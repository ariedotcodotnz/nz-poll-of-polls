"""Special votes: parsers of the Electoral Commission's files, the models' algebra, simulation and the page."""
import copy
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from pollofpolls.specials.data import (
    CATEGORIES,
    candidate_party,
    electorate_key,
    load_history,
    read_official_csv,
    read_preliminary_xlsx,
)
from pollofpolls.specials.model import (
    CATS,
    OfficialCountData,
    Posterior,
    SpecialsData,
    fit_specials,
    log_ratio,
    softmax0,
)
from pollofpolls.specials.simulate import Target, VolumeShift, simulate

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "specials.yml").read_text(encoding="utf-8"))
DATA = ROOT / CFG["data_dir"]
PAIR = CFG["pair"]
needs_data = pytest.mark.skipif(not DATA.is_dir(), reason="the special-vote data folder is not present")


# ------------------------------------------------------------------------------------------------ parsing
def test_names_and_parties():
    assert electorate_key("Mt Roskill") == electorate_key("Mount Roskill") == "mount roskill"
    assert electorate_key("Ōhāriu") == "ohariu"
    assert candidate_party("Labour Party") == "Labour" and candidate_party("National Party") == "National"
    assert candidate_party("New Zealand First Party") == "NZ First"
    assert candidate_party("The Opportunities Party (TOP)") == "TOP" == candidate_party("Opportunity")
    assert candidate_party("Te Pāti Māori") == candidate_party("Māori Party") == "Te Pāti Māori"
    assert candidate_party("") == "Independent"
    assert candidate_party("Animal Justice Party ") == "Animal Justice Party"


@needs_data
def test_every_official_file_adds_up():
    """The parser checks every row and the electorate total itself; here every file passes those checks."""
    history = load_history(DATA, CFG["election"])
    assert len(history.counts) >= 12 and history.years == [2017, 2020, 2023]
    for c in history.counts:
        assert set(c.specials) == set(CATEGORIES)
        assert np.array_equal(c.ordinary + c.special_total, c.total)
        assert c.party_index("National") is not None and c.party_index("Labour") is not None


@needs_data
def test_a_specialist_school_is_a_voting_place_not_a_special_vote():
    c = read_official_csv(DATA / "2023" / "candidate-votes-by-voting-place-25.csv", 2023)
    assert c.electorate == "Mt Roskill" and c.number == 25
    assert c.special_total.sum() == 1168 + 4206 + 3068           # the three special-vote rows exactly
    assert any("Specialist School" in p for p in c.booths.places)


@needs_data
def test_election_night_counts_attach_to_their_official_result():
    history = load_history(DATA, CFG["election"])
    hutt = {c.year: c for c in history.for_electorate("Hutt South")}
    assert hutt[2020].preliminary is None                          # not archived
    b = hutt[2023]
    assert b.preliminary[b.party_index("National")] == 14872 and b.preliminary[b.party_index("Labour")] == 13778
    assert b.ordinary.sum() - b.preliminary.sum() == 2706          # votes the official count added


def _bad_csv(tmp_path, change):
    src = (DATA / "2023" / "candidate-votes-by-voting-place-58.csv").read_text(encoding="utf-8-sig")
    path = tmp_path / "bad.csv"
    path.write_text(change(src), encoding="utf-8")
    return path


@needs_data
def test_a_file_that_does_not_add_up_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="add up|sum to"):
        read_official_csv(_bad_csv(tmp_path, lambda s: s.replace(",Wairarapa Total,1737,", ",Wairarapa Total,1738,")),
                          2023)
    with pytest.raises(ValueError, match="special-vote rows"):
        renamed = _bad_csv(tmp_path, lambda s: s.replace("Special Votes ON Polling Day", "Special Votes Later"))
        read_official_csv(renamed, 2023)


@needs_data
@pytest.mark.parametrize("seat", CFG["electorates"], ids=lambda s: s["name"])
def test_preliminary_workbooks(seat):
    c = read_preliminary_xlsx(DATA / seat["file"], CFG["election"], seat["candidates"])
    assert electorate_key(c.electorate) == electorate_key(seat["name"])
    assert c.total.sum() == c.advance.sum() + c.election_day.sum() > 20000      # counts stored as text are read
    assert c.informal and c.informal > 0
    # each contender's votes follow their party's vote across voting places, more than any other party's
    shares = lambda m: m / np.maximum(m.sum(1, keepdims=True), 1)              # noqa: E731
    big = c.booths.votes.sum(1) > 50
    cand, party = shares(c.booths.votes)[big], shares(c.party_booths.votes)[big]
    for p in PAIR:
        k = c.party_index(p)
        cor = [np.corrcoef(cand[:, k], party[:, j])[0, 1] if party[:, j].std() > 0 else -1
               for j in range(party.shape[1])]
        assert c.party_booths.columns[int(np.argmax(cor))] == p and max(cor) > 0.6, (seat["name"], p, max(cor))


@needs_data
def test_a_candidate_without_a_configured_party_is_an_error():
    seat = CFG["electorates"][0]
    partial = dict(list(seat["candidates"].items())[1:])
    with pytest.raises(ValueError, match="no party configured"):
        read_preliminary_xlsx(DATA / seat["file"], CFG["election"], partial)


# ---------------------------------------------------------------------------------------------- algebra
def test_log_ratio_and_softmax_invert():
    v = np.array([[1000.0, 2500.0, 300.0], [10.0, 1.0, 0.0]])
    shares = softmax0(log_ratio(v))
    assert np.allclose(shares, (v + 0.5) / (v + 0.5).sum(1, keepdims=True))


def test_official_count_jacobian_matches_finite_differences():
    prelim = np.array([[14872.0, 13778.0, 4119.0], [100.0, 300.0, 50.0]])
    d = OfficialCountData(prelim=prelim, ordinary=prelim * 1.01, labels=["a", "b"])
    eps = 1e-6
    for k in range(2):
        bump = np.zeros(2)
        bump[k] = eps
        numeric = (softmax0(log_ratio(prelim) + bump) - softmax0(log_ratio(prelim) - bump)) / (2 * eps)
        assert np.allclose(d.jacobian[:, :, k], numeric, atol=1e-6)
    assert np.allclose(d.residual, 0, atol=0.01)        # a change split like the count leaves no residual (in votes)


# --------------------------------------------------------------------------------------------- simulation
def _fake(n=400, E=2, T=2, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    C = len(CATS)
    draws = {"mu": np.full((n, C, 2), shift), "sigma_year": np.full((n, 2), 0.1),
             "L_year": np.broadcast_to(np.eye(2), (n, 2, 2)).copy(), "alpha": rng.normal(0, 0.1, (n, T, 2)),
             "sigma_yc": np.full((n, 2), 0.05), "gamma": np.zeros((n, T, C, 2)), "sigma_e": np.full((n, 2), 0.05),
             "beta": np.zeros((n, E, 2)), "sigma_eps": np.full((n, 2), 0.05), "vol_year": np.full((n, T), -1.6),
             "sigma_vol_b": np.full(n, 0.2), "vol_e": np.zeros((n, E)), "sigma_vol_c": np.full(n, 0.05),
             "ov_year": np.full((n, T), -1.9), "sigma_ov_b": np.full(n, 0.2), "ov_e": np.zeros((n, E)),
             "sigma_ov_c": np.full(n, 0.05)}
    official = {"jump_prob": np.full(n, 0.3), "small_sd": np.full(n, 0.002), "jump_log_median": np.full(n, -3.7),
                "jump_log_sd": np.full(n, 0.8), "sigma_xi": np.full((n, 2), 0.4), "recheck_sd": np.full(n, 0.1)}
    data = SpecialsData(pair=PAIR, electorates=["a", "b"], labels=["A", "B"], years=[2020, 2023],
                        e_idx=np.array([0, 1]), t_idx=np.array([0, 1]), base=np.zeros((2, 2)),
                        counts=np.ones((2, C, 3)), ordinary=np.ones((2, 3)), prelim=np.full((2, 3), np.nan))
    return Posterior(draws, {}), Posterior(official, {}), data


def test_simulation_adds_up_and_moves_seats_together():
    post, official, data = _fake()
    targets = [Target("A", "a", np.array([14000.0, 13800.0, 3000.0])),
               Target("New", "new", np.array([9000.0, 9100.0, 900.0]))]
    n = 3000
    vol = VolumeShift(2023, np.full(n, -0.4), np.zeros(n), 0.1)
    sim = simulate(post, data, targets, vol, official=official, n=n, seed=1)
    start = np.stack([t.start for t in targets])
    assert np.array_equal(sim.final, start[None] + sim.change + sim.specials.sum(2))
    assert (sim.specials >= 0).all() and (sim.final >= 0).all()
    ratio = sim.specials.sum((2, 3)) / (start.sum(1) + sim.change.sum(2))
    assert 0.12 < np.median(ratio[:, 0]) < 0.15                    # exp(-1.6 - 0.4)
    # one election-wide shift: the two seats' special-vote shares move together
    share = sim.specials[..., 1].sum(2) / sim.specials[..., :2].sum((2, 3))
    assert np.corrcoef(share[:, 0], share[:, 1])[0, 1] > 0.5
    # same seed, same simulation; a replayed past election uses that election's shift
    again = simulate(post, data, targets, vol, official=official, n=n, seed=1)
    assert np.array_equal(again.final, sim.final)
    past = simulate(post, data, targets, vol, official=official, n=n, seed=1, year=2020)
    replayed = past.election[:, 1, 0] - post["mu"][0, 1, 0]          # gamma is zero in the fake posterior
    assert np.isin(np.round(replayed, 9), np.round(post["alpha"][:, 0, 0], 9)).all()


def test_the_enrolment_shift_moves_domestic_specials_only():
    post, official, data = _fake(n=200)
    t = [Target("A", "a", np.array([10000.0, 10000.0, 2000.0]), official_change=False)]
    n = 20000
    vol = VolumeShift(2023, np.zeros(n), np.zeros(n))
    base = simulate(post, data, t, vol, n=n, seed=3)
    kappa = np.tile([-0.2, 0.0], (n, 1))                 # away from Labour's candidate
    moved = simulate(post, data, t, vol, n=n, seed=3, kappa=kappa)
    assert (base.change == 0).all()
    share = lambda s, c: s.specials[:, 0, c, 1].sum() / s.specials[:, 0, c, :2].sum()   # noqa: E731
    dom, ov = CATS.index("domestic"), CATS.index("overseas")
    assert share(moved, dom) < share(base, dom) - 0.03
    assert abs(share(moved, ov) - share(base, ov)) < 0.01


def test_national_draws_follow_the_config():
    from pollofpolls.specials.run import national_draws
    scfg = copy.deepcopy(CFG)
    scfg["volume"]["forecast"]["estimate_error"] = {"mean": 0.0, "sd": 0.0}
    scfg["volume"]["forecast"]["overseas_votes"] = {"mean": 78030, "sd": 0.0}
    nat = national_draws(scfg, 10, np.random.default_rng(0))
    ref, fc = scfg["volume"]["reference"], scfg["volume"]["forecast"]
    ratio = (fc["special_votes_estimate"] / (fc["total_votes_estimate"] - fc["special_votes_estimate"])) / \
        (ref["special_votes"] / (ref["total_votes"] - ref["special_votes"]))
    assert np.allclose(np.exp(nat["log_change"]), ratio)
    assert 0.3 < nat["removed_share"][0] < 0.5
    lean = scfg["rule_change"]["lean"]
    if lean["labour"]["mean"] > 0:
        assert nat["kappa"][:, 0].mean() < 0


# ------------------------------------------------------------------------------------------------ fitting
def test_fit_recovers_a_known_shift():
    """Synthetic seats whose specials lean 0.3 towards the second party: the fitted mean shift finds it."""
    rng = np.random.default_rng(5)
    T, E, C = 3, 4, len(CATS)
    e_idx, t_idx = np.repeat(np.arange(E), T), np.tile(np.arange(T), E)
    base = rng.normal(0, 0.2, (E * T, 2))
    eta = base[:, None, :] + np.array([0.3, 0.4]) + rng.normal(0, 0.03, (E * T, C, 2))
    counts = np.stack([rng.multinomial(n, p) for n, p in
                       zip(np.full(E * T * C, 3000), softmax0(eta).reshape(-1, 3))]).reshape(E * T, C, 3).astype(float)
    ordinary = 20000 * softmax0(base)
    data = SpecialsData(pair=PAIR, electorates=list("abcd"), labels=list("ABCD"), years=[2017, 2020, 2023],
                        e_idx=e_idx, t_idx=t_idx, base=base, counts=counts, ordinary=ordinary,
                        prelim=np.full((E * T, 3), np.nan))
    post = fit_specials(data, CFG["priors"], {"chains": 1, "warmup": 200, "samples": 200, "seed": 2})
    shift = (post["mu"][:, :, 0] + post["alpha"][:, :, 0].mean(1, keepdims=True)).mean()
    assert abs(shift - 0.3) < 0.08


# ------------------------------------------------------------------------------------------- stage and page
def test_page_without_a_forecast_says_how_to_make_one(tmp_path):
    from pollofpolls.report.specials import SpecialsPage, display_name
    shutil.copytree(ROOT / "config", tmp_path / "config")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    page = SpecialsPage(tmp_path)
    assert not page.available and "pollofpolls specials" in page.intro()
    assert page.tiles() == page.seat_table() == page.history_table() == "" and page.facts() == {}
    assert display_name("McANULTY, Kieran") == "Kieran McAnulty"
    assert display_name("WYLIE-VAN EERD, Ben") == "Ben Wylie-Van Eerd"
    assert display_name("WOOD, Michael Philip") == "Michael Philip Wood"


@needs_data
def test_stage_end_to_end(tmp_path):
    """A quick run of `pollofpolls specials` on a copy of the project, read back by the website page."""
    from pollofpolls.config import Config
    from pollofpolls.report.specials import SpecialsPage
    from pollofpolls.specials.run import forecast_specials
    shutil.copytree(ROOT / "config", tmp_path / "config")
    shutil.copytree(DATA, tmp_path / CFG["data_dir"])
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    cfg = Config(tmp_path)
    summary = forecast_specials(cfg, quick=True, run_backtest=False)
    assert [s["name"] for s in summary["seats"]] == [e["name"] for e in CFG["electorates"]]
    for s in summary["seats"]:
        p = s["p_win"]
        assert abs(p["National"] + p["Labour"] + p["tie"] - 1) < 1e-9
        assert s["margin"]["q05"] <= s["margin"]["q50"] <= s["margin"]["q95"]
        assert 0.5 < s["trailing"]["needs_share"] < 1
        assert sum(c["preliminary"] for c in s["candidates"]) == s["preliminary"]["votes"]
    assert abs(sum(summary["joint"]["wins"].values()) - 1) < 1e-9
    assert set(summary["scenarios"]) == {"model", "like_2017", "like_2020", "like_2023", "no_enrolment_adjustment"}
    out = cfg.paths.output / "specials"
    for f in ["forecast.json", "seats.csv", "candidates.csv", "history.csv", "margins.npz"]:
        assert (out / f).exists(), f
    page = SpecialsPage(tmp_path)
    assert page.available and "Hutt South" in page.seat_table() and "399,000" in page.intro()
    assert page.figure("margins").data and page.figure("needs", narrow=True).data
    assert page.facts()["n_cases"] == "12"
