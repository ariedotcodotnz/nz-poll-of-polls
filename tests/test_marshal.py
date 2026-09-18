from datetime import date

import numpy as np
import polars as pl
import pytest

from pollofpolls.config import Config
from pollofpolls.data.results import election_results_from_polls, load_reference_results
from pollofpolls.data.wikipedia import parse_page_file
from pollofpolls.forecast.simulate import simulate_seats
from pollofpolls.prep.marshal import alr, alr_inverse, build_dataset, result_vector
from pollofpolls.prep.polls_table import build_polls_table, tracked_parties

YEARS = [2011, 2014, 2017, 2020, 2023, 2026]


@pytest.fixture(scope="module")
def prepped(fixtures, root):
    cfg = Config(root)
    polls = [p for y in YEARS for p in parse_page_file(fixtures / f"{y}_party_vote.html", y)]
    table = build_polls_table(polls, cfg)
    scraped = election_results_from_polls(polls)
    reference = load_reference_results(root / "data/reference/election_results.csv")
    results = {y: {**scraped.get(y, {}), **reference.get(y, {})} for y in set(scraped) | set(reference)}
    return cfg, table, results


def test_alr_roundtrip():
    pi = np.array([[0.38, 0.27, 0.12, 0.09, 0.06, 0.03, 0.02, 0.03]])
    assert np.allclose(alr_inverse(alr(pi)), pi, atol=1e-6)


def test_polls_table(prepped):
    cfg, table, results = prepped
    assert table["poll_id"].n_unique() > 550
    assert "Labour–Talbot Mills" not in table["pollster_raw"].to_list()
    assert set(table["cycle"].unique().to_list()) <= {2011, 2014, 2017, 2020, 2023, 2026}
    # zeros are genuine "0%" reports (e.g. United Future), not invented values: a small fraction of rows
    assert table.filter(pl.col("share") == 0).height < 0.03 * table.height


def test_tracked_parties_2026(prepped):
    cfg, table, _ = prepped
    parties = tracked_parties(table, cfg, 2026)
    assert parties[:6] == ["National", "Labour", "Green", "ACT", "NZ First", "Te Pāti Māori"]
    assert "TOP" in parties


def test_dataset_2026(prepped):
    cfg, table, results = prepped
    ds = build_dataset(table, results, cfg, 2026)
    assert ds.parties[-1] == "Other" and ds.parties[0] == "National"
    assert ds.T == ds.target_t + 1
    assert ds.election_years == [2014, 2017, 2020, 2023]
    assert ds.N > 400 and ds.mask[:, -1].all() and ds.mask[:, 0].all()
    ratio = ds.y.sum(1) / ds.n
    assert ratio.min() > 0.95 and ratio.max() < 1.05   # polls sum to ~100% up to rounding
    assert ds.campaign.sum() == cfg.campaign_weeks * 5
    assert ds.cycle_years == [2014, 2017, 2020, 2023, 2026]
    assert ds.pm_party_idx == 0 and abs(ds.pm_prev_share - 0.3808) < 1e-6
    assert ((ds.cycle_frac >= 0) & (ds.cycle_frac <= 1)).all()
    assert set(np.unique(ds.round_unit)) <= {0.01, 0.005, 0.001}
    assert ds.anchors_t == (0, 147, 304, 464, 620)
    # fundamentals prior uses only elections before 2026: ten PM-party swings, 1996-2023
    assert ds.fund_n == 10 and abs(ds.fund_mean + 0.0198) < 0.001


def test_rounding_unit():
    from pollofpolls.prep.marshal import rounding_unit
    assert rounding_unit([0.38, 0.27, 0.12]) == 0.01
    assert rounding_unit([0.385, 0.27, 0.12]) == 0.005
    assert rounding_unit([0.381, 0.27]) == 0.001


def test_fundamentals_prior_has_no_look_ahead(prepped, root):
    from pollofpolls.data.results import load_reference_results
    from pollofpolls.prep.marshal import fundamentals_prior
    cfg = Config(root)
    res = load_reference_results(root / "data/reference/election_results.csv")
    mean, sd, n = fundamentals_prior(res, cfg.pm_by_year, 2020)
    assert n == 8                                   # 1996..2017, Ellis (2020): about -1.3 +/- 3.2 pp
    assert -0.02 < mean < -0.005 and 0.025 < sd < 0.045


def test_dataset_backtest_cutoff(prepped):
    cfg, table, results = prepped
    ds = build_dataset(table, results, cfg, 2023, cutoff=date(2023, 9, 1))
    assert ds.election_years == [2014, 2017, 2020]
    assert max(ds.mid_dates) <= date(2023, 9, 1)
    assert ds.target_t == ds.T - 1
    assert ds.last_data_t < ds.target_t


def test_simulate_seats_shapes(prepped):
    cfg, table, results = prepped
    parties = ["National", "Labour", "Green", "ACT", "NZ First", "Te Pāti Māori", "TOP", "Other"]
    rng = np.random.default_rng(0)
    pi = rng.dirichlet(np.array([30, 28, 11, 9, 11, 2.5, 6, 2.5]) * 40, size=500)
    sim = simulate_seats(pi, parties, cfg.electorates_cfg["electorates"], 500, rng)
    assert sim["seats"].shape == (500, 8)
    assert (sim["total"] >= 120).all() and (sim["seats"][:, -1] == 0).all()
    assert sim["electorates"][:, parties.index("ACT")].mean() > 0.8


def test_backtest_excludes_polls_not_yet_available(prepped):
    """A poll counts from fieldwork end (+ publication delay in backtests), not from its midpoint."""
    cfg, table, results = prepped
    cutoff = date(2023, 10, 7)                      # one week before the 2023 election
    ds = build_dataset(table, results, cfg, 2023, cutoff=cutoff, lagged=True)
    ids = set(ds.poll_ids)
    # Newshub-Reid Research 5-10 Oct 2023 has its midpoint on the cutoff but was not published until later
    assert not any("5–10 Oct 2023" in i for i in ids)
    avail = table.filter(pl.col("poll_id").is_in(list(ids)))
    assert avail["available"].max() <= cutoff and avail["date_to"].max() <= cutoff
    # live mode (no delay) still requires fieldwork to have ended
    live = build_dataset(table, results, cfg, 2023, cutoff=cutoff, lagged=False)
    assert table.filter(pl.col("poll_id").is_in(live.poll_ids))["date_to"].max() <= cutoff
    assert live.N >= ds.N


def test_dataset_fingerprint_tracks_model_inputs(prepped, root):
    cfg, table, results = prepped
    a = build_dataset(table, results, cfg, 2023, cutoff=date(2023, 8, 1), lagged=True)
    same = build_dataset(table, results, cfg, 2023, cutoff=date(2023, 8, 1), lagged=True)
    assert a.fingerprint() == same.fingerprint()
    # a later cutoff with no new eligible polls leaves the fingerprint alone
    later = build_dataset(table, results, cfg, 2023, cutoff=date(2023, 8, 2), lagged=True)
    if later.N == a.N:
        assert later.fingerprint() == a.fingerprint()
    # changing the campaign schedule changes it
    cfg2 = Config(root)
    cfg2.elections_cfg["campaign_weeks"] = 4
    b = build_dataset(table, results, cfg2, 2023, cutoff=date(2023, 8, 1), lagged=True)
    assert b.fingerprint() != a.fingerprint()
