"""The ``pollofpolls specials`` stage: forecast the official count of the close seats, and backtest the method.

Writes output/specials/:

    forecast.json     everything the website shows
    seats.csv         one row per seat: election-night count and the forecast official result
    candidates.csv    one row per candidate
    history.csv       every past seat and election: how its specials differed from its ordinary votes
    backtest.csv      leave-one-election-out forecasts of past seats, per method
    backtest_summary.csv
    margins.npz       simulated final margins and special-vote shares, for the charts
    party_votes.csv   the country's party vote: election night and the forecast official count, with seats
    routes.csv        how simulated counts changed the election-night House, and the minimal routes to a
                      change of majority
    party_vote_history.csv   past elections' ordinary and special party votes, as the model sees them
    party_vote_backtest.csv  leave-one-election-out forecasts of past elections' party votes
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from ..cache import StageCache
from ..config import Config
from . import national as natdata
from . import parliament as parl
from . import partyvote as pv
from .backtest import backtest, backtest_party_vote, summarise, summarise_party_vote
from .data import CATEGORIES, ElectorateCount, electorate_key, load_history, read_preliminary_xlsx
from .model import (
    CATS,
    Posterior,
    SpecialsData,
    build_official_count_data,
    build_specials_data,
    fit_official_count,
    fit_specials,
)
from .simulate import Simulation, Target, VolumeShift, simulate

OUT = "specials"
ELECTORATE_SEATS = Path("data") / "reference" / "electorate_seats.csv"     # electorates won at past elections
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def _log(msg: str) -> None:
    print(f"[pollofpolls] {msg}", flush=True)


def load_config(cfg: Config) -> dict:
    with open(cfg.paths.config / "specials.yml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Inputs:
    scfg: dict
    history: list[ElectorateCount]
    seats: list[ElectorateCount]
    data_dir: Path


def load_inputs(cfg: Config) -> Inputs:
    scfg = load_config(cfg)
    data_dir = cfg.paths.root / scfg["data_dir"]
    year = int(scfg["election"])
    pair = scfg["pair"]
    history = load_history(data_dir, year).counts
    seats = []
    for e in scfg["electorates"]:
        c = read_preliminary_xlsx(data_dir / e["file"], year, e["candidates"])
        if c.key != electorate_key(e["name"]):
            raise ValueError(f"config/specials.yml: {e['file']} is {c.electorate}, not {e['name']}")
        missing = [p for p in pair if c.party_index(p) is None]
        if missing:
            raise ValueError(f"config/specials.yml: {e['name']} has no candidate for {missing}")
        c.electorate = e["name"]
        seats.append(c)
    return Inputs(scfg, history, seats, data_dir)


# --------------------------------------------------------------------------------------- national inputs
def national_draws(scfg: dict, n: int, rng: np.random.Generator, groups: list[str] | None = None) -> dict[str, np.ndarray]:
    """The forecast election's national special-vote volume, overseas share and enrolment-change shift.

    With ``groups`` (the party-vote model's parties, the reference first) it also draws the shift of each party's
    domestic special party votes, ``kappa_party`` (n, K-1). Labour's candidates and Labour's party vote share
    one draw of how far the removed voters leaned.
    """
    v = scfg["volume"]
    ref, fc = v["reference"], v["forecast"]
    s_ref, t_ref, ov_ref = float(ref["special_votes"]), float(ref["total_votes"]), float(ref["overseas_votes"])
    o_ref = t_ref - s_ref
    err = fc["estimate_error"]
    specials = fc["special_votes_estimate"] * (1 + rng.normal(err["mean"], err["sd"], n))
    ordinary = float(fc["total_votes_estimate"] - fc["special_votes_estimate"])
    ov = np.clip(rng.normal(fc["overseas_votes"]["mean"], fc["overseas_votes"]["sd"], n), 1.0, 0.9 * specials)
    removed = np.clip(1 - ((specials - ov) / ordinary) / ((s_ref - ov_ref) / o_ref), 0.0, 1.0)
    lean = scfg["rule_change"]["lean"]
    z = {"labour": rng.standard_normal(n), "other": rng.standard_normal(n)}
    lean_draws = np.stack([lean[k]["mean"] + lean[k]["sd"] * z[k] for k in ("labour", "other")], axis=1)
    out = {"log_change": np.log((specials / ordinary) / (s_ref / o_ref)),
           "overseas_logit_change": np.log(ov / (specials - ov)) - np.log(ov_ref / (s_ref - ov_ref)),
           "specials": specials, "overseas": ov, "removed_share": removed,
           "kappa": -removed[:, None] * lean_draws}
    if groups is not None:
        party_lean = scfg["rule_change"].get("party_lean", {})
        second = scfg["pair"][1]
        cols = []
        for g in groups[1:]:
            m = party_lean.get(g, {"mean": 0.0, "sd": 0.0})
            cols.append(m["mean"] + m["sd"] * (z["labour"] if g == second else rng.standard_normal(n)))
        out["kappa_party"] = -removed[:, None] * np.stack(cols, axis=1)
    return out


# -------------------------------------------------------------------------------------------- summaries
def _q(x: np.ndarray) -> dict:
    qs = np.quantile(x, QUANTILES)
    return {"mean": float(np.mean(x)), **{f"q{int(q * 100):02d}": float(v) for q, v in zip(QUANTILES, qs)}}


def _share(x: np.ndarray) -> np.ndarray:
    """Labour's share of the two contenders' votes, (..., 3) -> (...)."""
    return x[..., 1] / np.maximum(x[..., 0] + x[..., 1], 1)


def _even_seat_pp(eta: np.ndarray) -> np.ndarray:
    """A log-ratio shift as percentage points of the two-candidate share in an evenly split seat."""
    return 100 * (1 / (1 + np.exp(-eta)) - 0.5)


def history_rows(data: SpecialsData, counts: list[ElectorateCount], pair: list[str]) -> list[dict]:
    rows = []
    for c in sorted(counts, key=lambda c: (c.key, c.year)):
        o = c.grouped(c.ordinary, pair)
        if o is None:
            continue
        s = c.grouped(c.special_total, pair)
        row = {"electorate": c.electorate, "year": c.year, "number": c.number,
               "ordinary_votes": int(c.ordinary.sum()), "special_votes": int(c.special_total.sum()),
               "specials_per_ordinary": float(c.special_total.sum() / c.ordinary.sum()),
               "overseas_share": float(c.specials["overseas"].sum() / c.special_total.sum()),
               "ordinary_share": float(_share(o)), "special_share": float(_share(s)),
               "special_shift_pp": float(100 * (_share(s) - _share(o))),
               "contenders_share_of_specials": float(s[:2].sum() / s.sum()),
               "contenders_share_of_ordinary": float(o[:2].sum() / o.sum())}
        for cat in CATEGORIES:
            row[f"special_share_{cat}"] = float(_share(c.grouped(c.specials[cat], pair)))
        t = c.grouped(c.total, pair)
        row["final_margin"] = int(t[1] - t[0])
        if c.preliminary is not None:
            p = c.grouped(c.preliminary, pair)
            row |= {"preliminary_margin": int(p[1] - p[0]), "ordinary_margin": int(o[1] - o[0]),
                    "official_ordinary_change": int(c.ordinary.sum() - c.preliminary.sum()),
                    "official_margin_change": int((o[1] - o[0]) - (p[1] - p[0]))}
        rows.append(row)
    return rows


def seat_summary(c: ElectorateCount, j: int, sim: Simulation, pair: list[str], close: int) -> dict:
    start = c.grouped(c.preliminary, pair)
    margin = sim.final[:, j, 1] - sim.final[:, j, 0]
    spec = sim.specials[:, j].sum(1)                               # (n, 3)
    n_spec = spec.sum(1)
    idx = {p: c.party_index(p) for p in pair}
    contenders = {p: c.candidates[idx[p]] for p in pair}
    lead = pair[1] if start[1] > start[0] else pair[0]
    trail = pair[0] if lead == pair[1] else pair[1]
    # what the trailing candidate needs from the contenders' special votes to overturn the election-night margin,
    # at the expected number of them (ignoring the smaller, two-way change to the ordinary votes)
    gap = abs(float(start[1] - start[0]))
    contender_specials = float(np.median(spec[:, 0] + spec[:, 1]))
    trail_k = pair.index(trail)
    need = 0.5 + gap / (2 * contender_specials)
    trail_share = spec[:, trail_k] / np.maximum(spec[:, 0] + spec[:, 1], 1)
    ordinary_trail_share = float(start[trail_k] / start[:2].sum())

    # every candidate: the "everyone else" group's specials split in proportion to their election-night votes
    others = [k for k in range(len(c.candidates)) if k not in idx.values()]
    other_votes = c.preliminary[others]
    cand = []
    for k, name in enumerate(c.candidates):
        if k == idx[pair[0]]:
            final, change, sp = sim.final[:, j, 0], sim.change[:, j, 0], spec[:, 0]
        elif k == idx[pair[1]]:
            final, change, sp = sim.final[:, j, 1], sim.change[:, j, 1], spec[:, 1]
        else:
            w = c.preliminary[k] / max(other_votes.sum(), 1)
            change, sp = sim.change[:, j, 2] * w, spec[:, 2] * w
            final = c.preliminary[k] + change + sp
        cand.append({"candidate": name, "party": c.parties[k], "preliminary": int(c.preliminary[k]),
                     "specials_mean": float(sp.mean()), "final": _q(final)})
    return {
        "name": c.electorate, "number": c.number, "contenders": contenders, "leader_election_night": lead,
        "candidates": cand,
        "preliminary": {"margin": int(start[1] - start[0]), "votes": int(c.preliminary.sum()),
                        "informal": c.informal, "advance": int(c.advance.sum()),
                        "election_day": int(c.election_day.sum()),
                        "contender_share": float(_share(start))},
        "specials": {**_q(n_spec), "overseas_mean": float(sim.specials[:, j, 0].sum(1).mean()),
                     "per_ordinary": float((n_spec / sim.final[:, j].sum(1).clip(1)).mean())},
        "official_change": {"votes": _q(sim.change[:, j].sum(1)),
                            "margin": _q(sim.change[:, j, 1] - sim.change[:, j, 0])},
        "special_share": _q(_share(spec)),
        "margin": {**_q(margin), "p_close": float((np.abs(margin) <= close).mean())},
        "p_win": {pair[0]: float((margin < 0).mean()), pair[1]: float((margin > 0).mean()),
                  "tie": float((margin == 0).mean())},
        "trailing": {"party": trail, "candidate": contenders[trail], "gap": gap,
                     "needs_share": need, "ordinary_share": ordinary_trail_share,
                     "forecast_share": _q(trail_share), "contender_specials": contender_specials},
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _posterior_summary(post: Posterior, data: SpecialsData, official: Posterior) -> dict:
    dom = CATS.index("domestic")
    out = {"years": {}, "electorates": {}}
    for y in data.years:
        t = data.year_index(y)
        eta = post["mu"][:, dom] + post["alpha"][:, t] + post["gamma"][:, t, dom]
        ov = post["mu"][:, 0] + post["alpha"][:, t] + post["gamma"][:, t, 0]
        out["years"][str(y)] = {"domestic_pp": _q(_even_seat_pp(eta[:, 0])), "overseas_pp": _q(_even_seat_pp(ov[:, 0]))}
    for k, label in zip(data.electorates, data.labels):
        e = data.electorate_index(k)
        out["electorates"][label] = {"persistent_pp": _q(_even_seat_pp(post["beta"][:, e, 0]))}
    out["sd_election_pp"] = _q(_even_seat_pp(post["sigma_year"][:, 0]))
    out["official_count"] = {k: _q(np.asarray(official[k]).reshape(len(official[k]), -1)[:, 0])
                             for k in ("jump_prob", "small_sd", "jump_log_median", "recheck_sd")}
    out["official_count"]["jump_median_share"] = _q(np.exp(official["jump_log_median"]))
    return out


# ------------------------------------------------------------------------------------------- the House
@dataclass
class PartyVote:
    """The party-vote side of the forecast: config/specials.yml's ``parliament`` section, loaded and fitted."""
    cfg: dict
    names: natdata.PartyNames
    history: natdata.NationalHistory
    data: pv.PartyVoteData
    post: Posterior
    start: np.ndarray               # election-night party votes (K,)
    change: dict                    # the official count's change to ordinary party votes, from past counts
    electorates: list[dict]         # every electorate's election-night leader
    valid: tuple[float, float]      # valid special party votes per special vote reported: (all, overseas)

    @property
    def groups(self) -> list[str]:
        return self.names.groups


def load_party_vote(cfg: Config, scfg: dict, mcmc: dict, progress: bool = False) -> PartyVote | None:
    pcfg = scfg.get("parliament")
    if not pcfg:
        return None
    pair, year = scfg["pair"], int(scfg["election"])
    names = natdata.PartyNames(pcfg["parties"])
    if names.groups[0] != pair[0] or pair[1] not in names.groups:
        raise ValueError(f"config/specials.yml: parliament.parties must start with {pair[0]} and include {pair[1]}")
    data_dir = cfg.paths.root / pcfg["data_dir"]
    history = natdata.load_history(data_dir, names, before=year)
    ref = scfg["volume"]["reference"]
    if int(ref["year"]) not in history.years:
        raise ValueError(f"{data_dir}/party-votes.csv has no {ref['year']} election (the volume reference)")
    t = history.year_index(int(ref["year"]))
    data = pv.build_party_vote_data(history, pair[1])
    post = pv.fit_party_vote(data, pcfg["priors"], mcmc, progress)
    return PartyVote(pcfg, names, history, data, post, natdata.load_preliminary(data_dir, year, names),
                     pv.ordinary_change(history), natdata.read_electorates(data_dir / f"electorates-{year}.csv"),
                     (float(history.specials[t].sum() / ref["special_votes"]),
                      float(history.specials[t, 0].sum() / ref["overseas_votes"])))


def extra_targets(party: PartyVote, pair: list[str], have: set[str]) -> list[tuple[dict, Target]]:
    """The other seats where the pair's candidates are first and second within ``modelled_margin`` votes."""
    out = []
    limit = int(party.cfg.get("modelled_margin", 0))
    for r in party.electorates:
        if electorate_key(r["electorate"]) in have or r["margin"] > limit:
            continue
        if {r["party"], r["runner_up_party"]} != set(pair):
            continue
        if r["national"] is None or r["labour"] is None or r["valid_votes"] is None:
            raise ValueError(f"electorates file: {r['electorate']} is within {limit} votes; add its candidate votes")
        start = np.array([r["national"], r["labour"], r["valid_votes"] - r["national"] - r["labour"]], dtype=float)
        if abs(start[1] - start[0]) != r["margin"] or (start[1] > start[0]) != (r["party"] == pair[1]):
            raise ValueError(f"electorates file: {r['electorate']}'s candidate votes do not give its margin")
        out.append((r, Target(r["electorate"], electorate_key(r["electorate"]), start, True)))
    return out


def electorate_seats(party: PartyVote, pair: list[str], targets: list[Target], sim: Simulation,
                     rng: np.random.Generator) -> np.ndarray:
    """Electorates won by each party in every simulation, (n, K)."""
    groups, n = party.groups, len(sim.final)
    rows = {electorate_key(r["electorate"]): r for r in party.electorates}
    won = np.zeros((n, len(groups)), dtype=int)
    for r in party.electorates:
        won[:, party.names.index(r["party"])] += 1
    k0, k1 = groups.index(pair[0]), groups.index(pair[1])
    for j, t in enumerate(targets):
        r = rows.get(t.key)
        if r is None:
            raise ValueError(f"{t.name} is not in the electorates file")
        if r["party"] not in pair:
            raise ValueError(f"{t.name}: the electorates file has {r['party']} leading, not one of {pair}")
        lead = k1 if r["party"] == pair[1] else k0
        other = k0 if lead == k1 else k1
        m = sim.final[:, j, 1] - sim.final[:, j, 0]
        flipped = (m < 0) if lead == k1 else (m > 0)
        won[:, lead] -= flipped
        won[:, other] += flipped
    for a in party.cfg.get("assumed_seats") or []:
        r = rows.get(electorate_key(a["electorate"]))
        if r is None:
            raise ValueError(f"config/specials.yml: assumed seat {a['electorate']} is not in the electorates file")
        flipped = rng.random(n) < float(a["p_flip"])
        won[:, party.names.index(r["party"])] -= flipped
        won[:, party.names.index(a["challenger"])] += flipped
    return won


def party_vote_history_rows(h: natdata.NationalHistory) -> list[dict]:
    rows = []
    for t, y in enumerate(h.years):
        o, s = h.ordinary[t], h.specials[t]
        for k, p in enumerate(h.parties):
            if o[k] < pv.MIN_VOTES:
                continue
            row = {"year": y, "party": p, "ordinary_votes": int(o[k]), "overseas_votes": int(s[0, k]),
                   "domestic_votes": int(s[1, k]), "ordinary_share": float(o[k] / o.sum()),
                   "overseas_share": float(s[0, k] / s[0].sum()), "domestic_share": float(s[1, k] / s[1].sum()),
                   "special_share": float(s[:, k].sum() / s.sum()), "final_share": float(h.final[t, k] / h.final[t].sum())}
            if not np.isnan(h.prelim[t]).any():
                row["election_night_share"] = float(h.prelim[t, k] / h.prelim[t].sum())
            rows.append(row)
    return rows


def house_summary(party: PartyVote, pvsim: pv.PartyVoteSim, house: parl.House, base: parl.House) -> dict:
    gov, opp = party.cfg["government"]["parties"], party.cfg["opposition"]["parties"]
    out = parl.summarise(house, base, gov, opp)
    out["routes"] = parl.paths(house, base, gov, opp)
    start, final, spec = party.start, pvsim.final, pvsim.specials
    share = final / final.sum(1, keepdims=True)
    spec_share = spec.sum(1) / spec.sum((1, 2))[:, None]
    threshold = float(party.cfg.get("threshold", 0.05))
    out["party_vote"] = [{
        "party": g, "election_night_votes": int(start[k]), "election_night_share": float(start[k] / start.sum()),
        "final_share": _q(share[:, k]), "change_pp": _q(100 * (share[:, k] - start[k] / start.sum())),
        "special_share": _q(spec_share[:, k]),
        "overseas_share": float((spec[:, 0, k] / spec[:, 0].sum(1)).mean()),
        "domestic_share": float((spec[:, 1, k] / spec[:, 1].sum(1)).mean()),
        "p_threshold": float((share[:, k] >= threshold).mean())} for k, g in enumerate(party.groups)]
    out["special_party_votes"] = {"valid": _q(spec.sum((1, 2))), "overseas": _q(spec[:, 0].sum(1)),
                                  "ordinary_added": _q(pvsim.ordinary.sum(1) - start.sum())}
    return out


# ------------------------------------------------------------------------------------------------- stage
def forecast_specials(cfg: Config, *, quick: bool = False, run_backtest: bool = True, progress: bool = False) -> dict:
    inp = load_inputs(cfg)
    scfg, pair = inp.scfg, inp.scfg["pair"]
    priors = scfg["priors"]
    mcmc = dict(scfg["mcmc"])
    n = int(scfg.get("n_sims", 20000))
    if quick:
        mcmc |= {"chains": 2, "warmup": 300, "samples": 300}
        n = min(n, 4000)
    df = priors.get("year_df")
    seed = int(mcmc.get("seed", 0))
    out = cfg.paths.output / OUT
    out.mkdir(parents=True, exist_ok=True)

    # the country's party vote first: the close seats' election-wide shift is tied to its lean
    party = load_party_vote(cfg, scfg, mcmc, progress)
    link = None
    if party is not None:
        link = {y: party.data.link[t] - party.data.link_mean for t, y in enumerate(party.data.years)}
        _log(f"specials: party vote fitted on {', '.join(map(str, party.data.years))} "
             f"(divergences {party.post.diagnostics['n_divergent']})")
    data = build_specials_data(inp.history, pair, link)
    post = fit_specials(data, priors, mcmc, progress)
    oc = build_official_count_data(data)
    official = fit_official_count(oc, priors, mcmc, progress)
    _log(f"specials: fitted on {data.n_rows} past seat-elections ({', '.join(map(str, data.years))}) in "
         f"{post.diagnostics['seconds']}s; official count on {len(oc.prelim)} archived election-night counts")

    rng = np.random.default_rng(seed)
    nat = national_draws(scfg, n, rng, party.groups if party else None)
    ref_year = int(scfg["volume"]["reference"]["year"])
    if ref_year not in data.years:
        raise ValueError(f"config/specials.yml: reference year {ref_year} has no official results in the data")
    volume = VolumeShift(ref_year, nat["log_change"], nat["overseas_logit_change"],
                         float(scfg["volume"].get("rule_change_sd", 0.0)))
    targets = [Target(c.electorate, c.key, c.grouped(c.preliminary, pair), True) for c in inp.seats]
    extra = extra_targets(party, pair, {t.key for t in targets}) if party else []
    targets += [t for _, t in extra]
    max_add = float(priors["official_count"].get("max_addition", 0.2))

    def run(year: int | None = None, enrolment: bool = True):
        """One set of simulated counts: the seats, and with the party vote the whole House."""
        r = np.random.default_rng([seed, year or 0, int(enrolment)])
        lean = z = None
        if party is not None:
            lean = pv.draw_lean(party.post, party.data, n, r, year=year, df=df)
            z = pv.link_covariate(lean, party.start, party.data.second) - party.data.link_mean
        sim = simulate(post, data, targets, volume, official=official, df=df, n=n, max_addition=max_add, seed=seed,
                       year=year, kappa=nat["kappa"] if enrolment else None,
                       national=z if data.national is not None else None)
        if party is None:
            return sim, None, None
        pvsim = pv.simulate_party_votes(party.start, lean, nat["specials"] * party.valid[0],
                                        nat["overseas"] * party.valid[1], r, change=party.change,
                                        kappa=nat["kappa_party"] if enrolment else None)
        house = parl.elect(pvsim.final, electorate_seats(party, pair, targets, sim, r), party.groups,
                           int(party.cfg.get("seats", 120)), float(party.cfg.get("threshold", 0.05)))
        return sim, pvsim, house

    sim, pvsim, house = run()
    close = int(scfg.get("close_margin", 100))
    seats = [seat_summary(c, j, sim, pair, close) for j, c in enumerate(inp.seats)]
    p_second = lambda s: (s.final[:, :, 1] > s.final[:, :, 0]).mean(0)  # noqa: E731
    gov = party.cfg["government"]["parties"] if party else []
    opp = party.cfg["opposition"]["parties"] if party else []
    house_odds = lambda h: {k: float((parl.outcome(h, gov, opp) == i).mean())  # noqa: E731
                            for i, k in enumerate(parl.OUTCOMES)}
    # scenarios: replay each past election's special-vote shift; and no enrolment-change adjustment
    scenarios = {"model": {s["name"]: s["p_win"][pair[1]] for s in seats}}
    house_scenarios = {"model": house_odds(house)} if party else {}
    alts = {f"like_{y}": {"year": y} for y in data.years} | {"no_enrolment_adjustment": {"enrolment": False}}
    for name, kw in alts.items():
        alt, _, alt_house = run(**kw)
        scenarios[name] = {c.electorate: float(p) for c, p in zip(inp.seats, p_second(alt))}
        if party:
            house_scenarios[name] = house_odds(alt_house)
    wins = (sim.final[:, :len(inp.seats), 1] > sim.final[:, :len(inp.seats), 0]).sum(1)
    dom = CATS.index("domestic")

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "election": int(scfg["election"]), "official_results_due": str(scfg.get("official_results_due", "")),
        "pair": pair, "close_margin": close, "n_sims": n,
        "seats": seats,
        "joint": {"party": pair[1], "wins": {str(k): float((wins == k).mean()) for k in range(len(seats) + 1)}},
        "scenarios": scenarios,
        "national": {"special_votes": _q(nat["specials"]), "overseas": _q(nat["overseas"]),
                     "removed_share": _q(nat["removed_share"]),
                     "estimate": scfg["volume"]["forecast"]["special_votes_estimate"],
                     "overseas_estimate": scfg["volume"]["forecast"]["overseas_votes"]["mean"],
                     "total_votes_estimate": scfg["volume"]["forecast"]["total_votes_estimate"],
                     "reference": scfg["volume"]["reference"],
                     "change_per_ordinary": _q(np.exp(nat["log_change"])),
                     "enrolment_shift_pp": _q(_even_seat_pp(nat["kappa"][:, 0]))},
        "election_shift_pp": {"forecast_domestic": _q(_even_seat_pp(sim.election[:, dom, 0])),
                              "forecast_overseas": _q(_even_seat_pp(sim.election[:, 0, 0]))},
        "posterior": _posterior_summary(post, data, official),
        "diagnostics": {"specials": post.diagnostics, "official_count": official.diagnostics},
        "data": {"years": data.years, "electorates": data.labels, "n_seat_elections": data.n_rows,
                 "n_election_night_counts": len(oc.prelim), "official_count_cases": oc.labels},
    }
    if "link_slope" in post.draws:
        summary["posterior"]["link_slope"] = _q(post["link_slope"][:, 0])

    if party is not None:
        leaders = np.bincount([party.names.index(r["party"]) for r in party.electorates],
                              minlength=len(party.groups))
        base = parl.elect(party.start, leaders, party.groups, int(party.cfg.get("seats", 120)),
                          float(party.cfg.get("threshold", 0.05)))
        hs = house_summary(party, pvsim, house, base)
        first = len(inp.seats)
        margins = sim.final[:, first:, 1] - sim.final[:, first:, 0]
        hs["other_seats"] = [{
            "name": r["electorate"], "leader": r["leader"], "leader_party": r["party"],
            "election_night_margin": int(t.start[1] - t.start[0]), "margin": _q(margins[:, j]),
            "p_flip": float(((margins[:, j] > 0) != (t.start[1] > t.start[0])).mean())}
            for j, (r, t) in enumerate(extra)]
        hs["assumed_seats"] = [{**a, "leader_party": next(r["party"] for r in party.electorates if electorate_key(
            r["electorate"]) == electorate_key(a["electorate"]))} for a in party.cfg.get("assumed_seats") or []]
        hs |= {"parties": party.groups, "government": party.cfg["government"], "opposition": party.cfg["opposition"],
               "colours": party.cfg.get("colours", {}), "modelled_margin": int(party.cfg.get("modelled_margin", 0)),
               "n_electorates": len(party.electorates), "scenarios": house_scenarios,
               "years": party.data.years, "ordinary_change": party.change,
               "model": {"common_sd": _q(party.post["sigma_f"]),
                         "party_sd": {g: _q(party.post["sigma_u"][:, k]) for k, g in enumerate(party.groups[1:])},
                         "category_sd": {c: _q(party.post["sigma_g"][:, k]) for k, c in enumerate(CATS)},
                         "usual_lean": {c: {g: float(party.post["mu"][:, i, k].mean())
                                            for k, g in enumerate(party.groups[1:])} for i, c in enumerate(CATS)}},
               "diagnostics": party.post.diagnostics}
        summary["house"] = hs
        _write_csv(out / "party_votes.csv", [{
            "party": r["party"], "election_night_votes": r["election_night_votes"],
            "election_night_share": r["election_night_share"],
            **{f"final_share_{k}": v for k, v in r["final_share"].items()},
            "special_share_mean": r["special_share"]["mean"], "p_threshold": r["p_threshold"],
            **next(({"seats_election_night": s["election_night"], "seats_mean": s["mean"],
                     "p_fewer_seats": s["p_fewer"], "p_more_seats": s["p_more"]}
                    for s in hs["seats"] if s["party"] == r["party"]), {})} for r in hs["party_vote"]])
        _write_csv(out / "routes.csv", [{"kind": kind, "route": "; ".join(r["steps"]), "probability": r["p"],
                                         "outcome": r.get("outcome", "opposition")}
                                        for kind in ("top", "minimal") for r in hs["routes"][kind]])
        _write_csv(out / "party_vote_history.csv", party_vote_history_rows(party.history))
        o = hs["outcome"]
        _log(f"specials: House: {party.cfg['government']['label']} keep a majority {o['government']:.2f}, "
             f"neither side {o['neither']:.2f}, {party.cfg['opposition']['label']} {o['opposition']:.2f}")

    hist = history_rows(data, inp.history, pair)
    if run_backtest:
        bmcmc = {**mcmc, **scfg.get("backtest", {}).get("mcmc", {})}
        if quick:
            bmcmc |= {"chains": 2, "warmup": 300, "samples": 300}
        b = scfg.get("backtest", {})
        nb = min(n, 8000)

        def linked(target_year: int, size: int, r: np.random.Generator):
            h = party.history
            if target_year not in h.years or len(h.years) < 3:
                return None
            d = pv.build_party_vote_data(h, pair[1], [y for y in h.years if y != target_year])
            fit = pv.fit_party_vote(d, party.cfg["priors"], bmcmc)
            draws = pv.draw_lean(fit, d, size, r, df=df)
            z = pv.link_covariate(draws, h.ordinary[h.year_index(target_year)], d.second) - d.link_mean
            return {y: d.link[t] - d.link_mean for t, y in enumerate(d.years)}, z

        rows = backtest(inp.history, pair, priors, bmcmc, n=nb, df=df,
                        volume_error_sd=float(b.get("volume_error_sd", 0.04)),
                        overseas_error_sd=float(b.get("overseas_error_sd", 0.10)), seed=7, log=_log,
                        link=linked if party is not None else None)
        _write_csv(out / "backtest.csv", rows)
        summary["backtest"] = summarise(rows)
        _write_csv(out / "backtest_summary.csv", summary["backtest"])
        if party is not None:
            won = natdata.read_electorate_seats(cfg.paths.root / ELECTORATE_SEATS, party.names)
            small = natdata.small_winners(cfg.paths.root / party.cfg["data_dir"], cfg.paths.root / ELECTORATE_SEATS,
                                          party.names)
            prow = backtest_party_vote(party.history, pair[1], party.cfg["priors"], bmcmc, electorates=won,
                                       small=small, n=nb,
                                       df=df, seed=11, log=_log, nseats=int(party.cfg.get("seats", 120)),
                                       threshold=float(party.cfg.get("threshold", 0.05)),
                                       volume_error_sd=float(party.cfg.get("backtest", {}).get("volume_error_sd", 0.04)))
            _write_csv(out / "party_vote_backtest.csv", prow)
            summary["house"]["backtest"] = summarise_party_vote(prow)
            (out / "party_vote_backtest.json").write_text(json.dumps(summary["house"]["backtest"]), encoding="utf-8")
    else:
        if (out / "backtest_summary.csv").exists():
            with open(out / "backtest_summary.csv", encoding="utf-8") as f:
                summary["backtest"] = [{k: _parse(v) for k, v in r.items()} for r in csv.DictReader(f)]
        if party is not None and (out / "party_vote_backtest.json").exists():
            summary["house"]["backtest"] = json.loads((out / "party_vote_backtest.json").read_text(encoding="utf-8"))

    (out / "forecast.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    _write_csv(out / "history.csv", hist)
    _write_csv(out / "seats.csv", [{
        "electorate": s["name"], "leader_election_night": s["leader_election_night"],
        "preliminary_margin": s["preliminary"]["margin"], "specials_mean": s["specials"]["mean"],
        "specials_q05": s["specials"]["q05"], "specials_q95": s["specials"]["q95"],
        **{f"margin_{k}": v for k, v in s["margin"].items()},
        **{f"p_{p}": s["p_win"][p] for p in pair}, "trailing_needs_share": s["trailing"]["needs_share"]}
        for s in seats])
    _write_csv(out / "candidates.csv", [{"electorate": s["name"], **{k: v for k, v in c.items() if k != "final"},
                                         **{f"final_{k}": v for k, v in c["final"].items()}}
                                        for s in seats for c in s["candidates"]])
    first = len(seats)
    np.savez_compressed(out / "margins.npz", names=np.array([s["name"] for s in seats]),
                        margin=(sim.final[:, :first, 1] - sim.final[:, :first, 0]).astype(np.int32),
                        special_share=_share(sim.specials[:, :first].sum(2)).astype(np.float32))
    for s in seats:
        _log(f"specials: {s['name']}: election night {s['preliminary']['margin']:+d} ({pair[1]} - {pair[0]}), "
             f"P({pair[1]}) {s['p_win'][pair[1]]:.2f}, final margin {s['margin']['q50']:+.0f} "
             f"({s['margin']['q05']:+.0f} to {s['margin']['q95']:+.0f})")
    return summary


def _parse(v: str):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def run_specials(cfg: Config, *, quick: bool = False, run_backtest: bool = True, force: bool = False) -> bool:
    """The stage, skipped when its inputs (data, config/specials.yml and this package) are unchanged."""
    scfg = load_config(cfg)
    out = cfg.paths.output / OUT
    extra = []
    if scfg.get("parliament"):                # the House also reads past electorates won and the seat allocator
        extra = [cfg.paths.root / scfg["parliament"]["data_dir"], cfg.paths.root / ELECTORATE_SEATS,
                 Path(__file__).parents[1] / "forecast" / "seats.py"]
    inputs = [cfg.paths.root / scfg["data_dir"], *extra, cfg.paths.config / "specials.yml", Path(__file__).parent,
              f"quick={quick}", f"backtest={run_backtest}"]
    return StageCache(cfg.paths.processed / "stamps").run(
        "specials", inputs, [out / "forecast.json"],
        lambda: forecast_specials(cfg, quick=quick, run_backtest=run_backtest), force=force)
