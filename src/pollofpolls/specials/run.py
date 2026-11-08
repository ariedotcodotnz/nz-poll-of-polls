"""The ``pollofpolls specials`` stage: forecast the official count of the close seats, and backtest the method.

Writes output/specials/:

    forecast.json     everything the website shows
    seats.csv         one row per seat: election-night count and the forecast official result
    candidates.csv    one row per candidate
    history.csv       every past seat and election: how its specials differed from its ordinary votes
    backtest.csv      leave-one-election-out forecasts of past seats, per method
    backtest_summary.csv
    margins.npz       simulated final margins and special-vote shares, for the charts
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
from .backtest import backtest, summarise
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
def national_draws(scfg: dict, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """The forecast election's national special-vote volume, overseas share and enrolment-change shift."""
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
    lean_draws = np.stack([rng.normal(lean[k]["mean"], lean[k]["sd"], n) for k in ("labour", "other")], axis=1)
    return {"log_change": np.log((specials / ordinary) / (s_ref / o_ref)),
            "overseas_logit_change": np.log(ov / (specials - ov)) - np.log(ov_ref / (s_ref - ov_ref)),
            "specials": specials, "overseas": ov, "removed_share": removed,
            "kappa": -removed[:, None] * lean_draws}


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
    out = cfg.paths.output / OUT
    out.mkdir(parents=True, exist_ok=True)

    data = build_specials_data(inp.history, pair)
    post = fit_specials(data, priors, mcmc, progress)
    oc = build_official_count_data(data)
    official = fit_official_count(oc, priors, mcmc, progress)
    _log(f"specials: fitted on {data.n_rows} past seat-elections ({', '.join(map(str, data.years))}) in "
         f"{post.diagnostics['seconds']}s; official count on {len(oc.prelim)} archived election-night counts")

    rng = np.random.default_rng(int(mcmc.get("seed", 0)))
    nat = national_draws(scfg, n, rng)
    ref_year = int(scfg["volume"]["reference"]["year"])
    if ref_year not in data.years:
        raise ValueError(f"config/specials.yml: reference year {ref_year} has no official results in the data")
    volume = VolumeShift(ref_year, nat["log_change"], nat["overseas_logit_change"],
                         float(scfg["volume"].get("rule_change_sd", 0.0)))
    targets = [Target(c.electorate, c.key, c.grouped(c.preliminary, pair), True) for c in inp.seats]
    max_add = float(priors["official_count"].get("max_addition", 0.2))
    run = lambda **kw: simulate(post, data, targets, volume, official=official, df=df, n=n,  # noqa: E731
                                max_addition=max_add, seed=int(mcmc.get("seed", 0)), **kw)
    sim = run(kappa=nat["kappa"])

    close = int(scfg.get("close_margin", 100))
    seats = [seat_summary(c, j, sim, pair, close) for j, c in enumerate(inp.seats)]
    # scenarios: replay each past election's special-vote shift; and no enrolment-change adjustment
    scenarios = {"model": {s["name"]: s["p_win"][pair[1]] for s in seats}}
    for y in data.years:
        alt = run(kappa=nat["kappa"], year=y)
        scenarios[f"like_{y}"] = {c.electorate: float((alt.final[:, j, 1] > alt.final[:, j, 0]).mean())
                                  for j, c in enumerate(inp.seats)}
    alt = run(kappa=None)
    scenarios["no_enrolment_adjustment"] = {c.electorate: float((alt.final[:, j, 1] > alt.final[:, j, 0]).mean())
                                            for j, c in enumerate(inp.seats)}
    wins = (sim.final[:, :, 1] > sim.final[:, :, 0]).sum(1)
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

    hist = history_rows(data, inp.history, pair)
    if run_backtest:
        bmcmc = {**mcmc, **scfg.get("backtest", {}).get("mcmc", {})}
        if quick:
            bmcmc |= {"chains": 2, "warmup": 300, "samples": 300}
        b = scfg.get("backtest", {})
        rows = backtest(inp.history, pair, priors, bmcmc, n=min(n, 8000), df=df,
                        volume_error_sd=float(b.get("volume_error_sd", 0.04)),
                        overseas_error_sd=float(b.get("overseas_error_sd", 0.10)), seed=7, log=_log)
        _write_csv(out / "backtest.csv", rows)
        summary["backtest"] = summarise(rows)
        _write_csv(out / "backtest_summary.csv", summary["backtest"])
    elif (out / "backtest_summary.csv").exists():
        with open(out / "backtest_summary.csv", encoding="utf-8") as f:
            summary["backtest"] = [{k: _parse(v) for k, v in r.items()} for r in csv.DictReader(f)]

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
    np.savez_compressed(out / "margins.npz", names=np.array([s["name"] for s in seats]),
                        margin=(sim.final[:, :, 1] - sim.final[:, :, 0]).astype(np.int32),
                        special_share=_share(sim.specials.sum(2)).astype(np.float32))
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
    inputs = [cfg.paths.root / scfg["data_dir"], cfg.paths.config / "specials.yml", Path(__file__).parent,
              f"quick={quick}", f"backtest={run_backtest}"]
    return StageCache(cfg.paths.processed / "stamps").run(
        "specials", inputs, [out / "forecast.json"],
        lambda: forecast_specials(cfg, quick=quick, run_backtest=run_backtest), force=force)
