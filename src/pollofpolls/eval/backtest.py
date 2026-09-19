"""Rolling-origin backtests: refit each variant with the polls available some weeks before a past election,
score the forecast against the result, and combine variants by CRPS stacking.

Layout under output/backtest/:
    fits/<variant>_<year>_h<weeks>.npz   cached fits (not committed)
    cases/<variant>_<year>_h<weeks>.json one scored case each (committed)
    scores.csv, summary.csv, summary_by_horizon.csv, ensemble.csv, stacking.json
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ..cache import fingerprint
from ..config import Config
from ..data.results import load_electorate_seats
from ..forecast.seats import allocate_seats_matrix
from ..model.fit import FitResult, fit
from ..prep.marshal import build_dataset, result_vector
from .calibration import factor, fit_spread, inflate
from .scoring import (brier, crps_components, crps_stacking_weights, keep_index, mixture_draws,
                      score_forecast)

DEFAULT_BLOCS = {"right": ["National", "ACT"], "left": ["Labour", "Green"]}


def blocs_for(cfg: Config, target: int) -> tuple[list[str], list[str]]:
    """The seat blocs compared by the backtest Brier score, from backtest.blocs in config/model.yml."""
    b = {int(k): v for k, v in (cfg.model_cfg.get("backtest", {}).get("blocs") or {}).items()}.get(target, DEFAULT_BLOCS)
    return list(b["right"]), list(b["left"])


def bloc_probability(draws: np.ndarray, parties: list[str], blocs: tuple[list[str], list[str]],
                     electorates: dict[str, int], outcome: np.ndarray) -> tuple[float, bool]:
    """P(right bloc has more seats than left bloc) using the electorates actually won, and what happened."""
    right, left = blocs
    idx = {p: i for i, p in enumerate(parties)}
    el = np.array([electorates.get(p, 0) for p in parties])

    def event(votes: np.ndarray) -> np.ndarray:
        v = np.clip(votes, 0.0, None)
        v[:, -1] = 0.0                                    # "Other" is many parties, none qualifying
        seats = allocate_seats_matrix(v, np.tile(el, (len(v), 1)))
        r = seats[:, [idx[p] for p in right if p in idx]].sum(1)
        lft = seats[:, [idx[p] for p in left if p in idx]].sum(1)
        return r > lft

    return float(event(draws.copy()).mean()), bool(event(outcome[None, :].copy())[0])


def fit_fingerprint(cfg: Config, ds, variant: dict, mcmc_cfg: dict) -> str:
    """Everything a fit depends on: the full dataset, the variant, priors, sampler settings and model code."""
    return fingerprint([ds.fingerprint(), json.dumps(variant, sort_keys=True), json.dumps(cfg.priors, sort_keys=True),
                        json.dumps(mcmc_cfg, sort_keys=True), json.dumps(cfg.model_cfg.get("election_obs_sd")),
                        cfg.paths.root / "src" / "pollofpolls" / "model"])


def _tag(variant: str, target: int, h: int) -> str:
    return f"{variant}_{target}_h{h}"


def backtest_case(cfg: Config, polls: pl.DataFrame, results: dict, target: int, horizon_weeks: int,
                  variant_name: str, out_dir: Path, mcmc_cfg: dict, force: bool = False,
                  progress: bool = False) -> dict:
    election = cfg.election(target)
    cutoff = election.date - timedelta(weeks=horizon_weeks)
    ds = build_dataset(polls, results, cfg, target, cutoff, lagged=True)
    variant = cfg.variant(variant_name)
    prefix = out_dir / "fits" / _tag(variant_name, target, horizon_weeks)
    fp = fit_fingerprint(cfg, ds, variant, mcmc_cfg)
    stamp = prefix.with_suffix(".stamp")
    if not force and stamp.exists() and stamp.read_text() == fp and prefix.with_suffix(".npz").exists():
        fr = FitResult.load(prefix)
    else:
        fr = fit(ds, variant, cfg.priors, mcmc_cfg, cfg.model_cfg.get("election_obs_sd", 0.01),
                 seed=int(mcmc_cfg.get("seed", 0)) + horizon_weeks, progress=progress)
        fr.save(prefix)
        stamp.write_text(fp)
    outcome = result_vector(results[target], ds.parties)
    draws = fr.pi_target
    el = load_electorate_seats(cfg.paths.reference / "electorate_seats.csv").get(target, {})
    p_event, actual = bloc_probability(draws, ds.parties, blocs_for(cfg, target), el, outcome)
    record = {
        "target": target, "horizon_weeks": horizon_weeks, "variant": variant_name, "cutoff": cutoff.isoformat(),
        "n_polls": int(ds.N), "parties": ds.parties, "p_right_bloc_ahead": p_event, "right_bloc_ahead": actual,
        "brier_bloc": brier(p_event, actual),
        "forecast_mean": {p: float(v) for p, v in zip(ds.parties, draws.mean(0))},
        "outcome": {p: float(v) for p, v in zip(ds.parties, outcome)},
        "diagnostics": fr.diagnostics, **score_forecast(draws, ds.parties, outcome),
    }
    (out_dir / "cases").mkdir(parents=True, exist_ok=True)
    (out_dir / "cases" / f"{_tag(variant_name, target, horizon_weeks)}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1))
    return record


def run_backtests(cfg: Config, polls: pl.DataFrame, results: dict, variants: list[str], targets: list[int],
                  horizons: list[int], out_dir: Path, mcmc_cfg: dict, force: bool = False,
                  progress: bool = False) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for target in targets:
        for h in horizons:
            for v in variants:
                print(f"[backtest] {target} h={h}w {v}", flush=True)
                r = backtest_case(cfg, polls, results, target, h, v, out_dir, mcmc_cfg, force, progress)
                rows.append(r)
                d = r["diagnostics"]
                print(f"   CRPS {r['crps_pp']:.2f}pp  MAE {r['mae_pp']:.2f}pp  cov90 {r['coverage_90']:.2f}  "
                      f"rhat {d.get('max_rhat') or float('nan'):.3f}  steps {d.get('mean_num_steps', 0):.0f}  "
                      f"{d.get('seconds')}s", flush=True)
    return rows


def load_cases(out_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((out_dir / "cases").glob("*.json"))]


def _flat(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows])


def _summary(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    return (df.group_by(by).agg([
        pl.col("crps_pp").mean(), pl.col("energy_pp").mean(), pl.col("mae_pp").mean(),
        pl.col("coverage_50").mean(), pl.col("coverage_90").mean(), pl.col("log_score_share").mean(),
        pl.col("brier_bloc").mean(), pl.len().alias("n_cases"),
    ]).sort(by[1:] + ["crps_pp"] if len(by) > 1 else ["crps_pp"]))


def aggregate(cfg: Config, out_dir: Path, ensemble: list[str], baseline: str = "legacy", seed: int = 7) -> dict:
    """Score every case, fit CRPS stacking weights, and evaluate the ensembles leave-one-election-out."""
    rows = load_cases(out_dir)
    if not rows:
        raise RuntimeError(f"no backtest cases under {out_dir / 'cases'}")
    by_case: dict[tuple[int, int], dict[str, dict]] = {}
    for r in rows:
        by_case.setdefault((r["target"], r["horizon_weeks"]), {})[r["variant"]] = r
    # ensemble members that have been backtested; cases missing any of them are left out of the stacking
    ens = [v for v in ensemble if any(v in c for c in by_case.values())]
    el_all = load_electorate_seats(cfg.paths.reference / "electorate_seats.csv")

    comps, draws_by_case = {}, {}
    for key, variants in sorted(by_case.items()):
        if not ens or not all(v in variants for v in ens):
            continue
        sets = [FitResult.load(out_dir / "fits" / _tag(v, *key)).pi_target for v in ens]
        parties = variants[ens[0]]["parties"]
        outcome = np.array([variants[ens[0]]["outcome"][p] for p in parties])
        draws_by_case[key] = (sets, parties, outcome)
        comps[key] = crps_components(sets, outcome, keep_index(parties), seed=seed)

    stacking = {"method": "crps", "variants": ens, "weights": {}, "loeo_weights": {}, "n_cases": len(comps)}
    ens_rows = []
    if ens and comps:
        w_all = crps_stacking_weights(list(comps.values()))
        stacking["weights"] = {v: float(x) for v, x in zip(ens, w_all)}
        targets = sorted({k[0] for k in comps})
        equal = np.full(len(ens), 1.0 / len(ens))

        def weights_without(excluded: set) -> np.ndarray:
            train = [c for k, c in comps.items() if k[0] not in excluded]
            return crps_stacking_weights(train) if train else equal

        def calibration_cases(elections: list[int], excluded: set) -> list[tuple]:
            """Stacked mixtures for ``elections``; each one's weights are fitted without it or ``excluded``."""
            out = []
            for f in elections:
                w_f = weights_without(excluded | {f})
                for key in [k for k in comps if k[0] == f]:
                    sets, parties, outcome = draws_by_case[key]
                    out.append((mixture_draws(sets, w_f, 4000, seed=seed), outcome, parties, float(key[1])))
            return out

        # stacked mixture per case, with weights fitted on the other elections
        loeo_mix = {}
        for target in targets:
            w = weights_without({target})
            stacking["loeo_weights"][str(target)] = {v: float(x) for v, x in zip(ens, w)}
            for key in [k for k in comps if k[0] == target]:
                loeo_mix[key] = mixture_draws(draws_by_case[key][0], w, 4000, seed=seed)
        # spread calibration. For 2026 it is fitted on every election's out-of-sample mixture. For evaluation it
        # is nested: when election E is held out, neither the spread nor the stacking weights inside its training
        # mixtures may see E's result.
        stacking["spread"] = fit_spread(calibration_cases(targets, set())) if len(targets) > 1 else None
        stacking["loeo_spread"] = {}
        for target in targets:
            inner = [f for f in targets if f != target]
            spread = fit_spread(calibration_cases(inner, {target})) if inner else None
            stacking["loeo_spread"][str(target)] = spread
            for key in [k for k in comps if k[0] == target]:
                sets, parties, outcome = draws_by_case[key]
                for label, mix in (("ensemble", loeo_mix[key]),
                                   ("calibrated", inflate(loeo_mix[key], factor(spread, key[1]))),
                                   ("equal", mixture_draws(sets, equal, 4000, seed=seed))):
                    p_event, actual = bloc_probability(mix, parties, blocs_for(cfg, key[0]),
                                                       el_all.get(key[0], {}), outcome)
                    rec = {"target": key[0], "horizon_weeks": key[1], "variant": label, "parties": parties,
                           "p_right_bloc_ahead": p_event, "right_bloc_ahead": actual,
                           "brier_bloc": brier(p_event, actual), **score_forecast(mix, parties, outcome)}
                    ens_rows.append(rec)
    (out_dir / "stacking.json").write_text(json.dumps(stacking, indent=1))

    all_rows = rows + ens_rows
    df = _flat(all_rows)
    df.write_csv(out_dir / "scores.csv")
    summary = _summary(df, ["variant"])
    summary.write_csv(out_dir / "summary.csv")
    _summary(df, ["variant", "horizon_weeks"]).write_csv(out_dir / "summary_by_horizon.csv")
    if ens_rows:
        _flat(ens_rows).write_csv(out_dir / "ensemble.csv")

    # head-to-head against the baseline, case by case
    h2h = []
    for (target, h), variants in sorted(by_case.items()):
        base = variants.get(baseline)
        if base is None:
            continue
        found = {r["variant"]: r for r in ens_rows if r["target"] == target and r["horizon_weeks"] == h}
        if "ensemble" in found:
            row = {"target": target, "horizon_weeks": h, "legacy_crps_pp": base["crps_pp"],
                   "legacy_cov90": base["coverage_90"], "ensemble_crps_pp": found["ensemble"]["crps_pp"],
                   "ensemble_cov90": found["ensemble"]["coverage_90"],
                   "ensemble_better": found["ensemble"]["crps_pp"] < base["crps_pp"]}
            if "calibrated" in found:
                row.update({"calibrated_crps_pp": found["calibrated"]["crps_pp"],
                            "calibrated_cov90": found["calibrated"]["coverage_90"],
                            "calibrated_better": found["calibrated"]["crps_pp"] < base["crps_pp"]})
            h2h.append(row)
    if h2h:
        pl.DataFrame(h2h).write_csv(out_dir / "head_to_head.csv")
    return {"summary": summary, "stacking": stacking, "head_to_head": h2h}
