"""Pipeline stages: fetch -> prep -> fit -> forecast -> backtest -> report."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from .cache import StageCache, fingerprint
from .config import Config
from .data.results import election_results_from_polls, load_reference_results, verify_results
from .data.wikipedia import Poll, fetch_page, parse_page_file
from .forecast.coalitions import coalition_table, kingmaker_probabilities, party_seat_summary
from .forecast.simulate import apply_fundamentals, simulate_seats
from .prep.marshal import Dataset, alr_inverse, build_dataset
from .prep.polls_table import build_polls_table

PAGE_YEARS = [2011, 2014, 2017, 2020, 2023, 2026]


def _log(msg: str) -> None:
    print(f"[pollofpolls] {msg}", flush=True)


# ----------------------------------------------------------------------------------------- fetch
def fetch(cfg: Config, force: bool = False) -> list[Path]:
    paths = []
    for y in PAGE_YEARS:
        p = fetch_page(y, cfg.paths.raw, force=force)
        _log(f"fetched {y}: {p.stat().st_size // 1024} KB")
        paths.append(p)
    return paths


# ------------------------------------------------------------------------------------------ prep
def load_polls(cfg: Config) -> list[Poll]:
    polls = []
    for y in PAGE_YEARS:
        p = cfg.paths.raw / f"{y}.html"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run `pollofpolls fetch` first")
        polls.extend(parse_page_file(p, y))
    return polls


def prep(cfg: Config, force: bool = False) -> tuple[pl.DataFrame, dict]:
    """Parse the cached pages into the poll table and the results table; verify results."""
    cfg.paths.processed.mkdir(parents=True, exist_ok=True)
    polls = load_polls(cfg)
    table = build_polls_table(polls, cfg)
    scraped = election_results_from_polls(polls)
    reference = load_reference_results(cfg.paths.reference / "election_results.csv")
    problems = verify_results(scraped, reference)
    if problems:
        raise RuntimeError("scraped election results disagree with data/reference/election_results.csv:\n  "
                           + "\n  ".join(problems))
    # prefer the vendored (2 d.p.) values, fall back to scraped ones for parties not vendored
    results = {y: {**scraped.get(y, {}), **reference.get(y, {})} for y in set(scraped) | set(reference)}
    table.write_parquet(cfg.paths.processed / "polls.parquet")
    (cfg.paths.processed / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    n_polls = table["poll_id"].n_unique()
    _log(f"poll table: {n_polls} polls, {table['pollster'].n_unique()} pollsters, latest {table['mid_date'].max()}")
    return table, results


def load_prepped(cfg: Config) -> tuple[pl.DataFrame, dict]:
    table = pl.read_parquet(cfg.paths.processed / "polls.parquet")
    results = {int(k): v for k, v in json.loads((cfg.paths.processed / "results.json").read_text()).items()}
    return table, results


# ------------------------------------------------------------------------------------------- fit
def ensemble_weights(cfg: Config, min_weight: float = 0.01) -> dict[str, float]:
    """Published ensemble: CRPS-stacking weights when every candidate has one, otherwise equal weights.

    Candidates with a stacking weight below ``min_weight`` are dropped (and not fitted).
    """
    candidates = cfg.ensemble
    stack = load_stacking(cfg)
    if stack and all(v in stack for v in candidates):
        kept = {v: w for v, w in stack.items() if v in candidates and w >= min_weight}
        total = sum(kept.values())
        return {v: w / total for v, w in kept.items()}
    if stack:
        _log(f"stacking weights do not cover {candidates}; using equal weights (run `pollofpolls backtest`)")
    return {v: 1.0 / len(candidates) for v in candidates}


def fit_variants(cfg: Config, variants: list[str] | None = None, force: bool = False,
                 mcmc_override: dict | None = None, progress: bool = True) -> dict[str, "FitResult"]:
    from .eval.backtest import fit_fingerprint
    from .model.fit import FitResult, fit

    variants = variants or list(ensemble_weights(cfg))
    table, results = load_prepped(cfg)
    target = cfg.forecast_election.year
    ds = build_dataset(table, results, cfg, target)
    ds.save(cfg.paths.processed / f"dataset_{target}")
    _log(f"dataset {target}: {ds.N} polls, {ds.K} parties {ds.parties}, {ds.T} weeks, cutoff {ds.cutoff}")
    mcmc_cfg = {**cfg.mcmc, **(mcmc_override or {})}
    cache = StageCache(cfg.paths.processed / ".stamps")
    fits = {}
    for name in variants:
        variant = cfg.variant(name)
        prefix = cfg.paths.processed / f"fit_{name}_{target}"
        inputs = [fit_fingerprint(cfg, ds, variant, mcmc_cfg)]

        def _run(variant=variant, prefix=prefix):
            _log(f"fitting variant {variant['name']} ...")
            fr = fit(ds, variant, cfg.priors, mcmc_cfg, cfg.model_cfg.get("election_obs_sd", 0.01), progress=progress)
            fr.save(prefix)
            dg = fr.diagnostics
            _log(f"  done in {dg['seconds']}s  rhat {dg['max_rhat']:.3f}  min ESS {dg['min_ess_bulk']:.0f}  "
                 f"divergences {dg['n_divergent']}  mean leapfrog steps {dg['mean_num_steps']:.0f}")

        ran = cache.run(f"fit_{name}_{target}", inputs, [prefix.with_suffix(".npz")], _run, force=force)
        if not ran:
            _log(f"variant {name}: cached")
        fits[name] = FitResult.load(prefix)
    return fits


# --------------------------------------------------------------------------------------- forecast
def _mixture(draw_sets: list[np.ndarray], weights: list[float], rng: np.random.Generator, size: int) -> np.ndarray:
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    counts = rng.multinomial(size, w)
    parts = [ds[rng.choice(len(ds), size=c, replace=True)] for ds, c in zip(draw_sets, counts) if c > 0]
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def load_stacking(cfg: Config) -> dict[str, float] | None:
    p = cfg.paths.output / "backtest" / "stacking.json"
    if p.exists():
        return json.loads(p.read_text()).get("weights") or None
    return None


def load_spread(cfg: Config) -> dict | None:
    """Spread-calibration parameters, if enabled in config/model.yml and fitted by the backtests."""
    if not cfg.model_cfg.get("forecast", {}).get("calibrate_spread", False):
        return None
    p = cfg.paths.output / "backtest" / "stacking.json"
    return json.loads(p.read_text()).get("spread") if p.exists() else None


def forecast(cfg: Config, variants: list[str] | None = None, seed: int = 2026) -> dict:
    """Combine fitted variants (stacked), simulate seats, write output tables and summary.json."""
    from .model.fit import FitResult, posterior_var

    weights = ensemble_weights(cfg)
    variants = variants or list(weights)
    w = [weights.get(v, 0.0) for v in variants]
    if sum(w) <= 0:
        w = [1.0 / len(variants)] * len(variants)
    target = cfg.forecast_election.year
    ds = Dataset.load(cfg.paths.processed / f"dataset_{target}")
    rng = np.random.default_rng(seed)
    fits = {v: FitResult.load(cfg.paths.processed / f"fit_{v}_{target}") for v in variants}

    prior = {"df": cfg.priors["fundamentals"]["df"], "mean": ds.fund_mean, "sd": ds.fund_sd}
    target_sets, now_sets, fund_ess = [], [], {}
    for v in variants:
        fr = fits[v]
        pt = fr.pi_target
        if cfg.variant(v).get("fundamentals") and cfg.variant(v).get("obs") == "kalman":
            pt, ess = apply_fundamentals(pt, ds.pm_party_idx, ds.pm_prev_share, prior, rng)
            fund_ess[v] = ess
        target_sets.append(pt)
        now_sets.append(fr.pi_last)
    size = int(cfg.model_cfg.get("seats", {}).get("n_sims", 4000))
    pi_target = _mixture(target_sets, w, rng, size)
    pi_now = _mixture(now_sets, w, rng, size)
    spread = load_spread(cfg)
    weeks_to_go = max((cfg.forecast_election.date - date.today()).days / 7.0, 1.0)
    if spread:
        from .eval.calibration import factor, inflate
        pi_target = inflate(pi_target, factor(spread, weeks_to_go))
        pi_now = inflate(pi_now, factor(spread, 1.0))
        _log(f"spread calibration: x{factor(spread, weeks_to_go):.2f} at {weeks_to_go:.1f} weeks, "
             f"x{factor(spread, 1.0):.2f} for 'held now'")

    electorate_cfg = cfg.electorates_cfg["electorates"]
    coal_cfg = cfg.electorates_cfg["coalitions"]
    sim_target = simulate_seats(pi_target, ds.parties, electorate_cfg, size, rng)
    sim_now = simulate_seats(pi_now, ds.parties, electorate_cfg, size, rng)

    out = cfg.paths.output
    out.mkdir(parents=True, exist_ok=True)
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]

    def share_rows(pi, when):
        rows = []
        for k, p in enumerate(ds.parties):
            q = np.quantile(pi[:, k], qs)
            rows.append({"when": when, "party": p, "mean": float(pi[:, k].mean()), "q05": q[0], "q25": q[1],
                         "q50": q[2], "q75": q[3], "q95": q[4], "p_over_5pct": float((pi[:, k] >= 0.05).mean())})
        return rows

    forecast_df = pl.DataFrame(share_rows(pi_target, "election_day") + share_rows(pi_now, "now"))
    forecast_df.write_csv(out / "forecast.csv")
    seats_rows = [{"when": "election_day", **r} for r in party_seat_summary(sim_target["seats"], ds.parties)] + \
                 [{"when": "now", **r} for r in party_seat_summary(sim_now["seats"], ds.parties)]
    pl.DataFrame(seats_rows).write_csv(out / "seats.csv")
    coal_rows = [{"when": "election_day", **r} for r in coalition_table(sim_target["seats"], ds.parties, coal_cfg, sim_target["total"])] + \
                [{"when": "now", **r} for r in coalition_table(sim_now["seats"], ds.parties, coal_cfg, sim_now["total"])]
    pl.DataFrame([{k: (", ".join(v) if isinstance(v, list) else v) for k, v in r.items()} for r in coal_rows]).write_csv(out / "coalitions.csv")
    np.savez_compressed(out / "seat_sims.npz", seats_election=sim_target["seats"], total_election=sim_target["total"],
                        seats_now=sim_now["seats"], total_now=sim_now["total"], pi_election=pi_target, pi_now=pi_now,
                        parties=np.array(ds.parties))

    # weekly trend of the ensemble: quantiles of the same weighted mixture used for the seat simulation
    mix_path = _mixture([fits[v].pi_thin for v in variants], w, rng, 2000)          # (2000, T, K)
    q_path = np.quantile(mix_path, qs, axis=0)                                        # (5, T, K)
    mean_path = mix_path.mean(0)
    trend_rows = []
    for k, p in enumerate(ds.parties):
        for t, wk in enumerate(ds.weeks):
            trend_rows.append({"party": p, "week": wk.isoformat(), "mean": float(mean_path[t, k]),
                               "q05": float(q_path[0, t, k]), "q25": float(q_path[1, t, k]),
                               "q50": float(q_path[2, t, k]), "q75": float(q_path[3, t, k]),
                               "q95": float(q_path[4, t, k]), "has_data": t <= ds.last_data_t})
    # 4 decimals (0.01 pp) keeps the weekly CI commit small
    pl.DataFrame(trend_rows).with_columns(pl.col(["mean", "q05", "q25", "q50", "q75", "q95"]).round(4)) \
        .write_csv(out / "trend.csv")

    # house effects in percentage points, evaluated at the current mean state
    ref = variants[0]
    theta_bar = fits[ref].theta_last.mean(0)
    base_pi = alr_inverse(theta_bar)
    house_rows = []
    hb = posterior_var(fits[ref].hyper, "house_base")
    if hb is not None:
        hb_mean = hb.reshape(-1, *hb.shape[2:]).mean(0)          # (houses, K) logit scale
        hb_alr = hb_mean[:, 1:] - hb_mean[:, :1]
        for h, label in enumerate(ds.houses):
            shifted = alr_inverse(theta_bar + hb_alr[h])
            for k, p in enumerate(ds.parties):
                house_rows.append({"house": label, "party": p, "effect_pp": float((shifted[k] - base_pi[k]) * 100)})
    pl.DataFrame(house_rows).write_csv(out / "house_effects.csv")

    blocs = {"Right bloc": ["National", "ACT", "NZ First"], "Left bloc": ["Labour", "Green", "Te Pāti Māori", "NZ First"]}
    kingmaker = kingmaker_probabilities(sim_target["seats"], ds.parties, sim_target["total"], blocs, "NZ First")
    table, _ = load_prepped(cfg)
    latest = table.filter(pl.col("cycle") == target)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "election_date": cfg.forecast_election.date.isoformat(),
        "days_to_election": (cfg.forecast_election.date - date.today()).days,
        "cutoff": ds.cutoff.isoformat(),
        "variants": variants, "weights": dict(zip(variants, w)), "fundamentals_ess": fund_ess,
        "spread_calibration": spread,
        "fundamentals_prior": {"mean": ds.fund_mean, "sd": ds.fund_sd, "n_elections": ds.fund_n},
        "n_polls_cycle": int(latest["poll_id"].n_unique()), "n_polls_total": int(ds.N),
        "latest_poll": str(latest["mid_date"].max()),
        "pollsters_cycle": sorted(latest["pollster"].unique().to_list()),
        "parties": ds.parties,
        "election_day": {r["party"]: r for r in share_rows(pi_target, "election_day")},
        "now": {r["party"]: r for r in share_rows(pi_now, "now")},
        "seats_election_day": party_seat_summary(sim_target["seats"], ds.parties),
        "coalitions_election_day": coalition_table(sim_target["seats"], ds.parties, coal_cfg, sim_target["total"]),
        "coalitions_now": coalition_table(sim_now["seats"], ds.parties, coal_cfg, sim_now["total"]),
        "kingmaker": kingmaker,
        "expected_house_size": float(sim_target["total"].mean()),
        "p_overhang": float((sim_target["total"] > 120).mean()),
        "diagnostics": {v: fits[v].diagnostics for v in variants},
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    _log("forecast written to output/")
    return summary


# --------------------------------------------------------------------------------------- backtest
def backtest(cfg: Config, variants: list[str] | None = None, targets: list[int] | None = None,
             horizons: list[int] | None = None, force: bool = False, progress: bool = False,
             run: bool = True, aggregate: bool = True) -> None:
    """Run (and/or aggregate) rolling-origin backtests. Cases are cached, so re-running is cheap."""
    from .eval.backtest import aggregate as _aggregate, run_backtests

    bt = cfg.model_cfg.get("backtest", {})
    variants = variants or bt.get("variants", ["legacy"] + cfg.ensemble)
    targets = targets or bt.get("targets", [2017, 2020, 2023])
    horizons = horizons or bt.get("horizons_weeks", [26, 8, 4, 1])
    out_dir = cfg.paths.output / "backtest"
    if run:
        mcmc_cfg = {**cfg.mcmc, **bt.get("mcmc", {})}
        table, results = load_prepped(cfg)
        run_backtests(cfg, table, results, variants, targets, horizons, out_dir, mcmc_cfg, force, progress)
    if aggregate:
        out = _aggregate(cfg, out_dir, cfg.ensemble)
        print(out["summary"])
        _log(f"stacking weights (all cases): {out['stacking']['weights']}")
        for row in out["head_to_head"]:
            _log(f"  {row['target']} h={row['horizon_weeks']}w: ensemble CRPS {row['ensemble_crps_pp']:.2f} vs "
                 f"legacy {row['legacy_crps_pp']:.2f}")
