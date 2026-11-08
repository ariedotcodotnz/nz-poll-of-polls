"""Leave-one-election-out backtests of the official-count forecast, against rules of thumb.

For each past election, the models are refitted on the other elections and every seat's official result is
forecast from its election-night count (or, where that count is not archived, from its official ordinary
votes). The forecast election-wide shift is a fresh draw, exactly as for the live forecast. The national
special-vote volume is given with the error of an election-night estimate.

Rules of thumb, each given the same special-vote volume:

    election night   specials split exactly like the starting count
    seat's last      specials differ from the starting count as they did in the seat's previous election
    average          specials differ from the starting count by the average of every past seat and election

Each rule of thumb becomes a forecast distribution with a normal error whose sd is its root-mean-square error on
the training elections, so all methods are scored on the same terms: CRPS of the final margin (votes), absolute
error, interval coverage, and the Brier score and log loss of the winner.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..eval.scoring import crps
from .data import ElectorateCount
from .model import (
    Posterior,
    build_official_count_data,
    build_specials_data,
    fit_official_count,
    fit_specials,
    log_ratio,
    softmax0,
)
from .simulate import Target, VolumeShift, simulate

METHODS = ["model", "election_night", "seat_last", "average"]
METHOD_LABELS = {"model": "This model", "election_night": "Specials split like election night",
                 "seat_last": "Seat's swing at the last election", "average": "Average swing of past seats"}


def _start(c: ElectorateCount, pair: list[str]) -> tuple[np.ndarray, bool]:
    if c.preliminary is not None:
        return c.grouped(c.preliminary, pair), True
    return c.grouped(c.ordinary, pair), False


def _pooled_volume(counts: list[ElectorateCount], year: int, pair: list[str]) -> tuple[float, float]:
    """Log specials per ordinary vote and logit of the overseas share, pooled over the seats of ``year``."""
    cs = [c for c in counts if c.year == year and c.grouped(c.ordinary, pair) is not None]
    s = sum(c.special_total.sum() for c in cs)
    o = sum(c.ordinary.sum() for c in cs)
    ov = sum(c.specials["overseas"].sum() for c in cs)
    return float(np.log(s / o)), float(np.log(ov / (s - ov)))


def _shift(c: ElectorateCount, pair: list[str]) -> np.ndarray:
    """How a seat's specials differed from its ordinary vote, on the log-ratio scale."""
    return log_ratio(c.grouped(c.special_total, pair)) - log_ratio(c.grouped(c.ordinary, pair))


def _rule(method: str, start: np.ndarray, volume: float, c: ElectorateCount, train: list[ElectorateCount],
          pair: list[str]) -> float | None:
    """A rule of thumb's final margin (pair[1] - pair[0]) from ``start`` and ``volume`` special votes."""
    if method == "election_night":
        d = np.zeros(2)
    elif method == "average":
        d = np.mean([_shift(t, pair) for t in train], axis=0)
    else:
        past = [t for t in train if t.key == c.key and t.year != c.year]
        if not past:
            return None
        before = [t for t in past if t.year < c.year]
        d = _shift(max(before, key=lambda t: t.year) if before else min(past, key=lambda t: t.year), pair)
    final = start + volume * softmax0(log_ratio(start) + d)
    return float(final[1] - final[0])


def _rule_sd(method: str, train: list[ElectorateCount], pair: list[str]) -> float:
    """Root-mean-square margin error of a rule of thumb on the training seats, given their actual volume."""
    errs = []
    for c in train:
        start, _ = _start(c, pair)
        pred = _rule(method, start, c.special_total.sum(), c, train, pair)
        if pred is not None:
            final = c.grouped(c.total, pair)
            errs.append(pred - (final[1] - final[0]))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def _scores(draws: np.ndarray, actual: float) -> dict:
    lo50, hi50, lo90, hi90 = np.quantile(draws, [0.25, 0.75, 0.05, 0.95])
    p = float(np.clip((draws > 0).mean(), 0.005, 0.995))
    won = actual > 0
    return {"crps_votes": crps(draws, actual), "abs_error": float(abs(np.median(draws) - actual)),
            "cover50": bool(lo50 <= actual <= hi50), "cover90": bool(lo90 <= actual <= hi90),
            "p_second": p, "brier": float((p - won) ** 2), "log_loss": float(-np.log(p if won else 1 - p)),
            "median": float(np.median(draws))}


def backtest(counts: list[ElectorateCount], pair: list[str], priors: dict, mcmc: dict, *, n: int = 8000,
             volume_error_sd: float = 0.04, overseas_error_sd: float = 0.10, df: float | None = 4,
             seed: int = 0, log=print) -> list[dict]:
    """One row per seat, past election and method."""
    usable = [c for c in counts if c.official and c.grouped(c.ordinary, pair) is not None]
    years = sorted({c.year for c in usable})
    rows: list[dict] = []
    for target_year in years:
        train = [c for c in usable if c.year != target_year]
        test = [c for c in usable if c.year == target_year]
        if len({c.year for c in train}) < 2:
            continue
        data = build_specials_data(train, pair)
        post = fit_specials(data, priors, mcmc)
        oc = build_official_count_data(data)
        official: Posterior | None = fit_official_count(oc, priors, mcmc) if len(oc.prelim) else None
        earlier = [y for y in data.years if y < target_year]
        ref = max(earlier) if earlier else min(data.years)
        rng = np.random.default_rng(seed + target_year)
        v_t, o_t = _pooled_volume(usable, target_year, pair)
        v_r, o_r = _pooled_volume(usable, ref, pair)
        volume = VolumeShift(ref, v_t - v_r + rng.normal(0, volume_error_sd, n),
                             o_t - o_r + rng.normal(0, overseas_error_sd, n))
        targets = [Target(c.electorate, c.key, *_start(c, pair)) for c in test]
        sim = simulate(post, data, targets, volume, official=official, df=df, n=n, seed=seed + target_year,
                       max_addition=float(priors["official_count"].get("max_addition", 0.2)))
        log(f"backtest {target_year}: fitted on {data.years} (divergences {post.diagnostics['n_divergent']}"
            f"{'' if official is None else ', ' + str(official.diagnostics['n_divergent'])})")
        sds = {m: _rule_sd(m, train, pair) for m in METHODS[1:]}
        for j, c in enumerate(test):
            start, from_prelim = targets[j].start, targets[j].official_change
            final = c.grouped(c.total, pair)
            actual = float(final[1] - final[0])
            spec_actual = c.grouped(c.special_total, pair)
            base = {"year": target_year, "electorate": c.electorate, "start": "election night" if from_prelim
                    else "official ordinary", "start_margin": float(start[1] - start[0]), "actual_margin": actual,
                    "actual_special_share": float(spec_actual[1] / spec_actual[:2].sum())}
            margin = sim.final[:, j, 1] - sim.final[:, j, 0]
            spec = sim.specials[:, j].sum(1)
            share = spec[:, 1] / np.maximum(spec[:, 0] + spec[:, 1], 1)
            rows.append({**base, "method": "model", **_scores(margin, actual),
                         "crps_special_share_pp": crps(share * 100, base["actual_special_share"] * 100)})
            # the rules of thumb get the special-vote volume the model expects
            volume_mean = float(sim.specials[:, j].sum((1, 2)).mean())
            for m in METHODS[1:]:
                point = _rule(m, start, volume_mean, c, train, pair)
                if point is None or not np.isfinite(sds[m]):
                    continue
                draws = stats.norm(point, sds[m]).rvs(size=n, random_state=rng)
                rows.append({**base, "method": m, **_scores(draws, actual), "crps_special_share_pp": None})
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    """Average scores per method over the seats every method could forecast."""
    cases = {}
    for r in rows:
        cases.setdefault((r["year"], r["electorate"]), set()).add(r["method"])
    common = {k for k, ms in cases.items() if set(METHODS) <= ms}
    out = []
    for m in METHODS:
        rs = [r for r in rows if r["method"] == m and (r["year"], r["electorate"]) in common]
        if not rs:
            continue
        flips = [r for r in rs if np.sign(r["start_margin"]) != np.sign(r["actual_margin"])]
        out.append({"method": m, "label": METHOD_LABELS[m], "n_cases": len(rs),
                    "crps_votes": float(np.mean([r["crps_votes"] for r in rs])),
                    "mae_votes": float(np.mean([r["abs_error"] for r in rs])),
                    "coverage_50": float(np.mean([r["cover50"] for r in rs])),
                    "coverage_90": float(np.mean([r["cover90"] for r in rs])),
                    "brier": float(np.mean([r["brier"] for r in rs])),
                    "log_loss": float(np.mean([r["log_loss"] for r in rs])),
                    "n_flipped": len(flips)})
    return out
