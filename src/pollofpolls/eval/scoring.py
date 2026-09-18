"""Proper scoring rules for vote-share forecasts, and CRPS stacking.

All scores are computed on the share scale over the tracked parties, excluding the residual "Other"
category. CRPS and the energy score are reported in percentage points (lower is better). A log score on the
log-ratio scale was tried first and rejected: it is dominated by parties on 0.1-0.5%, whose log-ratios swing
wildly for errors that are irrelevant to seats.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats


def keep_index(parties: list[str]) -> list[int]:
    return [k for k, p in enumerate(parties) if p != "Other"]


def crps(draws: np.ndarray, outcome: float) -> float:
    """Sample CRPS in the units of the draws: E|X - y| - E|X - X'| / 2 (sorted-sample formula, exact)."""
    x = np.sort(np.asarray(draws, dtype=float))
    n = x.size
    i = np.arange(1, n + 1)
    spread = 2.0 * np.sum((2 * i - n - 1) * x) / n ** 2
    return float(np.mean(np.abs(x - outcome)) - 0.5 * spread)


def energy_score(draws: np.ndarray, outcome: np.ndarray, max_pairs: int = 1000, seed: int = 0) -> float:
    """Multivariate energy score E||X - y|| - E||X - X'|| / 2 (Gneiting & Raftery 2007)."""
    d = np.asarray(draws, dtype=float)
    term1 = np.linalg.norm(d - outcome, axis=1).mean()
    rng = np.random.default_rng(seed)
    sub = d[rng.choice(len(d), size=min(len(d), max_pairs), replace=False)]
    term2 = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1).mean()
    return float(term1 - 0.5 * term2)


def gaussian_log_score(draws: np.ndarray, outcome: np.ndarray) -> float:
    """Log density of the outcome under a multivariate normal fitted to the draws (share scale)."""
    mean = draws.mean(0)
    cov = np.cov(draws, rowvar=False) + 1e-8 * np.eye(draws.shape[1])
    return float(stats.multivariate_normal(mean, cov, allow_singular=True).logpdf(outcome))


def interval_covers(draws: np.ndarray, outcome: float, level: float) -> bool:
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return bool(lo <= outcome <= hi)


def pit(draws: np.ndarray, outcome: float) -> float:
    return float((np.asarray(draws) <= outcome).mean())


def brier(prob: float, outcome: bool) -> float:
    return float((prob - float(outcome)) ** 2)


def score_forecast(draws: np.ndarray, parties: list[str], outcome: np.ndarray) -> dict:
    """All scores for one forecast: draws (S, K) proportions vs outcome (K,)."""
    keep = keep_index(parties)
    d, y = draws[:, keep], outcome[keep]
    out = {
        "crps_pp": float(np.mean([crps(d[:, j], y[j]) for j in range(len(keep))]) * 100),
        "energy_pp": energy_score(d * 100, y * 100),
        "log_score_share": gaussian_log_score(d, y),
        "mae_pp": float(np.abs(d.mean(0) - y).mean() * 100),
        "coverage_50": float(np.mean([interval_covers(d[:, j], y[j], 0.5) for j in range(len(keep))])),
        "coverage_90": float(np.mean([interval_covers(d[:, j], y[j], 0.9) for j in range(len(keep))])),
    }
    for j, k in enumerate(keep):
        p = parties[k]
        out[f"crps_pp[{p}]"] = crps(d[:, j], y[j]) * 100
        out[f"error_pp[{p}]"] = float((d[:, j].mean() - y[j]) * 100)
        out[f"sd_pp[{p}]"] = float(d[:, j].std() * 100)
        out[f"pit[{p}]"] = pit(d[:, j], y[j])
    return out


# ------------------------------------------------------------------------------------------ stacking
def crps_components(draw_sets: list[np.ndarray], outcome: np.ndarray, keep: list[int], m: int = 600,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pieces of the mixture CRPS, summed over parties: A[a] = E|X_a - y|, B[a, b] = E|X_a - X_b'|.

    For weights w the mixture CRPS is w @ A - w @ B @ w / 2, a convex quadratic on the simplex.
    """
    rng = np.random.default_rng(seed)
    subs = [s[rng.choice(len(s), size=m, replace=len(s) < m)][:, keep] for s in draw_sets]
    y = outcome[keep]
    M = len(subs)
    A = np.array([np.abs(s - y).mean(0).sum() for s in subs])
    B = np.zeros((M, M))
    for a in range(M):
        for b in range(a, M):
            B[a, b] = B[b, a] = np.abs(subs[a][:, None, :] - subs[b][None, :, :]).mean((0, 1)).sum()
    return A, B


def mixture_crps(w: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    return float(w @ A - 0.5 * w @ B @ w)


def crps_stacking_weights(components: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Weights on the simplex minimising the total mixture CRPS over the cases (stacking, Yao et al. 2018)."""
    M = len(components[0][0])
    if M == 1:
        return np.ones(1)

    def total(w):
        return sum(mixture_crps(w, A, B) for A, B in components)

    res = optimize.minimize(total, np.full(M, 1.0 / M), method="SLSQP", bounds=[(0.0, 1.0)] * M,
                            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}])
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


def mixture_draws(draw_sets: list[np.ndarray], weights: np.ndarray, size: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(size, np.asarray(weights) / np.sum(weights))
    parts = [s[rng.choice(len(s), size=c, replace=True)] for s, c in zip(draw_sets, counts) if c > 0]
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out
