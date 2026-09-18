"""Spread calibration of the ensemble forecast (EMOS-style post-processing, Gneiting et al. 2005).

The model's forecast distribution is rescaled about its mean on the log-ratio scale by a factor
c(h) = exp(a + b * log h), where h is the number of weeks before the election. (a, b) minimise the total
CRPS over backtest cases. Rescaling on the log-ratio scale keeps every draw a valid set of vote shares.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from ..prep.marshal import alr, alr_inverse
from .scoring import crps, keep_index


def inflate(draws: np.ndarray, c: float) -> np.ndarray:
    """Rescale share draws (S, K) about their log-ratio mean by factor c."""
    th = alr(draws)
    m = th.mean(0)
    return alr_inverse(m + c * (th - m))


def factor(params, h: float) -> float:
    """Spread factor for horizon h weeks; params is {"a", "b"}, an (a, b) sequence, or None (no calibration)."""
    if params is None:
        return 1.0
    a, b = (params["a"], params["b"]) if isinstance(params, dict) else (float(params[0]), float(params[1]))
    return float(np.exp(a + b * np.log(max(h, 1.0))))


def total_crps(params: tuple, cases: list[tuple[np.ndarray, np.ndarray, list[str], float]]) -> float:
    total = 0.0
    for draws, outcome, parties, h in cases:
        d = inflate(draws, factor(params, h))
        keep = keep_index(parties)
        total += np.mean([crps(d[:, k], outcome[k]) for k in keep])
    return total


def fit_spread(cases: list[tuple[np.ndarray, np.ndarray, list[str], float]]) -> dict:
    """Fit (a, b) by minimising the mean CRPS; bounded so the factor stays within [0.5, 4] for h in [1, 52]."""
    res = optimize.minimize(total_crps, x0=np.zeros(2), args=(cases,), method="Nelder-Mead",
                            options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 400})
    a, b = res.x
    b = float(np.clip(b, -0.5, 0.5))
    a = float(np.clip(a, np.log(0.5), np.log(4.0)))
    return {"a": a, "b": b}
