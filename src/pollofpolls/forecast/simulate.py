"""Turn election-day vote-share draws into seat and coalition simulations."""

from __future__ import annotations

import numpy as np
from scipy import stats

from .seats import allocate_seats_matrix


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def simulate_electorates(pi: np.ndarray, parties: list[str], electorate_cfg: list[dict],
                         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw electorate wins for the configured seats. Returns (electorates (S,K), independent_seats (S,))."""
    S, K = pi.shape
    idx = {p: i for i, p in enumerate(parties)}
    electorates = np.zeros((S, K), dtype=int)
    independents = np.zeros(S, dtype=int)
    for e in electorate_cfg:
        party = e["party"]
        if party not in idx:
            continue
        k = idx[party]
        p_ref, ind_ref = float(e["p"]), float(e.get("independent_p", 0.0))
        if p_ref + ind_ref > 1.0 + 1e-9:
            raise ValueError(f"{e['electorate']}: p + independent_p exceeds 1")
        # three mutually exclusive outcomes: the party, an independent, or anyone else. At the reference vote
        # share they have exactly the configured probabilities; as the party's vote moves, its chance moves
        # on the logit scale and the independent keeps the same share of the remaining probability.
        p_party = _sigmoid(_logit(p_ref) + e.get("slope", 0.0) * 100.0 * (pi[:, k] - e.get("at_share", pi[:, k].mean())))
        p_ind = ind_ref * (1.0 - p_party) / (1.0 - p_ref) if ind_ref > 0 else np.zeros(S)
        u = rng.uniform(size=S)
        party_win = u < p_party
        ind_win = (~party_win) & (u < p_party + p_ind)
        electorates[party_win, k] += 1
        independents[ind_win] += 1
    return electorates, independents


def simulate_seats(pi: np.ndarray, parties: list[str], electorate_cfg: list[dict], n_sims: int,
                   rng: np.random.Generator, other_name: str = "Other") -> dict:
    """Seat simulation for election-day share draws ``pi`` (S, K). Returns arrays keyed by name."""
    S = pi.shape[0]
    take = rng.choice(S, size=min(n_sims, S), replace=n_sims > S) if n_sims != S else np.arange(S)
    pi = pi[take]
    votes = np.clip(pi, 0.0, None)
    if other_name in parties:
        votes[:, parties.index(other_name)] = 0.0   # "Other" is many parties, none of which qualifies
    electorates, independents = simulate_electorates(pi, parties, electorate_cfg, rng)
    seats = allocate_seats_matrix(votes, electorates, independent_seats=independents)
    total = seats.sum(1) + independents
    return {"pi": pi, "seats": seats, "electorates": electorates, "independents": independents, "total": total}


def apply_fundamentals(pi: np.ndarray, pm_idx: int, prev_share: float, prior: dict,
                       rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Importance-resample election-day draws under the Student-t prior on the PM party's swing."""
    swing = pi[:, pm_idx] - prev_share
    logw = stats.t.logpdf(swing, df=prior["df"], loc=prior["mean"], scale=prior["sd"])
    w = np.exp(logw - logw.max())
    w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    take = rng.choice(len(pi), size=len(pi), replace=True, p=w)
    return pi[take], float(ess)
