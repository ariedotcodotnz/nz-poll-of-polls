"""Seat allocation under New Zealand's MMP rules (port of nzelect::allocate_seats).

* A party qualifies for list seats with >= 5% of the party vote or at least one electorate seat.
* 120 seats (less any electorates won by independents / non-qualifying candidates) are allocated
  among qualifying parties by the Sainte-Laguë method (divisors 1, 3, 5, ...).
* A party keeps all its electorate seats; if it won more electorates than its proportional
  entitlement the House grows (overhang).
"""

from __future__ import annotations

import numpy as np


def allocate_seats(votes: dict[str, float], electorates: dict[str, int] | None = None, nseats: int = 120,
                   threshold: float = 0.05, independent_seats: int = 0) -> dict[str, int]:
    """Return final seats per party (electorate + list), including overhang."""
    electorates = electorates or {}
    total = float(sum(votes.values()))
    shares = {p: v / total for p, v in votes.items()} if total > 0 else {p: 0.0 for p in votes}
    qualifying = [p for p in votes if shares[p] >= threshold or electorates.get(p, 0) > 0]
    pool = nseats - independent_seats
    alloc = {p: 0 for p in votes}
    if qualifying and pool > 0:
        q_votes = np.array([votes[p] for p in qualifying], dtype=float)
        divisors = 2 * np.arange(pool) + 1                      # 1, 3, 5, ...
        quotients = q_votes[:, None] / divisors[None, :]        # (parties, pool)
        order = np.argsort(-quotients, axis=None, kind="stable")[:pool]
        rows = order // pool
        counts = np.bincount(rows, minlength=len(qualifying))
        for p, c in zip(qualifying, counts):
            alloc[p] = int(c)
    return {p: max(alloc[p], int(electorates.get(p, 0))) for p in votes}


def allocate_seats_matrix(votes: np.ndarray, electorates: np.ndarray, nseats: int = 120, threshold: float = 0.05,
                          independent_seats: np.ndarray | None = None) -> np.ndarray:
    """Vectorised allocation for many simulations. votes (S, K) proportions; electorates (S, K) ints."""
    S, K = votes.shape
    independent_seats = np.zeros(S, dtype=int) if independent_seats is None else independent_seats
    out = np.zeros((S, K), dtype=int)
    qualifying = (votes >= threshold) | (electorates > 0)
    max_pool = int(nseats)
    divisors = 2 * np.arange(max_pool) + 1
    q_votes = np.where(qualifying, votes, 0.0)
    quotients = q_votes[:, :, None] / divisors[None, None, :]        # (S, K, pool)
    flat = quotients.reshape(S, -1)
    order = np.argsort(-flat, axis=1, kind="stable")                  # (S, K*pool)
    rows = order // max_pool                                          # party index of each ranked quotient
    for s in range(S):
        pool = max_pool - int(independent_seats[s])
        top = rows[s, :pool]
        out[s] = np.bincount(top, minlength=K)
    out = np.where(qualifying, out, 0)
    return np.maximum(out, electorates)
