"""Coalition and majority probabilities from simulated seat matrices."""

from __future__ import annotations

import numpy as np


def majority_threshold(total_seats: np.ndarray) -> np.ndarray:
    return total_seats // 2 + 1


def coalition_table(seats: np.ndarray, parties: list[str], coalitions: list[dict],
                    total_seats: np.ndarray) -> list[dict]:
    """For each configured coalition: seat mean, 90% interval and probability of a majority."""
    idx = {p: i for i, p in enumerate(parties)}
    need = majority_threshold(total_seats)
    rows = []
    for c in coalitions:
        cols = [idx[p] for p in c["parties"] if p in idx]
        s = seats[:, cols].sum(axis=1)
        rows.append({
            "name": c["name"], "parties": c["parties"], "seats_mean": float(s.mean()),
            "seats_q05": float(np.quantile(s, 0.05)), "seats_q95": float(np.quantile(s, 0.95)),
            "p_majority": float((s >= need).mean()),
        })
    return rows


def party_seat_summary(seats: np.ndarray, parties: list[str]) -> list[dict]:
    rows = []
    for k, p in enumerate(parties):
        s = seats[:, k]
        rows.append({"party": p, "seats_mean": float(s.mean()), "seats_q05": float(np.quantile(s, 0.05)),
                     "seats_q50": float(np.quantile(s, 0.5)), "seats_q95": float(np.quantile(s, 0.95)),
                     "p_any_seats": float((s > 0).mean())})
    return rows


def kingmaker_probabilities(seats: np.ndarray, parties: list[str], total_seats: np.ndarray,
                            blocs: dict[str, list[str]], pivot: str) -> dict[str, float]:
    """Probability that the pivot party is needed by either bloc, or that a bloc has a majority without it."""
    idx = {p: i for i, p in enumerate(parties)}
    need = majority_threshold(total_seats)
    out = {}
    for name, members in blocs.items():
        base = seats[:, [idx[p] for p in members if p in idx and p != pivot]].sum(1)
        with_pivot = base + (seats[:, idx[pivot]] if pivot in idx else 0)
        out[f"{name} majority without {pivot}"] = float((base >= need).mean())
        out[f"{name} majority only with {pivot}"] = float(((base < need) & (with_pivot >= need)).mean())
    return out
