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


def balance_of_power(seats: np.ndarray, parties: list[str], total_seats: np.ndarray,
                     blocs: dict[str, list[str]], pivots: list[str]) -> list[dict]:
    """Who holds the balance of power, per simulation, for each bloc.

    For each bloc the outcomes are exclusive and sum to one: a majority alone; short alone but a majority with
    at least one pivot party on its own (``p_with`` gives each pivot's chance of being enough); a majority only
    with every pivot together; or short even with all of them.
    """
    idx = {p: i for i, p in enumerate(parties)}
    need = majority_threshold(total_seats)
    zero = np.zeros(len(seats), dtype=int)
    piv = {p: (seats[:, idx[p]] if p in idx else zero) for p in pivots}
    all_piv = sum(piv.values(), zero)
    rows = []
    for name, members in blocs.items():
        base = seats[:, [idx[p] for p in members if p in idx]].sum(1)
        alone = base >= need
        enough = {p: (~alone) & (base + s >= need) for p, s in piv.items()}
        any_one = np.zeros(len(seats), dtype=bool)
        for v in enough.values():
            any_one |= v
        needs_all = (~alone) & (~any_one) & (base + all_piv >= need)
        rows.append({
            "bloc": name, "parties": list(members), "pivots": list(pivots),
            "p_alone": float(alone.mean()),
            "p_with": {p: float(v.mean()) for p, v in enough.items()},
            "p_any_one": float(any_one.mean()),
            "p_needs_all": float(needs_all.mean()),
            "p_short": float((base + all_piv < need).mean()),
        })
    return rows
