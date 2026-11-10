"""The House the official count would elect: seats by party, which side has a majority, and by what route.

Each simulated official count has the country's final party votes and the winner of every electorate. Seats
follow by the Sainte-Laguë formula with overhang (`forecast.seats`). The House is then one of three things:

    government   the governing parties have more than half the seats
    opposition   the parties named as the alternative have more than half
    neither      anything else: with two sides covering every seat, an exact tie

How a count gets from the election-night House to its own is written as a path: the seats each governing party
lost, split for a party in overhang into the seats its party vote entitles it to and the electorates it holds,
plus any change in the other side's overhang. Those are the only changes that move the margin between the sides:

    margin = government seats - everyone else's
    change in margin = 2 x (list seats a governing party loses) + (entitlement an overhang party loses)
                       + (electorates an overhang party loses) - (growth of the other side's overhang)

because a party in overhang keeps its electorates whatever its party vote (losing entitlement only hands the
seat to someone else), and loses an electorate without anyone gaining a seat.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from ..forecast.seats import entitlement_matrix

OUTCOMES = ("government", "neither", "opposition")


@dataclass
class House:
    parties: list[str]              # K groups, "Other" last
    seats: np.ndarray               # (n, K)
    entitlement: np.ndarray         # (n, K) seats by party vote, before overhang
    electorates: np.ndarray         # (n, K)

    @property
    def size(self) -> np.ndarray:
        return self.seats.sum(1)

    @property
    def overhang(self) -> np.ndarray:
        return np.maximum(self.electorates - self.entitlement, 0)


def elect(votes: np.ndarray, electorates: np.ndarray, parties: list[str], nseats: int = 120,
          threshold: float = 0.05) -> House:
    """Seats from party votes (n, K) and electorates won (n, K). The pooled "Other" group never qualifies."""
    votes = np.atleast_2d(np.asarray(votes, float))
    electorates = np.atleast_2d(np.asarray(electorates)).astype(int)
    shares = votes / votes.sum(1, keepdims=True)
    if "Other" in parties:
        shares[:, parties.index("Other")] = 0.0
    entitlement = entitlement_matrix(shares, electorates, nseats, threshold)
    return House(parties, np.maximum(entitlement, electorates), entitlement, electorates)


def outcome(house: House, government: list[str], opposition: list[str]) -> np.ndarray:
    """Index into OUTCOMES for each simulation."""
    gov = house.seats[:, [house.parties.index(p) for p in government]].sum(1)
    opp = house.seats[:, [house.parties.index(p) for p in opposition]].sum(1)
    return np.where(2 * gov > house.size, 0, np.where(2 * opp > house.size, 2, 1))


def margin(house: House, government: list[str]) -> np.ndarray:
    """Government seats minus everyone else's."""
    gov = house.seats[:, [house.parties.index(p) for p in government]].sum(1)
    return 2 * gov - house.size


@dataclass
class Component:
    key: str                        # "list:ACT", "entitlement:National", "electorates:National", "overhang:..."
    party: str
    kind: str                       # list | entitlement | electorates | overhang
    weight: int                     # seats of margin per unit
    lost: np.ndarray                # (n,) units lost by the government side (negative: gained)


def components(house: House, base: House, government: list[str]) -> list[Component]:
    """The changes from ``base`` (one row: the election-night House) that move the margin, per simulation."""
    out = []
    for k, p in enumerate(house.parties):
        in_overhang = base.electorates[0, k] > base.entitlement[0, k]
        if p in government:
            if in_overhang:
                out.append(Component(f"entitlement:{p}", p, "entitlement", 1,
                                     base.entitlement[0, k] - house.entitlement[:, k]))
                out.append(Component(f"electorates:{p}", p, "electorates", 1,
                                     base.electorates[0, k] - house.electorates[:, k]))
            elif base.seats[0, k] > 0:
                out.append(Component(f"list:{p}", p, "list", 2, base.seats[0, k] - house.seats[:, k]))
        elif in_overhang:
            out.append(Component(f"overhang:{p}", p, "overhang", 1, house.overhang[:, k] - base.overhang[0, k]))
    return out


def describe(c: Component, units: int) -> str:
    """A path step in words: "ACT loses 1 list seat"."""
    n, s = abs(units), "" if abs(units) == 1 else "s"
    if c.kind == "list":
        return f"{c.party} {'loses' if units > 0 else 'gains'} {n} list seat{s}"
    if c.kind == "entitlement":
        return f"{c.party}'s party-vote entitlement {'falls' if units > 0 else 'rises'} by {n}"
    if c.kind == "electorates":
        return f"{c.party} {'loses' if units > 0 else 'gains'} {n} electorate{s} net"
    return f"{c.party}'s overhang {'grows' if units > 0 else 'shrinks'} by {n}"


def paths(house: House, base: House, government: list[str], opposition: list[str], top: int = 12) -> dict:
    """The routes simulations took from the election-night House, and the minimal routes to a one-seat majority
    for the other side."""
    comps = components(house, base, government)
    n = len(house.seats)
    lost = np.stack([c.lost for c in comps], axis=1) if comps else np.zeros((n, 0), dtype=int)
    result = outcome(house, government, opposition)
    m0 = int(margin(base, government)[0])
    m = margin(house, government)
    keys, inverse, counts = np.unique(lost, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(-counts)

    def row(i: int) -> dict:
        sel = inverse == i
        steps = [describe(c, int(u)) for c, u in zip(comps, keys[i]) if u]
        return {"steps": steps or ["No change"], "units": {c.key: int(u) for c, u in zip(comps, keys[i]) if u},
                "p": float(counts[i] / n), "margin": float(np.median(m[sel])),
                "outcome": OUTCOMES[int(np.bincount(result[sel], minlength=3).argmax())]}

    # the minimal routes: combinations of losses whose weights add up exactly to the margin plus one
    need = m0 + 1
    weights = [c.weight for c in comps if c.kind != "overhang"]
    idx = [j for j, c in enumerate(comps) if c.kind != "overhang"]
    minimal = []
    if need > 0 and idx:
        for combo in itertools.product(*[range(need // w + 1) for w in weights]):
            if sum(u * w for u, w in zip(combo, weights)) != need:
                continue
            target = np.zeros(len(comps), dtype=int)
            target[idx] = combo
            hit = (lost == target).all(1)
            minimal.append({"steps": [describe(comps[j], u) for j, u in zip(idx, combo) if u],
                            "units": {comps[j].key: int(u) for j, u in zip(idx, combo) if u},
                            "p": float(hit.mean())})
        minimal.sort(key=lambda r: -r["p"])
    return {"margin_election_night": m0, "seats_of_margin_needed": need,
            "components": [{"key": c.key, "party": c.party, "kind": c.kind, "weight": c.weight} for c in comps],
            "top": [row(int(i)) for i in order[:top]], "n_routes": len(counts),
            "minimal": minimal, "p_minimal": float(sum(r["p"] for r in minimal)),
            "p_margin_exactly_needed": float((m0 - m == need).mean()),
            "p_margin_more_than_needed": float((m0 - m > need).mean())}


def distribution(x: np.ndarray) -> dict[str, float]:
    values, counts = np.unique(np.asarray(x).astype(int), return_counts=True)
    return {str(int(v)): float(c / len(x)) for v, c in zip(values, counts)}


def summarise(house: House, base: House, government: list[str], opposition: list[str]) -> dict:
    """Outcome probabilities, the margin's distribution and every party's seats."""
    result = outcome(house, government, opposition)
    seats = []
    for k, p in enumerate(house.parties):
        if p == "Other" and not house.seats[:, k].any():
            continue
        b = int(base.seats[0, k])
        seats.append({"party": p, "election_night": b, "election_night_electorates": int(base.electorates[0, k]),
                      "election_night_entitlement": int(base.entitlement[0, k]),
                      "mean": float(house.seats[:, k].mean()), "seats": distribution(house.seats[:, k]),
                      "p_fewer": float((house.seats[:, k] < b).mean()), "p_more": float((house.seats[:, k] > b).mean()),
                      "electorates": distribution(house.electorates[:, k]),
                      "entitlement": distribution(house.entitlement[:, k])})
    m = margin(house, government)
    return {"outcome": {name: float((result == i).mean()) for i, name in enumerate(OUTCOMES)},
            "margin": distribution(m), "margin_election_night": int(margin(base, government)[0]),
            "size": distribution(house.size), "size_election_night": int(base.size[0]),
            "seats": seats}
