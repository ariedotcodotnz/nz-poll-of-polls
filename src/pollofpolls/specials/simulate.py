"""Posterior predictive simulation of the official count of each seat.

Every simulation takes one posterior draw of the models' parameters and one draw of the election-wide special-vote
shift, shared by every seat (so the seats move together, as they do in a real count), then for each seat:

1. the official count's change to the ordinary votes (if the forecast starts from an election-night count);
2. how many special votes, and how many of them are overseas votes;
3. how each category of special vote splits between National's candidate, Labour's and everyone else;

and adds them to the starting count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import CATS, Posterior, SpecialsData, log_ratio, softmax0


@dataclass
class Target:
    """A seat to forecast, starting from ``start`` = [National, Labour, Other] votes."""
    name: str
    key: str                       # electorate key, to find the seat's own history
    start: np.ndarray              # (3,)
    official_change: bool = True   # False when ``start`` is already the official ordinary vote


@dataclass
class VolumeShift:
    """How the forecast election's special-vote volume differs from a reference election in the data."""
    ref_year: int
    log_change: np.ndarray         # (n,) national change in log specials per ordinary vote
    overseas_logit_change: np.ndarray   # (n,) national change in the logit of the overseas share of specials
    extra_sd: float = 0.0          # added per-seat sd of the log volume (the 2026 enrolment change)


@dataclass
class Simulation:
    final: np.ndarray              # (n, N, 3) official result
    specials: np.ndarray           # (n, N, C, 3) special votes by category
    change: np.ndarray             # (n, N, 3) official count's change to the ordinary votes
    shift: np.ndarray              # (n, N, C, 2) specials' log-ratio shift from the starting count
    election: np.ndarray           # (n, C, 2) election-wide shift of each category (mean + election + category)


def _resample(post: Posterior, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    pick = rng.integers(0, post.n, size=n)
    return {k: v[pick] for k, v in post.draws.items()}


def election_shift(p: dict[str, np.ndarray], year_index: int | None, df: float | None,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """The election-wide shift (n, 2) and the category deviations (n, C, 2): a fresh election, or a past one's."""
    n = len(p["sigma_year"])
    if year_index is not None:
        return p["alpha"][:, year_index], p["gamma"][:, year_index]
    z = rng.standard_normal((n, 2))
    if df:
        z *= np.sqrt(df / rng.chisquare(df, size=n))[:, None]       # multivariate Student-t
    alpha = np.einsum("nij,nj->ni", p["L_year"], z) * p["sigma_year"]
    gamma = rng.standard_normal((n, len(CATS), 2)) * p["sigma_yc"][:, None, :]
    return alpha, gamma


def official_change(start: np.ndarray, p: dict[str, np.ndarray], rng: np.random.Generator,
                    max_addition: float = 0.2) -> np.ndarray:
    """Draws (n, 3) of the official count's change to the ordinary votes of a seat whose count is ``start``."""
    n = len(p["jump_prob"])
    jump = rng.random(n) < p["jump_prob"]
    added = np.minimum(np.exp(p["jump_log_median"] + p["jump_log_sd"] * rng.standard_normal(n)), max_addition)
    v = np.where(jump, added, p["small_sd"] * rng.standard_normal(n))
    xi = rng.standard_normal((n, 2)) * p["sigma_xi"]
    shares = softmax0(log_ratio(start)[None, :] + xi)
    noise = rng.standard_normal((n, 3)) * p["recheck_sd"][:, None] * np.sqrt(start + 1.0)
    change = np.rint(v[:, None] * start.sum() * shares + noise)
    return np.maximum(change, -start)


def simulate(post: Posterior, data: SpecialsData, targets: list[Target], volume: VolumeShift, *,
             official: Posterior | None = None, year: int | None = None, kappa: np.ndarray | None = None,
             df: float | None = 4, max_addition: float = 0.2, n: int = 20000, seed: int = 0) -> Simulation:
    """Simulate the official count of ``targets``.

    ``year`` replays a past election's special-vote shift instead of drawing a new one. ``kappa`` (n, 2) shifts
    the domestic specials (the 2026 enrolment change). ``official`` is the official-count posterior; without it,
    or for a target with ``official_change`` False, the ordinary votes are not changed.
    """
    rng = np.random.default_rng(seed)
    p = _resample(post, n, rng)
    po = _resample(official, n, rng) if official is not None else None
    alpha, gamma = election_shift(p, None if year is None else data.year_index(year), df, rng)
    ref = data.year_index(volume.ref_year)
    dom = CATS.index("domestic")
    N, C = len(targets), len(CATS)
    final, specials = np.zeros((n, N, 3)), np.zeros((n, N, C, 3))
    change, shift = np.zeros((n, N, 3)), np.zeros((n, N, C, 2))
    for j, t in enumerate(targets):
        e = data.electorate_index(t.key)
        # 1. the ordinary votes in the official count
        if po is not None and t.official_change:
            change[:, j] = official_change(t.start, po, rng, max_addition)
        ordinary = t.start[None, :] + change[:, j]
        # 2. how many specials, and how many from overseas
        if e is not None:
            vol_e, ov_e, beta = p["vol_e"][:, e], p["ov_e"][:, e], p["beta"][:, e]
        else:                      # a seat with no history: its persistent differences are new draws
            vol_e = rng.standard_normal(n) * p["sigma_vol_b"]
            ov_e = rng.standard_normal(n) * p["sigma_ov_b"]
            beta = rng.standard_normal((n, 2)) * p["sigma_e"]
        log_vol = (p["vol_year"][:, ref] + vol_e + volume.log_change + rng.standard_normal(n) * p["sigma_vol_c"]
                   + rng.standard_normal(n) * volume.extra_sd)
        total = rng.poisson(ordinary.sum(1) * np.exp(log_vol))
        ov_logit = p["ov_year"][:, ref] + ov_e + volume.overseas_logit_change + rng.standard_normal(n) * p["sigma_ov_c"]
        overseas = rng.binomial(total, 1.0 / (1.0 + np.exp(-ov_logit)))
        by_cat = np.stack([overseas, total - overseas], axis=1)
        # 3. how they split
        eps = rng.standard_normal((n, C, 2)) * p["sigma_eps"][:, None, :]
        eta = p["mu"] + alpha[:, None, :] + gamma + beta[:, None, :] + eps
        if kappa is not None:
            eta[:, dom] += kappa
        shift[:, j] = eta
        shares = softmax0(log_ratio(t.start)[None, None, :] + eta)
        specials[:, j] = rng.multinomial(by_cat, shares)
        final[:, j] = ordinary + specials[:, j].sum(1)
    election = p["mu"] + alpha[:, None, :] + gamma
    if kappa is not None:
        election[:, dom] += kappa
    return Simulation(final=final, specials=specials, change=change, shift=shift, election=election)
