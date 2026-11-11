"""How the country's special party votes differ from its ordinary party votes, election by election (NumPyro).

For party p (every party but the reference, National; small parties are pooled as "Other"), category c
(overseas or domestic) and election t, the lean of the special votes is

    y_tcp = log(special votes of p / special votes of National) - log(ordinary votes of p / of National)

and the model is

    y_tcp ~ Normal(mu_cp + f_t + u_tp, sd = sqrt(sigma_g[c]^2 + sampling variance))

    mu_cp   the category's usual lean to the party
    f_t     the election's shift of every party against National (National's own good or bad year in the specials)
    u_tp    the election's shift of one party
    sigma_g how far overseas and domestic specials move apart within one election

For the forecast election f and u are new draws with Student-t tails, like the electorate model's election-wide
shift. The two models are linked: the electorate model's election-wide shift is a regression on this model's
domestic lean of Labour and of everyone else (`link_covariate`), so in a simulated count where the country's
special votes lean further from National than usual, the close seats' special votes do too.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from .model import CATS, Posterior, _run
from .national import NationalHistory

MIN_VOTES = 2000        # a party with fewer ordinary votes than this at an election is treated as not standing
DOM = CATS.index("domestic")


def lean(specials: np.ndarray, ordinary: np.ndarray) -> np.ndarray:
    """Log-ratio lean of ``specials`` (..., K) against ``ordinary`` (..., K), relative to the first party."""
    s, o = np.asarray(specials, float) + 0.5, np.asarray(ordinary, float) + 0.5
    return np.log(s[..., 1:] / s[..., :1]) - np.log(o[..., 1:] / o[..., :1])


def pooled(votes: np.ndarray, second: int, first: int = 0) -> np.ndarray:
    """Votes (..., K) as [first party, second party, everyone else]."""
    v = np.asarray(votes, float)
    rest = v.sum(-1) - v[..., first] - v[..., second]
    return np.stack([v[..., first], v[..., second], rest], axis=-1)


def pair_link(h: NationalHistory, second: int, first: int = 0) -> np.ndarray:
    """Each past election's domestic lean (T, 2) of [the second party, everyone else pooled] against the first:
    the covariate of an electorate model fitted to that pair of parties."""
    return lean(pooled(h.specials[:, DOM], second, first), pooled(h.ordinary, second, first))


@dataclass
class PartyVoteData:
    parties: list[str]              # K groups, the reference first
    years: list[int]
    y: np.ndarray                   # (T, C, K-1) lean of each category's specials
    var: np.ndarray                 # (T, C, K-1) its sampling variance
    stood: np.ndarray               # (T, K-1) whether the party stood (enough votes to measure a lean)
    second: int                     # index of the electorate model's second party (Labour)
    link: np.ndarray                # (T, 2) domestic lean of [second party, everyone else pooled]

    @property
    def link_mean(self) -> np.ndarray:
        return self.link.mean(0)

    def year_index(self, year: int) -> int:
        return self.years.index(year)


def build_party_vote_data(h: NationalHistory, second: str, years: list[int] | None = None) -> PartyVoteData:
    keep = [t for t, y in enumerate(h.years) if years is None or y in years]
    ordinary, specials = h.ordinary[keep], h.specials[keep]
    y = lean(specials, ordinary[:, None, :])
    var = 1.0 / (specials[..., 1:] + 0.5) + 1.0 / (specials[..., :1] + 0.5)
    k = h.parties.index(second)
    link = lean(pooled(specials[:, DOM], k), pooled(ordinary, k))            # pair_link, for these years
    return PartyVoteData(parties=h.parties, years=[h.years[t] for t in keep], y=y, var=var,
                         stood=ordinary[:, 1:] >= MIN_VOTES, second=k, link=link)


def party_vote_model(data: PartyVoteData, priors: dict):
    T, C, P = data.y.shape
    mu = numpyro.sample("mu", dist.Normal(0.0, priors["mean_sd"]).expand([C, P]).to_event(2))
    sigma_f = numpyro.sample("sigma_f", dist.HalfNormal(priors["common_sd_scale"]))
    f = numpyro.sample("f", dist.Normal(0.0, 1.0).expand([T]).to_event(1)) * sigma_f
    f = f - f.mean()                # past shifts are measured from their own average, which mu then carries:
    sigma_u = numpyro.sample("sigma_u", dist.HalfNormal(priors["party_sd_scale"]).expand([P]).to_event(1))
    u = numpyro.sample("u", dist.Normal(0.0, 1.0).expand([T, P]).to_event(2)) * sigma_u
    stood = jnp.asarray(data.stood)                                          # without it the two trade off freely
    u = jnp.where(stood, u - (u * stood).sum(0) / jnp.maximum(stood.sum(0), 1), u)
    sigma_g = numpyro.sample("sigma_g", dist.LogNormal(jnp.log(priors["category_sd_median"]), 1.0).expand([C]).to_event(1))
    numpyro.deterministic("f_t", f)
    numpyro.deterministic("u_t", u)
    mean = mu[None] + f[:, None, None] + u[:, None, :]
    sd = jnp.sqrt(sigma_g[None, :, None] ** 2 + jnp.asarray(data.var))
    mask = jnp.broadcast_to(jnp.asarray(data.stood)[:, None, :], (T, C, P))
    y = jnp.where(mask, jnp.asarray(data.y), 0.0)               # a party that did not stand has no lean to fit
    numpyro.sample("y", dist.Normal(mean, sd).mask(mask).to_event(3), obs=y)


KEEP = ["mu", "sigma_f", "sigma_u", "sigma_g", "f_t", "u_t"]


def fit_party_vote(data: PartyVoteData, priors: dict, mcmc_cfg: dict, progress: bool = False) -> Posterior:
    if len(data.years) < 2:
        raise ValueError("the party-vote model needs at least two past elections")
    # a few elections leave the spreads weakly identified; small steps keep NUTS free of divergences
    # and the model is small enough to run four times as long, which its slow-mixing spreads need
    mcmc_cfg = {**mcmc_cfg, "target_accept": max(0.995, float(mcmc_cfg.get("target_accept", 0.0))),
                "warmup": 2 * int(mcmc_cfg.get("warmup", 1000)), "samples": 4 * int(mcmc_cfg.get("samples", 1000))}
    return _run(party_vote_model, data, priors, mcmc_cfg, KEEP, progress)


def draw_lean(post: Posterior, data: PartyVoteData, n: int, rng: np.random.Generator, *,
              year: int | None = None, df: float | None = 4) -> np.ndarray:
    """Draws (n, C, K-1) of the special votes' lean at a new election, or at a past one (``year``).

    A past election's lean is what was counted; for a party that did not stand then it is the party's usual lean
    with that election's common shift.
    """
    pick = rng.integers(0, post.n, size=n)
    mu, sigma_u, sigma_g = post["mu"][pick], post["sigma_u"][pick], post["sigma_g"][pick]
    P = mu.shape[-1]
    g = rng.standard_normal((n, len(CATS), P)) * sigma_g[:, :, None]

    def tails(shape):
        z = rng.standard_normal(shape)
        return z * np.sqrt(df / rng.chisquare(df, size=shape)) if df else z

    u_new = tails((n, P)) * sigma_u
    if year is None:
        f = tails(n) * post["sigma_f"][pick]
        return mu + f[:, None, None] + u_new[:, None, :] + g
    t = data.year_index(year)
    modelled = mu + post["f_t"][pick, t][:, None, None] + u_new[:, None, :] + g
    return np.where(data.stood[t][None, None, :], data.y[t][None], modelled)


def against(shift: np.ndarray, ordinary: np.ndarray, second: int, first: int = 0) -> np.ndarray:
    """A shift (n, K-1) of every party's special votes against the reference, as the shift (n, 2) of
    [the second party, everyone else pooled] against the first, given the ordinary votes (K,) it applies to."""
    o = np.asarray(ordinary, float)
    full = np.concatenate([np.zeros((len(shift), 1)), shift], axis=1)       # the reference's own shift is zero
    rest = [k for k in range(len(o)) if k not in (first, second)]
    pooled_rest = np.log((o[rest][None, :] * np.exp(full[:, rest])).sum(1) / o[rest].sum())
    return np.stack([full[:, second] - full[:, first], pooled_rest - full[:, first]], axis=1)


def link_covariate(lean_draws: np.ndarray, ordinary: np.ndarray, second: int, first: int = 0) -> np.ndarray:
    """The electorate model's covariate (n, 2) for an election whose domestic lean is ``lean_draws[:, DOM]``:
    the lean of the second party and of everyone else pooled against the first (by default the reference),
    given the ordinary votes (K,) it applies to."""
    return against(lean_draws[:, DOM], ordinary, second, first)


@dataclass
class PartyVoteSim:
    final: np.ndarray               # (n, K) official party votes
    specials: np.ndarray            # (n, C, K)
    ordinary: np.ndarray            # (n, K) official ordinary votes


def ordinary_change(h: NationalHistory, floor_sd: float = 0.005) -> dict:
    """How past official counts changed the election-night party votes before any special vote: the growth in
    their number, and how each party's log-ratio to the reference moved (mean and spread across elections).
    Without an archived election-night count the change is taken as zero with the floor's uncertainty."""
    have = ~np.isnan(h.prelim).any(1)
    P = len(h.parties) - 1
    if not have.any():
        return {"growth_mean": 0.0, "growth_sd": floor_sd, "composition_mean": [0.0] * P,
                "composition_sd": [floor_sd] * P, "n": 0}
    growth = h.ordinary[have].sum(1) / h.prelim[have].sum(1) - 1
    moved = np.where(h.prelim[have][:, 1:] >= MIN_VOTES, lean(h.ordinary[have], h.prelim[have]), np.nan)
    seen = ~np.isnan(moved).all(0)
    mean = np.where(seen, np.nanmean(np.where(seen, moved, 0.0), axis=0), 0.0)
    sd = np.where(seen, np.nanstd(np.where(seen, moved, 0.0), axis=0), 0.0)
    return {"growth_mean": float(growth.mean()), "growth_sd": float(max(growth.std(), floor_sd)),
            "composition_mean": [float(x) for x in mean], "composition_sd": [float(max(x, floor_sd)) for x in sd],
            "n": int(have.sum())}


def simulate_party_votes(start: np.ndarray, lean_draws: np.ndarray, special_votes: np.ndarray,
                         overseas_votes: np.ndarray, rng: np.random.Generator, *, change: dict | None = None,
                         kappa: np.ndarray | None = None) -> PartyVoteSim:
    """The official party votes from the count ``start`` (K,).

    ``special_votes`` and ``overseas_votes`` (n,) are valid special party votes. ``change`` (from
    `ordinary_change`) moves the ordinary votes first; None leaves them as given. ``kappa`` (n, K-1) shifts the
    domestic specials (the 2026 enrolment change).
    """
    n, K = len(special_votes), len(start)
    start = np.asarray(start, float)
    if change is None:
        ordinary = np.broadcast_to(start, (n, K)).copy()
    else:
        growth = np.maximum(rng.normal(change["growth_mean"], change["growth_sd"], n), -0.01)
        moved = rng.normal(np.asarray(change["composition_mean"]), np.asarray(change["composition_sd"]), (n, K - 1))
        noise = np.concatenate([np.zeros((n, 1)), moved], axis=1)
        shares = start[None, :] * np.exp(noise)
        ordinary = np.rint(start.sum() * (1 + growth)[:, None] * shares / shares.sum(1, keepdims=True))
    eta = np.array(lean_draws, dtype=float)
    if kappa is not None:
        eta[:, DOM] += kappa
    base = np.log((ordinary[:, 1:] + 0.5) / (ordinary[:, :1] + 0.5))[:, None, :]
    logits = np.concatenate([np.zeros((n, len(CATS), 1)), base + eta], axis=-1)
    logits -= logits.max(-1, keepdims=True)
    shares = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    shares = np.where(ordinary[:, None, :] > 0, shares, 0.0)                # a party with no votes gets none
    shares /= shares.sum(-1, keepdims=True)
    overseas = np.clip(np.rint(overseas_votes), 0, np.rint(special_votes))
    by_cat = np.stack([overseas, np.rint(special_votes) - overseas], axis=1).astype(np.int64)
    specials = rng.multinomial(by_cat, shares).astype(float)
    return PartyVoteSim(final=ordinary + specials.sum(1), specials=specials, ordinary=ordinary)
