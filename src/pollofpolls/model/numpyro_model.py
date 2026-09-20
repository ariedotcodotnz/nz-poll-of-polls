"""Bayesian state-space poll-of-polls model (NumPyro).

State
    theta_t in R^{K-1}: weekly additive log-ratio vote intention against National. It follows a random walk
    with LKJ-correlated innovations, a campaign-period volatility multiplier and, optionally, heavy-tailed
    (multivariate Student-t) weekly shocks. Official election results pin the state exactly; between two
    results the path is sampled as a Brownian bridge, which removes the stiff direction that otherwise forces
    NUTS to its maximum tree depth.

Observation offsets (one logit-scale value per party, so no party is privileged; converted to log-ratios)
    house_base      persistent pollster bias (per pollster and method segment)
    house_cycle     per-term deviation of each pollster from its persistent bias
    industry        error shared by every pollster, moving linearly within a term from its value just after
                    the previous election (identified by the first polls of the term) to its value at the
                    next election (identified by the result). For the forecast term the election-day value
                    has no result yet, so it is a fresh draw whose spread is learned from past elections.

Likelihood
    dm        Dirichlet-multinomial on the reported categories with a per-pollster design effect
    gaussian  normal on the reported shares with variance deff * p(1-p)/n plus the variance of rounding
              the published figure (unit^2 / 12)
Unreported parties are unobserved (folded into Other for that poll), never zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import gammaln, logsumexp

from ..prep.marshal import Dataset, alr


@dataclass
class ModelData:
    K: int
    T: int
    y: jnp.ndarray            # (N, K) pseudo-counts
    share: jnp.ndarray        # (N, K) published shares (Other = residual)
    mask: jnp.ndarray         # (N, K)
    n: jnp.ndarray            # (N,)
    round_var: jnp.ndarray    # (N,) variance of the rounding error of each published share
    t: jnp.ndarray
    cycle_frac: jnp.ndarray   # (N,)
    pollster_idx: jnp.ndarray
    house_idx: jnp.ndarray
    cycle_idx: jnp.ndarray
    pc_idx: jnp.ndarray
    n_pollsters: int
    n_houses: int
    n_cycles: int
    anchors_t: tuple
    anchors_theta: jnp.ndarray  # (A, K-1)
    theta0: jnp.ndarray
    campaign: jnp.ndarray
    error_scale: jnp.ndarray   # (K,) per-party multiplier on the election-day error scale
    target_t: int
    pm_party_idx: int
    pm_prev_share: float
    fund_mean: float
    fund_sd: float

    @classmethod
    def from_dataset(cls, ds: Dataset, variant: dict) -> "ModelData":
        n = np.full(ds.N, float(variant["fixed_n"])) if variant.get("fixed_n") else ds.n.astype(float)
        share = ds.y / ds.n[:, None]
        anchors = np.vstack([ds.theta0[None, :], alr(ds.elections_pi)]) if len(ds.elections_t) else ds.theta0[None, :]
        return cls(
            K=ds.K, T=ds.T, y=jnp.asarray(share * n[:, None]), share=jnp.asarray(share), mask=jnp.asarray(ds.mask),
            n=jnp.asarray(n), round_var=jnp.asarray(ds.round_unit ** 2 / 12.0), t=jnp.asarray(ds.t),
            cycle_frac=jnp.asarray(ds.cycle_frac), pollster_idx=jnp.asarray(ds.pollster_idx),
            house_idx=jnp.asarray(ds.house_idx), cycle_idx=jnp.asarray(ds.cycle_idx), pc_idx=jnp.asarray(ds.pc_idx),
            n_pollsters=len(ds.pollsters), n_houses=len(ds.houses), n_cycles=ds.n_cycles,
            anchors_t=ds.anchors_t, anchors_theta=jnp.asarray(anchors), theta0=jnp.asarray(ds.theta0),
            campaign=jnp.asarray(ds.campaign), target_t=int(ds.target_t), pm_party_idx=int(ds.pm_party_idx),
            error_scale=jnp.asarray(np.ones(ds.K) if ds.error_scale is None else ds.error_scale),
            pm_prev_share=float(ds.pm_prev_share), fund_mean=float(ds.fund_mean), fund_sd=float(ds.fund_sd),
        )


# ------------------------------------------------------------------------------------------ helpers
def dm_logpmf(y: jnp.ndarray, alpha: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Dirichlet-multinomial log pmf per row, restricted to the masked categories (closed under aggregation)."""
    alpha = jnp.where(mask, alpha, 1.0)
    y = jnp.where(mask, y, 0.0)
    n = y.sum(-1)
    a_tot = jnp.where(mask, alpha, 0.0).sum(-1)
    per_cat = jnp.where(mask, gammaln(y + alpha) - gammaln(alpha) - gammaln(y + 1.0), 0.0)
    return gammaln(n + 1.0) + gammaln(a_tot) - gammaln(n + a_tot) + per_cat.sum(-1)


def softmax_with_reference(eta: jnp.ndarray) -> jnp.ndarray:
    """Map (..., K-1) log-ratios to (..., K) proportions with the FIRST category (National) as reference."""
    full = jnp.concatenate([jnp.zeros(eta.shape[:-1] + (1,)), eta], axis=-1)
    return jnp.exp(full - logsumexp(full, axis=-1, keepdims=True))


def bridge_path(W: jnp.ndarray, V: jnp.ndarray, anchors_t: tuple, anchors_vals: jnp.ndarray) -> jnp.ndarray:
    """Pin a random walk W (T, D) with cumulative variance V (T,) to known values at the anchor weeks.

    Between consecutive anchors the result is the Brownian bridge around the linear (in V) interpolation of
    the anchor values; after the last anchor the walk runs freely. This is the exact conditional distribution
    of a Gaussian random walk given the anchors, conditional on the per-week variances. The density of the
    anchors themselves must be added separately with :func:`anchor_transition_logprob`.
    """
    pieces = []
    for i, (a, b) in enumerate(zip(anchors_t[:-1], anchors_t[1:])):
        frac = ((V[a:b] - V[a]) / (V[b] - V[a]))[:, None]
        bridge = (W[a:b] - W[a]) - frac * (W[b] - W[a])
        pieces.append(anchors_vals[i] + frac * (anchors_vals[i + 1] - anchors_vals[i]) + bridge)
    last = anchors_t[-1]
    pieces.append(anchors_vals[-1] + (W[last:] - W[last]))
    return jnp.concatenate(pieces, axis=0)


def anchor_transition_logprob(anchors_vals: jnp.ndarray, V: jnp.ndarray, anchors_t: tuple,
                              L: jnp.ndarray) -> jnp.ndarray:
    """log p(anchor_{i+1} | anchor_i) under the random walk: N(0, (V_b - V_a) L L^T) for each interval.

    Pinning the path with :func:`bridge_path` conditions on the election results; this term is the probability
    of those results given the volatility parameters, i.e. what the election-to-election swings say about
    sigma and the correlations. Omitting it would let the sampler ignore those swings.
    """
    lp = 0.0
    zero = jnp.zeros(L.shape[0])
    for i, (a, b) in enumerate(zip(anchors_t[:-1], anchors_t[1:])):
        step = dist.MultivariateNormal(zero, scale_tril=jnp.sqrt(V[b] - V[a]) * L)
        lp = lp + step.log_prob(anchors_vals[i + 1] - anchors_vals[i])
    return lp


def offsets_from_params(d, variant: dict, p: dict) -> jnp.ndarray:
    """Per-poll observation offsets on the log-ratio scale (N, K-1) from per-party logit-scale parameters."""
    full = p["house_base"][d.house_idx]
    if variant.get("house") == "cycle":
        full = full + p["house_cycle"][d.pc_idx]
    if variant.get("industry_error"):
        f = d.cycle_frac[:, None]
        full = full + (1.0 - f) * p["industry_start"][d.cycle_idx] + f * p["industry_end"][d.cycle_idx]
    return full[:, 1:] - full[:, :1]


# ---------------------------------------------------------------------------------- model pieces
def sample_dynamics(d, variant: dict, priors: dict) -> jnp.ndarray:
    """Sample the weekly state path theta (T, K-1)."""
    Km1, T = d.K - 1, d.T
    sigma = numpyro.sample("sigma", dist.HalfNormal(priors["sigma_scale"]).expand([Km1]))
    L_corr = numpyro.sample("L_corr", dist.LKJCholesky(Km1, concentration=priors["lkj_eta"]))
    L = sigma[:, None] * L_corr
    var_t = jnp.ones(T - 1)
    if variant.get("campaign_kappa"):
        kappa = numpyro.sample("kappa", dist.LogNormal(0.0, priors["kappa_sd"]))
        var_t = var_t * (1.0 + (kappa - 1.0) * d.campaign[1:]) ** 2
    if variant.get("innovations") == "student_t":
        # multivariate-t weekly shocks via a shared precision per week (unit variance on average): big news
        # weeks move every party at once, quiet weeks barely move
        nu = float(variant.get("innovation_df", 4.0))
        w = numpyro.sample("shock_precision", dist.Gamma(nu / 2.0, nu / 2.0).expand([T - 1]))
        var_t = var_t * ((nu - 2.0) / nu) / w
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([T - 1, Km1]))
    innov = (z @ L.T) * jnp.sqrt(var_t)[:, None]
    W = jnp.concatenate([jnp.zeros((1, Km1)), jnp.cumsum(innov, axis=0)], axis=0)
    V = jnp.concatenate([jnp.zeros(1), jnp.cumsum(var_t)])
    numpyro.factor("anchor_transitions", anchor_transition_logprob(d.anchors_theta, V, d.anchors_t, L))
    return bridge_path(W, V, d.anchors_t, d.anchors_theta)


def industry_raw(name: str, df: float | None, shape: tuple[int, ...]) -> jnp.ndarray:
    """Unit-variance draws for the industry-wide polling error.

    With ``df`` the draws are Student-t rescaled to unit variance, so the scale keeps its meaning while rare,
    large misses stay possible: an error of three standard deviations is about four times as likely as under a
    normal. Four completed elections cannot rule such an election out, so the shape is a prior, stated here.
    """
    if not df:
        return numpyro.sample(name, dist.Normal(0.0, 1.0).expand(list(shape)))
    df = float(df)
    return numpyro.sample(name, dist.StudentT(df, 0.0, 1.0).expand(list(shape))) * jnp.sqrt((df - 2.0) / df)


def sample_offsets(d, variant: dict, priors: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample house effects, industry error and design effects. Returns (delta (N, K-1), deff (P,))."""
    K = d.K
    p = {"house_base": numpyro.sample("house_base", dist.Normal(0.0, priors["house_base_sd"]).expand([d.n_houses, K]))}
    if variant.get("house") == "cycle":
        hc_sd = numpyro.sample("house_cycle_sd", dist.HalfNormal(priors["house_cycle_sd_scale"]))
        hc_raw = numpyro.sample("house_cycle_raw", dist.Normal(0.0, 1.0).expand([d.n_pollsters * d.n_cycles, K]))
        p["house_cycle"] = numpyro.deterministic("house_cycle", hc_raw * hc_sd)
    if variant.get("industry_error"):
        df = priors.get("industry_df")
        s_sd = numpyro.sample("industry_start_sd", dist.HalfNormal(priors["industry_sd_scale"]))
        e_sd = numpyro.sample("industry_end_sd", dist.HalfNormal(priors["industry_sd_scale"]))
        s_raw = industry_raw("industry_start_raw", df, (d.n_cycles, K))
        e_raw = industry_raw("industry_end_raw", df, (d.n_cycles, K))
        p["industry_start"] = numpyro.deterministic("industry_start", s_raw * s_sd * d.error_scale)
        p["industry_end"] = numpyro.deterministic("industry_end", e_raw * e_sd * d.error_scale)
    if variant.get("design_effect"):
        deff = jnp.full((d.n_pollsters,), float(variant["design_effect"]))
    else:
        excess = numpyro.sample("deff_excess", dist.LogNormal(jnp.log(priors["design_effect_median"] - 1.0),
                                                              priors["design_effect_sd"]).expand([d.n_pollsters]))
        deff = numpyro.deterministic("design_effect", 1.0 + excess)
    return offsets_from_params(d, variant, p), deff


def poll_model(d: ModelData, variant: dict, priors: dict, election_obs_sd: float = 0.01) -> None:
    theta = sample_dynamics(d, variant, priors)
    numpyro.deterministic("theta", theta)
    pi = numpyro.deterministic("pi", softmax_with_reference(theta))

    delta, deff = sample_offsets(d, variant, priors)
    deff_i = deff[d.pollster_idx]

    p_full = softmax_with_reference(theta[d.t] + delta)       # (N, K)
    p_obs = jnp.where(d.mask, p_full, 0.0)
    p_obs = p_obs.at[:, -1].set(1.0 - p_obs[:, :-1].sum(-1))  # unreported parties are folded into Other
    if variant.get("obs", "dm") == "dm":
        phi = jnp.maximum((d.n - deff_i) / (deff_i - 1.0), 1.0)   # DM precision giving variance inflation deff
        alpha = phi[:, None] * jnp.maximum(p_obs, 1e-6)
        numpyro.factor("polls", dm_logpmf(d.y, alpha, d.mask).sum())
    else:
        p = jnp.clip(p_obs, 1e-6, 1.0)
        var = deff_i[:, None] * p * (1.0 - p) / d.n[:, None] + d.round_var[:, None]
        ll = dist.Normal(p, jnp.sqrt(var)).log_prob(d.share)
        numpyro.factor("polls", jnp.where(d.mask, ll, 0.0).sum())

    if variant.get("fundamentals"):
        swing = pi[d.target_t, d.pm_party_idx] - d.pm_prev_share
        numpyro.sample("fundamentals", dist.StudentT(priors["fundamentals"]["df"], d.fund_mean, d.fund_sd), obs=swing)
