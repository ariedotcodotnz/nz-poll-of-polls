"""Kalman-filter (Rao-Blackwellised) variant of the poll-of-polls model.

Observations are the additive log-ratios of each poll's reported shares against National, which are
linear in the latent state; their noise covariance comes from the multinomial delta method scaled by
a per-pollster design effect. The weekly state is integrated out exactly with a Kalman filter, so NUTS
only samples hyperparameters, house effects and design effects. The state posterior is recovered
afterwards with a Rauch-Tung-Striebel smoother for a subset of hyperparameter draws.

The filter runs in information form: all polls (and any election result) falling in the same week are
combined into one precision-weighted update before the scan, so the sequential part does two 7x7
Cholesky factorisations per week regardless of how many polls there are. Unreported parties simply
contribute zero precision, which is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import lax
from jax.scipy.linalg import cho_factor, cho_solve

from ..prep.marshal import Dataset, alr
from .numpyro_model import offsets_from_params, sample_offsets

P0_SD = 0.01       # state sd at the anchor election
UNROLL = 8         # scan unrolling; cuts XLA per-step dispatch overhead on CPU
LOG2PI = float(np.log(2 * np.pi))


@dataclass
class KalmanData:
    K: int
    T: int
    N: int
    ystar: jnp.ndarray      # (N, d) observed log-ratios vs National (0 where unobserved)
    mask: jnp.ndarray       # (N, d)
    jbase: jnp.ndarray      # (N, d, d) precision of the observed sub-vector for design effect 1, embedded
    logdet_base: jnp.ndarray  # (N,) log det of the observed-block covariance for design effect 1
    n_obs: jnp.ndarray      # (N,) number of observed dims
    t: jnp.ndarray          # (N,) week index
    cycle_frac: jnp.ndarray  # (N,)
    elec_flag: jnp.ndarray  # (T,)
    elec_y: jnp.ndarray     # (T, d)
    campaign: jnp.ndarray   # (T,)
    error_scale: jnp.ndarray  # (K,) per-party multiplier on the election-day error scale
    theta0: jnp.ndarray     # (d,)
    pollster_idx: jnp.ndarray
    house_idx: jnp.ndarray
    cycle_idx: jnp.ndarray
    pc_idx: jnp.ndarray
    n_pollsters: int
    n_houses: int
    n_cycles: int
    target_t: int
    pm_party_idx: int
    pm_prev_share: float

    @classmethod
    def from_dataset(cls, ds: Dataset, variant: dict) -> "KalmanData":
        K, T, N = ds.K, ds.T, ds.N
        d = K - 1
        n = np.full(N, float(variant["fixed_n"])) if variant.get("fixed_n") else ds.n.astype(float)
        share = ds.y / ds.n[:, None]
        s = np.maximum(share, 0.5 / n[:, None])
        all_tracked = ds.mask[:, :-1].all(axis=1)
        mask = ds.mask[:, 1:].copy()
        mask[:, -1] = mask[:, -1] & all_tracked   # Other is only observed when every tracked party was reported
        ystar = np.where(mask, np.log(s[:, 1:]) - np.log(s[:, :1]), 0.0)
        inv = 1.0 / s[:, 1:]
        rbase = (inv[:, :, None] * np.eye(d)[None] + (1.0 / s[:, :1])[:, :, None]) / n[:, None, None]
        jbase = np.zeros((N, d, d))
        logdet_base = np.zeros(N)
        for i in range(N):
            obs = np.where(mask[i])[0]
            if len(obs) == 0:
                continue
            block = rbase[i][np.ix_(obs, obs)]
            jbase[i][np.ix_(obs, obs)] = np.linalg.inv(block)
            logdet_base[i] = np.linalg.slogdet(block)[1]
        elec_flag = np.zeros(T, dtype=bool)
        elec_y = np.zeros((T, d))
        for te, pi in zip(ds.elections_t, ds.elections_pi):
            elec_flag[int(te)] = True
            elec_y[int(te)] = alr(pi)
        return cls(
            K=K, T=T, N=N, ystar=jnp.asarray(ystar), mask=jnp.asarray(mask), jbase=jnp.asarray(jbase),
            logdet_base=jnp.asarray(logdet_base), n_obs=jnp.asarray(mask.sum(1).astype(float)),
            t=jnp.asarray(ds.t), cycle_frac=jnp.asarray(ds.cycle_frac), elec_flag=jnp.asarray(elec_flag),
            elec_y=jnp.asarray(elec_y),
            campaign=jnp.asarray(ds.campaign), theta0=jnp.asarray(ds.theta0),
            error_scale=jnp.asarray(np.ones(K) if ds.error_scale is None else ds.error_scale),
            pollster_idx=jnp.asarray(ds.pollster_idx), house_idx=jnp.asarray(ds.house_idx),
            cycle_idx=jnp.asarray(ds.cycle_idx), pc_idx=jnp.asarray(ds.pc_idx),
            n_pollsters=len(ds.pollsters), n_houses=len(ds.houses), n_cycles=ds.n_cycles,
            target_t=int(ds.target_t), pm_party_idx=int(ds.pm_party_idx), pm_prev_share=float(ds.pm_prev_share),
        )


def weekly_information(kd: KalmanData, offset, deff_i, election_obs_sd):
    """Aggregate polls and election results into per-week precision (T,d,d), information vector (T,d)
    and the constant part of the log-likelihood (T,)."""
    d = kd.K - 1
    resid = kd.ystar - jnp.where(kd.mask, offset, 0.0)                      # (N, d)
    prec = kd.jbase / deff_i[:, None, None]                                  # R_i^{-1} (embedded)
    info = jnp.einsum("nij,nj->ni", prec, resid)                             # R_i^{-1} r_i
    quad = jnp.einsum("ni,ni->n", resid, info)
    logdet = kd.logdet_base + kd.n_obs * jnp.log(deff_i)
    const = -0.5 * (quad + logdet + kd.n_obs * LOG2PI)
    Lam = jax.ops.segment_sum(prec, kd.t, num_segments=kd.T)
    eta = jax.ops.segment_sum(info, kd.t, num_segments=kd.T)
    c = jax.ops.segment_sum(const, kd.t, num_segments=kd.T)
    ev = election_obs_sd ** 2
    ef = kd.elec_flag.astype(prec.dtype)
    Lam = Lam + ef[:, None, None] * jnp.eye(d)[None] / ev
    eta = eta + ef[:, None] * kd.elec_y / ev
    c = c - 0.5 * ef * ((kd.elec_y ** 2).sum(1) / ev + d * jnp.log(ev) + d * LOG2PI)
    return Lam, eta, c


def kalman_filter(kd: KalmanData, Q, scale2, offset, deff_i, election_obs_sd):
    """Information-form filter over weeks. Returns (loglik, m_filt (T,d), P_filt (T,d,d), P_pred (T,d,d))."""
    d = kd.K - 1
    I = jnp.eye(d)
    Lam, eta, c = weekly_information(kd, offset, deff_i, election_obs_sd)

    def step(carry, inputs):
        m, P = carry
        t, Lam_t, eta_t, c_t, s2 = inputs
        P_pred = P + jnp.where(t > 0, s2, 0.0) * Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        cf_prior = cho_factor(P_pred, lower=True)
        Pinv_m = cho_solve(cf_prior, m)
        b = Pinv_m + eta_t
        Prec_post = cho_solve(cf_prior, I) + Lam_t
        Prec_post = 0.5 * (Prec_post + Prec_post.T)
        cf_post = cho_factor(Prec_post, lower=True)
        m_post = cho_solve(cf_post, b)
        P_post = cho_solve(cf_post, I)
        P_post = 0.5 * (P_post + P_post.T)
        logdet_prior = 2.0 * jnp.log(jnp.diag(cf_prior[0])).sum()
        logdet_prec_post = 2.0 * jnp.log(jnp.diag(cf_post[0])).sum()
        ll = c_t - 0.5 * (logdet_prior + logdet_prec_post + m @ Pinv_m) + 0.5 * (b @ m_post)
        return (m_post, P_post), (ll, m_post, P_post, P_pred)

    init = (kd.theta0, P0_SD ** 2 * I)
    ts = jnp.arange(kd.T)
    (_, _), (lls, m_f, P_f, P_p) = lax.scan(step, init, (ts, Lam, eta, c, scale2), unroll=UNROLL)
    return lls.sum(), m_f, P_f, P_p


def rts_smoother(m_f, P_f, P_p):
    """Rauch-Tung-Striebel smoother. Returns (m_s (T,d), P_s (T,d,d))."""
    def step(carry, inputs):
        m_next, P_next = carry
        m_t, P_t, P_pred_next = inputs
        C = jnp.linalg.solve(P_pred_next, P_t).T      # P_t P_pred^{-1}
        m_s = m_t + C @ (m_next - m_t)
        P_s = P_t + C @ (P_next - P_pred_next) @ C.T
        P_s = 0.5 * (P_s + P_s.T)
        return (m_s, P_s), (m_s, P_s)

    T = m_f.shape[0]
    init = (m_f[T - 1], P_f[T - 1])
    xs = (m_f[:T - 1][::-1], P_f[:T - 1][::-1], P_p[1:][::-1])
    (_, _), (m_rev, P_rev) = lax.scan(step, init, xs, unroll=UNROLL)
    m_s = jnp.concatenate([m_rev[::-1], m_f[T - 1:]], axis=0)
    P_s = jnp.concatenate([P_rev[::-1], P_f[T - 1:]], axis=0)
    return m_s, P_s


def dynamics(kd: KalmanData, variant: dict, priors: dict):
    Km1 = kd.K - 1
    sigma = numpyro.sample("sigma", dist.HalfNormal(priors["sigma_scale"]).expand([Km1]))
    L_corr = numpyro.sample("L_corr", dist.LKJCholesky(Km1, concentration=priors["lkj_eta"]))
    L = sigma[:, None] * L_corr
    Q = L @ L.T
    if variant.get("campaign_kappa"):
        kappa = numpyro.sample("kappa", dist.LogNormal(0.0, priors["kappa_sd"]))
        scale = 1.0 + (kappa - 1.0) * kd.campaign
    else:
        scale = jnp.ones(kd.T)
    return Q, scale ** 2


def kalman_model(kd: KalmanData, variant: dict, priors: dict, election_obs_sd: float = 0.01) -> None:
    Q, scale2 = dynamics(kd, variant, priors)
    delta, deff = sample_offsets(kd, variant, priors)
    ll, _, _, _ = kalman_filter(kd, Q, scale2, delta, deff[kd.pollster_idx], election_obs_sd)
    numpyro.factor("kalman", ll)


def smooth_draw(kd: KalmanData, variant: dict, priors: dict, draw: dict, election_obs_sd: float):
    """Smoothed state marginals for one posterior draw of the hyperparameters (pure function, vmappable)."""
    sigma = draw["sigma"]
    L = sigma[:, None] * draw["L_corr"]
    Q = L @ L.T
    if variant.get("campaign_kappa"):
        scale = 1.0 + (draw["kappa"] - 1.0) * kd.campaign
    else:
        scale = jnp.ones(kd.T)
    delta = offsets_from_params(kd, variant, draw)
    deff = draw["design_effect"] if "design_effect" in draw else jnp.full((kd.n_pollsters,), float(variant["design_effect"]))
    _, m_f, P_f, P_p = kalman_filter(kd, Q, scale ** 2, delta, deff[kd.pollster_idx], election_obs_sd)
    return rts_smoother(m_f, P_f, P_p)
