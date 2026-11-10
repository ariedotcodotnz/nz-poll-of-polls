"""Hierarchical Bayesian models of the official count (NumPyro).

Votes are grouped per seat as [National's candidate, Labour's candidate, everyone else] (``pair`` in
config/specials.yml), and shares are written as log-ratios against National's candidate:
x = [log(Labour / National), log(Other / National)].

Special-vote model, fitted on past official results (one row per seat and election, i; category c is overseas
or domestic, domestic being the "before" and "on polling day" rows together):

    composition   specials of category c in seat-election i split as softmax([0, eta_ic]), with
                  eta_ic = x_i + mu_c + alpha_t + gamma_tc + beta_e + eps_ic
                  x_i      the seat's own ordinary vote that election (so the model learns how specials differ
                           from the ordinary vote, not how they vote)
                  mu_c     mean shift of the category
                  alpha_t  election-wide shift, correlated across the two log-ratios (LKJ). This is the
                           counterpart of the poll model's industry-wide error: for the forecast election it is a
                           fresh draw, with Student-t tails, from the spread of past elections. When the
                           country's party-vote lean is given (``national``, from the party-vote model),
                           alpha_t = slope * national_t + a residual with that structure, so the close seats
                           move with the national special vote and learn its spread from more elections
                  gamma_tc how far overseas and domestic specials move apart in one election
                  beta_e   the seat's persistent difference, eps_ic everything else
                  The counts are multinomial; eps absorbs any overdispersion.
    volume        log(specials / ordinary votes) = A_t + b_e + noise, and the overseas share of specials
                  logit = A'_t + b'_e + noise. For the forecast election A_t moves from the last election by
                  the national change the Electoral Commission reports.

Official-count model, fitted on the seats whose election-night count is archived: the official count rechecks
the ordinary votes and can add ordinary votes missing on election night. The relative change in ordinary votes
v is a mixture of a small symmetric change and, with probability pi, a log-normal addition. The added votes split
like the preliminary count shifted by xi ~ N(0, sigma_xi); every candidate's count also moves by recheck noise
with sd s_r * sqrt(votes).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from numpyro.infer import MCMC, NUTS, init_to_median

from .. import (
    model as _xla_flags,  # noqa: F401  sets the CPU device count before JAX starts, for parallel chains
)
from .data import ElectorateCount, electorate_key

CATEGORY_GROUPS = {"overseas": ("overseas",), "domestic": ("before", "on")}
CATS = list(CATEGORY_GROUPS)
GROUPS = 3                           # [pair[0], pair[1], other]


def log_ratio(v: np.ndarray) -> np.ndarray:
    """[log(v1 / v0), log(v2 / v0)] for grouped votes ``v`` (..., 3), with half a vote added to every cell."""
    v = np.asarray(v, dtype=float) + 0.5
    return np.log(v[..., 1:] / v[..., :1])


def softmax0(eta: np.ndarray) -> np.ndarray:
    """Shares from log-ratios against the first group: softmax([0, eta])."""
    full = np.concatenate([np.zeros(eta.shape[:-1] + (1,)), eta], axis=-1)
    full -= full.max(-1, keepdims=True)
    e = np.exp(full)
    return e / e.sum(-1, keepdims=True)


# ------------------------------------------------------------------------------------------------ data
@dataclass
class SpecialsData:
    """Past official results as arrays, one row per seat and election."""
    pair: list[str]
    electorates: list[str]          # electorate keys
    labels: list[str]               # names as published
    years: list[int]
    e_idx: np.ndarray               # (I,)
    t_idx: np.ndarray               # (I,)
    base: np.ndarray                # (I, 2) log-ratios of the ordinary vote
    counts: np.ndarray              # (I, C, 3) special votes by category and group
    ordinary: np.ndarray            # (I, 3) official ordinary votes by group
    prelim: np.ndarray              # (I, 3) election-night count, NaN where not archived
    national: np.ndarray | None = None   # (T, 2) the country's party-vote lean each election, centred

    @property
    def n_rows(self) -> int:
        return len(self.e_idx)

    @property
    def special_total(self) -> np.ndarray:
        return self.counts.sum((1, 2))

    @property
    def log_volume(self) -> np.ndarray:
        return np.log(self.special_total / self.ordinary.sum(1))

    @property
    def overseas_logit(self) -> np.ndarray:
        ov = self.counts[:, 0].sum(1)
        return np.log((ov + 0.5) / (self.special_total - ov + 0.5))

    def electorate_index(self, name: str) -> int | None:
        k = electorate_key(name)
        return self.electorates.index(k) if k in self.electorates else None

    def year_index(self, year: int) -> int:
        return self.years.index(year)


def build_specials_data(counts: list[ElectorateCount], pair: list[str],
                        national: dict[int, np.ndarray] | None = None) -> SpecialsData:
    """Arrays for the special-vote model from official results. Seats without a candidate from both parties of
    the pair are left out. ``national`` gives each election's national party-vote lean (2,), already centred;
    elections it does not cover are dropped from the link by leaving it out altogether."""
    rows = []
    for c in sorted(counts, key=lambda c: (c.year, c.key)):
        if not c.official:
            continue
        ordinary = c.grouped(c.ordinary, pair)
        if ordinary is None:
            continue
        by_cat = [sum(c.grouped(c.specials[k], pair) for k in CATEGORY_GROUPS[cat]) for cat in CATS]
        prelim = c.grouped(c.preliminary, pair) if c.preliminary is not None else np.full(GROUPS, np.nan)
        rows.append((c, ordinary, np.stack(by_cat), prelim))
    if not rows:
        raise ValueError(f"no past official results with candidates from both {pair[0]} and {pair[1]}")
    keys = sorted({c.key for c, *_ in rows})
    labels = {c.key: c.electorate for c, *_ in rows}
    years = sorted({c.year for c, *_ in rows})
    ordinary = np.array([r[1] for r in rows])
    return SpecialsData(
        pair=list(pair), electorates=keys, labels=[labels[k] for k in keys], years=years,
        e_idx=np.array([keys.index(c.key) for c, *_ in rows]), t_idx=np.array([years.index(c.year) for c, *_ in rows]),
        base=log_ratio(ordinary), counts=np.array([r[2] for r in rows]), ordinary=ordinary,
        prelim=np.array([r[3] for r in rows]),
        national=(np.array([national[y] for y in years], dtype=float)
                  if national is not None and all(y in national for y in years) else None))


@dataclass
class OfficialCountData:
    """Seats whose election-night count is archived: preliminary and official ordinary votes by group."""
    prelim: np.ndarray              # (J, 3)
    ordinary: np.ndarray            # (J, 3)
    labels: list[str]

    @property
    def change(self) -> np.ndarray:
        """Relative change in the ordinary votes: official / preliminary - 1."""
        return self.ordinary.sum(1) / self.prelim.sum(1) - 1

    @property
    def shares(self) -> np.ndarray:
        return softmax0(log_ratio(self.prelim))

    @property
    def residual(self) -> np.ndarray:
        """Change in each group's votes beyond a change that splits exactly like the preliminary count."""
        return self.ordinary - self.prelim - (self.change * self.prelim.sum(1))[:, None] * self.shares

    @property
    def jacobian(self) -> np.ndarray:
        """d shares / d log-ratios at the preliminary count, (J, 3, 2)."""
        s = self.shares
        eye = np.eye(GROUPS)[None, :, 1:]                                    # d(eta_k) enters group k + 1
        return s[:, :, None] * (eye - s[:, None, 1:])


def build_official_count_data(data: SpecialsData) -> OfficialCountData:
    have = ~np.isnan(data.prelim).any(1)
    labels = [f"{data.labels[e]} {data.years[t]}" for e, t in zip(data.e_idx[have], data.t_idx[have])]
    return OfficialCountData(prelim=data.prelim[have], ordinary=data.ordinary[have], labels=labels)


# ------------------------------------------------------------------------------------------------ models
def specials_model(data: SpecialsData, priors: dict):
    R, C = data.counts.shape[:2]
    T, E = len(data.years), len(data.electorates)
    e, t = jnp.asarray(data.e_idx), jnp.asarray(data.t_idx)

    # composition
    mu = numpyro.sample("mu", dist.Normal(0.0, priors["category_mean_sd"]).expand([C, 2]).to_event(2))
    linked = data.national is not None
    link = priors.get("link", {})
    year_scale = link.get("residual_sd_scale", priors["year_sd_scale"]) if linked else priors["year_sd_scale"]
    sigma_year = numpyro.sample("sigma_year", dist.HalfNormal(year_scale).expand([2]).to_event(1))
    L_year = numpyro.sample("L_year", dist.LKJCholesky(2, priors["lkj_eta"]))
    z_year = numpyro.sample("z_year", dist.Normal(0.0, 1.0).expand([T, 2]).to_event(2))
    alpha = (z_year @ L_year.T) * sigma_year
    if linked:
        slope = numpyro.sample("link_slope", dist.Normal(link.get("slope_mean", 1.0), link.get("slope_sd", 0.5))
                               .expand([2]).to_event(1))
        alpha = alpha + slope * jnp.asarray(data.national)
    alpha = numpyro.deterministic("alpha", alpha)
    sigma_yc = numpyro.sample("sigma_yc", dist.HalfNormal(priors["year_category_sd_scale"]).expand([2]).to_event(1))
    z_yc = numpyro.sample("z_yc", dist.Normal(0.0, 1.0).expand([T, C, 2]).to_event(3))
    gamma = numpyro.deterministic("gamma", z_yc * sigma_yc)
    sigma_e = numpyro.sample("sigma_e", dist.HalfNormal(priors["electorate_sd_scale"]).expand([2]).to_event(1))
    z_e = numpyro.sample("z_e", dist.Normal(0.0, 1.0).expand([E, 2]).to_event(2))
    beta = numpyro.deterministic("beta", z_e * sigma_e)
    sigma_eps = numpyro.sample("sigma_eps", dist.HalfNormal(priors["noise_sd_scale"]).expand([2]).to_event(1))
    z_eps = numpyro.sample("z_eps", dist.Normal(0.0, 1.0).expand([R, C, 2]).to_event(3))
    eta = (jnp.asarray(data.base)[:, None, :] + mu[None] + alpha[t][:, None, :] + gamma[t]
           + beta[e][:, None, :] + z_eps * sigma_eps)
    logits = jnp.concatenate([jnp.zeros((R, C, 1)), eta], axis=-1)
    counts = jnp.asarray(data.counts)
    numpyro.sample("specials", dist.Multinomial(total_count=counts.sum(-1), logits=logits).to_event(2), obs=counts)

    # volume: log specials per ordinary vote, and the overseas share of specials
    vp = priors["volume"]
    vol_year = numpyro.sample("vol_year", dist.Normal(-1.5, 1.0).expand([T]).to_event(1))
    sigma_vb = numpyro.sample("sigma_vol_b", dist.HalfNormal(vp["electorate_sd_scale"]))
    z_vol = numpyro.sample("z_vol", dist.Normal(0, 1).expand([E]).to_event(1))
    vol_e = numpyro.deterministic("vol_e", z_vol * sigma_vb)
    sigma_vc = numpyro.sample("sigma_vol_c", dist.HalfNormal(vp["noise_sd_scale"]))
    numpyro.sample("log_volume", dist.Normal(vol_year[t] + vol_e[e], sigma_vc).to_event(1),
                   obs=jnp.asarray(data.log_volume))
    ov_year = numpyro.sample("ov_year", dist.Normal(-2.0, 1.0).expand([T]).to_event(1))
    sigma_ob = numpyro.sample("sigma_ov_b", dist.HalfNormal(vp["electorate_sd_scale"]))
    z_ov = numpyro.sample("z_ov", dist.Normal(0, 1).expand([E]).to_event(1))
    ov_e = numpyro.deterministic("ov_e", z_ov * sigma_ob)
    sigma_oc = numpyro.sample("sigma_ov_c", dist.HalfNormal(vp["noise_sd_scale"] * 2))
    numpyro.sample("overseas_logit", dist.Normal(ov_year[t] + ov_e[e], sigma_oc).to_event(1),
                   obs=jnp.asarray(data.overseas_logit))


def official_count_model(data: OfficialCountData, priors: dict):
    p = priors["official_count"]
    P = jnp.asarray(data.prelim)
    v = jnp.asarray(data.change)
    a, b = p["jump_prob"]
    pi = numpyro.sample("jump_prob", dist.Beta(a, b))
    s_small = numpyro.sample("small_sd", dist.HalfNormal(p["small_sd_scale"]))
    m_jump = numpyro.sample("jump_log_median", dist.Normal(np.log(p["jump_median"]), 1.0))
    s_jump = numpyro.sample("jump_log_sd", dist.HalfNormal(p["jump_log_sd_scale"]))
    small = jnp.log1p(-pi) + dist.Normal(0.0, s_small).log_prob(v)
    jump = jnp.log(pi) + jnp.where(v > 0, dist.LogNormal(m_jump, s_jump).log_prob(jnp.clip(v, 1e-9)), -jnp.inf)
    numpyro.factor("change", jax.scipy.special.logsumexp(jnp.stack([small, jump]), axis=0).sum())

    # How the change splits between the two contenders. The added votes split as softmax(x + xi) with
    # xi ~ N(0, diag(sigma_xi^2)); linearising around the preliminary shares s, the contenders' residual
    # r = change - v * n * s is normal with covariance (v n)^2 G diag(sigma_xi^2) G' + s_r^2 diag(votes + 1), G the
    # softmax Jacobian. Integrating xi out leaves no per-seat latent variables, which only two or three seats
    # could identify. ("Other" is left out: given v, the three residuals sum to zero.)
    sigma_xi = numpyro.sample("sigma_xi", dist.HalfNormal(p["composition_sd_scale"]).expand([2]).to_event(1))
    s_r = numpyro.sample("recheck_sd", dist.HalfNormal(p["recheck_sd_scale"]))
    G = jnp.asarray(data.jacobian[:, :2, :])                                  # (J, 2, 2)
    scale = (v * P.sum(1))[:, None, None]
    cov = (scale ** 2) * (G * sigma_xi ** 2) @ jnp.swapaxes(G, 1, 2) + (s_r ** 2) * jax.vmap(jnp.diag)(P[:, :2] + 1.0)
    numpyro.sample("ordinary_change", dist.MultivariateNormal(jnp.zeros(2), covariance_matrix=cov).to_event(1),
                   obs=jnp.asarray(data.residual[:, :2]))


# ------------------------------------------------------------------------------------------------- fitting
KEEP = {"specials": ["mu", "sigma_year", "L_year", "alpha", "link_slope", "sigma_yc", "gamma", "sigma_e", "beta", "sigma_eps",
                     "vol_year", "sigma_vol_b", "vol_e", "sigma_vol_c", "ov_year", "sigma_ov_b", "ov_e",
                     "sigma_ov_c"],
        "official_count": ["jump_prob", "small_sd", "jump_log_median", "jump_log_sd", "sigma_xi", "recheck_sd"]}


@dataclass
class Posterior:
    """Posterior draws, each (S, ...) with chains stacked, and sampler diagnostics."""
    draws: dict[str, np.ndarray]
    diagnostics: dict

    def __getitem__(self, name: str) -> np.ndarray:
        return self.draws[name]

    @property
    def n(self) -> int:
        return len(next(iter(self.draws.values())))


def _diagnostics(samples: dict, extra: dict, names: list[str], seconds: float) -> dict:
    rhat, ess = [], []
    for k in names:
        x = np.asarray(samples[k])
        if x.ndim < 2 or x.shape[1] < 4:
            continue
        flat = x.reshape(x.shape[0], x.shape[1], -1)
        for j in range(flat.shape[-1]):
            col = flat[..., j]
            if np.ptp(col) == 0:                  # fixed elements, such as the 1 in a Cholesky factor
                continue
            if x.shape[0] > 1:
                rhat.append(float(split_gelman_rubin(col)))
            ess.append(float(effective_sample_size(col)))
    diverging = np.asarray(extra.get("diverging", np.zeros(1)))
    return {"max_rhat": max(rhat) if rhat else None, "min_ess_bulk": min(ess) if ess else None,
            "n_divergent": int(diverging.sum()), "seconds": round(seconds, 1)}


def _run(model, data, priors: dict, mcmc_cfg: dict, keep: list[str], progress: bool) -> Posterior:
    t0 = time.time()
    kernel = NUTS(model, target_accept_prob=mcmc_cfg.get("target_accept", 0.95),
                  max_tree_depth=mcmc_cfg.get("max_tree_depth", 10), init_strategy=init_to_median(num_samples=15))
    chains = int(mcmc_cfg.get("chains", 4))
    mcmc = MCMC(kernel, num_warmup=int(mcmc_cfg.get("warmup", 1000)), num_samples=int(mcmc_cfg.get("samples", 1000)),
                num_chains=chains, chain_method="parallel" if jax.local_device_count() >= chains else "sequential",
                progress_bar=progress)
    mcmc.run(jax.random.PRNGKey(int(mcmc_cfg.get("seed", 0))), data, priors, extra_fields=("diverging",))
    by_chain = mcmc.get_samples(group_by_chain=True)
    diag = _diagnostics(by_chain, mcmc.get_extra_fields(group_by_chain=True), list(by_chain), time.time() - t0)
    draws = {k: np.asarray(v).reshape((-1,) + np.asarray(v).shape[2:]) for k, v in by_chain.items() if k in keep}
    return Posterior(draws, diag)


def fit_specials(data: SpecialsData, priors: dict, mcmc_cfg: dict, progress: bool = False) -> Posterior:
    return _run(specials_model, data, priors, mcmc_cfg, KEEP["specials"], progress)


def fit_official_count(data: OfficialCountData, priors: dict, mcmc_cfg: dict, progress: bool = False) -> Posterior:
    if len(data.prelim) == 0:
        raise ValueError("no archived election-night counts to learn the official count's change from")
    # a handful of seats leave the mixture weakly identified; smaller steps keep NUTS free of divergences
    mcmc_cfg = {**mcmc_cfg, "target_accept": max(0.99, float(mcmc_cfg.get("target_accept", 0.0)))}
    return _run(official_count_model, data, priors, mcmc_cfg, KEEP["official_count"], progress)
