"""Re-implementation of the NZ Herald 2023 Poll of Polls model (legacy-r/stan/model2023.stan).

Used only as the baseline in backtests, so improvements are measured against what was actually run in 2023:

* latent vote shares on the raw proportion scale (they need not sum to one and can go negative);
* weekly random walk, innovations epsilon ~ MVN(0, Omega) scaled by sigma, sigma ~ N(0.002, 0.001) truncated
  at zero, Omega ~ LKJ(1);
* election results as (near-)exact observations of the state (here: exact, via the same bridge sampler plus
  the transition density of the results, which is what the Stan observations contributed);
* constant house effect per pollster and party, d ~ N(0, 0.03), with the Reid Research 2017 method change as
  a separate segment (the Stan model's reid_impact term);
* every poll assumed to have n = 1000; the standard error of each pollster-party pair is fixed at
  sqrt(pbar (1 - pbar) / 1000) from the pollster's average share, then multiplied by sqrt(2);
* tracked parties a poll did not report are recorded as 0% (values_fill = 0 in the R pipeline).

Deliberate differences: the per-party standard errors are matched to the right columns (the R code sorted
them alphabetically with "Other" in the middle, misaligning them with the data), and the same pollsters and
anchor election are used as for the new model, so the comparison isolates the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from ..prep.marshal import Dataset
from .numpyro_model import anchor_transition_logprob, bridge_path

INFLATOR = float(np.sqrt(2.0))


@dataclass
class LegacyData:
    K: int
    T: int
    share: jnp.ndarray       # (N, K) zero-filled published shares
    se: jnp.ndarray          # (N, K) fixed standard errors (before the inflator)
    t: jnp.ndarray
    house_idx: jnp.ndarray
    n_houses: int
    anchors_t: tuple
    anchors_mu: jnp.ndarray  # (A, K)

    @classmethod
    def from_dataset(cls, ds: Dataset) -> "LegacyData":
        share = ds.y / ds.n[:, None]                       # zero where a tracked party was not reported
        se = np.zeros_like(share)
        for j in range(len(ds.pollsters)):
            rows = ds.pollster_idx == j
            pbar = np.clip(share[rows].mean(axis=0), 1e-4, 1 - 1e-4)
            se[rows] = np.sqrt(pbar * (1 - pbar) / 1000.0)
        anchors = np.vstack([ds.pi0[None, :], ds.elections_pi]) if len(ds.elections_t) else ds.pi0[None, :]
        return cls(K=ds.K, T=ds.T, share=jnp.asarray(share), se=jnp.asarray(se), t=jnp.asarray(ds.t),
                   house_idx=jnp.asarray(ds.house_idx), n_houses=len(ds.houses), anchors_t=ds.anchors_t,
                   anchors_mu=jnp.asarray(anchors))


def legacy_model(ld: LegacyData) -> None:
    K, T = ld.K, ld.T
    sigma = numpyro.sample("sigma", dist.TruncatedNormal(0.002, 0.001, low=0.0).expand([K]))
    L_corr = numpyro.sample("L_corr", dist.LKJCholesky(K, concentration=1.0))
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([T - 1, K]))
    innov = (z @ L_corr.T) * sigma
    W = jnp.concatenate([jnp.zeros((1, K)), jnp.cumsum(innov, axis=0)], axis=0)
    V = jnp.arange(T, dtype=W.dtype)
    # the Stan model observed each result with sd 0.00145; pinning it exactly is equivalent once the
    # transition density of the results is kept
    numpyro.factor("anchor_transitions", anchor_transition_logprob(ld.anchors_mu, V, ld.anchors_t,
                                                                   sigma[:, None] * L_corr))
    mu = bridge_path(W, V, ld.anchors_t, ld.anchors_mu)
    numpyro.deterministic("pi", mu)
    house = numpyro.sample("legacy_house", dist.Normal(0.0, 0.03).expand([ld.n_houses, K]))
    mean = mu[ld.t] + house[ld.house_idx]
    numpyro.factor("polls", dist.Normal(mean, ld.se * INFLATOR).log_prob(ld.share).sum())
