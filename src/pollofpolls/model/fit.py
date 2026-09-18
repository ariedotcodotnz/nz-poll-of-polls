"""Run NUTS for a model variant on a Dataset and persist a compact FitResult."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from numpyro.infer import MCMC, NUTS, init_to_median

from ..prep.marshal import Dataset, alr, alr_inverse

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
HYPER_VARS = ["sigma", "kappa", "house_base", "house_cycle_sd", "industry_start_sd", "industry_end_sd",
              "industry_start", "industry_end", "design_effect", "legacy_house"]


@dataclass
class FitResult:
    variant: str
    target_year: int
    parties: list[str]
    weeks: list[str]
    pi_mean: np.ndarray            # (T, K)
    pi_q: np.ndarray               # (T, K, len(QUANTILES))
    pi_thin: np.ndarray            # (S_thin, T, K) thinned draws
    pi_last: np.ndarray            # (S, K) draws at the last week with data
    pi_target: np.ndarray          # (S, K) draws at the election week
    theta_last: np.ndarray         # (S, K-1)
    last_data_t: int
    target_t: int
    diagnostics: dict = field(default_factory=dict)
    hyper: dict | None = None      # hyperparameter draws, name -> (chain, draw, ...) array

    def save(self, prefix: Path) -> None:
        prefix.parent.mkdir(parents=True, exist_ok=True)
        hyper_arrays = {f"hyper__{k}": np.asarray(v) for k, v in (self.hyper or {}).items()}
        np.savez_compressed(prefix.with_suffix(".npz"), pi_mean=self.pi_mean, pi_q=self.pi_q,
                            pi_thin=self.pi_thin, pi_last=self.pi_last, pi_target=self.pi_target,
                            theta_last=self.theta_last, **hyper_arrays)
        prefix.with_suffix(".json").write_text(json.dumps({
            "variant": self.variant, "target_year": self.target_year, "parties": self.parties,
            "weeks": self.weeks, "last_data_t": self.last_data_t, "target_t": self.target_t,
            "quantiles": QUANTILES, "diagnostics": self.diagnostics,
        }, ensure_ascii=False, indent=1))

    @classmethod
    def load(cls, prefix: Path) -> "FitResult":
        raw = dict(np.load(prefix.with_suffix(".npz")))
        meta = json.loads(prefix.with_suffix(".json").read_text())
        hyper_draws = {k[len("hyper__"):]: v for k, v in raw.items() if k.startswith("hyper__")}
        arrays = {k: v for k, v in raw.items() if not k.startswith("hyper__")}
        hyper = hyper_draws or None
        return cls(variant=meta["variant"], target_year=meta["target_year"], parties=meta["parties"],
                   weeks=meta["weeks"], last_data_t=meta["last_data_t"], target_t=meta["target_t"],
                   diagnostics=meta["diagnostics"], hyper=hyper, **arrays)


def posterior_var(hyper: dict | None, name: str) -> np.ndarray | None:
    """Posterior draws of ``name`` shaped (chain, draw, ...), or None if absent."""
    if not hyper or name not in hyper:
        return None
    return np.asarray(hyper[name])


def posterior_vars(hyper: dict | None) -> list[str]:
    return list(hyper or {})


def _finite(fn, values) -> float | None:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(fn(v)) if v.size else None


def _diagnostics(hyper: dict, n_divergent: int, seconds: float, num_steps: np.ndarray) -> dict:
    """Split R-hat and effective sample size of every hyperparameter (NumPyro's implementations)."""
    rhats, esss = [], []
    for x in hyper.values():
        x = np.asarray(x, dtype=float)
        if x.ndim < 2 or x.shape[1] < 4:
            continue
        rhats.append(np.ravel(split_gelman_rubin(x)))
        esss.append(np.ravel(effective_sample_size(x)))
    rhat = np.concatenate(rhats) if rhats else np.array([])
    ess = np.concatenate(esss) if esss else np.array([])
    return {
        "n_divergent": int(n_divergent),
        "seconds": round(seconds, 1),
        "mean_num_steps": float(np.mean(num_steps)),
        "max_rhat": _finite(np.max, rhat),
        "min_ess_bulk": _finite(np.min, ess),
        "n_params_summarised": int(rhat.size),
    }


def _run_nuts(model, args: tuple, mcmc_cfg: dict, seed: int, progress: bool):
    kernel = NUTS(model, max_tree_depth=int(mcmc_cfg.get("max_tree_depth", 10)),
                  target_accept_prob=float(mcmc_cfg.get("target_accept", 0.85)),
                  init_strategy=init_to_median(num_samples=20), dense_mass=False)
    chains = int(mcmc_cfg.get("chains", 4))
    mcmc = MCMC(kernel, num_warmup=int(mcmc_cfg["warmup"]), num_samples=int(mcmc_cfg["samples"]),
                num_chains=chains, chain_method="parallel" if jax.local_device_count() >= chains else "sequential",
                progress_bar=progress)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(int(seed)), *args, extra_fields=("diverging", "num_steps"))
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    extra = mcmc.get_extra_fields(group_by_chain=True)
    seconds = time.time() - t0
    return samples, int(np.asarray(extra["diverging"]).sum()), np.asarray(extra["num_steps"]), seconds


def _result(ds: Dataset, variant: dict, pi: np.ndarray, theta_last: np.ndarray, pi_last: np.ndarray,
            pi_target: np.ndarray, samples: dict, n_div: int, num_steps, seconds: float,
            thin_to: int = 500) -> FitResult:
    hyper = {k: v for k, v in samples.items() if k in HYPER_VARS}
    step = max(1, pi.shape[0] // thin_to)
    return FitResult(
        variant=variant["name"], target_year=ds.target_year, parties=ds.parties,
        weeks=[w.isoformat() for w in ds.weeks], pi_mean=pi.mean(0),
        pi_q=np.quantile(pi, QUANTILES, axis=0).transpose(1, 2, 0), pi_thin=pi[::step], pi_last=pi_last,
        pi_target=pi_target, theta_last=theta_last, last_data_t=ds.last_data_t, target_t=ds.target_t,
        diagnostics=_diagnostics(hyper, n_div, seconds, num_steps), hyper=hyper,
    )


def _seed(mcmc_cfg: dict, seed: int | None) -> int:
    return int(seed if seed is not None else mcmc_cfg.get("seed", 0))


def fit_dataset(ds: Dataset, variant: dict, priors: dict, mcmc_cfg: dict, election_obs_sd: float = 0.01,
                seed: int | None = None, progress: bool = True) -> FitResult:
    """Full-state NUTS for the Dirichlet-multinomial and Gaussian variants."""
    from .numpyro_model import ModelData, poll_model

    data = ModelData.from_dataset(ds, variant)
    samples, n_div, steps, secs = _run_nuts(poll_model, (data, variant, priors, election_obs_sd), mcmc_cfg,
                                            _seed(mcmc_cfg, seed), progress)
    C_, S_, T, K = samples["pi"].shape
    pi = samples["pi"].reshape(C_ * S_, T, K)
    theta = samples["theta"].reshape(C_ * S_, T, K - 1)
    return _result(ds, variant, pi, theta[:, ds.last_data_t], pi[:, ds.last_data_t], pi[:, ds.target_t],
                   samples, n_div, steps, secs)


def fit_legacy(ds: Dataset, variant: dict, mcmc_cfg: dict, seed: int | None = None,
               progress: bool = True) -> FitResult:
    """The 2023 NZ Herald specification (see legacy.py)."""
    from .legacy import LegacyData, legacy_model

    ld = LegacyData.from_dataset(ds)
    samples, n_div, steps, secs = _run_nuts(legacy_model, (ld,), mcmc_cfg, _seed(mcmc_cfg, seed), progress)
    C_, S_, T, K = samples["pi"].shape
    pi = samples["pi"].reshape(C_ * S_, T, K)
    return _result(ds, variant, pi, alr(np.clip(pi[:, ds.last_data_t], 1e-4, None)), pi[:, ds.last_data_t],
                   pi[:, ds.target_t], samples, n_div, steps, secs)


def fit_kalman(ds: Dataset, variant: dict, priors: dict, mcmc_cfg: dict, election_obs_sd: float = 0.01,
               seed: int | None = None, n_state_draws: int = 400, draws_per_state: int = 10,
               progress: bool = True) -> FitResult:
    """NUTS over hyperparameters with the weekly state integrated out; states recovered by RTS smoothing."""
    from .kalman import KalmanData, kalman_model, smooth_draw

    kd = KalmanData.from_dataset(ds, variant)
    seed = _seed(mcmc_cfg, seed)
    samples, n_div, steps, secs = _run_nuts(kalman_model, (kd, variant, priors, election_obs_sd), mcmc_cfg,
                                            seed, progress)
    flat = {k: v.reshape((-1,) + v.shape[2:]) for k, v in samples.items()}
    pick = np.linspace(0, flat["sigma"].shape[0] - 1, min(n_state_draws, flat["sigma"].shape[0])).astype(int)
    keys = ["sigma", "L_corr", "kappa", "house_base", "house_cycle", "industry_start", "industry_end", "design_effect"]
    sub = {k: jnp.asarray(flat[k][pick]) for k in keys if k in flat}
    m_s, P_s = jax.jit(jax.vmap(lambda dr: smooth_draw(kd, variant, priors, dr, election_obs_sd)))(sub)
    m_s, P_s = np.asarray(m_s), np.asarray(P_s)
    S, T, d = m_s.shape
    rng = np.random.default_rng(seed + 1)
    chol = np.linalg.cholesky(P_s + 1e-9 * np.eye(d))

    def draw_theta(t_idx: int, reps: int) -> np.ndarray:
        eps = rng.standard_normal((S, reps, d))
        return (m_s[:, t_idx, None, :] + np.einsum("sij,srj->sri", chol[:, t_idx], eps)).reshape(-1, d)

    pi = alr_inverse(m_s + np.einsum("stij,stj->sti", chol, rng.standard_normal((S, T, d))))
    theta_last = draw_theta(ds.last_data_t, draws_per_state)
    return _result(ds, variant, pi, theta_last, alr_inverse(theta_last),
                   alr_inverse(draw_theta(ds.target_t, draws_per_state)), samples, n_div, steps, secs)


def fit(ds: Dataset, variant: dict, priors: dict, mcmc_cfg: dict, election_obs_sd: float = 0.01,
        seed: int | None = None, progress: bool = True) -> FitResult:
    """Dispatch on the variant's observation model."""
    obs = variant.get("obs", "dm")
    if obs == "legacy":
        return fit_legacy(ds, variant, mcmc_cfg, seed, progress)
    if obs == "kalman":
        return fit_kalman(ds, variant, priors, mcmc_cfg, election_obs_sd, seed, progress=progress)
    return fit_dataset(ds, variant, priors, mcmc_cfg, election_obs_sd, seed, progress)
