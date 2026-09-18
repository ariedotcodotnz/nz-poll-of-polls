"""End-to-end smoke tests on synthetic data: every inference path recovers a known path."""
from datetime import date, timedelta

import numpy as np
import pytest

import pollofpolls.model  # noqa: F401
from pollofpolls.model.fit import fit
from pollofpolls.prep.marshal import Dataset, alr, alr_inverse

PRIORS = {"sigma_scale": 0.06, "lkj_eta": 2.0, "house_base_sd": 0.12, "house_cycle_sd_scale": 0.05,
          "industry_sd_scale": 0.15, "kappa_sd": 0.3, "design_effect_median": 2.0, "design_effect_sd": 0.4,
          "fundamentals": {"df": 4}}
MCMC = {"chains": 1, "warmup": 80, "samples": 80, "max_tree_depth": 8, "seed": 3}


def synthetic(seed=0, T=60, n_polls=45):
    rng = np.random.default_rng(seed)
    parties = ["National", "Labour", "Green", "Other"]
    K = len(parties)
    pi0 = np.array([0.45, 0.35, 0.12, 0.08])
    theta0 = alr(pi0[None, :])[0]
    theta = theta0 + np.cumsum(rng.normal(0, 0.04, size=(T, K - 1)), axis=0)
    theta[0] = theta0
    pi = alr_inverse(theta)
    t_elec = T // 2
    house = np.array([[0.1, 0.0, -0.1], [-0.1, 0.05, 0.0]])
    t = np.sort(rng.integers(1, T, size=n_polls))
    pollster = rng.integers(0, 2, size=n_polls)
    n = np.full(n_polls, 1000.0)
    y = np.zeros((n_polls, K)); mask = np.ones((n_polls, K), dtype=bool)
    for i in range(n_polls):
        y[i] = rng.multinomial(int(n[i]), alr_inverse(theta[t[i]] + house[pollster[i]]))
    cycle = (t > t_elec).astype(int)
    frac = np.where(cycle == 0, t / t_elec, (t - t_elec) / (T - 1 - t_elec))
    return Dataset(
        parties=parties, weeks=[date(2020, 1, 5) + timedelta(weeks=i) for i in range(T)], y=y, mask=mask, n=n, t=t,
        pollster_idx=pollster, house_idx=pollster, cycle_idx=cycle, pc_idx=pollster * 2 + cycle, cycle_frac=frac,
        round_unit=np.full(n_polls, 0.001), pollsters=["A", "B"], houses=["A", "B"], cycle_years=[2021, 2023],
        elections_t=np.array([t_elec]), elections_pi=pi[t_elec][None, :], election_years=[2021], theta0=theta0,
        pi0=pi0, campaign=np.zeros(T), target_year=2023, target_t=T - 1, pm_party_idx=0,
        pm_prev_share=float(pi[t_elec, 0]), fund_mean=-0.01, fund_sd=0.05, fund_n=5, cutoff=date(2021, 3, 1),
        last_data_t=int(t.max()), poll_ids=[str(i) for i in range(n_polls)],
        mid_dates=[date(2020, 1, 5) + timedelta(weeks=int(k)) for k in t],
    ), pi


@pytest.mark.parametrize("variant", [
    {"name": "base", "obs": "dm", "house": "cycle", "industry_error": True, "campaign_kappa": True},
    {"name": "gauss", "obs": "gaussian", "house": "cycle", "industry_error": True, "campaign_kappa": True},
    {"name": "heavy", "obs": "gaussian", "house": "cycle", "industry_error": True, "innovations": "student_t",
     "innovation_df": 4},
    {"name": "fund", "obs": "gaussian", "house": "constant", "fundamentals": True},
    {"name": "kal", "obs": "kalman", "house": "cycle", "industry_error": True, "campaign_kappa": True},
    {"name": "legacy", "obs": "legacy"},
])
def test_recovers_path(variant):
    ds, truth = synthetic()
    fr = fit(ds, variant, PRIORS, MCMC, progress=False)
    assert fr.pi_mean.shape == (ds.T, ds.K)
    obs = slice(0, ds.last_data_t + 1)
    tol = 0.05 if variant["obs"] == "legacy" else 0.03   # legacy has a tight fixed-scale random walk prior
    assert np.abs(fr.pi_mean[obs] - truth[obs]).mean() < tol
    # the election result pins the state
    assert np.allclose(fr.pi_mean[ds.elections_t[0]], truth[ds.elections_t[0]], atol=0.01)
    if variant["obs"] != "legacy":
        assert np.allclose(fr.pi_target.sum(1), 1.0, atol=1e-5)
