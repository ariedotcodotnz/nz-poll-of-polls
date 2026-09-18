"""Regression tests for review findings: transition density, fit persistence, fingerprints, electorates."""
import numpy as np
import jax.numpy as jnp
from scipy import stats

import pollofpolls.model  # noqa: F401
from pollofpolls.cache import fingerprint
from pollofpolls.forecast.simulate import simulate_electorates
from pollofpolls.model.fit import FitResult, _diagnostics
from pollofpolls.model.numpyro_model import anchor_transition_logprob


def test_anchor_transition_density_matches_scipy():
    rng = np.random.default_rng(0)
    L = np.linalg.cholesky(np.array([[0.04, 0.01], [0.01, 0.09]]))
    anchors = rng.normal(size=(3, 2))
    V = np.cumsum(np.r_[0.0, rng.uniform(0.5, 2.0, size=29)])
    t = (0, 12, 29)
    got = float(anchor_transition_logprob(jnp.asarray(anchors), jnp.asarray(V), t, jnp.asarray(L)))
    want = sum(stats.multivariate_normal(np.zeros(2), (V[b] - V[a]) * L @ L.T).logpdf(anchors[i + 1] - anchors[i])
               for i, (a, b) in enumerate(zip(t[:-1], t[1:])))
    assert abs(got - want) < 1e-6


def test_fit_result_roundtrip_without_arviz(tmp_path):
    rng = np.random.default_rng(1)
    hyper = {"sigma": rng.normal(1, 0.1, size=(2, 100, 3)), "kappa": rng.normal(2, 0.1, size=(2, 100))}
    diag = _diagnostics(hyper, 0, 1.0, np.full((2, 100), 7))
    assert 0.9 < diag["max_rhat"] < 1.1 and diag["min_ess_bulk"] > 50 and diag["n_params_summarised"] == 4
    pi = rng.dirichlet(np.ones(4), size=(20, 5))
    fr = FitResult("x", 2026, ["National", "Labour", "Green", "Other"], [f"w{i}" for i in range(5)], pi.mean(0),
                   np.zeros((5, 4, 5)), pi, pi[:, 2], pi[:, 4], np.zeros((20, 3)), 2, 4, diag, hyper)
    fr.save(tmp_path / "fit")
    back = FitResult.load(tmp_path / "fit")
    assert set(back.hyper) == {"sigma", "kappa"} and np.allclose(back.hyper["sigma"], hyper["sigma"])
    assert back.diagnostics == diag


def test_fingerprint_ignores_bytecode(tmp_path):
    pkg = tmp_path / "model"
    pkg.mkdir()
    (pkg / "a.py").write_text("x = 1\n")
    before = fingerprint([pkg])
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "a.cpython-313.pyc").write_bytes(b"\x00\x01")
    assert fingerprint([pkg]) == before
    (pkg / "a.py").write_text("x = 2\n")
    assert fingerprint([pkg]) != before


def test_electorate_marginals_match_config_at_reference_share():
    parties = ["National", "Te Pāti Māori", "Other"]
    cfg = [{"electorate": "Te Tai Tonga", "party": "Te Pāti Māori", "p": 0.35, "at_share": 0.03, "slope": 0.4,
            "independent_p": 0.35}]
    S = 200_000
    pi = np.tile([0.5, 0.03, 0.47], (S, 1))
    el, ind = simulate_electorates(pi, parties, cfg, np.random.default_rng(2))
    assert abs(el[:, 1].mean() - 0.35) < 0.005
    assert abs(ind.mean() - 0.35) < 0.005
    assert not np.any((el[:, 1] > 0) & (ind > 0))          # mutually exclusive
    # when the party does better, its chance rises and the independent's falls
    el2, ind2 = simulate_electorates(np.tile([0.5, 0.05, 0.45], (S, 1)), parties, cfg, np.random.default_rng(3))
    assert el2[:, 1].mean() > 0.5 and ind2.mean() < 0.35
