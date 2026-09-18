import numpy as np
import jax.numpy as jnp

from pollofpolls.eval.scoring import (crps, crps_components, crps_stacking_weights, energy_score, mixture_crps,
                                      mixture_draws)
from pollofpolls.model.numpyro_model import bridge_path


def test_crps_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    brute = np.abs(x - 0.3).mean() - 0.5 * np.abs(x[:, None] - x[None, :]).mean()
    assert abs(crps(x, 0.3) - brute) < 1e-12


def test_crps_of_normal_close_to_closed_form():
    from scipy import stats
    x = np.random.default_rng(1).normal(size=20000)
    y = 0.7
    closed = y * (2 * stats.norm.cdf(y) - 1) + 2 * stats.norm.pdf(y) - 1 / np.sqrt(np.pi)
    assert abs(crps(x, y) - closed) < 0.01


def test_energy_score_reduces_to_crps_in_1d():
    x = np.random.default_rng(2).normal(size=(800, 1))
    assert abs(energy_score(x, np.array([0.2]), max_pairs=800) - crps(x[:, 0], 0.2)) < 1e-9


def test_mixture_crps_and_stacking():
    rng = np.random.default_rng(3)
    parties_keep = [0, 1]
    good = rng.normal([0.30, 0.20], 0.01, size=(2000, 2))
    bad = rng.normal([0.40, 0.10], 0.01, size=(2000, 2))
    outcome = np.array([0.30, 0.20])
    A, B = crps_components([good, bad], outcome, parties_keep, m=800)
    w = np.array([0.5, 0.5])
    mix = mixture_draws([good, bad], w, 4000)
    direct = sum(crps(mix[:, k], outcome[k]) for k in parties_keep)
    assert abs(mixture_crps(w, A, B) - direct) < 0.003
    weights = crps_stacking_weights([(A, B)])
    assert weights[0] > 0.95


def test_bridge_pins_anchors_and_runs_free_after():
    rng = np.random.default_rng(4)
    W = jnp.asarray(np.cumsum(rng.normal(size=(30, 2)), axis=0))
    W = W - W[0]
    V = jnp.arange(30.0)
    anchors = jnp.asarray([[0.0, 0.0], [1.0, -1.0], [2.0, 0.5]])
    path = np.asarray(bridge_path(W, V, (0, 10, 20), anchors))
    assert path.shape == (30, 2)
    assert np.allclose(path[[0, 10, 20]], np.asarray(anchors), atol=1e-6)
    assert np.allclose(path[21:] - path[20], np.asarray(W[21:] - W[20]), atol=1e-6)
