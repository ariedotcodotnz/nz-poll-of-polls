"""The information-form Kalman likelihood must equal brute-force Gaussian integration on a tiny problem."""
import jax.numpy as jnp
import numpy as np
from scipy import stats

import pollofpolls.model  # noqa: F401  (sets XLA flags)
from pollofpolls.model.kalman import KalmanData, kalman_filter, rts_smoother

D, T = 2, 4
POLLS = [  # (week, observed dims, y, covariance)
    (1, [True, True], np.array([0.3, -0.2]), np.array([[0.05, 0.01], [0.01, 0.06]])),
    (1, [True, False], np.array([0.1, 0.0]), np.array([[0.04, 0.0], [0.0, 1.0]])),
    (3, [True, True], np.array([0.5, -0.1]), np.array([[0.05, 0.02], [0.02, 0.07]])),
]
ELEC_WEEK, ELEC_Y, EPS = 2, np.array([0.2, -0.3]), 0.05
THETA0 = np.array([0.1, -0.1])
Q = np.array([[0.02, 0.005], [0.005, 0.03]])


def _kd():
    N = len(POLLS)
    mask = np.array([p[1] for p in POLLS])
    ystar = np.array([p[2] for p in POLLS])
    jbase = np.zeros((N, D, D)); logdet = np.zeros(N)
    for i, (_, m, _, R) in enumerate(POLLS):
        obs = np.where(m)[0]
        jbase[i][np.ix_(obs, obs)] = np.linalg.inv(R[np.ix_(obs, obs)])
        logdet[i] = np.linalg.slogdet(R[np.ix_(obs, obs)])[1]
    elec_flag = np.zeros(T, dtype=bool); elec_flag[ELEC_WEEK] = True
    elec_y = np.zeros((T, D)); elec_y[ELEC_WEEK] = ELEC_Y
    return KalmanData(K=D + 1, T=T, N=N, ystar=jnp.asarray(ystar), mask=jnp.asarray(mask), jbase=jnp.asarray(jbase),
                      logdet_base=jnp.asarray(logdet), n_obs=jnp.asarray(mask.sum(1).astype(float)),
                      t=jnp.asarray([p[0] for p in POLLS]), cycle_frac=jnp.zeros(N), elec_flag=jnp.asarray(elec_flag),
                      elec_y=jnp.asarray(elec_y),
                      campaign=jnp.zeros(T), theta0=jnp.asarray(THETA0), error_scale=jnp.ones(D + 1),
                      pollster_idx=jnp.zeros(N, dtype=int), house_idx=jnp.zeros(N, dtype=int),
                      cycle_idx=jnp.zeros(N, dtype=int), pc_idx=jnp.zeros(N, dtype=int),
                      n_pollsters=1, n_houses=1, n_cycles=1, target_t=T - 1, pm_party_idx=0, pm_prev_share=0.3)


def _brute_force():
    """Joint Gaussian over the stacked state (T*D): marginal likelihood and posterior in closed form."""
    P0 = 0.01 ** 2 * np.eye(D)
    dim = T * D
    mean = np.tile(THETA0, T)
    cov = np.zeros((dim, dim))
    for t1 in range(T):
        for t2 in range(T):
            cov[t1 * D:(t1 + 1) * D, t2 * D:(t2 + 1) * D] = P0 + min(t1, t2) * Q
    H_rows, y, R_blocks = [], [], []
    for (t, m, yy, R) in POLLS:
        obs = np.where(m)[0]
        for o in obs:
            row = np.zeros(dim); row[t * D + o] = 1; H_rows.append(row)
        y.extend(yy[obs]); R_blocks.append(R[np.ix_(obs, obs)])
    for o in range(D):
        row = np.zeros(dim); row[ELEC_WEEK * D + o] = 1; H_rows.append(row)
    y.extend(ELEC_Y); R_blocks.append(EPS ** 2 * np.eye(D))
    H = np.array(H_rows); y = np.array(y)
    R = np.zeros((len(y), len(y))); k = 0
    for b in R_blocks:
        R[k:k + b.shape[0], k:k + b.shape[0]] = b; k += b.shape[0]
    S = H @ cov @ H.T + R
    ll = stats.multivariate_normal(H @ mean, S).logpdf(y)
    K = cov @ H.T @ np.linalg.inv(S)
    post_mean = mean + K @ (y - H @ mean)
    post_cov = cov - K @ H @ cov
    return ll, post_mean.reshape(T, D), post_cov


def test_kalman_matches_brute_force():
    kd = _kd()
    ll, m_f, P_f, P_p = kalman_filter(kd, jnp.asarray(Q), jnp.ones(T), jnp.zeros((kd.N, D)), jnp.ones(kd.N), EPS)
    ll_bf, mean_bf, cov_bf = _brute_force()
    assert abs(float(ll) - ll_bf) < 1e-3, (float(ll), ll_bf)
    m_s, P_s = rts_smoother(m_f, P_f, P_p)
    assert np.allclose(np.asarray(m_s), mean_bf, atol=1e-4)
    for t in range(T):
        assert np.allclose(np.asarray(P_s)[t], cov_bf[t * D:(t + 1) * D, t * D:(t + 1) * D], atol=1e-4)
