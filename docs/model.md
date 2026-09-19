# Model

The model follows the state-space approach of Jackman (2005), *Pooling the polls over an election campaign*,
as adapted for New Zealand by Peter Ellis (2017, 2020) and the NZ Herald (2023). Latent vote intention moves
week by week; polls are noisy, biased measurements of it; official results measure it exactly. The main
changes from the 2023 model are listed at the end.

## Notation

| Symbol | Meaning |
|---|---|
| $k = 0, \dots, K-1$ | tracked parties; $k=0$ is National, $k = K-1$ is "Other" |
| $t = 0, \dots, T-1$ | weeks (starting Sunday); $t=0$ is the week of the anchor election (2011) |
| $\pi_t$ | vote shares in week $t$, a point on the simplex |
| $\theta_t \in \mathbb{R}^{K-1}$ | additive log-ratios, $\theta_{t,k} = \log(\pi_{t,k} / \pi_{t,0})$ |
| $i$ | a poll, with week $t_i$, pollster $j_i$, house segment $h_i$, term $c_i$ and sample size $n_i$ |

$\pi_t = \operatorname{softmax}([0, \theta_t])$, so shares are always positive and sum to one.

## Latent dynamics

The state is a random walk with correlated innovations:

$$\theta_t = \theta_{t-1} + \eta_t, \qquad \eta_t \sim \mathcal{N}(0,\ v_t\, Q), \qquad Q = \operatorname{diag}(\sigma)\, \Omega\, \operatorname{diag}(\sigma).$$

* $\sigma_k \sim \text{HalfNormal}(0.06)$ is the weekly volatility of each log-ratio.
* $\Omega$ is a correlation matrix with an LKJ(2) prior, so parties can move together.
* $v_t = \kappa^2$ in the eight weeks up to and including each election week, and $1$ otherwise, with
  $\kappa \sim \text{LogNormal}(0, 0.3)$. Support can move faster during campaigns; past terms estimate how much.
* In the `heavy` variant, $v_t$ is also multiplied by $\frac{\nu - 2}{\nu}\, w_t^{-1}$ with
  $w_t \sim \text{Gamma}(\nu/2, \nu/2)$ and $\nu = 4$. This makes the weekly shocks multivariate Student-t
  with the same average variance: most weeks are quiet, and a news week can move every party at once.

### Pinning the state to election results

At the anchor election and every completed election $e$ in the window, $\theta_{t_e} = \operatorname{alr}(\text{result}_e)$
exactly. Parties absent from a result, such as TOP in 2011, are floored at 0.2% before taking logs.

Sampling a random walk that must pass through fixed points is stiff: the 2023 Stan model needed a maximum tree
depth of 20. Here the path is sampled directly from its conditional distribution. With non-centred
innovations $W_t = \sum_{s \le t} \sqrt{v_s}\, L z_s$, where $L L^\top = Q$ and $z_s \sim \mathcal{N}(0, I)$, and
cumulative variance $V_t = \sum_{s \le t} v_s$, the path between consecutive anchors $a < b$ is the
Brownian bridge

$$\theta_t = \theta_a + \frac{V_t - V_a}{V_b - V_a}(\theta_b - \theta_a) + (W_t - W_a) - \frac{V_t - V_a}{V_b - V_a}(W_b - W_a), \qquad a \le t \le b.$$

After the last anchor the walk runs freely: $\theta_t = \theta_{\text{last}} + W_t - W_{\text{last}}$.

Conditioning on the anchors also requires their probability, which the bridge alone omits. The likelihood
therefore includes the transition density of every interval,

$$\sum_{\text{intervals } (a, b)} \log \mathcal{N}\big(\theta_b - \theta_a;\ 0,\ (V_b - V_a)\, Q\big),$$

so large swings between elections, such as 2017 to 2020 to 2023, inform $\sigma$, $\Omega$ and $\kappa$.

## Observation model

### Offsets

Each poll measures the state with an offset $\delta_i \in \mathbb{R}^{K}$ on the logit scale, with one value per
party so that no party is privileged:

$$\delta_i = \beta_{h_i} + \gamma_{j_i, c_i} + (1 - f_i)\, s_{c_i} + f_i\, e_{c_i}.$$

| Term | Meaning | Prior |
|---|---|---|
| $\beta_h$ | persistent house effect of a pollster, or of a pollster's method segment (Reid Research after its 2017 change) | $\mathcal{N}(0, 0.12^2)$ per party |
| $\gamma_{j,c}$ | how pollster $j$ deviates from its persistent effect in term $c$ | $\mathcal{N}(0, \tau_\gamma^2)$, $\tau_\gamma \sim \text{HalfNormal}(0.05)$ |
| $s_c$, $e_c$ | industry-wide error shared by all pollsters at the start and at the end of term $c$ | $\mathcal{N}(0, \tau_s^2)$, $\mathcal{N}(0, \tau_e^2)$, $\tau_s, \tau_e \sim \text{HalfNormal}(0.15)$ |

$f_i \in [0, 1]$ is how far through its term the poll was taken, from 0 at the previous election to 1 at the
next. The first polls of a term identify $s_c$ against the previous result, and each result identifies $e_c$.
For the term being forecast, $e_c$ has no result yet, so its posterior is its prior: a fresh election-day polling
error with spread $\tau_e$ learned from past elections.

The offsets enter on the log-ratio scale as $\Delta_{i,k} = \delta_{i,k} - \delta_{i,0}$. The expected poll
shares are

$$p_i = \operatorname{softmax}([0,\ \theta_{t_i} + \Delta_i]).$$

A poll that did not report a tracked party is treated as having folded it into "Other": that party is
unobserved, and $p_{i,\text{Other}}$ absorbs its expected share. It is never treated as 0%.

### Likelihood

Each pollster has a design effect $d_j = 1 + \epsilon_j$ with $\epsilon_j \sim \text{LogNormal}(0, 0.4)$, a prior
median of 2. This is the ratio of the pollster's true variance to simple random sampling, learned from the data
instead of the 2023 model's fixed $\sqrt{2}$ inflation of the standard error.

* **Dirichlet-multinomial** (`base`): the reported counts $y_i = n_i \times \text{share}_i$ follow
  $\text{DirMult}(n_i, \phi_i p_i)$ over the reported categories, with $\phi_i = (n_i - d_j)/(d_j - 1)$ so that the
  variance is $d_j$ times the multinomial variance.
* **Gaussian** (`gauss`, `heavy`, `fund`): each reported share, including Other, follows
  $\mathcal{N}\big(p_{i,k},\ d_j\, p_{i,k}(1 - p_{i,k})/n_i + u_i^2/12\big)$. Here $u_i$ is the rounding unit
  of the published figures: 1 point when every figure is a whole number, 0.5 when they are halves, otherwise
  0.1. Rounding matters for small parties, where "1%" can mean anything from 0.5% to 1.5%.

The real sample size is used when Wikipedia reports one; otherwise the pollster's default from
`config/pollsters.yml`.

### Fundamentals prior (`fund` only)

Ellis (2020) added a prior on the swing of the Prime Minister's party:

$$\pi_{T, \text{PM}} - \pi_{\text{prev}, \text{PM}} \sim \text{Student-}t_4(m, s),$$

where $m$ is the mean swing at elections since 1996 before the target, and $s = \sqrt{\text{sd}^2 + \text{sd}^2 / n}$.
For 2026 this gives $m = -2.0$ points and $s = 9.6$ points from ten elections. The prior is estimated without
look-ahead in backtests. It is not in the published ensemble because it made the 2020 and 2023 backtests worse
(see [Evaluation](evaluation.md)).

## Forecasts

* **Election day** is the posterior of $\pi$ at the election week. For the term being forecast this includes
  the random-walk movement still to come, the campaign volatility, and the fresh election-day industry error.
* **If held now** is the posterior of $\pi$ at the last week with polls.

The published forecast mixes variants by their stacking weights; see [Evaluation](evaluation.md).

## Seats

For each of 4,000 simulated elections:

1. **Electorates.** Each electorate in `config/electorates.yml` is won by the named party, an independent, or
   anyone else. The three outcomes are mutually exclusive and match the configured probabilities when the
   party's vote equals `at_share`. The party's chance moves on the logit scale by `slope` per percentage point
   of simulated party vote, and the independent keeps the same share of the remaining probability.
2. **Eligibility.** A party qualifies for list seats with at least 5% of the party vote or one electorate.
   "Other" never qualifies.
3. **Allocation.** 120 seats, less one for each electorate won by an independent, are divided among qualifying
   parties by the Sainte-Laguë method (divisors 1, 3, 5, ...).
4. **Overhang.** A party keeps every electorate it wins. If that exceeds its proportional entitlement, the House
   grows.
5. **Majorities.** A coalition has a majority when its seats reach half the simulated House size, rounded down,
   plus one.

The allocation reproduces the official 2011, 2014, 2017, 2020 and 2023 outcomes; see `tests/test_seats.py`.

## Inference

All variants are written in [NumPyro](https://num.pyro.ai) and sampled with NUTS on CPU through JAX: 4 chains of
1,000 warmup and 1,000 draws for the live forecast, and 2 chains of 500 and 500 for backtests. Each fit records
the number of divergences, the largest split R-hat and the smallest effective sample size over the
hyperparameters, from NumPyro's diagnostics. These appear in `output/summary.json` and at the foot of the report.

### Kalman variant (`kal`, experimental)

With a Gaussian likelihood on log-ratios, the model given its hyperparameters is linear and Gaussian, so the
weekly state can be integrated out exactly. Each poll is observed as the log-ratios of its published shares
against National, with covariance from the delta method:
$R_i = \tfrac{d_j}{n_i}\big(\operatorname{diag}(1/s_{i,k}) + \tfrac{1}{s_{i,0}} \mathbf{1}\mathbf{1}^\top\big)$
over the reported parties. An information-form Kalman filter combines all polls in a week in one update,
NUTS samples only the hyperparameters, and a Rauch-Tung-Striebel smoother recovers the states afterwards.
`tests/test_kalman.py` checks the likelihood and smoothed moments against brute-force Gaussian integration.

This is a different likelihood from `gauss`, not a faster route to the same posterior. It is not used in the
ensemble because each gradient of the sequential filter is slower on CPU than the full-state model. It is
the exact counterpart of the ensemble Kalman filters used for very high-dimensional states: with a
seven-dimensional state, the exact filter costs little and has no Monte Carlo error.

### 2023 model replica (`legacy`)

For backtests, `src/pollofpolls/model/legacy.py` re-implements the 2023 NZ Herald model
(`legacy-r/stan/model2023.stan`):

* a random walk on raw proportions with $\sigma \sim \mathcal{N}(0.002, 0.001)$ truncated at zero and LKJ(1);
* constant house effects $\mathcal{N}(0, 0.03)$ per pollster and party, with the Reid Research method change;
* $n = 1000$ for every poll, a fixed standard error per pollster and party from its average share, multiplied
  by $\sqrt{2}$;
* unreported parties recorded as 0%;
* election results as exact observations, including their transition density.

Two deliberate differences keep the comparison about the model: the standard errors are matched to the right
party columns, which the R code misaligned; and the replica sees the same pollsters and anchor election as the
new model.

## Changes from the 2023 model

| 2023 model | This model |
|---|---|
| Random walk on raw proportions, which can go negative | Log-ratio random walk; shares stay on the simplex |
| $n = 1000$ for every poll, standard error $\times \sqrt{2}$ | Real sample sizes, learned design effect per pollster, rounding error |
| Constant house effects plus one method-change dummy | Persistent plus per-term house effects, method segments |
| No industry-wide polling error | Start- and end-of-term industry error; fresh election-day error for 2026 |
| Constant volatility | Campaign multiplier; optional heavy-tailed weekly shocks |
| Unreported parties recorded as 0% | Unreported parties treated as unobserved |
| Stan, maximum tree depth 20 | NumPyro, bridge sampling, maximum tree depth 10 |
| Four hard-coded pollster slots | Any number of pollsters from `config/pollsters.yml` |
| No evaluation | Rolling-origin backtests, proper scoring rules, stacking |
