# NZ Poll of Polls 2026

Bayesian aggregation of published opinion polls, with a seat and coalition forecast, for the New Zealand
general election on **Saturday 7 November 2026**.

This is a Python rewrite and upgrade of the NZ Herald's 2023 *Poll of Polls* (R, Stan and `{targets}`),
which was itself a port of [Peter Ellis's](https://freerangestats.info/elections/state-space.html) 2017/2020
model in the tradition of Jackman (2005), *Pooling the polls over an election campaign*. The original R code
is kept unchanged, and no longer maintained, in [`legacy-r/`](legacy-r/).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                    # parsers, seat allocation, Kalman filter, scoring and model smoke tests
pollofpolls fetch         # cache the Wikipedia polling pages for 2011-2026
pollofpolls prep          # parse polls; verify election results against data/reference
pollofpolls fit           # NUTS fits of the ensemble variants (cached by input fingerprint)
pollofpolls forecast      # stacked ensemble -> seats, coalitions, output/*.csv and summary.json
pollofpolls report        # output/*.svg charts and site/index.html
pollofpolls backtest      # rolling-origin backtests on 2017/2020/2023 -> stacking weights (slow)
```

`pollofpolls all --quick` runs fetch to report with short MCMC chains for development. A full `fit`
takes roughly 5-10 minutes per variant on 4 CPU cores.

## The model

**State.** Weekly vote intention for the tracked parties and "Other" is a vector of additive log-ratios
against National. It follows a random walk with correlated innovations (LKJ prior), whose volatility is
multiplied by a learned factor in the last eight weeks of each campaign. The official result of every past
election pins the state exactly; between two results the path is sampled as a Brownian bridge, which
removes the stiff direction that made the Stan version need `max_treedepth = 20`. The `heavy` variant adds
a multivariate Student-t shock scale per week, so news weeks can move every party at once.

**Polls.** Each poll observes the state through three offsets, all on the logit scale with one value per
party so no party is privileged:

* a **persistent house effect** for each pollster (with a separate segment after Reid Research's 2017
  method change);
* a **per-term deviation** of each pollster from its persistent house effect (Ellis 2017's time-varying
  house effects, discretised by parliamentary term);
* an **industry-wide error** shared by all pollsters, which moves linearly over a term from its value
  just after the last election to its value at the next one. The first polls of a term identify the start
  value against the previous result, and each result identifies the end value. For 2026 the end value
  cannot be observed yet, so it is drawn from the spread of past election-day errors. This is how the
  forecast carries the finding of Shirani-Mehr, Rothschild, Goel and Gelman (2018) that total poll error
  is about twice the reported margin of error.

The likelihood is either **Dirichlet-multinomial** on the reported categories (`base`) or **Gaussian** on the
published shares with variance `deff * p(1-p)/n` plus the variance of rounding the published figure
(`gauss`, `heavy`, `fund`). Both use each poll's real sample size and a **design effect learned per
pollster**, replacing the fixed `n = 1000` and `sqrt(2)` inflator of the 2023 model. A party a poll did not
report is treated as unobserved, not as 0%.

**Fundamentals.** The `fund` variant adds Ellis's (2020) prior on the Prime Minister's party swing,
re-estimated at each target election from earlier elections only. It is kept for comparison but is not in
the published ensemble (see below).

**Seats.** Sainte-Laguë with the 5% threshold, the electorate lifeline, overhang, and independents
shrinking the proportional pool, as the Electoral Act 1993 requires. The allocation is unit-tested against the official
2011-2023 outcomes. Electorate wins by parties below the threshold, and by the two expelled Te Pāti Māori
MPs, are simulated from editorial assumptions in [`config/electorates.yml`](config/electorates.yml).

**Inference.** Everything runs in [NumPyro](https://num.pyro.ai) on JAX with the NUTS sampler on CPU. An
experimental `kal` variant integrates the weekly state out exactly with an information-form Kalman filter
and a Rauch-Tung-Striebel smoother, and samples only the hyperparameters. It is verified against
brute-force Gaussian integration in `tests/test_kalman.py`. It uses a Gaussian likelihood on log-ratios,
so it is a different model from `gauss`, not a faster route to the same posterior. It is not in the
ensemble because its sequential filter makes each gradient slower on CPU than full-state NUTS.

## Evaluation

Every variant, and a faithful re-implementation of the 2023 NZ Herald model (`legacy`, see
[`model/legacy.py`](src/pollofpolls/model/legacy.py)), is refitted using only the polls available 26, 8, 4
and 1 weeks before the 2017, 2020 and 2023 elections and scored against the official result. Scores are on
the vote-share scale over the tracked parties, excluding "Other": CRPS (the error of the whole forecast
distribution, in percentage points), mean absolute error, and 50% and 90% interval coverage. A Brier score
covers the seat-bloc lead. Variants are combined by **stacking on CRPS** (Yao, Vehtari, Simpson and Gelman
2018). The published ensemble is scored leave-one-election-out: its weights for each election are fitted on
the other two, so its scores are out of sample.

### Backtest results

Twelve cases: the 2017, 2020 and 2023 elections, each forecast 26, 8, 4 and 1 weeks out. Lower CRPS and error
are better; coverage should match its nominal level.

| Model | CRPS (pp) | Mean abs. error (pp) | 90% coverage | 50% coverage |
|---|---|---|---|---|
| Stacked ensemble (published; out of sample) | 1.39 | 1.95 | 0.86 | 0.54 |
| Ensemble + spread calibration (out of sample; not used) | 1.39 | 1.95 | 0.84 | 0.46 |
| Equal-weight ensemble | 1.40 | 1.95 | 0.89 | 0.52 |
| `gauss` | 1.37 | 1.91 | 0.90 | 0.54 |
| `heavy` | 1.38 | 1.93 | 0.87 | 0.51 |
| `base` (Dirichlet-multinomial) | 1.46 | 2.02 | 0.88 | 0.48 |
| **2023 NZ Herald model** (replica) | 1.70 | 2.14 | 0.56 | 0.25 |

CRPS in percentage points by weeks before the election, with 90% coverage in brackets:

| Model | 26 weeks | 8 weeks | 4 weeks | 1 week |
|---|---|---|---|---|
| Stacked ensemble (published; out of sample) | 2.22 (0.74) | 1.78 (0.88) | 0.76 (0.93) | 0.81 (0.89) |
| **2023 NZ Herald model** (replica) | 2.58 (0.53) | 1.94 (0.53) | 1.12 (0.67) | 1.14 (0.49) |

* The published ensemble has **18% lower CRPS** than the 2023 model and beat it in **10 of 12** cases. It lost on 2023 at 26 weeks and at 1 week.
* Its 90% intervals contained 86% of results against 56% for the 2023 model, whose fixed `n = 1000` and very smooth random walk made it overconfident at every horizon.
* Point accuracy improved less than calibration: mean absolute error fell from 2.14 to 1.95 pp. Late swings such as Labour's rise after Jacinda Ardern became leader seven weeks before the 2017 election cannot be forecast from polls.
* On the yes/no question of which bloc wins more seats, the 2023 model scored better (Brier 0.034 vs 0.139). It was confidently right in all three elections, partly because its smooth path barely reacted to the 2017 Labour surge. Three elections cannot separate that from luck.
* The variants differ little from each other, and three elections cannot rank them reliably. Stacking weights fitted on two elections and tested on the third came out close to the equal-weight blend and slightly worse than `gauss` alone. The 2026 weights, fitted on all three, are `gauss` 0.59, `heavy` 0.41 and `base` 0.00.
* Spread calibration (EMOS-style rescaling fitted on past elections) barely changed CRPS out of sample and made one-week intervals too narrow, so it is off (`forecast.calibrate_spread` in `config/model.yml`).
* The fundamentals prior made forecasts worse. For 2023 at 8 weeks its CRPS was 2.62 pp against 1.54 without it (`output/backtest/probe_fund_2023_h8.json`), because it pulled Labour towards its 2020 landslide. It is excluded from the ensemble.

Full tables: `output/backtest/summary.csv`, `summary_by_horizon.csv`, `head_to_head.csv` and one file per case in `output/backtest/cases/`.

## Research considered and what it means for this problem

The dataset is about 620 polls of eight series over 15 years, and the state has seven dimensions. That
scale decides which recent methods help.

* **Ensemble Kalman filters, including nested EnKF, replacing particle filters.** EnKF approximates the
  Kalman filter with a Monte Carlo ensemble so it can scale to states with thousands or millions of
  dimensions. With seven dimensions the exact filter is cheap and has no ensemble error, so the `kal`
  variant uses the exact filter. Where the likelihood is not Gaussian (Dirichlet-multinomial), HMC on the
  full state is used instead of a particle filter.
* **Deep latent dynamics (KVAE) and LLM components.** Neural state-space models need many long sequences to
  learn dynamics. One short multivariate series cannot identify them, and they would hide the quantities
  people read off this model: house effects, polling error and volatility. LLMs have no role in the
  likelihood, and poll extraction is a deterministic, tested parser.
* **Structured state-space duality, Mamba and hybrid attention architectures.** These are sequence-model
  layers for deep learning on very long token streams. Their "state space" is a learned linear recurrence
  inside a network, not a probabilistic latent state, and they need large training corpora.
* **Copula-based non-linear state-space models with HMC.** The useful part is already here. Cross-party
  dependence is a Gaussian dependence structure on the log-ratio scale (LKJ-correlated innovations), the
  observation model is non-linear, missing categories are handled exactly, and inference is HMC.
* **Probabilistic programming ecosystems (PyMC, NumPyro, dynestyx).** The model is written directly in
  NumPyro. The bridge conditioning on election results and the poll-specific offsets are simpler to express
  directly than through [dynestyx](https://github.com/BasisResearch/dynestyx) (June 2026), which could be
  adopted later without changing the model.

What was added instead: a compositional state, a Dirichlet-multinomial likelihood, learned design effects,
rounding error, time-varying house effects, an election-day polling error learned from past elections,
campaign volatility, heavy-tailed shocks, exact Kalman marginalisation as an option, proper scoring rules,
and stacking evaluated out of sample.

## Data

* **Polls.** The Wikipedia pages "Opinion polling for the *year* New Zealand general election", 2011–2026.
  The party-vote table is located by its header content, so column order and abbreviations (for example
  `OPP` for The Opportunities Party) may change between pages.
* **Results.** The "election result" rows on the same pages, verified against
  [`data/reference/election_results.csv`](data/reference/election_results.csv) (Electoral Commission
  figures, plus 1993–2005 National and Labour results for the fundamentals prior). The Commission's CSV
  endpoints that the R pipeline used now sit behind a JavaScript challenge and cannot be fetched by a script.
* **Pollsters.** Aliases, default sample sizes and method changes are in
  [`config/pollsters.yml`](config/pollsters.yml). Sponsored releases such as "Labour–Talbot Mills" are
  excluded because they are published selectively.

## Outputs

`output/summary.json`, `forecast.csv`, `seats.csv`, `coalitions.csv`, `trend.csv` and `house_effects.csv`.
Also the SVG charts under the file names the 2023 pipeline used (`voting_intention620.svg`,
`election_night375.svg`, ...), the backtest tables in `output/backtest/`, and the report at
`site/index.html`. The GitHub Actions workflow refreshes everything weekly and on every push, and publishes
the report to GitHub Pages; set Settings > Pages > Source to "GitHub Actions". Backtests run only when
triggered manually from the Actions tab, as three parallel jobs.

## License

GPL-3.0, as the original work by Peter Ellis and the NZ Herald.
