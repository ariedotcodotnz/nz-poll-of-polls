# Evaluation

Every modelling choice in this project was kept or dropped on evidence from backtests: refitting the model as
it would have run before past elections, and scoring the forecasts against what happened.

## Backtest design

* **Targets.** The 2017, 2020 and 2023 elections.
* **Horizons.** Forecasts made 26, 8, 4 and 1 weeks before each election: twelve cases per variant.
* **Information set.** Only the polls available at the cutoff, and only the results of earlier elections.
  Wikipedia records when fieldwork happened, not when a poll was published, so a poll counts once its
  fieldwork has ended and a typical publication delay for its pollster has passed
  (`publication_lag_days` in `config/pollsters.yml`: 2 days for Verian and Reid Research, 10 for Roy Morgan
  and Talbot Mills, 5 by default). The tracked parties, the pollsters kept and the fundamentals prior are all
  derived from that information set too.
* **Variants.** The 2023 model replica (`legacy`), `base`, `gauss` and `heavy`, with 2 chains of 500 warmup
  and 500 draws each. The fundamentals variant was tested separately; see below.

## Scores

Scores use the vote shares of the tracked parties, excluding the residual "Other" category.

| Score | What it measures | Better |
|---|---|---|
| CRPS | error of the whole forecast distribution, in percentage points; equals the absolute error for a point forecast | lower |
| Energy score | multivariate CRPS over all tracked parties jointly | lower |
| Mean absolute error | error of the forecast mean | lower |
| 90% and 50% coverage | share of party results inside the central 90% and 50% intervals | close to 0.9 and 0.5 |
| Brier score (bloc lead) | probability forecast of the right bloc winning more seats than the left bloc, using the electorates each party actually won | lower |

A log score on the log-ratio scale was tried first and rejected. Parties on 0.1% to 0.5%, such as United Future
in 2017, dominated it, although their errors do not affect seats.

## Combining variants

The published forecast is a weighted mixture of variants. The weights come from **stacking on CRPS**
(Yao, Vehtari, Simpson and Gelman 2018): choose weights $w$ on the simplex that minimise the total CRPS of the
mixture over all backtest cases. For a mixture the CRPS is a convex quadratic,

$$\text{CRPS}(w) = w^\top A - \tfrac{1}{2}\, w^\top B\, w, \qquad A_a = \mathbb{E}|X_a - y|, \quad B_{ab} = \mathbb{E}|X_a - X_b'|,$$

summed over parties and cases, and a small quadratic programme (SLSQP) finds its minimum.

The weights used for 2026 are fitted on all twelve cases: `gauss` 0.60, `heavy` 0.40.

To score the ensemble honestly, each election is held out in turn: its weights are fitted on the other two
elections only (2017: `gauss` 1.00; 2020: `gauss` 0.03, `heavy` 0.97; 2023: `gauss` 0.50, `heavy` 0.50). The equal-weight mixture of the candidates is scored for comparison.

## Spread calibration

A common post-processing step (EMOS, Gneiting and others 2005) rescales a forecast's spread by a factor learned
from past errors. Here the factor is $c(h) = \exp(a + b \log h)$ for a forecast $h$ weeks out, applied on the
log-ratio scale about the mixture mean, and fitted by minimising CRPS. It is evaluated with nested hold-outs:
when an election is held out, neither the spread factor nor the stacking weights inside its training mixtures
see that election's result.

It left CRPS unchanged and made one-week intervals too narrow, so it is off (`forecast.calibrate_spread`).

## Results

Twelve cases each, as of September 2026:

| Model | CRPS (pp) | Energy (pp) | MAE (pp) | 90% coverage | 50% coverage | Brier (bloc) |
|---|---|---|---|---|---|---|
| Stacked ensemble (the published method), out of sample | 1.60 | 5.54 | 2.29 | 0.87 | 0.47 | 0.123 |
| Stacked ensemble with spread calibration, out of sample | 1.60 | 5.50 | 2.31 | 0.84 | 0.46 | 0.126 |
| Equal-weight ensemble | 1.67 | 5.78 | 2.36 | 0.87 | 0.46 | 0.123 |
| `gauss` | 1.58 | 5.51 | 2.27 | 0.89 | 0.47 | 0.123 |
| `heavy` | 1.60 | 5.55 | 2.28 | 0.86 | 0.41 | 0.111 |
| `base` | 1.90 | 6.51 | 2.60 | 0.82 | 0.42 | 0.150 |
| 2023 model replica (`legacy`) | 1.97 | 6.60 | 2.47 | 0.55 | 0.18 | 0.041 |

CRPS by horizon, with 90% coverage in brackets:

| Model | 26 weeks | 8 weeks | 4 weeks | 1 week |
|---|---|---|---|---|
| Stacked ensemble | 2.74 (0.69) | 1.75 (0.85) | 1.04 (1.00) | 0.87 (0.93) |
| 2023 model replica | 3.22 (0.58) | 2.05 (0.51) | 1.41 (0.63) | 1.19 (0.49) |

* The ensemble's CRPS is 19% lower than the 2023 model's,
  and it scored better in 9 of 12 cases. The 2023 model was better only on the 2023
  election at 1, 8 and 26 weeks, by small margins.
* The biggest gain is calibration. The 2023 model's 90% intervals held 55% of results, because its
  fixed sample size and very smooth random walk made it overconfident at every horizon.
* On the bloc-lead question the 2023 model scored better. It was confidently right in all three elections,
  partly because its smooth path barely reacted to Labour's surge after Jacinda Ardern became leader in 2017.
  Three elections cannot separate that from luck.
* The Gaussian variants beat the Dirichlet-multinomial one, which gets no weight. The Gaussian and
  heavy-tailed variants are too close for three elections to rank.

### Fundamentals prior

Forecasting 2023 eight weeks out, the Gaussian variant scored a CRPS of 1.43 points without the fundamentals
prior and 2.61 with it. The prior pulled Labour up to 38.1% against 30.6% without it; Labour won 26.9%. Its
anchor is the PM party's previous result, which in 2020 was a landslide. The prior is excluded from the
ensemble. The two cases are in `output/backtest/probe_fundamentals.json`.

## Limitations

* **Three elections.** Twelve cases sound like more than they are: forecasts of the same election at different
  horizons share its surprises. Differences of a few percent in CRPS between variants are not meaningful.
* **Electorates are not tested.** The bloc-lead score uses the electorates each party actually won, so it tests
  the vote forecast only. The 2026 electorate assumptions in `config/electorates.yml` are editorial.
* **Publication delays are approximate**, so a backtest may occasionally include a poll a day or two before it
  was published, or exclude one after.
* **Single-poll pollsters are dropped** (`min_polls: 2`), for example Anacta in 2026.

## Running the backtests

```bash
pollofpolls backtest                 # all variants, targets and horizons in config/model.yml
pollofpolls backtest --no-run        # rescore cached cases, refit weights, rewrite the tables
```

A full run takes one to two hours on 8 cores. To use several processes, start one target per process and
aggregate at the end:

```bash
for y in 2017 2020 2023; do pollofpolls backtest --targets $y --no-aggregate & done; wait
pollofpolls backtest --no-run
```

Results are written to `output/backtest/`; see [Outputs](outputs.md). In GitHub Actions the backtests run only
on request, as three parallel jobs; see [Operations](operations.md).
