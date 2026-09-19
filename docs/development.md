# Development

## Code layout

| Module | Responsibility |
|---|---|
| `config.py` | loads the YAML files; paths; the election calendar |
| `cache.py` | fingerprints and stage stamps |
| `pipeline.py` | the stages behind the command line |
| `cli.py` | the `pollofpolls` command |
| `data/wikipedia.py` | downloading and parsing the polling tables |
| `data/dates.py`, `data/parties.py`, `data/pollsters.py` | date ranges, party names, pollster aliases and delays |
| `data/results.py` | election results and their verification |
| `prep/polls_table.py` | the tidy poll table, poll eligibility and tracked parties |
| `prep/marshal.py` | the `Dataset` the models read; the fundamentals prior |
| `model/numpyro_model.py` | the main model: dynamics, bridge sampling, offsets, likelihoods |
| `model/kalman.py` | the Kalman-filter variant |
| `model/legacy.py` | the 2023 model replica |
| `model/fit.py` | running NUTS, diagnostics, saving and loading fits |
| `forecast/seats.py`, `forecast/simulate.py`, `forecast/coalitions.py` | seat allocation, electorate simulation, coalition tables |
| `eval/scoring.py`, `eval/backtest.py`, `eval/calibration.py` | scores, backtests and stacking, spread calibration |
| `report/charts.py`, `report/render.py`, `templates/report.html` | charts and the HTML report |

Conventions: vote shares are proportions; the first tracked party is National, the log-ratio reference, and the
last is "Other"; weeks start on Sunday; everything that could change between elections is in `config/`.

## Tests

```bash
pytest -q                          # everything, about 2 minutes
pytest -q -k "not recovers_path"   # skip the slow model smoke tests
```

| File | Covers |
|---|---|
| `test_dates.py`, `test_parties.py`, `test_wikipedia.py` | parsing every page layout from 2011 to 2026 with saved fixtures; pollster aliases; result verification |
| `test_marshal.py` | the dataset, eligibility at a backtest cutoff, fingerprints, the fundamentals prior |
| `test_seats.py` | seat allocation against the official 2011–2023 outcomes |
| `test_kalman.py` | the Kalman likelihood and smoother against brute-force Gaussian integration |
| `test_scoring.py` | CRPS, energy score, mixture CRPS, stacking, the bridge sampler |
| `test_fixes.py` | regression tests: transition density, fit persistence, source fingerprints, electorate probabilities |
| `test_model_smoke.py` | every variant recovers a known path from simulated polls |
| `test_pipeline_e2e.py` | prep, forecast and report on the fixtures, including a hostile pollster name that must not inject script |

Tests make no network calls. The fixtures in `tests/fixtures/` hold only each page's party-vote table.

## Adding a model variant

1. If the variant only combines existing options, add an entry under `variants` in `config/model.yml`.
2. Otherwise add an option in `model/numpyro_model.py`: dynamics go in `sample_dynamics`, offsets in
   `sample_offsets` and `offsets_from_params`, and likelihoods in `poll_model`. Read it with `variant.get(...)`
   so existing variants are unaffected, and add a case to `tests/test_model_smoke.py`.
3. Add the variant to `backtest.variants`, run the backtests, and compare it in `output/backtest/summary.csv`.
4. If it helps, add it to `ensemble`. Stacking decides its weight.

## Performance

A NUTS iteration typically takes 255 leapfrog steps, tree depth 8. A production fit of 4 chains with 2,000
iterations each took 8 to 16 minutes on 4 cores of a shared machine; a backtest case with 2 chains takes 2 to 6
minutes. The four chains run in parallel because the package asks XLA for four host devices.

When running several processes at once, as for parallel backtests, stop each from spreading across every core:

```bash
export XLA_FLAGS="--xla_force_host_platform_device_count=4 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `no party-vote table found` | the Wikipedia layout changed. Save the new table as a fixture, fix `data/wikipedia.py` until `test_wikipedia.py` passes, then check the parsed polls. |
| `prep` reports a results mismatch | a result row on Wikipedia was edited, or a parser change broke the columns |
| A column is missing from `polls.parquet` | the table was written by an older version; run `pollofpolls prep` |
| Fits rerun every time | the data changed, which is expected when there are new polls, or a model source file changed |
| Only one chain runs at a time | `XLA_FLAGS` was set without `--xla_force_host_platform_device_count` |
| Charts are blank in the report | Plotly could not load from the CDN; check the internet connection |
