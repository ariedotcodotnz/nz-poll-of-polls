# Pipeline

The pipeline has six stages, each a `pollofpolls` subcommand. Every stage reads what the previous one wrote,
so stages can be rerun independently.

| Stage | Reads | Writes |
|---|---|---|
| `fetch` | Wikipedia | `data/raw/wikipedia/<year>.html`, `<year>.meta.json` |
| `prep` | raw pages, `data/reference/` | `data/processed/polls.parquet`, `results.json` |
| `fit` | processed data, `config/` | `data/processed/dataset_<year>.*`, `fit_<variant>_<year>.*` |
| `forecast` | fits, `output/backtest/stacking.json`, `config/electorates.yml` | `output/*.csv`, `summary.json`, `seat_sims.npz` |
| `report` | `output/`, `polls.parquet` | `output/*.svg`, `site/` |
| `backtest` | processed data | `output/backtest/` |

`pollofpolls all` runs fetch, prep, fit, forecast and report in order.

## fetch

Downloads the polling article for every election from `anchor_election` to the forecast election
(see [Configuration](configuration.md)); later calendar entries are ignored. Requests are conditional: the
`ETag` and `Last-Modified` headers are kept in `<year>.meta.json` with the article they came from, and an
unchanged page is not downloaded again. If an election's article changes, it is downloaded in full.

| Option | Effect |
|---|---|
| `--force` | ignore the cached validators and download every page |

## prep

Parses each page's party-vote table into one row per poll and party, then checks the election-result rows
against `data/reference/election_results.csv`. Any difference above 0.06 percentage points stops the
pipeline, because it means the page or the parser has changed. See [Data](data.md) for the parsing and
cleaning rules.

## fit

Builds the dataset for the forecast election from every poll whose fieldwork has ended, then fits each model
variant with NUTS.

| Option | Effect |
|---|---|
| `--variants a,b` | fit these variants instead of the ensemble |
| `--force` | refit even if the cached fit is current |
| `--quick` | 2 chains, 200 warmup and 200 draws; overwrites the cached fit |

By default the variants are the ensemble members whose stacking weight is at least 0.01. When the stacking
file is missing or does not cover every candidate in `ensemble`, all candidates are fitted and weighted equally.

## forecast

Mixes the fitted variants with their stacking weights and simulates election-day and "held now" outcomes:
4,000 draws of vote shares, electorate wins from `config/electorates.yml`, Sainte-Laguë seat allocation and
the coalition table. The weekly trend and the seat simulation use the same weighted mixture. If
`forecast.calibrate_spread` is on, the spread is rescaled first (see [How it is tested](https://ariedotcodotnz.github.io/nz-poll-of-polls/evaluation.html)).

| Option | Effect |
|---|---|
| `--variants a,b` | mix only these fitted variants |

## report

Builds the website with [Quarto](https://quarto.org) from `website/` into `site/`, which is cleared first.
Before rendering, it writes the election's year and date for the pages and draws the static SVG charts,
including those under the file names the 2023 pipeline used. Quarto then runs each page's Python cells, which
read `output/`: the forecast page with interactive Plotly charts, and the two pages on method. Afterwards the
CSV, JSON and SVG outputs are copied into `site/`. Text from Wikipedia is escaped before it reaches a page, so
it cannot inject script or run a Quarto shortcode. Needs Quarto 1.10 or newer.

## backtest

Refits variants on the polls available some weeks before past elections and scores them against the
results. Each case is a variant, a target election and a horizon.

| Option | Effect |
|---|---|
| `--variants`, `--targets`, `--horizons` | override the lists in `config/model.yml` |
| `--force` | refit cases even if cached |
| `--no-run` | only rescore and aggregate the cached cases |
| `--no-aggregate` | only fit cases; useful when running targets in parallel |

To use several machines or processes, run one target per process with `--no-aggregate`, then run
`pollofpolls backtest --no-run` once. See [How it is tested](https://ariedotcodotnz.github.io/nz-poll-of-polls/evaluation.html) for what the aggregation computes. A case scored with
other blocs than the current `backtest.blocs` is rescored from its cached fit; if the fit is gone, rerun it.

## Caching

Fits are expensive, so each one is keyed by a fingerprint of everything it depends on:

* the complete dataset, including polls, masks, sample sizes, weeks, election results, campaign weeks,
  pollster segments and the fundamentals prior, but not the cutoff date;
* the variant definition, the priors and the MCMC settings;
* the model's Python source files. Bytecode and cache directories are ignored.

A fit is reused only when its fingerprint matches. Live fits keep their stamps in `data/processed/.stamps/`
and backtest fits in `output/backtest/fits/*.stamp`. Use `--force`, or delete the stamp, to refit anyway.
Because the cutoff date is excluded, rerunning on a later day with no new polls does not refit.

## Environment variables

| Variable | Effect |
|---|---|
| `POLLOFPOLLS_ROOT` | project root; by default the first parent directory containing `pyproject.toml` and `config/` |
| `XLA_FLAGS` | JAX CPU settings. If unset, the package sets `--xla_force_host_platform_device_count=4` so four chains run in parallel. |
