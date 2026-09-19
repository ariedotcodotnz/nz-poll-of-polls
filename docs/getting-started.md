# Getting started

## Requirements

* Python 3.11 or newer.
* A few CPU cores. No GPU is needed. A production fit of one model variant takes about 8 minutes with
  4 chains on 4 cores; the full backtest takes 1 to 2 hours.
* Internet access for the `fetch` stage (Wikipedia) and for the report's charts (Plotly loads from a CDN).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest                 # about 100 tests, around 2 minutes
```

## Run everything

```bash
pollofpolls all        # fetch -> prep -> fit -> forecast -> report
```

`pollofpolls all --quick` uses short MCMC chains (2 chains, 200 warmup and 200 draws) for a fast check.
Quick fits overwrite the cached production fits, so run `pollofpolls fit --force` before publishing.

Stage by stage:

```bash
pollofpolls fetch      # download the Wikipedia polling pages into data/raw/wikipedia/
pollofpolls prep       # parse polls, verify election results -> data/processed/
pollofpolls fit        # NUTS fits of the ensemble variants -> data/processed/fit_*.npz
pollofpolls forecast   # stacked forecast, seats, coalitions -> output/
pollofpolls report     # charts and the HTML report -> output/*.svg, site/
```

## View the report

The report is a static page:

```bash
python -m http.server 8000 --directory site
```

Open <http://localhost:8000>. In PyCharm or WebStorm you can also open `site/index.html` and click the browser
icon at the top right of the editor.

## Where things are

| Path | Contents |
|---|---|
| `config/` | elections, pollsters, electorate assumptions, model variants and priors |
| `data/reference/` | official election results and electorate seats, committed |
| `data/raw/`, `data/processed/` | downloaded pages, parsed tables, datasets and fits; regenerated, not committed |
| `output/` | forecast tables, `summary.json`, SVG charts, backtest results |
| `site/` | the rendered report |
| `src/pollofpolls/` | the package |
| `tests/` | unit, regression and end-to-end tests, with Wikipedia table fixtures |

## Common tasks

| Task | Command |
|---|---|
| Refresh with new polls | `pollofpolls all` |
| Refit one variant from scratch | `pollofpolls fit --variants gauss --force` |
| Re-render the report only | `pollofpolls report` |
| Re-run the backtests | `pollofpolls backtest` (slow; see [Evaluation](evaluation.md)) |
| Rescore backtests without refitting | `pollofpolls backtest --no-run` |
