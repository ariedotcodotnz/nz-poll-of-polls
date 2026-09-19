# Getting started

## Requirements

* Python 3.11 or newer.
* [Quarto](https://quarto.org/docs/get-started/) 1.10 or newer, to build the website.
* A few CPU cores. No GPU is needed. A production fit of one model variant takes about 8 minutes with
  4 chains on 4 cores; the full backtest takes 1 to 2 hours.
* Internet access for the `fetch` stage (Wikipedia), and for the website's charts, maths and fonts, which load
  from CDNs.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest                 # about 110 tests, around 2 minutes
quarto check           # Quarto should find this environment's Python and Jupyter
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
pollofpolls report     # the website, built with Quarto -> output/*.svg, site/
```

## View the website

While editing, let Quarto render and serve it, reloading as pages change:

```bash
quarto preview website
```

Run it with the environment above active: the pages' Python cells import `pollofpolls`. To look at the site
`pollofpolls report` built, serve `site/`:

```bash
python -m http.server 8000 --directory site
```

and open <http://localhost:8000>. The site has the forecast and two pages on method, "How the model works" and
"How it is tested". This developer documentation is not part of it.

## Where things are

| Path | Contents |
|---|---|
| `config/` | elections, pollsters, electorate assumptions, model variants and priors |
| `data/reference/` | official election results and electorate seats, committed |
| `data/raw/`, `data/processed/` | downloaded pages, parsed tables, datasets and fits; regenerated, not committed |
| `output/` | forecast tables, `summary.json`, SVG charts, backtest results |
| `website/` | the website's Quarto source: the forecast page, the two pages on method, partials and theme |
| `site/` | the built website |
| `docs/` | this developer documentation |
| `src/pollofpolls/` | the package |
| `tests/` | unit, regression and end-to-end tests, with Wikipedia table fixtures |

## Common tasks

| Task | Command |
|---|---|
| Refresh with new polls | `pollofpolls all` |
| Refit one variant from scratch | `pollofpolls fit --variants gauss --force` |
| Rebuild the website only | `pollofpolls report` |
| Edit a website page with live reload | `quarto preview website` |
| Re-run the backtests | `pollofpolls backtest` (slow; see [Operations](operations.md#running-the-backtests)) |
| Rescore backtests without refitting | `pollofpolls backtest --no-run` |
