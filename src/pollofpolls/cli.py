"""Command line interface: pollofpolls fetch | prep | fit | forecast | backtest | report | all."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, help="NZ Poll of Polls: poll aggregation and seat forecast")


def _cfg(root: Optional[Path]):
    from .config import Config
    return Config(root)


@app.command()
def fetch(root: Optional[Path] = None, force: bool = False):
    """Download the Wikipedia polling pages (conditional requests; cached under data/raw)."""
    from .pipeline import fetch as _fetch
    _fetch(_cfg(root), force=force)


@app.command()
def prep(root: Optional[Path] = None):
    """Parse cached pages into data/processed/polls.parquet and verify election results."""
    from .pipeline import prep as _prep
    _prep(_cfg(root))


@app.command()
def fit(variants: Optional[str] = typer.Option(None, help="comma-separated variant names (default: ensemble)"),
        root: Optional[Path] = None, force: bool = False, quick: bool = False):
    """Fit the model variants for the forecast election (cached by input fingerprint)."""
    from .pipeline import fit_variants
    cfg = _cfg(root)
    override = {"chains": 2, "warmup": 200, "samples": 200} if quick else None
    fit_variants(cfg, variants.split(",") if variants else None, force=force, mcmc_override=override)


@app.command()
def forecast(variants: Optional[str] = None, root: Optional[Path] = None):
    """Combine fitted variants, simulate seats and coalitions, write output/."""
    from .pipeline import forecast as _forecast
    cfg = _cfg(root)
    _forecast(cfg, variants.split(",") if variants else None)


@app.command()
def backtest(variants: Optional[str] = None, targets: Optional[str] = None, horizons: Optional[str] = None,
             root: Optional[Path] = None, force: bool = False,
             aggregate: bool = typer.Option(True, help="score all cached cases and refit stacking weights"),
             run: bool = typer.Option(True, help="fit missing or stale cases (use --no-run to only aggregate)")):
    """Rolling-origin backtests on past elections; writes output/backtest/ and stacking weights."""
    from .pipeline import backtest as _backtest
    cfg = _cfg(root)
    _backtest(cfg, variants.split(",") if variants else None,
              [int(x) for x in targets.split(",")] if targets else None,
              [int(x) for x in horizons.split(",")] if horizons else None, force=force, run=run,
              aggregate=aggregate)


@app.command()
def report(root: Optional[Path] = None):
    """Render charts (output/*.svg) and the HTML report (site/index.html)."""
    from .report.render import render_report
    render_report(_cfg(root))


@app.command()
def all(root: Optional[Path] = None, force: bool = False, quick: bool = False, skip_fetch: bool = False):
    """Run fetch, prep, fit, forecast and report."""
    from .pipeline import fetch as _fetch, fit_variants, forecast as _forecast, prep as _prep
    from .report.render import render_report
    cfg = _cfg(root)
    if not skip_fetch:
        _fetch(cfg)
    _prep(cfg)
    override = {"chains": 2, "warmup": 200, "samples": 200} if quick else None
    fit_variants(cfg, force=force, mcmc_override=override)
    _forecast(cfg)
    render_report(cfg)


if __name__ == "__main__":
    app()
