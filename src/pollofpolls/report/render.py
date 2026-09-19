"""Build the website: static SVG charts, then Quarto renders website/ into site/, then the downloads are copied in.

`prepare` and `publish_files` are the website's Quarto pre- and post-render scripts (website/_scripts/), so
`quarto preview website` builds the same site as `pollofpolls report`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl
import yaml

from ..config import Config
from . import charts
from .site import BACKTEST_DOWNLOADS, CHARTS, DOWNLOADS, Site, long_date


def draw_static_charts(site: Site) -> None:
    """The SVG charts, under the file names the 2023 R pipeline used."""
    out, colours, dates = site.out, site.colours, site.election_dates
    polls = site.polls.filter(pl.col("mid_date") > site.anchor_date)
    charts.voting_intention_svg(site.trend, polls, dates, colours, out / "voting_intention620.svg", 2, 6.2, 8)
    charts.voting_intention_svg(site.trend, polls, dates, colours, out / "voting_intention375.svg", 1, 3.75, 10)
    charts.all_parties_svg(site.trend, dates, colours, out / "voting_intention_all620.svg", 6.2, 4.2, site.term_start)
    charts.all_parties_svg(site.trend, dates, colours, out / "voting_intention_all375.svg", 3.75, 4.6,
                           site.term_start, labels=False)
    coalitions, sims, parties = site.cfg.electorates_cfg["coalitions"], site.sims, site.parties
    for when, name in (("election", "election_night"), ("now", "saturday")):
        for ncol, width, height, size in ((2, 6.2, 5, 620), (1, 3.75, 10, 375)):
            charts.coalition_svg(sims[f"seats_{when}"], sims[f"total_{when}"], parties, coalitions,
                                 out / f"{name}{size}.svg", ncol, width, height, colours)


def prepare(root: Path | None = None) -> None:
    """Before Quarto renders: write website/_variables.yml from the config, and draw the static charts."""
    site = Site(root)
    if not (site.out / "summary.json").exists():
        raise SystemExit(f"{site.out / 'summary.json'} not found: run `pollofpolls forecast` before the website")
    project = Path(os.environ.get("QUARTO_PROJECT_DIR") or site.cfg.paths.root / "website")
    e = site.election
    variables = {"election": {"year": e.year, "date": long_date(e.date)},
                 "anchor": {"year": site.cfg.anchor_election}}
    (project / "_variables.yml").write_text("# Written by pollofpolls before each render; do not edit.\n"
                                            + yaml.safe_dump(variables, allow_unicode=True, sort_keys=False),
                                            encoding="utf-8")
    draw_static_charts(site)


def publish_files(root: Path | None = None) -> None:
    """After Quarto renders: copy the downloadable data, charts and backtest files into the site."""
    cfg = Config(root)
    out = cfg.paths.output
    site_dir = cfg.paths.site
    if os.environ.get("QUARTO_PROJECT_OUTPUT_DIR"):
        site_dir = Path(os.environ.get("QUARTO_PROJECT_DIR", ".")) / os.environ["QUARTO_PROJECT_OUTPUT_DIR"]
    for f in DOWNLOADS + CHARTS:
        if (out / f).exists():
            shutil.copyfile(out / f, site_dir / f)
    backtests = [f for f in BACKTEST_DOWNLOADS if (out / "backtest" / f).exists()]
    if backtests:
        (site_dir / "backtest").mkdir(exist_ok=True)
        for f in backtests:
            shutil.copyfile(out / "backtest" / f, site_dir / "backtest" / f)


def render_report(cfg: Config, quiet: bool = True) -> Path:
    """Build the website with Quarto into site/ and return the path of its front page."""
    quarto = shutil.which("quarto")
    if quarto is None:
        raise RuntimeError("Quarto is needed to build the website: install it from https://quarto.org/docs/get-started/")
    project = cfg.paths.root / "website"
    if cfg.paths.site.is_dir():
        shutil.rmtree(cfg.paths.site)        # Quarto does not clear an output directory outside its project
    # the pages' Python cells and the render scripts must run in this environment, against this checkout
    env = {**os.environ, "QUARTO_PYTHON": sys.executable, "POLLOFPOLLS_ROOT": str(cfg.paths.root)}
    subprocess.run([quarto, "render", str(project)] + (["--quiet"] if quiet else []), check=True, env=env)
    pages = sorted(p.name for p in cfg.paths.site.glob("*.html"))
    print(f"[pollofpolls] site written to {cfg.paths.site}: {', '.join(pages)}")
    return cfg.paths.site / "index.html"
