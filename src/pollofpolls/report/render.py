"""Render output/*.svg charts and the single-file HTML report (site/index.html)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..config import Config
from . import charts

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"


LABELS = {"legacy": "2023 model (replica)", "base": "Dirichlet-multinomial", "gauss": "Gaussian",
          "heavy": "Gaussian, heavy-tailed shocks", "fund": "Gaussian + fundamentals prior", "kal": "Kalman",
          "ensemble": "Stacked ensemble (out of sample)", "equal": "Equal-weight ensemble",
          "calibrated": "Stacked ensemble + spread calibration (out of sample)"}


def _backtest_context(bt_dir: Path) -> dict | None:
    """Tables for the report's backtest section, or None when no backtests have been run."""
    if not (bt_dir / "summary.csv").exists():
        return None
    order = ["ensemble", "calibrated", "equal", "heavy", "gauss", "base", "fund", "legacy"]
    rank = {v: i for i, v in enumerate(order)}
    summary = sorted(pl.read_csv(bt_dir / "summary.csv").to_dicts(), key=lambda r: rank.get(r["variant"], 99))
    for r in summary:
        r["label"] = LABELS.get(r["variant"], r["variant"])
    by_h = pl.read_csv(bt_dir / "summary_by_horizon.csv")
    horizons = sorted(by_h["horizon_weeks"].unique().to_list(), reverse=True)
    grid = []
    for v in [r["variant"] for r in summary]:
        sub = {row["horizon_weeks"]: row for row in by_h.filter(pl.col("variant") == v).to_dicts()}
        grid.append({"variant": v, "label": LABELS.get(v, v),
                     "cells": [sub.get(h) for h in horizons]})
    h2h = pl.read_csv(bt_dir / "head_to_head.csv").to_dicts() if (bt_dir / "head_to_head.csv").exists() else []
    stacking = json.loads((bt_dir / "stacking.json").read_text()) if (bt_dir / "stacking.json").exists() else {}
    targets = sorted(pl.read_csv(bt_dir / "scores.csv")["target"].unique().to_list())
    return {"summary": summary, "horizons": horizons, "targets": targets, "grid": grid, "h2h": h2h,
            "n_better": sum(1 for r in h2h if r["ensemble_better"]), "stacking": stacking,
            "labels": LABELS}


def render_report(cfg: Config) -> Path:
    out, site = cfg.paths.output, cfg.paths.site
    site.mkdir(parents=True, exist_ok=True)
    summary = json.loads((out / "summary.json").read_text())
    trend = pl.read_csv(out / "trend.csv")
    polls = pl.read_parquet(cfg.paths.processed / "polls.parquet")
    polls = polls.filter(pl.col("party").is_in(summary["parties"]))
    sims = np.load(out / "seat_sims.npz", allow_pickle=False)
    parties = list(summary["parties"])
    colours = cfg.colours
    coalitions = cfg.electorates_cfg["coalitions"]
    election_dates = [e.date for e in cfg.elections if e.year >= cfg.anchor_election]
    anchor_date = cfg.election(cfg.anchor_election).date
    polls_window = polls.filter(pl.col("mid_date") > anchor_date)

    # static SVGs (names kept from the 2023 R pipeline)
    charts.voting_intention_svg(trend, polls_window, election_dates, colours, out / "voting_intention620.svg", 2, 6.2, 8)
    charts.voting_intention_svg(trend, polls_window, election_dates, colours, out / "voting_intention375.svg", 1, 3.75, 10)
    charts.coalition_svg(sims["seats_election"], sims["total_election"], parties, coalitions, out / "election_night620.svg", 2, 6.2, 4)
    charts.coalition_svg(sims["seats_election"], sims["total_election"], parties, coalitions, out / "election_night375.svg", 1, 3.75, 7)
    charts.coalition_svg(sims["seats_now"], sims["total_now"], parties, coalitions, out / "saturday620.svg", 2, 6.2, 4)
    charts.coalition_svg(sims["seats_now"], sims["total_now"], parties, coalitions, out / "saturday375.svg", 1, 3.75, 7)

    # interactive figures
    since = cfg.election(cfg.forecast_election.year - 3).date - timedelta(days=60) if any(
        e.year == cfg.forecast_election.year - 3 for e in cfg.elections) else anchor_date
    figs = {}
    for ncol, suffix in ((2, ""), (1, "_1col")):
        figs["trend" + suffix] = charts.trend_figure(trend, polls_window, election_dates, colours, since, ncol=ncol)
        figs["trend_full" + suffix] = charts.trend_figure(trend, polls_window, election_dates, colours, anchor_date, ncol=ncol)
        figs["seats_election" + suffix] = charts.seats_figure(sims["seats_election"], sims["total_election"], parties, coalitions, ncol=ncol)
        figs["seats_now" + suffix] = charts.seats_figure(sims["seats_now"], sims["total_now"], parties, coalitions, ncol=ncol)
    house = pl.read_csv(out / "house_effects.csv") if (out / "house_effects.csv").stat().st_size > 30 else None
    if house is not None and house.height:
        figs["house"] = charts.house_effects_figure(house, colours)

    bt = _backtest_context(out / "backtest")

    recent_polls = (polls.filter(pl.col("cycle") == cfg.forecast_election.year)
                    .pivot(on="party", index=["mid_date", "pollster", "sample_size", "date_text"], values="share",
                           aggregate_function="first")
                    .sort("mid_date", descending=True).head(25))
    recent_cols = [p for p in parties if p in recent_polls.columns]

    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html"]))
    tpl = env.get_template("report.html")
    html = tpl.render(
        summary=summary, parties=parties, colours=colours, figs=figs,
        coalitions_election=summary["coalitions_election_day"], coalitions_now=summary["coalitions_now"],
        seats=summary["seats_election_day"], bt=bt, anchor_year=cfg.anchor_election,
        last_result_year=max(e.year for e in cfg.elections if not e.forecast),
        recent_polls=recent_polls.to_dicts(), recent_cols=recent_cols, today=date.today().isoformat(),
        election=cfg.forecast_election, house=house.to_dicts() if house is not None else [],
    )
    (site / "index.html").write_text(html, encoding="utf-8")
    for f in ["summary.json", "forecast.csv", "seats.csv", "coalitions.csv", "trend.csv", "house_effects.csv",
              "voting_intention620.svg", "voting_intention375.svg", "election_night620.svg", "election_night375.svg",
              "saturday620.svg", "saturday375.svg"]:
        if (out / f).exists():
            (site / f).write_bytes((out / f).read_bytes())
    print(f"[pollofpolls] report written to {site / 'index.html'}")
    return site / "index.html"
