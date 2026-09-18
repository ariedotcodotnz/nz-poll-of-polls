"""Static SVG charts (matplotlib) and interactive Plotly figures for the report."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_COLOUR = "#777777"
FONT = dict(family="-apple-system, 'Segoe UI', Roboto, 'Source Sans Pro', sans-serif", size=12)


def _party_order(parties: list[str], colours: dict[str, str]) -> list[str]:
    order = [p for p in colours if p in parties]
    return order + [p for p in parties if p not in order]


# ------------------------------------------------------------------------------- matplotlib SVGs
def voting_intention_svg(trend: pl.DataFrame, polls: pl.DataFrame, election_dates: list[date],
                         colours: dict[str, str], path: Path, ncol: int, width: float, height: float,
                         today: date | None = None) -> Path:
    today = today or date.today()
    parties = _party_order(trend["party"].unique().to_list(), colours)
    nrow = int(np.ceil(len(parties) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(width, height), squeeze=False)
    for ax, party in zip(axes.flat, parties):
        tr = trend.filter(pl.col("party") == party).sort("week")
        weeks = [date.fromisoformat(w) for w in tr["week"].to_list()]
        c = colours.get(party, DEFAULT_COLOUR)
        pp = polls.filter(pl.col("party") == party)
        ax.scatter(pp["mid_date"].to_list(), pp["share"].to_numpy() * 100, s=4, color="black", alpha=0.5, zorder=3)
        ax.fill_between(weeks, tr["q05"].to_numpy() * 100, tr["q95"].to_numpy() * 100, color=c, alpha=0.25, lw=0)
        ax.fill_between(weeks, tr["q25"].to_numpy() * 100, tr["q75"].to_numpy() * 100, color=c, alpha=0.35, lw=0)
        obs = [w <= today for w in weeks]
        ax.plot([w for w, o in zip(weeks, obs) if o], tr["mean"].to_numpy()[obs] * 100, color="black", lw=1)
        for d in election_dates:
            ax.axvline(d, color="grey", lw=0.6)
        ax.set_title(party, fontsize=9, loc="left", fontweight="bold")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in list(axes.flat)[len(parties):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, format="svg", dpi=100)
    plt.close(fig)
    return path


def coalition_svg(seats: np.ndarray, total: np.ndarray, parties: list[str], coalitions: list[dict], path: Path,
                  ncol: int, width: float, height: float) -> Path:
    idx = {p: i for i, p in enumerate(parties)}
    nrow = int(np.ceil(len(coalitions) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(width, height), squeeze=False)
    majority = int(np.median(total // 2 + 1))
    for ax, c in zip(axes.flat, coalitions):
        s = seats[:, [idx[p] for p in c["parties"] if p in idx]].sum(1)
        lab_based = c["parties"][0] == "Labour"
        colour = "#d82c20" if lab_based else "#065BAA"
        bins = np.arange(s.min() - 0.5, s.max() + 1.5, 1)
        ax.hist(s, bins=bins, color=colour, alpha=0.6, lw=0)
        ax.axvline(majority, color="black", lw=0.8)
        p_maj = (s >= (total // 2 + 1)).mean()
        ax.set_title(f"{c['name']}  ({p_maj:.0%} majority)", fontsize=8, loc="left", fontweight="bold")
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    for ax in list(axes.flat)[len(coalitions):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, format="svg", dpi=100)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------ plotly JSON
def trend_figure(trend: pl.DataFrame, polls: pl.DataFrame, election_dates: list[date], colours: dict[str, str],
                 since: date, today: date | None = None, ncol: int = 2) -> dict:
    """Plotly figure dict: one subplot per party with polls, mean and 50/90% bands."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    today = today or date.today()
    parties = _party_order(trend["party"].unique().to_list(), colours)
    nrow = int(np.ceil(len(parties) / ncol))
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=parties, vertical_spacing=0.35 / nrow,
                        horizontal_spacing=0.06)
    for i, party in enumerate(parties):
        r, c = i // ncol + 1, i % ncol + 1
        tr = trend.filter((pl.col("party") == party) & (pl.col("week") >= since.isoformat())).sort("week")
        weeks = tr["week"].to_list()
        col = colours.get(party, DEFAULT_COLOUR)
        rgba = _rgba(col, 0.25)
        fig.add_trace(go.Scatter(x=weeks + weeks[::-1], y=list(tr["q95"] * 100) + list((tr["q05"] * 100))[::-1],
                                 fill="toself", fillcolor=rgba, line=dict(width=0), hoverinfo="skip",
                                 showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(x=weeks + weeks[::-1], y=list(tr["q75"] * 100) + list((tr["q25"] * 100))[::-1],
                                 fill="toself", fillcolor=_rgba(col, 0.35), line=dict(width=0), hoverinfo="skip",
                                 showlegend=False), row=r, col=c)
        obs = tr.filter(pl.col("week") <= today.isoformat())
        fig.add_trace(go.Scatter(x=obs["week"].to_list(), y=list(obs["mean"] * 100), mode="lines",
                                 line=dict(color=col, width=2), name=party, showlegend=False,
                                 hovertemplate="%{x}: %{y:.1f}%<extra>" + party + "</extra>"), row=r, col=c)
        fut = tr.filter(pl.col("week") >= today.isoformat())
        fig.add_trace(go.Scatter(x=fut["week"].to_list(), y=list(fut["mean"] * 100), mode="lines",
                                 line=dict(color=col, width=2, dash="dot"), showlegend=False,
                                 hovertemplate="%{x}: %{y:.1f}%<extra>forecast</extra>"), row=r, col=c)
        pp = polls.filter((pl.col("party") == party) & (pl.col("mid_date") >= since))
        fig.add_trace(go.Scatter(x=[d.isoformat() for d in pp["mid_date"].to_list()], y=list(pp["share"] * 100),
                                 mode="markers", marker=dict(color="black", size=4, opacity=0.5), showlegend=False,
                                 text=pp["pollster"].to_list(),
                                 hovertemplate="%{text}<br>%{x}: %{y:.1f}%<extra></extra>"), row=r, col=c)
        for d in election_dates:
            if d >= since:
                fig.add_vline(x=d.isoformat(), line=dict(color="grey", width=1), row=r, col=c)
    fig.update_yaxes(ticksuffix="%", gridcolor="#eee")
    fig.update_xaxes(gridcolor="#eee")
    fig.update_layout(height=(260 if ncol > 1 else 220) * nrow, margin=dict(l=30, r=10, t=40, b=30),
                      plot_bgcolor="white", paper_bgcolor="white", font=FONT)
    return json.loads(fig.to_json())


def seats_figure(seats: np.ndarray, total: np.ndarray, parties: list[str], coalitions: list[dict],
                 ncol: int = 2) -> dict:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    idx = {p: i for i, p in enumerate(parties)}
    nrow = int(np.ceil(len(coalitions) / ncol))
    titles = []
    for c in coalitions:
        s = seats[:, [idx[p] for p in c["parties"] if p in idx]].sum(1)
        titles.append(f"{c['name']}<br><span style='font-weight:normal'>{(s >= total // 2 + 1).mean():.0%} chance of a majority</span>")
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=titles, vertical_spacing=0.45 / nrow,
                        horizontal_spacing=0.08)
    majority = int(np.median(total // 2 + 1))
    for i, c in enumerate(coalitions):
        r, cc = i // ncol + 1, i % ncol + 1
        s = seats[:, [idx[p] for p in c["parties"] if p in idx]].sum(1)
        colour = "#d82c20" if c["parties"][0] == "Labour" else "#065BAA"
        vals, counts = np.unique(s, return_counts=True)
        fig.add_trace(go.Bar(x=vals, y=counts / counts.sum(), marker_color=colour, opacity=0.7, showlegend=False,
                             hovertemplate="%{x} seats: %{y:.1%}<extra></extra>"), row=r, col=cc)
        fig.add_vline(x=majority - 0.5, line=dict(color="black", width=1), row=r, col=cc)
    fig.update_yaxes(tickformat=".0%", gridcolor="#eee")
    fig.update_annotations(font_size=12)
    fig.update_layout(height=(240 if ncol > 1 else 200) * nrow, margin=dict(l=30, r=10, t=50, b=30),
                      plot_bgcolor="white", paper_bgcolor="white", bargap=0.05, font=FONT)
    return json.loads(fig.to_json())


def house_effects_figure(house: pl.DataFrame, colours: dict[str, str]) -> dict:
    import plotly.graph_objects as go

    fig = go.Figure()
    parties = _party_order(house["party"].unique().to_list(), colours)
    houses = house["house"].unique(maintain_order=True).to_list()
    for party in parties:
        sub = house.filter(pl.col("party") == party)
        vals = {h: v for h, v in zip(sub["house"].to_list(), sub["effect_pp"].to_list())}
        fig.add_trace(go.Bar(name=party, x=houses, y=[vals.get(h, 0) for h in houses],
                             marker_color=colours.get(party, DEFAULT_COLOUR),
                             hovertemplate="%{x}<br>" + party + ": %{y:+.1f} pp<extra></extra>"))
    fig.update_layout(barmode="group", height=460, margin=dict(l=30, r=10, t=30, b=120), plot_bgcolor="white",
                      paper_bgcolor="white", yaxis=dict(ticksuffix=" pp", gridcolor="#eee", zeroline=True),
                      legend=dict(orientation="h", y=-0.45), font=FONT)
    return json.loads(fig.to_json())


def _rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
