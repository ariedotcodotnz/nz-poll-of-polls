"""Static SVG charts (matplotlib) and interactive Plotly figures for the report."""

from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

if TYPE_CHECKING:
    import plotly.graph_objects as go

DEFAULT_COLOUR = "#777777"
INK = "#1b1f23"          # primary text
MUTED = "#6a737d"        # secondary text, leader lines, crosshair
GRID = "#e5e7eb"         # hairline gridlines
RULE = "#9aa0a6"         # election-day rules
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


def _coalition_colour(coalition: dict, colours: dict[str, str]) -> str:
    """A coalition takes the colour of its lead party (National or Labour)."""
    return colours.get(coalition["parties"][0], DEFAULT_COLOUR)


def coalition_svg(seats: np.ndarray, total: np.ndarray, parties: list[str], coalitions: list[dict], path: Path,
                  ncol: int, width: float, height: float, colours: dict[str, str] | None = None) -> Path:
    idx = {p: i for i, p in enumerate(parties)}
    nrow = int(np.ceil(len(coalitions) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(width, height), squeeze=False)
    majority = int(np.median(total // 2 + 1))
    for ax, c in zip(axes.flat, coalitions):
        s = seats[:, [idx[p] for p in c["parties"] if p in idx]].sum(1)
        colour = _coalition_colour(c, colours or {})
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
                 since: date, today: date | None = None, ncol: int = 2) -> go.Figure:
    """One subplot per party with polls, mean and 50/90% bands."""
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
        fig.add_trace(go.Scatter(x=weeks + weeks[::-1], y=_pp(tr["q95"]) + _pp(tr["q05"])[::-1],
                                 fill="toself", fillcolor=rgba, line=dict(width=0), hoverinfo="skip",
                                 showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(x=weeks + weeks[::-1], y=_pp(tr["q75"]) + _pp(tr["q25"])[::-1],
                                 fill="toself", fillcolor=_rgba(col, 0.35), line=dict(width=0), hoverinfo="skip",
                                 showlegend=False), row=r, col=c)
        obs = tr.filter(pl.col("week") <= today.isoformat())
        fig.add_trace(go.Scatter(x=obs["week"].to_list(), y=_pp(obs["mean"]), mode="lines",
                                 line=dict(color=col, width=2), name=party, showlegend=False,
                                 hovertemplate="%{x}: %{y:.1f}%<extra>" + party + "</extra>"), row=r, col=c)
        fut = tr.filter(pl.col("week") >= today.isoformat())
        fig.add_trace(go.Scatter(x=fut["week"].to_list(), y=_pp(fut["mean"]), mode="lines",
                                 line=dict(color=col, width=2, dash="dot"), showlegend=False,
                                 hovertemplate="%{x}: %{y:.1f}%<extra>forecast</extra>"), row=r, col=c)
        pp = polls.filter((pl.col("party") == party) & (pl.col("mid_date") >= since))
        fig.add_trace(go.Scatter(x=[d.isoformat() for d in pp["mid_date"].to_list()], y=_pp(pp["share"]),
                                 mode="markers", marker=dict(color="black", size=4, opacity=0.5), showlegend=False,
                                 text=[escape(x) for x in pp["pollster"].to_list()],   # from Wikipedia
                                 hovertemplate="%{text}<br>%{x}: %{y:.1f}%<extra></extra>"), row=r, col=c)
        for d in election_dates:
            if d >= since:
                fig.add_vline(x=d.isoformat(), line=dict(color="grey", width=1), row=r, col=c)
    fig.update_yaxes(ticksuffix="%", gridcolor="#eee")
    fig.update_xaxes(gridcolor="#eee")
    fig.update_layout(height=(260 if ncol > 1 else 220) * nrow, margin=dict(l=30, r=10, t=40, b=30),
                      plot_bgcolor="white", paper_bgcolor="white", font=FONT)
    return fig


def seats_figure(seats: np.ndarray, total: np.ndarray, parties: list[str], coalitions: list[dict],
                 ncol: int = 2, colours: dict[str, str] | None = None) -> go.Figure:
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
        colour = _coalition_colour(c, colours or {})
        vals, counts = np.unique(s, return_counts=True)
        fig.add_trace(go.Bar(x=vals, y=counts / counts.sum(), marker_color=colour, opacity=0.7, showlegend=False,
                             hovertemplate="%{x} seats: %{y:.1%}<extra></extra>"), row=r, col=cc)
        fig.add_vline(x=majority - 0.5, line=dict(color="black", width=1), row=r, col=cc)
    fig.update_yaxes(tickformat=".0%", gridcolor="#eee")
    fig.update_annotations(font_size=12)
    fig.update_layout(height=(240 if ncol > 1 else 200) * nrow, margin=dict(l=30, r=10, t=50, b=30),
                      plot_bgcolor="white", paper_bgcolor="white", bargap=0.05, font=FONT)
    return fig


def house_effects_figure(house: pl.DataFrame, colours: dict[str, str]) -> go.Figure:
    import plotly.graph_objects as go

    fig = go.Figure()
    parties = _party_order(house["party"].unique().to_list(), colours)
    houses = house["house"].unique(maintain_order=True).to_list()
    labels = [escape(h) for h in houses]                     # pollster names come from Wikipedia
    for party in parties:
        sub = house.filter(pl.col("party") == party)
        vals = {h: v for h, v in zip(sub["house"].to_list(), sub["effect_pp"].to_list())}
        fig.add_trace(go.Bar(name=party, x=labels, y=[vals.get(h, 0) for h in houses],
                             marker_color=colours.get(party, DEFAULT_COLOUR),
                             hovertemplate="%{x}<br>" + party + ": %{y:+.1f} pp<extra></extra>"))
    fig.update_layout(barmode="group", height=460, margin=dict(l=30, r=10, t=30, b=120), plot_bgcolor="white",
                      paper_bgcolor="white", yaxis=dict(ticksuffix=" pp", gridcolor="#eee", zeroline=True),
                      legend=dict(orientation="h", y=-0.45), font=FONT)
    return fig


# ------------------------------------------------------------------------------- all parties together
def spread_labels(values: list[float], min_sep: float, floor: float | None = None) -> list[float]:
    """Label positions at least ``min_sep`` apart, each group centred on its members' values.

    Labels that would collide are merged into a group and spaced evenly around the group's mean, so a label
    moves only as far as needed; a leader line then joins each label to its line.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    groups = [[i] for i in order]

    def layout(g):
        centre = sum(values[i] for i in g) / len(g)
        start = centre - (len(g) - 1) * min_sep / 2
        if floor is not None:
            start = max(start, floor)
        return [start + k * min_sep for k in range(len(g))]

    merged = True
    while merged:
        merged = False
        out = []
        for g in groups:
            if out and layout(g)[0] - layout(out[-1])[-1] < min_sep - 1e-9:
                out[-1] = out[-1] + g
                merged = True
            else:
                out.append(g)
        groups = out
    pos = [0.0] * len(values)
    for g in groups:
        for i, y in zip(g, layout(g)):
            pos[i] = y
    return pos


def _all_parties_frame(trend: pl.DataFrame, colours: dict[str, str]) -> tuple[list[str], dict]:
    """Parties ordered by election-day support (Other last) and their series."""
    last_week = trend["week"].max()
    final = {r["party"]: r["mean"] for r in trend.filter(pl.col("week") == last_week).to_dicts()}
    parties = sorted((p for p in final if p != "Other"), key=lambda p: -final[p]) + (["Other"] if "Other" in final else [])
    series = {p: trend.filter(pl.col("party") == p).sort("week") for p in parties}
    return parties, series


def all_parties_figure(trend: pl.DataFrame, election_dates: list[date], colours: dict[str, str], since: date,
                       full_since: date, today: date | None = None, labels: bool = True) -> go.Figure:
    """Every party's estimated support on one axis, with a crosshair tooltip, range and interval toggles."""
    import plotly.graph_objects as go

    today = today or date.today()
    parties, series = _all_parties_frame(trend, colours)
    end_week = trend["week"].max()
    fig = go.Figure()
    band_idx = []
    for p in parties:
        tr = series[p]
        weeks = tr["week"].to_list()
        col = colours.get(p, DEFAULT_COLOUR)
        band_idx.append(len(fig.data))
        fig.add_trace(go.Scatter(x=weeks + weeks[::-1], y=_pp(tr["q95"]) + _pp(tr["q05"])[::-1],
                                 fill="toself", fillcolor=_rgba(col, 0.10), line=dict(width=0), hoverinfo="skip",
                                 showlegend=False, legendgroup=p, visible=False, name=f"{p} 90% interval"))
        obs = tr.filter(pl.col("week") <= today.isoformat())
        fig.add_trace(go.Scatter(x=obs["week"].to_list(), y=_pp(obs["mean"]), mode="lines", name=p,
                                 legendgroup=p, line=dict(color=col, width=2, shape="spline", smoothing=0.3),
                                 hovertemplate="%{y:.1f}%<extra>" + p + "</extra>"))
        fut = tr.filter(pl.col("week") >= obs["week"].max()) if obs.height else tr
        fig.add_trace(go.Scatter(x=fut["week"].to_list(), y=_pp(fut["mean"]), mode="lines", showlegend=False,
                                 legendgroup=p, line=dict(color=col, width=2, dash="dot"),
                                 hovertemplate="%{y:.1f}% (projection)<extra>" + p + "</extra>"))
    for d in election_dates:
        if d >= full_since:
            fig.add_vline(x=d.isoformat(), line=dict(color=RULE, width=1))

    def yrange(start: date) -> list[float]:
        window = trend.filter(pl.col("week") >= start.isoformat())
        return [0, float(np.ceil(window["q95"].max() * 100 / 5) * 5 + 2)]

    term_range, full_range = yrange(since), yrange(full_since)
    # buttons above the plot, legend below the axis: they never meet, whatever the width
    height, top, bottom = (520, 44, 84) if labels else (500, 44, 124)
    annotations = []
    if labels:
        values = [float(series[p]["mean"][-1] * 100) for p in parties]
        min_sep = 15 * (term_range[1] - term_range[0]) / (height - top - bottom)
        ys = spread_labels(values, min_sep, floor=min_sep / 2)
        for p, v, y in zip(parties, values, ys):
            annotations.append(dict(x=end_week, y=v, xref="x", yref="y", ax=16, ay=y, axref="pixel", ayref="y",
                                    text=f"{p} {v:.1f}%", showarrow=True, arrowhead=0, arrowwidth=1,
                                    arrowcolor=MUTED, xanchor="left", font=dict(color=INK, size=12)))
    fig.update_layout(
        height=height, margin=dict(l=40, r=150 if labels else 12, t=top, b=bottom),
        plot_bgcolor="white", paper_bgcolor="white", font=FONT, hovermode="x unified",
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK)),
        legend=dict(orientation="h", x=0, xanchor="left", y=-0.08, yanchor="top", font=dict(color=INK),
                    itemclick="toggle", itemdoubleclick="toggleothers"),
        xaxis=dict(range=[since.isoformat(), end_week], showgrid=False, showspikes=True, spikemode="across",
                   spikesnap="cursor", spikethickness=1, spikecolor=MUTED, spikedash="solid", linecolor=GRID,
                   tickfont=dict(color=MUTED)),
        yaxis=dict(range=term_range, ticksuffix="%", gridcolor=GRID, gridwidth=1, zeroline=False,
                   tickfont=dict(color=MUTED)),
        annotations=annotations,
        updatemenus=[
            dict(type="buttons", direction="right", x=0, xanchor="left", y=1.02, yanchor="bottom", showactive=True,
                 pad=dict(r=6, t=0), bgcolor="white", bordercolor=GRID, font=dict(color=INK, size=12),
                 buttons=[dict(label="This term", method="relayout",
                               args=[{"xaxis.range": [since.isoformat(), end_week], "yaxis.range": term_range}]),
                          dict(label=f"Since {full_since.year}", method="relayout",
                               args=[{"xaxis.range": [full_since.isoformat(), end_week], "yaxis.range": full_range}])]),
            dict(type="buttons", direction="right", x=1, xanchor="right", y=1.02, yanchor="bottom", showactive=False,
                 bgcolor="white", bordercolor=GRID, font=dict(color=INK, size=12),
                 buttons=[dict(label="90% intervals", method="restyle", args=[{"visible": True}, band_idx],
                               args2=[{"visible": False}, band_idx])]),
        ],
    )
    return fig


def all_parties_svg(trend: pl.DataFrame, election_dates: list[date], colours: dict[str, str], path: Path,
                    width: float, height: float, since: date, today: date | None = None, labels: bool = True) -> Path:
    """Static version for embedding: every party this term, end labels (wide) or a legend (narrow)."""
    today = today or date.today()
    parties, series = _all_parties_frame(trend, colours)
    fig, ax = plt.subplots(figsize=(width, height))
    end = date.fromisoformat(trend["week"].max())
    for p in parties:
        tr = series[p].filter(pl.col("week") >= since.isoformat())
        weeks = [date.fromisoformat(w) for w in tr["week"].to_list()]
        y = tr["mean"].to_numpy() * 100
        seen = [w <= today for w in weeks]
        last = max(i for i, s_ in enumerate(seen) if s_) if any(seen) else 0
        col = colours.get(p, DEFAULT_COLOUR)
        ax.plot(weeks[: last + 1], y[: last + 1], color=col, lw=1.5, solid_capstyle="round", solid_joinstyle="round",
                label=p)
        ax.plot(weeks[last:], y[last:], color=col, lw=1.5, ls=(0, (1, 1.6)), dash_capstyle="round")
    for d in election_dates:
        if since <= d <= end:
            ax.axvline(d, color=RULE, lw=0.6, zorder=0)
    top = np.ceil(max(series[p].filter(pl.col("week") >= since.isoformat())["mean"].max() for p in parties) * 100 / 5) * 5 + 2
    ax.set_ylim(0, top)
    ax.set_xlim(since, end)
    if labels:
        values = [float(series[p]["mean"][-1] * 100) for p in parties]
        ax_h_pt = ax.get_window_extent().height * 72 / fig.dpi
        min_sep = 11 * top / ax_h_pt
        for p, v, yl in zip(parties, values, spread_labels(values, min_sep, floor=min_sep / 2)):
            ax.annotate(f"{p} {v:.1f}%", xy=(end, v), xytext=(10, yl), textcoords=("offset points", "data"),
                        va="center", ha="left", fontsize=7.5, color=INK, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=0))
        fig.subplots_adjust(right=0.78)
    else:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=7,
                  labelcolor=INK, handlelength=1.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(labelsize=7, colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if not labels:
        fig.tight_layout()
    else:
        fig.subplots_adjust(left=0.08, bottom=0.1, top=0.97)
    fig.savefig(path, format="svg", dpi=100)
    plt.close(fig)
    return path


def _pp(shares: pl.Series) -> list[float]:
    """Shares as percentages rounded to 0.01 points: finer than any chart shows, and far less page weight."""
    return (shares * 100).round(2).to_list()


def _rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
