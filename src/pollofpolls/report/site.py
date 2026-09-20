"""What the website's pages compute.

The pages in website/ are Quarto documents. Their Python cells make one `Site`, which loads the pipeline's outputs
and returns Markdown (tables, tiles, lists) and Plotly figures. Text that came from Wikipedia, such as pollster
names, is escaped here: Quarto reads computed Markdown as Markdown, raw HTML and shortcodes, so a hostile edit
could otherwise put script into the page or read environment variables into it.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, timedelta
from functools import cached_property
from pathlib import Path

import numpy as np
import polars as pl

from ..config import Config
from ..forecast.coalitions import balance_of_power
from . import charts

REPO_URL = "https://github.com/ariedotcodotnz/nz-poll-of-polls"
DOWNLOADS = ["summary.json", "forecast.csv", "seats.csv", "coalitions.csv", "trend.csv", "house_effects.csv"]
CHARTS = ["voting_intention620.svg", "voting_intention375.svg", "voting_intention_all620.svg",
          "voting_intention_all375.svg", "election_night620.svg", "election_night375.svg", "saturday620.svg",
          "saturday375.svg"]
BACKTEST_DOWNLOADS = ["summary.csv", "summary_by_horizon.csv", "head_to_head.csv", "scores.csv", "stacking.json",
                      "probe_fundamentals.json"]
LABELS = {"legacy": "2023 model (replica)", "base": "Dirichlet-multinomial", "gauss": "Gaussian",
          "heavy": "Gaussian, heavy-tailed shocks", "fund": "Gaussian + fundamentals prior", "kal": "Kalman",
          "ensemble": "Stacked ensemble (out of sample)", "equal": "Equal-weight ensemble",
          "calibrated": "Stacked ensemble + spread calibration (out of sample)"}
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

_PUNCT = re.compile(r"([!-/:-@\[-`{-~])")


def esc(text: object) -> str:
    """Text as literal Markdown: ASCII punctuation backslash-escaped, whitespace collapsed to single spaces."""
    return _PUNCT.sub(r"\\\1", " ".join(str(text).split()))


def pct(x: float, digits: int = 0) -> str:
    return f"{x * 100:.{digits}f}%"


def prob(x: float) -> str:
    """A probability, never printed as 0% or 100%.

    These come from simulations, so a probability below the resolution of the simulation is "not seen", not
    "impossible": rare events are exactly where the model is least trustworthy, and saying 0% claims otherwise.
    """
    if x < 0.005:
        return "<1%"
    if x > 0.995:
        return ">99%"
    return f"{x * 100:.0f}%"


def num(x: float | None, digits: int = 0) -> str:
    return "–" if x is None or math.isnan(x) else f"{x:.{digits}f}"


def and_join(items: list) -> str:
    """"a", "a and b", "a, b and c"."""
    items = [str(i) for i in items]
    return " and ".join(items) if len(items) < 3 else ", ".join(items[:-1]) + " and " + items[-1]


def long_date(d: date | str) -> str:
    d = date.fromisoformat(d) if isinstance(d, str) else d
    return f"{d.day} {d:%B %Y}"


def table(header: list[str], rows: list[list[str]], align: str = "") -> str:
    """A Markdown pipe table from cells that are already Markdown, in a div that scrolls on narrow screens.
    ``align`` has one letter per column (l, r or c); by default the first column is left-aligned, the rest right."""
    align = align or "l" + "r" * (len(header) - 1)
    rule = {"l": ":---", "r": "---:", "c": ":---:"}
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(rule[a] for a in align) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "::: {.table-scroll}\n" + "\n".join(lines) + "\n:::\n"


def swatch(colour: str | None) -> str:
    """A colour chip before a party name. It is decorative: the name next to it carries the identity."""
    colour = colour if colour and re.fullmatch(r"#[0-9A-Fa-f]{6}", colour) else charts.DEFAULT_COLOUR
    return f'[]{{.swatch style="background-color:{colour}" aria-hidden="true"}}'


class Site:
    """The pipeline's outputs as the pieces the pages show. Data is loaded when a page first asks for it."""

    def __init__(self, root: Path | None = None, today: date | None = None):
        self.cfg = Config(root)
        self.out = self.cfg.paths.output
        self.election = self.cfg.forecast_election
        self.colours = self.cfg.colours
        self.today = today or date.today()
        self._plotly_loaded = False

    # ------------------------------------------------------------------------------------------------- data
    @cached_property
    def summary(self) -> dict:
        path = self.out / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run `pollofpolls forecast` before building the website")
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def parties(self) -> list[str]:
        return list(self.summary["parties"])

    @cached_property
    def trend(self) -> pl.DataFrame:
        return pl.read_csv(self.out / "trend.csv")

    @cached_property
    def polls(self) -> pl.DataFrame:
        polls = pl.read_parquet(self.cfg.paths.processed / "polls.parquet")
        return polls.filter(pl.col("party").is_in(self.parties))

    @cached_property
    def sims(self) -> dict[str, np.ndarray]:
        with np.load(self.out / "seat_sims.npz", allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    @cached_property
    def house(self) -> pl.DataFrame | None:
        path = self.out / "house_effects.csv"
        if not path.exists() or path.stat().st_size <= 30:          # header only: no pollsters this term
            return None
        house = pl.read_csv(path)
        return house if house.height else None

    @property
    def election_dates(self) -> list[date]:
        return [e.date for e in self.cfg.window_elections]

    @property
    def anchor_date(self) -> date:
        return self.cfg.election(self.cfg.anchor_election).date

    @property
    def last_result_year(self) -> int:
        return max(e.year for e in self.cfg.window_elections if not e.forecast)

    @property
    def term_start(self) -> date:
        """Where "this term" charts start: shortly before the previous election."""
        previous = [e.date for e in self.cfg.elections if e.date < self.election.date]
        return max(previous) - timedelta(days=60) if previous else self.anchor_date

    @property
    def n_simulations(self) -> str:
        return f"{len(self.sims['total_election']):,}"

    @property
    def days_to_election(self) -> int:
        return (self.election.date - self.today).days

    def balance_rows(self, when: str = "election_day") -> list[dict]:
        """Balance-of-power rows from summary.json. A summary written before the analysis existed, or before it
        split out ``p_needs_several``, is brought up to date from the seat simulations instead."""
        rows = (self.summary.get("balance_of_power") or {}).get(when)
        if rows and all("p_needs_several" in r for r in rows):
            return rows
        bop = self.cfg.electorates_cfg.get("balance_of_power", {})
        key = {"election_day": "election", "now": "now"}[when]
        return balance_of_power(self.sims[f"seats_{key}"], self.parties, self.sims[f"total_{key}"],
                                bop.get("blocs", {}), bop.get("pivots", []))

    # ---------------------------------------------------------------------------------------- forecast page
    def intro(self) -> str:
        s = self.summary
        latest = s.get("latest_poll")
        if not s.get("n_polls_cycle") or latest in (None, "", "None"):     # "None": written before 2026-09-19
            polls = "No polls have been published this term yet."
        else:
            polls = (f"{s['n_polls_cycle']} polls this term from {len(s['pollsters_cycle'])} pollsters; the latest "
                     f"was taken {long_date(latest)}.")
        return f"{self.days_to_election} days to go. {polls} Updated {long_date(s['generated_at'][:10])}.\n"

    def tiles(self) -> str:
        """One tile per coalition: its chance of a majority on election day, and its seats."""
        tiles = [f"::: {{.tile}}\n[{prob(c['p_majority'])}]{{.tile-value}}\n\n"
                 f"[{esc(c['name'])} majority]{{.tile-label}}\n\n"
                 f"[{num(c['seats_mean'])} seats ({num(c['seats_q05'])}–{num(c['seats_q95'])})]{{.tile-label}}\n:::"
                 for c in self.summary["coalitions_election_day"]]
        return ":::: {.tiles}\n" + "\n\n".join(tiles) + "\n::::\n"

    def balance_table(self, when: str = "election_day") -> str:
        rows = self.balance_rows(when)
        if not rows:
            return ""
        pivots = rows[0]["pivots"]
        header = ["Bloc", "Majority alone"] + [f"Enough with {esc(p)}" for p in pivots]
        if len(pivots) > 1:
            header.append(f"Needs {esc(pivots[0])} and {esc(pivots[1])}" if len(pivots) == 2 else
                          "Needs two or more of " + ", ".join(esc(p) for p in pivots))
        header.append("Short even then")
        body = []
        for r in rows:
            cells = [f"{esc(r['bloc'])} [({esc(', '.join(r['parties']))})]{{.muted}}", prob(r["p_alone"])]
            cells += [prob(r["p_with"][p]) for p in pivots]
            if len(pivots) > 1:
                cells.append(prob(r["p_needs_several"]))
            body.append(cells + [prob(r["p_short"])])
        return table(header, body)

    def house_size(self) -> str:
        s = self.summary
        return (f"Expected House size {num(s['expected_house_size'], 1)} seats "
                f"(chance of an overhang {prob(s['p_overhang'])}).\n")

    def party_table(self) -> str:
        s = self.summary
        seats = {r["party"]: r for r in s["seats_election_day"]}
        rows = []
        for p in self.parties:
            e, n, st = s["election_day"][p], s["now"][p], seats[p]
            rows.append([f"{swatch(self.colours.get(p))}{esc(p)}", pct(n["mean"], 1), f"**{pct(e['mean'], 1)}**",
                         f"{num(e['q05'] * 100, 1)}–{pct(e['q95'], 1)}", prob(e["p_over_5pct"]),
                         num(st["seats_mean"], 1), f"{num(st['seats_q05'])}–{num(st['seats_q95'])}"])
        return table(["Party", "Now", "Election day", "90% interval", "P(≥ 5%)", "Seats (mean)", "Seats 90%"], rows)

    def coalition_table(self) -> str:
        now = {c["name"]: c for c in self.summary["coalitions_now"]}
        rows = [[esc(c["name"]), prob(now[c["name"]]["p_majority"]), num(now[c["name"]]["seats_mean"]),
                 f"**{prob(c['p_majority'])}**", f"{num(c['seats_mean'])} ({num(c['seats_q05'])}–{num(c['seats_q95'])})"]
                for c in self.summary["coalitions_election_day"]]
        return table(["Coalition", "Now: majority", "Now: seats", "Election day: majority", "Election day: seats"],
                     rows)

    def recent_polls_table(self, n: int = 25) -> str:
        polls = self.polls.filter(pl.col("cycle") == self.election.year)
        if polls.is_empty():
            return "No polls have been published this term yet.\n"
        wide = (polls.pivot(on="party", index=["mid_date", "pollster", "sample_size", "date_text"], values="share",
                            aggregate_function="first")
                .sort("mid_date", descending=True).head(n))
        cols = [p for p in self.parties if p in wide.columns]
        rows = [[esc(r["date_text"]), esc(r["pollster"]), num(r["sample_size"])]
                + [num(r[p] * 100, 1) if r[p] is not None else "–" for p in cols] for r in wide.to_dicts()]
        return table(["Field dates", "Pollster", "n"] + [esc(p) for p in cols], rows)

    def downloads(self) -> str:
        """The files published beside the page, as a Markdown list."""
        def links(files: list[str], prefix: str = "") -> str:
            return " · ".join(f"[{esc(f)}]({prefix}{f})" for f in files)
        items = [("Forecast", [f for f in DOWNLOADS if (self.out / f).exists()], ""),
                 ("Charts", [f for f in CHARTS if (self.out / f).exists()], ""),
                 ("Backtests", [f for f in BACKTEST_DOWNLOADS if (self.out / "backtest" / f).exists()], "backtest/")]
        return "".join(f"- {label}: {links(files, prefix)}\n" for label, files, prefix in items if files)

    def diagnostics(self) -> str:
        parts = [f"{esc(v)} (max R̂ {num(d.get('max_rhat'), 3)}, min ESS {num(d.get('min_ess_bulk'))}, "
                 f"divergences {d.get('n_divergent', '–')})" for v, d in self.summary.get("diagnostics", {}).items()]
        return "Model diagnostics: " + "; ".join(parts) + ".\n" if parts else ""

    def wider_error_parties(self) -> str:
        """The parties whose election-day error is widened because they are small or have never been elected."""
        scale = self.summary.get("error_scale") or {}
        return and_join([p for p, x in scale.items() if x > 1]) or "no party"

    def electorate_sweep_risk(self, party: str) -> tuple[str, str]:
        """Chance the party holds none of its electorates: as simulated, and if each seat were an independent coin."""
        from ..forecast.simulate import simulate_electorates

        k = self.parties.index(party)
        seats = [e for e in self.cfg.electorates_cfg["electorates"] if e["party"] == party]
        shared = float(self.cfg.electorates_cfg.get("electorate_group_sd", 0.0))
        out = []
        for group_sd in (shared, 0.0):
            won = simulate_electorates(self.sims["pi_election"], self.parties, seats,
                                       np.random.default_rng(11), group_sd=group_sd)[0][:, k]
            out.append(prob(float((won == 0).mean())))
        return out[0], out[1]

    # -------------------------------------------------------------------------------------------- figures
    def show(self, name: str, narrow: bool = False) -> None:
        """Display a page figure (see `figure_html`)."""
        from IPython.display import HTML, display

        display(HTML(self.figure_html(name, narrow)))

    def figure_html(self, name: str, narrow: bool = False) -> str:
        """A page figure as HTML, without the toolbar and resizing with the page.

        Plotly's notebook renderer would load plotly.js (through require.js) and MathJax again for every figure.
        Instead the first figure of a page loads plotly.js, at the version this Plotly package writes for. That
        state belongs to the Site, which each run of a page makes afresh: Quarto's execution daemon keeps this
        module loaded from one render to the next.
        """
        import plotly.offline

        html = self.figure(name, narrow).to_html(full_html=False, include_plotlyjs=False, include_mathjax=False,
                                                 config=PLOTLY_CONFIG)
        if not self._plotly_loaded:
            src = f"https://cdn.plot.ly/plotly-{plotly.offline.get_plotlyjs_version()}.min.js"
            html = f'<script src="{src}" charset="utf-8"></script>\n{html}'
            self._plotly_loaded = True
        return html

    def figure(self, name: str, narrow: bool = False):
        """A page figure: all, trend, trend_full, seats_election, seats_now or house. ``narrow`` gives the
        one-column version for phones."""
        ncol = 1 if narrow else 2
        polls = self.polls.filter(pl.col("mid_date") > self.anchor_date) if name.startswith("trend") else None
        parties = self.parties
        coalitions = self.cfg.electorates_cfg["coalitions"]
        build = {
            "all": lambda: charts.all_parties_figure(self.trend, self.election_dates, self.colours, self.term_start,
                                                     self.anchor_date, labels=not narrow),
            "trend": lambda: charts.trend_figure(self.trend, polls, self.election_dates, self.colours,
                                                 self.term_start, ncol=ncol),
            "trend_full": lambda: charts.trend_figure(self.trend, polls, self.election_dates, self.colours,
                                                      self.anchor_date, ncol=ncol),
            "seats_election": lambda: charts.seats_figure(self.sims["seats_election"], self.sims["total_election"],
                                                          parties, coalitions, ncol=ncol, colours=self.colours),
            "seats_now": lambda: charts.seats_figure(self.sims["seats_now"], self.sims["total_now"], parties,
                                                     coalitions, ncol=ncol, colours=self.colours),
            "house": lambda: charts.house_effects_figure(self.house, self.colours),
        }
        return build[name]()

    # ------------------------------------------------------------------------------------------- backtests
    @cached_property
    def backtest(self) -> dict | None:
        """The backtest tables, or None when no backtests have been run."""
        bt_dir = self.out / "backtest"
        if not (bt_dir / "summary.csv").exists():
            return None
        order = ["ensemble", "calibrated", "equal", "heavy", "gauss", "base", "fund", "legacy"]
        rank = {v: i for i, v in enumerate(order)}
        summary = sorted(pl.read_csv(bt_dir / "summary.csv").to_dicts(), key=lambda r: rank.get(r["variant"], 99))
        by_h = pl.read_csv(bt_dir / "summary_by_horizon.csv")
        horizons = sorted(by_h["horizon_weeks"].unique().to_list(), reverse=True)
        grid = []
        for r in summary:
            cells = {row["horizon_weeks"]: row for row in by_h.filter(pl.col("variant") == r["variant"]).to_dicts()}
            grid.append((r["variant"], [cells.get(h) for h in horizons]))
        h2h_path, stacking_path = bt_dir / "head_to_head.csv", bt_dir / "stacking.json"
        return {"summary": summary, "horizons": horizons, "grid": grid,
                "targets": sorted(pl.read_csv(bt_dir / "scores.csv")["target"].unique().to_list()),
                "h2h": pl.read_csv(h2h_path).to_dicts() if h2h_path.exists() else [],
                "stacking": json.loads(stacking_path.read_text()) if stacking_path.exists() else {}}

    def backtest_tables(self) -> str:
        """The scores of every model, and CRPS by horizon."""
        bt = self.backtest
        if not bt:
            return "No backtest results have been generated yet.\n"
        header = ["Model", "CRPS (pp)", "Mean abs. error (pp)", "90% coverage", "50% coverage", "Brier (bloc lead)",
                  "Cases"]
        rows = []
        for r in bt["summary"]:
            label = esc(LABELS.get(r["variant"], r["variant"]))
            rows.append([f"**{label}**" if r["variant"] == "ensemble" else label, num(r["crps_pp"], 2),
                         num(r["mae_pp"], 2), num(r["coverage_90"], 2), num(r["coverage_50"], 2),
                         num(r["brier_bloc"], 3), str(r["n_cases"])])
        grid = [[esc(LABELS.get(v, v))] + [f"{num(c['crps_pp'], 2)} [({num(c['coverage_90'], 2)})]{{.muted}}"
                                           if c else "–" for c in cells] for v, cells in bt["grid"]]
        weeks = [f"{h} week{'s' if h != 1 else ''}" for h in bt["horizons"]]
        return (table(header, rows) + "\nCRPS by weeks before the election, with 90% coverage in brackets:\n\n"
                + table(["Model"] + weeks, grid) + "\n" + self.backtest_verdict())

    def backtest_verdict(self) -> str:
        bt = self.backtest
        parts = []
        if bt["h2h"]:
            better = sum(1 for r in bt["h2h"] if r["ensemble_better"])
            parts.append(f"The published ensemble beat the 2023 model on CRPS in {better} of {len(bt['h2h'])} "
                         "backtest cases.")
        if bt["stacking"].get("weights"):
            parts.append(f"Stacking weights for {self.election.year}, fitted on {bt['stacking'].get('n_cases', '–')} "
                         f"backtest cases: {self.weights_text(bt['stacking']['weights'])}.")
        return " ".join(parts) + "\n" if parts else ""

    @staticmethod
    def weights_text(weights: dict[str, float], labels: bool = True) -> str:
        """Non-negligible weights, largest first: "Gaussian 0.60, ..." with labels, else "gauss 0.60, ..."."""
        kept = sorted(((w, v) for v, w in weights.items() if w >= 0.005), reverse=True)
        return ", ".join(f"{esc(LABELS.get(v, v)) if labels else v} {num(w, 2)}" for w, v in kept)

    def backtest_facts(self) -> dict[str, str]:
        """Numbers quoted in the evaluation page's text, as plain text; empty without backtests."""
        bt = self.backtest
        if not bt:
            return {}
        by = {r["variant"]: r for r in bt["summary"]}
        facts = {"n_stacked": str(bt["stacking"].get("n_cases", "–")), "n_compared": str(len(bt["h2h"])),
                 "weights": self.weights_text(bt["stacking"].get("weights", {}), labels=False),
                 "loeo": "; ".join(f"{t}: {self.weights_text(w, labels=False)}"
                                   for t, w in bt["stacking"].get("loeo_weights", {}).items())}
        if "ensemble" in by and "legacy" in by:
            ens, leg = by["ensemble"], by["legacy"]
            worse = [r for r in bt["h2h"] if not r["ensemble_better"]]
            facts |= {"crps_gain": pct(1 - ens["crps_pp"] / leg["crps_pp"]), "n_better": str(len(bt["h2h"]) - len(worse)),
                      "legacy_cov90": pct(leg["coverage_90"]), "ensemble_cov90": pct(ens["coverage_90"]),
                      "legacy_brier": num(leg["brier_bloc"], 3), "ensemble_brier": num(ens["brier_bloc"], 3),
                      "legacy_wins": and_join([f"{r['target']} at {r['horizon_weeks']} week"
                                               f"{'s' if r['horizon_weeks'] != 1 else ''}" for r in worse]) or "none"}
        if "ensemble" in by and "calibrated" in by:
            h = min(bt["horizons"])
            cov = {v: next((c for c in cells if c and c["horizon_weeks"] == h), None) for v, cells in bt["grid"]}
            facts |= {"calibrated_crps": num(by["calibrated"]["crps_pp"], 2), "ensemble_crps": num(by["ensemble"]["crps_pp"], 2),
                      "shortest": f"{h} week{'s' if h != 1 else ''}"}
            if cov.get("ensemble") and cov.get("calibrated"):
                facts |= {"calibrated_cov_short": num(cov["calibrated"]["coverage_90"], 2),
                          "ensemble_cov_short": num(cov["ensemble"]["coverage_90"], 2)}
        return facts

    def design_facts(self) -> dict[str, str]:
        """The backtest design from config/model.yml, for the evaluation page."""
        bt = self.cfg.model_cfg.get("backtest", {})
        mcmc, targets, horizons = bt.get("mcmc", {}), bt.get("targets", []), bt.get("horizons_weeks", [])
        return {"targets": and_join(targets), "horizons": and_join(horizons),
                "n_cases": str(len(targets) * len(horizons)), "chains": str(mcmc.get("chains", "–")),
                "warmup": str(mcmc.get("warmup", "–")), "draws": str(mcmc.get("samples", "–"))}

    def prior_facts(self) -> dict[str, str]:
        """The fundamentals prior estimated for the forecast election, in percentage points."""
        prior = self.summary.get("fundamentals_prior") or {}
        if not prior.get("n_elections"):
            return {}
        return {"mean": f"{prior['mean'] * 100:.1f}".replace("-", "−"), "sd": num(prior["sd"] * 100, 1),
                "n": str(prior["n_elections"])}

    def probe_facts(self) -> dict[str, str]:
        """The fundamentals-prior probe: one past forecast with and without the prior."""
        path = self.out / "backtest" / "probe_fundamentals.json"
        cases = json.loads(path.read_text()).get("cases", {}) if path.exists() else {}
        if not {"fund", "gauss"} <= set(cases):
            return {}
        with_, without = cases["fund"], cases["gauss"]
        pm = self.cfg.pm_by_year.get(with_["target"], "")
        facts = {"target": str(with_["target"]), "weeks": str(with_["horizon_weeks"]),
                 "crps_with": num(with_["crps_pp"], 2), "crps_without": num(without["crps_pp"], 2), "pm": pm}
        if pm in with_["forecast_mean"]:
            facts |= {"pm_with": pct(with_["forecast_mean"][pm], 1), "pm_without": pct(without["forecast_mean"][pm], 1),
                      "pm_result": pct(with_["outcome"][pm], 1)}
        return facts
