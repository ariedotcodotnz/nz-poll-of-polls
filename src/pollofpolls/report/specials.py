"""The website's "Special votes" page: the official-count forecast of the close seats (output/specials/).

Like `Site`, every method returns Markdown or a Plotly figure, and text from data files (candidate names, party
labels) is escaped before it reaches Quarto. Without output/specials/forecast.json the page says how to make it.
"""

from __future__ import annotations

import csv
import json
import re
from functools import cached_property
from pathlib import Path

import numpy as np

from ..config import Config
from . import charts
from .site import PLOTLY_CONFIG, esc, long_date, num, prob, swatch, table

DOWNLOADS = ["forecast.json", "party_votes.csv", "routes.csv", "seats.csv", "candidates.csv", "history.csv",
             "party_vote_history.csv", "backtest.csv", "backtest_summary.csv", "party_vote_backtest.csv"]
OUTCOME_ORDER = ("government", "neither", "opposition")
YEAR_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross"]


def download_files(out: Path) -> list[str]:
    """The files in output/specials/ the page links to: the fixed list, and each other contest's history and
    backtest (history_<pair>.csv, backtest_<pair>.csv)."""
    extra = sorted(f.name for pattern in ("history_*.csv", "backtest_*.csv") for f in out.glob(pattern)
                   if not f.name.startswith("backtest_summary"))
    return [f for f in DOWNLOADS if (out / f).exists()] + [f for f in extra if f not in DOWNLOADS]


def display_name(name: str) -> str:
    """ "McANULTY, Kieran" -> "Kieran McAnulty"; "WYLIE-VAN EERD, Ben" -> "Ben Wylie-Van Eerd"."""
    surname, _, given = name.partition(",")

    def fix(word: str) -> str:
        if word.isupper():
            return word.capitalize()
        if word.startswith("Mc") and word[2:].isupper():
            return "Mc" + word[2:].capitalize()
        return word

    surname = re.sub(r"[^\W\d_]+", lambda m: fix(m.group(0)), surname.strip())
    return f"{given.strip()} {surname}".strip()


def votes(x: float) -> str:
    return f"{abs(x):,.0f}"


def signed(x: float, fmt: str = "+.1f") -> str:
    """A signed number with a typographic minus: "+1.4", "−0.6"."""
    return format(x, fmt).replace("-", "−")


class SpecialsPage:
    def __init__(self, root: Path | None = None):
        self.cfg = Config(root)
        self.out = self.cfg.paths.output / "specials"
        self.colours = self.cfg.colours
        self._plotly_loaded = False

    # ------------------------------------------------------------------------------------------- data
    @cached_property
    def data(self) -> dict | None:
        path = self.out / "forecast.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    @property
    def available(self) -> bool:
        return self.data is not None

    @property
    def pair(self) -> list[str]:
        return self.data["pair"]

    @property
    def seats(self) -> list[dict]:
        return self.data["seats"]

    @cached_property
    def contests(self) -> list[ContestPage]:
        """The close seats between other pairs of parties, each a page of its own data."""
        return [ContestPage(self, c) for c in (self.data or {}).get("contests") or []]

    def _csv(self, name: str) -> list[dict]:
        path = self.out / name
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @cached_property
    def history(self) -> list[dict]:
        return self._csv("history.csv")

    @cached_property
    def margins(self) -> dict[str, np.ndarray]:
        with np.load(self.out / "margins.npz", allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def colour(self, party: str) -> str:
        alias = ((self.data or {}).get("house") or {}).get("colours", {}).get(party, party)
        return self.colours.get(party) or self.colours.get(alias, charts.DEFAULT_COLOUR)

    def margin_text(self, m: float) -> str:
        """A margin (pair[1] minus pair[0]) in words: "Labour by 508"."""
        if round(m) == 0:
            return "tied"
        return f"{esc(self.pair[1] if m > 0 else self.pair[0])} by {votes(m)}"

    def contender(self, seat: dict, party: str) -> str:
        return esc(display_name(seat["contenders"][party]))

    # ------------------------------------------------------------------------------------------ text
    def missing(self) -> str:
        return ("The special-vote forecast has not been made yet: run `pollofpolls specials`, then rebuild the "
                "site.\n")

    def intro(self) -> str:
        if not self.available:
            return self.missing()
        d, nat = self.data, self.data["national"]
        names = [esc(s["name"]) for s in self.seats]
        seats = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]
        due = long_date(d["official_results_due"]) if d.get("official_results_due") else "in about three weeks"
        return (f"Election night's count left {seats} separated by fewer than "
                f"{votes(max(abs(s['preliminary']['margin']) for s in self.seats) + 1)} votes. The Electoral "
                f"Commission estimates that about {votes(nat['estimate'])} special votes are still to be counted, "
                f"out of {votes(nat['total_votes_estimate'])} cast. They and a recheck of every ordinary vote make "
                f"the official result, due on {due}. This page forecasts that count, from the preliminary results "
                f"by voting place and from how special votes behaved in these seats at the "
                f"{', '.join(map(str, d['data']['years'][:-1]))} and {d['data']['years'][-1]} elections. "
                f"Updated {long_date(d['generated_at'][:10])}.\n")

    def tiles(self) -> str:
        if not self.available:
            return ""
        out = []
        for s in self.seats:
            fav = max(self.pair, key=lambda p: s["p_win"][p])
            night = self.margin_text(s["preliminary"]["margin"])
            out.append(f"::: {{.tile}}\n[{esc(s['name'])}]{{.tile-label}}\n\n"
                       f"[{prob(s['p_win'][fav])}]{{.tile-value}}\n\n"
                       f"[{self.contender(s, fav)} ({esc(fav)}) to win]{{.tile-label}}\n\n"
                       f"[Election night: {night}]{{.tile-label}}\n:::")
        return ":::: {.tiles}\n" + "\n\n".join(out) + "\n::::\n"

    def seat_table(self) -> str:
        if not self.available:
            return ""
        a, b = self.pair
        rows = [[f"**{esc(s['name'])}**", self.margin_text(s["preliminary"]["margin"]),
                 f"{swatch(self.colour(b))}**{prob(s['p_win'][b])}**",
                 f"{swatch(self.colour(a))}**{prob(s['p_win'][a])}**", prob(s["margin"]["p_close"])]
                for s in self.seats]
        return table(["Electorate", "Election night", f"{esc(b)} wins", f"{esc(a)} wins",
                      f"Within {self.data['close_margin']} votes"], rows)

    def margin_table(self) -> str:
        """The margin chart as numbers: median and the middle 50% and 90% of simulated official counts."""
        if not self.available:
            return ""
        rows = []
        for s in self.seats:
            m = s["margin"]
            rows.append([f"**{esc(s['name'])}**", f"**{self.margin_text(m['q50'])}**",
                         f"{self.margin_text(m['q25'])} to {self.margin_text(m['q75'])}",
                         f"{self.margin_text(m['q05'])} to {self.margin_text(m['q95'])}"])
        return table(["Electorate", "Final margin (median)", "Middle 50%", "Middle 90%"], rows)

    def needs_table(self) -> str:
        if not self.available:
            return ""
        rows = []
        for s in self.seats:
            t, sp = s["trailing"], s["specials"]
            q = t["forecast_share"]
            rows.append([f"**{esc(s['name'])}**", esc(display_name(t["candidate"])),
                         f"{votes(sp['q50'])} [({votes(sp['q05'])}–{votes(sp['q95'])})]{{.muted}}",
                         f"{num(t['ordinary_share'] * 100, 1)}%", f"**{num(t['needs_share'] * 100, 1)}%**",
                         f"{num(q['q50'] * 100, 1)}% [({num(q['q05'] * 100, 0)}–{num(q['q95'] * 100, 0)}%)]{{.muted}}"])
        return table(["Electorate", "Trailing candidate", "Special votes (90%)", "Had", "Needs",
                      "Forecast (90%)"], rows)

    def scenario_table(self) -> str:
        if not self.available:
            return ""
        sc, years = self.data["scenarios"], self.data["data"]["years"]
        b = self.pair[1]
        header = ["Electorate", "This forecast"] + [f"Like {y}" for y in years] + ["No enrolment adjustment"]
        rows = [[esc(s["name"]), f"**{prob(sc['model'][s['name']])}**"]
                + [prob(sc[f"like_{y}"][s["name"]]) for y in years] + [prob(sc["no_enrolment_adjustment"][s["name"]])]
                for s in self.seats]
        return (f"The chance that {esc(b)}'s candidate wins each seat, if the special votes across the country shift "
                f"from the ordinary vote as they did at a past election:\n\n" + table(header, rows))

    def joint_text(self) -> str:
        if not self.available:
            return ""
        j = self.data["joint"]
        n = len(self.seats)
        dist = [(int(k), p) for k, p in j["wins"].items()]
        cells = [[f"{k} of {n}", prob(p)] for k, p in dist]
        expected = sum(k * p for k, p in dist)
        return (f"The seats are not independent: the same national shift in special votes moves all of them. "
                f"{esc(j['party'])} is expected to win {num(expected, 1)} of these {n} seats.\n\n"
                + table([f"Seats {esc(j['party'])} wins", "Chance"], [[c[0], c[1]] for c in cells]))

    def seat_tabs(self) -> str:
        """Every candidate's forecast, one tab per seat (no tabs for a single seat)."""
        if not self.available:
            return ""
        tabs = len(self.seats) > 1
        parts = ["::: {.panel-tabset}\n"] if tabs else []
        for s in self.seats:
            rows = []
            for c in sorted(s["candidates"], key=lambda c: -c["final"]["mean"]):
                f = c["final"]
                rows.append([f"{swatch(self.colours.get(c['party']))}{esc(display_name(c['candidate']))}",
                             esc(c["party"]), f"{c['preliminary']:,}", f"{c['specials_mean']:,.0f}",
                             f"**{f['q50']:,.0f}**", f"{f['q05']:,.0f}–{f['q95']:,.0f}"])
            p, oc = s["preliminary"], s["official_change"]
            split = (f" ({p['advance']:,} in advance, {p['election_day']:,} on election day)"
                     if p.get("advance") is not None else "")
            parts.append((f"## {esc(s['name'])}\n\n" if tabs else "")
                         + table(["Candidate", "Party", "Election night", "Special votes (mean)",
                                  "Official result (median)", "90% interval"], rows)
                         + f"\n{p['votes']:,} valid candidate votes on election night{split}. Expected special votes "
                           f"{votes(s['specials']['q50'])}, of which about {votes(s['specials']['overseas_mean'])} "
                           f"from overseas. The official count's change to the ordinary votes moves the margin by "
                           f"{votes(oc['margin']['q05'])} towards {esc(self.pair[0])} to "
                           f"{votes(oc['margin']['q95'])} towards {esc(self.pair[1])} (90%), and in half the "
                           f"simulations by fewer than "
                           f"{votes(max(abs(oc['margin']['q25']), abs(oc['margin']['q75'])) + 1)} votes. "
                           f"Other candidates' special votes are shared in proportion to their election-night "
                           f"votes.\n")
        return "\n".join(parts) + ("\n:::\n" if tabs else "")

    def history_table(self) -> str:
        rows = []
        for r in sorted(self.history, key=lambda r: (r["electorate"], int(r["year"]))):
            rows.append([esc(r["electorate"]), r["year"], f"{int(r['special_votes']):,}",
                         num(float(r["specials_per_ordinary"]) * 100, 1),
                         f"{num(float(r['ordinary_share']) * 100, 1)}%",
                         f"{num(float(r['special_share']) * 100, 1)}%", f"**{signed(float(r['special_shift_pp']))}**",
                         num(float(r["overseas_share"]) * 100, 0) + "%"])
        if not rows:
            return ""
        b = esc(self.pair[1]) if self.available else "Labour"
        return table(["Electorate", "Election", "Special votes", "Per 100 ordinary votes",
                      f"{b} share, ordinary votes", f"{b} share, special votes", "Shift (points)",
                      "From overseas"], rows)

    def official_count_table(self) -> str:
        rows = [r for r in self.history if r.get("preliminary_margin")]
        if not rows:
            return ""
        body = [[f"{esc(r['electorate'])} {r['year']}", self.margin_text(int(r["preliminary_margin"])),
                 self.margin_text(int(r["ordinary_margin"])), signed(int(r["official_ordinary_change"]), "+,"),
                 signed(int(r["official_margin_change"]), "+,")]
                for r in sorted(rows, key=lambda r: (int(r["year"]), r["electorate"]))]
        return table(["Seat", "Election night", "Official ordinary votes", "Ordinary votes added",
                      f"Margin change (towards {esc(self.pair[1]) if self.available else 'Labour'})"], body)

    def backtest_table(self) -> str:
        bt = (self.data or {}).get("backtest")
        if not bt:
            return "No backtest results have been generated yet.\n"
        rows = [[f"**{esc(r['label'])}**" if r["method"] == "model" else esc(r["label"]), num(r["crps_votes"]),
                 num(r["mae_votes"]), num(r["coverage_50"], 2), num(r["coverage_90"], 2), num(r["brier"], 3),
                 str(r["n_cases"])] for r in bt]
        return table(["Method", "CRPS (votes)", "Mean abs. error (votes)", "50% coverage", "90% coverage",
                      "Brier (winner)", "Seats"], rows)

    def facts(self) -> dict[str, str]:
        """Numbers and names quoted inline in the page's text. Quarto prints inline values as literal text, so they
        are not Markdown-escaped."""
        if not self.available:
            return {}
        d, nat, post = self.data, self.data["national"], self.data["posterior"]
        years = d["data"]["years"]
        shift = d["election_shift_pp"]["forecast_domestic"]
        bt = {r["method"]: r for r in d.get("backtest") or []}
        oc = post["official_count"]
        enrol = nat["enrolment_shift_pp"]
        f = {"n_cases": str(d["data"]["n_seat_elections"]),
             "years": ", ".join(map(str, years[:-1])) + f" and {years[-1]}",
             "n_prelim": str(d["data"]["n_election_night_counts"]),
             "estimate": votes(nat["estimate"]), "final_specials": votes(round(nat["special_votes"]["q50"], -3)),
             "ratio": num(nat["change_per_ordinary"]["q50"] * 100, 0),
             "removed": num(nat["removed_share"]["q50"] * 100, 0),
             "enrol_pp": num(abs(enrol["q50"]), 1), "enrol_lo": signed(enrol["q05"]), "enrol_hi": signed(enrol["q95"]),
             "shift_med": signed(shift["q50"]), "shift_lo": signed(shift["q05"]), "shift_hi": signed(shift["q95"]),
             "sd_year": num(post["sd_election_pp"]["q50"], 1),
             "jump_p": num(oc["jump_prob"]["q50"] * 100, 0),
             "jump_size": num(oc["jump_median_share"]["q50"] * 100, 1),
             "n_sims": f"{d['n_sims']:,}", "party": self.pair[1], "other_party": self.pair[0],
             "past": "; ".join(f"{y}: {signed(post['years'][str(y)]['domestic_pp']['q50'])}" for y in years)}
        # specials per ordinary vote by seat, averaged over past elections
        ratio: dict[str, list[float]] = {}
        for r in self.history:
            ratio.setdefault(r["electorate"], []).append(float(r["specials_per_ordinary"]))
        if len(ratio) >= 2:
            ranked = sorted(ratio, key=lambda k: np.mean(ratio[k]))
            f |= {"most_specials": ranked[-1], "fewest_specials": ranked[0],
                  "most_ratio": num(np.mean(ratio[ranked[-1]]) * 100, 0),
                  "fewest_ratio": num(np.mean(ratio[ranked[0]]) * 100, 0)}
        oc_rows = [r for r in self.history if r.get("preliminary_margin")]
        if oc_rows:
            def added(r: dict) -> float:     # votes the official count added, as a share of election night's
                change = int(r["official_ordinary_change"])
                return change / (int(r["ordinary_votes"]) - change)

            big = sorted((r for r in oc_rows if added(r) > 0.01), key=added, reverse=True)
            small = sum(abs(int(r["official_margin_change"])) < 25 for r in oc_rows)
            f |= {"oc_small": f"{small} of {len(oc_rows)}",
                  "oc_big": " and ".join(f"{num(added(r) * 100, 1)}% in {r['electorate']} {r['year']}" for r in big)}
        if "model" in bt:
            best = min((r for r in bt.values() if r["method"] != "model"), key=lambda r: r["crps_votes"])
            m = bt["model"]
            f |= {"bt_crps": num(m["crps_votes"]), "bt_best": best["label"].lower(),
                  "bt_best_crps": num(best["crps_votes"]), "bt_n": str(m["n_cases"]),
                  "bt_cov90": num(m["coverage_90"] * 100, 0), "bt_cov50": num(m["coverage_50"] * 100, 0)}
        return f | self.house_facts()

    @cached_property
    def _config(self) -> dict:
        import yaml
        path = self.cfg.paths.config / "specials.yml"
        return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}

    def backtest_text(self) -> str:
        """What the backtest shows, in words that follow from its numbers."""
        bt = {r["method"]: r for r in (self.data or {}).get("backtest") or []}
        if "model" not in bt:
            return ""
        model, rules = bt["model"], [r for r in bt.values() if r["method"] != "model"]
        best = min(rules, key=lambda r: r["crps_votes"])
        better = model["crps_votes"] < best["crps_votes"]
        parts = [f"The model's CRPS was {num(model['crps_votes'])} votes, {'lower than' if better else 'against'} "
                 f"{num(best['crps_votes'])} for the best rule of thumb ({esc(best['label'].lower())}). Its 50% "
                 f"and 90% intervals held {num(model['coverage_50'] * 100, 0)}% and "
                 f"{num(model['coverage_90'] * 100, 0)}% of the {model['n_cases']} results."]
        closer = [r for r in rules if r["mae_votes"] < model["mae_votes"]]
        if closer:
            r = min(closer, key=lambda r: r["mae_votes"])
            parts.append(f"Its median missed by {num(model['mae_votes'])} votes on average, more than "
                         f"{esc(r['label'].lower())} ({num(r['mae_votes'])}): a fixed rule is hard to beat in a year "
                         f"whose special votes behave like that rule, and the model keeps a wide range because no one "
                         f"knows in advance which year this is.")
        if all(r["n_flipped"] == 0 for r in bt.values()):
            parts.append("No seat in the backtest changed hands after election night, so the Brier score rewards the "
                         "most confident method.")
        parts.append(f"With {model['n_cases']} seat-elections the comparison is a sanity check, not proof.")
        return " ".join(parts) + "\n"

    def notes(self) -> str:
        """Notes on the seats from config/specials.yml, such as boundary changes."""
        notes = [e.get("note") for e in self._config.get("electorates", []) if e.get("note")]
        return " ".join(esc(n) for n in notes) + "\n" if notes else ""

    def official_count_text(self) -> str:
        """The paragraph on the official count's change to the ordinary votes, from the archived counts."""
        if not self.available:
            return ""
        f = self.facts()
        seen = (f"Election-night counts archived for {f['n_prelim']} of the past seat-elections show both: "
                f"{f['oc_small']} moved the margin by fewer than 25 votes" if "oc_small" in f else "")
        if seen and f.get("oc_big"):
            seen += f", but the official count added {esc(f['oc_big'])} more ordinary votes than election night"
        return ("**The ordinary votes in the official count.** The official count rechecks every ordinary vote, and "
                "sometimes adds votes that were missing on election night. " + (seen + ". " if seen else "")
                + f"The model gives each seat a {f['jump_p']}% chance of such an addition (median {f['jump_size']}% of "
                "its votes), split a little differently from the election-night count, plus recheck noise on every "
                "candidate.\n")

    def downloads(self) -> str:
        files = download_files(self.out)
        if not files:
            return ""
        return "- Special votes: " + " · ".join(f"[{esc(f)}](specials/{f})" for f in files) + "\n"

    # ------------------------------------------------------------------------------------- the House
    @property
    def house(self) -> dict | None:
        return (self.data or {}).get("house")

    def outcome_label(self, key: str, short: bool = False) -> str:
        h = self.house
        if key == "neither":
            if h.get("two_sided", True):       # the two sides hold every seat, so "neither" can only be a tie
                return "Exact tie" if short else "Neither side has a majority (an exact tie)"
            return "No majority" if short else "Neither side has a majority"
        label = esc(h[key]["label"])
        return label if short else f"{label} have a majority"

    def house_intro(self) -> str:
        h = self.house
        if not h:
            return ""
        gov, opp = h["government"], h["opposition"]
        seats = {s["party"]: s["election_night"] for s in h["seats"]}
        g, o = sum(seats.get(p, 0) for p in gov["parties"]), sum(seats.get(p, 0) for p in opp["parties"])
        size, nat = h["size_election_night"], self.data["national"]
        return (f"On election night {esc(gov['label'])} had {g} of {size} seats, {g - size // 2 - 1} more than "
                f"the {size // 2 + 1} a majority needs, and {esc(opp['label'])} had {o}. About "
                f"{votes(nat['estimate'])} special votes are still to be counted, {votes(nat['overseas_estimate'])} "
                f"of them from overseas. They change each party's share of the party vote, which sets how many "
                f"seats it is entitled to, and they decide the closest electorates. This section simulates both "
                f"together and counts how often each side ends with a majority.\n")

    def house_tiles(self) -> str:
        h = self.house
        if not h:
            return ""
        out = []
        for key in OUTCOME_ORDER:
            out.append(f"::: {{.tile}}\n[{self.outcome_label(key, short=True)}]{{.tile-label}}\n\n"
                       f"[{prob(h['outcome'][key])}]{{.tile-value}}\n\n"
                       f"[{'keep a majority' if key == 'government' else 'hung Parliament' if key == 'neither' else 'have a majority'}]"
                       f"{{.tile-label}}\n:::")
        return ":::: {.tiles}\n" + "\n\n".join(out) + "\n::::\n"

    def house_caveat(self) -> str:
        h = self.house
        if not h:
            return ""
        lost = h["outcome"]["neither"] + h["outcome"]["opposition"]
        return (f"The chance that {esc(h['government']['label'])} lose their majority is {prob(lost)}: "
                f"{'a tie' if h.get('two_sided', True) else 'no majority for either side'} or a "
                f"majority for the other parties. A majority for {esc(h['opposition']['label'])} is not a Labour-led "
                f"government unless all four agree to one, and Opportunity has said it could work with either "
                f"Labour or National.\n")

    def _seat_cell(self, dist: dict[str, float], base: int) -> str:
        items = sorted(((int(k), p) for k, p in dist.items() if p >= 0.03), key=lambda kp: -kp[1])[:5]
        return " · ".join((f"**{k}** ({prob(p)})" if k != base else f"{k} ({prob(p)})") for k, p in items)

    def party_vote_table(self) -> str:
        h = self.house
        if not h:
            return ""
        rows = []
        for r in h["party_vote"]:
            f, c = r["final_share"], r["change_pp"]
            rows.append([f"{swatch(self.colour(r['party']))}**{esc(r['party'])}**",
                         f"{num(r['election_night_share'] * 100, 2)}%",
                         f"{num(r['special_share']['q50'] * 100, 1)}%",
                         f"**{num(f['q50'] * 100, 2)}%** [({num(f['q05'] * 100, 1)}–{num(f['q95'] * 100, 1)})]{{.muted}}",
                         signed(c["q50"], "+.2f")])
        return table(["Party", "Election night", "Of special votes", "Official count (90%)", "Change (points)"], rows)

    def house_seats_table(self) -> str:
        h = self.house
        if not h:
            return ""
        rows = [[f"{swatch(self.colour(s['party']))}**{esc(s['party'])}**", str(s["election_night"]),
                 self._seat_cell(s["seats"], s["election_night"])] for s in h["seats"]]
        return table(["Party", "Election night", "Official count: seats (chance)"], rows)

    def routes_table(self) -> str:
        h = self.house
        if not h:
            return ""
        result = {"government": "Majority kept", "neither": self.outcome_label("neither", short=True),
                  "opposition": "Majority lost"}
        rows = [["; ".join(esc(x) for x in r["steps"]), signed(r["margin"], "+.0f") if r["margin"] else "0",
                 result[r["outcome"]], f"**{num(r['p'] * 100, 1)}%**"] for r in h["routes"]["top"]]
        return table(["What changes from election night", "Margin", "Result", "Chance"], rows)

    def minimal_routes_table(self) -> str:
        h = self.house
        if not h or not h["routes"]["minimal"]:
            return ""
        r = h["routes"]
        rows = [["; ".join(esc(x) for x in m["steps"]),
                 f"**{num(m['p'] * 100, 1)}%**" if m["p"] >= 0.0005 else "under 0.1%"] for m in r["minimal"]]
        rows.append(["**Any of these exactly**", f"**{num(r['p_minimal'] * 100, 1)}%**"])
        return table(["Route", "Chance"], rows)

    def routes_text(self) -> str:
        h = self.house
        if not h:
            return ""
        r, o = h["routes"], h["outcome"]
        if not r.get("minimal"):                # a party sits with neither side: a margin is not a majority
            return ""
        weights = {c["kind"]: c["weight"] for c in r["components"]}
        return (f"The Government's margin over every other party was {r['margin_election_night']} seats on election "
                f"night. A list seat lost by a governing party that has list seats moves the margin by "
                f"{weights.get('list', 2)}, because another party gains the seat. National holds more electorates "
                f"than its party vote entitles it to, so its changes count once: when its entitlement falls it "
                f"keeps its electorates and another party gains a seat, and when it loses an electorate nobody "
                f"gains one. A change of {r['seats_of_margin_needed']} gives the other parties a one-seat majority; "
                f"{r['seats_of_margin_needed'] - 1} is a tie. There are {len(r['minimal'])} ways to make exactly "
                f"{r['seats_of_margin_needed']}:\n\n" + self.minimal_routes_table()
                + f"\nA count can also reach that one-seat majority with offsetting changes (National gaining an "
                f"electorate while losing more elsewhere, or Te Pāti Māori losing one), which takes the chance of "
                f"exactly a one-seat majority to {prob(r['p_margin_exactly_needed'])}. A bigger majority for the "
                f"other parties has a chance of {prob(r['p_margin_more_than_needed'])}. Together these are the "
                f"{prob(o['opposition'])} above, so the routes that go further are already counted in it.\n")

    def other_seats_table(self) -> str:
        h = self.house
        if not h or not h.get("other_seats"):
            return ""
        a, b = self.pair
        rows = [[f"**{esc(s['name'])}**", f"{esc(s['leader'])} ({esc(s['leader_party'])})",
                 self.margin_text(s["election_night_margin"]),
                 f"{self.margin_text(s['margin']['q05'])} to {self.margin_text(s['margin']['q95'])}",
                 f"**{prob(s['p_flip'])}**"]
                for s in sorted(h["other_seats"], key=lambda s: -s["p_flip"])]
        text = (f"Every other seat where {esc(a)}'s and {esc(b)}'s candidates are first and second within "
                f"{h['modelled_margin']:,} votes is simulated the same way, from its election-night candidate "
                f"votes, without voting-place detail or the seat's own history:\n\n"
                + table(["Electorate", "Leader", "Election night", "Official margin (90%)", "Chance it changes hands"],
                        rows))
        if h.get("assumed_seats"):
            items = "; ".join(f"{esc(a['electorate'])} ({prob(float(a['p_flip']))} that {esc(a['challenger'])} "
                              f"overtakes {esc(a['leader_party'])})" for a in h["assumed_seats"])
            text += (f"\nThree close seats are not contests between those two parties, so the model cannot simulate "
                     f"them. Each has an assumed chance of changing hands, set in the configuration: {items}. "
                     f"The other {h['n_electorates'] - len(h['other_seats']) - len(self.seats) - len(h['assumed_seats']) - sum(len(c.seats) for c in self.contests)} "
                     f"electorates are taken as won by their election-night leaders.")
            for c in self.contests:
                names = " and ".join(esc(x["name"]) for x in c.seats)
                text += (f" {names}, between {esc(c.pair[0])} and {esc(c.pair[1])}, "
                         f"{'has' if len(c.seats) == 1 else 'have'} a forecast of "
                         f"{'its' if len(c.seats) == 1 else 'their'} own [below](#{c.anchor}).")
            text += "\n"
        return text

    def house_scenario_table(self) -> str:
        h = self.house
        if not h:
            return ""
        sc, years = h["scenarios"], self.data["data"]["years"]
        names = [("model", "This forecast")] + [(f"like_{y}", f"Special votes lean as in {y}") for y in years] \
            + [("no_enrolment_adjustment", "No enrolment adjustment")]
        rows = [[f"**{label}**" if key == "model" else label] + [prob(sc[key][o]) for o in OUTCOME_ORDER]
                for key, label in names if key in sc]
        return table(["Scenario", "Government keeps its majority", self.outcome_label("neither", short=True),
                      "Other parties have a majority"], rows)

    def house_history_table(self) -> str:
        """Past elections: each party's share of ordinary and of special votes."""
        rows = self._csv("party_vote_history.csv")
        h = self.house
        if not rows or not h:
            return ""
        parties = [p for p in h["parties"] if p != "Other"]
        years = sorted({r["year"] for r in rows})
        by = {(r["year"], r["party"]): r for r in rows}
        body = []
        for y in years:
            cells = [y]
            for p in parties:
                r = by.get((y, p))
                cells.append(signed(100 * (float(r["special_share"]) - float(r["ordinary_share"]))) if r else "–")
            body.append(cells)
        return table(["Election"] + [esc(p) for p in parties], body)

    def house_backtest_text(self) -> str:
        bt = (self.house or {}).get("backtest")
        if not bt:
            return ""
        mae, labels = bt["mae_pp"], bt["labels"]
        text = (f"**The party vote.** Each of the {len(bt['years'])} elections from {bt['years'][0]} to "
                f"{bt['years'][-1]} was forecast with the model refitted on the others. The model's median missed a "
                f"party's final share by {num(mae['model'], 2)} points on average, against "
                f"{num(mae['election_night'], 2)} for assuming {esc(labels['election_night'].lower())} and "
                f"{num(mae['average'], 2)} for the {esc(labels['average'].lower())}. Its 50% and 90% intervals held "
                f"{num(bt['coverage_50'] * 100, 0)}% and {num(bt['coverage_90'] * 100, 0)}% of the "
                f"{bt['n_cases']} results.")
        s = bt.get("seats")
        if s:
            text += (f" Given the electorates each party won, the model's most likely seat count was right for "
                     f"{num(s['model_right'] * 100, 0)}% of parties, against {num(s['election_night_right'] * 100, 0)}% "
                     f"for the election-night count; {s['n_changed']} of {s['n_cases']} seat counts changed.")
            if s.get("mean_p_actual_when_changed") is not None:
                text += (f" When a party's seats did change, the model had given the final number a "
                         f"{num(s['mean_p_actual_when_changed'] * 100, 0)}% chance on average.")
        return text + "\n"

    def house_facts(self) -> dict[str, str]:
        h = self.house
        if not h:
            return {}
        m, oc = h["model"], h["ordinary_change"]
        sp = h["special_party_votes"]
        return {"h_years": f"{h['years'][0]} to {h['years'][-1]}", "h_n_years": str(len(h["years"])),
                "h_common_sd": num(m["common_sd"]["q50"], 2),
                "h_valid": votes(round(sp["valid"]["q50"], -3)), "h_valid_lo": votes(round(sp["valid"]["q05"], -3)),
                "h_valid_hi": votes(round(sp["valid"]["q95"], -3)),
                "h_added": num(oc["growth_mean"] * 100, 1), "h_added_n": str(oc["n"]),
                "h_link": num(self.data["posterior"].get("link_slope", {}).get("q50", float("nan")), 1),
                "h_gov": h["government"]["label"], "h_opp": h["opposition"]["label"],
                "h_modelled": str(len(h.get("other_seats", [])) + len(self.seats))}

    def house_figure(self, narrow: bool = False):
        """The distribution of the Government's margin over every other party, coloured by who has a majority."""
        import plotly.graph_objects as go

        h = self.house
        gov, opp = h["government"], h["opposition"]
        dist = sorted((int(k), p * 100) for k, p in h["margin"].items())
        dist = [(k, p) for k, p in dist if p >= 0.05]
        # each margin's simulations by what they actually gave: a negative margin is a majority for the other
        # side only when no party sits outside the two sides
        split = h.get("margin_outcome") or {str(k): {"government": p / 100 if k > 0 else 0.0,
                                                     "neither": p / 100 if k == 0 else 0.0,
                                                     "opposition": p / 100 if k < 0 else 0.0} for k, p in dist}
        part = lambda key: [(k, split[str(k)][key] * 100) for k, _ in dist if split[str(k)][key] > 0]  # noqa: E731
        groups = [("government", part("government"), self.colour(gov["parties"][0]), f"{gov['label']} majority"),
                  ("neither", part("neither"), charts.MUTED, self.outcome_label("neither", short=True)),
                  ("opposition", part("opposition"), self.colour(opp["parties"][0]),
                   "Majority for the other parties" if narrow and h.get("two_sided", True)
                   else f"{opp['label']} majority")]
        fig = go.Figure()
        for _key, pts, colour, name in groups:
            if not pts:
                continue
            fig.add_trace(go.Bar(
                x=[k for k, _ in pts], y=[p for _, p in pts], name=name, width=0.84,
                marker=dict(color=colour, line=dict(width=0)),
                text=[f"{p:.0f}%" if p >= 1 else "" for _, p in pts], textposition="outside", cliponaxis=False,
                textfont=dict(size=11, color=charts.INK),
                hovertemplate=[(f"Government {k:+d} seats against the rest" if k else "Seats tied")
                               + f": {p:.1f}% of simulations<extra></extra>" for k, p in pts]))
        night = h["margin_election_night"]
        top = max(p for _, p in dist)
        fig.add_annotation(x=night, y=top * 1.2, text="election night", showarrow=True, arrowhead=0, ax=0,
                           ay=-18, yanchor="bottom", font=dict(size=11, color=charts.MUTED),
                           arrowcolor=charts.MUTED)
        fig.update_xaxes(tickmode="linear", dtick=1 if not narrow else 2, tickformat="+d", zeroline=False,
                         showgrid=False, title_text="Government seats minus every other party's")
        fig.update_yaxes(range=[0, top * 1.35], ticksuffix="%", gridcolor=charts.GRID, zeroline=False, nticks=5)
        fig.update_layout(height=340 if not narrow else 320, margin=dict(l=40, r=12, t=20, b=50), bargap=0.16,
                          barmode="stack",
                          plot_bgcolor="white", paper_bgcolor="white", font=charts.FONT, dragmode=False,
                          legend=dict(orientation="h", y=-0.28 if not narrow else -0.4, x=0, traceorder="normal"))
        return charts._no_zoom(fig)

    # --------------------------------------------------------------------------------------- figures
    def figure_html(self, name: str, narrow: bool = False) -> str:
        """A figure as an HTML fragment; the first one on the page also loads Plotly."""
        import plotly.offline

        html = self.figure(name, narrow).to_html(full_html=False, include_plotlyjs=False, include_mathjax=False,
                                                 config=PLOTLY_CONFIG)
        if not self._plotly_loaded:
            src = f"https://cdn.plot.ly/plotly-{plotly.offline.get_plotlyjs_version()}.min.js"
            html = f'<script src="{src}" charset="utf-8"></script>\n{html}'
            self._plotly_loaded = True
        return html

    def show(self, name: str, narrow: bool = False) -> None:
        from IPython.display import HTML, display

        if not self.available or (name == "house" and not self.house):
            return
        display(HTML(self.figure_html(name, narrow)))

    def figure(self, name: str, narrow: bool = False):
        return {"margins": self.margin_figure, "needs": self.needs_figure, "house": self.house_figure}[name](narrow)

    def margin_figure(self, narrow: bool = False):
        """The distribution of each seat's final margin: one row per seat, coloured by who is ahead."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        a, b = self.pair
        m = self.margins["margin"].astype(float)
        lo, hi = np.quantile(m, 0.003), np.quantile(m, 0.997)
        span = hi - lo
        width = next(w for w in (10, 20, 25, 50, 100, 200, 250, 500) if span / w <= (40 if narrow else 70))
        lo, hi = np.floor(lo / width) * width, np.ceil(hi / width) * width
        n_seats = len(self.seats)
        titles = [f"{s['name']}: {b} {prob(s['p_win'][b])}, {a} {prob(s['p_win'][a])}" for s in self.seats]
        fig = make_subplots(rows=n_seats, cols=1, shared_xaxes=True, subplot_titles=titles,
                            vertical_spacing=0.32 / n_seats)
        top = 0.0
        for j, s in enumerate(self.seats):
            x = m[:, j]
            k = np.where(x > 0, np.ceil(x / width), np.floor(x / width)).astype(int)   # (0, w] is bin 1, [-w, 0) is -1
            k = k[(x != 0) & (x >= lo) & (x <= hi)]
            bins, counts = np.unique(k, return_counts=True)
            share = counts / len(x) * 100
            top = max(top, share.max())
            for party, keep in ((a, bins < 0), (b, bins > 0)):
                kk = np.abs(bins[keep])                   # bin k covers a lead of (k - 1) * width + 1 to k * width
                labels = [f"{party} by {(k - 1) * width + 1:,.0f}–{k * width:,.0f}" for k in kk]
                centre = np.sign(bins[keep]) * (kk - 0.5) * width
                fig.add_trace(go.Bar(x=centre, y=share[keep], width=width * 0.86,
                                     marker=dict(color=self.colour(party), line=dict(width=0)), name=party,
                                     customdata=labels, showlegend=False,
                                     hovertemplate="%{customdata}: %{y:.1f}% of simulations<extra></extra>"),
                              row=j + 1, col=1)
            night = s["preliminary"]["margin"]
            fig.add_vline(x=night, line=dict(color=charts.MUTED, width=1), row=j + 1, col=1)
            fig.add_vline(x=0, line=dict(color=charts.INK, width=1), row=j + 1, col=1)
            ref = "" if j == 0 else str(j + 1)
            fig.add_annotation(x=night, y=1, xref=f"x{ref}", yref=f"y{ref} domain", text="election night",
                               showarrow=False, xanchor="left" if night >= 0 else "right", yanchor="top",
                               xshift=4 if night >= 0 else -4, font=dict(size=11, color=charts.MUTED))
        for j in range(n_seats):
            fig.update_yaxes(range=[0, top * 1.12], ticksuffix="%", gridcolor=charts.GRID, zeroline=False,
                             nticks=3, row=j + 1, col=1)
            fig.update_xaxes(range=[lo, hi], gridcolor=charts.GRID, zeroline=False, tickformat=",",
                             row=j + 1, col=1)
        title = f"← {a} · margin (votes) · {b} →" if narrow else f"Final margin in votes: ← {a} ahead · {b} ahead →"
        fig.update_xaxes(title_text=title, row=n_seats, col=1)
        fig.update_annotations(selector=dict(xref="paper"), font=dict(size=13, color=charts.INK), xanchor="left", x=0)
        fig.update_layout(height=(150 if narrow else 165) * n_seats + 60, margin=dict(l=40, r=12, t=40, b=50),
                          plot_bgcolor="white", paper_bgcolor="white", font=charts.FONT, bargap=0, dragmode=False)
        return charts._no_zoom(fig)

    def needs_figure(self, narrow: bool = False):
        """Past special-vote shifts in each seat, the forecast shift, and the shift that would flip the seat."""
        import plotly.graph_objects as go

        a, b = self.pair
        seats = self.seats
        names = [s["name"] for s in seats]
        ys = {n: i for i, n in enumerate(reversed(names))}
        past = {s.get("history", s["name"]): ys[s["name"]] for s in seats}     # the seat whose history is shown
        fig = go.Figure()
        xs = []
        years = sorted({int(r["year"]) for r in self.history})
        for k, y in enumerate(years):
            rows = [r for r in self.history if int(r["year"]) == y and r["electorate"] in past]
            x = [float(r["special_shift_pp"]) for r in rows]
            xs += x
            fig.add_trace(go.Scatter(
                x=x, y=[past[r["electorate"]] + 0.12 + 0.09 * k for r in rows], mode="markers", name=f"{y} election",
                marker=dict(symbol=YEAR_SYMBOLS[k % len(YEAR_SYMBOLS)], size=10, color=charts.MUTED,
                            line=dict(color="white", width=2)),
                hovertemplate=[f"{r['electorate']} {y}: specials {float(r['special_shift_pp']):+.1f} points "
                               f"more {b} than ordinary votes<extra></extra>" for r in rows]))
        for i, s in enumerate(seats):
            y0 = ys[s["name"]]
            base = s["preliminary"]["contender_share"]
            q = {k: (v - base) * 100 for k, v in s["special_share"].items()}
            t = s["trailing"]
            need_b = t["needs_share"] if t["party"] == b else 1 - t["needs_share"]
            need = (need_b - base) * 100
            xs += [q["q05"], q["q95"], need]
            colour = self.colour(t["party"])
            fig.add_shape(type="rect", x0=need, x1=need + (60 if need > 0 else -60), y0=y0 - 0.38, y1=y0 + 0.42,
                          fillcolor=charts._rgba(colour, 0.10), line=dict(width=0), layer="below")
            fig.add_shape(type="line", x0=need, x1=need, y0=y0 - 0.38, y1=y0 + 0.42, line=dict(color=colour, width=2))
            who = display_name(t["candidate"])
            label = (f"{who.split()[-1]} wins →" if need > 0 else f"← {who.split()[-1]} wins") if narrow \
                else f"{who} wins beyond here"
            fig.add_annotation(x=need, y=y0 + 0.42, text=label,
                               showarrow=False, yanchor="bottom", xanchor="left" if need > 0 else "right",
                               font=dict(size=11, color=charts.INK))
            show = i == 0
            fig.add_trace(go.Scatter(x=[q["q05"], q["q95"]], y=[y0 - 0.12] * 2, mode="lines", name="2026 forecast, 90%",
                                     line=dict(color=charts.INK, width=2), showlegend=show, legendgroup="f90",
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[q["q25"], q["q75"]], y=[y0 - 0.12] * 2, mode="lines", name="2026 forecast, 50%",
                                     line=dict(color=charts.INK, width=7), showlegend=show, legendgroup="f50",
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=[q["q50"]], y=[y0 - 0.12], mode="markers", name="2026 forecast, median", showlegend=False,
                marker=dict(size=11, color="white", line=dict(color=charts.INK, width=2)),
                hovertemplate=(f"{s['name']} 2026: specials {q['q50']:+.1f} points more {b} than election night "
                               f"(50%: {q['q25']:+.1f} to {q['q75']:+.1f}; 90%: {q['q05']:+.1f} to {q['q95']:+.1f}). "
                               f"{display_name(t['candidate'])} needs {need:+.1f}<extra></extra>")))
        lim = max(abs(np.nanmin(xs)), abs(np.nanmax(xs))) + 2
        fig.add_vline(x=0, line=dict(color=charts.RULE, width=1))
        fig.update_yaxes(tickvals=list(ys.values()), ticktext=list(ys), range=[-0.6, len(names) - 0.3],
                         showgrid=False, zeroline=False)
        fig.update_xaxes(range=[-lim, lim], gridcolor=charts.GRID, zeroline=False, ticksuffix="",
                         title_text=f"Shift towards {b} (points)" if narrow else
                         f"Special votes' shift towards {b}, in points of the two leaders' share")
        height = 120 * len(names) + 130
        legend = dict(orientation="h", y=-0.22 if not narrow else -0.3, x=0)
        if len(names) < 3:                       # a short chart: keep the legend clear of the axis title
            height += 40
            legend |= {"y": -(70 if not narrow else 80) / (height - 80), "yanchor": "top"}
        fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=50), plot_bgcolor="white",
                          paper_bgcolor="white", font=charts.FONT, dragmode=False,
                          legend=legend)
        return charts._no_zoom(fig)


class ContestPage(SpecialsPage):
    """The seats of a contest between another pair of parties (forecast.json's ``contests``): the same tables
    and figures as the main page, from the contest's own seats, history and simulated margins."""

    def __init__(self, parent: SpecialsPage, contest: dict):
        self.parent, self.contest = parent, contest
        self.cfg, self.out, self.colours = parent.cfg, parent.out, parent.colours

    @property
    def _plotly_loaded(self) -> bool:
        return self.parent._plotly_loaded

    @_plotly_loaded.setter
    def _plotly_loaded(self, value: bool) -> None:
        self.parent._plotly_loaded = value

    @cached_property
    def data(self) -> dict:
        return {**self.parent.data, "backtest": None, **self.contest, "contests": []}

    @property
    def slug(self) -> str:
        return self.contest["slug"]

    @property
    def anchor(self) -> str:
        return "contest-" + self.slug.replace("_", "-")

    @cached_property
    def history(self) -> list[dict]:
        return self._csv(f"history_{self.slug}.csv")

    @cached_property
    def margins(self) -> dict[str, np.ndarray]:
        with np.load(self.out / f"margins_{self.slug}.npz", allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def intro(self) -> str:
        d, (a, b) = self.data, self.pair
        parts = []
        for s in self.seats:
            lead = s["leader_election_night"]
            trail = b if lead == a else a
            parts.append(f"On election night {self.contender(s, lead)} ({esc(lead)}) led "
                         f"{self.contender(s, trail)} ({esc(trail)}) by {votes(s['preliminary']['margin'])} votes in "
                         f"{esc(s['name'])}, of {s['preliminary']['votes']:,} counted.")
        seats = d["data"]["electorates"]
        years = d["data"]["years"]
        text = " ".join(parts) + (
            f" The model above is fitted to {esc(self.parent.pair[0])} against {esc(self.parent.pair[1])}, so this "
            f"contest has its own: the same model, fitted to the official results of "
            f"{', '.join(map(esc, seats[:-1]))} and {esc(seats[-1])} at the "
            f"{', '.join(map(str, years[:-1]))} and {years[-1]} elections ({d['data']['n_seat_elections']} "
            f"seat-elections), with the {esc(a)} candidate in place of {esc(self.parent.pair[0])}'s. It shares each "
            f"simulation's national special-vote count and party-vote lean with the rest of the page.")
        notes = " ".join(esc(s["note"]) for s in self.seats if s.get("note"))
        for s in self.seats:                     # what the seat's past results, on its boundaries, came to
            rows = [r for r in self.history if r["electorate"] == s.get("history") != s["name"]]
            if rows and "notional" in s["history"]:
                last = max(rows, key=lambda r: int(r["year"]))
                notes += (f" On those boundaries the {last['year']} result would have been "
                          f"{self.margin_text(int(last['final_margin']))}.")
        if self.house:
            text += (f" The winner does not change the House: the {esc(a)} and {esc(b)} seat totals are set by "
                     f"their party votes, so the seat decides who sits, not how many.")
        return text + (" " + notes if notes else "") + "\n"

    def method_text(self) -> str:
        d = self.data
        post, shift, enrol = d["posterior"], d["election_shift_pp"]["forecast_domestic"], d["enrolment_shift_pp"]
        a, b = self.pair
        past = "; ".join(f"{y}: {signed(post['years'][str(y)]['domestic_pp']['q50'])}" for y in d["data"]["years"])
        moves = ("does not move this contest" if abs(enrol["q50"]) < 0.05
                 else f"moves this contest by {signed(enrol['q50'])} points")
        text = (f"Fitted to those past results, domestic special votes moved towards the {esc(b)} candidate, against "
                f"the {esc(a)} candidate, by {past} points in an evenly split seat. This year's shift is forecast at "
                f"{signed(shift['q50'])} points (90%: {signed(shift['q05'])} to {signed(shift['q95'])}). The "
                f"enrolment change is assumed to have removed voters who leaned to both parties alike, so on "
                f"average it {moves}, but it could go either way "
                f"(90%: {signed(enrol['q05'])} to {signed(enrol['q95'])} points). The official count's change to the "
                f"ordinary votes is the one learned from the {esc(self.parent.pair[0])}–{esc(self.parent.pair[1])} "
                f"seats.")
        bt = {r["method"]: r for r in d.get("backtest") or []}
        if "model" in bt and len(bt) > 1:
            m = bt["model"]
            best = min((r for r in bt.values() if r["method"] != "model"), key=lambda r: r["crps_votes"])
            text += (f" In a backtest that refits the model without each election in turn, its CRPS was "
                     f"{num(m['crps_votes'])} votes of final margin, against {num(best['crps_votes'])} for the best "
                     f"rule of thumb ({esc(best['label'].lower())}); its 90% intervals held "
                     f"{num(m['coverage_90'] * 100, 0)}% of the {m['n_cases']} results.")
        return text + "\n"

    def _figure_block(self, name: str) -> str:
        return "\n".join(f"::: {{.{cls}}}\n```{{=html}}\n{self.figure_html(name, narrow)}\n```\n:::\n"
                          for cls, narrow in (("wide-only", False), ("narrow-only", True)))

    def section(self) -> str:
        """The contest's whole section of the page, as Markdown with its figures embedded."""
        a, b = self.pair
        one = len(self.seats) == 1
        where = f" in {esc(self.seats[0]['history'])}" if one and self.seats[0].get("history") else ""
        parts = [f"## {esc(self.contest['title'])} {{#{self.anchor}}}\n", self.intro(), self.tiles(),
                 "### Who wins\n", self.seat_table(),
                 "### How the final margin could land\n",
                 (f"Each bar is the share of {self.data['n_sims']:,} simulated official counts that ended with that "
                  f"margin: the {esc(b)} candidate's votes minus the {esc(a)} candidate's. The dark line is a tie; "
                  f"the grey line is the election-night margin.\n"),
                 self._figure_block("margins"), self.margin_table(),
                 "### What the trailing candidate needs\n", self.needs_table(),
                 (f"\nThe grey shapes are how much more {esc(b)} the special votes were than the ordinary votes at "
                  f"past elections{where}. The black bar is this year's forecast, and the tinted zone is where the "
                  f"trailing candidate overtakes.\n"),
                 self._figure_block("needs"),
                 "### Under a past election's pattern\n", self.scenario_table(),
                 "### Every candidate\n" if one else "### Each seat\n", self.seat_tabs(),
                 "### How this forecast was made\n", self.method_text(),
                 f"\nSpecial votes in the {esc(a)}–{esc(b)} seats since {self.data['data']['years'][0]}:\n",
                 self.history_table()]
        if not one:
            parts.insert(-4, "### The seats together\n" + "\n" + self.joint_text())
        return "\n".join(parts) + "\n"
