"""Editorial assumptions: who wins which electorate, the coalitions reported, and the balance-of-power blocs.

They live in config/electorates.yml. Before a forecast uses them they are checked against the tracked parties and
the electorate boundaries in config/boundaries.yml, and every problem is reported at once, with what to change:
a misspelt party or electorate gets the nearest match, an electorate abolished before the forecast election gets
the names of the electorates that replaced it, and probabilities that cannot add up are caught. Names match
ignoring case and macrons, so `Te Pati Maori` and `tamaki makaurau` work.

Two things are allowed rather than errors, so one file serves forecasts of different elections: a party the
config knows but this forecast does not track (it holds no seats: its electorates are skipped, and it adds
nothing to a coalition or bloc), and an electorate that only exists at a later election than the one forecast.
Both are reported as notes.
"""

from __future__ import annotations

import difflib
import unicodedata
from dataclasses import dataclass, field

ELECTORATE_KEYS = {"electorate", "party", "p", "independent_p", "at_share", "slope", "group", "note", "source",
                   "updated"}
DEFAULT_KEYS = {"at_share", "slope", "group"}
DEFAULT_SLOPE = 0.3
OTHER = "Other"


class AssumptionError(ValueError):
    """Everything wrong with the editorial assumptions, one problem per line."""


def fold(name: object) -> str:
    """A name for matching: case, macrons and surrounding space ignored."""
    decomposed = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold().strip()


def _match(name: object, canonical: list[str]) -> str | None:
    return {fold(c): c for c in canonical}.get(fold(name))


def _suggest(name: object, canonical: list[str]) -> str:
    folded = {fold(c): c for c in canonical}
    close = difflib.get_close_matches(fold(name), list(folded), n=2, cutoff=0.7)
    return f"; did you mean {' or '.join(folded[c] for c in close)}?" if close else ""


def _join(items: list[str]) -> str:
    return " and ".join(items) if len(items) < 3 else ", ".join(items[:-1]) + " and " + items[-1]


@dataclass
class Boundaries:
    """The electorates in use at one election, and what the review that set them abolished."""
    name: str
    from_election: int
    general: list[str]
    maori: list[str]
    replaced: dict[str, list[str]] = field(default_factory=dict)

    @property
    def electorates(self) -> list[str]:
        return self.general + self.maori


def boundaries_for(boundaries_cfg: dict, year: int) -> Boundaries:
    """The boundary set in force at the election of ``year``: the latest one starting at or before it."""
    sets = [b for b in boundaries_cfg.get("boundary_sets", []) if int(b["from_election"]) <= year]
    if not sets:
        raise AssumptionError(f"config/boundaries.yml has no boundary set in force at the {year} election")
    b = max(sets, key=lambda s: int(s["from_election"]))
    return Boundaries(name=b["name"], from_election=int(b["from_election"]), general=list(b["general"]),
                      maori=list(b.get("maori", [])),
                      replaced={k: list(v) for k, v in (b.get("replaced") or {}).items()})


@dataclass
class Assumptions:
    """The checked assumptions, with names in their canonical spelling and party defaults filled in."""
    boundaries: Boundaries
    electorates: list[dict]
    coalitions: list[dict]
    blocs: dict[str, list[str]]
    pivots: list[str]
    group_sd: float
    notes: list[str] = field(default_factory=list)


def _placed(name: object, boundaries_cfg: dict, year: int) -> tuple[str, str]:
    """Where an electorate name stands relative to the election of ``year``: ("current", canonical),
    ("later", set name), ("abolished", message) or ("unknown", suggestion)."""
    sets = sorted(boundaries_cfg.get("boundary_sets", []), key=lambda b: int(b["from_election"]))
    current = boundaries_for(boundaries_cfg, year)
    found = _match(name, current.electorates)
    if found:
        return "current", found
    for b in sets:
        if int(b["from_election"]) > year and _match(name, list(b["general"]) + list(b.get("maori", []))):
            return "later", b["name"]
    for b in sets:
        if int(b["from_election"]) <= year:
            for old, successors in (b.get("replaced") or {}).items():
                if fold(old) == fold(name):
                    return "abolished", f"abolished at the {b['name']}; its area is now in {_join(successors)}"
    return "unknown", _suggest(name, current.electorates)


def load_assumptions(electorates_cfg: dict, boundaries_cfg: dict, parties: list[str], year: int,
                     known: list[str] | None = None) -> Assumptions:
    """Check config/electorates.yml against the tracked ``parties`` and the boundaries for ``year``.

    ``known`` are party names the configuration uses anywhere (colours, parliaments), so a party this forecast does
    not track is told apart from a misspelling. Raises AssumptionError listing every problem; returns the
    assumptions ready for the seat simulation.
    """
    problems: list[str] = []
    notes: list[str] = []
    bounds = boundaries_for(boundaries_cfg, year)
    named = [p for p in parties if p != OTHER]
    vocabulary = list(dict.fromkeys([*named, *(p for p in known or [] if p != OTHER)]))

    def party(name: object, where: str) -> str | None:
        """The canonical name of a tracked or known party; None, with a problem recorded, for anything else."""
        found = _match(name, vocabulary)
        if found is None:
            problems.append(f"{where}: `{name}` is not a party the model knows (tracked now: {_join(named)})"
                            f"{_suggest(name, vocabulary)}. A candidate outside the tracked parties is an "
                            "`independent_p` on another party's entry for that electorate.")
        return found

    defaults: dict[str, dict] = {}
    for name, values in (electorates_cfg.get("party_defaults") or {}).items():
        where = f"party_defaults: {name}"
        found = party(name, where)
        values = values or {}
        for key in sorted(set(values) - DEFAULT_KEYS):
            problems.append(f"{where}: unknown key `{key}` (defaults can set {_join(sorted(DEFAULT_KEYS))})")
        if found:
            defaults[found] = {k: v for k, v in values.items() if k in DEFAULT_KEYS}
    untracked = set()

    electorates, seen = [], {}
    for i, raw in enumerate(electorates_cfg.get("electorates") or [], start=1):
        if not isinstance(raw, dict) or "electorate" not in raw:
            problems.append(f"electorates, entry {i}: needs at least `electorate`, `party` and `p`")
            continue
        where = str(raw["electorate"])
        for key in sorted(set(raw) - ELECTORATE_KEYS):
            problems.append(f"{where}: unknown key `{key}`{_suggest(key, sorted(ELECTORATE_KEYS))}")
        status, detail = _placed(raw["electorate"], boundaries_cfg, year)
        name = detail if status == "current" else None
        if status == "later":
            notes.append(f"{where}: skipped, it is an electorate from the {detail}, after the {year} election")
            continue
        if status == "abolished":
            problems.append(f"{where}: {detail}. Use the name of the electorate the assumption is about.")
        elif status == "unknown":
            problems.append(f"{where}: not an electorate at the {year} election{detail}")
        elif fold(name) in seen:
            problems.append(f"{where}: listed twice (entries {seen[fold(name)]} and {i}); keep one")
        else:
            seen[fold(name)] = i
        if "party" not in raw or "p" not in raw:
            problems.append(f"{where}: needs `party` and `p`")
            continue
        found = party(raw["party"], where)
        if found and found not in named:
            untracked.add(found)
            continue
        e = {**defaults.get(found, {}), **raw, "electorate": name or where, "party": found or raw["party"]}
        e.setdefault("slope", DEFAULT_SLOPE)
        for key in ("p", "independent_p", "at_share", "slope"):
            if key in e and not isinstance(e[key], (int, float)):
                problems.append(f"{where}: `{key}` must be a number, not `{e[key]}`")
        if all(isinstance(e.get(k, 0), (int, float)) for k in ("p", "independent_p", "at_share", "slope")):
            p, ind = float(e["p"]), float(e.get("independent_p", 0.0))
            if not 0 <= p <= 1 or not 0 <= ind <= 1:
                problems.append(f"{where}: `p` and `independent_p` are probabilities, between 0 and 1")
            elif p + ind > 1 + 1e-9:
                problems.append(f"{where}: p ({p:g}) and independent_p ({ind:g}) add up to more than 1")
            if "at_share" in e and not 0 < float(e["at_share"]) < 1:
                problems.append(f"{where}: `at_share` is a share of the national vote, e.g. 0.03 for 3%")
        electorates.append(e)

    coalitions, names = [], set()
    for i, c in enumerate(electorates_cfg.get("coalitions") or [], start=1):
        where = f"coalitions: {c.get('name', f'entry {i}')}" if isinstance(c, dict) else f"coalitions, entry {i}"
        if not isinstance(c, dict) or not c.get("name") or not c.get("parties"):
            problems.append(f"{where}: needs a `name` and a list of `parties`")
            continue
        if c["name"] in names:
            problems.append(f"{where}: two coalitions have this name")
        names.add(c["name"])
        members = [party(p, where) for p in c["parties"]]
        coalitions.append({**c, "parties": [m for m in members if m]})

    bop = electorates_cfg.get("balance_of_power") or {}
    blocs = {name: [m for m in (party(p, f"balance_of_power: {name}") for p in members) if m]
             for name, members in (bop.get("blocs") or {}).items()}
    pivots = [m for m in (party(p, "balance_of_power: pivots") for p in bop.get("pivots") or []) if m]
    for name, members in blocs.items():
        for p in set(members) & set(pivots):
            problems.append(f"balance_of_power: {p} is both a pivot and in {name}; a pivot sits outside the blocs")

    for p in sorted(untracked):
        notes.append(f"{p} is not tracked in this forecast, so its electorates are skipped and it holds no seats")
    sd = electorates_cfg.get("electorate_group_sd", 0.0)
    if not isinstance(sd, (int, float)) or sd < 0:
        problems.append(f"electorate_group_sd: must be a number, 0 or more, not `{sd}`")
        sd = 0.0
    if problems:
        raise AssumptionError("\n".join(problems))
    return Assumptions(boundaries=bounds, electorates=electorates, coalitions=coalitions, blocs=blocs,
                       pivots=pivots, group_sd=float(sd), notes=notes)


def known_parties(cfg) -> list[str]:
    """The tracked parties of the latest forecast, or failing that the parties with a colour in elections.yml."""
    import json

    for path in (cfg.paths.output / "summary.json",
                 cfg.paths.processed / f"dataset_{cfg.forecast_election.year}.json"):
        if path.exists():
            return list(json.loads(path.read_text(encoding="utf-8"))["parties"])
    return [*cfg.colours, OTHER]


def preview_text(cfg, seed: int = 0) -> str:
    """The checked assumptions as a table, with what each implies at the latest forecast's simulated vote.

    Raises AssumptionError when the file has problems. Without a forecast yet, shows the configured values only.
    """
    import numpy as np

    from .forecast.simulate import simulate_electorates

    parties = known_parties(cfg)
    a = cfg.assumptions(parties)
    sims = cfg.paths.output / "seat_sims.npz"
    pi = None
    if sims.exists():
        with np.load(sims, allow_pickle=False) as z:
            if list(z["parties"]) == parties:
                pi = z["pi_election"]
    lines = [f"config/electorates.yml: {len(a.electorates)} electorates, {len(a.coalitions)} coalitions and "
             f"{len(a.blocs)} blocs; every check passed.",
             f"Boundaries: {a.boundaries.name}, in use from the {a.boundaries.from_election} election. "
             f"Shared error per party: {a.group_sd:g}.", *(f"Note: {n}" for n in a.notes), ""]
    head = f"{'Electorate':<19}{'Party':<15}{'p':>6}{'indep.':>8}{'at share':>10}"
    if pi is not None:
        head += f"{'chance now':>12}{'indep. now':>12}"
    lines.append(head)
    for e in a.electorates:
        row = (f"{e['electorate']:<19}{e['party']:<15}{e['p']:>6.2f}"
               f"{e.get('independent_p', 0) or 0:>8.2f}"
               f"{(format(e['at_share'], '.1%') if 'at_share' in e else 'current'):>10}")
        if pi is not None:
            won, indep = simulate_electorates(pi, parties, [e], np.random.default_rng(seed), a.group_sd)
            row += f"{won[:, parties.index(e['party'])].mean():>12.2f}{indep.mean():>12.2f}"
        lines.append(row)
    if pi is not None:
        won, indep = simulate_electorates(pi, parties, a.electorates, np.random.default_rng(seed), a.group_sd)
        lines += ["", "Electorates won, at the latest forecast's election-day vote (mean and 90% range):"]
        for party in sorted({e["party"] for e in a.electorates}, key=parties.index):
            w = won[:, parties.index(party)]
            lines.append(f"  {party:<15}{w.mean():>5.1f}  ({np.quantile(w, 0.05):.0f}-{np.quantile(w, 0.95):.0f})"
                         f"   at least one in {(w > 0).mean():.0%} of simulations")
        if indep.any():
            lines.append(f"  {'independents':<15}{indep.mean():>5.1f}  ({np.quantile(indep, 0.05):.0f}-"
                         f"{np.quantile(indep, 0.95):.0f})")
        lines += ["", "'chance now' averages each seat over the latest forecast's simulated national vote and the",
                  "shared error. Publish a change with: pollofpolls forecast && pollofpolls report"]
    else:
        lines += ["", "Run `pollofpolls forecast` to see what each assumption implies at the simulated vote."]
    return "\n".join(lines)
