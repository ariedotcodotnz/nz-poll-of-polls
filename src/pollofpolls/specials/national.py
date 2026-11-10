"""National party votes: past official counts split into ordinary and special votes, and election-night counts.

The Electoral Commission publishes, for every electorate, the party votes at each voting place and three or four
special-vote rows (overseas, before polling day, on polling day). Summed over the electorates they give the
country's ordinary and special party votes: `build_party_votes` does that from a folder of those files and writes
data/reference/special-votes/party-votes.csv, which the model reads. Run it again when an election is added:

    python -m pollofpolls.specials.national <folder with <year>/party-votes-by-voting-place-<n>.csv> <out.csv>
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import electorate_key

TYPES = ("ordinary", "overseas", "before", "on")
_SPECIAL = [(re.compile(r"^overseas special votes", re.IGNORECASE), "overseas"),
            (re.compile(r"^special votes before polling day", re.IGNORECASE), "before"),
            (re.compile(r"^special votes on polling day", re.IGNORECASE), "on")]
_TOTAL = re.compile(r"\btotal$", re.IGNORECASE)
_NOT_PARTIES = re.compile(r"^(total valid party votes|informal party votes)$", re.IGNORECASE)


def _int(cell: str) -> int:
    cell = cell.strip().replace(",", "")
    return int(float(cell)) if cell else 0


def read_party_csv(path: Path) -> tuple[str, dict[str, dict[str, int]]]:
    """One electorate's party votes: (electorate, {party: {ordinary, overseas, before, on}}).

    Every voting-place row and special-vote row must add up to the file's total row, party by party.
    """
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:                       # the Commission's files before 2017 are Windows-1252
        text = raw.decode("cp1252")
    rows = [[c.strip() for c in r] for r in csv.reader(text.splitlines())]
    header = next((i for i, r in enumerate(rows) if len(r) > 4 and not r[0] and not r[1]
                   and any(_NOT_PARTIES.match(c) for c in r)), None)
    if header is None:
        raise ValueError(f"{path}: no party header row")
    electorate = re.sub(r"\s+\d+$", "", rows[header - 1][0]) if header else ""
    cols = [k for k, c in enumerate(rows[header]) if k >= 2 and c and not _NOT_PARTIES.match(c)]
    parties = [rows[header][k] for k in cols]
    votes = {t: np.zeros(len(cols), dtype=np.int64) for t in TYPES}
    total = None
    for r in rows[header + 1:]:
        if len(r) <= max(cols) or not any(r[k] for k in cols):
            continue
        label = r[1] or r[0]
        counts = np.array([_int(r[k]) for k in cols], dtype=np.int64)
        if _TOTAL.search(label):
            total = counts
            break
        kind = next((t for rx, t in _SPECIAL if rx.match(label)), "ordinary")
        votes[kind] += counts
    if total is None:
        raise ValueError(f"{path}: no total row")
    if not np.array_equal(sum(votes.values()), total):
        raise ValueError(f"{path}: the voting places and special votes do not add up to the total row")
    if not votes["before"].sum() + votes["on"].sum():
        raise ValueError(f"{path}: no special-vote rows found")
    return electorate, {p: {t: int(votes[t][k]) for t in TYPES} for k, p in enumerate(parties)}


def build_party_votes(raw_dir: Path, out_csv: Path) -> list[dict]:
    """Sum every electorate's file under ``raw_dir``/<year>/ into national votes by party and vote type."""
    rows = []
    for year_dir in sorted(p for p in Path(raw_dir).iterdir() if p.is_dir() and p.name.isdigit()):
        national: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(TYPES, 0))
        files = sorted(year_dir.glob("*.csv"))
        names = set()
        for path in files:
            electorate, parties = read_party_csv(path)
            names.add(electorate)
            for party, v in parties.items():
                for t in TYPES:
                    national[party][t] += v[t]
        if len(names) != len(files):
            raise ValueError(f"{year_dir}: {len(files)} files but {len(names)} electorates")
        for party in sorted(national):
            v = national[party]
            rows.append({"year": int(year_dir.name), "party": party, **v, "total": sum(v.values()),
                         "electorates": len(files)})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


# ------------------------------------------------------------------------------------------- model inputs
class PartyNames:
    """Maps party names as published to the model's parties; anything else is "Other".

    ``parties`` is config/specials.yml's ``parliament.parties``: each of the model's parties with the other
    names it has been published under. Names match ignoring case and macrons.
    """

    def __init__(self, parties: dict[str, list[str] | None]):
        self.groups = [*parties, "Other"]
        self._lookup = {}
        for name, aliases in parties.items():
            for alias in [name, *(aliases or [])]:
                self._lookup[electorate_key(alias)] = name

    def group(self, published: str) -> str:
        return self._lookup.get(electorate_key(published), "Other")

    def index(self, published: str) -> int:
        return self.groups.index(self.group(published))


def _read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass
class NationalHistory:
    """Past elections' national party votes, grouped into the model's parties plus "Other"."""
    parties: list[str]              # the model's parties and "Other"; parties[0] is the reference
    years: list[int]
    ordinary: np.ndarray            # (T, K) official ordinary votes
    specials: np.ndarray            # (T, C, K) special votes: C = [overseas, domestic]
    prelim: np.ndarray              # (T, K) election-night count, NaN where not archived

    def year_index(self, year: int) -> int:
        return self.years.index(year)

    @property
    def final(self) -> np.ndarray:
        return self.ordinary + self.specials.sum(1)

    def subset(self, years: list[int]) -> NationalHistory:
        keep = [t for t, y in enumerate(self.years) if y in years]
        return NationalHistory(self.parties, [self.years[t] for t in keep], self.ordinary[keep],
                               self.specials[keep], self.prelim[keep])


def load_history(data_dir: Path, names: PartyNames, before: int | None = None) -> NationalHistory:
    """Official party votes of the elections before ``before``, with their election-night counts."""
    groups = names.groups
    official = [r for r in _read(Path(data_dir) / "party-votes.csv") if before is None or int(r["year"]) < before]
    years = sorted({int(r["year"]) for r in official})
    if not years:
        raise ValueError(f"{data_dir}/party-votes.csv has no election before {before}")
    T, K = len(years), len(groups)
    ordinary, specials, prelim = np.zeros((T, K)), np.zeros((T, 2, K)), np.full((T, K), np.nan)
    for r in official:
        t, k = years.index(int(r["year"])), names.index(r["party"])
        ordinary[t, k] += int(r["ordinary"])
        specials[t, 0, k] += int(r["overseas"])
        specials[t, 1, k] += int(r["before"]) + int(r["on"])
    for r in _read(Path(data_dir) / "preliminary-party-votes.csv"):
        if int(r["year"]) in years:
            t, k = years.index(int(r["year"])), names.index(r["party"])
            prelim[t, k] = np.nan_to_num(prelim[t, k]) + int(r["votes"])
    prelim[np.isnan(prelim).any(1)] = np.nan            # a year is archived whole or not at all
    return NationalHistory(groups, years, ordinary, specials, prelim)


def load_preliminary(data_dir: Path, year: int, names: PartyNames) -> np.ndarray:
    """The election-night party votes of ``year``, (K,) in the order of the model's parties plus "Other"."""
    out = np.zeros(len(names.groups))
    rows = [r for r in _read(Path(data_dir) / "preliminary-party-votes.csv") if int(r["year"]) == year]
    if not rows:
        raise ValueError(f"{data_dir}/preliminary-party-votes.csv has no rows for {year}")
    for r in rows:
        out[names.index(r["party"])] += int(r["votes"])
    return out


def read_electorates(path: Path) -> list[dict]:
    """Every electorate's election-night leader: electorate, leader, party, margin, and where published the
    runner-up's party and the votes of National's and Labour's candidates and of all candidates."""
    rows = []
    for r in _read(path):
        counts = {k: int(r[k]) if r.get(k) else None for k in ("national", "labour", "valid_votes")}
        rows.append({"electorate": r["electorate"], "leader": r["leader"], "party": r["party"],
                     "margin": int(r["margin"]), "runner_up_party": r.get("runner_up_party") or None, **counts})
    if len({electorate_key(r["electorate"]) for r in rows}) != len(rows):
        raise ValueError(f"{path}: an electorate is listed twice")
    return rows


def read_electorate_seats(path: Path, names: PartyNames) -> dict[int, np.ndarray]:
    """Electorates won by each of the model's parties at past elections, {year: (K,)}."""
    out: dict[int, np.ndarray] = {}
    for r in _read(path):
        out.setdefault(int(r["year"]), np.zeros(len(names.groups), dtype=int))[names.index(r["party"])] += int(
            r["electorates"])
    return out


def small_winners(data_dir: Path, seats_csv: Path, names: PartyNames) -> dict[int, list[dict]]:
    """Parties pooled as "Other" that won an electorate at a past election (United Future, Mana), with their own
    votes: {year: [{party, electorates, start, final}]}. ``start`` is the election-night count where archived,
    else the official ordinary votes. A party that wins an electorate takes part in the allocation of seats, so
    past Houses cannot be rebuilt from the pooled votes alone."""
    official = _read(Path(data_dir) / "party-votes.csv")
    prelim = _read(Path(data_dir) / "preliminary-party-votes.csv")
    out: dict[int, list[dict]] = {}
    for r in _read(seats_csv):
        if names.group(r["party"]) != "Other" or not int(r["electorates"]):
            continue
        year, key = int(r["year"]), electorate_key(r["party"])
        mine = [o for o in official if int(o["year"]) == year and electorate_key(o["party"]) == key]
        if not mine:
            raise ValueError(f"{seats_csv}: {r['party']} won an electorate in {year} but has no party votes on file")
        night = [int(x["votes"]) for x in prelim if int(x["year"]) == year and electorate_key(x["party"]) == key]
        out.setdefault(year, []).append({
            "party": r["party"], "electorates": int(r["electorates"]), "final": sum(int(o["total"]) for o in mine),
            "start": sum(night) if night else sum(int(o["ordinary"]) for o in mine)})
    return out


if __name__ == "__main__":
    built = build_party_votes(Path(sys.argv[1]), Path(sys.argv[2]))
    for y in sorted({r["year"] for r in built}):
        rs = [r for r in built if r["year"] == y]
        print(y, rs[0]["electorates"], "electorates,", sum(r["total"] for r in rs), "votes,",
              sum(r["total"] - r["ordinary"] for r in rs), "special")
