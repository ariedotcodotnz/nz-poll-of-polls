"""Electoral Commission results by voting place: official counts (CSV) and preliminary counts (xlsx).

Official results ("candidate-votes-by-voting-place-<n>.csv") list every voting place under "Advance Voting
Places" and "Voting Places", then three special-vote rows, a row for places that took fewer than six votes, and
the electorate total. The candidates' parties are listed under the total.

Preliminary results ("<n>_<Name>.xlsx") have a "Candidate" and a "Party" sheet with the same two sections and a
TOTAL row, but no special votes and no parties for the candidates; those come from config/specials.yml.

Every parser checks its arithmetic against the published totals and raises on any mismatch, so a changed file
layout fails loudly instead of silently dropping votes.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..data.parties import canonical_party
from .xlsx import read_workbook

CATEGORIES = ("overseas", "before", "on")
CATEGORY_LABELS = {"overseas": "Overseas (incl. Defence Force)", "before": "Before polling day",
                   "on": "On polling day"}
# Only the Commission's three special-vote rows: a voting place such as "Central Auckland Specialist School"
# must not match.
_SPECIAL_ROW = re.compile(r"^\s*(Overseas Special Votes|Special Votes (BEFORE|ON) Polling Day)\b", re.IGNORECASE)
_SMALL_PLACES = re.compile(r"^\s*Voting places where less than 6 votes were taken", re.IGNORECASE)
_EXTRA_PARTIES = {"opportunity": "TOP", "opportunity party": "TOP", "the opportunity party": "TOP",
                  "independent": "Independent", "": "Independent"}


def electorate_key(name: str) -> str:
    """A matching key for an electorate name: case, macrons, "Mt"/"Mount" and spacing ignored."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    plain = re.sub(r"\s+", " ", plain.strip().lower())
    return re.sub(r"^mt\.? ", "mount ", plain)


def candidate_key(name: str) -> str:
    """Surname of a candidate as the Commission prints it ("WOOD, Michael Philip" -> "wood")."""
    return electorate_key(name.split(",")[0])


def candidate_party(label: str | None) -> str:
    """Canonical party for a candidate's party label as the Commission prints it, or the label itself."""
    raw = re.sub(r"\s+", " ", (label or "").strip())
    if raw.lower() in _EXTRA_PARTIES:
        return _EXTRA_PARTIES[raw.lower()]
    tries = [raw, re.sub(r"\s*\([^)]*\)\s*$", "", raw)]
    tries += [re.sub(r"\s+Party$", "", re.sub(r"^The\s+", "", t), flags=re.IGNORECASE) for t in list(tries)]
    for t in tries:
        if t.lower() in _EXTRA_PARTIES:
            return _EXTRA_PARTIES[t.lower()]
        found = canonical_party(t)
        if found:
            return found
    return raw


@dataclass
class BoothTable:
    """Votes by voting place: ``votes[p, j]`` for place ``places[p]`` and column ``columns[j]``."""
    places: list[str]
    sections: list[str]          # "advance" or "election_day" per place
    columns: list[str]
    votes: np.ndarray


@dataclass
class ElectorateCount:
    """One electorate at one election, official or preliminary. Vectors are per candidate, in ``candidates`` order."""
    year: int
    electorate: str
    number: int | None
    candidates: list[str]
    parties: list[str]
    advance: np.ndarray
    election_day: np.ndarray
    ordinary: np.ndarray                         # every ordinary vote, including places with fewer than six
    specials: dict[str, np.ndarray]              # by category; empty for a preliminary count
    total: np.ndarray
    informal: int | None
    official: bool
    preliminary: np.ndarray | None = None        # the election-night count of this result, when known
    preliminary_informal: int | None = None
    preliminary_source: str | None = None
    booths: BoothTable | None = None
    party_booths: BoothTable | None = None
    source: str = ""

    @property
    def key(self) -> str:
        return electorate_key(self.electorate)

    @property
    def special_total(self) -> np.ndarray:
        return sum(self.specials.values()) if self.specials else np.zeros(len(self.candidates))

    def party_index(self, party: str) -> int | None:
        """The candidate standing for ``party``, or None. A party with two candidates is an error."""
        found = [i for i, p in enumerate(self.parties) if p == party]
        if len(found) > 1:
            raise ValueError(f"{self.electorate} {self.year}: {len(found)} candidates for {party}")
        return found[0] if found else None

    def grouped(self, votes: np.ndarray, pair: list[str]) -> np.ndarray | None:
        """``votes`` as [pair[0]'s candidate, pair[1]'s candidate, everyone else], or None if either is absent."""
        idx = [self.party_index(p) for p in pair]
        if any(i is None for i in idx):
            return None
        votes = np.asarray(votes, dtype=float)
        rest = votes.sum() - votes[idx[0]] - votes[idx[1]]
        return np.array([votes[idx[0]], votes[idx[1]], rest])


def _num(x) -> float | None:
    """A cell as a number. The Commission's workbooks store counts as text ("337") as well as numbers."""
    if isinstance(x, float):
        return x
    if isinstance(x, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", x.replace(",", "")):
        return float(x.replace(",", ""))
    return None


def _ints(cells: list[str]) -> list[int] | None:
    out = []
    for c in cells:
        c = (c or "").strip().replace(",", "")
        if not re.fullmatch(r"-?\d+", c):
            return None
        out.append(int(c))
    return out


# ------------------------------------------------------------------------------------------- official CSV
def read_official_csv(path: Path, year: int) -> ElectorateCount:
    """Parse an official "candidate-votes-by-voting-place" CSV and check it adds up."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    m = re.match(r"^\s*(.*?)\s+(\d+)\s*$", rows[1][0]) if len(rows) > 1 and rows[1] else None
    if not m:
        raise ValueError(f"{path}: no electorate name and number in the second line")
    name, number = m.group(1), int(m.group(2))
    header = rows[2]
    candidates = [h.strip() for h in header[2:] if h.strip() and not h.startswith(("Total", "Informal"))]
    C = len(candidates)
    if C < 2 or header[2 + C].strip() != "Total Valid Candidate Votes":
        raise ValueError(f"{path}: unexpected header {header}")

    advance, eday, small = np.zeros(C), np.zeros(C), np.zeros(C)
    specials: dict[str, np.ndarray] = {}
    places, sections, booth_rows = [], [], []
    section, total, informal = None, None, 0
    for r in rows[3:]:
        if not r or not any(x.strip() for x in r):
            continue
        head = r[0].strip()
        if head == "Advance Voting Places":
            section = "advance"
            continue
        if head == "Voting Places":
            section = "election_day"
            continue
        label = r[1].strip() if len(r) > 1 else ""
        if label.endswith(" Total") and not head:
            total = _ints(r[2:2 + C])
            break
        values = _ints(r[2:4 + C]) if len(r) >= 4 + C else None
        if values is None:
            if _SMALL_PLACES.match(label):        # 2017 leaves this row empty when no place took fewer than six
                continue
            continue
        v, row_total, row_informal = np.array(values[:C], float), values[C], values[C + 1]
        if v.sum() != row_total:
            raise ValueError(f"{path}: row {label!r} candidates sum to {v.sum():.0f}, total says {row_total}")
        m = _SPECIAL_ROW.match(label)
        if m:
            cat = "overseas" if m.group(1).lower().startswith("overseas") else m.group(2).lower()
            if cat in specials:
                raise ValueError(f"{path}: two {cat} special-vote rows")
            specials[cat] = v
        elif _SMALL_PLACES.match(label):
            small += v
        elif section is None:
            raise ValueError(f"{path}: voting place {label!r} before any section heading")
        else:
            (advance if section == "advance" else eday)[:] += v
            places.append(f"{head}, {label}" if head else label)
            sections.append(section)
            booth_rows.append(v)
        informal += row_informal
    if total is None:
        raise ValueError(f"{path}: no electorate total row")
    if set(specials) != set(CATEGORIES):
        raise ValueError(f"{path}: special-vote rows found {sorted(specials)}, expected {list(CATEGORIES)}")
    total = np.array(total, float)
    ordinary = advance + eday + small
    if not np.array_equal(ordinary + sum(specials.values()), total):
        raise ValueError(f"{path}: voting places and special votes do not add up to the electorate total")

    parties = {}
    for r in rows:
        if len(r) >= 3 and r[0].strip() in candidates and _ints([r[2]]) is not None:
            parties[r[0].strip()] = candidate_party(r[1])
    missing = [c for c in candidates if c not in parties]
    if missing:
        raise ValueError(f"{path}: no party listed for {missing}")
    return ElectorateCount(
        year=year, electorate=name, number=number, candidates=candidates,
        parties=[parties[c] for c in candidates], advance=advance, election_day=eday, ordinary=ordinary,
        specials=specials, total=total, informal=informal, official=True,
        booths=BoothTable(places, sections, candidates, np.array(booth_rows)), source=str(path))


# ----------------------------------------------------------------------------------------- preliminary xlsx
def _booth_sheet(rows: list[list], path: Path, sheet: str) -> tuple[str, int | None, BoothTable, np.ndarray, int]:
    title = next((str(r[0]) for r in rows if r and r[0]), "")
    m = re.match(r"^\s*(\d+)\s+(.*?)\s+-\s+\w+ Vote Details", title)
    if not m:
        raise ValueError(f"{path} [{sheet}]: unexpected title {title!r}")
    number, name = int(m.group(1)), m.group(2).strip()
    h = next((i for i, r in enumerate(rows) if r and r[0] and "Preliminary Results" in str(r[0])), None)
    if h is None:
        raise ValueError(f"{path} [{sheet}]: not a preliminary result (no 'Preliminary Results' header)")
    header = rows[h]
    cols = [(j, str(x).strip()) for j, x in enumerate(header) if j > 0 and x and str(x).strip()]
    names = [(j, x) for j, x in cols if not x.startswith(("Total", "Informal"))]
    j_total = next(j for j, x in cols if x.startswith("Total Valid"))
    j_inf = next((j for j, x in cols if x.startswith("Informal")), None)

    def cell(r, j):
        return r[j] if j < len(r) else None

    places, sections, votes = [], [], []
    section, total, informal = None, None, None
    for r in rows[h + 1:]:
        if not r or r[0] is None:
            continue
        head = str(r[0]).strip()
        if head.upper() == "ADVANCE VOTING PLACES":
            section = "advance"
            continue
        if head.upper() == "VOTING PLACES":
            section = "election_day"
            continue
        vals = [_num(cell(r, j)) for j, _ in names]
        if head.endswith("TOTAL"):
            if any(v is None for v in vals) or _num(cell(r, j_total)) != sum(vals):
                raise ValueError(f"{path} [{sheet}]: TOTAL row does not add up")
            total = np.array(vals)
            if j_inf is not None and _num(cell(r, j_inf)) is not None:
                informal = int(_num(cell(r, j_inf)))
            break
        if any(v is None for v in vals):
            continue
        if section is None:
            raise ValueError(f"{path} [{sheet}]: voting place {head!r} before any section heading")
        v = np.array(vals)
        if _num(cell(r, j_total)) != v.sum():
            raise ValueError(f"{path} [{sheet}]: row {head!r} does not add up")
        places.append(head)
        sections.append(section)
        votes.append(v)
    if total is None:
        raise ValueError(f"{path} [{sheet}]: no TOTAL row")
    if not places or "advance" not in sections or "election_day" not in sections:
        raise ValueError(f"{path} [{sheet}]: no advance or no election-day voting places found")
    booth = BoothTable(places, sections, [x for _, x in names], np.array(votes))
    # a place that took fewer than six votes is printed as zero, so the places can fall short of the TOTAL by a
    # few votes each, never exceed it
    short = total - booth.votes.sum(0)
    if np.any(short < 0) or short.sum() > 5 * len(places):
        raise ValueError(f"{path} [{sheet}]: voting places do not add up to the TOTAL row")
    return name, number, booth, total, informal


def read_preliminary_xlsx(path: Path, year: int, candidate_parties: dict[str, str]) -> ElectorateCount:
    """Parse a preliminary-results workbook. ``candidate_parties`` maps each candidate (as printed) to a party."""
    sheets = read_workbook(path)
    if "Candidate" not in sheets:
        raise ValueError(f"{path}: no 'Candidate' sheet (found {list(sheets)})")
    name, number, booths, total, informal = _booth_sheet(sheets["Candidate"], path, "Candidate")
    party_booths = None
    if "Party" in sheets:
        _, _, party_booths, _, _ = _booth_sheet(sheets["Party"], path, "Party")
        party_booths.columns = [candidate_party(c) for c in party_booths.columns]
    lookup = {candidate_key(k): v for k, v in candidate_parties.items()}
    missing = [c for c in booths.columns if candidate_key(c) not in lookup]
    if missing:
        raise ValueError(f"{path}: no party configured for {missing} (config/specials.yml)")
    extra = set(lookup) - {candidate_key(c) for c in booths.columns}
    if extra:
        raise ValueError(f"{path}: configured candidates not on the ballot: {sorted(extra)}")
    adv = booths.votes[np.array(booths.sections) == "advance"].sum(0)
    eday = total - adv               # election-day places, plus any votes too few to print by place
    return ElectorateCount(
        year=year, electorate=name, number=number, candidates=booths.columns,
        parties=[candidate_party(lookup[candidate_key(c)]) for c in booths.columns], advance=adv,
        election_day=eday, ordinary=total.copy(), specials={}, total=total.copy(), informal=informal,
        official=False, preliminary=total.copy(), preliminary_informal=informal, booths=booths,
        party_booths=party_booths, source=str(path))


# ---------------------------------------------------------------------------------- archived election night
def read_preliminary_counts(path: Path) -> dict[tuple[int, str], dict]:
    """Archived election-night candidate counts: (year, electorate key) -> {"votes": {candidate: n}, ...}."""
    out: dict[tuple[int, str], dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            k = (int(r["year"]), electorate_key(r["electorate"]))
            rec = out.setdefault(k, {"votes": {}, "informal": None, "source": r["source"]})
            if r["candidate"]:
                rec["votes"][r["candidate"]] = int(r["votes"])
            else:
                rec["informal"] = int(r["votes"])
    return out


@dataclass
class History:
    """Official results of past elections, with their election-night counts where archived."""
    counts: list[ElectorateCount] = field(default_factory=list)

    @property
    def years(self) -> list[int]:
        return sorted({c.year for c in self.counts})

    def for_electorate(self, name: str) -> list[ElectorateCount]:
        return sorted((c for c in self.counts if c.key == electorate_key(name)), key=lambda c: c.year)


def load_history(data_dir: Path, before: int) -> History:
    """Every official result in ``data_dir/<year>/`` for years before ``before``, with archived preliminary counts."""
    prelim = read_preliminary_counts(data_dir / "preliminary-counts.csv")
    counts = []
    for ydir in sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        year = int(ydir.name)
        if year >= before:
            continue
        for f in sorted(ydir.glob("candidate-votes-by-voting-place-*.csv")):
            c = read_official_csv(f, year)
            rec = prelim.get((year, c.key))
            if rec:
                missing = [x for x in c.candidates if x not in rec["votes"]]
                if missing:
                    raise ValueError(f"preliminary-counts.csv {year} {c.electorate}: no count for {missing}")
                c.preliminary = np.array([float(rec["votes"][x]) for x in c.candidates])
                c.preliminary_informal = rec["informal"]
                c.preliminary_source = rec["source"]
            counts.append(c)
    keys = [(c.year, c.key) for c in counts]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{data_dir}: an electorate appears twice in one year")
    return History(counts)
