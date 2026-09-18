"""Official election results: scraped from the "election result" rows and verified against vendored data."""

from __future__ import annotations

import csv
from pathlib import Path

from .parties import canonical_party
from .wikipedia import Poll


def _decimals(v: float) -> int:
    s = f"{v * 100:.4f}".rstrip("0").rstrip(".")
    return len(s.split(".")[1]) if "." in s else 0


def election_results_from_polls(polls: list[Poll]) -> dict[int, dict[str, float]]:
    """year -> {party: share}. When the same result appears on two pages, keep the more precise value."""
    out: dict[int, dict[str, float]] = {}
    for p in polls:
        if not p.is_election_result or p.election_year is None:
            continue
        row = out.setdefault(p.election_year, {})
        for party, v in p.shares.items():
            if party not in row or _decimals(v) > _decimals(row[party]):
                row[party] = v
    return out


def load_reference_results(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            party = canonical_party(r["party"]) or r["party"]
            out.setdefault(int(r["year"]), {})[party] = float(r["share"])
    return out


def load_electorate_seats(path: Path) -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            party = canonical_party(r["party"]) or r["party"]
            out.setdefault(int(r["year"]), {})[party] = int(r["electorates"])
    return out


def verify_results(scraped: dict[int, dict[str, float]], reference: dict[int, dict[str, float]],
                   tol: float = 0.0006) -> list[str]:
    """Discrepancies between scraped and vendored results (empty list means they agree)."""
    problems = []
    for year, ref in reference.items():
        if year not in scraped:
            continue
        for party, v in ref.items():
            if party == "Other" or party not in scraped[year]:
                continue
            if abs(scraped[year][party] - v) > tol:
                problems.append(f"{year} {party}: scraped {scraped[year][party]:.4f} vs reference {v:.4f}")
    return problems


def with_other(results: dict[int, dict[str, float]]) -> dict[int, dict[str, float]]:
    """Add an explicit Other = 1 - sum(listed parties) to each year."""
    out = {}
    for year, row in results.items():
        r = {k: v for k, v in row.items() if k != "Other"}
        r["Other"] = max(0.0, 1.0 - sum(r.values()))
        out[year] = r
    return out
