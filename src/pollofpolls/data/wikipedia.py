"""Fetch and parse the party-vote polling tables from Wikipedia.

The parser is deliberately layout-agnostic: it picks the largest table whose header has a pollster
column and at least three party columns, so it works for every page from 2011 to 2026 even though
the column order, abbreviations and presence of Sample size / Lead columns differ between pages.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .dates import clean_text, midpoint, parse_date_range
from .parties import canonical_party

URL_TEMPLATE = "https://en.wikipedia.org/wiki/Opinion_polling_for_the_{year}_New_Zealand_general_election"
USER_AGENT = "pollofpolls/2026 (NZ poll aggregation; https://github.com/ariedotcodotnz/nz-poll-of-polls)"

_POLLSTER_HEADERS = {"poll", "polling organisation", "pollster"}
_ELECTION_ROW = re.compile(r"(\d{4}) election result", re.I)
_MISSING = {"", "–", "—", "-", "n/a", "— n/a", "na", "?", "tbc", "tba"}


@dataclass
class Poll:
    """One published poll (or an election-result row) in long-ish form."""

    page_year: int
    pollster_raw: str
    date_text: str
    date_from: date
    date_to: date
    mid_date: date
    sample_size: int | None
    is_election_result: bool
    election_year: int | None
    shares: dict[str, float] = field(default_factory=dict)  # canonical party -> proportion (0-1)

    def to_record(self) -> dict:
        d = self.__dict__.copy()
        for k in ("date_from", "date_to", "mid_date"):
            d[k] = d[k].isoformat()
        return d


def page_url(year: int, page: str | None = None) -> str:
    """Article URL for an election's polling page; ``page`` overrides the standard title."""
    if page:
        return "https://en.wikipedia.org/wiki/" + page.replace(" ", "_")
    return URL_TEMPLATE.format(year=year)


def fetch_page(year: int, raw_dir: Path, force: bool = False, timeout: int = 60, page: str | None = None) -> Path:
    """Download the page for ``year`` into ``raw_dir`` using a conditional request; return the path."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    html_path = raw_dir / f"{year}.html"
    meta_path = raw_dir / f"{year}.meta.json"
    headers = {"User-Agent": USER_AGENT}
    meta = {}
    if meta_path.exists() and html_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        if meta.get("etag"):
            headers["If-None-Match"] = meta["etag"]
        if meta.get("last_modified"):
            headers["If-Modified-Since"] = meta["last_modified"]
    resp = requests.get(page_url(year, page), headers=headers, timeout=timeout)
    if resp.status_code == 304 and html_path.exists():
        return html_path
    resp.raise_for_status()
    html_path.write_text(resp.text, encoding="utf-8")
    meta_path.write_text(json.dumps({
        "etag": resp.headers.get("ETag"),
        "last_modified": resp.headers.get("Last-Modified"),
        "fetched_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "url": resp.url,
    }))
    return html_path


def _header_cells(table) -> list[str]:
    for tr in table.find_all("tr"):
        ths = tr.find_all("th")
        if ths and not tr.find("td"):
            return [clean_text(th.get_text(" ")) for th in ths]
    return []


def _body_rows(table):
    return [tr for tr in table.find_all("tr") if tr.find("td")]


def _table_layout(headers: list[str]) -> dict | None:
    """Return column indices for a party-vote table, or None if the table is something else."""
    pollster_col = next((i for i, h in enumerate(headers) if h.lower() in _POLLSTER_HEADERS), None)
    date_col = next((i for i, h in enumerate(headers) if h.lower().startswith("date")), None)
    if pollster_col is None or date_col is None:
        return None
    sample_col = next((i for i, h in enumerate(headers) if h.lower().startswith("sample")), None)
    party_cols = {i: canonical_party(h) for i, h in enumerate(headers) if canonical_party(h)}
    if len(party_cols) < 3:
        return None
    return {"pollster": pollster_col, "date": date_col, "sample": sample_col, "parties": party_cols,
            "n_cols": len(headers)}


def find_party_vote_table(soup: BeautifulSoup):
    """The party-vote table: largest wikitable with a pollster column and >= 3 party columns."""
    best, best_rows = None, -1
    for table in soup.select("table.wikitable"):
        layout = _table_layout(_header_cells(table))
        if layout is None:
            continue
        n = len(_body_rows(table))
        if n > best_rows:
            best, best_rows = (table, layout), n
    if best is None:
        raise ValueError("no party-vote table found")
    return best


def parse_share(text: str) -> float | None:
    s = clean_text(text).replace("%", "").replace(",", "").strip()
    if s.lower() in _MISSING:
        return None
    m = re.fullmatch(r"<\s*(\d+(?:\.\d+)?)", s)
    if m:  # "<1" reported as half the bound
        return float(m.group(1)) / 2 / 100
    m = re.fullmatch(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    return float(m.group(1)) / 100


def parse_sample_size(text: str | None) -> int | None:
    if text is None:
        return None
    digits = re.sub(r"[^\d]", "", clean_text(text))
    if not digits:
        return None
    n = int(digits)
    return n if 100 <= n <= 100_000 else None


def clean_pollster(text: str) -> str:
    t = clean_text(text)
    t = re.sub(r"\s*Archived.*$", "", t, flags=re.I)
    t = re.sub(r"\s*[–—-]\s*", "–", t)  # normalise dash spacing: "1 News – Colmar" -> "1 News–Colmar"
    return t.strip()


def parse_polls(html: str, page_year: int) -> list[Poll]:
    """Parse the party-vote table of one page into ``Poll`` objects (including election-result rows)."""
    soup = BeautifulSoup(html, "lxml")
    table, layout = find_party_vote_table(soup)
    polls: list[Poll] = []
    for tr in _body_rows(table):
        cells = [clean_text(c.get_text(" ")) for c in tr.find_all(["td", "th"])]
        if len(cells) < layout["n_cols"] - 3:  # event / commentary rows span the table
            continue
        pollster_raw = clean_pollster(cells[layout["pollster"]])
        if not pollster_raw:
            continue
        rng = parse_date_range(cells[layout["date"]])
        if rng is None:
            continue
        m = _ELECTION_ROW.search(pollster_raw)
        shares = {}
        for col, party in layout["parties"].items():
            if col < len(cells):
                v = parse_share(cells[col])
                if v is not None:
                    shares[party] = v
        if not shares:
            continue
        sample = parse_sample_size(cells[layout["sample"]]) if layout["sample"] is not None and layout["sample"] < len(cells) else None
        polls.append(Poll(
            page_year=page_year,
            pollster_raw=pollster_raw,
            date_text=cells[layout["date"]],
            date_from=rng[0],
            date_to=rng[1],
            mid_date=midpoint(*rng),
            sample_size=sample,
            is_election_result=m is not None,
            election_year=int(m.group(1)) if m else None,
            shares=shares,
        ))
    return polls


def parse_page_file(path: Path, page_year: int) -> list[Poll]:
    return parse_polls(path.read_text(encoding="utf-8"), page_year)
