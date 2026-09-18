"""Build the tidy poll table (one row per poll x party) from parsed Wikipedia pages."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from ..config import Config
from ..data.dates import week_start
from ..data.pollsters import PollsterMap
from ..data.wikipedia import Poll

PARTY_ORDER = ["National", "Labour", "Green", "ACT", "NZ First", "Te Pāti Māori", "TOP",
               "New Conservative", "United Future", "Mana", "Advance NZ", "Progressive", "Internet"]


def party_sort_key(name: str) -> tuple[int, str]:
    return (PARTY_ORDER.index(name) if name in PARTY_ORDER else len(PARTY_ORDER), name)


def build_polls_table(polls: list[Poll], cfg: Config) -> pl.DataFrame:
    """Long table of published polls (election-result rows excluded), canonical pollsters, filters applied."""
    pm = PollsterMap(cfg.pollsters_cfg)
    elections = cfg.elections
    rows = []
    for p in polls:
        if p.is_election_result:
            continue
        name = pm.canonical(p.pollster_raw)
        if name is None:
            continue
        if pm.polling_code_only and not pm.polling_code(name):
            continue
        cycle = next((e.year for e in elections if e.date >= p.mid_date), None)
        if cycle is None:
            continue
        n = p.sample_size or pm.sample_size(name)
        for party, share in p.shares.items():
            rows.append({
                "page_year": p.page_year, "pollster": name, "pollster_raw": p.pollster_raw,
                "segment": pm.segment(name, p.mid_date), "date_text": p.date_text,
                "date_from": p.date_from, "date_to": p.date_to, "mid_date": p.mid_date,
                "available": p.date_to + timedelta(days=pm.publication_lag(name)),
                "week": week_start(p.mid_date), "sample_size": int(n),
                "sample_reported": p.sample_size is not None, "cycle": cycle,
                "party": party, "share": float(share),
            })
    df = pl.DataFrame(rows)
    # de-duplicate polls that appear on two pages (same pollster, mid date and National share)
    key = ["pollster", "mid_date", "party"]
    df = df.unique(subset=key, keep="first", maintain_order=True)
    df = df.with_columns(
        pl.concat_str([pl.col("pollster"), pl.lit("|"), pl.col("mid_date").cast(pl.Utf8), pl.lit("|"),
                       pl.col("date_text")]).alias("poll_id")
    )
    return df.sort(["mid_date", "pollster", "party"])


def eligible_polls(df: pl.DataFrame, cutoff: date | None, lagged: bool, min_polls: int = 1) -> pl.DataFrame:
    """Polls that could have been used on ``cutoff``.

    A poll counts once its fieldwork has ended (``date_to``), or, when ``lagged`` (backtests), once it would
    typically have been published (``available``). The fieldwork midpoint is only used to place the poll on
    the weekly timeline. Pollsters with fewer than ``min_polls`` eligible polls are dropped.
    """
    if cutoff is not None:
        df = df.filter(pl.col("available" if lagged else "date_to") <= cutoff)
    if min_polls > 1 and df.height:
        counts = df.select(["pollster", "poll_id"]).unique().group_by("pollster").len()
        df = df.filter(pl.col("pollster").is_in(counts.filter(pl.col("len") >= min_polls)["pollster"].to_list()))
    return df


def tracked_parties(df: pl.DataFrame, cfg: Config, target_year: int, cutoff: date | None = None,
                    lagged: bool = False) -> list[str]:
    """Parties tracked for a target election: in Parliament at the cycle start, plus auto-tracked ones."""
    rule = cfg.auto_track
    sub = eligible_polls(df.filter(pl.col("cycle") == target_year), cutoff, lagged)
    auto = (sub.filter(pl.col("share") >= rule["min_share"]).group_by("party").len()
               .filter(pl.col("len") >= rule["min_polls"])["party"].to_list())
    parties = set(cfg.in_parliament(target_year)) | set(auto)
    return sorted(parties, key=party_sort_key)
