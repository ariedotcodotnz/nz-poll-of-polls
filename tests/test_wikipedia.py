from collections import Counter

import pytest

from pollofpolls.config import Config
from pollofpolls.data.pollsters import PollsterMap
from pollofpolls.data.results import (election_results_from_polls, load_reference_results,
                                      verify_results)
from pollofpolls.data.wikipedia import parse_page_file, parse_sample_size, parse_share

YEARS = [2011, 2014, 2017, 2020, 2023, 2026]


@pytest.mark.parametrize("text,expected", [
    ("38.08", 0.3808), ("38", 0.38), ("–", None), ("—", None), ("N/A", None), ("", None),
    ("— N/a", None), ("<1", 0.005), ("1.5[3]", 0.015), ("45%", 0.45),
])
def test_parse_share(text, expected):
    if expected is None:
        assert parse_share(text) is None
    else:
        assert parse_share(text) == pytest.approx(expected)


@pytest.mark.parametrize("text,expected", [
    ("1,000", 1000), ("1,000+", 1000), ("1001", 1001), ("N/A", None), ("", None), (None, None), ("1,701[b]", 1701),
])
def test_parse_sample_size(text, expected):
    assert parse_sample_size(text) == expected


@pytest.fixture(scope="module")
def all_polls(fixtures):
    return {y: parse_page_file(fixtures / f"{y}_party_vote.html", y) for y in YEARS}


def test_every_page_parses_with_election_rows(all_polls):
    for y, polls in all_polls.items():
        assert len(polls) >= 45, y
        years = sorted({p.election_year for p in polls if p.is_election_result})
        expected = [y - 3] if y == 2026 else [y - 3, y]
        assert years == expected, (y, years)


def test_2026_layout(all_polls):
    polls = [p for p in all_polls[2026] if not p.is_election_result]
    pollsters = Counter(p.pollster_raw for p in polls)
    assert pollsters["Roy Morgan"] >= 30
    assert pollsters["Taxpayers' Union–Curia"] >= 30
    assert pollsters["RNZ–Reid Research"] >= 5
    # OPP column is The Opportunities Party
    assert any("TOP" in p.shares for p in polls)
    # sample sizes are parsed from "1,000" style strings
    assert sum(1 for p in polls if p.sample_size) > 100
    # unreported parties are absent, never 0
    assert all(v > 0 for p in polls for v in p.shares.values() if p.pollster_raw == "Anacta")


def test_missing_values_are_not_zero(all_polls):
    # 2011: Roy Morgan did not report the Conservatives in 2008 -> key absent
    first = [p for p in all_polls[2011] if not p.is_election_result][0]
    assert "New Conservative" not in first.shares
    assert first.shares["National"] == pytest.approx(0.44)


def test_pollster_canonicalisation(all_polls, root):
    pm = PollsterMap(Config(root).pollsters_cfg)
    raw = {p.pollster_raw for polls in all_polls.values() for p in polls if not p.is_election_result}
    canon = {r: pm.canonical(r) for r in raw}
    assert canon["1 News–Colmar Brunton"] == "Verian"
    assert canon["1 News–Kantar Public"] == "Verian"
    assert canon["1 News–Verian"] == "Verian"
    assert canon["3 News Reid Research"] == "Reid Research"
    assert canon["Newshub–Reid Research"] == "Reid Research"
    assert canon["RNZ–Reid Research"] == "Reid Research"
    assert canon["Taxpayers' Union–Curia"] == "Curia"
    assert canon["Curia"] == "Curia"
    assert canon["Roy Morgan Research"] == "Roy Morgan"
    assert canon["The Post/Freshwater Strategy"] == "Freshwater Strategy"
    assert canon["Labour–Talbot Mills"] is None  # sponsored release excluded
    assert canon["Business NZ–Reid Research"] is None
    assert canon["Herald–DigiPoll"] == "DigiPoll"


def test_scraped_results_match_reference(all_polls, root):
    scraped = election_results_from_polls([p for polls in all_polls.values() for p in polls])
    reference = load_reference_results(root / "data" / "reference" / "election_results.csv")
    # the polling pages carry results from 2008 on; earlier rows are only used by the fundamentals prior
    assert {y for y in reference if y >= 2008} <= set(scraped)
    assert verify_results(scraped, reference) == []
