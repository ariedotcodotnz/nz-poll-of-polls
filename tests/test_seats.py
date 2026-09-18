import numpy as np
import pytest

from pollofpolls.data.results import load_electorate_seats, load_reference_results
from pollofpolls.forecast.seats import allocate_seats, allocate_seats_matrix

OFFICIAL = {
    2011: {"National": 59, "Labour": 34, "Green": 14, "NZ First": 8, "Te Pāti Māori": 3, "Mana": 1, "ACT": 1,
           "United Future": 1},
    2014: {"National": 60, "Labour": 32, "Green": 14, "NZ First": 11, "Te Pāti Māori": 2, "ACT": 1,
           "United Future": 1},
    2017: {"National": 56, "Labour": 46, "NZ First": 9, "Green": 8, "ACT": 1},
    2020: {"Labour": 65, "National": 33, "Green": 10, "ACT": 10, "Te Pāti Māori": 2},
    2023: {"National": 48, "Labour": 34, "Green": 15, "ACT": 11, "NZ First": 8, "Te Pāti Māori": 6},
}


def test_simple_example():
    assert allocate_seats({"a": 53000, "b": 24000, "c": 23000}, nseats=7, threshold=0) == {"a": 3, "b": 2, "c": 2}


@pytest.mark.parametrize("year", sorted(OFFICIAL))
def test_official_results(year, root):
    votes = load_reference_results(root / "data/reference/election_results.csv")[year]
    electorates = load_electorate_seats(root / "data/reference/electorate_seats.csv")[year]
    seats = allocate_seats(votes, electorates)
    got = {p: s for p, s in seats.items() if s > 0}
    assert got == OFFICIAL[year]
    assert sum(got.values()) == sum(OFFICIAL[year].values())


def test_independent_reduces_pool():
    votes = {"A": 0.5, "B": 0.5}
    assert sum(allocate_seats(votes, {}, independent_seats=1).values()) == 119


def test_matrix_matches_scalar(root):
    ref = load_reference_results(root / "data/reference/election_results.csv")
    el = load_electorate_seats(root / "data/reference/electorate_seats.csv")
    parties = ["National", "Labour", "Green", "NZ First", "ACT", "Te Pāti Māori", "TOP", "New Conservative",
               "United Future", "Mana", "Other"]
    rows, erows = [], []
    for year in [2014, 2017, 2020, 2023]:
        v = [ref[year].get(p, 0.0) for p in parties[:-1]]
        v.append(max(0.0, 1 - sum(v)))
        rows.append(v)
        erows.append([el[year].get(p, 0) for p in parties])
    votes = np.array(rows)
    seats = allocate_seats_matrix(votes, np.array(erows))
    for s, year in zip(seats, [2014, 2017, 2020, 2023]):
        got = {p: int(x) for p, x in zip(parties, s) if x > 0}
        assert got == OFFICIAL[year], year
