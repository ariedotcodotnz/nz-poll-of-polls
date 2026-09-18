from datetime import date

import pytest

from pollofpolls.data.dates import midpoint, parse_date_range, week_start


@pytest.mark.parametrize("text,expected", [
    ("17 Oct 2020", (date(2020, 10, 17), date(2020, 10, 17))),
    ("4–11 Sep 2026", (date(2026, 9, 4), date(2026, 9, 11))),
    ("27 Jul – 23 Aug 2026", (date(2026, 7, 27), date(2026, 8, 23))),
    ("28 Dec 2019 – 5 Jan 2020", (date(2019, 12, 28), date(2020, 1, 5))),
    ("24 Dec – 5 Jan 2020", (date(2019, 12, 24), date(2020, 1, 5))),
    ("Sep 2020", (date(2020, 9, 1), date(2020, 9, 30))),
    ("Early Apr 2017", (date(2017, 4, 1), date(2017, 4, 10))),
    ("Late Feb 2017", (date(2017, 2, 21), date(2017, 2, 28))),
    ("2–7, 14–15 Mar 2022", (date(2022, 3, 2), date(2022, 3, 15))),
    ("31 Sep – 11 Oct 2015", (date(2015, 9, 30), date(2015, 10, 11))),
    ("| Oct 2022", (date(2022, 10, 1), date(2022, 10, 31))),
    ("7–10 Oct 2023[a]", (date(2023, 10, 7), date(2023, 10, 10))),
    ("Date[a]", None),
    ("", None),
])
def test_parse_date_range(text, expected):
    assert parse_date_range(text) == expected


def test_midpoint_and_week():
    assert midpoint(date(2026, 9, 4), date(2026, 9, 11)) == date(2026, 9, 7)
    # 2026-09-07 is a Monday; Sunday-starting week begins 2026-09-06
    assert week_start(date(2026, 9, 7)) == date(2026, 9, 6)
    assert week_start(date(2026, 9, 6)) == date(2026, 9, 6)
