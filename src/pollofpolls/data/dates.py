"""Parse the date strings used in Wikipedia's NZ opinion-polling tables.

Formats seen across the 2011-2026 pages (footnote markers such as ``[a]`` or ``[nb 1]`` removed first):

* ``17 Oct 2020`` (single day)
* ``4–11 Sep 2026`` (day range within a month)
* ``27 Jul – 23 Aug 2026`` (range across months)
* ``28 Dec 2019 – 5 Jan 2020`` (range across years, explicit years)
* ``Sep 2020`` (whole month) and ``Early Apr 2017`` / ``Mid`` / ``Late``
* ``2–7, 14–15 Mar 2022`` (split field period)
* ``31 Sep – 11 Oct 2015`` (typo on the page; day clamped to the end of the month)
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_name) if m})
_MONTHS["sept"] = 9

_FOOTNOTE = re.compile(r"\[[^\]]*\]")
_DASH = re.compile(r"\s*[–—‒−-]\s*")
_WS = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Remove footnote markers, non-breaking spaces and stray table pipes; collapse whitespace."""
    t = _FOOTNOTE.sub("", text or "").replace("\xa0", " ")
    t = _WS.sub(" ", t).strip().lstrip("|").strip()
    return t


def _month(token: str | None) -> int | None:
    if not token:
        return None
    return _MONTHS.get(token.lower().rstrip("."))


def _safe_date(year: int, month: int, day: int) -> date:
    """Clamp an out-of-range day (e.g. 31 Sep) to the last day of the month."""
    last = calendar.monthrange(year, month)[1]
    return date(year, month, min(max(day, 1), last))


def _end_of_month(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def parse_date_range(text: str | None) -> tuple[date, date] | None:
    """Return ``(from_date, to_date)`` for a Wikipedia field-period string, or ``None`` if unparseable."""
    t = _DASH.sub(" - ", clean_text(text))
    if not t:
        return None

    m = re.fullmatch(r"(Early|Mid|Late) ([A-Za-z]+)\.? (\d{4})", t, flags=re.IGNORECASE)
    if m:
        mo, yr = _month(m.group(2)), int(m.group(3))
        if mo is None:
            return None
        part = m.group(1).lower()
        if part == "early":
            return date(yr, mo, 1), date(yr, mo, 10)
        if part == "mid":
            return date(yr, mo, 11), date(yr, mo, 20)
        return date(yr, mo, 21), _end_of_month(yr, mo)

    m = re.fullmatch(r"([A-Za-z]+)\.? (\d{4})", t)
    if m and _month(m.group(1)):
        mo, yr = _month(m.group(1)), int(m.group(2))
        return date(yr, mo, 1), _end_of_month(yr, mo)

    parts = [p.strip() for p in re.split(r" - |,", t) if p.strip()]
    tail = parts[-1]
    m = re.fullmatch(r"(\d{1,2}) ([A-Za-z]+)\.? (\d{4})", tail)
    if not m:
        return None
    to_mo = _month(m.group(2))
    if to_mo is None:
        return None
    year = int(m.group(3))
    to_date = _safe_date(year, to_mo, int(m.group(1)))
    if len(parts) == 1:
        return to_date, to_date

    head = parts[0]
    m = re.fullmatch(r"(\d{1,2})(?: ([A-Za-z]+)\.?)?(?: (\d{4}))?", head)
    if not m:
        return None
    from_day = int(m.group(1))
    from_mo = _month(m.group(2)) if m.group(2) else to_mo
    if from_mo is None:
        return None
    from_year = int(m.group(3)) if m.group(3) else year
    if not m.group(3) and from_mo > to_mo:
        from_year = year - 1
    from_date = _safe_date(from_year, from_mo, from_day)
    if from_date > to_date:
        return None
    return from_date, to_date


def midpoint(from_date: date, to_date: date) -> date:
    """Middle day of a field period (rounded down)."""
    return from_date + timedelta(days=(to_date - from_date).days // 2)


def week_start(d: date) -> date:
    """Sunday-starting week, matching lubridate's ``floor_date(unit = "week")`` in the R pipeline."""
    return d - timedelta(days=(d.weekday() + 1) % 7)
