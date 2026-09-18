"""Canonical party names for Wikipedia column headers and result tables."""

from __future__ import annotations

import re

# Order matters: first match wins. Patterns are matched against the cleaned header text.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(NAT|National|National Party)$", re.I), "National"),
    (re.compile(r"^(LAB|Labour|Labour Party)$", re.I), "Labour"),
    (re.compile(r"^(GRN|Green|Greens|Green Party)$", re.I), "Green"),
    (re.compile(r"^(ACT|ACT New Zealand)$", re.I), "ACT"),
    (re.compile(r"^(NZF|NZ First|New Zealand First)$", re.I), "NZ First"),
    (re.compile(r"^(MRI|TPM|M[āa]ori|M[āa]ori Party|Te P[āa]ti M[āa]ori)$", re.I), "Te Pāti Māori"),
    (re.compile(r"^(TOP|OPP|Opportunities|The Opportunities Party)$", re.I), "TOP"),
    (re.compile(r"^(CON|NCP|Conservative|New Conservative)$", re.I), "New Conservative"),
    (re.compile(r"^(UNF|United Future)$", re.I), "United Future"),
    (re.compile(r"^(MNA|Mana|Mana Movement|Internet ?Mana)$", re.I), "Mana"),
    (re.compile(r"^(Internet|Internet Party)$", re.I), "Internet"),
    (re.compile(r"^(Prog|Progressive)$", re.I), "Progressive"),
    (re.compile(r"^(ANZ|Advance NZ|Advance New Zealand)$", re.I), "Advance NZ"),
    (re.compile(r"^(DNZ|DemocracyNZ)$", re.I), "DemocracyNZ"),
    (re.compile(r"^(VNZ|Vision NZ)$", re.I), "Vision NZ"),
    (re.compile(r"^(NZL|NZ Loyal)$", re.I), "NZ Loyal"),
    (re.compile(r"^(SNZ|Sustainable NZ)$", re.I), "Sustainable NZ"),
    (re.compile(r"^(NCP|New Conservative Party)$", re.I), "New Conservative"),
]

# Header cells that are never parties.
_NOT_PARTIES = {
    "", "date", "poll", "polling organisation", "pollster", "sample size", "lead", "others", "other",
    "undecided", "source", "notes",
}


def canonical_party(header: str) -> str | None:
    """Map a Wikipedia column header (or results-table name) to a canonical party name."""
    h = (header or "").strip()
    if h.lower() in _NOT_PARTIES:
        return None
    for pattern, name in _PATTERNS:
        if pattern.match(h):
            return name
    return None


def is_party_header(header: str) -> bool:
    return canonical_party(header) is not None
