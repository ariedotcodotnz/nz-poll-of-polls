import pytest

from pollofpolls.data.parties import canonical_party


@pytest.mark.parametrize("header,expected", [
    ("NAT", "National"), ("National", "National"), ("LAB", "Labour"), ("GRN", "Green"),
    ("ACT", "ACT"), ("NZF", "NZ First"), ("NZ First", "NZ First"), ("MRI", "Te Pāti Māori"),
    ("TPM", "Te Pāti Māori"), ("Māori", "Te Pāti Māori"), ("TOP", "TOP"), ("OPP", "TOP"),
    ("CON", "New Conservative"), ("Con", "New Conservative"), ("NCP", "New Conservative"),
    ("UNF", "United Future"), ("United Future", "United Future"), ("MNA", "Mana"), ("Internet Mana", "Mana"),
    ("Prog", "Progressive"), ("ANZ", "Advance NZ"),
    ("Lead", None), ("Others", None), ("Sample size", None), ("Date", None), ("Poll", None), ("", None),
    ("Polling organisation", None), ("Luxon", None),
])
def test_canonical_party(header, expected):
    assert canonical_party(header) == expected
