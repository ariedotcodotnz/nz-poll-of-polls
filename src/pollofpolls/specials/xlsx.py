"""A minimal reader for the Electoral Commission's .xlsx results, using only the standard library.

The Commission's preliminary results by voting place are small workbooks of plain values (inline or shared
strings and numbers, no formulas that need evaluating), so a full spreadsheet library is not needed.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_COL = re.compile(r"[A-Z]+")


def _column(ref: str) -> int:
    n = 0
    for ch in _COL.match(ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def _text(node: ET.Element) -> str:
    return "".join(t.text or "" for t in node.iter(f"{_NS}t"))


def read_workbook(path: Path) -> dict[str, list[list[str | float | None]]]:
    """Every sheet as rows of cell values: str for text, float for numbers, None for empty cells.

    Rows keep their position (an empty row is an empty list) so section headings can be found by order.
    """
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            shared = [_text(si) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(f"{_NS}si")]
        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        targets = {}
        if "xl/_rels/workbook.xml.rels" in names:
            for rel in ET.fromstring(z.read("xl/_rels/workbook.xml.rels")).findall(f"{_PKG_REL}Relationship"):
                targets[rel.get("Id")] = "xl/" + rel.get("Target").lstrip("/").removeprefix("xl/")
        sheets = {}
        for i, sheet in enumerate(workbook.find(f"{_NS}sheets"), 1):
            target = targets.get(sheet.get(f"{_REL_NS}id"), f"xl/worksheets/sheet{i}.xml")
            sheets[sheet.get("name")] = _read_sheet(ET.fromstring(z.read(target)), shared)
    return sheets


def _read_sheet(root: ET.Element, shared: list[str]) -> list[list[str | float | None]]:
    rows: list[list[str | float | None]] = []
    for row in root.iter(f"{_NS}row"):
        index = int(row.get("r", len(rows) + 1)) - 1
        while len(rows) < index:
            rows.append([])
        cells: dict[int, str | float | None] = {}
        for c in row.findall(f"{_NS}c"):
            kind, ref = c.get("t"), c.get("r")
            col = _column(ref) if ref else len(cells)
            if kind == "inlineStr":
                cells[col] = _text(c)
                continue
            v = c.find(f"{_NS}v")
            if v is None or v.text is None:
                cells[col] = None
            elif kind == "s":
                cells[col] = shared[int(v.text)]
            elif kind in ("str", "e"):
                cells[col] = v.text
            elif kind == "b":
                cells[col] = float(v.text)
            else:
                cells[col] = float(v.text)
        rows.append([cells.get(j) for j in range(max(cells) + 1)] if cells else [])
    return rows
