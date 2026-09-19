"""The website's building blocks: escaping, tables, report-only rendering of older summaries, chart labels."""
import json
import re
import shutil
from itertools import pairwise

import numpy as np

from pollofpolls.report.charts import spread_labels
from pollofpolls.report.site import Site, and_join, esc, swatch, table


def test_escaped_text_is_inert_markdown():
    """Wikipedia text reaches Quarto as Markdown: no HTML, shortcodes, maths, citations or table breaks."""
    hostile = "Evil</script><script>alert(1)</script> {{< env HOME >}} *b* $x$ @cite | [l](javascript:x)\nnext"
    out = esc(hostile)
    assert re.search(r"(?<!\\)[<>{}*$@|\[\]()]", out) is None      # every special character escaped
    assert "\n" not in out and esc("Te Pāti Māori") == "Te Pāti Māori"


def test_table_and_swatch():
    md = table(["Party", "Share"], [[f"{swatch('#1F5FBF')}National", "29.1%"]])
    assert md.startswith("::: {.table-scroll}\n| Party | Share |\n|:---|---:|\n") and md.endswith(":::\n")
    assert 'style="background-color:#1F5FBF"' in md
    assert "#777777" in swatch('red" onmouseover="alert(1)')           # only a hex colour reaches the style
    assert and_join([2017]) == "2017" and and_join([1, 2]) == "1 and 2" and and_join([1, 2, 3]) == "1, 2 and 3"


def _project(tmp_path, root, summary: dict, seats: np.ndarray, parties: list[str]):
    shutil.copytree(root / "config", tmp_path / "config")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "summary.json").write_text(json.dumps({"parties": parties, **summary}))
    total = np.full(len(seats), 120)
    np.savez(tmp_path / "output" / "seat_sims.npz", seats_election=seats, total_election=total, seats_now=seats,
             total_now=total)
    return Site(tmp_path)


def test_report_renders_summaries_from_before_the_balance_of_power(tmp_path, root):
    """A summary.json without balance_of_power, or with the older fields, is brought up to date from the seat
    simulations, so `pollofpolls report` works on outputs written by an earlier version."""
    parties = ["National", "Labour", "Green", "ACT", "NZ First", "Te Pāti Māori", "TOP", "Other"]
    seats = np.array([[55, 35, 10, 8, 7, 3, 2, 0], [45, 40, 13, 8, 7, 3, 4, 0]])
    site = _project(tmp_path, root, {"kingmaker": {}}, seats, parties)
    rows = site.balance_rows()
    assert [r["bloc"] for r in rows] == ["Right bloc", "Left bloc"]
    right = rows[0]
    assert right["p_alone"] == 0.5 and right["p_needs_several"] == 0.5 and right["p_short"] == 0
    assert "Needs NZ First and TOP" in site.balance_table()

    old = {"balance_of_power": {"election_day": [{**r, "p_needs_all": 0.9} for r in rows]}}
    for r in old["balance_of_power"]["election_day"]:
        del r["p_needs_several"]
    site = _project(tmp_path / "old", root, old, seats, parties)
    assert site.balance_rows()[0]["p_needs_several"] == 0.5          # recomputed, not the stale field


def test_spread_labels_keeps_order_and_spacing():
    values = [29.1, 27.4, 11.4, 11.3, 9.0, 6.8, 2.7, 2.2]
    ys = spread_labels(values, min_sep=1.8, floor=0.9)
    by_value = [y for _, y in sorted(zip(values, ys))]
    assert all(b - a >= 1.8 - 1e-9 for a, b in pairwise(by_value))
    assert min(ys) >= 0.9
    # a label with room stays on its line; crowded ones move only as far as needed
    assert ys[5] == 6.8
    assert abs(ys[0] - 29.1) < 0.1 and abs(ys[1] - 27.4) < 0.1
