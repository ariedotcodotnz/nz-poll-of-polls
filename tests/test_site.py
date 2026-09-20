"""The website's building blocks: escaping, tables, report-only rendering of older summaries, chart labels."""
import json
import re
import shutil
from itertools import pairwise

import numpy as np
import polars as pl

from pollofpolls.report.charts import spread_labels
from pollofpolls.report.site import Site, and_join, esc, swatch, table


def test_escaped_text_is_inert_markdown():
    """Wikipedia text reaches Quarto as Markdown: no HTML, shortcodes, maths, citations or table breaks."""
    hostile = "Evil</script><script>alert(1)</script> {{< env HOME >}} *b* $x$ @cite | [l](javascript:x)\nnext"
    out = esc(hostile)
    assert re.search(r"(?<!\\)[<>{}*$@|\[\]()]", out) is None      # every special character escaped
    assert "\n" not in out and esc("Te Pāti Māori") == "Te Pāti Māori"


def test_probabilities_are_never_printed_as_0_or_100_percent():
    """A simulation that never produced an outcome has not shown it is impossible."""
    from pollofpolls.report.site import prob
    assert prob(0.0) == "<1%" and prob(0.0004) == "<1%" and prob(1.0) == ">99%" and prob(0.999) == ">99%"
    assert prob(100 / 20_000) == "<1%" and prob(19_900 / 20_000) == ">99%"
    assert prob(0.5745) == "57%" and prob(0.006) == "1%"
    assert prob(0.994) == "99%"


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


PARTIES = ["National", "Labour", "Green", "ACT", "NZ First", "Te Pāti Māori", "TOP", "Other"]
SEATS = np.array([[55, 35, 10, 8, 7, 3, 2, 0], [45, 40, 13, 8, 7, 3, 4, 0]])


def test_electorate_sweep_is_unavailable_for_an_untracked_party(tmp_path, root):
    parties = [p for p in PARTIES if p != "Te Pāti Māori"]
    site = _project(tmp_path, root, {}, np.delete(SEATS, PARTIES.index("Te Pāti Māori"), axis=1), parties)
    assert site.electorate_sweep_risk("Te Pāti Māori") is None


def test_electorate_sweep_is_unavailable_without_configured_seats(tmp_path, root):
    site = _project(tmp_path, root, {}, SEATS, PARTIES)
    site.cfg.electorates_cfg["electorates"] = []
    assert site.electorate_sweep_risk("Te Pāti Māori") is None


def test_report_renders_summaries_from_before_the_balance_of_power(tmp_path, root):
    """A summary.json without balance_of_power, or with the older fields, is brought up to date from the seat
    simulations, so `pollofpolls report` works on outputs written by an earlier version."""
    parties, seats = PARTIES, SEATS
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


def test_intro_before_the_first_poll_of_a_term(tmp_path, root):
    """A new term has no polls: summary.json has latest_poll null (or "None", as older versions wrote it)."""
    for i, latest in enumerate([None, "None"]):
        summary = {"n_polls_cycle": 0, "pollsters_cycle": [], "latest_poll": latest,
                   "generated_at": "2026-09-19T08:00:00+00:00"}
        site = _project(tmp_path / f"case{i}", root, summary, SEATS, PARTIES)
        assert "No polls have been published this term yet. Updated 19 September 2026." in site.intro()


def test_plotly_is_loaded_once_per_page_run(root, monkeypatch):
    """Quarto's execution daemon runs a page again in the same Python process; every run must load plotly.js."""
    import plotly.graph_objects as go
    monkeypatch.setattr(Site, "figure", lambda self, name, narrow=False: go.Figure(go.Scatter(x=[1], y=[1])))
    for _ in range(2):
        site = Site(root)                                   # what the page's setup cell does on each run
        first, second = site.figure_html("all"), site.figure_html("trend")
        assert first.count("cdn.plot.ly/plotly-") == 1 and "cdn.plot.ly" not in second
        assert "Plotly.newPlot" in first and "Plotly.newPlot" in second


def test_evaluation_counts_the_cases_behind_the_weights(tmp_path, root):
    """A partial backtest (gauss for 2020 at 4 weeks and 2023 at 1 week) is two cases, not two targets times two
    horizons; with no 2023-model runs there is nothing to compare against."""
    site = _project(tmp_path, root, {}, SEATS, PARTIES)
    bt = tmp_path / "output" / "backtest"
    bt.mkdir()
    pl.DataFrame({"variant": ["gauss"], "crps_pp": [1.5], "mae_pp": [2.0], "coverage_90": [0.9],
                  "coverage_50": [0.5], "brier_bloc": [0.1], "n_cases": [2]}).write_csv(bt / "summary.csv")
    pl.DataFrame({"variant": ["gauss", "gauss"], "horizon_weeks": [4, 1], "crps_pp": [1.6, 1.4],
                  "coverage_90": [0.9, 0.9]}).write_csv(bt / "summary_by_horizon.csv")
    pl.DataFrame({"target": [2020, 2023], "horizon_weeks": [4, 1], "variant": ["gauss", "gauss"]}) \
        .write_csv(bt / "scores.csv")
    (bt / "stacking.json").write_text(json.dumps({"weights": {"gauss": 1.0}, "loeo_weights": {}, "n_cases": 2}))
    facts = site.backtest_facts()
    assert facts["n_stacked"] == "2" and facts["n_compared"] == "0"
    assert "fitted on 2 backtest cases" in site.backtest_verdict()


def test_spread_labels_keeps_order_and_spacing():
    values = [29.1, 27.4, 11.4, 11.3, 9.0, 6.8, 2.7, 2.2]
    ys = spread_labels(values, min_sep=1.8, floor=0.9)
    by_value = [y for _, y in sorted(zip(values, ys))]
    assert all(b - a >= 1.8 - 1e-9 for a, b in pairwise(by_value))
    assert min(ys) >= 0.9
    # a label with room stays on its line; crowded ones move only as far as needed
    assert ys[5] == 6.8
    assert abs(ys[0] - 29.1) < 0.1 and abs(ys[1] - 27.4) < 0.1
