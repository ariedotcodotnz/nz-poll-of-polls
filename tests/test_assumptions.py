"""Editorial assumptions and electorate boundaries: the checks a hand edit of config/electorates.yml goes through."""
import copy

import pytest

from pollofpolls.assumptions import AssumptionError, boundaries_for, fold, load_assumptions
from pollofpolls.config import Config

PARTIES = ["National", "Labour", "Green", "ACT", "NZ First", "Te Pāti Māori", "TOP", "Other"]


@pytest.fixture(scope="module")
def cfg(root):
    return Config(root)


def _check(cfg, **changes):
    electorates = copy.deepcopy(cfg.electorates_cfg)
    for key, value in changes.items():
        electorates[key] = value
    return load_assumptions(electorates, cfg.boundaries_cfg, PARTIES, 2026)


def _problems(cfg, **changes) -> str:
    with pytest.raises(AssumptionError) as err:
        _check(cfg, **changes)
    return str(err.value)


def test_the_shipped_assumptions_pass(cfg):
    a = cfg.assumptions(PARTIES)
    assert a.boundaries.from_election == 2026 and len(a.electorates) >= 7
    tpm = [e for e in a.electorates if e["party"] == "Te Pāti Māori"]
    assert len(tpm) == 7 and all(e["at_share"] == 0.022 and e["slope"] == 0.40 for e in tpm)   # party defaults
    assert all(e["electorate"] in a.boundaries.electorates for e in a.electorates)


def test_boundary_sets_line_up(cfg):
    """The 2025 review: 65 general electorates became 64, and every abolished name maps onto new ones."""
    new, old = boundaries_for(cfg.boundaries_cfg, 2026), boundaries_for(cfg.boundaries_cfg, 2023)
    assert (len(new.general), len(new.maori), len(old.general)) == (64, 7, 65)
    assert new.maori == old.maori
    abolished = set(old.general) - set(new.general)
    assert abolished == set(new.replaced)                     # every abolished electorate says where it went
    for successors in new.replaced.values():
        assert set(successors) <= set(new.electorates)
    created = set(new.general) - set(old.general)
    assert created == {s for succ in new.replaced.values() for s in succ} - set(old.general)
    assert boundaries_for(cfg.boundaries_cfg, 2020).name == old.name


def test_an_abolished_electorate_names_its_successors(cfg):
    extra = cfg.electorates_cfg["electorates"] + [{"electorate": "Ohariu", "party": "Green", "p": 0.2}]
    msg = _problems(cfg, electorates=extra)
    assert "Ohariu: abolished at the 2025 boundary review" in msg and "Wellington North, Kenepuru and Hutt South" in msg


def test_typos_get_the_nearest_name(cfg):
    extra = cfg.electorates_cfg["electorates"] + [
        {"electorate": "Welington Bays", "party": "Te Party Maori", "p": 0.2, "indpendent_p": 0.1}]
    msg = _problems(cfg, electorates=extra)
    assert "did you mean Wellington Bays?" in msg
    assert "did you mean Te Pāti Māori?" in msg
    assert "unknown key `indpendent_p`; did you mean independent_p?" in msg


def test_names_match_without_macrons_and_come_back_canonical(cfg):
    entries = [{"electorate": "tamaki makaurau", "party": "Te Pati Maori", "p": 0.7}]
    a = _check(cfg, electorates=entries)
    assert a.electorates[0]["electorate"] == "Tāmaki Makaurau" and a.electorates[0]["party"] == "Te Pāti Māori"
    assert fold("Ōtāhuhu") == "otahuhu"


def test_probabilities_duplicates_and_blocs_are_checked(cfg):
    entries = [{"electorate": "Waiariki", "party": "Te Pāti Māori", "p": 0.7, "independent_p": 0.4},
               {"electorate": "waiariki", "party": "Te Pāti Māori", "p": 0.5}]
    msg = _problems(cfg, electorates=entries, coalitions=[{"name": "Nats", "parties": ["Nationl"]}],
                    balance_of_power={"blocs": {"Right": ["National", "NZ First"]}, "pivots": ["NZ First"]})
    assert "add up to more than 1" in msg and "listed twice" in msg
    assert "coalitions: Nats: `Nationl` is not a party the model knows" in msg and "did you mean National?" in msg
    assert "NZ First is both a pivot and in Right" in msg


def test_one_file_serves_a_forecast_of_an_earlier_election(cfg):
    """2020 boundaries, without Te Pāti Māori or TOP tracked: later electorates and untracked parties are notes."""
    tracked = ["National", "Labour", "Green", "ACT", "NZ First", "Other"]
    a = cfg.assumptions(tracked, year=2020)
    assert {e["electorate"] for e in a.electorates} == {"Epsom", "Auckland Central"}
    assert any("Wellington Bays: skipped" in n for n in a.notes)
    assert any(n.startswith("Te Pāti Māori is not tracked") for n in a.notes)
    assert any("TOP" in c["parties"] for c in a.coalitions) and a.pivots == ["NZ First", "TOP"]   # hold no seats


def test_a_party_nobody_has_heard_of_is_still_an_error(cfg):
    extra = cfg.electorates_cfg["electorates"] + [{"electorate": "Kapiti", "party": "Te Tai Tokerau Party", "p": 0.2}]
    with pytest.raises(AssumptionError, match="independent_p` on another party's entry"):
        load_assumptions({**cfg.electorates_cfg, "electorates": extra}, cfg.boundaries_cfg, PARTIES, 2026,
                         known=list(cfg.colours))


def test_a_misspelt_section_is_an_error_not_an_empty_list(cfg):
    """Renaming `electorates` by accident must not quietly drop every electorate assumption."""
    bad = copy.deepcopy(cfg.electorates_cfg)
    bad["electroates"] = bad.pop("electorates")
    bad["balance_of_power"]["pivot"] = bad["balance_of_power"].pop("pivots")
    with pytest.raises(AssumptionError) as err:
        load_assumptions(bad, cfg.boundaries_cfg, PARTIES, 2026)
    msg = str(err.value)
    assert "unknown section `electroates`; did you mean electorates?" in msg
    assert "the `electorates` section is missing; write `electorates: []` if there should be none" in msg
    assert "balance_of_power: unknown key `pivot`; did you mean pivots?" in msg
    empty = {**copy.deepcopy(cfg.electorates_cfg), "electorates": []}
    assert load_assumptions(empty, cfg.boundaries_cfg, PARTIES, 2026).electorates == []   # none, on purpose


def test_parties_from_past_parliaments_are_known_not_misspellings(cfg, monkeypatch):
    """United Future and Mana have no colour but sat in Parliament: untracked now, so they hold no seats."""
    edited = copy.deepcopy(cfg.electorates_cfg)
    edited["electorates"].append({"electorate": "Kenepuru", "party": "Mana", "p": 0.1})
    edited["coalitions"].append({"name": "National + United Future", "parties": ["National", "United Future"]})
    monkeypatch.setattr(cfg, "electorates_cfg", edited)
    a = cfg.assumptions(PARTIES)
    assert any(n.startswith("Mana is not tracked") for n in a.notes)
    assert ["National", "United Future"] in [c["parties"] for c in a.coalitions]


def test_an_electorate_can_override_its_party_defaults(cfg):
    entries = [{"electorate": "Waiariki", "party": "Te Pāti Māori", "p": 0.8, "slope": 0.1}]
    e = _check(cfg, electorates=entries).electorates[0]
    assert e["slope"] == 0.1 and e["at_share"] == 0.022        # own slope, the party's at_share


def test_seat_and_house_charts_do_not_zoom():
    """A drag on these charts must scroll the page on a phone; there is nothing to zoom into."""
    import numpy as np
    import polars as pl

    from pollofpolls.report import charts
    seats = charts.seats_figure(np.array([[60, 50], [58, 52]]), np.array([120, 120]), ["National", "Labour"],
                                [{"name": "National", "parties": ["National"]}], ncol=1)
    house = charts.house_effects_figure(pl.DataFrame({"house": ["A", "A"], "party": ["National", "Labour"],
                                                      "effect_pp": [1.0, -1.0]}), {})
    for fig in (seats, house):
        assert fig.layout.dragmode is False
        assert all(ax.fixedrange for ax in fig.select_xaxes()) and all(ax.fixedrange for ax in fig.select_yaxes())
