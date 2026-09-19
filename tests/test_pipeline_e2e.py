"""prep -> forecast -> report on the fixture pages, with synthetic posterior draws instead of MCMC."""
import json
import shutil

import numpy as np

from pollofpolls.config import Config
from pollofpolls.model.fit import QUANTILES, FitResult
from pollofpolls.pipeline import ensemble_weights, forecast, prep
from pollofpolls.prep.marshal import alr, build_dataset
from pollofpolls.report.render import render_report

YEARS = [2011, 2014, 2017, 2020, 2023, 2026]


def _fake_fit(ds, name, rng):
    """Draws around a straight-line path from the anchor result, shaped like a real FitResult."""
    T, K = ds.T, ds.K
    end = np.full(K, 1.0 / K)
    base = np.linspace(ds.pi0, end, T)
    base = base / base.sum(1, keepdims=True)
    S = 200
    noise = rng.normal(0, 0.05, size=(S, 1, K - 1))
    theta = alr(base)[None, :, :] + noise
    full = np.concatenate([np.zeros((S, T, 1)), theta], axis=-1)
    pi = np.exp(full) / np.exp(full).sum(-1, keepdims=True)
    hyper = {"house_base": rng.normal(0, 0.05, size=(2, 50, len(ds.houses), K))}
    return FitResult(variant=name, target_year=ds.target_year, parties=ds.parties,
                     weeks=[w.isoformat() for w in ds.weeks], pi_mean=pi.mean(0),
                     pi_q=np.quantile(pi, QUANTILES, axis=0).transpose(1, 2, 0), pi_thin=pi,
                     pi_last=pi[:, ds.last_data_t], pi_target=pi[:, ds.target_t],
                     theta_last=theta[:, ds.last_data_t], last_data_t=ds.last_data_t, target_t=ds.target_t,
                     diagnostics={"max_rhat": 1.0, "min_ess_bulk": 400.0, "n_divergent": 0, "seconds": 1.0,
                                  "mean_num_steps": 15.0}, hyper=hyper)


def test_prep_forecast_report(tmp_path, root, fixtures):
    shutil.copytree(root / "config", tmp_path / "config")
    shutil.copytree(root / "data" / "reference", tmp_path / "data" / "reference")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    raw = tmp_path / "data" / "raw" / "wikipedia"
    raw.mkdir(parents=True)
    for y in YEARS:
        shutil.copy(fixtures / f"{y}_party_vote.html", raw / f"{y}.html")
    # a hostile edit to the polling table: an unknown pollster whose name tries to close the script element
    evil = "Evil&lt;/script&gt;&lt;script&gt;alert(1)&lt;/script&gt;"
    page = (raw / "2026.html").read_text(encoding="utf-8")
    assert page.count(">Talbot Mills<") >= 2
    (raw / "2026.html").write_text(page.replace(">Talbot Mills<", f">{evil}<", 2), encoding="utf-8")
    cfg = Config(tmp_path)
    assert cfg.paths.root == tmp_path

    table, results = prep(cfg)
    assert (tmp_path / "data/processed/polls.parquet").exists()
    ds = build_dataset(table, results, cfg, 2026)
    ds.save(cfg.paths.processed / "dataset_2026")
    weights = ensemble_weights(cfg)                        # no backtests here -> equal weights
    assert set(weights) == set(cfg.ensemble) and abs(sum(weights.values()) - 1) < 1e-9
    rng = np.random.default_rng(0)
    for v in weights:
        _fake_fit(ds, v, rng).save(cfg.paths.processed / f"fit_{v}_2026")

    summary = forecast(cfg)
    assert summary["parties"][0] == "National"
    assert abs(sum(r["mean"] for r in summary["election_day"].values()) - 1) < 1e-6
    assert all(0 <= c["p_majority"] <= 1 for c in summary["coalitions_election_day"])
    assert summary["expected_house_size"] >= 120
    assert any("TOP" in c["parties"] for c in summary["coalitions_election_day"])
    for r in summary["balance_of_power"]["election_day"]:
        assert set(r["p_with"]) == {"NZ First", "TOP"}
        assert abs(r["p_alone"] + r["p_any_one"] + r["p_needs_all"] + r["p_short"] - 1) < 1e-9
    for f in ["forecast.csv", "seats.csv", "coalitions.csv", "trend.csv", "house_effects.csv", "summary.json"]:
        assert (tmp_path / "output" / f).exists(), f

    html = render_report(cfg).read_text()
    assert "NZ Poll of Polls 2026" in html and "Plotly.newPlot" in html
    assert "<script>alert(1)" not in html and "</script><script>" not in html
    assert "Evil" in html                                  # the label is shown, safely escaped
    assert "Who holds the balance of power" in html and "Enough with TOP" in html
    for f in ["voting_intention620.svg", "voting_intention375.svg", "election_night620.svg", "saturday375.svg"]:
        assert (tmp_path / "output" / f).stat().st_size > 1000
    assert json.loads((tmp_path / "site" / "summary.json").read_text())["election_date"] == "2026-11-07"
