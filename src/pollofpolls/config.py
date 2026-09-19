"""Configuration loading: paths, election calendar, model variants."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from functools import cached_property
from pathlib import Path

import yaml


def find_root(start: Path | None = None) -> Path:
    """Repository root: $POLLOFPOLLS_ROOT, else the first ancestor with pyproject.toml and config/."""
    env = os.environ.get("POLLOFPOLLS_ROOT")
    if env:
        return Path(env).resolve()
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "pyproject.toml").exists() and (cand / "config").is_dir():
            return cand
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw" / "wikipedia"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def reference(self) -> Path:
        return self.root / "data" / "reference"

    @property
    def output(self) -> Path:
        return self.root / "output"

    @property
    def site(self) -> Path:
        return self.root / "site"


@dataclass(frozen=True)
class Election:
    year: int
    date: date
    pm_party: str
    forecast: bool = False
    wikipedia_page: str | None = None   # override for the polling article's title


class Config:
    """All YAML configuration, loaded once."""

    def __init__(self, root: Path | None = None):
        self.paths = Paths(find_root(root))
        self.elections_cfg = self._load("elections.yml")
        self.pollsters_cfg = self._load("pollsters.yml")
        self.electorates_cfg = self._load("electorates.yml")
        self.model_cfg = self._load("model.yml")

    def _load(self, name: str) -> dict:
        with open(self.paths.config / name, encoding="utf-8") as f:
            return yaml.safe_load(f)

    @cached_property
    def elections(self) -> list[Election]:
        out = []
        for e in self.elections_cfg["elections"]:
            d = e["date"] if isinstance(e["date"], date) else date.fromisoformat(str(e["date"]))
            out.append(Election(int(e["year"]), d, e["pm_party"], bool(e.get("forecast", False)),
                                e.get("wikipedia_page")))
        return sorted(out, key=lambda e: e.date)

    def election(self, year: int) -> Election:
        for e in self.elections:
            if e.year == year:
                return e
        raise KeyError(year)

    @property
    def forecast_election(self) -> Election:
        return next(e for e in self.elections if e.forecast)

    @property
    def pm_by_year(self) -> dict[int, str]:
        """Party of the Prime Minister going into each election (history + calendar)."""
        out = {int(e["year"]): e["pm_party"] for e in self.elections_cfg.get("pm_history", [])}
        out.update({e.year: e.pm_party for e in self.elections})
        return out

    @property
    def polling_pages(self) -> list[tuple[int, str | None]]:
        """(year, article title override) for every polling page the model reads: anchor to forecast election."""
        return [(e.year, e.wikipedia_page) for e in self.elections if e.year >= self.anchor_election]

    @property
    def anchor_election(self) -> int:
        return int(self.elections_cfg.get("anchor_election", 2011))

    @property
    def campaign_weeks(self) -> int:
        return int(self.elections_cfg.get("campaign_weeks", 8))

    def in_parliament(self, cycle_year: int) -> list[str]:
        return list(self.elections_cfg.get("in_parliament", {}).get(cycle_year, []))

    @property
    def auto_track(self) -> dict:
        return dict(self.elections_cfg.get("auto_track", {"min_share": 0.025, "min_polls": 4}))

    @property
    def colours(self) -> dict[str, str]:
        return dict(self.elections_cfg.get("colours", {}))

    def variant(self, name: str) -> dict:
        v = dict(self.model_cfg["variants"][name])
        v["name"] = name
        return v

    @property
    def default_variant(self) -> str:
        return self.model_cfg.get("default_variant", "base")

    @property
    def ensemble(self) -> list[str]:
        return list(self.model_cfg.get("ensemble", [self.default_variant]))

    @property
    def priors(self) -> dict:
        return dict(self.model_cfg["priors"])

    @property
    def mcmc(self) -> dict:
        return dict(self.model_cfg["mcmc"])
