"""Map raw Wikipedia pollster strings to canonical pollsters, with metadata."""

from __future__ import annotations

import re
from datetime import date


class PollsterMap:
    def __init__(self, cfg: dict):
        self.defaults = cfg.get("defaults", {})
        self.exclude = [re.compile(p, re.I) for p in cfg.get("exclude_patterns", [])]
        self.entries = []
        for e in cfg.get("pollsters", []):
            changes = []
            for c in e.get("method_changes", []) or []:
                changes.append(c if isinstance(c, date) else date.fromisoformat(str(c)))
            self.entries.append({
                "name": e["name"],
                "match": re.compile(e["match"], re.I),
                "sample_size": int(e.get("sample_size", self.defaults.get("sample_size", 1000))),
                "polling_code": bool(e.get("polling_code", False)),
                "method_changes": sorted(changes),
                "publication_lag_days": int(e.get("publication_lag_days",
                                                  self.defaults.get("publication_lag_days", 5))),
            })
        self.by_name = {e["name"]: e for e in self.entries}

    def is_excluded(self, raw: str) -> bool:
        return any(p.search(raw) for p in self.exclude)

    def canonical(self, raw: str) -> str | None:
        """Canonical name, or None for sponsored/internal releases that are excluded."""
        if self.is_excluded(raw):
            return None
        for e in self.entries:
            if e["match"].search(raw):
                return e["name"]
        return raw  # unknown pollster: keep as-is; min_polls will drop true one-offs

    def sample_size(self, name: str) -> int:
        e = self.by_name.get(name)
        return e["sample_size"] if e else int(self.defaults.get("sample_size", 1000))

    def publication_lag(self, name: str) -> int:
        e = self.by_name.get(name)
        return e["publication_lag_days"] if e else int(self.defaults.get("publication_lag_days", 5))

    def polling_code(self, name: str) -> bool:
        e = self.by_name.get(name)
        return bool(e and e["polling_code"])

    def segment(self, name: str, when: date) -> int:
        """Method segment index: 0 before the first recorded method change, 1 after it, etc."""
        e = self.by_name.get(name)
        if not e:
            return 0
        return sum(1 for c in e["method_changes"] if when >= c)

    @property
    def min_polls(self) -> int:
        return int(self.defaults.get("min_polls", 2))

    @property
    def polling_code_only(self) -> bool:
        return bool(self.defaults.get("polling_code_only", False))
