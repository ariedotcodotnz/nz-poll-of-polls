"""Turn the poll table and election results into the arrays the model consumes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ..config import Config, Election
from ..data.dates import week_start
from ..data.pollsters import PollsterMap
from .polls_table import eligible_polls, tracked_parties

SHARE_FLOOR = 0.002  # avoids log(0) in the log-ratio transform for parties absent at the anchor


@dataclass
class Dataset:
    parties: list[str]                 # K names, first is National (log-ratio reference), last is "Other"
    weeks: list[date]                  # T Sunday week starts
    y: np.ndarray                      # (N, K) pseudo-counts share * n (0 where unobserved)
    mask: np.ndarray                   # (N, K) bool, observed categories (Other always True)
    n: np.ndarray                      # (N,) sample sizes
    t: np.ndarray                      # (N,) week index
    pollster_idx: np.ndarray           # (N,)
    house_idx: np.ndarray              # (N,) pollster x method-segment unit
    cycle_idx: np.ndarray              # (N,)
    pc_idx: np.ndarray                 # (N,) pollster x cycle index (= pollster * C + cycle)
    cycle_frac: np.ndarray             # (N,) position of the poll within its cycle: 0 = last election, 1 = next
    round_unit: np.ndarray             # (N,) rounding unit of the published shares (0.01, 0.005 or 0.001)
    pollsters: list[str]
    houses: list[str]
    cycle_years: list[int]             # election year that ends each cycle; last is the target
    elections_t: np.ndarray            # (E,) week index of anchored (completed) elections
    elections_pi: np.ndarray           # (E, K)
    election_years: list[int]
    theta0: np.ndarray                 # (K-1,) log-ratio state at the anchor election (vs National)
    pi0: np.ndarray                    # (K,) shares at the anchor election
    campaign: np.ndarray               # (T,) 1.0 in campaign weeks
    target_year: int
    target_t: int
    pm_party_idx: int
    pm_prev_share: float
    fund_mean: float                   # fundamentals prior on the PM party's swing (no look-ahead)
    fund_sd: float
    fund_n: int
    cutoff: date
    last_data_t: int
    error_scale: np.ndarray | None = None   # (K,) multiplier on the election-day error scale, per party
    poll_ids: list[str] = field(default_factory=list)
    mid_dates: list[date] = field(default_factory=list)

    @property
    def K(self) -> int:
        return len(self.parties)

    @property
    def T(self) -> int:
        return len(self.weeks)

    @property
    def N(self) -> int:
        return len(self.n)

    @property
    def n_cycles(self) -> int:
        return len(self.cycle_years)

    @property
    def anchors_t(self) -> tuple[int, ...]:
        """Week indices where the state is known exactly: the anchor election and completed elections."""
        return (0,) + tuple(int(t) for t in self.elections_t)

    def fingerprint(self) -> str:
        """Hash of everything the models read. The cutoff date is excluded: it only selects polls, whose
        effect is already in the arrays, so an unchanged poll set does not trigger a refit."""
        h = hashlib.sha256()
        for k, v in sorted(self.__dict__.items()):
            if isinstance(v, np.ndarray):
                h.update(k.encode() + str(v.dtype).encode() + str(v.shape).encode() + np.ascontiguousarray(v).tobytes())
        meta = {k: v for k, v in self.__dict__.items() if not isinstance(v, np.ndarray) and k != "cutoff"}
        h.update(json.dumps(meta, default=str, sort_keys=True, ensure_ascii=False).encode())
        return h.hexdigest()

    def save(self, path: Path) -> None:
        arrays = {k: v for k, v in self.__dict__.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        meta = {k: v for k, v in self.__dict__.items() if not isinstance(v, np.ndarray)}
        meta["weeks"] = [d.isoformat() for d in self.weeks]
        meta["mid_dates"] = [d.isoformat() for d in self.mid_dates]
        meta["cutoff"] = self.cutoff.isoformat()
        path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))

    @classmethod
    def load(cls, path: Path) -> "Dataset":
        arrays = dict(np.load(path.with_suffix(".npz"), allow_pickle=False))
        meta = json.loads(path.with_suffix(".json").read_text())
        meta["weeks"] = [date.fromisoformat(d) for d in meta["weeks"]]
        meta["mid_dates"] = [date.fromisoformat(d) for d in meta["mid_dates"]]
        meta["cutoff"] = date.fromisoformat(meta["cutoff"])
        return cls(**arrays, **meta)


def alr(pi: np.ndarray, floor: float = SHARE_FLOOR) -> np.ndarray:
    """Additive log-ratio against the FIRST category (National), giving K-1 values for parties[1:]."""
    p = np.maximum(np.asarray(pi, dtype=float), floor)
    p = p / p.sum(-1, keepdims=True)
    return np.log(p[..., 1:]) - np.log(p[..., :1])


def alr_inverse(theta: np.ndarray) -> np.ndarray:
    """Inverse of :func:`alr`: (..., K-1) log-ratios -> (..., K) proportions."""
    full = np.concatenate([np.zeros(theta.shape[:-1] + (1,)), theta], axis=-1)
    full = full - full.max(-1, keepdims=True)
    e = np.exp(full)
    return e / e.sum(-1, keepdims=True)


def result_vector(result: dict[str, float], parties: list[str]) -> np.ndarray:
    tracked = [result.get(p, 0.0) for p in parties[:-1]]
    other = max(0.0, 1.0 - sum(tracked))
    return np.array(tracked + [other])


def rounding_unit(shares: list[float]) -> float:
    """Precision the pollster published at: whole percentages, halves, or one decimal place."""
    pct = np.asarray(shares, dtype=float) * 100.0
    if pct.size == 0:
        return 0.01
    if np.all(np.abs(pct - np.round(pct)) < 1e-6):
        return 0.01
    if np.all(np.abs(pct * 2 - np.round(pct * 2)) < 1e-6):
        return 0.005
    return 0.001


def pm_swings(results: dict[int, dict[str, float]], pm_by_year: dict[int, str], before: int) -> list[float]:
    """Change in the Prime Minister's party vote at each election before ``before`` (Ellis 2020, prior.R)."""
    years = sorted(y for y in results if y < before)
    swings = []
    for prev, year in zip(years[:-1], years[1:]):
        party = pm_by_year.get(year)
        if party and party in results[year] and party in results[prev]:
            swings.append(results[year][party] - results[prev][party])
    return swings


def fundamentals_prior(results: dict[int, dict[str, float]], pm_by_year: dict[int, str],
                       target_year: int) -> tuple[float, float, int]:
    """Mean and predictive sd of the PM party's swing, from elections strictly before the target."""
    s = np.array(pm_swings(results, pm_by_year, target_year))
    if s.size < 3:
        return 0.0, 0.1, int(s.size)
    sd = float(s.std(ddof=1))
    se = sd / np.sqrt(s.size)
    return float(s.mean()), float(np.sqrt(sd ** 2 + se ** 2)), int(s.size)


def party_error_scale(parties: list[str], recent: np.ndarray, best_past: np.ndarray, priors: dict,
                      threshold: float = 0.05, other: str = "Other") -> np.ndarray:
    """How much wider the election-day error is for each party, as a multiplier on the common scale.

    Polls miss small parties by much more in relative terms: over 2011-2023, parties polling under 8% in the
    final fortnight missed their result by about 21% of their own support, against 8% for larger parties. A party
    that has never cleared the threshold is treated the same way, however well it is polling, because how much of
    its stated support turns into votes has never been tested. ``recent`` is each party's support in the last
    polls before the cutoff and ``best_past`` its best result at a completed election, so neither looks ahead.
    """
    small = float(priors.get("minor_party_share", 0.0))
    factor = float(priors.get("minor_party_factor", 1.0))
    scale = np.ones(len(parties))
    if factor == 1.0:
        return scale
    for k, party in enumerate(parties):
        if party == other:                                  # a residual category, not a party
            continue
        if recent[k] < small or best_past[k] < threshold:
            scale[k] = factor
    return scale


def build_dataset(polls: pl.DataFrame, results: dict[int, dict[str, float]], cfg: Config,
                  target_year: int, cutoff: date | None = None, lagged: bool = False) -> Dataset:
    """Marshal everything for forecasting ``target_year`` with the polls available on ``cutoff``.

    ``cutoff`` defaults to today for the forecast election. Backtests pass ``lagged=True`` so a poll only
    counts once it would typically have been published (fieldwork end + the pollster's publication delay).
    """
    elections = [e for e in cfg.elections if cfg.anchor_election <= e.year <= target_year]
    if len(elections) < 2:
        raise ValueError("need at least the anchor election and the target election")
    target: Election = elections[-1]
    cutoff = cutoff or (target.date if not target.forecast else date.today())
    if cutoff > target.date:
        cutoff = target.date
    anchor = elections[0]
    week0 = week_start(anchor.date)
    target_t = (week_start(target.date) - week0).days // 7
    weeks = [week0 + timedelta(weeks=i) for i in range(target_t + 1)]
    T = len(weeks)
    election_t = [(week_start(e.date) - week0).days // 7 for e in elections]

    parties = tracked_parties(polls, cfg, target_year, cutoff, lagged) + ["Other"]
    if parties[0] != "National":
        raise ValueError("National must be the first tracked party (log-ratio reference)")
    K = len(parties)
    pidx = {p: i for i, p in enumerate(parties)}

    df = polls.filter((pl.col("mid_date") > anchor.date) & (pl.col("cycle") <= target_year))
    df = eligible_polls(df, cutoff, lagged, PollsterMap(cfg.pollsters_cfg).min_polls)
    poll_ids = df["poll_id"].unique(maintain_order=True).to_list()
    attrs = (df.group_by("poll_id", maintain_order=True)
               .agg(pl.col("pollster").first(), pl.col("segment").first(), pl.col("mid_date").first(),
                    pl.col("sample_size").first(), pl.col("cycle").first()))
    attrs = {r["poll_id"]: r for r in attrs.to_dicts()}
    pollsters = sorted(df["pollster"].unique().to_list())
    pol_i = {p: i for i, p in enumerate(pollsters)}
    house_labels = sorted({(a["pollster"], a["segment"]) for a in attrs.values()})
    house_i = {h: i for i, h in enumerate(house_labels)}
    cycle_years = [e.year for e in elections[1:]]
    cyc_i = {y: i for i, y in enumerate(cycle_years)}
    C = len(cycle_years)

    N = len(poll_ids)
    y = np.zeros((N, K)); mask = np.zeros((N, K), dtype=bool)
    n = np.zeros(N); t = np.zeros(N, dtype=int)
    pollster_idx = np.zeros(N, dtype=int); house_idx = np.zeros(N, dtype=int)
    cycle_idx = np.zeros(N, dtype=int); pc_idx = np.zeros(N, dtype=int)
    cycle_frac = np.zeros(N); round_unit = np.zeros(N)
    mid_dates = []
    shares_by_poll: dict[str, dict[str, float]] = {pid: {} for pid in poll_ids}
    for r in df.select(["poll_id", "party", "share"]).iter_rows():
        shares_by_poll[r[0]][r[1]] = r[2]
    for i, pid in enumerate(poll_ids):
        a = attrs[pid]
        n[i] = a["sample_size"]
        wk = (week_start(a["mid_date"]) - week0).days // 7
        t[i] = min(max(wk, 0), T - 1)
        pollster_idx[i] = pol_i[a["pollster"]]
        house_idx[i] = house_i[(a["pollster"], a["segment"])]
        c = cyc_i[a["cycle"]]
        cycle_idx[i] = c
        pc_idx[i] = pollster_idx[i] * C + c
        start, end = election_t[c], election_t[c + 1]
        cycle_frac[i] = float(np.clip((t[i] - start) / max(end - start, 1), 0.0, 1.0))
        round_unit[i] = rounding_unit(list(shares_by_poll[pid].values()))
        mid_dates.append(a["mid_date"])
        observed_total = 0.0
        for party, share in shares_by_poll[pid].items():
            if party in pidx:
                k = pidx[party]
                y[i, k] = share * n[i]; mask[i, k] = True
                observed_total += share
        y[i, K - 1] = max(0.0, 1.0 - observed_total) * n[i]
        mask[i, K - 1] = True

    # completed elections inside the window (excluding the anchor, which sets theta0)
    anchored = [e for e in elections[1:] if e.date <= cutoff and e.year in results and not e.forecast]
    elections_t = np.array([(week_start(e.date) - week0).days // 7 for e in anchored], dtype=int)
    elections_pi = np.array([result_vector(results[e.year], parties) for e in anchored]).reshape(len(anchored), K)
    pi0 = result_vector(results[anchor.year], parties)

    campaign = np.zeros(T)
    for te in election_t[1:]:
        campaign[max(0, te - cfg.campaign_weeks + 1): te + 1] = 1.0

    prev = elections[-2]
    pm_prev_share = float(results[prev.year].get(target.pm_party, 0.0)) if prev.year in results else 0.0
    fund_mean, fund_sd, fund_n = fundamentals_prior(results, cfg.pm_by_year, target_year)
    last_data_t = int(t.max()) if N else 0
    recent = np.zeros(len(parties))
    if N:                                                   # support in the last eight weeks of polls
        late = t >= max(0, last_data_t - 8)
        seen = mask[late] & (n[late, None] > 0)
        shares = np.where(seen, y[late] / np.maximum(n[late, None], 1.0), np.nan)
        with np.errstate(invalid="ignore"):
            recent = np.nan_to_num(np.nanmean(shares, axis=0))
    # The anchor is known history even when no later election has been completed.
    best_past = np.vstack([pi0, elections_pi]).max(axis=0)
    error_scale = party_error_scale(parties, recent, best_past, cfg.priors)
    return Dataset(
        parties=parties, weeks=weeks, y=y, mask=mask, n=n, t=t, pollster_idx=pollster_idx,
        house_idx=house_idx, cycle_idx=cycle_idx, pc_idx=pc_idx, cycle_frac=cycle_frac, round_unit=round_unit,
        pollsters=pollsters, houses=[f"{p}" + (f" (segment {s})" if s else "") for p, s in house_labels],
        cycle_years=cycle_years, elections_t=elections_t, elections_pi=elections_pi,
        election_years=[e.year for e in anchored], theta0=alr(pi0), pi0=pi0, campaign=campaign,
        target_year=target_year, target_t=target_t, pm_party_idx=pidx.get(target.pm_party, 0),
        pm_prev_share=pm_prev_share, fund_mean=fund_mean, fund_sd=fund_sd, fund_n=fund_n,
        cutoff=cutoff, last_data_t=last_data_t, error_scale=error_scale, poll_ids=poll_ids, mid_dates=mid_dates,
    )
