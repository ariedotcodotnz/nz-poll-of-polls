"""Leave-one-election-out backtests of the official-count forecast, against rules of thumb.

For each past election, the models are refitted on the other elections and every seat's official result is
forecast from its election-night count (or, where that count is not archived, from its official ordinary
votes). The forecast election-wide shift is a fresh draw, exactly as for the live forecast. The national
special-vote volume is given with the error of an election-night estimate.

Rules of thumb, each given the same special-vote volume:

    election night   specials split exactly like the starting count
    seat's last      specials differ from the starting count as they did in the seat's previous election
    average          specials differ from the starting count by the average of every past seat and election

Each rule of thumb becomes a forecast distribution with a normal error whose sd is its root-mean-square error on
the training elections, so all methods are scored on the same terms: CRPS of the final margin (votes), absolute
error, interval coverage, and the Brier score and log loss of the winner.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..eval.scoring import crps
from . import parliament as parl
from . import partyvote as pv
from .data import ElectorateCount
from .model import (
    Posterior,
    build_official_count_data,
    build_specials_data,
    fit_official_count,
    fit_specials,
    log_ratio,
    softmax0,
)
from .national import NationalHistory
from .simulate import Target, VolumeShift, simulate

METHODS = ["model", "election_night", "seat_last", "average"]
METHOD_LABELS = {"model": "This model", "election_night": "Specials split like election night",
                 "seat_last": "Seat's swing at the last election", "average": "Average swing of past seats"}


def _start(c: ElectorateCount, pair: list[str]) -> tuple[np.ndarray, bool]:
    if c.preliminary is not None:
        return c.grouped(c.preliminary, pair), True
    return c.grouped(c.ordinary, pair), False


def _pooled_volume(counts: list[ElectorateCount], year: int, pair: list[str]) -> tuple[float, float]:
    """Log specials per ordinary vote and logit of the overseas share, pooled over the seats of ``year``."""
    cs = [c for c in counts if c.year == year and c.grouped(c.ordinary, pair) is not None]
    s = sum(c.special_total.sum() for c in cs)
    o = sum(c.ordinary.sum() for c in cs)
    ov = sum(c.specials["overseas"].sum() for c in cs)
    return float(np.log(s / o)), float(np.log(ov / (s - ov)))


def _shift(c: ElectorateCount, pair: list[str]) -> np.ndarray:
    """How a seat's specials differed from its ordinary vote, on the log-ratio scale."""
    return log_ratio(c.grouped(c.special_total, pair)) - log_ratio(c.grouped(c.ordinary, pair))


def _rule(method: str, start: np.ndarray, volume: float, c: ElectorateCount, train: list[ElectorateCount],
          pair: list[str]) -> float | None:
    """A rule of thumb's final margin (pair[1] - pair[0]) from ``start`` and ``volume`` special votes."""
    if method == "election_night":
        d = np.zeros(2)
    elif method == "average":
        d = np.mean([_shift(t, pair) for t in train], axis=0)
    else:
        past = [t for t in train if t.key == c.key and t.year != c.year]
        if not past:
            return None
        before = [t for t in past if t.year < c.year]
        d = _shift(max(before, key=lambda t: t.year) if before else min(past, key=lambda t: t.year), pair)
    final = start + volume * softmax0(log_ratio(start) + d)
    return float(final[1] - final[0])


def _rule_sd(method: str, train: list[ElectorateCount], pair: list[str]) -> float:
    """Root-mean-square margin error of a rule of thumb on the training seats, given their actual volume."""
    errs = []
    for c in train:
        start, _ = _start(c, pair)
        pred = _rule(method, start, c.special_total.sum(), c, train, pair)
        if pred is not None:
            final = c.grouped(c.total, pair)
            errs.append(pred - (final[1] - final[0]))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def _scores(draws: np.ndarray, actual: float) -> dict:
    lo50, hi50, lo90, hi90 = np.quantile(draws, [0.25, 0.75, 0.05, 0.95])
    p = float(np.clip((draws > 0).mean(), 0.005, 0.995))
    won = actual > 0
    return {"crps_votes": crps(draws, actual), "abs_error": float(abs(np.median(draws) - actual)),
            "cover50": bool(lo50 <= actual <= hi50), "cover90": bool(lo90 <= actual <= hi90),
            "p_second": p, "brier": float((p - won) ** 2), "log_loss": float(-np.log(p if won else 1 - p)),
            "median": float(np.median(draws))}


def backtest(counts: list[ElectorateCount], pair: list[str], priors: dict, mcmc: dict, *, n: int = 8000,
             volume_error_sd: float = 0.04, overseas_error_sd: float = 0.10, df: float | None = 4,
             seed: int = 0, log=print, link=None) -> list[dict]:
    """One row per seat, past election and method.

    ``link(target_year, n, rng)`` gives the national party-vote lean without the target election:
    ({training year: (2,)}, draws (n, 2) for the target), or None when the party-vote history lacks that year.
    """
    usable = [c for c in counts if c.official and c.grouped(c.ordinary, pair) is not None]
    years = sorted({c.year for c in usable})
    rows: list[dict] = []
    for target_year in years:
        train = [c for c in usable if c.year != target_year]
        test = [c for c in usable if c.year == target_year]
        if len({c.year for c in train}) < 2:
            continue
        rng = np.random.default_rng(seed + target_year)
        linked = link(target_year, n, rng) if link is not None else None
        data = build_specials_data(train, pair, linked[0] if linked else None)
        post = fit_specials(data, priors, mcmc)
        oc = build_official_count_data(data)
        official: Posterior | None = fit_official_count(oc, priors, mcmc) if len(oc.prelim) else None
        earlier = [y for y in data.years if y < target_year]
        ref = max(earlier) if earlier else min(data.years)
        v_t, o_t = _pooled_volume(usable, target_year, pair)
        v_r, o_r = _pooled_volume(usable, ref, pair)
        volume = VolumeShift(ref, v_t - v_r + rng.normal(0, volume_error_sd, n),
                             o_t - o_r + rng.normal(0, overseas_error_sd, n))
        targets = [Target(c.electorate, c.key, *_start(c, pair)) for c in test]
        sim = simulate(post, data, targets, volume, official=official, df=df, n=n, seed=seed + target_year,
                       national=linked[1] if data.national is not None else None,
                       max_addition=float(priors["official_count"].get("max_addition", 0.2)))
        log(f"backtest {target_year}: fitted on {data.years} (divergences {post.diagnostics['n_divergent']}"
            f"{'' if official is None else ', ' + str(official.diagnostics['n_divergent'])})")
        sds = {m: _rule_sd(m, train, pair) for m in METHODS[1:]}
        for j, c in enumerate(test):
            start, from_prelim = targets[j].start, targets[j].official_change
            final = c.grouped(c.total, pair)
            actual = float(final[1] - final[0])
            spec_actual = c.grouped(c.special_total, pair)
            base = {"year": target_year, "electorate": c.electorate, "start": "election night" if from_prelim
                    else "official ordinary", "start_margin": float(start[1] - start[0]), "actual_margin": actual,
                    "actual_special_share": float(spec_actual[1] / spec_actual[:2].sum())}
            margin = sim.final[:, j, 1] - sim.final[:, j, 0]
            spec = sim.specials[:, j].sum(1)
            share = spec[:, 1] / np.maximum(spec[:, 0] + spec[:, 1], 1)
            rows.append({**base, "method": "model", **_scores(margin, actual),
                         "crps_special_share_pp": crps(share * 100, base["actual_special_share"] * 100)})
            # the rules of thumb get the special-vote volume the model expects
            volume_mean = float(sim.specials[:, j].sum((1, 2)).mean())
            for m in METHODS[1:]:
                point = _rule(m, start, volume_mean, c, train, pair)
                if point is None or not np.isfinite(sds[m]):
                    continue
                draws = stats.norm(point, sds[m]).rvs(size=n, random_state=rng)
                rows.append({**base, "method": m, **_scores(draws, actual), "crps_special_share_pp": None})
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    """Average scores per method over the seats every method could forecast."""
    cases = {}
    for r in rows:
        cases.setdefault((r["year"], r["electorate"]), set()).add(r["method"])
    common = {k for k, ms in cases.items() if set(METHODS) <= ms}
    out = []
    for m in METHODS:
        rs = [r for r in rows if r["method"] == m and (r["year"], r["electorate"]) in common]
        if not rs:
            continue
        flips = [r for r in rs if np.sign(r["start_margin"]) != np.sign(r["actual_margin"])]
        out.append({"method": m, "label": METHOD_LABELS[m], "n_cases": len(rs),
                    "crps_votes": float(np.mean([r["crps_votes"] for r in rs])),
                    "mae_votes": float(np.mean([r["abs_error"] for r in rs])),
                    "coverage_50": float(np.mean([r["cover50"] for r in rs])),
                    "coverage_90": float(np.mean([r["cover90"] for r in rs])),
                    "brier": float(np.mean([r["brier"] for r in rs])),
                    "log_loss": float(np.mean([r["log_loss"] for r in rs])),
                    "n_flipped": len(flips)})
    return out


# ------------------------------------------------------------------------------------------ party vote
PARTY_METHODS = {"model": "This model", "election_night": "Specials split like election night",
                 "average": "Average lean of past elections"}


def split_small_winners(votes: np.ndarray, parties: list[str], small: list[dict], which: str,
                        reference: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Votes (..., K) with each small electorate-winning party taken out of "Other" into its own column.

    Each gets the share of "Other" it had in ``reference`` (K,), a count where its own votes ``small[which]`` are
    known; for that count itself the split is exact.
    """
    o = parties.index("Other")
    v = np.asarray(votes, float)
    if not small:
        return v, list(parties)
    cols = [v[..., o] * (m[which] / reference[o]) for m in small]
    rest = v.copy()
    rest[..., o] = v[..., o] - sum(cols)
    return np.concatenate([rest, np.stack(cols, axis=-1)], axis=-1), [*parties, *(m["party"] for m in small)]


def backtest_party_vote(h: NationalHistory, second: str, priors: dict, mcmc: dict, *,
                        electorates: dict[int, np.ndarray] | None = None,
                        small: dict[int, list[dict]] | None = None, n: int = 8000, df: float | None = 4,
                        volume_error_sd: float = 0.04, overseas_error_sd: float = 0.10, nseats: int = 120,
                        threshold: float = 0.05, seed: int = 0, log=print) -> list[dict]:
    """Leave one election out: forecast its official party vote from its election-night count (or, where that
    is not archived, from its official ordinary votes), with the model refitted on the other elections. One row
    per party and election, with two rules of thumb and, where the electorates won are on file, the seats.
    ``small`` (`national.small_winners`) names the parties pooled as "Other" that won an electorate: they are
    allocated seats in their own right, as they were."""
    rows: list[dict] = []
    for t, year in enumerate(h.years):
        train = [y for y in h.years if y != year]
        if len(train) < 2:
            continue
        d = pv.build_party_vote_data(h, second, train)
        post = pv.fit_party_vote(d, priors, mcmc)
        rng = np.random.default_rng(seed + year)
        from_prelim = not np.isnan(h.prelim[t]).any()
        start = h.prelim[t] if from_prelim else h.ordinary[t]
        change = pv.ordinary_change(h.subset(train)) if from_prelim else None
        lean = pv.draw_lean(post, d, n, rng, df=df)
        by_cat = h.specials[t].sum(1)                                        # [overseas, domestic]
        total = by_cat.sum() * np.exp(rng.normal(0, volume_error_sd, n))
        overseas = np.minimum(by_cat[0] * np.exp(rng.normal(0, overseas_error_sd, n)), 0.9 * total)
        sim = pv.simulate_party_votes(start, lean, total, overseas, rng, change=change)
        share = 100 * sim.final / sim.final.sum(1, keepdims=True)
        actual = 100 * h.final[t] / h.final[t].sum()
        night = 100 * start / start.sum()
        # the average rule: every party's specials lean as the average of the training elections
        avg = np.where(d.stood[:, None, :], d.y, np.nan)
        avg = np.nan_to_num(np.nanmean(avg, axis=0))                         # (C, K-1)
        rule = pv.simulate_party_votes(start, avg[None], by_cat.sum()[None], by_cat[:1], np.random.default_rng(0),
                                       change={**change, "growth_sd": 0.0,
                                               "composition_sd": [0.0] * (len(start) - 1)} if change else None)
        average = 100 * rule.final[0] / rule.final[0].sum()
        seats = None
        if electorates is not None and year in electorates:
            mine = (small or {}).get(year, [])
            e = np.concatenate([electorates[year], [m["electorates"] for m in mine]])
            e[h.parties.index("Other")] -= sum(m["electorates"] for m in mine)
            v_sim, names = split_small_winners(sim.final, h.parties, mine, "start", start)
            v_start, _ = split_small_winners(start, h.parties, mine, "start", start)
            v_final, _ = split_small_winners(h.final[t], h.parties, mine, "final", h.final[t])
            K = len(h.parties)
            seats = (parl.elect(v_sim, np.broadcast_to(e, v_sim.shape), names, nseats, threshold).seats[:, :K],
                     parl.elect(v_start, e, names, nseats, threshold).seats[0, :K],
                     parl.elect(v_final, e, names, nseats, threshold).seats[0, :K])
        log(f"party-vote backtest {year}: fitted on {train} (divergences {post.diagnostics['n_divergent']})")
        for k, party in enumerate(h.parties):
            if start[k] < pv.MIN_VOTES:
                continue
            lo50, hi50, lo90, hi90 = np.quantile(share[:, k], [0.25, 0.75, 0.05, 0.95])
            row = {"year": year, "party": party, "start": "election night" if from_prelim else "official ordinary",
                   "start_share": float(night[k]), "actual_share": float(actual[k]),
                   "model_median": float(np.median(share[:, k])), "model_q05": float(lo90), "model_q95": float(hi90),
                   "cover50": bool(lo50 <= actual[k] <= hi50), "cover90": bool(lo90 <= actual[k] <= hi90),
                   "crps_pp": crps(share[:, k], float(actual[k])),
                   "abs_error_model": float(abs(np.median(share[:, k]) - actual[k])),
                   "abs_error_election_night": float(abs(night[k] - actual[k])),
                   "abs_error_average": float(abs(average[k] - actual[k]))}
            if seats is not None and (seats[1][k] or seats[2][k]):
                sims, at_start, final = seats
                values, counts = np.unique(sims[:, k], return_counts=True)
                row |= {"seats_start": int(at_start[k]), "seats_actual": int(final[k]),
                        "seats_model": int(values[counts.argmax()]),
                        "p_seats_actual": float((sims[:, k] == final[k]).mean())}
            rows.append(row)
    return rows


def summarise_party_vote(rows: list[dict]) -> dict:
    """Mean absolute error of each method (percentage points of the party vote), the model's interval coverage,
    and how often each called the final number of seats."""
    if not rows:
        return {}
    seat_rows = [r for r in rows if "seats_actual" in r]
    changed = [r for r in seat_rows if r["seats_actual"] != r["seats_start"]]
    out = {"n_cases": len(rows), "years": sorted({r["year"] for r in rows}),
           "mae_pp": {m: float(np.mean([r[f"abs_error_{m}"] for r in rows])) for m in PARTY_METHODS},
           "labels": PARTY_METHODS,
           "coverage_50": float(np.mean([r["cover50"] for r in rows])),
           "coverage_90": float(np.mean([r["cover90"] for r in rows])),
           "crps_pp": float(np.mean([r["crps_pp"] for r in rows]))}
    if seat_rows:
        out["seats"] = {"n_cases": len(seat_rows), "n_changed": len(changed),
                        "model_right": float(np.mean([r["seats_model"] == r["seats_actual"] for r in seat_rows])),
                        "election_night_right": float(np.mean([r["seats_start"] == r["seats_actual"]
                                                               for r in seat_rows])),
                        "mean_p_actual": float(np.mean([r["p_seats_actual"] for r in seat_rows])),
                        "mean_p_actual_when_changed": (float(np.mean([r["p_seats_actual"] for r in changed]))
                                                       if changed else None)}
    return out
