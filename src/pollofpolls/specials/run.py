"""The ``pollofpolls specials`` stage: forecast the official count of the close seats, and backtest the method.

Writes output/specials/:

    forecast.json     everything the website shows
    seats.csv         one row per seat: election-night count and the forecast official result
    candidates.csv    one row per candidate
    history.csv       every past seat and election: how its specials differed from its ordinary votes
    backtest.csv      leave-one-election-out forecasts of past seats, per method
    backtest_summary.csv
    margins.npz       simulated final margins and special-vote shares, for the charts
    margin_draws.csv  5,000 of those margins per seat, for charts drawn outside Python
    party_votes.csv   the country's party vote: election night and the forecast official count, with seats
    routes.csv        how simulated counts changed the election-night House, and the minimal routes to a
                      change of majority
    party_vote_history.csv   past elections' ordinary and special party votes, as the model sees them
    party_vote_backtest.csv  leave-one-election-out forecasts of past elections' party votes

and, for each contest between another pair of parties (config/specials.yml's ``other_contests``),
history_<pair>.csv, margins_<pair>.npz, margin_draws_<pair>.csv and backtest_<pair>.csv, with its seats added
to forecast.json (``contests``), seats.csv and candidates.csv.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from ..cache import StageCache
from ..config import Config
from . import national as natdata
from . import parliament as parl
from . import partyvote as pv
from .backtest import backtest, backtest_party_vote, summarise, summarise_party_vote
from .data import (
    CATEGORIES,
    ElectorateCount,
    candidate_party,
    electorate_key,
    load_history,
    read_preliminary_xlsx,
)
from .model import (
    CATS,
    Posterior,
    SpecialsData,
    build_official_count_data,
    build_specials_data,
    fit_official_count,
    fit_specials,
    log_ratio,
    softmax0,
)
from .simulate import Simulation, Target, VolumeShift, simulate

OUT = "specials"
ELECTORATE_SEATS = Path("data") / "reference" / "electorate_seats.csv"     # electorates won at past elections
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def _log(msg: str) -> None:
    print(f"[pollofpolls] {msg}", flush=True)


def load_config(cfg: Config) -> dict:
    with open(cfg.paths.config / "specials.yml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Contest:
    """Close seats between another pair of parties, fitted on their own past results (``other_contests``)."""
    pair: list[str]
    title: str
    history: list[ElectorateCount]
    seats: list[ElectorateCount]
    keys: list[str]                 # per seat, the electorate key whose history stands for it
    data_dir: Path
    cfg: dict

    @property
    def slug(self) -> str:
        return "_".join(electorate_key(p).replace(" ", "_") for p in self.pair)


@dataclass
class Inputs:
    scfg: dict
    history: list[ElectorateCount]
    seats: list[ElectorateCount]
    data_dir: Path
    contests: list[Contest] = field(default_factory=list)


def count_from_config(e: dict, year: int) -> ElectorateCount:
    """An election-night count given in the configuration: ``candidates`` maps each name to {party, votes}."""
    names = list(e["candidates"])
    votes = np.array([float(e["candidates"][c]["votes"]) for c in names])
    if "valid_votes" in e and votes.sum() != e["valid_votes"]:
        raise ValueError(f"config/specials.yml: {e['name']}'s candidates have {votes.sum():.0f} votes, "
                         f"not the {e['valid_votes']} valid votes given")
    zero = np.zeros(len(names))
    return ElectorateCount(
        year=year, electorate=e["name"], number=e.get("number"), candidates=names,
        parties=[candidate_party(e["candidates"][c]["party"]) for c in names], advance=zero, election_day=zero,
        ordinary=votes.copy(), specials={}, total=votes.copy(), informal=e.get("informal"), official=False,
        preliminary=votes.copy(), preliminary_informal=e.get("informal"), source="config/specials.yml")


def _grouped_count(year: int, name: str, pair: list[str], ordinary: np.ndarray, specials: dict[str, np.ndarray],
                   source: str) -> ElectorateCount:
    """An official result held as [pair[0], pair[1], everyone else] instead of by candidate."""
    total = ordinary + sum(specials.values())
    zero = np.zeros(3)
    return ElectorateCount(year=year, electorate=name, number=None, candidates=[*pair, "Other"],
                           parties=[*pair, "Other"], advance=zero, election_day=zero, ordinary=ordinary,
                           specials=specials, total=total, informal=None, official=True, source=source)


def moved_places(c: ElectorateCount, patterns: list[str]) -> np.ndarray:
    """Which of an official result's voting places match any of ``patterns`` (in the place's name and address
    or the suburb it is listed under, ignoring case)."""
    b = c.booths
    text = [f"{sub} | {place}".lower() for sub, place in zip(b.suburbs or [""] * len(b.places), b.places)]
    return np.array([any(p.lower() in t for p in patterns) for t in text])


def notional_history(counts: list[ElectorateCount], name: str, parts: list[dict], pair: list[str]) -> list[ElectorateCount]:
    """Past results redrawn on a new electorate's boundaries.

    ``parts`` lists the old electorates it was made from: all of one, or the voting places of one that match
    ``places``. For each past election the new electorate, ``name``, gets those ordinary votes. Special votes
    are not counted by voting place, so a part-electorate's are shared out: the moved places take the same
    special votes per ordinary vote as the whole electorate, and they differ from the places' own ordinary
    votes as the whole electorate's special votes differed from its ordinary votes. The old electorates are
    replaced: one used whole disappears, one used in part keeps what is left as "<name> (rest)". Every result
    is held as [pair[0], pair[1], everyone else], so the votes are counted once and nothing is lost.
    """
    by = {electorate_key(p["electorate"]): p for p in parts}
    out = [c for c in counts if c.key not in by]
    for year in sorted({c.year for c in counts}):
        found = {c.key: c for c in counts if c.year == year and c.key in by}
        usable = {k: c for k, c in found.items() if c.grouped(c.ordinary, pair) is not None}
        if len(usable) < len(by):                 # without every part there is no notional result that year
            out += found.values()
            continue
        ordinary = np.zeros(3)
        specials = {k: np.zeros(3) for k in CATEGORIES}
        for key, c in usable.items():
            whole_o = c.grouped(c.ordinary, pair)
            whole_s = {k: c.grouped(c.specials[k], pair) for k in CATEGORIES}
            patterns = by[key].get("places")
            if not patterns:
                ordinary += whole_o
                for k in CATEGORIES:
                    specials[k] += whole_s[k]
                continue
            mask = moved_places(c, patterns)
            if not mask.any():
                raise ValueError(f"{c.electorate} {year}: no voting place matches {patterns}")
            moved_o = c.grouped(c.booths.votes[mask].sum(0), pair)
            rest_s = {}
            for k in CATEGORIES:
                shares = softmax0(log_ratio(moved_o) + log_ratio(whole_s[k]) - log_ratio(whole_o))
                moved_s = np.minimum(np.rint(whole_s[k].sum() * moved_o.sum() / whole_o.sum() * shares), whole_s[k])
                specials[k] += moved_s
                rest_s[k] = whole_s[k] - moved_s
            ordinary += moved_o
            out.append(_grouped_count(year, f"{c.electorate} (rest)", pair, whole_o - moved_o, rest_s, c.source))
        out.append(_grouped_count(year, name, pair, ordinary, specials, "notional"))
    return out


def read_seat(e: dict, data_dir: Path, year: int, pair: list[str]) -> ElectorateCount:
    """One seat of the forecast election: the Commission's workbook (``file``) or votes given in the config."""
    c = read_preliminary_xlsx(data_dir / e["file"], year, e["candidates"]) if e.get("file") else count_from_config(e, year)
    if c.key != electorate_key(e["name"]):
        raise ValueError(f"config/specials.yml: {e.get('file')} is {c.electorate}, not {e['name']}")
    missing = [p for p in pair if c.party_index(p) is None]
    if missing:
        raise ValueError(f"config/specials.yml: {e['name']} has no candidate for {missing}")
    c.electorate = e["name"]
    return c


def load_inputs(cfg: Config) -> Inputs:
    scfg = load_config(cfg)
    data_dir = cfg.paths.root / scfg["data_dir"]
    year = int(scfg["election"])
    history = load_history(data_dir, year).counts
    seats = [read_seat(e, data_dir, year, scfg["pair"]) for e in scfg["electorates"]]
    contests = []
    for c in scfg.get("other_contests") or []:
        cdir = cfg.paths.root / c["data_dir"]
        past = load_history(cdir, year).counts
        keys = []
        for e in c["electorates"]:
            hist = e.get("history", e["name"])
            if isinstance(hist, dict):            # past results redrawn on this seat's boundaries
                hist = hist.get("name", f"{e['name']} (notional)")
                past = notional_history(past, hist, e["history"]["parts"], c["pair"])
            keys.append(electorate_key(hist))
        have = {h.key for h in past if h.grouped(h.ordinary, c["pair"]) is not None}
        for e, k in zip(c["electorates"], keys):
            if "history" in e and k not in have:
                raise ValueError(f"config/specials.yml: {e['name']}'s history seat has no past results with "
                                 f"candidates from both {c['pair'][0]} and {c['pair'][1]} in {cdir}")
        contests.append(Contest(list(c["pair"]), c.get("title", " against ".join(c["pair"])), past,
                                [read_seat(e, cdir, year, c["pair"]) for e in c["electorates"]], keys, cdir, c))
    return Inputs(scfg, history, seats, data_dir, contests)


# --------------------------------------------------------------------------------------- national inputs
def national_draws(scfg: dict, n: int, rng: np.random.Generator, groups: list[str] | None = None) -> dict[str, np.ndarray]:
    """The forecast election's national special-vote volume, overseas share and enrolment-change shift.

    With ``groups`` (the party-vote model's parties, the reference first) it also draws the shift of each party's
    domestic special party votes, ``kappa_party`` (n, K-1). Labour's candidates and Labour's party vote share
    one draw of how far the removed voters leaned.
    """
    v = scfg["volume"]
    ref, fc = v["reference"], v["forecast"]
    s_ref, t_ref, ov_ref = float(ref["special_votes"]), float(ref["total_votes"]), float(ref["overseas_votes"])
    o_ref = t_ref - s_ref
    err = fc["estimate_error"]
    specials = fc["special_votes_estimate"] * (1 + rng.normal(err["mean"], err["sd"], n))
    ordinary = float(fc["total_votes_estimate"] - fc["special_votes_estimate"])
    ov = np.clip(rng.normal(fc["overseas_votes"]["mean"], fc["overseas_votes"]["sd"], n), 1.0, 0.9 * specials)
    removed = np.clip(1 - ((specials - ov) / ordinary) / ((s_ref - ov_ref) / o_ref), 0.0, 1.0)
    lean = scfg["rule_change"]["lean"]
    z = {"labour": rng.standard_normal(n), "other": rng.standard_normal(n)}
    lean_draws = np.stack([lean[k]["mean"] + lean[k]["sd"] * z[k] for k in ("labour", "other")], axis=1)
    out = {"log_change": np.log((specials / ordinary) / (s_ref / o_ref)),
           "overseas_logit_change": np.log(ov / (specials - ov)) - np.log(ov_ref / (s_ref - ov_ref)),
           "specials": specials, "overseas": ov, "removed_share": removed,
           "kappa": -removed[:, None] * lean_draws}
    if groups is not None:
        party_lean = scfg["rule_change"].get("party_lean", {})
        second = scfg["pair"][1]
        cols = []
        for g in groups[1:]:
            m = party_lean.get(g, {"mean": 0.0, "sd": 0.0})
            cols.append(m["mean"] + m["sd"] * (z["labour"] if g == second else rng.standard_normal(n)))
        out["kappa_party"] = -removed[:, None] * np.stack(cols, axis=1)
    return out


# -------------------------------------------------------------------------------------------- summaries
def _q(x: np.ndarray) -> dict:
    qs = np.quantile(x, QUANTILES)
    return {"mean": float(np.mean(x)), **{f"q{int(q * 100):02d}": float(v) for q, v in zip(QUANTILES, qs)}}


def _share(x: np.ndarray) -> np.ndarray:
    """Labour's share of the two contenders' votes, (..., 3) -> (...)."""
    return x[..., 1] / np.maximum(x[..., 0] + x[..., 1], 1)


def _even_seat_pp(eta: np.ndarray) -> np.ndarray:
    """A log-ratio shift as percentage points of the two-candidate share in an evenly split seat."""
    return 100 * (1 / (1 + np.exp(-eta)) - 0.5)


def history_rows(data: SpecialsData, counts: list[ElectorateCount], pair: list[str]) -> list[dict]:
    rows = []
    for c in sorted(counts, key=lambda c: (c.key, c.year)):
        o = c.grouped(c.ordinary, pair)
        if o is None:
            continue
        s = c.grouped(c.special_total, pair)
        row = {"electorate": c.electorate, "year": c.year, "number": c.number,
               "ordinary_votes": int(c.ordinary.sum()), "special_votes": int(c.special_total.sum()),
               "specials_per_ordinary": float(c.special_total.sum() / c.ordinary.sum()),
               "overseas_share": float(c.specials["overseas"].sum() / c.special_total.sum()),
               "ordinary_share": float(_share(o)), "special_share": float(_share(s)),
               "special_shift_pp": float(100 * (_share(s) - _share(o))),
               "contenders_share_of_specials": float(s[:2].sum() / s.sum()),
               "contenders_share_of_ordinary": float(o[:2].sum() / o.sum())}
        for cat in CATEGORIES:
            row[f"special_share_{cat}"] = float(_share(c.grouped(c.specials[cat], pair)))
        t = c.grouped(c.total, pair)
        row["final_margin"] = int(t[1] - t[0])
        if c.preliminary is not None:
            p = c.grouped(c.preliminary, pair)
            row |= {"preliminary_margin": int(p[1] - p[0]), "ordinary_margin": int(o[1] - o[0]),
                    "official_ordinary_change": int(c.ordinary.sum() - c.preliminary.sum()),
                    "official_margin_change": int((o[1] - o[0]) - (p[1] - p[0]))}
        rows.append(row)
    return rows


def seat_summary(c: ElectorateCount, j: int, sim: Simulation, pair: list[str], close: int) -> dict:
    start = c.grouped(c.preliminary, pair)
    margin = sim.final[:, j, 1] - sim.final[:, j, 0]
    spec = sim.specials[:, j].sum(1)                               # (n, 3)
    n_spec = spec.sum(1)
    idx = {p: c.party_index(p) for p in pair}
    contenders = {p: c.candidates[idx[p]] for p in pair}
    lead = pair[1] if start[1] > start[0] else pair[0]
    trail = pair[0] if lead == pair[1] else pair[1]
    # what the trailing candidate needs from the contenders' special votes to overturn the election-night margin,
    # at the expected number of them (ignoring the smaller, two-way change to the ordinary votes)
    gap = abs(float(start[1] - start[0]))
    contender_specials = float(np.median(spec[:, 0] + spec[:, 1]))
    trail_k = pair.index(trail)
    need = 0.5 + gap / (2 * contender_specials)
    trail_share = spec[:, trail_k] / np.maximum(spec[:, 0] + spec[:, 1], 1)
    ordinary_trail_share = float(start[trail_k] / start[:2].sum())

    # every candidate: the "everyone else" group's specials split in proportion to their election-night votes
    others = [k for k in range(len(c.candidates)) if k not in idx.values()]
    other_votes = c.preliminary[others]
    cand = []
    for k, name in enumerate(c.candidates):
        if k == idx[pair[0]]:
            final, change, sp = sim.final[:, j, 0], sim.change[:, j, 0], spec[:, 0]
        elif k == idx[pair[1]]:
            final, change, sp = sim.final[:, j, 1], sim.change[:, j, 1], spec[:, 1]
        else:
            w = c.preliminary[k] / max(other_votes.sum(), 1)
            change, sp = sim.change[:, j, 2] * w, spec[:, 2] * w
            final = c.preliminary[k] + change + sp
        cand.append({"candidate": name, "party": c.parties[k], "preliminary": int(c.preliminary[k]),
                     "specials_mean": float(sp.mean()), "final": _q(final)})
    return {
        "name": c.electorate, "number": c.number, "contenders": contenders, "leader_election_night": lead,
        "candidates": cand,
        "preliminary": {"margin": int(start[1] - start[0]), "votes": int(c.preliminary.sum()),
                        "informal": c.informal,       # by voting place only with the Commission's workbook
                        "advance": int(c.advance.sum()) if c.booths is not None else None,
                        "election_day": int(c.election_day.sum()) if c.booths is not None else None,
                        "contender_share": float(_share(start))},
        "specials": {**_q(n_spec), "overseas_mean": float(sim.specials[:, j, 0].sum(1).mean()),
                     "per_ordinary": float((n_spec / sim.final[:, j].sum(1).clip(1)).mean())},
        "official_change": {"votes": _q(sim.change[:, j].sum(1)),
                            "margin": _q(sim.change[:, j, 1] - sim.change[:, j, 0])},
        "special_share": _q(_share(spec)),
        "margin": {**_q(margin), "p_close": float((np.abs(margin) <= close).mean())},
        "p_win": {pair[0]: float((margin < 0).mean()), pair[1]: float((margin > 0).mean()),
                  "tie": float((margin == 0).mean())},
        "trailing": {"party": trail, "candidate": contenders[trail], "gap": gap,
                     "needs_share": need, "ordinary_share": ordinary_trail_share,
                     "forecast_share": _q(trail_share), "contender_specials": contender_specials},
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _posterior_summary(post: Posterior, data: SpecialsData, official: Posterior) -> dict:
    dom = CATS.index("domestic")
    out = {"years": {}, "electorates": {}}
    for y in data.years:
        t = data.year_index(y)
        eta = post["mu"][:, dom] + post["alpha"][:, t] + post["gamma"][:, t, dom]
        ov = post["mu"][:, 0] + post["alpha"][:, t] + post["gamma"][:, t, 0]
        out["years"][str(y)] = {"domestic_pp": _q(_even_seat_pp(eta[:, 0])), "overseas_pp": _q(_even_seat_pp(ov[:, 0]))}
    for k, label in zip(data.electorates, data.labels):
        e = data.electorate_index(k)
        out["electorates"][label] = {"persistent_pp": _q(_even_seat_pp(post["beta"][:, e, 0]))}
    out["sd_election_pp"] = _q(_even_seat_pp(post["sigma_year"][:, 0]))
    out["official_count"] = {k: _q(np.asarray(official[k]).reshape(len(official[k]), -1)[:, 0])
                             for k in ("jump_prob", "small_sd", "jump_log_median", "recheck_sd")}
    out["official_count"]["jump_median_share"] = _q(np.exp(official["jump_log_median"]))
    return out


# ------------------------------------------------------------------------------------------- the House
@dataclass
class PartyVote:
    """The party-vote side of the forecast: config/specials.yml's ``parliament`` section, loaded and fitted."""
    cfg: dict
    names: natdata.PartyNames
    history: natdata.NationalHistory
    data: pv.PartyVoteData
    post: Posterior
    start: np.ndarray               # election-night party votes (K,)
    change: dict                    # the official count's change to ordinary party votes, from past counts
    electorates: list[dict]         # every electorate's election-night leader
    valid: tuple[float, float]      # valid special party votes per special vote reported: (all, overseas)

    @property
    def groups(self) -> list[str]:
        return self.names.groups


def load_party_vote(cfg: Config, scfg: dict, mcmc: dict, progress: bool = False) -> PartyVote | None:
    pcfg = scfg.get("parliament")
    if not pcfg:
        return None
    pair, year = scfg["pair"], int(scfg["election"])
    names = natdata.PartyNames(pcfg["parties"])
    if names.groups[0] != pair[0] or pair[1] not in names.groups:
        raise ValueError(f"config/specials.yml: parliament.parties must start with {pair[0]} and include {pair[1]}")
    data_dir = cfg.paths.root / pcfg["data_dir"]
    history = natdata.load_history(data_dir, names, before=year)
    ref = scfg["volume"]["reference"]
    if int(ref["year"]) not in history.years:
        raise ValueError(f"{data_dir}/party-votes.csv has no {ref['year']} election (the volume reference)")
    t = history.year_index(int(ref["year"]))
    data = pv.build_party_vote_data(history, pair[1])
    post = pv.fit_party_vote(data, pcfg["priors"], mcmc, progress)
    return PartyVote(pcfg, names, history, data, post, natdata.load_preliminary(data_dir, year, names),
                     pv.ordinary_change(history), natdata.read_electorates(data_dir / f"electorates-{year}.csv"),
                     (float(history.specials[t].sum() / ref["special_votes"]),
                      float(history.specials[t, 0].sum() / ref["overseas_votes"])))


def extra_targets(party: PartyVote, pair: list[str], have: set[str]) -> list[tuple[dict, Target]]:
    """The other seats where the pair's candidates are first and second within ``modelled_margin`` votes."""
    out = []
    limit = int(party.cfg.get("modelled_margin", 0))
    for r in party.electorates:
        if electorate_key(r["electorate"]) in have or r["margin"] > limit:
            continue
        if {r["party"], r["runner_up_party"]} != set(pair):
            continue
        if r["national"] is None or r["labour"] is None or r["valid_votes"] is None:
            raise ValueError(f"electorates file: {r['electorate']} is within {limit} votes; add its candidate votes")
        start = np.array([r["national"], r["labour"], r["valid_votes"] - r["national"] - r["labour"]], dtype=float)
        if abs(start[1] - start[0]) != r["margin"] or (start[1] > start[0]) != (r["party"] == pair[1]):
            raise ValueError(f"electorates file: {r['electorate']}'s candidate votes do not give its margin")
        out.append((r, Target(r["electorate"], electorate_key(r["electorate"]), start, True)))
    return out


def electorate_seats(party: PartyVote, contests: list[tuple[list[str], list[Target], Simulation]],
                     rng: np.random.Generator) -> np.ndarray:
    """Electorates won by each party in every simulation, (n, K). ``contests`` are the simulated seats, each
    group with the pair of parties its margins are between."""
    groups, n = party.groups, len(contests[0][2].final)
    rows = {electorate_key(r["electorate"]): r for r in party.electorates}
    won = np.zeros((n, len(groups)), dtype=int)
    for r in party.electorates:
        won[:, party.names.index(r["party"])] += 1
    for pair, targets, sim in contests:
        k0, k1 = groups.index(pair[0]), groups.index(pair[1])
        for j, t in enumerate(targets):
            r = rows.get(electorate_key(t.name))
            if r is None:
                raise ValueError(f"{t.name} is not in the electorates file")
            if r["party"] not in pair:
                raise ValueError(f"{t.name}: the electorates file has {r['party']} leading, not one of {pair}")
            lead = k1 if r["party"] == pair[1] else k0
            other = k0 if lead == k1 else k1
            m = sim.final[:, j, 1] - sim.final[:, j, 0]
            flipped = (m < 0) if lead == k1 else (m > 0)
            won[:, lead] -= flipped
            won[:, other] += flipped
    for a in party.cfg.get("assumed_seats") or []:
        r = rows.get(electorate_key(a["electorate"]))
        if r is None:
            raise ValueError(f"config/specials.yml: assumed seat {a['electorate']} is not in the electorates file")
        flipped = rng.random(n) < float(a["p_flip"])
        won[:, party.names.index(r["party"])] -= flipped
        won[:, party.names.index(a["challenger"])] += flipped
    return won


def party_vote_history_rows(h: natdata.NationalHistory) -> list[dict]:
    rows = []
    for t, y in enumerate(h.years):
        o, s = h.ordinary[t], h.specials[t]
        for k, p in enumerate(h.parties):
            if o[k] < pv.MIN_VOTES:
                continue
            row = {"year": y, "party": p, "ordinary_votes": int(o[k]), "overseas_votes": int(s[0, k]),
                   "domestic_votes": int(s[1, k]), "ordinary_share": float(o[k] / o.sum()),
                   "overseas_share": float(s[0, k] / s[0].sum()), "domestic_share": float(s[1, k] / s[1].sum()),
                   "special_share": float(s[:, k].sum() / s.sum()), "final_share": float(h.final[t, k] / h.final[t].sum())}
            if not np.isnan(h.prelim[t]).any():
                row["election_night_share"] = float(h.prelim[t, k] / h.prelim[t].sum())
            rows.append(row)
    return rows


def house_summary(party: PartyVote, pvsim: pv.PartyVoteSim, house: parl.House, base: parl.House) -> dict:
    gov, opp = party.cfg["government"]["parties"], party.cfg["opposition"]["parties"]
    out = parl.summarise(house, base, gov, opp)
    out["routes"] = parl.paths(house, base, gov, opp)
    start, final, spec = party.start, pvsim.final, pvsim.specials
    share = final / final.sum(1, keepdims=True)
    spec_share = spec.sum(1) / spec.sum((1, 2))[:, None]
    threshold = float(party.cfg.get("threshold", 0.05))
    out["party_vote"] = [{
        "party": g, "election_night_votes": int(start[k]), "election_night_share": float(start[k] / start.sum()),
        "final_share": _q(share[:, k]), "change_pp": _q(100 * (share[:, k] - start[k] / start.sum())),
        "special_share": _q(spec_share[:, k]),
        "overseas_share": float((spec[:, 0, k] / spec[:, 0].sum(1)).mean()),
        "domestic_share": float((spec[:, 1, k] / spec[:, 1].sum(1)).mean()),
        "p_threshold": float((share[:, k] >= threshold).mean())} for k, g in enumerate(party.groups)]
    out["special_party_votes"] = {"valid": _q(spec.sum((1, 2))), "overseas": _q(spec[:, 0].sum(1)),
                                  "ordinary_added": _q(pvsim.ordinary.sum(1) - start.sum())}
    return out


# ------------------------------------------------------------------------------------------------- stage
def forecast_specials(cfg: Config, *, quick: bool = False, run_backtest: bool = True, progress: bool = False) -> dict:
    inp = load_inputs(cfg)
    scfg, pair = inp.scfg, inp.scfg["pair"]
    priors = scfg["priors"]
    mcmc = dict(scfg["mcmc"])
    n = int(scfg.get("n_sims", 20000))
    if quick:
        mcmc |= {"chains": 2, "warmup": 300, "samples": 300}
        n = min(n, 4000)
    df = priors.get("year_df")
    seed = int(mcmc.get("seed", 0))
    out = cfg.paths.output / OUT
    out.mkdir(parents=True, exist_ok=True)

    # the country's party vote first: the close seats' election-wide shift is tied to its lean
    party = load_party_vote(cfg, scfg, mcmc, progress)
    link = None
    if party is not None:
        link = {y: party.data.link[t] - party.data.link_mean for t, y in enumerate(party.data.years)}
        _log(f"specials: party vote fitted on {', '.join(map(str, party.data.years))} "
             f"(divergences {party.post.diagnostics['n_divergent']})")
    data = build_specials_data(inp.history, pair, link)
    post = fit_specials(data, priors, mcmc, progress)
    oc = build_official_count_data(data)
    official = fit_official_count(oc, priors, mcmc, progress)
    _log(f"specials: fitted on {data.n_rows} past seat-elections ({', '.join(map(str, data.years))}) in "
         f"{post.diagnostics['seconds']}s; official count on {len(oc.prelim)} archived election-night counts")

    rng = np.random.default_rng(seed)
    nat = national_draws(scfg, n, rng, party.groups if party else None)
    ref_year = int(scfg["volume"]["reference"]["year"])
    if ref_year not in data.years:
        raise ValueError(f"config/specials.yml: reference year {ref_year} has no official results in the data")
    volume = VolumeShift(ref_year, nat["log_change"], nat["overseas_logit_change"],
                         float(scfg["volume"].get("rule_change_sd", 0.0)))
    targets = [Target(c.electorate, c.key, c.grouped(c.preliminary, pair), True) for c in inp.seats]
    extra = extra_targets(party, pair, {t.key for t in targets}) if party else []
    targets += [t for _, t in extra]
    max_add = float(priors["official_count"].get("max_addition", 0.2))

    # the contests between other pairs of parties: the same model on their own past results
    fitted = []
    for i, c in enumerate(inp.contests):
        if party is None:
            raise ValueError("config/specials.yml: other_contests needs the parliament section")
        first, second = party.groups.index(c.pair[0]), party.groups.index(c.pair[1])
        lk = pv.pair_link(party.history, second, first)
        cdata = build_specials_data(c.history, c.pair,
                                    {y: lk[t] - lk.mean(0) for t, y in enumerate(party.history.years)})
        if ref_year not in cdata.years:
            raise ValueError(f"{c.data_dir}: no official results for the reference year {ref_year}")
        cpost = fit_specials(cdata, priors, {**mcmc, "seed": seed + 1 + i}, progress)
        ctargets = [Target(x.electorate, k, x.grouped(x.preliminary, c.pair), True) for x, k in zip(c.seats, c.keys)]
        fitted.append({"contest": c, "data": cdata, "post": cpost, "targets": ctargets, "first": first,
                       "second": second, "link_mean": lk.mean(0)})
        _log(f"specials: {' v '.join(c.pair)} fitted on {cdata.n_rows} past seat-elections "
             f"({', '.join(cdata.labels)}; divergences {cpost.diagnostics['n_divergent']})")

    def run(year: int | None = None, enrolment: bool = True):
        """One set of simulated counts: the seats, and with the party vote the whole House."""
        r = np.random.default_rng([seed, year or 0, int(enrolment)])
        lean = z = None
        if party is not None:
            lean = pv.draw_lean(party.post, party.data, n, r, year=year, df=df)
            z = pv.link_covariate(lean, party.start, party.data.second) - party.data.link_mean
        sim = simulate(post, data, targets, volume, official=official, df=df, n=n, max_addition=max_add, seed=seed,
                       year=year, kappa=nat["kappa"] if enrolment else None,
                       national=z if data.national is not None else None)
        if party is None:
            return sim, None, None, []
        csims = []                  # the same country-wide lean, volume and enrolment change as the seats above
        for i, ft in enumerate(fitted):
            zc = pv.link_covariate(lean, party.start, ft["second"], ft["first"]) - ft["link_mean"]
            kc = pv.against(nat["kappa_party"], party.start, ft["second"], ft["first"]) if enrolment else None
            csims.append(simulate(ft["post"], ft["data"], ft["targets"], volume, official=official, df=df, n=n,
                                  max_addition=max_add, seed=seed + 1 + i, year=year, kappa=kc,
                                  national=zc if ft["data"].national is not None else None))
        pvsim = pv.simulate_party_votes(party.start, lean, nat["specials"] * party.valid[0],
                                        nat["overseas"] * party.valid[1], r, change=party.change,
                                        kappa=nat["kappa_party"] if enrolment else None)
        simulated = [(pair, targets, sim)] + [(ft["contest"].pair, ft["targets"], cs) for ft, cs in zip(fitted, csims)]
        house = parl.elect(pvsim.final, electorate_seats(party, simulated, r), party.groups,
                           int(party.cfg.get("seats", 120)), float(party.cfg.get("threshold", 0.05)))
        return sim, pvsim, house, csims

    sim, pvsim, house, csims = run()
    close = int(scfg.get("close_margin", 100))
    seats = [seat_summary(c, j, sim, pair, close) for j, c in enumerate(inp.seats)]
    p_second = lambda s: (s.final[:, :, 1] > s.final[:, :, 0]).mean(0)  # noqa: E731
    gov = party.cfg["government"]["parties"] if party else []
    opp = party.cfg["opposition"]["parties"] if party else []
    house_odds = lambda h: {k: float((parl.outcome(h, gov, opp) == i).mean())  # noqa: E731
                            for i, k in enumerate(parl.OUTCOMES)}
    # scenarios: replay each past election's special-vote shift; and no enrolment-change adjustment
    scenarios = {"model": {s["name"]: s["p_win"][pair[1]] for s in seats}}
    house_scenarios = {"model": house_odds(house)} if party else {}
    for ft in fitted:
        missing = [y for y in data.years if y not in ft["data"].years]
        if missing:
            raise ValueError(f"{ft['contest'].data_dir}: no official results for {missing}")
    cscenarios = [{"model": {x.electorate: float(q) for x, q in zip(ft["contest"].seats, p_second(cs))}}
                  for ft, cs in zip(fitted, csims)]
    alts = {f"like_{y}": {"year": y} for y in data.years} | {"no_enrolment_adjustment": {"enrolment": False}}
    for name, kw in alts.items():
        alt, _, alt_house, alt_cs = run(**kw)
        for ft, sc, cs in zip(fitted, cscenarios, alt_cs):
            sc[name] = {x.electorate: float(q) for x, q in zip(ft["contest"].seats, p_second(cs))}
        scenarios[name] = {c.electorate: float(p) for c, p in zip(inp.seats, p_second(alt))}
        if party:
            house_scenarios[name] = house_odds(alt_house)
    wins = (sim.final[:, :len(inp.seats), 1] > sim.final[:, :len(inp.seats), 0]).sum(1)
    dom = CATS.index("domestic")

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "election": int(scfg["election"]), "official_results_due": str(scfg.get("official_results_due", "")),
        "pair": pair, "close_margin": close, "n_sims": n,
        "seats": seats,
        "joint": {"party": pair[1], "wins": {str(k): float((wins == k).mean()) for k in range(len(seats) + 1)}},
        "scenarios": scenarios,
        "national": {"special_votes": _q(nat["specials"]), "overseas": _q(nat["overseas"]),
                     "removed_share": _q(nat["removed_share"]),
                     "estimate": scfg["volume"]["forecast"]["special_votes_estimate"],
                     "overseas_estimate": scfg["volume"]["forecast"]["overseas_votes"]["mean"],
                     "total_votes_estimate": scfg["volume"]["forecast"]["total_votes_estimate"],
                     "reference": scfg["volume"]["reference"],
                     "change_per_ordinary": _q(np.exp(nat["log_change"])),
                     "enrolment_shift_pp": _q(_even_seat_pp(nat["kappa"][:, 0]))},
        "election_shift_pp": {"forecast_domestic": _q(_even_seat_pp(sim.election[:, dom, 0])),
                              "forecast_overseas": _q(_even_seat_pp(sim.election[:, 0, 0]))},
        "posterior": _posterior_summary(post, data, official),
        "diagnostics": {"specials": post.diagnostics, "official_count": official.diagnostics},
        "data": {"years": data.years, "electorates": data.labels, "n_seat_elections": data.n_rows,
                 "n_election_night_counts": len(oc.prelim), "official_count_cases": oc.labels},
    }
    if "link_slope" in post.draws:
        summary["posterior"]["link_slope"] = _q(post["link_slope"][:, 0])

    contests = []
    for ft, cs, sc in zip(fitted, csims, cscenarios):
        c, cdata, cpost = ft["contest"], ft["data"], ft["post"]
        cseats = []
        for j, (x, k, e) in enumerate(zip(c.seats, c.keys, c.cfg["electorates"])):
            row = seat_summary(x, j, cs, c.pair, close)
            if k != x.key:          # the seat whose past results stand for this one's
                row["history"] = cdata.labels[cdata.electorate_index(k)]
            cseats.append(row | {"note": e.get("note"), "source": x.source})
        cwins = (cs.final[:, :, 1] > cs.final[:, :, 0]).sum(1)
        csum = {"pair": c.pair, "slug": c.slug, "title": c.title, "seats": cseats, "scenarios": sc,
                "joint": {"party": c.pair[1], "wins": {str(k): float((cwins == k).mean()) for k in range(len(cseats) + 1)}},
                "election_shift_pp": {"forecast_domestic": _q(_even_seat_pp(cs.election[:, dom, 0])),
                                      "forecast_overseas": _q(_even_seat_pp(cs.election[:, 0, 0]))},
                "enrolment_shift_pp": _q(_even_seat_pp(pv.against(nat["kappa_party"], party.start, ft["second"],
                                                                 ft["first"])[:, 0])),
                "posterior": _posterior_summary(cpost, cdata, official),
                "diagnostics": {"specials": cpost.diagnostics},
                "data": {"years": cdata.years, "electorates": cdata.labels, "n_seat_elections": cdata.n_rows}}
        if "link_slope" in cpost.draws:
            csum["posterior"]["link_slope"] = _q(cpost["link_slope"][:, 0])
        contests.append(csum)
    if contests:
        summary["contests"] = contests

    if party is not None:
        leaders = np.bincount([party.names.index(r["party"]) for r in party.electorates],
                              minlength=len(party.groups))
        base = parl.elect(party.start, leaders, party.groups, int(party.cfg.get("seats", 120)),
                          float(party.cfg.get("threshold", 0.05)))
        hs = house_summary(party, pvsim, house, base)
        first = len(inp.seats)
        margins = sim.final[:, first:, 1] - sim.final[:, first:, 0]
        hs["other_seats"] = [{
            "name": r["electorate"], "leader": r["leader"], "leader_party": r["party"],
            "election_night_margin": int(t.start[1] - t.start[0]), "margin": _q(margins[:, j]),
            "p_flip": float(((margins[:, j] > 0) != (t.start[1] > t.start[0])).mean())}
            for j, (r, t) in enumerate(extra)]
        hs["assumed_seats"] = [{**a, "leader_party": next(r["party"] for r in party.electorates if electorate_key(
            r["electorate"]) == electorate_key(a["electorate"]))} for a in party.cfg.get("assumed_seats") or []]
        hs |= {"parties": party.groups, "government": party.cfg["government"], "opposition": party.cfg["opposition"],
               "colours": party.cfg.get("colours", {}), "modelled_margin": int(party.cfg.get("modelled_margin", 0)),
               "n_electorates": len(party.electorates), "scenarios": house_scenarios,
               "years": party.data.years, "ordinary_change": party.change,
               "model": {"common_sd": _q(party.post["sigma_f"]),
                         "party_sd": {g: _q(party.post["sigma_u"][:, k]) for k, g in enumerate(party.groups[1:])},
                         "category_sd": {c: _q(party.post["sigma_g"][:, k]) for k, c in enumerate(CATS)},
                         "usual_lean": {c: {g: float(party.post["mu"][:, i, k].mean())
                                            for k, g in enumerate(party.groups[1:])} for i, c in enumerate(CATS)}},
               "diagnostics": party.post.diagnostics}
        summary["house"] = hs
        _write_csv(out / "party_votes.csv", [{
            "party": r["party"], "election_night_votes": r["election_night_votes"],
            "election_night_share": r["election_night_share"],
            **{f"final_share_{k}": v for k, v in r["final_share"].items()},
            "special_share_mean": r["special_share"]["mean"], "p_threshold": r["p_threshold"],
            **next(({"seats_election_night": s["election_night"], "seats_mean": s["mean"],
                     "p_fewer_seats": s["p_fewer"], "p_more_seats": s["p_more"]}
                    for s in hs["seats"] if s["party"] == r["party"]), {})} for r in hs["party_vote"]])
        _write_csv(out / "routes.csv", [{"kind": kind, "route": "; ".join(r["steps"]), "probability": r["p"],
                                         "outcome": r.get("outcome", "opposition")}
                                        for kind in ("top", "minimal") for r in hs["routes"][kind]])
        _write_csv(out / "party_vote_history.csv", party_vote_history_rows(party.history))
        o = hs["outcome"]
        _log(f"specials: House: {party.cfg['government']['label']} keep a majority {o['government']:.2f}, "
             f"neither side {o['neither']:.2f}, {party.cfg['opposition']['label']} {o['opposition']:.2f}")

    hist = history_rows(data, inp.history, pair)
    if run_backtest:
        bmcmc = {**mcmc, **scfg.get("backtest", {}).get("mcmc", {})}
        if quick:
            bmcmc |= {"chains": 2, "warmup": 300, "samples": 300}
        b = scfg.get("backtest", {})
        nb = min(n, 8000)

        refits: dict[int, tuple] = {}            # the party-vote model without each election, fitted once

        def linked_for(first: int, second: int):
            """The national lean of ``second`` against ``first`` without the target election, for `backtest`."""
            def linked(target_year: int, size: int, r: np.random.Generator):
                h = party.history
                if target_year not in h.years or len(h.years) < 3:
                    return None
                if target_year not in refits:
                    d = pv.build_party_vote_data(h, pair[1], [y for y in h.years if y != target_year])
                    refits[target_year] = (d, pv.fit_party_vote(d, party.cfg["priors"], bmcmc))
                d, fit = refits[target_year]
                draws = pv.draw_lean(fit, d, size, r, df=df)
                lk = pv.pair_link(h.subset(d.years), second, first)
                z = pv.link_covariate(draws, h.ordinary[h.year_index(target_year)], second, first) - lk.mean(0)
                return {y: lk[t] - lk.mean(0) for t, y in enumerate(d.years)}, z
            return linked

        bkw = {"n": nb, "df": df, "volume_error_sd": float(b.get("volume_error_sd", 0.04)),
               "overseas_error_sd": float(b.get("overseas_error_sd", 0.10)), "seed": 7, "log": _log}
        rows = backtest(inp.history, pair, priors, bmcmc, **bkw,
                        link=linked_for(0, party.data.second) if party is not None else None)
        _write_csv(out / "backtest.csv", rows)
        summary["backtest"] = summarise(rows)
        _write_csv(out / "backtest_summary.csv", summary["backtest"])
        for ft, csum in zip(fitted, contests):
            crows = backtest(ft["contest"].history, csum["pair"], priors, bmcmc, **bkw,
                             link=linked_for(ft["first"], ft["second"]))
            _write_csv(out / f"backtest_{csum['slug']}.csv", crows)
            csum["backtest"] = summarise(crows)
            _write_csv(out / f"backtest_summary_{csum['slug']}.csv", csum["backtest"])
        if party is not None:
            won = natdata.read_electorate_seats(cfg.paths.root / ELECTORATE_SEATS, party.names)
            small = natdata.small_winners(cfg.paths.root / party.cfg["data_dir"], cfg.paths.root / ELECTORATE_SEATS,
                                          party.names)
            prow = backtest_party_vote(party.history, pair[1], party.cfg["priors"], bmcmc, electorates=won,
                                       small=small, n=nb,
                                       df=df, seed=11, log=_log, nseats=int(party.cfg.get("seats", 120)),
                                       threshold=float(party.cfg.get("threshold", 0.05)),
                                       volume_error_sd=float(party.cfg.get("backtest", {}).get("volume_error_sd", 0.04)))
            _write_csv(out / "party_vote_backtest.csv", prow)
            summary["house"]["backtest"] = summarise_party_vote(prow)
            (out / "party_vote_backtest.json").write_text(json.dumps(summary["house"]["backtest"]), encoding="utf-8")
    else:
        if (out / "backtest_summary.csv").exists():
            with open(out / "backtest_summary.csv", encoding="utf-8") as f:
                summary["backtest"] = [{k: _parse(v) for k, v in r.items()} for r in csv.DictReader(f)]
        for csum in contests:
            path = out / f"backtest_summary_{csum['slug']}.csv"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    csum["backtest"] = [{k: _parse(v) for k, v in r.items()} for r in csv.DictReader(f)]
        if party is not None and (out / "party_vote_backtest.json").exists():
            summary["house"]["backtest"] = json.loads((out / "party_vote_backtest.json").read_text(encoding="utf-8"))

    (out / "forecast.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    _write_csv(out / "history.csv", hist)
    every = [(pair, s) for s in seats] + [(c["pair"], s) for c in contests for s in c["seats"]]
    _write_csv(out / "seats.csv", [{
        "electorate": s["name"], "first": pr[0], "second": pr[1], "leader_election_night": s["leader_election_night"],
        "preliminary_margin": s["preliminary"]["margin"], "specials_mean": s["specials"]["mean"],
        "specials_q05": s["specials"]["q05"], "specials_q95": s["specials"]["q95"],
        **{f"margin_{k}": v for k, v in s["margin"].items()},
        **{f"p_{p}": s["p_win"][p] for p in pr}, "trailing_needs_share": s["trailing"]["needs_share"]}
        for pr, s in every])
    _write_csv(out / "candidates.csv", [{"electorate": s["name"], **{k: v for k, v in c.items() if k != "final"},
                                         **{f"final_{k}": v for k, v in c["final"].items()}}
                                        for _, s in every for c in s["candidates"]])
    first = len(seats)
    np.savez_compressed(out / "margins.npz", names=np.array([s["name"] for s in seats]),
                        margin=(sim.final[:, :first, 1] - sim.final[:, :first, 0]).astype(np.int32),
                        special_share=_share(sim.specials[:, :first].sum(2)).astype(np.float32))
    # the same margins as a table, thinned, for charts drawn outside Python (r/specials_charts.R)
    thin = np.linspace(0, n - 1, min(n, 5000)).astype(int)
    _write_csv(out / "margin_draws.csv", [{"electorate": s["name"], "margin": int(sim.final[i, j, 1] - sim.final[i, j, 0])}
                                          for j, s in enumerate(seats) for i in thin])
    for ft, cs, csum in zip(fitted, csims, contests):
        cm = (cs.final[:, :, 1] - cs.final[:, :, 0]).astype(np.int32)
        _write_csv(out / f"history_{csum['slug']}.csv", history_rows(ft["data"], ft["contest"].history, csum["pair"]))
        np.savez_compressed(out / f"margins_{csum['slug']}.npz", names=np.array([x["name"] for x in csum["seats"]]),
                            margin=cm, special_share=_share(cs.specials.sum(2)).astype(np.float32))
        _write_csv(out / f"margin_draws_{csum['slug']}.csv",
                   [{"electorate": x["name"], "margin": int(cm[i, j])} for j, x in enumerate(csum["seats"]) for i in thin])
    for pr, s in every:
        _log(f"specials: {s['name']}: election night {s['preliminary']['margin']:+d} ({pr[1]} - {pr[0]}), "
             f"P({pr[1]}) {s['p_win'][pr[1]]:.2f}, final margin {s['margin']['q50']:+.0f} "
             f"({s['margin']['q05']:+.0f} to {s['margin']['q95']:+.0f})")
    return summary


def _parse(v: str):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def run_specials(cfg: Config, *, quick: bool = False, run_backtest: bool = True, force: bool = False) -> bool:
    """The stage, skipped when its inputs (data, config/specials.yml and this package) are unchanged."""
    scfg = load_config(cfg)
    out = cfg.paths.output / OUT
    extra = []
    if scfg.get("parliament"):                # the House also reads past electorates won and the seat allocator
        extra = [cfg.paths.root / scfg["parliament"]["data_dir"], cfg.paths.root / ELECTORATE_SEATS,
                 Path(__file__).parents[1] / "forecast" / "seats.py"]
    extra += [cfg.paths.root / c["data_dir"] for c in scfg.get("other_contests") or []]
    inputs = [cfg.paths.root / scfg["data_dir"], *extra, cfg.paths.config / "specials.yml", Path(__file__).parent,
              f"quick={quick}", f"backtest={run_backtest}"]
    return StageCache(cfg.paths.processed / "stamps").run(
        "specials", inputs, [out / "forecast.json"],
        lambda: forecast_specials(cfg, quick=quick, run_backtest=run_backtest), force=force)
