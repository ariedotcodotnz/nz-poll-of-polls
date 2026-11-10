# Outputs

Vote shares are proportions between 0 and 1 unless a column name ends in `_pp`, which means percentage points.
Quantile columns `q05` to `q95` are the 5th, 25th, 50th, 75th and 95th percentiles. `when` is `election_day` for
the forecast of 7 November or `now` for "if the election were held now".

## output/ (forecast outputs)

Each forecast run generates these files. CI publishes fresh CSV and JSON downloads with the website; the copies
checked into the repository are snapshots from earlier runs.

### summary.json

Everything the website shows, in one file.

| Key | Meaning |
|---|---|
| `generated_at`, `election_date`, `days_to_election`, `cutoff` | timing |
| `variants`, `weights` | ensemble members and their weights |
| `spread_calibration` | calibration parameters, or `null` when off |
| `fundamentals_prior`, `fundamentals_ess` | the estimated prior on the PM party's swing (reported, not used) |
| `n_polls_cycle`, `n_polls_total`, `latest_poll`, `pollsters_cycle` | the data behind the forecast; `latest_poll` is null until the term's first poll |
| `parties` | tracked parties in order; National first, Other last |
| `election_day`, `now` | per party: `mean`, quantiles and `p_over_5pct` |
| `seats_election_day` | per party: seat mean and quantiles, and `p_any_seats` |
| `coalitions_election_day`, `coalitions_now` | per coalition: seats and `p_majority` |
| `balance_of_power` | for `election_day` and `now`, per bloc: `p_alone`, `p_with` (each pivot party alone enough), `p_any_one`, `p_needs_several` (a majority only with two or more pivots together), `p_needs_all` (every pivot needed) and `p_short`. `p_alone`, `p_any_one`, `p_needs_several` and `p_short` are exclusive and sum to one; with two pivots `p_needs_all` equals `p_needs_several` |
| `expected_house_size`, `p_overhang` | size of the House, including overhang |
| `diagnostics` | per variant: divergences, largest R-hat, smallest effective sample size, run time |

### forecast.csv

`when, party, mean, q05, q25, q50, q75, q95, p_over_5pct`: vote-share forecast per party.

### seats.csv

`when, party, seats_mean, seats_q05, seats_q50, seats_q95, p_any_seats`: seats per party, including electorates.

### coalitions.csv

`when, name, parties, seats_mean, seats_q05, seats_q95, p_majority`: one row per coalition in
`config/electorates.yml`. A majority is half the simulated House, rounded down, plus one.

### trend.csv

`party, week, mean, q05, q25, q50, q75, q95, has_data`: weekly support since the anchor election, from the same
weighted mixture as the forecast. `week` is the Sunday the week starts on. `has_data` is false for weeks after
the last poll, which are projections.

### house_effects.csv

`house, party, effect_pp`: each pollster's persistent house effect, converted to percentage points at current
support levels. Positive means the pollster tends to report a higher number for that party. Pollsters with a
method change appear once per segment, for example "Reid Research (segment 1)".

### Charts

The SVG file names are the ones the 2023 pipeline produced, so existing embeds keep working.

| File | Chart |
|---|---|
| `voting_intention620.svg`, `voting_intention375.svg` | support over time by party, in two columns (620 px) or one (375 px) |
| `voting_intention_all620.svg`, `voting_intention_all375.svg` | every party on one chart this term; end labels at 620 px, a legend at 375 px |
| `election_night620.svg`, `election_night375.svg` | seat distributions of each coalition on election day |
| `saturday620.svg`, `saturday375.svg` | the same if the election were held now |

`seat_sims.npz`, not committed, holds the raw simulations: `seats_election`, `total_election`, `seats_now`,
`total_now`, `pi_election`, `pi_now` and `parties`.

## output/specials/ (special votes)

Written by `pollofpolls specials`; see [Special votes](specials.md). Margins are `pair[1]` minus `pair[0]` in votes
(Labour minus National), and shares of "the two leaders" are of those two candidates' votes only.

| File | Contents |
|---|---|
| `forecast.json` | everything the Special votes page shows: per seat (`seats`) the election-night count, expected special votes, the official count's change, the final `margin` (mean, quantiles, `p_close`), `p_win`, what the trailing candidate needs (`trailing`) and every candidate's forecast; the joint distribution of seats won (`joint`); `scenarios` (each past election's shift replayed, and no enrolment adjustment); the `national` volume and enrolment-change draws; posterior summaries; sampler `diagnostics`; the `backtest` summary; and `house`: the chance each side has a majority (`outcome`), the distribution of the Government's `margin` over everyone else, every party's `party_vote` and `seats`, the `routes` from the election-night House (most simulated, and the minimal routes to a one-seat majority for the other side), the other simulated and assumed seats, `scenarios` and the party-vote backtest |
| `party_votes.csv` | one row per party: election-night votes and share, quantiles of the final share, its share of special votes, election-night and mean forecast seats, and the chance of fewer or more seats |
| `routes.csv` | the routes in `house.routes`: what changes from election night, its probability and result |
| `party_vote_history.csv` | one row per party and past election: ordinary, overseas and domestic special votes and shares, the final share and, where archived, the election-night share |
| `party_vote_backtest.csv` | one row per party and past election: start, actual and forecast final share, interval coverage, the errors of the model and two rules of thumb, and seats at the start, in the result and in the model |
| `seats.csv` | one row per seat: election-night leader and margin, special votes, margin quantiles, win probabilities and the share the trailing candidate needs |
| `candidates.csv` | one row per candidate: party, election-night votes, mean special votes, final-vote quantiles |
| `history.csv` | one row per past seat and election: special votes per ordinary vote, overseas share, Labour's share of the two leaders in ordinary and special votes (overall and per category), the shift, and, where the election-night count is archived, how the official count changed the ordinary votes and the margin |
| `backtest.csv` | one row per past seat, election and method: start and actual margins, CRPS, absolute error, interval coverage, Brier score and log loss |
| `backtest_summary.csv` | those scores averaged per method |
| `margins.npz` | simulated final margins (`margin`, simulations × seats) and Labour's share of the two leaders' special votes (`special_share`), for the charts |

The site copies the CSV and JSON files into `specials/`.

## output/backtest/ (committed)

| File | Contents |
|---|---|
| `cases/<variant>_<year>_h<weeks>.json` | one backtest case: cutoff, number of polls, forecast mean, outcome, scores, diagnostics, and the `blocs` its bloc-lead score used. Aggregation rescores a case from its cached fit when `backtest.blocs` has changed. |
| `scores.csv` | one row per case, including the `ensemble`, `calibrated` and `equal` mixtures; per-party columns such as `crps_pp[National]`, `error_pp[National]`, `sd_pp[National]` and `pit[National]` |
| `summary.csv`, `summary_by_horizon.csv` | mean scores per variant, overall and by horizon |
| `head_to_head.csv` | the ensemble and the calibrated ensemble against the 2023 model replica, case by case |
| `ensemble.csv` | the mixture rows of `scores.csv` only |
| `stacking.json` | stacking weights for 2026 (`weights`), per held-out election (`loeo_weights`), and spread-calibration parameters (`spread`, `loeo_spread`) |
| `probe_fundamentals.json` | the test of the fundamentals prior on 2023 at eight weeks |
| `fits/` | cached backtest fits and their fingerprints; not committed |

Score definitions are in [How it is tested](https://ariedotcodotnz.github.io/nz-poll-of-polls/evaluation.html).

## site/ (published to GitHub Pages)

Built by Quarto from `website/`; see [Development](development.md#the-website).

| File | Contents |
|---|---|
| `index.html` | the forecast, from `website/index.qmd`; chart data is embedded |
| `specials.html` | "Special votes", from `website/specials.qmd`: the official-count forecast: who has a majority, the party vote and seats, and the close seats |
| `model.html` | "How the model works", from `website/model.qmd` |
| `evaluation.html` | "How it is tested", from `website/evaluation.qmd`, with the latest backtest results |
| `site_libs/` | Quarto's scripts and styles |
| `sitemap.xml`, `robots.txt`, `search.json` | written by Quarto |

Plotly, KaTeX and the web fonts load from CDNs. The CSV, JSON and SVG outputs are copied next to the pages, and the backtest tables (`summary.csv`,
`summary_by_horizon.csv`, `head_to_head.csv`, `scores.csv`, `stacking.json`, `probe_fundamentals.json`) into
`backtest/`. The page's Data section links to all of them, so they can be downloaded from the published site.

## data/processed/ (regenerated, not committed)

| File | Contents |
|---|---|
| `polls.parquet` | one row per poll and party: `pollster`, `pollster_raw`, `segment`, `date_text`, `date_from`, `date_to`, `mid_date`, `available`, `week`, `sample_size`, `sample_reported`, `cycle`, `party`, `share`, `poll_id`, `page_year` |
| `results.json` | election results by year and party |
| `dataset_<year>.npz`, `.json` | the arrays the models read for the forecast election |
| `fit_<variant>_<year>.npz`, `.json` | fitted posterior summaries, thinned draws and hyperparameter draws |
| `.stamps/` | fingerprints of the cached fits |
