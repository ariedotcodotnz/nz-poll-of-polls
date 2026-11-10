# Special votes in close electorates

`pollofpolls specials` forecasts the official result from the election-night count: who wins the electorates still
in doubt once the special votes are counted and the ordinary votes rechecked, each party's final party vote and
seats, and how likely the Government is to keep its majority, tie or lose it. The website's
[Special votes](https://ariedotcodotnz.github.io/nz-poll-of-polls/specials.html) page shows it. Everything is set in
[`config/specials.yml`](../config/specials.yml); the code is in [`src/pollofpolls/specials/`](../src/pollofpolls/specials/).

```bash
pollofpolls specials             # fit, backtest, simulate; writes output/specials/ (several minutes)
pollofpolls specials --no-backtest --quick    # a fast check while editing
pollofpolls report               # rebuild the website, including the Special votes page
```

The stage reruns only when its inputs change (the data folders, `config/specials.yml` or the package); `--force`
reruns it anyway.

## Data

All in `data/reference/electorate-seats-small-margin/` (`data_dir` in the config):

| File | What it is |
|---|---|
| `<year>/candidate-votes-by-voting-place-<n>.csv` | Electoral Commission official results by voting place, one electorate each, for past elections. Ordinary votes by voting place, the three special-vote rows (overseas, before polling day, on polling day) and the total. |
| `<year>/<n>_<Name>.xlsx` | Preliminary (election-night) results by voting place for the election being forecast, with "Party" and "Candidate" sheets. |
| `preliminary-counts.csv` | Election-night candidate counts of past elections, read from the Commission's archived "electorate details" pages through the Internet Archive (the `source` column has each page). 2017 and 2023 are archived; 2020 is not. |

Every parser checks the file's arithmetic (each voting place, the special-vote rows and the electorate total) and
raises an error on any mismatch, so a changed layout cannot silently drop votes. Only the Commission's three
special-vote rows count as special votes: a voting place such as "Central Auckland Specialist School" does not.
The Commission's workbooks store counts as text, which the reader handles; it needs no spreadsheet library.

The party vote and the House use `data/reference/special-votes/` (`parliament.data_dir`):

| File | What it is |
|---|---|
| `party-votes.csv` | The country's party votes at each official count since 2008, by party: ordinary votes and overseas, before-polling-day and polling-day special votes. Summed from the Commission's party votes by voting place for every electorate (Internet Archive copies). |
| `preliminary-party-votes.csv` | Election-night party votes: the election being forecast, and 2017, 2020 and 2023 from the Commission's archived pages. |
| `electorates-<year>.csv` | Every electorate's election-night leader, party and margin; the runner-up's party where a seat is close; and National's, Labour's and all candidates' votes for the close National–Labour seats that have no workbook. |

The Commission's results site refuses automated requests, so the 2026 figures were read from RNZ's and Wikipedia's
copies of its preliminary results. To rebuild `party-votes.csv`, put each election's files in
`<folder>/<year>/party-votes-by-voting-place-<n>.csv` and run
`python -m pollofpolls.specials.national <folder> data/reference/special-votes/party-votes.csv`; every file is
checked against its own total row.

Any electorate whose files are in the folder joins the history automatically, so adding more past electorates
(all 72 seats of an election, say) sharpens the estimates of how much seats differ, at no other cost.

## The method

final = election-night count + official count's change to the ordinary votes + special votes. Votes are grouped
as [National's candidate, Labour's candidate, everyone else] (`pair`), and shares are log-ratios against National's.

**Special-vote volume.** log(specials / ordinary votes) = year + seat + noise, fitted to past elections. For the
forecast, the last election's level moves by the national change: the Commission's estimate of special votes still
to count (inflated by `estimate_error`, since its election-night estimates fell short by 5 to 6% in 2020 and 2023)
per ordinary vote, against the last official count. The overseas share is modelled the same way.
`rule_change_sd` adds seat-level uncertainty for the 2025 enrolment change.

**Special-vote split.** For seat *s*, category *c* (overseas or domestic), the specials' log-ratios are
`x_s + mu_c + alpha + gamma_c + beta_s + eps_sc`: the seat's own ordinary vote, a category mean, the election-wide
shift (LKJ-correlated between Labour's and Other's log-ratio), the election's category deviation, the seat's
persistent difference and noise. Counts are multinomial. For the forecast election, `alpha` is a fresh draw with
Student-t tails (`year_df`), as the poll model draws its election-day industry error.

**Enrolment change.** The Electoral Amendment Act 2025 closed enrolment 13 days before election day. The share of
domestic special votes this removed follows from the national figures; their lean against the domestic specials
that remain is an editorial assumption (`rule_change.lean`), applied to domestic specials only.

**Official count's change to ordinary votes.** Fitted to the archived election-night counts: a mixture of a small
symmetric change (rechecks) and, with probability `jump_prob`, a log-normal addition of votes missing on election
night, capped at `max_addition`. The added votes split like the election-night count shifted by a normal
deviation, which is integrated out analytically (a linearised softmax), so the model has no per-seat latent
variables that two or three seats could not identify.

**The party vote** (`partyvote.py`). For each party against National, the lean of a category's special votes is
log(special votes of the party / of National) minus the same for ordinary votes. Across the six elections it is
modelled as a usual lean per category and party, plus the election's shift common to every party, plus the
election's shift for that party, with overseas and domestic votes parting by a further amount. The forecast
election's shifts are new Student-t draws. The official count's addition to the ordinary party votes (about 2% in
each archived count, leaning slightly from National) is repeated from past counts. `rule_change.party_lean` applies
the enrolment change to each party's domestic specials.

**The link.** The electorate model's election-wide shift is `slope x national lean + remainder`, where the
national lean is the party-vote model's domestic lean of Labour, and of every other party pooled, centred on its
average (`priors.link`). The seats therefore borrow the spread of six elections instead of three and move with the
party vote inside each simulation.

**The House** (`parliament.py`). Each simulation has final party votes and a winner for every electorate: the
simulated winner in the seats under `electorates`, in every other National–Labour seat within
`parliament.modelled_margin` votes (from `electorates-<year>.csv`) and, by an assumed probability, in
`parliament.assumed_seats`; the election-night leader elsewhere. Seats follow by Sainte-Laguë with the threshold and
overhang. `government` has a majority with more than half the House, `opposition` likewise, and anything else is
"neither" (an exact tie when the two cover every seat). Routes describe each simulation's change from the
election-night House by the only things that move the margin: list seats lost by governing parties (2 each), and
for a party in overhang its entitlement and its electorates (1 each), less any growth in the other side's overhang.

All models are fitted with NUTS (NumPyro). Simulations draw one parameter set, one national lean and election-wide
shift shared by every seat, then each seat's change to ordinary votes, special-vote count, overseas share and
multinomial split, and the country's party votes.

## Backtest

Each past election is forecast with the models refitted on the others, from its election-night count (or its
official ordinary votes where none is archived), and scored on the final margin: CRPS, absolute error, 50% and 90%
coverage, Brier score and log loss of the winner. Three rules of thumb get the same expected number of specials:
specials split like election night, the seat's swing at its previous election, and the average swing. Each gets a
normal error with its root-mean-square error on the training elections. With twelve seat-elections and no seat
that changed hands, it is a sanity check: the model should be calibrated and at least as good on CRPS.

The party vote is backtested the same way over all six elections (`party_vote_backtest.csv`): each party's final
share against the model's median and intervals, against "specials split like election night" and against the
average past lean, and, given the electorates each party won, whether the seat count came out right.

## Changing it

| To | Edit |
|---|---|
| forecast another seat | add its preliminary workbook to `<data_dir>/<year>/` and an entry under `electorates`, with every candidate's party |
| update the national estimate | `volume.forecast` (`special_votes_estimate`, `total_votes_estimate`, `overseas_votes`) |
| change who counts as each side | `parliament.government`, `parliament.opposition` |
| change a close seat the model cannot simulate | `parliament.assumed_seats` (`p_flip`) |
| simulate more or fewer seats | `parliament.modelled_margin`, with candidate votes in `electorates-<year>.csv` |
| forecast the seats without the House | remove the `parliament` section |
| change the enrolment assumption | `rule_change.lean`; 0 means and sds switch it off |
| widen or narrow the election-wide shift | `priors.year_sd_scale`, `priors.year_df` |

**When the official results arrive**, add the forecast election's official CSVs to `<data_dir>/<year>/`, its
election-night counts to `preliminary-counts.csv`, rebuild `party-votes.csv` with the new election, and move
`election` (and the volume reference) on: the finished
election becomes history for the next one.

## Outputs

See [outputs](outputs.md#outputspecials-special-votes).
