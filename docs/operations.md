# Operations

## GitHub Actions

`.github/workflows/pipeline.yml` runs on every push to `main`, every Monday at 09:00 New Zealand standard time
(Sunday 21:00 UTC), and on demand from the Actions tab.

| Job | Runs | Does |
|---|---|---|
| `test` | always | installs the package and runs the test suite |
| `backtest-plan` | only on demand, with "Re-run the rolling-origin backtests" ticked | reads the backtest elections from `config/model.yml` |
| `backtest` | after `backtest-plan` | one job per election, in parallel. Each starts from an empty `output/backtest/` and uploads only its own election's cases and fits. |
| `backtest-aggregate` | after `backtest` | checks every configured election arrived, scores all cases, refits the stacking weights and commits `output/backtest/` |
| `run` | after `test`, and after `backtest-aggregate` when that ran | fetch, prep, fit, forecast and report; commits `output/*.csv` and `output/*.json`; uploads `site/` |
| `deploy` | after `run` | publishes `site/` to GitHub Pages |

Fits are cached between runs with `actions/cache`, keyed on the package source and `config/`. A fit is reused
only if its fingerprint matches, so a run with no new polls finishes quickly.

The bot's commits are pushed with the workflow's own token, which does not trigger another run.

### Publishing

Pages is enabled with Settings > Pages > Source set to "GitHub Actions", and the report is published at
<https://ariedotcodotnz.github.io/nz-poll-of-polls/>. A fork needs the same setting, and its site appears at
`https://<owner>.github.io/<repository>/`. No secrets are needed.

The `deploy` job runs whenever `run` succeeds, including the usual case where the optional backtest jobs were
skipped. Every run replaces the published site, so the page always matches the latest committed outputs.

### Running the backtests

Open Actions > Poll of Polls > Run workflow, tick the backtest box and run. The three election jobs take about an
hour or two each on hosted runners. The new stacking weights are committed first, and the `run` job that follows
uses them. Re-run the backtests after any change to the model code or priors; the weekly runs do not.

## During the campaign

| Situation | What to do |
|---|---|
| New polls on Wikipedia | nothing; the weekly run fetches them, or push to run now |
| An electorate outlook changes | edit `config/electorates.yml`, then `pollofpolls forecast && pollofpolls report`. No refit is needed. |
| A new pollster appears | it is kept under its Wikipedia name. Add an entry to `config/pollsters.yml` to give it a canonical name, a default sample size and a publication delay. |
| A pollster changes method | add the date to its `method_changes`; its house effect starts a new segment |
| A small party surges | it is tracked automatically once it reaches 2.5% in four polls of the term. Add its colour to `config/elections.yml`. |
| `prep` stops on a results mismatch | a result row on Wikipedia was edited. Check it against the Electoral Commission before changing anything. |
| `prep` finds no party-vote table | the page layout changed. See [Development](development.md). |

## Preparing the next election

After the 2026 results are official:

1. **Election calendar.** In `config/elections.yml`, remove `forecast: true` from 2026 and add the next election
   with `forecast: true`, its date or best estimate, and the Prime Minister's party. Add its `in_parliament`
   list. While the article is still titled "Opinion polling for the next New Zealand general election", set
   `wikipedia_page` to that title.
2. **Reference data.** Add the 2026 party-vote shares to `data/reference/election_results.csv` and the
   electorates won to `data/reference/electorate_seats.csv`.
3. **Assumptions.** Rewrite `config/electorates.yml` for the new Parliament, including the coalitions to report
   and the blocs and pivot parties for the balance-of-power table.
4. **Backtests.** Add 2026 to `backtest.targets` in `config/model.yml` and its seat blocs to `backtest.blocs`,
   then re-run the backtests to refresh the stacking weights. The GitHub Actions workflow reads the same list, so
   it starts a job for 2026 and checks that its results arrive.
5. **Tests.** Add a fixture of the new polling table to `tests/fixtures/`, add the new page year to the `YEARS`
   lists in the tests, and update the result rows expected from the 2026 page in
   `test_every_page_parses_with_election_rows`.

The page list, report title, chart window and CI backtest jobs all follow the configuration, so the only code
to change is the tests. Calendar entries after the forecast election are ignored, so the next election can be
added early without affecting the current forecast.
