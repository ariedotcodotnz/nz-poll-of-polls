# Configuration

All settings live in four YAML files under `config/`. Changing anything that affects a fit (the data, a
variant, the priors or the MCMC settings) changes the fit's fingerprint, so the next `pollofpolls fit` refits
automatically.

Party names must be the canonical names used throughout the package: National, Labour, Green, ACT,
NZ First, Te Pāti Māori, TOP, New Conservative, United Future, Mana, Internet, Progressive, Advance NZ,
DemocracyNZ, Vision NZ, NZ Loyal and Sustainable NZ. Column headers on Wikipedia, such as `NAT`, `MRI`, `TPM`
or `OPP`, are mapped to these names in `src/pollofpolls/data/parties.py`.

## elections.yml

| Key | Type | Meaning |
|---|---|---|
| `elections` | list | one entry per election; see below |
| `pm_history` | list of `{year, pm_party}` | Prime Minister's party at elections before 2008, for the fundamentals prior |
| `anchor_election` | year | first election used; the state starts at its result and earlier polls are ignored |
| `campaign_weeks` | integer | weeks up to each election whose volatility is scaled by the campaign multiplier |
| `in_parliament` | map of year to parties | parties always tracked in the term ending in that year |
| `auto_track` | `{min_share, min_polls}` | also track a party with at least `min_share` in at least `min_polls` polls of the term |
| `parliament_2026` | map | seats at dissolution; documentation only |
| `colours` | map of party to hex colour | chart colours; unlisted parties are grey. The defaults keep each party's usual hue but are re-stepped so every pair stays distinguishable on one chart, including for colour-blind readers. Check any change with a palette validator before using it. |

Each entry of `elections`:

| Key | Meaning |
|---|---|
| `year`, `date` | election year and polling day |
| `pm_party` | party of the Prime Minister going into the election |
| `forecast` | `true` for the election being forecast; exactly one entry |
| `wikipedia_page` | optional title of the polling article, when it is not "Opinion polling for the *year* New Zealand general election" |

Parties that are neither in Parliament nor auto-tracked are folded into "Other" for that term.

## pollsters.yml

| Key | Meaning |
|---|---|
| `defaults.sample_size` | sample size assumed when Wikipedia gives none and the pollster has no default |
| `defaults.min_polls` | pollsters with fewer eligible polls are dropped |
| `defaults.polling_code_only` | `true` keeps only signatories of the NZ Political Polling Code, as the 2023 model did |
| `defaults.publication_lag_days` | default days from the end of fieldwork to publication, used by backtests |
| `exclude_patterns` | regular expressions; matching releases are dropped. By default: polls commissioned by the Labour or National parties or by Business NZ, and UMR internal polls. Regular published series such as Taxpayers' Union–Curia are kept. |

Each entry of `pollsters`:

| Key | Meaning |
|---|---|
| `name` | canonical pollster name used in outputs |
| `match` | case-insensitive regular expression tested against the Wikipedia "Polling organisation" text. The first match wins; an unmatched name is kept as it is. |
| `sample_size` | default sample size for this pollster |
| `polling_code` | signatory of the NZ Political Polling Code |
| `publication_lag_days` | typical delay to publication, overriding the default |
| `method_changes` | dates of methodology changes. Each date starts a new house-effect segment, as for Reid Research's 2017 switch to a partly online sample. |

## electorates.yml

Electorate assumptions for the 2026 seat simulation. They are editorial judgements, not model output.

Each entry of `electorates`:

| Key | Meaning |
|---|---|
| `electorate` | name, for readability |
| `party` | the party whose win matters for the allocation |
| `p` | probability the party wins when its national vote equals `at_share` |
| `at_share` | the reference national vote share for `p` |
| `slope` | change in the logit of `p` per percentage point of simulated party vote above `at_share` |
| `independent_p` | optional probability that an independent or non-list candidate wins at `at_share` |

`p + independent_p` must not exceed 1; the remainder is the chance that anyone else wins. List an electorate
only if its result can change the proportional allocation: a party that may fall below 5%, or a possible
independent winner. Electorates won by parties safely above 5% make no difference.

`coalitions` lists the combinations reported, each with a `name` and a list of `parties`. It includes
combinations with TOP on either side, because TOP calls itself centrist and may clear the threshold.

`balance_of_power` defines the analysis of who decides the government:

| Key | Meaning |
|---|---|
| `blocs` | map of bloc name to its core parties, by default National + ACT and Labour + Green + Te Pāti Māori |
| `pivots` | parties that could support either bloc, by default NZ First and TOP; any number |

For each bloc the website gives the chance of a majority alone, with each pivot party on its own, only with
two or more pivots together (with two pivots, both), or not even then.

## model.yml

`variants` defines each model variant:

| Key | Values | Meaning |
|---|---|---|
| `obs` | `dm`, `gaussian`, `kalman`, `legacy` | likelihood; `legacy` is the 2023 model replica and ignores every other key |
| `house` | `constant`, `cycle` | persistent house effects only, or plus per-term deviations |
| `industry_error` | boolean | common polling error moving from the start to the end of each term |
| `campaign_kappa` | boolean | faster movement in the campaign weeks |
| `innovations` | `normal`, `student_t` | shape of the weekly shocks |
| `innovation_df` | number | degrees of freedom for `student_t`; default 4 |
| `fundamentals` | boolean | Ellis's prior on the Prime Minister's party swing |
| `fixed_n` | number | optional: treat every poll as this sample size |
| `design_effect` | number | optional: fix the design effect instead of learning it per pollster |

| Key | Meaning |
|---|---|
| `ensemble` | candidate variants for the published forecast, weighted by stacking; see [How it is tested](https://ariedotcodotnz.github.io/nz-poll-of-polls/evaluation.html) |
| `mcmc` | `chains`, `warmup`, `samples`, `max_tree_depth`, `target_accept` and `seed` for live fits |
| `priors` | prior scales; see the table below |
| `election_obs_sd` | log-ratio noise on election results, used only by the Kalman variant |
| `backtest.variants`, `targets`, `horizons_weeks` | what the backtests run. GitHub Actions starts one job per election in `targets`. |
| `backtest.blocs` | per target election, the `right` and `left` parties compared by the bloc-lead Brier score; others default to National + ACT against Labour + Green |
| `backtest.mcmc` | MCMC settings that override `mcmc` in backtests |
| `seats.n_sims` | number of simulated elections |
| `forecast.calibrate_spread` | apply the backtest-fitted spread calibration; off, see [How it is tested](https://ariedotcodotnz.github.io/nz-poll-of-polls/evaluation.html) |

Priors:

| Key | Default | Meaning |
|---|---|---|
| `sigma_scale` | 0.06 | HalfNormal scale of the weekly volatility of each log-ratio |
| `lkj_eta` | 2 | LKJ concentration of the innovation correlations |
| `house_base_sd` | 0.12 | sd of persistent house effects on the logit scale, about 3 points for a party on 40% |
| `house_cycle_sd_scale` | 0.05 | HalfNormal scale of the per-term house-effect sd |
| `industry_sd_scale` | 0.15 | HalfNormal scale of the start- and end-of-term industry-error sds |
| `kappa_sd` | 0.3 | LogNormal sd of the campaign multiplier |
| `design_effect_median`, `design_effect_sd` | 2, 0.4 | LogNormal prior on each pollster's design effect minus one |
| `fundamentals.df` | 4 | degrees of freedom of the fundamentals prior; its mean and sd are estimated |
