# Data

## Sources

| Source | Used for |
|---|---|
| Wikipedia, "Opinion polling for the *year* New Zealand general election", 2011 to 2026 | every published poll, and the official result rows |
| `data/reference/election_results.csv` | official party-vote shares for 2008 to 2023, plus National and Labour from 1993 to 2005 for the fundamentals prior (1993 was a first-past-the-post election) |
| `data/reference/electorate_seats.csv` | electorates won by each party from 2011 to 2023, for the seat-allocation tests and the backtests' bloc score |

The 2023 R pipeline downloaded results from the Electoral Commission's CSV files. Those URLs now sit behind a
JavaScript challenge that scripts cannot pass, so the reference files are committed instead and checked against
the Wikipedia result rows on every run.

## Parsing the polling tables

The table layout differs between pages. The pollster column is "Poll" or "Polling organisation", parties
appear as full names or abbreviations such as `NAT`, `MRI`, `TPM` or `OPP`, and only some pages have sample size
and lead columns. The parser does not rely on table order or position: it uses the largest table whose header has
a date column, a pollster column and at least three recognisable party columns.

Within that table:

* Rows with far fewer cells than the header are event notes, such as "Budget 2026 is delivered", and are
  skipped.
* Footnote markers like `[a]` or `[nb 1]` are removed from every cell.
* Rows labelled "*year* election result" become result rows, not polls.
* "–", "—", "N/A", "?" and empty cells mean the party was not reported. "<1" is read as 0.5%.
* Sample sizes such as `1,000` or `1,000+` are read as integers and kept if between 100 and 100,000.

Date formats handled:

| Example | Read as |
|---|---|
| `17 Oct 2020` | a single day |
| `4–11 Sep 2026` | range within a month |
| `27 Jul – 23 Aug 2026` | range across months |
| `28 Dec 2019 – 5 Jan 2020`, `24 Dec – 5 Jan 2020` | range across years |
| `Sep 2020` | the whole month |
| `Early Apr 2017`, `Mid …`, `Late …` | days 1–10, 11–20, 21 to month end |
| `2–7, 14–15 Mar 2022` | the whole span, 2 to 15 March |
| `31 Sep – 11 Oct 2015` | a typo on the page, clamped to 30 September |

A poll's midpoint places it on the weekly timeline. Weeks start on Sunday, as in the 2023 pipeline.

## Cleaning

**Pollsters.** Media partners and firms change names, so pollsters are canonicalised with the regular
expressions in `config/pollsters.yml`:

| Canonical name | Labels on Wikipedia |
|---|---|
| Verian | 1 News–Colmar Brunton, One News Colmar Brunton, 1 News–Kantar Public, 1 News–Verian |
| Reid Research | 3 News Reid Research, Newshub–Reid Research, RNZ–Reid Research |
| Curia | Taxpayers' Union–Curia, Curia |
| Roy Morgan | Roy Morgan Research, Roy Morgan |
| Freshwater Strategy | The Post–Freshwater Strategy, The Post/Freshwater Strategy |
| DigiPoll | Herald–DigiPoll |
| Ipsos | Fairfax Media Ipsos |
| Research International | Fairfax Media–Research International |
| Bauer Media Insights | Listener: Bauer Media Insights |

Talbot Mills, Horizon Research, Guardian Essential, YouGov, SSI and Anacta keep their own names.

**Exclusions.** Polls commissioned by the Labour or National parties or by Business NZ, and UMR internal polls,
are dropped because they are published selectively. For example, "Labour–Talbot Mills" polls are dropped, while
Talbot Mills' own series is kept. A pollster with fewer than two eligible polls is also dropped from a fit.

**Terms.** A poll belongs to the first election on or after its midpoint. Its position within the term is used
by the industry-error term of the model.

**Duplicates.** A poll that appears on two pages is kept once, matched on pollster, midpoint and party.

**Missing parties.** A party a poll did not report is recorded as unobserved, never as 0%. "Other" is one minus
the sum of the reported tracked parties.

**Availability.** Each poll has an `available` date: fieldwork end plus the pollster's typical publication delay.
Backtests use only polls available by their cutoff; the live forecast uses every poll on the page.

## Verifying results

`prep` compares the result rows scraped from every page with `data/reference/election_results.csv`. A difference
of more than 0.06 percentage points for any party stops the pipeline. When both sources have a value, the
reference value is used; the scraped values fill in smaller parties the reference file omits.

## What the data look like

As of 19 September 2026, after exclusions:

| Term ending | Polls | Sample size reported |
|---|---|---|
| 2011 | 127 | none; before the anchor election, so not used by the model |
| 2014 | 136 | none |
| 2017 | 79 | none |
| 2020 | 45 | 89% |
| 2023 | 119 | 86% |
| 2026 | 118 | 93% |

When no sample size is reported, the pollster's default from `config/pollsters.yml` is used. Roy Morgan
publishes most often, with 251 of the polls since 2008.

## Alternative source: NZPolls

[Nixinova/NZPolls](https://github.com/Nixinova/NZPolls) (public domain) hand-transcribes the same Wikipedia
tables into YAML with a browser paste tool. In September 2026 it agreed with this parser on 3,604 of the 3,632
party figures in the 467 polls both sources contain. Most disagreements were in a few Horizon Research polls.
One, a Verian poll in September 2023, looks like a copying slip in NZPolls: it shows decimals Verian does not
publish for large parties.

It is not used as the input for three reasons:

* It is updated by hand about monthly, and lagged the six most recent polls, which count most near an election.
* It adds no independent information, because it is copied from the same tables.
* Its unknown days are written as `-00`, and one date is impossible (`2015-09-31`), so standard YAML loaders
  reject the file.

It would be a useful automated cross-check against this parser.
