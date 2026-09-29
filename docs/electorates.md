# Editorial assumptions

The party-vote polls say nothing about who wins a particular electorate, yet in New Zealand a handful of
electorates can decide the result: winning one brings a party under the 5% threshold into Parliament, and a
seat won by a candidate outside the parties comes out of the 120 shared proportionally. Those outcomes are set
by hand in [`config/electorates.yml`](../config/electorates.yml), along with the coalitions reported and the
blocs in the balance-of-power table. This page is how to change them.

A change needs no refit. The whole loop takes about a minute:

```bash
# 1. edit config/electorates.yml
pollofpolls electorates                      # 2. check it, and see what each assumption now implies
pollofpolls forecast && pollofpolls report   # 3. rebuild the forecast and the website
git commit -am "Electorates: ..." && git push github main   # 4. publish; the workflow redeploys the site
```

## What `pollofpolls electorates` shows

It checks the whole file and lists every problem at once, or prints the assumptions and what they imply at the
latest forecast:

```text
config/electorates.yml: 11 electorates, 9 coalitions and 2 blocs; every check passed.
Boundaries: 2025 boundary review, in use from the 2026 election. Shared error per party: 0.8.

Electorate         Party               p  indep.  at share  chance now  indep. now
Epsom              ACT              0.90    0.00      8.5%        0.92        0.00
Waiariki           Te Pāti Māori    0.85    0.00      2.2%        0.84        0.00
...
Electorates won, at the latest forecast's election-day vote (mean and 90% range):
  Te Pāti Māori    3.7  (1-6)   at least one in 99% of simulations
```

*p* is what you set. *Chance now* is what the simulation actually uses: the seat's chance averaged over the
forecast's simulated national vote, which differs from *p* when the party is polling above or below the
`at_share` you judged it at.

## The file, section by section

### `electorates`: who wins which seat

```yaml
  - electorate: Te Tai Tonga
    party: Te Pāti Māori
    p: 0.15
    independent_p: 0.08
    note: Tākuta Ferris won it for Te Pāti Māori in 2023 and stands as an independent. ...
    source: Te Ao Māori News, 29 Sep 2026
    updated: 2026-09-29
```

| Key | Meaning |
|---|---|
| `electorate` | its name at the boundaries of the election being forecast; see [Boundaries](#boundaries) |
| `party` | the tracked party the seat is modelled for |
| `p` | the chance that party wins it, when its national vote is `at_share` |
| `independent_p` | optional: the chance a candidate outside the tracked parties wins it. The rest, 1 − p − independent_p, is everyone else. |
| `at_share`, `slope`, `group` | optional; normally set once per party in `party_defaults` |
| `note`, `source`, `updated` | your own record of why; the model ignores them |

List a seat only if it can change the seat count:

* **a party that could fall below 5% could win it**, like Te Pāti Māori's Māori seats or ACT's Epsom. An
  electorate brings such a party into Parliament with list seats in proportion to its party vote, and any
  electorates beyond that are overhang;
* **a candidate outside the tracked parties could win it**, like an independent MP or a party too small to be
  tracked (the Te Tai Tokerau Party in 2026). Put that chance in `independent_p` on the entry for the seat.

A party far above 5% winning an electorate changes nothing, so National's and Labour's seats need no entry.

### `party_defaults`: how a party's seats move with its vote

```yaml
party_defaults:
  Te Pāti Māori: {at_share: 0.022, slope: 0.40}
```

* `at_share` is the party's national vote you judged its `p` values at. Set it to the party's current poll
  average when you set the values, and update it when you revisit them.
* `slope` is how fast a seat's chance follows the national vote: the change in log-odds per percentage point.
  At 0.4, a seat that is an even chance at `at_share` is 60% with one more point and 40% with one fewer. 0 makes
  the seats ignore the national vote. Default 0.3.
* `group` names seats that share one error (next section). The default is the party, which is usually right.

An electorate entry can override any of them, for example a seat with a strong local candidate whose chances
depend less on the national vote: `slope: 0.1`.

### `electorate_group_sd`: how wrong the judgement can be

Judgement about a party's electorates tends to be wrong for all of them at once: Te Pāti Māori held none of the
Māori seats in 2017 and six in 2023. Each simulation shifts every seat in a group by one shared draw with this
standard deviation, on the log-odds scale. Each seat's average chance stays the `p` you set; what changes is
that the seats rise and fall together, which widens the range of seats a party can end up with. At 0.8, Te Pāti
Māori's chance of holding none of the seven is about four times what it would be with independent seats. 0 turns
it off.

### `coalitions`

```yaml
coalitions:
  - {name: "National + ACT + NZ First", parties: [National, ACT, NZ First]}
```

Any combination of parties. Each gets a majority probability on the website, judged per simulation against the
simulated House size, including overhang.

### `balance_of_power`

```yaml
balance_of_power:
  blocs:
    Right bloc: [National, ACT]
    Left bloc: [Labour, Green, Te Pāti Māori]
  pivots: [NZ First, TOP]
```

For each bloc: the chance of a majority alone, with each pivot on its own, only with two or more pivots
together, or not even then. A pivot cannot also be in a bloc.

## Recipes

**A new electorate poll.** Change `p` (and `independent_p` if an independent is standing), and record the poll
in `note` and `source` so the next person knows why. A single electorate poll a month out is weak evidence:
electorate polls underestimated Te Pāti Māori in 2023, when it won six of the seven Māori seats.

**A candidate withdraws, or a new one stands.** Adjust `p`; for an independent or a new small party, adjust
`independent_p` on that seat's entry.

**A new seat becomes relevant**, for example a small party's leader standing somewhere winnable. Add an entry;
if the party has no `party_defaults`, add one with its current poll average as `at_share`.

**A party is polling well away from where you judged it.** Nothing to do: `slope` moves the seats with the
simulated vote. Revisit `at_share` and the `p` values together when you next review them.

**A new coalition or bloc arrangement.** Edit `coalitions` or `balance_of_power`; any tracked party works.

## Boundaries

Electorate names must be ones in use at the election being forecast. The lists come from the Representation
Commission and live in [`config/boundaries.yml`](../config/boundaries.yml), with the boundary sets side by side:
the 2025 review for 2026, and the 2019–20 review for 2020 and 2023. For each review, `replaced` records where
every abolished electorate went. For 2026:

| 2023 electorate | Now in |
|---|---|
| Wellington Central | Wellington North; Brooklyn and Mount Cook to Wellington Bays |
| Rongotai | Wellington Bays |
| Ōhāriu | Wellington North, Kenepuru, Hutt South |
| Mana | Kenepuru, Kapiti |
| Ōtaki | Kapiti |
| Te Atatū, Kelston, New Lynn | Henderson, Glendene, Waitākere |
| Panmure-Ōtāhuhu | Ōtāhuhu |
| East Coast | East Cape (renamed) |
| Bay of Plenty | Mount Maunganui (renamed) |

After a boundary review, add a new boundary set at the top of `boundaries.yml` with its `from_election`, the
electorate lists and `replaced`, then run `pollofpolls electorates`: every assumption about an abolished seat is
reported with the seats that replaced it.

## When the check complains

Every problem is listed at once. Names match ignoring case and macrons, so `Te Pati Maori` is fine.

| Message | What to do |
|---|---|
| `Rongotai: abolished at the 2025 boundary review; its area is now in Wellington Bays` | use the new electorate's name, and check the judgement still holds for its new boundaries |
| `Welington Bays: not an electorate at the 2026 election; did you mean Wellington Bays?` | fix the spelling |
| `` `Te Tai Tokerau Party` is not a party the model knows `` | only tracked parties can hold a seat; put its candidate in `independent_p` on the seat's entry |
| `unknown key `indpendent_p`; did you mean independent_p?` | fix the key |
| `p (0.7) and independent_p (0.4) add up to more than 1` | the two are chances of different people winning the same seat |
| `listed twice` | keep one entry per electorate |
| `NZ First is both a pivot and in Right bloc` | a pivot sits outside the blocs |

Two things are notes rather than errors, so the same file works for a forecast of another election: a party the
configuration knows but the forecast does not track holds no seats (its electorates are skipped), and an
electorate from a later boundary set than the election forecast is skipped. `pollofpolls forecast` stops on an
error before simulating anything, and prints the notes.
