list(
  tar_target(election_dates, ymd(c("2014-09-20", "2017-09-23", "2020-10-17", "2023-10-14"))),
  tar_target(election_weeks, floor_date(election_dates, unit = 'week')),
  tar_target(weeks_between_elections, as.integer(diff(election_weeks)) / 7),
  tar_target(
    elections,
    results |>
      filter(Election >= year(min(election_weeks))) |>
      mutate(
        Party = fct_other(Party, keep = parties$Party) |>
          fct_relevel(parties$Party)
      ) |>
      count(Party, Election, wt = Percentage, name = 'Percentage') |>
      arrange(Party, Election) |>
      pivot_wider(names_from = Party, values_from = Percentage, values_fill = 0) |>
      select(-Election)
  ),
  tar_target(
    polls2,
    polls |>
  filter(Pollster %in% c(pollsters$Pollster), !is.na(MidPoint)) |>
  mutate(
    MidPoint = floor_date(MidPoint, unit = 'week'),
    Party = fct_other(Party, keep = parties$Party) |>
      fct_relevel(parties$Party)
  ) |>
  filter(Party != 'Other') |> 
  mutate(Polled = sum(VotingIntention), .by = c(Pollster, MidPoint)) |> 
  rename(MidDate = MidPoint, ElectionYear = Election) |>
  select(
    `Date range`,
    Party,
    Pollster,
    MidDate,
    ElectionYear,
    Polled,
    VotingIntention
  ) |>
  filter(ElectionYear %in% year(election_weeks[-1])) |>
  arrange(Party, MidDate) |>
  pivot_wider(
    names_from = Party,
    values_from = VotingIntention,
    names_sort = TRUE,
    values_fill = 0
  ) |>
  mutate(Other = pmax(0, 1 - Polled),
         MidDateNumber = 1 + as.numeric(MidDate - election_weeks[1]) / 7) |> 
  select(-Polled, -`Date range`)
),
  tar_target(
    polls3,
    polls2 |>
      arrange(Pollster, MidDate) |>
      group_split(Pollster)
  ),
  tar_target(parties_ss, names(elections)),
  tar_target(
    # estimate the standard errors.  Note we are pretending they all have a sample size of 1000 -
    # which the main five do, but not some of the smaller ones.  Improvement would be to better deal with this.
    
    all_ses,
    polls2 |>
      select(Pollster, ACT:Other) |>
      pivot_longer(ACT:Other, names_to = 'Party', values_to = 'p') |>
      summarise(
        p = mean(p, na.rm = T),
        se = sqrt(p * (1 - p) / 1000),
        .by = c(Pollster, Party)
      )
  ),
  tar_target(
    ses3,
    all_ses |>
      arrange(Pollster, Party) |>
      group_split(Pollster)
  ),
  tar_target(
    d1, list(mu_elect1 = as.numeric(elections[1, ]), 
           mu_elect2 = as.numeric(elections[2, ]),
           mu_elect3 = as.numeric(elections[3, ]), 

           n_parties = length(parties_ss),
           n_weeks = weeks_between_elections, 
           # multiply the variance of all polls by 2.  See my blog post of 9 July 2017.
           inflator = sqrt(2),
           
           y1_n = nrow(polls3[[1]]),
           y1_values = polls3[[1]][ , 4:11],
           y1_weeks = as.numeric(polls3[[1]]$MidDateNumber),
           y1_se = ses3[[1]]$se,
           
           y2_n = nrow(polls3[[2]]),
           y2_values = polls3[[2]][ , 4:11],
           y2_weeks = as.numeric(polls3[[2]]$MidDateNumber),
           y2_se = ses3[[2]]$se,
           
           y3_n = nrow(polls3[[3]]),
           y3_values = polls3[[3]][ , 4:11],
           y3_weeks = as.numeric(polls3[[3]]$MidDateNumber),
           y3_se = ses3[[3]]$se,
           
           y4_n = nrow(polls3[[4]]),
           y4_values = polls3[[4]][ , 4:11],
           y4_weeks = as.numeric(polls3[[4]]$MidDateNumber),
           y4_se = ses3[[4]]$se,
           reid_method = as.numeric(polls3[[4]]$MidDate >= as.Date("2017-01-01")),
           
           n_pollsters = 4)
  )
)

