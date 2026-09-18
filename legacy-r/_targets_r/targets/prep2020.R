list(
  tar_target(election_weeks2020, floor_date(ymd(
    c("2011-11-26", "2014-09-20", "2017-09-23", "2020-10-17")
  ), unit = 'week')),
  tar_target(weeks_between_elections2020, as.integer(diff(election_weeks)) / 7),
  tar_target(
    elections2020,
    results |>
      filter(Election >= year(min(election_weeks2020)) & 
               Election < year(max(election_weeks2020))) |>
      mutate(
        Party = fct_other(Party, keep = parties2020$Party) |>
          fct_relevel(parties2020$Party)
      ) |>
      count(Party, Election, wt = Percentage, name = 'Percentage') |>
      arrange(Party, Election) |>
      pivot_wider(names_from = Party, values_from = Percentage, values_fill = 0) |>
      select(-Election)
  ),
  tar_target(
    polls22020,
    polls |>
      filter(Election %in% c(2014, 2017, 2020),
             Pollster %in% c(pollsters2020$Pollster),
             !is.na(MidPoint)) |>
  mutate(
    MidPoint = floor_date(MidPoint, unit = 'week'),
    Party = fct_other(Party, keep = parties2020$Party) |>
      fct_relevel(parties2020$Party)
  ) |>
  filter(Party != 'Other') |> 
  mutate(Polled = sum(VotingIntention), .by = c(Pollster, MidPoint)) |> 
  rename(MidDate = MidPoint, ElectionYear = Election) |>
  count(
    Party,
    Pollster,
    MidDate,
    ElectionYear,
    Polled,
    wt = VotingIntention,
    name = "VotingIntention"
  ) |>
  filter(MidDate <= election_weeks2020[4]) |>
  arrange(Party, MidDate) |>
  pivot_wider(
    names_from = Party,
    values_from = VotingIntention,
    names_sort = TRUE,
    values_fill = 0
  ) |>
  mutate(Other = pmax(0, 1 - Polled),
         MidDateNumber = 1 + as.numeric(MidDate - election_weeks2020[1]) / 7) |> 
  select(-Polled)

  ),
  tar_target(
    polls32020,
    polls22020 |>
      arrange(Pollster, MidDate) |>
      group_split(Pollster)
  ),
  tar_target(parties_ss2020, names(elections2020)),
  tar_target(
    # estimate the standard errors.  Note we are pretending they all have a sample size of 1000 -
    # which the main five do, but not some of the smaller ones.  Improvement would be to better deal with this.
    
    all_ses2020,
    polls22020 |>
      select(Pollster, ACT:Other) |>
      pivot_longer(ACT:Other, names_to = 'Party', values_to = 'p') |>
      summarise(
        p = mean(p, na.rm = T),
        se = sqrt(p * (1 - p) / 1000),
        .by = c(Pollster, Party)
      )
  ),
  tar_target(
    ses32020,
    all_ses2020 |>
      arrange(Pollster, Party) |>
      group_split(Pollster)
  ),
  tar_target(
    d12020, list(mu_elect1 = as.numeric(elections2020[1, ]), 
           mu_elect2 = as.numeric(elections2020[2, ]),
           mu_elect3 = as.numeric(elections2020[3, ]), 

           n_parties = length(parties_ss2020),
           n_weeks = weeks_between_elections2020, 
           # multiply the variance of all polls by 2.  See my blog post of 9 July 2017.
           inflator = sqrt(2),
           
           y1_n = nrow(polls32020[[1]]),
           y1_values = polls32020[[1]][ , 4:9],
           y1_weeks = as.numeric(polls32020[[1]]$MidDateNumber),
           y1_se = ses32020[[1]]$se,
           
           y2_n = nrow(polls32020[[2]]),
           y2_values = polls32020[[2]][ , 4:9],
           y2_weeks = as.numeric(polls32020[[2]]$MidDateNumber),
           y2_se = ses32020[[2]]$se,
           reid_method = as.numeric(polls32020[[2]]$MidDate >= as.Date("2017-01-01")),
           
           y3_n = nrow(polls32020[[3]]),
           y3_values = polls32020[[3]][ , 4:9],
           y3_weeks = as.numeric(polls32020[[3]]$MidDateNumber),
           y3_se = ses32020[[3]]$se,
           
           n_pollsters = 3)
  )
)

