simulate_seats <- function(sims) {
  t(sapply(1:nrow(sims), function(i) {
    allocate_seats(
      votes = as.numeric(sims[i,]),
      electorate = c(1, 1, 1, 1, 0, 1, 0),
      parties = names(sims)
    )$seats_v
  })) |>
    as_tibble() |>
    mutate(
      NatCoal = ACT + National,
      LabGreen = Labour + Green,
      LabGreenMaori = Labour + Green + `Te Pāti Māori`,
      NatCoalMaori = NatCoal + `Te Pāti Māori`
    )
}

list(
  tar_target(
    weekly_mu,
    model2023_mcmc_model2023$draws('mu', format = 'draws_matrix')
  ),
  tar_target(
    sims_election_night,
    weekly_mu[, 1:8 * 473] |> as_tibble() |> set_names(parties_ss) |>
      select(all_of(sort(parties_ss))) |>
      select(-Other) 
  ),
  tar_target(
    sims_saturday,
    weekly_mu[, 1:8 * 473 - 24] |> as_tibble() |> set_names(parties_ss) |>
      select(all_of(sort(parties_ss))) |>
      select(-Other) 
  ),
  tar_target(seats_election_night, simulate_seats(sims_election_night)),
  tar_target(seats_saturday, simulate_seats(sims_saturday))
)
