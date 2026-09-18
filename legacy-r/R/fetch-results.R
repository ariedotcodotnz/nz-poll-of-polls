
fetch_results <- function(url, num_parties, year) {
  read_csv(
    url,
    skip = 6,
    col_names = c('Party', 'Seats', 'Votes', 'Percentage'),
    col_types = cols_only(
      Party = col_character(),
      Seats = col_integer(),
      Votes = col_integer(),
      Percentage = col_number()
    ),
    n_max = num_parties
  ) |>
    mutate(Election = year, Percentage = Percentage/100)
}

combine_results <- function(...) {
  rlang::list2(...) |>
    purrr::list_rbind() |>
    dplyr::mutate(
      Party = party_name(Party)
    )
}
