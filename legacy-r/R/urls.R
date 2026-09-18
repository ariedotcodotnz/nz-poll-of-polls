

wikipedia_poll_url <- function(election_year) {
  glue(
    "https://en.wikipedia.org/wiki/Opinion_polling_for_the_{election_year}_New_Zealand_general_election"
  )
}

election_results_summary <- function(election_year) {
  if (election_year < 2017) {
    return(
      glue(
        "https://www.electionresults.govt.nz/electionresults_{election_year}/e9/csv/e9_part1.csv"
      )
    )
  }
  glue(
    "https://www.electionresults.govt.nz/electionresults_{election_year}/statistics/csv/overall-results-summary.csv"
  )
}
