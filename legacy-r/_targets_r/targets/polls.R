list(
  tar_target(polls2011, extract_poll_results_2011(polls2011_url)),
  tar_target(polls2014, extract_poll_results_2014(polls2014_url)),
  tar_target(polls2017, extract_poll_results_2017(polls2017_url)),
  tar_target(polls2020, extract_poll_results_2020(polls2020_url)),
  tar_target(polls2023, extract_poll_results_2023(polls2023_url)),
  tar_target(polls, combine_polls(polls2011,polls2014,polls2017,polls2020,polls2023))
)
