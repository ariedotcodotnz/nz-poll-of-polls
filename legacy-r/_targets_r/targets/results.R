list(
  tar_target(results2020, fetch_results(results2020_url, 17, 2020)),
  tar_target(results2017, fetch_results(results2017_url, 16, 2017)),
  tar_target(results2014, fetch_results(results2014_url, 15, 2014)),
  tar_target(results2011, fetch_results(results2011_url, 13, 2011)),
  tar_target(results2008, fetch_results(results2008_url, 19, 2008)),
  tar_target(
    results,
    combine_results(
      results2008,
      results2011,
      results2014,
      results2017,
      results2020
    )
  )
  
)
