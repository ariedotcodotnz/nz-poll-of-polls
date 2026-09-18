skip_url_check <- FALSE
list(
  tar_url(polls2011_url, wikipedia_poll_url(2011), cue = tar_cue_skip(skip_url_check)),
  tar_url(polls2014_url, wikipedia_poll_url(2014), cue = tar_cue_skip(skip_url_check)),
  tar_url(polls2017_url, wikipedia_poll_url(2017), cue = tar_cue_skip(skip_url_check)),
  tar_url(polls2020_url, wikipedia_poll_url(2020), cue = tar_cue_skip(skip_url_check)),
  tar_url(polls2023_url, wikipedia_poll_url(2023), cue = tar_cue_skip(skip_url_check)),
  tar_url(results2008_url, election_results_summary(2008), cue = tar_cue_skip(skip_url_check)),
  tar_url(results2011_url, election_results_summary(2011), cue = tar_cue_skip(skip_url_check)),
  tar_url(results2014_url, election_results_summary(2014), cue = tar_cue_skip(skip_url_check)),
  tar_url(results2017_url, election_results_summary(2017), cue = tar_cue_skip(skip_url_check)),
  tar_url(results2020_url, election_results_summary(2020), cue = tar_cue_skip(skip_url_check))
)
