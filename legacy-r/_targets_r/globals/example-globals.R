library(targets)
library(tarchetypes)
library(stantargets)
options(tidyverse.quiet = TRUE)
tar_source() # grab all functions in R folder
tar_option_set(packages = c(
  "readr",
  "tidyr",
  "rvest",
  "glue",
  "dplyr",
  "stringr",
  "lubridate",
  "forcats",
  "ggplot2",
  "svglite",
  "scales",
  "janitor",
  "ggthemes",
  "nzelect",
  "purrr"
))
