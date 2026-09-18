



.months <-
  stringr::str_flatten(c(month.name, month.abb), collapse =  '|')

.extract_date_from <- function(dates_text) {
  from_raw <- dates_text[1]
  to_raw <- dates_text[2]
  from_part <- from_raw |> str_extract(glue('\\d+[ {.months}]*'))
  to_raw |> str_replace(
    replacement = glue('{from_part}'),
    pattern = if_else(grepl(.months, from_part),
                      glue('\\d+ [{.months}]+'),
                      '\\d+')
  ) |>
    dmy(quiet = T)
}


wikipedia_date_parser_from <- function(date_text) {
  date_text |> purrr::map(
    ~ case_when(
      . == '2–7, 14–15 Mar 2022' ~ ymd('2022-02-15'),
      . == '31 Sep – 11 Oct 2015' ~ ymd('2015-09-30'), # Assume it is supposed to be 30 Sep
      grepl('–', .) ~ str_split_1(., '–') |>
        str_trim() |>
        .extract_date_from(),
      #grepl(glue('^{.months} 20\\d\\d$'), .) ~ my(.),
      TRUE ~ coalesce(dmy(., quiet = T), my(., quiet = T))
    )
  ) |> purrr::reduce(c)
}

wikipedia_date_parser_to  <- function(date_text) {
  date_text |> purrr::map(
    ~ case_when(
      . == '2–7, 14–15 Mar 2022' ~ ymd('2022-03-15'),
      . == '31 Sep – 11 Oct 2015' ~ ymd('2015-10-11'),
      grepl('–', .) ~ str_split_1(., '–') |>
        str_trim() |>
        last() |>
        dmy(quiet = T),
      !is.na(my(., quiet = T)) ~ my(., quiet = T) + months(1) - days(1),
      TRUE ~ dmy(., quiet = T)
    )
  ) |> purrr::reduce(c)
}

extract_poll_results <-
  function(url,
           table_num,
           pollster_col,
           date_col,
           rep_col) {
    rvest::read_html(url) |>
      rvest::html_table() |>
      dplyr::nth(table_num) |>
      dplyr::filter({{ pollster_col }} != {{ rep_col }},
                    !grepl('Polling organisation|election result|^Poll$',
               {{ pollster_col }})) |>
      mutate(From = wikipedia_date_parser_from(
        str_remove_all({{ date_col }}, "\\[.*\\]")),
      To = wikipedia_date_parser_to(
        str_remove_all({{ date_col }}, "\\[.*\\]")),
      MidPoint = From + (To - From)/2) |>
      rename(`Date range` = {{ date_col }}) |>
      rename(`Polling organisation` = {{pollster_col}}) |>
      pivot_longer(
        matches("^[A-Z][A-Z][A-Z]$|^National$|^Labour$|^Green$|^NZ First$|^Māori$|^Mana$|^United Future$|^Con$|^Internet$|^InternetMana$|^Prog$",
                ignore.case = FALSE),
        names_to = 'Party',
        values_to = 'VotingIntention',
        values_transform = ~ parse_number(., na = c('–', 'N/A'))
      ) |>
      mutate(VotingIntention = VotingIntention / 100)
  }


extract_poll_results_2011 <- function(polls2011_url) {
  extract_poll_results(polls2011_url,
                       1,
                       Poll,
                       `Date[nb 1]`,
                       `Date[nb 1]`)
}


extract_poll_results_2014 <- function(polls2014_url) {
  extract_poll_results(polls2014_url,
                       1,
                       Poll,
                       `Date[nb 1]`,
                       `Date[nb 1]`)
}

extract_poll_results_2017 <- function(polls2017_url) {
  extract_poll_results(polls2017_url,
                       1,
                       Poll,
                       `Date[nb 1]`,
                       NAT)
}

extract_poll_results_2020 <- function(polls2020_url) {
  extract_poll_results(polls2020_url,
                       1,
                       `Polling organisation`,
                       `Date[nb 1]`,
                       `Sample size`) |>
    select(-Lead) |>
    mutate(`Sample size` = parse_integer(
      str_remove_all(`Sample size`, '\\[\\d+\\]'),
      na = c('N/A', '', '–')
    ))
}

extract_poll_results_2023 <- function(polls2023_url) {
  extract_poll_results(polls2023_url,
                       2,
                       `Polling organisation`,
                       `Date[a]`,
                       `Sample size`) |>
    select(-Lead) |>
    mutate(`Sample size` = parse_integer(
      str_remove_all(`Sample size`, '\\[\\d+\\]|,|\\+'),
      na = c('N/A', '', '–')
    ))
}



extract_poll_results_2023_ <- function(polls2023_url) {
  rvest::read_html(polls2023_url) |>
    rvest::html_table() |>
    dplyr::nth(2) |>
    dplyr::filter(
      `Polling organisation` != `Sample size`,!grepl(
        'Polling organisation|election result',
        `Polling organisation`
      )
    ) |>
    select(-Lead) |>
    mutate(
      To = wikipedia_date_parser_to(`Date[a]`),
      From = wikipedia_date_parser_from(`Date[a]`),
      `Sample size` = parse_number(`Sample size`, na = c('N/A', '', '–'))
    ) |>
    pivot_longer(
      matches("[A-Z][A-Z][A-Z]$", ignore.case = FALSE),
      names_to = 'Party',
      values_to = 'Percent',
      values_transform = ~ parse_number(., na = c('–', 'N/A'))
    )
}


extract_poll_results_2020_ <- function(polls2020_url) {
  rvest::read_html(polls2020_url) |>
    rvest::html_table() |>
    dplyr::nth(1) |>
    dplyr::filter(
      `Polling organisation` != `Sample size`,!grepl(
        'Polling organisation|election result',
        `Polling organisation`
      )
    ) |>
    select(-Lead) |>
    mutate(
      To = wikipedia_date_parser_to(`Date[nb 1]`),
      From = wikipedia_date_parser_from(`Date[nb 1]`),
      `Sample size` = parse_number(`Sample size`, na = c('N/A', '', '–'))
    ) |>
    pivot_longer(
      matches("[A-Z][A-Z][A-Z]$", ignore.case = FALSE),
      names_to = 'Party',
      values_to = 'Percent',
      values_transform = ~ parse_number(., na = c('–', 'N/A'))
    )
}

combine_polls <- function(polls2011,polls2014,polls2017,polls2020,polls2023) {
  polls2023 |>
    mutate(Election = 2023) |>
    bind_rows(polls2020 |> mutate(Election = 2020)) |>
    bind_rows(polls2017 |> mutate(Election = 2017)) |>
    bind_rows(polls2014 |> mutate(Election = 2014)) |>
    bind_rows(polls2011 |> mutate(Election = 2011)) |>
    mutate(`Polling organisation` = str_remove_all(`Polling organisation`, "\\[.+\\]"),
           Pollster = case_when(
             grepl('Kantar|Colmar', `Polling organisation`) ~ 'Kantar Public',
             grepl('Reid Research', `Polling organisation`) ~ 'Reid Research',
             grepl('Curia', `Polling organisation`) ~ 'Curia',
             grepl('Horizon', `Polling organisation`) ~ 'Horizon Research',
             grepl('Roy Morgan', `Polling organisation`) ~ 'Roy Morgan',
             grepl('Digipoll|DigiPoll', `Polling organisation`) ~ 'Digipoll',
             grepl('Talbot Mills', `Polling organisation`) ~ 'Talbot Mills',
             grepl('Ipsos', `Polling organisation`) ~ 'Ipsos',
             grepl('SSI', `Polling organisation`) ~ 'SSI',
             grepl('Bauer Media Insights', `Polling organisation`) ~ 'Bauer Media Insights',
             grepl('Research International', `Polling organisation`) ~ 'Research International',
             grepl('YouGov', `Polling organisation`) ~ 'YouGov',
           .default = NA_character_
           ),
           Party = party_name(Party)) |>

    filter(!is.na(VotingIntention))
}

