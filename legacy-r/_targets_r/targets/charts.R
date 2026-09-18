list(
  tar_target(
    voting_intention_chartdata,
    model2023_summary_model2023 |>
      filter(str_starts(variable, "mu")) |>
      mutate(
        Party = rep(parties_ss, each = sum(weeks_between_elections)),
        week = rep(1:sum(weeks_between_elections), length(parties_ss)),
        week = min(election_weeks) + weeks(week)
      )
  ),
  tar_target(
    voting_intention_chart,
    voting_intention_chartdata |>
      ggplot(aes(
        x = week,
        y = mean,
        colour = Party,
        fill = Party
      )) +
      geom_point(
        data = gather(
          polls2,
          Party,
          VotingIntention,-Pollster,-MidDate,-ElectionYear,-MidDateNumber
        ) |>
          filter(VotingIntention != 0),
        aes(x = MidDate, y = VotingIntention),
        colour = "black",
        size = 0.5
      ) +
      geom_vline(
        xintercept = as.numeric(election_dates),
        colour = "grey60"
      ) +
      scale_y_continuous(labels = percent) +
      scale_x_date(breaks = ym(
        c('2016-01', '2018-01', '2020-01', '2022-01')
      ), date_labels = '%Y') +
      scale_fill_manual(values = party_colours$Colour, breaks = party_colours$Party) +
      geom_line(
        data = voting_intention_chartdata |> filter(week <= today()),
        colour = "black"
      ) +
      geom_ribbon(aes(ymin = q5, ymax = q95), alpha = 0.3, colour = NA) +
      labs(y = NULL, x = NULL) +
      theme_clean() +
      theme(
        plot.title.position = "plot",
        legend.position = 'none',
        strip.background = element_rect(fill = '#121617'),
        strip.text = element_text(colour = 'white'),
        plot.background = element_blank()
      )
  ),
  tar_file(voting_intention620, {
    f <- "output/voting_intention620.svg"
    ggsave(
      f,
      plot = voting_intention_chart +
        facet_wrap(
          ~ factor(Party, levels = party_colours$Party),
          scales = "free",
          ncol = 2
        ),
      dpi = 100,
      width = 6.2,
      height = 8
    )
    f
  }),
  tar_file(voting_intention375, {
    f <- "output/voting_intention375.svg"
    ggsave(
      f,
      plot = voting_intention_chart +
        facet_wrap(
          ~ factor(Party, levels = party_colours$Party),
          scales = "free",
          ncol = 1
        ),
      dpi = 100,
      width = 3.75,
      height = 10
    )
    f
  })
)
