coalition_plot <- function(seats) {
  seats  |>
    select(National,
           Labour,
           NatCoal,
           LabGreen,
           LabGreenMaori,
           NatCoalMaori) |>
    gather(Coalition, Seats) |>
    mutate(lab_in = ifelse(grepl("^Lab", Coalition), "Labour-based", "Nationals-based")) |>
    mutate(
      Coalition = gsub("^LabGreen", "Labour, Greens", Coalition),
      Coalition = gsub("^NatCoal", "National, ACT", Coalition),
      Coalition = gsub("Maori", ",\nTe Pāti Māori", Coalition)
    ) |>
    ggplot(aes(x = Seats, fill = lab_in)) +
    geom_histogram(
      alpha = 0.5,
      binwidth = 1,
      position = "identity",
      colour = NA
    )  +
    scale_y_continuous() +
    scale_fill_manual(values = c('#d82c20', '#065BAA')) +
    theme_clean() +
    theme(
      legend.position = 'none',
      strip.background = element_rect(fill = '#121617'),
      strip.text = element_text(colour = 'white'),
      plot.background = element_blank()
    ) +
    labs(y = NULL) +
    annotate(
      'segment',
      x = 61,
      xend = 61,
      y = 0,
      yend = Inf
    )
}

list(
  tar_target(election_night_plot, coalition_plot(seats_election_night)),
  tar_target(saturday_plot, coalition_plot(seats_saturday)),
  tar_file(election_night620, {
    f <- "output/election_night620.svg"
    ggsave(
      f,
      plot = election_night_plot +
        facet_wrap(~ factor(
          Coalition,
          levels = c(
            "Labour",
            "National",
            "Labour, Greens",
            "National, ACT",
            "Labour, Greens,\nTe Pāti Māori",
            "National, ACT,\nTe Pāti Māori"
          )
        ), ncol = 2),
      dpi = 100,
      width = 6.2,
      height = 4
    )
    f
  }),
  tar_file(election_night375, {
    f <- "output/election_night375.svg"
    ggsave(
      f,
      plot = election_night_plot +
        facet_wrap(~ Coalition,
                   ncol = 1),
      dpi = 100,
      width = 3.75,
      height = 7
    )
    f
  }),
  tar_file(saturday620, {
    f <- "output/saturday620.svg"
    ggsave(
      f,
      plot = saturday_plot +
        facet_wrap(~ factor(
          Coalition,
          levels = c(
            "Labour",
            "National",
            "Labour, Greens",
            "National, ACT",
            "Labour, Greens,\nTe Pāti Māori",
            "National, ACT,\nTe Pāti Māori"
          )
        ), ncol = 2),
      dpi = 100,
      width = 6.2,
      height = 4
    )
    f
  }),
  tar_file(saturday375, {
    f <- "output/saturday375.svg"
    ggsave(
      f,
      plot = saturday_plot +
        facet_wrap(~ Coalition,
                   ncol = 1),
      dpi = 100,
      width = 3.75,
      height = 7
    )
    f
  })
)
