# Static charts of the special-vote forecast in the BBC's house style (ggplot2 + bbplot::bbc_style).
#
#   Rscript r/specials_charts.R            # reads output/specials/, writes output/specials/charts/*.svg and *.png
#
# Needs ggplot2, jsonlite, svglite, ragg and bbplot (remotes::install_github("bbc/bbplot")). Run it after
# `pollofpolls specials`. Every number comes from forecast.json and the CSVs beside it.

suppressPackageStartupMessages({
  library(ggplot2)
  library(jsonlite)
  library(bbplot)
})

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
root <- normalizePath(file.path(here, ".."))
src <- file.path(root, "output", "specials")
out <- file.path(src, "charts")
dir.create(out, showWarnings = FALSE, recursive = TRUE)

fc <- fromJSON(file.path(src, "forecast.json"), simplifyVector = FALSE)
house <- fc$house
if (is.null(house)) stop("forecast.json has no House forecast: run `pollofpolls specials` with a parliament section")

# bbc_style() asks for Helvetica. Where it is not installed, draw the PNGs with the nearest face present.
if (requireNamespace("systemfonts", quietly = TRUE) && !"Helvetica" %in% systemfonts::system_fonts()$family) {
  sf <- systemfonts::system_fonts()
  for (fam in c("Arial", "Liberation Sans", "Inter", "Roboto", "DejaVu Sans")) {
    reg <- sf[sf$family == fam & sf$style %in% c("Regular", "Book"), ]
    bold <- sf[sf$family == fam & sf$style %in% c("Bold", "Semi Bold", "SemiBold"), ]
    bold <- bold[order(bold$style), ]
    if (nrow(reg) && nrow(bold)) {
      systemfonts::register_font("Helvetica", plain = reg$path[1], bold = bold$path[1])
      break
    }
  }
}

INK <- "#222222"
MUTED <- "#6E6E6E"
GRID <- "#CBCBCB"
TIE <- "#8A8F98"
colours <- c(National = "#1F5FBF", Labour = "#D8342A", Green = "#44CA89", ACT = "#D9A300", `NZ First` = "#1F2328",
             Opportunity = "#079EC4", `Te Pāti Māori` = "#8E2F6B", Other = "#B4BAC0")
gov_col <- colours[["National"]]
opp_col <- colours[["Labour"]]
due <- format(as.Date(fc$official_results_due), "%e %B %Y")
due <- trimws(due)
source_line <- paste0("Source: NZ Poll of Polls special-vote model, ", format(fc$n_sims, big.mark = ","),
                      " simulated official counts; Electoral Commission preliminary results")

pct <- function(x, digits = 0) paste0(formatC(100 * x, format = "f", digits = digits), "%")
num <- function(x) format(round(x), big.mark = ",", trim = TRUE)

# bbc_style() plus the two things its finalise_plot() adds as an image: a left-aligned source line and titles
# aligned with the plot's edge.
style <- function() {
  bbc_style() +
    theme(plot.title.position = "plot", plot.caption.position = "plot",
          plot.title = element_text(size = 26, margin = margin(b = 6)),
          plot.subtitle = element_text(size = 17, colour = INK, lineheight = 1.15, margin = margin(b = 14)),
          plot.caption = element_text(family = "Helvetica", size = 11.5, colour = MUTED, hjust = 0, lineheight = 1.2,
                                      margin = margin(t = 16)),
          axis.text = element_text(size = 14, colour = INK),
          legend.text = element_text(size = 14, colour = INK),
          plot.margin = margin(16, 20, 12, 16))
}

save <- function(plot, name, width = 760, height = 520) {
  plot$labels$caption <- paste(strwrap(plot$labels$caption, width = round(width / 5.9)), collapse = "\n")
  ggsave(file.path(out, paste0(name, ".svg")), plot, width = width / 72, height = height / 72,
         device = svglite::svglite, bg = "white")
  ggsave(file.path(out, paste0(name, ".png")), plot, width = width / 72, height = height / 72, dpi = 216,
         device = ragg::agg_png, bg = "white")
  message("wrote ", name)
}

gov_label <- house$government$label
opp_label <- house$opposition$label
two_sided <- isTRUE(house$two_sided)
neither_label <- if (two_sided) "Exact tie" else "No majority"
o <- house$outcome

# ------------------------------------------------------------------------------------------ 1. the House
mo <- house$margin_outcome
d1 <- do.call(rbind, lapply(names(mo), function(m) data.frame(
  margin = as.integer(m), outcome = c("government", "neither", "opposition"),
  p = c(mo[[m]]$government, mo[[m]]$neither, mo[[m]]$opposition))))
d1 <- d1[d1$p > 0, ]
keep <- aggregate(p ~ margin, d1, sum)
keep <- keep$margin[keep$p >= 0.002]
d1 <- d1[d1$margin %in% keep, ]
d1$outcome <- factor(d1$outcome, c("government", "neither", "opposition"),
                     c(paste0("Government majority ", pct(o$government)), paste0(neither_label, " ", pct(o$neither)),
                       paste0("Majority for the other parties ", pct(o$opposition))))
tot <- aggregate(p ~ margin, d1, sum)
night <- house$margin_election_night
p1 <- ggplot(d1, aes(margin, p, fill = outcome)) +
  geom_col(width = 0.86) +
  geom_text(data = tot[tot$p >= 0.01, ], aes(margin, p, label = pct(p)), inherit.aes = FALSE, vjust = -0.5,
            size = 4.6, colour = INK, family = "Helvetica") +
  geom_hline(yintercept = 0, linewidth = 1, colour = "#333333") +
  annotate("segment", x = night, xend = night, y = max(tot$p) * 1.16, yend = tot$p[tot$margin == night] + 0.022,
           colour = MUTED, linewidth = 0.5) +
  annotate("text", x = night, y = max(tot$p) * 1.2, label = "Election night", colour = MUTED, size = 4.6,
           family = "Helvetica") +
  scale_fill_manual(values = setNames(c(gov_col, TIE, opp_col), levels(d1$outcome)), drop = FALSE) +
  scale_x_continuous(breaks = seq(min(d1$margin), max(d1$margin)),
                     labels = function(x) ifelse(x > 0, paste0("+", x), ifelse(x < 0, paste0("−", abs(x)), "0"))) +
  scale_y_continuous(labels = function(x) pct(x), expand = expansion(mult = c(0, 0.08))) +
  labs(title = "The Government's majority is a coin toss",
       subtitle = paste0("Chance of each final margin between ", gov_label, "\nand every other party, in seats, once special votes are counted"),
       caption = paste0(source_line, ". Official results are due ", due, ".")) +
  style()
save(p1, "01-house-margin", height = 560)

# ---------------------------------------------------------------------------------- 2. party vote change
pv <- do.call(rbind, lapply(house$party_vote, function(r) data.frame(
  party = r$party, night = 100 * r$election_night_share, lo = r$change_pp$q05, mid = r$change_pp$q50,
  hi = r$change_pp$q95, lo50 = r$change_pp$q25, hi50 = r$change_pp$q75)))
pv <- pv[pv$party != "Other", ]
pv$party <- factor(pv$party, rev(pv$party[order(-pv$mid)]))
sgn <- function(x) paste0(ifelse(x >= 0, "+", "−"), formatC(abs(x), format = "f", digits = 2))
p2 <- ggplot(pv, aes(y = party)) +
  geom_vline(xintercept = 0, linewidth = 1, colour = "#333333") +
  geom_segment(aes(x = lo, xend = hi, yend = party, colour = party), linewidth = 2.2, alpha = 0.35, lineend = "round") +
  geom_segment(aes(x = lo50, xend = hi50, yend = party, colour = party), linewidth = 5.5, lineend = "round") +
  geom_point(aes(x = mid), size = 4.2, shape = 21, fill = "white", colour = INK, stroke = 1.3) +
  geom_text(aes(x = ifelse(mid >= 0, hi, lo), label = sgn(mid), hjust = ifelse(mid >= 0, -0.3, 1.3)), size = 5,
            colour = INK, fontface = "bold", family = "Helvetica") +
  scale_colour_manual(values = colours, guide = "none") +
  scale_x_continuous(labels = function(x) ifelse(x > 0, paste0("+", x), ifelse(x < 0, paste0("−", abs(x)), "0")),
                     expand = expansion(mult = 0.16)) +
  labs(title = "Special votes should cost National most",
       subtitle = "Forecast change in each party's share of the party vote from election night\nto the official count, in percentage points. Thick bar: 50% of simulations; thin: 90%",
       caption = source_line) +
  style() +
  theme(panel.grid.major.y = element_blank(), panel.grid.major.x = element_line(colour = GRID),
        axis.text.y = element_text(face = "bold", size = 15))
save(p2, "02-party-vote-change", height = 520)

# ------------------------------------------------------------------------------------------ 3. seats
st <- do.call(rbind, lapply(house$seats, function(s) data.frame(
  party = s$party, night = s$election_night, fewer = s$p_fewer, more = s$p_more,
  likely = as.integer(names(which.max(unlist(s$seats)))))))
st <- st[st$party != "Other", ]
st$label <- paste0(st$party, "  ", st$night, ifelse(st$likely != st$night, paste0(" → ", st$likely), ""))
st$label <- factor(st$label, rev(st$label[order(st$more - st$fewer, decreasing = TRUE)]))
d3 <- rbind(data.frame(label = st$label, party = st$party, p = -st$fewer, side = "fewer"),
            data.frame(label = st$label, party = st$party, p = st$more, side = "more"))
p3 <- ggplot(d3, aes(p, label, fill = party)) +
  geom_col(width = 0.7) +
  geom_vline(xintercept = 0, linewidth = 1, colour = "#333333") +
  geom_text(data = d3[abs(d3$p) >= 0.02, ], aes(label = pct(abs(p)), hjust = ifelse(p < 0, 1.15, -0.15)), size = 4.8,
            colour = INK, family = "Helvetica") +
  annotate("text", x = -0.5, y = nrow(st) + 0.75, label = "← Chance of fewer seats", colour = MUTED, size = 4.8,
           family = "Helvetica") +
  annotate("text", x = 0.5, y = nrow(st) + 0.75, label = "Chance of more seats →", colour = MUTED, size = 4.8,
           family = "Helvetica") +
  scale_fill_manual(values = colours, guide = "none") +
  scale_x_continuous(limits = c(-1.12, 1.12), breaks = c(-1, -0.5, 0, 0.5, 1), labels = function(x) pct(abs(x))) +
  scale_y_discrete(expand = expansion(add = c(0.6, 1.2))) +
  labs(title = "NZ First's twelfth seat is the one most at risk",
       subtitle = "Chance each party ends with fewer or more seats than on election night.\nLabels show election-night seats and, where it differs, the most likely final count",
       caption = source_line) +
  style() +
  theme(panel.grid.major.y = element_blank(), panel.grid.major.x = element_line(colour = GRID),
        axis.text.y = element_text(face = "bold", size = 15, hjust = 0))
save(p3, "03-seats-at-risk", height = 540)

# ------------------------------------------------------------------------------------ 4. the close seats
pair <- unlist(fc$pair)
cs <- do.call(rbind, lapply(fc$seats, function(s) data.frame(
  seat = s$name, night = s$preliminary$margin, p_second = s$p_win[[pair[2]]])))
cs <- rbind(cs, do.call(rbind, lapply(house$other_seats, function(s) {
  lab_leads <- s$election_night_margin > 0
  data.frame(seat = s$name, night = s$election_night_margin, p_second = if (lab_leads) 1 - s$p_flip else s$p_flip)
})))
cs$lead <- ifelse(cs$night > 0, pair[2], pair[1])
cs$label <- paste0(cs$seat, "  (", cs$lead, " +", num(abs(cs$night)), ")")
cs$label <- factor(cs$label, cs$label[order(cs$p_second)])
# the second party's chance runs from the left edge, the first party's fills the rest
d4 <- rbind(data.frame(label = cs$label, party = pair[1], p = 1),
            data.frame(label = cs$label, party = pair[2], p = cs$p_second))
p4 <- ggplot(mapping = aes(p, label, fill = party)) +
  geom_col(data = d4[d4$party == pair[1], ], width = 0.72) +
  geom_col(data = d4[d4$party == pair[2], ], width = 0.72) +
  geom_vline(xintercept = 0.5, linewidth = 0.6, colour = "white", linetype = "22") +
  geom_text(data = cs, aes(x = 0.015, y = label, label = ifelse(p_second >= 0.12, pct(p_second), "")),
            inherit.aes = FALSE, hjust = 0, colour = "white", fontface = "bold", size = 4.8, family = "Helvetica") +
  geom_text(data = cs, aes(x = 0.985, y = label, label = ifelse(1 - p_second >= 0.12, pct(1 - p_second), "")),
            inherit.aes = FALSE, hjust = 1, colour = "white", fontface = "bold", size = 4.8, family = "Helvetica") +
  scale_fill_manual(values = colours[pair], breaks = rev(pair), labels = paste(rev(pair), "wins")) +
  scale_x_continuous(labels = function(x) pct(x), expand = c(0, 0)) +
  labs(title = "Two National seats hang on the special votes",
       subtitle = "Chance of winning each electorate within 2,500 votes in the official count.\nIn brackets: who led on election night, and by how many votes",
       caption = source_line) +
  style() +
  theme(panel.grid.major.y = element_blank(), axis.text.y = element_text(size = 14, hjust = 0),
        legend.justification = "left")
save(p4, "04-close-seats", height = 620)

# ------------------------------------------------------------------------------------- 5. the 14 routes
if (length(house$routes$minimal)) {
  rt <- do.call(rbind, lapply(house$routes$minimal, function(r) data.frame(
    route = paste(unlist(r$steps), collapse = " + "), p = r$p)))
  short <- function(x) {
    x <- gsub("'s party-vote entitlement falls by ([0-9]+)", " entitlement −\\1", x)
    x <- gsub(" loses ([0-9]+) electorates? net", " electorates −\\1", x)
    gsub(" loses ([0-9]+) list seats?", " list seats −\\1", x)
  }
  rt$route <- short(rt$route)
  rt$route <- factor(rt$route, rev(rt$route))
  rt$lab <- ifelse(rt$p >= 0.0005, pct(rt$p, 1), "under 0.1%")
  need <- house$routes$seats_of_margin_needed
  p5 <- ggplot(rt, aes(p, route)) +
    geom_col(width = 0.68, fill = opp_col) +
    geom_text(aes(label = lab), hjust = -0.15, size = 4.6, colour = INK, family = "Helvetica") +
    geom_vline(xintercept = 0, linewidth = 1, colour = "#333333") +
    scale_x_continuous(breaks = seq(0, 1, 0.02), labels = function(x) pct(x), expand = expansion(mult = c(0, 0.22))) +
    labs(title = paste0("The ", nrow(rt), " routes to a one-seat change of majority"),
         subtitle = paste0("Chance of each exact combination that moves the margin by ", need, " seats. Together ",
                           pct(house$routes$p_minimal), ";\nany one-seat majority ", pct(house$routes$p_margin_exactly_needed),
                           "; a bigger one ", pct(house$routes$p_margin_more_than_needed), "; in all ", pct(o$opposition)),
         caption = paste0(source_line, ". A governing party's list seat counts 2; National's entitlement and electorates count 1 each.")) +
    style() +
    theme(panel.grid.major.y = element_blank(), panel.grid.major.x = element_line(colour = GRID),
          axis.text.y = element_text(size = 13, hjust = 0),
          plot.caption = element_text(family = "Helvetica", size = 11, colour = MUTED, hjust = 0, lineheight = 1.2,
                                      margin = margin(t = 16)))
  save(p5, "05-routes", width = 820, height = 680)
}

# --------------------------------------------------------------------------------------- 6. scenarios
sc <- house$scenarios
years <- unlist(fc$data$years)
keys <- c("model", paste0("like_", years), "no_enrolment_adjustment")
labs6 <- c("This forecast", paste0("Special votes lean as in ", years), "No enrolment adjustment")
d6 <- do.call(rbind, lapply(seq_along(keys), function(i) data.frame(
  scenario = labs6[i], outcome = c("government", "neither", "opposition"),
  p = c(sc[[keys[i]]]$government, sc[[keys[i]]]$neither, sc[[keys[i]]]$opposition))))
d6$scenario <- factor(d6$scenario, rev(labs6))
d6$outcome <- factor(d6$outcome, c("government", "neither", "opposition"),
                     c("Government keeps majority", neither_label, "Other parties have majority"))
d6$pos <- ave(d6$p, d6$scenario, FUN = function(p) cumsum(p) - p / 2)
p6 <- ggplot(d6, aes(p, scenario, fill = outcome)) +
  geom_col(width = 0.7, position = position_stack(reverse = TRUE), colour = "white", linewidth = 0.8) +
  geom_text(data = d6[d6$p >= 0.06, ], aes(x = pos, label = pct(p)), colour = "white", fontface = "bold", size = 4.8,
            family = "Helvetica") +
  scale_fill_manual(values = setNames(c(gov_col, TIE, opp_col), levels(d6$outcome))) +
  scale_x_continuous(labels = function(x) pct(x), expand = c(0, 0)) +
  labs(title = "It depends which election this one resembles",
       subtitle = "Chance of each outcome if special votes differ from ordinary votes\nas they did at a past election, on this year's count and number of special votes",
       caption = source_line) +
  style() +
  theme(panel.grid.major.y = element_blank(), axis.text.y = element_text(size = 14, hjust = 0),
        legend.justification = "left")
save(p6, "06-scenarios", height = 500)

# ------------------------------------------------------------------------------ 7. the four seat margins
md <- read.csv(file.path(src, "margin_draws.csv"), check.names = FALSE, encoding = "UTF-8")
info <- do.call(rbind, lapply(fc$seats, function(s) data.frame(
  electorate = s$name, night = s$preliminary$margin,
  strip = paste0(s$name, ": ", pair[2], " ", pct(s$p_win[[pair[2]]]), ", ", pair[1], " ", pct(s$p_win[[pair[1]]])))))
md <- merge(md, info, by = "electorate")
md$strip <- factor(md$strip, info$strip)
info$strip <- factor(info$strip, info$strip)
lim <- quantile(abs(md$margin), 0.995)
md <- md[abs(md$margin) <= lim, ]
md$side <- ifelse(md$margin > 0, pair[2], pair[1])
p7 <- ggplot(md, aes(margin, fill = side)) +
  geom_histogram(aes(y = after_stat(count / 5000)), binwidth = 100, boundary = 0, colour = "white", linewidth = 0.2) +
  geom_vline(xintercept = 0, linewidth = 0.8, colour = "#333333") +
  geom_vline(data = info, aes(xintercept = night), colour = INK, linewidth = 0.6, linetype = "22") +
  geom_hline(yintercept = 0, linewidth = 1, colour = "#333333") +
  facet_wrap(~strip, ncol = 2) +
  scale_fill_manual(values = colours[pair], breaks = rev(pair), labels = paste(rev(pair), "ahead")) +
  scale_x_continuous(labels = function(x) ifelse(x == 0, "0", paste0(ifelse(x > 0, "+", "−"), num(abs(x))))) +
  scale_y_continuous(labels = function(x) pct(x), expand = expansion(mult = c(0, 0.08))) +
  labs(title = "Where the four closest seats could land",
       subtitle = paste0("Share of simulated official counts by final margin in votes, ", pair[2], " minus ", pair[1],
                         ".\nDashed line: the election-night margin"),
       caption = source_line) +
  style() +
  theme(strip.text = element_text(family = "Helvetica", size = 15, face = "bold", hjust = 0, margin = margin(b = 8, t = 6)),
        panel.spacing = unit(1.4, "lines"), axis.text = element_text(size = 12), legend.justification = "left")
save(p7, "07-seat-margins", width = 820, height = 640)

# ----------------------------------------------------------------------- 8. how special votes have leaned
hist <- read.csv(file.path(src, "party_vote_history.csv"), check.names = FALSE, encoding = "UTF-8")
hist <- hist[hist$party != "Other", ]
hist$diff <- 100 * (hist$special_share - hist$ordinary_share)
order8 <- names(sort(tapply(hist$diff, hist$party, mean)))
hist$party <- factor(hist$party, order8)
last <- hist[hist$year == max(hist$year), ]
last <- last[order(last$diff), ]
last$y <- last$diff                      # keep the end labels at least 0.62 points apart
for (i in seq_len(nrow(last))[-1]) last$y[i] <- max(last$y[i], last$y[i - 1] + 0.62)
last$y <- last$y - (mean(last$y) - mean(last$diff))
p8 <- ggplot(hist, aes(year, diff, colour = party)) +
  geom_hline(yintercept = 0, linewidth = 1, colour = "#333333") +
  geom_line(linewidth = 1.3) +
  geom_point(size = 2.8) +
  geom_text(data = last, aes(x = year + 0.5, y = y, label = party), hjust = 0, size = 4.8, fontface = "bold", family = "Helvetica") +
  scale_colour_manual(values = colours, guide = "none") +
  scale_x_continuous(breaks = sort(unique(hist$year)), expand = expansion(add = c(0.6, 6.2))) +
  scale_y_continuous(labels = function(x) ifelse(x > 0, paste0("+", x), ifelse(x < 0, paste0("−", abs(x)), "0"))) +
  labs(title = "Special voters have always been cooler on National",
       subtitle = "Each party's share of special votes minus its share of ordinary votes,\npercentage points, official counts 2008 to 2023",
       caption = "Source: Electoral Commission official results by voting place, summed over all electorates") +
  style()
save(p8, "08-special-vote-lean", height = 540)

# ------------------------------------------------- 9. seats between other pairs of parties (Wellington Bays)
# One chart per contest in forecast.json's `contests`: the same histogram as chart 7, in the two parties' colours.
for (ct in fc$contests) {
  cp <- unlist(ct$pair)
  draws_csv <- file.path(src, paste0("margin_draws_", ct$slug, ".csv"))
  if (!file.exists(draws_csv)) next
  cd <- read.csv(draws_csv, check.names = FALSE, encoding = "UTF-8")
  draws <- max(table(cd$electorate))
  ci <- do.call(rbind, lapply(ct$seats, function(s) data.frame(
    electorate = s$name, night = s$preliminary$margin, median = s$margin$q50,
    strip = paste0(s$name, ": ", cp[2], " ", pct(s$p_win[[cp[2]]]), ", ", cp[1], " ", pct(s$p_win[[cp[1]]])))))
  cd <- merge(cd, ci, by = "electorate")
  cd$strip <- factor(cd$strip, ci$strip)
  ci$strip <- factor(ci$strip, ci$strip)
  cd <- cd[abs(cd$margin) <= quantile(abs(cd$margin), 0.995), ]
  cd$side <- ifelse(cd$margin > 0, cp[2], cp[1])
  s1 <- ct$seats[[1]]
  fav <- if (s1$p_win[[cp[2]]] >= s1$p_win[[cp[1]]]) cp[2] else cp[1]
  who <- sub("^(.*), (.*)$", "\\2 \\1", s1$contenders[[fav]])
  who <- gsub("\\b([A-Z])([A-Z]+)\\b", "\\1\\L\\2", who, perl = TRUE)
  flips <- fav != s1$leader_election_night
  title <- if (length(ct$seats) == 1) {
    paste0(s1$name, ": ", who, if (flips) " is likely to overtake" else " is likely to hold on")
  } else paste0("Where the ", cp[1], "–", cp[2], " seats could land")
  p9 <- ggplot(cd, aes(margin, fill = side)) +
    geom_histogram(aes(y = after_stat(count / draws)), binwidth = 100, boundary = 0, colour = "white", linewidth = 0.2) +
    geom_vline(xintercept = 0, linewidth = 0.8, colour = "#333333") +
    geom_vline(data = ci, aes(xintercept = night), colour = INK, linewidth = 0.6, linetype = "22") +
    geom_hline(yintercept = 0, linewidth = 1, colour = "#333333") +
    scale_fill_manual(values = colours[cp], breaks = rev(cp), labels = paste(rev(cp), "ahead")) +
    scale_x_continuous(labels = function(x) ifelse(x == 0, "0", paste0(ifelse(x > 0, "+", "−"), num(abs(x))))) +
    scale_y_continuous(labels = function(x) pct(x), expand = expansion(mult = c(0, 0.08))) +
    labs(title = title,
         subtitle = paste0(if (length(ct$seats) == 1) paste0(cp[2], " ", pct(s1$p_win[[cp[2]]]), ", ", cp[1], " ",
                                                             pct(s1$p_win[[cp[1]]]), " to win. ") else "",
                           "Share of simulated official counts by final\nmargin in votes, ", cp[2], " minus ", cp[1],
                           ". Dashed line: the election-night margin"),
         caption = paste0(source_line, " as republished by RNZ")) +
    style() +
    theme(axis.text = element_text(size = 13), legend.justification = "left")
  if (length(ct$seats) > 1) {
    p9 <- p9 + facet_wrap(~strip, ncol = 2) +
      theme(strip.text = element_text(family = "Helvetica", size = 15, face = "bold", hjust = 0,
                                      margin = margin(b = 8, t = 6)), panel.spacing = unit(1.4, "lines"))
  }
  save(p9, paste0("09-", gsub("_", "-", ct$slug), "-seats"), height = 520)
}
