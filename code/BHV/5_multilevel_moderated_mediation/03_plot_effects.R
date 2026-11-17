script_frame <- Filter(function(f) !is.null(f$ofile), sys.frames())
if (length(script_frame)) {
  frame <- script_frame[[length(script_frame)]]
  script_path <- frame$ofile
  if (!is.null(frame$owd) && !grepl("^(/|[A-Za-z]:|\\\\\\\\)", script_path)) {
    script_path <- file.path(frame$owd, script_path)
  }
} else {
  script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L])
}
here <- dirname(normalizePath(script_path, mustWork = TRUE))
suppressWarnings(Sys.setlocale("LC_CTYPE", "English_United States.utf8"))
suppressPackageStartupMessages({library(dplyr); library(readr); library(tidyr); library(ggplot2)})
analysis_dir <- Sys.getenv("MEDIATION_ANALYSIS_DIR", here)
output_dir <- file.path(analysis_dir, "outputs")
figure_dir <- file.path(analysis_dir, "figures")
dir.create(figure_dir, showWarnings = FALSE)
dat <- read_csv(file.path(output_dir, "prepared_mediation_data.csv"), show_col_types = FALSE)
boot_summary <- read_csv(file.path(output_dir, "moderated_mediation_bootstrap_summary.csv"), show_col_types = FALSE)
dat <- rename(dat, reward = magnitude)
boot_summary <- rename(boot_summary, reward = magnitude)
dat$reward <- factor(dat$reward, levels = c("small", "large"))
boot_summary$reward <- factor(boot_summary$reward, levels = c("small", "large"))
mediators <- c("beta_mb", "beta_mf", "w")
mediator_order <- mediators
mediator_labels <- c(beta_mb = "MB weight", beta_mf = "MF weight", w = "Mixing weight")
boot_summary$mediator_label <- factor(boot_summary$mediator_label, levels = unname(mediator_labels))
reward_labels <- c(small = "Small reward", large = "Large reward")
pink <- "#346db1"
blue <- "#b03949"
reward_colors <- c(small = pink, large = blue)

plot_effect <- function(summary_data, estimate_col, low_col, high_col, y_label, file_stub) {
  pd <- position_dodge(width = 0.45)
  p <- ggplot(
    summary_data,
    aes(x = mediator_label, y = .data[[estimate_col]], color = reward, group = reward)
  ) +
    geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.5, color = "grey45") +
    geom_errorbar(
      aes(ymin = .data[[low_col]], ymax = .data[[high_col]]),
      width = 0.15, linewidth = 0.8, position = pd
    ) +
    geom_point(size = 2.7, position = pd) +
    scale_color_manual(values = reward_colors, labels = reward_labels, name = "Reward condition") +
    labs(x = NULL, y = y_label) +
    theme_classic(base_size = 12) +
    theme(
      axis.text.x = element_text(color = "black"),
      axis.text.y = element_text(color = "black"),
      axis.line = element_line(linewidth = 0.8, color = "black"),
      axis.ticks = element_line(linewidth = 0.8, color = "black"),
      legend.position = "top"
    )
  ggsave(file.path(figure_dir, paste0(file_stub, ".pdf")), p, width = 7, height = 4.5, device = grDevices::cairo_pdf, fallback_resolution = 600, dpi = 600)
  invisible(p)
}

plot_effect(boot_summary, "indirect_ab", "indirect_ci_low", "indirect_ci_high", "Indirect effect of WM load (high - low)", "indirect_effects_by_reward")
plot_effect(boot_summary, "a_load_to_mediator", "a_ci_low", "a_ci_high", "a path: WM load (high - low) -> mediator", "a_paths_by_reward")
plot_effect(boot_summary, "b_mediator_to_cr", "b_ci_low", "b_ci_high", "b path: mediator -> corrected reward rate", "b_paths_by_reward")

relationship_data <- dat %>%
  select(subid, load, reward, cr, beta_mb_within, beta_mf_within, w_within) %>%
  tidyr::pivot_longer(
    cols = c(beta_mb_within, beta_mf_within, w_within),
    names_to = "mediator",
    values_to = "mediator_within"
  ) %>%
  mutate(
    mediator_label = dplyr::recode(
      mediator,
      beta_mb_within = "MB weight (within-subject)",
      beta_mf_within = "MF weight (within-subject)",
      w_within = "Mixing weight (within-subject)"
    )
  )

p_relationship <- ggplot(relationship_data, aes(x = mediator_within, y = cr, color = reward)) +
  geom_point(alpha = 0.45, size = 1.8) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 0.9, alpha = 0.18) +
  facet_wrap(~ mediator_label, scales = "free_x") +
  scale_color_manual(values = reward_colors, labels = reward_labels, name = "Reward condition") +
  labs(x = "Within-subject mediator value", y = "Corrected reward rate") +
  theme_classic(base_size = 12) +
  theme(
    axis.line = element_line(linewidth = 0.8, color = "black"),
    axis.ticks = element_line(linewidth = 0.8, color = "black"),
    legend.position = "top"
  )
p_relationship <- p_relationship + theme(panel.spacing = grid::unit(1.5, "lines")) + scale_x_continuous(n.breaks = 4)

ggsave(file.path(figure_dir, "mediator_cr_relationships.pdf"), p_relationship, width = 9, height = 4.5, device = grDevices::cairo_pdf, fallback_resolution = 600, dpi = 600)

