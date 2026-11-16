suppressPackageStartupMessages({library(dplyr); library(ggplot2)})
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
source_paths <- unlist(lapply(sys.frames(), function(frame) frame$ofile))
script_path <- if (length(source_paths)) tail(source_paths, 1) else sub("^--file=", "", script_arg[1])
here <- dirname(normalizePath(script_path, mustWork = TRUE))
input <- file.path(here, "..", "..", "group_data", "MEG", "brain_behavior")
forest_dir <- file.path(here, "results", "Fig_5")
dir.create(forest_dir, recursive = TRUE, showWarnings = FALSE)

safe_name <- function(x) {
  x <- gsub("[^A-Za-z0-9]+", "_", as.character(x))
  x <- gsub("^_+|_+$", "", x)
  tolower(x)
}

phase_label <- function(x) {
  dplyr::case_when(
    x == "incentive" ~ "Incentive",
    x == "decision" ~ "Decision",
    x == "transition" ~ "Transition",
    x == "feedback" ~ "Feedback",
    TRUE ~ as.character(x)
  )
}

analysis_label <- function(x) {
  dplyr::case_when(
    x == "source_cortex_mask" ~ "ERF source",
    x == "tfa_sensor" ~ "TFA",
    TRUE ~ as.character(x)
  )
}

effect_label <- function(x) {
  dplyr::case_when(
    x %in% c("magnitude", "reward") ~ "Reward",
    x == "load" ~ "Load",
    x == "interaction" ~ "Interaction",
    TRUE ~ as.character(x)
  )
}

band_label <- function(analysis, target_label, source_file) {
  txt <- paste(target_label, source_file)
  dplyr::case_when(
    analysis == "source_cortex_mask" ~ "Source",
    grepl("theta", txt, ignore.case = TRUE) ~ "Theta",
    grepl("alpha", txt, ignore.case = TRUE) ~ "Alpha",
    grepl("low[_-]?gamma", txt, ignore.case = TRUE) ~ "Low gamma",
    grepl("gamma", txt, ignore.case = TRUE) ~ "Gamma",
    grepl("beta", txt, ignore.case = TRUE) ~ "Beta",
    TRUE ~ "TFA"
  )
}

phase_rank <- function(x) {
  match(as.character(x), c("incentive", "decision", "transition", "feedback"))
}

analysis_rank <- function(x) {
  match(as.character(x), c("source_cortex_mask", "tfa_sensor"))
}

predictor_rank <- function(x) {
  match(as.character(x), c("effect_matched", "b0_sensitivity"))
}

format_p <- function(p) {
  ifelse(
    is.na(p), "NA",
    ifelse(p < 0.001, "<.001", sprintf("%.3f", p))
  )
}

make_cluster_label <- function(df) {
  paste0(
    phase_label(df$phase), " | ",
    analysis_label(df$analysis),
    ifelse(df$analysis == "tfa_sensor", paste0(" ", df$band), ""), " | ",
    effect_label(df$effect), " | C", df$cluster_number, " | ",
    df$time_start_ms, "-", df$time_end_ms, " ms"
  )
}

make_forest_label <- function(df) {
  method_line <- paste0(
    phase_label(df$phase), " | ",
    analysis_label(df$analysis),
    ifelse(df$analysis == "tfa_sensor", paste0(" ", df$band), "")
  )
  detail_line <- paste0(
    effect_label(df$effect), " | C", df$cluster_number, " | ",
    df$time_start_ms, "-", df$time_end_ms, " ms"
  )
  paste0(method_line, "\n", detail_line)
}

data <- read.csv(file.path(input, "Fig_5a_MB_coefficients.csv"))
summary_all <- data %>% mutate(model_family = "model_based",
    phase_pretty = phase_label(phase),
    analysis_pretty = analysis_label(analysis),
    effect_pretty = effect_label(effect),
    cluster_label = make_cluster_label(pick(everything())),
    model_pretty = dplyr::case_when(
      model_family == "model_based" ~ "Model-based",
      model_family == "model_free" ~ "Model-free",
      TRUE ~ model_family
    ),
    predictor_pretty = dplyr::case_when(
      predictor_family == "effect_matched" ~ "Effect-matched",
      predictor_family == "b0_sensitivity" ~ "b0",
      TRUE ~ predictor_family
    ),
    significance = dplyr::case_when(
      !is.na(p_fdr_lm_within_phase) & p_fdr_lm_within_phase < 0.05 ~ "phasewise FDR < .05",
      !is.na(p_value) & p_value < 0.05 ~ "raw p < .05",
      TRUE ~ "n.s."
    ),
    direction = dplyr::case_when(
      is.na(meg_effect_std) ~ NA_character_,
      meg_effect_std > 0 ~ "positive",
      meg_effect_std < 0 ~ "negative",
      TRUE ~ "zero"
    )
  ) %>%
  arrange(
    model_family,
    predictor_rank(predictor_family),
    phase_rank(phase),
    analysis_rank(analysis),
    effect,
    band,
    time_start_ms,
    cluster_number
  )

plot_forest_one <- function(df, model_family_value) {
  plot_df <- df %>%
    filter(model_family == model_family_value) %>%
    mutate(
      y_label = make_forest_label(pick(everything())),
      y_label = factor(y_label, levels = rev(unique(y_label))),
      predictor_pretty = factor(predictor_pretty, levels = c("Effect-matched", "b0")),
      significance = factor(significance, levels = c("phasewise FDR < .05", "raw p < .05", "n.s."))
    )

  if (nrow(plot_df) == 0) return(invisible(NULL))

  p <- ggplot(
    plot_df,
    aes(
      x = meg_effect_std,
      y = y_label,
      xmin = ci_low,
      xmax = ci_high,
      color = effect_pretty,
      shape = significance
    )
  ) +
    geom_vline(xintercept = 0, color = "grey65", linewidth = 0.4) +
    geom_errorbar(width = 0, linewidth = 0.85, orientation = "y", position = position_dodge(width = 0.55)) +
    geom_point(size = 2.5, stroke = 1.05, position = position_dodge(width = 0.55)) +
    facet_grid(. ~ predictor_pretty) +
    scale_color_manual(
      values = c("Reward" = "#9a3258", "Load" = "#23488c", "Interaction" = "#acab89")
    ) +
    scale_shape_manual(
      values = c("phasewise FDR < .05" = 17, "raw p < .05" = 15, "n.s." = 1)
    ) +
    labs(
      title = paste0(unique(plot_df$model_pretty), " Behavior-From-MEG Associations"),
      subtitle = "Outcome: HBRL parameter; predictor: MEG cluster effect controlling age/sex",
      x = "Standardized regression coefficient for MEG effect",
      y = NULL,
      color = "MEG effect",
      shape = "Significance"
    ) +
    theme_classic(base_size = 10) +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      axis.title.x = element_text(size = 12),
      axis.text.x = element_text(size = 11),
      axis.text.y = element_text(size = 8.5, lineheight = 0.92),
      panel.spacing.x = grid::unit(1.2, "lines"),
      strip.background = element_rect(fill = "grey92", color = NA),
      strip.text = element_text(face = "bold"),
      legend.position = "right"
    )

  stem <- paste0("forest_behavior_from_meg_", safe_name(model_family_value))
  ggsave(file.path(forest_dir, paste0(stem, ".pdf")), p, width = 12.0, height = max(7, 0.23 * nrow(plot_df)), device = grDevices::cairo_pdf, dpi = 600, fallback_resolution = 600)
}

points <- read.csv(file.path(input, "Fig_5bc_residual_points.csv"))
panels <- read.csv(file.path(input, "Fig_5bc_index.csv"))
plot_scatter_for_row <- function(row, out_dir, suffix) {
  row <- as.list(row)
  plot_df <- points %>% filter(scatter_key == row$scatter_key) %>%
    transmute(meg_resid = x_meg_residual, behavior_resid = y_behavior_residual)
  partial_r <- suppressWarnings(cor(plot_df$meg_resid, plot_df$behavior_resid, use = "complete.obs"))

  title <- paste0(
    row$model_pretty, " | ", row$phase_pretty, " | ", row$analysis_pretty,
    ifelse(row$analysis == "tfa_sensor", paste0(" ", row$band), ""),
    " | ", row$effect_pretty, " | C", row$cluster_number,
    " | ", row$time_start_ms, "-", row$time_end_ms, " ms"
  )

  subtitle <- paste0(
    sub("_magnitude", "_reward", row$behavior_outcome),
    "; beta = ", sprintf("%.3f", row$meg_effect_std),
    ", p = ", format_p(row$p_value),
    ", phase-FDR p = ", format_p(row$p_fdr_lm_within_phase),
    ", partial r = ", sprintf("%.3f", partial_r)
  )

  p <- ggplot(plot_df, aes(x = meg_resid, y = behavior_resid)) +
    geom_hline(yintercept = 0, color = "grey86", linewidth = 0.35) +
    geom_vline(xintercept = 0, color = "grey86", linewidth = 0.35) +
    geom_point(
      shape = 21,
      size = 3.5,
      stroke = 1.5,
      alpha = 1,
      color = "#646464",
      fill = "#BDBDBD"
    ) +
    geom_smooth(
      method = "lm",
      se = TRUE,
      color = "#333333",
      fill = "#888888",
      alpha = 0.18,
      linewidth = 2
    ) +
    labs(
      title = title,
      subtitle = subtitle,
      x = "MEG cluster effect residual controlling age/sex",
      y = "HBRL behavior parameter residual controlling age/sex"
    ) +
    theme_classic(base_size = 10) +
    theme(
      plot.title = element_text(face = "bold", size = 18),
      plot.subtitle = element_text(size = 12),
      axis.title.x = element_text(size = 15),
      axis.title.y = element_text(size = 15),
      axis.text.x = element_text(size = 15),
      axis.text.y = element_text(size = 15),
      axis.line = element_line(linewidth = 1.5),
      axis.ticks = element_line(linewidth = 1),
      axis.ticks.length = unit(2.5, "mm")
    )

  stem <- paste0(
    safe_name(row$model_family), "_",
    safe_name(row$predictor_family), "_",
    safe_name(row$phase), "_",
    safe_name(row$analysis), "_",
    safe_name(row$band), "_",
    safe_name(row$effect), "_c", row$cluster_number, "_",
    row$time_start_ms, "_", row$time_end_ms, "_", suffix
  )

  ggsave(file.path(out_dir, paste0(stem, ".pdf")), p, width = 7, height = 5.2, device = grDevices::cairo_pdf, dpi = 600, fallback_resolution = 600)
  invisible(TRUE)
}

plot_forest_one(summary_all, "model_based")
for (i in seq_len(nrow(panels))) {
  panel <- panels[i, ]
  row <- filter(summary_all, map_key == panel$map_key, behavior_outcome == panel$outcome)
  stopifnot(nrow(row) == 1)
  row$scatter_key <- panel$scatter_key
  plot_scatter_for_row(row, forest_dir, "phase_fdr")
}
