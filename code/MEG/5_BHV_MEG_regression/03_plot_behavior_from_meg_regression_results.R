# Visualize behavior-from-MEG regression results.
#
# Direction:
#   HBRL behavior parameter ~ MEG cluster effect + age_z + sex
#
# Outputs:
#   1) combined regression/correlation summary tables
#   2) forest plots of standardized MEG coefficients
#   3) partial-residual scatter plots for raw-p and phasewise-FDR results
#   4) source/TFA cluster-mask location plots for significant rows
#
# In RStudio use Source, or run from the submission root:
#   rm(list = ls())
#   source("code/MEG/5_BHV_MEG_regression/03_plot_behavior_from_meg_regression_results.R", chdir = TRUE)

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
})

`%||%` <- function(x, y) {
  if (length(x) == 0 || is.na(x) || identical(x, "")) y else x
}

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
analysis_dir <- dirname(normalizePath(script_path, mustWork = TRUE))

output_dir <- Sys.getenv("BHV_FROM_MEG_PLOT_OUT") %||%
  file.path(analysis_dir, "results", "behavior_from_meg_visualizations")

table_dir <- file.path(output_dir, "summary_tables")
forest_dir <- file.path(output_dir, "forest_plots")
raw_scatter_dir <- file.path(output_dir, "raw_p_significant_scatter")
phase_scatter_dir <- file.path(output_dir, "phasewise_fdr_significant_scatter")

generate_heatmaps <- Sys.getenv("BHV_FROM_MEG_GENERATE_HEATMAPS", "FALSE") %in% c("TRUE", "true", "1")
heatmap_dir <- file.path(output_dir, "heatmaps")



message("Behavior-from-MEG visualization")
message("Output directory: ", output_dir)

result_sets <- tibble::tibble(
  model_family = c("model_based", "model_free"),
  parameter_prefix = c("beta_mb", "beta_mf"),
  result_dir = c(
    file.path(analysis_dir, "results", "batch_behavior_from_meg_planned_effects_covariates_phasewise"),
    file.path(analysis_dir, "results", "batch_behavior_from_meg_mf_planned_effects_covariates_phasewise")
  ),
  file_prefix = c(
    "behavior_from_meg_planned_effects_covariates_phasewise",
    "behavior_from_meg_mf_planned_effects_covariates_phasewise"
  )
) %>%
  filter(dir.exists(result_dir))

if (nrow(result_sets) == 0) {
  stop("No behavior-from-MEG result directories found under: ", analysis_dir)
}

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

standardize_if_possible <- function(x) {
  sx <- stats::sd(x, na.rm = TRUE)
  if (is.na(sx) || sx == 0) return(rep(NA_real_, length(x)))
  as.numeric(scale(x))
}

read_one_result_set <- function(model_family, parameter_prefix, result_dir, file_prefix) {
  paths <- list(
    matched_summary = file.path(result_dir, paste0(file_prefix, "_summary.csv")),
    b0_summary = file.path(result_dir, paste0(file_prefix, "_b0_sensitivity_summary.csv")),
    matched_cor = file.path(result_dir, paste0(file_prefix, "_correlation_results.csv")),
    b0_cor = file.path(result_dir, paste0(file_prefix, "_b0_sensitivity_correlation_results.csv")),
    matched_data = file.path(result_dir, paste0(file_prefix, "_regression_data.csv")),
    b0_data = file.path(result_dir, paste0(file_prefix, "_b0_sensitivity_regression_data.csv"))
  )

  missing <- paths[!file.exists(unlist(paths))]
  if (length(missing) > 0) {
    stop("Missing required file(s) for ", model_family, ":\n", paste(unlist(missing), collapse = "\n"))
  }

  matched_summary <- read_csv(paths$matched_summary, show_col_types = FALSE) %>%
    mutate(model_family = model_family, predictor_family = "effect_matched")
  b0_summary <- read_csv(paths$b0_summary, show_col_types = FALSE) %>%
    mutate(model_family = model_family, predictor_family = "b0_sensitivity")

  matched_cor <- read_csv(paths$matched_cor, show_col_types = FALSE) %>%
    mutate(model_family = model_family, predictor_family = "effect_matched")
  b0_cor <- read_csv(paths$b0_cor, show_col_types = FALSE) %>%
    mutate(model_family = model_family, predictor_family = "b0_sensitivity")

  matched_data <- read_csv(paths$matched_data, show_col_types = FALSE) %>%
    mutate(model_family = model_family, predictor_family = "effect_matched")
  b0_data <- read_csv(paths$b0_data, show_col_types = FALSE) %>%
    mutate(model_family = model_family, predictor_family = "b0_sensitivity")

  list(
    summary = bind_rows(matched_summary, b0_summary),
    correlation = bind_rows(matched_cor, b0_cor),
    regression_data = bind_rows(matched_data, b0_data)
  )
}

loaded <- lapply(seq_len(nrow(result_sets)), function(i) {
  args <- as.list(result_sets[i, ])
  do.call(read_one_result_set, args)
})

summary_all <- bind_rows(lapply(loaded, `[[`, "summary")) %>%
  mutate(
    band = band_label(analysis, target_label, source_file),
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

cor_all <- bind_rows(lapply(loaded, `[[`, "correlation")) %>%
  mutate(
    band = band_label(analysis, target_label, source_file),
    phase_pretty = phase_label(phase),
    analysis_pretty = analysis_label(analysis),
    effect_pretty = effect_label(effect),
    model_pretty = dplyr::case_when(
      model_family == "model_based" ~ "Model-based",
      model_family == "model_free" ~ "Model-free",
      TRUE ~ model_family
    ),
    predictor_pretty = dplyr::case_when(
      predictor_family == "effect_matched" ~ "Effect-matched",
      predictor_family == "b0_sensitivity" ~ "b0",
      TRUE ~ predictor_family
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
    cluster_number,
    method
  )

reg_data_all <- bind_rows(lapply(loaded, `[[`, "regression_data")) %>%
  mutate(
    band = band_label(analysis, target_label, source_file),
    behavior_predictor_z = ifelse(is.na(behavior_predictor_z), standardize_if_possible(behavior_predictor), behavior_predictor_z),
    meg_effect_z = ifelse(is.na(meg_effect_z), standardize_if_possible(meg_effect), meg_effect_z)
  )

compact_summary <- summary_all %>%
  transmute(
    model_family,
    predictor_family,
    phase = phase_pretty,
    analysis = analysis_pretty,
    band,
    effect = effect_pretty,
    cluster = paste0("C", cluster_number),
    time_window_ms = paste0(time_start_ms, "-", time_end_ms),
    behavior_outcome,
    n,
    beta_meg_std = meg_effect_std,
    beta_meg_std_se = meg_effect_std_se,
    t_value,
    p_lm = p_value,
    p_lm_phase_fdr = p_fdr_lm_within_phase,
    p_lm_all_fdr = p_fdr_lm_all_tests,
    ci_low,
    ci_high,
    r_squared,
    adj_r_squared,
    pearson_r,
    pearson_p,
    pearson_p_phase_fdr = p_fdr_pearson_within_phase,
    pearson_p_all_fdr = p_fdr_pearson_all_tests,
    spearman_rho,
    spearman_p,
    significance,
    direction,
    source_file
  )

for (d in c(output_dir, table_dir, forest_dir, raw_scatter_dir, phase_scatter_dir)) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}
if (generate_heatmaps) {
  dir.create(heatmap_dir, recursive = TRUE, showWarnings = FALSE)
}

write_csv(summary_all, file.path(table_dir, "behavior_from_meg_full_summary.csv"))
write_csv(compact_summary, file.path(table_dir, "behavior_from_meg_compact_summary.csv"))
write_csv(cor_all, file.path(table_dir, "behavior_from_meg_correlation_results_long.csv"))
write_csv(reg_data_all, file.path(table_dir, "behavior_from_meg_regression_data_long.csv"))

if (requireNamespace("openxlsx", quietly = TRUE)) {
  wb <- openxlsx::createWorkbook()
  openxlsx::addWorksheet(wb, "compact_summary")
  openxlsx::writeData(wb, "compact_summary", compact_summary)
  openxlsx::addWorksheet(wb, "full_summary")
  openxlsx::writeData(wb, "full_summary", summary_all)
  openxlsx::addWorksheet(wb, "correlations")
  openxlsx::writeData(wb, "correlations", cor_all)
  openxlsx::saveWorkbook(
    wb,
    file.path(table_dir, "behavior_from_meg_summary_tables.xlsx"),
    overwrite = TRUE
  )
}

message("Summary tables written to: ", table_dir)

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

plot_heatmap_one <- function(df, model_family_value, predictor_family_value = "effect_matched") {
  plot_df <- df %>%
    filter(model_family == model_family_value, predictor_family == predictor_family_value) %>%
    mutate(
      x_label = paste0(phase_pretty, " | ", analysis_pretty, ifelse(analysis == "tfa_sensor", paste0(" ", band), "")),
      y_label = paste0(effect_pretty, " | C", cluster_number, " | ", time_start_ms, "-", time_end_ms, " ms"),
      x_label = factor(x_label, levels = unique(x_label)),
      y_label = factor(y_label, levels = rev(unique(y_label))),
      sig_text = case_when(
        !is.na(p_fdr_lm_within_phase) & p_fdr_lm_within_phase < 0.05 ~ "**",
        !is.na(p_value) & p_value < 0.05 ~ "*",
        TRUE ~ ""
      )
    )

  if (nrow(plot_df) == 0) return(invisible(NULL))

  max_abs <- max(abs(plot_df$meg_effect_std), na.rm = TRUE)
  if (!is.finite(max_abs) || max_abs == 0) max_abs <- 1

  p <- ggplot(plot_df, aes(x = x_label, y = y_label, fill = meg_effect_std)) +
    geom_tile(color = "white", linewidth = 0.4) +
    geom_text(aes(label = sig_text), size = 4) +
    scale_fill_gradient2(
      low = "#2F5D7C",
      mid = "white",
      high = "#B45F3C",
      midpoint = 0,
      limits = c(-max_abs, max_abs),
      name = "Std. beta"
    ) +
    labs(
      title = paste0(unique(plot_df$model_pretty), " Behavior-From-MEG Heatmap"),
      subtitle = ifelse(predictor_family_value == "effect_matched", "Effect-matched HBRL outcomes", "b0 sensitivity outcomes"),
      x = NULL,
      y = NULL
    ) +
    theme_minimal(base_size = 9) +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
      panel.grid = element_blank()
    )

  stem <- paste0(
    "heatmap_behavior_from_meg_",
    safe_name(model_family_value),
    "_",
    safe_name(predictor_family_value)
  )
  ggsave(file.path(heatmap_dir, paste0(stem, ".pdf")), p, width = 10, height = max(5, 0.28 * nrow(plot_df)), device = grDevices::cairo_pdf, dpi = 600, fallback_resolution = 600)
}

make_partial_data <- function(df) {
  needed <- c("behavior_predictor_z", "meg_effect_z", "age_z", "sex")
  if (!all(needed %in% names(df))) return(NULL)
  df <- df %>% filter(if_all(all_of(needed), ~ !is.na(.x)))
  if (nrow(df) < 5) return(NULL)
  y_model <- lm(behavior_predictor_z ~ age_z + sex, data = df)
  x_model <- lm(meg_effect_z ~ age_z + sex, data = df)
  df %>%
    mutate(
      behavior_resid = residuals(y_model),
      meg_resid = residuals(x_model)
    )
}

plot_scatter_for_row <- function(row, out_dir, suffix) {
  row <- as.list(row)
  df <- reg_data_all %>%
    filter(
      model_family == row$model_family,
      predictor_family == row$predictor_family,
      source_file == row$source_file,
      analysis == row$analysis,
      phase == row$phase,
      effect == row$effect,
      cluster_number == row$cluster_number,
      behavior_outcome_name == row$behavior_outcome
    )

  plot_df <- make_partial_data(df)
  if (is.null(plot_df) || nrow(plot_df) < 5) return(invisible(FALSE))

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

for (mf in unique(summary_all$model_family)) {
  plot_forest_one(summary_all, mf)
  if (generate_heatmaps) {
    plot_heatmap_one(summary_all, mf, "effect_matched")
    plot_heatmap_one(summary_all, mf, "b0_sensitivity")
  }
}

raw_rows <- summary_all %>%
  filter(!is.na(p_value), p_value < 0.05) %>%
  arrange(p_value)

phase_rows <- summary_all %>%
  filter(!is.na(p_fdr_lm_within_phase), p_fdr_lm_within_phase < 0.05) %>%
  arrange(p_fdr_lm_within_phase, p_value)

write_csv(raw_rows, file.path(table_dir, "behavior_from_meg_raw_p_significant_rows.csv"))
write_csv(phase_rows, file.path(table_dir, "behavior_from_meg_phasewise_fdr_significant_rows.csv"))

if (nrow(raw_rows) > 0) {
  for (i in seq_len(nrow(raw_rows))) {
    plot_scatter_for_row(raw_rows[i, ], raw_scatter_dir, "raw_p")
  }
}

if (nrow(phase_rows) > 0) {
  for (i in seq_len(nrow(phase_rows))) {
    plot_scatter_for_row(phase_rows[i, ], phase_scatter_dir, "phase_fdr")
  }
}

message("Forest plots: ", forest_dir)
if (generate_heatmaps) {
  message("Heatmaps: ", heatmap_dir)
} else {
  message("Heatmaps skipped because BHV_FROM_MEG_GENERATE_HEATMAPS is FALSE.")
}
message("Raw-p scatter plots: ", raw_scatter_dir)
message("Phasewise-FDR scatter plots: ", phase_scatter_dir)

run_cluster_masks <- Sys.getenv("BHV_FROM_MEG_RUN_CLUSTER_MASKS", "TRUE") %in% c("TRUE", "true", "1")
mask_script <- file.path(analysis_dir, "04_plot_behavior_from_meg_cluster_masks.py")
mne_python <- Sys.getenv("MNE_PYTHON")
if (run_cluster_masks && !nzchar(mne_python)) {
  candidates <- unname(Sys.which(c("python", "python3")))
  if (.Platform$OS.type == "windows") {
    homes <- file.path(Sys.getenv("USERPROFILE"), c("anaconda3", "miniconda3", "miniforge3"))
    candidates <- c(candidates, file.path(homes, "envs", "mne", "python.exe"))
  }
  for (candidate in unique(candidates[file.exists(candidates)])) {
    status <- tryCatch(system2(candidate, c("-c", shQuote("import mne, pandas, matplotlib, pyvista")),
                               stdout = FALSE, stderr = FALSE), error = function(e) 1L)
    if (isTRUE(status == 0L)) {
      mne_python <- candidate
      break
    }
  }
}
if (run_cluster_masks && file.exists(mask_script) && file.exists(mne_python)) {
  message("Running cluster-mask plotting script: ", mask_script)
  mask_output <- system2(mne_python, args = shQuote(mask_script), stdout = TRUE, stderr = TRUE)
  cat(paste(mask_output, collapse = "\n"), "\n")
} else if (run_cluster_masks) {
  message(
    "Cluster masks were not generated. Check the script path and set MNE_PYTHON to the MNE environment executable.\n",
    "  Script: ", mask_script, "\n",
    "  MNE Python: ", mne_python
  )
}
message("Done.")
