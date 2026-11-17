# Planned-effect brain-behavior regression with behavior as the outcome.
#
# This script is the direction-reversed companion to the previous regression:
#   previous: MEG effect ~ matched HBRL beta_mb predictor + age_z + sex
#   current:  matched HBRL beta_mb predictor ~ MEG effect + age_z + sex
#
# Included targets:
#   incentive: magnitude
#   feedback: magnitude
#   decision: load
#   transition: load
#   any phase: interaction, if a phase-wise significant MEG cluster was extracted
#
# Behavior outcome mapping:
#   MEG reward effect      -> beta_mb_magnitude
#   MEG load effect        -> beta_mb_load
#   MEG interaction effect -> beta_mb_magnitude_load
#
# Main model:
#   matched HBRL beta_mb predictor ~ MEG cluster effect + age_z + sex
#
# b0 sensitivity model:
#   beta_mb_b0 ~ MEG cluster effect + age_z + sex
#
# subid is used only for BHV-MEG-covariate matching.
#
# In RStudio use Source, or run from the submission root:
#   rm(list = ls())
#   source("code/MEG/5_BHV_MEG_regression/02_batch_regress_behavior_from_meg_planned_effects_covariates_phasewise_mb.R", chdir = TRUE)

suppressPackageStartupMessages({
  library(rstan)
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

hbrl_fit_path <- file.path(dirname(dirname(analysis_dir)), "BHV", "3_HBRLmodel", "data_mfmblearning_magnitude&load.RData")

meg_effect_dir <- file.path(analysis_dir, "results", "meg_cluster_subject_effects_core_phases_phasewise")

covariate_file <- file.path(analysis_dir, "subject_covariates_48subs.csv")

meg_effect_col <- Sys.getenv("MEG_EFFECT_COL") %||% "meg_effect_raw"

output_dir <- Sys.getenv("BHV_FROM_MEG_BATCH_OUT") %||%
  file.path(analysis_dir, "results", "batch_behavior_from_meg_planned_effects_covariates_phasewise")
figure_dir <- file.path(output_dir, "Figures")
data_dir <- file.path(output_dir, "regression_data")

result_prefix <- Sys.getenv("RESULT_PREFIX") %||%
  "behavior_from_meg_planned_effects_covariates_phasewise"


message("Output directory: ", output_dir)
message("HBRL fit: ", hbrl_fit_path)
message("MEG effect directory: ", meg_effect_dir)
message("MEG effect column: ", meg_effect_col)
message("Covariate file: ", covariate_file)
message("Model: matched HBRL beta_mb predictor ~ MEG effect + age_z + sex")
message("b0 sensitivity: beta_mb_b0 ~ MEG effect + age_z + sex")

parse_subid <- function(x) {
  suppressWarnings(as.integer(gsub("[^0-9]", "", as.character(x))))
}

standardize_if_possible <- function(x) {
  if (all(is.na(x)) || isTRUE(sd(x, na.rm = TRUE) == 0)) return(x)
  as.numeric(scale(x))
}

safe_name <- function(x) {
  x <- gsub("[^A-Za-z0-9]+", "_", as.character(x))
  x <- gsub("^_+|_+$", "", x)
  x <- tolower(x)
  if (nchar(x) > 70) substr(x, 1, 70) else x
}

first_or_na <- function(x) {
  if (length(x) == 0) return(NA)
  x[[1]]
}

get_optional_col <- function(df, candidates) {
  hit <- candidates[candidates %in% names(df)]
  if (length(hit) == 0) return(NA_character_)
  hit[1]
}

format_p <- function(p) {
  if (is.na(p)) return("p = NA")
  if (p < 0.001) return("p < .001")
  paste0("p = ", sprintf("%.3f", p))
}

tidy_lm_base <- function(model, model_name) {
  model_summary <- summary(model)
  n_obs <- stats::nobs(model)
  r_squared <- model_summary$r.squared
  adj_r_squared <- model_summary$adj.r.squared

  coefs <- as.data.frame(model_summary$coefficients)
  coefs$term <- rownames(coefs)
  rownames(coefs) <- NULL
  names(coefs)[1:4] <- c("estimate", "std_error", "t_value", "p_value")

  ci <- as.data.frame(confint(model))
  ci$term <- rownames(ci)
  rownames(ci) <- NULL
  names(ci)[1:2] <- c("ci_low", "ci_high")

  left_join(coefs, ci, by = "term") %>%
    mutate(
      model = model_name,
      n = n_obs,
      r_squared = r_squared,
      adj_r_squared = adj_r_squared
    ) %>%
    select(model, n, term, estimate, std_error, t_value, p_value,
           ci_low, ci_high, r_squared, adj_r_squared)
}

fit_behavior_from_meg_models <- function(reg_data) {
  list(
    raw_covariate = lm(behavior_predictor ~ meg_effect + age_z + sex, data = reg_data),
    standardized_covariate = lm(behavior_predictor_z ~ meg_effect_z + age_z + sex, data = reg_data)
  )
}

safe_cor_test <- function(x, y, method) {
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]
  y <- y[ok]
  if (length(x) < 3 || sd(x) == 0 || sd(y) == 0) {
    return(tibble(method = method, estimate = NA_real_, p_value = NA_real_, n = length(x)))
  }
  out <- tryCatch(
    suppressWarnings(cor.test(x, y, method = method)),
    error = function(error) NULL
  )
  if (is.null(out)) {
    return(tibble(method = method, estimate = NA_real_, p_value = NA_real_, n = length(x)))
  }
  tibble(
    method = method,
    estimate = unname(out$estimate),
    p_value = out$p.value,
    n = length(x)
  )
}

metadata_from_rows <- function(df, path, cluster_value) {
  tibble(
    source_file = basename(path),
    analysis = first_or_na(df$analysis),
    phase = first_or_na(df$phase),
    effect = sub("^magnitude$", "reward", first_or_na(df$effect)),
    target_label = first_or_na(df$target_label),
    cluster_number = cluster_value,
    cluster_index = first_or_na(df$cluster_index),
    cluster_p = first_or_na(df$cluster_p),
    cluster_p_phasewise = first_or_na(df$cluster_p_phasewise),
    cluster_correction_multiplier = first_or_na(df$cluster_correction_multiplier),
    time_start_ms = first_or_na(df$time_start_ms),
    time_end_ms = first_or_na(df$time_end_ms),
    n_cluster_points = first_or_na(df$n_cluster_points),
    n_unique_space = first_or_na(df$n_unique_space),
    units = first_or_na(df$units),
    meg_effect_col = meg_effect_col
  )
}

is_planned_effect <- function(metadata) {
  phase <- as.character(metadata$phase)
  effect <- as.character(metadata$effect)

  if (effect == "interaction") return(TRUE)
  if (phase == "incentive" && effect == "reward") return(TRUE)
  if (phase == "feedback" && effect == "reward") return(TRUE)
  if (phase == "decision" && effect == "load") return(TRUE)
  if (phase == "transition" && effect == "load") return(TRUE)
  FALSE
}

predictor_col_for_effect <- function(effect) {
  effect <- as.character(effect)[1]
  if (!is.na(effect) && effect == "reward") return("beta_mb_magnitude")
  if (!is.na(effect) && effect == "load") return("beta_mb_load")
  if (!is.na(effect) && effect == "interaction") return("beta_mb_magnitude_load")
  "beta_mb_load"
}

add_metadata <- function(df, metadata) {
  bind_cols(metadata[rep(1, nrow(df)), ], df)
}

make_plot <- function(reg_data, metadata, out_stem, pearson_row, outcome_label) {
  plot_label <- paste0(
    "Pearson r = ", sprintf("%.2f", pearson_row$estimate),
    ", ", format_p(pearson_row$p_value),
    ", n = ", pearson_row$n
  )

  title <- paste0(metadata$phase, " ", metadata$analysis, " ", metadata$effect)
  subtitle <- paste0(
    "cluster ", metadata$cluster_number,
    ", ", sprintf("%.0f", metadata$time_start_ms), "-",
    sprintf("%.0f", metadata$time_end_ms), " ms"
  )

  p <- ggplot(reg_data, aes(x = meg_effect, y = behavior_predictor)) +
    geom_point(size = 2.4, alpha = 0.82, color = "#777777") +
    geom_smooth(method = "lm", se = TRUE, linewidth = 0.9, color = "#333333", fill = "#999999") +
    annotate(
      "text",
      x = -Inf,
      y = Inf,
      hjust = -0.05,
      vjust = 1.25,
      label = plot_label,
      size = 4
    ) +
    labs(
      x = metadata$meg_effect_col,
      y = sub("_magnitude", "_reward", outcome_label),
      title = title,
      subtitle = subtitle
    ) +
    theme_classic(base_size = 13)

  figure_stem <- paste(if (grepl("_b0$", outcome_label)) "b0" else "matched",
                       gsub("magnitude", "reward", metadata$target_label, fixed = TRUE),
                       paste0("C", metadata$cluster_number), meg_effect_col, sep = "_")
  ggsave(file.path(figure_dir, paste0(figure_stem, ".pdf")), p, width = 6, height = 4.6, device = grDevices::cairo_pdf, dpi = 600, fallback_resolution = 600)
}

make_regression_summary <- function(model_results, cor_results, focal_outcome_name) {
  if (nrow(model_results) == 0) return(tibble())

  model_results %>%
    filter(model == "standardized_covariate", term == "meg_effect_z") %>%
    select(
      source_file, analysis, phase, effect, target_label, cluster_number,
      cluster_index, cluster_p, cluster_p_phasewise, cluster_correction_multiplier,
      time_start_ms, time_end_ms, n, units,
      behavior_outcome = behavior_predictor,
      predictor_matched_to_meg_effect,
      meg_effect_col,
      meg_effect_std = estimate, meg_effect_std_se = std_error,
      t_value, p_value, ci_low, ci_high, r_squared, adj_r_squared
    ) %>%
    mutate(focal_outcome_name = focal_outcome_name) %>%
    left_join(
      cor_results %>%
        filter(method == "pearson") %>%
        select(
          source_file, target_label, cluster_number,
          pearson_r = estimate, pearson_p = p_value
        ),
      by = c("source_file", "target_label", "cluster_number")
    ) %>%
    left_join(
      cor_results %>%
        filter(method == "spearman") %>%
        select(
          source_file, target_label, cluster_number,
          spearman_rho = estimate, spearman_p = p_value
        ),
      by = c("source_file", "target_label", "cluster_number")
    ) %>%
    group_by(phase) %>%
    mutate(
      p_fdr_lm_within_phase = p.adjust(p_value, method = "BH"),
      p_fdr_pearson_within_phase = p.adjust(pearson_p, method = "BH")
    ) %>%
    ungroup() %>%
    group_by(analysis, effect) %>%
    mutate(
      p_fdr_lm_within_analysis_effect = p.adjust(p_value, method = "BH"),
      p_fdr_pearson_within_analysis_effect = p.adjust(pearson_p, method = "BH")
    ) %>%
    ungroup() %>%
    mutate(
      p_fdr_lm_all_tests = p.adjust(p_value, method = "BH"),
      p_fdr_pearson_all_tests = p.adjust(pearson_p, method = "BH")
    ) %>%
    arrange(p_value)
}

if (!file.exists(hbrl_fit_path)) {
  stop("Missing HBRL fit file: ", hbrl_fit_path)
}
if (!file.exists(covariate_file)) {
  stop("Missing covariate file: ", covariate_file)
}

fit_env <- new.env(parent = emptyenv())
load(hbrl_fit_path, envir = fit_env)
if (!exists("fit", envir = fit_env)) {
  stop("The HBRL RData file does not contain an object named 'fit'.")
}

posterior <- rstan::extract(fit_env$fit, pars = "beta_mb_effects", permuted = TRUE)

beta_mb_b0 <- apply(posterior$beta_mb_effects[, 1, ], 2, median)
beta_mb_magnitude <- apply(posterior$beta_mb_effects[, 2, ], 2, median)
beta_mb_load <- apply(posterior$beta_mb_effects[, 3, ], 2, median)
beta_mb_magnitude_load <- apply(posterior$beta_mb_effects[, 4, ], 2, median)

if (exists("data_files", envir = fit_env)) {
  data_files <- get("data_files", envir = fit_env)
  subids <- parse_subid(sub("_.*$", "", basename(data_files)))
} else {
  subids <- seq_along(beta_mb_load)
  warning("No data_files object found in HBRL RData; using subject index as subid.")
}

bhv_params <- tibble(
  subject_index = seq_along(beta_mb_load),
  subid = subids,
  beta_mb_b0 = beta_mb_b0,
  beta_mb_magnitude = beta_mb_magnitude,
  beta_mb_load = beta_mb_load,
  beta_mb_magnitude_load = beta_mb_magnitude_load
)

covariates <- read_csv(covariate_file, show_col_types = FALSE)
required_covariate_cols <- c("subid", "age", "sex")
missing_covariate_cols <- setdiff(required_covariate_cols, names(covariates))
if (length(missing_covariate_cols) > 0) {
  stop("Covariate file is missing required column(s): ", paste(missing_covariate_cols, collapse = ", "))
}

covariates <- covariates %>%
  transmute(
    subid = parse_subid(subid),
    age = suppressWarnings(as.numeric(age)),
    age_z = standardize_if_possible(age),
    sex = factor(sex, levels = c(0, 1), labels = c("female", "male"))
  )

if (any(is.na(covariates$subid))) {
  stop("Covariate file contains missing or unparsable subid values.")
}
if (any(duplicated(covariates$subid))) {
  stop("Covariate file contains duplicated subid values.")
}

bhv_params <- bhv_params %>%
  left_join(covariates, by = "subid")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
write_csv(bhv_params, file.path(output_dir, "hbrl_beta_mb_effects_with_covariates_52subs.csv"))
message("Extracted HBRL beta_mb effects for ", nrow(bhv_params), " subjects.")
message("Loaded covariates for ", nrow(covariates), " subjects.")

meg_files <- list.files(
  meg_effect_dir,
  pattern = "_subject_effects\\.csv$",
  full.names = TRUE
)

if (length(meg_files) == 0) {
  stop("No *_subject_effects.csv files found in: ", meg_effect_dir)
}

all_model_rows <- list()
all_cor_rows <- list()
all_reg_data <- list()
all_b0_model_rows <- list()
all_b0_cor_rows <- list()
all_b0_reg_data <- list()
skipped <- list()

for (path in meg_files) {
  message("\nReading MEG file: ", basename(path))
  meg_raw <- read_csv(path, show_col_types = FALSE)

  if (!meg_effect_col %in% names(meg_raw)) {
    skipped[[length(skipped) + 1]] <- tibble(
      source_file = basename(path),
      reason = paste0("Missing MEG_EFFECT_COL: ", meg_effect_col)
    )
    next
  }

  subject_col <- get_optional_col(
    meg_raw,
    c("subid", "subject", "Subject", "participant", "Participant", "id", "ID")
  )
  if (is.na(subject_col)) {
    skipped[[length(skipped) + 1]] <- tibble(
      source_file = basename(path),
      reason = "Missing subject column"
    )
    next
  }

  if ("cluster_number" %in% names(meg_raw)) {
    meg_raw$.cluster <- as.character(meg_raw$cluster_number)
  } else if ("Cluster" %in% names(meg_raw)) {
    meg_raw$.cluster <- as.character(meg_raw$Cluster)
  } else if ("Cluster #" %in% names(meg_raw)) {
    meg_raw$.cluster <- as.character(meg_raw[["Cluster #"]])
  } else {
    meg_raw$.cluster <- "1"
  }

  cluster_values <- sort(unique(meg_raw$.cluster))

  for (cluster_value in cluster_values) {
    df_cluster <- meg_raw %>% filter(.cluster == cluster_value)
    metadata <- metadata_from_rows(df_cluster, path, cluster_value)

    if (!is_planned_effect(metadata)) {
      skipped[[length(skipped) + 1]] <- metadata %>%
        mutate(reason = "Skipped by planned-effects filter")
      next
    }

    outcome_col <- predictor_col_for_effect(metadata$effect)
    metadata <- metadata %>%
      mutate(
        behavior_predictor = outcome_col,
        predictor_matched_to_meg_effect = TRUE
      )

    cluster_number_for_name <- suppressWarnings(as.integer(cluster_value))
    if (is.na(cluster_number_for_name)) cluster_number_for_name <- 0

    meg_effects <- df_cluster %>%
      mutate(subid = parse_subid(.data[[subject_col]])) %>%
      group_by(subid) %>%
      summarise(
        meg_effect = mean(.data[[meg_effect_col]], na.rm = TRUE),
        .groups = "drop"
      ) %>%
      filter(!is.na(subid), is.finite(meg_effect))

    reg_data <- inner_join(bhv_params, meg_effects, by = "subid") %>%
      mutate(
        behavior_predictor = .data[[outcome_col]],
        behavior_predictor_z = standardize_if_possible(behavior_predictor),
        meg_effect_z = standardize_if_possible(meg_effect)
      ) %>%
      filter(
        is.finite(meg_effect),
        is.finite(behavior_predictor),
        is.finite(age),
        !is.na(sex)
      )

    if (nrow(reg_data) < 10) {
      skipped[[length(skipped) + 1]] <- metadata %>%
        mutate(reason = paste0("Only ", nrow(reg_data), " matched subjects"))
      next
    }

    models <- fit_behavior_from_meg_models(reg_data)
    model_rows <- bind_rows(Map(tidy_lm_base, models, names(models))) %>%
      add_metadata(metadata)
    cor_rows <- bind_rows(
      safe_cor_test(reg_data$meg_effect, reg_data$behavior_predictor, "pearson"),
      safe_cor_test(reg_data$meg_effect, reg_data$behavior_predictor, "spearman")
    ) %>%
      add_metadata(metadata)

    all_model_rows[[length(all_model_rows) + 1]] <- model_rows
    all_cor_rows[[length(all_cor_rows) + 1]] <- cor_rows
    all_reg_data[[length(all_reg_data) + 1]] <- reg_data %>%
      mutate(
        source_file = basename(path),
        target_label = metadata$target_label,
        cluster_number = metadata$cluster_number,
        analysis = metadata$analysis,
        phase = metadata$phase,
        effect = metadata$effect,
        behavior_outcome_name = metadata$behavior_predictor,
        predictor_matched_to_meg_effect = TRUE,
        meg_effect_col = metadata$meg_effect_col
      )

    out_stem <- safe_name(paste0(
      "behavior_from_meg_",
      metadata$target_label,
      "_cluster_",
      sprintf("%02d", cluster_number_for_name),
      "_outcome_",
      outcome_col,
      "_",
      meg_effect_col
    ))

    write_csv(reg_data, file.path(data_dir, paste0(out_stem, "_regression_data.csv")))
    pearson_row <- cor_rows %>% filter(method == "pearson") %>% slice(1)
    make_plot(reg_data, metadata, out_stem, pearson_row, outcome_col)

    message(
      "  cluster ", cluster_value,
      ", outcome = ", outcome_col,
      ": n = ", nrow(reg_data),
      ", Pearson r = ", sprintf("%.3f", pearson_row$estimate),
      ", p = ", sprintf("%.3f", pearson_row$p_value)
    )

    metadata_b0 <- metadata %>%
      mutate(
        behavior_predictor = "beta_mb_b0",
        predictor_matched_to_meg_effect = FALSE
      )

    b0_reg_data <- inner_join(bhv_params, meg_effects, by = "subid") %>%
      mutate(
        behavior_predictor = beta_mb_b0,
        behavior_predictor_z = standardize_if_possible(behavior_predictor),
        meg_effect_z = standardize_if_possible(meg_effect)
      ) %>%
      filter(
        is.finite(meg_effect),
        is.finite(behavior_predictor),
        is.finite(age),
        !is.na(sex)
      )

    b0_models <- fit_behavior_from_meg_models(b0_reg_data)
    b0_model_rows <- bind_rows(Map(tidy_lm_base, b0_models, names(b0_models))) %>%
      add_metadata(metadata_b0)
    b0_cor_rows <- bind_rows(
      safe_cor_test(b0_reg_data$meg_effect, b0_reg_data$behavior_predictor, "pearson"),
      safe_cor_test(b0_reg_data$meg_effect, b0_reg_data$behavior_predictor, "spearman")
    ) %>%
      add_metadata(metadata_b0)

    all_b0_model_rows[[length(all_b0_model_rows) + 1]] <- b0_model_rows
    all_b0_cor_rows[[length(all_b0_cor_rows) + 1]] <- b0_cor_rows
    all_b0_reg_data[[length(all_b0_reg_data) + 1]] <- b0_reg_data %>%
      mutate(
        source_file = basename(path),
        target_label = metadata$target_label,
        cluster_number = metadata$cluster_number,
        analysis = metadata$analysis,
        phase = metadata$phase,
        effect = metadata$effect,
        behavior_outcome_name = "beta_mb_b0",
        predictor_matched_to_meg_effect = FALSE,
        meg_effect_col = metadata$meg_effect_col
      )

    out_stem_b0 <- safe_name(paste0(
      "behavior_from_meg_b0_sensitivity_",
      metadata$target_label,
      "_cluster_",
      sprintf("%02d", cluster_number_for_name),
      "_",
      meg_effect_col
    ))

    write_csv(b0_reg_data, file.path(data_dir, paste0(out_stem_b0, "_regression_data.csv")))
    b0_pearson_row <- b0_cor_rows %>% filter(method == "pearson") %>% slice(1)
    make_plot(b0_reg_data, metadata_b0, out_stem_b0, b0_pearson_row, "beta_mb_b0")

    message(
      "  cluster ", cluster_value,
      ", outcome = beta_mb_b0 sensitivity",
      ": n = ", nrow(b0_reg_data),
      ", Pearson r = ", sprintf("%.3f", b0_pearson_row$estimate),
      ", p = ", sprintf("%.3f", b0_pearson_row$p_value)
    )
  }
}

model_results <- bind_rows(all_model_rows)
cor_results <- bind_rows(all_cor_rows)
regression_data_all <- bind_rows(all_reg_data)
b0_model_results <- bind_rows(all_b0_model_rows)
b0_cor_results <- bind_rows(all_b0_cor_rows)
b0_regression_data_all <- bind_rows(all_b0_reg_data)
skipped_results <- bind_rows(skipped)

beta_summary <- make_regression_summary(model_results, cor_results, "matched_effect_beta_mb")
b0_summary <- make_regression_summary(b0_model_results, b0_cor_results, "beta_mb_b0")

model_results_path <- file.path(output_dir, paste0(result_prefix, "_regression_results.csv"))
cor_results_path <- file.path(output_dir, paste0(result_prefix, "_correlation_results.csv"))
summary_path <- file.path(output_dir, paste0(result_prefix, "_summary.csv"))
regression_data_path <- file.path(output_dir, paste0(result_prefix, "_regression_data.csv"))
skipped_path <- file.path(output_dir, paste0(result_prefix, "_skipped_targets.csv"))

b0_model_results_path <- file.path(output_dir, paste0(result_prefix, "_b0_sensitivity_regression_results.csv"))
b0_cor_results_path <- file.path(output_dir, paste0(result_prefix, "_b0_sensitivity_correlation_results.csv"))
b0_summary_path <- file.path(output_dir, paste0(result_prefix, "_b0_sensitivity_summary.csv"))
b0_regression_data_path <- file.path(output_dir, paste0(result_prefix, "_b0_sensitivity_regression_data.csv"))

write_csv(model_results, model_results_path)
write_csv(cor_results, cor_results_path)
write_csv(beta_summary, summary_path)
write_csv(regression_data_all, regression_data_path)
write_csv(skipped_results, skipped_path)

write_csv(b0_model_results, b0_model_results_path)
write_csv(b0_cor_results, b0_cor_results_path)
write_csv(b0_summary, b0_summary_path)
write_csv(b0_regression_data_all, b0_regression_data_path)

message("\nDone.")
message("Model results: ", model_results_path)
message("Correlation results: ", cor_results_path)
message("Summary: ", summary_path)
message("b0 sensitivity model results: ", b0_model_results_path)
message("b0 sensitivity correlation results: ", b0_cor_results_path)
message("b0 sensitivity summary: ", b0_summary_path)
message("Figures: ", figure_dir)
