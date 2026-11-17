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
# Evaluate HBRL parameter recovery by comparing generating and recovered values.

suppressPackageStartupMessages({
  library(rstan)
  library(dplyr)
  library(tidyr)
  library(readr)
})

analysis_dir <- file.path(here, "..", "..")
output_id <- "MEG_HBRL"
`%||%` <- function(x, y) {
  if (length(x) == 0 || is.na(x) || identical(x, "")) y else x
}
recovery_label <- Sys.getenv("HBRL_RECOVERY_LABEL") %||% output_id
recovery_dir <- file.path(analysis_dir, "validation", "parameter_recovery", paste0("outputs_", recovery_label))
fit_files_path <- file.path(recovery_dir, "recovery_fit_files.csv")
generating_path <- file.path(recovery_dir, "generating_parameters_long.csv")
figure_dir <- file.path(recovery_dir, "Figures", "single_panels")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(fit_files_path)) stop("Missing recovery fit table: ", fit_files_path)
if (!file.exists(generating_path)) stop("Missing generating parameters: ", generating_path)

fit_files <- readr::read_csv(fit_files_path, show_col_types = FALSE)
generating <- readr::read_csv(generating_path, show_col_types = FALSE)

extract_recovered_parameters <- function(fit_file, dataset_id) {
  load(fit_file)
  if (!exists("fit")) stop("Missing fit object in: ", fit_file)

  posterior <- rstan::extract(
    fit,
    pars = c(
      "alpha_hload", "alpha_lload",
      "beta_mf_effects", "beta_mb_effects",
      "beta_mf_small_lload", "beta_mf_large_lload",
      "beta_mf_small_hload", "beta_mf_large_hload",
      "beta_mb_small_lload", "beta_mb_large_lload",
      "beta_mb_small_hload", "beta_mb_large_hload",
      "st_hload", "st_lload",
      "respst_hload", "respst_lload"
    ),
    permuted = TRUE
  )

  # Label recovered Stan columns with the participant IDs in the same order used
  # during refitting. Simulated CSV files may be sorted lexicographically
  # (1, 10, 11, ...) when read by list.files(), so the Stan column index is not
  # always the same as the numeric participant ID. We keep both: subject_index is
  # the Stan column order, and subid is the participant ID parsed from data_files.
  n_subjects <- dim(posterior$alpha_hload)[2]
  subids <- seq_len(n_subjects)
  if (exists("data_files")) {
    parsed_subids <- suppressWarnings(as.integer(sub("_.*$", "", basename(data_files))))
    if (length(parsed_subids) == n_subjects && !any(is.na(parsed_subids))) {
      subids <- parsed_subids
    }
  }

  tibble(
    dataset_id = dataset_id,
    subject_index = seq_along(subids),
    subid = subids,
    alpha_hload = apply(posterior$alpha_hload, 2, median),
    alpha_lload = apply(posterior$alpha_lload, 2, median),
    beta_mf_b0 = apply(posterior$beta_mf_effects[, 1, ], 2, median),
    beta_mf_magnitude = apply(posterior$beta_mf_effects[, 2, ], 2, median),
    beta_mf_load = apply(posterior$beta_mf_effects[, 3, ], 2, median),
    beta_mf_magnitude_load = apply(posterior$beta_mf_effects[, 4, ], 2, median),
    beta_mb_b0 = apply(posterior$beta_mb_effects[, 1, ], 2, median),
    beta_mb_magnitude = apply(posterior$beta_mb_effects[, 2, ], 2, median),
    beta_mb_load = apply(posterior$beta_mb_effects[, 3, ], 2, median),
    beta_mb_magnitude_load = apply(posterior$beta_mb_effects[, 4, ], 2, median),
    beta_mf_small_lload = apply(posterior$beta_mf_small_lload, 2, median),
    beta_mf_large_lload = apply(posterior$beta_mf_large_lload, 2, median),
    beta_mf_small_hload = apply(posterior$beta_mf_small_hload, 2, median),
    beta_mf_large_hload = apply(posterior$beta_mf_large_hload, 2, median),
    beta_mb_small_lload = apply(posterior$beta_mb_small_lload, 2, median),
    beta_mb_large_lload = apply(posterior$beta_mb_large_lload, 2, median),
    beta_mb_small_hload = apply(posterior$beta_mb_small_hload, 2, median),
    beta_mb_large_hload = apply(posterior$beta_mb_large_hload, 2, median),
    st_hload = apply(posterior$st_hload, 2, median),
    st_lload = apply(posterior$st_lload, 2, median),
    respst_hload = apply(posterior$respst_hload, 2, median),
    respst_lload = apply(posterior$respst_lload, 2, median)
  )
}

recovered <- bind_rows(mapply(
  extract_recovered_parameters,
  fit_files$fit_file,
  fit_files$dataset_id,
  SIMPLIFY = FALSE
))

readr::write_csv(recovered, file.path(recovery_dir, "recovered_parameters_long.csv"))

parameter_names <- setdiff(intersect(names(generating), names(recovered)), c("dataset_id", "draw_id", "subject_index", "subid"))

generating_long <- generating %>%
  select(dataset_id, subject_index, subid, all_of(parameter_names)) %>%
  pivot_longer(all_of(parameter_names), names_to = "parameter", values_to = "generating")

recovered_long <- recovered %>%
  select(dataset_id, recovered_subject_index = subject_index, subid, all_of(parameter_names)) %>%
  pivot_longer(all_of(parameter_names), names_to = "parameter", values_to = "recovered")

comparison_long <- generating_long %>%
  inner_join(
    recovered_long,
    by = c("dataset_id", "subid", "parameter")
  ) %>%
  select(dataset_id, subject_index, recovered_subject_index, subid, parameter, generating, recovered)

if (nrow(comparison_long) != nrow(generating_long)) {
  warning(
    "Recovery comparison retained ", nrow(comparison_long),
    " rows, expected ", nrow(generating_long),
    ". Check participant IDs in generating/recovered tables."
  )
}

cor_p_value <- function(x, y, method = "pearson") {
  ok <- complete.cases(x, y)
  if (sum(ok) < 3 || sd(x[ok]) == 0 || sd(y[ok]) == 0) return(NA_real_)
  unname(stats::cor.test(x[ok], y[ok], method = method)$p.value)
}

p_stars <- function(p) {
  case_when(
    is.na(p) ~ "n/a",
    p < 0.001 ~ "***",
    p < 0.01 ~ "**",
    p < 0.05 ~ "*",
    TRUE ~ "n.s."
  )
}

recovery_summary <- comparison_long %>%
  group_by(parameter) %>%
  summarise(
    n = sum(complete.cases(generating, recovered)),
    pearson_r = cor(generating, recovered, use = "complete.obs", method = "pearson"),
    pearson_p = cor_p_value(generating, recovered, method = "pearson"),
    pearson_sig = p_stars(pearson_p),
    spearman_r = cor(generating, recovered, use = "complete.obs", method = "spearman"),
    spearman_p = cor_p_value(generating, recovered, method = "spearman"),
    spearman_sig = p_stars(spearman_p),
    bias = mean(recovered - generating, na.rm = TRUE),
    rmse = sqrt(mean((recovered - generating)^2, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  arrange(desc(pearson_r))

readr::write_csv(comparison_long, file.path(recovery_dir, "parameter_recovery_comparison_long.csv"))
readr::write_csv(recovery_summary, file.path(recovery_dir, "parameter_recovery_summary.csv"))

# Use the same standalone Matplotlib renderer as the random HBRL reference.
# This keeps the final figures identical whether launched from R or Python.
plot_script <- file.path(analysis_dir, "validation", "parameter_recovery", "plot_hbrl_recovery_figures.py")
requested_python <- Sys.getenv("HBRL_PLOT_PYTHON")
if (nzchar(requested_python)) {
  candidates <- requested_python
} else {
  candidates <- unname(Sys.which(c("python", "python3")))
}
plot_python <- NULL
for (candidate in unique(candidates[nzchar(candidates) & file.exists(candidates)])) {
  # --help loads the renderer and its dependencies without reading or writing results.
  status <- tryCatch(system2(candidate, c(shQuote(plot_script), "--help"),
                             stdout = FALSE, stderr = FALSE), error = function(e) 1L)
  if (isTRUE(status == 0L)) {
    plot_python <- candidate
    break
  }
}
if (is.null(plot_python)) {
  stop("Recovery statistics saved. Set HBRL_PLOT_PYTHON to a Python executable with numpy, pandas, scipy and matplotlib. To retry only the figures, run ", plot_script, " --label ", recovery_label)
}
message("Rendering recovery panels with ", plot_python)
plot_log <- suppressWarnings(system2(plot_python, c(shQuote(plot_script), "--label", shQuote(recovery_label)),
                                     stdout = TRUE, stderr = TRUE))
if (length(plot_log)) message(paste(plot_log, collapse = "\n"))
if (!is.null(attr(plot_log, "status")) && attr(plot_log, "status") != 0L) {
  stop("Recovery statistics saved. Renderer exited with code ", attr(plot_log, "status"),
       ". To retry only the figures, run ", plot_script, " --label ", recovery_label)
}
message("Parameter recovery summary saved to: ", file.path(recovery_dir, "parameter_recovery_summary.csv"))
message("Parameter recovery figures saved to: ", figure_dir)
print(recovery_summary)
