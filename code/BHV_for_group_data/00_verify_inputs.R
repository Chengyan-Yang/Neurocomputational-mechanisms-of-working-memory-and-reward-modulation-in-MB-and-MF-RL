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
input <- file.path(here, "..", "..", "group_data", "BHV")
cells <- readxl::read_excel(file.path(input, "analysis", "parameters_generated.xlsx"))
vars <- c("subid", "load", "magnitude", "cr", "w", "beta_mb", "beta_mf")
stopifnot(all(vars %in% names(cells)), !anyNA(cells[vars]),
          setequal(unique(cells$subid), 1:52),
          setequal(unique(cells$load), c("low", "high")),
          setequal(unique(cells$magnitude), c("small", "large")),
          all(table(cells$subid, cells$load, cells$magnitude) == 1))
demo <- read.csv(file.path(input, "analysis", "demographics_summary.csv"))
stopifnot(nrow(demo) == 1, demo$n == 52, demo$female_n + demo$male_n == demo$n)
draws <- read.csv(file.path(input, "posterior", "hbrl_group_posterior_draws.csv"))
stopifnot(nrow(draws) > 1, !anyDuplicated(draws$draw), !anyNA(draws),
          all(outer(c("beta_mb_", "beta_mf_"), c("small_lload", "large_lload",
              "small_hload", "large_hload"), paste0) %in% names(draws)))
for (model in c("RL", "HBRL")) {
  directory <- file.path(input, "validation", model)
  summary <- read.csv(file.path(directory, "parameter_recovery_summary.csv"))
  points <- read.csv(file.path(directory, "parameter_recovery_points.csv"))
  for (i in seq_len(nrow(summary))) {
    row <- summary[i, ]
    if (model == "RL") {
      selected <- points[points$load == row$load, ]
      x <- selected[[paste0("true_", row$param)]]
      y <- selected[[paste0("recovered_", row$param)]]
    } else {
      selected <- points[points$parameter == row$parameter, ]
      x <- selected$generating
      y <- selected$recovered
    }
    stopifnot(length(x) == row$n, abs(cor(x, y) - row$pearson_r) < 1e-8,
              abs(sqrt(mean((y - x)^2)) - row$rmse) < 1e-8)
  }
}
message("Verified 52 participants, 208 condition rows, ", nrow(draws),
        " matched posterior draws, and both recovery tables.")
