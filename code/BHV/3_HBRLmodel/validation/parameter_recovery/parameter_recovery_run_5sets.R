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
# One-step runner for 5 HBRL parameter recovery datasets.
#
# This version is adapted for the MEG 52-participant HBRL model.

analysis_dir <- file.path(here, "..", "..")
script_dir <- here
script_dir <- normalizePath(script_dir, winslash = "/", mustWork = TRUE)
`%||%` <- function(x, y) if (length(x) == 0 || is.na(x) || identical(x, "")) y else x

# RStudio often sources this file from another working directory. Keep script_dir
# anchored to this validation folder so the child scripts are never
# searched from getwd() or a reference-analysis directory.
if (!file.exists(file.path(script_dir, "parameter_recovery_generate_data.R"))) {
  stop("Cannot find HBRL parameter recovery scripts in: ", script_dir)
}

recovery_label <- Sys.getenv("HBRL_RECOVERY_LABEL") %||% "MEG_HBRL_5sets"
Sys.setenv(HBRL_RECOVERY_LABEL = recovery_label)
Sys.setenv(HBRL_RECOVERY_N_DATASETS = Sys.getenv("HBRL_RECOVERY_N_DATASETS") %||% "5")
Sys.setenv(HBRL_RECOVERY_ITER = Sys.getenv("HBRL_RECOVERY_ITER") %||% "4000")
Sys.setenv(HBRL_RECOVERY_WARMUP = Sys.getenv("HBRL_RECOVERY_WARMUP") %||% "2000")
Sys.setenv(HBRL_RECOVERY_CHAINS = Sys.getenv("HBRL_RECOVERY_CHAINS") %||% "4")
Sys.setenv(HBRL_RECOVERY_CORES = Sys.getenv("HBRL_RECOVERY_CORES") %||% "4")
Sys.setenv(HBRL_RECOVERY_OVERWRITE = Sys.getenv("HBRL_RECOVERY_OVERWRITE") %||% "FALSE")

manifest_file <- file.path(script_dir, paste0("outputs_", recovery_label), "recovery_datasets.csv")
if (file.exists(manifest_file) && Sys.getenv("HBRL_RECOVERY_OVERWRITE") != "TRUE") {
  manifest <- read.csv(manifest_file)
  if (nrow(manifest) != as.integer(Sys.getenv("HBRL_RECOVERY_N_DATASETS")) ||
      !all(dir.exists(manifest$data_dir))) stop("Existing simulations do not match the requested recovery run.")
  message("Step 1/3: reusing existing simulated datasets for a consistent resume.")
} else {
  message("Step 1/3: generating simulated recovery datasets ...")
  source(file.path(script_dir, "parameter_recovery_generate_data.R"), chdir = TRUE)
}

message("Step 2/3: refitting HBRL to simulated datasets ...")
source(file.path(script_dir, "parameter_recovery_fit.R"), chdir = TRUE)

message("Step 3/3: evaluating parameter recovery ...")
source(file.path(script_dir, "parameter_recovery_evaluate.R"), chdir = TRUE)

message("Done. Results saved to: ", file.path(analysis_dir, "validation", "parameter_recovery", paste0("outputs_", recovery_label)))

