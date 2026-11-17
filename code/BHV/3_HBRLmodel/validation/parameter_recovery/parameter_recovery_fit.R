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
# Refit the HBRL model to simulated datasets generated for parameter recovery.

suppressPackageStartupMessages({
  library(rstan)
  library(dplyr)
  library(readr)
})

`%||%` <- function(x, y) {
  if (length(x) == 0 || is.na(x) || identical(x, "")) y else x
}

analysis_dir <- file.path(here, "..", "..")
output_id <- "MEG_HBRL"
expected_N <- 52
recovery_label <- Sys.getenv("HBRL_RECOVERY_LABEL") %||% output_id
recovery_dir <- file.path(analysis_dir, "validation", "parameter_recovery", paste0("outputs_", recovery_label))
datasets_file <- file.path(recovery_dir, "recovery_datasets.csv")
fit_dir <- file.path(recovery_dir, "fits")
dir.create(fit_dir, recursive = TRUE, showWarnings = FALSE)

stan_file <- file.path(analysis_dir, "mfmblearning_magnitude&load.stan")

iter <- as.integer(Sys.getenv("HBRL_RECOVERY_ITER") %||% "4000")
warmup <- as.integer(Sys.getenv("HBRL_RECOVERY_WARMUP") %||% "2000")
chains <- as.integer(Sys.getenv("HBRL_RECOVERY_CHAINS") %||% "4")
cores <- as.integer(Sys.getenv("HBRL_RECOVERY_CORES") %||% as.character(chains))
seed <- as.integer(Sys.getenv("HBRL_RECOVERY_FIT_SEED") %||% "260220")
dataset_filter <- Sys.getenv("HBRL_RECOVERY_DATASETS")

if (!file.exists(datasets_file)) {
  stop("Missing recovery dataset table. Run parameter_recovery_generate_data.R first: ", datasets_file)
}

datasets <- readr::read_csv(datasets_file, show_col_types = FALSE)
if (nzchar(dataset_filter)) {
  keep <- trimws(strsplit(dataset_filter, ",")[[1]])
  datasets <- datasets %>% filter(dataset_id %in% keep)
}
if (nrow(datasets) == 0) stop("No recovery datasets selected.")

pars <- c(
  "mu_alpha_hload_logit", "sd_alpha_hload_logit",
  "mu_alpha_lload_logit", "sd_alpha_lload_logit",
  "mu_beta_mf_effects", "sd_beta_mf_effects",
  "Omega_mf",
  "mu_beta_mb_effects", "sd_beta_mb_effects",
  "Omega_mb",
  "mu_respst_hload", "sd_respst_hload",
  "mu_respst_lload", "sd_respst_lload",
  "mu_st_hload", "sd_st_hload",
  "mu_st_lload", "sd_st_lload",
  "alpha_hload", "alpha_lload",
  "beta_mf_effects", "beta_mf_small_hload", "beta_mf_large_hload", "beta_mf_small_lload", "beta_mf_large_lload",
  "beta_mb_effects", "beta_mb_small_hload", "beta_mb_large_hload", "beta_mb_small_lload", "beta_mb_large_lload",
  "st_hload", "st_lload", "respst_hload", "respst_lload",
  "alpha_hload_logit_z", "alpha_lload_logit_z",
  "st_hload_z", "st_lload_z", "respst_hload_z", "respst_lload_z"
)

build_stan_data <- function(folder_path) {
  data_files <- sort(list.files(path = folder_path, pattern = "\\.csv$", full.names = TRUE))
  parsed_ids <- suppressWarnings(as.integer(sub("_.*$", "", basename(data_files))))
  if (length(parsed_ids) == length(data_files) && !any(is.na(parsed_ids))) {
    data_files <- data_files[order(parsed_ids)]
  }
  N <- length(data_files)
  if (!is.na(expected_N) && N != expected_N) {
    stop("Expected ", expected_N, " participants, but found ", N, " csv files in ", folder_path)
  }

  stan_data <- NULL
  for (i in seq_along(data_files)) {
    raw_data <- read.csv(data_files[i], header = TRUE)
    raw_data <- raw_data[raw_data$probe != 1, ]

    nT <- length(raw_data$participant)
    s1 <- as.numeric(raw_data$pairs)
    s2 <- as.numeric(raw_data$animal)
    a <- ifelse(raw_data$choice %% 2 != 0, 1, 2)
    r <- raw_data$points
    magnitude <- ifelse(raw_data$magnitude == "small", 1, 5)
    load <- ifelse(raw_data$load == "low", 1, 2)
    part <- raw_data$part
    display <- with(raw_data, cbind(
      ifelse(stimuli_left %% 2 != 0, 1, 2),
      ifelse(stimuli_right %% 2 != 0, 1, 2)
    ))

    if (i == 1) {
      stan_data <- list(
        N = N, nT = nT, s1 = s1, s2 = s2, a = a, r = r,
        magnitude = magnitude, load = load, part = part, display = t(display)
      )
    } else {
      stan_data <- list(
        N = N,
        nT = c(stan_data$nT, nT),
        s1 = c(stan_data$s1, s1),
        s2 = c(stan_data$s2, s2),
        a = c(stan_data$a, a),
        r = c(stan_data$r, r),
        magnitude = c(stan_data$magnitude, magnitude),
        load = c(stan_data$load, load),
        part = c(stan_data$part, part),
        display = cbind(stan_data$display, t(display))
      )
    }
  }
  list(data = stan_data, data_files = data_files, folder_path = folder_path)
}

fit_one_dataset <- function(dataset_id, folder_path) {
  out_file <- file.path(fit_dir, paste0("fit_", dataset_id, ".RData"))
  if (file.exists(out_file) && Sys.getenv("HBRL_RECOVERY_OVERWRITE") != "TRUE") {
    message("Skipping existing fit: ", out_file)
    return(out_file)
  }

  message("Fitting ", dataset_id, " from: ", folder_path)
  built <- build_stan_data(folder_path)
  data <- built$data
  data_files <- built$data_files

  fit <- rstan::stan(
    file = stan_file,
    data = data,
    pars = pars,
    iter = iter,
    warmup = warmup,
    chains = chains,
    cores = cores,
    refresh = 10,
    init_r = 0.5,
    seed = seed
  )

  save(
    list = c("fit", "data", "data_files", "folder_path", "dataset_id", "iter", "warmup", "chains", "seed"),
    file = out_file
  )
  out_file
}

fit_files <- mapply(fit_one_dataset, datasets$dataset_id, datasets$data_dir)

readr::write_csv(
  tibble(dataset_id = datasets$dataset_id, fit_file = as.character(fit_files)),
  file.path(recovery_dir, "recovery_fit_files.csv")
)

message("Recovery fits saved to: ", fit_dir)


