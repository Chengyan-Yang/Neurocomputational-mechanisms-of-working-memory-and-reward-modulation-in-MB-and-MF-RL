# Export subject-level posterior draws from the saved MEG 52-subject stanfit.
# This only reads the fit object; it does not refit the model.

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) == 0) {
  stop("Run this script with Rscript so its validation directory can be resolved.")
}
script_path <- normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/")
validation_dir <- dirname(script_path)
analysis_dir <- dirname(validation_dir)
fit_path <- file.path(analysis_dir, "data_mfmblearning_magnitude&load.RData")
output_dir <- file.path(Sys.getenv("HBRL52_OUTPUT_DIR", unset=file.path(validation_dir, "outputs")), "mcmc")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(fit_path)) stop("Missing 52-subject fit file: ", fit_path)
fit_env <- new.env(parent = globalenv())
load(fit_path, envir = fit_env)
if (!exists("fit", envir = fit_env, inherits = FALSE)) stop("RData does not contain 'fit'.")
if (!exists("data", envir = fit_env, inherits = FALSE)) stop("RData does not contain Stan input 'data'.")

stan_data <- fit_env$data
if (stan_data$N != 52L) stop("Expected fit data for 52 participants; found N=", stan_data$N)
fit <- fit_env$fit
sim <- attributes(fit)[["sim"]]
if (is.null(sim) || is.null(sim$samples) || is.null(sim$warmup2)) {
  stop("Could not read saved Stan chains/warmup from the fit object.")
}

vars <- c(
  "alpha_lload", "alpha_hload",
  "beta_mf_small_lload", "beta_mf_large_lload",
  "beta_mf_small_hload", "beta_mf_large_hload",
  "beta_mb_small_lload", "beta_mb_large_lload",
  "beta_mb_small_hload", "beta_mb_large_hload",
  "st_lload", "st_hload", "respst_lload", "respst_hload"
)

extract_matrix <- function(chain_samples, var_name, keep) {
  cols <- names(chain_samples)[grepl(paste0("^", var_name, "\\["), names(chain_samples))]
  if (length(cols) != 52L) {
    stop("Expected 52 subject columns for ", var_name, "; found ", length(cols))
  }
  indices <- as.integer(sub(paste0("^", var_name, "\\[([0-9]+)\\]$"), "\\1", cols))
  if (!setequal(indices, 1:52)) stop("Invalid subject indices for ", var_name)
  cols <- cols[order(indices)]
  mat <- do.call(cbind, lapply(cols, function(nm) chain_samples[[nm]][keep]))
  colnames(mat) <- paste0(var_name, "_", seq_len(ncol(mat)))
  mat
}

samples <- sim$samples
if (length(sim$warmup2) != length(samples)) stop("Warmup metadata does not match chain count.")
rows <- vector("list", length(samples))
for (chain in seq_along(samples)) {
  n_iter <- length(samples[[chain]][[1]])
  keep <- seq.int(sim$warmup2[chain] + 1L, n_iter)
  mats <- lapply(vars, function(v) extract_matrix(samples[[chain]], v, keep))
  chain_df <- as.data.frame(do.call(cbind, mats), check.names = FALSE)
  chain_df$chain <- chain
  chain_df$draw <- seq_along(keep)
  rows[[chain]] <- chain_df[, c("chain", "draw", setdiff(names(chain_df), c("chain", "draw")))]
}

draws <- do.call(rbind, rows)
write.csv(draws, file.path(output_dir, "hbrl_subject_draws.csv"), row.names = FALSE)

data_files <- fit_env$data_files
if (is.null(data_files) || length(data_files) != 52L) {
  stop("Expected 52 saved data_files in the fit RData object.")
}
subject_ids <- suppressWarnings(as.integer(sub("_.*$", "", basename(data_files))))
if (anyNA(subject_ids) || anyDuplicated(subject_ids)) {
  stop("Could not derive 52 unique participant IDs from saved data_files.")
}
subject_map <- data.frame(
  subject_index = seq_along(subject_ids),
  subject_id = subject_ids,
  source_file = basename(data_files)
)
write.csv(subject_map, file.path(output_dir, "stan_subject_index.csv"), row.names = FALSE)
fit_trials <- data.frame(
  subject_index=rep(seq_len(stan_data$N), stan_data$nT),
  s1=stan_data$s1, s2=stan_data$s2, a=stan_data$a, r=stan_data$r,
  magnitude=stan_data$magnitude, load=stan_data$load, part=stan_data$part,
  display_left=stan_data$display[1,], display_right=stan_data$display[2,]
)
write.csv(fit_trials, file.path(output_dir, "stan_fit_trials.csv"), row.names=FALSE)

metadata <- data.frame(
  fit_file = normalizePath(fit_path, winslash = "/"),
  subjects = stan_data$N,
  chains = length(samples),
  warmup = sim$warmup2[1],
  retained_draws_per_chain = nrow(rows[[1]]),
  total_retained_draws = nrow(draws)
)
write.csv(metadata, file.path(output_dir, "stan_draw_export_metadata.csv"), row.names = FALSE)
cat("Exported subject posterior draws for 52 participants to:", output_dir, "\n")
