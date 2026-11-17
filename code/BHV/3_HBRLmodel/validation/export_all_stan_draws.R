# Export every saved Stan parameter for reference-style full MCMC diagnostics.
# The wide CSV can be large; this script only reads the existing fit object.

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
if (fit_env$data$N != 52L) stop("Expected fit data for 52 participants; found N=", fit_env$data$N)

sim <- attributes(fit_env$fit)[["sim"]]
if (is.null(sim) || is.null(sim$samples) || is.null(sim$warmup2) || is.null(sim$fnames_oi)) {
  stop("Could not read saved Stan chains, warmup or flat parameter names.")
}
samples <- sim$samples
fnames <- sim$fnames_oi
rows <- vector("list", length(samples))
for (chain in seq_along(samples)) {
  n_iter <- length(samples[[chain]][[1]])
  keep <- seq.int(sim$warmup2[chain] + 1L, n_iter)
  mat <- do.call(cbind, lapply(fnames, function(nm) samples[[chain]][[nm]][keep]))
  colnames(mat) <- fnames
  chain_df <- as.data.frame(mat, check.names = FALSE)
  chain_df$chain <- chain
  chain_df$draw <- seq_along(keep)
  rows[[chain]] <- chain_df[, c("chain", "draw", fnames)]
}

draws <- do.call(rbind, rows)
write.csv(draws, file.path(output_dir, "hbrl_all_stan_draws.csv"), row.names = FALSE)
metadata <- data.frame(
  fit_file = normalizePath(fit_path, winslash = "/"),
  subjects = fit_env$data$N,
  chains = length(samples),
  warmup = sim$warmup2[1],
  retained_draws_per_chain = nrow(rows[[1]]),
  total_retained_draws = nrow(draws),
  n_parameters = length(fnames)
)
write.csv(metadata, file.path(output_dir, "stan_all_draw_export_metadata.csv"), row.names = FALSE)
cat("Exported", length(fnames), "flat Stan parameters to:", output_dir, "\n")
