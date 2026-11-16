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
# Separate single-mediator MEG models with original load coding: high minus low.
suppressWarnings(Sys.setlocale("LC_CTYPE", "English_United States.utf8"))
suppressPackageStartupMessages(library(lme4))
dataset <- "MEG"
variant <- "load_original"
analysis_dir <- file.path(dirname(here), "results", "mediation")
source_dir <- analysis_dir
dest <- analysis_dir
out <- file.path(dest, "outputs")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
suffix <- ""
filename <- function(stem) paste0(stem, suffix, ".csv")
read_original <- function(stem) read.csv(file.path(source_dir, "outputs", filename(stem)), check.names = FALSE)
write_output <- function(value, stem) write.csv(value, file.path(out, filename(stem)), row.names = FALSE, na = "")
dat <- read_original("prepared_mediation_data")
reward <- "magnitude"
reward_c <- paste0(reward, "_c")
dat$subid <- factor(dat$subid)
stopifnot(!anyNA(dat), all(dat$load %in% c("low", "high")), all(dat[[reward]] %in% c("small", "large")))
stopifnot(all(table(dat$subid, interaction(dat$load, dat[[reward]])) == 1))
dat$load_c <- ifelse(dat$load == "high", 0.5, -0.5)
stopifnot(all(dat[[reward_c]] == ifelse(dat[[reward]] == "large", 0.5, -0.5)))
mediators <- c("beta_mb", "beta_mf", "w")
# Keep only the variables used in these analyses, including one mediator at a time in each outcome model.
dat <- dat[, c("subid", "load", reward, "cr", "load_c", reward_c, mediators,
               paste0(mediators, "_within"), paste0(mediators, "_between"))]
labels <- c(beta_mb = "MB weight", beta_mf = "MF weight", w = "Mixing weight", beta = "Inverse temperature", alpha = "Learning rate")
for (m in mediators) {
  expected_mean <- ave(dat[[m]], dat$subid, FUN = mean)
  stopifnot(max(abs(expected_mean - dat[[paste0(m, "_between")]])) < 1e-10)
  stopifnot(max(abs(dat[[m]] - expected_mean - dat[[paste0(m, "_within")]])) < 1e-10)
}
write_output(dat, "prepared_mediation_data")
x <- "load_c"
moderator <- "magnitude"
wc <- "magnitude_c"
w_values <- c(small = -0.5, large = 0.5)
a_col <- "a_load_to_mediator"
effect_cols <- c(a_col, "b_mediator_to_cr", "indirect_ab", "direct_cprime", "total_c", "prop_mediated")
coef_required <- function(fe, name) {
  if (!name %in% names(fe)) stop("Missing coefficient: ", name)
  unname(fe[[name]])
}
interaction_coef <- function(fe, a, b) {
  candidates <- c(paste(a, b, sep = ":"), paste(b, a, sep = ":"))
  name <- candidates[candidates %in% names(fe)]
  if (length(name) != 1) stop("Missing interaction: ", a, " / ", b)
  unname(fe[[name]])
}
fit_models <- function(data, id = "subid") {
  base <- paste(x, "*", wc)
  re <- paste0("(1 | ", id, ")")
  fit <- function(lhs, terms) lmer(as.formula(paste(lhs, "~", terms, "+", re)), data = data,
    REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
  ms <- setNames(lapply(mediators, function(m) fit(m, base)), mediators)
  ys <- setNames(lapply(mediators, function(m) {
    terms <- paste(c(base, paste0(m, "_within * ", wc), paste0(m, "_between")), collapse = " + ")
    fit("cr", terms)
  }), mediators)
  total <- fit("cr", base)
  ft <- fixef(total)
  rows <- list()
  for (m in mediators) {
    fm <- fixef(ms[[m]])
    fy <- fixef(ys[[m]])
    for (level in names(w_values)) {
      value <- w_values[[level]]
      a <- coef_required(fm, x) + value * interaction_coef(fm, x, wc)
      b <- coef_required(fy, paste0(m, "_within")) + value * interaction_coef(fy, paste0(m, "_within"), wc)
      direct <- coef_required(fy, x) + value * interaction_coef(fy, x, wc)
      tot <- coef_required(ft, x) + value * interaction_coef(ft, x, wc)
      row <- data.frame(mediator = m, mediator_label = labels[[m]], condition = level,
        condition_label = paste(level, "reward"),
        a = a, b_mediator_to_cr = b, indirect_ab = a*b, direct_cprime = direct,
        total_c = tot, prop_mediated = if (abs(tot) > .Machine$double.eps) a*b/tot else NA_real_)
      names(row)[3:5] <- c(moderator, paste0(moderator, "_label"), a_col)
      rows[[length(rows)+1]] <- row
    }
  }
  models <- c(ms, setNames(ys, paste0("outcome_", mediators)), list(total_cr = total))
  # Assert that every outcome includes only its own mediator.
  for (m in mediators) {
    vars <- all.vars(formula(ys[[m]]))
    other <- setdiff(mediators, m)
    stopifnot(!any(c(paste0(other, "_within"), paste0(other, "_between")) %in% vars))
  }
  diagnostics <- do.call(rbind, lapply(names(models), function(nm) {
    model <- models[[nm]]
    msg <- model@optinfo$conv$lme4$messages
    data.frame(model = nm, singular = isSingular(model), convergence_message = paste(msg, collapse = "; "))
  }))
  list(effects = do.call(rbind, rows), models = models, diagnostics = diagnostics)
}
point <- fit_models(dat)
write_output(point$effects, "moderated_mediation_point_estimates")
write_output(point$diagnostics, "model_diagnostics")
fixed <- do.call(rbind, lapply(names(point$models), function(nm) {
  fe <- fixef(point$models[[nm]])
  data.frame(model = nm, term = names(fe), estimate = unname(fe))
}))
write_output(fixed, "fixed_effects")
summary_lines <- c(paste("Dataset:", dataset), paste("Variant:", variant),
  "Load coding: low = -0.5, high = +0.5", "Reward coding: small = -0.5, large = +0.5",
  paste("Predictor:", x), paste("Moderator:", wc),
  "Three SEPARATE single-mediator models, not a parallel multiple-mediator model.",
  "Each outcome includes only the selected mediator within-participant deviation and its between-participant mean.",
  "Both a and b paths may be moderated. Conditional indirect effects are a(W)*b(W).",
  unlist(lapply(names(point$models), function(nm) c(paste("---", nm, "---"), capture.output(summary(point$models[[nm]]))))))
writeLines(summary_lines, file.path(out, paste0("model_summaries", suffix, ".txt")))
seed <- 260909L
n_boot <- as.integer(Sys.getenv("SINGLE_MEDIATION_N_BOOT", "2000"))
failed <- 0L
{
  set.seed(seed)
  ids <- unique(dat$subid)
  groups <- lapply(ids, function(id) which(dat$subid == id))
  results <- vector("list", n_boot)
  diagnostics <- vector("list", n_boot)
  for (i in seq_len(n_boot)) {
    sampled <- sample(seq_along(ids), length(ids), replace = TRUE)
    bd <- dat[unlist(groups[sampled]), ]
    bd$subid_boot <- factor(rep(seq_along(sampled), lengths(groups[sampled])))
    fit <- tryCatch(suppressMessages(suppressWarnings(fit_models(bd, "subid_boot"))), error = function(e) e)
    if (inherits(fit, "error")) {
      failed <- failed + 1L
      diagnostics[[i]] <- data.frame(model = "failed", singular = NA, convergence_message = conditionMessage(fit), boot_id = i)
    } else {
      fit$effects$boot_id <- i
      results[[i]] <- fit$effects
      fit$diagnostics$boot_id <- i
      diagnostics[[i]] <- fit$diagnostics
    }
    if (i %% 100 == 0) message(dataset, " ", variant, " bootstrap ", i, "/", n_boot)
  }
  boot <- do.call(rbind, results)
  write_output(do.call(rbind, diagnostics), "bootstrap_diagnostics")
  method <- paste("New subject-cluster bootstrap; resamples:", n_boot, "seed:", seed)
}
write_output(boot, "bootstrap_effects_long")
bootstrap_p <- function(v) {
  v <- v[is.finite(v)]
  if (!length(v)) return(NA_real_)
  min(1, 2 * min(mean(v >= 0), mean(v <= 0)))
}
prefixes <- c("a", "b", "indirect", "direct", "total", "prop_mediated")
summary_rows <- lapply(seq_len(nrow(point$effects)), function(i) {
  row <- point$effects[i, ]
  bb <- boot[boot$mediator == row$mediator & boot[[moderator]] == row[[moderator]], ]
  row$n_boot_ok <- nrow(bb)
  for (j in seq_along(effect_cols)) {
    v <- bb[[effect_cols[j]]]
    pre <- prefixes[j]
    row[[paste0(pre, "_mean")]] <- mean(v, na.rm = TRUE)
    row[[paste0(pre, "_ci_low")]] <- unname(quantile(v, .025, na.rm = TRUE))
    row[[paste0(pre, "_ci_high")]] <- unname(quantile(v, .975, na.rm = TRUE))
    row[[paste0(pre, "_p_boot")]] <- bootstrap_p(v)
  }
  row$indirect_significant <- row$indirect_ci_low > 0 | row$indirect_ci_high < 0
  row$bootstrap_failed <- failed
  row
})
write_output(do.call(rbind, summary_rows), "moderated_mediation_bootstrap_summary")
writeLines(c(method, paste("Failed resamples:", failed), capture.output(sessionInfo())), file.path(out, "run_provenance.txt"))

# Paired bootstrap contrasts: actual large reward minus small reward.
# Because both paths can be moderated, report this contrast rather than a single linear index.
contrast_rows <- list()
for (m in mediators) {
  pp <- point$effects[point$effects$mediator == m, ]
  small <- boot[boot$mediator == m & boot[[moderator]] == "small", ]
  large <- boot[boot$mediator == m & boot[[moderator]] == "large", ]
  large <- large[match(small$boot_id, large$boot_id), ]
  stopifnot(identical(small$boot_id, large$boot_id))
  for (j in seq_along(effect_cols)) {
    col <- effect_cols[j]
    delta <- large[[col]] - small[[col]]
    contrast_rows[[length(contrast_rows) + 1]] <- data.frame(
      mediator = m, moderator = moderator, contrast = "large - small", effect = prefixes[j],
      estimate = pp[pp[[moderator]] == "large", col] - pp[pp[[moderator]] == "small", col],
      bootstrap_mean = mean(delta, na.rm = TRUE),
      ci_low = unname(quantile(delta, .025, na.rm = TRUE)),
      ci_high = unname(quantile(delta, .975, na.rm = TRUE)),
      p_boot = bootstrap_p(delta), n_boot_ok = sum(is.finite(delta)))
  }
}
write_output(do.call(rbind, contrast_rows), "moderation_contrasts")
message("COMPLETE: ", dest)
