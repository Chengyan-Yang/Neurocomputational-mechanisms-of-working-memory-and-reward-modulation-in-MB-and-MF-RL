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
project_dir <- dirname(dirname(dirname(dirname(dirname(here)))))
# Generate simulated trial-level data for HBRL parameter recovery.
#
# Output datasets have the same column structure as EXP2 HBRL_code and can be
# refitted with parameter_recovery_fit.R.

suppressPackageStartupMessages({
  library(rstan)
  library(dplyr)
  library(readr)
})

`%||%` <- function(x, y) {
  if (length(x) == 0 || is.na(x) || identical(x, "")) y else x
}

analysis_dir <- file.path(here, "..", "..")
template_data_dir <- Sys.getenv("HBRL52_DATA_DIR", unset=file.path(project_dir, "data", "BHV", "2_stage_task", "preprocessed", "learning", "all", "HBRL_code"))
output_id <- "MEG_HBRL"
expected_N <- 52
recovery_label <- Sys.getenv("HBRL_RECOVERY_LABEL") %||% output_id

fit_path <- file.path(analysis_dir, "data_mfmblearning_magnitude&load.RData")
recovery_dir <- file.path(analysis_dir, "validation", "parameter_recovery", paste0("outputs_", recovery_label))
sim_root <- file.path(recovery_dir, "sim_data")
dir.create(sim_root, recursive = TRUE, showWarnings = FALSE)
if (file.exists(file.path(recovery_dir, "recovery_datasets.csv")) &&
    Sys.getenv("HBRL_RECOVERY_OVERWRITE") != "TRUE") {
  stop("Simulations already exist. Use the runner to resume fitting, or choose a new HBRL_RECOVERY_LABEL.")
}

n_datasets <- as.integer(Sys.getenv("HBRL_RECOVERY_N_DATASETS") %||% "1")
draw_ids_env <- Sys.getenv("HBRL_RECOVERY_DRAW_IDS")
set.seed(as.integer(Sys.getenv("HBRL_RECOVERY_SEED") %||% "260220"))

if (!file.exists(fit_path)) stop("Missing fit file: ", fit_path)
if (!dir.exists(template_data_dir)) stop("Missing template data folder: ", template_data_dir)

fit_env <- new.env()
load(fit_path, envir=fit_env)
fit <- fit_env$fit
if (is.null(fit)) stop("The RData file does not contain an object named 'fit'.")
if (is.null(fit_env$data) || fit_env$data$N != expected_N) {
  stop("The saved fit data must contain exactly ", expected_N, " subjects.")
}
fit_data_files <- fit_env$data_files

data_files <- sort(list.files(template_data_dir, pattern = "\\.csv$", full.names = TRUE))
if (length(data_files) != expected_N) {
  stop("Expected ", expected_N, " csv files, but found ", length(data_files), ".")
}
if (length(fit_data_files) != expected_N ||
    !identical(basename(fit_data_files), basename(data_files))) {
  stop("The 52-subject CSV order differs from the order saved in the fitted Stan data.")
}

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

n_available_draws <- nrow(posterior$alpha_hload)
if (nzchar(draw_ids_env)) {
  draw_ids <- as.integer(strsplit(draw_ids_env, ",")[[1]])
  if (any(is.na(draw_ids)) || any(draw_ids < 1) || any(draw_ids > n_available_draws)) {
    stop("HBRL_RECOVERY_DRAW_IDS must contain valid posterior draw indices.")
  }
} else {
  draw_ids <- sort(sample(seq_len(n_available_draws), n_datasets))
}
n_datasets <- length(draw_ids)

softmax2 <- function(q) {
  q <- q - max(q)
  exp(q) / sum(exp(q))
}

choice_from_action <- function(action, stimuli_left, stimuli_right) {
  left_action <- ifelse(as.integer(stimuli_left) %% 2L != 0L, 1L, 2L)
  ifelse(action == left_action, as.integer(stimuli_left), as.integer(stimuli_right))
}

animal_from_choice <- function(choice, load_code, part_code) {
  dplyr::case_when(
    load_code == 1L & choice %in% c(1L, 3L) ~ 1L,
    load_code == 1L & choice %in% c(2L, 4L) ~ 2L,
    load_code == 2L & part_code == 1L & choice %in% c(5L, 7L) ~ 3L,
    load_code == 2L & part_code == 1L & choice %in% c(6L, 8L) ~ 4L,
    load_code == 2L & part_code == 2L & choice %in% c(10L, 12L) ~ 5L,
    load_code == 2L & part_code == 2L & choice %in% c(9L, 11L) ~ 6L,
    TRUE ~ NA_integer_
  )
}

reward_from_choice <- function(choice, load_code, part_code, rews_1, rews_2, rews_3, rews_4) {
  dplyr::case_when(
    load_code == 1L & choice %in% c(1L, 3L) ~ rews_1,
    load_code == 1L & choice %in% c(2L, 4L) ~ rews_2,
    load_code == 2L & part_code == 1L & choice %in% c(5L, 7L) ~ rews_1,
    load_code == 2L & part_code == 1L & choice %in% c(6L, 8L) ~ rews_2,
    load_code == 2L & part_code == 2L & choice %in% c(10L, 12L) ~ rews_3,
    load_code == 2L & part_code == 2L & choice %in% c(9L, 11L) ~ rews_4,
    TRUE ~ NA_real_
  )
}

prepare_template <- function(path, subject_index) {
  raw_data <- readr::read_csv(path, show_col_types = FALSE) %>%
    filter(probe != 1) %>%
    mutate(
      subject_index = subject_index,
      subid = suppressWarnings(as.integer(first(na.omit(participant)))),
      s1 = as.integer(pairs),
      load_code = if_else(load == "low", 1L, 2L),
      magnitude_code = if_else(magnitude == "small", 1L, 5L),
      part_code = as.integer(part),
      display_left = if_else(as.integer(stimuli_left) %% 2L != 0L, 1L, 2L),
      rews_1 = as.numeric(rews_1),
      rews_2 = as.numeric(rews_2),
      rews_3 = as.numeric(rews_3),
      rews_4 = as.numeric(rews_4)
    )
  required <- c("participant", "pairs", "choice", "animal", "points", "load", "magnitude", "part",
                "stimuli_left", "stimuli_right", "rews_1", "rews_2", "rews_3", "rews_4")
  if (anyNA(raw_data[, required]) || any(raw_data$choice == 0)) {
    stop("Invalid template trials in ", path, "; preprocessing must match the fitted model.")
  }
  if (nrow(raw_data) != fit_env$data$nT[subject_index]) stop("Trial count differs from saved fit: ", path)
  raw_data
}

simulate_subject <- function(df, draw_id) {
  s <- unique(df$subject_index)
  n_trials <- nrow(df)
  sim_action <- integer(n_trials)
  sim_choice <- integer(n_trials)
  sim_s2 <- integer(n_trials)
  sim_reward <- numeric(n_trials)

  q2 <- rep(4.5, 6)
  qmf <- matrix(4.5, nrow = 6, ncol = 2)
  m_stick <- matrix(0, nrow = 6, ncol = 2)
  k_resp <- c(0, 0)
  lambda <- 1

  for (t in seq_len(n_trials)) {
    if (t == 1 || df$load_code[t] != df$load_code[t - 1]) {
      q2 <- rep(4.5, 6)
      qmf <- matrix(4.5, nrow = 6, ncol = 2)
      m_stick <- matrix(0, nrow = 6, ncol = 2)
      k_resp <- c(0, 0)
    }

    if (df$load_code[t] == 1L) {
      alpha <- posterior$alpha_lload[draw_id, s]
      st <- posterior$st_lload[draw_id, s]
      respst <- posterior$respst_lload[draw_id, s]
      if (df$magnitude_code[t] == 1L) {
        beta_mf <- posterior$beta_mf_small_lload[draw_id, s]
        beta_mb <- posterior$beta_mb_small_lload[draw_id, s]
      } else {
        beta_mf <- posterior$beta_mf_large_lload[draw_id, s]
        beta_mb <- posterior$beta_mb_large_lload[draw_id, s]
      }
    } else {
      alpha <- posterior$alpha_hload[draw_id, s]
      st <- posterior$st_hload[draw_id, s]
      respst <- posterior$respst_hload[draw_id, s]
      if (df$magnitude_code[t] == 1L) {
        beta_mf <- posterior$beta_mf_small_hload[draw_id, s]
        beta_mb <- posterior$beta_mb_small_hload[draw_id, s]
      } else {
        beta_mf <- posterior$beta_mf_large_hload[draw_id, s]
        beta_mb <- posterior$beta_mb_large_hload[draw_id, s]
      }
    }

    q2_subset <- if (df$s1[t] <= 2L) q2[1:2] else if (df$s1[t] <= 4L) q2[3:4] else q2[5:6]
    tm <- if (df$part_code[t] == 0L || df$part_code[t] == 1L) {
      matrix(c(1, 0, 0, 1), nrow = 2, byrow = TRUE)
    } else {
      matrix(c(0, 1, 1, 0), nrow = 2, byrow = TRUE)
    }
    qmb <- as.vector(tm %*% q2_subset)

    k_for_choice <- k_resp
    if (df$display_left[t] == 2L) k_for_choice <- rev(k_for_choice)

    q <- beta_mf * qmf[df$s1[t], ] +
      beta_mb * qmb +
      st * m_stick[df$s1[t], ] +
      respst * k_for_choice

    p <- softmax2(q)
    sim_action[t] <- rbinom(1, size = 1, prob = p[2]) + 1L
    sim_choice[t] <- choice_from_action(sim_action[t], df$stimuli_left[t], df$stimuli_right[t])
    sim_s2[t] <- animal_from_choice(sim_choice[t], df$load_code[t], df$part_code[t])
    sim_reward[t] <- reward_from_choice(
      sim_choice[t], df$load_code[t], df$part_code[t],
      df$rews_1[t], df$rews_2[t], df$rews_3[t], df$rews_4[t]
    )

    if (is.na(sim_s2[t]) || is.na(sim_reward[t])) {
      stop("Could not map simulated choice to animal/reward for subject ", s, ", trial ", t)
    }

    m_stick[df$s1[t], ] <- 0
    m_stick[df$s1[t], sim_action[t]] <- 1

    k_resp <- c(0, 0)
    if (sim_action[t] == df$display_left[t]) k_resp[1] <- 1 else k_resp[2] <- 1

    delta1 <- q2[sim_s2[t]] - qmf[df$s1[t], sim_action[t]]
    qmf[df$s1[t], sim_action[t]] <- qmf[df$s1[t], sim_action[t]] + alpha * delta1
    delta2 <- sim_reward[t] - q2[sim_s2[t]]
    q2[sim_s2[t]] <- q2[sim_s2[t]] + alpha * delta2
    qmf[df$s1[t], sim_action[t]] <- qmf[df$s1[t], sim_action[t]] + lambda * alpha * delta2
  }

  df %>%
    mutate(
      choice = sim_choice,
      action_index = sim_action,
      animal = sim_s2,
      points = sim_reward,
      correct = as.integer(sim_reward == as.numeric(max_rews))
    ) %>%
    select(-subject_index, -subid, -s1, -load_code, -magnitude_code, -part_code, -display_left)
}

generating_parameters <- function(draw_id, dataset_id) {
  subids <- vapply(template_data, function(x) unique(x$subid), integer(1))
  get_subject_vector <- function(x) {
    out <- vapply(seq_along(subids), function(s) as.numeric(x[draw_id, s]), numeric(1))
    if (length(out) != length(subids)) stop("Subject-level parameter extraction failed.")
    out
  }
  get_effect_vector <- function(x, effect_index) {
    out <- vapply(seq_along(subids), function(s) as.numeric(x[draw_id, effect_index, s]), numeric(1))
    if (length(out) != length(subids)) stop("Effect-level parameter extraction failed.")
    out
  }

  tibble(
    dataset_id = dataset_id,
    draw_id = draw_id,
    subject_index = seq_len(length(subids)),
    subid = subids,
    alpha_hload = get_subject_vector(posterior$alpha_hload),
    alpha_lload = get_subject_vector(posterior$alpha_lload),
    beta_mf_b0 = get_effect_vector(posterior$beta_mf_effects, 1),
    beta_mf_magnitude = get_effect_vector(posterior$beta_mf_effects, 2),
    beta_mf_load = get_effect_vector(posterior$beta_mf_effects, 3),
    beta_mf_magnitude_load = get_effect_vector(posterior$beta_mf_effects, 4),
    beta_mb_b0 = get_effect_vector(posterior$beta_mb_effects, 1),
    beta_mb_magnitude = get_effect_vector(posterior$beta_mb_effects, 2),
    beta_mb_load = get_effect_vector(posterior$beta_mb_effects, 3),
    beta_mb_magnitude_load = get_effect_vector(posterior$beta_mb_effects, 4),
    beta_mf_small_lload = get_subject_vector(posterior$beta_mf_small_lload),
    beta_mf_large_lload = get_subject_vector(posterior$beta_mf_large_lload),
    beta_mf_small_hload = get_subject_vector(posterior$beta_mf_small_hload),
    beta_mf_large_hload = get_subject_vector(posterior$beta_mf_large_hload),
    beta_mb_small_lload = get_subject_vector(posterior$beta_mb_small_lload),
    beta_mb_large_lload = get_subject_vector(posterior$beta_mb_large_lload),
    beta_mb_small_hload = get_subject_vector(posterior$beta_mb_small_hload),
    beta_mb_large_hload = get_subject_vector(posterior$beta_mb_large_hload),
    st_hload = get_subject_vector(posterior$st_hload),
    st_lload = get_subject_vector(posterior$st_lload),
    respst_hload = get_subject_vector(posterior$respst_hload),
    respst_lload = get_subject_vector(posterior$respst_lload)
  )
}

template_data <- lapply(seq_along(data_files), function(i) prepare_template(data_files[i], i))
combined <- bind_rows(template_data)
current_input <- list(
  s1=combined$s1, s2=combined$animal,
  a=ifelse(combined$choice %% 2 != 0, 1, 2), r=combined$points,
  magnitude=combined$magnitude_code, load=combined$load_code, part=combined$part_code,
  display=rbind(combined$display_left, ifelse(combined$stimuli_right %% 2 != 0, 1, 2))
)
for (nm in names(current_input)) {
  if (!isTRUE(all.equal(unname(current_input[[nm]]), unname(fit_env$data[[nm]]), check.attributes=FALSE))) {
    stop("Template data differ from fitted Stan input for field: ", nm)
  }
}
all_generating <- vector("list", n_datasets)

for (dataset_i in seq_along(draw_ids)) {
  dataset_id <- sprintf("recovery_%03d", dataset_i)
  draw_id <- draw_ids[dataset_i]
  dataset_dir <- file.path(sim_root, dataset_id, "HBRL_code")
  dir.create(dataset_dir, recursive = TRUE, showWarnings = FALSE)

  message("Generating ", dataset_id, " from posterior draw ", draw_id)
  for (s in seq_along(template_data)) {
    sim_df <- simulate_subject(template_data[[s]], draw_id)
    subid <- unique(template_data[[s]]$subid)
    out_file <- file.path(dataset_dir, sprintf("%03d_RL_HBRLprep_recovery.csv", subid))
    readr::write_csv(sim_df, out_file)
  }

  all_generating[[dataset_i]] <- generating_parameters(draw_id, dataset_id)
}

generating_df <- bind_rows(all_generating)
readr::write_csv(generating_df, file.path(recovery_dir, "generating_parameters_long.csv"))
readr::write_csv(
  tibble(
    dataset_id = sprintf("recovery_%03d", seq_along(draw_ids)),
    draw_id = draw_ids,
    data_dir = file.path(sim_root, sprintf("recovery_%03d", seq_along(draw_ids)), "HBRL_code")
  ),
  file.path(recovery_dir, "recovery_datasets.csv")
)

message("Generated recovery datasets in: ", sim_root)
message("Generating parameters saved to: ", file.path(recovery_dir, "generating_parameters_long.csv"))



