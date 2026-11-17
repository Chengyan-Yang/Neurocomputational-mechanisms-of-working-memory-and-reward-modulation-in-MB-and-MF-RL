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
project_dir <- dirname(dirname(dirname(here)))
# HBRL model for random design task ============================================

library(dplyr)
library(abind)

# Input sample and output paths.
output_id <- "MEG"
expected_N <- 52
run_model_fitting <- !tolower(Sys.getenv("HBRL_RUN_FIT", unset = "TRUE")) %in% c("false", "0", "no")
analysis_dir <- here
figure_dir <- file.path(analysis_dir, "Figures")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
stan_file <- file.path(analysis_dir, "mfmblearning_magnitude&load.stan")
fit_output_file <- file.path(analysis_dir, "data_mfmblearning_magnitude&load.RData")

# work path of preprocessed RL data
folder_path <- file.path(project_dir, "data", "BHV", "2_stage_task", "preprocessed", "learning", "all", "HBRL_code")
data_files <- list.files(path = folder_path, pattern = "\\.csv$", full.names = TRUE)

# get relevant data
N <- length(data_files) # number of participants
message("Input folder: ", folder_path)
message("Number of participant files: ", N)
if (!is.na(expected_N) && N != expected_N) {
  stop("Expected ", expected_N, " participants, but found ", N, " csv files in folder_path.")
}

# data extracting: based on mfmblearning_effects model fitting(Bolenz, 2022), update by Chengyan Yang, 2025
subject_ids <- integer(N)
for (i in seq_along(data_files)) {
  raw_data <- read.csv(data_files[i], header = TRUE) 
  id <- unique(na.omit(raw_data$participant))
  file_id <- as.integer(sub("_.*$", "", basename(data_files[i])))
  if (length(id) != 1L || !id %in% seq_len(expected_N) || !identical(as.integer(id), file_id)) {
    stop("Invalid participant ID or filename mismatch in ", data_files[i])
  }
  subject_ids[i] <- id
  raw_data <- raw_data[raw_data$probe != 1, ]
  
  nT = length(raw_data$participant) # number of trials per participant
  s1 <- as.numeric(raw_data$pairs) # first-stage states
  s2 <- as.numeric(raw_data$animal) # second-stage states
  a <- ifelse(raw_data$choice %% 2 != 0, 1, 2) # actions, (1,3,5,7,9,11 -> 1),(2,4,6,8,10,12 -> 2), The location for each action is shown in "display".
  r <- raw_data$points # reward
  magnitude <- ifelse(raw_data$magnitude == 'small', 1, 5) # magnitude condition 1 = small reward, 5 = large reward
  load <- ifelse(raw_data$load == 'low', 1, 2) # load condition 1 = low load, 2 = high load
  part <- raw_data$part # part
  display <- with(raw_data, cbind(ifelse(stimuli_left %% 2 != 0, 1, 2), ifelse(stimuli_right %% 2 != 0, 1, 2)))# left-1, right-2 or left-2, right-1

  # store data in list
  if(i==1){
    data <- list(N=N, nT=nT, s1=s1, s2=s2, a=a, r=r, magnitude=magnitude, load=load, part=part, display=t(display))
  } else{
    data <- list(N=N, nT=c(data$nT,nT), s1=c(data$s1,s1), s2=c(data$s2,s2), a=c(data$a,a),
                 r=c(data$r,r), magnitude=c(data$magnitude,magnitude), load=c(data$load,load), part=c(data$part,part),
                 display=cbind(data$display, t(display)))
  }
} 

stopifnot(setequal(subject_ids, seq_len(expected_N)))

library(rstan)

# define which model parameters to save
pars <- c("mu_alpha_hload_logit", "sd_alpha_hload_logit",
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
          "st_hload_z", "st_lload_z", "respst_hload_z", "respst_lload_z")
# ,"log_lik")


if (run_model_fitting) {
# fit model in Stan
fit <- stan(file=stan_file,
            data=data,
            pars=pars,
            iter=4000, # number of iterations
            warmup=2000, # number of warmup iterations
            # init=0, # initialize all parameters to 0
            chains=4,
            verbose = FALSE,
            refresh = 10 ,# Print progress every 10 iterations.
            cores=4,
            init_r = 0.5,  # Initial sampling radius.
            seed = 260220 # random seed
)

# save results
save(list=c("fit", "data", "subject_ids", "data_files", "folder_path", "output_id", "expected_N"), file=fit_output_file)
} else {
  message("Skipping Stan fitting and loading existing fit file: ", fit_output_file)
}

##############
# Start from here, if you do not want to reproduce model-fitting

# load packages
library(rstan) # 2.19.2
library(dplyr) # 0.7.6
library(ggplot2) # 3.0.0
library(loo)

# load workspace
load(fit_output_file)

## MCMC diganostics
# check model convergence, alls R-hats should be < 1.1
mon <- monitor(fit, print=F)
max(mon[,"Rhat"])

# check effective sample size
min(mon[,"Bulk_ESS"]) # bulk ess should be > 400
min(mon[,"Tail_ESS"]) # tail ess should be > 400

# check for divergent transitions
sampler_params <- get_sampler_params(fit, inc_warmup = FALSE)
sapply(sampler_params, function(x) sum(x[, "divergent__"]))


## Analyse results
# load function
source(file.path(analysis_dir, "samplesSummary.R"))

# extract samples from stanfit object
samples <- extract(fit)

# get log likelihodd
# logLik <- extract_log_lik(fit, merge_chains = FALSE)
# saveRDS(logLik, file="logLik_mfmb.rds")

# effects for model-based weights 
samplesSummary(samples$mu_beta_mb_effects[,1]) # mean model-based weight
samplesSummary(samples$mu_beta_mb_effects[,2]) # reward effect
samplesSummary(samples$mu_beta_mb_effects[,3]) # load effect
samplesSummary(samples$mu_beta_mb_effects[,4]) # reward x load interaction

samplesSummary(samples$mu_beta_mb_effects[,3] -0.5*samples$mu_beta_mb_effects[,4]) # load effect, separately for magnitude conditions: small reward
samplesSummary(samples$mu_beta_mb_effects[,3] +0.5*samples$mu_beta_mb_effects[,4]) # load effect, separately for magnitude conditions: large reward

samplesSummary(samples$Omega_mb[,1,2]) # correlation between mean model-based weight and reward effect


# effects for model-free weights
samplesSummary(samples$mu_beta_mf_effects[,1]) # mean model-free weight
samplesSummary(samples$mu_beta_mf_effects[,2]) # reward effect
samplesSummary(samples$mu_beta_mf_effects[,3]) # load effect
samplesSummary(samples$mu_beta_mf_effects[,4]) # reward x load interaction

samplesSummary(samples$mu_beta_mf_effects[,3] -0.5*samples$mu_beta_mf_effects[,4]) # load effect, separately for magnitude conditions: small reward
samplesSummary(samples$mu_beta_mf_effects[,3] +0.5*samples$mu_beta_mf_effects[,4]) # load effect, separately for magnitude conditions: large reward

samplesSummary(samples$Omega_mf[,1,2]) # correlation between mean model-free weight and reward effect


# compare effects for model-free and model-based weights
samplesSummary(samples$mu_beta_mb_effects[,1]-samples$mu_beta_mf_effects[,1]) # mean weight
samplesSummary(samples$mu_beta_mb_effects[,2]-samples$mu_beta_mf_effects[,2]) # reward effect
samplesSummary(samples$mu_beta_mb_effects[,3]-samples$mu_beta_mf_effects[,3]) # load effect
samplesSummary(samples$mu_beta_mb_effects[,4]-samples$mu_beta_mf_effects[,4]) # reward x load interaction

## all other parameters
samplesSummary(exp(samples$mu_alpha_hload_logit)/(1+exp(samples$mu_alpha_hload_logit))) # alpha high load
samplesSummary(exp(samples$mu_alpha_lload_logit)/(1+exp(samples$mu_alpha_lload_logit))) # alpha low load
samplesSummary(samples$mu_st_hload) # choice stickiness high load
samplesSummary(samples$mu_st_lload) # choice stickiness low load
samplesSummary(samples$mu_respst_hload) # response stickiness high load
samplesSummary(samples$mu_respst_lload) # response stickiness low load

