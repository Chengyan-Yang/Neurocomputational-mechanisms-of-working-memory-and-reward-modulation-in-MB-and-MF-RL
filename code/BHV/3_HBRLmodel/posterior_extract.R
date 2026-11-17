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
# Participant-level and group posterior summaries

# install.packages('rstan')
library(rstan)
library(dplyr)
library(purrr)
library(tidyr)
library(bayestestR) 
library(ggplot2)

dir.create(file.path(here, "Figures"), recursive = TRUE, showWarnings = FALSE)
load(file.path(here, "data_mfmblearning_magnitude&load.RData"))


# Median =======================================================================
get_median_hdi <- function(fit, param_name) {
  samples <- rstan::extract(fit, pars = param_name)[[1]]
  n_subjects <- ncol(samples)
  stopifnot(length(subject_ids) == n_subjects, !anyDuplicated(subject_ids))
  stats_df <- map_dfr(seq_len(n_subjects), function(i) {
    subj_samples <- samples[, i]
    med_val <- median(subj_samples)
    hdi_val <- hdi(subj_samples, ci = 0.95)
    data.frame(
      subject_id = subject_ids[i],
      parameter = param_name,
      median = med_val,
      hdi_low = hdi_val$CI_low,
      hdi_high = hdi_val$CI_high,
      stringsAsFactors = FALSE
    )
  })
  return(stats_df)
}

# -----------------------------------------------------------
# Median of Model-Free effect
# -----------------------------------------------------------
params_mf <- c("beta_mf_small_lload", "beta_mf_large_lload",
               "beta_mf_small_hload", "beta_mf_large_hload")

mf_data <- map_dfr(params_mf, ~ get_median_hdi(fit, .x)) %>%
  pivot_wider(
    names_from = parameter, 
    values_from = c(median, hdi_low, hdi_high),
    names_glue = "{parameter}_{.value}" 
  )

write.csv(mf_data, file.path(here, "mf_estimates_hdi.csv"), row.names = FALSE)

# -----------------------------------------------------------
# Median of Model-Based effect
# -----------------------------------------------------------
params_mb <- c("beta_mb_small_lload", "beta_mb_large_lload",
               "beta_mb_small_hload", "beta_mb_large_hload")

mb_data <- map_dfr(params_mb, ~ get_median_hdi(fit, .x)) %>%
  pivot_wider(
    names_from = parameter, 
    values_from = c(median, hdi_low, hdi_high),
    names_glue = "{parameter}_{.value}"
  )

write.csv(mb_data, file.path(here, "mb_estimates_hdi.csv"), row.names = FALSE)


# Posterior ====================================================================
get_combined_posterior <- function(fit, param_name, combined_label) {
  samples_mat <- rstan::extract(fit, pars = param_name)[[1]]
  group_means <- rowMeans(samples_mat)
  data.frame(
    value = group_means,
    Condition = combined_label,
    stringsAsFactors = FALSE
  )}

# -----------------------------------------------------------
# Posterior of Model-Free effect - beta_mf_[Magnitude]_[Load]
# -----------------------------------------------------------
d_lowload_small <- get_combined_posterior(fit, "beta_mf_small_lload",  "Low Load - small reward")
d_lowload_large <- get_combined_posterior(fit, "beta_mf_large_lload", "Low Load - large reward")
d_highload_small <- get_combined_posterior(fit, "beta_mf_small_hload",  "High Load - small reward")
d_highload_large <- get_combined_posterior(fit, "beta_mf_large_hload", "High Load - large reward")

plot_data_combined <- bind_rows(d_lowload_small, d_lowload_large, d_highload_small, d_highload_large)

plot_data_combined$Condition <- factor(plot_data_combined$Condition, levels = c(
  "Low Load - small reward", 
  "Low Load - large reward",
  "High Load - small reward", 
  "High Load - large reward"
))

mf <- ggplot(plot_data_combined, aes(x = value, fill = Condition, color = Condition)) +
  geom_density(alpha = 0.6, size = 0.8) +
  scale_fill_manual(values = c(
    "Low Load - small reward"  = "#a9c19c",
    "Low Load - large reward" = "#346db1",
    "High Load - small reward" = "#eacdd2",
    "High Load - large reward"= "#b03949" 
  ), guide = "none") +
  scale_color_manual(values = c(
    "Low Load - small reward"  = "#a9c19c",
    "Low Load - large reward" = "#346db1",
    "High Load - small reward" = "#eacdd2",
    "High Load - large reward"= "#b03949" 
  ), guide = "none") +
  
  
  labs(
    # title = "Posterior Distributions of MF effect",
    # subtitle = "Comparison across Load x Magnitude conditions",
    x = expression("Model-Free effect (" * italic(beta)[plain(MF)] * ")"),
    y = "Posterior Density",
    # fill = "Condition",
    # color = "Condition"
  ) +
  
  theme_classic(base_size = 14) +
  theme(
    # legend.position = "right",
    # legend.title = element_text(face = "bold"),
    axis.text = element_text(color = "black"),
    axis.line = element_line(linewidth = 1.2, color = "black"),
    axis.ticks = element_line(linewidth = 0.8, color = "black"),
    # Use a transparent plot background.
    plot.background = element_rect(fill = "transparent", color = NA),
    # Use a transparent panel background.
    panel.background = element_rect(fill = "transparent", color = NA)
  )

ggsave(file.path(here, "Figures", "posterior_mf.pdf"), 
       plot = mf,
       device = grDevices::cairo_pdf, fallback_resolution = 600,
       dpi = 600,
       width = 4, 
       height = 3,
       units = "in")

# -----------------------------------------------------------
# Posterior of Model-Based effect - beta_mb_[Magnitude]_[Load]
# -----------------------------------------------------------
d_lowload_small <- get_combined_posterior(fit, "beta_mb_small_lload",  "Low Load - small reward")
d_lowload_large <- get_combined_posterior(fit, "beta_mb_large_lload", "Low Load - large reward")
d_highload_small <- get_combined_posterior(fit, "beta_mb_small_hload",  "High Load - small reward")
d_highload_large <- get_combined_posterior(fit, "beta_mb_large_hload", "High Load - large reward")

plot_data_combined <- bind_rows(d_lowload_small, d_lowload_large, d_highload_small, d_highload_large)

plot_data_combined$Condition <- factor(plot_data_combined$Condition, levels = c(
  "Low Load - small reward", 
  "Low Load - large reward",
  "High Load - small reward", 
  "High Load - large reward"
))

mb <- ggplot(plot_data_combined, aes(x = value, fill = Condition, color = Condition)) +
  geom_density(alpha = 0.6, size = 0.8) +
  scale_fill_manual(values = c(
    "Low Load - small reward"  = "#a9c19c",
    "Low Load - large reward" = "#346db1",
    "High Load - small reward" = "#eacdd2",
    "High Load - large reward"= "#b03949" 
  ), guide = "none") +
  scale_color_manual(values = c(
    "Low Load - small reward"  = "#a9c19c",
    "Low Load - large reward" = "#346db1",
    "High Load - small reward" = "#eacdd2",
    "High Load - large reward"= "#b03949" 
  ), guide = "none") +
  
  
  labs(
    # title = "Posterior Distributions of MB effect",
    # subtitle = "Comparison across Load x Magnitude conditions",
    x = expression("Model-Based effect (" * italic(beta)[plain(MB)] * ")"),
    y = "Posterior Density",
    # fill = "Condition", 
    # color = "Condition"
  ) +
  
  theme_classic(base_size = 14) +
  theme(
    # legend.position = "right",
    # legend.title = element_text(face = "bold"),
    axis.text = element_text(color = "black"),
    axis.line = element_line(linewidth = 1.2, color = "black"),
    axis.ticks = element_line(linewidth = 0.8, color = "black"),
    # Use a transparent plot background.
    plot.background = element_rect(fill = "transparent", color = NA),
    # Use a transparent panel background.
    panel.background = element_rect(fill = "transparent", color = NA)
  )

ggsave(file.path(here, "Figures", "posterior_mb.pdf"), 
       plot = mb,
       device = grDevices::cairo_pdf, fallback_resolution = 600,
       dpi = 600,
       width = 4, 
       height = 3,
       units = "in")



# ============================================================
# Bayesian posterior contrasts:
# WM load, reward, and load x reward effects
# ============================================================

get_group_posterior <- function(fit, param_name) {
  samples_mat <- rstan::extract(fit, pars = param_name)[[1]]
  rowMeans(samples_mat)
}

summarise_posterior_effect <- function(samples, parameter, effect) {
  hdi_val <- bayestestR::hdi(samples, ci = 0.95)
  
  data.frame(
    Parameter = parameter,
    Effect = effect,
    Median = median(samples),
    Mean = mean(samples),
    HDI_low = hdi_val$CI_low,
    HDI_high = hdi_val$CI_high,
    P_gt_0 = mean(samples > 0),
    P_lt_0 = mean(samples < 0),
    Evidence = ifelse(
      hdi_val$CI_low > 0 | hdi_val$CI_high < 0,
      "95% HDI excludes 0",
      "95% HDI includes 0"
    )
  )
}

test_load_magnitude_interaction <- function(fit, prefix, parameter_label) {
  LS <- get_group_posterior(fit, paste0(prefix, "_small_lload"))
  LL <- get_group_posterior(fit, paste0(prefix, "_large_lload"))
  HS <- get_group_posterior(fit, paste0(prefix, "_small_hload"))
  HL <- get_group_posterior(fit, paste0(prefix, "_large_hload"))
  
  load_effect <- ((HS + HL) / 2) - ((LS + LL) / 2)
  
  reward_effect <- ((LL + HL) / 2) - ((LS + HS) / 2)
  
  interaction_effect <- (HL - HS) - (LL - LS)
  
  bind_rows(
    summarise_posterior_effect(
      load_effect,
      parameter_label,
      "WM load main effect: high load - low load"
    ),
    summarise_posterior_effect(
      reward_effect,
      parameter_label,
      "Reward main effect: large reward - small reward"
    ),
    summarise_posterior_effect(
      interaction_effect,
      parameter_label,
      "WM load x reward interaction"
    )
  )
}

mf_effect_tests <- test_load_magnitude_interaction(
  fit = fit,
  prefix = "beta_mf",
  parameter_label = "beta_mf"
)

mb_effect_tests <- test_load_magnitude_interaction(
  fit = fit,
  prefix = "beta_mb",
  parameter_label = "beta_mb"
)

effect_tests <- bind_rows(mf_effect_tests, mb_effect_tests)

print(effect_tests)

write.csv(
  effect_tests,
  file.path(here, "load_magnitude_interaction_posterior_tests.csv"),
  row.names = FALSE
)

# ============================================================
# Posterior density visualization of contrasts
# ============================================================

posterior_effect_colors <- c(
  "WM load" = "#9f9f9f",
  "Reward" = "#9f9f9f",
  "WM load x Reward" = "#9f9f9f"
)

get_effect_posterior_draws <- function(fit, prefix, parameter_label) {
  LS <- get_group_posterior(fit, paste0(prefix, "_small_lload"))
  LL <- get_group_posterior(fit, paste0(prefix, "_large_lload"))
  HS <- get_group_posterior(fit, paste0(prefix, "_small_hload"))
  HL <- get_group_posterior(fit, paste0(prefix, "_large_hload"))
  
  load_effect <- ((HS + HL) / 2) - ((LS + LL) / 2)
  reward_effect <- ((LL + HL) / 2) - ((LS + HS) / 2)
  interaction_effect <- (HL - HS) - (LL - LS)
  
  bind_rows(
    data.frame(
      Parameter = parameter_label,
      Effect_short = "WM load",
      value = load_effect
    ),
    data.frame(
      Parameter = parameter_label,
      Effect_short = "Reward",
      value = reward_effect
    ),
    data.frame(
      Parameter = parameter_label,
      Effect_short = "WM load x Reward",
      value = interaction_effect
    )
  )
}

effect_draws <- bind_rows(
  get_effect_posterior_draws(fit, "beta_mf", "beta_mf"),
  get_effect_posterior_draws(fit, "beta_mb", "beta_mb")
) %>%
  mutate(
    Effect_short = factor(
      Effect_short,
      levels = c("WM load", "Reward", "WM load x Reward")
    ),
    Parameter = factor(
      Parameter,
      levels = c("beta_mf", "beta_mb"),
      labels = c(
        "italic(beta)[plain(MF)]",
        "italic(beta)[plain(MB)]"
      )
    )
  )

plot_file_effect_densities <- file.path(here, "Figures", "posterior_load_magnitude_effect_densities.pdf")

p_effect_densities <- ggplot(
  effect_draws,
  aes(x = value, fill = Effect_short, color = Effect_short)
) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    linewidth = 0.5,
    color = "grey35"
  ) +
  geom_density(alpha = 0.55, linewidth = 0.75) +
  facet_grid(
    Parameter ~ Effect_short,
    scales = "free",
    labeller = labeller(Parameter = label_parsed)
  ) +
  scale_fill_manual(values = posterior_effect_colors, guide = "none") +
  scale_color_manual(values = posterior_effect_colors, guide = "none") +
  labs(
    x = "Posterior contrast",
    y = "Density"
  ) +
  theme_classic(base_size = 14) +
  theme(
    strip.text = element_text(face = "bold", size = 12),
    panel.spacing = unit(1.1, "lines"),
    axis.text = element_text(color = "black"),
    axis.line = element_line(linewidth = 1.2, color = "black"),
    axis.ticks = element_line(linewidth = 0.8, color = "black"),
    plot.margin = margin(8, 8, 8, 8),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA)
  )

ggsave(
  plot_file_effect_densities,
  plot = p_effect_densities,
  device = grDevices::cairo_pdf, fallback_resolution = 600,
  dpi = 600,
  width = 12,
  height = 3,
  units = "in",
  bg = "transparent"
)

cat("\nPosterior contrast density plot saved:", plot_file_effect_densities, "\n")