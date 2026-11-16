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
# This script performs generalized linear mixed-effects model (GLMM) analyses to examine the effects of 
# working memory (WM) load, reward magnitude, and their interaction on behavioral performance and reinforcement learning parameters. 
# Beta regression models are applied to corrected reward rate and mixing weight, while Gaussian models are used for model-based 
# and model-free weights, with participant-specific random intercepts. The analyses include Type III Wald tests, estimated marginal means, 
# and Bonferroni-adjusted simple effects comparisons. The script exports statistical results and model diagnostics to an Excel file and 
# generates raincloud plots illustrating the distributions and significant differences across experimental conditions.

library(glmmTMB)
library(emmeans)
library(car)
library(dplyr)
library(readxl)
library(openxlsx)
library(ggplot2)

# ============================================================
# 1. Input and output paths
# ============================================================

input_file <- file.path(here, "..", "..", "group_data", "BHV", "analysis", "parameters_generated.xlsx")
output_file <- file.path(here, "results", "GLMM", "GLMM_Results.xlsx")

plot_dir <- file.path(here, "results", "GLMM", "figures")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

data <- read_excel(input_file, sheet = "Sheet1")

# ============================================================
# 2. Validate input variables and configure factor contrasts
# ============================================================

required_cols <- c("subid", "load", "magnitude", "cr", "w", "beta_mb", "beta_mf")
missing_cols <- setdiff(required_cols, names(data))

if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

find_level <- function(x, target) {
  vals <- unique(na.omit(as.character(x)))
  vals_trim <- trimws(vals)
  
  exact_hits <- vals[base::tolower(vals_trim) == base::tolower(target)]
  
  if (length(exact_hits) == 1) {
    return(exact_hits)
  }
  
  partial_hits <- vals[grepl(base::tolower(target), base::tolower(vals_trim), fixed = TRUE)]
  
  if (length(partial_hits) == 1) {
    return(partial_hits)
  }
  
  stop(
    "Could not uniquely identify level '", target, "'.\n",
    "Available levels: ", paste(vals, collapse = ", "), "\n",
    "Expected low/high for load and small/large for magnitude."
  )
}

# Resolve potentially inconsistent case or whitespace in the two factor labels.
load_low_level <- find_level(data$load, "low")
load_high_level <- find_level(data$load, "high")
magnitude_small_level <- find_level(data$magnitude, "small")
magnitude_large_level <- find_level(data$magnitude, "large")

data <- data %>%
  mutate(
    subid = as.factor(subid),
    load = factor(load, levels = c(load_low_level, load_high_level)),
    magnitude = factor(magnitude, levels = c(magnitude_small_level, magnitude_large_level))
  )

if (any(is.na(data$load))) {
  stop("Some load values could not be matched to the low/high levels.")
}

if (any(is.na(data$magnitude))) {
  stop("Some magnitude values could not be matched to the small/large levels.")
}

# Sum-to-zero contrasts support Type III tests of main effects and interactions.
contrasts(data$load) <- contr.sum(nlevels(data$load))
contrasts(data$magnitude) <- contr.sum(nlevels(data$magnitude))

if (nlevels(data$load) * nlevels(data$magnitude) != 4) {
  stop("The plotting code requires four load-by-magnitude conditions. Check the factor levels.")
}

# ============================================================
# 3. Specify outcome families and helper functions
# ============================================================

# Beta regression requires 0 < y < 1. By default, stop at boundary values
# rather than silently transforming observations. Set to "ordbeta" only if justified.
beta_boundary_policy <- "stop"

# Fit beta models to bounded outcomes and Gaussian models to the two weights.
analysis_list <- list(
  list(dv = "cr",      family_type = "beta"),
  list(dv = "w",       family_type = "beta"),
  list(dv = "beta_mb", family_type = "gaussian"),
  list(dv = "beta_mf", family_type = "gaussian")
)

# Conditions: LS = low load/small reward; LL = low load/large reward;
# HS = high load/small reward; HL = high load/large reward.
condition_levels <- c("LS", "LL", "HS", "HL")

condition_colors <- c(
  "LS" = "#a9c19c",
  "LL" = "#346db1",
  "HS" = "#eacdd2",
  "HL" = "#b03949"
)

p_to_stars <- function(p) {
  dplyr::case_when(
    is.na(p) ~ "",
    p < 0.001 ~ "***",
    p < 0.01  ~ "**",
    p < 0.05  ~ "*",
    TRUE ~ "ns"
  )
}

make_condition_label <- function(load_value, magnitude_value) {
  load_value <- as.character(load_value)
  magnitude_value <- as.character(magnitude_value)
  
  dplyr::case_when(
    load_value == load_low_level  & magnitude_value == magnitude_small_level  ~ "LS",
    load_value == load_low_level  & magnitude_value == magnitude_large_level ~ "LL",
    load_value == load_high_level & magnitude_value == magnitude_small_level  ~ "HS",
    load_value == load_high_level & magnitude_value == magnitude_large_level ~ "HL",
    TRUE ~ NA_character_
  )
}

choose_family <- function(y, dv_name, family_type) {
  if (family_type == "gaussian") {
    return(gaussian())
  }
  
  if (family_type == "beta") {
    bad_n <- sum(y <= 0 | y >= 1, na.rm = TRUE)
    
    if (bad_n == 0) {
      return(beta_family(link = "logit"))
    }
    
    if (beta_boundary_policy == "ordbeta") {
      if ("ordbeta" %in% getNamespaceExports("glmmTMB")) {
        message(dv_name, " contains ", bad_n, " boundary observations (0 or 1); using ordbeta().")
        return(getExportedValue("glmmTMB", "ordbeta")(link = "logit"))
      } else {
        stop(dv_name, " contains boundary observations (0 or 1), but ordbeta() is unavailable in this glmmTMB version.")
      }
    }
    
    stop(
      dv_name, " contains ", bad_n, " observations outside (0, 1); beta_family() cannot be used directly.\n",
      "Verify whether these boundary values are valid. To use ordbeta(), set beta_boundary_policy to \"ordbeta\"."
    )
  }
  
  stop("Unknown family_type: ", family_type)
}

extract_anova <- function(model, dv_name) {
  res <- as.data.frame(car::Anova(model, type = "III"))
  res$Term <- gsub("magnitude", "reward", rownames(res), fixed = TRUE)
  rownames(res) <- NULL
  res$Dependent_Variable <- dv_name
  
  p_col <- grep("^Pr\\(", names(res), value = TRUE)[1]
  
  if (!is.na(p_col)) {
    res$P_value <- res[[p_col]]
    res$Significance <- p_to_stars(res$P_value)
  } else {
    res$P_value <- NA_real_
    res$Significance <- ""
  }
  
  res %>%
    dplyr::select(Dependent_Variable, Term, P_value, Significance, dplyr::everything())
}

extract_emmeans <- function(emm_obj, dv_name, effect_name) {
  res <- as.data.frame(summary(emm_obj, infer = c(TRUE, TRUE)))
  res$Dependent_Variable <- dv_name
  res$Effect <- effect_name
  
  res %>%
    dplyr::select(Dependent_Variable, Effect, dplyr::everything())
}

extract_pairs <- function(pairs_obj, dv_name, effect_name) {
  res <- as.data.frame(summary(
    pairs_obj,
    by = NULL,
    adjust = "bonferroni",
    infer = c(TRUE, TRUE)
  ))
  
  res$Dependent_Variable <- dv_name
  res$Effect <- effect_name
  
  p_col <- grep("^p\\.value", names(res), value = TRUE)[1]
  
  if (!is.na(p_col)) {
    res$P_value <- res[[p_col]]
    res$Significance <- p_to_stars(res$P_value)
  } else {
    res$P_value <- NA_real_
    res$Significance <- ""
  }
  
  res %>%
    dplyr::select(Dependent_Variable, Effect, P_value, Significance, dplyr::everything())
}

extract_brackets <- function(pairs_obj, dv_name, effect_name) {
  res <- as.data.frame(summary(
    pairs_obj,
    by = NULL,
    adjust = "bonferroni",
    infer = c(TRUE, TRUE)
  ))
  
  p_col <- grep("^p\\.value", names(res), value = TRUE)[1]
  
  if (is.na(p_col)) {
    return(data.frame(
      Dependent_Variable = character(),
      Effect = character(),
      Condition_1 = character(),
      Condition_2 = character(),
      P_value = numeric(),
      Significance = character()
    ))
  }
  
  contrast_parts <- strsplit(as.character(res$contrast), " - ", fixed = TRUE)
  valid_rows <- lengths(contrast_parts) == 2
  
  res <- res[valid_rows, , drop = FALSE]
  contrast_parts <- contrast_parts[valid_rows]
  
  if (nrow(res) == 0) {
    return(data.frame(
      Dependent_Variable = character(),
      Effect = character(),
      Condition_1 = character(),
      Condition_2 = character(),
      P_value = numeric(),
      Significance = character()
    ))
  }
  
  level_1 <- sapply(contrast_parts, `[`, 1)
  level_2 <- sapply(contrast_parts, `[`, 2)
  
  if (effect_name == "Reward within Load") {
    condition_1 <- make_condition_label(res$load, level_1)
    condition_2 <- make_condition_label(res$load, level_2)
  } else if (effect_name == "Load within Reward magnitude") {
    condition_1 <- make_condition_label(level_1, res$magnitude)
    condition_2 <- make_condition_label(level_2, res$magnitude)
  } else {
    stop("Unknown effect_name: ", effect_name)
  }
  
  data.frame(
    Dependent_Variable = dv_name,
    Effect = effect_name,
    Condition_1 = condition_1,
    Condition_2 = condition_2,
    P_value = res[[p_col]],
    Significance = p_to_stars(res[[p_col]])
  )
}

extract_diagnostics <- function(model, dv_name, n_used) {
  data.frame(
    Dependent_Variable = dv_name,
    N_Used = n_used,
    Convergence_Code = model$fit$convergence,
    Positive_Definite_Hessian = model$sdr$pdHess,
    AIC = AIC(model),
    BIC = BIC(model),
    LogLik = as.numeric(logLik(model))
  )
}

make_empty_raw_plot_data <- function() {
  data.frame(
    Dependent_Variable = character(),
    subid = character(),
    load = character(),
    magnitude = character(),
    Condition = character(),
    Value = numeric()
  )
}

make_half_violin_data <- function(plot_df, condition_levels, width = 0.32, n = 128) {
  out <- list()
  row_id <- 1
  
  for (cond in condition_levels) {
    y <- plot_df %>%
      dplyr::filter(Condition == cond) %>%
      dplyr::pull(Value)
    
    y <- y[is.finite(y)]
    
    if (length(y) < 3 || length(unique(y)) < 2) {
      next
    }
    
    dens <- density(
      y,
      n = n,
      from = min(y, na.rm = TRUE),
      to = max(y, na.rm = TRUE)
    )
    
    x0 <- match(cond, condition_levels)
    scaled_density <- dens$y / max(dens$y, na.rm = TRUE) * width
    
    out[[row_id]] <- data.frame(
      Condition = cond,
      Group_ID = cond,
      x = c(rep(x0, length(dens$x)), rev(x0 - scaled_density)),
      y = c(dens$x, rev(dens$x))
    )
    
    row_id <- row_id + 1
  }
  
  if (length(out) == 0) {
    return(data.frame(
      Condition = character(),
      Group_ID = character(),
      x = numeric(),
      y = numeric()
    ))
  }
  
  dplyr::bind_rows(out)
}

# ============================================================
# 4. Fit models and collect inferential results
# ============================================================

results <- list()

all_anova_results <- data.frame()
all_emmeans <- data.frame()
all_simple_effects <- data.frame()
all_diagnostics <- data.frame()
all_raw_plot_data <- make_empty_raw_plot_data()
all_brackets <- data.frame(
  Dependent_Variable = character(),
  Effect = character(),
  Condition_1 = character(),
  Condition_2 = character(),
  P_value = numeric(),
  Significance = character()
)
all_errors <- data.frame()

for (item in analysis_list) {
  dv_name <- item$dv
  
  cat("\n==================================================\n")
  cat("Analyzing outcome:", dv_name, "\n")
  
  tryCatch({
    model_data <- data %>%
      dplyr::select(subid, load, magnitude, dplyr::all_of(dv_name)) %>%
      dplyr::filter(!is.na(.data[[dv_name]]), !is.na(subid), !is.na(load), !is.na(magnitude)) %>%
      droplevels()
    
    if (nlevels(model_data$load) < 2) {
      stop("The load factor has fewer than two levels; analysis cannot proceed.")
    }
    
    if (nlevels(model_data$magnitude) < 2) {
      stop("The magnitude factor has fewer than two levels; analysis cannot proceed.")
    }
    
    contrasts(model_data$load) <- contr.sum(nlevels(model_data$load))
    contrasts(model_data$magnitude) <- contr.sum(nlevels(model_data$magnitude))
    
    model_family <- choose_family(
      y = model_data[[dv_name]],
      dv_name = dv_name,
      family_type = item$family_type
    )
    
    # Include load, reward magnitude, and their interaction as fixed effects,
    # with a participant-specific random intercept to account for repeated measures.
    formula_obj <- as.formula(
      paste(dv_name, "~ load * magnitude + (1 | subid)")
    )
    
    model <- glmmTMB(
      formula_obj,
      data = model_data,
      family = model_family
    )
    
    cat("--- Type III Wald Chi-square ANOVA ---\n")
    anova_res <- extract_anova(model, dv_name)
    print(anova_res)
    
    # Estimate marginal means and conditional contrasts for both simple effects.
    emm_magnitude_by_load <- emmeans(model, ~ magnitude | load)
    emm_load_by_magnitude <- emmeans(model, ~ load | magnitude)
    
    cat("\n--- Simple effects: Reward within Load ---\n")
    pairs_magnitude_by_load <- pairs(emm_magnitude_by_load, adjust = "none")
    simple_magnitude_by_load <- extract_pairs(
      pairs_magnitude_by_load,
      dv_name,
      "Reward within Load"
    )
    print(simple_magnitude_by_load)
    
    cat("\n--- Simple effects: Load within Reward magnitude ---\n")
    pairs_load_by_magnitude <- pairs(emm_load_by_magnitude, adjust = "none")
    simple_load_by_magnitude <- extract_pairs(
      pairs_load_by_magnitude,
      dv_name,
      "Load within Reward magnitude"
    )
    print(simple_load_by_magnitude)
    
    results[[dv_name]] <- list(
      model = model,
      emm_magnitude_by_load = emm_magnitude_by_load,
      emm_load_by_magnitude = emm_load_by_magnitude
    )
    
    raw_plot_part <- model_data %>%
      dplyr::mutate(
        Dependent_Variable = dv_name,
        Value = .data[[dv_name]],
        Condition = make_condition_label(load, magnitude),
        subid = as.character(subid),
        load = as.character(load),
        magnitude = as.character(magnitude)
      ) %>%
      dplyr::select(Dependent_Variable, subid, load, magnitude, Condition, Value)
    
    if (any(is.na(raw_plot_part$Condition))) {
      stop("Some conditions could not be mapped to LS/LL/HS/HL.")
    }
    
    bracket_magnitude_by_load <- extract_brackets(
      pairs_magnitude_by_load,
      dv_name,
      "Reward within Load"
    )
    
    bracket_load_by_magnitude <- extract_brackets(
      pairs_load_by_magnitude,
      dv_name,
      "Load within Reward magnitude"
    )
    
    all_raw_plot_data <- dplyr::bind_rows(all_raw_plot_data, raw_plot_part)
    all_brackets <- dplyr::bind_rows(all_brackets, bracket_magnitude_by_load, bracket_load_by_magnitude)
    all_anova_results <- dplyr::bind_rows(all_anova_results, anova_res)
    
    all_emmeans <- dplyr::bind_rows(
      all_emmeans,
      extract_emmeans(emm_magnitude_by_load, dv_name, "EMMs: Reward by Load"),
      extract_emmeans(emm_load_by_magnitude, dv_name, "EMMs: Load by Reward magnitude")
    )
    
    all_simple_effects <- dplyr::bind_rows(
      all_simple_effects,
      simple_magnitude_by_load,
      simple_load_by_magnitude
    )
    
    all_diagnostics <- dplyr::bind_rows(
      all_diagnostics,
      extract_diagnostics(model, dv_name, nrow(model_data))
    )
    
  }, error = function(e) {
    cat("Analysis of ", dv_name, " failed: ", conditionMessage(e), "\n", sep = "")
    
    all_errors <<- dplyr::bind_rows(
      all_errors,
      data.frame(
        Dependent_Variable = dv_name,
        Error_Message = conditionMessage(e)
      )
    )
  })
}

# ============================================================
# 5. Export model results and plotting data to Excel
# ============================================================

# Save Type III Wald tests, estimated marginal means, Bonferroni-adjusted
# simple effects, model diagnostics, and data used for figures.
wb <- createWorkbook()

addWorksheet(wb, "TypeIII_ANOVA")
writeData(wb, "TypeIII_ANOVA", all_anova_results)

addWorksheet(wb, "EMMs")
writeData(wb, "EMMs", all_emmeans)

addWorksheet(wb, "Simple_Effects")
writeData(wb, "Simple_Effects", all_simple_effects)

addWorksheet(wb, "Model_Diagnostics")
writeData(wb, "Model_Diagnostics", all_diagnostics)

addWorksheet(wb, "Raw_Plot_Data")
writeData(wb, "Raw_Plot_Data", all_raw_plot_data)

addWorksheet(wb, "Plot_Brackets")
writeData(wb, "Plot_Brackets", all_brackets)

if (nrow(all_errors) > 0) {
  addWorksheet(wb, "Errors")
  writeData(wb, "Errors", all_errors)
}

saveWorkbook(wb, output_file, overwrite = TRUE)

cat("\nExport completed. File: ", output_file, "\n")

# ============================================================
# 6. Save a separate raincloud plot for each outcome
# ============================================================

if (nrow(all_raw_plot_data) == 0) {
  stop("No data available for plotting. Check the Errors worksheet for model-fitting failures.")
}

for (dv_name in c("cr", "w", "beta_mb", "beta_mf")) {
  plot_data <- all_raw_plot_data %>%
    dplyr::filter(Dependent_Variable == dv_name) %>%
    dplyr::mutate(
      Condition = factor(Condition, levels = condition_levels),
      x_box = as.numeric(Condition) + 0.08,
      x_point = as.numeric(Condition) + 0.20
    ) %>%
    dplyr::filter(!is.na(Condition))
  
  if (nrow(plot_data) == 0) {
    warning(dv_name, " has no available plotting data; skipping.")
    next
  }
  
  violin_data <- make_half_violin_data(
    plot_df = plot_data,
    condition_levels = condition_levels
  ) %>%
    dplyr::mutate(
      Condition = factor(Condition, levels = condition_levels)
    )
  
  y_min <- min(plot_data$Value, na.rm = TRUE)
  y_max <- max(plot_data$Value, na.rm = TRUE)
  y_range <- y_max - y_min
  
  if (!is.finite(y_range) || y_range <= 0) {
    y_range <- 1
  }
  
  bracket_plot_data <- all_brackets %>%
    dplyr::filter(
      Dependent_Variable == dv_name,
      !is.na(P_value),
      P_value < 0.05
    ) %>%
    dplyr::mutate(
      x1 = match(Condition_1, condition_levels),
      x2 = match(Condition_2, condition_levels)
    ) %>%
    dplyr::filter(!is.na(x1), !is.na(x2), x1 != x2) %>%
    dplyr::mutate(
      xmin = pmin(x1, x2),
      xmax = pmax(x1, x2),
      xmid = (xmin + xmax) / 2
    ) %>%
    dplyr::arrange(xmin, xmax, P_value) %>%
    dplyr::mutate(
      bracket_id = dplyr::row_number(),
      y = y_max + y_range * (0.14 + 0.14 * (bracket_id - 1)),
      y_tip = y - y_range * 0.035,
      y_text = y + y_range * 0.04
    )
  
  # Combine a half-violin density, boxplot, jittered observations,
  # and brackets for Bonferroni-adjusted significant contrasts.
  p <- ggplot() +
    geom_polygon(
      data = violin_data,
      aes(
        x = x,
        y = y,
        group = Group_ID,
        fill = Condition,
        color = Condition
      ),
      alpha = 0.55,
      linewidth = 0.3
    ) +
    geom_boxplot(
      data = plot_data,
      aes(
        x = x_box,
        y = Value,
        fill = Condition,
        group = Condition
      ),
      width = 0.12,
      outlier.shape = NA,
      alpha = 0.78,
      color = "grey25",
      linewidth = 0.4
    ) +
    geom_point(
      data = plot_data,
      aes(
        x = x_point,
        y = Value,
        color = Condition
      ),
      position = position_jitter(width = 0.055, height = 0),
      alpha = 0.58,
      size = 1.4
    ) +
    geom_segment(
      data = bracket_plot_data,
      aes(x = xmin, xend = xmax, y = y, yend = y),
      inherit.aes = FALSE,
      linewidth = 0.45,
      color = "grey15"
    ) +
    geom_segment(
      data = bracket_plot_data,
      aes(x = xmin, xend = xmin, y = y_tip, yend = y),
      inherit.aes = FALSE,
      linewidth = 0.45,
      color = "grey15"
    ) +
    geom_segment(
      data = bracket_plot_data,
      aes(x = xmax, xend = xmax, y = y_tip, yend = y),
      inherit.aes = FALSE,
      linewidth = 0.45,
      color = "grey15"
    ) +
    geom_text(
      data = bracket_plot_data,
      aes(
        x = xmid,
        y = y_text,
        label = Significance
      ),
      inherit.aes = FALSE,
      size = 5,
      fontface = "bold",
      color = "grey10"
    ) +
    scale_x_continuous(
      breaks = 1:4,
      labels = condition_levels,
      limits = c(0.45, 4.55)
    ) +
    scale_y_continuous(
      expand = expansion(mult = c(0.08, 0.05))
    ) +
    scale_fill_manual(values = condition_colors, drop = FALSE) +
    scale_color_manual(values = condition_colors, drop = FALSE) +
    labs(
      x = "Condition",
      y = ifelse(dv_name == "beta_mb", "β_MB",
                 ifelse(dv_name == "beta_mf", "β_MF", dv_name))
    ) +
    coord_cartesian(clip = "off") +
    theme_classic(base_size = 14) +
    theme(
      axis.line = element_line(linewidth = 1.2, color = "black"),
      axis.ticks = element_line(linewidth = 0.8, color = "black"),
      axis.ticks.length = unit(0.18, "cm"),
      axis.text.x = element_text(size = 12),
      axis.text.y = element_text(size = 11),
      # axis.title = element_text(face = "bold", size = 14),
      legend.position = "none",
      plot.margin = margin(5.5, 5.5, 5.5, 5.5)
    )
  
  plot_file <- file.path(
    plot_dir,
    paste0("Raincloud_", dv_name, "_S2.pdf")
  )
  
  ggsave(
    filename = plot_file,
    plot = p,
    width = 4,
    height = 3,
    device = grDevices::cairo_pdf,
    fallback_resolution = 600,
    dpi = 600,
    units = "in"
  )
  
  cat("\nRaincloud plot saved: ", plot_file, "\n")
}
