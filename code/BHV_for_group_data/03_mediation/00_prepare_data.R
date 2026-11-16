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
# Prepare data for multilevel moderated mediation.
# Input rows should be one subject x one load x one magnitude condition.

required_packages <- c("readxl", "dplyr", "readr")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop(
    "Please install missing packages first: ",
    paste(missing_packages, collapse = ", "),
    "\nRun: install.packages(c(",
    paste(sprintf('"%s"', missing_packages), collapse = ", "),
    "))"
  )
}

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(readr)
})

analysis_dir <- file.path(dirname(here), "results", "mediation")
input_file <- file.path(here, "..", "..", "..", "group_data", "BHV", "analysis", "parameters_generated.xlsx")
output_dir <- file.path(analysis_dir, "outputs")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(input_file)) stop("Cannot find input file: ", input_file)

raw <- readxl::read_excel(input_file, sheet = 1)
if (ncol(raw) != 7) {
  stop("Expected 7 columns: subid, load, magnitude, cr, beta_MB, beta_MF, w. Found ", ncol(raw), " columns.")
}

# Rename by column position to avoid encoding problems with the Greek beta symbol in RStudio/Windows.
names(raw) <- c("subid", "load", "magnitude", "cr", "beta_mb", "beta_mf", "w")

mediation_data <- raw %>%
  mutate(
    subid = factor(subid),
    load = factor(tolower(as.character(load)), levels = c("low", "high")),
    magnitude = factor(tolower(as.character(magnitude)), levels = c("small", "large")),
    cr = as.numeric(cr),
    beta_mb = as.numeric(beta_mb),
    beta_mf = as.numeric(beta_mf),
    w = as.numeric(w),
    # Effect coding: coefficients for load_c are high-load minus low-load when magnitude_c = 0.
    load_c = ifelse(load == "high", 0.5, -0.5),
    magnitude_c = ifelse(magnitude == "large", 0.5, -0.5)
  ) %>%
  group_by(subid) %>%
  mutate(
    beta_mb_between = mean(beta_mb, na.rm = TRUE),
    beta_mf_between = mean(beta_mf, na.rm = TRUE),
    w_between = mean(w, na.rm = TRUE),
    beta_mb_within = beta_mb - beta_mb_between,
    beta_mf_within = beta_mf - beta_mf_between,
    w_within = w - w_between
  ) %>%
  ungroup()

checks <- list(
  n_rows = nrow(mediation_data),
  n_subjects = dplyr::n_distinct(mediation_data$subid),
  n_missing = sum(is.na(mediation_data)),
  load_levels = paste(levels(mediation_data$load), collapse = ", "),
  magnitude_levels = paste(levels(mediation_data$magnitude), collapse = ", ")
)

cell_counts <- mediation_data %>%
  count(load, magnitude, name = "n_rows")

if (checks$n_subjects != 52) warning("Expected 52 subjects, found ", checks$n_subjects, ".")
if (checks$n_rows != checks$n_subjects * 4) warning("Expected one row per subject x load x magnitude condition.")
if (any(is.na(mediation_data))) warning("Prepared data contain missing values; model fitting will drop incomplete rows.")

readr::write_csv(mediation_data, file.path(output_dir, "prepared_mediation_data.csv"))
readr::write_csv(cell_counts, file.path(output_dir, "condition_cell_counts.csv"))

message("Prepared mediation data saved to: ", file.path(output_dir, "prepared_mediation_data.csv"))
message("Rows: ", checks$n_rows, "; subjects: ", checks$n_subjects, "; missing cells: ", checks$n_missing)
print(cell_counts)
