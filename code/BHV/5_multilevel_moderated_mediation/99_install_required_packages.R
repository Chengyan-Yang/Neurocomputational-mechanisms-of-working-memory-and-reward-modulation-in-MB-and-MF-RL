# Install packages required for the multilevel moderated mediation scripts.
# Run this once in RStudio if the analysis script reports missing packages.

packages <- c("readxl", "dplyr", "readr", "lme4", "ggplot2", "tidyr")
missing_packages <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) {
  install.packages(missing_packages)
} else {
  message("All required packages are already installed.")
}
