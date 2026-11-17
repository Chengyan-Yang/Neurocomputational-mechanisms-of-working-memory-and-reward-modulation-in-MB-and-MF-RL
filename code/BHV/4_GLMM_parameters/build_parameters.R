# Build the GLMM input from current results. Run this script from any directory
# with Rscript, or source it from RStudio.
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
bhv <- dirname(here)
data_dir <- Sys.getenv("BHV_DATA_DIR", file.path(dirname(dirname(bhv)), "data", "BHV"))
ids <- 1:52

# Align every input to participant ID before selecting condition columns.
read_subjects <- function(path, id_col) {
  tab <- read.csv(path, check.names = FALSE)
  key <- tab[[id_col]]
  if (length(key) != length(ids) || !setequal(key, ids) || anyDuplicated(key))
    stop("Expected one row per participant (1-52) in ", path)
  tab[match(ids, key), , drop = FALSE]
}
column <- function(tab, name) {
  x <- tab[[name]]
  if (!is.numeric(x) || length(x) != length(ids) || any(!is.finite(x)))
    stop("Missing or invalid numeric column: ", name)
  x
}

cr <- read_subjects(file.path(data_dir, "2_stage_task", "preprocessed", "learning",
                              "performance_overall.csv"), "participant")
hbrl <- file.path(bhv, "3_HBRLmodel")
mb <- read_subjects(file.path(hbrl, "mb_estimates_hdi.csv"), "subject_id")
mf <- read_subjects(file.path(hbrl, "mf_estimates_hdi.csv"), "subject_id")
blocks <- list()
for (load in c("low", "high")) {
  load_code <- if (load == "low") "lload" else "hload"
  rl <- read_subjects(file.path(bhv, "3_RLmodel", "model_1", paste0(load, "_load"),
                                paste0("results_M1_", load, ".csv")), "id")
  for (magnitude in c("small", "large")) {
    magnitude_code <- if (magnitude == "small") "small_magnitude" else "large_magnitude"
    blocks[[length(blocks) + 1L]] <- data.frame(
      subid = ids, load = load, magnitude = magnitude,
      cr = column(cr, paste("cr", load_code, magnitude_code, sep = "_")),
      beta_mb = column(mb, paste("beta_mb", magnitude, load_code, "median", sep = "_")),
      beta_mf = column(mf, paste("beta_mf", magnitude, load_code, "median", sep = "_")),
      w = column(rl, paste0("w_", magnitude))
    )
  }
}
parameters <- do.call(rbind, blocks)
destination <- file.path(here, "parameters_generated.xlsx")
openxlsx::write.xlsx(list(Sheet1 = parameters), destination, overwrite = TRUE)
message("Saved ", nrow(parameters), " rows to ", destination)
