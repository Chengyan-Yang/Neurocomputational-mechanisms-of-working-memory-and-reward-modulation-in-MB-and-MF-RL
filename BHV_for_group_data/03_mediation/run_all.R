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
# Set MEDIATION_PLOTS_ONLY=TRUE to regenerate figures from existing fit outputs.
analysis_dir <- here
plots_only <- tolower(Sys.getenv("MEDIATION_PLOTS_ONLY", "false")) %in% c("true", "1", "yes")

# Verify plotting dependencies before starting the bootstrap.
python_exe <- Sys.getenv("MEDIATION_PYTHON", "")
if (!nzchar(python_exe)) {
  candidates <- unname(Sys.which(c("python", "python3")))
} else {
  candidates <- python_exe
}
python_exe <- NULL
for (candidate in unique(candidates[file.exists(candidates)])) {
  status <- tryCatch(system2(candidate, c("-c", shQuote("import numpy, pandas, matplotlib.pyplot")),
                             stdout = FALSE, stderr = FALSE), error = function(e) 1L)
  if (isTRUE(status == 0L)) {
    python_exe <- candidate
    break
  }
}
if (is.null(python_exe)) stop("Set MEDIATION_PYTHON to a working Python executable with numpy, pandas and matplotlib.")

if (!plots_only) {
  source(file.path(analysis_dir, "00_prepare_data.R"), chdir = TRUE, local = new.env(parent = environment()))
  source(file.path(analysis_dir, "01_fit_moderated_mediation.R"), chdir = TRUE, local = new.env(parent = environment()))
}
message("Rendering mediation diagrams with ", python_exe)
script <- file.path(analysis_dir, "02_plot_mediation_diagrams.py")
status <- system2(python_exe, shQuote(script))
if (status != 0L) stop("Python plotting failed (exit ", status, "). See the Python output above. To retry only the figures, set MEDIATION_PLOTS_ONLY=TRUE.")
source(file.path(analysis_dir, "03_plot_effects.R"), chdir = TRUE, local = new.env(parent = environment()))
message("Completed MEG single-mediator analysis: ", analysis_dir)
