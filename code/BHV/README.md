# Behavioral analyses from trial-level data

Fit RL and HBRL models to trial data from 52 participants. The MEG regression and RSA workflows use the resulting HBRL estimates. To start from saved parameter estimates, use [the group-data workflow](../BHV_for_group_data/README.md).

## Data and software

Keep `code/` and `data/` as siblings. The trial CSVs belong in `data/BHV/2_stage_task/original/`. Run the commands below from `code/BHV/`, except for the MATLAB block, which is entered in MATLAB with the same working directory.

Use R with `rstan` and a compatible C++ compiler, MATLAB with Optimization, Statistics and Machine Learning, and Parallel Computing toolboxes, and Python. R analyses use `abind`, `dplyr`, `tidyr`, `readxl`, `readr`, `purrr`, `loo`, `bayestestR`, `ggplot2`, `openxlsx`, `glmmTMB`, `emmeans`, `car`, and `lme4`. Make `Rscript` available to Python and install the validation dependencies in the Python environment used to run the scripts.

```sh
python -m pip install -r 3_RLmodel/model_1/validation/requirements_model1_validation.txt
```

## 1. Prepare trial records

The two R scripts prepare the RL and HBRL inputs independently. Both write to `data/BHV/2_stage_task/preprocessed/learning/`. The two MATLAB conversions are required only for the RL branch.

```sh
Rscript 1_preprocess/RLpreprocess_forRL.R
Rscript 1_preprocess/RLpreprocess_forHBRL.R
```

## 2. Fit and compare the RL models

```matlab
run("2_csv2mat_forRLfit/csv2mat_lload.m")
run("2_csv2mat_forRLfit/csv2mat_hload.m")
addpath("3_RLmodel")
run_all_rl_models()
```

The conversions produce `groupdata_lload.mat` and `groupdata_hload.mat`. The driver fits models 1–7 separately at low and high load (14 fits). After fitting, export the comparison table and validate model 1. The reported recovery analysis uses the participant-based `empirical_map` simulations for the mixing weights `w_small` and `w_large`.

```sh
python 3_RLmodel/model_comparison/_build_model_comparison_table.py
python 3_RLmodel/model_1/validation/run_all_diagnostics.py
```

Inspect `MEG_RL_model_comparison.csv` in the model-comparison directory and the likelihood, MCMC, posterior predictive check (PPC), and recovery outputs under `3_RLmodel/model_1/validation/outputs/`.

## 3. Fit and validate the HBRL model

This branch can start after step 1. Quote the fit script name because it contains an ampersand. Posterior extraction writes `mb_estimates_hdi.csv`, `mf_estimates_hdi.csv`, group posterior contrasts, and density plots. The validation flag includes all five Stan recovery datasets. The default full diagnostics also export the labeled key-parameter 95% highest-density interval (HDI) table to `validation/outputs/mcmc/`.

```sh
Rscript "3_HBRLmodel/mfmblearning_magnitude&load.R"
Rscript 3_HBRLmodel/posterior_extract.R
python 3_HBRLmodel/validation/run_all_diagnostics.py --include-stan-recovery
```

## 4. Test condition effects and moderated mediation

`build_parameters.R` joins corrected reward rate (CR), model-1 mixing weights, and HBRL model-based (MB) and model-free (MF) estimates by participant and condition. The resulting `parameters_generated.xlsx` contains 208 rows (52 participants × 2 loads × 2 reward levels). Both analyses below read that workbook; either can run first.

```sh
Rscript 4_GLMM_parameters/build_parameters.R
Rscript 4_GLMM_parameters/GLMM_for_parameters.R
Rscript 5_multilevel_moderated_mediation/run_all.R
```

The mixed-effects models export `4_GLMM_parameters/GLMM_Results.xlsx` and four raincloud plots. Mediation fits separate models for MB, MF, and the mixing weight, using 2,000 participant-level bootstrap resamples (seed 260909). Inspect `5_multilevel_moderated_mediation/outputs/moderated_mediation_bootstrap_summary.csv`, `outputs/run_provenance.txt`, and `figures/` in that directory. If R cannot locate Python for diagram rendering, set `MEDIATION_PYTHON` to an interpreter with NumPy, pandas, and Matplotlib.
