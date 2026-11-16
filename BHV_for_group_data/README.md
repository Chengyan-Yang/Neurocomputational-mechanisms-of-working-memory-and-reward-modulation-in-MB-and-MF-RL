# Behavioral inference from released parameter estimates

This workflow reproduces the behavioral mixed-effects models, group posterior contrasts, moderated mediation, and selected model-check figures. Its starting point is the saved learning-model estimates and diagnostic summaries. The [trial-level workflow](../BHV/README.md) describes fitting the RL and HBRL models and generating new predictive and recovery simulations.

## Inputs

Keep `group_data/BHV/` and `code/BHV_for_group_data/` in their project locations. `analysis/parameters_generated.xlsx` contains 208 participant–condition rows: 52 participants, two working-memory loads, and two levels of reward magnitude. The outcomes are corrected reward rate (`cr`), RL mixing weight (`w`), and HBRL model-based (`beta_mb`) and model-free (`beta_mf`) weights. `demographics_summary.csv` describes the cohort.

`posterior/` contains 8,000 matched HBRL draws of participant-averaged weights. `validation/RL/` and `validation/HBRL/` contain diagnostic summaries, posterior predictive replicates, observed–predicted points, and recovery pairs.

## Software and execution

Use R with `readxl`, `glmmTMB`, `emmeans`, `car`, `dplyr`, `openxlsx`, `ggplot2`, `bayestestR`, `purrr`, `tidyr`, `readr`, and `lme4`. Python requires NumPy, pandas, Matplotlib, and SciPy. Run the commands from `code/BHV_for_group_data/`. Begin with the input check; the remaining analyses are independent. Each writes under `results/`.

```sh
Rscript 00_verify_inputs.R
Rscript 01_GLMM.R
Rscript 02_HBRL_posterior.R
Rscript 03_mediation/run_all.R
python 04_RL_validation.py
python 04_HBRL_validation.py
```

## Analyses and verification

1. **Input check.** Check that each participant has four condition records, confirm cohort totals and posterior draw structure, and compare the correlation and RMSE recalculated from each set of recovery pairs with its saved summary.

2. **Mixed-effects models.** `01_GLMM.R` fits beta models for CR and `w`, and Gaussian models for the MB and MF weights, with participant random intercepts. Inspect `results/GLMM/GLMM_Results.xlsx` for Type III tests, estimated marginal means, and simple effects. Four raincloud plots accompany the workbook. Excel may display `#NUM!` in the degrees-of-freedom column for asymptotic *z* tests with infinite degrees of freedom; the *z* statistics and *p* values are reported separately.

3. **Posterior inference.** `02_HBRL_posterior.R` computes load, reward, and interaction contrasts within each draw, then summarizes their distributions with 95% highest-density intervals. Compare the six rows in `results/posterior/load_magnitude_interaction_posterior_tests.csv` with the identically named reference in the input `posterior/` directory. Three density figures are saved in `results/posterior/Figures/`.

4. **Moderated mediation.** The R driver fits three separate single-mediator models for MB, MF, and `w`. The default is 2,000 participant-level bootstrap resamples, seed 260909. It calls Python to render path diagrams and R to plot effects. If needed, set `MEDIATION_PYTHON` to the Python executable with the plotting packages. Check `results/mediation/outputs/run_provenance.txt` for the actual resample and failure counts and `moderated_mediation_bootstrap_summary.csv` in the same directory for inference.

5. **Model checks.** The two Python scripts redraw posterior predictive checks for action 2 choice rate, mean reward, stay rate, and optimal rate, including the available scatter plots and optimal-rate time course. RL recovery uses `empirical_map` for `w_small` and `w_large`. HBRL recovery pools the five saved simulation sets for the MB and MF weights. The scripts produce 9 RL and 13 HBRL panels in `results/validation/`. The seven-model comparison table, likelihood check, and HBRL key-parameter HDI table can be read directly from the input validation directories.
