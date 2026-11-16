# Behavioural analysis data

## Overview

This dataset contains the behavioural estimates and model-validation summaries for 52 participants. It supports mixed-effects modelling, posterior contrasts, moderated mediation and model-check figures. Each participant contributed four conditions formed by crossing working-memory load with reward magnitude.

## Data directories

### `analysis/` — Performance and fitted parameters

`parameters_generated.xlsx` contains one row per participant and condition (208 rows). The keys are `subid`, `load` and `magnitude`. The four measures are corrected reward rate (`cr`), the RL mixing weight (`w`), and the HBRL model-based and model-free weights (`beta_mb` and `beta_mf`). The mixed-effects and mediation analyses use this workbook. `demographics_summary.csv` reports cohort totals.

### `posterior/` — Group posterior contrasts

Each row of `hbrl_group_posterior_draws.csv` is one matched posterior draw. Each parameter column contains the mean across the 52 participants for that draw and condition. The suffixes `lload` and `hload` identify low and high load. Compute contrasts within each row before summarising their distributions across draws. `load_magnitude_interaction_posterior_tests.csv` provides the six reference contrasts.

### `validation/` — Model checks and parameter recovery

The `RL/` and `HBRL/` subdirectories contain MCMC diagnostics and posterior predictive checks. The replicate tables contain group-level simulations. Each row of `ppc_scatter_points.csv` pairs an observed and a predicted participant-level summary for mean reward or optimal rate. Participant identifiers are omitted.

In each subdirectory, `parameter_recovery_points.csv` pairs generating and fitted values, and `parameter_recovery_summary.csv` gives their correlations and root-mean-square errors. RL recovery uses the participant-based `empirical_map` simulations for the mixing weight. HBRL recovery covers five simulation sets and eight condition-specific MB and MF weights.

## Reproduction

Follow the [behavioural group-analysis guide](../../code/BHV_for_group_data/README.md) for input checks, execution order and output locations. The workflow starts from saved estimates and simulations. Refitting the learning models or recalculating observed trial-level diagnostics requires the original trial records.
