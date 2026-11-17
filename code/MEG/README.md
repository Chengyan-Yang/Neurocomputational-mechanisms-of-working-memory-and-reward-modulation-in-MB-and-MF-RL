# MEG analyses from participant-level derivatives

This workflow reproduces sensor- and source-level event-related fields (ERFs), sensor time–frequency analyses, brain–behavior regressions, and source-space searchlight representational similarity analysis (RSA) for 48 participants. For figure reproduction from released summaries alone, use [the group-data workflow](../MEG_for_group_data/README.md).

## Inputs and environment

Preserve the relative positions of `code/` and `data/`. The analyzed phases are `incentive`, `decision`, `transition`, and `feedback`. Participant IDs follow `5_BHV_MEG_regression/subject_covariates_48subs.csv` and retain their correspondence with the behavioral cohort.

The supplied starting points are `data/MEG/preprocessed/evokeds/`, `data/MEG/source/morphed_stcs/`, `data/MEG/preprocessed/time_frequency_epoch/`, and `data/MEG/time_frequency/sensor_band_power/`. Anatomical resources are under `data/MRI/surf/`. Use a writable project copy. Where the loader supports both sources, a regenerated derivative takes precedence over its supplied counterpart.

Run shell commands from `code/MEG/`. Use an MNE-Python environment with NumPy, SciPy, pandas, Matplotlib, seaborn, tqdm, Pillow, and the PyVista/Qt 3-D backend with OpenGL. The regression branch also uses R, RStan, dplyr, readr, and ggplot2. A fresh Python process picks up changes to phase configuration.

## 1. Sensor-level ERFs

Run the four independent phase scripts on condition-specific evokeds. Each runs the group cluster tests for its phase. Results are written to `2_ERF_sensor_level/results/sensor/`. The feedback interaction supplies Figure S1.

```sh
python 2_ERF_sensor_level/01_run_sensor_incentive.py
python 2_ERF_sensor_level/02_run_sensor_decision.py
python 2_ERF_sensor_level/03_run_sensor_transition.py
python 2_ERF_sensor_level/04_run_sensor_feedback.py
```

## 2. Source-level ERFs

In `3_ERF_source_level/cortex_mask_analysis_config.py`, set `PHASE` to one phase and run the pair below. Repeat for all four phases. The scripts select the corresponding analysis window automatically (incentive 0–1.0 s, decision 0–1.5 s, transition 0–0.5 s, feedback 0–1.5 s). Each phase includes three effects; the configured test uses 10,000 permutations, seed 42, and cluster-forming *p* = .01. Figure 3 is derived from these results in `3_ERF_source_level/results/source_cortex_mask/`.

```sh
python 3_ERF_source_level/04_source_cortex_mask_group_stats.py
python 3_ERF_source_level/05_cortex_mask_figures.py
```

## 3. Sensor time–frequency analysis

Use total band power without event-local baseline normalization. In `4_time_frequency/time_frequency_analysis/tfr_analysis_config.py`, choose `PHASE`. Run `02_group_cluster_test.py` once per phase. It crosses three effects with theta (4–7 Hz), alpha (8–12 Hz), beta (13–29 Hz), and low gamma (30–38 Hz). After all 48 phase–band–effect combinations are available, run the plotting script once for Figure 4. Results are in `4_time_frequency/time_frequency_analysis/results/sensor/`. To derive band power from the long epochs, first run `01_compute_subject_band_power.py` in the same directory.

```sh
python 4_time_frequency/time_frequency_analysis/02_group_cluster_test.py
python 4_time_frequency/time_frequency_analysis/03_plot_phase_results.py
```

## 4. Brain–behavior regression

Complete the source and time–frequency group statistics first. The extractor reads the selected clusters and calculates participant-level MEG effects. The two R batch scripts use the saved behavioral HBRL fit in `../BHV/3_HBRLmodel/` and the 48-participant covariate table to regress behavioral parameters on neural effects, adjusting for age and sex. The final R script produces the summary and scatter plots and calls the mask renderer by default. Results are under `5_BHV_MEG_regression/results/`; Figure 5 reports the MB associations.

```sh
python 5_BHV_MEG_regression/01_extract_meg_cluster_subject_effects.py
Rscript 5_BHV_MEG_regression/02_batch_regress_behavior_from_meg_planned_effects_covariates_phasewise_mb.R
Rscript 5_BHV_MEG_regression/02_batch_regress_behavior_from_meg_planned_effects_covariates_phasewise_mf.R
Rscript 5_BHV_MEG_regression/03_plot_behavior_from_meg_regression_results.R
```

## 5. Source-space searchlight RSA

The first script builds behavioral representational dissimilarity matrices (RDMs) from `mb_estimates_hdi.csv` and `mf_estimates_hdi.csv` in the behavioral HBRL directory. The searchlight uses source data and four phase-specific reference statistics: incentive/reward, decision/load, transition/load, and feedback/reward. Run the three scripts in order. Figure 6 outputs are under `6_RSA/results/searchlight_seed42/`.

```sh
python 6_RSA/01_build_behavior_rdms.py
python 6_RSA/02_compute_searchlight_seed42.py
python 6_RSA/03_plot_searchlight_mb_mf_seed42.py
```

## Starting earlier in the pipeline

`1_preprocess/` documents the acquisition-level route: run combination/ICA (01), ICA cleaning (02), epoch construction (03), trigger–metadata checks (04), bad-channel interpolation (05), resampling (06), reaction-time checks (07), and condition averaging (08). `03_epochs.py --kind erf` builds short ERF epochs, `--kind tfa` invokes the long-epoch builders, and `--kind all` runs both. Source scripts 01–03 cover coregistration, forward modeling, and individual source estimation/morphing. These upstream routes require the corresponding recordings, anatomical inputs, and participant-specific processing information; the supplied derivatives permit entry directly at sections 1–3 above.
