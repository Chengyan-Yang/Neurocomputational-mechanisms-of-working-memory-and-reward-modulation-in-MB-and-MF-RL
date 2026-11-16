# MEG analysis data

## Overview

This dataset contains the MEG statistics and plotting values for 48 participants used in Figures 3–6, Supplementary Figure S1 and Tables S6–S9. Condition labels give load first and reward magnitude second (`low_small`, `low_large`, `high_small`, `high_large`). Standard errors of the mean (SEM) are calculated across participants. Array indices such as `cluster_index` and `space_index` start at zero. Reported cluster numbers start at one.

## Data directories

### `sensor_erf/` — Sensor-level ERFs (Figure S1 and Table S6)

Table S6 covers the four phases. Figure S1 shows the feedback interaction at 545–630 ms (41 sensors, cluster-level FWE-corrected *p* = .042). Panel A contains condition means and SEM in fT. For the display traces, channels were equalised across participants, low-pass filtered at 20 Hz and cropped to −0.2–1.495 s. Sensor polarity was aligned to the peak effect before averaging within participants. No additional baseline correction was applied.

In panel B, each sensor value is the mean *t* across the cluster time samples. `in_cluster` marks sensors with at least one cluster member. Layout coordinates and the plotting sphere are in metres. Panel C uses unfiltered values at the exact sensor–time points in the cluster, with the original sign alignment. Each `panel_pair` links four condition values within this panel. The simple-effects file gives uncorrected and two-comparison Bonferroni-adjusted *p* values.

### `source_erf/` — Source-level ERFs (Figure 3 and Table S7)

`Fig_3_panel_index.csv` identifies nine plotted clusters by panel, effect, hemisphere, time window and region. Table S7 covers all 20 reported clusters. The NPZ archives contain template vertex IDs, times in seconds, sparse cluster indices and signed *t* values. `space_index` addresses the concatenated left- and right-hemisphere vertex arrays, left first.

The `mean_t` arrays contain the plotted group statistic averaged over cluster time samples in each bin, with zeros outside the cluster. Each archive includes the full-window map and any additional display bins. `saved_time_bins` gives the archived bin count, and the `time_s` arrays identify the bins. Colour limits use the 50th, 75th and 98th percentiles of absolute nonzero mean values across significant clusters for the same phase, effect and binning scheme. Figure 3e also includes four-condition dSPM means, SEM, paired observations and simple effects. Its pairing keys are local to that panel.

### `time_frequency/` — Sensor time–frequency analysis (Figure 4 and Table S8)

`cluster_key` links curves, topographies, significant time intervals and statistics for 23 clusters. Condition and contrast curves include pre-event samples. Contrast means and SEM are calculated across the 48 within-participant contrasts. Power is expressed in dB without event-local baseline normalisation.

Significant intervals preserve gaps in cluster membership. Each topographic value is the mean *t* across that sensor’s cluster-member samples, with zeros outside the cluster. Masks, ordered channel names, projected coordinates and plotting spheres accompany the values. Phase-wise and study-wide corrections use 12 and 48 tests, respectively. Use the unrounded *p* values to reproduce significance colouring.

### `brain_behavior/` — Brain–behaviour associations (Figure 5 and Table S9)

`Fig_5a_MB_coefficients.csv` contains 36 rows for the effect-matched and baseline MB forest plots. For the six associations surviving phase-wise FDR correction, scatter coordinates are residuals obtained by regressing each standardised variable on an intercept, standardised age and sex. Points are sorted by the horizontal coordinate within each panel and have no participant identifiers. Fitted lines are supplied at 80 positions with 95% confidence intervals for the mean.

Use `scatter_key` and `map_key` to match points, lines and insets. Sensor insets include the binary mask and coordinates projected using a 0.12 m sphere. The source inset gives binary membership in concatenated template space.

### `rsa/` — Searchlight RSA (Figure 6)

Eight phase-by-model archives contain group mean Spearman correlations, tested locations and significant cluster membership (cluster *p* < .05). In the full vertex array, the left hemisphere precedes the right. `tested_space_index` locates searchlight centres in that array. `cluster_member_center_index` indexes the tested centres, and `cluster_member_cluster_index` identifies their clusters. `masked_full_mean_rho` contains the surface map with zeros elsewhere. The inflated-surface display uses a transparent background, low-contrast cortex and a red-to-yellow scale (`#D73027` to `#FFD700`) with limits of 0, .25 and .5.

The representative matrices cover MB for four phases and MF for three feedback clusters. Mirror the six unique distances to obtain a symmetric 4 × 4 matrix with a zero diagonal. The RDM statistics table distinguishes the cluster mean correlation (`cluster_mean_rho`) from the correlation between group RDMs (`group_rdm_spearman_rho`).

### `shared/` — Surface template

`shared/fsaverage/surf/` contains standard FreeSurfer inflated, pial, white and sphere surfaces and curvature files for both hemispheres. For rendering, use `shared/` as `subjects_dir` and `fsaverage` as the subject. Sensor layouts are included with each topography.

## Reproduction

Follow the [MEG group-analysis guide](../../code/MEG_for_group_data/README.md) to verify the files against the byte sizes and SHA-256 digests in `manifest.csv`, regenerate the tables and export individual figure panels. Final figure assembly is a separate layout step. Paired observations and adjusted scatter coordinates can be used within their respective panels. The dataset provides no cross-figure participant key.

The `Table_S*_as_reported.csv` files transcribe the supplementary tables, including their rounding and blank repeated labels. Use `cluster_statistics.csv` and `Fig_5a_MB_coefficients.csv` for analysis values. `reported_p_value_differences.csv` lists 18 adjusted *p* values that differ from the manuscript beyond its stated precision or reporting bound. All retain the same significance decision at .05.
