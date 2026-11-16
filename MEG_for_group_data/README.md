# Reproducing MEG figures and tables from released group data

This workflow regenerates the numerical tables and individual figure panels for Figures 3–6, Supplementary Figure S1, and Tables S6–S9 from the saved results for 48 participants. To repeat the cluster permutations, regression analyses, or searchlight computations from participant-level inputs, follow [the full MEG workflow](../MEG/README.md).

## Inputs and software

Keep this directory at `code/MEG_for_group_data/` alongside the project’s `group_data/MEG/` tree. The latter contains sensor ERF, source ERF, time–frequency, brain–behavior, and RSA data, together with the standard `fsaverage` surfaces. The [data dictionary](../../group_data/MEG/README.md) describes units, condition order, masks, time bins, and pairing. Subject-level pairing keys are local to their panels.

Use Python with NumPy, pandas, SciPy, Matplotlib, seaborn, MNE-Python, Pillow, PyVista, pyvistaqt, and a compatible Qt binding. Surface rendering requires working OpenGL support. Figure 5 also uses R with dplyr, ggplot2, and Cairo PDF support. Run the commands below from `code/MEG_for_group_data/`.

## Verify the release and reproduce the tables

```sh
python 00_verify_data.py
```

The script verifies the files listed in `manifest.csv` by size and SHA-256 and checks the phase-wise Bonferroni calculations. It writes analysis-side values to `results/tables/`: S6 (18 sensor clusters), S7 (20 source clusters), S8 (23 time–frequency clusters), and S9 (six MB regression associations). For comparisons with the supplied manuscript transcription, consult `group_data/MEG/reported_p_value_differences.csv`; the analysis tables retain the numerical values from the saved results.

## Reproduce the figure panels

```sh
python 01_sensor_erf.py
python 02_source_erf.py
python 03_time_frequency.py
Rscript 04_brain_behavior.R
python 04_brain_behavior.py
python 05_rsa.py
```

For Figure S1, the sensor ERF script reads `sensor_erf/` and plots the feedback interaction at 545–630 ms. It saves the four condition waveforms, both with and without SEM, the sensor topography, and the paired interaction plot in `results/Fig_S1/`.

For Figure 3, the source ERF script maps nine selected clusters using the time bins and color limits stored in `source_erf/`. It saves the cortical maps, separate color bars, and the transition interaction plot in `results/Fig_3/`.

For Figure 4, the time–frequency script reads `time_frequency/` and produces three panels for each of 23 clusters. These show the four condition curves, the effect contrast, and the sensor topography. The 69 PDFs are organized by phase, frequency band, and effect in `results/Fig_4/`.

For Figure 5, both scripts read `brain_behavior/`. The R script produces the MB forest plot and six scatter plots adjusted for age and sex. The Python script draws the corresponding sensor and source masks. Both save their outputs in `results/Fig_5/`, where `Fig_5bc_index.csv` matches each scatter plot to its mask.

For Figure 6, the RSA script reads `rsa/` and saves cluster locations, MB and MF cortical maps for all four phases, and seven representative pairs of neural and behavioral dissimilarity matrices in `results/Fig_6/`.

Each figure can be generated independently after verification. Two-dimensional panels are exported as PDFs and cortical maps as transparent 600-dpi PNGs. Figure assembly in the manuscript is a separate layout step.
