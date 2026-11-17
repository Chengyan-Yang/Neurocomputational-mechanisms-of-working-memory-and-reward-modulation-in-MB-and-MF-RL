"""Configuration shared by cortex-masked group statistics and figures."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meg_project import DATA_ROOT, meg_subjects, SOURCE_RESULTS, source_stem



# Select one analysis phase here. Output names match the downstream analyses.
PHASE = "feedback"
PHASE_SETTINGS = {
    # analysis phase: (STC folder, window in seconds, ordered condition codes)
    "incentive": ("incentive", (0.0, 1.0), ("1", "2", "3", "4")),
    "decision": ("decision", (0.0, 1.5), ("11", "21", "31", "41")),
    "transition": ("transition", (0.0, 0.5), ("13", "23", "33", "43")),
    "feedback": ("feedback", (0.0, 1.5), ("51", "52", "53", "54")),
}
if PHASE not in PHASE_SETTINGS:
    raise ValueError(f"Unknown phase {PHASE!r}. Choose from {tuple(PHASE_SETTINGS)}.")
DATA_PHASE, (ANALYSIS_TMIN, ANALYSIS_TMAX), CONDITION_CODES = PHASE_SETTINGS[PHASE]
N_TIME_BINS = 1

# Conditions are ordered as small reward/low-load, large reward/low-load,
# small reward/high-load, and large reward/high-load.
EFFECT_WEIGHTS = {
    "load": (-0.5, -0.5, 0.5, 0.5),
    "magnitude": (-0.5, 0.5, -0.5, 0.5),
    "interaction": (0.5, -0.5, -0.5, 0.5),
}

EFFECT_LABELS = {
    "load": "High load - Low load",
    "magnitude": "Large reward - Small reward",
    "interaction": "Load x Reward interaction",
}


SUBJECTS_DIR = DATA_ROOT / "MRI" / "surf"

# These are the original STCs computed with the standard, unrestricted surface
# inverse solution. The cortex mask is applied only at group statistics.
MORPHED_DIR = DATA_ROOT / "MEG" / "source" / "morphed_stcs"

OUTPUT_ROOT = SOURCE_RESULTS / "source_cortex_mask"

TEMPLATE_SUBJECT = "fsaverage"
TEMPLATE_SPACING = "ico5"

CLUSTER_FORMING_P = 0.01
CLUSTER_ALPHA = 0.05
N_PERMUTATIONS = 10000
RANDOM_SEED = 42
N_JOBS = 4
BUFFER_SIZE = 256

SUBJECTS = meg_subjects()


# Figure settings used by code 8.
BRAIN_SURFACE = "inflated"
BRAIN_CORTEX = "low_contrast"
BRAIN_BACKGROUND = "white"
BRAIN_SIZE = (800, 800)
VIEWS = ("lateral", "medial")


def validate_config():
    if ANALYSIS_TMIN >= ANALYSIS_TMAX:
        raise ValueError("ANALYSIS_TMIN must be smaller than ANALYSIS_TMAX.")
    if N_TIME_BINS < 1:
        raise ValueError("N_TIME_BINS must be at least 1.")


def conditions():
    return CONDITION_CODES


def stc_stem(subject, condition):
    return source_stem(MORPHED_DIR, subject, DATA_PHASE, condition)


def output_dir(effect):
    return OUTPUT_ROOT / PHASE / effect


def result_path(effect):
    return output_dir(effect) / "Cluster_permutation_results_10000_seed42.pkl"


validate_config()
