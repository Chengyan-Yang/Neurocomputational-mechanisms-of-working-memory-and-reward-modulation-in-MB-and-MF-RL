"""Shared configuration for sensor-level time-frequency analyses."""

from pathlib import Path
from functools import lru_cache
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from meg_project import DATA_ROOT, meg_subjects, TFR_PREP_RESULTS, TFR_RESULTS, available



EPOCH_ROOT = DATA_ROOT / "MEG" / "preprocessed" / "time_frequency_epoch"
POWER_ROOT = TFR_RESULTS / "sensor_band_power"
RESULTS_ROOT = TFR_RESULTS / "sensor"

SUBJECTS = meg_subjects()
@lru_cache(maxsize=1)
def trial_selection():
    """Event sample numbers selected for the balanced time-frequency analyses."""
    path = DATA_ROOT / "MEG" / "time_frequency" / "balanced_trial_samples.json"
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def selected_indices(subject, phase, code, samples, expected_count):
    selected = trial_selection()[str(subject).zfill(2)][phase][code]
    selected_set = set(selected)
    indices = [i for i, sample in enumerate(samples) if sample in selected_set]
    if len(indices) != expected_count or [int(samples[i]) for i in indices] != selected:
        raise ValueError(f"Balanced trial selection does not match {subject}, {phase}, condition {code}")
    return indices


ANALYZED_PHASES = ('incentive', 'decision', 'transition', 'feedback')

PHASE_CONDITIONS = {'incentive': ('1', '2', '3', '4'),
 'decision': ('11', '21', '31', '41'),
 'transition': ('13', '23', '33', '43'),
 'feedback': ('51', '52', '53', '54')}


# These windows are cropped only after wavelet computation on the long epochs.
ANALYSIS_WINDOWS = {'incentive': (0.0, 1.0),
 'decision': (0.0, 1.5),
 'transition': (0.0, 0.5),
 'feedback': (0.0, 1.5)}

# Plotting-only windows. The pre-event segment is shown as context and is not
# included in the group-level cluster tests unless ANALYSIS_WINDOWS is changed.
DISPLAY_WINDOWS = {'incentive': (-0.2, 1.0),
 'decision': (-0.2, 1.5),
 'transition': (-0.2, 0.5),
 'feedback': (-0.2, 1.5)}

# The final integer is the number of Morlet cycles used in that band.
BANDS = {
    "theta": {"fmin": 4, "fmax": 7, "n_cycles": 3.0},
    "alpha": {"fmin": 8, "fmax": 12, "n_cycles": 4.0},
    "beta": {"fmin": 13, "fmax": 29, "n_cycles": 6.0},
    "low_gamma": {"fmin": 30, "fmax": 38, "n_cycles": 8.0},
}

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

# Choose the phase for script 02. All four bands and three effects run automatically.
PHASE = "feedback"

# Study power settings. Script 01 can prepare all retained phases in one run.
POWER_TYPE = "total"
PHASES_TO_COMPUTE = ANALYZED_PHASES
POWER_TYPES_TO_COMPUTE = (POWER_TYPE,)

DECIM = 2
N_JOBS = 4
RANDOM_SEED = 42
N_PERMUTATIONS = 10000
CLUSTER_FORMING_P = 0.01
CLUSTER_ALPHA = 0.05

# Conservative multiplicity factors if every phase, band, and effect is tested.
N_PLANNED_PHASES = len(ANALYZED_PHASES)
N_PLANNED_BANDS = len(BANDS)
N_PLANNED_EFFECTS = len(EFFECT_WEIGHTS)


def epoch_path(subject, phase):
    supplied = (
        EPOCH_ROOT
        / f"sub-{subject}"
        / f"sub-{subject}_{phase}_tfr-epo.fif"
    )


    return available(TFR_PREP_RESULTS / "time_frequency_epoch" / supplied.relative_to(EPOCH_ROOT), supplied)


def power_output_path(subject, phase, power_type):
    return (
        POWER_ROOT
        / f"sub-{subject}"
        / phase
        / f"sub-{subject}_{phase}_{power_type}_band-power.npz"
    )


def power_path(subject, phase, power_type):
    target = power_output_path(subject, phase, power_type)
    return available(target, DATA_ROOT / "MEG" / "time_frequency" / "sensor_band_power" / target.relative_to(POWER_ROOT))


def result_dir(phase, band, effect):
    return RESULTS_ROOT / phase / POWER_TYPE / band / effect


def validate_config():
    if set(ANALYZED_PHASES) != set(PHASE_CONDITIONS):
        raise ValueError("Phase and condition definitions do not match.")
    if set(ANALYZED_PHASES) != set(ANALYSIS_WINDOWS):
        raise ValueError("Phase and analysis-window definitions do not match.")
    if set(ANALYZED_PHASES) != set(DISPLAY_WINDOWS):
        raise ValueError("Phase and display-window definitions do not match.")
    if not set(PHASES_TO_COMPUTE).issubset(ANALYZED_PHASES):
        raise ValueError("PHASES_TO_COMPUTE contains an excluded phase.")
    if not set(POWER_TYPES_TO_COMPUTE).issubset({"total", "induced"}):
        raise ValueError("Unknown power type in POWER_TYPES_TO_COMPUTE.")
    if PHASE not in ANALYZED_PHASES:
        raise ValueError(f"Unknown or excluded PHASE: {PHASE}")
    if POWER_TYPE not in {"total", "induced"}:
        raise ValueError(f"Unknown POWER_TYPE: {POWER_TYPE}")


validate_config()
