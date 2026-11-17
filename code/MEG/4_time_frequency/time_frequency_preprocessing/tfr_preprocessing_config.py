"""Configuration for rebuilding epochs for sensor-level time-frequency work."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from meg_project import DATA_ROOT, meg_subjects, PREP_RESULTS, TFR_PREP_RESULTS



RAW_ROOT = PREP_RESULTS / "2_ICA" / "cleaned"
BEHAVIOR_ROOT = DATA_ROOT / "BHV" / "2_stage_task" / "preprocessed" / "learning" / "all" / "RL_code"
OUTPUT_ROOT = TFR_PREP_RESULTS / "time_frequency_epoch"

SUBJECTS = meg_subjects()

# Data-stage names are preserved. Decision and feedback use dedicated builders.
DIRECT_PHASES = ("incentive", "transition")
ANALYZED_PHASES = ('incentive', 'decision', 'transition', 'feedback')

PHASE_CONDITIONS = {'incentive': ('1', '2', '3', '4'),
 'decision': ('11', '21', '31', '41'),
 'transition': ('13', '23', '33', '43'),
 'feedback': ('51', '52', '53', '54')}

# Long epochs provide wavelet padding. Group statistics use shorter windows.
EPOCH_WINDOWS = {'incentive': (-1.0, 1.5),
 'decision': (-1.0, 2.5),
 'transition': (-1.0, 1.0),
 'feedback': (-1.0, 2.0)}

EXPECTED_TRIALS = 320
TARGET_SFREQ = 200.0
DETREND = 1
REJECT_CRITERIA = {"mag": 3000e-15}
OVERWRITE = False


def raw_path(subject):
    return RAW_ROOT / f"sub-{subject}_clean_raw.fif"


def behavior_path(subject):
    return BEHAVIOR_ROOT / f"{subject}_RL_prep.csv"


def epoch_path(subject, phase):
    return (
        OUTPUT_ROOT
        / f"sub-{subject}"
        / f"sub-{subject}_{phase}_tfr-epo.fif"
    )


def validate_config():
    if set(ANALYZED_PHASES) != set(PHASE_CONDITIONS):
        raise ValueError("ANALYZED_PHASES and PHASE_CONDITIONS do not match.")
    if set(ANALYZED_PHASES) != set(EPOCH_WINDOWS):
        raise ValueError("ANALYZED_PHASES and EPOCH_WINDOWS do not match.")
    if TARGET_SFREQ < 2 * 40:
        raise ValueError("TARGET_SFREQ is too low for frequencies near 40 Hz.")


validate_config()
