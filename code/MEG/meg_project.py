"""Shared locations and participant identifiers for the curated MEG project."""
import csv
from pathlib import Path

MEG_CODE = Path(__file__).resolve().parent
DATA_ROOT = MEG_CODE.parent.parent / "data"
PREP_RESULTS = MEG_CODE / "1_preprocess" / "results"
SOURCE_RESULTS = MEG_CODE / "3_ERF_source_level" / "results"
TFR_PREP_RESULTS = MEG_CODE / "4_time_frequency" / "time_frequency_preprocessing" / "results"
TFR_RESULTS = MEG_CODE / "4_time_frequency" / "time_frequency_analysis" / "results"
RSA_RESULTS = MEG_CODE / "6_RSA" / "results"


def available(generated, supplied):
    """Prefer a rebuilt input when present, otherwise use the supplied copy."""
    return generated if generated.exists() else supplied


def project_input(value):
    """Resolve a research input inside the submission data or code directory."""
    path = Path(value).resolve()
    if not any(path.is_relative_to(root) for root in (DATA_ROOT, MEG_CODE.parent)):
        raise ValueError(f"Input must be under {DATA_ROOT} or {MEG_CODE.parent}: {path}")
    return path


def behavior_rdm(model):
    if model not in ("mb", "mf"):
        raise ValueError(f"Unknown behavior model: {model}")
    return RSA_RESULTS / "searchlight_seed42" / "behavior_rdms" / f"hbrl_{model}_distances.csv"


def meg_subjects():
    """Read the curated cohort, retaining gaps in the behavioral identifiers."""
    path = MEG_CODE / "5_BHV_MEG_regression" / "subject_covariates_48subs.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        ids = [int(row["subid"]) for row in csv.DictReader(stream)]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Missing or duplicate subject IDs in {path}")
    return tuple(f"{value:02d}" for value in sorted(ids))


def source_stem(root, subject, phase, condition):
    relative = Path(f"sub-{subject}") / phase / f"{subject}_{condition}-morphed-stc.h5"
    if root == DATA_ROOT / "MEG" / "source" / "morphed_stcs":
        rebuilt = SOURCE_RESULTS / "morphed_stcs" / relative
        if all(Path(f"{rebuilt}-{side}.stc").is_file() for side in ("lh", "rh")):
            return rebuilt
    return root / relative
