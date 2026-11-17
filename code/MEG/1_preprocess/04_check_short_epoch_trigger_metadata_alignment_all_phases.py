# -*- coding: utf-8 -*-
"""Check MEG-trigger and behavior-metadata alignment in short ERF epochs.

The check is intentionally independent from the epoch-construction scripts. It
loads existing epoch files and compares each epoch's MEG event code with the
trigger code expected from the trial metadata.

By default, it scans all subjects and all short-epoch phases, including repaired,
reconstructed, and analysis-label phases:

    incentive, stage1, choose, transition, feedback_image, feedback,
    feedback_text, rest, decision, decision, feedback

Missing epoch files are reported as skipped. Mismatched rows are exported so
that sub-07 and sub-14 can be inspected after alignment repair.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meg_project import DATA_ROOT, meg_subjects, PREP_RESULTS


import mne
import numpy as np
import pandas as pd


mne.set_log_level("WARNING")


# =========================
# Configuration
# =========================
SUBJECTS = meg_subjects()

EPOCH_ROOT = PREP_RESULTS / "3_epochs"

OUTPUT_DIR = EPOCH_ROOT / "qc" / "trigger_metadata_alignment_all_phases"

PHASE_CONDITIONS = {'incentive': [1, 2, 3, 4],
 'decision': [11, 21, 31, 41],
 'transition': [13, 23, 33, 43],
 'feedback': [51, 52, 53, 54]}

PHASES_TO_CHECK = tuple(PHASE_CONDITIONS)



def epoch_path(subject, phase):
    return EPOCH_ROOT / f"sub-{subject}" / f"sub-{subject}_{phase}_epo.fif"


def normalize_level(value):
    """Normalize common low/high encodings to 'low' or 'high'."""
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    low_values = {"low", "l", "lo", "0", "1", "1.0", "smallmagnitude", "lowload"}
    high_values = {"high", "h", "hi", "5", "5.0", "2", "2.0", "largemagnitude", "highload"}

    if text in low_values:
        return "low"
    if text in high_values:
        return "high"
    return None


def get_magnitude_and_load(row):
    """Read magnitude/load from behavior-derived metadata.

    Prefer the original behavior columns (magnitude, load). Fall back to the helper
    columns (magnitude_level, load_level) only when the original columns are absent.
    """
    if "magnitude" in row.index and "load" in row.index:
        magnitude = str(row["magnitude"]).strip().lower()
        load = normalize_level(row["load"])
        source = "magnitude/load"
    elif "magnitude_level" in row.index and "load_level" in row.index:
        magnitude = str(row["magnitude_level"]).strip().lower()
        load = normalize_level(row["load_level"])
        source = "magnitude_level/load_level"
    else:
        magnitude = None
        load = None
        source = "missing"

    return magnitude, load, source


def condition_index_from_metadata(row):
    magnitude, load, source = get_magnitude_and_load(row)
    if magnitude is None or load is None:
        return np.nan, source

    if magnitude == "small" and load == "low":
        return 0, source
    if magnitude == "large" and load == "low":
        return 1, source
    if magnitude == "small" and load == "high":
        return 2, source
    if magnitude == "large" and load == "high":
        return 3, source
    return np.nan, source


def expected_trigger_from_metadata(row, phase):
    condition_index, source = condition_index_from_metadata(row)
    if not np.isfinite(condition_index):
        return np.nan, source

    conditions = PHASE_CONDITIONS[phase]
    return int(conditions[int(condition_index)]), source


def check_one_epoch_file(subject, phase):
    path = epoch_path(subject, phase)
    if not path.exists():
        return {
            "summary": {
                "Subject": subject,
                "Phase": phase,
                "Data phase": phase,
                "Status": "skipped_missing_file",
                "N epochs": 0,
                "N matched": np.nan,
                "N mismatched": np.nan,
                "Match rate": np.nan,
                "Epoch file": str(path),
            },
            "mismatches": [],
        }

    epochs = mne.read_epochs(path, preload=False, verbose=False)
    if epochs.metadata is None:
        raise RuntimeError(f"No metadata found in {path}")

    metadata = epochs.metadata.copy().reset_index(drop=True)
    if len(metadata) != len(epochs.events):
        raise RuntimeError(
            f"Metadata/event length mismatch in {path}: "
            f"metadata={len(metadata)}, events={len(epochs.events)}"
        )

    meg_triggers = epochs.events[:, 2].astype(int)
    expected = []
    metadata_sources = []
    for _, row in metadata.iterrows():
        code, source = expected_trigger_from_metadata(row, phase)
        expected.append(code)
        metadata_sources.append(source)

    expected = np.asarray(expected, dtype=float)
    comparable = np.isfinite(expected)
    matched = comparable & (meg_triggers == expected.astype(int))
    mismatched = comparable & ~matched
    unknown_expected = ~comparable

    summary = {
        "Subject": subject,
        "Phase": phase,
        "Data phase": phase,
        "Status": "checked",
        "N epochs": int(len(meg_triggers)),
        "N comparable": int(np.sum(comparable)),
        "N matched": int(np.sum(matched)),
        "N mismatched": int(np.sum(mismatched)),
        "N unknown expected trigger": int(np.sum(unknown_expected)),
        "Match rate": (
            float(np.mean(matched[comparable])) if np.any(comparable) else np.nan
        ),
        "Epoch file": str(path),
    }

    mismatch_rows = []
    bad_indices = np.where(mismatched | unknown_expected)[0]
    for idx in bad_indices:
        row = metadata.iloc[idx].to_dict()
        mismatch_rows.append(
            {
                "Subject": subject,
                "Phase": phase,
                "Data phase": phase,
                "Epoch index": int(idx),
                "trials_globe": row.get("trials_globe", np.nan),
                "MEG trigger": int(meg_triggers[idx]),
                "Expected trigger": expected[idx],
                "Metadata source": metadata_sources[idx],
                "magnitude": row.get("magnitude", np.nan),
                "load": row.get("load", np.nan),
                "magnitude_level": row.get("magnitude_level", np.nan),
                "load_level": row.get("load_level", np.nan),
                "miss": row.get("miss", np.nan),
                "condition_code": row.get("condition_code", np.nan),
                "event_sample": row.get("event_sample", np.nan),
                "Epoch file": str(path),
            }
        )

    return {"summary": summary, "mismatches": mismatch_rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    mismatches = []
    errors = []

    for subject in SUBJECTS:
        print(f"Checking sub-{subject}")
        for phase in PHASES_TO_CHECK:
            try:
                result = check_one_epoch_file(subject, phase)
                summaries.append(result["summary"])
                mismatches.extend(result["mismatches"])
            except Exception as error:
                errors.append(
                    {
                        "Subject": subject,
                        "Phase": phase,
                        "Error": repr(error),
                        "Epoch file": str(epoch_path(subject, phase)),
                    }
                )
                print(f"  {phase} failed: {error}")

    summary = pd.DataFrame(summaries)
    mismatch_table = pd.DataFrame(mismatches)
    error_table = pd.DataFrame(errors)

    summary_path = OUTPUT_DIR / "short_epoch_trigger_metadata_alignment_summary.csv"
    mismatch_path = OUTPUT_DIR / "short_epoch_trigger_metadata_alignment_mismatches.csv"
    error_path = OUTPUT_DIR / "short_epoch_trigger_metadata_alignment_errors.csv"

    summary.to_csv(summary_path, index=False)
    mismatch_table.to_csv(mismatch_path, index=False)
    error_table.to_csv(error_path, index=False)

    checked = summary[summary["Status"] == "checked"]
    total_mismatches = (
        int(checked["N mismatched"].fillna(0).sum()) if not checked.empty else 0
    )
    total_unknown = (
        int(checked["N unknown expected trigger"].fillna(0).sum())
        if not checked.empty
        else 0
    )

    print("\nDone.")
    print(f"Summary: {summary_path}")
    print(f"Mismatches: {mismatch_path}")
    print(f"Errors: {error_path}")
    print(f"Total mismatches: {total_mismatches}")
    print(f"Total unknown expected triggers: {total_unknown}")

    if total_mismatches == 0 and total_unknown == 0 and error_table.empty:
        print("All checked epoch files passed trigger-metadata alignment.")
    else:
        print("Inspect the mismatch/error tables before using these epochs.")


if __name__ == "__main__":
    main()
