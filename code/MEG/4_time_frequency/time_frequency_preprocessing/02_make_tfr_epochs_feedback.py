"""Build long TFR epochs for reconstructed feedback-image events.

This script is intentionally separate from 01_make_tfr_epochs.py because
feedback triggers are not written into the raw UPPT001
channel. They are stored in the supplied reconstructed event tables.

Use the supplied aligned epochs for subjects 07 and 14.

The raw MEG files and original behavior files are not modified.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from meg_project import DATA_ROOT, available

import json

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from tfr_preprocessing_config import (  # noqa: E402
    BEHAVIOR_ROOT,
    DETREND,
    EXPECTED_TRIALS,
    OUTPUT_ROOT,
    REJECT_CRITERIA,
    SUBJECTS,
    TARGET_SFREQ,
    behavior_path,
    raw_path,
)


RECONSTRUCTED_EVENT_ROOT = DATA_ROOT / "MEG" / "preprocessed" / "2_ICA" / "reconstructed_events"

PHASE = "feedback"
CONDITIONS = ("51", "52", "53", "54")
INCENTIVE_CODES = np.array([1, 2, 3, 4], dtype=int)
# Keep post-event padding so feedback analyses can test 0-1.5 s
# without placing the final tested samples directly on the epoch boundary.
EPOCH_WINDOW = (-1.0, 2.0)
OVERWRITE = True



def reconstructed_events_path(subject):
    return (
        RECONSTRUCTED_EVENT_ROOT
        / f"sub-{subject}"
        / f"sub-{subject}_events_with_feedback_image_reconstructed.npy"
    )


def output_epoch_path(subject):
    return (
        OUTPUT_ROOT
        / f"sub-{subject}"
        / f"sub-{subject}_{PHASE}_tfr-epo.fif"
    )


def load_behavior(subject):
    path = behavior_path(subject)
    if not path.exists():
        raise FileNotFoundError(f"Missing behavior file: {path}")

    behavior = pd.read_csv(path)
    required = {"trials_globe", "miss"}
    missing = required.difference(behavior.columns)
    if missing:
        raise RuntimeError(f"Behavior columns are missing: {sorted(missing)}")

    behavior = behavior.copy()
    behavior["trials_globe"] = pd.to_numeric(
        behavior["trials_globe"], errors="raise"
    ).astype(int)
    behavior = behavior.sort_values("trials_globe").reset_index(drop=True)

    expected_ids = np.arange(1, EXPECTED_TRIALS + 1)
    if not np.array_equal(behavior["trials_globe"].to_numpy(), expected_ids):
        raise RuntimeError(
            "Behavior trial IDs are not the complete ordered range "
            f"1-{EXPECTED_TRIALS} before correction."
        )

    return behavior, np.nan


def load_reconstructed_events(subject):
    path = reconstructed_events_path(subject)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing reconstructed event file: {path}. "
            "Supply the reconstructed event table for this subject."
        )
    events = np.load(path).astype(int)
    if events.ndim != 2 or events.shape[1] != 3:
        raise RuntimeError(f"Unexpected event-array shape: {events.shape}")
    return events[np.argsort(events[:, 0])]


def condition_metadata(row, condition, corrected_row, event_sample, dropped_trial):
    ordered = tuple(str(code) for code in CONDITIONS)
    condition = str(int(condition))
    condition_index = ordered.index(condition)
    metadata = row.to_dict()
    metadata.update(
        {
            "phase": PHASE,
            "condition_code": condition,
            "condition_index": condition_index,
            "magnitude_level": "small" if condition_index in (0, 2) else "large",
            "load_level": "low" if condition_index in (0, 1) else "high",
            "event_source": "feedback_text_minus_0.5s",
            "corrected_row": int(corrected_row),
            "event_sample": int(event_sample),
            "dropped_behavior_trial_for_alignment": dropped_trial,
            "local_baseline_applied": False,
        }
    )
    return metadata


def collect_feedback_image_events(raw, events, behavior, dropped_trial):
    incentive_events = events[np.isin(events[:, 2], INCENTIVE_CODES)]
    if len(incentive_events) != len(behavior):
        raise RuntimeError(
            "MEG incentive events and behavior rows do not match after correction: "
            f"incentive_events={len(incentive_events)}, behavior={len(behavior)}."
        )

    target_codes = np.array([int(code) for code in CONDITIONS], dtype=int)
    selected_events = []
    metadata_rows = []
    event_qc_rows = []

    for row_index, incentive_event in enumerate(incentive_events):
        row = behavior.iloc[row_index]
        original_trial = int(row["trials_globe"])
        corrected_row = row_index + 1

        trial_start = int(incentive_event[0])
        trial_stop = (
            int(incentive_events[row_index + 1, 0])
            if row_index + 1 < len(incentive_events)
            else int(raw.n_times + raw.first_samp)
        )

        qc_base = {
            "Original trial": original_trial,
            "Corrected row": corrected_row,
            "Behavior miss": int(row["miss"]),
            "Dropped behavior trial": dropped_trial,
        }

        if int(row["miss"]) == 1:
            event_qc_rows.append(
                {**qc_base, "Status": "behavior_miss", "Included": False}
            )
            continue

        matches = events[
            (events[:, 0] >= trial_start)
            & (events[:, 0] < trial_stop)
            & np.isin(events[:, 2], target_codes)
        ]
        if len(matches) != 1:
            event_qc_rows.append(
                {
                    **qc_base,
                    "Status": "missing_trigger" if len(matches) == 0 else "duplicate_trigger",
                    "Matches": int(len(matches)),
                    "Included": False,
                }
            )
            continue

        event = matches[0].astype(int)
        selected_events.append(event)
        metadata_rows.append(
            condition_metadata(
                row,
                condition=event[2],
                corrected_row=corrected_row,
                event_sample=event[0],
                dropped_trial=dropped_trial,
            )
        )
        event_qc_rows.append(
            {
                **qc_base,
                "Status": "included",
                "Matches": 1,
                "Event sample": int(event[0]),
                "Event code": int(event[2]),
                "Included": True,
            }
        )

    if not selected_events:
        raise RuntimeError("No usable reconstructed feedback-image events were found.")

    return (
        np.asarray(selected_events, dtype=int),
        pd.DataFrame(metadata_rows),
        event_qc_rows,
    )


def count_conditions(epochs):
    return {
        condition: len(epochs[condition]) if condition in epochs.event_id else 0
        for condition in CONDITIONS
    }


def process_subject(subject):
    source = raw_path(subject)
    if not source.exists():
        raise FileNotFoundError(f"Missing ICA-clean raw file: {source}")

    behavior, dropped_trial = load_behavior(subject)
    events = load_reconstructed_events(subject)
    raw = mne.io.read_raw_fif(source, preload=False, verbose=False)

    picks = mne.pick_types(
        raw.info,
        meg="mag",
        eeg=False,
        stim=False,
        exclude="bads",
    )
    if len(picks) == 0:
        raise RuntimeError("No good magnetometer channels were available.")

    phase_events, metadata, event_qc = collect_feedback_image_events(
        raw, events, behavior, dropped_trial
    )

    event_id = {condition: int(condition) for condition in CONDITIONS}
    epochs = mne.Epochs(
        raw,
        phase_events,
        event_id=event_id,
        tmin=EPOCH_WINDOW[0],
        tmax=EPOCH_WINDOW[1],
        picks=picks,
        baseline=None,
        detrend=DETREND,
        preload=True,
        reject_by_annotation=True,
        metadata=metadata,
        verbose=False,
    )
    counts_before = count_conditions(epochs)
    epochs.drop_bad(reject=REJECT_CRITERIA, verbose=False)
    counts_after = count_conditions(epochs)
    epochs.resample(TARGET_SFREQ, npad="auto", verbose=False)

    destination = output_epoch_path(subject)
    destination.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(destination, overwrite=OVERWRITE, verbose=False)

    rows = []
    for condition in CONDITIONS:
        rows.append(
            {
                "Subject": subject,
                "Phase": PHASE,
                "Condition": condition,
                "Before amplitude rejection": counts_before[condition],
                "After amplitude rejection": counts_after[condition],
                "Rejected": counts_before[condition] - counts_after[condition],
                "Channels": len(epochs.ch_names),
                "Sampling rate (Hz)": epochs.info["sfreq"],
                "Epoch tmin (s)": epochs.tmin,
                "Epoch tmax (s)": epochs.tmax,
                "Local baseline applied": False,
                "Dropped behavior trial": dropped_trial,
                "Output": str(destination),
            }
        )
    return rows, event_qc


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    qc_dir = OUTPUT_ROOT / "qc" / PHASE
    qc_dir.mkdir(parents=True, exist_ok=True)

    count_rows = []
    event_rows = []
    errors = []

    for subject in tqdm(SUBJECTS, desc=f"Building {PHASE} TFR epochs"):
        # These subjects use the supplied aligned epochs.
        if subject in {"07", "14"}:
            continue
        try:
            counts, event_qc = process_subject(subject)
            count_rows.extend(counts)
            event_rows.extend({"Subject": subject, **row} for row in event_qc)
        except Exception as error:
            errors.append({"Subject": subject, "Error": repr(error)})
            print(f"sub-{subject} failed: {error}")

    pd.DataFrame(count_rows).to_csv(
        qc_dir / f"{PHASE}_tfr_epoch_condition_counts.csv",
        index=False,
    )
    pd.DataFrame(event_rows).to_csv(
        qc_dir / f"{PHASE}_tfr_event_alignment_qc.csv",
        index=False,
    )
    (qc_dir / f"{PHASE}_tfr_epoch_errors.json").write_text(
        json.dumps(errors, indent=2),
        encoding="utf-8",
    )

    print(f"Epoch output: {OUTPUT_ROOT}")
    print(f"QC output: {qc_dir}")
    if errors:
        raise RuntimeError(
            f"{PHASE} TFR epoch construction failed for {len(errors)} subjects. "
            f"Inspect {qc_dir / f'{PHASE}_tfr_epoch_errors.json'}."
        )


if __name__ == "__main__":
    main()
