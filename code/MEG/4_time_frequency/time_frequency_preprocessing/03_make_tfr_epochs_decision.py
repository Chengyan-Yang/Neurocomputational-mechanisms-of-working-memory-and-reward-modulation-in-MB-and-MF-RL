"""Build long TFR epochs for the combined Stage1+Choose decision period.

The combined decision epoch is time-locked to Stage1/option onset and saved as
``decision``. The epoch is intentionally longer than the intended 0-2 s
analysis window to provide wavelet padding for time-frequency analysis.

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

PHASE = "decision"
INCENTIVE_CODES = np.array([1, 2, 3, 4], dtype=int)
STAGE1_CODES = np.array([11, 21, 31, 41], dtype=int)
CHOOSE_CODES = np.array([12, 22, 32, 42], dtype=int)
CONDITIONS = ("11", "21", "31", "41")

# Long TFR epoch. Group statistics can crop this to 0-2 s.
EPOCH_WINDOW = (-1.0, 2.5)
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

    expected = np.arange(1, EXPECTED_TRIALS + 1)
    if not np.array_equal(behavior["trials_globe"].to_numpy(), expected):
        raise RuntimeError(
            "Behavior trial IDs are not the complete ordered range "
            f"1-{EXPECTED_TRIALS} before correction."
        )

    return behavior, np.nan


def load_events(subject, raw):
    reconstructed = reconstructed_events_path(subject)
    if reconstructed.exists():
        events = np.load(reconstructed).astype(int)
    else:
        events = mne.find_events(
            raw,
            stim_channel="UPPT001",
            shortest_event=1,
            verbose=False,
        ).astype(int)
    if events.ndim != 2 or events.shape[1] != 3:
        raise RuntimeError(f"Unexpected event-array shape: {events.shape}")
    return events[np.argsort(events[:, 0])]


def summarize_optional_choose(stage1_event, choose_matches):
    stage1_sample = int(stage1_event[0])
    if len(choose_matches) == 1:
        choose_event = choose_matches[0].astype(int)
        choose_sample = int(choose_event[0])
        return {
            "choose_status": "present",
            "choose_event_sample": choose_sample,
            "choose_event_code": int(choose_event[2]),
            "choose_delay_samples": choose_sample - stage1_sample,
        }
    if len(choose_matches) == 0:
        status = "missing"
    else:
        status = "duplicate"
    return {
        "choose_status": status,
        "choose_event_sample": np.nan,
        "choose_event_code": np.nan,
        "choose_delay_samples": np.nan,
    }


def condition_metadata(row, condition, corrected_row, stage1_event, choose_info, dropped_trial):
    ordered = tuple(str(code) for code in CONDITIONS)
    condition = str(int(condition))
    condition_index = ordered.index(condition)
    stage1_sample = int(stage1_event[0])
    metadata = row.to_dict()
    metadata.update(
        {
            "phase": PHASE,
            "condition_code": condition,
            "condition_index": condition_index,
            "magnitude_level": "small" if condition_index in (0, 2) else "large",
            "load_level": "low" if condition_index in (0, 1) else "high",
            "event_source": "stage1_option_onset",
            "corrected_row": int(corrected_row),
            "event_sample": stage1_sample,
            **choose_info,
            "dropped_behavior_trial_for_alignment": dropped_trial,
            "local_baseline_applied": False,
        }
    )
    return metadata


def collect_decision_events(raw, events, behavior, dropped_trial):
    incentive_events = events[np.isin(events[:, 2], INCENTIVE_CODES)]
    if len(incentive_events) != len(behavior):
        raise RuntimeError(
            "MEG incentive events and behavior rows do not match after correction: "
            f"incentive_events={len(incentive_events)}, behavior={len(behavior)}."
        )

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

        stage1_matches = events[
            (events[:, 0] >= trial_start)
            & (events[:, 0] < trial_stop)
            & np.isin(events[:, 2], STAGE1_CODES)
        ]
        choose_matches = events[
            (events[:, 0] >= trial_start)
            & (events[:, 0] < trial_stop)
            & np.isin(events[:, 2], CHOOSE_CODES)
        ]

        qc_base = {
            "Original trial": original_trial,
            "Corrected row": corrected_row,
            "Behavior miss": int(row["miss"]),
            "Dropped behavior trial": dropped_trial,
            "Stage1 matches": int(len(stage1_matches)),
            "Choose matches": int(len(choose_matches)),
        }

        if int(row["miss"]) == 1:
            event_qc_rows.append(
                {**qc_base, "Status": "behavior_miss", "Included": False}
            )
            continue
        if len(stage1_matches) != 1:
            event_qc_rows.append(
                {
                    **qc_base,
                    "Status": "missing_or_duplicate_stage1",
                    "Included": False,
                }
            )
            continue

        stage1_event = stage1_matches[0].astype(int)
        choose_info = summarize_optional_choose(stage1_event, choose_matches)
        selected_events.append(stage1_event)
        metadata_rows.append(
            condition_metadata(
                row,
                condition=stage1_event[2],
                corrected_row=corrected_row,
                stage1_event=stage1_event,
                choose_info=choose_info,
                dropped_trial=dropped_trial,
            )
        )
        event_qc_rows.append(
            {
                **qc_base,
                "Status": "included",
                "Included": True,
                "Stage1 sample": int(stage1_event[0]),
                "Stage1 code": int(stage1_event[2]),
                "Choose status": choose_info["choose_status"],
                "Choose sample": choose_info["choose_event_sample"],
                "Choose code": choose_info["choose_event_code"],
                "Choose delay (s)": (
                    choose_info["choose_delay_samples"] / raw.info["sfreq"]
                    if np.isfinite(choose_info["choose_delay_samples"])
                    else np.nan
                ),
            }
        )

    if not selected_events:
        raise RuntimeError(f"No usable events were found for {PHASE}.")

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
    raw = mne.io.read_raw_fif(source, preload=False, verbose=False)
    events = load_events(subject, raw)

    picks = mne.pick_types(
        raw.info,
        meg="mag",
        eeg=False,
        stim=False,
        exclude="bads",
    )
    if len(picks) == 0:
        raise RuntimeError("No good magnetometer channels were available.")

    decision_events, metadata, event_qc = collect_decision_events(
        raw, events, behavior, dropped_trial
    )
    event_id = {condition: int(condition) for condition in CONDITIONS}
    epochs = mne.Epochs(
        raw,
        decision_events,
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
    error_path = qc_dir / f"{PHASE}_tfr_epoch_errors.json"
    error_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")

    print(f"Epoch output: {OUTPUT_ROOT}")
    print(f"QC output: {qc_dir}")
    if errors:
        raise RuntimeError(
            f"{PHASE} TFR epoch construction failed for {len(errors)} subjects. "
            f"Inspect {error_path}."
        )


if __name__ == "__main__":
    main()
