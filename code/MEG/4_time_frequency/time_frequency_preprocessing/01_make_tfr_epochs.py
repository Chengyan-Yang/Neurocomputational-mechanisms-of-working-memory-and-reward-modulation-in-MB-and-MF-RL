"""Rebuild long, non-baselined epochs for time-frequency analyses."""

import json
from pathlib import Path
import sys

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from tfr_preprocessing_config import (
    DIRECT_PHASES as ANALYZED_PHASES,
    DETREND,
    EPOCH_WINDOWS,
    EXPECTED_TRIALS,
    OUTPUT_ROOT,
    OVERWRITE,
    PHASE_CONDITIONS,
    REJECT_CRITERIA,
    SUBJECTS,
    TARGET_SFREQ,
    behavior_path,
    epoch_path,
    raw_path,
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
    if behavior["trials_globe"].duplicated().any():
        raise RuntimeError("Behavior contains duplicate trials_globe values.")
    behavior = behavior.sort_values("trials_globe").reset_index(drop=True)
    expected_ids = np.arange(1, EXPECTED_TRIALS + 1)
    if not np.array_equal(behavior["trials_globe"].to_numpy(), expected_ids):
        raise RuntimeError(
            "Behavior trial IDs are not the complete ordered range "
            f"1-{EXPECTED_TRIALS}."
        )
    return behavior


def condition_metadata(row, phase, condition):
    ordered = PHASE_CONDITIONS[phase]
    condition_index = ordered.index(str(condition))
    metadata = row.to_dict()
    metadata.update(
        {
            "phase": phase,
            "condition_code": str(condition),
            "condition_index": condition_index,
            "magnitude_level": "small" if condition_index in (0, 2) else "large",
            "load_level": "low" if condition_index in (0, 1) else "high",
        }
    )
    return metadata


def collect_phase_events(raw, behavior, phase):
    all_events = mne.find_events(
        raw,
        stim_channel="UPPT001",
        shortest_event=1,
        verbose=False,
    )
    incentive_codes = np.array([int(code) for code in PHASE_CONDITIONS["incentive"]])
    incentive_events = all_events[np.isin(all_events[:, 2], incentive_codes)]
    if len(incentive_events) != EXPECTED_TRIALS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRIALS} incentive events, found {len(incentive_events)}. "
            "Trial-to-behavior alignment would be unsafe."
        )

    target_codes = np.array([int(code) for code in PHASE_CONDITIONS[phase]])
    selected_events = []
    metadata_rows = []
    event_qc = []

    for trial_position, incentive_event in enumerate(incentive_events):
        trial_id = trial_position + 1
        row = behavior.iloc[trial_position]
        if int(row["trials_globe"]) != trial_id:
            raise RuntimeError(f"Behavior alignment failed at trial {trial_id}.")
        if int(row["miss"]) == 1:
            event_qc.append(
                {"Trial": trial_id, "Phase": phase, "Status": "behavior_miss"}
            )
            continue

        trial_start = int(incentive_event[0])
        trial_stop = (
            int(incentive_events[trial_position + 1, 0])
            if trial_position + 1 < len(incentive_events)
            else int(raw.n_times + raw.first_samp)
        )
        if phase == "incentive":
            matches = incentive_event[np.newaxis, :]
        else:
            matches = all_events[
                (all_events[:, 0] >= trial_start)
                & (all_events[:, 0] < trial_stop)
                & np.isin(all_events[:, 2], target_codes)
            ]

        if len(matches) != 1:
            event_qc.append(
                {
                    "Trial": trial_id,
                    "Phase": phase,
                    "Status": "missing_trigger" if len(matches) == 0 else "duplicate_trigger",
                    "Matches": len(matches),
                }
            )
            continue

        event = matches[0].astype(int)
        condition = str(event[2])
        selected_events.append(event)
        metadata = condition_metadata(row, phase, condition)
        metadata["meg_trial_id"] = trial_id
        metadata["event_sample"] = int(event[0])
        metadata_rows.append(metadata)
        event_qc.append(
            {"Trial": trial_id, "Phase": phase, "Status": "included"}
        )

    if not selected_events:
        raise RuntimeError(f"No usable events were found for {phase}.")
    return (
        np.asarray(selected_events, dtype=int),
        pd.DataFrame(metadata_rows),
        event_qc,
    )


def count_conditions(epochs, phase):
    counts = {}
    for condition in PHASE_CONDITIONS[phase]:
        counts[condition] = len(epochs[condition]) if condition in epochs.event_id else 0
    return counts


def process_subject(subject):
    source = raw_path(subject)
    if not source.exists():
        raise FileNotFoundError(f"Missing ICA-clean raw file: {source}")

    behavior = load_behavior(subject)
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

    subject_rows = []
    subject_event_qc = []
    for phase in ANALYZED_PHASES:
        destination = epoch_path(subject, phase)
        events, metadata, event_qc = collect_phase_events(
            raw, behavior, phase
        )
        subject_event_qc.extend(event_qc)
        if destination.exists() and not OVERWRITE:
            print(f"sub-{subject} {phase}: existing output retained")
            existing = mne.read_epochs(destination, preload=False, verbose=False)
            counts_after = count_conditions(existing, phase)
            counts_before = {
                condition: int(np.sum(events[:, 2] == int(condition)))
                for condition in PHASE_CONDITIONS[phase]
            }
            for condition in PHASE_CONDITIONS[phase]:
                subject_rows.append(
                    {
                        "Subject": subject,
                        "Phase": phase,
                        "Condition": condition,
                        "Before amplitude rejection": counts_before[condition],
                        "After amplitude rejection": counts_after[condition],
                        "Rejected": counts_before[condition] - counts_after[condition],
                        "Channels": len(existing.ch_names),
                        "Sampling rate (Hz)": existing.info["sfreq"],
                        "Epoch tmin (s)": existing.tmin,
                        "Epoch tmax (s)": existing.tmax,
                        "Local baseline applied": existing.baseline is not None,
                    }
                )
            del existing
            continue
        tmin, tmax = EPOCH_WINDOWS[phase]
        event_id = {
            condition: int(condition) for condition in PHASE_CONDITIONS[phase]
        }
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            picks=picks,
            baseline=None,
            detrend=DETREND,
            preload=True,
            reject_by_annotation=True,
            metadata=metadata,
            verbose=False,
        )
        counts_before = count_conditions(epochs, phase)
        epochs.drop_bad(reject=REJECT_CRITERIA, verbose=False)
        counts_after = count_conditions(epochs, phase)
        epochs.resample(TARGET_SFREQ, npad="auto", verbose=False)

        destination.parent.mkdir(parents=True, exist_ok=True)
        epochs.save(destination, overwrite=True, verbose=False)
        for condition in PHASE_CONDITIONS[phase]:
            subject_rows.append(
                {
                    "Subject": subject,
                    "Phase": phase,
                    "Condition": condition,
                    "Before amplitude rejection": counts_before[condition],
                    "After amplitude rejection": counts_after[condition],
                    "Rejected": counts_before[condition] - counts_after[condition],
                    "Channels": len(epochs.ch_names),
                    "Sampling rate (Hz)": epochs.info["sfreq"],
                    "Epoch tmin (s)": epochs.tmin,
                    "Epoch tmax (s)": epochs.tmax,
                    "Local baseline applied": False,
                }
            )
        del epochs

    return subject_rows, subject_event_qc


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    qc_dir = OUTPUT_ROOT / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    count_rows = []
    event_rows = []
    errors = []

    for subject in tqdm(SUBJECTS, desc="Building TFR epochs"):
        # These subjects use the supplied aligned epochs.
        if subject in {"07", "14"}:
            continue
        try:
            subject_counts, subject_events = process_subject(subject)
            count_rows.extend(subject_counts)
            event_rows.extend(
                {"Subject": subject, **row} for row in subject_events
            )
        except Exception as error:
            errors.append({"Subject": subject, "Error": repr(error)})
            print(f"sub-{subject} failed: {error}")

    pd.DataFrame(count_rows).to_csv(
        qc_dir / "tfr_epoch_condition_counts.csv", index=False
    )
    pd.DataFrame(event_rows).to_csv(
        qc_dir / "tfr_event_alignment_qc.csv", index=False
    )
    (qc_dir / "tfr_epoch_errors.json").write_text(
        json.dumps(errors, indent=2), encoding="utf-8"
    )
    print(f"Epoch output: {OUTPUT_ROOT}")
    print(f"QC output: {qc_dir}")
    if errors:
        raise RuntimeError(
            f"TFR epoch construction failed for {len(errors)} subjects. "
            "Inspect tfr_epoch_errors.json."
        )


if __name__ == "__main__":
    main()
