"""Build four-stage ERF epochs or launch the four-stage TFA epoch workflow."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meg_project import DATA_ROOT, meg_subjects, PREP_RESULTS

import argparse
import runpy
import os
import json

import mne
import numpy as np
import pandas as pd


mne.set_log_level("WARNING")


SUBJECTS = meg_subjects()

RAW_ROOT = PREP_RESULTS / "2_ICA" / "cleaned"
BEHAVIOR_ROOT = DATA_ROOT / "BHV" / "2_stage_task" / "preprocessed" / "learning" / "all" / "RL_code"
RECONSTRUCTED_EVENT_ROOT = DATA_ROOT / "MEG" / "preprocessed" / "2_ICA" / "reconstructed_events"
EPOCH_ROOT = PREP_RESULTS / "3_epochs"

PHASE = "decision"
EXPECTED_TRIALS = 320

INCENTIVE_CODES = np.array([1, 2, 3, 4], dtype=int)
STAGE1_CODES = np.array([11, 21, 31, 41], dtype=int)
CHOOSE_CODES = np.array([12, 22, 32, 42], dtype=int)
CONDITIONS = [11, 21, 31, 41]

# Short ERF/source epoch. Main analyses can crop this to 0-2 s.
TMIN = -0.2
TMAX = 2.0
BASELINE = (-0.1, 0.0)
DETREND = 1
REJECT_CRITERIA = {"mag": 3000e-15}
OVERWRITE = True


def raw_path(subject):
    return RAW_ROOT / f"sub-{subject}_clean_raw.fif"


def behavior_path(subject):
    return BEHAVIOR_ROOT / f"{subject}_RL_prep.csv"


def reconstructed_events_path(subject):
    return (
        RECONSTRUCTED_EVENT_ROOT
        / f"sub-{subject}"
        / f"sub-{subject}_events_with_feedback_image_reconstructed.npy"
    )


def epoch_path(subject):
    return EPOCH_ROOT / f"sub-{subject}" / f"sub-{subject}_{PHASE}_epo.fif"


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
            f"Behavior trial IDs are not the complete 1-{EXPECTED_TRIALS} range."
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


def condition_metadata(row, condition, meg_incentive_index, stage1_event, choose_info):
    ordered = [str(code) for code in CONDITIONS]
    condition = str(int(condition))
    condition_index = ordered.index(condition)
    metadata = row.to_dict()
    stage1_sample = int(stage1_event[0])
    metadata.update(
        {
            "phase": PHASE,
            "condition_code": condition,
            "condition_index": condition_index,
            "magnitude_level": "small" if condition_index in (0, 2) else "large",
            "load_level": "low" if condition_index in (0, 1) else "high",
            "meg_incentive_index": int(meg_incentive_index),
            "event_sample": stage1_sample,
            "event_source": "stage1_option_onset",
            **choose_info,
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
    qc_rows = []

    for position, incentive_event in enumerate(incentive_events):
        row = behavior.iloc[position]
        meg_incentive_index = position + 1
        behavior_trial = int(row["trials_globe"])
        trial_start = int(incentive_event[0])
        trial_stop = (
            int(incentive_events[position + 1, 0])
            if position + 1 < len(incentive_events)
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
            "MEG incentive index": meg_incentive_index,
            "Behavior trial": behavior_trial,
            "Dropped behavior trial for alignment": dropped_trial,
            "Behavior miss": int(row["miss"]),
            "Stage1 matches": int(len(stage1_matches)),
            "Choose matches": int(len(choose_matches)),
        }

        if int(row["miss"]) == 1:
            qc_rows.append({**qc_base, "Status": "behavior_miss", "Included": False})
            continue
        if len(stage1_matches) != 1:
            qc_rows.append(
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
                meg_incentive_index=meg_incentive_index,
                stage1_event=stage1_event,
                choose_info=choose_info,
            )
        )
        qc_rows.append(
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

    return np.asarray(selected_events, dtype=int), pd.DataFrame(metadata_rows), qc_rows


FEEDBACK_CODES = (51, 52, 53, 54)

def feedback_metadata(row, phase, condition, meg_incentive_index, event_sample):
    ordered = [str(code) for code in FEEDBACK_CODES]
    condition = str(int(condition))
    condition_index = ordered.index(condition)
    metadata = row.to_dict()
    metadata.update(
        {
            "phase": phase,
            "condition_code": condition,
            "condition_index": condition_index,
            "magnitude_level": "small" if condition_index in (0, 2) else "large",
            "load_level": "low" if condition_index in (0, 1) else "high",
            "meg_incentive_index": int(meg_incentive_index),
            "event_sample": int(event_sample),
        }
    )
    metadata["event_source"] = "feedback_text_minus_0.5s"
    return metadata


def collect_feedback_events(events, behavior):
    phase = "feedback"
    incentive_events = events[np.isin(events[:, 2], INCENTIVE_CODES)]
    expected_trials = len(behavior)
    if len(incentive_events) != expected_trials:
        raise RuntimeError(
            f"Expected {expected_trials} incentive events after behavior alignment "
            f"correction, found {len(incentive_events)}."
        )

    target_codes = np.array(FEEDBACK_CODES, dtype=int)
    selected_events = []
    metadata_rows = []
    qc_rows = []

    for position, incentive_event in enumerate(incentive_events):
        meg_incentive_index = position + 1
        behavior_row = behavior.iloc[position]
        behavior_trial = int(behavior_row["trials_globe"])

        trial_start = int(incentive_event[0])
        trial_stop = (
            int(incentive_events[position + 1, 0])
            if position + 1 < len(incentive_events)
            else int(events[-1, 0] + 1)
        )

        matches = events[
            (events[:, 0] >= trial_start)
            & (events[:, 0] < trial_stop)
            & np.isin(events[:, 2], target_codes)
        ]

        qc_base = {
            "MEG incentive index": meg_incentive_index,
            "Behavior trial": behavior_trial,
            "Phase": phase,
            "Behavior miss": int(behavior_row["miss"]),
        }

        if int(behavior_row["miss"]) == 1:
            qc_rows.append({**qc_base, "Status": "behavior_miss", "Included": False})
            continue

        if len(matches) != 1:
            qc_rows.append(
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
            feedback_metadata(
                behavior_row,
                phase=phase,
                condition=event[2],
                meg_incentive_index=meg_incentive_index,
                event_sample=event[0],
            )
        )
        qc_rows.append(
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
        raise RuntimeError(f"No usable events were found for {phase}.")

    return (
        np.asarray(selected_events, dtype=int),
        pd.DataFrame(metadata_rows),
        qc_rows,
    )


def count_conditions(epochs, conditions):
    return {
        str(condition): int(np.sum(epochs.events[:, 2] == int(condition)))
        for condition in conditions
    }


def process_subject(subject, phase):
    source = raw_path(subject)
    if not source.exists():
        raise FileNotFoundError(f"Missing ICA-clean raw file: {source}")

    behavior, dropped_trial = load_behavior(subject)
    raw = mne.io.read_raw_fif(source, preload=True, verbose=False)
    if phase == "feedback":
        event_file = reconstructed_events_path(subject)
        events = np.load(event_file, allow_pickle=False)
        events = events[np.argsort(events[:, 0])]
        selected, metadata, event_qc = collect_feedback_events(events, behavior)
        conditions, tmax = FEEDBACK_CODES, 1.5
    else:
        events = load_events(subject, raw)
        selected, metadata, event_qc = collect_decision_events(raw, events, behavior, dropped_trial)
        conditions, tmax = CONDITIONS, TMAX
    event_id = {str(code): int(code) for code in conditions}
    epochs = mne.Epochs(
        raw,
        selected,
        event_id=event_id,
        tmin=TMIN,
        tmax=tmax,
        picks="mag",
        baseline=None,
        detrend=DETREND,
        preload=True,
        metadata=metadata,
        reject_by_annotation=True,
        verbose=False,
    )
    epochs.apply_baseline(BASELINE)

    counts_before = count_conditions(epochs, conditions)
    n_before = len(epochs)
    epochs.drop_bad(REJECT_CRITERIA, verbose=False)
    counts_after = count_conditions(epochs, conditions)
    n_after = len(epochs)

    destination = EPOCH_ROOT / f"sub-{subject}" / f"sub-{subject}_{phase}_epo.fif"
    destination.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(destination, overwrite=OVERWRITE, verbose=False)

    summary_rows = []
    for condition in [str(code) for code in conditions]:
        summary_rows.append(
            {
                "Subject": subject,
                "Phase": phase,
                "Condition": condition,
                "Dropped behavior trial for alignment": dropped_trial,
                "Initial valid epochs": n_before,
                "Final epochs": n_after,
                "Rejected epochs": n_before - n_after,
                "Before amplitude rejection": counts_before[condition],
                "After amplitude rejection": counts_after[condition],
                "Epoch tmin (s)": epochs.tmin,
                "Epoch tmax (s)": epochs.tmax,
                "Baseline": str(BASELINE),
                "Epoch file": str(destination),
            }
        )
    return summary_rows, event_qc


def build_short_phase(phase):
    EPOCH_ROOT.mkdir(parents=True, exist_ok=True)
    qc_dir = EPOCH_ROOT / "qc" / phase
    qc_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    event_qc_rows = []
    errors = []

    for subject in SUBJECTS:
        # These subjects use the supplied aligned epochs.
        if subject in {"07", "14"}:
            continue
        print(f"Processing sub-{subject}: {phase}")
        try:
            subject_summary, subject_event_qc = process_subject(subject, phase)
            summary_rows.extend(subject_summary)
            event_qc_rows.extend({"Subject": subject, **row} for row in subject_event_qc)
        except Exception as error:
            errors.append({"Subject": subject, "Error": repr(error)})
            print(f"  sub-{subject} failed: {error}")

    summary_path = qc_dir / f"{phase}_short_epoch_summary.csv"
    event_qc_path = qc_dir / f"{phase}_short_epoch_event_qc.csv"
    error_path = qc_dir / f"{phase}_short_epoch_errors.json"

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(event_qc_rows).to_csv(event_qc_path, index=False)
    error_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Summary: {summary_path}")
    print(f"Event QC: {event_qc_path}")
    print(f"Errors: {error_path}")
    if errors:
        raise RuntimeError(
            f"{phase} short epoch construction failed for {len(errors)} subjects. "
            f"Inspect {error_path}."
        )


def build_incentive_transition():
    subjects = SUBJECTS
    rejection_summary = {}

    data_root = RAW_ROOT
    bhv_root = BEHAVIOR_ROOT
    epoch_root = EPOCH_ROOT

    phases = {'incentive': [1, 2, 3, 4], 'transition': [13, 23, 33, 43]}

    for sub in subjects:
        # These subjects use the supplied aligned epochs.
        if sub in {"07", "14"}:
            continue
        print(f"\n{'='*50}")
        print(f"Processing subject {sub}")
        print(f"{'='*50}")
    
        if sub not in rejection_summary:
            rejection_summary[sub] = {}
        
        ica_cleaned_path = os.path.join(data_root, f'sub-{sub}_clean_raw.fif')
        bhv_file = os.path.join(bhv_root, f'{sub}_RL_prep.csv')
    
        if not os.path.exists(ica_cleaned_path):
            print(f"Skipping {sub}: ICA-cleaned recording not found.")
            continue

        print(f"Loading data for {sub}")
        ica_clean = mne.io.read_raw_fif(ica_cleaned_path, preload=True)

        if not os.path.exists(bhv_file):
            print(f"Skipping epoch creation for {sub}: behavior file not found at {bhv_file}.")
            del ica_clean
            continue
        
        bhv_data = pd.read_csv(bhv_file)

        print(f"Aligning events to valid behavioral trials for {sub}")

        events = mne.find_events(ica_clean, stim_channel='UPPT001', shortest_event=1)
        sfreq = ica_clean.info['sfreq']
        incentive_events = events[np.isin(events[:, 2], phases['incentive'])]
        total_trials = len(incentive_events)
    
        for phase_name, event_ids in phases.items():
            print(f"Finding events for {phase_name}")
        
            valid_meg_events = []
            valid_bhv_indices = []

            for i in range(total_trials):
                trial_id = i + 1 
            
                bhv_row = bhv_data[bhv_data['trials_globe'] == trial_id]
                if bhv_row.empty:
                    continue
            
                is_miss = bhv_row.iloc[0]['miss'] == 1
                row_idx = bhv_row.index[0]
            
                # Search within the current trial's MEG event window.
                t_start = incentive_events[i, 0]
                t_end = incentive_events[i+1, 0] if (i+1) < total_trials else t_start + int(sfreq * 12)

                this_matches = events[(events[:, 0] >= t_start) & 
                                     (events[:, 0] < t_end) & 
                                     (np.isin(events[:, 2], event_ids))]

                # Keep trials with both a valid behavioral response and a matching MEG event.
                if not is_miss:
                    if len(this_matches) > 0:
                        valid_meg_events.append(this_matches[0])
                        valid_bhv_indices.append(row_idx)
                    else:
                        print(f"Dropped trial {trial_id}: behavioral response present but no {phase_name} event.")

            # Keep metadata rows in the same order as the selected events.
            if len(valid_meg_events) > 0:
                phase_events = np.array(valid_meg_events)
                current_bhv = bhv_data.loc[valid_bhv_indices].copy().reset_index(drop=True)
            
                epochs = mne.Epochs(ica_clean, phase_events, tmin=-0.2, tmax=1.5,
                                    picks='mag', baseline=None, detrend=1, preload=True,
                                    metadata=current_bhv)
            
                epochs.apply_baseline((-0.1, 0))
                reject_criteria = dict(mag=3000e-15) 
            
                # Count epochs before and after artifact rejection.
                n_before = len(epochs)
                epochs.drop_bad(reject_criteria)
                n_after = len(epochs)
                n_rejected = n_before - n_after
            
                rejection_summary[sub][phase_name] = {
                    "initial_valid_trials": n_before, 
                    "rejected_trials": n_rejected,   
                    "final_trials": n_after,          
                    "drop_log": epochs.drop_log      
                }
            
                out_dir = os.path.join(epoch_root, f'sub-{sub}')
                os.makedirs(out_dir, exist_ok=True)   
                out_name = os.path.join(out_dir, f'sub-{sub}_{phase_name}_epo.fif')
                epochs.save(out_name, overwrite=True)
                print(f"{phase_name}: retained {n_after} epochs; rejected {n_rejected}.")
            
            else:
                rejection_summary[sub][phase_name] = "No valid events"

        del ica_clean
        if 'epochs' in locals(): 
            del epochs
        print(f"Finished preprocessing {sub}.")

    summary_save_path = os.path.join(epoch_root, 'all_subjects_rejection_stats.json')
    with open(summary_save_path, 'w', encoding='utf-8') as f:
        json.dump(rejection_summary, f, indent=4, ensure_ascii=False)

    print(f"Saved the epoch rejection summary to {summary_save_path}")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("erf", "tfa", "all"), default="erf")
    kind = parser.parse_args().kind
    if kind in ("erf", "all"):
        build_incentive_transition()
        build_short_phase("feedback")
        build_short_phase("decision")
    if kind in ("tfa", "all"):
        folder = Path(__file__).resolve().parents[1] / "4_time_frequency" / "time_frequency_preprocessing"
        for script in ("01_make_tfr_epochs.py", "02_make_tfr_epochs_feedback.py",
                       "03_make_tfr_epochs_decision.py"):
            runpy.run_path(str(folder / script), run_name="__main__")


if __name__ == "__main__":
    main()
