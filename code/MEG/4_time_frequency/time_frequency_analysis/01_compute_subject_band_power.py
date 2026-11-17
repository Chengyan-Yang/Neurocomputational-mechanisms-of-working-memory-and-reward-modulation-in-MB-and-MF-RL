"""Compute balanced condition-level total or induced band power."""

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

from tfr_analysis_config import (
    selected_indices,
    ANALYSIS_WINDOWS,
    BANDS,
    DECIM,
    N_JOBS,
    PHASES_TO_COMPUTE,
    PHASE_CONDITIONS,
    POWER_ROOT,
    POWER_TYPES_TO_COMPUTE,
    SUBJECTS,
    epoch_path,
    power_output_path,
)


def balanced_condition_epochs(epochs, phase, subject):
    condition_codes = PHASE_CONDITIONS[phase]
    missing = [code for code in condition_codes if code not in epochs.event_id]
    if missing:
        raise RuntimeError(f"Missing conditions: {missing}")

    condition_epochs = [epochs[code].copy() for code in condition_codes]
    original_counts = [len(item) for item in condition_epochs]
    minimum_count = min(original_counts)
    if minimum_count < 2:
        raise RuntimeError(
            f"Too few trials after rejection: {dict(zip(condition_codes, original_counts))}"
        )

    balanced = []
    for code, item in zip(condition_codes, condition_epochs):
        keep = selected_indices(subject, phase, code, item.events[:, 0], minimum_count)
        balanced.append(item[keep])
    return balanced, original_counts, minimum_count


def compute_one_band(epochs, band_name, power_type, analysis_window):
    definition = BANDS[band_name]
    frequencies = np.arange(
        definition["fmin"], definition["fmax"] + 1, dtype=float
    )
    n_cycles = np.full(len(frequencies), definition["n_cycles"], dtype=float)
    working = epochs.copy()
    if power_type == "induced":
        working.subtract_evoked(working.average())

    tfr = mne.time_frequency.tfr_morlet(
        working,
        freqs=frequencies,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        average=True,
        decim=DECIM,
        n_jobs=N_JOBS,
        verbose=False,
    )
    tfr.crop(*analysis_window, include_tmax=True)
    linear_band_power = np.mean(tfr.data, axis=1)
    power_ft2 = np.maximum(linear_band_power * 1e30, np.finfo(float).tiny)
    log_power = 10.0 * np.log10(power_ft2)
    return log_power.astype(np.float32), tfr.times.copy(), tfr.ch_names


def process_subject_phase(subject, phase):
    source = epoch_path(subject, phase)
    if not source.exists():
        raise FileNotFoundError(f"Missing TFR epoch file: {source}")
    epochs = mne.read_epochs(source, preload=True, verbose=False)
    balanced, original_counts, balanced_count = balanced_condition_epochs(
        epochs, phase, subject
    )

    rows = []
    for power_type in POWER_TYPES_TO_COMPUTE:
        destination = power_output_path(subject, phase, power_type)
        output = {
            "subject": np.asarray(subject),
            "phase": np.asarray(phase),
            "power_type": np.asarray(power_type),
            "conditions": np.asarray(PHASE_CONDITIONS[phase]),
            "normalization": np.asarray(
                "10*log10 power in fT^2; no event-local baseline"
            ),
            "balanced_trial_count": np.asarray(balanced_count),
        }
        reference_times = None
        reference_channels = None

        for band_name in BANDS:
            condition_data = []
            for condition_epochs in balanced:
                data, times, channel_names = compute_one_band(
                    condition_epochs,
                    band_name,
                    power_type,
                    ANALYSIS_WINDOWS[phase],
                )
                if reference_times is None:
                    reference_times = times
                    reference_channels = channel_names
                elif not np.array_equal(reference_times, times):
                    raise RuntimeError("Band-power time axes do not match.")
                elif list(reference_channels) != list(channel_names):
                    raise RuntimeError("Band-power channel orders do not match.")
                condition_data.append(data)
            output[f"band_{band_name}"] = np.stack(condition_data)

        output["times"] = reference_times
        output["channel_names"] = np.asarray(reference_channels)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **output)

        for condition, original_count in zip(
            PHASE_CONDITIONS[phase], original_counts
        ):
            rows.append(
                {
                    "Subject": subject,
                    "Phase": phase,
                    "Power type": power_type,
                    "Condition": condition,
                    "Available trials": original_count,
                    "Balanced trials used": balanced_count,
                    "Output": str(destination),
                }
            )
    return rows


def main():
    qc_dir = POWER_ROOT / "qc"
    rows = []
    errors = []

    for subject in tqdm(SUBJECTS, desc="Computing subject band power"):
        for phase in PHASES_TO_COMPUTE:
            try:
                rows.extend(process_subject_phase(subject, phase))
            except Exception as error:
                errors.append(
                    {"Subject": subject, "Phase": phase, "Error": repr(error)}
                )
                print(f"sub-{subject} {phase} failed: {error}")

    qc_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        qc_dir / "band_power_trial_counts.csv", index=False
    )
    (qc_dir / "band_power_errors.json").write_text(
        json.dumps(errors, indent=2), encoding="utf-8"
    )
    print(f"Band-power output: {POWER_ROOT}")
    print(f"QC output: {qc_dir}")
    if errors:
        raise RuntimeError(
            f"Band-power computation failed for {len(errors)} subject-phase jobs."
        )


if __name__ == "__main__":
    main()
