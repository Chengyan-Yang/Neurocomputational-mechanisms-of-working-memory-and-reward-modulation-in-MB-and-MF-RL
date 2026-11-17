"""Replot cluster band-power timecourses without rerunning statistics.

This script reads cluster_results.pkl saved by 02_group_cluster_test.py and
recomputes display-only condition/contrast timecourses from the long epochs.
It does not run the cluster permutation test again. The pre-event segment is
shown only as visualization context.
"""

import json
import pickle

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import stats
from tqdm import tqdm


from tfr_analysis_config import (
    selected_indices,  # noqa: E402
    BANDS,
    DECIM,
    DISPLAY_WINDOWS,
    EFFECT_LABELS,
    EFFECT_WEIGHTS,
    N_JOBS,
    PHASE_CONDITIONS,
    POWER_TYPE,
    RESULTS_ROOT,
    epoch_path,
)
from meg_pdf import save_pdf


CONDITION_LABELS = (
    "Low load / small reward",
    "Low load / large reward",
    "High load / small reward",
    "High load / large reward",
)
CONDITION_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
CONTRAST_COLOR = "#B2182B"
SIGNIFICANT_SPAN_COLOR = "#F3D55B"
PLOT_PRE_EVENT_S = 0.2
OUTPUT_SUBDIR = "band_timecourses_pre200ms"

def current_result_dir(phase, band, effect):
    return RESULTS_ROOT / phase / POWER_TYPE / band / effect


def plot_display_window(phase):
    """Return the plotting window, always including 200 ms before event onset."""
    configured_tmin, configured_tmax = DISPLAY_WINDOWS[phase]
    return min(configured_tmin, -PLOT_PRE_EVENT_S), configured_tmax


def load_cluster_results(phase, band, effect):
    path = current_result_dir(phase, band, effect) / "cluster_results.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cluster results: {path}\n"
            "Run 02_group_cluster_test.py once for the same PHASE/POWER_TYPE/BAND/EFFECT."
        )
    with path.open("rb") as file:
        return pickle.load(file), path


def format_p(p_value):
    return "pFWE < .001" if p_value < 0.001 else f"pFWE = {p_value:.3f}"


def temporal_runs(time_indices):
    unique = np.unique(np.asarray(time_indices, dtype=int))
    split_points = np.where(np.diff(unique) > 1)[0] + 1
    return np.split(unique, split_points)


def build_contrast(effect, condition_data):
    weights = EFFECT_WEIGHTS[effect]
    return sum(weight * condition_data[:, index] for index, weight in enumerate(weights))


def balanced_condition_epochs(phase, epochs, subject):
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


def compute_display_band_timecourse(phase, band, condition_epochs, selected_channels):
    definition = BANDS[band]
    frequencies = np.arange(definition["fmin"], definition["fmax"] + 1, dtype=float)
    n_cycles = np.full(len(frequencies), definition["n_cycles"], dtype=float)

    working = condition_epochs.copy().pick(selected_channels)
    if POWER_TYPE == "induced":
        working.subtract_evoked(working.average())

    tfr = working.compute_tfr(
        method="morlet",
        freqs=frequencies,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        average=True,
        decim=DECIM,
        n_jobs=N_JOBS,
        verbose=False,
    )
    tfr.crop(*plot_display_window(phase), include_tmax=True)

    linear_band_power = np.mean(tfr.data, axis=(0, 1))
    power_ft2 = np.maximum(linear_band_power * 1e30, np.finfo(float).tiny)
    log_power = 10.0 * np.log10(power_ft2)
    return log_power.astype(np.float32), tfr.times.copy()


def recompute_display_condition_timecourses(phase, band, valid_subjects, selected_channels):
    subject_arrays = []
    reference_times = None
    trial_count_rows = []

    for subject in tqdm(valid_subjects, desc="Recomputing display timecourses"):
        epochs = mne.read_epochs(epoch_path(subject, phase), preload=True, verbose=False)
        missing_channels = [
            channel for channel in selected_channels if channel not in epochs.ch_names
        ]
        if missing_channels:
            raise RuntimeError(f"Missing cluster channels: {missing_channels[:5]}")

        balanced, original_counts, balanced_count = balanced_condition_epochs(
            phase,
            epochs, subject
        )
        condition_arrays = []
        for condition_epochs in balanced:
            values, display_times = compute_display_band_timecourse(
                phase, band,
                condition_epochs,
                selected_channels=selected_channels,
            )
            if reference_times is None:
                reference_times = display_times
            elif not np.array_equal(reference_times, display_times):
                raise RuntimeError("Display time axes do not match across subjects.")
            condition_arrays.append(values)
        subject_arrays.append(np.stack(condition_arrays))

        for code, original_count in zip(PHASE_CONDITIONS[phase], original_counts):
            trial_count_rows.append(
                {
                    "Subject": subject,
                    "Condition": code,
                    "Available trials": original_count,
                    "Balanced trials used": balanced_count,
                }
            )

    if not subject_arrays:
        raise RuntimeError("No subjects were available for display plotting.")
    return np.stack(subject_arrays), reference_times, trial_count_rows


def add_cluster_time_spans(axis, cluster_times, time_indices):
    for run in temporal_runs(time_indices):
        axis.axvspan(
            cluster_times[run[0]] * 1000,
            cluster_times[run[-1]] * 1000,
            color=SIGNIFICANT_SPAN_COLOR,
            alpha=0.28,
            linewidth=0,
        )


def style_axis(phase, axis):
    axis.axvline(0, color="0.35", linestyle="--", linewidth=0.85)
    display_tmin, display_tmax = plot_display_window(phase)
    axis.set_xlim(display_tmin * 1000, display_tmax * 1000)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", labelsize=9, width=1.0)


def plot_cluster_timecourses(
    phase, band, effect,
    report_number,
    cluster_index,
    cluster_results,
    selected_channels,
    output_dir,
):
    time_indices, _ = cluster_results["clusters"][cluster_index]
    time_indices = np.asarray(time_indices, dtype=int)
    cluster_times = np.asarray(cluster_results["times"], dtype=float)
    p_value = float(cluster_results["cluster_p_values"][cluster_index])
    valid_subjects = [f"{int(subject):02d}" for subject in cluster_results["valid_subjects"]]

    condition_data, display_times, trial_count_rows = (
        recompute_display_condition_timecourses(phase, band, valid_subjects, selected_channels)
    )
    contrast_data = build_contrast(effect, condition_data)

    fig, axis = plt.subplots(figsize=(6.4, 3.2))
    for condition_index, (label, color) in enumerate(
        zip(CONDITION_LABELS, CONDITION_COLORS)
    ):
        values = condition_data[:, condition_index, :]
        mean = values.mean(axis=0)
        sem = stats.sem(values, axis=0, nan_policy="omit")
        axis.plot(display_times * 1000, mean, color=color, lw=1.7, label=label)
        axis.fill_between(
            display_times * 1000,
            mean - sem,
            mean + sem,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    add_cluster_time_spans(axis, cluster_times, time_indices)
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Band power (dB re 1 fT$^2$)")
    axis.set_title(f"{phase} | {band} | cluster {report_number} (display only)")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    style_axis(phase, axis)
    fig.tight_layout()
    condition_pdf = output_dir / f"cluster_{report_number:02d}_condition_timecourses_display.pdf"
    save_pdf(fig, condition_pdf)
    plt.close(fig)

    mean_contrast = contrast_data.mean(axis=0)
    sem_contrast = stats.sem(contrast_data, axis=0, nan_policy="omit")
    fig, axis = plt.subplots(figsize=(6.4, 3.0))
    axis.plot(display_times * 1000, mean_contrast, color=CONTRAST_COLOR, lw=1.9)
    axis.fill_between(
        display_times * 1000,
        mean_contrast - sem_contrast,
        mean_contrast + sem_contrast,
        color=CONTRAST_COLOR,
        alpha=0.14,
        linewidth=0,
    )
    add_cluster_time_spans(axis, cluster_times, time_indices)
    axis.axhline(0, color="0.35", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Band-power contrast (dB)")
    axis.set_title(f"{EFFECT_LABELS[effect]} | cluster {report_number}")
    style_axis(phase, axis)
    fig.tight_layout()
    contrast_pdf = output_dir / f"cluster_{report_number:02d}_contrast_timecourse_display.pdf"
    save_pdf(fig, contrast_pdf)
    plt.close(fig)

    np.savez_compressed(
        output_dir / f"cluster_{report_number:02d}_display_timecourse_data.npz",
        condition_data=condition_data,
        contrast_data=contrast_data,
        condition_labels=np.asarray(CONDITION_LABELS),
        times=display_times,
        selected_channels=np.asarray(selected_channels),
        phase=np.asarray(phase),
        power_type=np.asarray(POWER_TYPE),
        band=np.asarray(band),
        effect=np.asarray(effect),
        cluster_index=np.asarray(cluster_index),
        cluster_p_value=np.asarray(p_value),
    )
    return {
        "Cluster": report_number,
        "Original Cluster Index": int(cluster_index),
        "N cluster sensors": len(selected_channels),
        "Time Window (ms)": (
            f"{cluster_times[time_indices.min()] * 1000:.0f} - "
            f"{cluster_times[time_indices.max()] * 1000:.0f}"
        ),
        "pFWE": p_value,
        "Condition Timecourse Figure": str(condition_pdf),
        "Contrast Timecourse Figure": str(contrast_pdf),
        "trial_count_rows": trial_count_rows,
    }


def main(phase, band, effect):
    cluster_results, cluster_path = load_cluster_results(phase, band, effect)
    good_cluster_inds = np.asarray(cluster_results["good_cluster_inds"], dtype=int)
    if len(good_cluster_inds) == 0:
        print(f"No significant clusters in {cluster_path}")
        return

    output_dir = current_result_dir(phase, band, effect) / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    channel_names = [str(name) for name in cluster_results["channel_names"]]

    summary_rows = []
    all_trial_rows = []
    for report_number, cluster_index in enumerate(good_cluster_inds, start=1):
        _, sensor_indices = cluster_results["clusters"][cluster_index]
        selected_channels = [
            channel_names[index]
            for index in np.unique(np.asarray(sensor_indices, dtype=int))
        ]
        row = plot_cluster_timecourses(
            phase, band, effect,
            report_number=report_number,
            cluster_index=cluster_index,
            cluster_results=cluster_results,
            selected_channels=selected_channels,
            output_dir=output_dir,
        )
        all_trial_rows.extend(row.pop("trial_count_rows"))
        summary_rows.append(row)

    import pandas as pd

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "band_timecourse_display_summary.csv", index=False
    )
    pd.DataFrame(all_trial_rows).to_csv(
        output_dir / "band_timecourse_display_trial_counts.csv", index=False
    )
    (output_dir / "plot_config.json").write_text(
        json.dumps(
            {
                "cluster_results": str(cluster_path),
                "phase": phase,
                "power_type": POWER_TYPE,
                "band": band,
                "effect": effect,
                "effect_label": EFFECT_LABELS[effect],
                "display_window": plot_display_window(phase),
                "note": (
                    "These line plots are regenerated from long epochs for display "
                    "only. No cluster permutation test is rerun."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Display window used: {plot_display_window(phase)}")
    print(f"Saved display-only band-power line plots to: {output_dir}")
