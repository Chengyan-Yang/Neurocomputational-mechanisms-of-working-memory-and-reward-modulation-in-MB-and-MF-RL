"""Plot condition-level frequency-by-time heatmaps for significant clusters.

This script reads the cluster statistics saved by 02_group_cluster_test.py.
For each significant cluster, it recomputes frequency-resolved TFRs within the
currently selected band and averages over the significant cluster sensors.
The plotted heatmaps show condition or condition-level power, not subtraction.
The black rectangle marks the time window of the significant band-level
sensor-by-time cluster.
"""

import json
import pickle

import matplotlib.pyplot as plt
from matplotlib import patches
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm


from tfr_analysis_config import (
    selected_indices,  # noqa: E402
    ANALYSIS_WINDOWS,
    BANDS,
    DECIM,
    DISPLAY_WINDOWS,
    EFFECT_LABELS,
    N_JOBS,
    PHASE_CONDITIONS,
    POWER_TYPE,
    epoch_path,
    result_dir,
)
from meg_pdf import save_pdf


CONDITION_LABELS = (
    "Low load / small reward",
    "Low load / large reward",
    "High load / small reward",
    "High load / large reward",
)

PLOT_PRE_EVENT_S = 0.2
OUTPUT_SUBDIR = "condition_band_tfr_heatmaps"


def plot_display_window(phase):
    """Return the plotting window, always including 200 ms before event onset."""
    configured_tmin, configured_tmax = DISPLAY_WINDOWS[phase]
    return min(configured_tmin, -PLOT_PRE_EVENT_S), configured_tmax


def balanced_condition_epochs(epochs, phase, subject):
    """Use the same balanced-trial sampling as 01_compute_subject_band_power.py."""
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


def compute_condition_tfr(phase, condition_epochs, selected_channels, frequencies, n_cycles):
    """Return cluster-sensor-averaged TFR data in dB."""
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

    # Average over the selected significant-cluster sensors.
    linear_power = np.mean(tfr.data, axis=0)
    power_ft2 = np.maximum(linear_power * 1e30, np.finfo(float).tiny)
    log_power = 10.0 * np.log10(power_ft2)
    return log_power.astype(np.float32), tfr.times.copy()


def load_cluster_results(phase, band, effect):
    path = result_dir(phase, band, effect) / "cluster_results.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cluster results: {path}\n"
            "Run 02_group_cluster_test.py for the same PHASE/POWER_TYPE/BAND/EFFECT first."
        )
    with path.open("rb") as file:
        return pickle.load(file), path


def cluster_time_runs(time_indices):
    unique = np.unique(np.asarray(time_indices, dtype=int))
    split_points = np.where(np.diff(unique) > 1)[0] + 1
    return np.split(unique, split_points)


def condition_panel_specs(effect):
    """Return panels to show without subtracting conditions."""
    if effect == "load":
        return [
            ("Low load", [0, 1]),
            ("High load", [2, 3]),
        ]
    if effect == "magnitude":
        return [
            ("small reward", [0, 2]),
            ("large reward", [1, 3]),
        ]
    return [
        ("Low load / small reward", [0]),
        ("Low load / large reward", [1]),
        ("High load / small reward", [2]),
        ("High load / large reward", [3]),
    ]


def robust_limits(matrices):
    values = np.concatenate([np.ravel(matrix) for matrix in matrices])
    vmin = float(np.nanpercentile(values, 2))
    vmax = float(np.nanpercentile(values, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    if vmin == vmax:
        vmax = vmin + np.finfo(float).eps
    return vmin, vmax


def add_significant_time_boxes(axis, cluster_times, time_indices, fmin, fmax):
    """Mark cluster time runs across the currently tested frequency band."""
    for run in cluster_time_runs(time_indices):
        start_ms = cluster_times[run[0]] * 1000
        end_ms = cluster_times[run[-1]] * 1000
        if len(cluster_times) > 1:
            step_ms = np.median(np.diff(cluster_times)) * 1000
            start_ms -= step_ms / 2.0
            end_ms += step_ms / 2.0
        rectangle = patches.Rectangle(
            (start_ms, fmin - 0.5),
            end_ms - start_ms,
            (fmax - fmin) + 1.0,
            fill=False,
            edgecolor="black",
            linewidth=1.8,
        )
        axis.add_patch(rectangle)


def plot_cluster_heatmaps(
    phase, band, effect,
    report_number,
    cluster_index,
    cluster_results,
    condition_tfr,
    tfr_times,
    frequencies,
    selected_channels,
    output_dir,
):
    time_indices, _ = cluster_results["clusters"][cluster_index]
    time_indices = np.asarray(time_indices, dtype=int)
    cluster_times = np.asarray(cluster_results["times"], dtype=float)
    p_value = float(cluster_results["cluster_p_values"][cluster_index])
    specs = condition_panel_specs(effect)

    matrices = []
    for _, condition_indices in specs:
        matrices.append(condition_tfr[:, condition_indices].mean(axis=(0, 1)))

    vmin, vmax = robust_limits(matrices)
    n_panels = len(specs)
    fig_width = 4.0 * n_panels
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(fig_width, 3.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    image = None
    extent = [
        tfr_times[0] * 1000,
        tfr_times[-1] * 1000,
        frequencies[0] - 0.5,
        frequencies[-1] + 0.5,
    ]
    for axis, (label, _), matrix in zip(axes, specs, matrices):
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="bilinear",
            extent=extent,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        add_significant_time_boxes(
            axis,
            cluster_times=cluster_times,
            time_indices=time_indices,
            fmin=float(frequencies[0]),
            fmax=float(frequencies[-1]),
        )
        axis.axvline(0, color="white", linestyle="--", linewidth=0.9, alpha=0.85)
        axis.set_title(label, fontsize=11, fontweight="bold")
        axis.set_xlabel("Time (ms)", fontsize=10)
        axis.tick_params(axis="both", labelsize=9, width=1.0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.1)
        axis.spines["bottom"].set_linewidth(1.1)

    axes[0].set_ylabel("Frequency (Hz)", fontsize=10)
    colorbar = fig.colorbar(image, ax=axes, pad=0.015, shrink=0.92)
    colorbar.set_label("Power (dB re 1 fT$^2$)", fontsize=10)
    colorbar.ax.tick_params(labelsize=9, width=1.0)

    start_ms = cluster_times[time_indices.min()] * 1000
    end_ms = cluster_times[time_indices.max()] * 1000
    p_text = "pFWE < .001" if p_value < 0.001 else f"pFWE = {p_value:.3f}"
    fig.suptitle(
        (
            f"{phase} | {band} | {EFFECT_LABELS[effect]} | cluster {report_number}\n"
            f"cluster sensors averaged, {start_ms:.0f}-{end_ms:.0f} ms, {p_text}"
        ),
        fontsize=12,
        fontweight="bold",
    )

    pdf_path = output_dir / f"cluster_{report_number:02d}_condition_band_tfr_heatmaps.pdf"
    save_pdf(fig, pdf_path)
    plt.close(fig)

    np.savez_compressed(
        output_dir / f"cluster_{report_number:02d}_condition_band_tfr_data.npz",
        condition_tfr=condition_tfr,
        condition_labels=np.asarray(CONDITION_LABELS),
        panel_labels=np.asarray([label for label, _ in specs]),
        panel_condition_indices=np.asarray(
            [",".join(str(index) for index in indices) for _, indices in specs]
        ),
        times=tfr_times,
        frequencies=frequencies,
        selected_channels=np.asarray(selected_channels),
        phase=np.asarray(phase),
        power_type=np.asarray(POWER_TYPE),
        band=np.asarray(band),
        effect=np.asarray(effect),
        cluster_index=np.asarray(cluster_index),
        cluster_p_value=np.asarray(p_value),
    )
    return pdf_path


def recompute_cluster_condition_tfrs(phase, band, valid_subjects, selected_channels):
    definition = BANDS[band]
    frequencies = np.arange(definition["fmin"], definition["fmax"] + 1, dtype=float)
    n_cycles = np.full(len(frequencies), definition["n_cycles"], dtype=float)

    subject_arrays = []
    count_rows = []
    skipped = []
    reference_times = None

    iterator = tqdm(valid_subjects, desc="Computing cluster-sensor TFRs")
    for subject in iterator:
        try:
            epochs = mne.read_epochs(epoch_path(subject, phase), preload=True, verbose=False)
            missing_channels = [ch for ch in selected_channels if ch not in epochs.ch_names]
            if missing_channels:
                raise RuntimeError(f"Missing cluster channels: {missing_channels[:5]}")
            balanced, original_counts, balanced_count = balanced_condition_epochs(
                epochs, phase, subject
            )
            condition_arrays = []
            for condition_epochs in balanced:
                data, times = compute_condition_tfr(
                    phase,
                    condition_epochs,
                    selected_channels=selected_channels,
                    frequencies=frequencies,
                    n_cycles=n_cycles,
                )
                if reference_times is None:
                    reference_times = times
                elif not np.array_equal(reference_times, times):
                    raise RuntimeError("TFR time axes do not match across subjects.")
                condition_arrays.append(data)
            subject_arrays.append(np.stack(condition_arrays))
            for code, original_count in zip(PHASE_CONDITIONS[phase], original_counts):
                count_rows.append(
                    {
                        "Subject": subject,
                        "Phase": phase,
                        "Condition": code,
                        "Available trials": original_count,
                        "Balanced trials used": balanced_count,
                    }
                )
        except Exception as error:
            skipped.append({"Subject": subject, "Error": repr(error)})
            print(f"sub-{subject} failed: {error}")

    if skipped:
        raise RuntimeError(
            f"Cluster TFR plotting failed for {len(skipped)} subjects. "
            "Inspect skipped_subjects.json in the output folder."
        )
    if not subject_arrays:
        raise RuntimeError("No subjects were available for cluster TFR plotting.")

    return (
        np.stack(subject_arrays),
        reference_times,
        frequencies,
        pd.DataFrame(count_rows),
        skipped,
    )


def main(phase, band, effect):
    cluster_results, cluster_path = load_cluster_results(phase, band, effect)
    good_cluster_inds = np.asarray(cluster_results["good_cluster_inds"], dtype=int)
    if len(good_cluster_inds) == 0:
        print(f"No significant clusters in {cluster_path}")
        return

    output_dir = result_dir(phase, band, effect) / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    channel_names = list(cluster_results["channel_names"])
    valid_subjects = [f"{int(subject):02d}" for subject in cluster_results["valid_subjects"]]

    saved_images = []
    cluster_rows = []
    for report_number, cluster_index in enumerate(good_cluster_inds, start=1):
        _, sensor_indices = cluster_results["clusters"][cluster_index]
        selected_channels = [
            channel_names[index] for index in np.unique(np.asarray(sensor_indices, dtype=int))
        ]
        cluster_dir = output_dir / f"cluster_{report_number:02d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Channel": selected_channels}).to_csv(
            cluster_dir / "selected_cluster_sensors.csv", index=False
        )

        condition_tfr, tfr_times, frequencies, counts, skipped = (
            recompute_cluster_condition_tfrs(phase, band, valid_subjects, selected_channels)
        )
        counts.to_csv(cluster_dir / "trial_counts.csv", index=False)
        (cluster_dir / "skipped_subjects.json").write_text(
            json.dumps(skipped, indent=2), encoding="utf-8"
        )

        image_path = plot_cluster_heatmaps(
            phase, band, effect,
            report_number=report_number,
            cluster_index=cluster_index,
            cluster_results=cluster_results,
            condition_tfr=condition_tfr,
            tfr_times=tfr_times,
            frequencies=frequencies,
            selected_channels=selected_channels,
            output_dir=cluster_dir,
        )
        saved_images.append(str(image_path))
        time_indices, _ = cluster_results["clusters"][cluster_index]
        cluster_times = np.asarray(cluster_results["times"], dtype=float)
        cluster_rows.append(
            {
                "Cluster": report_number,
                "Original Cluster Index": int(cluster_index),
                "N cluster sensors": len(selected_channels),
                "Time Window (ms)": (
                    f"{cluster_times[np.min(time_indices)] * 1000:.0f} - "
                    f"{cluster_times[np.max(time_indices)] * 1000:.0f}"
                ),
                "pFWE": float(cluster_results["cluster_p_values"][cluster_index]),
                "Figure": str(image_path),
            }
        )

    pd.DataFrame(cluster_rows).to_csv(
        output_dir / "condition_band_tfr_heatmap_summary.csv", index=False
    )
    (output_dir / "plot_config.json").write_text(
        json.dumps(
            {
                "cluster_results": str(cluster_path),
                "phase": phase,
                "power_type": POWER_TYPE,
                "band": band,
                "band_definition": BANDS[band],
                "effect": effect,
                "effect_label": EFFECT_LABELS[effect],
                "analysis_window": ANALYSIS_WINDOWS[phase],
        "display_window": plot_display_window(phase),
                "normalization": "10*log10 power in fT^2; no event-local baseline",
                "interpretation_note": (
                    "Heatmaps show condition-level cluster-sensor-averaged TFRs. "
                    "The pre-event segment is shown only for visualization. "
                    "The black box marks the significant time window from the "
                    "band-level sensor-by-time cluster test."
                ),
                "saved_images": saved_images,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Saved condition-level band TFR heatmaps:")
    for image in saved_images:
        print(image)
