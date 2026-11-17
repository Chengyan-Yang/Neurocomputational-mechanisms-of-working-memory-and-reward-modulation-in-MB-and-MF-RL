"""Run all band-by-effect sensor cluster tests for the selected phase."""

import json
import pickle
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import sparse, stats
from tqdm import tqdm

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

if __name__ == "__main__":
    # Spyder can reuse a kernel after the phase setting is edited.
    import importlib
    import tfr_analysis_config
    importlib.reload(tfr_analysis_config)

from tfr_analysis_config import (
    selected_indices,
    BANDS,
    CLUSTER_ALPHA,
    CLUSTER_FORMING_P,
    DECIM,
    DISPLAY_WINDOWS,
    EFFECT_LABELS,
    EFFECT_WEIGHTS,
    N_JOBS,
    N_PERMUTATIONS,
    N_PLANNED_BANDS,
    N_PLANNED_EFFECTS,
    N_PLANNED_PHASES,
    PHASE,
    PHASE_CONDITIONS,
    POWER_TYPE,
    RANDOM_SEED,
    SUBJECTS,
    epoch_path,
    power_path,
    result_dir,
)
from meg_pdf import save_pdf


CONDITION_LABELS = (
    "Low load / small reward",
    "Low load / large reward",
    "High load / small reward",
    "High load / large reward",
)
CONDITION_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
PLOT_PRE_EVENT_S = 0.2


def plot_display_window(phase):
    """Return the plotting window, always including 200 ms before event onset."""
    configured_tmin, configured_tmax = DISPLAY_WINDOWS[phase]
    return min(configured_tmin, -PLOT_PRE_EVENT_S), configured_tmax


def normalize_channel_name(name):
    return str(name).split("-")[0].replace(" ", "")


def load_group_condition_data(band):
    records = []
    skipped = []
    expected_conditions = tuple(PHASE_CONDITIONS[PHASE])

    for subject in tqdm(SUBJECTS, desc=f"Loading {PHASE} {band} power"):
        path = power_path(subject, PHASE, POWER_TYPE)
        if not path.exists():
            skipped.append({"Subject": subject, "Reason": "missing power file"})
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                conditions = tuple(str(value) for value in archive["conditions"])
                if conditions != expected_conditions:
                    raise RuntimeError(
                        f"Condition order {conditions} does not match {expected_conditions}."
                    )
                data = np.asarray(archive[f"band_{band}"], dtype=np.float32)
                times = np.asarray(archive["times"], dtype=float)
                channels = [str(value) for value in archive["channel_names"]]
            if data.shape != (4, len(channels), len(times)):
                raise RuntimeError(f"Unexpected band-power shape: {data.shape}")
            records.append(
                {
                    "subject": subject,
                    "data": data,
                    "times": times,
                    "channels": channels,
                }
            )
        except Exception as error:
            skipped.append({"Subject": subject, "Reason": repr(error)})

    if not records:
        raise RuntimeError("No complete subject power files were available.")

    reference_times = records[0]["times"]
    for record in records[1:]:
        if not np.array_equal(record["times"], reference_times):
            raise RuntimeError("Subject time axes do not match.")

    common = set(records[0]["channels"])
    for record in records[1:]:
        common.intersection_update(record["channels"])
    common_channels = [
        channel for channel in records[0]["channels"] if channel in common
    ]
    if not common_channels:
        raise RuntimeError("No common channels were found across subjects.")

    subject_arrays = []
    valid_subjects = []
    for record in records:
        index = {name: position for position, name in enumerate(record["channels"])}
        picks = [index[name] for name in common_channels]
        selected = record["data"][:, picks, :].transpose(0, 2, 1)
        subject_arrays.append(selected)
        valid_subjects.append(record["subject"])

    data = np.stack(subject_arrays)
    return data, reference_times, common_channels, valid_subjects, skipped


def build_info_and_adjacency(subject, data, common_channels):
    epochs = mne.read_epochs(epoch_path(subject, PHASE), preload=False, verbose=False)
    missing = [name for name in common_channels if name not in epochs.ch_names]
    if missing:
        raise RuntimeError(f"Template epochs are missing channels: {missing[:5]}")
    common_indices = [epochs.ch_names.index(name) for name in common_channels]
    common_info = mne.pick_info(epochs.info, common_indices, copy=True)

    full_adjacency, adjacency_names = mne.channels.find_ch_adjacency(
        common_info, ch_type="mag"
    )
    data_lookup = {
        normalize_channel_name(name): name for name in common_info.ch_names
    }
    adjacency_indices = []
    ordered_names = []
    for adjacency_index, adjacency_name in enumerate(adjacency_names):
        normalized = normalize_channel_name(adjacency_name)
        if normalized in data_lookup:
            adjacency_indices.append(adjacency_index)
            ordered_names.append(data_lookup[normalized])
    if not ordered_names:
        raise RuntimeError("No channels matched the CTF adjacency.")
    if len(set(ordered_names)) != len(ordered_names):
        raise RuntimeError("Adjacency matching produced duplicate channels.")

    adjacency = sparse.csr_matrix(
        full_adjacency[adjacency_indices][:, adjacency_indices]
    )
    data_order = [common_channels.index(name) for name in ordered_names]
    ordered_data = data[:, :, :, data_order]
    info_order = [common_info.ch_names.index(name) for name in ordered_names]
    ordered_info = mne.pick_info(common_info, info_order, copy=True)
    if adjacency.shape[0] != ordered_data.shape[-1]:
        raise RuntimeError("Adjacency and data channel dimensions do not match.")
    return ordered_data, ordered_info, adjacency, ordered_names


def build_contrast(condition_data, effect):
    weights = EFFECT_WEIGHTS[effect]
    return sum(
        weight * condition_data[:, index]
        for index, weight in enumerate(weights)
    )


def balanced_condition_epochs(phase, epochs, subject):
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
    return balanced


def compute_display_band_timecourse(phase, band, condition_epochs, selected_channels):
    """Return a display-window, cluster-sensor-averaged band-power timecourse."""
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
    """Recompute cluster-sensor timecourses from long epochs for plotting only."""
    subject_arrays = []
    reference_times = None

    for subject in valid_subjects:
        epochs = mne.read_epochs(epoch_path(subject, phase), preload=True, verbose=False)
        missing_channels = [
            channel for channel in selected_channels if channel not in epochs.ch_names
        ]
        if missing_channels:
            raise RuntimeError(f"Missing cluster channels: {missing_channels[:5]}")
        balanced = balanced_condition_epochs(phase, epochs, subject)
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

    if not subject_arrays:
        raise RuntimeError("No subjects were available for display timecourse plotting.")
    return np.stack(subject_arrays), reference_times


def temporal_runs(time_indices):
    unique = np.unique(np.asarray(time_indices, dtype=int))
    split_points = np.where(np.diff(unique) > 1)[0] + 1
    return np.split(unique, split_points)


def format_p(p_value):
    return "p < .001" if p_value < 0.001 else f"p = {p_value:.3f}"


def plot_cluster(
    phase, band, effect,
    report_number,
    cluster_index,
    t_obs,
    clusters,
    cluster_p_values,
    times,
    info,
    condition_data,
    contrast_data,
    valid_subjects,
    output_dir,
):
    time_indices, sensor_indices = clusters[cluster_index]
    time_indices = np.asarray(time_indices, dtype=int)
    sensor_indices = np.asarray(sensor_indices, dtype=int)
    unique_sensors = np.unique(sensor_indices)

    sums = np.zeros(t_obs.shape[1], dtype=float)
    counts = np.zeros(t_obs.shape[1], dtype=int)
    for time_index, sensor_index in zip(time_indices, sensor_indices):
        sums[sensor_index] += t_obs[time_index, sensor_index]
        counts[sensor_index] += 1
    t_map = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    sensor_mask = counts > 0
    vmax = max(float(np.max(np.abs(t_map))), np.finfo(float).eps)

    fig, axis = plt.subplots(figsize=(4.4, 3.7))
    image, _ = mne.viz.plot_topomap(
        t_map,
        info,
        axes=axis,
        show=False,
        cmap="RdBu_r",
        vlim=(-vmax, vmax),
        mask=sensor_mask,
        mask_params={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 4,
        },
    )
    start_ms = times[time_indices.min()] * 1000
    end_ms = times[time_indices.max()] * 1000
    p_value = float(cluster_p_values[cluster_index])
    axis.set_title(
        f"Cluster {report_number}: {start_ms:.0f}-{end_ms:.0f} ms\n"
        f"{format_p(p_value)}"
    )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.8, pad=0.05)
    colorbar.set_label("Mean cluster t-value")
    fig.suptitle(f"{phase} | {band} | {EFFECT_LABELS[effect]}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_pdf(fig, output_dir / f"cluster_{report_number:02d}_topomap.pdf")
    plt.close(fig)

    selected_channels = [info.ch_names[index] for index in unique_sensors]
    display_condition_data, display_times = recompute_display_condition_timecourses(
        phase, band,
        valid_subjects,
        selected_channels,
    )
    display_contrast_data = build_contrast(display_condition_data, effect)

    fig, axis = plt.subplots(figsize=(6.4, 3.2))
    for condition_index, (label, color) in enumerate(
        zip(CONDITION_LABELS, CONDITION_COLORS)
    ):
        values = display_condition_data[:, condition_index, :]
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
    for run in temporal_runs(time_indices):
        axis.axvspan(
            times[run[0]] * 1000,
            times[run[-1]] * 1000,
            color="#F3D55B",
            alpha=0.28,
            linewidth=0,
        )
    axis.axvline(0, color="0.3", ls="--", lw=0.9)
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Band power (dB re 1 fT$^2$)")
    axis.set_title(
        f"{phase} | {band} | cluster {report_number} (descriptive)"
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    save_pdf(fig,
        output_dir / f"cluster_{report_number:02d}_condition_timecourses.pdf"
    )
    plt.close(fig)

    contrast_values = display_contrast_data
    mean_contrast = contrast_values.mean(axis=0)
    sem_contrast = stats.sem(contrast_values, axis=0, nan_policy="omit")
    fig, axis = plt.subplots(figsize=(6.4, 3.0))
    axis.plot(display_times * 1000, mean_contrast, color="#B2182B", lw=1.9)
    axis.fill_between(
        display_times * 1000,
        mean_contrast - sem_contrast,
        mean_contrast + sem_contrast,
        color="#B2182B",
        alpha=0.14,
        linewidth=0,
    )
    for run in temporal_runs(time_indices):
        axis.axvspan(
            times[run[0]] * 1000,
            times[run[-1]] * 1000,
            color="#F3D55B",
            alpha=0.28,
            linewidth=0,
        )
    axis.axhline(0, color="0.35", ls="--", lw=0.8)
    axis.axvline(0, color="0.35", ls="--", lw=0.8)
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Band-power contrast (dB)")
    axis.set_title(f"{EFFECT_LABELS[effect]} | cluster {report_number}")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    save_pdf(fig, output_dir / f"cluster_{report_number:02d}_contrast_timecourse.pdf")
    plt.close(fig)

    # Frequency-resolved condition heatmaps are generated by
    # the plotting step after the statistics are saved.


def _tick_positions(n_items, max_ticks=14):
    if n_items <= max_ticks:
        return np.arange(n_items)
    return np.unique(np.linspace(0, n_items - 1, max_ticks).round().astype(int))


def plot_cluster_sensor_time_heatmaps(
    phase, band, effect,
    report_number,
    cluster_index,
    t_obs,
    clusters,
    times,
    channel_names,
    contrast_data,
    condition_data,
    output_dir,
):
    """Plot sensor-by-time heatmaps for one significant band-power cluster."""
    time_indices, sensor_indices = clusters[cluster_index]
    time_indices = np.asarray(time_indices, dtype=int)
    sensor_indices = np.asarray(sensor_indices, dtype=int)
    unique_sensors = np.unique(sensor_indices)

    # Keep sensors ordered by their strongest absolute t-value in the cluster,
    # which makes the heatmap easier to read than raw channel order.
    sensor_strength = {}
    for sensor in unique_sensors:
        in_sensor = sensor_indices == sensor
        sensor_strength[sensor] = np.max(
            np.abs(t_obs[time_indices[in_sensor], sensor])
        )
    ordered_sensors = np.array(
        sorted(unique_sensors, key=lambda sensor: sensor_strength[sensor], reverse=True)
    )

    t_matrix = t_obs[:, ordered_sensors].T
    contrast_matrix = contrast_data[:, :, ordered_sensors].mean(axis=0).T

    cluster_mask = np.zeros_like(t_matrix, dtype=bool)
    sensor_to_row = {sensor: row for row, sensor in enumerate(ordered_sensors)}
    for time_index, sensor_index in zip(time_indices, sensor_indices):
        cluster_mask[sensor_to_row[sensor_index], time_index] = True

    def format_sensor_axis(axis):
        tick_positions = _tick_positions(len(ordered_sensors))
        axis.set_yticks(tick_positions)
        axis.set_yticklabels(
            [
                normalize_channel_name(channel_names[ordered_sensors[pos]])
                for pos in tick_positions
            ],
            fontsize=7,
        )

    def add_cluster_contour(axis):
        if cluster_mask.any():
            axis.contour(
                times * 1000,
                np.arange(len(ordered_sensors)),
                cluster_mask.astype(float),
                levels=[0.5],
                colors="black",
                linewidths=0.9,
            )

    def draw_heatmap(matrix, title, colorbar_label, stem, cmap="RdBu_r"):
        fig, axis = plt.subplots(figsize=(7.0, 4.4))
        vmax = float(np.nanpercentile(np.abs(matrix), 98))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = float(np.nanmax(np.abs(matrix)))
        vmax = max(vmax, np.finfo(float).eps)
        if cmap == "RdBu_r":
            vmin = -vmax
        else:
            vmin = float(np.nanpercentile(matrix, 2))
            vmax = float(np.nanpercentile(matrix, 98))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin = float(np.nanmin(matrix))
                vmax = float(np.nanmax(matrix))
            if vmin == vmax:
                vmax = vmin + np.finfo(float).eps

        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[
                times[0] * 1000,
                times[-1] * 1000,
                -0.5,
                len(ordered_sensors) - 0.5,
            ],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        # Show the exact significant sensor-time points as a black contour.
        add_cluster_contour(axis)

        axis.axvline(0, color="0.25", ls="--", lw=0.8)
        format_sensor_axis(axis)
        axis.set_xlabel("Time (ms)")
        axis.set_ylabel("Cluster sensors")
        axis.set_title(title)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        colorbar = fig.colorbar(image, ax=axis, pad=0.02)
        colorbar.set_label(colorbar_label)
        fig.tight_layout()
        save_pdf(fig, output_dir / f"cluster_{report_number:02d}_{stem}.pdf")
        plt.close(fig)

    start_ms = times[time_indices.min()] * 1000
    end_ms = times[time_indices.max()] * 1000
    common_title = (
        f"{phase} | {band} | {EFFECT_LABELS[effect]}\n"
        f"cluster {report_number}: {start_ms:.0f}-{end_ms:.0f} ms"
    )

    draw_heatmap(
        t_matrix,
        f"{common_title} | t-values",
        "t-value",
        "sensor_time_tvalue_heatmap",
    )
    draw_heatmap(
        contrast_matrix,
        f"{common_title} | contrast",
        "Band-power contrast (dB)",
        "sensor_time_contrast_heatmap",
    )
    plot_condition_level_sensor_time_heatmaps(
        effect,
        report_number=report_number,
        common_title=common_title,
        condition_data=condition_data,
        ordered_sensors=ordered_sensors,
        times=times,
        cluster_mask=cluster_mask,
        channel_names=channel_names,
        format_sensor_axis=format_sensor_axis,
        add_cluster_contour=add_cluster_contour,
        output_dir=output_dir,
    )


def condition_level_specs(effect):
    """Return condition groups to visualize without subtracting conditions."""
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


def plot_condition_level_sensor_time_heatmaps(
    effect,
    report_number,
    common_title,
    condition_data,
    ordered_sensors,
    times,
    cluster_mask,
    channel_names,
    format_sensor_axis,
    add_cluster_contour,
    output_dir,
):
    """Plot non-subtracted condition heatmaps with the significant mask marked."""
    specs = condition_level_specs(effect)
    matrices = []
    for _, condition_indices in specs:
        matrix = (
            condition_data[:, condition_indices][:, :, :, ordered_sensors]
            .mean(axis=(0, 1))
            .T
        )
        matrices.append(matrix)

    all_values = np.concatenate([matrix.ravel() for matrix in matrices])
    vmin = float(np.nanpercentile(all_values, 2))
    vmax = float(np.nanpercentile(all_values, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(all_values))
        vmax = float(np.nanmax(all_values))
    if vmin == vmax:
        vmax = vmin + np.finfo(float).eps

    n_panels = len(specs)
    fig_width = 4.3 * n_panels
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(fig_width, 4.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    image = None
    for axis, (label, _), matrix in zip(axes, specs, matrices):
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[
                times[0] * 1000,
                times[-1] * 1000,
                -0.5,
                len(ordered_sensors) - 0.5,
            ],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        add_cluster_contour(axis)
        axis.axvline(0, color="0.25", ls="--", lw=0.8)
        axis.set_title(label)
        axis.set_xlabel("Time (ms)")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    format_sensor_axis(axes[0])
    axes[0].set_ylabel("Cluster sensors")
    fig.suptitle(f"{common_title} | condition power", fontsize=12)
    colorbar = fig.colorbar(image, ax=axes, pad=0.015, shrink=0.9)
    colorbar.set_label("Band power (dB re 1 fT$^2$)")
    save_pdf(fig,
        output_dir / f"cluster_{report_number:02d}_condition_sensor_time_heatmaps.pdf"
    )
    plt.close(fig)


def make_report(t_obs, clusters, cluster_p_values, times, channel_names, band, effect):
    rows = []
    good_indices = np.where(cluster_p_values < CLUSTER_ALPHA)[0]
    phase_factor = N_PLANNED_BANDS * N_PLANNED_EFFECTS
    study_factor = N_PLANNED_PHASES * phase_factor

    for report_number, cluster_index in enumerate(good_indices, start=1):
        time_indices, sensor_indices = clusters[cluster_index]
        time_indices = np.asarray(time_indices, dtype=int)
        sensor_indices = np.asarray(sensor_indices, dtype=int)
        values = t_obs[time_indices, sensor_indices]
        peak_position = int(np.argmax(np.abs(values)))
        peak_time_index = int(time_indices[peak_position])
        peak_sensor_index = int(sensor_indices[peak_position])
        p_value = float(cluster_p_values[cluster_index])
        unique_sensors = np.unique(sensor_indices)
        rows.append(
            {
                "Cluster #": report_number,
                "Original Cluster Index": int(cluster_index),
                "Phase": PHASE,
                "Power Type": POWER_TYPE,
                "Band": band,
                "Effect": EFFECT_LABELS[effect],
                "Time Window (ms)": (
                    f"{times[time_indices.min()] * 1000:.0f} - "
                    f"{times[time_indices.max()] * 1000:.0f}"
                ),
                "Peak Time (ms)": f"{times[peak_time_index] * 1000:.0f}",
                "Peak Sensor": normalize_channel_name(
                    channel_names[peak_sensor_index]
                ),
                "Peak T-value": f"{values[peak_position]:.2f}",
                "Size (Sensors)": len(unique_sensors),
                "Size (Spatiotemporal Points)": len(sensor_indices),
                "Touches Analysis Boundary": bool(
                    time_indices.min() == 0
                    or time_indices.max() == len(times) - 1
                ),
                "Cluster-level pFWE": p_value,
                f"Phase-wise Bonferroni p (x{phase_factor})": min(
                    p_value * phase_factor, 1.0
                ),
                f"Study-wise Bonferroni p (x{study_factor})": min(
                    p_value * study_factor, 1.0
                ),
            }
        )
    return pd.DataFrame(rows), good_indices


def run_effect(band, effect, condition_data, times, channel_names, valid_subjects, skipped, adjacency):
    output_dir = result_dir(PHASE, band, effect)
    contrast_data = build_contrast(condition_data, effect)
    degrees_of_freedom = len(valid_subjects) - 1
    threshold = stats.t.ppf(
        1.0 - CLUSTER_FORMING_P / 2.0, degrees_of_freedom
    )
    print(
        f"Data: {contrast_data.shape}; threshold: |t| >= {threshold:.3f}; "
        f"permutations: {N_PERMUTATIONS}"
    )

    t_obs, clusters, cluster_p_values, h0 = (
        mne.stats.spatio_temporal_cluster_1samp_test(
            contrast_data,
            adjacency=adjacency,
            threshold=threshold,
            tail=0,
            n_permutations=N_PERMUTATIONS,
            seed=RANDOM_SEED,
            n_jobs=N_JOBS,
            out_type="indices",
            verbose=True,
        )
    )
    report, good_indices = make_report(
        t_obs, clusters, cluster_p_values, times, channel_names, band, effect
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "significant_cluster_report.csv", index=False)
    pd.DataFrame({"Subject": valid_subjects}).to_csv(
        output_dir / "valid_subjects.csv", index=False
    )
    (output_dir / "skipped_subjects.json").write_text(
        json.dumps(skipped, indent=2), encoding="utf-8"
    )

    with (output_dir / "cluster_results.pkl").open("wb") as file:
        pickle.dump(
            {
                "T_obs": t_obs,
                "clusters": clusters,
                "cluster_p_values": cluster_p_values,
                "H0": h0,
                "good_cluster_inds": good_indices,
                "times": times,
                "channel_names": channel_names,
                "valid_subjects": valid_subjects,
                "condition_data": condition_data,
                "contrast_data": contrast_data,
                "conditions": PHASE_CONDITIONS[PHASE],
                "phase": PHASE,
                "power_type": POWER_TYPE,
                "band": band,
                "band_definition": BANDS[band],
                "effect": effect,
                "effect_weights": EFFECT_WEIGHTS[effect],
                "cluster_forming_p": CLUSTER_FORMING_P,
                "threshold": threshold,
                "n_permutations": N_PERMUTATIONS,
                "seed": RANDOM_SEED,
                "normalization": "10*log10 power in fT^2; no event-local baseline",
            },
            file,
        )

    config_record = {
        "phase": PHASE,
        "power_type": POWER_TYPE,
        "band": band,
        "band_definition": BANDS[band],
        "effect": effect,
        "effect_weights": EFFECT_WEIGHTS[effect],
        "n_subjects": len(valid_subjects),
        "n_channels": len(channel_names),
        "n_permutations": N_PERMUTATIONS,
        "cluster_forming_p": CLUSTER_FORMING_P,
        "phase_family_tests": N_PLANNED_BANDS * N_PLANNED_EFFECTS,
        "study_family_tests": (
            N_PLANNED_PHASES * N_PLANNED_BANDS * N_PLANNED_EFFECTS
        ),
    }
    (output_dir / "analysis_config.json").write_text(
        json.dumps(config_record, indent=2), encoding="utf-8"
    )
    print(report.to_string(index=False) if len(report) else "No significant clusters.")
    print(f"Results saved to: {output_dir}")


def main():
    for band in BANDS:
        data, times, common_channels, subjects, skipped = load_group_condition_data(band)
        condition_data, _, adjacency, channels = build_info_and_adjacency(subjects[0], data, common_channels)
        del data
        for effect in EFFECT_WEIGHTS:
            print(f"Running {PHASE} | {band} | {effect}", flush=True)
            run_effect(band, effect, condition_data, times, channels, subjects, skipped, adjacency)
        del condition_data


def plot_saved_clusters(phase, band, effect):
    output_dir = result_dir(phase, band, effect)
    with (output_dir / "cluster_results.pkl").open("rb") as file:
        result = pickle.load(file)
    if not len(result["good_cluster_inds"]):
        return
    subject = result["valid_subjects"][0]
    epochs = mne.read_epochs(epoch_path(subject, phase), preload=False, verbose=False)
    picks = [epochs.ch_names.index(name) for name in result["channel_names"]]
    info = mne.pick_info(epochs.info, picks, copy=True)
    for number, index in enumerate(result["good_cluster_inds"], start=1):
        plot_cluster(phase, band, effect, number, index, result["T_obs"], result["clusters"],
                     result["cluster_p_values"], result["times"], info,
                     result["condition_data"], result["contrast_data"],
                     result["valid_subjects"], output_dir)


if __name__ == "__main__":
    main()
