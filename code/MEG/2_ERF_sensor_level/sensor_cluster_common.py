# -*- coding: utf-8 -*-
"""Shared sensor-level ERF cluster analysis for all task phases."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meg_project import DATA_ROOT as PROJECT_DATA_ROOT, meg_subjects, MEG_CODE, PREP_RESULTS, available
from meg_pdf import save_pdf

from dataclasses import dataclass
import pickle
import re
from typing import Optional

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import sparse, stats
from tqdm import tqdm


DATA_ROOT = PROJECT_DATA_ROOT / "MEG" / "preprocessed" / "evokeds"
OUTPUT_ROOT = MEG_CODE / "2_ERF_sensor_level" / "results" / "sensor"

SUBJECTS = meg_subjects()

N_PERMUTATIONS = 10000
RANDOM_SEED = 42
CLUSTER_FORMING_P = 0.01
CLUSTER_ALPHA = 0.05
N_EFFECTS_PER_PHASE = 3
N_PHASES = 4
N_JOBS = -1
DISPLAY_LOW_PASS_HZ = 20.0
DISPLAY_TMIN = -0.2
BASELINE_TMIN = -0.1
BASELINE_TMAX = 0.0
WAVEFORM_FIGSIZE = (6.4, 3.1)
TOPOMAP_FIGSIZE = (4.4, 3.7)
AXIS_LINEWIDTH = 1.25
INTERACTION_EFFECT_NAME = "WM load x Reward interaction"
CONDITION_TIMECOURSE_COLORS = (
    "#a9c19c",
    "#346db1",
    "#eacdd2",
    "#b03949",
)

PHASE_TABLE_LABELS = {'incentive': 'Incentive',
 'decision': 'Decision 0-1500 ms',
 'transition': 'Transition',
 'feedback': 'Feedback'}

SCALP_SIDE_LABELS = {
    "L": "Left",
    "R": "Right",
    "Z": "Midline",
}

SCALP_REGION_LABELS = {
    "F": "frontal",
    "T": "temporal",
    "C": "central",
    "P": "parietal",
    "O": "occipital",
}


@dataclass(frozen=True)
class SensorAnalysisConfig:
    phase: str
    phase_label: str
    conditions: tuple
    tmin: float
    tmax: float
    data_phase: Optional[str] = None


def _safe_name(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def _data_phase(config):
    return config.data_phase or config.phase


def _style_waveform_axis(axis):
    """Apply consistent publication styling to waveform axes."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)
    axis.spines["left"].set_linewidth(AXIS_LINEWIDTH)
    axis.tick_params(
        axis="both",
        which="major",
        width=AXIS_LINEWIDTH,
        length=4,
        direction="out",
    )


def _add_waveform_time_references(axis, xmin, xmax):
    """Show the actual baseline interval and event onset."""
    baseline_start = max(BASELINE_TMIN, xmin)
    baseline_end = min(BASELINE_TMAX, xmax)
    if baseline_start < baseline_end:
        axis.axvspan(
            baseline_start,
            baseline_end,
            color="#DCEAF4",
            alpha=0.75,
            linewidth=0,
            zorder=0,
        )
    axis.axvline(0, color="0.25", lw=0.9, ls="--")
    axis.set_xlim(xmin, xmax)


def _condition_key(epochs_or_evokeds, condition):
    condition = str(condition)
    for key in epochs_or_evokeds:
        if str(key) == condition:
            return key
    return None


def _read_complete_subjects(config):
    """Read only subjects with all four condition Evokeds."""
    subject_evokeds = {}
    skipped = []

    data_phase = _data_phase(config)
    for subject in tqdm(SUBJECTS, desc=f"Reading {data_phase} Evokeds"):
        evoked_file = (
            DATA_ROOT
            / f"sub-{subject}"
            / f"sub-{subject}_{data_phase}-ave.fif"
        )
        evoked_file = available(PREP_RESULTS / "evokeds" / evoked_file.relative_to(DATA_ROOT), evoked_file)
        if not evoked_file.exists():
            skipped.append((subject, "missing file"))
            continue

        evokeds = mne.read_evokeds(evoked_file, verbose=False)
        evoked_by_comment = {str(evoked.comment): evoked for evoked in evokeds}
        missing = [
            condition
            for condition in config.conditions
            if _condition_key(evoked_by_comment, condition) is None
        ]
        if missing:
            skipped.append((subject, f"missing conditions: {missing}"))
            continue

        condition_evokeds = {}
        for condition in config.conditions:
            key = _condition_key(evoked_by_comment, condition)
            evoked = evoked_by_comment[key].copy()
            picks = mne.pick_types(
                evoked.info,
                meg="mag",
                eeg=False,
                stim=False,
                exclude="bads",
            )
            evoked.pick(picks)
            condition_evokeds[str(condition)] = evoked

        subject_evokeds[subject] = condition_evokeds

    if not subject_evokeds:
        raise RuntimeError(f"No complete subjects found for {config.phase}.")

    print(
        f"Complete subjects: {len(subject_evokeds)}; "
        f"skipped: {len(skipped)}"
    )
    for subject, reason in skipped:
        print(f"  skipped sub-{subject}: {reason}")

    return subject_evokeds


def _normalize_channel_name(channel_name):
    return str(channel_name).split("-")[0].replace(" ", "")


def _scalp_region(channel_name):
    """Translate a CTF channel code such as MRO44 into a scalp region."""
    short_name = _normalize_channel_name(channel_name).upper()
    match = re.match(r"^M([LRZ])([FTCPO])", short_name)
    if match is None:
        return "Unclassified"
    side = SCALP_SIDE_LABELS.get(match.group(1), match.group(1))
    region = SCALP_REGION_LABELS.get(match.group(2), match.group(2))
    return f"{side} {region}"


def _peak_scalp_location(channel_name):
    short_name = _normalize_channel_name(channel_name)
    return f"{_scalp_region(short_name)} ({short_name})"


def _cluster_scalp_distribution(sensor_indices, channel_names, max_regions=3):
    """Summarize the largest scalp-region contributions to one cluster."""
    unique_sensors = np.unique(np.asarray(sensor_indices, dtype=int))
    counts = {}
    for sensor_index in unique_sensors:
        region = _scalp_region(channel_names[int(sensor_index)])
        counts[region] = counts.get(region, 0) + 1

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    total = len(unique_sensors)
    displayed = ordered[:max_regions]
    parts = [
        f"{region}: {count}/{total} ({100.0 * count / total:.1f}%)"
        for region, count in displayed
    ]
    remaining = sum(count for _, count in ordered[max_regions:])
    if remaining:
        parts.append(
            f"Other: {remaining}/{total} ({100.0 * remaining / total:.1f}%)"
        )
    return "; ".join(parts)


def _format_p_value(p_value):
    return "<0.001" if p_value < 0.001 else f"{p_value:.3f}"


def _comparison_label(config, effect_name):
    phase = PHASE_TABLE_LABELS.get(config.phase, config.phase)
    if effect_name == "WM load main effect":
        contrast = "HL - LL"
    elif effect_name == "Reward main effect":
        contrast = "Large reward - Small reward"
    else:
        contrast = "WM load x Reward"
    return f"{phase}: {contrast}"


def _equalize_and_build_adjacency(subject_evokeds, conditions):
    """Equalize channels and force data order to match adjacency order."""
    all_evokeds = [
        subject_evokeds[subject][condition]
        for subject in subject_evokeds
        for condition in conditions
    ]
    mne.equalize_channels(all_evokeds, copy=False)

    template = all_evokeds[0]
    full_adjacency, adjacency_names = mne.channels.find_ch_adjacency(
        template.info, ch_type="mag"
    )

    data_name_lookup = {
        _normalize_channel_name(name): name for name in template.ch_names
    }
    adjacency_indices = []
    ordered_data_names = []
    for adjacency_index, adjacency_name in enumerate(adjacency_names):
        normalized = _normalize_channel_name(adjacency_name)
        if normalized in data_name_lookup:
            adjacency_indices.append(adjacency_index)
            ordered_data_names.append(data_name_lookup[normalized])

    if not ordered_data_names:
        raise RuntimeError("No data channels matched the sensor adjacency.")
    if len(set(ordered_data_names)) != len(ordered_data_names):
        raise RuntimeError("Channel matching produced duplicate sensor names.")

    adjacency = sparse.csr_matrix(
        full_adjacency[adjacency_indices][:, adjacency_indices]
    )

    for evoked in all_evokeds:
        evoked.pick(ordered_data_names)
        if evoked.ch_names != ordered_data_names:
            evoked.reorder_channels(ordered_data_names)

    if adjacency.shape != (len(ordered_data_names), len(ordered_data_names)):
        raise RuntimeError(
            f"Adjacency shape {adjacency.shape} does not match "
            f"{len(ordered_data_names)} data channels."
        )

    print(f"Channels entering statistics: {len(ordered_data_names)}")
    return adjacency, ordered_data_names, all_evokeds[0].info.copy()


def _plot_gfp(subject_evokeds, valid_subjects, config, output_dir):
    """Plot descriptive GFP without changing the data used for statistics."""
    labels = (
        "Low load / small reward",
        "Low load / large reward",
        "High load / small reward",
        "High load / large reward",
    )
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")

    fig, axis = plt.subplots(figsize=WAVEFORM_FIGSIZE)
    for condition, label, color in zip(config.conditions, labels, colors):
        grand = mne.grand_average(
            [subject_evokeds[sub][condition] for sub in valid_subjects]
        )
        display_grand = grand.copy()
        if DISPLAY_LOW_PASS_HZ is not None:
            display_grand.filter(
                l_freq=None,
                h_freq=DISPLAY_LOW_PASS_HZ,
                verbose=False,
            )
        gfp_ft = np.sqrt(np.mean(display_grand.data ** 2, axis=0)) * 1e15
        axis.plot(display_grand.times, gfp_ft, label=label, color=color, lw=1.8)

    axis.axhline(0, color="0.75", lw=0.7)
    display_tmin = max(DISPLAY_TMIN, display_grand.tmin)
    _add_waveform_time_references(
        axis,
        display_tmin,
        min(config.tmax, display_grand.tmax),
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Global field power (fT)")
    axis.set_title(f"{config.phase_label}\nSensor-level ERF", fontsize=10.5)
    _style_waveform_axis(axis)
    fig.tight_layout()
    save_pdf(fig, output_dir / "grand_average_gfp.pdf")
    plt.close(fig)


def _baseline_qc(subject_evokeds, valid_subjects, config, output_dir):
    """Quantify the saved Evoked baseline without altering the data."""
    rows = []
    for subject in valid_subjects:
        for condition in config.conditions:
            evoked = subject_evokeds[subject][condition]
            baseline_mask = (
                (evoked.times >= BASELINE_TMIN - 1e-12)
                & (evoked.times <= BASELINE_TMAX + 1e-12)
            )
            if baseline_mask.sum() < 2:
                raise RuntimeError(
                    f"Baseline samples are unavailable for sub-{subject}, "
                    f"condition {condition}."
                )
            channel_baseline_ft = (
                evoked.data[:, baseline_mask].mean(axis=1) * 1e15
            )
            rows.append(
                {
                    "Subject": subject,
                    "Condition": condition,
                    "Baseline tmin (s)": BASELINE_TMIN,
                    "Baseline tmax (s)": BASELINE_TMAX,
                    "Mean across channels (fT)": np.mean(channel_baseline_ft),
                    "RMS across channels (fT)": np.sqrt(
                        np.mean(channel_baseline_ft ** 2)
                    ),
                    "Maximum absolute channel mean (fT)": np.max(
                        np.abs(channel_baseline_ft)
                    ),
                }
            )

    baseline_table = pd.DataFrame(rows)
    baseline_file = output_dir / "baseline_qc_by_subject_condition.csv"
    baseline_table.to_csv(baseline_file, index=False)
    print(
        "Largest absolute channel baseline mean: "
        f"{baseline_table['Maximum absolute channel mean (fT)'].max():.4f} fT"
    )
    print(f"Baseline QC saved to: {baseline_file}")


def _build_condition_arrays(
    subject_evokeds,
    valid_subjects,
    config,
    tmin,
    tmax,
    display_low_pass_hz=None,
):
    arrays = {condition: [] for condition in config.conditions}
    reference_times = None

    for subject in valid_subjects:
        for condition in config.conditions:
            evoked = subject_evokeds[subject][condition].copy()
            if display_low_pass_hz is not None:
                evoked.filter(
                    l_freq=None,
                    h_freq=display_low_pass_hz,
                    verbose=False,
                )
            evoked.crop(tmin=tmin, tmax=tmax, include_tmax=True)
            if reference_times is None:
                reference_times = evoked.times.copy()
            elif not np.array_equal(evoked.times, reference_times):
                raise RuntimeError(
                    f"Time axis mismatch for sub-{subject}, condition {condition}."
                )
            arrays[condition].append(evoked.data.T)

    arrays = {
        condition: np.asarray(values) for condition, values in arrays.items()
    }
    expected_shape = arrays[config.conditions[0]].shape
    for condition, values in arrays.items():
        if values.shape != expected_shape:
            raise RuntimeError(
                f"Condition {condition} shape {values.shape} does not match "
                f"{expected_shape}."
            )

    return arrays, reference_times


def _build_factorial_contrasts(condition_arrays, conditions):
    c1, c2, c3, c4 = (condition_arrays[c] for c in conditions)
    return {
        "WM load main effect": ((c3 + c4) - (c1 + c2)) / 2.0,
        "Reward main effect": ((c2 + c4) - (c1 + c3)) / 2.0,
        "WM load x Reward interaction": ((c1 + c4) - (c2 + c3)) / 2.0,
    }


def _cluster_rows(
    config,
    effect_name,
    t_obs,
    clusters,
    cluster_p_values,
    times,
    channel_names,
):
    rows = []
    significant = np.where(cluster_p_values < CLUSTER_ALPHA)[0]
    for report_number, cluster_index in enumerate(significant, start=1):
        time_inds, sensor_inds = clusters[cluster_index]
        time_inds = np.asarray(time_inds, dtype=int)
        sensor_inds = np.asarray(sensor_inds, dtype=int)
        cluster_values = t_obs[time_inds, sensor_inds]
        peak_position = int(np.argmax(np.abs(cluster_values)))
        peak_time_index = int(time_inds[peak_position])
        peak_sensor_index = int(sensor_inds[peak_position])
        p_value = float(cluster_p_values[cluster_index])
        unique_sensor_inds = np.unique(sensor_inds)
        n_cluster_sensors = len(unique_sensor_inds)
        phase_adjusted_p = min(
            p_value * N_EFFECTS_PER_PHASE,
            1.0,
        )
        global_adjusted_p = min(
            p_value * N_EFFECTS_PER_PHASE * N_PHASES,
            1.0,
        )
        peak_sensor = channel_names[peak_sensor_index]

        rows.append(
            {
                "Comparison": _comparison_label(config, effect_name),
                "Effect": effect_name,
                "Cluster #": report_number,
                "Original Cluster Index": int(cluster_index),
                "Time Window (ms)": (
                    f"{times[time_inds.min()] * 1000:.0f} - "
                    f"{times[time_inds.max()] * 1000:.0f}"
                ),
                "Peak Time (ms)": f"{times[peak_time_index] * 1000:.0f}",
                "Peak Sensor": _normalize_channel_name(peak_sensor),
                "Peak Scalp Location (Sensor)": _peak_scalp_location(
                    peak_sensor
                ),
                "Peak T-value": f"{cluster_values[peak_position]:.2f}",
                "Size (Sensors)": n_cluster_sensors,
                "Sensors (%)": (
                    f"{100.0 * n_cluster_sensors / len(channel_names):.1f}"
                ),
                "Sensors, n (%)": (
                    f"{n_cluster_sensors} "
                    f"({100.0 * n_cluster_sensors / len(channel_names):.1f}%)"
                ),
                "Cluster Scalp Distribution": _cluster_scalp_distribution(
                    unique_sensor_inds,
                    channel_names,
                ),
                "Size (Spatiotemporal Points)": len(sensor_inds),
                "Touches Analysis Boundary": bool(
                    time_inds.min() == 0 or time_inds.max() == len(times) - 1
                ),
                "Cluster-level pFWE": _format_p_value(p_value),
                "Phase-wise Bonferroni-adjusted p": _format_p_value(
                    phase_adjusted_p
                ),
                "Study-wise Bonferroni-adjusted p": _format_p_value(
                    global_adjusted_p
                ),
                "Survives within-phase Bonferroni": (
                    "Yes" if phase_adjusted_p < CLUSTER_ALPHA else "No"
                ),
                "Survives study-wise Bonferroni": (
                    "Yes" if global_adjusted_p < CLUSTER_ALPHA else "No"
                ),
            }
        )
    return rows


def _plot_clusters(
    effect_name,
    t_obs,
    clusters,
    cluster_p_values,
    times,
    info,
    phase_label,
    output_dir,
):
    significant = np.where(cluster_p_values < CLUSTER_ALPHA)[0]
    effect_dir = output_dir / _safe_name(effect_name)
    effect_dir.mkdir(parents=True, exist_ok=True)

    for report_number, cluster_index in enumerate(significant, start=1):
        time_inds, sensor_inds = clusters[cluster_index]
        time_inds = np.asarray(time_inds, dtype=int)
        sensor_inds = np.asarray(sensor_inds, dtype=int)
        unique_times = np.unique(time_inds)
        unique_sensors = np.unique(sensor_inds)

        t_map = t_obs[unique_times].mean(axis=0)
        sensor_mask = np.zeros(len(info.ch_names), dtype=bool)
        sensor_mask[unique_sensors] = True
        vmax = float(np.max(np.abs(t_map)))
        if vmax == 0:
            vmax = 1.0

        fig, axis = plt.subplots(figsize=TOPOMAP_FIGSIZE)
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
                "linewidth": 0,
                "markersize": 5,
            },
        )
        p_value = float(cluster_p_values[cluster_index])
        p_text = "p < .001" if p_value < 0.001 else f"p = {p_value:.3f}"
        axis.set_title(
            f"Cluster {report_number}: "
            f"{times[unique_times.min()] * 1000:.0f}-"
            f"{times[unique_times.max()] * 1000:.0f} ms\n{p_text}",
            fontsize=11,
        )
        colorbar = fig.colorbar(image, ax=axis, shrink=0.78, pad=0.05)
        colorbar.set_label("Mean t-value")
        fig.suptitle(f"{phase_label}\n{effect_name}", fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        stem = f"cluster_{report_number:02d}"
        save_pdf(fig, effect_dir / f"{stem}_topomap.pdf")
        plt.close(fig)


def _plot_cluster_condition_timecourses(
    effect_name,
    t_obs,
    clusters,
    cluster_p_values,
    cluster_times,
    display_times,
    condition_arrays,
    conditions,
    phase_label,
    channel_names,
    output_dir,
):
    """Plot descriptive condition traces averaged over cluster sensors."""
    condition_labels = (
        "Low load / small reward",
        "Low load / large reward",
        "High load / small reward",
        "High load / large reward",
    )
    colors = CONDITION_TIMECOURSE_COLORS
    significant = np.where(cluster_p_values < CLUSTER_ALPHA)[0]
    effect_dir = output_dir / _safe_name(effect_name)
    effect_dir.mkdir(parents=True, exist_ok=True)

    for report_number, cluster_index in enumerate(significant, start=1):
        time_inds, sensor_inds = clusters[cluster_index]
        time_inds = np.asarray(time_inds, dtype=int)
        sensor_inds = np.asarray(sensor_inds, dtype=int)
        unique_times = np.unique(time_inds)
        unique_sensors = np.unique(sensor_inds)
        cluster_values = t_obs[time_inds, sensor_inds]
        peak_position = int(np.argmax(np.abs(cluster_values)))
        peak_sign = np.sign(cluster_values[peak_position])
        if peak_sign == 0:
            peak_sign = 1.0

        # Align each sensor's field polarity to the peak sensor before averaging.
        # This prevents opposite magnetic-field lobes from cancelling each other.
        sensor_weights = np.ones(len(unique_sensors), dtype=float)
        for position, sensor_index in enumerate(unique_sensors):
            sensor_cluster_points = time_inds[sensor_inds == sensor_index]
            sensor_mean_t = np.mean(t_obs[sensor_cluster_points, sensor_index])
            sensor_sign = np.sign(sensor_mean_t)
            if sensor_sign == 0:
                sensor_sign = peak_sign
            sensor_weights[position] = sensor_sign * peak_sign

        fig, axis = plt.subplots(figsize=WAVEFORM_FIGSIZE)
        fig_no_ci, axis_no_ci = plt.subplots(figsize=WAVEFORM_FIGSIZE)
        for condition, label, color in zip(
            conditions, condition_labels, colors
        ):
            # Shape after selection: subjects x times x cluster sensors.
            selected = condition_arrays[condition][:, :, unique_sensors]
            subject_traces = np.mean(
                selected * sensor_weights[np.newaxis, np.newaxis, :],
                axis=2,
            ) * 1e15
            mean_trace = np.mean(subject_traces, axis=0)
            sem_trace = stats.sem(subject_traces, axis=0, nan_policy="omit")
            axis.plot(
                display_times,
                mean_trace,
                color=color,
                lw=1.7,
                label=label,
            )
            axis_no_ci.plot(
                display_times,
                mean_trace,
                color=color,
                lw=1.7,
                label=label,
            )
            axis.fill_between(
                display_times,
                mean_trace - sem_trace,
                mean_trace + sem_trace,
                color=color,
                alpha=0.16,
                linewidth=0,
            )

        # A cluster can contain disconnected temporal runs; shade each run.
        split_points = np.where(np.diff(unique_times) > 1)[0] + 1
        for time_run in np.split(unique_times, split_points):
            axis.axvspan(
                cluster_times[time_run[0]],
                cluster_times[time_run[-1]],
                color="0.5",
                alpha=0.12,
                linewidth=0,
            )
            axis_no_ci.axvspan(
                cluster_times[time_run[0]],
                cluster_times[time_run[-1]],
                color="0.5",
                alpha=0.12,
                linewidth=0,
            )

        axis.axhline(0, color="0.7", lw=0.8)
        _add_waveform_time_references(
            axis,
            display_times[0],
            display_times[-1],
        )
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Sign-aligned cluster field (fT)")
        axis.set_title(
            f"{phase_label}\n{effect_name}, cluster {report_number}",
            fontsize=10.5,
        )
        _style_waveform_axis(axis)
        fig.tight_layout()
        stem = f"cluster_{report_number:02d}_condition_timecourses"
        save_pdf(fig, effect_dir / f"{stem}.pdf")
        plt.close(fig)

        axis_no_ci.axhline(0, color="0.7", lw=0.8)
        _add_waveform_time_references(
            axis_no_ci,
            display_times[0],
            display_times[-1],
        )
        axis_no_ci.set_xlabel("Time (s)")
        axis_no_ci.set_ylabel("Sign-aligned cluster field (fT)")
        axis_no_ci.set_title(
            f"{phase_label}\n{effect_name}, cluster {report_number}",
            fontsize=10.5,
        )
        _style_waveform_axis(axis_no_ci)
        fig_no_ci.tight_layout()
        no_ci_stem = f"cluster_{report_number:02d}_condition_timecourses_no_ci"
        save_pdf(fig_no_ci, effect_dir / f"{no_ci_stem}.pdf")
        plt.close(fig_no_ci)

        # Unaligned waveform at the peak sensor is a less transformed QC view.
        peak_sensor_index = int(sensor_inds[peak_position])
        peak_sensor_name = channel_names[peak_sensor_index]
        fig, axis = plt.subplots(figsize=WAVEFORM_FIGSIZE)
        for condition, label, color in zip(
            conditions, condition_labels, colors
        ):
            subject_traces = (
                condition_arrays[condition][:, :, peak_sensor_index] * 1e15
            )
            mean_trace = np.mean(subject_traces, axis=0)
            sem_trace = stats.sem(
                subject_traces,
                axis=0,
                nan_policy="omit",
            )
            axis.plot(display_times, mean_trace, color=color, lw=1.7)
            axis.fill_between(
                display_times,
                mean_trace - sem_trace,
                mean_trace + sem_trace,
                color=color,
                alpha=0.16,
                linewidth=0,
            )

        axis.axhline(0, color="0.7", lw=0.8)
        _add_waveform_time_references(
            axis,
            display_times[0],
            display_times[-1],
        )
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(f"Magnetic field at {peak_sensor_name} (fT)")
        axis.set_title(
            f"{phase_label}\n{effect_name}, cluster {report_number} peak sensor",
            fontsize=10.5,
        )
        _style_waveform_axis(axis)
        fig.tight_layout()
        peak_stem = f"cluster_{report_number:02d}_peak_sensor_timecourses"
        save_pdf(fig, effect_dir / f"{peak_stem}.pdf")
        plt.close(fig)


def _extract_interaction_cluster_values(
    t_obs,
    cluster,
    condition_arrays,
    conditions,
):
    """Extract sign-aligned condition means from exact cluster points."""
    time_inds, sensor_inds = cluster
    time_inds = np.asarray(time_inds, dtype=int)
    sensor_inds = np.asarray(sensor_inds, dtype=int)
    cluster_t_values = t_obs[time_inds, sensor_inds]

    peak_position = int(np.argmax(np.abs(cluster_t_values)))
    peak_sign = np.sign(cluster_t_values[peak_position])
    if peak_sign == 0:
        peak_sign = 1.0

    # MEG field lobes can have opposite polarities. Align each sensor to the
    # peak lobe before averaging, using only the interaction statistic.
    point_weights = np.ones(len(sensor_inds), dtype=float)
    for sensor_index in np.unique(sensor_inds):
        sensor_mask = sensor_inds == sensor_index
        sensor_mean_t = np.mean(
            t_obs[time_inds[sensor_mask], sensor_index]
        )
        sensor_sign = np.sign(sensor_mean_t)
        if sensor_sign == 0:
            sensor_sign = peak_sign
        point_weights[sensor_mask] = sensor_sign * peak_sign

    values = np.empty(
        (condition_arrays[conditions[0]].shape[0], len(conditions)),
        dtype=float,
    )
    for condition_position, condition in enumerate(conditions):
        exact_cluster_data = condition_arrays[condition][
            :, time_inds, sensor_inds
        ]
        values[:, condition_position] = np.mean(
            exact_cluster_data * point_weights[np.newaxis, :],
            axis=1,
        ) * 1e15
    return values


def _p_to_stars(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _add_significance_bracket(axis, x_center, y, height, label):
    x_start = x_center - 0.13
    x_end = x_center + 0.13
    axis.plot(
        [x_start, x_start, x_end, x_end],
        [y, y + height, y + height, y],
        color="black",
        linewidth=1.1,
        clip_on=False,
        zorder=4,
    )
    axis.text(
        x_center,
        y + height * 1.2,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def _plot_interaction_lines(
    effect_name,
    t_obs,
    clusters,
    cluster_p_values,
    times,
    condition_arrays,
    conditions,
    valid_subjects,
    phase_label,
    output_dir,
):
    """Create source-style descriptive 2 x 2 interaction plots."""
    if effect_name != INTERACTION_EFFECT_NAME:
        return

    significant = np.where(cluster_p_values < CLUSTER_ALPHA)[0]
    if len(significant) == 0:
        return

    effect_dir = output_dir / _safe_name(effect_name)
    effect_dir.mkdir(parents=True, exist_ok=True)
    subject_rows = []
    summary_rows = []
    simple_effect_rows = []

    magnitude_levels = ("small reward", "large reward")
    load_levels = ("Low WM", "High WM")
    palette = {"small reward": "#346DB1", "large reward": "#B03949"}
    markers = {"small reward": "o", "large reward": "s"}
    x_positions = {"Low WM": 0.0, "High WM": 1.0}
    dodge = {"small reward": -0.055, "large reward": 0.055}

    for report_number, cluster_index in enumerate(significant, start=1):
        values = _extract_interaction_cluster_values(
            t_obs,
            clusters[cluster_index],
            condition_arrays,
            conditions,
        )
        plot_table = pd.DataFrame(
            {
                "Subject": np.repeat(valid_subjects, 4),
                "Load": np.tile(
                    ["Low WM", "Low WM", "High WM", "High WM"],
                    len(valid_subjects),
                ),
                "Magnitude": np.tile(
                    ["small reward", "large reward"] * 2,
                    len(valid_subjects),
                ),
                "Sign-aligned cluster field (fT)": values.reshape(-1),
            }
        )
        summary_table = (
            plot_table.groupby(["Load", "Magnitude"], sort=False)[
                "Sign-aligned cluster field (fT)"
            ]
            .agg(["mean", "sem"])
            .reset_index()
        )

        for row in plot_table.to_dict(orient="records"):
            subject_rows.append({"Cluster #": report_number, **row})
        for row in summary_table.itertuples(index=False):
            summary_rows.append(
                {
                    "Cluster #": report_number,
                    "Load": row.Load,
                    "Magnitude": row.Magnitude,
                    "Mean field (fT)": row.mean,
                    "SEM (fT)": row.sem,
                }
            )

        simple_effects = (
            ("Low WM", values[:, 1], values[:, 0]),
            ("High WM", values[:, 3], values[:, 2]),
        )
        simple_effect_results = {}
        for load, large_magnitude, small_magnitude in simple_effects:
            t_value, uncorrected_p = stats.ttest_rel(
                large_magnitude,
                small_magnitude,
                nan_policy="omit",
            )
            adjusted_p = min(float(uncorrected_p) * 2.0, 1.0)
            simple_effect_results[load] = adjusted_p
            simple_effect_rows.append(
                {
                    "Cluster #": report_number,
                    "Simple effect": f"large reward - small reward at {load}",
                    "t-value": float(t_value),
                    "Uncorrected p-value": float(uncorrected_p),
                    "Bonferroni-adjusted p-value (2 tests)": adjusted_p,
                    "Significance label": _p_to_stars(adjusted_p),
                }
            )

        fig, axis = plt.subplots(figsize=(4.4, 3.55), dpi=300)
        wide_table = plot_table.pivot_table(
            index="Subject",
            columns=["Load", "Magnitude"],
            values="Sign-aligned cluster field (fT)",
        )
        for _, subject_row in wide_table.iterrows():
            for magnitude in magnitude_levels:
                axis.plot(
                    [
                        x_positions["Low WM"] + dodge[magnitude],
                        x_positions["High WM"] + dodge[magnitude],
                    ],
                    [
                        subject_row[("Low WM", magnitude)],
                        subject_row[("High WM", magnitude)],
                    ],
                    color="0.82",
                    linewidth=0.6,
                    alpha=0.45,
                    zorder=1,
                )

        for magnitude in magnitude_levels:
            magnitude_summary = summary_table[
                summary_table["Magnitude"] == magnitude
            ]
            magnitude_summary = magnitude_summary.set_index("Load").loc[
                list(load_levels)
            ]
            axis.errorbar(
                [x_positions[load] + dodge[magnitude] for load in load_levels],
                magnitude_summary["mean"].to_numpy(),
                yerr=magnitude_summary["sem"].to_numpy(),
                color=palette[magnitude],
                marker=markers[magnitude],
                markersize=6.0,
                markerfacecolor="white",
                markeredgewidth=1.4,
                linewidth=1.8,
                elinewidth=1.2,
                capsize=3.5,
                label=magnitude,
                zorder=3,
            )

        y_min = float(np.nanmin(values))
        y_max = float(np.nanmax(values))
        y_range = y_max - y_min if y_max > y_min else 1.0
        bracket_height = y_range * 0.04
        load_value_columns = {
            "Low WM": (0, 1),
            "High WM": (2, 3),
        }
        for load in load_levels:
            bracket_y = float(
                np.nanmax(values[:, load_value_columns[load]])
            ) + y_range * 0.07
            _add_significance_bracket(
                axis,
                x_positions[load],
                bracket_y,
                bracket_height,
                _p_to_stars(simple_effect_results[load]),
            )

        time_inds = np.asarray(clusters[cluster_index][0], dtype=int)
        start_ms = times[time_inds.min()] * 1000
        end_ms = times[time_inds.max()] * 1000
        p_value = float(cluster_p_values[cluster_index])
        p_text = "pFWE < .001" if p_value < 0.001 else f"pFWE = {p_value:.3f}"

        axis.axhline(0, color="0.25", linewidth=0.8, linestyle="--")
        axis.set_xticks([0, 1])
        axis.set_xticklabels(load_levels)
        axis.set_xlabel("WM load", fontsize=11, fontweight="bold")
        axis.set_ylabel(
            "Sign-aligned cluster field (fT)",
            fontsize=10.5,
            fontweight="bold",
        )
        axis.set_title(
            f"{phase_label}: interaction cluster {report_number}\n"
            f"{start_ms:.0f}-{end_ms:.0f} ms, {p_text}",
            fontsize=9.8,
            fontweight="bold",
            pad=7,
        )
        axis.legend(
            title="Reward magnitude",
            frameon=False,
            loc="best",
            fontsize=9,
            title_fontsize=9,
            handlelength=1.6,
        )
        axis.set_xlim(-0.35, 1.35)
        axis.set_ylim(
            y_min - y_range * 0.12,
            y_max + y_range * 0.23,
        )
        _style_waveform_axis(axis)
        fig.tight_layout(pad=1.1)

        stem = f"cluster_{report_number:02d}_interaction_plot"
        save_pdf(fig, effect_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame(subject_rows).to_csv(
        effect_dir / "interaction_plot_subject_values.csv",
        index=False,
    )
    pd.DataFrame(summary_rows).to_csv(
        effect_dir / "interaction_plot_summary.csv",
        index=False,
    )
    pd.DataFrame(simple_effect_rows).to_csv(
        effect_dir / "interaction_plot_simple_effects.csv",
        index=False,
    )


def plot_saved_sensor_interaction(config):
    """Plot a saved interaction result without rerunning permutations."""
    output_dir = OUTPUT_ROOT / config.phase
    effect_slug = _safe_name(INTERACTION_EFFECT_NAME.replace("Reward", "Magnitude"))
    result_path = output_dir / f"{effect_slug}_cluster_results.pkl"
    if not result_path.exists():
        raise FileNotFoundError(f"Saved interaction result not found: {result_path}")

    with open(result_path, "rb") as file:
        results = pickle.load(file)

    if results.get("phase") != config.phase:
        raise RuntimeError("Saved result phase does not match the plotting config.")
    if tuple(results.get("conditions", ())) != tuple(config.conditions):
        raise RuntimeError("Saved result conditions do not match the plotting config.")

    subject_evokeds = _read_complete_subjects(config)
    saved_subjects = [f"{int(subject):02d}" for subject in results["valid_subjects"]]
    missing_subjects = [
        subject for subject in saved_subjects if subject not in subject_evokeds
    ]
    if missing_subjects:
        raise RuntimeError(
            "Evokeds are now missing for saved valid subjects: "
            + ", ".join(missing_subjects)
        )
    subject_evokeds = {
        subject: subject_evokeds[subject] for subject in saved_subjects
    }
    _, channel_names, _ = _equalize_and_build_adjacency(
        subject_evokeds,
        config.conditions,
    )
    if list(channel_names) != list(results["channel_names"]):
        raise RuntimeError(
            "Current Evoked channel order does not match the saved statistic."
        )

    condition_arrays, times = _build_condition_arrays(
        subject_evokeds,
        saved_subjects,
        config,
        config.tmin,
        config.tmax,
    )
    display_condition_arrays, display_times = _build_condition_arrays(
        subject_evokeds,
        saved_subjects,
        config,
        DISPLAY_TMIN,
        config.tmax,
        display_low_pass_hz=DISPLAY_LOW_PASS_HZ,
    )
    if not np.allclose(times, np.asarray(results["times"]), atol=1e-12):
        raise RuntimeError(
            "Current Evoked time axis does not match the saved statistic."
        )

    _plot_cluster_condition_timecourses(
        INTERACTION_EFFECT_NAME,
        np.asarray(results["T_obs"]),
        results["clusters"],
        np.asarray(results["cluster_p_values"]),
        times,
        display_times,
        display_condition_arrays,
        config.conditions,
        config.phase_label,
        channel_names,
        output_dir,
    )
    _plot_interaction_lines(
        INTERACTION_EFFECT_NAME,
        np.asarray(results["T_obs"]),
        results["clusters"],
        np.asarray(results["cluster_p_values"]),
        times,
        condition_arrays,
        config.conditions,
        saved_subjects,
        config.phase_label,
        output_dir,
    )
    print(f"Interaction plot saved to: {output_dir / effect_slug}")


def run_sensor_analysis(config):
    """Run all three factorial sensor-level contrasts for one phase."""
    if len(config.conditions) != 4:
        raise ValueError("Exactly four ordered 2 x 2 conditions are required.")

    output_dir = OUTPUT_ROOT / config.phase

    subject_evokeds = _read_complete_subjects(config)
    valid_subjects = list(subject_evokeds)
    adjacency, channel_names, plot_info = _equalize_and_build_adjacency(
        subject_evokeds,
        config.conditions,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_gfp(subject_evokeds, valid_subjects, config, output_dir)
    _baseline_qc(subject_evokeds, valid_subjects, config, output_dir)

    condition_arrays, times = _build_condition_arrays(
        subject_evokeds,
        valid_subjects,
        config,
        config.tmin,
        config.tmax,
    )
    display_condition_arrays, display_times = _build_condition_arrays(
        subject_evokeds,
        valid_subjects,
        config,
        DISPLAY_TMIN,
        config.tmax,
        display_low_pass_hz=DISPLAY_LOW_PASS_HZ,
    )
    contrasts = _build_factorial_contrasts(
        condition_arrays,
        config.conditions,
    )

    print(
        f"Data shape: subjects={len(valid_subjects)}, times={len(times)}, "
        f"sensors={len(channel_names)}"
    )
    all_rows = []
    for effect_name, X_data in contrasts.items():
        degrees_of_freedom = X_data.shape[0] - 1
        t_threshold = stats.t.ppf(
            1.0 - CLUSTER_FORMING_P / 2.0,
            degrees_of_freedom,
        )
        print(
            f"\n{effect_name}: two-tailed cluster threshold "
            f"t={t_threshold:.3f}, permutations={N_PERMUTATIONS}"
        )

        t_obs, clusters, cluster_p_values, H0 = (
            mne.stats.spatio_temporal_cluster_1samp_test(
                X_data,
                threshold=t_threshold,
                tail=0,
                adjacency=adjacency,
                n_permutations=N_PERMUTATIONS,
                seed=RANDOM_SEED,
                n_jobs=N_JOBS,
                out_type="indices",
                verbose=True,
            )
        )

        effect_slug = _safe_name(effect_name.replace("Reward", "Magnitude"))
        result_path = output_dir / f"{effect_slug}_cluster_results.pkl"
        with open(result_path, "wb") as file:
            pickle.dump(
                {
                    "T_obs": t_obs,
                    "clusters": clusters,
                    "cluster_p_values": cluster_p_values,
                    "H0": H0,
                    "times": times,
                    "channel_names": channel_names,
                    "valid_subjects": valid_subjects,
                    "phase": config.phase,
                    "conditions": config.conditions,
                    "effect": effect_name,
                    "tmin": config.tmin,
                    "tmax": config.tmax,
                    "display_tmin": DISPLAY_TMIN,
                    "display_low_pass_hz": DISPLAY_LOW_PASS_HZ,
                    "baseline_tmin": BASELINE_TMIN,
                    "baseline_tmax": BASELINE_TMAX,
                    "cluster_forming_p": CLUSTER_FORMING_P,
                    "t_threshold": t_threshold,
                    "tail": 0,
                    "n_permutations": N_PERMUTATIONS,
                    "seed": RANDOM_SEED,
                },
                file,
            )

        rows = _cluster_rows(
            config,
            effect_name,
            t_obs,
            clusters,
            cluster_p_values,
            times,
            channel_names,
        )
        all_rows.extend(rows)
        print(f"Significant clusters at p < .05: {len(rows)}")
        _plot_clusters(
            effect_name,
            t_obs,
            clusters,
            cluster_p_values,
            times,
            plot_info,
            config.phase_label,
            output_dir,
        )
        _plot_cluster_condition_timecourses(
            effect_name,
            t_obs,
            clusters,
            cluster_p_values,
            times,
            display_times,
            display_condition_arrays,
            config.conditions,
            config.phase_label,
            channel_names,
            output_dir,
        )
        _plot_interaction_lines(
            effect_name,
            t_obs,
            clusters,
            cluster_p_values,
            times,
            condition_arrays,
            config.conditions,
            valid_subjects,
            config.phase_label,
            output_dir,
        )

    columns = [
        "Comparison",
        "Effect",
        "Cluster #",
        "Original Cluster Index",
        "Time Window (ms)",
        "Peak Time (ms)",
        "Peak Sensor",
        "Peak Scalp Location (Sensor)",
        "Peak T-value",
        "Size (Sensors)",
        "Sensors (%)",
        "Sensors, n (%)",
        "Cluster Scalp Distribution",
        "Size (Spatiotemporal Points)",
        "Touches Analysis Boundary",
        "Cluster-level pFWE",
        "Phase-wise Bonferroni-adjusted p",
        "Study-wise Bonferroni-adjusted p",
        "Survives within-phase Bonferroni",
        "Survives study-wise Bonferroni",
    ]
    summary = pd.DataFrame(all_rows, columns=columns)
    summary.to_csv(output_dir / "significant_cluster_summary.csv", index=False)

    publication_columns = [
        "Comparison",
        "Cluster #",
        "Time Window (ms)",
        "Peak Time (ms)",
        "Peak T-value",
        "Sensors, n (%)",
        "Peak Scalp Location (Sensor)",
        "Cluster Scalp Distribution",
        "Touches Analysis Boundary",
        "Cluster-level pFWE",
        "Phase-wise Bonferroni-adjusted p",
        "Study-wise Bonferroni-adjusted p",
    ]
    summary[publication_columns].to_csv(
        output_dir / "significant_cluster_summary_publication.csv",
        index=False,
    )

    notes = (
        "Cluster-level pFWE values are family-wise error corrected across "
        "sensors and time within each effect using permutation testing.\n"
        "Phase-wise Bonferroni-adjusted p values control the three planned "
        "effects within each phase (p x 3).\n"
        "Study-wise Bonferroni-adjusted p values control seven phases by three "
        "effects (p x 21).\n"
        "Cluster scalp distribution lists the three largest CTF scalp regions; "
        "remaining sensors are grouped as Other.\n"
        "Scalp regions describe sensor locations and are not anatomical source "
        "localizations.\n"
    )
    (output_dir / "significant_cluster_summary_notes.txt").write_text(
        notes,
        encoding="utf-8",
    )

    participant_table = pd.DataFrame({"Subject": valid_subjects})
    participant_table.to_csv(output_dir / "valid_subjects.csv", index=False)
    print(f"\nResults saved to: {output_dir}")
    print(
        "Interaction follow-up p-values are not computed inside a cluster "
        "selected from the same data. Use independent or whole-space "
        "simple-effect tests for confirmatory inference."
    )
