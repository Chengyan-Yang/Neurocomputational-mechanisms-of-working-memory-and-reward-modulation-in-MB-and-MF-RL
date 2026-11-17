"""Plot 2x2 interaction summaries for significant TFR band-power clusters.

This script reads the interaction cluster statistics saved by
02_group_cluster_test.py. For each significant Load x Reward cluster, it
averages the saved condition-level band-power data over the exact
sensor-by-time points in the cluster, then plots the 2x2 interaction pattern.
The plotted values are descriptive follow-ups to the cluster test, not a new
inferential test.
"""

import json
import importlib
import pickle
import sys

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats


from tfr_analysis_config import (
    selected_indices,  # noqa: E402
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
LOAD_LEVELS = ("Low load", "High load")
MAGNITUDE_LEVELS = ("small reward", "large reward")
MAGNITUDE_COLORS = ("#4C78A8", "#D55E00")
PLOT_PRE_EVENT_S = 0.2


def plot_display_window(phase):
    """Return the plotting window, always including 200 ms before event onset."""
    configured_tmin, configured_tmax = DISPLAY_WINDOWS[phase]
    return min(configured_tmin, -PLOT_PRE_EVENT_S), configured_tmax


def load_cluster_results(phase, band, effect):
    path = result_dir(phase, band, effect) / "cluster_results.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cluster results: {path}\n"
            "Run 02_group_cluster_test.py for the same PHASE/POWER_TYPE/BAND/EFFECT first."
        )
    install_numpy_pickle_compat()
    with path.open("rb") as file:
        return pickle.load(file), path


def install_numpy_pickle_compat():
    """Allow NumPy 1.x to read pickle files written by NumPy 2.x."""
    try:
        importlib.import_module("numpy._core")
        return
    except ModuleNotFoundError:
        pass

    np_core = importlib.import_module("numpy.core")
    sys.modules.setdefault("numpy._core", np_core)
    for name in (
        "multiarray",
        "numeric",
        "fromnumeric",
        "_multiarray_umath",
        "numerictypes",
        "_dtype",
        "_methods",
        "shape_base",
    ):
        try:
            module = importlib.import_module(f"numpy.core.{name}")
        except ModuleNotFoundError:
            continue
        sys.modules.setdefault(f"numpy._core.{name}", module)


def temporal_runs(time_indices):
    unique = np.unique(np.asarray(time_indices, dtype=int))
    split_points = np.where(np.diff(unique) > 1)[0] + 1
    return np.split(unique, split_points)


def format_p(p_value):
    return "pFWE < .001" if p_value < 0.001 else f"pFWE = {p_value:.3f}"


def p_to_stars(p_value):
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def one_sample_effect_test(values):
    """Test whether a subject-level simple effect differs from zero."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {"t": np.nan, "p": np.nan, "stars": "n/a"}
    result = stats.ttest_1samp(values, popmean=0.0)
    p_value = float(result.pvalue)
    return {
        "t": float(result.statistic),
        "p": p_value,
        "stars": p_to_stars(p_value),
    }


def mean_sem(values, axis=0):
    mean = np.nanmean(values, axis=axis)
    sem = stats.sem(values, axis=axis, nan_policy="omit")
    return mean, sem


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

    linear_band_power = np.mean(tfr.data, axis=1)
    power_ft2 = np.maximum(linear_band_power * 1e30, np.finfo(float).tiny)
    log_power = 10.0 * np.log10(power_ft2)
    return log_power.mean(axis=0).astype(np.float32), tfr.times.copy()


def recompute_display_condition_timecourses(phase, band, valid_subjects, selected_channels):
    """Recompute condition timecourses from long epochs for plotting only."""
    subject_arrays = []
    reference_times = None
    skipped = []

    for subject in valid_subjects:
        try:
            epochs = mne.read_epochs(epoch_path(subject, phase), preload=True, verbose=False)
            missing_channels = [
                channel for channel in selected_channels if channel not in epochs.ch_names
            ]
            if missing_channels:
                raise RuntimeError(f"Missing cluster channels: {missing_channels[:5]}")
            balanced = balanced_condition_epochs(epochs, phase, subject)
            condition_arrays = []
            for condition_epochs in balanced:
                values, times = compute_display_band_timecourse(
                    phase, band,
                    condition_epochs,
                    selected_channels=selected_channels,
                )
                if reference_times is None:
                    reference_times = times
                elif not np.array_equal(reference_times, times):
                    raise RuntimeError("Display time axes do not match across subjects.")
                condition_arrays.append(values)
            subject_arrays.append(np.stack(condition_arrays))
        except Exception as error:
            skipped.append({"Subject": subject, "Error": repr(error)})
            print(f"sub-{subject} display timecourse failed: {error}")

    if skipped:
        raise RuntimeError(
            f"Display timecourse plotting failed for {len(skipped)} subjects."
        )
    if not subject_arrays:
        raise RuntimeError("No subjects were available for display timecourse plotting.")

    return np.stack(subject_arrays), reference_times


def cluster_condition_values(condition_data, time_indices, sensor_indices):
    """Return subject x condition means over exact cluster sensor-time points."""
    point_values = condition_data[:, :, time_indices, sensor_indices]
    return point_values.mean(axis=2)


def cluster_timecourses(condition_data, unique_sensors):
    """Return subject x condition x time means over the cluster sensors."""
    return condition_data[:, :, :, unique_sensors].mean(axis=3)


def add_cluster_time_spans(axis, times, time_indices):
    for run in temporal_runs(time_indices):
        axis.axvspan(
            times[run[0]] * 1000,
            times[run[-1]] * 1000,
            color="#F3D55B",
            alpha=0.28,
            linewidth=0,
        )


def set_clean_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", labelsize=9, width=1.0)


def add_condition_panel_stars(axis, x_positions, condition_matrix, condition_sem, tests):
    """Place simple-effect stars above the corresponding load level."""
    upper = condition_matrix + condition_sem
    lower = condition_matrix - condition_sem
    y_span = np.nanmax(upper) - np.nanmin(lower)
    if not np.isfinite(y_span) or y_span == 0:
        y_span = 1.0
    star_offset = 0.07 * y_span
    star_y = np.nanmax(upper, axis=1) + star_offset

    for x_position, y_position, test in zip(x_positions, star_y, tests):
        axis.text(
            x_position,
            y_position,
            test["stars"],
            ha="center",
            va="bottom",
            fontsize=12 if test["stars"] != "n.s." else 9,
            fontweight="bold" if test["stars"] != "n.s." else "normal",
            color="#222222",
        )

    ymin, ymax = axis.get_ylim()
    axis.set_ylim(ymin, max(ymax, float(np.nanmax(star_y) + star_offset)))


def plot_interaction_summary(
    phase, band, effect,
    report_number,
    cluster_index,
    cluster_results,
    output_dir,
    selection_label,
    file_prefix,
):
    condition_data = np.asarray(cluster_results["condition_data"], dtype=float)
    times = np.asarray(cluster_results["times"], dtype=float)
    valid_subjects = [f"{int(subject):02d}" for subject in cluster_results["valid_subjects"]]
    channel_names = [str(name) for name in cluster_results["channel_names"]]
    time_indices, sensor_indices = cluster_results["clusters"][cluster_index]
    time_indices = np.asarray(time_indices, dtype=int)
    sensor_indices = np.asarray(sensor_indices, dtype=int)
    unique_sensors = np.unique(sensor_indices)
    p_value = float(cluster_results["cluster_p_values"][cluster_index])

    condition_means = cluster_condition_values(
        condition_data, time_indices, sensor_indices
    )
    reward_effect_low_load = condition_means[:, 1] - condition_means[:, 0]
    reward_effect_high_load = condition_means[:, 3] - condition_means[:, 2]
    low_load_simple_effect = one_sample_effect_test(reward_effect_low_load)
    high_load_simple_effect = one_sample_effect_test(reward_effect_high_load)
    difference_of_differences = reward_effect_high_load - reward_effect_low_load
    weighted_interaction = 0.5 * difference_of_differences

    selected_channels = [channel_names[index] for index in unique_sensors]
    timecourses, display_times = recompute_display_condition_timecourses(
        phase, band,
        valid_subjects,
        selected_channels,
    )
    low_load_magnitude_tc = timecourses[:, 1] - timecourses[:, 0]
    high_load_magnitude_tc = timecourses[:, 3] - timecourses[:, 2]
    interaction_tc = high_load_magnitude_tc - low_load_magnitude_tc

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)

    x = np.arange(2)
    condition_matrix = np.column_stack(
        [
            [condition_means[:, 0].mean(), condition_means[:, 2].mean()],
            [condition_means[:, 1].mean(), condition_means[:, 3].mean()],
        ]
    )
    condition_sem = np.column_stack(
        [
            [stats.sem(condition_means[:, 0]), stats.sem(condition_means[:, 2])],
            [stats.sem(condition_means[:, 1]), stats.sem(condition_means[:, 3])],
        ]
    )

    for magnitude_index, (label, color) in enumerate(
        zip(MAGNITUDE_LEVELS, MAGNITUDE_COLORS)
    ):
        axes[0].errorbar(
            x,
            condition_matrix[:, magnitude_index],
            yerr=condition_sem[:, magnitude_index],
            marker="o",
            markersize=5.5,
            linewidth=1.9,
            capsize=4,
            color=color,
            label=label,
        )
    simple_effect_tests = (low_load_simple_effect, high_load_simple_effect)
    add_condition_panel_stars(
        axes[0],
        x_positions=x,
        condition_matrix=condition_matrix,
        condition_sem=condition_sem,
        tests=simple_effect_tests,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(LOAD_LEVELS)
    axes[0].set_ylabel("Cluster mean band power (dB re 1 fT$^2$)")
    axes[0].set_title("2x2 condition means")
    axes[0].legend(frameon=False, fontsize=9)
    set_clean_axis(axes[0])

    effect_values = np.column_stack(
        [reward_effect_low_load, reward_effect_high_load]
    )
    rng = np.random.default_rng(1000 + int(report_number))
    jitter = rng.uniform(-0.025, 0.025, size=effect_values.shape)
    for subject_values, subject_jitter in zip(effect_values, jitter):
        axes[1].plot(
            x + subject_jitter,
            subject_values,
            color="0.72",
            linewidth=0.7,
            alpha=0.55,
            zorder=1,
        )
    mean_effect, sem_effect = mean_sem(effect_values, axis=0)
    axes[1].errorbar(
        x,
        mean_effect,
        yerr=sem_effect,
        marker="o",
        markersize=6,
        linewidth=2.0,
        capsize=4,
        color="#222222",
        zorder=3,
    )
    y_span = np.nanmax(effect_values) - np.nanmin(effect_values)
    if not np.isfinite(y_span) or y_span == 0:
        y_span = 1.0
    star_offset = 0.08 * y_span
    star_y = mean_effect + sem_effect + star_offset
    for x_position, y_position, test in zip(x, star_y, simple_effect_tests):
        axes[1].text(
            x_position,
            y_position,
            test["stars"],
            ha="center",
            va="bottom",
            fontsize=12 if test["stars"] != "n.s." else 9,
            fontweight="bold" if test["stars"] != "n.s." else "normal",
            color="#222222",
        )
    ymin, ymax = axes[1].get_ylim()
    axes[1].set_ylim(ymin, max(ymax, float(np.nanmax(star_y) + star_offset)))
    axes[1].axhline(0, color="0.45", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(LOAD_LEVELS)
    axes[1].set_ylabel("Reward effect (large - small reward, dB)")
    axes[1].set_title("Simple reward effects")
    set_clean_axis(axes[1])

    for values, label, color in (
        (low_load_magnitude_tc, "Low load", "#4C78A8"),
        (high_load_magnitude_tc, "High load", "#D55E00"),
        (
            interaction_tc,
            "Interaction (high-load - low-load reward effect)",
            "#222222",
        ),
    ):
        mean, sem = mean_sem(values, axis=0)
        axes[2].plot(
            display_times * 1000,
            mean,
            linewidth=1.8,
            color=color,
            label=label,
        )
        axes[2].fill_between(
            display_times * 1000,
            mean - sem,
            mean + sem,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    add_cluster_time_spans(axes[2], times, time_indices)
    axes[2].axhline(0, color="0.45", linestyle="--", linewidth=0.8)
    axes[2].axvline(0, color="0.45", linestyle="--", linewidth=0.8)
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Reward effect / interaction (dB)")
    axes[2].set_title("Cluster-sensor timecourse (display only)")
    axes[2].legend(frameon=False, fontsize=8)
    set_clean_axis(axes[2])

    start_ms = times[time_indices.min()] * 1000
    end_ms = times[time_indices.max()] * 1000
    fig.suptitle(
        (
            f"{phase} | {band} | {EFFECT_LABELS[effect]} | {selection_label} {report_number}\n"
            f"{start_ms:.0f}-{end_ms:.0f} ms, {len(unique_sensors)} sensors, "
            f"{format_p(p_value)}; descriptive interaction = "
            f"{np.mean(difference_of_differences):.3f} dB"
        ),
        fontsize=12,
        fontweight="bold",
    )

    pdf_path = output_dir / f"{file_prefix}_{report_number:02d}_interaction_plot.pdf"
    save_pdf(fig, pdf_path)
    plt.close(fig)

    subject_rows = []
    for subject, values, low_effect, high_effect, ddi, weighted in zip(
        valid_subjects,
        condition_means,
        reward_effect_low_load,
        reward_effect_high_load,
        difference_of_differences,
        weighted_interaction,
    ):
        subject_rows.append(
            {
                "Subject": subject,
                CONDITION_LABELS[0]: values[0],
                CONDITION_LABELS[1]: values[1],
                CONDITION_LABELS[2]: values[2],
                CONDITION_LABELS[3]: values[3],
                "Reward effect at low load": low_effect,
                "Reward effect at high load": high_effect,
                "Difference of reward effects": ddi,
                "Weighted interaction contrast": weighted,
                "Low-load simple effect p": low_load_simple_effect["p"],
                "Low-load simple effect stars": low_load_simple_effect["stars"],
                "High-load simple effect p": high_load_simple_effect["p"],
                "High-load simple effect stars": high_load_simple_effect["stars"],
            }
        )
    subject_table = pd.DataFrame(subject_rows)
    subject_table.to_csv(
        output_dir / f"{file_prefix}_{report_number:02d}_interaction_values_by_subject.csv",
        index=False,
    )

    return {
        "Cluster": report_number,
        "Original Cluster Index": int(cluster_index),
        "Selection": selection_label,
        "N subjects": len(valid_subjects),
        "N cluster sensors": len(unique_sensors),
        "N cluster sensor-time points": len(sensor_indices),
        "Time Window (ms)": f"{start_ms:.0f} - {end_ms:.0f}",
        "pFWE": p_value,
        "Mean low load / small reward": float(np.mean(condition_means[:, 0])),
        "Mean low load / large reward": float(np.mean(condition_means[:, 1])),
        "Mean high load / small reward": float(np.mean(condition_means[:, 2])),
        "Mean high load / large reward": float(np.mean(condition_means[:, 3])),
        "Reward effect at low load": float(np.mean(reward_effect_low_load)),
        "Reward effect at low load t": low_load_simple_effect["t"],
        "Reward effect at low load p": low_load_simple_effect["p"],
        "Reward effect at low load stars": low_load_simple_effect["stars"],
        "Reward effect at high load": float(np.mean(reward_effect_high_load)),
        "Reward effect at high load t": high_load_simple_effect["t"],
        "Reward effect at high load p": high_load_simple_effect["p"],
        "Reward effect at high load stars": high_load_simple_effect["stars"],
        "Difference of reward effects": float(np.mean(difference_of_differences)),
        "Difference of reward effects SEM": float(stats.sem(difference_of_differences)),
        "Weighted interaction contrast": float(np.mean(weighted_interaction)),
        "Cluster sensors": ", ".join(channel_names[index] for index in unique_sensors),
        "Display window (ms)": (
            f"{display_times[0] * 1000:.0f} - {display_times[-1] * 1000:.0f}"
        ),
        "Figure": str(pdf_path),
    }


def main(phase, band, effect):
    if effect != "interaction":
        raise RuntimeError(
            "Interaction plots require "
            "EFFECT = 'interaction'."
        )

    cluster_results, cluster_path = load_cluster_results(phase, band, effect)
    good_cluster_inds = np.asarray(cluster_results["good_cluster_inds"], dtype=int)
    selection_label = "significant cluster"
    file_prefix = "cluster"
    if len(good_cluster_inds) == 0:
        cluster_p_values = np.asarray(cluster_results["cluster_p_values"], dtype=float)
        finite = np.where(np.isfinite(cluster_p_values))[0]
        if len(finite) == 0:
            print(f"No thresholded interaction clusters in {cluster_path}")
            return
        best_cluster = finite[np.argmin(cluster_p_values[finite])]
        good_cluster_inds = np.asarray([best_cluster], dtype=int)
        selection_label = "best non-significant cluster"
        file_prefix = "best_cluster"
        print(
            "No significant interaction clusters; plotting the lowest-p "
            "cluster descriptively."
        )

    output_dir = result_dir(phase, band, effect) / "interaction_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    saved_images = []
    for report_number, cluster_index in enumerate(good_cluster_inds, start=1):
        row = plot_interaction_summary(
            phase, band, effect,
            report_number=report_number,
            cluster_index=cluster_index,
            cluster_results=cluster_results,
            output_dir=output_dir,
            selection_label=selection_label,
            file_prefix=file_prefix,
        )
        summary_rows.append(row)
        saved_images.append(row["Figure"])

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "interaction_plot_summary.csv", index=False
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
                "cluster_selection": selection_label,
                "cluster_average": (
                    "Condition means are averaged over the exact significant "
                    "or selected sensor-by-time points in each cluster."
                ),
                "display_window": plot_display_window(phase),
                "display_window_note": (
                    "The third-panel timecourse is recomputed from long, "
                    "non-baselined epochs for visualization and may include "
                    "pre-event time. The group-level cluster test still uses "
                    "the saved post-event analysis window."
                ),
                "difference_of_reward_effects": (
                    "(large reward - small reward at high load) - "
                    "(large reward - small reward at low load)"
                ),
                "weighted_interaction_contrast": (
                    "0.5 * difference_of_reward_effects, matching the contrast "
                    "weights used in the cluster test."
                ),
                "simple_effect_stars": (
                    "Stars on the simple reward effects panel come from "
                    "one-sample t-tests of each subject-level reward effect "
                    "against zero: * p < .05, ** p < .01, *** p < .001; "
                    "n.s. = p >= .05."
                ),
                "inference_note": (
                    "These plots and CSV values are descriptive follow-ups. "
                    "Use the cluster-level pFWE from 02_group_cluster_test.py "
                    "for inference."
                ),
                "saved_images": saved_images,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Saved interaction plots:")
    for image in saved_images:
        print(image)
