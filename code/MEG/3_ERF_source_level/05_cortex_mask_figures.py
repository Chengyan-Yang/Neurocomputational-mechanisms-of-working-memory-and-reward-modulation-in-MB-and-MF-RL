"""Create brain maps, contrast time courses, and interaction plots."""

import pickle

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm

from cortex_mask_analysis_config import (
    ANALYSIS_TMAX,
    ANALYSIS_TMIN,
    BRAIN_BACKGROUND,
    BRAIN_SIZE,
    BRAIN_SURFACE,
    EFFECT_LABELS,
    EFFECT_WEIGHTS,
    N_TIME_BINS,
    PHASE,
    SUBJECTS_DIR,
    BRAIN_CORTEX,
    TEMPLATE_SUBJECT,
    VIEWS,
    conditions,
    output_dir,
    result_path,
    stc_stem,
)

from meg_pdf import save_brain_png


def save_pdf(fig, path, **kwargs):
    with plt.rc_context({"pdf.fonttype": 42, "pdf.use14corefonts": False}):
        fig.savefig(path, format="pdf", dpi=600, **kwargs)


def load_results(effect):
    path = result_path(effect)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing result file: {path}. Run 04_source_cortex_mask_group_stats.py first."
        )
    with path.open("rb") as file:
        results = pickle.load(file)

    expected = {
        "phase": PHASE,
        "effect": effect,
        "analysis_tmin": ANALYSIS_TMIN,
        "analysis_tmax": ANALYSIS_TMAX,
    }
    mismatches = {
        key: (results.get(key), value)
        for key, value in expected.items()
        if results.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"The PKL does not match the current configuration: {mismatches}"
        )
    return results


def make_template_stc(results, data):
    times = np.asarray(results["times"], dtype=float)
    if len(times) < 2:
        raise RuntimeError("At least two time samples are required.")
    return mne.SourceEstimate(
        data,
        vertices=[np.asarray(v, dtype=int) for v in results["full_vertices"]],
        tmin=float(times[0]),
        tstep=float(np.median(np.diff(times))),
        subject=TEMPLATE_SUBJECT,
    )


def split_cluster_times(time_inds):
    unique_times = np.unique(np.asarray(time_inds, dtype=int))
    return [part for part in np.array_split(unique_times, N_TIME_BINS) if len(part)]


def cluster_spatiotemporal_data(results, cluster_index):
    T_obs = np.asarray(results["T_obs"])
    time_inds, space_inds = results["clusters"][cluster_index]
    time_inds = np.asarray(time_inds, dtype=int)
    space_inds = np.asarray(space_inds, dtype=int)
    data = np.zeros((T_obs.shape[1], T_obs.shape[0]), dtype=T_obs.dtype)
    data[space_inds, time_inds] = T_obs[time_inds, space_inds]
    return data, time_inds, space_inds


def make_color_limits(results, good_indices):
    values = []
    for cluster_index in good_indices:
        cluster_data, time_inds, _ = cluster_spatiotemporal_data(
            results, cluster_index
        )
        for time_bin in split_cluster_times(time_inds):
            mean_data = cluster_data[:, time_bin].mean(axis=1)
            nonzero = np.abs(mean_data[np.nonzero(mean_data)])
            if len(nonzero):
                values.append(nonzero)

    if not values:
        raise RuntimeError("No non-zero cluster values were available for plotting.")
    values = np.concatenate(values)
    limits = np.percentile(values, [50, 75, 98]).astype(float)
    epsilon = max(np.finfo(float).eps, limits[-1] * 1e-6)
    for index in range(1, len(limits)):
        if limits[index] <= limits[index - 1]:
            limits[index] = limits[index - 1] + epsilon
    if limits[0] <= 0:
        limits[0] = epsilon
    return limits.tolist()


def plot_brain_maps(results, good_indices, figure_dir):
    limits = make_color_limits(results, good_indices)
    times = np.asarray(results["times"])
    n_lh = len(results["full_vertices"][0])
    manifest = []
    clim = {"kind": "value", "pos_lims": limits}
    fig, axis = plt.subplots(figsize=(4.2, 0.6))
    try:
        mne.viz.plot_brain_colorbar(axis, clim, orientation="horizontal",
                                    label="t value", bgcolor="white")
        save_pdf(fig, figure_dir / "brain_colorbar.pdf", bbox_inches="tight", transparent=True)
    finally:
        plt.close(fig)
    print(
        "Unified color limits: "
        + ", ".join(f"{value:.3f}" for value in limits)
    )

    for report_number, cluster_index in enumerate(good_indices, start=1):
        cluster_data, time_inds, space_inds = cluster_spatiotemporal_data(
            results, cluster_index
        )
        hemispheres = []
        if np.any(space_inds < n_lh):
            hemispheres.append("lh")
        if np.any(space_inds >= n_lh):
            hemispheres.append("rh")

        for bin_number, time_bin in enumerate(
            split_cluster_times(time_inds), start=1
        ):
            mean_data = cluster_data[:, time_bin].mean(axis=1, keepdims=True)
            stc_mean = make_template_stc(results, mean_data)
            start_ms = times[time_bin[0]] * 1000
            end_ms = times[time_bin[-1]] * 1000

            for hemi in hemispheres:
                for view in VIEWS:
                    filename = (
                        f"cluster_{report_number:02d}_bin_{bin_number:02d}_"
                        f"{start_ms:.0f}-{end_ms:.0f}ms_{hemi}_{view}.png"
                    )
                    brain = stc_mean.plot(
                        subject=TEMPLATE_SUBJECT, subjects_dir=str(SUBJECTS_DIR),
                        surface=BRAIN_SURFACE, cortex=BRAIN_CORTEX, hemi=hemi,
                        views=view, initial_time=stc_mean.times[0], clim=clim,
                        time_viewer=False, time_label=None, colorbar=False,
                        background=BRAIN_BACKGROUND, size=BRAIN_SIZE,
                        backend="pyvistaqt", title="",
                    )
                    try:
                        save_brain_png(brain, figure_dir / filename)
                    finally:
                        brain.close()
                    manifest.append(
                        {
                            "Cluster #": report_number,
                            "Original Cluster Index": int(cluster_index),
                            "Time Bin": bin_number,
                            "Time Window (ms)": f"{start_ms:.0f} - {end_ms:.0f}",
                            "Hemi": hemi,
                            "View": view,
                            "File": filename,
                        }
                    )
    pd.DataFrame(manifest).to_csv(
        figure_dir / "brain_figure_manifest.csv", index=False
    )


def load_union_contrast_data(results, good_indices):
    """Reload data needed for contrast time courses and interaction plots."""
    union_vertices = np.unique(
        np.concatenate(
            [
                np.asarray(results["clusters"][index][1], dtype=int)
                for index in good_indices
            ]
        )
    )
    condition_codes = tuple(results["conditions"])
    weights = np.asarray(results["effect_weights"], dtype=float)
    expected_times = np.asarray(results["times"])
    subject_arrays = []
    interaction_values = []

    for subject in tqdm(
        results["valid_subjects"],
        desc="Loading cluster time-course data",
    ):
        stcs = [
            mne.read_source_estimate(str(stc_stem(subject, condition)))
            for condition in condition_codes
        ]
        for stc in stcs:
            stc.crop(
                results["analysis_tmin"],
                results["analysis_tmax"],
                include_tmax=True,
            )
        if any(not np.array_equal(stc.times, expected_times) for stc in stcs):
            raise RuntimeError(f"Time mismatch for sub-{subject}.")

        contrast = sum(
            weight * stc.data[union_vertices]
            for weight, stc in zip(weights, stcs)
        )
        subject_arrays.append(contrast.T)

        if results["effect"] == "interaction":
            subject_cluster_values = []
            for cluster_index in good_indices:
                time_inds, space_inds = results["clusters"][cluster_index]
                time_inds = np.asarray(time_inds, dtype=int)
                space_inds = np.asarray(space_inds, dtype=int)
                subject_cluster_values.append(
                    [
                        float(np.mean(stc.data[space_inds, time_inds]))
                        for stc in stcs
                    ]
                )
            interaction_values.append(subject_cluster_values)

    lookup = np.full(len(results["T_obs"].T), -1, dtype=int)
    lookup[union_vertices] = np.arange(len(union_vertices))
    if results["effect"] == "interaction":
        interaction_values = np.asarray(interaction_values, dtype=float)
    else:
        interaction_values = None
    return np.asarray(subject_arrays), union_vertices, lookup, interaction_values


def temporal_runs(time_inds):
    unique_times = np.unique(np.asarray(time_inds, dtype=int))
    split_points = np.where(np.diff(unique_times) > 1)[0] + 1
    return np.split(unique_times, split_points)


def p_to_stars(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def add_sig_bracket(axis, x_center, y, height, label):
    x_start = x_center - 0.13
    x_end = x_center + 0.13
    axis.plot(
        [x_start, x_start, x_end, x_end],
        [y, y + height, y + height, y],
        color="black",
        linewidth=1.2,
        clip_on=False,
    )
    axis.text(
        x_center,
        y + height * 1.25,
        label,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )


def plot_interaction_lines(
    results,
    good_indices,
    interaction_values,
    figure_dir,
):
    """Plot the four condition means as a 2 x 2 interaction."""
    if interaction_values is None:
        return

    subjects = list(results["valid_subjects"])
    p_values = np.asarray(results["cluster_p_values"])
    times_ms = np.asarray(results["times"]) * 1000
    subject_rows = []
    summary_rows = []
    simple_effect_rows = []

    palette = {
        "small reward": "#2F6DB3",
        "large reward": "#C53A32",
    }
    markers = {
        "small reward": "o",
        "large reward": "s",
    }
    x_map = {"Low load": 0, "High load": 1}
    dodge = {"small reward": -0.055, "large reward": 0.055}

    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    for report_position, cluster_index in enumerate(good_indices):
        report_number = report_position + 1
        values = interaction_values[:, report_position, :]
        plot_df = pd.DataFrame(
            {
                "Subject": np.repeat(subjects, 4),
                "Magnitude": np.tile(
                    ["small reward", "large reward"] * 2, len(subjects)
                ),
                "Load": np.tile(
                    ["Low load", "Low load", "High load", "High load"],
                    len(subjects),
                ),
                "Cluster-mean dSPM": values.reshape(-1),
            }
        )
        summary_df = (
            plot_df.groupby(["Load", "Magnitude"], sort=False)[
                "Cluster-mean dSPM"
            ]
            .agg(["mean", "sem"])
            .reset_index()
        )

        for _, row in plot_df.iterrows():
            subject_rows.append(
                {
                    "Cluster #": report_number,
                    "Subject": row["Subject"],
                    "Condition": f"{row['Magnitude']}, {row['Load']}",
                    "Cluster-mean dSPM": row["Cluster-mean dSPM"],
                }
            )
        for row in summary_df.itertuples(index=False):
            summary_rows.append(
                {
                    "Cluster #": report_number,
                    "Condition": f"{row.Magnitude}, {row.Load}",
                    "Mean dSPM": row.mean,
                    "SEM": row.sem,
                }
            )

        data_lowload_smallmagnitude = values[:, 0]
        data_lowload_largemagnitude = values[:, 1]
        data_highload_smallmagnitude = values[:, 2]
        data_highload_largemagnitude = values[:, 3]
        t_lowload, p_lowload = stats.ttest_rel(
            data_lowload_largemagnitude,
            data_lowload_smallmagnitude,
            nan_policy="omit",
        )
        t_highload, p_highload = stats.ttest_rel(
            data_highload_largemagnitude,
            data_highload_smallmagnitude,
            nan_policy="omit",
        )
        p_lowload_adjusted = min(p_lowload * 2, 1.0)
        p_highload_adjusted = min(p_highload * 2, 1.0)
        for load_label, t_value, p_value, p_adjusted in (
            ("Low load", t_lowload, p_lowload, p_lowload_adjusted),
            ("High load", t_highload, p_highload, p_highload_adjusted),
        ):
            simple_effect_rows.append(
                {
                    "Cluster #": report_number,
                    "Simple Effect": f"Large reward - Small reward at {load_label}",
                    "t-value": t_value,
                    "Uncorrected p-value": p_value,
                    "Bonferroni-adjusted p-value (2 tests)": p_adjusted,
                }
            )

        fig, axis = plt.subplots(figsize=(3.7, 3.25), dpi=300)

        wide_df = plot_df.pivot_table(
            index="Subject",
            columns=["Load", "Magnitude"],
            values="Cluster-mean dSPM",
        )
        for _, subject_row in wide_df.iterrows():
            for magnitude in ("small reward", "large reward"):
                axis.plot(
                    [
                        x_map["Low load"] + dodge[magnitude],
                        x_map["High load"] + dodge[magnitude],
                    ],
                    [
                        subject_row[("Low load", magnitude)],
                        subject_row[("High load", magnitude)],
                    ],
                    color="0.82",
                    linewidth=0.6,
                    alpha=0.45,
                    zorder=1,
                )

        for magnitude in ("small reward", "large reward"):
            magnitude_summary = summary_df[summary_df["Magnitude"] == magnitude]
            x_values = [
                x_map[load] + dodge[magnitude] for load in magnitude_summary["Load"]
            ]
            axis.errorbar(
                x_values,
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
        bracket_height = y_range * 0.045
        lowload_y = max(
            np.nanmax(data_lowload_smallmagnitude),
            np.nanmax(data_lowload_largemagnitude),
        ) + y_range * 0.12
        highload_y = max(
            np.nanmax(data_highload_smallmagnitude),
            np.nanmax(data_highload_largemagnitude),
        ) + y_range * 0.12
        add_sig_bracket(
            axis,
            x_map["Low load"],
            lowload_y,
            bracket_height,
            p_to_stars(p_lowload_adjusted),
        )
        add_sig_bracket(
            axis,
            x_map["High load"],
            highload_y,
            bracket_height,
            p_to_stars(p_highload_adjusted),
        )

        time_inds = np.asarray(
            results["clusters"][cluster_index][0], dtype=int
        )
        start_ms = times_ms[time_inds.min()]
        end_ms = times_ms[time_inds.max()]
        p_value = float(p_values[cluster_index])
        p_text = "p < .001" if p_value < 0.001 else f"p = {p_value:.3f}"
        axis.axhline(0, color="0.25", linewidth=0.8, linestyle="--", zorder=0)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["Low load", "High load"], fontsize=10)
        axis.set_xlabel("WM load", fontsize=11, fontweight="bold")
        axis.set_ylabel(
            "Mean dSPM source estimate", fontsize=11, fontweight="bold"
        )
        axis.set_title(
            f"{PHASE}: interaction cluster {report_number}\n"
            f"{start_ms:.0f}-{end_ms:.0f} ms, {p_text}",
            fontsize=10.5,
            fontweight="bold",
            pad=8,
        )
        axis.legend(
            title="Reward magnitude",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            fontsize=9,
            title_fontsize=9,
            handlelength=1.6,
        )
        sns.despine(ax=axis)
        axis.tick_params(labelsize=9)
        axis.set_xlim(-0.35, 1.35)
        axis.set_ylim(
            y_min - y_range * 0.15,
            max(lowload_y, highload_y) + y_range * 0.23,
        )
        axis.get_legend().set_in_layout(False)
        fig.tight_layout()
        axis.get_legend().set_in_layout(True)

        stem = f"cluster_{report_number:02d}_interaction_plot"
        save_pdf(fig, figure_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame(subject_rows).to_csv(
        figure_dir / "interaction_plot_subject_values.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        figure_dir / "interaction_plot_summary.csv", index=False
    )
    pd.DataFrame(simple_effect_rows).to_csv(
        figure_dir / "interaction_plot_simple_effects.csv", index=False
    )


def plot_contrast_time_courses(results, good_indices, figure_dir):
    X_union, _, lookup, interaction_values = load_union_contrast_data(
        results, good_indices
    )
    times_ms = np.asarray(results["times"]) * 1000
    p_values = np.asarray(results["cluster_p_values"])

    for report_number, cluster_index in enumerate(good_indices, start=1):
        time_inds, full_space_inds = results["clusters"][cluster_index]
        full_space_inds = np.asarray(full_space_inds, dtype=int)
        union_positions = lookup[np.unique(full_space_inds)]
        if np.any(union_positions < 0):
            raise RuntimeError("Could not map cluster vertices to time-course data.")

        subject_time_courses = X_union[:, :, union_positions].mean(axis=2)
        mean_time_course = subject_time_courses.mean(axis=0)
        sem_time_course = stats.sem(
            subject_time_courses,
            axis=0,
            nan_policy="omit",
        )

        fig, axis = plt.subplots(figsize=(7.0, 3.6))
        axis.plot(times_ms, mean_time_course, color="#C62828", lw=2.2)
        axis.fill_between(
            times_ms,
            mean_time_course - sem_time_course,
            mean_time_course + sem_time_course,
            color="#C62828",
            alpha=0.16,
            linewidth=0,
        )
        for run in temporal_runs(time_inds):
            axis.axvspan(
                times_ms[run[0]],
                times_ms[run[-1]],
                color="#F3D55B",
                alpha=0.35,
                linewidth=0,
            )

        axis.axhline(0, color="0.25", lw=0.9, ls="--")
        axis.axvline(0, color="0.25", lw=0.9)
        axis.set_xlim(times_ms[0], times_ms[-1])
        axis.set_xlabel("Time (ms)")
        axis.set_ylabel("Mean dSPM contrast")
        p_value = float(p_values[cluster_index])
        p_text = "p < .001" if p_value < 0.001 else f"p = {p_value:.3f}"
        axis.set_title(
            f"{PHASE}: {EFFECT_LABELS[results['effect']]}, cluster {report_number} "
            f"({p_text})",
            fontsize=11,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.2)
        axis.spines["bottom"].set_linewidth(1.2)
        axis.tick_params(width=1.2, length=4, direction="out")
        fig.tight_layout()

        stem = f"cluster_{report_number:02d}_contrast_timecourse"
        save_pdf(fig, figure_dir / f"{stem}.pdf")
        plt.close(fig)

    if results["effect"] == "interaction":
        plot_interaction_lines(
            results,
            good_indices,
            interaction_values,
            figure_dir,
        )


def run_effect(effect):
    results = load_results(effect)
    good_indices = np.asarray(results["good_cluster_inds"], dtype=int)
    if len(good_indices) == 0:
        print("No significant clusters are available for visualization.")
        return

    figure_dir = output_dir(effect) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_brain_maps(results, good_indices, figure_dir)
    plot_contrast_time_courses(results, good_indices, figure_dir)
    print(f"Figures saved to: {figure_dir}")


def main():
    for effect in EFFECT_WEIGHTS:
        print(f"Running {PHASE}: {effect}")
        run_effect(effect)


if __name__ == "__main__":
    main()
