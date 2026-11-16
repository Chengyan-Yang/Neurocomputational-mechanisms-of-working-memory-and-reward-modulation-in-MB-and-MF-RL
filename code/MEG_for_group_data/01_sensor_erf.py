import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
from group_plots import RESULTS, CONDITIONS, table
from meg_pdf import save_pdf

WAVEFORM_FIGSIZE = (6.4, 3.1)
TOPOMAP_FIGSIZE = (4.4, 3.7)
AXIS_LINEWIDTH = 1.25
BASELINE_TMIN, BASELINE_TMAX = -.1, 0


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

def main():
    effect_dir = RESULTS / "Fig_S1"
    effect_dir.mkdir(parents=True,exist_ok=True)
    phase_label = "Feedback"
    effect_name = "WM load x Reward interaction"
    report_number = 1
    conditions = CONDITIONS
    condition_labels = ("Low load / small reward","Low load / large reward",
                        "High load / small reward","High load / large reward")
    colors = ("#a9c19c","#346db1","#eacdd2","#b03949")
    frame = table("sensor_erf","Fig_S1A_waveforms")
    display_times = frame.time_s.to_numpy()
    times = np.array([.545,.630])
    cluster_times = times
    unique_times = np.arange(len(times))
    start_ms,end_ms = times*1000
    statistics = table("sensor_erf","cluster_statistics")
    p_value = statistics.query("phase=='feedback' and effect=='interaction'").pFWE_exact.item()
    fig, axis = plt.subplots(figsize=WAVEFORM_FIGSIZE)
    fig_no_ci, axis_no_ci = plt.subplots(figsize=WAVEFORM_FIGSIZE)
    for condition, label, color in zip(
        conditions, condition_labels, colors
    ):
        mean_trace = frame[f"{condition}_mean_fT"].to_numpy()
        sem_trace = frame[f"{condition}_sem_fT"].to_numpy()
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


    spatial = table("sensor_erf","Fig_S1B_topography")
    t_map = spatial.mean_t.to_numpy()
    positions = spatial[["x_m","y_m"]].to_numpy()
    sensor_mask = spatial.in_cluster.to_numpy()
    vmax = max(abs(t_map))
    fig, axis = plt.subplots(figsize=TOPOMAP_FIGSIZE)
    image, _ = mne.viz.plot_topomap(
        t_map,
        positions,
        ch_type="mag",
        sphere=tuple(spatial.iloc[0][f'sphere_{i}_m'] for i in ('x','y','z','radius')),
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

    magnitude_levels = ("small reward", "large reward")
    load_levels = ("Low WM", "High WM")
    palette = {"small reward": "#346DB1", "large reward": "#B03949"}
    markers = {"small reward": "o", "large reward": "s"}
    x_positions = {"Low WM": 0.0, "High WM": 1.0}
    dodge = {"small reward": -0.055, "large reward": 0.055}

    wide = table("sensor_erf","Fig_S1C_paired_amplitudes_fT")
    values = wide[list(CONDITIONS)].to_numpy()
    valid_subjects = wide.panel_pair
    pvals = table("sensor_erf","Fig_S1C_simple_effects")["Bonferroni-adjusted p-value (2 tests)"]
    simple_effect_results = dict(zip(load_levels,pvals))
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


if __name__ == "__main__":
    main()
