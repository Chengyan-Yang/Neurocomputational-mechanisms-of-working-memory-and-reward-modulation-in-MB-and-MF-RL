"""Figure 3 source maps and interaction estimates."""
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from group_plots import DATA, RESULTS, brains, table


def save_pdf(fig, path, **kwargs):
    with plt.rc_context({"pdf.fonttype": 42, "pdf.use14corefonts": False}):
        fig.savefig(path, format="pdf", dpi=600, **kwargs)

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
            f"transition: interaction cluster {report_number}\n"
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

def main():
    section = "source_erf"
    out = RESULTS / "Fig_3"
    pairs = table(section,"Fig_3e_paired_amplitudes").pivot(index="panel_pair",columns="Condition",values="Cluster-mean dSPM")
    pairs.columns = [s.split(", ")[1].split()[0].lower()+"_"+s.split()[0].lower() for s in pairs]
    out.mkdir(parents=True,exist_ok=True)
    stats_table = table(section,"cluster_statistics")
    interaction_row = stats_table.query("phase=='transition' and effect=='interaction'").iloc[0]
    with np.load(DATA/section/"Fig_3e_transition_interaction_C1.npz") as z:
        results = dict(valid_subjects=pairs.index,cluster_p_values=[interaction_row.pFWE_exact],
                       times=z["time_s"],clusters=[(z["cluster_time_index"],z["cluster_space_index"])])
    values = pairs[["low_small","low_large","high_small","high_large"]].to_numpy()
    with plt.rc_context():
        plot_interaction_lines(results,[0],values[:,None,:],out)
    for row in table(section,"Fig_3_panel_index").itertuples():
        print(row.file,flush=True)
        with np.load(DATA/section/row.file) as z:
            prefix = row.file[:-4]
            n = row.saved_time_bins
            limits = z[f"positive_color_limits_{n}bin"]
            fig, ax = plt.subplots(figsize=(4.2,.6))
            mne.viz.plot_brain_colorbar(ax,dict(kind="value",pos_lims=list(limits)),orientation="horizontal",label="t value",bgcolor="white")
            save_pdf(fig,out/(prefix+"_colorbar.pdf"),bbox_inches="tight",transparent=True)
            plt.close(fig)
            for b in range(1,n+1):
                suffix = f"{n}bin_{b}"
                times = z["time_s_"+suffix]*1000
                idx = z["cluster_space_index"]
                left = len(z["vertices_lh"])
                hemis = [h for h, present in (("lh",any(idx<left)),("rh",any(idx>=left))) if present]
                brains(z,z["mean_t_"+suffix],out/f"{prefix}_bin{b}_{times.min():.0f}-{times.max():.0f}ms","source",limits,hemis)


if __name__ == "__main__":
    main()
