import matplotlib.pyplot as plt
import numpy as np
import mne
from group_plots import RESULTS, CONDITIONS, table
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
DISPLAY_WINDOWS = {'incentive': (-0.2, 1.0),
 'decision': (-0.2, 1.5),
 'transition': (-0.2, 0.5),
 'feedback': (-0.2, 1.5)}

EFFECT_LABELS = {
    "load": "High load - Low load",
    "magnitude": "Large reward - Small reward",
    "interaction": "Load x Reward interaction",
}



def plot_display_window(phase):
    """Return the plotting window, always including 200 ms before event onset."""
    configured_tmin, configured_tmax = DISPLAY_WINDOWS[phase]
    return min(configured_tmin, -PLOT_PRE_EVENT_S), configured_tmax

def style_axis(phase, axis):
    axis.axvline(0, color="0.35", linestyle="--", linewidth=0.85)
    display_tmin, display_tmax = plot_display_window(phase)
    axis.set_xlim(display_tmin * 1000, display_tmax * 1000)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", labelsize=9, width=1.0)

def format_p(p_value):
    return "p < .001" if p_value < 0.001 else f"p = {p_value:.3f}"

def plot_cluster(row, frame, spatial, spans, output_dir):
    phase,band,effect = row["Phase"],row["Band"],row["effect_key"]
    report_number = int(row["Cluster #"])
    display_times = frame.time_s.to_numpy()
    t_map = spatial.mean_t.to_numpy()
    sensor_mask = spatial.in_cluster.to_numpy()
    positions = spatial[["x_m","y_m"]].to_numpy()
    vmax = max(np.max(np.abs(t_map)),np.finfo(float).eps)
    start_ms, end_ms = np.array([spans.start_s.min(),spans.end_s.max()])*1000
    p_value = row.pFWE_exact
    fig, axis = plt.subplots(figsize=(4.4, 3.7))
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
            "markersize": 4,
        },
    )
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

    fig, axis = plt.subplots(figsize=(6.4, 3.2))
    for condition_index, (label, color) in enumerate(
        zip(CONDITION_LABELS, CONDITION_COLORS)
    ):
        key = CONDITIONS[condition_index]
        mean = frame[f"{key}_mean_dB"].to_numpy()
        sem = frame[f"{key}_sem_dB"].to_numpy()
        axis.plot(display_times * 1000, mean, color=color, lw=1.7, label=label)
        axis.fill_between(
            display_times * 1000,
            mean - sem,
            mean + sem,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    for run in spans.itertuples():
        axis.axvspan(run.start_s*1000, run.end_s*1000, color=SIGNIFICANT_SPAN_COLOR, alpha=.28, linewidth=0)
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Band power (dB re 1 fT$^2$)")
    axis.set_title(f"{phase} | {band} | cluster {report_number} (display only)")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    style_axis(phase, axis)
    fig.tight_layout()
    condition_pdf = output_dir / f"cluster_{report_number:02d}_condition_timecourses_display.pdf"
    save_pdf(fig, condition_pdf)
    plt.close(fig)

    mean_contrast = frame.contrast_mean_dB.to_numpy()
    sem_contrast = frame.contrast_sem_dB.to_numpy()
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
    for run in spans.itertuples():
        axis.axvspan(run.start_s*1000, run.end_s*1000, color=SIGNIFICANT_SPAN_COLOR, alpha=.28, linewidth=0)
    axis.axhline(0, color="0.35", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Band-power contrast (dB)")
    axis.set_title(f"{EFFECT_LABELS[effect]} | cluster {report_number}")
    style_axis(phase, axis)
    fig.tight_layout()
    contrast_pdf = output_dir / f"cluster_{report_number:02d}_contrast_timecourse_display.pdf"
    save_pdf(fig, contrast_pdf)
    plt.close(fig)



def main():
    curves = table("time_frequency","Fig_4_curves").groupby("cluster_key")
    maps = table("time_frequency","Fig_4_topographies").groupby("cluster_key")
    spans = table("time_frequency","Fig_4_significant_spans").groupby("cluster_key")
    for _, row in table("time_frequency","cluster_statistics").iterrows():
        key = row.cluster_key
        out = RESULTS/"Fig_4"/row.Phase/row.Band/row.effect_key
        out.mkdir(parents=True,exist_ok=True)
        plot_cluster(row,curves.get_group(key),maps.get_group(key),spans.get_group(key),out)


if __name__ == "__main__":
    main()
