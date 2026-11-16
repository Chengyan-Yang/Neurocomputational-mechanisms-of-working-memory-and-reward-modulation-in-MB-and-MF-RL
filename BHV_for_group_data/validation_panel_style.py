"""Fixed-size single panels shared by the RL and HBRL validation workflows."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from scipy.stats import pearsonr


def save_panel(fig, directory, name, *, overview=False, timecourse=False):
    ax, = fig.axes
    fig.set_layout_engine(None)
    if timecourse:
        fig.set_size_inches(9, 5.5)
        ax.set_position([1.25 / 9, 1.8 / 5.5, 7.35 / 9, 3.15 / 5.5])
    elif overview:
        fig.set_size_inches(10, 6.2)
        ax.set_position([.12, .23, .84, .65])
    else:
        fig.set_size_inches(5.6, 6.2)
        ax.set_position([1.25 / 5.6, 1.45 / 6.2, 4.2 / 5.6, 4.2 / 6.2])
        ax.set_box_aspect(1)

    if ax.get_legend() is not None:
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
                   fontsize=12, bbox_to_anchor=(.5, .015), columnspacing=1.2)
    for axis in (ax.xaxis, ax.yaxis):
        if isinstance(axis.get_major_locator(), MaxNLocator):
            axis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        axis.get_offset_text().set_fontsize(14)
    ax.tick_params(axis="both", labelsize=14, pad=5)
    ax.set_title(ax.get_title(), fontsize=18, pad=14)
    ax.set_xlabel(ax.get_xlabel(), fontsize=16, labelpad=9)
    ax.set_ylabel(ax.get_ylabel(), fontsize=16, labelpad=10)
    for note in ax.texts:
        note.set_fontsize(13)
    # The reference scales text after legend handles and tick labels exist.
    fig.canvas.draw()
    for text in fig.findobj(match=Text):
        text.set_fontfamily("Arial")
        text.set_fontsize(text.get_fontsize() * 1.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overflow = 3 - ax.yaxis.label.get_window_extent(renderer).x0
    if overflow > 0 and ax.get_ylabel():
        ax.yaxis.labelpad = max(0, ax.yaxis.labelpad - overflow * 72 / fig.dpi)
        fig.canvas.draw()
        overflow = 3 - ax.yaxis.label.get_window_extent(renderer).x0
        if overflow > 0:
            ax.tick_params(axis="y", pad=5 - overflow * 72 / fig.dpi)
    directory.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"pdf.fonttype": 42, "pdf.use14corefonts": False, "savefig.bbox": None}):
        fig.savefig(directory / f"{name}.pdf", format="pdf", dpi=600, facecolor="white")
    plt.close(fig)


def save_subject_scatter(table, colors, directory, name):
    """Compare participant-level observations with posterior predictive means."""
    for metric, title in (("mean_reward", "mean reward PPC"), ("optimal_rate", "optimal rate PPC")):
        fig, ax = plt.subplots()
        for load in ("low", "high"):
            rows = table[(table.metric == metric) & (table.load == load)]
            if rows.empty:
                raise ValueError(f"No subject PPC values for {metric} at {load} load")
            ax.scatter(rows.observed, rows.pred_mean, s=24, alpha=.82,
                       color=colors[load], label=f"{load.title()} load")
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        limits = (min(xmin, ymin), max(xmax, ymax))
        ax.plot(limits, limits, color="#555555", lw=1, ls="--")
        ax.set(xlim=limits, ylim=limits, xlabel="Observed", ylabel="PPC mean", title=title)
        ax.legend(frameon=False)
        save_panel(fig, directory, f"{name}__{metric}")


def save_timecourse(table, colors, directory, name, *, expand_ylim=False):
    fig, ax = plt.subplots()
    for load in ("low", "high"):
        rows = table[(table.load == load) & (table.metric == "optimal_rate")].sort_values("bin")
        x = rows["bin"].to_numpy()
        if not np.array_equal(x, np.arange(1, 9)):
            raise ValueError(f"Expected eight optimal-rate progress bins for {load} load")
        color = colors[load]
        ax.fill_between(x, rows["pred_q2.5"].to_numpy(), rows["pred_q97.5"].to_numpy(),
                        color=color, alpha=.25)
        ax.plot(x, rows["pred_mean"], color=color, lw=2.2, label=f"{load.title()} load PPC")
        ax.plot(x, rows["observed"], color=color, lw=2.2, ls="--", label=f"{load.title()} load observed")
    lo, hi = .45, .75
    if expand_ylim:
        plotted = table[["observed", "pred_q2.5", "pred_q97.5"]].to_numpy()
        lo, hi = min(lo, np.nanmin(plotted) - .02), max(hi, np.nanmax(plotted) + .02)
    ax.set(title="learning curve PPC", xlabel="Progress bin", ylabel="Optimal rate",
           xticks=np.arange(1, 9), ylim=(lo, hi))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    save_panel(fig, directory, name, timecourse=True)


def p_label(p):
    for threshold, label in ((.001, "p<.001"), (.01, "p<.01"), (.05, "p<.05")):
        if p < threshold:
            return label
    return f"p={p:.2f}"


def recovery_panel(title, pairs_by_load, colors, *, hierarchical=False):
    """Plot paired generating and recovered values, one series per load."""
    fig, ax = plt.subplots()
    notes, values = [], []
    for load in ("low", "high"):
        pairs = np.asarray(pairs_by_load[load], dtype=float)
        pairs = pairs[np.isfinite(pairs).all(axis=1)]
        if not len(pairs):
            raise ValueError(f"No finite recovery pairs for {title} at {load} load")
        x, y = pairs.T
        ax.scatter(x, y, s=26, alpha=.82, color=colors[load],
                   edgecolors="none" if hierarchical else "face",
                   zorder=2 if hierarchical else 1)
        values.append(pairs)
        stats = "r=n/a, p=n/a"
        if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
            r, p = pearsonr(x, y)
            stats = f"r={r:.2f}, {p_label(p)}"
        notes.append(f"{load.title()} load {stats}")

    values = np.concatenate(values)
    lo, hi = values.min(), values.max()
    pad = .08 * (hi - lo) if hi > lo else .1
    lo -= pad
    hi = lo + (hi + pad - lo) / .76
    ax.set(xlim=(lo, hi), ylim=(lo, hi), title=title,
           xlabel="True parameter", ylabel="Recovered parameter")
    ax.plot([lo, hi], [lo, hi], "--", color="#555555",
            lw=1.1 if hierarchical else 1, zorder=1 if hierarchical else 2)
    if hierarchical:
        ax.tick_params(width=.8, length=3)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        axis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.text(.035, .965, "\n".join(notes), transform=ax.transAxes, ha="left", va="top",
            color="black", zorder=5, bbox=dict(boxstyle="square,pad=.35", facecolor="white",
                                               edgecolor="black", linewidth=.9))
    return fig
