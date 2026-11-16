from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from validation_panel_style import recovery_panel, save_panel, save_subject_scatter, save_timecourse

HERE = Path(__file__).resolve().parent
INPUT = HERE.parents[1] / "group_data" / "BHV" / "validation" / "RL"
FIG = HERE / "results" / "validation" / "RL"
COLORS = {"low": "#a9c19c", "high": "#346db1"}
LIGHT = COLORS.copy()
LABELS = {"low": "Low load", "high": "High load"}


WEIGHT_LABELS = {"w_small": r"$w_{\mathrm{small}}$", "w_large": r"$w_{\mathrm{large}}$"}



def save(fig, name):
    save_panel(fig, FIG, name, overview=name == "mcmc_posterior_means")


PPC_METRIC_LABELS = {
    "action_2_rate": "Action 2 choice rate",
    "mean_reward": "Mean reward",
    "stay_rate": "Stay rate",
    "optimal_rate": "Optimal rate",
}

def ppc_figures():
    group = pd.read_csv(INPUT / "ppc_group_metrics.csv")
    reps = pd.read_csv(INPUT / "ppc_group_replicates.csv")

    group = group[group.metric.isin(PPC_METRIC_LABELS) & group.load.isin(("low", "high"))].copy()
    for metric in PPC_METRIC_LABELS:
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        for j, load in enumerate(("low", "high")):
            row = group[(group.metric == metric) & (group.load == load)].iloc[0]
            vals = reps[(reps.metric == metric) & (reps.load == load)]["value"].to_numpy()
            parts = ax.violinplot(
                vals,
                positions=[j],
                widths=0.65,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(LIGHT[load])
                body.set_edgecolor(COLORS[load])
                body.set_alpha(0.85)
            parts["cmedians"].set_color(COLORS[load])
            ax.scatter(j, row.pred_mean, color=COLORS[load], s=24, zorder=3)
            ax.scatter(j, row.observed, marker="D", color="#222222", s=35, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Low load", "High load"])
        ax.set_title(PPC_METRIC_LABELS[metric].lower())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("Metric value")
        save(fig, f"ppc_group_{metric}")

    subject = pd.read_csv(INPUT / "ppc_scatter_points.csv")
    save_subject_scatter(subject, COLORS, FIG, "ppc_subject_scatter")
    save_timecourse(pd.read_csv(INPUT / "ppc_timecourse.csv"), COLORS, FIG,
                    "ppc_timecourse__learning_curve")
    return group

def recovery_figures():
    est = pd.read_csv(INPUT / "parameter_recovery_points.csv")
    rows = est
    for param, title in WEIGHT_LABELS.items():
        columns = [f"true_{param}", f"recovered_{param}"]
        pairs = {load: rows.loc[rows.load == load, columns] for load in ("low", "high")}
        fig = recovery_panel(title, pairs, COLORS)
        reward = "small" if param == "w_small" else "large"
        save(fig, f"recovery_scatter_empirical_map_w_{reward}")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    ppc_figures()
    recovery_figures()
