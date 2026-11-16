from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from validation_panel_style import recovery_panel, save_panel, save_subject_scatter, save_timecourse

HERE = Path(__file__).resolve().parent
INPUT = HERE.parents[1] / "group_data" / "BHV" / "validation" / "HBRL"
FIG = HERE / "results" / "validation" / "HBRL"
COLORS = {"low": "#eacdd2", "high": "#b03949"}
LIGHT = COLORS.copy()
LABELS = {"low": "Low load", "high": "High load"}
PPC_METRIC_LABELS = {
    "action_2_rate": "Action 2 choice rate",
    "mean_reward": "Mean reward",
    "stay_rate": "Stay rate",
    "optimal_rate": "Optimal rate",
}


def save(fig, name):
    save_panel(fig, FIG, name)

def make_figures():
    mcmc = pd.read_csv(INPUT / "hbrl_mcmc_parameter_diagnostics.csv")
    for diagnostic in ("rhat", "ess"):
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        for load in ("low", "high"):
            sub = mcmc[mcmc.load == load]
            x = range(len(sub)) if diagnostic == "rhat" else sub.min_bulk_ess
            y = sub.max_rhat if diagnostic == "rhat" else sub.min_tail_ess
            ax.scatter(x, y, color=COLORS[load], s=45, label=LABELS[load])
        if diagnostic == "rhat":
            ax.axhline(1.01, color="#555555", ls="--", lw=1)
            ax.set_xlabel("Parameter index")
            ax.set_ylabel("max Rhat")
        else:
            ax.axvline(400, color="#555555", ls="--", lw=1)
            ax.axhline(400, color="#555555", ls="--", lw=1)
            ax.set_xlabel("min bulk ESS")
            ax.set_ylabel("min tail ESS")
        ax.set_title("MCMC convergence" if diagnostic == "rhat" else "Effective sample size")
        ax.legend(frameon=False)
        save(fig, f"hbrl_mcmc_parameter_{diagnostic}")

    ppc = pd.read_csv(INPUT / "hbrl_ppc_group_metrics.csv")
    reps = pd.read_csv(INPUT / "hbrl_ppc_group_replicates.csv")
    for metric in PPC_METRIC_LABELS:
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        for j, load in enumerate(("low", "high")):
            vals = reps[(reps.load == load) & (reps.metric == metric)]["value"].to_numpy()
            parts = ax.violinplot(vals, positions=[j], widths=0.65, showmedians=True, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(LIGHT[load])
                body.set_edgecolor(COLORS[load])
                body.set_alpha(0.85)
            parts["cmedians"].set_color(COLORS[load])
            row = ppc[(ppc.load == load) & (ppc.metric == metric)].iloc[0]
            ax.scatter(j, row.pred_mean, color=COLORS[load], s=24, zorder=3)
            ax.scatter(j, row.observed, marker="D", color="#222222", s=36, zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Low load", "High load"])
        ax.set_title(PPC_METRIC_LABELS[metric].lower())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("Metric value")
        save(fig, f"hbrl_ppc_group_{metric}")

    subject = pd.read_csv(INPUT / "ppc_scatter_points.csv")
    save_subject_scatter(subject, COLORS, FIG, "hbrl_ppc_subject_scatter")
    save_timecourse(pd.read_csv(INPUT / "hbrl_ppc_optimal_rate_timecourse.csv"), COLORS, FIG,
                    "hbrl_ppc_optimal_rate_timecourse__learning_curve", expand_ylim=True)

def make_load_overlay_panels(comp, directory):
    for system in ("mf", "mb"):
        for reward in ("small", "large"):
            pairs = {}
            for load, suffix in (("low", "lload"), ("high", "hload")):
                param = f"beta_{system}_{reward}_{suffix}"
                pairs[load] = comp.loc[comp.parameter == param, ["generating", "recovered"]]
            title = rf"$\beta_{{\mathrm{{{system.upper()},\ {reward}}}}}$"
            fig = recovery_panel(title, pairs, COLORS, hierarchical=True)
            save_panel(fig, directory, f"parameter_recovery_beta_{system}_{reward}")

if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    make_figures()
    make_load_overlay_panels(pd.read_csv(INPUT / "parameter_recovery_points.csv"), FIG)
