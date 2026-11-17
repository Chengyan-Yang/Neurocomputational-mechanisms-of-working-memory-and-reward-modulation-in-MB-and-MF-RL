"""Create standalone validation panels and an English report; no refitting."""
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / "_mplconfig"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from model1_core import PARAMS
from run_ppc import PPC_METRIC_LABELS

BASE = Path(__file__).resolve().parent

from validation_panel_style import recovery_panel, save_panel, save_subject_scatter, save_timecourse
OUT = BASE / "outputs"
FIG = OUT / "figures" / "single_panels"

COLORS = {"low": "#a9c19c", "high": "#346db1"}
LIGHT = COLORS.copy()
LABELS = {"low": "Low load", "high": "High load"}


WEIGHT_LABELS = {"w_small": r"$w_{\mathrm{small}}$", "w_large": r"$w_{\mathrm{large}}$"}



def save(fig, name):
    save_panel(fig, FIG, name, overview=name == "mcmc_posterior_means")


def markdown_table(df, index=False):
    if index:
        df = df.reset_index()
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    headers = [str(c) for c in df.columns]
    rows = [[str(x) for x in row] for row in df.to_numpy()]
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def mcmc_parameter_table(summ):
    rows = []
    for (load, param), sub in summ.groupby(["load", "param"]):
        rows.append(
            {
                "load": load,
                "param": param,
                "n_subjects": int(sub["id"].nunique()),
                "max_rhat": float(sub.rhat.max()),
                "median_rhat": float(sub.rhat.median()),
                "min_bulk_ess": float(sub.ess_bulk.min()),
                "median_bulk_ess": float(sub.ess_bulk.median()),
                "min_tail_ess": float(sub.ess_tail.min()),
                "median_tail_ess": float(sub.ess_tail.median()),
                "median_mcse_mean": float(sub.mcse_mean.median()),
                "passed_parameter": bool(
                    (sub.rhat.max() <= 1.01)
                    and (sub.ess_bulk.min() >= 400)
                    and (sub.ess_tail.min() >= 400)
                ),
            }
        )
    out = pd.DataFrame(rows).sort_values(["load", "param"])
    out.to_csv(OUT / "mcmc" / "mcmc_parameter_diagnostics.csv", index=False)
    return out


def mcmc_figures():
    summ = pd.read_csv(OUT / "mcmc" / "mcmc_summary_final.csv")
    sub = pd.read_csv(OUT / "mcmc" / "mcmc_subject_diagnostics.csv")

    for diagnostic in ("rhat", "ess"):
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        for load in ("low", "high"):
            data = summ[summ.load == load]
            x = np.arange(len(data)) if diagnostic == "rhat" else data.ess_bulk
            y = data.rhat if diagnostic == "rhat" else data.ess_tail
            ax.scatter(x, y, s=15 if diagnostic == "rhat" else 18, alpha=0.82, color=COLORS[load], label=LABELS[load])
        if diagnostic == "rhat":
            ax.axhline(1.01, color="#555555", lw=1, ls="--")
            ax.set_title("MCMC convergence")
            ax.set_xlabel("Parameter index")
            ax.set_ylabel("Rank-normalized Rhat")
        else:
            ax.axvline(400, color="#555555", lw=1, ls="--")
            ax.axhline(400, color="#555555", lw=1, ls="--")
            ax.set_title("Effective sample size")
            ax.set_xlabel("Bulk ESS")
            ax.set_ylabel("Tail ESS")
        ax.legend(frameon=False)
        save(fig, f"mcmc_{diagnostic}")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    offset = {"low": -0.18, "high": 0.18}
    for i, p in enumerate(PARAMS):
        for load in ("low", "high"):
            vals = summ[(summ.param == p) & (summ.load == load)]["mean"]
            parts = ax.violinplot(
                vals,
                positions=[i + offset[load]],
                widths=0.30,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(LIGHT[load])
                body.set_edgecolor(COLORS[load])
                body.set_alpha(0.85)
            parts["cmedians"].set_color(COLORS[load])
            ax.scatter(
                np.full(len(vals), i + offset[load]),
                vals,
                s=8,
                alpha=0.82,
                color=COLORS[load],
            )
    ax.set_xticks(range(len(PARAMS)))
    ax.set_xticklabels([WEIGHT_LABELS.get(p, p) for p in PARAMS], rotation=25, ha="right")
    ax.set_ylabel("Posterior mean")
    ax.set_title("Posterior distributions by load")
    save(fig, "mcmc_posterior_means")
    param_diag = mcmc_parameter_table(summ)
    return summ, sub, param_diag


def ppc_figures():
    group = pd.read_csv(OUT / "ppc" / "ppc_group_metrics.csv")
    reps = pd.read_csv(OUT / "ppc" / "ppc_group_replicates.csv")

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

    subject = pd.read_csv(OUT / "ppc" / "ppc_subject_metrics.csv")
    save_subject_scatter(subject, COLORS, FIG, "ppc_subject_scatter")
    save_timecourse(pd.read_csv(OUT / "ppc" / "ppc_timecourse.csv"), COLORS, FIG,
                    "ppc_timecourse__learning_curve")
    return group



def recovery_figures():
    est = pd.read_csv(OUT / "recovery" / "recovery_estimates.csv")
    summ = pd.read_csv(OUT / "recovery" / "recovery_summary.csv")
    rows = est[est.truth_source == "empirical_map"]
    for param, title in WEIGHT_LABELS.items():
        columns = [f"true_{param}", f"recovered_{param}"]
        pairs = {load: rows.loc[rows.load == load, columns] for load in ("low", "high")}
        fig = recovery_panel(title, pairs, COLORS)
        reward = "small" if param == "w_small" else "large"
        save(fig, f"recovery_scatter_empirical_map_w_{reward}")
    return est, summ


def write_report(mcmc, sub, param_diag, ppc_group, rec):
    pass_counts = sub.groupby("load")["passed"].agg(["sum", "count"]).reset_index()
    subject_counts = sub.groupby("load")["id"].nunique().to_dict()
    likelihood_path = OUT / "likelihood_validation.csv"
    if likelihood_path.exists():
        likelihood = pd.read_csv(likelihood_path)
        likelihood_counts = likelihood.groupby("load")["id"].nunique().to_dict()
        max_abs_diff = float(likelihood["difference"].abs().max())
        likelihood_note = (
            "Likelihood validation completed: "
            + ", ".join(
                f"{load}: {count} subjects"
                for load, count in sorted(likelihood_counts.items())
            )
            + f"; maximum absolute log-likelihood difference: {max_abs_diff:.3g}."
        )
    else:
        likelihood_note = "Likelihood validation has not been run. Run `validate_likelihood.py` to generate the results."
    ppc_flags = ppc_group[ppc_group.bayes_p_two_sided < 0.05].copy()
    rec_focus = rec[(rec.truth_source == "empirical_map") & (rec.param.isin(WEIGHT_LABELS))]
    rec_pivot = rec_focus.pivot_table(index="param", columns="load", values="pearson_r")

    lines = []
    lines.append("# Model 1 MCMC, PPC and Parameter Recovery")
    lines.append("")
    load_n = ", ".join(
        f"{load}: {subject_counts.get(load, 0)} subjects"
        for load in ("low", "high")
    )
    lines.append(f"Model 1 low-load and high-load data ({load_n}).")
    lines.append("")
    lines.append("Behavioral CSVs come from `learning/low_load` and `learning/high_load`. Rows with missing values and probe trials are removed following MATLAB preprocessing. Participant IDs are matched to the model_1 MAP fits.")
    lines.append("")
    lines.append(likelihood_note)
    lines.append("")
    lines.append("## MCMC convergence checks")
    lines.append("")
    for _, r in pass_counts.iterrows():
        lines.append(f"- {r['load']}: {int(r['sum'])}/{int(r['count'])} subjects passed Rhat <= 1.01, bulk ESS >= 400 and tail ESS >= 400.")
    lines.append("")
    lines.append("Parameter-level diagnostics:")
    lines.append("")
    lines.append(markdown_table(param_diag, index=False))
    lines.append("")
    lines.append("Single panels: `figures/single_panels/mcmc_rhat.pdf`, `figures/single_panels/mcmc_ess.pdf`, `figures/single_panels/mcmc_posterior_means.pdf`.")
    lines.append("")
    lines.append("## Posterior predictive checks")
    lines.append("")
    lines.append(markdown_table(ppc_group, index=False))
    if len(ppc_flags):
        lines.append("")
        lines.append("PPC metrics with two-sided Bayesian p < .05:")
        lines.append("")
        lines.append(markdown_table(ppc_flags[["load", "metric", "observed", "pred_mean", "pred_q2.5", "pred_q97.5", "bayes_p_two_sided"]], index=False))
    else:
        lines.append("")
        lines.append("No group PPC metric had two-sided Bayesian p < .05.")
    lines.append("")
    lines.append("Four single panels: `figures/single_panels/ppc_group_<metric>.pdf`. Each panel compares low and high load.")
    lines.append("Participant-level observed values and PPC means are plotted in `figures/single_panels/ppc_subject_scatter__mean_reward.pdf` and `figures/single_panels/ppc_subject_scatter__optimal_rate.pdf`.")
    lines.append("The eight-bin optimal-rate learning curve compares observed behavior with posterior predictions: `figures/single_panels/ppc_timecourse__learning_curve.pdf`.")
    lines.append("")
    lines.append("## Parameter recovery")
    lines.append("")
    lines.append("Recovery was run in two designs: simulations from each subject's empirical MAP estimates, and a broader parameter-space simulation set. Reported below are empirical-MAP Pearson correlations.")
    lines.append("")
    lines.append(markdown_table(rec_pivot, index=True))
    lines.append("")
    lines.append("Full recovery statistics are in `recovery/recovery_summary.csv`; simulated estimates are in `recovery/recovery_estimates.csv`. The two empirical-MAP panels (w_small and w_large) overlay both loads and are saved as PNG and SVG in `figures/single_panels/`.")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append("- `mcmc/mcmc_summary_final.csv`: posterior summaries and diagnostics for every parameter, subject and load.")
    lines.append("- `mcmc/mcmc_parameter_diagnostics.csv`: parameter-level MCMC diagnostics without subject-level rows.")
    lines.append("- `mcmc/mcmc_subject_diagnostics.csv`: subject-level pass/fail summary.")
    lines.append("- `ppc/ppc_group_metrics.csv`, `ppc/ppc_group_replicates.csv`, `ppc/ppc_subject_metrics.csv`: PPC summaries.")
    lines.append("- `recovery/recovery_summary.csv`, `recovery/recovery_estimates.csv`: parameter recovery results.")
    lines.append("- `likelihood_validation.csv`: Python recursion checked against the original MATLAB MAP CSV log-likelihoods.")
    lines.append("")
    lines.append("Diagnostic definitions follow rank-normalized split Rhat/ESS recommendations from Vehtari et al. and standard posterior predictive comparison of observed summary statistics against replicated data.")
    (OUT / "REPORT_model1_validation.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    mcmc, sub, param_diag = mcmc_figures()
    ppc_group = ppc_figures()
    _, rec = recovery_figures()
    write_report(mcmc, sub, param_diag, ppc_group, rec)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
