"""MCMC diagnostics, PPC and parameter recovery for the HBRL model."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / "_mplconfig"),
)

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hbrl_core import MCMC_PARAM_GROUPS, PARAMS, load_data, recursion

BASE = Path(__file__).resolve().parent

from validation_panel_style import save_panel, save_subject_scatter, save_timecourse
from ppc_timecourse import optimal_by_bin, timecourse_rows

OUT = Path(os.environ.get("HBRL52_OUTPUT_DIR", str(BASE / "outputs")))
MCMC = OUT / "mcmc"
PPC = OUT / "ppc"
FIG = OUT / "figures" / "single_panels"
COLORS = {"low": "#eacdd2", "high": "#b03949"}
LIGHT = COLORS.copy()
LABELS = {"low": "Low load", "high": "High load"}
PPC_METRIC_LABELS = {
    "action_2_rate": "Action 2 choice rate",
    "mean_reward": "Mean reward",
    "stay_rate": "Stay rate",
    "optimal_rate": "Optimal rate",
}


def ensure_dirs():
    for d in (MCMC, PPC, FIG):
        d.mkdir(parents=True, exist_ok=True)


def load_draws():
    draws = pd.read_csv(MCMC / "hbrl_subject_draws.csv")
    meta = pd.read_csv(MCMC / "stan_draw_export_metadata.csv").iloc[0].to_dict()
    subject_map = pd.read_csv(MCMC / "stan_subject_index.csv")
    expected = load_data()
    expected_ids = [sid for _, sid, _, _ in expected]
    expected_files = [filename for _, _, _, filename in expected]
    if len(subject_map) != 52 or subject_map["subject_index"].tolist() != list(range(1, 53)):
        raise ValueError("The exported fit subject map must contain subject_index 1..52.")
    if subject_map["subject_id"].astype(int).tolist() != expected_ids:
        raise ValueError("Fit subject IDs do not match the sorted raw-data CSV IDs.")
    if subject_map["source_file"].astype(str).tolist() != expected_files:
        raise ValueError("Fit CSV order does not match the raw-data CSV order.")
    if int(meta["subjects"]) != 52:
        raise ValueError(f"The posterior draw export reports {meta['subjects']} subjects, expected 52.")
    fit_trials = pd.read_csv(MCMC / "stan_fit_trials.csv")
    if len(fit_trials) != sum(len(z) for _, _, z, _ in expected):
        raise ValueError("Raw-data trial count differs from the saved Stan fit.")
    for index, sid, z, _ in expected:
        stored = fit_trials.loc[fit_trials.subject_index == index]
        transformed = np.column_stack((z[:, 0] + 1, z[:, 1] + 1, z[:, 2] + 1,
            z[:, 3], 1 + 4 * z[:, 4], z[:, 5] + 1, z[:, 6], z[:, 7] + 1, 2 - z[:, 7]))
        fitted = stored[["s1", "s2", "a", "r", "magnitude", "load", "part", "display_left", "display_right"]].to_numpy()
        if transformed.shape != fitted.shape or not np.array_equal(transformed, fitted):
            raise ValueError(f"Raw trials differ from saved Stan input for participant {sid}.")
    if draws.duplicated(["chain", "draw"]).any() or draws.groupby("chain").size().nunique() != 1:
        raise ValueError("Posterior draws must have unique chain/draw pairs and equal chain lengths.")
    if len(draws.chain.unique()) < 2 or not np.isfinite(draws.to_numpy(dtype=float)).all():
        raise ValueError("At least two finite posterior chains are required.")
    return draws.sort_values(["chain", "draw"]).reset_index(drop=True), meta


def array_for(draws, param):
    cols = [f"{param}_{i}" for i in range(1, 53)]
    missing = [col for col in cols if col not in draws.columns]
    if missing:
        raise ValueError(f"Missing subject draw columns for {param}: {missing[:5]}")
    draws = draws.sort_values(["chain", "draw"])
    arr = draws[cols].to_numpy(dtype=float)
    chains = sorted(draws["chain"].unique())
    n_draw = int(draws.groupby("chain").size().iloc[0])
    return arr.reshape(len(chains), n_draw, 52)


def run_mcmc_diagnostics():
    draws, meta = load_draws()
    subject_map = pd.read_csv(MCMC / "stan_subject_index.csv")
    detail_rows = []
    param_rows = []
    for load, label, param in MCMC_PARAM_GROUPS:
        arr = array_for(draws, param)
        idata = az.from_dict(posterior={param: arr})
        summ = az.summary(idata, var_names=[param], kind="diagnostics", round_to="none")
        for idx, row in summ.iterrows():
            subject_index = int(idx.split("[")[1].split("]")[0]) + 1
            sid = int(subject_map.loc[subject_map.subject_index == subject_index, "subject_id"].iloc[0])
            detail_rows.append(
                {
                    "load": load,
                    "parameter": label,
                    "stan_parameter": param,
                    "subject_index": subject_index,
                    "subject_id": sid,
                    "rhat": float(row["r_hat"]),
                    "bulk_ess": float(row["ess_bulk"]),
                    "tail_ess": float(row["ess_tail"]),
                    "mcse_mean": float(row["mcse_mean"]),
                    "mcse_sd": float(row["mcse_sd"]),
                }
            )
        d = pd.DataFrame(detail_rows)
        sub = d[(d.load == load) & (d.stan_parameter == param)]
        param_rows.append(
            {
                "load": load,
                "parameter": label,
                "stan_parameter": param,
                "n_subjects": int(sub.subject_id.nunique()),
                "max_rhat": float(sub.rhat.max()),
                "min_bulk_ess": float(sub.bulk_ess.min()),
                "min_tail_ess": float(sub.tail_ess.min()),
                "passed_parameter": bool(
                    np.isfinite(sub[["rhat", "bulk_ess", "tail_ess"]].to_numpy()).all()
                    and (sub.rhat.max() <= 1.01)
                    and (sub.bulk_ess.min() >= 400)
                    and (sub.tail_ess.min() >= 400)
                ),
            }
        )
    detail = pd.DataFrame(detail_rows)
    param = pd.DataFrame(param_rows)
    detail.to_csv(MCMC / "hbrl_mcmc_subject_parameter_diagnostics.csv", index=False)
    param.to_csv(MCMC / "hbrl_mcmc_parameter_diagnostics.csv", index=False)
    return detail, param, meta


def subject_params_from_row(row, subject_index):
    return np.array([row[f"{p}_{subject_index}"] for p in PARAMS], dtype=float)


def optimal_action(data):
    better_outcome = (data[:, 9] > data[:, 8]).astype(int)
    reversed_mapping = ~np.isin(data[:, 6], (0, 1))
    return better_outcome ^ reversed_mapping, data[:, 8] != data[:, 9]


def choice_metrics(data):
    action = data[:, 2].astype(int)
    reward = data[:, 3].astype(float)
    best, eligible = optimal_action(data)
    same_state = data[1:, 0].astype(int) == data[:-1, 0].astype(int)
    same_state &= (np.diff(data[:, 14]) == 1) & (np.diff(data[:, 5]) == 0)
    stays = action[1:][same_state] == action[:-1][same_state]
    return {
        "action_2_rate": float(np.mean(action == 1)),
        "mean_reward": float(np.mean(reward)),
        "stay_rate": float(np.mean(stays)) if len(stays) else np.nan,
        "optimal_rate": float(np.mean(action[eligible] == best[eligible])) if np.any(eligible) else np.nan,
    }


def metric_dict(data, load_high):
    sub = data[data[:, 5] == load_high]
    return choice_metrics(sub)


def summarize_distribution(values, observed):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(observed) or not len(values):
        return dict(observed=float(observed), pred_mean=np.nan, **{
            "pred_q2.5": np.nan, "pred_q97.5": np.nan,
            "p_ge_observed": np.nan, "bayes_p_two_sided": np.nan})
    p_ge = float(np.mean(values >= observed))
    p_le = float(np.mean(values <= observed))
    return {
        "observed": float(observed),
        "pred_mean": float(np.mean(values)),
        "pred_q2.5": float(np.quantile(values, 0.025)),
        "pred_q97.5": float(np.quantile(values, 0.975)),
        "p_ge_observed": p_ge,
        "bayes_p_two_sided": float(min(1.0, 2 * min(p_ge, p_le))),
    }


def run_ppc(reps=500, seed=20260908):
    draws, _ = load_draws()
    datasets = load_data()
    rng = np.random.default_rng(seed)
    draw_rows = draws.iloc[rng.integers(0, len(draws), size=reps)].reset_index(drop=True)
    subject_rows = []
    group_rows = []
    group_reps = []
    time_rows = []
    for load_name, load_high in (("low", 0), ("high", 1)):
        observed_by_subject = []
        sim_by_metric = {}
        time_obs = []
        time_rep = []
        for subject_index, sid, data, _ in datasets:
            mask = data[:, 5] == load_high
            best, eligible = optimal_action(data[mask])
            time_obs.append(optimal_by_bin(data[mask, 2], best, eligible))
            sim_bins = []
            observed = metric_dict(data, load_high)
            observed_by_subject.append(observed)
            subj_sim = {k: [] for k in observed}
            for rep in range(reps):
                x = subject_params_from_row(draw_rows.iloc[rep], subject_index)
                _, sim_data, _ = recursion(x, data, True, rng.random(len(data)))
                sim_bins.append(optimal_by_bin(sim_data[mask, 2], best, eligible))
                metrics = metric_dict(sim_data, load_high)
                for k, v in metrics.items():
                    subj_sim[k].append(v)
            time_rep.append(sim_bins)
            for k, vals in subj_sim.items():
                subject_rows.append(
                    {
                        "load": load_name,
                        "subject_id": sid,
                        "metric": k,
                        **summarize_distribution(vals, observed[k]),
                    }
                )
                sim_by_metric.setdefault(k, []).append(vals)

        time_rows.extend(timecourse_rows(load_name, time_obs, time_rep, summarize_distribution))
        for metric, vals in sim_by_metric.items():
            rep_means = np.nanmean(np.asarray(vals, dtype=float), axis=0)
            obs_mean = np.nanmean([x[metric] for x in observed_by_subject])
            for rep, value in enumerate(rep_means):
                group_reps.append(
                    {"load": load_name, "metric": metric, "rep": rep, "value": float(value)}
                )
            group_rows.append(
                {
                    "load": load_name,
                    "metric": metric,
                    **summarize_distribution(rep_means, obs_mean),
                }
            )

    pd.DataFrame(subject_rows).to_csv(PPC / "hbrl_ppc_subject_metrics.csv", index=False)
    pd.DataFrame(group_rows).to_csv(PPC / "hbrl_ppc_group_metrics.csv", index=False)
    pd.DataFrame(group_reps).to_csv(PPC / "hbrl_ppc_group_replicates.csv", index=False)
    pd.DataFrame(time_rows).to_csv(PPC / "hbrl_ppc_optimal_rate_timecourse.csv", index=False)
    (PPC / "ppc_run_metadata.json").write_text(json.dumps({
        "subjects": len(datasets), "replicates": reps, "seed": seed,
        "total_trials": sum(len(z) for _, _, z, _ in datasets),
        "loads": ["low", "high"], "metrics": list(PPC_METRIC_LABELS),
        "group_aggregation": "Equal-weight mean of subject-level metrics",
        "stay_definition": "Repeated action on consecutive original trials in the same state and load",
        "optimal_definition": "Choice with higher available reward; reward ties excluded",
    }, indent=2), encoding="utf-8")
    return pd.DataFrame(group_rows), pd.DataFrame(group_reps)


def save(fig, name):
    save_panel(fig, FIG, name)


def make_figures():
    all_diag_path = MCMC / "hbrl_mcmc_all_parameter_diagnostics.csv"
    if all_diag_path.exists():
        all_diag = pd.read_csv(all_diag_path)
        for diagnostic in ("rhat", "ess"):
            fig, ax = plt.subplots(figsize=(5.6, 5.2))
            if diagnostic == "rhat":
                ax.scatter(np.arange(len(all_diag)), all_diag["Rhat"], color=COLORS["high"], s=10, alpha=0.78)
                ax.axhline(1.01, color="#555555", ls="--", lw=1)
                ax.set_xlabel("Parameter index")
                ax.set_ylabel("Rank-normalized Rhat")
            else:
                ax.scatter(all_diag["Bulk_ESS"], all_diag["Tail_ESS"], color=COLORS["low"], s=10, alpha=0.78)
                ax.axvline(400, color="#555555", ls="--", lw=1)
                ax.axhline(400, color="#555555", ls="--", lw=1)
                ax.set_xlabel("Bulk ESS")
                ax.set_ylabel("Tail ESS")
            ax.set_title("MCMC convergence" if diagnostic == "rhat" else "Effective sample size")
            save(fig, f"hbrl_mcmc_all_parameter_{diagnostic}")

    mcmc = pd.read_csv(MCMC / "hbrl_mcmc_parameter_diagnostics.csv")
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

    if not (PPC / "hbrl_ppc_group_metrics.csv").exists():
        return
    ppc = pd.read_csv(PPC / "hbrl_ppc_group_metrics.csv")
    reps = pd.read_csv(PPC / "hbrl_ppc_group_replicates.csv")
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

    subject = pd.read_csv(PPC / "hbrl_ppc_subject_metrics.csv")
    save_subject_scatter(subject, COLORS, FIG, "hbrl_ppc_subject_scatter")
    save_timecourse(pd.read_csv(PPC / "hbrl_ppc_optimal_rate_timecourse.csv"), COLORS, FIG,
                    "hbrl_ppc_optimal_rate_timecourse__learning_curve", expand_ylim=True)




def markdown_table(df):
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


def write_report():
    mcmc = pd.read_csv(MCMC / "hbrl_mcmc_parameter_diagnostics.csv")
    ppc_path = PPC / "hbrl_ppc_group_metrics.csv"
    ppc = pd.read_csv(ppc_path) if ppc_path.exists() else None
    lines = []
    lines.append("# HBRL Model MCMC, PPC and Parameter Recovery")
    lines.append("")
    lines.append("The MEG hierarchical Bayesian HBRL model includes 52 participants. Core subject parameters and all saved Stan parameters are diagnosed separately. PPC simulates choices from posterior draws and evaluates action_2_rate, optimal_rate, stay_rate and mean_reward.")
    lines.append("")
    lines.append("## MCMC convergence checks")
    lines.append("")
    full_summary_path = MCMC / "hbrl_mcmc_all_parameter_summary.csv"
    if full_summary_path.exists():
        lines.append("Full saved Stan-parameter diagnostics:")
        lines.append("")
        lines.append(markdown_table(pd.read_csv(full_summary_path)))
        lines.append("")
    lines.append("Key subject-level parameter diagnostics:")
    lines.append("")
    lines.append(markdown_table(mcmc))
    lines.append("")
    lines.append("## Posterior predictive checks")
    lines.append("")
    ppc_metadata = PPC / "ppc_run_metadata.json"
    if ppc_metadata.exists():
        lines.extend([f"Run settings: `{ppc_metadata.read_text(encoding='utf-8').strip()}`", ""])
    if ppc is not None:
        ppc = ppc[ppc.metric.isin(PPC_METRIC_LABELS) & ppc.load.isin(("low", "high"))].copy()
    lines.append(markdown_table(ppc) if ppc is not None else "PPC has not been run.")
    lines.append("")
    lines.append("Four PPC single panels compare low and high load: `figures/single_panels/hbrl_ppc_group_<metric>.pdf`.")
    lines.append("Participant-level observed values and PPC means are plotted in `figures/single_panels/hbrl_ppc_subject_scatter__mean_reward.pdf` and `figures/single_panels/hbrl_ppc_subject_scatter__optimal_rate.pdf`.")
    lines.append("The eight-bin optimal-rate learning curve compares observed behavior with posterior predictions: `figures/single_panels/hbrl_ppc_optimal_rate_timecourse__learning_curve.pdf`.")
    lines.append("")
    lines.append("## Parameter recovery")
    lines.append("")
    label = os.environ.get("HBRL_RECOVERY_LABEL", "MEG_HBRL_5sets")
    full_recovery = BASE / "parameter_recovery" / f"outputs_{label}" / "parameter_recovery_summary.csv"
    if full_recovery.exists():
        lines.extend([
            "Parameter recovery uses only the full hierarchical Stan workflow in parameter_recovery.",
            "Generating values and recovered posterior medians are paired by dataset_id, subid and parameter. The sample count represents simulated dataset-subject pairs, not independent real participants.",
            "Correlations quantify recovery of individual rankings; also inspect bias, RMSE and the identity line. P values pooled across simulated datasets are descriptive.",
            "", f"Source: `{full_recovery}`", "", markdown_table(pd.read_csv(full_recovery)),
            "", f"Figures: `../parameter_recovery/outputs_{label}/Figures/single_panels/`",
        ])
    else:
        lines.append("No parameter recovery summary found. Run parameter_recovery/parameter_recovery_run_5sets.R.")
    (OUT / "REPORT_HBRL_validation.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--skip-ppc", action="store_true")
    parser.add_argument("--ppc-reps", type=int, default=500)
    args = parser.parse_args()
    if args.ppc_reps < 1:
        parser.error("PPC reps must be positive.")
    ensure_dirs()
    if args.report_only:
        make_figures()
        label = os.environ.get("HBRL_RECOVERY_LABEL", "MEG_HBRL_5sets")
        recovery_dir = BASE / "parameter_recovery" / f"outputs_{label}"
        if (recovery_dir / "parameter_recovery_comparison_long.csv").exists():
            subprocess.run([sys.executable, str(BASE / "parameter_recovery" / "plot_hbrl_recovery_figures.py"), "--label", label], check=True)
        write_report()
        return
    detail, param, meta = run_mcmc_diagnostics()
    print(param.to_string(index=False))
    if not args.skip_ppc:
        ppc, _ = run_ppc(reps=args.ppc_reps)
        print(ppc.to_string(index=False))
    make_figures()
    write_report()
    print("Wrote", OUT)


if __name__ == "__main__":
    main()


