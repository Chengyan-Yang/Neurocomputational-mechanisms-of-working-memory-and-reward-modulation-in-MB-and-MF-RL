"""Posterior predictive checks for model_1."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / "_mplconfig"),
)

import numpy as np
import pandas as pd

from model1_core import PARAMS, load_data, recursion, validate_subject_alignment

OUT = Path(__file__).resolve().parent / "outputs"

from ppc_timecourse import optimal_by_bin, timecourse_rows

MCMC_DIR = OUT / "mcmc"
PPC_DIR = OUT / "ppc"
PPC_METRIC_LABELS = {
    "action_2_rate": "Action 2 choice rate",
    "mean_reward": "Mean reward",
    "stay_rate": "Stay rate",
    "optimal_rate": "Optimal rate",
}



def final_npz_path(load, sid, source):
    path = MCMC_DIR / f"{load}_subject_{sid:02d}_{source}.npz"
    if path.exists():
        return path
    if source == "long":
        return MCMC_DIR / f"{load}_subject_{sid:02d}_retry.npz"
    return path


def optimal_action(data):
    context = data[:, 0].astype(int)
    best_outcome = np.argmax(data[:, 7:9], axis=1)
    ties = data[:, 7] == data[:, 8]
    # In model_1, outcome = action XOR context. The optimal action is therefore
    # the better local outcome flipped by context. For low load context is 0,
    # so this reduces to the old action == best_outcome rule.
    best_action = best_outcome ^ context
    return best_action, ties


def metric_dict(data):
    action = data[:, 4].astype(int)
    reward = data[:, 6]
    best, ties = optimal_action(data)
    eligible = ~ties
    state = data[:, 0].astype(int) * 2 + data[:, 1].astype(int)
    same_adjacent_state = (state[1:] == state[:-1]) & (np.diff(data[:, 9]) == 1)
    stay = action[1:][same_adjacent_state] == action[:-1][same_adjacent_state]
    return {
        "action_2_rate": float(np.mean(action == 1)),
        "mean_reward": float(np.mean(reward)),
        "stay_rate": float(np.mean(stay)) if len(stay) else np.nan,
        "optimal_rate": float(np.mean(action[eligible] == best[eligible])) if np.any(eligible) else np.nan,
    }


def summarize_distribution(values, observed):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(observed) or not len(values):
        return dict(observed=float(observed), pred_mean=np.nan, **{
            "pred_q2.5": np.nan, "pred_q97.5": np.nan,
            "p_ge_observed": np.nan, "bayes_p_two_sided": np.nan})
    return {
        "observed": float(observed),
        "pred_mean": float(np.mean(values)),
        "pred_q2.5": float(np.quantile(values, 0.025)),
        "pred_q97.5": float(np.quantile(values, 0.975)),
        "p_ge_observed": float(np.mean(values >= observed)),
        "bayes_p_two_sided": float(min(1.0, 2 * min(np.mean(values >= observed), np.mean(values <= observed)))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("PPC reps must be positive.")
    PPC_DIR.mkdir(parents=True, exist_ok=True)

    diag = pd.read_csv(MCMC_DIR / "mcmc_subject_diagnostics.csv")
    rng = np.random.default_rng(args.seed)
    subject_rows = []
    group_rows = []
    group_replicate_rows = []
    time_rows = []
    replicate_store = {}

    for load in ("low", "high"):
        datasets = load_data(load)
        validate_subject_alignment(load, datasets, diag.loc[diag.load == load, "id"])
        group_rep = {}
        group_obs = {}
        time_obs = []
        time_rep = []
        for sid, data, _ in datasets:
            source = diag[(diag.load == load) & (diag.id == sid)]["source"].iloc[0]
            arr = np.load(final_npz_path(load, sid, source))
            samples = arr["samples"].reshape(-1, len(PARAMS))
            best, ties = optimal_action(data)
            eligible = ~ties
            time_obs.append(optimal_by_bin(data[:, 4], best, eligible))
            sim_bins = []
            obs_metrics = metric_dict(data)
            for key, val in obs_metrics.items():
                group_obs.setdefault(key, []).append(val)

            chosen = rng.integers(0, len(samples), size=args.reps)
            sim_metrics = {key: [] for key in obs_metrics}
            for rep, idx in enumerate(chosen):
                uniforms = rng.random(len(data))
                _, sim_data, _ = recursion(samples[idx], data, True, uniforms)
                sim_bins.append(optimal_by_bin(sim_data[:, 4], best, eligible))
                metrics = metric_dict(sim_data)
                for key, val in metrics.items():
                    sim_metrics[key].append(val)

            time_rep.append(sim_bins)
            for key, vals in sim_metrics.items():
                subject_rows.append(
                    {
                        "load": load,
                        "id": sid,
                        "metric": key,
                        **summarize_distribution(vals, obs_metrics[key]),
                    }
                )
                group_rep.setdefault(key, []).append(vals)

            replicate_store[f"{load}_{sid:02d}"] = np.array(
                [[sim_metrics[k][r] for k in obs_metrics] for r in range(args.reps)]
            )

        time_rows.extend(timecourse_rows(load, time_obs, time_rep, summarize_distribution))
        for key in PPC_METRIC_LABELS:
            reps = np.asarray(group_rep[key], dtype=float).T
            rep_means = np.nanmean(reps, axis=1)
            for rep_idx, value in enumerate(rep_means):
                group_replicate_rows.append(
                    {
                        "load": load,
                        "metric": key,
                        "rep": rep_idx,
                        "value": float(value),
                    }
                )
            group_rows.append(
                {
                    "load": load,
                    "metric": key,
                    **summarize_distribution(rep_means, np.nanmean(group_obs[key])),
                }
            )

    pd.DataFrame(time_rows).to_csv(PPC_DIR / "ppc_timecourse.csv", index=False)
    pd.DataFrame(subject_rows).to_csv(PPC_DIR / "ppc_subject_metrics.csv", index=False)
    pd.DataFrame(group_rows).to_csv(PPC_DIR / "ppc_group_metrics.csv", index=False)
    pd.DataFrame(group_replicate_rows).to_csv(PPC_DIR / "ppc_group_replicates.csv", index=False)
    np.savez_compressed(PPC_DIR / "ppc_subject_replicates.npz",
                        metric_names=np.array(list(PPC_METRIC_LABELS)), **replicate_store)
    (PPC_DIR / "ppc_run_metadata.json").write_text(json.dumps({
        "replicates": args.reps, "seed": args.seed, "loads": ["low", "high"],
        "metrics": list(PPC_METRIC_LABELS),
        "group_aggregation": "Equal-weight mean of subject-level metrics",
        "stay_definition": "Repeated action on consecutive original trials in the same state and load",
        "optimal_definition": "Choice with higher available reward; reward ties excluded",
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame(group_rows).to_string(index=False))
    print("Wrote", PPC_DIR)


if __name__ == "__main__":
    main()
