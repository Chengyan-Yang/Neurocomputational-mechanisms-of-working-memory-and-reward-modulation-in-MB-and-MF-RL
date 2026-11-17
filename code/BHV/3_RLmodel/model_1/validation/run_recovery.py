"""Parameter recovery for model_1 by simulation and refitting."""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / "_mplconfig"),
)

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import linregress, pearsonr, spearmanr

from model1_core import (
    PARAMS,
    load_data,
    objective,
    prior_draw,
    recursion,
    to_z,
    validate_subject_alignment,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "outputs"
REC_DIR = OUT / "recovery"


def load_map(load):
    path = ROOT / f"{load}_load" / f"results_M1_{load}.csv"
    return pd.read_csv(path).set_index("id")


def broad_draw(rng):
    return np.array(
        [
            np.exp(rng.uniform(np.log(0.12), np.log(8.0))),
            rng.uniform(0.04, 0.96),
            rng.uniform(0.04, 0.96),
            rng.uniform(0.04, 0.96),
            rng.uniform(0.04, 0.96),
            rng.normal(0, 1.0),
            rng.normal(0, 1.0),
        ],
        dtype=float,
    )


def fit_map(data, starts, rng):
    bounds = [
        (np.log(1e-4), np.log(19.999)),
        (-11.5129, 11.5129),
        (-11.5129, 11.5129),
        (-11.5129, 11.5129),
        (-11.5129, 11.5129),
        (-19.99, 19.99),
        (-19.99, 19.99),
    ]
    best = None
    for start in starts:
        res = minimize(
            objective,
            to_z(start),
            args=(data,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 800, "ftol": 1e-8},
        )
        if best is None or res.fun < best.fun:
            best = res
    x = np.empty(len(PARAMS))
    z = best.x
    x[0] = np.exp(z[0])
    x[1:5] = 1 / (1 + np.exp(-z[1:5]))
    x[5:] = z[5:]
    return x, -float(best.fun), bool(best.success)


def recovery_task(task):
    load, sid, template_data, truth, source, rep, args = task
    rng = np.random.default_rng(args.seed + sid * 311 + rep * 37 + (9000 if load == "high" else 0))
    _, sim_data, _ = recursion(truth, template_data, True, rng.random(len(template_data)))
    starts = [np.array([1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0]), broad_draw(rng)]
    starts.extend(prior_draw(rng) for _ in range(max(0, args.starts - len(starts))))
    est, logpost, ok = fit_map(sim_data, starts, rng)
    row = {
        "load": load,
        "template_id": sid,
        "truth_source": source,
        "rep": rep,
        "fit_success": ok,
        "logpost": logpost,
    }
    for p in ("w_small", "w_large"):
        i = PARAMS.index(p)
        row[f"true_{p}"] = float(truth[i])
        row[f"recovered_{p}"] = float(est[i])
    return row


def summarize(df):
    rows = []
    params = ("w_small", "w_large")
    for (load, source), sub in df.groupby(["load", "truth_source"]):
        for p in params:
            x = sub[f"true_{p}"].to_numpy()
            y = sub[f"recovered_{p}"].to_numpy()
            if np.std(x) == 0 or np.std(y) == 0:
                pear = np.nan
                spear = np.nan
                pear_p = np.nan
                spear_p = np.nan
                slope = np.nan
                intercept = np.nan
            else:
                pear_res = pearsonr(x, y)
                spear_res = spearmanr(x, y)
                pear = pear_res.statistic
                pear_p = pear_res.pvalue
                spear = spear_res.statistic
                spear_p = spear_res.pvalue
                lr = linregress(x, y)
                slope, intercept = lr.slope, lr.intercept
            rows.append(
                {
                    "load": load,
                    "truth_source": source,
                    "param": p,
                    "n": len(sub),
                    "pearson_r": pear,
                    "pearson_p": pear_p,
                    "spearman_r": spear,
                    "spearman_p": spear_p,
                    "bias": float(np.mean(y - x)),
                    "rmse": float(np.sqrt(np.mean((y - x) ** 2))),
                    "slope": slope,
                    "intercept": intercept,
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--empirical-reps", type=int, default=3)
    parser.add_argument("--broad", type=int, default=80)
    parser.add_argument("--starts", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    REC_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for load in ("low", "high"):
        maps = load_map(load)
        datasets = load_data(load)
        validate_subject_alignment(load, datasets, maps.index)
        by_id = {sid: data for sid, data, _ in datasets}
        for sid, data, _ in datasets:
            truth = maps.loc[sid, PARAMS].to_numpy(dtype=float)
            for rep in range(args.empirical_reps):
                tasks.append((load, sid, data, truth, "empirical_map", rep, args))
        rng = np.random.default_rng(args.seed + (40000 if load == "high" else 30000))
        ids = list(by_id)
        for rep in range(args.broad):
            sid = ids[rep % len(ids)]
            tasks.append((load, sid, by_id[sid], broad_draw(rng), "broad", rep, args))

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(recovery_task, task) for task in tasks]
        for fut in as_completed(futures):
            rows.append(fut.result())

    df = pd.DataFrame(rows)
    df.to_csv(REC_DIR / "recovery_estimates.csv", index=False)
    summary = summarize(df)
    summary.to_csv(REC_DIR / "recovery_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("Wrote", REC_DIR)


if __name__ == "__main__":
    main()
