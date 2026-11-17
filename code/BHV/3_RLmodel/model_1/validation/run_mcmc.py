"""Run posterior MCMC checks for model_1 in low- and high-load data."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / "_mplconfig"),
)

import arviz as az
import numpy as np
import pandas as pd

from model1_core import (
    PARAMS,
    load_data,
    prior_draw,
    sample_chain,
    to_z,
    validate_subject_alignment,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "outputs"
MCMC_DIR = OUT / "mcmc"


def load_map(load):
    path = ROOT / f"{load}_load" / f"results_M1_{load}.csv"
    return pd.read_csv(path).set_index("id")


def make_initials(map_x, chains, rng):
    initials = []
    base = to_z(map_x)
    initials.append(base)
    for _ in range(1, min(chains, 3)):
        jitter = rng.normal(0, [0.25, 0.75, 0.75, 0.75, 0.75, 0.35, 0.35])
        initials.append(base + jitter)
    while len(initials) < chains:
        initials.append(to_z(prior_draw(rng)))
    return initials


def diagnostics(samples):
    idata = az.from_dict(
        posterior={name: samples[:, :, idx] for idx, name in enumerate(PARAMS)}
    )
    diag = az.summary(
        idata,
        var_names=PARAMS,
        kind="diagnostics",
        round_to="none",
    )
    stat = az.summary(
        idata,
        var_names=PARAMS,
        kind="stats",
        hdi_prob=0.95,
        round_to="none",
    )
    rows = []
    for p in PARAMS:
        vals = samples[:, :, PARAMS.index(p)].reshape(-1)
        rows.append(
            {
                "param": p,
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "sd": float(np.std(vals, ddof=1)),
                "q2.5": float(np.quantile(vals, 0.025)),
                "q97.5": float(np.quantile(vals, 0.975)),
                "mcse_mean": float(diag.loc[p, "mcse_mean"]),
                "mcse_sd": float(diag.loc[p, "mcse_sd"]),
                "ess_bulk": float(diag.loc[p, "ess_bulk"]),
                "ess_tail": float(diag.loc[p, "ess_tail"]),
                "rhat": float(diag.loc[p, "r_hat"]),
                "hdi_2.5": float(stat.loc[p, "hdi_2.5%"]),
                "hdi_97.5": float(stat.loc[p, "hdi_97.5%"]),
            }
        )
    return rows


def one_subject(task):
    load, sid, data, map_x, args, retry = task
    rng = np.random.default_rng(args.seed + sid * 17 + (10000 if load == "high" else 0))
    warmup = args.warmup * (2 if retry else 1)
    draws = args.draws * (2 if retry else 1)
    suffix = "retry" if retry else "initial"
    MCMC_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = MCMC_DIR / f"{load}_subject_{sid:02d}_{suffix}.npz"
    meta_path = MCMC_DIR / f"{load}_subject_{sid:02d}_{suffix}.json"

    if npz_path.exists() and not args.force:
        arr = np.load(npz_path)
        samples = arr["samples"]
        lps = arr["logp"]
        accepts = arr["acceptance"]
    else:
        initials = make_initials(map_x, args.chains, rng)
        samples = np.empty((args.chains, draws, len(PARAMS)))
        lps = np.empty((args.chains, draws))
        accepts = np.empty(args.chains)
        for c, init in enumerate(initials):
            s, lp, acc = sample_chain(
                data,
                init,
                int(args.seed + sid * 100 + c + (1000 if load == "high" else 0)),
                int(warmup),
                int(draws),
            )
            samples[c] = s
            lps[c] = lp
            accepts[c] = acc
        np.savez_compressed(
            npz_path,
            samples=samples,
            logp=lps,
            acceptance=accepts,
            params=np.array(PARAMS),
        )
        meta_path.write_text(
            json.dumps(
                {
                    "load": load,
                    "subject": sid,
                    "warmup": warmup,
                    "draws": draws,
                    "chains": args.chains,
                    "seed": args.seed,
                    "source": suffix,
                    "acceptance": accepts.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    rows = diagnostics(samples)
    out_rows = []
    for row in rows:
        row = dict(row)
        row.update(
            {
                "load": load,
                "id": sid,
                "trials": int(data.shape[0]),
                "warmup": warmup,
                "draws": draws,
                "chains": args.chains,
                "source": suffix,
                "acceptance_mean": float(np.mean(accepts)),
                "logp_mean": float(np.mean(lps)),
            }
        )
        out_rows.append(row)
    return out_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=6000)
    parser.add_argument("--draws", type=int, default=12000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry", action="store_true", default=True)
    args = parser.parse_args()

    tasks = []
    for load in ("low", "high"):
        maps = load_map(load)
        datasets = load_data(load)
        validate_subject_alignment(load, datasets, maps.index)
        for sid, data, _ in datasets:
            map_x = maps.loc[sid, PARAMS].to_numpy(dtype=float)
            tasks.append((load, sid, data, map_x, args, False))

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one_subject, task) for task in tasks]
        for fut in as_completed(futures):
            rows.extend(fut.result())

    initial = pd.DataFrame(rows)
    initial_path = MCMC_DIR / "mcmc_summary_initial.csv"
    initial.to_csv(initial_path, index=False)

    final = initial.copy()
    if args.retry:
        bad = (
            initial.groupby(["load", "id"])
            .agg(max_rhat=("rhat", "max"), min_bulk=("ess_bulk", "min"), min_tail=("ess_tail", "min"))
            .reset_index()
        )
        bad = bad[(bad.max_rhat > 1.01) | (bad.min_bulk < 400) | (bad.min_tail < 400)]
        retry_tasks = [
            task[:-1] + (True,)
            for task in tasks
            if ((task[0] == bad.load) & (task[1] == bad.id)).any()
        ]
        retry_rows = []
        if retry_tasks:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(one_subject, task) for task in retry_tasks]
                for fut in as_completed(futures):
                    retry_rows.extend(fut.result())
            retry = pd.DataFrame(retry_rows)
            retry.to_csv(MCMC_DIR / "mcmc_summary_retry.csv", index=False)
            keys = set(zip(retry.load, retry.id))
            final = pd.concat(
                [
                    initial[~initial[["load", "id"]].apply(tuple, axis=1).isin(keys)],
                    retry,
                ],
                ignore_index=True,
            )

    final.to_csv(MCMC_DIR / "mcmc_summary_final.csv", index=False)
    by_sub = (
        final.groupby(["load", "id"])
        .agg(
            max_rhat=("rhat", "max"),
            min_bulk=("ess_bulk", "min"),
            min_tail=("ess_tail", "min"),
            mean_acceptance=("acceptance_mean", "mean"),
            source=("source", "first"),
        )
        .reset_index()
    )
    by_sub["passed"] = (
        (by_sub.max_rhat <= 1.01)
        & (by_sub.min_bulk >= 400)
        & (by_sub.min_tail >= 400)
    )
    by_sub.to_csv(MCMC_DIR / "mcmc_subject_diagnostics.csv", index=False)
    print(by_sub.groupby("load")["passed"].agg(["sum", "count"]))
    print("Worst Rhat", by_sub.sort_values("max_rhat", ascending=False).head(8).to_string(index=False))
    print("Wrote", MCMC_DIR)


if __name__ == "__main__":
    main()
