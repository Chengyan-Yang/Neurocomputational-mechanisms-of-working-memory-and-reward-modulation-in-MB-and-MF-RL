"""Run longer chains for selected MCMC cases and merge them into final tables."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from model1_core import PARAMS, load_data, validate_subject_alignment
from run_mcmc import MCMC_DIR, load_map, one_subject


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", choices=["low", "high"], required=True)
    parser.add_argument("--ids", type=int, nargs="+", required=True)
    parser.add_argument("--warmup", type=int, default=25000)
    parser.add_argument("--draws", type=int, default=50000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    maps = load_map(args.load)
    datasets = load_data(args.load)
    validate_subject_alignment(args.load, datasets, maps.index)
    data = {sid: z for sid, z, _ in datasets}
    run_args = SimpleNamespace(
        warmup=args.warmup,
        draws=args.draws,
        chains=args.chains,
        seed=args.seed,
        force=True,
    )

    rows = []
    for sid in args.ids:
        rows.extend(
            one_subject(
                (
                    args.load,
                    sid,
                    data[sid],
                    maps.loc[sid, PARAMS].to_numpy(dtype=float),
                    run_args,
                    True,
                )
            )
        )
    long_df = pd.DataFrame(rows)
    long_df["source"] = "long"
    long_df.to_csv(MCMC_DIR / f"mcmc_summary_long_{args.load}.csv", index=False)

    final = pd.read_csv(MCMC_DIR / "mcmc_summary_final.csv")
    keys = set((args.load, sid) for sid in args.ids)
    final = pd.concat(
        [
            final[~final[["load", "id"]].apply(tuple, axis=1).isin(keys)],
            long_df,
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
    print(by_sub[by_sub.load == args.load].sort_values("max_rhat", ascending=False).head(8).to_string(index=False))
    print(by_sub.groupby("load")["passed"].agg(["sum", "count"]))


if __name__ == "__main__":
    main()
