"""Validate the Python model_1 recursion against the original MATLAB MAP CSVs."""
from pathlib import Path

import numpy as np
import pandas as pd

from model1_core import PARAMS, ROOT, load_data, recursion, validate_subject_alignment


def main():
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for load in ("low", "high"):
        datasets = load_data(load)
        data = {sid: z for sid, z, _ in datasets}
        fit = pd.read_csv(ROOT / f"{load}_load" / f"results_M1_{load}.csv")
        validate_subject_alignment(load, datasets, fit.id)
        for _, row in fit.iterrows():
            sid = int(row.id)
            x = row[PARAMS].to_numpy(dtype=float)
            ll, _, _ = recursion(x, data[sid])
            rows.append(
                {
                    "load": load,
                    "id": sid,
                    "python_loglik": float(ll),
                    "matlab_csv_loglik": float(row.loglik),
                    "difference": float(ll - row.loglik),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "likelihood_validation.csv", index=False)
    print(df.groupby("load")["difference"].agg(["count", "min", "max", "mean"]))
    print("max_abs_diff", np.max(np.abs(df.difference)))


if __name__ == "__main__":
    main()
