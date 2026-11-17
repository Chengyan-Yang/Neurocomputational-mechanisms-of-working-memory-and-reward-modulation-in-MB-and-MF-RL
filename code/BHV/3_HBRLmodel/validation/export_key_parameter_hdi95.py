"""Export 95% posterior HDIs, preserving other key-parameter diagnostics.

Uses the shortest sorted-sample interval (the unimodal ArviZ HDI algorithm).
Reproduces the source 94% HDIs before writing the 95% version.
HDI95_low/high are highest-density endpoints, not equal-tail quantiles.
"""
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd


def hdi(values, probability):
    ordered = np.sort(np.asarray(values, dtype=float))
    offset = int(np.floor(probability * len(ordered)))
    widths = ordered[offset:] - ordered[:len(ordered) - offset]
    start = int(np.argmin(widths))
    return ordered[[start, start + offset]]


def main():
    folder = Path(os.environ.get("HBRL52_OUTPUT_DIR", Path(__file__).resolve().parent / "outputs")) / "mcmc"
    source = folder / "hbrl_mcmc_key_parameter_diagnostics.csv"
    target = folder / "hbrl_mcmc_key_parameter_diagnostics_95HDI_labeled.csv"
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)
    parameters = [row["parameter"] for row in rows]
    assert len(parameters) == len(set(parameters))
    draws = pd.read_csv(folder / "hbrl_all_stan_draws.csv", usecols=["chain", "draw"] + parameters)
    metadata = pd.read_csv(folder / "stan_all_draw_export_metadata.csv").iloc[0]
    assert int(metadata["subjects"]) == 52
    assert len(draws) == int(metadata["total_retained_draws"])
    assert draws["chain"].nunique() == int(metadata["chains"])
    assert (draws.groupby("chain").size() == int(metadata["retained_draws_per_chain"])).all()
    assert not draws.duplicated(["chain", "draw"]).any()
    output = []
    max_old_error = 0.0
    for row in rows:
        values = draws[row["parameter"]].to_numpy()
        assert np.isfinite(values).all()
        old = np.array([float(row["hdi_3%"]), float(row["hdi_97%"])])
        old_recomputed = hdi(values, 0.94)
        max_old_error = max(max_old_error, float(np.max(np.abs(old - old_recomputed))))
        np.testing.assert_allclose(old_recomputed, old, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(values.mean(), float(row["mean"]), rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(values.std(ddof=1), float(row["sd"]), rtol=1e-7, atol=1e-8)
        lower, upper = hdi(values, 0.95)
        assert lower <= upper
        assert np.mean((values >= lower) & (values <= upper)) >= 0.95
        assert upper - lower >= old_recomputed[1] - old_recomputed[0]
        updated = dict(row)
        del updated["hdi_3%"], updated["hdi_97%"]
        updated["HDI95_low"] = format(lower, ".15g")
        updated["HDI95_high"] = format(upper, ".15g")
        output.append(updated)
    columns = [{"hdi_3%": "HDI95_low", "hdi_97%": "HDI95_high"}.get(c, c) for c in columns]
    with target.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output)
    with target.open(newline="", encoding="utf-8-sig") as handle:
        saved = list(csv.DictReader(handle))
    assert saved == output
    print(f"Saved: {target}")
    print(f"Verified {len(rows)} parameters; {len(draws)} retained posterior draws each.")
    print(f"Maximum difference reproducing source 94% HDI endpoints: {max_old_error:.3g}")
    print(pd.DataFrame(output)[["parameter", "HDI95_low", "HDI95_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
