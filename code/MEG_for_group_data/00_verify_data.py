"""Verify the public MEG package and tabulate analysis-side values for S6-S9."""
import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "group_data" / "MEG"


def main():
    with (DATA/"manifest.csv").open(encoding="utf-8-sig",newline="") as stream:
        files = list(csv.DictReader(stream))
    for row in files:
        path = DATA/row["file"]
        if path.stat().st_size != int(row["bytes"]) or hashlib.sha256(path.read_bytes()).hexdigest()!=row["sha256"]:
            raise ValueError(f"File size or SHA-256 mismatch: {path}")
    out = HERE/"results"/"tables"
    out.mkdir(parents=True,exist_ok=True)
    for label,section,multiplier,corrected in (
        ("S6","sensor_erf",3,"p_phase_x3"),("S7","source_erf",3,"p_phase_x3"),
        ("S8","time_frequency",12,"p_phase_x12")):
        data = pd.read_csv(DATA/section/"cluster_statistics.csv")
        np.testing.assert_allclose(data[corrected],np.minimum(1,data.pFWE_exact*multiplier))
        for column in ("effect", "effect_key", "Effect", "Comparison"):
            if column in data:
                data[column] = data[column].str.replace("magnitude","reward",regex=False).str.replace("Magnitude","Reward",regex=False)
        data.to_csv(out/f"Table_{label}_analysis_values.csv",index=False)
    models = pd.read_csv(DATA/"brain_behavior"/"Fig_5a_MB_coefficients.csv")
    models["effect"] = models["effect"].replace({"magnitude":"reward"})
    models.loc[models.p_fdr_lm_within_phase<.05].to_csv(out/"Table_S9_analysis_values.csv",index=False)
    print(f"Verified {len(files)} input files. Tables saved to {out}")


if __name__ == "__main__":
    main()
