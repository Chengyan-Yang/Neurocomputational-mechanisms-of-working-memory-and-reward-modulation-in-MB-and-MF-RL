"""Build MB and MF behavioral RDMs for the four-phase searchlight analysis."""
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from meg_project import MEG_CODE, behavior_rdm, meg_subjects


def build_behavior_rdm(model):
    source = MEG_CODE.parent / "BHV" / "3_HBRLmodel" / f"{model}_estimates_hdi.csv"
    estimates = pd.read_csv(source).set_index("subject_id", verify_integrity=True)
    estimates = estimates.loc[[int(subject) for subject in meg_subjects()]]
    conditions = {}
    for load, suffix in (("low", "lload"), ("high", "hload")):
        for magnitude in ("small", "large"):
            column = f"beta_{model}_{magnitude}_{suffix}_median"
            values = estimates[column]
            if not np.isfinite(values).all():
                raise ValueError(f"Missing or invalid posterior median in {source}: {column}")
            conditions[f"{magnitude}_magnitude_{load}_load"] = values
    pairs = [
        pd.DataFrame({"subid": estimates.index, "rdm_pair": f"{a}__vs__{b}",
                      "rdm_distance": (conditions[a] - conditions[b]).abs().to_numpy()})
        for a, b in combinations(conditions, 2)
    ]
    path = behavior_rdm(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(pairs, ignore_index=True).to_csv(path, index=False)


if __name__ == "__main__":
    for model in ("mb", "mf"):
        build_behavior_rdm(model)
        print(f"Saved {model.upper()} behavior RDM: {behavior_rdm(model)}")
