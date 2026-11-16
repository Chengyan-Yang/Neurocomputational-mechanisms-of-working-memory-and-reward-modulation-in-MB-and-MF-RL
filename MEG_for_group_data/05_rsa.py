"""Figure 6 cluster locations, surface maps and representative RDMs."""
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from group_plots import DATA, RESULTS, brains, pdf, table
from rsa_style import RDM_PAIRS, plot_individual_rdm, plot_cluster_location_bubbles, save_standalone_colorbar

PHASES = ["incentive","decision","transition","feedback"]
KEYS = [f"{mag}_magnitude_{load}_load" for load in ("low","high") for mag in ("small","large")]


def main():
    out = RESULTS/"Fig_6"
    data = table("rsa","Fig_6_cluster_plotting_table")
    for model, rows in data.groupby("model"):
        out.mkdir(parents=True,exist_ok=True)
        rows = rows.copy()
        rows["phase"] = pd.Categorical(rows.phase,categories=PHASES,ordered=True)
        rows["peak_label_clean"] = rows.peak_aparc_label
        plot_cluster_location_bubbles(rows,{'mb':'Model-based','mf':'Model-free'}[model],out/(model+"_cluster_locations"))
    distances = table("rsa","Fig_6_RDM_distances").groupby(["model","phase","cluster_index"])
    for row in table("rsa","Fig_6_RDM_statistics").itertuples():
        frame = distances.get_group((row.model,row.phase,row.cluster_index)).set_index("rdm_pair")
        ordered = frame.loc[list(RDM_PAIRS)]
        plot_individual_rdm(row._asdict(),ordered.meg_distance.to_numpy(),
                            ordered.behavior_distance.to_numpy(),row.n_subjects,
                            {"mb":"Model-based","mf":"Model-free"}[row.model],out/row.model)
    for path in sorted((DATA/"rsa").glob("*_group_map.npz")):
        print(path.name,flush=True)
        with np.load(path) as z:
            brains(z,np.maximum(z["masked_full_mean_rho"],0),out/path.stem,"rsa",z["color_limits"],["lh","rh"])
            limits = z["color_limits"].tolist()
    save_standalone_colorbar(limits,out)


if __name__ == "__main__":
    main()
