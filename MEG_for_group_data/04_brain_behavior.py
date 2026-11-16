"""Figure 5 forest plots, added-variable plots and cluster masks."""
import numpy as np
from group_plots import DATA, RESULTS, brains, table
from sensor_mask import sensor_mask

def main():
    section = "brain_behavior"
    out = RESULTS/"Fig_5"
    def read(name):
        return table(section,name)
    for key,frame in read("Fig_5bc_sensor_masks").groupby("map_key"):
        sensor_mask(frame,out/(key+"_mask"))
    for path in (DATA/section).glob("*_mask.npz"):
        with np.load(path) as z:
            values = np.zeros(len(z["vertices_lh"])+len(z["vertices_rh"]))
            values[z["cluster_space_index"]] = 1
            brains(z,values,out/path.stem,"mask",[.5,.75,1],[hemi for hemi,data in zip(("lh","rh"),np.split(values,[len(z["vertices_lh"])])) if np.any(data)])


if __name__ == "__main__":
    main()
