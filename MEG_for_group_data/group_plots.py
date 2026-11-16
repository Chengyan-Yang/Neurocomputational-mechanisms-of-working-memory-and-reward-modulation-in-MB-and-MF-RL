"""Shared plotting helpers for the released MEG figure data."""
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from meg_pdf import save_brain_png, save_pdf

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "group_data" / "MEG"
RESULTS = HERE / "results"
CONDITIONS = ("low_small", "low_large", "high_small", "high_large")
LABELS = ("Low load, small reward", "Low load, large reward",
          "High load, small reward", "High load, large reward")


def table(section, name):
    return pd.read_csv(DATA / section / (name + ".csv"))


def pdf(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_pdf(fig, path.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    plt.close(fig)


def brains(z, values, path, kind, limits, hemis):
    import pyvista
    pyvista.OFF_SCREEN = True
    stc = mne.SourceEstimate(values[:,None], [z["vertices_lh"], z["vertices_rh"]], 0, 1, subject="fsaverage")
    options = dict(surface="inflated", cortex="low_contrast", size=(800,800))
    clim = dict(kind="value",pos_lims=list(limits))
    if kind=="rsa":
        colors = LinearSegmentedColormap.from_list("rsa",["#D73027","#FFD700"])(np.linspace(0,1,256))
        colors[0,3]=0
        options.update(colormap=ListedColormap(colors),transparent=False)
        clim = dict(kind="value",lims=list(limits))
    elif kind=="mask":
        options.update(surface="pial",cortex=(1.,1.,1.),size=(1050,850),
                       colormap=LinearSegmentedColormap.from_list("cluster_mask_yellow", ["#F2C500","#F2C500"], N=2),transparent=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    for hemi in hemis:
        brain = stc.plot(subjects_dir=str(DATA/"shared"), hemi=hemi, views="lateral",
                         time_viewer=False,time_label=None,colorbar=False,title="",background="white",
                         clim=clim, backend="pyvistaqt", brain_kwargs=dict(show=False), **options)
        try:
            for view in ("lateral","medial"):
                brain.show_view(view)
                save_brain_png(brain,path.parent/f"{path.name}_{hemi}_{view}.png",
                               base_alpha=.6 if kind=="rsa" else 1.)
        finally:
            brain.close()
