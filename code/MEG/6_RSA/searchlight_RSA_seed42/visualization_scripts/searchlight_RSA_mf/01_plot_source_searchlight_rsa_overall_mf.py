"""
Visualize source searchlight RSA results.

Inputs are produced by:
  01_run_source_searchlight_rsa_overall_mf.py

This script creates:
  1) cluster-thresholded source brain maps for each phase
  2) phase-level summary plots
  3) report-ready significant cluster tables with peak MNI and aparc labels

Run in the same MNE-Python/Spyder environment used for the source analysis:
  python 01_plot_source_searchlight_rsa_overall_mf.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from meg_project import DATA_ROOT, project_input, RSA_RESULTS, available
from meg_pdf import save_pdf, save_brain_png

import importlib
import json
import os
import pickle
import re

if sys.getrecursionlimit() < 50000:
    sys.setrecursionlimit(50000)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import pandas as pd

try:
    import mne
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "This plotting script requires MNE-Python. Please run it in the same "
        "Python/Spyder environment used for the ERF source analysis and source "
        "searchlight RSA."
    ) from error


# =============================================================================
# Configuration
# =============================================================================

SEARCHLIGHT_DIR = project_input(
    os.environ.get(
        "SEARCHLIGHT_RSA_MF_RESULT_DIR",
        str(RSA_RESULTS / "searchlight_seed42" / "overall_mf_source_searchlight"),
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "SEARCHLIGHT_RSA_MF_FIGURE_DIR",
        str(SEARCHLIGHT_DIR / "Figures_publication"),
    )
)

SUBJECTS_DIR = project_input(
    os.environ.get(
        "SEARCHLIGHT_SUBJECTS_DIR",
        f"{DATA_ROOT}/MRI/surf",
    )
)
TEMPLATE_SUBJECT = "fsaverage"

PHASE_ORDER = ("incentive", "decision", "transition", "feedback")
PHASE_LABELS = {
    "incentive": "Incentive",
    "decision": "Decision",
    "transition": "Transition",
    "feedback": "Feedback",
}

VIEWS = ("lateral", "medial")
HEMIS = ("lh", "rh")

# Publication style shared across phases and both models.
SURFACE = "inflated"
CORTEX = "low_contrast"
BACKGROUND = "white"
BRAIN_SIZE = (800, 800)
BRAIN_BASE_ALPHA = 0.6
BRAIN_COLORMAP = LinearSegmentedColormap.from_list("rsa_red_yellow", ["#D73027", "#FFD700"], N=256)
_brain_rgba = BRAIN_COLORMAP(np.linspace(0, 1, 256))
_brain_rgba[0, 3] = 0.0
BRAIN_RENDER_COLORMAP = ListedColormap(_brain_rgba)

CLUSTER_ALPHA = float(os.environ.get("SEARCHLIGHT_RSA_CLUSTER_ALPHA", "0.05"))
PLOT_VALUE = os.environ.get("SEARCHLIGHT_RSA_PLOT_VALUE", "mean_rho").lower()
if PLOT_VALUE not in {"mean_rho", "t_value"}:
    raise ValueError("SEARCHLIGHT_RSA_PLOT_VALUE must be 'mean_rho' or 't_value'.")


# =============================================================================
# Helpers
# =============================================================================


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()


def install_numpy_pickle_compat() -> None:
    try:
        numpy_core = importlib.import_module("numpy.core")
    except Exception:
        return
    sys.modules.setdefault("numpy._core", numpy_core)
    for name in ("multiarray", "numeric", "umath"):
        try:
            module = importlib.import_module(f"numpy.core.{name}")
        except Exception:
            continue
        sys.modules.setdefault(f"numpy._core.{name}", module)


def load_pickle(path: Path):
    try:
        with path.open("rb") as file:
            return pickle.load(file)
    except ModuleNotFoundError as error:
        if not str(error).startswith("No module named 'numpy._core"):
            raise
        install_numpy_pickle_compat()
        with path.open("rb") as file:
            return pickle.load(file)


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def phase_dir(phase: str) -> Path:
    return SEARCHLIGHT_DIR / phase


def get_cluster_space_indices(cluster) -> np.ndarray:
    if isinstance(cluster, tuple):
        return np.asarray(cluster[0], dtype=int)
    return np.asarray(cluster, dtype=int)


def significant_cluster_mask(phase: str, n_centers: int) -> tuple[np.ndarray, pd.DataFrame]:
    cluster_path = phase_dir(phase) / "cluster_permutation_results.pkl"
    if not cluster_path.exists():
        raise FileNotFoundError(f"Missing cluster permutation file: {cluster_path}")
    cluster_results = load_pickle(cluster_path)
    cluster_p_values = np.asarray(cluster_results["cluster_p_values"], dtype=float)
    clusters = cluster_results["clusters"]

    mask = np.zeros(n_centers, dtype=bool)
    rows = []
    for cluster_index, p_value in enumerate(cluster_p_values):
        space_indices = get_cluster_space_indices(clusters[cluster_index])
        is_sig = bool(p_value < CLUSTER_ALPHA)
        if is_sig:
            mask[space_indices] = True
        rows.append(
            {
                "phase": phase,
                "cluster_index": int(cluster_index),
                "cluster_p": float(p_value),
                "is_significant": is_sig,
                "n_vertices": int(space_indices.size),
            }
        )
    return mask, pd.DataFrame(rows, columns=["phase", "cluster_index", "cluster_p", "is_significant", "n_vertices"])


def make_full_source_data(group_maps: dict, values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    full_vertices_lh = np.asarray(group_maps["full_vertices_lh"], dtype=int)
    full_vertices_rh = np.asarray(group_maps["full_vertices_rh"], dtype=int)
    valid_indices = np.asarray(group_maps["valid_indices"], dtype=int)
    center_indices = np.asarray(group_maps["center_indices"], dtype=int)

    full_data = np.zeros(len(full_vertices_lh) + len(full_vertices_rh), dtype=float)
    plot_values = np.asarray(values, dtype=float).copy()
    if mask is not None:
        plot_values[~mask] = 0.0
    full_indices = valid_indices[center_indices]
    full_data[full_indices] = plot_values
    return full_data[:, np.newaxis]


def make_stc(group_maps: dict, data: np.ndarray) -> mne.SourceEstimate:
    return mne.SourceEstimate(
        data,
        vertices=[
            np.asarray(group_maps["full_vertices_lh"], dtype=int),
            np.asarray(group_maps["full_vertices_rh"], dtype=int),
        ],
        tmin=0.0,
        tstep=1.0,
        subject=TEMPLATE_SUBJECT,
    )


def collect_plot_values() -> np.ndarray:
    values = []
    for phase in PHASE_ORDER:
        maps_path = phase_dir(phase) / "source_searchlight_group_maps.npz"
        if not maps_path.exists():
            continue
        group_maps = load_npz(maps_path)
        mask, _ = significant_cluster_mask(phase, len(group_maps["center_indices"]))
        phase_values = np.asarray(group_maps[PLOT_VALUE], dtype=float)
        phase_values = phase_values[mask & np.isfinite(phase_values)]
        phase_values = phase_values[phase_values > 0]
        if phase_values.size:
            values.append(phase_values)
    if not values:
        return np.array([], dtype=float)
    return np.concatenate(values)


def color_limits(values: np.ndarray) -> list[float]:
    if PLOT_VALUE != "mean_rho":
        raise ValueError("This shared correlation color scale requires mean_rho.")
    vmax = float(os.environ.get("SEARCHLIGHT_RSA_RHO_VMAX", "0.5"))
    if not np.isfinite(vmax) or vmax <= 0 or (values.size and np.max(values) > vmax):
        raise ValueError("Rho color maximum must be positive and cover all plotted values.")
    return [0.0, vmax / 2.0, vmax]


def save_standalone_colorbar(limits: list[float], output_dir: Path) -> dict:
    """Save one shared colorbar for the searchlight brain maps."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    colorbar_dir = output_dir / "colorbar"
    colorbar_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(4.2, 0.58), dpi=300)
    norm = Normalize(vmin=limits[0], vmax=limits[-1])
    sm = ScalarMappable(norm=norm, cmap=BRAIN_COLORMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cbar.set_label("Mean Spearman rho (cluster-corrected p < 0.05)", fontsize=9)
    cbar.set_ticks([limits[0], limits[1], limits[2]])
    cbar.set_ticklabels([f"{value:.3f}" for value in limits])
    cbar.ax.tick_params(labelsize=8, length=2.5, width=0.7)
    cbar.outline.set_linewidth(0.7)
    fig.patch.set_alpha(0.0)
    save_pdf(fig, colorbar_dir / "source_searchlight_rsa_shared_colorbar.pdf", bbox_inches="tight", transparent=True)
    plt.close(fig)

    return {
        "file_pdf": str(colorbar_dir / "source_searchlight_rsa_shared_colorbar.pdf"),
        "limits": ";".join(f"{value:.6g}" for value in limits),
    }


def build_label_lookup() -> dict:
    labels = mne.read_labels_from_annot(
        TEMPLATE_SUBJECT,
        parc="aparc",
        subjects_dir=str(SUBJECTS_DIR),
        verbose=False,
    )
    lookup = {}
    for label in labels:
        clean = label.name.replace("-lh", "").replace("-rh", "")
        for vertex in label.vertices:
            lookup[(label.hemi, int(vertex))] = clean
    return lookup


def vertex_to_mni_safe(vertex: int, hemi: str) -> str:
    hemi_index = 0 if hemi == "lh" else 1
    try:
        mni = np.squeeze(
            mne.vertex_to_mni(
                int(vertex),
                hemi_index,
                subject=TEMPLATE_SUBJECT,
                subjects_dir=str(SUBJECTS_DIR),
            )
        )
        return ", ".join(f"{value:.1f}" for value in mni)
    except Exception:
        return ""


def plot_brain_maps() -> pd.DataFrame:
    brain_dir = OUTPUT_DIR / "brain_maps"
    brain_dir.mkdir(parents=True, exist_ok=True)
    values = collect_plot_values()
    limits = color_limits(values)
    print("Brain map color limits:", ", ".join(f"{v:.3f}" for v in limits))
    colorbar_manifest = save_standalone_colorbar(limits, brain_dir)
    manifest_rows = []
    for phase in PHASE_ORDER:
        maps_path = phase_dir(phase) / "source_searchlight_group_maps.npz"
        if not maps_path.exists():
            continue
        group_maps = load_npz(maps_path)
        mask, cluster_rows = significant_cluster_mask(phase, len(group_maps["center_indices"]))
        phase_values = np.asarray(group_maps[PLOT_VALUE], dtype=float)
        phase_values = np.where(phase_values > 0, phase_values, 0.0)
        clim = {"kind": "value", "lims": limits}

        stc = make_stc(group_maps, make_full_source_data(group_maps, phase_values, mask=mask))
        for hemi in HEMIS:
            for view in VIEWS:
                brain = stc.plot(
                    subject=TEMPLATE_SUBJECT,
                    subjects_dir=str(SUBJECTS_DIR),
                    surface=SURFACE,
                    hemi=hemi,
                    views=view,
                    initial_time=0.0,
                    time_viewer=False,
                    time_label=None,
                    colorbar=False,
                    background=BACKGROUND,
                    cortex=CORTEX,
                    alpha=1.0,
                    transparent=False,
                    size=BRAIN_SIZE,
                    clim=clim,
                    colormap=BRAIN_RENDER_COLORMAP,
                    title="",
                )
                stem = f"{safe_name(phase)}_{PLOT_VALUE}_cluster_masked_{hemi}_{view}"
                png_path = brain_dir / f"{stem}.png"
                try:
                    save_brain_png(brain, png_path, base_alpha=BRAIN_BASE_ALPHA)
                finally:
                    brain.close()
                manifest_rows.append(
                    {
                        "phase": phase,
                        "phase_label": PHASE_LABELS.get(phase, phase),
                        "plot_value": PLOT_VALUE,
                        "hemi": hemi,
                        "view": view,
                        "file_png": str(png_path),
                        "shared_colorbar_pdf": colorbar_manifest["file_pdf"],
                        "n_significant_clusters": int(cluster_rows["is_significant"].sum()),
                        "n_cluster_masked_vertices": int(mask.sum()),
                    }
                )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(brain_dir / "brain_map_manifest.csv", index=False, encoding="utf-8-sig")
    return manifest


def make_cluster_tables() -> pd.DataFrame:
    table_dir = OUTPUT_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    lookup = build_label_lookup()
    all_rows = []
    cluster_columns = [
        "phase",
        "phase_label",
        "cluster_index",
        "cluster_p",
        "n_vertices",
        "peak_valid_space_index",
        "peak_hemi",
        "peak_vertex",
        "peak_aparc_label",
        "peak_mni_xyz",
        "peak_t",
        "peak_mean_rho",
        "peak_fdr_bh",
        "cluster_mean_rho",
        "cluster_max_rho",
        "cluster_min_fdr_bh",
    ]

    for phase in PHASE_ORDER:
        maps_path = phase_dir(phase) / "source_searchlight_group_maps.npz"
        cluster_path = phase_dir(phase) / "cluster_permutation_results.pkl"
        vertex_path = phase_dir(phase) / "source_searchlight_rsa_vertex_summary.csv"
        if not maps_path.exists() or not cluster_path.exists() or not vertex_path.exists():
            continue

        group_maps = load_npz(maps_path)
        cluster_results = load_pickle(cluster_path)
        vertex_summary = pd.read_csv(vertex_path)
        center_indices = np.asarray(group_maps["center_indices"], dtype=int)
        valid_indices = np.asarray(group_maps["valid_indices"], dtype=int)
        full_vertices_lh = np.asarray(group_maps["full_vertices_lh"], dtype=int)
        full_vertices_rh = np.asarray(group_maps["full_vertices_rh"], dtype=int)
        n_lh = len(full_vertices_lh)
        t_values = np.asarray(group_maps["t_value"], dtype=float)
        mean_rho = np.asarray(group_maps["mean_rho"], dtype=float)
        p_fdr = np.asarray(group_maps["p_fdr_bh"], dtype=float)

        rows = []
        for cluster_index, cluster_p in enumerate(cluster_results["cluster_p_values"]):
            space_indices = get_cluster_space_indices(cluster_results["clusters"][cluster_index])
            if cluster_p >= CLUSTER_ALPHA:
                continue
            peak_local = int(space_indices[np.nanargmax(np.abs(t_values[space_indices]))])
            full_index = int(valid_indices[center_indices[peak_local]])
            if full_index < n_lh:
                hemi = "lh"
                vertex = int(full_vertices_lh[full_index])
            else:
                hemi = "rh"
                vertex = int(full_vertices_rh[full_index - n_lh])
            aparc = lookup.get((hemi, vertex), "Unknown")
            mni = vertex_to_mni_safe(vertex, hemi)
            cluster_mean_rho = float(np.nanmean(mean_rho[space_indices]))
            cluster_max_rho = float(np.nanmax(mean_rho[space_indices]))
            cluster_min_fdr = float(np.nanmin(p_fdr[space_indices]))
            peak_row = vertex_summary[vertex_summary["valid_space_index"] == peak_local]
            peak_vertex_fdr = (
                float(peak_row["p_fdr_bh"].iloc[0]) if len(peak_row) else float(p_fdr[peak_local])
            )
            row = {
                "phase": phase,
                "phase_label": PHASE_LABELS.get(phase, phase),
                "cluster_index": int(cluster_index),
                "cluster_p": float(cluster_p),
                "n_vertices": int(space_indices.size),
                "peak_valid_space_index": peak_local,
                "peak_hemi": hemi,
                "peak_vertex": vertex,
                "peak_aparc_label": aparc,
                "peak_mni_xyz": mni,
                "peak_t": float(t_values[peak_local]),
                "peak_mean_rho": float(mean_rho[peak_local]),
                "peak_fdr_bh": peak_vertex_fdr,
                "cluster_mean_rho": cluster_mean_rho,
                "cluster_max_rho": cluster_max_rho,
                "cluster_min_fdr_bh": cluster_min_fdr,
            }
            rows.append(row)
            all_rows.append(row)

        phase_table = pd.DataFrame(rows, columns=cluster_columns)
        if not phase_table.empty:
            phase_table = phase_table.sort_values("cluster_p")
        phase_table.to_csv(
            table_dir / f"{safe_name(phase)}_significant_cluster_table.csv",
            index=False,
            encoding="utf-8-sig",
        )

    all_table = pd.DataFrame(all_rows, columns=cluster_columns)
    if not all_table.empty:
        all_table = all_table.sort_values(["phase", "cluster_p", "cluster_index"])
    all_table.to_csv(
        table_dir / "all_phases_significant_cluster_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return all_table


def plot_phase_summary(cluster_table: pd.DataFrame) -> None:
    summary_dir = OUTPUT_DIR / "summary_figures"
    summary_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(SEARCHLIGHT_DIR / "source_searchlight_rsa_manifest.csv")
    manifest["phase_label"] = manifest["phase"].map(PHASE_LABELS)
    manifest["phase_label"] = pd.Categorical(
        manifest["phase_label"],
        categories=[PHASE_LABELS[p] for p in PHASE_ORDER],
        ordered=True,
    )
    manifest = manifest.sort_values("phase_label")
    if cluster_table.empty:
        cluster_counts = pd.DataFrame(columns=["phase", "n_significant_clusters_from_table"])
    else:
        cluster_counts = (
            cluster_table.groupby("phase", dropna=False)
            .size()
            .rename("n_significant_clusters_from_table")
            .reset_index()
        )
    manifest = manifest.merge(cluster_counts, on="phase", how="left")
    manifest["n_significant_clusters_from_table"] = (
        manifest["n_significant_clusters_from_table"].fillna(0).astype(int)
    )

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.0,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5), dpi=300)
    colors = ["#78909C", "#2F5D7C", "#2F7D59", "#A24D2E"]

    axes[0].bar(manifest["phase_label"], manifest["n_cluster_significant"], color=colors)
    axes[0].set_ylabel("Significant clusters")
    axes[0].set_title("Cluster-corrected RSA")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(manifest["phase_label"], manifest["n_fdr_vertices"], color=colors)
    axes[1].set_ylabel("FDR-significant vertices")
    axes[1].set_title("Vertex-level FDR")
    axes[1].tick_params(axis="x", rotation=25)

    axes[2].bar(manifest["phase_label"], -np.log10(manifest["min_p"].astype(float)), color=colors)
    axes[2].set_ylabel("-log10(min vertex p)")
    axes[2].set_title("Peak vertex strength")
    axes[2].tick_params(axis="x", rotation=25)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="0.88", linewidth=0.7)

    fig.suptitle("Source Searchlight RSA: HBRL MF RDM Alignment", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_pdf(fig, summary_dir / "source_searchlight_rsa_phase_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    manifest.to_csv(
        summary_dir / "source_searchlight_rsa_phase_summary_table.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:

    print("Searchlight RSA visualization")
    print(f"Input: {SEARCHLIGHT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Plot value: {PLOT_VALUE}")

    cluster_table = make_cluster_tables()
    plot_phase_summary(cluster_table)
    brain_manifest = plot_brain_maps()

    figure_manifest = {
        "input_dir": str(SEARCHLIGHT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "plot_value": PLOT_VALUE,
        "surface": SURFACE,
        "color_limits": color_limits(collect_plot_values()),
        "color_scale_note": "Linear red-to-yellow mean-rho scale shared across MB/MF and phases; zero background transparent",
        "cortex": str(CORTEX),
        "background": "transparent",
        "render_background": BACKGROUND,
        "brain_base_alpha": BRAIN_BASE_ALPHA,
        "brain_data_alpha": 1.0,
        "brain_data_transparent": False,
        "soften_white_brain_base_png": True,
        "colormap": "#D73027->#FFD700",
        "transparent_png": True,
        "add_cluster_peak_markers": False,
        "add_largest_cluster_marker": False,
        "add_largest_cluster_text": False,
        "add_largest_cluster_callout": False,
        "cluster_alpha": CLUSTER_ALPHA,
        "n_cluster_table_rows": int(len(cluster_table)),
        "n_brain_map_files": int(len(brain_manifest)),
        "main_outputs": {
            "brain_maps": str(OUTPUT_DIR / "brain_maps"),
            "brain_map_colorbar": str(OUTPUT_DIR / "brain_maps" / "colorbar"),
            "tables": str(OUTPUT_DIR / "tables"),
            "summary_figures": str(OUTPUT_DIR / "summary_figures"),
        },
    }
    (OUTPUT_DIR / "searchlight_rsa_figure_manifest.json").write_text(
        json.dumps(figure_manifest, indent=2),
        encoding="utf-8",
    )
    print("\nVisualization complete.")
    print(f"Brain maps: {OUTPUT_DIR / 'brain_maps'}")
    print(f"Tables: {OUTPUT_DIR / 'tables'}")
    print(f"Summary figures: {OUTPUT_DIR / 'summary_figures'}")


if __name__ == "__main__":
    main()


