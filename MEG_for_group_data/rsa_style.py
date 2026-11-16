from itertools import combinations
from pathlib import Path
import re
from matplotlib.backends.backend_pdf import RendererPdf
from meg_pdf import save_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from group_plots import pdf

PHASE_ORDER = ['incentive','decision','transition','feedback']
SUMMARY_CMAP = LinearSegmentedColormap.from_list('rsa_blue',['#fbfcff','#dce7f6','#6f8fc8'])
TOP_CLUSTERS_PER_PHASE = 8
CONDITION_LABELS = ('LS','LL','HS','HL')
RDM_VALUE_FONT_SIZE = 14
RDM_CONDITION_FONT_SIZE = 12
RDM_TITLE_FONT_SIZE = 13

def phase_label(value):
    return value.title()

def select_top_clusters(clusters: pd.DataFrame) -> pd.DataFrame:
    if clusters.empty:
        return clusters.copy()
    return (
        clusters.sort_values(["phase", "cluster_p", "n_vertices"], ascending=[True, True, False])
        .groupby("phase", observed=True, group_keys=False)
        .head(TOP_CLUSTERS_PER_PHASE)
        .reset_index(drop=True)
    )

def plot_cluster_location_bubbles(clusters: pd.DataFrame, model_label, path) -> None:
    plot_df = select_top_clusters(clusters)
    labels = (
        plot_df["peak_label_clean"]
        .astype(str)
        .str.replace("-", " ", regex=False)
        .str.replace("_", " ", regex=False)
    )
    plot_df = plot_df.assign(location_label=labels)

    unique_labels = list(dict.fromkeys(plot_df["location_label"]))
    y_map = {label: idx for idx, label in enumerate(unique_labels[::-1])}
    x_map = {phase: idx for idx, phase in enumerate(PHASE_ORDER)}

    phase_x = plot_df["phase"].astype(str).map(x_map).astype(float)
    within_slot = plot_df.groupby(["phase", "location_label"], observed=True).cumcount()
    slot_count = plot_df.groupby(["phase", "location_label"], observed=True)["cluster_index"].transform("count")
    offsets = (within_slot - (slot_count - 1) / 2) * 0.07
    plot_df = plot_df.assign(x_plot=phase_x + offsets)

    fig, ax = plt.subplots(figsize=(9.6, max(4.2, 0.34 * len(unique_labels) + 1.8)))
    sizes = np.sqrt(plot_df["n_vertices"].fillna(1).clip(lower=1)) * 15
    sizes = sizes.clip(85, 560)
    rho = plot_df["cluster_mean_rho"].fillna(plot_df["peak_mean_rho"])

    scatter = ax.scatter(
        plot_df["x_plot"],
        plot_df["location_label"].map(y_map),
        s=sizes,
        c=rho,
        cmap=SUMMARY_CMAP,
        edgecolors="none",
        linewidths=0,
        alpha=0.9,
    )

    for _, row in plot_df.iterrows():
        ax.text(
            row["x_plot"],
            y_map[row["location_label"]],
            f"C{row['cluster_index']}",
            ha="center",
            va="center",
            fontsize=7,
            color="#222222",
        )

    ax.set_xticks(range(len(PHASE_ORDER)))
    ax.set_xticklabels([phase_label(p) for p in PHASE_ORDER], fontsize=11)
    ax.set_yticks(range(len(unique_labels)))
    ax.set_yticklabels(unique_labels[::-1], fontsize=9)
    ax.set_xlim(-0.55, len(PHASE_ORDER) - 0.45)
    ax.grid(axis="x", color="#EAEAEA", linewidth=0.7)
    ax.set_title(
        f"{model_label} Source Searchlight RSA: Top Cluster Locations by Phase",
        fontsize=14,
        weight="bold",
    )
    ax.set_xlabel("Task phase", fontsize=11)
    ax.set_ylabel("Peak aparc label", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Cluster mean rho", fontsize=10)

    fig.set_size_inches(9.6, max(4.2, 0.34 * len(unique_labels) + 1.8))
    save_pdf(fig,path.with_suffix(".pdf"),bbox_inches="tight",facecolor="white",transparent=False)
    plt.close(fig)

def plot_matrix(ax: plt.Axes, matrix: np.ndarray, title: str, cmap, vmin: float, vmax: float):
    cmap = cmap.copy()
    cmap.set_bad(alpha=0.0)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Draw the complete lower triangle of the four-condition RDM.
    # The diagonal and upper triangle remain transparent.
    for row in range(len(CONDITION_LABELS)):
        for col in range(row):
            value = matrix[row, col]
            if not np.isfinite(value):
                continue
            facecolor = cmap(norm(value))
            ax.add_patch(
                plt.Rectangle(
                    (col, row),
                    1.0,
                    1.0,
                    facecolor=facecolor,
                    edgecolor="none",
                    zorder=1,
                )
            )
            ax.text(
                col + 0.5,
                row + 0.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=RDM_VALUE_FONT_SIZE,
                color="#1f2933",
                weight="normal",
                zorder=2,
            )

    n_conditions = len(CONDITION_LABELS)
    ax.set_xlim(0, n_conditions)
    ax.set_ylim(n_conditions, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n_conditions) + 0.5)
    ax.set_yticks(np.arange(n_conditions) + 0.5)
    ax.set_xticklabels(CONDITION_LABELS, rotation=0, ha="center", fontsize=RDM_CONDITION_FONT_SIZE)
    ax.set_yticklabels(CONDITION_LABELS, fontsize=RDM_CONDITION_FONT_SIZE)
    ax.set_title(title, fontsize=RDM_TITLE_FONT_SIZE, weight="bold", pad=10)
    ax.tick_params(axis="both", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    return scalar_mappable

CONDITION_KEYS = (
    "small_magnitude_low_load",
    "large_magnitude_low_load",
    "small_magnitude_high_load",
    "large_magnitude_high_load",
)
CONDITION_LABELS = ("LS", "LL", "HS", "HL")
RDM_PAIRS = tuple(
    f"{condition_a}__vs__{condition_b}"
    for condition_a, condition_b in combinations(CONDITION_KEYS, 2)
)
PAIR_INDICES = tuple(combinations(range(len(CONDITION_KEYS)), 2))

MEG_COLOR = "#6f8fc8"
BEHAVIOR_COLOR = "#c4879a"
MEG_CMAP = LinearSegmentedColormap.from_list("meg_rdm_soft_blue", ["#fbfcff", "#dce7f6", MEG_COLOR], N=256)
BEHAVIOR_CMAP = LinearSegmentedColormap.from_list(
    "behavior_rdm_soft_rose", ["#fffafd", "#f1d8e1", BEHAVIOR_COLOR], N=256
)

PLOT_RANK_SCALED = False
PHASE_LABELS = {phase: phase.title() for phase in PHASE_ORDER}
def safe_name(text: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()

def vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    matrix = np.full((len(CONDITION_KEYS), len(CONDITION_KEYS)), np.nan, dtype=float)
    for value, (i, j) in zip(vector, PAIR_INDICES):
        matrix[i, j] = value
        matrix[j, i] = value
    return matrix

def finite_max_or_one(vector: np.ndarray) -> float:
    values = np.asarray(vector, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return max(float(np.nanmax(finite)), 1e-12)

def plot_individual_rdm(row: pd.Series, meg_vector: np.ndarray, bhv_vector: np.ndarray, n_subjects: int, MODEL_LABEL, OUTPUT_DIR) -> dict:
    meg_plot = meg_vector
    bhv_plot = bhv_vector
    value_label = "Rank-scaled distance" if PLOT_RANK_SCALED else "Raw distance"


    meg_matrix = vector_to_matrix(meg_plot)
    bhv_matrix = vector_to_matrix(bhv_plot)

    phase = str(row["phase"])
    cluster_id = int(row["cluster_index"])
    peak_label = str(row.get("peak_aparc_label", "unknown"))
    cluster_mean_rho = float(row["cluster_mean_rho"]) if pd.notna(row.get("cluster_mean_rho", np.nan)) else np.nan
    peak_mean_rho = float(row["peak_mean_rho"]) if pd.notna(row.get("peak_mean_rho", np.nan)) else np.nan
    stem = f"{safe_name(phase)}_c{cluster_id}_{safe_name(peak_label)}_rdm"

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.8))
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.set_facecolor("none")
    im0 = plot_matrix(axes[0], meg_matrix, "MEG RDM", MEG_CMAP, 0, 1 if PLOT_RANK_SCALED else finite_max_or_one(meg_plot))
    im1 = plot_matrix(axes[1], bhv_matrix, "Behavior RDM", BEHAVIOR_CMAP, 0, 1 if PLOT_RANK_SCALED else finite_max_or_one(bhv_plot))
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).set_label(value_label, fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).set_label(value_label, fontsize=8)
    fig.suptitle(
        f"{MODEL_LABEL} searchlight representative RDM | "
        f"{PHASE_LABELS.get(phase, phase)} C{cluster_id} {peak_label}\n"
        f"cluster mean rho = {cluster_mean_rho:.3f}, peak mean rho = {peak_mean_rho:.3f}, "
        f"cluster p = {row['cluster_p']:.4g}, n = {n_subjects}",
        fontsize=13,
        y=1.03,
    )
    fig.tight_layout()

    pdf_path = OUTPUT_DIR / "individual_rdms" / f"{stem}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    # Measure the original labels before renaming so the crop and all plot
    # positions remain fixed, even though the new text is slightly narrower.
    for ax in axes:
        ax.set_xticklabels(CONDITION_LABELS, fontsize=RDM_CONDITION_FONT_SIZE)
        ax.set_yticklabels(CONDITION_LABELS, fontsize=RDM_CONDITION_FONT_SIZE)
    fig.set_dpi(72)
    fig.canvas.draw()
    renderer = RendererPdf(None, 72, *fig.get_size_inches()[::-1])
    original_bbox = fig.get_tightbbox(renderer).padded(plt.rcParams["savefig.pad_inches"])
    for ax in axes:
        ax.set_xticklabels(CONDITION_LABELS, fontsize=RDM_CONDITION_FONT_SIZE)
        ax.set_yticklabels(CONDITION_LABELS, fontsize=RDM_CONDITION_FONT_SIZE)
    save_pdf(fig, pdf_path, bbox_inches=original_bbox,
                transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)


BRAIN_COLORMAP = LinearSegmentedColormap.from_list("rsa_red_yellow", ["#D73027","#FFD700"], N=256)
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
