"""
Create pattern-RSA-style publication figures for source searchlight RSA.

This script reads the already computed source searchlight RSA outputs and makes
summary figures analogous to the pattern RSA publication figures:
  1) phase-ordered top-cluster lollipop plot
  2) phase-level cluster-count/extent summary
  3) phase-ordered cluster-location bubble plot
  4) source brain-map contact sheet using the existing phase maps
  5) all-significant-cluster lollipop plot for supplementary inspection

It does not recompute RSA statistics and does not overwrite the existing
brain-map files. Output goes to:
  <searchlight result>/Figures_publication/pattern_style_summary

Run in Spyder or a terminal:
  python 02_plot_source_searchlight_rsa_pattern_style_figures.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from meg_project import project_input, RSA_RESULTS
from meg_pdf import save_pdf

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name.endswith("_mf"):
    MODEL_TAG = "mf"
    RESULT_DIR_NAME = "overall_mf_source_searchlight"
    MODEL_LABEL = "Model-free"
elif SCRIPT_DIR.name.endswith("_mb"):
    MODEL_TAG = "mb"
    RESULT_DIR_NAME = "overall_mb_source_searchlight"
    MODEL_LABEL = "Model-based"
else:
    MODEL_TAG = os.environ.get("SEARCHLIGHT_RSA_MODEL_TAG", "mb").lower()
    RESULT_DIR_NAME = f"overall_{MODEL_TAG}_source_searchlight"
    MODEL_LABEL = "Model-free" if MODEL_TAG == "mf" else "Model-based"

SEARCHLIGHT_DIR = project_input(
    os.environ.get(
        "SEARCHLIGHT_RSA_RESULT_DIR",
        str(RSA_RESULTS / "searchlight_seed42" / RESULT_DIR_NAME),
    )
)
FIGURE_DIR = project_input(
    os.environ.get(
        "SEARCHLIGHT_RSA_FIGURE_DIR",
        str(SEARCHLIGHT_DIR / "Figures_publication"),
    )
)
OUTPUT_DIR = FIGURE_DIR / "pattern_style_summary"

PHASE_ORDER = ["incentive", "decision", "transition", "feedback"]
PHASE_LABELS = {
    "incentive": "Incentive",
    "decision": "Decision",
    "transition": "Transition",
    "feedback": "Feedback",
}
MEG_COLOR = "#6f8fc8"
BEHAVIOR_COLOR = "#c4879a"
LIGHT_MEG_COLOR = "#dce7f6"
SUMMARY_CMAP = LinearSegmentedColormap.from_list(
    "searchlight_summary_soft_blue",
    ["#fbfcff", "#dce7f6", MEG_COLOR],
    N=256,
)
PHASE_COLORS = {
    "incentive": MEG_COLOR,
    "decision": MEG_COLOR,
    "transition": BEHAVIOR_COLOR,
    "feedback": BEHAVIOR_COLOR,
}

TOP_CLUSTERS_PER_PHASE = int(os.environ.get("SEARCHLIGHT_RSA_TOP_CLUSTERS_PER_PHASE", "8"))


# =============================================================================
# Helpers
# =============================================================================


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()


def format_p(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "< .001"
    return f"= {p:.3f}".replace("0.", ".")


def phase_label(phase: str) -> str:
    return PHASE_LABELS.get(str(phase), str(phase))


def save_figure(fig: plt.Figure, name: str, width: float, height: float) -> None:
    pdf_path = OUTPUT_DIR / f"{name}.pdf"
    fig.set_size_inches(width, height)
    transparent = os.environ.get("SEARCHLIGHT_RSA_TRANSPARENT_PNG", "FALSE").upper() == "TRUE"
    facecolor = "none" if transparent else "white"
    save_pdf(fig, pdf_path, bbox_inches="tight", facecolor=facecolor, transparent=transparent)
    plt.close(fig)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_path = require_file(FIGURE_DIR / "tables" / "all_phases_significant_cluster_table.csv")
    summary_path = require_file(
        FIGURE_DIR / "summary_figures" / "source_searchlight_rsa_phase_summary_table.csv"
    )

    clusters = pd.read_csv(cluster_path)
    summary = pd.read_csv(summary_path)

    if clusters.empty:
        clusters = pd.DataFrame(
            columns=[
                "phase",
                "phase_label",
                "cluster_index",
                "cluster_p",
                "n_vertices",
                "peak_aparc_label",
                "peak_mean_rho",
                "cluster_mean_rho",
                "cluster_max_rho",
                "cluster_min_fdr_bh",
            ]
        )

    for frame in (clusters, summary):
        if "phase" in frame.columns:
            frame["phase"] = pd.Categorical(frame["phase"], categories=PHASE_ORDER, ordered=True)
            frame["phase_label_display"] = frame["phase"].astype(str).map(phase_label)

    numeric_cols = [
        "cluster_p",
        "n_vertices",
        "peak_t",
        "peak_mean_rho",
        "cluster_mean_rho",
        "cluster_max_rho",
        "cluster_min_rho",
        "cluster_min_fdr_bh",
        "n_cluster_significant",
        "n_significant_clusters_from_table",
        "n_fdr_vertices",
        "min_p",
    ]
    for col in numeric_cols:
        for frame in (clusters, summary):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if "phase_label" in clusters.columns:
        clusters["phase_label_original"] = clusters["phase_label"]
    clusters["phase_label"] = clusters["phase"].astype(str).map(phase_label)
    clusters["cluster_number"] = clusters["cluster_index"].astype(str)
    clusters["peak_label_clean"] = clusters.get("peak_aparc_label", pd.Series("", index=clusters.index))
    clusters["peak_label_clean"] = clusters["peak_label_clean"].fillna("unknown").replace("", "unknown")
    clusters["cluster_display"] = (
        clusters["phase_label"]
        + " | C"
        + clusters["cluster_number"]
        + " | "
        + clusters["peak_label_clean"]
        + " | n="
        + clusters["n_vertices"].fillna(0).astype(int).astype(str)
        + " | p "
        + clusters["cluster_p"].apply(format_p)
    )
    clusters = clusters.sort_values(["phase", "cluster_p", "n_vertices"], ascending=[True, True, False])

    if "n_significant_clusters_from_table" not in summary.columns:
        summary["n_significant_clusters_from_table"] = summary.get("n_cluster_significant", 0)
    summary["phase_label"] = summary["phase"].astype(str).map(phase_label)
    summary = summary.sort_values("phase")

    return clusters, summary


def select_top_clusters(clusters: pd.DataFrame) -> pd.DataFrame:
    if clusters.empty:
        return clusters.copy()
    return (
        clusters.sort_values(["phase", "cluster_p", "n_vertices"], ascending=[True, True, False])
        .groupby("phase", observed=True, group_keys=False)
        .head(TOP_CLUSTERS_PER_PHASE)
        .reset_index(drop=True)
    )


def add_vertical_phase_guides(ax: plt.Axes, y_positions: dict[str, list[int]]) -> None:
    for phase, positions in y_positions.items():
        if not positions:
            continue
        ymin, ymax = min(positions) - 0.45, max(positions) + 0.45
        ax.axhspan(ymin, ymax, color=PHASE_COLORS.get(phase, "#999999"), alpha=0.045, lw=0)


# =============================================================================
# Figures
# =============================================================================


def plot_cluster_lollipop(clusters: pd.DataFrame, name: str, title_suffix: str) -> None:
    if clusters.empty:
        fig, ax = plt.subplots(figsize=(8, 2.6))
        ax.text(0.5, 0.5, "No cluster-level significant results", ha="center", va="center")
        ax.axis("off")
        save_figure(fig, name, 8, 2.6)
        return

    plot_df = clusters.copy().reset_index(drop=True)
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot_df))
    colors = [PHASE_COLORS.get(str(p), MEG_COLOR) for p in plot_df["phase"].astype(str)]

    height = max(4.0, 0.33 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, height))

    y_positions: dict[str, list[int]] = {phase: [] for phase in PHASE_ORDER}
    for idx, phase in enumerate(plot_df["phase"].astype(str)):
        y_positions.setdefault(phase, []).append(idx)
    add_vertical_phase_guides(ax, y_positions)

    if {"cluster_min_rho", "cluster_max_rho"}.issubset(plot_df.columns):
        ax.hlines(
            y,
            plot_df["cluster_min_rho"],
            plot_df["cluster_max_rho"],
            color=colors,
            alpha=0.35,
            linewidth=1.4,
            zorder=1,
        )

    sizes = np.sqrt(plot_df["n_vertices"].fillna(1).clip(lower=1)) * 8
    sizes = sizes.clip(35, 260)
    ax.scatter(
        plot_df["cluster_mean_rho"],
        y,
        s=sizes,
        facecolors="#FFFFFF",
        edgecolors=colors,
        linewidths=1.6,
        zorder=3,
    )

    ax.axvline(0, color="#9A9A9A", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["cluster_display"], fontsize=8.5)
    ax.set_xlabel("Cluster mean Spearman rho", fontsize=12)
    ax.set_title(f"{MODEL_LABEL} Source Searchlight RSA: {title_suffix}", fontsize=14, weight="bold")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x_values = plot_df["cluster_mean_rho"].dropna()
    if len(x_values):
        pad = max(0.03, (x_values.max() - x_values.min()) * 0.15)
        ax.set_xlim(min(0, x_values.min() - pad), x_values.max() + pad)

    save_figure(fig, name, 11.5, height)


def plot_phase_extent_summary(summary: pd.DataFrame) -> None:
    plot_df = summary.copy()
    x = np.arange(len(plot_df))
    colors = [PHASE_COLORS.get(str(p), MEG_COLOR) for p in plot_df["phase"].astype(str)]

    fig, axes = plt.subplots(3, 1, figsize=(8.8, 7.2), sharex=True)

    metrics = [
        ("n_significant_clusters_from_table", "Cluster-level significant clusters", False),
        ("n_fdr_vertices", "FDR-significant vertices", False),
        ("min_p", "Minimum vertex-wise p", True),
    ]

    for ax, (col, ylabel, log_scale) in zip(axes, metrics):
        values = plot_df[col].fillna(0)
        ax.bar(x, values, color=colors, alpha=0.82, width=0.62)
        for xi, value in zip(x, values):
            label = f"{int(value)}" if col != "min_p" else format_p(float(value))
            ax.text(xi, value, label, ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if log_scale:
            positive = values[values > 0]
            if len(positive):
                ax.set_yscale("log")
                ax.invert_yaxis()
                ax.set_ylim(max(positive) * 1.8, min(positive) / 1.8)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(plot_df["phase_label"], fontsize=11)
    axes[0].set_title(f"{MODEL_LABEL} Source Searchlight RSA: Phase-Level Summary", fontsize=14, weight="bold")
    save_figure(fig, "fig2_phase_level_cluster_extent_summary", 8.8, 7.2)


def plot_cluster_location_bubbles(clusters: pd.DataFrame) -> None:
    if clusters.empty:
        fig, ax = plt.subplots(figsize=(8, 2.6))
        ax.text(0.5, 0.5, "No cluster-level significant results", ha="center", va="center")
        ax.axis("off")
        save_figure(fig, "fig3_phase_ordered_cluster_location_bubbles", 8, 2.6)
        return

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
        f"{MODEL_LABEL} Source Searchlight RSA: Top Cluster Locations by Phase",
        fontsize=14,
        weight="bold",
    )
    ax.set_xlabel("Task phase", fontsize=11)
    ax.set_ylabel("Peak aparc label", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Cluster mean rho", fontsize=10)

    save_figure(fig, "fig3_phase_ordered_cluster_location_bubbles", 9.6, max(4.2, 0.34 * len(unique_labels) + 1.8))


def write_plotting_tables(clusters: pd.DataFrame, summary: pd.DataFrame) -> None:
    clusters_out = clusters.copy()
    clusters_out["phase"] = clusters_out["phase"].astype(str)
    clusters_out = clusters_out[
        [
            col
            for col in [
                "phase",
                "phase_label",
                "cluster_index",
                "cluster_p",
                "n_vertices",
                "peak_hemi",
                "peak_vertex",
                "peak_aparc_label",
                "peak_mni_xyz",
                "peak_t",
                "peak_mean_rho",
                "cluster_mean_rho",
                "cluster_max_rho",
                "cluster_min_fdr_bh",
                "cluster_display",
            ]
            if col in clusters_out.columns
        ]
    ]
    clusters_out.to_csv(OUTPUT_DIR / "searchlight_pattern_style_plotting_table.csv", index=False)

    top_out = select_top_clusters(clusters).copy()
    if not top_out.empty:
        top_out["phase"] = top_out["phase"].astype(str)
    top_out.to_csv(OUTPUT_DIR / "searchlight_pattern_style_top_clusters.csv", index=False)

    summary_out = summary.copy()
    summary_out["phase"] = summary_out["phase"].astype(str)
    summary_out.to_csv(OUTPUT_DIR / "searchlight_pattern_style_phase_summary_table.csv", index=False)

    manifest = pd.DataFrame(
        {
            "figure": [
                "fig1_top_clusters_lollipop",
                "fig2_phase_level_cluster_extent_summary",
                "fig3_phase_ordered_cluster_location_bubbles",
                "figS1_all_significant_clusters_lollipop",
            ],
            "description": [
                f"Top {TOP_CLUSTERS_PER_PHASE} cluster-level significant source searchlight RSA clusters per phase.",
                "Phase-level counts of cluster-level significant clusters, FDR-significant vertices, and minimum vertex-wise p.",
                f"Top {TOP_CLUSTERS_PER_PHASE} cluster locations per phase, with point size indicating cluster extent.",
                "All cluster-level significant source searchlight RSA clusters.",
            ],
        }
    )
    manifest.to_csv(OUTPUT_DIR / "figure_manifest.csv", index=False)


def main() -> None:

    print("Pattern-style searchlight RSA visualization")
    print(f"Model: {MODEL_LABEL} ({MODEL_TAG})")
    print(f"Input: {SEARCHLIGHT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    clusters, summary = read_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_plotting_tables(clusters, summary)

    top_clusters = select_top_clusters(clusters)
    plot_cluster_lollipop(
        top_clusters,
        "fig1_top_clusters_lollipop",
        f"Top {TOP_CLUSTERS_PER_PHASE} Significant Clusters per Phase",
    )
    plot_phase_extent_summary(summary)
    plot_cluster_location_bubbles(clusters)
    plot_cluster_lollipop(
        clusters,
        "figS1_all_significant_clusters_lollipop",
        "All Cluster-Level Significant Clusters",
    )

    print("Done.")
    print(f"Figures: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
