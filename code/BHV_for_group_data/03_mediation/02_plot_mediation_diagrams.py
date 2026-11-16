"""Draw reward-effect moderated mediation diagrams.

This version places small- and large-reward conditional paths in the same panel so
reward moderation can be read directly from path differences.
"""
from __future__ import annotations

from pathlib import Path
import os
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "results" / "mediation"
OUTPUT_DIR = ANALYSIS_DIR / "outputs"
FIGURE_DIR = ANALYSIS_DIR / "figures"

SUMMARY_FILE = OUTPUT_DIR / "moderated_mediation_bootstrap_summary.csv"
BOOT_FILE = OUTPUT_DIR / "bootstrap_effects_long.csv"
PATH_FILE = OUTPUT_DIR / "diagram_path_estimates.csv"

# Palette of the selected MEG single-mediator analysis.
SMALL = "#346db1"
LARGE = "#b03949"
SMALL_TEXT = SMALL
LARGE_TEXT = LARGE
GRAY = "#666666"
BLACK = "#111111"

MEDIATOR_ORDER = ["beta_mb", "beta_mf", "w"]
MEDIATOR_LABELS = {"beta_mb": "MB weight", "beta_mf": "MF weight", "w": "Mixing weight"}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 12,
    "pdf.fonttype": 42,
    "pdf.use14corefonts": False,
})


def fmt_est(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.3f}".replace("-0.000", "0.000")


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "p = NA"
    if p < 0.001:
        return "p < .001"
    if p < 0.01:
        return f"p = {p:.3f}".replace("0.", ".")
    if p < 0.05:
        return f"p = {p:.3f}".replace("0.", ".")
    return f"p = {p:.3f}"


def stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt_star_est(x: float, p: float, prefix: str = "") -> str:
    if pd.isna(x):
        val = "NA"
    else:
        val = f"{x:.3f}".replace("-0.000", "0.000")
    return f"{prefix}{val}{stars(p)}"


def fmt_ci(lo: float, hi: float) -> str:
    return f"95% CI [{fmt_est(lo)}, {fmt_est(hi)}]"


def bootstrap_p(values: pd.Series | np.ndarray) -> float:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return math.nan
    return float(min(1.0, 2.0 * min(np.mean(vals >= 0), np.mean(vals <= 0))))


def build_path_table() -> pd.DataFrame:
    summary = pd.read_csv(SUMMARY_FILE)
    rows = []
    for _, row in summary.iterrows():
        rows.append(dict(
            mediator=row.mediator, mediator_label=MEDIATOR_LABELS[row.mediator], reward=row.magnitude,
            a_estimate=row.a_load_to_mediator, a_p_boot=row.a_p_boot,
            b_estimate=row.b_mediator_to_cr, b_p_boot=row.b_p_boot,
            cprime_estimate=row.direct_cprime, cprime_p_boot=row.direct_p_boot,
            indirect_estimate=row.indirect_ab, indirect_p_boot=row.indirect_p_boot,
            indirect_ci_low=row.indirect_ci_low, indirect_ci_high=row.indirect_ci_high,
            total_estimate=row.total_c, total_p_boot=row.total_p_boot))
    paths = pd.DataFrame(rows)
    paths.to_csv(PATH_FILE, index=False, encoding="utf-8-sig")
    return paths


def delta_table(paths: pd.DataFrame) -> pd.DataFrame:
    contrasts = pd.read_csv(OUTPUT_DIR / "moderation_contrasts.csv")
    rows = []
    for mediator in MEDIATOR_ORDER:
        row = {"mediator": mediator}
        for effect, short in [("a", "a"), ("b", "b"), ("indirect", "ab"), ("direct", "cprime"), ("total", "total")]:
            matches = contrasts[contrasts.mediator.eq(mediator) & contrasts.effect.eq(effect)]
            if len(matches) != 1:
                raise ValueError(f"Expected one contrast for {mediator}: {effect}")
            value = matches.iloc[0]
            row[f"delta_{short}_estimate"] = value.estimate
            for column in ("ci_low", "ci_high", "p_boot"):
                row[f"delta_{short}_{column}"] = value[column]
        rows.append(row)
    deltas = pd.DataFrame(rows)
    deltas.to_csv(OUTPUT_DIR / "diagram_reward_deltas.csv", index=False, encoding="utf-8-sig")
    return deltas


def add_node(ax, xy, w, h, text, face):
    x, y = xy
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face, edgecolor="none", linewidth=0, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=11.0, linespacing=1.22, color=BLACK, zorder=3)


def add_arrow(ax, start, end, color, rad=0.0, lw=2.1, ls="-"):
    patch = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=14,
        linewidth=lw, linestyle=ls, color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0, shrinkB=0, zorder=1,
    )
    ax.add_patch(patch)


def plain_text(ax, x, y, text, color=BLACK, size=9.0, ha="center", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color, linespacing=1.18, zorder=5)


def row_for(paths, med, reward):
    sub = paths[(paths["mediator"].astype(str) == med) & (paths["reward"].astype(str) == reward)]
    if sub.empty:
        raise ValueError(f"Missing path row for {med}/{reward}")
    return sub.iloc[0]


def draw_mediator_panel(ax, med: str, paths: pd.DataFrame, deltas: pd.DataFrame):
    small = row_for(paths, med, "small")
    large = row_for(paths, med, "large")
    delta = deltas[deltas["mediator"] == med].iloc[0]
    med_label = MEDIATOR_LABELS.get(med, str(small.get("mediator_label", med)))
    med_node_label = med_label.replace(" ", "\n")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(med_label, fontweight="normal", pad=6)

    node_fill = "#f7f7f7"
    add_node(ax, (0.12, 0.28), 0.20, 0.12, "WM load", node_fill)
    add_node(ax, (0.50, 0.72), 0.22, 0.14, med_node_label, node_fill)
    add_node(ax, (0.88, 0.28), 0.22, 0.14, "Corrected\nreward rate", node_fill)
    add_node(ax, (0.50, 0.43), 0.23, 0.12, "Reward\nmagnitude", node_fill)

    # Conditional paths: large reward is always drawn on the inner track,
    # small reward on the outer track, so path labels can be read consistently.
    add_arrow(ax, (0.19, 0.36), (0.42, 0.67), SMALL, rad=0.20, lw=2.2)
    add_arrow(ax, (0.20, 0.31), (0.42, 0.65), LARGE, rad=-0.03, lw=2.2)
    add_arrow(ax, (0.58, 0.67), (0.81, 0.36), SMALL, rad=-0.20, lw=2.2)
    add_arrow(ax, (0.58, 0.65), (0.80, 0.31), LARGE, rad=0.03, lw=2.2)
    add_arrow(ax, (0.22, 0.28), (0.78, 0.28), GRAY, rad=0.0, lw=1.5, ls="--")
    add_arrow(ax, (0.50, 0.48), (0.50, 0.64), BLACK, rad=0.0, lw=1.4)

    plain_text(ax, 0.29, 0.62, fmt_star_est(small["a_estimate"], small["a_p_boot"]), SMALL_TEXT, size=8.8, ha="right")
    plain_text(ax, 0.24, 0.52, fmt_star_est(large["a_estimate"], large["a_p_boot"]), LARGE_TEXT, size=8.8, ha="right")
    plain_text(ax, 0.71, 0.62, fmt_star_est(small["b_estimate"], small["b_p_boot"]), SMALL_TEXT, size=8.8, ha="left")
    plain_text(ax, 0.76, 0.52, fmt_star_est(large["b_estimate"], large["b_p_boot"]), LARGE_TEXT, size=8.8, ha="left")
    plain_text(
        ax, 0.50, 0.18,
        fmt_star_est(small["cprime_estimate"], small["cprime_p_boot"]) + " / " +
        fmt_star_est(large["cprime_estimate"], large["cprime_p_boot"]),
        GRAY, size=8.4, va="center"
    )

    plain_text(
        ax, 0.14, 0.08,
        fmt_star_est(small["indirect_estimate"], small["indirect_p_boot"], "Indirect: "),
        SMALL_TEXT, size=8.6, ha="left", va="bottom"
    )
    plain_text(
        ax, 0.86, 0.08,
        fmt_star_est(large["indirect_estimate"], large["indirect_p_boot"], "Indirect: "),
        LARGE_TEXT, size=8.6, ha="right", va="bottom"
    )
    plain_text(
        ax, 0.50, 0.96,
        "Large − small reward\n" +
        fmt_star_est(delta["delta_a_estimate"], delta["delta_a_p_boot"], "Δa ") + "   " +
        fmt_star_est(delta["delta_b_estimate"], delta["delta_b_p_boot"], "Δb ") + "\n" +
        fmt_star_est(delta["delta_ab_estimate"], delta["delta_ab_p_boot"], "Δab "),
        BLACK, size=8.4, ha="center", va="top"
    )


def save(fig, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_DIR / f"{name}.pdf"
    fig.savefig(pdf, format="pdf", dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(pdf)


def main():
    paths = build_path_table()
    deltas = delta_table(paths)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8))
    for ax, med in zip(axes, MEDIATOR_ORDER):
        draw_mediator_panel(ax, med, paths, deltas)
    handles = [
        Line2D([0], [0], color=SMALL, lw=3, label="Small reward"),
        Line2D([0], [0], color=LARGE, lw=3, label="Large reward"),
        Line2D([0], [0], color=GRAY, lw=2, ls="--", label="Direct path c′"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Reward moderation of multilevel mediation paths", y=1.08, fontsize=13, fontweight="normal")
    fig.subplots_adjust(wspace=0.12, top=0.78, bottom=0.04)
    save(fig, "moderated_mediation_reward_effect_all_mediators")

    for med in MEDIATOR_ORDER:
        fig, ax = plt.subplots(figsize=(7.0, 5.8))
        draw_mediator_panel(ax, med, paths, deltas)
        fig.legend(handles=handles[:2], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f"Reward moderation through {MEDIATOR_LABELS[med]}", y=1.08, fontsize=13, fontweight="normal")
        fig.subplots_adjust(top=0.80, bottom=0.04)
        save(fig, f"moderated_mediation_reward_effect_{med}")

    print(OUTPUT_DIR / "diagram_reward_deltas.csv")


if __name__ == "__main__":
    main()
