"""
Plot representative cluster RDMs for source searchlight RSA.

The main searchlight analysis saves spatial RSA maps, not RDM matrices. This
script reconstructs representative cluster-level RDMs by:
  1) selecting the top cluster-level significant cluster per phase by default;
  2) reading the same four condition source STCs used by the searchlight;
  3) averaging time within the phase window;
  4) computing MEG distances as 1 - Pearson r between four condition patterns
     inside the selected cluster;
  5) averaging subject-specific MEG and HBRL behavior RDMs for visualization.

Run in Spyder:
  python 03_plot_representative_searchlight_cluster_rdms.py

Optional before running:
  import os
  os.environ["SEARCHLIGHT_RSA_RDM_TOP_N"] = "2"
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from meg_project import DATA_ROOT, behavior_rdm, project_input, RSA_RESULTS, available, source_stem
from meg_pdf import save_pdf

import importlib
import json
import os
import pickle
import re
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import RendererPdf
from scipy import stats

if sys.getrecursionlimit() < 50000:
    sys.setrecursionlimit(50000)


SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name.endswith("_mf"):
    MODEL_TAG = "mf"
    MODEL_LABEL = "Model-free"
    RESULT_DIR_NAME = "overall_mf_source_searchlight"
else:
    MODEL_TAG = "mb"
    MODEL_LABEL = "Model-based"
    RESULT_DIR_NAME = "overall_mb_source_searchlight"

SEARCHLIGHT_DIR = project_input(
    os.environ.get(
        "SEARCHLIGHT_RSA_RESULT_DIR",
        str(RSA_RESULTS / "searchlight_seed42" / RESULT_DIR_NAME),
    )
)
FIGURE_DIR = SEARCHLIGHT_DIR / "Figures_publication"
OUTPUT_DIR = FIGURE_DIR / "representative_rdms"

DEFAULT_TOP_CLUSTERS_PER_PHASE = "3" if MODEL_TAG == "mf" else "1"
TOP_CLUSTERS_PER_PHASE_FOR_RDM = int(
    os.environ.get("SEARCHLIGHT_RSA_RDM_TOP_N", DEFAULT_TOP_CLUSTERS_PER_PHASE)
)
PLOT_RANK_SCALED = os.environ.get("SEARCHLIGHT_RSA_RDM_RANK_SCALE", "FALSE").upper() == "TRUE"

PHASE_ORDER = ["incentive", "decision", "transition", "feedback"]
PHASE_LABELS = {
    "incentive": "Incentive",
    "decision": "Decision",
    "transition": "Transition",
    "feedback": "Feedback",
}
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

RDM_VALUE_FONT_SIZE = 14
RDM_CONDITION_FONT_SIZE = 12
RDM_TITLE_FONT_SIZE = 13


def parse_subid(value) -> int | float:
    text = str(value).strip().removeprefix("sub-")
    if text.endswith(".0"):
        text = text[:-2]
    return int(text) if text.isdecimal() else np.nan


def safe_name(text: object) -> str:
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


def load_config() -> dict:
    config_path = SEARCHLIGHT_DIR / "source_searchlight_rsa_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing searchlight config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_manifest() -> pd.DataFrame:
    manifest_path = SEARCHLIGHT_DIR / "source_searchlight_rsa_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing searchlight manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    manifest["phase"] = pd.Categorical(manifest["phase"], categories=PHASE_ORDER, ordered=True)
    return manifest.sort_values("phase")


def load_cluster_table() -> pd.DataFrame:
    table_path = FIGURE_DIR / "tables" / "all_phases_significant_cluster_table.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing cluster table: {table_path}")
    table = pd.read_csv(table_path)
    if table.empty:
        return table
    table["phase"] = pd.Categorical(table["phase"], categories=PHASE_ORDER, ordered=True)
    for col in ("cluster_index", "cluster_p", "n_vertices", "cluster_mean_rho", "peak_mean_rho"):
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce")
    return table.sort_values(["phase", "cluster_p", "n_vertices"], ascending=[True, True, False])


def select_representative_clusters(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    return (
        table.groupby("phase", observed=True, group_keys=False)
        .head(TOP_CLUSTERS_PER_PHASE_FOR_RDM)
        .reset_index(drop=True)
    )


def parse_time_window(text: str) -> tuple[float, float]:
    values = re.split(r"\s*-\s*", str(text).strip(), maxsplit=1)
    if len(values) < 2:
        raise ValueError(f"Cannot parse time window: {text}")
    return float(values[0]), float(values[1])


def read_single_stc_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as file:
        tmin_ms = np.fromfile(file, dtype=">f4", count=1)
        tstep_ms = np.fromfile(file, dtype=">f4", count=1)
        n_vertices_header = np.fromfile(file, dtype=">u4", count=1)
        if tmin_ms.size != 1 or tstep_ms.size != 1 or n_vertices_header.size != 1:
            raise RuntimeError(f"Could not read STC header: {path}")
        n_vertices = int(n_vertices_header[0])
        vertices = np.fromfile(file, dtype=">u4", count=n_vertices)
        n_times_header = np.fromfile(file, dtype=">u4", count=1)
        if vertices.size != n_vertices or n_times_header.size != 1:
            raise RuntimeError(f"Could not read STC vertex/time counts: {path}")
        n_times = int(n_times_header[0])
        data = np.fromfile(file, dtype=">f4", count=n_vertices * n_times)
        if data.size != n_vertices * n_times:
            raise RuntimeError(f"Incomplete STC data: {path}")
    data = data.reshape((n_times, n_vertices)).T
    times = (float(tmin_ms[0]) + np.arange(n_times) * float(tstep_ms[0])) / 1000.0
    return data, times


def read_paired_stc(stem: Path) -> tuple[np.ndarray, np.ndarray]:
    lh_path = Path(f"{stem}-lh.stc")
    rh_path = Path(f"{stem}-rh.stc")
    if not lh_path.exists() or not rh_path.exists():
        raise FileNotFoundError(f"Missing paired STC files for stem: {stem}")
    lh_data, lh_times = read_single_stc_file(lh_path)
    rh_data, rh_times = read_single_stc_file(rh_path)
    if not np.allclose(lh_times, rh_times, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"Left/right STC times differ for {stem}")
    return np.vstack([lh_data, rh_data]), lh_times


def condition_stc_stem(source_morphed_dir: Path, subject: str, condition: str, data_phase: str) -> Path:
    return source_stem(source_morphed_dir, subject, data_phase, condition)


def average_condition_values(
    source_morphed_dir: Path,
    subject: str,
    data_phase: str,
    conditions: list[str],
    valid_indices: np.ndarray,
    time_window: tuple[float, float],
) -> np.ndarray:
    rows = []
    tmin, tmax = time_window
    for condition in conditions:
        data, times = read_paired_stc(condition_stc_stem(source_morphed_dir, subject, condition, data_phase))
        keep = (times >= tmin - 1e-9) & (times <= tmax + 1e-9)
        if not np.any(keep):
            raise RuntimeError(f"No STC time points inside {time_window} for subject {subject}, condition {condition}")
        rows.append(np.nanmean(data[:, keep], axis=1)[valid_indices])
    return np.asarray(rows, dtype=float)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def meg_rdm_vector(condition_values: np.ndarray, cluster_valid_indices: np.ndarray) -> np.ndarray:
    local = condition_values[:, cluster_valid_indices]
    distances = []
    for i, j in PAIR_INDICES:
        r = pearson_corr(local[i], local[j])
        distances.append(np.nan if np.isnan(r) else 1.0 - r)
    return np.asarray(distances, dtype=float)


def load_behavior_vectors(path: Path, subjects: list[str]) -> dict[int, np.ndarray]:
    behavior = pd.read_csv(path)
    behavior["subid"] = behavior["subid"].map(parse_subid)
    wanted = {parse_subid(subject) for subject in subjects}
    vectors: dict[int, np.ndarray] = {}
    for subid, df_subject in behavior[behavior["subid"].isin(wanted)].groupby("subid"):
        pair_map = dict(zip(df_subject["rdm_pair"], df_subject["rdm_distance"]))
        if all(pair in pair_map for pair in RDM_PAIRS):
            vectors[int(subid)] = np.asarray([pair_map[pair] for pair in RDM_PAIRS], dtype=float)
    return vectors


def vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    matrix = np.full((len(CONDITION_KEYS), len(CONDITION_KEYS)), np.nan, dtype=float)
    for value, (i, j) in zip(vector, PAIR_INDICES):
        matrix[i, j] = value
        matrix[j, i] = value
    return matrix


def rank_scale_vector(vector: np.ndarray) -> np.ndarray:
    out = np.full_like(vector, np.nan, dtype=float)
    ok = np.isfinite(vector)
    if ok.sum() <= 1:
        return out
    ranks = stats.rankdata(vector[ok], method="average") - 1.0
    out[ok] = ranks / (ok.sum() - 1.0)
    return out


def finite_max_or_one(vector: np.ndarray) -> float:
    values = np.asarray(vector, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return max(float(np.nanmax(finite)), 1e-12)


def lower_triangle(matrix: np.ndarray) -> np.ndarray:
    out = matrix.copy()
    out[np.triu_indices_from(out, k=0)] = np.nan
    return out


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


def compute_target_rdms(
    row: pd.Series,
    config: dict,
    manifest: pd.DataFrame,
    behavior_vectors: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    phase = str(row["phase"])
    phase_spec = config["phase_specs"][phase]
    group_maps = np.load(SEARCHLIGHT_DIR / phase / "source_searchlight_group_maps.npz", allow_pickle=True)
    valid_indices = np.asarray(group_maps["valid_indices"], dtype=int)
    subjects = [f"{int(subject):02d}" for subject in group_maps["subjects"]]
    time_window_text = manifest.loc[manifest["phase"].astype(str) == phase, "time_window"].iloc[0]
    time_window = parse_time_window(time_window_text)

    cluster_results = load_pickle(SEARCHLIGHT_DIR / phase / "cluster_permutation_results.pkl")
    cluster = cluster_results["clusters"][int(row["cluster_index"])]
    cluster_valid_indices = np.asarray(cluster[0] if isinstance(cluster, tuple) else cluster, dtype=int)

    source_morphed_dir = DATA_ROOT / "MEG" / "source" / "morphed_stcs"
    meg_vectors = []
    bhv_vectors = []
    for subject in subjects:
        subid = int(parse_subid(subject))
        if subid not in behavior_vectors:
            continue
        condition_values = average_condition_values(
            source_morphed_dir=source_morphed_dir,
            subject=subject,
            data_phase=phase_spec["data_phase"],
            conditions=list(phase_spec["conditions"]),
            valid_indices=valid_indices,
            time_window=time_window,
        )
        meg_vectors.append(meg_rdm_vector(condition_values, cluster_valid_indices))
        bhv_vectors.append(behavior_vectors[subid])

    if not meg_vectors:
        raise RuntimeError(f"No usable subjects for {phase}, cluster {row['cluster_index']}")

    return np.nanmean(np.vstack(meg_vectors), axis=0), np.nanmean(np.vstack(bhv_vectors), axis=0), len(meg_vectors)


def plot_individual_rdm(row: pd.Series, meg_vector: np.ndarray, bhv_vector: np.ndarray, n_subjects: int) -> dict:
    meg_plot = rank_scale_vector(meg_vector) if PLOT_RANK_SCALED else meg_vector
    bhv_plot = rank_scale_vector(bhv_vector) if PLOT_RANK_SCALED else bhv_vector
    value_label = "Rank-scaled distance" if PLOT_RANK_SCALED else "Raw distance"
    rho, p_value = stats.spearmanr(meg_vector, bhv_vector, nan_policy="omit")

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

    return {
        "phase": phase,
        "phase_label": PHASE_LABELS.get(phase, phase),
        "cluster_index": cluster_id,
        "cluster_p": row["cluster_p"],
        "n_vertices": row["n_vertices"],
        "peak_aparc_label": peak_label,
        "n_subjects": n_subjects,
        "cluster_mean_rho": cluster_mean_rho,
        "peak_mean_rho": peak_mean_rho,
        "group_rdm_spearman_rho": rho,
        "group_rdm_spearman_p": p_value,
        "file_pdf": str(pdf_path),
    }


def plot_contact_sheet(records: list[dict], vectors: list[tuple[np.ndarray, np.ndarray]]) -> None:
    if not records:
        return
    n_rows = len(records)
    fig, axes = plt.subplots(n_rows, 2, figsize=(9.5, max(4.4, 3.5 * n_rows)))
    fig.patch.set_alpha(0.0)
    if n_rows == 1:
        axes = np.asarray([axes])

    for row_axes in axes:
        for ax in row_axes:
            ax.set_facecolor("none")

    for row_idx, (record, (meg_vector, bhv_vector)) in enumerate(zip(records, vectors)):
        meg_plot = rank_scale_vector(meg_vector) if PLOT_RANK_SCALED else meg_vector
        bhv_plot = rank_scale_vector(bhv_vector) if PLOT_RANK_SCALED else bhv_vector
        plot_matrix(
            axes[row_idx, 0],
            vector_to_matrix(meg_plot),
            "MEG RDM",
            MEG_CMAP,
            0,
            1 if PLOT_RANK_SCALED else finite_max_or_one(meg_plot),
        )
        plot_matrix(
            axes[row_idx, 1],
            vector_to_matrix(bhv_plot),
            "Behavior RDM",
            BEHAVIOR_CMAP,
            0,
            1 if PLOT_RANK_SCALED else finite_max_or_one(bhv_plot),
        )
        axes[row_idx, 0].set_ylabel(
            f"{record['phase_label']} C{record['cluster_index']}\n{record['peak_aparc_label']}",
            fontsize=11,
            weight="bold",
        )

    fig.suptitle(
        f"{MODEL_LABEL} Searchlight RSA: Representative Cluster RDMs\n"
        +
        (
            "Lower triangles show rank-scaled group-mean distances"
            if PLOT_RANK_SCALED
            else "Lower triangles show raw group-mean distances"
        ),
        fontsize=14,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_pdf(fig,
        OUTPUT_DIR / "fig5_representative_cluster_rdms.pdf",
        bbox_inches="tight",
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    plt.close(fig)


def main() -> None:

    print("Representative searchlight cluster RDMs")
    print(f"Model: {MODEL_LABEL} ({MODEL_TAG})")
    print(f"Input: {SEARCHLIGHT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Representative clusters per phase: {TOP_CLUSTERS_PER_PHASE_FOR_RDM}")

    config = load_config()
    manifest = load_manifest()
    clusters = select_representative_clusters(load_cluster_table())
    if clusters.empty:
        print("No cluster-level significant clusters available.")
        return

    first_phase = str(clusters.iloc[0]["phase"])
    first_group_maps = np.load(SEARCHLIGHT_DIR / first_phase / "source_searchlight_group_maps.npz", allow_pickle=True)
    subjects = [f"{int(subject):02d}" for subject in first_group_maps["subjects"]]
    behavior_vectors = load_behavior_vectors(behavior_rdm(MODEL_TAG), subjects)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    distance_rows = []
    contact_vectors = []

    for _, row in clusters.iterrows():
        meg_vector, bhv_vector, n_subjects = compute_target_rdms(row, config, manifest, behavior_vectors)
        record = plot_individual_rdm(row, meg_vector, bhv_vector, n_subjects)
        manifest_rows.append(record)
        contact_vectors.append((meg_vector, bhv_vector))
        for pair, meg_distance, bhv_distance in zip(RDM_PAIRS, meg_vector, bhv_vector):
            distance_rows.append(
                {
                    **{key: record[key] for key in ["phase", "phase_label", "cluster_index", "peak_aparc_label"]},
                    "rdm_pair": pair,
                    "meg_distance": meg_distance,
                    "behavior_distance": bhv_distance,
                }
            )
        print(
            f"  {record['phase_label']} C{record['cluster_index']} "
            f"{record['peak_aparc_label']}: cluster_mean_rho={record['cluster_mean_rho']:.3f}"
        )

    plot_contact_sheet(manifest_rows, contact_vectors)
    pd.DataFrame(manifest_rows).to_csv(OUTPUT_DIR / "representative_rdm_manifest.csv", index=False)
    pd.DataFrame(distance_rows).to_csv(OUTPUT_DIR / "representative_rdm_distances.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
