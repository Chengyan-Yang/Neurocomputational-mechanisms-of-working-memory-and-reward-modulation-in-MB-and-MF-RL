"""Plot cluster-mask location figures for behavior-from-MEG regression results.

This script reads the significant-row CSV files produced by
03_plot_behavior_from_meg_regression_results.R and creates clean cluster masks:

  - ERF source clusters: binary spatial mask on a white fsaverage brain with
    visible gyri/sulci.
  - TFA sensor clusters: yellow-highlighted cluster sensors on a blank sensor
    layout.

Run in the MNE-Python/Spyder environment:
  %runfile 04_plot_behavior_from_meg_cluster_masks.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meg_project import DATA_ROOT, MEG_CODE, project_input
from meg_pdf import save_pdf, save_brain_png

import os
import pickle
import re
import traceback

if sys.getrecursionlimit() < 50000:
    sys.setrecursionlimit(50000)

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Circle, Polygon

try:
    import mne
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "This script requires MNE-Python. Please run it in the mne conda "
        "environment, or set MNE_PYTHON to that environment's python.exe."
    ) from error

try:
    from mne.channels.layout import _find_topomap_coords
except Exception:  # pragma: no cover - private MNE helper may move.
    _find_topomap_coords = None


ANALYSIS_DIR = project_input(
    os.environ.get(
        "BHV_MEG_REG_DIR",
        str(MEG_CODE / "5_BHV_MEG_regression"),
    )
)
RESULTS_ROOT = project_input(
    os.environ.get("MEG_RESULTS_DIR", str(MEG_CODE))
)
OUTPUT_ROOT = Path(
    os.environ.get(
        "BHV_FROM_MEG_MASK_OUT",
        str(ANALYSIS_DIR / "results" / "behavior_from_meg_visualizations" / "cluster_masks"),
    )
)
SUMMARY_DIR = ANALYSIS_DIR / "results" / "behavior_from_meg_visualizations" / "summary_tables"

SUBJECTS_DIR = project_input(
    os.environ.get(
        "SUBJECTS_DIR",
        f"{DATA_ROOT}/MRI/surf",
    )
)
TEMPLATE_SUBJECT = "fsaverage"
TEMPLATE_EPOCHS = project_input(
    os.environ.get(
        "TFA_TEMPLATE_EPOCHS",
        f"{DATA_ROOT}/MEG/preprocessed/time_frequency_epoch/sub-01/sub-01_incentive_tfr-epo.fif",
    )
)

PHASE_LABELS = {
    "incentive": "Incentive",
    "decision": "Decision",
    "transition": "Transition",
    "feedback": "Feedback",
}

ANALYSIS_LABELS = {
    "source_cortex_mask": "ERF source",
    "tfa_sensor": "TFA",
}

EFFECT_LABELS = {
    "magnitude": "Reward",
    "reward": "Reward",
    "load": "Load",
    "interaction": "Interaction",
}

HEMIS = ("lh", "rh")
VIEWS = ("lateral", "medial")

SOURCE_MASK_COLOR = os.environ.get("BHV_FROM_MEG_SOURCE_MASK_COLOR", "#F2C500")
TFA_MASK_COLOR = os.environ.get("BHV_FROM_MEG_TFA_MASK_COLOR", "#fff49d")
TFA_MASK_EDGE = os.environ.get("BHV_FROM_MEG_TFA_MASK_EDGE", "#B99D00")
TFA_SENSOR_EDGE = os.environ.get("BHV_FROM_MEG_TFA_SENSOR_EDGE", "#b8b8b8")
SOURCE_SURFACE = os.environ.get("BHV_FROM_MEG_SOURCE_SURFACE", "pial")
SOURCE_CORTEX = (1.0, 1.0, 1.0)
if "BHV_FROM_MEG_SOURCE_CORTEX" in os.environ:
    SOURCE_CORTEX = os.environ["BHV_FROM_MEG_SOURCE_CORTEX"]
SOURCE_BACKGROUND = os.environ.get("BHV_FROM_MEG_SOURCE_BACKGROUND", "white")
BRAIN_SIZE = tuple(
    int(value)
    for value in os.environ.get("BHV_FROM_MEG_SOURCE_SIZE", "1050,850").split(",")
)


def safe_name(text: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()


def install_numpy_pickle_compat() -> None:
    try:
        numpy_core = __import__("numpy.core", fromlist=[""])
    except Exception:
        return
    sys.modules.setdefault("numpy._core", numpy_core)
    for name in ("multiarray", "numeric", "umath"):
        old_name = f"numpy.core.{name}"
        new_name = f"numpy._core.{name}"
        try:
            module = __import__(old_name, fromlist=[""])
        except Exception:
            continue
        sys.modules.setdefault(new_name, module)


def load_pickle(path: Path) -> dict:
    try:
        with path.open("rb") as file:
            return pickle.load(file)
    except ModuleNotFoundError as error:
        if not str(error).startswith("No module named 'numpy._core"):
            raise
        install_numpy_pickle_compat()
        with path.open("rb") as file:
            return pickle.load(file)


def phase_pretty(phase: str) -> str:
    return PHASE_LABELS.get(str(phase), str(phase))


def analysis_pretty(analysis: str) -> str:
    return ANALYSIS_LABELS.get(str(analysis), str(analysis))


def effect_pretty(effect: str) -> str:
    return EFFECT_LABELS.get(str(effect), str(effect))


def band_to_path(band: str, source_file: str = "") -> str:
    text = f"{band} {source_file}".lower()
    if "low" in text and "gamma" in text:
        return "low_gamma"
    if "theta" in text:
        return "theta"
    if "alpha" in text:
        return "alpha"
    if "beta" in text:
        return "beta"
    if "gamma" in text:
        return "gamma"
    return safe_name(band)


def source_pkl_path(row: pd.Series) -> Path:
    return (
        RESULTS_ROOT / "3_ERF_source_level" / "results"
        / "source_cortex_mask"
        / str(row["phase"])
        / ("magnitude" if row["effect"] == "reward" else str(row["effect"]))
        / "Cluster_permutation_results_10000_seed42.pkl"
    )


def tfa_pkl_path(row: pd.Series) -> Path:
    band = band_to_path(row.get("band", ""), row.get("source_file", ""))
    return (
        RESULTS_ROOT / "4_time_frequency" / "time_frequency_analysis" / "results"
        / "sensor"
        / str(row["phase"])
        / "total"
        / band
        / ("magnitude" if row["effect"] == "reward" else str(row["effect"]))
        / "cluster_results.pkl"
    )


def load_significant_rows() -> list[tuple[str, pd.DataFrame]]:
    files = [
        ("raw_p_significant", SUMMARY_DIR / "behavior_from_meg_raw_p_significant_rows.csv"),
        (
            "phasewise_fdr_significant",
            SUMMARY_DIR / "behavior_from_meg_phasewise_fdr_significant_rows.csv",
        ),
    ]
    loaded = []
    for label, path in files:
        if not path.exists():
            print(f"Skipping {label}: missing {path}")
            continue
        table = pd.read_csv(path)
        if table.empty:
            print(f"Skipping {label}: no significant rows.")
            continue
        key_cols = [
            "analysis",
            "phase",
            "effect",
            "band",
            "cluster_number",
            "cluster_index",
            "time_start_ms",
            "time_end_ms",
            "source_file",
        ]
        keep = [col for col in key_cols if col in table.columns]
        table = table.drop_duplicates(subset=keep).copy()
        loaded.append((label, table))
    return loaded


def make_stc(results: dict, data: np.ndarray) -> mne.SourceEstimate:
    times = np.asarray(results["times"], dtype=float)
    tstep = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    return mne.SourceEstimate(
        data,
        vertices=[np.asarray(v, dtype=int) for v in results["full_vertices"]],
        tmin=0.0,
        tstep=tstep,
        subject=TEMPLATE_SUBJECT,
    )


def cluster_space_mask(results: dict, cluster_index: int) -> np.ndarray:
    n_vertices = int(np.asarray(results["T_obs"]).shape[1])
    _, space_indices = results["clusters"][cluster_index]
    mask = np.zeros(n_vertices, dtype=float)
    mask[np.unique(np.asarray(space_indices, dtype=int))] = 1.0
    return mask


def plot_source_mask(row: pd.Series, out_dir: Path) -> list[dict]:
    pkl_path = source_pkl_path(row)
    if not pkl_path.exists():
        return [{"status": "missing", "note": f"Missing source pkl: {pkl_path}"}]

    results = load_pickle(pkl_path)
    cluster_index = int(row["cluster_index"])
    mask = cluster_space_mask(results, cluster_index)
    if not np.any(mask):
        return [{"status": "empty", "note": "Cluster mask has no vertices."}]

    stc = make_stc(results, mask[:, np.newaxis])
    n_lh = len(results["full_vertices"][0])
    cluster_vertices = np.where(mask > 0)[0]
    hemis = []
    if np.any(cluster_vertices < n_lh):
        hemis.append("lh")
    if np.any(cluster_vertices >= n_lh):
        hemis.append("rh")

    cmap = LinearSegmentedColormap.from_list(
        "cluster_mask_yellow",
        [SOURCE_MASK_COLOR, SOURCE_MASK_COLOR],
        N=2,
    )
    title = ""
    outputs = []
    for hemi in hemis:
        for view in VIEWS:
            stem = (
                f"{safe_name(row['phase'])}_{safe_name(row['analysis'])}_"
                f"{safe_name(row['effect'])}_{safe_name(row.get('band', 'source'))}_"
                f"c{int(row['cluster_number'])}_{int(row['time_start_ms'])}_"
                f"{int(row['time_end_ms'])}_{hemi}_{view}_mask"
            )
            png_path = out_dir / "source" / f"{stem}.png"
            png_path.parent.mkdir(parents=True, exist_ok=True)
            brain = stc.plot(
                subject=TEMPLATE_SUBJECT,
                subjects_dir=str(SUBJECTS_DIR),
                surface=SOURCE_SURFACE,
                hemi=hemi,
                views=view,
                initial_time=stc.times[0],
                time_viewer=False,
                time_label=None,
                colorbar=False,
                background=SOURCE_BACKGROUND,
                cortex=SOURCE_CORTEX,
                alpha=1.0,
                transparent=True,
                size=BRAIN_SIZE,
                clim={"kind": "value", "pos_lims": [0.5, 0.75, 1.0]},
                colormap=cmap,
                title=title,
            )
            try:
                save_brain_png(brain, png_path)
            finally:
                brain.close()
            outputs.append(
                {
                    "status": "ok",
                    "analysis": row["analysis"],
                    "phase": row["phase"],
                    "effect": row["effect"],
                    "band": row.get("band", "Source"),
                    "cluster_number": int(row["cluster_number"]),
                    "cluster_index": cluster_index,
                    "hemi": hemi,
                    "view": view,
                    "n_mask_vertices": int(mask.sum()),
                    "file_png": str(png_path),
                    "transparent": True,
                    "note": "Binary spatial source mask.",
                }
            )
    return outputs


def template_info_for_channels(channel_names: list[str]) -> mne.Info:
    if not TEMPLATE_EPOCHS.exists():
        raise FileNotFoundError(f"Missing template epochs: {TEMPLATE_EPOCHS}")
    info = mne.io.read_info(str(TEMPLATE_EPOCHS), verbose="error")
    missing = [name for name in channel_names if name not in info.ch_names]
    if missing:
        raise RuntimeError(f"Template epochs are missing channels: {missing[:5]}")
    picks = [info.ch_names.index(name) for name in channel_names]
    return mne.pick_info(info, picks, copy=True)


def plot_sensor_mask(row: pd.Series, out_dir: Path) -> list[dict]:
    pkl_path = tfa_pkl_path(row)
    if not pkl_path.exists():
        return [{"status": "missing", "note": f"Missing TFA pkl: {pkl_path}"}]

    results = load_pickle(pkl_path)
    cluster_index = int(row["cluster_index"])
    _, sensor_indices = results["clusters"][cluster_index]
    sensor_indices = np.unique(np.asarray(sensor_indices, dtype=int))
    channel_names = [str(name) for name in results["channel_names"]]
    info = template_info_for_channels(channel_names)

    if _find_topomap_coords is None:
        raise RuntimeError("Could not access MNE topomap coordinates helper.")
    coords = _find_topomap_coords(
        info,
        np.arange(len(channel_names)),
        ignore_overlap=True,
        sphere=(0.0, 0.0, 0.0, 0.12),
    )

    fig, axis = plt.subplots(figsize=(4.3, 4.0), dpi=300)
    radius = float(np.max(np.linalg.norm(coords, axis=1)) * 1.08)
    axis.add_patch(
        Circle(
            (0, 0),
            radius=radius,
            facecolor="white",
            edgecolor="#6f6f6f",
            linewidth=1.1,
            zorder=1,
        )
    )
    nose = np.array(
        [
            [-0.08 * radius, 0.98 * radius],
            [0.0, 1.14 * radius],
            [0.08 * radius, 0.98 * radius],
        ]
    )
    axis.add_patch(
        Polygon(nose, closed=False, fill=False, edgecolor="#6f6f6f", linewidth=1.1, zorder=2)
    )
    axis.add_patch(
        Arc(
            (-1.02 * radius, 0),
            width=0.20 * radius,
            height=0.34 * radius,
            theta1=270,
            theta2=90,
            color="#6f6f6f",
            linewidth=1.0,
            zorder=2,
        )
    )
    axis.add_patch(
        Arc(
            (1.02 * radius, 0),
            width=0.20 * radius,
            height=0.34 * radius,
            theta1=90,
            theta2=270,
            color="#6f6f6f",
            linewidth=1.0,
            zorder=2,
        )
    )
    axis.scatter(
        coords[:, 0],
        coords[:, 1],
        s=20,
        facecolors="white",
        edgecolors=TFA_SENSOR_EDGE,
        linewidths=0.6,
        zorder=3,
    )
    axis.scatter(
        coords[sensor_indices, 0],
        coords[sensor_indices, 1],
        s=20,
        facecolors=TFA_MASK_COLOR,
        edgecolors=TFA_MASK_EDGE,
        linewidths=0.8,
        zorder=4,
    )
    axis.set_aspect("equal")
    lim = radius * 1.22
    axis.set_xlim(-lim, lim)
    axis.set_ylim(-lim * 0.98, lim * 1.08)
    axis.set_axis_off()

    stem = (
        f"{safe_name(row['phase'])}_{safe_name(row['analysis'])}_"
        f"{safe_name(row['effect'])}_{safe_name(row.get('band', 'tfa'))}_"
        f"c{int(row['cluster_number'])}_{int(row['time_start_ms'])}_"
        f"{int(row['time_end_ms'])}_sensor_mask"
    )
    pdf_path = out_dir / "tfa_sensor" / f"{stem}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    save_pdf(fig, pdf_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return [
        {
            "status": "ok",
            "analysis": row["analysis"],
            "phase": row["phase"],
            "effect": row["effect"],
            "band": row.get("band", "TFA"),
            "cluster_number": int(row["cluster_number"]),
            "cluster_index": cluster_index,
            "hemi": "",
            "view": "topomap",
            "n_mask_vertices": "",
            "n_mask_sensors": int(len(sensor_indices)),
            "file_pdf": str(pdf_path),
            "transparent": True,
            "note": "Yellow dots mark sensors included in the cluster.",
        }
    ]


def describe_row(row: pd.Series) -> str:
    band = row.get("band", "")
    band_text = f" {band}" if str(row["analysis"]) == "tfa_sensor" else ""
    return (
        f"{phase_pretty(row['phase'])} | {analysis_pretty(row['analysis'])}"
        f"{band_text} | {effect_pretty(row['effect'])} | "
        f"C{int(row['cluster_number'])} | "
        f"{int(row['time_start_ms'])}-{int(row['time_end_ms'])} ms"
    )


def plot_one(row: pd.Series, out_dir: Path) -> list[dict]:
    if row["analysis"] == "source_cortex_mask":
        return plot_source_mask(row, out_dir)
    if row["analysis"] == "tfa_sensor":
        return plot_sensor_mask(row, out_dir)
    return [{"status": "skipped", "note": f"Unsupported analysis: {row['analysis']}"}]


def main() -> None:
    print("Behavior-from-MEG cluster-mask plotting")
    print(f"Summary dir: {SUMMARY_DIR}")
    print(f"Output root: {OUTPUT_ROOT}")

    manifest_rows = []
    for significance_set, table in load_significant_rows():
        out_dir = OUTPUT_ROOT / significance_set
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{significance_set}: {len(table)} unique clusters/results")
        for _, row in table.iterrows():
            label = describe_row(row)
            print(f"  plotting {label}")
            try:
                outputs = plot_one(row, out_dir)
            except Exception as error:
                outputs = [
                    {
                        "status": "error",
                        "analysis": row.get("analysis", ""),
                        "phase": row.get("phase", ""),
                        "effect": row.get("effect", ""),
                        "band": row.get("band", ""),
                        "cluster_number": row.get("cluster_number", ""),
                        "cluster_index": row.get("cluster_index", ""),
                        "file_pdf": "",
                        "note": repr(error),
                        "traceback": traceback.format_exc(limit=3),
                    }
                ]
            for output in outputs:
                output.update(
                    {
                        "significance_set": significance_set,
                        "label": label,
                    }
                )
                manifest_rows.append(output)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = OUTPUT_ROOT / "cluster_mask_manifest.csv"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    n_ok = int((manifest.get("status", pd.Series(dtype=str)) == "ok").sum())
    n_error = int((manifest.get("status", pd.Series(dtype=str)) == "error").sum())
    print(f"\nManifest: {manifest_path}")
    print(f"Mask plotting complete: {n_ok} image entries ok, {n_error} errors.")


if __name__ == "__main__":
    main()
