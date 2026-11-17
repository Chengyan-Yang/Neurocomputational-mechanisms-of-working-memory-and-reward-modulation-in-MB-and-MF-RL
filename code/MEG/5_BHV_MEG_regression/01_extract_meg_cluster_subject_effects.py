# -*- coding: utf-8 -*-
"""Extract subject-level MEG cluster effects for brain-behavior regression.

This script does not rerun cluster permutation tests. It reads saved cluster
result files, extracts each subject's cluster-mean condition values, computes
the corresponding contrast, and writes CSV files that can be used by the
regression script.

Default preset:
    planned-load

The default extracts the main load clusters that are most natural to pair with
HBRL beta_mb_load: Decision 0-1500 ms and Transition, across ERF source and
selected TFA bands. ERF sensor results are intentionally excluded from this
brain-behavior regression extraction.

Examples:
    python 01_extract_meg_cluster_subject_effects.py

    python 01_extract_meg_cluster_subject_effects.py --preset all-tfa-load

    python 01_extract_meg_cluster_subject_effects.py --preset all-effects ^
      --output-dir "results/meg_cluster_subject_effects_all_phases"

    python 01_extract_meg_cluster_subject_effects.py --preset core-effects ^
      --output-dir "results/meg_cluster_subject_effects_core_phases_phasewise"

    python 01_extract_meg_cluster_subject_effects.py ^
      --single-pkl "../4_time_frequency/time_frequency_analysis/results/sensor/decision/total/alpha/load/cluster_results.pkl"

Notes:
    - TFA cluster result files contain condition_data and can be extracted
      directly from the pkl.
    - ERF source pkl files contain cluster masks but not full subject condition
      data; extracting them requires access to the morphed paired STC files
      in data/MEG/source/morphed_stcs. The STC files are read
      directly with NumPy to avoid importing MNE/SciPy during this extraction.
    - For TFA/source, meg_effect_raw is usually the more interpretable signed
      contrast. meg_effect_signed is retained as a diagnostic sign-aligned
      alternative.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meg_project import DATA_ROOT, MEG_CODE, project_input, available, source_stem

import argparse
import csv
import importlib
import pickle
import re
import traceback
from dataclasses import dataclass
from typing import Iterable

import numpy as np


RESULTS_ROOT_DEFAULT = MEG_CODE
OUTPUT_DIR_DEFAULT = Path(
    f"{MEG_CODE}/5_BHV_MEG_regression"
    "/results/meg_cluster_subject_effects_core_phases_phasewise"
)
CODE_ROOT_DEFAULT = Path(f"{MEG_CODE}")
SCRIPT_VERSION = "2026-08-20-phasewise-cluster-filter"

CONDITION_LABELS = (
    "small_magnitude_low_load",
    "large_magnitude_low_load",
    "small_magnitude_high_load",
    "large_magnitude_high_load",
)

CONDITION_LABELS_READABLE = (
    "small reward / Low load",
    "large reward / Low load",
    "small reward / High load",
    "large reward / High load",
)

WEIGHTS = {
    "load": np.array([-0.5, -0.5, 0.5, 0.5], dtype=float),
    "reward": np.array([-0.5, 0.5, -0.5, 0.5], dtype=float),
    "interaction": np.array([0.5, -0.5, -0.5, 0.5], dtype=float),
    "WM load main effect": np.array([-0.5, -0.5, 0.5, 0.5], dtype=float),
    "Reward main effect": np.array([-0.5, 0.5, -0.5, 0.5], dtype=float),
    "WM load x Reward interaction": np.array([0.5, -0.5, -0.5, 0.5], dtype=float),
}

PHASE_CONDITIONS = {'incentive': ('1', '2', '3', '4'),
 'decision': ('11', '21', '31', '41'),
 'transition': ('13', '23', '33', '43'),
 'feedback': ('51', '52', '53', '54')}


CORE_PHASES = ("incentive", "decision", "transition", "feedback")

PHASE_ALIASES = {
    "discussion_1500": "decision",
    "decision_1500": "decision",
    "decision": "decision",
    "feedback": "feedback",
}

SENSOR_PHASE_CONFIG = {
    'incentive': dict(data_phase="incentive", tmin=0.0, tmax=1.0),
    'decision': dict(data_phase="decision", tmin=0.0, tmax=1.5),
    'transition': dict(data_phase="transition", tmin=0.0, tmax=0.5),
    'feedback': dict(
        data_phase="feedback", tmin=0.0, tmax=1.5
    ),
}


@dataclass(frozen=True)
class ExtractionTarget:
    pkl_path: Path
    analysis: str
    phase: str
    effect: str
    label: str


def install_numpy_pickle_compat() -> None:
    """Allow NumPy 1.x to read pickle files written by some NumPy 2.x runs."""
    # Only install the missing NumPy 2.x module names when they are actually
    # needed. Do not alias numpy.core in NumPy 2.x environments, because that
    # can break SciPy/MNE lazy imports.
    try:
        numpy_core = importlib.import_module("numpy.core")
    except Exception:
        return

    sys.modules.setdefault("numpy._core", numpy_core)
    for name in ("multiarray", "numeric", "umath"):
        old_name = f"numpy.core.{name}"
        new_name = f"numpy._core.{name}"
        try:
            module = importlib.import_module(old_name)
        except Exception:
            continue
        sys.modules.setdefault(new_name, module)


def safe_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def parse_effect_from_path(path: Path, saved_effect: str | None) -> str:
    if saved_effect:
        return str(saved_effect)

    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if "interaction" in parts or "interaction" in name:
        return "interaction"
    if "magnitude" in parts or "magnitude_main_effect" in name:
        return "reward"
    if "load" in parts or "wm_load_main_effect" in name:
        return "load"
    return "unknown"


def standard_effect_key(effect: str) -> str:
    lower = str(effect).lower()
    if "interaction" in lower:
        return "interaction"
    if "magnitude" in lower or "reward" in lower:
        return "reward"
    if "load" in lower:
        return "load"
    return str(effect)


def effect_weights(results: dict, effect: str) -> np.ndarray:
    weights = results.get("effect_weights")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape == (4,):
            return weights

    key = standard_effect_key(effect)
    if key in WEIGHTS:
        return WEIGHTS[key]
    if effect in WEIGHTS:
        return WEIGHTS[effect]
    raise ValueError(f"Cannot infer effect weights for effect={effect!r}")


def load_pickle(path: Path) -> dict:
    # Source-cluster pickle files can contain deeply nested MNE/numpy objects.
    # The Python default recursion limit is sometimes too low to unpickle them.
    if sys.getrecursionlimit() < 10000:
        sys.setrecursionlimit(10000)
    try:
        with path.open("rb") as file:
            return pickle.load(file)
    except ModuleNotFoundError as error:
        if not str(error).startswith("No module named 'numpy._core"):
            raise
        install_numpy_pickle_compat()
        with path.open("rb") as file:
            return pickle.load(file)


def cluster_indices_to_extract(
    results: dict, cluster_alpha: float, include_all: bool
) -> list[int]:
    p_values = np.asarray(results["cluster_p_values"], dtype=float)
    if include_all:
        return list(range(len(p_values)))

    good = results.get("good_cluster_inds")
    if good is not None:
        return [int(index) for index in np.asarray(good, dtype=int)]

    return [int(index) for index in np.where(p_values < cluster_alpha)[0]]


def cluster_display_number(results: dict, cluster_index: int) -> int:
    """Keep the original report numbering after stricter corrected filtering."""
    good = results.get("good_cluster_inds")
    if good is None:
        return int(cluster_index) + 1
    good_indices = [int(index) for index in np.asarray(good, dtype=int)]
    if int(cluster_index) in good_indices:
        return good_indices.index(int(cluster_index)) + 1
    return int(cluster_index) + 1


def cluster_correction_multiplier(
    target: ExtractionTarget,
    args: argparse.Namespace,
) -> float:
    if args.cluster_correction == "raw":
        return 1.0
    if target.analysis == "tfa_sensor":
        return float(args.tfa_phasewise_multiplier)
    if target.analysis == "source_cortex_mask":
        return float(args.source_phasewise_multiplier)
    return 1.0


def cluster_entries_to_extract(
    target: ExtractionTarget,
    results: dict,
    args: argparse.Namespace,
) -> list[tuple[int, int, float, float]]:
    p_values = np.asarray(results["cluster_p_values"], dtype=float)
    multiplier = cluster_correction_multiplier(target, args)

    if args.include_all_clusters:
        candidate_indices = list(range(len(p_values)))
    else:
        candidate_indices = cluster_indices_to_extract(results, args.cluster_alpha, False)

    entries = []
    for cluster_index in candidate_indices:
        raw_p = float(p_values[int(cluster_index)])
        corrected_p = min(raw_p * multiplier, 1.0)
        if args.include_all_clusters or corrected_p < args.cluster_alpha:
            entries.append(
                (
                    int(cluster_index),
                    cluster_display_number(results, int(cluster_index)),
                    raw_p,
                    corrected_p,
                )
            )
    return entries


def target_band(target: ExtractionTarget) -> str:
    if target.analysis != "tfa_sensor":
        return ""
    parts = list(target.pkl_path.parts)
    try:
        sensor_i = [part.lower() for part in parts].index("sensor")
        return str(parts[sensor_i + 3])
    except Exception:
        return ""


def point_weights_from_t(
    t_obs: np.ndarray | None,
    time_indices: np.ndarray,
    space_indices: np.ndarray,
) -> np.ndarray:
    """Return sign-alignment weights for exact cluster points."""
    if t_obs is None:
        return np.ones(len(time_indices), dtype=float)

    cluster_t = np.asarray(t_obs)[time_indices, space_indices]
    if cluster_t.size == 0:
        return np.ones(len(time_indices), dtype=float)

    peak_sign = np.sign(cluster_t[np.argmax(np.abs(cluster_t))])
    if peak_sign == 0:
        peak_sign = 1.0

    weights = np.ones(len(space_indices), dtype=float)
    for space_index in np.unique(space_indices):
        mask = space_indices == space_index
        space_mean_t = np.mean(np.asarray(t_obs)[time_indices[mask], space_index])
        space_sign = np.sign(space_mean_t)
        if space_sign == 0:
            space_sign = peak_sign
        weights[mask] = space_sign * peak_sign
    return weights


def contrast_from_condition_means(
    condition_means: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    return condition_means @ weights


def condition_values_from_array(
    condition_data: np.ndarray,
    time_indices: np.ndarray,
    space_indices: np.ndarray,
    signed_point_weights: np.ndarray | None = None,
) -> np.ndarray:
    point_values = condition_data[:, :, time_indices, space_indices]
    if signed_point_weights is not None:
        point_values = point_values * signed_point_weights[np.newaxis, np.newaxis, :]
    return point_values.mean(axis=2)


def make_rows(
    *,
    target: ExtractionTarget,
    results: dict,
    cluster_index: int,
    cluster_number: int,
    cluster_p_corrected: float,
    cluster_correction_multiplier_value: float,
    condition_means_raw: np.ndarray,
    condition_means_signed: np.ndarray,
    units: str,
) -> list[dict]:
    subjects = [f"{int(subject):02d}" for subject in results["valid_subjects"]]
    weights = effect_weights(results, target.effect)
    p_value = float(np.asarray(results["cluster_p_values"])[cluster_index])
    time_indices, space_indices = results["clusters"][cluster_index]
    time_indices = np.asarray(time_indices, dtype=int)
    space_indices = np.asarray(space_indices, dtype=int)
    times = np.asarray(results["times"], dtype=float)
    time_start_ms = float(times[time_indices.min()] * 1000.0)
    time_end_ms = float(times[time_indices.max()] * 1000.0)

    meg_raw = contrast_from_condition_means(condition_means_raw, weights)
    meg_signed = contrast_from_condition_means(condition_means_signed, weights)

    rows = []
    for subject_i, subject in enumerate(subjects):
        row = {
            "subid": int(re.sub(r"[^0-9]", "", subject)),
            "Subject": subject,
            "analysis": target.analysis,
            "phase": target.phase,
            "effect": standard_effect_key(target.effect),
            "target_label": target.label,
            "band": target_band(target),
            "cluster_index": int(cluster_index),
            "cluster_number": int(cluster_number),
            "cluster_p": p_value,
            "cluster_p_phasewise": float(cluster_p_corrected),
            "cluster_correction_multiplier": float(
                cluster_correction_multiplier_value
            ),
            "time_start_ms": time_start_ms,
            "time_end_ms": time_end_ms,
            "n_cluster_points": int(len(time_indices)),
            "n_unique_space": int(len(np.unique(space_indices))),
            "units": units,
            "meg_effect_raw": float(meg_raw[subject_i]),
            "meg_effect_signed": float(meg_signed[subject_i]),
        }
        for col, value in zip(CONDITION_LABELS, condition_means_raw[subject_i]):
            row[f"{col}_raw"] = float(value)
        for col, value in zip(CONDITION_LABELS, condition_means_signed[subject_i]):
            row[f"{col}_signed"] = float(value)
        rows.append(row)
    return rows


def extract_from_condition_data(
    target: ExtractionTarget,
    results: dict,
    args: argparse.Namespace,
) -> list[dict]:
    condition_data = np.asarray(results["condition_data"], dtype=float)
    t_obs = results.get("T_obs")
    correction_multiplier = cluster_correction_multiplier(target, args)
    rows = []
    for (
        cluster_index,
        cluster_number,
        _raw_p,
        corrected_p,
    ) in cluster_entries_to_extract(target, results, args):
        time_indices, space_indices = results["clusters"][cluster_index]
        time_indices = np.asarray(time_indices, dtype=int)
        space_indices = np.asarray(space_indices, dtype=int)
        signed_weights = point_weights_from_t(t_obs, time_indices, space_indices)
        raw_means = condition_values_from_array(
            condition_data, time_indices, space_indices
        )
        signed_means = condition_values_from_array(
            condition_data, time_indices, space_indices, signed_weights
        )
        rows.extend(
            make_rows(
                target=target,
                results=results,
                cluster_index=cluster_index,
                cluster_number=cluster_number,
                cluster_p_corrected=corrected_p,
                cluster_correction_multiplier_value=correction_multiplier,
                condition_means_raw=raw_means,
                condition_means_signed=signed_means,
                units="dB_or_power",
            )
        )
    return rows


def import_mne_or_skip() -> object | None:
    try:
        import mne  # noqa: F401

        return mne
    except Exception as error:
        print(
            "[skip] mne is not available in this Python environment. "
            f"Sensor/source extraction skipped. Details: {error}"
        )
        return None


def infer_phase_from_results_or_path(results: dict, path: Path) -> str:
    if results.get("phase"):
        return str(results["phase"])
    for phase in PHASE_CONDITIONS:
        if phase in path.parts:
            return phase
    raise ValueError(f"Cannot infer phase from {path}")


def extract_sensor(
    target: ExtractionTarget,
    results: dict,
    args: argparse.Namespace,
    code_root: Path,
) -> list[dict]:
    mne = import_mne_or_skip()
    if mne is None:
        return []

    sensor_module_dir = code_root / "2_ERF_sensor_level"
    if str(sensor_module_dir) not in sys.path:
        sys.path.insert(0, str(sensor_module_dir))

    from sensor_cluster_common import (  # noqa: PLC0415
        SensorAnalysisConfig,
        _build_condition_arrays,
        _equalize_and_build_adjacency,
        _read_complete_subjects,
    )

    phase = infer_phase_from_results_or_path(results, target.pkl_path)
    phase_cfg = SENSOR_PHASE_CONFIG[phase]
    config = SensorAnalysisConfig(
        phase=phase,
        data_phase=phase_cfg["data_phase"],
        phase_label=phase,
        conditions=tuple(results.get("conditions", PHASE_CONDITIONS[phase])),
        tmin=float(results.get("tmin", phase_cfg["tmin"])),
        tmax=float(results.get("tmax", phase_cfg["tmax"])),
    )

    print(f"[sensor] Reading evokeds for {phase} ...")
    subject_evokeds = _read_complete_subjects(config)
    saved_subjects = [f"{int(subject):02d}" for subject in results["valid_subjects"]]
    subject_evokeds = {
        subject: subject_evokeds[subject]
        for subject in saved_subjects
        if subject in subject_evokeds
    }
    valid_subjects = list(subject_evokeds)
    if valid_subjects != saved_subjects:
        raise RuntimeError(
            "Reloaded sensor subjects do not match saved cluster subjects. "
            f"Reloaded={valid_subjects[:5]}..., saved={saved_subjects[:5]}..."
        )

    _, channel_names, _ = _equalize_and_build_adjacency(
        subject_evokeds,
        config.conditions,
    )
    if [str(name) for name in channel_names] != [str(name) for name in results["channel_names"]]:
        raise RuntimeError("Reloaded sensor channel order does not match cluster pkl.")

    condition_arrays_dict, times = _build_condition_arrays(
        subject_evokeds,
        valid_subjects,
        config,
        float(results.get("tmin", config.tmin)),
        float(results.get("tmax", config.tmax)),
    )
    if not np.array_equal(np.asarray(times), np.asarray(results["times"])):
        raise RuntimeError("Reloaded sensor time axis does not match cluster pkl.")

    condition_data = np.stack(
        [condition_arrays_dict[str(condition)] for condition in config.conditions],
        axis=1,
    )
    # Convert from T to fT.
    condition_data = condition_data * 1e15

    rows = []
    t_obs = np.asarray(results["T_obs"])
    correction_multiplier = cluster_correction_multiplier(target, args)
    for (
        cluster_index,
        cluster_number,
        _raw_p,
        corrected_p,
    ) in cluster_entries_to_extract(target, results, args):
        time_indices, space_indices = results["clusters"][cluster_index]
        time_indices = np.asarray(time_indices, dtype=int)
        space_indices = np.asarray(space_indices, dtype=int)
        signed_weights = point_weights_from_t(t_obs, time_indices, space_indices)
        raw_means = condition_values_from_array(
            condition_data, time_indices, space_indices
        )
        signed_means = condition_values_from_array(
            condition_data, time_indices, space_indices, signed_weights
        )
        rows.extend(
            make_rows(
                target=target,
                results=results,
                cluster_index=cluster_index,
                cluster_number=cluster_number,
                cluster_p_corrected=corrected_p,
                cluster_correction_multiplier_value=correction_multiplier,
                condition_means_raw=raw_means,
                condition_means_signed=signed_means,
                units="fT",
            )
        )
    return rows


def source_stc_path(
    subject: str,
    condition: str,
    phase: str,
    morphed_dir: Path,
) -> Path:
    return source_stem(morphed_dir, subject, phase, condition)


def source_stc_files_exist(stem: Path) -> bool:
    """Support MNE's paired STC files saved as <stem>-lh.stc/-rh.stc."""
    return (
        stem.exists()
        or Path(f"{stem}-lh.stc").exists()
        and Path(f"{stem}-rh.stc").exists()
    )


def read_single_stc_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one binary MNE .stc hemisphere file without importing MNE.

    The morphed files in this project are saved as paired files named like
    <stem>-lh.stc and <stem>-rh.stc. MNE's binary STC layout is simple enough
    for this extraction: tmin/tstep in milliseconds, vertex ids, then a
    vertices x times float matrix.
    """
    with path.open("rb") as file:
        tmin_ms = np.fromfile(file, dtype=">f4", count=1)
        tstep_ms = np.fromfile(file, dtype=">f4", count=1)
        if tmin_ms.size != 1 or tstep_ms.size != 1:
            raise RuntimeError(f"Could not read STC timing header: {path}")

        n_vertices_header = np.fromfile(file, dtype=">u4", count=1)
        if n_vertices_header.size != 1:
            raise RuntimeError(f"Could not read STC vertex count: {path}")
        n_vertices = int(n_vertices_header[0])
        vertices = np.fromfile(file, dtype=">u4", count=n_vertices)
        if vertices.size != n_vertices:
            raise RuntimeError(f"Could not read all STC vertices: {path}")

        n_times_header = np.fromfile(file, dtype=">u4", count=1)
        if n_times_header.size != 1:
            raise RuntimeError(f"Could not read STC time count: {path}")
        n_times = int(n_times_header[0])

        data = np.fromfile(file, dtype=">f4", count=n_vertices * n_times)
        expected_size = n_vertices * n_times
        if data.size != expected_size:
            raise RuntimeError(
                f"Could not read complete STC data from {path}: "
                f"{data.size}/{expected_size}"
            )

    data = data.reshape((n_times, n_vertices)).T
    times = (float(tmin_ms[0]) + np.arange(n_times) * float(tstep_ms[0])) / 1000.0
    return data, times, vertices.astype(int)


def read_paired_stc_data(stem: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read left/right hemisphere STC pair and concatenate as MNE does."""
    lh_path = Path(f"{stem}-lh.stc")
    rh_path = Path(f"{stem}-rh.stc")
    if not lh_path.exists() or not rh_path.exists():
        raise FileNotFoundError(f"Missing paired STC files for stem: {stem}")

    lh_data, lh_times, _ = read_single_stc_file(lh_path)
    rh_data, rh_times, _ = read_single_stc_file(rh_path)
    if not np.allclose(lh_times, rh_times, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"Left/right STC time axes differ for {stem}")
    return np.vstack([lh_data, rh_data]), lh_times


def crop_source_data_to_expected_times(
    data: np.ndarray,
    times: np.ndarray,
    expected_times: np.ndarray,
    analysis_tmin: float,
    analysis_tmax: float,
    *,
    subject: str,
    condition: str,
) -> np.ndarray:
    keep = (times >= analysis_tmin - 1e-9) & (times <= analysis_tmax + 1e-9)
    cropped_data = data[:, keep]
    cropped_times = times[keep]
    if cropped_times.shape != expected_times.shape or not np.allclose(
        cropped_times,
        expected_times,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError(
            "Time mismatch after direct STC read for "
            f"sub-{subject}, condition {condition}. "
            f"Read range={cropped_times[:3]}...{cropped_times[-3:]}, "
            f"expected={expected_times[:3]}...{expected_times[-3:]}"
        )
    return cropped_data


def extract_source(
    target: ExtractionTarget,
    results: dict,
    args: argparse.Namespace,
    morphed_dir: Path,
) -> list[dict]:
    phase = infer_phase_from_results_or_path(results, target.pkl_path)
    conditions = tuple(results.get("conditions", PHASE_CONDITIONS[phase]))
    subjects = [f"{int(subject):02d}" for subject in results["valid_subjects"]]
    expected_times = np.asarray(results["times"], dtype=float)
    analysis_tmin = float(results["analysis_tmin"])
    analysis_tmax = float(results["analysis_tmax"])
    clusters_to_extract = cluster_entries_to_extract(target, results, args)
    correction_multiplier = cluster_correction_multiplier(target, args)
    rows = []

    print(f"[source] Reading paired STC files directly for {phase} ...")
    for (
        cluster_index,
        cluster_number,
        _raw_p,
        corrected_p,
    ) in clusters_to_extract:
        time_indices, space_indices = results["clusters"][cluster_index]
        time_indices = np.asarray(time_indices, dtype=int)
        space_indices = np.asarray(space_indices, dtype=int)
        signed_weights = point_weights_from_t(
            np.asarray(results.get("T_obs")),
            time_indices,
            space_indices,
        )

        raw_means = np.empty((len(subjects), len(conditions)), dtype=float)
        signed_means = np.empty_like(raw_means)

        for subject_i, subject in enumerate(subjects):
            for condition_i, condition in enumerate(conditions):
                path = source_stc_path(subject, str(condition), phase, morphed_dir)
                if not source_stc_files_exist(path):
                    raise FileNotFoundError(f"Missing source estimate: {path}")
                data, times = read_paired_stc_data(path)
                data = crop_source_data_to_expected_times(
                    data,
                    times,
                    expected_times,
                    analysis_tmin,
                    analysis_tmax,
                    subject=subject,
                    condition=str(condition),
                )
                point_values = data[space_indices, time_indices]
                raw_means[subject_i, condition_i] = float(np.mean(point_values))
                signed_means[subject_i, condition_i] = float(
                    np.mean(point_values * signed_weights)
                )

        rows.extend(
            make_rows(
                target=target,
                results=results,
                cluster_index=cluster_index,
                cluster_number=cluster_number,
                cluster_p_corrected=corrected_p,
                cluster_correction_multiplier_value=correction_multiplier,
                condition_means_raw=raw_means,
                condition_means_signed=signed_means,
                units="dSPM",
            )
        )
    return rows


def infer_analysis(path: Path, results_root: Path) -> str:
    try:
        rel = path.relative_to(results_root)
        parts = rel.parts
    except ValueError:
        parts = path.parts
    joined = "/".join(parts).lower()
    if "4_time_frequency" in joined:
        return "tfa_sensor"
    if "source_cortex_mask" in joined:
        return "source_cortex_mask"
    if "/sensor/" in f"/{joined}/" or "\\sensor\\" in str(path).lower():
        return "erf_sensor"
    return "unknown"


def infer_target(
    path: Path,
    results_root: Path,
    results: dict | None = None,
) -> ExtractionTarget:
    if results is None:
        results = load_pickle(path)
    analysis = infer_analysis(path, results_root)
    phase = str(results.get("phase") or "unknown")
    effect = parse_effect_from_path(path, results.get("effect"))
    if analysis == "tfa_sensor":
        rel = path.relative_to(results_root)
        # time_frequency/sensor/<phase>/<power>/<band>/<effect>/cluster_results.pkl
        label = "_".join(rel.parts[rel.parts.index("sensor") + 1:-1])
    else:
        label = safe_stem("_".join(path.parts[path.parts.index("source_cortex_mask"):-1]) if "source_cortex_mask" in path.parts else "_".join(path.relative_to(results_root).parts[:-1]))
    return ExtractionTarget(
        pkl_path=path,
        analysis=analysis,
        phase=phase,
        effect=effect,
        label=label,
    )


def planned_load_targets(results_root: Path) -> list[Path]:
    paths = [
        results_root / "3_ERF_source_level" / "results"
        / "source_cortex_mask"
        / "decision"
        / "load"
        / "Cluster_permutation_results_10000_seed42.pkl",
        results_root / "3_ERF_source_level" / "results"
        / "source_cortex_mask"
        / "transition"
        / "load"
        / "Cluster_permutation_results_10000_seed42.pkl",
    ]
    for phase in ("decision", "transition"):
        for band in ("theta", "alpha", "beta", "low_gamma"):
            paths.append(
                results_root / "4_time_frequency" / "time_frequency_analysis" / "results"
                / "sensor"
                / phase
                / "total"
                / band
                / "load"
                / "cluster_results.pkl"
            )
    return paths


def find_all_tfa_load(results_root: Path) -> list[Path]:
    root = results_root / "4_time_frequency" / "time_frequency_analysis" / "results" / "sensor"
    return sorted(root.glob("*/*/*/load/cluster_results.pkl"))


def find_all_effect(results_root: Path, effect: str) -> list[Path]:
    paths: list[Path] = []
    source_effect = {
        "load": "load",
        "magnitude": "magnitude",
        "interaction": "interaction",
    }[effect]
    paths.extend(
        (results_root / "3_ERF_source_level" / "results" / "source_cortex_mask").glob(
            f"*/{source_effect}/Cluster_permutation_results_10000_seed42.pkl"
        )
    )
    paths.extend(
        (results_root / "4_time_frequency" / "time_frequency_analysis" / "results" / "sensor").glob(
            f"*/*/*/{source_effect}/cluster_results.pkl"
        )
    )
    return sorted(paths)


def find_all_supported_effects(results_root: Path) -> list[Path]:
    paths: list[Path] = []
    for effect in ("load", "magnitude", "interaction"):
        paths.extend(find_all_effect(results_root, effect))
    return sorted(set(paths))


def normalize_phase_name(phase: str) -> str:
    phase = phase.strip().lower()
    return PHASE_ALIASES.get(phase, phase)


def phase_from_result_path(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    for phase in sorted(PHASE_CONDITIONS, key=len, reverse=True):
        if phase.lower() in parts:
            return phase
    return None


def filter_paths_by_phases(paths: list[Path], phases: Iterable[str]) -> list[Path]:
    allowed = {normalize_phase_name(phase) for phase in phases if str(phase).strip()}
    if not allowed:
        return paths
    return [path for path in paths if phase_from_result_path(path) in allowed]


def targets_from_args(args: argparse.Namespace) -> list[Path]:
    if args.single_pkl:
        paths = [Path(args.single_pkl)]
    elif args.preset == "planned-load":
        paths = planned_load_targets(args.results_root)
    elif args.preset == "all-tfa-load":
        paths = find_all_tfa_load(args.results_root)
    elif args.preset == "all-load":
        paths = find_all_effect(args.results_root, "load")
    elif args.preset == "all-magnitude":
        paths = find_all_effect(args.results_root, "magnitude")
    elif args.preset == "all-interaction":
        paths = find_all_effect(args.results_root, "interaction")
    elif args.preset == "all-effects":
        paths = find_all_supported_effects(args.results_root)
    elif args.preset == "core-effects":
        paths = filter_paths_by_phases(
            find_all_supported_effects(args.results_root),
            CORE_PHASES,
        )
    else:
        raise ValueError(f"Unknown preset: {args.preset}")

    if args.phases:
        phases = [phase for phase in re.split(r"[,;\s]+", args.phases) if phase]
        paths = filter_paths_by_phases(paths, phases)
    return filter_paths_by_phases(paths, CORE_PHASES)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cluster_inspection_rows(
    target: ExtractionTarget,
    results: dict,
    args: argparse.Namespace,
) -> list[dict]:
    rows = []
    times = np.asarray(results.get("times", []), dtype=float)
    correction_multiplier = cluster_correction_multiplier(target, args)
    for cluster_index, cluster_number, raw_p, corrected_p in cluster_entries_to_extract(
        target, results, args
    ):
        time_indices, space_indices = results["clusters"][cluster_index]
        time_indices = np.asarray(time_indices, dtype=int)
        space_indices = np.asarray(space_indices, dtype=int)
        row = {
            "pkl_path": str(target.pkl_path),
            "analysis": target.analysis,
            "phase": target.phase,
            "effect": standard_effect_key(target.effect),
            "band": target_band(target),
            "target_label": target.label,
            "cluster_number": int(cluster_number),
            "cluster_index": int(cluster_index),
            "cluster_p": float(raw_p),
            "cluster_p_phasewise": float(corrected_p),
            "cluster_correction_multiplier": float(correction_multiplier),
            "n_cluster_points": int(len(time_indices)),
            "n_unique_space": int(len(np.unique(space_indices))),
        }
        if times.size:
            row["time_start_ms"] = float(times[time_indices.min()] * 1000.0)
            row["time_end_ms"] = float(times[time_indices.max()] * 1000.0)
        else:
            row["time_start_ms"] = ""
            row["time_end_ms"] = ""
        rows.append(row)
    return rows


def extract_target(
    target: ExtractionTarget,
    args: argparse.Namespace,
    results: dict | None = None,
) -> list[dict]:
    print(f"\nExtracting: {target.pkl_path}")
    if results is None:
        results = load_pickle(target.pkl_path)
    target = ExtractionTarget(
        pkl_path=target.pkl_path,
        analysis=target.analysis,
        phase=str(results.get("phase", target.phase)),
        effect=parse_effect_from_path(target.pkl_path, results.get("effect")),
        label=target.label,
    )

    clusters = cluster_entries_to_extract(target, results, args)
    print(
        "  selected clusters: "
        f"{[(index, number, raw_p, corrected_p) for index, number, raw_p, corrected_p in clusters]}"
    )
    if not clusters:
        return []

    if results.get("condition_data") is not None:
        return extract_from_condition_data(target, results, args)

    if target.analysis == "erf_sensor":
        print("[skip] ERF sensor extraction is disabled for this regression workflow.")
        return []

    if target.analysis == "source_cortex_mask":
        return extract_source(
            target,
            results,
            args,
            args.source_morphed_dir,
        )

    print(f"[skip] Unsupported target without condition_data: {target.pkl_path}")
    return []


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract subject-level MEG cluster effects."
    )
    parser.add_argument(
        "--results-root",
        type=project_input,
        default=RESULTS_ROOT_DEFAULT,
        help="Root MEG results directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help="Directory for extracted CSV files.",
    )
    parser.add_argument(
        "--code-root",
        type=project_input,
        default=CODE_ROOT_DEFAULT,
        help="Code root containing ERF_sensor_level helper scripts.",
    )
    parser.add_argument(
        "--source-morphed-dir",
        type=project_input,
        default=DATA_ROOT / "MEG" / "source" / "morphed_stcs",
        help="Directory containing morphed source estimates.",
    )
    parser.add_argument(
        "--preset",
        choices=(
            "planned-load",
            "all-tfa-load",
            "all-load",
            "all-magnitude",
            "all-interaction",
            "all-effects",
            "core-effects",
        ),
        default="core-effects",
        help="Which group of result files to extract.",
    )
    parser.add_argument(
        "--phases",
        default="",
        help=(
            "Optional phase filter, separated by commas/spaces. "
            "Choose incentive, decision, transition or feedback."
        ),
    )
    parser.add_argument(
        "--single-pkl",
        type=project_input,
        default=None,
        help="Extract one specific cluster result pkl instead of a preset.",
    )
    parser.add_argument(
        "--cluster-alpha",
        type=float,
        default=0.05,
        help="Corrected cluster p threshold.",
    )
    parser.add_argument(
        "--cluster-correction",
        choices=("phasewise", "raw"),
        default="phasewise",
        help=(
            "Cluster filter for regression extraction. phasewise uses source "
            "p x 3 and TFA p x 12 by default; raw reproduces the earlier "
            "single-test cluster p filtering."
        ),
    )
    parser.add_argument(
        "--source-phasewise-multiplier",
        type=float,
        default=3.0,
        help="Bonferroni multiplier for source clusters within each phase.",
    )
    parser.add_argument(
        "--tfa-phasewise-multiplier",
        type=float,
        default=12.0,
        help=(
            "Bonferroni multiplier for TFA clusters within each phase. The "
            "saved TFA reports label this as Phase-wise Bonferroni p (x12)."
        ),
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only write the corrected cluster selection table; do not extract subject effects.",
    )
    parser.add_argument(
        "--include-all-clusters",
        action="store_true",
        help="Extract/check every raw cluster regardless of corrected p.",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=50000,
        help="Python recursion limit for reading large saved MNE/pickle objects.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"01_extract_meg_cluster_subject_effects.py version: {SCRIPT_VERSION}")
    if sys.getrecursionlimit() < args.recursion_limit:
        sys.setrecursionlimit(args.recursion_limit)

    candidate_paths = targets_from_args(args)
    missing = [path for path in candidate_paths if not path.exists()]
    for path in missing:
        print(f"[missing] {path}")
    paths = [path for path in candidate_paths if path.exists()]
    if not paths:
        raise FileNotFoundError("No cluster result files were found for extraction.")

    all_rows: list[dict] = []
    inspection_rows: list[dict] = []
    manifest_rows: list[dict] = []
    for path in paths:
        try:
            fallback_label = safe_stem("_".join(path.relative_to(args.results_root).parts[:-1]))
        except ValueError:
            fallback_label = safe_stem("_".join(path.parts[:-1]))
        target = ExtractionTarget(
            pkl_path=path,
            analysis=infer_analysis(path, args.results_root),
            phase="unknown",
            effect=parse_effect_from_path(path, None),
            label=fallback_label,
        )
        try:
            results = load_pickle(path)
            target = infer_target(path, args.results_root, results)
            selected = cluster_inspection_rows(target, results, args)
            inspection_rows.extend(selected)
            if args.inspect_only:
                rows = []
                print(
                    f"\nInspecting: {path}\n"
                    f"  selected clusters: "
                    f"{[(row['cluster_index'], row['cluster_number'], row['cluster_p'], row['cluster_p_phasewise']) for row in selected]}"
                )
            else:
                rows = extract_target(target, args, results)
        except Exception as error:  # keep batch extraction moving
            print(f"[error] Failed to extract {path}: {error}")
            error_traceback = traceback.format_exc()
            print(error_traceback)
            manifest_rows.append(
                {
                    "pkl_path": str(path),
                    "analysis": target.analysis,
                    "phase": target.phase,
                    "effect": standard_effect_key(target.effect),
                    "label": target.label,
                    "status": "error",
                    "n_rows": 0,
                    "n_selected_clusters": 0,
                    "cluster_correction": args.cluster_correction,
                    "message": repr(error),
                    "traceback": error_traceback,
                }
            )
            continue

        if args.inspect_only:
            status = "inspect_only_selected" if selected else "inspect_only_no_clusters"
            message = ""
        elif rows:
            out_path = args.output_dir / f"{safe_stem(target.label)}_subject_effects.csv"
            write_csv(out_path, rows)
            print(f"  wrote {len(rows)} rows: {out_path}")
            all_rows.extend(rows)
            status = "ok"
            message = str(out_path)
        else:
            status = "no_clusters_or_skipped"
            message = ""

        manifest_rows.append(
            {
                "pkl_path": str(path),
                "analysis": target.analysis,
                "phase": target.phase,
                "effect": standard_effect_key(target.effect),
                "label": target.label,
                "status": status,
                "n_rows": len(rows),
                "n_selected_clusters": len(selected),
                "cluster_correction": args.cluster_correction,
                "message": message,
                "traceback": "",
            }
        )

    write_csv(args.output_dir / "phasewise_cluster_selection.csv", inspection_rows)
    write_csv(args.output_dir / "meg_cluster_subject_effects_all.csv", all_rows)
    write_csv(args.output_dir / "extraction_manifest.csv", manifest_rows)
    print(f"\nDone. Output directory: {args.output_dir}")
    print(f"Selected clusters listed: {len(inspection_rows)}")
    print(f"Total extracted rows: {len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
