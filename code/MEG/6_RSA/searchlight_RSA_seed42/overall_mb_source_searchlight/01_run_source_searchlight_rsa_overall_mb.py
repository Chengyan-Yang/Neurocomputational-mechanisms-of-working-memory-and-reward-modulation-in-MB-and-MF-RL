"""
Source-space searchlight RSA between MEG ERF patterns and overall HBRL MB RDMs.

This is an independent companion analysis to the previous cluster-based
follow-up RSA. It does not use significant MEG clusters as ROIs. Instead, it
slides a local source-space neighborhood across the whole cortex mask and tests
whether the local four-condition MEG representational geometry matches the
subject-specific HBRL four-condition model-based-weight RDM.

Default first-pass analysis:
  - ERF source only
  - four core phases: incentive, decision, transition, feedback
  - each phase uses the same analysis time window as the previous source ERF
    analysis, read from the corresponding source_cortex_mask result pkl
  - time is averaged within the phase window before local spatial RDMs are built
  - source searchlight is a 1-hop neighborhood on fsaverage ico5 cortex mask
  - subject-level RSA uses Spearman correlation across the six RDM distances
  - group inference uses one-sample t-tests on Fisher-z RSA values, plus FDR
    and optional spatial cluster permutation.

Run in Spyder:
  Open this file and click Run.

Fast smoke test in Spyder console:
  import os
  os.environ["SEARCHLIGHT_RSA_MAX_SUBJECTS"] = "3"
  os.environ["SEARCHLIGHT_RSA_MAX_VERTICES"] = "200"
  os.environ["SEARCHLIGHT_RSA_N_PERMUTATIONS"] = "0"
  python 01_run_source_searchlight_rsa_overall_mb.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from meg_project import DATA_ROOT, behavior_rdm, project_input, MEG_CODE, RSA_RESULTS, source_stem

import importlib
import json
import os
import pickle
import re
from dataclasses import dataclass
from itertools import combinations

# Some Windows Anaconda/MNE environments with NumPy 2.x can require a deeper
# import stack when SciPy imports NumPy compatibility modules.
if sys.getrecursionlimit() < 50000:
    sys.setrecursionlimit(50000)

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.stats import rankdata

try:
    import mne
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "This source searchlight RSA script requires MNE-Python because it "
        "uses the fsaverage ico5 source-space adjacency and spatial cluster "
        "permutation. Please run it in the same Python/Spyder environment that "
        "you used for the ERF source analysis scripts."
    ) from error


# =============================================================================
# Configuration
# =============================================================================

RESULTS_ROOT = MEG_CODE

OUTPUT_DIR = Path(
    os.environ.get(
        "SEARCHLIGHT_RSA_OUT",
        str(RSA_RESULTS / "searchlight_seed42" / "overall_mb_source_searchlight"),
    )
)

BEHAVIOR_RDM_PATH = project_input(
    os.environ.get(
        "SEARCHLIGHT_BEHAVIOR_RDM_PATH",
        behavior_rdm("mb"),
    )
)

SOURCE_MORPHED_DIR = project_input(
    os.environ.get(
        "SEARCHLIGHT_SOURCE_MORPHED_DIR",
        f"{DATA_ROOT}/MEG/source/morphed_stcs",
    )
)

SUBJECTS_DIR = project_input(
    os.environ.get(
        "SEARCHLIGHT_SUBJECTS_DIR",
        f"{DATA_ROOT}/MRI/surf",
    )
)

TEMPLATE_SUBJECT = "fsaverage"
TEMPLATE_SPACING = "ico5"

PHASE_SPECS = {
    "incentive": {
        "data_phase": "incentive",
        "conditions": ("1", "2", "3", "4"),
        "reference_effect": "magnitude",
    },
    "decision": {
        "data_phase": "decision",
        "conditions": ("11", "21", "31", "41"),
        "reference_effect": "load",
    },
    "transition": {
        "data_phase": "transition",
        "conditions": ("13", "23", "33", "43"),
        "reference_effect": "load",
    },
    "feedback": {
        "data_phase": "feedback",
        "conditions": ("51", "52", "53", "54"),
        "reference_effect": "magnitude",
    },
}

CONDITION_KEYS = (
    "small_magnitude_low_load",
    "large_magnitude_low_load",
    "small_magnitude_high_load",
    "large_magnitude_high_load",
)

RDM_PAIRS = tuple(
    f"{condition_a}__vs__{condition_b}"
    for condition_a, condition_b in combinations(CONDITION_KEYS, 2)
)

RSA_METHOD = os.environ.get("SEARCHLIGHT_RSA_METHOD", "spearman").lower()
NEIGHBORHOOD_HOPS = int(os.environ.get("SEARCHLIGHT_RSA_NEIGHBORHOOD_HOPS", "1"))
TIME_MODE = os.environ.get("SEARCHLIGHT_RSA_TIME_MODE", "mean").lower()
N_PERMUTATIONS = int(os.environ.get("SEARCHLIGHT_RSA_N_PERMUTATIONS", "5000"))
CLUSTER_FORMING_P = float(os.environ.get("SEARCHLIGHT_RSA_CLUSTER_FORMING_P", "0.01"))
CLUSTER_ALPHA = float(os.environ.get("SEARCHLIGHT_RSA_CLUSTER_ALPHA", "0.05"))
RANDOM_SEED = 42
N_JOBS = int(os.environ.get("SEARCHLIGHT_RSA_N_JOBS", "1"))

# Optional smoke-test limits. Leave empty/0 for the full analysis.
MAX_SUBJECTS = int(os.environ.get("SEARCHLIGHT_RSA_MAX_SUBJECTS", "0"))
MAX_VERTICES = int(os.environ.get("SEARCHLIGHT_RSA_MAX_VERTICES", "0"))


# =============================================================================
# Utilities
# =============================================================================


@dataclass(frozen=True)
class PhaseReference:
    phase: str
    data_phase: str
    conditions: tuple[str, str, str, str]
    pkl_path: Path
    analysis_tmin: float
    analysis_tmax: float
    times: np.ndarray
    valid_subjects: tuple[str, ...]
    valid_indices: np.ndarray
    full_vertices: list[np.ndarray]


def parse_subid(value) -> int | float:
    text = str(value).strip().removeprefix("sub-")
    if text.endswith(".0"):
        text = text[:-2]
    return int(text) if text.isdecimal() else np.nan


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, str(default)).lower()
    return value in {"1", "true", "yes", "y"}


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


def load_pickle(path: Path) -> dict:
    if sys.getrecursionlimit() < 50000:
        sys.setrecursionlimit(50000)
    try:
        with path.open("rb") as file:
            return pickle.load(file)
    except ModuleNotFoundError as error:
        if not str(error).startswith("No module named 'numpy._core"):
            raise
        install_numpy_pickle_compat()
        with path.open("rb") as file:
            return pickle.load(file)


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


def stc_stem(subject: str, condition: str, data_phase: str) -> Path:
    return source_stem(SOURCE_MORPHED_DIR, subject, data_phase, condition)


def reference_pkl_path(phase: str, reference_effect: str) -> Path:
    return (
        RESULTS_ROOT / "3_ERF_source_level" / "results"
        / "source_cortex_mask"
        / phase
        / reference_effect
        / "Cluster_permutation_results_10000_seed42.pkl"
    )


def load_phase_reference(phase: str, spec: dict) -> PhaseReference:
    pkl_path = reference_pkl_path(phase, spec["reference_effect"])
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing source reference pkl for {phase}: {pkl_path}")
    results = load_pickle(pkl_path)
    return PhaseReference(
        phase=phase,
        data_phase=spec["data_phase"],
        conditions=tuple(results.get("conditions", spec["conditions"])),
        pkl_path=pkl_path,
        analysis_tmin=float(results["analysis_tmin"]),
        analysis_tmax=float(results["analysis_tmax"]),
        times=np.asarray(results["times"], dtype=float),
        valid_subjects=tuple(f"{int(subject):02d}" for subject in results["valid_subjects"]),
        valid_indices=np.asarray(results["cortex_valid_indices"], dtype=int),
        full_vertices=[np.asarray(vertices, dtype=int) for vertices in results["full_vertices"]],
    )


def build_cortex_adjacency(reference: PhaseReference) -> sparse.csr_matrix:
    src = mne.setup_source_space(
        TEMPLATE_SUBJECT,
        spacing=TEMPLATE_SPACING,
        subjects_dir=str(SUBJECTS_DIR),
        add_dist=False,
        verbose=False,
    )
    for hemi in (0, 1):
        if not np.array_equal(src[hemi]["vertno"], reference.full_vertices[hemi]):
            raise RuntimeError("Reference STCs do not match fsaverage ico5 source space.")
    adjacency_full = sparse.csr_matrix(mne.spatial_src_adjacency(src))
    adjacency = adjacency_full[reference.valid_indices, :][:, reference.valid_indices].tocsr()
    return adjacency


def build_neighborhoods(adjacency: sparse.csr_matrix, hops: int, center_indices: np.ndarray) -> list[np.ndarray]:
    neighborhoods: list[np.ndarray] = []
    for center in center_indices:
        reached = {int(center)}
        frontier = {int(center)}
        for _ in range(hops):
            new_frontier = set()
            for vertex in frontier:
                new_frontier.update(int(x) for x in adjacency.getrow(vertex).indices)
            new_frontier.difference_update(reached)
            reached.update(new_frontier)
            frontier = new_frontier
        neighborhoods.append(np.asarray(sorted(reached), dtype=int))
    return neighborhoods


def load_behavior_rdm(path: Path) -> dict[int, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing behavior RDM file: {path}")
    behavior = pd.read_csv(path)
    required = {"subid", "rdm_pair", "rdm_distance"}
    missing = required.difference(behavior.columns)
    if missing:
        raise ValueError(f"Behavior RDM file is missing columns: {sorted(missing)}")

    behavior["subid"] = behavior["subid"].map(parse_subid)
    vectors: dict[int, np.ndarray] = {}
    for subid, df_subject in behavior.groupby("subid"):
        pair_map = dict(zip(df_subject["rdm_pair"], df_subject["rdm_distance"]))
        if not all(pair in pair_map for pair in RDM_PAIRS):
            continue
        vectors[int(subid)] = np.asarray([pair_map[pair] for pair in RDM_PAIRS], dtype=float)
    return vectors


def crop_and_average(data: np.ndarray, times: np.ndarray, expected_times: np.ndarray) -> np.ndarray:
    if times.shape != expected_times.shape or not np.allclose(times, expected_times, rtol=0.0, atol=1e-9):
        keep = (times >= expected_times[0] - 1e-9) & (times <= expected_times[-1] + 1e-9)
        data = data[:, keep]
        times = times[keep]
    if times.shape != expected_times.shape or not np.allclose(times, expected_times, rtol=0.0, atol=1e-9):
        raise RuntimeError(
            f"STC time axis mismatch: read {times.shape}, expected {expected_times.shape}"
        )
    if TIME_MODE != "mean":
        raise ValueError("Only SEARCHLIGHT_RSA_TIME_MODE='mean' is implemented in this first-pass script.")
    return np.nanmean(data, axis=1)


def condition_vertex_values(subject: str, reference: PhaseReference) -> np.ndarray:
    condition_rows = []
    for condition in reference.conditions:
        data, times = read_paired_stc(stc_stem(subject, condition, reference.data_phase))
        averaged = crop_and_average(data, times, reference.times)
        condition_rows.append(averaged[reference.valid_indices])
    return np.asarray(condition_rows, dtype=float)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def local_meg_rdm(condition_values: np.ndarray, neighborhood: np.ndarray) -> np.ndarray:
    distances = []
    local = condition_values[:, neighborhood]
    for i, j in combinations(range(local.shape[0]), 2):
        r = pearson_corr(local[i], local[j])
        distances.append(np.nan if np.isnan(r) else 1.0 - r)
    return np.asarray(distances, dtype=float)


def rsa_similarity(meg_rdm: np.ndarray, behavior_rdm: np.ndarray) -> float:
    ok = np.isfinite(meg_rdm) & np.isfinite(behavior_rdm)
    if ok.sum() < 3:
        return np.nan
    x = meg_rdm[ok]
    y = behavior_rdm[ok]
    if RSA_METHOD == "spearman":
        x = rankdata(x, method="average")
        y = rankdata(y, method="average")
    elif RSA_METHOD != "pearson":
        raise ValueError("SEARCHLIGHT_RSA_METHOD must be 'spearman' or 'pearson'.")
    return pearson_corr(x, y)


def fisher_z(rho: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(rho, -0.999999, 0.999999))


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    p_ok = p[ok]
    order = np.argsort(p_ok)
    ranked = p_ok[order]
    n = len(ranked)
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out_ok = np.empty_like(adjusted)
    out_ok[order] = adjusted
    out[ok] = out_ok
    return out


def vertex_table(reference: PhaseReference, center_indices: np.ndarray) -> pd.DataFrame:
    n_lh = len(reference.full_vertices[0])
    rows = []
    for valid_space_index in center_indices:
        full_index = int(reference.valid_indices[int(valid_space_index)])
        if full_index < n_lh:
            hemi = "lh"
            vertex = int(reference.full_vertices[0][full_index])
        else:
            hemi = "rh"
            vertex = int(reference.full_vertices[1][full_index - n_lh])
        rows.append(
            {
                "valid_space_index": int(valid_space_index),
                "full_space_index": full_index,
                "hemi": hemi,
                "vertex": vertex,
            }
        )
    return pd.DataFrame(rows)


def run_cluster_permutation(z_values: np.ndarray, adjacency: sparse.csr_matrix, phase_dir: Path) -> pd.DataFrame:
    if N_PERMUTATIONS <= 0:
        return pd.DataFrame()
    df = z_values.shape[0] - 1
    threshold = stats.t.ppf(1.0 - CLUSTER_FORMING_P / 2.0, df)
    print(
        f"  cluster permutation: n_perm={N_PERMUTATIONS}, "
        f"|t|>={threshold:.3f}, n_jobs={N_JOBS}"
    )
    t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
        z_values,
        threshold=threshold,
        tail=0,
        adjacency=adjacency,
        n_permutations=N_PERMUTATIONS,
        out_type="indices",
        seed=RANDOM_SEED,
        n_jobs=N_JOBS,
        verbose=True,
    )
    with (phase_dir / "cluster_permutation_results.pkl").open("wb") as file:
        pickle.dump(
            {
                "t_obs": t_obs,
                "clusters": clusters,
                "cluster_p_values": cluster_p_values,
                "h0": h0,
                "cluster_forming_p": CLUSTER_FORMING_P,
                "threshold": threshold,
                "n_permutations": N_PERMUTATIONS,
                "seed": RANDOM_SEED,
            },
            file,
        )

    rows = []
    for idx, cluster in enumerate(clusters):
        if isinstance(cluster, tuple):
            space_inds = np.asarray(cluster[0], dtype=int)
        else:
            space_inds = np.asarray(cluster, dtype=int)
        if space_inds.size == 0:
            continue
        peak_local = int(space_inds[np.argmax(np.abs(t_obs[space_inds]))])
        rows.append(
            {
                "cluster_index": idx,
                "cluster_p": float(cluster_p_values[idx]),
                "is_significant": bool(cluster_p_values[idx] < CLUSTER_ALPHA),
                "n_vertices": int(space_inds.size),
                "peak_local_index": peak_local,
                "peak_t": float(t_obs[peak_local]),
            }
        )
    return pd.DataFrame(rows)


def run_phase(reference: PhaseReference, behavior_vectors: dict[int, np.ndarray]) -> dict:
    phase_dir = OUTPUT_DIR / reference.phase

    adjacency = build_cortex_adjacency(reference)
    n_vertices = adjacency.shape[0]
    center_indices = np.arange(n_vertices, dtype=int)
    if MAX_VERTICES > 0:
        center_indices = center_indices[:MAX_VERTICES]
        adjacency_for_stats = adjacency[center_indices, :][:, center_indices].tocsr()
    else:
        adjacency_for_stats = adjacency

    neighborhoods = build_neighborhoods(adjacency, NEIGHBORHOOD_HOPS, center_indices)
    neighborhood_sizes = np.asarray([len(neigh) for neigh in neighborhoods], dtype=int)

    subjects = [subject for subject in reference.valid_subjects if int(parse_subid(subject)) in behavior_vectors]
    if MAX_SUBJECTS > 0:
        subjects = subjects[:MAX_SUBJECTS]
    if len(subjects) < 3:
        raise RuntimeError(f"Too few matched subjects for {reference.phase}: {len(subjects)}")

    print(f"\nPhase: {reference.phase}")
    print(f"  data phase: {reference.data_phase}")
    print(f"  time window: {reference.analysis_tmin:.3f}-{reference.analysis_tmax:.3f} s")
    print(f"  subjects: {len(subjects)}")
    print(f"  center vertices: {len(center_indices)}")
    print(f"  neighborhood hops: {NEIGHBORHOOD_HOPS}")
    print(f"  neighborhood size: min={neighborhood_sizes.min()}, median={np.median(neighborhood_sizes):.1f}, max={neighborhood_sizes.max()}")

    subject_rsa = np.full((len(subjects), len(center_indices)), np.nan, dtype=np.float32)

    for subject_i, subject in enumerate(subjects, start=1):
        subid = int(parse_subid(subject))
        behavior_rdm = behavior_vectors[subid]
        condition_values = condition_vertex_values(subject, reference)
        for center_i, neighborhood in enumerate(neighborhoods):
            meg_rdm = local_meg_rdm(condition_values, neighborhood)
            subject_rsa[subject_i - 1, center_i] = rsa_similarity(meg_rdm, behavior_rdm)
        print(f"  subject {subject_i:02d}/{len(subjects)} done: sub-{subject}")

    z_values = fisher_z(subject_rsa.astype(float))
    t_values, p_values = stats.ttest_1samp(z_values, popmean=0.0, axis=0, nan_policy="omit")
    mean_rho = np.nanmean(subject_rsa, axis=0)
    median_rho = np.nanmedian(subject_rsa, axis=0)
    mean_z = np.nanmean(z_values, axis=0)
    p_fdr = fdr_bh(p_values)

    vertices = vertex_table(reference, center_indices)
    summary = vertices.assign(
        phase=reference.phase,
        data_phase=reference.data_phase,
        analysis_tmin=reference.analysis_tmin,
        analysis_tmax=reference.analysis_tmax,
        n_subjects=len(subjects),
        neighborhood_hops=NEIGHBORHOOD_HOPS,
        neighborhood_size=neighborhood_sizes,
        rsa_method=RSA_METHOD,
        time_mode=TIME_MODE,
        mean_rho=mean_rho,
        median_rho=median_rho,
        mean_fisher_z=mean_z,
        t_value=t_values,
        p_value=p_values,
        p_fdr_bh=p_fdr,
    )
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(phase_dir / "source_searchlight_rsa_vertex_summary.csv", index=False, encoding="utf-8-sig")

    np.save(phase_dir / "source_searchlight_subject_rsa.npy", subject_rsa)
    np.savez_compressed(
        phase_dir / "source_searchlight_group_maps.npz",
        center_indices=center_indices,
        valid_indices=reference.valid_indices,
        full_vertices_lh=reference.full_vertices[0],
        full_vertices_rh=reference.full_vertices[1],
        subjects=np.asarray(subjects),
        mean_rho=mean_rho,
        median_rho=median_rho,
        mean_fisher_z=mean_z,
        t_value=t_values,
        p_value=p_values,
        p_fdr_bh=p_fdr,
        neighborhood_size=neighborhood_sizes,
    )

    cluster_summary = run_cluster_permutation(z_values, adjacency_for_stats, phase_dir)
    if not cluster_summary.empty:
        cluster_summary = cluster_summary.merge(
            vertices.reset_index().rename(columns={"index": "peak_local_index"}),
            on="peak_local_index",
            how="left",
        )
        cluster_summary.to_csv(
            phase_dir / "source_searchlight_cluster_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    top_vertices = summary.sort_values("p_value").head(200)
    top_vertices.to_csv(phase_dir / "source_searchlight_top_vertices.csv", index=False, encoding="utf-8-sig")

    return {
        "phase": reference.phase,
        "n_subjects": len(subjects),
        "n_vertices": len(center_indices),
        "time_window": f"{reference.analysis_tmin}-{reference.analysis_tmax}",
        "min_p": float(np.nanmin(p_values)),
        "n_fdr_vertices": int(np.nansum(p_fdr < 0.05)),
        "n_cluster_significant": int((cluster_summary["cluster_p"] < CLUSTER_ALPHA).sum()) if not cluster_summary.empty else 0,
        "phase_dir": str(phase_dir),
    }


def main() -> None:
    if RSA_METHOD not in {"spearman", "pearson"}:
        raise ValueError("SEARCHLIGHT_RSA_METHOD must be 'spearman' or 'pearson'.")
    if NEIGHBORHOOD_HOPS < 0:
        raise ValueError("SEARCHLIGHT_RSA_NEIGHBORHOOD_HOPS must be >= 0.")

    behavior_vectors = load_behavior_rdm(BEHAVIOR_RDM_PATH)

    print("Source searchlight RSA: overall HBRL MB RDM")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Behavior RDM: {BEHAVIOR_RDM_PATH}")
    print(f"Source morphed dir: {SOURCE_MORPHED_DIR}")
    print(f"Subjects dir: {SUBJECTS_DIR}")
    print(f"RSA method: {RSA_METHOD}")
    print(f"Time mode: {TIME_MODE}")
    print(f"Permutations: {N_PERMUTATIONS}")
    if MAX_SUBJECTS > 0 or MAX_VERTICES > 0:
        print(f"SMOKE TEST LIMITS: max_subjects={MAX_SUBJECTS}, max_vertices={MAX_VERTICES}")

    phase_rows = []
    for phase, spec in PHASE_SPECS.items():
        reference = load_phase_reference(phase, spec)
        phase_rows.append(run_phase(reference, behavior_vectors))

    manifest = pd.DataFrame(phase_rows)
    manifest.to_csv(OUTPUT_DIR / "source_searchlight_rsa_manifest.csv", index=False, encoding="utf-8-sig")
    config = {
        "analysis": "source_searchlight_rsa_overall_hbrl_mb",
        "behavior_rdm_path": str(BEHAVIOR_RDM_PATH),
        "source_morphed_dir": str(SOURCE_MORPHED_DIR),
        "subjects_dir": str(SUBJECTS_DIR),
        "template_subject": TEMPLATE_SUBJECT,
        "template_spacing": TEMPLATE_SPACING,
        "rsa_method": RSA_METHOD,
        "time_mode": TIME_MODE,
        "neighborhood_hops": NEIGHBORHOOD_HOPS,
        "n_permutations": N_PERMUTATIONS,
        "cluster_forming_p": CLUSTER_FORMING_P,
        "cluster_alpha": CLUSTER_ALPHA,
        "random_seed": RANDOM_SEED,
        "max_subjects": MAX_SUBJECTS,
        "max_vertices": MAX_VERTICES,
        "phase_specs": PHASE_SPECS,
        "note": (
            "This searchlight analysis does not use significant MEG clusters "
            "as ROIs. It tests local source-space MEG RDM alignment with "
            "subject-specific HBRL four-condition model-based-weight RDMs."
        ),
    }
    (OUTPUT_DIR / "source_searchlight_rsa_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\nSearchlight RSA complete.")
    print(manifest.to_string(index=False))
    print(f"Manifest: {OUTPUT_DIR / 'source_searchlight_rsa_manifest.csv'}")


if __name__ == "__main__":
    main()
