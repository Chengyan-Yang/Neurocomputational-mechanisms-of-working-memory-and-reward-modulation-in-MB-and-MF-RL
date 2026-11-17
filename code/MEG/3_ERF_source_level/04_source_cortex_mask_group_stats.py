"""Run configurable cortex-masked group statistics on standard-inverse STCs."""

from collections import Counter
import json
import pickle

import mne
import numpy as np
import pandas as pd
from scipy import sparse, stats
from tqdm import tqdm

from cortex_mask_analysis_config import (
    ANALYSIS_TMAX,
    ANALYSIS_TMIN,
    CLUSTER_ALPHA,
    CLUSTER_FORMING_P,
    EFFECT_LABELS,
    EFFECT_WEIGHTS,
    N_JOBS,
    N_PERMUTATIONS,
    BUFFER_SIZE,
    PHASE,
    RANDOM_SEED,
    SUBJECTS,
    SUBJECTS_DIR,
    TEMPLATE_SPACING,
    TEMPLATE_SUBJECT,
    conditions,
    output_dir,
    result_path,
    stc_stem,
)


UNKNOWN_REGION = "Unknown/Medial Wall"
INVALID_LABELS = {"unknown", "medial_wall", "??", "none", "corpuscallosum"}


def same_vertices(first, second):
    return all(
        np.array_equal(first.vertices[hemi], second.vertices[hemi])
        for hemi in (0, 1)
    )


def stc_files_exist(stem):
    """Support the paired STC format used by the original source pipeline."""
    return (
        stem.exists()
        or type(stem)(f"{stem}-lh.stc").exists()
        and type(stem)(f"{stem}-rh.stc").exists()
    )


def load_standard_inverse_contrasts(effect):
    """Read original morphed STCs and form one subject-level contrast."""
    condition_codes = conditions()
    weights = np.asarray(EFFECT_WEIGHTS[effect], dtype=float)
    contrasts = []
    valid_subjects = []
    skipped = []
    template_stc = None

    for subject in tqdm(
        SUBJECTS,
        desc=f"Loading standard-inverse {PHASE} {effect} STCs",
    ):
        stems = [stc_stem(subject, condition) for condition in condition_codes]
        missing = [
            condition
            for condition, stem in zip(condition_codes, stems)
            if not stc_files_exist(stem)
        ]
        if missing:
            skipped.append(
                {"Subject": subject, "Reason": f"missing conditions {missing}"}
            )
            continue

        try:
            stcs = [mne.read_source_estimate(str(stem)) for stem in stems]
            for stc in stcs:
                stc.crop(
                    ANALYSIS_TMIN,
                    ANALYSIS_TMAX,
                    include_tmax=True,
                )

            reference = stcs[0]
            if any(not same_vertices(reference, stc) for stc in stcs[1:]):
                raise RuntimeError("Condition source spaces do not match.")
            if any(
                not np.array_equal(reference.times, stc.times)
                for stc in stcs[1:]
            ):
                raise RuntimeError("Condition time axes do not match.")

            contrast = sum(
                weight * stc.data for weight, stc in zip(weights, stcs)
            )
            contrasts.append(contrast.T)
            valid_subjects.append(subject)
            if template_stc is None:
                template_stc = reference.copy()
        except Exception as error:
            skipped.append({"Subject": subject, "Reason": repr(error)})

    if not contrasts:
        print("\nNo subjects passed STC loading. First recorded reasons:")
        for item in skipped[:10]:
            print(f"  sub-{item['Subject']}: {item['Reason']}")
        raise RuntimeError("No complete subjects were available for statistics.")

    X_full = np.asarray(contrasts)
    if X_full.ndim != 3:
        raise RuntimeError(f"Expected a 3-D group array, got {X_full.shape}.")
    return X_full, valid_subjects, skipped, template_stc


def build_cortex_geometry(template_stc):
    """Build an fsaverage ico5 cortex mask and matching adjacency."""
    src = mne.setup_source_space(
        TEMPLATE_SUBJECT,
        spacing=TEMPLATE_SPACING,
        subjects_dir=str(SUBJECTS_DIR),
        add_dist=False,
        verbose=False,
    )
    for hemi in (0, 1):
        if not np.array_equal(src[hemi]["vertno"], template_stc.vertices[hemi]):
            raise RuntimeError(
                "The original morphed STCs do not match fsaverage ico5."
            )

    label_dir = SUBJECTS_DIR / TEMPLATE_SUBJECT / "label"
    cortex_labels = [
        mne.read_label(
            str(label_dir / f"{hemi}.cortex.label"),
            subject=TEMPLATE_SUBJECT,
            verbose=False,
        )
        for hemi in ("lh", "rh")
    ]

    valid_parts = [
        np.isin(source["vertno"], label.vertices)
        for source, label in zip(src, cortex_labels)
    ]
    valid_mask = np.concatenate(valid_parts)
    valid_indices = np.flatnonzero(valid_mask)

    adjacency_full = sparse.csr_matrix(mne.spatial_src_adjacency(src))
    adjacency = adjacency_full[valid_indices, :][:, valid_indices].tocsr()
    if adjacency.shape != (len(valid_indices), len(valid_indices)):
        raise RuntimeError("Masked adjacency has an unexpected shape.")

    print(
        f"Cortex mask retained {len(valid_indices)}/{len(valid_mask)} vertices "
        f"({100.0 * len(valid_indices) / len(valid_mask):.1f}%)."
    )
    return src, valid_mask, valid_indices, adjacency


def build_vertex_label_lookup():
    labels = mne.read_labels_from_annot(
        TEMPLATE_SUBJECT,
        parc="aparc",
        subjects_dir=str(SUBJECTS_DIR),
        verbose=False,
    )
    lookup = {}
    for label in labels:
        clean_name = label.name.replace("-lh", "").replace("-rh", "")
        if clean_name.lower() in INVALID_LABELS:
            continue
        for vertex in label.vertices:
            lookup[(label.hemi, int(vertex))] = clean_name
    return lookup


def make_report(
    T_obs,
    clusters,
    cluster_p_values,
    good_indices,
    template_stc,
    valid_indices,
):
    columns = [
        "Cluster #",
        "Original Cluster Index",
        "Time Window (ms)",
        "Region (Peak)",
        "Region (Spatial Majority)",
        "Spatial Majority (%)",
        "Spatial Unknown (%)",
        "Hemi",
        "Peak Time (ms)",
        "Peak T-value",
        "MNI (X, Y, Z)",
        "Size (Vertices)",
        "Size (Spatiotemporal Points)",
        "Touches Analysis Boundary",
        "p-value",
    ]
    lookup = build_vertex_label_lookup()
    n_lh = len(template_stc.vertices[0])

    def real_vertex(masked_space_index):
        full_index = int(valid_indices[int(masked_space_index)])
        if full_index < n_lh:
            return "lh", 0, int(template_stc.vertices[0][full_index])
        rh_index = full_index - n_lh
        return "rh", 1, int(template_stc.vertices[1][rh_index])

    rows = []
    for report_number, cluster_index in enumerate(good_indices, start=1):
        time_inds, space_inds = clusters[cluster_index]
        time_inds = np.asarray(time_inds, dtype=int)
        space_inds = np.asarray(space_inds, dtype=int)
        values = T_obs[time_inds, space_inds]
        peak_position = int(np.argmax(np.abs(values)))
        peak_time_index = int(time_inds[peak_position])
        peak_space_index = int(space_inds[peak_position])
        peak_t = float(values[peak_position])

        hemi, hemi_index, peak_vertex = real_vertex(peak_space_index)
        peak_region = lookup.get((hemi, peak_vertex), UNKNOWN_REGION)
        mni = np.squeeze(
            mne.vertex_to_mni(
                peak_vertex,
                hemi_index,
                subject=TEMPLATE_SUBJECT,
                subjects_dir=str(SUBJECTS_DIR),
            )
        )

        unique_space = np.unique(space_inds)
        regions = []
        for space_index in unique_space:
            region_hemi, _, vertex = real_vertex(space_index)
            regions.append(lookup.get((region_hemi, vertex), UNKNOWN_REGION))
        counts = Counter(regions)
        majority_region, majority_count = counts.most_common(1)[0]
        majority_percent = 100.0 * majority_count / len(regions)
        unknown_percent = 100.0 * counts.get(UNKNOWN_REGION, 0) / len(regions)

        p_value = float(cluster_p_values[cluster_index])
        rows.append(
            {
                "Cluster #": report_number,
                "Original Cluster Index": int(cluster_index),
                "Time Window (ms)": (
                    f"{template_stc.times[time_inds.min()] * 1000:.0f} - "
                    f"{template_stc.times[time_inds.max()] * 1000:.0f}"
                ),
                "Region (Peak)": peak_region,
                "Region (Spatial Majority)": majority_region,
                "Spatial Majority (%)": f"{majority_percent:.1f}",
                "Spatial Unknown (%)": f"{unknown_percent:.1f}",
                "Hemi": "L" if hemi == "lh" else "R",
                "Peak Time (ms)": (
                    f"{template_stc.times[peak_time_index] * 1000:.0f}"
                ),
                "Peak T-value": f"{peak_t:.2f}",
                "MNI (X, Y, Z)": ", ".join(f"{value:.1f}" for value in mni),
                "Size (Vertices)": len(unique_space),
                "Size (Spatiotemporal Points)": len(space_inds),
                "Touches Analysis Boundary": bool(
                    time_inds.min() == 0
                    or time_inds.max() == len(template_stc.times) - 1
                ),
                "p-value": (
                    "<0.001" if p_value < 0.001 else f"{p_value:.3f}"
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def run_effect(effect):
    save_dir = output_dir(effect)

    X_full, valid_subjects, skipped, template_stc = (
        load_standard_inverse_contrasts(effect)
    )
    src, valid_mask, valid_indices, adjacency = build_cortex_geometry(
        template_stc
    )
    X = X_full[:, :, valid_indices]
    if X.shape[2] != adjacency.shape[0]:
        raise RuntimeError("Masked data and adjacency dimensions do not match.")

    degrees_of_freedom = len(valid_subjects) - 1
    t_threshold = stats.t.ppf(
        1.0 - CLUSTER_FORMING_P / 2.0,
        degrees_of_freedom,
    )
    print(f"Valid subjects: {len(valid_subjects)}")
    print(f"Skipped subjects: {len(skipped)}")
    print(f"Masked group data shape: {X.shape}")
    print(
        f"Cluster-forming threshold: |t| >= {t_threshold:.3f} "
        f"(two-tailed p < {CLUSTER_FORMING_P})"
    )

    T_obs, clusters, cluster_p_values, H0 = (
        mne.stats.spatio_temporal_cluster_1samp_test(
            X,
            adjacency=adjacency,
            threshold=t_threshold,
            tail=0,
            n_permutations=N_PERMUTATIONS,
            n_jobs=N_JOBS,
            buffer_size=BUFFER_SIZE,
            seed=RANDOM_SEED,
            out_type="indices",
            verbose=True,
        )
    )
    good_indices = np.where(cluster_p_values < CLUSTER_ALPHA)[0]
    print(f"Significant clusters: {len(good_indices)}")

    n_full_vertices = len(valid_mask)
    T_obs_full = np.zeros((T_obs.shape[0], n_full_vertices), dtype=T_obs.dtype)
    T_obs_full[:, valid_indices] = T_obs
    clusters_full = [
        (
            np.asarray(time_inds, dtype=int),
            valid_indices[np.asarray(space_inds, dtype=int)],
        )
        for time_inds, space_inds in clusters
    ]

    result_file = result_path(effect)
    save_dir.mkdir(parents=True, exist_ok=True)
    with result_file.open("wb") as file:
        pickle.dump(
            {
                "T_obs_cortex": T_obs,
                "clusters_cortex": clusters,
                "cortex_valid_mask": valid_mask,
                "cortex_valid_indices": valid_indices,
                "T_obs": T_obs_full,
                "clusters": clusters_full,
                "cluster_p_values": cluster_p_values,
                "H0": H0,
                "good_cluster_inds": good_indices,
                "times": template_stc.times.copy(),
                "full_vertices": [
                    source["vertno"].copy() for source in src
                ],
                "valid_subjects": valid_subjects,
                "skipped_subjects": skipped,
                "phase": PHASE,
                "conditions": conditions(),
                "effect": effect,
                "effect_label": EFFECT_LABELS[effect],
                "effect_weights": EFFECT_WEIGHTS[effect],
                "analysis_tmin": ANALYSIS_TMIN,
                "analysis_tmax": ANALYSIS_TMAX,
                "cluster_forming_p": CLUSTER_FORMING_P,
                "t_threshold": t_threshold,
                "tail": 0,
                "n_permutations": N_PERMUTATIONS,
                "seed": RANDOM_SEED,
                "source_model": (
                    "standard surface inverse; fsaverage cortex mask applied "
                    "only at group statistics"
                ),
            },
            file,
        )

    report = make_report(
        T_obs,
        clusters,
        cluster_p_values,
        good_indices,
        template_stc,
        valid_indices,
    )
    report_file = save_dir / f"{PHASE}_{effect}_cortex_mask_report.csv"
    report.to_csv(report_file, index=False)
    pd.DataFrame({"Subject": valid_subjects}).to_csv(
        save_dir / "valid_subjects.csv", index=False
    )
    (save_dir / "skipped_subjects.json").write_text(
        json.dumps(skipped, indent=2), encoding="utf-8"
    )
    config_record = {
        "phase": PHASE,
        "effect": effect,
        "effect_label": EFFECT_LABELS[effect],
        "conditions": conditions(),
        "effect_weights": EFFECT_WEIGHTS[effect],
        "analysis_tmin": ANALYSIS_TMIN,
        "analysis_tmax": ANALYSIS_TMAX,
        "n_subjects": len(valid_subjects),
        "n_cortex_vertices": len(valid_indices),
        "cluster_forming_p": CLUSTER_FORMING_P,
        "cluster_alpha": CLUSTER_ALPHA,
        "n_permutations": N_PERMUTATIONS,
        "seed": RANDOM_SEED,
    }
    (save_dir / "analysis_config.json").write_text(
        json.dumps(config_record, indent=2), encoding="utf-8"
    )

    print(report.to_string(index=False) if len(report) else "No significant clusters.")
    print(f"\nPKL: {result_file}")
    print(f"Report: {report_file}")


def main():
    for effect in EFFECT_WEIGHTS:
        print(f"Running {PHASE}: {effect}")
        run_effect(effect)


if __name__ == "__main__":
    main()
