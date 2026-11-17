"""Likelihood and simulator matching the MEG 52-subject HBRL Stan model."""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(
    os.environ.get(
        "HBRL52_DATA_DIR",
        Path(__file__).resolve().parents[4] / "data" / "BHV" / "2_stage_task" / "preprocessed" / "learning" / "all" / "HBRL_code",
    )
)
EXPECTED_SUBJECTS = 52

PARAMS = [
    "alpha_lload",
    "alpha_hload",
    "beta_mf_small_lload",
    "beta_mf_large_lload",
    "beta_mf_small_hload",
    "beta_mf_large_hload",
    "beta_mb_small_lload",
    "beta_mb_large_lload",
    "beta_mb_small_hload",
    "beta_mb_large_hload",
    "st_lload",
    "st_hload",
    "respst_lload",
    "respst_hload",
]

MCMC_PARAM_GROUPS = [
    ("low", "Learning rate", "alpha_lload"),
    ("low", "MF weight small", "beta_mf_small_lload"),
    ("low", "MF weight large", "beta_mf_large_lload"),
    ("low", "MB weight small", "beta_mb_small_lload"),
    ("low", "MB weight large", "beta_mb_large_lload"),
    ("low", "Choice stickiness", "st_lload"),
    ("low", "Response stickiness", "respst_lload"),
    ("high", "Learning rate", "alpha_hload"),
    ("high", "MF weight small", "beta_mf_small_hload"),
    ("high", "MF weight large", "beta_mf_large_hload"),
    ("high", "MB weight small", "beta_mb_small_hload"),
    ("high", "MB weight large", "beta_mb_large_hload"),
    ("high", "Choice stickiness", "st_hload"),
    ("high", "Response stickiness", "respst_hload"),
]


def load_data():
    files = sorted(DATA_DIR.glob("*.csv"))
    if len(files) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} subject CSVs in {DATA_DIR}, found {len(files)}"
        )
    datasets = []
    seen_ids = set()
    for subject_index, path in enumerate(files, start=1):
        raw = pd.read_csv(path)
        required = {
            "participant", "probe", "pairs", "animal", "choice", "points", "magnitude",
            "load", "part", "stimuli_left", "stimuli_right", "rews_1", "rews_2", "rews_3", "rews_4",
            "trials", "trials_globe", "block", "correct", "miss",
        }
        missing_columns = sorted(required.difference(raw.columns))
        if missing_columns:
            raise ValueError(f"{path.name} is missing required columns: {missing_columns}")
        raw = raw[raw["probe"] != 1].reset_index(drop=True)
        if raw.empty:
            raise ValueError(f"No usable trials after removing probe trials in {path.name}")
        if raw[list(required)].isna().any().any():
            raise ValueError(
                f"Missing required values remain after removing probe trials in {path.name}; "
                "the Stan fitting script does not drop these rows."
            )
        participant_ids = pd.to_numeric(raw["participant"], errors="raise").astype(int).unique()
        if len(participant_ids) != 1:
            raise ValueError(f"Expected one participant ID in {path.name}, found {participant_ids}")
        sid = int(participant_ids[0])
        filename_match = re.match(r"^(\d+)_", path.name)
        if filename_match and int(filename_match.group(1)) != sid:
            raise ValueError(f"Filename ID and participant column disagree in {path.name}")
        if sid in seen_ids:
            raise ValueError(f"Duplicate participant ID {sid} in input folder")
        seen_ids.add(sid)
        if not set(raw["load"]).issubset({"low", "high"}) or not set(raw["magnitude"]).issubset({"small", "large"}):
            raise ValueError(f"Unknown load/magnitude labels in {path.name}")
        if set(raw["load"]) != {"low", "high"}:
            raise ValueError(f"Both loads are required in {path.name}")
        for col, lower, upper in [("pairs", 1, 6), ("animal", 1, 6), ("choice", 1, 12), ("part", 0, 2)]:
            vals = raw[col].to_numpy(dtype=float)
            if not np.all(np.isfinite(vals) & (vals == np.floor(vals)) & (vals >= lower) & (vals <= upper)):
                raise ValueError(f"Invalid {col} coding in {path.name}")
        if not ((raw.choice == raw.stimuli_left) | (raw.choice == raw.stimuli_right)).all():
            raise ValueError(f"Chosen stimulus is not displayed in {path.name}")
        if not ((raw.stimuli_left % 2) != (raw.stimuli_right % 2)).all():
            raise ValueError(f"Displayed stimuli must represent different actions in {path.name}")
        n = len(raw)
        z = np.zeros((n, 15), dtype=np.float64)
        # s1, s2, action, reward, magnitude-large, load-high, part, left-action,
        # reward local outcome 0, reward local outcome 1, trial, block, correct, miss
        z[:, 0] = raw["pairs"].to_numpy() - 1
        z[:, 1] = raw["animal"].to_numpy() - 1
        z[:, 2] = np.where(raw["choice"].to_numpy() % 2 != 0, 0, 1)
        z[:, 3] = raw["points"].to_numpy()
        z[:, 4] = (raw["magnitude"].to_numpy() == "large").astype(float)
        z[:, 5] = (raw["load"].to_numpy() == "high").astype(float)
        z[:, 6] = raw["part"].to_numpy()
        z[:, 7] = np.where(raw["stimuli_left"].to_numpy() % 2 != 0, 0, 1)
        rews = raw[["rews_1", "rews_2", "rews_3", "rews_4"]].to_numpy(dtype=float)
        s1 = z[:, 0].astype(int)
        for t in range(n):
            if s1[t] <= 3:
                z[t, 8] = rews[t, 0]
                z[t, 9] = rews[t, 1]
            else:
                z[t, 8] = rews[t, 2]
                z[t, 9] = rews[t, 3]
        z[:, 10] = raw["trials"].to_numpy()
        z[:, 11] = raw["block"].to_numpy()
        z[:, 12] = raw["correct"].to_numpy()
        z[:, 13] = raw["miss"].to_numpy()
        z[:, 14] = raw["trials_globe"].to_numpy()  # original adjacency, including gaps from excluded trials
        base = np.where(z[:, 0] <= 1, 0, np.where(z[:, 0] <= 3, 2, 4))
        mapped_outcome = base + np.where(z[:, 6] == 2, 1 - z[:, 2], z[:, 2])
        if not np.array_equal(mapped_outcome, z[:, 1]):
            raise ValueError(f"Action-to-animal mapping does not match the Stan transitions in {path.name}")
        expected = np.empty(n)
        for t in range(n):
            local_outcome = int(z[t, 1]) % 2
            expected[t] = z[t, 8 + local_outcome]
        if not np.allclose(expected, z[:, 3]):
            bad = int(np.where(np.abs(expected - z[:, 3]) > 1e-9)[0][0])
            raise ValueError(f"Reward mapping failed for {path.name} trial {bad + 1}")
        datasets.append((subject_index, sid, z, path.name))
    return datasets


@njit(cache=True, nogil=True)
def _select_params(x, load_high, magnitude_large):
    alpha = x[1] if load_high == 1 else x[0]
    st = x[11] if load_high == 1 else x[10]
    respst = x[13] if load_high == 1 else x[12]
    if load_high == 0 and magnitude_large == 0:
        beta_mf = x[2]
        beta_mb = x[6]
    elif load_high == 0 and magnitude_large == 1:
        beta_mf = x[3]
        beta_mb = x[7]
    elif load_high == 1 and magnitude_large == 0:
        beta_mf = x[4]
        beta_mb = x[8]
    else:
        beta_mf = x[5]
        beta_mb = x[9]
    return alpha, beta_mf, beta_mb, st, respst


@njit(cache=True, nogil=True)
def recursion(x, data, simulate=False, uniforms=np.zeros(1)):
    q2 = np.full(6, 4.5)
    qmf = np.full((6, 2), 4.5)
    m = np.zeros((6, 2))
    k = np.zeros(2)
    out = data.copy()
    probs = np.zeros(len(data))
    ll = 0.0
    previous_load = -1
    for t in range(len(data)):
        s1 = int(data[t, 0])
        load_high = int(data[t, 5])
        if t == 0 or load_high != previous_load:
            q2[:] = 4.5
            qmf[:, :] = 4.5
            m[:, :] = 0.0
            k[:] = 0.0
        previous_load = load_high

        magnitude_large = int(data[t, 4])
        part = int(data[t, 6])
        left_action = int(data[t, 7])
        alpha, beta_mf, beta_mb, st, respst = _select_params(x, load_high, magnitude_large)

        base = 0 if s1 <= 1 else (2 if s1 <= 3 else 4)
        if part == 0 or part == 1:
            qmb0 = q2[base]
            qmb1 = q2[base + 1]
        else:
            qmb0 = q2[base + 1]
            qmb1 = q2[base]

        if left_action == 1:
            k[0], k[1] = k[1], k[0]

        q0 = beta_mf * qmf[s1, 0] + beta_mb * qmb0 + st * m[s1, 0] + respst * k[0]
        q1 = beta_mf * qmf[s1, 1] + beta_mb * qmb1 + st * m[s1, 1] + respst * k[1]
        v = q1 - q0
        p = 1.0 / (1.0 + np.exp(-v)) if v >= 0 else np.exp(v) / (1.0 + np.exp(v))
        probs[t] = p
        action = int(uniforms[t] < p) if simulate else int(data[t, 2])
        local_outcome = action if (part == 0 or part == 1) else 1 - action
        outcome = base + local_outcome
        reward = data[t, 8 + local_outcome] if simulate else data[t, 3]
        ll += action * v - (max(v, 0.0) + np.log1p(np.exp(-abs(v))))

        m[s1, :] = 0.0
        m[s1, action] = 1.0
        k[:] = 0.0
        if action == left_action:
            k[0] = 1.0
        else:
            k[1] = 1.0

        delta1 = q2[outcome] - qmf[s1, action]
        delta2 = reward - q2[outcome]
        qmf[s1, action] += alpha * delta1 + alpha * delta2
        q2[outcome] += alpha * delta2

        out[t, 1] = outcome
        out[t, 2] = action
        out[t, 3] = reward
    return ll, out, probs
