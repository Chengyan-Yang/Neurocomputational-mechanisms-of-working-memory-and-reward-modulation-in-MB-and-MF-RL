"""Exact model_1 likelihood, forward simulator and adaptive Metropolis sampler.

Data are zero indexed locally. No block resets, exactly as the MATLAB source.
The same recursion drives likelihood and simulation to avoid update mismatches.
"""
import numpy as np
import pandas as pd
from numba import njit
from scipy.special import logit
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(__file__).resolve().parents[5] / "data" / "BHV" / "2_stage_task" / "preprocessed" / "learning"
DATA_FOLDERS = {
    "low": ("low_load",),
    "high": ("high_load",),
}
EXPECTED_SUBJECTS = 52
PARAMS = ['beta', 'alpha', 'lambda', 'w_small', 'w_large', 'st', 'respst']


def _subject_data_dir(load):
    if load not in DATA_FOLDERS:
        raise ValueError(f"Unknown load {load!r}; expected 'low' or 'high'.")
    candidates = [DATA_ROOT / name for name in DATA_FOLDERS[load]]
    for path in candidates:
        if path.is_dir():
            return path
    names = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No {load}-load 52-subject data folder found. Checked: {names}")


def _numeric_column(frame, name, source):
    try:
        return pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Column {name!r} in {source} is not numeric.") from exc


def load_data(load):
    folder = _subject_data_dir(load)
    suffix = "*_RL_lload.csv" if load == "low" else "*_RL_hload.csv"
    files = sorted(folder.glob(suffix))
    if len(files) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} {load}-load CSV files in {folder}, found {len(files)}."
        )

    required = {
        "trials", "trials_globe", "participant", "pairs", "stimuli_left", "choice", "animal",
        "rews_1", "rews_2", "points", "magnitude", "load", "part", "probe",
    }
    if load == "high":
        required.update({"rews_3", "rews_4"})

    result = []
    seen_ids = set()
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")

        # Match the preprocessing used to build the MATLAB groupdata: remove
        # rows containing any missing value, then remove probe trials.
        frame = frame.dropna(axis=0, how="any")
        frame = frame.loc[pd.to_numeric(frame["probe"], errors="coerce") != 1].copy()
        frame = frame.loc[pd.to_numeric(frame["trials"], errors="coerce").notna()].copy()
        if len(frame) < 130:
            raise ValueError(f"{path.name} has only {len(frame)} usable trials; expected at least 130.")

        subject_values = _numeric_column(frame, "participant", path.name)
        unique_subjects = np.unique(subject_values.astype(np.int64))
        if len(unique_subjects) != 1:
            raise ValueError(f"{path.name} contains multiple participant IDs: {unique_subjects.tolist()}")
        sid = int(unique_subjects[0])
        if sid in seen_ids:
            raise ValueError(f"Duplicate participant ID {sid} in {folder}.")
        seen_ids.add(sid)

        observed_loads = set(frame["load"].astype(str).str.lower())
        if observed_loads != {load}:
            raise ValueError(f"{path.name} has load values {sorted(observed_loads)}, expected {load!r}.")

        n = len(frame)
        if load == "low":
            ctx = np.zeros(n, dtype=np.int64)
        else:
            part = _numeric_column(frame, "part", path.name).astype(np.int64)
            if not np.isin(part, [1, 2]).all():
                raise ValueError(f"{path.name} has unexpected high-load part codes: {np.unique(part).tolist()}")
            ctx = part - 1

        pair = _numeric_column(frame, "pairs", path.name).astype(np.int64)
        left_stimulus = _numeric_column(frame, "stimuli_left", path.name).astype(np.int64)
        choice = _numeric_column(frame, "choice", path.name).astype(np.int64)
        animal = _numeric_column(frame, "animal", path.name).astype(np.int64)
        points = _numeric_column(frame, "points", path.name)
        if np.any(choice < 1):
            raise ValueError(f"{path.name} contains a missing/invalid response code in choice.")
        magnitude = frame["magnitude"].astype(str).str.lower().to_numpy()
        if not np.isin(magnitude, ["small", "large"]).all():
            raise ValueError(f"{path.name} contains unexpected magnitude labels: {np.unique(magnitude).tolist()}")

        rews_1 = _numeric_column(frame, "rews_1", path.name)
        rews_2 = _numeric_column(frame, "rews_2", path.name)
        if load == "low":
            reward0, reward1 = rews_1, rews_2
        else:
            rews_3 = _numeric_column(frame, "rews_3", path.name)
            rews_4 = _numeric_column(frame, "rews_4", path.name)
            reward0 = np.where(ctx == 0, rews_1, rews_3)
            reward1 = np.where(ctx == 0, rews_2, rews_4)

        # ctx, local pair, magnitude, left action, action, outcome, reward, rews0, rews1, original trial ID
        z = np.zeros((n, 10), dtype=np.float64)
        z[:, 0] = ctx
        z[:, 1] = (pair - 1) % 2
        z[:, 2] = magnitude == "large"
        z[:, 3] = (left_stimulus - 1) % 2
        z[:, 4] = (choice - 1) % 2
        z[:, 5] = (animal - 1) % 2
        z[:, 6] = points
        z[:, 7] = reward0
        z[:, 8] = reward1
        z[:, 9] = _numeric_column(frame, "trials_globe", path.name)

        if not np.array_equal(z[:, 5].astype(np.int64), z[:, 4].astype(np.int64) ^ ctx):
            raise ValueError(f"{path.name} violates model_1's deterministic choice-to-outcome mapping.")
        expected_points = z[np.arange(n), 7 + z[:, 5].astype(np.int64)]
        if not np.allclose(z[:, 6], expected_points, rtol=0.0, atol=1e-10):
            raise ValueError(f"{path.name} has points inconsistent with its selected outcome reward.")
        result.append((sid, z, path.name))

    if len(seen_ids) != EXPECTED_SUBJECTS:
        raise ValueError(f"Expected {EXPECTED_SUBJECTS} unique {load}-load participants, found {len(seen_ids)}.")
    return result


def validate_subject_alignment(load, datasets, fitted_ids):
    data_ids = {int(sid) for sid, _, _ in datasets}
    fitted_id_list = [int(sid) for sid in fitted_ids]
    fitted_ids = set(fitted_id_list)
    if len(data_ids) != EXPECTED_SUBJECTS:
        raise ValueError(f"{load} data contain {len(data_ids)} subjects; expected {EXPECTED_SUBJECTS}.")
    if len(fitted_id_list) != EXPECTED_SUBJECTS or len(fitted_ids) != len(fitted_id_list):
        raise ValueError(
            f"{load} MAP results contain {len(fitted_id_list)} rows and {len(fitted_ids)} unique IDs; "
            f"expected {EXPECTED_SUBJECTS} unique subjects."
        )
    if fitted_ids != data_ids:
        missing_data = sorted(fitted_ids - data_ids)
        missing_fit = sorted(data_ids - fitted_ids)
        raise ValueError(
            f"{load} fitted/data subject IDs do not match. "
            f"Fit-only IDs: {missing_data}; data-only IDs: {missing_fit}."
        )

@njit(cache=True, nogil=True)
def recursion(x, data, simulate=False, uniforms=np.zeros(1)):
    qmf = np.full((2, 2, 2), 4.5)
    q2 = np.full((2, 2), 4.5)
    m = np.zeros((2, 2, 2))
    r = np.zeros((2, 2))
    out = data.copy()
    probs = np.zeros(len(data))
    ll = 0.0
    for t in range(len(data)):
        c, s, magnitude, left = int(data[t,0]), int(data[t,1]), int(data[t,2]), int(data[t,3])
        if left == 1:
            r[c,0], r[c,1] = r[c,1], r[c,0]
        w = x[3+magnitude]
        qa = w*q2[c, c] + (1-w)*qmf[c,s,0] + x[5]*m[c,s,0]+x[6]*r[c,0]
        qb = w*q2[c, 1-c] + (1-w)*qmf[c,s,1] + x[5]*m[c,s,1]+x[6]*r[c,1]
        v = x[0]*(qb-qa)
        p = 1/(1+np.exp(-v)) if v >= 0 else np.exp(v)/(1+np.exp(v))
        probs[t] = p
        a = int(uniforms[t] < p) if simulate else int(data[t,4])
        outcome = (a ^ c) if simulate else int(data[t,5])
        reward = data[t,7+outcome] if simulate else data[t,6]
        ll += a*v - (max(v, 0.) + np.log1p(np.exp(-abs(v))))
        m[c,:,:] = 0.
        m[c,s,a] = 1.
        r[c,:] = 0.
        r[c,int(a != left)] = 1.
        dt1 = q2[c,outcome]-qmf[c,s,a]
        dt2 = reward-q2[c,outcome]
        qmf[c,s,a] += x[1]*dt1 + x[2]*x[1]*dt2
        q2[c,outcome] += x[1]*dt2
        out[t,4], out[t,5], out[t,6] = a, outcome, reward
    return ll, out, probs

@njit(cache=True, nogil=True)
def logprior(x):
    if x[0] <= 0 or x[0] >= 20 or abs(x[5]) >= 20 or abs(x[6]) >= 20:
        return -np.inf
    lp = 3.8*np.log(x[0])-x[0]/.88 - .5*(x[5]**2+x[6]**2)
    for k in range(1,5):
        if x[k] <= 0 or x[k] >= 1:
            return -np.inf
        lp += np.log(x[k])+np.log1p(-x[k])
    return lp

@njit(cache=True, nogil=True)
def from_z(z):
    x = z.copy()
    x[0] = np.exp(z[0])
    for k in range(1,5):
        x[k] = 1/(1+np.exp(-z[k]))
    return x

def to_z(x):
    z = np.array(x, dtype=float).copy()
    z[0] = np.log(z[0])
    z[1:5] = logit(np.clip(z[1:5],1e-9,1-1e-9))
    return z

@njit(cache=True, nogil=True)
def target(z, data, jacobian=True):
    x = from_z(z)
    lp = logprior(x)
    if not np.isfinite(lp):
        return -np.inf
    if jacobian:
        lp += z[0]
        for k in range(1,5):
            lp += np.log(x[k])+np.log1p(-x[k])
    return lp+recursion(x,data)[0]

@njit(cache=True, nogil=True)
def sample_chain(data, initial, seed, warmup, draws):
    np.random.seed(seed)
    z = initial.copy()
    lp = target(z,data)
    hist = np.empty((warmup,7))
    samples = np.empty((draws,7))
    lps = np.empty(draws)
    cov = np.eye(7)*.08
    chol = np.linalg.cholesky(cov)
    logscale = np.log(2.38/np.sqrt(7.))
    accepted, batch = 0, 0
    for i in range(warmup+draws):
        # Both kernels are symmetric. Adaptation ends before retained samples.
        if np.random.rand() < .05:
            proposal = z + .12*np.random.randn(7)
        else:
            proposal = z + np.exp(logscale)*(chol @ np.random.randn(7))
        newlp = target(proposal,data)
        accept = np.log(np.random.rand()) < newlp-lp
        if accept:
            z, lp = proposal, newlp
        if i < warmup:
            hist[i] = z
            batch += int(accept)
            if (i+1) % 100 == 0:
                logscale += min(.5, 4/np.sqrt((i+1)/100.))*(batch/100.-.234)
                batch = 0
                if i >= 499:
                    h = hist[100:i+1]
                    mean = np.zeros(7)
                    for j in range(len(h)):
                        mean += h[j]
                    mean /= len(h)
                    cov = np.zeros((7,7))
                    for j in range(len(h)):
                        delta = h[j]-mean
                        cov += np.outer(delta,delta)
                    cov = cov/(len(h)-1)+np.eye(7)*1e-5
                    chol = np.linalg.cholesky(cov)
        else:
            samples[i-warmup] = from_z(z)
            lps[i-warmup] = lp
            accepted += int(accept)
    return samples,lps,accepted/draws

def prior_draw(rng):
    x = np.r_[rng.gamma(4.8,.88),rng.beta(2,2,4),rng.normal(0,1,2)]
    while x[0] >= 20:
        x[0] = rng.gamma(4.8,.88)
    return x

def objective(z,data):
    return -target(z,data,False)
