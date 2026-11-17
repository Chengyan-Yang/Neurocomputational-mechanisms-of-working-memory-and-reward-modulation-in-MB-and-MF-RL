"""Compare seven RL models by load and export the participant-mean fit statistics."""

from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "model_comparison"
CSV_PATH = OUT / "MEG_RL_model_comparison.csv"

MODELS = {
    1: ["beta", "alpha", "lambda", "w_small", "w_large", "st", "respst"],
    2: ["beta", "alpha_small", "alpha_large", "lambda", "w", "st", "respst"],
    3: ["beta_small", "beta_large", "alpha", "lambda", "w", "st", "respst"],
    4: ["beta_small", "beta_large", "alpha_small", "alpha_large", "lambda", "w", "st", "respst"],
    5: ["beta_small", "beta_large", "alpha", "lambda", "w_small", "w_large", "st", "respst"],
    6: ["beta", "alpha_small", "alpha_large", "lambda", "w_small", "w_large", "st", "respst"],
    7: ["beta_small", "beta_large", "alpha_small", "alpha_large", "lambda", "w_small", "w_large", "st", "respst"],
}

SYMBOLS = {
    "beta": ("β", None),
    "alpha": ("α", None),
    "lambda": ("λ", None),
    "w": ("w", None),
    "st": ("π", None),
    "respst": ("ρ", None),
    "beta_small": ("β", "small"),
    "beta_large": ("β", "large"),
    "alpha_small": ("α", "small"),
    "alpha_large": ("α", "large"),
    "w_small": ("w", "small"),
    "w_large": ("w", "large"),
}


def parameter_plain(params):
    pieces = []
    for param in params:
        symbol, sub = SYMBOLS[param]
        pieces.append(symbol if sub is None else f"{symbol}_{sub}")
    return ", ".join(pieces)


def load_and_validate():
    records = []
    reference = {}
    for load in ("low", "high"):
        for model, params in MODELS.items():
            path = BASE / f"model_{model}" / f"{load}_load" / f"results_M{model}_{load}.csv"
            frame = pd.read_csv(path).sort_values("id").reset_index(drop=True)
            if len(frame) != 52:
                raise ValueError(f"Expected 52 participants in {path}, found {len(frame)}")
            ids_trials = frame[["id", "trials"]].copy()
            if load not in reference:
                reference[load] = ids_trials
            elif not ids_trials.equals(reference[load]):
                raise ValueError(f"Participant IDs or trial counts differ in {path}")
            k = len(params)
            expected_aic = -2 * frame["loglik"] + 2 * k
            expected_bic = -2 * frame["loglik"] + k * np.log(frame["trials"])
            expected_r2 = 1 - frame["loglik"] / frame["LL_random"]
            if not np.allclose(frame["aic"], expected_aic, atol=1e-8):
                raise ValueError(f"AIC formula check failed for {path}")
            if not np.allclose(frame["bic"], expected_bic, atol=1e-8):
                raise ValueError(f"BIC formula check failed for {path}")
            if not np.allclose(frame["Pseudo_R2"], expected_r2, atol=1e-8):
                raise ValueError(f"Pseudo R2 formula check failed for {path}")
            records.append({
                "load": load,
                "model": model,
                "parameters": parameter_plain(params),
                "K": k,
                "AIC": float(frame["aic"].mean()),
                "BIC": float(frame["bic"].mean()),
                "Pseudo_R2": float(frame["Pseudo_R2"].mean()),
                "n_subjects": len(frame),
            })
    return pd.DataFrame(records)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = load_and_validate()
    summary.to_csv(CSV_PATH, index=False, float_format="%.9f", encoding="utf-8-sig")
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
