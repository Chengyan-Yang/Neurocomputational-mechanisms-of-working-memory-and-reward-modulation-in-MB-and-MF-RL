"""Reference-style full-parameter MCMC diagnostics for the MEG 52-subject HBRL fit.

This mirrors the scope of rstan::monitor(fit) by evaluating every retained flat
parameter in the saved stanfit object. It uses draws exported by
export_all_stan_draws.R so it can run even when rstan is unavailable locally.
"""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "_mplconfig"))

import arviz as az
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
MCMC = Path(os.environ.get("HBRL52_OUTPUT_DIR", str(BASE / "outputs"))) / "mcmc"

KEY_PARAMETERS = [
    "mu_beta_mb_effects[1]", "mu_beta_mb_effects[2]",
    "mu_beta_mb_effects[3]", "mu_beta_mb_effects[4]",
    "mu_beta_mf_effects[1]", "mu_beta_mf_effects[2]",
    "mu_beta_mf_effects[3]", "mu_beta_mf_effects[4]",
    "mu_alpha_lload_logit", "mu_alpha_hload_logit",
]


def main():
    draw_path = MCMC / "hbrl_all_stan_draws.csv"
    if not draw_path.exists():
        raise FileNotFoundError(f"Run export_all_stan_draws.R first: {draw_path}")
    draws = pd.read_csv(draw_path)
    metadata_path = MCMC / "stan_all_draw_export_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing draw metadata: {metadata_path}")
    metadata = pd.read_csv(metadata_path).iloc[0]
    if int(metadata["subjects"]) != 52:
        raise ValueError(f"Expected 52-subject draws, found {metadata['subjects']} subjects.")
    draws = draws.sort_values(["chain", "draw"])
    chains = sorted(draws["chain"].unique())
    n_chain = len(chains)
    n_draw = int(draws.groupby("chain").size().iloc[0])
    if draws.groupby("chain").size().nunique() != 1:
        raise ValueError("Stan chains have different retained draw counts.")
    param_cols = [c for c in draws.columns if c not in {"chain", "draw"}]
    posterior = {}
    for col in param_cols:
        posterior[col] = draws[col].to_numpy(dtype=float).reshape(n_chain, n_draw)
    idata = az.from_dict(posterior=posterior)
    diag = az.summary(idata, var_names=param_cols, kind="all", round_to="none").reset_index()
    diag = diag.rename(columns={"index": "parameter", "r_hat": "Rhat", "ess_bulk": "Bulk_ESS", "ess_tail": "Tail_ESS"})
    constant = {c: bool(np.ptp(posterior[c]) == 0) for c in param_cols}
    diag["is_constant"] = diag.parameter.map(constant)
    diag.to_csv(MCMC / "hbrl_mcmc_all_parameter_diagnostics.csv", index=False)
    summary = pd.DataFrame([
        {
            "subjects": int(metadata["subjects"]),
            "n_parameters": len(param_cols),
            "n_undefined_rhat": int(diag["Rhat"].isna().sum()),
            "n_constant_parameters": int(diag["is_constant"].sum()),
            "max_rhat": float(np.nanmax(diag["Rhat"])),
            "n_rhat_gt_1_01": int(np.sum(diag["Rhat"] > 1.01)),
            "n_rhat_gt_1_05": int(np.sum(diag["Rhat"] > 1.05)),
            "min_bulk_ess": float(np.nanmin(diag["Bulk_ESS"])),
            "min_tail_ess": float(np.nanmin(diag["Tail_ESS"])),
        }
    ])
    summary.to_csv(MCMC / "hbrl_mcmc_all_parameter_summary.csv", index=False)
    key = diag[diag["parameter"].isin(KEY_PARAMETERS)].copy()
    key.to_csv(MCMC / "hbrl_mcmc_key_parameter_diagnostics.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
