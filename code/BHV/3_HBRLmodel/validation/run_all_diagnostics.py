"""Run saved-draw MCMC diagnostics and PPC for the MEG 52-subject HBRL fit."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("HBRL52_OUTPUT_DIR", str(BASE / "outputs")))
LOG_PATH = OUT / "run_all_diagnostics.log"
STATUS_PATH = OUT / "run_all_diagnostics_status.json"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def find_rscript():
    configured = os.environ.get("HBRL_RSCRIPT_EXE")
    if configured:
        candidate = Path(configured)
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"HBRL_RSCRIPT_EXE does not exist: {candidate}")
    discovered = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if not discovered and os.name == "nt":
        r_home = os.environ.get("R_HOME")
        candidates = [Path(r_home) / "bin" / "Rscript.exe"] if r_home else []
        program_files = os.environ.get("ProgramFiles")
        if program_files:
            candidates.extend(sorted((Path(program_files) / "R").glob("R-*/bin/Rscript.exe"), reverse=True))
        discovered = next((str(path) for path in candidates if path.is_file()), None)
    if discovered:
        return discovered
    raise FileNotFoundError(
        "Rscript was not found. Install R or set HBRL_RSCRIPT_EXE to the full Rscript.exe path."
    )


def save_status(status):
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppc-reps", type=int, default=500)
    parser.add_argument("--skip-full-parameters", action="store_true", help="Skip exporting/diagnosing all flat Stan parameters.")
    parser.add_argument("--include-stan-recovery", action="store_true", help="Also simulate and refit 5 full hierarchical Stan datasets (requires RStan).")
    args = parser.parse_args()
    if args.ppc_reps < 1:
        parser.error("PPC reps must be positive.")

    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(BASE / "_mplconfig"))
    rscript = find_rscript()
    steps = [
        ("Check Python dependencies", [sys.executable, "-c", "import numpy, pandas, scipy, numba, arviz, matplotlib; print('Python dependencies OK')"]),
        ("Export subject posterior draws", [rscript, str(BASE / "export_stan_draws.R")]),
    ]
    if args.include_stan_recovery:
        steps.insert(1, ("Check R recovery packages", [rscript, "-e", "p <- c('rstan','dplyr','readr','tidyr','ggplot2'); missing <- p[!vapply(p, requireNamespace, logical(1), quietly=TRUE)]; if(length(missing)) stop(paste('Install R packages:', paste(missing, collapse=', ')))"]))
    if not args.skip_full_parameters:
        steps.extend(
            [
                ("Export all saved Stan parameters", [rscript, str(BASE / "export_all_stan_draws.R")]),
                ("Full-parameter MCMC diagnostics", [sys.executable, "-u", str(BASE / "run_full_mcmc_diagnostics.py")]),
                ("Export key-parameter 95% HDIs", [sys.executable, "-u", str(BASE / "export_key_parameter_hdi95.py")]),
            ]
        )
    validation_args = ["--ppc-reps", str(args.ppc_reps)]
    steps.append(
        (
            "Core MCMC diagnostics, PPC, figures and report",
            [sys.executable, "-u", str(BASE / "run_hbrl_validation.py"), *validation_args],
        )
    )
    if args.include_stan_recovery:
        os.environ.setdefault("HBRL_PLOT_PYTHON", sys.executable)
        steps.extend([
            ("Full hierarchical Stan parameter recovery", [rscript, str(BASE / "parameter_recovery" / "parameter_recovery_run_5sets.R")]),
            ("Update report with Stan recovery", [sys.executable, "-u", str(BASE / "run_hbrl_validation.py"), "--report-only"]),
        ])

    status = {"status": "running", "started_at": now(), "steps": []}
    save_status(status)
    with LOG_PATH.open("w", encoding="utf-8", buffering=1) as log:
        log.write(f"Started: {status['started_at']}\nPython: {sys.executable}\nRscript: {rscript}\n")
        for title, command in steps:
            started = now()
            header = f"\n===== {title} | {started} =====\n$ {' '.join(command)}\n"
            print(header, end="", flush=True)
            log.write(header)
            step = {"name": title, "started_at": started, "command": command}
            status["steps"].append(step)
            save_status(status)
            try:
                proc = subprocess.Popen(
                    command,
                    cwd=BASE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                return_code = proc.wait()
            except Exception as exc:
                return_code = 1
                message = f"Runner error: {type(exc).__name__}: {exc}\n"
                print(message, end="", flush=True)
                log.write(message)

            step["finished_at"] = now()
            step["return_code"] = return_code
            if return_code != 0:
                status.update({"status": "failed", "failed_step": title, "finished_at": now()})
                save_status(status)
                message = f"\nStopped after {title} (exit code {return_code}). See {LOG_PATH}.\n"
                print(message, end="", flush=True)
                log.write(message)
                return return_code
            save_status(status)

        status.update({"status": "completed", "finished_at": now()})
        save_status(status)
        message = f"\nAll requested diagnostics completed. Report: {OUT / 'REPORT_HBRL_validation.md'}\n"
        print(message, end="", flush=True)
        log.write(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
