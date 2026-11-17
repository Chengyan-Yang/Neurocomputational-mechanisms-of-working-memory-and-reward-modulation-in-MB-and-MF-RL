"""Run the complete model_1 validation workflow and save a progress log."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
LOG_PATH = OUT / "run_all_diagnostics.log"
STATUS_PATH = OUT / "run_all_diagnostics_status.json"

STEPS = [
    ("Likelihood validation", "validate_likelihood.py", []),
    (
        "MCMC diagnostics",
        "run_mcmc.py",
        ["--warmup", "6000", "--draws", "12000", "--chains", "4", "--workers", "8"],
    ),
    ("Posterior predictive checks", "run_ppc.py", ["--reps", "500"]),
    (
        "Parameter recovery",
        "run_recovery.py",
        ["--empirical-reps", "3", "--broad", "80", "--starts", "16", "--workers", "8"],
    ),
    ("Figures and report", "make_figures_and_report.py", []),
]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_status(status):
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "started_at": now(), "steps": []}
    save_status(status)

    with LOG_PATH.open("w", encoding="utf-8", buffering=1) as log:
        log.write(f"Started: {status['started_at']}\n")
        log.write(f"Python: {sys.executable}\n\n")

        for title, script, args in STEPS:
            command = [sys.executable, "-u", str(BASE / script), *args]
            started = now()
            header = f"\n===== {title} | {started} =====\n$ {' '.join(command)}\n"
            print(header, end="", flush=True)
            log.write(header)
            step = {"name": title, "script": script, "started_at": started, "command": command}
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
        message = f"\nAll diagnostics completed. Report: {OUT / 'REPORT_model1_validation.md'}\n"
        print(message, end="", flush=True)
        log.write(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
