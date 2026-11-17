"""Compute four-phase source searchlight statistics for MB and MF using seed 42."""
import os
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from meg_project import behavior_rdm


def main():
    for model in ("mb", "mf"):
        if not behavior_rdm(model).is_file():
            raise FileNotFoundError(f"Run 01_build_behavior_rdms.py first: {behavior_rdm(model)}")

    # Ignore smoke-test limits or output overrides left in the Spyder session.
    env = {k: v for k, v in os.environ.items() if not k.startswith("SEARCHLIGHT_")}
    for model in ("mb", "mf"):
        script = (BASE / "searchlight_RSA_seed42" / f"overall_{model}_source_searchlight"
                  / f"01_run_source_searchlight_rsa_overall_{model}.py")
        print(f"Computing {model.upper()} searchlight RSA", flush=True)
        subprocess.run([sys.executable, "-u", str(script)], check=True, env=env)


if __name__ == "__main__":
    main()
