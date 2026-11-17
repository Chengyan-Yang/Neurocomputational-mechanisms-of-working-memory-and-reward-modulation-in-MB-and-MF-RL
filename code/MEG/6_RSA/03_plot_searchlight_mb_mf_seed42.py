"""Plot the seed-42 MB/MF results using the saved project plotting scripts.

Run this file in the MNE environment after both model fits.
Only Figures_publication inside searchlight_RSA_seed42 is written.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meg_project import DATA_ROOT, project_input, RSA_RESULTS

import argparse
from datetime import datetime
import hashlib
import json
import os
import runpy
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = RSA_RESULTS / "searchlight_seed42"
SCRIPTS = BASE / "searchlight_RSA_seed42" / "visualization_scripts"
STEPS = ("overall", "pattern", "rdm")


def script_path(model, step):
    name = {
        "overall": f"01_plot_source_searchlight_rsa_overall_{model}.py",
        "pattern": "02_plot_source_searchlight_rsa_pattern_style_figures.py",
        "rdm": "03_plot_representative_searchlight_cluster_rdms.py",
    }[step]
    return SCRIPTS / f"searchlight_RSA_{model}" / name


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_status(state):
    (ROOT / "visualization_status.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def worker(model, step):
    result = ROOT / f"overall_{model}_source_searchlight"
    config = json.loads((result / "source_searchlight_rsa_config.json").read_text())
    assert config["random_seed"] == 42
    for key in list(os.environ):
        if key.startswith("SEARCHLIGHT_"):
            del os.environ[key]
    os.environ.update({
        "SEARCHLIGHT_RSA_RESULT_DIR": str(result),
        "SEARCHLIGHT_RSA_MF_RESULT_DIR": str(result),
        "SEARCHLIGHT_RSA_FIGURE_DIR": str(result / "Figures_publication"),
        "SEARCHLIGHT_RSA_MF_FIGURE_DIR": str(result / "Figures_publication"),
        "SEARCHLIGHT_SUBJECTS_DIR": str(DATA_ROOT / "MRI" / "surf"),
        "SEARCHLIGHT_RSA_CLUSTER_ALPHA": str(config["cluster_alpha"]),
        "SEARCHLIGHT_RSA_RDM_TOP_N": "1" if model == "mb" else "3",
        "MPLBACKEND": "Agg", "PYVISTA_OFF_SCREEN": "true",
        "QT_QPA_PLATFORM": "windows", "PYTHONUNBUFFERED": "1",
    })
    sys.setrecursionlimit(max(50000, sys.getrecursionlimit()))
    if step == "overall":
        import pyvista
        pyvista.OFF_SCREEN = True
        import mne
        mne.viz.set_3d_backend("pyvistaqt")
    path = script_path(model, step)
    print(f"Executing {path}\nSeed42 input: {result}", flush=True)
    runpy.run_path(str(path), run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2, metavar=("MODEL", "STEP"))
    args = parser.parse_args()
    if args.worker:
        model, step = args.worker
        assert model in ("mb", "mf") and step in STEPS
        worker(model, step)
        return
    inputs = [p for model in ("mb", "mf")
              for p in (ROOT / f"overall_{model}_source_searchlight").rglob("*")
              if p.is_file() and "Figures_publication" not in p.parts]
    state = {"status": "running", "seed": 42,
             "started_at": datetime.now().astimezone().isoformat(), "completed": [],
             "input_sha256": {str(p): digest(p) for p in inputs}}
    logs = ROOT / "visualization_logs"
    logs.mkdir(exist_ok=True)
    write_status(state)
    try:
        for model in ("mb", "mf"):
            for step in STEPS:
                key = f"{model}/{step}"
                state["current"] = key
                write_status(state)
                log = logs / f"{model}_{step}.log"
                print(f"Starting {key}; log: {log}", flush=True)
                with log.open("w", encoding="utf-8") as stream:
                    subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", model, step],
                                   stdout=stream, stderr=subprocess.STDOUT, check=True)
                state["completed"].append(key)
                write_status(state)
                print(f"Completed {key}", flush=True)
        import pandas as pd
        from meg_pdf import pdf_size_inches
        outputs = []
        for model in ("mb", "mf"):
            result = ROOT / f"overall_{model}_source_searchlight"
            figures = result / "Figures_publication"
            table = pd.read_csv(figures / "tables" / "all_phases_significant_cluster_table.csv")
            manifest = pd.read_csv(result / "source_searchlight_rsa_manifest.csv")
            for row in manifest.itertuples():
                assert int((table.phase == row.phase).sum()) == int(row.n_cluster_significant)
            brain = pd.read_csv(figures / "brain_maps" / "brain_map_manifest.csv")
            assert len(brain) == 16
            brain_pngs = {Path(name) for name in brain["file_png"]}
            from PIL import Image
            for path in brain_pngs:
                with Image.open(path) as image:
                    assert image.mode == "RGBA" and image.getchannel("A").getextrema()[0] == 0, path
            per_phase = table.groupby("phase").size().clip(upper=1 if model == "mb" else 3)
            if per_phase.sum():
                rdms = pd.read_csv(figures / "representative_rdms" / "representative_rdm_manifest.csv")
                assert len(rdms) == int(per_phase.sum())
            for path in figures.rglob("*"):
                if path.is_file():
                    if path.suffix.lower() in (".png", ".tiff") and path not in brain_pngs:
                        continue
                    if path.with_suffix(".png") in brain_pngs and path.suffix.lower() == ".pdf":
                        continue
                    if path.stem == "fig4_phase_ordered_brain_map_contact_sheet":
                        continue
                    if path.suffix.lower() == ".pdf":
                        pdf_size_inches(path)
                    outputs.append({"model": model.upper(), "file": str(path), "bytes": path.stat().st_size})
        for path, expected in state["input_sha256"].items():
            assert digest(project_input(path)) == expected, f"Analysis input changed: {path}"
        pd.DataFrame(outputs).to_csv(ROOT / "visualization_file_manifest.csv", index=False, encoding="utf-8-sig")
        state.update(status="completed", analysis_inputs_unchanged=True, n_output_files=len(outputs),
                     finished_at=datetime.now().astimezone().isoformat())
        state.pop("current", None)
        print(f"All six plotting steps completed; {len(outputs)} output files verified.", flush=True)
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        write_status(state)


if __name__ == "__main__":
    main()
