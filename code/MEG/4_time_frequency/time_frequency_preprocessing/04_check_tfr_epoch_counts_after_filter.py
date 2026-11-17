"""Count saved TFR epochs after filtering/rejection.

This script reads the saved TFR epoch files and summarizes how many epochs are
available for each subject, phase, and condition after preprocessing, artifact
rejection, and resampling. It is read-only for epoch files.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from meg_project import DATA_ROOT, available

import mne
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from tfr_preprocessing_config import (  # noqa: E402
    ANALYZED_PHASES,
    OUTPUT_ROOT,
    PHASE_CONDITIONS,
    SUBJECTS,
    epoch_path,
)


def count_one_file(subject, phase):
    generated = epoch_path(subject, phase)
    path = available(generated, DATA_ROOT / "MEG" / "preprocessed" / "time_frequency_epoch" / generated.relative_to(OUTPUT_ROOT))

    rows = []

    if not path.exists():
        rows.append(
            {
                "Subject": subject,
                "Phase": phase,
                "Condition": "ALL",
                "N epochs": 0,
                "File exists": False,
                "File": str(path),
                "Notes": "missing file",
            }
        )
        return rows

    epochs = mne.read_epochs(path, preload=False, verbose=False)
    total = len(epochs)

    rows.append(
        {
            "Subject": subject,
            "Phase": phase,
            "Condition": "ALL",
            "N epochs": total,
            "File exists": True,
            "File": str(path),
            "Notes": "",
        }
    )

    for condition in PHASE_CONDITIONS[phase]:
        if str(condition) in epochs.event_id:
            n_epochs = len(epochs[str(condition)])
            notes = ""
        else:
            n_epochs = 0
            notes = "condition absent"

        rows.append(
            {
                "Subject": subject,
                "Phase": phase,
                "Condition": str(condition),
                "N epochs": n_epochs,
                "File exists": True,
                "File": str(path),
                "Notes": notes,
            }
        )

    return rows


def main():
    rows = []
    for subject in SUBJECTS:
        for phase in ANALYZED_PHASES:
            rows.extend(count_one_file(subject, phase))

    df = pd.DataFrame(rows)

    qc_dir = OUTPUT_ROOT / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    long_file = qc_dir / "tfr_epoch_counts_after_filter_long.csv"
    wide_file = qc_dir / "tfr_epoch_counts_after_filter_wide.csv"
    summary_file = qc_dir / "tfr_epoch_counts_after_filter_summary.csv"

    df.to_csv(long_file, index=False)

    wide = (
        df.pivot_table(
            index=["Subject", "Phase"],
            columns="Condition",
            values="N epochs",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    wide.to_csv(wide_file, index=False)

    condition_rows = df[df["Condition"] != "ALL"].copy()
    summary = (
        condition_rows.groupby(["Phase", "Condition"])["N epochs"]
        .agg(
            N_subjects="count",
            Mean="mean",
            SD="std",
            Min="min",
            Median="median",
            Max="max",
        )
        .reset_index()
    )
    summary.to_csv(summary_file, index=False)

    print("\nSaved epoch-count tables:")
    print(long_file)
    print(wide_file)
    print(summary_file)

    print("\nSubject x phase total epochs:")
    print(wide[["Subject", "Phase", "ALL"]].to_string(index=False))

    missing = df[(df["Condition"] == "ALL") & (~df["File exists"])]
    if len(missing):
        print("\nMissing epoch files:")
        print(missing[["Subject", "Phase", "File"]].to_string(index=False))


if __name__ == "__main__":
    main()
