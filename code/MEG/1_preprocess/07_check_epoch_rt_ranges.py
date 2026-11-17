# -*- coding: utf-8 -*-
"""Summarize the two behavioral RT variables stored in Epoch metadata."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meg_project import DATA_ROOT, MEG_CODE, meg_subjects, PREP_RESULTS
from meg_pdf import save_pdf

import re

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
EPOCHS_BASE_DIR = PREP_RESULTS / "5_epochs_resampled"
PHASE_NAME = 'decision'
SUBJECTS = meg_subjects()

# Enter the exact metadata column names if automatic detection does not find
# exactly two RT columns, for example: ("first_rt", "second_rt").
RT_COLUMNS = None

# Use "s" if RT is stored in seconds or "ms" if it is stored in milliseconds.
RT_UNIT = "s"
EXPECTED_MAX_RT_S = 2.0
FIXED_CHOOSING_PERIOD_S = 2.5

OUTPUT_FILE = PREP_RESULTS / "epoch_rt_range_summary.csv"
HISTOGRAM_FILE = OUTPUT_FILE.with_name("epoch_rt_histograms.pdf")


def find_rt_columns(metadata):
    """Return two likely RT columns, or request explicit configuration."""
    if RT_COLUMNS is not None:
        missing = [column for column in RT_COLUMNS if column not in metadata.columns]
        if missing:
            raise KeyError(
                f"Configured RT columns are absent from metadata: {missing}\n"
                f"Available columns: {metadata.columns.tolist()}"
            )
        return list(RT_COLUMNS)

    candidates = []
    for column in metadata.columns:
        name = str(column).lower()
        is_rt_name = bool(
            re.search(r"(^|[_\- ])rt\d*($|[_\- ])", name)
            or "reaction" in name
            or "response_time" in name
        )
        if is_rt_name:
            candidates.append(column)
    if len(candidates) != 2:
        raise RuntimeError(
            "Automatic detection did not find exactly two RT columns. "
            f"Candidates: {candidates}\n"
            f"Available columns: {metadata.columns.tolist()}\n"
            "Set RT_COLUMNS near the top of this script to the two exact names."
        )
    return candidates


def to_seconds(values):
    values = pd.to_numeric(values, errors="coerce")
    if RT_UNIT == "ms":
        return values / 1000.0
    if RT_UNIT == "s":
        return values
    raise ValueError("RT_UNIT must be either 's' or 'ms'.")


def describe_rt(data, column, label, valid_min, valid_max):
    """Create one publication-friendly summary row."""
    values = data[column].dropna()
    valid = values[(values > valid_min) & (values <= valid_max)]
    invalid = values[(values <= valid_min) | (values > valid_max)]

    row = {
        "Measure": label,
        "N total": len(data),
        "N valid": len(valid),
        "N missing": int(data[column].isna().sum()),
        "N outside expected range": len(invalid),
    }

    statistics = {
        "Min (s)": np.min,
        "5th percentile (s)": lambda x: np.percentile(x, 5),
        "25th percentile (s)": lambda x: np.percentile(x, 25),
        "Median (s)": np.median,
        "Mean (s)": np.mean,
        "75th percentile (s)": lambda x: np.percentile(x, 75),
        "95th percentile (s)": lambda x: np.percentile(x, 95),
        "Max (s)": np.max,
        "SD (s)": lambda x: np.std(x, ddof=1),
    }
    for name, function in statistics.items():
        row[name] = function(valid) if len(valid) else np.nan

    return row


def plot_histograms(data):
    """Plot overall distributions for RT1, RT2, and implied Choose duration."""
    plot_specs = [
        ("RT1_s", "RT1", 0.0, EXPECTED_MAX_RT_S, "#4472C4"),
        ("RT2_s", "RT2", 0.0, EXPECTED_MAX_RT_S, "#D65F5F"),
        (
            "Choose_duration_s",
            "Choose duration (2.5 - RT1)",
            FIXED_CHOOSING_PERIOD_S - EXPECTED_MAX_RT_S,
            FIXED_CHOOSING_PERIOD_S,
            "#4C956C",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for axis, (column, title, valid_min, valid_max, color) in zip(axes, plot_specs):
        values = data[column].dropna()
        values = values[(values > valid_min) & (values <= valid_max)]
        mean_value = values.mean()
        median_value = values.median()

        axis.hist(values, bins=50, color=color, edgecolor="white", linewidth=0.4)
        axis.axvline(
            mean_value,
            color="black",
            linestyle="-",
            linewidth=1.5,
            label=f"Mean = {mean_value:.3f} s",
        )
        axis.axvline(
            median_value,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Median = {median_value:.3f} s",
        )
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Trial count")
        axis.legend(frameon=False, fontsize=8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    save_pdf(fig, HISTOGRAM_FILE, bbox_inches="tight")
    print(f"Histograms saved to: {HISTOGRAM_FILE}")
    plt.show()


def main():
    all_rows = []
    missing_files = []
    rt_columns = None

    for subject in SUBJECTS:
        epoch_file = (
            EPOCHS_BASE_DIR
            / f"sub-{subject}"
            / f"sub-{subject}_{PHASE_NAME}_resamp200_epo.fif"
        )
        if not epoch_file.exists():
            missing_files.append(str(epoch_file))
            continue

        epochs = mne.read_epochs(epoch_file, preload=False, verbose=False)
        if epochs.metadata is None:
            raise RuntimeError(f"No metadata found in {epoch_file}")

        if rt_columns is None:
            rt_columns = find_rt_columns(epochs.metadata)
            print(f"Detected RT columns: {rt_columns}")

        metadata = epochs.metadata.loc[:, rt_columns].copy()
        all_rows.append(metadata)

    if not all_rows:
        raise RuntimeError(f"No {PHASE_NAME} Epoch files were found.")

    data = pd.concat(all_rows, ignore_index=True)
    rt1_name, rt2_name = rt_columns
    data["RT1_s"] = to_seconds(data[rt1_name])
    data["RT2_s"] = to_seconds(data[rt2_name])
    data["Choose_duration_s"] = FIXED_CHOOSING_PERIOD_S - data["RT1_s"]

    summary = [
        describe_rt(data, "RT1_s", f"RT1 ({rt1_name})", 0.0, EXPECTED_MAX_RT_S),
        describe_rt(data, "RT2_s", f"RT2 ({rt2_name})", 0.0, EXPECTED_MAX_RT_S),
        describe_rt(
            data,
            "Choose_duration_s",
            "Choose duration (2.5 - RT1)",
            FIXED_CHOOSING_PERIOD_S - EXPECTED_MAX_RT_S,
            FIXED_CHOOSING_PERIOD_S,
        ),
    ]

    summary_df = pd.DataFrame(summary)
    numeric_columns = summary_df.select_dtypes(include=[np.number]).columns
    summary_df[numeric_columns] = summary_df[numeric_columns].round(4)

    valid_choose_duration = data.loc[
        data["RT1_s"].between(0, EXPECTED_MAX_RT_S), "Choose_duration_s"
    ]

    print("\nRT summary (valid range: 0 < RT <= 2 s)")
    print(summary_df.to_string(index=False))
    print(
        "\nImplied Choose duration (2.5 - RT1): "
        f"{valid_choose_duration.min():.4f} to "
        f"{valid_choose_duration.max():.4f} s"
    )
    print(
        "RT1 + implied Choose duration check: "
        f"{(data['RT1_s'] + data['Choose_duration_s']).dropna().min():.4f} to "
        f"{(data['RT1_s'] + data['Choose_duration_s']).dropna().max():.4f} s"
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSummary saved to: {OUTPUT_FILE}")
    plot_histograms(data)

    if missing_files:
        print(f"Missing Epoch files: {len(missing_files)}")


if __name__ == "__main__":
    main()
