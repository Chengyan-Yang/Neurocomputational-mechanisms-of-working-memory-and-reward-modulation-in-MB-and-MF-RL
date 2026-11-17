# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 22:17:42 2026

@author: Cyan
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from meg_project import DATA_ROOT, available, TFR_PREP_RESULTS

import mne
import pandas as pd

SUBJECT = "38"
PHASE = 'incentive'

EPOCH_ROOT = DATA_ROOT / "MEG" / "preprocessed" / "time_frequency_epoch"

PHASE_SUFFIX = {'incentive': None, 'transition': 3}

def expected_trigger_from_metadata(magnitude, load, phase):
    magnitude = str(magnitude).lower()
    load = str(load).lower()

    if magnitude == "small" and load == "low":
        base = 1
    elif magnitude == "large" and load == "low":
        base = 2
    elif magnitude == "small" and load == "high":
        base = 3
    elif magnitude == "large" and load == "high":
        base = 4
    else:
        return None

    if phase == "incentive":
        return base
    return base * 10 + PHASE_SUFFIX[phase]

path = EPOCH_ROOT / f"sub-{SUBJECT}" / f"sub-{SUBJECT}_{PHASE}_tfr-epo.fif"
path = available(TFR_PREP_RESULTS / "time_frequency_epoch" / path.relative_to(EPOCH_ROOT), path)
epochs = mne.read_epochs(path, preload=False, verbose=False)

df = epochs.metadata.copy().reset_index(drop=True)
df["meg_trigger"] = epochs.events[:, 2]

df["expected_trigger"] = [
    expected_trigger_from_metadata(magnitude, load, PHASE)
    for magnitude, load in zip(df["magnitude"], df["load"])
]

df["trigger_match"] = df["meg_trigger"] == df["expected_trigger"]

print("=" * 80)
print(f"sub-{SUBJECT}, phase = {PHASE}")
print("file:", path)
print("n epochs:", len(df))
print("match:", df["trigger_match"].sum())
print("mismatch:", (~df["trigger_match"]).sum())
print("match rate:", df["trigger_match"].mean())

bad = df[~df["trigger_match"]]
print("\nFirst mismatch rows:")
print(
    bad[["trials_globe", "meg_trigger", "expected_trigger", "magnitude", "load", "miss"]]
    .head(30)
    .to_string(index=False)
)

print("\nShift check:")
for shift in range(-5, 6):
    shifted_expected = df["expected_trigger"].shift(shift)
    rate = (df["meg_trigger"] == shifted_expected).mean()
    print(f"shift {shift:+}: {rate:.3f}")