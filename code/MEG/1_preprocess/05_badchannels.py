# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:01:13 2026

@author: Cyan
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meg_project import DATA_ROOT, meg_subjects, PREP_RESULTS

import mne
import os

# ==========================================
# ==========================================
subjects = meg_subjects()

input_root = f'{PREP_RESULTS}/3_epochs'
output_root = f'{PREP_RESULTS}/4_epochs_interpolated'

phases = ['incentive', 'decision', 'transition', 'feedback']


bads_group_1 = ['MZF02-4504', 'MLF55-4504', 'MLO24-4504', 'MRT16-4504', 'MRT23-4504', 'MRT41-4504'] 
bads_group_2 = ['MLF31-4504', 'MLO42-4504', 'MRT52-4504', 'MZF02-4504', 'MLF21-4504', 'MLF41-4504', 
                'MLF22-4504', 'MLF32-4504', 'MLF42-4504'] 
# Group 2 includes noisy sensors MLF31-4504, MZF02-4504, MLF21-4504,
# MLF41-4504, MLF22-4504, MLF32-4504 and MLF42-4504; other listed sensors may be absent.

special_subs = ['01', '02', '03']

print(f"Interpolating bad channels for {len(subjects)} subjects across {len(phases)} phases")

success_count = 0

for sub in subjects:
    # Use the same bad-channel list across phases for each subject.
    target_bads = bads_group_1 if sub in special_subs else bads_group_2

    for phase in phases:

        input_file = os.path.join(input_root, f'sub-{sub}', f'sub-{sub}_{phase}_epo.fif')
        if not os.path.exists(input_file):
            print(f"Skipping sub-{sub} {phase}: epoch file not found.")
            continue

        print(f"Processing sub-{sub} {phase}", end=" ")

        epochs = mne.read_epochs(input_file, preload=True, verbose=False)

        # Select only bad channels present in this recording.
        valid_bads = [ch for ch in target_bads if ch in epochs.ch_names]
        epochs.info['bads'] = valid_bads

        # Use a fixed head origin for interpolation.
        if len(valid_bads) > 0:
            epochs.interpolate_bads(reset_bads=True,
                                    method={'meg': 'MNE'},
                                    origin=(0., 0., 0.04),
                                    verbose=False)
            print(f"Interpolated {len(valid_bads)} bad channels.", end=" ")
        else:
            print("No bad channels to interpolate.", end=" ")

        out_dir = os.path.join(output_root, f'sub-{sub}')
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f'sub-{sub}_{phase}_interp_epo.fif')
        epochs.save(out_file, overwrite=True, verbose=False)
        print("Saved.")
        success_count += 1

print(f"Interpolation complete. Saved {success_count} files.")
print(f"Interpolated epochs: {output_root}")