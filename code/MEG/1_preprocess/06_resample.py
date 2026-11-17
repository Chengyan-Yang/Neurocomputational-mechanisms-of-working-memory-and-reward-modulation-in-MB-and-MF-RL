# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:14:16 2026

@author: Cyan
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meg_project import DATA_ROOT, meg_subjects, PREP_RESULTS

import mne
import os
from tqdm import tqdm

# ==========================================
# ==========================================
subjects = meg_subjects()

input_root = f'{PREP_RESULTS}/4_epochs_interpolated'
output_root = f'{PREP_RESULTS}/5_epochs_resampled'

phases = ['incentive', 'decision', 'transition', 'feedback']



target_sfreq = 200

print(f"Resampling to {target_sfreq} Hz for {len(subjects)} subjects")
success_count = 0

# ==========================================
# ==========================================
for sub in tqdm(subjects, desc="Subjects", unit="subject"):
    
    out_dir = os.path.join(output_root, f'sub-{sub}')
    os.makedirs(out_dir, exist_ok=True)
    
    for phase in phases:

        input_file = os.path.join(input_root, f'sub-{sub}', f'sub-{sub}_{phase}_interp_epo.fif')
        
        if not os.path.exists(input_file):
            continue

        epochs = mne.read_epochs(input_file, preload=True, verbose=False)
        
        # MNE applies its anti-aliasing filter during resampling.
        epochs.resample(sfreq=target_sfreq, verbose=False)
        
        out_file = os.path.join(out_dir, f'sub-{sub}_{phase}_resamp200_epo.fif')
        epochs.save(out_file, overwrite=True, verbose=False)
        
        success_count += 1

print(f"Resampling complete. Saved {success_count} files.")
print(f"Resampled epochs: {output_root}")