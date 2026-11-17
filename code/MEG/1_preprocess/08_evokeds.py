# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:50:27 2026

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
subject_list = meg_subjects()

phases_dict = {'incentive': [1, 2, 3, 4],
 'decision': [11, 21, 31, 41],
 'transition': [13, 23, 33, 43],
 'feedback': [51, 52, 53, 54]}

# phases_dict = {
#     'decision':[11, 21, 31, 41]
    
# }

# MNE event_id keys are strings; convert condition codes before selection.
phases = {phase: [str(cond) for cond in conds] for phase, conds in phases_dict.items()}

epochs_base_dir = f'{PREP_RESULTS}/5_epochs_resampled'
evoked_base_dir = PREP_RESULTS / 'evokeds'
os.makedirs(evoked_base_dir, exist_ok=True)

print("\n" + "="*50)
print("Computing condition-wise evoked responses")
print("="*50)

# ==========================================
# ==========================================
for subject in subject_list:
    print(f"Processing sub-{subject}")
    
    sub_evoked_dir = os.path.join(evoked_base_dir, f'sub-{subject}')
    os.makedirs(sub_evoked_dir, exist_ok=True)
    
    for phase_name, conditions in phases.items():
        
        epochs_file = os.path.join(epochs_base_dir, f'sub-{subject}', f'sub-{subject}_{phase_name}_resamp200_epo.fif')
        
        if not os.path.exists(epochs_file):
            continue
            
        try:
            # Read epochs on demand to limit memory usage.
            epochs = mne.read_epochs(epochs_file, preload=False, verbose=False)
            
            evokeds_to_save = []
            
            for cond in conditions:
                if cond in epochs.event_id:
                    evoked = epochs[cond].average()
                    
                    # Store the condition code in evoked.comment for downstream selection.
                    evoked.comment = cond 
                    
                    evokeds_to_save.append(evoked)
                else:
                    print(f"sub-{subject}, {phase_name}: missing condition '{cond}'.")
            
            if evokeds_to_save:
                out_file = os.path.join(sub_evoked_dir, f'sub-{subject}_{phase_name}-ave.fif')
                mne.write_evokeds(out_file, evokeds_to_save, overwrite=True)
                print(f"Saved {phase_name} with {len(evokeds_to_save)} conditions.")
                
        except Exception as e:
            print(f"Failed to process {phase_name}: {e}")

print("\n" + "="*50)
print("Finished computing evoked responses.")
print("="*50)