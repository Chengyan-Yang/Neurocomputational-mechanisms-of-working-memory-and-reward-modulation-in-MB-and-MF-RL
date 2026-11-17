# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:32:26 2026

@author: Cyan
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meg_project import DATA_ROOT, meg_subjects, PREP_RESULTS, SOURCE_RESULTS, available

import mne
import os

# ==========================================
# ==========================================
subjects_dir = f'{DATA_ROOT}/MRI/surf'
spacing = 'oct6'  

subject_list = meg_subjects()

print("\n" + "="*50)
print(f"Starting forward modeling for {len(subject_list)} subjects")
print("="*50)

# ==========================================
# ==========================================
for subject in subject_list:
    
    meg_file = f'{PREP_RESULTS}/5_epochs_resampled/sub-{subject}/sub-{subject}_incentive_resamp200_epo.fif'
    trans_file = available(SOURCE_RESULTS / "trans" / f"sub-{subject}-trans.fif", DATA_ROOT / "MRI" / "trans" / f"sub-{subject}-trans.fif")
    fwd_file = f'{SOURCE_RESULTS}/forward/sub-{subject}-fwd.fif'
    
    print(f"Processing subject {subject}")

    # ==========================================
    # ==========================================
    
    if not os.path.exists(meg_file) or not os.path.exists(trans_file):
        print(f"Skipping {subject}: required MEG recording or transform is missing.")
        continue
        
    if os.path.exists(fwd_file):
        print(f"Skipping {subject}: forward solution already exists.")
        continue

    # ==========================================
    # ==========================================
    try:
        print("Creating the source space")
        src = mne.setup_source_space(subject, spacing=spacing, subjects_dir=subjects_dir, add_dist='patch', n_jobs=-1)

        print("Constructing the BEM conductor model")
        bem_model = mne.make_bem_model(subject, ico=4, subjects_dir=subjects_dir, conductivity=(0.3,))
        bem = mne.make_bem_solution(bem_model)

        print("Computing the forward solution with the MEG-MRI transform")
        fwd = mne.make_forward_solution(meg_file, trans=trans_file, src=src, bem=bem, meg=True, eeg=False, n_jobs=-1)

        print("Saving the forward solution")
        os.makedirs(os.path.dirname(fwd_file), exist_ok=True)
        mne.write_forward_solution(fwd_file, fwd, overwrite=True)
        print(f"Saved forward solution for {subject}.")

    except Exception as e:
        print(f"Forward modeling failed for {subject}: {e}")
        continue

print("\n" + "="*50)
print("Finished batch forward modeling.")
print("="*50)



# Inspect one subject's sensor and source alignment.
# ==========================================
# ==========================================
subjects_dir = f'{DATA_ROOT}/MRI/surf'
subject = '30'  # Subject selected for visual quality control

print(f"Inspecting 3D alignment for {subject}")

# ==========================================
# ==========================================
meg_file = f'{PREP_RESULTS}/5_epochs_resampled/sub-{subject}/sub-{subject}_incentive_resamp200_epo.fif'
trans_file = available(SOURCE_RESULTS / "trans" / f"sub-{subject}-trans.fif", DATA_ROOT / "MRI" / "trans" / f"sub-{subject}-trans.fif")
fwd_file = f'{SOURCE_RESULTS}/forward/sub-{subject}-fwd.fif'

if not os.path.exists(fwd_file):
    raise FileNotFoundError(f"Forward solution not found for {subject}. Check the input path and model output.")

# ==========================================
# ==========================================
print("Loading the source space and MEG metadata")
fwd = mne.read_forward_solution(fwd_file, verbose=False)
src = fwd['src']

info = mne.io.read_info(meg_file, verbose=False)

print("Rendering the alignment in MEG coordinates")

fig = mne.viz.plot_alignment(
    info=info,
    trans=trans_file,
    subject=subject,
    subjects_dir=subjects_dir,
    src=src,
    surfaces=['brain', 'inner_skull'],
    coord_frame='meg',
    sensor_colors='blue',
    show_axes=True,
    interaction='terrain'
)

print("Finished rendering the alignment.")

print("Explore the alignment in the 3D window.")
print("Drag with the left mouse button to rotate.")
print("Drag vertically with the right mouse button to zoom.")
print("Close the window when finished.")