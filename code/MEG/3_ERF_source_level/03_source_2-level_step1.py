# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:34:45 2026

@author: Cyan
"""

# -*- coding: utf-8 -*-
"""
Estimate source activity from evoked responses and morph it to the template.
"""

# -*- coding: utf-8 -*-
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meg_project import DATA_ROOT, meg_subjects, PREP_RESULTS, SOURCE_RESULTS, available

import mne
import os

# ==========================================
# ==========================================
subjects_dir = f'{DATA_ROOT}/MRI/surf'

subject_list = meg_subjects()

from cortex_mask_analysis_config import CONDITION_CODES, DATA_PHASE

phase_name = DATA_PHASE
conditions = CONDITION_CODES
template_brain = 'fsaverage'

epochs_dir = f'{PREP_RESULTS}/5_epochs_resampled'
evoked_dir = f'{DATA_ROOT}/MEG/preprocessed/evokeds'
output_dir = SOURCE_RESULTS / 'morphed_stcs'


print("\n" + "="*50)
print(f"Estimating and morphing evoked responses for {phase_name}")
print("="*50)

template_dir = Path(subjects_dir) / template_brain
if not template_dir.is_dir():
    raise FileNotFoundError(f"Missing local FreeSurfer template: {template_dir}")

# ==========================================
# ==========================================
for subject in subject_list:
    
    sub_output_dir = os.path.join(output_dir,f"sub-{subject}", phase_name)
    
    # Skip a subject only when both hemispheres exist for all four conditions.
    if os.path.exists(sub_output_dir) and \
       all(os.path.isfile(os.path.join(sub_output_dir, f"{subject}_{c}-morphed-stc.h5-{hemi}.stc"))
           for c in conditions for hemi in ("lh", "rh")):
        print(f"Skipping {subject}: morphed estimates already exist.")
        continue

    meg_file = os.path.join(epochs_dir, f'sub-{subject}', f'sub-{subject}_{phase_name}_resamp200_epo.fif')
    fwd_file = f'{SOURCE_RESULTS}/forward/sub-{subject}-fwd.fif'
    ave_file = os.path.join(evoked_dir, f'sub-{subject}', f'sub-{subject}_{phase_name}-ave.fif')
    
    ave_file = available(PREP_RESULTS / "evokeds" / Path(ave_file).relative_to(evoked_dir), Path(ave_file))
    if not all([os.path.exists(f) for f in [meg_file, fwd_file, ave_file]]):
        print(f"Skipping {subject}: one or more required input files are missing.")
        continue

    try:
        print(f"Processing subject {subject}")
        
        evokeds = mne.read_evokeds(ave_file, verbose=False)
        evoked_dict = {str(ev.comment): ev for ev in evokeds}
        
        epochs = mne.read_epochs(meg_file, preload=True, verbose=False)
        noise_cov = mne.compute_covariance(epochs, tmax=0, method='empirical', verbose=False)
        
        fwd = mne.read_forward_solution(fwd_file, verbose=False)
        inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov, verbose=False)
        
        print("Computing the source morph")
        dummy_stc = mne.minimum_norm.apply_inverse(evokeds[0], inv, lambda2=1.0/9.0, method='dSPM', verbose=False)
        morph = mne.compute_source_morph(dummy_stc, subject_from=subject, subject_to=template_brain, subjects_dir=subjects_dir, verbose=False)
        
        os.makedirs(sub_output_dir, exist_ok=True)
        
        for cond in conditions:
            if cond in evoked_dict:
                evoked_cond = evoked_dict[cond]
                stc_cond = mne.minimum_norm.apply_inverse(evoked_cond, inv, lambda2=1.0/9.0, method='dSPM', verbose=False)
                stc_fsa = morph.apply(stc_cond)
                
                out_file = os.path.join(sub_output_dir, f"{subject}_{cond}-morphed-stc.h5")
                stc_fsa.save(out_file, overwrite=True)
            else:
                print(f"{subject}: missing evoked condition {cond}.")
                
        print(f"Finished processing available conditions for {subject}.")

    except Exception as e:
        print(f"Source estimation failed for {subject}: {e}")

print("Finished batch source estimation.")

# ==========================================
# Inspect one subject's morphed source estimate.
# ==========================================
print("\n" + "="*50)
print("Searching for a source estimate for visual quality control")

sample_cond = conditions[0]
sample_base_path = None
sample_sub_found = None

for sub in subject_list:
    check_lh_file = os.path.join(output_dir, f"sub-{sub}", phase_name, f"{sub}_{sample_cond}-morphed-stc.h5-lh.stc")
    
    # Pass the common .h5 stem to MNE without the -lh.stc suffix.
    base_path = os.path.join(output_dir, f"sub-{sub}", phase_name, f"{sub}_{sample_cond}-morphed-stc.h5")
    
    if os.path.exists(check_lh_file):
        sample_base_path = base_path
        sample_sub_found = sub
        break

if sample_base_path:
    print(f"Rendering the source estimate for {sample_sub_found}")
    stc_check = mne.read_source_estimate(sample_base_path)
    
    brain = stc_check.plot(
        subject=template_brain,
        subjects_dir=subjects_dir,
        surface='pial',
        hemi='both',
        views='ventral',
        initial_time=0.333,
        time_viewer=True,
        colormap='hot',
        clim='auto',
        background='black',
        size=(800, 600),
        smoothing_steps=10
    )
else:
    print(f"No source estimates found for visual quality control.")
print("="*50)
