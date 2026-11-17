# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:34:48 2026

@author: Cyan
"""

# -*- coding: utf-8 -*-
"""
Review ICA artifact components and reconstruct cleaned MEG recordings.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meg_project import DATA_ROOT, meg_subjects, PREP_RESULTS

import mne
import os
import json

mne.set_log_level('WARNING')

in_raw_root = f'{PREP_RESULTS}/1_combined'
in_ica_root = f'{PREP_RESULTS}/2_ICA/run_ica'
out_clean_root = f'{PREP_RESULTS}/2_ICA/cleaned'

# Store the final component exclusions by subject.
log_file = os.path.join(out_clean_root, 'ica_exclusion_log.json')
os.makedirs(out_clean_root, exist_ok=True)

subjects = meg_subjects()

if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        exclusion_dict = json.load(f)
else:
    exclusion_dict = {}

for sub in subjects:
    print(f"Reviewing ICA components for {sub}")
    
    clean_save_path = os.path.join(out_clean_root, f'sub-{sub}_clean_200Hz.fif')
    if os.path.exists(clean_save_path):
        print(f"Skipping {sub}: cleaned recording already exists.")
        continue

    folder = Path(in_raw_root) / f"sub-{sub}"
    raw_file = folder / f"sub-{sub}_combined_200Hz.fif"
    if not raw_file.exists():
        raw_file = folder / f"sub-{sub}_combined_raw.fif"
    ica_file = os.path.join(in_ica_root, f'sub-{sub}_ica.fif')
    
    if not os.path.exists(raw_file) or not os.path.exists(ica_file):
        print(f"Skipping {sub}: required input files are missing.")
        continue
        
    raw = mne.io.read_raw_fif(raw_file, preload=True)
    if raw.info["sfreq"] != 200:
        raw.resample(200)
    ica = mne.preprocessing.read_ica(ica_file)
    
    # ==========================================
    # Identify candidate EOG and ECG components for manual review.
    # ==========================================
    
    # Use the EOG proxy only if it is present and not marked bad.
    # Inspect the recording to identify a channel sensitive to blinks.
    # raw.plot(picks='all', duration=10.0)
    eog_proxy_ch = 'MLT41-4504' 
    
    try:
        if eog_proxy_ch in raw.ch_names and eog_proxy_ch not in raw.info['bads']:
            eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_proxy_ch, threshold=3.0)
            print(f"EOG candidates from {eog_proxy_ch}: {eog_indices}")
        else:
            print(f"Skipping EOG detection: {eog_proxy_ch} is missing or marked bad.")
            eog_indices = []

        # Estimate ECG activity from MEG channels when no ECG channel is specified.
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method='ctps')
        print(f"ECG candidates from CTPS: {ecg_indices}")

    except Exception as e:
        print(f"Automatic artifact detection failed: {e}")
        eog_indices, ecg_indices = [], []

    auto_bads = list(set(eog_indices + ecg_indices))
    ica.exclude = auto_bads
    print(f"Candidate components for exclusion: {auto_bads}")

    # Review candidate components on the topographic maps.
    ica.plot_components(inst=raw)
    
    # Click component names in the source browser to toggle their exclusion.
    ica.plot_sources(raw, block=True)  # Close the browser to continue with the reviewed exclusions.
    
    final_bads = ica.exclude
    print(f"Confirmed component exclusions: {final_bads}")
    
    exclusion_dict[sub] = [int(i) for i in final_bads]
    with open(log_file, 'w') as f:
        json.dump(exclusion_dict, f, indent=4)

    # Apply the reviewed ICA solution to the analysis recording.
    raw_clean = raw.copy()
    ica.apply(raw_clean)
    
    raw_clean.save(clean_save_path, overwrite=True)
    print(f"Saved the cleaned recording for {sub}.")
    
    del raw, raw_clean, ica
    
    
    
# Apply the reviewed exclusions to the recording at its acquisition sampling rate.
in_original_raw_root = f'{PREP_RESULTS}/1_combined'
in_ica_root = f'{PREP_RESULTS}/2_ICA/run_ica'
out_clean_root = f'{PREP_RESULTS}/2_ICA/cleaned'

log_file = os.path.join(out_clean_root, 'ica_exclusion_log.json')

subjects = meg_subjects()

os.makedirs(out_clean_root, exist_ok=True)

if not os.path.exists(log_file):
    raise FileNotFoundError(f"ICA exclusion log not found: {log_file}")
    
with open(log_file, 'r') as f:
    exclusion_dict = json.load(f)

for sub in subjects:
    print(f"\n{'='*50}")
    print(f"Applying ICA to the original-rate recording for {sub}")
    print(f"{'='*50}")

    # Use the combined recording without the _200Hz suffix.
    raw_file = os.path.join(in_original_raw_root, f'sub-{sub}', f'sub-{sub}_combined_raw.fif')
    ica_file = os.path.join(in_ica_root, f'sub-{sub}_ica.fif')
    clean_save_path = os.path.join(out_clean_root, f'sub-{sub}_clean_raw.fif')

    if not os.path.exists(raw_file):
        print(f"Skipping: original-rate recording not found at {raw_file}.")
        continue
    if not os.path.exists(ica_file):
        print(f"Skipping: ICA fit not found at {ica_file}.")
        continue

    if os.path.exists(clean_save_path):
        print(f"Skipping {sub}: cleaned original-rate recording already exists.")
        continue

    print("Loading the original-rate recording and ICA fit")
    raw_orig = mne.io.read_raw_fif(raw_file, preload=True)
    ica = mne.preprocessing.read_ica(ica_file)

    if sub in exclusion_dict:
        ica.exclude = exclusion_dict[sub]
        print(f"Excluding components from the review log: {ica.exclude}")
    else:
        print(f"Skipping {sub}: no entry in the ICA exclusion log.")
        continue

    print("Applying ICA to the original-rate recording")
    raw_clean = raw_orig.copy()
    ica.apply(raw_clean)

    raw_clean.save(clean_save_path, overwrite=True)
    print(f"Saved the cleaned original-rate recording for {sub}.")

    del raw_orig, raw_clean, ica

print("Finished applying ICA to the original-rate recordings.")