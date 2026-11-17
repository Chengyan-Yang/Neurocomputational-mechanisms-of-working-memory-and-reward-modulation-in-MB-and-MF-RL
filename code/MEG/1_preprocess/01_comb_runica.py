# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:58:56 2026

@author: Cyan
"""

# -*- coding: utf-8 -*-
"""
Prepare empty-room projections and concatenate task recordings.
Mark known bad channels before computing projections and combining runs.
Retain the CTF third-order gradient compensation.
Compute six magnetometer SSP vectors and no gradiometer vectors.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meg_project import DATA_ROOT, meg_subjects, PREP_RESULTS

import mne
import os
import glob

subjects = meg_subjects()

data_root = f'{DATA_ROOT}/MEG'
noise_root = f'{DATA_ROOT}/MEG/Noise'
out_root = f'{PREP_RESULTS}'

bads_group_1 = []  # These 269-channel recordings already lack MZF02-4504, MLF55-4504, MLO24-4504, MRT16-4504, MRT23-4504 and MRT41-4504.
bads_group_2 = ['MZF02-4504', 'MLF21-4504', 'MLF41-4504', 'MLF22-4504', 'MLF32-4504', 'MLF42-4504']  # These 272-channel recordings already lack MLF31-4504, MLO42-4504 and MRT52-4504.

bads_dict = {}

for sub in subjects:
    if sub in ['01', '02', '03']:
        current_bads = bads_group_1
    else:
        current_bads = bads_group_2
        
    # Use the subject's bad-channel list for both empty-room and task recordings.
    bads_dict[sub] = {
        'noise': current_bads,
        '01': current_bads,
        '02': current_bads,
        '03': current_bads,
        '04': current_bads
    }

for sub in subjects:
    print(f"Processing subject {sub}")
    
    sub_bads = bads_dict.get(sub, {'noise': [], '01': [], '02': [], '03': [], '04': []})
    
    # ==========================================
    # ==========================================
    noise_pattern = os.path.join(noise_root, f'Noise_{sub}*.ds')
    noise_files = glob.glob(noise_pattern)
    
    if not noise_files:
        print(f"Skipping {sub}: empty-room recording not found.")
        continue
        
    noise_path = noise_files[0]
    print(f"Computing empty-room projections from {os.path.basename(noise_path)}")
    
    noise = mne.io.read_raw_ctf(noise_path, preload=True)
    
    # Mark bad channels before estimating noise projections.
    noise.info['bads'] = sub_bads.get('noise', [])
    
    noise.filter(0.1, 40, fir_design='firwin')
    projs_empty = mne.compute_proj_raw(noise)
    
    del noise
    
    # ==========================================
    # ==========================================
    raw_list = []
    for bid in ['01', '02', '03', '04']:
        search_pattern = os.path.join(data_root, 'raw_data', sub, f'{sub}_G27IP_*_{bid}.ds')
        matched_files = glob.glob(search_pattern)
        
        if not matched_files:
            print(f"Missing task block {bid}.")
            continue
            
        path = matched_files[0]
        print(f"Processing block {bid}")
        
        # Retain the CTF compensation when reading task recordings.
        raw_tmp = mne.io.read_raw_ctf(path, preload=True)
        
        raw_tmp.info['bads'] = sub_bads.get(bid, [])
        
        raw_tmp.filter(0.1, 40, l_trans_bandwidth='auto', h_trans_bandwidth='auto', 
                       filter_length='auto', phase='zero-double', fir_design='firwin')
        
        # Apply projections estimated from this subject's empty-room recording.
        raw_tmp.add_proj(projs_empty)
        raw_tmp.apply_proj()
        
        raw_list.append(raw_tmp)
    
    if not raw_list:
        print(f"Skipping {sub}: no task blocks available.")
        continue
        
    # ==========================================
    # ==========================================
    print(f"Concatenating task blocks for {sub}")
    raw_combined = mne.concatenate_raws(raw_list, on_mismatch='warn')
    del raw_list
    
    # Save the concatenated task recording at its acquisition sampling rate.
    # raw_combined.resample(200, npad='auto')
    
    out_dir = os.path.join(out_root, '1_combined', f'sub-{sub}')
    os.makedirs(out_dir, exist_ok=True)
    
    save_path = os.path.join(out_dir, f'sub-{sub}_combined_raw.fif')
    # save_path = os.path.join(out_dir, f'sub-{sub}_combined_200Hz.fif')
    subject_info = dict(raw_combined.info.get("subject_info") or {})
    subject_info["his_id"] = sub
    raw_combined.info["subject_info"] = subject_info
    raw_combined.save(save_path, overwrite=True)
    
    # del raw_combined
    print(f"Saved the combined recording for {sub}.")

print("Finished combining task recordings.")


# Inspect the combined recording for muscle artifacts.
sub_id = '05'
path = f'{PREP_RESULTS}/1_combined/sub-{sub_id}/sub-{sub_id}_combined_raw.fif'
raw = mne.io.read_raw_fif(path, preload=True)

# Press A in the raw browser and mark muscle artifacts as BAD_muscle.
# Mark high-frequency muscle bursts; leave blinks for ICA review.
raw.plot(duration=10, n_channels=50, block=True)
raw.compute_psd(fmax=50).plot()
raw.plot_psd_topomap()
raw.plot_sensors(show_names=True)
raw.plot(duration=100, n_channels=50, scalings='auto')

# Annotations from the browser are stored in this Raw object's annotations.

# Inspect ICA components for the example subject.
# Use a 1-Hz high-pass copy to fit ICA.
raw_ica_train = raw.copy().filter(l_freq=1.0, h_freq=40.0, verbose=False)
raw_ica_train.info['bads'] =  ['MZF02-4504', 'MLF21-4504', 'MLF41-4504', 'MLF22-4504', 'MLF32-4504', 'MLF42-4504']
raw_ica_train.info['bads'] = []
# Exclude annotated bad segments from the ICA fit.
ica = mne.preprocessing.ICA(n_components=25, random_state=42, method='fastica')
ica.fit(raw_ica_train, reject_by_annotation=True)

# Inspect topographies for components confined to a single sensor.
ica.plot_components(inst=raw)

# Inspect component time courses for activity synchronized with blinks.
ica.plot_sources(raw)
## ============================================================================


# ---- runica ----
subjects = meg_subjects()

in_root = f'{PREP_RESULTS}/1_combined'
ica_out_root = f'{PREP_RESULTS}/2_ICA/run_ica'

os.makedirs(ica_out_root, exist_ok=True)
print("Starting batch ICA fitting")

for sub in subjects:
    print(f"Fitting ICA for {sub}")
    
    folder = Path(in_root) / f"sub-{sub}"
    raw_file = folder / f"sub-{sub}_combined_200Hz.fif"
    if not raw_file.exists():
        raw_file = folder / f"sub-{sub}_combined_raw.fif"
    ica_save_file = os.path.join(ica_out_root, f'sub-{sub}_ica.fif')
    
    if not os.path.exists(raw_file):
        print(f"Skipping {sub}: combined recording not found.")
        continue
        
    if os.path.exists(ica_save_file):
        print(f"Skipping {sub}: ICA fit already exists.")
        continue
    
    try:
        raw = mne.io.read_raw_fif(raw_file, preload=True)
        
        # Filter an in-memory copy for ICA training.
        print("Preparing a 1-Hz high-pass copy for ICA")
        raw_ica_train = raw.copy()
        if raw_ica_train.info["sfreq"] != 200:
            raw_ica_train.resample(200)
        raw_ica_train.filter(l_freq=1.0, h_freq=None, verbose=False)
        
        ica = mne.preprocessing.ICA(n_components=25, 
                                    random_state=42, 
                                    method='fastica',
                                    max_iter='auto')
        # ica = mne.preprocessing.ICA(n_components=0.95, 
        #                     random_state=42, 
        #                     method='fastica',
        #                     max_iter='auto')
        
        print("Fitting ICA")
        ica.fit(raw_ica_train)
        
        ica.save(ica_save_file, overwrite=True)
        print(f"Saved ICA fit for {sub}.")
        
        del raw
        del raw_ica_train
        del ica
        
    except Exception as e:
        print(f"ICA failed for {sub}: {e}")
        continue

print("Finished batch ICA fitting.")

# Inspect the trigger events.
raw_file = os.path.join(in_root, f'sub-{sub}', f'sub-{sub}_combined_raw.fif')
raw = mne.io.read_raw_fif(raw_file, preload=True)
events = mne.find_events(raw, stim_channel='UPPT001', shortest_event=1)