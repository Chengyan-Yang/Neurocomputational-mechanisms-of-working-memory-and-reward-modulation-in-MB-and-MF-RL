# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:26:03 2026

@author: Cyan
"""

# -*- coding: utf-8 -*-
"""
Coregister MEG digitization points with the subject's MRI.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meg_project import DATA_ROOT, meg_subjects, PREP_RESULTS, SOURCE_RESULTS

import mne
import os

# ==========================================
# ==========================================
subjects_dir = f'{DATA_ROOT}/MRI/surf' 

# Use the preprocessed MEG recording for its digitization points.
meg_data_root = f'{PREP_RESULTS}/5_epochs_resampled'



subjects = meg_subjects()

# ==========================================
# ==========================================
print("Initializing MEG-MRI coregistration")

for sub in subjects:
    sub_mri_id = f'{sub}'  # FreeSurfer subject directory
    sub_meg_id = f'sub-{sub}'  # MEG subject directory
    
    meg_file = os.path.join(meg_data_root, sub_meg_id, f'{sub_meg_id}_incentive_resamp200_epo.fif')
    
    if not os.path.exists(meg_file):
        print(f"Skipping {sub}: MEG recording not found.")
        continue
    
    print(f"Opening coregistration for {sub}")
    print("Align the surfaces and digitization points, then click Save to write the transform.")
    print("Close the window to continue to the next subject.")
    
    trans_dir = SOURCE_RESULTS / "trans"
    trans_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save the transform as {trans_dir / f'sub-{sub}-trans.fif'}")
    mne.gui.coregistration(
        subject=sub_mri_id, 
        subjects_dir=subjects_dir, 
        inst=meg_file,
        block=True  # Wait for this subject's window to close.
    )

print("Finished the coregistration batch.")