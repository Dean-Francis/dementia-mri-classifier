import os
from pathlib import Path
import nibabel as nib 
import numpy as np
from PIL import Image
from typing import Any
from collections import Counter

def extract_slices(data_root: Path, output_dir: Path, cdr_threshold=0) -> Counter:
    output_dir.joinpath("CN").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("AD").mkdir(parents=True, exist_ok=True)

    slice_count = Counter({'CN': 0, 'AD': 0})

    subject_folders = sorted(data_root.glob('OAS1_*_MR1'))
    print(f"Found {len(subject_folders)} subjects")

    for subject_folder in subject_folders:
        subject_id = subject_folder.name

        txt_file = subject_folder / f"{subject_id}.txt"

        if not txt_file.exists():
            print(f"Text file does not exist for {subject_id}")
            continue

        cdr = None
        with open(txt_file, 'r') as f:
            for line in f:
                if 'CDR: ' in line:
                    cdr_str = line.split(':')[1].strip()
                    if cdr_str:  # Make sure it's not empty
                        cdr = float(cdr_str)
                    break
        
        if cdr is None:
            print(f"Skipping {subject_id}: CDR not found")
            continue

        # CDR = Clinical Dementia Rating
        # cdr = 0 -> Control/Normal
        # cdr = 0.5 -> Very Mild Dementia (AD)
        # cdr = 1.0 -> Mild Demntia (AD)
        # cdr = 2.0 -> Moderate Demntia (AD)
        # cdr = 3.0 -> Severe Dementia (AD)
        label = 'CN' if cdr <= cdr_threshold else 'AD'

        print(f"Processing {subject_id} CDR: {cdr}, label: {label}")

        subject_slices_folder = output_dir / label / subject_id
        # CACHING Check if we already processed this subject
        #TODO: Redo Caching
        
        
        if subject_slices_folder.exists():
            slices_count = sum(1 for _ in subject_slices_folder.iterdir())
            print(f"Already processed ({slices_count} slices found) - skipping")
            slice_count[label] += slices_count
            continue
        
        subject_slices_folder.mkdir(parents=True)


        # Loading the Brain Scans
        img_file = subject_folder / 'PROCESSED' / 'MPRAGE' / 'T88_111'
        img_files = list(img_file.glob('*t88_gfc.hdr'))

        if not img_files:
            print(f"Image file cannot be found from {subject_folder}")
            continue

        try:
            # Load the MRI Scan
            img_data = nib.load(img_files[0])
            volume: np.ndarray = img_data.get_fdata()
            
            # Remove Singleton Dimensions
            volume = np.squeeze(volume)

            print("Brain succesfully loaded. Extracting slices...")
            if volume.min() == volume.max():
                print(f"Empty Brain Scan: Skipping {subject_id}")
                continue

            volume = (volume - volume.min()) / (volume.max() - volume.min())
            volume = (volume * 255).astype(np.uint8)

            num_slices = volume.shape[2]
            slice_min = int(num_slices * 0.35)
            slice_max = int(num_slices * 0.68)
            # Extract each 2D slice
            for slice_idx in range(slice_min, slice_max+1):

                slice_2d = volume[:,:,slice_idx]

                filepath = subject_slices_folder / f"{subject_id}_slice{slice_idx}.png"
                
                # Save as PNG
                img_pil = Image.fromarray(slice_2d, mode='L')
                img_pil.save(filepath)
                
                slice_count[label] += 1
                
    

        except Exception as e:
            raise
            continue

    # Print final summary
    print(f"\n{'='*50}")
    print(f"Extraction Complete!")
    print(f"CN slices: {slice_count['CN']}")
    print(f"AD slices: {slice_count['AD']}")
    print(f"Total: {slice_count['CN'] + slice_count['AD']}")
    print(f"{'='*50}\n")
    return slice_count
