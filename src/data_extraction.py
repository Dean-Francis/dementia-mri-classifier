import os
from pathlib import Path
import nibabel as nib
import numpy as np
from PIL import Image

def extract_slices(data_root, output_dir, cdr_threshold=0, disc_max=None, disc_min=None):
    """
    Extract MRI slices from OASIS dataset.

    Args:
        data_root: Path to dataset root (e.g., ../dataset) - will find all disc1, disc2, etc.
        output_dir: Directory to save extracted slices
        cdr_threshold: CDR threshold for CN/AD classification
        disc_max: Maximum disc number to include (e.g., disc_max=11 uses disc1-disc11, skips disc12+)
        disc_min: Minimum disc number to include (e.g., disc_min=12 uses only disc12+)
    """
    os.makedirs(os.path.join(output_dir, 'CN'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'AD'), exist_ok=True)

    slice_count = {'CN': 0, 'AD': 0}

    # Find all disc directories (disc1, disc2, disc3, etc.)
    data_root_path = Path(data_root)
    disc_dirs = sorted([d for d in data_root_path.glob('disc*') if d.is_dir()])

    if not disc_dirs:
        # If no disc folders found, treat the input as a single disc
        disc_dirs = [data_root_path]

    # Filter discs by min/max number if specified
    if disc_min is not None or disc_max is not None:
        filtered_discs = []
        for disc_dir in disc_dirs:
            try:
                disc_num = int(disc_dir.name[4:])  # Extract number from 'disc1', 'disc2', etc.
                within_range = True
                if disc_min is not None and disc_num < disc_min:
                    within_range = False
                if disc_max is not None and disc_num > disc_max:
                    within_range = False
                if within_range:
                    filtered_discs.append(disc_dir)
            except (ValueError, IndexError):
                # If can't parse disc number, include it
                filtered_discs.append(disc_dir)
        disc_dirs = filtered_discs

    print(f"Found {len(disc_dirs)} disc(s): {[d.name for d in disc_dirs]}")

    for disc_dir in disc_dirs:
        print(f"\nProcessing {disc_dir.name}...")
        subject_folders = sorted(disc_dir.glob('OAS1_*_MR1'))
        print(f"  Found {len(subject_folders)} subjects")

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

            # CACHING Check if we already processed this subject
            output_folder = os.path.join(output_dir, label)
            existing_slices = [f for f in os.listdir(output_folder) if f.startswith(subject_id)]

            if existing_slices:
                print(f"Already processed ({len(existing_slices)} slices found) - skipping")
                slice_count[label] += len(existing_slices)
                continue

            # Loading the Brain Scans
            img_file = subject_folder / 'PROCESSED' / 'MPRAGE' / 'SUBJ_111'
            img_files = list(img_file.glob('*.img'))

            if not img_files:
                print(f"Image file cannot be found from {subject_folder}")
                continue

            try:
                # Load the MRI Scan
                img_data = nib.load(str(img_files[0]))
                volume = img_data.get_fdata()

                # Remove Singleton Dimensions
                volume = np.squeeze(volume)

                print("Brain succesfully loaded. Extracting slices...")

                volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
                volume = (volume * 255).astype(np.uint8)

                num_slices = volume.shape[2]

                # Extract each 2D slice
                for slice_idx in range(num_slices):
                    slice_2d = volume[:,:,slice_idx]

                    # Quality filtering: keep only slices with meaningful brain tissue
                    mean_intensity = np.mean(slice_2d)
                    std_intensity = np.std(slice_2d)

                    # Skip if:
                    # - Too dark (mean < 30): mostly air/skull edges with minimal brain tissue
                    # - Too bright (mean > 180): overexposed or artifact
                    # - No contrast (std < 20): uniform slice with no useful information
                    if mean_intensity < 30 or mean_intensity > 180 or std_intensity < 20:
                        continue

                    filename = f"{subject_id}_slice{slice_idx:03d}.png"
                    filepath = os.path.join(output_dir, label, filename)

                    # Save as PNG
                    img_pil = Image.fromarray(slice_2d, mode='L')
                    img_pil.save(filepath)

                    slice_count[label] += 1

            except Exception as e:
                print(e)
                continue

    # Print final summary
    print(f"\n{'='*50}")
    print(f"Extraction Complete!")
    print(f"CN slices: {slice_count['CN']}")
    print(f"AD slices: {slice_count['AD']}")
    print(f"Total: {slice_count['CN'] + slice_count['AD']}")
    print(f"{'='*50}\n")
    return slice_count