import nibabel as nib
import numpy as np

# Load one brain scan
img_path = r'C:\Users\user\Desktop\Data Mining\dataset\disc1\OAS1_0001_MR1\PROCESSED\MPRAGE\SUBJ_111\OAS1_0001_MR1_mpr_n4_anon_sbj_111.img'

# Load the file
img_data = nib.load(img_path)
volume = img_data.get_fdata()

print(f"Original shape: {volume.shape}")

# Squeeze out the singleton dimension
volume = np.squeeze(volume)  # Removes dimensions of size 1

print(f"After squeeze: {volume.shape}")
print(f"Data type: {volume.dtype}")
print(f"Min value: {volume.min():.2f}")
print(f"Max value: {volume.max():.2f}")

if len(volume.shape) == 3:
    print(f"\n✓ YES! 3D brain scan: {volume.shape[0]} x {volume.shape[1]} x {volume.shape[2]}")
    print(f"  We can extract {volume.shape[2]} slices!")