from data_extraction import extract_slices
from pathlib import Path

# Test on your dataset
root_dir = Path("dataset")
data_root = root_dir / "disc1"
output_dir = root_dir / "processed" / "oasis_slices"

print("Starting extraction test...\n")

# Run the extraction
slice_count = extract_slices(data_root, output_dir, cdr_threshold=0)

print("\nTest complete!")
print(f"Final counts: {slice_count}")

