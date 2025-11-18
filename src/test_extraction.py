from data_extraction import extract_slices

# Test on your dataset
data_root = r'C:\Users\user\Desktop\Year 4\Data Mining\dataset\disc1'
output_dir = r'C:\Users\user\Desktop\Year 4\Data Mining\dataset\processed\oasis_slices'

print("Starting extraction test...\n")

# Run the extraction
slice_count = extract_slices(data_root, output_dir, cdr_threshold=0)

print("\nTest complete!")
print(f"Final counts: {slice_count}")