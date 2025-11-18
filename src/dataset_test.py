from dataset import MRIDataset

data_path = r'C:\Users\user\Desktop\Year 4\Data Mining\dataset\processed\oasis_slices'

print("Creating Dataset")
dataset = MRIDataset(data_path)

print(f"Dataset length: {len(dataset)}")

print("Test __getitem()__")
image, label = dataset[0]
print(f"Image type: {type(image)}")
print(f"Image size: {image.size}")
print(f"Label: {label} ({'CN' if label == 0 else 'AD'})")

# INCASE
print(f"\nTesting a few more images:")
for i in [10, 100, 1000, 2000, 3000]:
    image, label = dataset[i]
    print(f"Image {i}: size={image.size}, label={label} ({'CN' if label == 0 else 'AD'})")