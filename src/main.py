# %%
import torch
from torch.utils.data import DataLoader, random_split
import torch
import matplotlib.pyplot as plt
import os
from utils import (
    load_segmentation_data,
    convert_dicom_to_numpy_slice_location,
    CustomDataset,
)
from models import SimpleUNet
from train import train_model

# %% Load the data
current_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(current_directory)
image_path = os.path.join(parent_directory, "Dataset\Images")
seg_path = os.path.join(parent_directory, "Dataset\Segmentations")

case_folders = [
    d for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))
]

# Dictionary to store numpy arrays for each case
case_arrays = {}

# Loop through each case and convert to numpy array
print("Loading images...")
for case in case_folders:
    case_path = os.path.join(image_path, case)
    case_arrays[case] = convert_dicom_to_numpy_slice_location(case_path)
    print(f"Converted {case} to numpy array with shape {case_arrays[case].shape}")
print("Images loaded.")

segmentation_data = load_segmentation_data(seg_path)
# %%
# Creating a list of paired data in the format (image, segmentation) and putting them all in this list to create a dataset
paired_data = []
for case in case_folders:
    image_array = case_arrays[case]
    segmentation_array = segmentation_data[case]

    # Check if the case exists in both image and segmentation data
    if case in segmentation_data:
        for i in range(image_array.shape[0]):  # Loop through each slice
            paired_data.append((image_array[i], segmentation_array[i]))


full_dataset = CustomDataset(paired_data)
# %%
# Define random seed for reproducibility
torch.manual_seed(25101999)

# Calculate sizes for train and test sets
train_size = int(2 / 3 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(25101999),
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset, batch_size=3, shuffle=True
)  # adjust batch_size as needed
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False)

model = SimpleUNet(
    in_channels=1, out_channels=1
)  # 1 input channel (grayscale), 1 output channel (binary mask)

# Train the model
losses, train_accuracies, test_accuracies = train_model(
    model, train_loader, test_loader, epochs=1, learning_rate=0.1
)

# %%
# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses, label="Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy", color="orange")
plt.plot(test_accuracies, label="Test Accuracy", color="green")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# %%
# Save the model

torch.save(model.state_dict(), "unet_model.pth")
