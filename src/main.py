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
    create_paired_data,
)
from models import SimpleUNet
from train import train_model
import random
import datetime
import json

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

seg_arrays = load_segmentation_data(seg_path)

# %% Train-test split
# First split the case_folders lists into training and test sets
case_folders = [
    d for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))
]
torch.manual_seed(25101999)
random.Random(25101999).shuffle(case_folders)  # fixing seed for reproducibility

# Calculate the number of cases for training
num_train_cases = int(2 / 3 * len(case_folders))

# Split the cases into training and test sets
train_cases = case_folders[:num_train_cases]
test_cases = case_folders[num_train_cases:]

train_paired_data = create_paired_data(train_cases, case_arrays, seg_arrays)
test_paired_data = create_paired_data(test_cases, case_arrays, seg_arrays)

# Create datasets
train_dataset = CustomDataset(train_paired_data)
test_dataset = CustomDataset(test_paired_data)

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False)

model = SimpleUNet(in_channels=1, out_channels=1)

# Training -------------------------------------------------------
losses, train_accuracies, test_accuracies = train_model(
    model, train_loader, test_loader, epochs=10, learning_rate=0.1
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
lr = 0.1  # Learning rate
epochs = 2  # Number of epochs
batch_size = 3  # Batch size
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
final_train_accuracy = train_accuracies[-1]
final_test_accuracy = test_accuracies[-1]

model_filename = f"unet_lr{lr}_epochs{epochs}_bs{batch_size}_trainacc{final_train_accuracy:.2f}_testacc{final_test_accuracy:.2f}_{timestamp}.pth"

# Saving the model in the 'saved_models' directory
saved_models_dir = os.path.join(parent_directory, "saved_models")
os.makedirs(saved_models_dir, exist_ok=True)  # Create the directory if it doesn't exist

model_save_path = os.path.join(saved_models_dir, model_filename)
torch.save(model.state_dict(), model_save_path)

print(f"Model saved to {model_save_path}")

# %%
# Convert the metrics to a dictionary
metrics = {
    "losses": losses,
    "train_accuracies": train_accuracies,
    "test_accuracies": test_accuracies,
}

# Save the metrics to a JSON file
metrics_filename = (
    f"metrics_{timestamp}.json"  # Use the same timestamp as for your model
)
metrics_path = os.path.join(saved_models_dir, metrics_filename)

with open(metrics_path, "w") as f:
    json.dump(metrics, f)

print(f"Metrics saved to {metrics_path}")
