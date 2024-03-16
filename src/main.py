# %%
import torch
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import os
from utils import (
    load_segmentation_data,
    CustomDataset,
    create_paired_data,
    save_metrics,
    load_image_data,
)
from models import SimpleUNet
from train import train_model
import random
import datetime
import json

# %%
current_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(current_directory)
image_path = os.path.join(parent_directory, "Dataset\Images")
seg_path = os.path.join(parent_directory, "Dataset\Segmentations")

# Load the data
case_arrays = load_image_data(image_path)
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

# Create paired data
train_paired_data = create_paired_data(train_cases, case_arrays, seg_arrays)
test_paired_data = create_paired_data(test_cases, case_arrays, seg_arrays)

# Create datasets
train_dataset = CustomDataset(train_paired_data)
test_dataset = CustomDataset(test_paired_data)

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False)

model = SimpleUNet(in_channels=1, out_channels=1)

# %%
# Training -------------------------------------------------------
losses, train_accuracies, test_accuracies = train_model(
    model,
    train_loader,
    test_loader,
    epochs=10,
    learning_rate=0.1,
    dice_threshold=0.0025 * 512 * 512,
    bce_weight=0.7,
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
    "train_cases": train_cases,
    "test_cases": test_cases,
    "bce_weight": 0.7,
}


def save_metrics(metrics, directory, timestamp):
    """
    Saves the provided metrics, including train and test cases, as a JSON file in the specified directory.

    Args:
        metrics (dict): A dictionary containing the metrics to save.
                        Expected to have keys like 'losses', 'train_accuracies', 'test_accuracies',
                        'train_cases', and 'test_cases'.
        directory (str): Path to the directory where the JSON file will be saved.
        timestamp (str): Timestamp to append to the filename for uniqueness.
    """
    metrics_filename = f"metrics_{timestamp}_new.json"
    metrics_path = os.path.join(directory, metrics_filename)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")


save_metrics(metrics, saved_models_dir, timestamp)

# Save the metrics to a JSON file
metrics_filename = f"metrics_{timestamp}.json"
metrics_path = os.path.join(saved_models_dir, metrics_filename)

with open(metrics_path, "w") as f:
    json.dump(metrics, f)

print(f"Metrics saved to {metrics_path}")
