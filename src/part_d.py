# %%
import os
import torch
import json
import matplotlib.pyplot as plt
from models import SimpleUNet
import matplotlib

from utils import (
    load_segmentation_data,
    CustomDataset,
    create_paired_data,
    save_metrics,
    load_image_data,
    generate_seg_preds,
)

# %%

# Define the current directory and parent directory
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)

# Construct paths to the saved model and metrics
model_filename = "unet_lr0.1_epochs2_bs3_trainacc1.00_testacc0.99_20240311-213847.pth"
metrics_filename = "metrics_20240311-213847.json"

model_path = os.path.join(parent_directory, "saved_models", model_filename)
metrics_path = os.path.join(parent_directory, "saved_models", metrics_filename)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleUNet(in_channels=1, out_channels=1)
model.load_state_dict(torch.load(model_path))
model.to(device)

# load the cases
image_path = os.path.join(parent_directory, "Dataset\Images")
seg_path = os.path.join(parent_directory, "Dataset\Segmentations")

# Load the data
case_arrays = load_image_data(image_path)
seg_arrays = load_segmentation_data(seg_path)

# %%
case_names = [
    d for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))
]

seg_preds = generate_seg_preds(model, case_arrays, case_names, device, threshold=0.5)

# %%
