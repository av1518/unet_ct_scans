# %% In this script we manually check the mask and image overlay for each case
import matplotlib.pyplot as plt
import pydicom
from pydicom import dcmread
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from utils import (
    convert_dicom_to_numpy,
    load_segmentation_data,
    convert_dicom_to_numpy_2,
    convert_dicom_to_numpy_slice_location,
)
from models import SimpleUNet

# %%

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
# plot every slice overlayed with mask number

case = "Case_006"
for slice in range(0, 122, 10):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(segmentation_data[case][slice], cmap="jet")
    axs[0].imshow(case_arrays[case][slice], cmap="gray", alpha=0.5)
    axs[0].set_title("Overlay")

    axs[1].imshow(case_arrays[case][slice], cmap="gray")
    axs[1].set_title(f"Image slice {slice} of {case}")
    plt.show()

# %%
plt.imshow(segmentation_data[case][42], cmap="jet")
print(np.unique(segmentation_data[case][42]))

# %%
model = SimpleUNet(in_channels=1, out_channels=1)

# print parameters
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
