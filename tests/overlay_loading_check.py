# %% In this script we manually check the mask and image overlay for each case
import os
import sys

current_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

import matplotlib.pyplot as plt
from src.utils import (
    load_segmentation_data,
    convert_dicom_to_numpy_slice_location,
)

# Case to be checked and the interval to check
case = "Case_006"
interval = 10
# %%


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
for slice in range(0, 122, interval):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(segmentation_data[case][slice], cmap="jet")
    axs[0].imshow(case_arrays[case][slice], cmap="gray", alpha=0.5)
    axs[0].set_title("Overlay")

    axs[1].imshow(case_arrays[case][slice], cmap="gray")
    axs[1].set_title(f"Image slice {slice} of {case}")
    plt.show()
