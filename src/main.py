# %%
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

# %%
# load one dicom file
# load path that is one level above the current directory

current_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(current_directory)

image_path = os.path.join(parent_directory, "Dataset\Images")
seg_path = os.path.join(parent_directory, "Dataset\Segmentations")


# %%
case_folders = [
    d for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))
]

# Dictionary to store numpy arrays for each case
case_arrays = {}

# Loop through each case and convert to numpy array
for case in case_folders:
    case_path = os.path.join(image_path, case)
    case_arrays[case] = convert_dicom_to_numpy_2(case_path)
    print(f"Converted {case} to numpy array with shape {case_arrays[case].shape}")

# %%
# Display the first slice of the first case
plt.imshow(case_arrays["Case_010"][85], cmap="gray")

# %%


npz_filename = (
    "Case_000_seg.npz"  # Replace with the name of an actual .npz file in the directory
)
npz_file_path = os.path.join(seg_path, npz_filename)

try:
    with np.load(npz_file_path) as data:
        print("Arrays in the .npz file:", list(data.keys()))
except Exception as e:
    print(f"An error occurred: {e}")
# %%

segmentation_data = load_segmentation_data(seg_path)
# %%
print(segmentation_data.keys())
# plot slice 42 of the first case
fig, axs = plt.subplots(1, 2)

slice = 50

axs[0].imshow(segmentation_data["Case_010_seg"][-slice, :, :], cmap="jet")
axs[0].set_title("Segmentation")

axs[1].imshow(case_arrays["Case_010"][slice, :, :], cmap="gray")
axs[1].set_title("Image")

plt.show()

# %%
# print slice number of each case
for case in case_folders:
    print(f"{case}: {case_arrays[case].shape[0]} slices")

# print slice number of each case in the segmentation data
for case in segmentation_data:
    print(f"{case}_seg: {segmentation_data[case].shape[0]} slices")

# overlay the segmentation on the image
# %%


# plot every slice overlayed with mask number
for slice in range(0, 122, 1):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(segmentation_data["Case_005_seg"][-slice], cmap="jet")
    axs[0].imshow(case_arrays["Case_005"][slice - 1], cmap="gray", alpha=0.5)
    axs[0].set_title("Overlay")

    axs[1].imshow(case_arrays["Case_005"][slice - 1], cmap="gray")
    axs[1].set_title(f"Image slice {slice} of")
    plt.show()

# %%
fig, axs = plt.subplots(1, 2)

case = "Case_007"

axs[0].imshow(segmentation_data["Case_007_seg"][-slice, :, :], cmap="jet")
axs[0].imshow(case_arrays[case][slice, :, :], cmap="gray", alpha=0.5)
axs[0].set_title("Overlay")

axs[1].imshow(case_arrays[case][slice, :, :], cmap="gray")
axs[1].set_title(f"Image slice {slice} of Case_007")

plt.show()
