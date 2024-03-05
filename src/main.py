# %%
import matplotlib.pyplot as plt
import pydicom
from pydicom import dcmread
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from utils import convert_dicom_to_numpy

# %%
# load one dicom file
# load path that is one level above the current directory

current_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(current_directory)

image_path = os.path.join(parent_directory, "Dataset\Images")
segmenation_path = os.path.join(parent_directory, "Dataset\Segmentations")


# %%
case_folders = [
    d for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))
]

# Dictionary to store numpy arrays for each case
case_arrays = {}

# Loop through each case and convert to numpy array
for case in case_folders:
    case_path = os.path.join(image_path, case)
    case_arrays[case] = convert_dicom_to_numpy(case_path)
    print(f"Converted {case} to numpy array with shape {case_arrays[case].shape}")

# %%
# Display the first slice of the first case
plt.imshow(case_arrays["Case_000"][50, :, :], cmap="gray")
