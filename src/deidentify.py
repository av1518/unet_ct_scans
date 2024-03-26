"""
@file deidentify.py
@brief Anonymizes DICOM files by modifying patient-related tags.

This script iterates over specified case IDs in a given dataset directory,
reads each DICOM file, and replaces patient ID and name with the case ID,
and clears the patient's birth date. Patient birth time is removed if present.
All changes are saved back to the file.
"""

# %%
from pydicom import dcmread
import os
from os import listdir
from os.path import join


# Define the current directory and parent directory
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
dataset_path = os.path.join(parent_directory, "Dataset\Images")

case_ids = ["Case_003", "Case_006", "Case_007"]
for case_id in case_ids:
    subdir_path = join(dataset_path, case_id)
    for file_name in listdir(subdir_path):
        dicom_file = join(subdir_path, file_name)
        metadata = dcmread(dicom_file)
        # Modify the tags that contain patient information
        metadata["PatientID"].value = case_id
        metadata["PatientName"].value = case_id
        metadata["PatientBirthDate"].value = ""
        # PatientBirthTime is optional, it might not be present
        try:
            del metadata["PatientBirthTime"]
        except:
            pass
        metadata.save_as(dicom_file)
