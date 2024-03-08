import pydicom
from pydicom import dcmread
import os
from os import listdir
from os.path import join
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


def convert_dicom_to_numpy(case_path):
    """
    Convert a series of DICOM files in a given directory into a 3D NumPy array.

    This function reads all DICOM (.dcm) files located in the specified directory,
    sorts them, and stacks their pixel data into a 3D NumPy array. The DICOM files
    should represent slices of a single scan, and they are sorted alphabetically
    to maintain the correct sequence.

    @param case_path: The path to the directory containing DICOM files. All files
                      in this directory with a .dcm extension are processed.
    @return: A 3D NumPy array containing the pixel data from the DICOM files.
             The array's shape is (number_of_slices, rows, columns), where
             number_of_slices is the number of DICOM files, and rows and columns
             are the dimensions of the DICOM images.
    @raise FileNotFoundError: If the specified directory does not exist or is
                              inaccessible.
    @raise ValueError: If there are no .dcm files in the given directory.

    Example usage:
    ```
    numpy_array = convert_dicom_to_numpy('/path/to/dicom/directory')
    ```
    """
    # list all DICOM  files in the directory
    dicom_files = [f for f in listdir(case_path) if f.endswith(".dcm")]
    dicom_files.sort()  # Ensure files are in the correct order

    # read the first file to get the image shape
    ref_metadata = dcmread(join(case_path, dicom_files[0]))
    image_shape = (len(dicom_files), int(ref_metadata.Rows), int(ref_metadata.Columns))

    # create a 3D numpy array to store the images
    case_images = np.empty(image_shape, dtype=ref_metadata.pixel_array.dtype)

    # loop through all the DICOM files and read them into the numpy array
    for i, file in enumerate(dicom_files):
        file_path = join(case_path, file)
        metadata = dcmread(file_path)
        case_images[i, :, :] = metadata.pixel_array

    return case_images


def convert_dicom_to_numpy_2(case_path):
    # list all DICOM files in the directory
    dicom_files = [f for f in listdir(case_path) if f.endswith(".dcm")]

    # Sort the files based on the numerical value after the hyphen
    dicom_files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))

    # Read the first file to get the image shape
    ref_metadata = dcmread(join(case_path, dicom_files[0]))
    image_shape = (len(dicom_files), int(ref_metadata.Rows), int(ref_metadata.Columns))

    # create a 3D numpy array to store the images
    case_images = np.empty(image_shape, dtype=ref_metadata.pixel_array.dtype)

    # loop through all the DICOM files and read them into the numpy array
    for i, file in enumerate(dicom_files):
        file_path = join(case_path, file)
        metadata = dcmread(file_path)
        case_images[i, :, :] = metadata.pixel_array

    return case_images


def load_segmentation_data(segmentation_path):
    segmentation_arrays = {}

    # List all .npz files in the directory
    npz_files = [f for f in listdir(segmentation_path) if f.endswith(".npz")]
    print("Loading segmentation data...")
    for file_name in tqdm(npz_files):
        case_name = os.path.splitext(file_name)[0].split("_seg")[
            0
        ]  # Remove the "_seg" so that the case name matches the image data
        npz_file_path = join(segmentation_path, file_name)

        # Load the numpy array from the .npz file
        with np.load(npz_file_path) as data:
            # Use the 'masks' key to access the data
            if "masks" in data:
                segmentation_arrays[case_name] = data["masks"]

            else:
                print(f"Key 'masks' not found in {npz_file_path}")
    print("Segmentation data loaded.")
    return segmentation_arrays


def convert_dicom_to_numpy_slice_location(case_path):
    # List all DICOM files in the directory
    dicom_files = [f for f in listdir(case_path) if f.endswith(".dcm")]

    # Read SliceLocation tag for each file and store in a list
    dicom_metadata = []
    for file in dicom_files:
        file_path = join(case_path, file)
        metadata = dcmread(file_path)
        slice_location = getattr(metadata, "SliceLocation", None)
        # print(f"SliceLocation: {slice_location}")
        if slice_location is not None:
            dicom_metadata.append((file, slice_location))
        else:
            print(f"SliceLocation not found in {file}")

    # Sort the list based on the slice location
    # (important because masks are sorted by slice location as well)
    dicom_metadata.sort(key=lambda x: x[1])

    # Create a 3D numpy array to store the images
    if dicom_metadata:
        num_slices = len(dicom_metadata)
        ref_metadata = dcmread(join(case_path, dicom_metadata[0][0]))
        image_shape = (num_slices, int(ref_metadata.Rows), int(ref_metadata.Columns))
        case_images = np.empty(image_shape, dtype=ref_metadata.pixel_array.dtype)

        # Load the images in the sorted order
        for i, (file, _) in enumerate(dicom_metadata):
            file_path = join(case_path, file)
            metadata = dcmread(file_path)
            case_images[i, :, :] = metadata.pixel_array
    else:
        raise ValueError("No DICOM files with SliceLocation found.")

    return case_images


class CustomDataset(Dataset):
    def __init__(self, paired_data):
        self.paired_data = paired_data

    def __len__(self):
        return len(self.paired_data)

    def __getitem__(self, idx):
        image, mask = self.paired_data[idx]

        # Convert numpy arrays to PyTorch tensors
        image_tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(
            0
        )  # Adding channel dimension
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        return image_tensor, mask_tensor
