import pydicom
from pydicom import dcmread
import os
from os import listdir
from os.path import join
import numpy as np


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
    dicom_files.sort()

    # read the first file to get the image shape
    ref_metadata = pydicom.dcmread(join(case_path, dicom_files[0]))
    image_shape = (len(dicom_files), int(ref_metadata.Rows), int(ref_metadata.Columns))

    # create a 3D numpy array to store the images
    case_images = np.empty(image_shape, dtype=ref_metadata.pixel_array.dtype)

    # loop through all the DICOM files and read them into the numpy array
    for i, file in enumerate(dicom_files):
        file_path = os.path.join(case_path, file)
        metadata = dcmread(file_path)
        case_images[i, :, :] = metadata.pixel_array

    return case_images
