import os
import sys
import pytest
import numpy as np
import torch


# Get the directory of the 'tests' folder
# Get the parent directory (project root)
# Append the project root to sys.path
tests_dir = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(tests_dir)
sys.path.append(parent_directory)

# load the cases
path_to_images = os.path.join(parent_directory, "Dataset\\Images")
seg_path = os.path.join(parent_directory, "Dataset\\Segmentations")


# # Load the data
# case_arrays = load_image_data(path_to_images)
# seg_arrays = load_segmentation_data(seg_path)

from src.utils import (
    load_image_data,
    load_segmentation_data,
    convert_dicom_to_numpy_slice_location,
    create_paired_data,
    dice_coeff,
)


def test_load_image_data_returns_dict():
    image_path = path_to_images
    result = load_image_data(image_path)
    assert isinstance(result, dict), "The function should return a dictionary"


def test_load_image_data_contains_case_keys():
    image_path = path_to_images
    result = load_image_data(image_path)
    assert all(
        isinstance(case, str) for case in result.keys()
    ), "All keys should be case names"


def test_load_image_data_values_are_numpy_arrays():
    image_path = path_to_images
    result = load_image_data(image_path)
    assert all(
        isinstance(arr, np.ndarray) for arr in result.values()
    ), "All values should be numpy arrays"


def test_load_image_data_handles_nonexistent_directory():
    image_path = "non/existent/path"
    with pytest.raises(Exception):
        _ = load_image_data(image_path)


def test_load_segmentation_data_nonexistent_directory():
    segmentation_path = "non/existent/directory"
    with pytest.raises(Exception):
        load_segmentation_data(segmentation_path)


def test_load_segmentation_data_keys_format():
    segmentation_path = seg_path
    result = load_segmentation_data(segmentation_path)
    assert all(
        "_seg" not in key for key in result.keys()
    ), "Keys should not contain '_seg'"


def generate_test_data(num_cases, num_slices, img_dim):
    case_arrays = {}
    segmentation_data = {}
    cases = [f"Case_{i:03}" for i in range(num_cases)]

    for case in cases:
        case_arrays[case] = np.random.rand(num_slices, img_dim, img_dim)
        segmentation_data[case] = np.random.randint(
            0, 2, (num_slices, img_dim, img_dim)
        )

    return cases, case_arrays, segmentation_data


@pytest.fixture  # Need this to pass this function first before the other functions
def test_data():
    return generate_test_data(5, 10, 256)  # 5 cases, 10 slices each, 256x256 images


def test_return_type(test_data):
    cases, case_arrays, segmentation_data = test_data
    result = create_paired_data(cases, case_arrays, segmentation_data)
    assert isinstance(result, list), "Function should return a list"


def test_pairing_correctness(test_data):
    cases, case_arrays, segmentation_data = test_data
    result = create_paired_data(cases, case_arrays, segmentation_data)
    for img, seg in result:
        assert (
            img.shape == seg.shape
        ), "Image and segmentation mask should have the same shape"


def test_perfect_match():
    pred = torch.tensor([[1, 1], [1, 1]])
    true = torch.tensor([[1, 1], [1, 1]])
    assert dice_coeff(pred, true) == pytest.approx(
        1.0
    ), "Dice coefficient should be 1 for a perfect match"


def test_no_overlap_with_threshold_handling():
    pred = torch.tensor([[1, 1], [1, 1]])
    true = torch.tensor([[0, 0], [0, 0]])
    threshold = 10
    assert dice_coeff(pred, true, threshold=threshold) == pytest.approx(
        1.0
    ), "Dice coefficient should be 1 due to threshold handling for no overlap"


def test_empty_masks():
    pred = torch.tensor([[0, 0], [0, 0]])
    true = torch.tensor([[0, 0], [0, 0]])
    assert dice_coeff(pred, true) == pytest.approx(
        1.0
    ), "Dice coefficient should be 1 for empty masks"


def test_threshold_effect():
    pred = torch.tensor([[1, 0], [0, 0]])
    true = torch.tensor([[0, 0], [0, 0]])
    threshold = 10
    assert dice_coeff(pred, true, threshold=threshold) == pytest.approx(
        1.0
    ), "Dice coefficient should be 1 due to threshold effect"


def test_data_types():
    pred = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    true = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)
    assert isinstance(
        dice_coeff(pred, true), torch.Tensor
    ), "Return type should be a torch tensor"
