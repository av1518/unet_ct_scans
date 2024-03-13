import pydicom
from pydicom import dcmread
import os
from os import listdir
from os.path import join
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import json
from torchmetrics.classification import BinaryAccuracy


def load_image_data(image_path):
    case_folders = [
        d for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))
    ]

    # Dictionary to store numpy arrays for each case
    case_arrays = {}

    # Loop through each case and convert to numpy array
    print("Loading images...")
    for case in tqdm(case_folders):
        case_path = os.path.join(image_path, case)
        case_arrays[case] = convert_dicom_to_numpy_slice_location(case_path)
        # print(f"Converted {case} to numpy array with shape {case_arrays[case].shape}")
    print("Images loaded.")

    return case_arrays


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


def create_paired_data(cases, case_arrays, segmentation_data):
    """
    Creates a list of paired data, where each pair consists of an image and its corresponding segmentation mask.

    This function iterates through the given cases, retrieves corresponding image and segmentation arrays,
    and pairs individual slices from these arrays. The pairing is done based on the index of the slices,
    ensuring that each image slice is paired with its corresponding segmentation mask slice.

    Args:
        cases (list): A list of case identifiers. Each identifier corresponds to a key in case_arrays and segmentation_data.
        case_arrays (dict): A dictionary where keys are case identifiers and values are image arrays (as NumPy arrays).
        segmentation_data (dict): A dictionary where keys are case identifiers and values are segmentation mask arrays (as NumPy arrays).

    Returns:
        list of tuples: A list where each tuple contains an image slice and its corresponding segmentation mask slice, both as NumPy arrays.
    """
    paired_data = []
    for case in cases:
        image_array = case_arrays[case]
        segmentation_array = segmentation_data[case]
        for i in range(image_array.shape[0]):
            paired_data.append((image_array[i], segmentation_array[i]))
    return paired_data


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
    metrics_filename = f"metrics_{timestamp}.json"
    metrics_path = os.path.join(directory, metrics_filename)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")


def predict_segmentation(model, image_array, device, threshold=0.5):
    model.to(device)
    # Add dimensions to match the model input shape [batch_size, channels, height, width]
    image_tensor = (
        torch.from_numpy(image_array.astype(np.float32))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        output = model(image_tensor)
    predicted = torch.sigmoid(output) > threshold
    return predicted.squeeze(0).squeeze(0).cpu()


def generate_seg_preds(model, case_arrays, case_names, device, threshold=0.5):
    seg_preds = {}
    model.eval()

    print("Generating segmentation predictions...")
    for case in tqdm(case_names):
        images = case_arrays[case]
        predictions = [
            predict_segmentation(model, image, device, threshold) for image in images
        ]
        seg_preds[case] = predictions
    print("Segmentation predictions generated.")

    return seg_preds


def calculate_pred_accuracy(seg_preds, seg_true, case_names):
    seg_acc = {}

    print(
        "Calculating segmentation prediction binary accuracies compared to ground truth..."
    )
    for case in tqdm(case_names):
        pred_masks = seg_preds[case]
        true_masks = seg_true[case]

        case_accuracies = []

        for pred_mask, true_mask in zip(pred_masks, true_masks):
            # Ensure both tensors are on the same device and of the same type
            pred_mask = pred_mask.to(dtype=torch.float32)
            true_mask = torch.Tensor(true_mask).to(dtype=torch.float32)

            # Calculate accuracy
            metric = BinaryAccuracy()
            metric.update(pred_mask, true_mask)
            accuracy = metric.compute().item()
            case_accuracies.append(accuracy)

        seg_acc[case] = case_accuracies
    print("Segmentation prediction accuracies calculated.")

    return seg_acc


# def dice_coeff(seg_pred, seg_true, smooth=1):
#     pred = seg_pred.view(-1).float()
#     true = seg_true.view(-1).float()

#     if pred.sum() == 0 and true.sum() == 0:
#         return torch.tensor(1.0)  # Perfect match for cases with no segmentation

#     intersection = (pred * true).sum()
#     dice = (2.0 * intersection + smooth) / (pred.sum() + true.sum() + smooth)
#     return dice


def dice_coeff(seg_pred, seg_true, smooth=1, threshold=10):
    pred = seg_pred.view(-1).float()
    true = seg_true.view(-1).float()

    # Consider prediction as effectively empty if below threshold
    if pred.sum() < threshold and true.sum() == 0:
        return torch.tensor(1.0)

    intersection = (pred * true).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + true.sum() + smooth)
    return dice


def calculate_dice_similarity(seg_preds, seg_true, case_names, pred_threshold=10):
    seg_dice = {}

    print("Calculating Dice Similarity Coefficients for segmentation predictions...")
    for case in tqdm(case_names):
        pred_masks = seg_preds[case]
        true_masks = seg_true[case]

        case_dice_scores = []

        for pred_mask, true_mask in zip(pred_masks, true_masks):
            pred_mask = pred_mask.to(dtype=torch.float32)
            true_mask = torch.Tensor(true_mask).to(dtype=torch.float32)

            dice_score = dice_coeff(pred_mask, true_mask, threshold=pred_threshold)
            case_dice_scores.append(dice_score.item())

        seg_dice[case] = case_dice_scores
    print("Dice Similarity Coefficients calculated.")
    return seg_dice


class CustomDataset(Dataset):
    """
    A custom Dataset class for loading paired medical imaging data (CT scans and corresponding segmentation masks).

    This class inherits from the PyTorch Dataset class and is used for easier data loading and manipulation in the training pipeline.

    Attributes:
        paired_data (list of tuples): A list where each tuple contains an image (as a NumPy array) and its corresponding segmentation mask.

    Methods:
        __init__(self, paired_data): Initializes the dataset with paired data.
        __len__(self): Returns the length of the dataset.
        __getitem__(self, idx): Retrieves a processed image-mask pair from the dataset by index.
    """

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
