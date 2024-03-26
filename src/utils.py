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


def convert_dicom_to_numpy_slice_location(case_path):
    """
    @brief Convert a series of DICOM files in a directory into a 3D NumPy array based
    on the "SliceLocation" metadata.

    This function processes all DICOM (.dcm) files located in the specified directory,
    extracting the slice location information and using it to sort the files in the
    correct sequence. It then reads the pixel data from each file and stacks them into
    a 3D NumPy array. The function used for ensuring that the slices are in the
    right order. This is important for the UNET training.

    @param case_path The path to the directory containing the DICOM files.
                     This directory should contain all the .dcm files for a single case.

    @return A 3D NumPy array containing the stacked pixel data from the DICOM files.
            The shape of the array is [number_of_slices, rows, columns], where
            number_of_slices is the total number of DICOM files in the directory, and
            rows and columns correspond to the dimensions of the images.

    @exception ValueError Thrown when no DICOM files with 'SliceLocation' metadata are
                          found or if the directory does not contain any .dcm files.

    Example usage:
    >>> numpy_array = convert_dicom_to_numpy_slice_location('/path/to/dicom/files')
    """
    # List all DICOM files in the directory
    dicom_files = [f for f in listdir(case_path) if f.endswith(".dcm")]

    # Read SliceLocation tag for each file and store in a list
    dicom_metadata = []
    for file in dicom_files:
        file_path = join(case_path, file)
        metadata = dcmread(file_path)
        slice_location = getattr(metadata, "SliceLocation", None)
        if slice_location is not None:
            dicom_metadata.append((file, slice_location))
        else:
            print(f"SliceLocation not found in {file}")

    # Sort the list based on the slice location
    # (important because masks are sorted by slice location as well)
    dicom_metadata.sort(key=lambda x: x[1])

    # Create an empty 3D numpy array in correct shape to store the images
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


def load_image_data(image_path):
    """
    @Brief Loads image data from all DICOM files located in a specified directory. Returns
    dictionary of 3D NumPy arrays representing the stacked images of each case.

    This function processes directories within the given image path, each representing
    a distinct case. It reads DICOM (.dcm) files from these directories, ensuring that
    the slices are sorted based on their 'SliceLocation' metadata for accurate 3D
    representation. The function uses 'convert_dicom_to_numpy_slice_location' for
    converting the DICOM files of each case into a 3D NumPy array.

    @param image_path: The path to the directory containing subdirectories for each case.
                       Each subdirectory should contain DICOM files for that particular case.

    @return: A dictionary where keys are case names derived from the subdirectory names,
             and values are 3D NumPy arrays representing the stacked images of each case.

    Example usage:
    >>> image_data = load_image_data('/path/to/image/data')
    """
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
    print("Images loaded.")

    return case_arrays


def load_segmentation_data(segmentation_path):
    """
    @brief Loads segmentation data from .npz files located in a specified directory.

    Each .npz file in the directory should contain segmentation data related to a specific case.
    The function expects the file naming convention to follow "Case_XXX_seg.npz", where "XXX"
    represents the case identifier. The segmentation data for each case is extracted and
    stored in a dictionary keyed by the case name.

    The function specifically looks for a key named 'masks' in each .npz file to retrieve
    the segmentation masks.

    @param segmentation_path: The path to the directory containing the .npz files with
                              segmentation data.

    @return: A dictionary where the keys are case identifiers, and the values are the
             corresponding segmentation mask arrays.

    @exception: If a .npz file does not contain the 'masks' key, a message is printed
                indicating that the key was not found in that file.

    Example usage:
    >>> segmentation_data = load_segmentation_data('/path/to/segmentation/data')
    """
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


def create_paired_data(cases, case_arrays, segmentation_data):
    """
    @brief Creates a list of paired data from images and segmentation masks.

    This function iterates through specified cases and pairs each image slice
    with its corresponding segmentation mask slice. The pairs are formed based
    on the index of the slices within their respective arrays, ensuring
    accurate alignment between each image and its segmentation mask.

    @param cases: A list of case identifiers. Each identifier corresponds to a
                  key in both case_arrays and segmentation_data dictionaries.
                  They are of the form ["Case_001", "Case_002", ...]
    @param case_arrays: A dictionary where keys are case identifiers and values
                        are image arrays represented as NumPy arrays.
    @param segmentation_data: A dictionary where keys are case identifiers and
                              values are segmentation mask arrays as NumPy arrays.

    @return: A list of tuples, where each tuple consists of an image slice and
             its corresponding segmentation mask slice, both as NumPy arrays.

    Example usage:
    >>> paired_data = create_paired_data(["Case_001, Case_011"], image_arrays, segmentation_arrays)
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
    @brief Saves the provided metrics, including train and test cases, as a JSON file in the specified directory.

    @param metrics (dict): A dictionary containing the metrics to save.
                           Expected to have keys like 'losses', 'train_accuracies', 'test_accuracies',
                           'train_cases', and 'test_cases'.
    @param directory (str): Path to the directory where the JSON file will be saved.
    @param timestamp (str): Timestamp to append to the filename to ensure unique filenames.
    """
    metrics_filename = f"metrics_{timestamp}_new.json"
    metrics_path = os.path.join(directory, metrics_filename)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")


def predict_segmentation(model, image_array, device, threshold=0.5):
    """
    @brief Predicts the segmentation mask for a given image using a trained model.

    This function processes a single image array through the provided model to generate
    a segmentation mask. It adapts the image array to match the input requirements of the
    model, specifically adding necessary dimensions to mimic batch size and channel depth.
    The model's prediction is then passed through a sigmoid activation function to convert
    logits into probabilities. A threshold is applied to these probabilities to generate
    a binary mask.

    @param model: The trained segmentation model used for prediction.
    @param image_array: A 2D numpy array representing the input image.
    @param device: The device (CPU or CUDA) on which the model and data are located.
    @param threshold: The threshold for converting sigmoid outputs to binary values.
                      Defaults to 0.5.

    @return A 2D tensor representing the predicted binary segmentation mask,
            moved back to the CPU.
    """
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
    """
    @brief Generates segmentation predictions for a set of images using a trained model.

    This function iterates over a specified list of cases, applying the trained model to
    generate segmentation predictions for each image within these cases. The function
    uses the `predict_segmentation` method for each image and collects the predictions
    in a dictionary, keyed by case names.

    @param model: The trained model used for generating segmentation predictions.
    @param case_arrays: A dictionary where keys are case identifiers and values are lists
                        of image arrays (NumPy arrays) for each case.
    @param case_names: A list of case identifiers for which predictions are to be generated.
    @param device: The device (CPU or CUDA) on which the model and data are located.
    @param threshold: The threshold used in `predict_segmentation` for binarising the
                      model's output. Defaults to 0.5.

    @return A dictionary where keys are case identifiers and values are lists of
            predicted segmentation masks (as tensors) for each image in the case.
    """

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
    """
    @brief Calculates binary accuracies of predicted segmentations compared to ground truth.

    This function iterates through each case specified in `case_names`, comparing the
    predicted segmentation masks (`seg_preds`) against the true masks (`seg_true`). It
    calculates the binary accuracy for each slice within a case using the BinaryAccuracy
    metric from torchmetrics. The accuracies for each slice in a case are compiled into a
    list and stored in a dictionary keyed by the case names.

    @param seg_preds: A dictionary containing the predicted segmentation masks. Keys are
                      case identifiers, and values are lists of predicted masks for each
                      image slice.
    @param seg_true: A dictionary containing the true segmentation masks. Keys are case
                     identifiers, and values are lists of true masks for each image slice.
    @param case_names: A list of case identifiers for which the accuracy is to be calculated.

    @return A dictionary where keys are case identifiers and values are lists of binary
            accuracies for each image slice in the respective case.
    """
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


def dice_coeff(seg_pred, seg_true, smooth=1, threshold=10):
    """
    @brief Calculates the Dice Coefficient between two segmentation masks.

    The Dice Coefficient is a measure of the area of overlap between two samples.

    @param seg_pred: The predicted segmentation mask. Should be a binary (or softmax/sigmoid probabilities converted to binary) tensor.
    @param seg_true: The ground truth segmentation mask. Should be a binary tensor.
    @param smooth (float, optional): A smoothing constant added to the numerator and denominator to avoid division by zero errors. Default is 1.
    @param threshold (float, optional): A threshold value below which the predicted mask is considered effectively empty.

    @return: The Dice Coefficient as a floating-point scalar. Higher values indicate greater similarity between the prediction and the ground truth.

    @note This implementation includes a threshold to handle the special case where both the prediction
    and the ground truth are effectively empty (e.g., no positive pixels in the mask).
    This scenario is common in medical image segmentation where some slices may not contain the region of interest.
    """
    pred = seg_pred.view(-1).float()
    true = seg_true.view(-1).float()

    # Consider prediction as effectively empty if below threshold
    if pred.sum() < threshold and true.sum() == 0:
        return torch.tensor(1.0)

    intersection = (pred * true).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + true.sum() + smooth)
    return dice


def calculate_dice_similarity(seg_preds, seg_true, case_names, pred_threshold=10):
    """
    @brief Calculates the Dice Similarity Coefficients between dictionaries of predicted and true segmentation masks.

    This function iterates through each case specified in `case_names`, comparing the predicted segmentation masks
    (`seg_preds`) against the true masks (`seg_true`). It calculates the Dice Similarity Coefficient for each slice
    within a case using the `dice_coeff` function. The coefficients for each slice in a case are compiled into a list
    and stored in a dictionary keyed by the case names.

    @param seg_preds: A dictionary containing the predicted segmentation masks. Keys are case identifiers, and values
                        are lists of predicted masks for each image slice.
    @param seg_true: A dictionary containing the true segmentation masks. Keys are case identifiers, and values are lists
                        of true masks for each image slice.
    @param case_names: A list of case identifiers for which the Dice Similarity Coefficients are to be calculated.
    @param pred_threshold: A threshold pixel value below which the predicted mask is considered effectively empty. Default is 10.

    @return: A dictionary where keys are case identifiers and values are lists of Dice Similarity Coefficients for each image slice in the respective case.

    """
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
