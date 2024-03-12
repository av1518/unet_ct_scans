# %%
import os
import torch
import json
import matplotlib.pyplot as plt
from models import SimpleUNet
import matplotlib

from utils import (
    load_segmentation_data,
    CustomDataset,
    create_paired_data,
    save_metrics,
    load_image_data,
    generate_seg_preds,
    calculate_pred_accuracy,
    calculate_dice_similarity,
)

# %%

# Define the current directory and parent directory
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)

# Construct paths to the saved model and metrics
model_filename = "unet_lr0.1_epochs2_bs3_trainacc1.00_testacc0.99_20240311-213847.pth"
metrics_filename = "metrics_20240311-213847.json"

model_path = os.path.join(parent_directory, "saved_models", model_filename)
metrics_path = os.path.join(parent_directory, "saved_models", metrics_filename)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleUNet(in_channels=1, out_channels=1)
model.load_state_dict(torch.load(model_path))
model.to(device)

# load the cases
image_path = os.path.join(parent_directory, "Dataset\Images")
seg_path = os.path.join(parent_directory, "Dataset\Segmentations")

# Load the data
case_arrays = load_image_data(image_path)
seg_arrays = load_segmentation_data(seg_path)

# %%
case_names = [
    d for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))
]

seg_preds = generate_seg_preds(model, case_arrays, case_names, device, threshold=0.5)

# %%
seg_acc = calculate_pred_accuracy(seg_preds, seg_arrays, case_names)
# %%
from tqdm import tqdm


def dice_coeff(seg_pred, seg_true, smooth=1):
    pred = seg_pred.view(-1).float()
    true = seg_true.view(-1).float()

    if pred.sum() == 0 and true.sum() == 0:
        return torch.tensor(1.0)  # Perfect match for cases with no segmentation

    intersection = (pred * true).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + true.sum() + smooth)
    return dice


def calculate_dice_similarity(seg_preds, seg_true, case_names):
    seg_dice = {}

    print("Calculating Dice Similarity Coefficients for segmentation predictions...")
    for case in tqdm(case_names):
        pred_masks = seg_preds[case]
        true_masks = seg_true[case]

        case_dice_scores = []

        for pred_mask, true_mask in zip(pred_masks, true_masks):
            pred_mask = pred_mask.to(dtype=torch.float32)
            true_mask = torch.Tensor(true_mask).to(dtype=torch.float32)

            dice_score = dice_coeff(pred_mask, true_mask)
            case_dice_scores.append(dice_score.item())

        seg_dice[case] = case_dice_scores
    print("Dice Similarity Coefficients calculated.")
    return seg_dice


seg_dice = calculate_dice_similarity(seg_preds, seg_arrays, case_names)

# %%
