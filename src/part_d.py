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

seg_preds = generate_seg_preds(model, case_arrays, case_names, device, threshold=0.3)

# %%
seg_acc = calculate_pred_accuracy(seg_preds, seg_arrays, case_names)
# %%
seg_dice = calculate_dice_similarity(
    seg_preds, seg_arrays, case_names, pred_threshold=0.005 * 512 * 512
)

# %% plot the accuracy and dice similarity

ax, fig = plt.subplots(1, 2, figsize=(20, 10))

# Plot the accuracy for case_000
case = "Case_001"
accuracies = seg_acc[case]
dice_scores = seg_dice[case]

fig[0].plot(accuracies, label="Accuracy")
fig[1].plot(dice_scores, label="Dice Similarity")

fig[0].set_title(f"Segmentation Prediction Accuracies for {case}")
fig[0].set_xlabel("Slice")
fig[0].set_ylabel("Accuracy")
fig[0].legend()

fig[1].set_title(f"Dice Similarity Coefficients for {case}")
fig[1].set_xlabel("Slice")
fig[1].set_ylabel("Dice Similarity")
fig[1].legend()

plt.show()


# %%
def visualize_segmentation(case, slice_num, case_arrays, seg_true, seg_preds):
    """
    Visualizes the CT scan slice along with ground truth and predicted masks.

    Args:
        case (str): The case identifier.
        slice_num (int): The slice number to visualize.
        case_arrays (dict): Dictionary containing original image data.
        seg_true (dict): Dictionary containing ground truth segmentation masks.
        seg_preds (dict): Dictionary containing predicted segmentation masks.
    """
    image = case_arrays[case][slice_num]
    true_mask = seg_true[case][slice_num]
    pred_mask = seg_preds[case][slice_num]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Display original image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Display ground truth mask overlay
    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(true_mask, cmap="jet", alpha=0.5)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    # Display predicted mask overlay
    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(pred_mask, cmap="jet", alpha=0.5)
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    plt.show()


# Visualize the segmentation for case_000
case = "Case_000"
slice_num = 150
visualize_segmentation(case, slice_num, case_arrays, seg_arrays, seg_preds)


# %%
def plot_dice_scores_histogram(seg_dice, selected_cases):
    """
    Plots a histogram of Dice scores for the selected cases.

    Args:
        seg_dice (dict): Dictionary containing the Dice scores, where keys are case names.
        selected_cases (list): List of case names for which to plot the histogram.
    """
    # Gather all Dice scores from the selected cases
    all_dice_scores = []
    for case in selected_cases:
        if case in seg_dice:
            all_dice_scores.extend(seg_dice[case])
        else:
            print(f"Case '{case}' not found in seg_dice")

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_dice_scores, bins=20, color="skyblue", edgecolor="black")
    plt.title("Histogram of Dice Scores")
    plt.xlabel("Dice Score")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.show()


plot_dice_scores_histogram(seg_dice, case_names)
