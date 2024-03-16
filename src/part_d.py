# %%
import os
import torch
import json
import matplotlib.pyplot as plt
from models import SimpleUNet

from utils import (
    load_segmentation_data,
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
model_filename = "unet_lr0.1_epochs2_bs3_trainacc1.00_testacc0.99_20240313-160230.pth"
metrics_filename = "metrics_20240313-160230_new.json"

model_path = os.path.join(parent_directory, "saved_models", model_filename)
metrics_path = os.path.join(parent_directory, "saved_models", metrics_filename)

# load the metrics
with open(metrics_path, "r") as f:
    metrics = json.load(f)

# Extracting metrics for plotting
losses = metrics["losses"]
train_accuracies = metrics["train_accuracies"]
test_accuracies = metrics["test_accuracies"]
train_cases = metrics["train_cases"]
test_cases = metrics["test_cases"]


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

seg_preds = generate_seg_preds(model, case_arrays, case_names, device, threshold=0.1)

# %%
seg_acc = calculate_pred_accuracy(seg_preds, seg_arrays, case_names)
# %%
seg_dice = calculate_dice_similarity(
    seg_preds, seg_arrays, case_names, pred_threshold=20
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
slice_num = 100
visualize_segmentation(case, slice_num, case_arrays, seg_arrays, seg_preds)


# %%
def gather_scores(seg_dice, selected_cases):
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

    return all_dice_scores


# %% Histogram of Dice scores
DSC_THRESHOLD_1 = 20
DSC_THRESHOLD_2 = 0.005 * 512 * 512
seg_dice = calculate_dice_similarity(
    seg_preds, seg_arrays, case_names, pred_threshold=DSC_THRESHOLD_1
)
seg_dice_2 = calculate_dice_similarity(
    seg_preds, seg_arrays, case_names, pred_threshold=DSC_THRESHOLD_2
)


# Plotting the histogram
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

axes[0].hist(
    gather_scores(seg_dice, train_cases),
    bins=20,
    color="skyblue",
    edgecolor="black",
    label=f"Train (DS Threshold = {DSC_THRESHOLD_1})",
)


axes[0].hist(
    gather_scores(seg_dice_2, train_cases),
    bins=20,
    color="orange",
    edgecolor="black",
    alpha=0.7,
    label=f"Train (DS Threshold = {DSC_THRESHOLD_2})",
)


axes[0].set_title("Histogram of Dice Scores for Train Cases")
axes[0].set_xlabel("Dice Score")
axes[0].set_ylabel("Frequency")
axes[0].grid(axis="y", alpha=1)
axes[0].legend()

axes[1].hist(
    gather_scores(seg_dice, test_cases),
    bins=20,
    color="firebrick",
    edgecolor="black",
    alpha=0.7,
    label=f"Test (DS Threshold = {DSC_THRESHOLD_1})",
)
axes[1].set_title("Histogram of Dice Scores for Test Cases")
axes[1].set_xlabel("Dice Score")
axes[1].set_ylabel("Frequency")
axes[1].grid(axis="y", alpha=1)

axes[1].hist(
    gather_scores(seg_dice_2, test_cases),
    bins=20,
    color="limegreen",
    edgecolor="black",
    alpha=0.7,
    label=f"Test (DS Threshold = {DSC_THRESHOLD_2})",
)

axes[1].legend()


plt.tight_layout()
plt.savefig("DSC_histogram.png", format="png", dpi=300)
plt.show()


# %% Histogram of accuracy scores
def gather_accuracy_scores(seg_acc, selected_cases):
    """
    Gathers all accuracy scores from the selected cases.

    Args:
        seg_acc (dict): Dictionary containing the accuracy scores, where keys are case names.
        selected_cases (list): List of case names for which to gather the scores.
    """
    all_accuracy_scores = []
    for case in selected_cases:
        if case in seg_acc:
            all_accuracy_scores.extend(seg_acc[case])
        else:
            print(f"Case '{case}' not found in seg_acc")

    return all_accuracy_scores


# Plotting the histogram
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Histogram for training accuracy scores
axes[0].hist(
    gather_accuracy_scores(seg_acc, train_cases),
    bins=20,
    color="blue",
    edgecolor="black",
)
axes[0].set_title("Histogram of Accuracy Scores for Train Cases")
axes[0].set_xlabel("Accuracy Score")
axes[0].set_ylabel("Frequency")
axes[0].grid(axis="y", alpha=0.75)

# %%
# Histogram for testing accuracy scores
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the accuracy for case_000
case = "Case_001"
dice_scores = seg_dice[case]

ax.plot(dice_scores, label="DSC")


ax.set_title(f"Dice Similarity Coefficients for {case}")
ax.set_xlabel("Slice")
ax.set_ylabel("Dice Similarity")
fig.legend()

plt.show()

# %%


def process_case_slices(seg_dice, num_examples=3):
    best_per_case = {}
    worst_per_case = {}
    middle_per_case = {}

    for case in seg_dice.keys():
        # Filter out slices with DSC = 1 and sort the rest
        case_scores = [
            (idx, score)
            for idx, score in enumerate(seg_dice[case])
            if score < 1 and score > 0.3
        ]
        sorted_scores = sorted(case_scores, key=lambda x: x[1])

        if sorted_scores:
            worst_per_case[case] = sorted_scores[0:num_examples]  # Worst slices
            best_per_case[case] = sorted_scores[-num_examples:]  # Best slices
            middle_index = len(sorted_scores) // 2
            middle_slices = sorted_scores[
                max(0, middle_index - num_examples // 2) : min(
                    len(sorted_scores), middle_index + num_examples // 2 + 1
                )
            ]
            middle_per_case[case] = middle_slices  # Median slices
    return best_per_case, worst_per_case, middle_per_case


num_examples = 3
best_slices, worst_slices, middle_slices = process_case_slices(seg_dice, num_examples)


def visualize_slice_with_score(case_name, slice_info, case_arrays, seg_true, seg_preds):
    slice_idx, dice_score = slice_info
    image = case_arrays[case_name][slice_idx]
    true_mask = seg_true[case_name][slice_idx]
    pred_mask = seg_preds[case_name][slice_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original Image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"Original Image\nCase: {case_name}, Slice: {slice_idx}")
    axes[0].axis("off")

    # Ground Truth Segmentation
    axes[1].imshow(true_mask, cmap="jet")  # Overlay with alpha
    axes[1].set_title("Ground Truth\nDice Score: {:.4f}".format(dice_score))
    axes[1].axis("off")

    # Predicted Segmentation
    axes[2].imshow(pred_mask, cmap="jet")  # Overlay with alpha
    axes[2].set_title("Predicted Segmentation\nDice Score: {:.4f}".format(dice_score))
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


## List of cases to visualize
cases_to_visualize = ["Case_000", "Case_001", "Case_002"]

# Visualize the best slices for selected cases
for case in cases_to_visualize:
    if case in best_slices:
        for slice_info in best_slices[case]:
            visualize_slice_with_score(
                case, slice_info, case_arrays, seg_arrays, seg_preds
            )
    else:
        print(f"Best slices for {case} are not available.")

# %%
# Visualize the worst slices for selected cases
for case in cases_to_visualize:
    if case in worst_slices:
        for slice_info in worst_slices[case]:
            visualize_slice_with_score(
                case, slice_info, case_arrays, seg_arrays, seg_preds
            )
    else:
        print(f"Worst slices for {case} are not available.")

# %% Visualize the middle slices for selected cases
for case in cases_to_visualize:
    if case in middle_slices:
        for slice_info in middle_slices[case]:
            visualize_slice_with_score(
                case, slice_info, case_arrays, seg_arrays, seg_preds
            )
    else:
        print(f"Middle slices for {case} are not available.")
