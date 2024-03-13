# %%
import os
import torch
import json
import matplotlib.pyplot as plt
from models import SimpleUNet
import matplotlib

# Set up matplotlib to use LaTeX for formatting
matplotlib.rcParams.update(
    {
        "font.size": 14,
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 14,
        "figure.autolayout": True,
        "savefig.dpi": 300,
        "figure.dpi": 300,
    }
)

# Define the current directory and parent directory
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)

# Construct paths to the saved model and metrics
model_filename = "unet_lr0.1_epochs2_bs3_trainacc0.99_testacc0.99_20240313-153629.pth"
metrics_filename = "metrics_20240313-153629_new.json"

model_path = os.path.join(parent_directory, "saved_models", model_filename)
metrics_path = os.path.join(parent_directory, "saved_models", metrics_filename)

# Load the model
model = SimpleUNet(in_channels=1, out_channels=1)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Load the metrics
with open(metrics_path, "r") as f:
    metrics = json.load(f)

# Extracting metrics for plotting
losses = metrics["losses"]
train_accuracies = metrics["train_accuracies"]
test_accuracies = metrics["test_accuracies"]
# %%
# Plotting the metrics
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses, label="Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy", color="orange")
plt.plot(test_accuracies, label="Test Accuracy", color="green")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()

fig_directory = os.path.join(parent_directory, "figures")
filename = "metrics_plot_new.png"
filepath = os.path.join(fig_directory, filename)

# Save the figure in a high-quality format
plt.savefig(filepath, format="png", dpi=300)

plt.show()
