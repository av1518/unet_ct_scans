import torch
from losses import CombinedLoss
from torchmetrics.classification import BinaryAccuracy
import wandb


def train_model(
    model, train_loader, test_loader, epochs, learning_rate, dice_threshold
):
    print("Training the model...")
    wandb.init(project="lung_segmentation", entity="av662")
    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": len(train_loader.batch_sampler),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CombinedLoss(
        dice_threshold=dice_threshold
    )  # Custom data loss that combines BCE and Dice loss
    accuracy_metric = BinaryAccuracy(threshold=0.5).to(device)

    losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        accuracy_metric.reset()

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = torch.sigmoid(outputs)
            accuracy_metric.update(predictions, masks)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = accuracy_metric.compute()

        losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy.item())

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}"
        )

        # Evaluation phase on test set
        model.eval()
        accuracy_metric.reset()
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                predictions = torch.sigmoid(outputs)
                accuracy_metric.update(predictions, masks)

        epoch_test_accuracy = accuracy_metric.compute()
        test_accuracies.append(epoch_test_accuracy.item())
        print(f", Test Accuracy: {epoch_test_accuracy:.4f}")
        wandb.log({"epoch": epoch, "loss": epoch_loss, "accuracy": epoch_accuracy})

    return losses, train_accuracies, test_accuracies
