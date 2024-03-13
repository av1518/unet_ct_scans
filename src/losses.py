import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, y_pred, y_true):
        bce = self.bce_loss(y_pred, y_true)
        dice_loss = soft_dice_loss(
            y_true, torch.sigmoid(y_pred)
        )  # Apply sigmoid before calculating Dice loss
        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss


def soft_dice_loss(y_true, y_pred, smooth=1):
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    intersection = (y_true_flat * y_pred_flat).sum()
    dice_score = (2.0 * intersection + smooth) / (
        y_true_flat.sum() + y_pred_flat.sum() + smooth
    )
    return 1 - dice_score
