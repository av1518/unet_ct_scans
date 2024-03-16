import torch
import torch.nn as nn
from utils import dice_coeff


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_threshold=0.005 * 512 * 512):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_threshold = dice_threshold

    def forward(self, y_pred, y_true):
        bce = self.bce_loss(y_pred, y_true)
        dice_loss = 1 - dice_coeff(y_true, torch.sigmoid(y_pred))
        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss
