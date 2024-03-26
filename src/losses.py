import torch
import torch.nn as nn
from utils import dice_coeff


class CombinedLoss(nn.Module):
    """
    @brief A custom loss function that combines Binary Cross-Entropy (BCE) Loss and Dice Loss for image segmentation tasks.

    This class defines a custom, loss function that balances the Binary Cross-Entropy (BCE) Loss and Dice Loss.
    It is used to balance pixel-wise accuracy and the overall shape of the segmented region.
    The loss is a weighted sum of Binary Cross-Entropy and Dice Loss.

    @param bce_weight (float, optional): The weight given to BCE Loss in
    the combined loss calculation. Default is 0.5.
    @param dice_threshold (float, optional): A threshold value used in
    Dice Loss calculation to consider predictions effectively empty. Default is 0.005 * 512 * 512,.
    This is 0.5% of the total number of pixels in a 512x512 image.

    @note The Dice Loss component is modified with a threshold to handle cases where the predicted segmentation mask is almost empty.
    In that case, the Dice coefficient is set to 1, and the Dice Loss is set to 0.
    """

    def __init__(self, bce_weight=0.5, dice_threshold=0.005 * 512 * 512):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_threshold = dice_threshold

    def forward(self, y_pred, y_true):
        """
        @brief Compute the combined loss given predictions and true labels.

        The function calculates the combined loss using Binary Cross-Entropy (BCE) Loss and Dice Loss.
        The BCE Loss is computed directly using `torch.nn.BCEWithLogitsLoss`,
        and the Dice Loss is computed using the custom `dice_coeff` function.

        @param y_pred: Predicted labels (logits) from the model.
        @param y_true: True binary labels for segmentation.

        @return: The combined loss value as a scalar.
        """
        bce = self.bce_loss(y_pred, y_true)
        dice_loss = 1 - dice_coeff(y_true, torch.sigmoid(y_pred))
        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss
