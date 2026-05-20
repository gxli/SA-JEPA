import torch.nn as nn


class LegacyDualJEPALoss(nn.Module):
    """Legacy combined JEPA + pixel-mask loss (not used by current training loop)."""

    def __init__(self, weight_jepa: float = 1.0, weight_pixel: float = 1.0):
        super().__init__()
        self.w_jepa = weight_jepa
        self.w_pixel = weight_pixel
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_latent, gt_latent, pred_mask, true_mask):
        loss_jepa = self.mse(pred_latent, gt_latent.detach())
        loss_pixel = self.bce(pred_mask, true_mask)
        total_loss = (self.w_jepa * loss_jepa) + (self.w_pixel * loss_pixel)
        return total_loss, loss_jepa, loss_pixel
