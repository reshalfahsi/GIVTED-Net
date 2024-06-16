import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFunction(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, mask):
        bce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        pred = torch.sigmoid(pred)
        inter = ((pred * mask)).sum(dim=(2, 3))
        union = ((pred + mask)).sum(dim=(2, 3))
        dice = 1.0 - (2 * inter + self.eps)/(union + self.eps)
        return dice.mean() + bce
