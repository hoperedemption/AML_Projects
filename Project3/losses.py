import torch 
import torch.nn as nn

class Dice(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, gt, pred, eps=1e-6):
        intersection = torch.sum(gt * pred)
        return 1 - (2 * intersection + eps) / (torch.sum(gt) + torch.sum(pred) + eps)