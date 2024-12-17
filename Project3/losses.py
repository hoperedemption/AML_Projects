import torch 
import torch.nn as nn
import torch.nn.functional as F

class Dice(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, gt, eps=1e-6):
        # gt, pred: shape (B, C, H, W)
        intersection = (gt * pred).sum((-2, -1))
        return (1 - (2 * intersection + eps) / (gt.sum((-2, -1)) + pred.sum((-2, -1)) + eps)).mean()
    
class IoU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, gt, eps=1):
        intersection = (gt * pred).sum((-2, -1))
        union = gt.sum((-2, -1)) + pred.sum((-2, -1)) - intersection
        return (1 - (intersection + eps) / (union + eps)).mean()

class BinaryFocalIoU(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = eps
    def forward(self, pred, gt):
        # 1. Binary Focal Loss
        BCE_loss = F.binary_cross_entropy(pred, gt, reduction='none')
        pt = torch.where(gt == 1, pred, 1 - pred)  # p_t for focal loss
        Focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
         # 2. Jaccard Loss
        intersection = (pred * gt).sum((1, 2, 3))  
        union = (pred + gt - pred * gt).sum((1, 2, 3))
        jaccard_loss = 1 - (intersection + self.smooth) / (union + self.smooth)
        
        return Focal_loss.mean() + jaccard_loss.mean()
              