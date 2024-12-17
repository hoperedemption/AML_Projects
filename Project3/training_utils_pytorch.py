import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
    
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112

class CustomTransform(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p 
        self.transform = transforms.Compose([
            transforms.ElasticTransform(alpha=5, sigma=10), 
            transforms.RandomAffine(degrees=45, translate=(0.15, 0.15)), 
            # transforms.RandomResizedCrop(size=(IMAGE_HEIGHT, IMAGE_WIDTH), scale=(0.8, 1.0))
        ])
        
    def forward(self, img, mask):
        cat = torch.cat([img, mask], dim=0)
        if torch.rand(1).item() < self.p:
            cat = self.transform(torch.cat([img, mask], dim=0))
        return cat[:-1, ...], cat[-1, ...]

class CustomDataset(Dataset):
    def __init__(self, image_list, mask_list, device, channels, apply_transform=False):
        assert image_list.shape[0]  == mask_list.shape[0]

        super().__init__()
        self.image_list = image_list 
        self.mask_list = mask_list
        self.apply_transform = apply_transform
        self.channels = channels
        self.device = device
        self.transform = CustomTransform()
        
    def __len__(self):
        return self.image_list.shape[0]
    
    def __getitem__(self, idx):
        img, mask = self.image_list[idx], self.mask_list[idx]
        view = lambda x, c: x.view(c, x.shape[-2], x.shape[-1])
        
        if self.apply_transform:
            img, mask = self.transform(view(img, self.channels), view(mask, 1))
        if img.device != self.device:
            img.to(self.device)
        if mask.device != self.device:
            mask.to(self.device)
        
        return view(img, self.channels), view(mask, 1)