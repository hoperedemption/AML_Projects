import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from torch.utils.data import Dataset

target_height, target_width = 224, 244

class Sharpen(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

    def apply(self, image, **params):
        # Define sharpening kernel
        sharpening_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        # Apply the kernel to the image
        sharpened_image = cv2.filter2D(image, -1, sharpening_kernel).clip(min=0, max=255.0)
        return sharpened_image
    
transform_img_ = A.Compose([
    Sharpen(p=1.0), 
    ToTensorV2()
])

def transform_img(image):
    return transform_img_(image=image.numpy())['image'].squeeze(0)

transform_ = A.Compose([
    A.VerticalFlip(p=0.3), 
    A.OneOf(
        [
            A.ElasticTransform(alpha=5, sigma=50, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0)
        ], p=0.5
    ), 
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.Rotate(limit=(-45, 45), p=0.5), 
    A.Normalize(
        mean=[0.0], 
        std=[1.0], 
        max_pixel_value=1.0
        ), 
    ToTensorV2()
], additional_targets={'mask': 'mask'})

def transform(image, mask):
    dict = transform_(image=image.numpy(), mask=mask.numpy())
    return dict['image'].squeeze(0), dict['mask']

class CustomDataset(Dataset):
    def __init__(self, image_list, mask_list, transform=False):
        assert image_list.shape[0]  == mask_list.shape[0]

        super().__init__()
        self.image_list = image_list 
        self.mask_list = mask_list
        self.transform = transform
        
    def __len__(self):
        return self.image_list.shape[0]
    
    def __getitem__(self, idx):
        img, mask = self.image_list[idx], self.mask_list[idx]
        if self.transform:
            img = transform_img(img)
            img, mask = transform(img, mask)
            img, mask = img.unsqueeze(0), mask.unsqueeze(0) # get back the channel dim
        return img, mask
