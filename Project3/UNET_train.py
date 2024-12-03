import torch
import torchvision 
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
from UNET import UNET 
from utils import (load_checkpoints, save_checkpoints, get_loaders, check_accuracy, save_predictions_as_imgs, print_evaluation_metrics)

LR = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True 
LOAD_MODEL = True 
IMAGE_HEIGHT = 280
IMAGE_WIDTH = 280

class VOCSegmentationTransform:
    def __init__(self, image_transform=None, mask_transform=None):
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __call__(self, image, mask):
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask
    
class CustomTransform(object):
    def __init__(self, transform, p=0.5):
        self.transform = transform()
        self.p = p
    def __call__(self, image):
        if np.random.uniform(low=0.0, high=1.0) < self.p:
            return self.transform(image)
        else:
            return image

transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH), 
        A.Rotate(limit=35, p=0.5),
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.1), 
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0), 
        ToTensorV2()
    ]
)

custom_transform = CustomTransform(transform)

voc_transform = VOCSegmentationTransform(image_transform=custom_transform, mask_transform=custom_transform)

model = UNET(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), in_channels=3, out_channels=1).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optmizer = optim.Adam(model.parameters(), lr=LR)

# get the segmentation datasets
voc_segmentation_train = torchvision.datasets.VOCSegmentation(root='data', image_set='train', download=True)
voc_segmentation_val = torchvision.datasets.VOCSegmentation(root='data', image_set='trainval', download=True)
voc_segmentation_test = torchvision.datasets.VOCSegmentation(root='data', image_set='val', download=True) 

# get the dataloaders
voc_segmentation_train_loader = torch.utils.data.DataLoader(voc_segmentation_train)
voc_segmentation_val_loader = torch.utils.data.DataLoader(voc_segmentation_val)
voc_segmentation_test_loader = torch.utils.data.DataLoader(voc_segmentation_test)

# create a scaler
scaler = torch.amp.GradScaler()

def train_fn(train_loader, model, optmizer, loss_fn, scaler, epoch, num_epochs):
    model.train()
    with tqdm(train_loader, total=len(train_loader), leave=False) as train_loop:
        for data, target in train_loop:
            # apply the transformation
            data, target = voc_transform(data, target)
            
            # get data and target
            data, target = data.to(torch.float32).to(DEVICE), target.to(torch.float32).to(DEVICE)
            
            # zero out all previous gradients
            optmizer.zero_grad()
            
            # run the forward pass with autocasting, float16 precision
            with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                output = model(data)
                loss = loss_fn(output, target)
                
            # scales loss, calls backward() on scaled loss 
            scaler.scale(loss).backward()
            
            # scaler step, optimizer step
            scaler.step(optmizer)
            
            # update the scale for next iteration
            scaler.update()
            
            # compute the accuracy
            acc = (output == target).mean()
            
            # update the tqdmp loop
            train_loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            train_loop.set_postfix(loss = loss.item(), acc = acc.item())

def val_fn(val_loader, model, loss_fn):
    model.eval()
    
    with torch.no_grad():
        for data, target in val_loader:
            # apply the transformation
            data, target = voc_transform(data, target)
            
            # get data and target 
            data, target = data.to(torch.float32).to(DEVICE), target.to(torch.float32).to(DEVICE)
            
            # get the output
            output = model(data)
            
            # evaluate the loss 
            loss = loss_fn(output, target)
            
            # compute the accuracy
            acc = (output == target).mean()
            
            # get metrics dict
            metrics = {'accuracy' : acc, 'loss': loss}
            
            # print the mtrics 
            print_evaluation_metrics(metrics)
               
        
for epoch in range(NUM_EPOCHS):
    train_fn(voc_segmentation_train_loader, model, optmizer, loss_fn, scaler)                

def main():
    pass
