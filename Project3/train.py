import torch
import torchvision 
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
import cv2
from PIL import Image
from SwinUNet import SwinUNET 
from utils import (load_zipped_pickle, overlay_segmentation_grayscale,
                   overlay_segmentation_countor_grayscale, print_evaluation_metrics, 
                   plot_training_history)
from training_utils import CustomDataset
from torch.utils.data import Dataset, DataLoader
from losses import Dice
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from collections import defaultdict

LR = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.device(DEVICE)
BATCH_SIZE = 5
NUM_EPOCHS = 100
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

def train_fn(train_loader, model, optimizer, scheduler, loss_fn, scaler, accumulation_steps, epoch, num_epochs):
    model.train()
    
    metrics = defaultdict(list)
    
    with tqdm(train_loader, total=len(train_loader), leave=False) as train_loop:
        for data, target in train_loop:
            # get data and target
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # zero out all previous gradients
            optimizer.zero_grad()
            
            # run the forward pass with autocasting, float16 precision
            with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                output = model(data)
                loss = loss_fn(target, output)
                
            # scales loss, calls backward() on scaled loss 
            scaler.scale(loss).backward()
            
            # Gradient accumulation step
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
            
            # update the metrics list
            metrics['loss'] += [loss]
            
            # update the tqdmp loop
            train_loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            train_loop.set_postfix(loss = loss.item())
        scheduler.step()
        
    metrics['loss'] = np.mean(metrics['loss'])
    return metrics

def iou(gt: torch.Tensor, mask: torch.Tensor, eps: float=1e-6):
    # gt, mask -> (B, 1, H, W)
    gt, mask = gt.to(torch.int32), mask.to(torch.int32)
    intersection = (gt & mask).to(torch.float32).sum((-2, -1))
    union = (gt | mask).to(torch.float32).sum((-2, -1))
    iou = (intersection + eps) / (union + eps)
    return iou.mean() # average across batches

def val_fn(val_loader, model, loss_fn):
    model.eval()
    
    metrics = defaultdict(list)
    
    with torch.no_grad():
        for data, target in val_loader:            
            # get data and target 
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # get the output
            output = model(data)
            
            # compute the segmentations mask
            output = torch.where(output >= 0.5, 1, 0)
            
            # evaluate the loss 
            loss = loss_fn(target, output)
            # evaluate the accuracy
            inter_over_union = iou(target, output)
            
            # get metrics dict
            metrics['iou'] += [inter_over_union]
            metrics['loss'] += [loss]
            
    
    metrics['iou'] = np.mean(metrics['iou'])
    metrics['loss'] = np.mean(metrics['loss'])
    print_evaluation_metrics(metrics)
    
    return metrics 

train_sparse = load_zipped_pickle('train_non_sparse.pkl')

imgs, boxes = train_sparse['imgs'], train_sparse['boxes']

target_height, target_width = IMAGE_HEIGHT, IMAGE_WIDTH
train_imgs, train_segs = np.zeros((len(imgs), target_height, target_width)), np.zeros((len(boxes), target_height, target_width))
for i, (img, seg) in enumerate(zip(imgs, boxes)):
    img = cv2.resize(img, (target_height, target_width), interpolation=cv2.INTER_LINEAR_EXACT)
    seg = Image.fromarray(seg).resize((target_height, target_width), resample=Image.BILINEAR)
    train_imgs[i], train_segs[i] = img, np.asarray(seg) 
    
train_imgs, val_imgs, train_segs, val_segs = train_test_split(train_imgs, train_segs, test_size=0.2, random_state=42)

convert_to_tensor = lambda x: torch.from_numpy(x).to(torch.float32)
train_imgs, val_imgs, train_segs, val_segs = convert_to_tensor(train_imgs), convert_to_tensor(val_imgs), convert_to_tensor(train_segs), convert_to_tensor(val_segs)

train_dataset = CustomDataset(train_imgs, train_segs, transform=True)
val_dataset = CustomDataset(val_imgs, val_segs, transform=True)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)

model = SwinUNET(in_dim=1, out_dim=1, img_size=224, patch_size=4, wdw_size=7, embed_dim=96, depth_list_enc=[2, 2, 2], 
                 depth_list_dec=[2, 2, 2], depth_bneck=2, n_heads=[2, 3, 6, 12], mlp_factor=2.0, bias=True, proj_drop=0.2, att_drop=0.2, drop_path=0.2)
loss_fn = Dice()
optimizer = Adam(params=model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer)

# create a scaler
scaler = torch.amp.GradScaler()

# train
history = defaultdict(list)
epochs = range(NUM_EPOCHS)
for epoch in epochs:
    train_metrics = train_fn(train_loader=train_dataloader, model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, scaler=scaler, accumulation_steps=5, epoch=epoch, num_epochs=NUM_EPOCHS)
    val_metrics = val_fn(val_loader=val_dataloader, model=model, loss_fn=loss_fn)
    history['train_loss'] += [train_metrics['loss']]
    history['val_loss'] += [train_metrics['loss']]
    history['val_iou'] += [train_metrics['iou']]
plot_training_history(history, list(epochs))
