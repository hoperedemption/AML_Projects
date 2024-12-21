import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import pandas as pd
from torch.nn import BCELoss
from PIL import Image
from SwinUNet import SwinUNET
from Models.DUCKNet import DUCKNet
from Models.UNET import UNET
from Models.TransUNET import TransUNet
from Models.UNETPP import UNETPP
from utils import (load_zipped_pickle, overlay_segmentation_grayscale,
                   overlay_segmentation_countor_grayscale, print_evaluation_metrics,
                   plot_training_history, preprocess_segmentation_data, resize_segmentation_data,
                   load_data_simple, resize_segmentation_data_simple, save_checkpoint, load_checkpoint,
                   preprocess_test_data, resize_images, get_sequences, load_data_smart)
from training_utils_pytorch import CustomDataset
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from losses import Dice, IoU, BinaryFocalIoU
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from collections import defaultdict

LR = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.device(DEVICE)
torch.set_default_device(DEVICE)
torch.autograd.set_detect_anomaly(True)
BATCH_SIZE = 5
NUM_EPOCHS_AMATEUR = 30
NUM_EPOCHS_EXPERT = 45
NUM_EPOCHS_LAST = 10
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
THRESHOLD = 0.75


def train_fn(train_loader, model, optimizer, scheduler, loss_fn, epoch, num_epochs):
    model.train()

    metrics = defaultdict(list)

    with tqdm(train_loader, total=len(train_loader), leave=False) as train_loop:
        for data, target in train_loop:
            # zero out all previous gradients
            optimizer.zero_grad()

            # run the forward pass and compute the loss
            output = model(data)

            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            metrics['loss'] += [loss]

            # update the tqdmp loop
            train_loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            train_loop.set_postfix(loss=loss.item())
    metrics['loss'] = torch.mean(torch.tensor(metrics['loss']))
    scheduler.step(metrics['loss'])
    metrics['loss'] = metrics['loss'].cpu().item()
    return metrics


def iou(gt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    # gt, mask -> (B, 1, H, W)
    gt, mask = gt.to(torch.int32), mask.to(torch.int32)
    intersection = (gt & mask).to(torch.float32).sum((-2, -1))
    union = (gt | mask).to(torch.float32).sum((-2, -1))
    iou = (intersection + eps) / (union + eps)
    return iou.mean()  # average across batches


def val_fn(val_loader, model, loss_fn, epoch=None):
    model.eval()

    metrics = defaultdict(list)

    with torch.no_grad():
        for data, target in val_loader:
            # get the output
            output = model(data)

            # evaluate the loss
            loss = loss_fn(output, target)
            # evaluate the accuracy
            inter_over_union = iou(
                target, torch.where(output >= THRESHOLD, 1, 0))

            # get metrics dict
            metrics['iou'] += [inter_over_union]
            metrics['loss'] += [loss]

    metrics['iou'] = torch.mean(torch.tensor(metrics['iou']))
    metrics['loss'] = torch.mean(torch.tensor(metrics['loss']))
    print_evaluation_metrics(metrics, epoch)

    metrics['iou'] = metrics['iou'].cpu().numpy()
    metrics['loss'] = metrics['loss'].cpu().numpy()
    return metrics


def train_one_go(train_dataloader, model, optimizer, scheduler, loss_fn, n_epochs, check=False, best_val=None):
    filepath = None
    history = defaultdict(list)
    epochs = range(n_epochs)
    for epoch in epochs:
        train_metrics = train_fn(train_loader=train_dataloader, model=model, optimizer=optimizer,
                                 scheduler=scheduler, loss_fn=loss_fn, epoch=epoch, num_epochs=n_epochs)
        val_metrics = val_fn(val_loader=val_expert_dataloader,
                             model=model, loss_fn=loss_fn, epoch=epoch)
        history['train_loss'] += [train_metrics['loss']]
        history['val_loss'] += [val_metrics['loss']]
        history['val_iou'] += [val_metrics['iou']]
        if check and best_val is not None:
            if val_metrics['iou'] >= best_val:
                best_val = val_metrics['iou']
                filepath = save_checkpoint(model, 'best_transunet')
    return history, epochs, best_val, filepath


def train_with_history(train_dataloader, model, optimizer, scheduler, loss_fn, n_epochs, filename, check=False, best_val=None):
    history, epochs, best_val, filepath = train_one_go(
        train_dataloader, model, optimizer, scheduler, loss_fn, n_epochs, check, best_val)
    plot_training_history(history, list(epochs), filename=filename)
    return filepath, best_val


print(' --- Loading Data ---')
dict = load_data_smart('../../data/train.pkl', device=DEVICE)
print(' --- End Load ---')

train_amateur_imgs, train_amateur_segs = dict['amateur_img'], dict['amateur_seg']
train_amateur_filtered, expert_filtered = dict['amateur_filt'], dict['expert_filt']
expert_imgs, expert_segs = dict['expert_img'], dict['expert_seg']

expert_imgs, expert_filtered, expert_segs = resize_segmentation_data_simple(
    expert_imgs, expert_filtered, expert_segs, IMAGE_HEIGHT, IMAGE_WIDTH)

train_amateur_imgs, train_amateur_segs, train_amateur_filtered, expert_filtered, expert_imgs, expert_segs = [np.array(arr)
                                                                                                             for arr in [train_amateur_imgs, train_amateur_segs, train_amateur_filtered, expert_filtered, expert_imgs, expert_segs]]
train_amateur_imgs, expert_imgs = preprocess_segmentation_data(train_amateur_imgs, train_amateur_filtered), \
    preprocess_segmentation_data(expert_imgs, expert_filtered)

train_expert_imgs, val_expert_imgs, train_expert_segs, val_expert_segs = train_test_split(
    expert_imgs, expert_segs, test_size=0.2, random_state=42)


def convert_to_tensor(x): return torch.from_numpy(
    x).to(torch.float32).to(DEVICE)


train_amateur_imgs, train_amateur_segs = convert_to_tensor(
    train_amateur_imgs), convert_to_tensor(train_amateur_segs)
train_expert_imgs, train_expert_segs = convert_to_tensor(
    train_expert_imgs), convert_to_tensor(train_expert_segs)
val_expert_imgs, val_expert_segs = convert_to_tensor(
    val_expert_imgs), convert_to_tensor(val_expert_segs)
expert_imgs, expert_segs = convert_to_tensor(
    expert_imgs), convert_to_tensor(expert_segs)

train_amateur_dataset = CustomDataset(
    train_amateur_imgs, train_amateur_segs, channels=3, device=DEVICE, apply_transform=True)
train_expert_dataset = CustomDataset(
    train_expert_imgs, train_expert_segs, channels=3, device=DEVICE, apply_transform=True)
last_expert_dataset = CustomDataset(
    expert_imgs, expert_segs, channels=3, device=DEVICE, apply_transform=False)
val_expert_dataset = CustomDataset(
    val_expert_imgs, val_expert_segs, channels=3, device=DEVICE, apply_transform=False)

train_amateur_dataloader = DataLoader(
    dataset=train_amateur_dataset, batch_size=BATCH_SIZE)
train_expert_dataloader = DataLoader(
    dataset=train_expert_dataset, batch_size=BATCH_SIZE)
last_expert_dataloader = DataLoader(
    dataset=last_expert_dataset, batch_size=BATCH_SIZE)
val_expert_dataloader = DataLoader(
    dataset=val_expert_dataset, batch_size=BATCH_SIZE)

kernel_size = (3, 3)
stride = (1, 1)
padding = (1, 1)

# model = UNET(in_channels=3, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding).to(DEVICE)
# model = DUCKNet(in_channels=4, out_channels=1, depth=5).to(DEVICE)
model = TransUNet(in_dim=3, out_dim=1, image_shape=(
    IMAGE_HEIGHT, IMAGE_WIDTH), n_patches=4, device=DEVICE).to(DEVICE)
loss_fn = BinaryFocalIoU()
optimizer = Adam(params=model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer)

# train first on amateur
train_with_history(train_amateur_dataloader, model, optimizer,
                   scheduler, loss_fn, NUM_EPOCHS_AMATEUR, 'amateur_history.png')

# # them train on experts
filepath, best_val = train_with_history(train_expert_dataloader, model, optimizer,
                                        scheduler, loss_fn, NUM_EPOCHS_EXPERT, 'expert_history.png', check=True, best_val=0.0)

# lastly train on whole expert dataset
filepath, _ = train_with_history(last_expert_dataloader, model, optimizer, scheduler,
                                 loss_fn, NUM_EPOCHS_LAST, 'last_history.png', check=True, best_val=best_val)

# predictions
print(filepath)
