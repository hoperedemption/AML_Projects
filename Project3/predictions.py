import torch
import cv2 
import numpy as np
import torch.nn as nn 
import pandas as pd
from tqdm import tqdm
from UNET import UNET
from TransUNET import TransUNet
from rnmfbreg import robust_nmf_breg
from torch.utils.data import TensorDataset, DataLoader
from utils import (resize_test_images, preprocess_segmentation_data, load_checkpoint, load_zipped_pickle, preprocess_test_data, 
                   save_zipped_pickle)

filepath = 'best_transunet_model.pth.tar'
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
THRESHOLD = 0.7
BATCH_SIZE = 5

convert_to_tensor = lambda x: torch.from_numpy(x).to(torch.float32).to(DEVICE)

kernel_size = (3, 3)
stride = (1, 1)
padding = (1, 1)

# model = UNET(in_channels=3, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding).to(DEVICE)
model = TransUNet(in_dim=3, out_dim=1, image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH), n_patches=4, device=DEVICE).to(DEVICE)

load_checkpoint(filepath, model)
model.eval()

# load the data
test_data = load_zipped_pickle('test.pkl')
names, videos = preprocess_test_data(test_data)

# do the predictions on the test data
print('----------- PREDICTIONS -----------')
mask_list = []

with tqdm(zip(names, videos), total=len(names), leave=False) as predictions_loop:
    for name, video in predictions_loop:
        video = np.array(resize_test_images(video, IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # normalize
        min_val, max_val = video.min(), video.max()
        video = (video - min_val) / (max_val - min_val)
        
        # RNMF
        video_height, video_width, video_frames = video.shape[0], video.shape[1], video.shape[2] 
        W, H, S = robust_nmf_breg(video, n_comp=2, iterations=250, λ=0.02, tol=1e-6, device=DEVICE)
        rec = (W @ H).reshape(video_height, video_width, video_frames)
        
        # Normalize rec
        rec = (rec - rec.min()) / (rec.max() - rec.min())  
        rest = np.abs(video - rec)
        
        video, rest = video.transpose(2, 0, 1), rest.transpose(2, 0, 1)
        
        test_images = preprocess_segmentation_data(video, rest)
        test_images = convert_to_tensor(test_images)
        
        tensor_dataset = TensorDataset(test_images)
        dataloader = DataLoader(tensor_dataset, batch_size=BATCH_SIZE)
        
        outputs = []
        
        for sample in dataloader:
            sample = sample[0]
            output = model(sample)
            outputs.append(torch.where(output >= THRESHOLD, 1, 0).squeeze(1))
        
        outputs = torch.cat(outputs, dim=0).cpu().numpy()
    
        mask_list.append(outputs)

print('----------- PREDICTIONS -----------')

save_zipped_pickle(mask_list, 'outputs.pkl')
print('Predictions saved successfully to outputs.pkl')

