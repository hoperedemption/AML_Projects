import torch 
import torchvision
import pickle
import gzip
import numpy as np
import os
import cv2
import pandas as pd
import io
import imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from colorama import Fore, Style
from PIL import Image
from IPython.display import display, Image as IPImage
from rnmfbreg import robust_nmf_breg
from skimage import measure, feature
from scipy import ndimage
from skimage.filters import threshold_multiotsu
from rnmfbreg import robust_nmf_breg

def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def filter(filter_expert, filter_amateur, item):
    return (not filter_expert and not filter_amateur)  \
                or (filter_expert and item['dataset'] == 'expert') \
                or (filter_amateur and item['dataset'] == 'amateur')

def load_data(file_path, filter_expert=False, filter_amateur=False):
    # Load the data
    data = load_zipped_pickle(file_path)
    
    # The arrays we will be using
    names = []
    images = []
    segmentations = []
    box_images = []
    box_segmentations = []
    bounding_boxes = []
    
    with tqdm(data) as loop:
        for item in loop:
            if filter(filter_expert, filter_amateur, item):
                name, box = item['name'], item['box']
                imgs = item['video'][:, :, item['frames']]
                segs = item['label'][:, :, item['frames']]
                
                bounding_box_indices = np.argwhere(box)
                top_left = bounding_box_indices.min(axis=0)
                bottom_right = bounding_box_indices.max(axis=0)
                
                box_imgs = imgs[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1], :]
                box_segs = segs[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1], :]
                         
                for i in range(len(item['frames']) ):
                    img, seg = imgs[:, :, i], segs[:, :, i]
                    box_img, box_seg = box_imgs[:, :, i], box_segs[:, :, i]
                    names.append(name)
                    images.append(img)
                    segmentations.append(seg)
                    bounding_boxes.append(box)
                    box_images.append(box_img)
                    box_segmentations.append(box_seg)
                    
    return names, images, segmentations, bounding_boxes, box_images, box_segmentations

def load_data_smart(file_path, device='cpu'):
    # Load the data
    data = load_zipped_pickle(file_path)
    
     # The arrays we will be using
    images_amateur = []
    filtered_amateur = []
    segmentations_amateur = []
    images_expert = []
    filtered_expert = []
    segmentations_expert = []
    
    with tqdm(data) as loop:
        for item in loop:
            # load the data 
            video = item['video']
            seg = item['label']
            
            # Normalize the video
            min_val, max_val = video.min(), video.max()
            video = (video - min_val) / (max_val - min_val)
                
            # RNMF
            video_height, video_width, video_frames = video.shape[0], video.shape[1], video.shape[2] 
            W, H, S = robust_nmf_breg(video, n_comp=2, iterations=250, λ=0.02, tol=1e-6, device=device)
            rec = (W @ H).reshape(video_height, video_width, video_frames)
            
            # Normalize rec
            rec = (rec - rec.min()) / (rec.max() - rec.min())  
            rest = np.abs(video - rec)
            
            imgs = video[:, :, item['frames']]
            filtered = rest[:, :, item['frames']]
            segs = seg[:, :, item['frames']]
            
            
            for i in range(len(item['frames']) ):
                img, filt, seg = imgs[:, :, i], filtered[:, :, i], segs[:, :, i]
                
                if item['dataset'] == 'amateur':
                    images_amateur.append(img)
                    filtered_amateur.append(filt)
                    segmentations_amateur.append(seg)
                elif item['dataset'] == 'expert':
                    images_expert.append(img)
                    filtered_expert.append(filt)
                    segmentations_expert.append(seg)
                else:
                    raise ValueError('Not amateur or expert')
    
    dict = {
        'amateur_img': images_amateur, 
        'amateur_filt': filtered_amateur,
        'amateur_seg': segmentations_amateur, 
        'expert_img': images_expert, 
        'expert_filt': filtered_expert,
        'expert_seg': segmentations_expert
    }
    
    return dict        
            
        
def load_data_simple(file_path):
    # Load the data
    data = load_zipped_pickle(file_path)
    
    # The arrays we will be using
    images_amateur = []
    segmentations_amateur = []
    images_expert = []
    segmentations_expert = []
    
    with tqdm(data) as loop:
        for item in loop:
            imgs = item['video'][:, :, item['frames']]
            segs = item['label'][:, :, item['frames']]
      
            for i in range(len(item['frames']) ):
                img, seg = imgs[:, :, i], segs[:, :, i]
                
                if item['dataset'] == 'amateur':
                    images_amateur.append(img)
                    segmentations_amateur.append(seg)
                elif item['dataset'] == 'expert':
                    images_expert.append(img)
                    segmentations_expert.append(seg)
                else:
                    raise ValueError('Not amateur or expert')
    
    dict = {
        'amateur_img': images_amateur, 
        'amateur_seg': segmentations_amateur, 
        'expert_img': images_expert, 
        'expert_seg': segmentations_expert
    }
    
    return dict

def load_videos(file_path, filter_expert=False, filter_amateur=False):
    # data loading
    data = load_zipped_pickle(file_path)

    videos = []
    
    with tqdm(data) as loop:
        for item in loop:
            if filter(filter_expert, filter_amateur, item):
                    videos.append(item['video'])
                    
    return videos

def load_training_segmentation_data(file_path, filter_expert=False, filter_amateur=False, sparse=False):
    # data loading
    data = load_zipped_pickle(file_path)
    
    images, segmentations = [], []
    
    with tqdm(data) as loop:
        for item in loop:
            if filter(filter_expert, filter_amateur, item):
                # compute sparse if sparse is set to true
                video = item['video']
                if sparse:
                    _, _, S = robust_nmf_breg(video, n_comp=2, iterations=150, λ=0.01)
                    video = S
                
                # extract images and correspoding segmentations
                imgs = video[:, :, item['frames']]
                segs = item['label'][:, :, item['frames']]
                
                # extract bounding box information
                box = item['box']
                bounding_box_indices = np.argwhere(box)
                top_left = bounding_box_indices.min(axis=0)
                bottom_right = bounding_box_indices.max(axis=0)
                
                box_imgs = imgs[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1], :]
                box_segs = segs[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1], :]
                
                # append them to the list
                for i in range(len(item['frames'])):
                    images.append(box_imgs[:, :, i])
                    segmentations.append(box_segs[:, :, i])
    
    return images, segmentations                

def preprocess_old(image):
    normalised = image / np.max(image)
    
    sobel_h = ndimage.sobel(normalised, 0)
    sobel_w = ndimage.sobel(normalised, 1)

    magnitude = np.sqrt(sobel_h ** 2 + sobel_w ** 2)
    magnitude *= 255.0 / np.max(magnitude)
    
    edges = feature.canny(normalised, sigma=1).astype(np.float32)

    thresholds = threshold_multiotsu(normalised, classes=2)

    regions = np.digitize(normalised, bins=thresholds)
    
    return normalised, magnitude, edges, regions

def preprocess_new(rest):
    thresholds = threshold_multiotsu(rest, classes=2)
    regions = np.digitize(rest, bins=thresholds)
    return regions

def resize_segmentation_data(imgs, boxes, target_height, target_width):
    train_imgs, train_segs = np.zeros((len(imgs), target_height, target_width)), np.zeros((len(boxes), target_height, target_width))
    for i, (img, seg) in enumerate(zip(imgs, boxes)):
        img = cv2.resize(img, (target_height, target_width), interpolation=cv2.INTER_LINEAR_EXACT)
        seg = Image.fromarray(seg).resize((target_height, target_width), resample=Image.BILINEAR)
        train_imgs[i], train_segs[i] = img, np.asarray(seg) 
    return train_imgs, train_segs

def resize_segmentation_data_simple(imgs, segs, target_height, target_width):
    train_imgs, train_segs = np.zeros((len(imgs), target_height, target_width)), np.zeros((len(segs), target_height, target_width))
    for i, (img, seg) in enumerate(zip(imgs, segs)):
        if (img.shape[-2] != target_height or img.shape[-1] != target_width):
            img = cv2.resize(img, (target_height, target_width), interpolation=cv2.INTER_LINEAR_EXACT)
        if (seg.shape[-2] != target_height or seg.shape[-1] != target_width):
            seg = np.asarray(Image.fromarray(seg).resize((target_height, target_width), resample=Image.BILINEAR))
        train_imgs[i], train_segs[i] = img, seg
   
    return train_imgs, train_segs

def resize_segmentation_data_simple(imgs, filtered, segs, target_height, target_width):
    train_imgs, train_filt, train_segs = np.zeros((len(imgs), target_height, target_width)), np.zeros((len(filtered), target_height, target_width)), \
        np.zeros((len(segs), target_height, target_width))
    for i, (img, filt, seg) in enumerate(zip(imgs, filtered, segs)):
        if (img.shape[-2] != target_height or img.shape[-1] != target_width):
            img = cv2.resize(img, (target_height, target_width), interpolation=cv2.INTER_LINEAR_EXACT)
        if (filt.shape[-2] != target_height or filt.shape[-1] != target_width):
            filt = cv2.resize(filt, (target_height, target_width), interpolation=cv2.INTER_LINEAR_EXACT)
        if (seg.shape[-2] != target_height or seg.shape[-1] != target_width):
            seg = np.asarray(Image.fromarray(seg).resize((target_height, target_width), resample=Image.BILINEAR))
        train_imgs[i], train_filt[i], train_segs[i] = img, filt, seg
   
    return train_imgs, train_filt, train_segs

def resize_images(imgs, target_height, target_width):
    resized_imgs = np.zeros((len(imgs), target_height, target_width))
    for i, img in enumerate(imgs):
        if (img.shape[-2] != target_height or img.shape[-1] != target_width):
            img = cv2.resize(img, (target_height, target_width), interpolation=cv2.INTER_LINEAR_EXACT)
        resized_imgs[i] = img
        
    return resized_imgs

def resize_test_images(imgs, target_height, target_width):
    resized_imgs = np.zeros((target_height, target_width, imgs.shape[-1]))
    for i in range(imgs.shape[-1]):
        img = imgs[..., i]
        if (img.shape[-2] != target_height or img.shape[-1] != target_width):
            img = cv2.resize(img, (target_height, target_width), interpolation=cv2.INTER_LINEAR_EXACT)
        resized_imgs[..., i] = img
        
    return resized_imgs

def preprocess_segmentation_data(imgs, rest):     
    augmented_imgs = np.zeros((imgs.shape[0], 3, imgs.shape[1], imgs.shape[2]))
    for i, (img, rest_img) in enumerate(zip(imgs, rest)):
        multiotsu = preprocess_new(rest_img)
        ndarrays = [img, rest_img, multiotsu]
        augmented_img = np.concatenate([ndarray[None, ...] for ndarray in ndarrays], axis=0)
        augmented_imgs[i] = augmented_img
    return augmented_imgs

def overlay_segmentation_countor_grayscale(image, mask, bounding_box=None, color='red', colormap=None, filename="segmented_image.png"):
     # Normalize grayscale image to [0, 1] for blending
    if image.max() > 1:
        image = image / 255.0
    
    # Convert grayscale to RGB for consistent blending
    image_rgb = np.stack([image] * 3, axis=-1)  # (H, W, 3)
    
    # Plot the overlay with bounding boxes
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Find contours
    contours = measure.find_contours(mask, level=0.5)

    # Plot the image
    if colormap is None:
        ax.imshow(image_rgb, interpolation='nearest')
    else: # only when using regions
        ax.imshow(image, interpolation='nearest', cmap=colormap)

    # Plot the contours on top of the image
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color=color)  # Adjust color and linewidth as needed
        
    if bounding_box is not None:
        # Compute the bounding box coordinates (min/max row and column)
        bounding_box_indices = np.argwhere(bounding_box)
        top_left = bounding_box_indices.min(axis=0)  # (min_row, min_col)
        bottom_right = bounding_box_indices.max(axis=0)  # (max_row, max_col)

        # Draw the bounding box on the plot
        rect = Rectangle(
            (top_left[1], top_left[0]),  # (x, y): (col, row)
            bottom_right[1] - top_left[1] + 1,  # width
            bottom_right[0] - top_left[0] + 1,  # height
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
    
    ax.axis('off')
    ax.set_title("Segmentation Image")
    fig.savefig(filename)
    plt.show()
    return fig, ax
    
def overlay_segmentation_countor_grayscale_for_gif(image, mask, bounding_box=None, color='red', colormap=None):
    fig, ax = overlay_segmentation_countor_grayscale(image, mask, bounding_box=bounding_box, color=color, colormap=colormap)
    # Remove axes and margins
    ax.axis('off')
    fig.tight_layout(pad=0)

    # Save the plot as an RGB array
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    overlay_image = imageio.imread(buf)  # Read the image as a NumPy array

    plt.close(fig)  # Close the figure to release memory
    return overlay_image
    
                
def overlay_segmentation_grayscale(image, mask, bounding_box=None, alpha=0.7, colormap='cividis'):
    # Normalize grayscale image to [0, 1] for blending
    if image.max() > 1:
        image = image / 255.0

    # Convert grayscale to RGB for consistent blending
    image_rgb = np.stack([image] * 3, axis=-1)  # (H, W, 3)
    
    # Apply colormap to the mask
    cmap = plt.get_cmap(colormap)
    colored_mask = cmap(mask / mask.max())  # Normalize mask to [0, 1]
    colored_mask = colored_mask[:, :, :3]  # Remove alpha channel if present

    # Overlay the colored mask on top of the grayscale image (replace pixels)
    overlay = image_rgb.copy()
    overlay[mask] = (1 - alpha) * image_rgb[mask] + alpha * colored_mask[mask]
    
    # Plot the overlay with bounding boxes
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if bounding_box is not None:
        # Compute the bounding box coordinates (min/max row and column)
        bounding_box_indices = np.argwhere(bounding_box)
        top_left = bounding_box_indices.min(axis=0)  # (min_row, min_col)
        bottom_right = bounding_box_indices.max(axis=0)  # (max_row, max_col)

        # Draw the bounding box on the plot
        rect = Rectangle(
            (top_left[1], top_left[0]),  # (x, y): (col, row)
            bottom_right[1] - top_left[1] + 1,  # width
            bottom_right[0] - top_left[0] + 1,  # height
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

    ax.imshow(overlay)
    ax.axis('off')
    ax.set_title("Segmentation Image")
    fig.savefig('segmented_image.png')
    plt.show()

def visualize_pixel_intensities(image, colormap='gray', title="Pixel Intensity Visualization", filename='pixel_intensity.png'):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Display the image with a colormap
    cax = ax.imshow(image, cmap=colormap, vmin=image.min(), vmax=image.max())
    
    # Add a color bar to the side
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Pixel Intensity Values")
    
    # Add title and remove axes for clarity
    ax.set_title(title)
    ax.axis('off')  # Remove axes for better visibility
    
    # Show the figure
    plt.show()
    plt.savefig(filename)

def convert_video_to_gif(video, display_in_jupyter=False):
     # Convert each frame to a PIL Image
    frames = [Image.fromarray(video[:, :, i]) for i in range(video.shape[2])]

    # Save as GIF
    output_path = "video.gif"
    frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0, duration=50)  # duration in milliseconds

    print(f"GIF saved at {output_path}")
    
    if display_in_jupyter:
        # Display in Jupyter Notebook (if using one)
        display(IPImage(filename=output_path))

def print_checkpoint_saving(filepath):
    print(f"{Fore.GREEN}{Style.BRIGHT}")
    print("=" * 50)
    print("💾 Saving PyTorch Checkpoint")
    print("-" * 50)
    print(f"📂 File: {filepath}")
    print("=" * 50)
    print(f"{Style.RESET_ALL}")

def print_checkpoint_loading(filepath):
    print(f"{Fore.CYAN}{Style.BRIGHT}")
    print("=" * 50)
    print("🚀 Loading PyTorch Checkpoint")
    print("-" * 50)
    print(f"📂 File: {filepath}")
    print("=" * 50)
    print(f"{Style.RESET_ALL}")
    
def print_evaluation_metrics(metrics, epoch=None):
    print(f"{Fore.YELLOW}{Style.BRIGHT}")
    print("=" * 50)
    print("📊 Model Evaluation Results")
    if epoch is not None:
        print(f"🕒 Epoch: {epoch}")
    print("=" * 50)
    for metric, value in metrics.items():
        emoji = "✅" if metric == "iou" else "📉" if "loss" in metric else "📈"
        color = Fore.GREEN if value > 0.9 else Fore.RED if value < 0.5 else Fore.WHITE
        print(f"{color}{emoji} {metric.capitalize()}: {value:.4f}")
    print("=" * 50)
    print(f"{Style.RESET_ALL}")

def save_checkpoint(model, model_name, filename='model.pth.tar'):
    filepath = f'{model_name}_' + filename
    print_checkpoint_saving(filepath)
    torch.save(model.state_dict(), filepath)
    return filepath
    
def load_checkpoint(filepath, model):
    print_checkpoint_loading(filepath)
    checkpoint = torch.load(filepath, weights_only=True, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    return model

def plot_training_history(history: dict, epochs, title="Training and Validation Metrics", filename='plot_metrics.png'):
    # Validate input
    required_keys = ['train_loss', 'val_loss', 'val_iou']
    assert all([any([rkey == key for rkey in required_keys]) for key in history.keys()])

    plt.figure(figsize=(14, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Training Loss', marker='o', linestyle='-', color='red')
    plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='o', linestyle='--', color='blue')
    plt.title("Loss", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_iou'], label='Validation IoU', marker='o', linestyle='--', color='green')
    plt.title("Accuracy", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)

    # Main title and layout
    plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.savefig(filename)

def preprocess_test_data(data):
    videos = []
    names = []
    for item in tqdm(data):
        video = item['video']
        video = video.astype(np.float32)
        videos.append(video)
        names.append([item['name']])
    return names, videos

def get_sequences(arr):
    first_indices, last_indices, lengths = [], [], []
    n, i = len(arr), 0
    arr = [0] + list(arr) + [0]
    for index, value in enumerate(arr[:-1]):
        if arr[index+1]-arr[index] == 1:
            first_indices.append(index)
        if arr[index+1]-arr[index] == -1:
            last_indices.append(index)
    lengths = list(np.array(last_indices)-np.array(first_indices))
    return first_indices, lengths