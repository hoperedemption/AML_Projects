import torch 
import torchvision
import pickle
import gzip
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from colorama import Fore, Style
from PIL import Image
from IPython.display import display, Image as IPImage
from rnmfbreg import robust_nmf_breg
from skimage import measure

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
                
def overlay_segmentation_countor_grayscale(image, mask, bounding_box=None, alpha=0.7, color='red'):
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
    ax.imshow(image_rgb, interpolation='nearest')

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
    fig.savefig('segmented_image.png')
    plt.show()
    
                
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

def save_checkpoint(checkpoint, model_name, filename='model.pth.tar'):
    filepath = 'checkpoints/' + f'{model_name}_' + filename
    print_checkpoint_saving(filepath)
    torch.save(checkpoint, filename)
    
def load_checkpoint(filepath, model, optimizer):
    print_checkpoint_loading(filepath)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['state_dict'])

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

