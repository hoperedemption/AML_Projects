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


def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object


def load_data(file_path, filter_expert=False):
    # Load the data
    data = load_zipped_pickle(file_path)

    # The arrays we will be using
    names = []
    images = []
    segmentations = []
    bounding_boxes = []

    with tqdm(data) as loop:
        for item in loop:
            if not filter_expert or item['dataset'] == 'expert':
                name, box = item['name'], item['box']
                imgs = item['video'][:, :, item['frames']]
                segs = item['label'][:, :, item['frames']]

                for i in range(len(item['frames'])):
                    img, seg = imgs[:, :, i], segs[:, :, i]
                    names.append(name)
                    images.append(img)
                    segmentations.append(seg)
                    bounding_boxes.append(box)

    return names, images, segmentations, bounding_boxes


def overlay_segmentation_grayscale(image, mask, bounding_box, alpha=0.7, colormap='cividis'):
    # Normalize grayscale image to [0, 1] for blending
    if image.max() > 1:
        image = image / 255.0

    # Convert grayscale to RGB for consistent blending
    image_rgb = np.stack([image] * 3, axis=-1)  # Shape: (H, W, 3)

    # Apply colormap to the mask
    cmap = plt.get_cmap(colormap)
    colored_mask = cmap(mask / mask.max())  # Normalize mask to [0, 1]
    colored_mask = colored_mask[:, :, :3]  # Remove alpha channel if present

    # Overlay the colored mask on top of the grayscale image (replace pixels)
    overlay = image_rgb.copy()
    overlay[mask] = (1 - alpha) * image_rgb[mask] + alpha * colored_mask[mask]

    # Plot the overlay with bounding boxes
    fig, ax = plt.subplots(figsize=(8, 8))

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
    plt.show()
    plt.savefig('segmented_image.png')


def convert_video_to_gif(video, display_in_jupyter=False):
    # Convert each frame to a PIL Image
    frames = [Image.fromarray(video[:, :, i]) for i in range(video.shape[2])]

    # Save as GIF
    output_path = "video.gif"
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   loop=0, duration=50)  # duration in milliseconds

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
        emoji = "✅" if metric == "accuracy" else "📉" if "loss" in metric else "📈"
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
