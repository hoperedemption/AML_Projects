import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
from utils import (load_zipped_pickle, preprocess_test_data, overlay_segmentation_countor_grayscale_for_gif)

def process_video(output, video):
    video_height, video_width = video.shape[-2], video.shape[-1]
    video_mask = np.zeros((output.shape[0], video_height, video_width))  # Initialize mask

    # Resize masks for the video
    for i, mask in enumerate(output):
        mask = cv2.resize(mask, (video_width, video_height), interpolation=cv2.INTER_LINEAR_EXACT)
        video_mask[i] = mask

    return video_mask

def create_gif_with_contours(video_array, mask_array, output_gif_path, fps=10, slowdown_factor=20):
    frames = []  # List to store the processed frames
    
    # Iterate over each frame
    for i in range(video_array.shape[0]):
        # Extract the video frame and mask
        frame = video_array[i]  # Shape: (height, width)
        mask = mask_array[i]    # Shape: (height, width)
        
        # Normalize and convert video frame to RGB for visualization
        frame_rgb = cv2.cvtColor((frame).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # Find contours in the mask
        mask_binary = (mask > 0).astype(np.uint8)  # Ensure mask is binary
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on the frame
        frame_with_contours = frame_rgb.copy()
        cv2.drawContours(frame_with_contours, contours, -1, color=(255,160,122), thickness=10)
        
        # Append the frame with contours to the list
        frames.append(frame_with_contours)
    
    # Save the list of frames as a GIF
    imageio.mimsave(output_gif_path, frames, duration=(1 / fps) * slowdown_factor )
    print(f"GIF saved to {output_gif_path}")
    
def create_gif_with_contours_overlay(video_array, mask_array, output_gif_path, fps=10, slowdown_factor=3, color='red'):
    frames = []  # List to store processed frames
    
    # Iterate over each frame and mask
    for i in range(video_array.shape[0]):
        frame = video_array[i]  # Grayscale video frame
        mask = mask_array[i]    # Corresponding binary mask

        # Overlay contours using the provided function
        overlay_frame = overlay_segmentation_countor_grayscale_for_gif(frame, mask, color=color)

        # Append the processed frame to the list
        frames.append(overlay_frame)
    
    # Adjust duration to slow down playback
    duration = (1 / fps) * slowdown_factor  # Multiply the frame duration by slowdown_factor
    
    # Save the list of frames as a GIF
    imageio.mimsave(output_gif_path, frames, duration=duration)
    print(f"GIF saved to {output_gif_path} with slowdown factor {slowdown_factor}")

if __name__ == "__main__":
    test_data = load_zipped_pickle('test.pkl')
    names, videos = preprocess_test_data(test_data)
    masks = load_zipped_pickle('outputs.pkl')
    
    first_video = videos[0].transpose(2, 0, 1)
    first_mask = process_video(masks[0], first_video)
    
    # Path to save the output GIF
    output_gif_path = "output_mistral_valve.gif"
    
    # Call the function to create GIF
    create_gif_with_contours_overlay(first_video, first_mask, output_gif_path)
