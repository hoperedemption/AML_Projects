import torch 
import numpy as np
import cv2
import pandas as pd
from joblib import Parallel, delayed

from utils import (load_zipped_pickle, preprocess_test_data, get_sequences)

THRESHOLD = 0.75

test_data = load_zipped_pickle('test.pkl')
names, videos = preprocess_test_data(test_data)

outputs = load_zipped_pickle('outputs.pkl')

def process_video(name, output, video, get_sequences_fn):
    video = video.transpose(2, 0, 1)
    video_height, video_width = video.shape[-2], video.shape[-1]
    video_mask = np.zeros((output.shape[0], video_height, video_width))  # Initialize mask

    # Resize masks for the video
    for i, mask in enumerate(output):
        mask = cv2.resize(mask, (video_width, video_height), interpolation=cv2.INTER_LINEAR_EXACT)
        video_mask[i] = mask

    # 0, 1, 2 -> 1, 2, 0
    video_mask = video_mask.transpose(1, 2, 0)
    # Flatten mask and get sequences
    first_indices, lengths = get_sequences_fn(video_mask.flatten())

    # Build name_ids and out_images for this video
    local_name_ids = [f"{name[0]}_{idx}" for idx in range(len(first_indices))]
    local_out_images = [f"[{first_indices[idx]}, {lengths[idx]}]" for idx in range(len(first_indices))]

    return local_name_ids, local_out_images

# Parallel execution
def parallel_process_videos(names, outputs, videos, get_sequences_fn, n_jobs=-1):
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_video)(name, output, video, get_sequences_fn)
        for name, output, video in zip(names, outputs, videos)
    )

    # Combine results
    name_ids, out_images = [], []
    for local_name_ids, local_out_images in results:
        name_ids.extend(local_name_ids)
        out_images.extend(local_out_images)

    return name_ids, out_images

# process_video('amog', outputs[0], videos[0], get_sequences)
name_ids, out_images = parallel_process_videos(names, outputs, videos, get_sequences, n_jobs=2)
    
df = pd.DataFrame({"id":name_ids, "value":out_images})
df.to_csv(f"mysubmissionfile_transformer_{THRESHOLD}.csv", index=False)