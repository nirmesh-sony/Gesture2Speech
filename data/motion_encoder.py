import os
import torch
import numpy as np
import torchvision.transforms as transforms
from decord import VideoReader, cpu
from PIL import Image
import torch.nn as nn

# Load Slow-R50 model
model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input
    transforms.ToTensor(),  # Convert to tensor after resizing
])

# Projection Layer (400 → 768)
projection = nn.Linear(400, 768)

# Parent Folder Containing Video Subfolders
parent_folder = "/hdd/lokesh/models/datasets/PATS/pats/data/video"
output_folder = "/hdd/lokesh/models/datasets/PATS/pats/data/motion_features1"

import torch.nn.functional as F

def extract_features(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    print(f"Total Frames in Video: {total_frames}")

    # Sample every 5th frame
    indices = list(range(0, total_frames, 5))

    # Convert frames properly
    frames = [transform(Image.fromarray(vr[i].asnumpy())) for i in indices]  # Convert to PIL

    # Stack frames into a batch
    video_tensor = torch.stack(frames).unsqueeze(0)  # Shape: [1, T, C, H, W]

    # Convert to (C, T, H, W) for Slow-R50 model
    video_tensor = video_tensor.permute(0, 2, 1, 3, 4)

    print(f"Input Tensor Shape: {video_tensor.shape}")

    # 🔹 Fix: Ensure at least 8 frames by padding
    min_frames = 8
    current_frames = video_tensor.shape[2]

    if current_frames < min_frames:
        pad_frames = min_frames - current_frames
        padding = video_tensor[:, :, -1:, :, :].repeat(1, 1, pad_frames, 1, 1)  # Repeat last frame
        video_tensor = torch.cat([video_tensor, padding], dim=2)  # Concatenate

        print(f"Padded to: {video_tensor.shape}")

    with torch.no_grad():
        features = model(video_tensor)  # ✅ Pass tensor directly
        features = projection(features)  # Apply linear projection (400 → 768)

    # return features.squeeze(0).numpy()  # Convert to NumPy Array
    return features.squeeze().unsqueeze(0).numpy()

# Process All Videos in Subfolders
for root, _, files in os.walk(parent_folder):
    for file in files:
        if file.endswith(('.mp4', '.avi', '.mov')):  # Add other formats if needed
            video_path = os.path.join(root, file)
            print(f"Processing: {video_path}")

            # Extract Features
            features = extract_features(video_path)

            # Create Corresponding Output Directory
            relative_path = os.path.relpath(root, parent_folder)
            save_dir = os.path.join(output_folder, relative_path)
            os.makedirs(save_dir, exist_ok=True)

            # Save as .npy File (Same Name as Video)
            feature_path = os.path.join(save_dir, os.path.splitext(file)[0] + ".npy")
            np.save(feature_path, features)

            print(f"Saved: {feature_path}")
