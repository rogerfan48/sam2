#!/usr/bin/env python3
"""
SAM2 Video Segmentation Script (2 Objects + Background)
========================================================

This script generates masks for video frames:
- obj1: First object of interest
- obj2: Second object of interest
- bg: Background (automatically generated as everything except obj1 and obj2)

Total: 2 objects + 1 background = 3 masks

PREREQUISITES:
--------------
Run setup_dataset.sh first to convert videos to frames.
Expected frame format: 00000.jpg, 00001.jpg, 00002.jpg, ... (5-digit, 0-indexed)

USAGE:
------
1. Configure the parameters in the CONFIG section below
2. Run: python create-masks-2obj-bg.py
3. Check the output in VIDEO_DIR/masks/

The script will:
- Load frames from VIDEO_DIR/frames/
- Generate masks for obj1 and obj2 based on your click points
- Automatically generate background mask (inverse of obj1 + obj2)
- Save masks to VIDEO_DIR/masks/obj1/, obj2/, bg/
- Create a visualization image

COORDINATE SYSTEM:
------------------
- Click coordinates are in pixels (x, y)
- Origin (0, 0) is at top-left corner
- x increases rightward, y increases downward
- Example: For 512x512 image, center is at (256, 256)

FRAME INDEXING:
---------------
- Frame indices are 0-based, matching the filename format
- Frame 0 = 00000.jpg, Frame 1 = 00001.jpg, etc.
- When specifying OBJ1_FRAME_IDX = 0, it uses 00000.jpg

TIPS:
-----
- Choose frames where each object is clearly visible
- Add multiple click points on different parts of each object for better coverage
- Use positive labels (1) for points inside the object
- Use negative labels (0) to exclude unwanted areas
"""

import os
import numpy as np
import torch
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image

# Suppress UserWarnings from SAM2 (CUDA kernel warnings that don't affect results)
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# CONFIG SECTION - MODIFY THESE PARAMETERS
# ============================================================================

# Video Directory (should contain frames/ subdirectory)
WORKING_DIR = "/work/rogerfan48/sam2"
VIDEO_NAME = "not-linear-trajectory3-skirt"  # Name of the video directory
VIDEO_DIR = f"{WORKING_DIR}/assets/ycmong-dataset/{VIDEO_NAME}/frames"
OUTPUT_DIR = f"{WORKING_DIR}/assets/ycmong-dataset/{VIDEO_NAME}/masks"

# SAM2 Model Configuration
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Frame Selection (0-indexed, matching SAM2 default format)
# Frame 0 = 00000.jpg, Frame 10 = 00010.jpg, etc.
OBJ1_FRAME_IDX = 0      # Frame number for obj1 (0 = 00000.jpg)
OBJ2_FRAME_IDX = 0      # Frame number for obj2 (10 = 00010.jpg)
VISUALIZATION_FRAME_IDX = 0  # Frame to use for visualization output

# Object 1: Click Points and Labels
# Format: [(x1, y1), (x2, y2), ...]
# Labels: 1 = positive (inside object), 0 = negative (outside object)
OBJ1_POINTS = [
    (185, 85),
    (160, 170),
    (220, 300),
    (210, 150),
    (277, 166)
]
OBJ1_LABELS = [1, 1, 1, 1, 0]  # All positive clicks

# Object 2: Click Points and Labels
OBJ2_POINTS = [
    (270, 205),
    (255, 200),
    (300, 215),
    (307, 266),
    (260, 185)
]
OBJ2_LABELS = [1, 1, 1, 0, 0]  # All positive clicks

# ============================================================================
# END OF CONFIG SECTION
# ============================================================================

print("=" * 70)
print("SAM2 Video Segmentation (2 Objects + Background)")
print("=" * 70)

# Setup device
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.chdir(WORKING_DIR)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# Load SAM2 video predictor
from sam2.build_sam import build_sam2_video_predictor

print("Loading SAM2 model...")
predictor = build_sam2_video_predictor(MODEL_CONFIG, SAM2_CHECKPOINT, device=device)
print("✓ Model loaded successfully")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Scan video frames (expecting JPG format from setup_dataset.sh)
frame_names = [
    p for p in os.listdir(VIDEO_DIR)
    if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

print(f"\nVideo Information:")
print(f"  Video: {VIDEO_NAME}")
print(f"  Frames directory: {VIDEO_DIR}")
print(f"  Total frames: {len(frame_names)}")
print(f"  Frame range: {frame_names[0]} to {frame_names[-1]}")

# Get frame dimensions
first_frame = Image.open(os.path.join(VIDEO_DIR, frame_names[0]))
width, height = first_frame.size
print(f"  Frame size: {width}x{height}")

# Validate frame indices
max_frame = len(frame_names)
if not (0 <= OBJ1_FRAME_IDX < max_frame):
    raise ValueError(f"OBJ1_FRAME_IDX ({OBJ1_FRAME_IDX}) out of range [0, {max_frame-1}]")
if not (0 <= OBJ2_FRAME_IDX < max_frame):
    raise ValueError(f"OBJ2_FRAME_IDX ({OBJ2_FRAME_IDX}) out of range [0, {max_frame-1}]")
if not (0 <= VISUALIZATION_FRAME_IDX < max_frame):
    raise ValueError(f"VISUALIZATION_FRAME_IDX ({VISUALIZATION_FRAME_IDX}) out of range [0, {max_frame-1}]")

# Convert click points to numpy arrays
obj1_points = np.array(OBJ1_POINTS, dtype=np.float32)
obj1_labels = np.array(OBJ1_LABELS, dtype=np.int32)
obj2_points = np.array(OBJ2_POINTS, dtype=np.float32)
obj2_labels = np.array(OBJ2_LABELS, dtype=np.int32)

# Validate labels
if len(obj1_points) != len(obj1_labels):
    raise ValueError("OBJ1_POINTS and OBJ1_LABELS must have same length")
if len(obj2_points) != len(obj2_labels):
    raise ValueError("OBJ2_POINTS and OBJ2_LABELS must have same length")

print(f"\nObject Configuration:")
print(f"  obj1: {len(obj1_points)} points on frame {OBJ1_FRAME_IDX} ({frame_names[OBJ1_FRAME_IDX]})")
print(f"  obj2: {len(obj2_points)} points on frame {OBJ2_FRAME_IDX} ({frame_names[OBJ2_FRAME_IDX]})")
print(f"  bg: Auto-generated (inverse of obj1 + obj2)")

# Initialize inference state
inference_state = predictor.init_state(video_path=VIDEO_DIR)

# Define objects info
objects_info = [
    ("obj1", obj1_points, obj1_labels, 1, OBJ1_FRAME_IDX),
    ("obj2", obj2_points, obj2_labels, 2, OBJ2_FRAME_IDX),
]

print("\n" + "=" * 70)
print("PROCESSING OBJECTS")
print("=" * 70)

# Add objects to SAM2
for obj_name, points, labels, obj_id, frame_idx in objects_info:
    print(f"\nAdding {obj_name}:")
    print(f"  Frame: {frame_idx} ({frame_names[frame_idx]})")
    print(f"  Points: {len(points)}")
    print(f"  Object ID: {obj_id}")

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
    )
    print(f"  ✓ Successfully added {obj_name}")

print("\nPropagating masks through all frames...")

# Run propagation throughout the video
# Use start_frame_idx=0 to ensure we get masks from frame 0 onwards
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
    inference_state,
    start_frame_idx=0
):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

print(f"✓ Object masks generated for {len(video_segments)} frames")

print("\n" + "=" * 70)
print("GENERATING BACKGROUND MASKS")
print("=" * 70)

# Generate background masks (inverse of obj1 + obj2)
for frame_idx in video_segments:
    # Get frame dimensions from first object mask
    frame_height, frame_width = video_segments[frame_idx][1].shape[-2:]
    background_mask = np.ones((frame_height, frame_width), dtype=bool)

    # Remove areas where obj1 or obj2 are detected
    for obj_id in [1, 2]:
        if obj_id in video_segments[frame_idx]:
            obj_mask = video_segments[frame_idx][obj_id].squeeze()
            background_mask = background_mask & (~obj_mask)

    # Add background mask with obj_id = 3
    video_segments[frame_idx][3] = background_mask

print(f"✓ Background masks generated for {len(video_segments)} frames")

# Create output directories
all_objects = [
    ("obj1", 1),
    ("obj2", 2),
    ("bg", 3)
]

for obj_name, _ in all_objects:
    obj_dir = os.path.join(OUTPUT_DIR, obj_name)
    os.makedirs(obj_dir, exist_ok=True)

print("\n" + "=" * 70)
print("SAVING MASKS")
print("=" * 70)

# Save masks for each frame (0-indexed naming to match input)
for frame_idx in range(len(frame_names)):
    if frame_idx in video_segments:
        for obj_name, obj_id in all_objects:
            if obj_id in video_segments[frame_idx]:
                # Get mask for this object
                mask = video_segments[frame_idx][obj_id].squeeze()

                # Convert mask to uint8 format (0-255)
                mask_uint8 = (mask * 255).astype(np.uint8)

                # Save mask with same naming as input (0-indexed, 5-digit)
                mask_filename = f"{frame_idx:05d}.png"
                obj_dir = os.path.join(OUTPUT_DIR, obj_name)
                mask_path = os.path.join(obj_dir, mask_filename)

                # Use PIL to save
                mask_pil = Image.fromarray(mask_uint8, mode='L')
                mask_pil.save(mask_path)

    if (frame_idx + 1) % 10 == 0:
        print(f"  Processed frame {frame_idx:05d}")

print(f"\n✓ All masks saved to: {OUTPUT_DIR}")
print(f"  Total frames processed: {len(frame_names)}")

# Create visualization
print("\n" + "=" * 70)
print("CREATING VISUALIZATION")
print("=" * 70)

visualization_frame = Image.open(os.path.join(VIDEO_DIR, frame_names[VISUALIZATION_FRAME_IDX]))
if VISUALIZATION_FRAME_IDX in video_segments:
    visualization_frame_array = np.array(visualization_frame)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Original frame
    axes[0, 0].imshow(visualization_frame_array)
    axes[0, 0].set_title(f"Original Frame {VISUALIZATION_FRAME_IDX:05d}", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Show each mask
    mask_positions = [(0, 1), (1, 0), (1, 1)]
    mask_titles = ["Object 1", "Object 2", "Background"]

    for idx, ((obj_name, obj_id), title) in enumerate(zip(all_objects, mask_titles)):
        row, col = mask_positions[idx]

        if obj_id in video_segments[VISUALIZATION_FRAME_IDX]:
            mask = video_segments[VISUALIZATION_FRAME_IDX][obj_id].squeeze()
            axes[row, col].imshow(visualization_frame_array)
            axes[row, col].imshow(mask, alpha=0.6, cmap='jet')
            axes[row, col].set_title(f"{title} ({obj_name})", fontsize=14, fontweight='bold')
        else:
            axes[row, col].text(0.5, 0.5, f"No {title} mask",
                               ha='center', va='center', fontsize=12)
            axes[row, col].set_title(f"{title} ({obj_name}) - Not Found",
                                    fontsize=14, fontweight='bold')

        axes[row, col].axis('off')

    viz_path = os.path.join(OUTPUT_DIR, "three_masks_visualization.png")
    plt.tight_layout()
    plt.savefig(viz_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"✓ Visualization saved to: {viz_path}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"✓ Successfully generated masks (2 objects + background):")
print(f"  - obj1: {OUTPUT_DIR}/obj1/")
print(f"  - obj2: {OUTPUT_DIR}/obj2/")
print(f"  - bg: {OUTPUT_DIR}/bg/")
print(f"\n✓ Total frames: {len(frame_names)}")
print(f"✓ Frame naming: {frame_names[0]} to {frame_names[-1]} (input)")
print(f"✓ Mask naming: 00000.png to {len(frame_names)-1:05d}.png (output)")
print(f"\n✓ Background generated using smart approach:")
print(f"  (everything NOT covered by obj1 or obj2)")
print("\n" + "=" * 70)
print("DONE! Masks ready for use.")
print("=" * 70)
