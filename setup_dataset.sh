#!/bin/bash

# =============================================================================
# Video Dataset Setup Script
# =============================================================================
# This script processes MP4 videos and prepares them for SAM2 processing
#
# USAGE:
#   1. Place your .mp4 files in SOURCE_DIR
#   2. Run: bash setup_dataset.sh
#   3. The script will create for each video (e.g., "myvideo.mp4"):
#      - myvideo/frames/00000.jpg, 00001.jpg, ... (extracted frames)
#      - myvideo/masks/ (empty, for SAM2 output)
#      - myvideo/myvideo.mp4 (original video)
#
# OUTPUT FORMAT:
#   Frame naming: 5-digit, 0-indexed (00000.jpg, 00001.jpg, ..., 00099.jpg)
#   Compatible with SAM2 default format
# =============================================================================

# --- Configuration ---
SOURCE_DIR="./assets/ycmong-dataset"
# --------------------

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null
then
    echo "Error: ffmpeg not found."
    echo "Please install ffmpeg first (e.g., apt install ffmpeg, brew install ffmpeg)"
    exit 1
fi

echo "========================================================================"
echo "Video Dataset Setup - Converting MP4 to Frames"
echo "========================================================================"
echo "Source directory: $SOURCE_DIR"
echo ""

# Count total MP4 files
mp4_count=$(find "$SOURCE_DIR" -maxdepth 1 -name "*.mp4" | wc -l)

if [ "$mp4_count" -eq 0 ]; then
    echo "No .mp4 files found in $SOURCE_DIR"
    echo "Please place your video files in the source directory first."
    exit 0
fi

echo "Found $mp4_count MP4 file(s) to process"
echo ""

processed=0

# Process each MP4 file in SOURCE_DIR
for video_file in "$SOURCE_DIR"/*.mp4
do
    [ -e "$video_file" ] || continue

    # Get base name without extension (e.g., "diagonal-2" from "diagonal-2.mp4")
    base_name=$(basename "$video_file" .mp4)

    # Create main directory for this video
    video_dir="$SOURCE_DIR/$base_name"

    # Check if directory already exists
    if [ -d "$video_dir" ]; then
        echo "⚠ Directory already exists: $video_dir"
        read -p "  Overwrite? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "  Skipped: $base_name"
            echo ""
            continue
        fi
        echo "  Cleaning existing directory..."
        rm -rf "$video_dir"
    fi

    echo "------------------------------------------------------------------------"
    echo "Processing: $base_name"
    echo "------------------------------------------------------------------------"

    # Create directory structure
    mkdir -p "$video_dir/frames"
    mkdir -p "$video_dir/masks"

    echo "✓ Created directories:"
    echo "  - $video_dir/frames/"
    echo "  - $video_dir/masks/"

    # Move original video to its directory
    mv "$video_file" "$video_dir/$base_name.mp4"
    echo "✓ Moved video to: $video_dir/$base_name.mp4"

    # Extract frames with 5-digit naming starting from 00000
    # -q:v 2 = high quality JPEG
    # -start_number 0 = start from 00000.jpg
    # %05d = 5-digit zero-padded format
    echo "  Extracting frames..."
    ffmpeg -i "$video_dir/$base_name.mp4" \
           -q:v 2 \
           -start_number 0 \
           "$video_dir/frames/%05d.jpg" \
           -loglevel error

    # Count extracted frames
    frame_count=$(ls "$video_dir/frames"/*.jpg 2>/dev/null | wc -l)

    if [ "$frame_count" -gt 0 ]; then
        echo "✓ Extracted $frame_count frames"
        echo "  Format: 00000.jpg to $(printf "%05d" $((frame_count - 1))).jpg"
    else
        echo "✗ Failed to extract frames from $base_name"
    fi

    processed=$((processed + 1))
    echo ""
done

echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"
echo "✓ Processed $processed/$mp4_count video(s)"
echo ""
echo "Directory structure created:"
echo "  $SOURCE_DIR/"
echo "    └── <video_name>/"
echo "        ├── frames/          (00000.jpg, 00001.jpg, ...)"
echo "        ├── masks/           (empty, for SAM2 output)"
echo "        └── <video_name>.mp4"
echo ""
echo "Ready for SAM2 processing!"
echo "========================================================================"
