# SAM2 Video Segmentation Workflow

Complete workflow for processing videos and generating object masks using SAM2 (Segment Anything Model 2).

This fork provides easy-to-use scripts for multi-object video segmentation with automatic background mask generation.

## 📦 Installation

### Prerequisites
- Python >= 3.10
- CUDA-compatible GPU (recommended)
- conda or miniconda

### Step 1: Install Conda (if not already installed)

```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
```

### Step 2: Create SAM2 Environment

```bash
# Create conda environment with Python 3.10
conda create -n SAM2 python=3.10 -y
conda activate SAM2

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio

# Install SAM2 and dependencies
pip install -e ".[notebooks]"
```

### Step 3: Download Model Checkpoints

```bash
# Download SAM2.1 checkpoints
cd checkpoints
bash ./download_ckpts.sh
cd ..
```

### Verify Installation

```bash
conda activate SAM2
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "from sam2.build_sam import build_sam2_video_predictor; print('SAM2 installed successfully')"
```

## 📁 Directory Structure

After processing, your dataset will have this structure:

```
assets/xxx-dataset/
├── video1.mp4                    # Place new videos here
├── video2.mp4
├── video1/
│   ├── frames/
│   │   ├── 00000.jpg            # Extracted frames (0-indexed, 5-digit)
│   │   ├── 00001.jpg
│   │   └── ...
│   ├── masks/
│   │   ├── obj1/
│   │   │   ├── 00000.png        # Masks for object 1
│   │   │   └── ...
│   │   ├── obj2/
│   │   │   ├── 00000.png        # Masks for object 2
│   │   │   └── ...
│   │   ├── bg/
│   │   │   ├── 00000.png        # Background masks
│   │   │   └── ...
│   │   └── three_masks_visualization.png
│   └── video1.mp4                # Original video
└── video2/
    └── ... (same structure)
```

## 🔄 Complete Workflow

### Step 1: Prepare Videos

Place your `.mp4` files in the source directory: `assets/xxx-dataset/`

### Step 2: Extract Frames

Configure the source directory in `setup_dataset.sh`:

```bash
# Edit setup_dataset.sh
SOURCE_DIR="assets/xxx-dataset"  # Change this to your dataset directory
```

Then run the script:

```bash
bash setup_dataset.sh
```

This will:
- Find all `.mp4` files in the source directory
- Create a directory for each video (e.g., `video-name/`)
- Extract frames as JPG using ffmpeg (00000.jpg, 00001.jpg, ...)
- Move the original video into its directory
- Create an empty `masks/` directory for later use

**Output format:**
- Frame naming: `00000.jpg, 00001.jpg, ..., 00099.jpg`
- 5-digit, 0-indexed (matches SAM2 default)
- Quality: `-q:v 2` (high quality JPEG)

### Step 3: Configure SAM2 Script

Open `create-masks-2obj-bg.py` (for 2 objects) or `create-masks-3obj-bg.py` (for 3 objects) and modify the CONFIG SECTION:

```python
# Video Directory
WORKING_DIR = "your-sam-dir"
VIDEO_NAME = "your-video-name"  # Name of the video directory

# Frame Selection (0-indexed)
OBJ1_FRAME_IDX = 0      # Frame 0 = 00000.jpg
OBJ2_FRAME_IDX = 10     # Frame 10 = 00010.jpg
OBJ3_FRAME_IDX = 5      # Only in 3obj version
VISUALIZATION_FRAME_IDX = 10

# Object 1: Click Points
OBJ1_POINTS = [
    (410, 200),  # (x, y) coordinates
    (380, 270),
    (420, 165)
]
OBJ1_LABELS = [1, 1, 1]  # 1 = inside object, 0 = outside object

# Object 2: Click Points
OBJ2_POINTS = [
    (450, 120),
    (450, 240)
]
OBJ2_LABELS = [1, 1]

# Object 3: Click Points (only in create-masks-3obj-bg.py)
OBJ3_POINTS = [
    (300, 300),
    (320, 320)
]
OBJ3_LABELS = [1, 1]
```

**How to choose click coordinates:**
1. Open a representative frame from `video-name/frames/` directory
2. Use an image viewer (e.g., GIMP, Photoshop, or online tools) to find pixel coordinates
3. Click on multiple points within each object for better segmentation
4. Use label `1` for points inside the object, `0` for points to exclude
5. Record the (x, y) coordinates

**Tips:**
- Choose frames where all objects are clearly visible
- Add 3-5 click points per object for better accuracy
- Objects can be marked on different frames if they appear at different times
- The script will propagate masks to **all frames** (from 00000.jpg to the last frame)

### Step 4: Generate Masks

Activate the SAM2 environment and run the appropriate script:

```bash
# Activate conda environment
conda activate SAM2

# For 2 objects + background
python create-masks-2obj-bg.py

# OR for 3 objects + background
python create-masks-3obj-bg.py
```

**What the script does:**
1. Loads all frames from `VIDEO_NAME/frames/`
2. Initializes SAM2 model on GPU (or CPU if GPU unavailable)
3. Processes your click points on the specified reference frames
4. Propagates object masks through **all frames** (0 to last frame)
5. Auto-generates background mask (everything NOT covered by objects)
6. Saves masks to separate directories: `VIDEO_NAME/masks/obj1/`, `obj2/`, `obj3/` (if 3obj), `bg/`
7. Creates a visualization image showing all masks

**Output:**
- Mask naming: `00000.png, 00001.png, ...` (matches input frames)
- Format: PNG grayscale (0 = black, 255 = white)
- One mask file per object per frame
- Visualization: `three_masks_visualization.png` or `four_masks_visualization.png`

## 📊 Frame and Mask Indexing

**Important:** This workflow uses **0-based indexing** (SAM2 default):

| Frame File | Frame Index | Mask File |
|-----------|-------------|-----------|
| 00000.jpg | 0           | 00000.png |
| 00001.jpg | 1           | 00001.png |
| 00010.jpg | 10          | 00010.png |
| 00099.jpg | 99          | 00099.png |

When specifying `OBJ1_FRAME_IDX = 0`, it uses `00000.jpg`.

## 📝 File Naming Convention

| Type | Format | Example | Notes |
|------|--------|---------|-------|
| Input frames | `%05d.jpg` | `00000.jpg` | 5-digit, 0-indexed |
| Output masks | `%05d.png` | `00000.png` | Matches input |
| Frame index | 0-based | 0, 1, 2, ... | In code |
| Object names | Fixed | obj1, obj2, (obj3), bg | 2-3 objects + background |

## 🛠️ Available Scripts

| Script | Description |
|--------|-------------|
| `setup_dataset.sh` | Extracts frames from MP4 videos |
| `create-masks-2obj-bg.py` | Generates masks for 2 objects + background |
| `create-masks-3obj-bg.py` | Generates masks for 3 objects + background |

## ⚠️ Troubleshooting

### CUDA Warnings
If you see CUDA kernel warnings, they can be safely ignored. The scripts already suppress these warnings with:
```python
warnings.filterwarnings('ignore', category=UserWarning)
```

### Missing Frames
If masks are not generated for all frames, ensure you're using the latest version of the scripts which include `start_frame_idx=0` in the propagation step.

### Environment Issues
If you encounter import errors:
```bash
conda activate SAM2
pip install -e ".[notebooks]" --force-reinstall
```

### GPU Out of Memory
If you run out of GPU memory:
- Reduce video resolution before extraction
- Process shorter videos
- Use CPU mode (will be slower)
