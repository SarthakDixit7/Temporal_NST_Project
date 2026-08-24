# Stabilising Video Neural Style Transfer: A Dense Optical Flow Approach to Temporal Coherence

This repository contains the official PyTorch implementation for the project **"Stabilising Video Neural Style Transfer: A Dense Optical Flow Approach to Temporal Coherence"**.

The project investigates the use of modern dense optical-flow estimation (RAFT) within an offline training framework for feed-forward video neural style transfer. The primary goal is to mitigate the severe temporal aliasing (texture boiling and flickering) inherent in frame-by-frame processing.

## Architecture & Methodology

The system is constructed upon three interacting architectural components, designed to restrict high computational costs to the offline training phase:

1. **Feed-Forward Transform Network:** A residual convolutional architecture designed for efficient feed-forward inference. It utilises reflection padding, instance normalisation, and integer striding to prevent spatial information loss.
2. **Dense Motion Estimator (RAFT):** A pre-trained Recurrent All-Pairs Field Transforms model used to compute bidirectional dense optical flow fields. This provides the exact vectors needed to track inter-frame pixel trajectories.
3. **Perceptual Loss Extractor (VGG-19):** A fixed image classification network used to evaluate deep feature representations (Content and Style loss) during training. Max-pooling layers are overridden with average-pooling for smoother gradient flow.

### Dynamic Style Routing
The project introduces Dynamic Style Routing to adapt spatial regularisation based on artistic frequency:
- **Pointillism (High-Frequency):** Restricts Gram matrix computations to shallow layers (`relu1_2` to `relu3_3`) and explicitly disables Total Variation (TV) loss (`tv_weight = 0.0`) to preserve fine dots.
- **Impressionism (Macro-Structural):** Expands style extraction to deeper layers (up to `relu5_1`) and applies TV loss (`tv_weight = 1e-3`) to encourage continuous brushstrokes.

---

## Installation & Setup

It is highly recommended to run this project inside an isolated virtual environment to prevent dependency conflicts (especially with PyTorch and Real-ESRGAN).

```bash
# 1. Create and activate a new virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip and install the requirements
pip install --upgrade pip
pip install -r requirements.txt
```

*Note: The `inference_video.py` script includes a built-in "monkey patch" for `torchvision.transforms.functional_tensor` to elegantly fix a known `basicsr` import issue on newer PyTorch versions. You do not need to manually edit `basicsr` files.*

---

## Model Training

To train the Temporal Neural Style Transfer model, use `train.py`. The script requires a dataset of sequential video frames (for temporal consistency) and a style directory containing your target style images. We recommend using the **MPI Sintel** dataset for video frames and **WikiArt** for styles.

```bash
# Basic training command
python train.py \
    --dataset_dir data/sintel \
    --style_dir data/wikiart \
    --checkpoint_dir checkpoints \
    --epochs 600 \
    --batch_size 8
```

### Key Training Arguments:
- `--style_mode`: Choose a specific artistic dynamic configuration (`pointillism` or `impressionism`).
- `--temporal_weight`: Adjust the strength of the temporal consistency loss (default: `1e5`).
- `--resume`: Add this flag to automatically resume training from the latest checkpoint in your `--checkpoint_dir`.
- `--checkpoint_interval`: How many steps between saving model weights (default: `500`).

*Note on Hardware:* This project disables Automatic Mixed Precision (AMP) in `train.py` because zero-TV constraints on high-frequency styles can trigger FP16 overflow resulting in tensor collapse. Optimization is handled entirely in FP32.

---

## Model Inference (Video)

To stylize and upscale a video using your trained checkpoints, use `inference_video.py`. This script employs a **downscale-stylize-upscale** pipeline:
1. Downscales the 4K/1080p video to enlarge abstract brushstrokes.
2. Processes it through the feed-forward Transform Network.
3. Uses **Real-ESRGAN** (`RRDBNet`) to upscale it back to the original high-resolution dimensions without blurring.

```bash
# Run inference on a video
python inference_video.py \
    --input data/dogs.mp4 \
    --output output/dogs_stylized.mp4 \
    --checkpoint_dir checkpoints \
    --max_size 1024
```

### Key Inference Arguments:
- `--input`: Path to your source `.mp4` video.
- `--output`: Path to save the stylized `.mp4` video.
- `--checkpoint_dir`: Directory containing your trained `.pth` models (the script automatically picks the latest).
- `--checkpoint`: Provide a specific `.pth` file path to explicitly load it.
- `--max_size`: The longest edge size for neural network processing (default: `1024`). Lower values produce thicker brush strokes.

---

## Evaluation

The project includes an `evaluate.py` script for calculating objective metrics over testing datasets:
- **Warping Error ($E_{warp}$):** Measures geometric frame-to-frame stability using a forward-backward occlusion mask (`epsilon = 1.0`).
- **LPIPS:** Measures perceptual spatial fidelity relative to the original video frames.

```bash
python evaluate.py --dataset_dir data/sintel_test --checkpoint_dir checkpoints
```
