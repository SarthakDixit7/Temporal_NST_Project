# Temporal_NST_Project

Quick setup and data instructions.

## Installation & Setup

It is highly recommended to run this project inside an isolated virtual environment to prevent dependency conflicts (especially with PyTorch and Real-ESRGAN).

```bash
# 1. Create and activate a pristine virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip and install the precise requirements
pip install --upgrade pip
pip install -r requirements.txt
```

*Note: The `inference_video.py` script includes a built-in "monkey patch" for `torchvision.transforms.functional_tensor` to elegantly fix a known `basicsr` import issue on newer PyTorch versions. You do not need to manually edit `basicsr` files.*

## Model Training

To train the Temporal Neural Style Transfer model, use `train.py`. The script requires a dataset of sequential video frames (for temporal consistency) and a style directory containing your target style images.

```bash
# Basic training command
python train.py \
    --dataset_dir data/sintel \
    --style_dir data/wikiart \
    --checkpoint_dir checkpoints \
    --epochs 2 \
    --batch_size 4
```

### Key Training Arguments:
- `--style_mode`: Choose a specific artistic dynamic configuration (`pointillism` or `impressionism`).
- `--temporal_weight`: Adjust the strength of the temporal consistency loss (default usually requires tuning based on motion).
- `--resume`: Add this flag to automatically resume training from the latest checkpoint in your `--checkpoint_dir`.
- `--checkpoint_interval`: How many steps between saving model weights.

*Note on Style Images:* Place your target style image in your `--style_dir`. The loader automatically caps the style image's longest side at 512 pixels before resizing. **Do not** feed raw 4K style images directly; the VGG receptive field will fail to capture macro brushstrokes properly.

---

## Model Inference (Video)

To stylize and upscale a video using your trained checkpoints, use `inference_video.py`. This script downscales the video to apply thick brushstrokes, processes it through your trained `TransformNetwork`, and then uses **Real-ESRGAN** to upscale it back to its original high-resolution (e.g., 4K or 1080p).

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
- `--checkpoint`: Provide a specific `.pth` file path if you don't want to use the latest.
- `--max_size`: The longest edge size for the neural network processing (default: `1024`). Lower values produce thicker brush strokes but lose fine detail before upscaling.
