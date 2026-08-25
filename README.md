# Stabilising Video Neural Style Transfer
**Dense Optical Flow Approach to Temporal Coherence**

This repository accompanies the MSc Artificial Intelligence dissertation at the University of Bath.

## Overview
Feed-forward neural style transfer networks produce visually impressive results for static images but suffer from severe temporal inconsistency (flickering) when applied independently to video frames. This project stabilises video neural style transfer by applying temporal supervision during offline training.

By estimating dense optical-flow correspondences between adjacent frames using a pretrained RAFT-Large network, we penalise temporal divergence in the generated output. Importantly, **RAFT and VGG-19 are training/evaluation components and are omitted from normal feed-forward inference**. The final trained Transform Network performs inference on single frames independently without requiring optical flow.

## Architecture
The system consists of three main components:
1. **Transform Network**: A feed-forward generator that stylises frames.
2. **VGG-19**: A frozen perceptual loss network used to enforce content and style fidelity during training.
3. **RAFT-Large**: A frozen pretrained network that estimates dense optical-flow correspondences for temporal training supervision.

## Temporal Training
During temporal training, the network processes consecutive MPI Sintel frame pairs. The temporal loss aligns the current stylised frame with the previous stylised frame, warped according to the forward optical flow. A forward-backward consistency mask (with epsilon = 1.0) ensures temporal loss is only applied to valid, non-occluded regions.

*Note for reproduction:* The original successful temporal training run multiplies the `[0,1]` image tensors by 255 before passing them to the RAFT network. This preprocessing convention is retained in the training and original evaluation scripts to reproduce the configuration used for the submitted experiments.

## RAFT Evaluation Protocols
There are two distinct evaluation protocols provided for the frozen line-art checkpoint pair:

1. **Original training-matched evaluation** (`evaluate.py`)
   Uses the original `[0,255]` RAFT convention exactly as seen during training.
2. **Post-hoc standard-preprocessing evaluation** (`evaluate_posthoc_raft.py`)
   Uses the standard preprocessing associated with the default pretrained torchvision RAFT-Large weights.

The two protocols change the optical-flow estimation preprocessing only. The separate Van Gogh "Starry Night" checkpoint pair was evaluated only under its original evaluation configuration.

## Quantitative Results

*Note: Results are style-specific. Different artistic styles use separately trained Transform Networks, and absolute metric values should not be compared directly across styles. Results are also sequence-dependent, as motion, occlusion, illumination, scene content, and optical-flow reliability affect measured values.*

### Line Art — Original Training-Matched
* Baseline Ewarp: 0.109265
* Temporal Ewarp: 0.079878 (**26.90% lower**)
* LPIPS: 0.680662 -> 0.683785

### Line Art — Post-Hoc Standard Preprocessing
* Baseline Ewarp: 0.229539
* Temporal Ewarp: 0.221234 (**3.62% lower**)
* LPIPS: 0.680662 -> 0.683785

### Van Gogh “Starry Night” — Separate Original Evaluation
* Baseline Ewarp: 0.156674
* Temporal Ewarp: 0.067721 (**Baseline-to-temporal ratio: 2.31x**)
* LPIPS: 0.652513 -> 0.658727

## Installation
Requires Python 3.10.x and a CUDA-capable GPU compatible with the supplied CUDA 12.1 PyTorch build.

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset
The project evaluates on the MPI Sintel dataset. The dataset is not included in this repository.
Place the extracted Sintel data such that the training frames are located at:
`<video_dir>/` (e.g. `sintel/training/final/`) containing subdirectories of sequences (`alley_1`, `ambush_2`, etc.).

## Training

### Baseline Training (Independent Frames)
```bash
python train_baseline.py \
    --dataset_dir /path/to/sintel/training/final \
    --style_dir /path/to/style_images \
    --epochs 600 \
    --batch_size 8
```

### Temporal Training (Adjacent Frame Pairs)
Applies RAFT temporal supervision with a default temporal weight of 1e5.
```bash
python train.py \
    --dataset_dir /path/to/sintel/training/final \
    --style_dir /path/to/style_images \
    --temporal_weight 1e5 \
    --epochs 600 \
    --batch_size 8
```

## Evaluation
Run the original training-matched evaluation:
```bash
python evaluate.py \
    --checkpoint_path /path/to/transform_net_iter_78000.pth \
    --video_dir /path/to/sintel/training/final
```

Run the post-hoc standard-preprocessing evaluation:
```bash
python evaluate_posthoc_raft.py \
    --baseline_checkpoint /path/to/baseline.pth \
    --temporal_checkpoint /path/to/temporal.pth \
    --video_dir /path/to/sintel/training/final
```

## Inference
Feed-forward inference does not require RAFT.

**Image Inference:**
```bash
python inference.py \
    --content_path /path/to/image.jpg \
    --checkpoint_path /path/to/model.pth \
    --output_path /path/to/output.png \
    --resolution 1080
```

**Video Inference:**
```bash
python inference_video.py \
    --input /path/to/video.mp4 \
    --checkpoint /path/to/model.pth \
    --output /path/to/output_video.mp4 \
    --max_size 1024
```
*The optional Real-ESRGAN stage restores output resolution for qualitative high-resolution demonstrations and may modify fine stylistic details.*

## Repository Structure
```text
.
├── data/
│   ├── dataset.py
│   └── wikiart_urls.txt
├── models/
│   ├── temporal.py
│   ├── transform_net.py
│   └── vgg_loss.py
├── evaluate.py
├── evaluate_posthoc_raft.py
├── inference.py
├── inference_video.py
├── train.py
├── train_baseline.py
├── requirements.txt
└── README.md
```

## Reproducibility Notes
* The final temporal model is trained in FP32 with AMP disabled.
* Baseline and temporal pipelines have slightly different sample/update counts due to adjacent-pair construction.
* Baseline training used 79,800 optimisation steps.
* Temporal training used 78,000 optimisation steps.
* The post-hoc evaluation modifies optical flow preprocessing only and does not constitute retraining.

## Limitations
* Evaluation uses MPI Sintel training sequences and is not a broad out-of-distribution validation.
* Temporal improvement varies by sequence and was not uniform across the evaluated videos.
* Optical-flow reliability affects Warping Error.
* Models are style-specific; there is no universal cross-style metric interpretation.
