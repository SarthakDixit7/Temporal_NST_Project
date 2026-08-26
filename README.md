Below is a cleaned-up README I’d use for submission. It keeps the technically important caveats, matches the dissertation, and avoids exaggerated claims.

````markdown
# Stabilising Video Neural Style Transfer

## A Dense Optical Flow Approach to Temporal Coherence

This repository accompanies the MSc Artificial Intelligence dissertation:

**Stabilising Video Neural Style Transfer: A Dense Optical Flow Approach to Temporal Coherence**

University of Bath, 2025–2026.

---

## Overview

Feed-forward neural style transfer can produce effective stylisation for individual images, but applying the same network independently to consecutive video frames can produce temporal inconsistency, including flickering and texture boiling.

This project investigates whether temporal consistency can be improved by incorporating dense optical-flow supervision during offline training.

A pretrained **RAFT-Large** model estimates motion between consecutive video frames. The previous stylised output is warped according to the estimated motion field and compared with the current stylised output through a temporal loss.

Importantly, **RAFT and VGG-19 are not required during normal inference**. Once training is complete, each input frame is processed independently by the trained Transform Network using a single feed-forward pass.

---

## Architecture

The system contains three principal components:

1. **Transform Network**  
   A feed-forward convolutional neural network that maps an input frame directly to a stylised output.

2. **VGG-19 Perceptual Network**  
   A frozen feature extractor used during training to calculate content and style losses.

3. **RAFT-Large Optical Flow Network**  
   A frozen pretrained optical-flow model used during temporal training to estimate dense motion correspondences between consecutive frames.

During normal inference, only the **Transform Network** is required.

---

## Temporal Training

Temporal training operates on consecutive MPI Sintel frame pairs:

```text
x(t-1), x(t)
````

RAFT estimates forward and backward optical-flow fields between the two unstylised frames.

The previous stylised output is warped using the estimated forward flow and compared with the current stylised output:

```text
y(t)  <->  Warp(y(t-1), F(t-1 -> t))
```

A forward-backward consistency check with a squared-distance threshold of:

```text
epsilon = 1.0
```

is used to reduce temporal supervision in regions containing unreliable or inconsistent optical-flow correspondences.

The temporal loss used during training is combined with the spatial content, style and optional Total Variation losses.

---

## Important RAFT Preprocessing Note

The successful temporal training configuration used for the submitted experiments supplied image tensors to RAFT after rescaling them from:

```text
[0, 1] -> [0, 255]
```

During final validation, this was identified as differing from the preprocessing associated with the default pretrained `torchvision` RAFT-Large weights, whose supplied inference transforms map inputs to approximately:

```text
[-1, 1]
```

The original preprocessing convention is retained in the original training and evaluation pipeline for reproducibility.

A separate post-hoc evaluation was therefore performed using the preprocessing associated with the pretrained RAFT-Large weights.

No retraining was performed for the post-hoc evaluation.

---

## RAFT Evaluation Protocols

Two evaluation protocols were applied to the same frozen **line-art baseline and temporal checkpoints**.

### 1. Original Training-Matched Evaluation

Implemented in:

```text
evaluate.py
```

This reproduces the `[0,255]` RAFT input convention used during temporal training.

### 2. Post-Hoc Standard-Preprocessing Evaluation

Implemented in:

```text
evaluate_corrected_raft.py
```

This uses the preprocessing associated with the default pretrained `torchvision` RAFT-Large weights.

Only the optical-flow preprocessing changes between the two evaluation protocols. The Transform Network checkpoints, stylised outputs, LPIPS procedure, warping operation and consistency-mask logic remain unchanged.

A separate Van Gogh **"Starry Night"** checkpoint pair was evaluated only under its original evaluation configuration.

---

## Dataset

The final baseline and temporal experiments use the **MPI Sintel training split**.

The dataset is not included in this repository.

The expected directory structure is similar to:

```text
data/
└── sintel/
    └── training/
        └── clean/
            ├── alley_1/
            ├── ambush_2/
            ├── bamboo_1/
            └── ...
```

The final quantitative evaluation contains:

```text
23 video sequences
1,041 consecutive frame pairs
```

The baseline model processes individual frames independently, while the temporal model uses adjacent frame pairs.

MPI Sintel ground-truth optical flow is **not** used for temporal training or the reported Warping Error calculations. Motion is estimated using RAFT.

---

## Quantitative Results

Results are **style-specific**.

Each artistic style is represented by separately trained Transform Networks, so absolute Warping Error and LPIPS values should not be used to rank different artistic styles directly.

Results can also vary between video sequences because of differences in scene content, motion, occlusion, illumination, non-rigid deformation and optical-flow reliability.

### Line Art — Original Training-Matched Evaluation

| Model                  | Mean Warping Error | Mean LPIPS |
| ---------------------- | -----------------: | ---------: |
| Baseline               |           0.109265 |   0.680662 |
| Temporally Constrained |       **0.079878** |   0.683785 |

Mean Warping Error was:

```text
26.90% lower
```

for the temporally constrained model.

LPIPS changed by:

```text
+0.003123
```

---

### Line Art — Post-Hoc Standard RAFT Preprocessing

| Model                  | Mean Warping Error | Mean LPIPS |
| ---------------------- | -----------------: | ---------: |
| Baseline               |           0.229539 |   0.680662 |
| Temporally Constrained |       **0.221234** |   0.683785 |

Mean Warping Error was:

```text
3.62% lower
```

for the temporally constrained model.

The smaller reduction demonstrates that the measured temporal improvement is sensitive to the preprocessing used by the optical-flow estimator.

LPIPS is unchanged between the two line-art evaluation protocols because LPIPS does not depend on RAFT-derived optical flow.

At sequence level, the post-hoc evaluation produced macro-average Warping Error values of:

```text
Baseline: 0.240520
Temporal: 0.233468
```

Improvement was not uniform across every sequence.

---

### Van Gogh "Starry Night" — Separate Evaluation

| Model                  | Mean Warping Error | Mean LPIPS |
| ---------------------- | -----------------: | ---------: |
| Baseline               |           0.156674 |   0.652513 |
| Temporally Constrained |       **0.067721** |   0.658727 |

Mean Warping Error was:

```text
56.78% lower
```

for the temporally constrained model.

LPIPS changed by:

```text
+0.006214
```

A corresponding post-hoc standard-preprocessing evaluation was **not performed** for this checkpoint pair.

---

## Interpretation of LPIPS

LPIPS is calculated between each stylised output and its corresponding **unstylised content frame**.

It therefore measures perceptual deviation from the source content rather than similarity to the reference artwork.

Consequently, LPIPS in this project should **not** be interpreted as a direct measure of artistic style fidelity.

---

## Dynamic Style Routing

Different style configurations use different spatial regularisation settings.

### High-Frequency / Pointillist Configuration

Style layers:

```text
relu1_2
relu2_2
relu3_3
```

Total Variation weight:

```text
0.0
```

This configuration is intended to preserve fine high-frequency dot structures.

### Impressionist / Macro-Structural Configuration

Style layers:

```text
relu1_2
relu2_2
relu3_3
relu4_1
relu5_1
```

Total Variation weight:

```text
1e-3
```

The deeper style features and small Total Variation penalty support broader and smoother brushstroke structures.

These configurations were developed from qualitative observations and should not be interpreted as universally optimal settings for all artistic styles.

---

## Numerical Stability

Mixed-precision execution produced non-finite losses in some high-frequency training configurations.

The precise numerical cause was not fully isolated.

The final temporal training configuration therefore disables Automatic Mixed Precision and performs optimisation entirely in:

```text
FP32
```

Under this configuration, the observed numerical instability did not recur.

---

## Installation

The submitted environment uses Python 3.10.x and CUDA-compatible PyTorch.

Create a virtual environment:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

### Baseline Training

The baseline model processes Sintel frames independently without temporal supervision.

```bash
python train_baseline.py \
    --dataset_dir /path/to/sintel/training/clean \
    --style_dir /path/to/style_images \
    --epochs 600 \
    --batch_size 8
```

### Temporal Training

The temporal model processes adjacent Sintel frame pairs and incorporates RAFT-derived temporal supervision.

```bash
python train.py \
    --dataset_dir /path/to/sintel/training/clean \
    --style_dir /path/to/style_images \
    --temporal_weight 1e5 \
    --epochs 600 \
    --batch_size 8 \
    --require_pretrained_raft
```

The `--require_pretrained_raft` option prevents the reported experiment configuration from silently falling back to a non-pretrained RAFT model if pretrained weights cannot be loaded.

---

## Evaluation

### Original Training-Matched Evaluation

```bash
python evaluate.py \
    --checkpoint_path /path/to/temporal_checkpoint.pth \
    --video_dir /path/to/sintel/training/clean \
    --require_pretrained_raft
```

This reproduces the original RAFT preprocessing convention used during training.

### Post-Hoc Standard-Preprocessing Evaluation

```bash
python evaluate_corrected_raft.py \
    --baseline_checkpoint /path/to/baseline_checkpoint.pth \
    --temporal_checkpoint /path/to/temporal_checkpoint.pth \
    --video_dir /path/to/sintel/training/clean
```

This evaluates the same frozen line-art checkpoints using the preprocessing associated with the pretrained `torchvision` RAFT-Large weights.

---

## Inference

Normal inference requires only the trained Transform Network.

RAFT and VGG-19 are not required.

### Image Inference

```bash
python inference.py \
    --content_path /path/to/image.jpg \
    --checkpoint_path /path/to/model.pth \
    --output_path /path/to/output.png \
    --resolution 1080
```

### Video Inference

```bash
python inference_video.py \
    --input /path/to/video.mp4 \
    --checkpoint /path/to/model.pth \
    --output /path/to/output_video.mp4 \
    --max_size 1024
```

Each video frame is processed independently by the Transform Network without runtime optical-flow computation or recurrent state.

---

## High-Resolution Demonstration Pipeline

For qualitative high-resolution demonstrations, an optional:

```text
downscale -> stylise -> upscale
```

pipeline is provided.

The stylised frame can be restored to the original output resolution using **Real-ESRGAN**.

Because Real-ESRGAN is a learned generative upsampler and may modify fine stylistic details, outputs produced using this stage were excluded from the quantitative evaluation.

---

## Principal Training Configuration

The principal settings used in the final experiments include:

| Parameter                   |           Value |
| --------------------------- | --------------: |
| Optimiser                   |            Adam |
| Learning Rate               |          `1e-3` |
| Batch Size                  |             `8` |
| Content Weight              |          `5000` |
| Style Weight                |          `1e10` |
| Temporal Weight             |           `1e5` |
| TV Weight                   | `0.0` or `1e-3` |
| Consistency Threshold       |           `1.0` |
| Temporal Training Precision |            FP32 |

---

## Training-Step Difference

The baseline and temporal data pipelines contain slightly different numbers of samples.

For each Sintel sequence containing `N` frames:

```text
Baseline samples = N
Temporal samples = N - 1
```

Across 23 sequences and 600 epochs, the reported configurations completed:

```text
Baseline: 79,800 optimisation steps
Temporal: 78,000 optimisation steps
```

This difference is acknowledged as an experimental limitation when comparing the final trained checkpoints.

---

## Repository Structure

```text
.
├── data/
│   ├── dataset.py
│   └── wikiart_urls.txt
│
├── models/
│   ├── temporal.py
│   ├── transform_net.py
│   └── vgg_loss.py
│
├── evaluate.py
├── evaluate_corrected_raft.py
├── inference.py
├── inference_video.py
├── train.py
├── train_baseline.py
├── requirements.txt
└── README.md
```

---

## Reproducibility Notes

* Final temporal training was performed entirely in FP32 with AMP disabled.
* All reported RAFT-based experiments used the default pretrained RAFT-Large weights.
* The original successful temporal training configuration used the `[0,255]` RAFT input convention.
* The post-hoc evaluation changes RAFT preprocessing only and does not constitute retraining.
* Baseline and temporal models have slightly different optimisation-step counts because temporal training requires adjacent frame pairs.
* Quantitative evaluation uses all 1,041 consecutive frame pairs from the 23 MPI Sintel training sequences.
* Line-art and Van Gogh results represent separate style-specific checkpoint pairs.
* The Van Gogh checkpoint pair was not evaluated using the post-hoc standard-preprocessing protocol.
* High-resolution Real-ESRGAN demonstration outputs are excluded from quantitative evaluation.

---

## Limitations

The principal limitations of the submitted experiments are:

* Evaluation is performed on the MPI Sintel **training split**, so the reported results do not establish broad out-of-distribution generalisation.
* Temporal improvement is sequence-dependent and was not uniform across all evaluated videos.
* Warping Error depends on RAFT-estimated motion and is therefore sensitive to optical-flow estimation quality and preprocessing.
* The original successful temporal training configuration used a RAFT input convention that differs from the preprocessing associated with the pretrained `torchvision` RAFT-Large weights.
* Artistic styles are represented by separately trained Transform Networks, so absolute metric values should not be compared directly across styles.
* The forward-backward consistency mask reduces unreliable temporal supervision but does not guarantee perfect identification of all occluded or invalid correspondences.
* High-frequency artistic detail can trade off against stronger temporal or spatial regularisation.
* A limited set of artistic style configurations was investigated.

---

## Dissertation

For the complete methodology, mathematical formulation, experimental discussion, qualitative analysis and limitations, refer to the accompanying MSc dissertation:

**Stabilising Video Neural Style Transfer: A Dense Optical Flow Approach to Temporal Coherence**
