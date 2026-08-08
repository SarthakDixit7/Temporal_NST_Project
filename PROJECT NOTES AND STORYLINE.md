Here is the fully expanded, comprehensive draft of your dissertation notes. I have fleshed out the technical explanations, deepened the architectural rationales, and integrated a dedicated subsection justifying the necessity of the dynamic style shift to prove the robustness of your temporal architecture.

This is structured to serve as the foundational text for your Methodology, Implementation, and Ablation chapters.

---

# Dissertation Technical Notes: Real-Time Video Neural Style Transfer

## 1. Project Motivation and the Temporal Coherence Problem

The fundamental objective of this research is to achieve real-time video neural style transfer without sacrificing frame-to-frame temporal coherence.

The baseline algorithm for Neural Style Transfer (NST), introduced by Gatys et al., relies on an offline, iterative per-frame pixel optimisation process. While this produces high-quality static images, it is computationally prohibitive for video applications, rendering real-time execution impossible.

Transitioning to a feed-forward Transform Network resolves the processing speed bottleneck by applying style in a single forward pass. However, treating a video sequence as a series of independent, isolated images exposes a critical failure point: the temporal coherence problem. Because the feed-forward network processes each frame independently, minor variations in the input video's pixel data—or slight changes in lighting and occlusion—result in vastly different stylised outputs. This manifests visually as severe, high-frequency flickering, destroying the illusion of a cohesive moving image.

The core intervention of this project is the synthesis of a high-speed feed-forward architecture with a custom, mathematically rigorous temporal consistency loss driven by dense optical flow estimation.

---

## 2. Architecture and System Pipeline

The pipeline is constructed upon three interacting architectural pillars, each explicitly designed to resolve a specific limitation of standard feed-forward style transfer.

### A. The Feed-Forward Transform Network

* **Architectural Role:** Replaces the expensive online optimization process with a single-pass inference model, unlocking the real-time execution speeds required for standard video frame rates (e.g., 24 to 30 fps).
* **Mechanism:** The network ingests a raw video content frame $x_t$ and maps it through a series of convolutional and residual layers to output the stylised frame $y_t = T(x_t)$ instantly.

### B. Dense Optical Flow Estimation via RAFT

* **Architectural Role:** Anchors the network's understanding of time, motion, and spatial displacement, effectively eliminating the high-frequency flickering inherent in single-frame processing.
* **Mechanism:** Traditional temporal smoothing relies on lightweight, sparse optical flow approximations that fail under fast motion. This pipeline integrates **RAFT (Recurrent All-Pairs Field Transforms)** to compute dense, highly accurate forward and backward optical flow fields between consecutive content frames ($x_{t-1}$ and $x_t$).
* **Integration:** By calculating the exact pixel trajectories, the network can warp the previously stylised output $y_{t-1}$ onto the spatial grid of the current frame $y_t$. This allows the loss function to enforce temporal consistency by directly penalising deviations along these calculated motion vectors.

### C. The VGG-19 Feature and Loss Extractor

* **Architectural Role:** Functions as a fixed, pre-trained feature extractor to evaluate perceptual content loss and artistic style loss during the training phase.
* **Mechanism:** Rather than evaluating pixel-to-pixel differences, the network compares the deep feature representations of the images. A critical architectural refinement involved shifting the feature extraction deeper into the network hierarchy (specifically targeting the `relu5_1` layer) to force the loss function to prioritise macro-level semantic structures and complex artistic motifs over low-level visual noise.

---

## 3. Methodological Justification: The Necessity of Dynamic Style Routing

While the primary objective of this dissertation is achieving temporal consistency, evaluating this temporal loss exclusively on standard, macro-structural art styles (e.g., standard Impressionism) introduces a critical methodological vulnerability. An examiner could argue that the temporal stability was artificially achieved because the spatial smoothing applied to such styles masked any underlying temporal failures.

To rigorously stress-test the RAFT-anchored temporal architecture, this project implements a **Dynamic Feature Extraction** and routing system based on the frequency of the target artistic texture. This "style mode shift" was not an arbitrary addition, but a necessary experimental condition serving two critical purposes:

1. **Stress-Testing Temporal Robustness:** By forcing the network to handle high-frequency, zero-TV styles (like Pointillism), the hardest possible spatial constraints were applied to the temporal loss. Proving that the temporal consistency holds even when the network generates sharp, chaotic, high-contrast pixels demonstrates that the optical flow solution is robust across the entire frequency spectrum of art.
2. **Aesthetic Validity (Avoiding the "Muddy Video" Trap):** Temporal consistency is mathematically meaningless if it destroys the artistic integrity of the frame. If Pointillist dots are smeared into a temporal blur, or Post-Impressionist brushstrokes dissolve into static grain, the style transfer has failed. Dynamic routing proves the model solves both the mathematical problem of flickering and the aesthetic problem of preserving highly specific artistic constraints in motion.

---

## 4. Implementation of Dynamic Feature Extraction and Spatial Regularisation

Because artistic textures are highly scale-variant, a one-size-fits-all VGG layer selection degrades output quality. The pipeline dynamically adjusts feature extraction depth and spatial regularisation based on the structural requirements of the target style.

### High-Frequency Textures (e.g., Georges Seurat’s Pointillism)

Styles characterised by microscopic, high-frequency visual noise require strict localisation to shallow network features.

* **Layer Selection:** Gram matrix computation is restricted to the early layers (`relu1_2`, `relu2_2`, and `relu3_3`). These act as low-level filter banks, capturing exact pixel-to-pixel colour variance and sharp dot textures without attempting to impose overarching spatial geometry.
* **Spatial Regularisation:** The Total Variation (TV) penalty is explicitly disabled ($\mathcal{L}_{\text{tv}} = 0$). Because TV loss functions as an anisotropic spatial smoother, its activation aggressively penalises the sharp, intentional pixel contrast required to render distinct dots. Disabling it prevents severe texture degradation and muddying.

### Macro-Structural Textures (e.g., Vincent van Gogh’s Post-Impressionism)

Styles defined by thick, sweeping, and connected brushstrokes demand macro-level structural coherence.

* **Layer Selection:** Style loss calculation is expanded deeper into the network architecture to include `relu4_1` and `relu5_1`. These deeper layers possess a broader receptive field, allowing the network to capture and reproduce large, continuous stylistic structures rather than isolated paint fissures.
* **Spatial Regularisation:** A minor Total Variation penalty ($\mathcal{L}_{\text{tv}} = 1 \times 10^{-3}$) is applied during training. This acts as a necessary smoothing agent between the sweeping brushstrokes, actively suppressing high-frequency checkerboard noise.

---

## 5. Dataset Preprocessing: Solving the Spatial Scale Mismatch

Early iterations of the network struggled to map the global composition of static reference paintings from the WikiArt dataset onto dynamic video frames. The network attempted to recreate the entire layout of a painting within every single video frame, resulting in distorted outputs.

* **Solution - Targeted Texture Cropping:** Instead of feeding entire paintings into the style loss calculation, the preprocessing pipeline crops the reference images to isolate specific brushstroke textures. This forces the Gram matrices to learn the physical, repeated texture of the medium (e.g., the specific shape of a brushstroke) rather than attempting to replicate the overarching layout of the original artwork.

---

## 6. Unified Loss Function Formulation

The final training loss is structured to dynamically balance three competing objectives within a single feed-forward pass: preserving the underlying video content, applying the cropped artistic textures, and maintaining strict frame-to-frame temporal stability.

$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{content}} + \beta \mathcal{L}_{\text{style}} + \gamma \mathcal{L}_{\text{temporal}}$$

1. **Content Loss ($\mathcal{L}_{\text{content}}$):** Computes the Euclidean distance between high-level feature maps of the content frame $x_t$ and generated frame $y_t$ extracted from the VGG network, ensuring the core geometry of the video is preserved.
2. **Style Loss ($\mathcal{L}_{\text{style}}$):** Measures discrepancies between the Gram matrices of feature maps from the cropped style reference image and the generated frame $y_t$, driving the application of the specific brushstrokes.
3. **Temporal Consistency Loss ($\mathcal{L}_{\text{temporal}}$):** Penalises pixel-wise differences between the prior stylised frame $\hat{y}_{t-1}$ (warped using RAFT optical flow vectors) and the current stylised frame $y_t$ in non-occluded regions. By embedding this directly into the feed-forward training phase, the model inherently learns temporal motion, successfully eliminating the need for post-processing filters.

---

## 7. Technical Challenge & Ablation: FP16 Overflow and NaN Propagation

During the implementation of the Pointillism style mode ($\mathcal{L}_{\text{tv}} = 0.0$), a severe numerical instability was observed, leading to total tensor collapse. This provided a crucial hardware-level ablation study regarding the interaction between spatial loss constraints and mixed-precision calculation.

### Diagnostic Analysis

The network initially produced localized black artifacts concentrated on areas of high geometric complexity and non-rigid motion (e.g., morphing faces). When the content loss weight was subsequently reduced to allow for stronger stylisation, the entire output tensor collapsed into pure black frames. This was diagnosed as localized `NaN` (Not a Number) generation caused by gradient overflow within PyTorch’s Automatic Mixed Precision (`autocast`) framework.

1. **The Spark (FP16 Overflow):** By explicitly disabling the Total Variation penalty to preserve high-frequency pointillist dots, all spatial smoothing constraints were removed. When attempting to force extreme pixel contrast onto non-rigid geometries, the mathematical gradients spiked violently. The 16-bit floating-point (FP16) architecture has a strict maximum value ceiling of 65,504. The high-frequency gradients exceeded this ceiling, resulting in a mathematical overflow that generated localized `NaN` values.
2. **The Collapse (Gram Matrix Propagation):** The total network collapse was triggered by the loss weighting shift. When content loss dominated, `NaN` values were quarantined to specific spatial coordinates, as content loss operates strictly pixel-by-pixel. However, when priority shifted to the Style Loss, the Gram Matrix—which computes the inner product of vectorized feature maps—cross-multiplied the localized `NaN` values against the entire spatial grid. This caused instantaneous global `NaN` propagation.

### Resolution

To resolve this instability, the `autocast` context manager was disabled for the forward pass, forcing the network to calculate in standard 32-bit floating-point precision (FP32). FP32 possesses a vastly higher numerical ceiling ($\approx 3.4 \times 10^{38}$), which successfully absorbed the extreme gradient spikes generated by the zero-TV pointillist styling without overflowing, entirely eliminating the tensor collapse.


### 1.X Summary of Contributions

This research modernizes video neural style transfer by addressing the computational, structural, and numerical limitations of legacy baseline architectures. The core contributions of this dissertation are as follows:

* **Integration of State-of-the-Art Optical Flow:** The pipeline replaces legacy motion estimation algorithms (e.g., DeepFlow) with Recurrent All-Pairs Field Transforms (RAFT). This modernizes the temporal loss calculation, providing highly accurate motion vectors and occlusion masking that effectively handle non-rigid motion and complex spatial deformations.
* **End-to-End Temporal Feed-Forward Fusion:** The research successfully synthesizes the computational speed of a feed-forward Transform Network with the frame-to-frame stability of optimization-based methods. By embedding the RAFT-derived temporal consistency constraints directly into the training loop, the network natively learns to generate temporally coherent stylizations without requiring post-processing stabilization.
* **Identification of Mixed-Precision (FP16) Numerical Instabilities:** The study investigates the behavior of high-frequency stylistic gradients on modern hardware, diagnosing a critical mathematical overflow when operating in mixed-precision (`autocast`) environments. The research documents how the removal of Total Variation (TV) penalties for micro-textural styling directly exceeds the FP16 memory ceiling, and implements an architectural resolution to prevent global tensor collapse.
* **Rigorous Quantitative Evaluation Framework:** The project moves beyond subjective visual assessment by establishing a strictly quantitative ablation framework. It utilizes Learned Perceptual Image Patch Similarity (LPIPS) and Temporal Warping Error metrics to mathematically evaluate the trade-off between perceptual stylistic fidelity and pixel-level temporal stability.


That is an incredibly observant question, and the answer highlights a fascinating side-effect of temporal training.

If your temporal model uses the exact same base architecture (and zero-padding) as your baseline but *doesn't* have those black borders, it is because **your temporal loss function explicitly killed them.**

Here is exactly how enforcing temporal consistency naturally fixes zero-padding artifacts:

### 1. Zero-Padding is "Screen-Locked"

In your baseline model, those black smudges are static. No matter what is happening in the video—whether a character is running or the camera is panning—the zero-padding is always fixed to the absolute edges of the screen.

### 2. Temporal Loss Relies on Motion

Your temporal model was trained using optical flow (which tracks how pixels physically move from Frame 1 to Frame 2). To calculate the temporal loss, the network takes the stylized Frame 1 and warps it to match the motion of Frame 2. It then compares this warped Frame 1 against the newly stylized Frame 2.

### 3. The Collision

Imagine a scene where the camera pans to the right. The optical flow dictates that all the pixels in the image should shift to the left.

* When the network warps Frame 1, that static black shadow on the right edge gets pulled *inward* toward the center of the image.
* However, when the network stylizes Frame 2, the zero-padding creates a *new* static black shadow back at the right edge.

When the loss function compares the two, it sees a massive error: a giant black shadow has suddenly appeared in the middle of the warped image, completely failing to match the newly stylized frame.

### The Result: An Accidental Regularizer

Because the temporal loss severely penalizes any visual element that stays frozen on the screen while the underlying scene moves (often called "texture sticking"), it creates a massive penalty for those black borders.

To lower this loss, the network is literally forced to learn how to compensate for its own zero-padding. It alters its internal weights to brighten those edges and fill them with color so that the edges flow naturally with the video's motion, completely erasing the artifacts in the process.

In short: **Your temporal loss is working so well that it didn't just fix video flickering; it acted as a strong regularizer that taught the network to fix its own spatial flaws.**

The difference in the total number of steps (and thus your final checkpoint numbers) comes down to how the two different dataset classes load your frames.

In your baseline script, you are using SingleFrameVideoDataset. This dataset loads every single image independently.

In your temporal script, you are using VideoDataset. Because your temporal loss requires calculating optical flow between frames, this dataset has to load pairs of adjacent frames (frame_t_minus_1 and frame_t). To make pairs, the dataset loses one frame of length for every individual video clip in the dataset. A clip with 50 frames only has 49 overlapping pairs.

Your default dataset is MPI Sintel, which contains roughly 24 different video sequences. Because the temporal dataset loses one frame per sequence to create pairs, it has exactly 24 fewer total items than the baseline dataset.

With a batch size of 8, having 24 fewer items results in exactly 3 fewer batches (or steps) per epoch for the temporal script.

When you multiply those 3 fewer steps per epoch by your total of 600 epochs, you get exactly 1,800 fewer steps over the entire training run.

79,800 baseline steps minus 1,800 steps equals exactly 78,000 steps for your temporal script.
You make a really great point. Because the temporal model intrinsically requires frame pairs to calculate optical flow, losing the first frame of a sequence is a natural, unavoidable characteristic of that specific architecture. It is not an arbitrary cut to the data.

You can absolutely keep the datasets and scripts exactly as they are. Comparing the models in their natural, out of the box state is a highly defensible approach for an ablation study. The baseline naturally uses every available frame, and the temporal model natively uses every available pair.

When you write up the methodology section for your main dissertation, you should just document this exact thought process. You can clearly explain that the difference in training iterations is a direct result of the temporal pairing constraint. The minor difference in total steps will not really be noticeable in your overall training time on the Hex cluster anyway.

