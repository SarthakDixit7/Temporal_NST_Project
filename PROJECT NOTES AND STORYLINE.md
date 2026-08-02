Here is a comprehensive, logically structured assembly of all your project notes. I have integrated your developmental phases, architectural decisions, mathematical formulations, and technical debugging into a unified document. This is formatted to be copied directly into your Obsidian vault to serve as the foundation for your methodology, implementation, and ablation chapters.

---

# Dissertation Technical Notes: Real-Time Video Neural Style Transfer

## 1. Project Motivation & Core Objective

The fundamental goal of this project is to achieve real-time video neural style transfer while maintaining strict temporal coherence across sequential frames.

Traditional neural style transfer methods rely on offline, iterative per-frame pixel optimisation (e.g., Gatys et al.), which is computationally heavy and far too slow for video applications. While implementing a feed-forward architecture resolves the processing speed bottleneck by applying style in a single pass, treating video as a sequence of independent images immediately exposes a classic temporal coherence problem: severe high-frequency flickering between frames.

The core intervention of this project is combining the inference speed of a feed-forward Transform Network with a custom temporal consistency loss driven by dense optical flow estimation.

---

## 2. Architecture & System Pipeline

The pipeline is built on three major architectural pillars, each addressing a specific limitation of standard single-frame style transfer.

### A. Feed-Forward Transform Network

* **Role:** Replaces online optimization to enable real-time execution speeds.
* **Mechanism:** Takes a raw video content frame $x_t$ as input and produces the stylised frame $y_t = T(x_t)$ in a single forward pass.

### B. Dense Optical Flow Estimation (RAFT)

* **Role:** Eliminates high-frequency flickering by anchoring the network's understanding of time and pixel movement.
* **Mechanism:** Instead of relying on basic, lightweight flow approximations, the architecture integrates **RAFT** (Recurrent All-Pairs Field Transforms). RAFT is highly effective at handling fast motion and large pixel displacements.
* **Implementation:** RAFT computes dense forward and backward optical flow fields between consecutive content frames ($x_{t-1}$ and $x_t$). This allows the network to warp the previously stylised output $y_{t-1}$ onto the current frame $y_t$, enforcing consistency by directly comparing the two.

### C. Feature & Loss Extractor (Pre-trained VGG Network)

* **Role:** Acts as a fixed feature extractor to evaluate perceptual content and artistic style loss.
* **Mechanism:** The network's loss function evaluates macro-level semantic structures and complex artistic motifs rather than low-level visual noise. A major architectural tweak was shifting the style feature extraction deeper into the network, explicitly targeting the `relu5_1` layer to refine the overall artistic quality.

---

## 3. Dataset & Preprocessing Strategy

A major conceptual shift during development was identifying and resolving a spatial scale mismatch. The network initially struggled to map the global composition of static reference paintings onto dynamic video frames. This was resolved via specific preprocessing techniques.

| Component | Source / Processing | Technical Rationale |
| --- | --- | --- |
| **Content Input** | Sequential video frames | Provides natural motion vectors necessary for accurate RAFT optical flow calculation. |
| **Style Input** | Static art images (WikiArt dataset) | Allows broad stylistic generalization across diverse artistic movements. |
| **Preprocessing** | Targeted texture cropping | Instead of feeding entire paintings into the style loss calculation, reference images are cropped to isolate specific brushstroke textures. This forces the Gram matrices to learn the physical texture of the art rather than attempting to replicate overarching spatial layout. |

---

## 4. Dynamic Feature Extraction & Spatial Regularisation

A fundamental challenge in generalised neural style transfer is the scale variance of artistic textures. The extraction of style representations via the VGG-19 Loss Network is highly dependent on the receptive field of the selected convolutional layers. To accommodate diverse artistic movements, the pipeline implements a dynamic routing system that adjusts feature extraction depth and spatial regularisation based on the structural requirements of the target style.

### High-Frequency Textures (e.g., Pointillism)

Styles characterised by microscopic, high-frequency visual noise, such as Georges Seurat’s Pointillism, require strict localisation to shallow network features.

* **Layer Selection:** Gram matrix computation is restricted to the early layers (`relu1_2`, `relu2_2`, and `relu3_3`). These act as low-level filter banks, capturing exact pixel-to-pixel colour variance and sharp dot textures without imposing overarching spatial geometry.
* **Spatial Regularisation:** The Total Variation (TV) penalty is explicitly disabled ($\mathcal{L}_{\text{tv}} = 0$). Because TV loss functions as an anisotropic spatial smoother, its activation would aggressively penalise the sharp, intentional pixel contrast required to render distinct pointillist dots, resulting in severe texture degradation and muddying.

### Macro-Structural Textures (e.g., Post-Impressionism)

Styles defined by thick, sweeping, and connected brushstrokes—such as Vincent van Gogh’s Post-Impressionism—demand macro-level structural coherence. Relying on shallow layers restricts the network to replicating microscopic paint fissures rather than sweeping global geometry.

* **Layer Selection:** Style loss calculation is expanded deeper into the network to include `relu4_1` and `relu5_1`. These deeper layers possess a broader receptive field, reproducing large, continuous stylistic structures.
* **Spatial Regularisation:** A minor Total Variation penalty ($\mathcal{L}_{\text{tv}} = 1 \times 10^{-3}$) is applied during training. This acts as a smoothing agent between sweeping brushstrokes, actively suppressing high-frequency checkerboard noise and preventing the stylised output from degenerating into static grain.

---

## 5. Loss Function Formulation

The final unified training loss is designed to balance three competing objectives dynamically within a single pass: preserving underlying video content, applying cropped artistic textures, and maintaining strict frame-to-frame stability.

$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{content}} + \beta \mathcal{L}_{\text{style}} + \gamma \mathcal{L}_{\text{temporal}}$$

1. **Content Loss ($\mathcal{L}_{\text{content}}$):** Keeps the core shapes intact. It measures the Euclidean distance between high-level feature maps of the content frame $x_t$ and generated frame $y_t$ extracted from VGG.
2. **Style Loss ($\mathcal{L}_{\text{style}}$):** Replicates the specific brushstrokes. It measures discrepancies between the Gram matrices of feature maps from the cropped style reference image and the generated frame $y_t$.
3. **Temporal Consistency Loss ($\mathcal{L}_{\text{temporal}}$):** Penalises differences in non-occluded regions. It calculates pixel-wise differences between the prior stylised frame $\hat{y}_{t-1}$ (warped using RAFT optical flow masks) and the current stylised frame $y_t$. By embedding this directly into the feed-forward training phase, the model inherently understands motion, eliminating the need for post-processing smoothing filters.

---

## 6. Technical Challenge: FP16 Overflow and NaN Propagation

During the implementation of the Pointillism style mode ($\mathcal{L}_{\text{tv}} = 0.0$), a severe numerical instability was observed. The network initially produced localized black artifacts concentrated on areas of high geometric complexity and non-rigid motion (e.g., faces and moving clouds). When the content loss weight was reduced to allow for stronger stylisation, the entire output tensor collapsed into pure black frames.

### Diagnostic Analysis

The localized artifacts were diagnosed as localized `NaN` (Not a Number) generation caused by gradient overflow within PyTorch’s Automatic Mixed Precision (`autocast`) framework.

1. **The Spark (FP16 Overflow):** By explicitly disabling the Total Variation penalty to preserve high-frequency pointillist dots, the network's spatial smoothing constraints were removed. When attempting to force extreme pixel contrast onto complex, non-rigid geometries, the gradients spiked violently. The 16-bit floating-point (FP16) architecture has a strict maximum value ceiling of 65,504. The high-frequency gradients exceeded this ceiling, resulting in a mathematical overflow that generated localized `NaN` values (rendered as black pixels).
2. **The Collapse (Gram Matrix Propagation):** The total network collapse was triggered by shifting the loss weighting. While content loss dominated, `NaN` values were quarantined to specific spatial coordinates, as content loss operates strictly pixel-by-pixel. However, when content weight was reduced, priority shifted to the Style Loss. Because style is calculated via a Gram Matrix—which computes the inner product of vectorized feature maps—the localized `NaN` values were cross-multiplied against the entire spatial grid. This caused an instantaneous global `NaN` propagation, collapsing the entire tensor.

### Resolution

To resolve this instability, the `autocast` mixed-precision context manager was disabled for the forward pass, forcing the network to calculate in standard 32-bit floating-point precision (FP32). FP32 possesses a vastly higher numerical ceiling ($\approx 3.4 \times 10^{38}$), which successfully absorbed the extreme gradient spikes generated by the zero-TV pointillist styling without overflowing, entirely eliminating both the localized artifacts and the global tensor collapse.

---

## 7. Project Novelty & Key Takeaways

* **Real-time Viability:** Proves that temporal loss integrated directly into feed-forward training eliminates the need for expensive, offline post-processing smoothing filters.
* **Temporal Stability:** Quantifies significant temporal stability improvements over standard single-frame feed-forward models, successfully resolving the classic flickering problem.
* **Dynamic Architectural Routing:** Demonstrates that a one-size-fits-all approach to layer selection and TV loss is insufficient for diverse art styles, proving the necessity of variable-depth feature extraction based on texture frequency.