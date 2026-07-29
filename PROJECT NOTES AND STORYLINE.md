# Project Storyline: Evolution of My Video Neural Style Transfer Model

## Phase 1: The Initial Concept and the Flickering Problem

I began this project with a clear objective to achieve real-time video neural style transfer. The immediate hurdle I faced was processing speed. Traditional methods relied on offline per-frame pixel optimisation which was far too slow for video applications.

To solve this I implemented a feed-forward Transform Network designed to apply style to a content frame $x_t$ in a single forward pass. While this achieved the necessary inference speed it immediately exposed the classic temporal coherence problem. Treating a video as a sequence of independent images resulted in severe high-frequency flickering between frames. I realised the network needed a way to understand time.

## Phase 2: Anchoring the Network with Optical Flow

To eliminate the flickering I knew the network needed to track pixel movement across sequential frames. My key intervention here was integrating dense optical flow estimation directly into the training pipeline.

Instead of relying on basic flow approximations I selected RAFT (Recurrent All-Pairs Field Transforms). This was a crucial architectural decision for me because RAFT handles fast motion and large pixel displacements effectively. I designed a system where RAFT computes the forward and backward flow vectors between frames $x_{t-1}$ and $x_t$. This allowed me to warp the previously stylised output and compare it directly to the current frame to enforce consistency.

## Phase 3: Tuning the VGG Feature Extractor

With the temporal constraints taking shape I noticed the actual artistic quality of the style transfer needed refinement. My initial outputs felt too noisy or superficial.

My thought process shifted to how the pre-trained VGG network was evaluating style. I made a significant architectural tweak by shifting the feature extraction deeper into the network specifically targeting the **`relu5_1`** layer. This forced the loss function to prioritise macro-level semantic structures and complex artistic motifs over low-level visual noise.

## Phase 4: Solving the Scale Mismatch via Cropping

Even with deeper VGG features I identified a spatial scale mismatch. My network was struggling to map the global composition of static reference paintings from the WikiArt dataset onto dynamic video frames.

My solution was a practical preprocessing input: targeted texture cropping. Instead of feeding entire paintings into the style loss calculation I cropped the reference images to isolate specific brushstroke textures. This was a vital conceptual shift. It forced the Gram matrices to learn the physical texture of the art rather than attempting to replicate the overarching layout of the original painting.

## Phase 5: Balancing the Tripartite Loss

The final stage of my development involved balancing the competing objectives within the loss function. I had to ensure the model preserved the underlying video content while applying the cropped artistic textures and maintaining strict frame-to-frame stability.

I structured a unified training loss that dynamically balanced these three elements:

1. **Content Loss:** Keeping the core shapes intact.
2. **Style Loss:** Replicating the cropped `relu5_1` brushstrokes.
3. **Temporal Consistency Loss:** Penalising differences in non-occluded regions using the RAFT optical flow masks.

By embedding this temporal loss directly into the feed-forward training phase I successfully eliminated the need for post-processing smoothing filters. My final architecture now inherently understands both artistic style and temporal motion in a single pass.



Here is a clean, structured set of study and summary notes covering your code development, architectural choices, and project iterations for your Video Neural Style Transfer dissertation. You can copy this directly into your Obsidian vault to draft your methodology and implementation sections.

---

# Dissertation Technical Notes: Real-Time Video Neural Style Transfer

## 1. Project Motivation & Core Objective

* **Goal:** Achieve real-time video neural style transfer while maintaining temporal coherence across sequential frames and eliminating high-frequency frame flickering.
* **Core Approach:** Train a feed-forward Transform Network capable of performing style transfer in a single forward pass per frame, rather than relying on slow, iterative per-frame pixel optimisation (e.g. Gatys et al.).
* **Key Differentiator:** Combining the inference speed of a feed-forward architecture with a temporal consistency loss driven by dense optical flow estimation.

---

## 2. Architecture & System Pipeline

### A. Feed-Forward Transform Network

* **Role:** Takes a content frame $x_t$ as input and produces the stylized frame $y_t = T(x_t)$ in a single forward pass.
* **Design Choice:** Feed-forward design replaces online optimization, enabling real-time execution speeds required for video processing pipelines.

### B. Feature & Loss Extractor (Pre-trained VGG Network)

* **Role:** Acts as a fixed feature extractor to evaluate perceptual content and artistic style loss.
* **Layer Selection & Fine-Tuning:**
* Experimented with deeper feature layers, specifically targeting **`relu5_1`** (and surrounding activations) to capture higher-level semantic structures and complex artistic features.
* Adjusting feature extraction depth helps balance local brushstroke textures with macro-level image structures.



### C. Dense Optical Flow Estimation (RAFT)

* **Role:** Utilises **RAFT** (Recurrent All-Pairs Field Transforms) to compute dense optical flow fields between consecutive content video frames ($x_{t-1}$ and $x_t$).
* **Function:** Provides forward/backward flow vectors and occlusion masks to warp the previous stylized output $y_{t-1}$ onto the current frame $y_t$.

---

## 3. Dataset & Preprocessing Strategy

| Component | Source / Processing | Technical Rationale |
| --- | --- | --- |
| **Content Input** | Sequential video frames | Provides natural motion vectors for optical flow calculation |
| **Style Input** | Static art images (WikiArt dataset) | Allows broad stylistic generalization across artistic movements |
| **Style Preprocessing** | Targeted texture cropping | Cropping style images focuses the Gram matrix calculations on detailed brushstroke textures rather than global spatial composition |

---

## 4. Loss Function Formulation

The total training loss combines three main components:

$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{content}} + \beta \mathcal{L}_{\text{style}} + \gamma \mathcal{L}_{\text{temporal}}$$

1. **Content Loss ($\mathcal{L}_{\text{content}}$):**
* Measures Euclidean distance between high-level feature maps of the content frame $x_t$ and generated frame $y_t$ extracted from VGG.


2. **Style Loss ($\mathcal{L}_{\text{style}}$):**
* Measures discrepancies between the Gram matrices of feature maps from the cropped style reference image and the generated frame $y_t$.


3. **Temporal Consistency Loss ($\mathcal{L}_{\text{temporal}}$):**
* Penalizes pixel-wise differences between the warped prior stylized frame $\hat{y}_{t-1}$ (warped using RAFT optical flow) and the current stylized frame $y_t$ in non-occluded regions.



---

## 5. Key Ideas, Changes & Iterative Refinements

1. **Shift to Dense Flow (RAFT):** Upgraded flow estimation to RAFT to handle fast motion and large displacements better than traditional sparse or lighter optical flow approximations.
2. **VGG Layer Selection (`relu5_1` Shift):** Shifted style evaluation deeper into VGG layers to capture richer stylistic motifs and reduce noisy high-frequency artifacts.
3. **Style Preprocessing Refinement:** Implemented style image cropping to prevent spatial scale mismatch between style references and video frames.

---

## 6. Novelty & Dissertation Key Points

* **Real-time Viability:** Proves that temporal loss integrated directly into feed-forward training eliminates the need for post-processing smoothing filters.
* **Temporal Stability:** Quantifies temporal stability improvements over standard single-frame feed-forward models (which suffer severe frame-to-frame flickering).

---



 Dynamic Feature Extraction and Spatial RegularisationA fundamental challenge in generalised neural style transfer is the scale variance of artistic textures. The extraction of style representations via the VGG-19 Loss Network is highly dependent on the receptive field of the selected convolutional layers. To accommodate diverse artistic movements, the pipeline implements a dynamic routing system that adjusts feature extraction depth and spatial regularisation based on the structural requirements of the target style.High-Frequency Textures (e.g., Pointillism)Styles characterised by microscopic, high-frequency visual noise, such as Georges Seurat’s Pointillism, require strict localisation to shallow network features. For such styles, Gram matrix computation is restricted to the early layers (relu1_2, relu2_2, and relu3_3). These layers act as low-level filter banks, capturing exact pixel-to-pixel colour variance and sharp dot textures without attempting to impose overarching spatial geometry. Furthermore, the Total Variation (TV) penalty is explicitly disabled ($\mathcal{L}_{\text{tv}} = 0$). Because TV loss functions as an anisotropic spatial smoother, its activation would aggressively penalise the sharp, intentional pixel contrast required to render distinct pointillist dots, resulting in severe texture degradation and muddying.Macro-Structural Textures (e.g., Post-Impressionism)Conversely, styles defined by thick, sweeping, and connected brushstrokes—such as Vincent van Gogh’s Post-Impressionism—demand macro-level structural coherence. Relying solely on shallow layers for these styles restricts the network to replicating microscopic paint fissures rather than the sweeping global geometry of the brushstrokes. Therefore, the style loss calculation is expanded deeper into the network architecture to include relu4_1 and relu5_1. These deeper layers possess a broader receptive field, allowing the network to capture and reproduce large, continuous stylistic structures. Additionally, a minor Total Variation penalty ($\mathcal{L}_{\text{tv}} = 1\times 10^{-3}$) is applied during training. This spatial regularisation acts as a smoothing agent between the sweeping brushstrokes, actively suppressing high-frequency checkerboard noise and preventing the stylised output from degenerating into static grain.