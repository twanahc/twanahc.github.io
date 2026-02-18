---
layout: post
title: "Video Inpainting: Editing Parts of Generated Video Without Regenerating Everything"
date: 2025-12-30
category: models
---

You generated a ten-second AI video clip. The scene, the motion, the lighting --- all correct. Except for one thing: there is a distracting object in the lower right, or the actor's shirt is the wrong color, or a three-frame artifact glitches across the background. You do not want to regenerate the entire clip. Regeneration is expensive, stochastic (you will get a completely different result), and wasteful. What you want is surgical editing: change only the region that needs fixing, leave everything else pixel-identical.

This is video inpainting. It is one of the most practically important capabilities for any AI video platform, and it sits at the intersection of diffusion models, optical flow estimation, temporal consistency, and mask generation. This post is the complete deep dive: mathematical foundations, the algorithms that make it work, the practical pipeline from mask to final composite, and the evaluation metrics that tell you whether you succeeded.

---

## Table of Contents

1. [What Video Inpainting Is](#what-video-inpainting-is)
2. [The Mathematical Formulation](#the-mathematical-formulation)
3. [Image Inpainting Foundations: Diffusion Approach](#image-inpainting-foundations-diffusion-approach)
4. [From Images to Video: The Temporal Consistency Problem](#from-images-to-video-the-temporal-consistency-problem)
5. [Approach 1: Per-Frame Inpainting with Temporal Smoothing](#approach-1-per-frame-inpainting-with-temporal-smoothing)
6. [Approach 2: Joint Spatiotemporal Inpainting](#approach-2-joint-spatiotemporal-inpainting)
7. [Approach 3: Flow-Guided Propagation](#approach-3-flow-guided-propagation)
8. [Mask Generation: Manual, SAM, and Text-Prompted](#mask-generation-manual-sam-and-text-prompted)
9. [The Complete Inpainting Pipeline](#the-complete-inpainting-pipeline)
10. [Practical Tools and Comparison](#practical-tools-and-comparison)
11. [Use Cases for Video Platforms](#use-cases-for-video-platforms)
12. [Quality Evaluation Metrics](#quality-evaluation-metrics)
13. [Implementation: Full Code Pipeline](#implementation-full-code-pipeline)
14. [Conclusion](#conclusion)

---

## What Video Inpainting Is

Video inpainting is the task of filling in a specified region of a video with new content that is spatially and temporally coherent with the surrounding context. The region to be filled is defined by a mask --- a binary (or soft) signal indicating which pixels should be replaced.

The term comes from art restoration, where conservators "inpaint" damaged regions of paintings by carefully matching the surrounding style, color, and texture. The computational analog must do the same, but for every frame of a video, maintaining consistency across time.

There are several distinct use cases:

- **Object removal**: Erase an unwanted element (a watermark, a passerby, a boom microphone) and fill the region with plausible background.
- **Object replacement**: Replace one object with another (swap a product, change clothing) guided by a text prompt or reference image.
- **Background replacement**: Keep the foreground subject, replace the entire background.
- **Artifact correction**: Fix localized generation artifacts (flickering, color bleeding, temporal glitches) in specific frames without touching the rest of the video.
- **Occlusion filling**: Complete regions that are occluded or missing due to view changes or sensor issues.

<svg viewBox="0 0 800 320" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 800px; display: block; margin: 2em auto;">
  <text x="400" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#333">Video Inpainting: Concept Overview</text>
  <!-- Original frame -->
  <rect x="30" y="50" width="160" height="100" rx="6" fill="#e3f2fd" stroke="#4fc3f7" stroke-width="2"/>
  <text x="110" y="80" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#333">Original Frame</text>
  <rect x="70" y="95" width="40" height="30" rx="3" fill="#ef5350" opacity="0.7"/>
  <text x="90" y="115" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="white">object</text>
  <rect x="50" y="130" width="100" height="10" rx="2" fill="#8bc34a" opacity="0.5"/>
  <!-- Plus sign -->
  <text x="215" y="105" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" fill="#333">+</text>
  <!-- Mask -->
  <rect x="240" y="50" width="160" height="100" rx="6" fill="#f5f5f5" stroke="#ffa726" stroke-width="2"/>
  <text x="320" y="80" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#333">Binary Mask M</text>
  <rect x="280" y="95" width="40" height="30" rx="3" fill="white" stroke="#ffa726" stroke-width="2"/>
  <text x="300" y="115" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#ffa726">M=1</text>
  <rect x="260" y="85" width="100" height="50" fill="none" stroke="#ffa726" stroke-width="1" stroke-dasharray="4"/>
  <!-- Arrow -->
  <line x1="420" y1="100" x2="470" y2="100" stroke="#333" stroke-width="2" marker-end="url(#inpaint-arrow)"/>
  <text x="445" y="90" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333">inpaint</text>
  <defs>
    <marker id="inpaint-arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>
  <!-- Result frame -->
  <rect x="490" y="50" width="160" height="100" rx="6" fill="#e8f5e9" stroke="#8bc34a" stroke-width="2"/>
  <text x="570" y="80" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#333">Inpainted Frame</text>
  <rect x="530" y="95" width="40" height="30" rx="3" fill="#8bc34a" opacity="0.3"/>
  <text x="550" y="115" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#333">filled</text>
  <rect x="510" y="130" width="100" height="10" rx="2" fill="#8bc34a" opacity="0.5"/>
  <!-- Temporal axis -->
  <text x="400" y="200" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="bold" fill="#333">Temporal Consistency Across Frames</text>
  <!-- Frame sequence -->
  <rect x="60" y="220" width="80" height="50" rx="4" fill="#e3f2fd" stroke="#4fc3f7" stroke-width="1.5"/>
  <text x="100" y="250" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333">Frame t-2</text>
  <rect x="170" y="220" width="80" height="50" rx="4" fill="#e3f2fd" stroke="#4fc3f7" stroke-width="1.5"/>
  <text x="210" y="250" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333">Frame t-1</text>
  <rect x="280" y="220" width="80" height="50" rx="4" fill="#fff3e0" stroke="#ffa726" stroke-width="2"/>
  <text x="320" y="245" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#ffa726" font-weight="bold">Frame t</text>
  <text x="320" y="258" text-anchor="middle" font-family="Arial, sans-serif" font-size="8" fill="#ffa726">(inpainted)</text>
  <rect x="390" y="220" width="80" height="50" rx="4" fill="#e3f2fd" stroke="#4fc3f7" stroke-width="1.5"/>
  <text x="430" y="250" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333">Frame t+1</text>
  <rect x="500" y="220" width="80" height="50" rx="4" fill="#e3f2fd" stroke="#4fc3f7" stroke-width="1.5"/>
  <text x="540" y="250" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333">Frame t+2</text>
  <!-- Arrows between frames -->
  <line x1="140" y1="245" x2="168" y2="245" stroke="#8bc34a" stroke-width="1.5" marker-end="url(#inpaint-arrow)"/>
  <line x1="250" y1="245" x2="278" y2="245" stroke="#8bc34a" stroke-width="1.5" marker-end="url(#inpaint-arrow)"/>
  <line x1="360" y1="245" x2="388" y2="245" stroke="#8bc34a" stroke-width="1.5" marker-end="url(#inpaint-arrow)"/>
  <line x1="470" y1="245" x2="498" y2="245" stroke="#8bc34a" stroke-width="1.5" marker-end="url(#inpaint-arrow)"/>
  <text x="400" y="300" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#666">Inpainted region must be temporally consistent with all neighboring frames</text>
</svg>

The challenge in video inpainting --- the thing that separates it from image inpainting --- is temporal consistency. Each frame must not only look plausible on its own, but must be coherent with its neighbors in time. A flicker, a sudden color shift, or a jump in the inpainted region's geometry across frames is immediately visible to the human eye, even if each individual frame looks fine in isolation.

---

## The Mathematical Formulation

Let $V = \{I_1, I_2, \ldots, I_T\}$ be a video consisting of $T$ frames, where each frame $I_t \in \mathbb{R}^{H \times W \times 3}$. Let $M = \{M_1, M_2, \ldots, M_T\}$ be the corresponding mask sequence, where $M_t \in \{0, 1\}^{H \times W}$ with $M_t(i,j) = 1$ indicating that pixel $(i,j)$ in frame $t$ should be inpainted.

Optionally, we have a conditioning signal $c$ (a text prompt describing what should fill the masked region, or nothing for pure removal).

The video inpainting problem is:

$$\text{Find } V' = \{I'_1, I'_2, \ldots, I'_T\} \text{ such that:}$$

**Constraint 1 --- Unmasked Preservation:**

$$I'_t(i,j) = I_t(i,j) \quad \forall (i,j) \text{ where } M_t(i,j) = 0$$

This is the fundamental constraint. Everything outside the mask must be pixel-identical to the original.

**Constraint 2 --- Spatial Coherence:**

$$I'_t \text{ is spatially plausible within frame } t$$

The inpainted region must blend seamlessly with the surrounding content in each frame. No visible boundaries, no texture discontinuities, no color mismatches.

**Constraint 3 --- Temporal Coherence:**

$$\|I'_t(i,j) - \text{warp}(I'_{t-1}, F_{t-1 \to t})(i,j)\| \text{ is small } \forall (i,j) \in M_t$$

where $F_{t-1 \to t}$ is the optical flow field from frame $t-1$ to $t$. In words: the inpainted content should follow the same motion patterns as the rest of the video. If the camera is panning left, the background fill should also pan left.

**Constraint 4 --- Semantic Coherence (if prompted):**

$$V'_M \text{ is consistent with conditioning signal } c$$

If a text prompt says "replace the car with a bicycle," the inpainted region should contain a bicycle that plausibly occupies the same spatial extent and follows the same motion trajectory as the original car.

We can write the optimization objective compactly as:

$$V'^* = \arg\min_{V'} \left[ \lambda_s \mathcal{L}_\text{spatial}(V') + \lambda_t \mathcal{L}_\text{temporal}(V') + \lambda_c \mathcal{L}_\text{condition}(V', c) \right] \quad \text{s.t.} \quad V'_{\bar{M}} = V_{\bar{M}}$$

where $\bar{M}$ denotes the complement of the mask (the unmasked region), and $\lambda_s, \lambda_t, \lambda_c$ are weights balancing spatial quality, temporal consistency, and prompt adherence.

---

## Image Inpainting Foundations: Diffusion Approach

Before tackling video, we need to understand how diffusion-based image inpainting works, since it forms the foundation for all video inpainting methods.

### The Standard Diffusion Training Loss

A diffusion model $\epsilon_\theta$ is trained to predict the noise added to a clean image $x_0$:

$$L = \mathbb{E}_{t, x_0, \epsilon \sim \mathcal{N}(0,I)} \left[ \|\epsilon - \epsilon_\theta(x_t, t, c)\|^2 \right]$$

where $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ is the noised image at timestep $t$, and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ is the cumulative noise schedule.

### Inpainting by Conditioning on the Unmasked Region

The key insight for diffusion inpainting: during the reverse process (denoising), at each step, we replace the unmasked region with the appropriately noised version of the original image. The model only generates content for the masked region, but it can "see" the surrounding context and use it to guide generation.

At each denoising step $t$, the update becomes:

$$x_{t-1} = M \odot \hat{x}_{t-1}^\text{gen} + (1 - M) \odot \hat{x}_{t-1}^\text{orig}$$

where:
- $\hat{x}_{t-1}^\text{gen}$ is the denoised prediction from the model
- $\hat{x}_{t-1}^\text{orig} = \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon$ is the original image re-noised to timestep $t-1$
- $M$ is the binary mask ($1$ = inpaint, $0$ = keep original)
- $\odot$ denotes element-wise multiplication

This is called **repaint-style** inpainting (from the RePaint paper by Lugmayr et al., 2022). The model denoises freely in the masked region while being anchored to the original content in the unmasked region.

### The Inpainting-Specific Training Loss

Dedicated inpainting models are trained with a modified loss that explicitly conditions on the masked input:

$$L_\text{inpaint} = \mathbb{E}_{t, x_0, \epsilon, M} \left[ \left\| \epsilon - \epsilon_\theta\left(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \; t, \; c, \; M \odot x_0, \; M \right) \right\|^2 \right]$$

Let us break down every term:

| Symbol | Meaning |
|--------|---------|
| $\epsilon$ | The noise sampled from $\mathcal{N}(0, I)$ |
| $\epsilon_\theta$ | The neural network predicting the noise |
| $\sqrt{\bar{\alpha}_t}$ | Signal scaling factor at timestep $t$ |
| $x_0$ | The clean original image |
| $\sqrt{1 - \bar{\alpha}_t}$ | Noise scaling factor at timestep $t$ |
| $t$ | The diffusion timestep |
| $c$ | The text conditioning (prompt) |
| $M \odot x_0$ | The masked image (only unmasked pixels visible) |
| $M$ | The binary mask itself (as a spatial channel) |

The model receives the noisy image, the timestep, the text prompt, the visible (unmasked) region of the original, and the mask as a spatial channel. This gives it explicit information about what is known and what needs to be generated.

The key difference from generic diffusion: **the model is trained from the start to understand the concept of a mask and to use unmasked context**. This produces much better boundary blending than the repaint-style approach, which retrofits inpainting onto a model that was never trained for it.

<svg viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 800px; display: block; margin: 2em auto;">
  <text x="400" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#333">Diffusion Inpainting: Denoising with Masked Context</text>
  <defs>
    <marker id="inp-arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>
  <!-- Timestep labels -->
  <text x="100" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#666">t = T (pure noise)</text>
  <text x="400" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#666">t = T/2</text>
  <text x="700" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#666">t = 0 (clean)</text>
  <!-- Frame at t=T -->
  <rect x="30" y="70" width="140" height="90" rx="5" fill="#f5f5f5" stroke="#ccc" stroke-width="1"/>
  <!-- Noise pattern -->
  <rect x="35" y="75" width="130" height="80" rx="3" fill="#e0e0e0"/>
  <text x="100" y="120" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#999">Random noise</text>
  <!-- Mask overlay -->
  <rect x="60" y="90" width="50" height="35" rx="2" fill="#ffa726" opacity="0.4" stroke="#ffa726" stroke-width="1" stroke-dasharray="3"/>
  <text x="85" y="112" text-anchor="middle" font-family="Arial, sans-serif" font-size="8" fill="#ffa726">mask</text>
  <!-- Arrow -->
  <line x1="180" y1="115" x2="230" y2="115" stroke="#333" stroke-width="1.5" marker-end="url(#inp-arrow)"/>
  <text x="205" y="107" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#333">denoise</text>
  <!-- Frame at t=T/2 -->
  <rect x="240" y="70" width="140" height="90" rx="5" fill="#f5f5f5" stroke="#ccc" stroke-width="1"/>
  <rect x="245" y="75" width="130" height="80" rx="3" fill="#e8f5e9"/>
  <!-- Unmasked area partially resolved -->
  <rect x="245" y="75" width="45" height="80" rx="3" fill="#c8e6c9"/>
  <rect x="320" y="75" width="55" height="80" rx="3" fill="#c8e6c9"/>
  <!-- Masked area still noisy -->
  <rect x="295" y="90" width="30" height="35" rx="2" fill="#fff3e0" stroke="#ffa726" stroke-width="1"/>
  <text x="310" y="112" text-anchor="middle" font-family="Arial, sans-serif" font-size="7" fill="#ffa726">forming</text>
  <text x="310" y="145" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#333">Context guides fill</text>
  <!-- Arrow -->
  <line x1="390" y1="115" x2="520" y2="115" stroke="#333" stroke-width="1.5" marker-end="url(#inp-arrow)"/>
  <text x="455" y="107" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#333">continue denoising</text>
  <!-- Replacement step annotation -->
  <rect x="395" y="125" width="120" height="30" rx="4" fill="#fff8e1" stroke="#ffa726" stroke-width="1"/>
  <text x="455" y="138" text-anchor="middle" font-family="Arial, sans-serif" font-size="8" fill="#ffa726">Replace unmasked with</text>
  <text x="455" y="148" text-anchor="middle" font-family="Arial, sans-serif" font-size="8" fill="#ffa726">re-noised original at each step</text>
  <!-- Frame at t=0 -->
  <rect x="530" y="70" width="140" height="90" rx="5" fill="#f5f5f5" stroke="#8bc34a" stroke-width="2"/>
  <rect x="535" y="75" width="130" height="80" rx="3" fill="#e8f5e9"/>
  <!-- Clean result -->
  <rect x="535" y="75" width="45" height="80" rx="3" fill="#c8e6c9"/>
  <rect x="620" y="75" width="45" height="80" rx="3" fill="#c8e6c9"/>
  <rect x="583" y="90" width="35" height="35" rx="2" fill="#8bc34a" opacity="0.3" stroke="#8bc34a" stroke-width="1"/>
  <text x="600" y="112" text-anchor="middle" font-family="Arial, sans-serif" font-size="8" fill="#333">filled</text>
  <text x="600" y="145" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#8bc34a" font-weight="bold">Seamless result</text>
  <!-- Bottom: Mathematical notation -->
  <rect x="30" y="200" width="740" height="170" rx="8" fill="#fafafa" stroke="#ddd" stroke-width="1"/>
  <text x="400" y="225" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="bold" fill="#333">At Each Denoising Step t:</text>
  <text x="400" y="255" text-anchor="middle" font-family="monospace" font-size="12" fill="#333">x_{t-1} = M * x_{t-1}^gen + (1-M) * x_{t-1}^orig</text>
  <text x="100" y="290" text-anchor="start" font-family="Arial, sans-serif" font-size="11" fill="#4fc3f7">x_{t-1}^gen: Model's denoised prediction (free generation in masked area)</text>
  <text x="100" y="310" text-anchor="start" font-family="Arial, sans-serif" font-size="11" fill="#8bc34a">x_{t-1}^orig: Original image re-noised to level t-1 (anchors unmasked area)</text>
  <text x="100" y="330" text-anchor="start" font-family="Arial, sans-serif" font-size="11" fill="#ffa726">M: Binary mask (1 = generate, 0 = keep original)</text>
  <text x="100" y="350" text-anchor="start" font-family="Arial, sans-serif" font-size="11" fill="#ef5350">Result: Generated content blends with preserved context at every noise level</text>
</svg>

### Boundary Artifacts and Solutions

The naive replace-at-each-step approach has a subtle problem: the generated region and the re-noised original region are statistically independent. They were produced by different processes (the model's generation vs. a forward noise process on the original). At the boundary between masked and unmasked regions, there can be visible seams because the two regions do not share a consistent noise realization.

Solutions include:

1. **Blurred masks**: Instead of a binary mask, use a soft mask with Gaussian-blurred edges. The transition zone blends generated and original content smoothly.

2. **RePaint resampling**: Jump back to a higher noise level periodically during denoising, re-noising the entire image and then re-replacing the unmasked region. This gives the model multiple chances to harmonize the boundary. Formally, instead of going $t \to t-1$ monotonically, the schedule includes jumps $t-1 \to t-1+j$ (re-noise) followed by $t-1+j \to t-1$ (denoise again), repeated $r$ times.

3. **Gradient-guided harmonization**: Add a gradient penalty that encourages the pixel gradients across the mask boundary to be smooth:

$$\mathcal{L}_\text{boundary} = \sum_{(i,j) \in \partial M} \|\nabla I'(i,j) - \nabla I(i,j)\|^2$$

where $\partial M$ is the set of pixels at the mask boundary.

---

## From Images to Video: The Temporal Consistency Problem

Applying image inpainting independently to each frame of a video produces results that look plausible in still frames but flicker catastrophically when played as a sequence. The reason is that diffusion is stochastic: each frame's inpainted region is sampled independently from the conditional distribution, and there is no mechanism ensuring that frame $t$'s sample is consistent with frame $t-1$'s sample.

Consider a simple example: inpainting a grass region across 30 frames. Each frame independently generates plausible grass texture. But the grass texture is different in every frame --- different blade positions, different shading, different color distributions. When played back, the region appears to "boil" or "shimmer" with random noise. This is immediately perceptible and unacceptable.

The temporal consistency problem can be formalized using optical flow. Let $F_{t \to t+1}: \mathbb{R}^{H \times W \times 2}$ be the optical flow field from frame $t$ to $t+1$. Define the warping operator:

$$\text{warp}(I_t, F_{t \to t+1})(i,j) = I_t\left(i + F_{t \to t+1}^x(i,j), \; j + F_{t \to t+1}^y(i,j)\right)$$

Temporal consistency requires:

$$\mathcal{L}_\text{temporal} = \sum_{t=1}^{T-1} \sum_{(i,j) \in M_t \cap M_{t+1}} \|I'_{t+1}(i,j) - \text{warp}(I'_t, F_{t \to t+1})(i,j)\|^2$$

This says: for every pixel in the inpainted region, the content at frame $t+1$ should match what you would get by warping frame $t$'s content forward according to the optical flow. In other words, the inpainted content should move the same way as everything else in the video.

---

## Approach 1: Per-Frame Inpainting with Temporal Smoothing

The simplest approach: inpaint each frame independently, then apply temporal smoothing as a post-processing step.

### Step 1: Independent Per-Frame Inpainting

Run a diffusion inpainting model on each frame:

$$I'_t = \text{Inpaint}(I_t, M_t, c) \quad \text{for } t = 1, \ldots, T$$

### Step 2: Temporal Smoothing

Apply a temporal filter to the inpainted regions. Common approaches include:

**Moving average**:

$$\hat{I}'_t(i,j) = \frac{1}{2k+1} \sum_{s=t-k}^{t+k} \text{warp}(I'_s, F_{s \to t})(i,j) \quad \forall (i,j) \in M_t$$

This warps $k$ neighboring frames to the current frame's coordinate system and averages them. The optical flow alignment is critical --- without it, you would simply blur the content.

**Temporal bilateral filter**: Weight the contributions by both temporal distance and color similarity:

$$\hat{I}'_t(i,j) = \frac{\sum_{s=t-k}^{t+k} w_s \cdot \text{warp}(I'_s, F_{s \to t})(i,j)}{\sum_{s=t-k}^{t+k} w_s}$$

where $w_s = \exp\left(-\frac{(s-t)^2}{2\sigma_t^2}\right) \cdot \exp\left(-\frac{\|I'_s - I'_t\|^2}{2\sigma_c^2}\right)$

### Pros and Cons

| Advantage | Disadvantage |
|-----------|-------------|
| Simple to implement | Temporal blurring reduces sharpness |
| Can use any image inpainting model | Flow errors cause ghosting artifacts |
| Parallelizable across frames | Does not handle large disoccluded regions |
| Low engineering complexity | Boundary artifacts between smoothed and unsmoothed regions |

This approach is viable for mild artifacts or small masks, but it fundamentally cannot handle large inpainted regions or complex motion.

---

## Approach 2: Joint Spatiotemporal Inpainting

The correct approach: treat the video as a 3D spatiotemporal volume and inpaint it jointly. This is what video diffusion models do natively.

### 3D Diffusion for Video Inpainting

A video diffusion model operates on the full spatiotemporal tensor $V \in \mathbb{R}^{T \times H \times W \times 3}$. The model's architecture includes temporal attention layers that allow each frame to attend to other frames, ensuring consistency.

The inpainting formulation extends naturally to 3D:

$$V_{t-1} = M_\text{3D} \odot \hat{V}_{t-1}^\text{gen} + (1 - M_\text{3D}) \odot \hat{V}_{t-1}^\text{orig}$$

where $M_\text{3D} \in \{0,1\}^{T \times H \times W}$ is the 3D spatiotemporal mask, and the replacement happens at each diffusion denoising step across all frames simultaneously.

The training loss for a video inpainting model is:

$$L = \mathbb{E}_{t, V_0, \epsilon, M_\text{3D}} \left[ \left\| \epsilon - \epsilon_\theta\left(\sqrt{\bar{\alpha}_t} V_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \; t, \; c, \; (1-M_\text{3D}) \odot V_0, \; M_\text{3D}\right) \right\|^2 \right]$$

where now $\epsilon \in \mathbb{R}^{T \times H \times W \times 3}$ is spatiotemporal noise.

### Temporal Attention Mechanism

The critical component is temporal self-attention. For a feature map at spatial position $(i,j)$, the model computes attention across the temporal dimension:

$$\text{TemporalAttn}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

where the queries, keys, and values are computed from features at the same spatial position across all frames:

$$Q = W_Q h_{:,i,j}, \quad K = W_K h_{:,i,j}, \quad V = W_V h_{:,i,j}$$

with $h_{:,i,j} \in \mathbb{R}^{T \times d}$ being the feature vector at position $(i,j)$ across all $T$ frames.

This allows the model to directly enforce temporal consistency: when generating frame $t$'s content in the masked region, it can attend to what it has generated (or is generating) in frames $t-1, t+1, t-2, t+2, \ldots$

### Computational Cost

The downside is compute. Joint spatiotemporal processing scales as $O(T^2 H W)$ for temporal attention (quadratic in the number of frames). For a 5-second, 30fps, 1080p video, this is $T=150$ frames at $1920 \times 1080$. Even in latent space (downsampled by 8x), this is computationally demanding.

Practical solutions:
- **Sliding window**: Process chunks of $k$ frames (typically 16--32) with overlap, blending the overlapping regions.
- **Sparse temporal attention**: Attend to a subset of frames (e.g., every 4th frame plus the immediate neighbors) instead of all frames.
- **Latent space processing**: Encode the video to latent space first, reducing spatial dimensions by 8x.

<svg viewBox="0 0 800 360" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 800px; display: block; margin: 2em auto;">
  <text x="400" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#333">Joint Spatiotemporal vs. Per-Frame Inpainting</text>
  <defs>
    <marker id="st-arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>
  <!-- Per-frame column -->
  <text x="200" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="bold" fill="#ef5350">Per-Frame (Independent)</text>
  <rect x="70" y="70" width="60" height="40" rx="4" fill="#ffebee" stroke="#ef5350" stroke-width="1.5"/>
  <text x="100" y="95" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#333">F1</text>
  <rect x="140" y="70" width="60" height="40" rx="4" fill="#ffebee" stroke="#ef5350" stroke-width="1.5"/>
  <text x="170" y="95" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#333">F2</text>
  <rect x="210" y="70" width="60" height="40" rx="4" fill="#ffebee" stroke="#ef5350" stroke-width="1.5"/>
  <text x="240" y="95" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#333">F3</text>
  <rect x="280" y="70" width="60" height="40" rx="4" fill="#ffebee" stroke="#ef5350" stroke-width="1.5"/>
  <text x="310" y="95" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#333">F4</text>
  <!-- No connections = independent -->
  <text x="200" y="130" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#ef5350">No information sharing</text>
  <text x="200" y="145" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#ef5350">= temporal flickering</text>
  <!-- Joint column -->
  <text x="600" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="bold" fill="#8bc34a">Joint Spatiotemporal</text>
  <rect x="460" y="65" width="280" height="55" rx="6" fill="#f1f8e9" stroke="#8bc34a" stroke-width="2"/>
  <rect x="470" y="72" width="55" height="38" rx="3" fill="#c8e6c9" stroke="#8bc34a" stroke-width="1"/>
  <text x="497" y="95" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#333">F1</text>
  <rect x="535" y="72" width="55" height="38" rx="3" fill="#c8e6c9" stroke="#8bc34a" stroke-width="1"/>
  <text x="562" y="95" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#333">F2</text>
  <rect x="600" y="72" width="55" height="38" rx="3" fill="#c8e6c9" stroke="#8bc34a" stroke-width="1"/>
  <text x="627" y="95" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#333">F3</text>
  <rect x="665" y="72" width="55" height="38" rx="3" fill="#c8e6c9" stroke="#8bc34a" stroke-width="1"/>
  <text x="692" y="95" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#333">F4</text>
  <!-- Bidirectional temporal connections -->
  <line x1="525" y1="91" x2="535" y2="91" stroke="#8bc34a" stroke-width="2"/>
  <line x1="590" y1="91" x2="600" y2="91" stroke="#8bc34a" stroke-width="2"/>
  <line x1="655" y1="91" x2="665" y2="91" stroke="#8bc34a" stroke-width="2"/>
  <text x="600" y="138" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#8bc34a">Temporal attention connects all frames</text>
  <text x="600" y="153" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#8bc34a">= temporal consistency built in</text>
  <!-- Comparison table below -->
  <rect x="50" y="175" width="700" height="170" rx="8" fill="#fafafa" stroke="#ddd" stroke-width="1"/>
  <text x="400" y="200" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="bold" fill="#333">Comparison</text>
  <!-- Headers -->
  <text x="200" y="225" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#333">Property</text>
  <text x="420" y="225" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#ef5350">Per-Frame</text>
  <text x="630" y="225" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#8bc34a">Joint</text>
  <line x1="80" y1="232" x2="720" y2="232" stroke="#ddd" stroke-width="1"/>
  <!-- Rows -->
  <text x="200" y="250" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333">Temporal consistency</text>
  <text x="420" y="250" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#ef5350">Poor (post-hoc)</text>
  <text x="630" y="250" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#8bc34a">Excellent (native)</text>
  <text x="200" y="270" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333">Compute cost</text>
  <text x="420" y="270" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#8bc34a">O(T * HW)</text>
  <text x="630" y="270" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#ef5350">O(T^2 * HW)</text>
  <text x="200" y="290" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333">Memory usage</text>
  <text x="420" y="290" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#8bc34a">Low (single frame)</text>
  <text x="630" y="290" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#ef5350">High (all frames)</text>
  <text x="200" y="310" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333">Large mask quality</text>
  <text x="420" y="310" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#ef5350">Inconsistent</text>
  <text x="630" y="310" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#8bc34a">Consistent</text>
  <text x="200" y="330" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333">Parallelization</text>
  <text x="420" y="330" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#8bc34a">Fully parallel</text>
  <text x="630" y="330" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#ffa726">Chunk-parallel</text>
</svg>

---

## Approach 3: Flow-Guided Propagation

The most practical approach for production systems: inpaint a small number of keyframes with high quality, then propagate the inpainted content to other frames using optical flow. This is the strategy used by ProPainter (Li et al., 2023) and related methods.

### The ProPainter Pipeline

**Step 1: Compute optical flow for the entire video.**

Use a pretrained optical flow model (RAFT, GMFlow, or similar) to compute forward and backward flow fields:

$$F_{t \to t+1}, \quad F_{t+1 \to t} \quad \text{for } t = 1, \ldots, T-1$$

**Step 2: Complete the flow in masked regions.**

The optical flow itself has missing values in the masked region (we cannot compute flow where we do not have content). ProPainter uses a flow completion network to hallucinate plausible flow in the masked area based on the surrounding flow field:

$$\hat{F}_{t \to t+1} = \text{FlowComplete}(F_{t \to t+1}, M_t, M_{t+1})$$

**Step 3: Propagate known pixels.**

Using the completed flow, propagate pixels from unmasked regions in neighboring frames into the masked region of the current frame. For each masked pixel $(i,j)$ at frame $t$, look for the nearest frame where that pixel's corresponding location (according to flow) is unmasked:

$$I'_t(i,j) = I_{t+\Delta t}\left(i + \sum_{s=t}^{t+\Delta t - 1} \hat{F}_{s \to s+1}^x(i,j), \; j + \sum_{s=t}^{t+\Delta t - 1} \hat{F}_{s \to s+1}^y(i,j)\right)$$

This chained flow propagation can fill many masked pixels, especially for object removal where the background is visible in other frames.

**Step 4: Inpaint remaining holes.**

After flow-based propagation, some pixels may still be missing (they were masked in all frames, or accumulated flow errors make the propagation unreliable). A transformer-based inpainting module fills these remaining holes using spatiotemporal attention.

**Step 5: Feature propagation refinement.**

Rather than propagating raw pixels (which can accumulate color drift), ProPainter propagates features from an intermediate layer of the inpainting network, allowing the decoder to produce harmonized output.

<svg viewBox="0 0 800 450" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 800px; display: block; margin: 2em auto;">
  <text x="400" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#333">Flow-Guided Video Inpainting Pipeline (ProPainter)</text>
  <defs>
    <marker id="flow-arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
    <marker id="flow-arrow-blue" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#4fc3f7"/>
    </marker>
  </defs>
  <!-- Step 1: Input -->
  <rect x="30" y="50" width="120" height="60" rx="6" fill="#e3f2fd" stroke="#4fc3f7" stroke-width="2"/>
  <text x="90" y="75" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#333" font-weight="bold">Input Video</text>
  <text x="90" y="90" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#666">V + Masks M</text>
  <!-- Arrow -->
  <line x1="150" y1="80" x2="180" y2="80" stroke="#333" stroke-width="1.5" marker-end="url(#flow-arrow)"/>
  <!-- Step 2: Flow Computation -->
  <rect x="185" y="50" width="120" height="60" rx="6" fill="#fff3e0" stroke="#ffa726" stroke-width="2"/>
  <text x="245" y="72" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333" font-weight="bold">Optical Flow</text>
  <text x="245" y="85" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#666">RAFT / GMFlow</text>
  <text x="245" y="98" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#ffa726">F_t->t+1</text>
  <!-- Arrow -->
  <line x1="305" y1="80" x2="335" y2="80" stroke="#333" stroke-width="1.5" marker-end="url(#flow-arrow)"/>
  <!-- Step 3: Flow Completion -->
  <rect x="340" y="50" width="130" height="60" rx="6" fill="#fce4ec" stroke="#ef5350" stroke-width="2"/>
  <text x="405" y="72" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333" font-weight="bold">Flow Completion</text>
  <text x="405" y="85" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#666">Hallucinate flow</text>
  <text x="405" y="98" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#ef5350">in masked regions</text>
  <!-- Arrow down -->
  <line x1="405" y1="110" x2="405" y2="150" stroke="#333" stroke-width="1.5" marker-end="url(#flow-arrow)"/>
  <!-- Step 4: Pixel Propagation -->
  <rect x="340" y="155" width="130" height="60" rx="6" fill="#e8f5e9" stroke="#8bc34a" stroke-width="2"/>
  <text x="405" y="177" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333" font-weight="bold">Pixel Propagation</text>
  <text x="405" y="190" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#666">Warp known pixels</text>
  <text x="405" y="203" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#8bc34a">into masked regions</text>
  <!-- Arrow down -->
  <line x1="405" y1="215" x2="405" y2="255" stroke="#333" stroke-width="1.5" marker-end="url(#flow-arrow)"/>
  <!-- Step 5: Transformer Inpaint -->
  <rect x="330" y="260" width="150" height="60" rx="6" fill="#e3f2fd" stroke="#4fc3f7" stroke-width="2"/>
  <text x="405" y="282" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333" font-weight="bold">Transformer Inpaint</text>
  <text x="405" y="295" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#666">Fill remaining holes</text>
  <text x="405" y="308" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#4fc3f7">Spatiotemporal attention</text>
  <!-- Arrow down -->
  <line x1="405" y1="320" x2="405" y2="360" stroke="#333" stroke-width="1.5" marker-end="url(#flow-arrow)"/>
  <!-- Step 6: Output -->
  <rect x="340" y="365" width="130" height="55" rx="6" fill="#f1f8e9" stroke="#8bc34a" stroke-width="2.5"/>
  <text x="405" y="388" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#333" font-weight="bold">Output Video V'</text>
  <text x="405" y="403" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#8bc34a">Temporally consistent</text>
  <!-- Side annotations -->
  <rect x="560" y="65" width="210" height="120" rx="6" fill="#fff8e1" stroke="#ffa726" stroke-width="1"/>
  <text x="665" y="85" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="#ffa726">Why Flow-Guided Works</text>
  <text x="575" y="105" font-family="Arial, sans-serif" font-size="9" fill="#333">1. Exploits temporal redundancy</text>
  <text x="575" y="120" font-family="Arial, sans-serif" font-size="9" fill="#333">2. Background often visible in</text>
  <text x="575" y="132" font-family="Arial, sans-serif" font-size="9" fill="#333">   other frames</text>
  <text x="575" y="147" font-family="Arial, sans-serif" font-size="9" fill="#333">3. Flow ensures natural motion</text>
  <text x="575" y="162" font-family="Arial, sans-serif" font-size="9" fill="#333">4. Inpainting only for truly</text>
  <text x="575" y="174" font-family="Arial, sans-serif" font-size="9" fill="#333">   novel content</text>
</svg>

### Why Flow-Guided Propagation Is Superior for Object Removal

Consider removing a person walking across a scene. In each frame, the person occludes a different portion of the background. But that background is visible in other frames (before the person walked there, or after they passed). Flow-guided propagation:

1. Identifies which background pixels are visible in which frames
2. Warps those pixels to the current frame using accurate optical flow
3. Only needs to hallucinate content for permanently occluded regions (e.g., where the person is standing throughout the entire clip)

This is dramatically more efficient and more accurate than generating all masked pixels from scratch. The propagated pixels are real pixels from the video --- they have the correct texture, lighting, and color. Only the small remaining holes need to be generated.

---

## Mask Generation: Manual, SAM, and Text-Prompted

The quality of inpainting depends critically on the quality of the mask. A mask that is too tight will leave artifacts at object edges. A mask that is too large will require unnecessary regeneration of content that was fine.

### Manual Masking

The user draws a mask on each frame (or on keyframes, with interpolation). This gives the most control but is labor-intensive. Practical for:
- Correcting a specific artifact in a known location
- Fine-tuning a small region

Tools: Runway's built-in masking, After Effects roto brush, DaVinci Resolve.

### SAM (Segment Anything Model)

Meta's Segment Anything Model can segment any object in an image given a point click, bounding box, or coarse mask. For video, SAM 2 extends this to track segments across frames.

The pipeline:
1. User clicks on the object in one frame
2. SAM generates a precise segmentation mask for that frame
3. SAM 2's memory mechanism propagates the mask across all frames, tracking the object through motion and deformation

```python
from sam2.build_sam import build_sam2_video_predictor

predictor = build_sam2_video_predictor(
    config_file="sam2_hiera_l.yaml",
    ckpt_path="sam2_hiera_large.pt"
)

# Initialize with video
inference_state = predictor.init_state(video_path="input_video.mp4")

# User provides a click on frame 0
frame_idx = 0
points = np.array([[x, y]])  # click coordinates
labels = np.array([1])       # 1 = foreground

# Add the prompt
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=frame_idx,
    obj_id=1,
    points=points,
    labels=labels,
)

# Propagate across entire video
video_segments = {}
for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
    masks = (mask_logits > 0.0).cpu().numpy()
    video_segments[frame_idx] = masks
```

### Text-Prompted Masking: GroundingDINO + SAM

The most automated approach: describe the object to remove in text, and the system generates the mask automatically.

1. **GroundingDINO** takes the text prompt (e.g., "the red car") and detects bounding boxes around matching objects in each frame
2. **SAM** refines each bounding box into a precise segmentation mask
3. **Temporal tracking** (SAM 2 or simple IoU matching) links masks across frames

```python
from groundingdino.util.inference import load_model, predict
from segment_anything import sam_model_registry, SamPredictor

# Load models
grounding_model = load_model("groundingdino_swinb_cogcoor.pth")
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam_predictor = SamPredictor(sam)

def get_mask_for_frame(frame, text_prompt, box_threshold=0.3):
    """Generate mask for a text-described object in a single frame."""
    # Step 1: Detect with GroundingDINO
    boxes, logits, phrases = predict(
        model=grounding_model,
        image=frame,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=0.25
    )

    # Step 2: Refine with SAM
    sam_predictor.set_image(frame)
    masks, scores, _ = sam_predictor.predict(
        box=boxes[0],  # best detection
        multimask_output=True
    )

    # Return highest-confidence mask
    best_mask = masks[scores.argmax()]
    return best_mask
```

<svg viewBox="0 0 800 300" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 800px; display: block; margin: 2em auto;">
  <text x="400" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#333">Mask Generation Methods</text>
  <defs>
    <marker id="mask-arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>
  <!-- Manual -->
  <rect x="30" y="50" width="220" height="100" rx="8" fill="#fce4ec" stroke="#ef5350" stroke-width="2"/>
  <text x="140" y="72" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#ef5350">Manual Masking</text>
  <text x="140" y="92" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333">User draws on each frame</text>
  <text x="140" y="107" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333">or keyframes + interpolation</text>
  <text x="50" y="130" font-family="Arial, sans-serif" font-size="9" fill="#666">Precision: High | Effort: High</text>
  <text x="50" y="142" font-family="Arial, sans-serif" font-size="9" fill="#666">Best for: Small corrections</text>
  <!-- SAM -->
  <rect x="280" y="50" width="220" height="100" rx="8" fill="#e3f2fd" stroke="#4fc3f7" stroke-width="2"/>
  <text x="390" y="72" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#4fc3f7">SAM 2 (Click)</text>
  <text x="390" y="92" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333">User clicks object once</text>
  <text x="390" y="107" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333">SAM 2 tracks across video</text>
  <text x="300" y="130" font-family="Arial, sans-serif" font-size="9" fill="#666">Precision: High | Effort: Low</text>
  <text x="300" y="142" font-family="Arial, sans-serif" font-size="9" fill="#666">Best for: Object removal</text>
  <!-- GroundingDINO + SAM -->
  <rect x="530" y="50" width="240" height="100" rx="8" fill="#e8f5e9" stroke="#8bc34a" stroke-width="2"/>
  <text x="650" y="72" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#8bc34a">Text-Prompted (GDINO+SAM)</text>
  <text x="650" y="92" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333">"Remove the red car"</text>
  <text x="650" y="107" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333">Fully automatic detection + mask</text>
  <text x="550" y="130" font-family="Arial, sans-serif" font-size="9" fill="#666">Precision: Medium | Effort: Minimal</text>
  <text x="550" y="142" font-family="Arial, sans-serif" font-size="9" fill="#666">Best for: Batch processing</text>
  <!-- Common output -->
  <line x1="140" y1="150" x2="140" y2="190" stroke="#333" stroke-width="1" marker-end="url(#mask-arrow)"/>
  <line x1="390" y1="150" x2="390" y2="190" stroke="#333" stroke-width="1" marker-end="url(#mask-arrow)"/>
  <line x1="650" y1="150" x2="650" y2="190" stroke="#333" stroke-width="1" marker-end="url(#mask-arrow)"/>
  <!-- Mask dilation step -->
  <rect x="100" y="195" width="580" height="50" rx="8" fill="#fff3e0" stroke="#ffa726" stroke-width="2"/>
  <text x="390" y="218" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#ffa726">Mask Post-Processing</text>
  <text x="390" y="235" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333">Dilate (expand by 5-15px) + Gaussian blur edges + Temporal smoothing</text>
  <!-- Output -->
  <line x1="390" y1="245" x2="390" y2="270" stroke="#333" stroke-width="1.5" marker-end="url(#mask-arrow)"/>
  <text x="390" y="290" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#333">Final Mask Sequence M = {M_1, M_2, ..., M_T}</text>
</svg>

### Mask Post-Processing

Regardless of how the mask is generated, post-processing improves inpainting quality:

1. **Dilation**: Expand the mask by 5--15 pixels to ensure the inpainting model covers any edge artifacts. Morphological dilation with a circular kernel:

$$M' = M \oplus B_r$$

where $B_r$ is a disk structuring element of radius $r$.

2. **Gaussian blur**: Apply a Gaussian blur to the dilated mask edges to create a soft transition zone:

$$M'' = G_\sigma * M'$$

where $G_\sigma$ is a Gaussian kernel with standard deviation $\sigma$, typically 3--7 pixels.

3. **Temporal smoothing**: Ensure the mask does not jitter frame-to-frame. Apply a 1D Gaussian filter along the temporal axis for each pixel:

$$M_t''(i,j) = \frac{\sum_{s=t-k}^{t+k} G_\sigma(s-t) \cdot M_s'(i,j)}{\sum_{s=t-k}^{t+k} G_\sigma(s-t)}$$

---

## The Complete Inpainting Pipeline

Here is the end-to-end pipeline for a production video inpainting system:

<svg viewBox="0 0 800 550" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 800px; display: block; margin: 2em auto;">
  <text x="400" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#333">Production Video Inpainting Pipeline</text>
  <defs>
    <marker id="pipe-arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>
  <!-- Step 1 -->
  <rect x="280" y="45" width="240" height="45" rx="8" fill="#e3f2fd" stroke="#4fc3f7" stroke-width="2"/>
  <text x="400" y="65" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#333">1. Generate or Receive Video</text>
  <text x="400" y="80" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#666">AI-generated or uploaded input V</text>
  <line x1="400" y1="90" x2="400" y2="110" stroke="#333" stroke-width="1.5" marker-end="url(#pipe-arrow)"/>
  <!-- Step 2 -->
  <rect x="280" y="115" width="240" height="45" rx="8" fill="#fff3e0" stroke="#ffa726" stroke-width="2"/>
  <text x="400" y="135" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#333">2. Identify Issue Region</text>
  <text x="400" y="150" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#666">User feedback or automated QA detection</text>
  <line x1="400" y1="160" x2="400" y2="180" stroke="#333" stroke-width="1.5" marker-end="url(#pipe-arrow)"/>
  <!-- Step 3 -->
  <rect x="280" y="185" width="240" height="45" rx="8" fill="#fce4ec" stroke="#ef5350" stroke-width="2"/>
  <text x="400" y="205" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#333">3. Generate Mask Sequence</text>
  <text x="400" y="220" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#666">SAM 2 / GDINO+SAM / manual</text>
  <line x1="400" y1="230" x2="400" y2="250" stroke="#333" stroke-width="1.5" marker-end="url(#pipe-arrow)"/>
  <!-- Step 4 -->
  <rect x="280" y="255" width="240" height="45" rx="8" fill="#f3e5f5" stroke="#ab47bc" stroke-width="2"/>
  <text x="400" y="275" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#333">4. Mask Post-Processing</text>
  <text x="400" y="290" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#666">Dilate + blur edges + temporal smooth</text>
  <line x1="400" y1="300" x2="400" y2="320" stroke="#333" stroke-width="1.5" marker-end="url(#pipe-arrow)"/>
  <!-- Step 5 -->
  <rect x="250" y="325" width="300" height="55" rx="8" fill="#e8f5e9" stroke="#8bc34a" stroke-width="2.5"/>
  <text x="400" y="345" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#333">5. Video Inpainting</text>
  <text x="400" y="358" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#666">Flow-guided propagation + diffusion fill</text>
  <text x="400" y="371" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#8bc34a">ProPainter / Video diffusion inpainting</text>
  <line x1="400" y1="380" x2="400" y2="400" stroke="#333" stroke-width="1.5" marker-end="url(#pipe-arrow)"/>
  <!-- Step 6 -->
  <rect x="280" y="405" width="240" height="45" rx="8" fill="#e3f2fd" stroke="#4fc3f7" stroke-width="2"/>
  <text x="400" y="425" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#333">6. Composite and Blend</text>
  <text x="400" y="440" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#666">Merge inpainted region with original</text>
  <line x1="400" y1="450" x2="400" y2="470" stroke="#333" stroke-width="1.5" marker-end="url(#pipe-arrow)"/>
  <!-- Step 7 -->
  <rect x="280" y="475" width="240" height="45" rx="8" fill="#fff3e0" stroke="#ffa726" stroke-width="2"/>
  <text x="400" y="495" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#333">7. Quality Evaluation</text>
  <text x="400" y="510" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#666">Temporal consistency + boundary check</text>
  <!-- Side annotation: optional prompt -->
  <rect x="560" y="325" width="200" height="55" rx="6" fill="#fff8e1" stroke="#ffa726" stroke-width="1" stroke-dasharray="4"/>
  <text x="660" y="345" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#ffa726">Optional: Text prompt c</text>
  <text x="660" y="360" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#666">"Replace with a bicycle"</text>
  <text x="660" y="373" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#666">"Fill with grass texture"</text>
  <line x1="560" y1="352" x2="550" y2="352" stroke="#ffa726" stroke-width="1" stroke-dasharray="3" marker-end="url(#pipe-arrow)"/>
</svg>

### Step 6: Compositing and Blending

The final composite combines the inpainted region with the original video. Using the (potentially soft) mask:

$$I'_\text{final}(i,j) = M''(i,j) \cdot I'_\text{inpainted}(i,j) + (1 - M''(i,j)) \cdot I_\text{original}(i,j)$$

where $M''$ is the blurred soft mask. The soft edges create a smooth gradient transition rather than a hard boundary.

For additional boundary harmonization, apply Poisson blending (Perez et al., 2003), which matches the gradients at the boundary while allowing the color to shift:

$$\arg\min_{I'} \sum_{(i,j) \in M} \|\nabla I'(i,j) - \nabla I_\text{inpainted}(i,j)\|^2 \quad \text{s.t.} \quad I'|_{\partial M} = I_\text{original}|_{\partial M}$$

This is a Laplacian equation and can be solved efficiently using a sparse linear system.

---

## Practical Tools and Comparison

Here is a comparison of available video inpainting tools and models as of late 2025:

| Tool / Model | Approach | Temporal Consistency | Speed | Open Source | Best For |
|--------------|----------|---------------------|-------|-------------|----------|
| **ProPainter** | Flow-guided + transformer | Excellent | ~3 fps (A100) | Yes | Object removal |
| **Runway Inpainting** | Proprietary diffusion | Very good | Real-time (API) | No | General editing |
| **Stable Diffusion Inpainting** | Per-frame diffusion | Poor (per-frame) | ~10 fps (A100) | Yes | Static scenes |
| **E2FGVI** | Flow-guided + propagation | Good | ~5 fps (A100) | Yes | Background fill |
| **STTN** | Spatiotemporal transformer | Good | ~4 fps (A100) | Yes | Medium holes |
| **CopyCat** | Copy-paste + harmonize | Moderate | ~8 fps | Yes | Object replacement |
| **Video-P2P** | Attention editing | Good | Slow (~0.5 fps) | Yes | Prompt-guided edits |

### RunwayML Inpainting

Runway's Gen-2 and Gen-3 models include native inpainting capabilities through their web interface and API. The user draws a mask, optionally provides a prompt, and the model generates the replacement content with temporal consistency.

Strengths:
- Tight integration with video generation (the model understands video natively)
- Good temporal consistency from joint spatiotemporal processing
- Easy-to-use interface for non-technical users

Weaknesses:
- Proprietary, API-only access
- Cost per generation can be significant for iterative editing
- Limited control over the inpainting process

### ProPainter (Open Source)

ProPainter is the state-of-the-art open-source solution for video inpainting, particularly strong for object removal. It uses the three-stage flow-guided approach described above.

```python
# ProPainter inference
import torch
from propainter.model import ProPainter
from propainter.utils import read_video, write_video, read_masks

# Load model
model = ProPainter()
model.load_state_dict(torch.load("propainter.pth"))
model.eval().cuda()

# Load video and masks
frames = read_video("input.mp4")       # [T, H, W, 3]
masks = read_masks("masks/")           # [T, H, W]

# Run inpainting
with torch.no_grad():
    result = model.inpaint(
        frames=frames,
        masks=masks,
        flow_model="raft",
        num_flow_complete_iters=20,
        subvideo_length=80,           # process in chunks
        neighbor_length=10,            # temporal context
    )

write_video(result, "output.mp4", fps=30)
```

---

## Use Cases for Video Platforms

### Object Removal

The most common use case. Examples:
- Remove watermarks from AI-generated preview clips
- Remove unwanted background elements (passersby, cars, signs)
- Clean up generation artifacts in specific regions

For watermark removal specifically, the mask is known a priori (the watermark location is fixed), making this a straightforward pipeline.

### Object Replacement

Replace one object with another while preserving the rest of the scene:

1. Generate mask for the target object (SAM 2)
2. Inpaint the region with a text prompt describing the replacement
3. The diffusion model generates the replacement object to fit the spatial extent of the mask

Example: "Replace the sedan with a pickup truck." The inpainting model generates a pickup truck that fills the sedan's mask region, matching the perspective, lighting, and motion of the original.

### Background Replacement

Keep the foreground subject, replace the entire background:

1. Segment the foreground subject (SAM 2 or matting model)
2. Invert the mask: everything except the subject is the inpainting region
3. Inpaint with a prompt describing the desired background

This is particularly powerful for AI video platforms where the subject was generated correctly but the background needs to change. Instead of regenerating the entire video (losing the subject), you inpaint only the background.

### Error Correction

AI-generated videos frequently have localized artifacts:
- A hand with six fingers in frames 15--25
- A flickering texture in the upper-left corner
- A color shift in a specific object for 10 frames

Inpainting can fix these surgically:

1. Detect the artifact region (manually or with an automated quality checker)
2. Generate a mask covering the artifact with some margin
3. Inpaint just that region, guided by the surrounding context
4. The fix is pixel-identical to the original everywhere outside the mask

This is dramatically cheaper and faster than regenerating the entire clip, and it preserves all the content you already approved.

---

## Quality Evaluation Metrics

How do you know if the inpainting succeeded? There are three dimensions to evaluate:

### 1. Temporal Consistency: Warping Error

The most important metric for video inpainting. Compute the optical flow between consecutive frames, warp frame $t$ to align with frame $t+1$, and measure the difference in the inpainted region:

$$E_\text{warp} = \frac{1}{|\Omega|} \sum_{t=1}^{T-1} \sum_{(i,j) \in \Omega_t} \|I'_{t+1}(i,j) - \text{warp}(I'_t, F_{t \to t+1})(i,j)\|^2$$

where $\Omega_t = M_t \cap M_{t+1}$ is the set of pixels that are inpainted in both frames, and $|\Omega| = \sum_t |\Omega_t|$ is the total count.

Lower warping error = better temporal consistency.

### 2. Visual Quality: FID and LPIPS

**FID (Frechet Inception Distance)** on the inpainted region measures whether the generated content has the same distribution as real content:

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

where $(\mu_r, \Sigma_r)$ and $(\mu_g, \Sigma_g)$ are the mean and covariance of Inception features from real and generated crops, respectively.

**LPIPS (Learned Perceptual Image Patch Similarity)** measures perceptual difference between the inpainted result and a reference (if available):

$$\text{LPIPS}(x, x') = \sum_l \frac{1}{H_l W_l} \sum_{i,j} \|w_l \odot (\hat{y}_{l}^{x}(i,j) - \hat{y}_{l}^{x'}(i,j))\|^2$$

where $\hat{y}_l^x$ are normalized activations from layer $l$ of a pretrained VGG network, and $w_l$ are learned weights.

### 3. Boundary Quality: Gradient Analysis

Check for visible seams at the mask boundary. Compute the gradient magnitude at the mask edge and compare it to the expected gradient from the surrounding region:

$$E_\text{boundary} = \frac{1}{|\partial M|} \sum_{(i,j) \in \partial M} \left| \|\nabla I'(i,j)\| - \mathbb{E}[\|\nabla I(i,j)\| \mid (i,j) \in \mathcal{N}(\partial M)] \right|$$

where $\mathcal{N}(\partial M)$ is a neighborhood of the mask boundary in the unmasked region. A high boundary error indicates visible seams; a low value indicates smooth blending.

### Combined Quality Score

For a production system, combine the metrics into a single quality gate:

$$Q = w_1 \cdot (1 - \hat{E}_\text{warp}) + w_2 \cdot (1 - \hat{\text{FID}}) + w_3 \cdot (1 - \hat{E}_\text{boundary})$$

where $\hat{\cdot}$ denotes normalization to $[0, 1]$ and $w_1 + w_2 + w_3 = 1$. Typical weights: $w_1 = 0.5$ (temporal consistency matters most), $w_2 = 0.3$ (visual quality), $w_3 = 0.2$ (boundary artifacts).

---

## Implementation: Full Code Pipeline

Here is a complete implementation of a video inpainting pipeline using ProPainter for the core inpainting and SAM 2 for mask generation:

```python
"""
Complete video inpainting pipeline.
Dependencies: torch, opencv-python, numpy, sam2, propainter
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Tuple


class VideoInpaintingPipeline:
    """End-to-end video inpainting: mask generation -> inpainting -> compositing."""

    def __init__(
        self,
        sam2_config: str = "sam2_hiera_l.yaml",
        sam2_checkpoint: str = "sam2_hiera_large.pt",
        propainter_checkpoint: str = "propainter.pth",
        device: str = "cuda",
    ):
        self.device = device
        self._load_sam2(sam2_config, sam2_checkpoint)
        self._load_propainter(propainter_checkpoint)

    def _load_sam2(self, config: str, checkpoint: str):
        from sam2.build_sam import build_sam2_video_predictor
        self.sam_predictor = build_sam2_video_predictor(config, checkpoint)

    def _load_propainter(self, checkpoint: str):
        from propainter.model import ProPainter
        self.inpainter = ProPainter()
        self.inpainter.load_state_dict(torch.load(checkpoint))
        self.inpainter.eval().to(self.device)

    def read_video(self, path: str) -> Tuple[np.ndarray, int]:
        """Read video to numpy array [T, H, W, 3] and return fps."""
        cap = cv2.VideoCapture(path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.stack(frames), fps

    def generate_mask_from_click(
        self,
        video_path: str,
        frame_idx: int,
        point: Tuple[int, int],
        foreground: bool = True,
    ) -> np.ndarray:
        """Generate mask sequence from a single click using SAM 2."""
        inference_state = self.sam_predictor.init_state(video_path=video_path)

        points = np.array([point])
        labels = np.array([1 if foreground else 0])

        self.sam_predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=1,
            points=points,
            labels=labels,
        )

        masks = {}
        for fid, obj_ids, mask_logits in self.sam_predictor.propagate_in_video(
            inference_state
        ):
            masks[fid] = (mask_logits[0] > 0.0).cpu().numpy().squeeze()

        # Convert dict to ordered array
        T = max(masks.keys()) + 1
        mask_array = np.zeros((T,) + masks[0].shape, dtype=np.uint8)
        for fid, mask in masks.items():
            mask_array[fid] = mask.astype(np.uint8)

        return mask_array

    def postprocess_masks(
        self,
        masks: np.ndarray,
        dilate_radius: int = 10,
        blur_sigma: float = 5.0,
        temporal_sigma: float = 1.0,
    ) -> np.ndarray:
        """Dilate, blur edges, and temporally smooth mask sequence."""
        T, H, W = masks.shape
        processed = np.zeros((T, H, W), dtype=np.float32)

        # Step 1: Dilate each frame
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_radius * 2 + 1, dilate_radius * 2 + 1)
        )
        for t in range(T):
            dilated = cv2.dilate(masks[t].astype(np.uint8), kernel, iterations=1)
            # Step 2: Gaussian blur the edges
            blurred = cv2.GaussianBlur(
                dilated.astype(np.float32),
                (0, 0),
                sigmaX=blur_sigma,
            )
            processed[t] = blurred

        # Step 3: Temporal smoothing
        if temporal_sigma > 0 and T > 1:
            from scipy.ndimage import gaussian_filter1d
            processed = gaussian_filter1d(processed, sigma=temporal_sigma, axis=0)

        return processed

    def inpaint(
        self,
        frames: np.ndarray,
        masks: np.ndarray,
        prompt: Optional[str] = None,
        subvideo_length: int = 80,
    ) -> np.ndarray:
        """Run inpainting on masked video regions."""
        with torch.no_grad():
            result = self.inpainter.inpaint(
                frames=torch.from_numpy(frames).to(self.device),
                masks=torch.from_numpy(masks).to(self.device),
                subvideo_length=subvideo_length,
            )
        return result.cpu().numpy()

    def composite(
        self,
        original: np.ndarray,
        inpainted: np.ndarray,
        soft_masks: np.ndarray,
    ) -> np.ndarray:
        """Blend inpainted region with original using soft mask."""
        # Expand mask to 3 channels
        masks_3c = soft_masks[..., np.newaxis]  # [T, H, W, 1]

        # Alpha blend
        result = masks_3c * inpainted + (1 - masks_3c) * original
        return result.astype(np.uint8)

    def evaluate_temporal_consistency(
        self,
        result: np.ndarray,
        masks: np.ndarray,
    ) -> float:
        """Compute warping error as a temporal consistency metric."""
        T, H, W, C = result.shape
        total_error = 0.0
        total_pixels = 0

        for t in range(T - 1):
            # Compute optical flow
            gray_t = cv2.cvtColor(result[t], cv2.COLOR_RGB2GRAY)
            gray_t1 = cv2.cvtColor(result[t + 1], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                gray_t, gray_t1, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )

            # Warp frame t to t+1
            h, w = flow.shape[:2]
            flow_map = np.column_stack((
                (np.arange(w)[np.newaxis, :] + flow[..., 0]).ravel(),
                (np.arange(h)[:, np.newaxis] + flow[..., 1]).ravel(),
            )).reshape(h, w, 2).astype(np.float32)

            warped = cv2.remap(
                result[t], flow_map[..., 0], flow_map[..., 1],
                cv2.INTER_LINEAR,
            )

            # Compute error in masked region
            mask_both = (masks[t] > 0.5) & (masks[t + 1] > 0.5)
            if mask_both.sum() > 0:
                error = np.mean(
                    (result[t + 1][mask_both].astype(float)
                     - warped[mask_both].astype(float)) ** 2
                )
                total_error += error * mask_both.sum()
                total_pixels += mask_both.sum()

        return total_error / max(total_pixels, 1)

    def run(
        self,
        video_path: str,
        output_path: str,
        frame_idx: int,
        click_point: Tuple[int, int],
        prompt: Optional[str] = None,
    ) -> dict:
        """Full pipeline: mask -> inpaint -> composite -> evaluate."""
        # Read video
        frames, fps = self.read_video(video_path)

        # Generate masks
        binary_masks = self.generate_mask_from_click(
            video_path, frame_idx, click_point
        )

        # Post-process masks
        soft_masks = self.postprocess_masks(binary_masks)

        # Inpaint
        inpainted = self.inpaint(frames, (soft_masks > 0.5).astype(np.uint8), prompt)

        # Composite
        result = self.composite(frames, inpainted, soft_masks)

        # Evaluate
        warp_error = self.evaluate_temporal_consistency(result, soft_masks)

        # Write output
        self.write_video(result, output_path, fps)

        return {
            "warp_error": warp_error,
            "frames_processed": len(frames),
            "mask_coverage": float(soft_masks.mean()),
        }

    def write_video(
        self, frames: np.ndarray, path: str, fps: int
    ):
        """Write numpy array [T, H, W, 3] to video file."""
        T, H, W, C = frames.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
        for t in range(T):
            writer.write(cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR))
        writer.release()


# Usage example
if __name__ == "__main__":
    pipeline = VideoInpaintingPipeline()

    metrics = pipeline.run(
        video_path="generated_video.mp4",
        output_path="inpainted_video.mp4",
        frame_idx=0,
        click_point=(350, 200),   # click on object to remove
        prompt=None,               # None = remove, or text for replacement
    )

    print(f"Warping error: {metrics['warp_error']:.4f}")
    print(f"Frames processed: {metrics['frames_processed']}")
    print(f"Mask coverage: {metrics['mask_coverage']:.2%}")
```

### FFmpeg Post-Processing

After inpainting, you may want to re-encode the result with optimal settings:

```bash
# Re-encode inpainted video with high quality
ffmpeg -i inpainted_video.mp4 \
  -c:v libx264 -crf 18 -preset slow \
  -pix_fmt yuv420p \
  -movflags +faststart \
  output_final.mp4

# If the original had audio, merge it back
ffmpeg -i inpainted_video.mp4 -i original_with_audio.mp4 \
  -c:v libx264 -crf 18 -preset slow \
  -c:a aac -b:a 192k \
  -map 0:v:0 -map 1:a:0 \
  -movflags +faststart \
  output_with_audio.mp4
```

---

## Conclusion

Video inpainting is the surgical tool that makes AI video platforms practical. Without it, every imperfection requires a full regeneration --- expensive, stochastic, and wasteful. With it, you can fix exactly what needs fixing and preserve everything that works.

The key technical insights:

1. **Temporal consistency is the hard problem.** Per-frame inpainting is easy but produces flickering. Joint spatiotemporal inpainting or flow-guided propagation is necessary for production quality.

2. **Flow-guided propagation is the practical sweet spot.** It exploits temporal redundancy (the same background content is often visible in other frames), minimizes the amount of content that needs to be generated from scratch, and naturally inherits the motion patterns of the video.

3. **Mask quality determines inpainting quality.** Invest in good mask generation (SAM 2 for interactive, GroundingDINO+SAM for automated) and always post-process with dilation and edge blurring.

4. **Evaluation must include temporal metrics.** FID and LPIPS on individual frames are necessary but not sufficient. Warping error across frames catches the temporal inconsistencies that are invisible in still images but obvious in playback.

For builders: integrating video inpainting into your platform's editing workflow transforms user experience. Instead of "regenerate and hope," users get "fix this specific thing." That is the difference between a toy and a tool.
