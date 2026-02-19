---
layout: post
title: "Perceptual Loss and Neural Distance Metrics: Why Pixel Loss Fails and What Replaces It for Video"
date: 2026-02-26
category: math
---

Train a video super-resolution model with mean squared error loss. It will produce blurry output. Train a video prediction model with L1 loss. It will produce blurry output. Train any generative model that maps an input to a pixel-space target using any per-pixel regression loss. It will produce blurry output.

This is not a bug in the training procedure. It is a mathematical inevitability: when the mapping is one-to-many (many valid outputs exist for a given input), the MSE-optimal prediction is the **conditional mean** --- the average over all valid outputs. Averaging sharp images with slightly different edges, textures, and details produces a blurry image. The math is clean. The result is unacceptable.

**Perceptual losses** fix this by comparing images in a learned feature space rather than pixel space. Instead of asking "are the pixel values the same?", they ask "do these images have the same high-level structure?" --- as judged by the internal representations of a pretrained neural network. This post derives why pixel loss leads to blur, builds up SSIM and its multi-scale variant, explains perceptual loss and style loss via Gram matrices, derives LPIPS, connects adversarial losses to perceptual quality, and addresses the unique challenges of temporal perceptual losses for video.

---

## Table of Contents

1. [Why Pixel Loss Produces Blur](#why-pixel-loss-produces-blur)
2. [SSIM: Structural Similarity](#ssim-structural-similarity)
3. [Feature Space Distances](#feature-space-distances)
4. [Perceptual Loss](#perceptual-loss)
5. [Style Loss and Gram Matrices](#style-loss-and-gram-matrices)
6. [LPIPS: Learned Perceptual Image Patch Similarity](#lpips-learned-perceptual-image-patch-similarity)
7. [Adversarial Loss as Perceptual Metric](#adversarial-loss-as-perceptual-metric)
8. [The Perception-Distortion Tradeoff](#the-perception-distortion-tradeoff)
9. [Temporal Perceptual Losses for Video](#temporal-perceptual-losses-for-video)
10. [The Loss Landscape of Video Generation](#the-loss-landscape-of-video-generation)
11. [Python: Feature Space Distances and Perceptual Similarity](#python-feature-space-distances-and-perceptual-similarity)

---

## Why Pixel Loss Produces Blur

Consider a conditional generation problem: given input \(\mathbf{c}\), generate output \(\mathbf{x}\). There may be many valid outputs for a given input (the mapping is one-to-many). For example, given a low-resolution image, many high-resolution images are consistent with it.

### The MSE-Optimal Prediction

A model trained with MSE loss minimizes:

$$\mathcal{L}_{\text{MSE}} = \mathbb{E}_{\mathbf{x}, \mathbf{c}}\!\left[\|\hat{\mathbf{x}}(\mathbf{c}) - \mathbf{x}\|^2\right]$$

The solution that minimizes this over all possible functions \(\hat{\mathbf{x}}(\cdot)\) is the **conditional mean**:

$$\hat{\mathbf{x}}^*(\mathbf{c}) = \mathbb{E}[\mathbf{x} | \mathbf{c}]$$

**Proof.** For any fixed \(\mathbf{c}\), we want to find the \(\hat{\mathbf{x}}\) that minimizes \(\mathbb{E}[\|\hat{\mathbf{x}} - \mathbf{x}\|^2 | \mathbf{c}]\). Expand:

$$\mathbb{E}[\|\hat{\mathbf{x}} - \mathbf{x}\|^2 | \mathbf{c}] = \|\hat{\mathbf{x}}\|^2 - 2\hat{\mathbf{x}}^T \mathbb{E}[\mathbf{x}|\mathbf{c}] + \mathbb{E}[\|\mathbf{x}\|^2 | \mathbf{c}]$$

Taking the derivative with respect to \(\hat{\mathbf{x}}\) and setting to zero:

$$2\hat{\mathbf{x}} - 2\mathbb{E}[\mathbf{x}|\mathbf{c}] = 0 \implies \hat{\mathbf{x}}^* = \mathbb{E}[\mathbf{x} | \mathbf{c}]$$

### Why the Mean Is Blurry

If the conditional distribution \(p(\mathbf{x} | \mathbf{c})\) is multimodal --- several distinct sharp images are plausible given \(\mathbf{c}\) --- then the mean averages over these modes. Each mode has sharp edges at slightly different positions. The average smears these edges out.

For video, this is even worse. Given the first 10 frames, many plausible continuations exist (the person could turn left or right, the ball could bounce or roll). The conditional mean averages over all futures, producing a ghostly, blurry mess.

Similarly, L1 loss yields the conditional **median**, which is slightly better (less sensitive to outliers) but still blurry for multimodal distributions.

<svg viewBox="0 0 700 250" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">Regression to the Mean: Why MSE → Blur</text>

  <!-- Mode 1 -->
  <rect x="50" y="50" width="120" height="90" rx="3" fill="none" stroke="#4fc3f7" stroke-width="1.5"/>
  <text x="110" y="75" text-anchor="middle" font-size="10" fill="#4fc3f7">Mode 1 (sharp)</text>
  <line x1="70" y1="100" x2="150" y2="100" stroke="#4fc3f7" stroke-width="3"/>
  <text x="110" y="155" text-anchor="middle" font-size="9" fill="#999">Edge at y=100</text>

  <!-- Mode 2 -->
  <rect x="200" y="50" width="120" height="90" rx="3" fill="none" stroke="#66bb6a" stroke-width="1.5"/>
  <text x="260" y="75" text-anchor="middle" font-size="10" fill="#66bb6a">Mode 2 (sharp)</text>
  <line x1="220" y1="110" x2="300" y2="110" stroke="#66bb6a" stroke-width="3"/>
  <text x="260" y="155" text-anchor="middle" font-size="9" fill="#999">Edge at y=110</text>

  <!-- Plus sign -->
  <text x="180" y="100" text-anchor="middle" font-size="20" fill="#666">+</text>

  <!-- Arrow -->
  <text x="360" y="100" text-anchor="middle" font-size="20" fill="#FF9800">→</text>
  <text x="360" y="120" text-anchor="middle" font-size="9" fill="#FF9800">MSE mean</text>

  <!-- Blurry result -->
  <rect x="400" y="50" width="120" height="90" rx="3" fill="none" stroke="#E53935" stroke-width="1.5"/>
  <text x="460" y="75" text-anchor="middle" font-size="10" fill="#E53935">Mean (blurry)</text>
  <!-- Blurry edge represented by gradient -->
  <rect x="420" y="98" width="80" height="2" fill="#E53935" opacity="0.3"/>
  <rect x="420" y="100" width="80" height="2" fill="#E53935" opacity="0.4"/>
  <rect x="420" y="102" width="80" height="2" fill="#E53935" opacity="0.5"/>
  <rect x="420" y="104" width="80" height="2" fill="#E53935" opacity="0.5"/>
  <rect x="420" y="106" width="80" height="2" fill="#E53935" opacity="0.4"/>
  <rect x="420" y="108" width="80" height="2" fill="#E53935" opacity="0.3"/>
  <text x="460" y="155" text-anchor="middle" font-size="9" fill="#999">Averaged edge → blur</text>

  <text x="550" y="190" text-anchor="start" font-size="10" fill="#d4d4d4">E[x|c] averages over</text>
  <text x="550" y="205" text-anchor="start" font-size="10" fill="#d4d4d4">modes → always blurry</text>
</svg>

---

## SSIM: Structural Similarity

**SSIM** (Wang et al., 2004) was the first widely-used metric that goes beyond pixel-level comparison by measuring structural similarity.

For two image patches \(\mathbf{x}\) and \(\mathbf{y}\), SSIM computes three components:

**Luminance comparison:**
$$l(\mathbf{x}, \mathbf{y}) = \frac{2\mu_x \mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}$$

**Contrast comparison:**
$$c(\mathbf{x}, \mathbf{y}) = \frac{2\sigma_x \sigma_y + C_2}{\sigma_x^2 + \sigma_y^2 + C_2}$$

**Structure comparison:**
$$s(\mathbf{x}, \mathbf{y}) = \frac{\sigma_{xy} + C_3}{\sigma_x \sigma_y + C_3}$$

where \(\mu_x, \mu_y\) are means, \(\sigma_x, \sigma_y\) are standard deviations, \(\sigma_{xy}\) is the covariance, and \(C_1, C_2, C_3\) are small stabilization constants.

The full SSIM is their product:

$$\text{SSIM}(\mathbf{x}, \mathbf{y}) = l \cdot c \cdot s = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

SSIM ranges from -1 to 1 (1 = identical). As a loss, we use \(\mathcal{L}_{\text{D-SSIM}} = (1 - \text{SSIM})/2\).

**MS-SSIM** extends this to multiple scales by computing SSIM at different resolutions (via downsampling) and combining them, capturing both fine details and coarse structure.

SSIM is better than MSE because it separates luminance, contrast, and structure. But it is still computed locally (within small patches) and does not capture high-level semantic similarity.

---

## Feature Space Distances

The breakthrough insight (Dosovitskiy & Brox, 2016; Johnson et al., 2016): compare images in the **feature space of a pretrained CNN** rather than in pixel space.

A CNN trained on ImageNet (e.g., VGG-16) learns a hierarchy of representations:
- **Early layers** (conv1, conv2): edges, textures, colors
- **Middle layers** (conv3, conv4): parts, patterns, spatial structure
- **Late layers** (conv5): objects, scenes, semantic content

Let \(\phi_l(\mathbf{x})\) denote the feature map of image \(\mathbf{x}\) at layer \(l\). These features are high-dimensional tensors (channels × height × width) that encode image content at a particular level of abstraction.

The key property: two images that are **perceptually similar** (similar content, structure, and appearance) have **similar feature representations**, even if their pixel values differ significantly. A slightly shifted image, a differently lit image, or a slightly different crop of the same scene will have different pixels but nearly identical mid-level features.

---

## Perceptual Loss

**Perceptual loss** (Johnson et al., 2016) measures the distance between images in feature space:

$$\mathcal{L}_{\text{perceptual}} = \sum_l w_l \|\phi_l(\hat{\mathbf{x}}) - \phi_l(\mathbf{x})\|_2^2$$

where \(\phi_l\) is the feature extractor at layer \(l\) of a pretrained VGG network, and \(w_l\) are per-layer weights.

Typical layer choices for VGG-19: relu1_1, relu2_1, relu3_1, relu4_1, relu5_1. Each layer contributes a different level of perceptual comparison.

**Why this works:**

1. Features are **locally invariant** to small perturbations. A one-pixel shift produces very different MSE but nearly identical features.

2. Features capture **semantics**. Two images of the same face from slightly different angles have similar features but very different pixels.

3. The loss gradient in pixel space from a perceptual loss encourages the model to match features, not pixels. This allows the model to produce sharp outputs that match the target's structure without needing exact pixel correspondence.

The perceptual loss was originally developed for style transfer and super-resolution, but it is now a standard component of autoencoder training (including the VAEs used in video diffusion).

---

## Style Loss and Gram Matrices

**Style loss** (Gatys et al., 2015) captures the **texture** and **style** of an image by comparing the statistics of feature maps, rather than the feature maps themselves.

### The Gram Matrix

For a feature map \(\phi_l(\mathbf{x}) \in \mathbb{R}^{C_l \times H_l \times W_l}\), reshape it to \(F \in \mathbb{R}^{C_l \times (H_l W_l)}\) (each row is a channel's spatial activation). The **Gram matrix** is:

$$G_l = \frac{1}{C_l H_l W_l} F F^T \in \mathbb{R}^{C_l \times C_l}$$

The entry \(G_l[i, j]\) is the inner product (correlation) between channels \(i\) and \(j\). This captures **which features co-occur spatially**: if channels for "horizontal edges" and "blue color" are correlated, the Gram matrix records this, indicating a texture with blue horizontal edges.

### Why Gram Matrices Capture Style

The Gram matrix computes the **second-order statistics** of the feature distribution. It is equivalent to the uncentered covariance of the feature vectors across spatial positions:

$$G_l[i, j] = \frac{1}{N_l}\sum_{k=1}^{N_l} \phi_l^{(i)}(x_k) \cdot \phi_l^{(j)}(x_k)$$

where \(N_l = H_l W_l\) and \(\phi_l^{(i)}\) is the \(i\)-th channel.

This discards spatial arrangement (where features are) but preserves texture patterns (which features co-occur). A field of grass and a close-up of a carpet might have different spatial layouts but similar Gram matrices if they share the same texture statistics.

**Style loss:**

$$\mathcal{L}_{\text{style}} = \sum_l w_l \|G_l(\hat{\mathbf{x}}) - G_l(\mathbf{x}_{\text{style}})\|_F^2$$

where \(\|\cdot\|_F\) is the Frobenius norm.

**Neural style transfer** (Gatys et al., 2015) combines content loss (perceptual loss matching a content image) with style loss (Gram matrix matching a style image) to synthesize an image with the content of one image and the style of another.

---

## LPIPS: Learned Perceptual Image Patch Similarity

**LPIPS** (Zhang et al., 2018) refines perceptual loss by learning which features matter for human perception.

### Architecture

Given two images \(\mathbf{x}, \hat{\mathbf{x}}\):

1. Extract features from a pretrained backbone (AlexNet, VGG, or SqueezeNet) at multiple layers
2. Normalize features channel-wise (unit variance per channel)
3. Compute squared differences: \(d_l = (\hat{\phi}_l(\mathbf{x}) - \hat{\phi}_l(\hat{\mathbf{x}}))^2\)
4. Apply **learned linear weights** \(w_l \in \mathbb{R}^{C_l}\) per channel: \(\text{LPIPS} = \sum_l \sum_c w_l^{(c)} \cdot \text{mean}(d_l^{(c)})\)

The key innovation: the channel weights \(w_l\) are trained on a dataset of **human perceptual judgments** (2AFC --- two-alternative forced choice: given a reference, which of two distortions is more similar?).

### Why Learned Weights Matter

Not all channels in a VGG feature map contribute equally to perceptual similarity. Some channels encode background texture that humans ignore; others encode salient edges that humans notice immediately. LPIPS learns to weight channels according to human importance.

LPIPS with a VGG backbone is now the standard perceptual metric for evaluating image and video generation quality. Lower LPIPS = more perceptually similar.

---

## Adversarial Loss as Perceptual Metric

A **discriminator** in a GAN can be viewed as a learned perceptual metric. The adversarial loss pushes generated images toward the manifold of real images, which inherently encourages perceptual realism.

### The PatchGAN

Instead of a single real/fake classification per image, a **PatchGAN** discriminator outputs a grid of real/fake predictions, each covering a receptive field of \(70 \times 70\) pixels (typically). This enforces realism at the texture level.

The adversarial loss:

$$\mathcal{L}_{\text{adv}} = -\mathbb{E}\!\left[\log D(\hat{\mathbf{x}})\right]$$

pushes the generator to produce images that the discriminator cannot distinguish from real. This implicitly encourages high-frequency detail, sharp edges, and realistic textures --- exactly what pixel loss suppresses.

### Multi-Scale Discriminators

For video, discriminators can operate at multiple scales (image-level and video-level) and across time (2D discriminator per frame + 3D discriminator across frames). The temporal discriminator penalizes flickering and temporal artifacts.

---

## The Perception-Distortion Tradeoff

**Theorem (Blau & Michaeli, 2018).** For any distribution-distortion pair, there is a fundamental tradeoff:

$$D(G) \geq D_{\min}(d(p_G, p_X))$$

where \(D(G)\) is the expected distortion (e.g., MSE), \(d(p_G, p_X)\) is a divergence between the generated and real distributions (perceptual quality), and \(D_{\min}\) is a monotonically decreasing function.

**In plain language:** you cannot simultaneously minimize distortion (MSE, PSNR) and maximize perceptual quality (make the distribution of outputs match the distribution of real data). Reducing one increases the other.

**Why?** The distortion-optimal prediction (conditional mean) is blurry --- it has low MSE but poor perceptual quality (does not look real). A sample from the conditional distribution has high perceptual quality (looks real) but higher MSE (it is one specific realization, not the optimal average).

The Pareto frontier traces the optimal tradeoff. Practical systems choose a point on this frontier via the loss function weighting:
- Pure MSE → minimum distortion, maximum blur
- Pure adversarial → maximum perceptual quality, higher distortion
- Weighted combination → intermediate point on the frontier

For video, this tradeoff has an additional temporal dimension: per-frame sharpness (perceptual quality) can conflict with temporal consistency (distortion across frames).

---

## Temporal Perceptual Losses for Video

Spatial perceptual losses applied per-frame do not enforce temporal consistency. A video where each frame independently looks great but flickers between frames is perceptually terrible.

### Warping Loss

Given optical flow \(\mathbf{u}_{t \to t+1}\) between consecutive frames, the warping loss penalizes temporal inconsistency:

$$\mathcal{L}_{\text{warp}} = \sum_t \|\mathbf{x}_{t+1} - \text{warp}(\mathbf{x}_t, \mathbf{u}_{t \to t+1})\|_1 \cdot (1 - O_t)$$

where \(O_t\) is an occlusion mask (regions visible in one frame but not the other are excluded). This loss says: if you warp frame \(t\) to frame \(t+1\) using the flow, the result should match frame \(t+1\) in non-occluded regions.

### Temporal Feature Loss

Apply perceptual loss across time: extract features from consecutive frames and penalize their difference:

$$\mathcal{L}_{\text{temp-feat}} = \sum_t \|\phi(\mathbf{x}_{t+1}) - \text{warp}(\phi(\mathbf{x}_t), \mathbf{u})\|_2^2$$

This enforces consistency in feature space, which is more robust to small spatial misalignments.

### 3D Perceptual Features

Use a video-pretrained backbone (e.g., I3D, VideoMAE) to extract spatiotemporal features. These features encode temporal patterns (motion, rhythm, consistency) that frame-level features miss:

$$\mathcal{L}_{\text{3D-perc}} = \|\Phi_{\text{3D}}(\hat{\mathbf{v}}) - \Phi_{\text{3D}}(\mathbf{v})\|_2^2$$

where \(\Phi_{\text{3D}}\) is the 3D feature extractor and \(\mathbf{v}\) is the video.

---

## The Loss Landscape of Video Generation

Modern video generation models combine multiple losses, each targeting a different aspect of quality:

$$\mathcal{L}_{\text{total}} = \lambda_1 \|\hat{\mathbf{x}} - \mathbf{x}\|_1 + \lambda_2 \mathcal{L}_{\text{perc}} + \lambda_3 \mathcal{L}_{\text{style}} + \lambda_4 \mathcal{L}_{\text{adv}} + \lambda_5 D_{\text{KL}} + \lambda_6 \mathcal{L}_{\text{warp}}$$

| Loss | What it encourages | What it discourages |
|------|-------------------|-------------------|
| L1 / MSE | Low distortion, color accuracy | Blur |
| Perceptual | Structural similarity | Pixel-exact matches |
| Style (Gram) | Texture realism | Content preservation |
| Adversarial | Sharpness, realism | Mode collapse |
| KL divergence | Smooth latent space | Overfitting |
| Warping | Temporal consistency | Independent frame processing |

The weights \(\lambda_i\) determine where on the perception-distortion Pareto frontier the model operates. These are typically tuned empirically --- there is no known closed-form optimal weighting.

---

## Python: Feature Space Distances and Perceptual Similarity

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def create_test_images():
    """Create pairs of images with different types of distortions."""
    size = 128
    y, x = np.mgrid[0:size, 0:size].astype(float) / size

    # Reference image: sharp edges and texture
    ref = np.zeros((size, size))
    ref[30:100, 30:100] = 0.8
    ref += 0.15 * np.sin(20 * np.pi * x) * np.sin(15 * np.pi * y)
    ref = np.clip(ref, 0, 1)

    # Distortion 1: Blur (low MSE, low perceptual quality)
    blur = gaussian_filter(ref, sigma=3.0)

    # Distortion 2: Noise (higher MSE, higher perceptual quality - still sharp)
    noise = ref + 0.15 * np.random.randn(size, size)
    noise = np.clip(noise, 0, 1)

    # Distortion 3: Shift (high MSE, high perceptual quality)
    shift = np.roll(ref, 3, axis=1)

    # Distortion 4: Texture change (moderate MSE, very different style)
    texture = np.zeros((size, size))
    texture[30:100, 30:100] = 0.8
    texture += 0.15 * np.sin(40 * np.pi * x) * np.sin(30 * np.pi * y)
    texture = np.clip(texture, 0, 1)

    return ref, blur, noise, shift, texture

def compute_simple_features(img, scales=[1, 2, 4]):
    """Simple multi-scale feature extraction (proxy for VGG features)."""
    features = []
    for s in scales:
        smoothed = gaussian_filter(img, sigma=s)
        # Gradient features (edge detection)
        gx = np.gradient(smoothed, axis=1)
        gy = np.gradient(smoothed, axis=0)
        mag = np.sqrt(gx**2 + gy**2)
        features.append(mag)
        features.append(smoothed)
    return features

def feature_distance(img1, img2):
    """Compute feature-space distance (simplified perceptual loss)."""
    f1 = compute_simple_features(img1)
    f2 = compute_simple_features(img2)
    return sum(np.mean((a - b)**2) for a, b in zip(f1, f2))

def compute_gram(feature_map):
    """Compute Gram matrix from a feature map."""
    h, w = feature_map.shape
    f = feature_map.flatten()
    return np.outer(f, f) / (h * w)

def ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """Compute SSIM between two images."""
    mu1 = gaussian_filter(img1, sigma=1.5)
    mu2 = gaussian_filter(img2, sigma=1.5)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu12 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1**2, sigma=1.5) - mu1_sq
    sigma2_sq = gaussian_filter(img2**2, sigma=1.5) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma=1.5) - mu12

    ssim_map = ((2*mu12 + C1) * (2*sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)

# Generate images
ref, blur, noise, shift, texture = create_test_images()
distortions = {'Blur': blur, 'Noise': noise, 'Shift': shift, 'Texture': texture}

# Compute metrics
metrics = {}
for name, img in distortions.items():
    mse = np.mean((ref - img)**2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    ssim_val = ssim(ref, img)
    feat_dist = feature_distance(ref, img)
    metrics[name] = {'MSE': mse, 'PSNR': psnr, 'SSIM': ssim_val, 'Feature': feat_dist}

# Visualization
fig, axes = plt.subplots(2, 5, figsize=(18, 7))

# Top row: images
images = [ref, blur, noise, shift, texture]
titles = ['Reference', 'Blur', 'Noise', 'Shift (3px)', 'Texture Change']
for i, (img, title) in enumerate(zip(images, titles)):
    axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0, i].set_title(title, fontsize=10)
    axes[0, i].axis('off')

# Bottom row: metrics comparison
metric_names = ['MSE', 'PSNR', 'SSIM', 'Feature']
colors = ['#E53935', '#4fc3f7', '#66bb6a', '#FF9800']
dist_names = list(distortions.keys())

for i, (metric, color) in enumerate(zip(metric_names, colors)):
    values = [metrics[d][metric] for d in dist_names]
    axes[1, i].bar(dist_names, values, color=color, alpha=0.7)
    axes[1, i].set_title(f'{metric}', fontsize=11, color=color)
    axes[1, i].tick_params(axis='x', rotation=30)
    axes[1, i].grid(True, alpha=0.3, axis='y')

# Summary text
axes[1, 4].axis('off')
summary = (
    "Key Insight:\n\n"
    "• Blur has LOW MSE but\n  looks terrible\n\n"
    "• Shift has HIGH MSE but\n  looks identical\n\n"
    "• Feature distance better\n  matches perception\n\n"
    "• SSIM is intermediate"
)
axes[1, 4].text(0.1, 0.9, summary, transform=axes[1, 4].transAxes,
               fontsize=10, verticalalignment='top', color='#d4d4d4',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

plt.suptitle('Pixel Loss vs Perceptual Metrics: Why MSE Fails', fontsize=14)
plt.tight_layout()
plt.savefig('perceptual_loss_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

The lesson is clear: **the choice of loss function determines the quality of generation**. Pixel-space losses produce pixel-accurate but perceptually bad output (blur). Feature-space losses produce perceptually good but pixel-inaccurate output (sharp, realistic). Adversarial losses push further toward realism at the cost of diversity. And the perception-distortion tradeoff theorem tells us that these compromises are fundamental --- you cannot have it all. The art of training video generation models is choosing the right point on this tradeoff curve for your application.
