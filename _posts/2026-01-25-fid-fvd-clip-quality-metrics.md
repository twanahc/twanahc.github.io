---
layout: post
title: "FID, FVD, and CLIP Score: The Mathematics of Measuring AI Video Quality"
date: 2026-01-25
category: math
---

How do you measure whether one video generation model is better than another? Not by vibes. Not by looking at cherry-picked samples. You measure it with metrics --- and the three metrics that dominate this space are FID, FVD, and CLIP Score.

Each measures something different. FID measures distributional similarity to real data. FVD extends that measurement to the temporal domain. CLIP Score measures whether the generated output actually matches what was asked for. Together they capture the two axes of quality that matter: realism and controllability.

But these metrics are frequently misunderstood, misused, and over-relied upon. This post is the complete mathematical treatment of all three, with worked examples, derivations, practical guidance, and an honest discussion of what they fail to capture.

---

## Table of Contents

1. [Why We Need Quantitative Metrics](#why-we-need-quantitative-metrics)
2. [Frechet Inception Distance (FID)](#frechet-inception-distance-fid)
3. [FID Worked Example](#fid-worked-example)
4. [FID Limitations and Pitfalls](#fid-limitations-and-pitfalls)
5. [Frechet Video Distance (FVD)](#frechet-video-distance-fvd)
6. [CLIP Score](#clip-score)
7. [When Metrics Disagree](#when-metrics-disagree)
8. [Benchmark Scores: Where Models Stand](#benchmark-scores-where-models-stand)
9. [Human Preference and ELO Scoring](#human-preference-and-elo-scoring)
10. [Building a Metric Dashboard](#building-a-metric-dashboard)
11. [The Future of Video Quality Metrics](#the-future-of-video-quality-metrics)

---

## Why We Need Quantitative Metrics

Human evaluation is the gold standard. But it is expensive, slow, non-reproducible, and hard to scale. You cannot run human evaluation on every training checkpoint, every hyperparameter sweep, every ablation.

Quantitative metrics enable:

- **Automated evaluation** during training (compute metrics on a validation set every N steps)
- **Reproducible comparisons** between models (same metric, same evaluation set, same protocol)
- **Optimization targets** for training objectives (some models directly optimize for FID or CLIP Score)
- **Regression detection** in production (if FID suddenly increases on your generation pipeline, something broke)

The challenge: no single metric captures human perception of "quality." Each metric is a proxy that correlates with quality along one dimension while being blind to others. Understanding what each metric actually measures --- and where it fails --- is essential.

---

## Frechet Inception Distance (FID)

FID was introduced by Heusel et al. (2017) and quickly became the de facto standard for evaluating image generation models. The core idea: compare the distribution of generated images to the distribution of real images in a learned feature space.

### Step 1: Feature Extraction with InceptionV3

Both real and generated images are passed through an InceptionV3 network pretrained on ImageNet. Features are extracted from the final pooling layer (before the classification head), producing a 2048-dimensional feature vector for each image.

Why InceptionV3? It was the best available image classifier when FID was proposed. The pool-2048 features capture high-level semantic information --- object identity, scene composition, texture statistics --- while being invariant to small spatial perturbations.

For a set of $N$ images $\{x_1, x_2, \ldots, x_N\}$, we compute:

$$f_i = \text{InceptionV3}(x_i) \in \mathbb{R}^{2048}$$

### Step 2: Fit Gaussian Distributions

We model the feature distribution as a multivariate Gaussian. For real images:

$$\mu_r = \frac{1}{N_r} \sum_{i=1}^{N_r} f_i^{(r)}, \qquad \Sigma_r = \frac{1}{N_r - 1} \sum_{i=1}^{N_r} (f_i^{(r)} - \mu_r)(f_i^{(r)} - \mu_r)^T$$

And for generated images:

$$\mu_g = \frac{1}{N_g} \sum_{i=1}^{N_g} f_i^{(g)}, \qquad \Sigma_g = \frac{1}{N_g - 1} \sum_{i=1}^{N_g} (f_i^{(g)} - \mu_g)(f_i^{(g)} - \mu_g)^T$$

where $\mu \in \mathbb{R}^{2048}$ is the mean vector and $\Sigma \in \mathbb{R}^{2048 \times 2048}$ is the covariance matrix.

### Step 3: Compute the Frechet Distance

The Frechet distance (also known as the Wasserstein-2 distance between Gaussians) measures the distance between two multivariate Gaussian distributions:

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)$$

Let us unpack each term:

**Term 1: $\|\mu_r - \mu_g\|^2$**

This is the squared Euclidean distance between the mean feature vectors. It measures whether the "center" of the generated distribution matches the "center" of the real distribution. If your model generates images that are, on average, in the right semantic neighborhood, this term is small.

**Term 2: $\text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$**

This measures the difference in spread and shape between the two distributions. Even if the means match perfectly, the generated distribution might be too narrow (mode collapse) or too wide (noisy outputs) or rotated in feature space.

The matrix square root $(\Sigma_r \Sigma_g)^{1/2}$ is the unique positive-definite matrix $M$ such that $M^2 = \Sigma_r \Sigma_g$. When $\Sigma_r = \Sigma_g$, we get $(\Sigma_r^2)^{1/2} = \Sigma_r$, and the trace term becomes $\text{Tr}(\Sigma_r + \Sigma_r - 2\Sigma_r) = 0$.

### The Full Derivation

The Frechet distance between two Gaussians $\mathcal{N}(\mu_1, \Sigma_1)$ and $\mathcal{N}(\mu_2, \Sigma_2)$ is defined as:

$$d_F^2 = \inf_{\gamma \in \Gamma(\mathcal{N}_1, \mathcal{N}_2)} \mathbb{E}_{(x,y) \sim \gamma}\left[\|x - y\|^2\right]$$

where $\Gamma$ is the set of all joint distributions (couplings) with the given marginals. The optimal coupling for Gaussians is known in closed form. The joint distribution that minimizes the expected squared distance is:

$$\gamma^* = \mathcal{N}\left(\begin{pmatrix} \mu_1 \\ \mu_2 \end{pmatrix}, \begin{pmatrix} \Sigma_1 & C \\ C^T & \Sigma_2 \end{pmatrix}\right)$$

where $C = \Sigma_1^{1/2}(\Sigma_1^{1/2} \Sigma_2 \Sigma_1^{1/2})^{1/2} \Sigma_1^{-1/2}$.

Evaluating the expected squared distance under this optimal coupling yields:

$$d_F^2 = \|\mu_1 - \mu_2\|^2 + \text{Tr}(\Sigma_1) + \text{Tr}(\Sigma_2) - 2\text{Tr}\left((\Sigma_1^{1/2} \Sigma_2 \Sigma_1^{1/2})^{1/2}\right)$$

Using the cyclic property of the trace and the identity $\text{Tr}(A^{1/2}) = \text{Tr}((A)^{1/2})$ for positive-definite matrices:

$$\text{Tr}\left((\Sigma_1^{1/2} \Sigma_2 \Sigma_1^{1/2})^{1/2}\right) = \text{Tr}\left((\Sigma_1 \Sigma_2)^{1/2}\right)$$

This gives us the standard FID formula:

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

<svg viewBox="0 0 740 480" xmlns="http://www.w3.org/2000/svg" style="max-width:740px; margin: 2em auto; display: block;">
  <rect width="740" height="480" fill="white"/>

  <text x="370" y="30" font-family="Georgia, serif" font-size="16" fill="#333" text-anchor="middle" font-weight="bold">FID Computation Pipeline</text>

  <!-- Real images box -->
  <rect x="30" y="60" width="140" height="60" rx="8" fill="#e3f2fd" stroke="#4fc3f7" stroke-width="2"/>
  <text x="100" y="85" font-family="Georgia, serif" font-size="13" fill="#333" text-anchor="middle">Real Images</text>
  <text x="100" y="105" font-family="monospace" font-size="11" fill="#666" text-anchor="middle">N = 50,000</text>

  <!-- Generated images box -->
  <rect x="30" y="360" width="140" height="60" rx="8" fill="#fce4ec" stroke="#ef5350" stroke-width="2"/>
  <text x="100" y="385" font-family="Georgia, serif" font-size="13" fill="#333" text-anchor="middle">Generated Images</text>
  <text x="100" y="405" font-family="monospace" font-size="11" fill="#666" text-anchor="middle">N = 50,000</text>

  <!-- InceptionV3 boxes -->
  <rect x="220" y="60" width="140" height="60" rx="8" fill="#f3e5f5" stroke="#ab47bc" stroke-width="2"/>
  <text x="290" y="85" font-family="Georgia, serif" font-size="13" fill="#333" text-anchor="middle">InceptionV3</text>
  <text x="290" y="105" font-family="monospace" font-size="11" fill="#666" text-anchor="middle">pool-2048</text>

  <rect x="220" y="360" width="140" height="60" rx="8" fill="#f3e5f5" stroke="#ab47bc" stroke-width="2"/>
  <text x="290" y="385" font-family="Georgia, serif" font-size="13" fill="#333" text-anchor="middle">InceptionV3</text>
  <text x="290" y="405" font-family="monospace" font-size="11" fill="#666" text-anchor="middle">pool-2048</text>

  <!-- Feature vectors -->
  <rect x="410" y="60" width="130" height="60" rx="8" fill="#e8f5e9" stroke="#8bc34a" stroke-width="2"/>
  <text x="475" y="80" font-family="Georgia, serif" font-size="12" fill="#333" text-anchor="middle">Features</text>
  <text x="475" y="97" font-family="monospace" font-size="10" fill="#666" text-anchor="middle">f_r ∈ R^2048</text>
  <text x="475" y="112" font-family="monospace" font-size="10" fill="#666" text-anchor="middle">N x 2048 matrix</text>

  <rect x="410" y="360" width="130" height="60" rx="8" fill="#e8f5e9" stroke="#8bc34a" stroke-width="2"/>
  <text x="475" y="380" font-family="Georgia, serif" font-size="12" fill="#333" text-anchor="middle">Features</text>
  <text x="475" y="397" font-family="monospace" font-size="10" fill="#666" text-anchor="middle">f_g ∈ R^2048</text>
  <text x="475" y="412" font-family="monospace" font-size="10" fill="#666" text-anchor="middle">N x 2048 matrix</text>

  <!-- Statistics -->
  <rect x="570" y="60" width="140" height="60" rx="8" fill="#fff3e0" stroke="#ffa726" stroke-width="2"/>
  <text x="640" y="80" font-family="Georgia, serif" font-size="12" fill="#333" text-anchor="middle">Fit Gaussian</text>
  <text x="640" y="97" font-family="monospace" font-size="10" fill="#666" text-anchor="middle">μ_r, Σ_r</text>
  <text x="640" y="112" font-family="monospace" font-size="10" fill="#666" text-anchor="middle">2048 + 2048²</text>

  <rect x="570" y="360" width="140" height="60" rx="8" fill="#fff3e0" stroke="#ffa726" stroke-width="2"/>
  <text x="640" y="380" font-family="Georgia, serif" font-size="12" fill="#333" text-anchor="middle">Fit Gaussian</text>
  <text x="640" y="397" font-family="monospace" font-size="10" fill="#666" text-anchor="middle">μ_g, Σ_g</text>
  <text x="640" y="412" font-family="monospace" font-size="10" fill="#666" text-anchor="middle">2048 + 2048²</text>

  <!-- Frechet Distance box -->
  <rect x="490" y="195" width="220" height="90" rx="10" fill="#e8eaf6" stroke="#5c6bc0" stroke-width="2.5"/>
  <text x="600" y="222" font-family="Georgia, serif" font-size="14" fill="#333" text-anchor="middle" font-weight="bold">Frechet Distance</text>
  <text x="600" y="245" font-family="monospace" font-size="10" fill="#444" text-anchor="middle">||μ_r - μ_g||² +</text>
  <text x="600" y="262" font-family="monospace" font-size="10" fill="#444" text-anchor="middle">Tr(Σ_r + Σ_g - 2(Σ_rΣ_g)^½)</text>
  <text x="600" y="278" font-family="Georgia, serif" font-size="12" fill="#5c6bc0" text-anchor="middle" font-weight="bold">= FID score</text>

  <!-- Arrows -->
  <defs>
    <marker id="arrowFid" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#666"/>
    </marker>
  </defs>

  <!-- Row 1 arrows -->
  <line x1="170" y1="90" x2="218" y2="90" stroke="#666" stroke-width="1.5" marker-end="url(#arrowFid)"/>
  <line x1="360" y1="90" x2="408" y2="90" stroke="#666" stroke-width="1.5" marker-end="url(#arrowFid)"/>
  <line x1="540" y1="90" x2="568" y2="90" stroke="#666" stroke-width="1.5" marker-end="url(#arrowFid)"/>

  <!-- Row 2 arrows -->
  <line x1="170" y1="390" x2="218" y2="390" stroke="#666" stroke-width="1.5" marker-end="url(#arrowFid)"/>
  <line x1="360" y1="390" x2="408" y2="390" stroke="#666" stroke-width="1.5" marker-end="url(#arrowFid)"/>
  <line x1="540" y1="390" x2="568" y2="390" stroke="#666" stroke-width="1.5" marker-end="url(#arrowFid)"/>

  <!-- Arrows to Frechet Distance -->
  <line x1="640" y1="122" x2="615" y2="193" stroke="#666" stroke-width="1.5" marker-end="url(#arrowFid)"/>
  <line x1="640" y1="358" x2="615" y2="287" stroke="#666" stroke-width="1.5" marker-end="url(#arrowFid)"/>
</svg>

### Properties of FID

**FID = 0** means the two distributions are identical. This never happens in practice --- even comparing a dataset to itself with finite samples gives a small positive FID due to estimation noise.

**FID is symmetric**: $\text{FID}(p, q) = \text{FID}(q, p)$, because the Frechet distance is symmetric.

**FID satisfies the triangle inequality**: It is a proper metric on the space of Gaussian distributions.

**Lower is better**: A model with FID = 3 is better than one with FID = 15 (assuming the same evaluation protocol).

---

## FID Worked Example

Let us walk through a toy FID calculation with 2D features to build intuition.

### Setup

Suppose we extract 2D features (instead of 2048D) from 5 real images and 5 generated images:

**Real image features:**

| Image | $f_1$ | $f_2$ |
|:-:|:-:|:-:|
| $r_1$ | 2.0 | 3.0 |
| $r_2$ | 2.5 | 3.5 |
| $r_3$ | 1.5 | 2.5 |
| $r_4$ | 2.2 | 3.2 |
| $r_5$ | 1.8 | 2.8 |

**Generated image features:**

| Image | $f_1$ | $f_2$ |
|:-:|:-:|:-:|
| $g_1$ | 3.0 | 4.0 |
| $g_2$ | 3.5 | 4.5 |
| $g_3$ | 2.5 | 3.5 |
| $g_4$ | 3.2 | 4.2 |
| $g_5$ | 2.8 | 3.8 |

### Step 1: Compute Means

$$\mu_r = \begin{pmatrix} \frac{2.0 + 2.5 + 1.5 + 2.2 + 1.8}{5} \\ \frac{3.0 + 3.5 + 2.5 + 3.2 + 2.8}{5} \end{pmatrix} = \begin{pmatrix} 2.0 \\ 3.0 \end{pmatrix}$$

$$\mu_g = \begin{pmatrix} \frac{3.0 + 3.5 + 2.5 + 3.2 + 2.8}{5} \\ \frac{4.0 + 4.5 + 3.5 + 4.2 + 3.8}{5} \end{pmatrix} = \begin{pmatrix} 3.0 \\ 4.0 \end{pmatrix}$$

### Step 2: Compute Mean Difference

$$\|\mu_r - \mu_g\|^2 = (2.0 - 3.0)^2 + (3.0 - 4.0)^2 = 1.0 + 1.0 = 2.0$$

### Step 3: Compute Covariance Matrices

For the real features, centering the data:

| | $f_1 - \bar{f}_1$ | $f_2 - \bar{f}_2$ |
|:-:|:-:|:-:|
| $r_1$ | 0.0 | 0.0 |
| $r_2$ | 0.5 | 0.5 |
| $r_3$ | -0.5 | -0.5 |
| $r_4$ | 0.2 | 0.2 |
| $r_5$ | -0.2 | -0.2 |

$$\Sigma_r = \frac{1}{4} \begin{pmatrix} 0.0^2 + 0.5^2 + 0.5^2 + 0.2^2 + 0.2^2 & \cdots \\ \cdots & \cdots \end{pmatrix}$$

Computing each entry:

$$(\Sigma_r)_{11} = \frac{0 + 0.25 + 0.25 + 0.04 + 0.04}{4} = \frac{0.58}{4} = 0.145$$

$$(\Sigma_r)_{22} = \frac{0 + 0.25 + 0.25 + 0.04 + 0.04}{4} = 0.145$$

$$(\Sigma_r)_{12} = (\Sigma_r)_{21} = \frac{0 + 0.25 + 0.25 + 0.04 + 0.04}{4} = 0.145$$

$$\Sigma_r = \begin{pmatrix} 0.145 & 0.145 \\ 0.145 & 0.145 \end{pmatrix}$$

The generated images have the same variance pattern (shifted by +1 in both dimensions), so:

$$\Sigma_g = \begin{pmatrix} 0.145 & 0.145 \\ 0.145 & 0.145 \end{pmatrix}$$

### Step 4: Compute the Matrix Square Root

Since $\Sigma_r = \Sigma_g = \Sigma$ in this example:

$$\Sigma_r \Sigma_g = \Sigma^2 = \begin{pmatrix} 0.145 & 0.145 \\ 0.145 & 0.145 \end{pmatrix}^2 = \begin{pmatrix} 0.042 & 0.042 \\ 0.042 & 0.042 \end{pmatrix}$$

The eigenvalues of $\Sigma^2$ are $\lambda_1 = 0.084$ and $\lambda_2 = 0$ (since $\Sigma$ is rank-1).

The square root $(\Sigma^2)^{1/2}$ has eigenvalues $\sqrt{0.084} = 0.290$ and $0$, with the same eigenvectors. This gives:

$$(\Sigma_r \Sigma_g)^{1/2} = \Sigma = \begin{pmatrix} 0.145 & 0.145 \\ 0.145 & 0.145 \end{pmatrix}$$

(This simplification holds because $\Sigma_r = \Sigma_g$ and both are positive semi-definite.)

### Step 5: Compute the Trace Term

$$\text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}) = \text{Tr}(2\Sigma - 2\Sigma) = 0$$

### Step 6: Final FID

$$\text{FID} = 2.0 + 0 = 2.0$$

In this toy example, the two distributions have identical shapes (same covariance) but different centers, so the entire FID comes from the mean difference term. In practice, the covariance term usually dominates for high-dimensional features.

### What Different FID Values Mean

| FID Range | Interpretation |
|:-:|:--|
| 0 -- 5 | Excellent; near-indistinguishable from real data |
| 5 -- 15 | Very good; state-of-the-art for most generative models |
| 15 -- 50 | Good; noticeable but acceptable difference from real data |
| 50 -- 100 | Mediocre; clearly synthetic but structurally coherent |
| 100+ | Poor; obvious artifacts, mode collapse, or distribution mismatch |

---

## FID Limitations and Pitfalls

FID is ubiquitous but deeply flawed. Understanding its limitations is as important as understanding its math.

### 1. The Gaussian Assumption

FID assumes that Inception features follow a multivariate Gaussian distribution. They do not. Real feature distributions are heavy-tailed, multi-modal, and have complex dependency structures. The Gaussian approximation discards higher-order statistics that may be perceptually important.

### 2. Sample Size Sensitivity

FID is biased for finite samples. The bias depends on sample size $N$ and feature dimensionality $d$:

$$\text{Bias} \approx \frac{d}{N}$$

For $d = 2048$ and $N = 10{,}000$, the bias is approximately 0.2 --- small but not negligible when comparing models with similar FID. The standard recommendation is $N \geq 50{,}000$ samples, but many papers use fewer.

| Sample Size | Approximate FID Bias (d=2048) |
|:-:|:-:|
| 1,000 | ~2.0 |
| 5,000 | ~0.4 |
| 10,000 | ~0.2 |
| 50,000 | ~0.04 |
| 100,000 | ~0.02 |

### 3. Inception Network Dependency

FID depends entirely on InceptionV3's feature space. This network was trained on ImageNet in 2015. It has known biases:

- Trained on 224x224 images (modern generators produce 1024x1024+)
- Trained on natural photographs (may not capture artistic styles, animations, etc.)
- Feature space emphasizes object classification features, not texture or aesthetic features
- Not trained on video --- frame-wise FID treats each frame independently

### 4. Mode Collapse Blindness

A model that generates only one perfect image (always the same output) would have low FID if that image is representative of the real distribution's mean. FID does not penalize lack of diversity directly --- the covariance term captures some diversity, but a narrow-but-well-centered distribution can still score well.

### 5. The Reference Set Problem

FID is measured relative to a reference dataset. Different reference sets give different FID values for the same model. This makes cross-paper comparisons unreliable unless the exact same reference set and preprocessing pipeline are used.

### 6. Not Sensitive to Fine Details

Small differences in texture, lighting, or color grading that humans easily perceive may not change FID significantly, because InceptionV3 features are invariant to these low-level properties (they were designed for classification, not quality assessment).

---

## Frechet Video Distance (FVD)

FVD (Unterthiner et al., 2019) extends FID to video by replacing the InceptionV3 image encoder with an I3D (Inflated 3D ConvNet) video encoder.

### The I3D Architecture

I3D (Carreira & Zisserman, 2017) inflates 2D convolution filters from InceptionV1 into 3D by adding a temporal dimension. A 2D filter of shape $(k, k)$ becomes a 3D filter of shape $(k, k, k)$ that processes spatial and temporal information jointly.

The network is pretrained on the Kinetics-400 action recognition dataset (400 classes of human actions in video). Features are extracted from the final average pooling layer, producing a 400-dimensional feature vector for each video clip.

Key difference from FID: the features capture spatiotemporal patterns, not just spatial ones. A video with coherent motion activates different I3D features than a video with flickering or temporal artifacts, even if each individual frame looks acceptable.

### The FVD Formula

FVD uses the same Frechet distance formula as FID, but with video features:

$$\text{FVD} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

where $\mu_r, \Sigma_r$ are computed from I3D features of real videos and $\mu_g, \Sigma_g$ from generated videos.

### Input Requirements

I3D expects video clips of a fixed duration (typically 16 frames). For longer generated videos, clips are either:
- Truncated to 16 frames
- Sampled at regular intervals (e.g., every 4th frame from a 64-frame video)
- Evaluated as sliding windows with results averaged

The standard evaluation protocol:
1. Sample 2,048 real video clips and 2,048 generated video clips
2. Resize all clips to 224x224 spatial resolution
3. Extract I3D features (400-dimensional)
4. Fit Gaussians and compute Frechet distance

### Why FVD Matters More Than Frame-FID for Video

Frame-FID (applying standard image FID to individual frames) misses temporal quality entirely:

| Scenario | Frame-FID | FVD | Human Rating |
|:--|:-:|:-:|:-:|
| Perfect frames, random order | Low (good) | High (bad) | Bad |
| Perfect frames, smooth motion | Low (good) | Low (good) | Good |
| Slightly blurry frames, smooth motion | Medium | Low-medium | Acceptable |
| Flickering between good frames | Low (good) | High (bad) | Bad |

The first and fourth rows illustrate why frame-FID is insufficient for video evaluation. Both would score well on frame-FID because each individual frame looks good. But the videos are unwatchable because the temporal coherence is broken. FVD captures this because I3D features encode motion patterns.

### FVD Limitations

FVD inherits all of FID's limitations plus additional ones:

**1. I3D was trained on action recognition.** Its feature space emphasizes human actions and motion patterns. It may not capture aesthetic quality, color accuracy, or scene composition as well as spatial features.

**2. Short temporal context.** I3D processes 16-frame clips. Long-range temporal coherence (e.g., consistent lighting over 10 seconds) is not well captured.

**3. Low feature dimensionality.** 400 dimensions (vs. FID's 2048) means less expressive power for capturing distributional differences.

**4. Small evaluation sets.** The standard protocol uses 2,048 videos, far fewer than FID's 50,000 images. This increases statistical noise.

**5. Resolution mismatch.** Like InceptionV3 for FID, I3D was trained on relatively low-resolution video. Modern generators produce much higher resolution.

### Correlation with Human Judgment

How well does FVD predict what humans actually prefer? Studies have found moderate-to-good correlation, but with important caveats:

<svg viewBox="0 0 700 450" xmlns="http://www.w3.org/2000/svg" style="max-width:700px; margin: 2em auto; display: block;">
  <rect width="700" height="450" fill="white"/>

  <text x="350" y="28" font-family="Georgia, serif" font-size="16" fill="#333" text-anchor="middle" font-weight="bold">FVD vs. Human Preference (Illustrative)</text>

  <!-- Axes -->
  <line x1="80" y1="380" x2="660" y2="380" stroke="#333" stroke-width="1.5"/>
  <line x1="80" y1="380" x2="80" y2="50" stroke="#333" stroke-width="1.5"/>

  <!-- Gridlines -->
  <line x1="80" y1="314" x2="660" y2="314" stroke="#eee" stroke-width="0.8"/>
  <line x1="80" y1="248" x2="660" y2="248" stroke="#eee" stroke-width="0.8"/>
  <line x1="80" y1="182" x2="660" y2="182" stroke="#eee" stroke-width="0.8"/>
  <line x1="80" y1="116" x2="660" y2="116" stroke="#eee" stroke-width="0.8"/>

  <!-- Y-axis labels -->
  <text x="70" y="385" font-family="monospace" font-size="11" fill="#666" text-anchor="end">40%</text>
  <text x="70" y="319" font-family="monospace" font-size="11" fill="#666" text-anchor="end">50%</text>
  <text x="70" y="253" font-family="monospace" font-size="11" fill="#666" text-anchor="end">60%</text>
  <text x="70" y="187" font-family="monospace" font-size="11" fill="#666" text-anchor="end">70%</text>
  <text x="70" y="121" font-family="monospace" font-size="11" fill="#666" text-anchor="end">80%</text>

  <!-- X-axis labels -->
  <text x="80" y="400" font-family="monospace" font-size="11" fill="#666" text-anchor="middle">0</text>
  <text x="225" y="400" font-family="monospace" font-size="11" fill="#666" text-anchor="middle">200</text>
  <text x="370" y="400" font-family="monospace" font-size="11" fill="#666" text-anchor="middle">400</text>
  <text x="515" y="400" font-family="monospace" font-size="11" fill="#666" text-anchor="middle">600</text>
  <text x="660" y="400" font-family="monospace" font-size="11" fill="#666" text-anchor="middle">800</text>

  <!-- Axis titles -->
  <text x="370" y="430" font-family="Georgia, serif" font-size="13" fill="#333" text-anchor="middle">FVD (lower is better)</text>
  <text x="25" y="215" font-family="Georgia, serif" font-size="13" fill="#333" text-anchor="middle" transform="rotate(-90 25 215)">Human Win Rate (%)</text>

  <!-- Trend line (negative correlation) -->
  <line x1="105" y1="100" x2="640" y2="370" stroke="#4fc3f7" stroke-width="2" stroke-dasharray="8,4" opacity="0.5"/>

  <!-- Scatter points (models) with labels -->
  <!-- Veo 3.1 - low FVD, high human pref -->
  <circle cx="130" cy="105" r="8" fill="#4fc3f7" stroke="white" stroke-width="2"/>
  <text x="145" y="100" font-family="Georgia, serif" font-size="11" fill="#0288d1" font-weight="bold">Veo 3.1</text>

  <!-- Kling 3.0 -->
  <circle cx="175" cy="130" r="8" fill="#4fc3f7" stroke="white" stroke-width="2"/>
  <text x="190" y="125" font-family="Georgia, serif" font-size="11" fill="#0288d1" font-weight="bold">Kling 3.0</text>

  <!-- Sora 2 -->
  <circle cx="200" cy="145" r="8" fill="#4fc3f7" stroke="white" stroke-width="2"/>
  <text x="215" y="145" font-family="Georgia, serif" font-size="11" fill="#0288d1" font-weight="bold">Sora 2</text>

  <!-- Wan 2.2 -->
  <circle cx="310" cy="230" r="8" fill="#8bc34a" stroke="white" stroke-width="2"/>
  <text x="325" y="225" font-family="Georgia, serif" font-size="11" fill="#558b2f" font-weight="bold">Wan 2.2</text>

  <!-- LTX-2 -->
  <circle cx="360" cy="260" r="8" fill="#8bc34a" stroke="white" stroke-width="2"/>
  <text x="375" y="260" font-family="Georgia, serif" font-size="11" fill="#558b2f" font-weight="bold">LTX-2</text>

  <!-- CogVideo -->
  <circle cx="420" cy="290" r="8" fill="#ffa726" stroke="white" stroke-width="2"/>
  <text x="435" y="290" font-family="Georgia, serif" font-size="11" fill="#e65100" font-weight="bold">CogVideo</text>

  <!-- AnimateDiff -->
  <circle cx="500" cy="330" r="8" fill="#ffa726" stroke="white" stroke-width="2"/>
  <text x="460" y="350" font-family="Georgia, serif" font-size="11" fill="#e65100" font-weight="bold">AnimateDiff</text>

  <!-- Outlier: high FVD but decent human pref (artistic style) -->
  <circle cx="480" cy="195" r="8" fill="#ef5350" stroke="white" stroke-width="2"/>
  <text x="495" y="190" font-family="Georgia, serif" font-size="11" fill="#c62828">Outlier A</text>
  <text x="495" y="204" font-family="Georgia, serif" font-size="10" fill="#999">(artistic style)</text>

  <!-- Correlation annotation -->
  <rect x="420" y="55" width="230" height="45" rx="4" fill="white" stroke="#ddd" stroke-width="1"/>
  <text x="535" y="73" font-family="Georgia, serif" font-size="12" fill="#333" text-anchor="middle">Spearman ρ ≈ -0.75</text>
  <text x="535" y="90" font-family="Georgia, serif" font-size="11" fill="#999" text-anchor="middle">Moderate-strong correlation</text>
</svg>

The Spearman rank correlation between FVD and human preference is typically $\rho \approx -0.7$ to $-0.8$ (negative because lower FVD = better, higher human preference = better). This is decent but far from perfect. The outliers are informative:

- **Artistic style models** may have high FVD (their outputs don't look like the real-video reference distribution) but high human preference (humans find them appealing)
- **Mode-collapsed models** may have low FVD (the few things they generate match the reference well) but low human preference (boring, repetitive)

---

## CLIP Score

CLIP Score measures something entirely different from FID/FVD. Instead of comparing generated outputs to real data, it measures how well a generated output matches its text prompt.

### The CLIP Architecture

CLIP (Radford et al., 2021) consists of two encoders trained jointly with contrastive learning:

- **Image encoder** $\text{CLIP}_\text{image}: \mathbb{R}^{H \times W \times 3} \to \mathbb{R}^{d}$, typically a Vision Transformer (ViT-L/14), producing a $d$-dimensional embedding ($d = 768$ for ViT-L/14)
- **Text encoder** $\text{CLIP}_\text{text}: \text{tokens} \to \mathbb{R}^{d}$, a Transformer that produces a $d$-dimensional embedding from tokenized text

Training objective: maximize the cosine similarity between matched image-text pairs and minimize it for unmatched pairs. After training on 400 million image-text pairs from the internet, CLIP learns a shared embedding space where semantically related images and text are close together.

### CLIP Score Definition

For a generated image $I$ and its text prompt $T$:

$$\text{CLIP Score}(I, T) = \cos\left(\text{CLIP}_\text{image}(I), \ \text{CLIP}_\text{text}(T)\right) = \frac{\text{CLIP}_\text{image}(I) \cdot \text{CLIP}_\text{text}(T)}{\|\text{CLIP}_\text{image}(I)\| \cdot \|\text{CLIP}_\text{text}(T)\|}$$

The score ranges from -1 to 1, but in practice for reasonable image-text pairs it falls between 0.15 and 0.40.

**Higher is better**: A CLIP Score of 0.35 indicates stronger text-image alignment than 0.25.

### CLIP Score for Video

CLIP was trained on images, not video. Applying it to video requires an adaptation:

**Frame-averaged CLIP Score**: Extract CLIP image features from each frame, average them, compute cosine similarity with the text:

$$\text{CLIP}_\text{video}(V, T) = \cos\left(\frac{1}{F}\sum_{f=1}^{F} \text{CLIP}_\text{image}(V_f), \ \text{CLIP}_\text{text}(T)\right)$$

where $V_f$ is the $f$-th frame and $F$ is the total number of frames.

**Per-frame CLIP Score**: Compute CLIP Score for each frame independently and report the mean:

$$\text{CLIP}_\text{video}(V, T) = \frac{1}{F}\sum_{f=1}^{F} \cos\left(\text{CLIP}_\text{image}(V_f), \ \text{CLIP}_\text{text}(T)\right)$$

These give similar but not identical results. The first averages in embedding space (the average feature might not correspond to any real frame). The second averages in score space (each frame's alignment is measured independently).

**Temporal CLIP extensions**: Some researchers have proposed video-aware CLIP models (e.g., VideoCLIP, X-CLIP) that encode temporal information. These are not yet standard in evaluation pipelines.

### What CLIP Score Captures and Misses

**Captures:**
- Semantic alignment (does the image contain the objects/scenes described in the text?)
- Compositional alignment (are the spatial relationships roughly correct?)
- Style alignment (to some degree --- "oil painting" vs "photograph")

**Misses:**
- Fine-grained spatial accuracy (text says "left" vs. "right" --- CLIP often cannot distinguish)
- Counting (text says "three cats" --- CLIP may score "two cats" equally well)
- Negation (text says "no people" --- CLIP may score images with people highly because "people" is mentioned)
- Temporal alignment (for video --- does the action happen in the right order?)
- Aesthetic quality (a blurry image that matches the text semantically can score well)

### CLIP Score Ranges

| CLIP Score | Interpretation |
|:-:|:--|
| 0.35+ | Excellent alignment; strongly matches the prompt |
| 0.30 -- 0.35 | Good alignment; most elements of the prompt are present |
| 0.25 -- 0.30 | Moderate alignment; main subject matches but details may differ |
| 0.20 -- 0.25 | Weak alignment; vaguely related to the prompt |
| < 0.20 | Poor alignment; effectively unrelated to the prompt |

---

## When Metrics Disagree

The most informative analysis happens when metrics disagree. Each disagreement tells you something specific about the model's failure mode.

### Case 1: Low FVD, Low CLIP Score

The model generates realistic-looking videos that do not match the text prompt. This is a controllability problem, not a quality problem. The model has learned the distribution of real video but cannot steer its generations based on text input.

**Diagnosis**: Weak text conditioning, underpowered cross-attention layers, insufficient text-video paired training data, or guidance scale too low.

### Case 2: High FVD, High CLIP Score

The model generates videos that match the text prompt well but do not look like real video. Common with models that over-rely on text conditioning at the expense of realism.

**Diagnosis**: Guidance scale too high, over-fitted to text-image alignment during training, insufficient diversity in training data. The model is essentially "illustrating" the text rather than generating natural video.

### Case 3: Low Frame-FID, High FVD

Each individual frame looks realistic, but the video has temporal artifacts. Flickering, jitter, inconsistent motion.

**Diagnosis**: Weak temporal modeling. The model generates good frames but poor inter-frame transitions. This is common in models that adapt image architectures to video without sufficient temporal modeling.

### Case 4: All Metrics Good, Human Preference Low

The numbers look great but humans don't like the output. This is the "uncanny valley" problem. The video may be technically correct but aesthetically unpleasant --- weird color grading, stiff motion, unnatural compositions.

**Diagnosis**: Metrics do not capture aesthetic preferences. Need human evaluation or learned aesthetic predictors.

### Summary Table

| Scenario | FVD | CLIP | Frame-FID | Likely Issue |
|:--|:-:|:-:|:-:|:--|
| Realistic but wrong | Low | Low | Low | Weak conditioning |
| Matches prompt but fake-looking | High | High | High | Over-guidance / unrealistic |
| Good frames, bad video | N/A | Medium | Low | Temporal modeling failure |
| Metrics good, humans disagree | Low | High | Low | Aesthetic / uncanny valley |
| Everything bad | High | Low | High | Fundamentally broken model |

---

## Benchmark Scores: Where Models Stand

Reported metrics vary wildly across papers due to different evaluation protocols, reference datasets, and preprocessing. The following table compiles approximate values from published benchmarks and should be interpreted with caution.

### Image Models (FID on COCO-30K)

| Model | FID | CLIP Score | Year |
|:--|:-:|:-:|:-:|
| DALL-E 2 | 10.39 | 0.31 | 2022 |
| Imagen | 7.27 | 0.33 | 2022 |
| Stable Diffusion 1.5 | 9.62 | 0.31 | 2022 |
| SDXL | 6.80 | 0.33 | 2023 |
| DALL-E 3 | ~5.5 | 0.35 | 2023 |
| Flux.1 [dev] | ~4.8 | 0.34 | 2024 |
| Flux 2.0 | ~3.5 | 0.36 | 2025 |

### Video Models (FVD on UCF-101 / Custom Benchmarks)

| Model | FVD (approx.) | CLIP Score (frame-avg) | Notes |
|:--|:-:|:-:|:--|
| Make-A-Video (Meta) | 367 | 0.28 | Early text-to-video |
| Imagen Video (Google) | 290 | 0.30 | Cascaded diffusion |
| VideoLDM | 292 | 0.29 | Latent diffusion for video |
| Sora 1 (OpenAI) | ~180 | 0.33 | Not independently verified |
| Sora 2 (OpenAI) | ~140 | 0.35 | Estimated from API outputs |
| Veo 2 (Google) | ~160 | 0.34 | Estimated |
| Veo 3.1 (Google) | ~110 | 0.36 | With audio generation |
| Kling 2.0 (Kuaishou) | ~190 | 0.32 | Strong motion quality |
| Kling 3.0 | ~135 | 0.35 | Major improvement |
| Wan 2.2 (Alibaba) | ~210 | 0.31 | Open-source, MoE |
| LTX-2 (Lightricks) | ~250 | 0.30 | Open-source |

*Values are approximate and compiled from various sources. Direct comparison requires identical evaluation protocols.*

<svg viewBox="0 0 700 500" xmlns="http://www.w3.org/2000/svg" style="max-width:700px; margin: 2em auto; display: block;">
  <rect width="700" height="500" fill="white"/>

  <text x="350" y="30" font-family="Georgia, serif" font-size="16" fill="#333" text-anchor="middle" font-weight="bold">Video Model Comparison: Multi-Metric Radar</text>

  <!-- Radar center and axes -->
  <!-- Center at (350, 280), radius 170 -->
  <!-- 5 axes: FVD (inverse), CLIP Score, Temporal Coherence, Resolution, Motion Quality -->

  <!-- Pentagon background rings -->
  <!-- Ring at 100% -->
  <polygon points="350,110 512,218 462,397 238,397 188,218" fill="none" stroke="#e0e0e0" stroke-width="0.8"/>
  <!-- Ring at 75% -->
  <polygon points="350,153 471,230 434,364 266,364 229,230" fill="none" stroke="#eee" stroke-width="0.8"/>
  <!-- Ring at 50% -->
  <polygon points="350,195 431,243 405,330 295,330 269,243" fill="none" stroke="#eee" stroke-width="0.8"/>
  <!-- Ring at 25% -->
  <polygon points="350,238 391,256 378,297 322,297 310,256" fill="none" stroke="#eee" stroke-width="0.8"/>

  <!-- Axes lines -->
  <line x1="350" y1="280" x2="350" y2="110" stroke="#ccc" stroke-width="1"/>
  <line x1="350" y1="280" x2="512" y2="218" stroke="#ccc" stroke-width="1"/>
  <line x1="350" y1="280" x2="462" y2="397" stroke="#ccc" stroke-width="1"/>
  <line x1="350" y1="280" x2="238" y2="397" stroke="#ccc" stroke-width="1"/>
  <line x1="350" y1="280" x2="188" y2="218" stroke="#ccc" stroke-width="1"/>

  <!-- Axis labels -->
  <text x="350" y="98" font-family="Georgia, serif" font-size="12" fill="#333" text-anchor="middle">FVD (inv.)</text>
  <text x="525" y="215" font-family="Georgia, serif" font-size="12" fill="#333" text-anchor="start">CLIP Score</text>
  <text x="475" y="412" font-family="Georgia, serif" font-size="12" fill="#333" text-anchor="start">Resolution</text>
  <text x="180" y="412" font-family="Georgia, serif" font-size="12" fill="#333" text-anchor="end">Motion</text>
  <text x="172" y="215" font-family="Georgia, serif" font-size="12" fill="#333" text-anchor="end">Temporal</text>

  <!-- Veo 3.1 polygon (high on most) -->
  <!-- FVD: 90%, CLIP: 90%, Resolution: 95%, Motion: 85%, Temporal: 90% -->
  <polygon points="350,127 496,224 453,391 248,382 207,224"
    fill="#4fc3f7" fill-opacity="0.15" stroke="#4fc3f7" stroke-width="2.5"/>

  <!-- Kling 3.0 polygon -->
  <!-- FVD: 82%, CLIP: 85%, Resolution: 85%, Motion: 90%, Temporal: 80% -->
  <polygon points="350,141 487,229 445,379 242,391 222,232"
    fill="#ef5350" fill-opacity="0.1" stroke="#ef5350" stroke-width="2"/>

  <!-- Wan 2.2 polygon -->
  <!-- FVD: 60%, CLIP: 70%, Resolution: 65%, Motion: 65%, Temporal: 60% -->
  <polygon points="350,178 463,243 423,356 280,356 253,243"
    fill="#8bc34a" fill-opacity="0.1" stroke="#8bc34a" stroke-width="2"/>

  <!-- Data points -->
  <circle cx="350" cy="127" r="4" fill="#4fc3f7"/>
  <circle cx="496" cy="224" r="4" fill="#4fc3f7"/>
  <circle cx="453" cy="391" r="4" fill="#4fc3f7"/>
  <circle cx="248" cy="382" r="4" fill="#4fc3f7"/>
  <circle cx="207" cy="224" r="4" fill="#4fc3f7"/>

  <circle cx="350" cy="141" r="4" fill="#ef5350"/>
  <circle cx="487" cy="229" r="4" fill="#ef5350"/>
  <circle cx="445" cy="379" r="4" fill="#ef5350"/>
  <circle cx="242" cy="391" r="4" fill="#ef5350"/>
  <circle cx="222" cy="232" r="4" fill="#ef5350"/>

  <circle cx="350" cy="178" r="4" fill="#8bc34a"/>
  <circle cx="463" cy="243" r="4" fill="#8bc34a"/>
  <circle cx="423" cy="356" r="4" fill="#8bc34a"/>
  <circle cx="280" cy="356" r="4" fill="#8bc34a"/>
  <circle cx="253" cy="243" r="4" fill="#8bc34a"/>

  <!-- Legend -->
  <rect x="490" y="430" width="195" height="60" rx="5" fill="white" stroke="#ddd" stroke-width="1"/>
  <line x1="503" y1="448" x2="533" y2="448" stroke="#4fc3f7" stroke-width="2.5"/>
  <text x="540" y="452" font-family="Georgia, serif" font-size="12" fill="#333">Veo 3.1</text>
  <line x1="503" y1="468" x2="533" y2="468" stroke="#ef5350" stroke-width="2"/>
  <text x="540" y="472" font-family="Georgia, serif" font-size="12" fill="#333">Kling 3.0</text>
  <line x1="503" y1="488" x2="533" y2="488" stroke="#8bc34a" stroke-width="2"/>
  <text x="540" y="492" font-family="Georgia, serif" font-size="12" fill="#333">Wan 2.2</text>
</svg>

---

## Human Preference and ELO Scoring

Ultimately, the question is: do humans like the output? Automated metrics are proxies. Direct human evaluation is the ground truth. But human evaluation needs its own rigorous methodology.

### Pairwise Comparison

The most reliable human evaluation method: show two videos side by side and ask which is better. This is simpler and more consistent than absolute rating scales.

Given prompt $T$, generate video $A$ from Model 1 and video $B$ from Model 2. Ask human raters: "Which video better matches the prompt and looks more realistic?" Record the winner.

### ELO Rating System

The ELO rating system, originally designed for chess, provides a principled way to aggregate pairwise comparisons into a global ranking. This is the same system used by LMSYS Chatbot Arena for LLM evaluation.

**The ELO probability model:**

Given two models with ratings $R_A$ and $R_B$, the expected win probability for Model A is:

$$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

This is a logistic function centered at $R_A - R_B = 0$. When the ratings are equal, $E_A = 0.5$ (equal chance of winning). A rating difference of 400 corresponds to a 10:1 expected win ratio.

**Rating update after a match:**

$$R_A' = R_A + K \cdot (S_A - E_A)$$

where:
- $S_A = 1$ if A wins, $S_A = 0$ if A loses, $S_A = 0.5$ for a tie
- $K$ is the update factor (controls how much each match changes the rating)
- $E_A$ is the expected score computed above

**Properties:**
- ELO is a zero-sum system: $R_A' + R_B' = R_A + R_B$ (total rating is conserved)
- Ratings converge to a stable equilibrium after enough matches
- Rating differences have a consistent probabilistic interpretation

### Bradley-Terry Model

The Bradley-Terry model provides a maximum-likelihood alternative to iterative ELO updates. It models the probability of Model $i$ beating Model $j$ as:

$$P(i \text{ beats } j) = \frac{\pi_i}{\pi_i + \pi_j}$$

where $\pi_i > 0$ is the "strength" parameter of Model $i$. Taking $\pi_i = 10^{R_i / 400}$ recovers the ELO model.

Given $n$ pairwise comparisons, the maximum-likelihood estimates of $\{\pi_i\}$ are found by solving:

$$\sum_{j \neq i} \frac{w_{ij}}{\hat{\pi}_i} = \sum_{j \neq i} \frac{w_{ij} + w_{ji}}{\hat{\pi}_i + \hat{\pi}_j}$$

where $w_{ij}$ is the number of times Model $i$ beat Model $j$. This system of equations can be solved by iterative algorithms (Zermelo's algorithm, MM algorithm, or standard gradient descent).

### ELO for Video Models: Practical Considerations

Applying ELO to video model evaluation requires careful design:

**1. Prompt diversity.** Evaluate across a diverse set of prompts (action, scenery, people, text, abstract). A model that excels at landscapes but fails at human faces should not rank higher than a balanced model.

**2. Aspect evaluation.** Consider separate ELO leaderboards for:
- Overall quality
- Text-prompt adherence
- Temporal coherence
- Aesthetic appeal
- Motion naturalness

**3. Rater calibration.** Different human raters have different standards. Use multiple raters per comparison and aggregate (e.g., majority vote or weighted average based on rater agreement rates).

**4. Confidence intervals.** Report uncertainty in ELO ratings. With $N$ comparisons, the standard error of the rating is approximately:

$$\text{SE}(R) \approx \frac{400}{\sqrt{N}} \cdot \frac{1}{\sqrt{\text{avg. information per match}}}$$

For practical purposes, you need 200+ comparisons per model pair for stable rankings.

### Sample ELO Rankings (Illustrative)

| Model | ELO Rating | 95% CI | Matches |
|:--|:-:|:-:|:-:|
| Veo 3.1 | 1285 | +/- 18 | 1,450 |
| Kling 3.0 | 1240 | +/- 20 | 1,380 |
| Sora 2 | 1220 | +/- 22 | 1,200 |
| Runway Gen-4 | 1180 | +/- 25 | 980 |
| Wan 2.2 (14B) | 1120 | +/- 28 | 820 |
| LTX-2 | 1080 | +/- 30 | 750 |
| AnimateDiff | 1010 | +/- 35 | 600 |

*These are illustrative rankings based on general model capabilities, not from a specific published arena.*

The ELO difference between Veo 3.1 (1285) and Wan 2.2 (1120) is 165 points, which translates to an expected win rate for Veo of:

$$E_\text{Veo} = \frac{1}{1 + 10^{-165/400}} = \frac{1}{1 + 10^{-0.4125}} = \frac{1}{1 + 0.387} = 0.721$$

Veo 3.1 would be expected to win approximately 72% of head-to-head comparisons against Wan 2.2.

---

## Building a Metric Dashboard

If you operate a video generation platform, monitoring these metrics is essential for detecting regressions, comparing model updates, and making routing decisions.

### Offline Evaluation Pipeline

```python
# Pseudocode for an automated evaluation pipeline

class VideoEvaluationPipeline:
    def __init__(self):
        self.inception = load_inception_v3()
        self.i3d = load_i3d_kinetics400()
        self.clip_model = load_clip_vit_l14()

    def evaluate_model(self, model, prompts, reference_videos):
        """Run full evaluation suite on a model."""
        generated = [model.generate(p) for p in prompts]

        return {
            'fid': self.compute_fid(reference_videos, generated),
            'fvd': self.compute_fvd(reference_videos, generated),
            'clip_score': self.compute_clip_scores(generated, prompts),
            'temporal_consistency': self.compute_temporal_consistency(generated),
        }

    def compute_fid(self, real_videos, gen_videos):
        """Frame-level FID using InceptionV3."""
        real_features = []
        gen_features = []
        for v in real_videos:
            for frame in sample_frames(v, n=8):
                real_features.append(self.inception(frame))
        for v in gen_videos:
            for frame in sample_frames(v, n=8):
                gen_features.append(self.inception(frame))

        mu_r, sigma_r = compute_statistics(real_features)
        mu_g, sigma_g = compute_statistics(gen_features)
        return frechet_distance(mu_r, sigma_r, mu_g, sigma_g)

    def compute_fvd(self, real_videos, gen_videos):
        """Video-level FVD using I3D."""
        real_features = [self.i3d(clip_to_16_frames(v)) for v in real_videos]
        gen_features = [self.i3d(clip_to_16_frames(v)) for v in gen_videos]

        mu_r, sigma_r = compute_statistics(real_features)
        mu_g, sigma_g = compute_statistics(gen_features)
        return frechet_distance(mu_r, sigma_r, mu_g, sigma_g)

    def compute_clip_scores(self, videos, prompts):
        """Per-video CLIP score averaged across frames."""
        scores = []
        for video, prompt in zip(videos, prompts):
            frame_embeddings = [self.clip_model.encode_image(f)
                               for f in sample_frames(video, n=8)]
            avg_embedding = mean(frame_embeddings)
            text_embedding = self.clip_model.encode_text(prompt)
            scores.append(cosine_similarity(avg_embedding, text_embedding))
        return mean(scores)
```

### Key Monitoring Thresholds

| Metric | Green | Yellow | Red |
|:--|:-:|:-:|:-:|
| FID (frame) | < 10 | 10 -- 20 | > 20 |
| FVD | < 300 | 300 -- 500 | > 500 |
| CLIP Score | > 0.30 | 0.25 -- 0.30 | < 0.25 |
| Temporal consistency | > 0.95 | 0.90 -- 0.95 | < 0.90 |

When a metric crosses from green to yellow, investigate. When it crosses to red, something is broken --- model update regression, preprocessing bug, or infrastructure issue.

---

## The Future of Video Quality Metrics

Current metrics are imperfect. The field is actively developing better alternatives.

### Learned Perceptual Metrics

Instead of using feature spaces designed for classification (Inception, I3D), train a network specifically to predict human quality judgments. Models like LPIPS (Learned Perceptual Image Patch Similarity) do this for images. Video equivalents are emerging but not yet standardized.

### Video Quality Assessment (VQA) Models

Traditional video quality assessment (VMAF, SSIM, PSNR) measures degradation relative to a reference. These do not apply directly to generative models (there is no reference video to compare against). But no-reference VQA models that predict human quality ratings directly from the video are improving rapidly.

### Compositional Evaluation

Rather than a single score, evaluate specific aspects of generation quality:

- Object presence: Are all described objects present?
- Attribute binding: Does each object have the correct attributes?
- Spatial relations: Are objects in the correct relative positions?
- Action accuracy: Does the described action occur?
- Temporal ordering: Do events happen in the described order?

Benchmarks like T2V-CompBench decompose evaluation into these dimensions.

### Self-Supervised Video Features

Replace I3D with modern self-supervised video representations (VideoMAE, InternVideo) that capture richer spatiotemporal features. Early results suggest these improve FVD's correlation with human preference.

---

## Summary

The three core metrics each capture a different axis of quality:

| Metric | What it Measures | Formula Core | Higher/Lower = Better |
|:--|:--|:--|:-:|
| FID | Distributional realism | Frechet distance in Inception features | Lower |
| FVD | Spatiotemporal realism | Frechet distance in I3D features | Lower |
| CLIP Score | Text-prompt adherence | Cosine similarity in CLIP space | Higher |

No single metric is sufficient. FID/FVD tell you whether outputs look real but not whether they match the prompt. CLIP Score tells you whether outputs match the prompt but not whether they look real. The intersection of low FVD and high CLIP Score is the sweet spot.

For production systems, combine automated metrics with periodic human evaluation. Use automated metrics for fast iteration and regression detection. Use human evaluation (via ELO-style pairwise comparisons) for definitive model comparisons and quality validation.

The mathematical foundations --- Frechet distance, cosine similarity, logistic rating models --- are straightforward. The hard part is knowing when to trust the numbers and when to look at the actual outputs with your own eyes.

---

## References

1. Heusel, M. et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.* NeurIPS 2017.
2. Unterthiner, T. et al. (2019). *FVD: A new Metric for Video Generation.* ICLR 2019 Workshop.
3. Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021.
4. Carreira, J. & Zisserman, A. (2017). *Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset.* CVPR 2017.
5. Kynkaanniemi, T. et al. (2019). *Improved Precision and Recall Metric for Assessing Generative Models.* NeurIPS 2019.
6. Szegedy, C. et al. (2016). *Rethinking the Inception Architecture for Computer Vision.* CVPR 2016.
7. Bradley, R.A. & Terry, M.E. (1952). *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons.* Biometrika.
8. Elo, A.E. (1978). *The Rating of Chessplayers, Past and Present.* Arco Publishing.
