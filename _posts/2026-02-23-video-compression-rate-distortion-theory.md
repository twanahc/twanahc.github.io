---
layout: post
title: "Video Compression and Rate-Distortion Theory: From DCT to Learned Latent Spaces"
date: 2026-02-23
category: math
---

Every video generation model operates in compressed space. Stable Video Diffusion compresses video by 8× spatially and 4× temporally before denoising. Sora uses a spatiotemporal patchifier that reduces a 1080p video to a small grid of latent tokens. The reason is simple arithmetic: a 5-second 1080p video at 24fps is \(5 \times 24 \times 1920 \times 1080 \times 3 \approx 746\) million values. No diffusion model can operate on that directly.

This makes video compression not just a practical necessity but a theoretical foundation of video generation. The quality of the compression --- how much information is preserved at a given bit budget --- determines the ceiling for generation quality. A latent space that discards too much detail produces blurry output no matter how good the diffusion model is.

This post develops the theory of video compression from first principles. We start with Shannon's rate-distortion theory (the fundamental limits of lossy compression), derive the rate-distortion function for Gaussian sources, explain the Discrete Cosine Transform and why it is nearly optimal, cover quantization theory, build up to motion-compensated video coding (I/P/B frames), and then bridge to learned compression and the 3D VAE architectures used in modern video generation.

---

## Table of Contents

1. [Why Compression Matters for Video Generation](#why-compression-matters-for-video-generation)
2. [Rate-Distortion Theory: The Fundamental Limits](#rate-distortion-theory-the-fundamental-limits)
3. [The Rate-Distortion Function for Gaussian Sources](#the-rate-distortion-function-for-gaussian-sources)
4. [Transform Coding: The DCT](#transform-coding-the-dct)
5. [Quantization Theory](#quantization-theory)
6. [Motion Compensation: Exploiting Temporal Redundancy](#motion-compensation-exploiting-temporal-redundancy)
7. [The Modern Video Codec Pipeline](#the-modern-video-codec-pipeline)
8. [Learned Image Compression](#learned-image-compression)
9. [The 3D VAE in Video Diffusion Models](#the-3d-vae-in-video-diffusion-models)
10. [VQ-VAE and Discrete Latent Spaces](#vq-vae-and-discrete-latent-spaces)
11. [Python: DCT Compression and Rate-Distortion Curves](#python-dct-compression-and-rate-distortion-curves)

---

## Why Compression Matters for Video Generation

Consider the computational cost of running diffusion on raw pixels. A video of \(F\) frames at resolution \(H \times W\) with 3 color channels has \(3FHW\) values. For each denoising step, a neural network processes all of these. Self-attention, the workhorse of modern architectures, scales as \(O(N^2)\) where \(N\) is the number of tokens.

The numbers are stark:

| Resolution | Frames | Raw values | Typical latent (8× spatial, 4× temporal) | Compression ratio |
|-----------|--------|------------|-------------------------------------------|-------------------|
| 512×512 | 16 | 12.6M | 49K | 256× |
| 1024×1024 | 24 | 75.5M | 74K | 1024× |
| 1920×1080 | 120 | 746M | 730K | 1024× |

A 1024× compression ratio means the latent space represents the video with 1/1024th the values. The question becomes: how much information can you preserve at that compression ratio? This is precisely what rate-distortion theory answers.

---

## Rate-Distortion Theory: The Fundamental Limits

**Rate-distortion theory**, developed by Claude Shannon in 1959, establishes the fundamental tradeoff between compression rate (bits per source symbol) and reconstruction quality (distortion).

### Setup

Let \(X\) be a source random variable with distribution \(p(x)\) (e.g., pixel values). A **lossy compression scheme** consists of:
- An **encoder** that maps a sequence of source symbols \(x^n = (x_1, \ldots, x_n)\) to a compressed representation using \(nR\) bits total (rate \(R\) bits per symbol)
- A **decoder** that produces a reconstruction \(\hat{x}^n\) from the compressed representation

The **distortion** is measured by a per-symbol distortion function \(d(x, \hat{x})\), averaged over the sequence:

$$D = \frac{1}{n} \sum_{i=1}^n \mathbb{E}[d(X_i, \hat{X}_i)]$$

Common choices: **squared error** \(d(x, \hat{x}) = (x - \hat{x})^2\) (MSE) and **absolute error** \(d(x, \hat{x}) = |x - \hat{x}|\).

### The Rate-Distortion Function

The **rate-distortion function** \(R(D)\) is the minimum rate required to achieve distortion at most \(D\):

$$R(D) = \min_{p(\hat{x}|x): \, \mathbb{E}[d(X,\hat{X})] \leq D} I(X; \hat{X})$$

where \(I(X; \hat{X})\) is the **mutual information** between the source and reconstruction. The minimization is over all conditional distributions \(p(\hat{x}|x)\) (all possible encodings) that achieve the target distortion.

Properties of \(R(D)\):
- \(R(D)\) is a convex, non-increasing function of \(D\)
- \(R(0) = H(X)\) for discrete sources (perfect reconstruction requires full entropy)
- \(R(D_{\max}) = 0\) where \(D_{\max}\) is the distortion achieved by ignoring the source entirely

Shannon's theorem says: any compression scheme achieving distortion \(D\) requires rate at least \(R(D)\), and there exist schemes that achieve \(R(D) + \epsilon\) for any \(\epsilon > 0\) (in the limit of long block lengths).

<svg viewBox="0 0 700 350" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <defs>
    <marker id="arrowRD" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#d4d4d4"/>
    </marker>
  </defs>
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">Rate-Distortion Curve</text>

  <!-- Axes -->
  <line x1="100" y1="300" x2="620" y2="300" stroke="#d4d4d4" stroke-width="2" marker-end="url(#arrowRD)"/>
  <line x1="100" y1="300" x2="100" y2="50" stroke="#d4d4d4" stroke-width="2" marker-end="url(#arrowRD)"/>
  <text x="370" y="335" text-anchor="middle" font-size="12" fill="#d4d4d4">Distortion D</text>
  <text x="50" y="175" text-anchor="middle" font-size="12" fill="#d4d4d4" transform="rotate(-90 50 175)">Rate R (bits/symbol)</text>

  <!-- R(D) curve -->
  <path d="M 100,70 C 130,75 160,90 200,120 S 300,200 400,250 S 500,280 580,290" fill="none" stroke="#4fc3f7" stroke-width="3"/>
  <text x="180" y="100" font-size="11" fill="#4fc3f7" font-weight="bold">R(D)</text>

  <!-- Achievable region -->
  <text x="450" y="150" font-size="11" fill="#66bb6a">Achievable</text>
  <text x="450" y="168" font-size="11" fill="#66bb6a">(R, D) pairs</text>
  <text x="200" y="250" font-size="11" fill="#E53935">Impossible</text>
  <text x="200" y="268" font-size="11" fill="#E53935">region</text>

  <!-- Operating points -->
  <circle cx="160" cy="108" r="6" fill="#FF9800" stroke="#FF9800" stroke-width="1"/>
  <text x="170" y="95" font-size="10" fill="#FF9800">High quality (low D, high R)</text>

  <circle cx="400" cy="250" r="6" fill="#CE93D8" stroke="#CE93D8" stroke-width="1"/>
  <text x="415" y="245" font-size="10" fill="#CE93D8">Low quality (high D, low R)</text>

  <!-- D_max -->
  <line x1="580" y1="290" x2="580" y2="300" stroke="#999" stroke-width="1"/>
  <text x="580" y="315" text-anchor="middle" font-size="10" fill="#999">D_max</text>
</svg>

---

## The Rate-Distortion Function for Gaussian Sources

For a Gaussian source \(X \sim \mathcal{N}(0, \sigma^2)\) with MSE distortion, the rate-distortion function has a beautiful closed form.

### Derivation

We seek the conditional distribution \(p(\hat{x}|x)\) that minimizes \(I(X; \hat{X})\) subject to \(\mathbb{E}[(X - \hat{X})^2] \leq D\).

The key insight: the optimal reconstruction is \(\hat{X} = X + Z\) where \(Z\) is independent Gaussian noise. More precisely, the optimal "test channel" is:

$$\hat{X} = X + Z, \quad Z \sim \mathcal{N}(0, D), \quad Z \perp X$$

Wait --- that adds noise, making things worse! The trick is that we are optimizing over the full joint distribution. The optimal scheme actually sends a coarsened version of \(X\):

$$\hat{X} = X - N = X - \text{(quantization noise)}$$

where the quantization noise \(N\) has variance \(D\). The mutual information is:

$$I(X; \hat{X}) = h(X) - h(X | \hat{X})$$

Since \(X\) is Gaussian with variance \(\sigma^2\):

$$h(X) = \frac{1}{2}\log(2\pi e \sigma^2)$$

Given \(\hat{X}\), the reconstruction error \(X - \hat{X}\) has variance \(D\), and by the optimality of the Gaussian, \(X | \hat{X}\) is Gaussian with variance \(D\):

$$h(X | \hat{X}) = \frac{1}{2}\log(2\pi e D)$$

Therefore:

$$R(D) = \frac{1}{2}\log\frac{\sigma^2}{D} \quad \text{bits (using log base 2)}$$

This is valid for \(0 \leq D \leq \sigma^2\). At \(D = 0\) (perfect reconstruction), \(R = \infty\) --- you need infinite bits to represent a continuous variable exactly. At \(D = \sigma^2\), \(R = 0\) --- the best zero-rate reconstruction is just the mean (zero).

Equivalently, solving for \(D\) as a function of \(R\):

$$D(R) = \sigma^2 \cdot 2^{-2R}$$

Every additional bit per symbol halves the MSE. This is a fundamental law of lossy compression.

### The Water-Filling Interpretation

For a vector Gaussian source with independent components \(X_i \sim \mathcal{N}(0, \sigma_i^2)\), the optimal rate allocation is given by **reverse water-filling**: allocate rate to component \(i\) as:

$$R_i = \max\left(0, \frac{1}{2}\log\frac{\sigma_i^2}{\theta}\right)$$

where \(\theta\) is chosen so that \(\sum R_i = R_{\text{total}}\). Components with variance below the "water level" \(\theta\) get zero rate (they are reconstructed as zero). High-variance components get more bits.

This is exactly what transform coding does: transform to a basis where components are decorrelated, then allocate bits proportional to log-variance.

---

## Transform Coding: The DCT

Natural images have strong spatial correlations --- neighboring pixels are similar. **Transform coding** exploits this by transforming the image to a domain where the energy is concentrated in a few coefficients, then keeping only those coefficients.

### The Discrete Cosine Transform

The **DCT** of a 1D signal \(x[n]\) for \(n = 0, 1, \ldots, N-1\) is:

$$X[k] = \sqrt{\frac{2}{N}} \, c_k \sum_{n=0}^{N-1} x[n] \cos\!\left(\frac{\pi(2n+1)k}{2N}\right)$$

where \(c_0 = 1/\sqrt{2}\) and \(c_k = 1\) for \(k > 0\). The DCT basis functions are cosines at integer multiples of a base frequency, with even symmetry.

### Why the DCT?

The DCT is nearly optimal for natural images because of the **Karhunen-Loeve theorem**: the optimal transform (in the MSE sense) for a stationary random process is the eigenbasis of its covariance matrix. For a first-order Markov process \(x[n] = \rho \, x[n-1] + w[n]\) (a reasonable model for rows of natural images, with \(\rho \approx 0.95\)), the eigenbasis approaches the DCT as the block size grows.

The 2D DCT (used in JPEG) is the tensor product of 1D DCTs applied to \(8 \times 8\) blocks:

$$X[k_1, k_2] = \sum_{n_1=0}^{7} \sum_{n_2=0}^{7} x[n_1, n_2] \cdot B_{k_1}[n_1] \cdot B_{k_2}[n_2]$$

where \(B_k[n]\) are the 1D DCT basis functions.

The **energy compaction** property means that for natural images, most of the energy is in the low-frequency DCT coefficients (top-left corner of the \(8 \times 8\) block). The high-frequency coefficients are small and can be quantized coarsely or discarded.

---

## Quantization Theory

After transforming, the DCT coefficients are **quantized** --- mapped from continuous values to a discrete set. This is the lossy step: information is irreversibly discarded.

### Scalar Quantization

A scalar quantizer maps a continuous value \(x\) to the nearest point in a finite set \(\{q_1, \ldots, q_K\}\) (reconstruction levels). The set \(\{q_i\}\) and the decision boundaries between them are the quantizer's design parameters.

### Lloyd-Max Optimal Quantizer

Given a source distribution \(p(x)\) and a target number of levels \(K\), the **Lloyd-Max algorithm** finds the optimal quantizer by alternating:

1. **Nearest-neighbor assignment:** Given reconstruction levels \(\{q_i\}\), the optimal decision boundaries are the midpoints: \(b_i = (q_i + q_{i+1})/2\).

2. **Centroid condition:** Given decision regions \([b_{i-1}, b_i)\), the optimal reconstruction level is the conditional mean:

$$q_i = \frac{\int_{b_{i-1}}^{b_i} x \, p(x) \, dx}{\int_{b_{i-1}}^{b_i} p(x) \, dx} = \mathbb{E}[X | b_{i-1} \leq X < b_i]$$

This is the quantization analogue of K-means clustering. The algorithm converges to a local optimum.

For a Gaussian source with \(K\) levels, the MSE of the optimal quantizer scales as:

$$D \approx \frac{\sqrt{3} \pi}{2} \sigma^2 \cdot 2^{-2R}$$

where \(R = \log_2 K\) bits. The factor \(\frac{\sqrt{3}\pi}{2} \approx 2.72\) is the **quantization penalty** --- the gap between actual quantization and the rate-distortion bound. This gap motivates vector quantization.

### Vector Quantization

**Vector quantization (VQ)** quantizes vectors jointly instead of element-by-element. A codebook of \(K\) codewords \(\{\mathbf{q}_1, \ldots, \mathbf{q}_K\}\) in \(\mathbb{R}^d\) partitions the space into \(K\) **Voronoi cells**. Each input vector is mapped to the nearest codeword.

VQ can close the gap to the rate-distortion bound as the vector dimension \(d \to \infty\) (Shannon's lossy source coding theorem). In practice, VQ with \(d = 8\)--\(16\) already provides significant gains over scalar quantization. The VQ-VAE (covered later) learns the codebook end-to-end as part of a neural network.

---

## Motion Compensation: Exploiting Temporal Redundancy

Images are spatially redundant; video is also **temporally** redundant. Consecutive frames are mostly the same --- only the moving parts change. Video codecs exploit this via **motion compensation**.

### I, P, B Frames

A video is divided into **Groups of Pictures (GOPs)**, each containing three types of frames:

- **I-frame** (Intra): Encoded independently, like a JPEG image. The anchor point. Larger but self-contained.
- **P-frame** (Predicted): Encoded as the difference from a previous reference frame. For each block, find the best-matching block in the reference (motion estimation), transmit the **motion vector** and the **residual** (prediction error).
- **B-frame** (Bidirectional): Encoded using both a past and future reference frame, with motion vectors pointing in both directions.

A typical GOP structure: `I B B P B B P B B P B B I ...`

### Motion Estimation

For each \(16 \times 16\) (or variable-size) block in the current frame, search the reference frame for the best match. The displacement is the **motion vector** \(\mathbf{v} = (v_x, v_y)\). The search minimizes the **sum of absolute differences (SAD)**:

$$\text{SAD}(\mathbf{v}) = \sum_{(i,j) \in \text{block}} |I_{\text{current}}(i, j) - I_{\text{ref}}(i + v_x, j + v_y)|$$

The residual is then: \(R(i, j) = I_{\text{current}}(i, j) - I_{\text{ref}}(i + v_x, j + v_y)\). Because the motion-compensated prediction is good, the residual has much less energy than the original block. DCT + quantize + entropy code the residual, transmitting far fewer bits than an I-frame.

Sub-pixel motion estimation (half-pixel, quarter-pixel) interpolates the reference frame and searches at fractional positions, capturing motion more precisely.

<svg viewBox="0 0 700 250" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">GOP Structure: I, P, B Frames</text>

  <!-- Frames -->
  <rect x="30" y="70" width="60" height="80" rx="4" fill="#E53935" opacity="0.3" stroke="#E53935" stroke-width="2"/>
  <text x="60" y="115" text-anchor="middle" font-size="14" fill="#E53935" font-weight="bold">I</text>
  <text x="60" y="170" text-anchor="middle" font-size="9" fill="#999">Key frame</text>

  <rect x="110" y="70" width="60" height="80" rx="4" fill="#2196F3" opacity="0.2" stroke="#2196F3" stroke-width="1.5"/>
  <text x="140" y="115" text-anchor="middle" font-size="14" fill="#2196F3" font-weight="bold">B</text>

  <rect x="190" y="70" width="60" height="80" rx="4" fill="#2196F3" opacity="0.2" stroke="#2196F3" stroke-width="1.5"/>
  <text x="220" y="115" text-anchor="middle" font-size="14" fill="#2196F3" font-weight="bold">B</text>

  <rect x="270" y="70" width="60" height="80" rx="4" fill="#66bb6a" opacity="0.2" stroke="#66bb6a" stroke-width="1.5"/>
  <text x="300" y="115" text-anchor="middle" font-size="14" fill="#66bb6a" font-weight="bold">P</text>

  <rect x="350" y="70" width="60" height="80" rx="4" fill="#2196F3" opacity="0.2" stroke="#2196F3" stroke-width="1.5"/>
  <text x="380" y="115" text-anchor="middle" font-size="14" fill="#2196F3" font-weight="bold">B</text>

  <rect x="430" y="70" width="60" height="80" rx="4" fill="#2196F3" opacity="0.2" stroke="#2196F3" stroke-width="1.5"/>
  <text x="460" y="115" text-anchor="middle" font-size="14" fill="#2196F3" font-weight="bold">B</text>

  <rect x="510" y="70" width="60" height="80" rx="4" fill="#66bb6a" opacity="0.2" stroke="#66bb6a" stroke-width="1.5"/>
  <text x="540" y="115" text-anchor="middle" font-size="14" fill="#66bb6a" font-weight="bold">P</text>

  <rect x="590" y="70" width="60" height="80" rx="4" fill="#E53935" opacity="0.3" stroke="#E53935" stroke-width="2"/>
  <text x="620" y="115" text-anchor="middle" font-size="14" fill="#E53935" font-weight="bold">I</text>
  <text x="620" y="170" text-anchor="middle" font-size="9" fill="#999">Key frame</text>

  <!-- Reference arrows -->
  <path d="M 300,70 Q 165,40 60,70" fill="none" stroke="#66bb6a" stroke-width="1" stroke-dasharray="4,3"/>
  <path d="M 540,70 Q 420,40 300,70" fill="none" stroke="#66bb6a" stroke-width="1" stroke-dasharray="4,3"/>
  <path d="M 140,70 Q 100,50 60,70" fill="none" stroke="#2196F3" stroke-width="1" stroke-dasharray="3,2"/>
  <path d="M 140,70 Q 220,45 300,70" fill="none" stroke="#2196F3" stroke-width="1" stroke-dasharray="3,2"/>

  <!-- Size comparison -->
  <text x="350" y="210" text-anchor="middle" font-size="10" fill="#d4d4d4">Typical sizes: I ≈ 100KB, P ≈ 30KB, B ≈ 15KB</text>
  <text x="350" y="230" text-anchor="middle" font-size="10" fill="#999">B-frames reference both past and future → smallest</text>
</svg>

---

## The Modern Video Codec Pipeline

Modern codecs (H.264/AVC, H.265/HEVC, AV1) follow the same conceptual pipeline with increasingly sophisticated tools:

1. **Block partitioning:** Divide each frame into blocks (CUs in HEVC: 64×64 down to 4×4, with flexible quad-tree + binary-tree partitioning)
2. **Prediction:** Intra-prediction (predict from neighboring already-decoded blocks within the same frame, using angular modes) or inter-prediction (motion-compensated prediction from reference frames)
3. **Transform:** DCT (or DST for small intra blocks) of the prediction residual
4. **Quantization:** Scalar quantization of transform coefficients, controlled by a quantization parameter (QP)
5. **Entropy coding:** Lossless compression of quantized coefficients and side information (motion vectors, modes) using CABAC (Context-Adaptive Binary Arithmetic Coding)

The quantization parameter QP controls the rate-distortion tradeoff: higher QP = coarser quantization = smaller file = more distortion.

---

## Learned Image Compression

Neural compression replaces the hand-crafted DCT + quantization + entropy coding pipeline with a learned autoencoder.

### The Architecture (Balle et al., 2018)

An **encoder** \(f_a(\mathbf{x})\) maps an image \(\mathbf{x}\) to a latent representation \(\mathbf{y} = f_a(\mathbf{x})\). The latents are **quantized**: \(\hat{\mathbf{y}} = \lfloor \mathbf{y} \rceil\) (round to nearest integer). A **decoder** \(f_s(\hat{\mathbf{y}})\) reconstructs the image: \(\hat{\mathbf{x}} = f_s(\hat{\mathbf{y}})\).

The training loss is the **rate-distortion Lagrangian**:

$$\mathcal{L} = \underbrace{-\log p_{\hat{\mathbf{y}}}(\hat{\mathbf{y}})}_{\text{rate (bits)}} + \lambda \underbrace{d(\mathbf{x}, \hat{\mathbf{x}})}_{\text{distortion}}$$

The rate term is the negative log-likelihood of the quantized latents under a learned **entropy model** \(p_{\hat{\mathbf{y}}}\). Minimizing this term encourages the encoder to produce latents that are easy to compress (predictable, low-entropy).

The distortion term can be MSE, MS-SSIM, or a perceptual loss. The Lagrange multiplier \(\lambda\) sweeps the rate-distortion tradeoff: large \(\lambda\) favors low distortion (high quality, high rate), small \(\lambda\) favors low rate (low quality, small file).

### The Hyperprior

A **hyperprior** (Balle et al., 2018) captures dependencies between latent elements. An additional small encoder \(h_a(\mathbf{y})\) extracts **hyper-latents** \(\mathbf{z}\) that parameterize the entropy model for \(\mathbf{y}\):

$$p_{\hat{\mathbf{y}}}(\hat{\mathbf{y}} | \hat{\mathbf{z}}) = \prod_i \left(\mathcal{N}(\mu_i(\hat{\mathbf{z}}), \sigma_i^2(\hat{\mathbf{z}})) * \mathcal{U}(-\frac{1}{2}, \frac{1}{2})\right)(\hat{y}_i)$$

Each latent element has its own predicted mean \(\mu_i\) and variance \(\sigma_i^2\) from the hyperprior, yielding a more accurate entropy model and lower bit rates.

The result: learned compression matches or exceeds traditional codecs (JPEG, BPG/HEVC-intra) at the same bit rate, with fewer visual artifacts.

---

## The 3D VAE in Video Diffusion Models

Video diffusion models do not learn compression for file size reduction. They learn it to make generation tractable. The **3D VAE** (variational autoencoder with 3D convolutions) compresses video into a compact latent space where diffusion operates.

### Architecture

The encoder takes raw video \(\mathbf{x} \in \mathbb{R}^{F \times H \times W \times 3}\) and produces latents \(\mathbf{z} \in \mathbb{R}^{F' \times H' \times W' \times C}\), where:

- Spatial downsampling: \(H' = H/f_s\), \(W' = W/f_s\) with \(f_s = 8\) typically
- Temporal downsampling: \(F' = F/f_t\) with \(f_t = 4\) typically
- Channel expansion: \(C = 4\) or \(16\) (more channels preserve more information per spatial location)

The encoder uses 3D convolutions (or factored 2D spatial + 1D temporal convolutions) with strided downsampling. The decoder mirrors this with transposed convolutions or sub-pixel upsampling.

### Why Quality Matters

The VAE is trained separately from the diffusion model (in most architectures). Its reconstruction quality is the **ceiling** for generation quality. If the VAE cannot faithfully reconstruct a sharp edge or a subtle color gradient, no diffusion model operating in that latent space can produce it.

This is why recent models invest heavily in VAE quality:
- **Stable Diffusion 3** uses a 16-channel VAE (up from 4 in SD 1.x) for higher fidelity
- **Sora** uses a "spacetime patchifier" --- essentially a learned video tokenizer with large compression ratios
- **CogVideoX** uses a causal 3D VAE that encodes frames sequentially, maintaining temporal coherence

### The Training Objective

The VAE is trained with:

$$\mathcal{L}_{\text{VAE}} = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|_1}_{\text{reconstruction}} + \lambda_{\text{perc}} \underbrace{\mathcal{L}_{\text{perceptual}}(\mathbf{x}, \hat{\mathbf{x}})}_{\text{VGG features}} + \lambda_{\text{adv}} \underbrace{\mathcal{L}_{\text{GAN}}(\hat{\mathbf{x}})}_{\text{adversarial}} + \lambda_{\text{KL}} \underbrace{D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{regularization}}$$

The KL term regularizes the latent distribution toward a standard Gaussian, which is the noise distribution used by diffusion. The adversarial term pushes reconstructions to be perceptually sharp (avoiding the blurriness inherent in pure MSE/L1 training).

---

## VQ-VAE and Discrete Latent Spaces

An alternative to continuous latent spaces is **discrete tokenization** via VQ-VAE.

### Vector Quantization in the Latent Space

Instead of a continuous \(\mathbf{z}\), the encoder output is quantized to the nearest entry in a learned codebook \(\mathcal{C} = \{\mathbf{e}_1, \ldots, \mathbf{e}_K\}\):

$$\mathbf{z}_q = \mathbf{e}_k \quad \text{where} \quad k = \arg\min_j \|\mathbf{z}_e - \mathbf{e}_j\|_2$$

The argmin is not differentiable, so gradients are passed through using the **straight-through estimator**: during backpropagation, the gradient of the output with respect to the input of the quantization step is simply the identity. The encoder receives the gradient as if quantization had not happened.

### The VQ-VAE Loss

$$\mathcal{L}_{\text{VQ}} = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2}_{\text{reconstruction}} + \underbrace{\|\text{sg}[\mathbf{z}_e] - \mathbf{e}_k\|_2^2}_{\text{codebook}} + \beta \underbrace{\|\mathbf{z}_e - \text{sg}[\mathbf{e}_k]\|_2^2}_{\text{commitment}}$$

where \(\text{sg}[\cdot]\) is the stop-gradient operator. The codebook loss moves the codebook vectors toward the encoder outputs. The commitment loss (with \(\beta \approx 0.25\)) prevents the encoder from oscillating by penalizing encoder outputs that are far from their assigned codeword.

### Finite Scalar Quantization (FSQ)

A recent simplification (Mentzer et al., 2023) replaces vector quantization with per-dimension scalar quantization: round each dimension of the latent to one of \(L\) fixed levels (e.g., \(L = 5\)). With a \(d\)-dimensional latent, the effective codebook size is \(L^d\) without needing to manage codebook vectors, avoiding collapse entirely.

---

## Python: DCT Compression and Rate-Distortion Curves

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn

def create_test_image(size=256):
    """Create a test image with edges, textures, and smooth gradients."""
    y, x = np.mgrid[0:size, 0:size].astype(float) / size
    # Smooth gradient
    img = 0.3 * x + 0.2 * y
    # Edges (rectangles)
    img += 0.5 * ((x > 0.2) & (x < 0.6) & (y > 0.3) & (y < 0.7)).astype(float)
    # Texture (high frequency)
    img += 0.1 * np.sin(30 * np.pi * x) * np.sin(25 * np.pi * y) * (x > 0.5)
    # Gaussian blob
    img += 0.4 * np.exp(-((x - 0.7)**2 + (y - 0.2)**2) / 0.01)
    return np.clip(img, 0, 1)

def dct_compress(image, quality):
    """DCT compression with given quality (0-100, higher = better)."""
    h, w = image.shape
    block_size = 8
    reconstructed = np.zeros_like(image)

    # Standard JPEG quantization matrix (luminance)
    Q_base = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68,109,103, 77],
        [24, 35, 55, 64, 81,104,113, 92],
        [49, 64, 78, 87,103,121,120,101],
        [72, 92, 95, 98,112,100,103, 99]
    ], dtype=float)

    # Scale quantization matrix by quality
    if quality < 50:
        scale = 5000.0 / quality
    else:
        scale = 200.0 - 2.0 * quality
    Q = np.clip(np.floor((Q_base * scale + 50) / 100), 1, 255)

    total_nonzero = 0
    total_coeffs = 0

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = image[i:i+block_size, j:j+block_size]
            # Forward DCT
            dct_block = dctn(block - 0.5, type=2, norm='ortho')
            # Quantize
            quantized = np.round(dct_block / Q) * Q
            # Inverse DCT
            reconstructed[i:i+block_size, j:j+block_size] = idctn(quantized, type=2, norm='ortho') + 0.5

            total_nonzero += np.count_nonzero(np.round(dct_block / Q))
            total_coeffs += block_size * block_size

    reconstructed = np.clip(reconstructed, 0, 1)
    rate = total_nonzero / total_coeffs  # approximate rate (fraction of nonzero coefficients)
    mse = np.mean((image - reconstructed)**2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))

    return reconstructed, rate, mse, psnr

# Generate test image
image = create_test_image(256)

# Compute R-D curve
qualities = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95]
rates, distortions, psnrs = [], [], []
reconstructions = {}

for q in qualities:
    recon, rate, mse, psnr = dct_compress(image, q)
    rates.append(rate)
    distortions.append(mse)
    psnrs.append(psnr)
    if q in [10, 50, 90]:
        reconstructions[q] = recon

# Gaussian R(D) theoretical bound
sigma2 = np.var(image)
D_theory = np.linspace(1e-5, sigma2 * 0.5, 100)
R_theory = 0.5 * np.log2(sigma2 / D_theory)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original
axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

# Low quality reconstruction
axes[0, 1].imshow(reconstructions[10], cmap='gray', vmin=0, vmax=1)
axes[0, 1].set_title('Quality = 10 (heavy compression)')
axes[0, 1].axis('off')

# High quality reconstruction
axes[0, 2].imshow(reconstructions[90], cmap='gray', vmin=0, vmax=1)
axes[0, 2].set_title('Quality = 90 (light compression)')
axes[0, 2].axis('off')

# Rate-Distortion curve
axes[1, 0].plot(distortions, rates, 'o-', color='#4fc3f7', linewidth=2, markersize=5, label='DCT (empirical)')
axes[1, 0].plot(D_theory, R_theory, '--', color='#E53935', linewidth=1.5, label=r'Gaussian $R(D) = \frac{1}{2}\log_2(\sigma^2/D)$')
axes[1, 0].set_xlabel(r'Distortion $D$ (MSE)')
axes[1, 0].set_ylabel(r'Rate $R$ (nonzero fraction)')
axes[1, 0].set_title('Rate-Distortion Curve')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# PSNR vs quality
axes[1, 1].plot(qualities, psnrs, 's-', color='#66bb6a', linewidth=2, markersize=5)
axes[1, 1].set_xlabel('Quality Parameter')
axes[1, 1].set_ylabel('PSNR (dB)')
axes[1, 1].set_title('PSNR vs Quality')
axes[1, 1].grid(True, alpha=0.3)

# DCT basis functions
dct_basis = np.zeros((64, 64))
for k1 in range(8):
    for k2 in range(8):
        basis = np.zeros((8, 8))
        basis[k1, k2] = 1.0
        patch = idctn(basis, type=2, norm='ortho')
        r, c = k1 * 8, k2 * 8
        dct_basis[r:r+8, c:c+8] = patch

axes[1, 2].imshow(dct_basis, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
axes[1, 2].set_title(r'8$\times$8 DCT Basis Functions')
axes[1, 2].set_xlabel(r'Horizontal frequency $k_2$')
axes[1, 2].set_ylabel(r'Vertical frequency $k_1$')
# Add grid lines
for i in range(1, 8):
    axes[1, 2].axhline(y=i*8-0.5, color='black', linewidth=0.5)
    axes[1, 2].axvline(x=i*8-0.5, color='black', linewidth=0.5)

plt.suptitle('Video Compression: DCT and Rate-Distortion Theory', fontsize=14)
plt.tight_layout()
plt.savefig('video_compression_rd.png', dpi=150, bbox_inches='tight')
plt.show()
```

The deep connection between classical video compression and modern video generation is this: both must solve the same problem --- represent high-dimensional spatiotemporal data in a compact form. Classical codecs use hand-crafted transforms (DCT) and prediction (motion compensation). Learned models use neural autoencoders trained end-to-end. But the theoretical limits are the same, set by rate-distortion theory. The 3D VAE in your favorite video model is, at its core, a learned video codec. Its quality determines everything downstream.
