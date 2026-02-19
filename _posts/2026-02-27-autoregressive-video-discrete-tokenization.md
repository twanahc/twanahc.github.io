---
layout: post
title: "Autoregressive Video Generation and Discrete Tokenization: From VQ-VAE to Next-Token Video Models"
date: 2026-02-27
category: math
---

Language models generate text one token at a time, left to right, by modeling the joint distribution as a product of conditionals: \(p(x_1, x_2, \ldots, x_n) = \prod_{i=1}^n p(x_i | x_{<i})\). This is exact --- no approximation, no variational bound, just the chain rule of probability. The question is: can we do the same for video?

The answer is yes, with a catch. Video is continuous (pixel values are real numbers), but autoregressive models work naturally with discrete tokens. The bridge is **vector quantization**: learn a codebook that maps continuous visual patches to discrete tokens, then model the token sequence autoregressively. This is the VQ-VAE → VQ-GAN → autoregressive generation pipeline, and it represents a fundamentally different approach to video generation than diffusion.

This post builds the theory of discrete video tokenization from scratch. We derive the VQ-VAE objective, explain the straight-through estimator, analyze codebook collapse and its solutions, introduce Finite Scalar Quantization, extend to video tokenization, and compare autoregressive and diffusion approaches for video generation.

---

## Table of Contents

1. [Autoregressive Models: The Chain Rule](#autoregressive-models-the-chain-rule)
2. [Vector Quantization: Discretizing Continuous Spaces](#vector-quantization-discretizing-continuous-spaces)
3. [VQ-VAE: Variational Autoencoders with Discrete Latents](#vq-vae-variational-autoencoders-with-discrete-latents)
4. [VQ-GAN: Adversarial Training for Sharp Reconstruction](#vq-gan-adversarial-training-for-sharp-reconstruction)
5. [Codebook Collapse and Solutions](#codebook-collapse-and-solutions)
6. [Finite Scalar Quantization](#finite-scalar-quantization)
7. [Residual Quantization](#residual-quantization)
8. [Tokenizing Video](#tokenizing-video)
9. [Autoregressive Generation of Video Tokens](#autoregressive-generation-of-video-tokens)
10. [Diffusion vs Autoregressive: The Fundamental Tradeoff](#diffusion-vs-autoregressive-the-fundamental-tradeoff)
11. [Python: Vector Quantization from Scratch](#python-vector-quantization-from-scratch)

---

## Autoregressive Models: The Chain Rule

The chain rule of probability factorizes any joint distribution into a product of conditionals:

$$p(\mathbf{x}) = p(x_1, x_2, \ldots, x_n) = p(x_1) \prod_{i=2}^{n} p(x_i | x_1, \ldots, x_{i-1})$$

This is not an approximation --- it is an identity, valid for any distribution. The power of autoregressive models is that they parameterize each conditional \(p(x_i | x_{<i})\) with a neural network (typically a Transformer) and train by maximizing the log-likelihood:

$$\mathcal{L} = -\sum_{i=1}^{n} \log p_\theta(x_i | x_{<i})$$

This is the **negative log-likelihood (NLL)** or cross-entropy loss. For discrete tokens with vocabulary size \(V\), each conditional is a softmax over \(V\) classes. The model outputs logits, the softmax converts to probabilities, and the loss is the negative log of the probability assigned to the correct token.

**Teacher forcing:** During training, the model receives the ground-truth tokens \(x_{<i}\) as input (not its own predictions). This is exact maximum likelihood estimation and avoids exposure bias.

**Generation:** At inference, the model generates one token at a time, feeding its own predictions back as input. This is inherently sequential --- each token depends on all previous tokens.

---

## Vector Quantization: Discretizing Continuous Spaces

To apply autoregressive models to images and video, we need to convert continuous pixel values to discrete tokens. **Vector quantization (VQ)** does this by maintaining a **codebook** \(\mathcal{C} = \{\mathbf{e}_1, \ldots, \mathbf{e}_K\} \subset \mathbb{R}^d\) of \(K\) prototype vectors (codewords) in a \(d\)-dimensional space.

Given a continuous vector \(\mathbf{z} \in \mathbb{R}^d\), the quantization operation maps it to the nearest codeword:

$$q(\mathbf{z}) = \mathbf{e}_k \quad \text{where} \quad k = \arg\min_j \|\mathbf{z} - \mathbf{e}_j\|_2$$

This partitions \(\mathbb{R}^d\) into \(K\) **Voronoi cells**, one per codeword. Each cell contains all points closer to that codeword than to any other.

The quantization is lossy: \(q(\mathbf{z}) \neq \mathbf{z}\) in general. The quantization error is \(\|\mathbf{z} - q(\mathbf{z})\|_2\). More codewords (larger \(K\)) and higher dimensions (larger \(d\)) reduce the error, at the cost of a larger codebook.

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">Vector Quantization: Voronoi Cells and Codebook</text>

  <!-- Voronoi-like cells -->
  <polygon points="100,50 250,60 280,150 200,200 80,170" fill="#2196F3" opacity="0.08" stroke="#2196F3" stroke-width="1"/>
  <polygon points="250,60 400,50 420,140 280,150" fill="#66bb6a" opacity="0.08" stroke="#66bb6a" stroke-width="1"/>
  <polygon points="400,50 550,70 530,180 420,140" fill="#FF9800" opacity="0.08" stroke="#FF9800" stroke-width="1"/>
  <polygon points="80,170 200,200 180,260 60,250" fill="#CE93D8" opacity="0.08" stroke="#CE93D8" stroke-width="1"/>
  <polygon points="200,200 280,150 420,140 400,250 180,260" fill="#E53935" opacity="0.08" stroke="#E53935" stroke-width="1"/>
  <polygon points="420,140 530,180 520,260 400,250" fill="#4fc3f7" opacity="0.08" stroke="#4fc3f7" stroke-width="1"/>

  <!-- Codebook vectors (centers) -->
  <circle cx="180" cy="130" r="6" fill="#2196F3"/>
  <circle cx="340" cy="100" r="6" fill="#66bb6a"/>
  <circle cx="480" cy="120" r="6" fill="#FF9800"/>
  <circle cx="140" cy="210" r="6" fill="#CE93D8"/>
  <circle cx="330" cy="200" r="6" fill="#E53935"/>
  <circle cx="470" cy="210" r="6" fill="#4fc3f7"/>

  <!-- Labels -->
  <text x="180" y="118" text-anchor="middle" font-size="9" fill="#2196F3">e₁</text>
  <text x="340" y="88" text-anchor="middle" font-size="9" fill="#66bb6a">e₂</text>
  <text x="480" y="108" text-anchor="middle" font-size="9" fill="#FF9800">e₃</text>
  <text x="140" y="230" text-anchor="middle" font-size="9" fill="#CE93D8">e₄</text>
  <text x="330" y="220" text-anchor="middle" font-size="9" fill="#E53935">e₅</text>
  <text x="470" y="230" text-anchor="middle" font-size="9" fill="#4fc3f7">e₆</text>

  <!-- Query point -->
  <circle cx="300" cy="170" r="4" fill="#d4d4d4"/>
  <text x="312" y="168" font-size="9" fill="#d4d4d4">z</text>
  <line x1="304" y1="170" x2="326" y2="198" stroke="#d4d4d4" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="340" y="175" font-size="8" fill="#999">→ maps to e₅</text>

  <!-- Info -->
  <text x="600" y="100" font-size="10" fill="#d4d4d4">K codewords</text>
  <text x="600" y="118" font-size="10" fill="#d4d4d4">d dimensions</text>
  <text x="600" y="140" font-size="10" fill="#999">Typical: K=8192</text>
  <text x="600" y="158" font-size="10" fill="#999">d=256</text>
</svg>

---

## VQ-VAE: Variational Autoencoders with Discrete Latents

**VQ-VAE** (van den Oord et al., 2017) combines an encoder-decoder architecture with vector quantization in the latent space.

### Architecture

1. **Encoder** \(f_e\): Maps input \(\mathbf{x}\) to continuous latent representations \(\mathbf{z}_e = f_e(\mathbf{x}) \in \mathbb{R}^{H' \times W' \times d}\)
2. **Quantization**: Each spatial position's \(d\)-dimensional vector is quantized to the nearest codebook entry
3. **Decoder** \(f_d\): Maps the quantized latents \(\mathbf{z}_q\) back to pixel space: \(\hat{\mathbf{x}} = f_d(\mathbf{z}_q)\)

### The VQ-VAE Objective

The loss function has three terms:

$$\mathcal{L} = \underbrace{\|\mathbf{x} - f_d(\mathbf{z}_q)\|_2^2}_{\text{reconstruction}} + \underbrace{\|\text{sg}[\mathbf{z}_e] - \mathbf{e}_k\|_2^2}_{\text{codebook}} + \beta \underbrace{\|\mathbf{z}_e - \text{sg}[\mathbf{e}_k]\|_2^2}_{\text{commitment}}$$

where \(\text{sg}[\cdot]\) is the **stop-gradient** operator (treats the argument as a constant during backpropagation).

**Reconstruction loss:** Standard \(L^2\) reconstruction. The decoder must reconstruct the input from the quantized latents.

**Codebook loss:** Moves each codeword \(\mathbf{e}_k\) toward the encoder outputs that are assigned to it. This is essentially K-means on the encoder outputs. The stop-gradient on \(\mathbf{z}_e\) means only the codebook updates (the encoder does not receive gradients from this term).

**Commitment loss:** Prevents the encoder outputs from drifting too far from the codebook. The stop-gradient on \(\mathbf{e}_k\) means only the encoder updates. The coefficient \(\beta \approx 0.25\) controls the strength.

### The Straight-Through Estimator

The quantization operation \(q(\mathbf{z})\) = argmin is **not differentiable** (it is a piecewise-constant function with zero gradient almost everywhere). How do gradients flow from the decoder loss to the encoder?

The **straight-through estimator (STE)**: during the forward pass, use the quantized values \(\mathbf{z}_q\). During the backward pass, copy the gradients from \(\mathbf{z}_q\) directly to \(\mathbf{z}_e\), as if the quantization had not happened:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_e} \approx \frac{\partial \mathcal{L}}{\partial \mathbf{z}_q}$$

In code: `z_q = z_e + (z_q - z_e).detach()`. The forward pass uses `z_q`, but the gradient flows through `z_e`.

This works because the commitment loss keeps \(\mathbf{z}_e\) close to \(\mathbf{z}_q\), so the gradient at \(\mathbf{z}_q\) is a reasonable approximation of the gradient at \(\mathbf{z}_e\).

### Why the KL Term Vanishes

In a standard VAE, the loss includes a KL divergence term \(D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))\). In VQ-VAE:
- The posterior \(q(\mathbf{z}|\mathbf{x})\) is a delta function on the nearest codebook entry (deterministic quantization)
- The prior \(p(\mathbf{z})\) is uniform over codebook entries

The KL divergence is:

$$D_{\text{KL}} = \log K - \log K = 0$$

It is constant and drops out of the optimization. This is a feature: VQ-VAE avoids the posterior collapse problem that plagues vanilla VAEs, because the discrete bottleneck forces the encoder to use the codebook.

---

## VQ-GAN: Adversarial Training for Sharp Reconstruction

VQ-VAE with an \(L^2\) reconstruction loss produces blurry reconstructions (same issue as always). **VQ-GAN** (Esser et al., 2021) adds perceptual and adversarial losses:

$$\mathcal{L}_{\text{VQ-GAN}} = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|_1}_{\text{L1}} + \underbrace{\lambda_p \mathcal{L}_{\text{perc}}}_{\text{perceptual}} + \underbrace{\lambda_a \mathcal{L}_{\text{GAN}}}_{\text{adversarial}} + \underbrace{\mathcal{L}_{\text{VQ}}}_{\text{codebook + commitment}}$$

The adversarial loss uses a PatchGAN discriminator. The adaptive weighting strategy balances the reconstruction and adversarial losses:

$$\lambda_a = \frac{\|\nabla_{\hat{\mathbf{x}}} \mathcal{L}_{\text{rec}}\|}{\|\nabla_{\hat{\mathbf{x}}} \mathcal{L}_{\text{GAN}}\| + \epsilon}$$

This keeps the adversarial gradient at a comparable magnitude to the reconstruction gradient, preventing either from dominating.

---

## Codebook Collapse and Solutions

A persistent problem: during training, many codebook entries go unused. The encoder learns to map everything to a small subset of codewords, and the unused entries drift away. This is **codebook collapse** (or codebook underutilization).

### Why It Happens

The codebook update (K-means-like) only moves codewords that are actively assigned to encoder outputs. If a codeword is far from any encoder output, it never gets assigned, never gets updated, and drifts further away. Positive feedback loop.

Mathematically, if codeword \(\mathbf{e}_j\) is never the nearest neighbor for any encoder output, then \(\partial \mathcal{L}_{\text{codebook}} / \partial \mathbf{e}_j = 0\) --- zero gradient, no update.

### Solutions

**Codebook reset:** Periodically replace unused codewords with randomly selected encoder outputs. If a codeword has not been used in the last \(N\) batches, replace it with a random \(\mathbf{z}_e\) from the current batch.

**EMA codebook updates:** Instead of gradient descent on the codebook loss, update codewords using an exponential moving average of the encoder outputs assigned to them:

$$\mathbf{e}_k \leftarrow \gamma \mathbf{e}_k + (1 - \gamma) \bar{\mathbf{z}}_k$$

where \(\bar{\mathbf{z}}_k\) is the mean of encoder outputs assigned to codeword \(k\) in the current batch, and \(\gamma \approx 0.99\). This is more stable than gradient-based updates.

**K-means initialization:** Initialize the codebook using K-means on the first batch of encoder outputs, ensuring the codewords start near the actual data distribution.

---

## Finite Scalar Quantization

**FSQ** (Mentzer et al., 2023) sidesteps the codebook entirely. Instead of vector quantization (nearest-neighbor in a learned codebook), independently quantize each dimension of the latent to one of \(L\) fixed levels.

For a \(d\)-dimensional latent with \(L\) levels per dimension, the effective codebook size is \(L^d\). For example, \(d = 8\) and \(L = 5\) gives \(5^8 = 390{,}625\) effective codewords --- without learning a single codebook vector.

### How It Works

1. The encoder outputs \(\mathbf{z}_e \in \mathbb{R}^d\)
2. Apply \(\tanh\) to bound each dimension to \([-1, 1]\)
3. Quantize each dimension independently: \(\hat{z}_i = \text{round}(z_i \cdot (L-1)/2) / ((L-1)/2)\)
4. Use the straight-through estimator for gradients

**Advantages:**
- No codebook to learn, manage, or collapse
- No commitment loss
- Simpler implementation
- Comparable reconstruction quality to VQ-VAE/VQ-GAN

**Lookup-Free Quantization (LFQ)** is a similar idea where each dimension is quantized to \(\{-1, +1\}\), giving a binary codebook of size \(2^d\).

---

## Residual Quantization

A single round of VQ may not capture all the detail. **Residual Quantization (RQ)** applies VQ iteratively to the quantization residual:

1. First round: \(\hat{\mathbf{z}}^{(1)} = q_1(\mathbf{z}_e)\), residual \(\mathbf{r}^{(1)} = \mathbf{z}_e - \hat{\mathbf{z}}^{(1)}\)
2. Second round: \(\hat{\mathbf{z}}^{(2)} = q_2(\mathbf{r}^{(1)})\), residual \(\mathbf{r}^{(2)} = \mathbf{r}^{(1)} - \hat{\mathbf{z}}^{(2)}\)
3. Repeat for \(D\) rounds

The final quantized representation is \(\hat{\mathbf{z}} = \sum_{d=1}^D \hat{\mathbf{z}}^{(d)}\).

Each spatial position now maps to \(D\) codebook indices instead of 1. With \(D = 8\) rounds and codebook size \(K = 1024\), the effective vocabulary for autoregressive modeling is \(1024^8\) --- too large for a flat softmax. Instead, the tokens at each depth are predicted sequentially or in parallel with a depth-conditional model.

**RQ-VAE** and **RQ-Transformer** use this approach for high-fidelity image and video generation.

---

## Tokenizing Video

Extending image tokenization to video adds a temporal dimension. A video tokenizer maps \(\mathbf{x} \in \mathbb{R}^{F \times H \times W \times 3}\) to a grid of discrete tokens \(\mathbf{t} \in \{1, \ldots, K\}^{F' \times H' \times W'}\).

### 3D Causal Tokenizers

**Causal** means the encoder only looks at past and present frames (not future frames), enabling streaming generation. The encoder uses causal 3D convolutions (masked to prevent looking ahead in time).

Compression ratios:
- Spatial: \(f_s = 8\) or \(16\) (reducing 256×256 to 32×32 or 16×16)
- Temporal: \(f_t = 4\) or \(8\) (reducing 16 frames to 4 or 2 tokens in time)

A 16-frame, 256×256 video becomes \(4 \times 32 \times 32 = 4{,}096\) tokens (at \(f_t = 4, f_s = 8\)).

### MAGVIT-v2

Google's MAGVIT-v2 uses a sophisticated tokenizer with:
- **Lookup-Free Quantization** with \(2^{18} = 262{,}144\) effective vocabulary
- Causal 3D encoder/decoder
- Joint image-video training (images are 1-frame videos)
- State-of-the-art reconstruction quality on video

The large vocabulary is crucial: more tokens per frame = higher reconstruction fidelity, but the autoregressive sequence becomes longer and harder to model.

---

## Autoregressive Generation of Video Tokens

With video tokens in hand, generation proceeds autoregressively:

$$p(\mathbf{t}) = \prod_{i=1}^{N} p(t_i | t_1, \ldots, t_{i-1})$$

The ordering is typically **raster scan**: left-to-right, top-to-bottom, frame-by-frame. A Transformer predicts the next token given all previous tokens.

### The Sequence Length Problem

A 5-second, 24fps, 256×256 video with 8× spatial and 4× temporal compression: \(30 \times 32 \times 32 = 30{,}720\) tokens. This is 10× longer than a typical language model context. Attention scales as \(O(N^2)\), making full self-attention impractical.

**Solutions:**

- **Sliding window attention:** Only attend to the most recent \(W\) tokens (local context). Long-range dependencies are captured through stacking layers.
- **Causal masking with block-parallel generation:** Generate one spatial frame at a time, attending to all previous frames but generating all tokens within the current frame in parallel (using a masked image model approach).
- **Hierarchical generation:** First generate a coarse token grid (low temporal/spatial resolution), then fill in details with a second-stage model.

### Masked Prediction (Non-Autoregressive)

An alternative to left-to-right generation: start with all tokens masked, and iteratively unmask tokens in parallel. **MaskGIT** predicts all tokens simultaneously, then keeps the most confident ones and re-predicts the rest. This is 5-10× faster than autoregressive generation because multiple tokens are generated per step.

For video, this can be combined with temporal autoregression: generate frame tokens left-to-right in time, but use masked prediction within each frame.

---

## Diffusion vs Autoregressive: The Fundamental Tradeoff

Two paradigms for video generation, each with distinct mathematical foundations:

| | Autoregressive | Diffusion |
|---|---|---|
| **Latent space** | Discrete tokens | Continuous vectors |
| **Likelihood** | Exact (chain rule) | Variational bound (ELBO) |
| **Generation** | Sequential (token by token) | Parallel (all tokens, iterative denoising) |
| **Quality scaling** | More tokens, larger model | More steps, larger model |
| **Temporal modeling** | Naturally causal | Requires explicit temporal attention |
| **Speed** | Slow (sequential) | Moderate (parallel but iterative) |
| **Diversity** | Temperature/top-k sampling | Noise injection |

**Advantages of autoregressive:**
- Exact log-likelihood training (no variational gap)
- Natural integration with language models (text and video share the same token space)
- Scaling laws from language (well-understood)
- Causal generation enables real-time streaming

**Advantages of diffusion:**
- Parallel generation (faster per step)
- Continuous latent space (no quantization artifacts)
- Flexible conditioning (classifier-free guidance)
- Strong inductive bias for image/video structure

**Hybrid approaches** (MAR, Transfusion) combine both: use autoregressive generation at the sequence level but diffusion within each token's continuous representation. This gets the best of both worlds --- autoregressive for temporal structure, diffusion for spatial detail.

---

## Python: Vector Quantization from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

class VectorQuantizer:
    """Simple vector quantizer with K-means-style codebook learning."""

    def __init__(self, n_embeddings, embedding_dim, beta=0.25):
        self.K = n_embeddings
        self.d = embedding_dim
        self.beta = beta
        # Initialize codebook randomly
        self.codebook = np.random.randn(n_embeddings, embedding_dim) * 0.1
        self.usage_count = np.zeros(n_embeddings)

    def quantize(self, z):
        """Quantize input vectors z (N, d) to nearest codebook entries."""
        # Compute distances (N, K)
        distances = np.sum(z**2, axis=1, keepdims=True) \
                    - 2 * z @ self.codebook.T \
                    + np.sum(self.codebook**2, axis=1, keepdims=True).T
        # Nearest neighbor
        indices = np.argmin(distances, axis=1)
        z_q = self.codebook[indices]
        return z_q, indices

    def compute_loss(self, z_e, z_q):
        """Compute VQ losses (codebook + commitment)."""
        codebook_loss = np.mean((z_e - z_q)**2)  # sg on z_e in practice
        commitment_loss = np.mean((z_e - z_q)**2)  # sg on z_q in practice
        return codebook_loss + self.beta * commitment_loss

    def update_codebook(self, z_e, indices, lr=0.1):
        """Update codebook entries using EMA-style update."""
        for k in range(self.K):
            mask = indices == k
            if np.any(mask):
                self.codebook[k] = (1 - lr) * self.codebook[k] + lr * np.mean(z_e[mask], axis=0)
                self.usage_count[k] += np.sum(mask)

    def reset_unused(self, z_e, threshold=5):
        """Reset unused codebook entries to random encoder outputs."""
        for k in range(self.K):
            if self.usage_count[k] < threshold:
                random_idx = np.random.randint(len(z_e))
                self.codebook[k] = z_e[random_idx] + 0.01 * np.random.randn(self.d)

# Generate 2D data (mixture of clusters)
np.random.seed(42)
n_points = 2000
centers = np.array([[-2, -1], [2, 1], [-1, 2], [1, -2], [0, 0]])
data = []
for c in centers:
    data.append(c + 0.4 * np.random.randn(n_points // 5, 2))
data = np.vstack(data)
np.random.shuffle(data)

# Train VQ
vq = VectorQuantizer(n_embeddings=16, embedding_dim=2, beta=0.25)
losses = []
codebook_history = [vq.codebook.copy()]

for epoch in range(50):
    # Mini-batch
    idx = np.random.choice(len(data), size=256, replace=False)
    z_e = data[idx]

    z_q, indices = vq.quantize(z_e)
    loss = vq.compute_loss(z_e, z_q)
    losses.append(loss)
    vq.update_codebook(z_e, indices, lr=0.2)

    if epoch % 10 == 0:
        vq.reset_unused(z_e, threshold=2)
        codebook_history.append(vq.codebook.copy())

# Final quantization
z_q_all, indices_all = vq.quantize(data)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original data
axes[0, 0].scatter(data[:, 0], data[:, 1], s=1, alpha=0.3, color='#4fc3f7')
axes[0, 0].set_title('Original Data')
axes[0, 0].set_aspect('equal')
axes[0, 0].grid(True, alpha=0.3)

# Codebook evolution
colors = plt.cm.Set3(np.linspace(0, 1, 16))
for i, cb in enumerate(codebook_history):
    alpha = 0.2 + 0.8 * i / len(codebook_history)
    axes[0, 1].scatter(cb[:, 0], cb[:, 1], s=50, alpha=alpha,
                      c=colors, marker='x', linewidths=2)
axes[0, 1].scatter(data[:, 0], data[:, 1], s=1, alpha=0.1, color='gray')
axes[0, 1].set_title('Codebook Evolution')
axes[0, 1].set_aspect('equal')
axes[0, 1].grid(True, alpha=0.3)

# Quantized data (colored by codebook index)
for k in range(vq.K):
    mask = indices_all == k
    if np.any(mask):
        axes[0, 2].scatter(z_q_all[mask, 0], z_q_all[mask, 1], s=5,
                          color=colors[k], alpha=0.5)
axes[0, 2].scatter(vq.codebook[:, 0], vq.codebook[:, 1], s=100,
                   c=colors, marker='*', edgecolors='white', linewidths=1.5, zorder=5)
axes[0, 2].set_title('Quantized Data (colored by index)')
axes[0, 2].set_aspect('equal')
axes[0, 2].grid(True, alpha=0.3)

# Voronoi cells
from scipy.spatial import Voronoi, voronoi_plot_2d
try:
    vor = Voronoi(vq.codebook)
    voronoi_plot_2d(vor, ax=axes[1, 0], show_vertices=False,
                    line_colors='#666', line_width=1)
    axes[1, 0].scatter(data[:, 0], data[:, 1], s=1, alpha=0.2, color='#4fc3f7')
    axes[1, 0].scatter(vq.codebook[:, 0], vq.codebook[:, 1], s=80,
                       c=colors, marker='*', edgecolors='white')
    axes[1, 0].set_xlim(-4, 4)
    axes[1, 0].set_ylim(-4, 4)
except:
    pass
axes[1, 0].set_title('Voronoi Cells')
axes[1, 0].set_aspect('equal')

# Training loss
axes[1, 1].plot(losses, color='#E53935', linewidth=1.5)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('VQ Loss')
axes[1, 1].set_title('Training Loss')
axes[1, 1].grid(True, alpha=0.3)

# Codebook usage histogram
usage_final = np.zeros(vq.K)
for k in range(vq.K):
    usage_final[k] = np.sum(indices_all == k)
axes[1, 2].bar(range(vq.K), usage_final, color=colors)
axes[1, 2].set_xlabel('Codebook Index')
axes[1, 2].set_ylabel('Usage Count')
axes[1, 2].set_title('Codebook Utilization')
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.suptitle('Vector Quantization: Codebook Learning and Voronoi Cells', fontsize=14)
plt.tight_layout()
plt.savefig('vector_quantization.png', dpi=150, bbox_inches='tight')
plt.show()
```

The autoregressive approach to video generation --- tokenize then predict --- has a beautiful theoretical foundation in exact likelihood modeling. But its practical challenge is clear: video is high-dimensional, tokens are many, and sequential generation is slow. Diffusion avoids the sequential bottleneck but pays with an approximate training objective. The field is converging toward hybrid architectures that combine the strengths of both. The mathematics is the same chain rule of probability that underlies all of generative modeling --- just applied at different granularities.
