---
layout: post
title: "Video Tokenization: How Diffusion Transformers Compress Spacetime Into Tokens"
date: 2026-01-27
category: math
---

You cannot feed a raw video into a transformer. A 5-second 1080p clip at 30fps is nearly a billion numbers. Self-attention over a billion tokens does not run on any hardware that exists or will exist in the near future. The entire field of video generation depends on one critical step: **tokenization** -- the compression of raw spatiotemporal data into a tractable number of tokens that a transformer can process.

This post is a complete mathematical treatment of how video tokenization works: the raw data problem, 3D VAE encoders, spatial patching, positional encoding in three dimensions, attention cost calculations, temporal compression strategies, and how architectural choices propagate into the generation quality and speed you experience as a user.

---

## Table of Contents

1. [The Raw Data Problem](#the-raw-data-problem)
2. [The 3D VAE: Compressing Video into Latent Space](#the-3d-vae)
3. [Spatial Patching: From Latent Frames to Tokens](#spatial-patching)
4. [Positional Encoding for Video](#positional-encoding-for-video)
5. [The Attention Cost](#the-attention-cost)
6. [Temporal Compression Strategies](#temporal-compression-strategies)
7. [Practical Impact on Generation](#practical-impact-on-generation)
8. [Comparison Across Models](#comparison-across-models)
9. [Conclusion](#conclusion)

---

## The Raw Data Problem

### Counting the Numbers

A digital video is a 4D tensor: height \(\times\) width \(\times\) time \(\times\) channels. Let us calculate the exact size of a typical AI-generated video clip.

**Parameters:**
- Duration: 5 seconds
- Frame rate: 30 fps
- Resolution: 1080p (1080 \(\times\) 1920)
- Color channels: 3 (RGB)

**Total values (pixels \(\times\) channels \(\times\) frames):**

$$
N_{\text{values}} = T \times H \times W \times C = 150 \times 1080 \times 1920 \times 3
$$

Let us compute this step by step:

$$
1080 \times 1920 = 2{,}073{,}600 \text{ pixels per frame}
$$

$$
2{,}073{,}600 \times 3 = 6{,}220{,}800 \text{ values per frame (RGB)}
$$

$$
6{,}220{,}800 \times 150 = 933{,}120{,}000 \text{ total values}
$$

That is **933.12 million** floating-point numbers for a 5-second clip. Nearly one billion values.

### Memory Requirements

Each value needs to be stored as a floating-point number. The memory requirement depends on the precision:

| Precision | Bytes/Value | Total Memory | Notes |
|-----------|-------------|-------------|-------|
| FP32 (float32) | 4 bytes | \(933{,}120{,}000 \times 4 = 3{,}732{,}480{,}000\) bytes = **3.73 GB** | Full precision training |
| FP16 (float16) | 2 bytes | \(933{,}120{,}000 \times 2 = 1{,}866{,}240{,}000\) bytes = **1.87 GB** | Mixed precision training |
| BF16 (bfloat16) | 2 bytes | **1.87 GB** | Same size as FP16, better dynamic range |
| INT8 (quantized) | 1 byte | **933 MB** | Inference only |

A single 5-second clip at FP16 occupies **1.87 GB**. A training batch of 8 such clips would require **14.93 GB** just for the input data alone -- before any model weights, activations, gradients, or optimizer states.

### Why Raw Pixels Are Intractable for Transformers

A standard vision transformer (ViT) treats each patch of an image as a token. If we naively extended this to video by treating each pixel as a token:

$$
N_{\text{tokens}} = T \times H \times W = 150 \times 1080 \times 1920 = 311{,}040{,}000
$$

Self-attention has quadratic complexity:

$$
\text{Attention cost} = O(N^2 \cdot d)
$$

where \(d\) is the embedding dimension (typically 1024-4096). For 311 million tokens:

$$
N^2 = (3.11 \times 10^8)^2 = 9.67 \times 10^{16}
$$

That is \(9.67 \times 10^{16}\) multiply-accumulate operations per attention layer. With \(d = 2048\):

$$
\text{FLOPs per layer} = 2 \times N^2 \times d = 2 \times 9.67 \times 10^{16} \times 2048 \approx 3.96 \times 10^{20} \text{ FLOPs}
$$

An H100 GPU performs approximately \(10^{15}\) FLOPs per second (1 PFLOP/s at FP16). A single attention layer would take:

$$
t = \frac{3.96 \times 10^{20}}{10^{15}} = 3.96 \times 10^5 \text{ seconds} \approx 4.6 \text{ days}
$$

**4.6 days for one attention layer on one clip.** A DiT with 28 layers, 50 denoising steps? That would be \(4.6 \times 28 \times 50 = 6{,}440\) days \(\approx\) **17.6 years**.

This is why tokenization is not optional. It is the fundamental enabler of video generation.

### The Compression Pipeline

Video tokenization compresses the raw data through two stages:

```
Raw Video                           Latent Representation        Token Sequence
[T, H, W, 3]                       [T', H', W', C_l]            [N_tokens, D]

150 x 1080 x 1920 x 3     ---->    38 x 135 x 240 x 4   ---->  ~123,120 x D

933,120,000 values                  4,924,800 values             123,120 vectors

     3D VAE Encoder                     Patchify + Flatten
     (189x compression)                 (40x compression)
```

The total compression ratio from raw pixels to tokens:

$$
\text{Compression ratio} = \frac{933{,}120{,}000}{123{,}120} \approx 7{,}579\times
$$

We compress the data by a factor of approximately **7,500x** before the transformer ever sees it. Let us now walk through each stage in detail.

---

## The 3D VAE

### What Is a Video VAE?

A Variational Autoencoder (VAE) learns to compress high-dimensional data (video frames) into a low-dimensional latent space, and then reconstruct the original data from that latent representation. The "3D" prefix indicates that the encoder and decoder operate on three spatial-temporal dimensions simultaneously, as opposed to processing each frame independently.

The VAE consists of two networks:

- **Encoder** \(\mathcal{E}\): Maps raw video \(\mathbf{x} \in \mathbb{R}^{T \times H \times W \times 3}\) to a latent representation \(\mathbf{z} \in \mathbb{R}^{T' \times H' \times W' \times C_l}\)
- **Decoder** \(\mathcal{D}\): Maps the latent representation back to pixel space: \(\hat{\mathbf{x}} = \mathcal{D}(\mathbf{z})\)

The training objective balances reconstruction fidelity with latent space regularity:

$$
\mathcal{L}_{\text{VAE}} = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2}_{\text{reconstruction}} + \underbrace{\beta \cdot D_{\text{KL}}\left(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})\right)}_{\text{KL divergence}} + \underbrace{\mathcal{L}_{\text{LPIPS}}(\mathbf{x}, \hat{\mathbf{x}})}_{\text{perceptual loss}} + \underbrace{\mathcal{L}_{\text{GAN}}(\mathbf{x}, \hat{\mathbf{x}})}_{\text{adversarial loss}}
$$

where:

- The **reconstruction loss** measures pixel-level accuracy.
- The **KL divergence** regularizes the latent distribution toward a standard normal \(p(\mathbf{z}) = \mathcal{N}(0, I)\), ensuring the latent space is smooth and continuous.
- The **perceptual loss** (LPIPS) measures similarity in feature space (typically VGG features), penalizing perceptually visible artifacts that pixel-level loss might miss.
- The **adversarial loss** uses a discriminator network to push reconstructions toward the manifold of realistic videos.

The weight \(\beta\) on the KL term controls the reconstruction-regularity tradeoff. Video VAEs typically use very small \(\beta\) (0.001 or less) because high-fidelity reconstruction is more important than perfectly regular latent spaces.

### 3D Convolutional Architecture

The encoder uses 3D convolutions that operate simultaneously over the spatial (H, W) and temporal (T) dimensions. A 3D convolution with kernel size \((k_t, k_h, k_w)\) processes a local spatiotemporal volume:

$$
\text{Conv3D}(\mathbf{x})_{t,h,w,c_{\text{out}}} = \sum_{c_{\text{in}}} \sum_{i=0}^{k_t-1} \sum_{j=0}^{k_h-1} \sum_{k=0}^{k_w-1} \mathbf{W}_{c_{\text{out}}, c_{\text{in}}, i, j, k} \cdot \mathbf{x}_{t+i, h+j, w+k, c_{\text{in}}} + b_{c_{\text{out}}}
$$

**Compression is achieved through strided convolutions** (or equivalently, convolution followed by downsampling). A stride of \((s_t, s_h, s_w)\) reduces each dimension:

$$
T' = \left\lfloor \frac{T}{s_t} \right\rfloor, \quad H' = \left\lfloor \frac{H}{s_h} \right\rfloor, \quad W' = \left\lfloor \frac{W}{s_w} \right\rfloor
$$

### Typical Compression Ratios

Most video VAEs apply multiple strided convolution layers to achieve the following total compression:

| Dimension | Raw Size | Compression Factor | Latent Size | How |
|-----------|---------|-------------------|-------------|-----|
| Temporal (\(T\)) | 150 frames | 4x | 38 | Two stride-2 temporal convolutions (\(2 \times 2 = 4\)) |
| Height (\(H\)) | 1080 pixels | 8x | 135 | Three stride-2 spatial convolutions (\(2^3 = 8\)) |
| Width (\(W\)) | 1920 pixels | 8x | 240 | Three stride-2 spatial convolutions (\(2^3 = 8\)) |
| Channels (\(C\)) | 3 (RGB) | 3 \(\to\) \(C_l\) | 4-16 | Learned channel projection |

**Note on temporal compression:** The first frame is often compressed differently. Some architectures (e.g., CogVideoX's 3D causal VAE) use causal temporal convolutions that only look backward in time, preserving the first frame at full spatial resolution for image-to-video conditioning.

### Worked Example: Full Compression Calculation

Let us trace the exact dimensions through a typical 3D VAE encoder for our 5-second 1080p video.

**Input**: \(\mathbf{x} \in \mathbb{R}^{150 \times 1080 \times 1920 \times 3}\)

**Encoder architecture** (simplified):

```
Layer 1: Conv3D(3 -> 64, kernel=3x3x3, stride=1x1x1, pad=1x1x1)
         Output: 150 x 1080 x 1920 x 64

Layer 2: Conv3D(64 -> 128, kernel=3x3x3, stride=1x2x2, pad=1x1x1)
         Output: 150 x 540 x 960 x 128          [spatial 2x downsample]

Layer 3: ResBlock3D(128 -> 128)
         Output: 150 x 540 x 960 x 128

Layer 4: Conv3D(128 -> 256, kernel=3x3x3, stride=2x2x2, pad=1x1x1)
         Output: 75 x 270 x 480 x 256            [temporal+spatial 2x]

Layer 5: ResBlock3D(256 -> 256)
         Output: 75 x 270 x 480 x 256

Layer 6: Conv3D(256 -> 512, kernel=3x3x3, stride=2x2x2, pad=1x1x1)
         Output: 38 x 135 x 240 x 512            [temporal+spatial 2x]

Layer 7: ResBlock3D(512 -> 512) + Attention
         Output: 38 x 135 x 240 x 512

Layer 8: Conv3D(512 -> 8, kernel=1x1x1, stride=1x1x1)
         Output: 38 x 135 x 240 x 8              [channel projection]

Layer 9: Split into mu and log_var (4 channels each)
         mu:      38 x 135 x 240 x 4
         log_var: 38 x 135 x 240 x 4

Reparameterize: z = mu + exp(0.5 * log_var) * epsilon
         Output: z ∈ R^{38 x 135 x 240 x 4}
```

**Compression summary:**

$$
\text{Temporal}: \frac{150}{38} \approx 4\times, \quad \text{Spatial}: \frac{1080}{135} = \frac{1920}{240} = 8\times, \quad \text{Channels}: 3 \to 4
$$

**Data volume comparison:**

$$
\text{Input}: 150 \times 1080 \times 1920 \times 3 = 933{,}120{,}000 \text{ values}
$$

$$
\text{Latent}: 38 \times 135 \times 240 \times 4 = 4{,}924{,}800 \text{ values}
$$

$$
\text{Compression}: \frac{933{,}120{,}000}{4{,}924{,}800} \approx 189\times
$$

The VAE compresses the video by approximately **189x** in total data volume. The memory footprint at FP16 drops from **1.87 GB** to **9.85 MB**.

### The Decoder (Reconstruction)

The decoder mirrors the encoder architecture, using transposed 3D convolutions (or upsampling + convolution) to expand back to full resolution:

```
Layer 1: Conv3D(4 -> 512, kernel=3x3x3, stride=1x1x1, pad=1x1x1)
         Input: 38 x 135 x 240 x 4
         Output: 38 x 135 x 240 x 512

Layer 2: ResBlock3D(512 -> 512) + Attention
         Output: 38 x 135 x 240 x 512

Layer 3: Upsample(2x2x2) + Conv3D(512 -> 256)
         Output: 76 x 270 x 480 x 256             [temporal+spatial 2x up]

Layer 4: ResBlock3D(256 -> 256)
         Output: 76 x 270 x 480 x 256

Layer 5: Upsample(2x2x2) + Conv3D(256 -> 128)
         Output: 152 x 540 x 960 x 128            [temporal+spatial 2x up]

Layer 6: ResBlock3D(128 -> 128)
         Output: 152 x 540 x 960 x 128

Layer 7: Upsample(1x2x2) + Conv3D(128 -> 64)
         Output: 152 x 1080 x 1920 x 64           [spatial 2x up only]

Layer 8: Conv3D(64 -> 3, kernel=3x3x3, stride=1x1x1, pad=1x1x1)
         Output: 152 x 1080 x 1920 x 3            [to RGB]

Crop/trim temporal: 150 x 1080 x 1920 x 3
```

The slight temporal overshoot (152 vs 150) is handled by cropping. Some architectures use padding strategies that maintain exact dimensions.

### Reconstruction Quality

The VAE is trained independently before the diffusion model. Its reconstruction quality sets a hard ceiling on the final generation quality -- **no diffusion model can produce details that the decoder cannot reconstruct.**

Typical reconstruction metrics for state-of-the-art video VAEs:

| Metric | Value | What It Measures |
|--------|-------|-----------------|
| PSNR | 32-38 dB | Pixel-level accuracy (higher = better) |
| SSIM | 0.95-0.98 | Structural similarity (higher = better) |
| LPIPS | 0.03-0.08 | Perceptual similarity (lower = better) |
| FVD | 15-50 | Frechet Video Distance (lower = better) |

A PSNR of 35 dB means the mean squared error between the original and reconstructed video is:

$$
\text{MSE} = 10^{-\text{PSNR}/10} = 10^{-3.5} \approx 3.16 \times 10^{-4}
$$

For pixel values normalized to \([0, 1]\), this corresponds to an average per-pixel error of:

$$
\text{RMSE} = \sqrt{3.16 \times 10^{-4}} \approx 0.0178
$$

On a 0-255 scale, that is approximately \(0.0178 \times 255 \approx 4.5\) intensity levels -- barely perceptible to the human eye for most content.

---

## Spatial Patching

### From Latent Volume to Token Sequence

The 3D VAE gives us a compressed latent volume \(\mathbf{z} \in \mathbb{R}^{T' \times H' \times W' \times C_l}\). This is still a 4D tensor. A transformer operates on a 1D sequence of tokens. The next step is **patchification**: dividing the latent volume into non-overlapping patches and flattening each patch into a token vector.

### The Patchification Process

Given a latent volume of size \((T', H', W', C_l)\) and a patch size of \((p_t, p_h, p_w)\) (where \(p_t\) is the temporal patch size and \(p_h, p_w\) are spatial patch sizes):

**Step 1: Divide into patches.**

The number of patches along each dimension:

$$
n_t = \frac{T'}{p_t}, \quad n_h = \frac{H'}{p_h}, \quad n_w = \frac{W'}{p_w}
$$

**Step 2: Total token count.**

$$
N_{\text{tokens}} = n_t \times n_h \times n_w = \frac{T'}{p_t} \times \frac{H'}{p_h} \times \frac{W'}{p_w}
$$

**Step 3: Token dimension.**

Each patch is a sub-volume of size \((p_t, p_h, p_w, C_l)\). Flattened, this becomes a vector of dimension:

$$
d_{\text{patch}} = p_t \times p_h \times p_w \times C_l
$$

This is then linearly projected to the model's hidden dimension \(D\) (typically 1024, 2048, or 4096):

$$
\mathbf{t}_i = \mathbf{W}_{\text{proj}} \cdot \text{flatten}(\text{patch}_i) + \mathbf{b}_{\text{proj}}
$$

where \(\mathbf{W}_{\text{proj}} \in \mathbb{R}^{D \times d_{\text{patch}}}\).

### Worked Example

Using our running example (\(T' = 38\), \(H' = 135\), \(W' = 240\), \(C_l = 4\)) with spatial patch size \(p_h = p_w = 2\) and temporal patch size \(p_t = 1\):

$$
n_t = \frac{38}{1} = 38, \quad n_h = \frac{135}{2} = 67.5 \to 67, \quad n_w = \frac{240}{2} = 120
$$

(Note: When dimensions are not evenly divisible, the latent is typically padded. For simplicity, assume \(H' = 136\) after padding.)

$$
n_h = \frac{136}{2} = 68
$$

$$
N_{\text{tokens}} = 38 \times 68 \times 120 = 309{,}840
$$

Each patch vector dimension:

$$
d_{\text{patch}} = 1 \times 2 \times 2 \times 4 = 16
$$

This is projected to the model dimension \(D = 2048\):

$$
\mathbf{W}_{\text{proj}} \in \mathbb{R}^{2048 \times 16}
$$

So our 5-second 1080p video, after VAE encoding and patchification, becomes a sequence of **309,840 tokens**, each of dimension 2048.

### Spacetime Patches (Sora's Approach)

Sora's key innovation (described in OpenAI's technical report) is **spacetime patches**: patches that span both spatial and temporal dimensions simultaneously. Instead of treating each latent frame independently (\(p_t = 1\)), Sora uses \(p_t > 1\).

With \(p_t = 2\):

$$
n_t = \frac{38}{2} = 19
$$

$$
N_{\text{tokens}} = 19 \times 68 \times 120 = 154{,}920
$$

The token count is halved. The patch vector dimension doubles:

$$
d_{\text{patch}} = 2 \times 2 \times 2 \times 4 = 32
$$

This is a fundamental tradeoff: **larger temporal patches reduce token count (and thus attention cost) but force the model to handle temporal dynamics within each patch rather than across the attention mechanism.**

### Comparison of Patch Strategies

| Strategy | \(p_t\) | \(p_h\) | \(p_w\) | \(N_{\text{tokens}}\) | \(d_{\text{patch}}\) | Attention Cost | Temporal Granularity |
|----------|-------|-------|-------|---------------------|--------------------|----|------|
| Spatial-only | 1 | 2 | 2 | 309,840 | 16 | Very High | Per-latent-frame |
| Spacetime (Sora) | 2 | 2 | 2 | 154,920 | 32 | High | Every 2 latent frames |
| Aggressive spacetime | 4 | 2 | 2 | 77,460 | 64 | Moderate | Every 4 latent frames |
| Large spatial | 1 | 4 | 4 | 77,520 | 64 | Moderate | Per-latent-frame |
| Large spacetime | 2 | 4 | 4 | 38,760 | 128 | Low | Every 2 latent frames |

### The Patchification Operation in Code

```python
import torch
import torch.nn as nn


class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding for video latent representations.

    Takes a latent volume [B, C, T, H, W] and produces a
    sequence of token embeddings [B, N, D].
    """

    def __init__(
        self,
        latent_channels: int = 4,
        patch_size: tuple = (2, 2, 2),  # (t, h, w)
        embed_dim: int = 2048,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Linear projection implemented as a 3D convolution
        # with kernel size = patch size and stride = patch size
        self.proj = nn.Conv3d(
            in_channels=latent_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W] latent video tensor
        Returns:
            tokens: [B, N, D] where N = (T/pt) * (H/ph) * (W/pw)
        """
        B, C, T, H, W = x.shape
        pt, ph, pw = self.patch_size

        # Verify dimensions are divisible by patch size
        assert T % pt == 0, f"T={T} not divisible by pt={pt}"
        assert H % ph == 0, f"H={H} not divisible by ph={ph}"
        assert W % pw == 0, f"W={W} not divisible by pw={pw}"

        # Apply 3D convolution: [B, C, T, H, W] -> [B, D, T/pt, H/ph, W/pw]
        x = self.proj(x)

        # Reshape to token sequence: [B, D, nt, nh, nw] -> [B, N, D]
        x = x.flatten(2)  # [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]

        return x
```

### Visual Representation of Patchification

Here is an ASCII diagram of how a single latent frame is divided into 2x2 patches:

```
Latent frame (H'=8, W'=8, C=4):

+----+----+----+----+
| p1 | p2 | p3 | p4 |   Each cell is a 2x2 spatial patch
+----+----+----+----+   containing 2 x 2 x 4 = 16 values
| p5 | p6 | p7 | p8 |
+----+----+----+----+   Token count per frame: 4 x 4 = 16
| p9 |p10 |p11 |p12 |
+----+----+----+----+   With spacetime patches (p_t=2),
|p13 |p14 |p15 |p16 |   two consecutive frames share
+----+----+----+----+   each patch position, so the token
                         count is halved for those pairs.

Spacetime patching across 4 latent frames:

Frame t=0:  [p1][p2][p3][p4]   }
Frame t=1:  [p1][p2][p3][p4]   }-- These 2 frames = 16 spacetime tokens
                                    (each token covers 2x2 spatial x 2 temporal)
Frame t=2:  [p1][p2][p3][p4]   }
Frame t=3:  [p1][p2][p3][p4]   }-- Next 16 spacetime tokens

Without spacetime patches: 4 frames x 16 patches = 64 tokens
With spacetime patches (p_t=2): 2 groups x 16 patches = 32 tokens
```

---

## Positional Encoding for Video

### The Problem: Tokens Have No Inherent Position

After patchification, we have a flat sequence of \(N\) tokens. The transformer's self-attention mechanism is **permutation-invariant** -- it does not inherently know which token came from which spatial location or which point in time. We must inject positional information.

For video, position is three-dimensional: \((x, y, t)\). Each token has a spatial position within its frame and a temporal position within the video.

### Sinusoidal Positional Encoding (3D Extension)

The original sinusoidal encoding from "Attention Is All You Need" can be extended to three dimensions. For a token at position \((x, y, t)\):

$$
\text{PE}(x, y, t) = \text{PE}_x(x) \oplus \text{PE}_y(y) \oplus \text{PE}_t(t)
$$

where \(\oplus\) denotes concatenation and each component uses the standard sinusoidal formula. For dimension \(i\) of the encoding:

$$
\text{PE}_x(x, 2i) = \sin\left(\frac{x}{10000^{2i/d_x}}\right), \quad \text{PE}_x(x, 2i+1) = \cos\left(\frac{x}{10000^{2i/d_x}}\right)
$$

where \(d_x\) is the number of dimensions allocated to the x-component (typically \(D/3\), splitting the model dimension evenly among the three axes).

**Worked example**: With \(D = 2048\) and \(d_x = d_y = d_t = 682\) (approximately $2048/3$, with 2 leftover dimensions padded or allocated to one axis), the positional encoding for a token at position \((5, 12, 3)\) contains:

For the \(x\)-component, dimension \(i = 0\):

$$
\text{PE}_x(5, 0) = \sin\left(\frac{5}{10000^{0/682}}\right) = \sin\left(\frac{5}{1}\right) = \sin(5) \approx -0.959
$$

$$
\text{PE}_x(5, 1) = \cos(5) \approx 0.284
$$

For dimension \(i = 100\):

$$
\text{PE}_x(5, 200) = \sin\left(\frac{5}{10000^{200/682}}\right) = \sin\left(\frac{5}{10000^{0.293}}\right) = \sin\left(\frac{5}{19.31}\right) = \sin(0.259) \approx 0.256
$$

The key property: tokens that are close in space or time have similar positional encodings (high dot product), while distant tokens have dissimilar encodings.

### Learned Positional Embeddings

An alternative is to learn the positional embeddings as parameters:

$$
\text{PE}(x, y, t) = \mathbf{E}_x[x] + \mathbf{E}_y[y] + \mathbf{E}_t[t]
$$

where \(\mathbf{E}_x \in \mathbb{R}^{n_w \times D}\), \(\mathbf{E}_y \in \mathbb{R}^{n_h \times D}\), and \(\mathbf{E}_t \in \mathbb{R}^{n_t \times D}\) are learnable embedding tables.

**Advantages**: Can learn arbitrary positional relationships.
**Disadvantages**: Cannot generalize to positions unseen during training. If trained on 5-second videos (38 temporal positions), cannot generate 10-second videos without interpolation or extrapolation of the temporal embeddings.

### Rotary Position Embeddings (RoPE)

RoPE has become the preferred positional encoding in modern video DiTs (including architectures used in Wan 2.2 and related models). Instead of adding positional information to the token embeddings, RoPE applies **rotation matrices** to the query and key vectors in attention.

For a 1D position \(m\), RoPE rotates pairs of dimensions by angles proportional to \(m\):

$$
\text{RoPE}(\mathbf{q}, m) = \begin{pmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \end{pmatrix} \odot \begin{pmatrix} \cos(m\theta_0) \\ \cos(m\theta_0) \\ \cos(m\theta_1) \\ \cos(m\theta_1) \\ \vdots \end{pmatrix} + \begin{pmatrix} -q_1 \\ q_0 \\ -q_3 \\ q_2 \\ \vdots \end{pmatrix} \odot \begin{pmatrix} \sin(m\theta_0) \\ \sin(m\theta_0) \\ \sin(m\theta_1) \\ \sin(m\theta_1) \\ \vdots \end{pmatrix}
$$

where \(\theta_i = 10000^{-2i/d}\) and \(\odot\) is element-wise multiplication.

**For 3D video positions**, RoPE is extended by partitioning the embedding dimensions among the three axes:

$$
d = d_x + d_y + d_t
$$

Dimensions \([0, d_x)\) are rotated by the \(x\)-position, dimensions \([d_x, d_x + d_y)\) by the \(y\)-position, and dimensions \([d_x + d_y, d)\) by the \(t\)-position.

The critical property of RoPE for video generation: the attention score between two tokens depends only on their **relative** position, not their absolute position:

$$
\mathbf{q}_m^T \mathbf{k}_n = f(\mathbf{q}, \mathbf{k}, m - n)
$$

This means a model trained on short videos can potentially generate longer videos, because the pairwise attention scores are position-independent. The model only needs to handle larger relative distances, not unseen absolute positions.

### RoPE Implementation for 3D Video

```python
import torch
import math


def get_3d_rope_freqs(
    dim: int,
    n_t: int, n_h: int, n_w: int,
    theta: float = 10000.0,
    dim_split: tuple = None,  # (d_t, d_h, d_w), must sum to dim
) -> torch.Tensor:
    """
    Compute 3D RoPE frequency tensor for video tokens.

    Args:
        dim: embedding dimension (must be even)
        n_t, n_h, n_w: number of positions in each dimension
        theta: base frequency
        dim_split: how to split dims among (t, h, w)

    Returns:
        freqs: [n_t * n_h * n_w, dim] complex tensor
    """
    if dim_split is None:
        # Default: split evenly (each must be even)
        d_t = dim // 3
        d_t = d_t - (d_t % 2)  # ensure even
        d_h = dim // 3
        d_h = d_h - (d_h % 2)
        d_w = dim - d_t - d_h
        d_w = d_w - (d_w % 2)
        dim_split = (d_t, d_h, d_w)

    d_t, d_h, d_w = dim_split

    # Frequencies for each axis
    freqs_t = 1.0 / (theta ** (torch.arange(0, d_t, 2).float() / d_t))
    freqs_h = 1.0 / (theta ** (torch.arange(0, d_h, 2).float() / d_h))
    freqs_w = 1.0 / (theta ** (torch.arange(0, d_w, 2).float() / d_w))

    # Position indices
    t_pos = torch.arange(n_t).float()
    h_pos = torch.arange(n_h).float()
    w_pos = torch.arange(n_w).float()

    # Outer products: [positions, frequencies]
    freqs_t = torch.outer(t_pos, freqs_t)  # [n_t, d_t/2]
    freqs_h = torch.outer(h_pos, freqs_h)  # [n_h, d_h/2]
    freqs_w = torch.outer(w_pos, freqs_w)  # [n_w, d_w/2]

    # Broadcast to full 3D grid
    # freqs_t: [n_t, 1, 1, d_t/2] -> [n_t, n_h, n_w, d_t/2]
    # etc.
    freqs_t = freqs_t[:, None, None, :].expand(n_t, n_h, n_w, -1)
    freqs_h = freqs_h[None, :, None, :].expand(n_t, n_h, n_w, -1)
    freqs_w = freqs_w[None, None, :, :].expand(n_t, n_h, n_w, -1)

    # Concatenate along frequency dimension
    freqs = torch.cat([freqs_t, freqs_h, freqs_w], dim=-1)  # [n_t, n_h, n_w, dim/2]

    # Flatten spatial dimensions
    freqs = freqs.reshape(-1, freqs.shape[-1])  # [N, dim/2]

    # Convert to complex: cos + i*sin
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_complex


def apply_rope_3d(
    x: torch.Tensor,  # [B, N, H, D/H] queries or keys
    freqs: torch.Tensor,  # [N, D/2H] per-head frequencies
) -> torch.Tensor:
    """Apply 3D RoPE to queries or keys."""
    # Convert to complex pairs
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2)
    )

    # Apply rotation
    x_rotated = x_complex * freqs

    # Convert back to real
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)
```

---

## The Attention Cost

### Self-Attention Complexity

The core operation of the transformer is self-attention:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}
$$

where \(\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d_k}\) (for a single head).

The computational cost breaks down as:

1. **QK^T computation**: \(O(N^2 \cdot d_k)\) FLOPs
2. **Softmax**: \(O(N^2)\) FLOPs
3. **Attention \(\times\) V**: \(O(N^2 \cdot d_k)\) FLOPs

Total per attention layer (all heads):

$$
\text{FLOPs}_{\text{attention}} = 4 \cdot N^2 \cdot d_k \cdot n_{\text{heads}} = 4 \cdot N^2 \cdot D
$$

where \(D = d_k \cdot n_{\text{heads}}\) is the total model dimension.

The **memory** for the attention matrix is:

$$
\text{Memory}_{\text{attn}} = N^2 \times n_{\text{heads}} \times \text{bytes\_per\_value}
$$

### Exact Calculations for Video

Let us compute the exact attention cost for our running example at different tokenization strategies.

**Setup:**
- Model dimension: \(D = 2048\)
- Number of attention heads: \(n_h = 16\)
- Head dimension: \(d_k = D / n_h = 128\)
- Number of DiT layers: \(L = 28\)
- Number of denoising steps: \(S = 50\)

#### Case 1: Spatial-Only Patches (\(p_t=1, p_h=2, p_w=2\))

$$
N = 38 \times 68 \times 120 = 309{,}840 \text{ tokens}
$$

**Attention matrix size:**

$$
N^2 = 309{,}840^2 = 9.60 \times 10^{10} \approx 96 \text{ billion entries}
$$

**Memory for attention matrix** (FP16, per head):

$$
96 \times 10^9 \times 2 \text{ bytes} = 192 \text{ GB per head}
$$

With 16 heads: \(192 \times 16 = 3{,}072\) GB. **This does not fit on any single GPU or even a typical multi-GPU node.** Full self-attention at this token count is infeasible.

#### Case 2: Spacetime Patches (\(p_t=2, p_h=2, p_w=2\))

$$
N = 19 \times 68 \times 120 = 154{,}920 \text{ tokens}
$$

$$
N^2 = 154{,}920^2 = 2.40 \times 10^{10} \approx 24 \text{ billion entries}
$$

**Memory**: \(24 \times 10^9 \times 2 \times 16 = 768\) GB. Still infeasible for full attention.

#### Case 3: Aggressive Compression + Smaller Resolution

Let us consider a more typical production setting: 720p resolution (\(720 \times 1280\)), 5 seconds at 24fps.

| Dimension | Raw | After VAE (8x spatial, 4x temporal) | Latent |
|-----------|-----|-------------------------------------|--------|
| T | 120 | \(\div 4\) | 30 |
| H | 720 | \(\div 8\) | 90 |
| W | 1280 | \(\div 8\) | 160 |
| C | 3 | \(\to\) | 4 |

With spacetime patches \(p_t = 2, p_h = 2, p_w = 2\):

$$
N = \frac{30}{2} \times \frac{90}{2} \times \frac{160}{2} = 15 \times 45 \times 80 = 54{,}000 \text{ tokens}
$$

**Attention matrix memory** (FP16, all heads):

$$
54{,}000^2 \times 16 \times 2 = 2{,}916{,}000{,}000 \times 32 = 93.3 \text{ GB}
$$

Still large, but with FlashAttention (which computes attention without materializing the full \(N \times N\) matrix), this is feasible on a multi-GPU setup.

**FLOPs per attention layer:**

$$
4 \times 54{,}000^2 \times 2048 = 4 \times 2.916 \times 10^9 \times 2048 = 2.39 \times 10^{13} \text{ FLOPs}
$$

**Total FLOPs for full generation** (28 layers, 50 steps):

$$
2.39 \times 10^{13} \times 28 \times 50 = 3.35 \times 10^{16} \text{ FLOPs}
$$

On an H100 at 1 PFLOP/s effective throughput:

$$
t = \frac{3.35 \times 10^{16}}{10^{15}} = 33.5 \text{ seconds}
$$

This accounts for attention FLOPs only. Adding feedforward layers (roughly equal compute to attention), the total is approximately **67 seconds** of H100 time. This aligns with observed generation times for 5-second 720p video on production systems.

### The Token Count vs. Quality Tradeoff

```
Generation Time (seconds)
    |
120 |                                           *
    |                                      *
100 |                                 *
    |                            *
 80 |                       *
    |                  *
 60 |             * <-- Sweet spot for most models
    |        *
 40 |   *
    | *
 20 |
    +----+----+----+----+----+----+----+----+--->
      10K  30K  50K  80K 100K 150K 200K 300K
                    Token Count (N)

    Note: Time scales approximately as N^2
    (quadratic, though FlashAttention reduces constant factor)
```

| Token Count | Attention FLOPs/Layer | Gen Time (est.) | Quality Trade-off |
|-------------|----------------------|-----------------|-------------------|
| 10,000 | \(8.2 \times 10^{11}\) | ~3 sec | Low res / short, limited detail |
| 30,000 | \(7.4 \times 10^{12}\) | ~12 sec | 480p quality, acceptable for previews |
| 54,000 | \(2.4 \times 10^{13}\) | ~35 sec | 720p quality, production usable |
| 100,000 | \(8.2 \times 10^{13}\) | ~90 sec | 1080p quality, high detail |
| 200,000 | \(3.3 \times 10^{14}\) | ~300 sec | High resolution, max quality |
| 300,000 | \(7.4 \times 10^{14}\) | ~600 sec | Approaching compute limits |

### Why Longer Videos Are Exponentially More Expensive

Doubling video duration doubles the temporal dimension, which (at fixed compression) doubles the token count. But attention cost is **quadratic** in token count:

$$
\text{Cost}(2T) = (2N)^2 = 4N^2 = 4 \times \text{Cost}(T)
$$

Doubling the duration **quadruples** the compute cost. This is the fundamental reason why generating a 20-second video is not 4x the cost of a 5-second video -- it is **16x** the cost (assuming full self-attention):

| Duration | Tokens (typical) | Relative Cost | Absolute Cost (est. H100-seconds) |
|----------|------------------|---------------|-----------------------------------|
| 2 sec | 21,600 | 1x | ~6 sec |
| 5 sec | 54,000 | 6.25x | ~35 sec |
| 10 sec | 108,000 | 25x | ~150 sec |
| 20 sec | 216,000 | 100x | ~600 sec |
| 60 sec | 648,000 | 900x | ~5,400 sec |

This is why no current model generates 60-second videos in a single pass. The \(N^2\) scaling makes it computationally prohibitive. Instead, longer videos are generated through:
- Temporal chunking with autoregressive sliding windows
- I2V chaining (as described in the Adobe post-production post)
- Hierarchical generation (generate keyframes, then interpolate)

---

## Temporal Compression Strategies

The temporal dimension is where video tokenization gets most interesting -- and most contentious. Different compression strategies produce dramatically different quality-compute tradeoffs.

### Strategy 1: Uniform Temporal Downsampling

The simplest approach: apply a fixed temporal stride in the VAE, reducing every \(k\) frames to 1 latent frame.

**Mechanism**: Strided 3D convolutions with temporal stride \(s_t = 2\) applied at multiple layers. Two layers of stride 2 yields \(4\times\) total temporal compression.

$$
T' = \frac{T}{s_t^L} = \frac{150}{4} = 37.5 \to 38 \text{ (with padding)}
$$

**Advantages:**
- Simple architecture
- Uniform treatment of all temporal regions
- Consistent quality across the video

**Disadvantages:**
- Wastes capacity on static regions (a still background doesn't need as many temporal tokens as an action sequence)
- Cannot adapt to content -- rapid motion and static scenes get the same compression

**Quality-Compute Tradeoff:**

| Temporal Compression | Latent Frames (5s@30fps) | Tokens (720p, \(p=2\)) | Temporal Quality |
|---------------------|------------------------|---------------------|-----------------|
| 1x (no compression) | 150 | 270,000 | Perfect |
| 2x | 75 | 135,000 | Very good |
| 4x | 38 | 54,000 | Good (standard) |
| 8x | 19 | 27,000 | Moderate (some motion blur) |
| 16x | 10 | 14,400 | Poor (significant temporal aliasing) |

### Strategy 2: Keyframe-Based Compression

Instead of uniform downsampling, allocate more latent capacity to keyframes (frames with significant change) and less to static/slowly-changing regions.

**Mechanism**: A content-aware temporal sampler that:
1. Computes optical flow magnitude between consecutive frames
2. Identifies frames with high motion as keyframes
3. Encodes keyframes at full temporal resolution
4. Encodes inter-keyframe regions at reduced resolution

```python
def keyframe_temporal_sampling(
    video: torch.Tensor,  # [T, H, W, 3]
    max_latent_frames: int = 38,
    keyframe_threshold: float = 0.15,  # optical flow magnitude threshold
) -> tuple:
    """
    Content-aware temporal sampling for video tokenization.

    Returns indices of frames to encode at full resolution
    and a sampling schedule for the rest.
    """
    import cv2

    T = video.shape[0]
    flow_magnitudes = []

    for i in range(1, T):
        prev = video[i-1].numpy().astype('uint8')
        curr = video[i].numpy().astype('uint8')
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
        flow_magnitudes.append(magnitude)

    flow_magnitudes = np.array(flow_magnitudes)

    # Identify keyframes (high motion frames + first/last)
    keyframes = set([0, T - 1])
    for i, mag in enumerate(flow_magnitudes):
        if mag > keyframe_threshold:
            keyframes.add(i + 1)

    # Distribute remaining budget among non-keyframe regions
    remaining_budget = max_latent_frames - len(keyframes)
    non_keyframe_indices = [i for i in range(T) if i not in keyframes]

    if remaining_budget > 0 and len(non_keyframe_indices) > 0:
        # Uniform sampling of non-keyframe regions
        step = max(1, len(non_keyframe_indices) // remaining_budget)
        sampled = non_keyframe_indices[::step][:remaining_budget]
        keyframes.update(sampled)

    selected_frames = sorted(keyframes)[:max_latent_frames]

    return selected_frames
```

**Advantages:**
- Better quality for high-motion scenes with the same token budget
- Allocates capacity where it matters most

**Disadvantages:**
- Content-dependent -- different videos produce different token counts (complicates batching)
- Requires an optical flow computation pass before encoding
- More complex decoder (needs to interpolate between non-uniformly spaced latent frames)

### Strategy 3: Learned Temporal Compression

The most sophisticated approach: let the model learn its own temporal compression through attention-based temporal aggregation.

**Mechanism**: Instead of fixed strided convolutions, use temporal attention layers in the VAE encoder that learn to aggregate temporal information adaptively. The CogVideoX approach uses a "3D causal VAE" where temporal convolutions are causal (only look backward), enabling autoregressive generation while maintaining temporal compression.

**Architecture sketch:**

```
Input frames: [f_0, f_1, f_2, ..., f_149]
                      |
            Spatial encoder (2D convolutions)
                      |
         Per-frame spatial features: [z_0, z_1, ..., z_149]
                      |
            Temporal attention + downsampling
            (learned which frames to "merge")
                      |
         Compressed temporal features: [z'_0, z'_1, ..., z'_37]
```

The temporal attention layer computes:

$$
\mathbf{z}'_i = \text{Attention}\left(\mathbf{Q}_i, \left\{\mathbf{K}_j, \mathbf{V}_j\right\}_{j \in \mathcal{W}(i)}\right)
$$

where \(\mathcal{W}(i)\) is a local temporal window around position \(i\). The stride of the output positions determines the compression ratio.

**Advantages:**
- Learns content-adaptive compression
- Can preserve fine temporal details where needed
- End-to-end trainable with the diffusion model

**Disadvantages:**
- More expensive to train
- Harder to analyze and debug
- May overfit to training data temporal statistics

### Strategy 4: Hierarchical Temporal Representation

Used in some recent architectures: represent video at multiple temporal resolutions simultaneously.

```
Full temporal resolution:    [z_0, z_1, z_2, z_3, z_4, z_5, z_6, z_7]  (8 frames)
                                     \    /         \    /
Half temporal resolution:            [z_01,  z_23,  z_45,  z_67]         (4 frames)
                                        \    /         \    /
Quarter temporal resolution:            [z_0123,       z_4567]            (2 frames)
                                             \        /
Eighth temporal resolution:                 [z_all]                       (1 frame)
```

Tokens from all levels are concatenated and fed to the transformer. This gives the model access to both coarse temporal structure (what is the overall motion?) and fine temporal detail (what happens in this exact frame?) simultaneously.

**Token count for this strategy:**

$$
N_{\text{hierarchical}} = \sum_{l=0}^{L} \frac{T'}{2^l} \times n_h \times n_w
$$

For \(T' = 38\), 4 levels, with \(n_h \times n_w = 5{,}400\):

$$
N = (38 + 19 + 10 + 5) \times 5{,}400 = 72 \times 5{,}400 = 388{,}800
$$

This is higher than single-level tokenization, but the lower levels contain far fewer tokens and the model can attend to them selectively.

---

## Practical Impact on Generation

### How Tokenization Determines Speed

The generation speed of a video diffusion model is dominated by the denoising loop, which runs the DiT forward pass \(S\) times (one per denoising step):

$$
t_{\text{generation}} = S \times t_{\text{forward}}
$$

The forward pass time is dominated by attention (for large token counts):

$$
t_{\text{forward}} \propto L \times \left(\underbrace{4 N^2 D}_{\text{attention}} + \underbrace{8 N D^2}_{\text{feedforward}}\right)
$$

For the attention term to dominate (which determines the scaling behavior), we need \(4N^2 D > 8ND^2\), i.e., \(N > 2D\). With \(D = 2048\), this means attention dominates when \(N > 4096\) tokens, which is always the case for video.

**Speed comparison for different tokenization strategies (same content):**

| Strategy | N (tokens) | Relative Speed | Estimated Time (H100, 50 steps) |
|----------|-----------|----------------|--------------------------------|
| No temporal compression, \(p=1\) | 486,000 | 0.01x | ~2,500 sec |
| 4x temporal, spatial \(p=2\) | 54,000 | 1x (baseline) | ~35 sec |
| 4x temporal, spacetime \(p=(2,2,2)\) | 27,000 | 4x | ~9 sec |
| 8x temporal, spacetime \(p=(2,2,2)\) | 13,500 | 16x | ~2.2 sec |

### How Tokenization Determines Quality Ceiling

Every compression step loses information. The quality ceiling is set by the **least lossy path** from raw video through the tokenization pipeline.

**Information loss at each stage:**

1. **VAE spatial compression (8x)**: Loses high-frequency spatial details. Fine textures, text rendering, and sharp edges are the first casualties.

2. **VAE temporal compression (4x)**: Loses rapid temporal changes. Fast motion, flickering lights, and particle effects degrade.

3. **Patchification**: Loses within-patch spatial correlations that the attention mechanism cannot reconstruct. Larger patches = more loss.

4. **Attention bandwidth**: Even with lossless tokenization, the transformer has limited capacity to model all token interactions. With more tokens, each token gets proportionally less "attention budget."

**Quality vs. compression tradeoff (empirical observations):**

```
Quality (FVD, lower = better)
    |
500 |
    |
400 |                                              * (16x temporal)
    |
300 |                              * (8x temporal)
    |
200 |                  * (4x temporal)
    |
100 |      * (2x temporal)
    |  * (1x temporal, theoretical optimum)
  0 +----+----+----+----+----+----+----+----+--->
    0    50K  100K 150K 200K 250K 300K 350K
         Token Count Reduction (from no compression)
```

### How Tokenization Determines Maximum Duration

The maximum video duration a model can generate in a single pass is directly determined by the token budget:

$$
T_{\text{max}} = \frac{N_{\text{budget}} \times p_t \times s_t}{f_{\text{fps}}}
$$

where:
- \(N_{\text{budget}}\) is the maximum number of tokens the model can handle (set by GPU memory and acceptable generation time)
- \(p_t\) is the temporal patch size
- \(s_t\) is the VAE temporal compression factor
- \(f_{\text{fps}}\) is the frame rate

**Example**: If the token budget is 100,000, spatial dimensions consume \(68 \times 120 = 8{,}160\) tokens per temporal position, patch temporal size is 2, and VAE temporal compression is 4x:

$$
n_t = \frac{N_{\text{budget}}}{n_h \times n_w} = \frac{100{,}000}{8{,}160} \approx 12 \text{ temporal token positions}
$$

$$
T_{\text{latent}} = n_t \times p_t = 12 \times 2 = 24 \text{ latent frames}
$$

$$
T_{\text{raw}} = T_{\text{latent}} \times s_t = 24 \times 4 = 96 \text{ raw frames}
$$

$$
\text{Duration} = \frac{96}{30} = 3.2 \text{ seconds}
$$

To generate a 10-second video at the same resolution:

$$
\text{Tokens needed} = \frac{10 \times 30}{4 \times 2} \times 8{,}160 = 37.5 \times 8{,}160 = 306{,}000 \text{ tokens}
$$

The attention cost scales as \(306{,}000^2 / 100{,}000^2 = 9.36\times\) -- nearly an order of magnitude more expensive.

**This is why a model with 4x temporal compression can generate 4x longer videos at the same compute cost** compared to a model with 1x temporal compression. The temporal compression factor directly multiplies the achievable duration within a fixed token budget.

---

## Comparison Across Models

### Model Tokenization Strategies

The following table compares the tokenization approaches of major video generation models, based on published papers, technical reports, and architectural analysis.

| Model | VAE Type | Spatial Comp. | Temporal Comp. | Patch Size | Latent Channels | Typical Tokens (5s, 720p) |
|-------|----------|--------------|---------------|------------|----------------|---------------------------|
| **Sora** (OpenAI) | 3D VAE | 8x | 4x (est.) | Spacetime: \((1,2,2)\) to \((2,2,2)\) | 16 (est.) | ~50,000-100,000 |
| **Veo 2/3** (Google) | Spatial VAE + temporal transformer | 8x | Learned (est. 4-8x) | \((1,2,2)\) | 8 (est.) | ~40,000-80,000 |
| **Kling 3.0** (Kuaishou) | 3D causal VAE | 8x | 4x | \((2,2,2)\) | 4 | ~27,000 |
| **Wan 2.2** (Alibaba) | 3D VAE | 8x | 4x | \((1,2,2)\) | 16 | ~54,000 |
| **CogVideoX** (Zhipu) | 3D causal VAE | 8x | 4x | \((1,2,2)\) | 16 | ~52,000 |
| **Runway Gen-3** | Proprietary | Est. 8x | Est. 4x | Proprietary | Proprietary | Est. ~40,000-60,000 |
| **LTX-2** (Lightricks) | 3D VAE | 32x | 8x | \((1,1,1)\) | 128 | ~21,600 |

### Key Architectural Differences

#### Sora: Spacetime Patches as the Core Innovation

Sora's technical report emphasizes "spacetime patches" as a unifying representation. The key insight: by using patches that span both space and time, Sora treats video generation as a single spatiotemporal problem rather than a sequence of spatial problems.

Sora operates on "visual patches" -- compressed tokens that can represent variable-duration, variable-resolution video. This enables:
- **Resolution flexibility**: The same model generates 1080p and 720p by adjusting spatial patch count
- **Duration flexibility**: Longer videos simply have more temporal patches
- **Aspect ratio flexibility**: Different aspect ratios are different spatial patch grid shapes

The estimated token count for Sora varies significantly by configuration. OpenAI has not published exact numbers, but based on generation times and reported compute requirements, the typical operating range appears to be 50,000-100,000 tokens for their standard outputs.

#### Veo: Separate Spatial and Temporal Processing

Google's Veo architecture (based on inferences from published research) appears to use a different strategy: spatial encoding with a 2D VAE, followed by temporal processing in the transformer itself.

This means temporal relationships are modeled entirely by the attention mechanism rather than being pre-compressed by the VAE. The advantage is that the transformer has full access to temporal dynamics. The disadvantage is higher token counts for the same duration.

Veo compensates for the higher token count with efficient attention mechanisms -- likely a combination of sliding window attention (local temporal) and global attention at reduced frequency.

#### Kling 3.0: Aggressive Compression for Speed

Kling 3.0 uses one of the most aggressive compression strategies: 8x spatial, 4x temporal, spacetime patches of \((2,2,2)\), and only 4 latent channels. This produces approximately 27,000 tokens for a 5-second 720p video -- roughly half what Sora or Wan use.

The result: Kling generates noticeably faster than competitors. The tradeoff is visible in fine temporal details and high-frequency spatial content, where Kling's output is slightly softer than Veo or Sora.

#### LTX-2: Extreme Spatial Compression

LTX-2 (Lightricks) takes an unconventional approach: very aggressive spatial compression (32x instead of the standard 8x) combined with a high latent channel count (128). This means each latent "pixel" represents a 32x32 spatial region in the original video, but carries 128 channels of information instead of 4.

$$
\text{LTX-2 latent}: \frac{720}{32} \times \frac{1280}{32} \times \frac{120}{8} = 22.5 \times 40 \times 15 = 13{,}500 \text{ spatial-temporal positions}
$$

With patch size \((1,1,1)\) (no additional patching), each latent position becomes one token with 128-dimensional patch content, projected to the model dimension.

The extremely low token count (~13,500) enables near-real-time generation but limits fine spatial detail. LTX-2 compensates with its high channel count, which carries more information per token than models with 4-channel latents.

### Quality Impact of Tokenization Choices

Here is how tokenization strategy maps to observable quality characteristics:

| Quality Aspect | High Spatial Comp. (32x) | Standard (8x) | Low Spatial Comp. (4x) |
|---------------|------------------------|----------------|----------------------|
| Fine textures | Blurry | Good | Excellent |
| Text in video | Illegible | Readable at large size | Clear |
| Hair/fur detail | Smooth blob | Visible strands | Individual strands |
| Skin pores | Not visible | Barely visible | Visible |
| Generation speed | Fast | Moderate | Slow |

| Quality Aspect | High Temporal Comp. (8x) | Standard (4x) | Low Temporal Comp. (2x) |
|---------------|--------------------------|----------------|------------------------|
| Smooth motion | Good for slow motion | Good | Excellent |
| Fast action | Motion blur / ghosting | Minor artifacts | Clean |
| Flickering | Possible | Rare | Very rare |
| Scene transitions | May skip frames | Clean | Perfect |
| Generation speed | Fast | Moderate | Slow |

### The Pareto Frontier

Plotting quality (FVD on a benchmark) against generation time reveals the Pareto frontier -- the set of configurations where you cannot improve quality without increasing time, or decrease time without losing quality:

```
Quality (FVD, lower = better)
    |
    |  * Veo 3.1 (high quality, slower)
 50 |
    |      * Sora 2 (balanced)
100 |
    |          * Wan 2.2 (good value)
150 |              * Kling 3.0 (fast, good quality)
    |
200 |                      * LTX-2 (fastest, lower quality)
    |
250 |
    +----+----+----+----+----+----+----+----+--->
      5s   10s  20s  40s  60s  90s 120s 180s
            Generation Time (5-second 720p clip)
```

The Pareto frontier suggests that current models are efficiently trading off quality and speed through tokenization choices. No model is obviously "wrong" in its approach -- each is targeting a different point on the frontier based on its intended use case.

---

## Conclusion

Video tokenization is the foundational bottleneck of video generation. Every architectural decision propagates through the pipeline:

**The compression cascade:**

$$
\underbrace{933M \text{ values}}_{\text{raw video}} \xrightarrow{\text{3D VAE (189x)}} \underbrace{4.9M \text{ values}}_{\text{latent}} \xrightarrow{\text{patchify (40x)}} \underbrace{54K-300K \text{ tokens}}_{\text{transformer input}}
$$

**What we have established:**

1. **Raw video is intractable.** A 5-second 1080p clip requires 933 million values. Self-attention over this many tokens would take years on current hardware.

2. **3D VAEs compress by ~189x** through spatial (8x) and temporal (4x) downsampling with strided 3D convolutions. The VAE reconstruction quality sets a hard ceiling on generation quality.

3. **Patchification reduces tokens by another 4-40x**, depending on patch size. Spacetime patches (Sora's approach) compress space and time simultaneously, halving token count per temporal factor of 2.

4. **Positional encoding in 3D** is essential for the transformer to understand spatial and temporal structure. RoPE is preferred for its relative position property, enabling potential generalization to longer sequences.

5. **Attention cost is quadratic** in token count. Doubling video duration quadruples compute cost. This is why longer videos are exponentially more expensive and why current models cap at 5-20 seconds per generation pass.

6. **Temporal compression is the key lever** for extending maximum duration. A model with 4x temporal compression generates 4x longer video at the same compute budget compared to 1x compression.

7. **Different models make different tradeoffs.** Kling prioritizes speed with aggressive compression. Veo prioritizes quality with more tokens. LTX-2 pushes extreme spatial compression for near-real-time generation.

For platform builders, understanding these tradeoffs informs model selection:

- **Preview / draft generation**: Choose models with aggressive tokenization (Kling, LTX-2) for speed
- **Final render / premium output**: Choose models with conservative tokenization (Veo, Sora) for quality
- **Long-form content**: Chain multiple short generations rather than fighting the \(O(N^2)\) scaling wall
- **Custom models**: If self-hosting, the VAE temporal compression factor is the single most impactful hyperparameter for balancing generation speed against temporal quality

The mathematics are clear: until attention mechanisms with sub-quadratic scaling (linear attention, state-space models) mature for video, tokenization strategy will remain the primary determinant of what is possible in video generation.
