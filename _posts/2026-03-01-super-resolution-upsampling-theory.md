---
layout: post
title: "Super-Resolution and Upsampling Theory: From Sampling Theorems to Diffusion-Based Video Upscaling"
date: 2026-03-01
category: math
---

Every video generation model faces a resolution problem. The diffusion process runs in a latent space that is 8× or 16× downsampled from the target resolution. The final step --- mapping from latent space back to full resolution --- is a form of super-resolution. And for models that generate at 512×512 and need to deliver at 4K, explicit super-resolution is the critical last mile.

Super-resolution is mathematically an **inverse problem**: recover a high-resolution signal from a low-resolution observation. It is fundamentally ill-posed --- infinitely many high-resolution images map to the same low-resolution image after downsampling. The richness of the field comes from the different ways to regularize this ill-posedness: from classical signal processing (Nyquist-Shannon, sinc interpolation) to learned approaches (SRCNN, ESRGAN) to modern diffusion-based methods that generate plausible high-resolution detail from a learned prior.

This post builds the theory from the Nyquist-Shannon sampling theorem, explains aliasing and why downsampled images lose information irreversibly, covers classical and learned upsampling methods, derives the perception-distortion tradeoff, and connects to diffusion-based video super-resolution.

---

## Table of Contents

1. [The Sampling Theorem](#the-sampling-theorem)
2. [Aliasing: When Sampling Goes Wrong](#aliasing-when-sampling-goes-wrong)
3. [Classical Upsampling Methods](#classical-upsampling-methods)
4. [Super-Resolution as an Inverse Problem](#super-resolution-as-an-inverse-problem)
5. [Regularization and MAP Estimation](#regularization-and-map-estimation)
6. [Sub-Pixel Convolution](#sub-pixel-convolution)
7. [Deep Residual Super-Resolution](#deep-residual-super-resolution)
8. [GAN-Based Super-Resolution](#gan-based-super-resolution)
9. [The Perception-Distortion Tradeoff](#the-perception-distortion-tradeoff)
10. [Diffusion-Based Super-Resolution](#diffusion-based-super-resolution)
11. [Video Super-Resolution](#video-super-resolution)
12. [Python: Sampling, Aliasing, and Upsampling](#python-sampling-aliasing-and-upsampling)

---

## The Sampling Theorem

The **Nyquist-Shannon sampling theorem** is the foundational result of signal processing. It tells us exactly when a continuous signal can be perfectly reconstructed from discrete samples --- and when it cannot.

### Statement

Let \(x(t)\) be a continuous signal that is **bandlimited** to frequency \(B\): its Fourier transform \(\hat{x}(f) = 0\) for all \(|f| > B\). Then \(x(t)\) is completely determined by its samples at rate \(f_s \geq 2B\):

$$x(t) = \sum_{n=-\infty}^{\infty} x[n] \, \text{sinc}\!\left(\frac{t - nT_s}{T_s}\right)$$

where \(T_s = 1/f_s\) is the sampling period and \(\text{sinc}(u) = \sin(\pi u) / (\pi u)\).

The **Nyquist rate** \(f_s = 2B\) is the minimum sampling rate for perfect reconstruction. Below this rate, the signal cannot be recovered --- information is irreversibly lost.

### Proof Sketch (via Fourier Transform)

Sampling a continuous signal \(x(t)\) at intervals \(T_s\) produces a discrete signal \(x[n] = x(nT_s)\). In the frequency domain, sampling corresponds to **periodization** of the spectrum:

$$\hat{x}_s(f) = \frac{1}{T_s} \sum_{k=-\infty}^{\infty} \hat{x}(f - k f_s)$$

The spectrum is replicated at integer multiples of \(f_s\). If \(f_s \geq 2B\), the replicas do not overlap, and we can recover \(\hat{x}(f)\) by applying an ideal low-pass filter at frequency \(B\). In the time domain, this filter is the sinc function, giving the reconstruction formula above.

If \(f_s < 2B\), the replicas overlap --- this is **aliasing**, and the original spectrum cannot be recovered.

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <defs>
    <marker id="arrowSR" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#d4d4d4"/>
    </marker>
  </defs>
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">Sampling Theorem: Adequate vs Insufficient Sampling</text>

  <!-- Top: adequate sampling (no aliasing) -->
  <text x="60" y="55" font-size="11" fill="#66bb6a" font-weight="bold">f_s ≥ 2B (no aliasing):</text>
  <line x1="80" y1="100" x2="630" y2="100" stroke="#666" stroke-width="1"/>
  <text x="635" y="104" font-size="9" fill="#999">f</text>
  <!-- Original spectrum -->
  <path d="M 280,100 L 300,60 L 340,55 L 380,60 L 400,100" fill="#4fc3f7" opacity="0.3" stroke="#4fc3f7" stroke-width="1.5"/>
  <text x="340" y="50" text-anchor="middle" font-size="9" fill="#4fc3f7">X(f)</text>
  <!-- Replicas (separated) -->
  <path d="M 80,100 L 100,70 L 140,65 L 180,70 L 200,100" fill="#4fc3f7" opacity="0.15" stroke="#4fc3f7" stroke-width="1" stroke-dasharray="3,2"/>
  <path d="M 480,100 L 500,70 L 540,65 L 580,70 L 600,100" fill="#4fc3f7" opacity="0.15" stroke="#4fc3f7" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="340" y="120" text-anchor="middle" font-size="9" fill="#66bb6a">Replicas don't overlap → perfect reconstruction</text>

  <!-- Bottom: insufficient sampling (aliasing) -->
  <text x="60" y="165" font-size="11" fill="#E53935" font-weight="bold">f_s &lt; 2B (aliasing!):</text>
  <line x1="80" y1="210" x2="630" y2="210" stroke="#666" stroke-width="1"/>
  <text x="635" y="214" font-size="9" fill="#999">f</text>
  <!-- Original spectrum -->
  <path d="M 240,210 L 270,170 L 340,165 L 410,170 L 440,210" fill="#4fc3f7" opacity="0.3" stroke="#4fc3f7" stroke-width="1.5"/>
  <!-- Overlapping replicas -->
  <path d="M 80,210 L 110,175 L 180,170 L 250,175 L 280,210" fill="#E53935" opacity="0.2" stroke="#E53935" stroke-width="1" stroke-dasharray="3,2"/>
  <path d="M 400,210 L 430,175 L 500,170 L 570,175 L 600,210" fill="#E53935" opacity="0.2" stroke="#E53935" stroke-width="1" stroke-dasharray="3,2"/>
  <!-- Overlap region -->
  <rect x="240" y="168" width="40" height="45" fill="#FF9800" opacity="0.15"/>
  <rect x="400" y="168" width="40" height="45" fill="#FF9800" opacity="0.15"/>
  <text x="260" y="250" text-anchor="middle" font-size="9" fill="#FF9800">overlap!</text>
  <text x="340" y="265" text-anchor="middle" font-size="9" fill="#E53935">Replicas overlap → information lost → cannot reconstruct</text>
</svg>

---

## Aliasing: When Sampling Goes Wrong

When a signal is sampled below the Nyquist rate, high-frequency components "fold" into lower frequencies, corrupting the signal. This is **aliasing**.

In images, aliasing appears as:
- **Moire patterns:** regular textures (brick walls, fences) produce wavy artifacts
- **Jagged edges:** diagonal lines show staircase patterns ("jaggies")
- **Color fringing:** high-frequency color transitions produce false colors

### The Anti-Aliasing Filter

To downsample an image without aliasing, first apply a **low-pass filter** that removes frequencies above the new Nyquist limit, then sample. For a 2× downsampling, the filter should cut off frequencies above half the original Nyquist rate.

In practice, a Gaussian blur is the standard anti-aliasing filter:

$$I_{\text{filtered}} = I * G_\sigma$$

where \(G_\sigma\) is a Gaussian kernel with \(\sigma \approx 0.5 \cdot (\text{downsample factor})\). Then subsample by keeping every \(k\)-th pixel.

**The information-theoretic point:** after anti-aliasing and downsampling, the high-frequency content is **gone**. It cannot be recovered from the downsampled image. Super-resolution must **hallucinate** the missing high frequencies using prior knowledge about what natural images look like.

---

## Classical Upsampling Methods

Given a low-resolution image \(\mathbf{y}\) at resolution \(H/k \times W/k\), produce a high-resolution image \(\hat{\mathbf{x}}\) at resolution \(H \times W\). Classical methods interpolate between known pixel values:

**Nearest neighbor:** Each output pixel takes the value of the nearest input pixel. Produces blocky, pixelated output. Zero frequency-domain smoothing --- preserves all existing frequencies but adds no new information.

**Bilinear interpolation:** Weighted average of the 4 nearest input pixels. Produces smoother output but still blurry --- the interpolation kernel is a triangle function, which has poor frequency-domain characteristics (slow rolloff, significant ripple).

**Bicubic interpolation:** Weighted average of the 16 nearest input pixels using a cubic polynomial kernel. Smoother than bilinear, with better frequency response. The standard default for image resizing.

**Lanczos interpolation:** Uses a windowed sinc function as the interpolation kernel. Closest to the ideal sinc reconstruction, with sharp frequency cutoff. Slightly overshoots at edges (ringing artifacts) but preserves the most detail.

All classical methods share the same fundamental limitation: they can only interpolate between known values. They cannot add high-frequency content that was lost during downsampling. The result is always blurry compared to a true high-resolution image.

---

## Super-Resolution as an Inverse Problem

The forward model for image degradation is:

$$\mathbf{y} = D H \mathbf{x} + \mathbf{n}$$

where:
- \(\mathbf{x} \in \mathbb{R}^{kH \times kW}\) is the unknown high-resolution image
- \(H\) is a blur operator (anti-aliasing filter)
- \(D\) is a downsampling operator (keep every \(k\)-th pixel)
- \(\mathbf{n}\) is observation noise
- \(\mathbf{y} \in \mathbb{R}^{H \times W}\) is the observed low-resolution image

This system is **ill-posed**: the operator \(DH\) maps a high-dimensional space to a low-dimensional space. Its null space is huge --- all the high-frequency components that were filtered out by \(H\) and lost by \(D\). Infinitely many \(\mathbf{x}\) values produce the same \(\mathbf{y}\).

For \(k = 4\) (4× super-resolution), each output pixel must be inferred from 1/16th of the information. The other 15/16ths must come from the **prior** --- our knowledge of what natural images look like.

---

## Regularization and MAP Estimation

To make the inverse problem well-posed, add a **regularizer** \(R(\mathbf{x})\) that encodes prior knowledge. The **MAP (maximum a posteriori)** estimate is:

$$\hat{\mathbf{x}} = \arg\min_\mathbf{x} \left[\underbrace{\|\mathbf{y} - DH\mathbf{x}\|_2^2}_{\text{data fidelity}} + \lambda \underbrace{R(\mathbf{x})}_{\text{regularizer}}\right]$$

The data fidelity term ensures the solution is consistent with the observation. The regularizer encodes what "natural" high-resolution images look like.

**Tikhonov regularization:** \(R(\mathbf{x}) = \|\mathbf{x}\|_2^2\). Penalizes large values, encouraging smoothness. Too simple for natural images.

**Total Variation (TV):** \(R(\mathbf{x}) = \|\nabla \mathbf{x}\|_1 = \sum_{i,j} |x_{i+1,j} - x_{i,j}| + |x_{i,j+1} - x_{i,j}|\). Penalizes the total gradient, encouraging piecewise-constant images. Produces sharp edges but cartoon-like textures.

**Sparse prior:** \(R(\mathbf{x}) = \|\Psi \mathbf{x}\|_1\) where \(\Psi\) is a sparsifying transform (wavelet, DCT). Natural images are sparse in these domains.

**Neural prior (deep image prior):** The regularization comes from the architecture of a neural network. Even an untrained CNN, when optimized to fit the observation, produces natural-looking images because its architecture biases toward smoothness at multiple scales.

---

## Sub-Pixel Convolution

**Sub-pixel convolution** (Shi et al., 2016, "Pixel Shuffle") is a computationally efficient upsampling layer used in many SR networks.

Instead of:
1. Upsampling to high resolution (computationally expensive)
2. Processing at high resolution (very expensive)

Sub-pixel convolution:
1. Processes at low resolution (cheap)
2. Outputs \(k^2 C\) channels instead of \(C\) channels
3. Rearranges (shuffles) these channels into a \(k \times k\) spatial grid, producing the high-resolution output

Mathematically, the pixel shuffle operation \(\mathcal{S}_k: \mathbb{R}^{H \times W \times k^2 C} \to \mathbb{R}^{kH \times kW \times C}\) is:

$$\mathcal{S}_k(\mathbf{Z})_{kh+i, kw+j, c} = \mathbf{Z}_{h, w, c \cdot k^2 + i \cdot k + j}$$

for \(i, j \in \{0, \ldots, k-1\}\). This is a deterministic rearrangement, not a learned operation. The learning happens in the convolution layers that produce the channel-expanded output.

**Advantage:** All computation happens at the low resolution. A \(4\times\) SR model processes features at 1/16th the pixel count compared to a model that upsamples first. This is 16× cheaper.

**Disadvantage:** The initial versions produced **checkerboard artifacts** because adjacent output pixels come from different convolution filters (different channel indices), creating systematic patterns. This is mitigated by careful initialization and by using sub-pixel convolution in the last layer only.

---

## Deep Residual Super-Resolution

Modern CNN-based SR models learn a **residual**: the difference between the bicubic-upsampled image and the ground-truth high-resolution image.

**SRCNN** (Dong et al., 2014): The first deep SR model. Three-layer CNN operating on the bicubic-upsampled input. Simple but effective.

**EDSR** (Lim et al., 2017): Deep residual network (32+ layers) with residual blocks, operating at the low resolution and using sub-pixel convolution for the final upsampling. Removes batch normalization (which hurts SR by normalizing away range information) and uses wide channels (256).

The residual learning formulation:

$$\hat{\mathbf{x}} = \text{bicubic}(\mathbf{y}) + F_\theta(\mathbf{y})$$

The network \(F_\theta\) only needs to predict the high-frequency detail (edges, textures) that bicubic interpolation misses. This is easier to learn than the full high-resolution image.

---

## GAN-Based Super-Resolution

**SRGAN** (Ledig et al., 2017) introduced adversarial training for SR, producing dramatically sharper results than MSE-trained models.

The generator loss combines:

$$\mathcal{L}_G = \lambda_{\text{pixel}} \|\hat{\mathbf{x}} - \mathbf{x}\|_1 + \lambda_{\text{perc}} \mathcal{L}_{\text{VGG}}(\hat{\mathbf{x}}, \mathbf{x}) + \lambda_{\text{adv}} \mathcal{L}_{\text{GAN}}(\hat{\mathbf{x}})$$

**ESRGAN** (Wang et al., 2018) improved on SRGAN with RRDB (Residual-in-Residual Dense Block) architecture, relativistic discriminator, and network interpolation between PSNR-oriented and perceptual-oriented models.

**Real-ESRGAN** (Wang et al., 2021) extended to real-world degradations: instead of training on clean downsampling (bicubic), train on realistic degradation pipelines (blur + noise + JPEG compression + downsampling in random order). This makes the model robust to real-world low-quality inputs.

---

## The Perception-Distortion Tradeoff

**Theorem (Blau & Michaeli, 2018).** Let \(G\) be an estimator of \(\mathbf{x}\) from \(\mathbf{y}\). Define:
- **Distortion:** \(\Delta = \mathbb{E}[d(\mathbf{x}, G(\mathbf{y}))]\) (e.g., MSE)
- **Perceptual quality:** \(d(p_G, p_X)\) where \(p_G\) is the distribution of \(G(\mathbf{y})\) and \(p_X\) is the real distribution

Then for any estimator:

$$\Delta \geq \Delta^*(d(p_G, p_X))$$

where \(\Delta^*\) is a monotonically decreasing function. You cannot simultaneously minimize distortion and make the output distribution match the real distribution.

**Proof intuition:** The distortion-optimal estimator is the conditional mean \(\mathbb{E}[\mathbf{x}|\mathbf{y}]\), which is blurry (has a different distribution from real images). To make \(p_G = p_X\), you must add variability (randomness), which increases MSE. The tradeoff is fundamental --- it holds for any estimator, learned or not.

**For video SR:** There is an additional temporal dimension to this tradeoff. Per-frame perceptual quality (sharpness) can conflict with temporal consistency (low flicker). A model that hallucinated different plausible high-frequency details for consecutive frames would score well on per-frame metrics but produce unwatchable flickering video.

---

## Diffusion-Based Super-Resolution

Diffusion models approach SR as **conditional generation**: given a low-resolution image \(\mathbf{y}\), sample from \(p(\mathbf{x} | \mathbf{y})\) --- the posterior distribution over high-resolution images consistent with \(\mathbf{y}\).

### Conditioning Mechanisms

**Concatenation:** Upsample \(\mathbf{y}\) to the target resolution (bicubic) and concatenate with the noisy image as additional input channels to the denoising network.

**Cross-attention:** Encode \(\mathbf{y}\) with a separate encoder and inject it via cross-attention into the denoising U-Net/DiT.

**SDEdit:** Start from a partially noised version of the upsampled \(\mathbf{y}\) (add noise to time \(t_0 < T\), not full noise) and denoise. The partial noise preserves the low-frequency content from \(\mathbf{y}\) while the denoising adds high-frequency detail.

### Why Diffusion SR Is Different

Unlike deterministic SR (EDSR, ESRGAN), diffusion SR is **stochastic**: each run produces a different plausible high-resolution image. This is a feature, not a bug --- it means the model samples from the posterior \(p(\mathbf{x}|\mathbf{y})\) rather than computing the posterior mean (which would be blurry).

Multiple samples from the same low-resolution input will have the same large-scale structure (same content, composition, colors) but different fine details (slightly different texture patterns, edge sharpness, noise structure). This is exactly the mathematical consequence of the ill-posedness: many high-res images are consistent with the same low-res input.

---

## Video Super-Resolution

Video SR has a crucial advantage over image SR: **multiple low-resolution frames contain complementary information.** Due to sub-pixel motion between frames (camera shake, object motion), different frames sample different sub-pixel positions, collectively providing more information than any single frame.

### Temporal Fusion

The basic pipeline:
1. **Align** neighboring frames to the reference frame using optical flow or deformable convolutions
2. **Fuse** the aligned frames (concatenation, attention, or recurrent processing)
3. **Upsample** the fused features using sub-pixel convolution

**BasicVSR** (Chan et al., 2021): Bidirectional recurrent network. Propagates features forward and backward in time using flow-based alignment, then fuses with a convolutional reconstruction module.

**RVRT** (Liang et al., 2022): Recurrent Video Restoration Transformer. Uses temporal attention for alignment (instead of explicit flow) and Transformer blocks for reconstruction.

### Diffusion-Based Video SR

**Upscale-A-Video** and similar methods extend diffusion SR to video:
1. Process video in overlapping temporal windows
2. Condition the diffusion model on the low-resolution video (concatenation)
3. Add temporal attention layers to maintain consistency across frames
4. Use flow-guided warping losses during training for temporal coherence

**NVIDIA RTX Video Super Resolution** takes a different approach: a lightweight CNN optimized for real-time processing (runs on the GPU's tensor cores at native display refresh rate). It sacrifices some quality for speed, targeting live upscaling of streaming video.

The key tradeoff for video SR: **quality vs temporal consistency vs speed**. Diffusion-based methods produce the highest quality but are slow and can flicker. CNN-based methods are fast and consistent but less detailed. Recurrent methods (BasicVSR) balance quality and consistency. Real-time methods (RTX VSR) sacrifice quality for speed. The right choice depends on the application.

---

## Python: Sampling, Aliasing, and Upsampling

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.ndimage import zoom

def create_test_signal(n_points=1024, freq_components=[(1, 1.0), (3, 0.7), (7, 0.4), (15, 0.3)]):
    """Create a 1D test signal with known frequency components."""
    t = np.linspace(0, 1, n_points, endpoint=False)
    signal = np.zeros(n_points)
    for freq, amp in freq_components:
        signal += amp * np.sin(2 * np.pi * freq * t)
    return t, signal

def downsample(signal, factor):
    """Downsample by taking every factor-th sample (with anti-aliasing)."""
    n = len(signal)
    # Anti-aliasing: low-pass filter
    from scipy.ndimage import gaussian_filter1d
    filtered = gaussian_filter1d(signal, sigma=factor/2)
    return filtered[::factor]

def downsample_no_aa(signal, factor):
    """Downsample without anti-aliasing (causes aliasing)."""
    return signal[::factor]

def upsample_methods(low_res, factor, original_length):
    """Compare different upsampling methods."""
    n_low = len(low_res)

    # Nearest neighbor
    nearest = np.repeat(low_res, factor)[:original_length]

    # Linear interpolation
    x_low = np.linspace(0, 1, n_low)
    x_high = np.linspace(0, 1, original_length)
    linear = np.interp(x_high, x_low, low_res)

    # Cubic (using scipy zoom)
    cubic = zoom(low_res, factor, order=3)[:original_length]

    # Sinc interpolation (ideal reconstruction)
    sinc = resample(low_res, original_length)

    return nearest, linear, cubic, sinc

# Create test signal
n = 1024
t, signal = create_test_signal(n)

# Downsample by factor 8 (with and without anti-aliasing)
factor = 8
low_res_aa = downsample(signal, factor)
low_res_no_aa = downsample_no_aa(signal, factor)

# Upsample using different methods
nearest, linear, cubic, sinc = upsample_methods(low_res_aa, factor, n)

# Also upsample the aliased version
_, _, _, sinc_aliased = upsample_methods(low_res_no_aa, factor, n)

# Compute frequency spectra
def spectrum(signal):
    """Compute normalized magnitude spectrum."""
    S = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal))
    return freqs, S / S.max()

# Visualization
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Original vs downsampled
axes[0, 0].plot(t[:200], signal[:200], 'b-', linewidth=1, alpha=0.5, label='Original')
t_low = np.linspace(0, 1, len(low_res_aa))
axes[0, 0].plot(t_low[:25], low_res_aa[:25], 'ro-', markersize=4, label=f'Downsampled {factor}×')
axes[0, 0].set_title('Signal and Samples')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlabel(r'Time $t$')

# Frequency domain: aliasing vs no aliasing
freqs_orig, spec_orig = spectrum(signal)
freqs_aa, spec_aa = spectrum(low_res_aa)
freqs_noaa, spec_noaa = spectrum(low_res_no_aa)

axes[0, 1].plot(freqs_orig * n, spec_orig, 'b-', linewidth=1, alpha=0.7, label='Original')
axes[0, 1].axvline(x=n/(2*factor), color='red', linestyle='--', alpha=0.5, label=f'Nyquist ({n/(2*factor):.0f} Hz)')
axes[0, 1].set_title('Frequency Spectrum')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlabel(r'Frequency $f$ (Hz)')
axes[0, 1].set_ylabel('Magnitude')
axes[0, 1].set_xlim(0, 30)

# Upsampling comparison (time domain)
methods = [('Nearest', nearest, '#E53935'),
           ('Linear', linear, '#FF9800'),
           ('Cubic', cubic, '#66bb6a'),
           ('Sinc', sinc, '#4fc3f7')]

for name, recon, color in methods:
    axes[1, 0].plot(t[:200], recon[:200], color=color, linewidth=1, alpha=0.7, label=name)
axes[1, 0].plot(t[:200], signal[:200], 'k--', linewidth=0.8, alpha=0.4, label='Original')
axes[1, 0].set_title('Upsampling Methods Comparison')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlabel(r'Time $t$')

# Error comparison
errors = {}
for name, recon, color in methods:
    err = np.mean((signal[:len(recon)] - recon[:len(signal)])**2)
    errors[name] = err

axes[1, 1].bar(errors.keys(), errors.values(),
               color=['#E53935', '#FF9800', '#66bb6a', '#4fc3f7'], alpha=0.7)
axes[1, 1].set_title('Reconstruction MSE')
axes[1, 1].set_ylabel('MSE')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Aliasing demonstration
axes[2, 0].plot(t[:200], signal[:200], 'b-', linewidth=1, alpha=0.4, label='Original')
axes[2, 0].plot(t[:200], sinc[:200], color='#66bb6a', linewidth=1.5, label='With anti-aliasing')
axes[2, 0].plot(t[:200], sinc_aliased[:200], color='#E53935', linewidth=1.5, alpha=0.7, label='Without anti-aliasing')
axes[2, 0].set_title('Effect of Anti-Aliasing on Reconstruction')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].set_xlabel(r'Time $t$')

# Perception-distortion tradeoff (conceptual plot)
distortion = np.linspace(0.01, 0.5, 100)
perception = 0.3 / distortion  # Conceptual tradeoff curve
perception = np.clip(perception, 0, 5)

axes[2, 1].plot(distortion, perception, color='#4fc3f7', linewidth=2.5)
axes[2, 1].fill_between(distortion, perception, 5, alpha=0.05, color='#66bb6a')
axes[2, 1].fill_between(distortion, 0, perception, alpha=0.05, color='#E53935')
axes[2, 1].text(0.35, 3.5, 'Achievable', fontsize=10, color='#66bb6a')
axes[2, 1].text(0.05, 1.0, 'Impossible', fontsize=10, color='#E53935')
axes[2, 1].scatter([0.4], [0.8], s=80, color='#FF9800', zorder=5, label='MSE-optimal (blurry)')
axes[2, 1].scatter([0.15], [2.5], s=80, color='#CE93D8', zorder=5, label='GAN-based (sharp)')
axes[2, 1].scatter([0.08], [4.0], s=80, color='#66bb6a', zorder=5, label='Diffusion (best)')
axes[2, 1].set_xlabel(r'Distortion $\Delta$ (MSE)')
axes[2, 1].set_ylabel(r'Perceptual Quality $1/d(p_G, p_X)$')
axes[2, 1].set_title('Perception-Distortion Tradeoff')
axes[2, 1].legend(fontsize=8)
axes[2, 1].grid(True, alpha=0.3)

plt.suptitle('Super-Resolution: Sampling Theory and Upsampling Methods', fontsize=14)
plt.tight_layout()
plt.savefig('super_resolution_theory.png', dpi=150, bbox_inches='tight')
plt.show()
```

Super-resolution for video generation sits at the intersection of classical signal processing and modern generative modeling. The sampling theorem tells us exactly what information is lost during downsampling. Classical upsampling can only interpolate --- it cannot add what was lost. The breakthrough of learned SR is using data-driven priors (CNNs, GANs, diffusion models) to hallucinate plausible high-frequency content. And the perception-distortion theorem tells us that this hallucination is not just a practical hack but a theoretical necessity: to produce sharp, realistic output, you must accept that it differs from the ground truth in MSE. The highest quality comes from diffusion-based SR, which samples from the full posterior distribution over high-resolution images --- but at the cost of stochasticity and compute. For real-time video, the tradeoff tips toward lightweight, deterministic methods that sacrifice some detail for speed and consistency.
