---
layout: post
title: "Temporal Attention: The Mathematical Architecture That Keeps Video Frames Consistent"
date: 2026-01-22
category: math
---

Why do AI-generated videos sometimes produce a person who changes face mid-scene? Why does a car spontaneously gain an extra wheel? And why have these artifacts become dramatically less common since 2024?

The answer is temporal attention -- a specific mathematical mechanism that allows each frame of a generated video to "look at" other frames and maintain consistency. This post is a full mathematical derivation of temporal attention: from the basic self-attention equation through spatial and temporal decomposition, causal masking, windowed attention strategies, the quadratic scaling problem, multi-head specialization, and Flash Attention.

Every equation is derived from first principles. Every complexity claim is justified. If you want to understand the machinery that makes video generation work at the architectural level, this is the reference.

---

## Table of Contents

1. [Self-Attention: The Foundation](#self-attention-the-foundation)
2. [Why Scale by sqrt(d_k): A Variance Analysis](#why-scale-by-sqrtd_k-a-variance-analysis)
3. [Spatial Attention: Within a Frame](#spatial-attention-within-a-frame)
4. [Temporal Attention: Across Frames](#temporal-attention-across-frames)
5. [Combined Spatiotemporal Attention](#combined-spatiotemporal-attention)
6. [Causal vs Bidirectional Temporal Attention](#causal-vs-bidirectional-temporal-attention)
7. [Attention Window Strategies](#attention-window-strategies)
8. [The Quadratic Problem: A Concrete Calculation](#the-quadratic-problem-a-concrete-calculation)
9. [Multi-Head Attention for Video](#multi-head-attention-for-video)
10. [Flash Attention for Video](#flash-attention-for-video)
11. [Putting It All Together](#putting-it-all-together)
12. [Conclusion](#conclusion)

---

## Self-Attention: The Foundation

Self-attention is the core operation in transformer architectures. Given an input sequence of $N$ tokens, each represented as a $d$-dimensional vector, self-attention allows every token to aggregate information from every other token.

### The Mechanism

Let the input be a matrix $X \in \mathbb{R}^{N \times d}$, where each row is a token embedding. Self-attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

where:
- $Q = XW^Q \in \mathbb{R}^{N \times d_k}$ are the **queries** ("what am I looking for?")
- $K = XW^K \in \mathbb{R}^{N \times d_k}$ are the **keys** ("what do I contain?")
- $V = XW^V \in \mathbb{R}^{N \times d_v}$ are the **values** ("what information do I provide?")
- $W^Q, W^K \in \mathbb{R}^{d \times d_k}$ and $W^V \in \mathbb{R}^{d \times d_v}$ are learned projection matrices

### Step-by-Step Computation

**Step 1: Compute attention scores.** The raw attention score between query $i$ and key $j$ is:

$$e_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} = \frac{\sum_{l=1}^{d_k} q_{il} k_{jl}}{\sqrt{d_k}}$$

This is a scaled dot product. The matrix $QK^T$ has dimensions $N \times N$ -- every query attends to every key.

**Step 2: Normalize with softmax.** The attention weights are:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{m=1}^{N} \exp(e_{im})}$$

Each row of the attention matrix sums to 1. Token $i$ distributes its attention across all tokens, with higher-scoring tokens receiving more weight.

**Step 3: Aggregate values.** The output for token $i$ is:

$$o_i = \sum_{j=1}^{N} \alpha_{ij} v_j$$

A weighted sum of all value vectors, where the weights are the attention probabilities.

### Complexity

| Operation | Computation | Memory |
|---|---|---|
| $QK^T$ | $O(N^2 d_k)$ | $O(N^2)$ for the attention matrix |
| Softmax | $O(N^2)$ | $O(N^2)$ |
| Attention $\times$ V | $O(N^2 d_v)$ | $O(N d_v)$ for output |
| **Total** | **$O(N^2 d_k)$** | **$O(N^2 + N d_v)$** |

The quadratic dependence on $N$ is the fundamental bottleneck for long sequences -- and video has very long sequences.

---

## Why Scale by sqrt(d_k): A Variance Analysis

The scaling factor $\frac{1}{\sqrt{d_k}}$ is not arbitrary. It prevents the softmax from saturating, which would kill gradients. Let us derive why.

### Setup

Assume the components of $q_i$ and $k_j$ are independent random variables with mean 0 and variance 1 (this is approximately true after standard initialization):

$$q_{il} \sim \mathcal{N}(0, 1), \quad k_{jl} \sim \mathcal{N}(0, 1), \quad \text{for } l = 1, \ldots, d_k$$

### Computing the Variance of the Dot Product

The unnormalized attention score is:

$$s_{ij} = q_i \cdot k_j = \sum_{l=1}^{d_k} q_{il} k_{jl}$$

Each term $q_{il} k_{jl}$ is a product of two independent standard normals:

$$\mathbb{E}[q_{il} k_{jl}] = \mathbb{E}[q_{il}] \cdot \mathbb{E}[k_{jl}] = 0 \cdot 0 = 0$$

$$\text{Var}(q_{il} k_{jl}) = \mathbb{E}[(q_{il} k_{jl})^2] - (\mathbb{E}[q_{il} k_{jl}])^2 = \mathbb{E}[q_{il}^2] \cdot \mathbb{E}[k_{jl}^2] - 0 = 1 \cdot 1 = 1$$

Since the $d_k$ terms are independent:

$$\mathbb{E}[s_{ij}] = 0$$

$$\text{Var}(s_{ij}) = d_k$$

So $s_{ij} \sim \mathcal{N}(0, d_k)$ approximately (by CLT for large $d_k$).

### The Problem Without Scaling

If $d_k = 512$, then $s_{ij}$ has standard deviation $\sqrt{512} \approx 22.6$. The softmax of values with magnitude ~22 is extremely peaked:

$$\text{softmax}(22) = \frac{e^{22}}{e^{22} + e^{-22} + \cdots} \approx 1.0$$

The attention becomes essentially a hard argmax -- one token gets all the weight, all others get approximately zero. Gradients through the softmax vanish because:

$$\frac{\partial \alpha_i}{\partial e_j} = \alpha_i (\delta_{ij} - \alpha_j) \approx 0 \text{ when } \alpha_i \approx 0 \text{ or } \alpha_i \approx 1$$

### The Fix: Scaling

Dividing by $\sqrt{d_k}$:

$$e_{ij} = \frac{s_{ij}}{\sqrt{d_k}}$$

$$\text{Var}(e_{ij}) = \frac{\text{Var}(s_{ij})}{d_k} = \frac{d_k}{d_k} = 1$$

Now $e_{ij} \sim \mathcal{N}(0, 1)$, regardless of $d_k$. The softmax receives inputs with unit variance, producing well-distributed attention weights and healthy gradients.

### Numerical Verification

| $d_k$ | $\text{Std}(s_{ij})$ unscaled | $\text{Std}(e_{ij})$ scaled | Softmax behavior |
|---|---|---|---|
| 16 | 4.0 | 1.0 | Healthy |
| 64 | 8.0 | 1.0 | Healthy |
| 256 | 16.0 | 1.0 | Healthy |
| 512 | 22.6 | 1.0 | Healthy |
| 1024 | 32.0 | 1.0 | Healthy |

Without scaling, the softmax saturates as $d_k$ grows. With scaling, the distribution of attention logits is stable regardless of $d_k$.

---

## Spatial Attention: Within a Frame

In video models, spatial attention operates **within a single frame**. Each frame is divided into patches (tokens), and these tokens attend to each other to capture spatial relationships.

### Tokenization

A frame of resolution $H \times W$ with patch size $p$ produces:

$$N_s = \frac{H}{p} \times \frac{W}{p}$$

spatial tokens per frame.

For a 1080p frame (1920 x 1080) with patch size 16:

$$N_s = \frac{1920}{16} \times \frac{1080}{16} = 120 \times 68 = 8{,}160 \text{ tokens}$$

For a 512x512 frame with patch size 8 (common in latent space, where the VAE has already compressed 4x to 8x):

$$N_s = \frac{512}{8} \times \frac{512}{8} = 64 \times 64 = 4{,}096 \text{ tokens}$$

### Spatial Attention Computation

For a single frame, spatial self-attention is:

$$\text{SpatialAttn}(X_f) = \text{softmax}\left(\frac{Q_f K_f^T}{\sqrt{d_k}}\right) V_f$$

where $X_f \in \mathbb{R}^{N_s \times d}$ contains the tokens from frame $f$.

**Complexity per frame**: $O(N_s^2 \cdot d_k)$

**Memory per frame**: $O(N_s^2)$ for the attention matrix

For $N_s = 4{,}096$ and $d_k = 64$:
- Attention matrix size: $4{,}096^2 = 16{,}777{,}216$ entries = 64 MB (float32)
- FLOPs for $QK^T$: $2 \times 4{,}096^2 \times 64 \approx 2.1 \times 10^9$

### What Spatial Attention Captures

Each patch can attend to every other patch in the same frame. This enables:

- **Global context**: A patch in the corner "knows" what is in the center
- **Spatial coherence**: Edges align, textures are consistent, objects have correct proportions
- **Compositional understanding**: The model can reason about spatial relationships between objects

<svg viewBox="0 0 700 500" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#333">Spatial vs Temporal Attention Patterns</text>
  <!-- Left: Spatial attention -->
  <text x="190" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#4fc3f7">Spatial Attention</text>
  <text x="190" y="72" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#999">Within each frame independently</text>
  <!-- Frame grid - Frame 1 -->
  <rect x="60" y="85" width="120" height="90" fill="none" stroke="#4fc3f7" stroke-width="2" rx="3"/>
  <text x="120" y="80" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Frame 1</text>
  <!-- 4x3 grid of tokens -->
  <rect x="65" y="90" width="25" height="25" fill="#4fc3f7" fill-opacity="0.3" stroke="#4fc3f7" stroke-width="0.5"/>
  <rect x="95" y="90" width="25" height="25" fill="#4fc3f7" fill-opacity="0.15" stroke="#4fc3f7" stroke-width="0.5"/>
  <rect x="125" y="90" width="25" height="25" fill="#4fc3f7" fill-opacity="0.15" stroke="#4fc3f7" stroke-width="0.5"/>
  <rect x="155" y="90" width="25" height="25" fill="#4fc3f7" fill-opacity="0.1" stroke="#4fc3f7" stroke-width="0.5"/>
  <rect x="65" y="120" width="25" height="25" fill="#4fc3f7" fill-opacity="0.15" stroke="#4fc3f7" stroke-width="0.5"/>
  <rect x="95" y="120" width="25" height="25" fill="#4fc3f7" fill-opacity="0.8" stroke="#4fc3f7" stroke-width="1.5"/>
  <rect x="125" y="120" width="25" height="25" fill="#4fc3f7" fill-opacity="0.2" stroke="#4fc3f7" stroke-width="0.5"/>
  <rect x="155" y="120" width="25" height="25" fill="#4fc3f7" fill-opacity="0.1" stroke="#4fc3f7" stroke-width="0.5"/>
  <rect x="65" y="150" width="25" height="25" fill="#4fc3f7" fill-opacity="0.1" stroke="#4fc3f7" stroke-width="0.5"/>
  <rect x="95" y="150" width="25" height="25" fill="#4fc3f7" fill-opacity="0.15" stroke="#4fc3f7" stroke-width="0.5"/>
  <rect x="125" y="150" width="25" height="25" fill="#4fc3f7" fill-opacity="0.1" stroke="#4fc3f7" stroke-width="0.5"/>
  <rect x="155" y="150" width="25" height="25" fill="#4fc3f7" fill-opacity="0.05" stroke="#4fc3f7" stroke-width="0.5"/>
  <!-- Highlight center token -->
  <circle cx="108" cy="133" r="3" fill="#ef5350"/>
  <!-- Arrows from center to all others -->
  <line x1="108" y1="133" x2="78" y2="103" stroke="#4fc3f7" stroke-width="1" opacity="0.5"/>
  <line x1="108" y1="133" x2="108" y2="103" stroke="#4fc3f7" stroke-width="1" opacity="0.5"/>
  <line x1="108" y1="133" x2="138" y2="103" stroke="#4fc3f7" stroke-width="1" opacity="0.4"/>
  <line x1="108" y1="133" x2="168" y2="103" stroke="#4fc3f7" stroke-width="1" opacity="0.3"/>
  <line x1="108" y1="133" x2="78" y2="133" stroke="#4fc3f7" stroke-width="1" opacity="0.5"/>
  <line x1="108" y1="133" x2="138" y2="133" stroke="#4fc3f7" stroke-width="1" opacity="0.5"/>
  <line x1="108" y1="133" x2="168" y2="133" stroke="#4fc3f7" stroke-width="1" opacity="0.3"/>
  <line x1="108" y1="133" x2="78" y2="163" stroke="#4fc3f7" stroke-width="1" opacity="0.3"/>
  <line x1="108" y1="133" x2="108" y2="163" stroke="#4fc3f7" stroke-width="1" opacity="0.4"/>
  <line x1="108" y1="133" x2="138" y2="163" stroke="#4fc3f7" stroke-width="1" opacity="0.3"/>
  <line x1="108" y1="133" x2="168" y2="163" stroke="#4fc3f7" stroke-width="1" opacity="0.2"/>
  <!-- Frame 2 (independent, no cross-frame connections) -->
  <rect x="210" y="85" width="120" height="90" fill="none" stroke="#4fc3f7" stroke-width="2" rx="3"/>
  <text x="270" y="80" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Frame 2</text>
  <rect x="215" y="90" width="25" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <rect x="245" y="90" width="25" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <rect x="275" y="90" width="25" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <rect x="305" y="90" width="25" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <rect x="215" y="120" width="25" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <rect x="245" y="120" width="25" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <rect x="275" y="120" width="25" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <rect x="305" y="120" width="25" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <rect x="215" y="150" width="25" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <rect x="245" y="150" width="25" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <rect x="275" y="150" width="25" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <rect x="305" y="150" width="25" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <text x="270" y="195" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#999">(processed separately)</text>
  <!-- Complexity -->
  <text x="190" y="220" text-anchor="middle" font-family="monospace" font-size="12" fill="#4fc3f7">O(N_s^2) per frame</text>
  <!-- Right: Temporal attention -->
  <text x="530" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#ef5350">Temporal Attention</text>
  <text x="530" y="72" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#999">Same position across frames</text>
  <!-- 4 frames stacked vertically -->
  <rect x="420" y="85" width="120" height="35" fill="none" stroke="#ccc" stroke-width="1" rx="2"/>
  <text x="410" y="107" text-anchor="end" font-family="Arial, sans-serif" font-size="10" fill="#666">t=1</text>
  <rect x="420" y="130" width="120" height="35" fill="none" stroke="#ccc" stroke-width="1" rx="2"/>
  <text x="410" y="152" text-anchor="end" font-family="Arial, sans-serif" font-size="10" fill="#666">t=2</text>
  <rect x="420" y="175" width="120" height="35" fill="none" stroke="#ccc" stroke-width="1" rx="2"/>
  <text x="410" y="197" text-anchor="end" font-family="Arial, sans-serif" font-size="10" fill="#666">t=3</text>
  <rect x="420" y="220" width="120" height="35" fill="none" stroke="#ccc" stroke-width="1" rx="2"/>
  <text x="410" y="242" text-anchor="end" font-family="Arial, sans-serif" font-size="10" fill="#666">t=4</text>
  <!-- Tokens in each frame - highlight the same spatial position -->
  <!-- Frame 1 tokens -->
  <rect x="425" y="90" width="22" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <rect x="452" y="90" width="22" height="25" fill="#ef5350" fill-opacity="0.3" stroke="#ef5350" stroke-width="1.5"/>
  <rect x="479" y="90" width="22" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <rect x="506" y="90" width="22" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <!-- Frame 2 tokens -->
  <rect x="425" y="135" width="22" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <rect x="452" y="135" width="22" height="25" fill="#ef5350" fill-opacity="0.6" stroke="#ef5350" stroke-width="1.5"/>
  <rect x="479" y="135" width="22" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <rect x="506" y="135" width="22" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <!-- Frame 3 tokens -->
  <rect x="425" y="180" width="22" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <rect x="452" y="180" width="22" height="25" fill="#ef5350" fill-opacity="0.8" stroke="#ef5350" stroke-width="1.5"/>
  <rect x="479" y="180" width="22" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <rect x="506" y="180" width="22" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <!-- Frame 4 tokens -->
  <rect x="425" y="225" width="22" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <rect x="452" y="225" width="22" height="25" fill="#ef5350" fill-opacity="1.0" stroke="#ef5350" stroke-width="1.5"/>
  <rect x="479" y="225" width="22" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <rect x="506" y="225" width="22" height="25" fill="#e0e0e0" stroke="#ccc" stroke-width="0.5"/>
  <!-- Temporal connections (vertical) -->
  <line x1="463" y1="115" x2="463" y2="135" stroke="#ef5350" stroke-width="2"/>
  <line x1="463" y1="160" x2="463" y2="180" stroke="#ef5350" stroke-width="2"/>
  <line x1="463" y1="205" x2="463" y2="225" stroke="#ef5350" stroke-width="2"/>
  <!-- Side arrows showing all attend to all -->
  <path d="M 550,103 C 570,103 570,148 550,148" fill="none" stroke="#ef5350" stroke-width="1.5"/>
  <path d="M 550,103 C 585,103 585,193 550,193" fill="none" stroke="#ef5350" stroke-width="1"/>
  <path d="M 550,103 C 600,103 600,238 550,238" fill="none" stroke="#ef5350" stroke-width="0.8"/>
  <text x="610" y="170" font-family="Arial, sans-serif" font-size="10" fill="#ef5350">All frames</text>
  <text x="610" y="182" font-family="Arial, sans-serif" font-size="10" fill="#ef5350">attend to</text>
  <text x="610" y="194" font-family="Arial, sans-serif" font-size="10" fill="#ef5350">each other</text>
  <!-- Complexity -->
  <text x="530" y="280" text-anchor="middle" font-family="monospace" font-size="12" fill="#ef5350">O(T^2) per position</text>
  <!-- Bottom: Combined -->
  <rect x="60" y="310" width="600" height="170" rx="8" fill="#f8f8f8" stroke="#ddd" stroke-width="1"/>
  <text x="360" y="340" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#333">Combined Spatiotemporal (Factored)</text>
  <text x="360" y="365" text-anchor="middle" font-family="monospace" font-size="13" fill="#333">Total: O(N_s^2 * T + N_s * T^2)</text>
  <text x="360" y="395" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#666">Alternating spatial and temporal layers instead of full 3D attention</text>
  <text x="360" y="425" text-anchor="middle" font-family="monospace" font-size="12" fill="#ef5350">Full 3D would be: O((N_s * T)^2) -- prohibitively expensive</text>
  <!-- Comparison numbers -->
  <text x="180" y="460" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#8bc34a">Factored: ~2.5 TFLOPs</text>
  <text x="480" y="460" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#ef5350">Full 3D: ~2,400 TFLOPs</text>
  <text x="360" y="475" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#999">(for 150 frames, 4096 spatial tokens)</text>
</svg>

---

## Temporal Attention: Across Frames

Temporal attention connects the same spatial position across different frames. This is the mechanism responsible for temporal consistency -- ensuring that an object at position $(x, y)$ looks the same from frame to frame.

### The Mechanism

Consider a video with $T$ frames, each containing $N_s$ spatial tokens. For temporal attention, we take all tokens at a specific spatial position $(h, w)$ across all frames and form a temporal sequence:

$$X^{(h,w)} = [x_1^{(h,w)}, x_2^{(h,w)}, \ldots, x_T^{(h,w)}] \in \mathbb{R}^{T \times d}$$

Temporal self-attention is then:

$$\text{TemporalAttn}(X^{(h,w)}) = \text{softmax}\left(\frac{Q^{(h,w)} {K^{(h,w)}}^T}{\sqrt{d_k}}\right) V^{(h,w)}$$

where $Q^{(h,w)}, K^{(h,w)}, V^{(h,w)}$ are computed from $X^{(h,w)}$.

### What Temporal Attention Captures

When frame $t$ attends to frame $t-5$, it "asks": What was happening at this spatial position 5 frames ago? The attention weights encode temporal dependencies:

- **Color consistency**: A blue car stays blue across frames
- **Motion continuity**: An object moving right continues moving right
- **Structural coherence**: A building does not gain or lose floors
- **Appearance stability**: A face maintains the same features

### Complexity

For each spatial position, temporal attention has complexity $O(T^2)$. Since there are $N_s$ spatial positions:

**Total temporal attention complexity**: $O(N_s \cdot T^2 \cdot d_k)$

**Memory**: $O(N_s \cdot T^2)$ for all the temporal attention matrices

For a 5-second, 30fps video in latent space ($N_s = 4{,}096$, $T = 150$):
- $N_s \cdot T^2 = 4{,}096 \times 22{,}500 = 92{,}160{,}000$ attention entries per layer
- Memory: $\sim$352 MB per layer (float32)

---

## Combined Spatiotemporal Attention

### The Full 3D Attention Problem

The ideal approach would be full 3D attention, where every spatiotemporal token attends to every other:

$$\text{Full3DAttn}(X) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

where $X \in \mathbb{R}^{(N_s \cdot T) \times d}$ contains all tokens from all frames.

**Complexity**: $O((N_s \cdot T)^2 \cdot d_k)$

For our 5-second, 30fps example:
- $N_s \cdot T = 4{,}096 \times 150 = 614{,}400$ tokens
- Attention matrix: $614{,}400^2 \approx 3.77 \times 10^{11}$ entries
- Memory: ~1.4 TB (float32) for the attention matrix alone

This is completely infeasible. No GPU can hold a 1.4 TB attention matrix. Even with float16, it is 700 GB.

### The Factored Decomposition

The solution is to **decompose** full 3D attention into alternating spatial and temporal attention layers:

```
For each transformer block:
    1. Spatial attention:  Each frame attends within itself    O(N_s^2 * T)
    2. Temporal attention: Each position attends across frames  O(N_s * T^2)
```

**Total complexity per block**: $O(N_s^2 \cdot T + N_s \cdot T^2)$

This is dramatically cheaper:

| Approach | Complexity | $N_s=4096, T=150$ | Memory |
|---|---|---|---|
| Full 3D | $O(N_s^2 T^2)$ | $\sim 3.8 \times 10^{11}$ | ~1.4 TB |
| Factored | $O(N_s^2 T + N_s T^2)$ | $\sim 2.6 \times 10^{9}$ | ~9.9 GB |
| **Ratio** | | **~146x cheaper** | **~146x less memory** |

The factored approach loses something: spatial position $(h_1, w_1)$ in frame $t_1$ cannot directly attend to spatial position $(h_2, w_2)$ in frame $t_2$ in a single attention operation. It requires two sequential operations -- first temporal attention aligns the time step, then spatial attention allows cross-position communication (or vice versa). In practice, with multiple layers, information can propagate across both space and time effectively.

### Mathematical Justification

The factored decomposition can be viewed as approximating the full 3D attention matrix $A \in \mathbb{R}^{(N_s T) \times (N_s T)}$ as:

$$A \approx A_s \otimes I_T + I_{N_s} \otimes A_t$$

where $A_s$ is the spatial attention matrix, $A_t$ is the temporal attention matrix, and $\otimes$ denotes the Kronecker product. This is a low-rank approximation to the full attention matrix.

A deeper justification comes from the observation that spatial and temporal correlations in video are approximately separable -- the spatial structure within a frame is largely independent of the temporal dynamics, except at object boundaries and during rapid motion. The factored approach captures each independently and lets information flow between them through multiple layers.

---

## Causal vs Bidirectional Temporal Attention

### Bidirectional Temporal Attention

In bidirectional attention, every frame can attend to every other frame -- both past and future:

$$\alpha_{t,t'} = \frac{\exp(e_{t,t'})}{\sum_{s=1}^{T} \exp(e_{t,s})}, \quad \forall t, t' \in \{1, \ldots, T\}$$

This is used when **all frames are generated simultaneously** (parallel generation), as in most video diffusion models. Since the denoising process refines all frames at once, there is no notion of "past" or "future" -- all frames exist simultaneously in the noisy state.

**Advantage**: Maximum information flow. Every frame benefits from every other frame.

**Use case**: Diffusion-based video models (Stable Video Diffusion, Wan 2.x, Sora, Veo).

### Causal Temporal Attention

In causal attention, each frame can only attend to itself and past frames:

$$\alpha_{t,t'} = \begin{cases} \frac{\exp(e_{t,t'})}{\sum_{s=1}^{t} \exp(e_{t,s})} & \text{if } t' \leq t \\ 0 & \text{if } t' > t \end{cases}$$

This is enforced by an attention mask that sets future positions to $-\infty$ before the softmax:

$$\text{CausalAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

where $M$ is the causal mask:

$$M_{t,t'} = \begin{cases} 0 & \text{if } t' \leq t \\ -\infty & \text{if } t' > t \end{cases}$$

**Advantage**: Enables autoregressive generation (frame by frame). Can generate videos of arbitrary length by processing frame by frame with KV caching.

**Use case**: Autoregressive video models, real-time/streaming generation.

<svg viewBox="0 0 700 380" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#333">Causal Attention Mask Visualization</text>
  <!-- Left: Causal mask matrix -->
  <text x="175" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="bold" fill="#333">Causal Mask (6 frames)</text>
  <!-- Column headers -->
  <text x="108" y="78" text-anchor="middle" font-family="monospace" font-size="9" fill="#999">K₁</text>
  <text x="138" y="78" text-anchor="middle" font-family="monospace" font-size="9" fill="#999">K₂</text>
  <text x="168" y="78" text-anchor="middle" font-family="monospace" font-size="9" fill="#999">K₃</text>
  <text x="198" y="78" text-anchor="middle" font-family="monospace" font-size="9" fill="#999">K₄</text>
  <text x="228" y="78" text-anchor="middle" font-family="monospace" font-size="9" fill="#999">K₅</text>
  <text x="258" y="78" text-anchor="middle" font-family="monospace" font-size="9" fill="#999">K₆</text>
  <!-- Row headers -->
  <text x="80" y="102" text-anchor="end" font-family="monospace" font-size="9" fill="#999">Q₁</text>
  <text x="80" y="132" text-anchor="end" font-family="monospace" font-size="9" fill="#999">Q₂</text>
  <text x="80" y="162" text-anchor="end" font-family="monospace" font-size="9" fill="#999">Q₃</text>
  <text x="80" y="192" text-anchor="end" font-family="monospace" font-size="9" fill="#999">Q₄</text>
  <text x="80" y="222" text-anchor="end" font-family="monospace" font-size="9" fill="#999">Q₅</text>
  <text x="80" y="252" text-anchor="end" font-family="monospace" font-size="9" fill="#999">Q₆</text>
  <!-- Row 1: [1, 0, 0, 0, 0, 0] -->
  <rect x="93" y="86" width="28" height="24" fill="#4fc3f7" fill-opacity="0.6" stroke="white" stroke-width="1"/>
  <rect x="123" y="86" width="28" height="24" fill="#ef5350" fill-opacity="0.15" stroke="white" stroke-width="1"/>
  <rect x="153" y="86" width="28" height="24" fill="#ef5350" fill-opacity="0.15" stroke="white" stroke-width="1"/>
  <rect x="183" y="86" width="28" height="24" fill="#ef5350" fill-opacity="0.15" stroke="white" stroke-width="1"/>
  <rect x="213" y="86" width="28" height="24" fill="#ef5350" fill-opacity="0.15" stroke="white" stroke-width="1"/>
  <rect x="243" y="86" width="28" height="24" fill="#ef5350" fill-opacity="0.15" stroke="white" stroke-width="1"/>
  <text x="107" y="102" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <text x="137" y="102" text-anchor="middle" font-family="monospace" font-size="9" fill="#ccc">-inf</text>
  <text x="167" y="102" text-anchor="middle" font-family="monospace" font-size="9" fill="#ccc">-inf</text>
  <text x="197" y="102" text-anchor="middle" font-family="monospace" font-size="9" fill="#ccc">-inf</text>
  <text x="227" y="102" text-anchor="middle" font-family="monospace" font-size="9" fill="#ccc">-inf</text>
  <text x="257" y="102" text-anchor="middle" font-family="monospace" font-size="9" fill="#ccc">-inf</text>
  <!-- Row 2 -->
  <rect x="93" y="116" width="28" height="24" fill="#4fc3f7" fill-opacity="0.4" stroke="white" stroke-width="1"/>
  <rect x="123" y="116" width="28" height="24" fill="#4fc3f7" fill-opacity="0.6" stroke="white" stroke-width="1"/>
  <rect x="153" y="116" width="28" height="24" fill="#ef5350" fill-opacity="0.15" stroke="white" stroke-width="1"/>
  <rect x="183" y="116" width="28" height="24" fill="#ef5350" fill-opacity="0.15" stroke="white" stroke-width="1"/>
  <rect x="213" y="116" width="28" height="24" fill="#ef5350" fill-opacity="0.15" stroke="white" stroke-width="1"/>
  <rect x="243" y="116" width="28" height="24" fill="#ef5350" fill-opacity="0.15" stroke="white" stroke-width="1"/>
  <text x="107" y="132" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <text x="137" y="132" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <text x="167" y="132" text-anchor="middle" font-family="monospace" font-size="9" fill="#ccc">-inf</text>
  <text x="197" y="132" text-anchor="middle" font-family="monospace" font-size="9" fill="#ccc">-inf</text>
  <text x="227" y="132" text-anchor="middle" font-family="monospace" font-size="9" fill="#ccc">-inf</text>
  <text x="257" y="132" text-anchor="middle" font-family="monospace" font-size="9" fill="#ccc">-inf</text>
  <!-- Row 3 -->
  <rect x="93" y="146" width="28" height="24" fill="#4fc3f7" fill-opacity="0.3" stroke="white" stroke-width="1"/>
  <rect x="123" y="146" width="28" height="24" fill="#4fc3f7" fill-opacity="0.4" stroke="white" stroke-width="1"/>
  <rect x="153" y="146" width="28" height="24" fill="#4fc3f7" fill-opacity="0.6" stroke="white" stroke-width="1"/>
  <rect x="183" y="146" width="28" height="24" fill="#ef5350" fill-opacity="0.15" stroke="white" stroke-width="1"/>
  <rect x="213" y="146" width="28" height="24" fill="#ef5350" fill-opacity="0.15" stroke="white" stroke-width="1"/>
  <rect x="243" y="146" width="28" height="24" fill="#ef5350" fill-opacity="0.15" stroke="white" stroke-width="1"/>
  <text x="107" y="162" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <text x="137" y="162" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <text x="167" y="162" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <text x="197" y="162" text-anchor="middle" font-family="monospace" font-size="9" fill="#ccc">-inf</text>
  <text x="227" y="162" text-anchor="middle" font-family="monospace" font-size="9" fill="#ccc">-inf</text>
  <text x="257" y="162" text-anchor="middle" font-family="monospace" font-size="9" fill="#ccc">-inf</text>
  <!-- Row 4 -->
  <rect x="93" y="176" width="28" height="24" fill="#4fc3f7" fill-opacity="0.2" stroke="white" stroke-width="1"/>
  <rect x="123" y="176" width="28" height="24" fill="#4fc3f7" fill-opacity="0.3" stroke="white" stroke-width="1"/>
  <rect x="153" y="176" width="28" height="24" fill="#4fc3f7" fill-opacity="0.4" stroke="white" stroke-width="1"/>
  <rect x="183" y="176" width="28" height="24" fill="#4fc3f7" fill-opacity="0.6" stroke="white" stroke-width="1"/>
  <rect x="213" y="176" width="28" height="24" fill="#ef5350" fill-opacity="0.15" stroke="white" stroke-width="1"/>
  <rect x="243" y="176" width="28" height="24" fill="#ef5350" fill-opacity="0.15" stroke="white" stroke-width="1"/>
  <text x="107" y="192" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <text x="137" y="192" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <text x="167" y="192" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <text x="197" y="192" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <text x="227" y="192" text-anchor="middle" font-family="monospace" font-size="9" fill="#ccc">-inf</text>
  <text x="257" y="192" text-anchor="middle" font-family="monospace" font-size="9" fill="#ccc">-inf</text>
  <!-- Row 5 -->
  <rect x="93" y="206" width="28" height="24" fill="#4fc3f7" fill-opacity="0.15" stroke="white" stroke-width="1"/>
  <rect x="123" y="206" width="28" height="24" fill="#4fc3f7" fill-opacity="0.2" stroke="white" stroke-width="1"/>
  <rect x="153" y="206" width="28" height="24" fill="#4fc3f7" fill-opacity="0.3" stroke="white" stroke-width="1"/>
  <rect x="183" y="206" width="28" height="24" fill="#4fc3f7" fill-opacity="0.4" stroke="white" stroke-width="1"/>
  <rect x="213" y="206" width="28" height="24" fill="#4fc3f7" fill-opacity="0.6" stroke="white" stroke-width="1"/>
  <rect x="243" y="206" width="28" height="24" fill="#ef5350" fill-opacity="0.15" stroke="white" stroke-width="1"/>
  <text x="107" y="222" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <text x="137" y="222" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <text x="167" y="222" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <text x="197" y="222" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <text x="227" y="222" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <text x="257" y="222" text-anchor="middle" font-family="monospace" font-size="9" fill="#ccc">-inf</text>
  <!-- Row 6 -->
  <rect x="93" y="236" width="28" height="24" fill="#4fc3f7" fill-opacity="0.1" stroke="white" stroke-width="1"/>
  <rect x="123" y="236" width="28" height="24" fill="#4fc3f7" fill-opacity="0.15" stroke="white" stroke-width="1"/>
  <rect x="153" y="236" width="28" height="24" fill="#4fc3f7" fill-opacity="0.2" stroke="white" stroke-width="1"/>
  <rect x="183" y="236" width="28" height="24" fill="#4fc3f7" fill-opacity="0.3" stroke="white" stroke-width="1"/>
  <rect x="213" y="236" width="28" height="24" fill="#4fc3f7" fill-opacity="0.4" stroke="white" stroke-width="1"/>
  <rect x="243" y="236" width="28" height="24" fill="#4fc3f7" fill-opacity="0.6" stroke="white" stroke-width="1"/>
  <text x="107" y="252" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <text x="137" y="252" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <text x="167" y="252" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <text x="197" y="252" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <text x="227" y="252" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <text x="257" y="252" text-anchor="middle" font-family="monospace" font-size="9" fill="#333">0</text>
  <!-- Legend -->
  <rect x="93" y="275" width="15" height="12" fill="#4fc3f7" fill-opacity="0.5"/>
  <text x="115" y="285" font-family="Arial, sans-serif" font-size="10" fill="#666">Attended (mask = 0)</text>
  <rect x="200" y="275" width="15" height="12" fill="#ef5350" fill-opacity="0.15"/>
  <text x="222" y="285" font-family="Arial, sans-serif" font-size="10" fill="#666">Masked (mask = -inf)</text>
  <!-- Right: explanation -->
  <text x="500" y="100" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="bold" fill="#333">Key Properties</text>
  <text x="370" y="125" font-family="Arial, sans-serif" font-size="11" fill="#666">Frame 1: sees only itself</text>
  <text x="370" y="145" font-family="Arial, sans-serif" font-size="11" fill="#666">Frame 2: sees frames 1-2</text>
  <text x="370" y="165" font-family="Arial, sans-serif" font-size="11" fill="#666">Frame 3: sees frames 1-3</text>
  <text x="370" y="185" font-family="Arial, sans-serif" font-size="11" fill="#666">...</text>
  <text x="370" y="205" font-family="Arial, sans-serif" font-size="11" fill="#666">Frame T: sees all frames</text>
  <rect x="365" y="225" width="270" height="55" rx="5" fill="#fff3e0" stroke="#ff9800" stroke-width="1"/>
  <text x="500" y="245" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#e65100">Causal = autoregressive generation.</text>
  <text x="500" y="262" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#e65100">Generate frame by frame, left to right.</text>
  <text x="500" y="279" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#e65100">Enables streaming and infinite length.</text>
  <!-- Bidirectional note -->
  <rect x="365" y="295" width="270" height="45" rx="5" fill="#e8f5e9" stroke="#4caf50" stroke-width="1"/>
  <text x="500" y="313" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#2e7d32">Bidirectional = all 0s (no mask).</text>
  <text x="500" y="330" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#2e7d32">All frames see all frames. Better quality.</text>
</svg>

---

## Attention Window Strategies

Full temporal attention (every frame attends to every other frame) becomes expensive as $T$ grows. Several windowed strategies reduce the cost while preserving most of the quality.

### Strategy 1: Full Temporal Attention

Every frame attends to every other frame.

**Complexity**: $O(T^2)$ per spatial position

**Effective receptive field**: Global -- frame 1 can directly influence frame $T$

**When to use**: Short videos ($T < 100$), or when global coherence is critical

### Strategy 2: Sliding Window Attention

Each frame attends only to the $K$ nearest frames (in both directions):

$$\alpha_{t,t'} = 0 \quad \text{if} \quad |t - t'| > K/2$$

**Complexity**: $O(T \cdot K)$ per spatial position

**Effective receptive field**: Local, $K$ frames. After $L$ layers, the effective receptive field is $L \cdot K$ frames.

**When to use**: Long videos where local consistency matters more than global coherence. $K = 16$ or $K = 32$ is typical.

### Strategy 3: Dilated Attention

Attend to frames at exponentially increasing intervals:

$$\text{Attended frames for frame } t: \{t-1, t-2, t-4, t-8, t-16, \ldots\}$$

**Complexity**: $O(T \cdot \log T)$ per spatial position

**Effective receptive field**: Global (logarithmic span), but sparse. Recent frames have dense coverage; distant frames have sparse coverage.

**When to use**: Long videos where both local detail and global structure matter. Particularly effective when combined with sliding window in alternating layers.

### Strategy 4: Chunk-Based Attention

Divide the video into chunks of $C$ frames. Full attention within each chunk; a global attention layer connects chunk representatives:

$$\text{Within chunk: } O(C^2), \quad \text{Across chunks: } O((T/C)^2)$$

**Complexity**: $O(T \cdot C + T^2/C)$, minimized when $C = \sqrt{T}$, giving $O(T^{3/2})$

**When to use**: Very long videos (minutes). This is the approach behind hierarchical generation.

### Complexity Comparison

| Strategy | Complexity (per pos.) | Memory (per pos.) | $T=150$ | $T=600$ | $T=2400$ |
|---|---|---|---|---|---|
| Full | $O(T^2)$ | $O(T^2)$ | 22.5K | 360K | 5.76M |
| Sliding ($K=32$) | $O(TK)$ | $O(TK)$ | 4.8K | 19.2K | 76.8K |
| Dilated | $O(T \log T)$ | $O(T \log T)$ | 1.1K | 5.4K | 27.4K |
| Chunk ($C=\sqrt{T}$) | $O(T^{3/2})$ | $O(T^{3/2})$ | 1.8K | 14.7K | 117.6K |

<svg viewBox="0 0 700 480" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#333">Attention Window Strategies Comparison</text>
  <text x="350" y="42" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#999">Each row = one frame's attention pattern across 16 frames</text>
  <!-- Strategy 1: Full Attention -->
  <text x="60" y="75" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#4fc3f7">Full Attention</text>
  <text x="60" y="90" font-family="Arial, sans-serif" font-size="10" fill="#999">O(T^2)</text>
  <!-- 16x16 grid, all filled -->
  <g transform="translate(180, 60)">
    <!-- All cells filled -->
    <rect x="0" y="0" width="320" height="48" fill="#4fc3f7" fill-opacity="0.25" stroke="#4fc3f7" stroke-width="0.5" rx="2"/>
    <!-- Grid lines -->
    <line x1="20" y1="0" x2="20" y2="48" stroke="#4fc3f7" stroke-width="0.3"/>
    <line x1="40" y1="0" x2="40" y2="48" stroke="#4fc3f7" stroke-width="0.3"/>
    <line x1="60" y1="0" x2="60" y2="48" stroke="#4fc3f7" stroke-width="0.3"/>
    <line x1="80" y1="0" x2="80" y2="48" stroke="#4fc3f7" stroke-width="0.3"/>
    <line x1="100" y1="0" x2="100" y2="48" stroke="#4fc3f7" stroke-width="0.3"/>
    <line x1="120" y1="0" x2="120" y2="48" stroke="#4fc3f7" stroke-width="0.3"/>
    <line x1="140" y1="0" x2="140" y2="48" stroke="#4fc3f7" stroke-width="0.3"/>
    <line x1="160" y1="0" x2="160" y2="48" stroke="#4fc3f7" stroke-width="0.3"/>
    <line x1="180" y1="0" x2="180" y2="48" stroke="#4fc3f7" stroke-width="0.3"/>
    <line x1="200" y1="0" x2="200" y2="48" stroke="#4fc3f7" stroke-width="0.3"/>
    <line x1="220" y1="0" x2="220" y2="48" stroke="#4fc3f7" stroke-width="0.3"/>
    <line x1="240" y1="0" x2="240" y2="48" stroke="#4fc3f7" stroke-width="0.3"/>
    <line x1="260" y1="0" x2="260" y2="48" stroke="#4fc3f7" stroke-width="0.3"/>
    <line x1="280" y1="0" x2="280" y2="48" stroke="#4fc3f7" stroke-width="0.3"/>
    <line x1="300" y1="0" x2="300" y2="48" stroke="#4fc3f7" stroke-width="0.3"/>
    <line x1="0" y1="16" x2="320" y2="16" stroke="#4fc3f7" stroke-width="0.3"/>
    <line x1="0" y1="32" x2="320" y2="32" stroke="#4fc3f7" stroke-width="0.3"/>
    <!-- Highlight query row -->
    <rect x="0" y="16" width="320" height="16" fill="#4fc3f7" fill-opacity="0.3" stroke="none"/>
    <text x="330" y="28" font-family="Arial, sans-serif" font-size="9" fill="#4fc3f7">query frame</text>
  </g>
  <text x="560" y="90" font-family="Arial, sans-serif" font-size="10" fill="#666">Every frame sees</text>
  <text x="560" y="102" font-family="Arial, sans-serif" font-size="10" fill="#666">every other frame</text>
  <!-- Strategy 2: Sliding Window (K=6) -->
  <text x="60" y="145" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#ef5350">Sliding Window</text>
  <text x="60" y="160" font-family="Arial, sans-serif" font-size="10" fill="#999">O(T*K), K=6</text>
  <g transform="translate(180, 130)">
    <rect x="0" y="0" width="320" height="48" fill="#f5f5f5" stroke="#ddd" stroke-width="0.5" rx="2"/>
    <!-- Grid lines -->
    <line x1="20" y1="0" x2="20" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="40" y1="0" x2="40" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="60" y1="0" x2="60" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="80" y1="0" x2="80" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="100" y1="0" x2="100" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="120" y1="0" x2="120" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="140" y1="0" x2="140" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="160" y1="0" x2="160" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="180" y1="0" x2="180" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="200" y1="0" x2="200" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="220" y1="0" x2="220" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="240" y1="0" x2="240" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="260" y1="0" x2="260" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="280" y1="0" x2="280" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="300" y1="0" x2="300" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="0" y1="16" x2="320" y2="16" stroke="#ddd" stroke-width="0.3"/>
    <line x1="0" y1="32" x2="320" y2="32" stroke="#ddd" stroke-width="0.3"/>
    <!-- Window around middle row (frames 4-10 for query at frame 7) -->
    <rect x="80" y="16" width="120" height="16" fill="#ef5350" fill-opacity="0.4" stroke="#ef5350" stroke-width="1"/>
    <!-- Other rows get their own windows -->
    <rect x="0" y="0" width="120" height="16" fill="#ef5350" fill-opacity="0.2" stroke="none"/>
    <rect x="200" y="32" width="120" height="16" fill="#ef5350" fill-opacity="0.2" stroke="none"/>
    <text x="330" y="28" font-family="Arial, sans-serif" font-size="9" fill="#ef5350">window</text>
  </g>
  <text x="560" y="158" font-family="Arial, sans-serif" font-size="10" fill="#666">Each frame sees K</text>
  <text x="560" y="170" font-family="Arial, sans-serif" font-size="10" fill="#666">nearest neighbors</text>
  <!-- Strategy 3: Dilated -->
  <text x="60" y="215" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#8bc34a">Dilated Attention</text>
  <text x="60" y="230" font-family="Arial, sans-serif" font-size="10" fill="#999">O(T*log T)</text>
  <g transform="translate(180, 200)">
    <rect x="0" y="0" width="320" height="48" fill="#f5f5f5" stroke="#ddd" stroke-width="0.5" rx="2"/>
    <!-- Grid lines -->
    <line x1="20" y1="0" x2="20" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="40" y1="0" x2="40" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="60" y1="0" x2="60" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="80" y1="0" x2="80" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="100" y1="0" x2="100" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="120" y1="0" x2="120" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="140" y1="0" x2="140" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="160" y1="0" x2="160" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="180" y1="0" x2="180" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="200" y1="0" x2="200" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="220" y1="0" x2="220" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="240" y1="0" x2="240" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="260" y1="0" x2="260" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="280" y1="0" x2="280" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="300" y1="0" x2="300" y2="48" stroke="#ddd" stroke-width="0.3"/>
    <line x1="0" y1="16" x2="320" y2="16" stroke="#ddd" stroke-width="0.3"/>
    <line x1="0" y1="32" x2="320" y2="32" stroke="#ddd" stroke-width="0.3"/>
    <!-- Dilated pattern: query at frame 8, attends to 7,6,4,0 (i.e., -1,-2,-4,-8) -->
    <rect x="140" y="16" width="20" height="16" fill="#8bc34a" fill-opacity="0.6" stroke="#8bc34a" stroke-width="1"/>
    <rect x="120" y="16" width="20" height="16" fill="#8bc34a" fill-opacity="0.5" stroke="#8bc34a" stroke-width="1"/>
    <rect x="100" y="16" width="20" height="16" fill="#8bc34a" fill-opacity="0.4" stroke="#8bc34a" stroke-width="1"/>
    <rect x="60" y="16" width="20" height="16" fill="#8bc34a" fill-opacity="0.3" stroke="#8bc34a" stroke-width="1"/>
    <rect x="160" y="16" width="20" height="16" fill="#8bc34a" fill-opacity="0.5" stroke="#8bc34a" stroke-width="1"/>
    <rect x="180" y="16" width="20" height="16" fill="#8bc34a" fill-opacity="0.4" stroke="#8bc34a" stroke-width="1"/>
    <rect x="220" y="16" width="20" height="16" fill="#8bc34a" fill-opacity="0.3" stroke="#8bc34a" stroke-width="1"/>
    <rect x="300" y="16" width="20" height="16" fill="#8bc34a" fill-opacity="0.2" stroke="#8bc34a" stroke-width="1"/>
    <text x="330" y="28" font-family="Arial, sans-serif" font-size="9" fill="#8bc34a">sparse</text>
  </g>
  <text x="560" y="225" font-family="Arial, sans-serif" font-size="10" fill="#666">Exponentially spaced:</text>
  <text x="560" y="237" font-family="Arial, sans-serif" font-size="10" fill="#666">t-1, t-2, t-4, t-8...</text>
  <!-- Strategy 4: Chunk-based -->
  <text x="60" y="285" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#ff9800">Chunk-Based</text>
  <text x="60" y="300" font-family="Arial, sans-serif" font-size="10" fill="#999">O(T^3/2)</text>
  <g transform="translate(180, 270)">
    <rect x="0" y="0" width="320" height="48" fill="#f5f5f5" stroke="#ddd" stroke-width="0.5" rx="2"/>
    <!-- Chunk boundaries -->
    <rect x="0" y="0" width="80" height="48" fill="#ff9800" fill-opacity="0.15" stroke="#ff9800" stroke-width="1.5" rx="2"/>
    <rect x="80" y="0" width="80" height="48" fill="#ff9800" fill-opacity="0.15" stroke="#ff9800" stroke-width="1.5" rx="2"/>
    <rect x="160" y="0" width="80" height="48" fill="#ff9800" fill-opacity="0.15" stroke="#ff9800" stroke-width="1.5" rx="2"/>
    <rect x="240" y="0" width="80" height="48" fill="#ff9800" fill-opacity="0.15" stroke="#ff9800" stroke-width="1.5" rx="2"/>
    <!-- Full attention within chunk 2 (highlighted) -->
    <rect x="80" y="16" width="80" height="16" fill="#ff9800" fill-opacity="0.4" stroke="#ff9800" stroke-width="1"/>
    <!-- Cross-chunk representatives -->
    <rect x="35" y="16" width="10" height="16" fill="#ff9800" fill-opacity="0.25" stroke="#ff9800" stroke-width="0.5"/>
    <rect x="195" y="16" width="10" height="16" fill="#ff9800" fill-opacity="0.25" stroke="#ff9800" stroke-width="0.5"/>
    <rect x="275" y="16" width="10" height="16" fill="#ff9800" fill-opacity="0.25" stroke="#ff9800" stroke-width="0.5"/>
    <!-- Chunk labels -->
    <text x="40" y="60" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#ff9800">Chunk 1</text>
    <text x="120" y="60" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#ff9800">Chunk 2</text>
    <text x="200" y="60" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#ff9800">Chunk 3</text>
    <text x="280" y="60" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#ff9800">Chunk 4</text>
  </g>
  <text x="560" y="295" font-family="Arial, sans-serif" font-size="10" fill="#666">Full within chunks,</text>
  <text x="560" y="307" font-family="Arial, sans-serif" font-size="10" fill="#666">representatives across</text>
  <!-- Cost scaling chart at bottom -->
  <text x="350" y="370" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="bold" fill="#333">Relative Cost Scaling (T=150 baseline = 1.0)</text>
  <!-- Bar chart -->
  <g transform="translate(100, 380)">
    <!-- Full -->
    <rect x="0" y="0" width="100" height="25" fill="#4fc3f7" fill-opacity="0.6" rx="3"/>
    <text x="105" y="17" font-family="Arial, sans-serif" font-size="11" fill="#333">Full: 1.0x</text>
    <!-- Sliding -->
    <rect x="0" y="30" width="21" height="25" fill="#ef5350" fill-opacity="0.6" rx="3"/>
    <text x="27" y="47" font-family="Arial, sans-serif" font-size="11" fill="#333">Sliding: 0.21x</text>
    <!-- Dilated -->
    <rect x="0" y="60" width="5" height="25" fill="#8bc34a" fill-opacity="0.6" rx="2"/>
    <text x="12" y="77" font-family="Arial, sans-serif" font-size="11" fill="#333">Dilated: 0.05x</text>
    <!-- Chunk -->
    <rect x="0" y="90" width="8" height="25" fill="#ff9800" fill-opacity="0.6" rx="2"/>
    <text x="15" y="107" font-family="Arial, sans-serif" font-size="11" fill="#333">Chunk: 0.08x</text>
  </g>
</svg>

---

## The Quadratic Problem: A Concrete Calculation

Let us calculate the exact cost of generating a 5-second, 30fps video at 1080p resolution, and show why doubling the length costs 4x more.

### Assumptions

| Parameter | Value | Notes |
|---|---|---|
| Resolution | 1920 x 1080 | 1080p |
| Frame rate | 30 fps | Standard |
| Duration | 5 seconds | 150 frames |
| VAE spatial compression | 8x | Each dimension compressed 8x |
| VAE temporal compression | 4x | 4 frames per latent frame |
| Patch size (in latent space) | 2 x 2 | Standard for DiTs |
| Latent channels | 16 | Common for video VAEs |
| Attention head dim ($d_k$) | 64 | Standard |
| Number of heads | 24 | Typical for large models |
| Number of transformer blocks | 28 | Typical for 14B models |

### Derived Quantities

**Spatial tokens per frame:**

$$N_s = \frac{1920/8}{2} \times \frac{1080/8}{2} = 120 \times 68 = 8{,}160$$

Wait -- let us be more careful. The VAE compresses by 8x spatially, then the DiT patchifies by 2x:

$$N_s = \frac{240}{2} \times \frac{135}{2} = 120 \times 67 = 8{,}040 \approx 8{,}000$$

**Temporal tokens (latent frames):**

$$T_l = \frac{150}{4} = 37.5 \approx 38$$

### Spatial Attention Cost (per block)

$$\text{FLOPs}_{\text{spatial}} = T_l \times 2 \times N_s^2 \times d_k \times n_{\text{heads}}$$

$$= 38 \times 2 \times 8{,}000^2 \times 64 \times 24$$

$$= 38 \times 2 \times 64{,}000{,}000 \times 64 \times 24$$

$$= 38 \times 2 \times 64 \times 10^6 \times 1{,}536$$

$$\approx 7.5 \times 10^{12} \text{ FLOPs} = 7.5 \text{ TFLOPs}$$

### Temporal Attention Cost (per block)

$$\text{FLOPs}_{\text{temporal}} = N_s \times 2 \times T_l^2 \times d_k \times n_{\text{heads}}$$

$$= 8{,}000 \times 2 \times 38^2 \times 64 \times 24$$

$$= 8{,}000 \times 2 \times 1{,}444 \times 1{,}536$$

$$\approx 3.6 \times 10^{10} \text{ FLOPs} = 0.036 \text{ TFLOPs}$$

### Total per Block and Overall

| Component | FLOPs per block | Fraction |
|---|---|---|
| Spatial attention | 7.5 TFLOPs | 99.5% |
| Temporal attention | 0.036 TFLOPs | 0.5% |
| **Total attention** | **~7.5 TFLOPs** | 100% |

Including FFN layers (roughly equal to attention in FLOPs):

$$\text{Total per block} \approx 15 \text{ TFLOPs}$$

$$\text{Total for 28 blocks} \approx 420 \text{ TFLOPs}$$

For 50 diffusion steps:

$$\text{Total generation} \approx 420 \times 50 = 21{,}000 \text{ TFLOPs}$$

### The Quadratic Scaling: 5 vs 10 Seconds

Now let us double the duration to 10 seconds (300 frames, 75 latent frames):

| Component | 5 seconds ($T_l=38$) | 10 seconds ($T_l=75$) | Ratio |
|---|---|---|---|
| Spatial attention | $38 \times 8000^2$ | $75 \times 8000^2$ | 1.97x |
| Temporal attention | $8000 \times 38^2$ | $8000 \times 75^2$ | 3.89x |
| Total (spatial-dominated) | ~7.5 TFLOPs/block | ~14.8 TFLOPs/block | ~1.97x |
| Memory (temporal attn matrix) | $8000 \times 38^2 = 11.6M$ | $8000 \times 75^2 = 45M$ | 3.89x |

Spatial attention scales linearly with $T$ (it is $O(N_s^2 T)$), but temporal attention scales quadratically ($O(N_s T^2)$). For short videos, spatial attention dominates. But as videos get longer, temporal attention becomes the bottleneck.

The **crossover point** where temporal attention exceeds spatial attention:

$$N_s T^2 > N_s^2 T \implies T > N_s$$

For $N_s = 8{,}000$, this occurs at $T = 8{,}000$ latent frames, which corresponds to 32,000 actual frames (about 17 minutes at 30fps). For shorter videos, spatial attention dominates. For very long videos, temporal attention becomes the bottleneck.

<svg viewBox="0 0 700 420" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#333">Attention Cost Scaling: Duration vs Compute</text>
  <!-- Axes -->
  <defs>
    <marker id="arrowAx" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#666"/>
    </marker>
  </defs>
  <line x1="80" y1="370" x2="650" y2="370" stroke="#666" stroke-width="1.5" marker-end="url(#arrowAx)"/>
  <line x1="80" y1="370" x2="80" y2="45" stroke="#666" stroke-width="1.5" marker-end="url(#arrowAx)"/>
  <text x="365" y="400" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#666">Video Duration (seconds)</text>
  <text x="25" y="210" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#666" transform="rotate(-90, 25, 210)">Relative Compute Cost</text>
  <!-- Grid lines -->
  <line x1="80" y1="310" x2="640" y2="310" stroke="#f0f0f0" stroke-width="1"/>
  <line x1="80" y1="250" x2="640" y2="250" stroke="#f0f0f0" stroke-width="1"/>
  <line x1="80" y1="190" x2="640" y2="190" stroke="#f0f0f0" stroke-width="1"/>
  <line x1="80" y1="130" x2="640" y2="130" stroke="#f0f0f0" stroke-width="1"/>
  <line x1="80" y1="70" x2="640" y2="70" stroke="#f0f0f0" stroke-width="1"/>
  <!-- X axis labels -->
  <text x="80" y="388" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#999">0</text>
  <text x="192" y="388" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#999">5s</text>
  <text x="304" y="388" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#999">10s</text>
  <text x="416" y="388" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#999">15s</text>
  <text x="528" y="388" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#999">20s</text>
  <text x="640" y="388" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#999">25s</text>
  <!-- Y axis labels -->
  <text x="70" y="374" text-anchor="end" font-family="Arial, sans-serif" font-size="10" fill="#999">0x</text>
  <text x="70" y="314" text-anchor="end" font-family="Arial, sans-serif" font-size="10" fill="#999">5x</text>
  <text x="70" y="254" text-anchor="end" font-family="Arial, sans-serif" font-size="10" fill="#999">10x</text>
  <text x="70" y="194" text-anchor="end" font-family="Arial, sans-serif" font-size="10" fill="#999">15x</text>
  <text x="70" y="134" text-anchor="end" font-family="Arial, sans-serif" font-size="10" fill="#999">20x</text>
  <text x="70" y="74" text-anchor="end" font-family="Arial, sans-serif" font-size="10" fill="#999">25x</text>
  <!-- Linear scaling (ideal) - dashed -->
  <path d="M 80,370 L 192,358 L 304,346 L 416,334 L 528,322 L 640,310" fill="none" stroke="#8bc34a" stroke-width="2" stroke-dasharray="6,4"/>
  <text x="645" y="305" font-family="Arial, sans-serif" font-size="10" fill="#8bc34a">Linear (ideal)</text>
  <!-- Actual quadratic scaling (temporal attention dominates at long durations) -->
  <!-- 5s=1x, 10s=~3.9x, 15s=~8.8x, 20s=~15.6x, 25s=~24.4x -->
  <path d="M 80,370 L 192,358 L 304,323 L 416,265 L 528,183 L 640,78" fill="none" stroke="#ef5350" stroke-width="2.5"/>
  <!-- Data points -->
  <circle cx="192" cy="358" r="5" fill="#ef5350" stroke="white" stroke-width="2"/>
  <circle cx="304" cy="323" r="5" fill="#ef5350" stroke="white" stroke-width="2"/>
  <circle cx="416" cy="265" r="5" fill="#ef5350" stroke="white" stroke-width="2"/>
  <circle cx="528" cy="183" r="5" fill="#ef5350" stroke="white" stroke-width="2"/>
  <circle cx="640" cy="78" r="5" fill="#ef5350" stroke="white" stroke-width="2"/>
  <!-- Labels on data points -->
  <text x="192" y="348" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#ef5350" font-weight="bold">1x</text>
  <text x="320" y="315" font-family="Arial, sans-serif" font-size="10" fill="#ef5350" font-weight="bold">~3.9x</text>
  <text x="432" y="258" font-family="Arial, sans-serif" font-size="10" fill="#ef5350" font-weight="bold">~8.8x</text>
  <text x="544" y="175" font-family="Arial, sans-serif" font-size="10" fill="#ef5350" font-weight="bold">~15.6x</text>
  <text x="620" y="68" font-family="Arial, sans-serif" font-size="10" fill="#ef5350" font-weight="bold">~24.4x</text>
  <text x="645" y="100" font-family="Arial, sans-serif" font-size="10" fill="#ef5350">Quadratic (actual)</text>
  <!-- Annotation -->
  <rect x="140" y="55" width="280" height="40" rx="5" fill="#fff3e0" stroke="#ff9800" stroke-width="1"/>
  <text x="280" y="72" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#e65100">10s costs ~4x more than 5s, not 2x.</text>
  <text x="280" y="87" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#e65100">This is why longer videos are so expensive.</text>
</svg>

---

## Multi-Head Attention for Video

### The Mechanism

Multi-head attention runs $h$ independent attention operations in parallel, each with its own learned projections:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

where:

$$\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)$$

Each head has dimensions $d_k = d_v = d_{\text{model}} / h$. With $d_{\text{model}} = 1536$ and $h = 24$ heads, each head operates on $d_k = 64$ dimensions.

### Why Multiple Heads Matter for Video

In language models, different heads learn to attend to different types of relationships (syntactic, semantic, positional). In video models, empirical analysis shows similar specialization:

**Head type 1: Color consistency heads.** These heads have high attention weights between tokens of the same object across frames, focusing on color channels. They ensure a red car stays red.

**Head type 2: Motion tracking heads.** These heads attend to spatially displaced tokens across frames, following the trajectory of moving objects. They predict where an object will be in the next frame.

**Head type 3: Structural coherence heads.** These heads attend to edges and boundaries, maintaining the structural integrity of objects. They prevent buildings from melting or faces from deforming.

**Head type 4: Global context heads.** These heads have broad, diffuse attention patterns, maintaining the overall scene layout and lighting consistency.

This specialization is not programmed -- it emerges from training. The multiple heads provide the model with enough "slots" to learn these different functions simultaneously.

### Mathematical Justification for Multi-Head

Why not just use a single head with $d_k = d_{\text{model}}$? The answer is that multi-head attention allows the model to attend to information from different representation subspaces at different positions simultaneously:

$$\text{SingleHead: } \alpha_{ij} = \text{softmax}\left(\frac{q_i \cdot k_j}{\sqrt{d_{\text{model}}}}\right)$$

With a single head, token $i$ has one attention distribution over all other tokens. It must compromise between attending to the token that provides color information and the token that provides motion information.

With multiple heads:

$$\alpha_{ij}^{(1)} \neq \alpha_{ij}^{(2)} \neq \cdots \neq \alpha_{ij}^{(h)}$$

Token $i$ can attend to color information through head 1 and motion information through head 3 simultaneously.

The concatenation and output projection $W^O$ then learns to combine these different types of information:

$$o_i = W^O \begin{bmatrix} \text{head}_1(i) \\ \text{head}_2(i) \\ \vdots \\ \text{head}_h(i) \end{bmatrix}$$

### Attention Head Analysis

For a trained video model, we can analyze what each head has learned by computing the average attention distance:

$$\bar{d}_{\text{head}_i} = \frac{1}{NT} \sum_{t=1}^{T} \sum_{n=1}^{N} \sum_{m=1}^{N} \alpha_{nm}^{(i)} |n - m|$$

Heads with small $\bar{d}$ are local (attending to nearby tokens), while heads with large $\bar{d}$ are global. In video models, a bimodal distribution emerges: some heads are very local (texture/edge consistency), others are very global (scene layout).

Similarly, the average temporal attention distance:

$$\bar{d}_{\text{temporal}_i} = \frac{1}{N_s T} \sum_{s=1}^{N_s} \sum_{t=1}^{T} \sum_{t'=1}^{T} \alpha_{tt'}^{(i,s)} |t - t'|$$

Heads with small temporal distance focus on frame-to-frame consistency (short-range motion), while heads with large temporal distance maintain long-range coherence (scene-level consistency).

---

## Flash Attention for Video

### The Memory Problem

Standard attention requires materializing the full $N \times N$ attention matrix in GPU memory (HBM). For video with $N = N_s \cdot T_l = 8{,}000 \times 38 = 304{,}000$ total tokens:

$$\text{Attention matrix size} = 304{,}000^2 \times 2 \text{ bytes (float16)} \approx 172 \text{ GB}$$

This exceeds the memory of any single GPU (A100: 80 GB, H100: 80 GB). Even the factored approach requires $N_s^2 = 64{,}000{,}000$ entries per frame, or about 128 MB per frame per head per layer.

### Flash Attention: The Key Insight

Flash Attention (Dao et al., 2022) observes that the attention computation is **IO-bound**, not compute-bound. The bottleneck is reading and writing the attention matrix to GPU HBM, not computing the matrix multiply.

The key insight: **we do not need to materialize the full attention matrix.** We can compute attention in blocks, keeping intermediate results in fast SRAM (on-chip memory) and only writing the final output to HBM.

### Algorithm

Flash Attention divides the $Q$, $K$, $V$ matrices into blocks of size $B_r$ (rows) and $B_c$ (columns):

```
For each block of queries Q_i (rows i*B_r to (i+1)*B_r):
    Initialize: m_i = -inf, l_i = 0, O_i = 0     (running max, normalizer, output)
    For each block of keys/values K_j, V_j (columns j*B_c to (j+1)*B_c):
        Load Q_i, K_j, V_j into SRAM
        Compute S_ij = Q_i * K_j^T / sqrt(d_k)    (block attention scores)
        Compute m_ij = rowmax(S_ij)                 (local max)
        Compute P_ij = exp(S_ij - m_ij)             (local softmax numerator)
        Compute l_ij = rowsum(P_ij)                 (local softmax denominator)
        Update running stats:
            m_new = max(m_i, m_ij)
            l_new = exp(m_i - m_new) * l_i + exp(m_ij - m_new) * l_ij
            O_i = (exp(m_i - m_new) * l_i * O_i + exp(m_ij - m_new) * P_ij * V_j) / l_new
            m_i = m_new, l_i = l_new
    Write O_i to HBM
```

### Memory Analysis

| Component | Standard Attention | Flash Attention |
|---|---|---|
| Attention matrix | $O(N^2)$ in HBM | $O(B_r \times B_c)$ in SRAM |
| Q, K, V | $O(Nd)$ in HBM | $O(Nd)$ in HBM (same) |
| Output | $O(Nd)$ in HBM | $O(Nd)$ in HBM (same) |
| **Total HBM** | **$O(N^2 + Nd)$** | **$O(Nd)$** |

The quadratic $O(N^2)$ HBM usage is eliminated. For our video example:

| Metric | Standard | Flash Attention | Savings |
|---|---|---|---|
| HBM for attention matrix (per head) | 172 GB | 0 GB | 172 GB |
| SRAM usage (per head) | 0 | ~256 KB | (SRAM is free) |
| Total HBM | ~200 GB | ~28 GB | 7x |

### Flash Attention for Temporal Attention

Flash Attention is particularly impactful for temporal attention in video models because:

1. **Temporal sequences are getting longer.** As video models generate longer outputs (5s to 30s to minutes), $T$ grows, and the temporal attention matrix grows as $T^2$.

2. **Many parallel temporal sequences.** There are $N_s$ independent temporal attention computations (one per spatial position). Flash Attention's block structure allows efficient batching.

3. **Causal masking is free.** Flash Attention can implement causal masking by simply skipping blocks where all entries would be masked. This provides a further 2x speedup for causal temporal attention.

### Flash Attention 2 and 3

Flash Attention 2 (2023) improved the algorithm by:
- Better parallelism across sequence length (not just batch/head dimensions)
- Reduced non-matmul FLOPs
- 2x speedup over Flash Attention 1

Flash Attention 3 (2024) added:
- Exploiting asynchrony and low-precision on Hopper GPUs (H100)
- FP8 quantization for attention scores
- 1.5-2x speedup over Flash Attention 2 on H100s

For video generation on H100s, Flash Attention 3 makes the difference between "fits in memory" and "does not fit in memory" for videos longer than 10 seconds.

---

## Putting It All Together

### The Full Temporal Attention Block

Here is the complete computation graph for a single transformer block in a video diffusion model:

```
Input: X ∈ R^(T_l * N_s, d)

1. Layer Norm
   X_norm = LayerNorm(X)

2. Reshape for spatial attention
   X_spatial = reshape(X_norm, (T_l, N_s, d))

3. Spatial self-attention (per frame)
   For each frame t:
       Q_t, K_t, V_t = X_spatial[t] @ W_Q, W_K, W_V
       attn_t = FlashAttention(Q_t, K_t, V_t)
   X_spatial_out = stack(attn_t for t in 1..T_l)

4. Residual connection + Layer Norm
   X_mid = X + reshape(X_spatial_out, (T_l * N_s, d))
   X_mid_norm = LayerNorm(X_mid)

5. Reshape for temporal attention
   X_temporal = reshape(X_mid_norm, (N_s, T_l, d))

6. Temporal self-attention (per spatial position)
   For each position s:
       Q_s, K_s, V_s = X_temporal[s] @ W_Q', W_K', W_V'
       attn_s = FlashAttention(Q_s, K_s, V_s, causal=False)
   X_temporal_out = stack(attn_s for s in 1..N_s)

7. Residual connection + Layer Norm
   X_out = X_mid + reshape(X_temporal_out, (T_l * N_s, d))
   X_out_norm = LayerNorm(X_out)

8. Feed-forward network
   X_final = X_out + FFN(X_out_norm)

Output: X_final ∈ R^(T_l * N_s, d)
```

### Memory Budget

For a 14B parameter model generating a 5-second 1080p video with 50 diffusion steps:

| Component | Memory | Notes |
|---|---|---|
| Model parameters (fp16) | 28 GB | 14B params x 2 bytes |
| Activations (with FlashAttn) | ~20 GB | Dominated by intermediate FFN |
| KV cache (not needed for diffusion) | 0 GB | Bidirectional, not autoregressive |
| Optimizer states (inference only) | 0 GB | Not training |
| Input/output tensors | ~2 GB | Video frames + latent codes |
| **Total** | **~50 GB** | Fits on single 80 GB GPU |

Without Flash Attention, activations would be ~200+ GB, requiring multi-GPU setups even for short videos.

### The Quality-Cost-Length Tradeoff

Every video generation system faces a fundamental tradeoff between quality, cost, and length:

$$\text{Quality} \propto \frac{\text{Compute per frame}}{\text{Number of frames}^{\alpha}}$$

where $\alpha > 0$ reflects the increasing difficulty of maintaining consistency over longer sequences.

In practice, model providers manage this tradeoff by:
1. Setting maximum duration limits (5s, 9s, 10s)
2. Pricing per second (encouraging shorter videos)
3. Using different model configurations for different durations
4. Applying aggressive attention window strategies for longer videos (sacrificing some global consistency)

This is why all current commercial video models cap duration at 5-10 seconds. The quadratic cost of temporal attention, combined with the increasing difficulty of maintaining consistency, means that generating a 60-second coherent video would require roughly 100x the compute of a 5-second video -- making it prohibitively expensive.

---

## Practical Implications for Builders

### Choosing Your Attention Strategy

If you are building or fine-tuning a video model:

| Video length | Recommended strategy | Reason |
|---|---|---|
| < 5 seconds (< 40 latent frames) | Full temporal attention | Small enough to be affordable, best quality |
| 5-15 seconds (40-120 latent frames) | Full + Flash Attention | Still manageable with memory optimization |
| 15-60 seconds (120-480 latent frames) | Sliding window ($K=32$) + dilated | Balance local consistency with global coherence |
| > 60 seconds | Chunk-based hierarchical | Generate in chunks, connect with overlap |

### API Implications

When building on top of video generation APIs:

1. **Expect pricing to scale super-linearly with duration.** A 10-second video should cost 3-4x a 5-second video, not 2x.

2. **Splitting long videos into short segments** and connecting them (with start/end frame conditioning) is often cheaper and higher quality than generating a single long video.

3. **Different providers may use different attention strategies**, which affects quality characteristics. A model using sliding window attention may have better frame-to-frame consistency but worse long-range coherence than a model using dilated attention.

4. **Resolution matters as much as duration** for cost. Doubling resolution quadruples $N_s$, which quadruples spatial attention cost. Going from 720p to 1080p is a 2.25x increase in spatial attention cost.

---

## Conclusion

Temporal attention is the mathematical machinery that transforms independent frame generation into coherent video generation. The key takeaways:

1. **Self-attention scales quadratically** with sequence length: $O(N^2)$. The $\sqrt{d_k}$ scaling prevents softmax saturation via a variance normalization argument.

2. **Factored spatiotemporal attention** decomposes the prohibitive $O(N_s^2 T^2)$ full 3D attention into manageable $O(N_s^2 T + N_s T^2)$ alternating spatial and temporal layers, achieving a ~100x cost reduction with minimal quality loss.

3. **Causal attention** enables autoregressive generation (frame-by-frame), while **bidirectional attention** enables higher quality parallel generation (all frames at once). Most current models use bidirectional attention within a diffusion framework.

4. **Windowed attention strategies** (sliding window, dilated, chunk-based) reduce temporal attention from $O(T^2)$ to $O(TK)$, $O(T \log T)$, or $O(T^{3/2})$, enabling longer video generation.

5. **Multi-head attention** allows specialization: different heads learn to track color, motion, structure, and global context independently.

6. **Flash Attention** eliminates the $O(N^2)$ memory bottleneck by computing attention in blocks within fast SRAM, reducing HBM usage to $O(N)$. This is essential for practical video generation.

7. **The quadratic scaling** means 10 seconds costs ~4x more than 5 seconds, not 2x. This is why commercial models cap duration and why pricing scales super-linearly with length.

Understanding these mechanics explains nearly every constraint and pricing decision in the current AI video generation market. The architecture is not a black box -- it is a set of well-understood mathematical operations with quantifiable costs and tradeoffs.

---

## References

1. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
2. Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS 2022*.
3. Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. *arXiv:2307.08691*.
4. Shah, J., et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. *arXiv:2407.08691*.
5. Ho, J., et al. (2022). Video Diffusion Models. *NeurIPS 2022*.
6. Blattmann, A., et al. (2023). Align Your Latents: High-Resolution Video Synthesis with Latent Diffusion Models. *CVPR 2023*.
7. Peebles, W., & Xie, S. (2023). Scalable Diffusion Models with Transformers. *ICCV 2023*.
8. Bertasius, G., et al. (2021). Is Space-Time Attention All You Need for Video Understanding? *ICML 2021*.
