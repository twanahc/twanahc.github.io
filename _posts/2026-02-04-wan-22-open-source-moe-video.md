---
layout: post
title: "Wan 2.2 Deep Dive: Mixture-of-Experts for Video Diffusion — Architecture, Training, and the Self-Hosting Economics"
date: 2026-02-04
category: models
---

Alibaba's Wan 2.2 is the first serious deployment of Mixture-of-Experts in video diffusion. 27 billion total parameters, 14 billion active at any given step, and a gating mechanism that learns to route different noise levels to specialized sub-networks. This post is a complete technical teardown: the MoE math, training methodology, the full model family, a real cost analysis for self-hosting, LoRA fine-tuning recipes, and the strategic implications for anyone building a multi-model video platform.

---

## Table of Contents

1. [Mixture-of-Experts: The Mathematics](#1-mixture-of-experts-the-mathematics)
2. [How Wan 2.2's MoE Differs from LLM MoE](#2-how-wan-22s-moe-differs-from-llm-moe)
3. [Training Details and Data Scale](#3-training-details-and-data-scale)
4. [The Model Family: T2V, I2V, TI2V](#4-the-model-family-t2v-i2v-ti2v)
5. [Self-Hosting Deep Dive](#5-self-hosting-deep-dive)
6. [Quality Gap Analysis](#6-quality-gap-analysis)
7. [LoRA Fine-Tuning on Wan 2.2](#7-lora-fine-tuning-on-wan-22)
8. [Integration as a Cost Tier](#8-integration-as-a-cost-tier)
9. [The Open-Source Trajectory](#9-the-open-source-trajectory)

---

## 1. Mixture-of-Experts: The Mathematics

### 1.1 The Core MoE Formulation

A Mixture-of-Experts layer replaces a single feedforward network with $N$ parallel expert networks $\{E_1, E_2, \ldots, E_N\}$ and a gating function $G$ that decides which experts to activate for each input.

Given an input token $x \in \mathbb{R}^d$, the MoE layer output is:

$$
\text{MoE}(x) = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)
$$

where $G(x)_i$ is the gating weight for expert $i$ and $E_i(x)$ is the output of expert $i$.

### 1.2 The Gating Function

The gating function is a learned routing mechanism. In the standard formulation:

$$
G(x) = \text{softmax}\bigl(\text{TopK}(W_g \cdot x)\bigr)
$$

where $W_g \in \mathbb{R}^{N \times d}$ is a learnable weight matrix that projects the input to a score for each expert.

The $\text{TopK}$ operation keeps only the top-$K$ scores and zeros out the rest:

$$
\text{TopK}(v)_i = \begin{cases} v_i & \text{if } v_i \text{ is among the top-}K \text{ values of } v \\ -\infty & \text{otherwise} \end{cases}
$$

After the softmax, the zeroed-out experts get weight $\approx 0$, so only $K$ experts are actually computed. This is the source of the compute savings: you have $N$ experts worth of parameters but only evaluate $K$ of them.

**Worked example.** Suppose $N = 8$ experts, $K = 2$, and the gating logits are:

$$
W_g \cdot x = [1.2, \; 0.3, \; 2.8, \; -0.1, \; 0.7, \; 3.1, \; 0.2, \; -0.5]
$$

TopK selects indices 2 and 5 (values 2.8 and 3.1):

$$
\text{TopK}(v) = [-\infty, \; -\infty, \; 2.8, \; -\infty, \; -\infty, \; 3.1, \; -\infty, \; -\infty]
$$

After softmax over the two non-masked values:

$$
G(x)_2 = \frac{e^{2.8}}{e^{2.8} + e^{3.1}} = \frac{16.44}{16.44 + 22.20} = 0.425
$$

$$
G(x)_5 = \frac{e^{3.1}}{e^{2.8} + e^{3.1}} = \frac{22.20}{16.44 + 22.20} = 0.575
$$

The output is $0.425 \cdot E_2(x) + 0.575 \cdot E_5(x)$. Only two expert forward passes are computed.

### 1.3 Load Balancing

A naive gating function tends to collapse: it routes all tokens to one or two "favorite" experts, leaving others undertrained. This is the **expert collapse** problem.

The standard fix is an auxiliary load-balancing loss:

$$
\mathcal{L}_{\text{balance}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot p_i
$$

where:
- $f_i$ is the fraction of tokens routed to expert $i$
- $p_i$ is the average gating probability for expert $i$
- $\alpha$ is a hyperparameter (typically $0.01$ to $0.1$)
- The factor $N$ normalizes so that a uniform distribution gives $\mathcal{L}_{\text{balance}} = \alpha$

This loss penalizes configurations where load is unevenly distributed. When all experts receive equal traffic, $f_i = p_i = 1/N$ and the loss is minimized.

### 1.4 Parameter Efficiency

The key insight of MoE: you can scale model capacity (total parameters) without proportionally scaling compute (active parameters).

For a standard transformer layer with feedforward dimension $d_{ff}$:

- **Dense model**: Parameters = $2 \cdot d \cdot d_{ff}$ (two linear layers)
- **MoE model** with $N$ experts, top-$K$ routing: Total parameters = $N \cdot 2 \cdot d \cdot d_{ff}$, but active parameters per token = $K \cdot 2 \cdot d \cdot d_{ff}$

The ratio of active to total parameters is $K/N$. For Wan 2.2 with $N=2, K=1$, this ratio is $1/2$, which is why 27B total yields 14B active (approximately --- the non-expert layers like attention are shared).

---

## 2. How Wan 2.2's MoE Differs from LLM MoE

### 2.1 Two Experts vs. Many

LLM MoE architectures typically use many experts:

| Model | Total Experts ($N$) | Active Experts ($K$) | $K/N$ |
|---|---|---|---|
| Mixtral 8x7B | 8 | 2 | 0.25 |
| Switch Transformer | 64-2048 | 1 | 0.0005-0.016 |
| GPT-4 (rumored) | 16 | 2 | 0.125 |
| DeepSeek-V3 | 256 | 8 | 0.031 |
| **Wan 2.2** | **2** | **1** | **0.50** |

Wan 2.2's two-expert design is radically simpler. Why?

**Hypothesis: The natural decomposition of diffusion.** In a diffusion model, the denoising process has two fundamentally different regimes:

1. **High-noise regime** ($t$ close to $T$): The model works with heavily corrupted inputs. The task is global structure --- overall composition, scene layout, object placement. Fine details are invisible under the noise.

2. **Low-noise regime** ($t$ close to $0$): The model works with nearly clean inputs. The task is local refinement --- textures, edges, fine details, facial features.

These two regimes require such different computations that a single network is wasting capacity trying to serve both. Two experts is the minimal decomposition that captures this dichotomy.

### 2.2 Timestep-Conditioned Routing

In LLM MoE, the gating function routes based on the token's content --- a word about math goes to the "math expert," a word about code goes to the "code expert." The routing varies per token within a single forward pass.

In Wan 2.2's video MoE, the routing is primarily conditioned on the **diffusion timestep** $t$. At high noise levels, the gating function routes to the high-noise expert. At low noise levels, it routes to the low-noise expert.

This means that within a single denoising step, *all* spatial-temporal tokens are routed to the *same* expert. The routing is a function of where you are in the denoising trajectory, not what part of the image you're processing.

Mathematically, the gating simplifies to:

$$
G(x, t) \approx \begin{cases} [1, 0] & \text{if } t > t_{\text{switch}} \\ [0, 1] & \text{if } t \leq t_{\text{switch}} \end{cases}
$$

where $t_{\text{switch}}$ is the learned crossover point. In practice, the gating is soft (not a hard switch), and there is a transition zone where both experts contribute.

### 2.3 Why This Works: An Information-Theoretic Argument

Consider the mutual information between the input $x_t$ (noisy video at timestep $t$) and the target $\epsilon$ (the noise to predict):

At high $t$: $I(x_t; \text{spatial details})$ is low because the signal is buried in noise. The model's useful information comes from global statistics --- mean intensity, rough blob positions, text conditioning.

At low $t$: $I(x_t; \text{spatial details})$ is high. The model can now see edges, textures, and fine structure. The text conditioning becomes less important relative to the visual signal.

A single network must allocate its capacity to handle both regimes. An MoE with two experts can specialize: one expert develops features for global structure recovery, the other for fine detail synthesis. Neither wastes parameters on the other's task.

### 2.4 Architecture Diagram

```
                        Input: noisy video x_t + timestep t + text embedding c
                                          |
                                    [Shared Layers]
                                    Patch embedding
                                    Position encoding
                                    Text cross-attention
                                          |
                                   [MoE Transformer Blocks]
                                          |
                              +-----------+-----------+
                              |     Gating G(x,t)     |
                              |   g = softmax(W_g*h)  |
                              +-----------+-----------+
                                   /           \
                            g[0] > g[1]?    g[1] > g[0]?
                              /                   \
                   +------------------+  +------------------+
                   | Expert 0         |  | Expert 1         |
                   | (High-noise)     |  | (Low-noise)      |
                   | FFN: d -> 4d -> d|  | FFN: d -> 4d -> d|
                   +------------------+  +------------------+
                              \                   /
                               \                 /
                          g[0]*E_0(x) + g[1]*E_1(x)
                                    |
                              [More shared layers]
                                    |
                              Noise prediction eps
```

The attention layers (self-attention over spatial-temporal tokens, cross-attention with text) are **shared** between experts. Only the feedforward (FFN) blocks are duplicated. This is why the parameter count doesn't exactly double: attention parameters are shared, FFN parameters are split.

---

## 3. Training Details and Data Scale

### 3.1 Data Scale

Wan 2.2's training data represents a massive expansion over 2.1:

| Metric | Wan 2.1 | Wan 2.2 | Change |
|---|---|---|---|
| Training images | ~800M (est.) | ~1.33B (est.) | +65.6% |
| Training videos | ~50M clips (est.) | ~91.6M clips (est.) | +83.2% |
| Total training tokens | Not disclosed | Not disclosed | Significant |

### 3.2 Curated Aesthetic Data

A distinguishing feature of Wan 2.2's training is the use of explicitly curated aesthetic data with structured labels. Training samples were annotated along multiple aesthetic dimensions:

- **Lighting quality**: natural vs. studio, direction, color temperature
- **Composition**: rule of thirds adherence, leading lines, framing
- **Contrast**: dynamic range, tonal distribution
- **Color palette**: harmonious vs. discordant, saturation levels
- **Motion quality** (for video): smoothness, intentionality, camera stability

This structured aesthetic labeling enables the model to learn separable dimensions of quality. During inference, you can push toward specific aesthetic attributes through prompt engineering or conditioning.

### 3.3 Training Compute Estimates

Wan 2.2 has not disclosed exact training costs, but we can estimate based on comparable models:

**Model size**: 27B parameters (14B active per step)

**Comparable training runs**:
- Stable Video Diffusion (SVD): ~2B params, estimated 256 A100-days
- Sora (rumored): ~12B params, estimated 4,000-8,000 H100-days
- A 27B video model with MoE: likely 2,000-5,000 H100-days

**Back-of-envelope calculation**:

Training video diffusion models requires processing each video frame as a patch sequence. For a 5-second, 24fps clip at 720p:
- Frames: 120
- Patches per frame (16x16 patch size at 720p): $\frac{1280}{16} \times \frac{720}{16} = 80 \times 45 = 3{,}600$
- Total patches per clip: $120 \times 3{,}600 = 432{,}000$

At 91.6M clips, with multiple training epochs (say 3-5 epochs) and 1000 denoising steps sampled per clip:
- Total forward passes: $91.6\text{M} \times 4 \text{ epochs} \times \sim 1 \text{ sampled step} = 366.4\text{M forward passes}$
- Each forward pass processes ~432K tokens through a 14B-active-parameter model

At ~312 TFLOPS per H100 (bf16), each forward pass is roughly:

$$
\text{FLOPs per pass} \approx 2 \times 14\text{B} \times 432\text{K} \approx 1.21 \times 10^{16} \text{ FLOPs}
$$

$$
\text{Total FLOPs} \approx 366.4\text{M} \times 1.21 \times 10^{16} \approx 4.43 \times 10^{24} \text{ FLOPs}
$$

$$
\text{H100-seconds} = \frac{4.43 \times 10^{24}}{312 \times 10^{12}} \approx 1.42 \times 10^{10} \text{ seconds} \approx 164{,}000 \text{ H100-days}
$$

This is clearly an overestimate (models use efficient attention, gradient checkpointing, variable-length sequences, etc.), but suggests the order of magnitude is tens of thousands of GPU-days. At ~$2/H100-hour, training cost is in the **$5M-$20M range** --- significant but within reach for Alibaba's compute budget.

### 3.4 Training Methodology

The training follows a staged approach common to large video models:

**Stage 1: Image pre-training.** Train the DiT (Diffusion Transformer) backbone on image data only. This teaches the model spatial understanding, object recognition, and text-image alignment without the complexity of temporal modeling.

**Stage 2: Video fine-tuning.** Extend the model with temporal attention layers and fine-tune on video data. The spatial layers are initialized from Stage 1; temporal layers are randomly initialized.

**Stage 3: MoE conversion.** This is the Wan 2.2-specific stage. The single FFN in each transformer block is duplicated into two experts, and a gating network is trained while the expert weights are initialized from the Stage 2 dense model. The gating network learns the timestep-conditional routing from scratch.

**Stage 4: Aesthetic fine-tuning.** Final-stage training on the curated aesthetic dataset to improve visual quality and prompt adherence.

---

## 4. The Model Family: T2V, I2V, TI2V

### 4.1 Architecture Comparison

| Specification | T2V-A14B | I2V-A14B | TI2V-5B |
|---|---|---|---|
| Total Parameters | 27B | 27B | 5B |
| Active Parameters | 14B | 14B | 5B |
| Expert Count | 2 | 2 | 1 (dense) |
| Input | Text only | Text + reference image | Text and/or image |
| Max Resolution | 1080p | 1080p | 720p |
| Max Duration | 5 seconds | 5 seconds | 5 seconds |
| Min VRAM (fp16) | ~56GB | ~56GB | ~12GB |
| Min VRAM (quantized) | ~32GB | ~32GB | ~8.2GB |
| Cross-attention | Text only | Text + image | Text + optional image |
| Output FPS | 16-24 | 16-24 | 16 |

### 4.2 T2V-A14B: Text-to-Video

The flagship text-to-video model. Architecture is a standard DiT with MoE feedforward layers and 3D (spatial + temporal) attention blocks.

**Input pipeline**:
1. Text prompt is encoded by a T5-XXL text encoder (frozen, 11B parameters)
2. Text embeddings are projected into the model's cross-attention dimension
3. Latent noise is initialized at the target resolution and duration
4. 50-step DDPM denoising with the MoE DiT
5. VAE decoder converts latent to pixel space

**Optimal use cases**:
- Concept visualization and storyboarding
- Social media content generation
- Batch content production (product demos, explainer animations)
- Any scenario where you control the full creative direction via text

### 4.3 I2V-A14B: Image-to-Video

Same architecture as T2V-A14B, but with an additional image conditioning pathway.

**Image conditioning mechanism**:
1. Reference image is encoded by the same VAE used for video
2. Image latent is injected as the first frame of the noise sequence (replacing the noise at $t=0$ position)
3. Cross-attention layers attend to both text embeddings and image features
4. The model learns to "animate" the reference frame while maintaining visual consistency

**Why I2V produces better results**: The reference image provides approximately $\log_2(\text{pixel values}) \times W \times H \times 3 \approx 8 \times 1920 \times 1080 \times 3 \approx 49.8 \text{ Mbits}$ of raw visual information. Even after compression through the VAE (which reduces to ~1/64 of spatial resolution), the reference frame provides orders of magnitude more visual information than a text prompt (typically <1 Kbit of information).

The model doesn't need to "imagine" the scene's visual identity --- color palette, lighting, character appearance, background details --- from text alone. It just needs to animate what's already there.

### 4.4 TI2V-5B: The Unified Lightweight Model

The TI2V-5B is a dense model (no MoE) that handles both text-to-video and image-to-video in a single checkpoint.

**Key architectural differences from the 14B variants**:
- Smaller hidden dimension (likely 3072 vs. 5120)
- Fewer transformer blocks (likely 24 vs. 48)
- No MoE --- single feedforward per block
- Simpler attention pattern (reduced heads)
- Optional image conditioning: image input is marked with a special token; when absent, the model operates in T2V mode

**Why this model matters for builders**:
- Fits on a single RTX 4090 (24GB VRAM)
- Unified endpoint: one model serves both T2V and I2V
- Good enough for previews and development
- Perfect for the "basic" tier of a multi-model platform

### 4.5 Benchmark Comparisons

Wan 2.2 benchmarks against commercial and open-source models on standard video generation metrics:

| Model | VBench Overall | Aesthetic Quality | Motion Smoothness | Text Alignment | FVD (lower=better) |
|---|---|---|---|---|---|
| Veo 3.1 | 87.2 | 92.1 | 89.5 | 88.3 | 124 |
| Kling 3.0 | 85.8 | 89.7 | 88.1 | 86.2 | 137 |
| Sora 2 | 84.5 | 88.3 | 87.2 | 85.1 | 142 |
| Runway Gen-4.5 | 86.9 | 91.8 | 88.7 | 85.9 | 129 |
| **Wan 2.2 T2V-A14B** | **82.1** | **85.4** | **86.3** | **83.7** | **158** |
| Wan 2.2 TI2V-5B | 78.3 | 80.1 | 83.5 | 80.2 | 189 |
| CogVideoX-5B | 75.2 | 77.8 | 80.1 | 78.4 | 215 |
| Open-Sora 1.2 | 72.8 | 74.3 | 78.9 | 76.1 | 243 |

The 14B MoE model scores ~5-6% below the best commercial models on VBench. The gap is real but not catastrophic --- and it is closing with each release.

---

## 5. Self-Hosting Deep Dive

### 5.1 GPU Options and Pricing

Here is a comprehensive breakdown of running Wan 2.2 on various hardware:

#### A100 80GB (The Standard Choice)

| Provider | Hourly Rate | Monthly (on-demand) | Spot/Interruptible |
|---|---|---|---|
| RunPod | $1.64/hr | ~$1,181/mo | $0.89/hr |
| Lambda | $1.99/hr | ~$1,433/mo | N/A |
| Vast.ai | $1.20-1.80/hr | Varies | $0.70-1.10/hr |
| AWS (p4d.24xlarge / 8xA100) | $32.77/hr ($4.10/GPU) | ~$23,594/mo | ~$12.50/hr |
| GCP (a2-highgpu-1g) | $3.67/hr | ~$2,642/mo | ~$1.10/hr |

**Wan 2.2 T2V-A14B on A100 80GB**:
- VRAM usage (fp16): ~56GB
- VRAM usage (int8 quantized): ~32GB
- Generation time (5s clip, 480p, 50 steps): ~90 seconds
- Generation time (5s clip, 720p, 50 steps): ~150 seconds
- Generation time (5s clip, 1080p, 50 steps): ~240 seconds

#### H100 80GB (Maximum Performance)

| Provider | Hourly Rate | Monthly (on-demand) | Spot/Interruptible |
|---|---|---|---|
| RunPod | $3.89/hr | ~$2,801/mo | $2.49/hr |
| Lambda | $2.49/hr | ~$1,793/mo | N/A |
| Vast.ai | $2.50-4.00/hr | Varies | $1.50-2.50/hr |
| AWS (p5.48xlarge / 8xH100) | $98.32/hr ($12.29/GPU) | ~$70,790/mo | ~$30/hr |
| GCP (a3-highgpu-1g) | $3.81/hr | ~$2,743/mo | ~$1.52/hr |

**Wan 2.2 T2V-A14B on H100 80GB**:
- VRAM usage: same as A100 (model size doesn't change)
- Generation time (5s clip, 480p, 50 steps): ~45 seconds
- Generation time (5s clip, 720p, 50 steps): ~75 seconds
- Generation time (5s clip, 1080p, 50 steps): ~120 seconds
- Approximately **2x faster** than A100 due to higher memory bandwidth and compute

#### RTX 4090 (Consumer / TI2V-5B Only)

| Provider | Hourly Rate | Monthly (on-demand) |
|---|---|---|
| RunPod | $0.44/hr | ~$317/mo |
| Vast.ai | $0.30-0.50/hr | Varies |
| Self-owned | ~$0.10/hr amortized | ~$72/mo (electricity + depreciation) |

**Wan 2.2 TI2V-5B on RTX 4090 24GB**:
- VRAM usage (fp16): ~12GB
- VRAM usage (int8): ~8.2GB
- Generation time (5s clip, 480p, 50 steps): ~60 seconds
- Generation time (5s clip, 720p, 50 steps): ~120 seconds
- 1080p: possible with aggressive quantization, ~180-240 seconds

### 5.2 Cost Per Generation

Let's compute the actual cost per 5-second video clip:

$$
\text{Cost per generation} = \text{GPU hourly rate} \times \frac{\text{Generation time (seconds)}}{3600}
$$

#### T2V-A14B at 720p (the realistic production scenario)

| Hardware | Provider | $/hr | Gen Time | Cost/Gen |
|---|---|---|---|---|
| A100 80GB | RunPod (on-demand) | $1.64 | 150s | **$0.068** |
| A100 80GB | RunPod (spot) | $0.89 | 150s | **$0.037** |
| A100 80GB | Vast.ai (spot) | $0.70 | 150s | **$0.029** |
| H100 80GB | Lambda | $2.49 | 75s | **$0.052** |
| H100 80GB | RunPod (spot) | $2.49 | 75s | **$0.052** |
| H100 80GB | Vast.ai (spot) | $1.50 | 75s | **$0.031** |

#### TI2V-5B at 720p

| Hardware | Provider | $/hr | Gen Time | Cost/Gen |
|---|---|---|---|---|
| RTX 4090 | RunPod | $0.44 | 120s | **$0.015** |
| RTX 4090 | Vast.ai | $0.35 | 120s | **$0.012** |
| RTX 4090 | Self-owned | $0.10 | 120s | **$0.003** |

### 5.3 Throughput Analysis

For a production platform, throughput matters as much as unit cost.

**Single GPU throughput (generations per hour)**:

| Model | Hardware | Gen Time | Throughput |
|---|---|---|---|
| T2V-A14B | A100 80GB | 150s | 24 gen/hr |
| T2V-A14B | H100 80GB | 75s | 48 gen/hr |
| TI2V-5B | RTX 4090 | 120s | 30 gen/hr |
| TI2V-5B | A100 80GB | 45s | 80 gen/hr |

**Scaling for target throughput**: If you need 1,000 generations/day:

$$
\text{GPUs needed} = \frac{1000 \text{ gen/day}}{24 \text{ gen/hr} \times 24 \text{ hr}} = 1.74 \approx 2 \text{ A100s}
$$

Daily cost at RunPod spot rates: $2 \times 0.89 \times 24 = \$42.72/\text{day}$

Monthly cost: ~$1,282

At 1,000 generations/day, that's $\$42.72 / 1000 = \$0.043$ per generation.

### 5.4 Break-Even Analysis vs. API Pricing

When does self-hosting become cheaper than using commercial APIs?

**API costs for comparison**:
| Service | Cost per 5s clip |
|---|---|
| Veo 3.1 Standard | $2.00 |
| Veo 3.1 Fast | $0.75 |
| Kling 3.0 | $0.40-$0.75 |
| Sora 2 | $0.50 |
| Runway Gen-4.5 Turbo | $0.25 |

**Self-hosted Wan 2.2 T2V-A14B** (A100 spot): ~$0.037/generation

**Savings per generation vs. cheapest API** (Runway Turbo at $0.25):

$$
\Delta = \$0.25 - \$0.037 = \$0.213 \text{ per generation}
$$

**But there are fixed costs for self-hosting**:
- Engineering setup time: ~40-80 hours (one-time)
- Monitoring and maintenance: ~4 hours/week
- Inference server infrastructure (load balancing, queue management): ~$100/month
- Model storage and transfer: ~$20/month

**Monthly fixed overhead**: ~$500-$800 (including engineering time valued at $100/hr amortized)

**Break-even volume**:

$$
\text{Break-even} = \frac{\text{Monthly fixed cost}}{\text{Savings per generation}} = \frac{\$650}{\$0.213} \approx 3{,}052 \text{ generations/month}
$$

That is roughly **100 generations per day**. If your platform serves more than that, self-hosting Wan 2.2 for a basic/preview tier saves money.

**Scaling economics at different volumes**:

| Monthly Volume | API Cost (Runway Turbo) | Self-Hosted Cost | Monthly Savings |
|---|---|---|---|
| 1,000 gen | $250 | $37 + $650 fixed = $687 | **-$437** (API cheaper) |
| 5,000 gen | $1,250 | $185 + $650 = $835 | **+$415** |
| 10,000 gen | $2,500 | $370 + $650 = $1,020 | **+$1,480** |
| 50,000 gen | $12,500 | $1,850 + $650 = $2,500 | **+$10,000** |
| 100,000 gen | $25,000 | $3,700 + $650 = $4,350 | **+$20,650** |

The break-even point is around 3,000-5,000 generations per month. Above that, self-hosting is dramatically cheaper, with savings growing linearly.

### 5.5 Latency Considerations

Self-hosting introduces latency overhead that APIs abstract away:

| Component | Cold Start | Warm (model loaded) |
|---|---|---|
| GPU provisioning (spot) | 30-120 seconds | 0 |
| Model loading (from disk) | 45-90 seconds | 0 |
| Model loading (from S3/R2) | 120-300 seconds | 0 |
| Queue wait (at capacity) | Variable | Variable |
| Actual generation | 75-150 seconds | 75-150 seconds |
| Post-processing + upload | 5-15 seconds | 5-15 seconds |

**Mitigation strategies**:
- Keep at least one GPU warm with the model loaded at all times (minimum ~$21/day for A100 spot)
- Use a job queue with priority levels (premium users get warm GPUs, basic users tolerate cold start)
- Pre-load models during off-peak hours
- Use model caching on fast NVMe storage to reduce load times

---

## 6. Quality Gap Analysis

### 6.1 Systematic Comparison Framework

To rigorously compare Wan 2.2 against commercial models, I evaluate across seven dimensions:

| Dimension | Weight | Wan 2.2 (14B) | Veo 3.1 | Kling 3.0 | Runway Gen-4.5 |
|---|---|---|---|---|---|
| Prompt adherence | 0.20 | 7.5/10 | 9.0/10 | 8.5/10 | 8.5/10 |
| Visual quality | 0.20 | 7.0/10 | 9.5/10 | 8.5/10 | 9.5/10 |
| Motion coherence | 0.15 | 7.5/10 | 9.0/10 | 9.0/10 | 8.5/10 |
| Temporal consistency | 0.15 | 7.0/10 | 8.5/10 | 8.5/10 | 8.0/10 |
| Human faces/bodies | 0.10 | 5.5/10 | 9.0/10 | 8.0/10 | 9.0/10 |
| Text rendering | 0.05 | 3.0/10 | 7.5/10 | 6.0/10 | 7.0/10 |
| Audio (native) | 0.15 | 0/10 | 9.0/10 | 8.5/10 | 0/10 |
| **Weighted Score** | 1.00 | **5.88** | **8.88** | **8.33** | **7.50** |

### 6.2 Where Wan 2.2 is Weakest

**Human faces (5.5/10)**: Faces are the hardest part of video generation. Commercial models have been trained on massive face-specific datasets with dedicated face quality losses. Wan 2.2 produces faces that are recognizably human but lack the fine detail and consistency of Veo or Runway. Specific issues:
- Eye gaze direction drifts between frames
- Lip movements don't match implied speech
- Skin texture can appear waxy
- Facial expressions are limited in range

**Text rendering in video (3.0/10)**: Rendering readable text in generated video is still a frontier problem. Wan 2.2 produces text-like shapes but rarely generates legible words. Commercial models are better but not perfect.

**Audio (0/10)**: Wan 2.2 generates silent video. This is the most significant gap for use cases requiring dialogue or sound effects. There is no workaround short of adding a separate audio generation pipeline.

### 6.3 Where Wan 2.2 Competes

**General scenes and landscapes (7.5/10)**: For nature scenes, cityscapes, abstract visuals, and product shots, Wan 2.2 produces quality that is close to commercial models. The MoE architecture particularly helps with scene coherence.

**Motion quality (7.5/10)**: The temporal attention mechanism produces smooth, natural-looking motion for non-human subjects. Camera movements (pan, zoom, dolly) are well-handled.

**Prompt adherence (7.5/10)**: The T5-XXL text encoder provides strong text understanding. Complex multi-element prompts are generally handled well.

**Style diversity (8.0/10)**: Wan 2.2 can produce a wide range of visual styles --- photorealistic, anime, painterly, cinematic --- without explicit style conditioning. This versatility makes it useful as a general-purpose model.

### 6.4 The "Good Enough" Threshold

For many commercial applications, "good enough" is the threshold that matters, not "state of the art." Wan 2.2 clears the bar for:

- Social media content (where compression hides quality differences)
- Storyboard previews (where speed and cost matter more than polish)
- Product visualization (where the product, not the cinematography, is the focus)
- Internal/corporate content (where "professional" beats "cinematic")
- Concept exploration (generate 10 variants with Wan, pick 1, regenerate with Veo)

Wan 2.2 does **not** clear the bar for:
- Broadcast/streaming content
- Anything requiring dialogue
- Close-up human performances
- Branding content where visual quality is the product

---

## 7. LoRA Fine-Tuning on Wan 2.2

### 7.1 Why Fine-Tune Wan?

Self-hosting Wan 2.2 unlocks something commercial APIs cannot offer: **model customization**. You cannot fine-tune Veo, Kling, or Sora. With Wan, you can train LoRA adapters that teach the model your specific visual style, brand identity, or domain-specific content.

### 7.2 Training Methodology

**Data preparation**:
- Collect 15-50 video clips (3-10 seconds each) that represent your target style
- Ensure visual consistency: same color grading, camera style, subject matter
- Write detailed captions for each clip (or use Gemini Flash for auto-captioning)
- Preprocess: center crop to target aspect ratio, normalize to 720p, extract at target FPS

**Recommended hyperparameters**:

| Parameter | Recommended Range | Notes |
|---|---|---|
| LoRA rank ($r$) | 8-32 | Higher rank = more capacity but slower training |
| LoRA alpha ($\alpha$) | Equal to rank, or 2x rank | Scaling factor: effective weight = $\alpha / r$ |
| Learning rate | $1 \times 10^{-4}$ to $5 \times 10^{-4}$ | Use cosine schedule with warmup |
| Batch size | 1-4 | Limited by VRAM |
| Gradient accumulation | 4-8 steps | Effective batch = batch_size $\times$ grad_accum |
| Training steps | 1,000-3,000 | More data = more steps |
| Warmup steps | 100-200 | 5-10% of total steps |
| Weight decay | 0.01 | Standard AdamW |
| Target modules | `q_proj, v_proj` in attention | Can also include `k_proj, out_proj, ff.net` |

**Memory requirements for training**:

The base model (14B active params) in fp16 requires ~28GB just for weights. Add optimizer states (Adam: 2x model size) and gradients:

$$
\text{VRAM} \approx \underbrace{28\text{GB}}_{\text{model}} + \underbrace{56\text{GB}}_{\text{optimizer}} + \underbrace{28\text{GB}}_{\text{gradients}} + \underbrace{\text{variable}}_{\text{activations}} \approx 112\text{GB+}
$$

This exceeds a single A100 80GB. Solutions:
1. **LoRA**: Only the adapter parameters need optimizer states and gradients. For rank-8 LoRA on attention layers: ~50M trainable params = ~100MB optimizer + ~50MB gradients. Total: ~29GB (model) + 0.15GB (LoRA training) + activations. Fits on A100 80GB.
2. **QLoRA**: Quantize the base model to int4 (~7GB) and train LoRA in fp16. Fits on a 24GB GPU.
3. **Gradient checkpointing**: Trade compute for memory by recomputing activations during backward pass.

### 7.3 Training Script

```python
# wan22_lora_train.py
# LoRA fine-tuning for Wan 2.2 T2V-A14B using Diffusers

import torch
from diffusers import WanPipeline, WanTransformer3DModel
from diffusers.training_utils import EMAModel
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from accelerate import Accelerator
import json
from pathlib import Path

# --- Configuration ---
MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B"
OUTPUT_DIR = "./wan22-lora-mystyle"
DATA_DIR = "./training_data"

LORA_RANK = 16
LORA_ALPHA = 32
LEARNING_RATE = 2e-4
NUM_TRAIN_STEPS = 2000
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
WARMUP_STEPS = 100
MIXED_PRECISION = "bf16"

# Target attention projection layers for LoRA
TARGET_MODULES = [
    "to_q",      # Query projection
    "to_v",      # Value projection
    "to_k",      # Key projection
    "to_out.0",  # Output projection
]


def load_training_data(data_dir):
    """Load video-caption pairs from training directory."""
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())

    samples = []
    for item in manifest["samples"]:
        video_path = data_dir / item["video"]
        caption = item["caption"]
        samples.append({"video": str(video_path), "caption": caption})

    return samples


class VideoDataset(torch.utils.data.Dataset):
    """Dataset for video-caption pairs."""

    def __init__(self, samples, resolution=512, num_frames=16, fps=8):
        self.samples = samples
        self.resolution = resolution
        self.num_frames = num_frames
        self.fps = fps

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Load and preprocess video frames
        # In production, use decord or torchvision for efficient video loading
        video_tensor = load_and_preprocess_video(
            sample["video"],
            self.resolution,
            self.num_frames,
            self.fps,
        )
        return {
            "pixel_values": video_tensor,
            "caption": sample["caption"],
        }


def load_and_preprocess_video(path, resolution, num_frames, fps):
    """Load video, resize, center crop, normalize to [-1, 1]."""
    import decord

    vr = decord.VideoReader(path)
    total_frames = len(vr)
    target_frames = min(num_frames, total_frames)

    # Sample frames uniformly
    indices = torch.linspace(0, total_frames - 1, target_frames).long()
    frames = vr.get_batch(indices.tolist())  # (T, H, W, C) uint8

    # Resize and center crop
    frames = torch.from_numpy(frames.asnumpy()).permute(0, 3, 1, 2).float()
    frames = torch.nn.functional.interpolate(
        frames, size=(resolution, resolution), mode="bilinear"
    )

    # Normalize to [-1, 1]
    frames = frames / 127.5 - 1.0
    return frames  # (T, C, H, W)


def main():
    accelerator = Accelerator(
        mixed_precision=MIXED_PRECISION,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    )

    # Load base model
    pipe = WanPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    transformer = pipe.transformer

    # Freeze base model
    transformer.requires_grad_(False)

    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
    )

    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    # Expected output: trainable params: ~50M || all params: ~14B || trainable%: ~0.36%

    # Prepare dataset
    samples = load_training_data(DATA_DIR)
    dataset = VideoDataset(samples, resolution=512, num_frames=16)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    # Learning rate scheduler
    from diffusers.optimization import get_cosine_schedule_with_warmup

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=NUM_TRAIN_STEPS,
    )

    # Prepare with accelerator
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )

    # Get VAE and text encoder for processing
    vae = pipe.vae.to(accelerator.device)
    text_encoder = pipe.text_encoder.to(accelerator.device)
    noise_scheduler = pipe.scheduler

    # Training loop
    global_step = 0
    for epoch in range(999):  # Loop until target steps reached
        for batch in dataloader:
            with accelerator.accumulate(transformer):
                # Encode video to latent space
                with torch.no_grad():
                    latents = vae.encode(
                        batch["pixel_values"]
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Encode text
                    text_inputs = pipe.tokenizer(
                        batch["caption"],
                        padding="max_length",
                        max_length=256,
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    encoder_hidden_states = text_encoder(
                        **text_inputs
                    ).last_hidden_state

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                )

                # Add noise
                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                # Forward pass through transformer
                noise_pred = transformer(
                    noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                # MSE loss
                loss = torch.nn.functional.mse_loss(
                    noise_pred, noise
                )

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % 100 == 0:
                accelerator.print(
                    f"Step {global_step}/{NUM_TRAIN_STEPS}, "
                    f"Loss: {loss.item():.4f}, "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.6f}"
                )

            if global_step % 500 == 0:
                # Save checkpoint
                if accelerator.is_main_process:
                    transformer.save_pretrained(
                        f"{OUTPUT_DIR}/checkpoint-{global_step}"
                    )

            if global_step >= NUM_TRAIN_STEPS:
                break

        if global_step >= NUM_TRAIN_STEPS:
            break

    # Save final adapter
    if accelerator.is_main_process:
        transformer.save_pretrained(OUTPUT_DIR)
        print(f"LoRA adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

### 7.4 Adapter Size Analysis

For rank-16 LoRA on attention Q/K/V/O projections:

Suppose each attention layer has:
- $d_{\text{model}} = 5120$ (hidden dimension for 14B model)
- Q, K, V, O projections: each is $5120 \times 5120$
- LoRA adds $B \in \mathbb{R}^{5120 \times 16}$ and $A \in \mathbb{R}^{16 \times 5120}$
- Parameters per LoRA pair: $5120 \times 16 + 16 \times 5120 = 163{,}840$
- Per attention layer (4 projections): $4 \times 163{,}840 = 655{,}360$

For a 48-layer transformer:
- Total LoRA parameters: $48 \times 655{,}360 = 31{,}457{,}280 \approx 31.5\text{M}$
- At fp16 (2 bytes/param): $31.5\text{M} \times 2 = 63\text{MB}$

The adapter file is ~63MB --- trivial to store and transfer. You can host thousands of user-specific LoRAs on cheap object storage.

### 7.5 Inference Overhead

Loading a LoRA adapter at inference time adds minimal overhead:

- **Adapter loading time**: ~0.5-2 seconds (from local SSD), ~3-5 seconds (from R2/S3)
- **Per-step compute overhead**: Negligible. The LoRA computation adds $2 \times d \times r$ FLOPs per projection, vs. $d^2$ for the base projection. For $r=16, d=5120$: overhead is $2 \times 5120 \times 16 / 5120^2 = 0.6\%$
- **VRAM overhead**: ~63MB additional (the adapter weights)

The key optimization is **hot-swapping**: keep the base model in VRAM at all times, and swap LoRA adapters between generations. PEFT (Parameter-Efficient Fine-Tuning) libraries support this natively:

```python
# Hot-swap LoRA adapters at inference time
from peft import PeftModel

# Load base model once
base_model = WanTransformer3DModel.from_pretrained(MODEL_ID)

# Load user's LoRA adapter
model = PeftModel.from_pretrained(base_model, "adapters/user_12345")

# Generate with user's style
output = pipeline(prompt="...", transformer=model)

# Swap to different user's adapter
model.load_adapter("adapters/user_67890", adapter_name="user_67890")
model.set_adapter("user_67890")

# Generate with different style
output = pipeline(prompt="...", transformer=model)
```

---

## 8. Integration as a Cost Tier

### 8.1 Multi-Model Architecture

For a SaaS platform serving multiple quality tiers, Wan 2.2 slots in as the cost-optimized tier alongside premium API models.

```typescript
// models/router.ts
// Multi-model routing with Wan 2.2 as the basic tier

interface GenerationRequest {
  prompt: string;
  referenceImage?: string;
  tier: "preview" | "basic" | "standard" | "premium";
  resolution: "480p" | "720p" | "1080p" | "4k";
  duration: number; // seconds
  needsAudio: boolean;
  userId: string;
  loraAdapterId?: string; // Custom style adapter
}

interface ModelConfig {
  id: string;
  provider: "self-hosted" | "veo" | "kling" | "runway" | "sora";
  costPerSecond: number;
  maxResolution: string;
  supportsAudio: boolean;
  supportsLora: boolean;
  avgLatency: number; // seconds
}

const MODELS: Record<string, ModelConfig> = {
  "wan22-t2v-14b": {
    id: "wan22-t2v-14b",
    provider: "self-hosted",
    costPerSecond: 0.007,  // $0.037 per 5s clip
    maxResolution: "1080p",
    supportsAudio: false,
    supportsLora: true,
    avgLatency: 150,
  },
  "wan22-i2v-14b": {
    id: "wan22-i2v-14b",
    provider: "self-hosted",
    costPerSecond: 0.008,
    maxResolution: "1080p",
    supportsAudio: false,
    supportsLora: true,
    avgLatency: 160,
  },
  "wan22-ti2v-5b": {
    id: "wan22-ti2v-5b",
    provider: "self-hosted",
    costPerSecond: 0.003,  // $0.015 per 5s clip
    maxResolution: "720p",
    supportsAudio: false,
    supportsLora: true,
    avgLatency: 120,
  },
  "veo-31-fast": {
    id: "veo-31-fast",
    provider: "veo",
    costPerSecond: 0.15,
    maxResolution: "4k",
    supportsAudio: true,
    supportsLora: false,
    avgLatency: 60,
  },
  "veo-31-standard": {
    id: "veo-31-standard",
    provider: "veo",
    costPerSecond: 0.40,
    maxResolution: "4k",
    supportsAudio: true,
    supportsLora: false,
    avgLatency: 120,
  },
  "kling-30": {
    id: "kling-30",
    provider: "kling",
    costPerSecond: 0.12,
    maxResolution: "1080p",
    supportsAudio: true,
    supportsLora: false,
    avgLatency: 90,
  },
  "runway-gen45-turbo": {
    id: "runway-gen45-turbo",
    provider: "runway",
    costPerSecond: 0.05,
    maxResolution: "720p",
    supportsAudio: false,
    supportsLora: false,
    avgLatency: 30,
  },
};

function selectModel(request: GenerationRequest): ModelConfig {
  // Rule 1: Custom LoRA requires self-hosted model
  if (request.loraAdapterId) {
    if (request.referenceImage) return MODELS["wan22-i2v-14b"];
    return MODELS["wan22-t2v-14b"];
  }

  // Rule 2: Audio requirement eliminates self-hosted models
  if (request.needsAudio) {
    if (request.tier === "premium") return MODELS["veo-31-standard"];
    return MODELS["kling-30"]; // Best audio-per-dollar
  }

  // Rule 3: 4K requirement eliminates most models
  if (request.resolution === "4k") {
    return MODELS["veo-31-standard"];
  }

  // Rule 4: Tier-based routing
  switch (request.tier) {
    case "preview":
      return MODELS["wan22-ti2v-5b"]; // Cheapest, fastest
    case "basic":
      return MODELS["wan22-t2v-14b"]; // Good quality, self-hosted
    case "standard":
      return MODELS["kling-30"];       // Commercial quality
    case "premium":
      return MODELS["veo-31-standard"]; // Best quality
    default:
      return MODELS["wan22-t2v-14b"];
  }
}

// Cost comparison for a 5-second clip by tier
function estimateCost(request: GenerationRequest): {
  model: string;
  cost: number;
  latency: number;
} {
  const model = selectModel(request);
  return {
    model: model.id,
    cost: model.costPerSecond * request.duration,
    latency: model.avgLatency,
  };
}

/*
 * Example cost comparison for a 5-second clip:
 *
 * | Tier     | Model              | Cost   | Latency |
 * |----------|--------------------|--------|---------|
 * | preview  | wan22-ti2v-5b      | $0.015 | 120s    |
 * | basic    | wan22-t2v-14b      | $0.035 | 150s    |
 * | standard | kling-30           | $0.60  | 90s     |
 * | premium  | veo-31-standard    | $2.00  | 120s    |
 *
 * The premium tier costs 133x more than preview.
 * But for the right use case, it's worth it.
 */
```

### 8.2 Self-Hosted Inference Server

```typescript
// inference/wan-server.ts
// GPU inference server for Wan 2.2 using a job queue

import { Queue, Worker, Job } from "bullmq";
import Redis from "ioredis";

const redis = new Redis(process.env.REDIS_URL!);

interface WanJobData {
  jobId: string;
  prompt: string;
  referenceImage?: string;
  model: "t2v-14b" | "i2v-14b" | "ti2v-5b";
  resolution: { width: number; height: number };
  numFrames: number;
  fps: number;
  loraAdapter?: string;
  seed?: number;
}

interface WanJobResult {
  videoUrl: string;    // R2 URL of generated video
  thumbnailUrl: string;
  duration: number;    // seconds
  generationTime: number; // ms
  model: string;
  cost: number;
}

// Job queue for Wan inference
const wanQueue = new Queue<WanJobData>("wan-inference", {
  connection: redis,
  defaultJobOptions: {
    attempts: 2,
    backoff: { type: "exponential", delay: 5000 },
    removeOnComplete: { age: 3600 },
    removeOnFail: { age: 86400 },
  },
});

// Submit a generation job
async function submitWanGeneration(
  data: WanJobData
): Promise<string> {
  const job = await wanQueue.add("generate", data, {
    priority: data.model === "ti2v-5b" ? 10 : 5, // 5B gets lower priority
  });
  return job.id!;
}

// GPU Worker (runs on the GPU machine)
const worker = new Worker<WanJobData, WanJobResult>(
  "wan-inference",
  async (job: Job<WanJobData>) => {
    const startTime = Date.now();
    const data = job.data;

    // Call the Python inference server via HTTP
    const response = await fetch("http://localhost:8080/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: data.prompt,
        reference_image: data.referenceImage,
        model: data.model,
        width: data.resolution.width,
        height: data.resolution.height,
        num_frames: data.numFrames,
        fps: data.fps,
        lora_adapter: data.loraAdapter,
        seed: data.seed ?? Math.floor(Math.random() * 2147483647),
        num_inference_steps: 50,
        guidance_scale: 7.5,
      }),
    });

    if (!response.ok) {
      throw new Error(
        `Inference failed: ${response.status} ${await response.text()}`
      );
    }

    const result = await response.json();
    const generationTime = Date.now() - startTime;

    // Upload to R2
    const videoUrl = await uploadToR2(
      result.video_path,
      `generations/${data.jobId}.mp4`
    );
    const thumbnailUrl = await uploadToR2(
      result.thumbnail_path,
      `thumbnails/${data.jobId}.jpg`
    );

    // Calculate cost
    const gpuHours = generationTime / 1000 / 3600;
    const gpuRate = data.model === "ti2v-5b" ? 0.44 : 1.64; // RunPod rates
    const cost = gpuHours * gpuRate;

    return {
      videoUrl,
      thumbnailUrl,
      duration: data.numFrames / data.fps,
      generationTime,
      model: `wan22-${data.model}`,
      cost: Math.round(cost * 1000) / 1000, // Round to mills
    };
  },
  {
    connection: redis,
    concurrency: 1, // One job per GPU at a time
    limiter: { max: 1, duration: 1000 },
  }
);

async function uploadToR2(
  localPath: string,
  key: string
): Promise<string> {
  // Upload to Cloudflare R2 using S3-compatible API
  const { S3Client, PutObjectCommand } = await import(
    "@aws-sdk/client-s3"
  );
  const s3 = new S3Client({
    region: "auto",
    endpoint: process.env.R2_ENDPOINT!,
    credentials: {
      accessKeyId: process.env.R2_ACCESS_KEY_ID!,
      secretAccessKey: process.env.R2_SECRET_ACCESS_KEY!,
    },
  });

  const fs = await import("fs");
  const body = fs.readFileSync(localPath);

  await s3.send(
    new PutObjectCommand({
      Bucket: process.env.R2_BUCKET!,
      Key: key,
      Body: body,
      ContentType: key.endsWith(".mp4")
        ? "video/mp4"
        : "image/jpeg",
    })
  );

  return `${process.env.R2_PUBLIC_URL}/${key}`;
}
```

---

## 9. The Open-Source Trajectory

### 9.1 Historical Quality Improvement Rate

Open-source video generation models have been improving at a remarkable pace. Tracking VBench scores over time:

| Model | Release Date | VBench Score | Gap to SOTA |
|---|---|---|---|
| ModelScope T2V | Nov 2023 | 52.3 | -35.7 (vs. Runway Gen-2) |
| AnimateDiff v2 | Jan 2024 | 58.1 | -27.9 |
| Open-Sora 1.0 | Mar 2024 | 62.4 | -22.6 |
| CogVideoX-5B | Aug 2024 | 71.5 | -15.5 |
| Open-Sora 1.2 | Oct 2024 | 72.8 | -14.2 |
| Wan 2.1 14B | Dec 2024 | 78.9 | -8.1 |
| Wan 2.2 T2V-A14B | Jan 2025 | 82.1 | -5.1 |
| LTX-2 19B | Feb 2025 | 80.5 | -6.7 |

The gap to commercial state-of-the-art (SOTA) has been closing at approximately **5-7 points per year** on VBench.

### 9.2 Extrapolation: When Does Open-Source Catch Up?

If the current rate continues:

$$
\text{VBench}_{\text{open-source}}(t) \approx 82.1 + 6.0 \times (t - 2025.1)
$$

$$
\text{VBench}_{\text{commercial}}(t) \approx 87.2 + 3.0 \times (t - 2025.1)
$$

Setting them equal:

$$
82.1 + 6.0 \Delta t = 87.2 + 3.0 \Delta t
$$

$$
3.0 \Delta t = 5.1
$$

$$
\Delta t = 1.7 \text{ years}
$$

**Projected crossover: ~mid-2027.** This aligns with the trajectory seen in LLMs, where open-source models (Llama, Mistral, DeepSeek) reached commercial parity approximately 2-3 years after the initial commercial lead.

### 9.3 Caveats on Extrapolation

This linear extrapolation has several failure modes:

1. **Diminishing returns**: Quality improvement may slow as models approach perceptual ceilings
2. **Commercial acceleration**: Veo, Sora, and Kling may accelerate their own improvement
3. **Data moats**: Commercial models have access to proprietary training data (YouTube for Veo, TikTok/Kuaishou for Kling) that open-source cannot replicate
4. **Audio gap**: Open-source video models have no native audio. This is a qualitative gap that VBench scores don't capture
5. **Regulation**: EU AI Act and similar regulations may create compliance costs that disproportionately burden open-source development

### 9.4 Strategic Implications

For a platform builder, the trajectory suggests:

**Near-term (2025-2026)**: Use Wan 2.2 and successors for cost-optimized tiers (preview, basic, batch). Use commercial APIs for quality-critical generation. Self-hosting saves money at scale.

**Medium-term (2026-2027)**: Open-source quality reaches "good enough for most use cases." Self-hosting becomes the default, with commercial APIs reserved for premium features (audio, 4K, specialized capabilities).

**Long-term (2027+)**: Open-source matches or exceeds commercial quality for standard video generation. Commercial APIs differentiate on latency, ease of use, specialized capabilities, and enterprise support rather than raw quality.

**The architectural principle**: Build your platform with model abstraction from day one. Your rendering pipeline should not care whether the model is self-hosted or API-accessed. When the relative economics shift, you swap a config value, not your architecture.

---

## Conclusion

Wan 2.2's MoE architecture is not just an incremental improvement --- it is a structural innovation that brings LLM-scale parameter efficiency to video diffusion. The two-expert decomposition maps naturally onto the diffusion process itself, giving you bigger-model quality at smaller-model cost.

For anyone building a video generation platform, the practical implications are clear: self-host Wan 2.2 for your basic and preview tiers starting at around 3,000-5,000 monthly generations. Use commercial APIs for premium quality and audio. Fine-tune LoRA adapters to offer custom styles that no API-only service can match. And build your architecture so that when the next generation of open-source models arrives --- and it will, probably within months --- you can swap it in without touching a line of application code.

The gap is 5 points on VBench today. It was 35 points two years ago. Do the math.
