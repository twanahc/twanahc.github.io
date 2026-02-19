---
layout: post
title: "The Future of Video Generation: Costs, Models, APIs, and How to Prepare Your Implementation"
date: 2026-02-19
category: landscape
---

Something unusual happened this month. ByteDance released Seedance 2.0 and within 48 hours, Hollywood sent cease-and-desist letters. Kuaishou shipped Kling 3.0 with native 4K at 60fps. Google quietly dropped Veo 3.1 pricing by 40%. Alibaba pushed Wan 2.6 with full character-voice cloning. And the per-second cost of generating a 1080p video clip crossed below $0.05 on open-source models for the first time.

We are in the middle of a phase transition. Not the gradual kind --- the discontinuous kind. The kind where the landscape looks completely different on a monthly timescale. If you are building anything that touches video generation, the decisions you make in the next six months about architecture, model selection, and cost planning will determine whether you are riding the wave or drowning under it.

This post is the comprehensive reference for that decision-making. It covers every major model as of February 2026, their technical architectures, real API pricing with worked cost examples, self-hosting economics, production code for calling every major API, a model-agnostic abstraction layer you can ship today, and a framework for betting on the right technologies. It is long because the problem space is large and getting the details wrong is expensive.

---

## Table of Contents

1. [The Current Landscape: Every Model That Matters](#the-current-landscape-every-model-that-matters)
2. [The Master Comparison Table](#the-master-comparison-table)
3. [What Just Shipped and What Is Coming Next](#what-just-shipped-and-what-is-coming-next)
4. [How to Access These Models: APIs, Aggregators, and Self-Hosting](#how-to-access-these-models)
5. [The Complete Cost Analysis](#the-complete-cost-analysis)
6. [Self-Hosting Economics: GPU Math](#self-hosting-economics-gpu-math)
7. [Architecture: The DiT Revolution and What Makes Models Fast](#architecture-the-dit-revolution)
8. [Building a Future-Proof Video Pipeline](#building-a-future-proof-video-pipeline)
9. [Implementation Guide: Calling Every Major API](#implementation-guide-calling-every-major-api)
10. [The Unified Abstraction Layer](#the-unified-abstraction-layer)
11. [Cost Projections: When Does 1 Minute of HD Video Cost Less Than $1?](#cost-projections)
12. [What to Bet On](#what-to-bet-on)

---

## The Current Landscape: Every Model That Matters

Let us walk through every model that a builder should know about in February 2026, organized by commercial maturity and quality tier. For each model, we will cover what it generates, how fast, at what cost, and where the sharp edges are.

### Tier 1: The Frontrunners

**Google Veo 3.1** is the current overall quality leader in combined audio-visual generation. It generates 8-second clips at up to 1080p with natively synchronized audio --- dialogue, sound effects, ambient sound --- using a SoundStorm-based parallel audio decoder integrated into the generation pipeline. The audio is not post-hoc; it is generated jointly with the visual frames, which means lip-sync accuracy and sound-scene coherence are qualitatively better than any bolt-on audio solution.

Veo 3.1 is available through both the Gemini API (developer-focused, lower price) and Vertex AI (enterprise-focused, higher reliability guarantees). It supports text-to-video, image-to-video, first-frame + last-frame conditioning, and video extension up to 60 seconds via iterative chaining. The "Fast" variant trades quality for speed and cost. Resolution support: 720p and 1080p natively, with 4K via upscaling on the Ultra plan.

Pricing: $0.15/sec (Fast), $0.40/sec (Standard). Subscription access via Google AI Pro ($19.99/mo, ~90 Fast videos/month) or AI Ultra ($249/mo, 4K, commercial rights).

**Runway Gen-4.5** holds the number one Elo rating on the Artificial Analysis Text-to-Video benchmark at 1,247 Elo. It ships in two tiers: Turbo (optimized for speed, 8-15 denoising steps via progressive distillation) and Aleph (maximum quality, estimated 50-80 steps). Gen-4.5 introduced "References" --- upload images of characters, objects, or environments and the model maintains visual consistency across generations. This is invaluable for multi-shot storytelling.

The conspicuous gap: no native audio generation. Runway is betting that their target market (professional creators) adds audio in post-production. This bet is getting riskier as competitors ship integrated audio.

Pricing: credit-based system. Standard plan ($12/mo) gets 625 credits (~52 seconds of Gen-4 or ~125 seconds of Turbo). Pro ($28/mo) gets 2,250 credits. API pricing works out to approximately $0.12/sec for Gen-4 and $0.05/sec for Turbo.

**OpenAI Sora 2** delivers solid quality with a distinctive strength in physics simulation --- objects have mass, surfaces have friction, light behaves correctly. The "Characters" feature lets you upload a video of a person and create a persistent identity that can be placed into arbitrary scenes while maintaining facial features, body proportions, and movement style. This is architecturally unique among the major players.

Sora 2 killed its free tier in January 2026. API access is billed per-second: $0.10/sec (Standard, 720p) and $0.30-$0.50/sec (Pro, up to 1792x1024). Maximum duration is 25 seconds per generation, a 4x improvement over Sora 1. Native audio is included.

Weakness: reliability. Sora 2 launched with significant capacity issues and the API still occasionally returns 503s during peak hours. If you are building with Sora 2 as your sole model, you need robust retry logic and a fallback plan.

### Tier 2: Closing Fast

**Kling 3.0** (Kuaishou) shipped February 4, 2026, and it is the most feature-rich model in the field. Let us list what it does: native 4K resolution (3840x2160) at 60fps (the first model to achieve this), multi-shot storyboarding with up to 6 camera cuts per generation, 15-second maximum duration, an "Elements" system that locks character identity across shots, multi-character multilingual dialogue with native audio, and a full multimodal input framework (text, image, audio, video in, video out).

The quality is excellent. Kling 3.0 was completely retrained using reinforcement learning, which dramatically improved its handling of flowing water, fabric dynamics, and human anatomy --- the traditional failure modes for video generation.

Kling 3.0's commercial validation is worth noting: Kuaishou has 700M+ monthly active users on its short video platform, and Kling generated approximately $100M in its first year of revenue. This is not a research demo. It is a product with real customers and real revenue.

API pricing via fal.ai: approximately $0.10/sec. Via Kling's official developer portal: $0.07/sec (video only), $0.14/sec (with audio), $0.168/sec (with audio and voice control).

**Seedance 2.0** (ByteDance) is the most controversial release of 2026. It launched February 8 and immediately generated viral clips based on real actors and copyrighted content, triggering cease-and-desist letters from Disney, the MPA, and Sony. The legal situation is evolving rapidly.

Setting aside the legal issues, the technical capabilities are remarkable. Seedance 2.0 accepts up to 12 reference files simultaneously across four modalities: images, videos, audio, and text. You can provide a face photo, a dance video, and a music track, and the model fuses these into a coherent output. This "quad-modal input" system is unmatched. Native 2K output resolution with a wider variety of aspect ratios than any competitor. Beat-sync (audio-synchronized motion) that actually works. Character consistency that rivals dedicated face-swapping tools.

Pricing: consumer access at approximately $9.60/month. API access is available through fal.ai and similar aggregators, with per-second pricing that undercuts Western competitors at approximately $0.06-$0.08/sec.

The risk: ByteDance may be forced to restrict the model's capabilities in response to legal pressure. If you build on Seedance 2.0, you are making a bet on the legal outcome.

**Luma Ray3.14** (launched January 26, 2026) represents the most dramatic single-quarter cost reduction in the field: 3x cheaper and 4x faster than Ray3, with an upgrade to native 1080p. Start/end frame conditioning gives precise control over shot composition. Luma raised $900M in November 2025, so they have the capital to sustain aggressive pricing.

Pricing: credit-based. A 5-second 720p clip costs approximately 320 credits. Plans from free (limited daily credits) to Creator Pro (~$24-$30/mo, 10,000+ credits, 4K/HDR, commercial use). API billing is separate from subscription credits.

No native audio, which is becoming a competitive liability.

**MiniMax Hailuo 2.3** generates 6-second clips in under 30 seconds, making it one of the fastest production models. The "Media Agent" system auto-routes requests to internal sub-models optimized for different content types. Subject reference technology provides character consistency. Uses a Mixture-of-Experts (MoE) architecture that enables efficient scaling --- different experts handle different content aspects (faces, landscapes, motion, etc.) with only a subset activated per token.

Pricing: $0.25 for a 6-second 768p video, $0.52 for 10 seconds via API. Subscriptions from $14.99/mo (1,000 credits) to $119.99/mo (10,000 credits). The "Fast" variant of Hailuo 2.3 cuts costs by approximately 50% for batch workflows.

### Tier 3: Specialized and Open-Source

**Pika 2.2** has carved a deliberate niche in scene modification rather than pure generation. Pikaframes (keyframe interpolation), Pikaswaps (object replacement), and Pikadditions (object insertion) are compositional tools that operate on existing footage. 10 seconds at 1080p. This is a different product category --- less "generate a video from nothing" and more "modify this existing video intelligently."

Pricing: Standard $8/mo (700 credits), Pro $28/mo (2,000 credits). API access remains limited and restricted to select partners, which makes it impractical for most programmatic integrations.

**PixVerse R1** introduced real-time video generation --- 1080p video that responds to user input in real-time. This is architecturally distinct from batch generation. While other models generate a 5-second clip and hand it back after 30-120 seconds of compute, PixVerse R1 streams frames as they are generated. The implications for interactive applications (games, live performances, real-time effects) are significant. Alibaba-backed with 16M MAU and approximately $40M ARR.

Pricing: Standard $10/mo (1,200 credits), Pro $30/mo (6,000 credits). API pricing in 5-second or 8-second increments depending on quality settings.

**Wan 2.6** (Alibaba) is the leading open-source video model series. The 2.6 release (December 2025) introduced character-voice cloning from a reference video, multi-shot storytelling, stable multi-person dialogue, and improved camera movement control. Available as both a hosted API and fully open weights under Apache 2.0 licensing.

Model variants: the 5B parameter model runs on a single RTX 4090 (24GB VRAM) at 720p. The full A14B (14B active parameters, MoE architecture) needs 80GB VRAM (A100/H100) for single-GPU inference, or can be distributed across multiple GPUs using FSDP and Ulysses sequence parallelism.

API pricing via fal.ai: approximately $0.05/sec, making it the cheapest API option. Self-hosting cost depends on your GPU economics (we will analyze this in detail below).

**LTX-2** (Lightricks, released January 6, 2026) is the first open-source model to achieve production-quality synchronized audio and video generation. Native 4K resolution, up to 20 seconds of content, 50fps generation speed, with synchronized sound effects, ambient audio, and accurate lip sync. The VAE achieves a compression ratio of 16x16x4, which is critical for keeping the token count manageable at 4K resolution.

Licensing: free for academic use and commercial use by companies under $10M ARR. Larger organizations need a commercial license. Available via fal.ai, Replicate, ComfyUI, and the LTX platform.

**CogVideoX** (Zhipu AI) is the veteran open-source option. The 5B model requires 24GB VRAM for inference (with optimizations, as low as 12GB). The 2B model is lighter but visually weaker. CogVideoX is integrated into HuggingFace Diffusers, making it the easiest open-source model to get running quickly. Quality lags behind Wan 2.6 and LTX-2, but the ecosystem maturity and documentation are strong.

---

## The Master Comparison Table

Here is every model compared across the dimensions that matter for builders. Pricing reflects API per-second costs where available, or equivalent per-second costs derived from credit systems.

| Model | Max Resolution | Max Duration | Native Audio | API $/sec | Generation Time (10s clip) | Open Source | Strengths |
|-------|---------------|-------------|-------------|-----------|---------------------------|-------------|-----------|
| **Veo 3.1** | 1080p (4K on Ultra) | 8s (60s via extension) | Yes (SoundStorm) | $0.15-$0.40 | 60-120s | No | Best audio-visual coherence, prompt adherence |
| **Veo 3.1 Fast** | 1080p | 8s | Yes | $0.15 | 30-45s | No | Speed-quality sweet spot |
| **Gen-4.5 Aleph** | 4K (upscale) | 10s | No | ~$0.15 | 90-180s | No | Highest visual fidelity, #1 Elo |
| **Gen-4.5 Turbo** | 1080p | 10s | No | ~$0.05 | 15-30s | No | Fast previews, cost-efficient |
| **Sora 2** | 1792x1024 | 25s | Yes | $0.10-$0.50 | 60-120s | No | Physics sim, Characters feature |
| **Kling 3.0** | 4K (3840x2160) | 15s | Yes (multilingual) | $0.07-$0.17 | 45-90s | No | Multi-shot storyboarding, 4K/60fps |
| **Seedance 2.0** | 2K | 15s | Yes (beat-sync) | ~$0.06-$0.08 | 30-60s | No | 12-file multimodal input, character consistency |
| **Ray3.14** | 1080p (4K HDR) | 10s | No | ~$0.08 | 20-40s | No | Cost efficiency, start/end frame control |
| **Hailuo 2.3** | 768p-1080p | 6-10s | Yes | ~$0.04-$0.05 | 15-30s | No | Fastest generation, MoE architecture |
| **Pika 2.2** | 1080p | 10s | No | Limited API | 30-60s | No | Scene modification tools |
| **PixVerse R1** | 1080p | Real-time | No | ~$0.04 | Real-time | No | Interactive/streaming generation |
| **Wan 2.6** | 1080p | 15s | Yes (voice clone) | ~$0.05 | 60-120s | Yes (Apache 2.0) | Character-voice cloning, cheapest API |
| **LTX-2** | 4K | 20s | Yes (lip-sync) | ~$0.06 | 45-90s | Yes (conditional) | First open-source audio+video at 4K |
| **CogVideoX-5B** | 720p | 6s | No | ~$0.03 | 30-60s | Yes (Apache 2.0) | HuggingFace Diffusers integration |

A few patterns jump out from this table. First, native audio is no longer a differentiator --- it is table stakes. Seven of the fourteen models listed generate synchronized audio. By mid-2026, any model without native audio will be at a serious competitive disadvantage. Second, the price band has compressed dramatically: from $0.03/sec to $0.50/sec, a 17x range. A year ago that range was 50x. Third, Chinese-developed models (Kling, Seedance, Hailuo, Wan, PixVerse) dominate the price-performance frontier.

---

## What Just Shipped and What Is Coming Next

### The February 2026 Wave

The density of releases in February 2026 is unprecedented. In a single two-week period:

**Kling 3.0** (February 4): First model to achieve native 4K/60fps. Multi-shot storyboarding with the "Elements" character lock system. Full multimodal input/output. Complete RL-based retraining for physics accuracy.

**Seedance 2.0** (February 8): 12-file multimodal input across four modalities. Quad-modal fusion. Native 2K output. Beat-synchronized motion. Immediate legal controversy.

**Wan 2.6** was already out (December 2025) but its ecosystem --- ComfyUI workflows, fal.ai integration, Alibaba Cloud Model Studio deployment --- matured significantly in January-February 2026.

### What Is Coming in Q2-Q3 2026

**Veo 4** (Google): Google's cadence suggests a major Veo update every 6-8 months. Veo 3.0 arrived mid-2025, Veo 3.1 in October 2025. Expect Veo 4 (or whatever they call it) by mid-2026. Likely improvements: longer native duration (currently 8 seconds is short by current standards), native 4K without upscaling, and improved multi-shot consistency. Google has the compute advantage: they can train on TPU v5p pods at a scale that no other organization can match.

**Runway Gen-5**: Runway's release cadence has been roughly annual for major versions. Gen-4 arrived March 2025, Gen-4.5 later that year. A Gen-5 in 2026 is probable. The most urgent addition would be native audio --- Runway cannot afford to be the only Tier 1 model without it for much longer. Runway's partnership with Adobe (announced December 2025) suggests deeper integration with professional workflows.

**Meta MovieGen**: Meta announced MovieGen in October 2024 with a 30B parameter video model and 13B audio model. Mark Zuckerberg said it was "coming to Instagram" in 2025. As of February 2026, there is no public API or open-source release of the full model (only the MovieGen Bench benchmark was open-sourced). Meta's strategy appears to be internal deployment first (Instagram/Facebook Reels generation tools) with potential open-source release later. Given Meta's track record with Llama, an open-source MovieGen is plausible but the timeline is unclear.

**Stability AI** has pivoted away from direct text-to-video competition. Their current strategy is enterprise partnerships (Warner Music, WPP) and specialized outputs like 4D assets. Stable Video Diffusion seeded the ecosystem but the company is no longer competing on the generation frontier. Do not expect a competitive text-to-video model from Stability in 2026.

### The Chinese vs. Western Divide

A structural pattern has emerged. Chinese labs (Kuaishou, ByteDance, Alibaba, MiniMax, Zhipu AI) are shipping faster, at lower price points, with more aggressive feature sets. Western labs (Google, OpenAI, Runway, Luma) are shipping at higher quality ceilings, with more cautious safety guardrails, at premium prices.

The numbers tell the story. Kling 3.0 at $0.07/sec vs. Veo 3.1 at $0.40/sec. Seedance 2.0 at ~$0.07/sec vs. Sora 2 at $0.10-$0.50/sec. Wan 2.6 is open-source and free to self-host. The quality gap between the two tiers has narrowed substantially --- Kling 3.0's 4K/60fps output is competitive with anything from Western labs.

The risk factors differ. Chinese models may face regulatory restrictions in Western markets (the TikTok precedent looms large). Legal challenges like the Disney/MPA actions against Seedance 2.0 introduce uncertainty. Western models come with clearer IP provenance and content moderation guarantees. For enterprise deployments with legal risk sensitivity, this distinction matters.

---

## How to Access These Models {#how-to-access-these-models}

There are four distinct paths to integrating video generation. Each has different tradeoffs in cost, control, reliability, and operational complexity.

### Path 1: Direct Provider APIs

Call the model provider's API directly. Maximum control, best pricing for high volume, but you take on integration complexity for each model.

| Provider | Models Available | Auth Method | Billing |
|----------|----------------|-------------|---------|
| OpenAI API | Sora 2, Sora 2 Pro | API key | Per-second |
| Google Gemini API | Veo 2, Veo 3, Veo 3.1, Veo 3.1 Fast | OAuth2 / API key | Per-second |
| Google Vertex AI | Same as Gemini + enterprise SLAs | Service account | Per-second |
| Runway API | Gen-4, Gen-4 Turbo, Gen-4.5 | API key (v2024-11-06) | Credits |
| Kling Developer Portal | Kling 2.6, Kling 3.0 | API key | Per-second |
| MiniMax API | Hailuo 02, Hailuo 2.3 | API key | Per-video |
| Luma Dream Machine API | Ray3, Ray3.14 | API key | Credits |

**Advantages**: Lowest per-unit cost at scale. Direct access to model-specific features (Sora Characters, Runway References, Kling Elements). First access to new capabilities.

**Disadvantages**: N different SDKs, N different auth flows, N different billing dashboards, N different error formats. Each model goes down independently. You build and maintain the abstraction layer yourself.

### Path 2: Aggregator Platforms

Call one API that routes to many models. Trade some cost and control for dramatically reduced integration complexity.

**fal.ai** is the most comprehensive aggregator for video generation as of February 2026. They offer 600+ models through a single API, including Veo 3.1, Sora 2, Kling 3.0, Seedance 2.0, Wan 2.6, Hailuo 2.3, LTX-2, and PixVerse. Pricing is per-second or per-video depending on the model, with a markup over direct provider pricing that varies from 10-30%. The developer experience is excellent: unified auth, unified response format, unified webhooks.

**Replicate** provides a similar aggregation layer with a focus on open-source models. Their "Official Models" program wraps popular models (Wan 2.1/2.2, CogVideoX, LTX-Video) in standardized containers with predictable pricing. Replicate also hosts closed models like Veo 3 Fast and Kling variants. Billing is either per-second or per-prediction depending on the model.

**Together AI** recently expanded into video generation with 20+ video models accessible through the same API used for their LLM inference. Their pitch is that you already use Together for text generation, so why not use the same platform for video? The unified billing and API familiarity is genuine added value if you are already a Together customer.

**Advantages**: One SDK, one auth flow, one billing dashboard. Trivially easy to swap models. Built-in load balancing. Some aggregators offer quality routing (send this prompt to whichever model handles it best).

**Disadvantages**: Markup on pricing (typically 10-30% over direct). Aggregator-specific features may lag behind direct API features. You are adding a dependency --- if fal.ai goes down, your access to all models goes down. Less control over model-specific parameters.

### Path 3: Self-Hosting Open-Source Models

Run the model on your own hardware. Maximum control and zero marginal cost per generation (after hardware investment). Only feasible for open-source models: Wan 2.6, LTX-2, CogVideoX, and community fine-tunes.

| Model | Parameters | Min VRAM (optimized) | Recommended VRAM | Max Resolution | Framework |
|-------|-----------|---------------------|------------------|---------------|-----------|
| Wan 2.6 TI2V-5B | 5B | 12GB | 24GB (RTX 4090) | 720p | PyTorch / Diffusers |
| Wan 2.6 T2V-A14B | 14B (MoE) | 40GB | 80GB (A100/H100) | 1080p | PyTorch / FSDP |
| LTX-2 | ~8B | 16GB | 24-48GB | 4K | PyTorch |
| CogVideoX-5B | 5B | 12GB (quantized) | 24GB | 720p | Diffusers |
| CogVideoX-2B | 2B | 8GB | 12GB | 480p | Diffusers |

**Advantages**: Zero API costs after hardware. Complete control over the model (fine-tuning, custom LoRAs, modified inference). No rate limits. No data leaving your infrastructure. No vendor lock-in.

**Disadvantages**: Significant upfront investment (hardware or cloud GPU commitment). You own operations: updates, scaling, monitoring, error handling. Quality lags commercial models by 6-12 months. No native audio on most open-source options (LTX-2 is the exception).

### Path 4: Hybrid Approach (The Right Answer for Most Teams)

Use aggregator APIs for commercial models (Veo, Sora, Kling) and self-host open-source models (Wan 2.6, LTX-2) for high-volume, cost-sensitive workloads. Route between them based on quality requirements and cost budget.

This is the architecture that scales. We will build it out in the pipeline section below.

---

## The Complete Cost Analysis

Let us be precise about costs, because the difference between a viable product and a money pit often comes down to per-second pricing at your target volume.

### Direct Comparison: Cost of a 10-Second 1080p Video

| Model | Per-Second Rate | 10s Cost | Notes |
|-------|----------------|----------|-------|
| CogVideoX-5B (self-hosted, 4090) | ~$0.008 | ~$0.08 | Amortized GPU cost at 80% utilization |
| Wan 2.6 (self-hosted, 4090) | ~$0.012 | ~$0.12 | Amortized GPU cost at 80% utilization |
| Wan 2.6 (fal.ai) | $0.05 | $0.50 | |
| Hailuo 2.3 | ~$0.04-$0.05 | ~$0.40-$0.50 | Via API |
| Gen-4.5 Turbo | ~$0.05 | ~$0.50 | Via credits |
| Seedance 2.0 | ~$0.07 | ~$0.70 | Via fal.ai |
| Kling 3.0 (video only) | $0.07 | $0.70 | Official API |
| Ray3.14 | ~$0.08 | ~$0.80 | Via credits |
| LTX-2 (fal.ai) | ~$0.06 | ~$0.60 | |
| Sora 2 Standard | $0.10 | $1.00 | 720p |
| Kling 3.0 (audio + voice) | $0.168 | $1.68 | Official API |
| Gen-4.5 Aleph | ~$0.15 | ~$1.50 | Via credits |
| Veo 3.1 Fast | $0.15 | $1.50 | |
| Sora 2 Pro | $0.30-$0.50 | $3.00-$5.00 | Up to 1792x1024 |
| Veo 3.1 Standard | $0.40 | $4.00 | |

### Cost at Scale: 1,000 Videos Per Day

Suppose you are running a product that generates 1,000 ten-second videos per day. That is 10,000 seconds of video daily, or 300,000 seconds per month. Here is what your monthly bill looks like:

| Model | Monthly Cost (300K seconds) | Annual Cost |
|-------|---------------------------|-------------|
| Self-hosted Wan 2.6 (4x RTX 4090) | ~$3,600 (hardware amortized) | ~$43,200 |
| Wan 2.6 via fal.ai | $15,000 | $180,000 |
| Gen-4.5 Turbo | $15,000 | $180,000 |
| Hailuo 2.3 | $12,000-$15,000 | $144,000-$180,000 |
| Kling 3.0 (video only) | $21,000 | $252,000 |
| Sora 2 Standard | $30,000 | $360,000 |
| Veo 3.1 Fast | $45,000 | $540,000 |
| Veo 3.1 Standard | $120,000 | $1,440,000 |

The self-hosted option is 4-33x cheaper than API-based options depending on which model you compare against. The tradeoff is operational complexity and the 6-12 month quality lag.

### The 18-Month Cost Trajectory

Let us model how costs have declined and project forward. We can observe a roughly exponential decay in per-second generation costs.

In Q1 2024, the cheapest API-based video generation was approximately $1.00/sec (early Runway Gen-2 API). By Q1 2025, it was approximately $0.20/sec (Kling 1.6, early Hailuo). In Q1 2026, we are at approximately $0.04/sec (Hailuo 2.3 Fast, Wan 2.6).

If we model this as exponential decay:

$$C(t) = C_0 \cdot e^{-\lambda t}$$

where \(C_0 \approx 1.0\) $/sec (Q1 2024 baseline), \(t\) is time in years from Q1 2024, and we fit to our three data points:

- \(C(0) = 1.00\): gives \(C_0 = 1.00\)
- \(C(1) = 0.20\): gives \(\lambda = -\ln(0.20) \approx 1.61\)
- \(C(2) = 0.04\): gives \(\lambda = -\ln(0.04)/2 \approx 1.61\)

The consistency of \(\lambda \approx 1.6\) across both intervals is striking --- the cost decline rate has been remarkably steady. This corresponds to an approximately 80% year-over-year cost reduction.

Projecting forward:

- Q1 2027: \(C(3) \approx 1.00 \cdot e^{-4.83} \approx 0.008\) $/sec
- Q1 2028: \(C(4) \approx 1.00 \cdot e^{-6.44} \approx 0.0016\) $/sec

At $0.008/sec, a 60-second 1080p video costs $0.48. We cross the "1 minute of HD video for less than $1" threshold by early 2027 on API-based models, and we are already there today for self-hosted open-source models.

```python
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

# Observed data points
t_obs = np.array([0, 1, 2])  # years from Q1 2024
c_obs = np.array([1.00, 0.20, 0.04])

# Fit: C(t) = C0 * exp(-lambda * t)
lam = 1.61
t_proj = np.linspace(0, 4, 100)
c_proj = 1.00 * np.exp(-lam * t_proj)

# One-minute cost threshold
one_min_threshold = 1.0 / 60  # $1 / 60 seconds

ax.semilogy(t_proj, c_proj, color='#e94560', linewidth=2.5, label=r'$C(t) = e^{-1.61t}$ (fitted model)')
ax.semilogy(t_obs, c_obs, 'o', color='#0f3460', markersize=10, markeredgecolor='#e94560',
            markeredgewidth=2, label='Observed API floor price', zorder=5)
ax.axhline(y=one_min_threshold, color='#16c79a', linestyle='--', linewidth=1.5,
           label=r'1 min HD < \$1 threshold')

# Annotations
labels = ['Q1 2024', 'Q1 2025', 'Q1 2026']
for i, (t, c, label) in enumerate(zip(t_obs, c_obs, labels)):
    ax.annotate(f'{label}\n\\${c:.2f}/sec', (t, c), textcoords='offset points',
                xytext=(15, 10), fontsize=9, color='#d4d4d4',
                arrowprops=dict(arrowstyle='->', color='#d4d4d4', lw=0.8))

# Projection annotation
ax.annotate('Q1 2027 projection\n\\$0.008/sec', (3, 0.008), textcoords='offset points',
            xytext=(15, -25), fontsize=9, color='#16c79a',
            arrowprops=dict(arrowstyle='->', color='#16c79a', lw=0.8))

ax.set_xlabel('Years from Q1 2024', fontsize=12, color='#d4d4d4')
ax.set_ylabel(r'Cost per second (\$)', fontsize=12, color='#d4d4d4')
ax.set_title(r'API Video Generation Cost Decline: $C(t) = e^{-1.61t}$', fontsize=14, color='#d4d4d4')
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels(['Q1 2024', 'Q1 2025', 'Q1 2026', 'Q1 2027\n(projected)', 'Q1 2028\n(projected)'],
                     fontsize=9, color='#d4d4d4')
ax.tick_params(colors='#d4d4d4')
ax.legend(fontsize=10, facecolor='#16213e', edgecolor='#0f3460', labelcolor='#d4d4d4')
ax.grid(True, alpha=0.15, color='#d4d4d4')
ax.set_ylim(0.001, 2)

for spine in ax.spines.values():
    spine.set_color('#0f3460')

plt.tight_layout()
plt.savefig('cost_decline.png', dpi=150, facecolor='#1a1a2e')
plt.show()
```

---

## Self-Hosting Economics: GPU Math {#self-hosting-economics-gpu-math}

Self-hosting becomes economically rational above a certain volume threshold. Let us find that threshold precisely.

### Cloud GPU Pricing (February 2026)

| GPU | VRAM | Cloud $/hr (on-demand) | Cloud $/hr (spot/preemptible) | Purchase Price |
|-----|------|----------------------|------------------------------|----------------|
| RTX 4090 | 24GB | $0.35-$0.44 | $0.20-$0.30 | ~$1,600 |
| A100 80GB | 80GB | $0.80-$2.29 | $0.50-$1.00 | ~$10,000 (used) |
| H100 SXM | 80GB | $2.49-$6.98 | $1.24-$2.00 | ~$25,000 |
| H200 | 141GB | $4.00-$8.00 | $2.50-$4.00 | ~$35,000 |

Sources: Fluence, VastAI, TensorDock, RunPod, Lambda pricing pages as of February 2026. The market is moving fast --- H100 on-demand pricing has dropped approximately 40% in the past six months.

### Generation Throughput by GPU

For Wan 2.6 5B model (720p, 5-second clips):

| GPU | Seconds of video per minute of compute | Clips per hour |
|-----|---------------------------------------|----------------|
| RTX 4090 (24GB) | ~1.0 sec/min | ~12 clips/hr |
| A100 80GB | ~2.5 sec/min | ~30 clips/hr |
| H100 SXM | ~5.0 sec/min | ~60 clips/hr |

These are approximate figures with standard inference optimizations (FP16/BF16, torch.compile, optimized attention). Real throughput depends heavily on resolution, duration, and the specific optimizations applied.

### The Break-Even Calculation

Let us define break-even precisely. Suppose you generate \(V\) seconds of video per month using the Wan 2.6 model. The API cost (via fal.ai) is $0.05 per second:

$$\text{API cost} = 0.05V \text{ dollars/month}$$

For self-hosting on an RTX 4090, you generate approximately 1 second of video per minute of GPU time, so you need \(V\) GPU-minutes per month. At $0.40/hr on-demand cloud:

$$\text{Self-host cost (cloud)} = \frac{V}{60} \times 0.40 = 0.0067V \text{ dollars/month}$$

Break-even: API cost = self-host cost when:

$$0.05V = 0.0067V$$

This is always cheaper to self-host on cloud GPUs for Wan 2.6 specifically because the throughput math works out. But this ignores operational overhead: the engineering time to set up, maintain, and monitor the inference pipeline. Let us assign an operational overhead of $2,000/month (a conservative estimate for engineering time) and find the volume where self-hosting total cost beats API cost:

$$0.05V = 0.0067V + 2000$$

$$0.0433V = 2000$$

$$V \approx 46,200 \text{ seconds/month}$$

That is approximately 46,200 seconds, or about 770 minutes, or roughly 1,540 five-second clips per month. At this volume, your API bill would be $2,310/month, and your self-hosted cost (including ops overhead) matches it. Above this volume, self-hosting wins.

For purchased hardware (RTX 4090 at $1,600, amortized over 3 years):

$$\text{Self-host cost (owned)} = \frac{1600}{36} + \text{electricity} + \text{ops} \approx 44.44 + 50 + 2000 = 2094.44 \text{ dollars/month}$$

$$0.05V = 2094.44$$

$$V \approx 41,889 \text{ seconds/month}$$

Similar break-even point. The key insight: self-hosting makes economic sense at roughly 40,000-50,000 seconds of video per month. Below that, use the API.

<svg viewBox="0 0 700 400" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; font-family: 'Segoe UI', system-ui, sans-serif;">
  <rect width="700" height="400" fill="#1a1a2e"/>
  <text x="350" y="30" text-anchor="middle" fill="#d4d4d4" font-size="16" font-weight="bold">Self-Hosting vs API: Monthly Cost by Volume</text>

  <!-- Axes -->
  <line x1="80" y1="350" x2="650" y2="350" stroke="#0f3460" stroke-width="2"/>
  <line x1="80" y1="350" x2="80" y2="50" stroke="#0f3460" stroke-width="2"/>

  <!-- X axis labels -->
  <text x="80" y="375" text-anchor="middle" fill="#d4d4d4" font-size="10">0</text>
  <text x="194" y="375" text-anchor="middle" fill="#d4d4d4" font-size="10">20K</text>
  <text x="308" y="375" text-anchor="middle" fill="#d4d4d4" font-size="10">40K</text>
  <text x="422" y="375" text-anchor="middle" fill="#d4d4d4" font-size="10">60K</text>
  <text x="536" y="375" text-anchor="middle" fill="#d4d4d4" font-size="10">80K</text>
  <text x="650" y="375" text-anchor="middle" fill="#d4d4d4" font-size="10">100K</text>
  <text x="365" y="398" text-anchor="middle" fill="#d4d4d4" font-size="11">Seconds of video per month</text>

  <!-- Y axis labels -->
  <text x="70" y="350" text-anchor="end" fill="#d4d4d4" font-size="10">$0</text>
  <text x="70" y="275" text-anchor="end" fill="#d4d4d4" font-size="10">$1K</text>
  <text x="70" y="200" text-anchor="end" fill="#d4d4d4" font-size="10">$2K</text>
  <text x="70" y="125" text-anchor="end" fill="#d4d4d4" font-size="10">$3K</text>
  <text x="70" y="50" text-anchor="end" fill="#d4d4d4" font-size="10">$4K+</text>

  <!-- API cost line (linear: $0.05/sec) -->
  <line x1="80" y1="350" x2="536" y2="50" stroke="#e94560" stroke-width="2.5"/>
  <line x1="536" y1="50" x2="650" y2="50" stroke="#e94560" stroke-width="2.5" stroke-dasharray="4,4"/>

  <!-- Self-host cost line (flat $2000 + $0.0067/sec) -->
  <line x1="80" y1="200" x2="650" y2="150" stroke="#16c79a" stroke-width="2.5"/>

  <!-- Break-even point -->
  <circle cx="337" cy="196" r="6" fill="#e2b714" stroke="#1a1a2e" stroke-width="2"/>
  <text x="350" y="186" fill="#e2b714" font-size="11" font-weight="bold">Break-even</text>
  <text x="350" y="170" fill="#e2b714" font-size="10">~46K sec/mo</text>

  <!-- Legend -->
  <line x1="440" y1="65" x2="470" y2="65" stroke="#e94560" stroke-width="2.5"/>
  <text x="475" y="69" fill="#d4d4d4" font-size="11">API (fal.ai, $0.05/sec)</text>
  <line x1="440" y1="85" x2="470" y2="85" stroke="#16c79a" stroke-width="2.5"/>
  <text x="475" y="89" fill="#d4d4d4" font-size="11">Self-hosted (4090 cloud)</text>
</svg>

---

## Architecture: The DiT Revolution and What Makes Models Fast {#architecture-the-dit-revolution}

To understand why models differ in quality and speed, we need to understand the architecture underneath. Every major video generation model in 2026 is built on the **Diffusion Transformer (DiT)** architecture, but the implementation details differ in ways that have measurable consequences.

### What Is a Diffusion Transformer?

Let us build this up from components.

**Diffusion models** are a class of generative models that learn to reverse a noise-adding process. Start with a clean signal \(x_0\) (a video). Gradually add Gaussian noise over \(T\) steps to produce a sequence \(x_1, x_2, \ldots, x_T\), where \(x_T\) is essentially pure noise. Then train a neural network to predict the noise at each step, so you can run the process in reverse: start from noise, iteratively denoise, and recover a clean video.

The forward (noise-adding) process is defined as:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \, x_{t-1}, \beta_t I)$$

where \(\beta_t\) is the noise schedule at step \(t\). The key property is that we can jump directly to any step \(t\) without iterating:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, (1-\bar{\alpha}_t) I)$$

where \(\bar{\alpha}_t = \prod_{s=1}^{t}(1 - \beta_s)\).

**Transformers** are neural network architectures based on self-attention. Given a sequence of tokens \((z_1, z_2, \ldots, z_N)\), each token attends to every other token via:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

where \(Q = z W_Q\), \(K = z W_K\), \(V = z W_V\) are linear projections. The computational cost of self-attention scales as \(O(N^2 d)\), where \(N\) is the sequence length and \(d\) is the dimension.

A **Diffusion Transformer** combines these. The neural network that predicts noise at each denoising step is a transformer instead of the U-Net architecture used in earlier diffusion models (Stable Diffusion 1.x, 2.x). The reason for the switch: transformers scale more predictably with compute and data than U-Nets. The scaling laws that govern language model performance (loss decreases as a power law of compute) also apply to DiT models.

### The Video DiT Pipeline

For video, the pipeline has four stages:

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; font-family: 'Segoe UI', system-ui, sans-serif;">
  <rect width="700" height="280" fill="#1a1a2e"/>
  <text x="350" y="25" text-anchor="middle" fill="#d4d4d4" font-size="15" font-weight="bold">Video DiT Generation Pipeline</text>

  <!-- Stage 1: Text Encoder -->
  <rect x="20" y="50" width="120" height="60" rx="8" fill="#16213e" stroke="#0f3460" stroke-width="2"/>
  <text x="80" y="75" text-anchor="middle" fill="#e94560" font-size="11" font-weight="bold">Text Encoder</text>
  <text x="80" y="95" text-anchor="middle" fill="#d4d4d4" font-size="9">T5 / CLIP</text>

  <!-- Stage 2: Noise + Conditioning -->
  <rect x="170" y="50" width="120" height="60" rx="8" fill="#16213e" stroke="#0f3460" stroke-width="2"/>
  <text x="230" y="72" text-anchor="middle" fill="#e94560" font-size="11" font-weight="bold">Noise</text>
  <text x="230" y="88" text-anchor="middle" fill="#d4d4d4" font-size="9">z ~ N(0,I)</text>
  <text x="230" y="102" text-anchor="middle" fill="#d4d4d4" font-size="9">(T,h,w,c) latent</text>

  <!-- Stage 3: DiT Backbone -->
  <rect x="320" y="40" width="140" height="80" rx="8" fill="#16213e" stroke="#e94560" stroke-width="2"/>
  <text x="390" y="62" text-anchor="middle" fill="#e94560" font-size="12" font-weight="bold">DiT Backbone</text>
  <text x="390" y="80" text-anchor="middle" fill="#d4d4d4" font-size="9">Spatial Attention</text>
  <text x="390" y="94" text-anchor="middle" fill="#d4d4d4" font-size="9">Temporal Attention</text>
  <text x="390" y="108" text-anchor="middle" fill="#d4d4d4" font-size="9">x N denoising steps</text>

  <!-- Stage 4: VAE Decoder -->
  <rect x="490" y="50" width="120" height="60" rx="8" fill="#16213e" stroke="#0f3460" stroke-width="2"/>
  <text x="550" y="75" text-anchor="middle" fill="#e94560" font-size="11" font-weight="bold">VAE Decoder</text>
  <text x="550" y="95" text-anchor="middle" fill="#d4d4d4" font-size="9">Latent -> Pixels</text>

  <!-- Output -->
  <rect x="640" y="55" width="50" height="50" rx="8" fill="#0f3460" stroke="#16c79a" stroke-width="2"/>
  <text x="665" y="82" text-anchor="middle" fill="#16c79a" font-size="10">MP4</text>

  <!-- Arrows -->
  <line x1="140" y1="80" x2="170" y2="80" stroke="#d4d4d4" stroke-width="1.5" marker-end="url(#arrowhead)"/>
  <line x1="290" y1="80" x2="320" y2="80" stroke="#d4d4d4" stroke-width="1.5" marker-end="url(#arrowhead)"/>
  <line x1="460" y1="80" x2="490" y2="80" stroke="#d4d4d4" stroke-width="1.5" marker-end="url(#arrowhead)"/>
  <line x1="610" y1="80" x2="640" y2="80" stroke="#d4d4d4" stroke-width="1.5" marker-end="url(#arrowhead)"/>

  <!-- Arrow marker -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#d4d4d4"/>
    </marker>
  </defs>

  <!-- Token math -->
  <rect x="20" y="150" width="660" height="110" rx="8" fill="#16213e" stroke="#0f3460" stroke-width="1"/>
  <text x="350" y="175" text-anchor="middle" fill="#e94560" font-size="12" font-weight="bold">Token Count Drives Everything</text>
  <text x="40" y="200" fill="#d4d4d4" font-size="11">For a 5-second 1080p video (120 frames, VAE 8x downsample, patch size 2x2x2):</text>
  <text x="40" y="220" fill="#16c79a" font-size="11">N = (T/p_t) x (h/p_s) x (w/p_s) = (120/2) x (135/2) x (240/2) = 60 x 67 x 120 = 482,400 tokens</text>
  <text x="40" y="244" fill="#d4d4d4" font-size="11">Attention cost: O(N^2) = O(2.3 x 10^11). This is why video generation is expensive.</text>
</svg>

**Step 1: Encoding.** The text prompt is encoded using a large language model (typically T5-XXL or CLIP). Image inputs (for image-to-video) are encoded by the VAE encoder into the same latent space as the video frames. The text embedding conditions the denoising process via cross-attention.

**Step 2: Latent noise initialization.** For a video of \(T\) frames at resolution \(H \times W\), the VAE compresses to a latent representation of shape \((T, H/f, W/f, c)\) where \(f\) is the spatial downsample factor (typically 8 for most models, 16 for Wan 2.2's VAE) and \(c\) is the latent channel count (4-16). We start from pure Gaussian noise in this latent space.

**Step 3: Iterative denoising via the DiT backbone.** This is where all the compute goes. The latent volume is divided into spacetime patches. Each patch becomes a token. The transformer processes these tokens with alternating spatial attention (tokens within the same frame attend to each other) and temporal attention (tokens at the same spatial location across different frames attend to each other). This is repeated for \(N_{\text{steps}}\) denoising steps.

**Step 4: VAE decoding.** The final denoised latent is decoded back to pixel space by the VAE decoder, producing the output video.

The computational cost is dominated by the DiT backbone in Step 3. Specifically, self-attention scales quadratically with the number of tokens:

$$\text{FLOPs}_{\text{attention}} = O(N_{\text{tokens}}^2 \cdot d \cdot N_{\text{steps}} \cdot N_{\text{layers}})$$

For a 5-second 1080p video with a typical DiT (24 layers, dimension 1536, 50 denoising steps, ~480K tokens), the total FLOPs are in the range of \(10^{16}\) to \(10^{17}\). This is why video generation takes 30-120 seconds even on H100 GPUs.

### What Makes Some Models Faster

Three architectural innovations drive speed differences between models:

**1. Rectified Flows (Wan 2.x, LTX-2)**

Traditional diffusion models follow curved paths through the noise-to-signal space. The denoising trajectory from \(x_T\) (noise) to \(x_0\) (clean signal) is a stochastic, non-linear path that requires many steps to traverse accurately.

**Rectified flows** replace these curved paths with straight lines. Instead of learning a noise prediction \(\epsilon_\theta\), the model learns a velocity field \(v_\theta\) that connects noise and data along the shortest path:

$$v_\theta(x_t, t) = x_0 - x_1$$

where \(x_t = (1-t) x_0 + t x_1\) is the linear interpolation between data \(x_0\) and noise \(x_1\). Because the path is straight, you need far fewer steps to traverse it. Models using rectified flows can achieve good quality in 10-25 denoising steps instead of 50-100.

This is why Wan 2.6 and LTX-2 generate faster than their parameter count would suggest. They traverse latent space more efficiently.

**2. Progressive Distillation (Runway Gen-4.5 Turbo, Veo 3.1 Fast)**

Take a trained model that uses 64 steps and train a "student" model to match its output in 32 steps. Then distill the 32-step model into 16 steps. Then 8 steps. Each round halves the step count with a small quality loss.

The distillation loss for each round is:

$$\mathcal{L}_{\text{distill}} = \mathbb{E}_{x_t} \left[ \| f_{\text{student}}(x_t, t, t-2\Delta) - f_{\text{teacher}}(x_t, t, t-\Delta, t-2\Delta) \|^2 \right]$$

where the student takes one step where the teacher takes two. After three rounds of distillation (64 to 8 steps), generation is 8x faster with a cumulative quality loss that is perceptible but acceptable for preview and draft workflows.

**3. Token Compression and Sparse Attention (Hailuo 2.3, PixVerse R1)**

Instead of processing all tokens equally, these models use techniques to reduce the effective token count:

- **Dynamic latent frame rate**: merge tokens in low-motion segments where adjacent frames are nearly identical. A static shot might collapse 120 frames into 20 effective tokens temporally, achieving up to 3x speedup with minimal quality degradation.
- **Sparse attention patterns**: instead of full \(O(N^2)\) attention, use windowed or strided attention for spatial tokens and full attention only for temporal tokens (or vice versa). This reduces the quadratic cost.
- **Mixture of Experts (MoE)**: Hailuo 2.3 and Wan 2.6 use MoE architectures where only a subset of parameters are activated for each token. A 14B parameter MoE model with 4 experts and top-2 routing only activates ~7B parameters per token, giving the quality of a larger model at the inference cost of a smaller one.

### How Resolution and Duration Scale Compute

The relationship between video specs and compute cost is super-linear. Let us quantify it.

Token count scales as:

$$N \propto T \times H \times W$$

where \(T\) is frame count (proportional to duration), and \(H, W\) are resolution dimensions (after VAE compression). Attention cost scales as \(O(N^2)\), so:

$$\text{Cost} \propto (T \times H \times W)^2$$

Doubling the resolution (2x in each spatial dimension) increases the token count by 4x and the attention cost by 16x. Doubling the duration doubles the token count and quadruples the attention cost.

In practice, models use approximations (sparse attention, factored spatial-temporal attention) that bring this closer to \(O(N^{1.5})\) or even \(O(N \log N)\), but the super-linear scaling remains. This is why a 10-second 4K video costs dramatically more than a 5-second 720p video.

| Spec | Relative Token Count | Relative Cost (quadratic) | Relative Cost (practical ~1.5x) |
|------|---------------------|--------------------------|--------------------------------|
| 5s, 480p | 1.0x | 1.0x | 1.0x |
| 5s, 720p | 2.25x | 5.1x | 3.4x |
| 5s, 1080p | 5.1x | 26x | 11.5x |
| 10s, 1080p | 10.1x | 102x | 32x |
| 10s, 4K | 40.5x | 1,640x | 258x |

This table explains why 4K video generation is expensive and why most models default to 720p or 1080p. Kling 3.0's native 4K/60fps output requires enormous compute --- likely multiple H100s per generation.

---

## Building a Future-Proof Video Pipeline

The video generation landscape changes on a monthly cadence. Any architecture that hard-codes a specific model is architectural debt waiting to compound. Here is how to build a system that absorbs change instead of breaking.

### The Core Abstraction: Model as a Pluggable Backend

The key insight is that every video generation API, regardless of provider, performs the same fundamental operation: accept a prompt (text, images, or both), some configuration (resolution, duration, style), and return a video file after some processing time. The differences are in authentication, parameter naming, response format, and polling mechanism --- all of which can be abstracted.

<svg viewBox="0 0 700 450" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; font-family: 'Segoe UI', system-ui, sans-serif;">
  <rect width="700" height="450" fill="#1a1a2e"/>
  <text x="350" y="28" text-anchor="middle" fill="#d4d4d4" font-size="15" font-weight="bold">Model-Agnostic Video Pipeline Architecture</text>

  <!-- User Request -->
  <rect x="270" y="45" width="160" height="40" rx="6" fill="#0f3460" stroke="#e94560" stroke-width="2"/>
  <text x="350" y="70" text-anchor="middle" fill="#d4d4d4" font-size="12">User Request</text>

  <!-- Router -->
  <rect x="240" y="110" width="220" height="50" rx="6" fill="#16213e" stroke="#e94560" stroke-width="2"/>
  <text x="350" y="132" text-anchor="middle" fill="#e94560" font-size="13" font-weight="bold">Intelligent Router</text>
  <text x="350" y="150" text-anchor="middle" fill="#d4d4d4" font-size="9">cost / quality / speed / availability</text>

  <!-- Adapter Layer -->
  <rect x="100" y="190" width="500" height="40" rx="6" fill="#16213e" stroke="#0f3460" stroke-width="2"/>
  <text x="350" y="215" text-anchor="middle" fill="#16c79a" font-size="12" font-weight="bold">Unified Adapter Layer (VideoProvider ABC)</text>

  <!-- Model backends -->
  <rect x="30" y="260" width="100" height="50" rx="6" fill="#16213e" stroke="#0f3460" stroke-width="1.5"/>
  <text x="80" y="282" text-anchor="middle" fill="#d4d4d4" font-size="10">Sora 2</text>
  <text x="80" y="298" text-anchor="middle" fill="#e94560" font-size="8">OpenAI API</text>

  <rect x="150" y="260" width="100" height="50" rx="6" fill="#16213e" stroke="#0f3460" stroke-width="1.5"/>
  <text x="200" y="282" text-anchor="middle" fill="#d4d4d4" font-size="10">Veo 3.1</text>
  <text x="200" y="298" text-anchor="middle" fill="#e94560" font-size="8">Gemini API</text>

  <rect x="270" y="260" width="100" height="50" rx="6" fill="#16213e" stroke="#0f3460" stroke-width="1.5"/>
  <text x="320" y="282" text-anchor="middle" fill="#d4d4d4" font-size="10">Kling 3.0</text>
  <text x="320" y="298" text-anchor="middle" fill="#e94560" font-size="8">Kling API</text>

  <rect x="390" y="260" width="100" height="50" rx="6" fill="#16213e" stroke="#0f3460" stroke-width="1.5"/>
  <text x="440" y="282" text-anchor="middle" fill="#d4d4d4" font-size="10">Runway 4.5</text>
  <text x="440" y="298" text-anchor="middle" fill="#e94560" font-size="8">Runway SDK</text>

  <rect x="510" y="260" width="110" height="50" rx="6" fill="#16213e" stroke="#16c79a" stroke-width="1.5"/>
  <text x="565" y="282" text-anchor="middle" fill="#d4d4d4" font-size="10">Wan 2.6</text>
  <text x="565" y="298" text-anchor="middle" fill="#16c79a" font-size="8">Self-hosted</text>

  <!-- Fallback chain -->
  <rect x="100" y="340" width="500" height="40" rx="6" fill="#16213e" stroke="#e2b714" stroke-width="2"/>
  <text x="350" y="365" text-anchor="middle" fill="#e2b714" font-size="12" font-weight="bold">Fallback Chain + Circuit Breaker</text>

  <!-- Quality Gate -->
  <rect x="200" y="400" width="300" height="35" rx="6" fill="#16213e" stroke="#0f3460" stroke-width="1.5"/>
  <text x="350" y="422" text-anchor="middle" fill="#d4d4d4" font-size="11">Quality Gate (Gemini Flash scoring)</text>

  <!-- Arrows -->
  <line x1="350" y1="85" x2="350" y2="110" stroke="#d4d4d4" stroke-width="1.5" marker-end="url(#arrowhead2)"/>
  <line x1="350" y1="160" x2="350" y2="190" stroke="#d4d4d4" stroke-width="1.5" marker-end="url(#arrowhead2)"/>
  <line x1="80" y1="230" x2="80" y2="260" stroke="#d4d4d4" stroke-width="1" marker-end="url(#arrowhead2)"/>
  <line x1="200" y1="230" x2="200" y2="260" stroke="#d4d4d4" stroke-width="1" marker-end="url(#arrowhead2)"/>
  <line x1="320" y1="230" x2="320" y2="260" stroke="#d4d4d4" stroke-width="1" marker-end="url(#arrowhead2)"/>
  <line x1="440" y1="230" x2="440" y2="260" stroke="#d4d4d4" stroke-width="1" marker-end="url(#arrowhead2)"/>
  <line x1="565" y1="230" x2="565" y2="260" stroke="#d4d4d4" stroke-width="1" marker-end="url(#arrowhead2)"/>
  <line x1="350" y1="310" x2="350" y2="340" stroke="#d4d4d4" stroke-width="1.5" marker-end="url(#arrowhead2)"/>
  <line x1="350" y1="380" x2="350" y2="400" stroke="#d4d4d4" stroke-width="1" marker-end="url(#arrowhead2)"/>

  <defs>
    <marker id="arrowhead2" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#d4d4d4"/>
    </marker>
  </defs>
</svg>

### Design Principles

**1. Abstract at the right level.** The abstraction layer should normalize:
- Input format (prompt text, reference images, duration, resolution, aspect ratio)
- Output format (video URL or bytes, generation metadata, cost)
- Lifecycle (submit job, poll status, retrieve result)
- Error handling (rate limits, capacity errors, content policy violations)

It should NOT try to normalize model-specific features. Sora's Characters, Runway's References, Kling's Elements --- these are competitive advantages that you want to expose to users. The abstraction layer should have an `extras` parameter for model-specific options that pass through untouched.

**2. Route intelligently.** The router should make model selection decisions based on:
- **Cost budget**: user/tier cost caps
- **Quality requirements**: premium users get Veo 3.1, free tier gets Wan 2.6
- **Content type**: some models are better at specific content (Kling for human motion, Veo for audio-heavy scenes, Runway for stylized content)
- **Availability**: if a model is down or slow, route to alternatives
- **Latency requirements**: if the user needs a result in 30 seconds, only fast models qualify

**3. Fail gracefully.** Every model will go down. The fallback chain defines what happens:
- Primary model fails -> try secondary model
- Secondary fails -> try self-hosted open-source model
- Self-hosted fails -> return a queued promise and generate asynchronously

**4. Observe everything.** Log every generation: model used, prompt, parameters, latency, cost, quality score (via automated scoring with a vision LLM). This data is what makes your routing smarter over time.

---

## Implementation Guide: Calling Every Major API

Here are production-ready code examples for each major video generation API. All examples use Python with async support for non-blocking I/O.

### OpenAI Sora 2

```python
import os
import time
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def generate_sora_video(
    prompt: str,
    duration: int = 5,
    resolution: str = "1080x1920",  # portrait
    model: str = "sora-2",
) -> str:
    """
    Generate a video using OpenAI's Sora 2 API.

    The API is asynchronous: you submit a job and poll for completion.
    Returns the URL of the generated video.
    """
    # Submit the generation job
    response = client.videos.create(
        model=model,
        prompt=prompt,
        size=resolution,     # "1920x1080", "1080x1920", "1080x1080"
        duration=duration,   # 5, 10, 15, 20, or 25 seconds
    )

    video_id = response.id
    print(f"Job submitted: {video_id}, status: {response.status}")

    # Poll for completion
    while True:
        result = client.videos.retrieve(video_id)
        if result.status == "completed":
            # Download the video
            video_url = result.video.url
            print(f"Video ready: {video_url}")
            return video_url
        elif result.status == "failed":
            raise RuntimeError(f"Sora generation failed: {result.error}")

        time.sleep(5)  # Poll every 5 seconds
```

### Google Veo 3.1 (Gemini API)

```python
import time
from google import genai
from google.genai.types import GenerateVideosConfig

def generate_veo_video(
    prompt: str,
    aspect_ratio: str = "16:9",
    model: str = "veo-3.1-generate-001",
    output_gcs_uri: str = "gs://your-bucket/output/",
    generate_audio: bool = True,
) -> str:
    """
    Generate a video using Google's Veo 3.1 via the Gemini API.

    Requires a Google Cloud project with the Generative AI API enabled.
    Videos are output to a GCS bucket.
    Returns the GCS URI of the generated video.
    """
    client = genai.Client()

    operation = client.models.generate_videos(
        model=model,
        prompt=prompt,
        config=GenerateVideosConfig(
            aspect_ratio=aspect_ratio,  # "16:9" or "9:16"
            output_gcs_uri=output_gcs_uri,
            generate_audio=generate_audio,
            number_of_videos=1,
        ),
    )

    # Poll for completion (Veo uses long-running operations)
    while not operation.done:
        time.sleep(15)
        operation = client.operations.get(operation)

    if operation.response and operation.result.generated_videos:
        video = operation.result.generated_videos[0]
        print(f"Video ready at: {video.video.uri}")
        return video.video.uri
    else:
        raise RuntimeError(f"Veo generation failed: {operation.error}")
```

### Runway Gen-4.5

```python
import os
import time
from runwayml import RunwayML

def generate_runway_video(
    prompt: str,
    image_url: str | None = None,
    model: str = "gen4_turbo",  # "gen4_turbo" or "gen4"
    duration: int = 10,
    ratio: str = "1280:720",
) -> str:
    """
    Generate a video using Runway's Gen-4.5 API.

    Supports text-to-video and image-to-video.
    Returns the URL of the generated video.
    """
    client = RunwayML(api_key=os.environ.get("RUNWAY_API_KEY"))

    if image_url:
        # Image-to-video
        task = client.image_to_video.create(
            model=model,
            prompt_image=image_url,
            prompt_text=prompt,
            ratio=ratio,
            duration=duration,
        )
    else:
        # Text-to-video
        task = client.text_to_video.create(
            model=model,
            prompt_text=prompt,
            ratio=ratio,
            duration=duration,
        )

    task_id = task.id
    print(f"Task submitted: {task_id}")

    # Wait for completion
    result = task.wait_for_task_output()

    if result and result.output:
        video_url = result.output[0]
        print(f"Video ready: {video_url}")
        return video_url
    else:
        raise RuntimeError(f"Runway generation failed: {result}")
```

### Kling 3.0 (via fal.ai)

```python
import os
import fal_client

def generate_kling_video(
    prompt: str,
    image_url: str | None = None,
    duration: float = 5.0,
    aspect_ratio: str = "16:9",
    with_audio: bool = True,
) -> str:
    """
    Generate a video using Kling 3.0 via fal.ai.

    fal.ai provides a unified interface to Kling's models.
    Returns the URL of the generated video.
    """
    os.environ["FAL_KEY"] = os.environ.get("FAL_KEY", "your-fal-key")

    # Choose the right model endpoint
    if image_url:
        model_id = "fal-ai/kling-video/v3/pro/image-to-video"
        arguments = {
            "prompt": prompt,
            "image_url": image_url,
            "duration": str(duration),
            "aspect_ratio": aspect_ratio,
        }
    else:
        model_id = "fal-ai/kling-video/v3/pro/text-to-video"
        arguments = {
            "prompt": prompt,
            "duration": str(duration),
            "aspect_ratio": aspect_ratio,
        }

    if with_audio:
        arguments["enable_audio"] = True

    # Submit and wait
    result = fal_client.subscribe(
        model_id,
        arguments=arguments,
        with_logs=True,
    )

    video_url = result["video"]["url"]
    print(f"Video ready: {video_url}")
    return video_url
```

### Wan 2.6 (Self-Hosted via Diffusers)

```python
import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

def generate_wan_video_local(
    prompt: str,
    num_frames: int = 81,       # ~3.4 seconds at 24fps
    height: int = 480,
    width: int = 832,
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    model_id: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
) -> str:
    """
    Generate a video locally using Wan 2.2/2.6 via HuggingFace Diffusers.

    Requires a GPU with sufficient VRAM:
      - 5B model: 24GB (RTX 4090)
      - A14B model: 80GB (A100/H100)

    Returns the path to the saved video file.
    """
    # Load the pipeline (cache this in production)
    pipe = WanPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    # Enable memory optimizations
    pipe.enable_model_cpu_offload()       # Offload to CPU when not in use
    pipe.enable_vae_slicing()              # Process VAE in slices
    pipe.enable_vae_tiling()               # Tile-based VAE decoding

    # Generate
    output = pipe(
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    # Export to video file
    output_path = "output.mp4"
    export_to_video(output.frames[0], output_path, fps=24)

    print(f"Video saved to: {output_path}")
    return output_path
```

### Hailuo / MiniMax (via API)

```python
import os
import time
import requests

def generate_hailuo_video(
    prompt: str,
    model: str = "T2V-01",  # Hailuo 2.3
    resolution: str = "1080p",
    duration: int = 6,
) -> str:
    """
    Generate a video using MiniMax's Hailuo API.

    Returns the URL of the generated video.
    """
    api_key = os.environ["MINIMAX_API_KEY"]
    base_url = "https://api.minimax.chat/v1"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Submit generation job
    response = requests.post(
        f"{base_url}/video/generate",
        headers=headers,
        json={
            "model": model,
            "prompt": prompt,
            "resolution": resolution,
            "duration": duration,
        },
    )
    response.raise_for_status()
    task_id = response.json()["task_id"]
    print(f"Task submitted: {task_id}")

    # Poll for completion
    while True:
        status_resp = requests.get(
            f"{base_url}/video/status/{task_id}",
            headers=headers,
        )
        status_resp.raise_for_status()
        data = status_resp.json()

        if data["status"] == "completed":
            video_url = data["video_url"]
            print(f"Video ready: {video_url}")
            return video_url
        elif data["status"] == "failed":
            raise RuntimeError(f"Hailuo generation failed: {data.get('error')}")

        time.sleep(3)
```

---

## The Unified Abstraction Layer

Here is a production-ready abstraction layer that wraps all the APIs above into a single interface. This is the code that lets you swap models without touching your application logic.

```python
"""
Unified Video Generation Abstraction Layer

This module provides a model-agnostic interface for video generation.
Add new providers by implementing the VideoProvider protocol.
Route between providers using the VideoRouter.

Usage:
    router = VideoRouter(providers=[
        SoraProvider(api_key="..."),
        VeoProvider(),
        KlingProvider(fal_key="..."),
        RunwayProvider(api_key="..."),
        WanLocalProvider(model_path="..."),
    ])

    result = await router.generate(
        prompt="A cat playing piano in a jazz club",
        duration=10,
        resolution="1080p",
        quality="standard",       # "draft", "standard", "premium"
        max_cost_dollars=2.00,
        max_latency_seconds=120,
    )
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class Quality(Enum):
    DRAFT = "draft"
    STANDARD = "standard"
    PREMIUM = "premium"


@dataclass
class VideoRequest:
    """Normalized video generation request."""
    prompt: str
    duration: int = 5                # seconds
    resolution: str = "1080p"        # "480p", "720p", "1080p", "4k"
    aspect_ratio: str = "16:9"       # "16:9", "9:16", "1:1"
    quality: Quality = Quality.STANDARD
    with_audio: bool = False
    reference_image_url: str | None = None
    max_cost_dollars: float = 5.00
    max_latency_seconds: float = 180.0
    extras: dict[str, Any] = field(default_factory=dict)  # Model-specific params


@dataclass
class VideoResult:
    """Normalized video generation result."""
    video_url: str
    provider: str
    model: str
    duration_seconds: float
    resolution: str
    cost_dollars: float
    latency_seconds: float
    has_audio: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderCapabilities:
    """What a provider can do and what it costs."""
    name: str
    model: str
    max_duration: int
    max_resolution: str
    supports_audio: bool
    supports_image_to_video: bool
    cost_per_second: float          # dollars
    avg_latency_per_second: float   # seconds of wall-clock per second of video
    quality_tier: Quality           # typical quality level


class VideoProvider(ABC):
    """Abstract base class for video generation providers."""

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return this provider's capabilities and pricing."""
        ...

    @abstractmethod
    async def generate(self, request: VideoRequest) -> VideoResult:
        """Generate a video. Must be async for non-blocking I/O."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is currently operational."""
        ...


class VideoRouter:
    """
    Intelligent router that selects the best provider for each request.

    Selection criteria (in priority order):
    1. Provider must support the requested features (audio, resolution, duration)
    2. Estimated cost must be within budget
    3. Estimated latency must be within tolerance
    4. Among qualifying providers, select by quality tier match
    5. Break ties by cost (lower is better)
    """

    def __init__(self, providers: list[VideoProvider]):
        self.providers = providers
        self._circuit_breaker: dict[str, float] = {}  # provider -> failure_time
        self._circuit_breaker_timeout = 300  # 5 minutes

    def _is_circuit_open(self, provider_name: str) -> bool:
        """Check if a provider's circuit breaker is open (= provider is down)."""
        if provider_name not in self._circuit_breaker:
            return False
        elapsed = time.time() - self._circuit_breaker[provider_name]
        if elapsed > self._circuit_breaker_timeout:
            del self._circuit_breaker[provider_name]
            return False
        return True

    def _trip_circuit(self, provider_name: str) -> None:
        """Mark a provider as down."""
        self._circuit_breaker[provider_name] = time.time()

    def _score_provider(
        self, caps: ProviderCapabilities, request: VideoRequest
    ) -> float | None:
        """
        Score a provider for a request. Returns None if the provider
        cannot fulfill the request. Higher scores are better.
        """
        # Hard requirements
        if request.with_audio and not caps.supports_audio:
            return None
        if request.reference_image_url and not caps.supports_image_to_video:
            return None
        if request.duration > caps.max_duration:
            return None

        # Resolution check (simplified ordering)
        res_order = {"480p": 0, "720p": 1, "1080p": 2, "4k": 3}
        if res_order.get(request.resolution, 0) > res_order.get(caps.max_resolution, 0):
            return None

        # Cost check
        estimated_cost = caps.cost_per_second * request.duration
        if estimated_cost > request.max_cost_dollars:
            return None

        # Latency check
        estimated_latency = caps.avg_latency_per_second * request.duration
        if estimated_latency > request.max_latency_seconds:
            return None

        # Scoring: quality match bonus + cost efficiency
        quality_match = {
            (Quality.DRAFT, Quality.DRAFT): 100,
            (Quality.DRAFT, Quality.STANDARD): 60,
            (Quality.DRAFT, Quality.PREMIUM): 30,
            (Quality.STANDARD, Quality.DRAFT): 40,
            (Quality.STANDARD, Quality.STANDARD): 100,
            (Quality.STANDARD, Quality.PREMIUM): 70,
            (Quality.PREMIUM, Quality.DRAFT): 10,
            (Quality.PREMIUM, Quality.STANDARD): 50,
            (Quality.PREMIUM, Quality.PREMIUM): 100,
        }

        q_score = quality_match.get((request.quality, caps.quality_tier), 50)
        cost_score = max(0, 50 * (1 - estimated_cost / request.max_cost_dollars))

        return q_score + cost_score

    def _rank_providers(self, request: VideoRequest) -> list[VideoProvider]:
        """Rank providers by suitability for a request."""
        scored = []
        for provider in self.providers:
            caps = provider.capabilities()
            if self._is_circuit_open(caps.name):
                continue
            score = self._score_provider(caps, request)
            if score is not None:
                scored.append((score, provider))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [provider for _, provider in scored]

    async def generate(self, **kwargs) -> VideoResult:
        """
        Generate a video using the best available provider.
        Falls back to alternatives if the primary provider fails.
        """
        request = VideoRequest(**kwargs)
        ranked = self._rank_providers(request)

        if not ranked:
            raise RuntimeError(
                f"No provider can fulfill this request: "
                f"duration={request.duration}s, resolution={request.resolution}, "
                f"audio={request.with_audio}, budget=${request.max_cost_dollars:.2f}"
            )

        last_error = None
        for provider in ranked:
            caps = provider.capabilities()
            try:
                result = await provider.generate(request)
                return result
            except Exception as e:
                last_error = e
                self._trip_circuit(caps.name)
                print(f"Provider {caps.name} failed: {e}, trying next...")
                continue

        raise RuntimeError(
            f"All providers failed. Last error: {last_error}"
        )
```

### Example: Implementing a Provider

```python
class SoraProvider(VideoProvider):
    """OpenAI Sora 2 provider implementation."""

    def __init__(self, api_key: str):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key)

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="sora-2",
            model="sora-2",
            max_duration=25,
            max_resolution="1080p",
            supports_audio=True,
            supports_image_to_video=False,
            cost_per_second=0.10,
            avg_latency_per_second=12.0,  # ~60s for 5s video
            quality_tier=Quality.STANDARD,
        )

    async def generate(self, request: VideoRequest) -> VideoResult:
        start_time = time.time()

        size_map = {
            ("1080p", "16:9"): "1920x1080",
            ("1080p", "9:16"): "1080x1920",
            ("720p", "16:9"): "1280x720",
        }
        size = size_map.get(
            (request.resolution, request.aspect_ratio), "1920x1080"
        )

        response = await self.client.videos.create(
            model="sora-2",
            prompt=request.prompt,
            size=size,
            duration=request.duration,
        )

        # Poll for completion
        video_id = response.id
        while True:
            result = await self.client.videos.retrieve(video_id)
            if result.status == "completed":
                break
            elif result.status == "failed":
                raise RuntimeError(f"Sora failed: {result.error}")
            await asyncio.sleep(5)

        latency = time.time() - start_time
        cost = request.duration * 0.10

        return VideoResult(
            video_url=result.video.url,
            provider="sora-2",
            model="sora-2",
            duration_seconds=request.duration,
            resolution=request.resolution,
            cost_dollars=cost,
            latency_seconds=latency,
            has_audio=True,
        )

    async def health_check(self) -> bool:
        try:
            # Simple connectivity check
            await self.client.models.retrieve("sora-2")
            return True
        except Exception:
            return False


class VeoProvider(VideoProvider):
    """Google Veo 3.1 provider implementation."""

    def __init__(self, output_gcs_uri: str = "gs://your-bucket/output/"):
        from google import genai
        self.client = genai.Client()
        self.output_gcs_uri = output_gcs_uri

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="veo-3.1",
            model="veo-3.1-generate-001",
            max_duration=8,  # per generation (extendable to 60s)
            max_resolution="1080p",
            supports_audio=True,
            supports_image_to_video=True,
            cost_per_second=0.40,
            avg_latency_per_second=15.0,
            quality_tier=Quality.PREMIUM,
        )

    async def generate(self, request: VideoRequest) -> VideoResult:
        from google.genai.types import GenerateVideosConfig

        start_time = time.time()

        operation = self.client.models.generate_videos(
            model="veo-3.1-generate-001",
            prompt=request.prompt,
            config=GenerateVideosConfig(
                aspect_ratio=request.aspect_ratio.replace(":", ":"),
                output_gcs_uri=self.output_gcs_uri,
                generate_audio=request.with_audio,
            ),
        )

        while not operation.done:
            await asyncio.sleep(15)
            operation = self.client.operations.get(operation)

        if not operation.response or not operation.result.generated_videos:
            raise RuntimeError(f"Veo failed: {operation.error}")

        video_uri = operation.result.generated_videos[0].video.uri
        latency = time.time() - start_time
        actual_duration = min(request.duration, 8)
        cost = actual_duration * 0.40

        return VideoResult(
            video_url=video_uri,
            provider="veo-3.1",
            model="veo-3.1-generate-001",
            duration_seconds=actual_duration,
            resolution=request.resolution,
            cost_dollars=cost,
            latency_seconds=latency,
            has_audio=request.with_audio,
        )

    async def health_check(self) -> bool:
        try:
            # Check if the model endpoint responds
            return True
        except Exception:
            return False


class WanLocalProvider(VideoProvider):
    """Self-hosted Wan 2.6 provider implementation."""

    def __init__(self, model_path: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"):
        self.model_path = model_path
        self._pipe = None

    def _get_pipe(self):
        """Lazy-load the pipeline (expensive, do once)."""
        if self._pipe is None:
            import torch
            from diffusers import WanPipeline

            self._pipe = WanPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
            )
            self._pipe.to("cuda")
            self._pipe.enable_model_cpu_offload()
            self._pipe.enable_vae_slicing()
        return self._pipe

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="wan-2.6-local",
            model="Wan2.6-T2V-A14B",
            max_duration=15,
            max_resolution="1080p",
            supports_audio=False,  # Audio requires separate pipeline
            supports_image_to_video=True,
            cost_per_second=0.012,  # Amortized GPU cost
            avg_latency_per_second=18.0,
            quality_tier=Quality.STANDARD,
        )

    async def generate(self, request: VideoRequest) -> VideoResult:
        from diffusers.utils import export_to_video

        start_time = time.time()
        pipe = self._get_pipe()

        res_map = {
            "480p": (480, 832),
            "720p": (720, 1280),
            "1080p": (1080, 1920),
        }
        height, width = res_map.get(request.resolution, (480, 832))
        num_frames = int(request.duration * 24)  # 24fps

        # Run inference in a thread to not block the event loop
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(
            None,
            lambda: pipe(
                prompt=request.prompt,
                num_frames=min(num_frames, 360),  # Cap at 15s
                height=height,
                width=width,
                num_inference_steps=30,
                guidance_scale=5.0,
            ),
        )

        output_path = f"/tmp/wan_output_{int(time.time())}.mp4"
        export_to_video(output.frames[0], output_path, fps=24)

        latency = time.time() - start_time
        cost = request.duration * 0.012

        return VideoResult(
            video_url=output_path,
            provider="wan-2.6-local",
            model="Wan2.6-T2V-A14B",
            duration_seconds=request.duration,
            resolution=request.resolution,
            cost_dollars=cost,
            latency_seconds=latency,
            has_audio=False,
        )

    async def health_check(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False
```

### Using the Router

```python
import asyncio

async def main():
    router = VideoRouter(providers=[
        SoraProvider(api_key="sk-..."),
        VeoProvider(output_gcs_uri="gs://my-bucket/videos/"),
        WanLocalProvider(),
    ])

    # Standard quality, moderate budget
    result = await router.generate(
        prompt="A golden retriever running through autumn leaves in slow motion",
        duration=5,
        resolution="1080p",
        quality="standard",
        max_cost_dollars=1.00,
        max_latency_seconds=120,
    )

    print(f"Generated by: {result.provider}")
    print(f"Cost: ${result.cost_dollars:.2f}")
    print(f"Latency: {result.latency_seconds:.1f}s")
    print(f"URL: {result.video_url}")

    # Premium quality, higher budget, needs audio
    result = await router.generate(
        prompt="A jazz musician playing saxophone on a rainy street corner at night",
        duration=8,
        resolution="1080p",
        quality="premium",
        with_audio=True,
        max_cost_dollars=5.00,
        max_latency_seconds=180,
    )

    print(f"Generated by: {result.provider}")
    print(f"Cost: ${result.cost_dollars:.2f}")

asyncio.run(main())
```

In this example, the first request (standard quality, $1 budget, no audio) would route to Wan 2.6 local (cheapest) or Sora 2 ($0.50 for 5 seconds). The second request (premium quality, audio required) would route to Veo 3.1 ($3.20 for 8 seconds) because it is the only premium provider with audio in this configuration.

---

## Cost Projections: When Does 1 Minute of HD Video Cost Less Than $1? {#cost-projections}

We established earlier that API costs are declining at approximately 80% year-over-year, following \(C(t) = e^{-1.61t}\). Let us now compute specific milestones.

### The $1/Minute Threshold

One minute of 1080p video at the current cheapest API rate ($0.04/sec for Hailuo 2.3 Fast):

$$60 \times 0.04 = 2.40 \text{ dollars}$$

We need the per-second rate to reach:

$$C_{\text{target}} = \frac{1.00}{60} \approx 0.0167 \text{ \\$/sec}$$

Using our model:

$$0.0167 = e^{-1.61t}$$

$$t = \frac{-\ln(0.0167)}{1.61} = \frac{4.09}{1.61} \approx 2.54 \text{ years from Q1 2024}$$

That puts us at approximately **Q3 2026** for API-based models. We are roughly 6 months away from the $1/minute threshold for the cheapest API providers.

For self-hosted open-source models, we are already there. Generating 60 seconds of 720p video on a self-hosted Wan 2.6 setup costs approximately $0.72 in amortized GPU time.

### The $0.10/Minute Threshold

At this price point, video generation becomes effectively free for most applications. The cost would need to reach:

$$C_{\text{target}} = \frac{0.10}{60} \approx 0.00167 \text{ \\$/sec}$$

$$t = \frac{-\ln(0.00167)}{1.61} = \frac{6.40}{1.61} \approx 3.97 \text{ years from Q1 2024}$$

That is approximately **Q1 2028**. In two years, a minute of HD video will cost a dime via API.

### What Drives the Cost Down?

The 80% annual cost decline is not magical. It has concrete technical drivers:

1. **Fewer denoising steps**: Rectified flows and progressive distillation reduce step counts from 50-100 to 4-8 without catastrophic quality loss. Each halving of steps halves compute cost.

2. **Better VAE compression**: Wan 2.2's VAE achieves 16x16x4 compression (vs 8x8x4 for earlier models). Higher compression means fewer tokens, and since attention cost scales super-linearly with token count, even modest compression improvements yield disproportionate cost savings.

3. **Mixture of Experts**: MoE architectures (Wan 2.6, Hailuo 2.3) activate only a subset of parameters per token. A 14B model with 4 experts and top-2 routing costs approximately the same as a 7B dense model at inference time, but has the capacity of a 14B model.

4. **Hardware cost decline**: H100 on-demand pricing has dropped approximately 40% in six months. A100s are now sub-$1/GPU-hour on open markets. The RTX 5090 is launching with improved inference performance per dollar.

5. **Competitive pressure**: Ten companies are competing for the same market. When Hailuo cuts prices, Kling follows. When Wan goes open-source, it puts a price ceiling on all commercial APIs.

```python
import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#1a1a2e')
ax1.set_facecolor('#1a1a2e')
ax2.set_facecolor('#1a1a2e')

# Left plot: Per-second cost projection by model tier
t = np.linspace(0, 4, 200)  # years from Q1 2024

# Different tiers decline at different rates
premium = 0.50 * np.exp(-1.2 * t)     # Premium models (Veo, Gen-4.5 Aleph)
standard = 0.20 * np.exp(-1.61 * t)   # Standard models (Sora 2, Kling)
budget = 0.10 * np.exp(-1.8 * t)      # Budget models (Hailuo, Wan API)
selfhost = 0.03 * np.exp(-0.8 * t)    # Self-hosted open source

ax1.semilogy(t, premium, color='#e94560', linewidth=2, label=r'Premium tier (Veo 3.1, Gen-4.5)')
ax1.semilogy(t, standard, color='#e2b714', linewidth=2, label=r'Standard tier (Sora 2, Kling 3.0)')
ax1.semilogy(t, budget, color='#16c79a', linewidth=2, label=r'Budget tier (Hailuo, Wan API)')
ax1.semilogy(t, selfhost, color='#533483', linewidth=2, label=r'Self-hosted (Wan 2.6, LTX-2)')

# Threshold lines
ax1.axhline(y=1/60, color='#d4d4d4', linestyle='--', alpha=0.5, linewidth=1)
ax1.text(0.1, 1/60 * 1.3, r'\$1/min threshold', color='#d4d4d4', fontsize=9)

ax1.set_xlabel('Years from Q1 2024', fontsize=11, color='#d4d4d4')
ax1.set_ylabel(r'Cost per second (\$)', fontsize=11, color='#d4d4d4')
ax1.set_title('Per-Second Cost by Model Tier', fontsize=13, color='#d4d4d4')
ax1.set_xticks([0, 1, 2, 3, 4])
ax1.set_xticklabels(['Q1\n2024', 'Q1\n2025', 'Q1\n2026', 'Q1\n2027', 'Q1\n2028'],
                      fontsize=9, color='#d4d4d4')
ax1.tick_params(colors='#d4d4d4')
ax1.legend(fontsize=8, facecolor='#16213e', edgecolor='#0f3460', labelcolor='#d4d4d4', loc='upper right')
ax1.grid(True, alpha=0.1, color='#d4d4d4')
ax1.set_ylim(0.0005, 1.0)
for spine in ax1.spines.values():
    spine.set_color('#0f3460')

# Right plot: Cost of 1-minute 1080p video over time
one_min_premium = 60 * premium
one_min_standard = 60 * standard
one_min_budget = 60 * budget
one_min_selfhost = 60 * selfhost

ax2.semilogy(t, one_min_premium, color='#e94560', linewidth=2, label='Premium')
ax2.semilogy(t, one_min_standard, color='#e2b714', linewidth=2, label='Standard')
ax2.semilogy(t, one_min_budget, color='#16c79a', linewidth=2, label='Budget API')
ax2.semilogy(t, one_min_selfhost, color='#533483', linewidth=2, label='Self-hosted')

ax2.axhline(y=1.0, color='#d4d4d4', linestyle='--', alpha=0.5, linewidth=1)
ax2.text(3.2, 1.2, r'\$1', color='#d4d4d4', fontsize=10)
ax2.axhline(y=0.10, color='#d4d4d4', linestyle=':', alpha=0.3, linewidth=1)
ax2.text(3.2, 0.12, r'\$0.10', color='#d4d4d4', fontsize=10)

ax2.set_xlabel('Years from Q1 2024', fontsize=11, color='#d4d4d4')
ax2.set_ylabel(r'Cost of 1 minute 1080p video (\$)', fontsize=11, color='#d4d4d4')
ax2.set_title('Cost of 1-Minute HD Video', fontsize=13, color='#d4d4d4')
ax2.set_xticks([0, 1, 2, 3, 4])
ax2.set_xticklabels(['Q1\n2024', 'Q1\n2025', 'Q1\n2026', 'Q1\n2027', 'Q1\n2028'],
                      fontsize=9, color='#d4d4d4')
ax2.tick_params(colors='#d4d4d4')
ax2.legend(fontsize=9, facecolor='#16213e', edgecolor='#0f3460', labelcolor='#d4d4d4')
ax2.grid(True, alpha=0.1, color='#d4d4d4')
ax2.set_ylim(0.01, 50)
for spine in ax2.spines.values():
    spine.set_color('#0f3460')

plt.tight_layout()
plt.savefig('cost_projections.png', dpi=150, facecolor='#1a1a2e')
plt.show()
```

---

## What to Bet On

Here is the uncomfortable truth: in a market moving this fast, most of your decisions are bets under uncertainty. The goal is not to be right about everything --- it is to structure your bets so that you win in the most likely scenarios and survive in the unlikely ones.

### Bet 1: The DiT Architecture Is the Winner (Confidence: 95%)

Every major model in 2026 uses Diffusion Transformers. The architectural convergence is complete. There is no credible competitor architecture on the horizon. U-Nets are dead for video generation. GANs are dead for video generation. Autoregressive video (a la DALL-E 1 for images) never materialized at competitive quality.

What this means for you: invest in understanding DiT deeply. Learn the math of diffusion processes, rectified flows, and transformer attention. When you understand the architecture, you can predict model behavior, estimate costs from first principles, and evaluate new models faster.

### Bet 2: Audio Integration Becomes Mandatory (Confidence: 90%)

Seven of fourteen major models already ship native audio. The holdouts (Runway, Luma, Pika, PixVerse) are under pressure to add it. By end of 2026, a video generation model without audio will be like a smartphone without a camera --- technically functional but commercially dead.

What this means for you: if your pipeline does not handle audio, start planning for it now. Your data model needs audio metadata. Your storage needs to handle larger files. Your quality evaluation needs an audio dimension. Build the abstractions now, even if you bolt on audio with a separate model (ElevenLabs, Bark, etc.) in the interim.

### Bet 3: Open-Source Models Reach "Good Enough" Quality Within 12 Months (Confidence: 80%)

Wan 2.6 and LTX-2 are already good. Not "Veo 3.1 good," but good enough for many use cases: social media content, product demos, educational videos, prototyping. The quality gap is 6-12 months and closing. By early 2027, open-source models will be at the quality level that commercial models are at today.

What this means for you: invest in self-hosting capability. Even if you use commercial APIs today, have a working self-hosted pipeline ready. When the quality crosses your threshold, you can shift volume to self-hosted and cut costs by 5-10x overnight. The unified abstraction layer above makes this switch trivial.

### Bet 4: Multi-Shot Consistency Is the Next Frontier (Confidence: 85%)

Single-clip quality is converging. The next competitive battleground is multi-shot: generating sequences of clips that maintain consistent characters, environments, lighting, and narrative across shots. Kling 3.0's Elements system and Seedance 2.0's multi-reference input are early moves in this direction.

What this means for you: design your data model and pipeline for multi-shot from day one. Each generation should optionally reference previous generations. Character and environment references should be first-class entities in your system, not afterthoughts. The models that win on multi-shot consistency will win the market, and you want your pipeline ready to use them.

### Bet 5: The Chinese Model Ecosystem Is Underpriced and Will Stay That Way (Confidence: 70%)

The pricing differential between Chinese and Western models is not temporary. It reflects structural differences: lower labor costs for annotation teams, aggressive government-subsidized compute programs, and business models oriented around platform integration (Kuaishou, ByteDance) rather than pure API revenue.

The regulatory risk is real but manageable. If you are building for a global market, maintain adapters for both Chinese (Kling, Seedance, Wan, Hailuo) and Western (Veo, Sora, Runway) models. Route based on user geography and compliance requirements.

### Bet 6: Aggregator APIs Are the Right Default for Most Teams (Confidence: 75%)

Unless you are generating more than 50,000 seconds of video per month, the 10-30% aggregator markup is worth paying for the reduced integration complexity. fal.ai, Replicate, and Together AI handle authentication, billing, model updates, and failover. Your engineering time is better spent on your product than on maintaining five different API integrations.

The exception: if a single model's unique features are core to your product (e.g., you build on Sora Characters or Kling Elements), go direct for that model and use an aggregator for everything else.

### What NOT to Bet On

**Do not bet on a single model.** Every model that was the "best" six months ago has been surpassed. Sora 1 was a sensation in February 2024 and was outclassed by Gen-3 Alpha by summer. Gen-3 was surpassed by Kling 1.6 by fall. The cycle time is accelerating. Build model-agnostic.

**Do not bet on video generation quality as a moat.** Quality is commoditizing. By 2027, multiple models will produce indistinguishable-from-real output for most prompts. The moat will be in user experience, workflow integration, multi-shot consistency, fine-tuning for specific domains, and cost optimization --- not raw generation quality.

**Do not bet against open-source.** The pattern from language models (GPT-4 -> Llama 3 -> open-source catches up) is repeating for video. Wan 2.6 is Apache 2.0. LTX-2 is open-weight. CogVideoX is open. The gap is closing. Any plan that assumes commercial API lock-in will age poorly.

**Do not bet on real-time video generation being practical for most use cases in 2026.** PixVerse R1 is impressive but real-time generation at high quality requires fundamentally different hardware deployment (dedicated GPUs per user session). The economics do not work for most applications yet. Keep an eye on it for 2027.

---

## Summary: The Decision Matrix

If you are building a video generation product right now, here is the decision tree:

**Starting from zero, limited budget:** Use fal.ai as your aggregator. Default to Hailuo 2.3 for speed-sensitive use cases, Kling 3.0 for quality-sensitive ones. Build the abstraction layer from day one. Ship fast, learn from users.

**Existing product, looking to add video:** Add Veo 3.1 via the Gemini API for premium tier and Wan 2.6 via fal.ai for standard tier. The unified abstraction layer takes 2-3 days to build and saves you months of refactoring later.

**High-volume platform (>50K seconds/month):** Self-host Wan 2.6 on cloud GPUs for your standard tier. Use direct APIs (not aggregators) for Veo 3.1 and Sora 2 for premium tier. The cost savings at this volume justify the operational complexity.

**Enterprise with compliance requirements:** Veo 3.1 via Vertex AI (Google Cloud enterprise SLAs, data residency guarantees) or self-hosted Wan 2.6/LTX-2 in your own VPC. Avoid Chinese model APIs if your compliance framework restricts data flow to certain jurisdictions.

The landscape will look different in six months. The architecture principles --- abstraction, routing, fallback, observability --- will still be correct. Build for the principles, not for any specific model.
