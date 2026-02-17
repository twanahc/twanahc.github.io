---
layout: post
title: "Wan 2.2's MoE Architecture: Open-Source Video Models Are Closing the Gap"
date: 2026-02-04
category: models
---

Alibaba's Wan 2.2 introduced Mixture-of-Experts into video diffusion models. 27 billion parameters, but only 14 billion active per inference step. That's a trick borrowed from LLMs — and it's the kind of architectural innovation that could reshape the cost structure of self-hosted video generation.

## What Wan 2.2 Does Differently

Traditional video diffusion models use a single network for the entire denoising process. Every step activates every parameter. Wan 2.2 splits this into two specialized experts:

- **High-noise expert**: Handles early denoising steps. Focuses on overall layout, composition, and scene structure.
- **Low-noise expert**: Handles later steps. Focuses on fine detail, textures, and coherence.

The result: 27B total parameters for quality, but only 14B compute cost per step. You get bigger-model quality at smaller-model prices.

Training data also scaled massively: +65.6% more images and +83.2% more videos compared to Wan 2.1. The model family includes text-to-video (T2V-A14B), image-to-video (I2V-A14B), and a lighter 5B variant for resource-constrained setups.

## The Self-Hosting Math

For a SaaS platform paying per-generation API costs to Veo or Kling, Wan 2.2 presents an interesting alternative for certain tiers:

**API costs (current):**
- Veo 3.1: ~$0.40/second of video
- Kling 3.0: ~$0.08–$0.15/second
- Sora 2: ~$0.10/second

**Self-hosted Wan 2.2 (estimated):**
- GPU rental (A100 80GB on RunPod): ~$1.64/hour
- Generation time per 5-second clip: ~60-120 seconds
- Cost per generation: ~$0.03–$0.06

The 5B variant is even more accessible — it runs on consumer GPUs with as little as 8.19GB VRAM. Not production-grade throughput, but viable for development and testing.

## Where Open Source Makes Sense

Don't replace your Veo/Kling integration with self-hosted Wan. Instead, use it strategically:

**Low-priority batch processing**: Scheduled content, bulk generation, background processing where latency doesn't matter. Run Wan on spot GPU instances at 50-70% savings.

**Preview tier**: Offer users a "fast preview" that generates with Wan at low cost, then upgrade to Veo/Kling for the final render when they approve the concept.

**Fine-tuned custom models**: Wan 2.2 supports LoRA fine-tuning. Enterprise clients who want brand-specific video styles can train custom adapters on your infrastructure. This is a premium feature that API-only models can't offer.

**Development and testing**: Run your full pipeline locally against Wan instead of burning API credits during development. The I2V-A14B variant is particularly useful for testing image-to-video workflows.

## The Quality Gap

Let's be honest: Wan 2.2 doesn't match Veo 3.1 or Kling 3.0 on raw output quality. The gap is most visible in:

- **Human faces**: Commercial models have more face-specific training data
- **Text rendering**: Still a weak point for most open-source models
- **Audio**: Wan doesn't generate native audio (Veo and Sora do)
- **Resolution**: Commercial models handle 4K natively; Wan peaks at 1080p

But for many use cases — product demos, social media content, storyboard previews, concept visualization — Wan's quality is sufficient. The gap is narrowing with each release, and the MoE architecture suggests Alibaba is investing seriously in catching up.

## The Wan 2.2 Model Family

| Model | Params (Active) | Use Case | VRAM |
|---|---|---|---|
| T2V-A14B | 27B (14B) | Text-to-video | 80GB A100 |
| I2V-A14B | 27B (14B) | Image-to-video | 80GB A100 |
| TI2V-5B | 5B (5B) | Text or image-to-video | ~16GB |

The TI2V-5B variant is the sleeper hit. It handles both text-to-video and image-to-video in a single model, making it perfect for a unified generation endpoint. At 5B parameters, it fits on a single consumer GPU.

## Integrating into a Multi-Model Platform

If you're building a multi-model routing architecture, Wan 2.2 slots in as your cost-optimized tier:

```typescript
function selectModel(request) {
  if (request.mode === 'preview') return 'wan-22-t2v';
  if (request.tier === 'basic') return 'wan-22-t2v';
  if (request.needsAudio) return 'veo-31-standard';
  if (request.quality === 'premium') return 'kling-30-omni';
  return 'wan-22-t2v'; // default: cheapest option
}
```

The adapter layer normalizes inputs and outputs across all models. When you add Wan 2.2, you write one adapter — the rest of your pipeline doesn't change.

## What's Next

Wan 2.6 has been referenced in Alibaba's research, suggesting continued rapid iteration. The open-source video generation space is following the same trajectory as open-source LLMs: each generation closes ~30% of the gap with commercial models.

For platform builders, the strategic play is clear: build your architecture to support multiple models from day one, use commercial APIs for quality-critical generation, and keep open-source options ready for cost optimization and custom fine-tuning.

The models that are "good enough" today will be "competitive" in 12 months. Plan accordingly.
