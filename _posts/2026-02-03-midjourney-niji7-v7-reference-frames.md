---
layout: post
title: "Midjourney Niji 7 and V7 Draft Mode: Video-Ready Reference Frames 10x Faster"
date: 2026-02-03
category: models
---

The quality of your AI video starts with the quality of your first frame. If you're running image-to-video pipelines — and you should be, because they produce more controllable output than text-to-video — then image generation speed and quality directly bottleneck your video generation workflow.

Midjourney V7 Draft Mode generates images 10x faster at half the cost. And Niji 7, their anime-focused model, just shipped with the best character consistency in the business. Here's what this means for video pipeline builders.

## V7 Draft Mode: The Speed Play

Midjourney V7 (default since June 2025) introduced Draft Mode — a speed-optimized generation path that trades a small amount of detail for massive throughput:

- **10x faster** generation than standard mode
- **Half the cost** in credits
- Quality is ~85-90% of standard mode — more than enough for reference frames

For a video generation pipeline, this changes the economics of the reference frame step. Previously, generating a high-quality starting frame took 30-60 seconds and consumed significant credits. With Draft Mode, you can generate 10 variations in the time it took to make one, let the user pick the best starting point, then only run the expensive video generation once.

The workflow becomes:

```
User prompt → Draft Mode (10 variations, ~6 seconds each)
    → User picks best → Video generation (Veo/Kling)
```

This "preview-then-commit" pattern cuts your total pipeline cost because you're only running video generation on frames the user has already approved.

## Niji 7: Anime + Character Consistency

Niji 7 launched January 9, 2026, built in collaboration with Spellbrush. The headline improvement: dramatically better **coherency** in character features.

What this means practically:

- **Eye details**: Consistent reflections, highlights, and pupil structure across generations
- **Facial proportions**: Same character actually looks like the same character in different poses
- **Style Reference (--sref)**: Enhanced to reduce style drift — apply a reference once, get consistent results across generations

The `--sref` parameter is the killer feature for multi-shot video projects. Define a visual style or character look once, reference it in every scene's image generation prompt, and get frames that feel like they belong to the same project. Feed those consistent frames into your image-to-video model and you've solved half of the character consistency problem.

## Prompt Strategy Differences

One gotcha: V7 and Niji 7 respond to prompts differently than previous versions.

**V7 (photorealistic):**
- Better coherence for bodies, hands, and complex objects
- Model personalization turned on by default
- Responds well to specific, technical descriptions

**Niji 7 (anime/stylized):**
- More "literal" in prompt interpretation than Niji 5
- Vague atmospheric prompts produce worse results
- Precise technical descriptions (camera angle, lighting setup, character pose) excel
- Flatter rendering emphasizing line art — closer to actual anime production

For your LLM prompt enhancement layer: when routing to Niji 7, output precise technical descriptions. When routing to V7, you have more latitude for atmospheric/mood-based prompts.

## Integration Architecture

For a platform using Midjourney for reference frames:

```
User input → Gemini (prompt enhancement)
    → Midjourney V7 Draft (10 variations)
    → User selection
    → Midjourney V7 Standard (upscale selected)
    → Video model (Veo I2V / Kling I2V)
    → Delivery
```

Midjourney doesn't have an official public API, but services like PiAPI and others provide API access. Since you're already using PiAPI for Kling, adding Midjourney through the same provider simplifies your integration.

Key considerations:
- **Store style references per project**: When a user sets a `--sref` for a project, persist it and apply it automatically to all frame generations in that project
- **Resolution matching**: Generate reference frames at the same resolution your video model expects (typically 1080p) to avoid quality loss from scaling
- **Batch generation**: Generate reference frames for all scenes in a storyboard simultaneously, not sequentially

## The V1 Video Model

Midjourney also launched their own V1 Video Model in June 2025 for animating static images into short clips. It's early — quality and duration are limited compared to Veo or Kling — but it signals that Midjourney sees image-to-video as a natural extension.

For now, Midjourney's video model is best used for subtle motion effects (parallax, gentle camera movement, atmospheric animation) rather than complex scene generation. Think "animated hero image" rather than "short film scene."

## The Cost Picture

For a multi-shot video project generating 10 scenes:

| Step | Tool | Cost per Scene | Total (10 scenes) |
|---|---|---|---|
| Draft variations (10 per scene) | V7 Draft | ~$0.05 | $0.50 |
| Final reference frame | V7 Standard | ~$0.10 | $1.00 |
| Video generation | Veo 3.1 (5 sec) | ~$2.00 | $20.00 |

The image generation step is 5-7% of total cost. Spending more on better reference frames is almost always worth it — a better starting frame means fewer video re-generations.

Don't skimp on the first frame. It's the cheapest part of the pipeline and the one that determines the quality of everything downstream.
