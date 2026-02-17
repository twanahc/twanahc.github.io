---
layout: post
title: "Building a Safety Layer: Content Moderation for AI Video Platforms"
date: 2026-02-01
category: infrastructure
---

In December 2025, users discovered they could generate explicit images through Grok with minimal prompt engineering. Millions of sexualized celebrity deepfakes flooded X before xAI tightened the filters. The reputational damage was immediate and severe.

This is the cautionary tale every AI video platform builder needs to internalize. One moderation failure can undo months of product work. Content moderation isn't a nice-to-have — it's existential.

## The Three-Point Moderation Architecture

You need moderation at three points in your pipeline, not one:

### 1. Input Screening (Prompts + Uploads)

Before you spend money on generation, screen what's coming in:

**Prompt screening**: Run every text prompt through a classifier before it hits your video model. This catches obvious policy violations ("generate a deepfake of...") and saves API costs on generations that would be rejected anyway.

**Upload screening**: If users can upload reference images for image-to-video, screen those too. An explicit reference image will produce an explicit video, regardless of the text prompt.

**Why before generation**: A rejected prompt costs $0. A generated-then-rejected video costs $0.50-$4.00 in wasted API calls plus the compute for moderation. Screen inputs first.

### 2. Output Screening (Generated Content)

Even with clean inputs, models can generate unexpected content. Screen every output before delivery:

- Frame sampling: Extract 5-10 frames from the generated video and run them through an image classifier
- Full video analysis: For higher-risk categories, analyze the complete video
- Audio screening: If the video has generated audio/speech, screen the audio content separately

### 3. User Reporting + Review

Automated moderation catches 95-99% of violations. The remaining 1-5% requires human review:

- Provide a "Report" button on every generated video
- Queue reported content for human review
- Track false positive rates — overly aggressive moderation hurts user experience

## The API Options

Several services specialize in content moderation at scale:

### Sightengine

- **Pricing**: Starts at $29/month, scales to $399/month
- **Free tier**: Available for testing
- **Capabilities**: Real-time moderation for images, video, and text
- **Strengths**: Good price/performance ratio for startups, easy API integration

### Hive Moderation

- **Pricing**: Enterprise (contact sales)
- **Capabilities**: 25+ subclasses across 5 content categories
- **Strengths**: Fine-grained classification, video frame sampling, live stream support
- **Best for**: Platforms that need detailed categorization beyond binary safe/unsafe

### Google Cloud Vision SafeSearch

- **Pricing**: Pay-per-use, ~$1.50 per 1000 images
- **Capabilities**: Adult, violence, racy, spoof, medical content detection
- **Strengths**: Good accuracy, integrates well if you're already on GCP
- **Best for**: Platforms already using Google Cloud

### Amazon Rekognition

- **Pricing**: Pay-per-use, ~$1.00 per 1000 images
- **Capabilities**: Content moderation with confidence scores
- **Strengths**: Customizable thresholds, AWS ecosystem integration

### DIY with Gemini Flash

For platforms already using Gemini in their pipeline:

```
Generated video → Extract frames → Gemini Flash analysis:
"Does this image contain explicit, violent, or policy-violating content?
Respond with: {safe: boolean, category: string, confidence: number}"
```

Cost: ~$0.001 per check. This won't match specialized moderation APIs on edge cases, but it's cheap enough to run on every generation as a first pass.

## The Kling Problem

If you're using Kling via PiAPI, be aware that Kling has some of the strictest built-in moderation in the industry. Users consistently report "aggressive and sometimes over-sensitive" filters that reject legitimate creative prompts.

This creates a hidden cost: failed generations due to overly strict moderation still consume queue time and user patience. Two strategies:

1. **Pre-screen prompts for Kling-specific triggers**: Build a list of terms and concepts that Kling's moderation flags, and either warn users or automatically rephrase before submission
2. **Route sensitive-but-legitimate prompts to other models**: If Kling rejects a medical education video prompt, route to Veo which may handle it correctly

## Cost Budgeting

For a platform processing 10,000 video generations per month:

| Moderation Layer | Method | Monthly Cost |
|---|---|---|
| Prompt screening | Gemini Flash | ~$10 |
| Upload screening | Sightengine or Gemini | ~$15-30 |
| Output screening (frame sampling) | Sightengine | ~$30-100 |
| Human review (1-5% escalation) | Internal or contract | Varies |

Total: roughly $55-140/month for automated moderation at 10K generations. That's $0.006-$0.014 per generation — negligible compared to your $0.50-$4.00 generation cost.

The cost of *not* moderating? One viral incident. Lost users. Lost enterprise contracts. Possible legal liability. App store removal. The math is obvious.

## Implementation Priority

If you're building moderation into a new platform:

1. **Start with input screening.** It's the cheapest and prevents the most expensive failures (wasted generation API calls).
2. **Add output frame sampling.** Extract 5 frames, run through a classifier. Catches most visual policy violations.
3. **Add user reporting.** Simple flag button, queue for review.
4. **Iterate on thresholds.** Track false positive and false negative rates. Adjust per content category.

Don't wait until you have a moderation incident to build moderation. By then it's a crisis, not an engineering project.
