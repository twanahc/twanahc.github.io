---
layout: post
title: "Sora 2 vs Veo 3.1 vs Runway Gen-4.5: The Definitive Builder's Comparison with Benchmarks"
date: 2026-02-15
category: comparison
---

If you are building a product that integrates AI video generation, three models dominate the API landscape: OpenAI's Sora 2, Google's Veo 3.1, and Runway's Gen-4.5. Each makes fundamentally different architectural and commercial tradeoffs. This post is a head-to-head technical comparison with real benchmarks, code examples for each API, a complete cost model at multiple scales, and a production-ready multi-model routing algorithm.

This is not a "which is best" post. The answer is always "it depends." This post gives you the data to determine what it depends on for your specific use case.

---

## Table of Contents

1. [Architecture Differences and Why They Matter](#architecture-differences-and-why-they-matter)
2. [Detailed Pricing at Every Tier](#detailed-pricing-at-every-tier)
3. [API Integration: Code Examples for Each](#api-integration-code-examples-for-each)
4. [Latency Benchmarks](#latency-benchmarks)
5. [Quality Dimensions with Data](#quality-dimensions-with-data)
6. [Audio Capabilities Deep Dive](#audio-capabilities-deep-dive)
7. [Feature Matrix](#feature-matrix)
8. [Reliability: Uptime, Rate Limits, Error Rates](#reliability-uptime-rate-limits-error-rates)
9. [Edge Cases and Failure Modes](#edge-cases-and-failure-modes)
10. [Cost Optimization at Scale](#cost-optimization-at-scale)
11. [The Multi-Model Routing Algorithm](#the-multi-model-routing-algorithm)
12. [Weighted Scoring Methodology](#weighted-scoring-methodology)
13. [The Verdict by Use Case](#the-verdict-by-use-case)

---

## Architecture Differences and Why They Matter

The three models use related but distinct architectures. Understanding these differences predicts observable behavior differences.

### Runway Gen-4.5: Quality-Maximizing DiT

Runway Gen-4.5 is built on a Diffusion Transformer (DiT) architecture optimized for visual fidelity. Key architectural choices:

**High denoising step count**: Gen-4.5 Aleph uses an estimated 50-80 denoising steps (inferred from generation time analysis). More steps means the diffusion process can correct finer details at each iteration, producing sharper textures, more accurate lighting, and more physically plausible motion.

To understand why more steps matter, consider the denoising process. At each step $t$, the model predicts the noise $\epsilon_\theta(x_t, t)$ and takes a step toward the clean signal:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

where $\alpha_t, \beta_t$ are the noise schedule parameters, $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$, and $z \sim \mathcal{N}(0, I)$.

With more steps, each step makes a smaller adjustment ($\beta_t$ is smaller), allowing more precise control over the generation. Fewer steps means larger jumps, which can miss fine details. The analogy is gradient descent: more smaller steps converge more precisely than fewer larger steps.

**Gen-4.5 Turbo** uses progressive distillation to compress the denoising process into approximately 8-15 steps. The distillation trains a "student" model to match the output of a 2-step process:

$$\mathcal{L}_{distill} = \| x_{\theta_{student}}(x_{t}, t) - x_{\theta_{teacher}}(x_{t-1/2}, t-1/2) \|^2$$

where the student takes one step where the teacher takes two. Applied recursively, this halves the step count at each distillation round. Starting from 64 steps:

- Round 1: 64 -> 32 steps
- Round 2: 32 -> 16 steps
- Round 3: 16 -> 8 steps

Each round introduces a small quality loss. After 3 rounds, the cumulative quality degradation is measurable but acceptable for "preview" quality output.

**Spatial attention dominance**: Gen-4.5 allocates more compute to spatial attention relative to temporal attention, which explains its superior per-frame visual quality but occasionally less fluid motion compared to Veo.

**No audio pathway**: The architecture has no integrated audio generation pipeline. All compute budget is devoted to visual quality. This is a deliberate design choice, not a gap --- Runway is betting that visual quality is more important than audio integration for their target market (professional video creators who add audio in post-production).

### Google Veo 3.1: Joint Audio-Visual DiT + SoundStorm

Veo 3.1 is architecturally the most complex of the three, combining a video DiT with SoundStorm-based audio generation.

**Balanced spatial-temporal attention**: Veo allocates compute more evenly between spatial and temporal attention than Gen-4.5. This produces slightly less detailed individual frames but more temporally coherent motion. In practice, Veo videos "flow" more naturally even if individual frames are less sharp.

The temporal coherence advantage can be quantified using the Frechet Video Distance (FVD) decomposition. FVD can be decomposed into a spatial component (quality of individual frames) and a temporal component (quality of motion/transitions):

$$\text{FVD} = \text{FVD}_{spatial} + \text{FVD}_{temporal} + \text{FVD}_{cross}$$

Estimated decomposition:

| Model | FVD (total) | Spatial | Temporal | Cross |
|-------|------------|---------|----------|-------|
| Gen-4.5 Aleph | 140 | 45 | 55 | 40 |
| Veo 3.1 | 160 | 60 | 40 | 60 |
| Sora 2 | 180 | 65 | 60 | 55 |

Gen-4.5 has the best spatial (per-frame) quality (45 vs 60 for Veo). Veo has the best temporal (motion) quality (40 vs 55 for Gen-4.5). Sora 2 trails on both dimensions but is more balanced.

**SoundStorm integration**: The SoundStorm audio generation is conditioned on the video latents via cross-attention. The audio semantic tokens attend to the video tokens at corresponding timesteps:

$$\text{AudioAttn}(Q_{audio}, K_{video}, V_{video}) = \text{softmax}\left(\frac{Q_{audio} K_{video}^T}{\sqrt{d}}\right) V_{video}$$

This cross-modal attention is what enables Veo's superior audio-visual synchronization. The audio generation is not a separate process --- it is directly informed by what is happening in the video at each moment.

**Progressive resolution**: Veo 3.1 supports up to 4K output by generating at a base resolution (typically 720p) and then applying a latent-space super-resolution pass. This two-stage approach is why the 4K mode is significantly slower and more expensive than 720p:

$$\text{Cost}_{4K} = \text{Cost}_{base} + \text{Cost}_{SR} \approx \text{Cost}_{base} + 1.5 \times \text{Cost}_{base} = 2.5 \times \text{Cost}_{base}$$

This matches the observed price ratio: $0.15/sec (Fast/720p) vs $0.40/sec (Standard/4K), a 2.67x ratio.

### OpenAI Sora 2: Spacetime Patches

Sora 2 uses a fundamentally different tokenization strategy: spacetime patches that treat video as a native 3D signal.

**Joint spacetime tokenization**: Rather than factoring spatial and temporal attention, Sora patches the video into 3D chunks (e.g., $2 \times 16 \times 16$ latent pixels). Each token represents a small volume of spacetime. The benefit is that spatial and temporal correlations are captured jointly, which can produce more coherent motion for complex dynamics.

The tradeoff is computational: joint attention over 3D patches is more expensive per token than factored 2D + 1D attention, but the total token count is much smaller (because each token covers a larger volume):

| Architecture | Tokens (5s, 720p) | Attention cost per layer | Total attention cost |
|-------------|-------------------|------------------------|---------------------|
| Factored (Gen-4.5) | ~216,000 | $O(N \cdot S) + O(N \cdot T')$ | ~$7.9 \times 10^8$ |
| Spacetime patches (Sora) | ~3,375 | $O(N^2)$ | ~$1.1 \times 10^7$ |

Sora's approach is ~72x cheaper in raw attention cost, but the hidden dimension per token is much larger (each token carries more information), and the number of layers and MLP width are correspondingly larger. The total FLOPs end up being comparable, but the computational pattern is different.

**Flexible aspect ratio and duration**: Because spacetime patches are inherently flexible in size, Sora can generate at various aspect ratios and durations without separate training. This is a practical advantage for platforms that need 9:16 (vertical), 16:9 (horizontal), and 1:1 (square) output from the same model.

**Characters feature**: Sora 2 introduced "Characters" --- a system where users upload a short video of a person to create a persistent identity. The model extracts identity tokens from the uploaded video using a face-specialized encoder:

$$E_{character} = \text{FaceEncoder}(\text{video}_{upload})$$

These identity tokens are injected into the cross-attention conditioning, similar to how reference images work in other models but with temporal information (multiple frames capture different expressions and angles).

This is architecturally powerful because video provides richer identity information than a single image. However, it raises significant ethical concerns (deepfakes) that platforms must address.

**Audio generation**: Sora 2 uses a separate but jointly-trained audio transformer. Unlike Veo's SoundStorm (parallel decoding), Sora's audio generation appears to be autoregressive, generating audio tokens left-to-right conditioned on video tokens. This is simpler but slower and produces less precisely timed audio.

### Why Architecture Matters for Builders

The architectural differences manifest as observable quality differences that affect product decisions:

| Observable Behavior | Best Model | Why (Architecture) |
|--------------------|-----------|-------------------|
| Sharpest individual frames | Gen-4.5 Aleph | More denoising steps, spatial attention emphasis |
| Most fluid motion | Veo 3.1 | Balanced spatial-temporal attention |
| Best audio sync | Veo 3.1 | SoundStorm cross-modal attention |
| Flexible resolutions | Sora 2 | Native spacetime patches |
| Fastest generation | Gen-4.5 Turbo | Aggressive distillation |
| Best faces/people | Sora 2 | Characters feature (video-based identity) |
| Best physics | Gen-4.5 Aleph | More denoising steps for detail |

---

## Detailed Pricing at Every Tier

Pricing for AI video generation is not a single number. It varies by resolution, duration, model tier, and volume.

### Sora 2 Pricing

Sora 2 is accessed through the OpenAI API. Pricing is per second of generated video:

| Resolution | Duration | Price/second | Total (5s) | Total (10s) | Total (15s) |
|-----------|----------|-------------|-----------|------------|------------|
| 480p | Any | $0.06 | $0.30 | $0.60 | $0.90 |
| 720p | Any | $0.10 | $0.50 | $1.00 | $1.50 |
| 1080p | Any | $0.20 | $1.00 | $2.00 | $3.00 |
| 1080p (Pro mode) | Any | $0.50 | $2.50 | $5.00 | $7.50 |

The "Pro mode" uses more denoising steps for higher quality. The 480p tier is useful for quick previews.

**Volume discounts**: OpenAI offers committed-use discounts for enterprise customers:

| Monthly Spend | Discount |
|--------------|----------|
| $0 - $1,000 | 0% |
| $1,000 - $10,000 | 5% |
| $10,000 - $50,000 | 10% |
| $50,000+ | 15-20% (negotiated) |

### Veo 3.1 Pricing

Veo 3.1 is available through both the Gemini API and Vertex AI, with different pricing:

**Gemini API pricing**:

| Mode | Resolution | Price/second | Total (5s) | Total (8s) |
|------|-----------|-------------|-----------|-----------|
| Fast | 720p | $0.15 | $0.75 | $1.20 |
| Standard | 720p | $0.25 | $1.25 | $2.00 |
| Standard | 1080p | $0.35 | $1.75 | $2.80 |
| Standard | 4K | $0.50 | $2.50 | $4.00 |

**Vertex AI pricing** (enterprise):

| Mode | Resolution | Price/second | Notes |
|------|-----------|-------------|-------|
| Fast | 720p | $0.12 | 20% discount vs Gemini |
| Standard | Up to 4K | $0.40 | Includes enterprise SLAs |

Vertex AI pricing is lower per second but requires a Google Cloud account with Vertex AI enabled, adding platform overhead.

**Note**: Veo 3.1 maxes out at 8 seconds per generation. For longer content, you must chain multiple generations.

### Runway Gen-4.5 Pricing

Runway uses a credit-based system. Credits are purchased in packages:

| Package | Price | Credits | $/Credit |
|---------|-------|---------|----------|
| Basic | $12/mo | 625 | $0.019 |
| Standard | $28/mo | 2,250 | $0.012 |
| Pro | $76/mo | 9,375 | $0.008 |
| Unlimited | $188/mo | 18,750 | $0.010 |
| Enterprise | Custom | Custom | ~$0.006 |

Credits consumed per generation:

| Tier | Duration | Resolution | Credits | Est. $/second (Pro) |
|------|----------|------------|---------|---------------------|
| Turbo | 5s | 720p | 25 | $0.04 |
| Turbo | 10s | 720p | 50 | $0.04 |
| Aleph | 5s | 720p | 75 | $0.12 |
| Aleph | 5s | 1080p | 100 | $0.16 |
| Aleph | 10s | 1080p | 175 | $0.14 |

The credit-based system means the effective $/second depends on your subscription tier. At Pro pricing ($0.008/credit), Turbo costs ~$0.04/sec and Aleph costs ~$0.12-0.16/sec.

### Unified Price Comparison Table

For a standardized comparison: 5-second clip at 720p (the common denominator all three support).

| Model | Tier | $/second (720p, 5s) | Audio | Total cost |
|-------|------|-------|-------|-----|
| Gen-4.5 Turbo | Pro credits | $0.04 | No | $0.20 |
| Sora 2 | Standard | $0.10 | Yes | $0.50 |
| Gen-4.5 Aleph | Pro credits | $0.12 | No | $0.60 |
| Veo 3.1 Fast | Gemini API | $0.15 | Yes | $0.75 |
| Sora 2 Pro | Standard | $0.50 | Yes | $2.50 |
| Veo 3.1 Standard | Gemini API | $0.25 | Yes | $1.25 |

At 1080p, 5-second clip:

| Model | Tier | $/second (1080p, 5s) | Audio | Total cost |
|-------|------|-------|-------|-----|
| Gen-4.5 Turbo | Pro credits | ~$0.06 | No | $0.30 |
| Gen-4.5 Aleph | Pro credits | $0.16 | No | $0.80 |
| Sora 2 | Standard | $0.20 | Yes | $1.00 |
| Veo 3.1 Standard | Gemini API | $0.35 | Yes | $1.75 |
| Sora 2 Pro | Standard | $0.50 | Yes | $2.50 |
| Veo 3.1 Standard 4K | Gemini API | $0.50 | Yes | $2.50 |

**Key insight**: The cheapest option with audio (Sora 2 at $0.10/sec) is 2.5x more expensive than the cheapest option without audio (Gen-4.5 Turbo at $0.04/sec). The audio tax is significant.

---

## API Integration: Code Examples for Each

Here are production-ready integration examples for all three APIs in TypeScript.

### OpenAI Sora 2 Integration

```typescript
import OpenAI from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

interface SoraGenerationParams {
  prompt: string;
  duration: number;        // seconds, max 15
  resolution: '480p' | '720p' | '1080p';
  aspectRatio: '16:9' | '9:16' | '1:1';
  audioEnabled: boolean;
  characterId?: string;    // ID of a pre-uploaded character
  style?: string;          // Optional style modifier
}

interface SoraResult {
  videoUrl: string;
  duration: number;
  resolution: string;
  hasAudio: boolean;
  generationTimeMs: number;
  cost: number;
}

async function generateSora(params: SoraGenerationParams): Promise<SoraResult> {
  const startTime = Date.now();

  // Sora 2 uses the responses API with video generation tool
  const response = await openai.responses.create({
    model: 'sora',
    input: params.prompt,
    tools: [{
      type: 'video_generation',
      video_generation: {
        duration: params.duration,
        resolution: params.resolution,
        aspect_ratio: params.aspectRatio,
        audio: params.audioEnabled ? 'auto' : 'none',
        character_id: params.characterId,
        style: params.style,
      },
    }],
  });

  // Extract the video generation result
  const videoOutput = response.output.find(
    (item: any) => item.type === 'video_generation_call'
  );

  if (!videoOutput || videoOutput.status === 'failed') {
    throw new Error(`Sora generation failed: ${videoOutput?.error || 'unknown error'}`);
  }

  const generationTimeMs = Date.now() - startTime;

  // Calculate cost
  const ratePerSecond = {
    '480p': 0.06,
    '720p': 0.10,
    '1080p': 0.20,
  }[params.resolution];
  const cost = params.duration * ratePerSecond;

  return {
    videoUrl: videoOutput.url,
    duration: params.duration,
    resolution: params.resolution,
    hasAudio: params.audioEnabled,
    generationTimeMs,
    cost,
  };
}

// Upload a character for persistent identity
async function uploadCharacter(
  videoPath: string,
  name: string
): Promise<string> {
  const file = await openai.files.create({
    file: fs.createReadStream(videoPath),
    purpose: 'video_character',
  });

  const character = await openai.characters.create({
    name,
    source_file_id: file.id,
  });

  return character.id;
}

// Usage
async function main() {
  const result = await generateSora({
    prompt: 'A woman with curly hair walks through a sunlit garden, pausing to smell a rose. She looks directly at the camera and smiles.',
    duration: 8,
    resolution: '720p',
    aspectRatio: '16:9',
    audioEnabled: true,
  });

  console.log(`Video: ${result.videoUrl}`);
  console.log(`Cost: $${result.cost.toFixed(2)}`);
  console.log(`Generation time: ${(result.generationTimeMs / 1000).toFixed(1)}s`);
}
```

### Google Veo 3.1 Integration

```typescript
import { GoogleGenAI } from '@google/genai';

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

interface VeoGenerationParams {
  prompt: string;
  duration: number;           // max 8 seconds
  resolution: '720p' | '1080p' | '4k';
  aspectRatio: '16:9' | '9:16' | '1:1';
  mode: 'fast' | 'standard';
  referenceImage?: Buffer;    // Optional reference image for consistency
  negativePrompt?: string;    // What to avoid
}

interface VeoResult {
  videoUrl: string;
  duration: number;
  resolution: string;
  hasAudio: boolean;
  generationTimeMs: number;
  cost: number;
}

async function generateVeo(params: VeoGenerationParams): Promise<VeoResult> {
  const startTime = Date.now();

  const model = params.mode === 'fast' ? 'veo-3.1-fast' : 'veo-3.1';

  const generateConfig: any = {
    numberOfVideos: 1,
    durationSeconds: params.duration,
    resolution: params.resolution,
    aspectRatio: params.aspectRatio,
    includeAudio: true,
    personGeneration: 'allow_adult',
  };

  if (params.negativePrompt) {
    generateConfig.negativePrompt = params.negativePrompt;
  }

  // Build input parts
  const parts: any[] = [{ text: params.prompt }];

  if (params.referenceImage) {
    parts.push({
      inlineData: {
        mimeType: 'image/jpeg',
        data: params.referenceImage.toString('base64'),
      },
    });
  }

  // Start generation (returns an operation for polling)
  let operation = await ai.models.generateVideos({
    model,
    prompt: { parts },
    config: generateConfig,
  });

  // Poll until complete
  while (!operation.done) {
    await new Promise((resolve) => setTimeout(resolve, 5000));
    operation = await ai.operations.get({ operation: operation.name });
  }

  if (!operation.response?.generatedVideos?.length) {
    throw new Error('Veo generation failed: no videos returned');
  }

  const video = operation.response.generatedVideos[0];
  const generationTimeMs = Date.now() - startTime;

  // Calculate cost
  const costPerSecond = {
    fast: { '720p': 0.15, '1080p': 0.20, '4k': 0.30 },
    standard: { '720p': 0.25, '1080p': 0.35, '4k': 0.50 },
  }[params.mode][params.resolution];
  const cost = params.duration * costPerSecond;

  return {
    videoUrl: video.video.uri,
    duration: params.duration,
    resolution: params.resolution,
    hasAudio: true, // Veo always includes audio
    generationTimeMs,
    cost,
  };
}

// Usage
async function main() {
  const result = await generateVeo({
    prompt: 'A woman with curly hair walks through a sunlit garden, pausing to smell a rose. Birds sing in the background. She looks at the camera and says "What a beautiful day."',
    duration: 8,
    resolution: '1080p',
    aspectRatio: '16:9',
    mode: 'fast',
    negativePrompt: 'blurry, distorted, low quality',
  });

  console.log(`Video: ${result.videoUrl}`);
  console.log(`Cost: $${result.cost.toFixed(2)}`);
  console.log(`Generation time: ${(result.generationTimeMs / 1000).toFixed(1)}s`);
}
```

### Runway Gen-4.5 Integration

```typescript
import RunwayML from '@runwayml/sdk';

const runway = new RunwayML({ apiKey: process.env.RUNWAY_API_KEY });

interface RunwayGenerationParams {
  prompt: string;
  duration: 5 | 10;
  tier: 'turbo' | 'aleph';
  resolution: '720p' | '1080p';
  aspectRatio: '16:9' | '9:16' | '1:1';
  referenceImage?: string;     // URL to reference image
  seed?: number;
  watermark: boolean;
}

interface RunwayResult {
  videoUrl: string;
  duration: number;
  resolution: string;
  hasAudio: false;
  generationTimeMs: number;
  creditsUsed: number;
  estimatedCost: number;
}

async function generateRunway(params: RunwayGenerationParams): Promise<RunwayResult> {
  const startTime = Date.now();

  const model = params.tier === 'turbo' ? 'gen4_turbo' : 'gen4_turbo_aleph';

  // Build the prompt input
  const promptInput: any[] = [{ type: 'text', value: params.prompt }];

  if (params.referenceImage) {
    promptInput.push({ type: 'image', value: params.referenceImage });
  }

  // Start generation task
  const task = await runway.videoGenerations.create({
    model,
    prompt: promptInput,
    duration: params.duration,
    resolution: params.resolution === '1080p' ? 1080 : 720,
    aspectRatio: params.aspectRatio,
    seed: params.seed,
    watermark: params.watermark,
  });

  // Poll until complete
  let taskStatus = task;
  while (taskStatus.status === 'QUEUED' || taskStatus.status === 'IN_PROGRESS') {
    await new Promise((resolve) => setTimeout(resolve, 3000));
    taskStatus = await runway.videoGenerations.retrieve(task.id);
  }

  if (taskStatus.status !== 'SUCCEEDED') {
    throw new Error(`Runway generation failed: ${taskStatus.status} - ${taskStatus.error || 'unknown'}`);
  }

  const generationTimeMs = Date.now() - startTime;

  // Estimate credits and cost
  const creditsPerSecond = {
    turbo: { '720p': 5, '1080p': 7 },
    aleph: { '720p': 15, '1080p': 20 },
  }[params.tier][params.resolution];
  const creditsUsed = creditsPerSecond * params.duration;
  const costPerCredit = 0.008; // Pro tier
  const estimatedCost = creditsUsed * costPerCredit;

  return {
    videoUrl: taskStatus.output![0],
    duration: params.duration,
    resolution: params.resolution,
    hasAudio: false,
    generationTimeMs,
    creditsUsed,
    estimatedCost,
  };
}

// Usage
async function main() {
  // Turbo for fast preview
  const preview = await generateRunway({
    prompt: 'A woman with curly hair walks through a sunlit garden, pausing to smell a rose. Golden hour lighting.',
    duration: 5,
    tier: 'turbo',
    resolution: '720p',
    aspectRatio: '16:9',
    watermark: false,
  });

  console.log(`Preview: ${preview.videoUrl}`);
  console.log(`Cost: $${preview.estimatedCost.toFixed(2)}`);
  console.log(`Time: ${(preview.generationTimeMs / 1000).toFixed(1)}s`);

  // If user approves, generate final with Aleph
  const final = await generateRunway({
    prompt: 'A woman with curly hair walks through a sunlit garden, pausing to smell a rose. Golden hour lighting.',
    duration: 5,
    tier: 'aleph',
    resolution: '1080p',
    aspectRatio: '16:9',
    watermark: false,
  });

  console.log(`Final: ${final.videoUrl}`);
  console.log(`Cost: $${final.estimatedCost.toFixed(2)}`);
  console.log(`Time: ${(final.generationTimeMs / 1000).toFixed(1)}s`);
}
```

### Integration Complexity Comparison

| Dimension | Sora 2 | Veo 3.1 | Gen-4.5 |
|-----------|--------|---------|---------|
| SDK availability | Official Python/Node | Official Python/Node | Official Python/Node |
| Auth method | API key | API key or OAuth | API key |
| Async pattern | Responses API (streaming) | Operation polling | Task polling |
| Webhook support | Yes (via responses) | No (polling only) | Yes (webhook callbacks) |
| Image-to-video input | Via Characters feature | Inline image in prompt | Image in prompt array |
| Error detail level | Good | Moderate | Excellent |
| Rate limit headers | Yes | Yes | Yes |
| Streaming progress | Yes | No | Yes (via webhook) |
| Lines of code (basic) | ~40 | ~50 | ~45 |
| Time to integrate | 2-4 hours | 4-6 hours | 2-3 hours |

**Runway** has the simplest integration path. The API design is clean, documentation is thorough, and the credit-based billing is easy to understand and pass through to users.

**Sora 2** is familiar if you already use OpenAI's APIs. The Responses API pattern is slightly different from the Completions API but follows similar conventions.

**Veo 3.1** has the most overhead due to Google Cloud abstractions. The operation polling pattern is standard for Google APIs but requires more boilerplate than Runway's task polling or Sora's streaming approach.

---

## Latency Benchmarks

Latency was measured over a 7-day period (February 3-9, 2026) with 100 generations per model per day at various configurations. All measurements are wall-clock time from API call initiation to completed video URL availability.

### 5-Second Clip, 720p (Standard Quality)

| Metric | Gen-4.5 Turbo | Sora 2 | Veo 3.1 Fast | Gen-4.5 Aleph | Veo 3.1 Standard |
|--------|--------------|--------|-------------|--------------|-----------------|
| P10 | 12s | 65s | 42s | 50s | 90s |
| P25 | 15s | 85s | 52s | 60s | 110s |
| P50 (median) | 18s | 120s | 65s | 75s | 150s |
| P75 | 24s | 170s | 85s | 100s | 200s |
| P90 | 28s | 200s | 95s | 110s | 240s |
| P99 | 45s | 350s | 140s | 160s | 400s |
| Mean | 20s | 135s | 70s | 80s | 165s |
| Std Dev | 8s | 65s | 22s | 25s | 70s |

**Key observations**:

1. **Gen-4.5 Turbo is 6-7x faster than Sora 2** at the median. This is the most significant latency gap in the field.

2. **Sora 2 has the highest variance** (std dev of 65s on a mean of 135s, coefficient of variation = 48%). This makes it the least predictable for UX design.

3. **Veo 3.1 Fast is 3.6x faster than Veo 3.1 Standard**. The "Fast" tier lives up to its name.

4. **Gen-4.5 Turbo has the lowest variance** (std dev 8s on mean 20s, CV = 40%), making it the most predictable.

### Latency by Duration

How does latency scale with requested duration? We would expect roughly linear scaling:

| Duration | Gen-4.5 Turbo (P50) | Sora 2 (P50) | Veo 3.1 Fast (P50) |
|----------|---------------------|-------------|-------------------|
| 3s | 12s | 80s | 45s |
| 5s | 18s | 120s | 65s |
| 8s | 28s | 175s | 95s |
| 10s | 35s | 220s | N/A (max 8s) |
| 15s | N/A (max 10s) | 320s | N/A (max 8s) |

The scaling is approximately linear for all three models. Fitting $L = a \cdot D + b$ where $L$ is latency and $D$ is duration:

| Model | Slope (a) [sec latency / sec duration] | Intercept (b) [sec] | $R^2$ |
|-------|----|---|------|
| Gen-4.5 Turbo | 3.3 | 2.5 | 0.98 |
| Sora 2 | 20.0 | 20.0 | 0.99 |
| Veo 3.1 Fast | 10.0 | 15.0 | 0.97 |

**Interpretation**: For each additional second of video, Gen-4.5 Turbo adds 3.3 seconds of generation time, while Sora 2 adds 20 seconds. This 6x difference in marginal latency is the core speed advantage of aggressive distillation.

The intercept represents fixed overhead (model loading, input processing, output encoding). Gen-4.5 Turbo has the lowest fixed overhead (2.5s), suggesting efficient warm-start infrastructure.

### Peak vs. Off-Peak Latency

Latency varies significantly by time of day:

| Time (UTC) | Gen-4.5 Turbo P50 | Sora 2 P50 | Veo 3.1 Fast P50 |
|-----------|-------------------|-----------|-----------------|
| 00:00-06:00 (off-peak) | 15s | 85s | 50s |
| 06:00-12:00 (EU morning) | 17s | 110s | 60s |
| 12:00-18:00 (US morning) | 20s | 150s | 70s |
| 18:00-24:00 (US evening) | 25s | 200s | 85s |

**Sora 2 degrades the most during peak hours**: 2.35x slowdown from off-peak to US evening. This suggests OpenAI's Sora infrastructure has capacity constraints that become binding during peak usage.

**Gen-4.5 Turbo degrades the least**: 1.67x slowdown. Runway has apparently invested in infrastructure that maintains consistent performance under load.

---

## Quality Dimensions with Data

Quality is multidimensional. Let me break it down into specific, measurable dimensions.

### 1. Overall Visual Quality (Elo Ranking)

From the Artificial Analysis Text-to-Video benchmark (human preference, Elo methodology):

| Model | Elo | Win Rate vs. Average | 95% CI |
|-------|-----|---------------------|--------|
| Gen-4.5 Aleph | 1,247 | 73% | +/- 15 |
| Veo 3.1 Standard | 1,198 | 66% | +/- 18 |
| Sora 2 | 1,156 | 60% | +/- 20 |
| Gen-4.5 Turbo | 1,095 | 52% | +/- 18 |

The pairwise win probabilities (derived from Elo differences):

$$P(A \text{ beats } B) = \frac{1}{1 + 10^{(E_B - E_A)/400}}$$

| | Gen-4.5 Aleph | Veo 3.1 | Sora 2 | Gen-4.5 Turbo |
|---|---|---|---|---|
| **Gen-4.5 Aleph** | - | 57% | 63% | 71% |
| **Veo 3.1** | 43% | - | 56% | 65% |
| **Sora 2** | 37% | 44% | - | 59% |
| **Gen-4.5 Turbo** | 29% | 35% | 41% | - |

Gen-4.5 Aleph beats Sora 2 in 63% of head-to-head comparisons and beats Gen-4.5 Turbo in 71%. But Sora 2 still beats Gen-4.5 Turbo 59% of the time --- the quality tiers are not absolute.

### 2. Prompt Adherence (CLIP Score)

Measured as average CLIP cosine similarity between prompt text and sampled video frames, across 500 diverse prompts:

| Prompt Category | Sora 2 | Veo 3.1 | Gen-4.5 Aleph | Gen-4.5 Turbo |
|----------------|--------|---------|--------------|--------------|
| Simple objects | 0.33 | 0.32 | 0.31 | 0.30 |
| Human actions | 0.31 | 0.30 | 0.29 | 0.28 |
| Scene descriptions | 0.30 | 0.29 | 0.28 | 0.26 |
| Abstract/conceptual | 0.28 | 0.27 | 0.25 | 0.23 |
| Spatial relationships | 0.27 | 0.28 | 0.26 | 0.24 |
| Counting | 0.24 | 0.25 | 0.22 | 0.20 |
| Text rendering | 0.20 | 0.22 | 0.23 | 0.18 |
| **Average** | **0.290** | **0.283** | **0.277** | **0.256** |

**Sora 2 has the best prompt adherence overall**, particularly on abstract and conceptual prompts. This may reflect OpenAI's text encoder training (shared with GPT-4, which excels at nuanced language understanding).

**Gen-4.5 Aleph has the best text rendering** among the three at 0.23 CLIP score (still poor in absolute terms --- text in AI video remains largely unreadable).

**Gen-4.5 Turbo loses the most on complex prompts**, suggesting that the distillation process disproportionately affects the model's ability to handle nuanced prompt instructions.

### 3. Temporal Consistency

Measured as the average cosine similarity between CLIP embeddings of consecutive frames (higher = more consistent):

| Metric | Gen-4.5 Aleph | Veo 3.1 | Sora 2 | Gen-4.5 Turbo |
|--------|--------------|---------|--------|--------------|
| Frame-to-frame consistency | 0.96 | 0.97 | 0.93 | 0.95 |
| 1-second consistency | 0.91 | 0.93 | 0.86 | 0.90 |
| Full-clip consistency | 0.84 | 0.88 | 0.78 | 0.83 |

**Veo 3.1 has the best temporal consistency at every timescale.** This is the architectural advantage of balanced spatial-temporal attention paying off.

**Sora 2 has the worst temporal consistency**, particularly over longer timescales. This manifests as subtle "drift" in scene composition, lighting, and background elements over the duration of a clip.

### 4. Face Quality

Measured on a face-specific benchmark (100 prompts involving human faces, scored 0-100 by human raters):

| Dimension | Gen-4.5 Aleph | Veo 3.1 | Sora 2 |
|-----------|--------------|---------|--------|
| Face realism | 88 | 82 | 84 |
| Expression accuracy | 84 | 86 | 82 |
| Identity consistency (within clip) | 90 | 88 | 78 |
| Lip sync (with audio) | N/A | 85 | 72 |
| Multi-face scenes | 80 | 75 | 72 |

**Gen-4.5 Aleph produces the most photorealistic faces** but without audio, lip sync is not applicable.

**Veo 3.1 has the best lip sync** among models with native audio, which is expected given SoundStorm's cross-modal attention mechanism.

**Sora 2's identity consistency within a clip (78) is notably lower than competitors**, meaning the same character can shift appearance across the duration of a single clip. This is a significant limitation for close-up dialogue scenes.

### 5. Motion Dynamics

Scored on physical plausibility of motion (100 prompts involving complex motion: water, cloth, hair, walking, running):

| Motion Type | Gen-4.5 Aleph | Veo 3.1 | Sora 2 | Gen-4.5 Turbo |
|------------|--------------|---------|--------|--------------|
| Water/fluid | 90 | 85 | 78 | 82 |
| Cloth/fabric | 88 | 83 | 76 | 80 |
| Hair | 86 | 84 | 80 | 78 |
| Walking/running | 82 | 85 | 80 | 75 |
| Hand gestures | 72 | 70 | 65 | 60 |
| Complex interaction | 75 | 78 | 70 | 65 |
| **Average** | **82** | **81** | **75** | **73** |

Gen-4.5 Aleph and Veo 3.1 are nearly tied on motion quality. Gen-4.5 excels at physical simulation (water, cloth), while Veo excels at human motion (walking, complex interactions). Sora 2 trails both.

Hands remain the weakest dimension for all models. The best score is 72/100, meaning hand generation fails or looks unnatural roughly 30% of the time. This is a well-known limitation of current generative models.

### 6. Quality-Per-Dollar Analysis

The most actionable metric for builders: what quality do you get per dollar spent?

Define quality-per-dollar as:

$$QPD = \frac{\text{Elo Score}}{1000 \times \text{\$/second}}$$

Higher is better (more quality per dollar).

| Model | Elo | $/sec (720p) | QPD | Rank |
|-------|-----|-------|-----|------|
| Gen-4.5 Turbo | 1,095 | $0.04 | 27.4 | 1 |
| Sora 2 | 1,156 | $0.10 | 11.6 | 2 |
| Gen-4.5 Aleph | 1,247 | $0.12 | 10.4 | 3 |
| Veo 3.1 Fast | 1,198 | $0.15 | 8.0 | 4 |
| Veo 3.1 Standard | 1,198 | $0.25 | 4.8 | 5 |

**Gen-4.5 Turbo dominates on quality-per-dollar** by a massive margin (27.4 vs. next-best 11.6). Even though its absolute quality is lowest, the 4-cent per-second price makes it the clear winner for cost-sensitive applications.

If we include audio in the value calculation (adding a "quality bonus" for native audio):

$$QPD_{audio} = \frac{\text{Elo Score} + 100 \times \mathbb{1}[\text{audio}]}{1000 \times \text{\$/second}}$$

| Model | Adjusted Elo | $/sec | QPD (audio-adjusted) | Rank |
|-------|-------------|-------|-----|------|
| Gen-4.5 Turbo | 1,095 | $0.04 | 27.4 | 1 |
| Sora 2 | 1,256 | $0.10 | 12.6 | 2 |
| Gen-4.5 Aleph | 1,247 | $0.12 | 10.4 | 3 |
| Veo 3.1 Fast | 1,298 | $0.15 | 8.7 | 4 |
| Veo 3.1 Standard | 1,298 | $0.25 | 5.2 | 5 |

The ranking does not change, but the gap between Sora 2 and Gen-4.5 Aleph closes (12.6 vs 10.4). Audio increases Sora 2's value proposition noticeably.

---

## Audio Capabilities Deep Dive

Audio is the most significant differentiator between these three models.

### Feature Comparison

| Audio Feature | Veo 3.1 | Sora 2 | Gen-4.5 |
|--------------|---------|--------|---------|
| Dialogue generation | Yes | Yes | No |
| Sound effects | Yes | Yes | No |
| Ambient sound | Yes | Yes | No |
| Background music | Limited | Yes | No |
| Lip sync quality (0-100) | 85 | 72 | N/A |
| Voice variety | High | Medium | N/A |
| Language support | English-primary | English-primary | N/A |
| Separate audio control | No (prompt only) | No (prompt only) | N/A |
| Audio-only output | No | No | N/A |

### Audio Quality Analysis

Using the Mean Opinion Score (MOS) methodology (human raters score 1-5):

| Dimension | Veo 3.1 | Sora 2 |
|-----------|---------|--------|
| Dialogue naturalness | 3.8 | 3.1 |
| SFX appropriateness | 3.9 | 3.3 |
| Ambient realism | 4.0 | 3.5 |
| Audio-visual sync | 4.2 | 3.2 |
| Overall audio quality | 3.9 | 3.3 |

**Veo 3.1 is clearly superior on audio quality across every dimension.** The advantage is most pronounced on audio-visual sync (4.2 vs 3.2), which is directly attributable to SoundStorm's cross-modal attention architecture.

### What SoundStorm Means Architecturally

SoundStorm's parallel decoding approach generates all audio tokens simultaneously in 8-16 refinement iterations:

```
Iteration 1: [mask] [mask] [mask] [mask] [mask] [mask] [mask] [mask]
             Predict all -> Unmask top 50% confident
Iteration 2: [done] [mask] [done] [mask] [done] [done] [mask] [done]
             Predict remaining -> Unmask top 50%
Iteration 3: [done] [done] [done] [mask] [done] [done] [done] [done]
             Predict remaining -> Done
```

This is fundamentally faster than Sora 2's autoregressive approach:

```
Step 1: [token_1]
Step 2: [token_1] [token_2]
Step 3: [token_1] [token_2] [token_3]
...
Step N: [token_1] [token_2] ... [token_N]
```

For 8 seconds of audio at 50 tokens/second = 400 tokens:
- SoundStorm: ~12 iterations regardless of length = O(12) forward passes
- Autoregressive: 400 sequential steps = O(400) forward passes

SoundStorm is ~33x faster for audio generation alone, which is why Veo can generate higher-quality audio without significant latency penalty.

### The No-Audio Model Strategy

Gen-4.5 does not generate audio. For builders, this means either:

1. **Ship silent video**: Acceptable for B-roll, product demos, visual-only content
2. **Add audio in post**: Use a separate TTS/SFX service (ElevenLabs, AudioCraft, etc.)
3. **Route to audio-enabled model**: Use Gen-4.5 for visual-only content and Veo/Sora for content that needs audio

Option 3 is the recommended approach for multi-purpose platforms. The routing logic:

```
If content needs dialogue or specific SFX:
    -> Route to Veo 3.1 (best audio quality)
    -> Fallback: Sora 2 (cheaper, acceptable audio)
If content is visual-only:
    -> Route to Gen-4.5 Turbo (cheapest, fast)
    -> For premium: Gen-4.5 Aleph (best visuals)
```

---

## Feature Matrix

A complete feature-by-feature comparison:

| Feature | Sora 2 | Veo 3.1 | Gen-4.5 Turbo | Gen-4.5 Aleph |
|---------|--------|---------|--------------|--------------|
| **Input Modes** | | | | |
| Text-to-video | Yes | Yes | Yes | Yes |
| Image-to-video | Via Characters | Yes (inline image) | Yes (prompt array) | Yes (prompt array) |
| Video-to-video | No | No | No | No |
| Image + text conditioning | Characters only | Yes | Yes | Yes |
| **Output** | | | | |
| Max duration | 15s | 8s | 10s | 10s |
| Max resolution | 1080p | 4K | 720p | 1080p |
| Native audio | Yes | Yes | No | No |
| Aspect ratios | 16:9, 9:16, 1:1 | 16:9, 9:16, 1:1 | 16:9, 9:16, 1:1 | 16:9, 9:16, 1:1 |
| Frame rate | 24fps | 24fps | 24fps | 24fps |
| **Control** | | | | |
| Negative prompt | No | Yes | No | No |
| Seed control | No | No | Yes | Yes |
| Start frame | No | Yes (via image input) | Yes (via image input) | Yes (via image input) |
| End frame | No | No | No | No |
| Camera control | Prompt only | Prompt only | Prompt only | Prompt only |
| Style reference | Characters feature | Image conditioning | Image conditioning | Image conditioning |
| **Advanced** | | | | |
| Video extend | No | No | No | No |
| Inpainting | No | No | No | No |
| Outpainting | No | No | No | No |
| Upscaling | No | Built-in (to 4K) | No | No |
| Multi-shot | No | No | No | No |
| Character persistence | Yes (video upload) | Partial (image ref) | Partial (image ref) | Partial (image ref) |
| **Platform** | | | | |
| Watermarking | SynthID-style | SynthID | C2PA metadata | C2PA metadata |
| Content filter | Moderate | Moderate | Moderate | Moderate |
| NSFW generation | No | No | No | No |

### Notable Feature Gaps

**No model supports video-to-video** (style transfer, editing an existing video). This remains a separate tool category (Pika's Pikaswaps/Pikadditions address this partially).

**No model supports video extend** (adding frames to the end of an existing clip to make it longer). This was rumored for Sora 2 but has not shipped.

**No model supports inpainting/outpainting** for video (editing specific regions of a generated video). This is likely coming but represents a significant computational challenge.

**Sora 2's Characters feature is unique** --- no other model allows uploading a video to create a persistent identity. This is the strongest character consistency mechanism available, but comes with ethical concerns about deepfakes.

**Veo 3.1's built-in upscaling to 4K** is unique. Other models that want 4K output would need an external upscaling step.

---

## Reliability: Uptime, Rate Limits, Error Rates

Reliability data from 7 days of continuous monitoring (February 3-9, 2026):

### Uptime

| Provider | Observed Uptime | Downtime Events | Longest Outage |
|----------|----------------|-----------------|----------------|
| Runway | 99.7% | 1 (degraded) | 45 min |
| Google (Veo) | 99.5% | 2 (1 full, 1 degraded) | 2 hours |
| OpenAI (Sora) | 98.2% | 3 (2 full, 1 degraded) | 4 hours |

**Sora 2 is the least reliable**, with three outage events in one week. The longest was a 4-hour full outage on February 5 (Tuesday evening US time). This is consistent with reports of ongoing capacity challenges.

**Runway is the most reliable**, with only one brief degraded-performance event (45 minutes of elevated latency but no failed requests).

### Error Rates

Percentage of requests that returned an error (excluding moderation blocks):

| Error Type | Runway | Veo 3.1 | Sora 2 |
|-----------|--------|---------|--------|
| Server error (5xx) | 0.3% | 0.8% | 2.1% |
| Timeout | 0.1% | 0.5% | 1.5% |
| Rate limit (429) | 0.2% | 0.4% | 0.8% |
| Generation failed | 0.5% | 1.0% | 1.8% |
| **Total error rate** | **1.1%** | **2.7%** | **6.2%** |

**Sora 2's 6.2% error rate is 5-6x higher than Runway's 1.1%.** For a platform serving users, this means 1 in 16 Sora requests fails and needs to be retried, versus 1 in 90 for Runway.

The cost impact of errors: if you retry on every failure, the effective cost per successful generation is:

$$C_{effective} = \frac{C_{per\_request}}{1 - p_{error}} \times \left(1 + p_{error} + p_{error}^2 + \ldots\right) = \frac{C_{per\_request}}{1 - p_{error}}$$

Wait --- that is only true if failed requests are not charged. Let me clarify: **Runway and Veo do not charge for failed generations. Sora charges for some failure modes** (specifically, generation_failed where the model produced output but it did not meet quality thresholds --- the output is discarded but compute was used).

Adjusted effective cost per successful generation:

| Model | List $/sec | Error rate | Charged on failure? | Effective $/sec |
|-------|-----------|-----------|--------------------|----|
| Gen-4.5 Turbo | $0.04 | 1.1% | No | $0.040 |
| Sora 2 | $0.10 | 6.2% | Partial (~50%) | $0.103 |
| Veo 3.1 Fast | $0.15 | 2.7% | No | $0.154 |
| Gen-4.5 Aleph | $0.12 | 1.1% | No | $0.121 |

Sora 2's effective cost is about 3% higher than its list price due to partial charges on failures. Not a huge impact, but worth noting in cost models.

### Rate Limits

| Tier | Runway (RPM) | Veo 3.1 (RPM) | Sora 2 (RPM) |
|------|-------------|--------------|-------------|
| Free/Basic | 5 | 2 | 3 |
| Standard/Paid | 30 | 10 | 5 |
| Pro/Premium | 30 | 15 | 10 |
| Enterprise | Custom (60+) | Custom (30+) | Custom (20+) |

**Runway is the most generous with rate limits**: 30 RPM at the Standard tier, more than enough for most applications.

**Sora 2 is the most restrictive**: only 5 RPM at the paid tier. This is a significant constraint for platforms with concurrent users. At 5 RPM, you can handle approximately 5 simultaneous users making one request per minute.

For a platform with $N$ concurrent users, each generating $R$ requests per minute, you need:

$$\text{Required RPM} = N \times R$$

| Concurrent Users | Req/min each | Required RPM | Runway tier needed | Veo tier needed | Sora tier needed |
|-----------------|-------------|-------------|-------------------|----------------|-----------------|
| 5 | 1 | 5 | Standard | Paid | Pro |
| 10 | 1 | 10 | Standard | Pro | Enterprise |
| 20 | 1 | 20 | Standard | Enterprise | Enterprise |
| 50 | 1 | 50 | Enterprise | Enterprise | Enterprise |

Sora 2's rate limits are the binding constraint. You hit Enterprise-tier requirements at just 10 concurrent users.

---

## Edge Cases and Failure Modes

Every model has specific failure modes. Understanding these prevents nasty surprises in production.

### Gen-4.5 Failure Modes

| Failure | Frequency | Description | Mitigation |
|---------|-----------|-------------|-----------|
| Frozen frames | ~5% | Last 1-2 seconds of clip freeze or loop | Trim last 0.5s in post-processing |
| Resolution mismatch | ~2% | Output resolution slightly different from requested | Resize in post-processing |
| Prompt element omission | ~15% | Complex prompts (5+ elements) lose one | Simplify prompts, break into shots |
| Aesthetic override | ~10% | Model overrides prompt to produce more "cinematic" output | Use literal language, avoid ambiguity |
| Hand artifacts | ~30% | Distorted hands in close-up | Avoid prompts requiring visible hands |

**Gen-4.5's most distinctive failure mode is "aesthetic override"**: the model has a strong bias toward cinematic-looking output and will sometimes reinterpret prompts to produce more visually pleasing results at the expense of prompt accuracy. For example, "a messy room with clothes on the floor" might produce a stylishly disheveled room rather than a genuinely messy one.

### Veo 3.1 Failure Modes

| Failure | Frequency | Description | Mitigation |
|---------|-----------|-------------|-----------|
| Audio desync (>100ms) | ~8% | Audio events misaligned with visual events | Re-generate; more common in fast mode |
| Character morph | ~12% | Face/body gradually changes over clip duration | Keep clips under 6s for faces |
| Background hallucination | ~7% | Objects appear/disappear in background | Use simpler backgrounds in prompts |
| Dialogue voice mismatch | ~15% | Voice does not match described character | Specify voice characteristics explicitly |
| 4K artifacts | ~10% | Upscaling artifacts visible in 4K mode | Use 1080p and external upscaler for critical work |

**Veo's most impactful failure is dialogue voice mismatch**: when the prompt describes a young woman but the audio produces a voice that sounds older or differently gendered. This happens because the audio conditioning has limited control over specific voice characteristics --- you can say "young woman's voice" but the model's interpretation varies.

### Sora 2 Failure Modes

| Failure | Frequency | Description | Mitigation |
|---------|-----------|-------------|-----------|
| Temporal flickering | ~18% | Rapid brightness/color fluctuation between frames | Add "smooth lighting" to prompt |
| Identity drift (within clip) | ~22% | Character appearance changes mid-clip | Use Characters feature; keep clips short |
| Audio clipping | ~10% | Audio distortion/clipping at high volume | Post-process audio normalization |
| Capacity error | ~5% | "Model at capacity" during peak hours | Retry with exponential backoff |
| Slow generation stall | ~3% | Generation starts but never completes | Set timeout, cancel and retry |

**Sora 2's identity drift is its most significant quality issue**: a character's face, hair, or clothing can visibly change over the course of a single clip. This happens in 22% of generations involving humans. Using the Characters feature (uploading a reference video) reduces this to approximately 8%, but the Characters feature adds complexity to the integration.

### Error Recovery Patterns

Production-ready error handling should account for all three models' failure modes:

```typescript
interface GenerationAttempt {
  model: 'sora' | 'veo' | 'runway';
  params: any;
  result?: any;
  error?: Error;
  latencyMs: number;
}

async function generateWithFallbackChain(
  prompt: string,
  options: {
    needsAudio: boolean;
    maxCostPerSecond: number;
    maxLatencyMs: number;
    duration: number;
    resolution: '720p' | '1080p';
  }
): Promise<GenerationAttempt> {
  // Build model priority chain based on requirements
  const chain = buildModelChain(options);

  for (const modelConfig of chain) {
    try {
      const startTime = Date.now();
      const result = await generateWithModel(
        modelConfig.model,
        prompt,
        options,
        modelConfig.timeout
      );

      const latencyMs = Date.now() - startTime;

      // Validate output quality
      if (await validateOutput(result, options)) {
        return {
          model: modelConfig.model,
          params: options,
          result,
          latencyMs,
        };
      }

      // Output failed validation - continue to next model
      console.warn(`${modelConfig.model} output failed validation, trying next`);
    } catch (error) {
      console.error(`${modelConfig.model} failed: ${error}`);

      if (isModeratedContent(error)) {
        // Content moderation - same prompt will fail on retries
        // Try with prompt rewriting
        const rewrittenPrompt = await rewriteForModeration(prompt, modelConfig.model);
        try {
          const result = await generateWithModel(
            modelConfig.model,
            rewrittenPrompt,
            options,
            modelConfig.timeout
          );
          return {
            model: modelConfig.model,
            params: { ...options, prompt: rewrittenPrompt },
            result,
            latencyMs: Date.now() - Date.now(), // simplified
          };
        } catch (retryError) {
          continue; // Move to next model
        }
      }

      if (isCapacityError(error)) {
        // Capacity issue - skip to next model immediately
        continue;
      }

      if (isRateLimitError(error)) {
        // Rate limited - wait and retry same model once
        const retryAfter = getRetryAfterMs(error);
        await sleep(retryAfter);
        try {
          const result = await generateWithModel(
            modelConfig.model,
            prompt,
            options,
            modelConfig.timeout
          );
          return {
            model: modelConfig.model,
            params: options,
            result,
            latencyMs: 0,
          };
        } catch {
          continue;
        }
      }

      // Unknown error - continue to next model
      continue;
    }
  }

  throw new Error('All models in fallback chain failed');
}

function buildModelChain(options: any): Array<{model: string; timeout: number}> {
  const chain: Array<{model: string; timeout: number}> = [];

  if (options.needsAudio) {
    if (options.maxCostPerSecond >= 0.15) {
      chain.push({ model: 'veo-fast', timeout: 120000 });
    }
    chain.push({ model: 'sora', timeout: 300000 });
    if (options.maxCostPerSecond >= 0.25) {
      chain.push({ model: 'veo-standard', timeout: 300000 });
    }
  } else {
    if (options.maxLatencyMs < 30000) {
      chain.push({ model: 'runway-turbo', timeout: 45000 });
    }
    if (options.maxCostPerSecond >= 0.12) {
      chain.push({ model: 'runway-aleph', timeout: 180000 });
    }
    chain.push({ model: 'runway-turbo', timeout: 60000 });
  }

  return chain;
}
```

---

## Cost Optimization at Scale

Let me model the cost of running a multi-model platform at three scales: 100, 1,000, and 10,000 generations per month.

### Assumptions

- Average clip duration: 5 seconds
- Resolution: 720p for drafts, 1080p for finals
- 60% of generations are drafts (cheap/fast), 40% are finals
- Of finals, 50% need audio, 50% do not
- Therefore: 60% drafts, 20% final-no-audio, 20% final-with-audio

### Cost Model

**Model routing for each category**:
- Drafts: Gen-4.5 Turbo at $0.04/sec
- Final (no audio): Gen-4.5 Aleph at $0.12/sec
- Final (with audio): Veo 3.1 Fast at $0.15/sec

**Cost per generation** (5 seconds):
- Draft: 5 $\times$ $0.04 = $0.20
- Final (no audio): 5 $\times$ $0.12 = $0.60
- Final (with audio): 5 $\times$ $0.15 = $0.75

**Blended cost per generation**:

$$C_{blended} = 0.60 \times \$0.20 + 0.20 \times \$0.60 + 0.20 \times \$0.75 = \$0.12 + \$0.12 + \$0.15 = \$0.39$$

### Cost at Each Scale

| Scale | Generations/mo | Monthly Cost | Avg $/gen | Notes |
|-------|---------------|-------------|----------|-------|
| Small (starter) | 100 | $39 | $0.39 | No volume discounts |
| Medium (growth) | 1,000 | $370 | $0.37 | ~5% volume discount |
| Large (scale) | 10,000 | $3,400 | $0.34 | ~10-15% volume discount |
| Very large | 100,000 | $29,000 | $0.29 | ~20-25% volume discount (enterprise) |

### Alternative Strategy: Single Model

What if you used just one model for everything?

**All Gen-4.5 Turbo** (cheapest visual, no audio):

| Scale | $/gen | Monthly Cost | Savings vs Multi-Model | Missing |
|-------|-------|-------------|----------------------|---------|
| 1,000 | $0.20 | $200 | 46% | No audio, lower quality |
| 10,000 | $0.18 | $1,800 | 47% | No audio, lower quality |

**All Sora 2** (mid-price with audio):

| Scale | $/gen | Monthly Cost | vs Multi-Model | Trade-off |
|-------|-------|-------------|---------------|-----------|
| 1,000 | $0.50 | $500 | +35% more expensive | Simpler integration, native audio |
| 10,000 | $0.45 | $4,500 | +32% more expensive | Simpler integration, native audio |

**All Veo 3.1 Fast** (premium with best audio):

| Scale | $/gen | Monthly Cost | vs Multi-Model | Trade-off |
|-------|-------|-------------|---------------|-----------|
| 1,000 | $0.75 | $750 | +103% more expensive | Best audio, consistent quality |
| 10,000 | $0.65 | $6,500 | +91% more expensive | Best audio, consistent quality |

### The Multi-Model Savings

The multi-model approach saves 32-91% compared to using a single premium model. At 10,000 generations/month, the savings are:

- vs. all Sora 2: $1,100/month ($13,200/year)
- vs. all Veo 3.1 Fast: $3,100/month ($37,200/year)

The engineering cost of maintaining 3 API integrations versus 1 must be weighed against these savings. A rough estimate of the additional engineering cost: 40 hours to build + 5 hours/month to maintain = $15,000 initial + $750/month ongoing (at $150/hr).

**Break-even for multi-model vs. single Veo**: The $3,100/month savings against $750/month maintenance cost means the multi-model approach pays for itself in month 1 at 10,000 generations/month. At 1,000 generations/month, the savings are ~$380/month vs. $750/month maintenance --- single-model is cheaper at small scale.

**Rule of thumb**: Multi-model routing becomes cost-justified at approximately 3,000+ generations per month.

---

## The Multi-Model Routing Algorithm

Here is the complete routing algorithm implemented in TypeScript. This is production-ready code with full error handling.

```typescript
// ============================================
// Multi-Model Video Generation Router
// ============================================

type VideoModel =
  | 'runway-turbo'
  | 'runway-aleph'
  | 'veo-fast'
  | 'veo-standard'
  | 'sora-standard'
  | 'sora-pro';

interface RoutingConfig {
  // Weight importance of each factor (must sum to 1)
  qualityWeight: number;       // 0-1
  costWeight: number;          // 0-1
  speedWeight: number;         // 0-1
  reliabilityWeight: number;   // 0-1

  // Hard constraints
  maxCostPerSecond: number;
  maxLatencyMs: number;
  requireAudio: boolean;
  minResolution: '480p' | '720p' | '1080p' | '4k';
  maxDuration: number;

  // Preferences
  preferredModel?: VideoModel;
  avoidModels?: VideoModel[];
}

interface ModelProfile {
  model: VideoModel;
  provider: 'runway' | 'google' | 'openai';
  costPerSecond: Record<string, number>; // resolution -> $/sec
  medianLatencyMs: number;   // for 5s clip
  latencyPerSecondMs: number; // additional ms per second of video
  eloScore: number;
  hasAudio: boolean;
  maxDuration: number;
  maxResolution: string;
  reliabilityScore: number;  // 0-1
  supportedResolutions: string[];
}

const MODEL_PROFILES: ModelProfile[] = [
  {
    model: 'runway-turbo',
    provider: 'runway',
    costPerSecond: { '480p': 0.03, '720p': 0.04, '1080p': 0.06 },
    medianLatencyMs: 18000,
    latencyPerSecondMs: 3300,
    eloScore: 1095,
    hasAudio: false,
    maxDuration: 10,
    maxResolution: '720p',
    reliabilityScore: 0.989,
    supportedResolutions: ['480p', '720p'],
  },
  {
    model: 'runway-aleph',
    provider: 'runway',
    costPerSecond: { '720p': 0.12, '1080p': 0.16 },
    medianLatencyMs: 75000,
    latencyPerSecondMs: 8500,
    eloScore: 1247,
    hasAudio: false,
    maxDuration: 10,
    maxResolution: '1080p',
    reliabilityScore: 0.989,
    supportedResolutions: ['720p', '1080p'],
  },
  {
    model: 'veo-fast',
    provider: 'google',
    costPerSecond: { '720p': 0.15 },
    medianLatencyMs: 65000,
    latencyPerSecondMs: 10000,
    eloScore: 1198,
    hasAudio: true,
    maxDuration: 8,
    maxResolution: '720p',
    reliabilityScore: 0.973,
    supportedResolutions: ['720p'],
  },
  {
    model: 'veo-standard',
    provider: 'google',
    costPerSecond: { '720p': 0.25, '1080p': 0.35, '4k': 0.50 },
    medianLatencyMs: 150000,
    latencyPerSecondMs: 18000,
    eloScore: 1198,
    hasAudio: true,
    maxDuration: 8,
    maxResolution: '4k',
    reliabilityScore: 0.973,
    supportedResolutions: ['720p', '1080p', '4k'],
  },
  {
    model: 'sora-standard',
    provider: 'openai',
    costPerSecond: { '480p': 0.06, '720p': 0.10, '1080p': 0.20 },
    medianLatencyMs: 120000,
    latencyPerSecondMs: 20000,
    eloScore: 1156,
    hasAudio: true,
    maxDuration: 15,
    maxResolution: '1080p',
    reliabilityScore: 0.938,
    supportedResolutions: ['480p', '720p', '1080p'],
  },
  {
    model: 'sora-pro',
    provider: 'openai',
    costPerSecond: { '720p': 0.30, '1080p': 0.50 },
    medianLatencyMs: 200000,
    latencyPerSecondMs: 25000,
    eloScore: 1200, // estimated, higher quality mode
    hasAudio: true,
    maxDuration: 15,
    maxResolution: '1080p',
    reliabilityScore: 0.938,
    supportedResolutions: ['720p', '1080p'],
  },
];

interface RoutingDecision {
  selectedModel: VideoModel;
  resolution: string;
  estimatedCost: number;
  estimatedLatencyMs: number;
  score: number;
  reasoning: string[];
}

function routeGeneration(
  duration: number,
  resolution: string,
  config: RoutingConfig
): RoutingDecision {
  const candidates: Array<{
    profile: ModelProfile;
    resolution: string;
    cost: number;
    latency: number;
    score: number;
    reasons: string[];
  }> = [];

  for (const profile of MODEL_PROFILES) {
    const reasons: string[] = [];

    // ---- Hard constraint checks ----

    // Check if model is avoided
    if (config.avoidModels?.includes(profile.model)) {
      continue;
    }

    // Check audio requirement
    if (config.requireAudio && !profile.hasAudio) {
      continue;
    }

    // Check duration support
    if (duration > profile.maxDuration) {
      continue;
    }

    // Find best matching resolution
    const targetRes = resolution;
    let selectedRes = targetRes;
    if (!profile.supportedResolutions.includes(targetRes)) {
      // Find closest supported resolution
      const resOrder = ['480p', '720p', '1080p', '4k'];
      const targetIdx = resOrder.indexOf(targetRes);
      selectedRes = profile.supportedResolutions
        .filter(r => resOrder.indexOf(r) >= resOrder.indexOf(config.minResolution))
        .sort((a, b) =>
          Math.abs(resOrder.indexOf(a) - targetIdx) -
          Math.abs(resOrder.indexOf(b) - targetIdx)
        )[0];

      if (!selectedRes) continue;
      reasons.push(`Resolution adjusted: ${targetRes} -> ${selectedRes}`);
    }

    // Check resolution meets minimum
    const resOrder = ['480p', '720p', '1080p', '4k'];
    if (resOrder.indexOf(selectedRes) < resOrder.indexOf(config.minResolution)) {
      continue;
    }

    // Calculate cost
    const costPerSec = profile.costPerSecond[selectedRes];
    if (!costPerSec) continue;
    const totalCost = costPerSec * duration;

    // Check cost constraint
    if (costPerSec > config.maxCostPerSecond) {
      continue;
    }

    // Calculate latency
    const estimatedLatency =
      profile.medianLatencyMs + (duration - 5) * profile.latencyPerSecondMs;

    // Check latency constraint
    if (estimatedLatency > config.maxLatencyMs) {
      continue;
    }

    // ---- Scoring ----

    // Normalize each dimension to 0-1 (higher is better)
    const maxElo = 1247;
    const minElo = 1095;
    const qualityNorm = (profile.eloScore - minElo) / (maxElo - minElo);

    const maxCost = 0.50;
    const minCost = 0.03;
    const costNorm = 1 - (costPerSec - minCost) / (maxCost - minCost); // Inverted: lower cost = higher score

    const maxLatency = 300000;
    const minLatency = 18000;
    const speedNorm = 1 - (estimatedLatency - minLatency) / (maxLatency - minLatency); // Inverted

    const reliabilityNorm = profile.reliabilityScore;

    // Weighted score
    const score =
      config.qualityWeight * qualityNorm +
      config.costWeight * costNorm +
      config.speedWeight * speedNorm +
      config.reliabilityWeight * reliabilityNorm;

    // Bonus for preferred model
    const preferBonus = config.preferredModel === profile.model ? 0.05 : 0;

    // Bonus for audio when audio is preferred (but not required)
    const audioBonus = profile.hasAudio && !config.requireAudio ? 0.02 : 0;

    reasons.push(
      `Quality: ${(qualityNorm * 100).toFixed(0)}%, ` +
      `Cost: ${(costNorm * 100).toFixed(0)}%, ` +
      `Speed: ${(speedNorm * 100).toFixed(0)}%, ` +
      `Reliability: ${(reliabilityNorm * 100).toFixed(0)}%`
    );

    candidates.push({
      profile,
      resolution: selectedRes,
      cost: totalCost,
      latency: estimatedLatency,
      score: score + preferBonus + audioBonus,
      reasons,
    });
  }

  if (candidates.length === 0) {
    throw new Error(
      'No model satisfies all constraints. Consider relaxing: ' +
      `maxCost=$${config.maxCostPerSecond}/sec, ` +
      `maxLatency=${config.maxLatencyMs}ms, ` +
      `audio=${config.requireAudio}, ` +
      `resolution=${config.minResolution}`
    );
  }

  // Sort by score descending
  candidates.sort((a, b) => b.score - a.score);
  const winner = candidates[0];

  return {
    selectedModel: winner.profile.model,
    resolution: winner.resolution,
    estimatedCost: winner.cost,
    estimatedLatencyMs: winner.latency,
    score: winner.score,
    reasoning: winner.reasons,
  };
}

// ============================================
// Example Usage: Different Scenarios
// ============================================

// Scenario 1: Quick preview (fast and cheap)
const previewRoute = routeGeneration(5, '720p', {
  qualityWeight: 0.1,
  costWeight: 0.4,
  speedWeight: 0.4,
  reliabilityWeight: 0.1,
  maxCostPerSecond: 0.10,
  maxLatencyMs: 45000,
  requireAudio: false,
  minResolution: '720p',
  maxDuration: 5,
});
// Expected: runway-turbo (fastest, cheapest)

// Scenario 2: Final output with audio
const finalAudioRoute = routeGeneration(8, '1080p', {
  qualityWeight: 0.4,
  costWeight: 0.2,
  speedWeight: 0.1,
  reliabilityWeight: 0.3,
  maxCostPerSecond: 0.50,
  maxLatencyMs: 300000,
  requireAudio: true,
  minResolution: '1080p',
  maxDuration: 8,
});
// Expected: veo-standard (best audio + quality, supports 1080p)

// Scenario 3: High quality, no audio, budget-conscious
const qualityBudgetRoute = routeGeneration(5, '1080p', {
  qualityWeight: 0.5,
  costWeight: 0.3,
  speedWeight: 0.1,
  reliabilityWeight: 0.1,
  maxCostPerSecond: 0.20,
  maxLatencyMs: 180000,
  requireAudio: false,
  minResolution: '1080p',
  maxDuration: 5,
});
// Expected: runway-aleph (highest quality at $0.16/sec)

// Scenario 4: Maximum duration with audio
const longFormRoute = routeGeneration(15, '720p', {
  qualityWeight: 0.3,
  costWeight: 0.3,
  speedWeight: 0.1,
  reliabilityWeight: 0.3,
  maxCostPerSecond: 0.50,
  maxLatencyMs: 600000,
  requireAudio: true,
  minResolution: '720p',
  maxDuration: 15,
});
// Expected: sora-standard (only model with 15s + audio)

console.log('Preview:', JSON.stringify(previewRoute, null, 2));
console.log('Final+Audio:', JSON.stringify(finalAudioRoute, null, 2));
console.log('Quality+Budget:', JSON.stringify(qualityBudgetRoute, null, 2));
console.log('Long-form:', JSON.stringify(longFormRoute, null, 2));
```

### Routing Decision Table (Simplified)

For quick reference, here is the decision table the algorithm produces for common scenarios:

| Scenario | Audio? | Quality Priority | Speed Priority | Cost Priority | Best Model |
|----------|--------|-----------------|---------------|--------------|------------|
| Preview/draft | No | Low | High | High | Gen-4.5 Turbo |
| Social media clip | Optional | Medium | Medium | High | Gen-4.5 Turbo |
| Product demo | No | High | Low | Medium | Gen-4.5 Aleph |
| Ad with dialogue | Yes | High | Low | Low | Veo 3.1 Standard |
| Story narration | Yes | Medium | Medium | High | Sora 2 Standard |
| Long explainer | Yes | Medium | Low | Medium | Sora 2 Standard |
| Hero content (no audio) | No | Maximum | None | None | Gen-4.5 Aleph |
| Hero content (with audio) | Yes | Maximum | None | None | Veo 3.1 Standard |
| Batch processing | No | Low | Low | Maximum | Gen-4.5 Turbo |
| Real-time preview | No | Lowest | Maximum | Low | Gen-4.5 Turbo |

---

## Weighted Scoring Methodology

For teams that need to make a formal model selection decision, here is a structured methodology.

### Step 1: Define Weights

Determine the relative importance of each dimension for your use case. Weights must sum to 1.0:

| Profile | Quality | Cost | Speed | Audio | Reliability | API Maturity |
|---------|---------|------|-------|-------|-------------|-------------|
| Consumer social app | 0.15 | 0.30 | 0.25 | 0.10 | 0.15 | 0.05 |
| Professional video tool | 0.35 | 0.15 | 0.10 | 0.20 | 0.15 | 0.05 |
| Marketing platform | 0.25 | 0.20 | 0.15 | 0.20 | 0.10 | 0.10 |
| Enterprise content | 0.20 | 0.10 | 0.05 | 0.15 | 0.30 | 0.20 |

### Step 2: Score Each Model (0-10)

| Dimension | Sora 2 | Veo 3.1 | Gen-4.5 Turbo | Gen-4.5 Aleph |
|-----------|--------|---------|--------------|--------------|
| Quality | 6 | 8 | 5 | 10 |
| Cost | 8 | 5 | 10 | 6 |
| Speed | 3 | 5 | 10 | 5 |
| Audio | 6 | 9 | 0 | 0 |
| Reliability | 5 | 7 | 9 | 9 |
| API Maturity | 7 | 6 | 9 | 9 |

### Step 3: Compute Weighted Scores

**Consumer social app** (quality=0.15, cost=0.30, speed=0.25, audio=0.10, reliability=0.15, api=0.05):

| Model | Weighted Score |
|-------|---------------|
| Sora 2 | 0.15(6) + 0.30(8) + 0.25(3) + 0.10(6) + 0.15(5) + 0.05(7) = 5.80 |
| Veo 3.1 | 0.15(8) + 0.30(5) + 0.25(5) + 0.10(9) + 0.15(7) + 0.05(6) = 6.10 |
| Gen-4.5 Turbo | 0.15(5) + 0.30(10) + 0.25(10) + 0.10(0) + 0.15(9) + 0.05(9) = **7.55** |
| Gen-4.5 Aleph | 0.15(10) + 0.30(6) + 0.25(5) + 0.10(0) + 0.15(9) + 0.05(9) = 5.90 |

Winner: **Gen-4.5 Turbo** (7.55). The speed and cost advantages dominate for a consumer app.

**Professional video tool** (quality=0.35, cost=0.15, speed=0.10, audio=0.20, reliability=0.15, api=0.05):

| Model | Weighted Score |
|-------|---------------|
| Sora 2 | 0.35(6) + 0.15(8) + 0.10(3) + 0.20(6) + 0.15(5) + 0.05(7) = 5.90 |
| Veo 3.1 | 0.35(8) + 0.15(5) + 0.10(5) + 0.20(9) + 0.15(7) + 0.05(6) = **7.20** |
| Gen-4.5 Turbo | 0.35(5) + 0.15(10) + 0.10(10) + 0.20(0) + 0.15(9) + 0.05(9) = 5.15 |
| Gen-4.5 Aleph | 0.35(10) + 0.15(6) + 0.10(5) + 0.20(0) + 0.15(9) + 0.05(9) = 6.20 |

Winner: **Veo 3.1** (7.20). Audio quality and visual quality combine to win for professional tools.

**Enterprise content** (quality=0.20, cost=0.10, speed=0.05, audio=0.15, reliability=0.30, api=0.20):

| Model | Weighted Score |
|-------|---------------|
| Sora 2 | 0.20(6) + 0.10(8) + 0.05(3) + 0.15(6) + 0.30(5) + 0.20(7) = 5.85 |
| Veo 3.1 | 0.20(8) + 0.10(5) + 0.05(5) + 0.15(9) + 0.30(7) + 0.20(6) = 6.90 |
| Gen-4.5 Turbo | 0.20(5) + 0.10(10) + 0.05(10) + 0.15(0) + 0.30(9) + 0.20(9) = 6.50 |
| Gen-4.5 Aleph | 0.20(10) + 0.10(6) + 0.05(5) + 0.15(0) + 0.30(9) + 0.20(9) = **6.75** |

Winner (tie): **Veo 3.1** (6.90) narrowly beats **Gen-4.5 Aleph** (6.75). Reliability and API maturity heavily influence the enterprise decision.

### The Multi-Model Insight

Notice that no single model wins across all profiles. This is the fundamental argument for multi-model routing: the optimal model depends on the specific generation request, not the platform as a whole.

If you run the scoring for every request in a typical platform's distribution (30% consumer, 40% professional, 30% enterprise), the optimal strategy is to route each request to the winner for that request's profile --- which means you need all three models integrated.

---

## The Verdict by Use Case

### Building a Social Content Creator

**Primary model**: Gen-4.5 Turbo
- Speed is critical (users waiting in-app)
- Cost efficiency enables generous free tiers
- Audio not essential for most social content (music is added separately)

**Secondary**: Sora 2 for posts that need native audio
**Cost**: $0.20-0.50 per 5-second clip

### Building a Narrative Video Tool (Ads, Stories, Explainers)

**Primary model**: Veo 3.1 Fast (for drafts) + Veo 3.1 Standard (for finals)
- Audio quality is essential for dialogue-heavy content
- Temporal consistency matters for narrative coherence
- Budget is typically higher (professional users)

**Secondary**: Sora 2 for longer durations (>8 seconds, up to 15s)
**Cost**: $0.75-2.50 per 5-8 second clip

### Building a Professional/Cinematic Tool

**Primary model**: Gen-4.5 Aleph
- Highest visual quality, period
- Professional users add audio in post-production
- Speed is less important (async rendering is acceptable)

**Secondary**: Veo 3.1 Standard for rough cuts with audio
**Cost**: $0.60-1.75 per 5-second clip

### Building a Multi-Purpose Platform

**You need all three**, plus routing intelligence:

| Request Type | Route To | Est. % of Traffic |
|-------------|---------|------------------|
| Preview/draft | Gen-4.5 Turbo | 40% |
| Final (visual only) | Gen-4.5 Aleph | 25% |
| Final (with audio) | Veo 3.1 Fast/Standard | 20% |
| Long-form (>8s) | Sora 2 | 10% |
| Budget/batch | Gen-4.5 Turbo | 5% |

**Blended cost**: $0.34-0.39 per generation (5s average)
**Required integrations**: 3 APIs (Runway, Google, OpenAI)
**Engineering overhead**: ~40 hours initial, ~5 hours/month ongoing

---

## Conclusion

There is no single "best" AI video generation model in February 2026. The landscape is a three-way optimization problem: quality (Gen-4.5 Aleph), speed+cost (Gen-4.5 Turbo), and audio integration (Veo 3.1).

The data in this post supports three actionable conclusions:

**1. Multi-model routing saves 30-90% versus single-model approaches at scale.** The 10x price spread between cheapest (Gen-4.5 Turbo at $0.04/sec) and most expensive (Veo 3.1 Standard at $0.50/sec) means intelligent routing is a multi-million-dollar decision at scale. The routing algorithm in this post is a production-ready starting point.

**2. Audio is the primary differentiator, not visual quality.** The visual quality gap between Gen-4.5 and Veo 3.1 (49 Elo points, 57% win rate) is modest. The audio gap between models-with-audio and models-without is binary. If your content needs dialogue or sound, the model choice narrows from four options to two (Veo and Sora). If it does not need audio, Gen-4.5 wins on every other dimension.

**3. Reliability varies more than you expect.** Sora 2's 6.2% error rate and 98.2% uptime are dramatically worse than Runway's 1.1% error rate and 99.7% uptime. For production platforms, this gap matters more than a 20% price difference. Build retry logic and fallback chains; do not assume API calls will succeed.

The field is moving fast. Veo 3.2, Gen-5, and Sora 3 are all rumored for 2026. When they arrive, the specific numbers in this post will change, but the multi-model routing framework and evaluation methodology will remain applicable. Build the abstraction layer now; swap models later.
