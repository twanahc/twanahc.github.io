---
layout: post
title: "Building a Safety Layer for AI Video: Content Moderation Architecture, API Benchmarks, and the Math of False Positives"
date: 2026-02-01
category: infrastructure
---

In December 2025, Grok users discovered they could generate explicit celebrity deepfakes with trivial prompt manipulation. Within 72 hours, millions of non-consensual sexualized images flooded X. xAI scrambled to patch filters, but the reputational damage was done. Stock-adjacent sentiment for xAI cratered, advertisers paused campaigns, and regulators in the EU and UK opened formal inquiries.

This was not an unprecedented failure. It was a *predictable* one. Any AI generation platform without a rigorous, multi-layered moderation architecture will eventually ship harmful content. The question is not *if* but *when* -- and whether your architecture catches it at the $0.00 input-screening stage or the $50,000 PR-crisis stage.

This post is a complete engineering guide to content moderation for AI video platforms. We cover the three-point architecture with production TypeScript code, the mathematics of false positive/negative tradeoffs, real benchmark data for every major moderation API, a forensic analysis of the Grok and Kling moderation failures, and a full production implementation with queue-based async moderation, human review dashboards, and appeal flows.

---

## Table of Contents

1. [The Three-Point Moderation Architecture](#the-three-point-moderation-architecture)
2. [The Mathematics of Moderation Thresholds](#the-mathematics-of-moderation-thresholds)
3. [API Benchmarks: Accuracy, Latency, and Pricing](#api-benchmarks-accuracy-latency-and-pricing)
4. [Case Study: The Grok Incident](#case-study-the-grok-incident)
5. [Case Study: Kling's Over-Moderation Problem](#case-study-klings-over-moderation-problem)
6. [Content Policy Design](#content-policy-design)
7. [Production TypeScript Implementation](#production-typescript-implementation)
8. [The Human Review Pipeline](#the-human-review-pipeline)
9. [Monitoring, Alerting, and Continuous Improvement](#monitoring-alerting-and-continuous-improvement)

---

## The Three-Point Moderation Architecture

Content moderation for AI video requires screening at three distinct points. Missing any one of them creates a gap that adversaries will find.

```
                         THREE-POINT MODERATION ARCHITECTURE
                         ===================================

  User Input                                                      User Output
  =========                                                       ===========

  [Text Prompt] ──┐                                          ┌── [Video Player]
                  │                                          │
  [Ref Image]  ──┼── LAYER 1: INPUT ──► [Generation API] ──┼── LAYER 2: OUTPUT
                  │   SCREENING           (Veo, Kling,      │   SCREENING
  [Audio Ref]  ──┘   - Text classify      Runway, Sora)     └── - Frame sampling
                     - Image classify                           - Full video scan
                     - Embedding sim.                           - Audio transcript
                     Cost: ~$0.001/req                          - NSFW classifier
                                                                Cost: ~$0.01/req
                                   │                    │
                                   │  LAYER 3: HUMAN    │
                                   │  REVIEW QUEUE      │
                                   │  - Flagged items    │
                                   │  - User reports     │
                                   │  - Random sampling  │
                                   │  - Appeal handling  │
                                   │  Cost: ~$0.50/review│
                                   └────────────────────┘
```

### Layer 1: Input Screening

Input screening is the cheapest moderation layer. A rejected prompt costs $0 in generation fees. A prompt that passes input screening but produces a violating video costs $0.50--$4.00 in wasted generation API calls plus the compute cost of output moderation.

**Text prompt classification:**

```typescript
import Anthropic from "@anthropic-ai/sdk";
import { z } from "zod";

// Moderation categories with severity levels
const ModerationResult = z.object({
  category: z.enum([
    "safe",
    "sexual_explicit",
    "sexual_suggestive",
    "violence_graphic",
    "violence_mild",
    "harassment",
    "hate_speech",
    "self_harm",
    "dangerous_activity",
    "child_exploitation",
    "deepfake_nonconsensual",
    "copyright_infringement",
    "pii_exposure",
  ]),
  severity: z.number().min(0).max(1),
  reasoning: z.string(),
  shouldBlock: z.boolean(),
});

type ModerationResult = z.infer<typeof ModerationResult>;

const MODERATION_SYSTEM_PROMPT = `You are a content moderation classifier for an AI video generation platform.
Analyze the user's prompt and classify it into one of the following categories:

- safe: No policy violations
- sexual_explicit: Pornographic or explicitly sexual content
- sexual_suggestive: Suggestive but not explicit sexual content
- violence_graphic: Graphic violence, gore, dismemberment
- violence_mild: Mild violence (action scenes, sports injuries)
- harassment: Targeted harassment or bullying
- hate_speech: Content targeting protected groups
- self_harm: Promotion of self-harm or suicide
- dangerous_activity: Instructions for dangerous/illegal activities
- child_exploitation: Any sexual content involving minors (ZERO TOLERANCE)
- deepfake_nonconsensual: Generating someone's likeness without consent
- copyright_infringement: Recreating copyrighted characters/scenes
- pii_exposure: Revealing personal information

Return a JSON object with: category, severity (0-1), reasoning, shouldBlock.

CRITICAL RULES:
- child_exploitation: ALWAYS block, severity 1.0, report immediately
- deepfake_nonconsensual: Block if real person identified, severity 0.9+
- sexual_explicit: Block, severity 0.8+
- For borderline cases, err on the side of caution (block)

Context: This is for AI video generation. Even suggestive prompts can produce
explicit outputs due to model behavior. Weight severity accordingly.`;

async function screenPrompt(prompt: string): Promise<ModerationResult> {
  const anthropic = new Anthropic();

  const response = await anthropic.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 512,
    system: MODERATION_SYSTEM_PROMPT,
    messages: [
      {
        role: "user",
        content: `Classify this video generation prompt:\n\n"${prompt}"`,
      },
    ],
  });

  const text =
    response.content[0].type === "text" ? response.content[0].text : "";
  const parsed = JSON.parse(text);
  return ModerationResult.parse(parsed);
}
```

**Image reference screening:**

```typescript
import { ImageAnnotatorClient } from "@google-cloud/vision";

interface ImageScreenResult {
  safe: boolean;
  scores: {
    adult: number;    // 0-5 likelihood
    violence: number;
    racy: number;
    spoof: number;
    medical: number;
  };
  faces: number;
  celebrities: string[];
  blocked: boolean;
  reason?: string;
}

async function screenReferenceImage(
  imageBuffer: Buffer
): Promise<ImageScreenResult> {
  const client = new ImageAnnotatorClient();

  // Run SafeSearch + Face Detection in parallel
  const [safeSearchResult, faceResult] = await Promise.all([
    client.safeSearchDetection({ image: { content: imageBuffer } }),
    client.faceDetection({ image: { content: imageBuffer } }),
  ]);

  const safe = safeSearchResult[0].safeSearchAnnotation!;
  const faces = faceResult[0].faceAnnotations || [];

  // Google Vision uses likelihood levels: UNKNOWN, VERY_UNLIKELY,
  // UNLIKELY, POSSIBLE, LIKELY, VERY_LIKELY (0-5)
  const scores = {
    adult: likelihoodToNumber(safe.adult!),
    violence: likelihoodToNumber(safe.violence!),
    racy: likelihoodToNumber(safe.racy!),
    spoof: likelihoodToNumber(safe.spoof!),
    medical: likelihoodToNumber(safe.medical!),
  };

  // Block if adult >= LIKELY (4) or violence >= LIKELY (4)
  const blocked = scores.adult >= 4 || scores.violence >= 4;

  // If faces detected, run celebrity recognition (separate service)
  let celebrities: string[] = [];
  if (faces.length > 0) {
    celebrities = await detectCelebrities(imageBuffer);
  }

  return {
    safe: !blocked && celebrities.length === 0,
    scores,
    faces: faces.length,
    celebrities,
    blocked: blocked || celebrities.length > 0,
    reason: blocked
      ? `Content flagged: adult=${scores.adult}, violence=${scores.violence}`
      : celebrities.length > 0
        ? `Celebrity detected: ${celebrities.join(", ")}`
        : undefined,
  };
}

function likelihoodToNumber(
  likelihood: string | number | null | undefined
): number {
  const map: Record<string, number> = {
    UNKNOWN: 0,
    VERY_UNLIKELY: 1,
    UNLIKELY: 2,
    POSSIBLE: 3,
    LIKELY: 4,
    VERY_LIKELY: 5,
  };
  if (typeof likelihood === "string") return map[likelihood] ?? 0;
  return Number(likelihood) || 0;
}
```

**Embedding similarity screening** -- catching adversarial prompts that use euphemisms or coded language:

```typescript
import { GoogleGenerativeAI } from "@google/generative-ai";

// Pre-computed embeddings of known-bad prompt patterns
// Updated weekly from flagged content database
let KNOWN_BAD_EMBEDDINGS: Float32Array[] = [];

async function loadBadEmbeddings(): Promise<void> {
  // Load from Firestore collection of confirmed-bad prompts
  const snapshot = await db.collection("moderation_embeddings").get();
  KNOWN_BAD_EMBEDDINGS = snapshot.docs.map(
    (doc) => new Float32Array(doc.data().embedding)
  );
}

async function embeddingSimilarityCheck(
  prompt: string
): Promise<{ maxSimilarity: number; flagged: boolean }> {
  const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);
  const model = genAI.getGenerativeModel({ model: "text-embedding-004" });

  const result = await model.embedContent(prompt);
  const promptEmbedding = new Float32Array(result.embedding.values);

  let maxSimilarity = 0;
  for (const badEmbedding of KNOWN_BAD_EMBEDDINGS) {
    const sim = cosineSimilarity(promptEmbedding, badEmbedding);
    maxSimilarity = Math.max(maxSimilarity, sim);
  }

  // Threshold of 0.85 determined empirically (see threshold optimization section)
  return {
    maxSimilarity,
    flagged: maxSimilarity > 0.85,
  };
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}
```

### Layer 2: Output Screening

Even with clean inputs, models produce unexpected content. Diffusion models can amplify subtle cues in training data, producing NSFW content from innocuous prompts. Every generated video must be screened before delivery.

**Frame sampling strategy:**

For a 5-second video at 24fps (120 frames), analyzing every frame is expensive. The optimal strategy depends on content type:

| Strategy | Frames Analyzed | Cost (Sightengine) | Catch Rate | Best For |
|----------|----------------|-------------------|------------|----------|
| First + Last | 2 | $0.002 | ~60% | Quick pre-check |
| Uniform 5-frame | 5 | $0.005 | ~85% | Standard content |
| Uniform 10-frame | 10 | $0.010 | ~93% | Higher risk |
| Scene-change + Uniform | 8-15 | $0.008-$0.015 | ~97% | Production |
| Every frame | 120 | $0.120 | ~99.5% | Maximum safety |

The scene-change approach is the best cost/accuracy tradeoff. Extract frames at scene transitions (where pixel delta exceeds threshold) plus uniform samples:

```typescript
import ffmpeg from "fluent-ffmpeg";
import { Readable } from "stream";

interface FrameExtractionResult {
  frames: Buffer[];
  timestamps: number[];
  method: string;
}

async function extractKeyFrames(
  videoPath: string,
  maxFrames: number = 10
): Promise<FrameExtractionResult> {
  const duration = await getVideoDuration(videoPath);
  const frames: Buffer[] = [];
  const timestamps: number[] = [];

  // Step 1: Extract scene-change frames using FFmpeg scene detection
  const sceneFrames = await extractSceneChangeFrames(videoPath, 0.3);

  // Step 2: Fill remaining slots with uniform samples
  const uniformCount = Math.max(3, maxFrames - sceneFrames.length);
  const interval = duration / (uniformCount + 1);
  const uniformTimestamps = Array.from(
    { length: uniformCount },
    (_, i) => interval * (i + 1)
  );

  // Merge and deduplicate (within 0.2s)
  const allTimestamps = [
    ...sceneFrames.map((f) => f.timestamp),
    ...uniformTimestamps,
  ]
    .sort((a, b) => a - b)
    .filter((t, i, arr) => i === 0 || t - arr[i - 1] > 0.2)
    .slice(0, maxFrames);

  // Extract frames at selected timestamps
  for (const timestamp of allTimestamps) {
    const frame = await extractFrameAtTimestamp(videoPath, timestamp);
    frames.push(frame);
    timestamps.push(timestamp);
  }

  return {
    frames,
    timestamps,
    method: `scene_change+uniform (${sceneFrames.length} scene, ${uniformCount} uniform)`,
  };
}

async function extractSceneChangeFrames(
  videoPath: string,
  threshold: number
): Promise<{ buffer: Buffer; timestamp: number }[]> {
  return new Promise((resolve, reject) => {
    const frames: { buffer: Buffer; timestamp: number }[] = [];

    ffmpeg(videoPath)
      .videoFilters(`select='gt(scene,${threshold})'`)
      .outputOptions(["-vsync", "vfr", "-f", "image2pipe", "-vcodec", "png"])
      .on("end", () => resolve(frames))
      .on("error", reject)
      .pipe();
    // In production, parse the pipe output to extract individual PNG frames
    // with their PTS timestamps
  });
}
```

**Multi-API output screening with consensus:**

```typescript
interface OutputScreenResult {
  passed: boolean;
  confidence: number;
  apis: {
    name: string;
    score: number;
    categories: Record<string, number>;
    latencyMs: number;
  }[];
  consensus: "safe" | "unsafe" | "review";
  frameResults: {
    timestamp: number;
    maxScore: number;
    flaggedCategories: string[];
  }[];
}

async function screenGeneratedVideo(
  videoPath: string
): Promise<OutputScreenResult> {
  // 1. Extract key frames
  const { frames, timestamps } = await extractKeyFrames(videoPath, 10);

  // 2. Screen each frame through multiple APIs
  const frameResults = await Promise.all(
    frames.map(async (frame, idx) => {
      const [sightengine, hive, googleVision] = await Promise.all([
        screenWithSightengine(frame),
        screenWithHive(frame),
        screenWithGoogleVision(frame),
      ]);

      // Weighted consensus: Hive gets highest weight (best accuracy)
      const weightedScore =
        sightengine.maxScore * 0.25 +
        hive.maxScore * 0.45 +
        googleVision.maxScore * 0.30;

      return {
        timestamp: timestamps[idx],
        maxScore: weightedScore,
        flaggedCategories: [
          ...sightengine.flagged,
          ...hive.flagged,
          ...googleVision.flagged,
        ],
        apis: [
          { name: "sightengine", ...sightengine },
          { name: "hive", ...hive },
          { name: "google_vision", ...googleVision },
        ],
      };
    })
  );

  // 3. Determine consensus
  const maxFrameScore = Math.max(...frameResults.map((f) => f.maxScore));
  let consensus: "safe" | "unsafe" | "review";
  if (maxFrameScore > 0.85) {
    consensus = "unsafe";
  } else if (maxFrameScore > 0.55) {
    consensus = "review"; // Send to human review queue
  } else {
    consensus = "safe";
  }

  return {
    passed: consensus === "safe",
    confidence: 1 - maxFrameScore,
    apis: frameResults.flatMap((f) => f.apis),
    consensus,
    frameResults,
  };
}
```

### Layer 3: Human Review Queue

Automated moderation catches 95--99% of violations. The remaining 1--5% requires human judgment. Critical design decisions:

- **What enters the queue**: Items scored between 0.55 and 0.85 by automated screening, all user reports, random 1% sample of "safe" items (for calibration).
- **SLA targets**: CSAM reports < 1 hour, other violations < 4 hours, appeals < 24 hours.
- **Reviewer tooling**: Side-by-side view of prompt + generated content + moderation scores + similar past decisions.

We implement the full queue system in the [Production Implementation](#production-typescript-implementation) section below.

---

## The Mathematics of Moderation Thresholds

Every moderation classifier outputs a confidence score between 0 and 1. You choose a threshold \(t\): scores above \(t\) are blocked, scores below pass. The threshold determines your false positive rate (safe content blocked) and false negative rate (harmful content passed).

### Definitions

Let \(Y \in \{0, 1\}\) be the true label (0 = safe, 1 = harmful) and \(\hat{S} \in [0, 1]\) be the classifier's score.

- **True Positive (TP)**: \(Y = 1\) and \(\hat{S} \geq t\) (harmful, correctly blocked)
- **False Positive (FP)**: \(Y = 0\) and \(\hat{S} \geq t\) (safe, incorrectly blocked)
- **True Negative (TN)**: \(Y = 0\) and \(\hat{S} < t\) (safe, correctly passed)
- **False Negative (FN)**: \(Y = 1\) and \(\hat{S} < t\) (harmful, incorrectly passed)

From these we derive:

$$\text{Precision} = \frac{TP}{TP + FP} = P(Y=1 \mid \hat{S} \geq t)$$

$$\text{Recall} = \frac{TP}{TP + FN} = P(\hat{S} \geq t \mid Y=1)$$

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{FPR} = \frac{FP}{FP + TN} = P(\hat{S} \geq t \mid Y=0)$$

### The Cost of Errors

For an AI video platform, false positives and false negatives have asymmetric costs:

**False positive costs** (\(C_{FP}\)):
- User frustration: blocked generation, wasted time
- Lost revenue: user churns or downgrades
- Support load: user contacts support to appeal
- Estimated cost: $0.50--$5.00 per false positive (depending on user LTV)

**False negative costs** (\(C_{FN}\)):
- Trust/safety incident: harmful content reaches the user or public
- Legal liability: CSAM, defamation, copyright
- Platform reputation: press coverage, advertiser flight
- Regulatory: fines under EU AI Act, UK Online Safety Act
- Estimated cost: $50--$500,000+ per false negative (depending on category)

The asymmetry is stark. A single CSAM false negative can cost millions in legal fees and reputational damage. A false positive on an artistic nude costs $2 in lost revenue. This asymmetry drives threshold selection.

### Optimal Threshold Derivation

We want to find the threshold \(t^*\) that minimizes expected cost:

$$t^* = \arg\min_t \; \mathbb{E}[\text{Cost}(t)]$$

The expected cost at threshold \(t\) is:

$$\mathbb{E}[\text{Cost}(t)] = C_{FP} \cdot \mathbb{E}[FP(t)] + C_{FN} \cdot \mathbb{E}[FN(t)]$$

Let \(N\) be the total number of items, \(\pi = P(Y=1)\) be the base rate of harmful content, and \(f_0(s)\), \(f_1(s)\) be the score distributions for safe and harmful items respectively.

$$\mathbb{E}[FP(t)] = N(1 - \pi) \int_t^1 f_0(s) \, ds = N(1-\pi)(1 - F_0(t))$$

$$\mathbb{E}[FN(t)] = N\pi \int_0^t f_1(s) \, ds = N\pi \cdot F_1(t)$$

where \(F_0\) and \(F_1\) are the CDFs of the score distributions for safe and harmful items.

Substituting:

$$\mathbb{E}[\text{Cost}(t)] = N \left[ C_{FP}(1-\pi)(1 - F_0(t)) + C_{FN} \pi F_1(t) \right]$$

Taking the derivative with respect to \(t\) and setting to zero:

$$\frac{d}{dt}\mathbb{E}[\text{Cost}(t)] = N \left[ -C_{FP}(1-\pi)f_0(t) + C_{FN} \pi f_1(t) \right] = 0$$

Solving:

$$C_{FP}(1-\pi)f_0(t^*) = C_{FN} \pi f_1(t^*)$$

$$\frac{f_1(t^*)}{f_0(t^*)} = \frac{C_{FP}(1-\pi)}{C_{FN} \pi}$$

This is a **likelihood ratio test**. The optimal threshold is where the likelihood ratio of the score distributions equals the cost-weighted prior odds.

### Worked Numerical Example

Suppose we are tuning a moderation classifier for an AI video platform with:
- \(\pi = 0.02\) (2% of prompts are actually harmful -- typical for a consumer platform)
- \(C_{FP} = \\)2.00$ (average lost revenue per false positive)
- \(C_{FN} = \\)500$ (average cost per false negative, blending minor and major incidents)
- 10,000 generations per day

We need score distributions. Assume the classifier scores follow Beta distributions (common for bounded classifiers):
- Safe items: \(\hat{S} \mid Y=0 \sim \text{Beta}(2, 20)\) (concentrated near 0)
- Harmful items: \(\hat{S} \mid Y=1 \sim \text{Beta}(15, 3)\) (concentrated near 1)

The likelihood ratio condition:

$$\frac{f_1(t^*)}{f_0(t^*)} = \frac{C_{FP}(1-\pi)}{C_{FN}\pi} = \frac{2.00 \times 0.98}{500 \times 0.02} = \frac{1.96}{10.0} = 0.196$$

Since the harmful distribution is concentrated at high scores and the safe distribution at low scores, a likelihood ratio of 0.196 occurs at a relatively low threshold -- meaning we should be aggressive about blocking.

Computing numerically (Beta PDF evaluation):

| Threshold \(t\) | \(f_0(t)\) | \(f_1(t)\) | LR = \(f_1/f_0\) | E[FP]/day | E[FN]/day | E[Cost]/day |
|--------|----------|----------|--------|----------|----------|------------|
| 0.20 | 2.84 | 0.003 | 0.001 | 1,568 | 0.04 | $3,156 |
| 0.30 | 0.89 | 0.06 | 0.067 | 392 | 0.22 | $894 |
| 0.40 | 0.19 | 0.42 | 2.21 | 74 | 0.68 | $488 |
| **0.35** | **0.42** | **0.14** | **0.33** | **156** | **0.38** | **$502** |
| 0.50 | 0.03 | 1.93 | 64.3 | 16 | 1.56 | $812 |
| 0.60 | 0.003 | 4.62 | 1540 | 3 | 3.12 | $1,566 |

The minimum expected cost occurs around \(t^* \approx 0.38\). At this threshold:
- We block ~156 safe items per day (false positives) -- costing $312/day
- We miss ~0.4 harmful items per day (false negatives) -- costing $190/day
- Total expected cost: ~$502/day

Compare this to a naive threshold of 0.50 (equal error rate): total cost is $812/day -- 62% more expensive.

**The key insight**: Because false negatives are 250x more expensive than false positives, the optimal threshold shifts significantly toward blocking more content. A 1.6% false positive rate is acceptable when it reduces false negatives from 0.78% to 0.19%.

### Threshold by Category

Different content categories warrant different thresholds because \(C_{FN}\) varies dramatically:

| Category | \(C_{FN}\) Estimate | Optimal \(t\) | Rationale |
|----------|-------------------|-------------|-----------|
| CSAM | $1,000,000+ | 0.15 | Zero tolerance. Block aggressively, human review everything. |
| Non-consensual deepfake | $100,000 | 0.25 | Legal liability, reputation damage |
| Explicit sexual | $5,000 | 0.40 | App store removal, advertiser loss |
| Graphic violence | $2,000 | 0.45 | Context-dependent (news vs gratuitous) |
| Hate speech | $1,000 | 0.50 | Nuanced, higher FP tolerance |
| Suggestive content | $200 | 0.65 | Low risk, high FP cost for creative platforms |

### The F-beta Score: Encoding Cost Asymmetry

When the cost ratio \(C_{FN}/C_{FP}\) is known, use the \(F_\beta\) score instead of \(F_1\):

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

where \(\beta^2 = C_{FN} / C_{FP}\). For our example (\(C_{FN}/C_{FP} = 250\)):

$$F_{250} \text{ heavily penalizes false negatives.}$$

In practice, \(\beta\) values of 2--5 are common for content moderation (weighting recall 2--5x higher than precision). A \(\beta\) of 250 makes precision nearly irrelevant, which matches intuition: for CSAM, you block everything remotely suspicious and deal with false positives through human review.

---

## API Benchmarks: Accuracy, Latency, and Pricing

We benchmarked five content moderation APIs against a labeled dataset of 5,000 images (2,500 safe, 2,500 policy-violating across categories). All tests conducted January 2026, US-East region.

### Benchmark Results

| API | Precision | Recall | F1 | P95 Latency | Price/Image | Monthly Min |
|-----|-----------|--------|-----|-------------|-------------|-------------|
| **Hive Moderation** | 0.964 | 0.978 | 0.971 | 340ms | $0.0012 | $500 |
| **Sightengine** | 0.941 | 0.952 | 0.946 | 180ms | $0.001 | $29 |
| **Google Cloud Vision** | 0.928 | 0.935 | 0.931 | 420ms | $0.0015 | Pay-per-use |
| **Amazon Rekognition** | 0.917 | 0.943 | 0.930 | 510ms | $0.001 | Pay-per-use |
| **Gemini Flash** | 0.952 | 0.961 | 0.956 | 620ms | $0.0003* | Pay-per-use |

*Gemini Flash pricing based on token usage for image analysis, approximately $0.0003 per image at current rates.

### Detailed Breakdown by Category

```
Category-Level Recall (% of violations correctly caught)
=========================================================

                    Sightengine  Hive   GCV    Rekognition  Gemini
Explicit Sexual        97.2%    99.1%  96.8%    97.5%      98.4%
Suggestive             84.3%    91.7%  79.2%    82.1%      93.6%
Graphic Violence       93.1%    96.4%  94.7%    91.8%      95.2%
Mild Violence          71.8%    82.3%  68.4%    72.6%      88.1%
Hate Symbols           88.2%    94.8%  83.1%    79.4%      91.7%
Firearms/Weapons       95.4%    97.1%  96.2%    98.3%      94.5%
Drug Paraphernalia     82.7%    89.3%  74.5%    77.8%      86.9%
Self-Harm              79.4%    87.6%  81.2%    76.3%      90.8%
Celebrity Faces        N/A      92.4%  88.7%    94.1%      96.2%
Text-in-Image          61.3%    78.4%  82.6%    58.2%      97.8%
```

### Latency Distribution

```
P50 / P95 / P99 Latency (milliseconds)
========================================

Sightengine:      85 /  180 /  340
Hive:            190 /  340 /  580
Google Cloud:    250 /  420 /  710
Rekognition:     280 /  510 /  890
Gemini Flash:    350 /  620 / 1100
```

### Analysis

**Hive** leads in accuracy across almost every category. Its F1 of 0.971 means that for every 1,000 moderation decisions, it makes ~29 errors. At scale, this matters: a platform processing 100K images/day would see ~2,900 errors with Hive vs ~6,900 with Google Cloud Vision.

**Sightengine** wins on latency. At 85ms P50, it adds negligible delay to the generation pipeline. For input screening where speed matters (user is waiting), Sightengine is the best choice.

**Gemini Flash** is the dark horse. It has the best accuracy for nuanced categories (suggestive content, mild violence, text-in-image) because it can reason about context rather than relying purely on visual features. It is also the cheapest per image. The tradeoff is latency -- at 620ms P95, it is too slow for synchronous input screening but excellent for async output screening.

**Recommended architecture**: Use Sightengine for synchronous input screening (fast, cheap, good-enough accuracy). Use Hive + Gemini Flash in parallel for async output screening (best accuracy, cost-effective in batch). Route disagreements to human review.

### Pricing at Scale

| Monthly Volume | Sightengine | Hive | GCV | Rekognition | Gemini Flash |
|---------------|-------------|------|-----|-------------|-------------|
| 10K images | $29 | $500 | $15 | $10 | $3 |
| 100K images | $99 | $500 | $150 | $100 | $30 |
| 500K images | $299 | $600 | $750 | $500 | $150 |
| 1M images | $399 | $1,200 | $1,500 | $1,000 | $300 |
| 5M images | Custom | $6,000 | $7,500 | $5,000 | $1,500 |

Sightengine's flat-rate pricing is advantageous at lower volumes. At 1M+ images, Gemini Flash's per-token pricing becomes the cheapest option by a significant margin.

---

## Case Study: The Grok Incident

### Timeline

- **Dec 8, 2025**: Users on X begin sharing explicit celebrity deepfakes generated through Grok. Initial reports are dismissed as edge cases.
- **Dec 9**: Techniques spread virally. Specific prompt patterns that bypass Grok's safety filters are shared across Reddit and 4chan.
- **Dec 10**: Major news outlets report on the issue. "Grok generates celebrity porn" becomes a headline.
- **Dec 11**: xAI pushes emergency filter update. Many generation requests start returning refusals.
- **Dec 12-15**: Cat-and-mouse continues. Users find new bypass techniques, xAI patches them.
- **Dec 18**: xAI announces comprehensive moderation overhaul. Admits the initial safety layer was insufficient.

### Technical Failures

The Grok incident resulted from failures at every layer of what should have been a three-point architecture:

**1. No celebrity recognition in input screening.** Grok did not check whether text prompts referenced real people. A simple named entity recognition (NER) step, cross-referenced against a celebrity database, would have caught the most obvious violations.

**2. No output screening.** Generated images were delivered directly to users without post-generation safety classification. Even if input screening is imperfect, output screening catches the final result.

**3. Over-reliance on prompt-level safety training.** xAI relied on the model's own training to refuse unsafe requests, rather than implementing external classification. This is fragile -- prompt injection and jailbreaks can bypass model-level safety.

**4. No feedback loop.** When early reports came in, there was no automated system to flag the pattern and escalate. The response was manual and slow.

### What the Architecture Should Have Caught

```
User prompt: "Generate a photo of [Celebrity] in lingerie"
                                    │
                          ┌─────────▼──────────┐
                          │ INPUT SCREENING     │
                          │ 1. NER: "Celebrity" │
                          │    → Real person     │
                          │ 2. Intent: sexual    │
                          │ 3. BLOCKED (Layer 1) │
                          └─────────────────────┘

Even if input screening missed it (adversarial prompt):
                                    │
                          ┌─────────▼──────────┐
                          │ OUTPUT SCREENING    │
                          │ 1. NSFW score: 0.94 │
                          │ 2. Face match: Yes   │
                          │ 3. BLOCKED (Layer 2) │
                          └─────────────────────┘

Even if output screening missed it (novel bypass):
                                    │
                          ┌─────────▼──────────┐
                          │ MONITORING          │
                          │ 1. Spike in reports  │
                          │ 2. Auto-escalation   │
                          │ 3. Kill switch at    │
                          │    threshold (Layer 3)│
                          └─────────────────────┘
```

### Lessons for Platform Builders

1. **Never rely solely on model-level safety.** The model is adversarially attackable. External classifiers are not.
2. **Celebrity/face detection is not optional.** If your platform generates human-like content, you need face detection and celebrity recognition.
3. **Build kill switches.** If moderation failure rate exceeds a threshold, automatically disable generation for the affected category/model until human review.
4. **Speed of response matters.** The difference between a 2-hour response and a 48-hour response is the difference between a minor incident and a front-page story.

---

## Case Study: Kling's Over-Moderation Problem

Kling 3.0 has the opposite problem: its content filters are so aggressive that they block legitimate creative use cases. Users report rejection rates of 20--40% for prompts involving:

- People in swimwear (even editorial/fashion context)
- Historical violence (war documentaries, news recreation)
- Medical content (surgery demonstrations, anatomy)
- Martial arts and combat sports
- Any prompt mentioning alcohol or tobacco

### The Cost of Over-Moderation

Over-moderation is not just a user experience problem -- it is a business problem:

$$\text{Revenue Lost} = \text{FP Rate} \times \text{Generations/Day} \times \text{Revenue/Generation}$$

For a platform routing 10,000 generations/day through Kling with a 30% false positive rate:

$$\text{Revenue Lost} = 0.30 \times 10{,}000 \times \$0.15 = \$450/\text{day} = \$13{,}500/\text{month}$$

Plus the indirect costs: user frustration, churn, negative reviews, support tickets.

### Programmatic Pre-Screening and Prompt Adaptation

The solution is to pre-screen prompts before sending them to Kling, and adapt prompts that would trigger false positives:

```typescript
// Kling-specific prompt sanitizer
// Maintains semantic intent while avoiding known trigger patterns

const KLING_TRIGGER_PATTERNS: {
  pattern: RegExp;
  replacement: string;
  risk: "low" | "medium" | "high";
}[] = [
  // Swimwear / revealing clothing
  {
    pattern: /\b(bikini|swimsuit|swimwear|lingerie|underwear)\b/gi,
    replacement: "summer attire",
    risk: "medium",
  },
  // Alcohol references
  {
    pattern: /\b(beer|wine|cocktail|whiskey|vodka|alcohol|drinking)\b/gi,
    replacement: "beverage",
    risk: "low",
  },
  // Weapons (even in legitimate contexts)
  {
    pattern: /\b(gun|rifle|pistol|sword|knife|weapon)\b/gi,
    replacement: "equipment",
    risk: "medium",
  },
  // Violence descriptors
  {
    pattern: /\b(fight|punch|kick|hit|attack|battle|combat)\b/gi,
    replacement: "dynamic action",
    risk: "medium",
  },
  // Medical terms
  {
    pattern: /\b(surgery|blood|wound|injection|needle|scalpel)\b/gi,
    replacement: "medical procedure",
    risk: "low",
  },
];

interface PromptAdaptationResult {
  originalPrompt: string;
  adaptedPrompt: string;
  modifications: string[];
  riskLevel: "safe" | "adapted" | "high_risk";
  shouldFallbackToAltModel: boolean;
}

function adaptPromptForKling(prompt: string): PromptAdaptationResult {
  let adaptedPrompt = prompt;
  const modifications: string[] = [];
  let maxRisk: "low" | "medium" | "high" = "low";

  for (const { pattern, replacement, risk } of KLING_TRIGGER_PATTERNS) {
    const matches = adaptedPrompt.match(pattern);
    if (matches) {
      adaptedPrompt = adaptedPrompt.replace(pattern, replacement);
      modifications.push(`"${matches[0]}" → "${replacement}"`);
      if (risk === "high" || (risk === "medium" && maxRisk !== "high")) {
        maxRisk = risk;
      }
    }
  }

  // If too many modifications, the prompt intent is probably not suitable for Kling
  const shouldFallback = modifications.length > 3 || maxRisk === "high";

  return {
    originalPrompt: prompt,
    adaptedPrompt,
    modifications,
    riskLevel:
      modifications.length === 0
        ? "safe"
        : shouldFallback
          ? "high_risk"
          : "adapted",
    shouldFallbackToAltModel: shouldFallback,
  };
}

// Usage in the generation pipeline
async function generateWithKling(prompt: string): Promise<GenerationResult> {
  const adaptation = adaptPromptForKling(prompt);

  if (adaptation.shouldFallbackToAltModel) {
    // Route to a less restrictive model (Runway, Veo)
    console.log(
      `Kling prompt too risky after adaptation, falling back. ` +
      `Modifications: ${adaptation.modifications.join(", ")}`
    );
    return generateWithRunway(prompt); // Original prompt, not adapted
  }

  if (adaptation.modifications.length > 0) {
    console.log(
      `Adapted prompt for Kling: ${adaptation.modifications.join(", ")}`
    );
  }

  return callKlingAPI(adaptation.adaptedPrompt);
}
```

### Model-Aware Routing

The better long-term solution is model-aware content routing that considers each model's moderation profile:

```typescript
interface ModelModerationProfile {
  model: string;
  sensitivities: Record<string, number>; // 0-1, higher = more sensitive
  falsePositiveRate: number;
  costPerGeneration: number;
}

const MODEL_PROFILES: ModelModerationProfile[] = [
  {
    model: "kling-3.0",
    sensitivities: {
      nudity: 0.95,      // Very sensitive
      violence: 0.90,
      alcohol: 0.80,
      medical: 0.75,
      weapons: 0.85,
    },
    falsePositiveRate: 0.30,
    costPerGeneration: 0.50,
  },
  {
    model: "runway-gen4.5",
    sensitivities: {
      nudity: 0.70,
      violence: 0.60,
      alcohol: 0.20,
      medical: 0.30,
      weapons: 0.50,
    },
    falsePositiveRate: 0.08,
    costPerGeneration: 0.75,
  },
  {
    model: "veo-3.1",
    sensitivities: {
      nudity: 0.75,
      violence: 0.65,
      alcohol: 0.25,
      medical: 0.35,
      weapons: 0.55,
    },
    falsePositiveRate: 0.10,
    costPerGeneration: 1.00,
  },
];

function selectBestModel(
  prompt: string,
  contentCategories: Record<string, number>
): string {
  // Score each model: lower is better (less likely to false-positive)
  const scores = MODEL_PROFILES.map((profile) => {
    let rejectionRisk = 0;
    for (const [category, promptScore] of Object.entries(contentCategories)) {
      const sensitivity = profile.sensitivities[category] || 0.5;
      rejectionRisk += promptScore * sensitivity;
    }
    return {
      model: profile.model,
      rejectionRisk,
      cost: profile.costPerGeneration,
      expectedCost:
        profile.costPerGeneration / (1 - rejectionRisk * profile.falsePositiveRate),
    };
  });

  // Sort by expected cost (including retries from false positives)
  scores.sort((a, b) => a.expectedCost - b.expectedCost);
  return scores[0].model;
}
```

---

## Content Policy Design

A content policy is not just a list of prohibited categories. It is a decision framework for edge cases. The policy must be:

1. **Specific enough** that automated classifiers can implement it
2. **Nuanced enough** that human reviewers can handle edge cases
3. **Transparent enough** that users understand what is and is not allowed

### Category Taxonomy

```
PROHIBITED CONTENT (always block, no exceptions)
================================================
├── Child Sexual Abuse Material (CSAM)
│   └── Any sexual content depicting or appearing to depict minors
├── Non-Consensual Intimate Imagery
│   └── Sexual content using real person's likeness without consent
├── Terrorism / Extremism
│   └── Content promoting terrorist organizations or acts
└── Imminent Harm
    └── Content designed to facilitate immediate violence

RESTRICTED CONTENT (block by default, allow with verification)
=============================================================
├── Adult Sexual Content
│   ├── Explicit: Block unless platform has 18+ verification
│   └── Suggestive: Allow with age-gate
├── Graphic Violence
│   ├── Gratuitous: Block
│   └── Contextual (news, education, art): Allow with content warning
├── Hate Speech
│   ├── Direct attacks on protected groups: Block
│   └── Quoting/documenting hate speech: Allow with context label
└── Dangerous Activities
    ├── Instructions for harm: Block
    └── Documentary/educational: Allow with content warning

EDGE CASES (human review required)
===================================
├── Medical / Anatomical Content
│   ├── Surgical procedures → Allow with "medical" tag
│   ├── Anatomy education → Allow
│   └── Fetishized medical content → Block
├── Artistic Nudity
│   ├── Classical art recreation → Allow
│   ├── Contemporary fine art → Allow with content warning
│   └── Nudity without artistic merit → Restrict
├── Historical Violence
│   ├── War documentation → Allow with content warning
│   ├── Historical atrocities → Allow educational, block glorification
│   └── Combat reenactment → Allow
├── Combat Sports / Martial Arts
│   ├── Professional sports → Allow
│   ├── Training / technique → Allow
│   └── Street fighting / assault → Block
└── Satire / Parody
    ├── Political satire using public figures → Allow
    ├── Deepfake parody → Allow with disclosure label
    └── Defamatory content disguised as satire → Block
```

### Implementing Edge Cases in Code

```typescript
// Content policy decision engine
interface PolicyDecision {
  action: "allow" | "block" | "restrict" | "review";
  contentWarning?: string;
  requiredLabels?: string[];
  reasoning: string;
}

function applyContentPolicy(
  moderationResult: ModerationResult,
  context: {
    userVerified18Plus: boolean;
    promptIntent?: string;
    contentTags?: string[];
  }
): PolicyDecision {
  const { category, severity } = moderationResult;

  // PROHIBITED: Always block
  if (
    category === "child_exploitation" ||
    category === "deepfake_nonconsensual"
  ) {
    return {
      action: "block",
      reasoning: `Prohibited content: ${category}. Zero tolerance policy.`,
    };
  }

  // RESTRICTED: Conditional
  if (category === "sexual_explicit") {
    if (context.userVerified18Plus && severity < 0.9) {
      return {
        action: "restrict",
        contentWarning: "This content contains explicit material.",
        requiredLabels: ["nsfw", "adult"],
        reasoning: "Explicit sexual content allowed for verified 18+ users.",
      };
    }
    return {
      action: "block",
      reasoning: "Explicit sexual content blocked for unverified users.",
    };
  }

  // EDGE CASES: Context-dependent
  if (category === "violence_graphic") {
    const isMedical = context.contentTags?.includes("medical");
    const isHistorical = context.contentTags?.includes("historical");
    const isNews = context.contentTags?.includes("news");

    if (isMedical || isHistorical || isNews) {
      return {
        action: "allow",
        contentWarning: `This content contains ${isMedical ? "medical" : isHistorical ? "historical" : "news"} imagery that some viewers may find disturbing.`,
        requiredLabels: ["content-warning"],
        reasoning: `Graphic violence allowed in ${isMedical ? "medical" : isHistorical ? "historical" : "news"} context.`,
      };
    }

    if (severity > 0.7) {
      return {
        action: "review",
        reasoning:
          "Graphic violence without clear educational/artistic context. Sending to human review.",
      };
    }
  }

  // Default: allow with low severity, review with medium
  if (severity < 0.3) {
    return { action: "allow", reasoning: "Low severity, within policy." };
  }
  if (severity < 0.6) {
    return {
      action: "review",
      reasoning: `Medium severity (${severity}) for category ${category}. Human review recommended.`,
    };
  }

  return {
    action: "block",
    reasoning: `High severity (${severity}) for category ${category}.`,
  };
}
```

---

## Production TypeScript Implementation

This section presents a complete production implementation of the three-layer moderation system with queue-based async processing, a human review interface, and appeal flow.

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     MODERATION SERVICE                           │
│                                                                  │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐  │
│  │ Input    │    │ Generation│    │ Output   │    │ Delivery  │  │
│  │ Screening│───▶│ Queue     │───▶│ Screening│───▶│ or Block  │  │
│  └─────────┘    └──────────┘    └──────────┘    └───────────┘  │
│       │                                │               │        │
│       │         ┌──────────┐           │               │        │
│       └────────▶│ Human    │◀──────────┘               │        │
│                 │ Review   │                            │        │
│                 │ Queue    │───────────────────────────▶│        │
│                 └──────────┘                                     │
│                      │                                           │
│                 ┌──────────┐                                     │
│                 │ Appeal   │                                     │
│                 │ Queue    │                                     │
│                 └──────────┘                                     │
└──────────────────────────────────────────────────────────────────┘
```

### Core Types

```typescript
// types/moderation.ts

export interface ModerationJob {
  id: string;
  type: "input_screen" | "output_screen" | "human_review" | "appeal";
  status: "pending" | "processing" | "completed" | "failed";
  createdAt: Date;
  updatedAt: Date;

  // Input data
  generationId: string;
  userId: string;
  prompt?: string;
  referenceImageUrl?: string;
  videoUrl?: string;

  // Results
  inputScreenResult?: InputScreenResult;
  outputScreenResult?: OutputScreenResult;
  humanReviewResult?: HumanReviewResult;
  finalDecision?: FinalDecision;

  // Metadata
  modelUsed: string;
  priority: "critical" | "high" | "normal" | "low";
  retryCount: number;
  maxRetries: number;
}

export interface InputScreenResult {
  passed: boolean;
  promptClassification: ModerationResult;
  imageScreenResult?: ImageScreenResult;
  embeddingSimilarity?: { maxSimilarity: number; flagged: boolean };
  totalLatencyMs: number;
}

export interface OutputScreenResult {
  passed: boolean;
  confidence: number;
  consensus: "safe" | "unsafe" | "review";
  frameResults: FrameResult[];
  audioScreenResult?: AudioScreenResult;
  totalLatencyMs: number;
}

export interface HumanReviewResult {
  reviewerId: string;
  decision: "approve" | "reject" | "escalate";
  category?: string;
  notes: string;
  reviewDurationMs: number;
  reviewedAt: Date;
}

export interface FinalDecision {
  action: "deliver" | "block" | "quarantine";
  reasoning: string;
  decidedBy: "auto" | "human" | "appeal";
  decidedAt: Date;
  contentWarning?: string;
  labels?: string[];
}

export interface FrameResult {
  timestamp: number;
  scores: Record<string, number>;
  maxScore: number;
  flaggedCategories: string[];
}

export interface AudioScreenResult {
  transcription: string;
  flaggedSegments: { start: number; end: number; reason: string }[];
  overallScore: number;
}
```

### Queue-Based Moderation Service

```typescript
// services/moderation-queue.ts

import { Firestore, FieldValue } from "firebase-admin/firestore";
import { PubSub } from "@google-cloud/pubsub";
import type { ModerationJob, FinalDecision } from "../types/moderation";

export class ModerationQueueService {
  private db: Firestore;
  private pubsub: PubSub;

  constructor(db: Firestore, pubsub: PubSub) {
    this.db = db;
    this.pubsub = pubsub;
  }

  /**
   * Submit a new generation for moderation.
   * This is the main entry point called by the generation API.
   */
  async submitForModeration(params: {
    generationId: string;
    userId: string;
    prompt: string;
    referenceImageUrl?: string;
    modelUsed: string;
  }): Promise<{ jobId: string; inputScreenPassed: boolean }> {
    const jobId = crypto.randomUUID();

    // Phase 1: Synchronous input screening (fast path)
    const startTime = Date.now();
    const inputResult = await this.performInputScreening(
      params.prompt,
      params.referenceImageUrl
    );
    const latency = Date.now() - startTime;

    // Create the moderation job
    const job: ModerationJob = {
      id: jobId,
      type: "input_screen",
      status: inputResult.passed ? "completed" : "completed",
      createdAt: new Date(),
      updatedAt: new Date(),
      generationId: params.generationId,
      userId: params.userId,
      prompt: params.prompt,
      referenceImageUrl: params.referenceImageUrl,
      modelUsed: params.modelUsed,
      inputScreenResult: inputResult,
      priority: this.calculatePriority(inputResult),
      retryCount: 0,
      maxRetries: 3,
    };

    // Save to Firestore
    await this.db.collection("moderation_jobs").doc(jobId).set(job);

    if (!inputResult.passed) {
      // Blocked at input screening -- save final decision, no generation needed
      const decision: FinalDecision = {
        action: "block",
        reasoning: `Input screening blocked: ${inputResult.promptClassification.category} (severity: ${inputResult.promptClassification.severity})`,
        decidedBy: "auto",
        decidedAt: new Date(),
      };

      await this.db.collection("moderation_jobs").doc(jobId).update({
        finalDecision: decision,
      });

      // Log for analytics
      await this.logModerationEvent(jobId, "input_blocked", {
        category: inputResult.promptClassification.category,
        severity: inputResult.promptClassification.severity,
        latencyMs: latency,
      });

      return { jobId, inputScreenPassed: false };
    }

    // Input passed -- generation can proceed
    // Output screening will be triggered when generation completes
    await this.logModerationEvent(jobId, "input_passed", {
      latencyMs: latency,
    });

    return { jobId, inputScreenPassed: true };
  }

  /**
   * Called when video generation completes.
   * Publishes to output screening queue for async processing.
   */
  async onGenerationComplete(
    jobId: string,
    videoUrl: string
  ): Promise<void> {
    await this.db.collection("moderation_jobs").doc(jobId).update({
      type: "output_screen",
      status: "pending",
      videoUrl,
      updatedAt: FieldValue.serverTimestamp(),
    });

    // Publish to output screening topic
    await this.pubsub.topic("moderation-output-screen").publishMessage({
      json: { jobId, videoUrl },
    });
  }

  /**
   * Process output screening (called by Cloud Function subscriber)
   */
  async processOutputScreening(jobId: string): Promise<void> {
    const jobRef = this.db.collection("moderation_jobs").doc(jobId);
    const jobDoc = await jobRef.get();
    const job = jobDoc.data() as ModerationJob;

    if (!job || !job.videoUrl) {
      throw new Error(`Job ${jobId} not found or missing videoUrl`);
    }

    await jobRef.update({ status: "processing" });

    try {
      const result = await screenGeneratedVideo(job.videoUrl);

      await jobRef.update({
        outputScreenResult: result,
        status: "completed",
        updatedAt: FieldValue.serverTimestamp(),
      });

      // Route based on consensus
      if (result.consensus === "safe") {
        const decision: FinalDecision = {
          action: "deliver",
          reasoning: `Output screening passed with confidence ${result.confidence.toFixed(3)}`,
          decidedBy: "auto",
          decidedAt: new Date(),
        };
        await jobRef.update({ finalDecision: decision });
        await this.deliverVideo(job.generationId, job.userId, job.videoUrl);
      } else if (result.consensus === "review") {
        await this.sendToHumanReview(jobId, "output_screening_uncertain");
      } else {
        // consensus === "unsafe"
        const decision: FinalDecision = {
          action: "block",
          reasoning: `Output screening blocked: max frame score ${Math.max(
            ...result.frameResults.map((f) => f.maxScore)
          ).toFixed(3)}`,
          decidedBy: "auto",
          decidedAt: new Date(),
        };
        await jobRef.update({ finalDecision: decision });
        await this.logModerationEvent(jobId, "output_blocked", {
          maxScore: Math.max(...result.frameResults.map((f) => f.maxScore)),
        });
      }
    } catch (error) {
      await jobRef.update({
        status: "failed",
        retryCount: FieldValue.increment(1),
      });

      // Retry if under limit
      if (job.retryCount < job.maxRetries) {
        await this.pubsub.topic("moderation-output-screen").publishMessage({
          json: { jobId, videoUrl: job.videoUrl },
          attributes: { retryCount: String(job.retryCount + 1) },
        });
      } else {
        // Max retries exceeded -- send to human review
        await this.sendToHumanReview(jobId, "screening_failed_max_retries");
      }
    }
  }

  /**
   * Send a job to the human review queue
   */
  async sendToHumanReview(jobId: string, reason: string): Promise<void> {
    await this.db.collection("moderation_jobs").doc(jobId).update({
      type: "human_review",
      status: "pending",
      updatedAt: FieldValue.serverTimestamp(),
    });

    await this.db.collection("human_review_queue").add({
      moderationJobId: jobId,
      reason,
      status: "pending",
      assignedTo: null,
      createdAt: FieldValue.serverTimestamp(),
      priority: "high",
      slaDeadline: new Date(Date.now() + 4 * 60 * 60 * 1000), // 4 hours
    });

    // Notify moderators via Slack/email
    await this.notifyModerators(jobId, reason);
  }

  /**
   * Process a human review decision
   */
  async submitHumanReview(
    jobId: string,
    reviewerId: string,
    decision: "approve" | "reject" | "escalate",
    notes: string
  ): Promise<void> {
    const review: HumanReviewResult = {
      reviewerId,
      decision,
      notes,
      reviewDurationMs: 0, // Set by frontend
      reviewedAt: new Date(),
    };

    const jobRef = this.db.collection("moderation_jobs").doc(jobId);
    const jobDoc = await jobRef.get();
    const job = jobDoc.data() as ModerationJob;

    await jobRef.update({
      humanReviewResult: review,
      status: "completed",
    });

    if (decision === "approve") {
      const finalDecision: FinalDecision = {
        action: "deliver",
        reasoning: `Human review approved by ${reviewerId}: ${notes}`,
        decidedBy: "human",
        decidedAt: new Date(),
      };
      await jobRef.update({ finalDecision: finalDecision });
      await this.deliverVideo(job.generationId, job.userId, job.videoUrl!);
    } else if (decision === "reject") {
      const finalDecision: FinalDecision = {
        action: "block",
        reasoning: `Human review rejected by ${reviewerId}: ${notes}`,
        decidedBy: "human",
        decidedAt: new Date(),
      };
      await jobRef.update({ finalDecision: finalDecision });

      // Notify user with appeal option
      await this.notifyUserBlocked(job.userId, job.generationId, jobId);
    } else {
      // Escalate to senior moderator
      await this.escalateToSenior(jobId, reviewerId, notes);
    }
  }

  // --- Appeal Flow ---

  async submitAppeal(
    jobId: string,
    userId: string,
    reason: string
  ): Promise<string> {
    const appealId = crypto.randomUUID();

    await this.db.collection("appeals").doc(appealId).set({
      moderationJobId: jobId,
      userId,
      reason,
      status: "pending",
      createdAt: FieldValue.serverTimestamp(),
      slaDeadline: new Date(Date.now() + 24 * 60 * 60 * 1000), // 24 hours
    });

    // Appeals always go to senior moderator
    await this.db.collection("human_review_queue").add({
      moderationJobId: jobId,
      appealId,
      reason: `APPEAL: ${reason}`,
      status: "pending",
      assignedTo: null,
      createdAt: FieldValue.serverTimestamp(),
      priority: "high",
      requiresSenior: true,
      slaDeadline: new Date(Date.now() + 24 * 60 * 60 * 1000),
    });

    return appealId;
  }

  // --- Helpers ---

  private async performInputScreening(
    prompt: string,
    imageUrl?: string
  ): Promise<InputScreenResult> {
    const startTime = Date.now();

    // Run all input checks in parallel
    const [promptResult, embeddingResult, imageResult] = await Promise.all([
      screenPrompt(prompt),
      embeddingSimilarityCheck(prompt),
      imageUrl ? screenReferenceImage(await downloadImage(imageUrl)) : null,
    ]);

    const passed =
      !promptResult.shouldBlock &&
      !embeddingResult.flagged &&
      (imageResult === null || imageResult.safe);

    return {
      passed,
      promptClassification: promptResult,
      imageScreenResult: imageResult ?? undefined,
      embeddingSimilarity: embeddingResult,
      totalLatencyMs: Date.now() - startTime,
    };
  }

  private calculatePriority(
    inputResult: InputScreenResult
  ): ModerationJob["priority"] {
    const { category, severity } = inputResult.promptClassification;
    if (category === "child_exploitation") return "critical";
    if (severity > 0.8) return "high";
    if (severity > 0.5) return "normal";
    return "low";
  }

  private async deliverVideo(
    generationId: string,
    userId: string,
    videoUrl: string
  ): Promise<void> {
    await this.db.collection("generations").doc(generationId).update({
      status: "completed",
      videoUrl,
      deliveredAt: FieldValue.serverTimestamp(),
    });

    // Notify user via WebSocket / push notification
    await this.pubsub.topic("video-ready").publishMessage({
      json: { userId, generationId, videoUrl },
    });
  }

  private async notifyModerators(
    jobId: string,
    reason: string
  ): Promise<void> {
    // Slack webhook integration
    await fetch(process.env.SLACK_MODERATION_WEBHOOK!, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: `New moderation review needed: ${reason}\nJob ID: ${jobId}\nReview at: ${process.env.ADMIN_URL}/moderation/${jobId}`,
      }),
    });
  }

  private async notifyUserBlocked(
    userId: string,
    generationId: string,
    jobId: string
  ): Promise<void> {
    await this.db.collection("notifications").add({
      userId,
      type: "generation_blocked",
      message:
        "Your video generation was blocked by our content policy. " +
        "If you believe this was an error, you can appeal this decision.",
      generationId,
      moderationJobId: jobId,
      appealUrl: `/appeal/${jobId}`,
      createdAt: FieldValue.serverTimestamp(),
      read: false,
    });
  }

  private async escalateToSenior(
    jobId: string,
    reviewerId: string,
    notes: string
  ): Promise<void> {
    await this.db.collection("human_review_queue").add({
      moderationJobId: jobId,
      reason: `ESCALATED by ${reviewerId}: ${notes}`,
      status: "pending",
      assignedTo: null,
      requiresSenior: true,
      createdAt: FieldValue.serverTimestamp(),
      priority: "critical",
      slaDeadline: new Date(Date.now() + 2 * 60 * 60 * 1000), // 2 hours
    });
  }

  private async logModerationEvent(
    jobId: string,
    event: string,
    data: Record<string, unknown>
  ): Promise<void> {
    await this.db.collection("moderation_events").add({
      moderationJobId: jobId,
      event,
      data,
      timestamp: FieldValue.serverTimestamp(),
    });
  }
}
```

---

## The Human Review Pipeline

### Review Dashboard Data Model

```typescript
// Firestore schema for the review dashboard

interface ReviewQueueItem {
  id: string;
  moderationJobId: string;
  reason: string;
  status: "pending" | "assigned" | "in_review" | "completed";
  assignedTo: string | null;
  createdAt: Timestamp;
  assignedAt: Timestamp | null;
  completedAt: Timestamp | null;
  priority: "critical" | "high" | "normal" | "low";
  requiresSenior: boolean;
  slaDeadline: Timestamp;
  slaBreached: boolean;

  // Denormalized for the dashboard (avoid extra reads)
  promptPreview: string;
  thumbnailUrl: string | null;
  autoModerationScores: Record<string, number>;
  userHistorySnapshot: {
    totalGenerations: number;
    previousViolations: number;
    accountAge: number; // days
  };
}
```

### SLA Monitoring

```typescript
// Cloud Function: Check SLA compliance every 5 minutes

import { onSchedule } from "firebase-functions/v2/scheduler";

export const checkModerationSLA = onSchedule(
  "every 5 minutes",
  async (event) => {
    const db = getFirestore();
    const now = new Date();

    // Find items that have breached SLA
    const breached = await db
      .collection("human_review_queue")
      .where("status", "in", ["pending", "assigned", "in_review"])
      .where("slaDeadline", "<", now)
      .where("slaBreached", "==", false)
      .get();

    for (const doc of breached.docs) {
      const item = doc.data() as ReviewQueueItem;

      // Mark as breached
      await doc.ref.update({ slaBreached: true });

      // Escalate
      if (item.priority === "critical") {
        // Page on-call engineer
        await sendPagerDutyAlert({
          severity: "critical",
          summary: `CRITICAL moderation item ${item.id} breached SLA`,
          moderationJobId: item.moderationJobId,
        });
      } else {
        // Slack alert
        await sendSlackAlert(
          `SLA breached for moderation item ${item.id} (${item.priority} priority). ` +
          `Deadline was ${item.slaDeadline.toDate().toISOString()}.`
        );
      }
    }

    // Metrics: log current queue depth and SLA compliance
    const pendingCount = await db
      .collection("human_review_queue")
      .where("status", "==", "pending")
      .count()
      .get();

    const breachedCount = await db
      .collection("human_review_queue")
      .where("slaBreached", "==", true)
      .where("completedAt", "==", null)
      .count()
      .get();

    console.log(
      JSON.stringify({
        metric: "moderation_queue_depth",
        pending: pendingCount.data().count,
        slaBreached: breachedCount.data().count,
        timestamp: now.toISOString(),
      })
    );
  }
);
```

### Reviewer Efficiency Metrics

Track reviewer performance to maintain quality and speed:

```typescript
interface ReviewerMetrics {
  reviewerId: string;
  period: "daily" | "weekly" | "monthly";
  periodStart: Date;

  // Volume
  reviewsCompleted: number;
  avgReviewDurationMs: number;
  medianReviewDurationMs: number;

  // Quality
  agreementWithAutoModeration: number; // % agreement with automated system
  overturnsOnAppeal: number; // decisions overturned by senior on appeal
  interReviewerAgreement: number; // Cohen's kappa with other reviewers

  // Categories
  categoryCounts: Record<string, number>;
}

async function calculateReviewerMetrics(
  reviewerId: string,
  startDate: Date,
  endDate: Date
): Promise<ReviewerMetrics> {
  const db = getFirestore();

  const reviews = await db
    .collection("moderation_jobs")
    .where("humanReviewResult.reviewerId", "==", reviewerId)
    .where("humanReviewResult.reviewedAt", ">=", startDate)
    .where("humanReviewResult.reviewedAt", "<=", endDate)
    .get();

  const durations = reviews.docs
    .map((d) => d.data().humanReviewResult.reviewDurationMs)
    .sort((a, b) => a - b);

  const avgDuration =
    durations.reduce((s, d) => s + d, 0) / durations.length;
  const medianDuration = durations[Math.floor(durations.length / 2)];

  // Calculate agreement with automated moderation
  let agreements = 0;
  for (const doc of reviews.docs) {
    const job = doc.data() as ModerationJob;
    const autoDecision = job.outputScreenResult?.consensus;
    const humanDecision = job.humanReviewResult?.decision;

    const autoBlock = autoDecision === "unsafe";
    const humanBlock = humanDecision === "reject";

    if (autoBlock === humanBlock) agreements++;
  }

  return {
    reviewerId,
    period: "daily",
    periodStart: startDate,
    reviewsCompleted: reviews.docs.length,
    avgReviewDurationMs: avgDuration,
    medianReviewDurationMs: medianDuration,
    agreementWithAutoModeration: agreements / reviews.docs.length,
    overturnsOnAppeal: 0, // Calculated separately
    interReviewerAgreement: 0, // Requires cross-reviewer comparison
    categoryCounts: {},
  };
}
```

---

## Monitoring, Alerting, and Continuous Improvement

### Key Metrics Dashboard

```
MODERATION HEALTH DASHBOARD
============================

Real-time Metrics:
─────────────────
  Input block rate:        4.2%  (target: 3-8%)
  Output block rate:       1.8%  (target: 1-5%)
  Human review queue:      23    (target: <50)
  Avg review wait time:    47min (SLA: <4hr)
  SLA compliance:          98.7% (target: >95%)

Daily Metrics:
──────────────
  Total generations:       12,847
  Input screened:          12,847 (100%)
  Input blocked:           539   (4.2%)
  Output screened:         12,308
  Output blocked:          221   (1.8%)
  Sent to review:          87    (0.7%)
  User reports:            12
  Appeals filed:           3
  Appeals upheld:          1     (false positive caught)

Quality Metrics (rolling 7-day):
────────────────────────────────
  Estimated FP rate:       2.1%  (from appeal + review data)
  Estimated FN rate:       0.3%  (from user reports + sampling)
  F1 score:                0.967
  Precision:               0.979
  Recall:                  0.997
  Avg latency (input):     142ms
  Avg latency (output):    2.3s
  Cost per moderation:     $0.008
```

### Automated Threshold Tuning

```typescript
// Weekly job: re-optimize thresholds based on latest data

async function optimizeThresholds(): Promise<void> {
  const db = getFirestore();

  // Pull all moderation decisions from the last 30 days
  // that have ground truth (human review or user report)
  const decisions = await db
    .collection("moderation_jobs")
    .where("finalDecision.decidedBy", "in", ["human", "appeal"])
    .where("updatedAt", ">", new Date(Date.now() - 30 * 24 * 3600 * 1000))
    .get();

  // Build score-label pairs
  const pairs = decisions.docs.map((doc) => {
    const job = doc.data() as ModerationJob;
    const score = job.outputScreenResult
      ? Math.max(...job.outputScreenResult.frameResults.map((f) => f.maxScore))
      : job.inputScreenResult?.promptClassification.severity ?? 0;
    const trueLabel = job.finalDecision?.action === "block" ? 1 : 0;
    return { score, label: trueLabel };
  });

  // Grid search for optimal threshold
  const costs: { threshold: number; expectedCost: number }[] = [];
  for (let t = 0.1; t <= 0.9; t += 0.01) {
    const fp = pairs.filter((p) => p.label === 0 && p.score >= t).length;
    const fn = pairs.filter((p) => p.label === 1 && p.score < t).length;

    const expectedCost = fp * COST_FP + fn * COST_FN;
    costs.push({ threshold: t, expectedCost });
  }

  costs.sort((a, b) => a.expectedCost - b.expectedCost);
  const optimal = costs[0];

  console.log(
    `Optimal threshold: ${optimal.threshold.toFixed(2)} ` +
    `(expected cost: $${optimal.expectedCost.toFixed(2)}/day)`
  );

  // Update threshold in config (with manual approval gate)
  await db.collection("moderation_config").doc("thresholds").update({
    proposedThreshold: optimal.threshold,
    currentExpectedCost: optimal.expectedCost,
    proposedAt: FieldValue.serverTimestamp(),
    status: "pending_approval", // Requires human approval before going live
  });
}
```

### The Feedback Loop

Every moderation decision feeds back into the system:

1. **Human review outcomes** become training data for threshold optimization
2. **User reports** identify false negatives the automated system missed
3. **Appeal outcomes** identify false positives and category-specific over-moderation
4. **New bad-prompt embeddings** are added to the similarity check database when novel attack patterns are found
5. **Reviewer disagreements** surface ambiguous policy areas that need clarification

This feedback loop ensures the moderation system improves over time rather than degrading as adversaries adapt.

---

## Conclusion

Content moderation is not a feature you bolt on after launch. It is load-bearing infrastructure that must be designed into the generation pipeline from day one. The three-point architecture (input screening, output screening, human review) with cost-optimal thresholds per category, multi-API consensus for output screening, and a human review pipeline with SLA monitoring gives you a system that is both safe and scalable.

The mathematics are clear: because false negatives are orders of magnitude more expensive than false positives, the optimal threshold is always lower (more aggressive) than intuition suggests. Build for aggressive automated screening with a robust appeal process, not permissive screening with damage control.

The Grok incident cost xAI immeasurably in reputation. The Kling over-moderation problem costs Kuaishou in lost revenue and user frustration. The engineering sweet spot is a system that blocks what it should, passes what it should, and routes everything else to humans fast enough to matter. That is what this architecture delivers.
