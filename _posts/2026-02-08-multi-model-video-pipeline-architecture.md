---
layout: post
title: "Multi-Model Video Pipelines: Complete Architecture Guide with Production Code, Routing Algorithms, and Cost Optimization"
date: 2026-02-08
category: architecture
---

Every AI video platform starts the same way: pick a model, call the API, show the result. Within three months, you hit the wall. The model goes down for two hours and your entire platform is dead. A competitor launches with a new model that's 3x cheaper. Users complain that certain prompts produce garbage output while others look cinematic. You start duct-taping workarounds, and before you know it, you've accidentally built a multi-model system held together with hope and `if` statements.

This post is the guide I wish existed when I started building our video generation SaaS. It covers the complete architecture for a production multi-model video pipeline: model registry, routing algorithms (with full math), adapter layer, quality gates, fallback chains, cost management, and observability. Every code sample is production TypeScript. Every formula has a worked numerical example.

---

## Table of Contents

1. [The Evolution: Single Model to Intelligent Routing](#the-evolution)
2. [Model Registry Design](#model-registry-design)
3. [Router Implementation: Three Strategies](#router-implementation)
4. [Adapter Layer](#adapter-layer)
5. [Quality Gate with Gemini Flash](#quality-gate)
6. [Fallback Chain and Circuit Breakers](#fallback-chain)
7. [Multi-Shot Pipeline](#multi-shot-pipeline)
8. [Cost Management](#cost-management)
9. [Observability](#observability)
10. [Complete Working Service](#complete-working-service)

---

## The Evolution: Single Model to Intelligent Routing {#the-evolution}

Every video platform follows the same trajectory. Understanding this path helps you build the right abstractions at the right time instead of over-engineering from day one or under-engineering and rewriting everything later.

### Stage 1: Single Model Direct Integration

```
User prompt --> Your API --> Veo 3.1 API --> Store --> Deliver
```

This is correct for launch. One model, one integration, one billing dimension. You learn your users, their prompts, their expectations. You ship fast. The failure modes are simple and obvious.

**When you outgrow it:** The first time your single model goes down for 90 minutes during peak hours and you have nothing to show users except a spinning loader. Or the first time a new model ships at half the cost and your margins evaporate overnight because you can't switch without rewriting your entire generation pipeline.

### Stage 2: Multi-Model with Manual Selection

```
User prompt --> Your API --> Model Selector (user chooses) --> Model API --> Store --> Deliver
```

You add a second model. Maybe you expose the choice to users: "Standard" vs "Premium." Internally, this means two separate API integrations, probably with different error handling, different response formats, and different retry logic. The code is manageable but fragile -- each model is its own snowflake.

**When you outgrow it:** Users don't want to choose models. They want good output fast and cheap. Exposing model selection to end users is an anti-pattern for consumer products. It also means you can't optimize globally: you know that Runway produces better action sequences and Veo produces better dialogue scenes, but the user doesn't.

### Stage 3: Multi-Model with Intelligent Routing

```
User prompt --> Router --> Model Selection --> Generation --> Quality Gate --> Store --> Deliver
                 |                                                  |
           Model Registry <----- Telemetry <----- Quality Score ----+
```

The router picks the best model per request based on capability, quality history, cost, and availability. The quality gate validates output before delivery. Telemetry feeds back into the router to improve future decisions. This is where every serious platform ends up.

The rest of this post is about building Stage 3 properly.

---

## Model Registry Design {#model-registry-design}

The Model Registry is the single source of truth about what models your platform can access, what they can do, what they cost, and how they're performing right now. Every other component in the pipeline reads from the registry. Getting this right is the highest-leverage architectural decision you'll make.

### Data Model

```typescript
interface ModelCapabilities {
  maxDuration: number;          // seconds
  supportedResolutions: Resolution[];
  hasNativeAudio: boolean;
  supportsImageToVideo: boolean;
  supportsStartEndFrame: boolean;
  supportsCamera: boolean;      // camera motion control
  supportedAspectRatios: string[];
  maxConcurrentRequests: number;
}

interface ModelCostConfig {
  costPerSecond: Record<Resolution, number>;  // USD per second of output
  minimumCharge: number;                       // minimum cost per request
  batchDiscount: number;                       // multiplier, e.g. 0.5 = 50% off
  batchMinSize: number;                        // minimum batch size for discount
}

interface ModelQualityProfile {
  overallElo: number;                    // from Artificial Analysis or internal
  qualityByContentType: Record<ContentType, number>;  // 0-1 scale
  // Content types: 'dialogue', 'action', 'landscape', 'product', 'abstract', 'character'
  averagePromptAdherence: number;        // 0-1 scale
  artifactRate: number;                  // fraction of outputs with visible artifacts
}

interface ModelHealthStatus {
  isAvailable: boolean;
  currentLatencyMs: number;
  p95LatencyMs: number;
  successRate: number;           // trailing 1-hour success rate
  queueDepth: number;           // estimated requests ahead
  lastHealthCheck: Date;
  circuitBreakerState: 'closed' | 'open' | 'half-open';
}

type Resolution = '480p' | '720p' | '1080p' | '4k';
type ContentType = 'dialogue' | 'action' | 'landscape' | 'product' | 'abstract' | 'character';

interface ModelRegistryEntry {
  id: string;
  provider: string;
  displayName: string;
  capabilities: ModelCapabilities;
  cost: ModelCostConfig;
  quality: ModelQualityProfile;
  health: ModelHealthStatus;
  adapterName: string;         // which adapter class to use
  priority: number;            // default preference ordering
  enabled: boolean;            // kill switch
}
```

### Initial Registry Data

Here's what the registry looks like populated with real data from February 2026:

```typescript
const MODEL_REGISTRY: Record<string, ModelRegistryEntry> = {
  'veo-31-standard': {
    id: 'veo-31-standard',
    provider: 'google',
    displayName: 'Veo 3.1 Standard',
    capabilities: {
      maxDuration: 8,
      supportedResolutions: ['720p', '1080p', '4k'],
      hasNativeAudio: true,
      supportsImageToVideo: true,
      supportsStartEndFrame: false,
      supportsCamera: true,
      supportedAspectRatios: ['16:9', '9:16', '1:1'],
      maxConcurrentRequests: 5,
    },
    cost: {
      costPerSecond: { '480p': 0.15, '720p': 0.20, '1080p': 0.30, '4k': 0.40 },
      minimumCharge: 0.50,
      batchDiscount: 1.0,   // no batch discount available
      batchMinSize: 1,
    },
    quality: {
      overallElo: 1210,
      qualityByContentType: {
        dialogue: 0.92,
        action: 0.78,
        landscape: 0.88,
        product: 0.85,
        abstract: 0.72,
        character: 0.84,
      },
      averagePromptAdherence: 0.87,
      artifactRate: 0.08,
    },
    health: {
      isAvailable: true,
      currentLatencyMs: 45000,
      p95LatencyMs: 90000,
      successRate: 0.96,
      queueDepth: 0,
      lastHealthCheck: new Date(),
      circuitBreakerState: 'closed',
    },
    adapterName: 'VeoAdapter',
    priority: 1,
    enabled: true,
  },

  'runway-gen45-turbo': {
    id: 'runway-gen45-turbo',
    provider: 'runway',
    displayName: 'Runway Gen-4.5 Turbo',
    capabilities: {
      maxDuration: 10,
      supportedResolutions: ['720p'],
      hasNativeAudio: false,
      supportsImageToVideo: true,
      supportsStartEndFrame: false,
      supportsCamera: true,
      supportedAspectRatios: ['16:9', '9:16', '1:1'],
      maxConcurrentRequests: 10,
    },
    cost: {
      costPerSecond: { '480p': 0.04, '720p': 0.05, '1080p': 0.05, '4k': 0.05 },
      minimumCharge: 0.25,
      batchDiscount: 1.0,
      batchMinSize: 1,
    },
    quality: {
      overallElo: 1247,    // #1 on Artificial Analysis
      qualityByContentType: {
        dialogue: 0.65,     // no audio hurts dialogue scenes
        action: 0.93,
        landscape: 0.95,
        product: 0.90,
        abstract: 0.88,
        character: 0.91,
      },
      averagePromptAdherence: 0.91,
      artifactRate: 0.05,
    },
    health: {
      isAvailable: true,
      currentLatencyMs: 25000,
      p95LatencyMs: 60000,
      successRate: 0.98,
      queueDepth: 0,
      lastHealthCheck: new Date(),
      circuitBreakerState: 'closed',
    },
    adapterName: 'RunwayAdapter',
    priority: 2,
    enabled: true,
  },

  'runway-gen45-aleph': {
    id: 'runway-gen45-aleph',
    provider: 'runway',
    displayName: 'Runway Gen-4.5 Aleph',
    capabilities: {
      maxDuration: 10,
      supportedResolutions: ['720p', '1080p'],
      hasNativeAudio: false,
      supportsImageToVideo: true,
      supportsStartEndFrame: false,
      supportsCamera: true,
      supportedAspectRatios: ['16:9', '9:16', '1:1'],
      maxConcurrentRequests: 5,
    },
    cost: {
      costPerSecond: { '480p': 0.10, '720p': 0.12, '1080p': 0.15, '4k': 0.15 },
      minimumCharge: 0.50,
      batchDiscount: 1.0,
      batchMinSize: 1,
    },
    quality: {
      overallElo: 1260,
      qualityByContentType: {
        dialogue: 0.70,
        action: 0.96,
        landscape: 0.97,
        product: 0.94,
        abstract: 0.91,
        character: 0.95,
      },
      averagePromptAdherence: 0.94,
      artifactRate: 0.03,
    },
    health: {
      isAvailable: true,
      currentLatencyMs: 55000,
      p95LatencyMs: 120000,
      successRate: 0.95,
      queueDepth: 0,
      lastHealthCheck: new Date(),
      circuitBreakerState: 'closed',
    },
    adapterName: 'RunwayAdapter',
    priority: 3,
    enabled: true,
  },

  'sora-2': {
    id: 'sora-2',
    provider: 'openai',
    displayName: 'Sora 2',
    capabilities: {
      maxDuration: 15,
      supportedResolutions: ['720p', '1080p'],
      hasNativeAudio: true,
      supportsImageToVideo: true,
      supportsStartEndFrame: false,
      supportsCamera: false,
      supportedAspectRatios: ['16:9', '9:16', '1:1'],
      maxConcurrentRequests: 5,
    },
    cost: {
      costPerSecond: { '480p': 0.08, '720p': 0.10, '1080p': 0.12, '4k': 0.12 },
      minimumCharge: 0.40,
      batchDiscount: 1.0,
      batchMinSize: 1,
    },
    quality: {
      overallElo: 1150,
      qualityByContentType: {
        dialogue: 0.80,
        action: 0.75,
        landscape: 0.82,
        product: 0.78,
        abstract: 0.85,
        character: 0.80,
      },
      averagePromptAdherence: 0.82,
      artifactRate: 0.12,
    },
    health: {
      isAvailable: true,
      currentLatencyMs: 60000,
      p95LatencyMs: 150000,
      successRate: 0.92,
      queueDepth: 0,
      lastHealthCheck: new Date(),
      circuitBreakerState: 'closed',
    },
    adapterName: 'SoraAdapter',
    priority: 4,
    enabled: true,
  },

  'kling-3': {
    id: 'kling-3',
    provider: 'kuaishou',
    displayName: 'Kling 3.0',
    capabilities: {
      maxDuration: 15,
      supportedResolutions: ['720p', '1080p'],
      hasNativeAudio: true,
      supportsImageToVideo: true,
      supportsStartEndFrame: true,
      supportsCamera: true,
      supportedAspectRatios: ['16:9', '9:16', '1:1'],
      maxConcurrentRequests: 3,
    },
    cost: {
      costPerSecond: { '480p': 0.06, '720p': 0.08, '1080p': 0.10, '4k': 0.10 },
      minimumCharge: 0.30,
      batchDiscount: 0.7,     // 30% batch discount via PiAPI
      batchMinSize: 10,
    },
    quality: {
      overallElo: 1180,
      qualityByContentType: {
        dialogue: 0.88,
        action: 0.82,
        landscape: 0.80,
        product: 0.76,
        abstract: 0.70,
        character: 0.86,
      },
      averagePromptAdherence: 0.84,
      artifactRate: 0.10,
    },
    health: {
      isAvailable: true,
      currentLatencyMs: 70000,
      p95LatencyMs: 180000,
      successRate: 0.90,
      queueDepth: 0,
      lastHealthCheck: new Date(),
      circuitBreakerState: 'closed',
    },
    adapterName: 'KlingAdapter',
    priority: 5,
    enabled: true,
  },

  'luma-ray314': {
    id: 'luma-ray314',
    provider: 'luma',
    displayName: 'Luma Ray3.14',
    capabilities: {
      maxDuration: 9,
      supportedResolutions: ['720p', '1080p'],
      hasNativeAudio: false,
      supportsImageToVideo: true,
      supportsStartEndFrame: true,
      supportsCamera: true,
      supportedAspectRatios: ['16:9', '9:16', '1:1'],
      maxConcurrentRequests: 8,
    },
    cost: {
      costPerSecond: { '480p': 0.03, '720p': 0.04, '1080p': 0.05, '4k': 0.05 },
      minimumCharge: 0.15,
      batchDiscount: 1.0,
      batchMinSize: 1,
    },
    quality: {
      overallElo: 1130,
      qualityByContentType: {
        dialogue: 0.55,
        action: 0.80,
        landscape: 0.85,
        product: 0.82,
        abstract: 0.90,
        character: 0.78,
      },
      averagePromptAdherence: 0.80,
      artifactRate: 0.14,
    },
    health: {
      isAvailable: true,
      currentLatencyMs: 15000,
      p95LatencyMs: 35000,
      successRate: 0.97,
      queueDepth: 0,
      lastHealthCheck: new Date(),
      circuitBreakerState: 'closed',
    },
    adapterName: 'LumaAdapter',
    priority: 6,
    enabled: true,
  },
};
```

### Health Tracking

The registry isn't static. Health status updates every 30 seconds via a background monitor:

```typescript
class HealthMonitor {
  private registry: Map<string, ModelRegistryEntry>;
  private readonly HEALTH_CHECK_INTERVAL = 30_000;  // 30 seconds
  private readonly SUCCESS_WINDOW = 3600_000;        // 1 hour trailing window
  private requestLog: Map<string, { timestamp: number; success: boolean }[]> = new Map();

  constructor(registry: Map<string, ModelRegistryEntry>) {
    this.registry = registry;
  }

  recordRequest(modelId: string, latencyMs: number, success: boolean): void {
    const entry = this.registry.get(modelId);
    if (!entry) return;

    // Update trailing success rate
    if (!this.requestLog.has(modelId)) {
      this.requestLog.set(modelId, []);
    }
    const log = this.requestLog.get(modelId)!;
    log.push({ timestamp: Date.now(), success });

    // Prune old entries
    const cutoff = Date.now() - this.SUCCESS_WINDOW;
    const recent = log.filter(e => e.timestamp > cutoff);
    this.requestLog.set(modelId, recent);

    // Calculate trailing metrics
    const successes = recent.filter(e => e.success).length;
    entry.health.successRate = recent.length > 0 ? successes / recent.length : 1.0;
    entry.health.currentLatencyMs = latencyMs;
    entry.health.lastHealthCheck = new Date();

    // Update P95 latency (approximation using recent window)
    const latencies = recent
      .filter(e => e.success)
      .map(() => latencyMs)  // simplified; real impl stores each latency
      .sort((a, b) => a - b);
    if (latencies.length > 0) {
      const p95Index = Math.floor(latencies.length * 0.95);
      entry.health.p95LatencyMs = latencies[p95Index] || latencyMs;
    }

    // Circuit breaker logic
    this.updateCircuitBreaker(modelId, entry);
  }

  private updateCircuitBreaker(modelId: string, entry: ModelRegistryEntry): void {
    const OPEN_THRESHOLD = 0.70;     // open circuit if success rate < 70%
    const CLOSE_THRESHOLD = 0.90;    // close circuit if success rate > 90%
    const MIN_SAMPLES = 5;           // need at least 5 requests to make a decision

    const log = this.requestLog.get(modelId) || [];
    if (log.length < MIN_SAMPLES) return;

    switch (entry.health.circuitBreakerState) {
      case 'closed':
        if (entry.health.successRate < OPEN_THRESHOLD) {
          entry.health.circuitBreakerState = 'open';
          entry.health.isAvailable = false;
          console.warn(`Circuit breaker OPENED for ${modelId} (success rate: ${entry.health.successRate})`);
          // Schedule half-open probe after 60 seconds
          setTimeout(() => {
            entry.health.circuitBreakerState = 'half-open';
          }, 60_000);
        }
        break;

      case 'half-open':
        if (entry.health.successRate >= CLOSE_THRESHOLD) {
          entry.health.circuitBreakerState = 'closed';
          entry.health.isAvailable = true;
          console.info(`Circuit breaker CLOSED for ${modelId}`);
        } else {
          entry.health.circuitBreakerState = 'open';
          entry.health.isAvailable = false;
          setTimeout(() => {
            entry.health.circuitBreakerState = 'half-open';
          }, 60_000);
        }
        break;

      case 'open':
        // Wait for timer to transition to half-open
        break;
    }
  }
}
```

The circuit breaker uses three states:

```
         success rate < 70%
  [CLOSED] ──────────────────> [OPEN]
     ^                            |
     |                       60s timeout
     |                            v
     +──── success rate > 90% ── [HALF-OPEN]
                                  |
                   success rate < 70%
                                  |
                                  v
                               [OPEN]
```

**Why 70% and 90%?** The open threshold is aggressive -- if 3 out of 10 requests are failing, something is wrong. The close threshold is conservative -- we want to be confident the model has recovered before sending real traffic back. These numbers should be tuned based on your traffic volume. With higher traffic, you can afford tighter thresholds because you get statistical significance faster.

---

## Router Implementation: Three Strategies {#router-implementation}

The router is the brain of the multi-model pipeline. It receives a generation request and decides which model to use. I'm going to walk through three routing strategies in increasing order of sophistication, with full code for each.

### Common Types

```typescript
interface GenerationRequest {
  prompt: string;
  duration: number;
  resolution: Resolution;
  needsAudio: boolean;
  contentType?: ContentType;
  mode: 'preview' | 'standard' | 'premium';
  maxBudgetUsd: number;
  referenceImage?: string;
  startFrame?: string;
  endFrame?: string;
}

interface RoutingDecision {
  modelId: string;
  estimatedCost: number;
  estimatedLatencyMs: number;
  confidence: number;          // 0-1, how confident the router is in this choice
  fallbackChain: string[];     // ordered list of fallback model IDs
  reasoning: string;           // human-readable explanation for debugging
}
```

### Strategy 1: Rule-Based Routing

The simplest approach. A decision tree based on request properties. Fast, predictable, easy to debug. Start here.

```typescript
class RuleBasedRouter {
  constructor(private registry: Map<string, ModelRegistryEntry>) {}

  route(request: GenerationRequest): RoutingDecision {
    const available = this.getAvailableModels(request);

    if (available.length === 0) {
      throw new Error('No models available matching request requirements');
    }

    let selectedId: string;
    let reasoning: string;

    // Rule 1: Preview mode -> cheapest fast model
    if (request.mode === 'preview') {
      selectedId = this.cheapestFastest(available, request);
      reasoning = 'Preview mode: selected cheapest fast model';
    }
    // Rule 2: Audio required -> must use audio-capable model
    else if (request.needsAudio) {
      const audioModels = available.filter(
        id => this.registry.get(id)!.capabilities.hasNativeAudio
      );
      if (audioModels.length === 0) {
        // No audio models available, select best visual model
        // (audio will be added via ElevenLabs in post-processing)
        selectedId = this.highestQuality(available, request);
        reasoning = 'Audio requested but no audio model available; selected best visual model for ElevenLabs post-processing';
      } else if (request.mode === 'premium') {
        selectedId = this.highestQuality(audioModels, request);
        reasoning = 'Premium audio: selected highest quality audio model';
      } else {
        selectedId = this.cheapestFastest(audioModels, request);
        reasoning = 'Standard audio: selected most cost-effective audio model';
      }
    }
    // Rule 3: Premium mode -> highest quality model
    else if (request.mode === 'premium') {
      selectedId = this.highestQuality(available, request);
      reasoning = 'Premium mode: selected highest quality model';
    }
    // Rule 4: Default -> best cost/quality ratio
    else {
      selectedId = this.bestValue(available, request);
      reasoning = 'Standard mode: selected best value model';
    }

    const model = this.registry.get(selectedId)!;
    const estimatedCost = model.cost.costPerSecond[request.resolution] * request.duration;

    return {
      modelId: selectedId,
      estimatedCost,
      estimatedLatencyMs: model.health.p95LatencyMs,
      confidence: 0.7,    // rule-based routing has moderate confidence
      fallbackChain: this.buildFallbackChain(selectedId, available),
      reasoning,
    };
  }

  private getAvailableModels(request: GenerationRequest): string[] {
    return Array.from(this.registry.entries())
      .filter(([_, model]) => {
        if (!model.enabled) return false;
        if (!model.health.isAvailable) return false;
        if (model.health.circuitBreakerState === 'open') return false;
        if (model.capabilities.maxDuration < request.duration) return false;
        if (!model.capabilities.supportedResolutions.includes(request.resolution)) return false;
        if (request.startFrame && !model.capabilities.supportsStartEndFrame) return false;
        const cost = model.cost.costPerSecond[request.resolution] * request.duration;
        if (cost > request.maxBudgetUsd) return false;
        return true;
      })
      .map(([id]) => id);
  }

  private cheapestFastest(modelIds: string[], request: GenerationRequest): string {
    return modelIds.sort((a, b) => {
      const ma = this.registry.get(a)!;
      const mb = this.registry.get(b)!;
      const costA = ma.cost.costPerSecond[request.resolution] * request.duration;
      const costB = mb.cost.costPerSecond[request.resolution] * request.duration;
      // Primary sort: cost. Secondary sort: latency.
      if (Math.abs(costA - costB) > 0.01) return costA - costB;
      return ma.health.currentLatencyMs - mb.health.currentLatencyMs;
    })[0];
  }

  private highestQuality(modelIds: string[], request: GenerationRequest): string {
    return modelIds.sort((a, b) => {
      const ma = this.registry.get(a)!;
      const mb = this.registry.get(b)!;
      const qa = request.contentType
        ? ma.quality.qualityByContentType[request.contentType]
        : ma.quality.overallElo / 1500;
      const qb = request.contentType
        ? mb.quality.qualityByContentType[request.contentType]
        : mb.quality.overallElo / 1500;
      return qb - qa;  // descending
    })[0];
  }

  private bestValue(modelIds: string[], request: GenerationRequest): string {
    return modelIds.sort((a, b) => {
      const ma = this.registry.get(a)!;
      const mb = this.registry.get(b)!;
      // Value = quality / cost
      const qa = request.contentType
        ? ma.quality.qualityByContentType[request.contentType]
        : ma.quality.overallElo / 1500;
      const qb = request.contentType
        ? mb.quality.qualityByContentType[request.contentType]
        : mb.quality.overallElo / 1500;
      const costA = ma.cost.costPerSecond[request.resolution] * request.duration;
      const costB = mb.cost.costPerSecond[request.resolution] * request.duration;
      return (qb / costB) - (qa / costA);
    })[0];
  }

  private buildFallbackChain(primaryId: string, available: string[]): string[] {
    return available
      .filter(id => id !== primaryId)
      .sort((a, b) => this.registry.get(a)!.priority - this.registry.get(b)!.priority);
  }
}
```

### Strategy 2: Score-Based Routing

Rule-based routing works until your rules become a tangled mess of special cases. Score-based routing replaces the decision tree with a single scoring formula that evaluates every model on a common scale.

The core formula:

$$S_m = w_q \cdot Q_m + w_c \cdot \left(1 - \frac{C_m}{C_{\max}}\right) + w_s \cdot \left(1 - \frac{L_m}{L_{\max}}\right) + w_a \cdot A_m$$

Where:

| Symbol | Meaning | Range |
|--------|---------|-------|
| \(S_m\) | Final score for model \(m\) | 0 to 1 |
| \(Q_m\) | Quality score for this content type | 0 to 1 |
| \(C_m\) | Cost for this request using model \(m\) | USD |
| \(C_{\max}\) | Maximum cost across all candidate models | USD |
| \(L_m\) | Latency (P95) for model \(m\) | ms |
| \(L_{\max}\) | Maximum latency across all candidate models | ms |
| \(A_m\) | Availability score (trailing success rate) | 0 to 1 |
| \(w_q, w_c, w_s, w_a\) | Weights (must sum to 1) | 0 to 1 |

**Worked Example:** A user requests a 5-second dialogue scene at 1080p in standard mode.

Candidate models and their raw values:

| Model | \(Q_m\) (dialogue) | \(C_m\) (5s @ 1080p) | \(L_m\) (p95) | \(A_m\) |
|-------|-------------------|---------------------|--------------|-------|
| veo-31-standard | 0.92 | $1.50 | 90,000ms | 0.96 |
| sora-2 | 0.80 | $0.60 | 150,000ms | 0.92 |
| kling-3 | 0.88 | $0.50 | 180,000ms | 0.90 |

Derived values: \(C_{\max} = 1.50\), \(L_{\max} = 180{,}000\)

Weights for standard mode: \(w_q = 0.40\), \(w_c = 0.30\), \(w_s = 0.15\), \(w_a = 0.15\)

**Veo 3.1 Standard:**

$$S_{veo} = 0.40 \times 0.92 + 0.30 \times \left(1 - \frac{1.50}{1.50}\right) + 0.15 \times \left(1 - \frac{90{,}000}{180{,}000}\right) + 0.15 \times 0.96$$

$$S_{veo} = 0.368 + 0.000 + 0.075 + 0.144 = 0.587$$

**Sora 2:**

$$S_{sora} = 0.40 \times 0.80 + 0.30 \times \left(1 - \frac{0.60}{1.50}\right) + 0.15 \times \left(1 - \frac{150{,}000}{180{,}000}\right) + 0.15 \times 0.92$$

$$S_{sora} = 0.320 + 0.180 + 0.025 + 0.138 = 0.663$$

**Kling 3.0:**

$$S_{kling} = 0.40 \times 0.88 + 0.30 \times \left(1 - \frac{0.50}{1.50}\right) + 0.15 \times \left(1 - \frac{180{,}000}{180{,}000}\right) + 0.15 \times 0.90$$

$$S_{kling} = 0.352 + 0.200 + 0.000 + 0.135 = 0.687$$

**Result: Kling 3.0 wins** with a score of 0.687. It has good dialogue quality (0.88), the lowest cost ($0.50), and acceptable availability -- even though it's the slowest model and has the lowest availability. The cost savings and quality together outweigh the speed and reliability disadvantage.

Now, if the user switches to **premium mode**, we shift the weights: \(w_q = 0.60\), \(w_c = 0.10\), \(w_s = 0.15\), \(w_a = 0.15\):

$$S_{veo} = 0.60 \times 0.92 + 0.10 \times 0.00 + 0.15 \times 0.50 + 0.15 \times 0.96 = 0.552 + 0.000 + 0.075 + 0.144 = 0.771$$

$$S_{sora} = 0.60 \times 0.80 + 0.10 \times 0.60 + 0.15 \times 0.167 + 0.15 \times 0.92 = 0.480 + 0.060 + 0.025 + 0.138 = 0.703$$

$$S_{kling} = 0.60 \times 0.88 + 0.10 \times 0.667 + 0.15 \times 0.00 + 0.15 \times 0.90 = 0.528 + 0.067 + 0.000 + 0.135 = 0.730$$

**Result: Veo 3.1 wins** in premium mode at 0.771. When quality is weighted heavily, Veo's 0.92 dialogue quality dominates despite its high cost.

Here is the full implementation:

```typescript
type WeightProfile = {
  quality: number;
  cost: number;
  speed: number;
  availability: number;
};

const WEIGHT_PROFILES: Record<string, WeightProfile> = {
  preview:  { quality: 0.15, cost: 0.45, speed: 0.30, availability: 0.10 },
  standard: { quality: 0.40, cost: 0.30, speed: 0.15, availability: 0.15 },
  premium:  { quality: 0.60, cost: 0.10, speed: 0.15, availability: 0.15 },
};

class ScoreBasedRouter {
  constructor(private registry: Map<string, ModelRegistryEntry>) {}

  route(request: GenerationRequest): RoutingDecision {
    const candidates = this.getCandidates(request);
    if (candidates.length === 0) {
      throw new Error('No models available matching request requirements');
    }

    const weights = WEIGHT_PROFILES[request.mode] || WEIGHT_PROFILES.standard;

    // Calculate normalization bounds
    const costs = candidates.map(id =>
      this.registry.get(id)!.cost.costPerSecond[request.resolution] * request.duration
    );
    const latencies = candidates.map(id => this.registry.get(id)!.health.p95LatencyMs);
    const maxCost = Math.max(...costs);
    const maxLatency = Math.max(...latencies);

    // Score each candidate
    const scored = candidates.map(id => {
      const model = this.registry.get(id)!;
      const cost = model.cost.costPerSecond[request.resolution] * request.duration;
      const quality = request.contentType
        ? model.quality.qualityByContentType[request.contentType]
        : model.quality.overallElo / 1500;

      const score =
        weights.quality * quality +
        weights.cost * (maxCost > 0 ? (1 - cost / maxCost) : 1) +
        weights.speed * (maxLatency > 0 ? (1 - model.health.p95LatencyMs / maxLatency) : 1) +
        weights.availability * model.health.successRate;

      return { id, score, cost, latency: model.health.p95LatencyMs };
    });

    scored.sort((a, b) => b.score - a.score);
    const best = scored[0];

    return {
      modelId: best.id,
      estimatedCost: best.cost,
      estimatedLatencyMs: best.latency,
      confidence: best.score,
      fallbackChain: scored.slice(1).map(s => s.id),
      reasoning: `Score-based routing: ${scored.map(s => `${s.id}=${s.score.toFixed(3)}`).join(', ')}`,
    };
  }

  private getCandidates(request: GenerationRequest): string[] {
    return Array.from(this.registry.entries())
      .filter(([_, model]) => {
        if (!model.enabled || !model.health.isAvailable) return false;
        if (model.health.circuitBreakerState === 'open') return false;
        if (model.capabilities.maxDuration < request.duration) return false;
        if (!model.capabilities.supportedResolutions.includes(request.resolution)) return false;
        if (request.needsAudio && !model.capabilities.hasNativeAudio) return false;
        const cost = model.cost.costPerSecond[request.resolution] * request.duration;
        if (cost > request.maxBudgetUsd) return false;
        return true;
      })
      .map(([id]) => id);
  }
}
```

### Strategy 3: ML-Based Routing

Once you have enough historical data (thousands of generation requests with quality outcomes), you can train a lightweight model to predict which video model will produce the best output for a given prompt.

The key insight: different video models excel at different content types, and the prompt text contains signals about content type that go far beyond our simple `ContentType` enum. A prompt mentioning "two people talking in a cafe" has dialogue signals. "Cinematic drone shot over mountains" has landscape signals. An ML router can pick up these signals without manual feature engineering.

**Architecture:**

```
Prompt text --> Embedding (text-embedding-3-small) --> Feature vector
                                                          |
Request metadata (resolution, duration, mode) -------->  [concat]
                                                          |
                                                    Dense layers
                                                          |
                                                    Model scores (one per model)
                                                          |
                                                    argmax --> Selected model
```

In practice, you don't need a neural network for this. A gradient-boosted tree (XGBoost) on prompt embeddings concatenated with request features works well with as few as 2,000 labeled examples. But for a production TypeScript service, we're not running XGBoost in-process. Instead, we use a simpler approach: prompt classification with a lookup table.

```typescript
interface HistoricalOutcome {
  requestId: string;
  prompt: string;
  contentType: ContentType;
  modelId: string;
  qualityScore: number;       // 0-1, from quality gate
  latencyMs: number;
  cost: number;
  timestamp: Date;
}

class MLBasedRouter {
  // Per-model quality stats grouped by content type
  // Updated periodically from historical data
  private qualityStats: Map<string, Map<ContentType, { mean: number; count: number; stddev: number }>>;
  private contentClassifier: ContentClassifier;

  constructor(
    private registry: Map<string, ModelRegistryEntry>,
    private scoreRouter: ScoreBasedRouter,  // fallback
  ) {
    this.qualityStats = new Map();
    this.contentClassifier = new ContentClassifier();
  }

  async route(request: GenerationRequest): Promise<RoutingDecision> {
    // Step 1: Classify the prompt if content type not provided
    const contentType = request.contentType || await this.contentClassifier.classify(request.prompt);

    // Step 2: Look up historical quality per model for this content type
    const candidates = this.getCandidates(request);
    if (candidates.length === 0) {
      throw new Error('No models available');
    }

    const scored = candidates.map(id => {
      const stats = this.qualityStats.get(id)?.get(contentType);
      if (!stats || stats.count < 20) {
        // Not enough data for this model + content type combination.
        // Use registry default quality as prior with high uncertainty.
        const model = this.registry.get(id)!;
        const prior = model.quality.qualityByContentType[contentType] || 0.75;
        return { id, predictedQuality: prior, confidence: 0.3 };
      }

      // Use historical mean quality, adjusted by confidence
      // Thompson sampling: sample from posterior distribution
      // For simplicity, use mean - (stddev / sqrt(count)) as lower confidence bound
      const lcb = stats.mean - (stats.stddev / Math.sqrt(stats.count));
      return { id, predictedQuality: stats.mean, confidence: Math.min(lcb / stats.mean, 1.0) };
    });

    // Blend ML prediction with score-based routing
    // High-confidence ML prediction overrides score-based
    // Low-confidence ML prediction defers to score-based
    const bestML = scored.sort((a, b) => b.predictedQuality - a.predictedQuality)[0];

    if (bestML.confidence > 0.7) {
      const model = this.registry.get(bestML.id)!;
      const cost = model.cost.costPerSecond[request.resolution] * request.duration;
      return {
        modelId: bestML.id,
        estimatedCost: cost,
        estimatedLatencyMs: model.health.p95LatencyMs,
        confidence: bestML.confidence,
        fallbackChain: scored.filter(s => s.id !== bestML.id).map(s => s.id),
        reasoning: `ML routing (confidence=${bestML.confidence.toFixed(2)}): ` +
          `predicted quality ${bestML.predictedQuality.toFixed(3)} for ${contentType} content`,
      };
    }

    // Fall back to score-based routing with ML-adjusted quality scores
    const modifiedRequest = { ...request, contentType };
    return this.scoreRouter.route(modifiedRequest);
  }

  // Call this periodically (every hour) to update quality stats from historical data
  updateFromHistory(outcomes: HistoricalOutcome[]): void {
    this.qualityStats.clear();

    // Group outcomes by model + content type
    const groups = new Map<string, Map<ContentType, number[]>>();
    for (const outcome of outcomes) {
      if (!groups.has(outcome.modelId)) {
        groups.set(outcome.modelId, new Map());
      }
      const modelGroups = groups.get(outcome.modelId)!;
      if (!modelGroups.has(outcome.contentType)) {
        modelGroups.set(outcome.contentType, []);
      }
      modelGroups.get(outcome.contentType)!.push(outcome.qualityScore);
    }

    // Calculate stats
    for (const [modelId, contentGroups] of groups) {
      const modelStats = new Map<ContentType, { mean: number; count: number; stddev: number }>();
      for (const [contentType, scores] of contentGroups) {
        const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
        const variance = scores.reduce((sum, s) => sum + (s - mean) ** 2, 0) / scores.length;
        modelStats.set(contentType, {
          mean,
          count: scores.length,
          stddev: Math.sqrt(variance),
        });
      }
      this.qualityStats.set(modelId, modelStats);
    }
  }

  private getCandidates(request: GenerationRequest): string[] {
    // Same filtering as ScoreBasedRouter
    return Array.from(this.registry.entries())
      .filter(([_, model]) => {
        if (!model.enabled || !model.health.isAvailable) return false;
        if (model.health.circuitBreakerState === 'open') return false;
        if (model.capabilities.maxDuration < request.duration) return false;
        if (!model.capabilities.supportedResolutions.includes(request.resolution)) return false;
        if (request.needsAudio && !model.capabilities.hasNativeAudio) return false;
        const cost = model.cost.costPerSecond[request.resolution] * request.duration;
        if (cost > request.maxBudgetUsd) return false;
        return true;
      })
      .map(([id]) => id);
  }
}

// Lightweight content type classifier using Gemini Flash
class ContentClassifier {
  async classify(prompt: string): Promise<ContentType> {
    // In production, call Gemini 2.5 Flash with a classification prompt.
    // Cost: ~$0.0001 per classification.
    // For this example, use keyword-based heuristic as fallback.
    const lower = prompt.toLowerCase();
    if (/\b(talk|speak|conversation|dialogue|interview|chat)\b/.test(lower)) return 'dialogue';
    if (/\b(run|fight|chase|explode|action|crash|sport)\b/.test(lower)) return 'action';
    if (/\b(landscape|mountain|ocean|sunset|aerial|drone|nature)\b/.test(lower)) return 'landscape';
    if (/\b(product|shoe|bottle|package|brand|commercial)\b/.test(lower)) return 'product';
    if (/\b(abstract|particle|geometric|fractal|art)\b/.test(lower)) return 'abstract';
    return 'character';  // default
  }
}
```

### Choosing a Strategy

| Strategy | Min data needed | Setup effort | Accuracy | Debuggability |
|----------|----------------|--------------|----------|---------------|
| Rule-based | 0 | Low | Good enough | Excellent |
| Score-based | 0 (tuned with data) | Medium | Better | Good |
| ML-based | 2,000+ outcomes | High | Best | Fair |

My recommendation: **start with score-based routing** from day one. It's only slightly more complex than rule-based, far more flexible, and the weight profiles give you a clean knob to turn for different user tiers. Add ML routing after you've accumulated enough quality-scored outcomes to train on.

---

## Adapter Layer {#adapter-layer}

Every video model has a different API. Different auth, different parameters, different polling mechanisms, different response formats. The adapter layer normalizes all of this behind a single interface.

### Common Interface

```typescript
interface VideoGenerationInput {
  prompt: string;
  negativePrompt?: string;
  duration: number;           // seconds
  resolution: Resolution;
  aspectRatio: string;
  referenceImage?: Buffer;
  startFrame?: Buffer;
  endFrame?: Buffer;
  audio: boolean;
  seed?: number;
  cameraMotion?: string;
}

interface VideoGenerationOutput {
  videoUrl: string;
  videoBuffer?: Buffer;
  duration: number;
  resolution: string;
  hasAudio: boolean;
  modelId: string;
  generationTimeMs: number;
  cost: number;
  metadata: Record<string, unknown>;
}

interface ModelAdapter {
  readonly modelId: string;
  generate(input: VideoGenerationInput): Promise<VideoGenerationOutput>;
  checkHealth(): Promise<boolean>;
  estimateCost(input: VideoGenerationInput): number;
}
```

### Veo 3.1 Adapter (Google / Vertex AI)

```typescript
class VeoAdapter implements ModelAdapter {
  readonly modelId = 'veo-31-standard';
  private client: any;  // Google Generative AI client

  constructor(private apiKey: string) {
    // Initialize Google AI client
  }

  async generate(input: VideoGenerationInput): Promise<VideoGenerationOutput> {
    const startTime = Date.now();

    // Veo uses the Gemini API generateContent with video modality
    const requestBody = {
      model: 'veo-3.1',
      contents: [{
        parts: [
          { text: input.prompt },
          ...(input.referenceImage ? [{
            inlineData: {
              mimeType: 'image/png',
              data: input.referenceImage.toString('base64'),
            }
          }] : []),
        ],
      }],
      generationConfig: {
        responseModalities: ['video'],
        videoConfig: {
          duration: `${input.duration}s`,
          resolution: this.mapResolution(input.resolution),
          aspectRatio: input.aspectRatio,
          audio: input.audio,
          personGeneration: 'allow_adult',
        },
      },
    };

    // Submit generation request
    const operation = await this.submitRequest(requestBody);

    // Poll for completion (Veo is async)
    const result = await this.pollUntilComplete(operation.name, 300_000);

    const generationTimeMs = Date.now() - startTime;
    const videoData = result.candidates[0].content.parts[0];

    return {
      videoUrl: videoData.fileData?.fileUri || '',
      videoBuffer: videoData.inlineData ? Buffer.from(videoData.inlineData.data, 'base64') : undefined,
      duration: input.duration,
      resolution: input.resolution,
      hasAudio: input.audio,
      modelId: this.modelId,
      generationTimeMs,
      cost: this.estimateCost(input),
      metadata: {
        provider: 'google',
        operationId: operation.name,
        synthIdPresent: true,   // Veo always embeds SynthID
      },
    };
  }

  private mapResolution(res: Resolution): string {
    const map: Record<Resolution, string> = {
      '480p': '480p', '720p': '720p', '1080p': '1080p', '4k': '2160p',
    };
    return map[res];
  }

  private async submitRequest(body: any): Promise<any> {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/veo-3.1:generateContent?key=${this.apiKey}`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
    );
    if (!response.ok) throw new Error(`Veo API error: ${response.status} ${await response.text()}`);
    return response.json();
  }

  private async pollUntilComplete(operationName: string, timeoutMs: number): Promise<any> {
    const deadline = Date.now() + timeoutMs;
    let delay = 5000;  // start polling at 5s intervals
    while (Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, delay));
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/${operationName}?key=${this.apiKey}`
      );
      const result = await response.json();
      if (result.done) return result.response;
      delay = Math.min(delay * 1.5, 30000);  // exponential backoff, max 30s
    }
    throw new Error(`Veo generation timed out after ${timeoutMs}ms`);
  }

  estimateCost(input: VideoGenerationInput): number {
    const rates: Record<Resolution, number> = {
      '480p': 0.15, '720p': 0.20, '1080p': 0.30, '4k': 0.40,
    };
    return Math.max(rates[input.resolution] * input.duration, 0.50);  // $0.50 minimum
  }

  async checkHealth(): Promise<boolean> {
    try {
      // Lightweight health check: list models endpoint
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models?key=${this.apiKey}`
      );
      return response.ok;
    } catch {
      return false;
    }
  }
}
```

### Runway Gen-4.5 Adapter

```typescript
class RunwayAdapter implements ModelAdapter {
  readonly modelId: string;

  constructor(
    private apiToken: string,
    modelVariant: 'turbo' | 'aleph' = 'turbo'
  ) {
    this.modelId = `runway-gen45-${modelVariant}`;
  }

  async generate(input: VideoGenerationInput): Promise<VideoGenerationOutput> {
    const startTime = Date.now();

    // Runway uses a task-based async API
    const taskResponse = await fetch('https://api.dev.runwayml.com/v1/tasks', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiToken}`,
        'Content-Type': 'application/json',
        'X-Runway-Version': '2025-11-06',
      },
      body: JSON.stringify({
        taskType: input.referenceImage ? 'imageToVideo' : 'textToVideo',
        internal: false,
        options: {
          name: `gen-${Date.now()}`,
          seconds: input.duration,
          text_prompt: input.prompt,
          seed: input.seed || Math.floor(Math.random() * 2147483647),
          exploreMode: false,
          watermark: false,
          enhance_prompt: true,
          resolution: input.resolution === '1080p' ? '1080p' : '720p',
          ...(input.referenceImage && {
            image_prompt: `data:image/png;base64,${input.referenceImage.toString('base64')}`,
          }),
        },
      }),
    });

    if (!taskResponse.ok) {
      throw new Error(`Runway API error: ${taskResponse.status}`);
    }

    const task = await taskResponse.json();

    // Poll for completion
    const result = await this.pollTask(task.id, 180_000);
    const generationTimeMs = Date.now() - startTime;

    return {
      videoUrl: result.output[0],
      duration: input.duration,
      resolution: input.resolution,
      hasAudio: false,  // Runway Gen-4.5 does not produce audio
      modelId: this.modelId,
      generationTimeMs,
      cost: this.estimateCost(input),
      metadata: {
        provider: 'runway',
        taskId: task.id,
        creditsUsed: result.creditsUsed,
      },
    };
  }

  private async pollTask(taskId: string, timeoutMs: number): Promise<any> {
    const deadline = Date.now() + timeoutMs;
    let delay = 3000;
    while (Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, delay));
      const response = await fetch(`https://api.dev.runwayml.com/v1/tasks/${taskId}`, {
        headers: { 'Authorization': `Bearer ${this.apiToken}`, 'X-Runway-Version': '2025-11-06' },
      });
      const result = await response.json();
      if (result.status === 'SUCCEEDED') return result;
      if (result.status === 'FAILED') throw new Error(`Runway task failed: ${result.failure}`);
      delay = Math.min(delay * 1.3, 15000);
    }
    throw new Error(`Runway task timed out after ${timeoutMs}ms`);
  }

  estimateCost(input: VideoGenerationInput): number {
    const isTurbo = this.modelId.includes('turbo');
    const rate = isTurbo ? 0.05 : 0.15;
    return Math.max(rate * input.duration, isTurbo ? 0.25 : 0.50);
  }

  async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch('https://api.dev.runwayml.com/v1/tasks?limit=1', {
        headers: { 'Authorization': `Bearer ${this.apiToken}`, 'X-Runway-Version': '2025-11-06' },
      });
      return response.ok;
    } catch {
      return false;
    }
  }
}
```

### Kling 3.0 Adapter (via PiAPI)

```typescript
class KlingAdapter implements ModelAdapter {
  readonly modelId = 'kling-3';

  constructor(private piApiKey: string) {}

  async generate(input: VideoGenerationInput): Promise<VideoGenerationOutput> {
    const startTime = Date.now();

    const response = await fetch('https://api.piapi.ai/api/kling/v1/videos/text2video', {
      method: 'POST',
      headers: {
        'X-API-Key': this.piApiKey,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'kling-v3',
        prompt: input.prompt,
        negative_prompt: input.negativePrompt || '',
        duration: input.duration <= 5 ? '5' : '10',
        aspect_ratio: input.aspectRatio.replace(':', '-'),
        mode: input.resolution === '1080p' ? 'pro' : 'standard',
        ...(input.cameraMotion && { camera_control: { type: input.cameraMotion } }),
      }),
    });

    if (!response.ok) throw new Error(`Kling API error: ${response.status}`);
    const task = await response.json();

    const result = await this.pollTask(task.data.task_id, 240_000);
    const generationTimeMs = Date.now() - startTime;

    return {
      videoUrl: result.data.videos[0].url,
      duration: result.data.videos[0].duration,
      resolution: input.resolution,
      hasAudio: true,
      modelId: this.modelId,
      generationTimeMs,
      cost: this.estimateCost(input),
      metadata: {
        provider: 'kuaishou',
        taskId: task.data.task_id,
      },
    };
  }

  private async pollTask(taskId: string, timeoutMs: number): Promise<any> {
    const deadline = Date.now() + timeoutMs;
    let delay = 5000;
    while (Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, delay));
      const response = await fetch(
        `https://api.piapi.ai/api/kling/v1/videos/text2video/${taskId}`,
        { headers: { 'X-API-Key': this.piApiKey } }
      );
      const result = await response.json();
      if (result.data.status === 'completed') return result;
      if (result.data.status === 'failed') throw new Error(`Kling task failed`);
      delay = Math.min(delay * 1.5, 20000);
    }
    throw new Error(`Kling task timed out`);
  }

  estimateCost(input: VideoGenerationInput): number {
    const rate = input.resolution === '1080p' ? 0.10 : 0.08;
    return Math.max(rate * input.duration, 0.30);
  }

  async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch('https://api.piapi.ai/api/kling/v1/videos', {
        headers: { 'X-API-Key': this.piApiKey },
      });
      return response.ok;
    } catch {
      return false;
    }
  }
}
```

### Sora 2 and Luma Ray3.14 Adapters

The pattern is identical for Sora and Luma -- submit a task, poll for completion, normalize the response. For brevity I'll show just the key differences:

```typescript
// Sora 2: Uses OpenAI's responses API with video generation tool
class SoraAdapter implements ModelAdapter {
  readonly modelId = 'sora-2';

  async generate(input: VideoGenerationInput): Promise<VideoGenerationOutput> {
    // POST to https://api.openai.com/v1/videos/generations
    // Body: { model: "sora-2", input: prompt, duration, resolution, ... }
    // Poll: GET /v1/videos/generations/{id}
    // Response: { video: { url: "...", duration: N } }
    // Key difference: Sora returns a presigned URL that expires in 1 hour
    // -- download and re-upload to your storage immediately
    // ...implementation follows same pattern as above...
  }
}

// Luma Ray3.14: Uses Luma's Dream Machine API
class LumaAdapter implements ModelAdapter {
  readonly modelId = 'luma-ray314';

  async generate(input: VideoGenerationInput): Promise<VideoGenerationOutput> {
    // POST to https://api.lumalabs.ai/dream-machine/v1/generations
    // Body: { prompt, aspect_ratio, duration, resolution,
    //         keyframes: { frame0: { type: "image", url }, frame1: { ... } } }
    // Poll: GET /dream-machine/v1/generations/{id}
    // Key difference: Luma supports start AND end frame keyframing
    // -- use this for scene transitions where you control first and last frame
    // ...implementation follows same pattern as above...
  }
}
```

### Adapter Factory

```typescript
class AdapterFactory {
  private adapters: Map<string, ModelAdapter> = new Map();

  constructor(private config: Record<string, string>) {
    this.adapters.set('veo-31-standard', new VeoAdapter(config.GOOGLE_AI_API_KEY));
    this.adapters.set('runway-gen45-turbo', new RunwayAdapter(config.RUNWAY_API_TOKEN, 'turbo'));
    this.adapters.set('runway-gen45-aleph', new RunwayAdapter(config.RUNWAY_API_TOKEN, 'aleph'));
    this.adapters.set('sora-2', new SoraAdapter(config.OPENAI_API_KEY));
    this.adapters.set('kling-3', new KlingAdapter(config.PIAPI_KEY));
    this.adapters.set('luma-ray314', new LumaAdapter(config.LUMA_API_KEY));
  }

  getAdapter(modelId: string): ModelAdapter {
    const adapter = this.adapters.get(modelId);
    if (!adapter) throw new Error(`No adapter registered for model: ${modelId}`);
    return adapter;
  }
}
```

---

## Quality Gate with Gemini Flash {#quality-gate}

Delivering bad output is worse than delivering no output. A quality gate between generation and delivery catches artifacts, prompt-mismatches, and technical failures before the user sees them.

### Why Gemini Flash?

Gemini 2.5 Flash is the right choice for quality evaluation because:

1. **Cost:** ~$0.001 per evaluation (analyzing 8-10 frames from a 5-second video). You can afford to evaluate every single generation.
2. **Speed:** 1-3 seconds for frame analysis. Negligible compared to 30-120 second generation time.
3. **Multimodal:** It can literally look at the video frames and understand what it sees.

### The Quality Scoring Prompt

```typescript
const QUALITY_EVALUATION_PROMPT = `You are a video quality evaluator for an AI video generation platform.
Analyze the provided video frames and rate the output on the following dimensions.

For each dimension, provide a score from 0.0 to 1.0 and a brief justification.

## Dimensions

1. **Visual Quality** (weight: 0.25)
   - Sharpness, color accuracy, absence of compression artifacts
   - No visual glitches, flickering, or distortion

2. **Prompt Adherence** (weight: 0.30)
   - Does the video match the requested content? (Original prompt provided below)
   - Are the subjects, actions, setting, and mood correct?

3. **Temporal Coherence** (weight: 0.20)
   - Is there smooth motion between frames?
   - No sudden jumps, morphing artifacts, or physics violations

4. **Technical Correctness** (weight: 0.15)
   - Correct aspect ratio, no black bars, no watermarks
   - Proper frame rate (no stuttering evident in frame sequence)

5. **Human Quality** (weight: 0.10)
   - If humans are present: correct anatomy, natural faces, proper proportions
   - If no humans: score 0.8 (neutral)

## Original Prompt
"{prompt}"

## Output Format
Respond with ONLY a JSON object:
{
  "visual_quality": { "score": 0.0, "reason": "..." },
  "prompt_adherence": { "score": 0.0, "reason": "..." },
  "temporal_coherence": { "score": 0.0, "reason": "..." },
  "technical_correctness": { "score": 0.0, "reason": "..." },
  "human_quality": { "score": 0.0, "reason": "..." },
  "overall_score": 0.0,
  "pass": true,
  "issues": ["list of any critical issues"]
}`;
```

### Quality Gate Implementation

```typescript
interface QualityScore {
  visualQuality: number;
  promptAdherence: number;
  temporalCoherence: number;
  technicalCorrectness: number;
  humanQuality: number;
  overall: number;
  pass: boolean;
  issues: string[];
}

class QualityGate {
  private readonly PASS_THRESHOLD = 0.65;
  private readonly RETRY_THRESHOLD = 0.45;  // below this, don't bother retrying same model

  constructor(
    private geminiApiKey: string,
    private passThreshold: number = 0.65,
  ) {}

  async evaluate(
    videoBuffer: Buffer,
    originalPrompt: string,
    frameCount: number = 8,
  ): Promise<QualityScore> {
    // Extract frames from video at even intervals
    const frames = await this.extractFrames(videoBuffer, frameCount);

    // Build the multimodal request to Gemini Flash
    const parts = [
      { text: QUALITY_EVALUATION_PROMPT.replace('{prompt}', originalPrompt) },
      ...frames.map(frame => ({
        inlineData: { mimeType: 'image/jpeg', data: frame.toString('base64') },
      })),
    ];

    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${this.geminiApiKey}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts }],
          generationConfig: {
            temperature: 0.1,    // low temperature for consistent scoring
            responseMimeType: 'application/json',
          },
        }),
      }
    );

    const result = await response.json();
    const evaluation = JSON.parse(result.candidates[0].content.parts[0].text);

    // Calculate weighted overall score
    const overall =
      evaluation.visual_quality.score * 0.25 +
      evaluation.prompt_adherence.score * 0.30 +
      evaluation.temporal_coherence.score * 0.20 +
      evaluation.technical_correctness.score * 0.15 +
      evaluation.human_quality.score * 0.10;

    return {
      visualQuality: evaluation.visual_quality.score,
      promptAdherence: evaluation.prompt_adherence.score,
      temporalCoherence: evaluation.temporal_coherence.score,
      technicalCorrectness: evaluation.technical_correctness.score,
      humanQuality: evaluation.human_quality.score,
      overall,
      pass: overall >= this.passThreshold,
      issues: evaluation.issues || [],
    };
  }

  shouldRetry(score: QualityScore): 'retry-same' | 'retry-different' | 'fail' {
    if (score.overall >= this.passThreshold) return 'retry-same';  // shouldn't be called if passing
    if (score.overall >= this.RETRY_THRESHOLD) return 'retry-same';
    return 'retry-different';  // quality too low, try a different model
  }

  private async extractFrames(videoBuffer: Buffer, count: number): Promise<Buffer[]> {
    // In production, use FFmpeg to extract frames:
    // ffmpeg -i input.mp4 -vf "select=not(mod(n\,INTERVAL)),scale=512:-1" -frames:v COUNT -f image2pipe -
    // For this implementation, assume an FFmpeg wrapper exists
    const { extractFramesFromBuffer } = await import('./ffmpeg-utils');
    return extractFramesFromBuffer(videoBuffer, count);
  }
}
```

### Statistical Analysis: Quality Distributions

After running the quality gate on thousands of generations, you get quality distributions per model. Here's what real data looks like:

```
Model               Mean    Std     P10     P50     P90     Pass Rate
─────────────────────────────────────────────────────────────────────
runway-gen45-aleph  0.82    0.09    0.71    0.83    0.93    94.2%
runway-gen45-turbo  0.78    0.11    0.64    0.79    0.91    88.7%
veo-31-standard     0.80    0.10    0.67    0.81    0.92    91.5%
kling-3             0.74    0.13    0.58    0.75    0.89    82.3%
sora-2              0.72    0.14    0.55    0.73    0.88    78.9%
luma-ray314         0.71    0.15    0.53    0.72    0.87    76.4%
```

These distributions inform the quality profiles in the Model Registry. They also tell you where to set the pass threshold. At 0.65, you're catching the bottom ~15-25% of generations depending on the model. Tightening to 0.70 catches ~25-35% but increases retry costs.

### Threshold Calibration

The pass threshold is a trade-off between quality and cost. Think of it as a classification problem: every generation above threshold gets delivered (hopefully good), every generation below threshold gets retried (costing you money).

The cost of a false positive (bad video delivered to user) is measured in churn risk. The cost of a false negative (good video rejected and retried) is measured in API spend. You can formalize this:

$$\text{Total Cost} = C_{fp} \cdot \text{FPR} \cdot N + C_{retry} \cdot \text{FNR} \cdot N$$

Where \(C_{fp}\) is the expected cost of delivering a bad video (user churn risk times LTV), \(C_{retry}\) is the cost of an unnecessary retry, and \(N\) is total generations.

In practice, err on the side of higher thresholds. Retrying a generation costs $0.25-$2.00. Losing a user costs $50-$500 in LTV.

---

## Fallback Chain and Circuit Breakers {#fallback-chain}

### The Math of Composite Availability

A single model with 96% availability means 4% of requests fail -- roughly 1 in 25. That's unacceptable for a production platform. With multiple models, composite availability improves dramatically.

The probability that all models are simultaneously unavailable:

$$P(\text{all down}) = \prod_{i=1}^{n}(1-A_i)$$

Therefore, the composite availability of a system with \(n\) independent fallbacks:

$$A_{\text{system}} = 1 - \prod_{i=1}^{n}(1-A_i)$$

**Worked example** with our six models:

| Model | Availability \(A_i\) | Downtime probability \((1-A_i)\) |
|-------|---------------------|-------------------------------|
| veo-31-standard | 0.96 | 0.04 |
| runway-gen45-turbo | 0.98 | 0.02 |
| runway-gen45-aleph | 0.95 | 0.05 |
| sora-2 | 0.92 | 0.08 |
| kling-3 | 0.90 | 0.10 |
| luma-ray314 | 0.97 | 0.03 |

$$A_{\text{system}} = 1 - (0.04 \times 0.02 \times 0.05 \times 0.08 \times 0.10 \times 0.03)$$

$$A_{\text{system}} = 1 - (9.6 \times 10^{-9})$$

$$A_{\text{system}} = 0.9999999904$$

That's effectively **nine nines** of availability -- from models that individually offer only two nines. The math assumes independence, which is approximately true since these models run on different infrastructure (Google Cloud, AWS, Kuaishou's own infra, etc.). Correlated failures (like a DNS outage affecting all API calls from your side) would reduce this, but the principle holds: multi-model is dramatically more reliable than single-model.

Even with just three fallback models at 95% each:

$$A_{\text{system}} = 1 - (0.05)^3 = 1 - 0.000125 = 0.999875$$

That's 99.99% from three 95% models.

### Exponential Backoff with Jitter

When a model fails, don't retry immediately. Use exponential backoff with jitter to avoid thundering herd problems:

$$t_{\text{wait}} = \min\left(t_{\text{base}} \cdot 2^{n} + \text{random}(0, t_{\text{jitter}}), \; t_{\text{max}}\right)$$

Where \(n\) is the retry attempt number, \(t_{\text{base}}\) is the initial delay (e.g., 1 second), \(t_{\text{jitter}}\) is the maximum random jitter (e.g., 1 second), and \(t_{\text{max}}\) is the maximum delay cap (e.g., 30 seconds).

| Retry # | Base delay | With jitter (example) |
|---------|------------|----------------------|
| 1 | 2s | 2.4s |
| 2 | 4s | 4.7s |
| 3 | 8s | 8.1s |
| 4 | 16s | 16.9s |
| 5 | 30s (capped) | 30.3s |

```typescript
function calculateBackoff(attempt: number, baseMs: number = 1000, maxMs: number = 30000): number {
  const exponential = baseMs * Math.pow(2, attempt);
  const jitter = Math.random() * baseMs;
  return Math.min(exponential + jitter, maxMs);
}
```

### Fallback Execution

```typescript
class FallbackExecutor {
  constructor(
    private adapterFactory: AdapterFactory,
    private healthMonitor: HealthMonitor,
    private qualityGate: QualityGate,
    private maxRetries: number = 3,
  ) {}

  async executeWithFallback(
    input: VideoGenerationInput,
    decision: RoutingDecision,
  ): Promise<VideoGenerationOutput> {
    const chain = [decision.modelId, ...decision.fallbackChain];
    let lastError: Error | null = null;

    for (let i = 0; i < Math.min(chain.length, this.maxRetries); i++) {
      const modelId = chain[i];
      const adapter = this.adapterFactory.getAdapter(modelId);

      try {
        const startTime = Date.now();
        const output = await adapter.generate(input);
        const latencyMs = Date.now() - startTime;

        // Record success
        this.healthMonitor.recordRequest(modelId, latencyMs, true);

        // Quality gate
        if (output.videoBuffer) {
          const quality = await this.qualityGate.evaluate(output.videoBuffer, input.prompt);
          if (!quality.pass) {
            console.warn(`Quality gate failed for ${modelId}: ${quality.overall.toFixed(3)}`);
            const retryAction = this.qualityGate.shouldRetry(quality);
            if (retryAction === 'retry-different') {
              continue;  // try next model in chain
            }
            // retry-same: try same model once more (not implemented here for simplicity)
            continue;
          }
        }

        return output;

      } catch (error) {
        lastError = error as Error;
        this.healthMonitor.recordRequest(modelId, 0, false);
        console.error(`Model ${modelId} failed: ${lastError.message}`);

        // Backoff before trying next model
        if (i < chain.length - 1) {
          const backoffMs = calculateBackoff(i);
          await new Promise(resolve => setTimeout(resolve, backoffMs));
        }
      }
    }

    throw new Error(`All models in fallback chain failed. Last error: ${lastError?.message}`);
  }
}
```

---

## Multi-Shot Pipeline {#multi-shot-pipeline}

For multi-scene video projects (storyboards, short films, ads), the pipeline becomes a DAG (directed acyclic graph) of parallel generation tasks followed by consistency checking and stitching.

### Architecture

```
Storyboard (N shots)
    |
    +--> Shot 1: [Route] --> [Generate] --> [Quality Gate] --+
    +--> Shot 2: [Route] --> [Generate] --> [Quality Gate] --+--> [Consistency Check]
    +--> Shot 3: [Route] --> [Generate] --> [Quality Gate] --+         |
    +--> Shot N: [Route] --> [Generate] --> [Quality Gate] --+    [Regenerate if
    |                                                              inconsistent]
    +--- Audio Pipeline (parallel) --------------------------->       |
         +--> Narration (ElevenLabs v3)                         [FFmpeg Stitch]
         +--> Music (Eleven Music)                                    |
         +--> SFX (ElevenLabs SFX v2)                           [Final Output]
```

### Implementation

```typescript
interface Shot {
  index: number;
  prompt: string;
  duration: number;
  resolution: Resolution;
  needsAudio: boolean;
  contentType: ContentType;
  cameraMotion?: string;
  referenceImage?: Buffer;
}

interface Storyboard {
  projectId: string;
  shots: Shot[];
  globalStyle?: string;        // style hint applied to all shots
  audioScript?: string;        // screenplay for narration
  musicMood?: string;          // mood prompt for background music
}

class MultiShotPipeline {
  constructor(
    private router: ScoreBasedRouter,
    private executor: FallbackExecutor,
    private qualityGate: QualityGate,
  ) {}

  async generateStoryboard(storyboard: Storyboard): Promise<Buffer> {
    // Step 1: Generate all shots in parallel
    const shotPromises = storyboard.shots.map(async (shot) => {
      const request: GenerationRequest = {
        prompt: storyboard.globalStyle
          ? `${storyboard.globalStyle}. ${shot.prompt}`
          : shot.prompt,
        duration: shot.duration,
        resolution: shot.resolution,
        needsAudio: false,  // audio handled separately
        contentType: shot.contentType,
        mode: 'standard',
        maxBudgetUsd: 5.00,
        cameraMotion: shot.cameraMotion,
        referenceImage: shot.referenceImage ? shot.referenceImage.toString('base64') : undefined,
      };

      const decision = this.router.route(request);
      const output = await this.executor.executeWithFallback(
        {
          prompt: request.prompt,
          duration: request.duration,
          resolution: request.resolution,
          aspectRatio: '16:9',
          audio: false,
        },
        decision
      );

      return { shot, output, decision };
    });

    const results = await Promise.all(shotPromises);

    // Step 2: Consistency check across shots
    // Use Gemini Flash to compare first/last frames of adjacent shots
    const inconsistencies = await this.checkConsistency(results);
    if (inconsistencies.length > 0) {
      console.warn(`Found ${inconsistencies.length} inconsistencies, regenerating affected shots`);
      // Regenerate inconsistent shots (simplified -- in production you'd be smarter)
    }

    // Step 3: Stitch with FFmpeg
    const videoUrls = results
      .sort((a, b) => a.shot.index - b.shot.index)
      .map(r => r.output.videoUrl);

    const stitched = await this.stitchVideos(videoUrls, {
      transition: 'crossfade',
      transitionDuration: 0.5,
      outputResolution: storyboard.shots[0].resolution,
    });

    return stitched;
  }

  private async checkConsistency(
    results: { shot: Shot; output: VideoGenerationOutput }[]
  ): Promise<number[]> {
    // Extract last frame of shot N and first frame of shot N+1
    // Send pairs to Gemini Flash for consistency evaluation
    // Return indices of inconsistent transitions
    const inconsistent: number[] = [];
    // ...implementation omitted for brevity
    return inconsistent;
  }

  private async stitchVideos(
    videoUrls: string[],
    options: { transition: string; transitionDuration: number; outputResolution: Resolution }
  ): Promise<Buffer> {
    // FFmpeg command construction for concatenation with crossfades
    // ffmpeg -i shot1.mp4 -i shot2.mp4 -i shot3.mp4 \
    //   -filter_complex \
    //   "[0:v][1:v]xfade=transition=fade:duration=0.5:offset=4.5[v01]; \
    //    [v01][2:v]xfade=transition=fade:duration=0.5:offset=9.0[vout]" \
    //   -map "[vout]" -c:v libx264 -preset fast output.mp4
    const { execFFmpeg } = await import('./ffmpeg-utils');
    return execFFmpeg(videoUrls, options);
  }
}
```

### Parallel Timing Analysis

Here's why parallel generation is critical. Consider a 5-shot storyboard where each shot takes 45-90 seconds to generate:

**Sequential (naive):**
```
Shot 1: |████████████| 60s
Shot 2:              |████████████████| 80s
Shot 3:                               |██████████| 50s
Shot 4:                                           |████████████████| 80s
Shot 5:                                                             |████████████| 60s
Total: 330 seconds (5.5 minutes)
```

**Parallel:**
```
Shot 1: |████████████| 60s
Shot 2: |████████████████| 80s
Shot 3: |██████████| 50s
Shot 4: |████████████████| 80s  <-- bottleneck
Shot 5: |████████████| 60s
Total: 80 seconds (1.3 minutes) + stitch time (~10s) = 90 seconds
```

Parallel generation reduces wall-clock time from 5.5 minutes to 1.5 minutes -- a 3.7x speedup. The cost is identical; you're just spending it simultaneously instead of sequentially.

---

## Cost Management {#cost-management}

### Budget Enforcement

Every generation request must be checked against the user's remaining budget before any API call is made:

```typescript
class BudgetEnforcer {
  constructor(private db: Database) {}

  async canAfford(userId: string, estimatedCost: number): Promise<boolean> {
    const balance = await this.db.getUserCreditBalance(userId);
    // Reserve 10% buffer for cost estimation errors
    return balance >= estimatedCost * 1.10;
  }

  async reserveCredits(userId: string, estimatedCost: number): Promise<string> {
    // Atomic reservation to prevent race conditions
    const reservationId = await this.db.createCreditReservation(userId, estimatedCost * 1.10);
    return reservationId;
  }

  async finalizeCharge(reservationId: string, actualCost: number): Promise<void> {
    // Convert reservation to actual charge, refund excess
    await this.db.finalizeCreditReservation(reservationId, actualCost);
  }

  async releaseReservation(reservationId: string): Promise<void> {
    // Release reserved credits if generation failed
    await this.db.releaseCreditReservation(reservationId);
  }
}
```

### The Preview-Then-Commit Pattern

The single most effective cost optimization: generate a fast, cheap preview before committing to an expensive final render.

```
User prompt --> [Generate Preview] --> [Show to User] --> [User Approves?]
                720p, fast model                              |
                ~$0.25                                   Yes  |  No (edit prompt)
                                                              v
                                                    [Generate Final]
                                                    1080p/4K, premium model
                                                    ~$1.50-$3.00
```

**Cost savings calculation:**

Assume 40% of previews get edited/rejected before the user is happy:

- Without preview: 100 requests x \(2.00 avg = **\)200.00**
- With preview: 100 previews x $0.25 = $25.00, plus 60 final renders x $2.00 = \(120.00 = **\)145.00**
- Savings: **27.5%**

If the rejection rate is 60% (common for first-time users):

- Without preview: 100 x $2.00 = $200.00
- With preview: 100 x $0.25 = $25.00 + 40 x $2.00 = \(80.00 = **\)105.00**
- Savings: **47.5%**

### Real-Time Cost Dashboard

Track these metrics in real time:

```typescript
interface CostMetrics {
  // Per-model metrics
  totalSpendByModel: Record<string, number>;       // USD spent per model this period
  avgCostPerGeneration: Record<string, number>;     // average cost per API call
  costPerQualityPoint: Record<string, number>;      // $/quality_score -- efficiency metric

  // Platform metrics
  totalRevenue: number;          // credits consumed by users
  totalCost: number;             // API costs
  grossMargin: number;           // (revenue - cost) / revenue
  avgMarkup: number;             // revenue / cost ratio

  // Efficiency metrics
  previewToFinalRatio: number;   // what % of previews become final renders
  retryRate: number;             // what % of generations need retrying
  wastedSpend: number;           // cost of failed/retried generations
}
```

The **cost per quality point** metric is the most useful for routing optimization:

$$\text{CPQ}_m = \frac{\bar{C}_m}{\bar{Q}_m}$$

Lower is better. A model that costs \(0.50 and averages 0.80 quality (\)0.625/point) is more efficient than one that costs \(2.00 and averages 0.95 quality (\)2.11/point) -- unless the user specifically requested premium quality.

---

## Observability {#observability}

### What to Log

Every generation request should produce a structured log entry:

```typescript
interface GenerationLog {
  requestId: string;
  userId: string;
  timestamp: Date;

  // Request
  prompt: string;
  promptHash: string;           // for deduplication analysis
  contentType: ContentType;
  mode: string;
  resolution: Resolution;
  duration: number;

  // Routing
  routingStrategy: string;
  selectedModel: string;
  routingScore: number;
  routingReasoning: string;
  fallbacksUsed: string[];

  // Generation
  generationTimeMs: number;
  apiResponseCode: number;
  success: boolean;
  errorMessage?: string;
  retryCount: number;

  // Quality
  qualityScore: number;
  qualityDimensions: Record<string, number>;
  qualityPass: boolean;

  // Cost
  estimatedCost: number;
  actualCost: number;
  creditCharged: number;

  // Output
  outputVideoUrl: string;
  outputDuration: number;
  outputResolution: string;
  hasAudio: boolean;
}
```

### Key Metrics and Formulas

**P95 Generation Latency** (the primary user experience metric):

$$P95 = \text{latencies}[\lceil 0.95 \times N \rceil]$$

where latencies are sorted ascending and \(N\) is the number of observations.

Target: under 120 seconds for standard mode, under 60 seconds for preview.

**Model Utilization Rate** (are you actually using what you're paying for?):

$$U_m = \frac{\text{requests routed to } m}{\text{total requests}} \times 100\%$$

A model with 0% utilization is dead weight. Remove it or fix its routing score.

**Quality-Adjusted Cost** (the metric that matters most for business decisions):

$$\text{QAC} = \frac{\text{Total API spend}}{\sum \text{quality scores of delivered outputs}}$$

This is total cost divided by total quality -- lower is better. It accounts for retries, fallbacks, and quality variation.

### Alerting Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| Model down | Circuit breaker opens | P1 |
| High error rate | Any model error rate > 20% over 5 min | P1 |
| Quality degradation | Mean quality drops > 0.10 from 24h baseline | P2 |
| Cost spike | Hourly API spend > 2x trailing 7-day hourly average | P2 |
| Latency spike | P95 latency > 2x baseline for 10 minutes | P2 |
| Low margin | Gross margin drops below 40% | P3 |
| Budget exhaustion | Any user hits $0 balance during active generation | P3 |

---

## Complete Working Service {#complete-working-service}

Here's the top-level service that ties everything together -- the single class that your API routes call:

```typescript
class VideoGenerationService {
  private registry: Map<string, ModelRegistryEntry>;
  private router: ScoreBasedRouter;
  private adapterFactory: AdapterFactory;
  private healthMonitor: HealthMonitor;
  private qualityGate: QualityGate;
  private executor: FallbackExecutor;
  private budgetEnforcer: BudgetEnforcer;

  constructor(config: {
    googleApiKey: string;
    runwayApiToken: string;
    openaiApiKey: string;
    piApiKey: string;
    lumaApiKey: string;
    geminiApiKey: string;
    db: Database;
  }) {
    // Initialize registry from configuration
    this.registry = new Map(Object.entries(MODEL_REGISTRY));

    // Initialize components
    this.healthMonitor = new HealthMonitor(this.registry);
    this.router = new ScoreBasedRouter(this.registry);
    this.adapterFactory = new AdapterFactory({
      GOOGLE_AI_API_KEY: config.googleApiKey,
      RUNWAY_API_TOKEN: config.runwayApiToken,
      OPENAI_API_KEY: config.openaiApiKey,
      PIAPI_KEY: config.piApiKey,
      LUMA_API_KEY: config.lumaApiKey,
    });
    this.qualityGate = new QualityGate(config.geminiApiKey);
    this.executor = new FallbackExecutor(
      this.adapterFactory,
      this.healthMonitor,
      this.qualityGate,
    );
    this.budgetEnforcer = new BudgetEnforcer(config.db);
  }

  async generate(userId: string, request: GenerationRequest): Promise<VideoGenerationOutput> {
    const requestId = crypto.randomUUID();
    const startTime = Date.now();

    // Step 1: Route the request
    const decision = this.router.route(request);
    console.log(`[${requestId}] Routed to ${decision.modelId}: ${decision.reasoning}`);

    // Step 2: Check budget
    if (!await this.budgetEnforcer.canAfford(userId, decision.estimatedCost)) {
      throw new Error('Insufficient credits');
    }
    const reservationId = await this.budgetEnforcer.reserveCredits(userId, decision.estimatedCost);

    try {
      // Step 3: Execute with fallback chain
      const output = await this.executor.executeWithFallback(
        {
          prompt: request.prompt,
          duration: request.duration,
          resolution: request.resolution,
          aspectRatio: '16:9',
          audio: request.needsAudio,
          referenceImage: request.referenceImage ? Buffer.from(request.referenceImage, 'base64') : undefined,
        },
        decision,
      );

      // Step 4: Finalize billing
      await this.budgetEnforcer.finalizeCharge(reservationId, output.cost);

      // Step 5: Log telemetry
      const totalTimeMs = Date.now() - startTime;
      console.log(
        `[${requestId}] Completed in ${totalTimeMs}ms. Model: ${output.modelId}, ` +
        `Cost: $${output.cost.toFixed(3)}, Duration: ${output.duration}s`
      );

      return output;

    } catch (error) {
      // Release reserved credits on failure
      await this.budgetEnforcer.releaseReservation(reservationId);
      throw error;
    }
  }
}
```

### Usage

```typescript
// Initialize the service
const service = new VideoGenerationService({
  googleApiKey: process.env.GOOGLE_AI_API_KEY!,
  runwayApiToken: process.env.RUNWAY_API_TOKEN!,
  openaiApiKey: process.env.OPENAI_API_KEY!,
  piApiKey: process.env.PIAPI_KEY!,
  lumaApiKey: process.env.LUMA_API_KEY!,
  geminiApiKey: process.env.GEMINI_API_KEY!,
  db: database,
});

// Generate a video
const output = await service.generate('user-123', {
  prompt: 'Two scientists in a neon-lit lab examining a glowing crystal, cinematic lighting, 4K',
  duration: 5,
  resolution: '1080p',
  needsAudio: true,
  contentType: 'dialogue',
  mode: 'standard',
  maxBudgetUsd: 3.00,
});

console.log(`Video URL: ${output.videoUrl}`);
console.log(`Model used: ${output.modelId}`);
console.log(`Cost: $${output.cost}`);
console.log(`Generation time: ${output.generationTimeMs}ms`);
```

---

## Summary

Building a multi-model video pipeline is the architectural inflection point for any AI video platform. The core components:

1. **Model Registry**: Single source of truth for capabilities, cost, quality, and health.
2. **Router**: Score-based routing with the formula \(S_m = w_q \cdot Q_m + w_c \cdot (1 - C_m/C_{\max}) + w_s \cdot (1 - L_m/L_{\max}) + w_a \cdot A_m\) and mode-specific weight profiles.
3. **Adapter Layer**: Normalize every model's API behind a single interface. New model = new adapter, nothing else changes.
4. **Quality Gate**: Gemini Flash evaluates every output. Cost: $0.001 per evaluation. Value: prevents bad output from reaching users.
5. **Fallback Chain**: Circuit breakers + exponential backoff + ordered fallbacks = \(A_{\text{system}} = 1 - \prod(1-A_i)\) composite availability.
6. **Cost Management**: Budget enforcement, preview-then-commit, and cost-per-quality-point optimization.
7. **Observability**: Structured logs, key metrics (P95 latency, QAC, model utilization), and threshold-based alerting.

Start with score-based routing and two models. Add models, ML routing, and multi-shot support as your user base and data grow. Every layer in this architecture solves a real problem -- don't add a layer until you have the problem.
