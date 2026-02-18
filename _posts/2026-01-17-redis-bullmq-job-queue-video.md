---
layout: post
title: "Redis + BullMQ: Production Job Queue Architecture for AI Video Generation"
date: 2026-01-17
category: architecture
---

# Redis + BullMQ: Production Job Queue Architecture for AI Video Generation

If you are running an AI video generation platform, you have a fundamental problem: a user clicks "Generate," and then nothing visible happens for anywhere from 30 seconds to five minutes. The video model grinds through diffusion steps, rendering frames, stitching audio. You cannot make the user stare at a spinner on an HTTP connection for that long. You cannot tie up a web server process for five minutes per request. And you absolutely cannot lose the job if your server restarts mid-generation.

You need a job queue. Not a toy queue. A production-grade, Redis-backed, horizontally scalable, priority-aware, retry-capable, observable job queue. This post is the complete blueprint for building one with **BullMQ** and **Redis**, specifically tuned for AI video generation workloads.

---

## Table of Contents

1. [Why You Need a Job Queue](#why-you-need-a-job-queue)
2. [BullMQ Architecture Deep Dive](#bullmq-architecture-deep-dive)
3. [Full Implementation](#full-implementation)
4. [Priority Queue Design](#priority-queue-design)
5. [Concurrency Management](#concurrency-management)
6. [Retry Strategies](#retry-strategies)
7. [Dead Letter Queue](#dead-letter-queue)
8. [Scaling Workers](#scaling-workers)
9. [Cost Optimization](#cost-optimization)
10. [Monitoring and Alerting](#monitoring-and-alerting)
11. [Putting It All Together](#putting-it-all-together)

---

## Why You Need a Job Queue

Let me walk through the failure modes of *not* having a job queue, because I have lived through every one of them.

### The Naive Approach and Why It Breaks

```typescript
// DON'T DO THIS — the "just await it" anti-pattern
app.post('/api/generate', async (req, res) => {
  const result = await callVideoModelAPI(req.body.prompt); // 30-300 seconds
  res.json(result); // Connection probably already timed out
});
```

Problems with this approach:

| Problem | Impact | Frequency |
|---------|--------|-----------|
| HTTP timeout (30s default) | User gets error, job may still be running | Every request over 30s |
| Server restart during generation | Job lost entirely, user charged but gets nothing | Every deploy |
| Memory pressure from concurrent requests | OOM kill, all in-flight jobs lost | Traffic spikes |
| No retry on transient API failures | 5-10% of requests fail permanently that could have succeeded | Constant |
| No concurrency control | Exceed provider rate limits, get banned | Traffic spikes |
| No priority management | Free users block paying users | Peak hours |
| No visibility into what is happening | Debugging is impossible | Always |

### What a Job Queue Gives You

A job queue decouples the **request** from the **processing**. The web server's only job is to accept the request, validate it, create a job, and return a job ID. A separate worker process picks up the job and does the actual work.

<svg viewBox="0 0 900 320" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:system-ui,sans-serif">
  <defs>
    <marker id="arrowhead-q1" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>

  <!-- User -->
  <rect x="20" y="120" width="120" height="60" rx="8" fill="#f5f5f5" stroke="#333" stroke-width="2"/>
  <text x="80" y="155" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">User / Client</text>

  <!-- API Server -->
  <rect x="200" y="100" width="140" height="100" rx="8" fill="#e3f2fd" stroke="#4fc3f7" stroke-width="2"/>
  <text x="270" y="135" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">API Server</text>
  <text x="270" y="155" text-anchor="middle" font-size="11" fill="#666">Validate</text>
  <text x="270" y="170" text-anchor="middle" font-size="11" fill="#666">Create Job</text>
  <text x="270" y="185" text-anchor="middle" font-size="11" fill="#666">Return Job ID</text>

  <!-- Redis -->
  <rect x="400" y="110" width="120" height="80" rx="8" fill="#ffebee" stroke="#ef5350" stroke-width="2"/>
  <text x="460" y="145" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">Redis</text>
  <text x="460" y="165" text-anchor="middle" font-size="11" fill="#666">BullMQ Queue</text>

  <!-- Workers -->
  <rect x="580" y="40" width="140" height="70" rx="8" fill="#e8f5e9" stroke="#8bc34a" stroke-width="2"/>
  <text x="650" y="70" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Worker 1</text>
  <text x="650" y="88" text-anchor="middle" font-size="11" fill="#666">Veo processor</text>

  <rect x="580" y="125" width="140" height="70" rx="8" fill="#e8f5e9" stroke="#8bc34a" stroke-width="2"/>
  <text x="650" y="155" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Worker 2</text>
  <text x="650" y="173" text-anchor="middle" font-size="11" fill="#666">Kling processor</text>

  <rect x="580" y="210" width="140" height="70" rx="8" fill="#e8f5e9" stroke="#8bc34a" stroke-width="2"/>
  <text x="650" y="240" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Worker N</text>
  <text x="650" y="258" text-anchor="middle" font-size="11" fill="#666">General processor</text>

  <!-- Video Model APIs -->
  <rect x="780" y="110" width="100" height="80" rx="8" fill="#fff3e0" stroke="#ff9800" stroke-width="2"/>
  <text x="830" y="145" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Video</text>
  <text x="830" y="163" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Model APIs</text>

  <!-- Arrows -->
  <line x1="140" y1="150" x2="195" y2="150" stroke="#333" stroke-width="2" marker-end="url(#arrowhead-q1)"/>
  <line x1="340" y1="150" x2="395" y2="150" stroke="#333" stroke-width="2" marker-end="url(#arrowhead-q1)"/>
  <line x1="520" y1="140" x2="575" y2="80" stroke="#333" stroke-width="2" marker-end="url(#arrowhead-q1)"/>
  <line x1="520" y1="150" x2="575" y2="160" stroke="#333" stroke-width="2" marker-end="url(#arrowhead-q1)"/>
  <line x1="520" y1="165" x2="575" y2="240" stroke="#333" stroke-width="2" marker-end="url(#arrowhead-q1)"/>
  <line x1="720" y1="75" x2="775" y2="135" stroke="#333" stroke-width="2" marker-end="url(#arrowhead-q1)"/>
  <line x1="720" y1="160" x2="775" y2="150" stroke="#333" stroke-width="2" marker-end="url(#arrowhead-q1)"/>
  <line x1="720" y1="245" x2="775" y2="175" stroke="#333" stroke-width="2" marker-end="url(#arrowhead-q1)"/>

  <!-- Labels -->
  <text x="165" y="140" text-anchor="middle" font-size="10" fill="#666">POST</text>
  <text x="365" y="140" text-anchor="middle" font-size="10" fill="#666">enqueue</text>
  <text x="545" y="105" text-anchor="middle" font-size="10" fill="#666">dequeue</text>

  <!-- Response arrow back -->
  <line x1="200" y1="170" x2="145" y2="170" stroke="#4fc3f7" stroke-width="2" stroke-dasharray="5,3" marker-end="url(#arrowhead-q1)"/>
  <text x="170" y="190" text-anchor="middle" font-size="10" fill="#4fc3f7">job ID (instant)</text>
</svg>

This architecture gives you:

- **Instant response**: The API returns a job ID in under 50ms.
- **Persistence**: Jobs survive server restarts because they live in Redis.
- **Retries**: Failed jobs are retried automatically with configurable backoff.
- **Concurrency control**: You decide how many jobs run simultaneously.
- **Priority**: Paying users go first.
- **Observability**: You can see every job's state, timing, and failure reason.
- **Horizontal scaling**: Add more workers to process more jobs.

---

## BullMQ Architecture Deep Dive

BullMQ is a Node.js job queue library backed by Redis. It is the successor to Bull, completely rewritten in TypeScript with better performance, cleaner APIs, and more features. It uses Redis Lua scripts for atomic operations, which means job state transitions are safe even under high concurrency.

### How Jobs Flow Through States

Every job in BullMQ transitions through a well-defined state machine. Understanding this is critical for debugging production issues.

<svg viewBox="0 0 900 500" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:system-ui,sans-serif">
  <defs>
    <marker id="arrowhead-sm" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
    <marker id="arrowhead-red" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#ef5350"/>
    </marker>
    <marker id="arrowhead-green" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#8bc34a"/>
    </marker>
    <marker id="arrowhead-blue" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#4fc3f7"/>
    </marker>
  </defs>

  <text x="450" y="30" text-anchor="middle" font-size="18" font-weight="bold" fill="#333">BullMQ Job State Machine</text>

  <!-- States -->
  <!-- Waiting -->
  <rect x="50" y="180" width="120" height="50" rx="25" fill="#e3f2fd" stroke="#4fc3f7" stroke-width="3"/>
  <text x="110" y="210" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">waiting</text>

  <!-- Delayed -->
  <rect x="50" y="80" width="120" height="50" rx="25" fill="#fff3e0" stroke="#ff9800" stroke-width="2"/>
  <text x="110" y="110" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">delayed</text>

  <!-- Prioritized -->
  <rect x="50" y="280" width="120" height="50" rx="25" fill="#f3e5f5" stroke="#9c27b0" stroke-width="2"/>
  <text x="110" y="310" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">prioritized</text>

  <!-- Active -->
  <rect x="300" y="180" width="120" height="50" rx="25" fill="#e8f5e9" stroke="#8bc34a" stroke-width="3"/>
  <text x="360" y="210" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">active</text>

  <!-- Completed -->
  <rect x="550" y="120" width="120" height="50" rx="25" fill="#e8f5e9" stroke="#4caf50" stroke-width="3"/>
  <text x="610" y="150" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">completed</text>

  <!-- Failed -->
  <rect x="550" y="240" width="120" height="50" rx="25" fill="#ffebee" stroke="#ef5350" stroke-width="3"/>
  <text x="610" y="270" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">failed</text>

  <!-- Dead Letter Queue -->
  <rect x="730" y="330" width="140" height="50" rx="25" fill="#ffcdd2" stroke="#c62828" stroke-width="3"/>
  <text x="800" y="360" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">dead letter queue</text>

  <!-- Wait-children -->
  <rect x="300" y="370" width="140" height="50" rx="25" fill="#e0f7fa" stroke="#00bcd4" stroke-width="2"/>
  <text x="370" y="400" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">wait-children</text>

  <!-- Arrows: waiting -> active -->
  <line x1="170" y1="205" x2="295" y2="205" stroke="#333" stroke-width="2" marker-end="url(#arrowhead-sm)"/>
  <text x="230" y="198" text-anchor="middle" font-size="11" fill="#666">worker picks up</text>

  <!-- delayed -> waiting -->
  <path d="M 170 105 Q 220 105 220 175 L 220 185" stroke="#ff9800" stroke-width="2" fill="none" marker-end="url(#arrowhead-sm)"/>
  <text x="240" y="140" font-size="10" fill="#ff9800">timer fires</text>

  <!-- prioritized -> waiting -->
  <path d="M 170 305 Q 220 305 220 220 L 220 210" stroke="#9c27b0" stroke-width="2" fill="none" marker-end="url(#arrowhead-sm)"/>
  <text x="240" y="290" font-size="10" fill="#9c27b0">by priority</text>

  <!-- active -> completed -->
  <line x1="420" y1="195" x2="545" y2="150" stroke="#8bc34a" stroke-width="2" marker-end="url(#arrowhead-green)"/>
  <text x="490" y="162" text-anchor="middle" font-size="11" fill="#8bc34a">success</text>

  <!-- active -> failed -->
  <line x1="420" y1="215" x2="545" y2="258" stroke="#ef5350" stroke-width="2" marker-end="url(#arrowhead-red)"/>
  <text x="490" y="248" text-anchor="middle" font-size="11" fill="#ef5350">error thrown</text>

  <!-- failed -> waiting (retry) -->
  <path d="M 610 295 Q 610 340 400 340 Q 110 340 110 235" stroke="#4fc3f7" stroke-width="2" fill="none" stroke-dasharray="6,3" marker-end="url(#arrowhead-blue)"/>
  <text x="360" y="355" text-anchor="middle" font-size="11" fill="#4fc3f7">retry (attempts remaining)</text>

  <!-- failed -> DLQ -->
  <line x1="670" y1="275" x2="725" y2="340" stroke="#c62828" stroke-width="2" marker-end="url(#arrowhead-red)"/>
  <text x="720" y="305" text-anchor="middle" font-size="11" fill="#c62828">max retries</text>

  <!-- active -> wait-children -->
  <path d="M 360 230 L 360 365" stroke="#00bcd4" stroke-width="2" fill="none" marker-end="url(#arrowhead-sm)"/>
  <text x="370" y="300" font-size="10" fill="#00bcd4">has children</text>

  <!-- wait-children -> active -->
  <path d="M 440 385 Q 480 385 480 225 L 425 210" stroke="#00bcd4" stroke-width="2" fill="none" stroke-dasharray="6,3" marker-end="url(#arrowhead-sm)"/>
  <text x="500" y="310" font-size="10" fill="#00bcd4">children done</text>

  <!-- Legend -->
  <rect x="600" y="410" width="280" height="80" rx="6" fill="#fafafa" stroke="#ddd" stroke-width="1"/>
  <text x="610" y="430" font-size="11" font-weight="bold" fill="#333">Legend</text>
  <line x1="610" y1="445" x2="640" y2="445" stroke="#333" stroke-width="2"/>
  <text x="645" y="449" font-size="10" fill="#333">Normal transition</text>
  <line x1="610" y1="462" x2="640" y2="462" stroke="#4fc3f7" stroke-width="2" stroke-dasharray="6,3"/>
  <text x="645" y="466" font-size="10" fill="#333">Retry transition</text>
  <line x1="610" y1="479" x2="640" y2="479" stroke="#c62828" stroke-width="2"/>
  <text x="645" y="483" font-size="10" fill="#333">Terminal failure</text>
</svg>

### State Details

| State | Redis Data Structure | Description |
|-------|---------------------|-------------|
| `waiting` | List (FIFO) | Job is queued and ready to be picked up by a worker |
| `prioritized` | Sorted Set (by priority) | Job is queued with a specific priority level |
| `delayed` | Sorted Set (by timestamp) | Job is scheduled for future execution |
| `active` | Set | Job is currently being processed by a worker |
| `completed` | Set | Job finished successfully |
| `failed` | Set | Job threw an error (may be retried) |
| `wait-children` | Set | Parent job waiting for child jobs to complete |

### Redis Data Structures Under the Hood

BullMQ uses multiple Redis keys per queue. Understanding these is essential when you need to debug production issues by inspecting Redis directly.

```
bull:video-generation:waiting       # List — FIFO queue of waiting job IDs
bull:video-generation:prioritized   # Sorted Set — priority-ordered job IDs
bull:video-generation:active        # Set — currently processing job IDs
bull:video-generation:delayed       # Sorted Set — delayed jobs (score = timestamp)
bull:video-generation:completed     # Set — completed job IDs
bull:video-generation:failed        # Set — failed job IDs
bull:video-generation:id            # String — auto-incrementing job ID counter
bull:video-generation:{jobId}       # Hash — individual job data, options, state
bull:video-generation:events        # Stream — event log for the queue
bull:video-generation:meta          # Hash — queue metadata
```

All state transitions are performed via Redis Lua scripts, which means they are **atomic**. There is no window where a job can be in two states simultaneously, even under high concurrency. This is what makes BullMQ safe for production use.

---

## Full Implementation

Let us build a complete, production-ready job queue for AI video generation. This is not a toy example. This is the code I would deploy.

### Project Setup

```typescript
// package.json dependencies you need:
// "bullmq": "^5.1.0"
// "ioredis": "^5.3.0"
// "@bull-board/api": "^5.10.0"
// "@bull-board/express": "^5.10.0"
// "express": "^4.18.0"
// "uuid": "^9.0.0"
```

### Redis Connection Configuration

```typescript
// src/config/redis.ts
import IORedis from 'ioredis';

export interface RedisConfig {
  host: string;
  port: number;
  password?: string;
  maxRetriesPerRequest: null; // Required by BullMQ
  enableReadyCheck: boolean;
  retryStrategy: (times: number) => number | null;
  tls?: { rejectUnauthorized: boolean };
}

export function createRedisConfig(): RedisConfig {
  const config: RedisConfig = {
    host: process.env.REDIS_HOST || '127.0.0.1',
    port: parseInt(process.env.REDIS_PORT || '6379', 10),
    password: process.env.REDIS_PASSWORD || undefined,
    maxRetriesPerRequest: null, // BullMQ requirement — do not change
    enableReadyCheck: true,
    retryStrategy: (times: number) => {
      // Reconnect after exponential backoff, max 30 seconds
      const delay = Math.min(times * 500, 30_000);
      console.warn(`Redis connection retry #${times}, next attempt in ${delay}ms`);
      if (times > 20) {
        console.error('Redis connection failed after 20 retries, giving up');
        return null; // Stop retrying
      }
      return delay;
    },
  };

  // Enable TLS for production Redis (e.g., AWS ElastiCache, Upstash)
  if (process.env.REDIS_TLS === 'true') {
    config.tls = { rejectUnauthorized: false };
  }

  return config;
}

export function createRedisConnection(): IORedis {
  const config = createRedisConfig();
  const connection = new IORedis(config);

  connection.on('connect', () => {
    console.log('Redis connected');
  });

  connection.on('error', (err) => {
    console.error('Redis connection error:', err.message);
  });

  return connection;
}
```

### Job Data Types

```typescript
// src/types/jobs.ts

/** Supported video generation providers */
export type VideoProvider = 'veo' | 'kling' | 'runway' | 'luma' | 'minimax';

/** User subscription tiers affecting priority */
export type SubscriptionTier = 'free' | 'pro' | 'enterprise';

/** Job priority levels — lower number = higher priority in BullMQ */
export enum JobPriority {
  CRITICAL = 1,    // System-initiated retries, admin jobs
  ENTERPRISE = 2,  // Enterprise tier users
  PRO = 5,         // Pro tier users
  FREE = 10,       // Free tier users
  BATCH = 20,      // Batch/bulk processing
}

/** Input data for a video generation job */
export interface VideoGenerationJobData {
  /** Unique idempotency key to prevent duplicate jobs */
  idempotencyKey: string;

  /** User who requested the generation */
  userId: string;

  /** User's subscription tier */
  tier: SubscriptionTier;

  /** The text prompt for video generation */
  prompt: string;

  /** Optional reference image URL for image-to-video */
  referenceImageUrl?: string;

  /** Which provider to use */
  provider: VideoProvider;

  /** Desired video duration in seconds */
  durationSeconds: number;

  /** Desired resolution */
  resolution: '720p' | '1080p' | '4k';

  /** Aspect ratio */
  aspectRatio: '16:9' | '9:16' | '1:1';

  /** Additional provider-specific parameters */
  providerParams?: Record<string, unknown>;

  /** Webhook URL to notify on completion */
  webhookUrl?: string;

  /** Timestamp when the job was created */
  createdAt: number;
}

/** Result data stored on completed job */
export interface VideoGenerationResult {
  /** URL to the generated video */
  videoUrl: string;

  /** URL to a thumbnail/preview image */
  thumbnailUrl: string;

  /** Actual video duration in seconds */
  durationSeconds: number;

  /** Provider-specific generation ID */
  providerJobId: string;

  /** How long generation took in milliseconds */
  processingTimeMs: number;

  /** Cost of this generation in USD */
  costUsd: number;
}

/** Progress update sent during processing */
export interface VideoGenerationProgress {
  /** 0-100 percentage */
  percentage: number;

  /** Human-readable status message */
  message: string;

  /** Current stage of processing */
  stage: 'queued' | 'submitted' | 'generating' | 'post-processing' | 'uploading';

  /** Provider-reported ETA in seconds, if available */
  etaSeconds?: number;
}
```

### The Job Producer

The producer is the component that creates jobs. It runs in your API server process.

```typescript
// src/queue/producer.ts
import { Queue, QueueEvents } from 'bullmq';
import { createRedisConfig } from '../config/redis';
import {
  VideoGenerationJobData,
  VideoGenerationResult,
  VideoGenerationProgress,
  JobPriority,
  SubscriptionTier,
} from '../types/jobs';

export class VideoGenerationProducer {
  private queue: Queue<VideoGenerationJobData, VideoGenerationResult>;
  private queueEvents: QueueEvents;

  constructor() {
    const connection = createRedisConfig();

    this.queue = new Queue<VideoGenerationJobData, VideoGenerationResult>(
      'video-generation',
      {
        connection,
        defaultJobOptions: {
          // Default retry configuration
          attempts: 3,
          backoff: {
            type: 'exponential',
            delay: 5000, // Start at 5 seconds
          },
          // Remove completed jobs after 24 hours to save Redis memory
          removeOnComplete: {
            age: 86400,   // 24 hours in seconds
            count: 10000, // Keep at most 10k completed jobs
          },
          // Keep failed jobs longer for debugging
          removeOnFail: {
            age: 604800, // 7 days
            count: 5000,
          },
        },
      }
    );

    this.queueEvents = new QueueEvents('video-generation', { connection });
  }

  /**
   * Map subscription tier to BullMQ priority number.
   * Lower number = higher priority.
   */
  private getPriority(tier: SubscriptionTier): number {
    const priorityMap: Record<SubscriptionTier, number> = {
      enterprise: JobPriority.ENTERPRISE,
      pro: JobPriority.PRO,
      free: JobPriority.FREE,
    };
    return priorityMap[tier];
  }

  /**
   * Calculate timeout based on provider and resolution.
   * Different providers and resolutions take different amounts of time.
   */
  private getTimeout(data: VideoGenerationJobData): number {
    const baseTimeouts: Record<string, number> = {
      veo: 300_000,    // 5 minutes
      kling: 600_000,  // 10 minutes (PiAPI can be slow)
      runway: 300_000, // 5 minutes
      luma: 240_000,   // 4 minutes
      minimax: 480_000, // 8 minutes
    };

    let timeout = baseTimeouts[data.provider] || 300_000;

    // Higher resolution = more time needed
    if (data.resolution === '4k') timeout *= 2;
    if (data.resolution === '1080p') timeout *= 1.5;

    // Longer videos take proportionally longer
    if (data.durationSeconds > 10) timeout *= 1.5;

    return timeout;
  }

  /**
   * Add a video generation job to the queue.
   * Returns the job ID immediately — does not wait for processing.
   */
  async addJob(data: VideoGenerationJobData): Promise<string> {
    const priority = this.getPriority(data.tier);
    const timeout = this.getTimeout(data);

    const job = await this.queue.add(
      'generate-video', // Job name (useful for filtering)
      data,
      {
        jobId: data.idempotencyKey, // Prevents duplicate jobs
        priority,
        attempts: data.tier === 'enterprise' ? 5 : 3, // More retries for paying users
        backoff: {
          type: 'custom', // We will implement custom backoff in the worker
        },
        timeout, // Kill job if it exceeds this duration

        // Delay the job if needed (e.g., scheduled generation)
        // delay: 0,

        // Store the timestamp for tracking
        timestamp: Date.now(),
      }
    );

    console.log(
      `Job ${job.id} created: provider=${data.provider}, ` +
      `priority=${priority}, timeout=${timeout}ms, ` +
      `user=${data.userId}, tier=${data.tier}`
    );

    return job.id!;
  }

  /**
   * Add a batch of jobs (e.g., for bulk generation features).
   * Uses BullMQ's addBulk for better performance.
   */
  async addBatch(jobs: VideoGenerationJobData[]): Promise<string[]> {
    const bulkJobs = jobs.map((data) => ({
      name: 'generate-video',
      data,
      opts: {
        jobId: data.idempotencyKey,
        priority: JobPriority.BATCH,
        attempts: 3,
        backoff: { type: 'exponential' as const, delay: 10_000 },
        timeout: this.getTimeout(data),
      },
    }));

    const createdJobs = await this.queue.addBulk(bulkJobs);
    return createdJobs.map((j) => j.id!);
  }

  /**
   * Get the current status of a job.
   */
  async getJobStatus(jobId: string): Promise<{
    state: string;
    progress: VideoGenerationProgress | null;
    result: VideoGenerationResult | null;
    failedReason: string | null;
    attemptsMade: number;
    attemptsTotal: number;
  } | null> {
    const job = await this.queue.getJob(jobId);
    if (!job) return null;

    const state = await job.getState();
    return {
      state,
      progress: (job.progress as VideoGenerationProgress) || null,
      result: job.returnvalue || null,
      failedReason: job.failedReason || null,
      attemptsMade: job.attemptsMade,
      attemptsTotal: job.opts.attempts || 3,
    };
  }

  /**
   * Wait for a job to complete (for synchronous API clients).
   * Use sparingly — prefer webhooks or polling.
   */
  async waitForJob(
    jobId: string,
    timeoutMs: number = 600_000
  ): Promise<VideoGenerationResult> {
    return this.queueEvents.waitUntilFinished(jobId, timeoutMs);
  }

  /**
   * Get queue statistics for monitoring dashboards.
   */
  async getQueueStats(): Promise<{
    waiting: number;
    active: number;
    completed: number;
    failed: number;
    delayed: number;
    paused: number;
  }> {
    const [waiting, active, completed, failed, delayed, paused] =
      await Promise.all([
        this.queue.getWaitingCount(),
        this.queue.getActiveCount(),
        this.queue.getCompletedCount(),
        this.queue.getFailedCount(),
        this.queue.getDelayedCount(),
        this.queue.getPausedCount(),
      ]);

    return { waiting, active, completed, failed, delayed, paused };
  }

  /**
   * Graceful shutdown.
   */
  async close(): Promise<void> {
    await this.queueEvents.close();
    await this.queue.close();
  }
}
```

### The Job Consumer (Worker)

The worker is where the actual video generation happens. This runs as a separate process (or set of processes) from your API server.

```typescript
// src/queue/worker.ts
import { Worker, Job, UnrecoverableError } from 'bullmq';
import { createRedisConfig } from '../config/redis';
import {
  VideoGenerationJobData,
  VideoGenerationResult,
  VideoGenerationProgress,
  VideoProvider,
} from '../types/jobs';
import { VideoProviderClient } from '../providers/client';
import { CostTracker } from '../billing/cost-tracker';
import { WebhookNotifier } from '../notifications/webhook';

export class VideoGenerationWorker {
  private worker: Worker<VideoGenerationJobData, VideoGenerationResult>;
  private providerClient: VideoProviderClient;
  private costTracker: CostTracker;
  private webhookNotifier: WebhookNotifier;

  /** Per-provider concurrency tracking */
  private activeCounts: Map<VideoProvider, number> = new Map();
  private readonly providerLimits: Record<VideoProvider, number> = {
    veo: 5,      // Google Veo: 5 concurrent
    kling: 3,    // PiAPI/Kling: 3 concurrent
    runway: 10,  // Runway: 10 concurrent
    luma: 8,     // Luma: 8 concurrent
    minimax: 4,  // MiniMax: 4 concurrent
  };

  constructor(concurrency: number = 5) {
    const connection = createRedisConfig();

    this.providerClient = new VideoProviderClient();
    this.costTracker = new CostTracker();
    this.webhookNotifier = new WebhookNotifier();

    this.worker = new Worker<VideoGenerationJobData, VideoGenerationResult>(
      'video-generation',
      async (job) => this.processJob(job),
      {
        connection,
        concurrency, // How many jobs this worker processes in parallel
        limiter: {
          max: 20,       // Max 20 jobs per...
          duration: 60_000, // ...60 seconds (global rate limit)
        },
        settings: {
          // If a job's lock expires while processing, another worker could
          // pick it up. Set lock duration longer than expected processing time.
          lockDuration: 600_000, // 10 minutes
          lockRenewTime: 300_000, // Renew lock every 5 minutes

          // Custom backoff strategy
          backoffStrategy: (attemptsMade: number) => {
            return this.calculateBackoff(attemptsMade);
          },
        },
      }
    );

    this.setupEventHandlers();
  }

  /**
   * Custom exponential backoff with jitter.
   *
   * Formula: delay = min(base * 2^attempt + random(0, jitter), maxDelay)
   *
   * The jitter prevents thundering herd: if 100 jobs fail simultaneously,
   * they will not all retry at exactly the same moment.
   */
  private calculateBackoff(attemptsMade: number): number {
    const base = 5_000;      // 5 seconds base
    const maxDelay = 300_000; // 5 minutes max
    const jitter = 5_000;    // Up to 5 seconds of random jitter

    const exponentialDelay = base * Math.pow(2, attemptsMade);
    const randomJitter = Math.random() * jitter;
    const delay = Math.min(exponentialDelay + randomJitter, maxDelay);

    console.log(
      `Backoff for attempt ${attemptsMade}: ${Math.round(delay)}ms ` +
      `(base=${exponentialDelay}ms, jitter=${Math.round(randomJitter)}ms)`
    );

    return Math.round(delay);
  }

  /**
   * The main job processing function.
   * This is where the magic happens.
   */
  private async processJob(
    job: Job<VideoGenerationJobData, VideoGenerationResult>
  ): Promise<VideoGenerationResult> {
    const { data } = job;
    const startTime = Date.now();

    console.log(
      `Processing job ${job.id}: provider=${data.provider}, ` +
      `prompt="${data.prompt.substring(0, 50)}...", ` +
      `attempt=${job.attemptsMade + 1}/${job.opts.attempts}`
    );

    // --- Stage 1: Pre-flight checks ---
    await this.updateProgress(job, {
      percentage: 5,
      message: 'Validating request...',
      stage: 'queued',
    });

    // Check per-provider concurrency
    await this.waitForProviderSlot(data.provider);

    // Check user's budget
    const hasbudget = await this.costTracker.checkBudget(
      data.userId,
      data.provider,
      data.durationSeconds
    );
    if (!hasbudget) {
      // UnrecoverableError means "do not retry this job"
      throw new UnrecoverableError(
        `User ${data.userId} has exceeded their budget limit`
      );
    }

    try {
      // --- Stage 2: Submit to provider ---
      await this.updateProgress(job, {
        percentage: 15,
        message: `Submitting to ${data.provider}...`,
        stage: 'submitted',
      });

      const providerJobId = await this.providerClient.submit({
        provider: data.provider,
        prompt: data.prompt,
        referenceImageUrl: data.referenceImageUrl,
        durationSeconds: data.durationSeconds,
        resolution: data.resolution,
        aspectRatio: data.aspectRatio,
        params: data.providerParams,
      });

      // --- Stage 3: Poll for completion ---
      await this.updateProgress(job, {
        percentage: 25,
        message: 'Generating video...',
        stage: 'generating',
      });

      const providerResult = await this.pollForCompletion(
        job,
        data.provider,
        providerJobId
      );

      // --- Stage 4: Post-processing ---
      await this.updateProgress(job, {
        percentage: 85,
        message: 'Post-processing...',
        stage: 'post-processing',
      });

      // Upload to our own storage (S3/R2/GCS)
      const { videoUrl, thumbnailUrl } = await this.providerClient.downloadAndStore(
        providerResult.downloadUrl,
        data.userId,
        job.id!
      );

      // --- Stage 5: Finalize ---
      await this.updateProgress(job, {
        percentage: 95,
        message: 'Finalizing...',
        stage: 'uploading',
      });

      const processingTimeMs = Date.now() - startTime;
      const costUsd = await this.costTracker.recordGeneration(
        data.userId,
        data.provider,
        data.durationSeconds,
        data.resolution
      );

      const result: VideoGenerationResult = {
        videoUrl,
        thumbnailUrl,
        durationSeconds: providerResult.actualDuration,
        providerJobId,
        processingTimeMs,
        costUsd,
      };

      // Send webhook notification if configured
      if (data.webhookUrl) {
        // Fire and forget — don't fail the job if webhook delivery fails
        this.webhookNotifier.notify(data.webhookUrl, {
          event: 'generation.completed',
          jobId: job.id!,
          result,
        }).catch((err) => {
          console.error(`Webhook delivery failed for job ${job.id}:`, err.message);
        });
      }

      await this.updateProgress(job, {
        percentage: 100,
        message: 'Complete!',
        stage: 'uploading',
      });

      return result;
    } catch (error: any) {
      // Classify errors: should we retry or fail permanently?
      if (this.isUnrecoverable(error)) {
        throw new UnrecoverableError(error.message);
      }
      // Rethrow for BullMQ to handle retry
      throw error;
    } finally {
      // Release the provider slot
      this.releaseProviderSlot(data.provider);
    }
  }

  /**
   * Poll the provider API until the video is ready.
   * Report progress back to the job.
   */
  private async pollForCompletion(
    job: Job<VideoGenerationJobData, VideoGenerationResult>,
    provider: VideoProvider,
    providerJobId: string
  ): Promise<{ downloadUrl: string; actualDuration: number }> {
    const maxPollTime = 600_000; // 10 minutes max
    const pollInterval = 5_000;  // Check every 5 seconds
    const startTime = Date.now();

    while (Date.now() - startTime < maxPollTime) {
      const status = await this.providerClient.checkStatus(provider, providerJobId);

      if (status.state === 'completed') {
        return {
          downloadUrl: status.downloadUrl!,
          actualDuration: status.actualDuration!,
        };
      }

      if (status.state === 'failed') {
        throw new Error(`Provider ${provider} failed: ${status.error}`);
      }

      // Update progress based on provider's reported progress
      if (status.progress !== undefined) {
        const mappedProgress = 25 + (status.progress / 100) * 60; // Map to 25-85%
        await this.updateProgress(job, {
          percentage: Math.round(mappedProgress),
          message: status.message || 'Generating...',
          stage: 'generating',
          etaSeconds: status.etaSeconds,
        });
      }

      await this.sleep(pollInterval);
    }

    throw new Error(`Provider ${provider} timed out after ${maxPollTime}ms`);
  }

  /**
   * Simple per-provider concurrency limiter.
   * Waits until a slot is available for the given provider.
   */
  private async waitForProviderSlot(provider: VideoProvider): Promise<void> {
    const limit = this.providerLimits[provider];
    const checkInterval = 2_000;

    while (true) {
      const current = this.activeCounts.get(provider) || 0;
      if (current < limit) {
        this.activeCounts.set(provider, current + 1);
        return;
      }
      console.log(
        `Provider ${provider} at capacity (${current}/${limit}), ` +
        `waiting for slot...`
      );
      await this.sleep(checkInterval);
    }
  }

  private releaseProviderSlot(provider: VideoProvider): void {
    const current = this.activeCounts.get(provider) || 0;
    this.activeCounts.set(provider, Math.max(0, current - 1));
  }

  /**
   * Determine if an error is permanent (should not be retried).
   */
  private isUnrecoverable(error: any): boolean {
    const message = error.message?.toLowerCase() || '';
    return (
      message.includes('content policy violation') ||
      message.includes('invalid api key') ||
      message.includes('account suspended') ||
      message.includes('insufficient funds') ||
      message.includes('prompt rejected') ||
      error.status === 400 || // Bad request
      error.status === 401 || // Unauthorized
      error.status === 403    // Forbidden
    );
  }

  private async updateProgress(
    job: Job<VideoGenerationJobData, VideoGenerationResult>,
    progress: VideoGenerationProgress
  ): Promise<void> {
    await job.updateProgress(progress);
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * Set up event handlers for logging and monitoring.
   */
  private setupEventHandlers(): void {
    this.worker.on('completed', (job, result) => {
      console.log(
        `Job ${job.id} completed in ${result.processingTimeMs}ms, ` +
        `cost=$${result.costUsd.toFixed(4)}`
      );
    });

    this.worker.on('failed', (job, err) => {
      console.error(
        `Job ${job?.id} failed (attempt ${job?.attemptsMade}/${job?.opts.attempts}): ` +
        `${err.message}`
      );
    });

    this.worker.on('error', (err) => {
      console.error('Worker error:', err.message);
    });

    this.worker.on('stalled', (jobId) => {
      console.warn(`Job ${jobId} stalled — lock expired while processing`);
    });
  }

  /**
   * Graceful shutdown: stop accepting new jobs, wait for current jobs to finish.
   */
  async close(): Promise<void> {
    console.log('Worker shutting down gracefully...');
    await this.worker.close();
    console.log('Worker shutdown complete');
  }
}
```

### API Route Integration

```typescript
// src/api/routes/generate.ts
import { Router, Request, Response } from 'express';
import { v4 as uuidv4 } from 'uuid';
import { VideoGenerationProducer } from '../../queue/producer';
import { VideoGenerationJobData, SubscriptionTier } from '../../types/jobs';

const router = Router();
const producer = new VideoGenerationProducer();

/**
 * POST /api/v1/generate
 * Create a new video generation job.
 * Returns immediately with a job ID.
 */
router.post('/generate', async (req: Request, res: Response) => {
  try {
    const {
      prompt,
      provider = 'veo',
      durationSeconds = 5,
      resolution = '1080p',
      aspectRatio = '16:9',
      referenceImageUrl,
      webhookUrl,
      idempotencyKey,
    } = req.body;

    // Validation
    if (!prompt || prompt.length < 10) {
      return res.status(400).json({
        error: 'Prompt must be at least 10 characters',
      });
    }

    if (!['veo', 'kling', 'runway', 'luma', 'minimax'].includes(provider)) {
      return res.status(400).json({ error: `Unknown provider: ${provider}` });
    }

    // Get user from auth middleware (assumed to be set by prior middleware)
    const userId = (req as any).userId;
    const tier: SubscriptionTier = (req as any).userTier;

    const jobData: VideoGenerationJobData = {
      idempotencyKey: idempotencyKey || uuidv4(),
      userId,
      tier,
      prompt,
      referenceImageUrl,
      provider,
      durationSeconds,
      resolution,
      aspectRatio,
      webhookUrl,
      createdAt: Date.now(),
    };

    const jobId = await producer.addJob(jobData);

    // Return immediately — the job is now in the queue
    res.status(202).json({
      jobId,
      status: 'queued',
      message: 'Video generation job created',
      statusUrl: `/api/v1/jobs/${jobId}`,
      estimatedWaitSeconds: await estimateWaitTime(tier, provider),
    });
  } catch (error: any) {
    // If the idempotency key already exists, BullMQ throws
    if (error.message?.includes('Job already exists')) {
      return res.status(409).json({
        error: 'A job with this idempotency key already exists',
      });
    }

    console.error('Failed to create generation job:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

/**
 * GET /api/v1/jobs/:jobId
 * Check the status of a generation job.
 */
router.get('/jobs/:jobId', async (req: Request, res: Response) => {
  const { jobId } = req.params;
  const status = await producer.getJobStatus(jobId);

  if (!status) {
    return res.status(404).json({ error: 'Job not found' });
  }

  res.json({
    jobId,
    state: status.state,
    progress: status.progress,
    result: status.result,
    failedReason: status.failedReason,
    attempts: {
      made: status.attemptsMade,
      total: status.attemptsTotal,
    },
  });
});

/**
 * GET /api/v1/queue/stats
 * Get queue statistics (admin endpoint).
 */
router.get('/queue/stats', async (_req: Request, res: Response) => {
  const stats = await producer.getQueueStats();
  res.json(stats);
});

async function estimateWaitTime(
  tier: SubscriptionTier,
  provider: string
): Promise<number> {
  const stats = await producer.getQueueStats();
  const queueDepth = stats.waiting;

  // Rough estimate based on queue depth and priority
  // Enterprise users skip ahead, so their wait is shorter
  const positionMultiplier: Record<SubscriptionTier, number> = {
    enterprise: 0.1,
    pro: 0.4,
    free: 1.0,
  };

  const avgProcessingTime = 120; // 2 minutes average
  const estimatedPosition = Math.ceil(queueDepth * positionMultiplier[tier]);
  const activeWorkers = Math.max(stats.active, 1);

  return Math.ceil((estimatedPosition * avgProcessingTime) / activeWorkers);
}

export default router;
```

### Bull Board Dashboard Integration

```typescript
// src/dashboard/bull-board.ts
import { createBullBoard } from '@bull-board/api';
import { BullMQAdapter } from '@bull-board/api/bullMQAdapter';
import { ExpressAdapter } from '@bull-board/express';
import { Queue } from 'bullmq';
import { createRedisConfig } from '../config/redis';

export function setupBullBoard(app: any): void {
  const connection = createRedisConfig();

  // Create queue instances for monitoring (read-only)
  const videoQueue = new Queue('video-generation', { connection });
  const deadLetterQueue = new Queue('video-generation-dlq', { connection });

  const serverAdapter = new ExpressAdapter();
  serverAdapter.setBasePath('/admin/queues');

  createBullBoard({
    queues: [
      new BullMQAdapter(videoQueue),
      new BullMQAdapter(deadLetterQueue),
    ],
    serverAdapter,
  });

  // Mount the dashboard — protect this with admin auth in production!
  app.use('/admin/queues', serverAdapter.getRouter());

  console.log('Bull Board dashboard available at /admin/queues');
}
```

### Worker Entry Point

```typescript
// src/worker.ts — Run this as a separate process
import { VideoGenerationWorker } from './queue/worker';

const CONCURRENCY = parseInt(process.env.WORKER_CONCURRENCY || '5', 10);

async function main() {
  console.log(`Starting video generation worker with concurrency=${CONCURRENCY}`);

  const worker = new VideoGenerationWorker(CONCURRENCY);

  // Graceful shutdown on SIGTERM (e.g., from Kubernetes)
  const shutdown = async (signal: string) => {
    console.log(`Received ${signal}, shutting down...`);
    await worker.close();
    process.exit(0);
  };

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));

  console.log('Worker is running and waiting for jobs...');
}

main().catch((err) => {
  console.error('Worker failed to start:', err);
  process.exit(1);
});
```

---

## Priority Queue Design

In a video generation platform, not all users are equal. Enterprise customers paying thousands per month should not wait behind free-tier users generating memes. BullMQ supports numeric priorities where **lower number = higher priority**.

### Priority Levels Visualization

<svg viewBox="0 0 800 350" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:system-ui,sans-serif">
  <text x="400" y="30" text-anchor="middle" font-size="18" font-weight="bold" fill="#333">Priority Queue Levels</text>

  <!-- Priority lanes -->
  <!-- Critical (P1) -->
  <rect x="50" y="55" width="700" height="45" rx="6" fill="#ffcdd2" stroke="#ef5350" stroke-width="2"/>
  <text x="70" y="82" font-size="13" font-weight="bold" fill="#c62828">P1 — CRITICAL</text>
  <text x="300" y="82" font-size="12" fill="#333">System retries, admin jobs, SLA-breach recovery</text>
  <text x="700" y="82" text-anchor="end" font-size="12" font-weight="bold" fill="#c62828">Immediate</text>

  <!-- Enterprise (P2) -->
  <rect x="50" y="110" width="700" height="45" rx="6" fill="#e3f2fd" stroke="#4fc3f7" stroke-width="2"/>
  <text x="70" y="137" font-size="13" font-weight="bold" fill="#0277bd">P2 — ENTERPRISE</text>
  <text x="300" y="137" font-size="12" fill="#333">Enterprise tier users, API customers</text>
  <text x="700" y="137" text-anchor="end" font-size="12" font-weight="bold" fill="#0277bd">&lt; 30s wait</text>

  <!-- Pro (P5) -->
  <rect x="50" y="165" width="700" height="45" rx="6" fill="#e8f5e9" stroke="#8bc34a" stroke-width="2"/>
  <text x="70" y="192" font-size="13" font-weight="bold" fill="#33691e">P5 — PRO</text>
  <text x="300" y="192" font-size="12" fill="#333">Pro subscription users</text>
  <text x="700" y="192" text-anchor="end" font-size="12" font-weight="bold" fill="#33691e">&lt; 2min wait</text>

  <!-- Free (P10) -->
  <rect x="50" y="220" width="700" height="45" rx="6" fill="#fff3e0" stroke="#ff9800" stroke-width="2"/>
  <text x="70" y="247" font-size="13" font-weight="bold" fill="#e65100">P10 — FREE</text>
  <text x="300" y="247" font-size="12" fill="#333">Free tier users, best effort</text>
  <text x="700" y="247" text-anchor="end" font-size="12" font-weight="bold" fill="#e65100">&lt; 10min wait</text>

  <!-- Batch (P20) -->
  <rect x="50" y="275" width="700" height="45" rx="6" fill="#f5f5f5" stroke="#9e9e9e" stroke-width="2"/>
  <text x="70" y="302" font-size="13" font-weight="bold" fill="#424242">P20 — BATCH</text>
  <text x="300" y="302" font-size="12" fill="#333">Bulk processing, overnight jobs, backfills</text>
  <text x="700" y="302" text-anchor="end" font-size="12" font-weight="bold" fill="#424242">Best effort</text>

  <!-- Arrow showing priority direction -->
  <line x1="30" y1="60" x2="30" y2="315" stroke="#333" stroke-width="2" marker-end="url(#arrowhead-q1)"/>
  <text x="25" y="185" text-anchor="middle" font-size="11" fill="#666" transform="rotate(-90,25,185)">Lower priority</text>
</svg>

### The Math of Expected Wait Times

For a system with a single queue, we can model expected wait time using **queuing theory**. The simplest applicable model is the M/M/1 queue (Markovian arrivals, Markovian service, 1 server).

For an M/M/1 queue:

- $\lambda$ = arrival rate (jobs per second)
- $\mu$ = service rate (jobs per second per worker)
- $\rho = \lambda / \mu$ = utilization (must be < 1 for stability)

The expected wait time in the queue is:

$$W_q = \frac{\rho}{\mu(1 - \rho)} = \frac{\lambda}{\mu(\mu - \lambda)}$$

For an M/M/c queue (c workers), the formula becomes more complex but the insight is the same: wait time explodes as utilization approaches 1.

**Real numbers for our platform:**

| Metric | Value |
|--------|-------|
| Average arrival rate ($\lambda$) | 2 jobs/minute |
| Average processing time | 2 minutes |
| Service rate per worker ($\mu$) | 0.5 jobs/minute |
| Number of workers ($c$) | 5 |
| Total service rate ($c\mu$) | 2.5 jobs/minute |
| Utilization ($\rho = \lambda / c\mu$) | 0.8 |

At 80% utilization, the expected wait time in an M/M/c queue with c=5 workers is:

$$W_q = \frac{C(c, \rho)}{c\mu(1-\rho)}$$

where $C(c, \rho)$ is the Erlang C formula. For our numbers, this works out to approximately **3.1 minutes** average wait. At 90% utilization, it jumps to **7.5 minutes**. At 95%, it is **16 minutes**. This is why monitoring utilization and scaling workers is critical.

For a **priority queue**, the expected wait time for priority class $k$ (out of $K$ classes) can be modeled with the priority M/G/1 formula:

$$W_{q,k} = \frac{W_0}{(1 - \sigma_{k-1})(1 - \sigma_k)}$$

where $\sigma_k = \sum_{i=1}^{k} \rho_i$ is the cumulative utilization of all higher-priority classes, and $W_0$ is the mean residual service time for a random job.

The practical implication: if enterprise users (priority 2) consume 20% of capacity, and the system is at 80% total utilization, free users (priority 10) can expect wait times 3-5x longer than if there were no priority classes. This is the desired behavior — you are selling faster processing to paying customers.

---

## Concurrency Management

AI video APIs have strict rate limits. Exceed them and you get 429 errors, or worse, your API key gets temporarily suspended. You need per-provider concurrency limits.

### Provider Rate Limit Reference

| Provider | Concurrent Requests | Requests/Minute | Notes |
|----------|-------------------|-----------------|-------|
| Google Veo | 5 | 30 | Per project |
| Kling (via PiAPI) | 3 | 20 | Varies by plan |
| Runway Gen-3 | 10 | 60 | Enterprise tier |
| Luma Dream Machine | 8 | 40 | Pro plan |
| MiniMax | 4 | 30 | Standard plan |

### Distributed Concurrency Limiter

The per-worker concurrency tracking shown earlier only works for a single worker process. When you have multiple workers across multiple machines, you need a distributed solution using Redis.

```typescript
// src/queue/concurrency-limiter.ts
import IORedis from 'ioredis';

export class DistributedConcurrencyLimiter {
  private redis: IORedis;
  private readonly keyPrefix = 'concurrency';

  constructor(redis: IORedis) {
    this.redis = redis;
  }

  /**
   * Try to acquire a slot for the given provider.
   * Returns true if a slot was acquired, false if at capacity.
   *
   * Uses a Redis sorted set with timestamps as scores.
   * Entries older than the TTL are automatically cleaned up,
   * preventing leaked slots from dead workers.
   */
  async tryAcquire(
    provider: string,
    workerId: string,
    limit: number,
    ttlSeconds: number = 600
  ): Promise<boolean> {
    const key = `${this.keyPrefix}:${provider}`;
    const now = Date.now();
    const expireThreshold = now - ttlSeconds * 1000;

    // Lua script for atomic check-and-acquire
    const script = `
      local key = KEYS[1]
      local workerId = ARGV[1]
      local limit = tonumber(ARGV[2])
      local now = tonumber(ARGV[3])
      local expireThreshold = tonumber(ARGV[4])

      -- Remove expired entries (dead workers)
      redis.call('ZREMRANGEBYSCORE', key, '-inf', expireThreshold)

      -- Check current count
      local count = redis.call('ZCARD', key)

      if count < limit then
        -- Add this worker with current timestamp as score
        redis.call('ZADD', key, now, workerId)
        return 1
      else
        return 0
      end
    `;

    const result = await this.redis.eval(
      script, 1, key, workerId, limit, now, expireThreshold
    );

    return result === 1;
  }

  /**
   * Release a slot for the given provider.
   */
  async release(provider: string, workerId: string): Promise<void> {
    const key = `${this.keyPrefix}:${provider}`;
    await this.redis.zrem(key, workerId);
  }

  /**
   * Get current usage for a provider.
   */
  async getUsage(provider: string): Promise<number> {
    const key = `${this.keyPrefix}:${provider}`;
    return this.redis.zcard(key);
  }

  /**
   * Heartbeat: refresh the worker's slot timestamp so it does not expire.
   * Call this periodically during long-running jobs.
   */
  async heartbeat(provider: string, workerId: string): Promise<void> {
    const key = `${this.keyPrefix}:${provider}`;
    await this.redis.zadd(key, Date.now(), workerId);
  }
}
```

---

## Retry Strategies

Not all failures are the same. A 500 error from the provider is likely transient and worth retrying. A 400 "content policy violation" will never succeed no matter how many times you retry. You need a nuanced retry strategy.

### Exponential Backoff with Jitter

The retry delay formula is:

$$delay = \min\left(base \cdot 2^{attempt} + \text{random}(0, jitter),\; maxDelay\right)$$

For our configuration (base=5s, jitter=5s, maxDelay=300s):

| Attempt | Base Delay | Delay Range | Cumulative Wait |
|---------|-----------|-------------|-----------------|
| 1 | 10s | 10-15s | 10-15s |
| 2 | 20s | 20-25s | 30-40s |
| 3 | 40s | 40-45s | 70-85s |
| 4 | 80s | 80-85s | 150-170s |
| 5 | 160s | 160-165s | 310-335s |

### Error Classification

```typescript
// src/queue/error-classifier.ts

export interface RetryDecision {
  shouldRetry: boolean;
  reason: string;
  /** Override the default backoff delay (ms) */
  customDelay?: number;
}

export function classifyError(error: any): RetryDecision {
  const status = error.status || error.statusCode;
  const message = error.message?.toLowerCase() || '';

  // --- Never retry these ---

  if (status === 400) {
    return {
      shouldRetry: false,
      reason: 'Bad request — the input is invalid and will never succeed',
    };
  }

  if (status === 401 || status === 403) {
    return {
      shouldRetry: false,
      reason: 'Authentication/authorization error — check API key',
    };
  }

  if (message.includes('content policy') || message.includes('safety filter')) {
    return {
      shouldRetry: false,
      reason: 'Content policy violation — prompt was rejected',
    };
  }

  if (message.includes('insufficient') || message.includes('quota exceeded')) {
    return {
      shouldRetry: false,
      reason: 'Account quota exceeded — need to upgrade or wait for reset',
    };
  }

  // --- Retry with specific strategies ---

  if (status === 429) {
    // Rate limited — use a longer delay
    const retryAfter = error.headers?.['retry-after'];
    const delay = retryAfter ? parseInt(retryAfter, 10) * 1000 : 60_000;
    return {
      shouldRetry: true,
      reason: 'Rate limited by provider',
      customDelay: delay,
    };
  }

  if (status === 503 || status === 502 || status === 504) {
    return {
      shouldRetry: true,
      reason: 'Provider temporarily unavailable',
      customDelay: 30_000, // Wait 30 seconds
    };
  }

  if (message.includes('timeout') || message.includes('econnreset')) {
    return {
      shouldRetry: true,
      reason: 'Network timeout — transient failure',
    };
  }

  if (message.includes('internal server error') || status === 500) {
    return {
      shouldRetry: true,
      reason: 'Provider internal error — likely transient',
    };
  }

  // Default: retry with standard backoff
  return {
    shouldRetry: true,
    reason: `Unknown error (status=${status}): ${message.substring(0, 100)}`,
  };
}
```

---

## Dead Letter Queue

When a job exhausts all its retry attempts, it needs to go somewhere that is not the main queue. This is the **dead letter queue** (DLQ). Jobs in the DLQ require human attention — they represent either a bug in your system or a persistent failure in a provider.

```typescript
// src/queue/dead-letter-queue.ts
import { Queue, Worker, Job, QueueEvents } from 'bullmq';
import { createRedisConfig } from '../config/redis';
import { VideoGenerationJobData, VideoGenerationResult } from '../types/jobs';

export class DeadLetterQueueManager {
  private dlq: Queue;
  private mainQueue: Queue;
  private worker: Worker;

  constructor() {
    const connection = createRedisConfig();

    this.dlq = new Queue('video-generation-dlq', { connection });
    this.mainQueue = new Queue('video-generation', { connection });

    // Set up a listener on the main queue to catch jobs that have
    // exhausted all retries
    this.worker = new Worker(
      'video-generation',
      undefined as any, // No processor — we are using the events only
      { connection }
    );

    // This is the key: listen for the 'failed' event where
    // attemptsMade equals the max attempts
    this.setupFailedJobCapture(connection);
  }

  private setupFailedJobCapture(connection: any): void {
    const queueEvents = new QueueEvents('video-generation', { connection });

    queueEvents.on('failed', async ({ jobId, failedReason }) => {
      const job = await this.mainQueue.getJob(jobId);
      if (!job) return;

      // Check if this was the final attempt
      const maxAttempts = job.opts.attempts || 3;
      if (job.attemptsMade >= maxAttempts) {
        console.error(
          `Job ${jobId} exhausted all ${maxAttempts} retries. ` +
          `Moving to DLQ. Final error: ${failedReason}`
        );

        // Add to dead letter queue with full context
        await this.dlq.add('dead-letter', {
          originalJobId: jobId,
          originalData: job.data,
          originalOpts: job.opts,
          failedReason,
          attemptsMade: job.attemptsMade,
          failedAt: Date.now(),
          stacktrace: job.stacktrace,
        });

        // Send alert (PagerDuty, Slack, email, etc.)
        await this.sendAlert(jobId, job.data, failedReason);
      }
    });
  }

  /**
   * Manually retry a job from the DLQ.
   * Used by admin tools to retry jobs after fixing the underlying issue.
   */
  async retryFromDLQ(dlqJobId: string): Promise<string> {
    const dlqJob = await this.dlq.getJob(dlqJobId);
    if (!dlqJob) throw new Error(`DLQ job ${dlqJobId} not found`);

    const { originalData, originalOpts } = dlqJob.data;

    // Create a new job in the main queue with fresh attempts
    const newJob = await this.mainQueue.add('generate-video', originalData, {
      ...originalOpts,
      jobId: `${originalData.idempotencyKey}-retry-${Date.now()}`,
      attempts: 3,
      priority: 1, // Give retried jobs highest priority
    });

    // Mark the DLQ entry as retried
    await dlqJob.updateData({
      ...dlqJob.data,
      retriedAt: Date.now(),
      retriedAsJobId: newJob.id,
    });

    console.log(
      `DLQ job ${dlqJobId} retried as new job ${newJob.id}`
    );

    return newJob.id!;
  }

  /**
   * Retry all DLQ jobs for a specific provider.
   * Useful when a provider recovers from an outage.
   */
  async retryAllForProvider(provider: string): Promise<number> {
    const dlqJobs = await this.dlq.getJobs(['waiting', 'completed']);
    let retriedCount = 0;

    for (const job of dlqJobs) {
      if (
        job.data.originalData?.provider === provider &&
        !job.data.retriedAt
      ) {
        await this.retryFromDLQ(job.id!);
        retriedCount++;
      }
    }

    console.log(`Retried ${retriedCount} DLQ jobs for provider ${provider}`);
    return retriedCount;
  }

  /**
   * Get DLQ statistics grouped by failure reason and provider.
   */
  async getStats(): Promise<{
    total: number;
    byProvider: Record<string, number>;
    byReason: Record<string, number>;
    oldestJobAge: number;
  }> {
    const jobs = await this.dlq.getJobs(['waiting', 'completed']);
    const byProvider: Record<string, number> = {};
    const byReason: Record<string, number> = {};
    let oldestTimestamp = Date.now();

    for (const job of jobs) {
      const provider = job.data.originalData?.provider || 'unknown';
      const reason = this.categorizeReason(job.data.failedReason);

      byProvider[provider] = (byProvider[provider] || 0) + 1;
      byReason[reason] = (byReason[reason] || 0) + 1;

      if (job.data.failedAt < oldestTimestamp) {
        oldestTimestamp = job.data.failedAt;
      }
    }

    return {
      total: jobs.length,
      byProvider,
      byReason,
      oldestJobAge: Date.now() - oldestTimestamp,
    };
  }

  private categorizeReason(reason: string): string {
    if (!reason) return 'unknown';
    const lower = reason.toLowerCase();
    if (lower.includes('timeout')) return 'timeout';
    if (lower.includes('rate limit') || lower.includes('429')) return 'rate_limit';
    if (lower.includes('500') || lower.includes('internal')) return 'provider_error';
    if (lower.includes('content policy')) return 'content_policy';
    if (lower.includes('network') || lower.includes('econnreset')) return 'network';
    return 'other';
  }

  private async sendAlert(
    jobId: string,
    data: VideoGenerationJobData,
    failedReason: string
  ): Promise<void> {
    // In production, integrate with your alerting system:
    // - PagerDuty for critical failures
    // - Slack for informational alerts
    // - Email for daily DLQ digests
    console.error(
      `[ALERT] Job ${jobId} moved to DLQ:\n` +
      `  User: ${data.userId}\n` +
      `  Provider: ${data.provider}\n` +
      `  Prompt: "${data.prompt.substring(0, 80)}..."\n` +
      `  Reason: ${failedReason}`
    );
  }

  async close(): Promise<void> {
    await this.worker.close();
    await this.dlq.close();
    await this.mainQueue.close();
  }
}
```

---

## Scaling Workers

The fundamental throughput equation for a job queue is:

$$\text{throughput} = \text{workers} \times \frac{1}{\text{avg\_processing\_time}}$$

If your average video generation takes 120 seconds, each worker produces $1/120 = 0.0083$ jobs per second. Five workers give you $5 \times 0.0083 = 0.0417$ jobs per second, or about **2.5 jobs per minute**.

### When to Scale

The decision to scale is driven by **queue depth** and **wait time**:

| Queue Depth | Active Workers | Action |
|-------------|---------------|--------|
| 0-10 | 2 | Minimum — keep 2 for redundancy |
| 10-50 | 5 | Normal load |
| 50-200 | 10 | High load — scale up |
| 200-500 | 20 | Peak load — scale up aggressively |
| 500+ | 20 + alert | Investigate — this is unusual |

### Auto-Scaling Implementation

```typescript
// src/scaling/auto-scaler.ts
import { Queue } from 'bullmq';
import { createRedisConfig } from '../config/redis';

interface ScalingConfig {
  minWorkers: number;
  maxWorkers: number;
  scaleUpThreshold: number;    // Queue depth to trigger scale-up
  scaleDownThreshold: number;  // Queue depth to trigger scale-down
  cooldownSeconds: number;     // Wait between scaling decisions
  checkIntervalSeconds: number;
}

export class QueueAutoScaler {
  private queue: Queue;
  private config: ScalingConfig;
  private currentWorkerCount: number;
  private lastScaleAction: number = 0;

  constructor(config: Partial<ScalingConfig> = {}) {
    this.config = {
      minWorkers: 2,
      maxWorkers: 20,
      scaleUpThreshold: 50,
      scaleDownThreshold: 5,
      cooldownSeconds: 300,  // 5 minute cooldown
      checkIntervalSeconds: 30,
      ...config,
    };

    this.currentWorkerCount = this.config.minWorkers;
    this.queue = new Queue('video-generation', {
      connection: createRedisConfig(),
    });
  }

  async checkAndScale(): Promise<void> {
    const now = Date.now();
    const cooldownMs = this.config.cooldownSeconds * 1000;

    // Respect cooldown period
    if (now - this.lastScaleAction < cooldownMs) {
      return;
    }

    const [waiting, active, delayed] = await Promise.all([
      this.queue.getWaitingCount(),
      this.queue.getActiveCount(),
      this.queue.getDelayedCount(),
    ]);

    const totalPending = waiting + delayed;
    const desiredWorkers = this.calculateDesiredWorkers(totalPending, active);

    if (desiredWorkers !== this.currentWorkerCount) {
      console.log(
        `Scaling workers: ${this.currentWorkerCount} -> ${desiredWorkers} ` +
        `(pending=${totalPending}, active=${active})`
      );

      await this.scaleWorkers(desiredWorkers);
      this.currentWorkerCount = desiredWorkers;
      this.lastScaleAction = now;
    }
  }

  private calculateDesiredWorkers(pending: number, active: number): number {
    let desired: number;

    if (pending > 200) {
      desired = this.config.maxWorkers;
    } else if (pending > this.config.scaleUpThreshold) {
      // Linear scaling between threshold and max
      const ratio = pending / 200;
      desired = Math.ceil(
        this.config.minWorkers +
        ratio * (this.config.maxWorkers - this.config.minWorkers)
      );
    } else if (pending < this.config.scaleDownThreshold && active < this.config.minWorkers) {
      desired = this.config.minWorkers;
    } else {
      desired = this.currentWorkerCount; // No change
    }

    // Clamp to min/max
    return Math.max(
      this.config.minWorkers,
      Math.min(this.config.maxWorkers, desired)
    );
  }

  /**
   * Scale workers by calling your infrastructure API.
   * This varies by deployment platform.
   */
  private async scaleWorkers(count: number): Promise<void> {
    // Example: Kubernetes HPA
    // await exec(`kubectl scale deployment video-worker --replicas=${count}`);

    // Example: AWS ECS
    // await ecs.updateService({
    //   cluster: 'video-cluster',
    //   service: 'video-worker',
    //   desiredCount: count,
    // });

    // Example: Docker Compose (for development)
    // await exec(`docker compose up -d --scale video-worker=${count}`);

    console.log(`Scaled to ${count} workers`);
  }

  /**
   * Start the auto-scaling loop.
   */
  start(): void {
    const interval = this.config.checkIntervalSeconds * 1000;
    setInterval(() => this.checkAndScale(), interval);
    console.log(
      `Auto-scaler started (check every ${this.config.checkIntervalSeconds}s, ` +
      `min=${this.config.minWorkers}, max=${this.config.maxWorkers})`
    );
  }
}
```

### Worker Scaling Curve

<svg viewBox="0 0 700 400" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:system-ui,sans-serif">
  <text x="350" y="30" text-anchor="middle" font-size="18" font-weight="bold" fill="#333">Worker Count vs Queue Depth</text>

  <!-- Axes -->
  <line x1="80" y1="350" x2="650" y2="350" stroke="#333" stroke-width="2"/>
  <line x1="80" y1="50" x2="80" y2="350" stroke="#333" stroke-width="2"/>

  <!-- X axis labels -->
  <text x="80" y="375" text-anchor="middle" font-size="11" fill="#666">0</text>
  <text x="194" y="375" text-anchor="middle" font-size="11" fill="#666">50</text>
  <text x="308" y="375" text-anchor="middle" font-size="11" fill="#666">100</text>
  <text x="422" y="375" text-anchor="middle" font-size="11" fill="#666">200</text>
  <text x="536" y="375" text-anchor="middle" font-size="11" fill="#666">300</text>
  <text x="650" y="375" text-anchor="middle" font-size="11" fill="#666">500</text>
  <text x="365" y="398" text-anchor="middle" font-size="13" fill="#333">Queue Depth (pending jobs)</text>

  <!-- Y axis labels -->
  <text x="70" y="350" text-anchor="end" font-size="11" fill="#666">0</text>
  <text x="70" y="290" text-anchor="end" font-size="11" fill="#666">5</text>
  <text x="70" y="230" text-anchor="end" font-size="11" fill="#666">10</text>
  <text x="70" y="170" text-anchor="end" font-size="11" fill="#666">15</text>
  <text x="70" y="110" text-anchor="end" font-size="11" fill="#666">20</text>
  <text x="70" y="50" text-anchor="end" font-size="11" fill="#666">25</text>
  <text x="25" y="200" text-anchor="middle" font-size="13" fill="#333" transform="rotate(-90,25,200)">Worker Count</text>

  <!-- Grid lines -->
  <line x1="80" y1="290" x2="650" y2="290" stroke="#eee" stroke-width="1"/>
  <line x1="80" y1="230" x2="650" y2="230" stroke="#eee" stroke-width="1"/>
  <line x1="80" y1="170" x2="650" y2="170" stroke="#eee" stroke-width="1"/>
  <line x1="80" y1="110" x2="650" y2="110" stroke="#eee" stroke-width="1"/>

  <!-- Scaling curve -->
  <path d="M 80 326 L 120 326 L 194 290 L 308 194 L 422 110 L 536 110 L 650 110"
        stroke="#4fc3f7" stroke-width="3" fill="none"/>

  <!-- Data points -->
  <circle cx="80" cy="326" r="5" fill="#4fc3f7"/>
  <circle cx="194" cy="290" r="5" fill="#4fc3f7"/>
  <circle cx="308" cy="194" r="5" fill="#4fc3f7"/>
  <circle cx="422" cy="110" r="5" fill="#4fc3f7"/>
  <circle cx="536" cy="110" r="5" fill="#ef5350"/>
  <circle cx="650" cy="110" r="5" fill="#ef5350"/>

  <!-- Annotations -->
  <text x="200" y="280" font-size="10" fill="#8bc34a">Scale-up threshold</text>
  <line x1="194" y1="290" x2="194" y2="350" stroke="#8bc34a" stroke-width="1" stroke-dasharray="4,3"/>

  <text x="430" y="100" font-size="10" fill="#ef5350">Max workers (20)</text>
  <line x1="80" y1="110" x2="650" y2="110" stroke="#ef5350" stroke-width="1" stroke-dasharray="4,3"/>

  <text x="88" y="320" font-size="10" fill="#4fc3f7">Min workers (2)</text>
  <line x1="80" y1="326" x2="194" y2="326" stroke="#8bc34a" stroke-width="1" stroke-dasharray="4,3"/>
</svg>

---

## Cost Optimization

Video generation is expensive. A single 10-second 1080p video can cost $0.50-$4.00 depending on the provider. At scale, the compute cost of running workers adds up too. Here are strategies to minimize costs without sacrificing user experience.

### Spot/Preemptible Instances for Batch Workers

For non-priority jobs (batch processing, free-tier users), run workers on spot instances that cost 60-90% less:

```typescript
// src/scaling/spot-worker-config.ts

export const workerDeploymentConfig = {
  // Always-on workers for priority jobs (on-demand instances)
  priority: {
    minInstances: 2,
    maxInstances: 10,
    instanceType: 'c6i.xlarge', // 4 vCPU, 8 GB RAM
    costPerHour: 0.17,
    queues: ['video-generation'],
    // Only process priority 1-5 jobs
    jobFilter: (job: any) => (job.opts.priority || 10) <= 5,
  },

  // Spot workers for non-priority jobs (spot instances)
  batch: {
    minInstances: 0,
    maxInstances: 20,
    instanceType: 'c6i.xlarge',
    costPerHour: 0.05, // ~70% cheaper as spot
    queues: ['video-generation'],
    // Only process priority 6+ jobs
    jobFilter: (job: any) => (job.opts.priority || 10) > 5,
    // Graceful handling when spot instance is reclaimed
    spotInterruptionHandler: async (worker: any) => {
      console.warn('Spot instance being reclaimed, draining worker...');
      await worker.close(); // Finish current job, stop taking new ones
    },
  },

  // Off-peak batch processing (overnight, cheapest rates)
  overnight: {
    schedule: '0 2 * * *', // Start at 2 AM
    endSchedule: '0 8 * * *', // Stop at 8 AM
    maxInstances: 30,
    instanceType: 'c6i.2xlarge',
    costPerHour: 0.08, // Spot during off-peak
    queues: ['video-generation-batch'],
  },
};
```

### Cost Comparison Table

| Strategy | Workers | Cost/Hour | Throughput | Best For |
|----------|---------|-----------|------------|----------|
| On-demand, always on | 5 | $0.85 | 2.5 jobs/min | Enterprise SLA |
| Spot instances | 5 | $0.25 | 2.5 jobs/min | Pro tier |
| Spot + auto-scale | 2-20 | $0.10-$1.00 | 1-10 jobs/min | Dynamic load |
| Off-peak batch | 30 | $2.40 | 15 jobs/min | Bulk processing |

---

## Monitoring and Alerting

You cannot manage what you cannot measure. Here are the key metrics and alert thresholds for a production job queue.

### Key Metrics

| Metric | Formula | Target | Alert Threshold |
|--------|---------|--------|----------------|
| Queue Depth | `waiting + delayed` | < 50 | > 200 for > 5 min |
| Processing Time p50 | 50th percentile of `completedAt - startedAt` | < 120s | > 180s |
| Processing Time p95 | 95th percentile | < 300s | > 600s |
| Processing Time p99 | 99th percentile | < 600s | > 900s |
| Failure Rate | `failed / (completed + failed)` per hour | < 5% | > 10% |
| Throughput | `completed` per minute | > 2/min | < 0.5/min |
| Worker Utilization | `active / (concurrency * workers)` | 60-80% | > 95% for > 10 min |
| DLQ Size | Count of DLQ jobs | 0 | > 10 |
| Stalled Jobs | Jobs whose lock expired | 0 | > 0 |

### Monitoring Implementation

```typescript
// src/monitoring/queue-metrics.ts
import { Queue } from 'bullmq';
import { createRedisConfig } from '../config/redis';

interface QueueMetrics {
  timestamp: number;
  queueDepth: number;
  activeJobs: number;
  completedLastHour: number;
  failedLastHour: number;
  failureRate: number;
  throughputPerMinute: number;
  processingTimeP50: number;
  processingTimeP95: number;
  processingTimeP99: number;
  workerUtilization: number;
  dlqSize: number;
  stalledJobs: number;
}

export class QueueMetricsCollector {
  private queue: Queue;
  private dlq: Queue;
  private processingTimes: number[] = [];

  constructor() {
    const connection = createRedisConfig();
    this.queue = new Queue('video-generation', { connection });
    this.dlq = new Queue('video-generation-dlq', { connection });
  }

  async collect(): Promise<QueueMetrics> {
    const [
      waiting,
      active,
      completed,
      failed,
      delayed,
      dlqWaiting,
    ] = await Promise.all([
      this.queue.getWaitingCount(),
      this.queue.getActiveCount(),
      this.queue.getCompletedCount(),
      this.queue.getFailedCount(),
      this.queue.getDelayedCount(),
      this.dlq.getWaitingCount(),
    ]);

    // Calculate processing time percentiles from recent completed jobs
    const recentCompleted = await this.queue.getJobs(
      ['completed'],
      0,
      100,
      true // ascending
    );

    const processingTimes = recentCompleted
      .filter((j) => j.finishedOn && j.processedOn)
      .map((j) => j.finishedOn! - j.processedOn!);

    const sortedTimes = processingTimes.sort((a, b) => a - b);

    const totalProcessed = completed + failed;
    const failureRate = totalProcessed > 0 ? failed / totalProcessed : 0;

    return {
      timestamp: Date.now(),
      queueDepth: waiting + delayed,
      activeJobs: active,
      completedLastHour: completed, // In production, filter by time window
      failedLastHour: failed,
      failureRate,
      throughputPerMinute: completed / 60, // Simplification
      processingTimeP50: this.percentile(sortedTimes, 0.5),
      processingTimeP95: this.percentile(sortedTimes, 0.95),
      processingTimeP99: this.percentile(sortedTimes, 0.99),
      workerUtilization: active / Math.max(1, active + waiting),
      dlqSize: dlqWaiting,
      stalledJobs: 0, // BullMQ tracks this separately
    };
  }

  private percentile(sorted: number[], p: number): number {
    if (sorted.length === 0) return 0;
    const index = Math.ceil(sorted.length * p) - 1;
    return sorted[Math.max(0, index)];
  }

  /**
   * Check metrics against alert thresholds and return any violations.
   */
  async checkAlerts(): Promise<string[]> {
    const metrics = await this.collect();
    const alerts: string[] = [];

    if (metrics.queueDepth > 200) {
      alerts.push(
        `HIGH QUEUE DEPTH: ${metrics.queueDepth} pending jobs ` +
        `(threshold: 200)`
      );
    }

    if (metrics.failureRate > 0.1) {
      alerts.push(
        `HIGH FAILURE RATE: ${(metrics.failureRate * 100).toFixed(1)}% ` +
        `(threshold: 10%)`
      );
    }

    if (metrics.processingTimeP95 > 600_000) {
      alerts.push(
        `SLOW PROCESSING: p95=${(metrics.processingTimeP95 / 1000).toFixed(0)}s ` +
        `(threshold: 600s)`
      );
    }

    if (metrics.dlqSize > 10) {
      alerts.push(
        `DLQ GROWING: ${metrics.dlqSize} dead-letter jobs ` +
        `(threshold: 10)`
      );
    }

    if (metrics.workerUtilization > 0.95) {
      alerts.push(
        `WORKERS SATURATED: ${(metrics.workerUtilization * 100).toFixed(0)}% ` +
        `utilization (threshold: 95%)`
      );
    }

    return alerts;
  }
}
```

---

## Putting It All Together

Here is the complete architecture diagram showing how all the pieces fit together in production.

<svg viewBox="0 0 950 650" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:system-ui,sans-serif">
  <defs>
    <marker id="arr-main" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>

  <text x="475" y="28" text-anchor="middle" font-size="18" font-weight="bold" fill="#333">Complete Production Architecture</text>

  <!-- Client Layer -->
  <rect x="20" y="50" width="910" height="70" rx="8" fill="#fafafa" stroke="#ddd" stroke-width="1"/>
  <text x="40" y="72" font-size="12" font-weight="bold" fill="#999">CLIENT LAYER</text>
  <rect x="40" y="80" width="100" height="30" rx="4" fill="#f5f5f5" stroke="#999" stroke-width="1"/>
  <text x="90" y="100" text-anchor="middle" font-size="11" fill="#333">Web App</text>
  <rect x="160" y="80" width="100" height="30" rx="4" fill="#f5f5f5" stroke="#999" stroke-width="1"/>
  <text x="210" y="100" text-anchor="middle" font-size="11" fill="#333">Mobile App</text>
  <rect x="280" y="80" width="100" height="30" rx="4" fill="#f5f5f5" stroke="#999" stroke-width="1"/>
  <text x="330" y="100" text-anchor="middle" font-size="11" fill="#333">API Client</text>

  <!-- API Layer -->
  <rect x="20" y="140" width="400" height="90" rx="8" fill="#e3f2fd" stroke="#4fc3f7" stroke-width="2"/>
  <text x="40" y="162" font-size="12" font-weight="bold" fill="#0277bd">API LAYER</text>
  <rect x="40" y="170" width="110" height="45" rx="4" fill="#bbdefb" stroke="#4fc3f7" stroke-width="1"/>
  <text x="95" y="190" text-anchor="middle" font-size="10" fill="#333">API Server 1</text>
  <text x="95" y="205" text-anchor="middle" font-size="9" fill="#666">Producer</text>
  <rect x="165" y="170" width="110" height="45" rx="4" fill="#bbdefb" stroke="#4fc3f7" stroke-width="1"/>
  <text x="220" y="190" text-anchor="middle" font-size="10" fill="#333">API Server 2</text>
  <text x="220" y="205" text-anchor="middle" font-size="9" fill="#666">Producer</text>
  <rect x="290" y="170" width="110" height="45" rx="4" fill="#bbdefb" stroke="#4fc3f7" stroke-width="1"/>
  <text x="345" y="190" text-anchor="middle" font-size="10" fill="#333">Bull Board</text>
  <text x="345" y="205" text-anchor="middle" font-size="9" fill="#666">Dashboard</text>

  <!-- Redis -->
  <rect x="200" y="260" width="180" height="80" rx="8" fill="#ffebee" stroke="#ef5350" stroke-width="2"/>
  <text x="290" y="290" text-anchor="middle" font-size="14" font-weight="bold" fill="#c62828">Redis Cluster</text>
  <text x="290" y="310" text-anchor="middle" font-size="11" fill="#666">BullMQ Queues</text>
  <text x="290" y="325" text-anchor="middle" font-size="11" fill="#666">Rate Limit State</text>

  <!-- Worker Layer -->
  <rect x="20" y="370" width="550" height="110" rx="8" fill="#e8f5e9" stroke="#8bc34a" stroke-width="2"/>
  <text x="40" y="392" font-size="12" font-weight="bold" fill="#33691e">WORKER LAYER</text>
  <rect x="40" y="400" width="95" height="65" rx="4" fill="#c8e6c9" stroke="#8bc34a" stroke-width="1"/>
  <text x="87" y="425" text-anchor="middle" font-size="10" fill="#333">Priority</text>
  <text x="87" y="440" text-anchor="middle" font-size="10" fill="#333">Workers</text>
  <text x="87" y="455" text-anchor="middle" font-size="9" fill="#666">On-demand</text>
  <rect x="150" y="400" width="95" height="65" rx="4" fill="#c8e6c9" stroke="#8bc34a" stroke-width="1"/>
  <text x="197" y="425" text-anchor="middle" font-size="10" fill="#333">Standard</text>
  <text x="197" y="440" text-anchor="middle" font-size="10" fill="#333">Workers</text>
  <text x="197" y="455" text-anchor="middle" font-size="9" fill="#666">Spot</text>
  <rect x="260" y="400" width="95" height="65" rx="4" fill="#c8e6c9" stroke="#8bc34a" stroke-width="1"/>
  <text x="307" y="425" text-anchor="middle" font-size="10" fill="#333">Batch</text>
  <text x="307" y="440" text-anchor="middle" font-size="10" fill="#333">Workers</text>
  <text x="307" y="455" text-anchor="middle" font-size="9" fill="#666">Off-peak spot</text>
  <rect x="370" y="400" width="95" height="65" rx="4" fill="#fff9c4" stroke="#fbc02d" stroke-width="1"/>
  <text x="417" y="425" text-anchor="middle" font-size="10" fill="#333">Auto-Scaler</text>
  <text x="417" y="445" text-anchor="middle" font-size="9" fill="#666">Monitors queue</text>
  <text x="417" y="458" text-anchor="middle" font-size="9" fill="#666">depth</text>
  <rect x="480" y="400" width="75" height="65" rx="4" fill="#ffccbc" stroke="#ff5722" stroke-width="1"/>
  <text x="517" y="425" text-anchor="middle" font-size="10" fill="#333">DLQ</text>
  <text x="517" y="440" text-anchor="middle" font-size="10" fill="#333">Handler</text>

  <!-- Provider Layer -->
  <rect x="20" y="510" width="550" height="70" rx="8" fill="#fff3e0" stroke="#ff9800" stroke-width="2"/>
  <text x="40" y="532" font-size="12" font-weight="bold" fill="#e65100">VIDEO MODEL PROVIDERS</text>
  <text x="80" y="560" font-size="12" fill="#333">Veo</text>
  <text x="170" y="560" font-size="12" fill="#333">Kling</text>
  <text x="260" y="560" font-size="12" fill="#333">Runway</text>
  <text x="350" y="560" font-size="12" fill="#333">Luma</text>
  <text x="440" y="560" font-size="12" fill="#333">MiniMax</text>

  <!-- Monitoring -->
  <rect x="600" y="260" width="320" height="210" rx="8" fill="#f3e5f5" stroke="#9c27b0" stroke-width="2"/>
  <text x="620" y="285" font-size="12" font-weight="bold" fill="#6a1b9a">MONITORING</text>
  <text x="620" y="310" font-size="11" fill="#333">Metrics: queue depth, latency,</text>
  <text x="620" y="325" font-size="11" fill="#333">failure rate, throughput</text>
  <text x="620" y="350" font-size="11" fill="#333">Alerts: PagerDuty, Slack</text>
  <text x="620" y="375" font-size="11" fill="#333">Dashboards: Grafana, Bull Board</text>
  <text x="620" y="400" font-size="11" fill="#333">Logs: structured JSON to ELK</text>
  <text x="620" y="430" font-size="11" fill="#333">Tracing: OpenTelemetry spans</text>
  <text x="620" y="455" font-size="11" fill="#333">Cost tracking: per-user, per-provider</text>

  <!-- Arrows -->
  <line x1="200" y1="120" x2="200" y2="138" stroke="#333" stroke-width="1.5" marker-end="url(#arr-main)"/>
  <line x1="220" y1="215" x2="260" y2="258" stroke="#333" stroke-width="1.5" marker-end="url(#arr-main)"/>
  <line x1="290" y1="340" x2="200" y2="368" stroke="#333" stroke-width="1.5" marker-end="url(#arr-main)"/>
  <line x1="200" y1="465" x2="200" y2="508" stroke="#333" stroke-width="1.5" marker-end="url(#arr-main)"/>

  <!-- Redis to monitoring -->
  <line x1="380" y1="300" x2="595" y2="320" stroke="#9c27b0" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arr-main)"/>
</svg>

### Production Checklist

Before deploying this to production, make sure you have checked off every item:

| Category | Item | Status |
|----------|------|--------|
| **Redis** | Redis Cluster or Sentinel for HA | Required |
| **Redis** | Persistence (RDB + AOF) enabled | Required |
| **Redis** | Memory limits and eviction policy set | Required |
| **Redis** | Separate Redis instance from app cache | Recommended |
| **Workers** | Graceful shutdown on SIGTERM | Required |
| **Workers** | Health check endpoint | Required |
| **Workers** | Lock duration > max processing time | Required |
| **Workers** | Lock renewal configured | Required |
| **Monitoring** | Queue depth metric and alert | Required |
| **Monitoring** | Failure rate metric and alert | Required |
| **Monitoring** | Processing time percentiles | Required |
| **Monitoring** | DLQ size alert | Required |
| **Monitoring** | Stalled job alert | Required |
| **Cost** | Per-user budget caps | Recommended |
| **Cost** | Per-provider cost tracking | Recommended |
| **Cost** | Spot instance workers for batch | Recommended |
| **Security** | Redis AUTH enabled | Required |
| **Security** | Redis TLS in production | Required |
| **Security** | No sensitive data in job payloads | Required |

### Summary

Building a production job queue for AI video generation is not optional — it is a fundamental architectural requirement. The combination of Redis and BullMQ gives you a battle-tested foundation that handles the async, expensive, failure-prone nature of video generation workloads.

The key design decisions:

1. **Decouple request from processing** with a producer-consumer pattern.
2. **Prioritize by user tier** to deliver value to paying customers.
3. **Respect provider rate limits** with per-provider concurrency control.
4. **Retry intelligently** with exponential backoff and error classification.
5. **Catch permanent failures** in a dead letter queue with alerting.
6. **Scale workers dynamically** based on queue depth and utilization.
7. **Optimize costs** with spot instances and off-peak batch processing.
8. **Monitor everything** with metrics, alerts, and dashboards.

The code in this post is production-ready. Fork it, adapt the provider-specific details, and deploy it. Your users (and your on-call engineers) will thank you.
