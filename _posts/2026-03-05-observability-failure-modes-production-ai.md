---
layout: post
title: "Observability and Failure Modes in Production AI Systems"
date: 2026-03-05
category: infra
---

# Observability and Failure Modes in Production AI Systems

*This is Part 4 of 4 in the series on taking vibe-coded AI projects to production. [Part 1](/2026/01/17/redis-bullmq-job-queue-video/) covered job queue architecture, [Part 2](/2026/01/14/database-schema-ai-video-platform/) covered database schema design, and [Part 3](/2026/01/20/websocket-architecture-generation-status/) covered real-time status delivery. This final article covers the thing that keeps production systems alive after launch: observability and failure resilience.*

Your video platform is deployed. It handles load. It has CI/CD. And then, at 2 AM on a Saturday, it stops working. Not with a crash. Not with an error message. It just... gets slow. Requests take 30 seconds instead of 200ms. Some succeed, some time out. The database says it is fine. Redis says it is fine. Your logs say nothing useful because you logged `"error occurred"` without context. You have no idea what is happening, and your users are churning while you SSH into a VPS and run `htop` like a caveman.

This is the observability problem. And it is the difference between a production system and a deployed hobby project.

The gap between "it works on my machine" and "it works reliably at 3 AM when I am asleep" is almost entirely an observability gap. You cannot fix what you cannot see. You cannot debug what you did not instrument. And in an AI video platform, where a single generation request might touch six services and cost $0.10 in API fees, the cost of not seeing is measured in dollars, churned users, and sleepless weekends.

This post is the complete guide to making your system visible, resilient, and debuggable. We will build everything from scratch, define every term, and by the end, you will have a production observability stack that would make an SRE team nod approvingly.

---

## Table of Contents

1. [The Three Pillars of Observability](#1-the-three-pillars-of-observability)
2. [Structured Logging](#2-structured-logging)
3. [Metrics with Prometheus](#3-metrics-with-prometheus)
4. [Dashboards with Grafana](#4-dashboards-with-grafana)
5. [Alerting That Does Not Cry Wolf](#5-alerting-that-does-not-cry-wolf)
6. [The Failure Taxonomy](#6-the-failure-taxonomy)
7. [Distributed Tracing with OpenTelemetry](#7-distributed-tracing-with-opentelemetry)
8. [The Observability Maturity Model](#8-the-observability-maturity-model)
9. [The Production Readiness Checklist](#9-the-production-readiness-checklist)

---

## 1. The Three Pillars of Observability

Before we write a single line of instrumentation code, we need to understand what observability actually means and why it requires three distinct mechanisms working together.

**Observability** is the ability to understand the internal state of a system by examining its external outputs. The term comes from control theory: a system is "observable" if you can determine its internal state from its outputs alone. In software, the "outputs" are the data your system emits about itself --- logs, metrics, and traces.

Here is the critical insight: you need all three. Each one answers a different question, and no single pillar can substitute for the others.

| Pillar | Question It Answers | Example |
|--------|---------------------|---------|
| **Logs** | *What happened?* | "Generation gen_abc123 failed with error: model API returned 503 at 02:14:37 UTC" |
| **Metrics** | *How much? How fast? How often?* | "P95 generation latency is 4.2 seconds. Error rate is 3.7%. Queue depth is 47." |
| **Traces** | *Where in the request path?* | "The request spent 12ms in the API, 340ms waiting in the queue, 2800ms in the model API, 90ms in storage upload" |

Let me make this concrete with a scenario. A user reports that video generation is slow. Here is what each pillar tells you:

- **Metrics** show that P95 latency spiked from 3s to 12s starting at 14:00 UTC. So you know *when* it started and *how bad* it is.
- **Traces** show that the latency increase is entirely in the "model API call" span --- everything else (queue wait, upload, database writes) is normal. So you know *where* the bottleneck is.
- **Logs** for those slow traces show that the model API is returning rate-limit headers and the system is retrying with backoff. So you know *why* it is slow.

No single pillar gives you the full picture. Metrics without traces tell you something is slow but not where. Traces without logs tell you where but not why. Logs without metrics give you individual data points but no aggregate view.

<svg viewBox="0 0 900 420" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:system-ui,sans-serif">
  <defs>
    <marker id="arr-obs" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>

  <text x="450" y="30" text-anchor="middle" font-size="18" font-weight="bold" fill="#333">The Three Pillars of Observability</text>

  <!-- Logs -->
  <rect x="50" y="60" width="230" height="160" rx="10" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="165" y="90" text-anchor="middle" font-size="16" font-weight="bold" fill="#1976d2">Logs</text>
  <text x="165" y="115" text-anchor="middle" font-size="12" fill="#333">WHAT happened</text>
  <line x1="80" y1="125" x2="250" y2="125" stroke="#1976d2" stroke-width="1" opacity="0.3"/>
  <text x="165" y="145" text-anchor="middle" font-size="11" fill="#555">Discrete events with context</text>
  <text x="165" y="163" text-anchor="middle" font-size="11" fill="#555">Error messages, stack traces</text>
  <text x="165" y="181" text-anchor="middle" font-size="11" fill="#555">Request/response details</text>
  <text x="165" y="199" text-anchor="middle" font-size="11" fill="#555">Structured JSON with IDs</text>

  <!-- Metrics -->
  <rect x="335" y="60" width="230" height="160" rx="10" fill="#e8f5e9" stroke="#388e3c" stroke-width="2"/>
  <text x="450" y="90" text-anchor="middle" font-size="16" font-weight="bold" fill="#388e3c">Metrics</text>
  <text x="450" y="115" text-anchor="middle" font-size="12" fill="#333">HOW MUCH / HOW FAST</text>
  <line x1="365" y1="125" x2="535" y2="125" stroke="#388e3c" stroke-width="1" opacity="0.3"/>
  <text x="450" y="145" text-anchor="middle" font-size="11" fill="#555">Aggregated numeric values</text>
  <text x="450" y="163" text-anchor="middle" font-size="11" fill="#555">Counters, gauges, histograms</text>
  <text x="450" y="181" text-anchor="middle" font-size="11" fill="#555">Time-series data</text>
  <text x="450" y="199" text-anchor="middle" font-size="11" fill="#555">Alerting thresholds</text>

  <!-- Traces -->
  <rect x="620" y="60" width="230" height="160" rx="10" fill="#fff3e0" stroke="#f57c00" stroke-width="2"/>
  <text x="735" y="90" text-anchor="middle" font-size="16" font-weight="bold" fill="#f57c00">Traces</text>
  <text x="735" y="115" text-anchor="middle" font-size="12" fill="#333">WHERE in the path</text>
  <line x1="650" y1="125" x2="820" y2="125" stroke="#f57c00" stroke-width="1" opacity="0.3"/>
  <text x="735" y="145" text-anchor="middle" font-size="11" fill="#555">Request flow across services</text>
  <text x="735" y="163" text-anchor="middle" font-size="11" fill="#555">Span timing breakdown</text>
  <text x="735" y="181" text-anchor="middle" font-size="11" fill="#555">Parent-child relationships</text>
  <text x="735" y="199" text-anchor="middle" font-size="11" fill="#555">Waterfall visualization</text>

  <!-- Connecting arrows to center insight -->
  <rect x="275" y="280" width="350" height="60" rx="8" fill="#f5f5f5" stroke="#333" stroke-width="2"/>
  <text x="450" y="305" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Complete Understanding</text>
  <text x="450" y="323" text-anchor="middle" font-size="11" fill="#555">WHEN + HOW BAD + WHERE + WHY</text>

  <line x1="165" y1="220" x2="350" y2="280" stroke="#1976d2" stroke-width="2" marker-end="url(#arr-obs)"/>
  <line x1="450" y1="220" x2="450" y2="275" stroke="#388e3c" stroke-width="2" marker-end="url(#arr-obs)"/>
  <line x1="735" y1="220" x2="550" y2="280" stroke="#f57c00" stroke-width="2" marker-end="url(#arr-obs)"/>

  <!-- Bottom note -->
  <text x="450" y="380" text-anchor="middle" font-size="12" fill="#777" font-style="italic">Each pillar alone gives partial visibility.</text>
  <text x="450" y="400" text-anchor="middle" font-size="12" fill="#777" font-style="italic">Together, they give you full observability.</text>
</svg>

---

## 2. Structured Logging

Let us start with the most basic pillar --- and the one most vibe-coded projects get catastrophically wrong.

### The Problem with console.log

Here is what logging looks like in most hobby projects:

```typescript
// This is what Level 0 observability looks like
app.post('/api/generate', async (req, res) => {
  try {
    console.log('generation started');
    const result = await generateVideo(req.body);
    console.log('generation done');
    res.json(result);
  } catch (err) {
    console.log('error occurred');  // Which error? For whom? When? Why?
    res.status(500).json({ error: 'failed' });
  }
});
```

At 3 AM when your platform is on fire, this log tells you absolutely nothing. You know *an* error occurred. You do not know which user was affected. You do not know what model was being used. You do not know what the error actually was. You do not know how long the request had been running. You cannot correlate this log line with any other log line from the same request.

**Structured logging** means emitting log entries as machine-parseable data (typically JSON) with consistent fields that let you search, filter, and aggregate across millions of log lines.

### Key Concepts

Before we implement anything, let me define the terms:

- **Log level**: A severity classification. `debug` for development noise, `info` for normal operations, `warn` for concerning-but-not-broken situations, `error` for things that failed and need attention, `fatal` for things that crashed the process.
- **Structured log**: A log entry emitted as a JSON object with named fields, as opposed to a freeform string. `{"level":"error","userId":"u_123","msg":"generation failed"}` instead of `"Error: generation failed for user"`.
- **Correlation ID**: A unique identifier (usually a UUID) assigned to each incoming request and propagated through every service that handles it. This lets you search for a single correlation ID and see every log line from every service that participated in handling that request.
- **Request context**: The set of metadata (user ID, request ID, route, method) attached to every log line within the scope of a single request.

### Setting Up Pino

**Pino** is the standard structured logger for Node.js. It is significantly faster than Winston (which blocks the event loop during serialization) because it uses a worker thread for log processing. In a video generation platform handling hundreds of concurrent requests, this performance difference matters.

```typescript
// src/logger.ts
import pino from 'pino';

// Create the base logger with default configuration
export const logger = pino({
  level: process.env.LOG_LEVEL || 'info',

  // Format timestamps as ISO 8601 strings for human readability
  // and machine parseability
  timestamp: pino.stdTimeFunctions.isoTime,

  // Base fields included in every log line from this process
  base: {
    service: 'video-api',
    env: process.env.NODE_ENV || 'development',
    version: process.env.APP_VERSION || 'unknown',
  },

  // In development, pretty-print for human consumption
  // In production, emit raw JSON for log aggregation tools
  transport: process.env.NODE_ENV === 'development'
    ? { target: 'pino-pretty', options: { colorize: true } }
    : undefined,

  // Redact sensitive fields so they never appear in logs
  redact: {
    paths: ['req.headers.authorization', 'req.headers.cookie', 'body.apiKey'],
    censor: '[REDACTED]',
  },
});

// Type for the request-scoped logger
export type RequestLogger = pino.Logger;
```

### Request-Scoped Logging with Correlation IDs

The real power of structured logging comes from attaching context to every log line within a request. We do this with Express middleware that creates a **child logger** --- a logger instance that inherits the base configuration but adds request-specific fields to every line it emits.

```typescript
// src/middleware/requestLogger.ts
import { randomUUID } from 'crypto';
import { Request, Response, NextFunction } from 'express';
import { logger, RequestLogger } from '../logger';

// Extend Express Request to carry the logger
declare global {
  namespace Express {
    interface Request {
      log: RequestLogger;
      requestId: string;
    }
  }
}

export function requestLoggerMiddleware(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  const requestId = (req.headers['x-request-id'] as string) || randomUUID();
  const startTime = Date.now();

  // Create a child logger with request-scoped context.
  // Every log line emitted with req.log will include these fields
  // automatically, so you never have to remember to add them.
  req.requestId = requestId;
  req.log = logger.child({
    requestId,
    method: req.method,
    path: req.path,
    userId: (req as any).userId || 'anonymous',
    userAgent: req.headers['user-agent'],
    ip: req.ip,
  });

  // Set the correlation ID on the response header so the client
  // can reference it in bug reports
  res.setHeader('x-request-id', requestId);

  req.log.info('request started');

  // When the response finishes, log the outcome
  res.on('finish', () => {
    const duration = Date.now() - startTime;
    const logData = {
      statusCode: res.statusCode,
      duration,
      contentLength: res.getHeader('content-length'),
    };

    if (res.statusCode >= 500) {
      req.log.error(logData, 'request failed with server error');
    } else if (res.statusCode >= 400) {
      req.log.warn(logData, 'request failed with client error');
    } else {
      req.log.info(logData, 'request completed');
    }
  });

  next();
}
```

### Good vs Bad Logging in Practice

Here is the difference this makes. A generation fails. Here is what you see with unstructured logging:

```
error occurred
generation failed
```

Here is what you see with structured logging:

```json
{
  "level": "error",
  "time": "2026-03-05T02:14:37.892Z",
  "service": "video-api",
  "requestId": "req_8f3a2b1c-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
  "userId": "usr_p4k7m2",
  "method": "POST",
  "path": "/api/generate",
  "generationId": "gen_abc123",
  "model": "kling-v3",
  "prompt": "A cat riding a bicycle through Tokyo at sunset",
  "duration": 4823,
  "error": {
    "code": "MODEL_API_503",
    "message": "Kling API returned 503 Service Unavailable",
    "retryCount": 3,
    "lastRetryDelay": 4000
  },
  "msg": "generation failed after exhausting retries"
}
```

With this log entry, you know: who was affected (`usr_p4k7m2`), what they were trying to do (generate with Kling v3), how long it took before failing (4.8 seconds), why it failed (Kling API returned 503), how many retries were attempted (3), and the exact request ID for cross-referencing with traces and other log lines.

### Propagating Correlation IDs Across Services

In our video platform, a single generation request touches multiple services: the API server, the job queue, the worker process, and potentially external APIs. The correlation ID must flow through all of them.

```typescript
// When adding a job to the queue, include the requestId
import { Queue } from 'bullmq';

const generationQueue = new Queue('video-generation', {
  connection: redisConnection,
});

async function enqueueGeneration(
  req: Request,
  params: GenerationParams
): Promise<string> {
  const generationId = `gen_${randomUUID().slice(0, 12)}`;

  await generationQueue.add(
    'generate',
    {
      ...params,
      generationId,
      // Propagate the correlation ID into the job data
      // so the worker can continue the logging chain
      requestId: req.requestId,
      userId: (req as any).userId,
    },
    {
      jobId: generationId,
      priority: params.isPremiumUser ? 1 : 10,
    }
  );

  req.log.info(
    { generationId, model: params.model, priority: params.isPremiumUser ? 1 : 10 },
    'generation job enqueued'
  );

  return generationId;
}
```

```typescript
// In the worker, reconstruct the logger with the same correlation ID
import { Worker, Job } from 'bullmq';
import { logger } from '../logger';

const worker = new Worker(
  'video-generation',
  async (job: Job) => {
    // Create a child logger with the correlation ID from the original request.
    // Now every log line from this worker run is linked to the original
    // HTTP request that triggered it.
    const log = logger.child({
      requestId: job.data.requestId,
      generationId: job.data.generationId,
      userId: job.data.userId,
      model: job.data.model,
      jobId: job.id,
      worker: `worker-${process.pid}`,
    });

    log.info('worker picked up generation job');

    try {
      const startTime = Date.now();
      const result = await callModelAPI(job.data, log);
      const duration = Date.now() - startTime;

      log.info(
        { duration, outputUrl: result.videoUrl, cost: result.cost },
        'generation completed successfully'
      );

      return result;
    } catch (err) {
      log.error(
        { error: { message: (err as Error).message, stack: (err as Error).stack } },
        'generation failed in worker'
      );
      throw err;
    }
  },
  { connection: redisConnection, concurrency: 5 }
);
```

Now, when something goes wrong, you search for a single `requestId` and get the complete story across every service boundary: the API received the request, the queue accepted the job, the worker picked it up, the model API was called, and here is exactly what happened at each stage.

---

## 3. Metrics with Prometheus

Logs tell you what happened to individual requests. Metrics tell you what is happening to your system as a whole, right now, over time. They are the vital signs of your platform.

### What Prometheus Is and How It Works

**Prometheus** is a time-series database and monitoring system. Unlike most monitoring tools that require your application to *push* data to them, Prometheus uses a *pull* model: your application exposes an HTTP endpoint (typically `/metrics`) that returns all current metric values, and Prometheus periodically scrapes that endpoint (usually every 15 seconds).

Why pull instead of push? Three reasons:
1. Your application does not need to know where Prometheus lives or whether it is up. It just serves metrics on demand.
2. Prometheus can detect when a target is down (the scrape fails), which is itself a useful signal.
3. You can spin up a new Prometheus instance and point it at existing targets without changing any application configuration.

### The Four Metric Types

Prometheus defines four fundamental metric types. Each one models a different kind of measurement.

**Counter**: A value that only goes up. It counts the total number of times something has happened since the process started. Examples: total HTTP requests served, total errors, total generations completed. You never read a counter's raw value --- you query its *rate of change* over time (requests per second, errors per minute).

**Gauge**: A value that goes up and down. It represents a current measurement at a point in time. Examples: current queue depth, active database connections, CPU usage, memory usage. Unlike counters, you read gauges directly: "there are 47 jobs in the queue right now."

**Histogram**: A distribution of observed values, sorted into configurable buckets. It answers questions like "what percentage of requests completed in under 200ms?" and "what is the 95th percentile latency?" Each observation is counted into a bucket, and Prometheus can compute quantiles from the bucket counts.

**Summary**: Similar to a histogram but computes quantiles on the client side. Histograms are generally preferred because they can be aggregated across instances; summaries cannot.

### Instrumenting the Video Platform

Here is a complete metrics setup for our AI video generation platform using `prom-client`, the standard Prometheus client for Node.js:

```typescript
// src/metrics.ts
import client, {
  Registry,
  Counter,
  Histogram,
  Gauge,
  collectDefaultMetrics,
} from 'prom-client';

// Create a dedicated registry. The default registry works fine for
// single-service setups, but a dedicated registry gives you explicit
// control over what gets exposed.
export const registry = new Registry();

// Collect Node.js runtime metrics automatically:
// process CPU usage, memory usage, event loop lag, GC stats, etc.
collectDefaultMetrics({ register: registry });

// ---------- HTTP Metrics ----------

// Histogram for HTTP request duration.
// The buckets define the boundaries for latency classification.
// We care about fast (<100ms), normal (<500ms), slow (<2s), and very slow (>2s).
export const httpRequestDuration = new Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'status_code'] as const,
  buckets: [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30],
  registers: [registry],
});

// Counter for total HTTP requests (used for calculating request rate)
export const httpRequestsTotal = new Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status_code'] as const,
  registers: [registry],
});

// ---------- Generation Pipeline Metrics ----------

// Gauge: how many generations are currently in progress
// This tells you the current load on your worker fleet
export const activeGenerations = new Gauge({
  name: 'video_generations_active',
  help: 'Number of video generations currently in progress',
  labelNames: ['model'] as const,
  registers: [registry],
});

// Counter: total generations completed, labeled by model and outcome
// Rate of this counter = generations per second
export const generationsTotal = new Counter({
  name: 'video_generations_total',
  help: 'Total video generations completed',
  labelNames: ['model', 'status'] as const, // status: success, failure, timeout
  registers: [registry],
});

// Histogram: generation duration by model
// This tells you the P50, P95, P99 generation time per model
export const generationDuration = new Histogram({
  name: 'video_generation_duration_seconds',
  help: 'Duration of video generation in seconds',
  labelNames: ['model'] as const,
  buckets: [1, 5, 10, 30, 60, 120, 300, 600],
  registers: [registry],
});

// Counter: total cost incurred by model API calls
export const generationCostTotal = new Counter({
  name: 'video_generation_cost_dollars_total',
  help: 'Total cost of video generation API calls in dollars',
  labelNames: ['model'] as const,
  registers: [registry],
});

// ---------- Queue Metrics ----------

// Gauge: current depth of each queue
export const queueDepth = new Gauge({
  name: 'job_queue_depth',
  help: 'Current number of jobs waiting in the queue',
  labelNames: ['queue_name', 'state'] as const, // state: waiting, active, delayed
  registers: [registry],
});

// ---------- Infrastructure Metrics ----------

// Gauge: database connection pool utilization
export const dbPoolUtilization = new Gauge({
  name: 'db_connection_pool_utilization_ratio',
  help: 'Database connection pool usage as a ratio (0 to 1)',
  registers: [registry],
});

// Gauge: Redis memory usage in bytes
export const redisMemoryUsage = new Gauge({
  name: 'redis_memory_usage_bytes',
  help: 'Redis memory usage in bytes',
  registers: [registry],
});

// Counter: circuit breaker state transitions
export const circuitBreakerTransitions = new Counter({
  name: 'circuit_breaker_transitions_total',
  help: 'Number of circuit breaker state transitions',
  labelNames: ['service', 'from_state', 'to_state'] as const,
  registers: [registry],
});
```

### Express Middleware for HTTP Metrics

Now we wire the HTTP metrics into Express so every request is automatically measured:

```typescript
// src/middleware/metricsMiddleware.ts
import { Request, Response, NextFunction } from 'express';
import { httpRequestDuration, httpRequestsTotal } from '../metrics';

export function metricsMiddleware(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  // Normalize the route path so that /api/generations/gen_abc123
  // becomes /api/generations/:id. Without this, you get a unique
  // metric label for every generation ID, which causes unbounded
  // cardinality and kills Prometheus.
  const route = normalizeRoute(req.route?.path || req.path);
  const startTime = Date.now();

  res.on('finish', () => {
    const duration = (Date.now() - startTime) / 1000; // Convert to seconds
    const labels = {
      method: req.method,
      route,
      status_code: String(res.statusCode),
    };

    httpRequestDuration.observe(labels, duration);
    httpRequestsTotal.inc(labels);
  });

  next();
}

function normalizeRoute(path: string): string {
  // Replace UUIDs and common ID patterns with :id
  return path
    .replace(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/g, ':id')
    .replace(/gen_[a-z0-9]+/g, ':id')
    .replace(/usr_[a-z0-9]+/g, ':id')
    .replace(/\d{10,}/g, ':id');
}
```

### Exposing the Metrics Endpoint

```typescript
// In your Express app setup
import express from 'express';
import { registry } from './metrics';

const app = express();

// The /metrics endpoint that Prometheus will scrape
app.get('/metrics', async (_req, res) => {
  try {
    res.set('Content-Type', registry.contentType);
    res.end(await registry.metrics());
  } catch (err) {
    res.status(500).end('Error collecting metrics');
  }
});
```

### Instrumenting the Generation Worker

The worker process is where the most important metrics are recorded --- generation duration, cost, and success/failure rates:

```typescript
// In the worker processor
import {
  activeGenerations,
  generationsTotal,
  generationDuration,
  generationCostTotal,
} from '../metrics';

async function processGeneration(job: Job): Promise<GenerationResult> {
  const { model } = job.data;

  // Track that a generation is in progress
  activeGenerations.inc({ model });
  const startTime = Date.now();

  try {
    const result = await callModelAPI(job.data);
    const durationSec = (Date.now() - startTime) / 1000;

    // Record successful completion
    generationsTotal.inc({ model, status: 'success' });
    generationDuration.observe({ model }, durationSec);
    generationCostTotal.inc({ model }, result.cost);

    return result;
  } catch (err) {
    const durationSec = (Date.now() - startTime) / 1000;

    // Record failure --- still record duration so we can see if failures
    // are fast (immediate rejection) or slow (timeout after waiting)
    generationsTotal.inc({ model, status: 'failure' });
    generationDuration.observe({ model }, durationSec);

    throw err;
  } finally {
    // Always decrement the active gauge, regardless of success or failure
    activeGenerations.dec({ model });
  }
}
```

### Periodic Infrastructure Metrics Collection

Some metrics cannot be collected in the request path. Queue depth, connection pool utilization, and Redis memory usage need to be sampled periodically:

```typescript
// src/metricsCollector.ts
import { Queue } from 'bullmq';
import { Pool } from 'pg';
import Redis from 'ioredis';
import { queueDepth, dbPoolUtilization, redisMemoryUsage } from './metrics';

export function startMetricsCollector(
  queues: Record<string, Queue>,
  dbPool: Pool,
  redis: Redis
): void {
  // Collect every 15 seconds to match the Prometheus scrape interval.
  // There is no benefit to collecting more frequently.
  setInterval(async () => {
    // Queue depths
    for (const [name, queue] of Object.entries(queues)) {
      const counts = await queue.getJobCounts(
        'waiting', 'active', 'delayed', 'failed'
      );
      queueDepth.set({ queue_name: name, state: 'waiting' }, counts.waiting);
      queueDepth.set({ queue_name: name, state: 'active' }, counts.active);
      queueDepth.set({ queue_name: name, state: 'delayed' }, counts.delayed);
    }

    // Database connection pool
    // pool.totalCount = total connections, pool.idleCount = idle connections
    const poolTotal = dbPool.totalCount;
    const poolIdle = dbPool.idleCount;
    const poolMax = 20; // Your configured max pool size
    const utilization = poolTotal > 0 ? (poolTotal - poolIdle) / poolMax : 0;
    dbPoolUtilization.set(utilization);

    // Redis memory
    const info = await redis.info('memory');
    const memMatch = info.match(/used_memory:(\d+)/);
    if (memMatch) {
      redisMemoryUsage.set(parseInt(memMatch[1], 10));
    }
  }, 15_000);
}
```

---

## 4. Dashboards with Grafana

Metrics are useless if nobody looks at them. Dashboards turn raw numbers into visual understanding. **Grafana** is the standard tool for this: it connects to Prometheus (and many other data sources) and lets you build dashboards with graphs, tables, gauges, and alerts.

### Dashboard Architecture

The connection between your application and Grafana looks like this:

<svg viewBox="0 0 900 200" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:system-ui,sans-serif">
  <defs>
    <marker id="arr-dash" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>

  <!-- App -->
  <rect x="30" y="60" width="150" height="80" rx="8" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="105" y="95" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Your App</text>
  <text x="105" y="115" text-anchor="middle" font-size="11" fill="#666">GET /metrics</text>

  <!-- Prometheus -->
  <rect x="280" y="60" width="150" height="80" rx="8" fill="#ffebee" stroke="#ef5350" stroke-width="2"/>
  <text x="355" y="95" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Prometheus</text>
  <text x="355" y="115" text-anchor="middle" font-size="11" fill="#666">Scrapes + Stores</text>

  <!-- Grafana -->
  <rect x="530" y="60" width="150" height="80" rx="8" fill="#e8f5e9" stroke="#388e3c" stroke-width="2"/>
  <text x="605" y="95" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Grafana</text>
  <text x="605" y="115" text-anchor="middle" font-size="11" fill="#666">Queries + Visualizes</text>

  <!-- You -->
  <rect x="750" y="60" width="110" height="80" rx="8" fill="#f5f5f5" stroke="#333" stroke-width="2"/>
  <text x="805" y="100" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">You</text>

  <!-- Arrows -->
  <line x1="280" y1="90" x2="185" y2="90" stroke="#ef5350" stroke-width="2" marker-end="url(#arr-dash)"/>
  <text x="230" y="82" text-anchor="middle" font-size="10" fill="#ef5350">pull (every 15s)</text>

  <line x1="530" y1="100" x2="435" y2="100" stroke="#388e3c" stroke-width="2" marker-end="url(#arr-dash)"/>
  <text x="483" y="92" text-anchor="middle" font-size="10" fill="#388e3c">PromQL queries</text>

  <line x1="680" y1="100" x2="745" y2="100" stroke="#333" stroke-width="2" marker-end="url(#arr-dash)"/>
  <text x="713" y="92" text-anchor="middle" font-size="10" fill="#333">dashboards</text>
</svg>

### The Four Essential Dashboards

For an AI video generation platform, you need exactly four dashboards. More than four and people stop looking. Fewer than four and you miss critical signals.

#### Dashboard 1: System Overview

This is the dashboard you look at first during an incident. It answers: "Is the system healthy right now?"

| Panel | PromQL Query | Visualization |
|-------|-------------|---------------|
| Request rate | `rate(http_requests_total[5m])` | Time series graph |
| Error rate (%) | `rate(http_requests_total{status_code=~"5.."}[5m]) / rate(http_requests_total[5m]) * 100` | Time series with threshold at 5% |
| P50 latency | `histogram_quantile(0.5, rate(http_request_duration_seconds_bucket[5m]))` | Time series |
| P95 latency | `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))` | Time series |
| P99 latency | `histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))` | Time series |
| Active generations | `video_generations_active` | Gauge |

#### Dashboard 2: Generation Pipeline

This dashboard tells you how the core product is performing.

| Panel | PromQL Query | Visualization |
|-------|-------------|---------------|
| Generations per minute by model | `rate(video_generations_total[5m]) * 60` by `model` | Stacked time series |
| Success rate by model | `rate(video_generations_total{status="success"}[5m]) / rate(video_generations_total[5m]) * 100` by `model` | Time series |
| Generation duration P95 by model | `histogram_quantile(0.95, rate(video_generation_duration_seconds_bucket[5m]))` by `model` | Time series |
| Queue depth | `job_queue_depth` by `state` | Stacked area chart |
| Cost accumulation | `increase(video_generation_cost_dollars_total[1h])` | Time series, $/hour |

#### Dashboard 3: Infrastructure

This dashboard shows the health of the underlying systems.

| Panel | PromQL Query | Visualization |
|-------|-------------|---------------|
| CPU usage | `rate(process_cpu_seconds_total[5m])` | Time series |
| Memory usage | `process_resident_memory_bytes` | Time series with max line |
| Event loop lag | `nodejs_eventloop_lag_p99_seconds` | Time series |
| DB connection pool | `db_connection_pool_utilization_ratio` | Gauge (0-100%) |
| Redis memory | `redis_memory_usage_bytes` | Time series with max line |

#### Dashboard 4: Business Metrics

This is the dashboard that ties infrastructure to revenue.

| Panel | PromQL Query | Visualization |
|-------|-------------|---------------|
| Generations per hour | `increase(video_generations_total{status="success"}[1h])` | Time series |
| Cost per generation | `rate(video_generation_cost_dollars_total[1h]) / rate(video_generations_total{status="success"}[1h])` | Time series |
| Total spend today | `increase(video_generation_cost_dollars_total[24h])` | Stat panel |
| Cost by model | `increase(video_generation_cost_dollars_total[1h])` by `model` | Pie chart |

### Dashboard Layout

<svg viewBox="0 0 900 500" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#1a1a2e;font-family:system-ui,sans-serif">
  <text x="450" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#e0e0e0">Grafana Dashboard Layout: System Overview</text>

  <!-- Row 1: Key stats -->
  <rect x="20" y="50" width="200" height="80" rx="6" fill="#16213e" stroke="#333" stroke-width="1"/>
  <text x="120" y="75" text-anchor="middle" font-size="11" fill="#888">Request Rate</text>
  <text x="120" y="105" text-anchor="middle" font-size="28" font-weight="bold" fill="#4fc3f7">247/s</text>

  <rect x="235" y="50" width="200" height="80" rx="6" fill="#16213e" stroke="#333" stroke-width="1"/>
  <text x="335" y="75" text-anchor="middle" font-size="11" fill="#888">Error Rate</text>
  <text x="335" y="105" text-anchor="middle" font-size="28" font-weight="bold" fill="#66bb6a">0.3%</text>

  <rect x="450" y="50" width="200" height="80" rx="6" fill="#16213e" stroke="#333" stroke-width="1"/>
  <text x="550" y="75" text-anchor="middle" font-size="11" fill="#888">P95 Latency</text>
  <text x="550" y="105" text-anchor="middle" font-size="28" font-weight="bold" fill="#ffa726">312ms</text>

  <rect x="665" y="50" width="210" height="80" rx="6" fill="#16213e" stroke="#333" stroke-width="1"/>
  <text x="770" y="75" text-anchor="middle" font-size="11" fill="#888">Active Generations</text>
  <text x="770" y="105" text-anchor="middle" font-size="28" font-weight="bold" fill="#ce93d8">12</text>

  <!-- Row 2: Request rate + Error rate graphs -->
  <rect x="20" y="145" width="430" height="160" rx="6" fill="#16213e" stroke="#333" stroke-width="1"/>
  <text x="235" y="170" text-anchor="middle" font-size="12" fill="#888">Request Rate (req/s)</text>
  <!-- Simulated graph line -->
  <polyline points="40,270 80,260 120,255 160,265 200,250 240,245 280,260 320,240 360,235 400,250 430,242" fill="none" stroke="#4fc3f7" stroke-width="2"/>
  <line x1="40" y1="280" x2="430" y2="280" stroke="#333" stroke-width="1"/>

  <rect x="465" y="145" width="410" height="160" rx="6" fill="#16213e" stroke="#333" stroke-width="1"/>
  <text x="670" y="170" text-anchor="middle" font-size="12" fill="#888">Error Rate (%)</text>
  <polyline points="485,270 525,272 565,268 605,274 645,270 685,260 725,265 765,270 805,268 845,272" fill="none" stroke="#66bb6a" stroke-width="2"/>
  <!-- Threshold line -->
  <line x1="485" y1="200" x2="855" y2="200" stroke="#ef5350" stroke-width="1" stroke-dasharray="4,4"/>
  <text x="860" y="204" font-size="9" fill="#ef5350">5% threshold</text>
  <line x1="485" y1="280" x2="855" y2="280" stroke="#333" stroke-width="1"/>

  <!-- Row 3: Latency percentiles + Queue depth -->
  <rect x="20" y="320" width="430" height="160" rx="6" fill="#16213e" stroke="#333" stroke-width="1"/>
  <text x="235" y="345" text-anchor="middle" font-size="12" fill="#888">Latency Percentiles (seconds)</text>
  <polyline points="40,440 80,435 120,430 160,438 200,425 240,420 280,435 320,415 360,425 400,420 430,418" fill="none" stroke="#66bb6a" stroke-width="2"/>
  <polyline points="40,420 80,415 120,410 160,418 200,405 240,400 280,415 320,395 360,405 400,400 430,398" fill="none" stroke="#ffa726" stroke-width="2"/>
  <polyline points="40,390 80,385 120,380 160,388 200,375 240,370 280,385 320,365 360,375 400,370 430,368" fill="none" stroke="#ef5350" stroke-width="2"/>
  <line x1="40" y1="455" x2="430" y2="455" stroke="#333" stroke-width="1"/>
  <!-- Legend -->
  <line x1="60" y1="462" x2="80" y2="462" stroke="#66bb6a" stroke-width="2"/>
  <text x="85" y="466" font-size="9" fill="#888">P50</text>
  <line x1="120" y1="462" x2="140" y2="462" stroke="#ffa726" stroke-width="2"/>
  <text x="145" y="466" font-size="9" fill="#888">P95</text>
  <line x1="180" y1="462" x2="200" y2="462" stroke="#ef5350" stroke-width="2"/>
  <text x="205" y="466" font-size="9" fill="#888">P99</text>

  <rect x="465" y="320" width="410" height="160" rx="6" fill="#16213e" stroke="#333" stroke-width="1"/>
  <text x="670" y="345" text-anchor="middle" font-size="12" fill="#888">Queue Depth</text>
  <polyline points="485,440 525,435 565,425 605,430 645,420 685,415 725,425 765,410 805,420 845,415" fill="none" stroke="#ce93d8" stroke-width="2"/>
  <line x1="485" y1="455" x2="855" y2="455" stroke="#333" stroke-width="1"/>
</svg>

---

## 5. Alerting That Does Not Cry Wolf

Dashboards require a human to look at them. Alerts tell you when to look. But alerting is one of those things that is worse than useless when done poorly, because of a phenomenon called **alert fatigue**.

### Alert Fatigue

**Alert fatigue** is what happens when your alerting system fires too many false or low-priority alerts. The human response is predictable: you start ignoring all alerts. You mute Slack channels. You silence PagerDuty. And then, when a real incident happens, the critical alert drowns in a sea of noise that you have trained yourself to ignore.

The goal of alerting is not to notify you of every anomaly. It is to wake you up only when a human needs to act. Every alert should pass this test: "If I receive this alert at 3 AM, will I need to do something right now?"

### Rules of Good Alerting

1. **Alert on symptoms, not causes.** Alert on "error rate above 5%" (a symptom users experience), not on "CPU above 80%" (a cause that might or might not affect users). High CPU is not a problem if latency is normal.

2. **Alert on user-impacting conditions.** If it does not affect users, it is a dashboard metric, not an alert. Database replication lag of 500ms? Dashboard. Database replication lag of 30 seconds causing stale reads that users can see? Alert.

3. **Use severity levels.** Not every alert needs to wake someone up at 3 AM. CRITICAL means page someone now. WARNING means look at it during business hours. INFO means it is logged for trend analysis.

4. **Require sustained conditions.** Do not alert on a single data point. A momentary spike to 6% error rate that resolves in 30 seconds is not worth waking someone. Alert when the condition persists for a defined duration.

### Alert Rules for the Video Platform

Here are the concrete alert rules, expressed as Prometheus alerting rules. I will explain the reasoning behind each threshold.

```yaml
# prometheus/alerts.yml
groups:
  - name: video-platform-critical
    rules:
      # CRITICAL: Error rate above 5% for 5 minutes
      # WHY 5%: Below 5%, most users are unaffected and it could be
      # transient API hiccups. Above 5%, a significant fraction of
      # users are experiencing failures.
      # WHY 5 minutes: Eliminates momentary spikes from deploys or
      # brief upstream hiccups. If it persists for 5 minutes, it is real.
      - alert: HighErrorRate
        expr: |
          (
            rate(http_requests_total{status_code=~"5.."}[5m])
            / rate(http_requests_total[5m])
          ) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Error rate above 5%"
          description: >
            Error rate is {{ $value | humanizePercentage }} over the
            last 5 minutes. Users are experiencing failures.

      # CRITICAL: P95 latency above 5 seconds for 5 minutes
      # WHY 5s: Our SLA target is P95 < 500ms for API responses
      # (not generation time, which is expected to be 30-300s).
      # 5 seconds means something is fundamentally broken:
      # connection pool exhaustion, Redis slowdown, etc.
      # WHY 5 minutes: Same reasoning as above --- sustained means real.
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            rate(http_request_duration_seconds_bucket[5m])
          ) > 5
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "P95 latency above 5 seconds"
          description: >
            P95 API latency is {{ $value | humanizeDuration }}.
            This indicates systemic slowdown.

  - name: video-platform-warning
    rules:
      # WARNING: Queue depth above 100 for 10 minutes
      # WHY 100: With our worker concurrency of 5 workers x 3
      # concurrent jobs = 15 jobs processing at once, a queue of 100
      # means ~7 minutes of backlog. Users will wait too long.
      # WHY 10 minutes: Queue depth fluctuates. A burst of signups
      # might spike it briefly. 10 minutes of sustained depth means
      # we are not keeping up with demand.
      - alert: HighQueueDepth
        expr: job_queue_depth{state="waiting"} > 100
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Generation queue depth above 100"
          description: >
            Queue has {{ $value }} waiting jobs. Users may experience
            excessive wait times. Consider scaling workers.

      # WARNING: Database connection pool above 80% for 5 minutes
      # WHY 80%: At 80% utilization, you have very little headroom
      # for traffic spikes. At 100%, new requests queue for a
      # connection and latency spikes dramatically.
      # WHY 5 minutes: Connection pool usage is bursty. 80% for
      # a few seconds during a traffic spike is fine. 80% sustained
      # means you are at structural capacity.
      - alert: HighDBPoolUtilization
        expr: db_connection_pool_utilization_ratio > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Database connection pool above 80%"
          description: >
            Pool utilization at {{ $value | humanizePercentage }}.
            Risk of connection exhaustion under load.

      # WARNING: Hourly spend exceeds budget
      # WHY: This is specific to AI platforms. A bug or attack
      # that triggers thousands of generations can cost real money.
      # Set your threshold based on your expected peak hourly spend.
      - alert: HighHourlySpend
        expr: increase(video_generation_cost_dollars_total[1h]) > 50
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Hourly generation spend exceeds $50"
          description: >
            Spent ${{ $value | printf "%.2f" }} on generation APIs
            in the last hour. Normal peak is ~$20/hour.
```

### Routing Alerts

Alerts are only useful if they reach the right person through the right channel at the right time.

| Severity | Channel | Who | When |
|----------|---------|-----|------|
| CRITICAL | PagerDuty + SMS | On-call engineer | 24/7, wakes you up |
| WARNING | Slack #alerts channel | Engineering team | Business hours, reviewed daily |
| INFO | Dashboard annotation | Nobody directly | Visible on graphs for context |

---

## 6. The Failure Taxonomy

This is the most important section of this article. Understanding *how* production systems fail --- not just *that* they fail --- is what separates reliable platforms from fragile ones. Every failure mode described here has burned real money in production, and every mitigation is battle-tested.

### 6.1 Timeouts

A **timeout** is a limit on how long you will wait for an operation to complete before giving up. This sounds simple. It is not.

Without timeouts, a single slow dependency can consume all your resources. Imagine your API server has a connection pool of 20 database connections. One of those connections is stuck on a slow query. Without a timeout, that connection is consumed forever. If this happens to 20 connections, your entire server stops. This is called **resource exhaustion**, and it is one of the most common production failure modes.

There are three distinct timeouts you must configure, and confusing them causes subtle bugs:

- **Connect timeout**: How long to wait for the TCP connection to be established. If this times out, the remote server is unreachable. Typical value: 3-5 seconds.
- **Read timeout** (also called socket timeout): How long to wait for data after the connection is established. If this times out, the remote server is alive but not responding. Typical value: 10-30 seconds depending on the operation.
- **Total timeout**: The maximum wall-clock time for the entire operation, including retries. This is your last line of defense against operations that technically make progress but never finish. Typical value: 60-120 seconds.

```typescript
// src/resilience/timeouts.ts

interface TimeoutConfig {
  connectTimeoutMs: number;
  readTimeoutMs: number;
  totalTimeoutMs: number;
}

// Different operations have very different timeout profiles.
// A database query should complete in milliseconds.
// A video model API call takes 30-300 seconds.
const TIMEOUT_CONFIGS: Record<string, TimeoutConfig> = {
  database: {
    connectTimeoutMs: 3_000,
    readTimeoutMs: 10_000,
    totalTimeoutMs: 15_000,
  },
  redis: {
    connectTimeoutMs: 2_000,
    readTimeoutMs: 5_000,
    totalTimeoutMs: 8_000,
  },
  modelApi: {
    connectTimeoutMs: 5_000,
    readTimeoutMs: 300_000,  // 5 minutes: model generation is slow
    totalTimeoutMs: 360_000, // 6 minutes: allows for one retry
  },
  storageUpload: {
    connectTimeoutMs: 5_000,
    readTimeoutMs: 60_000,
    totalTimeoutMs: 120_000,
  },
};

// A helper that wraps any async operation with a total timeout.
// The connect and read timeouts should be configured on the
// HTTP client or database driver. This is the outer safety net.
export async function withTimeout<T>(
  operation: () => Promise<T>,
  timeoutMs: number,
  operationName: string
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(
        `Operation "${operationName}" timed out after ${timeoutMs}ms`
      ));
    }, timeoutMs);

    operation()
      .then((result) => {
        clearTimeout(timer);
        resolve(result);
      })
      .catch((err) => {
        clearTimeout(timer);
        reject(err);
      });
  });
}
```

**Cascading timeouts** are the insidious version of this problem. If your API has a 30-second timeout, your database query has a 10-second timeout, and the model API has a 300-second timeout, what happens? The API times out the user at 30 seconds, but the model API call is still running. The worker is still waiting. Resources are still consumed. You need to ensure that when an outer timeout fires, it cancels all inner operations. AbortController in Node.js is the mechanism for this:

```typescript
// Using AbortController for cascading timeout cancellation
async function handleGenerationRequest(req: Request): Promise<void> {
  // Create an AbortController that will propagate cancellation
  // to all child operations
  const controller = new AbortController();
  const { signal } = controller;

  // Set the outer timeout
  const timeout = setTimeout(() => controller.abort(), 30_000);

  try {
    // Every child operation checks the signal
    await enqueueJob(req.body, { signal });
  } catch (err) {
    if (signal.aborted) {
      req.log.warn('request cancelled due to timeout');
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}
```

---

### 6.2 Retries

When an operation fails, the natural instinct is to try again. This is often correct --- many failures are transient (network blip, brief overload, temporary unavailability). But naive retries are one of the most dangerous patterns in distributed systems.

**The retry storm problem**: Your model API briefly returns 503 errors. You have 50 workers, each processing a request. All 50 get a 503 and immediately retry. Now the model API receives 100 requests instead of 50. It was already overloaded. This makes it worse. All 100 fail. All 100 retry again. Now 200 requests. This positive feedback loop can take down both your system and the upstream service. This is called a **retry storm** or **thundering herd**.

The solution is **exponential backoff with jitter**.

**Exponential backoff** means each successive retry waits longer: first 1 second, then 2 seconds, then 4 seconds, then 8 seconds. This spreads out the retries over time, giving the upstream service time to recover.

**Jitter** adds randomness to the backoff delay. Without jitter, if 50 workers all start their first retry at exactly 1 second, you still get a burst of 50 at the 1-second mark. Jitter randomizes the delay so the 50 retries are spread across a window instead of arriving simultaneously.

The math is straightforward. For retry attempt \(n\) (starting at 0), the delay is:

$$
\text{delay}(n) = \min\left(\text{baseDelay} \times 2^n + \text{random}(0, \text{jitter}), \;\text{maxDelay}\right)
$$

where `baseDelay` is typically 1 second, `jitter` is typically equal to `baseDelay`, and `maxDelay` caps the exponential growth (typically 30-60 seconds).

**Retry budgets** add a second layer of protection. Instead of allowing infinite retries, you define a budget: "no more than 20% of requests can be retries." If your system is sending more than 20% retries, something is systematically broken and retrying will not help --- you need to alert and back off.

```typescript
// src/resilience/retry.ts
import { logger } from '../logger';

interface RetryConfig {
  maxRetries: number;
  baseDelayMs: number;
  maxDelayMs: number;
  jitterMs: number;
  retryableErrors: (error: Error) => boolean;
}

const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxRetries: 3,
  baseDelayMs: 1_000,
  maxDelayMs: 30_000,
  jitterMs: 1_000,
  // Only retry transient errors. A 400 Bad Request will never succeed
  // on retry, so retrying it wastes resources and money.
  retryableErrors: (err: Error) => {
    const message = err.message.toLowerCase();
    return (
      message.includes('503') ||
      message.includes('429') ||
      message.includes('timeout') ||
      message.includes('econnreset') ||
      message.includes('econnrefused') ||
      message.includes('socket hang up')
    );
  },
};

// Retry budget: track retry ratio over a sliding window
class RetryBudget {
  private requests = 0;
  private retries = 0;
  private readonly maxRetryRatio: number;
  private readonly windowMs: number;

  constructor(maxRetryRatio = 0.2, windowMs = 60_000) {
    this.maxRetryRatio = maxRetryRatio;
    this.windowMs = windowMs;

    // Reset counters every window
    setInterval(() => {
      this.requests = 0;
      this.retries = 0;
    }, this.windowMs);
  }

  recordRequest(): void {
    this.requests++;
  }

  canRetry(): boolean {
    if (this.requests === 0) return true;
    const currentRatio = this.retries / this.requests;
    return currentRatio < this.maxRetryRatio;
  }

  recordRetry(): void {
    this.retries++;
  }
}

const retryBudget = new RetryBudget();

export async function withRetry<T>(
  operation: () => Promise<T>,
  operationName: string,
  config: Partial<RetryConfig> = {}
): Promise<T> {
  const cfg = { ...DEFAULT_RETRY_CONFIG, ...config };

  retryBudget.recordRequest();

  let lastError: Error | undefined;

  for (let attempt = 0; attempt <= cfg.maxRetries; attempt++) {
    try {
      return await operation();
    } catch (err) {
      lastError = err as Error;

      // Do not retry non-retryable errors
      if (!cfg.retryableErrors(lastError)) {
        logger.warn(
          { operationName, attempt, error: lastError.message },
          'non-retryable error, not retrying'
        );
        throw lastError;
      }

      // Do not retry if we have exhausted our retry budget
      if (!retryBudget.canRetry()) {
        logger.warn(
          { operationName, attempt },
          'retry budget exhausted, not retrying'
        );
        throw lastError;
      }

      // Do not retry if we have exhausted all attempts
      if (attempt === cfg.maxRetries) {
        break;
      }

      // Calculate delay with exponential backoff + jitter
      const exponentialDelay = cfg.baseDelayMs * Math.pow(2, attempt);
      const jitter = Math.random() * cfg.jitterMs;
      const delay = Math.min(exponentialDelay + jitter, cfg.maxDelayMs);

      retryBudget.recordRetry();

      logger.info(
        { operationName, attempt, delay: Math.round(delay), error: lastError.message },
        `retrying after delay`
      );

      await sleep(delay);
    }
  }

  throw lastError!;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
```

---

### 6.3 Backpressure

**Backpressure** is what happens when a system receives more work than it can process. Think of it like water flowing through a pipe: if the inflow rate exceeds the outflow rate, pressure builds up. In software, that pressure manifests as growing queues, increasing memory usage, and rising latency.

Without backpressure management, overload looks like this: requests pile up in memory, the system slows down under the weight of accumulated work, latency increases for everyone (including the requests that arrived when the system was healthy), and eventually the system crashes from memory exhaustion, killing everything in flight.

The correct response to overload is **load shedding**: explicitly refusing new work when you cannot handle it, so that the work you are already doing can complete successfully. It is better to reject 20% of requests cleanly than to degrade service for 100% of requests.

```typescript
// src/resilience/backpressure.ts
import { Request, Response, NextFunction } from 'express';
import { queueDepth } from '../metrics';

interface BackpressureConfig {
  // Maximum queue depth before rejecting new work
  maxQueueDepth: number;
  // Maximum concurrent requests the API server will process
  maxConcurrentRequests: number;
}

const config: BackpressureConfig = {
  maxQueueDepth: 200,
  maxConcurrentRequests: 100,
};

let currentConcurrentRequests = 0;

export function backpressureMiddleware(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  // Check 1: Are we at concurrent request capacity?
  if (currentConcurrentRequests >= config.maxConcurrentRequests) {
    req.log.warn(
      { currentConcurrent: currentConcurrentRequests },
      'rejecting request: concurrent request limit reached'
    );

    // 429 Too Many Requests tells the client to back off.
    // The Retry-After header tells it how long to wait.
    res.status(429).set('Retry-After', '5').json({
      error: 'too_many_requests',
      message: 'Server is at capacity. Please retry in a few seconds.',
      retryAfter: 5,
    });
    return;
  }

  currentConcurrentRequests++;

  res.on('finish', () => {
    currentConcurrentRequests--;
  });

  next();
}

// For the generation endpoint specifically, also check queue depth
export async function checkQueueBackpressure(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  // We read queue depth from our metrics gauge, but you could also
  // query BullMQ directly with queue.getWaitingCount()
  const currentDepth = (await getQueueWaitingCount()) || 0;

  if (currentDepth >= config.maxQueueDepth) {
    req.log.warn(
      { queueDepth: currentDepth, maxQueueDepth: config.maxQueueDepth },
      'rejecting generation: queue depth limit reached'
    );

    res.status(429).set('Retry-After', '30').json({
      error: 'queue_full',
      message: 'Generation queue is full. Please try again in 30 seconds.',
      retryAfter: 30,
      queueDepth: currentDepth,
    });
    return;
  }

  next();
}

// Placeholder --- wire this to your actual BullMQ queue
async function getQueueWaitingCount(): Promise<number> {
  // In practice: return await generationQueue.getWaitingCount();
  return 0;
}
```

The key insight is that a 429 response is infinitely better than a timeout. The 429 returns in milliseconds, clearly communicates what happened, and tells the client exactly when to retry. A timeout wastes the client's time, gives no useful information, and may cause the client to retry immediately (making things worse).

---

### 6.4 Idempotency

**Idempotency** means that performing the same operation multiple times produces the same result as performing it once. Mathematically, a function \(f\) is idempotent if \(f(f(x)) = f(x)\).

Why does this matter? Because retries exist. When a network timeout occurs, the client does not know whether the server received and processed the request or not. The request might have been processed successfully, but the response was lost in transit. The client retries. If the operation is not idempotent, the user gets charged twice, or two videos are generated, or two database rows are created.

Here is the concrete scenario: a user clicks "Generate Video." The API receives the request, enqueues the job, charges the user's credits, and sends back a response. But the response never reaches the client (network hiccup). The client retries. Without idempotency protection, the user is charged twice and gets two identical generation jobs.

The solution is **idempotency keys**: the client sends a unique key with each request, and the server uses this key to deduplicate.

```typescript
// src/resilience/idempotency.ts
import Redis from 'ioredis';
import { Request, Response, NextFunction } from 'express';

const redis = new Redis(process.env.REDIS_URL!);

// Idempotency keys expire after 24 hours. This means:
// - A retry within 24 hours will be deduplicated (correct)
// - A genuinely new request after 24 hours with a recycled key will
//   go through (acceptable, since keys should be UUIDs)
const IDEMPOTENCY_TTL_SECONDS = 86_400; // 24 hours

interface StoredResponse {
  statusCode: number;
  body: unknown;
  completedAt: string;
}

export function idempotencyMiddleware(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  // Only apply to state-changing operations
  if (req.method !== 'POST' && req.method !== 'PUT') {
    next();
    return;
  }

  const idempotencyKey = req.headers['idempotency-key'] as string | undefined;

  // If no key is provided, skip idempotency checking.
  // This is a design decision: you could require keys on all
  // POST endpoints. For generation, I recommend requiring them.
  if (!idempotencyKey) {
    next();
    return;
  }

  const cacheKey = `idempotency:${idempotencyKey}`;

  // Check if we have already processed this key
  redis.get(cacheKey)
    .then((cached) => {
      if (cached) {
        // We already processed this request. Return the stored response
        // without executing the handler again.
        const stored: StoredResponse = JSON.parse(cached);
        req.log.info(
          { idempotencyKey },
          'returning cached response for idempotent request'
        );
        res.status(stored.statusCode).json(stored.body);
        return;
      }

      // First time seeing this key. Process normally, but intercept
      // the response to cache it.
      const originalJson = res.json.bind(res);
      res.json = function (body: unknown) {
        // Store the response so future retries get the same result
        const stored: StoredResponse = {
          statusCode: res.statusCode,
          body,
          completedAt: new Date().toISOString(),
        };

        redis
          .setex(cacheKey, IDEMPOTENCY_TTL_SECONDS, JSON.stringify(stored))
          .catch((err) => {
            req.log.error(
              { error: err.message, idempotencyKey },
              'failed to cache idempotent response'
            );
          });

        return originalJson(body);
      };

      next();
    })
    .catch((err) => {
      // If Redis is down, skip idempotency rather than blocking all
      // requests. This is a deliberate tradeoff: we accept the risk
      // of duplicate processing over the certainty of total outage.
      req.log.error(
        { error: err.message },
        'idempotency check failed, proceeding without'
      );
      next();
    });
}
```

On the client side, generating and sending idempotency keys is straightforward:

```typescript
// Client-side: generate a unique key per user action
async function generateVideo(prompt: string): Promise<GenerationResult> {
  // Generate the key ONCE when the user clicks the button.
  // If the request fails and is retried, the SAME key is reused.
  const idempotencyKey = crypto.randomUUID();

  return fetchWithRetry('/api/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Idempotency-Key': idempotencyKey,
    },
    body: JSON.stringify({ prompt }),
  });
}
```

---

### 6.5 Circuit Breakers

A **circuit breaker** is a pattern borrowed from electrical engineering. In a home electrical system, a circuit breaker trips (opens) when too much current flows, preventing a fire. In software, a circuit breaker trips when a downstream service fails too frequently, preventing your system from wasting resources on requests that will almost certainly fail.

Without a circuit breaker, here is what happens: the Kling API is down. Every generation request that uses Kling sends a request, waits for the timeout (say, 30 seconds), and fails. Your workers are tied up for 30 seconds each, doing nothing useful. Queue depth grows. Users of *other* models (Veo, Runway) are delayed because workers are stuck waiting for Kling to time out.

The circuit breaker has three states:

<svg viewBox="0 0 900 380" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:system-ui,sans-serif">
  <defs>
    <marker id="arr-cb" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
    <marker id="arr-green" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#388e3c"/>
    </marker>
    <marker id="arr-red" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#d32f2f"/>
    </marker>
    <marker id="arr-orange" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#f57c00"/>
    </marker>
  </defs>

  <text x="450" y="30" text-anchor="middle" font-size="18" font-weight="bold" fill="#333">Circuit Breaker State Machine</text>

  <!-- CLOSED state -->
  <circle cx="150" cy="200" r="80" fill="#e8f5e9" stroke="#388e3c" stroke-width="3"/>
  <text x="150" y="190" text-anchor="middle" font-size="16" font-weight="bold" fill="#388e3c">CLOSED</text>
  <text x="150" y="210" text-anchor="middle" font-size="11" fill="#555">Requests flow</text>
  <text x="150" y="225" text-anchor="middle" font-size="11" fill="#555">normally</text>

  <!-- OPEN state -->
  <circle cx="750" cy="200" r="80" fill="#ffebee" stroke="#d32f2f" stroke-width="3"/>
  <text x="750" y="190" text-anchor="middle" font-size="16" font-weight="bold" fill="#d32f2f">OPEN</text>
  <text x="750" y="210" text-anchor="middle" font-size="11" fill="#555">Requests fail</text>
  <text x="750" y="225" text-anchor="middle" font-size="11" fill="#555">immediately</text>

  <!-- HALF-OPEN state -->
  <circle cx="450" cy="200" r="80" fill="#fff3e0" stroke="#f57c00" stroke-width="3"/>
  <text x="450" y="190" text-anchor="middle" font-size="16" font-weight="bold" fill="#f57c00">HALF-OPEN</text>
  <text x="450" y="210" text-anchor="middle" font-size="11" fill="#555">Test with one</text>
  <text x="450" y="225" text-anchor="middle" font-size="11" fill="#555">request</text>

  <!-- CLOSED -> OPEN -->
  <path d="M 230 170 Q 450 60 670 170" fill="none" stroke="#d32f2f" stroke-width="2" marker-end="url(#arr-red)"/>
  <text x="450" y="95" text-anchor="middle" font-size="12" fill="#d32f2f" font-weight="bold">Failure threshold exceeded</text>

  <!-- OPEN -> HALF-OPEN -->
  <path d="M 670 240 Q 600 310 530 240" fill="none" stroke="#f57c00" stroke-width="2" marker-end="url(#arr-orange)"/>
  <text x="620" y="310" text-anchor="middle" font-size="12" fill="#f57c00" font-weight="bold">Reset timeout expires</text>

  <!-- HALF-OPEN -> CLOSED -->
  <path d="M 370 240 Q 300 310 230 240" fill="none" stroke="#388e3c" stroke-width="2" marker-end="url(#arr-green)"/>
  <text x="280" y="310" text-anchor="middle" font-size="12" fill="#388e3c" font-weight="bold">Test request succeeds</text>

  <!-- HALF-OPEN -> OPEN -->
  <path d="M 530 170 Q 600 100 670 170" fill="none" stroke="#d32f2f" stroke-width="2" marker-end="url(#arr-red)"/>
  <text x="610" y="135" text-anchor="middle" font-size="12" fill="#d32f2f" font-weight="bold">Test fails</text>

  <!-- Legend -->
  <text x="450" y="370" text-anchor="middle" font-size="12" fill="#777" font-style="italic">The circuit breaker protects your system from wasting resources on a failing dependency.</text>
</svg>

- **Closed** (normal operation): Requests flow through to the downstream service. The breaker counts failures. When the failure rate exceeds a threshold within a time window, the breaker trips to Open.

- **Open** (protecting): All requests fail immediately with a "circuit open" error, without contacting the downstream service. This is fast (milliseconds instead of timeout seconds) and preserves resources. After a configured reset timeout, the breaker transitions to Half-Open.

- **Half-Open** (testing): The breaker allows a single test request through. If it succeeds, the service is probably healthy --- transition back to Closed. If it fails, the service is still down --- transition back to Open.

```typescript
// src/resilience/circuitBreaker.ts
import { circuitBreakerTransitions } from '../metrics';
import { logger } from '../logger';

type CircuitState = 'closed' | 'open' | 'half-open';

interface CircuitBreakerConfig {
  // How many failures in the window trigger the breaker
  failureThreshold: number;
  // Time window for counting failures (ms)
  failureWindowMs: number;
  // How long to stay open before testing (ms)
  resetTimeoutMs: number;
  // Name for logging and metrics
  name: string;
}

export class CircuitBreaker {
  private state: CircuitState = 'closed';
  private failures: number[] = []; // timestamps of recent failures
  private lastStateChange: number = Date.now();
  private readonly config: CircuitBreakerConfig;

  constructor(config: CircuitBreakerConfig) {
    this.config = config;
  }

  async execute<T>(operation: () => Promise<T>): Promise<T> {
    // Check if the circuit allows this request
    if (this.state === 'open') {
      // Has enough time passed to try again?
      const elapsed = Date.now() - this.lastStateChange;
      if (elapsed < this.config.resetTimeoutMs) {
        // Still in cooldown. Fail fast.
        throw new CircuitOpenError(
          `Circuit breaker "${this.config.name}" is open. ` +
          `Will retry in ${Math.ceil((this.config.resetTimeoutMs - elapsed) / 1000)}s.`
        );
      }
      // Transition to half-open: allow one test request
      this.transitionTo('half-open');
    }

    try {
      const result = await operation();

      // Success: reset the circuit
      if (this.state === 'half-open') {
        this.transitionTo('closed');
      }
      // In closed state, a success does not reset the failure window.
      // Failures naturally age out of the window.

      return result;
    } catch (err) {
      this.recordFailure();

      if (this.state === 'half-open') {
        // The test request failed. Back to open.
        this.transitionTo('open');
      } else if (this.state === 'closed') {
        // Check if we have exceeded the failure threshold
        if (this.getRecentFailureCount() >= this.config.failureThreshold) {
          this.transitionTo('open');
        }
      }

      throw err;
    }
  }

  private recordFailure(): void {
    this.failures.push(Date.now());
    // Prune old failures outside the window
    const cutoff = Date.now() - this.config.failureWindowMs;
    this.failures = this.failures.filter((t) => t > cutoff);
  }

  private getRecentFailureCount(): number {
    const cutoff = Date.now() - this.config.failureWindowMs;
    return this.failures.filter((t) => t > cutoff).length;
  }

  private transitionTo(newState: CircuitState): void {
    const oldState = this.state;
    this.state = newState;
    this.lastStateChange = Date.now();

    if (newState === 'closed') {
      this.failures = [];
    }

    // Record the transition for metrics and logging
    circuitBreakerTransitions.inc({
      service: this.config.name,
      from_state: oldState,
      to_state: newState,
    });

    logger.info(
      { breaker: this.config.name, from: oldState, to: newState },
      'circuit breaker state transition'
    );
  }

  getState(): CircuitState {
    return this.state;
  }
}

export class CircuitOpenError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CircuitOpenError';
  }
}
```

Using the circuit breaker with a model API:

```typescript
// Create one circuit breaker per downstream service
const klingBreaker = new CircuitBreaker({
  name: 'kling-api',
  failureThreshold: 5,     // 5 failures in the window triggers the breaker
  failureWindowMs: 60_000, // Count failures over 1-minute windows
  resetTimeoutMs: 30_000,  // Wait 30 seconds before testing again
});

const veoBreaker = new CircuitBreaker({
  name: 'veo-api',
  failureThreshold: 5,
  failureWindowMs: 60_000,
  resetTimeoutMs: 30_000,
});

async function callModelAPI(
  params: GenerationParams,
  log: RequestLogger
): Promise<GenerationResult> {
  const breaker = getBreaker(params.model);

  try {
    return await breaker.execute(() =>
      withRetry(
        () => makeAPICall(params),
        `${params.model}-generation`,
        { maxRetries: 2 }
      )
    );
  } catch (err) {
    if (err instanceof CircuitOpenError) {
      log.warn(
        { model: params.model, error: err.message },
        'circuit breaker is open, failing fast'
      );
      // Optionally: try a fallback model
      // return await callFallbackModel(params, log);
    }
    throw err;
  }
}

function getBreaker(model: string): CircuitBreaker {
  switch (model) {
    case 'kling-v3': return klingBreaker;
    case 'veo-3': return veoBreaker;
    default: throw new Error(`Unknown model: ${model}`);
  }
}
```

---

### 6.6 Cost Explosions

This failure mode is unique to AI platforms and deserves its own section because it burns real money.

Consider: your AI video platform uses model APIs that charge per generation. Kling charges $0.08 per 5-second clip. Veo charges $0.10. Runway charges $0.05. At normal traffic (50 generations per hour), your hourly API cost is about $4. Manageable.

Now a bug slips into production. Maybe the frontend retries on every error without backoff. Maybe a worker enters an infinite retry loop. Maybe a free-tier user discovers they can hit the API directly and scripts 10,000 requests. Suddenly you are sending 10,000 generation requests at $0.10 each. That is $1,000 gone in minutes. And because model API charges are not refundable, that money is simply gone.

The defenses are layered:

**Layer 1: Per-user rate limits**

```typescript
// src/resilience/costProtection.ts
import Redis from 'ioredis';

const redis = new Redis(process.env.REDIS_URL!);

interface SpendingLimits {
  // Maximum generations per user per hour
  maxGenerationsPerHour: number;
  // Maximum spend per user per day (dollars)
  maxSpendPerUserPerDay: number;
  // Maximum total platform spend per hour (dollars)
  maxPlatformSpendPerHour: number;
}

const LIMITS: SpendingLimits = {
  maxGenerationsPerHour: 20,     // Even a power user rarely needs >20/hr
  maxSpendPerUserPerDay: 10,     // $10/day per user is very generous
  maxPlatformSpendPerHour: 100,  // $100/hr is 25x normal traffic
};

export async function checkSpendingLimits(
  userId: string,
  estimatedCost: number
): Promise<{ allowed: boolean; reason?: string }> {
  const now = Date.now();
  const hourKey = `ratelimit:gen:${userId}:${Math.floor(now / 3_600_000)}`;
  const dayKey = `spend:${userId}:${Math.floor(now / 86_400_000)}`;
  const platformHourKey = `spend:platform:${Math.floor(now / 3_600_000)}`;

  // Check per-user generation rate
  const userGenCount = await redis.incr(hourKey);
  if (userGenCount === 1) {
    await redis.expire(hourKey, 3600);
  }
  if (userGenCount > LIMITS.maxGenerationsPerHour) {
    return {
      allowed: false,
      reason: `Rate limit exceeded: ${LIMITS.maxGenerationsPerHour} generations per hour`,
    };
  }

  // Check per-user daily spend
  const userDailySpend = parseFloat(
    (await redis.get(dayKey)) || '0'
  );
  if (userDailySpend + estimatedCost > LIMITS.maxSpendPerUserPerDay) {
    return {
      allowed: false,
      reason: `Daily spending limit of $${LIMITS.maxSpendPerUserPerDay} reached`,
    };
  }

  // Check platform-wide hourly spend (the kill switch layer)
  const platformSpend = parseFloat(
    (await redis.get(platformHourKey)) || '0'
  );
  if (platformSpend + estimatedCost > LIMITS.maxPlatformSpendPerHour) {
    return {
      allowed: false,
      reason: 'Platform spending limit reached. All generations paused.',
    };
  }

  // Record the spend
  await redis.incrbyfloat(dayKey, estimatedCost);
  if ((await redis.ttl(dayKey)) === -1) {
    await redis.expire(dayKey, 86_400);
  }

  await redis.incrbyfloat(platformHourKey, estimatedCost);
  if ((await redis.ttl(platformHourKey)) === -1) {
    await redis.expire(platformHourKey, 3_600);
  }

  return { allowed: true };
}
```

**Layer 2: The Kill Switch**

Every AI platform needs a kill switch --- a single toggle that stops all generation requests immediately. This is not something you hope to never use. It is something you will use, and when you need it, you need it to work instantly.

```typescript
// src/resilience/killSwitch.ts
import Redis from 'ioredis';
import { logger } from '../logger';

const redis = new Redis(process.env.REDIS_URL!);
const KILL_SWITCH_KEY = 'killswitch:generations';

// Check the kill switch before every generation
export async function isKillSwitchActive(): Promise<boolean> {
  const value = await redis.get(KILL_SWITCH_KEY);
  return value === 'active';
}

// Activate the kill switch. This should be callable from:
// 1. An admin API endpoint
// 2. A Slack bot command
// 3. An automated alert response
export async function activateKillSwitch(reason: string): Promise<void> {
  await redis.set(KILL_SWITCH_KEY, 'active');
  logger.fatal(
    { reason },
    'KILL SWITCH ACTIVATED --- all generations halted'
  );
  // Send immediate notification to all engineering channels
  // await notifySlack('#engineering', `Kill switch activated: ${reason}`);
  // await notifyPagerDuty('Kill switch activated', reason);
}

export async function deactivateKillSwitch(): Promise<void> {
  await redis.del(KILL_SWITCH_KEY);
  logger.info('Kill switch deactivated --- generations resumed');
}
```

Here is how all the failure mode defenses compose together in a single generation request flow:

<svg viewBox="0 0 900 600" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:system-ui,sans-serif">
  <defs>
    <marker id="arr-flow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
    <marker id="arr-reject" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#d32f2f"/>
    </marker>
  </defs>

  <text x="450" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">Defense Layers for a Generation Request</text>

  <!-- Request arrives -->
  <rect x="350" y="40" width="200" height="40" rx="6" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="450" y="65" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Incoming Request</text>
  <line x1="450" y1="80" x2="450" y2="105" stroke="#333" stroke-width="2" marker-end="url(#arr-flow)"/>

  <!-- Layer 1: Kill Switch -->
  <rect x="300" y="110" width="300" height="40" rx="6" fill="#ffebee" stroke="#d32f2f" stroke-width="2"/>
  <text x="450" y="135" text-anchor="middle" font-size="12" font-weight="bold" fill="#d32f2f">Layer 1: Kill Switch Check</text>
  <line x1="600" y1="130" x2="720" y2="130" stroke="#d32f2f" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arr-reject)"/>
  <text x="730" y="134" font-size="10" fill="#d32f2f">503 Service Unavailable</text>
  <line x1="450" y1="150" x2="450" y2="175" stroke="#333" stroke-width="2" marker-end="url(#arr-flow)"/>

  <!-- Layer 2: Backpressure -->
  <rect x="300" y="180" width="300" height="40" rx="6" fill="#fff3e0" stroke="#f57c00" stroke-width="2"/>
  <text x="450" y="205" text-anchor="middle" font-size="12" font-weight="bold" fill="#f57c00">Layer 2: Backpressure / Queue Depth</text>
  <line x1="600" y1="200" x2="720" y2="200" stroke="#d32f2f" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arr-reject)"/>
  <text x="730" y="204" font-size="10" fill="#d32f2f">429 Too Many Requests</text>
  <line x1="450" y1="220" x2="450" y2="245" stroke="#333" stroke-width="2" marker-end="url(#arr-flow)"/>

  <!-- Layer 3: Spending Limits -->
  <rect x="300" y="250" width="300" height="40" rx="6" fill="#fff3e0" stroke="#f57c00" stroke-width="2"/>
  <text x="450" y="275" text-anchor="middle" font-size="12" font-weight="bold" fill="#f57c00">Layer 3: Spending Limits</text>
  <line x1="600" y1="270" x2="720" y2="270" stroke="#d32f2f" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arr-reject)"/>
  <text x="730" y="274" font-size="10" fill="#d32f2f">429 Limit Exceeded</text>
  <line x1="450" y1="290" x2="450" y2="315" stroke="#333" stroke-width="2" marker-end="url(#arr-flow)"/>

  <!-- Layer 4: Idempotency -->
  <rect x="300" y="320" width="300" height="40" rx="6" fill="#e8f5e9" stroke="#388e3c" stroke-width="2"/>
  <text x="450" y="345" text-anchor="middle" font-size="12" font-weight="bold" fill="#388e3c">Layer 4: Idempotency Check</text>
  <line x1="600" y1="340" x2="720" y2="340" stroke="#388e3c" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arr-flow)"/>
  <text x="730" y="344" font-size="10" fill="#388e3c">200 Cached Response</text>
  <line x1="450" y1="360" x2="450" y2="385" stroke="#333" stroke-width="2" marker-end="url(#arr-flow)"/>

  <!-- Layer 5: Circuit Breaker -->
  <rect x="300" y="390" width="300" height="40" rx="6" fill="#ffebee" stroke="#d32f2f" stroke-width="2"/>
  <text x="450" y="415" text-anchor="middle" font-size="12" font-weight="bold" fill="#d32f2f">Layer 5: Circuit Breaker</text>
  <line x1="600" y1="410" x2="720" y2="410" stroke="#d32f2f" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arr-reject)"/>
  <text x="730" y="414" font-size="10" fill="#d32f2f">503 Circuit Open</text>
  <line x1="450" y1="430" x2="450" y2="455" stroke="#333" stroke-width="2" marker-end="url(#arr-flow)"/>

  <!-- Layer 6: Retry with Backoff -->
  <rect x="300" y="460" width="300" height="40" rx="6" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="450" y="485" text-anchor="middle" font-size="12" font-weight="bold" fill="#1976d2">Layer 6: Retry with Exp. Backoff</text>
  <line x1="450" y1="500" x2="450" y2="525" stroke="#333" stroke-width="2" marker-end="url(#arr-flow)"/>

  <!-- Model API call -->
  <rect x="350" y="530" width="200" height="40" rx="6" fill="#e8f5e9" stroke="#388e3c" stroke-width="2"/>
  <text x="450" y="555" text-anchor="middle" font-size="13" font-weight="bold" fill="#388e3c">Model API Call</text>
</svg>

Each layer either passes the request through or rejects it. The rejections get progressively more expensive to reach: the kill switch is a simple Redis GET (sub-millisecond), backpressure is an in-memory check, spending limits are a Redis lookup, idempotency is a Redis lookup, and the circuit breaker is an in-memory check. Only requests that pass all six layers make the expensive API call.

---

## 7. Distributed Tracing with OpenTelemetry

When a single user request touches multiple services --- API server, Redis, PostgreSQL, job queue, worker, external model API, object storage --- and something is slow, you need to know *where* in that chain the time is being spent. This is what distributed tracing gives you.

### Concepts

**OpenTelemetry** (often abbreviated OTel) is the industry-standard framework for collecting traces (and metrics and logs). It is vendor-neutral: you instrument your code once with OpenTelemetry, and you can send the data to any backend (Jaeger, Zipkin, Grafana Tempo, Datadog, etc.).

A **trace** represents the complete journey of a single request through your system. It has a unique trace ID.

A **span** represents a single operation within a trace. A span has a name ("database query"), a start time, a duration, and optional attributes (the SQL query, the number of rows returned). Spans can be nested: a parent span "handle generation request" might contain child spans for "validate input," "enqueue job," and "write to database."

The trace ID is propagated across service boundaries via HTTP headers (typically `traceparent`), so that spans created in different services are linked into a single trace.

### Setting Up OpenTelemetry

```typescript
// src/tracing.ts
// This file MUST be imported before any other code ---
// OpenTelemetry works by monkey-patching libraries at import time.

import { NodeSDK } from '@opentelemetry/sdk-node';
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { Resource } from '@opentelemetry/resources';
import {
  ATTR_SERVICE_NAME,
  ATTR_SERVICE_VERSION,
} from '@opentelemetry/semantic-conventions';

const sdk = new NodeSDK({
  resource: new Resource({
    [ATTR_SERVICE_NAME]: 'video-api',
    [ATTR_SERVICE_VERSION]: process.env.APP_VERSION || '0.0.0',
  }),

  // Export traces to an OpenTelemetry Collector or directly to
  // a backend like Grafana Tempo or Jaeger
  traceExporter: new OTLPTraceExporter({
    url: process.env.OTEL_EXPORTER_OTLP_ENDPOINT || 'http://localhost:4318/v1/traces',
  }),

  // Auto-instrumentation automatically creates spans for:
  // - HTTP requests (incoming and outgoing)
  // - Express routes
  // - PostgreSQL queries
  // - Redis commands
  // - DNS lookups
  // This means you get useful traces with zero manual instrumentation.
  instrumentations: [
    getNodeAutoInstrumentations({
      // Disable fs instrumentation --- it is noisy and rarely useful
      '@opentelemetry/instrumentation-fs': { enabled: false },
    }),
  ],
});

sdk.start();

// Gracefully shut down on process exit to flush pending spans
process.on('SIGTERM', () => {
  sdk.shutdown().then(() => process.exit(0));
});
```

### Adding Custom Spans

Auto-instrumentation gives you spans for HTTP, database, and Redis operations. But it does not know about your business logic. For that, you create custom spans:

```typescript
// src/services/generationService.ts
import { trace, SpanStatusCode, context } from '@opentelemetry/api';

const tracer = trace.getTracer('video-generation');

export async function processGeneration(
  params: GenerationParams
): Promise<GenerationResult> {
  // Create a parent span for the entire generation flow
  return tracer.startActiveSpan('generation.process', async (span) => {
    span.setAttributes({
      'generation.id': params.generationId,
      'generation.model': params.model,
      'generation.prompt_length': params.prompt.length,
      'user.id': params.userId,
    });

    try {
      // Each sub-operation gets its own child span
      const validated = await tracer.startActiveSpan(
        'generation.validate',
        async (validateSpan) => {
          const result = await validatePrompt(params.prompt);
          validateSpan.setAttributes({
            'validation.passed': result.isValid,
            'validation.flags': result.flags.join(','),
          });
          validateSpan.end();
          return result;
        }
      );

      // Call the model API with its own span
      const apiResult = await tracer.startActiveSpan(
        'generation.model_api_call',
        async (apiSpan) => {
          apiSpan.setAttributes({
            'model.name': params.model,
            'model.resolution': params.resolution,
            'model.duration_seconds': params.durationSec,
          });

          const result = await callModelAPI(params);

          apiSpan.setAttributes({
            'model.cost': result.cost,
            'model.processing_time_ms': result.processingTimeMs,
          });
          apiSpan.end();
          return result;
        }
      );

      // Upload the result to object storage
      const uploadResult = await tracer.startActiveSpan(
        'generation.upload',
        async (uploadSpan) => {
          const result = await uploadToStorage(apiResult.videoBuffer);
          uploadSpan.setAttributes({
            'upload.size_bytes': apiResult.videoBuffer.length,
            'upload.storage_key': result.key,
          });
          uploadSpan.end();
          return result;
        }
      );

      span.setAttributes({
        'generation.status': 'success',
        'generation.total_cost': apiResult.cost,
        'generation.output_url': uploadResult.url,
      });
      span.setStatus({ code: SpanStatusCode.OK });

      return {
        videoUrl: uploadResult.url,
        cost: apiResult.cost,
        generationId: params.generationId,
      };
    } catch (err) {
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: (err as Error).message,
      });
      span.recordException(err as Error);
      throw err;
    } finally {
      span.end();
    }
  });
}
```

### The Trace Waterfall

When you view a trace in a tool like Jaeger or Grafana Tempo, you see a **waterfall view** --- a timeline showing every span, nested by parent-child relationships, with each span's duration represented by its horizontal length. Here is what a generation trace looks like:

```
Trace ID: abc123def456

|-- generation.process (3,240ms) ----------------------------------|
  |-- generation.validate (12ms) -|
  |-- pg.query (8ms) --|
  |-- generation.model_api_call (2,890ms) -------------------------|
    |-- http.request POST api.kling.ai/v3/generate (2,890ms) -----|
  |-- generation.upload (185ms) ----------|
    |-- http.request PUT r2.storage/video.mp4 (185ms) ------------|
  |-- pg.query (6ms) --|
  |-- redis.set (2ms) |
```

From this single view, you can immediately see that the model API call consumed 89% of the total time (2,890ms out of 3,240ms). The validation, database queries, Redis operations, and upload are negligible. If this trace is slower than expected, you know exactly where to look.

### Connecting Traces to Logs

The final piece is linking traces to structured logs. When a trace ID is available, include it in every log line:

```typescript
// Add trace context to the pino logger
import { trace, context } from '@opentelemetry/api';

function getTraceContext(): { traceId?: string; spanId?: string } {
  const span = trace.getSpan(context.active());
  if (!span) return {};

  const ctx = span.spanContext();
  return {
    traceId: ctx.traceId,
    spanId: ctx.spanId,
  };
}

// Modify the request logger middleware to include trace IDs
req.log = logger.child({
  requestId,
  ...getTraceContext(),
  method: req.method,
  path: req.path,
});
```

Now you can search logs by trace ID and see every log line from every service that participated in a specific request. You can also click a trace in Grafana Tempo and jump directly to the associated logs in Grafana Loki. The three pillars are connected.

---

## 8. The Observability Maturity Model

Not everyone needs distributed tracing on day one. Observability is a spectrum, and you should invest proportionally to your scale and risk. Here is the maturity model, based on what I have seen across dozens of deployed AI platforms:

### Level 0: console.log and SSH

**Where you are**: `console.log("error")` scattered through code. Debugging means SSH-ing into the server and running `htop`, `tail -f`, or `grep`. No metrics. No dashboards. No alerts. You find out about outages when users tweet at you.

**Risk**: Every outage requires manual investigation from scratch. No historical data. No aggregate visibility. Mean time to resolution (MTTR) is measured in hours.

**Most vibe-coded projects live here.** It works when you have 10 users. It does not work when you have 100.

### Level 1: Structured Logging + Basic Metrics

**What to implement**: Replace all `console.log` calls with a structured logger (Pino). Add correlation IDs. Add the four core Prometheus metrics: request rate, error rate, latency histogram, and active generation gauge. Expose a `/metrics` endpoint.

**Effort**: 1-2 days for an existing codebase.

**What you gain**: You can search logs by user ID, request ID, or error type. You can answer "what is our error rate right now?" without guessing. When a user reports a problem, you can find the exact request and see what happened.

### Level 2: Dashboards + Alerting

**What to implement**: Set up Prometheus scraping and Grafana dashboards (the four dashboards described in Section 4). Add alerting rules for error rate, latency, and queue depth. Route critical alerts to PagerDuty or your phone.

**Effort**: 1-2 days if you already have Level 1.

**What you gain**: You are notified of problems before users report them. You have historical graphs showing how the system behaves over time. You can see the impact of deployments. You can capacity-plan based on real usage data.

**Get to Level 2 before you call yourself production-ready.** This is the minimum bar.

### Level 3: Distributed Tracing + Anomaly Detection

**What to implement**: Add OpenTelemetry instrumentation. Set up a trace backend (Grafana Tempo, Jaeger). Connect traces to logs. Add anomaly detection on cost metrics (alert when hourly spend deviates more than 2 standard deviations from the rolling 7-day average).

**Effort**: 2-3 days for OpenTelemetry setup. Ongoing tuning for anomaly detection.

**What you gain**: You can diagnose latency issues down to the individual span level. You can see how often circuit breakers trip, how retry rates correlate with upstream outages, and whether your cost controls are working. MTTR drops from hours to minutes.

| Level | MTTR | Setup Effort | Required Scale |
|-------|------|-------------|----------------|
| 0 | Hours | None | 0-10 users |
| 1 | 30-60 min | 1-2 days | 10-100 users |
| 2 | 5-15 min | 1-2 days | 100-1,000 users |
| 3 | 1-5 min | 2-3 days | 1,000+ users |

The effort numbers are cumulative building time, not elapsed calendar time. You do not need to wait until you hit 1,000 users to implement Level 3 --- if you can invest the time early, it pays dividends immediately. But if you are resource-constrained, this prioritization tells you where to start.

---

## 9. The Production Readiness Checklist

This section ties together all four articles in this series. Use it as a gate: before you tell anyone your AI video platform is "production-ready," every item should be checked.

### Infrastructure (from Article 1: Job Queue Architecture)

- [ ] Job queue (BullMQ + Redis) for all long-running operations
- [ ] Separate worker processes from API server processes
- [ ] Job persistence across server restarts
- [ ] Dead letter queue for jobs that fail all retries
- [ ] Concurrency limits per worker
- [ ] Priority queues (paid users before free users)

### Data Layer (from Article 2: Database Schema Design)

- [ ] Database schema handles all core entities (users, projects, scenes, generations, assets, billing)
- [ ] Indexes on every query pattern
- [ ] Connection pooling with max pool size configured
- [ ] Database backups (automated, tested restore)
- [ ] Migration strategy for schema changes

### Real-Time Communication (from Article 3: WebSocket Architecture)

- [ ] Real-time status updates for generation progress
- [ ] Connection resilience (auto-reconnect, state catch-up)
- [ ] Authentication on WebSocket/SSE connections
- [ ] Graceful degradation if real-time transport fails

### Observability (this article)

- [ ] Structured logging with correlation IDs
- [ ] Log levels used correctly (not everything is `info`)
- [ ] Prometheus metrics: request rate, error rate, latency, queue depth, generation metrics
- [ ] Grafana dashboards: system overview, generation pipeline, infrastructure, business
- [ ] Alerting: error rate, latency, queue depth, cost anomalies
- [ ] Alert routing: critical to PagerDuty, warning to Slack

### Resilience (this article)

- [ ] Timeouts on every external call (connect, read, total)
- [ ] Retries with exponential backoff and jitter
- [ ] Retry budgets to prevent retry storms
- [ ] Backpressure: 429 responses when overloaded
- [ ] Idempotency keys on state-changing endpoints
- [ ] Circuit breakers on downstream service calls
- [ ] Cost controls: per-user rate limits, daily spend caps, platform-wide spend cap
- [ ] Kill switch for emergency generation halt

### Deployment

- [ ] CI/CD pipeline (tests run on every push)
- [ ] Zero-downtime deployments (rolling or blue-green)
- [ ] Environment parity (staging mirrors production configuration)
- [ ] Secrets management (not hardcoded, not in git)
- [ ] HTTPS everywhere
- [ ] Health check endpoint (`/health`)
- [ ] Graceful shutdown (drain in-flight requests before stopping)

### Business Continuity

- [ ] Error budget defined (e.g., 99.9% uptime = 43 minutes downtime per month)
- [ ] On-call rotation (even if it is just you, define when you check alerts)
- [ ] Incident response playbook (what to do when each alert fires)
- [ ] Post-mortem process (learn from every outage)
- [ ] Cost monitoring dashboard reviewed weekly

---

## Bringing It All Together

Here is the complete request flow through a production-ready AI video platform, with every defense layer and observability mechanism in place:

<svg viewBox="0 0 900 720" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:system-ui,sans-serif">
  <defs>
    <marker id="arr-final" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>

  <text x="450" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">Production AI Video Platform: Complete Architecture</text>

  <!-- User -->
  <rect x="370" y="40" width="160" height="45" rx="6" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="450" y="68" text-anchor="middle" font-size="13" font-weight="bold" fill="#1976d2">User Request</text>
  <line x1="450" y1="85" x2="450" y2="108" stroke="#333" stroke-width="1.5" marker-end="url(#arr-final)"/>

  <!-- API Layer -->
  <rect x="250" y="113" width="400" height="195" rx="8" fill="#f5f5f5" stroke="#999" stroke-width="1.5" stroke-dasharray="5,3"/>
  <text x="450" y="133" text-anchor="middle" font-size="12" font-weight="bold" fill="#666">API Server</text>

  <!-- Defense layers inside API -->
  <rect x="310" y="142" width="280" height="28" rx="4" fill="#ffebee" stroke="#ef5350" stroke-width="1"/>
  <text x="450" y="161" text-anchor="middle" font-size="10" fill="#d32f2f">Kill Switch + Backpressure + Rate Limits</text>

  <rect x="310" y="178" width="280" height="28" rx="4" fill="#e8f5e9" stroke="#66bb6a" stroke-width="1"/>
  <text x="450" y="197" text-anchor="middle" font-size="10" fill="#388e3c">Idempotency + Validation</text>

  <rect x="310" y="214" width="280" height="28" rx="4" fill="#e3f2fd" stroke="#42a5f5" stroke-width="1"/>
  <text x="450" y="233" text-anchor="middle" font-size="10" fill="#1976d2">Structured Logging + Metrics + Tracing</text>

  <rect x="310" y="250" width="280" height="28" rx="4" fill="#fff3e0" stroke="#ffa726" stroke-width="1"/>
  <text x="450" y="269" text-anchor="middle" font-size="10" fill="#f57c00">Enqueue Job (BullMQ + Redis)</text>

  <line x1="450" y1="308" x2="450" y2="335" stroke="#333" stroke-width="1.5" marker-end="url(#arr-final)"/>

  <!-- Queue -->
  <rect x="370" y="340" width="160" height="45" rx="6" fill="#ffebee" stroke="#ef5350" stroke-width="2"/>
  <text x="450" y="368" text-anchor="middle" font-size="13" font-weight="bold" fill="#ef5350">Redis Queue</text>
  <line x1="450" y1="385" x2="450" y2="410" stroke="#333" stroke-width="1.5" marker-end="url(#arr-final)"/>

  <!-- Worker Layer -->
  <rect x="250" y="415" width="400" height="155" rx="8" fill="#f5f5f5" stroke="#999" stroke-width="1.5" stroke-dasharray="5,3"/>
  <text x="450" y="435" text-anchor="middle" font-size="12" font-weight="bold" fill="#666">Worker Process</text>

  <rect x="310" y="444" width="280" height="28" rx="4" fill="#ffebee" stroke="#ef5350" stroke-width="1"/>
  <text x="450" y="463" text-anchor="middle" font-size="10" fill="#d32f2f">Circuit Breaker + Retry w/ Backoff</text>

  <rect x="310" y="480" width="280" height="28" rx="4" fill="#e3f2fd" stroke="#42a5f5" stroke-width="1"/>
  <text x="450" y="499" text-anchor="middle" font-size="10" fill="#1976d2">Structured Logging + Metrics + Tracing</text>

  <rect x="310" y="516" width="280" height="28" rx="4" fill="#e8f5e9" stroke="#66bb6a" stroke-width="1"/>
  <text x="450" y="535" text-anchor="middle" font-size="10" fill="#388e3c">Timeout Enforcement (connect + read + total)</text>

  <line x1="450" y1="570" x2="450" y2="595" stroke="#333" stroke-width="1.5" marker-end="url(#arr-final)"/>

  <!-- Model API -->
  <rect x="370" y="600" width="160" height="45" rx="6" fill="#e8f5e9" stroke="#388e3c" stroke-width="2"/>
  <text x="450" y="628" text-anchor="middle" font-size="13" font-weight="bold" fill="#388e3c">Model API</text>

  <!-- Observability sidebar -->
  <rect x="20" y="113" width="180" height="195" rx="8" fill="#e8eaf6" stroke="#5c6bc0" stroke-width="1.5"/>
  <text x="110" y="138" text-anchor="middle" font-size="12" font-weight="bold" fill="#5c6bc0">Observability Stack</text>
  <text x="110" y="165" text-anchor="middle" font-size="10" fill="#333">Prometheus (metrics)</text>
  <text x="110" y="185" text-anchor="middle" font-size="10" fill="#333">Grafana (dashboards)</text>
  <text x="110" y="205" text-anchor="middle" font-size="10" fill="#333">Pino/Loki (logs)</text>
  <text x="110" y="225" text-anchor="middle" font-size="10" fill="#333">OTel + Tempo (traces)</text>
  <text x="110" y="255" text-anchor="middle" font-size="10" fill="#333">Alertmanager (alerts)</text>
  <text x="110" y="280" text-anchor="middle" font-size="10" fill="#333">PagerDuty (paging)</text>

  <line x1="200" y1="210" x2="245" y2="210" stroke="#5c6bc0" stroke-width="1.5" stroke-dasharray="3,3" marker-end="url(#arr-final)"/>

  <!-- Alert sidebar -->
  <rect x="700" y="113" width="180" height="195" rx="8" fill="#fce4ec" stroke="#e91e63" stroke-width="1.5"/>
  <text x="790" y="138" text-anchor="middle" font-size="12" font-weight="bold" fill="#e91e63">Alert Rules</text>
  <text x="790" y="165" text-anchor="middle" font-size="10" fill="#333">Error rate > 5%</text>
  <text x="790" y="185" text-anchor="middle" font-size="10" fill="#333">P95 latency > 5s</text>
  <text x="790" y="205" text-anchor="middle" font-size="10" fill="#333">Queue depth > 100</text>
  <text x="790" y="225" text-anchor="middle" font-size="10" fill="#333">DB pool > 80%</text>
  <text x="790" y="255" text-anchor="middle" font-size="10" fill="#333">Hourly spend > $50</text>
  <text x="790" y="280" text-anchor="middle" font-size="10" fill="#333">Circuit breaker open</text>

  <line x1="700" y1="210" x2="655" y2="210" stroke="#e91e63" stroke-width="1.5" stroke-dasharray="3,3" marker-end="url(#arr-final)"/>

  <!-- Bottom note -->
  <text x="450" y="680" text-anchor="middle" font-size="12" fill="#777" font-style="italic">Every request is logged, measured, traced, and protected at every layer.</text>
  <text x="450" y="700" text-anchor="middle" font-size="12" fill="#777" font-style="italic">This is what production-ready actually means.</text>
</svg>

---

## Final Thoughts

Here is the uncomfortable truth: building the features of your AI video platform --- the generation, the editing, the slick UI --- is maybe 30% of the work of running it in production. The other 70% is everything in this article series: the queue architecture that prevents lost jobs, the database schema that does not crumble under query load, the real-time communication that keeps users informed, and the observability and resilience that keeps the whole thing running at 2 AM when you are asleep.

Most vibe-coded projects skip all of this. They deploy a Next.js app to Vercel, wire up a model API, and call it production. It works beautifully for 5 users. Then 50 users sign up, a generation request fails silently, the developer has no logs to diagnose it, no metrics to see the pattern, no alert to notify them, and no retry mechanism to recover. The user churns. The developer never even knows.

The tools are not complicated. Pino takes 30 minutes to set up. Prometheus metrics take an afternoon. Grafana dashboards take a day. Circuit breakers, retries, and backpressure are each under 100 lines of code. The total investment to go from Level 0 to Level 2 is about a week. That week buys you the ability to sleep through the night, diagnose issues in minutes instead of hours, and retain users who would otherwise silently leave.

The gap between a deployed hobby project and a production system is not features. It is the infrastructure you build to know when things are breaking and the mechanisms you put in place to prevent one failure from cascading into a system-wide outage.

Build the observability. Build the resilience. Then sleep soundly.
