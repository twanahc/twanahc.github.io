---
layout: post
title: "From Vibe Code to Production Code: Performance Engineering for AI Video Platforms"
date: 2026-03-02
category: infra
---

You built a working AI video platform in a weekend. Cursor wrote your Express routes. Claude scaffolded your Firestore queries. GPT generated your React components. You clicked "Generate Video" and a beautiful clip appeared on screen. You deployed to Vercel, shared the link with three friends, and everything worked perfectly. Congratulations --- you have a prototype.

Now fifty users sign up on Monday. By Wednesday morning, the dashboard takes eleven seconds to load. By Thursday, your database connection pool is exhausted and the API returns 503s for twenty minutes. By Friday, a user reports that clicking "Generate" does nothing --- their request sat in memory on a server that restarted during a deploy, and the job vanished forever.

This is not a failure of the AI tools that helped you write the code. The code is correct. It returns the right result for one user in isolation. The problem is that **production is not isolation**. Production is five hundred concurrent users, a database with two million rows, network partitions, cold starts, garbage collection pauses, and a deploy happening at 3 AM while someone in Tokyo is mid-generation.

The gap between prototype and production is not more code. It is a different way of thinking. The mental shift from "does it work?" to "how does it fail?" This post is the first in a four-part series on making that shift, and we are starting with the most immediate, most measurable dimension: **performance**.

---

## Table of Contents

1. [The Mental Model Shift](#1-the-mental-model-shift)
2. [Profiling Before Optimizing](#2-profiling-before-optimizing)
3. [The N+1 Query Problem](#3-the-n1-query-problem)
4. [Connection Pool Exhaustion](#4-connection-pool-exhaustion)
5. [Caching Strategy](#5-caching-strategy)
6. [Blocking vs Non-Blocking I/O](#6-blocking-vs-non-blocking-io)
7. [The Performance Checklist](#7-the-performance-checklist)
8. [Series Roadmap](#series-roadmap)

---

## 1. The Mental Model Shift

When you vibe-code a feature, the implicit question is: **does this return the right result?** You call the database, get the data, render it, done. You test it by clicking around in the browser. If it looks right, it is right.

Production thinking asks a fundamentally different question: **does this return the right result at the 99th percentile, under 200ms, with 500 concurrent users, while a worker process is restarting?**

That question contains several concepts that most vibe-coded prototypes never encounter. Let us define them precisely, because you cannot fix what you cannot name.

### Latency, Throughput, and the Percentile Zoo

**Latency** is the time between a request entering your system and the response leaving it. When a user clicks "Load Dashboard" and sees their projects appear 340ms later, the latency of that request was 340ms. Latency is measured per-request --- every single request has its own latency value.

**Throughput** is the number of requests your system can handle per unit time. If your API server can process 200 requests per second before response times start degrading, your throughput is 200 req/s. Latency and throughput are related but not the same thing. You can have low latency at low throughput (one user gets fast responses) and high latency at high throughput (many users all get slow responses).

Now, the critical nuance: **averages lie**.

Suppose your API endpoint has an average latency of 120ms. That sounds great. But averages can hide catastrophic problems. Consider two systems:

| System | Request 1 | Request 2 | Request 3 | Request 4 | Request 5 | Average |
|--------|-----------|-----------|-----------|-----------|-----------|---------|
| A | 110ms | 115ms | 125ms | 120ms | 130ms | **120ms** |
| B | 50ms | 50ms | 50ms | 50ms | **400ms** | **120ms** |

Same average. Radically different user experience. System B has a user who waited 400ms while everyone else breezed through at 50ms.

This is why we use **percentiles** instead of averages.

**P50** (the median): 50% of requests complete faster than this. It tells you what the "typical" experience looks like.

**P95**: 95% of requests complete faster than this. Only 1 in 20 users sees something worse.

**P99**: 99% of requests complete faster than this. Only 1 in 100 users sees something worse.

The P99 is sometimes called the **tail latency**, and it is the number that matters most in production. Here is why.

### Why Tail Latency Matters More Than You Think

Imagine a user loading the project dashboard on your AI video platform. The dashboard makes 5 parallel API calls: fetch user profile, fetch projects list, fetch recent generations, fetch credit balance, fetch notification count. The page is not "loaded" until all 5 complete. The user sees the latency of the **slowest** of those 5 calls.

What is the probability that at least one of those 5 calls hits the P99 latency?

The probability that a single call does NOT hit the P99 is 0.99. The probability that NONE of 5 independent calls hit the P99 is:

$$P(\text{all fast}) = 0.99^5 = 0.951$$

So the probability that at least one call is slow is:

$$P(\text{at least one slow}) = 1 - 0.99^5 = 0.049 \approx 5\%$$

That is 1 in 20 page loads. And this is the optimistic case with only 5 parallel calls. If your dashboard makes 10 calls (which is common once you add analytics, feature flags, A/B test assignments):

$$P(\text{at least one slow}) = 1 - 0.99^{10} = 0.096 \approx 10\%$$

One in ten users sees a slow load on every single visit. And this is your P99, not some catastrophic failure --- just the normal tail of your latency distribution. If your P99 is 2 seconds instead of 200ms, 10% of your users are waiting 2 seconds every time they load the dashboard.

This is why production engineers obsess over tail latency. The user experience is determined not by your average, but by your worst case --- and with fan-out, the worst case happens far more often than intuition suggests.

<svg viewBox="0 0 800 420" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:Arial,Helvetica,sans-serif;">
  <defs>
    <marker id="arrowhead-perf" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>
  <text x="400" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">Latency Distribution: Why Averages Lie</text>
  <!-- Axes -->
  <line x1="80" y1="350" x2="750" y2="350" stroke="#333" stroke-width="2" marker-end="url(#arrowhead-perf)"/>
  <line x1="80" y1="350" x2="80" y2="50" stroke="#333" stroke-width="2" marker-end="url(#arrowhead-perf)"/>
  <text x="415" y="395" text-anchor="middle" font-size="13" fill="#333">Response Time (ms)</text>
  <text x="25" y="200" text-anchor="middle" font-size="13" fill="#333" transform="rotate(-90 25 200)">Request Count</text>
  <!-- X axis labels -->
  <text x="130" y="370" text-anchor="middle" font-size="11" fill="#555">50</text>
  <text x="230" y="370" text-anchor="middle" font-size="11" fill="#555">100</text>
  <text x="330" y="370" text-anchor="middle" font-size="11" fill="#555">200</text>
  <text x="430" y="370" text-anchor="middle" font-size="11" fill="#555">500</text>
  <text x="530" y="370" text-anchor="middle" font-size="11" fill="#555">1000</text>
  <text x="630" y="370" text-anchor="middle" font-size="11" fill="#555">2000</text>
  <text x="730" y="370" text-anchor="middle" font-size="11" fill="#555">5000</text>
  <!-- Distribution curve (long-tailed) -->
  <path d="M 100 340 Q 130 340 150 300 Q 170 200 190 120 Q 210 80 230 100 Q 260 160 300 260 Q 340 310 400 335 Q 500 345 600 348 Q 700 349 740 349" stroke="#4fc3f7" stroke-width="3" fill="none"/>
  <!-- Filled area under curve -->
  <path d="M 100 340 Q 130 340 150 300 Q 170 200 190 120 Q 210 80 230 100 Q 260 160 300 260 Q 340 310 400 335 Q 500 345 600 348 Q 700 349 740 349 L 740 350 L 100 350 Z" fill="#4fc3f720"/>
  <!-- P50 line -->
  <line x1="200" y1="60" x2="200" y2="350" stroke="#4caf50" stroke-width="2" stroke-dasharray="6,4"/>
  <text x="200" y="55" text-anchor="middle" font-size="12" font-weight="bold" fill="#4caf50">P50 = 85ms</text>
  <!-- P95 line -->
  <line x1="380" y1="60" x2="380" y2="350" stroke="#ff9800" stroke-width="2" stroke-dasharray="6,4"/>
  <text x="380" y="55" text-anchor="middle" font-size="12" font-weight="bold" fill="#ff9800">P95 = 420ms</text>
  <!-- P99 line -->
  <line x1="560" y1="60" x2="560" y2="350" stroke="#ef5350" stroke-width="2" stroke-dasharray="6,4"/>
  <text x="560" y="55" text-anchor="middle" font-size="12" font-weight="bold" fill="#ef5350">P99 = 1200ms</text>
  <!-- Tail annotation -->
  <rect x="580" y="280" width="150" height="45" rx="4" fill="#fff3e0" stroke="#ff9800"/>
  <text x="655" y="300" text-anchor="middle" font-size="11" fill="#333">The "tail" — rare but</text>
  <text x="655" y="315" text-anchor="middle" font-size="11" fill="#333">felt by real users</text>
  <line x1="580" y1="300" x2="560" y2="330" stroke="#ff9800" stroke-width="1.5"/>
</svg>

### The Production Questions

Here is a concrete checklist for shifting from prototype thinking to production thinking. Every feature, every endpoint, every database query should be evaluated against these:

| Prototype Question | Production Question |
|---|---|
| Does it return the correct data? | Does it return correct data under concurrent writes? |
| Does it load? | Does it load in under 200ms at P99? |
| Does the database query work? | Does the query work when the table has 5 million rows? |
| Does error handling work? | Does the system degrade gracefully under partial failure? |
| Can I deploy it? | Can I deploy it with zero downtime during peak traffic? |
| Does the cache work? | Does the cache invalidate correctly under race conditions? |

This is not about writing more code. It is about asking different questions of the same code. And the first step to answering those questions is measurement.

---

## 2. Profiling Before Optimizing

There is a universal law in performance engineering that is violated more often than any other: **you cannot optimize what you have not measured**. The instinct when something is slow is to guess where the bottleneck is and start optimizing. This instinct is wrong roughly 80% of the time.

Human intuition about performance bottlenecks is notoriously unreliable. You think the problem is your complex sorting algorithm. It is actually a DNS resolution happening inside a library you forgot about. You think the database is slow. It is actually your JSON serialization on a 50KB response payload. You think the external API call is the bottleneck. It is actually the 200ms you spend constructing the request body because you are doing N+1 lookups to build it.

The only way to find the real bottleneck is to **profile**.

### What Profiling Actually Measures

A **profiler** records what your program is doing over time and aggregates that information into a report. There are two main types:

**CPU profiling** answers the question: "where is my program spending its compute time?" It periodically samples the call stack (typically every 1ms) and records which function is currently executing. After thousands of samples, you get a statistical picture of where time is spent.

**Memory profiling** answers: "where is my program allocating memory, and is it releasing it?" This matters for Node.js because garbage collection pauses can cause latency spikes.

For our purposes --- a Node.js/TypeScript API server powering an AI video platform --- CPU profiling will find 90% of performance problems.

### Using the Node.js Built-In Profiler

Node.js ships with a V8 profiler that requires zero dependencies. You start your server with the `--prof` flag:

```bash
node --prof dist/server.js
```

This produces a binary log file (`isolate-0x....-v8.log`). Process it into a human-readable report:

```bash
node --prof-process isolate-0xnnnnnnnnnnnn-v8.log > profile.txt
```

The output looks something like this (abbreviated):

```
[Summary]:
   ticks  total  nonlib   name
   3245   34.2%   41.8%  JavaScript
   2876   30.3%   37.1%  C++
    892    9.4%   11.5%  GC
   2478   26.1%          Shared libraries

[JavaScript]:
   ticks  total  nonlib   name
    612    6.5%    7.9%  LazyCompile: *serializeGeneration /app/src/serializers/generation.ts:24
    543    5.7%    7.0%  LazyCompile: *buildProjectTree /app/src/services/dashboard.ts:89
    421    4.4%    5.4%  LazyCompile: *validatePermissions /app/src/middleware/auth.ts:15
    ...
```

This immediately tells a story. The `serializeGeneration` function is consuming 6.5% of total CPU time. That is a serialization function --- it should be trivial. If it is the top item in your profile, something is wrong inside it.

But text output is hard to navigate. For real investigation, you want a **flame graph**.

### Flame Graphs: The Visual Profiler

A **flame graph** is a visualization invented by Brendan Gregg that displays profiling data as a stack of horizontal bars. Each bar represents a function. The width of the bar is proportional to the time spent in that function (including time spent in functions it calls). The vertical axis represents the call stack: the bottom is the entry point, and each layer up is a function called by the layer below.

Here is how to generate one for your Node.js server. First, record a CPU profile using Chrome DevTools protocol:

```bash
# Start your server with the inspect flag
node --inspect dist/server.js

# In another terminal, generate load against the endpoint you want to profile
npx autocannon -c 50 -d 30 http://localhost:3000/api/dashboard
```

Then open `chrome://inspect` in Chrome, click your Node.js process, go to the "Performance" tab, and hit record. After 30 seconds of load, stop recording.

Alternatively, use the `0x` tool for a fully automated flame graph:

```bash
npx 0x dist/server.js
# In another terminal, generate load
npx autocannon -c 50 -d 30 http://localhost:3000/api/dashboard
# Stop the server with Ctrl+C — 0x opens the flame graph in your browser
```

### Reading a Flame Graph: A Concrete Example

Let me walk through a realistic scenario. You are profiling the `/api/dashboard` endpoint of your AI video platform. This endpoint loads the user's project list with their latest generation status --- the most-hit endpoint in the entire application.

<svg viewBox="0 0 900 480" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:monospace,sans-serif;">
  <text x="450" y="25" text-anchor="middle" font-size="15" font-weight="bold" fill="#333" font-family="Arial">Flame Graph — GET /api/dashboard (simplified)</text>
  <text x="450" y="45" text-anchor="middle" font-size="11" fill="#888" font-family="Arial">Width = time spent. Wider bars = more time consumed.</text>

  <!-- Bottom layer: HTTP handler -->
  <rect x="20" y="420" width="860" height="30" rx="2" fill="#4fc3f7"/>
  <text x="450" y="440" text-anchor="middle" font-size="11" fill="#fff" font-weight="bold">handleDashboardRequest (100% — 1420ms)</text>

  <!-- Layer 2: main operations -->
  <rect x="20" y="385" width="120" height="30" rx="2" fill="#81d4fa"/>
  <text x="80" y="405" text-anchor="middle" font-size="10" fill="#333">auth (8%)</text>

  <rect x="145" y="385" width="570" height="30" rx="2" fill="#ef5350"/>
  <text x="430" y="405" text-anchor="middle" font-size="11" fill="#fff" font-weight="bold">fetchProjectsWithGenerations (68% — 965ms)</text>

  <rect x="720" y="385" width="160" height="30" rx="2" fill="#81d4fa"/>
  <text x="800" y="405" text-anchor="middle" font-size="10" fill="#333">serialize (12%)</text>

  <!-- Layer 3: inside fetchProjectsWithGenerations -->
  <rect x="145" y="350" width="80" height="30" rx="2" fill="#e57373"/>
  <text x="185" y="370" text-anchor="middle" font-size="9" fill="#fff">getProjects</text>

  <rect x="230" y="350" width="485" height="30" rx="2" fill="#c62828"/>
  <text x="472" y="370" text-anchor="middle" font-size="11" fill="#fff" font-weight="bold">getLatestGenerationForEachScene (58% — 824ms)</text>

  <!-- Layer 4: the N+1 revealed -->
  <rect x="230" y="315" width="48" height="30" rx="2" fill="#d32f2f"/>
  <text x="254" y="335" text-anchor="middle" font-size="8" fill="#fff">query</text>
  <rect x="282" y="315" width="48" height="30" rx="2" fill="#d32f2f"/>
  <text x="306" y="335" text-anchor="middle" font-size="8" fill="#fff">query</text>
  <rect x="334" y="315" width="48" height="30" rx="2" fill="#d32f2f"/>
  <text x="358" y="335" text-anchor="middle" font-size="8" fill="#fff">query</text>
  <rect x="386" y="315" width="48" height="30" rx="2" fill="#d32f2f"/>
  <text x="410" y="335" text-anchor="middle" font-size="8" fill="#fff">query</text>
  <rect x="438" y="315" width="48" height="30" rx="2" fill="#d32f2f"/>
  <text x="462" y="335" text-anchor="middle" font-size="8" fill="#fff">query</text>
  <rect x="490" y="315" width="48" height="30" rx="2" fill="#d32f2f"/>
  <text x="514" y="335" text-anchor="middle" font-size="8" fill="#fff">query</text>
  <rect x="542" y="315" width="48" height="30" rx="2" fill="#d32f2f"/>
  <text x="566" y="335" text-anchor="middle" font-size="8" fill="#fff">query</text>
  <rect x="594" y="315" width="48" height="30" rx="2" fill="#d32f2f"/>
  <text x="618" y="335" text-anchor="middle" font-size="8" fill="#fff">query</text>
  <rect x="646" y="315" width="48" height="30" rx="2" fill="#d32f2f"/>
  <text x="670" y="335" text-anchor="middle" font-size="8" fill="#fff">query</text>

  <!-- Annotation -->
  <rect x="300" y="240" width="310" height="55" rx="6" fill="#fff3e0" stroke="#ff9800" stroke-width="2"/>
  <text x="455" y="258" text-anchor="middle" font-size="11" fill="#333" font-family="Arial" font-weight="bold">N+1 QUERY DETECTED</text>
  <text x="455" y="275" text-anchor="middle" font-size="10" fill="#555" font-family="Arial">Each "query" bar is a separate DB round-trip.</text>
  <text x="455" y="290" text-anchor="middle" font-size="10" fill="#555" font-family="Arial">47 scenes = 47 individual queries = 824ms</text>
  <line x1="455" y1="295" x2="455" y2="313" stroke="#ff9800" stroke-width="2" marker-end="url(#arrowhead-perf)"/>

  <!-- Legend -->
  <rect x="20" y="460" width="12" height="12" fill="#4fc3f7"/>
  <text x="37" y="471" font-size="10" fill="#555" font-family="Arial">HTTP handler</text>
  <rect x="150" y="460" width="12" height="12" fill="#81d4fa"/>
  <text x="167" y="471" font-size="10" fill="#555" font-family="Arial">Fast operations</text>
  <rect x="280" y="460" width="12" height="12" fill="#ef5350"/>
  <text x="297" y="471" font-size="10" fill="#555" font-family="Arial">Slow path</text>
  <rect x="400" y="460" width="12" height="12" fill="#c62828"/>
  <text x="417" y="471" font-size="10" fill="#555" font-family="Arial">Bottleneck (N+1)</text>
</svg>

The flame graph tells the story immediately. The `handleDashboardRequest` function takes 1420ms total. Of that, 68% is spent in `fetchProjectsWithGenerations`. And within that function, 58% of total time is spent in `getLatestGenerationForEachScene` --- which is just the same database query executed 47 times in sequence, one for each scene across the user's projects.

This is the "aha moment." You did not guess where the bottleneck was. You measured it. And the measurement pointed directly at an N+1 query problem. Without profiling, you might have spent days optimizing serialization (12% of time) or adding indexes to the projects table (8% of time) and barely moved the needle.

### Profiling Methodology: Brendan Gregg's Approach

Brendan Gregg, the engineer who invented flame graphs, codified a systematic methodology for performance analysis. The core idea is to work from the top down:

1. **Start with the USE method**: For every resource (CPU, memory, network, disk), check **U**tilization, **S**aturation, and **E**rrors.
2. **If CPU utilization is high**: Profile with flame graphs to find the hot functions.
3. **If CPU utilization is low but latency is high**: The bottleneck is I/O (database, network, disk). Profile off-CPU time or trace I/O operations.
4. **If saturation is high**: You have a queuing problem --- too many requests for available resources.

For a typical Node.js video platform API, the pattern is almost always #3: CPU is idle, but the process is waiting on database queries, Redis calls, or external API responses. This is why flame graphs for Node.js often reveal I/O patterns (like N+1 queries) rather than CPU-bound algorithms.

---

## 3. The N+1 Query Problem

The N+1 query problem is the single most common performance issue in web applications, and vibe-coded applications are especially vulnerable to it because the code looks clean and correct --- it just happens to be catastrophically slow at scale.

### What N+1 Means

The name describes the pattern: you execute **1** query to fetch a list of N items, then execute **N** additional queries to fetch related data for each item. Total: N+1 queries.

The problem is not that each individual query is slow. The problem is that database round-trips have a fixed overhead. Every query, regardless of how simple, incurs:

- **Network round-trip latency**: typically 1-5ms to a local database, 5-20ms to a cloud database
- **Connection acquisition**: getting a connection from the pool (more on this in section 4)
- **Query parsing and planning**: the database parses SQL, builds an execution plan
- **Result serialization**: the database serializes results, the driver deserializes them

Even if each query returns instantly, the fixed overhead per round-trip adds up. At 10ms per round-trip and 50 queries, that is 500ms of pure overhead --- half a second of the user staring at a loading spinner, caused entirely by the number of queries rather than the complexity of any single one.

### The AI Video Platform Example

Let us look at the most critical page on any AI video platform: the project dashboard. A user opens the dashboard and expects to see:

- Their list of projects
- For each project, the scene count
- For each scene, the latest generation's status and thumbnail
- The total credit cost per project

Here is how this gets vibe-coded. It looks clean. It is correct. And it is a performance disaster.

```typescript
// The N+1 anti-pattern — DO NOT USE IN PRODUCTION
interface DashboardProject {
  id: string;
  title: string;
  scenes: DashboardScene[];
  totalCost: number;
}

interface DashboardScene {
  id: string;
  order: number;
  latestGeneration: Generation | null;
}

async function getDashboardData(userId: string): Promise<DashboardProject[]> {
  // Query 1: Get all projects for the user
  const projects = await db.query<Project>(
    'SELECT * FROM projects WHERE user_id = $1 ORDER BY updated_at DESC',
    [userId]
  );

  const dashboardProjects: DashboardProject[] = [];

  for (const project of projects) {
    // Query 2..N: Get scenes for EACH project
    const scenes = await db.query<Scene>(
      'SELECT * FROM scenes WHERE project_id = $1 ORDER BY "order" ASC',
      [project.id]
    );

    const dashboardScenes: DashboardScene[] = [];

    for (const scene of scenes) {
      // Query 2N..3N: Get latest generation for EACH scene
      const [latestGen] = await db.query<Generation>(
        `SELECT * FROM generations
         WHERE scene_id = $1
         ORDER BY created_at DESC
         LIMIT 1`,
        [scene.id]
      );

      dashboardScenes.push({
        id: scene.id,
        order: scene.order,
        latestGeneration: latestGen || null,
      });
    }

    // Query 3N..4N: Get total cost for EACH project
    const [costResult] = await db.query<{ total: number }>(
      `SELECT COALESCE(SUM(g.cost), 0) as total
       FROM generations g
       JOIN scenes s ON g.scene_id = s.id
       WHERE s.project_id = $1 AND g.status = 'completed'`,
      [project.id]
    );

    dashboardProjects.push({
      id: project.id,
      title: project.title,
      scenes: dashboardScenes,
      totalCost: costResult.total,
    });
  }

  return dashboardProjects;
}
```

Let us count the queries for a realistic user with 8 projects, averaging 6 scenes each (48 scenes total):

| Query Type | Count | Time Per Query | Total Time |
|---|---|---|---|
| Fetch projects | 1 | 5ms | 5ms |
| Fetch scenes (per project) | 8 | 5ms | 40ms |
| Fetch latest generation (per scene) | 48 | 8ms | 384ms |
| Fetch cost (per project) | 8 | 12ms | 96ms |
| **Total** | **65 queries** | | **525ms** |

Sixty-five database queries for a single page load. And this is a user with just 8 projects. A power user with 30 projects and 150 scenes would trigger 211 queries. At that point, you are looking at multi-second page loads purely from query overhead.

### The Fix: Batched Queries with JOINs

The solution is to fetch all the data you need in a small, fixed number of queries --- ideally 2-3 regardless of how many projects or scenes exist. This is called **eager loading** or **batched loading**.

```typescript
// The fix: batched queries — O(1) queries regardless of data size
async function getDashboardData(userId: string): Promise<DashboardProject[]> {
  // Query 1: Get all projects with scene counts and costs in one query
  const projects = await db.query<ProjectWithStats>(`
    SELECT
      p.id,
      p.title,
      p.updated_at,
      COUNT(DISTINCT s.id) as scene_count,
      COALESCE(SUM(
        CASE WHEN g.status = 'completed' THEN g.cost ELSE 0 END
      ), 0) as total_cost
    FROM projects p
    LEFT JOIN scenes s ON s.project_id = p.id
    LEFT JOIN generations g ON g.scene_id = s.id
    WHERE p.user_id = $1
    GROUP BY p.id
    ORDER BY p.updated_at DESC
  `, [userId]);

  if (projects.length === 0) return [];

  const projectIds = projects.map(p => p.id);

  // Query 2: Get all scenes for all projects at once
  const scenes = await db.query<Scene>(`
    SELECT * FROM scenes
    WHERE project_id = ANY($1)
    ORDER BY project_id, "order" ASC
  `, [projectIds]);

  const sceneIds = scenes.map(s => s.id);

  // Query 3: Get the latest generation for all scenes at once
  // Using DISTINCT ON (PostgreSQL) to get the most recent per scene
  const latestGenerations = await db.query<Generation>(`
    SELECT DISTINCT ON (scene_id) *
    FROM generations
    WHERE scene_id = ANY($1)
    ORDER BY scene_id, created_at DESC
  `, [sceneIds]);

  // Build lookup maps for O(1) access
  const scenesByProject = new Map<string, Scene[]>();
  for (const scene of scenes) {
    const list = scenesByProject.get(scene.project_id) || [];
    list.push(scene);
    scenesByProject.set(scene.project_id, list);
  }

  const genByScene = new Map<string, Generation>();
  for (const gen of latestGenerations) {
    genByScene.set(gen.scene_id, gen);
  }

  // Assemble the dashboard data — pure in-memory operations, no I/O
  return projects.map(project => ({
    id: project.id,
    title: project.title,
    totalCost: project.total_cost,
    scenes: (scenesByProject.get(project.id) || []).map(scene => ({
      id: scene.id,
      order: scene.order,
      latestGeneration: genByScene.get(scene.id) || null,
    })),
  }));
}
```

The query count comparison:

| Metric | N+1 Version | Batched Version | Improvement |
|---|---|---|---|
| Queries (8 projects, 48 scenes) | 65 | 3 | **21.7x fewer** |
| Queries (30 projects, 150 scenes) | 211 | 3 | **70.3x fewer** |
| Latency (8 projects) | ~525ms | ~35ms | **15x faster** |
| Latency (30 projects) | ~1680ms | ~45ms | **37x faster** |
| Scales with data size? | O(N) queries | O(1) queries | Fixed cost |

The batched version issues exactly 3 queries regardless of whether the user has 1 project or 100 projects. The data volume grows (larger result sets), but the number of round-trips stays constant. This is the critical insight: **database round-trips are the expensive part, not the data transfer within a single query**.

### Detecting N+1 in Your Codebase

You do not need to wait for profiling to find N+1 patterns. They have a distinctive code signature: **a database query inside a loop**. In TypeScript, look for:

```typescript
// Red flag: query inside for...of
for (const item of items) {
  const related = await db.query('SELECT ...', [item.id]);
}

// Red flag: query inside .map() with Promise.all
const results = await Promise.all(
  items.map(item => db.query('SELECT ...', [item.id]))
);

// Red flag: query inside forEach/reduce
items.forEach(async (item) => {
  const related = await db.query('SELECT ...', [item.id]);
});
```

The `Promise.all` version is slightly better because it runs queries in parallel rather than sequentially, but it still creates N connections and N round-trips. The fix is always the same: batch the IDs and use `WHERE id = ANY($1)` or `WHERE id IN (...)`.

### N+1 with Firestore

If you are using Firestore (as many AI video platforms do for v1, as I covered in the [database schema post](/2026/01/14/database-schema-ai-video-platform)), the N+1 problem manifests as individual `getDoc()` calls inside a loop. The fix is `getAll()` or restructuring your data model to use subcollections with collection group queries:

```typescript
// N+1 in Firestore — same anti-pattern, different syntax
for (const sceneId of sceneIds) {
  const snap = await getDoc(doc(db, 'generations', sceneId));
  // Each getDoc() is a separate network round-trip to Firestore
}

// Fix: batch read with getAll (Admin SDK)
const refs = sceneIds.map(id => db.collection('generations').doc(id));
const snaps = await db.getAll(...refs);
// Single network round-trip for all documents
```

---

## 4. Connection Pool Exhaustion

The second most common production failure I see in vibe-coded platforms is **connection pool exhaustion**. It is insidious because it does not happen during development. It does not happen with 5 users. It happens at 2 AM when traffic spikes and suddenly every request fails with a timeout error that has nothing to do with your code logic.

### What Is a Connection Pool and Why Does It Exist

To understand connection pools, you need to understand what happens when your application talks to a database. Every database query requires a **connection** --- a persistent TCP socket between your application and the database server. Creating a new connection involves:

1. **TCP handshake**: 3 packets, typically 1-3ms on a local network
2. **TLS negotiation**: if encrypted (which it should be), another 2-4 round trips, adding 5-15ms
3. **Authentication**: the database verifies credentials, allocates server-side resources, sets session parameters
4. **Memory allocation**: both sides allocate memory buffers for the connection

Total time to establish a new connection: **20-50ms** on a cloud database. And the database server has a hard limit on how many simultaneous connections it can handle. A small PostgreSQL instance (like the ones on Supabase's free tier or a small Cloud SQL instance) typically supports **100-300 concurrent connections**. Each connection consumes roughly 5-10MB of server memory.

A **connection pool** solves this by maintaining a set of pre-established connections that your application reuses. Instead of:

```
Request arrives -> Create connection -> Execute query -> Close connection -> Return response
```

It becomes:

```
Request arrives -> Borrow connection from pool -> Execute query -> Return connection to pool -> Return response
```

The pool keeps connections alive and hands them out on demand. This eliminates the 20-50ms connection setup overhead on every query and prevents your application from accidentally opening thousands of connections.

### How Exhaustion Happens

Here is a typical failure sequence on an AI video platform:

<svg viewBox="0 0 900 520" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:Arial,Helvetica,sans-serif;">
  <text x="450" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">Connection Pool Exhaustion Timeline</text>

  <!-- Timeline axis -->
  <line x1="60" y1="460" x2="860" y2="460" stroke="#333" stroke-width="2" marker-end="url(#arrowhead-perf)"/>
  <text x="460" y="490" text-anchor="middle" font-size="12" fill="#555">Time</text>

  <!-- Time markers -->
  <line x1="100" y1="455" x2="100" y2="465" stroke="#333" stroke-width="2"/>
  <text x="100" y="480" text-anchor="middle" font-size="10" fill="#555">T+0s</text>
  <line x1="260" y1="455" x2="260" y2="465" stroke="#333" stroke-width="2"/>
  <text x="260" y="480" text-anchor="middle" font-size="10" fill="#555">T+5s</text>
  <line x1="420" y1="455" x2="420" y2="465" stroke="#333" stroke-width="2"/>
  <text x="420" y="480" text-anchor="middle" font-size="10" fill="#555">T+15s</text>
  <line x1="580" y1="455" x2="580" y2="465" stroke="#333" stroke-width="2"/>
  <text x="580" y="480" text-anchor="middle" font-size="10" fill="#555">T+30s</text>
  <line x1="740" y1="455" x2="740" y2="465" stroke="#333" stroke-width="2"/>
  <text x="740" y="480" text-anchor="middle" font-size="10" fill="#555">T+60s</text>

  <!-- Pool capacity line -->
  <line x1="60" y1="120" x2="860" y2="120" stroke="#ef5350" stroke-width="2" stroke-dasharray="8,4"/>
  <text x="55" y="120" text-anchor="end" font-size="11" fill="#ef5350" font-weight="bold">Pool Max (20)</text>

  <!-- Connection usage curve -->
  <path d="M 100 400 L 180 380 L 260 300 L 340 200 L 380 150 L 420 120 L 500 120 L 580 120 L 660 120 L 740 120 L 820 120" stroke="#4fc3f7" stroke-width="3" fill="none"/>
  <path d="M 100 400 L 180 380 L 260 300 L 340 200 L 380 150 L 420 120 L 500 120 L 580 120 L 660 120 L 740 120 L 820 120 L 820 460 L 100 460 Z" fill="#4fc3f720"/>

  <!-- Request queue curve (starts rising at saturation) -->
  <path d="M 420 400 L 500 350 L 580 250 L 660 180 L 740 140 L 820 120" stroke="#ff9800" stroke-width="3" fill="none" stroke-dasharray="5,3"/>

  <!-- Phase labels -->
  <rect x="100" y="55" width="140" height="35" rx="4" fill="#e8f5e9" stroke="#4caf50"/>
  <text x="170" y="77" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">Normal Traffic</text>

  <rect x="260" y="55" width="140" height="35" rx="4" fill="#fff3e0" stroke="#ff9800"/>
  <text x="330" y="77" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">Traffic Spike</text>

  <rect x="430" y="55" width="160" height="35" rx="4" fill="#ffebee" stroke="#ef5350"/>
  <text x="510" y="77" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">Pool Exhausted</text>

  <rect x="630" y="55" width="180" height="35" rx="4" fill="#f3e5f5" stroke="#9c27b0"/>
  <text x="720" y="77" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">Cascading Failure</text>

  <!-- Annotations -->
  <text x="340" y="170" font-size="10" fill="#555">Connections climbing...</text>
  <text x="510" y="105" font-size="10" fill="#ef5350" font-weight="bold">All 20 connections in use</text>
  <text x="660" y="160" font-size="10" fill="#ff9800">Requests queue up (orange)</text>
  <text x="730" y="108" font-size="10" fill="#9c27b0">Timeouts begin</text>

  <!-- Legend -->
  <line x1="100" y1="510" x2="130" y2="510" stroke="#4fc3f7" stroke-width="3"/>
  <text x="135" y="514" font-size="10" fill="#555">Active connections</text>
  <line x1="270" y1="510" x2="300" y2="510" stroke="#ff9800" stroke-width="3" stroke-dasharray="5,3"/>
  <text x="305" y="514" font-size="10" fill="#555">Queued requests</text>
  <line x1="440" y1="510" x2="470" y2="510" stroke="#ef5350" stroke-width="2" stroke-dasharray="8,4"/>
  <text x="475" y="514" font-size="10" fill="#555">Pool maximum</text>
</svg>

The cascade works like this:

1. **Traffic spikes.** Maybe a marketing email goes out. Maybe it is Monday morning and everyone opens the dashboard.
2. **Each request borrows a connection** from the pool to run its queries. If the query is fast (5ms), the connection is returned quickly. But if queries are slow (because of N+1, missing indexes, or complex JOINs), connections stay borrowed longer.
3. **The pool runs out of connections.** New requests cannot get a connection and start queuing.
4. **Queued requests start timing out.** After 30 seconds (the default timeout on most pool libraries), queued requests fail with a connection timeout error.
5. **The cascade begins.** Users see errors, retry, creating even more requests. The retry traffic makes the pool starvation worse. Health checks start failing. Load balancers mark servers as unhealthy.

The dangerous part: **the root cause is often not the traffic spike itself**. The spike just exposed a pre-existing problem --- slow queries that hold connections too long. At normal traffic, the slow queries finish before the pool fills up. Under load, they cannot.

### Configuring the Pool Correctly

Here is a production PostgreSQL connection pool configuration for a Node.js video platform, using the `pg` library:

```typescript
import { Pool, PoolConfig } from 'pg';

const poolConfig: PoolConfig = {
  // Connection details
  host: process.env.DB_HOST,
  port: parseInt(process.env.DB_PORT || '5432'),
  database: process.env.DB_NAME,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  ssl: { rejectUnauthorized: true },

  // Pool sizing
  // Rule of thumb: connections = (2 * CPU cores) + effective_spindle_count
  // For cloud databases: start with 10-20 per Node.js process
  min: 2,                    // Minimum idle connections to keep alive
  max: 20,                   // Maximum total connections

  // Timeouts — these prevent cascading failures
  connectionTimeoutMillis: 5000,   // Max wait for a connection from pool
  idleTimeoutMillis: 30000,        // Close idle connections after 30s
  query_timeout: 10000,            // Kill queries running longer than 10s

  // Connection lifecycle
  allowExitOnIdle: true,           // Let process exit if pool is idle
};

const pool = new Pool(poolConfig);

// CRITICAL: Handle pool errors to prevent process crash
pool.on('error', (err) => {
  console.error('Unexpected pool error:', err);
  // Do NOT process.exit() here — let the pool recover
});

// Monitor pool health
setInterval(() => {
  console.log({
    totalConnections: pool.totalCount,
    idleConnections: pool.idleCount,
    waitingRequests: pool.waitingCount,
  });
}, 10000);
```

Let me explain each configuration choice:

**`max: 20`** --- Why 20? The PostgreSQL wiki recommends `connections = (2 * CPU cores) + disk_spindles` per client. A typical 4-core Node.js server process gets about 10 connections. But you have to account for headroom and bursty traffic. 20 is conservative for a single server. If you have 4 servers each with `max: 20`, your database sees up to 80 connections --- well within the 100-300 limit.

**`connectionTimeoutMillis: 5000`** --- If a request cannot get a connection within 5 seconds, it fails fast rather than waiting indefinitely. This is crucial for preventing the cascade. A 5-second timeout means the user sees an error in 5 seconds and can retry, rather than hanging for 30 seconds and timing out at the load balancer level.

**`idleTimeoutMillis: 30000`** --- Connections sitting unused for 30 seconds are closed. This returns resources to the database during low-traffic periods.

**`query_timeout: 10000`** --- Any query running longer than 10 seconds is killed. This prevents runaway queries (like unindexed full-table scans) from holding a connection hostage. If a query legitimately needs more than 10 seconds, it should be offloaded to a background job.

### Pool Sizing Formula

The ideal pool size depends on your query latency profile. The formula is:

$$\text{connections needed} = \text{throughput (req/s)} \times \text{avg queries per request} \times \text{avg query duration (s)}$$

Example: your API serves 100 requests/second, each request makes 3 queries, and the average query takes 10ms (0.01s):

$$\text{connections needed} = 100 \times 3 \times 0.01 = 3$$

You only need 3 connections! But this is the average case. You need to account for variance. Queries during a busy analytics aggregation might take 50ms. A burst of 200 requests/second might hit during peak. So:

$$\text{pool max} = \text{peak throughput} \times \text{peak queries per request} \times \text{P99 query duration}$$

$$\text{pool max} = 200 \times 3 \times 0.05 = 30$$

Set `max: 30` and `min: 5` (to handle the average case without cold-starting connections).

---

## 5. Caching Strategy

Caching is the single most effective performance optimization available to any web application. A well-designed caching layer can reduce database load by 90% and cut response times from hundreds of milliseconds to single-digit milliseconds. But caching is also the source of some of the most confusing bugs in production, because **cache invalidation is genuinely hard**.

Let us build this from scratch.

### Cache Fundamentals

A **cache** is a fast storage layer that holds copies of data so that future requests can be served without accessing the slower primary storage (database). Three things happen with caches:

**Cache hit**: The requested data is in the cache. You return it immediately. Fast.

**Cache miss**: The requested data is NOT in the cache. You fetch it from the database, return it to the user, and store a copy in the cache for next time. Slow (same as no cache) but the next request will be a hit.

**Cache eviction**: The cache removes data, either because it has expired (TTL --- time to live), the cache is full and needs space (eviction policy), or the data was explicitly invalidated (you deleted it because the source changed).

The **cache hit rate** is the percentage of requests served from cache. A hit rate of 95% means only 5% of requests touch the database. The hit rate is the single most important metric for your caching layer.

### The Three-Layer Caching Architecture

For an AI video platform, you want three caching layers, each optimized for a different use case:

<svg viewBox="0 0 900 500" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:Arial,Helvetica,sans-serif;">
  <defs>
    <marker id="arrowhead-cache" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>
  <text x="450" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">Three-Layer Caching Architecture</text>

  <!-- Layer 1: In-memory LRU -->
  <rect x="80" y="60" width="740" height="100" rx="8" fill="#e8f5e9" stroke="#4caf50" stroke-width="2"/>
  <text x="450" y="85" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">Layer 1: In-Memory LRU Cache (Node.js Process)</text>
  <text x="450" y="105" text-anchor="middle" font-size="11" fill="#555">Latency: &lt;0.1ms | Capacity: 50-200MB | TTL: 30-60s</text>
  <text x="450" y="125" text-anchor="middle" font-size="11" fill="#555">Best for: Hot data accessed on every request (user sessions, feature flags, config)</text>
  <text x="450" y="145" text-anchor="middle" font-size="10" fill="#888">Shared: NO (per process) | Survives restart: NO</text>

  <!-- Arrow -->
  <line x1="450" y1="163" x2="450" y2="188" stroke="#333" stroke-width="2" marker-end="url(#arrowhead-cache)"/>
  <text x="470" y="180" font-size="10" fill="#ef5350">MISS</text>

  <!-- Layer 2: Redis -->
  <rect x="80" y="190" width="740" height="100" rx="8" fill="#e3f2fd" stroke="#4fc3f7" stroke-width="2"/>
  <text x="450" y="215" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">Layer 2: Redis Cache (Shared)</text>
  <text x="450" y="235" text-anchor="middle" font-size="11" fill="#555">Latency: 1-5ms | Capacity: 1-16GB | TTL: 60s-1hr</text>
  <text x="450" y="255" text-anchor="middle" font-size="11" fill="#555">Best for: Shared data across servers (generation status, project metadata, API responses)</text>
  <text x="450" y="275" text-anchor="middle" font-size="10" fill="#888">Shared: YES (all processes) | Survives restart: YES (with persistence)</text>

  <!-- Arrow -->
  <line x1="450" y1="293" x2="450" y2="318" stroke="#333" stroke-width="2" marker-end="url(#arrowhead-cache)"/>
  <text x="470" y="310" font-size="10" fill="#ef5350">MISS</text>

  <!-- Layer 3: CDN -->
  <rect x="80" y="320" width="740" height="100" rx="8" fill="#fff3e0" stroke="#ff9800" stroke-width="2"/>
  <text x="450" y="345" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">Layer 3: CDN Edge Cache (Cloudflare / Vercel Edge)</text>
  <text x="450" y="365" text-anchor="middle" font-size="11" fill="#555">Latency: 5-50ms (global) | Capacity: Unlimited | TTL: 5min-24hr</text>
  <text x="450" y="385" text-anchor="middle" font-size="11" fill="#555">Best for: Static and semi-static content (thumbnails, video files, public API responses)</text>
  <text x="450" y="405" text-anchor="middle" font-size="10" fill="#888">Shared: YES (global edge) | Survives restart: YES</text>

  <!-- Arrow to DB -->
  <line x1="450" y1="423" x2="450" y2="448" stroke="#333" stroke-width="2" marker-end="url(#arrowhead-cache)"/>
  <text x="470" y="440" font-size="10" fill="#ef5350">MISS</text>

  <!-- Database -->
  <rect x="300" y="450" width="300" height="40" rx="8" fill="#f5f5f5" stroke="#333" stroke-width="2"/>
  <text x="450" y="475" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">PostgreSQL / Firestore (Source of Truth)</text>
</svg>

### Caching Video Generation Status: The Concrete Example

On any AI video platform, the single most frequently polled piece of data is **generation status**. When a user starts a generation, the frontend polls for status updates. If you followed the advice in the [WebSocket architecture post](/2026/01/20/websocket-architecture-generation-status), you are using WebSockets or SSE for push-based updates. But even with WebSockets, you still need a cache layer because:

1. Users refresh the page (WebSocket reconnection needs current state)
2. The dashboard loads all recent generation statuses in bulk
3. Multiple components on the same page may request the same generation status

Here is a complete two-layer caching implementation:

```typescript
import { LRUCache } from 'lru-cache';
import Redis from 'ioredis';

// Layer 1: In-memory LRU cache
// LRU = "Least Recently Used" — when the cache is full, the item
// accessed least recently gets evicted to make room for new items.
const memoryCache = new LRUCache<string, string>({
  max: 5000,               // Max 5000 entries
  maxSize: 50 * 1024 * 1024,  // Max 50MB total
  sizeCalculation: (value) => Buffer.byteLength(value, 'utf8'),
  ttl: 30 * 1000,          // 30 second TTL
});

// Layer 2: Redis cache
const redis = new Redis({
  host: process.env.REDIS_HOST,
  port: parseInt(process.env.REDIS_PORT || '6379'),
  password: process.env.REDIS_PASSWORD,
  maxRetriesPerRequest: 3,
  retryStrategy: (times) => Math.min(times * 200, 5000),
});

interface GenerationStatus {
  id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;         // 0-100
  stage: string;            // 'analyzing_prompt' | 'generating_frames' | etc
  thumbnailUrl?: string;
  videoUrl?: string;
  error?: string;
  updatedAt: string;
}

// Read-through cache: check memory -> Redis -> database
async function getGenerationStatus(
  generationId: string
): Promise<GenerationStatus> {
  const cacheKey = `gen:status:${generationId}`;

  // Layer 1: Check in-memory cache (< 0.1ms)
  const memoryCached = memoryCache.get(cacheKey);
  if (memoryCached) {
    return JSON.parse(memoryCached);
  }

  // Layer 2: Check Redis cache (1-5ms)
  const redisCached = await redis.get(cacheKey);
  if (redisCached) {
    // Promote to memory cache for subsequent reads
    memoryCache.set(cacheKey, redisCached);
    return JSON.parse(redisCached);
  }

  // Layer 3: Database query (10-50ms)
  const status = await db.query<GenerationStatus>(
    `SELECT id, status, progress, stage, thumbnail_url,
            video_url, error, updated_at
     FROM generations WHERE id = $1`,
    [generationId]
  );

  if (!status[0]) {
    throw new Error(`Generation ${generationId} not found`);
  }

  // Write back to both caches
  const serialized = JSON.stringify(status[0]);

  // Redis: longer TTL based on generation state
  const ttl = status[0].status === 'completed' || status[0].status === 'failed'
    ? 3600   // Completed/failed: cache for 1 hour (won't change)
    : 10;    // In-progress: cache for 10 seconds (changes frequently)

  await redis.setex(cacheKey, ttl, serialized);
  memoryCache.set(cacheKey, serialized);

  return status[0];
}
```

The TTL strategy here is intentional and worth explaining. A generation that is still `queued` or `processing` changes its status frequently --- every few seconds as it progresses through stages. So we cache it with a short 10-second TTL. A generation that is `completed` or `failed` will never change again, so we cache it for an hour. This gives us high hit rates on the data that is read most (users refreshing to see their completed videos) while keeping in-progress data fresh.

### Cache Invalidation: The Hard Part

There is a famous quote in computer science, attributed to Phil Karlton: "There are only two hard things in Computer Science: cache invalidation and naming things."

Cache invalidation is hard because you have to answer the question: **when the source data changes, how do all the caches learn about it?** There are three strategies:

**1. TTL-based expiration (simplest, least consistent)**

You set a TTL on every cached entry and accept that data may be stale for up to that duration. This is what we used above. It works well when:
- Brief staleness is acceptable (generation status being 10s stale is fine)
- The data changes infrequently (user profile cached for 5 minutes)
- The cost of serving stale data is low (showing the old thumbnail for 30s is not harmful)

**2. Write-through invalidation (more complex, more consistent)**

When you write to the database, you also invalidate (delete or update) the cache entry. This ensures the cache is always current, but requires discipline: every write path must know about the cache.

```typescript
// Write-through: update database AND invalidate cache atomically
async function updateGenerationStatus(
  generationId: string,
  update: Partial<GenerationStatus>
): Promise<void> {
  // Update the source of truth
  await db.query(
    `UPDATE generations
     SET status = COALESCE($2, status),
         progress = COALESCE($3, progress),
         stage = COALESCE($4, stage),
         updated_at = NOW()
     WHERE id = $1`,
    [generationId, update.status, update.progress, update.stage]
  );

  // Invalidate all cache layers
  const cacheKey = `gen:status:${generationId}`;
  memoryCache.delete(cacheKey);
  await redis.del(cacheKey);

  // Also invalidate the project dashboard cache if status changed
  // to completed or failed (the dashboard shows latest status)
  if (update.status === 'completed' || update.status === 'failed') {
    const [gen] = await db.query<{ scene_id: string }>(
      'SELECT scene_id FROM generations WHERE id = $1',
      [generationId]
    );
    const [scene] = await db.query<{ project_id: string }>(
      'SELECT project_id FROM scenes WHERE id = $1',
      [gen.scene_id]
    );
    await redis.del(`dashboard:project:${scene.project_id}`);
  }
}
```

Notice the cascade: updating a generation status requires invalidating not just the generation cache, but also the project dashboard cache (because the dashboard shows the latest generation status). This is where cache invalidation gets complex --- you need to track all the cache keys that depend on a given piece of data.

**3. Event-driven invalidation (most robust, most complex)**

Instead of the write path directly invalidating caches, you publish a change event and let cache subscribers handle invalidation. This decouples the write path from the caching layer. If you are already using the [BullMQ job queue architecture](/2026/01/17/redis-bullmq-job-queue-video) from an earlier post, you can leverage Redis Pub/Sub:

```typescript
// Publisher: the worker that processes generations
async function onGenerationComplete(
  generationId: string,
  videoUrl: string
): Promise<void> {
  await db.query(
    `UPDATE generations SET status = 'completed',
     video_url = $2, updated_at = NOW() WHERE id = $1`,
    [generationId, videoUrl]
  );

  // Publish event — cache layer subscribes
  await redis.publish('generation:status_changed', JSON.stringify({
    generationId,
    newStatus: 'completed',
  }));
}

// Subscriber: runs in every API server process
const subscriber = redis.duplicate();
subscriber.subscribe('generation:status_changed');
subscriber.on('message', (channel, message) => {
  const { generationId } = JSON.parse(message);
  const cacheKey = `gen:status:${generationId}`;
  memoryCache.delete(cacheKey);
  // Note: Redis cache is shared, so only one server needs to delete it
  redis.del(cacheKey);
});
```

### Measuring Cache Effectiveness

You need to track your cache hit rates. Without measurement, you do not know if your caching is working or just consuming memory.

```typescript
// Simple cache metrics
const cacheMetrics = {
  memoryHits: 0,
  redisHits: 0,
  misses: 0,

  get hitRate(): number {
    const total = this.memoryHits + this.redisHits + this.misses;
    return total === 0 ? 0 : (this.memoryHits + this.redisHits) / total;
  },

  get memoryHitRate(): number {
    const total = this.memoryHits + this.redisHits + this.misses;
    return total === 0 ? 0 : this.memoryHits / total;
  },

  log(): void {
    console.log({
      totalHitRate: `${(this.hitRate * 100).toFixed(1)}%`,
      memoryHitRate: `${(this.memoryHitRate * 100).toFixed(1)}%`,
      memoryHits: this.memoryHits,
      redisHits: this.redisHits,
      dbMisses: this.misses,
    });
  },
};
```

Target hit rates for a healthy AI video platform:

| Cache Layer | Target Hit Rate | If Below Target |
|---|---|---|
| In-memory (L1) | > 60% | TTL too short, or cache too small |
| Redis (L2) | > 85% (L1 + L2 combined) | TTL too short, or invalidation too aggressive |
| CDN (L3) | > 90% for static assets | Check Cache-Control headers, vary headers |
| Overall | > 90% | Review what queries are still hitting the database |

---

## 6. Blocking vs Non-Blocking I/O

This section matters enormously for Node.js applications and is often the least understood aspect of Node.js performance. If you do not understand the event loop, you will write code that looks asynchronous but silently blocks your entire server.

### The Event Loop: What It Actually Is

Node.js runs your JavaScript in a **single thread**. There is one thread executing your code, one call stack, and one instruction pointer. This sounds like a terrible idea for a server that needs to handle hundreds of concurrent requests --- and it would be, if your code actually ran synchronously.

The trick is that most of what a web server does is **waiting**. Waiting for a database response. Waiting for a file to be read from disk. Waiting for an HTTP response from an external API. While one request is waiting for a database query to return, the event loop can process other requests.

Here is the mental model. Think of the event loop as a single chef in a kitchen:

1. The chef picks up a ticket (an incoming request)
2. The chef starts the request: parses the body, validates the input
3. The chef needs to wait for something (a database query), so they put the ticket aside and note "when the database responds, continue here"
4. The chef picks up the next ticket and starts working on it
5. The database response arrives --- the chef finishes that earlier ticket when they have a moment

This works beautifully as long as the chef never gets stuck on a single task. The chef can juggle hundreds of tickets because most of each ticket is waiting time. But if one ticket requires the chef to do 5 minutes of uninterrupted chopping (a CPU-intensive synchronous operation), **every other ticket stops**. The entire kitchen freezes while the chef chops.

<svg viewBox="0 0 900 400" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:Arial,Helvetica,sans-serif;">
  <text x="450" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">Node.js Event Loop: Blocking vs Non-Blocking</text>

  <!-- Non-blocking side -->
  <rect x="30" y="55" width="400" height="30" rx="4" fill="#e8f5e9" stroke="#4caf50" stroke-width="2"/>
  <text x="230" y="75" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Non-Blocking (Correct)</text>

  <!-- Timeline bars: interleaved requests -->
  <rect x="50" y="100" width="60" height="20" rx="3" fill="#4fc3f7"/>
  <text x="80" y="114" text-anchor="middle" font-size="9" fill="#fff">Req A</text>
  <rect x="115" y="100" width="60" height="20" rx="3" fill="#8bc34a"/>
  <text x="145" y="114" text-anchor="middle" font-size="9" fill="#fff">Req B</text>
  <rect x="180" y="100" width="60" height="20" rx="3" fill="#ff9800"/>
  <text x="210" y="114" text-anchor="middle" font-size="9" fill="#fff">Req C</text>
  <rect x="245" y="100" width="50" height="20" rx="3" fill="#4fc3f7"/>
  <text x="270" y="114" text-anchor="middle" font-size="9" fill="#fff">A cb</text>
  <rect x="300" y="100" width="50" height="20" rx="3" fill="#8bc34a"/>
  <text x="325" y="114" text-anchor="middle" font-size="9" fill="#fff">B cb</text>
  <rect x="355" y="100" width="50" height="20" rx="3" fill="#ff9800"/>
  <text x="380" y="114" text-anchor="middle" font-size="9" fill="#fff">C cb</text>

  <text x="50" y="145" font-size="10" fill="#555">All 3 requests complete in ~200ms total.</text>
  <text x="50" y="160" font-size="10" fill="#555">While A waits for DB, B and C start processing.</text>

  <!-- Blocking side -->
  <rect x="470" y="55" width="400" height="30" rx="4" fill="#ffebee" stroke="#ef5350" stroke-width="2"/>
  <text x="670" y="75" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Blocking (Broken)</text>

  <!-- Timeline bars: sequential, one blocking -->
  <rect x="490" y="100" width="280" height="20" rx="3" fill="#ef5350"/>
  <text x="630" y="114" text-anchor="middle" font-size="9" fill="#fff">Req A — CPU-bound image resize (180ms)</text>
  <rect x="775" y="100" width="80" height="20" rx="3" fill="#8bc34a"/>
  <text x="815" y="114" text-anchor="middle" font-size="9" fill="#fff">Req B</text>

  <text x="490" y="145" font-size="10" fill="#555">Req B waits 180ms before it even starts.</text>
  <text x="490" y="160" font-size="10" fill="#555">Req C is still queued. Health checks fail.</text>

  <!-- Throughput comparison -->
  <rect x="30" y="190" width="400" height="80" rx="6" fill="#f8f9fa" stroke="#ddd"/>
  <text x="230" y="215" text-anchor="middle" font-size="12" font-weight="bold" fill="#4caf50">2,000 requests/second</text>
  <text x="230" y="235" text-anchor="middle" font-size="11" fill="#555">P99 latency: 120ms</text>
  <text x="230" y="255" text-anchor="middle" font-size="11" fill="#555">Event loop lag: &lt;5ms</text>

  <rect x="470" y="190" width="400" height="80" rx="6" fill="#f8f9fa" stroke="#ddd"/>
  <text x="670" y="215" text-anchor="middle" font-size="12" font-weight="bold" fill="#ef5350">5.5 requests/second</text>
  <text x="670" y="235" text-anchor="middle" font-size="11" fill="#555">P99 latency: 9,100ms</text>
  <text x="670" y="255" text-anchor="middle" font-size="11" fill="#555">Event loop lag: 180ms+</text>

  <!-- Arrow showing 360x difference -->
  <rect x="350" y="300" width="200" height="40" rx="6" fill="#fff3e0" stroke="#ff9800" stroke-width="2"/>
  <text x="450" y="325" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">360x throughput difference</text>
</svg>

### The Blocking Catastrophe

Here is a realistic example. Your AI video platform needs to generate thumbnails for uploaded reference images. The vibe-coded version uses a synchronous image processing library:

```typescript
import sharp from 'sharp';
import express from 'express';

const app = express();

// THIS BLOCKS THE EVENT LOOP — DO NOT DO THIS
app.post('/api/upload-reference', async (req, res) => {
  const imageBuffer = req.body.image; // Let's say 5MB JPEG

  // sharp's resize is async, BUT...
  // This pipeline does synchronous CPU work between the async steps:
  // 1. Decode JPEG (CPU-bound: ~50ms for 5MB)
  // 2. Resize (CPU-bound: ~30ms)
  // 3. Re-encode as WebP (CPU-bound: ~100ms)
  const thumbnail = await sharp(imageBuffer)
    .resize(256, 256, { fit: 'cover' })
    .webp({ quality: 80 })
    .toBuffer();

  // The await above looks async, but sharp does heavy CPU work
  // on the main thread for decode/encode operations.

  await uploadToStorage(thumbnail);
  res.json({ success: true });
});
```

`sharp` is actually one of the better libraries because it offloads to native code via libvips. But many image processing operations still consume significant CPU time. During those ~180ms of CPU-bound work, the event loop is **completely blocked**. No other request can be processed. No WebSocket messages can be sent. No health check can be responded to.

At 50 concurrent uploads, you have 50 requests each needing ~180ms of CPU time. With one thread, they execute sequentially. The 50th request waits \(50 \times 180\text{ms} = 9\text{ seconds}\) before it even starts processing.

Here is what this looks like to your users:

| Scenario | Without blocking work | With blocking image resize |
|---|---|---|
| P50 response time | 45ms | 2,400ms |
| P99 response time | 120ms | 9,100ms |
| Requests/second capacity | 2,000 | 5.5 |
| WebSocket status updates | Real-time | Delayed 2-9s |
| Health check response | Instant | May timeout (server looks dead) |

The server throughput drops from 2,000 req/s to 5.5 req/s because each request monopolizes the single thread for 180ms.

### The Fix: Offload to Worker Threads or a Job Queue

There are two solutions, and the right one depends on whether the blocking work needs to happen synchronously (the user is waiting for the result) or can happen asynchronously (the user does not need the result immediately).

**Option 1: Worker Threads (user needs the result immediately)**

Node.js Worker Threads let you run CPU-intensive operations on separate threads, keeping the event loop free:

```typescript
// thumbnail-worker.ts — runs in a separate thread
import { parentPort, workerData } from 'worker_threads';
import sharp from 'sharp';

async function processImage(): Promise<void> {
  const { imageBuffer, width, height, quality } = workerData;

  const thumbnail = await sharp(Buffer.from(imageBuffer))
    .resize(width, height, { fit: 'cover' })
    .webp({ quality })
    .toBuffer();

  parentPort?.postMessage(thumbnail);
}

processImage();
```

```typescript
// In your Express handler — the event loop stays free
import { Worker } from 'worker_threads';
import path from 'path';

// Pre-create a pool of workers to avoid spawning overhead
import { Pool } from 'tinypool';

const workerPool = new Pool({
  filename: path.resolve(__dirname, 'thumbnail-worker.js'),
  minThreads: 2,
  maxThreads: 4,   // Match your CPU core count
  idleTimeout: 60000,
});

app.post('/api/upload-reference', async (req, res) => {
  const imageBuffer = req.body.image;

  // This runs on a SEPARATE thread — event loop stays free
  const thumbnail = await workerPool.run({
    imageBuffer: Buffer.from(imageBuffer),
    width: 256,
    height: 256,
    quality: 80,
  });

  await uploadToStorage(Buffer.from(thumbnail));
  res.json({ success: true });
});
```

Now the event loop never blocks. The CPU work happens on worker threads, and the main thread can continue processing other requests. Four worker threads on a 4-core machine can process 4 images in parallel while the main thread handles hundreds of API requests.

**Option 2: Job Queue (user does not need the result immediately)**

If the thumbnail does not need to be ready before the API responds (common for background processing), offload to your job queue. This is exactly the pattern described in the [Redis + BullMQ post](/2026/01/17/redis-bullmq-job-queue-video):

```typescript
import { Queue } from 'bullmq';

const thumbnailQueue = new Queue('thumbnails', {
  connection: { host: process.env.REDIS_HOST },
});

app.post('/api/upload-reference', async (req, res) => {
  // Store the original image immediately
  const imageKey = await uploadToStorage(req.body.image);

  // Enqueue thumbnail generation — returns instantly
  await thumbnailQueue.add('generate-thumbnail', {
    imageKey,
    width: 256,
    height: 256,
    quality: 80,
  });

  // Respond immediately — thumbnail will be ready later
  res.json({
    success: true,
    imageKey,
    thumbnailStatus: 'processing',
  });
});
```

A separate worker process picks up the job and generates the thumbnail without affecting the API server's event loop at all. The frontend can poll for the thumbnail or receive a WebSocket notification when it is ready.

### How to Detect Event Loop Blocking

You cannot always tell from reading the code whether something blocks the event loop. Some operations that look async actually perform synchronous work. The definitive test is to measure event loop lag:

```typescript
// Measure event loop lag — if this number spikes, something is blocking
let lastCheck = process.hrtime.bigint();

setInterval(() => {
  const now = process.hrtime.bigint();
  const elapsed = Number(now - lastCheck) / 1_000_000; // Convert to ms
  const lag = elapsed - 100; // Expected interval is 100ms

  if (lag > 50) {
    console.warn(`Event loop lag: ${lag.toFixed(1)}ms — something is blocking!`);
  }

  lastCheck = now;
}, 100);
```

If the interval fires every 100ms and the measured elapsed time is 250ms, you have 150ms of event loop blocking. This metric should be in your monitoring dashboard. Any lag consistently above 50ms warrants investigation.

Common hidden blockers in Node.js video platforms:

| Operation | Why It Blocks | Fix |
|---|---|---|
| `JSON.parse()` on large payloads (>1MB) | Synchronous CPU work | Stream parsing with `json-stream` |
| `JSON.stringify()` on large objects | Synchronous CPU work | Stream serialization or paginate response |
| `fs.readFileSync()` | Synchronous disk I/O | Use `fs.promises.readFile()` |
| `crypto.pbkdf2Sync()` | Synchronous CPU work | Use `crypto.pbkdf2()` (async version) |
| RegExp on long strings | Synchronous CPU work | Limit input length, use streaming regex |
| `Array.sort()` on large arrays (>100K) | Synchronous CPU work | Worker thread or database-side sort |
| Template rendering (EJS, Handlebars) | Synchronous CPU work | Pre-compile templates, or use React SSR with streaming |

---

## 7. The Performance Checklist

Here is a practical checklist you can run through on any codebase, whether it was vibe-coded in a weekend or carefully architected over months. Each item takes 5-30 minutes to check and can reveal problems before your users find them.

### Database Layer

```
[ ] Run EXPLAIN ANALYZE on your 5 most-hit queries
    — Are any doing sequential scans on large tables?
    — Is the estimated cost >> actual cost? (statistics may be stale: ANALYZE)

[ ] Check for N+1 queries
    — Search codebase for database calls inside loops
    — Enable query logging: SET log_min_duration_statement = 0;
    — Count queries per endpoint using middleware logging

[ ] Verify indexes exist for all WHERE and JOIN columns
    — Especially: user_id, project_id, scene_id, status, created_at
    — Check composite indexes for multi-column WHERE clauses
    — Use: SELECT * FROM pg_stat_user_indexes WHERE idx_scan = 0;
      (shows indexes that exist but are never used — remove them)

[ ] Check query timeout configuration
    — Is statement_timeout set? (It should be: 10-30s max)
    — Are long-running analytics queries isolated to read replicas?
```

### Connection Management

```
[ ] Pool sizing matches your workload
    — Current pool size: SELECT count(*) FROM pg_stat_activity;
    — Max connections: SHOW max_connections;
    — Ratio of pool max to max_connections across all servers < 80%

[ ] Pool monitoring is in place
    — Are you logging: total, idle, waiting counts every 10s?
    — Do you have alerts for waitingCount > 0 sustained for 30s?

[ ] Connection timeouts are configured
    — connectionTimeoutMillis < 10s (fail fast)
    — query_timeout is set (prevent runaway queries)
    — idleTimeoutMillis is set (release unused connections)
```

### Caching

```
[ ] Hot read paths have caching
    — Dashboard/listing endpoints
    — User session/profile data
    — Generation status (the most polled data)
    — Configuration and feature flags

[ ] Cache hit rates are being measured
    — Target: > 90% overall
    — Check for cache keys with 0% hit rate (wasted memory)
    — Check for keys with very short TTL that get evicted before reuse

[ ] Cache invalidation paths are complete
    — Every database write that affects cached data must invalidate
    — Test: update a record, immediately read via API — do you see the update?
    — Watch for stale data complaints from users (cache invalidation bug)

[ ] CDN is configured for static assets
    — Video files, thumbnails, and images served via CDN
    — Cache-Control headers set correctly
    — Vary headers configured properly for authenticated endpoints
```

### Async Processing

```
[ ] No CPU-intensive work on the event loop
    — Image processing: offloaded to worker threads or job queue
    — Video encoding: always a background job
    — Large JSON serialization: check with event loop lag monitoring
    — PDF generation, file compression: worker threads

[ ] Event loop lag monitoring is active
    — Baseline lag: < 10ms
    — Alert threshold: > 50ms sustained
    — Investigate: > 100ms (something is definitely blocking)

[ ] Background jobs use a proper queue (not setTimeout)
    — Jobs survive server restarts
    — Failed jobs retry with backoff
    — Dead letter queue captures permanently failed jobs
```

### Payload and Compression

```
[ ] API responses are gzip/brotli compressed
    — Enable compression middleware: app.use(compression())
    — Verify with: curl -H "Accept-Encoding: gzip" -v your-api-url
    — Typical reduction: 60-80% for JSON responses

[ ] Response payloads are right-sized
    — Are you returning fields the client does not use? (over-fetching)
    — Are listing endpoints paginated? (never return unbounded lists)
    — Are large blobs (base64 images, logs) returned inline?

[ ] Request validation rejects oversized payloads early
    — Set body parser limits: express.json({ limit: '1mb' })
    — File uploads have size limits enforced at the middleware level
    — Reject before processing, not after
```

### Monitoring and Alerting

```
[ ] Response time percentiles are tracked (P50, P95, P99)
    — Per-endpoint, not just global averages
    — Historical trends to detect gradual degradation

[ ] Error rates are tracked per endpoint
    — 4xx vs 5xx separation
    — Alert on 5xx rate > 1% sustained for 5 minutes

[ ] Database metrics are visible
    — Query count per endpoint
    — Slow query log (> 100ms)
    — Connection pool utilization

[ ] Synthetic monitoring for critical paths
    — Automated test that loads the dashboard every 60s
    — Automated test that starts a generation every 5 minutes
    — Alerts if latency exceeds threshold or test fails
```

---

## 8. Series Roadmap

This post is the first in a four-part series on taking vibe-coded projects to production. Here is where we are going:

| Part | Title | Focus |
|---|---|---|
| **1 (this post)** | Performance Engineering | Profiling, N+1, connection pools, caching, event loop |
| 2 | Error Handling and Resilience | Circuit breakers, retry strategies, graceful degradation, chaos testing |
| 3 | Observability and Debugging | Structured logging, distributed tracing, metrics pipelines, debugging production issues |
| 4 | Deployment and Operations | Zero-downtime deploys, feature flags, rollback strategies, incident response |

Each post builds on the previous one. Performance engineering (this post) gives you the tools to measure and optimize your system. Error handling gives you the patterns to survive when things go wrong. Observability gives you the visibility to understand what is happening in production. And deployment gives you the operational practices to ship changes safely.

The common thread across all four posts: **production is not a feature you add. It is a mindset you adopt.** The AI tools that helped you build your prototype are extraordinary --- they let you move at a speed that was unimaginable five years ago. But they optimize for correctness in isolation, not resilience under load. That last mile is still on you.

The good news: the performance patterns in this post are not hard to implement once you know to look for them. A single afternoon of profiling and query optimization can take a vibe-coded prototype from "collapses at 50 users" to "handles 5,000 comfortably." The hard part was knowing where to look. Now you do.

---

*This is Part 1 of the "Vibe Code to Production Code" series. The next post covers error handling and resilience patterns for AI video platforms --- circuit breakers, retry strategies, graceful degradation, and how to build systems that fail gracefully instead of catastrophically.*
