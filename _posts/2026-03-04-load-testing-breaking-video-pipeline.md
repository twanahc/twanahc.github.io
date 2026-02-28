---
layout: post
title: "Load Testing to Destruction: Breaking Your AI Video Pipeline Before Users Do"
date: 2026-03-04
category: infra
---

# Load Testing to Destruction: Breaking Your AI Video Pipeline Before Users Do

There is a specific kind of confidence that comes from watching your application handle 1,000 concurrent users without breaking a sweat --- and a specific kind of horror that comes from watching it crumble under 50. Performance is not intuition. It is measurement. And the only way to measure is to break things deliberately, in a controlled environment, before your users do it for you in production.

This is the third article in a four-part series on taking vibe-coded projects to production. In the [first post](/2026/01/17/redis-bullmq-job-queue-video.html), we built a production job queue with BullMQ and Redis. In the [second post](/2026/01/20/websocket-architecture-generation-status.html), we wired up real-time status updates over WebSockets. Now we stress-test the entire system until something breaks --- and then we fix it, and break it again, until we can state with numerical precision exactly how much load our platform can handle.

Every number in this post comes from running tests against a real AI video platform. If you are building one of these systems, the failure modes I describe here are not hypothetical. They are the exact failures you will encounter, roughly in the order you will encounter them.

---

## Table of Contents

1. [Why Load Testing Matters](#1-why-load-testing-matters)
2. [k6: The Tool](#2-k6-the-tool)
3. [Designing Test Scenarios for an AI Video Platform](#3-designing-test-scenarios-for-an-ai-video-platform)
4. [The First Load Test: Watching Everything Break](#4-the-first-load-test-watching-everything-break)
5. [Diagnosing the Failures](#5-diagnosing-the-failures)
6. [Fixing and Retesting](#6-fixing-and-retesting)
7. [Advanced Patterns](#7-advanced-patterns)
8. [Interpreting Results and Setting SLOs](#8-interpreting-results-and-setting-slos)
9. [The Load Testing Checklist](#9-the-load-testing-checklist)

---

## 1. Why Load Testing Matters

Your video platform works with 1 user. It works with 10 users. It works with 50. Probably. You have never actually checked, because in development you are the only user, and in your staging environment you have at most a handful of QA testers clicking around at human speed.

But you have no idea what happens at 200 concurrent users, because you have never tried. Load testing is the practice of finding out.

Let me define the four types of load testing, because they test fundamentally different failure modes:

**Load testing** applies a realistic, sustained level of traffic to your system and measures how it performs. The question it answers is: "Can our system handle the expected peak traffic?" You ramp up to a target number of concurrent users, hold that level for a duration, and measure response times and error rates.

**Stress testing** pushes beyond expected load to find the breaking point. The question it answers is: "At what point does our system fail, and how does it fail?" You keep increasing load until errors appear, response times become unacceptable, or the system stops responding entirely.

**Soak testing** (also called endurance testing) applies moderate load for a long duration --- typically 4 to 24 hours. The question it answers is: "Does our system have slow-degradation bugs like memory leaks, connection pool exhaustion, or disk space depletion?" These bugs are invisible in short tests because the resource drain is small per-request but cumulative over time.

**Spike testing** applies sudden, extreme bursts of traffic. The question it answers is: "What happens when load goes from zero to maximum instantly?" Think: you get featured on Hacker News, your product launches, or a viral TikTok sends 10,000 users to your site in two minutes.

Each type catches a different class of bug. If you only run one, you miss three categories of failure. We will run all four in this post.

### Why AI Video Platforms Are Especially Vulnerable

An AI video platform has characteristics that make it uniquely susceptible to load-related failures:

1. **Long-running operations**: A single video generation ties up resources for 30 to 300 seconds. In a standard web application, a request completes in under 200ms. That three-orders-of-magnitude difference means that even modest concurrency creates enormous resource pressure.

2. **Mixed workload profiles**: Your API handles everything from sub-10ms health checks to 5-minute video generations. A single "average response time" metric is meaningless when your distribution is this bimodal.

3. **Expensive external API calls**: Each generation hits a third-party video model API (Veo, Kling, Runway, Sora) that costs real money. You cannot just fire off thousands of real requests in a load test.

4. **State-heavy operations**: Users are constantly polling for generation status, fetching project lists, loading scene editors. These read-heavy operations hit your database and cache layers hard.

5. **WebSocket connections**: Real-time status updates mean persistent connections. Each connection consumes server memory and file descriptors. At scale, this is a resource limit most people have never thought about.

<svg viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:system-ui,sans-serif">
  <defs>
    <marker id="arr-lt1" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>

  <text x="400" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">AI Video Platform: Load Pressure Points</text>

  <!-- Client layer -->
  <rect x="30" y="60" width="140" height="80" rx="8" fill="#e3f2fd" stroke="#4fc3f7" stroke-width="2"/>
  <text x="100" y="90" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Clients</text>
  <text x="100" y="108" text-anchor="middle" font-size="10" fill="#666">HTTP + WebSocket</text>
  <text x="100" y="122" text-anchor="middle" font-size="10" fill="#666">200 concurrent</text>

  <!-- API Server -->
  <rect x="220" y="60" width="150" height="80" rx="8" fill="#fff3e0" stroke="#ff9800" stroke-width="2"/>
  <text x="295" y="85" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">API Server</text>
  <text x="295" y="102" text-anchor="middle" font-size="10" fill="#e65100">Event loop blocking</text>
  <text x="295" y="116" text-anchor="middle" font-size="10" fill="#e65100">Memory pressure</text>
  <text x="295" y="130" text-anchor="middle" font-size="10" fill="#e65100">File descriptor limits</text>

  <!-- Database -->
  <rect x="420" y="60" width="150" height="80" rx="8" fill="#ffebee" stroke="#ef5350" stroke-width="2"/>
  <text x="495" y="85" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">PostgreSQL</text>
  <text x="495" y="102" text-anchor="middle" font-size="10" fill="#c62828">Connection pool exhausted</text>
  <text x="495" y="116" text-anchor="middle" font-size="10" fill="#c62828">Missing indexes</text>
  <text x="495" y="130" text-anchor="middle" font-size="10" fill="#c62828">N+1 queries</text>

  <!-- Redis -->
  <rect x="620" y="60" width="150" height="80" rx="8" fill="#ffebee" stroke="#ef5350" stroke-width="2"/>
  <text x="695" y="85" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Redis</text>
  <text x="695" y="102" text-anchor="middle" font-size="10" fill="#c62828">Connection limits</text>
  <text x="695" y="116" text-anchor="middle" font-size="10" fill="#c62828">Memory ceiling</text>
  <text x="695" y="130" text-anchor="middle" font-size="10" fill="#c62828">Pub/Sub fan-out</text>

  <!-- Arrows -->
  <line x1="170" y1="100" x2="215" y2="100" stroke="#333" stroke-width="2" marker-end="url(#arr-lt1)"/>
  <line x1="370" y1="90" x2="415" y2="90" stroke="#333" stroke-width="2" marker-end="url(#arr-lt1)"/>
  <line x1="370" y1="110" x2="415" y2="110" stroke="#333" stroke-width="2" marker-end="url(#arr-lt1)"/>
  <line x1="570" y1="100" x2="615" y2="100" stroke="#333" stroke-width="2" marker-end="url(#arr-lt1)"/>

  <!-- Pressure indicators -->
  <rect x="30" y="180" width="740" height="50" rx="6" fill="#f5f5f5" stroke="#ddd"/>
  <text x="50" y="205" font-size="12" fill="#333" font-weight="bold">Pressure gradient (200 concurrent users):</text>
  <rect x="50" y="215" width="60" height="6" rx="3" fill="#4caf50"/>
  <text x="120" y="221" font-size="10" fill="#666">Low</text>
  <rect x="160" y="215" width="120" height="6" rx="3" fill="#ff9800"/>
  <text x="290" y="221" font-size="10" fill="#666">Medium</text>
  <rect x="330" y="215" width="200" height="6" rx="3" fill="#f44336"/>
  <text x="540" y="221" font-size="10" fill="#666">High</text>
  <rect x="570" y="215" width="180" height="6" rx="3" fill="#b71c1c"/>
  <text x="760" y="221" font-size="10" fill="#666">Critical</text>

  <!-- Workers -->
  <rect x="220" y="270" width="150" height="80" rx="8" fill="#e8f5e9" stroke="#8bc34a" stroke-width="2"/>
  <text x="295" y="295" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Job Workers</text>
  <text x="295" y="312" text-anchor="middle" font-size="10" fill="#666">BullMQ processors</text>
  <text x="295" y="326" text-anchor="middle" font-size="10" fill="#666">Concurrency-limited</text>
  <text x="295" y="340" text-anchor="middle" font-size="10" fill="#666">Memory-heavy</text>

  <!-- External APIs -->
  <rect x="420" y="270" width="150" height="80" rx="8" fill="#f3e5f5" stroke="#ab47bc" stroke-width="2"/>
  <text x="495" y="295" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Video Model APIs</text>
  <text x="495" y="312" text-anchor="middle" font-size="10" fill="#666">Veo, Kling, Runway</text>
  <text x="495" y="326" text-anchor="middle" font-size="10" fill="#666">Rate limited</text>
  <text x="495" y="340" text-anchor="middle" font-size="10" fill="#666">$0.05-0.50 per call</text>

  <!-- Storage -->
  <rect x="620" y="270" width="150" height="80" rx="8" fill="#e8eaf6" stroke="#5c6bc0" stroke-width="2"/>
  <text x="695" y="295" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Object Storage</text>
  <text x="695" y="312" text-anchor="middle" font-size="10" fill="#666">R2 / S3</text>
  <text x="695" y="326" text-anchor="middle" font-size="10" fill="#666">Presigned URL gen</text>
  <text x="695" y="340" text-anchor="middle" font-size="10" fill="#666">Bandwidth limits</text>

  <line x1="295" y1="140" x2="295" y2="265" stroke="#333" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arr-lt1)"/>
  <line x1="370" y1="310" x2="415" y2="310" stroke="#333" stroke-width="1.5" marker-end="url(#arr-lt1)"/>
  <line x1="570" y1="310" x2="615" y2="310" stroke="#333" stroke-width="1.5" marker-end="url(#arr-lt1)"/>
</svg>

---

## 2. k6: The Tool

There are many load testing tools --- JMeter, Locust, Gatling, Artillery, Vegeta. I use **k6**, and let me explain why.

k6 is an open-source load testing tool built by Grafana Labs (formerly Load Impact). Tests are written in JavaScript, which means you get a real programming language with functions, loops, conditionals, imports, and the ability to model complex user behavior. It runs from the command line, produces clean metrics output, and is designed from the ground up for scriptability.

The alternatives and why they fall short for our use case:

| Tool | Language | Why Not |
|------|----------|---------|
| **JMeter** | XML / GUI | GUI-based, XML config files, poor version control, slow for complex scenarios |
| **Locust** | Python | Good choice, but Python's GIL limits single-machine throughput |
| **Gatling** | Scala | Excellent performance, but Scala is a steep learning curve |
| **Artillery** | YAML + JS | Good for simple tests, but limited scenario modeling |
| **k6** | JavaScript | CLI-native, scriptable, fast (Go runtime), excellent metrics |

k6 scripts run inside a Go runtime, not Node.js, which means they generate enormous load from a single machine. A laptop can simulate thousands of virtual users. The JavaScript you write is not full Node.js --- there is no `fs`, no `require()` for npm packages --- but it covers everything you need for HTTP requests, WebSocket connections, checks, thresholds, and custom metrics.

### Installation

```bash
# macOS
brew install k6

# Linux (Debian/Ubuntu)
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg \
  --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D68
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" \
  | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update && sudo apt-get install k6

# Windows
choco install k6

# Docker (works everywhere)
docker run --rm -i grafana/k6 run - < script.js
```

### Your First k6 Script

Let us start with the simplest possible test: hitting a health endpoint.

```javascript
// health-check.js
import http from 'k6/http';
import { check, sleep } from 'k6';

// Options configure the load profile
export const options = {
  // VUs = Virtual Users = concurrent simulated users
  vus: 10,
  // Duration = how long the test runs
  duration: '30s',
};

// The default function runs once per iteration for each VU
export default function () {
  const response = http.get('http://localhost:3000/api/health');

  // Checks are assertions — they do not stop the test on failure,
  // they record pass/fail rates
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 200ms': (r) => r.timings.duration < 200,
    'body contains status ok': (r) => r.json().status === 'ok',
  });

  // Think time: real users do not send requests in a tight loop.
  // This 1-second sleep means each VU sends ~1 request per second.
  sleep(1);
}
```

Run it:

```bash
k6 run health-check.js
```

Let me walk through the key concepts in this script.

**Virtual Users (VUs)** are simulated concurrent users. Each VU runs the `default` function in a loop for the specified duration. If you have 10 VUs and each iteration takes ~1 second (because of the `sleep(1)`), you get roughly 10 requests per second. The VU count is the primary knob for controlling load.

**Iterations** are individual executions of the `default` function. If a VU completes its function and the duration has not expired, it loops and runs again. The total iterations depend on VU count, duration, and how long each iteration takes.

**Think time** is the `sleep()` call. Without it, each VU would fire requests as fast as the server can respond --- which is not realistic. Real users read the screen, click around, type things. Typical think times range from 1 to 10 seconds depending on the action.

**Checks** are soft assertions. Unlike assertions in a unit test, a failed check does not abort the test. Instead, k6 tracks the pass rate. At the end of the test, you see something like `checks: 98.5% (985/1000)`, telling you that 1.5% of checks failed. This is how you detect degradation without binary pass/fail.

### Understanding k6 Output

When a k6 test completes, you get output like this:

```
     checks.........................: 100.00% ✓ 300  ✗ 0
     data_received..................: 45 kB   1.5 kB/s
     data_sent......................: 27 kB   900 B/s
     http_req_blocked...............: avg=1.2ms  min=0.5ms  med=0.8ms  max=15ms   p(90)=1.5ms  p(95)=2.1ms
     http_req_connecting............: avg=0.6ms  min=0.2ms  med=0.4ms  max=12ms   p(90)=0.8ms  p(95)=1.3ms
     http_req_duration..............: avg=12ms   min=3ms    med=8ms    max=145ms  p(90)=25ms   p(95)=42ms
     http_req_failed................: 0.00%   ✓ 0    ✗ 300
     http_req_receiving.............: avg=0.1ms  min=0ms    med=0.1ms  max=1ms    p(90)=0.2ms  p(95)=0.3ms
     http_req_sending...............: avg=0.05ms min=0ms    med=0.04ms max=0.5ms  p(90)=0.1ms  p(95)=0.1ms
     http_req_tls_handshaking.......: avg=0ms    min=0ms    med=0ms    max=0ms    p(90)=0ms    p(95)=0ms
     http_req_waiting...............: avg=11.8ms min=2.8ms  med=7.8ms  max=144ms  p(90)=24ms   p(95)=41ms
     http_reqs......................: 300     10/s
     iteration_duration.............: avg=1.01s  min=1.003s med=1.008s max=1.15s  p(90)=1.025s p(95)=1.042s
     iterations.....................: 300     10/s
     vus............................: 10      min=10  max=10
     vus_max........................: 10      min=10  max=10
```

The metrics that matter most:

- **`http_req_duration`**: Total time from sending the request to receiving the full response. The `p(95)` value is what you should focus on --- it means 95% of requests completed faster than that number. If P95 is 42ms, that means only 5% of requests took longer than 42ms.
- **`http_req_failed`**: Percentage of requests that returned non-2xx status codes. This is your error rate.
- **`http_reqs`**: Total request count and requests per second.
- **`checks`**: Pass rate for all your check assertions.

Why P95 and not average? Because averages lie. If 99 requests take 10ms and one request takes 10,000ms, the average is 109ms, which sounds fine. But the P95 is 10ms (also fine) and the P99 is 10,000ms (terrible). Percentiles tell you what the tail of your distribution looks like, and that tail is where user frustration lives.

---

## 3. Designing Test Scenarios for an AI Video Platform

A health check test is useful for validating your setup. But real load testing requires modeling what users actually do. For an AI video platform, the user journey is not "hit one endpoint repeatedly." It is a multi-step workflow with dependencies between steps.

### The User Journey

Here is the typical flow for a user on an AI video generation platform:

1. **Login** --- Authenticate, receive a session token
2. **Dashboard** --- Fetch list of projects
3. **Open Project** --- Fetch project details and list of scenes with generation status
4. **Trigger Generation** --- Submit a video generation request
5. **Poll Status** --- Check generation progress (or receive WebSocket updates)
6. **View Result** --- Fetch the completed video URL and metadata

Not every user completes the full journey. Some login and browse. Some trigger generations. Some just check on previously submitted generations. A realistic load test models this distribution.

### The Complete k6 Test Script

```javascript
// video-platform-load-test.js
import http from 'k6/http';
import { check, group, sleep, fail } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// ─── Custom Metrics ───────────────────────────────────────────
// k6 has built-in metrics, but we want to track business-specific
// measurements separately from generic HTTP timing.
const loginDuration = new Trend('login_duration', true);
const dashboardDuration = new Trend('dashboard_duration', true);
const generationTriggerDuration = new Trend('generation_trigger_duration', true);
const statusPollDuration = new Trend('status_poll_duration', true);
const errorRate = new Rate('business_error_rate');

// ─── Test Configuration ───────────────────────────────────────
export const options = {
  scenarios: {
    // Scenario 1: Ramp-up load test
    // Start at 0 users, ramp to target, hold, ramp down.
    ramp_up: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 },   // Ramp to 50 users over 2 minutes
        { duration: '5m', target: 50 },   // Hold at 50 for 5 minutes
        { duration: '2m', target: 100 },  // Ramp to 100 users
        { duration: '5m', target: 100 },  // Hold at 100
        { duration: '2m', target: 200 },  // Ramp to 200 users
        { duration: '5m', target: 200 },  // Hold at 200
        { duration: '3m', target: 0 },    // Ramp down to 0
      ],
      gracefulRampDown: '30s',
    },
  },

  // Thresholds define pass/fail criteria for the entire test.
  // If any threshold is breached, k6 exits with a non-zero code.
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1500'],  // P95 < 500ms, P99 < 1.5s
    http_req_failed: ['rate<0.01'],                    // Error rate < 1%
    login_duration: ['p(95)<300'],                     // Login P95 < 300ms
    dashboard_duration: ['p(95)<400'],                 // Dashboard P95 < 400ms
    business_error_rate: ['rate<0.05'],                // Business errors < 5%
  },
};

// ─── Test Data ────────────────────────────────────────────────
// Pre-generated test users. In a real setup, you would create these
// in a setup() function or load from a CSV file.
const TEST_USERS = [
  { email: 'loadtest-001@test.com', password: 'test-password-001' },
  { email: 'loadtest-002@test.com', password: 'test-password-002' },
  { email: 'loadtest-003@test.com', password: 'test-password-003' },
  // ... in practice, generate 500+ test users
];

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3000';

// ─── Helper Functions ─────────────────────────────────────────

function getTestUser() {
  // Each VU gets a deterministic user based on its ID.
  // __VU is the 1-based VU number assigned by k6.
  const index = (__VU - 1) % TEST_USERS.length;
  return TEST_USERS[index];
}

function apiRequest(method, path, body, token) {
  const url = `${BASE_URL}${path}`;
  const headers = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  let response;
  if (method === 'GET') {
    response = http.get(url, { headers });
  } else if (method === 'POST') {
    response = http.post(url, JSON.stringify(body), { headers });
  }

  return response;
}

// ─── Main Test Function ───────────────────────────────────────

export default function () {
  const user = getTestUser();
  let token = null;

  // ─── Step 1: Login ──────────────────────────────────────────
  group('01_login', function () {
    const response = apiRequest('POST', '/api/auth/login', {
      email: user.email,
      password: user.password,
    });

    loginDuration.add(response.timings.duration);

    const loginOk = check(response, {
      'login: status 200': (r) => r.status === 200,
      'login: has token': (r) => {
        try {
          return r.json().token !== undefined;
        } catch (e) {
          return false;
        }
      },
    });

    if (!loginOk) {
      errorRate.add(1);
      fail('Login failed — cannot continue user journey');
    }

    errorRate.add(0);
    token = response.json().token;
  });

  // Think time: user sees dashboard loading
  sleep(Math.random() * 2 + 1); // 1-3 seconds

  // ─── Step 2: Dashboard (List Projects) ──────────────────────
  let projects = [];
  group('02_dashboard', function () {
    const response = apiRequest('GET', '/api/projects', null, token);

    dashboardDuration.add(response.timings.duration);

    const dashOk = check(response, {
      'dashboard: status 200': (r) => r.status === 200,
      'dashboard: has projects array': (r) => {
        try {
          return Array.isArray(r.json().projects);
        } catch (e) {
          return false;
        }
      },
    });

    if (dashOk) {
      projects = response.json().projects;
    }
    errorRate.add(dashOk ? 0 : 1);
  });

  // Think time: user scans the project list
  sleep(Math.random() * 3 + 2); // 2-5 seconds

  // ─── Step 3: Open a Project (List Scenes) ───────────────────
  if (projects.length > 0) {
    const project = projects[Math.floor(Math.random() * projects.length)];

    group('03_open_project', function () {
      const response = apiRequest(
        'GET',
        `/api/projects/${project.id}/scenes`,
        null,
        token
      );

      check(response, {
        'project: status 200': (r) => r.status === 200,
        'project: has scenes': (r) => {
          try {
            return Array.isArray(r.json().scenes);
          } catch (e) {
            return false;
          }
        },
      });
    });

    // Think time: user reviews scenes
    sleep(Math.random() * 3 + 2); // 2-5 seconds

    // ─── Step 4: Trigger Generation ───────────────────────────
    // Only 30% of users trigger a generation per session.
    // The rest are browsing/reviewing.
    if (Math.random() < 0.3) {
      let generationId = null;

      group('04_trigger_generation', function () {
        const response = apiRequest(
          'POST',
          `/api/projects/${project.id}/generate`,
          {
            sceneId: project.scenes?.[0]?.id || 'scene-1',
            prompt: 'A serene mountain landscape at golden hour',
            model: 'veo-2',
            resolution: '1080p',
            duration: 5,
          },
          token
        );

        generationTriggerDuration.add(response.timings.duration);

        const genOk = check(response, {
          'generate: status 201 or 202': (r) =>
            r.status === 201 || r.status === 202,
          'generate: has generation id': (r) => {
            try {
              return r.json().generationId !== undefined;
            } catch (e) {
              return false;
            }
          },
        });

        if (genOk) {
          generationId = response.json().generationId;
        }
        errorRate.add(genOk ? 0 : 1);
      });

      // ─── Step 5: Poll Generation Status ──────────────────────
      if (generationId) {
        group('05_poll_status', function () {
          // Poll up to 10 times, then give up.
          // In production, this would be a WebSocket, but polling
          // is easier to model in a load test and creates more
          // server pressure (which is what we want to measure).
          for (let i = 0; i < 10; i++) {
            const response = apiRequest(
              'GET',
              `/api/generations/${generationId}/status`,
              null,
              token
            );

            statusPollDuration.add(response.timings.duration);

            check(response, {
              'poll: status 200': (r) => r.status === 200,
              'poll: has status field': (r) => {
                try {
                  return r.json().status !== undefined;
                } catch (e) {
                  return false;
                }
              },
            });

            // Check if generation completed or failed
            try {
              const status = response.json().status;
              if (status === 'completed' || status === 'failed') {
                break;
              }
            } catch (e) {
              // Response was not valid JSON — continue polling
            }

            // Poll interval: 3 seconds
            sleep(3);
          }
        });
      }
    }
  }

  // Final think time before next iteration
  sleep(Math.random() * 5 + 3); // 3-8 seconds
}

// ─── Setup Function ───────────────────────────────────────────
// Runs once before the test starts. Use this to seed test data.
export function setup() {
  console.log(`Starting load test against ${BASE_URL}`);
  console.log(`Test users: ${TEST_USERS.length}`);

  // Verify the API is reachable
  const health = http.get(`${BASE_URL}/api/health`);
  if (health.status !== 200) {
    fail(`API is not reachable: ${health.status}`);
  }

  return { startTime: Date.now() };
}

// ─── Teardown Function ────────────────────────────────────────
// Runs once after the test ends. Use for cleanup.
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`Test completed in ${duration.toFixed(1)} seconds`);
}
```

Let me unpack the design decisions in this script.

**Scenarios with `ramping-vus`**: Instead of a flat load, we ramp up in stages. This lets us see exactly where performance degrades. At 50 VUs everything might be fine; at 100 we might see latency creep; at 200 errors might appear. The staircase pattern makes it trivial to correlate metrics with load level.

**Groups**: Each step of the user journey is wrapped in a `group()`. This gives us per-step timing in the k6 output, so we can see that login takes 50ms but the dashboard takes 800ms. Without groups, all HTTP requests are lumped together.

**Think times with randomness**: `sleep(Math.random() * 3 + 2)` sleeps between 2 and 5 seconds. This models realistic user behavior and prevents all VUs from hitting the server in synchronized waves (which would be a spike test, not a load test). The randomness is important --- synchronized VUs create artificial thundering-herd effects.

**Probabilistic branching**: Only 30% of users trigger a generation. This matches real usage patterns where most sessions are browse-heavy. If you have 200 VUs and 30% trigger generations, that is 60 concurrent generation attempts --- which is plenty to stress the system.

**Custom metrics**: The built-in `http_req_duration` tells you total request timing, but not which endpoint is slow. Custom `Trend` metrics let you track per-endpoint performance independently.

---

## 4. The First Load Test: Watching Everything Break

Let me walk through what actually happens when you run this test for the first time against a platform that has never been load tested.

The first two minutes feel fine. You are ramping from 0 to 50 VUs. Response times are fast. Error rate is zero. You start feeling confident.

Then you hit 50 VUs and hold. For about a minute, everything looks stable. Then the P95 response time starts drifting upward. Not dramatically --- from 80ms to 120ms. You think: "That is fine, still well under the threshold."

You ramp to 100 VUs. Within 30 seconds, the dashboard endpoint (which lists all projects with their latest generation status) starts taking over a second. Your generation status polling endpoint, which was at 15ms, is now at 400ms. And then the errors start.

First, a trickle: 0.5% error rate. Then 2%. Then 8%. By the time you reach 200 VUs, the system is returning 503 errors on 40% of requests, the P95 is over 5 seconds for endpoints that should be sub-100ms, and your database CPU is pegged at 100%.

Here is what the metrics look like at each stage:

### Performance Degradation Table

| Metric | 10 VUs | 50 VUs | 100 VUs | 200 VUs |
|--------|--------|--------|---------|---------|
| **Dashboard P50** | 45ms | 82ms | 340ms | 2,800ms |
| **Dashboard P95** | 78ms | 190ms | 1,200ms | 8,400ms |
| **Dashboard P99** | 120ms | 350ms | 2,300ms | timeout |
| **Status Poll P50** | 8ms | 15ms | 85ms | 1,200ms |
| **Status Poll P95** | 15ms | 42ms | 400ms | 5,300ms |
| **Generation Trigger P50** | 120ms | 180ms | 580ms | 3,100ms |
| **Generation Trigger P95** | 200ms | 320ms | 1,400ms | timeout |
| **Error Rate** | 0% | 0.1% | 4.2% | 38.6% |
| **Req/s (actual)** | 9.8 | 42 | 68 | 31 |
| **DB Connections (active)** | 5 | 18 | 48 | 50 (maxed) |
| **DB CPU** | 3% | 15% | 62% | 100% |
| **API Server Memory** | 180MB | 240MB | 420MB | 890MB |

Notice something critical in the "Req/s" row. At 100 VUs, you are getting 68 requests per second. At 200 VUs, you are getting only 31. That is not a typo. You *doubled* the load and the throughput *halved*. This is the signature of a system past its saturation point --- adding more concurrent users does not increase throughput, it *decreases* it because the system is spending all its time managing contention rather than doing useful work.

<svg viewBox="0 0 800 450" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:system-ui,sans-serif">
  <text x="400" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">Response Time vs. Concurrent Users (Dashboard Endpoint)</text>

  <!-- Axes -->
  <line x1="80" y1="380" x2="750" y2="380" stroke="#333" stroke-width="2"/>
  <line x1="80" y1="380" x2="80" y2="50" stroke="#333" stroke-width="2"/>

  <!-- X axis labels -->
  <text x="80" y="405" text-anchor="middle" font-size="11" fill="#666">0</text>
  <text x="230" y="405" text-anchor="middle" font-size="11" fill="#666">50</text>
  <text x="400" y="405" text-anchor="middle" font-size="11" fill="#666">100</text>
  <text x="570" y="405" text-anchor="middle" font-size="11" fill="#666">150</text>
  <text x="740" y="405" text-anchor="middle" font-size="11" fill="#666">200</text>
  <text x="415" y="435" text-anchor="middle" font-size="13" fill="#333">Concurrent Users (VUs)</text>

  <!-- Y axis labels -->
  <text x="70" y="384" text-anchor="end" font-size="11" fill="#666">0</text>
  <text x="70" y="314" text-anchor="end" font-size="11" fill="#666">500ms</text>
  <text x="70" y="244" text-anchor="end" font-size="11" fill="#666">1s</text>
  <text x="70" y="174" text-anchor="end" font-size="11" fill="#666">2s</text>
  <text x="70" y="104" text-anchor="end" font-size="11" fill="#666">5s</text>
  <text x="70" y="64" text-anchor="end" font-size="11" fill="#666">8s+</text>
  <text x="25" y="220" text-anchor="middle" font-size="13" fill="#333" transform="rotate(-90 25 220)">Response Time</text>

  <!-- Grid lines -->
  <line x1="80" y1="310" x2="750" y2="310" stroke="#eee" stroke-width="1" stroke-dasharray="4,4"/>
  <line x1="80" y1="240" x2="750" y2="240" stroke="#eee" stroke-width="1" stroke-dasharray="4,4"/>
  <line x1="80" y1="170" x2="750" y2="170" stroke="#eee" stroke-width="1" stroke-dasharray="4,4"/>
  <line x1="80" y1="100" x2="750" y2="100" stroke="#eee" stroke-width="1" stroke-dasharray="4,4"/>

  <!-- P50 line (blue) -->
  <polyline points="80,375 230,368 400,335 570,260 740,100" fill="none" stroke="#4fc3f7" stroke-width="3"/>
  <circle cx="80" cy="375" r="4" fill="#4fc3f7"/>
  <circle cx="230" cy="368" r="4" fill="#4fc3f7"/>
  <circle cx="400" cy="335" r="4" fill="#4fc3f7"/>
  <circle cx="570" cy="260" r="4" fill="#4fc3f7"/>
  <circle cx="740" cy="100" r="4" fill="#4fc3f7"/>

  <!-- P95 line (orange) -->
  <polyline points="80,372 230,352 400,280 570,165 740,60" fill="none" stroke="#ff9800" stroke-width="3"/>
  <circle cx="80" cy="372" r="4" fill="#ff9800"/>
  <circle cx="230" cy="352" r="4" fill="#ff9800"/>
  <circle cx="400" cy="280" r="4" fill="#ff9800"/>
  <circle cx="570" cy="165" r="4" fill="#ff9800"/>
  <circle cx="740" cy="60" r="4" fill="#ff9800"/>

  <!-- SLO line -->
  <line x1="80" y1="310" x2="750" y2="310" stroke="#f44336" stroke-width="2" stroke-dasharray="8,4"/>
  <text x="755" y="314" font-size="11" fill="#f44336">500ms SLO</text>

  <!-- Danger zone shading -->
  <rect x="400" y="50" width="350" height="330" fill="#f44336" opacity="0.05"/>
  <text x="575" y="78" text-anchor="middle" font-size="12" fill="#f44336" opacity="0.6">Danger Zone</text>

  <!-- Legend -->
  <rect x="530" y="415" width="220" height="30" rx="4" fill="#f9f9f9" stroke="#ddd"/>
  <line x1="545" y1="432" x2="575" y2="432" stroke="#4fc3f7" stroke-width="3"/>
  <text x="580" y="436" font-size="11" fill="#666">P50</text>
  <line x1="620" y1="432" x2="650" y2="432" stroke="#ff9800" stroke-width="3"/>
  <text x="655" y="436" font-size="11" fill="#666">P95</text>
  <line x1="695" y1="432" x2="725" y2="432" stroke="#f44336" stroke-width="2" stroke-dasharray="4,2"/>
  <text x="730" y="436" font-size="11" fill="#666">SLO</text>
</svg>

This chart tells a clear story. Up to about 75 VUs, response times are within SLO. Between 75 and 100, the P95 crosses the 500ms threshold. Beyond 100, the system is in freefall.

### Little's Law: Why This Happens

There is an elegant mathematical relationship governing queueing systems called **Little's Law**. It states:

$$
L = \lambda W
$$

Where:
- \(L\) is the average number of items in the system (concurrent requests being processed)
- \(\lambda\) is the average arrival rate (requests per second)
- \(W\) is the average time each item spends in the system (response time)

This is not an approximation. It is an exact identity that holds for any stable queueing system in steady state, regardless of the arrival distribution or service time distribution. John Little proved this in 1961, and it applies to everything from grocery store checkout lines to API servers.

Let us apply it to our situation. At 50 VUs with 1-second think time, each VU sends roughly 1 request per second, so \(\lambda \approx 50\) req/s. If average response time \(W = 82\text{ms} = 0.082\text{s}\), then:

$$
L = 50 \times 0.082 = 4.1
$$

So on average, 4.1 requests are being processed concurrently. Our database connection pool of 50 can handle that easily.

At 200 VUs, even with longer response times reducing the actual request rate to about 31 req/s, and average response time \(W = 2.8\text{s}\):

$$
L = 31 \times 2.8 = 86.8
$$

Now 87 requests are being processed concurrently, but our connection pool only has 50 connections. The excess requests queue up, wait for a connection, and that waiting time adds to \(W\), which increases \(L\), which causes more queuing. This is a positive feedback loop --- the system spirals into failure.

---

## 5. Diagnosing the Failures

The load test told us *that* the system fails. Now we need to figure out *why*. Let me walk through the usual suspects for an AI video platform, in roughly the order you will encounter them.

### 5.1 Database Connection Pool Exhaustion

**Symptom**: At high concurrency, requests start timing out waiting for a database connection. Error logs show messages like `TimeoutError: Connection acquisition timed out` or `error: remaining connection slots are reserved for superuser connections`.

**What is happening**: Every database query needs a connection. Opening a new connection to PostgreSQL takes 20-50ms (TCP handshake, authentication, SSL negotiation). To avoid that overhead on every query, applications use a **connection pool** --- a set of pre-opened connections that are checked out when needed and returned when done.

Most connection pool libraries default to a pool size of 10 to 20. PostgreSQL defaults to a maximum of 100 connections total. When all connections in the pool are checked out and a new request needs one, it waits. If the pool is saturated for long enough, the wait times out and the request fails.

**How to diagnose**:

```sql
-- Check current connections by state
SELECT state, count(*)
FROM pg_stat_activity
GROUP BY state;

-- Check max connections setting
SHOW max_connections;

-- See what queries are running right now
SELECT pid, state, query, now() - query_start AS duration
FROM pg_stat_activity
WHERE state = 'active'
ORDER BY duration DESC;
```

If `pg_stat_activity` shows all connections as `active` and your pool size matches `max_connections`, you have found the bottleneck.

### 5.2 Missing Database Indexes

**Symptom**: Specific queries that should be fast (looking up a project by ID, filtering generations by status) take hundreds of milliseconds under load. CPU usage on the database server spikes.

**What is happening**: Without an index, PostgreSQL performs a **sequential scan** --- it reads every single row in the table to find matches. With 1,000 rows, this takes a few milliseconds and you never notice. With 100,000 rows and 50 concurrent queries doing sequential scans, you have 50 processes each reading the entire table, fighting for disk I/O and CPU.

An **index** is a separate data structure (usually a B-tree) that maps column values to row locations. Looking up a row by an indexed column is \(O(\log n)\) instead of \(O(n)\). For 100,000 rows, that is roughly 17 comparisons instead of 100,000.

**How to diagnose**:

```sql
-- Enable slow query logging (in postgresql.conf or per-session)
SET log_min_duration_statement = 100;  -- Log queries over 100ms

-- Find the slowest queries using pg_stat_statements extension
SELECT query, calls, mean_exec_time, total_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 20;

-- For a specific slow query, check if it uses an index
EXPLAIN ANALYZE
SELECT g.*, s.prompt, s.order_index
FROM generations g
JOIN scenes s ON g.scene_id = s.id
WHERE s.project_id = 'proj_abc123'
AND g.status = 'completed'
ORDER BY g.created_at DESC;
```

If `EXPLAIN ANALYZE` shows `Seq Scan` instead of `Index Scan` for a table with more than a few thousand rows, you need an index.

### 5.3 N+1 Queries

**Symptom**: The dashboard endpoint (list projects) is disproportionately slow. Database logs show bursts of nearly identical queries executing in rapid succession.

**What is happening**: An **N+1 query** is a pattern where you run 1 query to fetch a list of N items, then run N additional queries to fetch related data for each item. For a dashboard that shows 20 projects, each with their latest generation status:

```typescript
// THE N+1 ANTI-PATTERN
// 1 query to fetch projects
const projects = await db.query(
  'SELECT * FROM projects WHERE user_id = $1',
  [userId]
);

// Then N queries — one per project — to fetch latest generation
for (const project of projects) {
  const latest = await db.query(
    `SELECT status, created_at FROM generations
     WHERE project_id = $1
     ORDER BY created_at DESC LIMIT 1`,
    [project.id]
  );
  project.latestGeneration = latest.rows[0];
}
```

With 20 projects, that is 21 queries. Each query takes ~5ms, so it takes ~105ms. Tolerable for a single user. But at 100 concurrent users, you are running 2,100 queries per dashboard load. Each query needs a database connection, and suddenly your 50-connection pool is handling 2,100 near-simultaneous queries for just one endpoint.

**How to diagnose**: Enable query logging and look for repetitive patterns. A single endpoint that generates dozens of nearly identical queries is an N+1.

### 5.4 Redis Connection Limits

**Symptom**: BullMQ starts failing to enqueue or process jobs. Job status updates stop flowing. Error messages reference `MaxRetriesPerRequestError` or `ECONNREFUSED`.

**What is happening**: Redis has a default maximum connection limit (`maxclients`, default 10,000). Each BullMQ worker, each Pub/Sub subscriber, and each cache client opens its own connection. Under load, you can exhaust this limit, especially if connections are not being properly released.

**How to diagnose**:

```bash
# Check current connection count
redis-cli info clients

# Check max clients setting
redis-cli config get maxclients

# See connected clients with details
redis-cli client list | wc -l
```

### 5.5 Event Loop Blocking

**Symptom**: All endpoints slow down simultaneously, even simple ones like the health check. Node.js CPU usage is 100% on a single core.

**What is happening**: Node.js runs JavaScript on a single thread --- the **event loop**. The event loop processes one task at a time: handle an incoming request, parse JSON, execute a callback, send a response. As long as each task completes quickly (under a few milliseconds), the event loop cycles fast enough to handle thousands of concurrent connections.

But if any task takes a long time --- a synchronous JSON parse of a large payload, a CPU-intensive computation, an accidental synchronous file read --- it **blocks the event loop**. While that task is running, no other request can be processed. Every other connection just waits.

**How to diagnose**:

```bash
# Run your application with the built-in Node.js profiler
node --prof app.js

# After the test, process the profiler output
node --prof-process isolate-0x*.log > processed-profile.txt
```

Look for functions that consume a disproportionate amount of CPU time. Common culprits:
- `JSON.parse()` on large payloads (e.g., serialized generation results with embedded base64 video data)
- Synchronous crypto operations (`crypto.pbkdf2Sync` instead of `crypto.pbkdf2`)
- Large array operations (sorting thousands of items in-memory)
- Template rendering with complex data

### 5.6 Memory Leaks Under Sustained Load

**Symptom**: The system works fine initially but degrades over hours. Memory usage grows steadily. Eventually, the process crashes with an out-of-memory error or becomes so slow that it is effectively dead (due to excessive garbage collection).

**What is happening**: A **memory leak** occurs when objects are allocated but never freed. In garbage-collected languages like JavaScript, this typically means objects are unintentionally kept reachable --- stored in a growing array, referenced in a closure, cached in a Map that never evicts entries.

Common sources in a video platform:
- WebSocket connections that are closed but not removed from an in-memory tracking Set
- Event listeners registered on each request but never removed
- In-memory caching without size limits or TTL
- BullMQ job data retained after completion

**How to diagnose**:

```bash
# Take a heap snapshot using the V8 inspector
# Start your app with the inspector enabled:
node --inspect app.js

# Connect Chrome DevTools to chrome://inspect
# Take a heap snapshot, run load for a few minutes, take another snapshot
# Compare the two snapshots to see what grew
```

You can also add a simple memory monitoring endpoint:

```typescript
// Add to your API server
app.get('/debug/memory', (req, res) => {
  const used = process.memoryUsage();
  res.json({
    rss: `${Math.round(used.rss / 1024 / 1024)}MB`,
    heapTotal: `${Math.round(used.heapTotal / 1024 / 1024)}MB`,
    heapUsed: `${Math.round(used.heapUsed / 1024 / 1024)}MB`,
    external: `${Math.round(used.external / 1024 / 1024)}MB`,
  });
});
```

<svg viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:system-ui,sans-serif">
  <text x="400" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">Diagnostic Decision Tree: Identifying Performance Bottlenecks</text>

  <!-- Root -->
  <rect x="280" y="50" width="240" height="40" rx="6" fill="#4fc3f7" stroke="#0288d1" stroke-width="2"/>
  <text x="400" y="75" text-anchor="middle" font-size="12" font-weight="bold" fill="#fff">Requests slow under load?</text>

  <!-- Level 1 branches -->
  <line x1="340" y1="90" x2="180" y2="120" stroke="#333" stroke-width="1.5"/>
  <line x1="460" y1="90" x2="620" y2="120" stroke="#333" stroke-width="1.5"/>

  <text x="245" y="108" font-size="10" fill="#666">All endpoints</text>
  <text x="540" y="108" font-size="10" fill="#666">Specific endpoints</text>

  <!-- All endpoints slow -->
  <rect x="70" y="120" width="220" height="40" rx="6" fill="#ff9800" stroke="#e65100" stroke-width="1.5"/>
  <text x="180" y="145" text-anchor="middle" font-size="11" font-weight="bold" fill="#fff">Check: Event loop blocked?</text>

  <line x1="130" y1="160" x2="80" y2="190" stroke="#333" stroke-width="1"/>
  <line x1="230" y1="160" x2="270" y2="190" stroke="#333" stroke-width="1"/>

  <text x="90" y="182" font-size="9" fill="#666">Yes</text>
  <text x="250" y="182" font-size="9" fill="#666">No</text>

  <rect x="10" y="190" width="150" height="35" rx="4" fill="#f44336"/>
  <text x="85" y="212" text-anchor="middle" font-size="10" fill="#fff">node --prof, find CPU hog</text>

  <rect x="195" y="190" width="160" height="35" rx="4" fill="#ff9800" stroke="#e65100" stroke-width="1"/>
  <text x="275" y="212" text-anchor="middle" font-size="10" fill="#fff">Check: DB pool exhausted?</text>

  <line x1="235" y1="225" x2="195" y2="255" stroke="#333" stroke-width="1"/>
  <line x1="315" y1="225" x2="345" y2="255" stroke="#333" stroke-width="1"/>

  <text x="200" y="247" font-size="9" fill="#666">Yes</text>
  <text x="335" y="247" font-size="9" fill="#666">No</text>

  <rect x="120" y="255" width="155" height="35" rx="4" fill="#f44336"/>
  <text x="198" y="277" text-anchor="middle" font-size="10" fill="#fff">Increase pool + fix queries</text>

  <rect x="290" y="255" width="145" height="35" rx="4" fill="#f44336"/>
  <text x="363" y="277" text-anchor="middle" font-size="10" fill="#fff">Check Redis / memory</text>

  <!-- Specific endpoints slow -->
  <rect x="510" y="120" width="220" height="40" rx="6" fill="#ff9800" stroke="#e65100" stroke-width="1.5"/>
  <text x="620" y="145" text-anchor="middle" font-size="11" font-weight="bold" fill="#fff">Check: Which endpoint?</text>

  <line x1="570" y1="160" x2="510" y2="190" stroke="#333" stroke-width="1"/>
  <line x1="670" y1="160" x2="730" y2="190" stroke="#333" stroke-width="1"/>

  <text x="520" y="182" font-size="9" fill="#666">List endpoints</text>
  <text x="710" y="182" font-size="9" fill="#666">Write endpoints</text>

  <rect x="430" y="190" width="160" height="35" rx="4" fill="#f44336"/>
  <text x="510" y="212" text-anchor="middle" font-size="10" fill="#fff">N+1 queries / missing index</text>

  <rect x="650" y="190" width="160" height="35" rx="4" fill="#f44336"/>
  <text x="730" y="212" text-anchor="middle" font-size="10" fill="#fff">Lock contention / slow writes</text>

  <!-- Memory leak branch -->
  <rect x="250" y="320" width="300" height="40" rx="6" fill="#8bc34a" stroke="#558b2f" stroke-width="2"/>
  <text x="400" y="345" text-anchor="middle" font-size="12" font-weight="bold" fill="#fff">Degrades over time? -> Memory leak (soak test)</text>

  <line x1="400" y1="295" x2="400" y2="320" stroke="#333" stroke-width="1" stroke-dasharray="4,3"/>
  <text x="415" y="312" font-size="9" fill="#666">If gradual</text>
</svg>

---

## 6. Fixing and Retesting

This is where load testing transforms from "interesting diagnostic tool" to "the thing that actually makes your system fast." The methodology is:

1. Identify the worst bottleneck from your load test results
2. Fix that single bottleneck
3. Rerun the exact same load test
4. Measure the improvement
5. Repeat

Fix one thing at a time. If you fix three things simultaneously, you cannot attribute improvement to any specific change. You need to know which fixes matter and by how much, because in the future you will need to make trade-off decisions about engineering effort versus performance gain.

### Fix 1: Add Database Indexes

The dashboard query was doing sequential scans on the `generations` table to find the latest generation for each project. Let us add the missing indexes.

```sql
-- Index for looking up generations by project (via scene)
-- This supports the dashboard query that joins scenes to generations
CREATE INDEX CONCURRENTLY idx_generations_scene_id_created_at
ON generations (scene_id, created_at DESC);

-- Index for filtering generations by status (for the status polling endpoint)
CREATE INDEX CONCURRENTLY idx_generations_status
ON generations (status)
WHERE status IN ('queued', 'processing');

-- Index for listing projects by user (for the dashboard)
CREATE INDEX CONCURRENTLY idx_projects_user_id_updated_at
ON projects (user_id, updated_at DESC);

-- Index for listing scenes by project (for the project editor)
CREATE INDEX CONCURRENTLY idx_scenes_project_id_order
ON scenes (project_id, order_index);
```

A note on `CREATE INDEX CONCURRENTLY`: without the `CONCURRENTLY` keyword, PostgreSQL takes an exclusive lock on the table during index creation, blocking all writes. On a production table with ongoing traffic, this can cause downtime. `CONCURRENTLY` builds the index without blocking writes, at the cost of taking slightly longer.

**Result after retest**:

| Metric | Before Indexes | After Indexes | Improvement |
|--------|---------------|---------------|-------------|
| Dashboard P50 @ 100 VUs | 340ms | 65ms | 5.2x faster |
| Dashboard P95 @ 100 VUs | 1,200ms | 180ms | 6.7x faster |
| Dashboard P99 @ 100 VUs | 2,300ms | 400ms | 5.8x faster |
| Status Poll P95 @ 100 VUs | 400ms | 120ms | 3.3x faster |
| DB CPU @ 100 VUs | 62% | 18% | 3.4x lower |
| Error Rate @ 100 VUs | 4.2% | 0.8% | 5.3x lower |

Indexes alone dropped the dashboard P95 from 1.2 seconds to 180ms. This is the single highest-impact change you can make, and it is four lines of SQL.

### Fix 2: Eliminate N+1 Queries

Replace the per-project generation lookup with a single query:

```typescript
// BEFORE: N+1 pattern (21 queries for 20 projects)
const projects = await db.query(
  'SELECT * FROM projects WHERE user_id = $1 ORDER BY updated_at DESC',
  [userId]
);
for (const project of projects) {
  const gen = await db.query(
    `SELECT status, created_at FROM generations
     WHERE scene_id IN (SELECT id FROM scenes WHERE project_id = $1)
     ORDER BY created_at DESC LIMIT 1`,
    [project.id]
  );
  project.latestGeneration = gen.rows[0];
}

// AFTER: Single query with lateral join (1 query total)
const result = await db.query(`
  SELECT
    p.*,
    latest_gen.status AS latest_gen_status,
    latest_gen.created_at AS latest_gen_created_at,
    latest_gen.model AS latest_gen_model
  FROM projects p
  LEFT JOIN LATERAL (
    SELECT g.status, g.created_at, g.model
    FROM generations g
    JOIN scenes s ON g.scene_id = s.id
    WHERE s.project_id = p.id
    ORDER BY g.created_at DESC
    LIMIT 1
  ) latest_gen ON true
  WHERE p.user_id = $1
  ORDER BY p.updated_at DESC
`, [userId]);
```

A **lateral join** is a powerful SQL feature that lets a subquery reference columns from preceding tables in the `FROM` clause. In this case, the subquery finds the latest generation *for each project*, all in a single round trip to the database. Without the lateral join, you either do N+1 queries or pull all generations into application memory and filter there.

**Result after retest**:

| Metric | After Indexes Only | After N+1 Fix | Improvement |
|--------|-------------------|---------------|-------------|
| Dashboard P50 @ 100 VUs | 65ms | 28ms | 2.3x faster |
| Dashboard P95 @ 100 VUs | 180ms | 62ms | 2.9x faster |
| DB Queries/sec @ 100 VUs | 4,200 | 890 | 4.7x fewer |
| DB Connections (active) @ 100 VUs | 32 | 12 | 2.7x fewer |

The N+1 fix reduced database queries by nearly 5x. Each dashboard load now executes 1 query instead of 21. This is why the connection pool usage dropped so dramatically --- fewer concurrent queries means fewer connections needed.

### Fix 3: Connection Pool Tuning

Even with the query optimizations, we should properly size the connection pool. The formula is:

$$
\text{pool\_size} = \left\lceil \frac{\text{concurrent\_requests} \times \text{avg\_query\_time}}{\text{target\_wait\_time}} \right\rceil
$$

But there is a simpler heuristic from the PostgreSQL documentation: for most workloads, the optimal pool size is:

$$
\text{pool\_size} = \text{CPU\_cores} \times 2 + \text{disk\_spindles}
$$

For an SSD-backed database on a 4-core instance, that gives \(4 \times 2 + 1 = 9\). In practice, you should benchmark, but 10 to 20 connections per application instance is a reasonable starting point. If you have 4 application instances, that is 40 to 80 total database connections, leaving headroom below PostgreSQL's default `max_connections = 100`.

```typescript
// Connection pool configuration
import { Pool } from 'pg';

const pool = new Pool({
  host: process.env.DB_HOST,
  database: process.env.DB_NAME,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,

  // Pool sizing
  min: 5,                    // Keep at least 5 connections warm
  max: 20,                   // Never exceed 20 connections
  idleTimeoutMillis: 30000,  // Close idle connections after 30s
  connectionTimeoutMillis: 5000, // Fail fast if no connection available in 5s

  // Statement timeout prevents runaway queries from hogging connections
  statement_timeout: 10000,  // Kill queries that take > 10s
});

// Monitor pool health
setInterval(() => {
  console.log({
    pool_total: pool.totalCount,
    pool_idle: pool.idleCount,
    pool_waiting: pool.waitingCount,
  });
}, 10000);
```

**Result after retest**:

| Metric | Before Pool Tuning | After Pool Tuning | Improvement |
|--------|-------------------|-------------------|-------------|
| Error Rate @ 200 VUs | 12.3% | 0.1% | 123x lower |
| Connection Timeout Errors | 847 | 2 | Effectively eliminated |
| P99 @ 200 VUs | 1,800ms | 320ms | 5.6x faster |

The key change here was not the pool size itself, but the `connectionTimeoutMillis` and `statement_timeout`. Before, requests would wait indefinitely for a connection, causing cascading timeouts. Now they fail fast after 5 seconds, which prevents the positive feedback loop we described with Little's Law.

### Fix 4: Redis Caching for Hot Paths

The dashboard and project listing endpoints are read-heavy. The data changes infrequently (only when a generation completes or a user creates/edits a project). This is a textbook case for caching.

```typescript
import Redis from 'ioredis';

const redis = new Redis(process.env.REDIS_URL);

// Cache wrapper with automatic invalidation
async function cachedQuery<T>(
  key: string,
  ttlSeconds: number,
  queryFn: () => Promise<T>
): Promise<T> {
  // Try cache first
  const cached = await redis.get(key);
  if (cached) {
    return JSON.parse(cached);
  }

  // Cache miss — run the query
  const result = await queryFn();

  // Store in cache with TTL
  await redis.setex(key, ttlSeconds, JSON.stringify(result));

  return result;
}

// Usage in the dashboard endpoint
app.get('/api/projects', async (req, res) => {
  const userId = req.user.id;

  const projects = await cachedQuery(
    `user:${userId}:projects`,
    60,  // Cache for 60 seconds
    () => db.query(`
      SELECT p.*, latest_gen.status AS latest_gen_status,
             latest_gen.created_at AS latest_gen_created_at
      FROM projects p
      LEFT JOIN LATERAL (
        SELECT g.status, g.created_at
        FROM generations g
        JOIN scenes s ON g.scene_id = s.id
        WHERE s.project_id = p.id
        ORDER BY g.created_at DESC LIMIT 1
      ) latest_gen ON true
      WHERE p.user_id = $1
      ORDER BY p.updated_at DESC
    `, [userId])
  );

  res.json({ projects: projects.rows });
});

// Invalidate cache when data changes
async function invalidateUserProjectCache(userId: string) {
  await redis.del(`user:${userId}:projects`);
}

// Call this after generation completes, project created, etc.
bullmqWorker.on('completed', async (job) => {
  await invalidateUserProjectCache(job.data.userId);
});
```

**Result after retest**:

| Metric | Before Caching | After Caching | Improvement |
|--------|---------------|---------------|-------------|
| Dashboard P50 @ 200 VUs | 28ms | 4ms | 7x faster |
| Dashboard P95 @ 200 VUs | 62ms | 12ms | 5.2x faster |
| Status Poll P95 @ 200 VUs | 85ms | 18ms | 4.7x faster |
| DB Queries/sec @ 200 VUs | 1,420 | 210 | 6.8x fewer |
| DB CPU @ 200 VUs | 38% | 6% | 6.3x lower |

Redis serves cached results in under 1ms (network round trip included on localhost). Queries that were hitting the database are now served from memory. The database CPU dropped to 6% because it is only handling cache misses and writes.

### Cumulative Improvement

Here is the full before/after comparison, showing the cumulative effect of all four fixes:

| Metric | Original (no fixes) | All Fixes Applied | Total Improvement |
|--------|---------------------|-------------------|-------------------|
| **Dashboard P50 @ 100 VUs** | 340ms | 4ms | 85x faster |
| **Dashboard P95 @ 100 VUs** | 1,200ms | 10ms | 120x faster |
| **Dashboard P95 @ 200 VUs** | 8,400ms | 12ms | 700x faster |
| **Status Poll P95 @ 200 VUs** | 5,300ms | 18ms | 294x faster |
| **Generation Trigger P95 @ 200 VUs** | timeout | 95ms | From dead to fast |
| **Error Rate @ 200 VUs** | 38.6% | 0.08% | 483x lower |
| **Throughput @ 200 VUs** | 31 req/s | 185 req/s | 6x higher |
| **DB CPU @ 200 VUs** | 100% | 6% | 16.7x lower |

The system went from crumbling at 50 users to handling 200 users with headroom to spare. The total engineering effort: four SQL `CREATE INDEX` statements, one query refactor, a pool configuration change, and a caching layer. Maybe two days of work. The alternative --- discovering these issues in production with real users --- would have cost weeks of firefighting, customer churn, and lost revenue.

---

## 7. Advanced Patterns

The basic ramp-up test we ran in section 4 catches the most common failures. But there are entire categories of bugs that only surface under specific conditions. Let us design tests for those.

### 7.1 Soak Testing: Finding Memory Leaks

A soak test runs moderate load for an extended period --- typically 4 to 24 hours. The goal is to find bugs that accumulate over time: memory leaks, connection leaks, file descriptor exhaustion, disk space depletion, log file growth.

```javascript
// soak-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend } from 'k6/metrics';

const responseTime = new Trend('soak_response_time', true);

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3000';

export const options = {
  scenarios: {
    soak: {
      executor: 'constant-vus',
      vus: 50,           // Moderate load — not trying to break the system
      duration: '4h',    // Run for 4 hours
    },
  },
  thresholds: {
    soak_response_time: ['p(95)<300'],
    http_req_failed: ['rate<0.01'],
  },
};

function getAuthToken() {
  // In practice, cache this per VU to avoid re-authenticating every iteration
  return 'test-token-for-soak';
}

export default function () {
  const token = getAuthToken();

  // Rotate through different endpoints to simulate real usage
  const endpoints = [
    '/api/projects',
    '/api/projects/proj_001/scenes',
    '/api/generations/gen_001/status',
    '/api/health',
  ];

  const path = endpoints[Math.floor(Math.random() * endpoints.length)];
  const response = http.get(`${BASE_URL}${path}`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  responseTime.add(response.timings.duration);
  check(response, {
    'status is 200': (r) => r.status === 200,
  });

  sleep(Math.random() * 3 + 1);
}
```

Run it, then monitor your server's memory over time:

```bash
# In a separate terminal, record memory every 30 seconds
while true; do
  TIMESTAMP=$(date -u +%H:%M:%S)
  MEM=$(curl -s http://localhost:3000/debug/memory)
  echo "[$TIMESTAMP] $MEM"
  sleep 30
done
```

A healthy system has memory that grows during startup, stabilizes, and stays flat. A leaking system shows steady upward growth. If your heap goes from 200MB to 800MB over 4 hours with constant load, you have a leak.

### 7.2 Spike Testing: Simulating Viral Traffic

A spike test slams your system with maximum load instantaneously, with no warm-up.

```javascript
// spike-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3000';

export const options = {
  scenarios: {
    spike: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '10s', target: 5 },     // Baseline: 5 users
        { duration: '1m', target: 5 },       // Hold baseline for 1 minute
        { duration: '10s', target: 500 },    // SPIKE: 0 to 500 in 10 seconds
        { duration: '3m', target: 500 },     // Hold spike for 3 minutes
        { duration: '10s', target: 5 },      // Drop back to baseline
        { duration: '3m', target: 5 },       // Recovery period — does it stabilize?
        { duration: '10s', target: 0 },      // Ramp down
      ],
    },
  },
  thresholds: {
    // During a spike, we accept higher latency but still track it.
    // The key question is: does the system RECOVER after the spike ends?
    http_req_duration: ['p(99)<5000'],   // Even during spike, P99 < 5s
    http_req_failed: ['rate<0.10'],      // Accept up to 10% errors during spike
  },
};

function getTestToken() {
  return 'test-token-for-spike';
}

export default function () {
  const response = http.get(`${BASE_URL}/api/projects`, {
    headers: { Authorization: `Bearer ${getTestToken()}` },
  });

  check(response, {
    'status is 200 or 429': (r) => r.status === 200 || r.status === 429,
  });

  sleep(Math.random() * 2 + 0.5);
}
```

The critical metric in a spike test is not peak performance --- it is **recovery time**. After the spike ends and load drops back to baseline:
- Do response times return to normal? How quickly?
- Is the error rate back to zero?
- Is memory back to pre-spike levels?

If the system does not recover, you have a state corruption bug --- probably a connection pool that is full of dead connections, a queue that is backed up and not draining, or a cache that got thrashed.

### 7.3 Testing WebSocket Connections

If your platform uses WebSocket for real-time generation status (which it should, per the [earlier post in this series](/2026/01/20/websocket-architecture-generation-status.html)), you need to test that connection layer separately. k6 has built-in WebSocket support.

```javascript
// websocket-load-test.js
import ws from 'k6/ws';
import { check } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const wsConnectDuration = new Trend('ws_connect_duration', true);
const wsMessageReceived = new Rate('ws_message_received');

export const options = {
  scenarios: {
    websockets: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 100 },
        { duration: '5m', target: 100 },
        { duration: '1m', target: 500 },
        { duration: '5m', target: 500 },
        { duration: '2m', target: 1000 },
        { duration: '5m', target: 1000 },
        { duration: '2m', target: 0 },
      ],
    },
  },
  thresholds: {
    ws_connect_duration: ['p(95)<500'],
    ws_message_received: ['rate>0.95'],
  },
};

function getTestToken() {
  return `test-token-ws-${__VU}`;
}

export default function () {
  const token = getTestToken();
  const url = `ws://localhost:3000/ws?token=${token}`;

  const startTime = Date.now();

  const response = ws.connect(url, {}, function (socket) {
    wsConnectDuration.add(Date.now() - startTime);

    // Subscribe to a generation status channel
    socket.on('open', () => {
      socket.send(JSON.stringify({
        type: 'subscribe',
        generationId: `gen_test_${__VU}`,
      }));
    });

    socket.on('message', (data) => {
      const msg = JSON.parse(data);
      const messageOk = check(msg, {
        'message has type': (m) => m.type !== undefined,
        'message has timestamp': (m) => m.timestamp !== undefined,
      });
      wsMessageReceived.add(messageOk ? 1 : 0);
    });

    socket.on('error', (e) => {
      console.error(`WebSocket error: ${e.error()}`);
      wsMessageReceived.add(0);
    });

    // Keep the connection open for 60 seconds
    socket.setTimeout(function () {
      socket.close();
    }, 60000);
  });

  check(response, {
    'ws status is 101': (r) => r && r.status === 101,
  });
}
```

This test reveals:
- **File descriptor limits**: Each WebSocket connection consumes a file descriptor. Most Linux systems default to 1,024 per process. At 1,000+ concurrent connections, you will hit this.
- **Memory per connection**: Each WebSocket connection maintains buffer state. At 1,000 connections with Node.js `ws` library, expect roughly 2-10KB per connection (2-10MB total).
- **Redis Pub/Sub fan-out**: If each WebSocket subscriber creates a Redis subscription, you can hit Redis connection limits.

**Fix file descriptor limits before testing**:

```bash
# Check current limit
ulimit -n

# Increase for the current session
ulimit -n 65535

# Permanent fix: edit /etc/security/limits.conf
# youruser  soft  nofile  65535
# youruser  hard  nofile  65535
```

### 7.4 Geographic Distribution

If your users are global, local testing only tells you about latency to your server from one location. Network round-trip time from New York to a server in `us-east-1` is ~5ms. From Tokyo, it is ~200ms. That 200ms gets added to every HTTP request.

k6 Cloud (the commercial version) can run tests from multiple geographic locations simultaneously. If you are on the free tier, you can simulate geographic latency locally:

```javascript
// Add artificial latency to simulate geographic distance
import { sleep } from 'k6';

function simulateNetworkLatency() {
  // Simulate a distribution of network round-trip times
  // matching a 70% US, 20% EU, 10% Asia user base
  const rand = Math.random();
  if (rand < 0.7) {
    sleep(0.01);   // US: ~10ms RTT
  } else if (rand < 0.9) {
    sleep(0.08);   // EU: ~80ms RTT
  } else {
    sleep(0.2);    // Asia: ~200ms RTT
  }
}
```

---

## 8. Interpreting Results and Setting SLOs

Load test results are only useful if you know what "good" looks like. This is where **SLOs** come in.

### Definitions

An **SLI (Service Level Indicator)** is a quantitative measurement of some aspect of your service. Examples:
- Request latency P95
- Error rate
- Throughput (requests per second)
- Availability (percentage of successful health checks)

An **SLO (Service Level Objective)** is a target value for an SLI. Examples:
- "API request latency P95 shall be below 200ms"
- "Error rate shall be below 0.1%"
- "The service shall be available 99.9% of the time"

An **SLA (Service Level Agreement)** is a contractual commitment with consequences for missing it. You probably do not need SLAs unless you are selling to enterprise customers. But you absolutely need SLOs, because they give you a decision framework for when to invest in performance work versus features.

An **error budget** is the inverse of your SLO. If your availability SLO is 99.9%, your error budget is 0.1% --- which translates to roughly 43 minutes of allowed downtime per month. When you have consumed your error budget, you stop shipping features and fix reliability. When you have budget remaining, you ship features with confidence.

### SLOs for an AI Video Platform

Based on the load test results and the nature of an AI video platform, here are recommended SLOs:

| SLI | SLO | Rationale |
|-----|-----|-----------|
| **API response time (non-generation)** | P95 < 200ms | Users perceive anything under 200ms as "instant." Above that, the UI feels sluggish. |
| **Dashboard load time** | P95 < 300ms | The dashboard is the first thing users see. Slow dashboards kill retention. |
| **Generation trigger response** | P95 < 500ms | This only enqueues a job, so it should be fast. The actual generation takes minutes and is expected to. |
| **Status poll response** | P99 < 50ms | Status polling happens every few seconds. Even small latency adds up to perceived sluggishness. |
| **WebSocket message delivery** | P99 < 100ms | Real-time status must feel real-time. Over 100ms and the progress bar feels choppy. |
| **Error rate (all endpoints)** | < 0.1% | One in a thousand requests can fail. More than that and users notice. |
| **Availability** | 99.9% | 43 minutes of downtime per month. Achievable without heroic infrastructure investment. |
| **Generation success rate** | > 98% | External API failures are not always our fault, but users do not care whose fault it is. |

### Translating SLOs into k6 Thresholds

The SLOs translate directly into k6 threshold configuration:

```javascript
export const options = {
  thresholds: {
    // API response time P95 < 200ms
    http_req_duration: ['p(95)<200'],

    // Dashboard P95 < 300ms (using tagged requests)
    'http_req_duration{endpoint:dashboard}': ['p(95)<300'],

    // Status poll P99 < 50ms
    'http_req_duration{endpoint:status_poll}': ['p(99)<50'],

    // Generation trigger P95 < 500ms
    'http_req_duration{endpoint:trigger_generation}': ['p(95)<500'],

    // Error rate < 0.1%
    http_req_failed: ['rate<0.001'],

    // Business error rate < 0.5%
    business_error_rate: ['rate<0.005'],
  },
};
```

When k6 runs with thresholds, it exits with a non-zero status code if any threshold is breached. This means you can integrate load tests into your CI/CD pipeline and fail a deployment if performance has regressed.

### Amdahl's Law: The Limits of Optimization

There is a point at which further optimization of individual components yields diminishing returns. **Amdahl's Law** quantifies this:

$$
S(n) = \frac{1}{(1 - p) + \frac{p}{n}}
$$

Where:
- \(S(n)\) is the theoretical speedup of the system when you parallelize the portion \(p\) of it across \(n\) processors (or threads, or instances)
- \(p\) is the fraction of the workload that can be parallelized
- \(1 - p\) is the fraction that is inherently sequential

Suppose 80% of your request processing time is spent waiting for the database (parallelizable, since you can add read replicas), and 20% is sequential application logic. Even with infinite database performance (\(n \to \infty\)):

$$
S(\infty) = \frac{1}{(1 - 0.8) + 0} = \frac{1}{0.2} = 5
$$

The maximum speedup is 5x. You can never make the system more than 5x faster by optimizing only the database, because the sequential 20% becomes the bottleneck.

This is why our optimization sequence in section 6 worked so well: we were not just making one component faster, we were reducing the sequential fraction (by caching results that previously required query + serialization + response construction) and eliminating unnecessary work entirely (by fixing N+1 queries).

---

## 9. The Load Testing Checklist

Here is the complete checklist for load testing an AI video platform. Tape this to your monitor before launch.

### When to Run Load Tests

- [ ] Before every production deployment that touches data access patterns, connection management, or caching logic
- [ ] After adding a new database table or changing schema
- [ ] After adding or modifying database indexes
- [ ] After significant changes to caching strategy
- [ ] Before any expected traffic event (launch, marketing campaign, press coverage)
- [ ] Weekly automated soak test (4-hour, via CI)
- [ ] Monthly comprehensive stress test (full ramp to breaking point)

### What to Test

- [ ] Complete user journey (login through generation status polling)
- [ ] Dashboard / project listing (highest traffic endpoint)
- [ ] Generation trigger (the most expensive operation)
- [ ] Status polling (highest frequency operation)
- [ ] WebSocket connections at scale
- [ ] Authentication under load (token validation, session management)
- [ ] File upload / download (if users upload reference images)

### Minimum Scenarios

- [ ] **Ramp-up**: 0 to expected peak, hold, measure
- [ ] **Stress**: 0 to 2x expected peak, find breaking point
- [ ] **Soak**: 50% of expected peak for 4 hours minimum
- [ ] **Spike**: 0 to maximum instantly, verify recovery

### What to Monitor During Tests

- [ ] k6 metrics: P50, P95, P99 response times, error rate, throughput
- [ ] Database: connection count, active queries, CPU, disk I/O, slow query log
- [ ] Redis: connection count, memory usage, ops/sec, Pub/Sub channels
- [ ] Application: memory (RSS, heap), CPU, event loop lag, open file descriptors
- [ ] Infrastructure: network I/O, disk space, container restart count

### How to Interpret Results

| Signal | What It Means | What to Fix |
|--------|--------------|-------------|
| P95 degrades linearly with VU count | Resource contention (DB, CPU, memory) | Add indexes, fix queries, increase pool sizes |
| P95 degrades exponentially with VU count | Saturated resource causing cascading failure | Find the saturated resource (usually DB connections) and fix it |
| Error rate spikes at specific VU count | Hard resource limit hit | Increase the limit (connections, file descriptors, memory) |
| Response times drift upward over hours | Memory leak or resource exhaustion | Profile memory, check for connection leaks |
| System does not recover after spike | State corruption or queue backup | Implement circuit breakers, add backpressure mechanisms |
| One endpoint is 10x slower than others | N+1 queries or missing index on that endpoint | EXPLAIN ANALYZE the query, add indexes |

### The Numbers You Need to Know

Before launch, you should be able to state with confidence:

1. **Maximum concurrent users**: "Our system handles X concurrent users with all SLOs met."
2. **Breaking point**: "At Y concurrent users, error rate exceeds 1% / latency exceeds SLO."
3. **Headroom**: "We have Z% headroom above expected peak traffic."
4. **Recovery time**: "After a traffic spike, the system recovers to normal within N seconds."
5. **Soak stability**: "Under sustained moderate load for 4 hours, memory stays flat and error rate stays below 0.1%."

If you cannot state these five things with numbers, you are not ready to launch. Run the tests. Get the numbers. Fix what breaks. Then launch with the specific kind of confidence that comes from having watched your system handle everything you could throw at it --- because you were the one doing the throwing.

---

## Series Navigation

This is **Part 3** of a 4-part series on taking vibe-coded projects to production:

1. [Redis + BullMQ: Production Job Queue Architecture](/2026/01/17/redis-bullmq-job-queue-video.html) --- Decoupling requests from processing
2. [WebSocket Architecture for Real-Time Status](/2026/01/20/websocket-architecture-generation-status.html) --- Pushing updates to clients
3. **Load Testing to Destruction** (this post) --- Breaking your system before users do
4. *Observability and Incident Response* (coming soon) --- Monitoring, alerting, and debugging in production
