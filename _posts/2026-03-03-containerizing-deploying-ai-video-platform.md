---
layout: post
title: "Containerizing and Deploying an AI Video Platform: Docker, Reverse Proxies, and CI/CD from Scratch"
date: 2026-03-03
category: tools
mathjax: false
---

# Containerizing and Deploying an AI Video Platform: Docker, Reverse Proxies, and CI/CD from Scratch

*This is Part 2 of a 4-part series on taking vibe-coded projects to production.*

Your app works on your machine. It works in development. You run `npm run dev`, everything compiles, the API responds, videos generate, life is good. Then you try to deploy it to a server and discover that "it works on my machine" is not a deployment strategy. Deployment is where vibe-coded projects go to die---not because the code is bad, but because nobody thought about the *environment* the code runs in.

Your Node.js version is different. Your system has `ffmpeg` installed globally but the server does not. You have environment variables set in your `.bashrc` that you forgot about. The file paths are wrong. Redis is not running. PostgreSQL expects a different user. The port is already taken. Each of these is a five-minute fix in isolation and a three-day nightmare when they all hit you at once on a Friday evening while a demo is scheduled for Monday.

This post is the complete, ground-up guide to containerizing and deploying an AI video generation platform. We will start from what a container actually is (not a metaphor), build production Dockerfiles, wire up multi-service development with Docker Compose, set up Nginx as a reverse proxy with TLS, lock down a server with a firewall, and automate the entire pipeline with GitHub Actions CI/CD. Every configuration file is real. Every decision is explained.

---

## Table of Contents

1. [What Containers Actually Are](#what-containers-actually-are)
2. [Writing Production Dockerfiles](#writing-production-dockerfiles)
3. [Docker Compose for Local Development](#docker-compose-for-local-development)
4. [Networking Fundamentals You Must Know](#networking-fundamentals-you-must-know)
5. [Reverse Proxies: Nginx](#reverse-proxies-nginx)
6. [TLS and HTTPS](#tls-and-https)
7. [Firewall Configuration](#firewall-configuration)
8. [CI/CD with GitHub Actions](#cicd-with-github-actions)
9. [The Deployment Checklist](#the-deployment-checklist)

---

## What Containers Actually Are

Before we use Docker, let us understand what it actually does. The word "container" is used so loosely that most people think of it as a "lightweight VM." It is not. The distinction matters, and understanding it will save you hours of debugging.

### Processes, Not Machines

A **process** is a running program. When you type `node server.js`, the Linux kernel creates a process. That process has a process ID (PID), it can see the filesystem, it can see other processes, it can open network ports, and it can use CPU and memory without restriction (up to whatever the system has).

A **container** is a process (or group of processes) with *restricted visibility*. It is still a normal Linux process running on the host kernel. There is no separate operating system. There is no hypervisor. The difference between a container and a regular process is that the container has been given a restricted view of the system using two Linux kernel features: **namespaces** and **cgroups**.

**Namespaces** control *what a process can see*:
- **PID namespace**: The container sees only its own processes. PID 1 inside the container is not PID 1 on the host.
- **Network namespace**: The container gets its own network stack---its own IP address, its own ports. Port 3000 inside the container is not port 3000 on the host unless you explicitly map it.
- **Mount namespace**: The container sees its own filesystem. It cannot see the host's files unless you explicitly share them.
- **User namespace**: The container can have its own user IDs. Root inside the container does not have to be root on the host.

**Cgroups** (control groups) control *how much a process can use*:
- CPU limits: "This container gets at most 2 CPU cores."
- Memory limits: "This container gets at most 512MB of RAM. If it tries to use more, kill it."
- I/O limits: Restrict disk and network bandwidth.

The combination of these two features means a container is an isolated, resource-limited process. It thinks it is the only thing running on the system. It has its own filesystem, its own network, its own process tree. But underneath, it is sharing the host's kernel with every other container on that machine.

### Why This Matters for Deployment

The reason this matters is **reproducibility**. When you run your application inside a container, you are specifying *everything* about its environment: the base operating system, every installed package, every file, every environment variable, every port. The container does not depend on what is installed on the host. It does not depend on the host's Node.js version, or whether `ffmpeg` is available, or what version of `libc` is installed. It carries all of that with it.

This solves the "it works on my machine" problem by making your machine irrelevant. The container is the machine.

### Key Terminology

Before we go further, let me define the terms precisely:

| Term | Definition |
|------|-----------|
| **Image** | A read-only template containing the filesystem and configuration for a container. Think of it as a snapshot of a fully configured system. |
| **Container** | A running instance of an image. You can run multiple containers from the same image. |
| **Layer** | Images are built from stacked layers. Each instruction in a Dockerfile creates a layer. Layers are cached and shared between images. |
| **Registry** | A server that stores and distributes images. Docker Hub is the default public registry. GitHub Container Registry (ghcr.io) is another. |
| **Dockerfile** | A text file containing instructions for building an image. Each line is a step: install this, copy that, run this command. |
| **Volume** | A mechanism for persisting data outside the container's filesystem. When a container is destroyed, its filesystem is destroyed with it---unless you use a volume. |

<svg viewBox="0 0 880 400" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:system-ui,sans-serif">
  <defs>
    <marker id="arr-c" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>

  <text x="440" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#222">Container vs VM Architecture</text>

  <!-- VM Side -->
  <text x="220" y="60" text-anchor="middle" font-size="14" font-weight="bold" fill="#555">Virtual Machine</text>
  <rect x="40" y="300" width="360" height="50" rx="4" fill="#e0e0e0" stroke="#999" stroke-width="1.5"/>
  <text x="220" y="330" text-anchor="middle" font-size="13" fill="#333">Host Hardware</text>
  <rect x="40" y="250" width="360" height="50" rx="4" fill="#b0bec5" stroke="#999" stroke-width="1.5"/>
  <text x="220" y="280" text-anchor="middle" font-size="13" fill="#333">Hypervisor</text>

  <rect x="50" y="80" width="105" height="165" rx="4" fill="#fff3e0" stroke="#ff9800" stroke-width="1.5"/>
  <rect x="55" y="85" width="95" height="40" rx="3" fill="#ffe0b2"/>
  <text x="102" y="110" text-anchor="middle" font-size="10" fill="#333">App A</text>
  <rect x="55" y="130" width="95" height="30" rx="3" fill="#ffcc80"/>
  <text x="102" y="150" text-anchor="middle" font-size="10" fill="#333">Bins/Libs</text>
  <rect x="55" y="165" width="95" height="35" rx="3" fill="#ffb74d"/>
  <text x="102" y="187" text-anchor="middle" font-size="10" fill="#fff">Guest OS</text>
  <rect x="55" y="205" width="95" height="30" rx="3" fill="#f57c00"/>
  <text x="102" y="225" text-anchor="middle" font-size="10" fill="#fff">Virtual HW</text>

  <rect x="168" y="80" width="105" height="165" rx="4" fill="#fff3e0" stroke="#ff9800" stroke-width="1.5"/>
  <rect x="173" y="85" width="95" height="40" rx="3" fill="#ffe0b2"/>
  <text x="220" y="110" text-anchor="middle" font-size="10" fill="#333">App B</text>
  <rect x="173" y="130" width="95" height="30" rx="3" fill="#ffcc80"/>
  <text x="220" y="150" text-anchor="middle" font-size="10" fill="#333">Bins/Libs</text>
  <rect x="173" y="165" width="95" height="35" rx="3" fill="#ffb74d"/>
  <text x="220" y="187" text-anchor="middle" font-size="10" fill="#fff">Guest OS</text>
  <rect x="173" y="205" width="95" height="30" rx="3" fill="#f57c00"/>
  <text x="220" y="225" text-anchor="middle" font-size="10" fill="#fff">Virtual HW</text>

  <rect x="285" y="80" width="105" height="165" rx="4" fill="#fff3e0" stroke="#ff9800" stroke-width="1.5"/>
  <rect x="290" y="85" width="95" height="40" rx="3" fill="#ffe0b2"/>
  <text x="337" y="110" text-anchor="middle" font-size="10" fill="#333">App C</text>
  <rect x="290" y="130" width="95" height="30" rx="3" fill="#ffcc80"/>
  <text x="337" y="150" text-anchor="middle" font-size="10" fill="#333">Bins/Libs</text>
  <rect x="290" y="165" width="95" height="35" rx="3" fill="#ffb74d"/>
  <text x="337" y="187" text-anchor="middle" font-size="10" fill="#fff">Guest OS</text>
  <rect x="290" y="205" width="95" height="30" rx="3" fill="#f57c00"/>
  <text x="337" y="225" text-anchor="middle" font-size="10" fill="#fff">Virtual HW</text>

  <!-- Container Side -->
  <text x="660" y="60" text-anchor="middle" font-size="14" font-weight="bold" fill="#555">Container</text>
  <rect x="480" y="300" width="360" height="50" rx="4" fill="#e0e0e0" stroke="#999" stroke-width="1.5"/>
  <text x="660" y="330" text-anchor="middle" font-size="13" fill="#333">Host Hardware</text>
  <rect x="480" y="250" width="360" height="50" rx="4" fill="#b0bec5" stroke="#999" stroke-width="1.5"/>
  <text x="660" y="280" text-anchor="middle" font-size="13" fill="#333">Host OS + Kernel</text>
  <rect x="480" y="200" width="360" height="50" rx="4" fill="#90caf9" stroke="#42a5f5" stroke-width="1.5"/>
  <text x="660" y="230" text-anchor="middle" font-size="13" fill="#333">Container Runtime (Docker)</text>

  <rect x="490" y="80" width="105" height="115" rx="4" fill="#e3f2fd" stroke="#42a5f5" stroke-width="1.5"/>
  <rect x="495" y="85" width="95" height="40" rx="3" fill="#bbdefb"/>
  <text x="542" y="110" text-anchor="middle" font-size="10" fill="#333">App A</text>
  <rect x="495" y="130" width="95" height="30" rx="3" fill="#90caf9"/>
  <text x="542" y="150" text-anchor="middle" font-size="10" fill="#333">Bins/Libs</text>
  <text x="542" y="183" text-anchor="middle" font-size="9" fill="#666">No Guest OS</text>

  <rect x="608" y="80" width="105" height="115" rx="4" fill="#e3f2fd" stroke="#42a5f5" stroke-width="1.5"/>
  <rect x="613" y="85" width="95" height="40" rx="3" fill="#bbdefb"/>
  <text x="660" y="110" text-anchor="middle" font-size="10" fill="#333">App B</text>
  <rect x="613" y="130" width="95" height="30" rx="3" fill="#90caf9"/>
  <text x="660" y="150" text-anchor="middle" font-size="10" fill="#333">Bins/Libs</text>
  <text x="660" y="183" text-anchor="middle" font-size="9" fill="#666">No Guest OS</text>

  <rect x="725" y="80" width="105" height="115" rx="4" fill="#e3f2fd" stroke="#42a5f5" stroke-width="1.5"/>
  <rect x="730" y="85" width="95" height="40" rx="3" fill="#bbdefb"/>
  <text x="777" y="110" text-anchor="middle" font-size="10" fill="#333">App C</text>
  <rect x="730" y="130" width="95" height="30" rx="3" fill="#90caf9"/>
  <text x="777" y="150" text-anchor="middle" font-size="10" fill="#333">Bins/Libs</text>
  <text x="777" y="183" text-anchor="middle" font-size="9" fill="#666">No Guest OS</text>

  <!-- Size annotations -->
  <text x="220" y="380" text-anchor="middle" font-size="11" fill="#888">Each VM: 1-10+ GB (includes full OS)</text>
  <text x="660" y="380" text-anchor="middle" font-size="11" fill="#888">Each container: 50-500 MB (shares host kernel)</text>
</svg>

This is why containers start in milliseconds (they are just processes) while VMs take seconds to minutes (they boot an entire operating system). For our AI video platform, this means we can restart our API server in under a second during deployments, while a VM-based approach would mean seconds of downtime per restart.

---

## Writing Production Dockerfiles

Let us build a Dockerfile for our AI video platform API. We will start with the naive version---the one most tutorials teach you---and then fix every problem with it, one at a time, until we have a production-grade image.

### The Naive Dockerfile

```dockerfile
FROM node:20
WORKDIR /app
COPY . .
RUN npm install
RUN npm run build
EXPOSE 3000
CMD ["node", "dist/server.js"]
```

This works. It will build. It will run. And it has at least six serious problems.

### Problem Analysis

| Problem | Why It Is Bad | Impact |
|---------|---------------|--------|
| `FROM node:20` (full image) | Includes compilers, man pages, Python, and hundreds of packages you do not need | Image is ~1.1 GB instead of ~180 MB |
| `COPY . .` before `npm install` | Every code change invalidates the npm install cache, causing full reinstall every build | Build time goes from 10s to 120s+ |
| `npm install` (not `npm ci`) | Does not use lockfile, may install different versions | Non-reproducible builds |
| No `.dockerignore` | Copies `node_modules`, `.git`, `.env`, test files into the image | Image bloat, potential secret leak |
| Runs as root | The container process has root privileges; if compromised, attacker has full control | Security vulnerability |
| No health check | Orchestrators cannot tell if the app is healthy or stuck | Zombie containers |

Let us fix each one.

### Step 1: Add .dockerignore

Before touching the Dockerfile, create a `.dockerignore` file. This is the Docker equivalent of `.gitignore`---it tells Docker which files to exclude from the build context.

```text
# .dockerignore
node_modules
.git
.gitignore
.env
.env.*
*.md
tests/
coverage/
.nyc_output/
docker-compose*.yml
Dockerfile*
.dockerignore
.vscode/
.idea/
*.log
dist/
```

Why does this matter? When you run `docker build`, Docker sends the entire directory (the "build context") to the Docker daemon. If `node_modules` contains 500MB of dependencies and `.git` contains 200MB of history, you are sending 700MB of data that will never be used in the build. The `.dockerignore` file prevents this.

### Step 2: Multi-Stage Build

A **multi-stage build** uses multiple `FROM` statements in a single Dockerfile. Each `FROM` starts a new "stage." You can copy files from one stage to another, but only what you explicitly copy makes it into the final image. This means you can have a "builder" stage with all your dev tools (TypeScript compiler, build tools, dev dependencies) and a "runner" stage with only the production code and runtime dependencies.

```dockerfile
# ============================================
# Stage 1: Builder
# ============================================
FROM node:20-slim AS builder

WORKDIR /app

# Copy dependency manifests first (layer caching)
COPY package.json package-lock.json ./

# Use npm ci for reproducible installs from lockfile
RUN npm ci

# Now copy source code
COPY tsconfig.json ./
COPY src/ ./src/
COPY prisma/ ./prisma/

# Generate Prisma client and build TypeScript
RUN npx prisma generate
RUN npm run build

# ============================================
# Stage 2: Production runner
# ============================================
FROM node:20-slim AS runner

WORKDIR /app

# Install only production dependencies
COPY package.json package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force

# Copy built artifacts from builder stage
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules/.prisma ./node_modules/.prisma
COPY --from=builder /app/prisma ./prisma

# Install system dependencies needed at runtime
# (ffmpeg for video processing, curl for health checks)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE 3000

# Health check: hit the health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Start the application
CMD ["node", "dist/server.js"]
```

Let me walk through every decision.

**`FROM node:20-slim AS builder`**: The `-slim` variant of the Node image strips out compilers and documentation, reducing the base from ~1.1GB to ~240MB. We use `AS builder` to name this stage so we can reference it later.

**Copying `package.json` before source code**: Docker caches each layer. If the files in a `COPY` instruction have not changed, Docker reuses the cached layer and everything after it. By copying the dependency manifests first and running `npm ci`, we ensure that the expensive dependency installation step is cached as long as `package.json` and `package-lock.json` do not change. Code changes (which happen constantly) only invalidate the layers *after* `npm ci`.

**`npm ci` instead of `npm install`**: The `ci` command installs *exactly* what is in the lockfile. `npm install` may resolve to different versions and update the lockfile. In a CI/CD environment, you want byte-for-byte reproducible installs.

**Separate runner stage**: The final image does not contain TypeScript source code, the TypeScript compiler, build tools, or dev dependencies. It contains only the compiled JavaScript, production `node_modules`, and runtime system dependencies.

**Non-root user**: By default, processes in Docker containers run as root. If an attacker exploits a vulnerability in your application, they get root access inside the container. Running as a non-root user limits the blast radius.

**Health check**: The `HEALTHCHECK` instruction tells Docker how to determine if the container is healthy. Docker will run the specified command every 30 seconds. If it fails 3 consecutive times, the container is marked as unhealthy. This is critical for orchestration---Docker Compose and Kubernetes use health checks to decide whether to route traffic to a container or restart it.

### Image Size Reduction Progress

| Configuration | Image Size | Notes |
|--------------|-----------|-------|
| Naive (`node:20`, copy all, `npm install`) | ~1.2 GB | Full Node image + all deps + source |
| Add `.dockerignore` | ~1.0 GB | No `node_modules`/`.git` in context |
| Switch to `node:20-slim` | ~450 MB | Smaller base image |
| Multi-stage build (production deps only) | ~280 MB | No dev deps, no TS source |
| Add `npm cache clean` | ~240 MB | Remove npm's download cache |
| Using `node:20-alpine` (optional) | ~160 MB | Alpine Linux base (see caveats below) |

A note on Alpine: The `node:20-alpine` base image uses musl libc instead of glibc. This makes it much smaller (~50MB base vs ~180MB for slim), but some native Node modules fail to compile on musl, and FFmpeg builds may behave differently. For video processing workloads, I recommend sticking with `-slim` unless you have thoroughly tested Alpine with your specific dependencies.

### The Health Endpoint

Your Dockerfile references a `/health` endpoint. Here is what that should look like:

```typescript
// src/routes/health.ts
import { Router, Request, Response } from 'express';
import { prisma } from '../lib/prisma';
import { redis } from '../lib/redis';

const router = Router();

router.get('/health', async (req: Request, res: Response) => {
  const checks = {
    uptime: process.uptime(),
    timestamp: Date.now(),
    database: 'unknown',
    redis: 'unknown',
  };

  try {
    // Check database connectivity
    await prisma.$queryRaw`SELECT 1`;
    checks.database = 'healthy';
  } catch (err) {
    checks.database = 'unhealthy';
  }

  try {
    // Check Redis connectivity
    await redis.ping();
    checks.redis = 'healthy';
  } catch (err) {
    checks.redis = 'unhealthy';
  }

  const isHealthy = checks.database === 'healthy' && checks.redis === 'healthy';
  res.status(isHealthy ? 200 : 503).json(checks);
});

export default router;
```

The health endpoint does not just return 200. It checks every dependency the application needs---database, Redis, and anything else---and returns 503 if any of them are unhealthy. This way, orchestrators know to stop sending traffic to a container whose database connection has died.

---

## Docker Compose for Local Development

Our AI video platform is not a single process. It needs:
- An **API server** (the Node.js/TypeScript application)
- A **worker process** (picks jobs off the queue and calls video generation APIs)
- **Redis** (job queue backing store, caching, pub/sub for real-time status)
- **PostgreSQL** (persistent data: users, projects, generations, billing)

Running these individually is painful. You would need four separate terminal windows, manual coordination of startup order, and hardcoded connection strings. **Docker Compose** is a tool that lets you define all of these services in a single YAML file and start everything with one command.

### The Complete docker-compose.yml

```yaml
# docker-compose.yml
version: '3.8'

services:
  # ==========================================
  # PostgreSQL Database
  # ==========================================
  postgres:
    image: postgres:16-alpine
    container_name: vidgen-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-vidgen}
      POSTGRES_USER: ${POSTGRES_USER:-vidgen_user}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?Database password is required}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-vidgen_user} -d ${POSTGRES_DB:-vidgen}"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ==========================================
  # Redis
  # ==========================================
  redis:
    image: redis:7-alpine
    container_name: vidgen-redis
    restart: unless-stopped
    command: redis-server --requirepass ${REDIS_PASSWORD:?Redis password is required} --maxmemory 256mb --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ==========================================
  # API Server
  # ==========================================
  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: runner
    container_name: vidgen-api
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      NODE_ENV: production
      PORT: 3000
      DATABASE_URL: postgresql://${POSTGRES_USER:-vidgen_user}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-vidgen}
      REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379
      JWT_SECRET: ${JWT_SECRET:?JWT secret is required}
      # Video API keys -- injected from .env, never baked into image
      VEO_API_KEY: ${VEO_API_KEY}
      KLING_API_KEY: ${KLING_API_KEY}
      RUNWAY_API_KEY: ${RUNWAY_API_KEY}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - vidgen-network

  # ==========================================
  # Worker Process
  # ==========================================
  worker:
    build:
      context: .
      dockerfile: Dockerfile
      target: runner
    container_name: vidgen-worker
    restart: unless-stopped
    command: ["node", "dist/worker.js"]
    environment:
      NODE_ENV: production
      DATABASE_URL: postgresql://${POSTGRES_USER:-vidgen_user}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-vidgen}
      REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379
      WORKER_CONCURRENCY: ${WORKER_CONCURRENCY:-3}
      VEO_API_KEY: ${VEO_API_KEY}
      KLING_API_KEY: ${KLING_API_KEY}
      RUNWAY_API_KEY: ${RUNWAY_API_KEY}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - vidgen-network

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local

networks:
  vidgen-network:
    driver: bridge
```

Let me explain each decision.

### Environment Variables and Secrets

Notice that no secrets appear in the `docker-compose.yml` file itself. Every secret is referenced via `${VARIABLE_NAME}`, which Docker Compose reads from a `.env` file in the same directory. The `:?` syntax (e.g., `${POSTGRES_PASSWORD:?Database password is required}`) means "if this variable is not set, abort with this error message." This is a safety net: you cannot accidentally start the stack with missing credentials.

The `.env` file:

```bash
# .env (NEVER commit this file)
POSTGRES_DB=vidgen
POSTGRES_USER=vidgen_user
POSTGRES_PASSWORD=your_secure_password_here
REDIS_PASSWORD=your_redis_password_here
JWT_SECRET=your_jwt_secret_here

# Video generation API keys
VEO_API_KEY=veo_key_here
KLING_API_KEY=kling_key_here
RUNWAY_API_KEY=runway_key_here

# Worker configuration
WORKER_CONCURRENCY=3
```

Add `.env` to your `.gitignore` immediately. Secrets baked into images or committed to version control are the number one cause of security incidents in containerized applications. Commit a `.env.example` file with placeholder values instead.

### Volumes: Why Data Survives Container Restarts

When a container is destroyed, its filesystem is destroyed with it. If PostgreSQL stores data inside the container's filesystem and you run `docker compose down`, your database is gone. **Volumes** solve this.

`postgres_data:/var/lib/postgresql/data` tells Docker to create a **named volume** called `postgres_data` and mount it at the path where PostgreSQL stores its data files. Named volumes persist across container restarts and rebuilds. The data lives on the host machine, managed by Docker.

The `./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql` line is a **bind mount**---it maps a specific file on your host machine into the container. PostgreSQL's official image runs any `.sql` files in `/docker-entrypoint-initdb.d/` when the database is initialized for the first time. This is how you seed your schema.

### depends_on with Health Checks

The `depends_on` section with `condition: service_healthy` ensures that the API server does not start until PostgreSQL and Redis are not just *running* but *healthy*. Without the health condition, Docker only waits for the container to start, which is not the same as the service inside being ready to accept connections. PostgreSQL can take several seconds to initialize, and if your API tries to connect during that window, it crashes.

### Running the Stack

```bash
# Start everything in the background
docker compose up -d

# Watch the logs
docker compose logs -f

# Check health status
docker compose ps

# Stop everything (volumes preserved)
docker compose down

# Stop everything AND delete volumes (destroys data)
docker compose down -v
```

---

## Networking Fundamentals You Must Know

Networking is where most Docker beginners get stuck. The API cannot connect to Redis. Redis works locally but not in Docker. The worker can reach the database but the API cannot. These problems all have the same root cause: misunderstanding how Docker networking works.

### Ports: Host vs Container

A **port** is a number (0--65535) that identifies a specific process on a machine that is listening for network connections. When your Node.js server calls `app.listen(3000)`, it tells the operating system "I want to receive any network traffic arriving on port 3000."

Inside a Docker container, the container has its own network namespace. Port 3000 inside the container is completely separate from port 3000 on your host machine. The `-p 3000:3000` flag in Docker (or the `ports` section in Compose) creates a **port mapping**: traffic arriving at port 3000 on the host is forwarded to port 3000 inside the container.

```
Host machine
+-- Port 80    -->  Nginx container, port 80
+-- Port 3000  -->  API container, port 3000
+-- Port 5432  -->  PostgreSQL container, port 5432
+-- Port 6379  -->  Redis container, port 6379
```

You can also remap ports: `-p 8080:3000` means "traffic on host port 8080 goes to container port 3000."

### Bridge Networks and DNS

When you define a network in Docker Compose (like our `vidgen-network`), Docker creates a **bridge network**. A bridge network is a virtual network switch: all containers attached to it can communicate with each other, and Docker provides built-in DNS resolution so containers can reach each other by **service name**.

In our `docker-compose.yml`, the API connects to PostgreSQL using the connection string:

```
postgresql://vidgen_user:password@postgres:5432/vidgen
```

The hostname `postgres` is resolved by Docker's internal DNS to the IP address of the PostgreSQL container. This only works because both containers are on the same network. If you put the API on one network and PostgreSQL on another, the DNS resolution fails and you get `ECONNREFUSED` or `ENOTFOUND`.

Here is what the network topology looks like:

<svg viewBox="0 0 860 420" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:system-ui,sans-serif">
  <defs>
    <marker id="arr-n" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>

  <text x="430" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#222">Docker Bridge Network: vidgen-network</text>

  <!-- Bridge Network Box -->
  <rect x="30" y="120" width="800" height="250" rx="12" fill="#f5f5f5" stroke="#90caf9" stroke-width="2" stroke-dasharray="8,4"/>
  <text x="60" y="148" font-size="12" fill="#42a5f5" font-weight="bold">vidgen-network (172.18.0.0/16)</text>

  <!-- Host Machine Box -->
  <rect x="10" y="45" width="840" height="365" rx="12" fill="none" stroke="#ccc" stroke-width="1.5" stroke-dasharray="4,4"/>
  <text x="30" y="70" font-size="12" fill="#999">Host Machine</text>

  <!-- API -->
  <rect x="60" y="170" width="150" height="80" rx="8" fill="#e3f2fd" stroke="#42a5f5" stroke-width="2"/>
  <text x="135" y="200" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">API Server</text>
  <text x="135" y="220" text-anchor="middle" font-size="10" fill="#666">172.18.0.3:3000</text>
  <text x="135" y="238" text-anchor="middle" font-size="10" fill="#666">hostname: api</text>

  <!-- Worker -->
  <rect x="240" y="170" width="150" height="80" rx="8" fill="#e8f5e9" stroke="#66bb6a" stroke-width="2"/>
  <text x="315" y="200" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Worker</text>
  <text x="315" y="220" text-anchor="middle" font-size="10" fill="#666">172.18.0.4</text>
  <text x="315" y="238" text-anchor="middle" font-size="10" fill="#666">hostname: worker</text>

  <!-- Redis -->
  <rect x="470" y="170" width="150" height="80" rx="8" fill="#ffebee" stroke="#ef5350" stroke-width="2"/>
  <text x="545" y="200" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Redis</text>
  <text x="545" y="220" text-anchor="middle" font-size="10" fill="#666">172.18.0.5:6379</text>
  <text x="545" y="238" text-anchor="middle" font-size="10" fill="#666">hostname: redis</text>

  <!-- PostgreSQL -->
  <rect x="650" y="170" width="150" height="80" rx="8" fill="#fff3e0" stroke="#ffa726" stroke-width="2"/>
  <text x="725" y="200" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">PostgreSQL</text>
  <text x="725" y="220" text-anchor="middle" font-size="10" fill="#666">172.18.0.6:5432</text>
  <text x="725" y="238" text-anchor="middle" font-size="10" fill="#666">hostname: postgres</text>

  <!-- Connections from API -->
  <line x1="210" y1="210" x2="240" y2="210" stroke="#333" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="210" y1="215" x2="470" y2="215" stroke="#333" stroke-width="1.5" marker-end="url(#arr-n)"/>
  <text x="340" y="208" text-anchor="middle" font-size="9" fill="#666">redis://redis:6379</text>

  <line x1="135" y1="250" x2="135" y2="290" stroke="#333" stroke-width="1"/>
  <line x1="135" y1="290" x2="725" y2="290" stroke="#333" stroke-width="1.5"/>
  <line x1="725" y1="290" x2="725" y2="250" stroke="#333" stroke-width="1.5" marker-end="url(#arr-n)"/>
  <text x="430" y="305" text-anchor="middle" font-size="9" fill="#666">postgresql://postgres:5432</text>

  <!-- Worker connections -->
  <line x1="390" y1="200" x2="470" y2="200" stroke="#66bb6a" stroke-width="1.5" marker-end="url(#arr-n)"/>
  <line x1="390" y1="230" x2="430" y2="230" stroke="#66bb6a" stroke-width="1"/>
  <line x1="430" y1="230" x2="430" y2="340" stroke="#66bb6a" stroke-width="1"/>
  <line x1="430" y1="340" x2="725" y2="340" stroke="#66bb6a" stroke-width="1"/>
  <line x1="725" y1="340" x2="725" y2="250" stroke="#66bb6a" stroke-width="1" stroke-dasharray="3,3"/>

  <!-- Port mapping from host -->
  <rect x="80" y="80" width="110" height="30" rx="4" fill="#fff" stroke="#42a5f5" stroke-width="1.5"/>
  <text x="135" y="100" text-anchor="middle" font-size="10" fill="#333">Host :3000</text>
  <line x1="135" y1="110" x2="135" y2="170" stroke="#42a5f5" stroke-width="1.5" marker-end="url(#arr-n)"/>

  <!-- Docker DNS -->
  <rect x="280" y="370" width="300" height="25" rx="4" fill="#e1f5fe" stroke="#4fc3f7" stroke-width="1"/>
  <text x="430" y="387" text-anchor="middle" font-size="10" fill="#0277bd">Docker DNS: service names resolve to container IPs</text>
</svg>

### The Most Common Networking Mistake

Here is the scenario that traps nearly everyone. You set up Redis locally, and in your `.env` file you have:

```
REDIS_URL=redis://localhost:6379
```

This works on your machine because Redis is running on `localhost`. But inside the API container, `localhost` refers to *the container itself*, not the host machine. The API container does not have Redis running inside it. The connection fails.

The fix: use the service name as the hostname. Inside Docker Compose, `redis` resolves to the Redis container's IP. Your connection string becomes:

```
REDIS_URL=redis://:password@redis:6379
```

This is why the `docker-compose.yml` above uses `@postgres:5432` and `@redis:6379` instead of `@localhost`.

---

## Reverse Proxies: Nginx

At this point we have a containerized application that runs locally. But in production, you do not expose your Node.js application directly to the internet. You put a **reverse proxy** in front of it.

### What Is a Reverse Proxy

A **proxy** is something that acts on behalf of something else. A **forward proxy** acts on behalf of clients (like a VPN---your requests go through the proxy before reaching the server). A **reverse proxy** acts on behalf of servers---client requests hit the proxy first, and the proxy decides which backend server to forward them to.

Why do you need one? Several reasons, all critical for a production video platform:

| Function | Why It Matters |
|----------|---------------|
| **TLS termination** | Handle HTTPS encryption/decryption at the proxy. Your backend speaks plain HTTP internally, which is simpler and faster. |
| **Load balancing** | Distribute requests across multiple API server instances. |
| **Static file serving** | Serve images, videos, CSS, and JavaScript directly from the proxy without hitting your application. Nginx serves static files 10--50x faster than Node.js. |
| **Rate limiting** | Protect your backend from abuse. Limit requests per IP per second. |
| **Request buffering** | Nginx receives the full client request before forwarding it, protecting your backend from slow clients tying up connections. |
| **WebSocket proxying** | Forward WebSocket connections (for real-time generation status updates) to your backend. |
| **Security headers** | Add security headers (HSTS, CSP, X-Frame-Options) in one place rather than in every backend. |

### Full Nginx Configuration

Here is a complete, annotated Nginx configuration for our AI video platform:

```nginx
# /etc/nginx/nginx.conf

# Run as a non-root user
user nginx;

# Auto-detect the number of CPU cores
worker_processes auto;

# Error log location
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    # Maximum simultaneous connections per worker
    worker_connections 1024;
}

http {
    # ==========================================
    # Basic Settings
    # ==========================================
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging format with request timing
    log_format main '$remote_addr - $remote_user [$time_local] '
                    '"$request" $status $body_bytes_sent '
                    '"$http_referer" "$http_user_agent" '
                    'rt=$request_time';

    access_log /var/log/nginx/access.log main;

    # Performance: send files directly from kernel space
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;

    # Keep connections alive for 65 seconds
    keepalive_timeout 65;

    # Hide Nginx version in response headers (security)
    server_tokens off;

    # Gzip compression for text-based responses
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css application/json application/javascript
               text/xml application/xml application/xml+rss text/javascript
               image/svg+xml;

    # ==========================================
    # Rate Limiting Zones
    # ==========================================
    # 10 requests per second per IP for general API endpoints
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    # 2 requests per second per IP for video generation (expensive)
    limit_req_zone $binary_remote_addr zone=generate:10m rate=2r/s;

    # ==========================================
    # Upstream Backend Servers
    # ==========================================
    upstream api_backend {
        # If you scale to multiple API containers, add them here
        server api:3000;
        # server api-2:3000;
        # server api-3:3000;

        # Keep persistent connections to the backend
        keepalive 32;
    }

    # ==========================================
    # Redirect HTTP to HTTPS
    # ==========================================
    server {
        listen 80;
        server_name vidgen.example.com;

        # Allow Let's Encrypt challenge verification over HTTP
        location /.well-known/acme-challenge/ {
            root /var/www/certbot;
        }

        # Redirect everything else to HTTPS
        location / {
            return 301 https://$host$request_uri;
        }
    }

    # ==========================================
    # Main HTTPS Server
    # ==========================================
    server {
        listen 443 ssl http2;
        server_name vidgen.example.com;

        # ======================================
        # SSL Configuration
        # ======================================
        ssl_certificate /etc/letsencrypt/live/vidgen.example.com/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/vidgen.example.com/privkey.pem;

        # Modern SSL settings
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;

        # OCSP stapling for faster SSL handshakes
        ssl_stapling on;
        ssl_stapling_verify on;
        resolver 8.8.8.8 8.8.4.4 valid=300s;

        # ======================================
        # Security Headers
        # ======================================
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-Frame-Options "DENY" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "strict-origin-when-cross-origin" always;

        # ======================================
        # Client Upload Size (for reference images/videos)
        # ======================================
        client_max_body_size 500M;

        # ======================================
        # Static Assets
        # ======================================
        location /static/ {
            alias /var/www/vidgen/static/;
            expires 30d;
            add_header Cache-Control "public, immutable";
            access_log off;
        }

        # Generated video files served directly by Nginx
        location /media/ {
            alias /var/www/vidgen/media/;
            expires 7d;
            add_header Cache-Control "public";
            # Prevent hotlinking
            valid_referers none blocked vidgen.example.com;
            if ($invalid_referer) {
                return 403;
            }
        }

        # ======================================
        # WebSocket Endpoint (generation status)
        # ======================================
        location /ws {
            proxy_pass http://api_backend;
            proxy_http_version 1.1;

            # These two headers upgrade HTTP to WebSocket
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";

            # Pass real client IP to backend
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # WebSocket connections can be long-lived
            proxy_read_timeout 86400s;
            proxy_send_timeout 86400s;
        }

        # ======================================
        # Video Generation API (rate limited)
        # ======================================
        location /api/generate {
            limit_req zone=generate burst=5 nodelay;

            proxy_pass http://api_backend;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Connection "";
        }

        # ======================================
        # All Other API Routes
        # ======================================
        location /api/ {
            limit_req zone=api burst=20 nodelay;

            proxy_pass http://api_backend;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Connection "";

            # Timeouts for slow API operations
            proxy_connect_timeout 10s;
            proxy_send_timeout 30s;
            proxy_read_timeout 60s;
        }

        # ======================================
        # Health Check (no rate limiting)
        # ======================================
        location /health {
            proxy_pass http://api_backend;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            access_log off;
        }
    }
}
```

Let me explain the parts that are not obvious.

**`upstream api_backend`**: This block defines the pool of backend servers. Right now it is a single server (`api:3000`), but when you need to scale, you add more entries and Nginx round-robins between them. The `keepalive 32` directive maintains a pool of persistent connections to the backend, avoiding the overhead of opening a new TCP connection for every request.

**WebSocket proxying**: WebSocket connections start as regular HTTP requests and then "upgrade" to the WebSocket protocol. For Nginx to handle this, it needs to forward the `Upgrade` and `Connection` headers. Without `proxy_set_header Upgrade $http_upgrade` and `proxy_set_header Connection "upgrade"`, the WebSocket handshake fails silently and your real-time generation status updates will not work. The long `proxy_read_timeout` (86400 seconds = 24 hours) is needed because WebSocket connections are long-lived---the default 60-second timeout would kill the connection prematurely.

**Rate limiting**: The `/api/generate` endpoint is rate-limited to 2 requests per second per IP. This is because each video generation costs real money (API calls to Veo, Kling, or Runway can cost $0.05--$0.50 each). The `burst=5` parameter allows short bursts above the rate (up to 5 queued requests), and `nodelay` processes burst requests immediately rather than spacing them out.

**Static file serving**: Nginx serves files from `/static/` and `/media/` directly, without forwarding the request to the Node.js backend. Nginx is purpose-built for serving static files. It uses `sendfile` to transfer files directly from the kernel's file cache to the network socket, bypassing userspace entirely. Node.js would read the file into a buffer, then write it to the response---orders of magnitude slower for large video files.

### Adding Nginx to Docker Compose

Add this service to your `docker-compose.yml`:

```yaml
  # ==========================================
  # Nginx Reverse Proxy
  # ==========================================
  nginx:
    image: nginx:1.25-alpine
    container_name: vidgen-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/letsencrypt:ro
      - ./static:/var/www/vidgen/static:ro
      - ./media:/var/www/vidgen/media:ro
      - certbot_webroot:/var/www/certbot
    depends_on:
      api:
        condition: service_healthy
    networks:
      - vidgen-network
```

The `:ro` suffix on volumes means "read-only"---the Nginx container can read the configuration and certificate files but cannot modify them. This is a security best practice: even if the Nginx process is compromised, it cannot alter its own configuration or your TLS private keys.

---

## TLS and HTTPS

**TLS** (Transport Layer Security) encrypts the connection between the user's browser and your server. The older name for TLS is SSL (Secure Sockets Layer), and you will see both terms used interchangeably, though SSL is technically deprecated. **HTTPS** is simply HTTP over TLS---the same request-response protocol, but encrypted.

You might think "my API just returns JSON, encryption is overkill." It is not, for several reasons:

1. **Security**: Without TLS, anyone on the same network (coffee shop WiFi, corporate network, ISP) can read every request and response in plain text. That includes authentication tokens, user data, and API keys sent in headers.
2. **HTTP/2**: Browsers only use HTTP/2 over TLS. HTTP/2 multiplexes multiple requests over a single TCP connection, which is significantly faster for pages that make many API calls (like a dashboard polling generation status).
3. **Browser trust**: Modern browsers flag HTTP sites as "Not Secure" in the address bar. For a paid SaaS product, this kills user trust instantly.
4. **SEO**: Google's search ranking algorithm factors in HTTPS. HTTP-only sites rank lower.
5. **WebSocket security**: The `wss://` (WebSocket Secure) protocol requires TLS. Without it, many corporate firewalls and proxy servers block WebSocket connections entirely.

### Let's Encrypt with Certbot

**Let's Encrypt** is a nonprofit certificate authority (CA) that issues TLS certificates for free, with automated validation. Before Let's Encrypt, TLS certificates cost $50--$300 per year and required manual verification. **Certbot** is the official command-line tool that automates the process of obtaining and renewing Let's Encrypt certificates.

The flow works like this:

<svg viewBox="0 0 860 350" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:system-ui,sans-serif">
  <defs>
    <marker id="arr-t" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>

  <text x="430" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#222">TLS Certificate Flow: DNS + Certbot + Nginx</text>

  <!-- Step 1: Domain registrar -->
  <rect x="30" y="60" width="150" height="70" rx="8" fill="#e3f2fd" stroke="#42a5f5" stroke-width="2"/>
  <text x="105" y="88" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">1. DNS Provider</text>
  <text x="105" y="108" text-anchor="middle" font-size="10" fill="#666">A record points</text>
  <text x="105" y="120" text-anchor="middle" font-size="10" fill="#666">domain to server IP</text>

  <!-- Step 2: Certbot requests -->
  <rect x="220" y="60" width="150" height="70" rx="8" fill="#e8f5e9" stroke="#66bb6a" stroke-width="2"/>
  <text x="295" y="88" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">2. Certbot</text>
  <text x="295" y="108" text-anchor="middle" font-size="10" fill="#666">Requests certificate</text>
  <text x="295" y="120" text-anchor="middle" font-size="10" fill="#666">from Let's Encrypt</text>

  <!-- Step 3: Challenge -->
  <rect x="410" y="60" width="150" height="70" rx="8" fill="#fff3e0" stroke="#ffa726" stroke-width="2"/>
  <text x="485" y="88" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">3. HTTP Challenge</text>
  <text x="485" y="108" text-anchor="middle" font-size="10" fill="#666">LE verifies you own</text>
  <text x="485" y="120" text-anchor="middle" font-size="10" fill="#666">the domain via HTTP</text>

  <!-- Step 4: Certificate issued -->
  <rect x="600" y="60" width="150" height="70" rx="8" fill="#e8eaf6" stroke="#7986cb" stroke-width="2"/>
  <text x="675" y="88" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">4. Cert Issued</text>
  <text x="675" y="108" text-anchor="middle" font-size="10" fill="#666">fullchain.pem +</text>
  <text x="675" y="120" text-anchor="middle" font-size="10" fill="#666">privkey.pem saved</text>

  <!-- Arrows -->
  <line x1="180" y1="95" x2="220" y2="95" stroke="#333" stroke-width="1.5" marker-end="url(#arr-t)"/>
  <line x1="370" y1="95" x2="410" y2="95" stroke="#333" stroke-width="1.5" marker-end="url(#arr-t)"/>
  <line x1="560" y1="95" x2="600" y2="95" stroke="#333" stroke-width="1.5" marker-end="url(#arr-t)"/>

  <!-- Nginx uses certificate -->
  <rect x="350" y="190" width="160" height="70" rx="8" fill="#ffebee" stroke="#ef5350" stroke-width="2"/>
  <text x="430" y="218" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">5. Nginx</text>
  <text x="430" y="238" text-anchor="middle" font-size="10" fill="#666">Loads certs, serves</text>
  <text x="430" y="250" text-anchor="middle" font-size="10" fill="#666">HTTPS on port 443</text>

  <line x1="675" y1="130" x2="675" y2="160" stroke="#333" stroke-width="1"/>
  <line x1="675" y1="160" x2="430" y2="160" stroke="#333" stroke-width="1"/>
  <line x1="430" y1="160" x2="430" y2="190" stroke="#333" stroke-width="1.5" marker-end="url(#arr-t)"/>

  <!-- Users connect -->
  <rect x="100" y="190" width="150" height="70" rx="8" fill="#f5f5f5" stroke="#999" stroke-width="2"/>
  <text x="175" y="218" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">6. Users</text>
  <text x="175" y="238" text-anchor="middle" font-size="10" fill="#666">Connect via HTTPS</text>
  <text x="175" y="250" text-anchor="middle" font-size="10" fill="#666">TLS encrypted</text>

  <line x1="250" y1="225" x2="350" y2="225" stroke="#333" stroke-width="1.5" marker-end="url(#arr-t)"/>

  <!-- Auto-renewal -->
  <rect x="600" y="190" width="150" height="70" rx="8" fill="#e8f5e9" stroke="#66bb6a" stroke-width="2"/>
  <text x="675" y="218" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">7. Auto-Renew</text>
  <text x="675" y="238" text-anchor="middle" font-size="10" fill="#666">Cron job renews</text>
  <text x="675" y="250" text-anchor="middle" font-size="10" fill="#666">every 60 days</text>

  <line x1="510" y1="225" x2="600" y2="225" stroke="#66bb6a" stroke-width="1.5" marker-end="url(#arr-t)"/>

  <!-- Note -->
  <rect x="100" y="290" width="660" height="40" rx="6" fill="#f8f9fa" stroke="#ddd" stroke-width="1"/>
  <text x="430" y="315" text-anchor="middle" font-size="11" fill="#555">Let's Encrypt certs expire after 90 days. Certbot auto-renewal runs twice daily and renews when less than 30 days remain.</text>
</svg>

### Step-by-Step Setup

**1. Point your domain to your server**

In your DNS provider (Cloudflare, Namecheap, Route53, etc.), create an **A record**. An A record maps a domain name to an IPv4 address:

```
Type: A
Name: vidgen (or @ for root domain)
Value: 203.0.113.42 (your server's public IP)
TTL: 300
```

Wait for DNS propagation (usually 5--30 minutes, but can take up to 48 hours). Verify with:

```bash
dig +short vidgen.example.com
# Should return: 203.0.113.42
```

**2. Install Certbot and obtain certificates**

```bash
# Install Certbot
sudo apt update
sudo apt install -y certbot

# Obtain certificate using the webroot method.
# Nginx must be running and serving the .well-known directory on port 80.
sudo certbot certonly \
  --webroot \
  --webroot-path /var/www/certbot \
  -d vidgen.example.com \
  --email you@example.com \
  --agree-tos \
  --non-interactive
```

The `--webroot` method works by placing a challenge file in `/var/www/certbot/.well-known/acme-challenge/`. Let's Encrypt then makes an HTTP request to `http://vidgen.example.com/.well-known/acme-challenge/<token>` to verify you control the domain. This is why our Nginx config has a `location /.well-known/acme-challenge/` block that serves from that directory---without it, the challenge request would be redirected to HTTPS (which does not work yet because we do not have a certificate yet).

After success, your certificates are at:
```
/etc/letsencrypt/live/vidgen.example.com/fullchain.pem  (certificate + chain)
/etc/letsencrypt/live/vidgen.example.com/privkey.pem     (private key)
```

**3. Set up auto-renewal**

Let's Encrypt certificates expire after 90 days. Certbot includes an auto-renewal mechanism. You run `certbot renew` periodically, and it checks whether any certificates are within 30 days of expiry. If they are, it renews them automatically.

```bash
# Test renewal (dry run, does not actually renew)
sudo certbot renew --dry-run

# Add a cron job for automatic renewal
sudo crontab -e
```

Add this line:

```
0 3 * * * certbot renew --quiet --deploy-hook "docker exec vidgen-nginx nginx -s reload"
```

This runs at 3:00 AM daily. The `--deploy-hook` runs *only* when a certificate is actually renewed---it tells Nginx to reload its configuration and pick up the new certificate files. The reload is graceful: existing connections continue uninterrupted while new connections use the new certificate.

---

## Firewall Configuration

Your server has dozens of ports that could be listening for connections. PostgreSQL on 5432, Redis on 6379, the Docker daemon, SSH on 22, and potentially many others. Every port reachable from the internet is an attack surface. A **firewall** is a program that blocks network traffic except on ports you explicitly allow.

### UFW (Uncomplicated Firewall)

**UFW** is a user-friendly frontend for Linux's `iptables` firewall. `iptables` is powerful but notoriously difficult to configure correctly. UFW wraps it in simple commands.

```bash
# Install UFW (usually pre-installed on Ubuntu)
sudo apt install -y ufw

# Default policy: deny all incoming traffic, allow all outgoing
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (CRITICAL: do this BEFORE enabling the firewall,
# or you will lock yourself out of your own server)
sudo ufw allow 22/tcp comment 'SSH'

# Allow HTTP (needed for Let's Encrypt challenges and HTTPS redirect)
sudo ufw allow 80/tcp comment 'HTTP'

# Allow HTTPS (the only port end users should reach)
sudo ufw allow 443/tcp comment 'HTTPS'

# Enable the firewall
sudo ufw enable

# Verify the rules
sudo ufw status verbose
```

The output should look like:

```
Status: active
Logging: on (low)
Default: deny (incoming), allow (outgoing), disabled (routed)

To                         Action      From
--                         ------      ----
22/tcp                     ALLOW IN    Anywhere       # SSH
80/tcp                     ALLOW IN    Anywhere       # HTTP
443/tcp                    ALLOW IN    Anywhere       # HTTPS
22/tcp (v6)                ALLOW IN    Anywhere (v6)  # SSH
80/tcp (v6)                ALLOW IN    Anywhere (v6)  # HTTP
443/tcp (v6)               ALLOW IN    Anywhere (v6)  # HTTPS
```

Why is this critical? Without a firewall, your PostgreSQL port (5432) is reachable from the internet. Even though PostgreSQL requires authentication, it is constantly being probed by automated bots trying default credentials, brute-forcing passwords, and exploiting known vulnerabilities. Same with Redis on port 6379---and many Redis instances deployed with Docker have no authentication at all. By only exposing 22 (SSH), 80 (HTTP), and 443 (HTTPS), you reduce your attack surface to the absolute minimum.

### Docker and UFW: A Crucial Gotcha

There is a well-known and dangerous interaction between Docker and UFW. Docker manipulates `iptables` directly to set up its port mappings, and these rules **bypass UFW entirely**. This means that even with UFW configured to block port 5432, if you have `ports: - "5432:5432"` in your `docker-compose.yml`, PostgreSQL is accessible from the internet.

The fix is simple: do not publish ports that should not be accessible from outside. In production, only Nginx needs published ports. The API server, workers, PostgreSQL, and Redis communicate over the internal Docker bridge network and do not need port mappings at all.

Update your production `docker-compose.yml` to remove unnecessary port mappings:

```yaml
services:
  postgres:
    # REMOVE the ports section in production
    # ports:
    #   - "5432:5432"
    ...

  redis:
    # REMOVE the ports section in production
    # ports:
    #   - "6379:6379"
    ...

  api:
    # REMOVE the ports section in production
    # Nginx proxies to api:3000 over the internal Docker network
    # ports:
    #   - "3000:3000"
    ...

  nginx:
    # ONLY Nginx publishes ports to the host
    ports:
      - "80:80"
      - "443:443"
    ...
```

The only container that publishes ports to the host is Nginx. Everything else communicates internally over the Docker bridge network, invisible to the outside world.

---

## CI/CD with GitHub Actions

At this point you can deploy manually: SSH into the server, pull the latest code, rebuild the Docker images, restart the containers. This works for a prototype. It does not work for a production system where you need to deploy multiple times per day and cannot afford to SSH into a server every time.

**CI/CD** stands for Continuous Integration / Continuous Deployment:
- **Continuous Integration (CI)**: Every push to the repository triggers automated builds and tests. Broken code is caught before it reaches production.
- **Continuous Deployment (CD)**: After CI passes, the application is automatically deployed to production without manual intervention.

**GitHub Actions** is GitHub's built-in CI/CD system. You define workflows as YAML files in a `.github/workflows/` directory in your repository. Each workflow is triggered by events (push, pull request, tag, schedule) and runs a series of steps on GitHub's cloud-hosted runners (Ubuntu VMs with common tools pre-installed).

### The Complete CI/CD Pipeline

Here is the pipeline we are building:

<svg viewBox="0 0 900 200" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:system-ui,sans-serif">
  <defs>
    <marker id="arr-ci" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>

  <text x="450" y="25" text-anchor="middle" font-size="15" font-weight="bold" fill="#222">CI/CD Pipeline: Push to Production in 5 Minutes</text>

  <!-- Git Push -->
  <rect x="20" y="60" width="110" height="60" rx="8" fill="#f5f5f5" stroke="#999" stroke-width="2"/>
  <text x="75" y="85" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">Git Push</text>
  <text x="75" y="102" text-anchor="middle" font-size="10" fill="#666">to main</text>

  <!-- Lint + Type Check -->
  <rect x="165" y="60" width="120" height="60" rx="8" fill="#e3f2fd" stroke="#42a5f5" stroke-width="2"/>
  <text x="225" y="85" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">Lint + Types</text>
  <text x="225" y="102" text-anchor="middle" font-size="10" fill="#666">eslint, tsc</text>

  <!-- Test -->
  <rect x="320" y="60" width="110" height="60" rx="8" fill="#e8f5e9" stroke="#66bb6a" stroke-width="2"/>
  <text x="375" y="85" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">Test</text>
  <text x="375" y="102" text-anchor="middle" font-size="10" fill="#666">jest / vitest</text>

  <!-- Build Docker Image -->
  <rect x="465" y="60" width="120" height="60" rx="8" fill="#fff3e0" stroke="#ffa726" stroke-width="2"/>
  <text x="525" y="85" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">Build Image</text>
  <text x="525" y="102" text-anchor="middle" font-size="10" fill="#666">docker build</text>

  <!-- Push to Registry -->
  <rect x="620" y="60" width="120" height="60" rx="8" fill="#e8eaf6" stroke="#7986cb" stroke-width="2"/>
  <text x="680" y="85" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">Push Image</text>
  <text x="680" y="102" text-anchor="middle" font-size="10" fill="#666">ghcr.io</text>

  <!-- Deploy to VPS -->
  <rect x="775" y="60" width="110" height="60" rx="8" fill="#ffebee" stroke="#ef5350" stroke-width="2"/>
  <text x="830" y="85" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">Deploy</text>
  <text x="830" y="102" text-anchor="middle" font-size="10" fill="#666">SSH to VPS</text>

  <!-- Arrows -->
  <line x1="130" y1="90" x2="165" y2="90" stroke="#333" stroke-width="1.5" marker-end="url(#arr-ci)"/>
  <line x1="285" y1="90" x2="320" y2="90" stroke="#333" stroke-width="1.5" marker-end="url(#arr-ci)"/>
  <line x1="430" y1="90" x2="465" y2="90" stroke="#333" stroke-width="1.5" marker-end="url(#arr-ci)"/>
  <line x1="585" y1="90" x2="620" y2="90" stroke="#333" stroke-width="1.5" marker-end="url(#arr-ci)"/>
  <line x1="740" y1="90" x2="775" y2="90" stroke="#333" stroke-width="1.5" marker-end="url(#arr-ci)"/>

  <!-- Time annotations -->
  <text x="225" y="145" text-anchor="middle" font-size="10" fill="#888">~30s</text>
  <text x="375" y="145" text-anchor="middle" font-size="10" fill="#888">~60s</text>
  <text x="525" y="145" text-anchor="middle" font-size="10" fill="#888">~90s</text>
  <text x="680" y="145" text-anchor="middle" font-size="10" fill="#888">~30s</text>
  <text x="830" y="145" text-anchor="middle" font-size="10" fill="#888">~45s</text>

  <text x="450" y="180" text-anchor="middle" font-size="11" fill="#555">Total pipeline: approximately 4-5 minutes from push to production</text>
</svg>

### The Workflow File

```yaml
# .github/workflows/deploy.yml
name: Build, Test, and Deploy

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # ==========================================
  # Job 1: Lint and Type Check
  # ==========================================
  lint:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run ESLint
        run: npm run lint

      - name: Run TypeScript type check
        run: npx tsc --noEmit

  # ==========================================
  # Job 2: Run Tests
  # ==========================================
  test:
    runs-on: ubuntu-latest
    needs: lint

    services:
      # Spin up PostgreSQL for integration tests
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_DB: vidgen_test
          POSTGRES_USER: test_user
          POSTGRES_PASSWORD: test_password
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      # Spin up Redis for integration tests
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run database migrations
        run: npx prisma migrate deploy
        env:
          DATABASE_URL: postgresql://test_user:test_password@localhost:5432/vidgen_test

      - name: Run tests
        run: npm test -- --coverage
        env:
          DATABASE_URL: postgresql://test_user:test_password@localhost:5432/vidgen_test
          REDIS_URL: redis://localhost:6379
          JWT_SECRET: test-secret-do-not-use-in-production

      - name: Upload coverage report
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: coverage/

  # ==========================================
  # Job 3: Build and Push Docker Image
  # ==========================================
  build:
    runs-on: ubuntu-latest
    needs: test
    # Only build and push on push to main, not on PRs
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    permissions:
      contents: read
      packages: write

    outputs:
      image_tag: ${{ steps.meta.outputs.tags }}

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata for Docker
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=
            type=raw,value=latest

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ==========================================
  # Job 4: Deploy to VPS
  # ==========================================
  deploy:
    runs-on: ubuntu-latest
    needs: build
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    steps:
      - name: Deploy to production server via SSH
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.VPS_HOST }}
          username: ${{ secrets.VPS_USER }}
          key: ${{ secrets.VPS_SSH_KEY }}
          script: |
            set -e

            # Navigate to application directory
            cd /opt/vidgen

            # Login to GitHub Container Registry
            echo ${{ secrets.GHCR_TOKEN }} | docker login ghcr.io -u ${{ github.actor }} --password-stdin

            # Pull the latest image
            docker pull ghcr.io/${{ github.repository }}:latest

            # Rolling update: restart API and worker with new image
            # --no-deps prevents restarting postgres/redis/nginx
            docker compose pull api worker
            docker compose up -d --no-deps api worker

            # Wait for the API container to pass health checks
            echo "Waiting for API to become healthy..."
            for i in $(seq 1 30); do
              if docker inspect --format='{{.State.Health.Status}}' vidgen-api 2>/dev/null | grep -q healthy; then
                echo "API is healthy after ${i} checks."
                break
              fi
              if [ "$i" -eq 30 ]; then
                echo "ERROR: API failed to become healthy within 60 seconds."
                echo "Rolling back..."
                docker compose up -d --no-deps api worker
                exit 1
              fi
              sleep 2
            done

            # Clean up old, unused Docker images
            docker image prune -f

            echo "Deployment complete!"
```

Let me walk through the critical design decisions.

### Jobs and Dependencies

The workflow has four jobs: `lint`, `test`, `build`, and `deploy`. The `needs` keyword creates a dependency chain:

```
lint --> test --> build --> deploy
```

If linting fails, tests never run. If tests fail, no Docker image is built. If the image build fails, there is no deployment. This prevents broken code from reaching production. Each step is a gate: you must pass to proceed.

### Service Containers

GitHub Actions can spin up Docker containers alongside your test runner. The `services` section in the `test` job starts PostgreSQL and Redis containers that are accessible at `localhost` from the test runner. This means your integration tests run against *real* PostgreSQL and Redis instances, not mocks. This is critical for catching issues like bad SQL queries, incorrect Redis commands, or connection handling bugs that mocks would not reveal.

### Docker Layer Caching with GitHub Actions

The `cache-from: type=gha` and `cache-to: type=gha,mode=max` lines in the build step use GitHub Actions' built-in cache to store Docker build layers between workflow runs. Without caching, every build starts from scratch: pull the base image, install all npm dependencies, compile TypeScript. With caching, a build where only your source code changed (but dependencies did not) can complete in under 30 seconds instead of 2--3 minutes, because the `npm ci` layer is cached.

### Secrets Configuration

The workflow references several secrets. These are configured in your GitHub repository under Settings, then Secrets and variables, then Actions. They are encrypted at rest and masked in build logs so they never appear in plain text.

| Secret | Value | How to Generate |
|--------|-------|-----------------|
| `VPS_HOST` | Server IP or hostname | Your VPS provider's dashboard |
| `VPS_USER` | SSH username | `sudo adduser deploy` on the server |
| `VPS_SSH_KEY` | Private SSH key (full PEM content) | `ssh-keygen -t ed25519 -f deploy_key` |
| `GHCR_TOKEN` | GitHub personal access token | GitHub Settings: Developer settings: Personal access tokens |

The SSH key pair: generate it locally with `ssh-keygen`, paste the *private* key into the GitHub secret, and add the *public* key to `~/.ssh/authorized_keys` on the server for the deploy user.

### The Deployment Strategy

The deploy script uses `docker compose up -d --no-deps api worker` to restart only the API and worker containers without touching PostgreSQL, Redis, or Nginx. The `--no-deps` flag is important: without it, Docker Compose would also recreate the database and Redis containers, causing a brief outage and potentially losing in-memory data.

The health check loop waits up to 60 seconds (30 iterations x 2 seconds) for the new API container to become healthy. If it fails, the script outputs an error. In a more sophisticated setup, you would keep track of the previous image tag and roll back to it explicitly. For a single-server deployment, this approach provides near-zero downtime---typically under 2 seconds of interrupted service while Docker stops the old container and starts the new one.

---

## The Deployment Checklist

Here is the checklist I go through before every production deployment. Print it. Pin it to your wall. Do not skip items.

### Container Security

- [ ] **Non-root user**: Container runs as a non-root user (`USER appuser` in Dockerfile)
- [ ] **Read-only volumes**: Configuration and certificate mounts use `:ro` flag
- [ ] **No secrets in images**: API keys and passwords come from environment variables at runtime, never baked in at build time
- [ ] **Minimal base image**: Using `-slim` or `-alpine`, not the full base image
- [ ] **Pinned versions**: Base image uses a specific tag (`node:20-slim`), not `latest`
- [ ] **`.dockerignore`**: Excludes `.env`, `.git`, `node_modules`, test files, documentation

### Health and Observability

- [ ] **Health check endpoint**: `/health` verifies database and Redis connectivity, returns 503 if either is down
- [ ] **Docker HEALTHCHECK**: Defined in Dockerfile with appropriate interval (30s), timeout (5s), start period (10s), and retries (3)
- [ ] **Structured logging**: Application emits logs in JSON format for log aggregation tools
- [ ] **Error tracking**: Sentry or equivalent configured and tested for production environment
- [ ] **Uptime monitoring**: External service (UptimeRobot, Better Stack, Checkly) pings `/health` every 60 seconds

### Networking and Security

- [ ] **Firewall enabled**: UFW active, allowing only ports 22, 80, and 443
- [ ] **No unnecessary port mappings**: Only Nginx publishes ports to the host; database and Redis are internal only
- [ ] **TLS configured**: Let's Encrypt certificate installed, Nginx serving HTTPS with modern cipher suites
- [ ] **Certificate auto-renewal**: Certbot cron job tested with `--dry-run` and deploy hook configured
- [ ] **Security headers**: HSTS, X-Content-Type-Options, X-Frame-Options, Referrer-Policy set in Nginx
- [ ] **Rate limiting**: Nginx rate limits configured for API endpoints and generation endpoints

### CI/CD Pipeline

- [ ] **Automated linting**: ESLint and TypeScript type checking run on every push
- [ ] **Automated tests**: Test suite with integration tests runs on every push, with real database and Redis
- [ ] **Automated image build**: Docker image built and pushed to registry on merge to main
- [ ] **Automated deployment**: Deploy triggered automatically after successful build
- [ ] **Rollback plan**: Know the previous image tag and how to revert with one command
- [ ] **Secrets managed**: All credentials stored in GitHub Actions secrets, never in source code

### Data and Backup

- [ ] **Database backups**: Automated daily `pg_dump` to compressed files
- [ ] **Backup verification**: Periodically restore from a backup to confirm it actually works
- [ ] **Volume persistence**: PostgreSQL and Redis data on named Docker volumes
- [ ] **Off-site backup**: Backups copied to object storage (S3, R2, GCS), not just local disk
- [ ] **Media storage**: Generated videos stored on cloud object storage, not the application server's local disk

### Production Database Backup Script

```bash
#!/bin/bash
# scripts/backup-db.sh
# Run daily via cron: 0 2 * * * /opt/vidgen/scripts/backup-db.sh

set -euo pipefail

BACKUP_DIR="/opt/vidgen/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/vidgen_${TIMESTAMP}.sql.gz"
RETENTION_DAYS=30

# Create backup directory if it does not exist
mkdir -p "${BACKUP_DIR}"

# Dump the database from the running PostgreSQL container
docker exec vidgen-postgres pg_dump \
  -U vidgen_user \
  -d vidgen \
  --no-owner \
  --no-privileges \
  | gzip > "${BACKUP_FILE}"

# Verify the backup file is not empty
if [ ! -s "${BACKUP_FILE}" ]; then
  echo "ERROR: Backup file is empty! Something went wrong."
  exit 1
fi

BACKUP_SIZE=$(du -h "${BACKUP_FILE}" | cut -f1)
echo "Backup created: ${BACKUP_FILE} (${BACKUP_SIZE})"

# Delete backups older than the retention period
find "${BACKUP_DIR}" -name "vidgen_*.sql.gz" -mtime +${RETENTION_DAYS} -delete
echo "Cleaned up backups older than ${RETENTION_DAYS} days."

# Optional: upload to cloud storage for off-site backup
# aws s3 cp "${BACKUP_FILE}" "s3://vidgen-backups/${TIMESTAMP}.sql.gz"
# rclone copy "${BACKUP_FILE}" r2:vidgen-backups/
```

---

## The Full Deployment Topology

Putting it all together, here is the complete architecture of our deployed AI video platform:

<svg viewBox="0 0 900 620" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:system-ui,sans-serif">
  <defs>
    <marker id="arr-f" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
    <marker id="arr-g" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#66bb6a"/>
    </marker>
  </defs>

  <text x="450" y="28" text-anchor="middle" font-size="17" font-weight="bold" fill="#222">Complete Deployment Topology</text>

  <!-- Internet / Users -->
  <rect x="350" y="50" width="200" height="50" rx="10" fill="#f5f5f5" stroke="#999" stroke-width="2"/>
  <text x="450" y="80" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">Users (Internet)</text>

  <!-- Arrow to Firewall -->
  <line x1="450" y1="100" x2="450" y2="130" stroke="#333" stroke-width="2" marker-end="url(#arr-f)"/>

  <!-- Firewall -->
  <rect x="330" y="130" width="240" height="40" rx="6" fill="#ffcdd2" stroke="#e53935" stroke-width="2"/>
  <text x="450" y="155" text-anchor="middle" font-size="12" font-weight="bold" fill="#c62828">Firewall (UFW): ports 80, 443 only</text>

  <!-- Arrow to Nginx -->
  <line x1="450" y1="170" x2="450" y2="200" stroke="#333" stroke-width="2" marker-end="url(#arr-f)"/>

  <!-- VPS Box -->
  <rect x="30" y="190" width="840" height="400" rx="12" fill="#fafafa" stroke="#ccc" stroke-width="1.5" stroke-dasharray="6,3"/>
  <text x="50" y="212" font-size="11" fill="#999" font-weight="bold">VPS (Ubuntu 22.04)</text>

  <!-- Docker network box -->
  <rect x="50" y="285" width="800" height="285" rx="10" fill="#f0f4f8" stroke="#90caf9" stroke-width="1.5" stroke-dasharray="4,3"/>
  <text x="75" y="305" font-size="10" fill="#42a5f5" font-weight="bold">Docker Bridge Network: vidgen-network</text>

  <!-- Nginx -->
  <rect x="340" y="220" width="220" height="55" rx="8" fill="#e8eaf6" stroke="#7986cb" stroke-width="2"/>
  <text x="450" y="243" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Nginx</text>
  <text x="450" y="260" text-anchor="middle" font-size="10" fill="#666">TLS termination, rate limiting, static files</text>

  <!-- Arrows from Nginx -->
  <line x1="390" y1="275" x2="200" y2="325" stroke="#333" stroke-width="1.5" marker-end="url(#arr-f)"/>
  <line x1="450" y1="275" x2="450" y2="325" stroke="#333" stroke-width="1.5" marker-end="url(#arr-f)"/>
  <text x="310" y="300" font-size="9" fill="#666">HTTP proxy</text>
  <text x="480" y="300" font-size="9" fill="#666">WebSocket</text>

  <!-- API Server -->
  <rect x="80" y="325" width="200" height="70" rx="8" fill="#e3f2fd" stroke="#42a5f5" stroke-width="2"/>
  <text x="180" y="352" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">API Server</text>
  <text x="180" y="370" text-anchor="middle" font-size="10" fill="#666">Node.js + Express</text>
  <text x="180" y="384" text-anchor="middle" font-size="10" fill="#666">:3000 (internal only)</text>

  <!-- WebSocket Server (part of API) -->
  <rect x="350" y="325" width="200" height="70" rx="8" fill="#e3f2fd" stroke="#42a5f5" stroke-width="2"/>
  <text x="450" y="352" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">WebSocket Server</text>
  <text x="450" y="370" text-anchor="middle" font-size="10" fill="#666">Real-time status updates</text>
  <text x="450" y="384" text-anchor="middle" font-size="10" fill="#666">via Redis Pub/Sub</text>

  <!-- Worker -->
  <rect x="620" y="325" width="200" height="70" rx="8" fill="#e8f5e9" stroke="#66bb6a" stroke-width="2"/>
  <text x="720" y="352" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Worker Process</text>
  <text x="720" y="370" text-anchor="middle" font-size="10" fill="#666">BullMQ consumer</text>
  <text x="720" y="384" text-anchor="middle" font-size="10" fill="#666">Video generation jobs</text>

  <!-- Redis -->
  <rect x="280" y="440" width="160" height="60" rx="8" fill="#ffebee" stroke="#ef5350" stroke-width="2"/>
  <text x="360" y="465" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Redis</text>
  <text x="360" y="482" text-anchor="middle" font-size="10" fill="#666">Queue + Pub/Sub + Cache</text>

  <!-- PostgreSQL -->
  <rect x="100" y="440" width="160" height="60" rx="8" fill="#fff3e0" stroke="#ffa726" stroke-width="2"/>
  <text x="180" y="465" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">PostgreSQL</text>
  <text x="180" y="482" text-anchor="middle" font-size="10" fill="#666">Users, projects, billing</text>

  <!-- External APIs -->
  <rect x="620" y="440" width="200" height="60" rx="8" fill="#f3e5f5" stroke="#ab47bc" stroke-width="2"/>
  <text x="720" y="465" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Video Model APIs</text>
  <text x="720" y="482" text-anchor="middle" font-size="10" fill="#666">Veo, Kling, Runway</text>

  <!-- Connections: API to databases -->
  <line x1="180" y1="395" x2="180" y2="440" stroke="#333" stroke-width="1.5" marker-end="url(#arr-f)"/>
  <line x1="230" y1="395" x2="320" y2="440" stroke="#333" stroke-width="1.5" marker-end="url(#arr-f)"/>

  <!-- Connections: Worker to databases -->
  <line x1="670" y1="395" x2="440" y2="440" stroke="#333" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="720" y1="395" x2="720" y2="440" stroke="#ab47bc" stroke-width="1.5" marker-end="url(#arr-f)"/>

  <!-- WebSocket to Redis -->
  <line x1="450" y1="395" x2="400" y2="440" stroke="#ef5350" stroke-width="1.5" marker-end="url(#arr-f)"/>

  <!-- Volumes -->
  <rect x="100" y="520" width="120" height="35" rx="4" fill="#e0e0e0" stroke="#999" stroke-width="1"/>
  <text x="160" y="542" text-anchor="middle" font-size="10" fill="#555">postgres_data vol</text>

  <rect x="300" y="520" width="120" height="35" rx="4" fill="#e0e0e0" stroke="#999" stroke-width="1"/>
  <text x="360" y="542" text-anchor="middle" font-size="10" fill="#555">redis_data vol</text>

  <line x1="180" y1="500" x2="160" y2="520" stroke="#999" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="360" y1="500" x2="360" y2="520" stroke="#999" stroke-width="1" stroke-dasharray="3,3"/>

  <!-- GitHub Actions -->
  <rect x="680" y="195" width="170" height="55" rx="8" fill="#e8f5e9" stroke="#66bb6a" stroke-width="2"/>
  <text x="765" y="218" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">GitHub Actions</text>
  <text x="765" y="235" text-anchor="middle" font-size="10" fill="#666">CI/CD: build, test, deploy</text>

  <line x1="765" y1="250" x2="765" y2="290" stroke="#66bb6a" stroke-width="1.5" stroke-dasharray="4,3"/>
  <line x1="765" y1="290" x2="720" y2="325" stroke="#66bb6a" stroke-width="1.5" marker-end="url(#arr-g)"/>
  <text x="790" y="275" font-size="9" fill="#66bb6a">SSH deploy</text>
</svg>

The flow from end to end:
1. Users connect to your domain over HTTPS.
2. The firewall allows traffic only on ports 80 and 443, blocking everything else.
3. Nginx terminates TLS, serves static files and cached video assets directly, and proxies API and WebSocket requests to the backend.
4. The API server handles authentication, business logic, and creates generation jobs in Redis via BullMQ.
5. Worker processes pick up jobs from the Redis queue and call external video generation APIs (Veo, Kling, Runway).
6. Real-time status updates flow through Redis Pub/Sub to WebSocket connections, so users see "Generating frames (24/48)..." in real time.
7. All persistent data---users, projects, generations, billing---lives in PostgreSQL backed by a Docker volume.
8. GitHub Actions automatically builds, tests, and deploys new code on every push to main, with health-check verification before declaring success.

---

## Where to Go from Here

This post covered the deployment fundamentals: containers, Docker Compose, Nginx, TLS, firewall, and CI/CD. These are sufficient for a single-server deployment handling moderate traffic---say, a few hundred concurrent users and a few thousand video generations per day.

When you outgrow a single server, the next steps are:

- **Container orchestration** (Kubernetes or Docker Swarm) for running across multiple servers with automatic scaling, self-healing, and rolling deployments.
- **Managed databases** (RDS, Cloud SQL, Supabase) instead of running PostgreSQL in Docker on the same machine as your application. Managed databases handle backups, replication, and failover automatically.
- **CDN** (Cloudflare, CloudFront) for caching and serving generated videos from edge locations globally, reducing latency for users far from your server.
- **Log aggregation** (Grafana Loki, ELK stack, Datadog) for centralized logging across multiple containers and servers. When something breaks, you need to search logs across your entire fleet, not SSH into individual machines.
- **Metrics and alerting** (Prometheus + Grafana, Datadog) for monitoring CPU, memory, response times, error rates, and queue depths, with automatic alerts when things go wrong.

But do not start there. Start with a single server, the architecture in this post, and real users generating real videos. You can always add complexity later when the bottlenecks become clear. You cannot easily remove complexity once you have committed to it.

*Next in the series: Part 3 covers monitoring, observability, and error tracking---turning your black-box deployment into a system you can actually debug when things go wrong at 2 AM.*
