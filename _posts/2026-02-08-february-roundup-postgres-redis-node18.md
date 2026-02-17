---
layout: post
title: "February 2026 Roundup: PostgreSQL CVEs, Redis Goes AGPL, Node 18 Sunset, and Python Gets Threads"
date: 2026-02-08
category: roundup
urgency: important
score: 82
type: weekly
---

A lot happened in the database and runtime layers this month. PostgreSQL shipped critical security patches, Redis reversed its licensing course, Node.js 18 is being forcibly retired across the ecosystem, and Python 3.14 made free-threading official. Here's what matters and what to do about it.

## PostgreSQL: 5 CVEs Including a Network-Accessible Buffer Overflow

On February 12, the PostgreSQL Global Development Group released security updates across all active lines: 18.2, 17.8, 16.12, 15.16, and 14.21. Five CVEs, 65+ bug fixes.

The critical one: **CVE-2026-2007** (CVSS 8.2). A heap buffer overflow in the `pg_trgm` extension — the one you're probably using for fuzzy text search with `LIKE`, `ILIKE`, or trigram similarity queries. No authentication required. Network-accessible. Affects PostgreSQL 18.0 and 18.1.

The others worth knowing:

- **CVE-2026-2004:** Code execution through the `intarray` selectivity estimator via `pgcrypto`. Affects PostgreSQL 14-18.
- **CVE-2026-2003:** `oidvector` type validation issue that leaks bytes of server memory.
- **CVE-2026-2006:** Missing multibyte character length validation enabling buffer overruns.

**Important timing note:** An out-of-cycle patch release is scheduled for February 26, 2026 to fix regressions introduced by the February 12 update. If you can wait, hold off until 18.2.1 / 17.8.1 drops. If you're running pg_trgm on a public-facing PostgreSQL 18 instance, patch now anyway — the CVE is worse than the regressions.

**Action items:**
- Check if you use `pg_trgm`: `SELECT * FROM pg_extension WHERE extname = 'pg_trgm';`
- Check your PostgreSQL version: `SELECT version();`
- Plan the upgrade, preferably after the Feb 26 follow-up release
- If you're on a managed service (AWS RDS, Supabase), check whether they've applied the patches

## Redis Goes AGPL: The Open Source Return

In what might be the most significant licensing reversal in recent open source history, Redis Ltd. added AGPLv3 as a third licensing option starting with Redis 8. Redis is now available under an OSI-approved open source license again for the first time since Redis 7.4.

The backstory: In March 2024, Redis switched from BSD to a dual RSALv2/SSPLv1 license — neither of which is considered open source by the OSI. The move was widely criticized. It spawned Valkey, a Linux Foundation fork backed by AWS, Google Cloud, Oracle, and ~50 other companies. Valkey has been gaining serious enterprise traction, and Redis clearly felt the pressure.

**What AGPL means for you:**

If you use Redis as a cache, session store, or pub/sub layer in your application — which is how most developers use it — AGPL changes nothing for you. AGPL's copyleft requirement triggers when you modify Redis itself and offer it as a network service. Running unmodified Redis behind your application doesn't trigger the copyleft clause.

If you're building a managed Redis service or redistributing a modified Redis, AGPL requires you to release your modifications. That's the whole point — it prevents cloud providers from taking the code, adding proprietary features, and offering it as a competing service without contributing back.

**Valkey isn't going away:**

Despite Redis's relicensing, Valkey has its own momentum now. AWS ElastiCache, Google Memorystore, and Oracle Cloud all offer Valkey-based services. Valkey maintains full Redis protocol compatibility and has been adding features independently. For new projects, Valkey is the lower-risk choice — it's BSD-licensed with no copyleft concerns and has multi-company governance.

For existing Redis deployments: Redis 8 with AGPL simplifies compliance, but check whether your cloud provider's managed offering has updated. AWS ElastiCache has been migrating to Valkey; you might already be running Valkey without knowing it.

## Node.js 18 End of Support Cascade

Node.js 18 officially reached end-of-life in April 2025, but the real enforcement is hitting now:

- **AWS SDK for JavaScript v3** dropped Node 18 support in January 2026. Installing newer SDK versions on Node 18 produces engine warnings (errors with `engine-strict=true`).
- **AWS Lambda** extended its enforcement deadline to March 2026, after which new deployments on Node 18 will be blocked.
- **Vercel** deprecated Node 18 for Builds and Functions in September 2025.
- **AWS CDK** announced end of support for Node 18.x.

This is a forced migration. If any part of your stack runs Node 18 — Lambda functions, Docker images, CI runners, local dev — you need to move.

**The right target is Node.js 22 LTS**, not Node.js 20. Here's why: Node 20 LTS reaches end-of-life in April 2026 — just two months from now. Migrating to Node 20 buys you almost nothing before you'd need to migrate again. Skip it entirely.

**Migration checklist:**
- Update `FROM node:22-alpine` in your Dockerfiles
- Update `node-version: '22'` in GitHub Actions workflows
- Update `engines.node` in `package.json`
- Update Lambda runtime to `nodejs22.x`
- Run your test suite on Node 22 — breaking changes are minimal but test anyway
- Update `@types/node` to match: `npm install -D @types/node@22`

## January Security Releases: Node.js 8 CVEs

Alongside the EOL cascade, Node.js shipped security updates for all active lines (20.x, 22.x, 24.x, 25.x) fixing 8 CVEs. Three are High severity:

- **CVE-2025-55131:** Buffer initialization race condition in the `vm` module exposes uninitialized memory. Relevant if you use `vm.runInContext` with timeout options.
- **CVE-2025-55130:** Symlink path bypass defeats `--allow-fs-read` / `--allow-fs-write` in the Permissions model. Relevant if you're using Node's experimental permissions in containers or sandboxed environments.
- **CVE-2025-59465:** Malformed HTTP/2 HEADERS frame crashes the server via unhandled TLSSocket error. This is a remote, unauthenticated DoS — relevant to any HTTP/2 endpoint.

**Action:** Update to latest patch: `nvm install 22` or update your Docker base images. The HTTP/2 DoS (CVE-2025-59465) is particularly concerning for production servers.

## Python 3.14: Free-Threading Goes Official

Python 3.14 made free-threaded Python (no GIL) officially supported via PEP 779. This was experimental in 3.13 — now it's the real deal.

**Why this matters for FastAPI:**

FastAPI runs on Uvicorn with async I/O, which handles concurrent requests well for I/O-bound work. But CPU-bound tasks — data processing, ML inference, complex validation — still hit the GIL wall. The workaround has always been `ProcessPoolExecutor`, which has its own overhead and complexity.

With free-threaded Python, CPU-bound work can actually parallelize across threads. For FastAPI endpoints that do heavy computation, this could mean real throughput improvements without the multiprocessing dance.

**Other Python 3.14 highlights:**

- **T-strings (PEP 750):** Template string literals with `t""` syntax, similar to f-strings but with custom processing hooks. Useful for building SQL queries, HTML, and prompt strings safely.
- **Deferred annotation evaluation (PEP 649):** Resolves the longstanding `from __future__ import annotations` saga. Forward references in type hints now work predictably. This is particularly good for Pydantic models with circular references.
- **Multiple interpreters in stdlib (PEP 734):** Run isolated Python interpreters in the same process. Think of it as lightweight multiprocessing without the process spawn overhead.

**Caveat:** Free-threaded Python requires extension modules to be built without the GIL. Core libraries like numpy and pydantic-core need to ship GIL-free wheels before the ecosystem fully benefits. Check compatibility before upgrading production FastAPI apps.

## GitHub Actions: New Charges and Supply Chain Compromise

Two things happened with GitHub Actions in January:

**New pricing:** A $0.002/minute platform charge started January 1, 2026 for all private repo usage. Not devastating, but it adds up for CI-heavy projects with long build times. Audit your workflow minutes if you haven't recently.

**Supply chain attack:** `tj-actions/changed-files` — used in 23,000+ repositories — was compromised. A malicious commit injected code that scans GitHub Runner memory for secrets (tokens, AWS credentials, etc.) and exfiltrates them. If you use this action, check your repository's Actions logs for suspicious network activity and rotate any secrets that may have been exposed.

**Prevention:** Pin all third-party GitHub Actions to commit SHAs, not tags. Tags can be force-pushed; commit SHAs can't. Use Dependabot or Renovate to track action version updates.

## Docker Desktop: Patch Your AI Assistant

Docker patched critical vulnerabilities in Ask Gordon, its AI assistant in Docker Desktop. CVE-2025-9074 (CVSS 9.3) allows container escape and host compromise through prompt injection in the chat interface. Update to Docker Desktop 4.44.3+.

The broader lesson: AI assistants embedded in developer tools create a new attack surface. They have access to your local files, environment variables, and Docker daemon. Treat them with the same caution as browser extensions.

## What to Do This Week

1. **Patch Next.js** for CVE-2026-23864 (if you haven't already)
2. **Plan PostgreSQL upgrade** (after Feb 26 out-of-cycle release)
3. **Migrate off Node.js 18** to Node.js 22 LTS
4. **Pin GitHub Actions** to commit SHAs
5. **Update Docker Desktop** to 4.44.3+
6. **Test TypeScript 6.0 beta** against your codebase: `npm install -D typescript@beta`
7. **Evaluate Prisma 7** if you're on Prisma 6 — the upgrade is worth it
