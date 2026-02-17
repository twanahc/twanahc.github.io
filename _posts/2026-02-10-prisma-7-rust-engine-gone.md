---
layout: post
title: "Prisma 7: The Rust Engine Is Gone (And Everything Is Faster)"
date: 2026-02-10
category: tooling
urgency: important
score: 85
type: cluster
---

Prisma has been the TypeScript ORM that people love to complain about. Too slow. Too many generated types. That 14 MB Rust binary that breaks on every new platform. Cold starts on Lambda that make you question your life choices. Prisma 7.0 addresses all of it by doing something drastic: removing the Rust engine entirely.

The results speak for themselves: 3.4x faster query execution, 90% smaller bundles, 70% faster type checks. And no more platform-specific binary compatibility headaches.

## What Changed

Since its inception, Prisma has used a Rust-based query engine as a sidecar process. Your TypeScript code talks to the Prisma Client, which serializes your query, sends it over a local HTTP connection to the Rust binary, which parses it, generates SQL, talks to your database, serializes the result, and sends it back. Every query pays the cost of cross-language serialization twice.

Prisma 7 replaces all of this with a pure TypeScript query engine. Your code talks to the Prisma Client, which generates SQL directly and talks to your database using Node.js native drivers. No sidecar process. No serialization boundary. No Rust binary.

The numbers:

- **Query execution: up to 3.4x faster** — eliminating the serialization round-trip is the biggest single performance improvement in Prisma's history
- **Bundle size: 14 MB → 1.6 MB** — the Rust binary was the vast majority of Prisma's installed weight
- **Generated types: 98% fewer** — Prisma's type generation was infamously verbose. The new client generates dramatically less code.
- **Full type checks: 70% faster** — directly caused by the type reduction. Your IDE is snappier, your CI is faster.

## Why This Matters for Deployments

The Rust engine was Prisma's Achilles' heel on edge and serverless platforms. Every Lambda cold start had to load a 14 MB binary. Every Vercel Edge Function deployment had to include a platform-specific Rust binary. Every new platform (ARM64 Linux, Windows ARM, etc.) needed a separate binary build.

With Prisma 7:

- **Lambda cold starts drop significantly** — no binary to load, just JavaScript modules
- **Vercel Edge Functions** work without special configuration — no more binary compatibility dance
- **Cloudflare Workers** are now a realistic deployment target — bundle size under 2 MB fits comfortably
- **Docker images shrink** — no more multi-stage builds to include the right binary
- **No platform matrix** — the TypeScript engine runs anywhere Node.js runs

## Breaking Changes You Need to Handle

This isn't a drop-in upgrade. Prisma 7 changes where generated code lives:

**Before (Prisma 6):** Generated client code goes to `node_modules/.prisma/client/`

**After (Prisma 7):** Generated client code goes to your project source tree (configurable location)

This means:
- Your `.gitignore` needs updating if you don't want generated files committed
- Your import paths may need updating depending on your setup
- CI pipelines that run `prisma generate` need to account for the new output location

The `prisma generate` command still works the same way, but the output destination is different. Check the migration guide for your specific setup.

**Other breaking changes:**
- The query engine no longer accepts raw SQL through the engine protocol — use `$queryRaw` and `$executeRaw` as before, but some edge cases in raw query handling have changed
- `PrismaClient` constructor options related to the engine binary (`binaryTargets`, `engineType`) are removed
- The `@prisma/engines` package no longer exists

## Prisma 7.3.0: Further Tuning

The follow-up release (January 21, 2026) adds a `compilerBuild` option:

- `fast` — Optimizes the query compiler for speed. Better for development and Lambda where startup time matters.
- `small` — Optimizes for bundle size. Better for edge deployments where every KB counts.

This is a nice touch — rather than forcing a one-size-fits-all tradeoff, they let you optimize for your deployment target.

## The Drizzle Question

The elephant in the room: does this change the Prisma vs. Drizzle calculus?

Drizzle's core selling point has been "closer to SQL, no engine overhead, smaller bundles." Prisma 7 neutralizes two of those three arguments. The bundle size gap is now minimal (1.6 MB vs. Drizzle's sub-1 MB). The performance gap has narrowed dramatically — Prisma 7 is genuinely competitive on raw query speed.

What Drizzle still has: a more SQL-like API that some developers prefer, and a relational query builder that maps more directly to how you think about SQL. What Prisma still has: the schema-first workflow with migrations, a more mature ecosystem of tools (Prisma Studio, Prisma Accelerate), and arguably better type safety for complex queries.

The honest answer: both are excellent choices now. Prisma 7 removed the technical arguments against it. The remaining choice is about API preference and workflow.

## Migration Checklist

1. **Check compatibility:** `npx prisma@7 validate` against your existing schema
2. **Update dependencies:** `npm install prisma@7 @prisma/client@7`
3. **Run generate:** `npx prisma generate` — check where output goes
4. **Update .gitignore:** Add the new generated client location if needed
5. **Remove engine configs:** Delete `binaryTargets` from your schema if present
6. **Test queries:** Run your test suite. Pay attention to raw SQL queries and edge cases around JSON fields
7. **Benchmark:** Compare query latencies before and after. You should see immediate improvement.
8. **Update Dockerfiles:** Remove any multi-stage build steps that were copying Rust binaries

## Looking Ahead

Prisma's roadmap after 7.0 is focused on the new architecture's capabilities. Without the Rust engine as a bottleneck, they can iterate faster on the query compiler and add features that would have been impractical before — like streaming query results, better connection pooling, and native support for database-specific features.

The deeper story is about the JavaScript ecosystem maturing. Prisma tried the "rewrite the hot path in Rust" approach that's popular in JS tooling (Turbopack, SWC, etc.), and concluded it wasn't worth the complexity for their use case. The serialization boundary between JavaScript and Rust cost more than Rust's speed advantage saved. Sometimes the right answer is just writing better JavaScript.

For your projects: if you've been hesitant about Prisma because of bundle size or cold starts, those objections are gone. If you're already using Prisma, upgrade — you'll get free performance improvements across the board.
