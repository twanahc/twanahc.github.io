---
layout: post
title: "TypeScript 6.0: The End of an Era"
date: 2026-02-15
category: tooling
urgency: important
score: 88
type: cluster
---

On February 11, Microsoft shipped the TypeScript 6.0 Beta. The headline feature isn't a new type — it's a farewell. This is explicitly the last TypeScript release built on JavaScript. TypeScript 7.0 will be rewritten in Go, and 6.0 is the bridge that gets you there.

If you've been writing TypeScript for any length of time, this is worth paying attention to. Not because 6.0 is revolutionary, but because everything it deprecates tells you exactly what TypeScript 7.0 will demand.

## The Defaults Change

TypeScript 6.0 flips several defaults that the community has been manually setting for years:

- **`strict` now defaults to `true`** — Every serious project already sets this. Now it's the default. If you've been running with strict off, your build will break. Fix: add `"strict": false` to `tsconfig.json` (but really, fix your types).

- **`module` defaults to `esnext`** — CommonJS is no longer the assumed output. This matters for Node.js projects that haven't migrated to ESM yet.

- **`target` defaults to `es2025`** — No more ES5 output by default. If you still need to target older environments, set it explicitly.

- **`types` defaults to `[]`** — This is the sneaky one. Previously, TypeScript auto-included type definitions from `@types/*` packages in `node_modules`. Now it includes nothing by default. If your project relies on `@types/node` being automatically available, your build will break with mysterious "Cannot find name 'Buffer'" errors. Fix: add `"types": ["node"]` to `tsconfig.json`.

The `types: []` change alone will break thousands of projects on upgrade. But it also improves build performance by 20-50% because TypeScript no longer crawls `node_modules/@types/` on startup. It's the right default — it just hurts.

## What's Deprecated (And Removed in 7.0)

These options still work in 6.0 with warnings. In TypeScript 7.0 (Go-based), they're gone:

- **`--outFile`** — The single-bundle output mode. If you're using this, you've been swimming against the current for years. Use a bundler.
- **`--baseUrl`** — For import resolution without `paths`. Rarely needed if you're using `paths` correctly.
- **`--moduleResolution node`** — The old Node.js resolution algorithm. Use `node16`, `nodenext`, or `bundler` instead.
- **`target: es5`** — ES5 output is going away entirely. If you need ES5, use a bundler's downleveling.
- **AMD, UMD, SystemJS module formats** — The module format wars are over. ESM won.

You can temporarily silence deprecation warnings with `"ignoreDeprecations": "6.0"`, but treat this as a migration tool, not a permanent solution. Every one of these deprecations will become an error in 7.0.

## The Go Rewrite: What We Know

TypeScript 7.0 — internally called "Corsa" or the "native port" — is being rewritten in Go for one reason: performance. The JavaScript-based compiler has reached its ceiling. Despite years of optimization, large projects still wait 30-60 seconds for a full type check.

The Go rewrite promises:

- **10x faster type checking** through native parallelism (Go's goroutines vs. JavaScript's single-threaded event loop)
- **Lower memory usage** — Go's garbage collector is more predictable than V8's for long-running compiler processes
- **Faster IDE integration** — `tsserver` responsiveness directly improves the editing experience

The `--stableTypeOrdering` flag in 6.0 previews TypeScript 7.0 behavior. When enabled, type declarations are emitted in a deterministic order (alphabetical) rather than source order. This is necessary for the Go compiler's parallel processing. Currently, this flag *reduces* performance by up to 25% in the JS compiler — but it will be the default in the Go compiler where parallel processing makes it net positive.

## Migration Strategy

**Right now (before 6.0 stable on March 17):**

1. Run `npx tsc@beta --noEmit` against your project to see what breaks
2. Add explicit `"types": ["node"]` (or whatever `@types/*` packages you use) to `tsconfig.json`
3. Replace `--moduleResolution node` with `node16` or `bundler`
4. If you use `--outFile`, start planning the migration to a bundler

**Before TypeScript 7.0 (estimated mid-2026):**

1. Eliminate all deprecated options from your tsconfig
2. Test with `--stableTypeOrdering` to catch any code that depends on declaration order
3. Audit your build pipeline — if any tooling depends on TypeScript's JavaScript API, it will need updating for the Go-based compiler

**What not to worry about:**

TypeScript's type system itself isn't changing. Your types, interfaces, generics, conditional types — all of it works the same. The rewrite is about the compiler's implementation language, not the language it compiles. Your code is safe. Your tooling config needs attention.

## The Bigger Picture

TypeScript choosing Go over Rust is interesting. The team explicitly cited Go's simpler concurrency model and faster compile times (for the compiler itself) as deciding factors. Rust would have given them more control over memory, but Go gives them faster iteration speed during the rewrite.

This mirrors a broader trend: developer tools are moving to compiled languages. Bun (Zig), Turbopack (Rust), esbuild (Go), now TypeScript (Go). The JavaScript toolchain is increasingly written in everything except JavaScript. The reason is simple — JavaScript is great for application code but fundamentally limited for CPU-bound compiler workloads where parallelism matters.

For TypeScript users, the practical impact is: your editor gets faster, your CI gets faster, and your tsconfig gets a spring cleaning. That's a good trade.

Install the beta and start testing: `npm install -D typescript@beta`
