---
layout: post
title: "Claude Code for Production: A Complete Guide to Building Software That Doesn't Become Technical Debt"
date: 2026-03-10
category: tools
mathjax: false
---

A developer ships a feature with Claude Code in an hour. The endpoint handles authentication, validates input with Zod schemas, queries the database, and returns a well-structured response. Tests pass. The TypeScript compiles without errors. The PR merges at 4 PM. Two months later, that endpoint is responsible for 30 percent of the team's production incidents. The retry logic retries on non-idempotent operations. The validation trusts client-provided user IDs without checking authorization. The database query does a sequential scan on a table that has grown to 2 million rows since launch. None of this is because Claude wrote bad code. The code is syntactically clean, well-typed, and follows every naming convention. The problem is that nobody told Claude about the retry semantics, the authorization model, or the table growth projections. The same tool that generated the code could have prevented every one of these issues --- if the project had been set up correctly.

Claude Code is not autocomplete. It is not a fancy snippet generator. It is a development environment with access to your entire codebase, your terminal, your test suite, and any external tool you connect to it. That access makes it extraordinarily productive when configured correctly and extraordinarily dangerous when configured loosely. The difference is not in Claude's capabilities --- it is in the engineering discipline that the project configuration enforces.

The configuration surface has five components: CLAUDE.md (persistent instructions that define your standards), hooks (automated quality gates that run on every tool invocation), MCP servers (external tool integrations that extend Claude's capabilities), test-driven workflows (where humans define correctness and Claude implements it), and code review patterns (where humans verify what Claude produced). Get all five right and Claude Code becomes the most productive engineering environment that exists. Get them wrong --- or skip them --- and Claude Code is a technical debt accelerator that produces plausible-looking code that silently violates every assumption your system depends on.

This article is the setup guide.

---

## Table of Contents

1. [The Problem Claude Code Solves (and the One It Creates)](#the-problem-claude-code-solves-and-the-one-it-creates)
2. [Project Initialization: CLAUDE.md as Your Constitution](#project-initialization-claudemd-as-your-constitution)
3. [Hook Systems: Automated Quality Gates](#hook-systems-automated-quality-gates)
4. [MCP Servers: Extending Claude's Capabilities](#mcp-servers-extending-claudes-capabilities)
5. [Test-Driven Development with Claude Code](#test-driven-development-with-claude-code)
6. [Code Review Discipline](#code-review-discipline)
7. [Project Structure and Architecture Patterns](#project-structure-and-architecture-patterns)
8. [Skills and Workflows](#skills-and-workflows)
9. [Performance-Aware Development](#performance-aware-development)
10. [The Production Readiness Checklist](#the-production-readiness-checklist)

---

## The Problem Claude Code Solves (and the One It Creates)

Before AI-assisted development, the bottleneck was typing. Not literally typing speed --- the bottleneck was the mechanical work of translating an idea into code. Writing boilerplate, implementing CRUD endpoints, constructing test harnesses, wiring up configurations. A senior engineer might spend 30 percent of their day on implementation and 70 percent on design, debugging, and review. The mechanical work was the tax on productivity.

Claude Code removed that tax. Implementation that took an afternoon now takes minutes. A well-prompted session produces working endpoints, test suites, database migrations, and configuration files at a speed that was not possible before. The typing bottleneck is gone.

The new bottleneck is quality at speed.

When a developer writes 50 lines of code per day, they review each line multiple times. When Claude Code produces 500 lines in a session, the review dynamics change fundamentally. The code looks correct --- Claude is excellent at producing syntactically valid, well-structured, properly named code that follows visible patterns. The errors are in the invisible assumptions: retry semantics, authorization boundaries, performance characteristics under load, failure modes at 3 AM when the database connection pool is exhausted and the rate limiter is rejecting half the requests.

Without guardrails, Claude Code is a debt accelerator. It produces code faster than anyone can review it, and the code's surface quality makes errors hard to spot. Every line looks intentional. Every function has a reasonable name. The patterns are plausible. But plausible is not correct, and the gap between plausible and correct is where production incidents live.

With the right setup --- CLAUDE.md conventions that define your standards, hooks that enforce them automatically, tests that define correctness before implementation begins, and review practices calibrated for AI-generated code --- Claude Code becomes something different. It becomes an environment where the developer focuses entirely on the hard problems (what to build, why, and what assumptions must hold) while the implementation is handled by a system that already knows the project's constraints, conventions, and forbidden patterns.

The rest of this article describes that setup.

---

## Project Initialization: CLAUDE.md as Your Constitution

CLAUDE.md is a Markdown file at the root of your repository. Claude Code reads it at the start of every session and treats its contents as persistent instructions --- a system prompt that applies to every interaction within that project. If Claude Code is a new engineer on your team, CLAUDE.md is the onboarding document that tells them how things work here.

This is the single most important file in a Claude Code project. Everything else --- hooks, tests, reviews --- is downstream of what CLAUDE.md defines. Get this wrong and every other guardrail is working against drift. Get it right and Claude Code already knows your coding standards, your architectural decisions, your forbidden patterns, and your domain-specific terminology before a single line of code is written.

### What Goes in CLAUDE.md

**Coding standards.** Not "use TypeScript" --- that is too vague to constrain behavior. Specific, enforceable rules:

```markdown
# Coding Standards
- Language: TypeScript 5.x, strict mode
- Framework: Next.js 15 with App Router
- Database: PostgreSQL via Prisma ORM
- NEVER use `any` type — use `unknown` with type guards
- NEVER use `console.log` — use the structured logger from `lib/logger.ts`
- All API routes must validate input with Zod schemas
- All database queries must use parameterized queries (no string interpolation)
- Error responses follow RFC 7807 Problem Details format
```

Every rule in this block is a constraint that Claude Code will follow in every session. "NEVER use `any` type" is not a suggestion --- it is a rule that Claude will not violate because it reads this file before generating any code.

**Architectural decisions.** Document the decisions that are already made so Claude does not relitigate them:

```markdown
# Architecture
- Authentication: JWT tokens with refresh rotation, 15-minute access token TTL
- Authorization: RBAC with role hierarchy defined in `lib/auth/roles.ts`
- Caching: Redis for session data, no application-level caching of database queries
- Queue: BullMQ for background jobs, all jobs must be idempotent
- NEVER add new npm dependencies without explicit approval
- NEVER modify the database schema without a migration file
```

**Forbidden patterns.** The patterns that cause production incidents in your specific system. These are learned from experience and are the highest-value content in CLAUDE.md:

```markdown
# Forbidden Patterns
- NEVER retry non-idempotent operations (POST, PUT, DELETE)
- NEVER trust client-provided user IDs — always derive from JWT
- NEVER use `SELECT *` — always specify columns explicitly
- NEVER catch errors silently — log with correlation ID and re-throw
- NEVER use `setTimeout` or `setInterval` for scheduling — use BullMQ
- NEVER store PII in logs — use the redaction middleware from `lib/logger.ts`
```

**Test requirements.** Define what "tested" means in your project:

```markdown
# Testing
- Every new function must have at least one unit test
- Business logic must have 100% branch coverage
- API routes must have integration tests covering happy path and error cases
- Database queries must be tested against a real database (not mocked)
- Use `vitest` for unit tests, `supertest` for API tests
```

**Domain terminology.** If your project uses terms that have specific meanings different from their common usage, define them:

```markdown
# Domain Terms
- "workspace" = a tenant's isolated environment (not a VS Code workspace)
- "member" = a user with access to a workspace (not a database record)
- "plan" = a billing tier (not a project plan or a database query plan)
```

### What Does NOT Go in CLAUDE.md

**Session-specific context.** "I am currently working on the billing module" is conversation context, not a project standard. It belongs in the conversation, not in CLAUDE.md.

**Temporary state or work-in-progress notes.** "TODO: refactor the auth module" is a task tracker item. CLAUDE.md is for stable rules, not transient notes.

**Personal preferences that change frequently.** "I prefer tabs over spaces" is a formatting rule and belongs in a formatter configuration (.prettierrc, .editorconfig), not in CLAUDE.md. Put it in CLAUDE.md only if it is a team standard.

**Secrets or credentials.** CLAUDE.md is checked into version control. Never put API keys, database passwords, or tokens in it.

### Layered CLAUDE.md

CLAUDE.md files are hierarchical. A repository root CLAUDE.md sets project-wide standards. A directory-level CLAUDE.md in `src/api/` can add API-specific rules without repeating the project-wide ones. Claude Code walks up the directory tree and loads all CLAUDE.md files it encounters, with more specific files supplementing (not replacing) more general ones.

This is useful for monorepos: the root CLAUDE.md defines shared standards, and each package's CLAUDE.md adds package-specific rules. It is also useful for areas of the codebase with different constraints --- a `migrations/` directory might have a CLAUDE.md that says "NEVER delete a column, only add columns and mark old ones as deprecated."

### The New Engineer Test

Here is a heuristic for completeness: if a new engineer joined your team and only had CLAUDE.md to understand your standards, could they write code that passes code review? If not, the CLAUDE.md is incomplete. The gaps between your CLAUDE.md and what a new engineer would need to know are exactly the gaps where Claude Code will produce plausible but incorrect code.

The test works because Claude Code is, in a meaningful sense, a new engineer on every session. It has no memory of previous sessions (unless you configure persistence). It knows nothing about your project except what CLAUDE.md and the codebase tell it. The more explicit and complete your CLAUDE.md, the less time you spend correcting assumptions in conversation, and the fewer reviews fail because Claude made a decision that violated an undocumented convention.

### Effective vs. Ineffective

The difference between an effective CLAUDE.md and an ineffective one is specificity. An ineffective CLAUDE.md describes the project:

```markdown
# Project
This is a web app. Use TypeScript. Follow best practices.
```

"Follow best practices" is meaningless --- it does not tell Claude which practices, whose definition of best, or what trade-offs to make when best practices conflict. An effective CLAUDE.md prescribes behavior:

```markdown
# Coding Standards
- Language: TypeScript 5.x, strict mode
- Framework: Next.js 15 with App Router
- Database: PostgreSQL via Prisma ORM
- NEVER use `any` type — use `unknown` with type guards
- NEVER use `console.log` — use the structured logger from `lib/logger.ts`
- All API routes must validate input with Zod schemas
- All database queries must use parameterized queries (no string interpolation)
- Error responses follow RFC 7807 Problem Details format
- All endpoints must return within 200ms at P99
- Background jobs must be idempotent and include a deduplication key
```

The effective version is three times longer but eliminates an entire class of review failures. Every rule Claude follows automatically is a rule you do not have to catch manually.

---

## Hook Systems: Automated Quality Gates

CLAUDE.md tells Claude Code what to do. Hooks make sure it actually did it.

A hook is a shell command that executes automatically at specific points in Claude Code's tool execution lifecycle. When Claude edits a file, a hook can run the linter. When Claude is about to execute a bash command, a hook can check whether the command is safe. When Claude finishes a task, a hook can run the full test suite. The key word is *automatic* --- hooks do not require the developer to remember to run checks. They run every time, on every operation, without exception.

### Hook Types

Claude Code supports three hook types, each triggered at a different point in the tool lifecycle:

**PreToolUse** --- runs before a tool is executed. Use these to validate or block operations before they happen. A PreToolUse hook on the Bash tool can prevent dangerous commands (rm -rf, dropping database tables, pushing to production). A PreToolUse hook on Edit or Write can check whether the target file is in a protected directory.

**PostToolUse** --- runs after a tool completes. Use these to validate the result. A PostToolUse hook on Edit or Write can run the linter, the type checker, or the relevant test suite immediately after a file change. This is the most commonly used hook type because it enforces quality at the point of change.

**Notification** --- runs when Claude Code produces a notification event, such as when it finishes a long-running task or needs user attention.

### Configuration

Hooks are configured in `.claude/settings.json` (shared with the team via version control) or `.claude/settings.local.json` (local overrides, git-ignored). The format:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "npm run lint -- --fix 2>&1 | tail -20"
          }
        ]
      },
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "npx tsc --noEmit 2>&1 | tail -30"
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "echo \"$CLAUDE_TOOL_INPUT\" | grep -qE '(rm -rf|drop table|git push.*--force)' && echo 'BLOCKED: Dangerous command detected' && exit 1 || exit 0"
          }
        ]
      }
    ]
  }
}
```

The `matcher` field is a regex pattern that determines which tool invocations trigger the hook. `"Edit|Write"` triggers on both Edit and Write tools. `"Bash"` triggers on all bash commands. You can make matchers as specific as needed.

### Practical Hook Examples

**Auto-lint on file change.** Every file edit triggers the linter. If the linter fails, Claude sees the failure and fixes it immediately, before moving on:

```json
{
  "matcher": "Edit|Write",
  "hooks": [{
    "type": "command",
    "command": "npx eslint --fix $(echo $CLAUDE_FILE_PATH) 2>&1 | tail -20"
  }]
}
```

**Type check on file change.** TypeScript type checking catches errors that the linter misses --- incorrect argument types, missing properties, incompatible return types:

```json
{
  "matcher": "Edit|Write",
  "hooks": [{
    "type": "command",
    "command": "npx tsc --noEmit 2>&1 | tail -30"
  }]
}
```

**Run relevant tests after changes.** Instead of running the full test suite on every edit (which can be slow), run only the tests relevant to the changed file:

```json
{
  "matcher": "Edit|Write",
  "hooks": [{
    "type": "command",
    "command": "npx vitest related $(echo $CLAUDE_FILE_PATH) --run 2>&1 | tail -40"
  }]
}
```

**Block dangerous commands.** Prevent Claude from executing commands that could damage the system:

```json
{
  "matcher": "Bash",
  "hooks": [{
    "type": "command",
    "command": "echo \"$CLAUDE_TOOL_INPUT\" | grep -qiE '(rm -rf /|drop database|git push.*force|truncate table)' && echo 'BLOCKED' && exit 1 || exit 0"
  }]
}
```

### The Hook Philosophy

If a standard is important enough to be in CLAUDE.md, it should be enforced by a hook. CLAUDE.md says "never use `any` type." The PostToolUse hook runs `tsc --noEmit` after every edit and catches it immediately. CLAUDE.md says "all functions must have tests." The PostToolUse hook runs the test suite and flags untested code.

The relationship between CLAUDE.md and hooks is the same as the relationship between a company's values statement and its org chart: the values tell you what matters, the structure makes it happen. CLAUDE.md is declarative (what the standards are). Hooks are imperative (how the standards are enforced).

The goal is simple: Claude Code cannot produce a commit that violates your standards. Not because Claude is perfectly compliant --- it is not --- but because the automated checks catch every deviation at the point of change, giving Claude immediate feedback and the opportunity to fix the issue before moving on.

---

## MCP Servers: Extending Claude's Capabilities

Claude Code ships with a set of built-in tools: file reading and editing, bash command execution, web search, and a few others. These cover general-purpose development. They do not cover your project's specific needs. If your workflow requires querying a database, interacting with a project management tool, accessing a documentation site, or running domain-specific analysis, you need to extend Claude's toolkit.

MCP --- Model Context Protocol --- is the mechanism for this extension. It is an open protocol, developed by Anthropic, that standardizes how AI applications connect to external tools and data sources. Think of it as USB for AI: a universal interface that any tool can implement to become accessible to any MCP-compatible host. Claude Code is an MCP host. MCP servers provide tools. The protocol handles the communication.

### How MCP Works

The architecture is client-server. Claude Code (the host) connects to one or more MCP servers, each of which exposes a set of tools. When Claude needs to perform an action that an MCP server handles --- querying a database, fetching documentation, creating a GitHub issue --- it calls the tool through the MCP protocol. The server executes the action and returns the result. From Claude's perspective, MCP tools work identically to built-in tools.

MCP servers run locally on your machine. They are processes that start when Claude Code launches and communicate over standard I/O. This means they have the permissions of the user running them, which is both powerful (they can access local databases, file systems, and network services) and dangerous (they can access anything the user can access).

### Configuration

MCP servers are configured in `.claude/settings.json` or added via the CLI:

```bash
# Add an MCP server via CLI
claude mcp add postgres-db -- npx -y @modelcontextprotocol/server-postgres \
  "postgresql://localhost:5432/mydb"

# Add a filesystem server
claude mcp add project-docs -- npx -y @modelcontextprotocol/server-filesystem \
  /path/to/documentation
```

The equivalent configuration in `.claude/settings.json`:

```json
{
  "mcpServers": {
    "postgres-db": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres",
               "postgresql://localhost:5432/mydb"]
    },
    "project-docs": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem",
               "/path/to/documentation"]
    }
  }
}
```

### Essential MCP Servers

**Database access.** The PostgreSQL and SQLite MCP servers let Claude query your database directly. Instead of asking Claude to write a query and then running it yourself, Claude executes the query, sees the result, and reasons about it in context. This is useful for debugging data issues, verifying migration results, and understanding the current state of your data during development.

**GitHub integration.** The GitHub MCP server provides access to issues, pull requests, repositories, and code review. Claude can read PR comments, check CI status, and understand the context of code review feedback without leaving the development session.

**Browser and documentation access.** The Puppeteer MCP server enables browser automation, and the filesystem server provides read access to local documentation directories. Together, these let Claude read project documentation, API references, and style guides that are not in the codebase itself.

**Memory and knowledge graph.** The memory MCP server provides Claude with persistent storage for key-value pairs and simple knowledge graphs. This is useful for maintaining context across sessions without putting everything in CLAUDE.md.

### Building Custom MCP Servers

When your project needs a tool that no existing MCP server provides, you build one. The MCP specification defines the protocol; you implement the tools.

**When to build one.** When Claude needs to interact with a domain-specific tool --- your internal deployment system, your company's documentation wiki, a custom data pipeline, or a specialized analysis tool. The threshold is simple: if you find yourself repeatedly copying output from one tool and pasting it into Claude's conversation, that tool should be an MCP server.

**Basic structure.** An MCP server is a program that listens on standard I/O, declares its available tools, and handles tool invocation requests. The `@modelcontextprotocol/sdk` npm package (or the Python equivalent) provides the scaffolding:

```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new Server({
  name: "my-custom-tool",
  version: "1.0.0"
}, {
  capabilities: { tools: {} }
});

server.setRequestHandler("tools/list", async () => ({
  tools: [{
    name: "query_metrics",
    description: "Query application metrics from the monitoring system",
    inputSchema: {
      type: "object",
      properties: {
        metric: { type: "string", description: "Metric name" },
        timeRange: { type: "string", description: "Time range (e.g., '1h', '24h')" }
      },
      required: ["metric"]
    }
  }]
}));

server.setRequestHandler("tools/call", async (request) => {
  if (request.params.name === "query_metrics") {
    const result = await fetchMetrics(request.params.arguments);
    return { content: [{ type: "text", text: JSON.stringify(result) }] };
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
```

### Security Considerations

**Principle of least privilege.** Only expose the operations Claude actually needs. If Claude needs to read the database but not write to it, configure read-only access. If Claude needs to query one table, do not give it access to all tables.

**Credential management.** Never pass secrets through MCP server configuration in `.claude/settings.json` --- that file is checked into version control. Use `.claude/settings.local.json` (git-ignored) for configurations that include credentials, or use environment variables:

```json
{
  "mcpServers": {
    "database": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "${DATABASE_URL}"
      }
    }
  }
}
```

**Network isolation.** MCP servers run locally with your user permissions. A malicious or poorly written MCP server can access your file system, make network requests, and read environment variables. Audit third-party MCP server code before using it, the same way you would audit any third-party dependency.

---

## Test-Driven Development with Claude Code

TDD is not a testing practice. It is a specification practice. The test defines what "correct" means. When you write the test before the implementation, you are not testing code that does not exist yet --- you are specifying behavior that the implementation must satisfy. This distinction matters enormously for AI-assisted development because it inverts the usual bottleneck.

In traditional development, the bottleneck is implementation: you know what the code should do, and you spend time writing it. With Claude Code, implementation is fast --- Claude can write a function in seconds. The new bottleneck is specification: making sure the code does the *right* thing, not just *a* thing. TDD addresses this directly. The human specifies correctness (by writing the test). Claude implements it (by writing the code that passes the test). The human reviews the implementation for domain correctness and edge cases.

### Red-Green-Refactor for AI

The classic TDD cycle is Red-Green-Refactor. With Claude Code, each phase has a clear owner:

**Red (human writes).** Write a failing test that defines the desired behavior. This is the specification step. It requires domain knowledge --- you must know what correct behavior looks like:

```typescript
// Human writes this test
describe("calculateShippingCost", () => {
  it("returns free shipping for orders over $100", () => {
    expect(calculateShippingCost({ subtotal: 150, weight: 5 })).toBe(0);
  });

  it("charges $5.99 flat rate for orders under $100 and under 2kg", () => {
    expect(calculateShippingCost({ subtotal: 50, weight: 1.5 })).toBe(5.99);
  });

  it("charges $5.99 plus $2 per kg over 2kg for heavy orders under $100", () => {
    expect(calculateShippingCost({ subtotal: 50, weight: 4 })).toBe(9.99);
  });

  it("throws on negative subtotal", () => {
    expect(() => calculateShippingCost({ subtotal: -10, weight: 1 })).toThrow();
  });
});
```

The test is a contract. It says: this function takes a subtotal and weight, returns a number, has three pricing tiers, and rejects invalid input. Claude does not need to know the business logic --- it is encoded in the test.

**Green (Claude implements).** Ask Claude to implement the function that passes all tests. Claude reads the test, understands the specification, and writes the minimum code to pass:

```typescript
// Claude writes this implementation
export function calculateShippingCost({
  subtotal,
  weight,
}: {
  subtotal: number;
  weight: number;
}): number {
  if (subtotal < 0) throw new Error("Subtotal cannot be negative");
  if (subtotal >= 100) return 0;
  const baseCost = 5.99;
  const extraWeight = Math.max(0, weight - 2);
  return baseCost + extraWeight * 2;
}
```

The PostToolUse hook runs the tests immediately. If they pass, the implementation is correct by definition --- correct relative to the specification, which is all the tests can verify.

**Refactor (human reviews).** Claude may propose a refactor --- extracting constants, improving naming, adding input validation. The human reviews for domain correctness: does the free shipping threshold match the business rule? Is the weight unit correct (kilograms, not pounds)? Are there edge cases the tests missed (zero weight, exactly $100 subtotal, maximum weight limits)?

### Property-Based Testing

Claude is particularly effective at generating property-based tests. In traditional testing, you specify concrete examples (input X should produce output Y). In property-based testing, you specify *invariants* --- properties that must hold for all possible inputs --- and a testing framework generates hundreds of random inputs to verify them.

The human defines the invariants. Claude implements the test harness:

```python
# Human defines: "Shipping cost must never be negative"
# Claude generates:
from hypothesis import given, strategies as st

@given(
    subtotal=st.floats(min_value=0, max_value=10000),
    weight=st.floats(min_value=0, max_value=100)
)
def test_shipping_cost_never_negative(subtotal, weight):
    result = calculate_shipping_cost(subtotal=subtotal, weight=weight)
    assert result >= 0

@given(
    subtotal=st.floats(min_value=100, max_value=10000),
    weight=st.floats(min_value=0, max_value=100)
)
def test_free_shipping_above_threshold(subtotal, weight):
    result = calculate_shipping_cost(subtotal=subtotal, weight=weight)
    assert result == 0
```

Property-based testing catches edge cases that example-based tests miss. The human identifies what *must always be true*. The framework finds inputs that violate those invariants. Claude writes the test structure and the strategy definitions.

### Integration Testing

Unit tests verify components. Integration tests verify wiring --- that components work together correctly. Claude writes integration tests well when the system's boundaries and interactions are defined in CLAUDE.md:

```markdown
# Integration Testing Patterns
- API tests use `supertest` against the real server (not mocks)
- Database tests run against a test database seeded with fixtures from `tests/fixtures/`
- External API calls are recorded with `nock` — never hit real external services in tests
- Each test file creates and tears down its own data — no shared state between test files
```

With these rules in CLAUDE.md, Claude generates integration tests that follow the team's patterns: real database, real server, isolated fixtures, no shared state.

---

## Code Review Discipline

Code review for AI-generated code requires a different focus than review of human-written code. Traditional reviews check formatting, naming, structure, and adherence to patterns. Claude handles all of these well --- its code is consistently formatted, reasonably named, and structurally plausible. Reviewing AI-generated code for formatting is like proofreading a spell-checker's output: you will rarely find errors, and the time spent checking is wasted.

The errors in AI-generated code are semantic, not syntactic. The function is named correctly and typed correctly, but it retries non-idempotent operations. The database query returns the right data, but it does a table scan instead of using an index. The error handler catches the exception, but it swallows the stack trace. These are the errors that cause production incidents, and they require a different kind of review.

### The Semantic Review Checklist

Six questions to ask about every piece of AI-generated code. For each, the key is not whether the code looks right --- it is whether the code *is* right for your specific system.

**1. Domain correctness.** Does this abstraction map to your actual domain, or is it a plausible-looking generic? Claude generates code that fits common patterns. If your domain has uncommon constraints --- a billing system where partial refunds must preserve the original transaction's tax rate, a scheduling system where time zones are per-user rather than per-organization --- Claude will produce code that works for the common case and fails for yours. Check that the code's assumptions match your domain's rules.

**2. Failure modes.** What happens at 3 AM with a full queue, a slow database, and a rate-limited API? Claude writes for the happy path with reasonable error handling. It rarely anticipates cascading failures: what happens when the retry exhausts its budget, the circuit breaker opens, and the fallback service is also degraded? Walk through the failure scenarios that matter for your system and check that the code handles them.

**3. Observability.** Where is the logging? Where are the metrics? Can you debug this in production? Claude often omits logging because CLAUDE.md did not specify a logging pattern. Check that every error path produces a structured log entry with a correlation ID, that business-critical operations emit metrics, and that the code is debuggable without adding logging after the fact.

**4. Edge cases.** What happens with empty input, null values, maximum-size payloads, concurrent access, and clock skew? Claude handles obvious edge cases (null checks, empty arrays) but misses domain-specific ones (what happens when two users submit the same form simultaneously, what happens when the clock rolls over midnight during a transaction).

**5. Assumptions.** What does this code assume about input shape, system state, ordering, or availability? Every function makes assumptions. Claude's assumptions are implicit --- derived from patterns in its training data rather than from your system's actual constraints. Make them explicit: what does this code assume about the database connection being available, the user object having a certain shape, the queue being FIFO?

**6. Dependencies.** Did Claude add a new package? Is it necessary? Is it maintained? Claude sometimes reaches for a library when a few lines of code would suffice. Check whether the dependency is actually needed, whether it is actively maintained, whether it has known vulnerabilities, and whether it aligns with your project's dependency policy.

### The Ownership Model

You prompted it, you own it. This principle is simple and non-negotiable.

Ownership means three things:

**You have read every line.** Not skimmed --- read. You understand what each function does, what each condition checks, and what each error handler catches.

**You can explain every decision.** Why did the code use a Map instead of an object? Why does the retry use exponential backoff with jitter? Why is the cache TTL 300 seconds? If you cannot explain these decisions, you do not understand the code well enough to own it.

**You can modify without re-prompting.** If the requirements change, can you update the code yourself? Or would you need to go back to Claude and re-prompt from scratch? If the latter, you do not have sufficient understanding to maintain the code.

The PR review is where ownership transfers from model to human. Before that transfer, the code is Claude's suggestion. After the review, it is your code. Treat the review accordingly --- with the same rigor you would apply to code written by a junior engineer who is talented but does not know your system.

---

## Project Structure and Architecture Patterns

How you organize your codebase affects how effectively Claude Code can work with it. Claude reads files to understand context. The more context it needs to load for a given task, the slower and less accurate it becomes. File organization is not just a human readability concern --- it is a context window efficiency concern.

### Feature-Based vs. Layer-Based

Layer-based organization groups files by technical role:

```
src/
  controllers/
    userController.ts
    orderController.ts
    productController.ts
  services/
    userService.ts
    orderService.ts
    productService.ts
  models/
    user.ts
    order.ts
    product.ts
  routes/
    userRoutes.ts
    orderRoutes.ts
    productRoutes.ts
```

Feature-based organization groups files by domain concept:

```
src/
  users/
    controller.ts
    service.ts
    model.ts
    routes.ts
    users.test.ts
  orders/
    controller.ts
    service.ts
    model.ts
    routes.ts
    orders.test.ts
  products/
    controller.ts
    service.ts
    model.ts
    routes.ts
    products.test.ts
```

For Claude Code, feature-based wins. When Claude is working on the orders feature, it needs the controller, service, model, routes, and tests. In a feature-based structure, these are all in one directory --- Claude reads the directory and has full context. In a layer-based structure, Claude needs to read files from four different directories, loading unrelated code (user and product controllers) along the way. The wasted context is not just inefficient; it increases the chance that Claude will confuse patterns from unrelated features.

### Context Window Efficiency

Claude Code has a large but finite context window. Every file it reads consumes part of that window. Three practices keep context usage efficient:

**Small, focused files.** A 200-line file that does one thing is better than a 2,000-line file that does ten things. When Claude needs to modify one behavior, it reads only the relevant file instead of the entire module.

**Clear module boundaries.** Each directory should have a clear public interface. A `users/index.ts` that re-exports the public API lets Claude understand the module's contract without reading its internals.

**Dependency direction.** Code should depend in one direction --- features depend on shared utilities, not on each other. When Claude modifies the orders feature, it should not need to understand the users feature. Circular dependencies between features mean that modifying one requires understanding both, doubling the context load.

### Dependency Management

Claude Code will add dependencies if you do not tell it not to. It has seen millions of npm packages in its training data and will reach for a library whenever one exists for the task at hand. This is often the wrong choice --- a library that saves ten lines of code adds a maintenance burden, a security surface, and a build dependency that outlasts the ten lines.

CLAUDE.md should include explicit dependency rules:

```markdown
# Dependencies
- NEVER add new npm dependencies without explicit approval
- Prefer standard library solutions over third-party packages
- If a dependency is necessary, check: is it actively maintained? Does it have known vulnerabilities? Is it < 50KB?
- Allowed HTTP client: `fetch` (built-in). NEVER use axios, got, or node-fetch.
- Allowed ORM: Prisma. NEVER use TypeORM, Sequelize, or Drizzle.
```

---

## Skills and Workflows

A skill is a reusable development workflow packaged as a prompt. When you type `/commit` in Claude Code, you are invoking a skill: a predefined set of instructions that tells Claude how to analyze changes, write a commit message, and execute the commit. Skills turn repetitive multi-step workflows into single commands.

### Built-In Skills

Claude Code ships with skills for common operations:

**`/commit`** analyzes staged and unstaged changes, drafts a commit message following repository conventions, and creates the commit. It reads recent commit history to match the project's message style.

**`/review-pr`** reviews a pull request by reading the diff, checking for the semantic issues described in the Code Review Discipline section, and producing structured feedback.

These built-in skills are useful but generic. The real power is in custom skills.

### Custom Skills

A custom skill encodes your team's specific workflow as a prompt. The prompt defines the steps, the quality criteria, and the expected output. Skills can be composed --- one skill can invoke another --- creating multi-step workflows that maintain consistency across the team.

Example: a custom deploy skill for a team that deploys via GitHub Actions:

```markdown
## Deploy Skill

1. Run the full test suite: `npm test`
2. If tests fail, stop and report failures
3. Build the production bundle: `npm run build`
4. Check bundle size against the budget in CLAUDE.md (max 500KB gzipped)
5. If bundle exceeds budget, identify the largest chunks and suggest optimizations
6. Create a git tag with the next semantic version
7. Push the tag to trigger the deployment pipeline
8. Monitor the GitHub Actions deployment workflow
9. Report success or failure with the deployment URL
```

This skill runs nine steps. Without it, the developer either executes each step manually (error-prone, slow) or writes a shell script (rigid, hard to modify). As a skill, it is flexible --- Claude adapts to the current state (different test failures, different bundle sizes) while following the same workflow.

### Subagents

Claude Code can dispatch work to subagents --- separate Claude instances that handle specific tasks in their own context. Subagents are useful when:

**The task is independent.** Researching a codebase question while also implementing a feature --- two independent tasks that do not share context.

**The context should be isolated.** Reviewing code benefits from a fresh perspective. A subagent that reviews code does not carry the bias of having written it.

**The work can be parallelized.** Multiple independent research tasks (searching for files, reading documentation, checking test coverage) can run simultaneously as subagents, returning results to the main session.

Subagents should not be used for tasks that require shared state or sequential coordination. If step 2 depends on the result of step 1, do both in the main session.

---

## Performance-Aware Development

Claude Code produces correct code by default. It does not produce performant code by default. Performance requires constraints, and constraints come from CLAUDE.md.

Without explicit performance rules, Claude makes reasonable but generic choices: it uses the first algorithm that works, the first data structure that fits, the first query that returns the right result. These choices are often fine for prototypes and catastrophic at scale. A sequential scan on a 100-row table returns instantly. The same scan on a 10-million-row table takes seconds and locks the table.

### Teaching Claude About Performance

Performance constraints go in CLAUDE.md as non-negotiable rules:

```markdown
# Performance Requirements
- All API endpoints must respond in < 200ms at P99
- All database queries must use indexes — sequential scans are bugs
- No N+1 query patterns — use eager loading or batch queries
- Pagination is mandatory for list endpoints (max 100 items per page)
- Background jobs must complete within 30 seconds or be broken into subtasks
- Memory usage per request must not exceed 50MB
```

These rules are testable. A PostToolUse hook can run `EXPLAIN ANALYZE` on new queries and flag sequential scans. A load testing script can verify P99 latency. The rules in CLAUDE.md define the targets; hooks and tests enforce them.

### Profiling Integration

Profiling should be part of the development workflow, not an afterthought triggered by a production incident. CLAUDE.md can mandate profiling for performance-critical paths:

```markdown
# Profiling Rules
- Any change to a query in `src/api/search/` must include EXPLAIN ANALYZE output
- Any change to the rendering pipeline must include a lighthouse score comparison
- New endpoints must include a k6 load test script in `tests/load/`
```

A PostToolUse hook can enforce this by checking whether the developer (or Claude) included profiling output with changes to performance-critical files.

### Load Testing as a Claude Code Workflow

Claude Code is effective at writing load test scripts when given the test scenarios in CLAUDE.md. Tools like k6 and Artillery use declarative configurations that Claude generates well:

```markdown
# Load Testing
- k6 scripts go in `tests/load/`
- Each API endpoint must have a load test
- Standard test profile: 50 concurrent users, 5-minute ramp-up, 10-minute sustained load
- Success criteria: P99 < 200ms, error rate < 0.1%, no memory leaks
```

With this in CLAUDE.md, asking Claude to "write a load test for the search endpoint" produces a k6 script that follows the team's conventions, targets the right metrics, and uses the standard test profile.

---

## The Production Readiness Checklist

Everything in this article reduces to a checklist. Before shipping a Claude Code project to production, verify each item.

### CLAUDE.md

- [ ] Coding standards defined (language version, framework, formatting rules)
- [ ] Architectural decisions documented (patterns to use, patterns to avoid)
- [ ] Forbidden patterns listed (with specific "NEVER do X" rules)
- [ ] Test requirements specified (coverage targets, testing patterns)
- [ ] Performance constraints defined (latency budgets, resource limits)
- [ ] Domain terminology defined (terms that have project-specific meanings)
- [ ] Dependency policy documented (approval process, allowed packages)
- [ ] Error handling conventions specified (logging, retries, circuit breakers)

### Hooks

- [ ] Linting runs on every file change (PostToolUse on Edit/Write)
- [ ] Type checking runs on every file change
- [ ] Relevant tests run on every file change
- [ ] Dangerous commands are blocked (PreToolUse on Bash)
- [ ] Formatting is automated (not left to manual review)

### Testing

- [ ] Unit test coverage exceeds 80% on business logic
- [ ] Integration tests exist for critical API paths
- [ ] Property-based tests verify key invariants
- [ ] Load test scripts exist for performance-critical endpoints
- [ ] Tests run against real dependencies (database, not mocks)

### Code Review

- [ ] Semantic review checklist used for all AI-generated PRs
- [ ] Domain correctness verified (not just syntax)
- [ ] Failure modes evaluated (not just happy path)
- [ ] Observability verified (logging, metrics, tracing)
- [ ] Ownership model followed (can explain every line)

### CI/CD

- [ ] All hooks run in CI (not just locally)
- [ ] Deployment is automated (no manual steps)
- [ ] Rollback procedure documented and tested
- [ ] Feature flags available for risky changes

### Observability

- [ ] Structured logging configured with correlation IDs
- [ ] Metrics exported (latency, error rate, throughput)
- [ ] Alerting configured for error rate and latency anomalies
- [ ] Dashboard exists for key business and system metrics

### Security

- [ ] No secrets in source code or CLAUDE.md
- [ ] Input validation on all external boundaries (API endpoints, webhooks)
- [ ] Dependencies audited for known vulnerabilities
- [ ] MCP servers configured with least privilege
- [ ] Sensitive MCP configurations in `.claude/settings.local.json` (git-ignored)

This checklist is not exhaustive. It is the minimum. Every item on it has cost teams production incidents when omitted. Claude Code does not make these items easier or harder to implement --- it makes them easier to forget, because the code it produces looks so polished that the gaps are invisible until 3 AM.

The antidote to invisible gaps is visible checklists. Run through this list before every production deployment, and the gaps will not survive to production.
