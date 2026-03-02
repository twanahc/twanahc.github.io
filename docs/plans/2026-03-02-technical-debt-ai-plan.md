# Technical Debt and AI Usage — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Write a comprehensive standalone blog post about technical debt in the age of AI-assisted coding, with rigorous definitions, quantitative models, and practical management strategies.

**Architecture:** Single markdown post in `_posts/` following Jekyll conventions. Six sections building from definitions to math models to AI-specific patterns to strategies. Two matplotlib visualizations embedded as Python code blocks. Two SVG diagrams inline. Moderate math via MathJax.

**Tech Stack:** Jekyll markdown, MathJax (inline: `\\(...\\)`, display: `$$...$$`), matplotlib + numpy for plots, inline SVG for diagrams.

**Design doc:** `docs/plans/2026-03-02-technical-debt-ai-design.md`

---

### Task 1: Create the post file with front matter and Table of Contents

**Files:**
- Create: `_posts/2026-03-02-technical-debt-ai-usage.md`

**Step 1: Write the file with front matter and TOC**

Create the post file with this exact content:

```markdown
---
layout: post
title: "Technical Debt in the Age of AI: The Compounding Cost of Code You Didn't Write"
date: 2026-03-02
category: infra
---

[Opening paragraph — Task 2]

---

## Table of Contents

1. [What Technical Debt Actually Means](#what-technical-debt-actually-means)
2. [The Mathematics of Debt Accumulation](#the-mathematics-of-debt-accumulation)
3. [How AI Tools Change the Debt Equation](#how-ai-tools-change-the-debt-equation)
4. [The Six Patterns of AI-Generated Debt](#the-six-patterns-of-ai-generated-debt)
5. [The Detection Problem](#the-detection-problem)
6. [Quantifying the Cost: When Debt Kills You](#quantifying-the-cost-when-debt-kills-you)
7. [Managing AI-Generated Debt](#managing-ai-generated-debt)
8. [Conclusion](#conclusion)

---
```

**Step 2: Commit**

```bash
git add _posts/2026-03-02-technical-debt-ai-usage.md
git commit -m "feat: scaffold technical debt article with front matter and TOC"
```

---

### Task 2: Write Section 1 — Opening & Setup

**Files:**
- Modify: `_posts/2026-03-02-technical-debt-ai-usage.md`

**Step 1: Write the opening**

Replace `[Opening paragraph — Task 2]` with the opening content. Requirements:

- **Hook:** A concrete scenario — shipped a feature in 2 hours with Cursor/Claude, a month later it's the source of 40% of production incidents. Code "works" but nobody can explain why it does what it does.
- **Setup:** Technical debt is the most abused metaphor in software engineering. Everyone uses it, almost nobody defines it precisely. This post fixes that.
- **Thesis:** "AI tools don't create a new kind of debt. They accelerate the accumulation of every existing kind, while simultaneously making the debt harder to detect because the code looks right."
- **Tone:** Match the opening of `2026-03-02-vibe-code-to-production-performance-engineering.md` — concrete scenario first, then zoom out to the problem, then state what the post will do.
- **Length:** 3-4 paragraphs, ~200-300 words.

**Step 2: Commit**

```bash
git add _posts/2026-03-02-technical-debt-ai-usage.md
git commit -m "feat: write opening section for technical debt article"
```

---

### Task 3: Write Section 2 — What Technical Debt Actually Means

**Files:**
- Modify: `_posts/2026-03-02-technical-debt-ai-usage.md`

**Step 1: Write the rigorous definition section**

Append after the TOC/opening. Requirements:

- **Section header:** `## What Technical Debt Actually Means`
- **Ward Cunningham's original metaphor:** Briefly credit the origin (1992), then formalize it.
- **Financial mapping as a definition list or bold terms:**
  - **Principal** — the shortcut itself. The gap between what you built and what you should have built. Quantifiable as refactoring cost.
  - **Interest** — the ongoing cost of working around the shortcut. Every feature takes longer, every bug harder to find. Measured as velocity drag.
  - **Interest rate** — how fast debt compounds. Determined by coupling. Isolated debt = low rate. Foundational debt (auth, data models, core abstractions) = crushing rate.
  - **Compounding** — debt creates more debt. Bad abstraction forces workarounds, workarounds become load-bearing, two layers of debt.
- **Martin Fowler debt quadrant** as an SVG diagram or HTML table. 2x2 grid:
  - Rows: Deliberate / Inadvertent
  - Columns: Prudent / Reckless
  - Cells with one-line examples:
    - Deliberate+Prudent: "We know this is a shortcut, ship now, refactor next sprint"
    - Deliberate+Reckless: "We don't have time for design"
    - Inadvertent+Prudent: "Now we know how this should have been built"
    - Inadvertent+Reckless: "What's layering?"
- **Why the quadrant matters:** Set up that AI tools interact differently with each cell. Tease this for the next section.
- **End with `---` horizontal rule.**
- **Math:** No MathJax equations in this section — definitions only.
- **Length:** ~400-500 words.

**Step 2: Commit**

```bash
git add _posts/2026-03-02-technical-debt-ai-usage.md
git commit -m "feat: write technical debt definitions section"
```

---

### Task 4: Write Section 3 — The Mathematics of Debt Accumulation

**Files:**
- Modify: `_posts/2026-03-02-technical-debt-ai-usage.md`

**Step 1: Write the math section**

Append after Section 2. Requirements:

- **Section header:** `## The Mathematics of Debt Accumulation`
- **Opening:** "The financial metaphor is not just an analogy — it can be formalized."
- **Velocity decay model:** Define velocity \\(V(t)\\) as a function of accumulated debt \\(D(t)\\):

  $$V(t) = V_0 \cdot e^{-\alpha D(t)}$$

  Explain each term:
  - \\(V_0\\) = initial development velocity (features per unit time)
  - \\(D(t)\\) = accumulated debt at time \\(t\\)
  - \\(\alpha\\) = debt sensitivity coefficient (how much debt slows you down)
  - Explain WHY exponential: debt doesn't reduce velocity linearly. The first 10 shortcuts barely slow you down. The 100th makes everything take 3x longer. This is the compounding effect.

- **Debt dynamics ODE:**

  $$\frac{dD}{dt} = \beta \cdot R(t) - \gamma \cdot P(t)$$

  Where:
  - \\(R(t)\\) = rate of new code production (features shipped per unit time)
  - \\(P(t)\\) = paydown effort (refactoring, cleanup)
  - \\(\beta\\) = debt creation rate per unit of new code (quality coefficient, 0 = perfect code, 1 = pure debt)
  - \\(\gamma\\) = paydown efficiency

  Explain: when \\(\beta R(t) > \gamma P(t)\\), debt grows. When the inequality flips, debt shrinks. Most teams never flip it.

- **The feedback loop:** Combine the two equations. Higher \\(D(t)\\) reduces \\(V(t)\\), which pressures the team to ship faster (increase \\(R(t)\\)) with less care (increase \\(\beta\\)), which increases \\(D(t)\\) further. This is the debt spiral.

- **Caveat:** This is not a predictive model. You cannot measure \\(\alpha\\) or \\(\beta\\) precisely. The value is in formalizing the qualitative dynamics: debt compounds, velocity decays exponentially, and there is a feedback loop that makes things worse.

- **End with `---` horizontal rule.**

- **MathJax notes:**
  - Inline math: `\\(...\\)` (double-escaped for kramdown)
  - Display math: `$$...$$`
  - Escape `\*` inside inline math where needed
  - Use `\cdot` for multiplication, not `*`

- **Length:** ~400-500 words.

**Step 2: Commit**

```bash
git add _posts/2026-03-02-technical-debt-ai-usage.md
git commit -m "feat: write debt accumulation math section"
```

---

### Task 5: Write Section 4 — How AI Tools Change the Debt Equation

**Files:**
- Modify: `_posts/2026-03-02-technical-debt-ai-usage.md`

**Step 1: Write the AI throughput section**

Append after Section 3. Requirements:

- **Section header:** `## How AI Tools Change the Debt Equation`
- **The throughput multiplier argument:**
  - AI tools (Cursor, Claude, Copilot, ChatGPT) increase \\(R(t)\\) by 3-10x.
  - If debt creation rate is proportional to code output, you accumulate debt 3-10x faster.
  - The denominator hasn't changed: team's capacity to understand, review, and refactor code is the same.
  - Reference the ODE: \\(\beta \cdot R(t)\\) just got multiplied. \\(\gamma \cdot P(t)\\) didn't.
  - This alone shifts the equilibrium toward perpetual debt growth.
- **The deeper problem:** AI doesn't just increase \\(R(t)\\) — it also increases \\(\beta\\) (the debt creation rate per unit of code). Human-written code at speed creates debt at a known rate. AI-generated code creates debt at a higher rate because of specific patterns we'll examine next.
- **Bridge to next section:** "What are those patterns? There are six that recur across every codebase we've seen built with AI assistance."
- **End with `---` horizontal rule.**
- **Length:** ~250-350 words. Keep this section tight — it's the bridge between math and patterns.

**Step 2: Commit**

```bash
git add _posts/2026-03-02-technical-debt-ai-usage.md
git commit -m "feat: write AI throughput multiplier section"
```

---

### Task 6: Write Section 5 — The Six Patterns of AI-Generated Debt

**Files:**
- Modify: `_posts/2026-03-02-technical-debt-ai-usage.md`

**Step 1: Write the six patterns**

Append after Section 4. Requirements:

- **Section header:** `## The Six Patterns of AI-Generated Debt`
- **Each pattern gets a subsection (H3)** with: the pattern name, a 1-2 paragraph explanation of what it is and why AI produces it, and a concrete example (description of the code smell, not actual code blocks — keep it prose).
- **The six patterns:**

  **### 1. Plausible but Wrong Abstractions**
  AI generates code that "looks like" good design — proper class hierarchy, separation of concerns, design patterns. But the abstractions don't map to your actual domain. You get an elegant solution to the wrong problem. AI has no context about your business domain, your team's conventions, or what will change next. It pattern-matches from training data, producing architecturally beautiful code that forces your real requirements into the wrong shape.

  **### 2. Cargo-Cult Error Handling**
  try/catch blocks everywhere, generic error messages, swallowed exceptions. The code handles errors the way a student who read about error handling would: structurally correct, operationally useless. When something fails at 3 AM, the logs say "An error occurred" and nothing else. The error handling exists to satisfy the appearance of robustness, not to actually help you diagnose failures.

  **### 3. Dependency Hoarding**
  AI reaches for libraries by default. A 5-line string parsing function becomes an import of a 200KB package. A date formatting task pulls in moment.js. Multiply across a codebase and your dependency tree is a liability — each dependency is a maintenance burden, a security surface, and a breaking-change risk you didn't sign up for.

  **### 4. Implicit Coupling Through Copy-Paste Evolution**
  AI generates similar-but-not-identical code across files because each prompt is independent. No shared abstractions, no DRY. You end up with five slightly different implementations of the same business rule. Changes require finding and updating N copies, and you won't find all of them.

  **### 5. Missing Observability**
  AI writes code that produces correct outputs but is a black box in production. No structured logging, no metrics emission, no distributed tracing context. When it fails, you have no signal. When it's slow, you don't know where. The code was written to satisfy a functional requirement, not an operational one.

  **### 6. Confident Nonsense in Edge Cases**
  AI handles the happy path beautifully and generates plausible-looking code for edge cases that is subtly wrong. Off-by-one errors wrapped in clean syntax. Race conditions in code that reads like it was written by someone who understands concurrency. The confidence of the code's style does not match the correctness of its logic.

- **End with `---` horizontal rule.**
- **No math in this section.**
- **Length:** ~600-800 words total across all six patterns.

**Step 2: Commit**

```bash
git add _posts/2026-03-02-technical-debt-ai-usage.md
git commit -m "feat: write six AI debt patterns section"
```

---

### Task 7: Write Section 6 — The Detection Problem

**Files:**
- Modify: `_posts/2026-03-02-technical-debt-ai-usage.md`

**Step 1: Write the detection problem section**

Append after Section 5. Requirements:

- **Section header:** `## The Detection Problem`
- **Core argument:** Traditional tech debt is often visibly ugly — long functions, obvious hacks, TODO comments. AI-generated debt reads well. It passes code review because reviewers pattern-match on surface quality signals (naming, structure, formatting) and AI nails all of those. The debt is semantic, not syntactic.
- **SVG diagram:** A comparison showing two columns:
  - Left column: "Surface Metrics" — formatting, naming conventions, code structure, cyclomatic complexity, lint score. AI scores HIGH on all.
  - Right column: "Semantic Metrics" — domain alignment, operational fitness, edge-case correctness, abstraction accuracy, failure-mode coverage. AI scores LOW on these.
  - Use a simple bar chart or radar-style SVG. Dark background colors consistent with blog style. Max width 700px, centered.
- **The implication:** Code review processes designed for human-written code are insufficient for AI-generated code. You need to review for different things. (Bridge to the strategies section.)
- **End with `---` horizontal rule.**
- **No math in this section.**
- **Length:** ~300-400 words + the SVG diagram.

**Step 2: Commit**

```bash
git add _posts/2026-03-02-technical-debt-ai-usage.md
git commit -m "feat: write detection problem section with SVG diagram"
```

---

### Task 8: Write Section 7 — Quantifying the Cost

**Files:**
- Modify: `_posts/2026-03-02-technical-debt-ai-usage.md`

**Step 1: Write the quantification section**

Append after Section 6. Requirements:

- **Section header:** `## Quantifying the Cost: When Debt Kills You`

- **Break-even analysis:**
  - Define time saved \\(S\\) per feature as the difference between manual and AI-assisted development time.
  - Define cumulative interest paid \\(I(t)\\) as total extra time spent debugging, working around, and extending indebted code.
  - Break-even point: \\(I(t) = S\\). After this, you're net negative.
  - Key insight: for well-isolated code (utility functions, one-off scripts), the crossover may never come — AI is pure upside. For foundational code (data models, auth, core pipeline), crossover can happen in weeks.

- **The "300-line threshold" observation:**
  - AI-generated code that nobody on the team has manually read, line by line, is not an asset — it's a liability with unknown magnitude.
  - The moment your codebase has more unreviewed AI code than reviewed code, you've lost the ability to reason about your own system.

- **Python visualization: Velocity decay plot.** Three scenarios plotted on the same axes:
  1. No AI tools, steady debt management (slow linear growth, stable velocity)
  2. AI tools with active review and paydown (faster growth, moderate velocity decay, sustainable)
  3. AI tools with no debt management (massive spike then cliff)

  Plot requirements:
  - Use `fig, ax = plt.subplots()` pattern
  - Use `plt.rcParams['text.usetex'] = False` with matplotlib mathtext for labels
  - X-axis: `r'Time (weeks)'`
  - Y-axis: `r'Team Velocity $V(t)$'`
  - Title: `r'Velocity Decay Under Three Debt Management Scenarios'`
  - Three curves with legend, clean publication style
  - Use numpy for computation
  - Save to PNG with dpi=150

- **Python visualization: Break-even crossover plot.**
  - X-axis: `r'Months After Shipping'`
  - Y-axis: `r'Cumulative Hours'`
  - Two curves per scenario: time saved (flat line) vs cumulative interest (growing curve)
  - Show 3 coupling levels: low (utility code), medium (feature code), high (foundational code)
  - Mark the crossover points
  - Same matplotlib style requirements as above

- **End with `---` horizontal rule.**
- **Length:** ~400-500 words + two Python code blocks.

**Step 2: Commit**

```bash
git add _posts/2026-03-02-technical-debt-ai-usage.md
git commit -m "feat: write cost quantification section with visualizations"
```

---

### Task 9: Write Section 8 — Managing AI-Generated Debt

**Files:**
- Modify: `_posts/2026-03-02-technical-debt-ai-usage.md`

**Step 1: Write the strategies section**

Append after Section 7. Requirements:

- **Section header:** `## Managing AI-Generated Debt`
- **Opening:** "The solution is not to stop using AI tools. The solution is to understand the debt equation and manage it deliberately."
- **Six strategies, each as an H3 subsection.** Each gets 1-2 paragraphs — concise, actionable, no fluff.

  **### 1. The Ownership Rule**
  If you prompted AI to write it, you own it. Ownership means you've read every line, you can explain why each decision was made, and you can modify it without re-prompting. If you can't do that, it's not your code — it's a dependency with no maintainer.

  **### 2. Prompt-Then-Rewrite**
  Use AI for the first draft, then rewrite it by hand. Not "edit" — rewrite. The AI gives you the structure and approach; you produce the code you'll actually maintain. The 2x speed reduction from maximum AI throughput is paid back immediately in comprehension.

  **### 3. Foundational Code Gets Human Supervision**
  Simple rule: the more coupled a piece of code is to the rest of the system, the less AI should write it unsupervised. Utility functions, tests, boilerplate — let AI rip. Data models, core abstractions, auth — write it yourself or treat the AI output as a rough sketch.

  **### 4. Review for Semantics, Not Syntax**
  Change what code review means when AI is involved. Stop checking formatting, naming, and structure — AI already nails those. Start asking: Does this abstraction map to our domain? What happens when this fails at 3 AM? Where's the logging? What are the implicit assumptions?

  **### 5. Scheduled Debt Audits**
  Every N sprints, pick a module that was primarily AI-generated and do a full read-through. Not refactoring — just reading. Build shared understanding. The read-through itself surfaces debt that pattern-matching review missed.

  **### 6. Track the Ratio**
  Maintain rough awareness of how much of your codebase was AI-generated vs human-written vs human-reviewed-AI-code. You don't need precise numbers. "Most of auth was hand-written, most of the API routes were AI-generated and reviewed, the test suite was AI-generated and never carefully read" is enough to know where your risk is.

- **End with `---` horizontal rule.**
- **No math in this section.**
- **Length:** ~500-600 words.

**Step 2: Commit**

```bash
git add _posts/2026-03-02-technical-debt-ai-usage.md
git commit -m "feat: write debt management strategies section"
```

---

### Task 10: Write Section 9 — Conclusion

**Files:**
- Modify: `_posts/2026-03-02-technical-debt-ai-usage.md`

**Step 1: Write the conclusion**

Append after Section 8. Requirements:

- **Section header:** `## Conclusion`
- **Three beats, one paragraph each:**
  1. Technical debt is a precisely definable quantity with compounding dynamics. Not a vibes thing — it has principal, interest, and a rate that depends on coupling.
  2. AI tools multiply your debt creation rate while making the debt harder to see. The code looks clean; the debt is semantic.
  3. The solution isn't to stop using AI tools — it's to understand the debt equation and manage it deliberately. Speed without comprehension is a loan you'll repay with interest.
- **Closing line:** "The best engineers using AI tools aren't the ones who ship the fastest. They're the ones who know exactly how much debt they're taking on and why."
- **No trailing `---` after conclusion.**
- **Length:** ~150-200 words. Short and sharp.

**Step 2: Commit**

```bash
git add _posts/2026-03-02-technical-debt-ai-usage.md
git commit -m "feat: write conclusion for technical debt article"
```

---

### Task 11: Final review and polish

**Files:**
- Modify: `_posts/2026-03-02-technical-debt-ai-usage.md`

**Step 1: Read the entire post end-to-end**

Read the complete post and check:
- All section headers match the TOC links (anchor IDs)
- All inline math uses `\\(...\\)` (double-escaped for kramdown)
- All `*` inside inline math is escaped as `\*`
- Display math uses `$$...$$`
- No `$...$` for inline math
- Horizontal rules (`---`) between all major sections
- Python code blocks use `fig, ax = plt.subplots()` pattern
- Python code uses `plt.rcParams['text.usetex'] = False` with matplotlib mathtext
- Tone is consistent throughout — conversational but precise, no fluff
- Every term is defined before use
- No trailing whitespace issues

**Step 2: Fix any issues found**

**Step 3: Final commit**

```bash
git add _posts/2026-03-02-technical-debt-ai-usage.md
git commit -m "polish: final review pass on technical debt article"
```

**Step 4: Push**

```bash
git push
```
