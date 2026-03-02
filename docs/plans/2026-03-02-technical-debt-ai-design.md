# Design: Technical Debt and AI Usage

**Date:** 2026-03-02
**Category:** infra
**Type:** Standalone article (not part of vibe code series)
**Math level:** Moderate (velocity decay model, break-even analysis, plots)
**Stance:** Pragmatic realist

---

## Thesis

AI tools don't create a new kind of technical debt. They accelerate the accumulation of every existing kind, while simultaneously making the debt harder to detect because the code "looks right."

## Article Structure

### Section 1: Opening & Setup
- **Hook:** Shipped a feature in 2 hours with Cursor, a month later it's 40% of production incidents. Code "works" but nobody understands why.
- **Setup:** Technical debt is the most abused metaphor in software. We define it rigorously, build a quantitative model, then examine how AI tools changed the equation.
- **Thesis statement** as above.

### Section 2: Defining Technical Debt Rigorously
- Map the financial metaphor precisely: principal, interest, interest rate, compounding.
- Quantitative model:
  - Velocity decay: V(t) = V_0 * exp(-alpha * D(t))
  - Debt dynamics: dD/dt = beta * R(t) - gamma * P(t)
  - Not predictive — formalizes the intuition that debt compounds exponentially.
- Martin Fowler's debt quadrant (deliberate/inadvertent x prudent/reckless) as a table.
- Set up the quadrant because AI tools interact differently with each cell.

### Section 3: How AI Tools Change the Debt Equation
Three parts:

**Part A — The throughput multiplier:**
AI increases R(t) by 3-10x. Debt creation is proportional to code output. Team's review/refactor capacity is unchanged. The equilibrium shifts.

**Part B — Six specific AI debt patterns:**
1. Plausible but wrong abstractions
2. Cargo-cult error handling
3. Dependency hoarding
4. Implicit coupling through copy-paste evolution
5. Missing observability
6. Confident nonsense in edge cases

Each with concrete explanation and code-smell examples.

**Part C — The detection problem:**
AI-generated debt reads well. Passes review because reviewers pattern-match on surface signals (naming, structure, formatting). The debt is semantic, not syntactic. Diagram: AI code scores high on surface metrics, low on deeper ones (domain alignment, operational fitness, edge-case correctness).

### Section 4: Quantifying the Cost — When Debt Kills You
- **Break-even analysis:** Time saved S vs cumulative interest I(t). Crossover point depends on coupling. Isolated code = AI is pure upside. Foundational code = crossover in weeks.
- **Velocity decay plot:** Three scenarios (no AI steady management, AI + active review, AI + no management). Scenario 3 shows spike then cliff.
- **"300-line threshold":** Unreviewed AI code is a liability with unknown magnitude. When unreviewed > reviewed, you've lost the ability to reason about your system.

### Section 5: Managing AI-Generated Debt — Strategies
1. **The ownership rule** — if you prompted it, you own it (meaning you've read and can explain every line)
2. **Prompt-then-rewrite** — AI for first draft, you rewrite by hand
3. **Foundational code is off-limits** — coupling determines AI supervision level
4. **Review for semantics, not syntax** — stop checking formatting, start asking domain/failure/logging questions
5. **Scheduled debt audits** — periodic full read-throughs of AI-generated modules
6. **Track the ratio** — rough awareness of AI-generated vs human-written vs reviewed-AI code

### Section 6: Conclusion
Three beats:
1. Technical debt is precisely definable with compounding dynamics.
2. AI tools multiply creation rate while hiding the debt. Clean syntax, semantic debt.
3. Solution isn't to stop using AI — it's to understand the equation and manage deliberately.

Closing line: "The best engineers using AI tools aren't the ones who ship the fastest. They're the ones who know exactly how much debt they're taking on and why."

## Visuals

1. **SVG diagram:** Martin Fowler debt quadrant
2. **Matplotlib plot:** Velocity decay V(t) under three scenarios
3. **Matplotlib plot:** Break-even crossover for different coupling levels
4. **SVG diagram:** Surface metrics vs semantic metrics comparison for AI-generated code

## Style Notes
- Per CLAUDE.md: conversational but precise, no fluff, no buzzwords
- Define every term before using it
- MathJax inline with \\(...\\), display with $$...$$
- Escape * as \* inside inline math (kramdown compatibility)
- Open with concrete scenario, build intuition, then formalize
