---
layout: post
title: "Measure Theory: The Rigorous Foundation Under Probability and Why It Matters"
date: 2026-01-07
category: math
---

Probability theory, at first glance, seems like it should be simple. You have events, you assign numbers between 0 and 1 to them, and those numbers obey certain rules. But the moment you try to make this rigorous --- the moment you ask "which events can we assign probabilities to?" and "what exactly is an expectation?" --- you run headfirst into measure theory.

This is not an accident. Measure theory is not some unnecessary abstraction bolted onto probability. It is the **only** known way to build a consistent, general theory of probability. Without it, you cannot rigorously define continuous random variables, prove convergence theorems that underpin machine learning, or even state what a "density function" really is. Every time you write $\mathbb{E}[f(X)]$ or invoke the law of large numbers, you are implicitly relying on measure theory.

This post builds measure theory from scratch, motivated at every step by the question: why do we need this?

---

## Table of Contents

1. [The Problem: Not All Sets Can Be Measured](#the-problem-not-all-sets-can-be-measured)
2. [Sigma-Algebras: The Sets We Can Measure](#sigma-algebras-the-sets-we-can-measure)
3. [Measures: Assigning Sizes to Sets](#measures-assigning-sizes-to-sets)
4. [Measurable Functions](#measurable-functions)
5. [The Lebesgue Integral: A Better Way to Integrate](#the-lebesgue-integral-a-better-way-to-integrate)
6. [Probability Spaces as Measure Spaces](#probability-spaces-as-measure-spaces)
7. [Random Variables as Measurable Functions](#random-variables-as-measurable-functions)
8. [Expectation as Lebesgue Integration](#expectation-as-lebesgue-integration)
9. [Convergence Theorems That Power ML](#convergence-theorems-that-power-ml)
10. [Radon-Nikodym: Densities as Derivatives of Measures](#radon-nikodym-densities-as-derivatives-of-measures)
11. [Product Measures and Fubini's Theorem](#product-measures-and-fubinis-theorem)
12. [Conclusion](#conclusion)

---

## The Problem: Not All Sets Can Be Measured

Here is a question that seems like it should have an obvious answer: can we assign a "length" to every subset of the real line?

Length, here, means a function $\lambda$ that takes a subset $A \subseteq \mathbb{R}$ and returns a number $\lambda(A) \geq 0$ satisfying three intuitive properties:

1. **Intervals have the right length:** $\lambda([a, b]) = b - a$.
2. **Translation invariance:** shifting a set does not change its length: $\lambda(A + x) = \lambda(A)$ for all $x \in \mathbb{R}$.
3. **Countable additivity:** if $A_1, A_2, A_3, \ldots$ are pairwise disjoint sets, then $\lambda\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} \lambda(A_i)$.

These seem completely reasonable. Surely every subset of $\mathbb{R}$ can be measured with these rules?

No. The answer is no, and the proof is one of the most important results in all of analysis.

### The Vitali Set

The construction, due to Giuseppe Vitali (1905), goes like this. Define an equivalence relation on $[0, 1]$: two numbers $x$ and $y$ are equivalent if $x - y$ is rational. That is, $x \sim y$ if and only if $x - y \in \mathbb{Q}$.

This partitions $[0, 1]$ into equivalence classes. For example, all the rationals in $[0, 1]$ form one class. The number $\sqrt{2}/2$ belongs to a different class, along with $\sqrt{2}/2 + 1/3$, $\sqrt{2}/2 - 1/7$, and so on.

Now, using the **Axiom of Choice**, pick exactly one representative from each equivalence class. Call this collection $V$. This is the Vitali set.

Now consider the translates $V_q = \{v + q \pmod{1} : v \in V\}$ for each rational $q \in \mathbb{Q} \cap [0, 1)$. Since every real number in $[0, 1]$ belongs to exactly one equivalence class, and $V$ contains exactly one representative from each class, the translates $\{V_q\}$ form a partition of $[0, 1]$:

$$[0, 1] = \bigcup_{q \in \mathbb{Q} \cap [0,1)} V_q$$

These are countably many disjoint sets (since $\mathbb{Q}$ is countable). By countable additivity:

$$\lambda([0, 1]) = \sum_{q \in \mathbb{Q} \cap [0,1)} \lambda(V_q)$$

By translation invariance, all $\lambda(V_q)$ are equal. Call this value $c$. Then:

$$1 = \sum_{q \in \mathbb{Q} \cap [0,1)} c$$

If $c = 0$, the right side is $0$. If $c > 0$, the right side is $\infty$. Either way, we get a contradiction.

**The conclusion is inescapable: the Vitali set cannot be assigned a length.** No value of $\lambda(V)$ is consistent with the three properties above. The set is **non-measurable**.

This is not a pathological curiosity. It is telling us something fundamental: if we want a consistent notion of "size" (or probability), we **cannot** apply it to all subsets of $\mathbb{R}$. We need to be careful about which sets we are allowed to measure.

This is why we need sigma-algebras.

---

## Sigma-Algebras: The Sets We Can Measure

A **sigma-algebra** (also written $\sigma$-algebra) on a set $\Omega$ is a collection $\mathcal{F}$ of subsets of $\Omega$ that satisfies three properties:

1. **Contains the whole space:** $\Omega \in \mathcal{F}$.
2. **Closed under complements:** If $A \in \mathcal{F}$, then $A^c = \Omega \setminus A \in \mathcal{F}$.
3. **Closed under countable unions:** If $A_1, A_2, A_3, \ldots \in \mathcal{F}$, then $\bigcup_{i=1}^{\infty} A_i \in \mathcal{F}$.

The pair $(\Omega, \mathcal{F})$ is called a **measurable space**. The sets in $\mathcal{F}$ are called **measurable sets** --- these are the sets we are allowed to measure.

Note what follows from the axioms:
- $\emptyset \in \mathcal{F}$ (take the complement of $\Omega$).
- $\mathcal{F}$ is closed under countable intersections (by De Morgan's law: $\bigcap A_i = (\bigcup A_i^c)^c$).
- $\mathcal{F}$ is closed under set differences (since $A \setminus B = A \cap B^c$).

The sigma-algebra is the bookkeeping device that tells us which sets we are allowed to ask "how big is this?" The whole point is to **exclude** pathological sets like the Vitali set while keeping all the sets we actually care about.

### Examples

**The trivial sigma-algebra:** $\mathcal{F} = \{\emptyset, \Omega\}$. This is the smallest sigma-algebra on $\Omega$. You can only measure "everything" or "nothing." Useless for most purposes, but technically valid.

**The power set:** $\mathcal{F} = 2^\Omega$ (all subsets of $\Omega$). This is the largest sigma-algebra. When $\Omega$ is finite or countable, you can get away with this. When $\Omega = \mathbb{R}$, the Vitali argument shows you cannot.

**The Borel sigma-algebra on $\mathbb{R}$:** This is the most important sigma-algebra in practice. It is defined as the smallest sigma-algebra containing all open intervals $(a, b)$. We denote it $\mathcal{B}(\mathbb{R})$.

The Borel sigma-algebra contains:
- All open sets (unions of open intervals).
- All closed sets (complements of open sets).
- All countable intersections of open sets ($G_\delta$ sets).
- All countable unions of closed sets ($F_\sigma$ sets).
- And so on, through a transfinite hierarchy.

It contains every set you will ever encounter in practice. It does **not** contain the Vitali set. Problem solved.

<svg viewBox="0 0 700 320" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-size="15" font-weight="bold" fill="#333">Sigma-Algebra Hierarchy on Ω = ℝ</text>
  <!-- Power set -->
  <ellipse cx="350" cy="175" rx="320" ry="130" fill="#f0f0f0" stroke="#999" stroke-width="1.5"/>
  <text x="350" y="300" text-anchor="middle" font-size="13" fill="#666">Power set 2^ℝ (includes non-measurable sets like Vitali set)</text>
  <!-- Borel -->
  <ellipse cx="350" cy="175" rx="220" ry="95" fill="#ddeeff" stroke="#4488cc" stroke-width="2"/>
  <text x="350" y="108" text-anchor="middle" font-size="13" fill="#336699">Borel σ-algebra B(ℝ)</text>
  <!-- Open/Closed -->
  <ellipse cx="280" cy="185" rx="90" ry="50" fill="#bbddff" stroke="#4488cc" stroke-width="1.5"/>
  <text x="280" y="180" text-anchor="middle" font-size="12" fill="#336699">Open sets</text>
  <text x="280" y="196" text-anchor="middle" font-size="12" fill="#336699">Closed sets</text>
  <ellipse cx="430" cy="185" rx="80" ry="50" fill="#bbddff" stroke="#4488cc" stroke-width="1.5"/>
  <text x="430" y="180" text-anchor="middle" font-size="12" fill="#336699">G_δ, F_σ</text>
  <text x="430" y="196" text-anchor="middle" font-size="12" fill="#336699">sets</text>
  <!-- Intervals -->
  <ellipse cx="280" cy="195" rx="45" ry="22" fill="#99ccff" stroke="#336699" stroke-width="1.5"/>
  <text x="280" y="199" text-anchor="middle" font-size="11" fill="#003366">Intervals</text>
  <!-- Trivial -->
  <circle cx="280" cy="195" r="8" fill="#336699"/>
  <text x="293" y="218" text-anchor="start" font-size="10" fill="#333">{∅, ℝ}</text>
  <!-- Non-measurable region label -->
  <text x="625" y="155" text-anchor="middle" font-size="11" fill="#cc3333" font-style="italic">Vitali set</text>
  <text x="625" y="170" text-anchor="middle" font-size="11" fill="#cc3333" font-style="italic">lives here</text>
  <line x1="590" y1="160" x2="575" y2="165" stroke="#cc3333" stroke-width="1"/>
</svg>

---

## Measures: Assigning Sizes to Sets

Now that we know which sets we are allowed to measure (those in a sigma-algebra $\mathcal{F}$), we can define a **measure**: the function that assigns sizes.

A **measure** on a measurable space $(\Omega, \mathcal{F})$ is a function $\mu: \mathcal{F} \to [0, \infty]$ satisfying:

1. **Non-negativity:** $\mu(A) \geq 0$ for all $A \in \mathcal{F}$.
2. **Null empty set:** $\mu(\emptyset) = 0$.
3. **Countable additivity ($\sigma$-additivity):** If $A_1, A_2, \ldots$ are pairwise disjoint sets in $\mathcal{F}$, then:

$$\mu\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} \mu(A_i)$$

The triple $(\Omega, \mathcal{F}, \mu)$ is called a **measure space**.

### Key Examples

**Counting measure:** $\Omega = \mathbb{N}$ (or any countable set), $\mathcal{F} = 2^\Omega$, and $\mu(A) = |A|$ (the number of elements in $A$). This is the simplest non-trivial measure and it is exactly what it sounds like: you count elements.

**Lebesgue measure:** $\Omega = \mathbb{R}$, $\mathcal{F} = \mathcal{B}(\mathbb{R})$ (Borel sigma-algebra, extended slightly to the Lebesgue sigma-algebra), and $\lambda$ assigns to each set its "length" in the intuitive sense. For intervals: $\lambda([a, b]) = b - a$. For more complicated sets, the definition involves careful limiting constructions with covers of intervals.

The Lebesgue measure is the rigorous version of "length" that avoids the Vitali paradox --- by restricting to measurable sets, we dodge the contradiction entirely.

**Dirac (point mass) measure:** $\delta_x(A) = 1$ if $x \in A$, $0$ otherwise. All the measure is concentrated at a single point. This is what lets us treat discrete and continuous distributions in one framework.

### Important Properties

From the axioms, one can prove:

- **Monotonicity:** If $A \subseteq B$, then $\mu(A) \leq \mu(B)$.
- **Continuity from below:** If $A_1 \subseteq A_2 \subseteq \cdots$, then $\mu\left(\bigcup A_n\right) = \lim_{n \to \infty} \mu(A_n)$.
- **Continuity from above:** If $A_1 \supseteq A_2 \supseteq \cdots$ and $\mu(A_1) < \infty$, then $\mu\left(\bigcap A_n\right) = \lim_{n \to \infty} \mu(A_n)$.
- **Subadditivity:** $\mu\left(\bigcup A_n\right) \leq \sum \mu(A_n)$ (for not necessarily disjoint sets).

---

## Measurable Functions

We need one more piece before we can integrate: **measurable functions**. These are functions that are "compatible" with the sigma-algebras on the domain and codomain.

Let $(\Omega, \mathcal{F})$ and $(S, \mathcal{S})$ be measurable spaces. A function $f: \Omega \to S$ is **$(\mathcal{F}, \mathcal{S})$-measurable** if for every $B \in \mathcal{S}$:

$$f^{-1}(B) = \{\omega \in \Omega : f(\omega) \in B\} \in \mathcal{F}$$

In words: the preimage of every measurable set in the codomain is a measurable set in the domain.

When $S = \mathbb{R}$ with the Borel sigma-algebra, we call $f$ a **Borel measurable function**. A convenient equivalent condition: $f: \Omega \to \mathbb{R}$ is measurable if and only if for every $a \in \mathbb{R}$:

$$\{\omega \in \Omega : f(\omega) \leq a\} \in \mathcal{F}$$

This single condition (for all $a$) is enough because the sets $(-\infty, a]$ generate the Borel sigma-algebra.

Why do we care about measurability? Because we want to integrate $f$ with respect to a measure, and integration requires us to compute things like $\mu(\{f > a\})$. If the set $\{f > a\}$ is not in $\mathcal{F}$, we cannot compute its measure, and the integral is undefined. Measurability is the condition that makes integration possible.

The good news: almost every function you will ever encounter is measurable. Continuous functions, piecewise continuous functions, limits of measurable functions, sums, products, compositions --- all measurable. You essentially have to use the Axiom of Choice to construct a non-measurable function.

---

## The Lebesgue Integral: A Better Way to Integrate

The Riemann integral, the one from calculus, works by slicing the **domain** (the $x$-axis) into small intervals and summing the function values times interval widths:

$$\int_a^b f(x) \, dx \approx \sum_{i} f(x_i^*) \cdot \Delta x_i$$

This works beautifully for continuous functions and even for functions with finitely many discontinuities. But it fails for highly discontinuous functions, and more importantly, it does not interact well with limits.

The **Lebesgue integral** takes a fundamentally different approach: it slices the **range** (the $y$-axis) instead.

### The Idea

Consider a non-negative measurable function $f: \Omega \to [0, \infty)$. Instead of asking "what is the function value at each point?", we ask "how much of the domain does the function spend at each height?"

For each height $y$, we look at the set $\{x : f(x) > y\}$ --- the region where the function is above $y$ --- and measure its size. The integral is then:

$$\int f \, d\mu = \int_0^\infty \mu(\{x : f(x) > y\}) \, dy$$

This is the "layer cake" representation of the Lebesgue integral.

<svg viewBox="0 0 700 550" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 700px; display: block; margin: 2em auto;">
  <!-- Title -->
  <text x="350" y="25" text-anchor="middle" font-size="15" font-weight="bold" fill="#333">Riemann vs Lebesgue Integration</text>

  <!-- Left panel: Riemann -->
  <text x="175" y="55" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">Riemann: Slice the x-axis</text>
  <!-- Axes -->
  <line x1="40" y1="260" x2="320" y2="260" stroke="#333" stroke-width="1.5"/>
  <line x1="40" y1="260" x2="40" y2="70" stroke="#333" stroke-width="1.5"/>
  <text x="180" y="278" text-anchor="middle" font-size="12" fill="#333">x</text>
  <text x="28" y="165" text-anchor="middle" font-size="12" fill="#333">y</text>
  <!-- Riemann bars -->
  <rect x="60" y="220" width="40" height="40" fill="#4488cc" fill-opacity="0.4" stroke="#4488cc" stroke-width="1"/>
  <rect x="100" y="180" width="40" height="80" fill="#4488cc" fill-opacity="0.4" stroke="#4488cc" stroke-width="1"/>
  <rect x="140" y="130" width="40" height="130" fill="#4488cc" fill-opacity="0.4" stroke="#4488cc" stroke-width="1"/>
  <rect x="180" y="100" width="40" height="160" fill="#4488cc" fill-opacity="0.4" stroke="#4488cc" stroke-width="1"/>
  <rect x="220" y="150" width="40" height="110" fill="#4488cc" fill-opacity="0.4" stroke="#4488cc" stroke-width="1"/>
  <rect x="260" y="200" width="40" height="60" fill="#4488cc" fill-opacity="0.4" stroke="#4488cc" stroke-width="1"/>
  <!-- Curve -->
  <path d="M60,240 Q110,180 160,120 Q200,90 230,140 Q270,190 300,220" fill="none" stroke="#cc3333" stroke-width="2.5"/>
  <!-- Vertical slices annotation -->
  <line x1="100" y1="260" x2="100" y2="265" stroke="#333" stroke-width="1"/>
  <line x1="140" y1="260" x2="140" y2="265" stroke="#333" stroke-width="1"/>
  <line x1="180" y1="260" x2="180" y2="265" stroke="#333" stroke-width="1"/>
  <line x1="220" y1="260" x2="220" y2="265" stroke="#333" stroke-width="1"/>
  <line x1="260" y1="260" x2="260" y2="265" stroke="#333" stroke-width="1"/>
  <text x="175" y="295" text-anchor="middle" font-size="11" fill="#4488cc">Σ f(xᵢ*) · Δxᵢ</text>

  <!-- Right panel: Lebesgue -->
  <text x="525" y="55" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">Lebesgue: Slice the y-axis</text>
  <!-- Axes -->
  <line x1="390" y1="260" x2="670" y2="260" stroke="#333" stroke-width="1.5"/>
  <line x1="390" y1="260" x2="390" y2="70" stroke="#333" stroke-width="1.5"/>
  <text x="530" y="278" text-anchor="middle" font-size="12" fill="#333">x</text>
  <text x="378" y="165" text-anchor="middle" font-size="12" fill="#333">y</text>
  <!-- Horizontal slices (colored bands) -->
  <rect x="408" y="220" width="242" height="20" fill="#cc7733" fill-opacity="0.15" stroke="none"/>
  <rect x="418" y="200" width="212" height="20" fill="#cc7733" fill-opacity="0.2" stroke="none"/>
  <rect x="435" y="180" width="175" height="20" fill="#cc7733" fill-opacity="0.25" stroke="none"/>
  <rect x="455" y="160" width="130" height="20" fill="#cc7733" fill-opacity="0.3" stroke="none"/>
  <rect x="475" y="140" width="85" height="20" fill="#cc7733" fill-opacity="0.35" stroke="none"/>
  <rect x="490" y="120" width="55" height="20" fill="#cc7733" fill-opacity="0.4" stroke="none"/>
  <rect x="500" y="100" width="35" height="20" fill="#cc7733" fill-opacity="0.45" stroke="none"/>
  <!-- Curve -->
  <path d="M410,240 Q460,180 510,120 Q550,90 580,140 Q620,190 650,220" fill="none" stroke="#cc3333" stroke-width="2.5"/>
  <!-- Horizontal lines -->
  <line x1="387" y1="220" x2="660" y2="220" stroke="#cc7733" stroke-width="0.5" stroke-dasharray="3,3"/>
  <line x1="387" y1="200" x2="660" y2="200" stroke="#cc7733" stroke-width="0.5" stroke-dasharray="3,3"/>
  <line x1="387" y1="180" x2="660" y2="180" stroke="#cc7733" stroke-width="0.5" stroke-dasharray="3,3"/>
  <line x1="387" y1="160" x2="660" y2="160" stroke="#cc7733" stroke-width="0.5" stroke-dasharray="3,3"/>
  <line x1="387" y1="140" x2="660" y2="140" stroke="#cc7733" stroke-width="0.5" stroke-dasharray="3,3"/>
  <line x1="387" y1="120" x2="660" y2="120" stroke="#cc7733" stroke-width="0.5" stroke-dasharray="3,3"/>
  <text x="525" y="295" text-anchor="middle" font-size="11" fill="#cc7733">∫₀^∞ μ({f > y}) dy</text>

  <!-- Bottom explanation -->
  <text x="350" y="330" text-anchor="middle" font-size="12" fill="#333">Riemann: partition the domain, sum heights × widths</text>
  <text x="350" y="350" text-anchor="middle" font-size="12" fill="#333">Lebesgue: partition the range, sum heights × measure of level sets</text>

  <!-- Pathological function example -->
  <text x="350" y="385" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">Why it matters: the Dirichlet function 1_Q(x)</text>
  <!-- Left: Riemann fails -->
  <text x="175" y="410" text-anchor="middle" font-size="13" fill="#cc3333">Riemann: UNDEFINED</text>
  <line x1="40" y1="500" x2="320" y2="500" stroke="#333" stroke-width="1.5"/>
  <line x1="40" y1="500" x2="40" y2="420" stroke="#333" stroke-width="1.5"/>
  <text x="35" y="445" text-anchor="end" font-size="10" fill="#333">1</text>
  <text x="35" y="500" text-anchor="end" font-size="10" fill="#333">0</text>
  <!-- Dense dots at both 0 and 1 -->
  <g fill="#4488cc" fill-opacity="0.5">
    <circle cx="55" cy="440" r="1.5"/><circle cx="65" cy="440" r="1.5"/><circle cx="75" cy="440" r="1.5"/><circle cx="85" cy="440" r="1.5"/><circle cx="95" cy="440" r="1.5"/><circle cx="105" cy="440" r="1.5"/><circle cx="115" cy="440" r="1.5"/><circle cx="125" cy="440" r="1.5"/><circle cx="135" cy="440" r="1.5"/><circle cx="145" cy="440" r="1.5"/><circle cx="155" cy="440" r="1.5"/><circle cx="165" cy="440" r="1.5"/><circle cx="175" cy="440" r="1.5"/><circle cx="185" cy="440" r="1.5"/><circle cx="195" cy="440" r="1.5"/><circle cx="205" cy="440" r="1.5"/><circle cx="215" cy="440" r="1.5"/><circle cx="225" cy="440" r="1.5"/><circle cx="235" cy="440" r="1.5"/><circle cx="245" cy="440" r="1.5"/><circle cx="255" cy="440" r="1.5"/><circle cx="265" cy="440" r="1.5"/><circle cx="275" cy="440" r="1.5"/><circle cx="285" cy="440" r="1.5"/><circle cx="295" cy="440" r="1.5"/><circle cx="305" cy="440" r="1.5"/>
  </g>
  <g fill="#cc3333" fill-opacity="0.5">
    <circle cx="60" cy="495" r="1.5"/><circle cx="70" cy="495" r="1.5"/><circle cx="80" cy="495" r="1.5"/><circle cx="90" cy="495" r="1.5"/><circle cx="100" cy="495" r="1.5"/><circle cx="110" cy="495" r="1.5"/><circle cx="120" cy="495" r="1.5"/><circle cx="130" cy="495" r="1.5"/><circle cx="140" cy="495" r="1.5"/><circle cx="150" cy="495" r="1.5"/><circle cx="160" cy="495" r="1.5"/><circle cx="170" cy="495" r="1.5"/><circle cx="180" cy="495" r="1.5"/><circle cx="190" cy="495" r="1.5"/><circle cx="200" cy="495" r="1.5"/><circle cx="210" cy="495" r="1.5"/><circle cx="220" cy="495" r="1.5"/><circle cx="230" cy="495" r="1.5"/><circle cx="240" cy="495" r="1.5"/><circle cx="250" cy="495" r="1.5"/><circle cx="260" cy="495" r="1.5"/><circle cx="270" cy="495" r="1.5"/><circle cx="280" cy="495" r="1.5"/><circle cx="290" cy="495" r="1.5"/><circle cx="300" cy="495" r="1.5"/>
  </g>
  <text x="175" y="530" text-anchor="middle" font-size="10" fill="#333">Every interval has both rationals and irrationals — upper/lower sums never converge</text>

  <!-- Right: Lebesgue works -->
  <text x="525" y="410" text-anchor="middle" font-size="13" fill="#339933">Lebesgue: = 0</text>
  <line x1="390" y1="500" x2="670" y2="500" stroke="#333" stroke-width="1.5"/>
  <line x1="390" y1="500" x2="390" y2="420" stroke="#333" stroke-width="1.5"/>
  <text x="385" y="445" text-anchor="end" font-size="10" fill="#333">1</text>
  <text x="385" y="500" text-anchor="end" font-size="10" fill="#333">0</text>
  <!-- Level set at y=1: rationals, measure 0 -->
  <rect x="395" y="432" width="265" height="16" fill="#339933" fill-opacity="0.1" stroke="#339933" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="530" y="444" text-anchor="middle" font-size="10" fill="#339933">μ({f = 1}) = λ(Q ∩ [0,1]) = 0</text>
  <!-- Level set at y=0: irrationals, measure 1 -->
  <rect x="395" y="488" width="265" height="16" fill="#339933" fill-opacity="0.3" stroke="#339933" stroke-width="1"/>
  <text x="530" y="500" text-anchor="middle" font-size="10" fill="#339933">μ({f = 0}) = λ(Q^c ∩ [0,1]) = 1</text>
  <text x="525" y="530" text-anchor="middle" font-size="10" fill="#333">∫ 1_Q dλ = 1 · λ(Q) + 0 · λ(Q^c) = 1 · 0 + 0 · 1 = 0</text>
</svg>

### Formal Construction

The formal construction builds up in stages:

**Step 1: Simple functions.** A **simple function** is a measurable function that takes only finitely many values:

$$s = \sum_{i=1}^{n} a_i \cdot \mathbf{1}_{A_i}$$

where $a_i \geq 0$ and $A_i \in \mathcal{F}$. Its integral is defined as:

$$\int s \, d\mu = \sum_{i=1}^{n} a_i \cdot \mu(A_i)$$

**Step 2: Non-negative measurable functions.** For any non-negative measurable function $f \geq 0$, there exists a sequence of simple functions $0 \leq s_1 \leq s_2 \leq \cdots$ with $s_n \uparrow f$ pointwise. The integral is defined as:

$$\int f \, d\mu = \sup_n \int s_n \, d\mu = \lim_{n \to \infty} \int s_n \, d\mu$$

**Step 3: General measurable functions.** Write $f = f^+ - f^-$ where $f^+ = \max(f, 0)$ and $f^- = \max(-f, 0)$. Both are non-negative. Define:

$$\int f \, d\mu = \int f^+ \, d\mu - \int f^- \, d\mu$$

provided at least one of the two integrals is finite. If both are finite, $f$ is called **integrable** (or $L^1$).

### Python: Lebesgue vs Riemann on a Pathological Function

```python
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

def is_approximately_rational(x, max_denom=1000):
    """Check if x is close to a rational with small denominator."""
    try:
        frac = Fraction(x).limit_denominator(max_denom)
        return abs(float(frac) - x) < 1e-10
    except (ValueError, OverflowError):
        return False

# Thomae's function: f(p/q) = 1/q for rationals, f(x) = 0 for irrationals
# (More visualizable than the Dirichlet function)
def thomae(x, max_denom=200):
    """Thomae's function: 1/q at p/q in lowest terms, 0 at irrationals."""
    if x == 0 or x == 1:
        return 1.0
    frac = Fraction(x).limit_denominator(max_denom)
    if abs(float(frac) - x) < 1e-12:
        return 1.0 / frac.denominator
    return 0.0

# Riemann integration attempt: vary partition size
partition_sizes = [10, 50, 100, 500, 1000, 5000]
upper_sums = []
lower_sums = []

for n in partition_sizes:
    dx = 1.0 / n
    upper = 0.0
    lower = 0.0
    for i in range(n):
        a, b = i * dx, (i + 1) * dx
        # Sample many points in [a, b]
        samples = np.linspace(a, b, 200)
        vals = [thomae(s) for s in samples]
        upper += max(vals) * dx
        lower += min(vals) * dx
    upper_sums.append(upper)
    lower_sums.append(lower)

# Lebesgue integral: sum y * measure({f = y})
# For Thomae: {f = 1/q} has measure 0 for each q, so integral = 0
lebesgue_value = 0.0

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Thomae's function
ax1 = axes[0]
xs = np.array(sorted(set(
    list(np.linspace(0, 1, 2000)) +
    [p/q for q in range(1, 80) for p in range(0, q+1)]
)))
ys = np.array([thomae(x) for x in xs])
ax1.scatter(xs, ys, s=0.5, c='#4488cc', alpha=0.6)
ax1.set_title("Thomae's Function on [0, 1]", fontsize=13)
ax1.set_xlabel("x")
ax1.set_ylabel("f(x)")
ax1.set_ylim(-0.05, 1.05)

# Right: Riemann upper/lower sums vs partition size
ax2 = axes[1]
ax2.semilogx(partition_sizes, upper_sums, 'o-', color='#cc3333', label='Upper Riemann sum', linewidth=2)
ax2.semilogx(partition_sizes, lower_sums, 's-', color='#339933', label='Lower Riemann sum', linewidth=2)
ax2.axhline(y=lebesgue_value, color='#cc7733', linestyle='--', linewidth=2, label=f'Lebesgue integral = {lebesgue_value}')
ax2.set_title("Riemann Sums vs Lebesgue Integral", fontsize=13)
ax2.set_xlabel("Number of partitions")
ax2.set_ylabel("Integral estimate")
ax2.legend(fontsize=11)
ax2.set_ylim(-0.1, 1.0)

plt.tight_layout()
plt.savefig("lebesgue_vs_riemann.png", dpi=150, bbox_inches='tight')
plt.show()
```

Thomae's function is Riemann integrable (the upper and lower sums do converge, though slowly). But the Dirichlet function $\mathbf{1}_\mathbb{Q}$ is **not** Riemann integrable: the upper sum is always 1 and the lower sum is always 0, regardless of partition size. The Lebesgue integral handles it effortlessly: $\int \mathbf{1}_\mathbb{Q} \, d\lambda = 1 \cdot \lambda(\mathbb{Q}) + 0 \cdot \lambda(\mathbb{Q}^c) = 0$.

---

## Probability Spaces as Measure Spaces

A **probability space** is simply a measure space $(\Omega, \mathcal{F}, P)$ where the measure $P$ satisfies $P(\Omega) = 1$. That is it. The entire edifice of probability theory is a special case of measure theory with a normalized measure.

- $\Omega$ is the **sample space**: the set of all possible outcomes.
- $\mathcal{F}$ is the **event space**: a sigma-algebra of subsets of $\Omega$. Events are the sets we can assign probabilities to.
- $P$ is the **probability measure**: a measure with $P(\Omega) = 1$.

Every property of measures carries over:

| Measure theory | Probability theory |
|---|---|
| Measure space $(\Omega, \mathcal{F}, \mu)$ | Probability space $(\Omega, \mathcal{F}, P)$ |
| Measurable set $A \in \mathcal{F}$ | Event |
| $\mu(A)$ | $P(A)$ (probability) |
| $\mu(\Omega) = 1$ | Normalization axiom |
| Countable additivity | $P(A \cup B) = P(A) + P(B)$ for disjoint events |
| Measurable function | Random variable |
| Integral $\int f \, d\mu$ | Expectation $\mathbb{E}[X]$ |

This unification is powerful. It means every theorem from measure theory automatically applies to probability. We do not need separate proofs for the probabilistic versions.

### Example: A Coin Flip

$\Omega = \{H, T\}$, $\mathcal{F} = \{\emptyset, \{H\}, \{T\}, \{H, T\}\}$ (the power set), $P(\{H\}) = p$, $P(\{T\}) = 1 - p$. Simple, and the sigma-algebra is just the power set because $\Omega$ is finite.

### Example: A Uniform Random Variable on $[0, 1]$

$\Omega = [0, 1]$, $\mathcal{F} = \mathcal{B}([0,1])$ (Borel sigma-algebra restricted to $[0,1]$), $P = \lambda$ (Lebesgue measure). This is where the Vitali argument bites: we cannot use the power set as $\mathcal{F}$ because there exist non-measurable subsets. The Borel sigma-algebra is the right choice.

---

## Random Variables as Measurable Functions

A **random variable** $X$ on a probability space $(\Omega, \mathcal{F}, P)$ is a measurable function $X: \Omega \to \mathbb{R}$. That is, $X$ is $(\mathcal{F}, \mathcal{B}(\mathbb{R}))$-measurable:

$$\{X \leq a\} := \{\omega \in \Omega : X(\omega) \leq a\} \in \mathcal{F} \quad \text{for all } a \in \mathbb{R}$$

Why do we need measurability? Because we want to compute $P(X \leq a) = P(\{\omega : X(\omega) \leq a\})$. For this to be defined, the set $\{X \leq a\}$ must be in $\mathcal{F}$ --- it must be an event that $P$ can evaluate. Measurability guarantees this.

The **cumulative distribution function (CDF)** is then:

$$F_X(a) = P(X \leq a) = P(X^{-1}((-\infty, a]))$$

This is just the pushforward of $P$ through $X$. The **distribution** (or **law**) of $X$ is the probability measure $\mu_X$ on $(\mathbb{R}, \mathcal{B}(\mathbb{R}))$ defined by:

$$\mu_X(B) = P(X^{-1}(B)) = P(X \in B)$$

This is where the distinction between a random variable (a function) and its distribution (a measure on $\mathbb{R}$) becomes important and precise.

---

## Expectation as Lebesgue Integration

The **expectation** of a random variable $X$ is its Lebesgue integral with respect to $P$:

$$\mathbb{E}[X] = \int_\Omega X(\omega) \, dP(\omega)$$

This is the most general definition of expectation. It unifies discrete and continuous cases:

- **Discrete case:** If $X$ takes values $x_1, x_2, \ldots$ with probabilities $p_1, p_2, \ldots$, then $P$ is a sum of point masses and the integral reduces to $\mathbb{E}[X] = \sum_i x_i p_i$.

- **Continuous case:** If $X$ has a density $f$ (more on this below), then the integral reduces to $\mathbb{E}[X] = \int_{-\infty}^{\infty} x \, f(x) \, dx$.

- **Mixed case:** $X$ could have a distribution that is partly discrete and partly continuous. The Lebesgue integral handles this seamlessly; the Riemann integral cannot.

More generally, for any measurable function $g$:

$$\mathbb{E}[g(X)] = \int_\Omega g(X(\omega)) \, dP(\omega) = \int_\mathbb{R} g(x) \, d\mu_X(x)$$

The second equality is the **change of variables formula** (also called the law of the unconscious statistician in probability). It says we can compute the expectation either by integrating over the original sample space $\Omega$ or by integrating over $\mathbb{R}$ with respect to the distribution of $X$.

---

## Convergence Theorems That Power ML

The real payoff of measure-theoretic integration is the convergence theorems. These are the theorems that let you swap limits and integrals, which is essential for justifying virtually every computation in machine learning that involves taking gradients through expectations or training on infinite-dimensional function spaces.

### Monotone Convergence Theorem (MCT)

**Statement:** Let $f_1 \leq f_2 \leq f_3 \leq \cdots$ be a sequence of non-negative measurable functions with $f_n \uparrow f$ pointwise. Then:

$$\lim_{n \to \infty} \int f_n \, d\mu = \int \lim_{n \to \infty} f_n \, d\mu = \int f \, d\mu$$

In words: for an increasing sequence of non-negative functions, you can swap the limit and the integral.

**Why it matters:** This is the foundational theorem. Almost every other convergence result is proved using MCT. It tells you that if your approximations are getting bigger (or better, monotonically), the integral of the approximation converges to the integral of the limit.

### Fatou's Lemma

**Statement:** For non-negative measurable functions $f_n$:

$$\int \liminf_{n \to \infty} f_n \, d\mu \leq \liminf_{n \to \infty} \int f_n \, d\mu$$

The integral of the limit is at most the limit of the integrals. Equality may not hold --- some "mass" can escape to infinity.

### Dominated Convergence Theorem (DCT)

**Statement:** Let $f_n \to f$ pointwise, and suppose there exists an integrable function $g$ (called the **dominating function**) with $|f_n| \leq g$ for all $n$ and $\int g \, d\mu < \infty$. Then:

$$\lim_{n \to \infty} \int f_n \, d\mu = \int f \, d\mu$$

and moreover $\int |f_n - f| \, d\mu \to 0$.

**Why it matters for ML:** DCT is the workhorse. Every time you differentiate under an expectation --- which is what happens when you compute gradients of loss functions that involve expectations (policy gradients, variational inference, score matching) --- you need DCT to justify swapping the derivative and integral:

$$\frac{d}{d\theta} \mathbb{E}[f(X; \theta)] = \frac{d}{d\theta} \int f(x; \theta) \, d\mu(x) \stackrel{\text{DCT}}{=} \int \frac{\partial f}{\partial \theta}(x; \theta) \, d\mu(x) = \mathbb{E}\left[\frac{\partial f}{\partial \theta}(X; \theta)\right]$$

This interchange is valid when $|\partial f / \partial \theta|$ is bounded by an integrable function --- exactly the DCT hypothesis. Without this theorem, gradient-based optimization of expected losses would have no rigorous foundation.

### Python: Convergence Theorems in Action

```python
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- MCT demonstration ---
ax1 = axes[0]
x = np.linspace(0, 5, 1000)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, 8))

for i, n in enumerate([1, 2, 3, 5, 8, 12, 20, 50]):
    # f_n(x) = min(x^2, n) -- increasing sequence converging to x^2
    fn = np.minimum(x**2, n)
    integral_n = np.trapz(fn, x)
    ax1.plot(x, fn, color=colors[i], label=f'n={n}, ∫={integral_n:.1f}', linewidth=1.5)

ax1.plot(x, x**2, 'k--', linewidth=2, label='$f(x) = x^2$')
ax1.set_title('Monotone Convergence Theorem', fontsize=13)
ax1.set_xlabel('x')
ax1.set_ylabel('$f_n(x)$')
ax1.set_ylim(0, 30)
ax1.legend(fontsize=8, loc='upper left')

# --- Fatou's Lemma: mass can escape ---
ax2 = axes[1]
x = np.linspace(0, 20, 2000)

for i, n in enumerate([1, 2, 4, 8, 16]):
    # f_n = n * indicator of [n, n + 1/n] -- integral = 1 always
    fn = np.where((x >= n) & (x <= n + 1.0/n), n, 0)
    color = colors[i] if i < len(colors) else 'black'
    ax2.plot(x, fn, color=color, label=f'n={n}, ∫={np.trapz(fn, x):.2f}', linewidth=2)

ax2.set_title("Fatou's Lemma: Mass Escapes to ∞", fontsize=13)
ax2.set_xlabel('x')
ax2.set_ylabel('$f_n(x)$')
ax2.legend(fontsize=9)
ax2.annotate('$f_n \\to 0$ pointwise\nbut $\\int f_n = 1$ always!',
            xy=(10, 6), fontsize=11, color='#cc3333',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# --- DCT demonstration ---
ax3 = axes[2]
x = np.linspace(-5, 5, 1000)
g = np.exp(-x**2 / 2)  # dominating function

integrals = []
ns = list(range(1, 51))
for n in ns:
    # f_n(x) = sin(nx)/n * exp(-x^2/2) -- converges to 0, dominated by g
    fn = np.sin(n * x) / n * np.exp(-x**2 / 2)
    integrals.append(np.trapz(fn, x))

ax3.plot(ns, integrals, 'o-', color='#4488cc', markersize=3, linewidth=1.5, label='$\\int f_n \\, d\\mu$')
ax3.axhline(y=0, color='#cc3333', linestyle='--', linewidth=2, label='$\\int \\lim f_n = 0$')
ax3.set_title('DCT: $f_n = \\frac{\\sin(nx)}{n} e^{-x^2/2}$', fontsize=13)
ax3.set_xlabel('n')
ax3.set_ylabel('$\\int f_n$')
ax3.legend(fontsize=11)

plt.tight_layout()
plt.savefig("convergence_theorems.png", dpi=150, bbox_inches='tight')
plt.show()
```

---

## Radon-Nikodym: Densities as Derivatives of Measures

Here is a question that arises constantly: what is a probability density function, really?

We say a continuous random variable $X$ has density $f$ if:

$$P(X \in A) = \int_A f(x) \, dx$$

But what is $f$ in terms of the underlying measures? The answer is the **Radon-Nikodym theorem**, which is one of the deepest results in measure theory.

### Absolute Continuity

Let $\mu$ and $\nu$ be two measures on $(\Omega, \mathcal{F})$. We say $\nu$ is **absolutely continuous** with respect to $\mu$, written $\nu \ll \mu$, if:

$$\mu(A) = 0 \implies \nu(A) = 0 \quad \text{for all } A \in \mathcal{F}$$

In words: every set that is "invisible" to $\mu$ is also "invisible" to $\nu$. The measure $\nu$ does not put mass on sets that $\mu$ considers to have zero size.

### The Theorem

**Radon-Nikodym Theorem:** If $\nu \ll \mu$ and both are $\sigma$-finite measures, then there exists a non-negative measurable function $f$ such that:

$$\nu(A) = \int_A f \, d\mu \quad \text{for all } A \in \mathcal{F}$$

This function $f$ is called the **Radon-Nikodym derivative** of $\nu$ with respect to $\mu$, written:

$$f = \frac{d\nu}{d\mu}$$

The notation is deliberate: it is a derivative of one measure with respect to another, analogous to how $dy/dx$ is a derivative of one function with respect to another.

### Why This Matters

**Density functions are Radon-Nikodym derivatives.** If $X$ is a continuous random variable with distribution $\mu_X$ and the Lebesgue measure is $\lambda$, and if $\mu_X \ll \lambda$ (the distribution does not concentrate mass at single points), then the density function $f$ is:

$$f = \frac{d\mu_X}{d\lambda}$$

The PDF is literally the derivative of the probability measure with respect to Lebesgue measure. This is why density values can be greater than 1 --- they are not probabilities, they are rates of change.

**KL divergence uses the Radon-Nikodym derivative.** The Kullback-Leibler divergence between two distributions $P$ and $Q$ with $P \ll Q$ is:

$$D_{\text{KL}}(P \| Q) = \int \log \frac{dP}{dQ} \, dP = \mathbb{E}_P\left[\log \frac{dP}{dQ}\right]$$

When both have densities $p$ and $q$ with respect to Lebesgue measure, the Radon-Nikodym derivative $dP/dQ = p/q$, and this reduces to the familiar:

$$D_{\text{KL}}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx$$

But the measure-theoretic version is more general: it works even when $P$ and $Q$ do not have densities (e.g., for discrete distributions, or mixtures of discrete and continuous).

**Importance sampling.** In ML, importance sampling reweights samples from one distribution to estimate expectations under another:

$$\mathbb{E}_P[f(X)] = \mathbb{E}_Q\left[f(X) \cdot \frac{dP}{dQ}(X)\right]$$

The importance weight $dP/dQ$ is the Radon-Nikodym derivative. This is the rigorous basis for techniques like off-policy reinforcement learning and variational inference.

---

## Product Measures and Fubini's Theorem

In machine learning, we constantly deal with joint distributions of multiple random variables: $(X, Y)$, datasets $\{(x_i, y_i)\}_{i=1}^n$, latent variables and observations $(z, x)$. Product measures are the measure-theoretic foundation for joint distributions.

### Product Sigma-Algebras

Given measurable spaces $(\Omega_1, \mathcal{F}_1)$ and $(\Omega_2, \mathcal{F}_2)$, the **product sigma-algebra** $\mathcal{F}_1 \otimes \mathcal{F}_2$ on $\Omega_1 \times \Omega_2$ is the sigma-algebra generated by all "rectangles" $A \times B$ where $A \in \mathcal{F}_1$ and $B \in \mathcal{F}_2$.

### Product Measures

Given measures $\mu_1$ on $(\Omega_1, \mathcal{F}_1)$ and $\mu_2$ on $(\Omega_2, \mathcal{F}_2)$, the **product measure** $\mu_1 \otimes \mu_2$ is the unique measure on $(\Omega_1 \times \Omega_2, \mathcal{F}_1 \otimes \mathcal{F}_2)$ satisfying:

$$(\mu_1 \otimes \mu_2)(A \times B) = \mu_1(A) \cdot \mu_2(B)$$

For probability, this is exactly the joint distribution of **independent** random variables: $P(X \in A, Y \in B) = P(X \in A) \cdot P(Y \in B)$ when $X$ and $Y$ are independent.

### Fubini's Theorem

**Statement:** Let $f$ be a measurable function on $(\Omega_1 \times \Omega_2, \mathcal{F}_1 \otimes \mathcal{F}_2, \mu_1 \otimes \mu_2)$. If $f \geq 0$ or $\int |f| \, d(\mu_1 \otimes \mu_2) < \infty$, then:

$$\int_{\Omega_1 \times \Omega_2} f \, d(\mu_1 \otimes \mu_2) = \int_{\Omega_1} \left(\int_{\Omega_2} f(\omega_1, \omega_2) \, d\mu_2(\omega_2)\right) d\mu_1(\omega_1)$$

$$= \int_{\Omega_2} \left(\int_{\Omega_1} f(\omega_1, \omega_2) \, d\mu_1(\omega_1)\right) d\mu_2(\omega_2)$$

In words: **you can compute a double integral by doing iterated single integrals in either order.** The order of integration does not matter, as long as the integrability condition is satisfied.

**Why this matters for ML:**

- **Computing marginal distributions:** $p(x) = \int p(x, z) \, dz$ is justified by Fubini.
- **The ELBO derivation** in variational inference involves swapping integration order.
- **Expectations of functions of multiple random variables** reduce to iterated expectations: $\mathbb{E}[g(X, Y)] = \mathbb{E}_X[\mathbb{E}_Y[g(X, Y) \mid X]]$ when $X, Y$ are independent.
- **Monte Carlo estimation:** the law of large numbers, which justifies estimating $\mathbb{E}[f(X)]$ with $\frac{1}{n}\sum f(X_i)$, ultimately relies on Fubini applied to the product measure of $n$ independent copies.

---

## Conclusion

Measure theory is not an abstraction imposed on probability from the outside. It is the natural language that resolves the fundamental inconsistencies that arise when you try to assign probabilities to continuous outcomes.

The chain of reasoning is clean:

1. Not all subsets of $\mathbb{R}$ can be consistently measured (Vitali set).
2. Therefore, we restrict to a sigma-algebra of measurable sets.
3. A measure assigns sizes to these sets; a probability measure is one normalized to 1.
4. Random variables are measurable functions; expectation is the Lebesgue integral.
5. The Lebesgue integral, unlike the Riemann integral, handles pathological functions and interacts cleanly with limits (MCT, DCT).
6. Density functions are Radon-Nikodym derivatives --- rates of change of one measure with respect to another.
7. Joint distributions arise from product measures; Fubini's theorem lets us do iterated integration.

Every time you write a loss function involving $\mathbb{E}$, differentiate through an expectation, compute a KL divergence, or invoke the law of large numbers, you are standing on this foundation. Understanding it does not change your code. But it tells you exactly when your code is mathematically justified --- and when it is not.
