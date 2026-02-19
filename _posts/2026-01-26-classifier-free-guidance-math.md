---
layout: post
title: "Classifier-Free Guidance: The Complete Mathematical Theory Behind Better Video Generation"
date: 2026-01-26
category: math
---

Every time you type a prompt into a video generation model and get back something that actually matches what you described, classifier-free guidance (CFG) is doing the heavy lifting. It is arguably the single most important inference-time technique in diffusion models. Without it, conditional generation would be blurry, generic, and barely responsive to your text prompts.

This post is the complete mathematical treatment. We start from Bayes' theorem, derive the score function formulation, arrive at the CFG equation, analyze the quality-diversity tradeoff, and then explore the practical implications for video generation specifically. If you build on top of diffusion models --- whether through APIs or self-hosted inference --- understanding CFG at this level will make you a better practitioner.

---

## Table of Contents

1. [The Problem: Conditional Generation is Hard](#the-problem-conditional-generation-is-hard)
2. [Classifier-Based Guidance: The First Attempt](#classifier-based-guidance-the-first-attempt)
3. [From Bayes' Theorem to Score Functions](#from-bayes-theorem-to-score-functions)
4. [Deriving the CFG Equation](#deriving-the-cfg-equation)
5. [The Guidance Scale: What It Actually Controls](#the-guidance-scale-what-it-actually-controls)
6. [The Quality-Diversity Tradeoff](#the-quality-diversity-tradeoff)
7. [Practical Guidance Scales by Model](#practical-guidance-scales-by-model)
8. [CFG in Video Generation](#cfg-in-video-generation)
9. [Dynamic Guidance Schedules](#dynamic-guidance-schedules)
10. [Negative Prompts as CFG](#negative-prompts-as-cfg)
11. [CFG Rescale and Advanced Variants](#cfg-rescale-and-advanced-variants)
12. [Implementation Notes](#implementation-notes)

---

## The Problem: Conditional Generation is Hard

The goal of conditional generation is to sample from the distribution \(p(x \mid c)\), where \(x\) is the data (an image or video) and \(c\) is the conditioning signal (a text prompt, reference image, class label, etc.).

Diffusion models learn to reverse a noise process. During training, a model \(\epsilon_\theta\) learns to predict the noise \(\epsilon\) that was added to a clean sample \(x_0\) at some timestep \(t\):

$$L = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2\right]$$

This works. You get a model that can generate images or video conditioned on \(c\). But the generations tend to be conservative. The model hedges its bets --- producing outputs that are vaguely related to the prompt but lack the sharp, specific adherence that users expect.

Why? Because the model is trained to minimize mean squared error across the entire distribution. It learns the average behavior conditional on \(c\), not the sharpest, most distinctive conditional behavior. The probability mass spreads across many plausible outputs rather than concentrating on the outputs that most strongly match the condition.

We need a way to amplify the influence of the conditioning signal at inference time. This is where guidance comes in.

---

## Classifier-Based Guidance: The First Attempt

The first solution, proposed by Dhariwal and Nichol (2021), was classifier guidance. The idea: train a separate classifier \(p_\phi(c \mid x_t)\) that can classify noisy images at every timestep \(t\). Then use the gradient of the classifier's log-probability to steer the diffusion process toward outputs that the classifier confidently assigns to class \(c\).

The modified score function becomes:

$$\nabla_{x_t} \log p(x_t \mid c) = \nabla_{x_t} \log p(x_t) + \gamma \nabla_{x_t} \log p_\phi(c \mid x_t)$$

where \(\gamma\) controls the guidance strength. The first term is the unconditional score (the direction the diffusion model would naturally go), and the second term pushes toward regions that the classifier identifies as belonging to class \(c\).

This works surprisingly well. But it has serious practical problems:

**Problem 1: You need a separate classifier.** Not just any classifier --- one trained on noisy data at every noise level. This is a separate model you have to train, maintain, and run at inference time. For text-conditioned generation, you'd need a classifier that maps noisy images to text descriptions, which is itself a hard problem.

**Problem 2: Noisy gradients.** The classifier operates on noisy inputs \(x_t\), especially at high noise levels early in the denoising process. Its gradients can be unreliable, leading to artifacts and instability.

**Problem 3: Adversarial behavior.** Gradient-based steering can find adversarial inputs --- images that the classifier confidently labels as class \(c\) but that don't actually look like class \(c\) to humans. You are optimizing the classifier's confidence, not actual semantic content.

**Problem 4: Scalability.** For text-to-image or text-to-video generation with open-ended text prompts, you cannot train a classifier over an unbounded label space.

Ho and Salimans (2022) proposed an elegant alternative that solves all of these problems: classifier-free guidance.

---

## From Bayes' Theorem to Score Functions

The derivation of CFG starts from first principles. Let us walk through it step by step.

### Step 1: Bayes' Theorem

We want to sample from \(p(x \mid c)\). By Bayes' theorem:

$$p(x \mid c) = \frac{p(c \mid x) \cdot p(x)}{p(c)}$$

Since \(p(c)\) does not depend on \(x\), we can write:

$$p(x \mid c) \propto p(c \mid x) \cdot p(x)$$

### Step 2: Take the Logarithm

Taking the log of both sides:

$$\log p(x \mid c) = \log p(c \mid x) + \log p(x) + \text{const}$$

The constant term \(-\log p(c)\) is independent of \(x\) and will vanish when we take gradients.

### Step 3: Score Functions

The score function of a distribution \(p(x)\) is defined as the gradient of its log-density with respect to \(x\):

$$s(x) = \nabla_x \log p(x)$$

In diffusion models, the score function at timestep \(t\) is:

$$s(x_t, t) = \nabla_{x_t} \log p_t(x_t)$$

This is the direction in which to move \(x_t\) to increase its log-probability under the data distribution at noise level \(t\). The denoising process follows these score functions iteratively.

Taking the gradient of our Bayes' equation with respect to \(x_t\):

$$\nabla_{x_t} \log p_t(x_t \mid c) = \nabla_{x_t} \log p_t(x_t) + \nabla_{x_t} \log p_t(c \mid x_t)$$

This is the fundamental identity. The conditional score is the unconditional score plus the gradient of the implicit classifier \(\log p_t(c \mid x_t)\).

### Step 4: The Connection to Noise Prediction

In score-based diffusion models, there is a direct relationship between the score function and the noise prediction network:

$$\nabla_{x_t} \log p_t(x_t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

where \(\bar{\alpha}_t\) is the cumulative noise schedule parameter. The noise prediction \(\epsilon_\theta\) and the score function point in opposite directions (the score points toward data, the noise prediction points toward noise), scaled by \(\frac{1}{\sqrt{1 - \bar{\alpha}_t}}\).

This means we can rewrite the conditional score as:

$$-\frac{\epsilon_\theta(x_t, t, c)}{\sqrt{1 - \bar{\alpha}_t}} = -\frac{\epsilon_\theta(x_t, t, \emptyset)}{\sqrt{1 - \bar{\alpha}_t}} + \nabla_{x_t} \log p_t(c \mid x_t)$$

where \(\epsilon_\theta(x_t, t, \emptyset)\) is the unconditional noise prediction (no conditioning) and \(\epsilon_\theta(x_t, t, c)\) is the conditional noise prediction.

Rearranging:

$$\nabla_{x_t} \log p_t(c \mid x_t) = -\frac{1}{\sqrt{1 - \bar{\alpha}_t}}\left[\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)\right]$$

This is a key insight. The difference between the conditional and unconditional noise predictions is proportional to the gradient of the implicit classifier. **We do not need a separate classifier.** The diffusion model itself implicitly contains one.

---

## Deriving the CFG Equation

Now we have all the pieces. The classifier guidance equation was:

$$\nabla_{x_t} \log p_t(x_t \mid c) = \nabla_{x_t} \log p_t(x_t) + \gamma \nabla_{x_t} \log p_t(c \mid x_t)$$

The parameter \(\gamma\) controls guidance strength. When \(\gamma = 1\), this is just standard Bayes' theorem. When \(\gamma > 1\), we amplify the classifier signal.

Substituting the noise-prediction equivalences and using guidance scale \(s\) (where \(s = \gamma\)):

$$-\frac{\hat{\epsilon}(x_t, t, c)}{\sqrt{1 - \bar{\alpha}_t}} = -\frac{\epsilon_\theta(x_t, t, \emptyset)}{\sqrt{1 - \bar{\alpha}_t}} + s \cdot \left(-\frac{1}{\sqrt{1 - \bar{\alpha}_t}}\right)\left[\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)\right]$$

The \(-\frac{1}{\sqrt{1 - \bar{\alpha}_t}}\) factor cancels on all terms:

$$\hat{\epsilon}(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + s \cdot \left[\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)\right]$$

This is the **classifier-free guidance equation**:

$$\boxed{\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot \left[\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset)\right]}$$

An equivalent rearrangement that is sometimes more intuitive:

$$\hat{\epsilon}_\theta(x_t, c) = (1 - s) \cdot \epsilon_\theta(x_t, \emptyset) + s \cdot \epsilon_\theta(x_t, c)$$

When \(s = 1\): \(\hat{\epsilon} = \epsilon_\theta(x_t, c)\). Standard conditional generation, no guidance.

When \(s = 0\): \(\hat{\epsilon} = \epsilon_\theta(x_t, \emptyset)\). Unconditional generation, condition is ignored.

When \(s > 1\): The conditional prediction is amplified beyond the trained distribution. The model moves further in the direction of the conditioning signal than the standard conditional model would.

When \(0 < s < 1\): The conditioning signal is dampened. Generations are more diverse but less prompt-adherent.

When \(s < 0\): Negative guidance --- the model actively avoids the condition. This is mathematically valid but rarely useful in practice.

### Training for CFG

For classifier-free guidance to work, the model must be able to produce both conditional and unconditional predictions. During training, the conditioning signal \(c\) is randomly dropped (replaced with a null token \(\emptyset\)) with some probability \(p_\text{uncond}\), typically \(10\text{--}20\%\):

$$c_\text{train} = \begin{cases} c & \text{with probability } 1 - p_\text{uncond} \\ \emptyset & \text{with probability } p_\text{uncond} \end{cases}$$

This means a single model learns both \(\epsilon_\theta(x_t, t, c)\) and \(\epsilon_\theta(x_t, t, \emptyset)\). No separate unconditional model is needed.

The unconditional dropout rate \(p_\text{uncond}\) is a hyperparameter. Too low and the unconditional predictions are poor (not enough training signal). Too high and the conditional predictions suffer (too much data wasted on unconditional training). The sweet spot is usually \(p_\text{uncond} = 0.1\) to $0.2$.

---

## The Guidance Scale: What It Actually Controls

Let us build geometric intuition for what the guidance scale does.

Consider the noise prediction as a vector in a high-dimensional space. The unconditional prediction \(\epsilon_\theta(x_t, \emptyset)\) points in a direction determined by the general data distribution. The conditional prediction \(\epsilon_\theta(x_t, c)\) points in a slightly different direction, biased by the conditioning.

The difference vector \(\Delta = \epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset)\) represents the "pure conditioning signal" --- the direction in which the conditioning pushes the prediction away from the unconditional baseline.

CFG takes this difference vector and scales it:

$$\hat{\epsilon} = \epsilon_\theta(x_t, \emptyset) + s \cdot \Delta$$

<svg viewBox="0 0 700 400" xmlns="http://www.w3.org/2000/svg" style="max-width:700px; margin: 2em auto; display: block;">
  <!-- Grid -->
  <defs>
    <pattern id="grid1" width="40" height="40" patternUnits="userSpaceOnUse">
      <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#333" stroke-width="0.5"/>
    </pattern>
  </defs>
  <rect width="700" height="400" fill="url(#grid1)"/>

  <!-- Origin -->
  <circle cx="200" cy="280" r="5" fill="#d4d4d4"/>
  <text x="188" y="305" font-family="Georgia, serif" font-size="14" fill="#d4d4d4">x_t</text>

  <!-- Unconditional vector -->
  <line x1="200" y1="280" x2="380" y2="200" stroke="#9e9e9e" stroke-width="2.5" marker-end="url(#arrowGray)"/>
  <text x="260" y="260" font-family="Georgia, serif" font-size="13" fill="#999">ε(x_t, ∅)</text>

  <!-- Conditional vector -->
  <line x1="200" y1="280" x2="420" y2="160" stroke="#4fc3f7" stroke-width="2.5" marker-end="url(#arrowBlue)"/>
  <text x="340" y="195" font-family="Georgia, serif" font-size="13" fill="#0288d1">ε(x_t, c)</text>

  <!-- Delta vector -->
  <line x1="380" y1="200" x2="420" y2="160" stroke="#8bc34a" stroke-width="2" stroke-dasharray="6,3" marker-end="url(#arrowGreen)"/>
  <text x="415" y="170" font-family="Georgia, serif" font-size="12" fill="#558b2f">Δ</text>

  <!-- Guided vector s=1 (same as conditional) -->

  <!-- Guided vector s=2 -->
  <line x1="200" y1="280" x2="460" y2="120" stroke="#ef5350" stroke-width="2.5" stroke-dasharray="8,4" marker-end="url(#arrowRed)"/>
  <text x="445" y="108" font-family="Georgia, serif" font-size="13" fill="#c62828" font-weight="bold">s = 2</text>

  <!-- Guided vector s=3 -->
  <line x1="200" y1="280" x2="500" y2="80" stroke="#ffa726" stroke-width="2.5" stroke-dasharray="8,4" marker-end="url(#arrowOrange)"/>
  <text x="490" y="68" font-family="Georgia, serif" font-size="13" fill="#e65100" font-weight="bold">s = 3</text>

  <!-- Guided vector s=7 -->
  <line x1="200" y1="280" x2="580" y2="30" stroke="#ab47bc" stroke-width="2" stroke-dasharray="4,4" marker-end="url(#arrowPurple)"/>
  <text x="568" y="22" font-family="Georgia, serif" font-size="13" fill="#7b1fa2" font-weight="bold">s = 7</text>

  <!-- Arrow markers -->
  <defs>
    <marker id="arrowGray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#9e9e9e"/>
    </marker>
    <marker id="arrowBlue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#4fc3f7"/>
    </marker>
    <marker id="arrowGreen" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#8bc34a"/>
    </marker>
    <marker id="arrowRed" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#ef5350"/>
    </marker>
    <marker id="arrowOrange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#ffa726"/>
    </marker>
    <marker id="arrowPurple" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#ab47bc"/>
    </marker>
  </defs>

  <!-- Legend -->
  <rect x="20" y="15" width="260" height="130" rx="6" fill="#1e1e1e" stroke="#444" stroke-width="1"/>
  <text x="35" y="38" font-family="Georgia, serif" font-size="13" fill="#d4d4d4" font-weight="bold">Vector Interpretation of CFG</text>
  <line x1="35" y1="55" x2="65" y2="55" stroke="#9e9e9e" stroke-width="2.5"/>
  <text x="72" y="59" font-family="Georgia, serif" font-size="12" fill="#999">Unconditional ε(x_t, ∅)</text>
  <line x1="35" y1="75" x2="65" y2="75" stroke="#4fc3f7" stroke-width="2.5"/>
  <text x="72" y="79" font-family="Georgia, serif" font-size="12" fill="#999">Conditional ε(x_t, c) [s=1]</text>
  <line x1="35" y1="95" x2="65" y2="95" stroke="#8bc34a" stroke-width="2" stroke-dasharray="6,3"/>
  <text x="72" y="99" font-family="Georgia, serif" font-size="12" fill="#999">Conditioning delta Δ</text>
  <line x1="35" y1="115" x2="65" y2="115" stroke="#ef5350" stroke-width="2" stroke-dasharray="8,4"/>
  <text x="72" y="119" font-family="Georgia, serif" font-size="12" fill="#999">Guided predictions (s > 1)</text>
  <line x1="35" y1="135" x2="65" y2="135" stroke="#ffa726" stroke-width="2" stroke-dasharray="8,4"/>
  <text x="72" y="139" font-family="Georgia, serif" font-size="12" fill="#999">Higher s = more amplification</text>
</svg>

At \(s = 1\), you get the standard conditional prediction. As \(s\) increases, you extrapolate further along the conditioning direction. The model generates outputs that are "more conditional than conditional" --- sharper, more specific, more saturated.

But there is a limit. Push \(s\) too high and you move into regions of the noise-prediction space where the model was never trained. The outputs become oversaturated, distorted, or exhibit repetitive artifacts. This is the fundamental tension of CFG.

### The Implicit Distribution

What distribution are we actually sampling from when we use CFG with scale \(s\)? We can work this out from the modified score:

$$\nabla_{x_t} \log \tilde{p}(x_t \mid c) = \nabla_{x_t} \log p(x_t) + s \cdot \nabla_{x_t} \log p(c \mid x_t)$$

This corresponds to sampling from:

$$\tilde{p}(x \mid c) \propto p(x) \cdot p(c \mid x)^s$$

When \(s = 1\), this is just \(p(x \mid c)\) by Bayes' theorem. When \(s > 1\), we are sampling from a distribution where the likelihood term \(p(c \mid x)\) is raised to a power greater than 1. This sharpens the distribution around the modes where \(p(c \mid x)\) is highest --- the samples most strongly associated with condition \(c\).

This is analogous to temperature scaling in language models. Lower temperature (higher \(s\) in CFG) concentrates sampling around the most likely outputs. The analogy is not exact --- CFG operates on score functions in continuous space rather than logit distributions over a discrete vocabulary --- but the intuition transfers.

---

## The Quality-Diversity Tradeoff

CFG creates a fundamental tradeoff between sample quality (measured by FID or human preference) and sample diversity (measured by recall, coverage, or intra-class variance).

### Mathematical Analysis

As guidance scale \(s\) increases:

1. **FID improves** (decreases) up to a point, then degrades. The optimal \(s\) for FID is typically in the range \([1.5, 8]\) depending on the model and task.

2. **Precision increases monotonically** with \(s\) (up to a saturation point). Each sample is more likely to be a high-quality member of the conditional distribution.

3. **Recall decreases monotonically** with \(s\). The model covers less of the true conditional distribution, concentrating on the modes.

4. **CLIP score increases** with \(s\), reflecting better prompt adherence, until artifacts cause CLIP to degrade.

<svg viewBox="0 0 700 420" xmlns="http://www.w3.org/2000/svg" style="max-width:700px; margin: 2em auto; display: block;">

  <text x="350" y="28" font-family="Georgia, serif" font-size="16" fill="#d4d4d4" text-anchor="middle" font-weight="bold">FID vs. Guidance Scale</text>

  <!-- Axes -->
  <line x1="80" y1="360" x2="660" y2="360" stroke="#d4d4d4" stroke-width="1.5"/>
  <line x1="80" y1="360" x2="80" y2="50" stroke="#d4d4d4" stroke-width="1.5"/>

  <!-- Gridlines -->
  <line x1="80" y1="295" x2="660" y2="295" stroke="#333" stroke-width="0.8"/>
  <line x1="80" y1="230" x2="660" y2="230" stroke="#333" stroke-width="0.8"/>
  <line x1="80" y1="165" x2="660" y2="165" stroke="#333" stroke-width="0.8"/>
  <line x1="80" y1="100" x2="660" y2="100" stroke="#333" stroke-width="0.8"/>

  <!-- Y-axis labels -->
  <text x="70" y="365" font-family="monospace" font-size="11" fill="#999" text-anchor="end">0</text>
  <text x="70" y="300" font-family="monospace" font-size="11" fill="#999" text-anchor="end">5</text>
  <text x="70" y="235" font-family="monospace" font-size="11" fill="#999" text-anchor="end">10</text>
  <text x="70" y="170" font-family="monospace" font-size="11" fill="#999" text-anchor="end">15</text>
  <text x="70" y="105" font-family="monospace" font-size="11" fill="#999" text-anchor="end">20</text>

  <!-- X-axis labels -->
  <text x="80" y="385" font-family="monospace" font-size="11" fill="#999" text-anchor="middle">0</text>
  <text x="174" y="385" font-family="monospace" font-size="11" fill="#999" text-anchor="middle">2</text>
  <text x="268" y="385" font-family="monospace" font-size="11" fill="#999" text-anchor="middle">4</text>
  <text x="362" y="385" font-family="monospace" font-size="11" fill="#999" text-anchor="middle">6</text>
  <text x="456" y="385" font-family="monospace" font-size="11" fill="#999" text-anchor="middle">8</text>
  <text x="550" y="385" font-family="monospace" font-size="11" fill="#999" text-anchor="middle">10</text>
  <text x="644" y="385" font-family="monospace" font-size="11" fill="#999" text-anchor="middle">12</text>

  <!-- Axis titles -->
  <text x="370" y="410" font-family="Georgia, serif" font-size="13" fill="#d4d4d4" text-anchor="middle">Guidance Scale (s)</text>
  <text x="25" y="210" font-family="Georgia, serif" font-size="13" fill="#d4d4d4" text-anchor="middle" transform="rotate(-90 25 210)">FID (lower is better)</text>

  <!-- FID curve - U-shaped: starts high, drops, then rises -->
  <!-- Points: s=0 FID=18, s=1 FID=12, s=2 FID=7, s=3 FID=4.5, s=4 FID=3.2, s=5 FID=3.0, s=6 FID=3.5, s=7 FID=4.5, s=8 FID=6, s=10 FID=10, s=12 FID=15 -->
  <polyline points="80,126 127,204 174,269 221,301 268,319 315,322 362,314 409,301 456,282 550,230 644,165"
    fill="none" stroke="#4fc3f7" stroke-width="3" stroke-linejoin="round"/>

  <!-- Optimal point marker -->
  <circle cx="315" cy="322" r="6" fill="#ef5350" stroke="white" stroke-width="2"/>
  <text x="325" y="345" font-family="Georgia, serif" font-size="12" fill="#ef5350" font-weight="bold">Optimal s ≈ 5</text>

  <!-- Annotation: underfitting region -->
  <text x="140" y="80" font-family="Georgia, serif" font-size="11" fill="#999" text-anchor="middle">Weak conditioning</text>
  <text x="140" y="94" font-family="Georgia, serif" font-size="11" fill="#999" text-anchor="middle">(blurry outputs)</text>
  <line x1="140" y1="100" x2="140" y2="195" stroke="#444" stroke-width="1" stroke-dasharray="4,3"/>

  <!-- Annotation: overfitting region -->
  <text x="560" y="80" font-family="Georgia, serif" font-size="11" fill="#999" text-anchor="middle">Over-saturation</text>
  <text x="560" y="94" font-family="Georgia, serif" font-size="11" fill="#999" text-anchor="middle">(artifacts)</text>
  <line x1="560" y1="100" x2="560" y2="225" stroke="#444" stroke-width="1" stroke-dasharray="4,3"/>

  <!-- Data points -->
  <circle cx="80" cy="126" r="3" fill="#4fc3f7"/>
  <circle cx="127" cy="204" r="3" fill="#4fc3f7"/>
  <circle cx="174" cy="269" r="3" fill="#4fc3f7"/>
  <circle cx="221" cy="301" r="3" fill="#4fc3f7"/>
  <circle cx="268" cy="319" r="3" fill="#4fc3f7"/>
  <circle cx="362" cy="314" r="3" fill="#4fc3f7"/>
  <circle cx="409" cy="301" r="3" fill="#4fc3f7"/>
  <circle cx="456" cy="282" r="3" fill="#4fc3f7"/>
  <circle cx="550" cy="230" r="3" fill="#4fc3f7"/>
  <circle cx="644" cy="165" r="3" fill="#4fc3f7"/>
</svg>

<svg viewBox="0 0 700 420" xmlns="http://www.w3.org/2000/svg" style="max-width:700px; margin: 2em auto; display: block;">

  <text x="350" y="28" font-family="Georgia, serif" font-size="16" fill="#d4d4d4" text-anchor="middle" font-weight="bold">Diversity (Recall) vs. Guidance Scale</text>

  <!-- Axes -->
  <line x1="80" y1="360" x2="660" y2="360" stroke="#d4d4d4" stroke-width="1.5"/>
  <line x1="80" y1="360" x2="80" y2="50" stroke="#d4d4d4" stroke-width="1.5"/>

  <!-- Gridlines -->
  <line x1="80" y1="298" x2="660" y2="298" stroke="#333" stroke-width="0.8"/>
  <line x1="80" y1="236" x2="660" y2="236" stroke="#333" stroke-width="0.8"/>
  <line x1="80" y1="174" x2="660" y2="174" stroke="#333" stroke-width="0.8"/>
  <line x1="80" y1="112" x2="660" y2="112" stroke="#333" stroke-width="0.8"/>

  <!-- Y-axis labels -->
  <text x="70" y="365" font-family="monospace" font-size="11" fill="#999" text-anchor="end">0.0</text>
  <text x="70" y="303" font-family="monospace" font-size="11" fill="#999" text-anchor="end">0.2</text>
  <text x="70" y="241" font-family="monospace" font-size="11" fill="#999" text-anchor="end">0.4</text>
  <text x="70" y="179" font-family="monospace" font-size="11" fill="#999" text-anchor="end">0.6</text>
  <text x="70" y="117" font-family="monospace" font-size="11" fill="#999" text-anchor="end">0.8</text>

  <!-- X-axis labels -->
  <text x="80" y="385" font-family="monospace" font-size="11" fill="#999" text-anchor="middle">0</text>
  <text x="174" y="385" font-family="monospace" font-size="11" fill="#999" text-anchor="middle">2</text>
  <text x="268" y="385" font-family="monospace" font-size="11" fill="#999" text-anchor="middle">4</text>
  <text x="362" y="385" font-family="monospace" font-size="11" fill="#999" text-anchor="middle">6</text>
  <text x="456" y="385" font-family="monospace" font-size="11" fill="#999" text-anchor="middle">8</text>
  <text x="550" y="385" font-family="monospace" font-size="11" fill="#999" text-anchor="middle">10</text>
  <text x="644" y="385" font-family="monospace" font-size="11" fill="#999" text-anchor="middle">12</text>

  <!-- Axis titles -->
  <text x="370" y="410" font-family="Georgia, serif" font-size="13" fill="#d4d4d4" text-anchor="middle">Guidance Scale (s)</text>
  <text x="25" y="210" font-family="Georgia, serif" font-size="13" fill="#d4d4d4" text-anchor="middle" transform="rotate(-90 25 210)">Recall (higher = more diverse)</text>

  <!-- Diversity curve: monotonically decreasing, starts at 0.85, drops steeply then flattens -->
  <!-- Points: s=0: 0.85, s=1: 0.78, s=2: 0.65, s=3: 0.52, s=4: 0.42, s=5: 0.34, s=6: 0.28, s=7: 0.24, s=8: 0.20, s=10: 0.15, s=12: 0.12 -->
  <polyline points="80,97 127,118 174,158 221,198 268,230 315,254 362,273 409,285 456,298 550,314 644,323"
    fill="none" stroke="#8bc34a" stroke-width="3" stroke-linejoin="round"/>

  <!-- Data points -->
  <circle cx="80" cy="97" r="3" fill="#8bc34a"/>
  <circle cx="127" cy="118" r="3" fill="#8bc34a"/>
  <circle cx="174" cy="158" r="3" fill="#8bc34a"/>
  <circle cx="221" cy="198" r="3" fill="#8bc34a"/>
  <circle cx="268" cy="230" r="3" fill="#8bc34a"/>
  <circle cx="315" cy="254" r="3" fill="#8bc34a"/>
  <circle cx="362" cy="273" r="3" fill="#8bc34a"/>
  <circle cx="409" cy="285" r="3" fill="#8bc34a"/>
  <circle cx="456" cy="298" r="3" fill="#8bc34a"/>
  <circle cx="550" cy="314" r="3" fill="#8bc34a"/>
  <circle cx="644" cy="323" r="3" fill="#8bc34a"/>

  <!-- Annotation -->
  <rect x="380" y="130" width="220" height="55" rx="4" fill="#1e1e1e" stroke="#444" stroke-width="1"/>
  <text x="490" y="150" font-family="Georgia, serif" font-size="12" fill="#999" text-anchor="middle">Higher guidance = less diversity</text>
  <text x="490" y="170" font-family="Georgia, serif" font-size="12" fill="#999" text-anchor="middle">Mode collapse at extreme scales</text>
</svg>

### The Precision-Recall Perspective

The precision-recall framework (Kynkaanniemi et al., 2019) gives us a clean way to quantify this tradeoff:

- **Precision**: Fraction of generated samples that fall within the support of the real data distribution. High precision means generated samples look realistic.
- **Recall**: Fraction of real data distribution covered by the generated samples. High recall means the model captures the full diversity of the data.

| Guidance Scale | Precision | Recall | FID | CLIP Score |
|:-:|:-:|:-:|:-:|:-:|
| 0.0 (unconditional) | 0.45 | 0.85 | 28.5 | 0.18 |
| 1.0 (standard) | 0.62 | 0.72 | 12.4 | 0.25 |
| 2.0 | 0.73 | 0.58 | 6.8 | 0.29 |
| 3.0 | 0.80 | 0.48 | 4.5 | 0.31 |
| 5.0 | 0.86 | 0.32 | 3.1 | 0.33 |
| 7.5 | 0.88 | 0.22 | 4.8 | 0.32 |
| 10.0 | 0.87 | 0.16 | 8.2 | 0.30 |
| 15.0 | 0.82 | 0.10 | 14.6 | 0.26 |

*Representative values for a class-conditional ImageNet model. Exact numbers vary by architecture.*

Notice the FID sweet spot around \(s = 5\). Below that, lack of conditioning hurts quality. Above that, over-conditioning introduces artifacts that FID penalizes. Meanwhile, recall drops monotonically --- every increase in guidance sacrifices some diversity.

---

## Practical Guidance Scales by Model

Different models use different default guidance scales. This is not arbitrary --- it reflects the model architecture, training procedure, conditioning mechanism, and target use case.

| Model | Type | Default \(s\) | Recommended Range | Notes |
|:--|:--|:-:|:--|:--|
| Stable Diffusion 1.5 | Image | 7.5 | 5 -- 12 | Classic default, well-studied |
| Stable Diffusion XL | Image | 7.0 | 5 -- 10 | Slightly lower than SD 1.5 |
| Flux.1 (Black Forest Labs) | Image | 3.5 | 2.5 -- 5.0 | Guidance-distilled, needs lower \(s\) |
| DALL-E 3 (via API) | Image | N/A | Fixed internally | Not user-configurable |
| Wan 2.2 T2V | Video | 5.0 | 3.5 -- 7.0 | MoE architecture, moderate guidance |
| Kling 3.0 | Video | ~4.5 | N/A (API) | Estimated from outputs |
| Sora 2 | Video | ~4.0 | N/A (API) | Estimated from outputs |
| LTX-2 | Video | 5.0 | 3.0 -- 7.0 | Open-source, configurable |
| AnimateDiff | Video | 7.5 | 5.0 -- 9.0 | Inherits from SD backbone |
| CogVideo | Video | 6.0 | 4.0 -- 8.0 | Moderate default |

### Why Video Models Use Lower Guidance

Video models typically use lower guidance scales than image models (\(s = 4\text{--}6\) vs \(s = 7\text{--}8\)). There are several reasons:

**1. Temporal coherence penalty.** Higher guidance amplifies frame-level prompt adherence but can create inconsistencies between frames. Each frame is pushed harder toward the text condition independently, potentially at the expense of smooth temporal transitions.

**2. Accumulation of artifacts.** In a 5-second video at 24 fps, you have 120 frames. Subtle artifacts from over-guidance in a single image become jarring flickering or pulsing in video.

**3. Architectural differences.** Video models often use temporal attention layers that create inter-frame dependencies. Strong CFG guidance on the spatial dimensions can fight against the temporal consistency learned by these layers.

**4. Guidance-aware training.** Newer video models (especially Flux and its derivatives) are trained with guidance-aware loss functions or distillation. They bake some of the guidance effect into the model weights, requiring less inference-time guidance. This is sometimes called "guidance distillation."

---

## CFG in Video Generation

The interaction between CFG and temporal consistency deserves deeper analysis, as this is where video diverges most significantly from image generation.

### The Temporal Coherence Problem

Consider a video diffusion model denoising a sequence of \(F\) frames jointly. At each denoising step, the model predicts noise for all frames simultaneously:

$$\hat{\epsilon}_\theta(x_t^{1:F}, c) = \epsilon_\theta(x_t^{1:F}, \emptyset) + s \cdot \left[\epsilon_\theta(x_t^{1:F}, c) - \epsilon_\theta(x_t^{1:F}, \emptyset)\right]$$

The model has learned temporal correlations through temporal attention layers. The unconditional prediction \(\epsilon_\theta(x_t^{1:F}, \emptyset)\) respects these correlations --- it produces temporally smooth noise predictions even without conditioning.

The conditioning difference \(\Delta = \epsilon_\theta(x_t^{1:F}, c) - \epsilon_\theta(x_t^{1:F}, \emptyset)\) also generally respects temporal coherence, but less perfectly. The text condition can push different frames in slightly different directions depending on which frame "matches" the prompt best at a given denoising step.

When \(s > 1\), this per-frame variation in \(\Delta\) is amplified. The result: each frame is pushed harder toward prompt adherence, but the temporal smoothness is degraded. Visually, this manifests as:

- **Flickering**: Rapid brightness or color changes between adjacent frames
- **Jitter**: Small spatial displacements of objects frame-to-frame
- **Temporal aliasing**: Motion that appears choppy rather than smooth
- **Semantic inconsistency**: An object changing shape or color mid-sequence

### The Guidance Sweet Spot for Video

The optimal guidance scale for video is lower than for images, and the penalty for going too high is steeper:

| Quality Metric | Image (optimal \(s\)) | Video (optimal \(s\)) |
|:--|:-:|:-:|
| FID / FVD | 5 -- 8 | 3.5 -- 6 |
| CLIP Score | 6 -- 10 | 4 -- 7 |
| Human preference | 5 -- 8 | 4 -- 6 |
| Temporal consistency | N/A | 2 -- 5 |

For video, you often want to prioritize temporal consistency over prompt adherence. A slightly less "perfect" match to the prompt that moves smoothly is more watchable than a frame-accurate but flickery video.

### Per-Frame vs. Sequence-Level Guidance

Some implementations apply CFG differently along temporal and spatial dimensions:

**Uniform guidance**: Same \(s\) for all frames. Simple but treats the first and last frames identically.

**Keyframe-weighted guidance**: Higher \(s\) on keyframes (first, middle, last), lower \(s\) on intermediate frames. The keyframes anchor the semantic content while intermediate frames prioritize smooth interpolation.

**Temporal-decay guidance**: Higher \(s\) on the first frame, decreasing toward the end. Ensures the opening frame strongly matches the prompt while allowing the sequence to evolve naturally.

This is an active area of research. Most production models still use uniform guidance, but the next generation of video models will likely incorporate temporal-aware guidance schedules.

---

## Dynamic Guidance Schedules

Rather than using a fixed guidance scale \(s\) throughout the entire denoising process, dynamic schedules vary \(s\) as a function of the timestep \(t\).

### The Intuition

Early denoising steps (high \(t\), high noise) determine the global structure --- composition, layout, major semantic elements. Late denoising steps (low \(t\), low noise) refine fine details --- textures, edges, small features.

High guidance is most valuable during early steps, where it steers the global structure toward the prompt. During late steps, the structure is already established, and high guidance mainly causes over-saturation and artifacts on fine details.

### The Linear Schedule

The simplest dynamic schedule linearly decreases guidance from a maximum \(s_\text{max}\) to a minimum \(s_\text{min}\) over the denoising steps:

$$s(t) = s_\text{min} + (s_\text{max} - s_\text{min}) \cdot \frac{t}{T}$$

where \(T\) is the total number of denoising steps, and \(t\) counts down from \(T\) (high noise) to $0$ (clean output).

### The Cosine Schedule

A smoother alternative uses a cosine decay:

$$s(t) = s_\text{min} + \frac{s_\text{max} - s_\text{min}}{2} \cdot \left(1 + \cos\left(\pi \cdot \frac{T - t}{T}\right)\right)$$

This spends more steps at high guidance during the early (high-noise) phase, then rapidly drops during the late (low-noise) phase.

### The Step-Function Schedule

Some practitioners use a simple two-phase approach:

$$s(t) = \begin{cases} s_\text{high} & \text{if } t > t_\text{switch} \\ s_\text{low} & \text{if } t \leq t_\text{switch} \end{cases}$$

where \(t_\text{switch}\) is typically around $0.5T$ to $0.7T$.

<svg viewBox="0 0 700 420" xmlns="http://www.w3.org/2000/svg" style="max-width:700px; margin: 2em auto; display: block;">

  <text x="350" y="28" font-family="Georgia, serif" font-size="16" fill="#d4d4d4" text-anchor="middle" font-weight="bold">Dynamic Guidance Schedules</text>

  <!-- Axes -->
  <line x1="80" y1="360" x2="660" y2="360" stroke="#d4d4d4" stroke-width="1.5"/>
  <line x1="80" y1="360" x2="80" y2="50" stroke="#d4d4d4" stroke-width="1.5"/>

  <!-- Gridlines -->
  <line x1="80" y1="310" x2="660" y2="310" stroke="#333" stroke-width="0.8"/>
  <line x1="80" y1="260" x2="660" y2="260" stroke="#333" stroke-width="0.8"/>
  <line x1="80" y1="210" x2="660" y2="210" stroke="#333" stroke-width="0.8"/>
  <line x1="80" y1="160" x2="660" y2="160" stroke="#333" stroke-width="0.8"/>
  <line x1="80" y1="110" x2="660" y2="110" stroke="#333" stroke-width="0.8"/>

  <!-- Y-axis labels (guidance scale) -->
  <text x="70" y="365" font-family="monospace" font-size="11" fill="#999" text-anchor="end">1</text>
  <text x="70" y="310" font-family="monospace" font-size="11" fill="#999" text-anchor="end">3</text>
  <text x="70" y="260" font-family="monospace" font-size="11" fill="#999" text-anchor="end">5</text>
  <text x="70" y="210" font-family="monospace" font-size="11" fill="#999" text-anchor="end">7</text>
  <text x="70" y="160" font-family="monospace" font-size="11" fill="#999" text-anchor="end">9</text>
  <text x="70" y="110" font-family="monospace" font-size="11" fill="#999" text-anchor="end">11</text>

  <!-- X-axis labels (denoising step, left=high noise, right=clean) -->
  <text x="80" y="385" font-family="monospace" font-size="11" fill="#999" text-anchor="middle">T</text>
  <text x="225" y="385" font-family="monospace" font-size="11" fill="#999" text-anchor="middle">0.75T</text>
  <text x="370" y="385" font-family="monospace" font-size="11" fill="#999" text-anchor="middle">0.5T</text>
  <text x="515" y="385" font-family="monospace" font-size="11" fill="#999" text-anchor="middle">0.25T</text>
  <text x="660" y="385" font-family="monospace" font-size="11" fill="#999" text-anchor="middle">0</text>

  <!-- Axis titles -->
  <text x="370" y="410" font-family="Georgia, serif" font-size="13" fill="#d4d4d4" text-anchor="middle">Denoising Progress (high noise --> clean)</text>
  <text x="25" y="210" font-family="Georgia, serif" font-size="13" fill="#d4d4d4" text-anchor="middle" transform="rotate(-90 25 210)">Guidance Scale s(t)</text>

  <!-- Constant schedule (baseline) -->
  <line x1="80" y1="210" x2="660" y2="210" stroke="#9e9e9e" stroke-width="2" stroke-dasharray="8,5"/>

  <!-- Linear schedule: from s=10 at t=T to s=2 at t=0 -->
  <line x1="80" y1="135" x2="660" y2="335" stroke="#4fc3f7" stroke-width="2.5"/>

  <!-- Cosine schedule: starts high, holds, then drops -->
  <path d="M 80,135 C 200,140 280,145 370,210 C 450,280 550,330 660,340" fill="none" stroke="#ef5350" stroke-width="2.5"/>

  <!-- Step function: high then low -->
  <line x1="80" y1="135" x2="370" y2="135" stroke="#8bc34a" stroke-width="2.5"/>
  <line x1="370" y1="135" x2="370" y2="310" stroke="#8bc34a" stroke-width="2.5" stroke-dasharray="3,2"/>
  <line x1="370" y1="310" x2="660" y2="310" stroke="#8bc34a" stroke-width="2.5"/>

  <!-- Legend -->
  <rect x="420" y="60" width="230" height="105" rx="6" fill="#1e1e1e" stroke="#444" stroke-width="1"/>
  <line x1="435" y1="82" x2="475" y2="82" stroke="#9e9e9e" stroke-width="2" stroke-dasharray="8,5"/>
  <text x="482" y="86" font-family="Georgia, serif" font-size="12" fill="#999">Constant (s = 7)</text>
  <line x1="435" y1="105" x2="475" y2="105" stroke="#4fc3f7" stroke-width="2.5"/>
  <text x="482" y="109" font-family="Georgia, serif" font-size="12" fill="#999">Linear decay</text>
  <line x1="435" y1="128" x2="475" y2="128" stroke="#ef5350" stroke-width="2.5"/>
  <text x="482" y="132" font-family="Georgia, serif" font-size="12" fill="#999">Cosine decay</text>
  <line x1="435" y1="151" x2="475" y2="151" stroke="#8bc34a" stroke-width="2.5"/>
  <text x="482" y="155" font-family="Georgia, serif" font-size="12" fill="#999">Step function</text>
</svg>

### Why Dynamic Schedules Work

The mathematical justification comes from analyzing the signal-to-noise ratio at each timestep.

At high noise levels (early steps), the signal-to-noise ratio is low. The model's conditional and unconditional predictions are both heavily influenced by the noise, and the conditioning difference \(\Delta\) is small relative to the overall prediction magnitude. High guidance is needed to make the conditioning signal "audible" above the noise.

At low noise levels (late steps), the signal-to-noise ratio is high. The conditioning is already strongly expressed in the partially-denoised sample. The difference \(\Delta\) is now operating on fine details, and amplifying it causes:

- **Color over-saturation**: Push toward extreme RGB values
- **Edge ringing**: Gibbs-like artifacts along sharp boundaries
- **Texture repetition**: Fine-grained patterns become excessively regular

Reducing guidance in late steps avoids these artifacts while retaining the structural benefits of high guidance in early steps.

### Practical Impact

In experiments across multiple image models, dynamic guidance schedules (particularly cosine decay from \(s_\text{max} = 10\) to \(s_\text{min} = 2\)) improve FID by 5-15% relative to the optimal constant guidance scale. The improvement is even more pronounced for video, where late-step over-guidance causes visible temporal artifacts.

```python
# Example: implementing dynamic cosine guidance
import math

def cosine_guidance_schedule(step, total_steps, s_max=10.0, s_min=2.0):
    """
    Returns guidance scale for a given denoising step.
    step: current step (0 = first/noisiest, total_steps-1 = last/cleanest)
    """
    progress = step / (total_steps - 1)  # 0 to 1
    s = s_min + (s_max - s_min) * 0.5 * (1 + math.cos(math.pi * progress))
    return s

# Usage during denoising loop:
# for step in range(total_steps):
#     s = cosine_guidance_schedule(step, total_steps)
#     eps_guided = eps_uncond + s * (eps_cond - eps_uncond)
```

---

## Negative Prompts as CFG

One of the most popular user-facing features of modern image and video generators --- negative prompts --- is a direct application of CFG with a modified unconditional term.

### Standard CFG Recap

In standard CFG, the unconditional prediction \(\epsilon_\theta(x_t, \emptyset)\) is obtained by feeding a null conditioning token. The guided prediction extrapolates away from this unconditional baseline toward the positive prompt.

### Negative Prompt Modification

With a negative prompt \(c_\text{neg}\), we replace the null unconditional prediction with the negative-conditioned prediction:

$$\hat{\epsilon}_\theta(x_t, c_\text{pos}, c_\text{neg}) = \epsilon_\theta(x_t, c_\text{neg}) + s \cdot \left[\epsilon_\theta(x_t, c_\text{pos}) - \epsilon_\theta(x_t, c_\text{neg})\right]$$

This changes the direction of extrapolation. Instead of moving away from "generic unconditional output" toward the positive prompt, we move away from the negative prompt toward the positive prompt.

### Geometric Interpretation

The standard CFG direction is:

$$\Delta_\text{standard} = \epsilon_\theta(x_t, c_\text{pos}) - \epsilon_\theta(x_t, \emptyset)$$

The negative-prompt direction is:

$$\Delta_\text{negative} = \epsilon_\theta(x_t, c_\text{pos}) - \epsilon_\theta(x_t, c_\text{neg})$$

These are different vectors. \(\Delta_\text{standard}\) points from "generic" toward "matching the prompt." \(\Delta_\text{negative}\) points from "matching the negative prompt" toward "matching the positive prompt."

If your negative prompt describes blurriness, the direction \(\Delta_\text{negative}\) not only moves toward your positive prompt but also actively moves away from blur. This is why negative prompts like "blurry, low quality, distorted" are so effective --- they add an additional repulsive force that complements the attractive force of the positive prompt.

### The Math of Why Negative Prompts Work

We can decompose \(\Delta_\text{negative}\) as:

$$\Delta_\text{negative} = \underbrace{[\epsilon_\theta(x_t, c_\text{pos}) - \epsilon_\theta(x_t, \emptyset)]}_{\text{positive guidance}} + \underbrace{[\epsilon_\theta(x_t, \emptyset) - \epsilon_\theta(x_t, c_\text{neg})]}_{\text{negative avoidance}}$$

$$= \Delta_\text{standard} + \Delta_\text{avoidance}$$

The negative prompt adds a new component \(\Delta_\text{avoidance}\) that pushes away from the negative condition. This component is independent of (or at least different from) the positive guidance direction, which means negative prompts can address failure modes that positive prompts cannot.

For example, the positive prompt "a sharp, detailed photograph of a cat" and the negative prompt "blurry, out of focus" do not produce identical results even though they seem to describe the same intent. The positive prompt activates features in the model associated with "sharp" and "detailed." The negative prompt deactivates features associated with "blurry" and "out of focus." These are overlapping but distinct sets of features in the model's representation space.

### Dual CFG: Separate Scales for Positive and Negative

Some implementations allow separate guidance scales for positive and negative prompts:

$$\hat{\epsilon} = \epsilon_\theta(x_t, \emptyset) + s_\text{pos} \cdot [\epsilon_\theta(x_t, c_\text{pos}) - \epsilon_\theta(x_t, \emptyset)] - s_\text{neg} \cdot [\epsilon_\theta(x_t, c_\text{neg}) - \epsilon_\theta(x_t, \emptyset)]$$

This requires three forward passes (unconditional, positive, negative) instead of two, but gives finer control. You can strongly avoid the negative prompt (\(s_\text{neg}\) high) while moderately following the positive prompt (\(s_\text{pos}\) moderate), or vice versa.

---

## CFG Rescale and Advanced Variants

Standard CFG has a known failure mode: at high guidance scales, the guided noise prediction \(\hat{\epsilon}\) can have a much larger magnitude than the trained range. This causes oversaturation and loss of contrast.

### The Magnitude Problem

The trained noise predictions \(\epsilon_\theta(x_t, c)\) and \(\epsilon_\theta(x_t, \emptyset)\) have magnitudes that the denoising sampler expects. When \(s > 1\), the guided prediction:

$$\hat{\epsilon} = \epsilon_\theta(x_t, \emptyset) + s \cdot \Delta$$

can have \(\|\hat{\epsilon}\| \gg \|\epsilon_\theta(x_t, c)\|\). The sampler was designed for predictions of a certain magnitude, and oversized predictions cause:

- Pixel values clipping to [0, 1] extremes
- Loss of mid-tone detail
- Oversaturated colors
- Flat, posterized regions

### CFG Rescale (Lin et al., 2024)

The CFG rescale technique normalizes the guided prediction to match the expected magnitude:

$$\hat{\epsilon}_\text{rescaled} = \hat{\epsilon} \cdot \frac{\|\epsilon_\theta(x_t, c)\|}{\|\hat{\epsilon}\|} \cdot \phi + \hat{\epsilon} \cdot (1 - \phi)$$

where \(\phi \in [0, 1]\) controls how much rescaling to apply. At \(\phi = 1\), the magnitude is fully normalized. At \(\phi = 0\), no rescaling is applied. Typical values: \(\phi = 0.7\).

This preserves the direction of the guided prediction (which encodes the CFG effect) while controlling its magnitude (which causes artifacts). The result: you can use higher guidance scales without oversaturation.

### Perp-Neg (Armandpour et al., 2023)

Another variant decomposes negative prompts into components perpendicular and parallel to the positive guidance direction. Only the perpendicular component is used for negative guidance, avoiding interference with the positive signal:

$$\Delta_\text{neg}^\perp = \Delta_\text{neg} - \frac{\Delta_\text{neg} \cdot \Delta_\text{pos}}{\|\Delta_\text{pos}\|^2} \Delta_\text{pos}$$

$$\hat{\epsilon} = \epsilon_\theta(x_t, \emptyset) + s_\text{pos} \cdot \Delta_\text{pos} - s_\text{neg} \cdot \Delta_\text{neg}^\perp$$

This prevents the negative prompt from accidentally reducing the positive prompt's influence when the two conditioning directions are not orthogonal.

### Self-Attention Guidance (SAG)

An even more recent variant replaces the classifier-free approach with self-attention map analysis. Instead of comparing conditional vs. unconditional predictions, SAG analyzes the self-attention maps within the model to identify regions that need enhancement. This is beyond the scope of CFG proper but represents the evolution of the guidance concept.

---

## Implementation Notes

### Computational Cost

CFG doubles the computational cost of inference. Each denoising step requires two forward passes through the model:

1. \(\epsilon_\theta(x_t, \emptyset)\) --- unconditional pass
2. \(\epsilon_\theta(x_t, c)\) --- conditional pass

For a model with \(N\) parameters and \(T\) denoising steps, the total FLOPs are approximately \(2 \times T \times \text{FLOPs}(N)\) instead of \(T \times \text{FLOPs}(N)\).

This is why guidance distillation is valuable. Models like Flux and LCM (Latent Consistency Models) train with guidance baked into the weights, requiring only a single forward pass at inference time. The distilled model approximates \(\hat{\epsilon}_\theta(x_t, c, s)\) directly, without needing two separate passes.

### Batched CFG

In practice, the two forward passes are batched together as a single forward pass with batch size 2:

```python
# Efficient batched CFG implementation
def cfg_step(model, x_t, t, c_pos, c_null, guidance_scale):
    # Batch both predictions into a single forward pass
    x_batch = torch.cat([x_t, x_t], dim=0)
    c_batch = torch.cat([c_null, c_pos], dim=0)
    t_batch = torch.cat([t, t], dim=0)

    # Single forward pass with batch size 2
    eps_batch = model(x_batch, t_batch, c_batch)

    # Split predictions
    eps_uncond, eps_cond = eps_batch.chunk(2, dim=0)

    # Apply CFG
    eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    return eps_guided
```

This uses the same memory as a batch-size-2 forward pass and benefits from GPU parallelism, making CFG roughly 1.5x (not 2x) the cost of non-guided inference in practice.

### Guidance with Multiple Conditions

When multiple conditioning signals are present (e.g., text prompt + reference image + camera control), CFG can be applied independently to each:

$$\hat{\epsilon} = \epsilon_\theta(x_t, \emptyset) + \sum_i s_i \cdot [\epsilon_\theta(x_t, c_i) - \epsilon_\theta(x_t, \emptyset)]$$

This requires \(N+1\) forward passes for \(N\) conditions, which quickly becomes expensive. In practice, multi-condition models usually encode all conditions jointly and apply a single CFG scale to the combined conditioning.

### Numerical Stability

At extreme guidance scales (\(s > 20\)), numerical overflow can occur in float16 inference. The guided prediction magnitude can exceed the representable range. Mitigations:

- Use float32 for the CFG calculation even if the model runs in float16
- Apply CFG rescale with \(\phi \geq 0.5\) at high scales
- Clamp the guided prediction magnitude to a reasonable range (e.g., 4 standard deviations)

---

## Summary

Classifier-free guidance is elegant in its derivation and profound in its impact. From Bayes' theorem through score function decomposition to the final equation, the math tells a clean story: amplify the difference between conditional and unconditional predictions to sharpen conditional generation.

The key equations to remember:

**The CFG equation:**

$$\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot [\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset)]$$

**The implicit distribution:**

$$\tilde{p}(x \mid c) \propto p(x) \cdot p(c \mid x)^s$$

**Negative prompt extension:**

$$\hat{\epsilon}_\theta = \epsilon_\theta(x_t, c_\text{neg}) + s \cdot [\epsilon_\theta(x_t, c_\text{pos}) - \epsilon_\theta(x_t, c_\text{neg})]$$

For video generation specifically:
- Use lower guidance scales (\(s = 4\text{--}6\)) than image models (\(s = 7\text{--}8\)) to preserve temporal coherence
- Consider dynamic guidance schedules that decrease \(s\) in later denoising steps
- Guidance distillation (single-pass models) halves inference cost and is where the field is heading

Understanding CFG at this level is not just academic. When you tune generation quality for a production video pipeline, you are adjusting these parameters. When you debug why a video flickers or looks oversaturated, you are diagnosing CFG artifacts. When you evaluate whether to use an open-source model with configurable CFG vs. an API model with fixed internal guidance, you are making an informed architectural decision.

---

## References

1. Ho, J. & Salimans, T. (2022). *Classifier-Free Diffusion Guidance.* NeurIPS 2022 Workshop on Score-Based Methods.
2. Dhariwal, P. & Nichol, A. (2021). *Diffusion Models Beat GANs on Image Synthesis.* NeurIPS 2021.
3. Kynkaanniemi, T. et al. (2019). *Improved Precision and Recall Metric for Assessing Generative Models.* NeurIPS 2019.
4. Lin, S. et al. (2024). *Common Diffusion Noise Schedules and Sample Steps are Flawed.* WACV 2024.
5. Armandpour, M. et al. (2023). *Re-imagine the Negative Prompt Algorithm: Transform 2D Diffusion into 3D, alleviate Janus Problem and Beyond.* arXiv:2304.04968.
6. Song, Y. et al. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021.
