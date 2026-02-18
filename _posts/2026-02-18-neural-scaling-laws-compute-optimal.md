---
layout: post
title: "Neural Scaling Laws: Power Laws, Chinchilla, and the Economics of Making Models Bigger"
date: 2026-02-18
category: math
---

There is an uncomfortable truth at the heart of modern deep learning: the single most reliable way to make a model better is to make it bigger and feed it more data. Not a better architecture. Not a cleverer training trick. Just more. And the relationship between "more" and "better" follows a remarkably clean mathematical pattern --- a power law --- that holds across orders of magnitude, across model families, and across modalities.

This post is a thorough mathematical treatment of neural scaling laws. We will define exactly what a power law is, walk through the two landmark papers --- Kaplan et al. (2020) and Hoffmann et al. (2022, "Chinchilla") --- derive the compute-optimal training recipe using Lagrange multipliers, explore why power laws emerge at all, extend to vision and video, and then confront the brutal economics of diminishing returns. Along the way we will build intuition with Python simulations and visualizations.

---

## Table of Contents

1. [What Is a Power Law?](#what-is-a-power-law)
2. [The Empirical Observation: Loss Scales Predictably](#the-empirical-observation-loss-scales-predictably)
3. [Kaplan Scaling Laws (2020)](#kaplan-scaling-laws-2020)
4. [Chinchilla (2022): The Correction](#chinchilla-2022-the-correction)
5. [Deriving the Chinchilla Optimal Allocation](#deriving-the-chinchilla-optimal-allocation)
6. [Why Power Laws? The Deep Question](#why-power-laws-the-deep-question)
7. [Beyond Language: Vision and Video](#beyond-language-vision-and-video)
8. [The Economics of Diminishing Returns](#the-economics-of-diminishing-returns)
9. [Compute-Optimal vs. Inference-Optimal](#compute-optimal-vs-inference-optimal)
10. [Python Simulations](#python-simulations)
11. [Conclusion](#conclusion)

---

## What Is a Power Law?

Before we talk about neural networks, we need to be precise about what a power law is, because the term gets used loosely.

A **power law** is a functional relationship of the form:

$$f(x) = a \cdot x^{-\alpha}$$

where $a > 0$ is a constant (the scale factor) and $\alpha > 0$ is the **exponent** (sometimes called the scaling exponent). The defining characteristic: on a log-log plot, a power law appears as a straight line. Take the logarithm of both sides:

$$\log f(x) = \log a - \alpha \log x$$

This is just $y = b + mx$ with slope $-\alpha$. So if you plot your data on log-log axes and see a straight line, you have a power law.

Power laws show up everywhere in nature: earthquake magnitudes (Gutenberg-Richter law), city populations (Zipf's law), income distributions (Pareto), species extinction rates, citations of academic papers, and --- as it turns out --- the loss of neural networks as a function of their size.

For neural scaling laws, there is an important refinement. The loss does not go to zero as the model grows infinitely large. There is a floor --- the **irreducible loss** $L_\infty$ --- representing the entropy of the data itself. No model, no matter how large, can predict truly random content. So the scaling law takes the form:

$$L(x) = a \cdot x^{-\alpha} + L_\infty$$

The quantity $L(x) - L_\infty$ is the **reducible loss**: the part that better modeling can actually shrink. It is this reducible component that follows the power law.

<svg viewBox="0 0 700 380" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 700px; display: block; margin: 2em auto;">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>
  <!-- Title -->
  <text x="350" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" font-weight="bold" fill="#333">Power Law with Irreducible Loss</text>
  <!-- Axes -->
  <line x1="80" y1="320" x2="650" y2="320" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="80" y1="320" x2="80" y2="40" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  <text x="380" y="360" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#333">log(x) — e.g. log(Parameters) or log(Data)</text>
  <text x="25" y="180" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#333" transform="rotate(-90, 25, 180)">Loss L(x)</text>
  <!-- Irreducible loss line -->
  <line x1="80" y1="270" x2="640" y2="270" stroke="#e57373" stroke-width="2" stroke-dasharray="8,4"/>
  <text x="645" y="266" font-family="Arial, sans-serif" font-size="12" fill="#e57373" text-anchor="start">L&#x221E;</text>
  <!-- Power law curve -->
  <path d="M 100,80 C 140,100 180,140 220,175 S 320,220 400,240 S 500,255 620,262" fill="none" stroke="#4fc3f7" stroke-width="3"/>
  <!-- Reducible loss bracket -->
  <line x1="160" y1="145" x2="160" y2="270" stroke="#66bb6a" stroke-width="1.5" stroke-dasharray="4,3"/>
  <line x1="155" y1="145" x2="165" y2="145" stroke="#66bb6a" stroke-width="1.5"/>
  <line x1="155" y1="270" x2="165" y2="270" stroke="#66bb6a" stroke-width="1.5"/>
  <text x="175" y="210" font-family="Arial, sans-serif" font-size="11" fill="#66bb6a" font-weight="bold">Reducible</text>
  <text x="175" y="225" font-family="Arial, sans-serif" font-size="11" fill="#66bb6a" font-weight="bold">loss</text>
  <!-- Curve label -->
  <text x="300" y="145" font-family="Arial, sans-serif" font-size="12" fill="#4fc3f7" font-weight="bold">L(x) = a x^(-&#x3B1;) + L&#x221E;</text>
  <!-- Slope annotation -->
  <line x1="130" y1="110" x2="230" y2="110" stroke="#999" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="230" y1="110" x2="230" y2="175" stroke="#999" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="245" y="148" font-family="Arial, sans-serif" font-size="11" fill="#999">slope = -&#x3B1;</text>
</svg>

---

## The Empirical Observation: Loss Scales Predictably

Here is what researchers at OpenAI, DeepMind, and other labs found when they trained language models of many different sizes on many different amounts of data, and plotted the resulting cross-entropy loss.

**Cross-entropy loss** measures how well a model predicts the next token in a sequence. If the model assigns probability $p_i$ to the correct next token, the loss for that token is $-\log p_i$. Averaged over many tokens, this gives the model's loss $L$. Lower is better. A loss of 0 would mean perfect prediction; the entropy of natural language sets a floor around $L_\infty \approx 1.6$ nats for English text (roughly corresponding to a perplexity of 5).

The observation, reported first systematically by Hestness et al. (2017) and then in much greater detail by Kaplan et al. (2020), is this:

> When you plot loss against model size $N$ (number of parameters), dataset size $D$ (number of tokens), or total compute $C$ (floating point operations), **the data falls on a straight line in log-log space**. This holds over at least 6-7 orders of magnitude.

This is surprising for several reasons:

1. **Neural networks are complicated.** They have nonlinear activations, attention mechanisms, layer norms, residual connections. There is no obvious reason why the aggregate performance of such a system should follow a simple mathematical law.

2. **The relationship holds across architectures.** Whether you use a 12-layer transformer or a 96-layer transformer, whether you use different learning rates or different batch sizes (within reason), the loss falls on approximately the same power law curve. Architecture details shift the curve slightly, but the exponent stays nearly the same.

3. **The exponent is small.** The improvements are real but gradual. You need orders of magnitude more resources for modest gains. This means the curve is not steep --- it is gentle, relentless, and (as we will see) economically punishing.

---

## Kaplan Scaling Laws (2020)

The landmark paper is Kaplan et al., "Scaling Laws for Neural Language Models" (2020). They trained over 1,000 transformer language models ranging from around 768 parameters to 1.5 billion parameters and proposed three independent power laws.

### Three Power Laws

**Loss vs. Parameters** (holding data fixed and large):

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076, \quad N_c \approx 8.8 \times 10^{13}$$

**Loss vs. Data** (holding parameters fixed and large):

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095, \quad D_c \approx 5.4 \times 10^{13}$$

**Loss vs. Compute** (using compute-optimal allocation):

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050, \quad C_c \approx 3.1 \times 10^8$$

Here $N$ is the number of non-embedding parameters, $D$ is the number of training tokens, and $C$ is the total number of floating-point operations (FLOPs) used during training.

The notation $N_c$, $D_c$, $C_c$ are characteristic scales --- fitting constants that set where the power law "starts." The important quantities are the exponents $\alpha_N$, $\alpha_D$, $\alpha_C$.

### The Compute Approximation

For a decoder-only transformer with $N$ parameters trained on $D$ tokens, the total compute is approximately:

$$C \approx 6ND$$

Where does the 6 come from? Each token requires a forward pass and a backward pass. The forward pass through a dense layer with $N$ parameters takes approximately $2N$ FLOPs (one multiply and one add per parameter). The backward pass takes approximately $4N$ FLOPs (computing gradients with respect to both weights and activations). So the total per token is $\approx 6N$ FLOPs, and over $D$ tokens we get $C \approx 6ND$.

This is an approximation. It ignores attention computation (which scales quadratically with sequence length but is usually a small fraction of total compute for large models), embedding layers, and layer norms. But it is accurate to within a factor of 2 for most practical transformer architectures, and it is extremely useful for reasoning about the tradeoff between model size and data.

### Kaplan's Compute-Optimal Allocation

Given a fixed compute budget $C$, how should you split it between model size $N$ and dataset size $D$? This is a constrained optimization problem.

Kaplan et al. proposed a combined loss function:

$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}$$

They found that minimizing this subject to $C = 6ND$ yields:

$$N^* \propto C^{0.73}, \quad D^* \propto C^{0.27}$$

The exponents sum to 1 (since $C \propto ND$, $0.73 + 0.27 = 1.00$). But the asymmetry is dramatic: **parameters should scale much faster than data**. If your compute budget increases by $10\times$, you should make the model $\approx 5.4\times$ bigger but only use $\approx 1.9\times$ more data.

This finding directly influenced the design of GPT-3 (2020): 175 billion parameters trained on only 300 billion tokens. By the Chinchilla analysis we will derive below, this was significantly undertrained --- the model was far too large for the amount of data it saw.

---

## Chinchilla (2022): The Correction

Two years later, Hoffmann et al. at DeepMind published "Training Compute-Optimal Large Language Models" (2022), commonly known as the "Chinchilla paper." They argued that Kaplan's scaling laws were biased because the models in the Kaplan study were **not trained to convergence**. Each model was trained with a fixed number of tokens regardless of its size, and the learning rate schedules were not individually tuned.

When they corrected for this --- training each model size with its own optimal learning rate schedule and enough data to approach convergence --- the scaling exponents changed significantly.

### The Chinchilla Loss Function

Chinchilla proposed a decomposable loss function:

$$L(N, D) = \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} + L_\infty$$

with fitted parameters:

$$A \approx 406.4, \quad B \approx 410.7, \quad \alpha \approx 0.34, \quad \beta \approx 0.28, \quad L_\infty \approx 1.69$$

This has a clean interpretation. The loss decomposes into three additive terms:

1. **$A / N^{\alpha}$** --- the approximation loss. The model is too small to represent the true data distribution. More parameters reduce this.

2. **$B / D^{\beta}$** --- the estimation loss. The model has not seen enough data to learn the parameters it has. More data reduces this.

3. **$L_\infty$** --- the irreducible loss. The inherent entropy of the data. No amount of modeling can reduce this. For natural language, this reflects genuine unpredictability --- the next word in a sentence is often not deterministic.

The key structural insight: these three sources of loss are **additive and independent**. Making the model bigger only helps the first term. Adding more data only helps the second term. Neither helps the third.

<svg viewBox="0 0 700 350" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" font-weight="bold" fill="#333">Decomposition of Loss</text>
  <!-- Three boxes -->
  <rect x="30" y="50" width="190" height="260" rx="10" fill="#e3f2fd" stroke="#4fc3f7" stroke-width="2"/>
  <rect x="250" y="50" width="190" height="260" rx="10" fill="#fce4ec" stroke="#e57373" stroke-width="2"/>
  <rect x="470" y="50" width="190" height="260" rx="10" fill="#f3e5f5" stroke="#ab47bc" stroke-width="2"/>
  <!-- Plus signs -->
  <text x="235" y="185" text-anchor="middle" font-family="Arial, sans-serif" font-size="28" font-weight="bold" fill="#333">+</text>
  <text x="455" y="185" text-anchor="middle" font-family="Arial, sans-serif" font-size="28" font-weight="bold" fill="#333">+</text>
  <!-- Box 1: Approximation -->
  <text x="125" y="80" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#1976d2">Approximation</text>
  <text x="125" y="130" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" fill="#1976d2">A / N^&#x3B1;</text>
  <text x="125" y="170" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#555">Model too small to</text>
  <text x="125" y="186" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#555">represent the true</text>
  <text x="125" y="202" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#555">distribution</text>
  <text x="125" y="240" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#1976d2" font-weight="bold">Fix: more parameters</text>
  <text x="125" y="265" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#555">&#x3B1; &#x2248; 0.34</text>
  <text x="125" y="285" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#555">Diminishes as N grows</text>
  <!-- Box 2: Estimation -->
  <text x="345" y="80" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#c62828">Estimation</text>
  <text x="345" y="130" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" fill="#c62828">B / D^&#x3B2;</text>
  <text x="345" y="170" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#555">Not enough data to</text>
  <text x="345" y="186" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#555">learn the parameters</text>
  <text x="345" y="202" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#555">the model has</text>
  <text x="345" y="240" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#c62828" font-weight="bold">Fix: more data</text>
  <text x="345" y="265" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#555">&#x3B2; &#x2248; 0.28</text>
  <text x="345" y="285" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#555">Diminishes as D grows</text>
  <!-- Box 3: Irreducible -->
  <text x="565" y="80" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#7b1fa2">Irreducible</text>
  <text x="565" y="130" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" fill="#7b1fa2">L&#x221E;</text>
  <text x="565" y="170" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#555">Inherent entropy of</text>
  <text x="565" y="186" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#555">the data itself</text>
  <text x="565" y="202" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#555">(randomness in language)</text>
  <text x="565" y="240" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#7b1fa2" font-weight="bold">No fix possible</text>
  <text x="565" y="265" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#555">&#x2248; 1.69 nats</text>
  <text x="565" y="285" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#555">Constant floor</text>
</svg>

### Where Kaplan Went Wrong

The critical difference between Kaplan and Chinchilla lies in how the training runs were configured.

Kaplan trained all models for a fixed "token budget" (or close to it), then observed how loss varied with model size. But a small model converges quickly on a given dataset, while a large model needs more data to converge. By using the same dataset size for all model sizes, the large models were undertrained --- they had not seen enough data to realize the benefit of their extra parameters. This made model size look disproportionately effective compared to data.

Chinchilla corrected for this by training each model size on varying amounts of data and observing the loss surface $L(N, D)$ directly. They found that the marginal return from data was much higher than Kaplan had estimated --- meaning data was more valuable than previously thought.

The practical consequence was stark: the 70-billion parameter Chinchilla model, trained on 1.4 trillion tokens, matched or outperformed the 280-billion parameter Gopher model trained on 300 billion tokens. Four times fewer parameters, nearly five times more data, same compute budget, better results.

---

## Deriving the Chinchilla Optimal Allocation

Now for the main derivation. We want to find the compute-optimal model size $N^*$ and dataset size $D^*$ that minimize the loss for a given compute budget. This is a classic constrained optimization problem, and we solve it with **Lagrange multipliers**.

### The Setup

We want to minimize:

$$L(N, D) = \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} + L_\infty$$

subject to the compute constraint:

$$g(N, D) = 6ND - C = 0$$

where $C$ is the fixed compute budget.

**Lagrange multipliers** is a technique for finding the extrema of a function subject to constraints. The idea: at an optimum of $L$ subject to $g = 0$, the gradient of $L$ must be proportional to the gradient of $g$ --- otherwise you could move along the constraint surface and improve $L$. Formally, we require:

$$\nabla L = \lambda \nabla g$$

for some scalar $\lambda$ (the Lagrange multiplier).

### The Lagrangian

We form the Lagrangian:

$$\mathcal{L}(N, D, \lambda) = \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} + L_\infty + \lambda(6ND - C)$$

### Taking Partial Derivatives

**With respect to $N$:**

$$\frac{\partial \mathcal{L}}{\partial N} = -\frac{\alpha A}{N^{\alpha + 1}} + 6\lambda D = 0$$

This gives us:

$$6\lambda D = \frac{\alpha A}{N^{\alpha + 1}} \quad \Rightarrow \quad \lambda = \frac{\alpha A}{6 D N^{\alpha + 1}} \tag{1}$$

**With respect to $D$:**

$$\frac{\partial \mathcal{L}}{\partial D} = -\frac{\beta B}{D^{\beta + 1}} + 6\lambda N = 0$$

This gives us:

$$6\lambda N = \frac{\beta B}{D^{\beta + 1}} \quad \Rightarrow \quad \lambda = \frac{\beta B}{6 N D^{\beta + 1}} \tag{2}$$

### Equating

Setting equations (1) and (2) equal (both equal $\lambda$):

$$\frac{\alpha A}{6 D N^{\alpha + 1}} = \frac{\beta B}{6 N D^{\beta + 1}}$$

Simplify (the factors of 6 cancel):

$$\frac{\alpha A}{D N^{\alpha + 1}} = \frac{\beta B}{N D^{\beta + 1}}$$

Cross-multiply:

$$\alpha A \cdot N \cdot D^{\beta + 1} = \beta B \cdot D \cdot N^{\alpha + 1}$$

$$\alpha A \cdot D^{\beta} = \beta B \cdot N^{\alpha}$$

This is the **optimality condition**: at the compute-optimal point, the marginal reduction in loss from increasing $N$ (weighted by the compute cost) equals the marginal reduction from increasing $D$.

Solving for the ratio:

$$\frac{N^{\alpha}}{D^{\beta}} = \frac{\alpha A}{\beta B} \tag{3}$$

This tells us the relationship between the optimal $N$ and $D$. Now we use the constraint $C = 6ND$ to express everything in terms of $C$.

### Expressing $N^*$ and $D^*$ in Terms of $C$

From the constraint: $D = C / (6N)$. Substitute into equation (3):

$$N^{\alpha} = \frac{\alpha A}{\beta B} \cdot D^{\beta} = \frac{\alpha A}{\beta B} \cdot \left(\frac{C}{6N}\right)^{\beta}$$

$$N^{\alpha} = \frac{\alpha A}{\beta B} \cdot \frac{C^{\beta}}{6^{\beta} N^{\beta}}$$

$$N^{\alpha + \beta} = \frac{\alpha A}{\beta B \cdot 6^{\beta}} \cdot C^{\beta}$$

$$N = \left(\frac{\alpha A}{\beta B \cdot 6^{\beta}}\right)^{\frac{1}{\alpha + \beta}} \cdot C^{\frac{\beta}{\alpha + \beta}}$$

Similarly, from $N = C / (6D)$:

$$D = \left(\frac{\beta B \cdot 6^{\alpha}}{\alpha A}\right)^{\frac{1}{\alpha + \beta}} \cdot C^{\frac{\alpha}{\alpha + \beta}}$$

### The Scaling Exponents

The compute-scaling exponents for $N$ and $D$ are:

$$N^* \propto C^{\frac{\beta}{\alpha + \beta}}, \quad D^* \propto C^{\frac{\alpha}{\alpha + \beta}}$$

With Chinchilla's values $\alpha \approx 0.34$ and $\beta \approx 0.28$:

$$\frac{\beta}{\alpha + \beta} = \frac{0.28}{0.62} \approx 0.45, \quad \frac{\alpha}{\alpha + \beta} = \frac{0.34}{0.62} \approx 0.55$$

So $N^* \propto C^{0.45}$ and $D^* \propto C^{0.55}$.

These exponents are close to $0.5$ --- meaning **parameters and data should scale roughly equally with compute**. This is the Chinchilla result. For every doubling of parameters, you need approximately a doubling of data.

Compare with Kaplan: $N^* \propto C^{0.73}$, $D^* \propto C^{0.27}$. Kaplan said scale parameters $2.7\times$ faster than data. Chinchilla says scale them approximately equally.

### The Optimal Token-to-Parameter Ratio

We can derive the optimal ratio $D^*/N^*$. Since $C = 6N^*D^*$ and both scale as $\sim C^{0.5}$, the ratio $D^*/N^*$ is approximately constant --- it does not change with scale.

Using the Chinchilla fits:

$$\frac{D^*}{N^*} \approx 20$$

That is: the compute-optimal training recipe uses about 20 tokens per parameter. A 7B model should train on ~140B tokens. A 70B model should train on ~1.4T tokens. A 700B model would need ~14T tokens.

---

## Why Power Laws? The Deep Question

We have established the empirical fact: loss follows power laws. But why? This is, honestly, not fully understood. But there are several compelling hypotheses that each illuminate part of the picture.

### Hypothesis 1: The Data Has Multi-Scale Structure

Natural data --- language, images, video --- has structure at many scales simultaneously. In language: character-level patterns (spelling), word-level patterns (grammar), sentence-level patterns (syntax), paragraph-level patterns (rhetoric), document-level patterns (narrative). Each scale has its own statistical regularities.

A model with $N$ parameters has a certain capacity to resolve statistical regularities. As you add parameters, the model can resolve progressively finer-grained features. But the number of features at each scale follows its own distribution, and the contribution of each scale to the total loss is roughly power-law distributed.

Think of it like a fractal. A fractal has detail at every scale of magnification, and the amount of detail at scale $\epsilon$ is proportional to $\epsilon^{-d_f}$ where $d_f$ is the fractal dimension. If the data manifold is fractal-like --- having structure at every scale --- then a model that "resolves" down to scale $\epsilon$ (requiring $N \sim \epsilon^{-d_f}$ parameters) will capture a fraction of variance that also follows a power law.

### Hypothesis 2: Connection to Statistical Physics

There is a deep connection to statistical mechanics, specifically to the theory of **critical phenomena**.

In statistical physics, a system at a **critical point** (like water at exactly the boiling temperature and pressure) exhibits power-law correlations. The correlation length diverges, fluctuations happen at all scales, and the system is described by **universal** exponents that depend only on the symmetry class and dimensionality of the system --- not on microscopic details.

The analogy to neural scaling: the loss exponents ($\alpha_N \approx 0.076$, etc.) are remarkably insensitive to architectural details. They are the same for GPT-style models, for different layer counts, for different attention head configurations. This universality is reminiscent of critical exponents in physics.

The hypothesis is that the learning process itself is a kind of critical phenomenon. The neural network, as it grows, is performing a form of coarse-graining (in the renormalization group sense) of the data distribution. Each layer of the network resolves one more level of the hierarchy. The power-law exponent reflects the fractal dimension of the data distribution in some appropriate sense.

This is not just an analogy. Roberts et al. (2022) and Bahri et al. (2024) have made this connection precise, showing that certain scaling exponents can be derived from the spectral properties of the data covariance matrix and the network's effective kernel.

### Hypothesis 3: The Power Law of Learning Curves

There is a classical result in learning theory. For a model class with $N$ effective parameters learning from $D$ data points, the generalization error (how much worse the model does on test data vs. training data) scales as:

$$\epsilon_{gen} \sim \frac{N}{D}$$

This is the bias-variance tradeoff. Too few parameters (high bias): the model cannot fit the data. Too few data points (high variance): the model overfits. But if we layer in the approximation error --- which depends on the complexity of the true function relative to the model class --- we get a richer picture. If the true function has a certain smoothness (belonging to a Sobolev space of order $s$ in dimension $d$), the optimal approximation rate is:

$$L \sim N^{-2s/d}$$

This is a power law, and the exponent depends on the intrinsic complexity ($s$, $d$) of the function being learned. The neural scaling exponents, from this viewpoint, encode information about the intrinsic complexity of natural language.

### The "Broken Power Law" Phenomenon

One important caveat: the power law is not perfectly smooth. There are hints of **broken power laws** --- regions where the exponent changes. This may correspond to capability thresholds: the model transitions from not being able to do a certain task to being able to do it, and this discrete jump modifies the smooth scaling trend.

Think of it as a phase transition in capability space. Below some critical size, the model cannot do multi-digit arithmetic at all. Above it, the capability emerges suddenly. This is related to the "emergent abilities" discussion (Wei et al., 2022), though the exact nature of these transitions --- whether they are genuine phase transitions or artifacts of how we measure --- remains debated.

---

## Beyond Language: Vision and Video

Do the same power laws hold for other modalities?

### Image Generation

For image generation models (GANs, diffusion models), the quality metric is typically FID (Frechet Inception Distance) rather than cross-entropy loss. FID measures the statistical distance between generated images and real images in the feature space of an InceptionV3 network. Lower is better.

The evidence is that yes, power laws hold here too, but the exponents are different. Henighan et al. (2020) found scaling laws for autoregressive image models with exponents comparable to language models. For diffusion-based generators, the scaling is typically characterized as:

$$\text{FID}(C) \propto C^{-\alpha_{FID}}$$

with $\alpha_{FID}$ values in the range 0.05--0.15 depending on the architecture and dataset.

The core principle transfers: scaling compute (through larger models or more training) reliably improves image quality, and the improvement follows a power law.

### Video Generation

Video adds a new dimension --- literally. A video model must not only generate plausible individual frames but maintain **temporal coherence**: objects should move smoothly, physics should be consistent, and the visual style should not flicker between frames.

The scaling behavior for video models has some distinctive features:

1. **Spatial quality scales similarly to images.** Per-frame visual quality follows the same kind of power law as image models.

2. **Temporal consistency is harder to scale.** Maintaining coherence across frames requires the model to learn long-range dependencies in time. This appears to require disproportionately more data than simply scaling the model. Preliminary evidence suggests that the $\beta$ exponent (data scaling) is relatively larger for video than for language or images --- meaning video models are more data-hungry.

3. **Video data is harder to curate.** This is perhaps the most important practical constraint. The internet contains trillions of tokens of text, but high-quality video data at scale is much harder to source. Text can be scraped, filtered, and deduplicated relatively easily. Video requires downloading, decoding, filtering for quality and content, handling copyright issues, and dealing with enormous storage and bandwidth requirements. A single hour of 1080p video at 30fps contains about 108,000 frames --- the data cost per "meaningful unit" is far higher than text.

The net effect: video models are likely to be data-limited before they are compute-limited. The Chinchilla optimal point (where the model is neither too big for its data nor too small) may be harder to reach for video because acquiring enough high-quality training data is the binding constraint.

---

## The Economics of Diminishing Returns

Now we turn to the question that keeps CFOs awake at night: given that scaling works, what does it cost?

### The Cost-to-Improve Formula

Suppose the current state of the art achieves loss $L_1$ using compute $C_1$, and we want to reach a new loss $L_2 < L_1$. From the scaling law:

$$L = \left(\frac{C_c}{C}\right)^{\alpha_C} + L_\infty$$

The reducible component is $L - L_\infty = (C_c/C)^{\alpha_C}$, so:

$$C = C_c \cdot (L - L_\infty)^{-1/\alpha_C}$$

The ratio of compute needed:

$$\frac{C_2}{C_1} = \left(\frac{L_1 - L_\infty}{L_2 - L_\infty}\right)^{1/\alpha_C}$$

### What Does Halving the Reducible Loss Cost?

If we want to halve the reducible loss ($L_2 - L_\infty = \frac{1}{2}(L_1 - L_\infty)$):

$$\frac{C_2}{C_1} = 2^{1/\alpha_C}$$

With $\alpha_C \approx 0.05$:

$$\frac{C_2}{C_1} = 2^{20} = 1,048,576$$

That is: **halving the reducible loss requires approximately one million times more compute**. This is the brutal economics of power laws with small exponents.

### More Realistic Improvements

Halving the loss is an extreme target. What about more modest improvements?

A 10% reduction in reducible loss ($L_2 - L_\infty = 0.9 (L_1 - L_\infty)$):

$$\frac{C_2}{C_1} = \left(\frac{1}{0.9}\right)^{1/0.05} = \left(\frac{10}{9}\right)^{20} \approx 6.7$$

So a 10% loss improvement requires about $7\times$ more compute. A 20% improvement:

$$\frac{C_2}{C_1} = \left(\frac{1}{0.8}\right)^{20} = 1.25^{20} \approx 86.7$$

About $87\times$ more compute for a 20% gain. And a 50% improvement:

$$\frac{C_2}{C_1} = 2^{20} \approx 10^6$$

These numbers make clear why the scaling game has limits. At current GPU prices (roughly $1--2 per A100-hour), a training run costing $100 million delivers a certain loss. Getting a 10% improvement costs $700 million. A 20% improvement costs $8.7 billion. The curve is exponential in cost for linear improvements in loss.

### The Inference Alternative

This cost wall has motivated a different strategy: instead of spending all compute at training time, spend some at **inference time**. Techniques include:

- **Chain-of-thought prompting**: Have the model reason step by step, using more tokens (and hence more FLOPs) per query
- **Best-of-N sampling**: Generate $N$ candidate answers and select the best one (using a verifier or reward model)
- **Tree search**: Explore multiple reasoning paths and select the most promising

The key insight is that inference compute scales differently. If you generate $N$ samples and pick the best, the effective quality improvement follows:

$$L_{eff}(N_{samples}) \sim L_{base} - c \cdot \log(N_{samples})$$

This is logarithmic, not power-law --- which means diminishing returns kick in even faster for any single query. But the advantage is that you only pay the inference cost when you actually serve a query, not upfront in a massive training run. For many applications, the total cost (training + all inference) can be lower with a smaller, cheaper-to-run model that uses more inference-time compute.

---

## Compute-Optimal vs. Inference-Optimal

The Chinchilla result tells you how to train the best model for a given compute budget. But the best model to **deploy** is not necessarily the compute-optimal one. This is because training is a one-time cost, but inference happens millions or billions of times.

### The Total Cost Model

Let $C_{train}$ be the training compute and $c_{inf}$ be the inference compute per token. If the model will serve $T$ total tokens over its lifetime, the total compute is:

$$C_{total} = C_{train} + c_{inf} \cdot T$$

For a transformer with $N$ parameters, the inference compute per token is approximately $2N$ FLOPs (one forward pass). So:

$$C_{total} = 6ND + 2N \cdot T$$

where $D$ is the number of training tokens.

### The Chinchilla-Optimal Case

At the Chinchilla-optimal point, $D^* \approx 20N$, so:

$$C_{train} = 6N \cdot 20N = 120N^2$$

$$C_{total} = 120N^2 + 2NT$$

### The Overtrained (Inference-Optimal) Case

Now suppose we train a smaller model $N' < N^*$ but on much more data $D' \gg 20N'$, keeping the training budget the same: $6N'D' = 6N^*D^*$. The model is "overtrained" --- past the Chinchilla-optimal point. Its loss will be slightly worse than the Chinchilla-optimal model (since we are not at the minimum of the loss surface for this compute budget). But inference is cheaper because the model is smaller.

The total cost becomes:

$$C_{total}' = 6N'D' + 2N' \cdot T = C_{train} + 2N' \cdot T$$

Since $N' < N^*$, the inference term $2N'T$ is smaller. If $T$ is large enough, the savings in inference outweigh the slight increase in loss.

### The Llama Philosophy

This is precisely the philosophy behind Meta's Llama models. Llama 1 (2023) trained a 7B parameter model on 1 trillion tokens --- about 140 tokens per parameter, which is $7\times$ the Chinchilla-optimal ratio of 20 tokens per parameter. The model was heavily overtrained. Its loss was not the best achievable for that training compute, but its small size made it extremely cheap to deploy.

Llama 2 pushed further: 2 trillion tokens for the 7B model (nearly 300 tokens per parameter). Llama 3 used 15 trillion tokens for the 8B model --- almost 2000 tokens per parameter, far beyond Chinchilla-optimal.

The tradeoff calculation: for a model that will be served to millions of users, each generating thousands of tokens per session, the total inference compute $T$ easily reaches $10^{18}$ or more. At that scale, cutting the model size by $4\times$ (and inference cost by $4\times$) is worth a modest loss increase, because the inference savings dominate.

### The Breakeven Point

We can compute the breakeven number of inference tokens $T^*$ where overtaining becomes worth it. Suppose the overtrained model has loss $L' = L^* + \Delta L$ (slightly worse) but uses a model of size $N' = N^*/k$ for some factor $k > 1$. The training cost is the same ($C_{train}$ fixed), but inference savings are:

$$\Delta C_{inf} = 2(N^* - N'/k) \cdot T = 2N^*(1 - 1/k) \cdot T$$

This is positive for all $T > 0$ --- meaning overtraining is always better for inference cost, and the only question is whether the loss penalty $\Delta L$ is acceptable for your application. In practice, the loss penalty from moderate overtraining (say $2\times$ to $5\times$ the Chinchilla-optimal data) is small, on the order of a few percent.

---

## Python Simulations

Let us make this concrete with code. We will fit power laws, visualize the Chinchilla frontier, and plot the cost-to-improve curve.

### Simulation 1: Fitting Power Laws

```python
import numpy as np
import matplotlib.pyplot as plt

# Chinchilla loss model: L(N, D) = A/N^alpha + B/D^beta + L_inf
A = 406.4
B = 410.7
alpha = 0.34
beta = 0.28
L_inf = 1.69

# Generate loss vs. parameters (with D large enough to not matter)
N_range = np.logspace(7, 12, 200)  # 10M to 1T parameters
D_fixed = 1e15  # Very large, so B/D^beta ~ 0
L_N = A / N_range**alpha + B / D_fixed**beta + L_inf

# Generate loss vs. data (with N large enough to not matter)
D_range = np.logspace(8, 14, 200)  # 100M to 100T tokens
N_fixed = 1e14
L_D = A / N_fixed**alpha + B / D_range**beta + L_inf

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss vs Parameters
axes[0].loglog(N_range, L_N - L_inf, color='#4fc3f7', linewidth=2.5)
axes[0].set_xlabel(r'Parameters $N$', fontsize=13)
axes[0].set_ylabel(r'Reducible Loss $L - L_\infty$', fontsize=13)
axes[0].set_title(r'Loss vs. Model Size (Power Law)', fontsize=14)
axes[0].grid(True, alpha=0.3, which='both')

# Fit line for reference
log_N = np.log10(N_range)
log_L = np.log10(L_N - L_inf)
axes[0].text(1e9, 0.5, r'slope $= -\alpha = -$' + f'{alpha}', fontsize=12,
             color='#e57373', fontweight='bold')

# Loss vs Data
axes[1].loglog(D_range, L_D - L_inf, color='#e57373', linewidth=2.5)
axes[1].set_xlabel(r'Training Tokens $D$', fontsize=13)
axes[1].set_ylabel(r'Reducible Loss $L - L_\infty$', fontsize=13)
axes[1].set_title(r'Loss vs. Dataset Size (Power Law)', fontsize=14)
axes[1].grid(True, alpha=0.3, which='both')
axes[1].text(1e10, 0.3, r'slope $= -\beta = -$' + f'{beta}', fontsize=12,
             color='#4fc3f7', fontweight='bold')

plt.tight_layout()
plt.savefig('scaling_power_laws.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Simulation 2: Chinchilla Optimal Frontier

This is the key visualization. We plot iso-loss contours in $(N, D)$ space, overlaid with compute-constraint lines $C = 6ND$. The Chinchilla-optimal point for each compute budget is where the compute line is tangent to an iso-loss contour.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

A = 406.4
B = 410.7
alpha = 0.34
beta = 0.28
L_inf = 1.69

# Create grid in log space
log_N = np.linspace(7, 12, 500)   # 10M to 1T params
log_D = np.linspace(8, 14, 500)   # 100M to 100T tokens
LOG_N, LOG_D = np.meshgrid(log_N, log_D)
N = 10**LOG_N
D = 10**LOG_D

# Compute loss at each (N, D) pair
L = A / N**alpha + B / D**beta + L_inf

# Compute-optimal frontier: N* and D* for each compute level
C_levels = np.logspace(18, 26, 200)
ratio = (alpha * A) / (beta * B)

# From derivation: N* = k * C^(beta/(alpha+beta))
exp_N = beta / (alpha + beta)
exp_D = alpha / (alpha + beta)
k_N = (ratio / 6**beta) ** (1 / (alpha + beta))
k_D = (1 / (ratio * 6**alpha)) ** (1 / (alpha + beta))

N_opt = k_N * C_levels**exp_N
D_opt = k_D * C_levels**exp_D

fig, ax = plt.subplots(figsize=(10, 8))

# Iso-loss contours
loss_levels = [1.8, 1.85, 1.9, 2.0, 2.2, 2.5, 3.0, 4.0]
CS = ax.contour(LOG_N, LOG_D, L, levels=loss_levels,
                colors='#4fc3f7', linewidths=1.5, alpha=0.7)
ax.clabel(CS, inline=True, fontsize=10, fmt=r'$L=%.2f$')

# Compute budget lines (C = 6ND, so log D = log(C/6) - log N)
C_budgets = [1e19, 1e21, 1e23, 1e25]
for C_b in C_budgets:
    log_d_line = np.log10(C_b / 6) - log_N
    mask = (log_d_line >= 8) & (log_d_line <= 14)
    ax.plot(log_N[mask], log_d_line[mask], '--', color='#999',
            linewidth=1, alpha=0.6)
    # Label
    idx = np.argmin(np.abs(log_d_line - 11))
    if mask[idx]:
        ax.text(log_N[idx], log_d_line[idx] + 0.2,
                r'$C=10^{' + str(int(np.log10(C_b))) + r'}$',
                fontsize=9, color='#666', ha='center')

# Chinchilla optimal frontier
mask_opt = (np.log10(N_opt) >= 7) & (np.log10(N_opt) <= 12) & \
           (np.log10(D_opt) >= 8) & (np.log10(D_opt) <= 14)
ax.plot(np.log10(N_opt[mask_opt]), np.log10(D_opt[mask_opt]),
        color='#e57373', linewidth=3, label=r'Chinchilla optimal frontier')

# Reference point: Chinchilla itself (70B params, 1.4T tokens)
ax.plot(np.log10(7e10), np.log10(1.4e12), 'o', color='#66bb6a',
        markersize=12, zorder=5, label=r'Chinchilla (70B, 1.4T)')

# Reference point: GPT-3 (175B params, 300B tokens)
ax.plot(np.log10(1.75e11), np.log10(3e11), 's', color='#ab47bc',
        markersize=12, zorder=5, label=r'GPT-3 (175B, 300B)')

# Reference point: Llama 3 8B (8B params, 15T tokens)
ax.plot(np.log10(8e9), np.log10(1.5e13), '^', color='#ff9800',
        markersize=12, zorder=5, label=r'Llama 3 8B (8B, 15T)')

ax.set_xlabel(r'$\log_{10}(N)$ (Parameters)', fontsize=14)
ax.set_ylabel(r'$\log_{10}(D)$ (Training Tokens)', fontsize=14)
ax.set_title(r'Chinchilla Optimal Frontier with Iso-Loss Contours', fontsize=15)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.2)
ax.set_xlim(7, 12)
ax.set_ylim(8, 14)

plt.tight_layout()
plt.savefig('chinchilla_frontier.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Simulation 3: The Cost-to-Improve Curve

```python
import numpy as np
import matplotlib.pyplot as plt

alpha_C = 0.05  # Compute scaling exponent

# Fractional reduction in reducible loss
reduction = np.linspace(0.01, 0.60, 500)  # 1% to 60% reduction

# Compute multiplier needed
# If L_new - L_inf = (1 - r) * (L_old - L_inf), then
# C_new / C_old = (1 / (1 - r))^(1/alpha_C)
compute_multiplier = (1 / (1 - reduction))**(1 / alpha_C)

fig, ax = plt.subplots(figsize=(10, 6))

ax.semilogy(reduction * 100, compute_multiplier, color='#e57373',
            linewidth=2.5)

# Annotate key points
key_reductions = [0.10, 0.20, 0.30, 0.50]
for r in key_reductions:
    mult = (1 / (1 - r))**(1 / alpha_C)
    ax.plot(r * 100, mult, 'o', color='#333', markersize=8, zorder=5)
    if r == 0.50:
        ax.annotate(f'{r*100:.0f}%: {mult:.1e}x',
                    xy=(r * 100, mult), xytext=(r * 100 - 10, mult * 5),
                    fontsize=11, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#333'))
    else:
        ax.annotate(f'{r*100:.0f}%: {mult:.0f}x',
                    xy=(r * 100, mult), xytext=(r * 100 + 2, mult * 2),
                    fontsize=11, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#333'))

ax.set_xlabel(r'Reduction in Reducible Loss (\%)', fontsize=14)
ax.set_ylabel(r'Compute Multiplier Required ($C_{\mathrm{new}} / C_{\mathrm{old}}$)', fontsize=14)
ax.set_title(r'The Brutal Economics of Scaling: Cost to Improve', fontsize=15)
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(0, 62)
ax.set_ylim(1, 1e8)

# Add a shaded "feasible" region
ax.axhspan(1, 100, alpha=0.1, color='#66bb6a', label=r'Feasible ($< 100\times$)')
ax.axhspan(100, 1e4, alpha=0.1, color='#ff9800', label=r'Expensive ($100$--$10{,}000\times$)')
ax.axhspan(1e4, 1e8, alpha=0.1, color='#e57373', label=r'Prohibitive ($> 10{,}000\times$)')
ax.legend(fontsize=11, loc='upper left')

plt.tight_layout()
plt.savefig('cost_to_improve.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Simulation 4: Kaplan vs. Chinchilla Allocation

```python
import numpy as np
import matplotlib.pyplot as plt

# Compare how Kaplan and Chinchilla allocate compute
C_range = np.logspace(18, 26, 100)

# Kaplan allocation
N_kaplan = 1e-3 * C_range**0.73  # Proportionality constants chosen for
D_kaplan = C_range / (6 * N_kaplan)  # reasonable absolute values

# Chinchilla allocation
N_chinchilla = 0.02 * C_range**0.50
D_chinchilla = C_range / (6 * N_chinchilla)

# Token-to-parameter ratio
ratio_kaplan = D_kaplan / N_kaplan
ratio_chinchilla = D_chinchilla / N_chinchilla

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: N and D vs C
axes[0].loglog(C_range, N_kaplan, '--', color='#4fc3f7', linewidth=2,
               label=r'$N$ (Kaplan)')
axes[0].loglog(C_range, D_kaplan, '--', color='#e57373', linewidth=2,
               label=r'$D$ (Kaplan)')
axes[0].loglog(C_range, N_chinchilla, '-', color='#4fc3f7', linewidth=2.5,
               label=r'$N$ (Chinchilla)')
axes[0].loglog(C_range, D_chinchilla, '-', color='#e57373', linewidth=2.5,
               label=r'$D$ (Chinchilla)')
axes[0].set_xlabel(r'Compute Budget $C$ (FLOPs)', fontsize=13)
axes[0].set_ylabel(r'Optimal $N$ or $D$', fontsize=13)
axes[0].set_title(r'Compute Allocation: Kaplan vs. Chinchilla', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, which='both')

# Panel 2: Token-to-parameter ratio
axes[1].semilogx(C_range, ratio_kaplan, '--', color='#ab47bc', linewidth=2,
                 label=r'$D/N$ (Kaplan)')
axes[1].semilogx(C_range, ratio_chinchilla, '-', color='#ab47bc',
                 linewidth=2.5, label=r'$D/N$ (Chinchilla)')
axes[1].axhline(y=20, color='#66bb6a', linewidth=1.5, linestyle=':',
                label=r'Chinchilla rule: $D/N \approx 20$')
axes[1].set_xlabel(r'Compute Budget $C$ (FLOPs)', fontsize=13)
axes[1].set_ylabel(r'Tokens per Parameter ($D/N$)', fontsize=13)
axes[1].set_title(r'Tokens per Parameter at Optimal Allocation', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, which='both')
axes[1].set_ylim(0, 100)

plt.tight_layout()
plt.savefig('kaplan_vs_chinchilla.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Conclusion

The scaling laws tell a story that is simultaneously empowering and sobering.

**Empowering:** we have a predictive theory. Before training a model that costs $100 million, you can train a small version for $1,000 and extrapolate. The power law lets you predict the loss of a 100B-parameter model from the losses of 1M, 10M, and 100M-parameter models. This is extraordinary --- it turns an empirical science into something closer to engineering.

**Sobering:** the exponents are small. $\alpha_C \approx 0.05$ means that the game of "just make it bigger" hits diminishing returns brutally fast. A $10\times$ increase in compute buys you only a $\sim 12\%$ reduction in reducible loss. The era of easy gains from scaling is approaching its limits, at least along the pure training-compute axis.

The field is responding in several ways:

1. **Better data**, not just more data. Careful curation, deduplication, and quality filtering can effectively shift the scaling curve --- getting the same loss with less compute.

2. **Inference-time scaling.** Instead of making the base model bigger, use more compute at inference time through chain-of-thought reasoning, search, and verification.

3. **Overtraining for deployment.** Train smaller models past the Chinchilla-optimal point, accepting a slight loss penalty for dramatically cheaper inference.

4. **Architectural improvements.** While the scaling exponents are universal, the constant factors are not. Better architectures (mixture of experts, state-space models, etc.) can shift the curve downward --- achieving the same loss at lower compute.

The power law is not a wall. It is a landscape. Understanding its shape --- the exponents, the optimal allocations, the economic tradeoffs --- is what separates informed scaling decisions from expensive guesswork.

The mathematics laid out here --- from the Lagrange multiplier derivation of the Chinchilla frontier to the cost-to-improve formula --- gives you the tools to reason quantitatively about these tradeoffs. Whether you are deciding how much to spend on training, how big to make your model, or whether to invest in inference-time compute instead, the scaling laws provide the framework.

The future of AI is not just about making things bigger. It is about making things bigger *wisely*.
