---
layout: post
title: "Stochastic Differential Equations: The Mathematical Engine Behind Diffusion Models"
date: 2026-01-13
category: math
---

Diffusion models generate images by reversing a noise process. They start with pure Gaussian noise and iteratively denoise it until a coherent image emerges. The mathematical framework that makes this possible is **stochastic calculus** --- the calculus of random processes. Specifically, the forward noising process and the reverse denoising process are both described by **stochastic differential equations (SDEs)**, and the connection between them is one of the most beautiful results in probability theory.

This post builds stochastic calculus from the ground up. We start with Brownian motion (the random walk that underlies everything), develop the Ito integral (because the usual calculus integral fails for random processes), derive Ito's lemma (the stochastic chain rule, where a term that vanishes in ordinary calculus suddenly matters), solve the Ornstein-Uhlenbeck process explicitly, and then connect the entire framework to diffusion models --- forward noising, reverse denoising, score matching, and the DDPM equations.

---

## Table of Contents

1. [Brownian Motion: The Wiener Process](#brownian-motion-the-wiener-process)
2. [Why Ordinary Calculus Fails](#why-ordinary-calculus-fails)
3. [The Ito Integral](#the-ito-integral)
4. [Ito's Lemma: The Stochastic Chain Rule](#itos-lemma-the-stochastic-chain-rule)
5. [Stochastic Differential Equations](#stochastic-differential-equations)
6. [The Fokker-Planck Equation](#the-fokker-planck-equation)
7. [The Ornstein-Uhlenbeck Process](#the-ornstein-uhlenbeck-process)
8. [The Forward Diffusion Process](#the-forward-diffusion-process)
9. [The Reverse-Time SDE](#the-reverse-time-sde)
10. [Score Matching](#score-matching)
11. [The DDPM Connection](#the-ddpm-connection)
12. [Python: Simulating Stochastic Processes](#python-simulating-stochastic-processes)

---

## Brownian Motion: The Wiener Process

**Brownian motion** --- named after the botanist Robert Brown, who observed the erratic motion of pollen grains suspended in water --- is the fundamental random process from which all of stochastic calculus is built. Mathematically, it is called the **Wiener process** and denoted \(W_t\) or \(B_t\).

A standard Wiener process \(W_t\) (for \(t \geq 0\)) satisfies four properties:

1. **\(W_0 = 0\)** --- it starts at the origin.

2. **Independent increments.** For any \(0 \leq s < t \leq u < v\), the increments \(W_t - W_s\) and \(W_v - W_u\) are independent random variables. What happens in one time interval tells you nothing about what happens in a non-overlapping interval.

3. **Gaussian increments.** \(W_t - W_s \sim \mathcal{N}(0, t - s)\). Each increment is normally distributed with mean zero and variance equal to the length of the time interval. Longer intervals have more variance.

4. **Continuous paths.** \(W_t\) is a continuous function of \(t\) with probability 1.

From these properties, several remarkable consequences follow:

**\(W_t \sim \mathcal{N}(0, t)\).** The process at time \(t\) is Gaussian with variance \(t\). It spreads out as \(\sqrt{t}\) --- not \(t\). The standard deviation grows as the square root of time, which is why diffusion is slow (the particles in Brown's pollen experiment meander rather than march).

**\(\text{Cov}(W_s, W_t) = \min(s, t)\).** The covariance between the process at two times equals the smaller time. This follows from writing \(W_t = W_s + (W_t - W_s)\) for \(s < t\) and using independence of increments.

**Continuous but nowhere differentiable.** This is the most striking property. Brownian paths are continuous --- they never jump --- but they are so jagged that they have no derivative at any point. Formally, the limit \(\lim_{h \to 0} (W_{t+h} - W_t)/h\) does not exist because \(W_{t+h} - W_t \sim \mathcal{N}(0, h)\), so the ratio is \(\mathcal{N}(0, 1/h)\), whose variance diverges as \(h \to 0\).

This nowhere-differentiability is not a mathematical pathology --- it is the fundamental reason why stochastic calculus must be rebuilt from scratch rather than adapted from ordinary calculus.

<svg viewBox="0 0 700 350" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <defs>
    <marker id="arrowBM" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#d4d4d4"/>
    </marker>
  </defs>
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">Brownian Motion: Key Properties</text>
  <!-- Axes -->
  <line x1="60" y1="180" x2="660" y2="180" stroke="#d4d4d4" stroke-width="1" marker-end="url(#arrowBM)"/>
  <line x1="60" y1="320" x2="60" y2="40" stroke="#d4d4d4" stroke-width="1" marker-end="url(#arrowBM)"/>
  <text x="660" y="200" text-anchor="end" font-size="12" fill="#d4d4d4">t</text>
  <text x="40" y="45" text-anchor="middle" font-size="12" fill="#d4d4d4">W(t)</text>
  <!-- Sample path (jagged) -->
  <polyline points="60,180 80,175 95,185 110,170 125,180 140,165 155,175 170,155 185,160 200,140 215,150 230,135 245,145 260,130 275,140 290,125 305,135 320,120 335,130 350,115 365,125 380,110 395,120 410,100 425,115 440,95 455,105 470,90 485,100 500,85 515,95 530,80 545,95 560,75 575,90 590,70 605,85 620,65 635,80" fill="none" stroke="#2196F3" stroke-width="1.8"/>
  <!-- Standard deviation envelope -->
  <path d="M 60,180 Q 200,140 350,120 Q 500,100 640,85" fill="none" stroke="#F44336" stroke-width="1" stroke-dasharray="6,3"/>
  <path d="M 60,180 Q 200,220 350,240 Q 500,260 640,275" fill="none" stroke="#F44336" stroke-width="1" stroke-dasharray="6,3"/>
  <text x="645" y="82" font-size="10" fill="#F44336">+√t</text>
  <text x="645" y="278" font-size="10" fill="#F44336">-√t</text>
  <!-- Labels -->
  <text x="350" y="310" text-anchor="middle" font-size="11" fill="#999">Continuous but nowhere differentiable. Std. dev. grows as √t.</text>
  <text x="350" y="330" text-anchor="middle" font-size="11" fill="#999">Increments W(t)-W(s) ~ N(0, t-s) are independent for non-overlapping intervals.</text>
</svg>

---

## Why Ordinary Calculus Fails

In ordinary calculus, the chain rule says that if \(f\) is a smooth function and \(x(t)\) is a differentiable function of time, then:

$$\frac{d}{dt} f(x(t)) = f'(x(t)) \cdot x'(t)$$

What happens if we replace \(x(t)\) with the Wiener process \(W_t\)? Since \(W_t\) is not differentiable, \(W_t'\) does not exist, and the chain rule as stated is meaningless.

But the problem runs deeper than just non-differentiability. Consider trying to define the integral \(\int_0^T g(t) \, dW_t\). In the Riemann-Stieltjes approach, we would write:

$$\int_0^T g(t) \, dW_t = \lim_{n \to \infty} \sum_{i=0}^{n-1} g(\tau_i) \left(W_{t_{i+1}} - W_{t_i}\right)$$

where \(\tau_i\) is some point in the interval \([t_i, t_{i+1}]\). For ordinary (bounded variation) integrands, this limit exists and does not depend on the choice of \(\tau_i\) within each subinterval.

For Brownian motion, **the limit depends on where you evaluate \(g\) in each subinterval**. Choosing \(\tau_i = t_i\) (the left endpoint) gives one answer. Choosing \(\tau_i = (t_i + t_{i+1})/2\) (the midpoint) gives a different answer. This is not a minor technical issue --- the two choices differ by a term of order 1, not order \(\Delta t\).

The reason is the **quadratic variation** of Brownian motion. For a smooth function \(x(t)\), the quadratic variation:

$$[x]_T = \lim_{n \to \infty} \sum_{i=0}^{n-1} \left(x(t_{i+1}) - x(t_i)\right)^2$$

is zero (because each term is \(O(\Delta t^2)\) and there are \(O(1/\Delta t)\) terms, so the sum is \(O(\Delta t) \to 0\)). But for Brownian motion:

$$[W]_T = \lim_{n \to \infty} \sum_{i=0}^{n-1} \left(W_{t_{i+1}} - W_{t_i}\right)^2 = T$$

Each term \((W_{t_{i+1}} - W_{t_i})^2\) has expectation \(\Delta t\) and variance \(2(\Delta t)^2\). There are \(n = T/\Delta t\) terms. By the law of large numbers, the sum converges to \(T\). The quadratic variation is non-zero and finite --- it equals the time elapsed.

This is the fundamental fact that distinguishes stochastic calculus: **\((dW_t)^2 = dt\)**, not zero. In ordinary calculus, \((dx)^2\) is infinitesimally small compared to \(dx\) and can be ignored. In stochastic calculus, \((dW_t)^2\) is of the same order as \(dt\) and cannot be dropped. This single fact explains why Ito's lemma has an extra term.

---

## The Ito Integral

The **Ito integral** resolves the ambiguity by fixing the convention: always evaluate the integrand at the **left endpoint** of each subinterval.

$$\int_0^T g(t) \, dW_t = \lim_{n \to \infty} \sum_{i=0}^{n-1} g(t_i) \left(W_{t_{i+1}} - W_{t_i}\right)$$

This is not the only possible choice. The **Stratonovich integral** uses the midpoint:

$$\int_0^T g(t) \circ dW_t = \lim_{n \to \infty} \sum_{i=0}^{n-1} g\left(\frac{t_i + t_{i+1}}{2}\right) \left(W_{t_{i+1}} - W_{t_i}\right)$$

Both conventions are mathematically consistent. The Ito convention is more natural for probability theory and for applications to finance and machine learning (because the integrand does not "peek into the future" --- it uses only information available at time \(t_i\)). The Stratonovich convention obeys the ordinary chain rule, making it more natural for physics.

The key properties of the Ito integral \(I = \int_0^T g(t) \, dW_t\):

1. **Zero mean.** \(\mathbb{E}[I] = 0\). This follows from the independence of \(g(t_i)\) and \(W_{t_{i+1}} - W_{t_i}\) in the Ito convention (the integrand depends only on past information, not the current increment).

2. **Ito isometry.** \(\mathbb{E}[I^2] = \int_0^T \mathbb{E}[g(t)^2] \, dt\). The variance of the Ito integral equals the time integral of the squared integrand. This is the analog of Parseval's theorem.

3. **Martingale property.** The Ito integral defines a **martingale** --- a process whose expected future value, given all information up to now, equals its current value: \(\mathbb{E}[I_t \mid \mathcal{F}_s] = I_s\) for \(s < t\).

---

## Ito's Lemma: The Stochastic Chain Rule

**Ito's lemma** is the chain rule for stochastic calculus. It tells us how a smooth function of a stochastic process evolves. It is the single most important tool in the field.

Let \(X_t\) be a stochastic process satisfying:

$$dX_t = \mu(X_t, t) \, dt + \sigma(X_t, t) \, dW_t$$

where \(\mu\) is the **drift** (deterministic tendency) and \(\sigma\) is the **diffusion coefficient** (noise intensity). Let \(f(x, t)\) be a twice-differentiable function. Then \(Y_t = f(X_t, t)\) satisfies:

$$df = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} dX_t + \frac{1}{2} \frac{\partial^2 f}{\partial x^2} (dX_t)^2$$

Now we use the multiplication rules for stochastic differentials:

$$(dt)^2 = 0, \quad dt \cdot dW_t = 0, \quad (dW_t)^2 = dt$$

So \((dX_t)^2 = (\mu \, dt + \sigma \, dW_t)^2 = \sigma^2 (dW_t)^2 = \sigma^2 \, dt\) (dropping the higher-order terms \(\mu^2 (dt)^2\) and \(\mu\sigma \, dt \, dW_t\)).

Substituting:

$$\boxed{df = \left(\frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{1}{2}\sigma^2 \frac{\partial^2 f}{\partial x^2}\right) dt + \sigma \frac{\partial f}{\partial x} dW_t}$$

This is **Ito's lemma**. Compare with the ordinary chain rule, which would give only the first two terms in the \(dt\) coefficient. The extra term \(\frac{1}{2}\sigma^2 \frac{\partial^2 f}{\partial x^2}\) exists because \((dW_t)^2 = dt\) rather than zero. This term is sometimes called the **Ito correction** or **noise-induced drift**.

### Example: Computing \(d(W_t^2)\)

Let \(f(x) = x^2\) and \(X_t = W_t\) (so \(\mu = 0\), \(\sigma = 1\)). Then:

$$d(W_t^2) = \frac{\partial (x^2)}{\partial x} \bigg|_{x=W_t} dW_t + \frac{1}{2} \frac{\partial^2 (x^2)}{\partial x^2} \bigg|_{x=W_t} dt = 2W_t \, dW_t + dt$$

In integral form: \(W_T^2 = 2\int_0^T W_t \, dW_t + T\).

The ordinary chain rule would give \(d(W_t^2) = 2W_t \, dW_t\). The extra \(dt\) term is the Ito correction. Rearranging: \(\int_0^T W_t \, dW_t = \frac{1}{2}(W_T^2 - T)\). This is the stochastic analog of \(\int x \, dx = x^2/2\), with a correction term \(-T/2\).

---

## Stochastic Differential Equations

A **stochastic differential equation (SDE)** describes how a random process evolves:

$$dX_t = \mu(X_t, t) \, dt + \sigma(X_t, t) \, dW_t$$

The drift \(\mu\) determines the deterministic tendency (where the process wants to go on average), and the diffusion \(\sigma\) determines the noise intensity (how much it is jostled at each instant).

The canonical example is the **Langevin equation**, which describes the velocity of a particle subject to friction and random kicks:

$$dv_t = -\gamma v_t \, dt + \sigma \, dW_t$$

The first term \(-\gamma v_t\) is friction (the velocity is pulled toward zero). The second term \(\sigma \, dW_t\) represents random thermal kicks. The balance between friction and noise determines the equilibrium behavior.

SDEs are solved in a distributional sense. A solution is a stochastic process \(X_t\) that satisfies the integral equation:

$$X_t = X_0 + \int_0^t \mu(X_s, s) \, ds + \int_0^t \sigma(X_s, s) \, dW_s$$

where the first integral is an ordinary Riemann integral and the second is an Ito integral.

Existence and uniqueness of solutions is guaranteed under Lipschitz and linear growth conditions on \(\mu\) and \(\sigma\) --- the stochastic analog of the Picard-Lindelof theorem for ODEs.

---

## The Fokker-Planck Equation

While an SDE describes the trajectory of a single random particle, the **Fokker-Planck equation** (also called the Kolmogorov forward equation) describes the evolution of the probability density \(p(x, t)\) of the entire ensemble.

If \(X_t\) satisfies \(dX_t = \mu(x, t) \, dt + \sigma(x, t) \, dW_t\), then \(p(x, t)\) satisfies:

$$\frac{\partial p}{\partial t} = -\frac{\partial}{\partial x}\left[\mu(x, t) \, p(x, t)\right] + \frac{1}{2}\frac{\partial^2}{\partial x^2}\left[\sigma^2(x, t) \, p(x, t)\right]$$

Let us derive this using Ito's lemma. We want to compute \(\mathbb{E}[f(X_t)]\) for an arbitrary smooth test function \(f\). By Ito's lemma:

$$df(X_t) = \left(\mu f' + \frac{1}{2}\sigma^2 f''\right) dt + \sigma f' \, dW_t$$

Taking expectations (the Ito integral has zero mean):

$$\frac{d}{dt}\mathbb{E}[f(X_t)] = \mathbb{E}\left[\mu f'(X_t) + \frac{1}{2}\sigma^2 f''(X_t)\right]$$

The left side is \(\frac{d}{dt}\int f(x) p(x,t) \, dx = \int f(x) \frac{\partial p}{\partial t} dx\). The right side is \(\int \left(\mu f' + \frac{1}{2}\sigma^2 f''\right) p \, dx\). Integrating by parts (assuming \(p\) and its derivatives vanish at infinity):

$$\int \mu f' p \, dx = -\int f \frac{\partial}{\partial x}(\mu p) \, dx$$

$$\int \frac{1}{2}\sigma^2 f'' p \, dx = \int f \frac{1}{2}\frac{\partial^2}{\partial x^2}(\sigma^2 p) \, dx$$

Since this holds for all test functions \(f\), the integrands must be equal:

$$\frac{\partial p}{\partial t} = -\frac{\partial}{\partial x}(\mu p) + \frac{1}{2}\frac{\partial^2}{\partial x^2}(\sigma^2 p)$$

This is the Fokker-Planck equation. It converts a stochastic problem (tracking random trajectories) into a deterministic PDE (tracking a probability density). The first term on the right is the **drift** or **advection** term --- it transports the probability. The second term is the **diffusion** term --- it spreads the probability out.

For diffusion models, the Fokker-Planck equation tells us how the data distribution evolves under the forward noising process. Starting from the data distribution \(p_{\text{data}}(x)\) at \(t=0\), the forward SDE gradually transforms it into a Gaussian at \(t=T\).

---

## The Ornstein-Uhlenbeck Process

The **Ornstein-Uhlenbeck (OU) process** is the simplest SDE with a non-trivial stationary distribution. It is defined by:

$$dX_t = -\theta X_t \, dt + \sigma \, dW_t$$

where \(\theta > 0\) is the **mean-reversion rate** and \(\sigma > 0\) is the noise intensity. The drift \(-\theta X_t\) pulls the process toward zero (mean reversion), while the noise \(\sigma \, dW_t\) pushes it away.

This SDE can be solved exactly. Multiply both sides by the integrating factor \(e^{\theta t}\):

$$d(e^{\theta t} X_t) = e^{\theta t} dX_t + \theta e^{\theta t} X_t \, dt = e^{\theta t} \sigma \, dW_t$$

(using the product rule for stochastic differentials, where the Ito correction vanishes because \(e^{\theta t}\) is deterministic). Integrating from $0$ to \(t\):

$$e^{\theta t} X_t - X_0 = \sigma \int_0^t e^{\theta s} \, dW_s$$

$$X_t = X_0 e^{-\theta t} + \sigma \int_0^t e^{-\theta(t-s)} \, dW_s$$

The first term \(X_0 e^{-\theta t}\) is the deterministic decay of the initial condition. The second term is a stochastic integral that accumulates noise over the history.

Since the stochastic integral of a deterministic function against \(dW_s\) is Gaussian, \(X_t\) is Gaussian at every time:

$$X_t \sim \mathcal{N}\left(X_0 e^{-\theta t}, \, \frac{\sigma^2}{2\theta}(1 - e^{-2\theta t})\right)$$

As \(t \to \infty\), the mean decays to zero and the variance converges to:

$$\text{Var}(X_\infty) = \frac{\sigma^2}{2\theta}$$

The **stationary distribution** is \(\mathcal{N}\left(0, \frac{\sigma^2}{2\theta}\right)\). Regardless of where the process starts, it always converges to this distribution. The mean-reversion parameter \(\theta\) controls how fast it forgets its initial condition, and the balance \(\sigma^2/(2\theta)\) between noise intensity and reversion determines the spread of the equilibrium.

The OU process is the prototype for the forward diffusion process in diffusion models: start from data, apply a mean-reverting SDE, and converge to a known Gaussian. The only difference is that diffusion models use time-dependent coefficients.

---

## The Forward Diffusion Process

In a diffusion model, the **forward process** gradually destroys data by adding noise. Starting from a data sample \(x_0 \sim p_{\text{data}}\), the forward SDE is:

$$dx_t = f(x_t, t) \, dt + g(t) \, dW_t$$

where \(f\) is the drift and \(g\) is the diffusion coefficient. Two standard formulations are used:

### Variance-Preserving (VP) SDE

$$dx_t = -\frac{1}{2}\beta(t) x_t \, dt + \sqrt{\beta(t)} \, dW_t$$

Here \(\beta(t) > 0\) is a noise schedule (typically increasing with \(t\)). The drift \(-\frac{1}{2}\beta(t) x_t\) is mean-reverting, and the diffusion \(\sqrt{\beta(t)}\) adds noise. The name "variance-preserving" comes from the fact that if \(x_0\) has unit variance and the schedule is chosen correctly, the marginal variance \(\text{Var}(x_t)\) stays close to 1 for all \(t\).

The transition kernel (conditional distribution of \(x_t\) given \(x_0\)) can be computed in closed form:

$$p(x_t \mid x_0) = \mathcal{N}\left(x_t; \, \sqrt{\bar{\alpha}_t} \, x_0, \, (1 - \bar{\alpha}_t) I\right)$$

where \(\bar{\alpha}_t = e^{-\int_0^t \beta(s) \, ds}\). This means we can sample \(x_t\) directly without simulating the SDE step by step:

$$x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

At \(t = 0\): \(x_0\) is the original data. As \(t\) increases, \(\bar{\alpha}_t\) decreases toward zero, the signal \(\sqrt{\bar{\alpha}_t} x_0\) fades, and the noise \(\sqrt{1 - \bar{\alpha}_t} \epsilon\) dominates. At \(t \to \infty\): \(x_t \sim \mathcal{N}(0, I)\), pure noise.

### Variance-Exploding (VE) SDE

$$dx_t = \sqrt{\frac{d[\sigma^2(t)]}{dt}} \, dW_t$$

No drift, only noise. The noise level \(\sigma(t)\) increases with time. The transition kernel is:

$$p(x_t \mid x_0) = \mathcal{N}\left(x_t; \, x_0, \, \sigma^2(t) I\right)$$

The signal is not attenuated --- it is drowned out by increasing noise. The variance of \(x_t\) grows without bound (hence "variance-exploding").

Both formulations achieve the same goal: transform the data distribution into a simple Gaussian. They differ in the path they take through distribution space. VP is more numerically stable and is the formulation underlying DDPM.

---

## The Reverse-Time SDE

The key theoretical result that makes diffusion models possible is **Anderson's theorem** (1982): **every forward-time SDE has a corresponding reverse-time SDE**.

If the forward process is:

$$dx_t = f(x_t, t) \, dt + g(t) \, dW_t$$

then the reverse process, running from time \(T\) back to $0$, is:

$$dx_t = \left[f(x_t, t) - g(t)^2 \nabla_x \log p_t(x_t)\right] dt + g(t) \, d\bar{W}_t$$

where \(\bar{W}_t\) is a reverse-time Wiener process and \(\nabla_x \log p_t(x)\) is the **score function** --- the gradient of the log-probability density at time \(t\).

Let us unpack what this equation says. The reverse drift has two components:

1. **\(f(x_t, t)\)**: the forward drift, unchanged. If the forward process was pulling the data toward zero, the reverse process has this same pull.

2. **\(-g(t)^2 \nabla_x \log p_t(x_t)\)**: the score correction. This is the term that guides the reverse process. The score \(\nabla_x \log p_t(x)\) points in the direction of increasing probability density. Subtracting \(g^2\) times the score from the drift means the reverse process moves toward regions of higher probability --- it denoises.

The mathematical derivation uses the Fokker-Planck equation for the forward process and the fact that the time-reversed probability density satisfies its own Fokker-Planck equation. Working out the drift of the reversed Fokker-Planck equation and using the identity:

$$\frac{\nabla_x p_t(x)}{p_t(x)} = \nabla_x \log p_t(x)$$

yields the reverse-time SDE above.

The practical implication is profound: **if we know the score function \(\nabla_x \log p_t(x)\) for all \(x\) and \(t\), we can run the reverse SDE and generate samples from the data distribution starting from pure noise**.

---

## Score Matching

The score function \(\nabla_x \log p_t(x)\) is not known in closed form --- it depends on the data distribution \(p_{\text{data}}\), which is what we are trying to model. So we train a neural network \(s_\theta(x, t)\) to approximate it.

**Naive score matching** would minimize:

$$\mathcal{L} = \mathbb{E}_{t, x_t}\left[\left\|s_\theta(x_t, t) - \nabla_{x_t} \log p_t(x_t)\right\|^2\right]$$

But \(\nabla_{x_t} \log p_t(x_t)\) is unknown --- that is the whole point. The trick is **denoising score matching** (Vincent, 2011). Since we know the transition kernel \(p(x_t \mid x_0)\) in closed form, we can write:

$$p_t(x_t) = \int p(x_t \mid x_0) p_{\text{data}}(x_0) \, dx_0$$

and it can be shown that minimizing the denoising score matching objective:

$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_{t, x_0, x_t}\left[\left\|s_\theta(x_t, t) - \nabla_{x_t} \log p(x_t \mid x_0)\right\|^2\right]$$

is equivalent to minimizing the original objective (up to a constant that does not depend on \(\theta\)).

For the VP-SDE, \(p(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)I)\), so:

$$\nabla_{x_t} \log p(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{1 - \bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$

where \(\epsilon = (x_t - \sqrt{\bar{\alpha}_t} x_0) / \sqrt{1 - \bar{\alpha}_t}\) is the noise that was added. So the denoising score matching loss becomes:

$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_{t, x_0, \epsilon}\left[\left\|s_\theta(x_t, t) + \frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}\right\|^2\right]$$

If we reparameterize \(s_\theta(x_t, t) = -\epsilon_\theta(x_t, t) / \sqrt{1 - \bar{\alpha}_t}\), the loss simplifies to:

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[\left\|\epsilon_\theta(x_t, t) - \epsilon\right\|^2\right]$$

This is the **DDPM training objective**: predict the noise \(\epsilon\) that was added to the clean data \(x_0\) to produce the noisy data \(x_t\). The noise prediction network \(\epsilon_\theta\) is equivalent to a score network (up to a known scaling factor).

---

## The DDPM Connection

The **Denoising Diffusion Probabilistic Model** (DDPM, Ho et al. 2020) is a discrete-time version of the VP-SDE framework. It uses a discrete noise schedule \(\beta_1, \beta_2, \ldots, \beta_T\) and defines:

**Forward process:** \(q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \, \beta_t I)\)

**Cumulative forward:** \(q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, \, (1 - \bar{\alpha}_t) I)\), where \(\alpha_t = 1 - \beta_t\) and \(\bar{\alpha}_t = \prod_{s=1}^t \alpha_s\).

**Training:** minimize \(\mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon_\theta(x_t, t) - \epsilon\|^2\right]\) where \(x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon\).

**Sampling:** the reverse process is:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sqrt{\beta_t} \, z$$

where \(z \sim \mathcal{N}(0, I)\) for \(t > 1\) and \(z = 0\) for \(t = 1\).

This sampling formula is the **Euler-Maruyama discretization** of the reverse-time SDE. To see this, start from the reverse SDE with VP drift:

$$dx_t = \left[-\frac{1}{2}\beta(t) x_t - \beta(t) \nabla_x \log p_t(x_t)\right] dt + \sqrt{\beta(t)} \, d\bar{W}_t$$

Substitute \(\nabla_x \log p_t(x_t) \approx -\epsilon_\theta(x_t, t)/\sqrt{1 - \bar{\alpha}_t}\) and discretize with step \(\Delta t = 1\):

$$x_{t-1} = x_t + \left[-\frac{1}{2}\beta_t x_t + \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right] + \sqrt{\beta_t} \, z$$

For small \(\beta_t\), \(1 - \frac{1}{2}\beta_t \approx 1/\sqrt{\alpha_t}\), and simplifying yields the DDPM sampling formula. The \(T = 1000\) steps in the original DDPM paper correspond to 1000 Euler-Maruyama steps of the reverse SDE.

The entire pipeline is now clear: the forward SDE destroys data by adding noise, the Fokker-Planck equation governs how the probability density evolves, the reverse-time SDE shows how to undo the process, score matching trains a network to approximate the score function needed by the reverse SDE, and Euler-Maruyama discretization turns the continuous reverse SDE into the discrete DDPM sampling formula.

---

## Python: Simulating Stochastic Processes

### Brownian Motion Paths

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulate Brownian motion paths
T = 2.0       # Total time
N = 10000     # Number of time steps
dt = T / N
t = np.linspace(0, T, N+1)
n_paths = 5

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Multiple sample paths
for i in range(n_paths):
    dW = np.sqrt(dt) * np.random.randn(N)
    W = np.concatenate([[0], np.cumsum(dW)])
    axes[0, 0].plot(t, W, linewidth=0.7, alpha=0.8)

# Standard deviation envelope
axes[0, 0].fill_between(t, -2*np.sqrt(t), 2*np.sqrt(t), alpha=0.1, color='red')
axes[0, 0].plot(t, np.sqrt(t), 'r--', linewidth=1, label=r'$\pm\sqrt{t}$ (std dev)')
axes[0, 0].plot(t, -np.sqrt(t), 'r--', linewidth=1)
axes[0, 0].set_xlabel(r'Time $t$')
axes[0, 0].set_ylabel(r'$W(t)$')
axes[0, 0].set_title(r'Brownian Motion: Sample Paths')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Distribution at fixed times
dW_all = np.sqrt(dt) * np.random.randn(5000, N)
W_all = np.concatenate([np.zeros((5000, 1)), np.cumsum(dW_all, axis=1)], axis=1)

time_indices = [N//4, N//2, 3*N//4, N]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
for idx, ti in enumerate(time_indices):
    time_val = t[ti]
    samples = W_all[:, ti]
    axes[0, 1].hist(samples, bins=60, density=True, alpha=0.4, color=colors[idx],
                     label=rf'$t = {time_val:.1f}$')
    # Theoretical density
    x_range = np.linspace(-4, 4, 200)
    theoretical = np.exp(-x_range**2 / (2*time_val)) / np.sqrt(2*np.pi*time_val)
    axes[0, 1].plot(x_range, theoretical, color=colors[idx], linewidth=2)

axes[0, 1].set_xlabel(r'$W(t)$')
axes[0, 1].set_ylabel(r'Probability density')
axes[0, 1].set_title(r'Distribution of $W(t) \sim \mathcal{N}(0, t)$')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Panel 3: Quadratic variation convergence
partitions = [10, 50, 100, 500, 1000, 5000]
dW_one = np.sqrt(dt) * np.random.randn(N)
W_one = np.concatenate([[0], np.cumsum(dW_one)])

qv_estimates = []
for n_part in partitions:
    step = N // n_part
    indices = np.arange(0, N+1, step)
    W_sampled = W_one[indices]
    qv = np.sum(np.diff(W_sampled)**2)
    qv_estimates.append(qv)

axes[1, 0].semilogx(partitions, qv_estimates, 'o-', color='#9C27B0', linewidth=2, markersize=8)
axes[1, 0].axhline(y=T, color='red', linestyle='--', linewidth=1.5, label=rf'$[W]_T = T = {T}$')
axes[1, 0].set_xlabel(r'Number of partition points')
axes[1, 0].set_ylabel(r'Quadratic variation estimate')
axes[1, 0].set_title(r'Quadratic Variation: $\sum (\Delta W)^2 \to T$')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: Nowhere differentiability (zoom into path)
axes[1, 1].plot(t[:500], W_one[:500], linewidth=0.8, color='#2196F3')
axes[1, 1].set_xlabel(r'Time $t$')
axes[1, 1].set_ylabel(r'$W(t)$')
axes[1, 1].set_title(r'Brownian Motion Close-Up: Jagged at Every Scale')
axes[1, 1].grid(True, alpha=0.3)

# Inset: zoom further
inset = axes[1, 1].inset_axes([0.55, 0.55, 0.4, 0.4])
inset.plot(t[:50], W_one[:50], linewidth=0.8, color='#F44336')
inset.set_title('Further zoom', fontsize=9)
inset.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('brownian_motion.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Forward Diffusion: Progressive Noising

```python
import numpy as np
import matplotlib.pyplot as plt

def forward_diffusion_1d(x0_samples, T=100, beta_min=0.0001, beta_max=0.02):
    """
    Simulate the forward VP diffusion process on 1D data.

    x0_samples: initial data samples
    Returns snapshots of the evolving distribution.
    """
    betas = np.linspace(beta_min, beta_max, T)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)

    snapshots = [x0_samples.copy()]
    snapshot_times = [0]

    for t_idx in [T//10, T//4, T//2, 3*T//4, T-1]:
        ab = alpha_bars[t_idx]
        # Direct sampling: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * eps
        eps = np.random.randn(len(x0_samples))
        x_t = np.sqrt(ab) * x0_samples + np.sqrt(1 - ab) * eps
        snapshots.append(x_t)
        snapshot_times.append(t_idx + 1)

    return snapshots, snapshot_times, alpha_bars

# Create bimodal data distribution
np.random.seed(42)
n_samples = 10000
x0 = np.concatenate([
    np.random.randn(n_samples // 2) * 0.5 + 3,   # mode at x=3
    np.random.randn(n_samples // 2) * 0.5 - 3     # mode at x=-3
])

snapshots, times, alpha_bars = forward_diffusion_1d(x0, T=200)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

titles = [r'$t = 0$' + '\n(Data)', rf'$t = {times[1]}$' + '\n(Light noise)',
          rf'$t = {times[2]}$' + '\n(Moderate)', rf'$t = {times[3]}$' + '\n(Heavy)',
          rf'$t = {times[4]}$' + '\n(Nearly destroyed)', rf'$t = {times[5]}$' + '\n(Pure noise)']

for i, (snap, title) in enumerate(zip(snapshots, titles)):
    axes[i].hist(snap, bins=80, density=True, alpha=0.7, color='#2196F3', edgecolor='none')
    axes[i].set_xlim(-8, 8)
    axes[i].set_ylim(0, 0.6)
    axes[i].set_title(title, fontsize=11)
    axes[i].grid(True, alpha=0.2)
    if i >= 3:
        axes[i].set_xlabel(r'$x$')

    # Overlay Gaussian for reference
    x_plot = np.linspace(-8, 8, 200)
    gaussian = np.exp(-x_plot**2/2) / np.sqrt(2*np.pi)
    axes[i].plot(x_plot, gaussian, 'r--', alpha=0.5, linewidth=1, label=r'$\mathcal{N}(0,1)$')
    if i == 0:
        axes[i].legend(fontsize=9)

plt.suptitle(r'Forward Diffusion Process: Data $\to$ Noise', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('forward_diffusion.png', dpi=150, bbox_inches='tight')
plt.show()

# Signal-to-noise ratio over time
fig, ax = plt.subplots(figsize=(10, 5))
t_range = np.arange(len(alpha_bars))
signal = np.sqrt(alpha_bars)
noise = np.sqrt(1 - alpha_bars)
snr = signal / noise

ax.plot(t_range, signal, label=r'Signal: $\sqrt{\bar{\alpha}_t}$', color='#2196F3', linewidth=2)
ax.plot(t_range, noise, label=r'Noise: $\sqrt{1-\bar{\alpha}_t}$', color='#F44336', linewidth=2)
ax.plot(t_range, snr, label=r'SNR', color='#4CAF50', linewidth=2, linestyle='--')
ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel(r'Diffusion step $t$', fontsize=12)
ax.set_ylabel(r'Coefficient', fontsize=12)
ax.set_title(r'Signal vs Noise in the Forward Diffusion Process', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('diffusion_snr.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Ornstein-Uhlenbeck Process

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_ou(x0, theta, sigma, T, dt, n_paths=1):
    """Simulate Ornstein-Uhlenbeck process: dX = -theta*X*dt + sigma*dW."""
    N = int(T / dt)
    t = np.linspace(0, T, N+1)
    X = np.zeros((n_paths, N+1))
    X[:, 0] = x0

    for i in range(N):
        dW = np.sqrt(dt) * np.random.randn(n_paths)
        X[:, i+1] = X[:, i] - theta * X[:, i] * dt + sigma * dW

    return t, X

np.random.seed(42)

theta = 2.0   # mean-reversion rate
sigma = 1.0   # noise intensity
T = 5.0
dt = 0.001

# Theoretical stationary distribution
stat_var = sigma**2 / (2 * theta)
stat_std = np.sqrt(stat_var)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Sample paths from different starting points
starting_points = [-3, -1, 0, 1, 3]
colors = plt.cm.viridis(np.linspace(0, 1, len(starting_points)))
for x0, color in zip(starting_points, colors):
    t, X = simulate_ou(x0, theta, sigma, T, dt, n_paths=1)
    axes[0].plot(t, X[0], linewidth=0.7, alpha=0.8, color=color, label=rf'$x_0 = {x0}$')

axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
axes[0].fill_between(np.linspace(0, T, 100), -2*stat_std, 2*stat_std,
                      alpha=0.1, color='red', label=rf'$\pm 2\sigma_\mathrm{{stat}} = \pm{2*stat_std:.2f}$')
axes[0].set_xlabel(r'Time $t$', fontsize=12)
axes[0].set_ylabel(r'$X(t)$', fontsize=12)
axes[0].set_title(r'OU Paths: Mean Reversion', fontsize=13)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Panel 2: Many paths converging to stationary distribution
t, X = simulate_ou(3.0, theta, sigma, T, dt, n_paths=500)

# Plot density evolution
time_slices = [0, int(0.5/dt), int(1.0/dt), int(2.0/dt), int(T/dt)]
colors2 = ['#F44336', '#FF9800', '#4CAF50', '#2196F3', '#9C27B0']
for ti, color in zip(time_slices, colors2):
    time_val = ti * dt
    samples = X[:, ti]
    axes[1].hist(samples, bins=40, density=True, alpha=0.3, color=color)

    # Theoretical distribution at time t
    mean_t = 3.0 * np.exp(-theta * time_val)
    var_t = (sigma**2 / (2*theta)) * (1 - np.exp(-2*theta*time_val))
    if var_t > 0:
        x_range = np.linspace(-3, 5, 200)
        pdf = np.exp(-(x_range - mean_t)**2 / (2*var_t)) / np.sqrt(2*np.pi*var_t)
        axes[1].plot(x_range, pdf, color=color, linewidth=2, label=rf'$t = {time_val:.1f}$')

axes[1].set_xlabel(r'$X$', fontsize=12)
axes[1].set_ylabel(r'Density', fontsize=12)
axes[1].set_title(r'OU: Convergence to Stationary Distribution', fontsize=13)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# Panel 3: Mean and variance over time (theory vs simulation)
t_theory = np.linspace(0, T, 200)
mean_theory = 3.0 * np.exp(-theta * t_theory)
var_theory = (sigma**2 / (2*theta)) * (1 - np.exp(-2*theta*t_theory))

mean_sim = np.mean(X, axis=0)
var_sim = np.var(X, axis=0)
t_sim = np.linspace(0, T, X.shape[1])

axes[2].plot(t_theory, mean_theory, 'b-', linewidth=2, label=r'$\mathbb{E}[X(t)]$ theory')
axes[2].plot(t_sim[::100], mean_sim[::100], 'b.', markersize=3, alpha=0.5, label=r'$\mathbb{E}[X(t)]$ sim')
axes[2].plot(t_theory, var_theory, 'r-', linewidth=2, label=r'$\mathrm{Var}[X(t)]$ theory')
axes[2].plot(t_sim[::100], var_sim[::100], 'r.', markersize=3, alpha=0.5, label=r'$\mathrm{Var}[X(t)]$ sim')
axes[2].axhline(y=stat_var, color='red', linestyle='--', alpha=0.5, label=rf'$\sigma^2/(2\theta) = {stat_var:.3f}$')
axes[2].set_xlabel(r'Time $t$', fontsize=12)
axes[2].set_ylabel(r'Value', fontsize=12)
axes[2].set_title(r'OU: Mean and Variance vs Theory', fontsize=13)
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ou_process.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Simple 1D Score Matching

```python
import numpy as np
import matplotlib.pyplot as plt

def score_matching_1d():
    """
    Train a simple neural network to learn the score function of a 1D distribution
    via denoising score matching, then use it to generate samples.
    """
    np.random.seed(42)

    # Target distribution: mixture of two Gaussians
    def sample_data(n):
        mask = np.random.rand(n) < 0.5
        samples = np.where(mask, np.random.randn(n)*0.5 + 3, np.random.randn(n)*0.5 - 3)
        return samples

    # Simple MLP for score estimation (using numpy for transparency)
    # Architecture: input (x, sigma) -> hidden -> score
    hidden_size = 64

    # Xavier initialization
    W1 = np.random.randn(2, hidden_size) / np.sqrt(2)
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
    b2 = np.zeros(hidden_size)
    W3 = np.random.randn(hidden_size, 1) / np.sqrt(hidden_size)
    b3 = np.zeros(1)

    def relu(x):
        return np.maximum(0, x)

    def relu_grad(x):
        return (x > 0).astype(float)

    def forward(x, sigma):
        inp = np.column_stack([x, sigma])  # (batch, 2)
        h1 = relu(inp @ W1 + b1)  # (batch, hidden)
        h2 = relu(h1 @ W2 + b2)   # (batch, hidden)
        out = (h2 @ W3 + b3).flatten()  # (batch,)
        return out, (inp, h1, h2)

    def backward(x, sigma, target, lr=0.001):
        nonlocal W1, b1, W2, b2, W3, b3
        out, (inp, h1, h2) = forward(x, sigma)
        batch = len(x)

        # Loss gradient
        dout = 2 * (out - target) / batch  # (batch,)

        # Layer 3
        dW3 = h2.T @ dout.reshape(-1, 1)
        db3 = np.sum(dout)
        dh2 = dout.reshape(-1, 1) @ W3.T

        # Layer 2
        dh2 = dh2 * relu_grad(h1 @ W2 + b2)
        dW2 = h1.T @ dh2
        db2 = np.sum(dh2, axis=0)
        dh1 = dh2 @ W2.T

        # Layer 1
        dh1 = dh1 * relu_grad(inp @ W1 + b1)
        dW1 = inp.T @ dh1
        db1 = np.sum(dh1, axis=0)

        # SGD update
        W3 -= lr * dW3; b3 -= lr * db3
        W2 -= lr * dW2; b2 -= lr * db2
        W1 -= lr * dW1; b1 -= lr * db1

        return np.mean((out - target)**2)

    # Training via denoising score matching
    # At noise level sigma: score = -(x_noisy - x_clean) / sigma^2
    sigma_levels = np.array([0.1, 0.3, 0.5, 1.0, 2.0])
    losses = []

    print("Training score network...")
    for epoch in range(5000):
        # Sample data
        x_clean = sample_data(256)
        # Random noise level
        sigma_idx = np.random.randint(0, len(sigma_levels), size=256)
        sigma = sigma_levels[sigma_idx]
        # Add noise
        x_noisy = x_clean + sigma * np.random.randn(256)
        # Target score: -(x_noisy - x_clean) / sigma^2
        target_score = -(x_noisy - x_clean) / sigma**2

        loss = backward(x_noisy, sigma, target_score, lr=0.0005)
        losses.append(loss)

        if (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1}: loss = {loss:.4f}")

    # Generate samples using Langevin dynamics with learned score
    def langevin_sample(n_samples, n_steps=1000, step_size=0.01):
        x = np.random.randn(n_samples) * 3  # Start from broad distribution
        sigma_val = np.ones(n_samples) * 0.1  # Use smallest noise level

        for i in range(n_steps):
            score, _ = forward(x, sigma_val)
            noise = np.random.randn(n_samples)
            x = x + step_size * score + np.sqrt(2 * step_size) * noise
        return x

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Training loss
    axes[0, 0].semilogy(losses[::10], color='#2196F3', linewidth=0.5)
    axes[0, 0].set_xlabel(r'Epoch ($\times 10$)')
    axes[0, 0].set_ylabel(r'Loss')
    axes[0, 0].set_title(r'Denoising Score Matching Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # Learned score function vs true score
    x_eval = np.linspace(-6, 6, 500)

    # True score of the mixture: d/dx log p(x)
    def true_density(x):
        return 0.5 * np.exp(-(x-3)**2/(2*0.25)) / np.sqrt(2*np.pi*0.25) + \
               0.5 * np.exp(-(x+3)**2/(2*0.25)) / np.sqrt(2*np.pi*0.25)

    def true_score(x):
        p = true_density(x)
        dp = 0.5 * (-(x-3)/0.25) * np.exp(-(x-3)**2/(2*0.25)) / np.sqrt(2*np.pi*0.25) + \
             0.5 * (-(x+3)/0.25) * np.exp(-(x+3)**2/(2*0.25)) / np.sqrt(2*np.pi*0.25)
        return dp / (p + 1e-10)

    sigma_eval = np.ones(500) * 0.1
    learned_score, _ = forward(x_eval, sigma_eval)

    axes[0, 1].plot(x_eval, true_score(x_eval), 'b-', linewidth=2, label=r'True score')
    axes[0, 1].plot(x_eval, learned_score, 'r--', linewidth=2, label=r'Learned score')
    axes[0, 1].set_xlabel(r'$x$')
    axes[0, 1].set_ylabel(r'$\nabla_x \log p(x)$')
    axes[0, 1].set_title(r'Score Function: True vs Learned')
    axes[0, 1].legend()
    axes[0, 1].set_ylim(-30, 30)
    axes[0, 1].grid(True, alpha=0.3)

    # True density
    axes[1, 0].plot(x_eval, true_density(x_eval), 'b-', linewidth=2)
    axes[1, 0].fill_between(x_eval, true_density(x_eval), alpha=0.3, color='#2196F3')
    axes[1, 0].set_xlabel(r'$x$')
    axes[1, 0].set_ylabel(r'$p(x)$')
    axes[1, 0].set_title(r'Target Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # Generated samples
    print("Generating samples via Langevin dynamics...")
    generated = langevin_sample(5000, n_steps=2000, step_size=0.005)
    axes[1, 1].hist(generated, bins=80, density=True, alpha=0.7, color='#F44336',
                     edgecolor='none', label=r'Generated')
    axes[1, 1].plot(x_eval, true_density(x_eval), 'b-', linewidth=2, label=r'True')
    axes[1, 1].set_xlabel(r'$x$')
    axes[1, 1].set_ylabel(r'Density')
    axes[1, 1].set_title(r'Generated Samples vs True Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('score_matching.png', dpi=150, bbox_inches='tight')
    plt.show()

score_matching_1d()
```

---

## Conclusion

The mathematical chain connecting stochastic calculus to diffusion models is long but each link is logical and necessary:

1. **Brownian motion** provides the random driving force. Its key property --- non-zero quadratic variation, \((dW)^2 = dt\) --- forces us to rebuild calculus.

2. **The Ito integral** defines integration against Brownian motion using left-endpoint evaluation, producing a well-defined stochastic integral with the martingale property.

3. **Ito's lemma** is the chain rule with a correction term: the second derivative of any function gets multiplied by the noise variance and contributes to the drift. This is not an artifact --- it is a physical effect (noise-induced drift).

4. **The Fokker-Planck equation** translates the trajectory-level SDE into a density-level PDE, telling us how probability distributions evolve under drift and diffusion.

5. **The forward diffusion** uses a specific SDE to transform any data distribution into a Gaussian --- a known, easy-to-sample distribution.

6. **Anderson's reverse-time SDE** shows that this process can be reversed, provided we know the score function \(\nabla \log p_t(x)\) at every time and location.

7. **Denoising score matching** trains a neural network to approximate this score by predicting the noise that was added --- which is the DDPM training objective.

8. **Euler-Maruyama discretization** of the reverse SDE yields the DDPM sampling formula, and higher-order discretizations yield faster samplers.

Every piece of the diffusion model pipeline --- training, sampling, fast sampling, guidance --- rests on this mathematical foundation. The theory is not optional background. It is the engineering specification.
