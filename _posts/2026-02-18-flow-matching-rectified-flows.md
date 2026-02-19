---
layout: post
title: "Flow Matching and Rectified Flows: The Elegant Alternative to Diffusion That's Taking Over"
date: 2026-02-18
category: math
---

Diffusion models generate stunning images and videos, but they pay a steep computational tax. The forward process adds noise along curved, stochastic paths. The reverse process must trace those paths back step by step, requiring dozens or hundreds of neural network evaluations to produce a single sample. The math is beautiful --- stochastic differential equations, score functions, the Fokker-Planck equation --- but the paths themselves are wasteful. They wander.

What if you could transport noise to data along **straight lines** instead?

This is the core idea behind **flow matching** and **rectified flows**. Instead of defining a stochastic process and learning to reverse it, you define a deterministic ODE that pushes a simple distribution (Gaussian noise) to the data distribution along the most direct route. The velocity field of this ODE is learned by simple regression --- no ODE solver in the training loop, no score function estimation, no reverse-time SDE. The math is simpler, the training is simpler, and the generation is faster: 4--8 steps instead of 50--1000, with comparable quality. In the limit, a single step suffices.

This post builds the entire framework from the ground up. We start with continuous normalizing flows and why they were historically expensive, derive the flow matching objective and the key theorem that makes it tractable, explain how rectified flows straighten paths for few-step generation, and connect everything back to diffusion models as a special case. By the end, you will understand why every major image and video model released in the last two years --- Stable Diffusion 3, Flux, and beyond --- has switched from diffusion to flow matching.

---

## Table of Contents

1. [The Problem with Diffusion](#the-problem-with-diffusion)
2. [Continuous Normalizing Flows](#continuous-normalizing-flows)
3. [The Flow Matching Objective](#the-flow-matching-objective)
4. [The Key Theorem: Conditional Flow Matching](#the-key-theorem-conditional-flow-matching)
5. [Rectified Flows: Straightening the Paths](#rectified-flows-straightening-the-paths)
6. [Comparison to Diffusion Models](#comparison-to-diffusion-models)
7. [OT Path vs. VP/VE Paths: Unifying the Framework](#ot-path-vs-vpve-paths-unifying-the-framework)
8. [Practical Impact: Why the Industry Switched](#practical-impact-why-the-industry-switched)
9. [Python Simulation: Flow Matching in 2D](#python-simulation-flow-matching-in-2d)

---

## The Problem with Diffusion

In the [SDE post](/2026/01/13/stochastic-differential-equations-diffusion.html), we built the full mathematical machinery of diffusion models. Here is a compressed recap of the core pipeline:

**Forward process.** Start with a data sample $x_0 \sim p_\text{data}$ and progressively add noise via an SDE:

$$dx_t = f(x_t, t)\,dt + g(t)\,dW_t$$

Over time $t \in [0, T]$, this transforms the data distribution into (approximately) a standard Gaussian $\mathcal{N}(0, I)$.

**Reverse process.** To generate, start from noise $x_T \sim \mathcal{N}(0, I)$ and run the reverse-time SDE:

$$dx_t = \left[f(x_t, t) - g(t)^2 \nabla_x \log p_t(x_t)\right]dt + g(t)\,d\bar{W}_t$$

The score function $\nabla_x \log p_t(x)$ is approximated by a neural network trained via denoising score matching.

**The problem:** this reverse process is *stochastic* and the paths it traces through data space are *curved*. Think of a particle doing a random walk that gradually drifts toward a target --- it meanders, doubles back, takes detours. Each of these detours requires a small step, and each step requires a neural network evaluation.

In practice, DDPM uses $T = 1000$ steps. Faster samplers like DDIM reduce this to 50--100 steps by converting the SDE to a probability flow ODE (removing the noise term $d\bar{W}_t$), but the paths are still curved because the velocity field inherited from the diffusion process is not straight. The ODE solver must take many small steps to track these curves accurately.

<svg viewBox="0 0 700 300" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <defs>
    <marker id="arrowR" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#E53935"/>
    </marker>
    <marker id="arrowB" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#1E88E5"/>
    </marker>
  </defs>
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">Diffusion vs. Flow Matching: Path Geometry</text>

  <!-- Left: Diffusion (curved, stochastic) -->
  <text x="175" y="55" text-anchor="middle" font-size="12" font-weight="bold" fill="#E53935">Diffusion (SDE/ODE)</text>
  <circle cx="60" cy="150" r="6" fill="#E53935" opacity="0.6"/>
  <text x="60" y="180" text-anchor="middle" font-size="10" fill="#999">Noise</text>
  <circle cx="290" cy="150" r="6" fill="#E53935"/>
  <text x="290" y="180" text-anchor="middle" font-size="10" fill="#999">Data</text>
  <!-- Curved path -->
  <path d="M 66,150 C 100,100 130,200 160,120 C 190,40 220,180 250,130 C 265,100 280,140 284,150" fill="none" stroke="#E53935" stroke-width="2" stroke-dasharray="5,3" marker-end="url(#arrowR)"/>
  <text x="175" y="220" text-anchor="middle" font-size="10" fill="#666">Curved paths, many steps needed</text>
  <text x="175" y="235" text-anchor="middle" font-size="10" fill="#666">~50-1000 NFE</text>

  <!-- Right: Flow Matching (straight, deterministic) -->
  <text x="525" y="55" text-anchor="middle" font-size="12" font-weight="bold" fill="#1E88E5">Flow Matching (ODE)</text>
  <circle cx="410" cy="150" r="6" fill="#1E88E5" opacity="0.6"/>
  <text x="410" y="180" text-anchor="middle" font-size="10" fill="#999">Noise</text>
  <circle cx="640" cy="150" r="6" fill="#1E88E5"/>
  <text x="640" y="180" text-anchor="middle" font-size="10" fill="#999">Data</text>
  <!-- Straight path -->
  <line x1="416" y1="150" x2="634" y2="150" stroke="#1E88E5" stroke-width="2.5" marker-end="url(#arrowB)"/>
  <text x="525" y="220" text-anchor="middle" font-size="10" fill="#666">Straight paths, few steps suffice</text>
  <text x="525" y="235" text-anchor="middle" font-size="10" fill="#666">~1-8 NFE</text>

  <!-- Divider -->
  <line x1="350" y1="45" x2="350" y2="270" stroke="#444" stroke-width="1" stroke-dasharray="4,4"/>
</svg>

The fundamental question is: can we design a generative process that transports noise to data along *straight* paths? If the paths are straight, even a crude ODE solver (a single Euler step) can follow them exactly. This would mean one-step generation with no quality loss.

The answer is yes. The framework that achieves this is called **flow matching**, and its path-straightening extension is called **rectified flows**.

---

## Continuous Normalizing Flows

Before we get to flow matching, we need to understand the object it trains: a **continuous normalizing flow** (CNF). This is the mathematical backbone.

### The Setup

A CNF defines a time-dependent velocity field $v_\theta(x, t)$ parameterized by a neural network. This velocity field generates a flow --- a deterministic mapping that transports points through space over time $t \in [0, 1]$. The flow is defined by the ordinary differential equation:

$$\frac{dx_t}{dt} = v_\theta(x_t, t) \tag{1}$$

Starting from an initial point $x_0$ at time $t = 0$, this ODE traces a deterministic trajectory $x_t$ through space. There is no randomness at all --- no Brownian motion, no stochastic term. Given the same starting point, you always get the same path.

The idea is to set things up so that if $x_0 \sim p_0$ (a simple base distribution, say $\mathcal{N}(0, I)$), then the endpoint $x_1$ is distributed according to the data distribution $p_\text{data}$. The velocity field $v_\theta$ must be chosen to make this happen.

### The Flow Map

The ODE (1) defines a **flow map** $\phi_t: \mathbb{R}^d \to \mathbb{R}^d$ that takes any starting point $x_0$ and returns its position at time $t$:

$$\phi_t(x_0) = x_0 + \int_0^t v_\theta(\phi_s(x_0), s)\,ds$$

This map is a diffeomorphism (a smooth, invertible mapping with a smooth inverse) as long as $v_\theta$ is Lipschitz continuous in $x$. The map $\phi_1$ is the full transport: it takes noise and returns data.

### The Change of Variables Formula

Here is the crucial property. If we push a distribution $p_0$ through the flow map $\phi_t$, we get a new distribution $p_t$. The relationship between them is governed by the **instantaneous change of variables formula**.

To derive it, start from the conservation of probability. If a small volume element $dx_0$ around point $x_0$ gets mapped to a volume element $dx_t$ around point $x_t = \phi_t(x_0)$, then the probability mass is conserved:

$$p_0(x_0)\,dx_0 = p_t(x_t)\,dx_t$$

The ratio of the volume elements is the absolute value of the Jacobian determinant:

$$\frac{dx_t}{dx_0} = \left|\det\frac{\partial \phi_t(x_0)}{\partial x_0}\right|$$

So the density transforms as:

$$p_t(x_t) = p_0(x_0) \left|\det\frac{\partial \phi_t}{\partial x_0}\right|^{-1}$$

Taking logarithms and differentiating with respect to $t$ gives the **instantaneous change of log-density**. The derivation (which uses Jacobi's formula for the derivative of a determinant) yields:

$$\frac{d}{dt}\log p_t(x_t) = -\nabla \cdot v_\theta(x_t, t) = -\text{tr}\left(\frac{\partial v_\theta}{\partial x}\right) \tag{2}$$

where $\nabla \cdot v_\theta$ is the **divergence** of the velocity field --- the sum of its partial derivatives along each coordinate. Equation (2) says that the log-density along a trajectory changes at a rate equal to minus the divergence of the velocity field.

This is the continuous analog of the discrete change-of-variables formula used in normalizing flows. The divergence acts as a "compression factor": where the velocity field converges ($\nabla \cdot v < 0$), probability density increases; where it diverges ($\nabla \cdot v > 0$), probability density decreases.

### Why CNFs Were Historically Expensive

The log-density equation (2) is beautiful, but it creates a computational problem. To train a CNF via maximum likelihood, you need to compute $\log p_1(x)$ for data points $x$. This requires:

1. Running the ODE (1) backward from $t = 1$ to $t = 0$ to find the corresponding noise point $x_0$.
2. Simultaneously integrating the divergence (2) along the trajectory.
3. Both require an ODE solver in the training loop.

ODE solvers are expensive: they require multiple evaluations of $v_\theta$ per step, they are sequential (cannot be parallelized across time), and they must satisfy numerical accuracy requirements. Backpropagating through the ODE solver (for gradient computation) is even more expensive, requiring either storing the entire trajectory or solving an adjoint ODE.

This is why the original CNF paper (Chen et al., 2018, "Neural ODEs") was a conceptual breakthrough but not a practical workhorse for generation. Training was too slow compared to GANs or even diffusion models.

Flow matching solves this problem entirely. It trains the same object --- a velocity field $v_\theta$ that defines an ODE --- but uses a completely different training objective that requires **no ODE solver**.

---

## The Flow Matching Objective

### The Goal

We want to learn a velocity field $v_\theta(x, t)$ that generates a flow transporting $p_0 = \mathcal{N}(0, I)$ to $p_1 = p_\text{data}$. The velocity field defines a **marginal probability path** $p_t$ --- the distribution of $x_t$ at each time $t$ --- and a corresponding **marginal velocity field** $u_t(x)$, which is the true velocity that generates this path.

The **marginal velocity field** $u_t(x)$ is defined implicitly: it is the velocity field whose flow generates the probability path $p_t$. Formally, $u_t$ and $p_t$ are linked by the **continuity equation** (the conservation of probability in continuous time):

$$\frac{\partial p_t(x)}{\partial t} + \nabla \cdot \left(p_t(x)\,u_t(x)\right) = 0 \tag{3}$$

This is the lossless transport version of the Fokker-Planck equation --- no diffusion term, only advection. It says that the rate of change of probability density at any point equals the negative divergence of the probability current $p_t u_t$.

### The Marginal Flow Matching Loss

If we knew $u_t(x)$ for all $x$ and $t$, we could simply regress to it:

$$\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\, x \sim p_t}\left[\|v_\theta(x, t) - u_t(x)\|^2\right] \tag{4}$$

This is the **flow matching loss**: sample a random time $t$, sample a point $x$ from the distribution $p_t$ at that time, and minimize the squared error between the model's velocity $v_\theta(x, t)$ and the true velocity $u_t(x)$.

The problem is that $u_t(x)$ and $p_t(x)$ are both intractable. The marginal probability path $p_t$ is a mixture over all data points --- it involves an integral over $p_\text{data}$ that we cannot compute in closed form. And $u_t(x)$ is even harder because it is defined only implicitly through the continuity equation.

This is where the key insight of Lipman et al. (2023) enters.

---

## The Key Theorem: Conditional Flow Matching

### Conditional Probability Paths

Instead of working with the intractable marginal path $p_t$, we define **conditional** probability paths. For each data point $x_1 \sim p_\text{data}$, we define a conditional path $p_t(x \mid x_1)$ that interpolates from noise to that specific data point:

- At $t = 0$: $p_0(x \mid x_1) = \mathcal{N}(x; 0, I)$ (standard Gaussian, independent of $x_1$).
- At $t = 1$: $p_1(x \mid x_1) = \delta(x - x_1)$ (a point mass at $x_1$, or in practice, a very narrow Gaussian centered at $x_1$).

The simplest and most natural choice is the **optimal transport (OT) path** --- a linear interpolation:

$$x_t = (1 - t)x_0 + t x_1, \quad x_0 \sim \mathcal{N}(0, I) \tag{5}$$

This is literally a straight line from the noise point $x_0$ to the data point $x_1$. At $t = 0$, you are at the noise point. At $t = 1$, you are at the data point. At $t = 0.5$, you are exactly halfway between.

The conditional probability path for this interpolation is Gaussian:

$$p_t(x \mid x_1) = \mathcal{N}(x;\, t x_1,\, (1-t)^2 I) \tag{6}$$

To verify: at $t = 0$, this is $\mathcal{N}(0, I)$. At $t = 1$, the mean is $x_1$ and the variance is $0$ --- a delta function at $x_1$. The mean moves linearly from $0$ to $x_1$ while the variance shrinks from $1$ to $0$.

### The Conditional Velocity Field

Given the interpolation $x_t = (1-t)x_0 + tx_1$, what is the velocity $dx_t/dt$? Just differentiate:

$$u_t(x_t \mid x_1) = \frac{dx_t}{dt} = x_1 - x_0 \tag{7}$$

The conditional velocity is constant in time --- it is simply the displacement from the noise point to the data point. Every point moves at constant speed along a straight line. This is as simple as a velocity field can be.

### The Marginal Path as a Mixture

The marginal probability path is obtained by averaging the conditional paths over all data points:

$$p_t(x) = \int p_t(x \mid x_1)\,p_\text{data}(x_1)\,dx_1 = \mathbb{E}_{x_1 \sim p_\text{data}}\left[p_t(x \mid x_1)\right] \tag{8}$$

Similarly, the marginal velocity field can be recovered from the conditional velocity fields (weighted by their probability):

$$u_t(x) = \frac{\int u_t(x \mid x_1)\,p_t(x \mid x_1)\,p_\text{data}(x_1)\,dx_1}{p_t(x)} \tag{9}$$

Both of these are intractable integrals. But we do not need to compute them. The key theorem says that we can match the marginal velocity field by only using the conditional velocity fields.

### The Conditional Flow Matching Theorem

**Theorem** (Lipman et al., 2023). *The conditional flow matching (CFM) loss*

$$\mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t,\, x_1 \sim p_\text{data},\, x_0 \sim \mathcal{N}(0,I)}\left[\|v_\theta(x_t, t) - u_t(x_t \mid x_1)\|^2\right] \tag{10}$$

*where $x_t = (1-t)x_0 + tx_1$, has the same gradients with respect to $\theta$ as the marginal flow matching loss $\mathcal{L}_\text{FM}(\theta)$.*

This is the central result. Let us walk through the proof.

### Proof Sketch

We need to show that $\nabla_\theta \mathcal{L}_\text{CFM} = \nabla_\theta \mathcal{L}_\text{FM}$.

**Step 1.** Expand the marginal flow matching loss:

$$\mathcal{L}_\text{FM} = \mathbb{E}_{t,\, x \sim p_t}\left[\|v_\theta(x, t) - u_t(x)\|^2\right]$$

$$= \mathbb{E}_t \int \|v_\theta(x, t) - u_t(x)\|^2\,p_t(x)\,dx$$

**Step 2.** Expand the squared norm:

$$\|v_\theta - u_t\|^2 = \|v_\theta\|^2 - 2\langle v_\theta, u_t\rangle + \|u_t\|^2$$

The gradient with respect to $\theta$ only acts on terms involving $v_\theta$:

$$\nabla_\theta \mathcal{L}_\text{FM} = \mathbb{E}_t \int \left(2 v_\theta(x,t) - 2 u_t(x)\right) \nabla_\theta v_\theta(x, t)\,p_t(x)\,dx$$

Wait --- actually, let us be more careful. We only need to show equivalence of gradients, so we can ignore terms independent of $\theta$. The $\|u_t\|^2$ term drops out. We are left with:

$$\nabla_\theta \mathcal{L}_\text{FM} = \mathbb{E}_t \int \nabla_\theta\|v_\theta(x,t)\|^2\,p_t(x)\,dx - 2\mathbb{E}_t \int u_t(x)\,\nabla_\theta v_\theta(x,t)\,p_t(x)\,dx$$

**Step 3.** Now expand the conditional flow matching loss:

$$\mathcal{L}_\text{CFM} = \mathbb{E}_{t,\, x_1,\, x_0}\left[\|v_\theta(x_t, t) - u_t(x_t \mid x_1)\|^2\right]$$

The $\|v_\theta\|^2$ term, after marginalizing over $x_1$ and $x_0$, gives:

$$\mathbb{E}_{t,\, x_1,\, x_0}\left[\|v_\theta(x_t, t)\|^2\right] = \mathbb{E}_t \int \|v_\theta(x, t)\|^2 p_t(x)\,dx$$

This is because $x_t = (1-t)x_0 + tx_1$ has marginal distribution $p_t(x)$ when $x_0 \sim \mathcal{N}(0, I)$ and $x_1 \sim p_\text{data}$. So the $\|v_\theta\|^2$ terms match exactly.

**Step 4.** For the cross term, consider:

$$\mathbb{E}_{x_1, x_0}\left[u_t(x_t \mid x_1) \nabla_\theta v_\theta(x_t, t)\right]$$

We can write this as:

$$\int \int u_t(x_t \mid x_1)\,\nabla_\theta v_\theta(x_t, t)\,p_0(x_0)\,p_\text{data}(x_1)\,dx_0\,dx_1$$

Now change variables from $(x_0, x_1)$ to $(x, x_1)$ where $x = x_t = (1-t)x_0 + tx_1$. For fixed $x_1$ and $t$, this is a linear change with unit Jacobian (up to scaling by $(1-t)^{-d}$ which is absorbed into the density). We get:

$$\int \left[\int u_t(x \mid x_1)\,p_t(x \mid x_1)\,p_\text{data}(x_1)\,dx_1\right] \nabla_\theta v_\theta(x, t)\,dx$$

By equation (9), the bracketed expression is exactly $u_t(x)\,p_t(x)$. Therefore:

$$\mathbb{E}_{x_1, x_0}\left[u_t(x_t \mid x_1) \nabla_\theta v_\theta(x_t, t)\right] = \int u_t(x)\,\nabla_\theta v_\theta(x, t)\,p_t(x)\,dx$$

This matches the corresponding term in $\nabla_\theta \mathcal{L}_\text{FM}$. $\square$

### What This Means Practically

The theorem says we can replace the intractable marginal flow matching loss with the conditional version, and the neural network will learn the same thing (in the sense that gradient descent will converge to the same optimum). The conditional version is trivially computable:

1. **Sample** $t \sim \mathcal{U}[0, 1]$, $x_0 \sim \mathcal{N}(0, I)$, $x_1 \sim p_\text{data}$.
2. **Compute** $x_t = (1-t)x_0 + tx_1$.
3. **Compute target** $u_t = x_1 - x_0$.
4. **Regress** $v_\theta(x_t, t)$ to $u_t$:

$$\mathcal{L} = \|v_\theta(x_t, t) - (x_1 - x_0)\|^2$$

No ODE solver. No score function. No divergence computation. Just a simple regression loss. The training loop looks almost identical to diffusion model training --- sample noise, sample data, compute an interpolation, regress a network --- but the target is a velocity (displacement) rather than a noise prediction.

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">Flow Matching Training Pipeline</text>

  <!-- Step boxes -->
  <rect x="20" y="50" width="130" height="60" rx="8" fill="#E3F2FD" stroke="#1E88E5" stroke-width="1.5"/>
  <text x="85" y="75" text-anchor="middle" font-size="11" fill="#1565C0" font-weight="bold">Sample</text>
  <text x="85" y="95" text-anchor="middle" font-size="10" fill="#d4d4d4">x₀ ~ N(0,I)</text>
  <text x="85" y="105" text-anchor="middle" font-size="10" fill="#d4d4d4">x₁ ~ p_data</text>

  <rect x="175" y="50" width="130" height="60" rx="8" fill="#E8F5E9" stroke="#43A047" stroke-width="1.5"/>
  <text x="240" y="75" text-anchor="middle" font-size="11" fill="#2E7D32" font-weight="bold">Interpolate</text>
  <text x="240" y="95" text-anchor="middle" font-size="10" fill="#d4d4d4">t ~ U[0,1]</text>
  <text x="240" y="105" text-anchor="middle" font-size="10" fill="#d4d4d4">xₜ = (1-t)x₀ + tx₁</text>

  <rect x="330" y="50" width="130" height="60" rx="8" fill="#FFF3E0" stroke="#FB8C00" stroke-width="1.5"/>
  <text x="395" y="75" text-anchor="middle" font-size="11" fill="#E65100" font-weight="bold">Target</text>
  <text x="395" y="95" text-anchor="middle" font-size="10" fill="#d4d4d4">uₜ = x₁ - x₀</text>
  <text x="395" y="105" text-anchor="middle" font-size="10" fill="#d4d4d4">(constant velocity)</text>

  <rect x="485" y="50" width="190" height="60" rx="8" fill="#FCE4EC" stroke="#E53935" stroke-width="1.5"/>
  <text x="580" y="75" text-anchor="middle" font-size="11" fill="#C62828" font-weight="bold">Loss</text>
  <text x="580" y="95" text-anchor="middle" font-size="10" fill="#d4d4d4">L = ||v_θ(xₜ, t) - uₜ||²</text>
  <text x="580" y="105" text-anchor="middle" font-size="10" fill="#d4d4d4">Simple MSE regression</text>

  <!-- Arrows -->
  <line x1="150" y1="80" x2="175" y2="80" stroke="#999" stroke-width="1.5" marker-end="url(#arrowG)"/>
  <line x1="305" y1="80" x2="330" y2="80" stroke="#999" stroke-width="1.5" marker-end="url(#arrowG)"/>
  <line x1="460" y1="80" x2="485" y2="80" stroke="#999" stroke-width="1.5" marker-end="url(#arrowG)"/>

  <defs>
    <marker id="arrowG" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#999"/>
    </marker>
  </defs>

  <!-- Comparison note -->
  <rect x="80" y="140" width="540" height="120" rx="8" fill="#1e1e1e" stroke="#444" stroke-width="1"/>
  <text x="350" y="165" text-anchor="middle" font-size="12" font-weight="bold" fill="#d4d4d4">Compare: Diffusion Training vs. Flow Matching Training</text>
  <text x="100" y="190" font-size="11" fill="#E53935" font-weight="bold">Diffusion:</text>
  <text x="185" y="190" font-size="11" fill="#999">xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε,   target = ε,   loss = ||ε_θ(xₜ,t) - ε||²</text>
  <text x="100" y="215" font-size="11" fill="#1E88E5" font-weight="bold">Flow Matching:</text>
  <text x="215" y="215" font-size="11" fill="#999">xₜ = (1-t) x₀ + t x₁,   target = x₁-x₀,   loss = ||v_θ(xₜ,t) - (x₁-x₀)||²</text>
  <text x="350" y="245" text-anchor="middle" font-size="10" fill="#666">Both are simple MSE regression. The difference is the interpolation schedule and the target.</text>
</svg>

---

## Rectified Flows: Straightening the Paths

Flow matching with the OT interpolation gives us straight *conditional* paths --- each individual $(x_0, x_1)$ pair travels in a straight line. But the *learned* velocity field $v_\theta(x, t)$ must handle all pairs simultaneously, and at any point $x_t$, multiple different conditional paths may pass through the same location with different velocities. The network must average over these, which means the resulting *marginal* flow paths (obtained by integrating the learned $v_\theta$) are generally not perfectly straight.

**Rectified flows** (Liu et al., 2023) address this by iteratively straightening the paths.

### The Reflow Procedure

The idea is elegant:

**Step 1: Initial flow.** Train a flow matching model $v_\theta$ on pairs $(x_0, x_1)$ where $x_0 \sim \mathcal{N}(0, I)$ and $x_1 \sim p_\text{data}$. These are *independent* --- there is no particular relationship between a noise sample and the data sample it is paired with.

**Step 2: Generate new couplings.** Use the trained model to create *coupled* pairs. For each noise sample $x_0 \sim \mathcal{N}(0, I)$, solve the ODE $dx_t/dt = v_\theta(x_t, t)$ from $t = 0$ to $t = 1$ to get the corresponding output $\hat{x}_1 = \phi_1(x_0)$. Now the pair $(x_0, \hat{x}_1)$ is coupled --- $\hat{x}_1$ is the specific data point that the flow assigns to $x_0$.

**Step 3: Retrain.** Train a new flow matching model on the coupled pairs $(x_0, \hat{x}_1)$. Because these pairs are *correlated* (each noise point maps to its "correct" data point under the first model), the conditional paths $(1-t)x_0 + t\hat{x}_1$ will be more aligned with the marginal flow. The paths that the new model learns will be straighter.

**Repeat.** Each iteration of this "reflow" procedure produces straighter paths. In the limit of infinite reflowing, the paths converge to straight lines, and the velocity field becomes nearly constant along each trajectory.

### Why Reflowing Straightens Paths

Here is the mathematical intuition. Consider a point $x$ at time $t$. Under the marginal flow, many different conditional paths $(1-t)x_0^{(i)} + tx_1^{(i)}$ pass through or near $x$, each with velocity $x_1^{(i)} - x_0^{(i)}$. The learned velocity $v_\theta(x, t)$ is approximately the average of these velocities, weighted by probability:

$$v_\theta(x, t) \approx \frac{\sum_i (x_1^{(i)} - x_0^{(i)}) \cdot p_t(x \mid x_1^{(i)})}{\sum_i p_t(x \mid x_1^{(i)})}$$

When $x_0$ and $x_1$ are independent (as in the initial training), there is a lot of "crossing" --- paths that pass through the same region of space but head in different directions. The average velocity at that point is a compromise, and the resulting marginal flow paths curve.

After reflowing, $x_0$ and $\hat{x}_1$ are coupled by the flow itself. Paths that previously crossed now tend to be parallel (because $\hat{x}_1$ is the destination that the flow already assigns to $x_0$). With less crossing, there is less velocity averaging, less compromise, and the paths are straighter.

<svg viewBox="0 0 700 350" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">Rectified Flow: Path Straightening via Reflow</text>

  <!-- Left panel: Initial (crossing paths) -->
  <text x="175" y="55" text-anchor="middle" font-size="12" font-weight="bold" fill="#E53935">Before Reflow</text>
  <text x="175" y="70" text-anchor="middle" font-size="10" fill="#666">Independent coupling (x₀, x₁)</text>

  <!-- Noise points (left side) -->
  <circle cx="40" cy="120" r="4" fill="#E53935" opacity="0.5"/>
  <circle cx="40" cy="180" r="4" fill="#43A047" opacity="0.5"/>
  <circle cx="40" cy="240" r="4" fill="#1E88E5" opacity="0.5"/>
  <circle cx="40" cy="300" r="4" fill="#FB8C00" opacity="0.5"/>

  <!-- Data points (right side of left panel) -->
  <circle cx="310" cy="280" r="4" fill="#E53935"/>
  <circle cx="310" cy="120" r="4" fill="#43A047"/>
  <circle cx="310" cy="180" r="4" fill="#1E88E5"/>
  <circle cx="310" cy="240" r="4" fill="#FB8C00"/>

  <!-- Crossing paths -->
  <path d="M 44,120 C 120,120 200,260 306,280" fill="none" stroke="#E53935" stroke-width="1.5" opacity="0.7"/>
  <path d="M 44,180 C 120,170 200,130 306,120" fill="none" stroke="#43A047" stroke-width="1.5" opacity="0.7"/>
  <path d="M 44,240 C 120,230 200,190 306,180" fill="none" stroke="#1E88E5" stroke-width="1.5" opacity="0.7"/>
  <path d="M 44,300 C 120,290 200,250 306,240" fill="none" stroke="#FB8C00" stroke-width="1.5" opacity="0.7"/>

  <!-- Crossing indicator -->
  <circle cx="155" cy="195" r="12" fill="none" stroke="#d4d4d4" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="155" y="198" text-anchor="middle" font-size="8" fill="#d4d4d4">cross</text>

  <!-- Labels -->
  <text x="40" y="330" text-anchor="middle" font-size="10" fill="#999">x₀</text>
  <text x="310" y="330" text-anchor="middle" font-size="10" fill="#999">x₁</text>

  <!-- Divider -->
  <line x1="350" y1="45" x2="350" y2="320" stroke="#444" stroke-width="1" stroke-dasharray="4,4"/>

  <!-- Right panel: After reflow (parallel paths) -->
  <text x="525" y="55" text-anchor="middle" font-size="12" font-weight="bold" fill="#1E88E5">After Reflow</text>
  <text x="525" y="70" text-anchor="middle" font-size="10" fill="#666">Coupled (x₀, x̂₁) from learned flow</text>

  <!-- Noise points -->
  <circle cx="390" cy="120" r="4" fill="#E53935" opacity="0.5"/>
  <circle cx="390" cy="180" r="4" fill="#43A047" opacity="0.5"/>
  <circle cx="390" cy="240" r="4" fill="#1E88E5" opacity="0.5"/>
  <circle cx="390" cy="300" r="4" fill="#FB8C00" opacity="0.5"/>

  <!-- Data points (now reordered to avoid crossing) -->
  <circle cx="660" cy="140" r="4" fill="#E53935"/>
  <circle cx="660" cy="195" r="4" fill="#43A047"/>
  <circle cx="660" cy="250" r="4" fill="#1E88E5"/>
  <circle cx="660" cy="305" r="4" fill="#FB8C00"/>

  <!-- Nearly straight, non-crossing paths -->
  <line x1="394" y1="120" x2="656" y2="140" stroke="#E53935" stroke-width="1.8" opacity="0.8"/>
  <line x1="394" y1="180" x2="656" y2="195" stroke="#43A047" stroke-width="1.8" opacity="0.8"/>
  <line x1="394" y1="240" x2="656" y2="250" stroke="#1E88E5" stroke-width="1.8" opacity="0.8"/>
  <line x1="394" y1="300" x2="656" y2="305" stroke="#FB8C00" stroke-width="1.8" opacity="0.8"/>

  <!-- Labels -->
  <text x="390" y="330" text-anchor="middle" font-size="10" fill="#999">x₀</text>
  <text x="660" y="330" text-anchor="middle" font-size="10" fill="#999">x̂₁</text>
</svg>

### Why Straighter Paths Need Fewer Steps

Consider solving the ODE $dx_t/dt = v(x_t, t)$ numerically. The simplest method is the **Euler method**: take a step of size $\Delta t$ in the direction of the current velocity:

$$x_{t+\Delta t} = x_t + \Delta t \cdot v(x_t, t)$$

The Euler method is exact when $v$ is constant along the trajectory (i.e., the path is a straight line). In that case, a single step from $t = 0$ to $t = 1$ recovers the exact solution:

$$x_1 = x_0 + 1 \cdot v(x_0, 0) = x_0 + (x_1 - x_0) = x_1 \quad \checkmark$$

When the path is curved, the Euler method accumulates error at each step. The local truncation error of the Euler method is $O((\Delta t)^2)$ per step, and the global error is $O(\Delta t)$. To achieve error $\epsilon$, you need $O(1/\epsilon)$ steps.

We can quantify the "straightness" of a flow. Define the **straightness** of a trajectory as:

$$S = \frac{\|x_1 - x_0\|}{\int_0^1 \|v(x_t, t)\|\,dt}$$

For a straight-line path at constant speed, $S = 1$ (the length of the displacement equals the length of the path). For a curved path, $S < 1$ (the path is longer than the displacement). The closer $S$ is to 1, the fewer ODE steps are needed for accurate integration.

After multiple rounds of reflowing, $S$ approaches 1, and even a single Euler step gives high-quality samples. This is the promise of rectified flows: **one-step generation** without distillation or any other trick --- just learn a sufficiently straight flow.

### Connection to Optimal Transport

The name "optimal transport" for the linear interpolation path is not accidental. In the theory of optimal transport, the **Wasserstein-2 (W2) distance** between two distributions $p_0$ and $p_1$ is:

$$W_2(p_0, p_1)^2 = \inf_{\pi \in \Pi(p_0, p_1)} \int \|x_0 - x_1\|^2\,d\pi(x_0, x_1)$$

where $\Pi(p_0, p_1)$ is the set of all joint distributions (couplings) with marginals $p_0$ and $p_1$. The optimal coupling $\pi^*$ minimizes the total squared transport distance.

The displacement interpolation under the optimal coupling traces straight lines from each $x_0$ to its matched $x_1$, and these paths do not cross. The rectified flow procedure approximates this optimal transport coupling: each reflow step brings the coupling closer to the OT coupling, reducing path crossing and increasing straightness.

---

## Comparison to Diffusion Models

Let us now put diffusion and flow matching side by side, concretely.

### The Interpolation

**Diffusion (VP-SDE):**

$$x_t = \sqrt{\bar{\alpha}_t}\,x_1 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Here $\bar{\alpha}_t$ is determined by the noise schedule $\beta(t)$ and varies nonlinearly from $\bar{\alpha}_0 = 1$ (pure data) to $\bar{\alpha}_T \approx 0$ (pure noise). Note the convention difference: in diffusion, $x_0$ is data and $x_T$ is noise. In flow matching, $x_0$ is noise and $x_1$ is data. We will use the flow matching convention throughout.

**Flow Matching (OT path):**

$$x_t = (1-t)\,x_0 + t\,x_1, \quad x_0 \sim \mathcal{N}(0, I)$$

Linear interpolation. The coefficient on data is $t$ and the coefficient on noise is $(1-t)$.

### The Training Target

**Diffusion:** Predict the noise $\epsilon$ that was added. The loss is $\|\epsilon_\theta(x_t, t) - \epsilon\|^2$.

**Flow Matching:** Predict the velocity $x_1 - x_0$. The loss is $\|v_\theta(x_t, t) - (x_1 - x_0)\|^2$.

These are related by a change of parameterization. In fact, $x_1 - x_0 = x_1 - \epsilon$ when $x_0 = \epsilon$ is the noise. The velocity prediction can be converted to a noise prediction and vice versa. But the OT interpolation schedule makes the velocity field smoother and easier to learn.

### The Sampling Process

**Diffusion:** Run the reverse SDE (stochastic, requires noise injection at each step) or the probability flow ODE (deterministic, but with curved paths inherited from the diffusion process). Needs 20--100 steps with good samplers (DPM-Solver, DDIM).

**Flow Matching:** Run the ODE $dx_t/dt = v_\theta(x_t, t)$ from $t = 0$ to $t = 1$. Deterministic. Paths are approximately straight, so even a basic ODE solver works well in 4--8 steps. With rectified flows, 1 step suffices.

### The Path Geometry

**Diffusion:** The paths through data space are curved because the VP/VE noise schedules create nonlinear interpolation between noise and data. The signal-to-noise ratio changes at a non-constant rate, and the velocity field must change direction to track this. Even the probability flow ODE has curved streamlines.

**Flow Matching:** The OT paths are straight lines by construction. The learned velocity field approximates these straight lines, and each reflow iteration makes the approximation better.

| Property | Diffusion (VP-SDE) | Flow Matching (OT) |
|:--|:--|:--|
| Process type | Stochastic (SDE) or deterministic (ODE) | Deterministic (ODE) |
| Path geometry | Curved | Straight (approximately) |
| Training target | Noise $\epsilon$ | Velocity $x_1 - x_0$ |
| Training cost | Same (simple regression) | Same (simple regression) |
| Sampling steps | 20--100 (ODE), 50--1000 (SDE) | 1--8 (ODE) |
| Score function needed | Yes | No |
| ODE solver in training | No (denoising score matching) | No (conditional flow matching) |
| Quality | State-of-the-art | State-of-the-art |

The bottom line: flow matching achieves the same generation quality with dramatically fewer sampling steps and simpler math. The training cost is identical --- both are simple regression losses. The only disadvantage of flow matching is the need for reflowing to get truly one-step generation, which requires generating a dataset of coupled pairs (an additional cost at training time, but a one-time investment).

---

## OT Path vs. VP/VE Paths: Unifying the Framework

One of the most elegant aspects of the flow matching framework is its generality. The OT (linear interpolation) path is just one choice among many. By choosing different conditional probability paths $p_t(x \mid x_1)$, you can recover diffusion models as special cases.

### The General Conditional Path

A general Gaussian conditional path takes the form:

$$p_t(x \mid x_1) = \mathcal{N}(x;\, \alpha_t x_1,\, \sigma_t^2 I)$$

where $\alpha_t$ and $\sigma_t$ are time-dependent schedules satisfying:
- $\alpha_0 = 0$, $\sigma_0 = 1$ (at $t = 0$: centered at origin, unit variance = standard Gaussian)
- $\alpha_1 = 1$, $\sigma_1 = 0$ (at $t = 1$: centered at data, zero variance = point mass at $x_1$)

The corresponding interpolation is:

$$x_t = \alpha_t x_1 + \sigma_t x_0, \quad x_0 \sim \mathcal{N}(0, I)$$

and the conditional velocity field is:

$$u_t(x_t \mid x_1) = \frac{d x_t}{dt} = \dot{\alpha}_t x_1 + \dot{\sigma}_t x_0$$

where $\dot{\alpha}_t = d\alpha_t/dt$ and $\dot{\sigma}_t = d\sigma_t/dt$.

### Recovering the OT Path

The OT path corresponds to:

$$\alpha_t = t, \quad \sigma_t = 1 - t$$

$$x_t = t\,x_1 + (1-t)\,x_0$$

$$u_t = x_1 - x_0$$

The velocity is constant. The coefficients change linearly. The signal-to-noise ratio $\alpha_t / \sigma_t = t/(1-t)$ increases monotonically from $0$ to $\infty$.

### Recovering the VP-SDE Path

The VP-SDE forward process gives:

$$x_t = \sqrt{\bar{\alpha}_t}\,x_1 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon$$

In the flow matching framework (reversing the time convention so that $t=0$ is noise and $t=1$ is data), this corresponds to:

$$\alpha_t = \sqrt{\bar{\alpha}_{1-t}}, \quad \sigma_t = \sqrt{1 - \bar{\alpha}_{1-t}}$$

The conditional velocity field is:

$$u_t = \dot{\alpha}_t x_1 + \dot{\sigma}_t x_0$$

where the derivatives of $\alpha_t$ and $\sigma_t$ are determined by the noise schedule $\beta(t)$. These derivatives are **not** constant --- they depend on $t$ through the noise schedule. This means the velocity changes along the path, making the path curved.

### Recovering the VE-SDE Path

The VE-SDE has:

$$x_t = x_1 + \sigma(t)\,\epsilon$$

In flow matching notation:

$$\alpha_t = 1, \quad \sigma_t = \sigma(1-t)$$

The signal coefficient is constant (no signal attenuation), and only the noise level changes. The conditional velocity is:

$$u_t = \dot{\sigma}_t\,x_0$$

This points purely in the noise direction --- the path moves "radially" from data outward into noise space, with no component along the data direction.

### Why OT is Best

Among all choices of $(\alpha_t, \sigma_t)$, the OT path is special:

1. **Constant velocity.** $u_t = x_1 - x_0$ does not depend on $t$. The velocity field is easier for the network to learn (it does not need to predict time-dependent changes).

2. **Straight paths.** Constant velocity means straight trajectories, which means fewer ODE steps for sampling.

3. **Minimal transport cost.** The expected squared path length $\mathbb{E}\left[\int_0^1 \|u_t\|^2 dt\right] = \mathbb{E}\left[\|x_1 - x_0\|^2\right]$ is minimized (among all paths connecting the same endpoints) by the straight-line path.

4. **Uniform SNR progression.** The signal-to-noise ratio $\alpha_t/\sigma_t = t/(1-t)$ increases smoothly from $0$ to $\infty$ without the extreme compression that VP and VE schedules exhibit near the endpoints.

The VP and VE paths spend a disproportionate amount of "time" at very high or very low noise levels (depending on the schedule), creating a mismatch between the time variable $t$ and the informational content of $x_t$. The OT path distributes the informational content uniformly across $t \in [0, 1]$, making every time step equally "useful" for learning.

<svg viewBox="0 0 700 350" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">Signal and Noise Coefficients: OT vs VP vs VE</text>

  <!-- Axes -->
  <line x1="60" y1="290" x2="660" y2="290" stroke="#333" stroke-width="1"/>
  <line x1="60" y1="290" x2="60" y2="50" stroke="#333" stroke-width="1"/>
  <text x="660" y="310" text-anchor="end" font-size="11" fill="#666">t (0 = noise, 1 = data)</text>
  <text x="55" y="45" text-anchor="end" font-size="11" fill="#666">Coeff.</text>

  <!-- Grid -->
  <line x1="60" y1="170" x2="660" y2="170" stroke="#eee" stroke-width="0.5"/>

  <!-- Tick marks -->
  <text x="60" y="308" font-size="10" fill="#666" text-anchor="middle">0</text>
  <text x="360" y="308" font-size="10" fill="#666" text-anchor="middle">0.5</text>
  <text x="660" y="308" font-size="10" fill="#666" text-anchor="middle">1</text>
  <text x="50" y="294" font-size="10" fill="#666" text-anchor="end">0</text>
  <text x="50" y="174" font-size="10" fill="#666" text-anchor="end">0.5</text>
  <text x="50" y="58" font-size="10" fill="#666" text-anchor="end">1</text>

  <!-- OT: alpha_t = t (linear), sigma_t = 1-t (linear) -->
  <line x1="60" y1="290" x2="660" y2="50" stroke="#1E88E5" stroke-width="2.5"/>
  <line x1="60" y1="50" x2="660" y2="290" stroke="#1E88E5" stroke-width="2.5" stroke-dasharray="6,4"/>

  <!-- VP: alpha_t = sqrt(alpha_bar), sigma_t = sqrt(1-alpha_bar) — S-curve -->
  <path d="M 60,290 C 180,285 300,260 360,170 C 420,80 540,55 660,50" fill="none" stroke="#E53935" stroke-width="2"/>
  <path d="M 60,50 C 180,55 300,80 360,170 C 420,260 540,285 660,290" fill="none" stroke="#E53935" stroke-width="2" stroke-dasharray="6,4"/>

  <!-- VE: alpha_t = 1 (constant), sigma_t decreases -->
  <line x1="60" y1="50" x2="660" y2="50" stroke="#43A047" stroke-width="2"/>
  <path d="M 60,50 C 200,60 400,150 660,290" fill="none" stroke="#43A047" stroke-width="2" stroke-dasharray="6,4"/>

  <!-- Legend -->
  <rect x="420" y="55" width="220" height="110" rx="6" fill="white" stroke="#ddd" stroke-width="1" opacity="0.95"/>
  <line x1="435" y1="75" x2="470" y2="75" stroke="#1E88E5" stroke-width="2.5"/>
  <text x="478" y="79" font-size="11" fill="#333">OT (linear)</text>
  <line x1="435" y1="98" x2="470" y2="98" stroke="#E53935" stroke-width="2"/>
  <text x="478" y="102" font-size="11" fill="#333">VP-SDE (S-curve)</text>
  <line x1="435" y1="121" x2="470" y2="121" stroke="#43A047" stroke-width="2"/>
  <text x="478" y="125" font-size="11" fill="#333">VE-SDE</text>
  <text x="500" y="150" font-size="10" fill="#999">Solid = α_t, Dashed = σ_t</text>
</svg>

---

## Practical Impact: Why the Industry Switched

The theoretical advantages of flow matching translate directly into practical gains. Here is how the transition has played out.

### Stable Diffusion 3 and SD 3.5

Stability AI's Stable Diffusion 3 (March 2024) was the first major image model to publicly adopt flow matching. The architecture uses a "Multimodal Diffusion Transformer" (MMDiT) with the rectified flow formulation. Key changes from SD 1.5/SDXL:

- **Training:** Uses the flow matching loss with the OT interpolation path. The model predicts velocities, not noise.
- **Sampling:** Uses a basic ODE solver (Euler or Heun's method) with 20--30 steps for high quality, or as few as 4--8 steps with quality that matches the previous generation of 50-step diffusion samplers.
- **Guidance:** Classifier-free guidance still applies (the CFG equation works identically --- you just replace noise predictions with velocity predictions).

### Flux (Black Forest Labs)

Flux, from the team that originally built Stable Diffusion, uses rectified flows explicitly. The model is trained with the flow matching objective and uses a guidance-distilled variant that bakes CFG into the weights, enabling single-pass inference (no separate unconditional evaluation needed). Flux can generate high-quality 1024x1024 images in 4 steps.

### Video Models

The impact on video generation is even more dramatic. Video models must generate dozens or hundreds of frames coherently, and every sampling step is proportionally more expensive (the network processes all frames jointly). Reducing from 50 steps to 4--8 steps means a 6--12x speedup in wall-clock generation time.

Modern video architectures (including those powering commercial platforms) overwhelmingly use flow matching or rectified flow formulations. The combination of:
- **Fewer steps** (lower latency, lower compute cost)
- **Deterministic sampling** (reproducible outputs, easier debugging)
- **Simpler training** (no noise schedule tuning, no score function subtleties)

makes flow matching the default choice for new model development.

### Why Not Switch Earlier?

The ideas behind flow matching and optimal transport paths existed before 2023. Continuous normalizing flows date to 2018. Linear interpolation between distributions is textbook optimal transport theory. What changed?

1. **The conditional flow matching theorem.** Before Lipman et al. (2023) and the concurrent work by Liu et al. (2023) on rectified flows, training CNFs required expensive ODE solvers. The realization that conditional paths could be used to avoid this --- and that the resulting loss has the same gradients as the intractable marginal loss --- was the key innovation.

2. **Scalability proof.** It took large-scale experiments (SD3, Flux) to demonstrate that flow matching works at the quality frontier, not just on toy problems. Once these models matched or exceeded diffusion quality with fewer steps, the industry switched rapidly.

3. **Simplicity.** Flow matching is genuinely simpler to implement, tune, and debug than diffusion with its zoo of noise schedules, score parameterizations, and SDE/ODE solver choices. Simplicity wins in production.

---

## Python Simulation: Flow Matching in 2D

Let us implement flow matching from scratch on a 2D toy problem to build concrete intuition. We will train a velocity field to transport a standard Gaussian to a mixture of Gaussians, and visualize the learned flow lines. We will compare the OT (straight) path with a VP-style (curved) path to see the difference in path geometry.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# ===== Data Distribution: Mixture of 8 Gaussians in a Ring =====

def sample_target(n):
    """Sample from a mixture of 8 Gaussians arranged in a circle."""
    n_modes = 8
    angles = np.linspace(0, 2 * np.pi, n_modes, endpoint=False)
    centers = 4.0 * np.column_stack([np.cos(angles), np.sin(angles)])
    idx = np.random.randint(0, n_modes, size=n)
    samples = centers[idx] + 0.4 * np.random.randn(n, 2)
    return samples

# ===== Simple MLP for velocity prediction =====

class SimpleMLP:
    """A 3-layer MLP with ReLU, trained with Adam, implemented in numpy."""

    def __init__(self, input_dim=3, hidden_dim=128, output_dim=2, lr=1e-3):
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * scale2
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, output_dim) * scale2
        self.b3 = np.zeros(output_dim)
        self.lr = lr
        # Adam state
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.step_count = 0

    def forward(self, x):
        """x: (batch, input_dim). Returns (batch, output_dim) and cache."""
        h1_pre = x @ self.W1 + self.b1
        h1 = np.maximum(0, h1_pre)
        h2_pre = h1 @ self.W2 + self.b2
        h2 = np.maximum(0, h2_pre)
        out = h2 @ self.W3 + self.b3
        cache = (x, h1_pre, h1, h2_pre, h2)
        return out, cache

    def backward_and_step(self, out, target, cache):
        """Compute loss, backprop, and Adam update. Returns loss value."""
        x, h1_pre, h1, h2_pre, h2 = cache
        batch = x.shape[0]
        diff = out - target
        loss = np.mean(np.sum(diff ** 2, axis=1))

        # Gradient of MSE
        d_out = 2.0 * diff / batch

        # Layer 3
        dW3 = h2.T @ d_out
        db3 = np.sum(d_out, axis=0)
        d_h2 = d_out @ self.W3.T

        # ReLU
        d_h2 = d_h2 * (h2_pre > 0).astype(float)

        # Layer 2
        dW2 = h1.T @ d_h2
        db2 = np.sum(d_h2, axis=0)
        d_h1 = d_h2 @ self.W2.T

        # ReLU
        d_h1 = d_h1 * (h1_pre > 0).astype(float)

        # Layer 1
        dW1 = x.T @ d_h1
        db1 = np.sum(d_h1, axis=0)

        # Adam update
        grads = [dW1, db1, dW2, db2, dW3, db3]
        self.step_count += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for i, g in enumerate(grads):
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * g ** 2
            m_hat = self.m[i] / (1 - beta1 ** self.step_count)
            v_hat = self.v[i] / (1 - beta2 ** self.step_count)
            self.params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

        # Sync references
        self.W1, self.b1 = self.params[0], self.params[1]
        self.W2, self.b2 = self.params[2], self.params[3]
        self.W3, self.b3 = self.params[4], self.params[5]
        return loss

    def predict(self, x):
        out, _ = self.forward(x)
        return out

# ===== Training: OT Flow Matching =====

np.random.seed(42)
model_ot = SimpleMLP(input_dim=3, hidden_dim=256, output_dim=2, lr=3e-4)

losses_ot = []
n_steps = 15000
batch_size = 512

for step in range(n_steps):
    # Sample noise and data
    x0 = np.random.randn(batch_size, 2)
    x1 = sample_target(batch_size)
    t = np.random.rand(batch_size, 1)

    # OT interpolation
    xt = (1 - t) * x0 + t * x1

    # Target velocity
    ut = x1 - x0

    # Network input: (xt, t)
    inp = np.concatenate([xt, t], axis=1)
    out, cache = model_ot.forward(inp)
    loss = model_ot.backward_and_step(out, ut, cache)
    losses_ot.append(loss)

    if (step + 1) % 3000 == 0:
        print(f"OT Step {step+1}/{n_steps}, Loss: {loss:.4f}")

# ===== Training: VP-style Flow Matching (curved paths) =====

np.random.seed(42)
model_vp = SimpleMLP(input_dim=3, hidden_dim=256, output_dim=2, lr=3e-4)

losses_vp = []

def vp_schedule(t):
    """VP-style schedule: alpha_t and sigma_t."""
    # Cosine schedule inspired by improved DDPM
    alpha_t = np.cos(0.5 * np.pi * (1 - t)) ** 2
    sigma_t = np.sin(0.5 * np.pi * (1 - t)) ** 2
    # Derivatives
    dalpha_t = np.pi * np.cos(0.5 * np.pi * (1 - t)) * np.sin(0.5 * np.pi * (1 - t))
    dsigma_t = -np.pi * np.sin(0.5 * np.pi * (1 - t)) * np.cos(0.5 * np.pi * (1 - t))
    return alpha_t, sigma_t, dalpha_t, dsigma_t

for step in range(n_steps):
    x0 = np.random.randn(batch_size, 2)
    x1 = sample_target(batch_size)
    t = np.random.rand(batch_size, 1)

    alpha_t, sigma_t, dalpha_t, dsigma_t = vp_schedule(t)
    xt = alpha_t * x1 + sigma_t * x0
    ut = dalpha_t * x1 + dsigma_t * x0

    inp = np.concatenate([xt, t], axis=1)
    out, cache = model_vp.forward(inp)
    loss = model_vp.backward_and_step(out, ut, cache)
    losses_vp.append(loss)

    if (step + 1) % 3000 == 0:
        print(f"VP Step {step+1}/{n_steps}, Loss: {loss:.4f}")

# ===== Sampling: Euler ODE Solver =====

def sample_ode(model, n_samples, n_ode_steps, schedule='ot'):
    """Generate samples by solving the ODE with Euler method."""
    x = np.random.randn(n_samples, 2)
    dt = 1.0 / n_ode_steps
    trajectory = [x.copy()]

    for i in range(n_ode_steps):
        t_val = i * dt
        t_arr = np.full((n_samples, 1), t_val)
        inp = np.concatenate([x, t_arr], axis=1)
        v = model.predict(inp)
        x = x + dt * v
        trajectory.append(x.copy())

    return x, trajectory

# ===== Visualization =====

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# --- Row 1: OT Flow Matching ---

# Panel 1: Training loss
ax = axes[0, 0]
window = 100
smoothed = np.convolve(losses_ot, np.ones(window)/window, mode='valid')
ax.semilogy(smoothed, color='#1E88E5', linewidth=0.8)
ax.set_xlabel(r'Training Step')
ax.set_ylabel(r'Loss $\mathcal{L}$')
ax.set_title(r'OT Flow Matching: Training Loss')
ax.grid(True, alpha=0.3)

# Panel 2: Flow trajectories (OT)
ax = axes[0, 1]
n_traj = 200
x0_traj = np.random.randn(n_traj, 2)
x = x0_traj.copy()
dt_vis = 1.0 / 50
trajectories = [x.copy()]
for i in range(50):
    t_val = i * dt_vis
    t_arr = np.full((n_traj, 1), t_val)
    inp = np.concatenate([x, t_arr], axis=1)
    v = model_ot.predict(inp)
    x = x + dt_vis * v
    trajectories.append(x.copy())

trajectories = np.array(trajectories)  # (51, n_traj, 2)
for j in range(n_traj):
    colors_line = cm.viridis(np.linspace(0, 1, 50))
    for k in range(50):
        ax.plot(trajectories[k:k+2, j, 0], trajectories[k:k+2, j, 1],
                color=colors_line[k], linewidth=0.5, alpha=0.6)

# Show target distribution
x1_show = sample_target(500)
ax.scatter(x1_show[:, 0], x1_show[:, 1], s=5, c='red', alpha=0.3, zorder=5)
ax.set_title(r'OT Flow: Learned Trajectories')
ax.set_xlim(-7, 7)
ax.set_ylim(-7, 7)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)

# Panel 3: Generated samples at different step counts (OT)
ax = axes[0, 2]
for n_steps_sample, color, label in [(1, '#E53935', '1 step'),
                                      (4, '#FB8C00', '4 steps'),
                                      (8, '#43A047', '8 steps'),
                                      (50, '#1E88E5', '50 steps')]:
    samples, _ = sample_ode(model_ot, 800, n_steps_sample, schedule='ot')
    ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.4, c=color, label=label)
ax.set_title(r'OT Flow: Sample Quality vs. Steps')
ax.set_xlim(-7, 7)
ax.set_ylim(-7, 7)
ax.set_aspect('equal')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.2)

# --- Row 2: VP-style Flow Matching ---

# Panel 4: Training loss
ax = axes[1, 0]
smoothed_vp = np.convolve(losses_vp, np.ones(window)/window, mode='valid')
ax.semilogy(smoothed_vp, color='#E53935', linewidth=0.8)
ax.set_xlabel(r'Training Step')
ax.set_ylabel(r'Loss $\mathcal{L}$')
ax.set_title(r'VP-style Flow: Training Loss')
ax.grid(True, alpha=0.3)

# Panel 5: Flow trajectories (VP)
ax = axes[1, 1]
x0_traj_vp = np.random.randn(n_traj, 2)
x = x0_traj_vp.copy()
trajectories_vp = [x.copy()]
for i in range(50):
    t_val = i * dt_vis
    t_arr = np.full((n_traj, 1), t_val)
    inp = np.concatenate([x, t_arr], axis=1)
    v = model_vp.predict(inp)
    x = x + dt_vis * v
    trajectories_vp.append(x.copy())

trajectories_vp = np.array(trajectories_vp)
for j in range(n_traj):
    colors_line = cm.magma(np.linspace(0, 1, 50))
    for k in range(50):
        ax.plot(trajectories_vp[k:k+2, j, 0], trajectories_vp[k:k+2, j, 1],
                color=colors_line[k], linewidth=0.5, alpha=0.6)

ax.scatter(x1_show[:, 0], x1_show[:, 1], s=5, c='red', alpha=0.3, zorder=5)
ax.set_title(r'VP-style Flow: Learned Trajectories (Curved)')
ax.set_xlim(-7, 7)
ax.set_ylim(-7, 7)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)

# Panel 6: Generated samples at different step counts (VP)
ax = axes[1, 2]
for n_steps_sample, color, label in [(1, '#E53935', '1 step'),
                                      (4, '#FB8C00', '4 steps'),
                                      (8, '#43A047', '8 steps'),
                                      (50, '#1E88E5', '50 steps')]:
    samples_vp, _ = sample_ode(model_vp, 800, n_steps_sample, schedule='vp')
    ax.scatter(samples_vp[:, 0], samples_vp[:, 1], s=3, alpha=0.4, c=color, label=label)
ax.set_title(r'VP-style Flow: Sample Quality vs. Steps')
ax.set_xlim(-7, 7)
ax.set_ylim(-7, 7)
ax.set_aspect('equal')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('flow_matching_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

### What to Observe

The simulation reveals several key points:

**Row 1 (OT paths):** The learned trajectories are approximately straight --- they fan outward from the Gaussian center toward the eight target modes with minimal curvature. Even with just 4 Euler steps, the generated samples land cleanly on the target modes. With 1 step, the samples are slightly spread out (the paths are not *perfectly* straight because the learned marginal flow averages over crossing conditional paths), but the mode structure is already clearly visible.

**Row 2 (VP-style paths):** The trajectories are visibly curved. They spend early time steps (dark colors) moving slowly through the high-noise region and then accelerate toward the target modes at later steps. With 1 Euler step, the samples are far from the target --- the single step overshoots or undershoots because the velocity changes dramatically along the path. Even with 4 steps, the quality is worse than the OT flow with the same step count. You need 20--50 steps to match the OT flow's 4-step quality.

This is the path geometry argument made concrete. Straight paths tolerate coarse ODE solvers. Curved paths demand fine solvers.

### A Reflow Iteration

To see the straightening effect of reflowing, here is an additional code block that performs one reflow iteration on the OT model:

```python
# ===== Reflow: Generate coupled pairs and retrain =====

# Step 1: Generate coupled (x0, x1_hat) pairs using the trained OT model
n_reflow = 20000
x0_reflow = np.random.randn(n_reflow, 2)
x1_hat = x0_reflow.copy()
dt_reflow = 1.0 / 100  # fine solver for generating couplings

for i in range(100):
    t_val = i * dt_reflow
    t_arr = np.full((n_reflow, 1), t_val)
    inp = np.concatenate([x1_hat, t_arr], axis=1)
    v = model_ot.predict(inp)
    x1_hat = x1_hat + dt_reflow * v

# Step 2: Retrain on coupled pairs
np.random.seed(123)
model_reflow = SimpleMLP(input_dim=3, hidden_dim=256, output_dim=2, lr=3e-4)
losses_reflow = []

for step in range(10000):
    idx = np.random.randint(0, n_reflow, size=batch_size)
    x0_batch = x0_reflow[idx]
    x1_batch = x1_hat[idx]
    t = np.random.rand(batch_size, 1)

    xt = (1 - t) * x0_batch + t * x1_batch
    ut = x1_batch - x0_batch

    inp = np.concatenate([xt, t], axis=1)
    out, cache = model_reflow.forward(inp)
    loss = model_reflow.backward_and_step(out, ut, cache)
    losses_reflow.append(loss)

# Compare 1-step generation: original vs reflowed
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

samples_orig_1, _ = sample_ode(model_ot, 2000, 1)
samples_reflow_1, _ = sample_ode(model_reflow, 2000, 1)
x1_ref = sample_target(2000)

ax = axes[0]
ax.scatter(x1_ref[:, 0], x1_ref[:, 1], s=5, alpha=0.5, c='#1E88E5', label=r'Target')
ax.set_title(r'Target Distribution')
ax.set_xlim(-7, 7); ax.set_ylim(-7, 7); ax.set_aspect('equal')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2)

ax = axes[1]
ax.scatter(samples_orig_1[:, 0], samples_orig_1[:, 1], s=5, alpha=0.5, c='#E53935')
ax.set_title(r'Original Flow: 1 Euler Step')
ax.set_xlim(-7, 7); ax.set_ylim(-7, 7); ax.set_aspect('equal')
ax.grid(True, alpha=0.2)

ax = axes[2]
ax.scatter(samples_reflow_1[:, 0], samples_reflow_1[:, 1], s=5, alpha=0.5, c='#43A047')
ax.set_title(r'After Reflow: 1 Euler Step')
ax.set_xlim(-7, 7); ax.set_ylim(-7, 7); ax.set_aspect('equal')
ax.grid(True, alpha=0.2)

plt.suptitle(r'Reflow Straightens Paths for 1-Step Generation', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('reflow_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

After a single reflow iteration, the 1-step samples are dramatically sharper. The modes are tighter and more clearly separated. A second reflow iteration would make them even better, approaching the quality of a 50-step ODE solve.

---

## Summary

The progression from diffusion models to flow matching is a progression toward simplicity:

1. **Diffusion models** define a stochastic forward process (SDE), learn to reverse it via score matching, and sample by running the reverse SDE or probability flow ODE. The paths are curved and stochastic. Sampling requires many steps.

2. **Continuous normalizing flows** define a deterministic ODE that transports noise to data. Clean mathematically, but historically required ODE solvers in the training loop, making them impractical.

3. **Flow matching** (Lipman et al., 2023) solves the training problem. The key theorem shows that you can regress to *conditional* velocity fields (trivially computable) instead of the intractable *marginal* velocity field, and the learned model converges to the same thing. Training becomes simple MSE regression, identical in complexity to diffusion training.

4. **The OT path** --- linear interpolation $x_t = (1-t)x_0 + tx_1$ with constant velocity $u_t = x_1 - x_0$ --- is the optimal choice. It gives straight conditional paths, minimal transport cost, and uniform information progression. The VP and VE paths from diffusion are special cases with suboptimal (curved) geometry.

5. **Rectified flows** (Liu et al., 2023) iteratively straighten the learned flow by retraining on coupled pairs generated by the previous model. Each reflow iteration reduces path curvature. In the limit, the paths are straight, and a single Euler step generates high-quality samples.

The mathematical core is beautiful in its simplicity: a straight line from noise to data, with a constant-velocity ODE connecting them. Everything else --- the conditional flow matching theorem, the reflow procedure, the connection to optimal transport --- exists to make this simple idea practical and provably correct.

This is why the industry switched. Not because of a marginal improvement in FID scores, but because flow matching does the same job with fewer steps, simpler math, simpler training, and simpler sampling. In engineering, simplicity is not a luxury. It is the goal.

---

## References

1. Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). *Flow Matching for Generative Modeling.* ICLR 2023.
2. Liu, X., Gong, C., & Liu, Q. (2023). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR 2023.
3. Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). *Neural Ordinary Differential Equations.* NeurIPS 2018.
4. Albergo, M. S. & Vanden-Eijnden, E. (2023). *Building Normalizing Flows with Stochastic Interpolants.* ICLR 2023.
5. Esser, P., Kulal, S., Blattmann, A., et al. (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.* ICML 2024.
6. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021.
