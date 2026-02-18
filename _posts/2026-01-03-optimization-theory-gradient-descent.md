---
layout: post
title: "Optimization Theory: From Gradient Descent to Adam and Why Convergence Happens"
date: 2026-01-03
category: math
---

Training a neural network means finding the parameters that minimize a loss function. That sentence is easy to write and extraordinarily difficult to execute. The loss landscape of a modern video generation model has billions of dimensions, is non-convex, is riddled with saddle points, and is only accessible through noisy stochastic estimates. Yet we reliably find good minima. Understanding why --- and understanding what each optimizer is actually doing mathematically --- is essential for anyone who trains models or diagnoses training failures.

This post builds optimization theory from scratch. We define what optimization means, establish the conditions for minima, derive gradient descent from Taylor expansion, prove convergence for convex functions, then systematically build up the modern optimizer stack: SGD, momentum, RMSprop, and Adam. Every equation is derived and every design choice is motivated.

---

## Table of Contents

1. [What Optimization Means](#what-optimization-means)
2. [Convex vs Non-Convex Functions](#convex-vs-non-convex-functions)
3. [Conditions for Minima](#conditions-for-minima)
4. [Gradient Descent from Taylor Expansion](#gradient-descent-from-taylor-expansion)
5. [Convergence Proof for Convex Functions](#convergence-proof-for-convex-functions)
6. [Stochastic Gradient Descent](#stochastic-gradient-descent)
7. [Momentum: The Ball Rolling Downhill](#momentum-the-ball-rolling-downhill)
8. [RMSprop: Adaptive Learning Rates](#rmsprop-adaptive-learning-rates)
9. [Adam: Combining Momentum and RMSprop](#adam-combining-momentum-and-rmsprop)
10. [Learning Rate Schedules](#learning-rate-schedules)
11. [Python Visualizations](#python-visualizations)
12. [Conclusion](#conclusion)

---

## What Optimization Means

An **optimization problem** asks: given a function $f: \mathbb{R}^d \to \mathbb{R}$, find the point $\theta^*$ that minimizes it:

$$\theta^* = \arg\min_{\theta \in \mathbb{R}^d} f(\theta)$$

In machine learning, $\theta$ is the vector of all model parameters (weights and biases), $f(\theta)$ is the **loss function** (measuring how poorly the model performs on the training data), and $d$ is the number of parameters --- often in the billions for modern models.

We call $f(\theta)$ the **objective function** or **loss landscape**. The word "landscape" is deliberate: if $d = 2$, you can visualize $f$ as a 3D surface, with hills, valleys, ridges, and saddle points. Optimization is the process of navigating this landscape to find the lowest valley.

A few important definitions:

- **Global minimum**: $\theta^*$ such that $f(\theta^*) \leq f(\theta)$ for all $\theta$. The absolute lowest point.
- **Local minimum**: $\theta^*$ such that $f(\theta^*) \leq f(\theta)$ for all $\theta$ in some neighborhood of $\theta^*$. A valley that may not be the deepest.
- **Saddle point**: A point where the gradient is zero but the point is neither a local maximum nor a local minimum. Nearby, the function curves up in some directions and down in others. Think of a mountain pass.

<svg viewBox="0 0 700 300" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">Loss Landscape Features</text>
  <!-- Curve representing loss landscape -->
  <path d="M30,200 Q80,60 150,170 Q200,240 250,180 Q280,140 320,160 Q370,190 400,100 Q430,30 460,100 Q500,190 540,180 Q580,170 620,150 Q660,135 680,140" stroke="#2563eb" stroke-width="2.5" fill="none"/>
  <!-- Global min marker -->
  <circle cx="430" cy="30" r="5" fill="#16a34a"/>
  <text x="430" y="55" text-anchor="middle" font-size="11" fill="#16a34a" font-weight="bold">Global min</text>
  <!-- Local min markers -->
  <circle cx="150" cy="170" r="4" fill="#dc2626"/>
  <text x="150" y="195" text-anchor="middle" font-size="10" fill="#dc2626">Local min</text>
  <circle cx="250" cy="180" r="4" fill="#dc2626"/>
  <text x="250" y="205" text-anchor="middle" font-size="10" fill="#dc2626">Local min</text>
  <!-- Saddle point -->
  <circle cx="320" cy="160" r="4" fill="#9333ea"/>
  <text x="320" y="145" text-anchor="middle" font-size="10" fill="#9333ea">Saddle point</text>
  <!-- Axes -->
  <line x1="30" y1="260" x2="680" y2="260" stroke="#999" stroke-width="1"/>
  <text x="350" y="285" text-anchor="middle" font-size="12" fill="#666">Parameter space θ</text>
  <text x="15" y="150" text-anchor="middle" font-size="12" fill="#666" transform="rotate(-90, 15, 150)">Loss f(θ)</text>
</svg>

---

## Convex vs Non-Convex Functions

**Convexity** is the single most important structural property a function can have for optimization. A function $f: \mathbb{R}^d \to \mathbb{R}$ is **convex** if, for all $\theta_1, \theta_2 \in \mathbb{R}^d$ and all $\lambda \in [0, 1]$:

$$f(\lambda \theta_1 + (1-\lambda)\theta_2) \leq \lambda f(\theta_1) + (1-\lambda)f(\theta_2)$$

Geometrically: if you draw a line segment between any two points on the graph of $f$, the function lies on or below that line segment. The function curves upward (or is flat), never downward. A bowl is convex; a hill is not; a landscape with multiple valleys is not.

Equivalently, for twice-differentiable functions: $f$ is convex if and only if its **Hessian matrix** $H = \nabla^2 f$ is positive semidefinite everywhere. The Hessian is the matrix of all second partial derivatives:

$$H_{ij} = \frac{\partial^2 f}{\partial \theta_i \partial \theta_j}$$

"Positive semidefinite" means $v^T H v \geq 0$ for all vectors $v$ --- the function curves upward in every direction.

A stronger condition is **$\mu$-strong convexity**: there exists $\mu > 0$ such that

$$f(\lambda \theta_1 + (1-\lambda)\theta_2) \leq \lambda f(\theta_1) + (1-\lambda)f(\theta_2) - \frac{\mu}{2}\lambda(1-\lambda)\|\theta_1 - \theta_2\|^2$$

This means the function is at least as curved as a quadratic with curvature $\mu$. Strong convexity guarantees a unique global minimum and faster convergence.

**Why convexity matters:** For convex functions, every local minimum is a global minimum. There are no suboptimal valleys to get trapped in. Gradient descent is guaranteed to find the optimum.

**The reality:** Neural network loss functions are emphatically not convex. They have exponentially many local minima and saddle points. Yet optimization works in practice. Understanding the convex case gives us the theoretical foundation, and the non-convex case builds on it with additional insights about why modern architectures and optimizers succeed despite the lack of convexity guarantees.

---

## Conditions for Minima

### First-Order Necessary Condition

If $\theta^*$ is a local minimum of a differentiable function $f$, then:

$$\nabla f(\theta^*) = 0$$

The gradient (vector of all partial derivatives) must vanish. This is the multivariable generalization of "set the derivative to zero and solve."

Why? If $\nabla f(\theta^*) \neq 0$, then the direction $-\nabla f(\theta^*)$ is a **descent direction**: moving a small step in that direction decreases $f$. So $\theta^*$ could not be a minimum.

Formally: by Taylor expansion, $f(\theta^* + \epsilon d) \approx f(\theta^*) + \epsilon \nabla f(\theta^*)^T d$. If we choose $d = -\nabla f(\theta^*)$, the second term becomes $-\epsilon \|\nabla f(\theta^*)\|^2 < 0$, so $f$ decreases.

Points where $\nabla f = 0$ are called **critical points** or **stationary points**. But not all critical points are minima --- they could be maxima or saddle points.

### Second-Order Sufficient Conditions

To distinguish, we look at the Hessian $H = \nabla^2 f(\theta^*)$:

- If $H$ is **positive definite** (all eigenvalues strictly positive): $\theta^*$ is a **strict local minimum**. The function curves upward in every direction.
- If $H$ is **negative definite** (all eigenvalues strictly negative): $\theta^*$ is a **strict local maximum**.
- If $H$ has both positive and negative eigenvalues: $\theta^*$ is a **saddle point**. The function curves up in some directions and down in others.

The derivation uses the second-order Taylor expansion:

$$f(\theta^* + \delta) \approx f(\theta^*) + \underbrace{\nabla f(\theta^*)^T \delta}_{= 0} + \frac{1}{2}\delta^T H \delta$$

Since the gradient is zero at a critical point, the behavior near $\theta^*$ is determined by the quadratic form $\delta^T H \delta$. If $H$ is positive definite, this quadratic form is positive for all $\delta \neq 0$, meaning $f(\theta^* + \delta) > f(\theta^*)$ for all small perturbations --- a minimum.

In high-dimensional non-convex optimization (like training neural networks), most critical points are saddle points, not local minima. At a random critical point in $d$ dimensions, each Hessian eigenvalue is positive or negative with roughly equal probability, so the chance of all $d$ eigenvalues being positive is approximately $2^{-d}$ --- astronomically small. Gradient descent naturally escapes saddle points (it follows the negative gradient, which has a component along the negative curvature direction), which is one reason it works well in practice.

---

## Gradient Descent from Taylor Expansion

Now we derive the gradient descent update rule. The question: given a current position $\theta_t$, where should we move to decrease $f$?

The first-order Taylor expansion of $f$ around $\theta_t$ is:

$$f(\theta_t + \delta) \approx f(\theta_t) + \nabla f(\theta_t)^T \delta$$

We want to choose $\delta$ to make $f(\theta_t + \delta)$ as small as possible. But the linear approximation is only valid for small $\delta$, so we add a constraint: $\|\delta\| \leq \eta$ for some step size $\eta > 0$.

The inner product $\nabla f(\theta_t)^T \delta$ is minimized when $\delta$ points in the opposite direction of $\nabla f(\theta_t)$. By the Cauchy-Schwarz inequality:

$$\nabla f(\theta_t)^T \delta \geq -\|\nabla f(\theta_t)\| \cdot \|\delta\|$$

with equality when $\delta = -\frac{\eta}{\|\nabla f(\theta_t)\|}\nabla f(\theta_t)$.

The standard gradient descent update absorbs the normalization into the step size, giving:

$$\boxed{\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t)}$$

where $\eta > 0$ is the **learning rate**. This is not an arbitrary algorithm --- it is the optimal direction for a linear approximation to $f$, with the learning rate controlling how far we trust that approximation.

The learning rate $\eta$ plays a critical role:

- **Too small**: Convergence is painfully slow. The algorithm takes tiny steps and wastes computation.
- **Too large**: The algorithm overshoots, jumping past the minimum. If $\eta$ is large enough, the iterates can diverge entirely.
- **Just right**: The algorithm converges efficiently. For a quadratic function $f(\theta) = \frac{1}{2}\theta^T A \theta$, the optimal learning rate is $\eta = 2/(\lambda_{\max} + \lambda_{\min})$, where $\lambda_{\max}$ and $\lambda_{\min}$ are the largest and smallest eigenvalues of $A$.

The **condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$ of the Hessian determines how difficult optimization is. A large condition number means the landscape is shaped like a long, narrow valley --- the gradient points mostly across the valley (toward the nearest wall) rather than along it (toward the minimum). This is why gradient descent can be slow on ill-conditioned problems.

---

## Convergence Proof for Convex Functions

Let us prove that gradient descent converges for convex, $L$-smooth functions. A function is **$L$-smooth** if its gradient is Lipschitz continuous:

$$\|\nabla f(\theta_1) - \nabla f(\theta_2)\| \leq L \|\theta_1 - \theta_2\| \quad \text{for all } \theta_1, \theta_2$$

This bounds how fast the gradient can change. Equivalently, the Hessian eigenvalues are bounded above by $L$.

**Theorem:** For a convex, $L$-smooth function $f$, gradient descent with step size $\eta = 1/L$ satisfies:

$$f(\theta_T) - f(\theta^*) \leq \frac{L \|\theta_0 - \theta^*\|^2}{2T}$$

This says the suboptimality decreases as $O(1/T)$ --- to get within $\epsilon$ of the optimum, you need $T = O(1/\epsilon)$ iterations.

**Proof sketch.** The $L$-smoothness condition implies the **descent lemma**:

$$f(\theta_{t+1}) \leq f(\theta_t) + \nabla f(\theta_t)^T(\theta_{t+1} - \theta_t) + \frac{L}{2}\|\theta_{t+1} - \theta_t\|^2$$

Substituting the gradient descent update $\theta_{t+1} - \theta_t = -\eta \nabla f(\theta_t)$ with $\eta = 1/L$:

$$f(\theta_{t+1}) \leq f(\theta_t) - \frac{1}{L}\|\nabla f(\theta_t)\|^2 + \frac{L}{2} \cdot \frac{1}{L^2}\|\nabla f(\theta_t)\|^2 = f(\theta_t) - \frac{1}{2L}\|\nabla f(\theta_t)\|^2$$

So each step decreases the function value by at least $\frac{1}{2L}\|\nabla f(\theta_t)\|^2$. This is called **sufficient decrease**.

By convexity: $f(\theta^*) \geq f(\theta_t) + \nabla f(\theta_t)^T(\theta^* - \theta_t)$, which rearranges to:

$$\nabla f(\theta_t)^T(\theta_t - \theta^*) \leq f(\theta_t) - f(\theta^*)$$

Combining and telescoping over $T$ iterations (using the identity $\|\theta_{t+1} - \theta^*\|^2 = \|\theta_t - \theta^*\|^2 - 2\eta\nabla f(\theta_t)^T(\theta_t - \theta^*) + \eta^2\|\nabla f(\theta_t)\|^2$), we obtain:

$$\sum_{t=0}^{T-1}[f(\theta_t) - f(\theta^*)] \leq \frac{L\|\theta_0 - \theta^*\|^2}{2}$$

Since $f(\theta_T) \leq f(\theta_t)$ for all $t \leq T$ (by sufficient decrease), we have $T[f(\theta_T) - f(\theta^*)] \leq \frac{L\|\theta_0 - \theta^*\|^2}{2}$, giving the result. $\blacksquare$

For **strongly convex** functions with parameter $\mu$, the convergence is exponential:

$$f(\theta_T) - f(\theta^*) \leq \left(1 - \frac{\mu}{L}\right)^T [f(\theta_0) - f(\theta^*)]$$

The ratio $\kappa = L/\mu$ (condition number) determines the convergence rate. Large $\kappa$ means slow convergence --- the ill-conditioned valley problem.

---

## Stochastic Gradient Descent

In practice, computing the full gradient $\nabla f(\theta) = \frac{1}{N}\sum_{i=1}^{N}\nabla f_i(\theta)$ requires passing through the entire dataset of $N$ examples. For large datasets, this is prohibitively expensive.

**Stochastic Gradient Descent (SGD)** replaces the full gradient with the gradient of a single randomly chosen example (or a small mini-batch):

$$\theta_{t+1} = \theta_t - \eta \, \nabla f_{i_t}(\theta_t)$$

where $i_t$ is sampled uniformly from $\{1, \ldots, N\}$.

Why does this work? Because the stochastic gradient is an **unbiased estimator** of the true gradient:

$$\mathbb{E}_{i_t}[\nabla f_{i_t}(\theta_t)] = \frac{1}{N}\sum_{i=1}^{N}\nabla f_i(\theta_t) = \nabla f(\theta_t)$$

On average, the stochastic gradient points in the right direction. Individual updates are noisy --- they may point away from the optimum --- but the noise averages out over many steps.

The noise is characterized by the **variance** of the stochastic gradient:

$$\sigma^2 = \mathbb{E}\left[\|\nabla f_{i_t}(\theta_t) - \nabla f(\theta_t)\|^2\right]$$

Using mini-batches of size $B$ reduces the variance by a factor of $B$: $\sigma^2_B = \sigma^2 / B$. This is the variance reduction from averaging independent random variables.

SGD converges, but with a subtle difference: you must **decay the learning rate** over time. With constant learning rate, the noise prevents convergence to the exact minimum --- the iterates fluctuate in a neighborhood whose size is proportional to $\eta\sigma^2$. With a decaying schedule like $\eta_t = \eta_0 / \sqrt{t}$, convergence to the optimum is guaranteed (for convex functions) at rate $O(1/\sqrt{T})$.

The noise in SGD is not purely a nuisance --- it can help escape sharp local minima and find flatter minima that generalize better. This is an active area of research, but the intuition is that the noise scale $\eta\sigma^2/B$ acts as an implicit regularizer.

---

## Momentum: The Ball Rolling Downhill

Standard gradient descent has a problem: in long, narrow valleys (high condition number), the gradient points mostly across the valley, causing oscillation. The trajectory zigzags back and forth instead of progressing along the valley floor.

**Momentum** fixes this by maintaining a running average of past gradients, accumulating velocity in consistent directions and damping oscillations in inconsistent directions.

The physical analogy: imagine a ball rolling down a hilly landscape. The ball has mass, so it builds up speed when moving consistently in one direction and does not immediately reverse when it encounters a small uphill. Friction gradually slows it down.

### Polyak Momentum (Heavy Ball Method)

Polyak (1964) proposed:

$$v_{t+1} = \beta v_t + \nabla f(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta \, v_{t+1}$$

where $\beta \in [0, 1)$ is the **momentum coefficient** (typically 0.9) and $v_t$ is the **velocity** vector, initialized to zero.

The velocity is an exponentially weighted moving average of past gradients. Expanding the recursion:

$$v_{t+1} = \nabla f(\theta_t) + \beta \nabla f(\theta_{t-1}) + \beta^2 \nabla f(\theta_{t-2}) + \cdots$$

Gradients from the recent past contribute most; the contribution of a gradient $k$ steps ago is weighted by $\beta^k$. The effective window length is approximately $1/(1-\beta)$: with $\beta = 0.9$, the optimizer "remembers" roughly the last 10 gradients.

Why does this help? In the zigzag scenario:
- The component of the gradient **across** the valley oscillates in sign (positive, negative, positive, ...). These components cancel out in the running average.
- The component of the gradient **along** the valley is consistent (always the same sign). These components accumulate in the running average.

The result: the velocity vector builds up along the valley floor and the oscillations are damped. Momentum effectively reduces the condition number of the problem.

### Nesterov Momentum

Nesterov (1983) proposed a subtle but important modification: compute the gradient at the **lookahead position** $\theta_t - \eta\beta v_t$ rather than the current position:

$$v_{t+1} = \beta v_t + \nabla f(\theta_t - \eta\beta v_t)$$
$$\theta_{t+1} = \theta_t - \eta \, v_{t+1}$$

The idea: since we know the momentum will carry us to approximately $\theta_t - \eta\beta v_t$, we should evaluate the gradient there rather than at our current position. This "look-ahead" provides a correction that improves convergence. For convex functions, Nesterov momentum achieves the optimal convergence rate of $O(1/T^2)$, compared to $O(1/T)$ for standard gradient descent.

---

## RMSprop: Adaptive Learning Rates

Momentum addresses the direction of gradient steps. **RMSprop** (Hinton, 2012, introduced in a lecture, never formally published) addresses a different problem: the **scale** of the gradient varies dramatically across different parameters.

Consider a neural network where some parameters have consistently large gradients and others have consistently small gradients. A single global learning rate is suboptimal: too large for the high-gradient parameters (overshooting) and too small for the low-gradient parameters (slow progress).

RMSprop maintains a per-parameter estimate of the gradient magnitude and divides by it:

$$s_{t+1} = \beta \, s_t + (1-\beta) \, [\nabla f(\theta_t)]^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_{t+1}} + \epsilon} \odot \nabla f(\theta_t)$$

Here $s_t$ is the **exponentially weighted moving average of squared gradients** (element-wise), $\beta$ is typically 0.999, $\epsilon \approx 10^{-8}$ is a small constant to prevent division by zero, and $\odot$ denotes element-wise operations.

What is $\sqrt{s_{t+1}}$ estimating? It is an exponential moving average estimate of the **root mean square (RMS)** of recent gradients for each parameter. Dividing by this normalizes each parameter's update by its typical gradient scale.

Parameters with large gradients get divided by a large number (small effective learning rate). Parameters with small gradients get divided by a small number (large effective learning rate). This automatic per-parameter learning rate adaptation is what makes RMSprop so effective.

The intuition: RMSprop performs an approximate second-order optimization by estimating the diagonal of the Hessian from the gradient magnitudes. The diagonal Hessian entries $\partial^2 f / \partial \theta_i^2$ determine the curvature along each parameter axis. Large curvature means the gradient changes rapidly --- you should take smaller steps. RMSprop's denominator $\sqrt{s_t}$ approximates $\sqrt{\text{diag}(H)}$, so dividing by it is like preconditioning with the inverse square root of the diagonal Hessian.

---

## Adam: Combining Momentum and RMSprop

**Adam** (Adaptive Moment Estimation, Kingma & Ba, 2015) combines momentum (first moment estimation) with RMSprop (second moment estimation):

$$m_{t+1} = \beta_1 m_t + (1-\beta_1) \nabla f(\theta_t) \qquad \text{(first moment estimate)}$$
$$v_{t+1} = \beta_2 v_t + (1-\beta_2) [\nabla f(\theta_t)]^2 \qquad \text{(second moment estimate)}$$

$$\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}} \qquad \text{(bias-corrected first moment)}$$
$$\hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}} \qquad \text{(bias-corrected second moment)}$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon} \odot \hat{m}_{t+1}$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

### Deriving the Bias Correction

The bias correction terms are crucial and their derivation is elegant. Consider the second moment estimate $v_t$ initialized to $v_0 = 0$. Unrolling the recursion:

$$v_t = (1-\beta_2)\sum_{k=0}^{t-1}\beta_2^{t-1-k}[\nabla f(\theta_k)]^2$$

Taking the expectation (assuming the true second moment $\mathbb{E}[g_k^2]$ is approximately constant):

$$\mathbb{E}[v_t] \approx \mathbb{E}[g^2] \cdot (1-\beta_2)\sum_{k=0}^{t-1}\beta_2^{t-1-k} = \mathbb{E}[g^2] \cdot (1-\beta_2) \cdot \frac{1 - \beta_2^t}{1 - \beta_2} = \mathbb{E}[g^2](1 - \beta_2^t)$$

So $\mathbb{E}[v_t] = \mathbb{E}[g^2](1 - \beta_2^t)$, which is biased toward zero, especially for small $t$. Dividing by $(1 - \beta_2^t)$ corrects this:

$$\mathbb{E}\left[\frac{v_t}{1 - \beta_2^t}\right] = \mathbb{E}[g^2]$$

The same argument applies to $m_t$ with $\beta_1$. Without bias correction, the first few steps of Adam would use severely underestimated moment estimates, leading to excessively large update steps (since we divide by $\sqrt{v_t}$, and $v_t \approx 0$ initially).

With $\beta_2 = 0.999$, the bias is significant for the first approximately 1000 steps: $\beta_2^{1000} = 0.999^{1000} \approx 0.368$, so the bias correction factor is $1/(1 - 0.368) \approx 1.58$. It takes thousands of steps for the bias to become negligible.

### Why Adam Works So Well in Practice

Adam combines the best of both worlds:

1. **Momentum** ($m_t$): smooths out noisy gradients and accelerates through consistent gradient directions.
2. **Adaptive scaling** ($v_t$): normalizes the update magnitude per-parameter, handling different scales across parameters automatically.
3. **Bias correction**: ensures good behavior from the very first step, not just after warmup.

The effective step size for each parameter is approximately $\eta \cdot m / \sqrt{v} \approx \eta \cdot \text{sign}(g)$ when the gradient is consistent (since $m \approx g$ and $\sqrt{v} \approx |g|$). This means Adam effectively takes steps of size $\pm\eta$ regardless of the gradient magnitude --- a form of sign gradient descent with adaptive magnitude.

---

## Learning Rate Schedules

Even with adaptive optimizers, the learning rate $\eta$ is the most important hyperparameter. Modern training typically uses a **learning rate schedule** that varies $\eta$ over the course of training.

### Warmup

**Problem:** At the start of training, the model's parameters are randomly initialized, the gradients are large and noisy, and the adaptive moment estimates ($m_t$, $v_t$) have not yet converged. Large learning rates at this stage can cause instability.

**Solution:** Start with a very small learning rate and linearly increase it over the first $T_{\text{warmup}}$ steps:

$$\eta_t = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}}, \qquad t \leq T_{\text{warmup}}$$

This gives the moment estimates time to stabilize before taking aggressive steps. For large-batch training and transformer architectures, warmup is essentially mandatory --- without it, training often diverges.

### Cosine Annealing

After warmup, the learning rate is gradually decreased. **Cosine annealing** (Loshchilov & Hutter, 2017) uses a cosine schedule:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi \cdot (t - T_{\text{warmup}})}{T_{\text{total}} - T_{\text{warmup}}}\right)\right)$$

Why cosine? Several reasons:

1. It starts with a plateau near $\eta_{\max}$ (the cosine is flat near its peak), allowing aggressive exploration early on.
2. It decreases smoothly toward $\eta_{\min}$, with the fastest decrease in the middle of training.
3. It ends with a plateau near $\eta_{\min}$ (the cosine is flat near its trough), allowing fine-tuning at the end.

Compared to step decay (sudden drops in learning rate), cosine annealing avoids the abrupt transitions that can destabilize training. Compared to linear decay, it spends more time at both high and low learning rates, which empirically produces better results.

### Why Schedules Help

The intuition: early in training, you want a large learning rate to explore the loss landscape broadly and escape bad regions. Late in training, you want a small learning rate to settle into a precise minimum. The schedule implements this tradeoff automatically.

More formally: the noise from SGD creates a fluctuation scale proportional to $\eta$. A large $\eta$ means large fluctuations, which helps cross barriers between basins but prevents settling into a sharp minimum. Decreasing $\eta$ over time gradually reduces the fluctuation scale, eventually allowing convergence.

---

## Python Visualizations

### Visualization 1: Gradient Descent on a 2D Loss Landscape

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a 2D loss function: Rosenbrock-like with adjustable condition number
def loss_fn(theta):
    x, y = theta
    return (1 - x)**2 + 10 * (y - x**2)**2

def grad_fn(theta):
    x, y = theta
    dx = -2*(1-x) + 10*2*(y - x**2)*(-2*x)
    dy = 10*2*(y - x**2)
    return np.array([dx, dy])

# Create contour data
x = np.linspace(-2, 2, 200)
y = np.linspace(-1, 3, 200)
X, Y = np.meshgrid(x, y)
Z = (1 - X)**2 + 10*(Y - X**2)**2

# Run gradient descent with different learning rates
def run_gd(lr, n_steps=200, start=np.array([-1.5, 2.5])):
    path = [start.copy()]
    theta = start.copy()
    for _ in range(n_steps):
        g = grad_fn(theta)
        theta = theta - lr * g
        path.append(theta.copy())
        if np.any(np.abs(theta) > 10):
            break
    return np.array(path)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
lrs = [0.001, 0.005, 0.02]
titles = [r'$\eta = 0.001$ (too small)', r'$\eta = 0.005$ (good)', r'$\eta = 0.02$ (too large)']

for ax, lr, title in zip(axes, lrs, titles):
    ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.6)
    path = run_gd(lr)
    ax.plot(path[:, 0], path[:, 1], 'r.-', markersize=3, linewidth=0.8, alpha=0.8)
    ax.plot(path[0, 0], path[0, 1], 'go', markersize=8, label='Start')
    ax.plot(1, 1, 'r*', markersize=12, label=r'Optimum $(1,1)$')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.legend(fontsize=8)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 3)

plt.suptitle(r'Gradient Descent: Effect of Learning Rate $\eta$ on Convergence', fontsize=14)
plt.tight_layout()
plt.savefig('gd_learning_rate.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Visualization 2: Comparing GD vs Momentum vs Adam

```python
import numpy as np
import matplotlib.pyplot as plt

# Ill-conditioned quadratic: f(x,y) = 50*x^2 + y^2 (condition number = 50)
def loss_fn(theta):
    return 50*theta[0]**2 + theta[1]**2

def grad_fn(theta):
    return np.array([100*theta[0], 2*theta[1]])

start = np.array([1.0, 1.0])
n_steps = 100

# Vanilla GD
def run_gd(lr):
    path = [start.copy()]
    theta = start.copy()
    for _ in range(n_steps):
        theta = theta - lr * grad_fn(theta)
        path.append(theta.copy())
    return np.array(path)

# GD with momentum
def run_momentum(lr, beta=0.9):
    path = [start.copy()]
    theta = start.copy()
    v = np.zeros(2)
    for _ in range(n_steps):
        g = grad_fn(theta)
        v = beta * v + g
        theta = theta - lr * v
        path.append(theta.copy())
    return np.array(path)

# Adam
def run_adam(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
    path = [start.copy()]
    theta = start.copy()
    m = np.zeros(2)
    v = np.zeros(2)
    for t in range(1, n_steps + 1):
        g = grad_fn(theta)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(theta.copy())
    return np.array(path)

# Create contour plot
x = np.linspace(-1.2, 1.2, 200)
y = np.linspace(-1.2, 1.2, 200)
X, Y = np.meshgrid(x, y)
Z = 50*X**2 + Y**2

fig, ax = plt.subplots(figsize=(10, 8))
ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)

path_gd = run_gd(lr=0.005)
path_mom = run_momentum(lr=0.002, beta=0.9)
path_adam = run_adam(lr=0.1)

ax.plot(path_gd[:, 0], path_gd[:, 1], 'r.-', markersize=2, linewidth=1,
        label=r'GD ($\eta=0.005$)', alpha=0.8)
ax.plot(path_mom[:, 0], path_mom[:, 1], 'b.-', markersize=2, linewidth=1,
        label=r'Momentum ($\eta=0.002$, $\beta=0.9$)', alpha=0.8)
ax.plot(path_adam[:, 0], path_adam[:, 1], 'g.-', markersize=2, linewidth=1,
        label=r'Adam ($\eta=0.1$)', alpha=0.8)

ax.plot(start[0], start[1], 'ko', markersize=10, label='Start')
ax.plot(0, 0, 'r*', markersize=15, label=r'Optimum $\theta^*$')
ax.set_xlabel(r'$\theta_1$ (high curvature)')
ax.set_ylabel(r'$\theta_2$ (low curvature)')
ax.set_title(r'Optimizer Comparison on Ill-Conditioned Quadratic ($\kappa = 50$)', fontsize=13)
ax.legend(fontsize=10)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

Notice how vanilla GD oscillates wildly along the high-curvature ($\theta_1$) direction while making slow progress along the low-curvature ($\theta_2$) direction. Momentum dampens the oscillation and accelerates along $\theta_2$. Adam handles both directions efficiently by adapting the learning rate per parameter.

### Visualization 3: Learning Rate Effect on Convergence

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple 1D convex function: f(x) = x^2
def f(x): return x**2
def grad_f(x): return 2*x

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: trajectories for different learning rates
lrs = [0.01, 0.1, 0.5, 0.9, 1.05]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

ax = axes[0]
x_plot = np.linspace(-3, 3, 100)
ax.plot(x_plot, f(x_plot), 'k-', linewidth=2, alpha=0.3)

for lr, color in zip(lrs, colors):
    x = 2.5
    xs = [x]
    for _ in range(30):
        x = x - lr * grad_f(x)
        xs.append(x)
        if abs(x) > 10:
            break
    ax.plot(range(len(xs)), [f(xi) for xi in xs], '.-', color=color,
            label=rf'$\eta = {lr}$', markersize=4, linewidth=1)

ax.set_xlabel(r'Iteration')
ax.set_ylabel(r'$f(\theta)$')
ax.set_title(r'Convergence for Different Learning Rates')
ax.legend(fontsize=9)
ax.set_ylim(-0.5, 10)
ax.grid(True, alpha=0.3)

# Right: cosine annealing schedule
ax2 = axes[1]
T_warmup = 1000
T_total = 10000
eta_max = 1e-3
eta_min = 1e-5

t = np.arange(T_total)
eta = np.zeros(T_total)

# Warmup phase
warmup_mask = t < T_warmup
eta[warmup_mask] = eta_max * t[warmup_mask] / T_warmup

# Cosine annealing phase
cos_mask = ~warmup_mask
t_cos = t[cos_mask] - T_warmup
T_cos = T_total - T_warmup
eta[cos_mask] = eta_min + 0.5*(eta_max - eta_min)*(1 + np.cos(np.pi * t_cos / T_cos))

ax2.plot(t, eta * 1000, 'b-', linewidth=1.5)  # Scale for readability
ax2.axvline(x=T_warmup, color='r', linestyle='--', alpha=0.5, label='End of warmup')
ax2.set_xlabel(r'Training Step')
ax2.set_ylabel(r'Learning Rate $\eta$ ($\times 10^{-3}$)')
ax2.set_title(r'Cosine Annealing with Warmup')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lr_schedules.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Conclusion

Optimization is the engine of deep learning. The key ideas, in summary:

1. **Gradient descent** follows the steepest descent direction, derived from the first-order Taylor expansion. The learning rate controls the trust region.

2. **Convergence is guaranteed** for convex, smooth functions at rate $O(1/T)$, and $O(\exp(-T/\kappa))$ for strongly convex functions.

3. **SGD** replaces the full gradient with a stochastic estimate, trading accuracy per step for dramatically reduced cost per step. The noise is unbiased and can help generalization.

4. **Momentum** accumulates velocity in consistent gradient directions and dampens oscillations, effectively reducing the condition number of the problem.

5. **RMSprop** adapts the learning rate per-parameter by dividing by the RMS of recent gradients, implementing approximate diagonal second-order optimization.

6. **Adam** combines momentum and RMSprop with bias correction, producing an optimizer that works well out of the box across a wide range of problems.

7. **Learning rate schedules** (warmup + cosine annealing) manage the exploration-exploitation tradeoff over the course of training: explore broadly early, refine precisely late.

Every training run you launch --- whether it is fine-tuning a LoRA adapter or training a video generation model from scratch --- is executing these algorithms. Understanding them at this level lets you diagnose training failures (divergence means learning rate too high, slow progress means learning rate too low or poor conditioning), choose optimizer hyperparameters with principled reasoning rather than blind search, and understand why certain training recipes work.
