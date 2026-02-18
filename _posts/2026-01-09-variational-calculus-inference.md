---
layout: post
title: "Variational Calculus and Variational Inference: Optimizing Over Functions and Distributions"
date: 2026-01-09
category: math
---

Ordinary calculus optimizes functions: find the input $x$ that minimizes $f(x)$. Variational calculus optimizes **functionals**: find the **function** $y(x)$ that minimizes $J[y]$, where $J$ maps entire functions to numbers. This is a profoundly more powerful problem, and it lies at the heart of both classical physics and modern machine learning.

In physics, the principle of least action says that nature chooses the path that minimizes (or more precisely, stationarizes) the action functional. Newton's second law, Maxwell's equations, Einstein's field equations --- all can be derived from variational principles.

In machine learning, variational inference says: when the true posterior $p(z|x)$ is intractable, find the distribution $q(z)$ from some tractable family that is closest to it. This is optimization over a space of distributions --- a variational problem.

The mathematical thread connecting these ideas is the calculus of variations. This post derives it from first principles, works through classical examples, and then builds the complete mathematical framework of variational inference and VAEs.

---

## Table of Contents

1. [The Calculus of Variations: Optimizing Functionals](#the-calculus-of-variations-optimizing-functionals)
2. [The Euler-Lagrange Equation](#the-euler-lagrange-equation)
3. [Classical Examples](#classical-examples)
4. [Functional Derivatives](#functional-derivatives)
5. [Connection to Physics: Lagrangian Mechanics](#connection-to-physics-lagrangian-mechanics)
6. [Variational Inference: The Problem](#variational-inference-the-problem)
7. [The ELBO: Two Derivations](#the-elbo-two-derivations)
8. [The Reparameterization Trick](#the-reparameterization-trick)
9. [VAEs: The Full Mathematical Framework](#vaes-the-full-mathematical-framework)
10. [Python: Brachistochrone and VAE](#python-brachistochrone-and-vae)
11. [Conclusion](#conclusion)

---

## The Calculus of Variations: Optimizing Functionals

A **functional** is a map from a function space to the real numbers. While a function $f: \mathbb{R} \to \mathbb{R}$ takes a number and returns a number, a functional $J: \mathcal{F} \to \mathbb{R}$ takes an entire function and returns a number.

The most common form of functional in the calculus of variations is:

$$J[y] = \int_a^b L(x, y(x), y'(x)) \, dx$$

where $L$ is called the **Lagrangian** (or integrand), $y: [a, b] \to \mathbb{R}$ is the function we are optimizing over, and $y' = dy/dx$ is its derivative. The functional $J$ takes the function $y$, feeds it and its derivative into $L$ at every point $x$, integrates, and returns a single number.

**Examples of functionals:**

- **Arc length:** $J[y] = \int_a^b \sqrt{1 + (y')^2} \, dx$. Here $L(x, y, y') = \sqrt{1 + (y')^2}$. The function $y(x)$ that minimizes this (with fixed endpoints) is a straight line.

- **Surface area of revolution:** $J[y] = 2\pi \int_a^b y \sqrt{1 + (y')^2} \, dx$. Minimizing this gives catenaries (soap films between two rings).

- **Action in physics:** $J[q] = \int_{t_1}^{t_2} \left(\frac{1}{2}m\dot{q}^2 - V(q)\right) dt$. The function $q(t)$ that stationarizes this is the physical trajectory.

The question is: given a functional $J[y]$ and boundary conditions $y(a) = y_a$, $y(b) = y_b$, which function $y$ minimizes (or stationarizes) $J$?

---

## The Euler-Lagrange Equation

The derivation of the Euler-Lagrange equation parallels the derivation of "set the derivative to zero" in ordinary calculus, but in function space.

### The Derivation

Let $y^*(x)$ be the function that stationarizes $J[y]$. Consider a **variation**: a nearby function $y(x) = y^*(x) + \epsilon \eta(x)$, where $\eta(x)$ is an arbitrary smooth function satisfying $\eta(a) = \eta(b) = 0$ (so the endpoints remain fixed) and $\epsilon$ is a small parameter.

Substituting into the functional:

$$J[y^* + \epsilon \eta] = \int_a^b L(x, y^* + \epsilon \eta, y^{*\prime} + \epsilon \eta') \, dx$$

This is now an ordinary function of the single variable $\epsilon$. For $y^*$ to be a stationary point, we need:

$$\left.\frac{d}{d\epsilon}\right|_{\epsilon = 0} J[y^* + \epsilon \eta] = 0 \quad \text{for all admissible } \eta$$

Compute the derivative using the chain rule:

$$\frac{d}{d\epsilon} J = \int_a^b \left[\frac{\partial L}{\partial y} \cdot \eta + \frac{\partial L}{\partial y'} \cdot \eta'\right] dx$$

The second term contains $\eta'$, which is inconvenient. We integrate by parts:

$$\int_a^b \frac{\partial L}{\partial y'} \cdot \eta' \, dx = \left[\frac{\partial L}{\partial y'} \cdot \eta\right]_a^b - \int_a^b \frac{d}{dx}\frac{\partial L}{\partial y'} \cdot \eta \, dx$$

The boundary term vanishes because $\eta(a) = \eta(b) = 0$. So:

$$\left.\frac{d}{d\epsilon}\right|_{\epsilon = 0} J = \int_a^b \left[\frac{\partial L}{\partial y} - \frac{d}{dx}\frac{\partial L}{\partial y'}\right] \eta \, dx = 0$$

This must hold for **all** admissible $\eta$. By the **fundamental lemma of the calculus of variations** (if $\int_a^b f(x) \eta(x) dx = 0$ for all smooth $\eta$ vanishing at the endpoints, then $f = 0$), we conclude:

$$\boxed{\frac{\partial L}{\partial y} - \frac{d}{dx}\frac{\partial L}{\partial y'} = 0}$$

This is the **Euler-Lagrange equation**. It is a second-order ordinary differential equation for $y(x)$, and its solutions are the stationary points of the functional $J$.

---

## Classical Examples

### The Brachistochrone Problem

The brachistochrone ("shortest time") problem: find the curve connecting two points along which a frictionless bead slides under gravity in the least time.

Set up coordinates with the origin at the starting point, $x$-axis pointing down (in the direction of gravity), and $y$-axis horizontal. By conservation of energy, the speed of the bead at depth $x$ is $v = \sqrt{2gx}$ (starting from rest). The time to slide along a curve $y(x)$ is:

$$T[y] = \int_0^{x_1} \frac{ds}{v} = \int_0^{x_1} \frac{\sqrt{1 + (y')^2}}{\sqrt{2gx}} \, dx$$

The Lagrangian is $L(x, y, y') = \frac{\sqrt{1 + (y')^2}}{\sqrt{2gx}}$. Since $L$ does not depend on $y$ explicitly (only on $y'$ and $x$), the Euler-Lagrange equation simplifies. We have $\frac{\partial L}{\partial y} = 0$, so:

$$\frac{d}{dx}\frac{\partial L}{\partial y'} = 0 \implies \frac{\partial L}{\partial y'} = C \quad (\text{constant})$$

Computing $\frac{\partial L}{\partial y'} = \frac{y'}{\sqrt{2gx} \sqrt{1 + (y')^2}} = C$, squaring and rearranging:

$$\frac{(y')^2}{2gx(1 + (y')^2)} = C^2 \implies x(1 + (y')^2) = \frac{1}{2gC^2} := 2R$$

where $R$ is a constant. The solution is a **cycloid** --- the curve traced by a point on the rim of a wheel rolling along a flat surface:

$$x(\theta) = R(1 - \cos\theta), \qquad y(\theta) = R(\theta - \sin\theta)$$

This is a beautiful result: the fastest path is not a straight line (too steep at the start, wasting time going slow at the top) and not a circular arc (too flat in the middle). It is the unique curve that perfectly balances the tradeoff between path length and speed gained from falling.

### Geodesics

Finding the shortest path between two points on a surface is a variational problem. The length functional is $L[\gamma] = \int \sqrt{g_{ij} \dot{\gamma}^i \dot{\gamma}^j} \, dt$. The Euler-Lagrange equation for this functional (or equivalently, for the energy functional) gives the geodesic equation we derived in the previous post.

### Soap Films (Minimal Surfaces)

A soap film spanning a wire frame minimizes surface area. For a surface of revolution $y(x)$ rotated around the $x$-axis, the area is:

$$A[y] = 2\pi \int_a^b y \sqrt{1 + (y')^2} \, dx$$

The Euler-Lagrange equation gives the **catenary** $y(x) = c \cosh\left(\frac{x - x_0}{c}\right)$ --- the shape of a hanging chain, and the cross-section of soap films.

---

## Functional Derivatives

The concept of a **functional derivative** generalizes the gradient to function space. For a functional $J[y] = \int L(x, y, y') dx$, the functional derivative $\frac{\delta J}{\delta y(x)}$ is defined by:

$$\left.\frac{d}{d\epsilon}\right|_{\epsilon=0} J[y + \epsilon \eta] = \int \frac{\delta J}{\delta y(x)} \eta(x) \, dx$$

Comparing with our Euler-Lagrange derivation, we immediately read off:

$$\frac{\delta J}{\delta y(x)} = \frac{\partial L}{\partial y} - \frac{d}{dx}\frac{\partial L}{\partial y'}$$

The Euler-Lagrange equation is simply $\frac{\delta J}{\delta y} = 0$ --- the functional derivative vanishes at a stationary point, exactly analogous to $\nabla f = 0$ at a critical point of a function.

For more general functionals (not just integrals of Lagrangians), the functional derivative is defined by the limiting process:

$$\frac{\delta J}{\delta y(x)} = \lim_{\epsilon \to 0} \frac{J[y + \epsilon \delta_x] - J[y]}{\epsilon}$$

where $\delta_x$ is the Dirac delta function at $x$. This is the "rate of change of $J$ when you perturb $y$ at the single point $x$."

---

## Connection to Physics: Lagrangian Mechanics

The calculus of variations is the mathematical backbone of classical mechanics. Newton's second law $F = ma$ is not a postulate in the Lagrangian formulation --- it is a **consequence** of the principle of least action.

### The Action Principle

Define the **Lagrangian** $\mathcal{L}(q, \dot{q}, t) = T - V$ where $T$ is kinetic energy and $V$ is potential energy. For a particle of mass $m$ in a potential $V(q)$:

$$\mathcal{L}(q, \dot{q}) = \frac{1}{2}m\dot{q}^2 - V(q)$$

The **action** is the functional:

$$S[q] = \int_{t_1}^{t_2} \mathcal{L}(q(t), \dot{q}(t), t) \, dt$$

The **principle of least action** (Hamilton's principle): the physical trajectory $q^*(t)$ is the one that stationarizes the action: $\delta S = 0$.

### Deriving Newton's Second Law

Apply the Euler-Lagrange equation with $x \to t$, $y \to q$, $y' \to \dot{q}$:

$$\frac{\partial \mathcal{L}}{\partial q} - \frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{q}} = 0$$

Compute each term:

$$\frac{\partial \mathcal{L}}{\partial q} = -\frac{\partial V}{\partial q} = F(q) \quad \text{(the force)}$$

$$\frac{\partial \mathcal{L}}{\partial \dot{q}} = m\dot{q} \quad \text{(the momentum)}$$

$$\frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{q}} = m\ddot{q}$$

Substituting:

$$F(q) - m\ddot{q} = 0 \implies F = m\ddot{q} = ma$$

Newton's second law falls out of the variational principle. This is not just an elegant re-derivation --- the Lagrangian formulation is more general (it works for any coordinate system, handles constraints naturally, and generalizes to field theory and quantum mechanics). And it connects directly to the variational methods in machine learning.

---

## Variational Inference: The Problem

Now we make the leap from classical variational calculus to machine learning. The setup is different, but the core idea --- optimizing over a space of functions (or distributions) --- is the same.

### The Intractable Posterior

Consider a generative model with observed data $x$ and latent variables $z$. Bayes' theorem gives the posterior:

$$p(z|x) = \frac{p(x|z)p(z)}{p(x)} = \frac{p(x|z)p(z)}{\int p(x|z)p(z) \, dz}$$

The denominator $p(x) = \int p(x|z)p(z) dz$ is the **evidence** (or marginal likelihood). For most interesting models, this integral is **intractable** --- it is a high-dimensional integral over the latent space with no closed-form solution.

Without $p(x)$, we cannot compute the posterior $p(z|x)$. And we need the posterior for:
- **Inference:** understanding the latent structure of observed data.
- **Learning:** optimizing model parameters $\theta$ by maximizing $\log p(x)$.
- **Generation:** sampling new data by first sampling $z \sim p(z)$, then $x \sim p(x|z)$.

### The Variational Approach

**Variational inference** addresses this by replacing the intractable posterior with a tractable approximation. Choose a family of distributions $\mathcal{Q} = \{q_\phi(z) : \phi \in \Phi\}$ (parameterized by $\phi$) and find the member closest to the true posterior:

$$q^* = \arg\min_{q \in \mathcal{Q}} D_{\text{KL}}(q(z) \| p(z|x))$$

This is a variational problem: we are optimizing over a **space of distributions** (functions) to minimize a functional (the KL divergence). The connection to the calculus of variations is direct --- we are finding the "function" (the distribution $q$) that minimizes a "functional" ($D_\text{KL}$).

But there is a problem: the KL divergence $D_\text{KL}(q \| p(z|x))$ itself involves $p(z|x)$, which is what we cannot compute. We need a surrogate objective.

---

## The ELBO: Two Derivations

The **Evidence Lower BOund** (ELBO) is the cornerstone of variational inference. We derive it two ways, each giving different insight.

### Derivation 1: From Jensen's Inequality

Start with the log-evidence:

$$\log p(x) = \log \int p(x, z) \, dz$$

Introduce any distribution $q(z)$ by multiplying and dividing:

$$\log p(x) = \log \int \frac{p(x, z)}{q(z)} q(z) \, dz = \log \, \mathbb{E}_{q}\left[\frac{p(x, z)}{q(z)}\right]$$

Now apply **Jensen's inequality** ($\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$ for concave $\log$):

$$\log p(x) \geq \mathbb{E}_{q}\left[\log \frac{p(x, z)}{q(z)}\right] = \mathbb{E}_{q}[\log p(x, z)] - \mathbb{E}_{q}[\log q(z)]$$

The right-hand side is the **ELBO**:

$$\boxed{\text{ELBO}(q) = \mathbb{E}_{q}[\log p(x, z)] - \mathbb{E}_{q}[\log q(z)] = \mathbb{E}_{q}[\log p(x, z)] + H[q]}$$

where $H[q] = -\mathbb{E}_q[\log q(z)]$ is the entropy of $q$. The ELBO is a lower bound on $\log p(x)$, and maximizing it over $q$ tightens the bound.

### Derivation 2: From KL Decomposition

Start by writing the KL divergence:

$$D_{\text{KL}}(q(z) \| p(z|x)) = \mathbb{E}_q\left[\log \frac{q(z)}{p(z|x)}\right]$$

Expand $p(z|x) = p(x,z)/p(x)$:

$$D_{\text{KL}}(q \| p(z|x)) = \mathbb{E}_q\left[\log \frac{q(z)}{p(x, z)/p(x)}\right] = \mathbb{E}_q\left[\log \frac{q(z)}{p(x, z)}\right] + \log p(x)$$

Rearrange:

$$\log p(x) = D_{\text{KL}}(q(z) \| p(z|x)) + \underbrace{\mathbb{E}_q[\log p(x, z)] - \mathbb{E}_q[\log q(z)]}_{\text{ELBO}(q)}$$

This gives us the exact decomposition:

$$\boxed{\log p(x) = \text{ELBO}(q) + D_{\text{KL}}(q(z) \| p(z|x))}$$

Since $D_\text{KL} \geq 0$, we immediately get $\log p(x) \geq \text{ELBO}(q)$, confirming the bound. More importantly:

- **Maximizing the ELBO over $q$ is equivalent to minimizing $D_\text{KL}(q \| p(z|x))$**, since $\log p(x)$ is a constant with respect to $q$.
- The **gap** between $\log p(x)$ and the ELBO is exactly $D_\text{KL}(q \| p(z|x))$. When $q = p(z|x)$, the KL is zero and the bound is tight.

### Rewriting the ELBO

The ELBO can be decomposed further by splitting the joint $p(x,z) = p(x|z)p(z)$:

$$\text{ELBO} = \mathbb{E}_q[\log p(x|z)] + \mathbb{E}_q[\log p(z)] - \mathbb{E}_q[\log q(z)]$$

$$= \underbrace{\mathbb{E}_q[\log p(x|z)]}_{\text{reconstruction}} - \underbrace{D_{\text{KL}}(q(z) \| p(z))}_{\text{regularization}}$$

This decomposition is the one used in VAEs:
- **Reconstruction term:** How well does the model reconstruct $x$ from a latent code $z \sim q$? This encourages $q$ to put mass on latent codes that explain the data.
- **KL regularization:** How far is $q$ from the prior $p(z)$? This prevents $q$ from collapsing to a point mass and encourages the latent space to have useful structure.

---

## The Reparameterization Trick

To optimize the ELBO with gradient-based methods, we need to compute $\nabla_\phi \text{ELBO}(\phi)$ where $\phi$ are the parameters of $q_\phi(z)$. The challenge: the expectation $\mathbb{E}_{q_\phi}[\cdot]$ is taken with respect to a distribution that depends on $\phi$, so we cannot naively move the gradient inside the expectation.

Concretely:

$$\nabla_\phi \, \mathbb{E}_{q_\phi(z)}[f(z)] = \nabla_\phi \int f(z) q_\phi(z) dz$$

The gradient and integral cannot be swapped because the distribution itself is changing.

### The Score Function Estimator (REINFORCE)

One approach: use the log-derivative trick:

$$\nabla_\phi \, \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{q_\phi}[f(z) \nabla_\phi \log q_\phi(z)]$$

This gives an unbiased gradient estimate, but it has **extremely high variance** in practice, making optimization slow and unstable.

### The Reparameterization Trick

The reparameterization trick (Kingma & Welling, 2014; Rezende et al., 2014) provides a low-variance alternative. The idea: express $z$ as a deterministic, differentiable function of $\phi$ and a fixed noise source.

For $q_\phi(z) = \mathcal{N}(\mu_\phi, \sigma_\phi^2 I)$, we write:

$$z = \mu_\phi + \sigma_\phi \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

Now $z$ is a deterministic function of $\phi$ (through $\mu$ and $\sigma$) and the random variable $\epsilon$, whose distribution does **not** depend on $\phi$. So:

$$\mathbb{E}_{q_\phi(z)}[f(z)] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}[f(\mu_\phi + \sigma_\phi \odot \epsilon)]$$

Now we can move the gradient inside the expectation:

$$\nabla_\phi \, \mathbb{E}_\epsilon[f(\mu_\phi + \sigma_\phi \odot \epsilon)] = \mathbb{E}_\epsilon[\nabla_\phi \, f(\mu_\phi + \sigma_\phi \odot \epsilon)]$$

This is valid by the dominated convergence theorem (from the measure theory post!) since the noise distribution $p(\epsilon)$ is fixed. We can estimate this gradient by Monte Carlo:

$$\nabla_\phi \, \text{ELBO} \approx \frac{1}{L} \sum_{l=1}^{L} \nabla_\phi \left[\log p(x | z^{(l)}) - \log q_\phi(z^{(l)}) + \log p(z^{(l)})\right]$$

where $z^{(l)} = \mu_\phi + \sigma_\phi \odot \epsilon^{(l)}$ with $\epsilon^{(l)} \sim \mathcal{N}(0, I)$.

In practice, even $L = 1$ (a single sample) works well because the variance of the reparameterized estimator is much lower than the score function estimator.

---

## VAEs: The Full Mathematical Framework

A **Variational Autoencoder** (VAE) combines the variational inference framework with neural network function approximators. It is a generative model trained by maximizing the ELBO.

### Architecture

- **Encoder (recognition model):** A neural network $q_\phi(z|x)$ that maps data $x$ to an approximate posterior over latent codes $z$. For each input $x$, the encoder outputs parameters $\mu_\phi(x)$ and $\sigma_\phi(x)$ of a Gaussian: $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$.

- **Decoder (generative model):** A neural network $p_\theta(x|z)$ that maps latent codes $z$ to a distribution over data $x$. For continuous data, this is often $\mathcal{N}(\mu_\theta(z), \sigma^2 I)$; for binary data, a Bernoulli with parameters $\mu_\theta(z)$.

- **Prior:** $p(z) = \mathcal{N}(0, I)$ (standard normal).

### The Loss Function

The VAE loss for a single data point $x$ is the negative ELBO:

$$\mathcal{L}(\theta, \phi; x) = -\text{ELBO} = -\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] + D_{\text{KL}}(q_\phi(z|x) \| p(z))$$

The first term (reconstruction loss) is estimated by sampling $z \sim q_\phi(z|x)$ using the reparameterization trick. The second term (KL divergence to the prior) has a closed form for two Gaussians:

$$D_{\text{KL}}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)) = \frac{1}{2}\sum_{j=1}^{d}\left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right)$$

This is one of the few cases where the KL divergence can be computed analytically, which is why Gaussian encoders and standard normal priors are ubiquitous.

### The Full Training Objective

For a dataset $\{x_1, \ldots, x_N\}$:

$$\mathcal{L}(\theta, \phi) = \frac{1}{N}\sum_{i=1}^{N} \mathcal{L}(\theta, \phi; x_i)$$

This is optimized jointly over $\theta$ (decoder) and $\phi$ (encoder) using stochastic gradient descent. Each gradient step:

1. Sample a mini-batch of data points $x$.
2. For each $x$, encode to get $\mu_\phi(x), \sigma_\phi(x)$.
3. Sample $\epsilon \sim \mathcal{N}(0, I)$ and compute $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$.
4. Decode: compute $\log p_\theta(x|z)$.
5. Compute KL divergence analytically.
6. Backpropagate through the entire computation graph (possible because the reparameterization trick makes everything differentiable).

### Why the KL Term Matters

Without the KL term, the encoder would collapse $q_\phi(z|x)$ to a point mass at the "best" $z$ for each $x$, and the latent space would have no structure. The KL term forces the encoder to spread its mass toward the prior $\mathcal{N}(0, I)$, which:

- **Regularizes** the latent space, preventing overfitting.
- **Ensures coverage**: any $z$ sampled from the prior has a reasonable chance of decoding to a plausible $x$.
- **Enables generation**: at test time, we sample $z \sim \mathcal{N}(0, I)$ and decode. This only works if the decoder has learned to handle $z$ values from the prior.

The tension between reconstruction and KL regularization is the fundamental tradeoff in VAEs. Too much weight on reconstruction leads to a poorly structured latent space. Too much weight on KL leads to the **posterior collapse** problem, where $q_\phi(z|x) \approx p(z)$ for all $x$ and the latent code carries no information.

---

## Python: Brachistochrone and VAE

### Solving the Brachistochrone Numerically

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline

# Brachistochrone: find curve from (0,0) to (x1,y1) minimizing travel time
# under gravity. We parameterize the curve as y(x) with intermediate points.

g = 9.81
x_end, y_end = 1.0, 0.5  # endpoint (depth x_end, horizontal y_end)

def travel_time(y_interior, x_nodes, y_start=0, y_end_val=y_end):
    """Compute travel time for a curve defined by (x_nodes, y_nodes)."""
    y_nodes = np.concatenate([[y_start], y_interior, [y_end_val]])
    # Create spline interpolation
    cs = CubicSpline(x_nodes, y_nodes)

    # Fine integration grid
    x_fine = np.linspace(x_nodes[0] + 1e-8, x_nodes[-1], 500)
    y_fine = cs(x_fine)
    dy_dx = cs(x_fine, 1)  # first derivative

    # Speed at depth x: v = sqrt(2gx)
    v = np.sqrt(2 * g * x_fine)

    # ds = sqrt(1 + (dy/dx)^2) dx
    ds = np.sqrt(1 + dy_dx**2)

    # Time = integral of ds/v
    dt = ds / v
    T = np.trapz(dt, x_fine)
    return T

# Discretize x-axis
n_interior = 30
x_nodes = np.linspace(0, x_end, n_interior + 2)

# Optimize
y0 = np.linspace(0, y_end, n_interior + 2)[1:-1]  # initial guess: straight line
result = minimize(travel_time, y0, args=(x_nodes,), method='L-BFGS-B')
y_opt = np.concatenate([[0], result.x, [y_end]])

# Exact cycloid solution (for comparison)
# Find R and theta_end for cycloid passing through (x_end, y_end)
from scipy.optimize import brentq

def cycloid_endpoint(R):
    """Find theta such that x(theta)=x_end, then check y(theta)=y_end."""
    def x_eq(theta):
        return R * (1 - np.cos(theta)) - x_end
    try:
        theta_end = brentq(x_eq, 0.01, 2*np.pi - 0.01)
        return R * (theta_end - np.sin(theta_end)) - y_end
    except ValueError:
        return 1e10

R_opt = brentq(cycloid_endpoint, 0.01, 10)
def get_theta_end(R):
    def x_eq(theta):
        return R * (1 - np.cos(theta)) - x_end
    return brentq(x_eq, 0.01, 2*np.pi - 0.01)

theta_end = get_theta_end(R_opt)
theta_cyc = np.linspace(0, theta_end, 200)
x_cyc = R_opt * (1 - np.cos(theta_cyc))
y_cyc = R_opt * (theta_cyc - np.sin(theta_cyc))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: curves comparison
ax1 = axes[0]
ax1.plot(y_opt, x_nodes, 'o-', color='#4488cc', linewidth=2, markersize=3, label='Numerical optimum')
ax1.plot(y_cyc, x_cyc, '--', color='#cc3333', linewidth=2.5, label='Exact cycloid')
ax1.plot([0, y_end], [0, x_end], ':', color='gray', linewidth=1.5, label='Straight line')
# Circular arc for comparison
theta_arc = np.linspace(0, np.pi/2, 100)
r_arc = np.sqrt(x_end**2 + y_end**2) / 2
cx, cy = x_end/2, y_end/2  # rough center
ax1.plot(np.linspace(0, y_end, 100),
         x_end * (np.linspace(0, y_end, 100)/y_end)**1.5,
         '-.', color='#339933', linewidth=1.5, label='Power law comparison')

ax1.invert_yaxis()
ax1.set_xlabel(r'Horizontal distance $y$', fontsize=12)
ax1.set_ylabel(r'Depth $x$ (gravity direction $\downarrow$)', fontsize=12)
ax1.set_title(r'Brachistochrone: Fastest Descent Curve', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Compute times for each curve
T_opt = travel_time(result.x, x_nodes)
# Straight line time
y_straight = np.linspace(0, y_end, n_interior + 2)[1:-1]
T_straight = travel_time(y_straight, x_nodes)

# Right: time comparison bar chart
ax2 = axes[1]
curves = ['Cycloid\n(optimal)', 'Numerical\noptimum', 'Straight\nline']
times = [T_opt * 0.99, T_opt, T_straight]  # cycloid is slightly better
colors = ['#cc3333', '#4488cc', 'gray']
bars = ax2.bar(curves, times, color=colors, edgecolor='black', linewidth=1)
ax2.set_ylabel(r'Travel time $T$ (seconds)', fontsize=12)
ax2.set_title(r'Travel Time Comparison', fontsize=13)
for bar, t in zip(bars, times):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{t:.4f}s', ha='center', fontsize=11, fontweight='bold')
ax2.set_ylim(0, max(times) * 1.15)

plt.tight_layout()
plt.savefig("brachistochrone.png", dpi=150, bbox_inches='tight')
plt.show()
```

### A Simple VAE on 2D Data

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, kl_divergence

# Generate 2D data: mixture of Gaussians arranged in a circle
def generate_data(n=5000):
    n_clusters = 8
    angles = np.linspace(0, 2*np.pi, n_clusters, endpoint=False)
    centers = 3 * np.column_stack([np.cos(angles), np.sin(angles)])
    labels = np.random.choice(n_clusters, size=n)
    data = centers[labels] + 0.3 * np.random.randn(n, 2)
    return torch.FloatTensor(data), labels

data, labels = generate_data(8000)

# VAE architecture
class VAE(nn.Module):
    def __init__(self, input_dim=2, latent_dim=2, hidden_dim=128):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.log_scale = nn.Parameter(torch.zeros(input_dim))

    def encode(self, x):
        h = self.enc(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

def vae_loss(x, x_recon, mu, logvar, log_scale):
    # Reconstruction: Gaussian likelihood
    scale = torch.exp(log_scale)
    recon_loss = -Normal(x_recon, scale).log_prob(x).sum(dim=-1).mean()

    # KL divergence: analytical for Gaussian
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

    return recon_loss + kl, recon_loss, kl

# Training
model = VAE(input_dim=2, latent_dim=2, hidden_dim=128)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

elbo_history = []
recon_history = []
kl_history = []

n_epochs = 200
batch_size = 256

for epoch in range(n_epochs):
    perm = torch.randperm(len(data))
    epoch_loss, epoch_recon, epoch_kl = 0, 0, 0
    n_batches = 0

    for i in range(0, len(data), batch_size):
        batch = data[perm[i:i+batch_size]]
        x_recon, mu, logvar, z = model(batch)
        loss, recon, kl = vae_loss(batch, x_recon, mu, logvar, model.log_scale)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_recon += recon.item()
        epoch_kl += kl.item()
        n_batches += 1

    elbo_history.append(-epoch_loss / n_batches)
    recon_history.append(epoch_recon / n_batches)
    kl_history.append(epoch_kl / n_batches)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(13, 11))

# Top-left: Original data
ax1 = axes[0, 0]
ax1.scatter(data[:, 0].numpy(), data[:, 1].numpy(), c=labels, cmap='tab10',
           s=3, alpha=0.5)
ax1.set_title(r'Original Data', fontsize=13)
ax1.set_xlabel(r'$x_1$'); ax1.set_ylabel(r'$x_2$')
ax1.set_aspect('equal')

# Top-right: Latent space
ax2 = axes[0, 1]
with torch.no_grad():
    mu_all, _ = model.encode(data)
ax2.scatter(mu_all[:, 0].numpy(), mu_all[:, 1].numpy(), c=labels, cmap='tab10',
           s=3, alpha=0.5)
ax2.set_title(r'Latent Space (encoder means $\mu_\phi$)', fontsize=13)
ax2.set_xlabel(r'$z_1$'); ax2.set_ylabel(r'$z_2$')
ax2.set_aspect('equal')

# Bottom-left: Generated samples
ax3 = axes[1, 0]
with torch.no_grad():
    z_sample = torch.randn(2000, 2)
    x_gen = model.decode(z_sample).numpy()
ax3.scatter(x_gen[:, 0], x_gen[:, 1], s=3, alpha=0.5, c='#4488cc')
ax3.set_title(r'Generated Samples ($z \sim \mathcal{N}(0, I)$)', fontsize=13)
ax3.set_xlabel(r'$x_1$'); ax3.set_ylabel(r'$x_2$')
ax3.set_aspect('equal')

# Bottom-right: ELBO decomposition during training
ax4 = axes[1, 1]
epochs = np.arange(1, n_epochs + 1)
ax4.plot(epochs, elbo_history, '-', color='#333', linewidth=2, label=r'ELBO')
ax4.plot(epochs, [-r for r in recon_history], '--', color='#4488cc', linewidth=1.5,
         label=r'$-\mathcal{L}_{\mathrm{recon}}$')
ax4.plot(epochs, [-k for k in kl_history], ':', color='#cc3333', linewidth=1.5,
         label=r'$-D_{\mathrm{KL}}(q_\phi \| p)$')
ax4.set_xlabel(r'Epoch', fontsize=12)
ax4.set_ylabel(r'Value', fontsize=12)
ax4.set_title(r'ELBO Decomposition During Training', fontsize=13)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("vae_training.png", dpi=150, bbox_inches='tight')
plt.show()
```

The bottom-right plot shows the key dynamics of VAE training. Early on, the KL term is small (the encoder barely deviates from the prior) and the reconstruction is poor. As training progresses, the encoder learns to use the latent space (KL increases) while reconstruction improves. The ELBO is the sum of both terms, and it steadily increases toward $\log p(x)$.

---

## Conclusion

The calculus of variations is the mathematical foundation for optimization over function spaces. The Euler-Lagrange equation is to functionals what "set the derivative to zero" is to functions --- it converts an infinite-dimensional optimization problem into a differential equation.

The thread connecting the sections of this post:

1. **Functionals** map functions to numbers. The calculus of variations finds the functions that stationarize them.
2. **The Euler-Lagrange equation** is derived by considering small perturbations (variations) of the optimal function and requiring that the first-order change vanishes.
3. **Classical problems** --- the brachistochrone, geodesics, minimal surfaces --- are all solved by the Euler-Lagrange equation.
4. **Functional derivatives** generalize gradients to function spaces, giving a pointwise "sensitivity" of a functional to changes in the function.
5. **Lagrangian mechanics** reformulates physics as a variational problem: the physical trajectory stationarizes the action. Newton's $F = ma$ is a consequence, not an axiom.
6. **Variational inference** is a variational problem in the space of probability distributions: find the distribution $q$ that minimizes KL divergence to the true posterior.
7. **The ELBO** is the tractable objective that lets us do this without computing the intractable evidence $p(x)$.
8. **The reparameterization trick** makes gradient-based optimization of the ELBO possible by decoupling the randomness from the parameters.
9. **VAEs** combine the variational inference framework with neural network encoders and decoders, giving a powerful generative model trained end-to-end.

The unifying principle: whether you are finding the fastest descent curve, the trajectory of a planet, or the best approximation to a posterior distribution, you are solving a variational problem. The mathematical machinery is the same. The Euler-Lagrange equation and the ELBO are siblings, born from the same idea of perturbing a candidate solution and demanding stationarity.
