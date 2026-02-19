---
layout: post
title: "Optimal Transport for Video: Displacement Interpolation, Wasserstein Geodesics, and Motion as Transport"
date: 2026-02-28
category: math
---

When you watch a video of a ball rolling across a table, you perceive **motion** --- mass being transported from one location to another. Not a fade. Not a dissolve. A physical displacement. The mathematical framework that formalizes this is **optimal transport (OT)**: given two distributions of mass, find the cheapest way to move one into the other.

Optimal transport is not just an abstract curiosity. It is the theoretical foundation of **flow matching** (the training paradigm behind Stable Diffusion 3, Flux, and most modern generative models), the correct way to interpolate between video frames (displacement interpolation vs cross-fading), and the natural metric for comparing probability distributions (the Wasserstein distance used in WGANs).

This post builds optimal transport from the Monge problem (1781) through the Kantorovich relaxation, derives the Wasserstein distance and the Brenier theorem, introduces displacement interpolation and explains why it is the right way to interpolate frames, derives the Benamou-Brenier dynamic formulation, and connects everything to flow matching and video generation.

---

## Table of Contents

1. [The Optimal Transport Problem](#the-optimal-transport-problem)
2. [Wasserstein Distances](#wasserstein-distances)
3. [The Brenier Theorem](#the-brenier-theorem)
4. [Displacement Interpolation](#displacement-interpolation)
5. [The Benamou-Brenier Formula](#the-benamou-brenier-formula)
6. [Displacement Interpolation as Frame Interpolation](#displacement-interpolation-as-frame-interpolation)
7. [Entropic Optimal Transport and Sinkhorn](#entropic-optimal-transport-and-sinkhorn)
8. [OT Conditional Paths in Flow Matching](#ot-conditional-paths-in-flow-matching)
9. [Wasserstein GANs](#wasserstein-gans)
10. [Python: Sinkhorn OT and Displacement Interpolation](#python-sinkhorn-ot-and-displacement-interpolation)

---

## The Optimal Transport Problem

### Monge's Formulation (1781)

Gaspard Monge asked: given a pile of sand (source distribution \(\mu\)) and a hole to fill (target distribution \(\nu\)), what is the cheapest way to move the sand? The cost of moving a grain from position \(\mathbf{x}\) to position \(\mathbf{y}\) is \(c(\mathbf{x}, \mathbf{y})\) (typically the squared Euclidean distance \(c(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|^2\)).

A **transport map** \(T: \mathbb{R}^d \to \mathbb{R}^d\) pushes each point \(\mathbf{x}\) to \(T(\mathbf{x})\). The constraint is that the pushed-forward distribution \(T_\# \mu\) must equal \(\nu\):

$$T_\# \mu = \nu \quad \Leftrightarrow \quad \nu(B) = \mu(T^{-1}(B)) \text{ for all measurable sets } B$$

**Monge's problem:**

$$\inf_{T: T_\# \mu = \nu} \int c(\mathbf{x}, T(\mathbf{x})) \, d\mu(\mathbf{x})$$

### Why Monge Fails

Monge's formulation requires a **deterministic map**: each grain of sand goes to exactly one destination. This is problematic when \(\mu\) is a point mass and \(\nu\) is spread out --- you cannot split a point mass using a map. More precisely, if \(\mu = \delta_{\mathbf{x}_0}\) (all mass at one point) and \(\nu\) has support on multiple points, no transport map \(T\) satisfies \(T_\# \mu = \nu\).

---

## Wasserstein Distances

### Kantorovich Relaxation

Leonid Kantorovich (1942) relaxed Monge's problem by allowing mass to **split**. Instead of a deterministic map, use a **transport plan** (or coupling) \(\pi \in \Gamma(\mu, \nu)\) --- a joint distribution on \(\mathbb{R}^d \times \mathbb{R}^d\) with marginals \(\mu\) and \(\nu\):

$$\Gamma(\mu, \nu) = \{\pi \in \mathcal{P}(\mathbb{R}^d \times \mathbb{R}^d) : \pi(\cdot, \mathbb{R}^d) = \mu, \, \pi(\mathbb{R}^d, \cdot) = \nu\}$$

The value \(\pi(\mathbf{x}, \mathbf{y})\) represents how much mass is moved from \(\mathbf{x}\) to \(\mathbf{y}\).

**Kantorovich's problem:**

$$W_p^p(\mu, \nu) = \inf_{\pi \in \Gamma(\mu, \nu)} \int \|\mathbf{x} - \mathbf{y}\|^p \, d\pi(\mathbf{x}, \mathbf{y})$$

The **\(p\)-Wasserstein distance** is:

$$W_p(\mu, \nu) = \left(\inf_{\pi \in \Gamma(\mu, \nu)} \int \|\mathbf{x} - \mathbf{y}\|^p \, d\pi(\mathbf{x}, \mathbf{y})\right)^{1/p}$$

**\(W_1\)** is the **Earth Mover's Distance**: the minimum total distance that mass must travel.

**\(W_2\)** (quadratic cost) is the most mathematically rich. It has a unique optimal transport plan (under mild conditions), a beautiful geometric structure, and direct connections to fluid mechanics.

### Properties

- \(W_p\) is a metric on the space of probability distributions with finite \(p\)-th moments
- \(W_p\) metrizes weak convergence plus convergence of \(p\)-th moments
- \(W_p\) is sensitive to the **geometry** of the underlying space (unlike KL divergence, which depends only on density ratios)

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">Monge Map vs Kantorovich Coupling</text>

  <!-- Monge: deterministic map -->
  <text x="175" y="55" text-anchor="middle" font-size="12" font-weight="bold" fill="#4fc3f7">Monge: Transport Map T</text>
  <circle cx="80" cy="120" r="8" fill="#E53935" opacity="0.7"/>
  <circle cx="80" cy="170" r="8" fill="#E53935" opacity="0.7"/>
  <circle cx="80" cy="220" r="8" fill="#E53935" opacity="0.7"/>
  <circle cx="270" cy="100" r="8" fill="#66bb6a" opacity="0.7"/>
  <circle cx="270" cy="160" r="8" fill="#66bb6a" opacity="0.7"/>
  <circle cx="270" cy="220" r="8" fill="#66bb6a" opacity="0.7"/>
  <line x1="88" y1="120" x2="260" y2="100" stroke="#4fc3f7" stroke-width="1.5"/>
  <line x1="88" y1="170" x2="260" y2="220" stroke="#4fc3f7" stroke-width="1.5"/>
  <line x1="88" y1="220" x2="260" y2="160" stroke="#4fc3f7" stroke-width="1.5"/>
  <text x="175" y="260" text-anchor="middle" font-size="10" fill="#999">Each point → one destination</text>

  <!-- Kantorovich: coupling -->
  <text x="525" y="55" text-anchor="middle" font-size="12" font-weight="bold" fill="#FF9800">Kantorovich: Coupling π</text>
  <circle cx="430" cy="120" r="8" fill="#E53935" opacity="0.7"/>
  <circle cx="430" cy="170" r="8" fill="#E53935" opacity="0.7"/>
  <circle cx="430" cy="220" r="8" fill="#E53935" opacity="0.7"/>
  <circle cx="620" cy="100" r="8" fill="#66bb6a" opacity="0.7"/>
  <circle cx="620" cy="160" r="8" fill="#66bb6a" opacity="0.7"/>
  <circle cx="620" cy="220" r="8" fill="#66bb6a" opacity="0.7"/>
  <!-- Split connections -->
  <line x1="438" y1="120" x2="610" y2="100" stroke="#FF9800" stroke-width="2" opacity="0.7"/>
  <line x1="438" y1="120" x2="610" y2="160" stroke="#FF9800" stroke-width="1" opacity="0.4"/>
  <line x1="438" y1="170" x2="610" y2="160" stroke="#FF9800" stroke-width="1.5" opacity="0.6"/>
  <line x1="438" y1="170" x2="610" y2="220" stroke="#FF9800" stroke-width="1" opacity="0.4"/>
  <line x1="438" y1="220" x2="610" y2="220" stroke="#FF9800" stroke-width="2" opacity="0.7"/>
  <text x="525" y="260" text-anchor="middle" font-size="10" fill="#999">Mass can split across destinations</text>
</svg>

---

## The Brenier Theorem

For the quadratic cost \(c(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|^2\), there is a unique optimal transport map (under mild conditions), and it has a beautiful structure.

**Theorem (Brenier, 1991).** If \(\mu\) is absolutely continuous (no point masses), then the optimal transport map from \(\mu\) to \(\nu\) with quadratic cost is the gradient of a convex function:

$$T(\mathbf{x}) = \nabla \phi(\mathbf{x})$$

where \(\phi: \mathbb{R}^d \to \mathbb{R}\) is a convex function (called the **Brenier potential** or **Kantorovich potential**).

**Implications:**

1. The optimal map pushes each point in the gradient direction of a convex potential. Since the gradient of a convex function is a monotone map (it does not "cross" transport paths), the optimal transport avoids path crossings.

2. The convex function \(\phi\) satisfies the **Monge-Ampere equation**:

$$\det(\nabla^2 \phi(\mathbf{x})) = \frac{d\mu}{d\nu}(\nabla \phi(\mathbf{x}))$$

This PDE relates the curvature of the potential to the density ratio.

3. For the special case of two Gaussians \(\mu = \mathcal{N}(\mathbf{m}_0, \Sigma_0)\) and \(\nu = \mathcal{N}(\mathbf{m}_1, \Sigma_1)\), the optimal map is affine:

$$T(\mathbf{x}) = \mathbf{m}_1 + \Sigma_0^{-1/2}(\Sigma_0^{1/2} \Sigma_1 \Sigma_0^{1/2})^{1/2} \Sigma_0^{-1/2}(\mathbf{x} - \mathbf{m}_0)$$

And the Wasserstein-2 distance has a closed form (the **Bures metric**):

$$W_2^2(\mu, \nu) = \|\mathbf{m}_0 - \mathbf{m}_1\|^2 + \text{tr}\!\left(\Sigma_0 + \Sigma_1 - 2(\Sigma_0^{1/2} \Sigma_1 \Sigma_0^{1/2})^{1/2}\right)$$

---

## Displacement Interpolation

Given distributions \(\mu_0\) and \(\mu_1\) and an optimal transport map \(T\), the **displacement interpolation** (McCann, 1997) at time \(t \in [0, 1]\) is:

$$\mu_t = ((1-t)\text{Id} + tT)_\# \mu_0$$

Each point \(\mathbf{x}\) in \(\mu_0\) moves linearly toward its destination \(T(\mathbf{x})\) in \(\mu_1\):

$$\mathbf{x}_t = (1-t)\mathbf{x} + t \cdot T(\mathbf{x})$$

The distribution \(\mu_t\) is the distribution of \(\mathbf{x}_t\).

### Why Displacement Interpolation Is Special

Displacement interpolation is the **geodesic** in the Wasserstein-2 space. It is the shortest path between \(\mu_0\) and \(\mu_1\) in the space of probability distributions, measured by the Wasserstein-2 distance:

$$W_2(\mu_0, \mu_t) = t \cdot W_2(\mu_0, \mu_1), \quad W_2(\mu_t, \mu_1) = (1-t) \cdot W_2(\mu_0, \mu_1)$$

### Comparison with Linear Interpolation

The **linear (mixture) interpolation** between densities is:

$$\rho_t^{\text{linear}} = (1-t) \rho_0 + t \rho_1$$

This is a **cross-fade**: at each point in space, the density is a weighted average of the two distributions. For two separated Gaussian blobs, linear interpolation creates a bimodal distribution at \(t = 0.5\) --- the mass exists at both locations simultaneously, like a ghost.

**Displacement interpolation** instead moves the mass physically from one location to the other. At \(t = 0.5\), the blob is halfway between the two positions --- a single, coherent blob, not a ghostly superposition.

For video: cross-fading between frames produces ghosting artifacts (transparent overlapping images). Displacement interpolation produces actual motion --- the content physically moves from the first frame's position to the second frame's position.

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">Linear Interpolation vs Displacement Interpolation</text>

  <!-- Linear interpolation (top row) -->
  <text x="80" y="55" font-size="11" fill="#E53935" font-weight="bold">Cross-fade (linear):</text>
  <!-- t=0 -->
  <ellipse cx="140" cy="100" rx="30" ry="20" fill="#4fc3f7" opacity="0.5"/>
  <text x="140" y="130" text-anchor="middle" font-size="9" fill="#999">t=0</text>
  <!-- t=0.5 -->
  <ellipse cx="300" cy="100" rx="30" ry="20" fill="#4fc3f7" opacity="0.25"/>
  <ellipse cx="400" cy="100" rx="30" ry="20" fill="#4fc3f7" opacity="0.25"/>
  <text x="350" y="130" text-anchor="middle" font-size="9" fill="#E53935">t=0.5 (ghosts!)</text>
  <!-- t=1 -->
  <ellipse cx="560" cy="100" rx="30" ry="20" fill="#4fc3f7" opacity="0.5"/>
  <text x="560" y="130" text-anchor="middle" font-size="9" fill="#999">t=1</text>

  <!-- Displacement interpolation (bottom row) -->
  <text x="80" y="165" font-size="11" fill="#66bb6a" font-weight="bold">Displacement (OT):</text>
  <!-- t=0 -->
  <ellipse cx="140" cy="210" rx="30" ry="20" fill="#66bb6a" opacity="0.5"/>
  <text x="140" y="240" text-anchor="middle" font-size="9" fill="#999">t=0</text>
  <!-- t=0.5 -->
  <ellipse cx="350" cy="210" rx="30" ry="20" fill="#66bb6a" opacity="0.5"/>
  <text x="350" y="240" text-anchor="middle" font-size="9" fill="#66bb6a">t=0.5 (moved!)</text>
  <!-- t=1 -->
  <ellipse cx="560" cy="210" rx="30" ry="20" fill="#66bb6a" opacity="0.5"/>
  <text x="560" y="240" text-anchor="middle" font-size="9" fill="#999">t=1</text>
  <!-- Motion arrow -->
  <line x1="170" y1="210" x2="320" y2="210" stroke="#66bb6a" stroke-width="1" stroke-dasharray="4,3"/>
  <line x1="380" y1="210" x2="530" y2="210" stroke="#66bb6a" stroke-width="1" stroke-dasharray="4,3"/>
</svg>

---

## The Benamou-Brenier Formula

The **Benamou-Brenier formula** (2000) provides a dynamic formulation of optimal transport. Instead of finding a static transport plan, find a **velocity field** \(\mathbf{v}(\mathbf{x}, t)\) that moves the mass from \(\mu_0\) to \(\mu_1\) along the cheapest path.

The density \(\rho(\mathbf{x}, t)\) must satisfy the **continuity equation** (conservation of mass):

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0$$

with boundary conditions \(\rho(\cdot, 0) = \mu_0\) and \(\rho(\cdot, 1) = \mu_1\).

The Benamou-Brenier formula says:

$$W_2^2(\mu_0, \mu_1) = \inf_{(\rho, \mathbf{v})} \int_0^1 \int_{\mathbb{R}^d} \rho(\mathbf{x}, t) \|\mathbf{v}(\mathbf{x}, t)\|^2 \, d\mathbf{x} \, dt$$

The cost being minimized is the total **kinetic energy** of the transport. The optimal velocity field has constant speed along each particle's path (straight lines at constant velocity), which is precisely the displacement interpolation.

This is the fluid mechanics interpretation: \(W_2^2\) is the minimum kinetic energy required to morph one density into another while conserving mass.

---

## Displacement Interpolation as Frame Interpolation

The connection to video is direct. Consider two video frames as distributions of "visual content" in pixel space. Frame interpolation asks: what should the intermediate frame at time \(t\) look like?

**Cross-fade** (linear interpolation of pixel values): \(I_t = (1-t) I_0 + t I_1\). This produces ghosting --- both frames visible simultaneously, semi-transparent.

**Displacement interpolation** (OT-based): move the content from its position in \(I_0\) to its position in \(I_1\) along straight paths. At time \(t\), each pixel is at the linearly interpolated position. This produces natural-looking motion without ghosting.

In practice, this requires knowing the optical flow (the transport map between frames). Methods like FILM and RIFE estimate bidirectional flow and use it for frame interpolation, which is conceptually an approximation to displacement interpolation.

---

## Entropic Optimal Transport and Sinkhorn

Computing exact optimal transport is expensive: \(O(n^3 \log n)\) for discrete distributions with \(n\) support points (via linear programming). **Entropic regularization** makes it dramatically faster.

Add an entropy term to the Kantorovich problem:

$$W_\epsilon(\mu, \nu) = \inf_{\pi \in \Gamma(\mu, \nu)} \int c(\mathbf{x}, \mathbf{y}) \, d\pi(\mathbf{x}, \mathbf{y}) - \epsilon H(\pi)$$

where \(H(\pi) = -\int \pi \log \pi\) is the entropy of the coupling. The parameter \(\epsilon > 0\) controls the regularization strength. As \(\epsilon \to 0\), we recover the exact OT solution.

The entropy penalty encourages the coupling to be "spread out" (less deterministic). The regularized problem has a unique solution of the form:

$$\pi^*_{ij} = a_i K_{ij} b_j$$

where \(K_{ij} = \exp(-c_{ij}/\epsilon)\) is the Gibbs kernel and \(a_i, b_j\) are scaling factors determined by the marginal constraints.

### The Sinkhorn Algorithm

The scaling factors are found by alternating:

$$a_i \leftarrow \frac{\mu_i}{\sum_j K_{ij} b_j}, \quad b_j \leftarrow \frac{\nu_j}{\sum_i a_i K_{ij}}$$

This is **Sinkhorn iteration** (also called the IPFP algorithm). Each iteration enforces one marginal constraint. After \(L\) iterations, the coupling approximately satisfies both marginals.

**Complexity:** Each iteration is a matrix-vector product \(O(n^2)\), and \(O(1/\epsilon)\) iterations typically suffice. Total: \(O(n^2/\epsilon)\), which is dramatically faster than the \(O(n^3)\) exact solver for moderate \(\epsilon\).

---

## OT Conditional Paths in Flow Matching

The connection between optimal transport and flow matching is the central reason OT matters for modern video generation.

In the [flow matching post](/2026/02/18/flow-matching-rectified-flows.html), we derived that flow matching trains a velocity field to transport noise \(\mathbf{x}_0 \sim \mathcal{N}(0, I)\) to data \(\mathbf{x}_1 \sim p_{\text{data}}\) along straight paths:

$$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t \mathbf{x}_1$$

This is exactly the **displacement interpolation** between a point mass at \(\mathbf{x}_0\) and a point mass at \(\mathbf{x}_1\). The velocity field is:

$$\mathbf{v}_t(\mathbf{x}_t) = \mathbf{x}_1 - \mathbf{x}_0$$

constant along the path (straight line, constant speed).

The **conditional flow matching** objective:

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\!\left[\|\mathbf{v}_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0)\|^2\right]$$

trains the model to predict the OT velocity field. This is why flow matching produces straighter generation paths than diffusion (which follows curved SDE trajectories): it is explicitly trained to follow the OT geodesic.

The **OT conditional path** is the cheapest (minimum kinetic energy) way to transport each noise-data pair. By the Benamou-Brenier formula, this minimizes \(\int \|\mathbf{v}\|^2 dt\) per pair, which is achieved by constant-velocity straight lines.

---

## Wasserstein GANs

**WGAN** (Arjovsky et al., 2017) uses the Wasserstein-1 distance as the generator's training objective:

$$\min_G W_1(p_{\text{data}}, p_G)$$

The Kantorovich-Rubinstein duality provides a computable form:

$$W_1(\mu, \nu) = \sup_{\|f\|_L \leq 1} \left[\mathbb{E}_{\mathbf{x} \sim \mu}[f(\mathbf{x})] - \mathbb{E}_{\mathbf{y} \sim \nu}[f(\mathbf{y})]\right]$$

where the supremum is over all 1-Lipschitz functions \(f\). The **critic** (discriminator) approximates this optimal \(f\).

**Why \(W_1\) is better than KL/JS for GANs:**

KL and JS divergences are infinite (or saturated) when the supports of \(p_{\text{data}}\) and \(p_G\) do not overlap --- which is common early in training when the generator is poor. \(W_1\) provides meaningful gradients even for non-overlapping distributions, because it measures the physical distance between them.

The Lipschitz constraint is enforced via weight clipping (original WGAN), gradient penalty (WGAN-GP), or spectral normalization.

---

## Python: Sinkhorn OT and Displacement Interpolation

```python
import numpy as np
import matplotlib.pyplot as plt

def sinkhorn(a, b, C, epsilon=0.1, n_iter=100):
    """Sinkhorn algorithm for entropic optimal transport.

    Args:
        a: Source distribution (n,)
        b: Target distribution (m,)
        C: Cost matrix (n, m)
        epsilon: Regularization strength
        n_iter: Number of Sinkhorn iterations

    Returns:
        Transport plan (n, m)
    """
    K = np.exp(-C / epsilon)
    u = np.ones_like(a)
    v = np.ones_like(b)

    for _ in range(n_iter):
        u = a / (K @ v + 1e-30)
        v = b / (K.T @ u + 1e-30)

    return np.diag(u) @ K @ np.diag(v)

def create_distributions():
    """Create two 2D distributions for OT demonstration."""
    np.random.seed(42)
    n = 200

    # Source: cluster on the left
    source = np.column_stack([
        np.random.randn(n) * 0.5 - 2,
        np.random.randn(n) * 0.8
    ])

    # Target: two clusters on the right
    target_1 = np.column_stack([
        np.random.randn(n // 2) * 0.4 + 2,
        np.random.randn(n // 2) * 0.4 + 1
    ])
    target_2 = np.column_stack([
        np.random.randn(n // 2) * 0.4 + 2,
        np.random.randn(n // 2) * 0.4 - 1
    ])
    target = np.vstack([target_1, target_2])

    return source, target

def displacement_interpolation(source, target, plan, t):
    """Compute displacement interpolation at time t."""
    n = len(source)
    # For each source point, compute the weighted average of its destinations
    # Normalize plan rows to get conditional transport
    plan_normalized = plan / (plan.sum(axis=1, keepdims=True) + 1e-30)

    # Weighted destination for each source point
    destinations = plan_normalized @ target

    # Interpolate
    return (1 - t) * source + t * destinations

def linear_interpolation_samples(source, target, t):
    """Simple linear mixing (cross-fade approximation)."""
    n = min(len(source), len(target))
    return (1 - t) * source[:n] + t * target[:n]

# Create distributions
source, target = create_distributions()
n = len(source)
m = len(target)

# Uniform weights
a = np.ones(n) / n
b = np.ones(m) / m

# Cost matrix (squared Euclidean)
C = np.sum((source[:, None, :] - target[None, :, :]) ** 2, axis=2)

# Solve OT
plan = sinkhorn(a, b, C, epsilon=0.5, n_iter=200)

# Visualize
fig, axes = plt.subplots(2, 5, figsize=(18, 7))

times = [0.0, 0.25, 0.5, 0.75, 1.0]

# Top row: Displacement interpolation
for i, t in enumerate(times):
    ax = axes[0, i]
    if t == 0:
        pts = source
    elif t == 1:
        pts = target
    else:
        pts = displacement_interpolation(source, target, plan, t)

    ax.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.5, color='#4fc3f7')
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.set_title(f't = {t}', fontsize=11)
    ax.grid(True, alpha=0.2)
    if i == 0:
        ax.set_ylabel('Displacement\n(OT geodesic)', fontsize=10, color='#4fc3f7')

# Bottom row: Linear interpolation (cross-fade)
for i, t in enumerate(times):
    ax = axes[1, i]
    if t == 0:
        pts = source
    elif t == 1:
        pts = target
    else:
        pts = linear_interpolation_samples(source, target, t)

    ax.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.5, color='#E53935')
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.set_title(f't = {t}', fontsize=11)
    ax.grid(True, alpha=0.2)
    if i == 0:
        ax.set_ylabel('Linear\n(cross-fade)', fontsize=10, color='#E53935')

plt.suptitle('Displacement Interpolation (OT) vs Linear Interpolation (Cross-Fade)', fontsize=14)
plt.tight_layout()
plt.savefig('ot_displacement_interpolation.png', dpi=150, bbox_inches='tight')
plt.show()
```

Optimal transport provides the mathematical foundation for understanding motion in video as the transport of visual content through space and time. The displacement interpolation is the natural way to move between two distributions --- and by extension, between two video frames --- without ghosting or cross-fading. The Benamou-Brenier formula connects this to the velocity fields learned by flow matching models. And the Wasserstein distance provides a geometrically meaningful metric for comparing distributions of generated and real video. The deepest insight: flow matching IS optimal transport, applied one sample at a time.
