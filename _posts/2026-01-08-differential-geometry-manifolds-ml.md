---
layout: post
title: "Differential Geometry: Manifolds, Tangent Spaces, and the Curved Geometry of Data"
date: 2026-01-08
category: math
---

Most of machine learning assumes the world is flat. Data points live in $\mathbb{R}^n$, distances are Euclidean, and optimization happens by moving along straight lines (gradient descent). This works surprisingly well for many problems. But it is fundamentally wrong for many others.

The Earth is not flat, and neither is the space of probability distributions, the space of positive-definite matrices, the latent space of a generative model, or the parameter space of a neural network. These are all **curved spaces** --- manifolds --- and using flat-space intuition leads to real problems: geodesic distances differ dramatically from Euclidean distances, straight-line interpolation leaves the space entirely, and the "steepest descent" direction depends on the geometry.

This post builds the mathematical framework of differential geometry from scratch: manifolds, tangent spaces, Riemannian metrics, geodesics, curvature. At each stage, we connect the abstractions to concrete problems in machine learning.

---

## Table of Contents

1. [What Is a Manifold?](#what-is-a-manifold)
2. [Charts, Atlases, and Coordinate Systems](#charts-atlases-and-coordinate-systems)
3. [Tangent Spaces: Velocity Vectors on Curved Surfaces](#tangent-spaces-velocity-vectors-on-curved-surfaces)
4. [The Tangent Bundle](#the-tangent-bundle)
5. [Riemannian Metrics: Measuring on Curved Spaces](#riemannian-metrics-measuring-on-curved-spaces)
6. [Geodesics: Shortest Paths on Curved Surfaces](#geodesics-shortest-paths-on-curved-surfaces)
7. [Curvature: How Space Bends](#curvature-how-space-bends)
8. [Parallel Transport](#parallel-transport)
9. [The Exponential Map](#the-exponential-map)
10. [Connections to Machine Learning](#connections-to-machine-learning)
11. [Conclusion](#conclusion)

---

## What Is a Manifold?

A **manifold** is a space that is curved globally but looks flat locally. The canonical example is the surface of the Earth. If you stand in a field, the ground looks flat --- locally, it is well-approximated by a flat plane $\mathbb{R}^2$. But globally, it is a sphere $S^2$, and flat-Earth geometry gives wrong answers at large scales. A straight line on a flat map is not the shortest path between two cities on a sphere.

Formally, a **topological manifold** of dimension $n$ is a topological space $M$ such that:

1. **Locally Euclidean:** Every point $p \in M$ has a neighborhood that is homeomorphic (continuously deformable, with a continuous inverse) to an open subset of $\mathbb{R}^n$.
2. **Hausdorff:** Any two distinct points have disjoint neighborhoods (no pathological point-identification).
3. **Second countable:** The topology has a countable basis (a technical condition that prevents the manifold from being "too large").

The key word is **locally Euclidean**. A manifold is a space where, if you zoom in enough, it looks like ordinary flat $\mathbb{R}^n$. The dimension $n$ is the number of coordinates you need locally.

### Examples

**The circle $S^1$:** A 1-dimensional manifold. Locally, each small arc looks like a line segment (a piece of $\mathbb{R}^1$). Globally, it is closed and curved --- you can walk forever without hitting a boundary.

**The sphere $S^2$:** A 2-dimensional manifold embedded in $\mathbb{R}^3$. Each small patch looks like a piece of the plane $\mathbb{R}^2$. This is what maps exploit: a small-enough map of your city is approximately correct.

**The space of unit vectors in $\mathbb{R}^n$:** This is the sphere $S^{n-1}$, an $(n-1)$-dimensional manifold. Normalized word embeddings live here.

**The space of $n \times n$ positive definite matrices:** This is a manifold of dimension $n(n+1)/2$ (the number of unique entries in a symmetric matrix). Covariance matrices live here, and this space has rich non-Euclidean geometry.

**What is NOT a manifold:** A figure-eight is not a manifold because at the crossing point, no neighborhood looks like a line. A cone tip is not a manifold because the tip has no flat neighborhood.

---

## Charts, Atlases, and Coordinate Systems

The "locally Euclidean" condition means we can put coordinates on small patches of a manifold. But different patches need different coordinate systems, and we need them to be compatible where they overlap.

A **chart** on a manifold $M$ is a pair $(U, \varphi)$ where:
- $U \subseteq M$ is an open set (a patch of the manifold).
- $\varphi: U \to \varphi(U) \subseteq \mathbb{R}^n$ is a homeomorphism (a continuous bijection with continuous inverse).

The function $\varphi$ assigns coordinates to each point in the patch $U$. If $p \in U$, then $\varphi(p) = (x^1, x^2, \ldots, x^n) \in \mathbb{R}^n$ are the **local coordinates** of $p$ in this chart.

An **atlas** is a collection of charts $\{(U_\alpha, \varphi_\alpha)\}$ that cover the entire manifold: $\bigcup_\alpha U_\alpha = M$.

When two charts $(U_\alpha, \varphi_\alpha)$ and $(U_\beta, \varphi_\beta)$ overlap (i.e., $U_\alpha \cap U_\beta \neq \emptyset$), we get a **transition map**:

$$\varphi_\beta \circ \varphi_\alpha^{-1}: \varphi_\alpha(U_\alpha \cap U_\beta) \to \varphi_\beta(U_\alpha \cap U_\beta)$$

This is a map from $\mathbb{R}^n$ to $\mathbb{R}^n$ --- it tells you how to translate coordinates from one chart to another. A **smooth manifold** requires all transition maps to be smooth ($C^\infty$).

### Example: The Sphere Needs Multiple Charts

Consider $S^2$ (the 2-sphere). Can we cover it with a single chart? No. A single chart would be a homeomorphism from $S^2$ to an open subset of $\mathbb{R}^2$, and $S^2$ is compact while open subsets of $\mathbb{R}^2$ are not. More intuitively: you cannot make a perfectly flat map of the entire Earth without tearing it.

The standard atlas for $S^2$ uses **stereographic projection** from two charts:
- Project from the North Pole onto the plane tangent to the South Pole. This covers everything except the North Pole itself.
- Project from the South Pole onto the plane tangent to the North Pole. This covers everything except the South Pole.

Together, they cover all of $S^2$.

<svg viewBox="0 0 700 380" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-size="15" font-weight="bold" fill="#d4d4d4">Stereographic Projection: A Chart on S²</text>

  <!-- Sphere -->
  <ellipse cx="250" cy="200" rx="120" ry="120" fill="none" stroke="#d4d4d4" stroke-width="2"/>
  <ellipse cx="250" cy="200" rx="120" ry="30" fill="none" stroke="#d4d4d4" stroke-width="1" stroke-dasharray="5,5"/>

  <!-- North pole -->
  <circle cx="250" cy="80" r="5" fill="#cc3333"/>
  <text x="268" y="78" font-size="12" fill="#cc3333" font-weight="bold">N</text>

  <!-- South pole -->
  <circle cx="250" cy="320" r="5" fill="#336699"/>
  <text x="268" y="325" font-size="12" fill="#336699" font-weight="bold">S</text>

  <!-- Point on sphere -->
  <circle cx="325" cy="145" r="5" fill="#339933"/>
  <text x="335" y="142" font-size="12" fill="#339933" font-weight="bold">p</text>

  <!-- Projection line from N through p -->
  <line x1="250" y1="80" x2="500" y2="360" stroke="#cc3333" stroke-width="1.5" stroke-dasharray="6,3"/>

  <!-- Projection plane -->
  <line x1="100" y1="360" x2="560" y2="360" stroke="#d4d4d4" stroke-width="2"/>
  <text x="570" y="365" font-size="12" fill="#d4d4d4">ℝ²</text>

  <!-- Projected point -->
  <circle cx="500" cy="360" r="5" fill="#339933"/>
  <text x="505" y="355" font-size="12" fill="#339933" font-weight="bold">φ(p)</text>

  <!-- Labels -->
  <text x="420" y="210" font-size="11" fill="#cc3333">projection line</text>
  <text x="420" y="226" font-size="11" fill="#cc3333">from N through p</text>

  <!-- Right side: coordinate explanation -->
  <rect x="560" y="60" width="130" height="120" rx="8" fill="#1e1e1e" stroke="#444" stroke-width="1"/>
  <text x="625" y="82" text-anchor="middle" font-size="12" font-weight="bold" fill="#d4d4d4">Chart (U, φ)</text>
  <text x="625" y="102" text-anchor="middle" font-size="11" fill="#d4d4d4">U = S² \ {N}</text>
  <text x="625" y="122" text-anchor="middle" font-size="11" fill="#d4d4d4">φ: U → ℝ²</text>
  <text x="625" y="142" text-anchor="middle" font-size="11" fill="#d4d4d4">φ(p) = (x, y)</text>
  <text x="625" y="162" text-anchor="middle" font-size="11" fill="#999">Covers all of</text>
  <text x="625" y="177" text-anchor="middle" font-size="11" fill="#999">S² except N</text>

  <!-- Second chart note -->
  <rect x="560" y="200" width="130" height="80" rx="8" fill="#1e1e1e" stroke="#444" stroke-width="1"/>
  <text x="625" y="222" text-anchor="middle" font-size="11" fill="#336699">Second chart:</text>
  <text x="625" y="242" text-anchor="middle" font-size="11" fill="#336699">project from S</text>
  <text x="625" y="262" text-anchor="middle" font-size="11" fill="#336699">covers S² \ {S}</text>

  <text x="625" y="310" text-anchor="middle" font-size="11" fill="#d4d4d4">Together: full</text>
  <text x="625" y="326" text-anchor="middle" font-size="11" fill="#d4d4d4">atlas for S²</text>
</svg>

---

## Tangent Spaces: Velocity Vectors on Curved Surfaces

On flat $\mathbb{R}^n$, vectors are simple: they are arrows pointing from one point to another. On a curved manifold, this does not work. There is no global notion of "direction" because the space is curved --- a vector at one point cannot be directly compared to a vector at another point without additional structure.

The **tangent space** $T_pM$ at a point $p \in M$ is the set of all "velocity vectors" of curves passing through $p$. It is a vector space of the same dimension as $M$, and it represents all the directions you can move from $p$.

### Rigorous Definition

There are three equivalent ways to define tangent vectors. Here is the most intuitive one.

**Definition via curves:** Let $\gamma: (-\epsilon, \epsilon) \to M$ be a smooth curve with $\gamma(0) = p$. The **tangent vector** to $\gamma$ at $p$ is the "velocity" $\gamma'(0)$.

But what does $\gamma'(0)$ mean on a manifold? We need coordinates. Choose a chart $(U, \varphi)$ around $p$. In local coordinates, $\gamma$ becomes $\varphi \circ \gamma: (-\epsilon, \epsilon) \to \mathbb{R}^n$, which is an ordinary curve in $\mathbb{R}^n$. Its derivative at 0 is an ordinary vector in $\mathbb{R}^n$:

$$\left.\frac{d}{dt}\right|_{t=0} (\varphi \circ \gamma)(t) = \left(\dot{\gamma}^1(0), \dot{\gamma}^2(0), \ldots, \dot{\gamma}^n(0)\right)$$

Two curves $\gamma_1$ and $\gamma_2$ with $\gamma_1(0) = \gamma_2(0) = p$ define the **same** tangent vector if and only if they have the same derivative in local coordinates:

$$\left.\frac{d}{dt}\right|_{t=0} (\varphi \circ \gamma_1)(t) = \left.\frac{d}{dt}\right|_{t=0} (\varphi \circ \gamma_2)(t)$$

This is chart-independent (thanks to the chain rule and smooth transition maps).

The tangent space $T_pM$ is the set of all equivalence classes of curves through $p$, where two curves are equivalent if they have the same velocity. It is a vector space of dimension $n$ (the same as the manifold), and in local coordinates $(x^1, \ldots, x^n)$, a basis is:

$$\left\{\frac{\partial}{\partial x^1}\bigg|_p, \frac{\partial}{\partial x^2}\bigg|_p, \ldots, \frac{\partial}{\partial x^n}\bigg|_p\right\}$$

Any tangent vector $v \in T_pM$ can be written as $v = v^i \frac{\partial}{\partial x^i}\bigg|_p$ (using Einstein summation convention: repeated indices are summed).

<svg viewBox="0 0 700 380" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-size="15" font-weight="bold" fill="#d4d4d4">Tangent Plane to S² at Point p</text>

  <defs>
    <marker id="arrowTangent" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#cc3333"/>
    </marker>
    <marker id="arrowBlue" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#336699"/>
    </marker>
  </defs>

  <!-- Sphere -->
  <ellipse cx="280" cy="220" rx="150" ry="150" fill="#1a2a40" fill-opacity="0.3" stroke="#d4d4d4" stroke-width="2"/>
  <ellipse cx="280" cy="220" rx="150" ry="40" fill="none" stroke="#d4d4d4" stroke-width="1" stroke-dasharray="5,5"/>

  <!-- Point p (on upper part of sphere) -->
  <circle cx="340" cy="120" r="6" fill="#339933"/>
  <text x="352" y="115" font-size="13" fill="#339933" font-weight="bold">p</text>

  <!-- Tangent plane (parallelogram) -->
  <polygon points="240,60 420,80 460,180 280,160" fill="#cc7733" fill-opacity="0.15" stroke="#cc7733" stroke-width="1.5"/>
  <text x="450" y="75" font-size="12" fill="#cc7733" font-weight="bold">TₚM</text>

  <!-- Tangent vectors -->
  <line x1="340" y1="120" x2="410" y2="110" stroke="#cc3333" stroke-width="2.5" marker-end="url(#arrowTangent)"/>
  <text x="418" y="108" font-size="11" fill="#cc3333">∂/∂x¹</text>

  <line x1="340" y1="120" x2="365" y2="165" stroke="#336699" stroke-width="2.5" marker-end="url(#arrowBlue)"/>
  <text x="372" y="172" font-size="11" fill="#336699">∂/∂x²</text>

  <!-- Curve on sphere through p -->
  <path d="M220,170 Q280,100 340,120 Q400,140 410,200" fill="none" stroke="#339933" stroke-width="2" stroke-dasharray="6,3"/>
  <text x="205" y="168" font-size="11" fill="#339933">γ(t)</text>

  <!-- Annotation -->
  <rect x="490" y="120" width="190" height="100" rx="8" fill="#1e1e1e" stroke="#444" stroke-width="1"/>
  <text x="585" y="142" text-anchor="middle" font-size="11" fill="#d4d4d4">TₚM is the plane</text>
  <text x="585" y="158" text-anchor="middle" font-size="11" fill="#d4d4d4">tangent to M at p.</text>
  <text x="585" y="180" text-anchor="middle" font-size="11" fill="#d4d4d4">dim(TₚM) = dim(M)</text>
  <text x="585" y="202" text-anchor="middle" font-size="11" fill="#d4d4d4">Vectors live IN TₚM,</text>
  <text x="585" y="218" text-anchor="middle" font-size="11" fill="#d4d4d4">not in M itself.</text>
</svg>

---

## The Tangent Bundle

The **tangent bundle** $TM$ of a manifold $M$ is the disjoint union of all tangent spaces:

$$TM = \bigsqcup_{p \in M} T_pM = \{(p, v) : p \in M, \, v \in T_pM\}$$

This is itself a manifold of dimension $2n$ (if $M$ has dimension $n$): you need $n$ coordinates to specify the point $p$ and $n$ more to specify the tangent vector $v$ at $p$.

The tangent bundle is the natural setting for dynamics. A point in $TM$ specifies both a **position** and a **velocity** --- exactly the information you need to describe the state of a physical system (or the state of an optimization algorithm on a manifold).

A **vector field** is a smooth assignment of a tangent vector to each point: a smooth map $X: M \to TM$ with $X(p) \in T_pM$ for each $p$. Gradient fields, velocity fields, and the "directions of steepest descent" in optimization are all vector fields.

---

## Riemannian Metrics: Measuring on Curved Spaces

A bare manifold has topology (which points are "near" which) but no notion of distance, angle, or length. To measure things, we need additional structure: a **Riemannian metric**.

A **Riemannian metric** $g$ on a smooth manifold $M$ is a smoothly varying inner product on each tangent space. That is, for each point $p \in M$, $g_p: T_pM \times T_pM \to \mathbb{R}$ is:

1. **Bilinear:** $g_p(\alpha u + \beta v, w) = \alpha g_p(u, w) + \beta g_p(v, w)$.
2. **Symmetric:** $g_p(u, v) = g_p(v, u)$.
3. **Positive definite:** $g_p(v, v) > 0$ for $v \neq 0$.

In local coordinates $(x^1, \ldots, x^n)$, the metric is specified by the **metric tensor** $g_{ij}(p)$:

$$g_p\left(\frac{\partial}{\partial x^i}, \frac{\partial}{\partial x^j}\right) = g_{ij}(p)$$

The metric tensor is a symmetric positive definite $n \times n$ matrix at each point. For two tangent vectors $u = u^i \frac{\partial}{\partial x^i}$ and $v = v^j \frac{\partial}{\partial x^j}$:

$$g_p(u, v) = g_{ij}(p) \, u^i v^j$$

### What the Metric Gives You

**Length of a curve:** If $\gamma: [a, b] \to M$ is a smooth curve, its length is:

$$L(\gamma) = \int_a^b \sqrt{g_{\gamma(t)}(\dot{\gamma}(t), \dot{\gamma}(t))} \, dt = \int_a^b \sqrt{g_{ij} \dot{\gamma}^i \dot{\gamma}^j} \, dt$$

**Distance between points:** $d(p, q) = \inf_\gamma L(\gamma)$, where the infimum is over all smooth curves from $p$ to $q$.

**Angle between vectors:** $\cos \theta = \frac{g_p(u, v)}{\sqrt{g_p(u, u)} \sqrt{g_p(v, v)}}$.

**Volume:** $dV = \sqrt{\det g} \, dx^1 \wedge \cdots \wedge dx^n$.

### Example: The Sphere

For the unit sphere $S^2$ with spherical coordinates $(\theta, \phi)$ (where $\theta \in [0, \pi]$ is the polar angle from the North Pole and $\phi \in [0, 2\pi)$ is the azimuthal angle):

$$ds^2 = d\theta^2 + \sin^2\theta \, d\phi^2$$

The metric tensor is:

$$g = \begin{pmatrix} 1 & 0 \\ 0 & \sin^2\theta \end{pmatrix}$$

Notice: the metric depends on position. Near the poles ($\theta \approx 0$ or $\pi$), the $\phi$ direction is "squished" ($\sin^2\theta \approx 0$), reflecting the fact that lines of longitude converge at the poles. This is the curvature of the sphere encoded in the metric.

### Example: Flat Space

In $\mathbb{R}^n$ with standard Cartesian coordinates, $g_{ij} = \delta_{ij}$ (the identity matrix). The metric is the same everywhere, and everything reduces to the familiar Euclidean geometry: $ds^2 = dx_1^2 + dx_2^2 + \cdots + dx_n^2$.

---

## Geodesics: Shortest Paths on Curved Surfaces

A **geodesic** is a curve that locally minimizes length. It is the generalization of a "straight line" to curved spaces. On the sphere, geodesics are great circles (the equator, lines of longitude, and any other circle whose center is the center of the sphere).

### Deriving the Geodesic Equation

We find geodesics by **variational calculus** --- minimizing the length functional. It is technically easier to minimize the **energy functional** (which gives the same paths when parameterized by arc length):

$$E(\gamma) = \frac{1}{2} \int_a^b g_{ij}(\gamma(t)) \dot{\gamma}^i(t) \dot{\gamma}^j(t) \, dt$$

We want to find the curve $\gamma$ that minimizes $E(\gamma)$ with fixed endpoints $\gamma(a) = p$, $\gamma(b) = q$.

Applying the Euler-Lagrange equations (which we will derive in full in the next post), we get the **geodesic equation**:

$$\ddot{\gamma}^k + \Gamma^k_{ij} \dot{\gamma}^i \dot{\gamma}^j = 0$$

where $\Gamma^k_{ij}$ are the **Christoffel symbols**:

$$\Gamma^k_{ij} = \frac{1}{2} g^{kl}\left(\frac{\partial g_{il}}{\partial x^j} + \frac{\partial g_{jl}}{\partial x^i} - \frac{\partial g_{ij}}{\partial x^l}\right)$$

The Christoffel symbols encode how the coordinate basis vectors change from point to point. In flat space with Cartesian coordinates, all $\Gamma^k_{ij} = 0$, and the geodesic equation reduces to $\ddot{\gamma}^k = 0$ --- straight lines.

On curved spaces, the $\Gamma^k_{ij} \dot{\gamma}^i \dot{\gamma}^j$ term acts like a "fictitious force" that curves the path. This is exactly analogous to how objects in a rotating reference frame experience Coriolis and centrifugal forces, which are really just Christoffel symbols in disguise.

### Python: Geodesics on the Sphere

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

def geodesic_sphere(t, y):
    """Geodesic equations on S^2 in spherical coordinates (theta, phi)."""
    theta, phi, dtheta, dphi = y
    # Christoffel symbols for S^2:
    # Gamma^theta_{phi,phi} = -sin(theta)cos(theta)
    # Gamma^phi_{theta,phi} = Gamma^phi_{phi,theta} = cos(theta)/sin(theta)
    ddtheta = np.sin(theta) * np.cos(theta) * dphi**2
    ddphi = -2 * np.cos(theta) / (np.sin(theta) + 1e-15) * dtheta * dphi
    return [dtheta, dphi, ddtheta, ddphi]

def spherical_to_cartesian(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

fig = plt.figure(figsize=(14, 6))

# --- Plot 1: Multiple geodesics from a point ---
ax1 = fig.add_subplot(121, projection='3d')

# Draw sphere wireframe
u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi, 20)
xs = np.outer(np.sin(v), np.cos(u))
ys = np.outer(np.sin(v), np.sin(u))
zs = np.outer(np.cos(v), np.ones_like(u))
ax1.plot_surface(xs, ys, zs, alpha=0.1, color='lightblue')
ax1.plot_wireframe(xs, ys, zs, alpha=0.15, color='gray', linewidth=0.5)

# Start point: (theta0, phi0)
theta0, phi0 = np.pi/4, 0.0
p0 = spherical_to_cartesian(theta0, phi0)
ax1.scatter(*p0, color='red', s=80, zorder=5, label=r'Start point $p$')

# Shoot geodesics in different directions
colors = plt.cm.viridis(np.linspace(0.1, 0.9, 8))
for i, angle in enumerate(np.linspace(0, 2*np.pi, 8, endpoint=False)):
    dtheta0 = 0.8 * np.cos(angle)
    dphi0 = 0.8 * np.sin(angle) / np.sin(theta0)

    sol = solve_ivp(geodesic_sphere, [0, 4], [theta0, phi0, dtheta0, dphi0],
                    t_eval=np.linspace(0, 4, 200), max_step=0.01)

    gx, gy, gz = spherical_to_cartesian(sol.y[0], sol.y[1])
    ax1.plot(gx, gy, gz, color=colors[i], linewidth=2)

ax1.set_title(r'Geodesics on $S^2$ from a Point', fontsize=13)
ax1.set_xlim([-1.2, 1.2]); ax1.set_ylim([-1.2, 1.2]); ax1.set_zlim([-1.2, 1.2])
ax1.legend(fontsize=9)

# --- Plot 2: Geodesic vs Euclidean distance ---
ax2 = fig.add_subplot(122)

# Compare geodesic and Euclidean distances between two points on S^2
# Fix point A, move point B along the sphere
theta_A, phi_A = np.pi/2, 0.0
angular_seps = np.linspace(0, np.pi, 200)

geodesic_dists = angular_seps  # On unit sphere, geodesic distance = angle
euclidean_dists = 2 * np.sin(angular_seps / 2)  # Chord length

ax2.plot(np.degrees(angular_seps), geodesic_dists, '-', color='#cc3333',
         linewidth=2.5, label=r'Geodesic distance (arc)')
ax2.plot(np.degrees(angular_seps), euclidean_dists, '--', color='#4488cc',
         linewidth=2.5, label=r'Euclidean distance (chord)')
ax2.fill_between(np.degrees(angular_seps), euclidean_dists, geodesic_dists,
                 alpha=0.15, color='orange', label=r'Discrepancy')
ax2.set_xlabel(r'Angular separation (degrees)', fontsize=12)
ax2.set_ylabel(r'Distance', fontsize=12)
ax2.set_title(r'Geodesic vs Euclidean Distance on $S^2$', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Annotate the maximum discrepancy
ax2.annotate(r'At antipodes:' + '\n' + r'geodesic $= \pi \approx 3.14$' + '\n' + r'Euclidean $= 2$',
            xy=(180, 2), xytext=(120, 2.6),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig("geodesics_sphere.png", dpi=150, bbox_inches='tight')
plt.show()
```

---

## Curvature: How Space Bends

Curvature quantifies how a space deviates from being flat. There are several notions of curvature, each capturing different aspects of the geometry.

### Gaussian Curvature

For a 2-dimensional surface embedded in $\mathbb{R}^3$, the **Gaussian curvature** $K$ at a point $p$ is the product of the two **principal curvatures** $\kappa_1$ and $\kappa_2$:

$$K = \kappa_1 \cdot \kappa_2$$

The principal curvatures are the maximum and minimum curvatures of curves on the surface passing through $p$. Intuitively:

- $K > 0$: The surface curves the same way in all directions (like a sphere). A small circle on the surface has circumference **less** than $2\pi r$.
- $K = 0$: The surface is flat (locally isometric to $\mathbb{R}^2$). A cylinder has $K = 0$ everywhere because you can unroll it flat.
- $K < 0$: The surface curves in opposite directions (like a saddle or a Pringle chip). A small circle has circumference **greater** than $2\pi r$.

Gauss's Theorema Egregium ("remarkable theorem") states that Gaussian curvature depends only on the metric tensor $g_{ij}$ --- it is an **intrinsic** property of the surface. A being living on the surface could measure $K$ without knowing about the ambient 3D space. This is why it matters: curvature is not about how the manifold sits inside a bigger space, but about the geometry of the manifold itself.

### The Riemann Curvature Tensor

For manifolds of dimension $n > 2$, the full curvature is captured by the **Riemann curvature tensor** $R^l_{\ ijk}$, a rank-4 tensor defined by:

$$R^l_{\ ijk} = \frac{\partial \Gamma^l_{jk}}{\partial x^i} - \frac{\partial \Gamma^l_{ik}}{\partial x^j} + \Gamma^l_{im}\Gamma^m_{jk} - \Gamma^l_{jm}\Gamma^m_{ik}$$

This tensor measures how much a vector changes when you **parallel transport** it around a small loop --- a concept we will define shortly.

### Ricci Curvature

The **Ricci curvature tensor** is a contraction (trace) of the Riemann tensor:

$$R_{ij} = R^k_{\ ikj}$$

It measures how the volume of a small ball around a point deviates from the Euclidean volume:

$$\frac{\text{Vol}(B_r(p))}{V_{\text{Eucl}}(r)} \approx 1 - \frac{R_{ij} v^i v^j}{6(n+2)} r^2 + \cdots$$

Positive Ricci curvature means volumes are **smaller** than Euclidean (the space is "pinched"), negative means **larger** (the space is "spread out"). This matters for ML because it controls the concentration of measure on the manifold --- how probability mass distributes.

---

## Parallel Transport

On flat $\mathbb{R}^n$, you can compare vectors at different points directly: just slide one vector to the other point without changing it. On a curved manifold, there is no canonical way to do this. You must **transport** vectors along a specific path, keeping them "as constant as possible" --- this is **parallel transport**.

A vector field $V(t)$ along a curve $\gamma(t)$ is **parallel** if:

$$\frac{DV^k}{dt} = \frac{dV^k}{dt} + \Gamma^k_{ij} \dot{\gamma}^i V^j = 0$$

The operator $D/dt$ is the **covariant derivative** along the curve. It adjusts the ordinary derivative by the Christoffel terms to account for the curvature of the space.

The key property of parallel transport on a curved manifold: **the result depends on the path.** If you parallel transport a vector from $A$ to $B$ along two different paths, you may get different results. The difference is exactly measured by the Riemann curvature tensor.

<svg viewBox="0 0 700 350" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-size="15" font-weight="bold" fill="#d4d4d4">Parallel Transport on S²: Path Dependence</text>

  <defs>
    <marker id="arrowRed" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#cc3333"/>
    </marker>
    <marker id="arrowGreen" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#339933"/>
    </marker>
  </defs>

  <!-- Sphere outline -->
  <ellipse cx="280" cy="200" rx="160" ry="160" fill="#1a2a40" fill-opacity="0.2" stroke="#d4d4d4" stroke-width="2"/>
  <ellipse cx="280" cy="200" rx="160" ry="45" fill="none" stroke="#d4d4d4" stroke-width="1" stroke-dasharray="4,4"/>

  <!-- Triangle path on sphere: North Pole -> Equator(0) -> Equator(90) -> North Pole -->
  <!-- Path A -> B (along meridian) -->
  <path d="M280,45 Q310,120 375,195" fill="none" stroke="#4488cc" stroke-width="2.5"/>
  <text x="345" y="110" font-size="11" fill="#4488cc">meridian φ=0</text>

  <!-- Path B -> C (along equator) -->
  <path d="M375,195 Q340,210 280,235 Q220,210 185,195" fill="none" stroke="#4488cc" stroke-width="2.5"/>
  <text x="275" y="260" font-size="11" fill="#4488cc">equator (90°)</text>

  <!-- Path C -> A (along meridian) -->
  <path d="M185,195 Q250,120 280,45" fill="none" stroke="#4488cc" stroke-width="2.5"/>
  <text x="195" y="110" font-size="11" fill="#4488cc">meridian φ=90°</text>

  <!-- Points -->
  <circle cx="280" cy="45" r="6" fill="#d4d4d4"/>
  <text x="290" y="38" font-size="12" fill="#d4d4d4" font-weight="bold">N (start)</text>
  <circle cx="375" cy="195" r="5" fill="#d4d4d4"/>
  <text x="385" y="193" font-size="11" fill="#d4d4d4">B</text>
  <circle cx="185" cy="195" r="5" fill="#d4d4d4"/>
  <text x="160" y="193" font-size="11" fill="#d4d4d4">C</text>

  <!-- Initial vector at N pointing "east" -->
  <line x1="280" y1="45" x2="320" y2="45" stroke="#cc3333" stroke-width="3" marker-end="url(#arrowRed)"/>
  <text x="325" y="42" font-size="11" fill="#cc3333" font-weight="bold">v (start)</text>

  <!-- After transport around loop, vector rotated 90 degrees -->
  <line x1="280" y1="45" x2="280" y2="85" stroke="#339933" stroke-width="3" marker-end="url(#arrowGreen)"/>
  <text x="287" y="92" font-size="11" fill="#339933" font-weight="bold">v (after loop)</text>

  <!-- Angle annotation -->
  <path d="M300,45 A20,20 0 0,1 280,65" fill="none" stroke="#cc7733" stroke-width="1.5"/>
  <text x="305" y="62" font-size="11" fill="#cc7733">90°</text>

  <!-- Explanation -->
  <rect x="460" y="60" width="220" height="140" rx="8" fill="#1e1e1e" stroke="#444" stroke-width="1"/>
  <text x="570" y="82" text-anchor="middle" font-size="12" font-weight="bold" fill="#d4d4d4">Path dependence</text>
  <text x="570" y="102" text-anchor="middle" font-size="11" fill="#d4d4d4">Transport v around a</text>
  <text x="570" y="118" text-anchor="middle" font-size="11" fill="#d4d4d4">triangle with 3 right angles.</text>
  <text x="570" y="142" text-anchor="middle" font-size="11" fill="#d4d4d4">After returning to start,</text>
  <text x="570" y="158" text-anchor="middle" font-size="11" fill="#d4d4d4">v has rotated by 90°!</text>
  <text x="570" y="182" text-anchor="middle" font-size="11" fill="#cc3333">Rotation angle = area/R²</text>
  <text x="570" y="198" text-anchor="middle" font-size="11" fill="#cc3333">= enclosed solid angle</text>
</svg>

This is perhaps the most striking consequence of curvature. On a flat surface, parallel transporting a vector around any closed loop returns it unchanged. On a curved surface, the vector rotates. The amount of rotation is proportional to the curvature enclosed by the loop. This is the geometric meaning of the Riemann curvature tensor.

---

## The Exponential Map

The **exponential map** $\exp_p: T_pM \to M$ "shoots" a geodesic from $p$ in the direction $v \in T_pM$ for unit time:

$$\exp_p(v) = \gamma(1)$$

where $\gamma$ is the unique geodesic with $\gamma(0) = p$ and $\dot{\gamma}(0) = v$.

More generally, $\exp_p(tv)$ walks along the geodesic in direction $v$ for time $t$. The length of the geodesic from $p$ to $\exp_p(v)$ is $\|v\|_g = \sqrt{g_p(v, v)}$.

The exponential map has a crucial property: it is a **local diffeomorphism** near the origin. For small enough $v$, $\exp_p$ is a smooth bijection from a neighborhood of $0 \in T_pM$ to a neighborhood of $p \in M$. This gives us **normal coordinates** (also called geodesic coordinates or Riemann normal coordinates) centered at $p$.

In normal coordinates, the metric is Euclidean to first order:

$$g_{ij}(x) = \delta_{ij} - \frac{1}{3} R_{ikjl} x^k x^l + O(|x|^3)$$

This is the precise mathematical statement that a curved manifold looks flat locally: the curvature only appears at second order.

### The Logarithmic Map

The inverse of the exponential map (where it exists) is the **logarithmic map** $\log_p: M \to T_pM$:

$$\log_p(q) = v \quad \text{where} \quad \exp_p(v) = q$$

The logarithmic map finds the initial velocity of the geodesic from $p$ to $q$. This is the key operation for doing "arithmetic" on manifolds: to compute $q - p$ on a manifold, you compute $\log_p(q)$, which gives you a vector in $T_pM$.

### Python: Exponential Map Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left: Exp map on S^2 (Mollweide-like visualization) ---
ax1 = axes[0]

# Base point: (theta=pi/3, phi=pi/4) on S^2
theta0, phi0 = np.pi/3, np.pi/4

# Grid of tangent vectors
n_rings = 6
n_dirs = 24
colors_ring = plt.cm.plasma(np.linspace(0.1, 0.9, n_rings))

for i, r in enumerate(np.linspace(0.3, 2.5, n_rings)):
    thetas_exp = []
    phis_exp = []
    for angle in np.linspace(0, 2*np.pi, n_dirs, endpoint=False):
        # Tangent vector components
        dtheta = r * np.cos(angle)
        dphi = r * np.sin(angle) / np.sin(theta0)

        # Geodesic on sphere: use rotation approach
        # For visualization, numerically integrate
        from scipy.integrate import solve_ivp

        def geo(t, y):
            th, ph, dth, dph = y
            sin_th = np.sin(th) + 1e-15
            ddth = np.sin(th) * np.cos(th) * dph**2
            ddph = -2 * np.cos(th) / sin_th * dth * dph
            return [dth, dph, ddth, ddph]

        sol = solve_ivp(geo, [0, 1], [theta0, phi0, dtheta, dphi],
                       max_step=0.01)
        thetas_exp.append(sol.y[0][-1])
        phis_exp.append(sol.y[1][-1])

    thetas_exp = np.array(thetas_exp)
    phis_exp = np.array(phis_exp)

    # Convert to Cartesian for plotting
    x = np.sin(thetas_exp) * np.cos(phis_exp)
    y_coord = np.sin(thetas_exp) * np.sin(phis_exp)

    ax1.plot(np.append(phis_exp, phis_exp[0]),
             np.append(thetas_exp, thetas_exp[0]),
             '-', color=colors_ring[i], linewidth=1.5,
             label=r'$r=$' + f'{r:.1f}' if i % 2 == 0 else None)

# Base point
ax1.plot(phi0, theta0, 'ro', markersize=10, zorder=5, label=r'Base point $p$')
ax1.set_xlabel(r'$\varphi$ (azimuthal)', fontsize=12)
ax1.set_ylabel(r'$\theta$ (polar)', fontsize=12)
ax1.set_title(r'Exponential Map on $S^2$' + '\n' + r'(concentric circles in $T_pM$ mapped to $S^2$)', fontsize=13)
ax1.legend(fontsize=9)
ax1.invert_yaxis()
ax1.set_xlim(-1, 2*np.pi+1)
ax1.set_ylim(np.pi+0.3, -0.3)
ax1.grid(True, alpha=0.3)

# --- Right: Distortion comparison ---
ax2 = axes[1]

# For various radii, compare tangent space distance vs geodesic distance
radii = np.linspace(0.01, np.pi, 200)

# Geodesic distance on unit sphere = angle between points
# If we shoot a geodesic with speed r for time 1, geodesic distance = r
# Euclidean distance (chord) between start and endpoint
euclidean_dists = []
for r in radii:
    # Endpoint of geodesic: theta changes by r (for simplicity, along meridian)
    theta_end = theta0 + r
    if theta_end > np.pi:
        theta_end = 2*np.pi - theta_end  # Wrap
    # Both points on same meridian
    p1 = np.array([np.sin(theta0)*np.cos(phi0), np.sin(theta0)*np.sin(phi0), np.cos(theta0)])
    p2 = np.array([np.sin(theta_end)*np.cos(phi0), np.sin(theta_end)*np.sin(phi0), np.cos(theta_end)])
    euclidean_dists.append(np.linalg.norm(p2 - p1))

ax2.plot(radii, radii, '-', color='#cc3333', linewidth=2.5,
         label=r'Tangent space distance $\|v\|$')
ax2.plot(radii, radii, '--', color='#339933', linewidth=2.5,
         label=r'Geodesic distance ($= \|v\|$)', alpha=0.7)
ax2.plot(radii, euclidean_dists, '-.', color='#4488cc', linewidth=2.5,
         label=r'Euclidean (chord) distance')
ax2.set_xlabel(r'Tangent vector norm $\|v\|$', fontsize=12)
ax2.set_ylabel(r'Distance', fontsize=12)
ax2.set_title(r'Tangent Space vs Geodesic vs Euclidean', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.annotate(r'$\exp_p$ preserves' + '\n' + r'geodesic distance' + '\n' + r'by construction',
            xy=(1.5, 1.5), xytext=(2.0, 0.8),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig("exponential_map.png", dpi=150, bbox_inches='tight')
plt.show()
```

---

## Connections to Machine Learning

The geometry described above is not abstract mathematics disconnected from practice. It appears throughout machine learning in concrete, consequential ways.

### Data Manifolds and Dimensionality Reduction

The **manifold hypothesis** states that high-dimensional data (images, text embeddings, molecular structures) lies near a low-dimensional manifold embedded in the ambient space. Techniques like t-SNE, UMAP, and autoencoders attempt to learn this manifold structure. The latent space of a VAE is an explicit parameterization of the data manifold.

The key implication: **Euclidean distance in the ambient space is a poor measure of similarity.** Two points might be close in $\mathbb{R}^D$ but far apart along the manifold (imagine two points on opposite sides of a thin spiral). Geodesic distance along the manifold is the right metric.

### Natural Gradient Descent

Standard gradient descent updates parameters by $\theta \leftarrow \theta - \eta \nabla_\theta L$. But this treats the parameter space as flat $\mathbb{R}^n$, which it is not. Different parameterizations of the same model lead to different gradient directions, even though the underlying function is the same.

**Natural gradient descent** (Amari, 1998) fixes this by using the **Fisher information metric** as a Riemannian metric on the space of probability distributions. The Fisher information matrix is:

$$F_{ij}(\theta) = \mathbb{E}_{p(x|\theta)}\left[\frac{\partial \log p(x|\theta)}{\partial \theta^i} \frac{\partial \log p(x|\theta)}{\partial \theta^j}\right]$$

This is a Riemannian metric tensor. The natural gradient is:

$$\tilde{\nabla}_\theta L = F^{-1}(\theta) \nabla_\theta L$$

This is the steepest descent direction measured in the Fisher metric --- it is the direction that decreases the loss fastest per unit of KL divergence in parameter space, rather than per unit of Euclidean distance.

In practice, computing $F^{-1}$ is expensive for large models. Approximations like K-FAC (Kronecker-Factored Approximate Curvature) approximate the Fisher information to make natural gradient tractable. Adam and other adaptive optimizers can be interpreted as diagonal approximations to the natural gradient.

### Riemannian Optimization

Many ML problems have constraints that define a manifold:

- **Orthogonal matrices** (in recurrent networks for stability): the Stiefel manifold.
- **Low-rank matrices** (in matrix completion, LoRA): a manifold of fixed-rank matrices.
- **Positive definite matrices** (covariance estimation): a manifold with a natural hyperbolic-like geometry.
- **The probability simplex** (topic models, softmax outputs): a manifold with the Fisher metric.

Riemannian optimization algorithms (Riemannian gradient descent, Riemannian conjugate gradient, Riemannian BFGS) perform optimization directly on these manifolds. The key operations are:

1. Compute the **Riemannian gradient** (project the Euclidean gradient onto the tangent space).
2. **Retract** (use the exponential map or an approximation to move along the manifold in the gradient direction).

This is more principled and often more efficient than projecting back onto the constraint set after each Euclidean update.

### Why Euclidean Distance Fails in Latent Spaces

Consider a VAE with a Gaussian encoder $q(z|x) = \mathcal{N}(\mu(x), \sigma^2(x)I)$. The latent space $\mathbb{R}^d$ has a non-trivial geometry induced by the decoder: the "natural" metric is the pullback of some metric on the data space through the decoder.

Two latent codes $z_1$ and $z_2$ may be close in Euclidean distance but decode to very different outputs if they lie on opposite sides of a "fold" in the decoder's mapping. Conversely, two codes far apart in Euclidean distance might decode to similar outputs if the decoder is nearly constant in some region.

The right approach: define a Riemannian metric on the latent space using the Jacobian of the decoder $J = \frac{\partial D(z)}{\partial z}$:

$$g_{ij}(z) = J^T J = \sum_k \frac{\partial D_k}{\partial z_i} \frac{\partial D_k}{\partial z_j}$$

This is the **pullback metric**: it measures distances in latent space according to how much the decoded output changes. Geodesics in this metric correspond to smooth, perceptually uniform interpolations in data space.

---

## Conclusion

Differential geometry provides the right language for spaces that are not flat. The core objects --- manifolds, tangent spaces, metrics, geodesics, curvature --- are not abstract luxuries. They are the tools you need when Euclidean geometry gives wrong answers.

The chain of ideas:

1. **Manifolds** are spaces that look flat locally but curve globally. Data lives on manifolds.
2. **Tangent spaces** are where vectors (gradients, velocities) live. You cannot add vectors at different points without a connection.
3. **Riemannian metrics** let you measure distances and angles. Different metrics give different geometries on the same underlying space.
4. **Geodesics** are the shortest paths. They replace straight lines and depend on the metric.
5. **Curvature** measures how the space deviates from flatness. It affects everything from volume estimates to parallel transport to the convergence of optimization algorithms.
6. **The exponential and logarithmic maps** translate between the tangent space (where linear algebra works) and the manifold (where the data lives).

The practical takeaway: whenever your data, parameters, or distributions have constraints or non-Euclidean structure, the tools of differential geometry tell you how to work with them correctly. Natural gradient descent, Riemannian optimization, geodesic interpolation in latent spaces, and manifold learning all rest on this foundation.
