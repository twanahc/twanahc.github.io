---
layout: post
title: "3D Gaussian Splatting and Differentiable Rendering: From Point Clouds to Real-Time Novel Views"
date: 2026-02-21
category: math
---

For decades, computer graphics has rendered 3D scenes using triangle meshes --- explicit surfaces defined by vertices and faces. This works beautifully when you have a hand-crafted 3D model, but it fails when you want to reconstruct a scene from photographs alone. You do not know the mesh. You do not know the surface. You have pixels.

**3D Gaussian Splatting** (Kerbl et al., 2023) offers a radically different representation. Instead of triangles, represent a scene as a cloud of millions of tiny 3D Gaussian blobs, each with a position, shape (covariance), opacity, and view-dependent color. To render an image, project these Gaussians onto the camera's image plane and blend them front-to-back using alpha compositing. The entire pipeline --- projection, sorting, blending --- is differentiable, so you can optimize all the Gaussian parameters by comparing rendered images to real photographs using gradient descent.

The result: photorealistic novel view synthesis at real-time frame rates (100+ FPS), trained from a set of photographs in minutes. This is transforming video generation, enabling explicit 3D camera control, world models, and 4D dynamic scene reconstruction.

This post derives the full mathematical pipeline. We start with multivariate Gaussians in 3D, derive the projection formula that maps a 3D covariance to a 2D covariance via the Jacobian of perspective projection, explain alpha compositing from the volume rendering integral, introduce spherical harmonics for view-dependent color, and cover the adaptive densification strategy. We then extend to dynamic 4D Gaussian splatting and connect to video generation.

---

## Table of Contents

1. [The Representation Problem](#the-representation-problem)
2. [3D Gaussian Primitives](#3d-gaussian-primitives)
3. [Projecting Gaussians: From 3D to 2D](#projecting-gaussians-from-3d-to-2d)
4. [Alpha Compositing and the Rendering Equation](#alpha-compositing-and-the-rendering-equation)
5. [Spherical Harmonics for View-Dependent Color](#spherical-harmonics-for-view-dependent-color)
6. [The Full Rendering Pipeline](#the-full-rendering-pipeline)
7. [Differentiable Rendering and Optimization](#differentiable-rendering-and-optimization)
8. [Adaptive Density Control](#adaptive-density-control)
9. [Training: Loss Functions and Initialization](#training-loss-functions-and-initialization)
10. [Dynamic Gaussian Splatting for Video](#dynamic-gaussian-splatting-for-video)
11. [Connection to Video Generation](#connection-to-video-generation)
12. [Python: 2D Gaussian Splatting Simulation](#python-2d-gaussian-splatting-simulation)

---

## The Representation Problem

To render a 3D scene from a novel camera viewpoint, you need a 3D representation. The major options are:

**Triangle meshes.** Explicit surfaces defined by vertices, edges, and faces. Fast to render (hardware-accelerated rasterization), but hard to optimize --- moving vertices requires careful handling of topology, and gradients through rasterization are discontinuous.

**Voxel grids.** Divide 3D space into a regular grid, store density and color per voxel. Simple and differentiable, but memory scales as \(O(N^3)\) --- a \(512^3\) grid requires 134 million voxels, most of which are empty.

**Implicit functions (NeRF).** Represent the scene as a neural network \(F_\theta(x, y, z, \theta, \phi) \to (\sigma, \mathbf{c})\) mapping 3D coordinates and viewing direction to density and color. Compact and flexible, but rendering requires marching rays through the volume with hundreds of network evaluations per ray. Slow.

**Point clouds / Gaussian splats.** Represent the scene as a collection of points, each carrying attributes. Memory is proportional to the scene complexity, not the volume. Rendering is done by **splatting** (projecting points onto the image plane), which is fast and parallelizable.

3D Gaussian Splatting chooses the last option and enriches each point with a full 3D Gaussian distribution (giving it spatial extent and shape), opacity, and spherical harmonic color coefficients.

---

## 3D Gaussian Primitives

Each Gaussian primitive \(i\) in the scene is parameterized by:

**Position** \(\boldsymbol{\mu}_i \in \mathbb{R}^3\): the center of the Gaussian in world coordinates.

**Covariance** \(\Sigma_i \in \mathbb{R}^{3 \times 3}\): a symmetric positive-definite matrix defining the shape and orientation of the Gaussian. The Gaussian's density in 3D is:

$$G(\mathbf{x}) = \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

The covariance matrix \(\Sigma\) encodes an **ellipsoid**. Its eigendecomposition \(\Sigma = R \Lambda R^T\) reveals:
- The eigenvectors (columns of \(R\)) give the **orientation** of the ellipsoid's principal axes
- The eigenvalues \(\lambda_1, \lambda_2, \lambda_3\) in \(\Lambda\) give the **squared semi-axis lengths** (variances along each axis)

To ensure \(\Sigma\) stays positive definite during gradient-based optimization, it is parameterized as:

$$\Sigma = R S S^T R^T$$

where \(R\) is a rotation matrix (stored as a unit quaternion \(\mathbf{q} \in \mathbb{R}^4\)) and \(S = \text{diag}(s_1, s_2, s_3)\) is a diagonal scaling matrix with \(s_i > 0\). This guarantees positive definiteness by construction.

**Opacity** \(\alpha_i \in (0, 1)\): how opaque the Gaussian is at its center. Parameterized via a sigmoid to keep it in range.

**Color** \(\mathbf{c}_i\): represented via spherical harmonic coefficients (explained below) to capture view-dependent appearance.

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">3D Gaussian Primitive: Parameterization</text>

  <!-- Ellipsoid (tilted ellipse as projection) -->
  <ellipse cx="200" cy="160" rx="80" ry="40" fill="#2196F3" opacity="0.15" stroke="#2196F3" stroke-width="2" transform="rotate(-25 200 160)"/>
  <ellipse cx="200" cy="160" rx="80" ry="40" fill="none" stroke="#2196F3" stroke-width="1" stroke-dasharray="4,3" transform="rotate(-25 200 160)"/>

  <!-- Center point -->
  <circle cx="200" cy="160" r="4" fill="#FF9800"/>
  <text x="215" y="155" font-size="11" fill="#FF9800">μ (position)</text>

  <!-- Principal axes -->
  <line x1="200" y1="160" x2="272" y2="130" stroke="#E53935" stroke-width="2"/>
  <text x="278" y="128" font-size="10" fill="#E53935">s₁ (scale)</text>
  <line x1="200" y1="160" x2="220" y2="200" stroke="#66bb6a" stroke-width="2"/>
  <text x="225" y="210" font-size="10" fill="#66bb6a">s₂</text>

  <!-- Rotation arc -->
  <path d="M 270,160 A 70,70 0 0,0 255,120" fill="none" stroke="#d4d4d4" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="280" y="145" font-size="10" fill="#d4d4d4">R (rotation)</text>

  <!-- Parameter list -->
  <text x="480" y="80" font-size="12" fill="#d4d4d4" font-weight="bold">Per-Gaussian Parameters:</text>
  <text x="480" y="105" font-size="11" fill="#FF9800">μ ∈ ℝ³   — position (3)</text>
  <text x="480" y="125" font-size="11" fill="#E53935">q ∈ ℝ⁴   — rotation quaternion (4)</text>
  <text x="480" y="145" font-size="11" fill="#66bb6a">s ∈ ℝ³   — scale (3)</text>
  <text x="480" y="165" font-size="11" fill="#4fc3f7">α ∈ (0,1) — opacity (1)</text>
  <text x="480" y="185" font-size="11" fill="#CE93D8">SH ∈ ℝ⁴⁸  — color coefficients (48)</text>
  <line x1="480" y1="195" x2="650" y2="195" stroke="#666" stroke-width="1"/>
  <text x="480" y="215" font-size="11" fill="#d4d4d4" font-weight="bold">Total: 59 floats per Gaussian</text>
  <text x="480" y="240" font-size="10" fill="#999">Typical scene: 1-5 million Gaussians</text>
</svg>

---

## Projecting Gaussians: From 3D to 2D

To render, we need to project each 3D Gaussian onto the 2D image plane. The key mathematical result: **the projection of a 3D Gaussian through a perspective camera is approximately a 2D Gaussian**, and its 2D covariance can be computed in closed form.

### The Projection Pipeline

Let \(\boldsymbol{\mu}\) be the Gaussian's center in world coordinates and \(\Sigma\) its 3D covariance. The camera has a view matrix \(W\) (world-to-camera transform) and projection matrix \(P\).

**Step 1: Transform to camera space.** The Gaussian center in camera coordinates is:

$$\boldsymbol{\mu}' = W \boldsymbol{\mu}$$

The 3D covariance transforms as:

$$\Sigma' = W \Sigma W^T$$

(covariance matrices transform by conjugation, not by simple multiplication, because they are quadratic forms).

**Step 2: Project to image plane.** Perspective projection is nonlinear: \((x, y, z) \mapsto (fx/z, fy/z)\) where \(f\) is the focal length. A Gaussian passed through a nonlinear function is not exactly Gaussian, but we can approximate by linearizing.

The Jacobian of the perspective projection at the Gaussian center \(\boldsymbol{\mu}' = (\mu_x', \mu_y', \mu_z')\) is:

$$J = \begin{pmatrix} f / \mu_z' & 0 & -f \mu_x' / (\mu_z')^2 \\ 0 & f / \mu_z' & -f \mu_y' / (\mu_z')^2 \end{pmatrix}$$

This is a \(2 \times 3\) matrix (projecting from 3D to 2D).

**Step 3: Compute the 2D covariance.** Using the first-order approximation (local affine), the projected 2D covariance is:

$$\Sigma_{2D} = J \Sigma' J^T = J W \Sigma W^T J^T$$

This \(2 \times 2\) symmetric positive-definite matrix defines an ellipse on the image plane --- the "footprint" of the projected Gaussian. Its eigenvalues give the semi-axis lengths of the ellipse, and its eigenvectors give the orientation.

The projected 2D Gaussian at pixel position \(\mathbf{p}\) is:

$$G_{2D}(\mathbf{p}) = \exp\!\left(-\frac{1}{2}(\mathbf{p} - \boldsymbol{\mu}_{2D})^T \Sigma_{2D}^{-1} (\mathbf{p} - \boldsymbol{\mu}_{2D})\right)$$

where \(\boldsymbol{\mu}_{2D}\) is the projected center.

<svg viewBox="0 0 700 300" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <defs>
    <marker id="arrowProj" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#FF9800"/>
    </marker>
  </defs>
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">3D Gaussian → 2D Projection</text>

  <!-- 3D ellipsoid -->
  <ellipse cx="150" cy="140" rx="70" ry="35" fill="#2196F3" opacity="0.15" stroke="#2196F3" stroke-width="2" transform="rotate(-20 150 140)"/>
  <circle cx="150" cy="140" r="3" fill="#FF9800"/>
  <text x="150" y="200" text-anchor="middle" font-size="11" fill="#2196F3">3D Gaussian (Σ₃ₓ₃)</text>

  <!-- Projection lines -->
  <line x1="190" y1="125" x2="420" y2="100" stroke="#FF9800" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arrowProj)"/>
  <line x1="190" y1="155" x2="420" y2="180" stroke="#FF9800" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arrowProj)"/>

  <!-- Camera/image plane -->
  <rect x="420" y="60" width="180" height="200" rx="3" fill="none" stroke="#666" stroke-width="2"/>
  <text x="510" y="55" text-anchor="middle" font-size="11" fill="#999">Image plane</text>

  <!-- 2D ellipse on image plane -->
  <ellipse cx="510" cy="150" rx="50" ry="30" fill="#4fc3f7" opacity="0.2" stroke="#4fc3f7" stroke-width="2" transform="rotate(-10 510 150)"/>
  <circle cx="510" cy="150" r="3" fill="#FF9800"/>
  <text x="510" y="210" text-anchor="middle" font-size="11" fill="#4fc3f7">2D Gaussian (Σ₂ₓ₂)</text>

  <!-- Formula -->
  <text x="310" y="250" text-anchor="middle" font-size="12" fill="#d4d4d4">Σ₂D = J W Σ Wᵀ Jᵀ</text>
  <text x="310" y="275" text-anchor="middle" font-size="10" fill="#999">J = Jacobian of perspective projection</text>
</svg>

---

## Alpha Compositing and the Rendering Equation

With all Gaussians projected to 2D, we need to blend them together to produce the final pixel color. This is done via **front-to-back alpha compositing**, which is derived from the volume rendering integral.

### From Volume Rendering to Alpha Compositing

The volume rendering equation for a ray passing through a participating medium is:

$$C = \int_0^\infty T(t) \, \sigma(t) \, \mathbf{c}(t) \, dt$$

where \(T(t) = \exp\!\left(-\int_0^t \sigma(s) \, ds\right)\) is the **transmittance** (probability of the ray reaching distance \(t\) without being absorbed) and \(\sigma(t)\) is the density at distance \(t\).

For a discrete set of \(N\) Gaussians sorted by depth (front-to-back), this integral discretizes to:

$$C = \sum_{i=1}^{N} T_i \, \alpha_i \, \mathbf{c}_i$$

where:

$$\alpha_i = o_i \cdot G_{2D}^{(i)}(\mathbf{p})$$

is the effective opacity of Gaussian \(i\) at pixel \(\mathbf{p}\) (base opacity \(o_i\) times the Gaussian's value at that pixel), and the transmittance is:

$$T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)$$

This is the standard **over** operator from computer graphics: each Gaussian contributes its color weighted by its opacity, attenuated by all the Gaussians in front of it.

The transmittance \(T_i\) decreases monotonically as we accumulate Gaussians, and once it drops below a threshold (e.g., \(T_i < 0.001\)), we can stop --- the remaining Gaussians contribute negligibly. This early termination is crucial for performance.

### Why Sorting Matters

The formula above requires processing Gaussians in depth order. 3D Gaussian Splatting uses **tile-based rendering**: divide the image into \(16 \times 16\) pixel tiles, assign each Gaussian to the tiles it overlaps, sort Gaussians per tile by depth, then composite within each tile in parallel. This is extremely GPU-friendly.

---

## Spherical Harmonics for View-Dependent Color

Real-world surfaces exhibit view-dependent appearance: specular highlights, reflections, subsurface scattering. The color of a surface point changes depending on your viewing angle. To capture this, each Gaussian stores not a single RGB color but a set of **spherical harmonic (SH) coefficients** that encode a function of viewing direction.

### What Are Spherical Harmonics?

Spherical harmonics \(Y_l^m(\theta, \phi)\) are an orthonormal basis for functions on the sphere, analogous to Fourier series for functions on a circle. They are indexed by **degree** \(l \geq 0\) and **order** \(-l \leq m \leq l\).

The first few (real) spherical harmonics:

**Band 0** (constant, 1 function):
$$Y_0^0 = \frac{1}{2\sqrt{\pi}}$$

**Band 1** (linear, 3 functions):
$$Y_1^{-1} = \sqrt{\frac{3}{4\pi}} y, \quad Y_1^0 = \sqrt{\frac{3}{4\pi}} z, \quad Y_1^1 = \sqrt{\frac{3}{4\pi}} x$$

where \((x, y, z)\) is the unit viewing direction vector.

**Band 2** (quadratic, 5 functions):
$$Y_2^{-2} = \sqrt{\frac{15}{4\pi}} xy, \quad Y_2^{-1} = \sqrt{\frac{15}{4\pi}} yz, \quad Y_2^0 = \sqrt{\frac{5}{16\pi}}(3z^2 - 1), \quad \ldots$$

Each degree \(l\) has \(2l + 1\) basis functions. Through degree \(l_{\max}\), there are \((l_{\max} + 1)^2\) total coefficients. 3D Gaussian Splatting typically uses degree 3, giving \(16\) coefficients per color channel, or \(48\) total for RGB.

The color for a given viewing direction \(\mathbf{d}\) is:

$$\mathbf{c}(\mathbf{d}) = \sum_{l=0}^{l_{\max}} \sum_{m=-l}^{l} \mathbf{k}_l^m \, Y_l^m(\mathbf{d})$$

where \(\mathbf{k}_l^m \in \mathbb{R}^3\) are the per-Gaussian, per-channel SH coefficients.

Band 0 captures the average color (diffuse). Band 1 adds directional shading. Bands 2+ add specular-like effects. This provides a compact, differentiable representation of view-dependent appearance.

---

## The Full Rendering Pipeline

Putting it all together, here is the complete pipeline for rendering a single image:

1. **For each Gaussian:** Transform center to camera coordinates, compute 2D projected covariance \(\Sigma_{2D} = J W \Sigma W^T J^T\), compute 2D center \(\boldsymbol{\mu}_{2D}\).

2. **Tile assignment:** Determine which \(16 \times 16\) pixel tiles each projected Gaussian overlaps (using the 2D ellipse's bounding box).

3. **Sort by depth:** Within each tile, sort assigned Gaussians by their camera-space depth \(\mu_z'\).

4. **Composite per pixel:** For each pixel \(\mathbf{p}\) in a tile, iterate through sorted Gaussians front-to-back:
   - Evaluate the 2D Gaussian: \(G_{2D}^{(i)}(\mathbf{p})\)
   - Compute effective opacity: \(\alpha_i = o_i \cdot G_{2D}^{(i)}(\mathbf{p})\)
   - Evaluate view-dependent color: \(\mathbf{c}_i = \text{SH}(\mathbf{d}_i)\)
   - Accumulate: \(C \mathrel{+}= T \cdot \alpha_i \cdot \mathbf{c}_i\), then \(T \mathrel{*}= (1 - \alpha_i)\)
   - Early termination: stop if \(T < \epsilon\)

5. **Output pixel color** \(C\).

This entire pipeline runs on GPU. The tile-based approach avoids global sorting (only local per-tile sorting) and enables massive parallelism.

<svg viewBox="0 0 700 200" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="20" text-anchor="middle" font-size="13" font-weight="bold" fill="#d4d4d4">Tile-Based Rasterization Pipeline</text>

  <!-- Pipeline boxes -->
  <rect x="20" y="50" width="110" height="50" rx="5" fill="#2196F3" opacity="0.2" stroke="#2196F3" stroke-width="1.5"/>
  <text x="75" y="72" text-anchor="middle" font-size="10" fill="#2196F3">Project to 2D</text>
  <text x="75" y="88" text-anchor="middle" font-size="9" fill="#999">Σ₂D = JWΣWᵀJᵀ</text>

  <text x="145" y="78" text-anchor="middle" font-size="16" fill="#666">→</text>

  <rect x="160" y="50" width="110" height="50" rx="5" fill="#FF9800" opacity="0.2" stroke="#FF9800" stroke-width="1.5"/>
  <text x="215" y="72" text-anchor="middle" font-size="10" fill="#FF9800">Tile Assignment</text>
  <text x="215" y="88" text-anchor="middle" font-size="9" fill="#999">16×16 pixel tiles</text>

  <text x="285" y="78" text-anchor="middle" font-size="16" fill="#666">→</text>

  <rect x="300" y="50" width="110" height="50" rx="5" fill="#E53935" opacity="0.2" stroke="#E53935" stroke-width="1.5"/>
  <text x="355" y="72" text-anchor="middle" font-size="10" fill="#E53935">Sort by Depth</text>
  <text x="355" y="88" text-anchor="middle" font-size="9" fill="#999">Per-tile sorting</text>

  <text x="425" y="78" text-anchor="middle" font-size="16" fill="#666">→</text>

  <rect x="440" y="50" width="110" height="50" rx="5" fill="#66bb6a" opacity="0.2" stroke="#66bb6a" stroke-width="1.5"/>
  <text x="495" y="72" text-anchor="middle" font-size="10" fill="#66bb6a">α-Composite</text>
  <text x="495" y="88" text-anchor="middle" font-size="9" fill="#999">Front-to-back</text>

  <text x="565" y="78" text-anchor="middle" font-size="16" fill="#666">→</text>

  <rect x="580" y="50" width="100" height="50" rx="5" fill="#CE93D8" opacity="0.2" stroke="#CE93D8" stroke-width="1.5"/>
  <text x="630" y="72" text-anchor="middle" font-size="10" fill="#CE93D8">Output Image</text>
  <text x="630" y="88" text-anchor="middle" font-size="9" fill="#999">100+ FPS</text>

  <!-- Key insight -->
  <text x="350" y="140" text-anchor="middle" font-size="11" fill="#d4d4d4">Every step is differentiable — gradients flow back through compositing,</text>
  <text x="350" y="158" text-anchor="middle" font-size="11" fill="#d4d4d4">projection, and into the 3D Gaussian parameters.</text>
</svg>

---

## Differentiable Rendering and Optimization

The key innovation is that the entire rendering pipeline is **differentiable**. Given a rendered image \(\hat{I}\) and a ground-truth photograph \(I^*\), we can compute a loss \(\mathcal{L}(\hat{I}, I^*)\) and backpropagate gradients all the way to the Gaussian parameters.

### Gradient Flow

The chain rule propagates gradients through:

$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}} = \frac{\partial \mathcal{L}}{\partial C} \cdot \frac{\partial C}{\partial \alpha_i} \cdot \frac{\partial \alpha_i}{\partial G_{2D}} \cdot \frac{\partial G_{2D}}{\partial \boldsymbol{\mu}_{2D}} \cdot \frac{\partial \boldsymbol{\mu}_{2D}}{\partial \boldsymbol{\mu}}$$

Each factor is computed in closed form:

- \(\partial C / \partial \alpha_i\): from the alpha compositing formula
- \(\partial \alpha_i / \partial G_{2D}\): simply \(o_i\) (the base opacity)
- \(\partial G_{2D} / \partial \boldsymbol{\mu}_{2D}\): gradient of a Gaussian (which is the Gaussian times the Mahalanobis direction)
- \(\partial \boldsymbol{\mu}_{2D} / \partial \boldsymbol{\mu}\): the Jacobian of projection

Similarly, gradients flow to the covariance parameters (via \(\Sigma_{2D}\)), the opacity, and the SH coefficients.

The gradient through alpha compositing is particularly elegant. For the accumulated color at pixel \(\mathbf{p}\):

$$\frac{\partial C}{\partial \alpha_i} = T_i \left(\mathbf{c}_i - \frac{1}{1 - \alpha_i} \sum_{j=i+1}^{N} T_j \alpha_j \mathbf{c}_j \right)$$

The term in parentheses compares Gaussian \(i\)'s color to the weighted average of everything behind it. If Gaussian \(i\) has the "wrong" color, the gradient pushes its opacity down.

---

## Adaptive Density Control

Starting from an initial set of Gaussians (typically from Structure-from-Motion point clouds), the scene needs more Gaussians in detailed regions and fewer in empty space. 3D Gaussian Splatting uses **adaptive density control** during training:

**Densification.** Every \(N\) iterations (e.g., 100), check each Gaussian's accumulated position gradient magnitude \(\|\partial \mathcal{L} / \partial \boldsymbol{\mu}_{2D}\|\). Large gradients indicate the Gaussian is trying to move a lot --- it is either too big (covering a region with fine detail) or in the wrong place.

- If the Gaussian is **large** (its scale exceeds a threshold) and has large gradients: **split** it into two smaller Gaussians, each with half the scale.
- If the Gaussian is **small** and has large gradients: **clone** it, creating a copy shifted in the gradient direction. This fills in under-represented regions.

**Pruning.** Remove Gaussians with very low opacity (\(\alpha < \epsilon_\alpha\)). Also periodically reset opacities to a low value and let the optimization decide which Gaussians are truly needed.

<svg viewBox="0 0 700 200" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="20" text-anchor="middle" font-size="13" font-weight="bold" fill="#d4d4d4">Adaptive Density Control</text>

  <!-- Split -->
  <ellipse cx="100" cy="100" rx="60" ry="30" fill="#E53935" opacity="0.2" stroke="#E53935" stroke-width="1.5"/>
  <text x="100" y="105" text-anchor="middle" font-size="10" fill="#E53935">Large + high grad</text>
  <text x="100" y="150" text-anchor="middle" font-size="11" fill="#E53935" font-weight="bold">SPLIT</text>
  <text x="175" y="100" font-size="16" fill="#666">→</text>
  <ellipse cx="225" cy="85" rx="30" ry="15" fill="#E53935" opacity="0.2" stroke="#E53935" stroke-width="1"/>
  <ellipse cx="225" cy="115" rx="30" ry="15" fill="#E53935" opacity="0.2" stroke="#E53935" stroke-width="1"/>

  <!-- Clone -->
  <ellipse cx="380" cy="100" rx="20" ry="20" fill="#4fc3f7" opacity="0.2" stroke="#4fc3f7" stroke-width="1.5"/>
  <text x="380" y="105" text-anchor="middle" font-size="10" fill="#4fc3f7">Small + high grad</text>
  <text x="380" y="150" text-anchor="middle" font-size="11" fill="#4fc3f7" font-weight="bold">CLONE</text>
  <text x="420" y="100" font-size="16" fill="#666">→</text>
  <ellipse cx="460" cy="90" rx="20" ry="20" fill="#4fc3f7" opacity="0.2" stroke="#4fc3f7" stroke-width="1"/>
  <ellipse cx="480" cy="110" rx="20" ry="20" fill="#4fc3f7" opacity="0.2" stroke="#4fc3f7" stroke-width="1"/>

  <!-- Prune -->
  <ellipse cx="600" cy="100" rx="25" ry="25" fill="#999" opacity="0.1" stroke="#999" stroke-width="1.5" stroke-dasharray="4,3"/>
  <text x="600" y="105" text-anchor="middle" font-size="10" fill="#999">Low opacity</text>
  <text x="600" y="150" text-anchor="middle" font-size="11" fill="#999" font-weight="bold">PRUNE</text>
  <text x="640" y="100" font-size="16" fill="#666">→</text>
  <text x="670" y="105" font-size="14" fill="#666">×</text>
</svg>

---

## Training: Loss Functions and Initialization

### Loss Function

The training loss combines an L1 photometric loss with a structural similarity term:

$$\mathcal{L} = (1 - \lambda) \|\hat{I} - I^*\|_1 + \lambda \cdot \mathcal{L}_{\text{D-SSIM}}(\hat{I}, I^*)$$

where \(\lambda = 0.2\) typically. D-SSIM captures structural similarity at a perceptual level, preventing the optimizer from ignoring low-contrast regions.

### Initialization

Gaussians are initialized from a sparse point cloud produced by **Structure from Motion** (SfM), typically using COLMAP. Each SfM point becomes a Gaussian with:
- Position: the 3D point
- Scale: proportional to the distance to the nearest neighbor
- Opacity: initialized to a moderate value (e.g., 0.1 after sigmoid)
- SH coefficients: initialized from the point's color (band 0 only)

Training runs for 30,000 iterations on a single GPU, taking 20--40 minutes for a typical scene. The optimizer is Adam with different learning rates per parameter group (positions get a spatially adaptive rate based on scene extent).

---

## Dynamic Gaussian Splatting for Video

Static 3D Gaussian Splatting captures a frozen moment. For video, we need Gaussians that move and deform over time. Several approaches exist:

### Deformation Fields

**4D Gaussian Splatting (Wu et al., 2023):** Add a neural deformation network \(D_\theta(t, \boldsymbol{\mu}_i) \to (\Delta \boldsymbol{\mu}, \Delta \mathbf{q}, \Delta \mathbf{s})\) that predicts per-Gaussian, per-timestep offsets to position, rotation, and scale. The canonical Gaussians live in a reference frame, and the deformation field warps them to each timestep.

$$\boldsymbol{\mu}_i(t) = \boldsymbol{\mu}_i^{\text{canonical}} + \Delta \boldsymbol{\mu}_i(t)$$

This is analogous to the D-NeRF approach, but applied to explicit Gaussians instead of implicit fields. The advantages: rendering remains fast (just splat the deformed Gaussians), and the deformation field can be lightweight since most of the scene structure is captured by the canonical Gaussians.

### Per-Timestep Gaussians

An alternative: simply optimize independent Gaussians per timestep, with a temporal regularization loss that penalizes large differences between consecutive frames. This is simpler but uses more memory and can produce temporal artifacts.

### Gaussian Flows

Attach a velocity field to each Gaussian: \(\mathbf{v}_i \in \mathbb{R}^3\). Between frames, Gaussians move according to \(\boldsymbol{\mu}_i(t + \Delta t) = \boldsymbol{\mu}_i(t) + \mathbf{v}_i \Delta t\). This enforces physically plausible motion and enables frame interpolation by evaluating at intermediate times.

---

## Connection to Video Generation

3D Gaussian Splatting is converging with generative video models in several ways:

**Camera control.** Video generation models like Sora and Veo3 can generate videos with specified camera trajectories. Under the hood, this requires 3D understanding. Models can be trained on data rendered from 3D Gaussian reconstructions, providing ground-truth camera poses and multi-view consistency.

**World models.** A "world model" predicts how a 3D scene evolves over time, including physics, occlusion, and novel view synthesis. 3D Gaussian Splatting provides the representational backbone: the world is a set of Gaussians that move, deform, appear, and disappear.

**3D-aware generation.** Instead of generating video frames in 2D pixel space, generate or refine a 3D Gaussian scene and render it from a camera path. This guarantees multi-view consistency by construction and allows post-hoc camera changes.

**Feed-forward reconstruction.** Recent work (e.g., pixelSplat, MVSplat) trains networks to predict 3D Gaussians directly from one or two input images in a single forward pass, without per-scene optimization. Combined with video generation, this enables: generate a keyframe with a 2D model → lift to 3D Gaussians → render novel views → animate.

---

## Python: 2D Gaussian Splatting Simulation

```python
import numpy as np
import matplotlib.pyplot as plt

def gaussian_2d(pos, mu, cov_inv, det_cov):
    """Evaluate 2D Gaussian at positions pos given mean mu and inverse covariance."""
    diff = pos - mu  # shape: (N, 2)
    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
    return np.exp(exponent)

def render_gaussians(gaussians, width=256, height=256):
    """Render a set of 2D Gaussians via alpha compositing."""
    # Create pixel grid
    y, x = np.mgrid[0:height, 0:width]
    pixels = np.stack([x.ravel(), y.ravel()], axis=-1).astype(float)  # (H*W, 2)

    # Sort gaussians by depth (z-order)
    gaussians = sorted(gaussians, key=lambda g: g['depth'])

    # Alpha compositing
    canvas = np.zeros((height * width, 3))
    transmittance = np.ones(height * width)

    for g in gaussians:
        mu = np.array(g['mu'])
        cov = np.array(g['cov'])
        cov_inv = np.linalg.inv(cov)
        base_alpha = g['opacity']
        color = np.array(g['color'])

        # Evaluate Gaussian at all pixels
        gauss_val = gaussian_2d(pixels, mu, cov_inv, np.linalg.det(cov))

        # Effective alpha
        alpha = base_alpha * gauss_val

        # Alpha compositing
        contribution = transmittance * alpha
        canvas += contribution[:, None] * color[None, :]
        transmittance *= (1 - alpha)

        # Early termination check (skip if all transmittance is tiny)
        if np.max(transmittance) < 0.001:
            break

    return canvas.reshape(height, width, 3)

# Define some 2D Gaussians (simulating a simple scene)
np.random.seed(42)
gaussians = []
for i in range(30):
    angle = np.random.uniform(0, 2 * np.pi)
    scale_x = np.random.uniform(5, 30)
    scale_y = np.random.uniform(5, 30)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    S = np.diag([scale_x**2, scale_y**2])
    cov = R @ S @ R.T

    gaussians.append({
        'mu': [np.random.uniform(30, 226), np.random.uniform(30, 226)],
        'cov': cov,
        'opacity': np.random.uniform(0.3, 0.95),
        'color': np.random.uniform(0.1, 1.0, size=3),
        'depth': np.random.uniform(0, 10)
    })

# Render
image = render_gaussians(gaussians, width=256, height=256)
image = np.clip(image, 0, 1)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Rendered image
axes[0].imshow(image)
axes[0].set_title('Rendered Scene (Alpha Compositing)')
axes[0].axis('off')

# Show individual Gaussian centers and ellipses
axes[1].set_xlim(0, 256)
axes[1].set_ylim(256, 0)
axes[1].set_aspect('equal')
for g in gaussians:
    mu = g['mu']
    cov = g['cov']
    # Eigendecomposition for ellipse
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle_deg = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    from matplotlib.patches import Ellipse
    ell = Ellipse(xy=mu, width=2*np.sqrt(eigvals[0]), height=2*np.sqrt(eigvals[1]),
                  angle=angle_deg, facecolor=g['color'], alpha=g['opacity']*0.4,
                  edgecolor='white', linewidth=0.5)
    axes[1].add_patch(ell)
    axes[1].plot(mu[0], mu[1], 'w.', markersize=2)
axes[1].set_title('Gaussian Ellipses (2D)')
axes[1].set_facecolor('#1a1a1a')

# Transmittance map
y, x = np.mgrid[0:256, 0:256]
pixels = np.stack([x.ravel(), y.ravel()], axis=-1).astype(float)
transmittance = np.ones(256 * 256)
for g in sorted(gaussians, key=lambda g: g['depth']):
    mu = np.array(g['mu'])
    cov_inv = np.linalg.inv(np.array(g['cov']))
    gauss_val = gaussian_2d(pixels, mu, cov_inv, 1.0)
    alpha = g['opacity'] * gauss_val
    transmittance *= (1 - alpha)
transmittance_img = transmittance.reshape(256, 256)
axes[2].imshow(transmittance_img, cmap='inferno', vmin=0, vmax=1)
axes[2].set_title(r'Transmittance $T$ (dark = opaque)')
axes[2].axis('off')

plt.suptitle('2D Gaussian Splatting Simulation', fontsize=14)
plt.tight_layout()
plt.savefig('gaussian_splatting_2d.png', dpi=150, bbox_inches='tight')
plt.show()
```

The key takeaway: 3D Gaussian Splatting replaces the implicit, slow, ray-marching approach of NeRF with an explicit, fast, rasterization-based approach. The Gaussians are a flexible, differentiable representation that can be optimized to match real photographs, rendered in real time, and extended to dynamic scenes. For video generation, this opens the door to explicit 3D control, physically grounded camera motion, and world models that understand geometry --- not just pixel patterns.
