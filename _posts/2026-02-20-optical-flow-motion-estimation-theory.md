---
layout: post
title: "Optical Flow and Motion Estimation: The Mathematics of Movement in Video Generation"
date: 2026-02-20
category: math
---

Video is not a stack of independent images. It is a record of motion --- objects translating, rotating, deforming, appearing, and disappearing across time. Every video generation model, whether it knows it explicitly or not, must learn to produce coherent motion. The mathematical framework that formalizes motion between frames is **optical flow**: a dense vector field that assigns a velocity to every pixel, telling you where that pixel "went" in the next frame.

Optical flow predates deep learning by decades. It was formulated in the 1980s by Horn and Schunck, refined by Lucas and Kanade, and has been a cornerstone of computer vision ever since. But its relevance to modern video generation is deeper than historical. Understanding optical flow gives you the language to talk about temporal consistency, motion compensation, frame interpolation, and the implicit motion representations that diffusion-based video models learn internally.

This post builds the theory of optical flow from scratch. We start with the brightness constancy assumption, derive the fundamental constraint equation, explain why it is underdetermined (the aperture problem), and then develop two classical solutions: the local Lucas-Kanade method and the global Horn-Schunck variational method. We extend to deep optical flow (RAFT), connect flow to video generation via warping and motion-conditioned synthesis, and implement both classical methods from scratch in Python.

---

## Table of Contents

1. [What Is Optical Flow?](#what-is-optical-flow)
2. [The Brightness Constancy Assumption](#the-brightness-constancy-assumption)
3. [The Aperture Problem](#the-aperture-problem)
4. [The Lucas-Kanade Method](#the-lucas-kanade-method)
5. [The Horn-Schunck Method](#the-horn-schunck-method)
6. [Multi-Scale Coarse-to-Fine Estimation](#multi-scale-coarse-to-fine-estimation)
7. [The General Variational Framework](#the-general-variational-framework)
8. [Deep Optical Flow: FlowNet to RAFT](#deep-optical-flow-flownet-to-raft)
9. [Warping and Flow-Based Frame Synthesis](#warping-and-flow-based-frame-synthesis)
10. [Optical Flow in Video Diffusion Models](#optical-flow-in-video-diffusion-models)
11. [Python: Lucas-Kanade and Horn-Schunck from Scratch](#python-lucas-kanade-and-horn-schunck-from-scratch)

---

## What Is Optical Flow?

Consider two consecutive video frames, \(I(x, y, t)\) and \(I(x, y, t + \Delta t)\), where \(I\) is the image intensity (brightness) at spatial position \((x, y)\) and time \(t\). Some pixels in the first frame have "moved" to new positions in the second frame --- a person walks, a car drives, the camera pans.

**Optical flow** is a 2D vector field \(\mathbf{u}(x, y) = (u(x, y), v(x, y))\) that assigns to every pixel \((x, y)\) in the first frame a displacement vector telling you where that pixel moved to in the second frame. The component \(u\) is the horizontal displacement and \(v\) is the vertical displacement.

More precisely, if a point at position \((x, y)\) in frame \(t\) corresponds to position \((x + u, y + v)\) in frame \(t + \Delta t\), then \((u, v)\) is the optical flow at that point.

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <defs>
    <marker id="arrowFlow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#4fc3f7"/>
    </marker>
  </defs>
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">Optical Flow Between Two Frames</text>

  <!-- Frame 1 -->
  <rect x="30" y="50" width="200" height="180" rx="4" fill="none" stroke="#666" stroke-width="2"/>
  <text x="130" y="250" text-anchor="middle" font-size="12" fill="#999">Frame t</text>
  <circle cx="100" cy="130" r="20" fill="#E53935" opacity="0.7"/>
  <circle cx="170" cy="170" r="12" fill="#66bb6a" opacity="0.7"/>
  <circle cx="80" cy="80" r="8" fill="#FF9800" opacity="0.7"/>

  <!-- Frame 2 -->
  <rect x="470" y="50" width="200" height="180" rx="4" fill="none" stroke="#666" stroke-width="2"/>
  <text x="570" y="250" text-anchor="middle" font-size="12" fill="#999">Frame t + Δt</text>
  <circle cx="560" cy="120" r="20" fill="#E53935" opacity="0.7"/>
  <circle cx="610" cy="160" r="12" fill="#66bb6a" opacity="0.7"/>
  <circle cx="510" cy="90" r="8" fill="#FF9800" opacity="0.7"/>

  <!-- Flow arrows -->
  <line x1="120" y1="130" x2="538" y2="120" stroke="#4fc3f7" stroke-width="2" marker-end="url(#arrowFlow)" stroke-dasharray="6,3"/>
  <line x1="182" y1="170" x2="596" y2="160" stroke="#4fc3f7" stroke-width="2" marker-end="url(#arrowFlow)" stroke-dasharray="6,3"/>
  <line x1="88" y1="80" x2="500" y2="90" stroke="#4fc3f7" stroke-width="2" marker-end="url(#arrowFlow)" stroke-dasharray="6,3"/>

  <text x="350" y="110" text-anchor="middle" font-size="11" fill="#4fc3f7">u(x,y) = displacement vectors</text>
</svg>

There is a subtle but important distinction: optical flow is not the same as the **true 3D motion field** projected onto the image plane. Consider a perfectly uniform sphere rotating in place --- the 3D motion is nonzero everywhere, but the optical flow is zero because no brightness pattern changes. Conversely, a fixed scene with a moving light source produces apparent brightness changes (and thus nonzero optical flow) even though nothing is physically moving. Optical flow measures **apparent motion** --- the motion of brightness patterns --- not physical motion. In practice, for typical video content, the two are very close.

---

## The Brightness Constancy Assumption

The foundation of all classical optical flow methods is a single assumption: **a pixel's brightness does not change as it moves**. If a pixel at \((x, y)\) in frame \(t\) has intensity \(I(x, y, t)\), and it moves to \((x + u, y + v)\) in frame \(t + \Delta t\), then:

$$I(x + u, y + v, t + \Delta t) = I(x, y, t)$$

This is the **brightness constancy equation**. It says that the intensity is conserved along the motion trajectory. This is reasonable when illumination is constant and surfaces are Lambertian (diffuse reflectors), but it breaks down under shadows, specular reflections, transparency, and occlusion.

Now we linearize. Taylor-expand the left side around \((x, y, t)\):

$$I(x + u, y + v, t + \Delta t) \approx I(x, y, t) + \frac{\partial I}{\partial x} u + \frac{\partial I}{\partial y} v + \frac{\partial I}{\partial t} \Delta t$$

Setting this equal to \(I(x, y, t)\) and dividing by \(\Delta t\):

$$\frac{\partial I}{\partial x} \frac{u}{\Delta t} + \frac{\partial I}{\partial y} \frac{v}{\Delta t} + \frac{\partial I}{\partial t} = 0$$

Define the flow velocities \(u' = u / \Delta t\) and \(v' = v / \Delta t\) (but by convention we drop the primes and just call them \(u, v\), understanding they are velocities per frame). Introduce the shorthand \(I_x = \partial I / \partial x\), \(I_y = \partial I / \partial y\), \(I_t = \partial I / \partial t\). The result is the **optical flow constraint equation**:

$$I_x u + I_y v + I_t = 0$$

Or in vector form, with \(\nabla I = (I_x, I_y)\) being the spatial image gradient:

$$\nabla I \cdot \mathbf{u} + I_t = 0$$

This is a single linear equation in two unknowns \((u, v)\). One equation, two unknowns. The system is **underdetermined** --- there are infinitely many flow vectors \((u, v)\) consistent with the constraint at any single pixel. This is the **aperture problem**.

---

## The Aperture Problem

The optical flow constraint equation tells us the component of the flow in the direction of the image gradient \(\nabla I\), but tells us nothing about the component perpendicular to it.

To see why, rewrite the constraint:

$$\nabla I \cdot \mathbf{u} = -I_t$$

The left side is the dot product of the flow with the gradient direction. This means we can only recover the **normal flow** --- the component of \(\mathbf{u}\) in the direction of \(\nabla I\):

$$u_n = \frac{-I_t}{|\nabla I|}$$

The component of \(\mathbf{u}\) tangent to the image edge (perpendicular to \(\nabla I\)) is completely unconstrained.

<svg viewBox="0 0 700 300" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <defs>
    <marker id="arrowAp1" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#E53935"/>
    </marker>
    <marker id="arrowAp2" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#4fc3f7"/>
    </marker>
    <marker id="arrowAp3" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#66bb6a"/>
    </marker>
  </defs>
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">The Aperture Problem</text>

  <!-- Aperture circle -->
  <circle cx="200" cy="170" r="80" fill="none" stroke="#666" stroke-width="2" stroke-dasharray="6,3"/>
  <text x="200" y="270" text-anchor="middle" font-size="11" fill="#999">Local aperture view</text>

  <!-- Edge line through aperture -->
  <line x1="130" y1="210" x2="270" y2="130" stroke="#d4d4d4" stroke-width="3"/>

  <!-- Gradient direction (perpendicular to edge) -->
  <line x1="200" y1="170" x2="240" y2="210" stroke="#E53935" stroke-width="2" marker-end="url(#arrowAp1)"/>
  <text x="260" y="220" font-size="10" fill="#E53935">∇I (gradient)</text>

  <!-- Normal flow (along gradient) -->
  <line x1="200" y1="170" x2="220" y2="190" stroke="#4fc3f7" stroke-width="2.5" marker-end="url(#arrowAp2)"/>
  <text x="235" y="195" font-size="10" fill="#4fc3f7">u_n (known)</text>

  <!-- Tangent direction (unknown) -->
  <line x1="200" y1="170" x2="240" y2="140" stroke="#66bb6a" stroke-width="2" marker-end="url(#arrowAp3)" stroke-dasharray="4,3"/>
  <text x="250" y="135" font-size="10" fill="#66bb6a">tangent (unknown)</text>

  <!-- Right side: explanation -->
  <text x="480" y="100" text-anchor="middle" font-size="12" fill="#d4d4d4">Through a small aperture,</text>
  <text x="480" y="120" text-anchor="middle" font-size="12" fill="#d4d4d4">an edge appears to move</text>
  <text x="480" y="140" text-anchor="middle" font-size="12" fill="#d4d4d4">only perpendicular to itself.</text>
  <text x="480" y="175" text-anchor="middle" font-size="11" fill="#999">The motion along the edge</text>
  <text x="480" y="195" text-anchor="middle" font-size="11" fill="#999">is invisible locally.</text>
  <text x="480" y="230" text-anchor="middle" font-size="11" fill="#FF9800">One equation, two unknowns.</text>
  <text x="480" y="250" text-anchor="middle" font-size="11" fill="#FF9800">Need additional constraints!</text>
</svg>

Think about looking at a moving barber pole through a small window. The stripes appear to move upward, but the pole is actually rotating horizontally. Through the small aperture, you can only see the component of motion perpendicular to the stripes. The motion along the stripes is invisible.

To recover the full flow, we need additional constraints. The two classical approaches differ in what constraint they add: **Lucas-Kanade** assumes flow is locally constant (spatial smoothness within a patch), while **Horn-Schunck** imposes a global smoothness penalty on the flow field.

---

## The Lucas-Kanade Method

The Lucas-Kanade (LK) method (1981) resolves the aperture problem by assuming that the flow \((u, v)\) is constant within a small spatial neighborhood \(\Omega\) of each pixel. If the flow is the same for all pixels in a patch, we get one equation per pixel but only two unknowns, giving us an overdetermined system.

For a window of \(N\) pixels, the brightness constancy equation at each pixel \((x_i, y_i)\) gives:

$$I_x(x_i, y_i) \, u + I_y(x_i, y_i) \, v = -I_t(x_i, y_i) \quad \text{for } i = 1, \ldots, N$$

In matrix form:

$$A \mathbf{u} = \mathbf{b}$$

where:

$$A = \begin{pmatrix} I_x(x_1, y_1) & I_y(x_1, y_1) \\ I_x(x_2, y_2) & I_y(x_2, y_2) \\ \vdots & \vdots \\ I_x(x_N, y_N) & I_y(x_N, y_N) \end{pmatrix}, \quad \mathbf{u} = \begin{pmatrix} u \\ v \end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix} -I_t(x_1, y_1) \\ -I_t(x_2, y_2) \\ \vdots \\ -I_t(x_N, y_N) \end{pmatrix}$$

This is an overdetermined system (\(N \gg 2\)), so we solve it in the least-squares sense by solving the **normal equations**:

$$A^T A \, \mathbf{u} = A^T \mathbf{b}$$

The matrix \(A^T A\) is the **structure tensor** (also called the second moment matrix):

$$M = A^T A = \begin{pmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{pmatrix}$$

where all sums run over the window \(\Omega\). The solution is:

$$\mathbf{u} = M^{-1} A^T \mathbf{b} = \begin{pmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{pmatrix}^{-1} \begin{pmatrix} -\sum I_x I_t \\ -\sum I_y I_t \end{pmatrix}$$

### When Does This Work?

The solution exists and is well-conditioned when \(M\) is invertible, which requires both eigenvalues \(\lambda_1, \lambda_2\) to be large. The eigenvalue structure of \(M\) reveals the local image geometry:

- **Both eigenvalues small** (\(\lambda_1 \approx \lambda_2 \approx 0\)): Flat region --- no gradient, no flow information. The system is degenerate.
- **One eigenvalue large, one small** (\(\lambda_1 \gg \lambda_2 \approx 0\)): An edge --- the gradient has one dominant direction. We can recover normal flow but not tangential flow. This is the aperture problem again.
- **Both eigenvalues large** (\(\lambda_1, \lambda_2 \gg 0\)): A corner or textured region --- gradients in multiple directions. The system is well-determined and the flow estimate is reliable.

This eigenvalue analysis is precisely the **Harris corner detector**. Corners are good features for flow estimation; edges and flat regions are not. The quantity \(\min(\lambda_1, \lambda_2)\) (the Shi-Tomasi criterion) tells you how reliable the LK flow estimate is at each point.

In practice, the window is often weighted by a Gaussian to give more influence to the center pixel:

$$M = \sum_{(x_i, y_i) \in \Omega} w_i \begin{pmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{pmatrix}_{(x_i, y_i)}$$

where \(w_i = \exp\left(-\frac{(x_i - x_0)^2 + (y_i - y_0)^2}{2\sigma^2}\right)\).

---

## The Horn-Schunck Method

While Lucas-Kanade makes a local assumption (constant flow in a patch), the Horn-Schunck method (1981) takes a **global** approach: find the flow field that satisfies the brightness constancy equation as well as possible while being as smooth as possible everywhere.

This is formulated as a **variational problem**. Define the energy functional:

$$E(u, v) = \iint \left[ \underbrace{(I_x u + I_y v + I_t)^2}_{\text{data term}} + \alpha^2 \underbrace{\left(|\nabla u|^2 + |\nabla v|^2\right)}_{\text{smoothness term}} \right] dx \, dy$$

The **data term** penalizes violations of brightness constancy. The **smoothness term** penalizes spatial variation in the flow field, encouraging neighboring pixels to have similar flow vectors. The parameter \(\alpha^2 > 0\) controls the tradeoff: larger \(\alpha\) produces smoother flow (at the cost of accuracy at motion boundaries).

The gradient terms are:

$$|\nabla u|^2 = u_x^2 + u_y^2, \quad |\nabla v|^2 = v_x^2 + v_y^2$$

where subscripts denote partial derivatives of the flow components.

### Euler-Lagrange Equations

To minimize this functional, we apply the calculus of variations. The integrand is of the form \(\mathcal{L}(u, v, u_x, u_y, v_x, v_y)\), and the Euler-Lagrange equations for \(u\) and \(v\) are:

$$\frac{\partial \mathcal{L}}{\partial u} - \frac{\partial}{\partial x}\frac{\partial \mathcal{L}}{\partial u_x} - \frac{\partial}{\partial y}\frac{\partial \mathcal{L}}{\partial u_y} = 0$$

$$\frac{\partial \mathcal{L}}{\partial v} - \frac{\partial}{\partial x}\frac{\partial \mathcal{L}}{\partial v_x} - \frac{\partial}{\partial y}\frac{\partial \mathcal{L}}{\partial v_y} = 0$$

Let us compute each term. The integrand is:

$$\mathcal{L} = (I_x u + I_y v + I_t)^2 + \alpha^2(u_x^2 + u_y^2 + v_x^2 + v_y^2)$$

For the \(u\) equation:

$$\frac{\partial \mathcal{L}}{\partial u} = 2 I_x (I_x u + I_y v + I_t)$$

$$\frac{\partial \mathcal{L}}{\partial u_x} = 2\alpha^2 u_x \implies \frac{\partial}{\partial x}\frac{\partial \mathcal{L}}{\partial u_x} = 2\alpha^2 u_{xx}$$

$$\frac{\partial \mathcal{L}}{\partial u_y} = 2\alpha^2 u_y \implies \frac{\partial}{\partial y}\frac{\partial \mathcal{L}}{\partial u_y} = 2\alpha^2 u_{yy}$$

The Euler-Lagrange equation for \(u\) becomes:

$$I_x(I_x u + I_y v + I_t) - \alpha^2 \nabla^2 u = 0$$

where \(\nabla^2 u = u_{xx} + u_{yy}\) is the Laplacian. Similarly for \(v\):

$$I_y(I_x u + I_y v + I_t) - \alpha^2 \nabla^2 v = 0$$

These are two coupled partial differential equations for the flow field \((u, v)\). They can be solved iteratively using a Gauss-Seidel or Jacobi scheme. A standard approach approximates the Laplacian using the difference between the local average and the center value:

$$\nabla^2 u \approx \bar{u} - u$$

where \(\bar{u}\) is the average of \(u\) over the 4-neighborhood (or 8-neighborhood) of the pixel. Substituting and solving for \(u\) and \(v\):

$$u^{k+1} = \bar{u}^k - \frac{I_x(I_x \bar{u}^k + I_y \bar{v}^k + I_t)}{\alpha^2 + I_x^2 + I_y^2}$$

$$v^{k+1} = \bar{v}^k - \frac{I_y(I_x \bar{u}^k + I_y \bar{v}^k + I_t)}{\alpha^2 + I_x^2 + I_y^2}$$

These are iterated until convergence. Typically 100--500 iterations suffice.

<svg viewBox="0 0 700 250" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">Lucas-Kanade (Local) vs Horn-Schunck (Global)</text>

  <!-- LK side -->
  <rect x="30" y="50" width="280" height="170" rx="4" fill="none" stroke="#4fc3f7" stroke-width="1.5"/>
  <text x="170" y="75" text-anchor="middle" font-size="12" font-weight="bold" fill="#4fc3f7">Lucas-Kanade</text>
  <text x="170" y="100" text-anchor="middle" font-size="10" fill="#999">Solve independently per patch</text>
  <!-- Small patches -->
  <rect x="60" y="115" width="40" height="40" fill="#4fc3f7" opacity="0.15" stroke="#4fc3f7" stroke-width="1"/>
  <rect x="120" y="115" width="40" height="40" fill="#4fc3f7" opacity="0.15" stroke="#4fc3f7" stroke-width="1"/>
  <rect x="180" y="115" width="40" height="40" fill="#4fc3f7" opacity="0.15" stroke="#4fc3f7" stroke-width="1"/>
  <rect x="240" y="115" width="40" height="40" fill="#4fc3f7" opacity="0.15" stroke="#4fc3f7" stroke-width="1"/>
  <rect x="60" y="165" width="40" height="40" fill="#4fc3f7" opacity="0.15" stroke="#4fc3f7" stroke-width="1"/>
  <rect x="120" y="165" width="40" height="40" fill="#4fc3f7" opacity="0.15" stroke="#4fc3f7" stroke-width="1"/>
  <rect x="180" y="165" width="40" height="40" fill="#4fc3f7" opacity="0.15" stroke="#4fc3f7" stroke-width="1"/>
  <rect x="240" y="165" width="40" height="40" fill="#4fc3f7" opacity="0.15" stroke="#4fc3f7" stroke-width="1"/>
  <text x="170" y="230" text-anchor="middle" font-size="10" fill="#999">Sparse, accurate at corners</text>

  <!-- HS side -->
  <rect x="390" y="50" width="280" height="170" rx="4" fill="none" stroke="#66bb6a" stroke-width="1.5"/>
  <text x="530" y="75" text-anchor="middle" font-size="12" font-weight="bold" fill="#66bb6a">Horn-Schunck</text>
  <text x="530" y="100" text-anchor="middle" font-size="10" fill="#999">Global smoothness regularization</text>
  <!-- Smooth field -->
  <rect x="400" y="110" width="260" height="100" fill="#66bb6a" opacity="0.08"/>
  <line x1="420" y1="140" x2="445" y2="135" stroke="#66bb6a" stroke-width="1.5"/>
  <line x1="460" y1="140" x2="485" y2="134" stroke="#66bb6a" stroke-width="1.5"/>
  <line x1="500" y1="138" x2="525" y2="132" stroke="#66bb6a" stroke-width="1.5"/>
  <line x1="540" y1="136" x2="565" y2="130" stroke="#66bb6a" stroke-width="1.5"/>
  <line x1="580" y1="134" x2="605" y2="128" stroke="#66bb6a" stroke-width="1.5"/>
  <line x1="420" y1="175" x2="445" y2="170" stroke="#66bb6a" stroke-width="1.5"/>
  <line x1="460" y1="175" x2="485" y2="169" stroke="#66bb6a" stroke-width="1.5"/>
  <line x1="500" y1="173" x2="525" y2="167" stroke="#66bb6a" stroke-width="1.5"/>
  <line x1="540" y1="171" x2="565" y2="165" stroke="#66bb6a" stroke-width="1.5"/>
  <line x1="580" y1="169" x2="605" y2="163" stroke="#66bb6a" stroke-width="1.5"/>
  <text x="530" y="230" text-anchor="middle" font-size="10" fill="#999">Dense, smooth, blurs boundaries</text>
</svg>

---

## Multi-Scale Coarse-to-Fine Estimation

Both classical methods assume **small displacements** --- the Taylor expansion is only valid when the pixel motion between frames is small (a few pixels). For large motions (fast-moving objects, low frame rates), the linearization breaks down.

The solution is a **coarse-to-fine** approach using an **image pyramid**:

1. Build a Gaussian pyramid for both frames: downsample each image by factor 2 repeatedly, creating \(L\) levels. At the coarsest level, images are tiny (e.g., 16×12 pixels).

2. At the coarsest level, displacements are small (because the image is small), so the linearization is valid. Estimate the flow at this level using LK or HS.

3. Upsample the coarse flow estimate to the next finer level (multiply by 2 and resize). **Warp** the first frame toward the second using this preliminary flow estimate.

4. Compute the **residual flow** between the warped first frame and the actual second frame. The residual displacement is now small (the coarse flow already captured the bulk of the motion).

5. Add the residual flow to the upsampled coarse flow. Repeat down to the finest level.

This hierarchical approach can handle displacements of hundreds of pixels. It is standard in all classical flow methods and remains the conceptual backbone even for deep methods (which process features at multiple scales via encoder-decoder architectures).

The warping step at level \(l\) is:

$$I_1^{\text{warped}}(x, y) = I_1(x + u^{l+1}(x,y), \, y + v^{l+1}(x,y))$$

where \((u^{l+1}, v^{l+1})\) is the flow estimate from the coarser level, upsampled to the current resolution.

---

## The General Variational Framework

Horn-Schunck is the simplest variational optical flow method, but it has a well-known weakness: the quadratic smoothness term \(|\nabla u|^2\) penalizes flow discontinuities everywhere, including at **motion boundaries** (where the flow genuinely changes sharply --- the edge between a moving foreground and a stationary background).

The modern variational framework generalizes the energy to:

$$E(u, v) = \iint \left[ \Psi_D\!\left((I_x u + I_y v + I_t)^2\right) + \alpha \, \Psi_S\!\left(|\nabla u|^2 + |\nabla v|^2\right) \right] dx \, dy$$

where \(\Psi_D\) and \(\Psi_S\) are **penalty functions** (also called robust estimators) that control how violations of the data constraint and smoothness constraint are penalized.

**Horn-Schunck** uses \(\Psi(s) = s\) (quadratic penalty). The problem: outliers and motion boundaries get penalized quadratically, causing over-smoothing.

**Robust alternatives:**

- **Charbonnier penalty:** \(\Psi(s) = \sqrt{s + \epsilon^2}\) (a smooth approximation to \(L^1\)). This penalizes large deviations less than quadratic, allowing the flow to have sharp boundaries.

- **Lorentzian:** \(\Psi(s) = \log(1 + s / (2\sigma^2))\). Even more robust to outliers.

- **Total Variation (TV):** \(\Psi_S(|\nabla u|^2) = |\nabla u|\). This promotes piecewise-constant flow, which is appropriate when objects move rigidly.

The TV-L1 optical flow method (Zach, Pock, Bischof, 2007) uses an \(L^1\) data term with TV regularization:

$$E = \iint \left[ |I_x u + I_y v + I_t| + \alpha (|\nabla u| + |\nabla v|) \right] dx \, dy$$

This is convex and can be solved efficiently using primal-dual algorithms. It is robust to outliers in the data term (occlusions, illumination changes) and produces piecewise-smooth flow with sharp motion boundaries.

---

## Deep Optical Flow: FlowNet to RAFT

Classical methods optimize an energy functional per frame pair. Deep optical flow methods learn to estimate flow directly from image pairs using neural networks, trained on large synthetic datasets (FlyingChairs, FlyingThings3D, Sintel) where ground-truth flow is available.

### FlowNet (2015)

The first end-to-end deep flow method. Two variants:

- **FlowNetS** ("simple"): Concatenate two frames along the channel dimension, pass through a CNN encoder-decoder. The network implicitly computes correspondences.
- **FlowNetC** ("correlation"): Compute features separately for each image, then compute a **correlation volume** (the dot product of feature vectors at all spatial positions), which explicitly encodes matching costs.

The correlation volume at position \((x, y)\) for displacement \((d_x, d_y)\) is:

$$C(x, y, d_x, d_y) = \sum_c f_1^c(x, y) \cdot f_2^c(x + d_x, y + d_y)$$

where \(f_1, f_2\) are feature maps and \(c\) indexes channels.

### RAFT (2020)

Recurrent All-Pairs Field Transforms. The state-of-the-art architecture before the transformer era. Key ideas:

1. **All-pairs correlation volume.** Compute the full 4D correlation volume between all pairs of feature vectors (not just local displacements). This is \(H \times W \times H \times W\), which is expensive but comprehensive.

2. **Correlation pyramid.** Pool the correlation volume at multiple scales to capture both small and large displacements.

3. **Iterative refinement.** Start with an initial flow estimate (e.g., zero). At each iteration, look up the correlation volume at the current flow-warped positions, concatenate with context features, and pass through a **GRU** (gated recurrent unit) to produce a flow update \(\Delta \mathbf{u}\).

4. **Repeat for \(K\) iterations** (typically 12--32), progressively refining the flow.

This iterative approach is analogous to the iterative optimization in variational methods, but the update rule is learned rather than derived from an energy functional. RAFT achieves dramatically better accuracy than classical methods, especially at motion boundaries and in occluded regions.

---

## Warping and Flow-Based Frame Synthesis

Once you have an optical flow field, you can use it to **warp** one frame toward another. This is fundamental for:

- **Frame interpolation** (generating intermediate frames)
- **Video prediction** (synthesizing future frames)
- **Temporal consistency losses** in video generation

### Backward Warping

Given a flow field \(\mathbf{u} = (u, v)\) from frame \(I_0\) to frame \(I_1\), the **backward warp** produces a reconstructed frame:

$$\hat{I}_1(x, y) = I_0(x + u(x,y), \, y + v(x,y))$$

Since \((x + u, y + v)\) generally falls between pixel grid points, we use **bilinear interpolation**:

$$I_0(x', y') = (1-\alpha)(1-\beta) \, I_0(\lfloor x' \rfloor, \lfloor y' \rfloor) + \alpha(1-\beta) \, I_0(\lceil x' \rceil, \lfloor y' \rfloor) + (1-\alpha)\beta \, I_0(\lfloor x' \rfloor, \lceil y' \rceil) + \alpha \beta \, I_0(\lceil x' \rceil, \lceil y' \rceil)$$

where \(\alpha = x' - \lfloor x' \rfloor\) and \(\beta = y' - \lfloor y' \rfloor\).

Backward warping is **differentiable** with respect to both the flow and the source image (via the spatial transformer network formulation of Jaderberg et al. 2015), which is why it can be used as a layer in trainable networks.

### Occlusion Handling

Warping breaks at **occlusions** --- regions visible in one frame but not the other. When an object moves, it reveals previously hidden background. No amount of warping can synthesize content that was never visible.

Occlusion maps can be estimated from flow consistency: compute forward flow \(\mathbf{u}_{0 \to 1}\) and backward flow \(\mathbf{u}_{1 \to 0}\), and check if they are consistent:

$$\text{occluded}(x, y) = |\mathbf{u}_{0 \to 1}(x, y) + \mathbf{u}_{1 \to 0}(x + u, y + v)| > \tau$$

If the forward and backward flows don't cancel, the point is likely occluded.

### Frame Interpolation

To interpolate a frame at time \(t \in (0, 1)\) between frames \(I_0\) (at \(t=0\)) and \(I_1\) (at \(t=1\)):

1. Estimate flows \(\mathbf{u}_{0 \to 1}\) and \(\mathbf{u}_{1 \to 0}\).
2. Approximate intermediate flows: \(\mathbf{u}_{0 \to t} \approx t \cdot \mathbf{u}_{0 \to 1}\), \(\mathbf{u}_{1 \to t} \approx (1-t) \cdot \mathbf{u}_{1 \to 0}\).
3. Warp both frames to time \(t\): \(\hat{I}_t^0 = \text{warp}(I_0, \mathbf{u}_{0 \to t})\), \(\hat{I}_t^1 = \text{warp}(I_1, \mathbf{u}_{1 \to t})\).
4. Blend (with occlusion-aware weighting): \(\hat{I}_t = (1-t) \hat{I}_t^0 + t \hat{I}_t^1\).

This is the basis of methods like FILM and RIFE for video frame interpolation.

---

## Optical Flow in Video Diffusion Models

Modern video generation models (Stable Video Diffusion, Sora, Kling, Runway Gen-3) do not explicitly compute optical flow. They operate in latent space and generate all frames jointly. But the concept of optical flow is deeply embedded in how these models work:

**Implicit motion learning.** The temporal attention layers in video DiT models learn to correlate features across frames. The attention weights implicitly encode correspondences --- which token in frame \(t\) corresponds to which token in frame \(t+1\). This is a soft, learned version of optical flow operating in feature space.

**Flow-conditioned generation.** Some architectures accept explicit flow maps as conditioning. You provide the model with a desired motion field, and it generates video that follows that motion. This allows user control over camera movement and object trajectories.

**Motion guidance (AnimateDiff, MotionCtrl).** These methods add **temporal LoRA layers** or **motion modules** that inject motion priors into a pretrained image diffusion model. The motion module learns temporal correlations from video data, effectively encoding a distribution over optical flow fields.

**Temporal consistency losses.** During training or fine-tuning, some methods use flow-based warping losses:

$$\mathcal{L}_{\text{temporal}} = \sum_t \| I_{t+1} - \text{warp}(I_t, \mathbf{u}_{t \to t+1}) \|_1 \cdot (1 - O_t)$$

where \(O_t\) is an occlusion mask. This loss penalizes temporal inconsistency in non-occluded regions.

**Flow in latent space.** An intriguing recent direction: estimate optical flow not in pixel space but in the latent space of the video autoencoder. Since the latent space is 8× downsampled spatially and may be downsampled temporally, "latent flow" captures motion at a semantic level rather than pixel level.

---

## Python: Lucas-Kanade and Horn-Schunck from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, uniform_filter

def create_synthetic_frames(size=128):
    """Create two frames with a moving Gaussian blob."""
    y, x = np.mgrid[0:size, 0:size].astype(float)

    # Frame 1: Gaussian blob centered at (40, 50)
    cx1, cy1 = 50.0, 40.0
    I1 = np.exp(-((x - cx1)**2 + (y - cy1)**2) / (2 * 15**2))

    # Frame 2: Blob moved to (47, 53) — displacement (u=3, v=7)
    cx2, cy2 = 53.0, 47.0
    I2 = np.exp(-((x - cx2)**2 + (y - cy2)**2) / (2 * 15**2))

    return I1, I2, 3.0, 7.0  # frames and true flow

def compute_derivatives(I1, I2, sigma=1.0):
    """Compute spatial and temporal image derivatives."""
    # Smooth first for stable derivatives
    I1s = gaussian_filter(I1, sigma)
    I2s = gaussian_filter(I2, sigma)
    # Temporal derivative
    It = I2s - I1s
    # Spatial derivatives (average of both frames for accuracy)
    Iavg = 0.5 * (I1s + I2s)
    Ix = np.gradient(Iavg, axis=1)
    Iy = np.gradient(Iavg, axis=0)
    return Ix, Iy, It

def lucas_kanade(Ix, Iy, It, window=15):
    """Lucas-Kanade optical flow with local window."""
    half_w = window // 2
    h, w = Ix.shape
    u = np.zeros_like(Ix)
    v = np.zeros_like(Ix)

    # Use uniform filter to compute windowed sums efficiently
    Ix2 = uniform_filter(Ix * Ix, size=window)
    Iy2 = uniform_filter(Iy * Iy, size=window)
    IxIy = uniform_filter(Ix * Iy, size=window)
    IxIt = uniform_filter(Ix * It, size=window)
    IyIt = uniform_filter(Iy * It, size=window)

    # Determinant of structure tensor
    det = Ix2 * Iy2 - IxIy**2
    # Threshold to avoid division by zero in flat regions
    valid = np.abs(det) > 1e-6

    u[valid] = (IxIy[valid] * IyIt[valid] - Iy2[valid] * IxIt[valid]) / det[valid]
    v[valid] = (IxIy[valid] * IxIt[valid] - Ix2[valid] * IyIt[valid]) / det[valid]
    return u, v

def horn_schunck(Ix, Iy, It, alpha=1.0, num_iter=200):
    """Horn-Schunck optical flow with global smoothness."""
    u = np.zeros_like(Ix)
    v = np.zeros_like(Ix)

    # Precompute denominators
    denom = alpha**2 + Ix**2 + Iy**2

    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]]) / 4.0

    for _ in range(num_iter):
        # Compute local averages
        u_avg = uniform_filter(u, size=3)  # approximate Laplacian neighborhood
        v_avg = uniform_filter(v, size=3)

        # Update equations
        P = Ix * u_avg + Iy * v_avg + It
        u = u_avg - Ix * P / denom
        v = v_avg - Iy * P / denom

    return u, v

# Generate synthetic data
I1, I2, true_u, true_v = create_synthetic_frames()
Ix, Iy, It = compute_derivatives(I1, I2)

# Compute flow with both methods
u_lk, v_lk = lucas_kanade(Ix, Iy, It, window=21)
u_hs, v_hs = horn_schunck(Ix, Iy, It, alpha=1.0, num_iter=300)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# Top row: frames and derivatives
axes[0, 0].imshow(I1, cmap='gray')
axes[0, 0].set_title(r'Frame $I_1$')
axes[0, 0].axis('off')

axes[0, 1].imshow(I2, cmap='gray')
axes[0, 1].set_title(r'Frame $I_2$')
axes[0, 1].axis('off')

mag = np.sqrt(Ix**2 + Iy**2)
axes[0, 2].imshow(mag, cmap='hot')
axes[0, 2].set_title(r'$|\nabla I|$ (gradient magnitude)')
axes[0, 2].axis('off')

# Bottom row: flow fields
step = 6
Y, X = np.mgrid[0:128:step, 0:128:step]

axes[1, 0].imshow(I1, cmap='gray', alpha=0.5)
axes[1, 0].quiver(X, Y, u_lk[::step, ::step], v_lk[::step, ::step],
                  color='cyan', scale=100, width=0.004)
axes[1, 0].set_title('Lucas-Kanade Flow')
axes[1, 0].axis('off')

axes[1, 1].imshow(I1, cmap='gray', alpha=0.5)
axes[1, 1].quiver(X, Y, u_hs[::step, ::step], v_hs[::step, ::step],
                  color='lime', scale=100, width=0.004)
axes[1, 1].set_title('Horn-Schunck Flow')
axes[1, 1].axis('off')

# Flow magnitude comparison
flow_mag_lk = np.sqrt(u_lk**2 + v_lk**2)
flow_mag_hs = np.sqrt(u_hs**2 + v_hs**2)
axes[1, 2].plot(flow_mag_lk[64, :], label='LK', color='cyan')
axes[1, 2].plot(flow_mag_hs[64, :], label='HS', color='lime')
axes[1, 2].axhline(y=np.sqrt(true_u**2 + true_v**2), color='red',
                   linestyle='--', label=r'True $|\mathbf{u}|$')
axes[1, 2].set_xlabel(r'$x$ position')
axes[1, 2].set_ylabel(r'Flow magnitude $|\mathbf{u}|$')
axes[1, 2].set_title('Flow Magnitude (row 64)')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('Optical Flow: Lucas-Kanade vs Horn-Schunck', fontsize=14)
plt.tight_layout()
plt.savefig('optical_flow_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

This implementation highlights the fundamental difference: Lucas-Kanade produces sparse, accurate flow at well-textured points (where the structure tensor has two large eigenvalues), while Horn-Schunck produces dense, globally smooth flow that may blur motion boundaries. Modern deep methods like RAFT combine the best of both worlds --- dense estimation with sharp boundaries --- by learning the correspondence and regularization jointly from data.

The mathematical framework of optical flow remains central to video understanding and generation. Whether a model estimates flow explicitly or learns it implicitly through attention, the core question is the same: where did each pixel go?
