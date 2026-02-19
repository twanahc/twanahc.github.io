---
layout: post
title: "Neural Radiance Fields and Volumetric Rendering: From Ray Marching to Dynamic 4D Scenes"
date: 2026-02-22
category: math
---

A photograph is a 2D projection of a 3D world. To generate video with controllable cameras, we need to reconstruct that 3D world from 2D observations and render it from arbitrary viewpoints. **Neural Radiance Fields (NeRF)** accomplish this by representing a scene as a continuous volumetric function --- mapping every point in 3D space to a density and color --- and rendering images by tracing rays through this volume using classical radiative transfer theory.

NeRF's impact on video generation is profound. It demonstrated that neural networks can learn complete 3D scene representations from photographs alone, enabling photorealistic novel view synthesis. Its descendants power camera control in video generation models, world models that understand 3D geometry, and 4D dynamic scene reconstruction from monocular video.

This post derives the full pipeline from first principles. We start with the radiative transfer equation from physics, derive the volume rendering integral, discretize it into the alpha compositing formula used in practice, explain positional encoding and why it is necessary, derive hierarchical importance sampling, and extend to dynamic scenes. We then connect to video generation.

---

## Table of Contents

1. [The Volume Rendering Equation](#the-volume-rendering-equation)
2. [Discretizing the Integral: Alpha Compositing](#discretizing-the-integral-alpha-compositing)
3. [Neural Radiance Fields: The MLP](#neural-radiance-fields-the-mlp)
4. [Positional Encoding and Spectral Bias](#positional-encoding-and-spectral-bias)
5. [Hierarchical Volume Sampling](#hierarchical-volume-sampling)
6. [Training NeRF](#training-nerf)
7. [Instant-NGP: Hash Grid Encoding](#instant-ngp-hash-grid-encoding)
8. [Extending to Dynamic Scenes](#extending-to-dynamic-scenes)
9. [Factored 4D Representations](#factored-4d-representations)
10. [NeRF Meets Video Generation](#nerf-meets-video-generation)
11. [Python: Volume Rendering in 2D](#python-volume-rendering-in-2d)

---

## The Volume Rendering Equation

The mathematical foundation of NeRF is the **volume rendering equation**, which comes from **radiative transfer theory** --- the physics of light traveling through a participating medium (fog, smoke, clouds, or in our case, a neural density field).

### The Differential Equation

Consider a ray \(\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}\) originating at a camera center \(\mathbf{o}\) and traveling in direction \(\mathbf{d}\). As the ray passes through the medium, light is:

- **Absorbed** at a rate proportional to the density \(\sigma(t)\) at each point
- **Emitted** with color \(\mathbf{c}(t)\) at each point (in NeRF, the "emission" is the learned radiance)

The **Beer-Lambert law** describes absorption. If light of intensity \(L_0\) enters a medium of density \(\sigma\) over an infinitesimal distance \(dt\), the fraction absorbed is:

$$dL = -\sigma(t) \, L(t) \, dt$$

This is a first-order ODE. The solution is exponential decay:

$$L(t) = L_0 \exp\!\left(-\int_0^t \sigma(s) \, ds\right)$$

The exponential factor is the **transmittance** --- the probability that a photon travels from 0 to \(t\) without being absorbed:

$$T(t) = \exp\!\left(-\int_0^t \sigma(s) \, ds\right)$$

Properties of transmittance:
- \(T(0) = 1\) (nothing absorbed at the start)
- \(T(t)\) is monotonically decreasing
- \(T(t) \to 0\) as \(t \to \infty\) if \(\sigma > 0\) everywhere (all light eventually absorbed)

### The Full Rendering Integral

Including emission, the radiance accumulated along the ray from near plane \(t_n\) to far plane \(t_f\) is:

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \, \sigma(\mathbf{r}(t)) \, \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt$$

This integral has a beautiful physical interpretation:
- At each point \(t\) along the ray, the medium emits light of color \(\mathbf{c}(t)\) with intensity proportional to the density \(\sigma(t)\)
- This emitted light is attenuated by the transmittance \(T(t)\) --- the fraction of light that survives the journey from point \(t\) back to the camera
- The final color is the sum of all these attenuated emissions along the ray

<svg viewBox="0 0 700 300" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <defs>
    <marker id="arrowRay" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#4fc3f7"/>
    </marker>
  </defs>
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">Volume Rendering: Ray Marching Through a Scene</text>

  <!-- Camera -->
  <polygon points="40,140 70,120 70,160" fill="#FF9800" opacity="0.8"/>
  <text x="40" y="175" font-size="10" fill="#FF9800">Camera (o)</text>

  <!-- Ray -->
  <line x1="70" y1="140" x2="650" y2="140" stroke="#4fc3f7" stroke-width="2" marker-end="url(#arrowRay)"/>
  <text x="660" y="145" font-size="10" fill="#4fc3f7">d</text>

  <!-- Sample points -->
  <circle cx="150" cy="140" r="5" fill="#66bb6a" opacity="0.3"/>
  <circle cx="230" cy="140" r="5" fill="#66bb6a" opacity="0.5"/>
  <circle cx="310" cy="140" r="5" fill="#66bb6a" opacity="0.9"/>
  <circle cx="390" cy="140" r="5" fill="#66bb6a" opacity="0.7"/>
  <circle cx="470" cy="140" r="5" fill="#66bb6a" opacity="0.2"/>
  <circle cx="550" cy="140" r="5" fill="#66bb6a" opacity="0.1"/>

  <!-- Density profile -->
  <text x="350" y="75" text-anchor="middle" font-size="11" fill="#E53935">σ(t) — density</text>
  <polyline points="150,100 190,95 230,85 270,80 310,60 350,70 390,75 430,90 470,98 510,100 550,102" fill="none" stroke="#E53935" stroke-width="1.5"/>

  <!-- Transmittance profile -->
  <text x="350" y="235" text-anchor="middle" font-size="11" fill="#2196F3">T(t) — transmittance</text>
  <polyline points="150,195 190,197 230,202 270,210 310,230 350,240 390,248 430,255 470,258 510,260 550,261" fill="none" stroke="#2196F3" stroke-width="1.5"/>

  <!-- Labels -->
  <text x="150" y="270" text-anchor="middle" font-size="9" fill="#999">t_n</text>
  <text x="550" y="270" text-anchor="middle" font-size="9" fill="#999">t_f</text>
  <text x="310" y="50" text-anchor="middle" font-size="9" fill="#E53935">high density</text>
  <text x="350" y="280" text-anchor="middle" font-size="10" fill="#d4d4d4">C(r) = ∫ T(t) σ(t) c(t) dt</text>
</svg>

---

## Discretizing the Integral: Alpha Compositing

The continuous integral cannot be evaluated analytically (since \(\sigma\) and \(\mathbf{c}\) come from a neural network), so we approximate it numerically.

### Stratified Sampling

Divide the ray interval \([t_n, t_f]\) into \(N\) bins and sample one point uniformly within each bin:

$$t_i \sim \mathcal{U}\!\left[t_n + \frac{i-1}{N}(t_f - t_n), \, t_n + \frac{i}{N}(t_f - t_n)\right]$$

The jittering within each bin is crucial --- it turns a deterministic quadrature (which aliases at regular frequencies) into a stochastic estimator that converges without systematic bias.

### The Discrete Formula

With sample points \(t_1 < t_2 < \cdots < t_N\) and spacings \(\delta_i = t_{i+1} - t_i\), the integral is approximated by:

$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \, \alpha_i \, \mathbf{c}_i$$

where the discrete transmittance and alpha are:

$$T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)$$

$$\alpha_i = 1 - \exp(-\sigma_i \, \delta_i)$$

### Derivation

This formula comes from approximating the continuous transmittance as piecewise constant within each interval. If \(\sigma(t) = \sigma_i\) for \(t \in [t_i, t_{i+1}]\), then:

$$T(t_{i+1}) = T(t_i) \exp(-\sigma_i \delta_i) = T(t_i)(1 - \alpha_i)$$

The contribution of interval \(i\) to the integral is approximately:

$$\int_{t_i}^{t_{i+1}} T(t) \sigma(t) \mathbf{c}(t) \, dt \approx T(t_i) \mathbf{c}_i \int_{t_i}^{t_{i+1}} \sigma_i \exp(-\sigma_i (t - t_i)) \, dt = T(t_i) \mathbf{c}_i \left[1 - \exp(-\sigma_i \delta_i)\right] = T_i \alpha_i \mathbf{c}_i$$

This is exactly the **front-to-back alpha compositing** formula from computer graphics. The connection is not a coincidence --- alpha compositing was originally derived as an approximation to volume rendering.

Note that \(\alpha_i = 1 - \exp(-\sigma_i \delta_i)\) maps the density-times-distance product to an opacity in \([0, 1)\):
- \(\sigma_i \delta_i = 0 \implies \alpha_i = 0\) (transparent)
- \(\sigma_i \delta_i \to \infty \implies \alpha_i \to 1\) (opaque)

---

## Neural Radiance Fields: The MLP

NeRF represents the scene as a multilayer perceptron (MLP) \(F_\theta\) that maps a 3D position \(\mathbf{x} = (x, y, z)\) and viewing direction \(\mathbf{d} = (\theta, \phi)\) to a density \(\sigma\) and RGB color \(\mathbf{c}\):

$$F_\theta: (\mathbf{x}, \mathbf{d}) \to (\sigma, \mathbf{c})$$

A critical design choice: **density is independent of viewing direction, but color is not.** The density function \(\sigma(\mathbf{x})\) encodes the scene geometry (where is there stuff?), which does not change with viewpoint. The color function \(\mathbf{c}(\mathbf{x}, \mathbf{d})\) encodes appearance, which can be view-dependent (specular highlights, reflections).

In practice, the MLP first processes position \(\mathbf{x}\) through 8 layers (256 units each) to produce \(\sigma\) and a feature vector, then concatenates the viewing direction and passes through 1 additional layer to produce color. The density branch is deeper because geometry requires understanding global structure, while color is a local, view-dependent modulation.

<svg viewBox="0 0 700 200" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="20" text-anchor="middle" font-size="13" font-weight="bold" fill="#d4d4d4">NeRF MLP Architecture</text>

  <!-- Input: position -->
  <rect x="20" y="60" width="80" height="40" rx="4" fill="#FF9800" opacity="0.2" stroke="#FF9800" stroke-width="1.5"/>
  <text x="60" y="77" text-anchor="middle" font-size="9" fill="#FF9800">γ(x,y,z)</text>
  <text x="60" y="90" text-anchor="middle" font-size="8" fill="#999">60-dim</text>

  <!-- MLP layers -->
  <text x="120" y="83" font-size="14" fill="#666">→</text>
  <rect x="135" y="55" width="180" height="50" rx="4" fill="#2196F3" opacity="0.15" stroke="#2196F3" stroke-width="1.5"/>
  <text x="225" y="78" text-anchor="middle" font-size="10" fill="#2196F3">8 × FC(256) + ReLU</text>
  <text x="225" y="93" text-anchor="middle" font-size="8" fill="#999">+ skip connection at layer 5</text>

  <!-- Density output -->
  <text x="330" y="60" font-size="14" fill="#666">→</text>
  <rect x="345" y="40" width="60" height="30" rx="4" fill="#E53935" opacity="0.2" stroke="#E53935" stroke-width="1.5"/>
  <text x="375" y="60" text-anchor="middle" font-size="10" fill="#E53935">σ</text>

  <!-- Feature + direction -->
  <rect x="345" y="80" width="60" height="30" rx="4" fill="#2196F3" opacity="0.2" stroke="#2196F3" stroke-width="1.5"/>
  <text x="375" y="100" text-anchor="middle" font-size="9" fill="#2196F3">feat</text>

  <!-- Direction input -->
  <rect x="20" y="130" width="80" height="35" rx="4" fill="#66bb6a" opacity="0.2" stroke="#66bb6a" stroke-width="1.5"/>
  <text x="60" y="147" text-anchor="middle" font-size="9" fill="#66bb6a">γ(θ,φ)</text>
  <text x="60" y="160" text-anchor="middle" font-size="8" fill="#999">24-dim</text>

  <!-- Concat -->
  <line x1="100" y1="148" x2="415" y2="148" stroke="#66bb6a" stroke-width="1" stroke-dasharray="4,3"/>
  <line x1="405" y1="95" x2="435" y2="140" stroke="#2196F3" stroke-width="1" stroke-dasharray="4,3"/>

  <!-- Color head -->
  <rect x="440" y="125" width="120" height="40" rx="4" fill="#CE93D8" opacity="0.15" stroke="#CE93D8" stroke-width="1.5"/>
  <text x="500" y="143" text-anchor="middle" font-size="10" fill="#CE93D8">FC(128) + ReLU</text>
  <text x="500" y="157" text-anchor="middle" font-size="8" fill="#999">feat + γ(d)</text>

  <text x="575" y="148" font-size="14" fill="#666">→</text>
  <rect x="590" y="130" width="70" height="35" rx="4" fill="#4fc3f7" opacity="0.2" stroke="#4fc3f7" stroke-width="1.5"/>
  <text x="625" y="148" text-anchor="middle" font-size="10" fill="#4fc3f7">c (RGB)</text>
</svg>

---

## Positional Encoding and Spectral Bias

A plain MLP \(F_\theta(\mathbf{x})\) with smooth activations (ReLU, etc.) has a **spectral bias**: it learns low-frequency functions first and struggles to represent high-frequency detail. This is a problem for NeRF, where scenes contain sharp edges, fine textures, and intricate geometry.

The reason is well-understood from the theory of the **Neural Tangent Kernel (NTK)**. An MLP's NTK kernel has a spectrum that decays rapidly with frequency, meaning the network's effective learning rate for high-frequency components is much lower than for low-frequency ones. Training converges first on the coarse structure and only slowly (if ever) fits the fine details.

**Positional encoding** (Fourier feature mapping) is the solution. Instead of feeding raw coordinates \(\mathbf{x}\) to the MLP, we map them through sinusoidal functions at geometrically spaced frequencies:

$$\gamma(\mathbf{x}) = \left(\sin(2^0 \pi \mathbf{x}), \cos(2^0 \pi \mathbf{x}), \sin(2^1 \pi \mathbf{x}), \cos(2^1 \pi \mathbf{x}), \ldots, \sin(2^{L-1} \pi \mathbf{x}), \cos(2^{L-1} \pi \mathbf{x})\right)$$

For a 3D input and \(L = 10\) frequency bands, this maps \(\mathbb{R}^3 \to \mathbb{R}^{60}\).

**Why this works:** The key theorem (Tancik et al., 2020) shows that the NTK of a network with Fourier features has a **tunable bandwidth**. By choosing the frequencies in \(\gamma\), you directly control which spatial frequencies the network can learn efficiently. The geometric spacing \(2^0, 2^1, \ldots, 2^{L-1}\) covers frequencies from scene scale down to fine detail.

For viewing direction, a smaller encoding is used (\(L = 4\), giving 24 dimensions), since view-dependent effects are typically low-frequency (broad specular lobes).

---

## Hierarchical Volume Sampling

Evaluating the MLP at \(N\) points per ray is expensive. Worse, most sample points fall in empty space or behind occluding surfaces, contributing nothing to the final color. **Hierarchical sampling** allocates samples where they matter.

### Coarse Network

First, evaluate a "coarse" network at \(N_c = 64\) stratified sample points along each ray. This produces a rough density profile. The discrete weights:

$$w_i = T_i \alpha_i$$

represent how much each sample contributes to the final color. Normalize them:

$$\hat{w}_i = \frac{w_i}{\sum_j w_j}$$

These weights define a piecewise-constant probability distribution along the ray.

### Fine Network: Importance Sampling

Sample \(N_f = 128\) additional points from this distribution using **inverse CDF sampling**:

1. Compute the cumulative distribution: \(F_i = \sum_{j=1}^{i} \hat{w}_j\)
2. Draw uniform samples \(u_k \sim \mathcal{U}[0, 1]\)
3. For each \(u_k\), find the bin \(i\) such that \(F_{i-1} \leq u_k < F_i\)
4. Sample uniformly within that bin: \(t_k = t_i + (u_k - F_{i-1}) / \hat{w}_i \cdot \delta_i\)

This concentrates samples near surfaces (where \(\sigma\) is large) and away from empty space. The fine network is evaluated at all \(N_c + N_f\) points (coarse + fine) and produces the final rendering.

<svg viewBox="0 0 700 250" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-size="13" font-weight="bold" fill="#d4d4d4">Hierarchical Sampling</text>

  <!-- Coarse samples -->
  <text x="60" y="55" font-size="11" fill="#4fc3f7">Coarse (uniform):</text>
  <line x1="150" y1="55" x2="650" y2="55" stroke="#666" stroke-width="1"/>
  <circle cx="185" cy="55" r="3" fill="#4fc3f7"/>
  <circle cx="235" cy="55" r="3" fill="#4fc3f7"/>
  <circle cx="285" cy="55" r="3" fill="#4fc3f7"/>
  <circle cx="335" cy="55" r="3" fill="#4fc3f7"/>
  <circle cx="385" cy="55" r="3" fill="#4fc3f7"/>
  <circle cx="435" cy="55" r="3" fill="#4fc3f7"/>
  <circle cx="485" cy="55" r="3" fill="#4fc3f7"/>
  <circle cx="535" cy="55" r="3" fill="#4fc3f7"/>
  <circle cx="585" cy="55" r="3" fill="#4fc3f7"/>
  <circle cx="635" cy="55" r="3" fill="#4fc3f7"/>

  <!-- Weight distribution -->
  <text x="60" y="125" font-size="11" fill="#E53935">Density profile:</text>
  <polyline points="185,130 235,125 285,100 310,75 335,65 360,70 385,95 435,125 485,130 535,130 585,130 635,130" fill="none" stroke="#E53935" stroke-width="2"/>
  <rect x="270" y="60" width="120" height="75" fill="#E53935" opacity="0.05" stroke="#E53935" stroke-width="0.5" stroke-dasharray="3,3"/>
  <text x="330" y="150" text-anchor="middle" font-size="9" fill="#E53935">surface here</text>

  <!-- Fine samples (importance sampled) -->
  <text x="60" y="195" font-size="11" fill="#66bb6a">Fine (importance):</text>
  <line x1="150" y1="195" x2="650" y2="195" stroke="#666" stroke-width="1"/>
  <circle cx="275" cy="195" r="3" fill="#66bb6a"/>
  <circle cx="290" cy="195" r="3" fill="#66bb6a"/>
  <circle cx="302" cy="195" r="3" fill="#66bb6a"/>
  <circle cx="315" cy="195" r="3" fill="#66bb6a"/>
  <circle cx="325" cy="195" r="3" fill="#66bb6a"/>
  <circle cx="338" cy="195" r="3" fill="#66bb6a"/>
  <circle cx="348" cy="195" r="3" fill="#66bb6a"/>
  <circle cx="360" cy="195" r="3" fill="#66bb6a"/>
  <circle cx="372" cy="195" r="3" fill="#66bb6a"/>
  <circle cx="385" cy="195" r="3" fill="#66bb6a"/>
  <circle cx="230" cy="195" r="3" fill="#66bb6a"/>
  <circle cx="400" cy="195" r="3" fill="#66bb6a"/>
  <circle cx="485" cy="195" r="3" fill="#66bb6a"/>
  <circle cx="210" cy="195" r="3" fill="#66bb6a"/>

  <text x="350" y="230" text-anchor="middle" font-size="10" fill="#999">Samples concentrate near the surface, where density is high</text>
</svg>

---

## Training NeRF

### The Loss

The training loss is simply the mean squared error between rendered pixel colors and ground-truth pixel colors:

$$\mathcal{L} = \sum_{\mathbf{r} \in \mathcal{R}} \left\| \hat{C}_c(\mathbf{r}) - C^*(\mathbf{r}) \right\|_2^2 + \left\| \hat{C}_f(\mathbf{r}) - C^*(\mathbf{r}) \right\|_2^2$$

where \(\mathcal{R}\) is a batch of rays, \(\hat{C}_c\) and \(\hat{C}_f\) are the coarse and fine rendered colors, and \(C^*\) is the ground-truth pixel color.

That is it. No 3D supervision, no depth maps, no segmentation masks. Just photometric error. The 3D structure emerges because it is the only way to consistently explain the 2D observations from multiple viewpoints.

### Camera Poses

NeRF requires known camera poses (position and orientation for each training image). These are typically obtained via **Structure from Motion** (SfM) using COLMAP. Each camera has:
- An extrinsic matrix \([R | \mathbf{t}] \in \mathbb{R}^{3 \times 4}\) (rotation and translation)
- An intrinsic matrix with focal length \(f\) and principal point

Rays are cast from the camera center through each pixel: \(\mathbf{r}(t) = \mathbf{o} + t \cdot \mathbf{d}\), where \(\mathbf{o}\) is the camera position and \(\mathbf{d}\) is the pixel's direction in world coordinates.

### Training Details

- Batch size: 4096 rays per iteration (not full images --- random rays from random views)
- Optimizer: Adam with learning rate \(5 \times 10^{-4}\), decaying exponentially
- Training: ~100K--300K iterations (hours on a single GPU)
- The network has ~1.2M parameters (tiny by modern standards)

---

## Instant-NGP: Hash Grid Encoding

NeRF is slow. Training takes hours; rendering takes seconds per frame. **Instant Neural Graphics Primitives (Instant-NGP)** by Muller et al. (2022) replaces the positional encoding with a **multi-resolution hash grid**, reducing training to minutes and enabling real-time rendering.

### The Idea

Instead of encoding position with sinusoidal functions, store **learnable feature vectors** on a multi-resolution grid. For a query point \(\mathbf{x}\):

1. Define \(L\) resolution levels, with grid spacing geometrically increasing from fine to coarse
2. At each level, find the grid cell containing \(\mathbf{x}\) and look up the feature vectors at its corners
3. Trilinearly interpolate the corner features to get a per-level feature
4. Concatenate features from all levels and pass through a small MLP (2 layers, 64 units)

### Hash Encoding

At fine resolutions, a full dense grid would be enormous. The trick: use a **hash table** of fixed size \(T\) (e.g., \(2^{19}\)) at each level. Grid vertices are mapped to hash table entries via a spatial hash function:

$$h(\mathbf{x}_{\text{grid}}) = \left(\bigoplus_{d=1}^{3} x_d \cdot \pi_d \right) \mod T$$

where \(\bigoplus\) is XOR and \(\pi_d\) are large prime numbers.

Collisions are inevitable at fine resolutions (many grid points share a hash entry), but this is handled gracefully: the gradients from different collision partners average out, and the small MLP learns to disambiguate. The theory shows that fine-scale details that collide are typically spatially distant and do not interfere perceptually.

The result: training in ~5 minutes (vs. hours), real-time rendering at 30+ FPS.

---

## Extending to Dynamic Scenes

For video, the scene changes over time. Several approaches extend NeRF to the temporal domain:

### D-NeRF: Deformation Fields

Add a time-conditioned deformation network \(\mathcal{D}_\theta(\mathbf{x}, t) \to \Delta \mathbf{x}\) that warps points from the observation time \(t\) to a canonical frame:

$$\mathbf{x}_{\text{canonical}} = \mathbf{x} + \mathcal{D}_\theta(\mathbf{x}, t)$$

Then evaluate the static NeRF at the deformed position: \(F_\theta(\mathbf{x}_{\text{canonical}}, \mathbf{d}) \to (\sigma, \mathbf{c})\).

This separates geometry (canonical NeRF) from motion (deformation field). The canonical NeRF captures the scene's structure, and the deformation field captures how it moves over time.

### HyperNeRF: Higher-Dimensional Lifting

Some deformations are topological (a person opening their mouth creates new surfaces). HyperNeRF handles this by lifting the ambient space to higher dimensions: map \((\mathbf{x}, t)\) to a higher-dimensional "hyper-space" \((\mathbf{x}, \mathbf{w}(t))\) where the topology change becomes a smooth deformation. The additional dimensions \(\mathbf{w}(t)\) provide enough room for the surface to continuously deform without tearing.

---

## Factored 4D Representations

Full 4D representations (3D space + time) are expensive. Factored approaches decompose the 4D volume into lower-dimensional components:

**K-Planes** (Fridovich-Keil et al., 2023): Represent the 4D field as a product of six 2D plane features (one for each pair of axes: xy, xz, xt, yz, yt, zt). For a query point \((x, y, z, t)\), look up features from each plane and combine them:

$$f(x,y,z,t) = \prod_{k=1}^{6} P_k(\text{project}_{k}(x,y,z,t))$$

Each plane \(P_k\) is a 2D grid of feature vectors, interpolated bilinearly. This reduces storage from \(O(N^4)\) to \(O(N^2)\) and supports efficient optimization.

**HexPlane** and **TensoRF** use similar factorizations but with different combination operations (concatenation, summation, or Hadamard product).

These factored representations train in minutes and render at interactive rates, making them practical for video-length dynamic scene reconstruction.

---

## NeRF Meets Video Generation

The connection between 3D reconstruction (NeRF) and video generation runs both ways:

**3D data for training video models.** Large-scale 3D reconstructions from video datasets provide training data with explicit camera poses, depth maps, and multi-view consistency. Video generation models trained on this data learn better 3D structure.

**Camera control in video generation.** Models like Sora generate video with camera movement. Internally, they benefit from 3D understanding --- either explicit (3D Gaussian Splatting backbone) or implicit (learned from 3D-aware training data). NeRF-style representations provide the language and data for specifying camera trajectories.

**World models.** A world model predicts how a scene evolves, including novel viewpoints, physics, and interaction. NeRF provides the 3D representation; a dynamics model (diffusion, autoregressive, or physics-based) provides temporal evolution. Together, they form a complete generative world model.

**NeRF-guided video editing.** Reconstruct a scene from video using NeRF, edit the 3D representation (add objects, change colors, alter geometry), and re-render. This provides geometrically consistent video editing that a purely 2D model cannot guarantee.

---

## Python: Volume Rendering in 2D

```python
import numpy as np
import matplotlib.pyplot as plt

def create_scene_2d(grid_size=256):
    """Create a 2D density field (simulating a cross-section of a 3D scene)."""
    y, x = np.mgrid[0:grid_size, 0:grid_size].astype(float) / grid_size

    # Two blobs (objects) and a gradient background
    blob1 = 5.0 * np.exp(-((x - 0.3)**2 + (y - 0.5)**2) / (2 * 0.05**2))
    blob2 = 8.0 * np.exp(-((x - 0.7)**2 + (y - 0.4)**2) / (2 * 0.04**2))
    density = blob1 + blob2

    # Colors for each region (RGB)
    color = np.zeros((grid_size, grid_size, 3))
    color[:, :, 0] = 0.9 * blob1 / (blob1.max() + 1e-8) + 0.2 * blob2 / (blob2.max() + 1e-8)
    color[:, :, 1] = 0.2 * blob1 / (blob1.max() + 1e-8) + 0.8 * blob2 / (blob2.max() + 1e-8)
    color[:, :, 2] = 0.3 * blob1 / (blob1.max() + 1e-8) + 0.3 * blob2 / (blob2.max() + 1e-8)

    return density, color

def volume_render_row(density, color, row, n_samples=128):
    """Render a single horizontal ray through the 2D density field."""
    grid_size = density.shape[1]
    t_vals = np.linspace(0, grid_size - 1, n_samples).astype(int)
    t_vals = np.clip(t_vals, 0, grid_size - 1)
    delta = grid_size / n_samples

    sigmas = density[row, t_vals]
    colors = color[row, t_vals]

    # Compute alpha and transmittance
    alpha = 1.0 - np.exp(-sigmas * delta / grid_size * 10)
    transmittance = np.ones(n_samples)
    for i in range(1, n_samples):
        transmittance[i] = transmittance[i-1] * (1 - alpha[i-1])

    # Accumulate color
    weights = transmittance * alpha
    rendered_color = np.sum(weights[:, None] * colors, axis=0)

    return rendered_color, transmittance, weights, alpha

# Create scene
density, color = create_scene_2d(256)

# Render all rows (horizontal rays from left to right)
rendered_image = np.zeros((256, 3))
all_weights = np.zeros((256, 128))
all_transmittance = np.zeros((256, 128))

for row in range(256):
    rendered_image[row], trans, w, _ = volume_render_row(density, color, row)
    all_weights[row] = w
    all_transmittance[row] = trans

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Density field
axes[0, 0].imshow(density, cmap='hot', aspect='auto')
axes[0, 0].set_title(r'Density field $\sigma(x, y)$')
axes[0, 0].set_xlabel(r'$x$ (ray direction)')
axes[0, 0].set_ylabel(r'$y$ (pixel row)')

# Color field
color_display = np.clip(color / (color.max() + 1e-8), 0, 1)
axes[0, 1].imshow(color_display, aspect='auto')
axes[0, 1].set_title(r'Color field $\mathbf{c}(x, y)$')
axes[0, 1].set_xlabel(r'$x$')

# Rendered 1D image (shown as vertical strip)
rendered_strip = np.clip(rendered_image, 0, 1)
axes[0, 2].imshow(rendered_strip[:, None, :].repeat(30, axis=1), aspect='auto')
axes[0, 2].set_title('Rendered Image (1D)')
axes[0, 2].set_xlabel('')
axes[0, 2].set_ylabel(r'Pixel row $y$')

# Weight distribution for a specific row
test_row = 128
_, trans_row, weights_row, alpha_row = volume_render_row(density, color, test_row)
t_vals = np.linspace(0, 1, 128)

axes[1, 0].fill_between(t_vals, weights_row, alpha=0.5, color='#4fc3f7')
axes[1, 0].plot(t_vals, weights_row, color='#4fc3f7', linewidth=1.5)
axes[1, 0].set_xlabel(r'Ray parameter $t$')
axes[1, 0].set_ylabel(r'Weight $w_i = T_i \alpha_i$')
axes[1, 0].set_title(f'Sample Weights (row {test_row})')
axes[1, 0].grid(True, alpha=0.3)

# Transmittance
axes[1, 1].plot(t_vals, trans_row, color='#66bb6a', linewidth=2)
axes[1, 1].set_xlabel(r'Ray parameter $t$')
axes[1, 1].set_ylabel(r'Transmittance $T(t)$')
axes[1, 1].set_title(f'Transmittance (row {test_row})')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim(0, 1.05)

# Weight map (all rows)
axes[1, 2].imshow(all_weights, cmap='viridis', aspect='auto')
axes[1, 2].set_xlabel(r'Sample index along ray')
axes[1, 2].set_ylabel(r'Pixel row $y$')
axes[1, 2].set_title(r'Weight map $w_i$ (all rows)')

plt.suptitle('Volume Rendering: From Density Field to Image', fontsize=14)
plt.tight_layout()
plt.savefig('nerf_volume_rendering.png', dpi=150, bbox_inches='tight')
plt.show()
```

NeRF demonstrated something remarkable: a simple MLP, trained only on 2D photographs with known camera poses, can reconstruct a complete 3D scene that can be rendered from any viewpoint. The mathematics is grounded in classical radiative transfer, but the representation is entirely learned. For video generation, NeRF and its successors provide the bridge between 2D generative models and 3D-consistent output --- the foundation for camera control, world models, and physically grounded video synthesis.
