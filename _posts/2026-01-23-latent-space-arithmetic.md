---
layout: post
title: "Latent Space Arithmetic: Why Video Models Understand 'Style + Subject' and How to Exploit It"
date: 2026-01-23
category: math
---

Every time you type a prompt into a video generation model and get something coherent back, you are witnessing computation in a latent space. Not pixel space. Not text space. A learned, compressed, continuous representation where the model has organized visual concepts into a geometry that supports arithmetic. Style plus subject. Motion minus camera shake. Happy scene shifted toward melancholy.

This post is a deep mathematical dive into latent spaces: what they are, why they have structure, how interpolation and arithmetic work within them, and how to exploit these properties for video generation. We will derive the key equations from first principles, build geometric intuition with visualizations, and connect the theory to practical techniques you can use today.

---

## Table of Contents

1. [What Latent Spaces Are](#what-latent-spaces-are)
2. [The Manifold Hypothesis](#the-manifold-hypothesis)
3. [The VAE Latent Space: Full Derivation](#the-vae-latent-space-full-derivation)
4. [Why KL Divergence Creates Structure](#why-kl-divergence-creates-structure)
5. [Linear Interpolation (LERP)](#linear-interpolation-lerp)
6. [Spherical Interpolation (SLERP)](#spherical-interpolation-slerp)
7. [The Curse of Dimensionality and Why SLERP Wins](#the-curse-of-dimensionality-and-why-slerp-wins)
8. [Disentangled Representations](#disentangled-representations)
9. [Arithmetic in Latent Space](#arithmetic-in-latent-space)
10. [Applications for Video Generation](#applications-for-video-generation)
11. [Practical Recipes](#practical-recipes)
12. [Conclusion](#conclusion)

---

## What Latent Spaces Are

A latent space is a learned compressed representation of data. The word "latent" means hidden -- these are variables that we never observe directly, but that the model learns to use internally.

Consider a 512x512 RGB video frame. In pixel space, this is a vector in \(\mathbb{R}^{786432}\) (512 x 512 x 3 = 786,432 dimensions). Most of this space is empty. Random vectors in \(\mathbb{R}^{786432}\) do not look like natural images -- they look like static noise. The set of natural images occupies an infinitesimally thin subregion of pixel space.

A latent space compresses this. A typical video model encoder maps each frame to a vector \(z \in \mathbb{R}^{d}\) where \(d\) might be 4, 8, 16, or at most a few hundred dimensions. The critical property is that this compression is not random -- it is **learned** to preserve semantic structure.

Formally, an encoder \(E: \mathcal{X} \rightarrow \mathcal{Z}\) maps from input space \(\mathcal{X}\) (images, video frames) to latent space \(\mathcal{Z}\), and a decoder \(D: \mathcal{Z} \rightarrow \mathcal{X}\) maps back. The pair is trained so that \(D(E(x)) \approx x\) -- reconstruction fidelity -- while the latent space \(\mathcal{Z}\) is simultaneously constrained to have useful geometric properties.

```
Input Space (High-D)          Latent Space (Low-D)         Output Space (High-D)
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│ 512x512x3        │  E(x)   │ z ∈ R^d          │  D(z)   │ 512x512x3        │
│ = 786,432 dims   │ ──────> │ d = 4 to 512     │ ──────> │ ≈ Original input │
│                  │         │                  │         │                  │
│ Mostly empty     │         │ Dense, structured│         │ Reconstruction   │
│ (random = noise) │         │ (nearby = similar)│        │                  │
└──────────────────┘         └──────────────────┘         └──────────────────┘
```

The key insight: **semantically similar inputs map to nearby points in latent space**. Two images of the same person with slightly different expressions will have latent vectors that are close together. Two images of entirely different scenes will be far apart. This is not an accident -- it emerges from training because the decoder needs nearby codes to produce similar outputs for the reconstruction loss to be low.

---

## The Manifold Hypothesis

The manifold hypothesis states that high-dimensional real-world data (images, video, audio, text) lies on or near a low-dimensional manifold embedded in the high-dimensional ambient space.

Formally: if \(\mathcal{X} \subset \mathbb{R}^{D}\) is the space of all possible pixel configurations and \(\mathcal{M} \subset \mathcal{X}\) is the set of natural images, then \(\mathcal{M}\) is approximately a smooth manifold of dimension \(d \ll D\).

Why should this be true? Consider what determines a face image: the identity of the person (a handful of parameters), their expression (a few more), the lighting direction (3 parameters), the camera angle (3 parameters), the background (a few parameters). Even a generous count gives perhaps 50 to 100 meaningful degrees of freedom. Yet the image lives in a space of 786,432 dimensions. The ratio of intrinsic to ambient dimensionality is roughly 100 : 786,432 or about 0.013%.

This means the data manifold is incredibly thin compared to the ambient space. A latent space model attempts to learn a coordinate system for this manifold -- a set of \(d\) coordinates that parameterize the manifold smoothly.

The implications are profound:

1. **Interpolation is meaningful.** Moving along the manifold (in latent space) corresponds to smooth semantic changes. Moving off the manifold (in pixel space) produces noise.

2. **Arithmetic is meaningful.** Directions in latent space correspond to semantic attributes. You can add, subtract, and combine these directions.

3. **Compression is not lossy in the way you might expect.** Reducing from 786,432 to 512 dimensions does not lose 99.9% of the information, because 99.9% of the ambient dimensions were noise anyway.

<svg viewBox="0 0 700 400" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <defs>
    <marker id="arrow1" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#d4d4d4"/>
    </marker>
  </defs>
  <!-- Title -->
  <text x="350" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#d4d4d4">The Manifold Hypothesis: Data Lives on a Low-D Surface</text>
  <!-- Ambient space box -->
  <rect x="30" y="40" width="300" height="320" rx="8" fill="#1e1e1e" stroke="#444" stroke-width="1"/>
  <text x="180" y="65" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#999">Ambient Space ℝᴰ</text>
  <!-- Random noise dots -->
  <circle cx="50" cy="100" r="3" fill="#444"/>
  <circle cx="90" cy="320" r="3" fill="#444"/>
  <circle cx="270" cy="140" r="3" fill="#444"/>
  <circle cx="300" cy="300" r="3" fill="#444"/>
  <circle cx="150" cy="90" r="3" fill="#444"/>
  <circle cx="200" cy="340" r="3" fill="#444"/>
  <circle cx="60" cy="250" r="3" fill="#444"/>
  <circle cx="310" cy="200" r="3" fill="#444"/>
  <text x="310" y="95" font-family="Arial, sans-serif" font-size="10" fill="#666">noise</text>
  <text x="50" cy="280" font-family="Arial, sans-serif" font-size="10" fill="#666">noise</text>
  <!-- Manifold curve -->
  <path d="M 60,180 C 100,120 140,250 180,200 S 250,150 300,220" fill="none" stroke="#4fc3f7" stroke-width="3"/>
  <text x="160" y="275" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#4fc3f7" font-weight="bold">Data Manifold M</text>
  <!-- Data points on manifold -->
  <circle cx="80" cy="164" r="5" fill="#4fc3f7"/>
  <circle cx="120" cy="178" r="5" fill="#4fc3f7"/>
  <circle cx="160" cy="215" r="5" fill="#4fc3f7"/>
  <circle cx="200" cy="196" r="5" fill="#4fc3f7"/>
  <circle cx="240" cy="178" r="5" fill="#4fc3f7"/>
  <circle cx="280" cy="210" r="5" fill="#4fc3f7"/>
  <!-- Arrow to latent space -->
  <line x1="340" y1="200" x2="400" y2="200" stroke="#d4d4d4" stroke-width="2" marker-end="url(#arrow1)"/>
  <text x="370" y="190" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#d4d4d4">E(x)</text>
  <!-- Latent space -->
  <rect x="410" y="80" width="260" height="240" rx="8" fill="#162030" stroke="#4fc3f7" stroke-width="2"/>
  <text x="540" y="105" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#4fc3f7" font-weight="bold">Latent Space ℝᵈ</text>
  <!-- Latent points spread nicely -->
  <circle cx="450" cy="160" r="6" fill="#4fc3f7"/>
  <circle cx="480" cy="180" r="6" fill="#4fc3f7"/>
  <circle cx="510" cy="200" r="6" fill="#4fc3f7"/>
  <circle cx="540" cy="215" r="6" fill="#4fc3f7"/>
  <circle cx="575" cy="230" r="6" fill="#4fc3f7"/>
  <circle cx="610" cy="250" r="6" fill="#4fc3f7"/>
  <!-- Labels -->
  <text x="540" y="290" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#999">d = 4 to 512 dimensions</text>
  <text x="540" y="305" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#999">Dense, structured, navigable</text>
  <!-- Dimension labels -->
  <text x="180" y="355" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#999">D = 786,432 dimensions</text>
  <text x="180" y="370" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#999">~99.99% empty</text>
</svg>

---

## The VAE Latent Space: Full Derivation

The Variational Autoencoder (VAE) is the most principled framework for building latent spaces. It combines neural network function approximation with Bayesian inference. Let us derive the full objective from first principles.

### The Generative Model

We posit a generative model for data \(x\):

1. Sample a latent variable: \(z \sim p(z) = \mathcal{N}(0, I)\)
2. Generate data from the latent: \(x \sim p_\theta(x|z)\)

The marginal likelihood of the data is:

$$p_\theta(x) = \int p_\theta(x|z) \, p(z) \, dz$$

This integral is intractable -- we cannot compute it exactly because it requires integrating over all possible latent codes \(z\).

### The Variational Lower Bound

Since we cannot compute \(p_\theta(x)\) directly, we introduce an approximate posterior \(q_\phi(z|x)\) and derive a lower bound on \(\log p_\theta(x)\).

Start with the log-marginal-likelihood:

$$\log p_\theta(x) = \log \int p_\theta(x|z) \, p(z) \, dz$$

Multiply and divide by \(q_\phi(z|x)\):

$$\log p_\theta(x) = \log \int \frac{p_\theta(x|z) \, p(z)}{q_\phi(z|x)} \, q_\phi(z|x) \, dz$$

$$= \log \mathbb{E}_{q_\phi(z|x)} \left[ \frac{p_\theta(x|z) \, p(z)}{q_\phi(z|x)} \right]$$

By Jensen's inequality (\(\log \mathbb{E}[X] \geq \mathbb{E}[\log X]\) for concave \(\log\)):

$$\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)} \left[ \log \frac{p_\theta(x|z) \, p(z)}{q_\phi(z|x)} \right]$$

Expanding:

$$\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$$

This is the **Evidence Lower Bound (ELBO)**:

$$\text{ELBO} = \underbrace{\mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)]}_{\text{Reconstruction term}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{Regularization term}}$$

The first term encourages the decoder to reconstruct \(x\) from \(z\). The second term encourages the encoder's posterior to stay close to the prior \(p(z) = \mathcal{N}(0, I)\).

### The Encoder Parameterization

The encoder \(q_\phi(z|x)\) is parameterized as a diagonal Gaussian:

$$q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$$

The neural network takes input \(x\) and outputs two vectors: \(\mu_\phi(x) \in \mathbb{R}^d\) and \(\log \sigma_\phi^2(x) \in \mathbb{R}^d\).

### Deriving the KL Divergence Term

For the specific case where \(q_\phi(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))\) and \(p(z) = \mathcal{N}(0, I)\), the KL divergence has a closed-form solution. Let us derive it.

The KL divergence between two distributions is:

$$D_{KL}(q \| p) = \int q(z) \log \frac{q(z)}{p(z)} dz = \mathbb{E}_q[\log q(z) - \log p(z)]$$

For multivariate Gaussians, the general formula is:

$$D_{KL}(\mathcal{N}(\mu_1, \Sigma_1) \| \mathcal{N}(\mu_2, \Sigma_2)) = \frac{1}{2} \left[ \log \frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr}(\Sigma_2^{-1} \Sigma_1) + (\mu_2 - \mu_1)^T \Sigma_2^{-1} (\mu_2 - \mu_1) \right]$$

Substituting \(\mu_1 = \mu\), \(\Sigma_1 = \text{diag}(\sigma^2)\), \(\mu_2 = 0\), \(\Sigma_2 = I\):

$$D_{KL} = \frac{1}{2} \left[ \log \frac{|I|}{|\text{diag}(\sigma^2)|} - d + \text{tr}(I^{-1} \cdot \text{diag}(\sigma^2)) + \mu^T I^{-1} \mu \right]$$

Since \(|I| = 1\), \(|\text{diag}(\sigma^2)| = \prod_j \sigma_j^2\), and \(\text{tr}(\text{diag}(\sigma^2)) = \sum_j \sigma_j^2\):

$$D_{KL} = \frac{1}{2} \left[ -\sum_{j=1}^{d} \log \sigma_j^2 - d + \sum_{j=1}^{d} \sigma_j^2 + \sum_{j=1}^{d} \mu_j^2 \right]$$

$$\boxed{D_{KL}(q(z|x) \| p(z)) = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)}$$

This is the KL loss term used in every VAE implementation. Let us verify the intuition for each component:

| Component | When is it small? | Meaning |
|---|---|---|
| \(\mu_j^2\) | When \(\mu_j \approx 0\) | Posterior mean is near the prior mean |
| \(\sigma_j^2\) | When \(\sigma_j \approx 1\) | Posterior variance matches the prior |
| \(\log \sigma_j^2\) | When \(\sigma_j \approx 1\) | Balances the \(\sigma_j^2\) term |
| $1$ | Always | Constant offset making KL = 0 when \(\mu=0, \sigma=1\) |

When \(\mu_j = 0\) and \(\sigma_j = 1\) for all \(j\), the KL divergence is exactly zero -- the posterior matches the prior perfectly.

### The Reparameterization Trick

To backpropagate through the stochastic sampling step, we use the reparameterization trick:

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This moves the stochasticity into \(\epsilon\), which is independent of the parameters \(\phi\). The gradient with respect to \(\phi\) can now flow through \(\mu_\phi\) and \(\sigma_\phi\):

$$\frac{\partial z}{\partial \phi} = \frac{\partial \mu_\phi}{\partial \phi} + \frac{\partial \sigma_\phi}{\partial \phi} \odot \epsilon$$

Without this trick, we would need REINFORCE-style gradient estimators, which have much higher variance.

---

## Why KL Divergence Creates Structure

The KL divergence term is not just a regularizer -- it is the reason latent spaces have the geometric properties that make arithmetic possible. Let us understand why.

### Continuity

The KL term penalizes the encoder for placing the posterior \(q_\phi(z|x)\) too far from \(\mathcal{N}(0, I)\). This means different inputs cannot be mapped to arbitrarily separated regions of latent space. The posteriors overlap, creating a **continuous** space where intermediate points are also valid.

Without KL regularization (a plain autoencoder), the encoder can map different inputs to isolated islands in latent space. Points between the islands decode to garbage. With KL regularization, the posteriors are forced to overlap with each other and with the prior, filling in the gaps.

### Completeness

Any point sampled from the prior \(z \sim \mathcal{N}(0, I)\) should decode to a plausible output. The KL term ensures this by keeping the aggregate posterior \(q(z) = \mathbb{E}_{p(x)}[q_\phi(z|x)]\) close to \(p(z)\). If the aggregate posterior matches the prior, then sampling from the prior samples from the same distribution the decoder was trained on.

### Smoothness

Small changes in \(z\) produce small changes in \(D(z)\). This follows from the continuity of neural networks and the fact that nearby latent codes often came from similar inputs (due to the overlap enforced by KL).

These three properties -- continuity, completeness, smoothness -- are what make interpolation and arithmetic meaningful in latent space.

```
Without KL (Autoencoder)          With KL (VAE)
┌────────────────────┐            ┌────────────────────┐
│                    │            │  ░░░░░░░░░░░░░░░░  │
│   ●●●              │            │  ░░░●●●░░░░░░░░░░  │
│   cats              │            │  ░░●cats●░░░░░░░░  │
│                    │            │  ░░░●●●░░░░░░░░░░  │
│            ●●●      │            │  ░░░░░░●●●░░░░░░  │
│            dogs     │            │  ░░░░●dogs●░░░░░  │
│                    │            │  ░░░░░░●●●░░░░░░  │
│  Gap = garbage      │            │  ░░░░░░░░░░░░░░░░  │
│  (no valid images) │            │  Smooth transitions │
└────────────────────┘            └────────────────────┘
```

---

## Linear Interpolation (LERP)

The simplest way to move between two latent codes is linear interpolation (LERP):

$$z_{\text{interp}}(t) = (1-t) \cdot z_1 + t \cdot z_2, \quad t \in [0, 1]$$

At \(t = 0\), we get \(z_1\). At \(t = 1\), we get \(z_2\). At \(t = 0.5\), we get the midpoint.

LERP traces a **straight line** through latent space. When decoded, this line produces a sequence of outputs that smoothly transition from the input corresponding to \(z_1\) to the input corresponding to \(z_2\).

### Properties of LERP

**Constant velocity.** The derivative is constant: \(\frac{dz}{dt} = z_2 - z_1\). The interpolated point moves at uniform speed along the line.

**Norm variation.** The norm of the interpolated point varies non-monotonically:

$$\|z_{\text{interp}}(t)\|^2 = (1-t)^2 \|z_1\|^2 + 2t(1-t)(z_1 \cdot z_2) + t^2 \|z_2\|^2$$

If \(z_1\) and \(z_2\) are nearly orthogonal (as they tend to be in high dimensions), then at \(t = 0.5\):

$$\|z_{\text{interp}}(0.5)\|^2 \approx \frac{\|z_1\|^2 + \|z_2\|^2}{4} \approx \frac{\|z_1\|^2}{2}$$

The midpoint has roughly \(\frac{1}{\sqrt{2}} \approx 0.707\) times the norm of the endpoints. **The interpolated path dips toward the origin.** In high-dimensional spaces, this dip passes through a region that the model never saw during training, potentially producing low-quality or blurry outputs.

### When LERP Works

LERP works well when:
- The latent space is low-dimensional (d < 10 or so)
- The two endpoints are close together
- The latent space was trained with very strong regularization

LERP struggles when:
- The latent space is high-dimensional (d > 100)
- The endpoints are far apart
- The data distribution concentrates near a shell (as Gaussians do in high dimensions)

---

## Spherical Interpolation (SLERP)

Spherical Linear Interpolation (SLERP) moves along the **great circle** connecting two points on a hypersphere, rather than cutting through the interior:

$$z_{\text{interp}}(t) = \frac{\sin((1-t)\Omega)}{\sin \Omega} z_1 + \frac{\sin(t\Omega)}{\sin \Omega} z_2$$

where \(\Omega\) is the angle between \(z_1\) and \(z_2\):

$$\Omega = \arccos \left( \frac{z_1 \cdot z_2}{\|z_1\| \|z_2\|} \right)$$

### Derivation

SLERP can be derived from the requirement that the interpolation:
1. Stays on the surface of the sphere
2. Moves at constant angular velocity

Consider two unit vectors \(\hat{z}_1\) and \(\hat{z}_2\) on the unit sphere \(S^{d-1}\). We want a curve \(\gamma(t)\) on the sphere such that \(\gamma(0) = \hat{z}_1\), \(\gamma(1) = \hat{z}_2\), and \(\|\gamma(t)\| = 1\) for all \(t\).

We can decompose \(\hat{z}_2\) into components parallel and perpendicular to \(\hat{z}_1\):

$$\hat{z}_2 = (\hat{z}_1 \cdot \hat{z}_2) \hat{z}_1 + \hat{z}_\perp$$

where \(\hat{z}_\perp\) is the unit vector perpendicular to \(\hat{z}_1\) in the plane spanned by \(\hat{z}_1\) and \(\hat{z}_2\).

The great circle in this plane is parameterized by:

$$\gamma(t) = \cos(t\Omega) \hat{z}_1 + \sin(t\Omega) \hat{e}_\perp$$

where \(\hat{e}_\perp = \frac{\hat{z}_2 - (\hat{z}_1 \cdot \hat{z}_2)\hat{z}_1}{\|\hat{z}_2 - (\hat{z}_1 \cdot \hat{z}_2)\hat{z}_1\|}\) and \(\Omega = \arccos(\hat{z}_1 \cdot \hat{z}_2)\).

Substituting and simplifying (using \(\sin \Omega = \|\hat{z}_2 - \cos\Omega \cdot \hat{z}_1\|\)), we obtain:

$$\gamma(t) = \frac{\sin((1-t)\Omega)}{\sin \Omega} \hat{z}_1 + \frac{\sin(t\Omega)}{\sin \Omega} \hat{z}_2$$

For non-unit vectors, we can either normalize first and then scale, or apply SLERP directly with an interpolated radius: \(r(t) = (1-t)\|z_1\| + t\|z_2\|\).

### Properties of SLERP

**Constant norm (on unit sphere).** If \(\|z_1\| = \|z_2\| = 1\), then \(\|\gamma(t)\| = 1\) for all \(t\). The interpolation stays on the sphere.

**Constant angular velocity.** The angle from \(\hat{z}_1\) to \(\gamma(t)\) is exactly \(t\Omega\). The point moves at uniform angular speed.

**No norm collapse.** Unlike LERP, the midpoint has the same norm as the endpoints. This avoids the "dip toward the origin" problem.

<svg viewBox="0 0 700 420" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#d4d4d4">LERP vs SLERP: Interpolation on a Circle</text>
  <!-- Left diagram - LERP -->
  <text x="175" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#ef5350">Linear Interpolation (LERP)</text>
  <!-- Circle -->
  <circle cx="175" cy="235" r="130" fill="none" stroke="#444" stroke-width="1.5" stroke-dasharray="4,3"/>
  <!-- Origin -->
  <circle cx="175" cy="235" r="3" fill="#999"/>
  <text x="185" y="250" font-family="Arial, sans-serif" font-size="10" fill="#999">origin</text>
  <!-- z1 and z2 -->
  <circle cx="101" cy="122" r="7" fill="#4fc3f7" stroke="white" stroke-width="2"/>
  <text x="75" y="113" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#4fc3f7">z₁</text>
  <circle cx="289" cy="162" r="7" fill="#8bc34a" stroke="white" stroke-width="2"/>
  <text x="300" y="155" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#8bc34a">z₂</text>
  <!-- LERP line (straight) -->
  <line x1="101" y1="122" x2="289" y2="162" stroke="#ef5350" stroke-width="2.5"/>
  <!-- Midpoint on LERP -->
  <circle cx="195" cy="142" r="5" fill="#ef5350"/>
  <text x="195" y="132" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#ef5350">t=0.5</text>
  <!-- Dashed line from origin to midpoint showing reduced norm -->
  <line x1="175" y1="235" x2="195" y2="142" stroke="#ef5350" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="145" y="195" font-family="Arial, sans-serif" font-size="10" fill="#ef5350">|z| ≈ 0.71r</text>
  <!-- Arrow showing dip -->
  <text x="175" y="315" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#ef5350">Cuts through interior</text>
  <text x="175" y="330" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#ef5350">Norm dips at midpoint</text>
  <text x="175" y="345" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#ef5350">May produce blurry outputs</text>
  <!-- Right diagram - SLERP -->
  <text x="525" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#4fc3f7">Spherical Interpolation (SLERP)</text>
  <!-- Circle -->
  <circle cx="525" cy="235" r="130" fill="none" stroke="#444" stroke-width="1.5" stroke-dasharray="4,3"/>
  <!-- Origin -->
  <circle cx="525" cy="235" r="3" fill="#999"/>
  <text x="535" y="250" font-family="Arial, sans-serif" font-size="10" fill="#999">origin</text>
  <!-- z1 and z2 -->
  <circle cx="451" cy="122" r="7" fill="#4fc3f7" stroke="white" stroke-width="2"/>
  <text x="425" y="113" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#4fc3f7">z₁</text>
  <circle cx="639" cy="162" r="7" fill="#8bc34a" stroke="white" stroke-width="2"/>
  <text x="650" y="155" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#8bc34a">z₂</text>
  <!-- SLERP arc (along the circle) -->
  <path d="M 451,122 A 130,130 0 0,1 639,162" fill="none" stroke="#4fc3f7" stroke-width="2.5"/>
  <!-- Midpoint on SLERP (on the circle) -->
  <circle cx="556" cy="107" r="5" fill="#4fc3f7"/>
  <text x="556" y="97" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#4fc3f7">t=0.5</text>
  <!-- Dashed line from origin to midpoint showing preserved norm -->
  <line x1="525" y1="235" x2="556" y2="107" stroke="#4fc3f7" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="570" y="180" font-family="Arial, sans-serif" font-size="10" fill="#4fc3f7">|z| = r</text>
  <!-- Labels -->
  <text x="525" y="315" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#4fc3f7">Follows the great circle</text>
  <text x="525" y="330" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#4fc3f7">Constant norm throughout</text>
  <text x="525" y="345" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#4fc3f7">Stays on the data manifold</text>
  <!-- Divider -->
  <line x1="350" y1="45" x2="350" y2="380" stroke="#444" stroke-width="1" stroke-dasharray="6,4"/>
</svg>

---

## The Curse of Dimensionality and Why SLERP Wins

In high-dimensional spaces, the geometry of Gaussian distributions is deeply counterintuitive. Understanding this geometry explains why SLERP is essential for high-dimensional latent spaces.

### Gaussian Shell Concentration

For \(z \sim \mathcal{N}(0, I_d)\) in \(d\) dimensions, the squared norm \(\|z\|^2\) follows a chi-squared distribution with \(d\) degrees of freedom:

$$\|z\|^2 \sim \chi^2_d$$

The mean is \(\mathbb{E}[\|z\|^2] = d\) and the variance is \(\text{Var}(\|z\|^2) = 2d\).

Therefore, the norm \(\|z\|\) concentrates around \(\sqrt{d}\):

$$\mathbb{E}[\|z\|] \approx \sqrt{d} - \frac{1}{2\sqrt{d}}, \quad \text{Var}(\|z\|) \approx \frac{1}{2}$$

The standard deviation of \(\|z\|\) is approximately \(\frac{1}{\sqrt{2}}\), **independent of \(d\)**. As \(d\) grows, the ratio of standard deviation to mean shrinks as \(O(1/\sqrt{d})\). In \(d = 512\) dimensions:

$$\|z\| \approx 22.6 \pm 0.71$$

Virtually all the probability mass is concentrated in a thin **spherical shell** of radius \(\approx \sqrt{d}\) and thickness \(\approx O(1)\).

### Why This Breaks LERP

If two points \(z_1\) and \(z_2\) are sampled from \(\mathcal{N}(0, I_{512})\), they both have norm \(\approx 22.6\). But their LERP midpoint has norm:

$$\|z_{\text{mid}}\| = \left\|\frac{z_1 + z_2}{2}\right\|$$

In high dimensions, random vectors are nearly orthogonal (\(z_1 \cdot z_2 \approx 0\)), so:

$$\|z_{\text{mid}}\|^2 \approx \frac{\|z_1\|^2 + \|z_2\|^2}{4} \approx \frac{d}{2}$$

$$\|z_{\text{mid}}\| \approx \sqrt{d/2} \approx \frac{\sqrt{d}}{\sqrt{2}} \approx 16.0$$

The midpoint has norm $16.0$ instead of $22.6$. This is **deep inside the sphere**, in a region that has essentially zero probability under the prior. The decoder has never been trained on points with this norm. The output is often blurry, washed out, or semantically incoherent.

SLERP avoids this entirely by staying on the spherical shell of radius \(\sqrt{d}\).

### Numerical Example

| Dimension \(d\) | Shell radius \(\sqrt{d}\) | LERP midpoint norm | Ratio | Probability of norm < midpoint |
|---|---|---|---|---|
| 2 | 1.41 | 1.00 | 0.707 | ~39% |
| 10 | 3.16 | 2.24 | 0.707 | ~6.8% |
| 100 | 10.0 | 7.07 | 0.707 | ~\(10^{-7}\) |
| 512 | 22.6 | 16.0 | 0.707 | ~\(10^{-35}\) |
| 4096 | 64.0 | 45.3 | 0.707 | ~\(10^{-280}\) |

In 512 dimensions, the probability of a random sample having a norm as small as the LERP midpoint is approximately \(10^{-35}\). The midpoint is in a region of essentially zero probability. This is not a theoretical curiosity -- it directly causes visible artifacts in generated outputs.

---

## Disentangled Representations

A representation is **disentangled** when individual dimensions (or small groups of dimensions) correspond to independent, semantically meaningful factors of variation.

### Formal Definition

Let \(z = (z_1, z_2, \ldots, z_d)\) be a latent representation, and let \(v = (v_1, v_2, \ldots, v_K)\) be the true generative factors (pose, lighting, identity, expression, etc.). A representation is disentangled if:

1. **Each \(z_j\) depends on at most one \(v_k\)**: Changes to \(z_j\) only affect one factor.
2. **Each \(v_k\) is captured by at most a few \(z_j\)'s**: Each factor has a localized representation.
3. **The factors are independent**: \(p(z) = \prod_j p(z_j)\).

In a perfectly disentangled space, moving along dimension \(z_3\) might change only the lighting, while moving along \(z_7\) might change only the facial expression.

### beta-VAE: Pressuring Disentanglement

The beta-VAE modifies the standard VAE objective by weighting the KL term:

$$\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)] - \beta \cdot D_{KL}(q_\phi(z|x) \| p(z))$$

When \(\beta > 1\), the model faces stronger pressure to match the prior \(\mathcal{N}(0, I)\). Since the prior factorizes as \(p(z) = \prod_j p(z_j) = \prod_j \mathcal{N}(0, 1)\), this encourages the aggregate posterior to also factorize, which implies statistical independence between dimensions, which promotes disentanglement.

The tradeoff: higher \(\beta\) means better disentanglement but worse reconstruction. The model cannot use correlated dimensions to encode complex features, so it must sacrifice detail.

The information bottleneck interpretation:

$$I(x; z) \leq \frac{1}{\beta} \left[ \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)] + \text{const} \right]$$

Higher \(\beta\) tightens the bottleneck, forcing the model to encode only the most important, independent factors.

### Metrics for Disentanglement

Several metrics have been proposed:

| Metric | What it measures | Approach |
|---|---|---|
| **DCI** (Disentanglement, Completeness, Informativeness) | Whether each code dimension captures one factor | Train regressors from \(z\) to \(v\) |
| **MIG** (Mutual Information Gap) | Gap in mutual information between top-2 codes per factor | \(\frac{1}{K} \sum_k \frac{1}{H(v_k)} (I_{top1} - I_{top2})\) |
| **SAP** (Separated Attribute Predictability) | Whether factors can be predicted from individual codes | Classification accuracy gap |
| **Factor VAE metric** | Whether individual dimensions encode individual factors | Majority vote classifier |

In practice, disentanglement in video models is partial. Modern latent video diffusion models (like those used in Sora, Veo, Kling) use high-dimensional latent spaces (typically \(d = 4\) to $16$ per spatial position, but with spatial dimensions preserved, so the total latent dimensionality is very high). Full disentanglement is neither achieved nor necessary -- partial disentanglement is sufficient for the arithmetic operations we care about.

---

## Arithmetic in Latent Space

The most striking property of well-structured latent spaces is that **vector arithmetic corresponds to semantic operations**. This was first demonstrated at scale with word2vec, and it applies directly to visual and video latent spaces.

### The word2vec Analogy

In word2vec, word embeddings exhibit linear structure:

$$\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$$

This works because the embedding space encodes a "royalty" direction and a "gender" direction as approximately orthogonal linear subspaces. The vector \(\vec{\text{king}} - \vec{\text{man}}\) isolates the "royalty" component, and adding it to \(\vec{\text{woman}}\) produces the royal female.

### Why Arithmetic Works

Consider a simplified model where the latent space has been trained to encode two independent factors: content \(c\) and style \(s\). If the encoding is approximately linear:

$$z = W_c \cdot c + W_s \cdot s + b$$

Then:

$$z_{\text{content}} - z_{\text{neutral}} + z_{\text{style target}} = W_c \cdot c_{\text{content}} + W_s \cdot s_{\text{neutral}} - W_c \cdot c_{\text{neutral}} - W_s \cdot s_{\text{neutral}} + W_c \cdot c_{\text{style target}} + W_s \cdot s_{\text{style target}}$$

If we choose \(c_{\text{neutral}} = c_{\text{style target}}\) (same content), this simplifies to:

$$= W_c \cdot c_{\text{content}} + W_s \cdot s_{\text{style target}}$$

We get the content of the first input with the style of the second. This is style transfer via vector arithmetic.

### Style Transfer as Vector Arithmetic

The general formula for style transfer in latent space:

$$z_{\text{output}} = z_{\text{content}} + (z_{\text{style reference}} - z_{\text{neutral}})$$

where:
- \(z_{\text{content}}\) is the encoding of the content you want to keep
- \(z_{\text{style reference}}\) is the encoding of something in the target style
- \(z_{\text{neutral}}\) is the encoding of something in a neutral / default style with similar content to the style reference

The difference \((z_{\text{style reference}} - z_{\text{neutral}})\) isolates the **style direction** -- the vector that points from "neutral" to "stylized" while keeping content fixed.

### Attribute Vectors

More generally, we can extract **attribute vectors** by averaging:

$$v_{\text{attribute}} = \frac{1}{|S_+|} \sum_{x \in S_+} E(x) - \frac{1}{|S_-|} \sum_{x \in S_-} E(x)$$

where \(S_+\) is a set of inputs that have the attribute and \(S_-\) is a set that lacks it. The resulting vector \(v_{\text{attribute}}\) points in the direction of "having the attribute."

Examples for video:

| Attribute | \(S_+\) (has attribute) | \(S_-\) (lacks attribute) | Application |
|---|---|---|---|
| "cinematic" | Professional movie clips | Amateur footage | Make any video look cinematic |
| "slow motion" | Slow-mo clips | Normal speed clips | Add slow-mo feel |
| "warm color" | Warm-graded footage | Neutral footage | Color grading in latent space |
| "dynamic camera" | Moving camera shots | Static camera shots | Add camera motion feel |

<svg viewBox="0 0 700 500" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#d4d4d4">Style Arithmetic in Latent Space</text>
  <!-- Axes -->
  <defs>
    <marker id="arrowStyle" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#999"/>
    </marker>
  </defs>
  <line x1="60" y1="440" x2="660" y2="440" stroke="#444" stroke-width="1" marker-end="url(#arrowStyle)"/>
  <line x1="60" y1="440" x2="60" y2="50" stroke="#444" stroke-width="1" marker-end="url(#arrowStyle)"/>
  <text x="660" y="460" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#999">Content axis</text>
  <text x="30" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#999" transform="rotate(-90, 30, 200)">Style axis</text>
  <!-- Grid -->
  <line x1="60" y1="340" x2="650" y2="340" stroke="#222" stroke-width="1"/>
  <line x1="60" y1="240" x2="650" y2="240" stroke="#222" stroke-width="1"/>
  <line x1="60" y1="140" x2="650" y2="140" stroke="#222" stroke-width="1"/>
  <line x1="210" y1="50" x2="210" y2="440" stroke="#222" stroke-width="1"/>
  <line x1="360" y1="50" x2="360" y2="440" stroke="#222" stroke-width="1"/>
  <line x1="510" y1="50" x2="510" y2="440" stroke="#222" stroke-width="1"/>
  <!-- Neutral style, Subject A: z_neutral -->
  <circle cx="200" cy="360" r="12" fill="#2a2a2a" stroke="#999" stroke-width="2"/>
  <text x="200" y="365" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#999">A</text>
  <text x="200" y="395" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#999">z_neutral</text>
  <text x="200" y="410" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#999">(subject A, neutral style)</text>
  <!-- Style reference: z_style -->
  <circle cx="200" cy="140" r="12" fill="#ef5350" stroke="#c62828" stroke-width="2"/>
  <text x="200" y="145" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="white">S</text>
  <text x="200" y="115" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#ef5350">z_style</text>
  <text x="200" y="100" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#ef5350">(subject A, cinematic style)</text>
  <!-- Style vector arrow -->
  <line x1="215" y1="355" x2="215" y2="155" stroke="#ef5350" stroke-width="2.5" marker-end="url(#arrowStyle)" stroke-dasharray="8,4"/>
  <text x="245" y="260" font-family="Arial, sans-serif" font-size="11" fill="#ef5350" font-weight="bold">Style vector</text>
  <text x="245" y="275" font-family="Arial, sans-serif" font-size="10" fill="#ef5350">z_style - z_neutral</text>
  <!-- Content: z_content (subject B, neutral) -->
  <circle cx="500" cy="360" r="12" fill="#4fc3f7" stroke="#0288d1" stroke-width="2"/>
  <text x="500" y="365" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="white">B</text>
  <text x="500" y="395" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#4fc3f7">z_content</text>
  <text x="500" y="410" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#4fc3f7">(subject B, neutral style)</text>
  <!-- Output: z_output (subject B, cinematic) -->
  <circle cx="500" cy="140" r="14" fill="#8bc34a" stroke="#558b2f" stroke-width="2.5"/>
  <text x="500" y="145" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="white" font-weight="bold">B*</text>
  <text x="500" y="115" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#8bc34a" font-weight="bold">z_output</text>
  <text x="500" y="100" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#8bc34a">(subject B, cinematic style)</text>
  <!-- Applied style vector -->
  <line x1="515" y1="355" x2="515" y2="155" stroke="#8bc34a" stroke-width="2.5" marker-end="url(#arrowStyle)"/>
  <text x="545" y="260" font-family="Arial, sans-serif" font-size="11" fill="#8bc34a" font-weight="bold">Same vector</text>
  <text x="545" y="275" font-family="Arial, sans-serif" font-size="10" fill="#8bc34a">applied to B</text>
  <!-- Horizontal connection -->
  <line x1="215" y1="360" x2="485" y2="360" stroke="#999" stroke-width="1" stroke-dasharray="4,4"/>
  <line x1="215" y1="140" x2="485" y2="140" stroke="#999" stroke-width="1" stroke-dasharray="4,4"/>
  <!-- Formula -->
  <rect x="130" y="445" width="440" height="40" rx="5" fill="#1e1e1e" stroke="#444" stroke-width="1"/>
  <text x="350" y="470" text-anchor="middle" font-family="monospace" font-size="13" fill="#d4d4d4">z_output = z_content + (z_style - z_neutral)</text>
</svg>

### Composing Multiple Attributes

Attribute vectors can be composed by addition:

$$z_{\text{output}} = z_{\text{base}} + \alpha_1 v_{\text{cinematic}} + \alpha_2 v_{\text{warm}} + \alpha_3 v_{\text{slow motion}}$$

where \(\alpha_i\) controls the strength of each attribute. This works because:

1. The attribute vectors are approximately orthogonal (different factors of variation)
2. The latent space is approximately linear in the relevant directions
3. The decoder smoothly maps nearby latent codes to nearby outputs

The coefficients \(\alpha_i\) are typically in the range \([-2, 2]\). Values beyond this range push the latent code outside the training distribution and can produce artifacts.

### When Arithmetic Fails

Latent arithmetic is an approximation. It fails when:

1. **The space is not linearly structured.** Deep non-linearities in the encoding can create curved manifolds where linear operations are not meaningful.

2. **The attributes are not independent.** If "cinematic" and "warm lighting" are highly correlated in the training data, their attribute vectors will not be orthogonal, and composing them may double-count shared features.

3. **The operation pushes too far off-manifold.** Large \(\alpha\) values or many composed attributes can push the latent code into regions the decoder has not learned, producing artifacts.

4. **The space has holes.** Real latent spaces are not perfectly smooth. There can be discontinuities or regions where the decoder behaves erratically.

---

## Applications for Video Generation

Latent space arithmetic is not just a mathematical curiosity. It is the foundation of several practical video generation techniques.

### Scene Interpolation

SLERP between the latent codes of two scenes produces smooth transitions:

```python
import numpy as np

def slerp(z1, z2, t):
    """Spherical linear interpolation between z1 and z2."""
    z1_norm = z1 / np.linalg.norm(z1)
    z2_norm = z2 / np.linalg.norm(z2)

    omega = np.arccos(np.clip(np.dot(z1_norm, z2_norm), -1.0, 1.0))

    if omega < 1e-10:
        # Vectors are nearly parallel, fall back to LERP
        return (1 - t) * z1 + t * z2

    sin_omega = np.sin(omega)
    coeff1 = np.sin((1 - t) * omega) / sin_omega
    coeff2 = np.sin(t * omega) / sin_omega

    # Interpolate on the unit sphere, then interpolate the radius
    z_interp = coeff1 * z1_norm + coeff2 * z2_norm

    # Interpolate radii
    r1 = np.linalg.norm(z1)
    r2 = np.linalg.norm(z2)
    r_interp = (1 - t) * r1 + t * r2

    return z_interp * r_interp

# Generate transition frames
num_frames = 30
transition = []
for i in range(num_frames):
    t = i / (num_frames - 1)
    z_frame = slerp(z_scene1, z_scene2, t)
    frame = decoder(z_frame)
    transition.append(frame)
```

For video generation models, scene interpolation in the temporal latent space produces smooth, semantically meaningful transitions that would be impossible to achieve by pixel-level blending.

### Style Mixing

Combine the content of one reference with the style of another:

```python
def style_transfer_latent(z_content, z_style_ref, z_style_neutral, strength=1.0):
    """
    Apply style from z_style_ref to z_content.

    Args:
        z_content: Latent code of the content to preserve
        z_style_ref: Latent code of the style reference
        z_style_neutral: Latent code of a neutral-style version of style_ref's content
        strength: How strongly to apply the style (0.0 = no change, 1.0 = full)

    Returns:
        z_output: Content of z_content with style of z_style_ref
    """
    style_direction = z_style_ref - z_style_neutral
    z_output = z_content + strength * style_direction
    return z_output
```

In practice, the "neutral" reference can be generated by encoding a prompt like "a plain video of [content description]" and the style reference can be generated from "a [style] video of [similar content]."

### Mood Gradients

Smoothly shift the emotional tone of a video by interpolating between mood vectors:

```python
def mood_gradient(z_base, z_happy, z_neutral_happy, z_sad, z_neutral_sad, t):
    """
    Smoothly interpolate mood from happy (t=0) to sad (t=1).

    Uses SLERP on the mood direction to maintain manifold position.
    """
    mood_happy_vec = z_happy - z_neutral_happy
    mood_sad_vec = z_sad - z_neutral_sad

    # SLERP between mood vectors
    mood_vec = slerp(mood_happy_vec, mood_sad_vec, t)

    return z_base + mood_vec
```

This produces a gradient where the content stays fixed but the mood smoothly shifts. This is invaluable for creating emotional arcs in generated video -- a character's scene can gradually shift from hopeful to anxious without any discontinuity.

### Temporal Latent Interpolation

In video diffusion models, each frame (or group of frames) has a latent representation. Interpolation between temporal latent codes produces new in-between frames:

$$z_{\text{frame}_{i+0.5}} = \text{SLERP}(z_{\text{frame}_i}, z_{\text{frame}_{i+1}}, 0.5)$$

This is the basis of **temporal super-resolution** -- generating intermediate frames to increase the frame rate of generated video. SLERP produces more natural intermediate frames than LERP because it preserves the norm structure.

### Multi-Reference Blending

Given \(N\) reference latent codes \(\{z_1, \ldots, z_N\}\) with weights \(\{w_1, \ldots, w_N\}\) where \(\sum_i w_i = 1\), we can blend:

$$z_{\text{blend}} = \sum_{i=1}^{N} w_i z_i$$

For high dimensions, a spherical version uses iterative SLERP:

```python
def multi_slerp(latents, weights):
    """Weighted spherical interpolation of multiple latent codes."""
    assert abs(sum(weights) - 1.0) < 1e-6

    result = latents[0]
    cumulative_weight = weights[0]

    for i in range(1, len(latents)):
        t = weights[i] / (cumulative_weight + weights[i])
        result = slerp(result, latents[i], t)
        cumulative_weight += weights[i]

    return result
```

This enables blending references from multiple source videos with precise control over each contribution.

---

## Practical Recipes

### Recipe 1: Finding Style Directions

To extract a style direction from a trained model:

1. Collect 50-100 images/frames in the target style (\(S_+\))
2. Collect 50-100 images/frames in neutral style (\(S_-\))
3. Encode all images: \(\{E(x) : x \in S_+\}\) and \(\{E(x) : x \in S_-\}\)
4. Compute the mean difference: \(v_{\text{style}} = \frac{1}{|S_+|}\sum_{S_+} E(x) - \frac{1}{|S_-|}\sum_{S_-} E(x)\)
5. Optionally normalize: \(\hat{v}_{\text{style}} = v_{\text{style}} / \|v_{\text{style}}\|\)

The resulting direction can be applied with a scalar multiplier: \(z_{\text{output}} = z_{\text{input}} + \alpha \hat{v}_{\text{style}}\).

### Recipe 2: Smooth Video Transitions

For seamlessly transitioning between two generated videos:

1. Generate video A with latent sequence \(\{z^A_1, \ldots, z^A_T\}\)
2. Generate video B with latent sequence \(\{z^B_1, \ldots, z^B_T\}\)
3. Create transition by SLERP-ing the final frames of A with the first frames of B:

```python
def create_transition(z_A_end, z_B_start, num_transition_frames=15):
    """Create smooth transition between two video latents."""
    frames = []
    for i in range(num_transition_frames):
        t = i / (num_transition_frames - 1)
        # Ease in/out for smoother perceptual transition
        t_eased = 0.5 - 0.5 * np.cos(np.pi * t)
        z_frame = slerp(z_A_end, z_B_start, t_eased)
        frames.append(z_frame)
    return frames
```

The cosine easing function \(t_{\text{eased}} = \frac{1}{2}(1 - \cos(\pi t))\) produces a perceptually smoother transition by moving slowly at the start and end (where changes are most noticeable) and faster in the middle.

### Recipe 3: Attribute Strength Tuning

```python
def tune_attribute(z_base, attribute_vector, strength_range=(-2.0, 2.0), steps=20):
    """
    Generate a grid of outputs with varying attribute strength.
    Useful for finding the right strength for a given attribute.
    """
    results = []
    for i in range(steps):
        alpha = strength_range[0] + (strength_range[1] - strength_range[0]) * i / (steps - 1)
        z_modified = z_base + alpha * attribute_vector
        output = decoder(z_modified)
        results.append((alpha, output))
    return results
```

### Recipe 4: Conditional Generation with Latent Guidance

Rather than using classifier-free guidance in pixel/noise space, you can guide generation in latent space:

$$z_{t-1} = z_{t-1}^{\text{uncond}} + s \cdot (z_{t-1}^{\text{cond}} - z_{t-1}^{\text{uncond}})$$

where \(s\) is the guidance scale. This is mathematically equivalent to classifier-free guidance when the denoising happens in latent space (as in Latent Diffusion Models), and it is the standard approach in modern video generation models.

---

## The Geometry of Video Latent Spaces

Video latent spaces have additional structure beyond image latent spaces because they must encode temporal relationships.

### Temporal Coherence Manifold

In a well-trained video model, the latent codes for consecutive frames lie on a smooth trajectory in latent space. The velocity vector of this trajectory encodes motion:

$$v_t = z_{t+1} - z_t$$

If \(v_t\) is constant, the motion is uniform. If \(v_t\) changes smoothly, the motion accelerates or decelerates naturally. Abrupt changes in \(v_t\) produce jerky motion.

This means the **second derivative** of the latent trajectory matters for video quality:

$$a_t = v_{t+1} - v_t = z_{t+2} - 2z_{t+1} + z_t$$

Small \(\|a_t\|\) implies smooth motion. Video generation models are implicitly trained to produce trajectories with small second derivatives.

### Resolution-Dependent Latent Dimensions

Modern video VAEs (like the 3D VAEs used in Wan, CogVideo, and similar architectures) compress along both spatial and temporal dimensions:

| Compression | Input | Latent | Ratio |
|---|---|---|---|
| Spatial only | \(512 \times 512 \times 3\) | \(64 \times 64 \times 4\) | 48x |
| Spatiotemporal | \(16 \times 512 \times 512 \times 3\) | \(4 \times 64 \times 64 \times 4\) | 192x |

The temporal compression (typically 4x to 8x) means each latent "frame" actually represents multiple output frames. Interpolation in this space produces temporally smooth results because the decoder is trained to generate smooth frame sequences from each latent code.

<svg viewBox="0 0 700 450" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#333">2D Latent Space: Clusters, Interpolation, and Arithmetic</text>
  <!-- Axes -->
  <line x1="60" y1="410" x2="670" y2="410" stroke="#ddd" stroke-width="1"/>
  <line x1="60" y1="410" x2="60" y2="50" stroke="#ddd" stroke-width="1"/>
  <text x="670" y="430" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#bbb">z₁</text>
  <text x="45" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#bbb">z₂</text>
  <!-- Grid -->
  <line x1="60" y1="310" x2="670" y2="310" stroke="#f5f5f5" stroke-width="1"/>
  <line x1="60" y1="210" x2="670" y2="210" stroke="#f5f5f5" stroke-width="1"/>
  <line x1="60" y1="110" x2="670" y2="110" stroke="#f5f5f5" stroke-width="1"/>
  <line x1="210" y1="50" x2="210" y2="410" stroke="#f5f5f5" stroke-width="1"/>
  <line x1="360" y1="50" x2="360" y2="410" stroke="#f5f5f5" stroke-width="1"/>
  <line x1="510" y1="50" x2="510" y2="410" stroke="#f5f5f5" stroke-width="1"/>
  <!-- Cluster: Nature scenes (blue) -->
  <ellipse cx="180" cy="300" rx="70" ry="50" fill="#4fc3f7" fill-opacity="0.12" stroke="#4fc3f7" stroke-width="1.5" stroke-dasharray="4,3"/>
  <circle cx="155" cy="285" r="5" fill="#4fc3f7"/>
  <circle cx="175" cy="310" r="5" fill="#4fc3f7"/>
  <circle cx="200" cy="295" r="5" fill="#4fc3f7"/>
  <circle cx="165" cy="320" r="5" fill="#4fc3f7"/>
  <circle cx="195" cy="280" r="5" fill="#4fc3f7"/>
  <circle cx="210" cy="315" r="5" fill="#4fc3f7"/>
  <circle cx="150" cy="305" r="5" fill="#4fc3f7"/>
  <text x="180" y="365" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#4fc3f7" font-weight="bold">Nature scenes</text>
  <!-- Cluster: Urban scenes (red) -->
  <ellipse cx="520" cy="130" rx="75" ry="45" fill="#ef5350" fill-opacity="0.12" stroke="#ef5350" stroke-width="1.5" stroke-dasharray="4,3"/>
  <circle cx="495" cy="120" r="5" fill="#ef5350"/>
  <circle cx="520" cy="140" r="5" fill="#ef5350"/>
  <circle cx="545" cy="125" r="5" fill="#ef5350"/>
  <circle cx="505" cy="145" r="5" fill="#ef5350"/>
  <circle cx="535" cy="110" r="5" fill="#ef5350"/>
  <circle cx="555" cy="140" r="5" fill="#ef5350"/>
  <circle cx="490" cy="135" r="5" fill="#ef5350"/>
  <text x="520" y="90" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#ef5350" font-weight="bold">Urban scenes</text>
  <!-- Cluster: Portraits (green) -->
  <ellipse cx="480" cy="330" rx="60" ry="50" fill="#8bc34a" fill-opacity="0.12" stroke="#8bc34a" stroke-width="1.5" stroke-dasharray="4,3"/>
  <circle cx="460" cy="320" r="5" fill="#8bc34a"/>
  <circle cx="485" cy="340" r="5" fill="#8bc34a"/>
  <circle cx="500" cy="315" r="5" fill="#8bc34a"/>
  <circle cx="470" cy="345" r="5" fill="#8bc34a"/>
  <circle cx="505" cy="335" r="5" fill="#8bc34a"/>
  <circle cx="465" cy="310" r="5" fill="#8bc34a"/>
  <text x="480" y="395" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#8bc34a" font-weight="bold">Portraits</text>
  <!-- SLERP path (curved) from Nature to Urban -->
  <path d="M 200,295 C 280,180 400,100 520,130" fill="none" stroke="#ff9800" stroke-width="2.5" stroke-dasharray="6,3"/>
  <!-- Interpolation points along path -->
  <circle cx="270" cy="215" r="4" fill="#ff9800"/>
  <circle cx="340" cy="165" r="4" fill="#ff9800"/>
  <circle cx="420" cy="135" r="4" fill="#ff9800"/>
  <text x="310" y="155" font-family="Arial, sans-serif" font-size="10" fill="#ff9800" font-weight="bold">SLERP path</text>
  <!-- Style arithmetic arrow -->
  <defs>
    <marker id="arrowOrange" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#9c27b0"/>
    </marker>
  </defs>
  <!-- Style vector from Nature to Nature+Cinematic -->
  <circle cx="180" cy="190" r="7" fill="#9c27b0" fill-opacity="0.7"/>
  <line x1="180" y1="280" x2="180" y2="200" stroke="#9c27b0" stroke-width="2" marker-end="url(#arrowOrange)"/>
  <text x="145" y="185" font-family="Arial, sans-serif" font-size="10" fill="#9c27b0">+cinematic</text>
  <!-- Same style vector applied to Portrait cluster -->
  <circle cx="480" cy="220" r="7" fill="#9c27b0" fill-opacity="0.7"/>
  <line x1="480" y1="310" x2="480" y2="230" stroke="#9c27b0" stroke-width="2" marker-end="url(#arrowOrange)"/>
  <text x="510" y="218" font-family="Arial, sans-serif" font-size="10" fill="#9c27b0">+cinematic</text>
  <!-- Legend -->
  <rect x="90" y="55" width="180" height="80" rx="5" fill="white" stroke="#eee" stroke-width="1"/>
  <circle cx="105" cy="72" r="5" fill="#ff9800"/>
  <text x="118" y="76" font-family="Arial, sans-serif" font-size="10" fill="#666">SLERP interpolation</text>
  <line x1="100" y1="92" x2="115" y2="92" stroke="#9c27b0" stroke-width="2"/>
  <text x="118" y="96" font-family="Arial, sans-serif" font-size="10" fill="#666">Style arithmetic</text>
  <circle cx="105" cy="112" r="5" fill="#4fc3f7" fill-opacity="0.3" stroke="#4fc3f7"/>
  <text x="118" y="116" font-family="Arial, sans-serif" font-size="10" fill="#666">Semantic clusters</text>
</svg>

---

## Mathematical Comparison: LERP vs SLERP Quality

To quantify the difference, consider the reconstruction quality along an interpolation path. Let \(Q(z)\) be the quality of the decoded output (measured, for instance, by FID against the training distribution). In a well-structured latent space:

$$Q(z) \propto p(z)$$

The quality is proportional to the density of the learned distribution at that point. Points in high-density regions decode well; points in low-density regions decode poorly.

For a standard Gaussian prior in \(d\) dimensions, the log-density is:

$$\log p(z) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\|z\|^2$$

The density depends only on the norm. The maximum of \(p(z) \cdot r^{d-1}\) (the radial density) occurs at \(\|z\| = \sqrt{d-1} \approx \sqrt{d}\).

For a LERP midpoint with \(\|z_{\text{mid}}\| \approx \sqrt{d/2}\), the log-density penalty relative to the shell is:

$$\Delta \log p = -\frac{1}{2}\left(\frac{d}{2}\right) + \frac{1}{2}(d) + (d-1)\log\frac{\sqrt{d/2}}{\sqrt{d}}$$

$$= \frac{d}{4} + (d-1)\log\frac{1}{\sqrt{2}}$$

$$= \frac{d}{4} - \frac{d-1}{2}\log 2$$

For \(d = 512\):

$$\Delta \log p \approx 128 - 177 = -49$$

The density at the LERP midpoint is \(e^{-49} \approx 10^{-21}\) times lower than at the shell. The quality degradation is severe.

SLERP, by staying on the shell, maintains \(\|z\| = \sqrt{d}\) throughout, so \(\Delta \log p = 0\). The quality is uniformly high along the entire interpolation path.

---

## Advanced Topic: Latent Diffusion and the Two-Level Latent Space

Modern video generation models like Stable Video Diffusion, Wan 2.x, and the architectures behind Sora and Veo use a two-level latent space:

1. **VAE latent space**: The 3D VAE compresses video from pixel space to a spatial-temporal latent space (e.g., from \(T \times H \times W \times 3\) to \(T/4 \times H/8 \times W/8 \times 4\)).

2. **Diffusion latent space**: The diffusion process operates within the VAE latent space, starting from pure noise and iteratively denoising to a clean latent code.

Arithmetic in the VAE latent space affects the overall content. Arithmetic during the diffusion process (at intermediate noise levels) affects different aspects depending on the noise level:

| Noise level \(t\) | What arithmetic affects | Analogy |
|---|---|---|
| \(t \approx T\) (high noise) | Global structure, composition, layout | Rough sketch |
| \(t \approx T/2\) (medium noise) | Style, color palette, medium-scale features | Color blocking |
| \(t \approx 0\) (low noise) | Fine details, textures, edges | Finishing touches |

This gives a powerful tool: by performing arithmetic at different noise levels, you can control which aspects of the output are modified.

```python
def arithmetic_at_noise_level(pipe, z_content, z_style, z_neutral,
                                target_noise_level=0.5, strength=1.0):
    """
    Apply style arithmetic at a specific noise level during diffusion.

    target_noise_level: 0.0 = modify fine details, 1.0 = modify global structure
    """
    T = pipe.scheduler.num_timesteps
    target_step = int(target_noise_level * T)

    style_direction = z_style - z_neutral

    # Run diffusion normally until target step
    z_t = pipe.forward_steps(z_content, steps=range(T, target_step, -1))

    # Apply arithmetic at this noise level
    z_t_modified = z_t + strength * style_direction

    # Continue diffusion from the modified state
    z_0 = pipe.forward_steps(z_t_modified, steps=range(target_step, 0, -1))

    return pipe.vae.decode(z_0)
```

---

## Conclusion

Latent space arithmetic is not a trick -- it is a mathematical consequence of how neural networks organize learned representations. The key ideas:

1. **The manifold hypothesis** tells us that high-dimensional data lives on low-dimensional manifolds, making compression meaningful rather than lossy.

2. **VAE training** with KL divergence regularization produces continuous, complete, smooth latent spaces where interpolation and arithmetic are well-defined operations.

3. **SLERP is superior to LERP** in high-dimensional spaces because Gaussian distributions concentrate on thin shells, and LERP cuts through the low-density interior while SLERP stays on the shell.

4. **Disentanglement** means individual directions correspond to semantic attributes, enabling targeted style transfer, mood adjustment, and attribute composition.

5. **Vector arithmetic** (\(z_{\text{output}} = z_{\text{content}} + (z_{\text{style}} - z_{\text{neutral}})\)) is the foundation of style transfer, mood gradients, and multi-reference blending in video generation.

6. **Arithmetic at different noise levels** in diffusion models provides fine-grained control over which aspects of the output are modified.

For builders working with video generation APIs, understanding latent space arithmetic gives you a mental model for why certain techniques work (and why others do not). When you blend two keyframe references and get a smooth transition, that is SLERP on the latent manifold. When you apply a style reference and the content is preserved, that is vector subtraction isolating the style direction. When interpolation produces blurry midpoints, that is the LERP norm collapse in high dimensions.

The math is not just theory. It is the operating system of every video generation model.

---

## References

1. Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. *arXiv:1312.6114*.
2. Higgins, I., et al. (2017). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. *ICLR 2017*.
3. White, T. (2016). Sampling Generative Networks. *arXiv:1609.04468*.
4. Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space. *arXiv:1301.3781*.
5. Shoemake, K. (1985). Animating Rotation with Quaternion Curves. *SIGGRAPH 1985*.
6. Rombach, L., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR 2022*.
7. Blattmann, A., et al. (2023). Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. *arXiv:2311.15127*.
