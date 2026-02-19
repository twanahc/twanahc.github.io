---
layout: post
title: "Noise Schedules, Sampling Strategies, and Distillation: Making Video Diffusion Fast"
date: 2026-02-24
category: math
---

A video diffusion model trained with 1000 denoising steps does not need 1000 steps to generate. The art --- and the theory --- of making diffusion fast lies in three interrelated ideas: choosing the right **noise schedule** (how noise is added and removed), designing efficient **samplers** (how to traverse the denoising trajectory in fewer steps), and **distillation** (training a student model to match the teacher's output in far fewer steps).

This post builds all three from the mathematical foundations. We define the signal-to-noise ratio framework that unifies all noise schedules, derive the DDIM sampler from scratch, explain DPM-Solver's exponential integrator approach, and cover the distillation methods (progressive distillation, consistency models, adversarial distillation) that have reduced video generation from hundreds of steps to 1--4. The stakes are high: a 4-second video at 50 steps might take 30 seconds. At 4 steps, it takes 2.5 seconds. At 1 step, it is real-time.

---

## Table of Contents

1. [The Signal-to-Noise Ratio Framework](#the-signal-to-noise-ratio-framework)
2. [VP, VE, and Sub-VP Schedules](#vp-ve-and-sub-vp-schedules)
3. [Continuous-Time Noise Schedules](#continuous-time-noise-schedules)
4. [The DDIM Sampler](#the-ddim-sampler)
5. [DPM-Solver: Fast ODE Solvers for Diffusion](#dpm-solver-fast-ode-solvers-for-diffusion)
6. [Stochastic vs Deterministic Sampling](#stochastic-vs-deterministic-sampling)
7. [Progressive Distillation](#progressive-distillation)
8. [Consistency Models](#consistency-models)
9. [Adversarial Distillation](#adversarial-distillation)
10. [Video-Specific Challenges](#video-specific-challenges)
11. [Python: Comparing Samplers on a 1D Diffusion Model](#python-comparing-samplers-on-a-1d-diffusion-model)

---

## The Signal-to-Noise Ratio Framework

Every diffusion model defines a **noising process** that adds Gaussian noise to data. At time \(t\), the noisy sample is:

$$\mathbf{x}_t = \alpha_t \, \mathbf{x}_0 + \sigma_t \, \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$$

where \(\alpha_t\) is the signal coefficient and \(\sigma_t\) is the noise coefficient. Different schedules define different functions \(\alpha_t\) and \(\sigma_t\).

The **signal-to-noise ratio (SNR)** at time \(t\) is:

$$\text{SNR}(t) = \frac{\alpha_t^2}{\sigma_t^2}$$

This single quantity captures everything about the noise level. At \(t = 0\) (clean data), \(\text{SNR} = \infty\). At \(t = T\) (pure noise), \(\text{SNR} \approx 0\). The denoising process traverses from low SNR to high SNR.

The **log-SNR** \(\lambda_t = \log \text{SNR}(t) = \log(\alpha_t^2 / \sigma_t^2)\) is the natural parameterization for analysis. It maps the entire noising process to a monotonically decreasing function from \(+\infty\) to \(-\infty\).

A key theorem (Kingma et al., 2021): **the diffusion training objective, weighted appropriately, depends on the noise schedule only through the SNR schedule.** Two schedules with the same SNR(t) function produce the same trained model, even if their \(\alpha_t\) and \(\sigma_t\) differ individually. This is because the model learns to predict the clean signal given a noisy input at a particular SNR, regardless of how that SNR was achieved.

---

## VP, VE, and Sub-VP Schedules

The three classical noise schedule families correspond to different choices of \(\alpha_t\) and \(\sigma_t\):

### Variance-Preserving (VP)

Used by DDPM. The constraint is \(\alpha_t^2 + \sigma_t^2 = 1\), so the noisy sample always has unit variance (assuming the data has unit variance). The DDPM schedule defines:

$$\alpha_t^2 = \bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$$

where \(\beta_t\) is a linearly increasing sequence from \(\beta_1 = 10^{-4}\) to \(\beta_T = 0.02\). Then \(\sigma_t^2 = 1 - \bar{\alpha}_t\).

The SNR decreases from \(\bar{\alpha}_1 / (1 - \bar{\alpha}_1) \approx 10^4\) to \(\bar{\alpha}_T / (1 - \bar{\alpha}_T) \approx 0.006\).

### Variance-Exploding (VE)

Used by SMLD (Song & Ermon, 2019). Here \(\alpha_t = 1\) (signal is unchanged) and noise of increasing variance is added:

$$\mathbf{x}_t = \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}$$

where \(\sigma_t\) increases geometrically from \(\sigma_{\min} \approx 0.01\) to \(\sigma_{\max} \approx 100\). The variance of \(\mathbf{x}_t\) grows without bound (hence "exploding"). SNR \(= 1/\sigma_t^2\).

### Sub-VP

A variant where \(\alpha_t^2 + \sigma_t^2 < 1\) (the variance decreases slightly). This avoids the endpoint problem where VP schedules have \(\sigma_T^2 \approx 1\) but not exactly 1, creating a mismatch between the diffusion endpoint and the prior \(\mathcal{N}(0, I)\).

<svg viewBox="0 0 700 300" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <defs>
    <marker id="arrowSNR" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#d4d4d4"/>
    </marker>
  </defs>
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">Noise Schedules: log-SNR Over Time</text>

  <!-- Axes -->
  <line x1="80" y1="250" x2="650" y2="250" stroke="#d4d4d4" stroke-width="1.5" marker-end="url(#arrowSNR)"/>
  <line x1="80" y1="250" x2="80" y2="40" stroke="#d4d4d4" stroke-width="1.5" marker-end="url(#arrowSNR)"/>
  <text x="370" y="280" text-anchor="middle" font-size="12" fill="#d4d4d4">Time t (0 → T)</text>
  <text x="35" y="145" text-anchor="middle" font-size="12" fill="#d4d4d4" transform="rotate(-90 35 145)">log SNR(t)</text>

  <!-- Zero line -->
  <line x1="80" y1="170" x2="640" y2="170" stroke="#666" stroke-width="0.5" stroke-dasharray="4,4"/>
  <text x="72" y="174" text-anchor="end" font-size="9" fill="#666">0</text>

  <!-- VP schedule (DDPM) -->
  <path d="M 100,60 C 200,65 300,80 400,140 S 530,220 620,240" fill="none" stroke="#4fc3f7" stroke-width="2.5"/>
  <text x="200" y="55" font-size="11" fill="#4fc3f7" font-weight="bold">VP (DDPM)</text>

  <!-- VE schedule -->
  <path d="M 100,70 C 180,72 280,95 380,155 S 520,230 620,245" fill="none" stroke="#E53935" stroke-width="2.5"/>
  <text x="280" y="85" font-size="11" fill="#E53935" font-weight="bold">VE (SMLD)</text>

  <!-- EDM / Cosine schedule -->
  <path d="M 100,55 C 200,75 350,145 450,185 S 570,235 620,242" fill="none" stroke="#66bb6a" stroke-width="2.5"/>
  <text x="430" y="165" font-size="11" fill="#66bb6a" font-weight="bold">EDM (Karras)</text>

  <!-- Annotations -->
  <text x="105" y="50" font-size="9" fill="#999">Clean</text>
  <text x="610" y="260" font-size="9" fill="#999">Noise</text>
</svg>

---

## Continuous-Time Noise Schedules

The discrete-time schedules above can be unified in a continuous-time framework. Define the noise schedule as a continuous function \(\text{SNR}(t)\) for \(t \in [0, 1]\), with \(\text{SNR}(0) = \text{SNR}_{\max}\) and \(\text{SNR}(1) = \text{SNR}_{\min}\).

### The EDM Parameterization (Karras et al., 2022)

EDM defines the noise schedule directly in terms of \(\sigma(t)\):

$$\sigma(t) = \left(\sigma_{\min}^{1/\rho} + t \cdot (\sigma_{\max}^{1/\rho} - \sigma_{\min}^{1/\rho})\right)^\rho$$

with \(\rho = 7\), \(\sigma_{\min} = 0.002\), \(\sigma_{\max} = 80\). The parameter \(\rho\) controls how time is distributed across noise levels. With \(\rho = 7\), more timesteps are allocated to intermediate noise levels (where the denoising is most challenging) rather than the extreme ends.

EDM also sets \(\alpha_t = 1\) (VE-like) and focuses the entire design on the sigma schedule. The corresponding SNR is \(1/\sigma(t)^2\).

### Optimal Time Discretization

When sampling with \(N\) steps, the timesteps \(t_1 > t_2 > \cdots > t_N\) should be chosen to equalize the "difficulty" of each step. Karras et al. show that spacing the \(\sigma\) values equally in \(\sigma^{1/\rho}\) (which corresponds to equal steps in the EDM parameterization) is near-optimal. This places more steps at medium noise levels and fewer at the extremes.

---

## The DDIM Sampler

The **Denoising Diffusion Implicit Model (DDIM)** (Song et al., 2020) is the key to fast sampling. It defines a **deterministic** generative process that shares the same marginal distributions as DDPM but requires far fewer steps.

### Derivation

DDPM defines a Markovian forward process:

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} \, \mathbf{x}_{t-1}, \beta_t I)$$

DDIM observes that the marginals \(q(\mathbf{x}_t | \mathbf{x}_0)\) are:

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} \, \mathbf{x}_0, (1 - \bar{\alpha}_t) I)$$

There are **many** different forward processes that produce these same marginals. DDIM constructs a **non-Markovian** forward process parameterized by \(\eta \geq 0\):

$$q_\eta(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\!\left(\sqrt{\bar{\alpha}_{t-1}} \, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \eta^2 \beta_t} \cdot \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0}{\sqrt{1 - \bar{\alpha}_t}}, \, \eta^2 \beta_t I\right)$$

Let us unpack this. The mean of the reverse step is a weighted combination of:
1. The predicted clean image \(\mathbf{x}_0\) (scaled by \(\sqrt{\bar{\alpha}_{t-1}}\))
2. The "direction pointing to \(\mathbf{x}_t\)" --- the noise component rescaled from noise level \(t\) to noise level \(t-1\)

The variance is \(\eta^2 \beta_t\).

**When \(\eta = 1\):** This reduces to DDPM (full stochasticity).

**When \(\eta = 0\):** The variance is zero --- the process is **deterministic**. Given \(\mathbf{x}_t\) and the model's prediction of \(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\), the DDIM update is:

$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left(\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t} \, \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{predicted } \mathbf{x}_0} + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$$

This is a deterministic map from \(\mathbf{x}_t\) to \(\mathbf{x}_{t-1}\). The same initial noise always produces the same output. And because it is deterministic, you can skip steps --- go directly from \(\mathbf{x}_{1000}\) to \(\mathbf{x}_{950}\) to \(\mathbf{x}_{900}\) etc., using any subsequence of timesteps.

### Why DDIM Can Skip Steps

DDPM's stochasticity requires small steps (each step adds noise that the next step must remove). DDIM's deterministic steps only need to track a smooth ODE trajectory. If the trajectory is not too curved, large steps are accurate. In practice, DDIM with 50 steps matches DDPM with 1000 steps.

---

## DPM-Solver: Fast ODE Solvers for Diffusion

DDIM is a first-order ODE solver (Euler method). **DPM-Solver** (Lu et al., 2022) achieves the same quality in fewer steps by using higher-order methods.

### The Probability Flow ODE

Any diffusion process with score function \(\nabla_{\mathbf{x}} \log p_t(\mathbf{x})\) has an equivalent deterministic ODE (the probability flow ODE):

$$\frac{d\mathbf{x}}{dt} = f(\mathbf{x}, t) - \frac{1}{2}g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$

For the VP schedule, this simplifies to:

$$\frac{d\mathbf{x}}{d\lambda} = \frac{\sigma_\lambda}{2} \left(\mathbf{x} - \hat{\mathbf{x}}_0(\mathbf{x}, \lambda)\right)$$

where \(\lambda = \log(\alpha/\sigma)\) is the log-SNR and \(\hat{\mathbf{x}}_0\) is the denoised prediction.

### Exponential Integrator

DPM-Solver writes the ODE in a form where the linear part can be integrated exactly. The key change of variables: let \(\hat{\boldsymbol{\epsilon}}_\theta\) be the noise prediction. Then the exact solution from time \(s\) to time \(t\) (in log-SNR parameterization) is:

$$\mathbf{x}_t = \frac{\alpha_t}{\alpha_s} \mathbf{x}_s - \sigma_t \int_{\lambda_s}^{\lambda_t} e^{-\lambda} \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_\lambda, \lambda) \, d\lambda$$

DPM-Solver approximates the integral by Taylor-expanding \(\hat{\boldsymbol{\epsilon}}_\theta\) around \(\lambda_s\):

**First order (DPM-Solver-1, equivalent to DDIM):**

$$\mathbf{x}_t \approx \frac{\alpha_t}{\alpha_s} \mathbf{x}_s - \sigma_t (e^{h} - 1) \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_s, \lambda_s)$$

where \(h = \lambda_t - \lambda_s\).

**Second order (DPM-Solver-2):** Evaluate \(\hat{\boldsymbol{\epsilon}}_\theta\) at a midpoint to get a better approximation:

$$\mathbf{x}_t \approx \frac{\alpha_t}{\alpha_s} \mathbf{x}_s - \sigma_t (e^h - 1) \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_s, \lambda_s) - \frac{\sigma_t}{2h}(e^h - 1 - h)(\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_m, \lambda_m) - \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_s, \lambda_s))$$

Each order increase reduces the number of steps needed by roughly 2×. DPM-Solver-2 with 15--20 steps matches DDIM with 50+ steps.

---

## Stochastic vs Deterministic Sampling

ODE-based samplers (DDIM, DPM-Solver) are deterministic: the same noise input always produces the same output. SDE-based samplers (DDPM, ancestral sampling) add noise at each step.

**When is stochasticity helpful?**

- **Diversity:** Stochastic samplers explore more of the distribution, producing more diverse outputs from the same starting noise.
- **Error correction:** Added noise can "wash out" accumulated errors from imperfect denoising, improving robustness.
- **Quality at low steps:** At very few steps (1--4), stochastic samplers often produce artifacts because the noise injection is too large relative to the signal evolution.

**The \(\eta\) parameter** (from DDIM) and **churn** (from EDM) interpolate between deterministic and stochastic. For video, deterministic sampling is generally preferred because stochastic noise injection at each step can break temporal consistency.

---

## Progressive Distillation

**Progressive distillation** (Salimans & Ho, 2022) trains a student model that matches the teacher's output in half the number of steps, then repeats.

### The Algorithm

1. Start with a teacher model \(\boldsymbol{\epsilon}_\text{teacher}\) that generates in \(N\) steps.
2. Train a student \(\boldsymbol{\epsilon}_\text{student}\) that takes one step to match the teacher's two steps:

$$\mathcal{L} = \mathbb{E}\!\left[\left\| \boldsymbol{\epsilon}_\text{student}(\mathbf{x}_{t}, t \to t-2) - \hat{\boldsymbol{\epsilon}}_{\text{target}} \right\|^2 \right]$$

where \(\hat{\boldsymbol{\epsilon}}_{\text{target}}\) is computed by running the teacher for two steps from \(\mathbf{x}_t\).

3. The student now generates in \(N/2\) steps. Use the student as the new teacher and repeat.

After \(k\) rounds, the model generates in \(N / 2^k\) steps. Starting from \(N = 1024\):
- Round 1: 512 steps
- Round 2: 256 steps
- ...
- Round 8: 4 steps
- Round 9: 2 steps
- Round 10: 1 step

### v-Prediction

Progressive distillation works best with the **v-prediction** parameterization:

$$\mathbf{v}_t = \alpha_t \boldsymbol{\epsilon} - \sigma_t \mathbf{x}_0$$

which is a rotation of the \((\mathbf{x}_0, \boldsymbol{\epsilon})\) pair. The v-prediction is numerically better behaved than \(\epsilon\)-prediction at high and low noise levels, which matters when distillation pushes the model to make large jumps in noise space.

---

## Consistency Models

**Consistency models** (Song et al., 2023) take a different approach: instead of iterative distillation, train the model to satisfy a **self-consistency** property.

### The Key Idea

Consider the probability flow ODE trajectory starting from any noisy point \(\mathbf{x}_t\). All points on this trajectory map to the same clean output \(\mathbf{x}_0\) when the ODE is solved to completion. A **consistency function** \(f_\theta(\mathbf{x}_t, t)\) maps any point on the trajectory directly to the endpoint:

$$f_\theta(\mathbf{x}_t, t) = \mathbf{x}_0 \quad \text{for all } t \text{ along the same ODE trajectory}$$

The self-consistency property is:

$$f_\theta(\mathbf{x}_t, t) = f_\theta(\mathbf{x}_{t'}, t') \quad \text{for all } t, t' \text{ on the same trajectory}$$

### Boundary Condition

At \(t = \epsilon\) (near-clean data), the consistency function must be the identity:

$$f_\theta(\mathbf{x}_\epsilon, \epsilon) = \mathbf{x}_\epsilon$$

This is enforced architecturally: parameterize \(f_\theta(\mathbf{x}_t, t) = c_\text{skip}(t) \mathbf{x}_t + c_\text{out}(t) F_\theta(\mathbf{x}_t, t)\) where \(c_\text{skip}(\epsilon) = 1\) and \(c_\text{out}(\epsilon) = 0\).

### Training

**Consistency distillation:** Given a pretrained diffusion model, enforce consistency between adjacent time steps:

$$\mathcal{L} = \mathbb{E}\!\left[\| f_\theta(\mathbf{x}_{t_{n+1}}, t_{n+1}) - f_{\theta^-}(\hat{\mathbf{x}}_{t_n}, t_n) \|^2 \right]$$

where \(\hat{\mathbf{x}}_{t_n}\) is obtained by one ODE step from \(\mathbf{x}_{t_{n+1}}\) using the teacher, and \(\theta^-\) is an EMA of \(\theta\).

**Consistency training:** Train from scratch without a teacher, using the data itself to define the ODE trajectories. This avoids needing a pretrained model but is harder to train.

### Sampling

A trained consistency model generates in **one step**: sample \(\mathbf{x}_T \sim \mathcal{N}(0, I)\) and compute \(f_\theta(\mathbf{x}_T, T)\). For better quality, use 2--4 steps with a "denoise-then-noise-then-denoise" strategy.

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">Consistency Model: All Points Map to x₀</text>

  <!-- Axes -->
  <line x1="60" y1="240" x2="650" y2="240" stroke="#666" stroke-width="1"/>
  <text x="350" y="265" text-anchor="middle" font-size="11" fill="#999">Data space</text>

  <!-- ODE trajectory -->
  <path d="M 150,230 C 200,180 280,120 350,90 S 480,60 580,50" fill="none" stroke="#4fc3f7" stroke-width="2.5"/>
  <text x="590" y="45" font-size="10" fill="#4fc3f7">x_T (noise)</text>
  <text x="140" y="245" font-size="10" fill="#4fc3f7">x_0 (clean)</text>

  <!-- Points on trajectory mapping to x_0 -->
  <circle cx="580" cy="50" r="5" fill="#E53935"/>
  <circle cx="450" cy="70" r="5" fill="#FF9800"/>
  <circle cx="350" cy="90" r="5" fill="#66bb6a"/>
  <circle cx="250" cy="150" r="5" fill="#CE93D8"/>
  <circle cx="150" cy="230" r="7" fill="#4fc3f7"/>

  <!-- Arrows to x_0 -->
  <line x1="575" y1="55" x2="158" y2="225" stroke="#E53935" stroke-width="1" stroke-dasharray="4,3"/>
  <line x1="445" y1="74" x2="155" y2="226" stroke="#FF9800" stroke-width="1" stroke-dasharray="4,3"/>
  <line x1="345" y1="94" x2="155" y2="226" stroke="#66bb6a" stroke-width="1" stroke-dasharray="4,3"/>
  <line x1="247" y1="154" x2="153" y2="226" stroke="#CE93D8" stroke-width="1" stroke-dasharray="4,3"/>

  <!-- Label -->
  <text x="350" y="180" text-anchor="middle" font-size="11" fill="#d4d4d4">f(x_t, t) = x₀ for all t on the trajectory</text>
  <text x="350" y="200" text-anchor="middle" font-size="10" fill="#999">One model evaluation → direct jump to clean data</text>
</svg>

---

## Adversarial Distillation

The latest distillation methods use a **discriminator** to match distributions in 1--4 steps, bypassing the slow iterative distillation process.

### SDXL-Turbo / ADD (Adversarial Diffusion Distillation)

Train a student generator \(G_\theta\) with two losses:
1. **Diffusion loss:** The student's output, when noised to an intermediate level, should fool the teacher's denoiser (the student's generation should look like a plausible intermediate state of the teacher's denoising process).
2. **Adversarial loss:** A discriminator \(D_\phi\) is trained to distinguish real images from the student's 1--4 step outputs.

$$\mathcal{L}_\text{student} = \lambda_\text{diff} \mathcal{L}_\text{diffusion} + \lambda_\text{adv} \mathcal{L}_\text{adversarial}$$

The adversarial loss is crucial: it ensures the student's outputs lie on the manifold of natural images, even when the diffusion loss alone would produce blurry averages.

### Distribution Matching Distillation (DMD2)

Instead of training a discriminator from scratch, use the pretrained diffusion model itself as the discriminator. The key insight: the score function \(\nabla_\mathbf{x} \log p(\mathbf{x})\) can distinguish real from fake images (real images have high log-likelihood; fake images do not). The loss pushes the student's generated distribution toward the teacher's.

### LADD (Latent Adversarial Diffusion Distillation)

Apply adversarial distillation in the latent space of the VAE, which is lower-dimensional and smoother. The discriminator operates on latent features rather than pixels, which is more computationally efficient and produces more stable training.

These methods achieve 1--4 step generation with quality approaching 50-step sampling, enabling near-real-time video generation.

---

## Video-Specific Challenges

Video diffusion has unique challenges that affect noise schedule and sampler design:

**Temporal coherence requires more steps.** A single image needs only spatial consistency. Video needs temporal consistency across frames. Each denoising step must produce frames that are not only individually clean but also coherent with their neighbors. Aggressive step reduction (below ~8 steps) can produce temporally flickering video even when individual frames look fine.

**Joint noise schedules.** Some architectures add different amounts of noise to spatial and temporal dimensions. For example, add full noise spatially but less noise temporally, allowing the model to maintain temporal structure even at high noise levels. This is equivalent to a 4D noise schedule that varies across (x, y, t, channel).

**Cascaded generation.** Many video models generate at low resolution first (fast) and then upsample (using a separate super-resolution diffusion model). Each stage has its own noise schedule and sampler. The base model might use 50 steps; the SR model might use 10.

**Sliding window / autoregressive.** Long videos are generated chunk by chunk, with overlapping frames for consistency. Each chunk undergoes full denoising, but the overlapping frames are initialized from the previous chunk's output rather than pure noise, requiring a modified noise schedule that starts from a partially-noised state.

---

## Python: Comparing Samplers on a 1D Diffusion Model

```python
import numpy as np
import matplotlib.pyplot as plt

# Target distribution: mixture of Gaussians
def target_score(x, t):
    """Score function of a noised mixture of Gaussians."""
    sigma2 = t  # noise variance at time t
    mu1, mu2 = -2.0, 2.0
    w1, w2 = 0.4, 0.6

    # p(x|t) = w1 * N(mu1, 1+sigma2) + w2 * N(mu2, 1+sigma2)
    var = 1.0 + sigma2
    p1 = w1 * np.exp(-0.5 * (x - mu1)**2 / var) / np.sqrt(2 * np.pi * var)
    p2 = w2 * np.exp(-0.5 * (x - mu2)**2 / var) / np.sqrt(2 * np.pi * var)
    p = p1 + p2 + 1e-30

    # Score = d/dx log p
    dp1 = p1 * (-(x - mu1) / var)
    dp2 = p2 * (-(x - mu2) / var)
    return (dp1 + dp2) / p

def eps_prediction(x, t, alpha_t, sigma_t):
    """Convert score to epsilon prediction."""
    score = target_score(x, t)
    return -sigma_t * score

# VP noise schedule
def vp_schedule(t, T=1.0, beta_min=0.1, beta_max=20.0):
    """Continuous VP schedule."""
    beta_t = beta_min + t * (beta_max - beta_min)
    log_alpha2 = -(beta_min * t + 0.5 * (beta_max - beta_min) * t**2)
    alpha_t = np.exp(0.5 * log_alpha2)
    sigma_t = np.sqrt(1 - np.exp(log_alpha2))
    return alpha_t, sigma_t

# DDPM sampler (stochastic)
def sample_ddpm(n_samples, n_steps):
    ts = np.linspace(1.0, 0.001, n_steps + 1)
    x = np.random.randn(n_samples)  # start from noise

    for i in range(n_steps):
        t = ts[i]
        t_prev = ts[i + 1]
        alpha_t, sigma_t = vp_schedule(t)
        alpha_prev, sigma_prev = vp_schedule(t_prev)

        eps = eps_prediction(x, sigma_t**2, alpha_t, sigma_t)

        # Predict x0
        x0_pred = (x - sigma_t * eps) / alpha_t

        # DDPM update
        mean = alpha_prev * x0_pred + sigma_prev * np.sqrt(1 - (sigma_prev/sigma_t)**2 * 0.5) * eps
        if i < n_steps - 1:
            noise = np.random.randn(n_samples)
            std = np.sqrt(sigma_prev**2 - sigma_prev**2 * (alpha_prev * sigma_t / (alpha_t * sigma_prev))**2 * 0.1)
            x = mean + std * noise
        else:
            x = mean
    return x

# DDIM sampler (deterministic)
def sample_ddim(n_samples, n_steps):
    ts = np.linspace(1.0, 0.001, n_steps + 1)
    x = np.random.randn(n_samples)

    for i in range(n_steps):
        t = ts[i]
        t_prev = ts[i + 1]
        alpha_t, sigma_t = vp_schedule(t)
        alpha_prev, sigma_prev = vp_schedule(t_prev)

        eps = eps_prediction(x, sigma_t**2, alpha_t, sigma_t)

        # Predict x0
        x0_pred = (x - sigma_t * eps) / alpha_t

        # DDIM update (deterministic, eta=0)
        x = alpha_prev * x0_pred + sigma_prev * eps

    return x

# Compare samplers at different step counts
step_counts = [5, 10, 25, 50, 100, 500]
n_samples = 5000

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
x_range = np.linspace(-6, 6, 300)

# True target density
var_true = 1.0
p_true = (0.4 * np.exp(-0.5 * (x_range + 2)**2 / var_true) / np.sqrt(2*np.pi*var_true) +
          0.6 * np.exp(-0.5 * (x_range - 2)**2 / var_true) / np.sqrt(2*np.pi*var_true))

for idx, n_steps in enumerate(step_counts):
    ax = axes[idx // 3, idx % 3]

    samples_ddim = sample_ddim(n_samples, n_steps)
    samples_ddpm = sample_ddpm(n_samples, n_steps)

    ax.plot(x_range, p_true, 'k-', linewidth=2, label='Target', alpha=0.7)
    ax.hist(samples_ddim, bins=80, range=(-6, 6), density=True,
            alpha=0.4, color='#4fc3f7', label='DDIM')
    ax.hist(samples_ddpm, bins=80, range=(-6, 6), density=True,
            alpha=0.4, color='#E53935', label='DDPM')

    ax.set_title(f'{n_steps} steps')
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel('Density')

plt.suptitle('DDIM vs DDPM: Convergence with Number of Steps', fontsize=14)
plt.tight_layout()
plt.savefig('sampler_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

The practical message is clear: the mathematical structure of the noise schedule and sampler matters enormously for video generation. A good schedule allocates denoising effort where it is most needed (medium noise levels). A good sampler (DPM-Solver-2+, DDIM) traverses the ODE trajectory efficiently. And distillation (consistency models, adversarial distillation) collapses the entire trajectory into 1--4 steps. Together, these tools have reduced video generation latency from minutes to seconds --- the difference between a research demo and a product.
