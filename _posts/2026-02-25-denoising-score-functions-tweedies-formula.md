---
layout: post
title: "Denoising, Score Functions, and Tweedie's Formula: The Statistical Heart of Diffusion Models"
date: 2026-02-25
category: math
---

At the center of every diffusion model is a neural network that takes a noisy input and predicts what was there before the noise was added. This sounds like a straightforward regression problem, but it conceals a deep mathematical structure. The optimal denoiser is not just a good guess at the clean data --- it is intimately connected to the **score function** (the gradient of the log probability density), to **Tweedie's formula** from Bayesian statistics, and to the entire theoretical apparatus of score-based generative modeling.

Understanding this connection is not merely academic. It explains why diffusion models work at all, why different model parameterizations (\(\epsilon\)-prediction, \(\mathbf{x}_0\)-prediction, v-prediction) are equivalent, and why the denoiser's behavior at different noise levels reveals the structure of the data distribution at different scales.

This post derives everything from first principles. We define the score function and explain its geometric meaning, derive the original score matching objective and the integration-by-parts trick, prove the denoising score matching theorem, derive Tweedie's formula from Bayes' theorem, and connect all of this to the practical training of video diffusion models.

---

## Table of Contents

1. [The Score Function](#the-score-function)
2. [Score Matching: The Original Objective](#score-matching-the-original-objective)
3. [Denoising Score Matching](#denoising-score-matching)
4. [Tweedie's Formula](#tweedies-formula)
5. [The Denoising Autoencoder Connection](#the-denoising-autoencoder-connection)
6. [Noise-Conditioned Score Networks](#noise-conditioned-score-networks)
7. [From Score Matching to Diffusion](#from-score-matching-to-diffusion)
8. [Epsilon, Score, x₀, and v-Prediction](#epsilon-score-x0-and-v-prediction)
9. [The Optimal Denoiser for Video](#the-optimal-denoiser-for-video)
10. [Tweedie in Practice](#tweedie-in-practice)
11. [Python: Score Estimation and Langevin Dynamics](#python-score-estimation-and-langevin-dynamics)

---

## The Score Function

Let \(p(\mathbf{x})\) be the probability density of the data distribution over \(\mathbf{x} \in \mathbb{R}^d\). The **score function** is the gradient of the log-density:

$$\mathbf{s}(\mathbf{x}) = \nabla_\mathbf{x} \log p(\mathbf{x}) = \frac{\nabla_\mathbf{x} \, p(\mathbf{x})}{p(\mathbf{x})}$$

The score is a vector field over data space. At each point \(\mathbf{x}\), it points in the direction of **steepest increase** of the log-density --- toward regions of higher probability.

### Geometric Interpretation

Think of \(\log p(\mathbf{x})\) as a landscape. Peaks correspond to modes of the distribution (the most likely data points). The score at any point is the uphill direction on this landscape. If you follow the score from any starting point, you climb toward a mode.

This is exactly **gradient ascent on the log-density**, which is the continuous-time version of a hill-climbing algorithm for finding high-probability regions.

### Score of a Gaussian

For a Gaussian \(p(\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \Sigma)\), the log-density is:

$$\log p(\mathbf{x}) = -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) + \text{const.}$$

The score is:

$$\nabla_\mathbf{x} \log p(\mathbf{x}) = -\Sigma^{-1}(\mathbf{x} - \boldsymbol{\mu})$$

This points from \(\mathbf{x}\) toward the mean \(\boldsymbol{\mu}\), with magnitude proportional to the distance from the mean (weighted by the inverse covariance). Points far from the mean have large scores --- strong "pull" back toward the center.

For the special case of an isotropic Gaussian \(\mathcal{N}(\boldsymbol{\mu}, \sigma^2 I)\):

$$\nabla_\mathbf{x} \log p(\mathbf{x}) = -\frac{\mathbf{x} - \boldsymbol{\mu}}{\sigma^2}$$

The score is simply the negative displacement from the mean, scaled by \(1/\sigma^2\).

### Why the Score?

The score avoids the **normalizing constant problem**. The density \(p(\mathbf{x})\) requires computing the partition function \(Z = \int p^*(\mathbf{x}) d\mathbf{x}\) (where \(p^* = p \cdot Z\) is the unnormalized density), which is intractable in high dimensions. But the score:

$$\nabla_\mathbf{x} \log p(\mathbf{x}) = \nabla_\mathbf{x} \log p^*(\mathbf{x}) - \nabla_\mathbf{x} \log Z = \nabla_\mathbf{x} \log p^*(\mathbf{x})$$

The normalizing constant \(Z\) is a constant with respect to \(\mathbf{x}\), so its gradient is zero. The score depends only on the unnormalized density. This is what makes score-based modeling tractable.

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <defs>
    <marker id="arrowScore" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#4fc3f7"/>
    </marker>
  </defs>
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">Score Field: Arrows Point Toward High Density</text>

  <!-- Density contours (two modes) -->
  <ellipse cx="220" cy="150" rx="80" ry="60" fill="#4fc3f7" opacity="0.05" stroke="#4fc3f7" stroke-width="0.5"/>
  <ellipse cx="220" cy="150" rx="55" ry="40" fill="#4fc3f7" opacity="0.08" stroke="#4fc3f7" stroke-width="0.5"/>
  <ellipse cx="220" cy="150" rx="30" ry="20" fill="#4fc3f7" opacity="0.12" stroke="#4fc3f7" stroke-width="0.5"/>
  <circle cx="220" cy="150" r="4" fill="#4fc3f7"/>

  <ellipse cx="480" cy="140" rx="90" ry="70" fill="#66bb6a" opacity="0.05" stroke="#66bb6a" stroke-width="0.5"/>
  <ellipse cx="480" cy="140" rx="60" ry="45" fill="#66bb6a" opacity="0.08" stroke="#66bb6a" stroke-width="0.5"/>
  <ellipse cx="480" cy="140" rx="30" ry="22" fill="#66bb6a" opacity="0.12" stroke="#66bb6a" stroke-width="0.5"/>
  <circle cx="480" cy="140" r="4" fill="#66bb6a"/>

  <!-- Score arrows pointing inward -->
  <line x1="110" y1="150" x2="145" y2="150" stroke="#4fc3f7" stroke-width="1.5" marker-end="url(#arrowScore)"/>
  <line x1="135" y1="90" x2="160" y2="115" stroke="#4fc3f7" stroke-width="1.5" marker-end="url(#arrowScore)"/>
  <line x1="135" y1="210" x2="160" y2="190" stroke="#4fc3f7" stroke-width="1.5" marker-end="url(#arrowScore)"/>
  <line x1="310" y1="130" x2="280" y2="140" stroke="#4fc3f7" stroke-width="1.5" marker-end="url(#arrowScore)"/>

  <line x1="600" y1="140" x2="560" y2="140" stroke="#66bb6a" stroke-width="1.5" marker-end="url(#arrowScore)"/>
  <line x1="580" y1="80" x2="545" y2="105" stroke="#66bb6a" stroke-width="1.5" marker-end="url(#arrowScore)"/>
  <line x1="580" y1="200" x2="545" y2="180" stroke="#66bb6a" stroke-width="1.5" marker-end="url(#arrowScore)"/>
  <line x1="380" y1="130" x2="415" y2="135" stroke="#66bb6a" stroke-width="1.5" marker-end="url(#arrowScore)"/>

  <!-- Between modes -->
  <line x1="350" y1="145" x2="370" y2="143" stroke="#FF9800" stroke-width="1.2" marker-end="url(#arrowScore)"/>

  <text x="350" y="260" text-anchor="middle" font-size="11" fill="#d4d4d4">∇ log p(x) points toward modes of the distribution</text>
</svg>

---

## Score Matching: The Original Objective

To learn the score function from data, we need an objective. The **Fisher divergence** between the model score \(\mathbf{s}_\theta(\mathbf{x})\) and the true score \(\nabla_\mathbf{x} \log p(\mathbf{x})\) is:

$$J(\theta) = \frac{1}{2} \mathbb{E}_{p(\mathbf{x})}\!\left[\|\mathbf{s}_\theta(\mathbf{x}) - \nabla_\mathbf{x} \log p(\mathbf{x})\|^2\right]$$

This is the ideal loss, but it requires the unknown true score. Hyvarinen (2005) showed that this can be rewritten in a form that does not require the true score.

### The Integration-by-Parts Trick

Expand the squared norm:

$$J(\theta) = \frac{1}{2}\mathbb{E}\!\left[\|\mathbf{s}_\theta\|^2\right] - \mathbb{E}\!\left[\mathbf{s}_\theta \cdot \nabla \log p\right] + \frac{1}{2}\mathbb{E}\!\left[\|\nabla \log p\|^2\right]$$

The third term is a constant (independent of \(\theta\)). The key is the second term:

$$\mathbb{E}_{p(\mathbf{x})}\!\left[\mathbf{s}_\theta(\mathbf{x}) \cdot \nabla_\mathbf{x} \log p(\mathbf{x})\right] = \int p(\mathbf{x}) \, \mathbf{s}_\theta(\mathbf{x}) \cdot \frac{\nabla p(\mathbf{x})}{p(\mathbf{x})} \, d\mathbf{x} = \int \mathbf{s}_\theta(\mathbf{x}) \cdot \nabla p(\mathbf{x}) \, d\mathbf{x}$$

Integrating by parts (with the boundary term vanishing for well-behaved densities):

$$\int \mathbf{s}_\theta \cdot \nabla p \, d\mathbf{x} = -\int p \, \nabla \cdot \mathbf{s}_\theta \, d\mathbf{x} = -\mathbb{E}\!\left[\nabla \cdot \mathbf{s}_\theta(\mathbf{x})\right]$$

where \(\nabla \cdot \mathbf{s}_\theta = \sum_i \partial s_{\theta,i} / \partial x_i\) is the divergence (trace of the Jacobian) of the model score.

Putting it together and dropping the constant:

$$J(\theta) \propto \mathbb{E}_{p(\mathbf{x})}\!\left[\frac{1}{2}\|\mathbf{s}_\theta(\mathbf{x})\|^2 + \nabla \cdot \mathbf{s}_\theta(\mathbf{x})\right]$$

This is **explicit score matching**: it requires only samples from \(p\) (not the density itself), but it requires computing the divergence of the model, which involves second derivatives of the network --- computationally expensive.

---

## Denoising Score Matching

**Denoising score matching (DSM)** (Vincent, 2011) avoids the divergence computation entirely by connecting score matching to denoising.

### The Setup

Instead of matching the score of the data distribution \(p(\mathbf{x})\), match the score of a **noised** data distribution:

$$p_\sigma(\tilde{\mathbf{x}}) = \int p(\mathbf{x}) \, q_\sigma(\tilde{\mathbf{x}} | \mathbf{x}) \, d\mathbf{x}$$

where \(q_\sigma(\tilde{\mathbf{x}} | \mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^2 I)\) is Gaussian noise. The noisy observation is \(\tilde{\mathbf{x}} = \mathbf{x} + \sigma \boldsymbol{\epsilon}\).

### The Key Theorem

**Theorem (Vincent, 2011).** The Fisher divergence with respect to the noised distribution:

$$\frac{1}{2}\mathbb{E}_{p_\sigma(\tilde{\mathbf{x}})}\!\left[\|\mathbf{s}_\theta(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}})\|^2\right]$$

equals (up to a constant independent of \(\theta\)):

$$\frac{1}{2}\mathbb{E}_{p(\mathbf{x})} \mathbb{E}_{q_\sigma(\tilde{\mathbf{x}}|\mathbf{x})}\!\left[\|\mathbf{s}_\theta(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} | \mathbf{x})\|^2\right]$$

### Proof

Write the Fisher divergence:

$$J_\sigma = \frac{1}{2}\int p_\sigma(\tilde{\mathbf{x}}) \|\mathbf{s}_\theta(\tilde{\mathbf{x}}) - \nabla \log p_\sigma(\tilde{\mathbf{x}})\|^2 d\tilde{\mathbf{x}}$$

Expand the squared norm:

$$= \frac{1}{2}\int p_\sigma \|\mathbf{s}_\theta\|^2 - \int p_\sigma \, \mathbf{s}_\theta \cdot \nabla \log p_\sigma + \frac{1}{2}\int p_\sigma \|\nabla \log p_\sigma\|^2$$

Now consider the alternative objective:

$$\tilde{J}_\sigma = \frac{1}{2}\int p(\mathbf{x}) q_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) \|\mathbf{s}_\theta(\tilde{\mathbf{x}}) - \nabla \log q_\sigma(\tilde{\mathbf{x}}|\mathbf{x})\|^2 \, d\mathbf{x} \, d\tilde{\mathbf{x}}$$

Since \(p_\sigma(\tilde{\mathbf{x}}) = \int p(\mathbf{x}) q_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) d\mathbf{x}\), the first term \(\frac{1}{2}\int p_\sigma \|\mathbf{s}_\theta\|^2\) is the same in both.

The cross-term in \(\tilde{J}_\sigma\) is:

$$-\int p(\mathbf{x}) q_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) \, \mathbf{s}_\theta(\tilde{\mathbf{x}}) \cdot \nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) \, d\mathbf{x} \, d\tilde{\mathbf{x}}$$

$$= -\int \mathbf{s}_\theta(\tilde{\mathbf{x}}) \cdot \left[\int p(\mathbf{x}) \nabla_{\tilde{\mathbf{x}}} q_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) d\mathbf{x}\right] d\tilde{\mathbf{x}}$$

$$= -\int \mathbf{s}_\theta(\tilde{\mathbf{x}}) \cdot \nabla_{\tilde{\mathbf{x}}} p_\sigma(\tilde{\mathbf{x}}) \, d\tilde{\mathbf{x}}$$

$$= -\int p_\sigma(\tilde{\mathbf{x}}) \, \mathbf{s}_\theta(\tilde{\mathbf{x}}) \cdot \nabla \log p_\sigma(\tilde{\mathbf{x}}) \, d\tilde{\mathbf{x}}$$

This is exactly the cross-term in \(J_\sigma\). So the two objectives differ only by \(\theta\)-independent constants: \(J_\sigma = \tilde{J}_\sigma + \text{const}\). QED.

### The Practical Objective

Since \(q_\sigma(\tilde{\mathbf{x}} | \mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^2 I)\), the conditional score is:

$$\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} | \mathbf{x}) = -\frac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma^2} = -\frac{\boldsymbol{\epsilon}}{\sigma}$$

So the DSM objective is:

$$\mathcal{L}_{\text{DSM}} = \frac{1}{2}\mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}}\!\left[\left\|\mathbf{s}_\theta(\mathbf{x} + \sigma\boldsymbol{\epsilon}) + \frac{\boldsymbol{\epsilon}}{\sigma}\right\|^2\right]$$

The model learns to predict \(-\boldsymbol{\epsilon}/\sigma\) from the noisy input. No integration by parts, no divergence computation. Just add noise, predict the noise, and take gradients. This is the loss function that trains every diffusion model.

---

## Tweedie's Formula

**Tweedie's formula** is a result from Bayesian statistics that connects the posterior mean of a noised signal to the score function. It is the theoretical backbone of the "\(\mathbf{x}_0\)-prediction" interpretation of diffusion models.

### Statement

Let \(\tilde{\mathbf{x}} = \mathbf{x} + \sigma \boldsymbol{\epsilon}\) where \(\mathbf{x} \sim p(\mathbf{x})\) and \(\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)\). Then the posterior mean of \(\mathbf{x}\) given \(\tilde{\mathbf{x}}\) is:

$$\mathbb{E}[\mathbf{x} | \tilde{\mathbf{x}}] = \tilde{\mathbf{x}} + \sigma^2 \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}})$$

### Derivation from Bayes' Theorem

The posterior is:

$$p(\mathbf{x} | \tilde{\mathbf{x}}) = \frac{q_\sigma(\tilde{\mathbf{x}} | \mathbf{x}) \, p(\mathbf{x})}{p_\sigma(\tilde{\mathbf{x}})}$$

The posterior mean is:

$$\mathbb{E}[\mathbf{x} | \tilde{\mathbf{x}}] = \int \mathbf{x} \, p(\mathbf{x} | \tilde{\mathbf{x}}) \, d\mathbf{x} = \frac{\int \mathbf{x} \, q_\sigma(\tilde{\mathbf{x}} | \mathbf{x}) \, p(\mathbf{x}) \, d\mathbf{x}}{p_\sigma(\tilde{\mathbf{x}})}$$

Now use the identity: for \(q_\sigma(\tilde{\mathbf{x}} | \mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^2 I)\):

$$\nabla_{\tilde{\mathbf{x}}} q_\sigma(\tilde{\mathbf{x}} | \mathbf{x}) = q_\sigma(\tilde{\mathbf{x}} | \mathbf{x}) \cdot \frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma^2}$$

Therefore:

$$\mathbf{x} \cdot q_\sigma(\tilde{\mathbf{x}} | \mathbf{x}) = \tilde{\mathbf{x}} \cdot q_\sigma(\tilde{\mathbf{x}} | \mathbf{x}) + \sigma^2 \nabla_{\tilde{\mathbf{x}}} q_\sigma(\tilde{\mathbf{x}} | \mathbf{x})$$

Integrate both sides against \(p(\mathbf{x})\):

$$\int \mathbf{x} \, q_\sigma p \, d\mathbf{x} = \tilde{\mathbf{x}} \int q_\sigma p \, d\mathbf{x} + \sigma^2 \int \nabla_{\tilde{\mathbf{x}}} q_\sigma \, p \, d\mathbf{x}$$

$$= \tilde{\mathbf{x}} \, p_\sigma(\tilde{\mathbf{x}}) + \sigma^2 \nabla_{\tilde{\mathbf{x}}} p_\sigma(\tilde{\mathbf{x}})$$

Dividing by \(p_\sigma(\tilde{\mathbf{x}})\):

$$\mathbb{E}[\mathbf{x} | \tilde{\mathbf{x}}] = \tilde{\mathbf{x}} + \sigma^2 \frac{\nabla_{\tilde{\mathbf{x}}} p_\sigma(\tilde{\mathbf{x}})}{p_\sigma(\tilde{\mathbf{x}})} = \tilde{\mathbf{x}} + \sigma^2 \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}})$$

This is Tweedie's formula. The optimal denoiser (in the MSE sense) equals the noisy input plus a correction term proportional to the score of the noised distribution. **The optimal denoiser IS the score function** (up to a linear transformation).

<svg viewBox="0 0 700 250" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <defs>
    <marker id="arrowTw" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#66bb6a"/>
    </marker>
    <marker id="arrowTw2" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#E53935"/>
    </marker>
  </defs>
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">Tweedie's Formula: Denoising = Score Evaluation</text>

  <!-- Noisy point -->
  <circle cx="200" cy="140" r="6" fill="#E53935"/>
  <text x="200" y="125" text-anchor="middle" font-size="11" fill="#E53935">x̃ (noisy)</text>

  <!-- Clean point (posterior mean) -->
  <circle cx="450" cy="140" r="6" fill="#66bb6a"/>
  <text x="450" y="125" text-anchor="middle" font-size="11" fill="#66bb6a">E[x|x̃] (denoised)</text>

  <!-- Arrow from noisy to clean -->
  <line x1="210" y1="140" x2="438" y2="140" stroke="#66bb6a" stroke-width="2.5" marker-end="url(#arrowTw)"/>

  <!-- Decomposition -->
  <text x="325" y="170" text-anchor="middle" font-size="11" fill="#d4d4d4">= x̃ + σ² ∇ log p_σ(x̃)</text>

  <!-- Score direction label -->
  <text x="325" y="195" text-anchor="middle" font-size="10" fill="#4fc3f7">↑ score points toward data manifold</text>

  <!-- Data manifold (symbolic) -->
  <path d="M 420,200 C 440,180 460,170 500,175 S 560,195 580,200" fill="none" stroke="#999" stroke-width="1.5" stroke-dasharray="4,3"/>
  <text x="500" y="215" text-anchor="middle" font-size="10" fill="#999">data manifold</text>
</svg>

---

## The Denoising Autoencoder Connection

Vincent (2011) also connected denoising to autoencoders. A **denoising autoencoder (DAE)** is trained to reconstruct clean data from noisy input:

$$\mathcal{L}_{\text{DAE}} = \mathbb{E}_{\mathbf{x}, \tilde{\mathbf{x}}}\!\left[\|\mathbf{r}_\theta(\tilde{\mathbf{x}}) - \mathbf{x}\|^2\right]$$

where \(\tilde{\mathbf{x}} = \mathbf{x} + \sigma\boldsymbol{\epsilon}\). The optimal DAE output is the posterior mean:

$$\mathbf{r}^*(\tilde{\mathbf{x}}) = \mathbb{E}[\mathbf{x} | \tilde{\mathbf{x}}]$$

By Tweedie's formula:

$$\mathbf{r}^*(\tilde{\mathbf{x}}) = \tilde{\mathbf{x}} + \sigma^2 \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}})$$

Rearranging:

$$\nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}}) = \frac{\mathbf{r}^*(\tilde{\mathbf{x}}) - \tilde{\mathbf{x}}}{\sigma^2}$$

The score of the noised distribution is proportional to the **residual** of the optimal denoiser --- the vector pointing from the noisy input to the denoised output. Training a DAE implicitly learns the score function. This is the deep reason why diffusion models, which are essentially denoising autoencoders trained at many noise levels, learn to generate.

---

## Noise-Conditioned Score Networks

A single noise level \(\sigma\) is not enough. In regions of low data density, there is very little training data, so the score estimate is poor. The solution (Song & Ermon, 2019): use **many noise levels** simultaneously.

Define a sequence of noise levels \(\sigma_1 < \sigma_2 < \cdots < \sigma_L\), with \(\sigma_1\) small (nearly clean data) and \(\sigma_L\) large (data buried in noise). Train a single network \(\mathbf{s}_\theta(\mathbf{x}, \sigma)\) conditioned on the noise level:

$$\mathcal{L} = \sum_{l=1}^{L} \lambda(\sigma_l) \, \mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}}\!\left[\left\|\mathbf{s}_\theta(\mathbf{x} + \sigma_l \boldsymbol{\epsilon}, \sigma_l) + \frac{\boldsymbol{\epsilon}}{\sigma_l}\right\|^2\right]$$

The weighting \(\lambda(\sigma_l) = \sigma_l^2\) is a common choice that makes each term contribute equally.

**Why multiple noise levels help:**

- At **high noise** (\(\sigma_L\) large), the noised distribution \(p_{\sigma_L}\) is nearly Gaussian and covers the full data space. The score is well-defined everywhere, but carries only coarse information about the data.
- At **low noise** (\(\sigma_1\) small), the noised distribution closely resembles the true data distribution. The score carries fine detail, but is only accurate near the data manifold.
- By training at all levels simultaneously, the model learns a **multi-scale representation**: coarse structure from high noise, fine detail from low noise.

### Annealed Langevin Dynamics

To generate samples, run **Langevin dynamics** (gradient ascent on the log-density with added noise) starting from the highest noise level and gradually decreasing:

$$\mathbf{x}_{k+1} = \mathbf{x}_k + \frac{\delta}{2} \mathbf{s}_\theta(\mathbf{x}_k, \sigma_l) + \sqrt{\delta} \, \boldsymbol{\epsilon}_k$$

Run for many steps at each noise level before decreasing \(\sigma_l\). This progressively refines the sample from coarse to fine.

---

## From Score Matching to Diffusion

The continuous limit of "many noise levels" is a continuous noise schedule \(\sigma(t)\) for \(t \in [0, T]\). This is exactly the framework of diffusion models.

The forward noising process is:

$$\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}$$

The score at time \(t\) is \(\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)\), where \(p_t\) is the distribution of \(\mathbf{x}_t\). Training minimizes:

$$\mathcal{L} = \mathbb{E}_{t \sim \mathcal{U}[0,T]} \, \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}}\!\left[\lambda(t) \left\|\mathbf{s}_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0)\right\|^2\right]$$

By DSM, the conditional score is known:

$$\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0) = -\frac{\mathbf{x}_t - \alpha_t \mathbf{x}_0}{\sigma_t^2} = -\frac{\boldsymbol{\epsilon}}{\sigma_t}$$

So:

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\!\left[\lambda(t) \left\|\mathbf{s}_\theta(\alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}, t) + \frac{\boldsymbol{\epsilon}}{\sigma_t}\right\|^2\right]$$

This is the standard diffusion training objective. Denoising score matching at all noise levels simultaneously IS training a diffusion model.

---

## Epsilon, Score, x₀, and v-Prediction

Different parameterizations of the model prediction are all equivalent:

**\(\boldsymbol{\epsilon}\)-prediction:** The model predicts the noise \(\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t)\).

**Score-prediction:** The model predicts the score \(\hat{\mathbf{s}}_\theta(\mathbf{x}_t, t) \approx \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)\).

**\(\mathbf{x}_0\)-prediction:** The model predicts the clean data \(\hat{\mathbf{x}}_0(\mathbf{x}_t, t)\).

**v-prediction:** The model predicts \(\hat{\mathbf{v}} = \alpha_t \boldsymbol{\epsilon} - \sigma_t \mathbf{x}_0\).

The conversions between them are linear:

$$\hat{\mathbf{s}} = -\hat{\boldsymbol{\epsilon}} / \sigma_t$$

$$\hat{\mathbf{x}}_0 = (\mathbf{x}_t - \sigma_t \hat{\boldsymbol{\epsilon}}) / \alpha_t = \mathbf{x}_t + \sigma_t^2 \hat{\mathbf{s}} / \alpha_t$$

$$\hat{\mathbf{v}} = \alpha_t \hat{\boldsymbol{\epsilon}} - \sigma_t \hat{\mathbf{x}}_0$$

Mathematically equivalent, they differ in **training dynamics**:

- **\(\boldsymbol{\epsilon}\)-prediction** has uniform scale across noise levels but the target is noisy (high variance) at low noise levels
- **\(\mathbf{x}_0\)-prediction** is numerically unstable at high noise levels (the prediction oscillates wildly when very little signal remains)
- **v-prediction** is the most numerically stable: it is the velocity of the interpolation between \(\mathbf{x}_0\) and \(\boldsymbol{\epsilon}\), with bounded magnitude at all noise levels

---

## The Optimal Denoiser for Video

For video, the optimal denoiser \(\mathbb{E}[\mathbf{x}_0 | \mathbf{x}_t]\) operates over the full spatiotemporal volume. This is the posterior mean of the entire clean video given the noisy video.

**Why video denoising is harder:**

1. **The posterior is highly multimodal.** Given a noisy video, many clean videos are plausible. The posterior mean (which minimizes MSE) averages over these modes, producing blur. This is the "regression to the mean" problem, and it is worse for video because the space of plausible continuations is enormous.

2. **Temporal consistency requires joint processing.** Denoising each frame independently produces flickering. The score function for video must capture correlations across frames --- the score at frame \(t\) depends on all other frames.

3. **The score is spatiotemporally structured.** In regions of fast motion, the score must change rapidly across frames. In static regions, the temporal score is nearly zero. The network must learn to allocate capacity adaptively.

This is why video diffusion models use temporal attention (to capture cross-frame dependencies) and why they need more denoising steps than image models for comparable quality.

---

## Tweedie in Practice

In a trained diffusion model, Tweedie's formula gives the one-step denoised prediction from any noise level:

$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t + \sigma_t^2 \, \hat{\mathbf{s}}_\theta(\mathbf{x}_t, t)}{\alpha_t} = \frac{\mathbf{x}_t - \sigma_t \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t)}{\alpha_t}$$

At **high noise** (\(\sigma_t\) large): \(\hat{\mathbf{x}}_0\) is the blurry global average of the data distribution. The score provides only coarse directional information.

At **low noise** (\(\sigma_t\) small): \(\hat{\mathbf{x}}_0\) is a sharp, detailed prediction close to the true clean data. The score captures fine structure.

This reveals the diffusion process as a **coarse-to-fine** refinement: early steps (high noise) establish the global composition and layout; later steps (low noise) add textures, edges, and details. The Tweedie prediction at each step shows you exactly what the model "sees" at that noise level.

---

## Python: Score Estimation and Langevin Dynamics

```python
import numpy as np
import matplotlib.pyplot as plt

# Target: mixture of Gaussians in 2D
def target_density(x, y):
    """Density of a mixture of 3 Gaussians."""
    centers = [(-2, 0), (2, 1), (0, -2)]
    weights = [0.3, 0.4, 0.3]
    sigma = 0.6
    p = 0.0
    for (cx, cy), w in zip(centers, weights):
        p += w * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    p /= (2 * np.pi * sigma**2)
    return p

def target_score(x, y, noise_sigma=0.0):
    """Analytical score of the noised mixture."""
    centers = [(-2, 0), (2, 1), (0, -2)]
    weights = [0.3, 0.4, 0.3]
    sigma2 = 0.6**2 + noise_sigma**2  # convolved variance
    p = 0.0
    dpx = 0.0
    dpy = 0.0
    for (cx, cy), w in zip(centers, weights):
        gi = w * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma2))
        p += gi
        dpx += gi * (-(x - cx) / sigma2)
        dpy += gi * (-(y - cy) / sigma2)
    p = np.maximum(p, 1e-30)
    return dpx / p, dpy / p

def langevin_dynamics(score_fn, n_samples=500, n_steps=200, step_size=0.05, noise_sigma=0.0):
    """Run Langevin dynamics using the score function."""
    x = np.random.randn(n_samples) * 3
    y = np.random.randn(n_samples) * 3
    trajectory_x = [x.copy()]
    trajectory_y = [y.copy()]

    for _ in range(n_steps):
        sx, sy = score_fn(x, y, noise_sigma)
        x = x + step_size / 2 * sx + np.sqrt(step_size) * np.random.randn(n_samples)
        y = y + step_size / 2 * sy + np.sqrt(step_size) * np.random.randn(n_samples)
        trajectory_x.append(x.copy())
        trajectory_y.append(y.copy())

    return x, y, trajectory_x, trajectory_y

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Grid for density/score visualization
xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))

# Plot 1: Target density
p = target_density(xx, yy)
axes[0, 0].contourf(xx, yy, p, levels=30, cmap='viridis')
axes[0, 0].set_title(r'Target density $p(\mathbf{x})$')
axes[0, 0].set_aspect('equal')

# Plot 2: Score field
sx, sy = target_score(xx, yy, noise_sigma=0.0)
# Subsample for quiver
step = 8
axes[0, 1].quiver(xx[::step, ::step], yy[::step, ::step],
                  sx[::step, ::step], sy[::step, ::step],
                  color='cyan', alpha=0.8, scale=40)
axes[0, 1].contour(xx, yy, p, levels=10, colors='white', alpha=0.3)
axes[0, 1].set_title(r'Score field $\nabla \log p(\mathbf{x})$')
axes[0, 1].set_aspect('equal')
axes[0, 1].set_facecolor('#1a1a1a')

# Plot 3: Score field at high noise (sigma=1.5)
sx_noisy, sy_noisy = target_score(xx, yy, noise_sigma=1.5)
p_noisy = target_density(xx / np.sqrt(1 + 1.5**2/0.36), yy / np.sqrt(1 + 1.5**2/0.36))
axes[0, 2].quiver(xx[::step, ::step], yy[::step, ::step],
                  sx_noisy[::step, ::step], sy_noisy[::step, ::step],
                  color='orange', alpha=0.8, scale=20)
axes[0, 2].set_title(r'Score at $\sigma = 1.5$ (blurred)')
axes[0, 2].set_aspect('equal')
axes[0, 2].set_facecolor('#1a1a1a')

# Plot 4-6: Langevin dynamics at different step counts
for idx, (n_steps, title) in enumerate([(10, '10 steps'), (50, '50 steps'), (500, '500 steps')]):
    x_samples, y_samples, _, _ = langevin_dynamics(
        target_score, n_samples=2000, n_steps=n_steps, step_size=0.02)

    ax = axes[1, idx]
    ax.scatter(x_samples, y_samples, s=1, alpha=0.3, color='#4fc3f7')
    ax.contour(xx, yy, p, levels=8, colors='white', alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title(f'Langevin samples ({title})')
    ax.set_facecolor('#1a1a1a')

plt.suptitle('Score Functions and Langevin Dynamics', fontsize=14)
plt.tight_layout()
plt.savefig('score_langevin.png', dpi=150, bbox_inches='tight')
plt.show()
```

The mathematical chain is complete: **denoising = score estimation = diffusion model training**. The optimal denoiser points toward the data manifold (Tweedie). This direction IS the score function (Vincent). Training a denoiser at all noise levels IS score matching (Song & Ermon). And running the reverse-time SDE using the learned score IS generation (Song et al., 2020). Every step in this chain is a rigorous mathematical identity, not an approximation. This is why diffusion models have such clean theoretical foundations --- and why they work so well.
