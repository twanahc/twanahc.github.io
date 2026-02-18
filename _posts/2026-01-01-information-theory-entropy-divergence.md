---
layout: post
title: "Information Theory: Entropy, Divergence, and Why Your Loss Function Works"
date: 2026-01-01
category: math
---

Every time you train a classifier with cross-entropy loss, you are doing information theory. Every time a VAE minimizes its ELBO objective, it is minimizing a KL divergence. Every time you compress a model, quantize weights, or distill knowledge from a large model into a small one, the fundamental limits are set by information-theoretic quantities. These are not loose analogies. The loss functions of modern machine learning are literally information-theoretic measures, derived from a question Claude Shannon asked in 1948: how do you quantify information?

This post builds information theory from that question. We derive the logarithmic measure of information from axioms, define entropy as expected surprise, construct cross-entropy and KL divergence, prove their key properties, and then connect every piece to the loss functions you use in practice. The goal is not to survey formulas but to make you understand *why* these specific quantities appear in ML --- why cross-entropy is the natural loss for classification, why KL divergence is the natural regularizer for variational inference, and why mutual information measures what a representation has learned.

---

## Table of Contents

1. [Shannon's Question](#shannons-question)
2. [Self-Information: The Axioms](#self-information-the-axioms)
3. [Entropy: Expected Surprise](#entropy-expected-surprise)
4. [Entropy of Common Distributions](#entropy-of-common-distributions)
5. [Cross-Entropy: Measuring Mismatch](#cross-entropy-measuring-mismatch)
6. [KL Divergence: The Extra Bits](#kl-divergence-the-extra-bits)
7. [Properties of KL Divergence](#properties-of-kl-divergence)
8. [Mutual Information](#mutual-information)
9. [The Data Processing Inequality](#the-data-processing-inequality)
10. [Rate-Distortion Theory: Compression Limits](#rate-distortion-theory-compression-limits)
11. [Why Cross-Entropy Loss Works](#why-cross-entropy-loss-works)
12. [KL Divergence in VAEs](#kl-divergence-in-vaes)
13. [Simulations](#simulations)

---

## Shannon's Question

In 1948, Claude Shannon asked a deceptively simple question: **how much information does a message contain?**

Not the semantic content --- not whether the message is interesting, true, or useful. Shannon wanted a mathematical measure of the *uncertainty resolved* by receiving a message. If you already know the outcome, a message confirming it gives you zero information. If the outcome was highly uncertain, a message resolving that uncertainty gives you a lot.

This leads to a fundamental insight: **information is the reduction of uncertainty**. A message is informative precisely to the extent that it was surprising. A weather report saying "it will be sunny" in the Sahara contains almost no information (you already expected that). The same report in London in November is genuinely informative (it could easily have been rain).

To make this precise, we need a mathematical definition of "surprise" that satisfies certain natural requirements.

---

## Self-Information: The Axioms

Consider an event that occurs with probability $p$. We want a function $I(p)$ that quantifies the surprise (or information content) of observing this event. What properties should $I$ have?

**Axiom 1: $I$ is a decreasing function of $p$.**
More probable events are less surprising. Certain events ($p = 1$) carry zero information. Impossible events ($p \to 0$) carry infinite information.

**Axiom 2: $I(1) = 0$.**
An event that always happens provides no information.

**Axiom 3: $I$ is continuous.**
A small change in probability produces a small change in information.

**Axiom 4 (Additivity): For independent events, $I(p_1 \cdot p_2) = I(p_1) + I(p_2)$.**
If two independent events occur, the total information is the sum of the individual informations. Learning that a coin flip is heads and a die roll is 6 gives you the sum of the information from each event separately.

Now here is the key derivation. We need a continuous, decreasing function $I$ satisfying $I(1) = 0$ and $I(p_1 p_2) = I(p_1) + I(p_2)$.

Axiom 4 says $I$ turns multiplication into addition. The only continuous function that does this is the logarithm:

$$I(p) = -\log p$$

Why the negative sign? Because $\log p \leq 0$ for $p \in (0, 1]$, and we want information to be non-negative. The negative sign flips the sign so that rare events (small $p$) have large information and certain events ($p = 1$) have zero.

Let's verify: $I(p_1 p_2) = -\log(p_1 p_2) = -\log p_1 - \log p_2 = I(p_1) + I(p_2)$. Additivity holds.

**Proof of uniqueness.** Define $g(x) = I(e^x)$ so that $g$ satisfies $g(x + y) = g(x) + g(y)$ for all real $x, y$. By Axiom 3 (continuity), the only solution to this Cauchy functional equation is $g(x) = cx$ for some constant $c$. Therefore $I(e^x) = cx$, which means $I(p) = c \ln p$. Since $I$ must be decreasing ($c < 0$) and we conventionally choose $c = -1/\ln 2$ to measure in bits, we get $I(p) = -\log_2 p$.

The choice of logarithm base determines the unit:
- Base 2: information measured in **bits** (binary digits)
- Base $e$: information measured in **nats** (natural units)
- Base 10: information measured in **hartleys**

In ML, we almost always use natural logarithm (base $e$, measured in nats) because it plays nicely with calculus. The conversion is $\log_2 p = \frac{\ln p}{\ln 2}$.

**Self-information** of an event with probability $p$ is:

$$I(p) = -\log p$$

A fair coin flip ($p = 1/2$): $I = -\log_2(1/2) = 1$ bit. You learn exactly one bit of information. A die roll ($p = 1/6$): $I = -\log_2(1/6) \approx 2.585$ bits. A 1-in-a-million event: $I = -\log_2(10^{-6}) \approx 19.93$ bits.

---

## Entropy: Expected Surprise

Self-information measures the surprise of a single event. **Entropy** is the average surprise over an entire distribution. If a random variable $X$ has possible outcomes $x_1, x_2, \ldots, x_n$ with probabilities $p_1, p_2, \ldots, p_n$, its entropy is:

$$H(X) = E[I(X)] = -\sum_{i=1}^n p_i \log p_i$$

This is the expected value of self-information under the distribution of $X$.

Entropy measures the **average uncertainty** you have about $X$ before observing it. Equivalently, it measures the average number of bits (or nats) of information you gain upon observing the outcome.

**Key properties of entropy:**

**Non-negativity:** $H(X) \geq 0$. You can verify: each term $-p_i \log p_i \geq 0$ since $p_i \in [0, 1]$ implies $\log p_i \leq 0$.

**Maximum entropy:** For a discrete distribution over $n$ outcomes, entropy is maximized when all outcomes are equally likely ($p_i = 1/n$ for all $i$):

$$H_{\max} = -\sum_{i=1}^n \frac{1}{n} \log \frac{1}{n} = \log n$$

This is the uniform distribution. Maximum uncertainty. No outcome is more or less expected than any other.

**Minimum entropy:** Entropy is zero if and only if one outcome has probability 1 (and all others have probability 0). Zero uncertainty --- you already know what will happen.

**Why entropy matters for ML:** Entropy quantifies the irreducible randomness in a distribution. You cannot compress data below its entropy (Shannon's source coding theorem). A generative model that perfectly captures the data distribution produces samples with entropy equal to the data entropy. A classifier's output distribution has low entropy when it is confident (probability concentrated on one class) and high entropy when it is uncertain (probability spread across classes).

For the continuous case, **differential entropy** is defined as:

$$h(X) = -\int_{-\infty}^{\infty} f(x) \log f(x) \, dx$$

Unlike discrete entropy, differential entropy can be negative (a uniform distribution on $[0, 1/2]$ has $h = -\log 2 < 0$ in nats). This is a subtlety that does not affect any of the ML applications, because differences of differential entropies (like KL divergence) are always well-defined and non-negative.

---

## Entropy of Common Distributions

**Bernoulli($p$):**

$$H = -p \log p - (1-p) \log(1-p)$$

This ranges from 0 (when $p = 0$ or $p = 1$, certain outcome) to $\log 2$ bits (when $p = 1/2$, maximum uncertainty for a binary variable). We will plot this curve in the simulations section.

**Gaussian($\mu, \sigma^2$):**

$$h = \frac{1}{2} \log(2\pi e \sigma^2)$$

The Gaussian has the maximum differential entropy among all distributions with the same mean and variance. This is a deep result: if you only know the mean and variance of a quantity, the least-committal (maximum entropy) assumption is that it is Gaussian. This provides a principled justification for the ubiquitous Gaussian assumptions in ML.

Derivation:

$$h = -\int_{-\infty}^{\infty} f(x) \log f(x) \, dx$$

where $f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$. So:

$$\log f(x) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}$$

Therefore:

$$h = -\int f(x) \left[-\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}\right] dx$$

$$= \frac{1}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2} E[(X-\mu)^2]$$

$$= \frac{1}{2}\log(2\pi\sigma^2) + \frac{1}{2}$$

$$= \frac{1}{2}\log(2\pi e \sigma^2)$$

The entropy depends only on the variance, not the mean. Shifting a distribution doesn't change its uncertainty --- only its spread matters.

---

## Cross-Entropy: Measuring Mismatch

Suppose the true distribution of a random variable is $p$, but you *believe* (or model) it to be $q$. The **cross-entropy** from $p$ to $q$ is:

$$H(p, q) = -\sum_x p(x) \log q(x)$$

or in the continuous case:

$$H(p, q) = -\int p(x) \log q(x) \, dx$$

What does this measure? If the true distribution is $p$ and you use a code designed for distribution $q$, the cross-entropy is the average number of bits you need per symbol. Since $q$ is "wrong," you will use more bits than the optimal $H(p)$, which corresponds to using a code designed for the true distribution $p$.

The relationship between entropy and cross-entropy:

$$H(p, q) = H(p) + D_{\text{KL}}(p \| q)$$

where $D_{\text{KL}}$ is the KL divergence, which we define next. Since $D_{\text{KL}} \geq 0$, cross-entropy is always at least as large as entropy: $H(p, q) \geq H(p)$, with equality if and only if $p = q$.

**Cross-entropy in classification.** In a classification task with $C$ classes, the true label for a data point is a one-hot distribution $p$: $p(y_{\text{true}}) = 1$ and $p(y) = 0$ for all other classes. The model predicts a distribution $q(y)$ (the softmax output). The cross-entropy loss for one data point is:

$$H(p, q) = -\sum_{c=1}^C p(c) \log q(c) = -\log q(y_{\text{true}})$$

Since $p$ is one-hot, only the true class contributes. Minimizing this loss means maximizing $q(y_{\text{true}})$ --- making the model assign high probability to the correct class. This is exactly maximum likelihood estimation from the previous post: $-\log q(y_{\text{true}})$ is the negative log-likelihood of the correct label under the model.

---

## KL Divergence: The Extra Bits

The **Kullback-Leibler divergence** (also called relative entropy) from distribution $p$ to distribution $q$ is:

$$D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = E_p\left[\log \frac{p(x)}{q(x)}\right]$$

This measures how much extra information (in bits or nats) is needed to encode samples from $p$ using a code optimized for $q$, beyond what is needed using a code optimized for $p$ itself.

Alternative forms:

$$D_{\text{KL}}(p \| q) = \sum_x p(x) \log p(x) - \sum_x p(x) \log q(x) = -H(p) + H(p, q)$$

This confirms: KL divergence is the difference between cross-entropy and entropy. It is the "excess cost" of using the wrong distribution.

**Derivation from cross-entropy:** We already showed $H(p, q) = H(p) + D_{\text{KL}}(p \| q)$. Rearranging: $D_{\text{KL}}(p \| q) = H(p, q) - H(p)$. Cross-entropy minus entropy equals the divergence. Cross-entropy combines two things: the irreducible randomness $H(p)$ and the modeling error $D_{\text{KL}}(p \| q)$. When you minimize cross-entropy loss in training, you are minimizing KL divergence, because $H(p)$ (the entropy of the true data distribution) is a constant with respect to your model parameters.

---

## Properties of KL Divergence

**Non-negativity (Gibbs' inequality):** $D_{\text{KL}}(p \| q) \geq 0$, with equality if and only if $p = q$.

This is the most important property. Let's prove it.

We use the inequality $\ln x \leq x - 1$, which holds for all $x > 0$ with equality only at $x = 1$. (This follows from the concavity of $\ln$: the tangent line at $x = 1$ is $y = x - 1$, and a concave function lies below any tangent line.)

$$-D_{\text{KL}}(p \| q) = -\sum_x p(x) \log\frac{p(x)}{q(x)} = \sum_x p(x) \log\frac{q(x)}{p(x)}$$

Applying $\ln x \leq x - 1$ with $x = q(x)/p(x)$:

$$\sum_x p(x) \log\frac{q(x)}{p(x)} \leq \sum_x p(x) \left(\frac{q(x)}{p(x)} - 1\right) = \sum_x q(x) - \sum_x p(x) = 1 - 1 = 0$$

Therefore $-D_{\text{KL}}(p \| q) \leq 0$, which means $D_{\text{KL}}(p \| q) \geq 0$.

**Asymmetry:** $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$ in general. KL divergence is not a distance metric. It is a directed measure.

The asymmetry has practical consequences:
- **Forward KL** $D_{\text{KL}}(p \| q)$: penalizes $q$ for placing low probability where $p$ has high probability. This encourages $q$ to be "mean-seeking" --- it tries to cover all the modes of $p$.
- **Reverse KL** $D_{\text{KL}}(q \| p)$: penalizes $q$ for placing high probability where $p$ has low probability. This encourages $q$ to be "mode-seeking" --- it concentrates on one mode of $p$ rather than trying to cover all of them.

This distinction matters hugely in variational inference. VAEs minimize $D_{\text{KL}}(q_\phi(z|x) \| p_\theta(z|x))$, which is the reverse KL. This tends to make the approximate posterior $q$ underestimate the uncertainty of the true posterior, concentrating on a single mode.

**KL divergence between two Gaussians.** For $p = \mathcal{N}(\mu_1, \sigma_1^2)$ and $q = \mathcal{N}(\mu_2, \sigma_2^2)$:

$$D_{\text{KL}}(p \| q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

This is one of the most-used formulas in ML. Let's derive it.

$$D_{\text{KL}}(p \| q) = \int p(x) \log\frac{p(x)}{q(x)} dx = \int p(x) [\log p(x) - \log q(x)] dx$$

We already computed $\int p(x) \log p(x) \, dx = -h(p) = -\frac{1}{2}\log(2\pi e \sigma_1^2)$.

For the second term:

$$\int p(x) \log q(x) \, dx = \int p(x) \left[-\frac{1}{2}\log(2\pi\sigma_2^2) - \frac{(x - \mu_2)^2}{2\sigma_2^2}\right] dx$$

$$= -\frac{1}{2}\log(2\pi\sigma_2^2) - \frac{1}{2\sigma_2^2}E_p[(X - \mu_2)^2]$$

Expand $E_p[(X - \mu_2)^2]$:

$$E_p[(X - \mu_2)^2] = E_p[(X - \mu_1 + \mu_1 - \mu_2)^2] = \sigma_1^2 + (\mu_1 - \mu_2)^2$$

Putting it together:

$$D_{\text{KL}} = -\frac{1}{2}\log(2\pi e \sigma_1^2) + \frac{1}{2}\log(2\pi\sigma_2^2) + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2}$$

$$= \frac{1}{2}\left[\log\frac{\sigma_2^2}{\sigma_1^2} - 1 + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{\sigma_2^2}\right]$$

$$= \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

In the special case of a VAE's KL term, where $p = q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2)$ and $q = p(z) = \mathcal{N}(0, 1)$:

$$D_{\text{KL}}(q_\phi \| p) = -\frac{1}{2}\left(1 + \log\sigma^2 - \mu^2 - \sigma^2\right)$$

This closed-form expression is what makes VAE training computationally tractable.

---

## Mutual Information

Given two random variables $X$ and $Y$ with joint distribution $p(x, y)$ and marginals $p(x)$ and $p(y)$, the **mutual information** is:

$$I(X; Y) = D_{\text{KL}}(p(x, y) \| p(x)p(y)) = \sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}$$

Mutual information measures how much knowing one variable tells you about the other. It is the KL divergence between the joint distribution and the product of the marginals. If $X$ and $Y$ are independent, $p(x, y) = p(x)p(y)$, and $I(X; Y) = 0$. The more dependent they are, the larger $I(X; Y)$.

Equivalent expressions:

$$I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = H(X) + H(Y) - H(X, Y)$$

where $H(X \mid Y) = -\sum_{x,y} p(x,y) \log p(x \mid y)$ is the **conditional entropy** --- the remaining uncertainty about $X$ after observing $Y$.

The first expression is the most intuitive: mutual information is the reduction in uncertainty about $X$ achieved by observing $Y$. It is the amount of information that $Y$ provides about $X$.

**Key properties:**
- $I(X; Y) \geq 0$ (from non-negativity of KL divergence)
- $I(X; Y) = I(Y; X)$ (symmetric, unlike KL divergence)
- $I(X; Y) = 0$ if and only if $X$ and $Y$ are independent
- $I(X; X) = H(X)$ (a variable provides maximum information about itself)

In representation learning, mutual information between the input $X$ and a learned representation $Z$ measures how much of the input information the representation preserves. Methods like InfoNCE (used in contrastive learning) optimize lower bounds on mutual information.

---

## The Data Processing Inequality

If $X \to Y \to Z$ forms a **Markov chain** (meaning $Z$ is conditionally independent of $X$ given $Y$: all information about $X$ that $Z$ has must pass through $Y$), then:

$$I(X; Z) \leq I(X; Y)$$

Processing data can only destroy information, never create it. No matter how clever your processing of $Y$ is, the result $Z$ cannot contain more information about $X$ than $Y$ did.

This has profound implications for deep learning. Each layer of a neural network processes the output of the previous layer. By the data processing inequality, information about the input can only decrease (or stay the same) as you go deeper. A network cannot "recover" information that was lost at an earlier layer.

This is why architectural decisions about early layers matter so much. If the first convolutional layer has too few channels or too aggressive pooling, information is lost that no later layer can recover. Skip connections (as in ResNets) help by providing alternative paths for information to bypass potentially lossy layers.

The data processing inequality also explains why lossless compression is hard and lossy compression is everywhere. You cannot compress data and then recover information that was not in the compressed version. Rate-distortion theory makes this precise.

---

## Rate-Distortion Theory: Compression Limits

Rate-distortion theory asks: what is the minimum number of bits needed to represent a source with a given level of distortion?

Define a **distortion measure** $d(x, \hat{x})$ between the original data $x$ and the reconstruction $\hat{x}$ (e.g., mean squared error). The **rate-distortion function** $R(D)$ gives the minimum number of bits per symbol needed to achieve expected distortion at most $D$:

$$R(D) = \min_{p(\hat{x}|x): E[d(X,\hat{X})] \leq D} I(X; \hat{X})$$

The minimum is over all conditional distributions (encoding schemes) that achieve distortion $\leq D$. The answer is the mutual information between the source and the reconstruction under the optimal encoding.

Key insights:
- $R(0) = H(X)$ for discrete sources: lossless compression requires entropy bits.
- As $D$ increases (more distortion allowed), $R(D)$ decreases (fewer bits needed).
- $R(D) = 0$ when $D$ is large enough that you can just output the mean.

For a Gaussian source with variance $\sigma^2$ and squared-error distortion:

$$R(D) = \frac{1}{2}\log\frac{\sigma^2}{D}, \quad 0 \leq D \leq \sigma^2$$

This appears directly in VAEs. The ELBO objective balances a rate term (KL divergence from the approximate posterior to the prior, measured in nats) against a distortion term (reconstruction loss). The VAE is implicitly operating on the rate-distortion curve, trading off compression quality against reconstruction fidelity.

---

## Why Cross-Entropy Loss Works

We can now give a complete, information-theoretic explanation of why cross-entropy is the right loss function for classification.

**The setup.** You have a true data distribution $p_{\text{data}}(x, y)$ over inputs $x$ and labels $y$. Your model predicts $q_\theta(y \mid x)$. You want to find $\theta$ such that $q_\theta$ is as close to the true conditional $p_{\text{data}}(y \mid x)$ as possible.

**Step 1.** "Close" means small KL divergence. For a given input $x$, the mismatch is:

$$D_{\text{KL}}(p(y \mid x) \| q_\theta(y \mid x)) = \sum_y p(y \mid x) \log \frac{p(y \mid x)}{q_\theta(y \mid x)}$$

$$= -H(p(y \mid x)) - \sum_y p(y \mid x) \log q_\theta(y \mid x)$$

The first term $-H(p(y \mid x))$ does not depend on $\theta$. So minimizing KL divergence is equivalent to minimizing:

$$-\sum_y p(y \mid x) \log q_\theta(y \mid x) = H(p(y|x), q_\theta(y|x))$$

which is the cross-entropy.

**Step 2.** Average over inputs:

$$E_{x \sim p_{\text{data}}} \left[H(p(y|x), q_\theta(y|x))\right] = -E_{x \sim p_{\text{data}}} \sum_y p(y|x) \log q_\theta(y|x)$$

**Step 3.** In practice, we approximate this expectation with a training set. For each training example $(x_i, y_i)$ where $y_i$ is the true label, $p(y \mid x_i)$ is a one-hot distribution, so:

$$H(p(y|x_i), q_\theta(y|x_i)) = -\log q_\theta(y_i \mid x_i)$$

The average over the training set is:

$$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^N \log q_\theta(y_i \mid x_i)$$

This is the cross-entropy loss. It is simultaneously:
1. The negative log-likelihood (MLE objective from the probability post)
2. The cross-entropy between the empirical data distribution and the model
3. The KL divergence from the true conditional to the model (up to a constant)

Three different derivations, all arriving at the same formula. This convergence is not a coincidence --- it reflects the deep connection between probability theory, information theory, and statistical estimation.

---

## KL Divergence in VAEs

The Variational Autoencoder (VAE) provides a clean example of information theory at work in a generative model.

**The generative model:** $p_\theta(x) = \int p_\theta(x \mid z) p(z) \, dz$, where $z$ is a latent variable with prior $p(z) = \mathcal{N}(0, I)$.

**The problem:** The posterior $p_\theta(z \mid x) = \frac{p_\theta(x \mid z) p(z)}{p_\theta(x)}$ is intractable because $p_\theta(x)$ requires integrating over all possible $z$.

**The solution:** Introduce an approximate posterior $q_\phi(z \mid x)$ (the encoder) and derive the Evidence Lower Bound (ELBO).

Start from the log-likelihood:

$$\log p_\theta(x) = \log \int p_\theta(x \mid z) p(z) \, dz$$

Multiply and divide by $q_\phi(z \mid x)$ inside the integral:

$$= \log \int q_\phi(z \mid x) \frac{p_\theta(x \mid z) p(z)}{q_\phi(z \mid x)} \, dz$$

Apply Jensen's inequality ($\log$ is concave, so $\log E[Y] \geq E[\log Y]$):

$$\geq \int q_\phi(z \mid x) \log \frac{p_\theta(x \mid z) p(z)}{q_\phi(z \mid x)} \, dz$$

$$= E_{q_\phi(z|x)}[\log p_\theta(x \mid z)] - D_{\text{KL}}(q_\phi(z \mid x) \| p(z))$$

This is the ELBO. The gap between $\log p_\theta(x)$ and the ELBO is exactly $D_{\text{KL}}(q_\phi(z \mid x) \| p_\theta(z \mid x))$ --- the KL divergence from the approximate posterior to the true posterior.

The ELBO has two terms:
1. **Reconstruction term** $E_{q_\phi}[\log p_\theta(x \mid z)]$: measures how well the decoder reconstructs $x$ from $z$. This is a (negative) distortion measure.
2. **KL term** $D_{\text{KL}}(q_\phi(z \mid x) \| p(z))$: measures how much the encoder deviates from the prior. This is a rate measure --- the number of nats used to encode the latent representation.

The VAE objective is literally rate-distortion optimization. Maximizing the ELBO trades off reconstruction quality against the cost of representing information in the latent code. The $\beta$-VAE modifies this by weighting the KL term with a factor $\beta$, explicitly controlling where you sit on the rate-distortion curve.

---

## Simulations

### Bernoulli Entropy

The entropy of a Bernoulli random variable with parameter $p$ is $H(p) = -p\log_2 p - (1-p)\log_2(1-p)$. This curve captures the essence of entropy in a single picture:

```python
import numpy as np
import matplotlib.pyplot as plt

p = np.linspace(0.001, 0.999, 500)
H = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(p, H, 'b-', linewidth=2.5)
ax.set_xlabel('p (probability of success)', fontsize=13)
ax.set_ylabel('H(p) in bits', fontsize=13)
ax.set_title('Entropy of a Bernoulli Random Variable', fontsize=14)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='H = 1 bit')
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='p = 0.5')
ax.annotate('Maximum at p=0.5\n(maximum uncertainty)',
            xy=(0.5, 1), xytext=(0.65, 0.7),
            fontsize=11, arrowprops=dict(arrowstyle='->', color='red'),
            color='red')
ax.annotate('H → 0 as p → 0\n(certain outcome: always fail)',
            xy=(0.05, 0.05), xytext=(0.15, 0.3),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='green'),
            color='green')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('bernoulli_entropy.png', dpi=150, bbox_inches='tight')
plt.show()
```

At $p = 0$ or $p = 1$, entropy is zero --- the outcome is certain. At $p = 0.5$, entropy reaches its maximum of 1 bit --- maximum uncertainty. This parabola-like shape (it is not actually a parabola, but a sum of $x \log x$ terms) is the canonical entropy curve. Every classification model implicitly navigates this curve: confident predictions correspond to the low-entropy wings, uncertain predictions correspond to the peak.

### KL Divergence Between Two Gaussians

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def kl_gaussians(mu1, sigma1, mu2, sigma2):
    """KL(N(mu1,sigma1^2) || N(mu2,sigma2^2))"""
    return (np.log(sigma2 / sigma1)
            + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2)
            - 0.5)

x = np.linspace(-6, 10, 500)

# Fixed q = N(0,1), vary p
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

cases = [
    (0, 1, 'p = N(0,1) [same as q]'),
    (1, 1, 'p = N(1,1) [shifted mean]'),
    (3, 1, 'p = N(3,1) [large shift]'),
    (0, 0.5, 'p = N(0,0.25) [narrow]'),
    (0, 2, 'p = N(0,4) [wide]'),
    (2, 2, 'p = N(2,4) [shifted + wide]'),
]

for ax, (mu_p, sigma_p, label) in zip(axes.flatten(), cases):
    mu_q, sigma_q = 0, 1

    p_pdf = norm.pdf(x, mu_p, sigma_p)
    q_pdf = norm.pdf(x, mu_q, sigma_q)

    ax.fill_between(x, p_pdf, alpha=0.3, color='blue', label='p')
    ax.fill_between(x, q_pdf, alpha=0.3, color='red', label='q = N(0,1)')
    ax.plot(x, p_pdf, 'b-', linewidth=2)
    ax.plot(x, q_pdf, 'r-', linewidth=2)

    kl_fwd = kl_gaussians(mu_p, sigma_p, mu_q, sigma_q)
    kl_rev = kl_gaussians(mu_q, sigma_q, mu_p, sigma_p)

    ax.set_title(f'{label}\nKL(p||q)={kl_fwd:.3f}, KL(q||p)={kl_rev:.3f}',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(-6, 8)
    ax.set_ylim(0, max(p_pdf.max(), q_pdf.max()) * 1.2)

plt.suptitle('KL Divergence Between Gaussians (Note: KL(p||q) ≠ KL(q||p))',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('kl_divergence_gaussians.png', dpi=150, bbox_inches='tight')
plt.show()
```

The plots demonstrate asymmetry visually. When $p$ and $q$ are the same, both KL divergences are zero. When $p$ is narrow and $q$ is wide, $D_{\text{KL}}(p \| q)$ is moderate (the narrow $p$ sits comfortably within the wide $q$), but $D_{\text{KL}}(q \| p)$ is large (the wide $q$ places significant probability where the narrow $p$ has almost none). This asymmetry is not a bug --- it reflects the fact that "how well does $q$ explain $p$" is a fundamentally different question from "how well does $p$ explain $q$."

### Estimating Mutual Information

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def estimate_mi_binned(x, y, bins=20):
    """Estimate mutual information using histogram binning."""
    # Joint histogram
    hist_xy, x_edges, y_edges = np.histogram2d(x, y, bins=bins, density=True)
    # Marginal histograms
    hist_x, _ = np.histogram(x, bins=x_edges, density=True)
    hist_y, _ = np.histogram(y, bins=y_edges, density=True)

    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]

    mi = 0
    for i in range(bins):
        for j in range(bins):
            if hist_xy[i, j] > 0 and hist_x[i] > 0 and hist_y[j] > 0:
                mi += hist_xy[i, j] * np.log(hist_xy[i, j] / (hist_x[i] * hist_y[j])) * dx * dy
    return mi

n = 5000

# Generate data with varying correlation
correlations = [0.0, 0.3, 0.6, 0.9, 0.99]
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for ax, rho in zip(axes, correlations):
    # Generate bivariate Gaussian with correlation rho
    cov = [[1, rho], [rho, 1]]
    data = np.random.multivariate_normal([0, 0], cov, n)
    x, y = data[:, 0], data[:, 1]

    # True MI for bivariate Gaussian
    mi_true = -0.5 * np.log(1 - rho**2) if abs(rho) < 1 else float('inf')
    mi_est = estimate_mi_binned(x, y, bins=30)

    ax.scatter(x, y, s=1, alpha=0.3, color='steelblue')
    ax.set_title(f'ρ = {rho}\nMI_true = {mi_true:.3f}\nMI_est = {mi_est:.3f}',
                 fontsize=11)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

plt.suptitle('Mutual Information: How Much Does Y Tell You About X?',
             fontsize=14, y=1.08)
plt.tight_layout()
plt.savefig('mutual_information.png', dpi=150, bbox_inches='tight')
plt.show()
```

At $\rho = 0$ (independent), mutual information is zero --- knowing $X$ tells you nothing about $Y$. As $\rho$ increases, the joint distribution becomes more concentrated along a line, and mutual information increases. At $\rho \to 1$ (perfect correlation), $MI \to \infty$ for continuous variables, because knowing $X$ perfectly determines $Y$.

For bivariate Gaussians, the true mutual information has a clean formula: $I(X; Y) = -\frac{1}{2}\log(1 - \rho^2)$. This diverges as $\rho \to \pm 1$, confirming that perfectly correlated Gaussian variables have infinite mutual information.

---

The three posts in this series --- linear algebra, probability, and information theory --- form the mathematical bedrock of modern machine learning. Linear algebra gives you the language of computation: vectors, matrices, transformations, decompositions. Probability gives you the language of uncertainty: distributions, expectations, likelihoods, posterior inference. Information theory gives you the language of learning itself: how much uncertainty exists, how much a model reduces it, and what the optimal reduction looks like.

Every loss function you will encounter in deep learning is a combination of these three. Cross-entropy loss is a probability-weighted sum (linear algebra) of log-probabilities (information theory) derived from maximum likelihood (probability). The VAE objective is a KL divergence (information theory) computed between Gaussian distributions (probability) parameterized by neural network outputs (linear algebra). Understanding these foundations does not just satisfy intellectual curiosity --- it gives you the ability to derive new loss functions, diagnose training failures, and reason about model behavior from first principles rather than from intuition alone.
