---
layout: post
title: "Probability Theory from First Principles: The Mathematics of Uncertainty in Generative Models"
date: 2025-12-29
category: math
---

Generative models learn probability distributions. A diffusion model learns $p(\text{image})$. A language model learns $p(\text{next token} \mid \text{context})$. A VAE learns an approximate posterior $q(z \mid x)$. If you want to understand these models at a level deeper than API calls, you need probability theory --- not as a collection of formulas to memorize, but as a coherent mathematical framework for reasoning about uncertainty.

This post builds that framework from scratch. We start with the axioms --- the minimal rules that any valid notion of probability must satisfy --- and derive everything else: random variables, distributions, expectation, variance, Bayes' theorem, maximum likelihood estimation, and the Central Limit Theorem. Every concept is defined before it is used, every formula is derived rather than asserted, and every abstraction is connected to the generative modeling context where it matters.

---

## Table of Contents

1. [Sample Spaces and Events](#sample-spaces-and-events)
2. [The Axioms of Probability](#the-axioms-of-probability)
3. [Conditional Probability and Independence](#conditional-probability-and-independence)
4. [Random Variables](#random-variables)
5. [Discrete Distributions: Bernoulli, Binomial, Poisson](#discrete-distributions-bernoulli-binomial-poisson)
6. [Continuous Distributions: Uniform, Exponential, Gaussian](#continuous-distributions-uniform-exponential-gaussian)
7. [Expectation, Variance, and Moments](#expectation-variance-and-moments)
8. [Joint Distributions and Marginalization](#joint-distributions-and-marginalization)
9. [Bayes' Theorem](#bayes-theorem)
10. [Maximum Likelihood Estimation](#maximum-likelihood-estimation)
11. [The Central Limit Theorem](#the-central-limit-theorem)
12. [Connections to Generative Models](#connections-to-generative-models)

---

## Sample Spaces and Events

Before we can define probability, we need to define what we are assigning probability to.

A **sample space** $\Omega$ is the set of all possible outcomes of a random experiment. Every possible thing that could happen is one element of $\Omega$.

Examples:
- Coin flip: $\Omega = \\{H, T\\}$
- Six-sided die: $\Omega = \\{1, 2, 3, 4, 5, 6\\}$
- Generating a 256x256 grayscale image: $\Omega = [0, 255]^{65536}$ (each pixel takes a value in $[0, 255]$, and there are $256 \times 256 = 65536$ pixels)

An **event** is a subset of the sample space --- a collection of outcomes we might care about. "The die shows an even number" is the event $\\{2, 4, 6\\} \subset \Omega$. "The generated image contains a face" is some (complicated, hard-to-describe) subset of $[0, 255]^{65536}$.

We denote events with capital letters like $A$, $B$, $C$. The key set operations are:
- **Union** $A \cup B$: the event that $A$ or $B$ (or both) occurs
- **Intersection** $A \cap B$: the event that both $A$ and $B$ occur
- **Complement** $A^c$: the event that $A$ does not occur

---

## The Axioms of Probability

Andrey Kolmogorov formalized probability theory in 1933 with three axioms. A **probability measure** $P$ is a function that assigns a number to each event, satisfying:

**Axiom 1 (Non-negativity):** For any event $A$:
$$P(A) \geq 0$$

Probabilities are never negative. This seems obvious, but it is an axiom, not a theorem. (Quantum mechanics uses "quasi-probabilities" that can be negative, which is exactly why quantum computing is weird.)

**Axiom 2 (Normalization):** The probability of the entire sample space is 1:
$$P(\Omega) = 1$$

Something must happen. The probabilities of all possible outcomes sum (or integrate) to 1.

**Axiom 3 (Countable additivity):** For any countable sequence of mutually exclusive events $A_1, A_2, A_3, \ldots$ (meaning $A_i \cap A_j = \emptyset$ for $i \neq j$):
$$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

If events cannot happen simultaneously, the probability of at least one occurring is the sum of their individual probabilities.

Everything else in probability theory is a consequence of these three axioms. Let's derive some immediate results.

**The complement rule:** Since $A$ and $A^c$ are mutually exclusive and $A \cup A^c = \Omega$:
$$P(A) + P(A^c) = P(\Omega) = 1$$
$$P(A^c) = 1 - P(A)$$

**The addition rule:** For any two events (not necessarily mutually exclusive):
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

We subtract $P(A \cap B)$ because it gets counted once in $P(A)$ and once in $P(B)$, so we have double-counted it.

---

## Conditional Probability and Independence

**Conditional probability** answers: "Given that $B$ has occurred, what is the probability of $A$?" The definition is:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

provided $P(B) > 0$. The intuition: once we know $B$ has occurred, the effective sample space shrinks from $\Omega$ to $B$. We need to rescale so that probabilities within $B$ still sum to 1, and the fraction of $B$ that also lies in $A$ is $P(A \cap B) / P(B)$.

Rearranging gives the **multiplication rule**: $P(A \cap B) = P(A \mid B) \cdot P(B)$. The probability that both events occur equals the probability of the first times the probability of the second given the first.

**Independence.** Two events $A$ and $B$ are **independent** if knowing that $B$ occurred tells you nothing about $A$:

$$P(A \mid B) = P(A)$$

Substituting into the multiplication rule: $P(A \cap B) = P(A) \cdot P(B)$. This is the equivalent definition --- independent events have joint probability equal to the product of their individual probabilities.

Independence is a modeling assumption that can be powerful or dangerous. The "naive" in Naive Bayes means assuming that features are conditionally independent given the class label. Language models exist precisely because words are *not* independent --- the probability of the next word depends heavily on the previous words.

**The law of total probability.** If $B_1, B_2, \ldots, B_n$ form a partition of $\Omega$ (they are mutually exclusive and exhaustive), then for any event $A$:

$$P(A) = \sum_{i=1}^{n} P(A \mid B_i) P(B_i)$$

This is a powerful decomposition: compute the probability of $A$ by conditioning on all the different ways the world could be partitioned. In machine learning, marginalization over latent variables is exactly this --- summing over all possible values of a hidden variable, weighted by their probabilities.

---

## Random Variables

A **random variable** is a function that assigns a numerical value to each outcome in the sample space:

$$X: \Omega \to \mathbb{R}$$

This is a formal definition that often confuses people, so let's unpack it. The random variable $X$ is not a number --- it is a *function*. Given an outcome $\omega \in \Omega$, it produces a number $X(\omega)$. "The sum of two dice" is a random variable: the outcome is a pair like $(3, 5)$, and the random variable maps it to $8$.

We write $P(X = x)$ as shorthand for $P(\\{\omega \in \Omega : X(\omega) = x\\})$ --- the probability of the set of outcomes where $X$ takes the value $x$.

**Discrete random variables** take values in a countable set ($\\{0, 1, 2, \ldots\\}$ or $\\{-3, 0.5, 7\\}$, etc.). They are described by a **probability mass function** (PMF):

$$p(x) = P(X = x)$$

The PMF satisfies $p(x) \geq 0$ for all $x$ and $\sum_x p(x) = 1$.

**Continuous random variables** take values in an interval (or all of $\mathbb{R}$). They are described by a **probability density function** (PDF) $f(x)$. The critical distinction: for a continuous random variable, $P(X = x) = 0$ for any specific value $x$. Probabilities are only defined for intervals:

$$P(a \leq X \leq b) = \int_a^b f(x) \, dx$$

The PDF satisfies $f(x) \geq 0$ and $\int_{-\infty}^{\infty} f(x) \, dx = 1$.

The **cumulative distribution function** (CDF) works for both discrete and continuous:

$$F(x) = P(X \leq x)$$

For continuous variables, $F(x) = \int_{-\infty}^{x} f(t) \, dt$, and $f(x) = F'(x)$.

---

## Discrete Distributions: Bernoulli, Binomial, Poisson

### The Bernoulli Distribution

The simplest non-trivial distribution. A **Bernoulli** random variable $X$ takes the value 1 ("success") with probability $p$ and 0 ("failure") with probability $1 - p$:

$$P(X = x) = p^x (1-p)^{1-x}, \quad x \in \\{0, 1\\}$$

Verify: when $x = 1$, this gives $p^1(1-p)^0 = p$. When $x = 0$, this gives $p^0(1-p)^1 = 1-p$. The formula is compact and it generalizes nicely.

Every binary classification output, every coin flip, every pixel in a binary image is a Bernoulli random variable.

### The Binomial Distribution

If you perform $n$ independent Bernoulli trials, each with success probability $p$, and count the total number of successes $X$, then $X$ follows the **Binomial distribution** $X \sim \text{Binomial}(n, p)$.

The PMF is:

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ is the binomial coefficient --- the number of ways to choose which $k$ of the $n$ trials are successes. The derivation: each specific sequence with $k$ successes and $n-k$ failures has probability $p^k(1-p)^{n-k}$ (by independence), and there are $\binom{n}{k}$ such sequences.

### The Poisson Distribution

The **Poisson distribution** models the number of events occurring in a fixed interval of time or space, when events happen independently at a constant average rate $\lambda$.

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

Where does this come from? Derive it as a limit of the Binomial. Divide the interval into $n$ tiny sub-intervals, each with success probability $p = \lambda / n$. As $n \to \infty$:

$$\binom{n}{k}\left(\frac{\lambda}{n}\right)^k\left(1 - \frac{\lambda}{n}\right)^{n-k}$$

The binomial coefficient: $\binom{n}{k} = \frac{n(n-1)\cdots(n-k+1)}{k!} \to \frac{n^k}{k!}$ as $n \to \infty$.

The $p^k$ term: $\left(\frac{\lambda}{n}\right)^k = \frac{\lambda^k}{n^k}$.

The $(1-p)^{n-k}$ term: $\left(1 - \frac{\lambda}{n}\right)^{n-k} \to e^{-\lambda}$ (since $\lim_{n\to\infty}(1 - x/n)^n = e^{-x}$).

Combining: $\frac{n^k}{k!} \cdot \frac{\lambda^k}{n^k} \cdot e^{-\lambda} = \frac{\lambda^k e^{-\lambda}}{k!}$.

The Poisson distribution appears in ML whenever you model counts: word frequencies, the number of objects in an image region, event arrivals in a time series.

---

## Continuous Distributions: Uniform, Exponential, Gaussian

### The Uniform Distribution

$X \sim \text{Uniform}(a, b)$ assigns equal probability density to all points in $[a, b]$:

$$f(x) = \frac{1}{b-a}, \quad a \leq x \leq b$$

The density must be constant (that is the meaning of "uniform") and must integrate to 1 over $[a, b]$, which forces the height to be $\frac{1}{b - a}$.

### The Exponential Distribution

The **exponential distribution** models the time between events in a Poisson process. If events arrive at rate $\lambda$ per unit time, the waiting time $T$ until the first event has PDF:

$$f(t) = \lambda e^{-\lambda t}, \quad t \geq 0$$

Derivation: The probability of waiting at least $t$ is the probability of zero events in time $t$, which is the Poisson probability with mean $\lambda t$: $P(T > t) = e^{-\lambda t}$. The CDF is $F(t) = 1 - e^{-\lambda t}$, and the PDF is $f(t) = F'(t) = \lambda e^{-\lambda t}$.

The exponential distribution has a unique property: **memorylessness**. $P(T > t + s \mid T > t) = P(T > s)$. If you have already waited $t$ units, the remaining wait time has the same distribution as if you just started. No other continuous distribution has this property.

### The Gaussian (Normal) Distribution

The most important distribution in all of statistics and machine learning. A random variable $X \sim \mathcal{N}(\mu, \sigma^2)$ has PDF:

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

where $\mu$ is the mean (center) and $\sigma^2$ is the variance (spread). The standard normal has $\mu = 0$, $\sigma = 1$.

Where does this formula come from? There are several derivations; here is one from symmetry and independence.

We want a distribution in 2D that is: (1) rotationally symmetric (the density depends only on the distance from the origin, not the direction), and (2) independent in the $x$ and $y$ coordinates --- $f(x, y) = g(x) \cdot g(y)$ for some function $g$.

Condition (1): $f(x, y) = h(x^2 + y^2)$ for some function $h$.

Condition (2): $f(x, y) = g(x) \cdot g(y)$.

So we need $g(x) \cdot g(y) = h(x^2 + y^2)$.

Set $y = 0$: $g(x) \cdot g(0) = h(x^2)$. So $g(x) = \frac{h(x^2)}{g(0)}$.

Set $x = 0$: $g(0) \cdot g(y) = h(y^2)$. So $g(y) = \frac{h(y^2)}{g(0)}$.

Substituting back: $\frac{h(x^2)}{g(0)} \cdot \frac{h(y^2)}{g(0)} = h(x^2 + y^2)$.

This is a functional equation of the form $h(a) \cdot h(b) \propto h(a + b)$. The only continuous solution is the exponential: $h(r) = C e^{-\alpha r}$ for some constants $C$ and $\alpha > 0$ (we need $\alpha > 0$ for the integral to converge).

Therefore $g(x) \propto e^{-\alpha x^2}$, and normalizing gives the Gaussian PDF with $\alpha = \frac{1}{2\sigma^2}$.

This derivation reveals why Gaussians are everywhere: they are the **unique** distributions that are simultaneously rotationally symmetric and have independent coordinates. This is a deep mathematical fact, not a coincidence.

---

## Expectation, Variance, and Moments

The **expected value** (or mean) of a random variable is its long-run average:

$$E[X] = \begin{cases} \sum_x x \cdot p(x) & \text{(discrete)} \\ \int_{-\infty}^{\infty} x \cdot f(x) \, dx & \text{(continuous)} \end{cases}$$

Each possible value is weighted by its probability. This is a weighted average over the entire distribution. It tells you where the distribution is "centered."

Key properties (all provable from the definition):
- **Linearity:** $E[aX + bY] = aE[X] + bE[Y]$. This holds regardless of whether $X$ and $Y$ are independent. Linearity of expectation is the single most useful property in probability.
- **For a function:** $E[g(X)] = \sum_x g(x) p(x)$ (discrete) or $\int g(x) f(x) \, dx$ (continuous). This is the **law of the unconscious statistician** --- you don't need to find the distribution of $g(X)$.

The **variance** measures spread --- how far values typically deviate from the mean:

$$\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$

The second formula follows by expanding the square:

$$E[(X - \mu)^2] = E[X^2 - 2\mu X + \mu^2] = E[X^2] - 2\mu E[X] + \mu^2 = E[X^2] - \mu^2$$

The **standard deviation** $\sigma = \sqrt{\text{Var}(X)}$ is in the same units as $X$.

Properties:
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$. Shifting by $b$ doesn't change spread; scaling by $a$ scales variance by $a^2$.
- If $X$ and $Y$ are **independent**: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$.

**Moments** generalize these. The $k$-th moment is $E[X^k]$, and the $k$-th central moment is $E[(X - \mu)^k]$. The third central moment measures **skewness** (asymmetry), and the fourth measures **kurtosis** (tail heaviness). The Gaussian is completely determined by its first two moments --- this is special and one reason it is so analytically tractable.

---

## Joint Distributions and Marginalization

Two random variables $X$ and $Y$ have a **joint distribution** that describes their simultaneous behavior.

For discrete variables, the **joint PMF** is $p(x, y) = P(X = x, Y = y)$.

For continuous variables, the **joint PDF** $f(x, y)$ satisfies $P((X, Y) \in A) = \iint_A f(x, y) \, dx \, dy$.

**Marginalization** recovers the distribution of one variable by summing (or integrating) over the other:

$$p(x) = \sum_y p(x, y) \quad \text{(discrete)}$$
$$f_X(x) = \int_{-\infty}^{\infty} f(x, y) \, dy \quad \text{(continuous)}$$

This is the law of total probability applied to joint distributions. It says: to find the probability of $X = x$, consider all possible values of $Y$ and add up the joint probabilities.

Marginalization is one of the most important operations in probabilistic ML. In latent variable models (VAEs, mixture models, diffusion models), the data likelihood is obtained by marginalizing over the latent variable:

$$p(x) = \int p(x, z) \, dz = \int p(x \mid z) p(z) \, dz$$

This integral is often intractable, which is why variational inference exists --- but the mathematical operation is marginalization.

**Conditional distribution.** The conditional distribution of $Y$ given $X = x$ is:

$$p(y \mid x) = \frac{p(x, y)}{p(x)}$$

This is the definition of conditional probability applied to random variables. It tells you the distribution of $Y$ once you know the value of $X$.

The **covariance** measures linear association between two variables:

$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]$$

If $X$ and $Y$ are independent, $\text{Cov}(X, Y) = 0$ (but the converse is false --- zero covariance does not imply independence in general). The **correlation** normalizes covariance to $[-1, 1]$: $\rho = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$.

---

## Bayes' Theorem

Start from the definition of conditional probability, applied in two ways:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B \mid A) = \frac{P(A \cap B)}{P(A)}$$

Both have $P(A \cap B)$ in the numerator. Solve for it from the second equation: $P(A \cap B) = P(B \mid A) P(A)$. Substitute into the first:

$$P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}$$

This is **Bayes' theorem**. It looks simple, but it is the most important formula in probabilistic machine learning.

Rename the variables to make the ML interpretation clear. Let $\theta$ be a model parameter (or hypothesis) and $D$ be observed data:

$$P(\theta \mid D) = \frac{P(D \mid \theta) \, P(\theta)}{P(D)}$$

- $P(\theta \mid D)$: the **posterior** --- our updated belief about $\theta$ after seeing data
- $P(D \mid \theta)$: the **likelihood** --- how probable the data is under parameter $\theta$
- $P(\theta)$: the **prior** --- our belief about $\theta$ before seeing data
- $P(D)$: the **evidence** (or marginal likelihood) --- the total probability of the data

Bayes' theorem is a learning rule. You start with a prior belief, observe data, and update to a posterior. The evidence $P(D) = \int P(D \mid \theta) P(\theta) \, d\theta$ is a normalizing constant that ensures the posterior integrates to 1.

Here is a Python visualization of Bayesian updating --- watching a prior belief about a coin's bias evolve as we observe flips:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Bayesian updating for coin flip bias
# Prior: Beta(alpha, beta)
# After observing h heads and t tails: Beta(alpha + h, beta + t)

alpha_prior, beta_prior = 2, 2  # mildly informed prior centered at 0.5
observations = [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]  # biased coin

theta = np.linspace(0, 1, 500)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

checkpoints = [0, 1, 3, 5, 10, 20]

for ax, n in zip(axes.flatten(), checkpoints):
    h = sum(observations[:n])
    t = n - h
    a_post = alpha_prior + h
    b_post = beta_prior + t

    # Plot prior
    ax.plot(theta, beta.pdf(theta, alpha_prior, beta_prior),
            'b--', linewidth=1.5, alpha=0.5, label='Prior')
    # Plot posterior
    ax.plot(theta, beta.pdf(theta, a_post, b_post),
            'r-', linewidth=2.5, label=f'Posterior (n={n})')
    # True value
    ax.axvline(x=0.7, color='green', linestyle=':', linewidth=1.5, label='True p=0.7')

    ax.set_xlabel('θ (coin bias)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'After {n} observations ({h}H, {t}T)', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)

plt.suptitle('Bayesian Updating: Learning a Coin Bias', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('bayesian_updating.png', dpi=150, bbox_inches='tight')
plt.show()
```

The posterior starts broad (uncertain) and concentrates around the true value as more data arrives. This is learning. This is what every Bayesian neural network, every posterior inference algorithm, and every uncertainty-aware model is doing --- using Bayes' theorem to turn data into beliefs.

---

## Maximum Likelihood Estimation

Bayesian inference gives you the full posterior distribution over parameters. But often you just want a single "best" parameter value. **Maximum likelihood estimation** (MLE) finds the parameter that makes the observed data most probable:

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta P(D \mid \theta)$$

For independent observations $x_1, x_2, \ldots, x_n$, the likelihood factorizes:

$$P(D \mid \theta) = \prod_{i=1}^n p(x_i \mid \theta)$$

Products are numerically unstable and analytically awkward. Taking the logarithm (which is monotonic, so the maximizer is unchanged) gives the **log-likelihood**:

$$\ell(\theta) = \sum_{i=1}^n \log p(x_i \mid \theta)$$

Maximizing the log-likelihood means finding where its derivative is zero.

**Example: MLE for Gaussian parameters.** Given data $x_1, \ldots, x_n$ drawn from $\mathcal{N}(\mu, \sigma^2)$:

$$\ell(\mu, \sigma^2) = \sum_{i=1}^n \left[-\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x_i - \mu)^2}{2\sigma^2}\right]$$

$$= -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

Take the derivative with respect to $\mu$ and set to zero:

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0 \quad \Rightarrow \quad \hat{\mu} = \frac{1}{n}\sum_{i=1}^n x_i$$

The MLE for $\mu$ is the sample mean. Now for $\sigma^2$:

$$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^n (x_i - \mu)^2 = 0$$

$$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \hat{\mu})^2$$

The MLE for $\sigma^2$ is the sample variance (with $\frac{1}{n}$ rather than $\frac{1}{n-1}$; the difference is a bias correction that matters for small samples).

Here is a simulation that demonstrates MLE finding the best-fit Gaussian:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate data from a true distribution
np.random.seed(42)
true_mu, true_sigma = 3.0, 1.5
data = np.random.normal(true_mu, true_sigma, size=100)

# MLE estimates
mu_mle = np.mean(data)
sigma_mle = np.std(data)  # uses 1/n by default

# Visualize
x = np.linspace(-2, 8, 300)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: data histogram with MLE fit
axes[0].hist(data, bins=20, density=True, alpha=0.6, color='steelblue', label='Data')
axes[0].plot(x, norm.pdf(x, true_mu, true_sigma), 'g--', linewidth=2,
             label=f'True: μ={true_mu}, σ={true_sigma}')
axes[0].plot(x, norm.pdf(x, mu_mle, sigma_mle), 'r-', linewidth=2,
             label=f'MLE: μ={mu_mle:.2f}, σ={sigma_mle:.2f}')
axes[0].set_xlabel('x', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('MLE Fit to Data', fontsize=14)
axes[0].legend(fontsize=11)

# Right: log-likelihood surface
mu_range = np.linspace(1, 5, 100)
sigma_range = np.linspace(0.5, 3, 100)
MU, SIGMA = np.meshgrid(mu_range, sigma_range)
LL = np.zeros_like(MU)
for i in range(len(sigma_range)):
    for j in range(len(mu_range)):
        LL[i, j] = np.sum(norm.logpdf(data, MU[i, j], SIGMA[i, j]))

axes[1].contour(MU, SIGMA, LL, levels=30, cmap='RdYlBu_r')
axes[1].plot(mu_mle, sigma_mle, 'r*', markersize=15, label='MLE')
axes[1].plot(true_mu, true_sigma, 'g^', markersize=12, label='True')
axes[1].set_xlabel('μ', fontsize=12)
axes[1].set_ylabel('σ', fontsize=12)
axes[1].set_title('Log-Likelihood Surface', fontsize=14)
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig('mle_gaussian.png', dpi=150, bbox_inches='tight')
plt.show()
```

MLE is the foundation of most training procedures in deep learning. When you minimize cross-entropy loss (for classification) or mean squared error (for regression with Gaussian assumptions), you are doing maximum likelihood estimation. The loss function is the negative log-likelihood.

---

## The Central Limit Theorem

The **Central Limit Theorem** (CLT) is one of the most remarkable results in all of mathematics.

**Statement:** Let $X_1, X_2, \ldots, X_n$ be independent, identically distributed (i.i.d.) random variables with mean $\mu$ and finite variance $\sigma^2$. Then as $n \to \infty$, the standardized sample mean converges in distribution to a standard normal:

$$\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

where $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$.

In words: the average of many independent random variables is approximately Gaussian, regardless of the original distribution. It could be Bernoulli, Poisson, uniform, exponential --- it doesn't matter. Average enough of them, and you get a bell curve.

**Why is this true?** The deep reason involves characteristic functions (Fourier transforms of probability distributions), but the intuition is this: adding independent random variables corresponds to convolving their distributions. Convolution is a smoothing operation. Repeated convolution of any reasonable shape converges to a Gaussian, because the Gaussian is the **fixed point of convolution** in a precise sense (it is the distribution that maximizes entropy for a given mean and variance).

The CLT explains why Gaussians appear everywhere:
- Measurement errors (sum of many small independent perturbations)
- Thermal fluctuations
- Financial returns (approximately, over short timescales)
- The noise in diffusion models (the noise schedule adds many small Gaussian perturbations, and by the CLT, the result is Gaussian regardless of the data distribution)

Here is a simulation that shows the CLT in action:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)

# Different source distributions
distributions = {
    'Uniform(0,1)': lambda size: np.random.uniform(0, 1, size),
    'Exponential(1)': lambda size: np.random.exponential(1, size),
    'Bernoulli(0.3)': lambda size: np.random.binomial(1, 0.3, size),
    'Poisson(2)': lambda size: np.random.poisson(2, size),
}

sample_sizes = [1, 2, 5, 30]
n_simulations = 10000

fig, axes = plt.subplots(4, 4, figsize=(16, 14))

for row, (dist_name, sampler) in enumerate(distributions.items()):
    for col, n in enumerate(sample_sizes):
        # Generate n_simulations sample means, each from n observations
        means = np.array([sampler(n).mean() for _ in range(n_simulations)])

        ax = axes[row, col]
        ax.hist(means, bins=50, density=True, alpha=0.7, color='steelblue')

        # Overlay the CLT Gaussian prediction
        mu = means.mean()
        sigma = means.std()
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        ax.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2)

        if row == 0:
            ax.set_title(f'n = {n}', fontsize=13)
        if col == 0:
            ax.set_ylabel(dist_name, fontsize=11)
        ax.set_yticks([])

plt.suptitle('Central Limit Theorem: Sample Means Converge to Gaussian',
             fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('clt_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
```

Each row is a different source distribution. Each column increases the sample size $n$. By $n = 30$, every row looks Gaussian. The red curve is the theoretical Gaussian prediction from the CLT. The match is excellent even for $n = 30$, which is often cited as the practical threshold for CLT approximation.

---

## Connections to Generative Models

Every concept in this post maps directly to the machinery of modern generative models.

**Diffusion models** start with data from some unknown distribution $p_{\text{data}}(x)$ and gradually add Gaussian noise until the distribution becomes $\mathcal{N}(0, I)$. The model learns to reverse this process. The forward process is a sequence of conditional Gaussians. The training objective is a weighted sum of denoising losses, which is closely related to maximizing a variational lower bound on the log-likelihood (which is MLE). The CLT explains why Gaussian noise is the natural choice: after enough independent perturbations, the result is Gaussian regardless of the starting distribution.

**VAEs** define a generative model $p_\theta(x) = \int p_\theta(x \mid z) p(z) \, dz$. This is marginalization over the latent variable $z$. The integral is intractable, so we introduce an approximate posterior $q_\phi(z \mid x)$ and optimize a lower bound (the ELBO) using Bayes' theorem. Training minimizes a combination of reconstruction loss (related to the likelihood $p_\theta(x \mid z)$) and a KL divergence that regularizes the posterior toward the prior.

**Language models** estimate $p(\text{next token} \mid \text{context})$. Training is MLE: maximize the log-likelihood of observed sequences. The loss function is cross-entropy, which we will derive from information-theoretic first principles in the next post.

**Bayesian neural networks** place priors on weights $p(\theta)$ and compute (or approximate) the posterior $p(\theta \mid D)$ using Bayes' theorem. This gives uncertainty estimates for predictions.

The mathematical foundation is complete. We have the tools to define sample spaces, assign probabilities, work with random variables and their distributions, update beliefs with data, and estimate parameters from observations. The next post in this series --- information theory --- will build on top of this probability framework to explain entropy, divergence, and why your loss function takes the specific form it does.
