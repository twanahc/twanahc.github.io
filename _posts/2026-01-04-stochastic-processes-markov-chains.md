---
layout: post
title: "Stochastic Processes and Markov Chains: The Mathematics of Sequential Randomness"
date: 2026-01-04
category: math
---

Randomness is not the same as chaos. A stochastic process is a mathematical framework for describing how random quantities evolve over time, and the remarkable discovery is that even when individual outcomes are unpredictable, the collective statistical behavior often follows precise, derivable laws. Markov chains are the simplest and most powerful class of stochastic processes --- they underlie PageRank, MCMC sampling, reinforcement learning, and, crucially for this blog, the diffusion process at the heart of modern video generation models.

This post builds the theory from the ground up. We start with what a stochastic process is, work through random walks and their properties, formalize the Markov property, derive stationary distributions, prove the conditions for convergence, and then connect everything to MCMC and diffusion models. Every key result is derived, not stated.

---

## Table of Contents

1. [What a Stochastic Process Is](#what-a-stochastic-process-is)
2. [Random Walks: The Simplest Stochastic Process](#random-walks-the-simplest-stochastic-process)
3. [The Markov Property](#the-markov-property)
4. [Transition Matrices and the Chapman-Kolmogorov Equation](#transition-matrices-and-the-chapman-kolmogorov-equation)
5. [Stationary Distributions](#stationary-distributions)
6. [Detailed Balance](#detailed-balance)
7. [Ergodicity and Mixing Time](#ergodicity-and-mixing-time)
8. [MCMC: Metropolis-Hastings Algorithm](#mcmc-metropolis-hastings-algorithm)
9. [Gibbs Sampling](#gibbs-sampling)
10. [Connection to Diffusion Models](#connection-to-diffusion-models)
11. [Python Simulations](#python-simulations)
12. [Conclusion](#conclusion)

---

## What a Stochastic Process Is

A **stochastic process** is a collection of random variables indexed by time (or some other ordered set):

$$\{X_t\}_{t \in \mathcal{T}}$$

Each \(X_t\) is a random variable taking values in some **state space** \(\mathcal{S}\). The index set \(\mathcal{T}\) is the "time" parameter.

Let us unpack this with concrete examples.

**Discrete time, discrete state space.** \(\mathcal{T} = \{0, 1, 2, \ldots\}\) and \(\mathcal{S}\) is a finite or countable set. Example: the sequence of states visited by a chess piece performing a random walk on a board. At each integer time step, the piece is on some square. \(X_t\) is the square at time \(t\).

**Discrete time, continuous state space.** \(\mathcal{T} = \{0, 1, 2, \ldots\}\) and \(\mathcal{S} = \mathbb{R}^d\). Example: a stock price recorded at market close each day. \(X_t\) is the price on day \(t\), a real number.

**Continuous time, continuous state space.** \(\mathcal{T} = [0, \infty)\) and \(\mathcal{S} = \mathbb{R}^d\). Example: Brownian motion --- the position of a pollen grain in water, recorded at every instant.

The challenge is that a stochastic process is an infinite collection of random variables, and they are generally not independent. The value of \(X_5\) depends on the value of \(X_4\), which depends on \(X_3\), and so on. To fully specify a stochastic process, you would need the joint distribution of \((X_{t_1}, X_{t_2}, \ldots, X_{t_n})\) for every finite collection of times. This is generally intractable.

The solution: impose structure. The most powerful structural assumption is the **Markov property**, which we will formalize shortly. But first, let us build intuition with the simplest stochastic process.

---

## Random Walks: The Simplest Stochastic Process

A **random walk** is a stochastic process defined by cumulative independent random steps:

$$X_t = X_0 + \sum_{i=1}^{t} Z_i$$

where \(Z_1, Z_2, \ldots\) are independent, identically distributed (i.i.d.) random variables. The simplest case: \(Z_i = +1\) with probability \(p\) and \(Z_i = -1\) with probability \(q = 1-p\). This is the **simple random walk** on the integers.

### Properties of the 1D Simple Random Walk

**Mean position.** Since \(\mathbb{E}[Z_i] = p - q = 2p - 1\):

$$\mathbb{E}[X_t] = X_0 + t(2p - 1)$$

When \(p = 1/2\) (symmetric walk), the expected position stays at \(X_0\). The walk has no drift.

**Variance.** Since \(\text{Var}(Z_i) = 1 - (2p-1)^2 = 4pq\) and the steps are independent:

$$\text{Var}(X_t) = t \cdot 4pq$$

For the symmetric walk (\(p = q = 1/2\)): \(\text{Var}(X_t) = t\). The standard deviation grows as \(\sqrt{t}\). This is **diffusive scaling** --- the walker spreads out, but only as the square root of time, not linearly.

**Root-mean-square displacement.** For the symmetric walk: \(\sqrt{\mathbb{E}[(X_t - X_0)^2]} = \sqrt{t}\). After 100 steps, the typical displacement is only 10 steps from the start. After 10,000 steps, only 100. Random walks are surprisingly slow at exploring space.

**Recurrence in 1D.** A beautiful result: the symmetric 1D random walk returns to the origin with probability 1. It will return infinitely many times, in fact. The expected time to return is infinite (the walk returns, but it takes a long time on average). This is Polya's recurrence theorem for \(d = 1\).

### The 2D Random Walk

In two dimensions, at each step the walker moves to one of the four adjacent grid points with equal probability (up, down, left, right, each with probability 1/4).

Polya proved in 1921 that the 2D random walk is also **recurrent** --- it returns to the origin with probability 1. But in 3D and higher, the walk is **transient** --- there is a positive probability (about 34% in 3D) that the walker never returns to the origin. This is sometimes paraphrased as "a drunk man will find his way home, but a drunk bird may not."

<svg viewBox="0 0 700 350" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#e8e8e8">Random Walk: 1D Diffusive Spreading</text>
  <!-- Axes -->
  <line x1="60" y1="175" x2="670" y2="175" stroke="#444" stroke-width="1"/>
  <line x1="60" y1="30" x2="60" y2="320" stroke="#444" stroke-width="1"/>
  <text x="370" y="340" text-anchor="middle" font-size="12" fill="#999">Time step t</text>
  <text x="20" y="175" text-anchor="middle" font-size="12" fill="#999" transform="rotate(-90, 20, 175)">Position X(t)</text>
  <!-- Standard deviation envelope: ±sqrt(t) -->
  <path d="M60,175 Q200,145 365,120 Q530,100 670,85" stroke="#2563eb" stroke-width="1.5" fill="none" stroke-dasharray="6,4"/>
  <path d="M60,175 Q200,205 365,230 Q530,250 670,265" stroke="#2563eb" stroke-width="1.5" fill="none" stroke-dasharray="6,4"/>
  <text x="675" y="80" font-size="10" fill="#2563eb">+√t</text>
  <text x="675" y="270" font-size="10" fill="#2563eb">−√t</text>
  <!-- Sample path (jagged) -->
  <polyline points="60,175 80,165 100,175 120,165 140,155 160,165 180,155 200,145 220,155 240,165 260,155 280,145 300,155 320,165 340,155 360,145 380,155 400,145 420,135 440,145 460,155 480,145 500,135 520,125 540,135 560,145 580,135 600,125 620,135 640,145 660,135 670,140" stroke="#dc2626" stroke-width="1.5" fill="none" opacity="0.7"/>
  <!-- Another sample path -->
  <polyline points="60,175 80,185 100,195 120,185 140,195 160,205 180,215 200,205 220,215 240,225 260,215 280,205 300,215 320,225 340,215 360,225 380,235 400,225 420,215 440,225 460,235 480,225 500,215 520,225 540,235 560,225 580,215 600,225 620,215 640,225 660,220 670,225" stroke="#16a34a" stroke-width="1.5" fill="none" opacity="0.7"/>
  <!-- Legend -->
  <line x1="200" y1="310" x2="230" y2="310" stroke="#2563eb" stroke-width="1.5" stroke-dasharray="6,4"/>
  <text x="235" y="314" font-size="10" fill="#d4d4d4">±σ = ±√t envelope</text>
  <line x1="400" y1="310" x2="430" y2="310" stroke="#dc2626" stroke-width="1.5"/>
  <text x="435" y="314" font-size="10" fill="#d4d4d4">Sample paths</text>
</svg>

---

## The Markov Property

The **Markov property** is the assumption that the future depends on the present but not on the past:

$$P(X_{t+1} = x \mid X_t = x_t, X_{t-1} = x_{t-1}, \ldots, X_0 = x_0) = P(X_{t+1} = x \mid X_t = x_t)$$

In words: given the current state, the entire past history is irrelevant for predicting the future. The present state contains all the information needed.

This is sometimes called **memorylessness**. The process has no memory beyond its current state.

Why is this so powerful? Without the Markov property, specifying the process requires the conditional distribution of \(X_{t+1}\) given the entire history \((X_0, X_1, \ldots, X_t)\) --- a function of \(t+1\) variables. With the Markov property, you only need \(P(X_{t+1} \mid X_t)\) --- a function of 2 variables. The complexity collapses from exponential in the history length to constant.

A stochastic process with the Markov property is called a **Markov chain** (when the state space is discrete) or a **Markov process** (general state space).

**Important:** The Markov property does not mean the states are independent. \(X_{t+1}\) depends on \(X_t\), which depends on \(X_{t-1}\), etc. There are correlations across time. What the Markov property says is that these correlations are mediated entirely through the current state: knowing \(X_t\) makes \(X_{t+1}\) conditionally independent of \(X_{t-1}, X_{t-2}, \ldots\)

**Example: Random walk.** The simple random walk is Markov because \(X_{t+1} = X_t + Z_{t+1}\), and \(Z_{t+1}\) is independent of everything that came before. Given \(X_t\), \(X_{t+1}\) depends only on \(X_t\) (plus the independent noise \(Z_{t+1}\)).

**Non-example:** A process where the step size depends on the last two positions: \(X_{t+1} = 2X_t - X_{t-1} + Z_{t+1}\). This violates the Markov property because you need both \(X_t\) and \(X_{t-1}\) to predict \(X_{t+1}\). However, you can always convert it to a Markov chain by **state augmentation**: define \(Y_t = (X_t, X_{t-1})\). Then \(Y_{t+1}\) depends only on \(Y_t\), and the process in the augmented state space is Markov.

---

## Transition Matrices and the Chapman-Kolmogorov Equation

For a discrete-state Markov chain with states \(\{1, 2, \ldots, n\}\), the one-step dynamics are fully described by the **transition matrix** \(P\):

$$P_{ij} = P(X_{t+1} = j \mid X_t = i)$$

Each row of \(P\) sums to 1 (the chain must go somewhere), so \(P\) is a **stochastic matrix**. The entry \(P_{ij}\) is the probability of transitioning from state \(i\) to state \(j\) in one step.

If the initial distribution is \(\pi_0\) (a row vector where \((\pi_0)_j = P(X_0 = j)\)), then the distribution after one step is:

$$\pi_1 = \pi_0 P$$

After two steps:

$$\pi_2 = \pi_1 P = \pi_0 P^2$$

After \(t\) steps:

$$\pi_t = \pi_0 P^t$$

The \(t\)-step transition probabilities are given by the matrix power \(P^t\). Specifically, \((P^t)_{ij}\) is the probability of going from state \(i\) to state \(j\) in exactly \(t\) steps.

### The Chapman-Kolmogorov Equation

This matrix multiplication rule implies the **Chapman-Kolmogorov equation**:

$$(P^{m+n})_{ij} = \sum_{k} (P^m)_{ik} (P^n)_{kj}$$

In probability language:

$$P(X_{m+n} = j \mid X_0 = i) = \sum_{k} P(X_m = k \mid X_0 = i) \cdot P(X_{m+n} = j \mid X_m = k)$$

This says: to go from \(i\) to \(j\) in \(m + n\) steps, you must pass through some intermediate state \(k\) at time \(m\). Sum over all possible intermediate states. It is the law of total probability applied to the intermediate time point.

This is not an additional assumption --- it is a consequence of the Markov property. But it is powerful because it gives us a recursive way to compute multi-step transition probabilities from one-step probabilities.

---

## Stationary Distributions

A **stationary distribution** (or equilibrium distribution, or invariant distribution) is a probability distribution \(\pi\) over the states such that if the chain starts in distribution \(\pi\), it stays in distribution \(\pi\) forever:

$$\boxed{\pi = \pi P}$$

This is a left eigenvector equation: \(\pi\) is a left eigenvector of \(P\) with eigenvalue 1.

Why does eigenvalue 1 always exist? Because \(P\) is a stochastic matrix (rows sum to 1), the vector \(\mathbf{1} = (1, 1, \ldots, 1)^T\) satisfies \(P\mathbf{1} = \mathbf{1}\), so 1 is always an eigenvalue of \(P\). By the Perron-Frobenius theorem, 1 is the largest eigenvalue, and there exists a corresponding non-negative left eigenvector \(\pi\).

### Deriving the Stationary Distribution

For a concrete example, consider a two-state Markov chain (e.g., weather: sunny = 1, rainy = 2) with transition matrix:

$$P = \begin{pmatrix} 1-\alpha & \alpha \\ \beta & 1-\beta \end{pmatrix}$$

where \(\alpha\) is the probability of switching from sunny to rainy, and \(\beta\) is the probability of switching from rainy to sunny.

The stationary distribution satisfies \(\pi P = \pi\) with \(\pi_1 + \pi_2 = 1\):

$$\pi_1(1-\alpha) + \pi_2 \beta = \pi_1$$
$$\pi_1 \alpha + \pi_2(1-\beta) = \pi_2$$

From the first equation: \(-\pi_1\alpha + \pi_2\beta = 0\), so \(\pi_1\alpha = \pi_2\beta\), giving:

$$\frac{\pi_1}{\pi_2} = \frac{\beta}{\alpha}$$

With the normalization constraint:

$$\pi_1 = \frac{\beta}{\alpha + \beta}, \qquad \pi_2 = \frac{\alpha}{\alpha + \beta}$$

The interpretation: if sunny-to-rainy transitions are rare (\(\alpha\) small) and rainy-to-sunny transitions are common (\(\beta\) large), then the stationary distribution is mostly sunny (\(\pi_1 \gg \pi_2\)). The stationary distribution reflects the long-run fraction of time spent in each state.

### Convergence to Stationarity

Under mild conditions (irreducibility and aperiodicity, defined below), the chain converges to the stationary distribution regardless of where it starts:

$$\lim_{t \to \infty} \pi_0 P^t = \pi \qquad \text{for any initial distribution } \pi_0$$

This is the **fundamental theorem of Markov chains** (or the convergence theorem). The eigenvalue structure explains why: if we decompose \(\pi_0\) in the eigenbasis of \(P\), the component along \(\pi\) (eigenvalue 1) persists while all other components (eigenvalues \(|\lambda_i| < 1\)) decay exponentially.

---

## Detailed Balance

A Markov chain satisfies **detailed balance** with respect to a distribution \(\pi\) if:

$$\pi_i P_{ij} = \pi_j P_{ji} \qquad \text{for all states } i, j$$

The left side is the probability flow from \(i\) to \(j\) in stationarity (\(\pi_i\) times the transition rate \(P_{ij}\)). The right side is the flow from \(j\) to \(i\). Detailed balance means these flows are equal for every pair of states --- every individual transition is in equilibrium.

**Claim:** If a distribution \(\pi\) satisfies detailed balance, then \(\pi\) is a stationary distribution.

**Proof:** Sum detailed balance over \(i\):

$$\sum_i \pi_i P_{ij} = \sum_i \pi_j P_{ji} = \pi_j \sum_i P_{ji} = \pi_j \cdot 1 = \pi_j$$

The left side is \((\pi P)_j\). So \(\pi P = \pi\), confirming \(\pi\) is stationary. \(\blacksquare\)

Detailed balance is a **sufficient** condition for stationarity, not necessary. Chains can be stationary without detailed balance (such chains have net circulation of probability flow --- they are called non-reversible).

Why does detailed balance matter? It is the key to constructing MCMC algorithms. If we want to sample from a target distribution \(\pi\), we design a Markov chain whose transitions satisfy detailed balance with respect to \(\pi\). Then \(\pi\) is guaranteed to be the stationary distribution, and running the chain long enough produces samples from \(\pi\).

---

## Ergodicity and Mixing Time

For the convergence theorem to hold, the chain needs two properties:

**Irreducibility:** Every state can be reached from every other state with positive probability (in some finite number of steps). Formally: for all \(i, j\), there exists \(t\) such that \((P^t)_{ij} > 0\). This ensures the chain does not get trapped in a subset of states.

**Aperiodicity:** The chain does not cycle through states in a deterministic pattern. The **period** of state \(i\) is \(d_i = \gcd\{t \geq 1 : (P^t)_{ii} > 0\}\). The chain is aperiodic if \(d_i = 1\) for all \(i\). A simple way to ensure aperiodicity: include self-loops (\(P_{ii} > 0\) for some state \(i\)).

A chain that is both irreducible and aperiodic is called **ergodic**. The ergodic theorem states:

**Ergodic Theorem:** For an ergodic Markov chain with stationary distribution \(\pi\), and for any function \(g\) of the state:

$$\frac{1}{T}\sum_{t=1}^{T} g(X_t) \xrightarrow{T \to \infty} \sum_i \pi_i g(i) = \mathbb{E}_\pi[g]$$

almost surely. Time averages converge to ensemble averages. This is the Markov chain analog of the law of large numbers.

### Mixing Time

**Mixing time** quantifies how long the chain takes to "forget" its initial state and reach (approximately) the stationary distribution. Define the **total variation distance** between the chain's distribution at time \(t\) and the stationary distribution:

$$d(t) = \max_{x_0} \frac{1}{2}\sum_j \left|(P^t)_{x_0, j} - \pi_j\right|$$

The mixing time (at tolerance \(\epsilon\)) is:

$$t_{\text{mix}}(\epsilon) = \min\{t : d(t) \leq \epsilon\}$$

typically with \(\epsilon = 1/4\) or \(\epsilon = 1/(2e)\).

The mixing time is controlled by the **spectral gap** \(\gamma = 1 - |\lambda_2|\), where \(\lambda_2\) is the second-largest eigenvalue of \(P\) in absolute value (recall \(\lambda_1 = 1\) is the largest). The relationship is:

$$t_{\text{mix}} = \Theta\left(\frac{1}{\gamma}\right)$$

Large spectral gap means fast mixing (the non-stationary components decay quickly). Small spectral gap means slow mixing. For a chain with \(n\) states, the mixing time can range from \(O(\log n)\) (fast) to \(O(n^2)\) or worse (slow).

This has enormous practical implications for MCMC: if the mixing time is too long, the samples from the chain are highly correlated and provide little independent information about the target distribution.

---

## MCMC: Metropolis-Hastings Algorithm

**Markov Chain Monte Carlo (MCMC)** is a family of algorithms that construct a Markov chain whose stationary distribution is a target distribution \(\pi\) that we want to sample from. The idea is simple in retrospect: design the transition probabilities so that detailed balance holds with respect to \(\pi\), then run the chain and collect samples.

### The Problem

We want to sample from a distribution \(\pi(x)\), but we can only evaluate \(\pi(x)\) up to a normalizing constant. That is, we know a function \(\tilde{\pi}(x) \propto \pi(x)\), but computing \(Z = \sum_x \tilde{\pi}(x)\) (or \(Z = \int \tilde{\pi}(x)dx\)) is intractable.

This situation arises constantly in Bayesian statistics (posterior distributions), statistical physics (Boltzmann distributions), and machine learning (energy-based models).

### The Metropolis-Hastings Algorithm

Metropolis et al. (1953) and Hastings (1970) gave the following construction:

1. Choose a **proposal distribution** \(q(x' \mid x)\): given the current state \(x\), propose a new state \(x'\) drawn from \(q\).
2. **Accept or reject** the proposal with probability:

$$\boxed{A(x \to x') = \min\left(1, \frac{\tilde{\pi}(x') \, q(x \mid x')}{\tilde{\pi}(x) \, q(x' \mid x)}\right)}$$

3. If accepted, move to \(x'\). If rejected, stay at \(x\).

### Deriving the Acceptance Probability

The acceptance probability is specifically designed so that detailed balance holds. The effective transition probability from \(x\) to \(x'\) (for \(x' \neq x\)) is:

$$T(x \to x') = q(x' \mid x) \cdot A(x \to x')$$

We want \(\pi(x) T(x \to x') = \pi(x') T(x' \to x)\) (detailed balance). Substituting:

$$\pi(x) \, q(x' \mid x) \, A(x \to x') = \pi(x') \, q(x \mid x') \, A(x' \to x)$$

Rearranging:

$$\frac{A(x \to x')}{A(x' \to x)} = \frac{\pi(x') \, q(x \mid x')}{\pi(x) \, q(x' \mid x)}$$

Call the right side \(r\). We need \(A(x \to x') / A(x' \to x) = r\), and both acceptance probabilities must be in \([0, 1]\). The solution that maximizes the acceptance rate (to reduce rejected steps) is:

$$A(x \to x') = \min(1, r), \qquad A(x' \to x) = \min(1, 1/r)$$

This can be verified: if \(r \leq 1\), then \(A(x \to x') = r\) and \(A(x' \to x) = 1\), giving ratio \(r\). If \(r > 1\), then \(A(x \to x') = 1\) and \(A(x' \to x) = 1/r\), also giving ratio \(r\). \(\blacksquare\)

The beautiful thing: the normalizing constant \(Z\) cancels in the ratio \(\tilde{\pi}(x')/\tilde{\pi}(x)\), so we never need to compute it.

### Choosing the Proposal Distribution

The choice of \(q\) dramatically affects the mixing time. Common choices:

- **Random walk proposal:** \(q(x' \mid x) = \mathcal{N}(x, \sigma^2 I)\). The proposal is symmetric (\(q(x' \mid x) = q(x \mid x')\)), so the acceptance ratio simplifies to \(\min(1, \tilde{\pi}(x')/\tilde{\pi}(x))\).
- **If \(\sigma\) is too small:** almost all proposals are accepted, but the chain moves slowly (high acceptance, low movement per step).
- **If \(\sigma\) is too large:** the chain proposes distant points that are usually in low-probability regions, leading to frequent rejection (low acceptance, no movement).
- **The optimal acceptance rate** for random walk Metropolis in \(d\) dimensions is approximately 23.4% (Roberts, Gelman, and Gilks, 1997).

---

## Gibbs Sampling

**Gibbs sampling** is a special case of MCMC for multivariate distributions. Instead of proposing a move in all dimensions simultaneously, it updates one variable at a time, drawing from the **full conditional distribution**.

Given a target \(\pi(x_1, x_2, \ldots, x_d)\), one iteration of Gibbs sampling cycles through the variables:

$$x_1^{(t+1)} \sim \pi(x_1 \mid x_2^{(t)}, x_3^{(t)}, \ldots, x_d^{(t)})$$
$$x_2^{(t+1)} \sim \pi(x_2 \mid x_1^{(t+1)}, x_3^{(t)}, \ldots, x_d^{(t)})$$
$$\vdots$$
$$x_d^{(t+1)} \sim \pi(x_d \mid x_1^{(t+1)}, x_2^{(t+1)}, \ldots, x_{d-1}^{(t+1)})$$

Each conditional update is a Metropolis-Hastings step with acceptance probability 1 (the proposal is exactly the conditional distribution, so the ratio is always 1). This means Gibbs sampling never rejects a proposal.

The catch: you need to be able to sample from the full conditional distributions \(\pi(x_i \mid x_{-i})\), where \(x_{-i}\) denotes all variables except \(x_i\). For many models (especially those with conjugate priors in Bayesian statistics), these conditionals have standard forms and are easy to sample from.

Gibbs sampling can be slow when variables are highly correlated --- updating one variable at a time does not move the joint sample very far when variables are tightly coupled. In such cases, block Gibbs sampling (updating groups of variables together) or other MCMC methods may be more efficient.

---

## Connection to Diffusion Models

The forward and reverse processes of a **diffusion model** are Markov chains, and this is the mathematical foundation of their generative capability.

### The Forward Process

The forward (noising) process starts with a clean data sample \(x_0 \sim q(x_0)\) and adds Gaussian noise at each step:

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \, x_{t-1}, \beta_t I)$$

This is a Markov chain: \(x_t\) depends only on \(x_{t-1}\) (and the noise schedule parameter \(\beta_t\)). After sufficiently many steps (\(t = T\)), \(x_T\) is approximately standard Gaussian noise, regardless of the initial image \(x_0\).

The transition kernel has a special structure: it is a Gaussian centered near the previous state (scaled by \(\sqrt{1-\beta_t}\)) with variance \(\beta_t\). The scaling factor \(\sqrt{1-\beta_t}\) ensures the variance does not explode --- at each step, the signal is slightly attenuated and noise is added, gradually converting signal to noise.

A crucial property: the multi-step transition can be computed in closed form:

$$q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, (1-\bar{\alpha}_t)I)$$

where \(\alpha_t = 1 - \beta_t\) and \(\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s\). This is the Chapman-Kolmogorov equation in action --- we can jump from \(x_0\) directly to \(x_t\) without simulating the intermediate steps, because the composition of Gaussians is still Gaussian.

### The Reverse Process

The generative (denoising) process runs the Markov chain backward:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

This is also a Markov chain: \(x_{t-1}\) depends only on \(x_t\) and the learned mean \(\mu_\theta\). The neural network parameterizes the transition kernel of a reverse-time Markov chain.

The key theoretical result (Anderson, 1982): for a continuous diffusion process, the reverse-time process is also a diffusion process (and hence Markov), with a drift term that depends on the **score function** \(\nabla_{x_t} \log q(x_t)\) --- the gradient of the log probability density. The neural network learns to approximate this score function.

### The Stationary Distribution Connection

At \(t = T\), the forward process has reached (approximately) its stationary distribution: the standard Gaussian \(\mathcal{N}(0, I)\). This is the equilibrium of the noising Markov chain, analogous to the stationary distribution \(\pi\) we derived earlier.

Generation starts from this stationary distribution and runs the learned reverse Markov chain. The reverse chain's stationary distribution is (ideally) the data distribution \(q(x_0)\). The mixing time of the reverse chain determines how many denoising steps are needed --- this is why faster samplers (DDIM, DPM-Solver) that reduce the number of steps are so valuable.

The entire framework is a direct application of Markov chain theory: construct a chain with a known stationary distribution (the noise), learn the reverse chain, and use it to generate samples from the data distribution.

---

## Python Simulations

### Simulation 1: Random Walks in 1D and 2D

```python
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1D Random Walk
np.random.seed(42)
n_steps = 1000
n_walks = 5

ax = axes[0]
for i in range(n_walks):
    steps = np.random.choice([-1, 1], size=n_steps)
    position = np.cumsum(steps)
    ax.plot(range(n_steps), position, linewidth=0.8, alpha=0.7)

# Plot ±sqrt(t) envelope
t = np.arange(1, n_steps + 1)
ax.plot(t, np.sqrt(t), 'k--', linewidth=1.5, label=r'$\pm\sqrt{t}$')
ax.plot(t, -np.sqrt(t), 'k--', linewidth=1.5)
ax.fill_between(t, -np.sqrt(t), np.sqrt(t), alpha=0.1, color='gray')
ax.set_xlabel(r'Time step $t$')
ax.set_ylabel(r'Position $X_t$')
ax.set_title(r'1D Random Walk (5 trajectories)')
ax.legend()
ax.grid(True, alpha=0.3)

# 2D Random Walk
ax2 = axes[1]
np.random.seed(123)
n_steps_2d = 5000

for i in range(3):
    directions = np.random.choice(4, size=n_steps_2d)
    dx = np.where(directions == 0, 1, np.where(directions == 1, -1, 0))
    dy = np.where(directions == 2, 1, np.where(directions == 3, -1, 0))
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    ax2.plot(x, y, linewidth=0.5, alpha=0.6)
    ax2.plot(x[0], y[0], 'go', markersize=6)
    ax2.plot(x[-1], y[-1], 'rs', markersize=6)

ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$y$')
ax2.set_title(r'2D Random Walk (3 trajectories, 5000 steps)')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('random_walks.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Simulation 2: Convergence to Stationary Distribution

```python
import numpy as np
import matplotlib.pyplot as plt

# 3-state Markov chain
P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.1, 0.3, 0.6]
])

# Compute stationary distribution (left eigenvector with eigenvalue 1)
eigenvalues, eigenvectors = np.linalg.eig(P.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
pi_stationary = np.real(eigenvectors[:, idx])
pi_stationary = pi_stationary / pi_stationary.sum()
print(f"Stationary distribution: {pi_stationary}")

# Track convergence from different initial distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: distribution over time from different starts
initial_dists = [
    np.array([1.0, 0.0, 0.0]),  # Start in state 1
    np.array([0.0, 1.0, 0.0]),  # Start in state 2
    np.array([0.0, 0.0, 1.0]),  # Start in state 3
]
colors = ['#2563eb', '#dc2626', '#16a34a']
labels = ['Start in state 1', 'Start in state 2', 'Start in state 3']
n_steps = 30

ax = axes[0]
for pi0, color, label in zip(initial_dists, colors, labels):
    pi_t = pi0.copy()
    history = [pi_t.copy()]
    for _ in range(n_steps):
        pi_t = pi_t @ P
        history.append(pi_t.copy())
    history = np.array(history)

    # Plot probability of state 1 over time
    ax.plot(range(n_steps + 1), history[:, 0], '-', color=color,
            linewidth=1.5, label=label)

ax.axhline(y=pi_stationary[0], color='black', linestyle='--',
           label=r'$\pi_1 = $' + f'{pi_stationary[0]:.3f}')
ax.set_xlabel(r'Time step $t$')
ax.set_ylabel(r'$P(\text{state } 1)$')
ax.set_title(r'Convergence to Stationary Distribution (State 1)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: total variation distance over time
ax2 = axes[1]
for pi0, color, label in zip(initial_dists, colors, labels):
    pi_t = pi0.copy()
    tv_distances = []
    for _ in range(n_steps):
        pi_t = pi_t @ P
        tv = 0.5 * np.sum(np.abs(pi_t - pi_stationary))
        tv_distances.append(tv)

    ax2.semilogy(range(1, n_steps + 1), tv_distances, '-', color=color,
                 linewidth=1.5, label=label)

ax2.set_xlabel(r'Time step $t$')
ax2.set_ylabel(r'Total Variation Distance $d(t)$')
ax2.set_title(r'Convergence Rate (Log Scale)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('markov_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
```

The total variation distance decays exponentially, with the rate determined by the spectral gap. The eigenvalues of this transition matrix are \(\{1.0, \lambda_2, \lambda_3\}\), and the convergence rate is \(|\lambda_2|^t\).

### Simulation 3: Metropolis-Hastings for a 2D Gaussian Mixture

```python
import numpy as np
import matplotlib.pyplot as plt

# Target distribution: mixture of 3 Gaussians
def log_target(x):
    """Log of unnormalized target density (mixture of Gaussians)."""
    mu1, mu2, mu3 = np.array([-2, -2]), np.array([2, 2]), np.array([-1, 3])
    sigma = 0.8
    p1 = np.exp(-0.5 * np.sum((x - mu1)**2) / sigma**2)
    p2 = np.exp(-0.5 * np.sum((x - mu2)**2) / sigma**2)
    p3 = np.exp(-0.5 * np.sum((x - mu3)**2) / sigma**2)
    return np.log(p1 + p2 + p3 + 1e-300)

# Metropolis-Hastings with random walk proposal
def metropolis_hastings(log_target, n_samples, proposal_std, x0):
    d = len(x0)
    samples = np.zeros((n_samples, d))
    x = x0.copy()
    n_accepted = 0

    for i in range(n_samples):
        # Propose
        x_proposal = x + proposal_std * np.random.randn(d)

        # Accept/reject
        log_alpha = log_target(x_proposal) - log_target(x)
        if np.log(np.random.rand()) < log_alpha:
            x = x_proposal
            n_accepted += 1

        samples[i] = x

    acceptance_rate = n_accepted / n_samples
    return samples, acceptance_rate

np.random.seed(42)
n_samples = 50000
x0 = np.array([0.0, 0.0])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
proposal_stds = [0.1, 0.8, 5.0]
titles_suffix = ['too small', 'good', 'too large']

# True density for contour plot
xg = np.linspace(-5, 5, 200)
yg = np.linspace(-5, 6, 200)
Xg, Yg = np.meshgrid(xg, yg)
Zg = np.zeros_like(Xg)
for i in range(Xg.shape[0]):
    for j in range(Xg.shape[1]):
        Zg[i, j] = np.exp(log_target(np.array([Xg[i, j], Yg[i, j]])))

for ax, std, tsuf in zip(axes, proposal_stds, titles_suffix):
    samples, acc_rate = metropolis_hastings(log_target, n_samples, std, x0)

    ax.contour(Xg, Yg, Zg, levels=10, cmap='Blues', alpha=0.5)
    ax.plot(samples[:2000, 0], samples[:2000, 1], 'r.', markersize=0.5, alpha=0.3)
    ax.set_title(r'$\sigma = $' + f'{std} ({tsuf})' + f'\nAccept rate: {acc_rate:.1%}', fontsize=11)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 6)
    ax.set_aspect('equal')

plt.suptitle(r'Metropolis-Hastings: Effect of Proposal Scale $\sigma$', fontsize=14)
plt.tight_layout()
plt.savefig('metropolis_hastings.png', dpi=150, bbox_inches='tight')
plt.show()
```

With \(\sigma = 0.1\), the chain explores very slowly --- it discovers the nearest mode and stays there for a long time before finding others (high acceptance, low movement). With \(\sigma = 5.0\), most proposals land in low-probability regions and are rejected (low acceptance, occasional large jumps). With \(\sigma = 0.8\), the chain efficiently explores all three modes.

### Simulation 4: Mixing Time and the Spectral Gap

```python
import numpy as np
import matplotlib.pyplot as plt

def make_chain(epsilon):
    """Create a 2-state chain with controllable mixing rate.
    P = [[1-epsilon, epsilon], [epsilon, 1-epsilon]]
    Spectral gap = 2*epsilon, mixing time ~ 1/(2*epsilon)
    """
    return np.array([[1-epsilon, epsilon], [epsilon, 1-epsilon]])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epsilons = [0.5, 0.1, 0.01]
colors = ['#2563eb', '#dc2626', '#16a34a']
n_steps = 200

ax = axes[0]
for eps, color in zip(epsilons, colors):
    P = make_chain(eps)
    pi_t = np.array([1.0, 0.0])  # Start in state 1
    pi_stat = np.array([0.5, 0.5])
    tv_distances = []
    for _ in range(n_steps):
        pi_t = pi_t @ P
        tv = 0.5 * np.sum(np.abs(pi_t - pi_stat))
        tv_distances.append(tv)

    ax.semilogy(range(1, n_steps + 1), tv_distances, color=color,
                linewidth=1.5, label=r'$\varepsilon = $' + f'{eps}, gap = {2*eps:.2f}')

ax.set_xlabel(r'Time step $t$')
ax.set_ylabel(r'Total Variation Distance $d(t)$')
ax.set_title(r'Mixing Time vs Spectral Gap $\gamma$')
ax.legend()
ax.grid(True, alpha=0.3)

# Right: simulate chains and show histogram convergence
ax2 = axes[1]
eps = 0.05
P = make_chain(eps)
n_chains = 10000
n_sim_steps = [5, 20, 100]

for t, alpha_val in zip(n_sim_steps, [0.3, 0.6, 1.0]):
    # Start all chains in state 0, run for t steps
    states = np.zeros(n_chains, dtype=int)
    for _ in range(t):
        transitions = np.random.rand(n_chains)
        for i in range(n_chains):
            if transitions[i] < P[states[i], 1]:
                states[i] = 1 - states[i]

    frac_state1 = np.mean(states == 1)
    ax2.bar([f't={t}'], [1 - frac_state1], color='#2563eb', alpha=alpha_val,
            label=r'$t=$' + f'{t}: ' + r'$P(\mathrm{state}\;0) = $' + f'{1-frac_state1:.2f}' if t == n_sim_steps[0] else None)
    ax2.bar([f't={t}'], [frac_state1], bottom=[1-frac_state1], color='#dc2626',
            alpha=alpha_val)

ax2.axhline(y=0.5, color='black', linestyle='--', label=r'Stationary ($\pi = 0.5$)')
ax2.set_ylabel(r'Fraction of chains')
ax2.set_title(r'Distribution Over ' + f'{n_chains}' + r' Chains ($\varepsilon = $' + f'{eps})')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig('mixing_time.png', dpi=150, bbox_inches='tight')
plt.show()
```

The chain with \(\epsilon = 0.5\) (spectral gap = 1.0) mixes in a single step --- it is actually i.i.d. sampling. The chain with \(\epsilon = 0.01\) (spectral gap = 0.02) takes about 50 steps to mix. The relationship is clear: mixing time is inversely proportional to the spectral gap.

---

## Conclusion

Stochastic processes and Markov chains provide the mathematical language for describing systems that evolve randomly over time. The key ideas:

1. **A stochastic process** is a collection of random variables indexed by time. Random walks are the canonical example, exhibiting diffusive \(\sqrt{t}\) spreading and dimension-dependent recurrence.

2. **The Markov property** --- the future depends only on the present --- reduces the complexity of specifying a process from exponential to constant in the history length. This is what makes analysis tractable.

3. **Transition matrices** encode the one-step dynamics, and the Chapman-Kolmogorov equation gives multi-step behavior through matrix powers.

4. **Stationary distributions** are left eigenvectors of the transition matrix with eigenvalue 1. Under ergodicity conditions, the chain converges to the stationary distribution regardless of initialization.

5. **Detailed balance** provides a sufficient condition for stationarity and is the design principle behind MCMC algorithms.

6. **Mixing time** --- how long until the chain approximately reaches stationarity --- is controlled by the spectral gap and determines the practical efficiency of MCMC sampling.

7. **Metropolis-Hastings** constructs a Markov chain with any desired stationary distribution by carefully choosing acceptance probabilities to satisfy detailed balance. The proposal scale controls the exploration-exploitation tradeoff.

8. **Diffusion models** are Markov chains in disguise. The forward process is a noise-adding Markov chain converging to a Gaussian stationary distribution. The reverse process is a learned Markov chain whose stationary distribution is the data distribution. The number of denoising steps is essentially the mixing time of the reverse chain.

These concepts form the theoretical backbone of generative modeling. When you run a diffusion model to generate a video, you are running a learned Markov chain, and the quality of the output depends on how well that chain mixes and how accurately it approximates the true reverse process.
