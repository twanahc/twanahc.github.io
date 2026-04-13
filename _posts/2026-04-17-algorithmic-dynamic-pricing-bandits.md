---
layout: post
title: "Algorithmic Dynamic Pricing: Bandits, Bayesian Learning, and Pricing at Scale"
date: 2026-04-17
category: business
---

*This is Part 5 of a 5-part series on pricing strategy. [Part 1: Demand, Elasticity & Markup](/2026/04/13/demand-elasticity-monopolist-markup.html) | [Part 2: Price Discrimination](/2026/04/14/price-discrimination-extracting-surplus.html) | [Part 3: Game Theory of Pricing](/2026/04/15/game-theory-competitive-pricing.html) | [Part 4: Causal Demand Estimation](/2026/04/16/causal-demand-estimation-ml.html) | **Part 5: Algorithmic Dynamic Pricing***

We've built the full theoretical stack. Part 1 gave us the Lerner index: the optimal markup given known elasticity is \(1/|\varepsilon|\). Part 2 showed how to extract more surplus through price discrimination — segment customers, charge each segment closer to their willingness to pay. Part 3 analyzed how competitors warp the game, driving margins toward zero in Bertrand competition but sustaining them through differentiation and repeated interaction. Part 4 showed how to estimate demand from observational data using causal methods, solving the price endogeneity problem that ruins naive regressions. But there is a final problem we haven't addressed: what if you don't know the demand function at all when you start, and you need to *learn* it while simultaneously *earning* revenue?

Every price you set is both a business decision and an experiment. Charge too high and you miss sales — customers walk away and you observe nothing about what they would have paid at a lower price. Charge too low and you leave money on the table — you make the sale but never learn that the customer would have paid more. Experiment too much and your cumulative revenue suffers while you gather data. Experiment too little and you stay stuck at a suboptimal price forever, never discovering the peak of the revenue curve.

This is the **explore-exploit tradeoff**, and it is the problem that Amazon, Uber, airlines, and every serious e-commerce platform solve every second of every day. The mathematical framework is **multi-armed bandits**, and this post develops it from scratch — the formal theory, the key algorithms, the Bayesian machinery — then shows how it powers dynamic pricing at planetary scale. By the end, we'll connect every piece of the five-part series into the unified pipeline that runs modern algorithmic pricing.

---

## Table of Contents

1. [The Explore-Exploit Tradeoff](#1-the-explore-exploit-tradeoff)
2. [Multi-Armed Bandits: Formal Framework](#2-multi-armed-bandits-formal-framework)
3. [Upper Confidence Bound (UCB)](#3-upper-confidence-bound-ucb) — *with full proof of the regret bound*
4. [Thompson Sampling](#4-thompson-sampling) — *with information-theoretic analysis (Russo & Van Roy)*
5. [Bayesian Dynamic Pricing with Conjugate Demand Models](#45-bayesian-dynamic-pricing-with-conjugate-demand-models)
6. [Contextual Bandits for Personalized Pricing](#5-contextual-bandits-for-personalized-pricing)
7. [Continuous-Armed Bandits and Lipschitz Optimization](#6-continuous-armed-bandits-and-lipschitz-optimization)
8. [Gaussian Process Bandits and Bayesian Optimization for Pricing](#7-gaussian-process-bandits-and-bayesian-optimization-for-pricing)
9. [Non-Stationary Bandits — When Demand Shifts](#8-non-stationary-bandits--when-demand-shifts)
10. [Adversarial Bandits — When Competitors Fight Back](#9-adversarial-bandits--when-competitors-fight-back)
11. [Multi-Product Dynamic Pricing](#10-multi-product-dynamic-pricing)
12. [Deep Reinforcement Learning for Pricing](#11-deep-reinforcement-learning-for-pricing)
13. [Offline Reinforcement Learning for Pricing](#115-offline-reinforcement-learning-for-pricing)
14. [Fairness Constraints in Algorithmic Pricing](#12-fairness-constraints-in-algorithmic-pricing)
15. [Python — GP-Bandit and Non-Stationary Demand](#13-python--gp-bandit-and-non-stationary-demand)
16. [The Pricing MDP: When Bandits Aren't Enough](#14-the-pricing-mdp-when-bandits-arent-enough)
17. [Airline Revenue Management](#15-airline-revenue-management) — *with network revenue management*
18. [Uber's Surge Pricing](#16-ubers-surge-pricing)
19. [Amazon's Pricing Engine](#17-amazons-pricing-engine)
20. [The Buy Box Game](#18-the-buy-box-game)
21. [Implementation at Scale](#19-implementation-at-scale)
22. [Python Simulations](#20-python-simulations) — *including Bayesian dynamic pricing simulation*
23. [The Complete Stack](#21-the-complete-stack)

---

## 1. The Explore-Exploit Tradeoff

Here is the setup. You have a product to sell — say a new SaaS tool. You don't know the demand curve. You have no historical pricing data. Each day, you set a price and observe whether customers buy. That's all the information you get: a price, and the resulting sales count.

If you always charge the same price — say $29/month because it "feels right" — you learn *nothing* about demand at other price points. Maybe $49/month would yield fewer customers but higher revenue. Maybe $19/month would flood you with so many users that revenue doubles. You'll never know, because you never tested.

If you experiment wildly with different prices — $5 on Monday, $99 on Tuesday, $37 on Wednesday — you gather a lot of information about the demand curve, but your cumulative revenue is terrible. You spent days charging $5 when the optimal price might be $45. Your investors are unhappy.

The core tension is between **exploitation** — charging your current best estimate of the optimal price — and **exploration** — trying other prices to improve your estimate. This is *not* a traditional optimization problem. In standard optimization, you evaluate the objective function (revenue as a function of price) and then choose the maximum. But here, every evaluation is costly — it's a real price you charged to a real customer. And each evaluation generates two things simultaneously: a **reward** (revenue) and **information** (data about the demand curve at that price point). The two objectives are in direct conflict.

The formal measure of performance is **regret**. Suppose the true optimal price is \(p^*\) and it generates expected revenue \(r^* = r(p^*)\) per period. At each time step \(t\), you choose price \(p_t\) and earn expected revenue \(r(p_t)\). The **cumulative regret** after \(T\) periods is:

$$
R_T = \sum_{t=1}^{T} \left[ r(p^*) - r(p_t) \right]
$$

This is the total revenue you lost, compared to an oracle that knows the optimal price from the start. A good algorithm has regret that grows **sublinearly** in \(T\) — meaning the per-period regret \(R_T / T \to 0\). You make worse decisions early on, but you learn, and eventually your decisions approach optimal. The best algorithms achieve \(O(\sqrt{T})\) regret, which means per-period regret decays like \(O(1/\sqrt{T})\). After 10,000 rounds, you're losing roughly 1% per period. After 1,000,000 rounds, roughly 0.1%.

An algorithm with *linear* regret — \(R_T = \Theta(T)\) — is one that never learns. It keeps making the same mistake every period. A constant-price strategy that happens to be wrong has linear regret. Our goal is to do much better.

---

## 2. Multi-Armed Bandits: Formal Framework

The name comes from slot machines, which are colloquially called "one-armed bandits" (because they rob you). Imagine you're standing in front of \(K\) slot machines, each with a different, unknown payout distribution. You can pull one arm per round and observe the reward. Over \(T\) rounds, which arms should you pull to maximize your total payout?

For pricing: each candidate price point is an "arm." The reward from pulling arm \(k\) is the revenue from selling at price \(p_k\). You choose one price per period, observe sales, and compute revenue.

Formally, we have:

- \(K\) arms (price points), indexed \(k = 1, 2, \ldots, K\)
- Arm \(k\) has an unknown reward distribution \(\mathcal{D}_k\) with mean \(\mu_k\)
- At time \(t\), the learner chooses arm \(A_t \in \{1, \ldots, K\}\)
- The learner observes reward \(X_t \sim \mathcal{D}_{A_t}\), drawn independently
- The **optimal arm** is \(k^* = \arg\max_k \mu_k\), with mean reward \(\mu^* = \mu_{k^*}\)

The cumulative regret becomes:

$$
R_T = T \cdot \mu^* - \sum_{t=1}^{T} \mu_{A_t} = \sum_{t=1}^{T} (\mu^* - \mu_{A_t})
$$

Define the **gap** for arm \(k\) as \(\Delta_k = \mu^* - \mu_k\). This is how much worse arm \(k\) is compared to the best. Regret can be rewritten as:

$$
R_T = \sum_{k=1}^{K} \Delta_k \cdot \mathbb{E}[N_k(T)]
$$

where \(N_k(T)\) is the number of times arm \(k\) has been pulled up to time \(T\). The regret is large when you pull suboptimal arms (high \(\Delta_k\)) many times (high \(N_k(T)\)).

There's a fundamental lower bound, due to Lai and Robbins (1985), that constrains *any* algorithm. For any policy that achieves sublinear regret on all bandit instances:

$$
R_T \geq \sum_{k: \mu_k < \mu^*} \frac{\Delta_k}{\text{KL}(\mu_k \| \mu^*)} \cdot \ln T
$$

where \(\text{KL}(\mu_k \| \mu^*)\) is the **Kullback-Leibler divergence** between the reward distributions of arm \(k\) and the optimal arm. The KL divergence measures how statistically "distinguishable" two distributions are — if arm \(k\) and the optimal arm have very similar reward distributions, you need many samples to tell them apart, and you incur more regret. This gives a **logarithmic lower bound** on regret: even the best possible algorithm cannot do better than \(\Omega(\ln T)\).

So the target is clear: find algorithms whose regret grows like \(O(\ln T)\) — matching this lower bound. Let's look at two that achieve it.

---

## 3. Upper Confidence Bound (UCB)

The UCB family of algorithms is built on a beautiful principle: **optimism in the face of uncertainty**. The idea is: when you're unsure about an arm, give it the benefit of the doubt. Assume it's as good as your data allows. This automatically balances exploration and exploitation — arms you know well have tight estimates, and arms you've barely tried have wide confidence intervals that extend upward, making them attractive to try.

The **UCB1** algorithm (Auer, Cesa-Bianchi, and Fischer, 2002) works as follows:

1. **Initialize**: pull each arm once.
2. **At time \(t\)**: choose the arm \(k\) that maximizes:

$$
\text{UCB}_k(t) = \hat{\mu}_k(t) + \sqrt{\frac{2 \ln t}{N_k(t)}}
$$

where \(\hat{\mu}_k(t)\) is the sample mean reward of arm \(k\) from the \(N_k(t)\) times it has been pulled.

The first term, \(\hat{\mu}_k(t)\), is **exploitation** — it favors the arm with the highest observed average reward. The second term, \(\sqrt{2 \ln t / N_k(t)}\), is **exploration** — it's large when \(N_k(t)\) is small (the arm hasn't been tried much) and shrinks as you gather more data about arm \(k\).

Why this specific form? It comes from **Hoeffding's inequality**, a concentration inequality that bounds how far a sample mean can deviate from the true mean. For bounded rewards in \([0, 1]\), Hoeffding's inequality says:

$$
\Pr\left(|\hat{\mu}_k - \mu_k| \geq c\right) \leq 2 \exp\left(-2 N_k c^2\right)
$$

We want this probability to be at most \(1/t^2\) (a schedule that decays fast enough to give logarithmic regret). Setting \(2 \exp(-2 N_k c^2) = 1/t^2\) and solving:

$$
c = \sqrt{\frac{2 \ln t}{N_k}}
$$

So the UCB is an **upper confidence bound** at roughly the \(1/t^2\) significance level. With high probability, the true mean \(\mu_k\) is below the UCB. By being optimistic about uncertain arms, UCB1 is guaranteed to explore them enough to either confirm they're good (and exploit them) or discover they're bad (and stop pulling them).

**Regret guarantee**: UCB1 achieves:

$$
R_T \leq \sum_{k: \Delta_k > 0} \left(\frac{8 \ln T}{\Delta_k} + (1 + \frac{\pi^2}{3}) \Delta_k\right)
$$

This is \(O(K \ln T)\), which matches the Lai-Robbins lower bound up to constants. UCB1 is essentially **optimal** in the minimax sense.

### Full Proof of the UCB1 Regret Bound

The statement above is clean, but the proof is where the real understanding lives. Let's walk through every step, because the argument reveals *why* optimism works and exactly where the logarithmic dependence on \(T\) comes from.

**What we want to bound.** The regret is \(R_T = \sum_k \Delta_k \, \mathbb{E}[N_k(T)]\), so it suffices to bound \(\mathbb{E}[N_k(T)]\) — the expected number of times each suboptimal arm \(k\) is pulled.

**Step 1: Decomposing the pulls of a suboptimal arm.** Fix a suboptimal arm \(k\) with gap \(\Delta_k = \mu^* - \mu_k > 0\). Choose a threshold:

$$
\ell = \left\lceil \frac{8 \ln T}{\Delta_k^2} \right\rceil
$$

We split the total pulls of arm \(k\) into "early" pulls (the first \(\ell\) times) and "late" pulls (every time after that):

$$
N_k(T) = \ell + \sum_{t=K+1}^{T} \mathbf{1}\{A_t = k, \; N_k(t-1) \geq \ell\}
$$

The first \(\ell\) pulls are "budgeted" — we accept them as the cost of gathering enough data about arm \(k\). The key is bounding the second term: how often does UCB1 pull arm \(k\) even after it has been tried \(\ell\) times?

**Step 2: When does arm \(k\) beat the optimal arm?** Arm \(k\) is pulled at time \(t\) only if its UCB index exceeds the UCB index of the optimal arm \(k^*\):

$$
\hat{\mu}_k(t-1) + \sqrt{\frac{2 \ln t}{N_k(t-1)}} \geq \hat{\mu}_{k^*}(t-1) + \sqrt{\frac{2 \ln t}{N_{k^*}(t-1)}}
$$

For this to happen when \(N_k(t-1) \geq \ell\), at least one of the following "bad events" must occur:

- **Event \(E_1\)**: The sample mean of arm \(k\) is inflated above its true mean plus the confidence width: \(\hat{\mu}_k \geq \mu_k + \sqrt{2 \ln t / N_k}\).
- **Event \(E_2\)**: The sample mean of the optimal arm is deflated below its true mean minus the confidence width: \(\hat{\mu}_{k^*} \leq \mu^* - \sqrt{2 \ln t / N_{k^*}}\).
- **Event \(E_3\)**: Both confidence intervals are correct (arm \(k\)'s true mean is within its CI, optimal arm's true mean is within its CI), but arm \(k\)'s inflated upper bound still beats the optimal arm's upper bound.

**Step 3: Ruling out Event \(E_3\).** Suppose both confidence intervals hold:

$$
\hat{\mu}_k \leq \mu_k + \sqrt{\frac{2 \ln t}{N_k}}, \qquad \hat{\mu}_{k^*} \geq \mu^* - \sqrt{\frac{2 \ln t}{N_{k^*}}}
$$

Then the UCB of arm \(k\) is bounded above by:

$$
\text{UCB}_k = \hat{\mu}_k + \sqrt{\frac{2 \ln t}{N_k}} \leq \mu_k + 2\sqrt{\frac{2 \ln t}{N_k}}
$$

And the UCB of the optimal arm is bounded below by:

$$
\text{UCB}_{k^*} = \hat{\mu}_{k^*} + \sqrt{\frac{2 \ln t}{N_{k^*}}} \geq \mu^*
$$

For arm \(k\) to be pulled, we need \(\text{UCB}_k \geq \text{UCB}_{k^*}\), which requires:

$$
\mu_k + 2\sqrt{\frac{2 \ln t}{N_k}} \geq \mu^*
$$

Rearranging: \(2\sqrt{2 \ln t / N_k} \geq \Delta_k\), which means:

$$
N_k \leq \frac{8 \ln t}{\Delta_k^2} \leq \frac{8 \ln T}{\Delta_k^2} \leq \ell
$$

But we assumed \(N_k \geq \ell\). Contradiction! So when \(N_k \geq \ell\) and both confidence intervals hold, arm \(k\) *cannot* be pulled. Event \(E_3\) is impossible.

This is the crux of the proof. The threshold \(\ell = \lceil 8 \ln T / \Delta_k^2 \rceil\) is chosen precisely so that after \(\ell\) observations, the exploration bonus of arm \(k\) is small enough that its optimistic estimate cannot exceed the true mean of the optimal arm — *provided* the confidence intervals haven't failed.

**Step 4: Bounding the probability of confidence interval failure.** We need to bound the probability of events \(E_1\) and \(E_2\). By **Hoeffding's inequality**, for rewards bounded in \([0, 1]\):

$$
\Pr\left(\hat{\mu}_k - \mu_k \geq \sqrt{\frac{2 \ln t}{s}}\right) \leq \exp\left(-2s \cdot \frac{2 \ln t}{s}\right) = \exp(-4 \ln t) = t^{-4}
$$

where \(s = N_k\) is the number of observations. This is a one-sided bound. Similarly for the lower tail. So:

$$
\Pr(E_1 \text{ for a specific } (t, s) \text{ pair}) \leq t^{-4}
$$

$$
\Pr(E_2 \text{ for a specific } (t, s') \text{ pair}) \leq t^{-4}
$$

**Step 5: The union bound over all time steps and sample sizes.** The indicator \(\mathbf{1}\{A_t = k, N_k(t-1) \geq \ell\}\) can only be 1 if \(E_1\) or \(E_2\) holds at time \(t\) for some sample sizes \(s \in \{\ell, \ldots, t-1\}\) for arm \(k\) and \(s' \in \{1, \ldots, t-1\}\) for the optimal arm. Taking a union bound:

$$
\sum_{t=K+1}^{T} \Pr(A_t = k, N_k(t-1) \geq \ell) \leq \sum_{t=1}^{T} \sum_{s=\ell}^{t-1} t^{-4} + \sum_{t=1}^{T} \sum_{s'=1}^{t-1} t^{-4}
$$

Each inner sum has at most \(t\) terms, so:

$$
\leq \sum_{t=1}^{T} t \cdot t^{-4} + \sum_{t=1}^{T} t \cdot t^{-4} = 2\sum_{t=1}^{T} t^{-3}
$$

The series \(\sum_{t=1}^{\infty} t^{-3}\) converges to \(\zeta(3) \approx 1.202\), and is bounded by \(\pi^2/6 \approx 1.645\). So:

$$
\sum_{t=K+1}^{T} \Pr(A_t = k, N_k(t-1) \geq \ell) \leq \frac{\pi^2}{3}
$$

**Step 6: Putting it all together.** Combining the early and late pulls:

$$
\mathbb{E}[N_k(T)] \leq \ell + \frac{\pi^2}{3} = \left\lceil \frac{8 \ln T}{\Delta_k^2} \right\rceil + \frac{\pi^2}{3} \leq \frac{8 \ln T}{\Delta_k^2} + 1 + \frac{\pi^2}{3}
$$

Multiplying by the gap \(\Delta_k\) and summing over all suboptimal arms:

$$
R_T = \sum_{k: \Delta_k > 0} \Delta_k \, \mathbb{E}[N_k(T)] \leq \sum_{k: \Delta_k > 0} \left(\frac{8 \ln T}{\Delta_k} + \left(1 + \frac{\pi^2}{3}\right) \Delta_k\right)
$$

This is the stated bound. The first term is the dominant one — it grows logarithmically with \(T\) and inversely with \(\Delta_k\). Arms that are nearly as good as the optimal arm (\(\Delta_k\) small) are harder to distinguish and contribute more regret. Arms that are clearly suboptimal (\(\Delta_k\) large) are quickly identified and contribute little regret. The second term is a constant that doesn't grow with \(T\) — it's the "price of admission" from the union bound.

**The deep insight.** The proof reveals *why* the logarithmic bound works. The exploration bonus \(\sqrt{2 \ln t / N_k}\) decays as \(1/\sqrt{N_k}\), so after \(O(\ln T / \Delta_k^2)\) pulls, it's small enough that a suboptimal arm can no longer compete with the optimal arm (Step 3). The confidence intervals fail with probability \(O(t^{-4})\) per time step, and summing this geometric tail over all \(T\) rounds gives a convergent series (Step 5). The \(\ln T\) in the final bound comes from the \(\ln t\) in the exploration bonus — if we used a larger bonus (say \(\sqrt{\ln^2 t / N_k}\)), we'd get a worse regret bound; if we used a smaller one, the confidence intervals would fail too often.

**For pricing**: discretize the price space into \(K\) prices (e.g., $9.99, $10.99, $11.99, ..., $29.99). Run UCB1, treating each price as an arm. The algorithm will initially try each price once, then gradually focus on the revenue-maximizing price while occasionally revisiting others to tighten its estimates.

**Limitation**: UCB1 treats each price as an independent arm. It doesn't exploit the structure that nearby prices should have similar expected revenues — if $14.99 generates high revenue, $15.99 probably does too. This is wasteful. We'll address it with contextual bandits and parametric models later, but first, let's look at the Bayesian alternative.

---

## 4. Thompson Sampling

Thompson Sampling, first proposed by Thompson in 1933, takes a completely different approach. Instead of constructing confidence bounds, it maintains a **posterior distribution** over the mean reward of each arm and uses randomization to balance exploration and exploitation.

The algorithm is remarkably simple:

1. **Initialize**: set a prior distribution for each arm's mean reward.
2. **At time \(t\)**:
   - For each arm \(k\), draw a sample \(\theta_k \sim \text{Posterior}_k\)
   - Choose the arm with the highest sample: \(A_t = \arg\max_k \theta_k\)
3. **Update**: after observing reward \(X_t\), update the posterior of arm \(A_t\) using Bayes' rule.

For **Bernoulli rewards** — the customer either buys (1) or doesn't (0) at the given price — the natural choice is the **Beta-Bernoulli** model:

- **Prior**: \(\text{Beta}(\alpha_k, \beta_k)\) for arm \(k\). Start with \(\alpha_k = \beta_k = 1\), which is a uniform prior on \([0,1]\) (we have no prior knowledge about the purchase probability).
- **After observing a purchase at price \(k\)**: update \(\alpha_k \leftarrow \alpha_k + 1\)
- **After observing no purchase at price \(k\)**: update \(\beta_k \leftarrow \beta_k + 1\)
- **Posterior mean**: \(\hat{p}_k = \alpha_k / (\alpha_k + \beta_k)\), which is just the empirical purchase rate with a small correction from the prior.

Why does Thompson Sampling work? The mechanism is elegant. When you're **uncertain** about an arm, its posterior is wide — it has high variance. A wide distribution sometimes produces very high samples, which causes the arm to be selected for exploration. As you observe more data, the posterior **concentrates** around the true mean — it becomes narrow. For the best arm, the posterior concentrates around a high value, so it's sampled high most of the time and gets exploited. For bad arms, the posterior concentrates around a low value, so they're rarely sampled high and thus rarely chosen.

Exploration happens *automatically*, proportional to uncertainty. No tuning of exploration parameters is needed (unlike UCB, where the constant in front of the exploration term matters in practice).

**Regret guarantee**: Thompson Sampling achieves \(O(K \ln T)\) regret, matching UCB1 and the Lai-Robbins lower bound. But empirically, it often has **lower constants** — it converges to the optimal arm faster in practice. This has made it the go-to algorithm in industry.

**For pricing with revenue**: the reward from price \(p_k\) is not just the purchase probability — it's the **revenue** \(p_k \times \mathbb{1}[\text{sale}]\). If the purchase probability at price \(p_k\) follows a Beta posterior \(\text{Beta}(\alpha_k, \beta_k)\), then the expected revenue from sampling this arm is \(p_k \times \theta_k\) where \(\theta_k\) is the sampled purchase probability. You draw \(\theta_k \sim \text{Beta}(\alpha_k, \beta_k)\) for each arm, compute \(p_k \times \theta_k\), and choose the arm with the highest product.

Thompson Sampling also provides a natural way to incorporate **prior knowledge**. If you have beliefs about the shape of the demand curve — perhaps from similar products or market research — you encode them in the prior. A strong prior (large \(\alpha\) and \(\beta\)) means you're confident in your initial estimate and the algorithm explores less. A weak prior (small \(\alpha\) and \(\beta\)) means you're uncertain and the algorithm explores more. This is exactly the Bayesian way to handle the cold-start problem.

### Information-Theoretic Analysis: Why Thompson Sampling Works (Russo & Van Roy, 2016)

The regret bound \(O(K \ln T)\) tells us Thompson Sampling performs well, but it doesn't explain *why*. The deepest insight into Thompson Sampling comes from an information-theoretic analysis by Russo and Van Roy (2016) that reveals the fundamental mechanism: Thompson Sampling achieves a near-optimal tradeoff between **regret incurred** and **information gained** at every single round.

**The information ratio.** Define the **information ratio** at round \(t\) as:

$$
\Gamma_t = \frac{\left(\mathbb{E}[\Delta_{A_t} \mid \mathcal{H}_{t-1}]\right)^2}{I_t(k^*; (A_t, R_t) \mid \mathcal{H}_{t-1})}
$$

The numerator is the squared expected instantaneous regret — how much reward we expect to lose this round by not playing the optimal arm. The denominator is the **mutual information** between the identity of the optimal arm \(k^*\) and the observation \((A_t, R_t)\) at round \(t\), conditioned on the history \(\mathcal{H}_{t-1}\).

Mutual information, if you haven't encountered it, measures how much observing one random variable reduces uncertainty about another. Formally, \(I(X; Y) = H(X) - H(X \mid Y)\), where \(H\) is Shannon entropy. In this context, \(I_t\) measures how much the observation at round \(t\) reduces our uncertainty about which arm is optimal.

The information ratio captures the **efficiency of exploration**. A high \(\Gamma_t\) means the algorithm is incurring a lot of regret (numerator large) relative to how much it's learning (denominator small) — wasteful exploration. A low \(\Gamma_t\) means the algorithm is learning a lot per unit of regret incurred — efficient exploration.

**The fundamental inequality.** If the information ratio is bounded — \(\Gamma_t \leq \Gamma\) for all \(t\) — then the Bayesian regret satisfies:

$$
\mathbb{E}[R_T] \leq \sqrt{\Gamma \cdot H(k^*) \cdot T}
$$

where \(H(k^*)\) is the entropy of the prior distribution over the identity of the optimal arm. The proof is elegant. By the chain rule of mutual information:

$$
\sum_{t=1}^{T} I_t(k^*; (A_t, R_t) \mid \mathcal{H}_{t-1}) = I(k^*; \mathcal{H}_T) \leq H(k^*)
$$

The total information gained over all \(T\) rounds is bounded by the entropy of the thing we're trying to learn. This is an information-theoretic budget constraint — you can't learn more about \(k^*\) than the entropy of \(k^*\).

From the definition of \(\Gamma_t\):

$$
\left(\mathbb{E}[\Delta_{A_t} \mid \mathcal{H}_{t-1}]\right)^2 \leq \Gamma \cdot I_t
$$

By Jensen's inequality (square root is concave) and Cauchy-Schwarz:

$$
\mathbb{E}[R_T] = \sum_{t=1}^{T} \mathbb{E}[\Delta_{A_t}] \leq \sum_{t=1}^{T} \sqrt{\Gamma \cdot I_t} \leq \sqrt{\Gamma \cdot T \cdot \sum_{t=1}^{T} I_t} \leq \sqrt{\Gamma \cdot T \cdot H(k^*)}
$$

The second inequality is Cauchy-Schwarz applied to the vectors \((\sqrt{I_1}, \ldots, \sqrt{I_T})\) and \((1, \ldots, 1)\). That's the entire proof.

**Thompson Sampling has a small information ratio.** For \(K\)-armed Bernoulli bandits, Russo and Van Roy show that Thompson Sampling has \(\Gamma \leq K/2\). With a uniform prior, \(H(k^*) = \ln K\). Substituting:

$$
\mathbb{E}[R_T] \leq \sqrt{\frac{K \ln K}{2} \cdot T}
$$

This is near-minimax-optimal — the lower bound for \(K\)-armed bandits is \(\Omega(\sqrt{KT})\), and Thompson Sampling matches it up to a \(\sqrt{\ln K}\) factor.

**Why UCB can be less efficient.** UCB algorithms always play the arm with the highest upper confidence bound. Consider a situation where arm 3 has the highest UCB, but you're already fairly confident about arm 3's mean — the UCB is high only because of a lucky early observation, not because of genuine uncertainty. Playing arm 3 incurs regret (it's suboptimal) and provides little information (you already have many observations of arm 3). The information ratio is poor.

Thompson Sampling avoids this because it samples from the posterior. Arms with narrow posteriors (low uncertainty) are sampled near their true mean — they're played frequently if they're genuinely good, rarely if they're genuinely bad. Arms with wide posteriors (high uncertainty) sometimes produce high samples, triggering exploration, but the exploration is *proportional to the remaining uncertainty*, which is exactly what minimizes the information ratio.

**The deep interpretation.** Thompson Sampling is approximately the algorithm that minimizes the information ratio at each step. It achieves the best tradeoff between regret incurred and information gained — every exploratory pull is maximally informative about which arm is optimal, relative to the regret it costs. This is why Thompson Sampling consistently outperforms UCB in practice: not because its worst-case bound is better (it isn't — both are \(O(K \ln T)\)), but because it wastes less exploration on low-information pulls.

---

## 4.5. Bayesian Dynamic Pricing with Conjugate Demand Models

Before we move to contextual bandits, let's develop the full Bayesian pricing framework that connects Thompson Sampling to parametric demand models. This is the bridge between the "model-free" bandit approach (treat each price as an independent arm) and the "model-based" approach (assume demand follows a parametric form and learn the parameters).

### Setup: The Linear Demand Model

Assume demand \(Q\) at price \(p\) follows a linear model with Gaussian noise:

$$
Q_t = \alpha + \beta p_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)
$$

Here \(\alpha > 0\) is the base demand (demand when price is zero — a theoretical quantity, but it anchors the intercept), \(\beta < 0\) is the **price sensitivity** (each dollar increase in price reduces demand by \(|\beta|\) units), and \(\sigma^2\) is the noise variance (which we assume is known, for analytic tractability — it can be estimated separately).

The revenue at price \(p\) is:

$$
r(p) = p \cdot Q(p) = p(\alpha + \beta p) = \alpha p + \beta p^2
$$

This is a downward-opening parabola (since \(\beta < 0\)), with a unique maximum at:

$$
p^* = -\frac{\alpha}{2\beta}
$$

The challenge: \(\alpha\) and \(\beta\) are **unknown**. We need to learn them from data while simultaneously pricing well.

### The Bayesian Approach: Prior and Posterior

Define the parameter vector \(\boldsymbol{\theta} = (\alpha, \beta)^\top\) and the feature vector at time \(t\) as \(\mathbf{x}_t = (1, p_t)^\top\). The demand model is then:

$$
Q_t = \mathbf{x}_t^\top \boldsymbol{\theta} + \varepsilon_t
$$

This is a standard Bayesian linear regression. Place a **conjugate prior** on \(\boldsymbol{\theta}\):

$$
\boldsymbol{\theta} \sim \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)
$$

where \(\boldsymbol{\mu}_0 = (\mu_\alpha, \mu_\beta)^\top\) is our prior mean (best guess for the demand parameters before seeing any data) and \(\boldsymbol{\Sigma}_0\) is the prior covariance (how uncertain we are).

For example, a reasonable prior might be \(\mu_\alpha = 40\) (we think demand at zero price would be about 40 units), \(\mu_\beta = -0.5\) (we think each dollar of price costs half a unit of demand), and a diagonal covariance with large variances (reflecting high uncertainty).

**The conjugate posterior.** After observing \(T\) price-quantity pairs \(\{(p_1, Q_1), \ldots, (p_T, Q_T)\}\), the posterior is also Gaussian:

$$
\boldsymbol{\theta} \mid \text{data} \sim \mathcal{N}(\boldsymbol{\mu}_T, \boldsymbol{\Sigma}_T)
$$

Let's derive the posterior parameters. Collect the data into matrices: \(\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_T]^\top\) is \(T \times 2\), and \(\mathbf{Q} = (Q_1, \ldots, Q_T)^\top\) is \(T \times 1\). The likelihood is:

$$
p(\mathbf{Q} \mid \boldsymbol{\theta}) \propto \exp\left(-\frac{1}{2\sigma^2}(\mathbf{Q} - \mathbf{X}\boldsymbol{\theta})^\top(\mathbf{Q} - \mathbf{X}\boldsymbol{\theta})\right)
$$

The prior is:

$$
p(\boldsymbol{\theta}) \propto \exp\left(-\frac{1}{2}(\boldsymbol{\theta} - \boldsymbol{\mu}_0)^\top \boldsymbol{\Sigma}_0^{-1} (\boldsymbol{\theta} - \boldsymbol{\mu}_0)\right)
$$

By Bayes' theorem, the posterior is proportional to the product. Expanding the exponents and completing the square (the standard trick for Gaussian conjugacy):

$$
\boldsymbol{\Sigma}_T^{-1} = \boldsymbol{\Sigma}_0^{-1} + \frac{1}{\sigma^2}\mathbf{X}^\top\mathbf{X}
$$

$$
\boldsymbol{\mu}_T = \boldsymbol{\Sigma}_T\left(\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\mu}_0 + \frac{1}{\sigma^2}\mathbf{X}^\top\mathbf{Q}\right)
$$

The posterior precision (inverse covariance) is the sum of the prior precision and the data precision. The posterior mean is a precision-weighted average of the prior mean and the data-driven estimate. As \(T \to \infty\), the data term dominates, and \(\boldsymbol{\mu}_T \to (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{Q}\) — the ordinary least squares estimate. The prior washes out with enough data, as it should.

**Sequential update form.** In an online setting, we observe one data point at a time. After observing \((p_t, Q_t)\) with \(\mathbf{x}_t = (1, p_t)^\top\), the update from posterior at time \(t-1\) to posterior at time \(t\) is:

$$
\boldsymbol{\Sigma}_t^{-1} = \boldsymbol{\Sigma}_{t-1}^{-1} + \frac{1}{\sigma^2}\mathbf{x}_t\mathbf{x}_t^\top
$$

$$
\boldsymbol{\mu}_t = \boldsymbol{\Sigma}_t\left(\boldsymbol{\Sigma}_{t-1}^{-1}\boldsymbol{\mu}_{t-1} + \frac{Q_t}{\sigma^2}\mathbf{x}_t\right)
$$

This is computationally cheap — a rank-1 update to a \(2 \times 2\) matrix at each step.

### Myopic vs. Farsighted Pricing

Given the current posterior \((\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)\), how should we set the next price?

**Myopic pricing** maximizes the expected immediate revenue given the current posterior mean:

$$
\mathbb{E}[r(p) \mid \boldsymbol{\mu}_t] = p \cdot (\mu_\alpha^t + \mu_\beta^t \cdot p) = \mu_\alpha^t \cdot p + \mu_\beta^t \cdot p^2
$$

Taking the first-order condition \(\partial / \partial p = \mu_\alpha^t + 2\mu_\beta^t \cdot p = 0\):

$$
p_{\text{myopic}}^* = -\frac{\mu_\alpha^t}{2\mu_\beta^t}
$$

This is the "certainty equivalent" price — the optimal price if the posterior mean were the true parameters. It completely ignores uncertainty and the value of information. If \(\mu_\beta^t\) is far from the true \(\beta\) (because we haven't gathered enough data yet), the myopic price can be very wrong.

Worse, myopic pricing can get **stuck**. If the initial prior underestimates price sensitivity (|\(\mu_\beta\)| too small), the myopic price is too high. At that high price, demand is low and noisy, providing little information to correct the estimate. The posterior barely updates, and the algorithm stays stuck at a suboptimal price. This is the classic explore-exploit failure: pure exploitation with no exploration.

**Farsighted pricing** accounts for how the current price choice affects the posterior (and hence future pricing decisions). The full solution is a dynamic program:

$$
V_t(\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t) = \max_{p} \left[\mathbb{E}[r(p)] + \gamma \cdot \mathbb{E}_{Q|p}\left[V_{t+1}(\boldsymbol{\mu}_{t+1}, \boldsymbol{\Sigma}_{t+1})\right]\right]
$$

where the expectation is over the demand \(Q\) that we'd observe at price \(p\), and the future value \(V_{t+1}\) depends on the updated posterior after observing \((p, Q)\). This is the **Gittins index** problem — the optimal solution balances the immediate revenue from price \(p\) against the information value of observing demand at \(p\).

Solving this DP exactly is intractable for continuous state spaces (the posterior covariance \(\boldsymbol{\Sigma}_t\) lives in an infinite-dimensional space as \(T\) grows). Two practical approximations are widely used:

### The Knowledge Gradient

The **knowledge gradient** (KG) measures the value of a single additional observation at price \(p\):

$$
\text{KG}(p) = \mathbb{E}\left[\max_{p'} \mathbb{E}[r(p') \mid \boldsymbol{\mu}_{t+1}]\right] - \max_{p'} \mathbb{E}[r(p') \mid \boldsymbol{\mu}_t]
$$

This is the expected improvement in the optimal future revenue from learning at price \(p\). The first term is the best revenue we can achieve with the updated posterior (after observing demand at price \(p\)), averaged over the random observation. The second term is the best revenue under the current posterior. The difference is the **information value** of experimenting at price \(p\).

For the Normal linear demand model, the posterior update after observing \(Q\) at price \(p\) gives a new posterior mean that is itself random (it depends on the noisy observation \(Q\)). The KG has a semi-closed form involving the standard normal CDF \(\Phi\) and PDF \(\phi\):

$$
\text{KG}(p) = \tilde{\sigma}(p) \cdot \nu\!\left(\frac{\tilde{\sigma}(p)}{\sigma_0}\right)
$$

where \(\tilde{\sigma}(p)\) measures how much the observation at price \(p\) reduces posterior uncertainty about the optimal price, and \(\nu(z) = z\Phi(z) + \phi(z)\) is the **knowledge gradient factor** — a function that captures the expected improvement from a Gaussian observation.

The KG policy chooses price \(p\) to maximize:

$$
p_{\text{KG}} = \arg\max_p \left[r(p) + \nu_{\text{KG}} \cdot \text{KG}(p)\right]
$$

where \(\nu_{\text{KG}}\) is a parameter controlling the exploitation-exploration tradeoff. When \(\nu_{\text{KG}} = 0\), this reduces to myopic pricing; when \(\nu_{\text{KG}} \to \infty\), the algorithm explores purely for information.

### Thompson Sampling for Bayesian Pricing

Thompson Sampling provides a simpler and often equally effective alternative. At each round:

1. Draw \(\boldsymbol{\theta}_{\text{sample}} = (\alpha_{\text{sample}}, \beta_{\text{sample}}) \sim \mathcal{N}(\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)\)
2. Compute the optimal price for the sampled parameters: \(p_t = -\alpha_{\text{sample}} / (2\beta_{\text{sample}})\)
3. Observe demand \(Q_t\) at price \(p_t\)
4. Update the posterior using the sequential update formulas

This is elegant because it automatically explores in proportion to posterior uncertainty. When the posterior is wide (early rounds), the sampled parameters vary widely, producing diverse prices — exploration. As the posterior concentrates (later rounds), the sampled parameters cluster near the truth, and the prices cluster near the optimal — exploitation.

The connection to the knowledge gradient: Thompson Sampling implicitly computes something close to the KG by sampling from the posterior. The KG explicitly evaluates information value; Thompson Sampling achieves a similar effect through randomization. In practice, Thompson Sampling is simpler to implement and performs nearly as well, making it the preferred choice for Bayesian dynamic pricing.

### Worked Numerical Example

Suppose the true parameters are \(\alpha = 50\), \(\beta = -1.2\), and \(\sigma = 5\). The true optimal price is:

$$
p^* = -\frac{50}{2 \times (-1.2)} = 20.83
$$

We start with a vague prior: \(\boldsymbol{\mu}_0 = (40, -0.5)^\top\), \(\boldsymbol{\Sigma}_0 = \text{diag}(100, 1)\). The prior thinks demand sensitivity is lower than it actually is (\(-0.5\) vs. \(-1.2\)), so the initial myopic price is \(-40/(2 \times (-0.5)) = 40\) — much too high.

After 10 observations at various prices, suppose the data matrix is:

$$
\mathbf{X}^\top\mathbf{X} = \begin{pmatrix} 10 & \sum p_i \\ \sum p_i & \sum p_i^2 \end{pmatrix}, \qquad \mathbf{X}^\top\mathbf{Q} = \begin{pmatrix} \sum Q_i \\ \sum p_i Q_i \end{pmatrix}
$$

The posterior precision becomes \(\boldsymbol{\Sigma}_{10}^{-1} = \boldsymbol{\Sigma}_0^{-1} + \sigma^{-2}\mathbf{X}^\top\mathbf{X}\), and the posterior mean shifts toward the OLS estimate. With enough price variation, the posterior quickly concentrates around the true values, and the myopic/TS prices converge to \(p^* \approx 20.83\).

The Simulation 4 in Section 20 implements this exact setup and visualizes the posterior learning process, price convergence, and regret dynamics.

---

## 5. Contextual Bandits for Personalized Pricing

The bandit formulation so far assumes a single optimal price for all customers. But in practice, the optimal price depends on **context**: who the customer is, when they're buying, what device they're using, what competitors are charging, and how much inventory you have.

A **contextual bandit** extends the standard bandit to incorporate side information. At each time step \(t\):

1. Observe a context vector \(\mathbf{x}_t \in \mathbb{R}^d\) (customer features, time features, market features)
2. Choose an arm \(a_t \in \{1, \ldots, K\}\) (a price)
3. Observe reward \(r_t\) (revenue)

The expected reward is now a function of both context and action: \(\mu(\mathbf{x}, a)\). The goal is to learn this function while maximizing cumulative reward.

**LinUCB** (Li et al., 2010) assumes the expected reward is linear in context features for each arm:

$$
\mu(\mathbf{x}, a) = \mathbf{x}^\top \boldsymbol{\theta}_a
$$

For each arm \(a\), we maintain a regularized least-squares estimate of \(\boldsymbol{\theta}_a\) and a **confidence ellipsoid** that quantifies uncertainty. Given the data collected so far for arm \(a\), the estimate is:

$$
\hat{\boldsymbol{\theta}}_a = \mathbf{A}_a^{-1} \mathbf{b}_a
$$

where \(\mathbf{A}_a = \mathbf{I} + \sum_{t: A_t = a} \mathbf{x}_t \mathbf{x}_t^\top\) is the regularized **Gram matrix** and \(\mathbf{b}_a = \sum_{t: A_t = a} r_t \mathbf{x}_t\).

The UCB for arm \(a\) at context \(\mathbf{x}_t\) is:

$$
\text{UCB}_a(\mathbf{x}_t) = \mathbf{x}_t^\top \hat{\boldsymbol{\theta}}_a + \alpha \sqrt{\mathbf{x}_t^\top \mathbf{A}_a^{-1} \mathbf{x}_t}
$$

The first term is the predicted reward. The second term is the **exploration bonus** — it's large when the current context is in a region of feature space where you have little data for arm \(a\) (because \(\mathbf{A}_a^{-1}\) is large in that direction).

This is where **machine learning meets bandits**. The linear model is just the starting point. You can replace it with neural networks, gradient-boosted trees, or any function approximator, as long as you can construct reasonable confidence intervals. The key insight is that any supervised learning model can be turned into a contextual bandit by adding an exploration mechanism.

**Application to personalized pricing**: the context vector \(\mathbf{x}_t\) might include customer income, browsing history, geographic location, time of day, and competitor prices. The bandit learns that high-income customers in urban areas have lower price sensitivity (recalling the Lerner index from Part 1, they have lower \(|\varepsilon|\) so the optimal markup is higher), while price-sensitive customers respond to discounts. The algorithm automatically charges **different prices to different segments** — this is third-degree price discrimination from Part 2, but learned online from data rather than designed by a pricing analyst.

---

## 6. Continuous-Armed Bandits and Lipschitz Optimization

Everything we've done so far discretizes the price space: pick 10 or 20 candidate prices, treat each as an arm, and run UCB or Thompson Sampling. But real pricing is **continuous**. You can charge $19.99, $20.00, $20.01, or anything in between. Discretizing throws away structure — it treats $20.00 and $20.01 as completely unrelated arms, even though they obviously generate nearly identical revenue. And it forces a tradeoff between resolution and sample efficiency: more price points give finer resolution but require more exploration rounds to learn about each one.

The natural formulation is the **continuous-armed bandit**. The action space is a continuous interval \(\mathcal{A} = [p_{\min}, p_{\max}] \subset \mathbb{R}\) instead of a finite set. At each round \(t\), the learner chooses a price \(p_t \in [p_{\min}, p_{\max}]\) and observes a noisy reward \(r_t = \mu(p_t) + \epsilon_t\), where \(\mu: \mathcal{A} \to \mathbb{R}\) is the unknown expected revenue function and \(\epsilon_t\) is zero-mean noise. The goal, as before, is to minimize cumulative regret:

$$
R_T = \sum_{t=1}^{T} \left[\mu(p^*) - \mu(p_t)\right]
$$

where \(p^* = \arg\max_{p \in \mathcal{A}} \mu(p)\).

The fundamental challenge is that the action space is **uncountably infinite**. You cannot try every price even once. Instead, you must exploit **structure** in the reward function to generalize from observed prices to unobserved ones. Without any assumptions on \(\mu\), the problem is hopeless — the revenue at $20.00 tells you nothing about the revenue at $20.01 if the function can be arbitrary.

### Lipschitz Bandits

The most natural structural assumption for pricing is **Lipschitz continuity** (Kleinberg, 2004; Bubeck, Munos, Stoltz, and Szepesvari, 2011). A function \(\mu\) is Lipschitz continuous with constant \(L > 0\) if:

$$
|\mu(p_1) - \mu(p_2)| \leq L |p_1 - p_2| \quad \forall \, p_1, p_2 \in \mathcal{A}
$$

This says that the revenue function cannot change faster than a linear rate. If you know the revenue at $20.00, then the revenue at $20.01 is within \(\pm L \times 0.01\) of that value. For pricing, this is entirely reasonable: if charging $20.00 generates expected revenue \(R\), then charging $20.01 generates revenue very close to \(R\). Customers don't exhibit wildly different behavior in response to a one-cent price difference.

The Lipschitz constant \(L\) quantifies the **maximum sensitivity** of revenue to price changes. A large \(L\) means the revenue curve has steep slopes — small price changes cause big revenue swings. A small \(L\) means the curve is gentle. In practice, \(L\) depends on the product and market: luxury goods with status-signaling value might have very steep revenue curves (customers are sensitive around certain price thresholds), while commodities might have gentler curves.

### The Zooming Algorithm

A naive approach to Lipschitz bandits is **uniform discretization**: divide \([p_{\min}, p_{\max}]\) into \(N\) equally spaced price points and run a finite-armed bandit (like UCB1) on those \(N\) points. The discretization error at each point is at most \(L \cdot (p_{\max} - p_{\min}) / (2N)\), and the regret of UCB1 on \(N\) arms over \(T\) rounds is \(O(N \log T)\). Balancing the discretization error against the bandit regret by setting \(N \sim T^{1/3}\) gives a total regret of \(O(T^{2/3})\). This is already much better than linear regret, but it wastes effort: it explores uniformly even in regions that are clearly suboptimal.

The **Zooming algorithm** (Kleinberg, Slivkins, and Upfal, 2008) does better by **adaptively discretizing** the price space. The core idea is:

1. Start with a coarse covering of the price space.
2. Maintain a UCB-style index for each "active" price point: \(\text{UCB}(p) = \hat{\mu}(p) + \text{confidence\_radius}(p) + \text{discretization\_radius}(p)\).
3. At each round, play the price with the highest index.
4. When a price point has been played enough times that its confidence radius shrinks below its discretization radius, **zoom in**: split that region into finer sub-regions and activate new price points.
5. Crucially, only zoom in where the UCB is high — where the region is either genuinely promising or too uncertain to dismiss. Regions that are clearly suboptimal (low UCB) are left coarse and eventually ignored.

The zooming algorithm concentrates its exploration where it matters most: near the optimum and in high-uncertainty regions. It achieves regret \(O(T^{(d+1)/(d+2)})\) for a \(d\)-dimensional action space, where \(d\) is the **zooming dimension** — a data-dependent quantity that can be much smaller than the ambient dimension when the near-optimal region is small. For \(d = 1\) (single price), the worst-case regret is \(O(T^{2/3})\), matching uniform discretization, but in favorable cases (a sharp peak in the revenue curve), the zooming dimension is small and the algorithm does better.

### Hierarchical Optimistic Optimization (HOO)

An alternative approach is **HOO** (Bubeck, Munos, Stoltz, and Szepesvari, 2011), which organizes the price space as a **binary tree**. The root node covers the entire interval \([p_{\min}, p_{\max}]\). Each internal node is split into two children covering the left and right halves of the parent's interval. The leaves of the tree represent increasingly fine partitions of the price space.

HOO traverses this tree like a UCB algorithm:

1. Start at the root.
2. At each internal node, compute a UCB-style score: \(B(h, i) = \hat{\mu}(h, i) + \sqrt{2 \ln t / N(h, i)} + \nu \rho^h\), where \(h\) is the depth, \(i\) is the node index, \(N(h, i)\) is the visit count, and \(\nu \rho^h\) is a bonus that accounts for the Lipschitz variation within the node (shrinking with depth because finer nodes cover smaller intervals).
3. Descend to the child with the higher score, continuing until reaching the current frontier (deepest expanded level).
4. Play the price at the center of the frontier node, observe the reward, and propagate the update back up the tree.
5. Expand the frontier node if it has been visited enough times.

The tree structure means HOO automatically allocates exponentially more effort to promising regions. A branch of the tree that consistently yields low rewards is rarely visited, while branches near the optimum are expanded to finer and finer resolution. The regret bound is \(O(\sqrt{T \log T})\) under additional smoothness assumptions (e.g., the reward function has a unique maximum with a polynomial rate of decrease away from it), which is dramatically better than \(O(T^{2/3})\).

For pricing, HOO is intuitive: start with a rough sense of whether the optimal price is in the low, medium, or high range. As data accumulates, zoom into the right neighborhood. Within that neighborhood, zoom in further. The tree automatically handles the exploration-exploitation tradeoff at every scale simultaneously.

### The Regret Cost of Continuity

It is worth pausing to appreciate the **price of continuity**. For finite-armed bandits with \(K\) arms, the Lai-Robbins lower bound gives \(\Omega(\log T)\) regret — essentially negligible compared to \(T\). But for continuous-armed bandits even with Lipschitz structure, the lower bound is \(\Omega(T^{2/3})\) for one-dimensional action spaces. This is a qualitative difference: going from finite to continuous arms makes the problem fundamentally harder.

The reason is information-theoretic. With \(K\) finite arms, each observation directly reduces uncertainty about one arm's mean reward. After \(O(\log T)\) observations per arm, you've identified the best one with high probability. With a continuum of arms, each observation only reduces uncertainty in a neighborhood of the observed price (by Lipschitz continuity). To pin down the optimal price to within \(\epsilon\), you need to observe prices in an \(\epsilon\)-neighborhood of it, and finding that neighborhood requires coarse exploration first. This multi-scale search is what the zooming and HOO algorithms formalize.

---

## 7. Gaussian Process Bandits and Bayesian Optimization for Pricing

Lipschitz bandits assume minimal structure — just that the revenue function doesn't change too fast. But in many pricing settings, you have stronger beliefs. You might believe the revenue curve is smooth (differentiable, not just Lipschitz), or that it has a single peak (unimodal), or that it roughly resembles revenue curves you've seen for similar products. **Gaussian Processes** provide a framework to encode these beliefs precisely and to quantify uncertainty over the *entire* revenue function, not just at discrete price points.

### Gaussian Processes as Distributions Over Functions

A **Gaussian Process** (GP) is a probability distribution over functions. Just as a Gaussian distribution describes uncertainty about a single number (with a mean and a variance), a GP describes uncertainty about an entire function (with a mean function and a covariance function).

Formally, a GP is specified by:
- A **mean function** \(m(p) = \mathbb{E}[f(p)]\), encoding the prior belief about the function's average value at each price \(p\). Often set to zero or a constant for simplicity.
- A **kernel function** (or covariance function) \(k(p, p') = \text{Cov}[f(p), f(p')]\), encoding the prior belief about how function values at different prices are correlated.

The kernel is the heart of the GP. It encodes your structural assumptions about the revenue function. The most common choice is the **squared exponential** (or RBF — Radial Basis Function) kernel:

$$
k(p, p') = \sigma_f^2 \exp\left(-\frac{|p - p'|^2}{2\ell^2}\right)
$$

Here, \(\sigma_f^2\) is the **signal variance** (how much the function varies overall) and \(\ell\) is the **length scale** (how quickly the function changes with price). A large \(\ell\) means the function is very smooth — prices far apart still have correlated revenues. A small \(\ell\) means the function can vary rapidly — only very nearby prices have correlated revenues.

For pricing, the length scale has a direct economic interpretation. A long length scale means customers are relatively insensitive to small price changes — the revenue curve is smooth and broad. A short length scale means customers are very price-sensitive around certain thresholds — perhaps there's a psychological barrier at $50 or $100 that causes a sharp drop in demand.

### The GP Posterior: Learning from Data

The power of GPs is that conditioning on data gives a **closed-form posterior** that is also a GP. Suppose you've observed data \(\mathcal{D} = \{(p_1, r_1), \ldots, (p_n, r_n)\}\) where \(r_i = \mu(p_i) + \epsilon_i\) and \(\epsilon_i \sim \mathcal{N}(0, \sigma_{\text{noise}}^2)\). Then the posterior distribution over \(\mu(p)\) at any new price \(p\) is Gaussian with:

$$
\mu_n(p) = \mathbf{k}(p)^\top \left[\mathbf{K} + \sigma_{\text{noise}}^2 \mathbf{I}\right]^{-1} \mathbf{r}
$$

$$
\sigma_n^2(p) = k(p, p) - \mathbf{k}(p)^\top \left[\mathbf{K} + \sigma_{\text{noise}}^2 \mathbf{I}\right]^{-1} \mathbf{k}(p)
$$

where \(\mathbf{K}\) is the \(n \times n\) kernel matrix with entries \(K_{ij} = k(p_i, p_j)\), \(\mathbf{k}(p)\) is the \(n\)-vector with entries \(k(p_i, p)\), and \(\mathbf{r} = (r_1, \ldots, r_n)^\top\).

The posterior mean \(\mu_n(p)\) is the **best estimate** of revenue at price \(p\) given the data — it interpolates through observed data points and regresses toward the prior mean in unobserved regions. The posterior variance \(\sigma_n^2(p)\) quantifies **uncertainty** — it's small near observed data points (where the GP is confident) and large far from any observations (where the GP is uncertain). This is exactly what we need for pricing: a complete picture of where we know the revenue curve and where we don't.

### GP-UCB: Optimistic Pricing with Gaussian Processes

**GP-UCB** (Srinivas, Krause, Kakade, and Seeger, 2010) combines the GP posterior with the UCB principle. At each round \(t\), choose the price that maximizes:

$$
p_t = \arg\max_{p \in \mathcal{A}} \left[\mu_{t-1}(p) + \beta_t \, \sigma_{t-1}(p)\right]
$$

where \(\beta_t\) is an exploration parameter that grows as \(\beta_t = O(\sqrt{\log t})\). The first term exploits (choose prices with high predicted revenue) and the second term explores (choose prices where uncertainty is high).

The regret bound for GP-UCB is:

$$
R_T = O^*\left(\sqrt{T \, \gamma_T}\right)
$$

where the \(O^*\) hides logarithmic factors and \(\gamma_T\) is the **maximum information gain** after \(T\) observations. The information gain measures how much information \(T\) observations can provide about the function \(\mu\) under the GP prior. It depends on the kernel:

- For the **RBF kernel**: \(\gamma_T = O((\log T)^{d+1})\), where \(d\) is the input dimension. For one-dimensional pricing (\(d = 1\)), this gives \(\gamma_T = O((\log T)^2)\), and the regret bound becomes \(O^*(\sqrt{T (\log T)^2})\) — nearly \(O(\sqrt{T})\), which is dramatically better than the \(O(T^{2/3})\) of Lipschitz bandits.
- For the **Matern kernel** with smoothness parameter \(\nu\): \(\gamma_T = O(T^{d(d+1)/(2\nu + d(d+1))} (\log T))\). Smoother kernels (larger \(\nu\)) give smaller information gain and better regret.

The improvement over Lipschitz bandits comes from the stronger smoothness assumption encoded in the kernel. The RBF kernel assumes the function is infinitely differentiable, which constrains the function class much more than Lipschitz continuity alone.

### Expected Improvement: An Alternative Acquisition Function

GP-UCB is not the only way to select the next price. **Expected Improvement (EI)** is a classic acquisition function from the Bayesian optimization literature. Let \(r_{\text{best}} = \max_{i \leq t} r_i\) be the best observed revenue so far. The expected improvement at price \(p\) is:

$$
\text{EI}(p) = \mathbb{E}\left[\max\left(0, \, \mu(p) - r_{\text{best}}\right)\right]
$$

Under the GP posterior, this has a closed-form expression:

$$
\text{EI}(p) = (\mu_n(p) - r_{\text{best}}) \, \Phi(z) + \sigma_n(p) \, \phi(z)
$$

where \(z = (\mu_n(p) - r_{\text{best}}) / \sigma_n(p)\), \(\Phi\) is the standard normal CDF, and \(\phi\) is the standard normal PDF.

The first term \((\mu_n(p) - r_{\text{best}}) \Phi(z)\) rewards exploitation — it's large when the predicted revenue exceeds the current best. The second term \(\sigma_n(p) \phi(z)\) rewards exploration — it's large when uncertainty is high. EI naturally balances these two objectives and tends to converge to the optimum more aggressively than UCB, which makes it well-suited for pricing where you want to find the best price quickly rather than uniformly reducing uncertainty everywhere.

### Why GPs Are Perfect for Pricing

The GP posterior gives you something that no other bandit algorithm provides: a **complete visualization** of what you know and don't know about the revenue curve. You can plot the posterior mean (your best estimate of revenue at every price) with shaded uncertainty bands (where you're confident vs. uncertain). This is invaluable for a pricing team that needs to understand and trust the algorithm.

The kernel's length scale \(\ell\) encodes a key economic quantity: how quickly revenue changes with price. By fitting \(\ell\) from data (via marginal likelihood maximization), you're implicitly estimating price sensitivity — a short \(\ell\) means customers react sharply to small price changes, a long \(\ell\) means they don't. This connects directly to the elasticity concept from Part 1, but expressed in the function-space language of GPs rather than the point-estimate language of demand curves.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 380" style="max-width:700px; width:100%; height:auto;">
  <rect width="700" height="380" fill="#1a1a2e" rx="8"/>
  <!-- Title -->
  <text x="350" y="30" text-anchor="middle" fill="#d4d4d4" font-size="15" font-weight="bold">GP Posterior Over Revenue Curve</text>
  <!-- Axes -->
  <line x1="80" y1="320" x2="660" y2="320" stroke="#636e72" stroke-width="1.5"/>
  <line x1="80" y1="50" x2="80" y2="320" stroke="#636e72" stroke-width="1.5"/>
  <text x="370" y="355" text-anchor="middle" fill="#d4d4d4" font-size="12">Price p</text>
  <text x="30" y="185" text-anchor="middle" fill="#d4d4d4" font-size="12" transform="rotate(-90, 30, 185)">Revenue μ(p)</text>
  <!-- Uncertainty band (wide far from data, narrow near data) -->
  <path d="M100,260 Q200,230 280,140 Q340,80 380,90 Q420,100 460,150 Q540,240 640,280" fill="none" stroke="none"/>
  <!-- Upper band -->
  <path d="M100,220 Q200,170 280,80 Q340,30 380,45 Q420,60 460,110 Q540,200 640,240
           L640,310 Q540,280 460,200 Q420,150 380,140 Q340,130 280,200 Q200,270 100,290 Z"
        fill="#3498db" fill-opacity="0.2" stroke="none"/>
  <!-- Mean function -->
  <path d="M100,255 Q200,220 280,140 Q340,85 380,92 Q420,105 460,155 Q540,240 640,275"
        fill="none" stroke="#3498db" stroke-width="2.5"/>
  <!-- True function (dashed) -->
  <path d="M100,250 Q200,215 280,130 Q340,75 380,80 Q420,95 460,150 Q540,235 640,270"
        fill="none" stroke="#e74c3c" stroke-width="2" stroke-dasharray="6,4"/>
  <!-- Observed data points -->
  <circle cx="150" cy="240" r="5" fill="#2ecc71" stroke="#fff" stroke-width="1"/>
  <circle cx="250" cy="170" r="5" fill="#2ecc71" stroke="#fff" stroke-width="1"/>
  <circle cx="320" cy="110" r="5" fill="#2ecc71" stroke="#fff" stroke-width="1"/>
  <circle cx="370" cy="95" r="5" fill="#2ecc71" stroke="#fff" stroke-width="1"/>
  <circle cx="400" cy="100" r="5" fill="#2ecc71" stroke="#fff" stroke-width="1"/>
  <circle cx="500" cy="195" r="5" fill="#2ecc71" stroke="#fff" stroke-width="1"/>
  <!-- Next query point (star) -->
  <polygon points="570,255 574,243 586,243 576,236 580,224 570,231 560,224 564,236 554,243 566,243"
           fill="#f1c40f" stroke="#f1c40f" stroke-width="1"/>
  <text x="570" y="218" text-anchor="middle" fill="#f1c40f" font-size="10">next query</text>
  <!-- Legend -->
  <line x1="100" y1="365" x2="125" y2="365" stroke="#3498db" stroke-width="2.5"/>
  <text x="130" y="369" fill="#d4d4d4" font-size="10">GP mean</text>
  <line x1="210" y1="365" x2="235" y2="365" stroke="#e74c3c" stroke-width="2" stroke-dasharray="6,4"/>
  <text x="240" y="369" fill="#d4d4d4" font-size="10">True revenue</text>
  <circle cx="330" cy="365" r="4" fill="#2ecc71" stroke="#fff" stroke-width="1"/>
  <text x="340" y="369" fill="#d4d4d4" font-size="10">Observed data</text>
  <rect x="420" y="358" width="15" height="12" fill="#3498db" fill-opacity="0.2" stroke="#3498db" stroke-width="0.5"/>
  <text x="440" y="369" fill="#d4d4d4" font-size="10">±2σ band</text>
  <polygon points="540,365 543,359 549,359 544,355 546,349 540,353 534,349 536,355 531,359 537,359"
           fill="#f1c40f" stroke="#f1c40f" stroke-width="0.5"/>
  <text x="555" y="369" fill="#d4d4d4" font-size="10">Next query (UCB/EI)</text>
</svg>

The diagram above illustrates the GP posterior after a handful of observations. Near the data points, the uncertainty band is narrow — the GP is confident about the revenue. Far from data (at the right edge), the band is wide — the GP acknowledges its ignorance. The next query point is chosen where the UCB (mean + exploration bonus) is maximized, which in this case is in the under-explored region where the uncertainty band is wide but the mean is not yet obviously low.

---

## 8. Non-Stationary Bandits — When Demand Shifts

Every bandit algorithm we've discussed so far assumes **stationarity**: the reward distribution of each arm doesn't change over time. The optimal price today is the optimal price tomorrow. In real pricing, this assumption is almost never true. Demand shifts due to seasonality (winter coats in July vs. January), competitor actions (a rival launches a cheaper alternative), trends (a product goes viral on social media), macroeconomic changes (recession reduces willingness to pay), and external shocks (a pandemic, a supply chain disruption).

A pricing algorithm that ignores non-stationarity will converge to the optimal price for the *old* demand curve and then keep charging that price long after demand has shifted. The result is linear regret in the new regime — exactly the behavior we designed bandits to avoid.

### Two Models of Non-Stationarity

The literature distinguishes two fundamentally different types of change:

**1. Abruptly changing (piecewise stationary)**: the reward distribution is constant for a period of time, then suddenly jumps to a new distribution at an unknown **changepoint**, stays constant again, then jumps again, and so on. There are \(M\) changepoints in \(T\) rounds. Example: a competitor launches a similar product on day 500, and your demand curve suddenly shifts downward. Or a raw material shortage doubles your production costs overnight, requiring a price adjustment.

**2. Slowly drifting**: the reward distribution changes gradually every round. At each time \(t\), the mean reward of arm \(k\) is \(\mu_k(t)\), and the total variation is bounded: \(\sum_{t=1}^{T-1} \max_k |\mu_k(t+1) - \mu_k(t)| \leq V_T\). Example: seasonal demand gradually shifting over months, or a slow change in customer preferences as a product ages.

### Discounted UCB

The simplest approach to non-stationarity is to **discount old observations**. Instead of the standard sample mean, use an exponentially weighted average that gives more weight to recent data:

$$
\hat{\mu}_k^\gamma(t) = \frac{\sum_{s \leq t: A_s = k} \gamma^{t-s} \, r_s}{\sum_{s \leq t: A_s = k} \gamma^{t-s}}
$$

where \(\gamma \in (0, 1)\) is the **discount factor**. When \(\gamma\) is close to 1, old data is nearly as important as new data (the algorithm adapts slowly but has low variance). When \(\gamma\) is small, only very recent data matters (the algorithm adapts quickly but is noisy).

The effective number of observations contributing to the estimate of arm \(k\) is:

$$
N_k^\gamma(t) = \sum_{s \leq t: A_s = k} \gamma^{t-s}
$$

This is a geometric series bounded by \(1/(1-\gamma)\), regardless of how many times the arm has been pulled. The discount factor creates an implicit **sliding window** of effective sample size.

The **Discounted UCB** algorithm replaces the standard UCB with:

$$
\text{D-UCB}_k(t) = \hat{\mu}_k^\gamma(t) + c\sqrt{\frac{\ln t}{N_k^\gamma(t)}}
$$

The exploration bonus now depends on the *effective* sample size rather than the total count. Arms that haven't been pulled recently have a small \(N_k^\gamma(t)\) (because old observations are discounted), so they get a large exploration bonus and are revisited. This ensures the algorithm keeps re-exploring arms to detect changes.

### Sliding Window UCB (SW-UCB)

An even simpler approach: only use the **last \(W\) observations** for each arm, discarding everything older. The sliding-window sample mean for arm \(k\) at time \(t\) is:

$$
\hat{\mu}_k^W(t) = \frac{1}{N_k^W(t)} \sum_{\substack{s \in [t-W+1, t] \\ A_s = k}} r_s
$$

where \(N_k^W(t)\) is the number of times arm \(k\) was pulled in the last \(W\) rounds. SW-UCB then uses:

$$
\text{SW-UCB}_k(t) = \hat{\mu}_k^W(t) + c\sqrt{\frac{\ln t}{N_k^W(t)}}
$$

The window size \(W\) controls the speed-accuracy tradeoff. A small \(W\) adapts quickly to changes but has high variance (few observations per estimate). A large \(W\) is more stable but slow to adapt. The optimal window depends on the rate of change: if changes happen every \(\tau\) rounds, setting \(W \approx \tau\) is roughly optimal.

### Sliding Window Thompson Sampling

Thompson Sampling adapts to non-stationarity via the same sliding-window idea. For Bernoulli rewards (buy or don't buy), maintain the Beta posterior using only the last \(W\) observations:

$$
\alpha_k^W(t) = 1 + \sum_{\substack{s \in [t-W+1, t] \\ A_s = k}} r_s, \qquad \beta_k^W(t) = 1 + \sum_{\substack{s \in [t-W+1, t] \\ A_s = k}} (1 - r_s)
$$

At each round, sample \(\theta_k \sim \text{Beta}(\alpha_k^W, \beta_k^W)\) for each arm and play the arm with the highest \(p_k \times \theta_k\). The sliding window ensures the posterior reflects only recent demand, so the algorithm naturally forgets outdated information and adapts to the current environment.

### Changepoint Detection + Restart

A more sophisticated approach is to run a standard (stationary) bandit algorithm, but monitor the data stream for **changepoints**. When a change is detected, reset the algorithm and start fresh.

Common changepoint detection methods include:

- **CUSUM test**: maintain a running sum \(S_t = \max(0, S_{t-1} + r_t - \hat{\mu} - \delta)\). When \(S_t\) exceeds a threshold \(h\), declare a change. The parameters \(\delta\) (minimum detectable change) and \(h\) (sensitivity threshold) control the tradeoff between false alarms and detection delay.
- **Bayesian changepoint detection**: maintain a posterior distribution over the location of the most recent changepoint. At each round, compute the posterior probability that a changepoint occurred in the last few rounds. If this probability exceeds a threshold, declare a change.
- **Page-Hinkley test**: a variant of CUSUM that monitors the cumulative deviation from the running mean.

The advantage of changepoint detection is that it combines the efficiency of stationary algorithms (when nothing changes, you get \(O(\log T)\) regret) with adaptivity when changes occur. The disadvantage is detection delay — there's an unavoidable gap between when the change happens and when you detect it, during which you're using the wrong model.

### Regret Bounds for Non-Stationary Settings

For the piecewise-stationary model with \(M\) changepoints in \(T\) rounds, the best algorithms achieve regret:

$$
R_T = O\left(\sqrt{M T \log T}\right)
$$

This is proportional to \(\sqrt{M}\) — more changepoints mean more regret, because each changepoint requires a period of re-exploration. Note that if \(M = 0\) (stationary), this recovers the \(O(\sqrt{T \log T})\) bound of standard bandits.

For the slowly drifting model with total variation \(V_T\), the best algorithms achieve:

$$
R_T = O\left(V_T^{1/3} T^{2/3}\right)
$$

This degrades gracefully with the rate of change \(V_T\).

### Practical Importance for Pricing

A pricing algorithm that doesn't handle non-stationarity will keep charging last season's optimal price into the new season. If your competitor exits the market and demand surges, a stationary bandit will take a very long time to discover that higher prices are now optimal — it has already "converged" to the old optimum and barely explores anymore.

Discounted Thompson Sampling is the industry standard for this reason: it adapts automatically, requires minimal tuning (just the discount factor or window size), and retains the strong empirical performance of Thompson Sampling. In practice, most production pricing systems use a discount factor of \(\gamma \in [0.99, 0.999]\), corresponding to an effective memory of 100 to 1000 observations.

---

## 9. Adversarial Bandits — When Competitors Fight Back

Non-stationary bandits assume demand changes, but not *in response to your actions*. What if the environment is **adversarial** — what if a competitor observes your pricing behavior and deliberately undercuts you? This is worse than random drift or seasonal change: the environment is trying to make you lose.

In an adversarial bandit model, the rewards are not drawn from fixed (or even smoothly changing) distributions. Instead, at each round \(t\), an **adversary** chooses a reward vector \(\mathbf{r}_t = (r_{t,1}, \ldots, r_{t,K})\) — one reward for each arm. The learner then chooses arm \(A_t\) and observes only \(r_{t, A_t}\) — the reward of the arm they pulled. The adversary can be **oblivious** (chooses all reward vectors before the game starts) or **adaptive** (chooses \(\mathbf{r}_t\) based on the learner's past actions \(A_1, \ldots, A_{t-1}\)).

In pricing: imagine a competitor with a repricing bot that monitors your prices. When you charge $25, the competitor undercuts to $24. When you raise to $30, the competitor responds by pricing at $29. The competitor's strategy is a function of your past pricing, making it adaptive. Standard UCB or Thompson Sampling — designed for stochastic environments — will fail catastrophically because they assume rewards are drawn from fixed distributions.

### The EXP3 Algorithm

**EXP3** (Exponential-weight algorithm for Exploration and Exploitation; Auer, Cesa-Bianchi, Freund, and Schapire, 2002) is the foundational algorithm for adversarial bandits. Instead of maintaining mean estimates and confidence bounds, EXP3 maintains a **probability distribution** over arms and updates it using exponential weights.

The algorithm proceeds as follows:

**Initialize**: set weights \(w_k(1) = 1\) for all \(k = 1, \ldots, K\), giving a uniform distribution \(p_k(1) = 1/K\).

**At each round \(t\)**:
1. Compute the mixed strategy: \(p_k(t) = (1 - \gamma) \frac{w_k(t)}{\sum_{j=1}^K w_j(t)} + \frac{\gamma}{K}\) where \(\gamma \in (0, 1]\) mixes with the uniform distribution to ensure minimum exploration probability \(\gamma/K\) for every arm.
2. Sample arm \(A_t\) from the distribution \(p(t) = (p_1(t), \ldots, p_K(t))\).
3. Observe reward \(r_{t, A_t}\).
4. Construct the **importance-weighted reward estimator**: for each arm \(k\),

$$
\hat{r}_{t,k} = \begin{cases} r_{t,k} / p_k(t) & \text{if } k = A_t \\ 0 & \text{otherwise} \end{cases}
$$

5. Update weights: \(w_k(t+1) = w_k(t) \cdot \exp\left(\eta \, \hat{r}_{t,k}\right)\) where \(\eta = \sqrt{\ln K / (TK)}\) is the learning rate.

### Why Importance Weighting?

The importance-weighted estimator is the key innovation. The problem is that you only observe the reward of the arm you pulled — you don't see the counterfactual rewards of the other arms. If you simply used \(r_{t, A_t}\) as the reward estimate for arm \(A_t\) and zero for others, your estimates would be **biased**: arms that you pull frequently would have artificially high estimated rewards (because you observe their actual rewards) while arms you rarely pull would have artificially low estimates (because you mostly assign them zeros).

The correction \(\hat{r}_{t,k} = r_{t,k} / p_k(t)\) fixes this. The key property is:

$$
\mathbb{E}[\hat{r}_{t,k} | p(t)] = p_k(t) \cdot \frac{r_{t,k}}{p_k(t)} + (1 - p_k(t)) \cdot 0 = r_{t,k}
$$

The estimator is **unbiased**: its expected value equals the true reward, regardless of the probability with which the arm was played. Arms that are rarely played get their rewards amplified (divided by a small \(p_k(t)\)) to compensate for being observed infrequently. This is the same inverse propensity weighting used in causal inference (Part 4) — we're correcting for selection bias in which arm we chose to observe.

The cost of unbiasedness is **variance**. When \(p_k(t)\) is small, \(\hat{r}_{t,k}\) can be very large (reward divided by a small number), introducing high variance. The mixing with the uniform distribution (the \(\gamma/K\) term) puts a floor on \(p_k(t)\), bounding the maximum variance. This is the exploration-exploitation tradeoff in adversarial settings: more uniform mixing (larger \(\gamma\)) reduces variance but wastes more reward on exploration.

### Regret Guarantee

EXP3 achieves the following regret bound against **any** adversary, including adaptive ones:

$$
R_T \leq O\left(\sqrt{TK \log K}\right)
$$

This is a **worst-case** guarantee: no matter what the adversary does — even if it's an omniscient competitor that sees your algorithm's internal state (minus the current random seed) — EXP3's cumulative regret is at most \(O(\sqrt{TK \log K})\). Over \(T = 10{,}000\) rounds with \(K = 10\) price points, this is roughly \(\sqrt{10{,}000 \times 10 \times \ln 10} \approx 480\), which means the per-round regret averages about 0.048 — less than 5% of what the best fixed arm earns.

The comparison point for adversarial regret is the **best fixed arm in hindsight** — the single arm that, had you always played it, would have given the highest cumulative reward. EXP3 doesn't compete against the best *adaptive* strategy (that would require even stronger algorithms); it competes against the best constant action. But this is already remarkable: even against an adversary that's trying to make you lose, you're guaranteed to do almost as well as the best fixed price.

### The Stochastic-Adversarial Tradeoff

EXP3's guarantee comes at a cost. In a **stochastic** environment (where rewards are i.i.d.), UCB and Thompson Sampling achieve \(O(\log T)\) regret, while EXP3 achieves \(O(\sqrt{T})\) — exponentially worse. EXP3 explores more than necessary because it can't trust past data as much; it hedges against the possibility that the environment is adversarial.

This creates a practical dilemma: if you use EXP3, you're robust to adversarial behavior but waste regret in benign environments. If you use UCB/Thompson Sampling, you're efficient in stochastic environments but vulnerable to adversarial ones.

The solution is **best-of-both-worlds** algorithms (Bubeck and Slivkins, 2012) that adapt their behavior to the environment: they achieve \(O(\log T)\) regret in stochastic settings and \(O(\sqrt{T})\) regret in adversarial settings, without knowing which type of environment they face. These algorithms monitor statistical tests for stochasticity (e.g., checking if reward sequences are consistent with i.i.d. draws) and switch between UCB-style and EXP3-style updates accordingly. For pricing in competitive markets, this is the ideal approach: be efficient when competitors are stable, and robust when they're aggressive.

---

## 10. Multi-Product Dynamic Pricing

Everything up to this point has been about pricing **one product**. But real firms sell portfolios. Amazon has millions of SKUs. A grocery store has tens of thousands. A SaaS company has multiple tiers and add-ons. The prices **interact**: raising the price of Product A might push customers toward substitute Product B, or it might reduce demand for complement Product C (because customers buy them together).

Ignoring these interactions and pricing each product independently can leave significant money on the table — or worse, actively destroy value.

### The Multi-Product Pricing Problem

Consider a firm selling \(J\) products with price vector \(\mathbf{p} = (p_1, \ldots, p_J)\) and marginal cost vector \(\mathbf{c} = (c_1, \ldots, c_J)\). The demand for product \(j\) depends on **all** prices, not just \(p_j\):

$$
Q_j = Q_j(\mathbf{p}) = Q_j(p_1, p_2, \ldots, p_J)
$$

This is because customers make portfolio decisions — they consider the entire menu of prices when deciding what to buy. Total profit is:

$$
\pi(\mathbf{p}) = \sum_{j=1}^{J} (p_j - c_j) \, Q_j(\mathbf{p})
$$

The first-order condition for the optimal price of product \(j\) is obtained by differentiating with respect to \(p_j\):

$$
\frac{\partial \pi}{\partial p_j} = Q_j + \sum_{k=1}^{J} (p_k - c_k) \frac{\partial Q_k}{\partial p_j} = 0
$$

This equation has a profound structure. The first term, \(Q_j\), is the direct effect of raising \(p_j\) — you earn more per unit sold. The second term captures the **cross-effects**: the summation runs over *all* products \(k\), including \(k = j\) (own-price effect) and \(k \neq j\) (cross-price effects).

The cross-price derivative \(\partial Q_k / \partial p_j\) classifies the relationship between products \(j\) and \(k\):

- **Substitutes** (\(\partial Q_k / \partial p_j > 0\)): raising the price of product \(j\) increases demand for product \(k\). Customers switch from \(j\) to \(k\). Examples: Coke and Pepsi, iPhone and Samsung Galaxy, Standard and Premium SaaS tiers.
- **Complements** (\(\partial Q_k / \partial p_j < 0\)): raising the price of product \(j\) decreases demand for product \(k\). Customers buy them together. Examples: printers and ink cartridges, razors and blades, game consoles and games.
- **Independent** (\(\partial Q_k / \partial p_j = 0\)): the products don't interact. Pricing can be done independently.

### Portfolio Effects on Optimal Prices

The cross-effects have systematic consequences for pricing:

**For substitutes**: when \(\partial Q_k / \partial p_j > 0\) for substitute \(k\), the term \((p_k - c_k) \partial Q_k / \partial p_j > 0\) in the FOC for product \(j\). This means the firm's marginal profit from raising \(p_j\) is *higher* than it would be for a single-product monopolist, because the diverted demand generates profit on the substitute. The optimal price of \(p_j\) is therefore **higher** than the single-product optimum. A multi-product firm internalizes the substitution effect and raises prices on substitutable products.

**For complements**: when \(\partial Q_k / \partial p_j < 0\) for complement \(k\), the term \((p_k - c_k) \partial Q_k / \partial p_j < 0\). Raising \(p_j\) reduces demand for complement \(k\), destroying profit there. The firm internalizes this and sets \(p_j\) **lower** than the single-product optimum. This is why Amazon can sell Kindle e-readers at cost (or even a loss) — the reduced price increases demand for Kindle books, which have high margins. It's why printer manufacturers sell printers cheaply and charge premium prices for ink cartridges.

### The Multi-Product Lerner Index

The first-order conditions can be written in elegant matrix form. Define the \(J \times J\) **Jacobian matrix** of demand:

$$
\mathbf{J} = \frac{\partial \mathbf{Q}}{\partial \mathbf{p}^\top}, \quad J_{jk} = \frac{\partial Q_j}{\partial p_k}
$$

The first-order conditions become:

$$
\mathbf{Q} + \mathbf{J}^\top (\mathbf{p} - \mathbf{c}) = \mathbf{0}
$$

Solving for the optimal markup:

$$
\mathbf{p} - \mathbf{c} = -(\mathbf{J}^\top)^{-1} \mathbf{Q}
$$

This is the **multi-product Lerner index**. For a single product (\(J = 1\)), this reduces to \(p - c = -Q / (\partial Q / \partial p) = Q / |Q'|\), which is the familiar Lerner index \((p - c)/p = 1/|\varepsilon|\). For multiple products, the inverse Jacobian captures all the cross-elasticity effects simultaneously.

### The Curse of Dimensionality

The computational challenge is severe. With \(J\) products, the price vector lives in \(\mathbb{R}^J\), and the demand Jacobian has \(J^2\) entries — each an elasticity that must be estimated from data. For \(J = 100\) products, that's 10,000 elasticities. For \(J = 1{,}000\), it's a million. The bandit problem has a \(J\)-dimensional action space (each action is a complete price vector), and the number of rounds needed to explore grows exponentially with \(J\).

Practical approaches to taming this curse:

**Decomposition**: if the product catalog can be partitioned into groups with no significant cross-effects between groups (e.g., electronics and groceries), optimize each group independently. Within each group, the dimensionality is manageable.

**Coordinate descent**: cycle through products one at a time, optimizing each price holding the others fixed. At each step, you solve a one-dimensional problem. Under mild conditions on the demand Jacobian (diagonal dominance — own-price effects dominate cross-price effects), coordinate descent converges to the global optimum.

**Structured demand models**: instead of estimating \(J^2\) free elasticities, impose structure. A **nested logit** demand model, for instance, groups products into nests (substitution classes) and parameterizes cross-elasticities within and between nests using a few parameters. This reduces the estimation burden from \(J^2\) to \(O(J)\).

**Neural bandits for portfolio pricing**: use a neural network to model the joint demand function \(\mathbf{Q}(\mathbf{p})\) and optimize the entire price vector using gradient-based methods (backpropagation through the network) with added exploration noise. The neural network can capture complex nonlinear interactions between prices that linear models miss.

Amazon's approach is representative: it doesn't jointly optimize all millions of SKUs. Instead, it clusters products into competitive groups (substitutes within a category), identifies complementary pairs (frequently bought together), and optimizes within clusters while maintaining consistency constraints across the catalog. The cluster structure reduces the effective dimensionality from millions to hundreds of small, manageable problems.

---

## 11. Deep Reinforcement Learning for Pricing

When the state space is large — many products, complex context, long time horizons with inventory dynamics — exact dynamic programming (the Bellman equation from Section 14) becomes computationally infeasible. The state space is too large to enumerate, and the transition dynamics are too complex to model analytically. **Deep reinforcement learning** (deep RL) provides scalable approximations by using neural networks to represent value functions and policies.

### Deep Q-Networks (DQN) for Pricing

Recall from the MDP framework that the **Q-function** \(Q(s, a)\) gives the expected cumulative discounted reward from being in state \(s\), taking action \(a\), and thereafter following the optimal policy. If you knew the Q-function, optimal pricing would be trivial: in every state, charge the price \(a^* = \arg\max_a Q(s, a)\).

A **Deep Q-Network** (DQN; Mnih et al., 2015) approximates \(Q(s, a)\) with a neural network \(Q_\theta(s, a)\) parameterized by weights \(\theta\). The state \(s\) might include current inventory levels for all products, time features (day of week, month, season), recent demand signals (sales velocity, search volume), competitor prices, and macroeconomic indicators. The action \(a\) is the price (or price vector for multi-product settings), discretized into a manageable set.

**Training** proceeds by minimizing the **Bellman error**. The target for a transition \((s, a, r, s')\) is \(y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')\), where \(\theta^-\) are the weights of a **target network** — a slowly-updating copy of the Q-network that stabilizes training. The loss function is:

$$
\mathcal{L}(\theta) = \mathbb{E}\left[\left(Q_\theta(s, a) - \left[r + \gamma \max_{a'} Q_{\theta^-}(s', a')\right]\right)^2\right]
$$

Two key tricks make DQN work:
1. **Experience replay**: store all observed transitions \((s, a, r, s')\) in a replay buffer and sample random minibatches for training. This breaks the temporal correlation between consecutive samples and improves data efficiency.
2. **Target network**: update \(\theta^-\) only every \(C\) steps (by copying \(\theta\) to \(\theta^-\)). Without this, the target \(y\) changes with every gradient step, creating a moving-target problem that destabilizes learning.

### Policy Gradient Methods

DQN requires discretizing the action space, which is limiting for continuous pricing. **Policy gradient methods** directly parameterize the policy \(\pi_\theta(a | s)\) as a neural network that outputs a distribution over prices given the state. For continuous prices, the network might output the mean \(\mu_\theta(s)\) and standard deviation \(\sigma_\theta(s)\) of a Gaussian distribution: \(\pi_\theta(a | s) = \mathcal{N}(a; \mu_\theta(s), \sigma_\theta(s)^2)\).

The objective is to maximize expected cumulative reward:

$$
J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]
$$

The **policy gradient theorem** (Sutton et al., 2000) gives the gradient:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t | s_t) \, A(s_t, a_t)\right]
$$

where \(A(s_t, a_t) = Q(s_t, a_t) - V(s_t)\) is the **advantage function** — how much better action \(a_t\) is compared to the average action in state \(s_t\). If the advantage is positive (the price we chose was better than average), the gradient update increases the probability of choosing that price in similar states. If negative, it decreases the probability.

The REINFORCE algorithm estimates the advantage using Monte Carlo returns (run full episodes and compute cumulative reward), but this has high variance. Modern algorithms like **PPO** (Proximal Policy Optimization; Schulman et al., 2017) use a learned value function to estimate the advantage with lower variance and clip the policy update to prevent catastrophically large steps.

### Actor-Critic Architecture

The **actor-critic** framework combines the best of both worlds:
- The **actor** is the policy network \(\pi_\theta(a | s)\) that selects prices.
- The **critic** is a value network \(V_\phi(s)\) that estimates the expected cumulative reward from state \(s\).

The critic provides low-variance estimates of the advantage \(A(s, a) \approx r + \gamma V_\phi(s') - V_\phi(s)\), which stabilizes the policy gradient update. The actor and critic are trained simultaneously: the critic minimizes the TD error \((r + \gamma V_\phi(s') - V_\phi(s))^2\), and the actor maximizes the expected reward using the critic's advantage estimates.

For pricing, the actor outputs a price distribution given the current state (inventory, demand signals, competitor prices, time), and the critic evaluates whether the resulting state trajectory leads to good cumulative revenue. Over time, the actor learns to charge high prices when inventory is scarce and demand is strong, and low prices when inventory is plentiful or demand is weak.

### Practical Challenges

**Simulation-to-real gap**: you typically train the RL agent in a simulated demand environment because exploration in the real market is expensive (every suboptimal price costs real revenue). But the simulated environment never perfectly matches reality. **Domain randomization** — randomly varying the simulator's parameters (elasticities, arrival rates, competitor behavior) during training — helps the agent learn a robust policy that works across a range of environments, not just the specific simulated one.

**Safety constraints**: an unconstrained RL agent might explore dangerously low prices (giving products away for free to learn about demand at low prices) or dangerously high prices (alienating customers). **Constrained RL** adds penalties or hard constraints: the Lagrangian approach maximizes \(J(\theta) - \lambda \cdot g(\theta)\) where \(g(\theta)\) measures constraint violation (e.g., \(g = \mathbb{E}[\max(0, p_{\min} - a)] + \mathbb{E}[\max(0, a - p_{\max})]\)). The multiplier \(\lambda\) is updated via dual ascent to enforce feasibility.

**Sample efficiency**: real pricing data is expensive. Each exploratory price change has a revenue opportunity cost. **Offline RL** (Levine et al., 2020) addresses this by learning a policy entirely from historical data — past pricing decisions and their outcomes — without any further exploration. The challenge is distribution shift: the historical data was generated by a different policy (the old pricing algorithm), and the new policy might query state-action pairs that are poorly represented in the data. Conservative algorithms like CQL (Conservative Q-Learning) address this by penalizing Q-values for out-of-distribution actions.

### State of Practice

Despite the theoretical appeal of deep RL for pricing, most production systems use simpler methods — contextual bandits, heuristic dynamic programming, or rule-based systems. The reasons are practical: simpler methods are easier to debug (when the price looks wrong, you can trace back to which feature or which confidence bound drove the decision), easier to explain to stakeholders (a neural network Q-function is a black box), and easier to constrain (business rules are straightforward to implement as hard constraints on bandit actions, but tricky to encode in a neural network's loss function).

Deep RL is most valuable for complex, multi-product, multi-period problems where the state space is genuinely high-dimensional and the inter-temporal dynamics are too complex for tabular methods. Examples include managing pricing for a portfolio of 100+ SKUs with inventory interactions over a multi-week horizon — problems where the Bellman equation has billions of states and the transition dynamics involve complex substitution patterns.

---

## 11.5. Offline Reinforcement Learning for Pricing

The deep RL methods above assume you can explore — try different prices and observe outcomes in real time. But real-time exploration is expensive and risky. Every suboptimal price costs real revenue. Charging $5 to "explore" when the optimal price is $25 means you just lost $20 on that customer. For a retailer processing millions of transactions daily, even a small exploration rate can cost millions of dollars per year.

**The question**: can we learn a good pricing policy from **historical data** — a log of past prices, customer features, and outcomes — without any further exploration?

### The Offline RL Problem

You have a historical dataset \(\mathcal{D} = \{(s_i, a_i, r_i, s_i')\}_{i=1}^n\) collected by a **behavior policy** \(\mu\) — the old pricing system (perhaps rule-based, perhaps a previous-generation algorithm, perhaps human analysts). Each record is a state (customer features, inventory, time), the action taken (price charged), the reward observed (revenue), and the next state. You want to learn a new policy \(\pi\) that achieves higher revenue, but you cannot deploy \(\pi\) to collect new data. You must evaluate and optimize \(\pi\) entirely from the historical data.

This is fundamentally different from the online setting. In online RL, if your estimate of some state-action pair is wrong, the agent will eventually visit that pair and correct the error. In offline RL, if the historical data never contains a particular state-action pair (e.g., charging $99 for a product that was always priced at $20-$30), you have no information about it. Extrapolating to unseen actions is dangerous.

### Off-Policy Evaluation (OPE)

Before deploying a new policy, you need to estimate its expected revenue from historical data. This is **off-policy evaluation** — evaluating one policy (the new policy \(\pi\)) using data generated by a different policy (the behavior policy \(\mu\)).

**Importance sampling (IS)** is the foundational technique. The key identity:

$$
V(\pi) = \mathbb{E}_\mu\left[\prod_{t=0}^{H-1} \frac{\pi(a_t \mid s_t)}{\mu(a_t \mid s_t)} \sum_{t=0}^{H-1} \gamma^t r_t\right]
$$

where the product \(\prod_t \pi(a_t|s_t)/\mu(a_t|s_t)\) is the **importance weight** (or likelihood ratio) that reweights trajectories from the behavior policy to "look like" trajectories from the target policy. For a single-step (bandit) setting, this simplifies to:

$$
\hat{V}_{\text{IS}}(\pi) = \frac{1}{n}\sum_{i=1}^{n} \frac{\pi(a_i \mid s_i)}{\mu(a_i \mid s_i)} \cdot r_i
$$

This estimator is **unbiased** — \(\mathbb{E}[\hat{V}_{\text{IS}}(\pi)] = V(\pi)\) — but it can have astronomical variance. If the new policy \(\pi\) frequently selects actions that the behavior policy \(\mu\) rarely took, the importance weight \(\pi(a|s)/\mu(a|s)\) is huge, amplifying noise. Over a multi-step horizon, these ratios multiply, and the product can be astronomically large for even a modest number of steps. This is the **curse of horizon** in off-policy evaluation.

**The doubly robust (DR) estimator** combines a model-based estimate with an IS correction:

$$
\hat{V}_{\text{DR}}(\pi) = \frac{1}{n}\sum_{i=1}^{n}\left[\hat{V}_{\text{model}}(s_i) + \frac{\pi(a_i \mid s_i)}{\mu(a_i \mid s_i)}\left(r_i - \hat{Q}_{\text{model}}(s_i, a_i)\right)\right]
$$

where \(\hat{V}_{\text{model}}\) and \(\hat{Q}_{\text{model}}\) are model-based estimates of the value function and Q-function. The DR estimator is unbiased if either the model or the importance weights are correct (hence "doubly robust"), and it has lower variance than pure IS when the model is approximately correct. The IS correction term only activates when the model's prediction \(\hat{Q}_{\text{model}}(s_i, a_i)\) differs from the observed reward \(r_i\) — it corrects for model errors using the reweighting trick.

### Conservative Q-Learning (CQL)

The most dangerous failure mode of offline RL is **overestimation of out-of-distribution actions**. Standard Q-learning uses the Bellman update:

$$
Q(s, a) \leftarrow r + \gamma \max_{a'} Q(s', a')
$$

The \(\max\) operator is the problem. If the Q-values for actions not well-represented in the data are initialized randomly (or are poorly estimated), the \(\max\) selects the largest of these noisy estimates, biasing the value upward. The agent then tries to take these overestimated actions, which the historical data cannot validate. In pricing: the Q-function might estimate that charging $999 yields enormous revenue, simply because no data exists at that price to contradict the estimate.

**Conservative Q-Learning** (CQL; Kumar et al., 2020) fixes this by adding a regularization term that pushes down Q-values for out-of-distribution actions:

$$
Q_{\text{CQL}} = \arg\min_Q \; \alpha \left(\mathbb{E}_{a \sim \pi}[Q(s, a)] - \mathbb{E}_{a \sim \mu}[Q(s, a)]\right) + \frac{1}{2}\mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[\left(Q(s, a) - \hat{\mathcal{B}}^\pi Q(s, a)\right)^2\right]
$$

The first term is the CQL penalty. It penalizes Q-values under the policy distribution \(\pi\) (which might query unseen actions) while rewarding Q-values under the data distribution \(\mu\) (which are well-supported). The net effect: actions that appear in the data retain their estimated value, while actions that don't appear get their Q-values pushed down. The parameter \(\alpha\) controls the conservatism — larger \(\alpha\) means more aggressive penalization.

The result is a **lower bound** on the true Q-function for the learned policy. The deployed policy might underestimate the revenue from some prices, but it won't overestimate the revenue from prices it has never tried. This is exactly the right kind of conservatism for pricing — you'd rather underestimate the revenue from a novel price (and not try it) than overestimate it (and lose money).

### Batch-Constrained Q-Learning (BCQ)

An alternative approach is **BCQ** (Fujimoto, Meger, and Precup, 2019), which directly constrains the policy to only select actions that are "close" to actions observed in the data:

$$
\pi(s) = \arg\max_{a : G(s, a) > \tau} Q(s, a)
$$

where \(G(s, a)\) is a generative model (e.g., a VAE or conditional density estimator) trained on the data distribution, and \(\tau\) is a threshold. The policy only considers actions that \(G\) deems plausible given the state — actions that the behavior policy might have taken. If the historical data never charged above $50, BCQ won't consider prices above $50.

### Application to Pricing

Consider a retailer with 2 years of pricing data from a rule-based system. The data contains millions of transactions: for each, the product, customer features, time, the price that was charged, and whether a sale occurred. The retailer wants to deploy an ML-based pricing engine.

The pipeline:

1. **Estimate the behavior policy** \(\mu(a|s)\) from historical data — what price did the old system tend to charge in each context? This is needed for importance weighting.
2. **Train an offline RL agent** (CQL or BCQ) on the historical data. The agent learns a Q-function and derives a new policy \(\pi\).
3. **Evaluate \(\pi\) off-policy** using the DR estimator to estimate expected revenue improvement.
4. **Deploy with guardrails**: start by deploying \(\pi\) on a small fraction of traffic (A/B test), compare against the old policy, and gradually roll out if performance matches the off-policy estimate.

The key advantage: the retailer can train, evaluate, and iterate on the pricing policy *before* any customer sees a new price. The risk of catastrophic exploration is zero. The tradeoff: offline RL can only be as good as the data allows — if the historical data never explored a promising pricing region, the offline agent can't discover it either. This is why offline RL is typically used for the initial deployment, followed by cautious online fine-tuning (online RL with safety constraints) once the system is live.

---

## 12. Fairness Constraints in Algorithmic Pricing

Pricing algorithms optimize revenue. But unconstrained optimization can produce outcomes that society, regulators, and customers consider **unfair**. When an algorithm charges different prices to different people, the line between efficient personalized pricing and discriminatory exploitation becomes blurred.

### Price Discrimination and the Law

As we discussed in Part 2, price discrimination — charging different prices to different customers for the same product — is economically efficient: it moves prices closer to each customer's willingness to pay, extracting more surplus and often serving customers who would be priced out of a uniform market. But legal and ethical frameworks place constraints on *which* customer characteristics can be used for pricing.

In the United States, the Robinson-Patman Act (1936) prohibits price discrimination between *businesses* that harms competition, but generally does not apply to consumer pricing. However, charging different prices based on **protected attributes** — race, sex, religion, national origin, and in some jurisdictions age or disability — is illegal under various civil rights statutes.

The subtlety is that algorithmic pricing systems can learn **proxies** for protected attributes from behavioral data without ever explicitly using the protected attribute. A contextual bandit that uses ZIP code as a feature may effectively discriminate by race due to residential segregation. An algorithm that uses browsing behavior (device type, time of browsing, website referrer) may indirectly infer income, age, or race. The algorithm doesn't "know" it's discriminating — it simply found that customers with certain behavioral patterns are less price-sensitive and charged them more.

### Fairness Definitions for Pricing

The machine learning fairness literature offers several definitions that can be adapted to pricing:

**Demographic parity**: the price distribution should be the same across protected groups. Formally, for protected groups \(A\) and \(B\):

$$
\mathbb{E}[P \mid \text{group} = A] = \mathbb{E}[P \mid \text{group} = B]
$$

This is the simplest definition but also the most restrictive. It forbids *any* price difference between groups, even if the groups have genuinely different willingness to pay for legitimate reasons (e.g., different product usage patterns, different outside options).

**Equalized welfare**: instead of equalizing prices, equalize **consumer surplus** — the welfare each customer derives from the transaction. A customer who pays a price close to their willingness to pay gets little surplus; one who pays much less gets a lot. Equalized welfare allows price differences that reflect cost-to-serve differences but prohibits price differences that disproportionately reduce one group's welfare.

**Individual fairness** (Dwork, Hardt, Pitassi, Reingold, and Zemel, 2012): similar customers should receive similar prices. Formally:

$$
|P(\mathbf{x}) - P(\mathbf{x}')| \leq L \cdot d(\mathbf{x}, \mathbf{x}')
$$

for a task-specific metric \(d(\mathbf{x}, \mathbf{x}')\) that measures the "similarity" of customers \(\mathbf{x}\) and \(\mathbf{x}'\). If two customers have the same demand characteristics (same usage, same alternatives, same cost-to-serve), they should get the same price, regardless of their protected attributes. The metric \(d\) defines what "same demand characteristics" means — and choosing it well is the hard part.

**Envy-freeness**: no customer should prefer another customer's price-product bundle to their own. Formally, customer \(i\) does not envy customer \(j\) if \(u_i(q_i, p_i) \geq u_i(q_j, p_j)\), where \(u_i\) is customer \(i\)'s utility and \((q_j, p_j)\) is the bundle offered to customer \(j\). Envy-freeness is a strong condition that's closely related to incentive compatibility in mechanism design — it ensures that customers don't want to "pretend" to be a different type to get a better deal.

### Constrained Optimization: The Lagrangian Approach

To incorporate fairness into a pricing algorithm, add fairness constraints to the optimization objective. The general framework is:

$$
\max_\pi \; \mathbb{E}[\text{Revenue}(\pi)] \quad \text{subject to} \quad g(\pi) \leq 0
$$

where \(g(\pi)\) is a fairness violation measure (e.g., \(g = |\mathbb{E}[P | A] - \mathbb{E}[P | B]|\) for demographic parity).

The **Lagrangian relaxation** converts this to an unconstrained problem:

$$
\max_\pi \min_{\lambda \geq 0} \; \mathbb{E}[\text{Revenue}(\pi)] - \lambda \cdot g(\pi)
$$

Dual ascent alternates between:
1. Fix \(\lambda\), optimize \(\pi\) (this is a modified bandit/MDP where the reward includes a penalty for unfairness).
2. Fix \(\pi\), update \(\lambda \leftarrow \lambda + \eta \cdot g(\pi)\) (increase the penalty if the fairness constraint is violated, decrease it if satisfied).

This converges to a policy that maximizes revenue subject to the fairness constraint. The Lagrange multiplier \(\lambda^*\) at convergence represents the **shadow price of fairness** — the marginal revenue cost of tightening the fairness constraint by one unit.

### The Price of Fairness

Imposing fairness constraints reduces achievable revenue. The gap between unconstrained optimal revenue and constrained optimal revenue is the **price of fairness** — the economic cost of being fair.

Empirical studies (Kallus and Zhou, 2021; Cohen, Perakis, and Puspita, 2021) find that the Pareto frontier between revenue and fairness is typically **concave**: the first few percent of fairness improvement cost very little revenue, but strict equality can be very expensive. Intuitively, the most egregious price differences (charging a vulnerable group 3x more) are also the most inefficient (those customers have high elasticity at the inflated price), so eliminating them improves both fairness and revenue. It's only when you push toward strict equality across groups with genuinely different demand characteristics that the revenue cost becomes significant.

This means firms often face a reasonable tradeoff: a modest fairness constraint eliminates the worst discriminatory outcomes with minimal revenue impact, making it a relatively cheap way to manage legal and reputational risk.

### The Regulatory Landscape

Regulation is catching up to algorithmic pricing:

- The **EU AI Act** (2024) classifies AI-based pricing in certain high-risk domains (insurance, credit, essential services) as requiring transparency, human oversight, and bias auditing.
- **California's Automated Decision Systems** regulations require businesses to explain how automated systems affect consumers, including pricing.
- The **FTC** (US Federal Trade Commission) has investigated personalized pricing practices and has authority to challenge unfair or deceptive pricing under Section 5 of the FTC Act.
- The **UK Competition and Markets Authority** has published guidance on algorithmic pricing and fairness, focusing on whether personalized pricing harms consumer welfare.

The trend is clear: firms deploying algorithmic pricing systems will increasingly need to demonstrate that their algorithms do not unfairly discriminate, that customers can understand why they received a particular price, and that human oversight exists. Building fairness constraints into the algorithm from the start is both ethically sound and a smart regulatory strategy.

---

## 13. Python — GP-Bandit and Non-Stationary Demand

### Part A: Gaussian Process Bandit for Continuous Pricing

We implement GP-UCB from scratch to find the revenue-maximizing price on a continuous interval. The true revenue function is bell-shaped: \(\text{revenue}(p) = p \cdot \sigma(-0.15(p - 25))\), where \(\sigma\) is the sigmoid function. The GP learns this function from noisy observations, concentrating its exploration around the optimum.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- True revenue function ---
def true_revenue(p):
    """Revenue = price * sigmoid(-0.15 * (price - 25))"""
    return p / (1.0 + np.exp(0.15 * (p - 25.0)))

# Find true optimum on a fine grid
p_fine = np.linspace(5, 50, 1000)
r_fine = true_revenue(p_fine)
p_star = p_fine[np.argmax(r_fine)]
r_star = np.max(r_fine)

# --- GP-UCB implementation ---
# RBF kernel
def rbf_kernel(p1, p2, sigma_f=10.0, length_scale=5.0):
    """Squared exponential kernel."""
    p1 = np.atleast_1d(p1)
    p2 = np.atleast_1d(p2)
    sqdist = (p1[:, None] - p2[None, :]) ** 2
    return sigma_f**2 * np.exp(-sqdist / (2 * length_scale**2))

# GP posterior
def gp_posterior(X_obs, y_obs, X_pred, sigma_noise=1.0,
                 sigma_f=10.0, length_scale=5.0):
    """Compute GP posterior mean and variance at X_pred."""
    K = rbf_kernel(X_obs, X_obs, sigma_f, length_scale)
    K += sigma_noise**2 * np.eye(len(X_obs))
    K_s = rbf_kernel(X_obs, X_pred, sigma_f, length_scale)
    K_ss = rbf_kernel(X_pred, X_pred, sigma_f, length_scale)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
    mu = K_s.T @ alpha

    v = np.linalg.solve(L, K_s)
    var = np.diag(K_ss) - np.sum(v**2, axis=0)
    var = np.maximum(var, 1e-10)  # numerical stability
    return mu, var

# --- Run GP-UCB ---
n_rounds = 50
sigma_noise = 2.0
p_grid = np.linspace(5, 50, 200)  # grid for acquisition function

X_obs = np.array([])
y_obs = np.array([])

# Initial observations: two endpoints
for p_init in [10.0, 40.0]:
    r_init = true_revenue(p_init) + np.random.randn() * sigma_noise
    X_obs = np.append(X_obs, p_init)
    y_obs = np.append(y_obs, r_init)

for t in range(n_rounds):
    mu, var = gp_posterior(X_obs, y_obs, p_grid,
                           sigma_noise=sigma_noise)
    sigma = np.sqrt(var)

    # GP-UCB acquisition: beta_t = 2 * log(t+3)
    beta_t = 2.0 * np.log(t + 3)
    ucb = mu + np.sqrt(beta_t) * sigma
    p_next = p_grid[np.argmax(ucb)]

    # Observe noisy reward
    r_next = true_revenue(p_next) + np.random.randn() * sigma_noise
    X_obs = np.append(X_obs, p_next)
    y_obs = np.append(y_obs, r_next)

# --- Final posterior for plotting ---
mu_final, var_final = gp_posterior(X_obs, y_obs, p_fine,
                                    sigma_noise=sigma_noise)
sigma_final = np.sqrt(var_final)

# --- Part B: Non-stationary pricing ---
np.random.seed(123)

T_ns = 1500
K_ns = 10
prices_ns = np.linspace(5, 50, K_ns)

def purchase_prob_ns(price, wtp):
    return 1.0 / (1.0 + np.exp(0.2 * (price - wtp)))

def get_optimal_arm(wtp):
    rev = prices_ns * purchase_prob_ns(prices_ns, wtp)
    return np.argmax(rev), np.max(rev)

# WTP shifts at t=500: from 20 to 35
wtp_schedule = np.where(np.arange(T_ns) < 500, 20.0, 35.0)

# Standard Thompson Sampling
ts_alpha = np.ones(K_ns)
ts_beta = np.ones(K_ns)
ts_regret = np.zeros(T_ns)

for t in range(T_ns):
    wtp = wtp_schedule[t]
    opt_arm, opt_rev = get_optimal_arm(wtp)

    sampled = np.random.beta(ts_alpha, ts_beta)
    arm = np.argmax(prices_ns * sampled)

    prob = purchase_prob_ns(prices_ns[arm], wtp)
    sale = np.random.rand() < prob
    reward = prices_ns[arm] * sale

    if sale:
        ts_alpha[arm] += 1
    else:
        ts_beta[arm] += 1

    ts_regret[t] = opt_rev - prices_ns[arm] * prob

# Sliding Window Thompson Sampling (W=100)
W = 100
sw_history = []  # list of (arm, sale)
sw_regret = np.zeros(T_ns)

for t in range(T_ns):
    wtp = wtp_schedule[t]
    opt_arm, opt_rev = get_optimal_arm(wtp)

    # Compute posterior from last W observations
    sw_alpha = np.ones(K_ns)
    sw_beta = np.ones(K_ns)
    window_start = max(0, len(sw_history) - W)
    for arm_h, sale_h in sw_history[window_start:]:
        if sale_h:
            sw_alpha[arm_h] += 1
        else:
            sw_beta[arm_h] += 1

    sampled = np.random.beta(sw_alpha, sw_beta)
    arm = np.argmax(prices_ns * sampled)

    prob = purchase_prob_ns(prices_ns[arm], wtp)
    sale = np.random.rand() < prob
    reward = prices_ns[arm] * sale

    sw_history.append((arm, int(sale)))
    sw_regret[t] = opt_rev - prices_ns[arm] * prob

# --- Plot side by side ---
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# Left: GP posterior
ax = axes[0]
ax.fill_between(p_fine, mu_final - 2 * sigma_final,
                mu_final + 2 * sigma_final,
                alpha=0.25, color='#3498db', label=r'$\pm 2\sigma$ band')
ax.plot(p_fine, r_fine, '--', color='#e74c3c', linewidth=2,
        label='True revenue')
ax.plot(p_fine, mu_final, color='#3498db', linewidth=2,
        label='GP posterior mean')
ax.scatter(X_obs, y_obs, c='#2ecc71', s=30, zorder=5,
           edgecolors='white', linewidths=0.5, label='Observations')
ax.axvline(p_star, color='gray', linestyle=':', alpha=0.5)
ax.annotate(rf'$p^* = {p_star:.1f}$', xy=(p_star, r_star),
            xytext=(p_star + 5, r_star + 2),
            fontsize=10, color='#d4d4d4',
            arrowprops=dict(arrowstyle='->', color='#d4d4d4'))
ax.set_xlabel(r'Price $p$', fontsize=12)
ax.set_ylabel(r'Revenue $\mu(p)$', fontsize=12)
ax.set_title('GP-UCB for Continuous Pricing (50 rounds)', fontsize=13)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

# Right: Non-stationary regret
ax = axes[1]
ax.plot(np.cumsum(ts_regret), color='#e74c3c', linewidth=1.5,
        label='Standard Thompson Sampling')
ax.plot(np.cumsum(sw_regret), color='#2ecc71', linewidth=1.5,
        label=f'Sliding Window TS ($W={W}$)')
ax.axvline(500, color='#f1c40f', linestyle='--', alpha=0.7,
           label='Demand shift at $t=500$')
ax.set_xlabel(r'Round $t$', fontsize=12)
ax.set_ylabel(r'Cumulative Regret $R_t$', fontsize=12)
ax.set_title('Non-Stationary Demand: Standard vs Sliding-Window TS',
             fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gp_bandit_nonstationary.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Left panel**: the GP posterior after 50 rounds of GP-UCB. The blue shaded band shows the uncertainty — narrow near observed data points, wider elsewhere. The GP mean closely tracks the true revenue curve (dashed red) in the region around the optimum. Most observations cluster near the peak, showing that GP-UCB efficiently focused its exploration on the revenue-maximizing region rather than wasting rounds at clearly suboptimal prices.

**Right panel**: cumulative regret under a demand shift. At \(t = 500\), the optimal price jumps from ~$15 to ~$30 (due to a WTP shift from $20 to $35). Standard Thompson Sampling, which uses all historical data equally, is slow to adapt — its posterior is dominated by the 500 pre-shift observations, and it keeps charging the old optimal price, accumulating linear regret after the shift. Sliding-window Thompson Sampling (\(W = 100\)) quickly forgets the pre-shift data and re-learns the new optimal price, recovering within roughly 100 rounds. The divergence in cumulative regret after the shift clearly demonstrates why non-stationary methods are essential for production pricing systems.

---

## 14. The Pricing MDP: When Bandits Aren't Enough

Bandits assume each round is **independent** — today's price choice doesn't affect tomorrow's state of the world. But in many pricing problems, this assumption breaks down:

- **Inventory depletes**: airline seats, hotel rooms, concert tickets — when you sell a unit, it's gone. Today's sale reduces tomorrow's available supply, which should change tomorrow's price.
- **Reference price effects**: customers form expectations about your price. If you charge $20 for weeks and then raise to $30, there's a backlash (anchoring). Today's price affects tomorrow's demand curve.
- **Competitor dynamics**: your price cut today triggers a competitor's response tomorrow.

When the current action affects the future state, the right framework is a **Markov Decision Process** (MDP). An MDP is defined by:

- **State space** \(\mathcal{S}\): the state \(s_t\) captures everything relevant about the current situation (remaining inventory, time, competitor prices, recent demand signals)
- **Action space** \(\mathcal{A}\): the available prices \(a_t\)
- **Transition function** \(P(s_{t+1} | s_t, a_t)\): how the state evolves given the current state and action
- **Reward function** \(r(s_t, a_t)\): the immediate revenue from choosing action \(a_t\) in state \(s_t\)
- **Discount factor** \(\gamma \in [0, 1]\): how much we value future rewards relative to current ones

The goal is to find a **policy** \(\pi(s)\) — a mapping from states to actions — that maximizes the expected cumulative discounted reward:

$$
V^\pi(s) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \, r(s_t, \pi(s_t)) \;\middle|\; s_0 = s\right]
$$

The optimal policy satisfies the **Bellman equation**:

$$
V^*(s) = \max_{a \in \mathcal{A}} \left[ r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' | s, a) \, V^*(s') \right]
$$

This is a recursive equation: the value of a state equals the best immediate reward plus the discounted value of the next state. The optimal action in each state is the one that maximizes this sum.

For **revenue management** with finite inventory: the state is \(s_t = (\tau, n)\) where \(\tau\) is the time remaining (days until departure, end of season, etc.) and \(n\) is the remaining inventory. The state space is finite, so the Bellman equation can be solved exactly via **backward induction** — start from the terminal state (no time left), compute the value of every state at time \(\tau = 1\), then \(\tau = 2\), and so on. For large state spaces (many products, high-dimensional context), reinforcement learning methods like Q-learning and policy gradient algorithms approximate the solution.

---

## 15. Airline Revenue Management

Airline revenue management is the original dynamic pricing problem — and arguably the most sophisticated. American Airlines pioneered it in the 1980s, and the practice is credited with generating hundreds of millions of dollars in additional annual revenue.

The setting is: you have a fixed capacity (say 200 seats on a flight), a finite horizon (the departure date), and multiple fare classes (economy, premium economy, business, first). Once the plane departs, unsold seats have zero value. Demand arrives stochastically over time, and typically the willingness to pay *increases* as departure approaches (business travelers book late and are less price-sensitive; leisure travelers book early and hunt for deals).

The foundational result is **Littlewood's rule** (1972), which handles the simplest case: two fare classes. You have a high fare \(f_H\) (business class) and a low fare \(f_L\) (discount economy). Low-fare demand comes first (early bookers), and you need to decide how many seats to "protect" for late-arriving high-fare customers.

Littlewood's rule says: **accept a low-fare booking if and only if the low fare exceeds the expected marginal revenue of saving that seat for a potential high-fare customer**. Formally:

$$
\text{Accept } f_L \iff f_L \geq f_H \cdot \Pr(\text{high-fare demand} > \text{remaining seats})
$$

The right-hand side is the **option value** of the seat — the expected revenue you'd get by holding it for a high-fare customer. If there are many seats remaining relative to expected high-fare demand, the option value is low (most seats won't be needed), so you should accept the low fare. If seats are scarce, the option value is high, and you should protect them.

The generalization to multiple fare classes is the **EMSR** (Expected Marginal Seat Revenue) method. EMSR-b, the most common variant, computes protection levels for each fare class by comparing the fare of lower classes against the expected marginal revenue from higher classes.

Modern airline revenue management goes far beyond these heuristics. The full problem is a **stochastic dynamic program** where the state is \((\tau, \mathbf{n})\) — time remaining and available seats by class — and the action is which fares to offer. Airlines estimate demand curves per route, per departure date, using historical booking data, controlling for seasonality, day-of-week, competition, and special events. This is exactly the **causal estimation** from Part 4 — demand for a specific flight depends on the prices offered (endogenous), on observable features (exogenous), and on unobservable shocks. Getting the causal effect right is critical: if you naively regress bookings on price, you get biased estimates because prices are set in response to demand signals.

### Network Revenue Management: The Multi-Leg Problem

A single flight is one "leg." But airlines sell **itineraries**: a customer flying New York to London to Paris uses two legs (JFK-LHR and LHR-CDG). Accepting this booking consumes one seat on *each* leg. The revenue management problem must consider the **network** of legs simultaneously — accepting a cheap connecting itinerary might block a more profitable direct booking on one of the legs.

**The network LP.** Let \(i\) index itineraries (origin-destination pairs with routing) and \(j\) index legs. Define:

- \(f_i\) — the fare for itinerary \(i\)
- \(\mathbf{A}\) — the **incidence matrix** where \(A_{ji} = 1\) if itinerary \(i\) uses leg \(j\), and 0 otherwise
- \(c_j\) — the capacity (seats) of leg \(j\)
- \(d_i\) — the expected demand for itinerary \(i\)

The **deterministic linear program** (DLP) for network revenue management is:

$$
\max_{\mathbf{x}} \sum_{i} f_i x_i
$$

$$
\text{subject to:} \quad \sum_{i} A_{ji} x_i \leq c_j \quad \forall j, \qquad 0 \leq x_i \leq d_i \quad \forall i
$$

The decision variable \(x_i\) represents how many bookings to accept for itinerary \(i\). The first constraint says total bookings using leg \(j\) cannot exceed its capacity. The second says you can't sell more than the demand.

**Bid prices from the dual.** The dual variables \(\pi_j\) of the capacity constraints are the **bid prices** — they represent the marginal value of one additional seat on leg \(j\). The economic interpretation is clean: \(\pi_j\) is the shadow price of capacity on leg \(j\), measuring how much the optimal revenue would increase if you had one more seat.

The bid-price control rule is: **accept a booking request for itinerary \(i\) if and only if the fare exceeds the sum of bid prices for the legs used**:

$$
\text{Accept itinerary } i \iff f_i \geq \sum_{j} A_{ji} \pi_j
$$

The right-hand side, \(\sum_j A_{ji} \pi_j\), is the **displacement cost** — the opportunity cost of the seats consumed by itinerary \(i\) across all legs. A JFK-LHR-CDG booking at fare $800 should be accepted only if $800 exceeds the bid price for the JFK-LHR seat plus the bid price for the LHR-CDG seat. If those bid prices are $500 and $400 respectively (because both legs are scarce), the connecting booking should be rejected — those seats are worth more if sold separately to direct passengers.

**Dynamic bid prices.** The LP gives static bid prices based on expected demand. But in reality, as bookings arrive stochastically and capacities change, the bid prices need updating. Airlines re-solve the LP periodically (every few hours, or after every booking) to get updated bid prices that reflect the current capacity situation.

The DLP has a beautiful theoretical property: it provides an **upper bound** on the optimal revenue from the stochastic problem. The gap between DLP revenue and optimal revenue shrinks as capacity grows — a consequence of the law of large numbers. When capacities are large, the stochastic fluctuations in demand average out, and the deterministic approximation becomes tight.

**Randomized LP.** Talluri and van Ryzin proposed a refinement: solve the DLP, then use the bid prices to decompose the network problem into independent single-leg problems. Each leg uses its bid price to decide which itineraries to accept. Randomization handles the integrality issue — the LP solution may say "accept 15.7 bookings for itinerary \(i\)," which isn't implementable. The randomized approach converts these fractional decisions into probabilistic accept/reject rules.

**The computational scale.** A major airline like Delta or United operates roughly 5,000 flights per day, selling tickets on hundreds of thousands of itineraries (when you count all origin-destination pairs with connections). The incidence matrix \(\mathbf{A}\) has 5,000 rows (legs) and hundreds of thousands of columns (itineraries). The LP must be re-solved frequently as bookings arrive. Modern implementations use column generation and decomposition techniques to make this tractable — the full LP is too large to solve directly, but most itineraries are either clearly profitable or clearly unprofitable, so only a subset of "interesting" columns need to be actively managed.

**Revenue impact.** Network revenue management typically generates 2-5% additional revenue compared to managing each leg independently, which translates to hundreds of millions of dollars annually for a major carrier. The insight is that single-leg management systematically undervalues connecting seats (because it doesn't account for the displacement cost on other legs) and overaccepts cheap connecting fares that consume scarce hub capacity.

---

## 16. Uber's Surge Pricing

Uber operates a **two-sided market**: riders (demand) and drivers (supply). The price affects *both* sides simultaneously. When the price rises, some riders decide it's too expensive and cancel (demand decreases), while more drivers see the higher earnings and come online (supply increases). The price that balances the two is the **market-clearing price**, and Uber's surge pricing algorithm finds it in real time.

The **surge multiplier** works as follows: \(\text{price} = \text{base\_fare} \times \text{surge\_multiplier}\). When the multiplier is 1.0x, there's no surge. At 2.0x, the rider pays double and the driver earns roughly double.

What the algorithm actually does: Uber estimates the **supply and demand elasticities** in real time, per geographic zone. The supply elasticity tells you how many additional drivers come online when earnings increase by 1%. The demand elasticity tells you how many riders drop off when the price increases by 1%. The optimal surge multiplier balances these:

At the market-clearing multiplier \(m^*\), the quantity of rides demanded at price \(m^* \times \text{base\_fare}\) equals the quantity of rides drivers are willing to provide at the earnings \(m^*\) implies. In practice, Uber adds a margin above the strict market-clearing price.

The technical implementation: the city is divided into **hexagonal zones** (H3 geospatial indexing). Each zone has an independent pricing algorithm. The algorithm observes real-time request rates, driver availability, and estimated ETAs (a proxy for supply-demand imbalance — long ETAs mean supply is scarce). It adjusts the multiplier every few minutes. The whole system processes millions of events per second and updates prices across thousands of zones simultaneously.

Surge pricing is economically efficient — higher prices allocate rides to those who value them most and incentivize more supply. But it's also controversial. During emergencies (hurricanes, terrorist attacks), prices spike dramatically, which looks like price gouging. The economic argument is that high prices bring more drivers to the area, increasing supply where it's most needed. The fairness argument is that exploiting desperate people is unconscionable, regardless of supply effects. Uber has responded by capping surge during declared emergencies — a departure from pure market-clearing pricing in response to social pressure.

---

## 17. Amazon's Pricing Engine

Amazon changes prices approximately **2.5 million times per day** across its catalog. This isn't a team of analysts making decisions — it's a fully automated system that ingests data, estimates demand, and sets prices at machine speed.

The system ingests: competitor prices (via web scraping and data feeds), internal demand patterns, current and projected inventory levels, fulfillment costs (which vary by warehouse, shipping method, and distance), time features (hour, day-of-week, season), and customer behavior signals (search volume, click-through rates, cart additions).

For each SKU, the pipeline is roughly:

1. **Estimate demand elasticity** using causal methods from Part 4 — instrumental variables, difference-in-differences, or double ML — to isolate the true price effect from confounders.
2. **Forecast demand** given a candidate price, using time-series models that incorporate seasonality, trends, and external features.
3. **Optimize price** using a combination of the Lerner-style markup (\(p^* = c / (1 + 1/\varepsilon)\), from Part 1) and competitive positioning relative to other sellers.

**The inventory-price feedback loop** is particularly important. As inventory decreases, the algorithm incrementally raises prices to slow sales velocity and prevent stockouts — the opportunity cost of selling the last unit is high because a future customer might value it more. When inventory is replenished, prices decrease to stimulate demand and clear new stock. This is a direct application of the dynamic programming framework from Section 14: the state includes inventory, and the optimal price depends on how much stock remains.

**Loss leaders**: Amazon deliberately prices popular, highly elastic items at or below cost to drive traffic and Prime subscriptions. When customers come for the cheap electronics deal, they also buy Amazon Basics batteries, subscribe to Audible, and use AWS. Profit comes from **less elastic categories** where customers aren't price-comparing. This is the Lerner index in action at a strategic level: zero or negative markup on goods with \(|\varepsilon| \to \infty\) (perfectly elastic — customers will buy from whoever is cheapest), fat margins on goods with low \(|\varepsilon|\) (inelastic — customers buy from Amazon regardless of price).

---

## 18. The Buy Box Game

Most products on Amazon have **multiple sellers** offering the same item. The **Buy Box** — the prominent "Add to Cart" button — goes to one seller at a time. Winning the Buy Box is everything: it captures roughly 82% of Amazon's sales. If you don't have the Buy Box, your offer is buried in the "Other Sellers" section that almost nobody clicks.

Amazon's Buy Box algorithm considers: price (lower is better), seller metrics (ratings, on-time delivery, defect rate), Prime eligibility (strongly favored), and fulfillment method (FBA — Fulfillment by Amazon — is preferred over FBM — Fulfillment by Merchant). Price is necessary but not sufficient — the lowest price doesn't automatically win if the seller has poor metrics.

The Buy Box creates a **game-theoretic structure** among sellers, connecting directly to Part 3. Multiple sellers with repricing algorithms are playing a repeated game. But the game has a twist: **Buy Box rotation**. Amazon doesn't always give the Buy Box to the single lowest-priced eligible seller. Instead, it rotates among sellers with competitive prices and good metrics, giving each a share of Buy Box time roughly proportional to their competitiveness.

This rotation mechanism has a profound consequence. In a standard Bertrand competition (Part 3), sellers undercut each other until prices reach marginal cost — the Bertrand paradox. But Buy Box rotation converts this into something closer to a **tacit collusion equilibrium**. Here's why: if aggressive undercutting doesn't guarantee you the Buy Box (because Amazon rotates anyway), and if matching competitors' prices gives you a fair share of rotation time, then the rational strategy is **mutual restraint**. Sellers learn that price wars aren't rewarded and that keeping prices modestly competitive yields a steady share of the Buy Box.

This creates the conditions for **algorithmic collusion through the platform** — a phenomenon we discussed theoretically in Part 3. Repricing algorithms (like RepricerExpress, Feedvisor, and Amazon's own Automate Pricing tool) adopt similar strategies: match the current Buy Box price rather than aggressively undercutting, raise prices incrementally when others do, and avoid triggering price wars. The result is supra-competitive prices sustained without any explicit coordination between sellers. The platform architecture itself serves as the coordination mechanism.

This is a live, large-scale example of the theoretical concern from Part 3: algorithms can achieve collusive outcomes by independently learning the same equilibrium strategy, facilitated by a platform that rewards restraint over aggression.

---

## 19. Implementation at Scale

Building a production pricing system at Amazon-scale involves a full ML pipeline. Here's how the pieces fit together:

**1. Feature Engineering**: For each SKU, construct features including competitor prices (scraped or from data feeds), current inventory level and restock schedule, historical sales velocity (units per day at each price point), time features (hour, day-of-week, month, holiday flags), product attributes (category, brand, rating, review count), customer behavior signals (search impressions, click-through rate, cart-add rate), and fulfillment costs.

**2. Demand Estimation**: Use the causal methods from Part 4 — double/debiased ML, instrumental variables, or natural experiments — to estimate unbiased price elasticity per product. This is critical. Naive demand estimation overstates elasticity (because prices are correlated with demand shocks), leading to prices that are systematically too low.

**3. Demand Forecasting**: Time-series models (Transformers, Prophet, DeepAR) predict demand given a candidate price. These models capture seasonality, trends, and the impact of external events.

**4. Online Optimization**: Contextual bandits for continuous price experimentation. Each price change is both a revenue-generating action and a data-collecting experiment. Thompson Sampling with a prior informed by the demand model handles the explore-exploit tradeoff.

**5. Constraint Enforcement**: Business rules overlay the optimization: minimum price floors (to maintain brand value), maximum price ceilings (to avoid PR disasters), inventory-aware adjustments, MAP (Minimum Advertised Price) compliance, and fairness rules (e.g., no surge pricing during emergencies).

**6. Deployment**: Prices updated every 10-15 minutes for millions of SKUs. The system must be low-latency, fault-tolerant, and auditable.

**The cold-start problem**: when a new product launches, there's no historical data. Thompson Sampling handles this naturally — use priors from similar products (same category, similar price range, comparable brand). The prior is informative enough to avoid wildly suboptimal initial prices but uncertain enough to allow rapid learning.

**The model refresh problem**: demand changes over time. Consumer preferences shift, competitors enter or exit, macroeconomic conditions evolve. The system uses continuous online learning with **exponential forgetting** — recent observations are weighted more than old ones. This ensures the model adapts to non-stationary demand without being whipsawed by noise.

---

## 20. Python Simulations

### Simulation 1: UCB vs Thompson Sampling for Pricing

We simulate a pricing problem with 10 discrete price points and an unknown demand curve. Customers have a willingness to pay (WTP) that follows a logistic purchase probability: \(P(\text{sale} | p) = 1 / (1 + \exp(\beta(p - \text{WTP})))\). Revenue at each price is \(p \times P(\text{sale} | p)\), which is bell-shaped — too low a price means high sales but low per-unit revenue; too high means high per-unit revenue but few sales.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Problem setup
prices = np.linspace(5, 50, 10)  # 10 candidate prices
wtp = 25.0    # true mean willingness to pay
beta = 0.2    # demand sensitivity

# True purchase probability and expected revenue at each price
purchase_prob = 1.0 / (1.0 + np.exp(beta * (prices - wtp)))
true_revenue = prices * purchase_prob
optimal_arm = np.argmax(true_revenue)
optimal_revenue = true_revenue[optimal_arm]

T = 10000  # rounds
K = len(prices)

# --- UCB1 ---
ucb_rewards = np.zeros(T)
ucb_counts = np.zeros(K)
ucb_sum = np.zeros(K)
ucb_arm_history = np.zeros(T, dtype=int)

for t in range(T):
    if t < K:
        arm = t  # pull each arm once
    else:
        ucb_values = ucb_sum / ucb_counts + np.sqrt(2 * np.log(t) / ucb_counts)
        arm = np.argmax(ucb_values)
    
    # Simulate: does the customer buy?
    sale = np.random.rand() < purchase_prob[arm]
    reward = prices[arm] * sale
    
    ucb_counts[arm] += 1
    ucb_sum[arm] += reward
    ucb_rewards[t] = reward
    ucb_arm_history[t] = arm

ucb_regret = np.cumsum(optimal_revenue - true_revenue[ucb_arm_history])

# --- Thompson Sampling ---
ts_rewards = np.zeros(T)
ts_alpha = np.ones(K)  # Beta prior: alpha = 1
ts_beta_param = np.ones(K)   # Beta prior: beta = 1
ts_arm_history = np.zeros(T, dtype=int)

for t in range(T):
    # Sample from Beta posterior, compute sampled revenue
    sampled_prob = np.random.beta(ts_alpha, ts_beta_param)
    sampled_revenue = prices * sampled_prob
    arm = np.argmax(sampled_revenue)
    
    # Simulate
    sale = np.random.rand() < purchase_prob[arm]
    reward = prices[arm] * sale
    
    if sale:
        ts_alpha[arm] += 1
    else:
        ts_beta_param[arm] += 1
    
    ts_rewards[t] = reward
    ts_arm_history[t] = arm

ts_regret = np.cumsum(optimal_revenue - true_revenue[ts_arm_history])

# --- Plot cumulative regret ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(ucb_regret, label='UCB1', color='#e74c3c', linewidth=1.5)
ax.plot(ts_regret, label='Thompson Sampling', color='#2ecc71', linewidth=1.5)
ax.set_xlabel(r'Round $t$', fontsize=12)
ax.set_ylabel(r'Cumulative Regret $R_t$', fontsize=12)
ax.set_title('Cumulative Regret: UCB1 vs Thompson Sampling', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# --- Plot arm selection frequency over time ---
ax = axes[1]
window = 500
for alg_name, arm_hist, color in [('UCB1', ucb_arm_history, '#e74c3c'),
                                    ('Thompson', ts_arm_history, '#2ecc71')]:
    opt_frac = np.convolve(arm_hist == optimal_arm, np.ones(window)/window, mode='valid')
    ax.plot(opt_frac, label=f'{alg_name}', color=color, linewidth=1.5)

ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Optimal')
ax.set_xlabel(r'Round $t$', fontsize=12)
ax.set_ylabel(f'Fraction choosing optimal price (window={window})', fontsize=12)
ax.set_title(f'Convergence to Optimal Price (${prices[optimal_arm]:.2f})', fontsize=13)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ucb_vs_thompson.png', dpi=150, bbox_inches='tight')
plt.show()
```

Both algorithms achieve sublinear regret, but Thompson Sampling typically converges faster — its Bayesian uncertainty tracking is more efficient than UCB's frequentist confidence bounds. By round 10,000, both spend the vast majority of their time at the optimal price.

### Simulation 2: Contextual Bandit for Personalized Pricing

We simulate customers with different incomes. High-income customers have a higher willingness to pay. A LinUCB algorithm learns to charge different prices to different segments.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

# Setup: 5 price arms, customer features = [1, income]
K = 5
prices = np.array([10, 20, 30, 40, 50], dtype=float)
d = 2  # feature dimension: [1, normalized_income]

# True model: WTP = 15 + 30 * income (income in [0, 1])
# Purchase prob: logistic with beta=0.15
def true_purchase_prob(price, income):
    wtp = 15 + 30 * income
    return 1.0 / (1.0 + np.exp(0.15 * (price - wtp)))

T = 8000
alpha_param = 1.5  # exploration parameter

# LinUCB state: one model per arm
A = [np.eye(d) for _ in range(K)]
b = [np.zeros(d) for _ in range(K)]

chosen_prices = np.zeros(T)
incomes = np.zeros(T)
rewards = np.zeros(T)

for t in range(T):
    income = np.random.rand()
    x = np.array([1.0, income])
    incomes[t] = income
    
    # Compute UCB for each arm
    ucbs = np.zeros(K)
    for k in range(K):
        A_inv = np.linalg.inv(A[k])
        theta_hat = A_inv @ b[k]
        ucbs[k] = x @ theta_hat + alpha_param * np.sqrt(x @ A_inv @ x)
    
    arm = np.argmax(ucbs)
    price = prices[arm]
    chosen_prices[t] = price
    
    # Simulate sale
    prob = true_purchase_prob(price, income)
    sale = np.random.rand() < prob
    reward = price * sale
    rewards[t] = reward
    
    # Update LinUCB
    A[arm] = A[arm] + np.outer(x, x)
    b[arm] = b[arm] + reward * x

# --- Compute true optimal price per income level ---
income_grid = np.linspace(0, 1, 100)
optimal_prices_true = np.zeros(len(income_grid))
for i, inc in enumerate(income_grid):
    revs = [p * true_purchase_prob(p, inc) for p in prices]
    optimal_prices_true[i] = prices[np.argmax(revs)]

# --- Compute learned policy (last 2000 rounds) ---
fig, ax = plt.subplots(figsize=(8, 5))

# Scatter: chosen prices in last 2000 rounds
late = slice(T - 2000, T)
scatter = ax.scatter(incomes[late], chosen_prices[late], alpha=0.08,
                     c='#3498db', s=10, label='Learned policy (last 2000 rounds)')

# True optimal
ax.plot(income_grid, optimal_prices_true, color='#e74c3c', linewidth=2.5,
        label='True optimal price', zorder=5)

ax.set_xlabel('Customer Income (normalized)', fontsize=12)
ax.set_ylabel(r'Price Charged $p$', fontsize=12)
ax.set_title('LinUCB Learns Personalized Pricing', fontsize=13)
ax.set_yticks(prices)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('contextual_bandit_pricing.png', dpi=150, bbox_inches='tight')
plt.show()
```

The scatter plot shows the algorithm learning to charge higher prices to high-income customers and lower prices to low-income customers — it has independently discovered third-degree price discrimination from Part 2, purely from online experimentation.

### Simulation 3: Dynamic Pricing with Inventory (Airline Revenue Management)

We simulate an airline-style problem: 100 seats, 30 days to departure, with demand intensity that increases as departure approaches (late bookers are less price-sensitive). We compare three strategies: fixed price, two-price (low early, high late), and full dynamic pricing via backward induction on the Bellman equation.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(456)

# Parameters
total_seats = 100
total_days = 30
prices_available = np.array([100, 150, 200, 250, 300, 400, 500], dtype=float)

# Demand model: arrival rate * purchase probability
# Arrival rate increases as departure approaches
# Purchase probability decreases with price, but late arrivals are less price-sensitive

def demand_rate(price, day, days_total):
    """Expected number of customers willing to buy at this price on this day.
    day=0 is earliest, day=days_total-1 is day before departure."""
    # Arrival intensity increases toward departure
    lam = 3 + 7 * (day / days_total) ** 1.5
    # WTP increases toward departure (business travelers book late)
    wtp = 200 + 150 * (day / days_total)
    prob = 1.0 / (1.0 + np.exp(0.015 * (price - wtp)))
    return lam * prob

# --- Solve Bellman equation via backward induction ---
# V[tau][n] = optimal expected revenue with tau days remaining and n seats left
V = np.zeros((total_days + 1, total_seats + 1))
policy = np.zeros((total_days + 1, total_seats + 1), dtype=int)

for tau in range(1, total_days + 1):
    day = total_days - tau  # which calendar day this corresponds to
    for n in range(1, total_seats + 1):
        best_val = -np.inf
        best_price_idx = 0
        for ip, price in enumerate(prices_available):
            # Expected demand at this price on this day
            d = demand_rate(price, day, total_days)
            # Probability of selling k seats (Poisson approximation, cap at n)
            max_sell = min(n, 20)  # truncate Poisson
            expected_rev = 0
            for k in range(max_sell + 1):
                prob_k = np.exp(-d) * d**k / np.math.factorial(k)
                k_actual = min(k, n)
                expected_rev += prob_k * (k_actual * price + V[tau - 1][n - k_actual])
            # Remaining probability (k > max_sell)
            prob_remaining = 1.0 - sum(
                np.exp(-d) * d**j / np.math.factorial(j) for j in range(max_sell + 1)
            )
            expected_rev += prob_remaining * (n * price + V[tau - 1][0])
            
            if expected_rev > best_val:
                best_val = expected_rev
                best_price_idx = ip
        
        V[tau][n] = best_val
        policy[tau][n] = best_price_idx

# --- Simulate revenue for each strategy ---
n_simulations = 1000

def simulate_fixed_price(price, n_sims):
    revenues = np.zeros(n_sims)
    for sim in range(n_sims):
        seats = total_seats
        rev = 0
        for day in range(total_days):
            if seats <= 0:
                break
            d = demand_rate(price, day, total_days)
            arrivals = np.random.poisson(d)
            sold = min(arrivals, seats)
            rev += sold * price
            seats -= sold
        revenues[sim] = rev
    return revenues

def simulate_two_price(price_early, price_late, switch_day, n_sims):
    revenues = np.zeros(n_sims)
    for sim in range(n_sims):
        seats = total_seats
        rev = 0
        for day in range(total_days):
            if seats <= 0:
                break
            price = price_early if day < switch_day else price_late
            d = demand_rate(price, day, total_days)
            arrivals = np.random.poisson(d)
            sold = min(arrivals, seats)
            rev += sold * price
            seats -= sold
        revenues[sim] = rev
    return revenues

def simulate_dynamic(n_sims):
    revenues = np.zeros(n_sims)
    for sim in range(n_sims):
        seats = total_seats
        rev = 0
        for day in range(total_days):
            if seats <= 0:
                break
            tau = total_days - day
            price_idx = policy[tau][seats]
            price = prices_available[price_idx]
            d = demand_rate(price, day, total_days)
            arrivals = np.random.poisson(d)
            sold = min(arrivals, seats)
            rev += sold * price
            seats -= sold
        revenues[sim] = rev
    return revenues

rev_fixed = simulate_fixed_price(250, n_simulations)
rev_two = simulate_two_price(150, 350, 20, n_simulations)
rev_dynamic = simulate_dynamic(n_simulations)

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

for data, label, color in [(rev_fixed, f'Fixed ($250) — mean ${rev_fixed.mean():,.0f}',
                             '#e74c3c'),
                            (rev_two, f'Two-price ($150/$350) — mean ${rev_two.mean():,.0f}',
                             '#f39c12'),
                            (rev_dynamic, f'Dynamic (Bellman) — mean ${rev_dynamic.mean():,.0f}',
                             '#2ecc71')]:
    ax.hist(data, bins=40, alpha=0.5, color=color, label=label, edgecolor='white')

ax.set_xlabel('Total Revenue ($)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Revenue Distribution: Fixed vs Two-Price vs Dynamic Pricing\n'
             f'(100 seats, 30 days, {n_simulations} simulations)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dynamic_pricing_airline.png', dpi=150, bbox_inches='tight')
plt.show()
```

The dynamic pricing strategy — which adjusts prices optimally based on remaining inventory and time — consistently outperforms both the fixed-price and two-price strategies. The revenue distribution is shifted right and has lower variance. The dynamic policy charges low prices early (when inventory is plentiful and arrivals have low WTP) and ratchets up prices as departure approaches and seats become scarce. This is exactly Littlewood's rule in action: protect capacity for high-value late arrivals.

### Simulation 4: Bayesian Dynamic Pricing with Posterior Updates

This simulation implements the full Bayesian pricing framework from Section 4.5. We have a product with linear demand \(Q = \alpha + \beta p + \varepsilon\), where the true parameters \(\alpha = 50\) and \(\beta = -1.2\) are unknown. We start with a **deliberately wrong prior** — one that underestimates price sensitivity (\(\beta_{\text{prior}} = -0.5\) vs. the true \(-1.2\)). Thompson Sampling must learn the true parameters from pricing experiments while simultaneously earning revenue.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(2026)

# True demand: Q = alpha + beta * P + noise
alpha_true = 50.0
beta_true = -1.2  # true price sensitivity
sigma = 5.0       # demand noise

# Prior on (alpha, beta)
mu_prior = np.array([40.0, -0.5])  # vague prior, underestimates sensitivity
Sigma_prior = np.array([[100.0, 0.0],
                         [0.0, 1.0]])  # wide prior

T = 200
prices_history = np.zeros(T)
quantities_history = np.zeros(T)
revenues_history = np.zeros(T)
mu_history = np.zeros((T + 1, 2))
optimal_price_history = np.zeros(T)

mu_t = mu_prior.copy()
Sigma_t = Sigma_prior.copy()
mu_history[0] = mu_t

for t in range(T):
    # Thompson Sampling: draw (alpha, beta) from posterior
    theta_sample = np.random.multivariate_normal(mu_t, Sigma_t)
    alpha_sample, beta_sample = theta_sample
    
    # Optimal price given sampled parameters: p* = -alpha / (2*beta)
    if beta_sample < -0.01:  # ensure negative slope
        p_optimal = -alpha_sample / (2 * beta_sample)
    else:
        p_optimal = 25.0  # fallback
    
    # Clip to reasonable range
    p_t = np.clip(p_optimal, 5.0, 60.0)
    prices_history[t] = p_t
    
    # Observe demand
    Q_t = alpha_true + beta_true * p_t + np.random.normal(0, sigma)
    Q_t = max(0, Q_t)
    quantities_history[t] = Q_t
    revenues_history[t] = p_t * Q_t
    
    # Bayesian update (Normal-Normal conjugate)
    x_t = np.array([1.0, p_t])
    Sigma_t_inv = np.linalg.inv(Sigma_t)
    Sigma_new_inv = Sigma_t_inv + np.outer(x_t, x_t) / sigma**2
    Sigma_t = np.linalg.inv(Sigma_new_inv)
    mu_t = Sigma_t @ (Sigma_t_inv @ mu_t + x_t * Q_t / sigma**2)
    mu_history[t + 1] = mu_t
    
    # Current best price given posterior mean
    if mu_t[1] < -0.01:
        optimal_price_history[t] = -mu_t[0] / (2 * mu_t[1])
    else:
        optimal_price_history[t] = 25.0

# True optimal price
p_star = -alpha_true / (2 * beta_true)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: posterior evolution of beta
ax = axes[0, 0]
ax.plot(mu_history[:, 1], color='#3498db', linewidth=1.5,
        label=r'Posterior mean of $\beta$')
ax.axhline(beta_true, color='#e74c3c', linewidth=2, linestyle='--',
           label=rf'True $\beta = {beta_true}$')
ax.set_xlabel(r'Round $t$', fontsize=12)
ax.set_ylabel(r'$\beta$ (price sensitivity)', fontsize=12)
ax.set_title(r'Learning $\beta$: Posterior Mean Over Time', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Top-right: price convergence
ax = axes[0, 1]
ax.plot(prices_history, alpha=0.4, color='#2ecc71', linewidth=0.8,
        label='Chosen price')
ax.plot(optimal_price_history, color='#3498db', linewidth=1.5,
        label='Posterior-optimal price')
ax.axhline(p_star, color='#e74c3c', linewidth=2, linestyle='--',
           label=rf'True optimal $p^* = {p_star:.1f}$')
ax.set_xlabel(r'Round $t$', fontsize=12)
ax.set_ylabel(r'Price $p$', fontsize=12)
ax.set_title('Price Convergence to Optimum', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Bottom-left: cumulative revenue vs oracle
ax = axes[1, 0]
oracle_revenue = p_star * (alpha_true + beta_true * p_star)
cumrev = np.cumsum(revenues_history)
oracle_cumrev = np.cumsum(np.full(T, oracle_revenue))
ax.plot(cumrev, color='#3498db', linewidth=1.5, label='Bayesian TS')
ax.plot(oracle_cumrev, color='#e74c3c', linewidth=1.5, linestyle='--',
        label='Oracle')
ax.set_xlabel(r'Round $t$', fontsize=12)
ax.set_ylabel('Cumulative Revenue', fontsize=12)
ax.set_title('Cumulative Revenue: Learner vs Oracle', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Bottom-right: cumulative regret
ax = axes[1, 1]
per_period_regret = oracle_revenue - revenues_history
cumregret = np.cumsum(per_period_regret)
ax.plot(cumregret, color='#e74c3c', linewidth=1.5)
ax.set_xlabel(r'Round $t$', fontsize=12)
ax.set_ylabel(r'Cumulative Regret $R_t$', fontsize=12)
ax.set_title('Regret: Flattening as Learning Completes', fontsize=13)
ax.grid(True, alpha=0.3)

plt.suptitle('Bayesian Dynamic Pricing with Thompson Sampling\n'
             rf'True: $Q = {alpha_true} + ({beta_true})P + \epsilon$, '
             rf'$p^* = {p_star:.1f}$', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('bayesian_pricing.png', dpi=150, bbox_inches='tight')
plt.show()
```

The four panels reveal the full anatomy of Bayesian learning for pricing:

**Top-left (learning \(\beta\))**: the posterior mean of the price sensitivity parameter \(\beta\) starts at the prior value \(-0.5\) and converges toward the true value \(-1.2\) over roughly 50 rounds. The learning is rapid initially — each observation provides a lot of information because the prior is vague — and then slows as the posterior concentrates. This is Bayesian learning in action: the posterior update weights new evidence against prior belief, and when the prior is wrong, the data overwhelms it.

**Top-right (price convergence)**: because the prior underestimates price sensitivity, the algorithm initially overprices — it thinks demand isn't very sensitive to price, so it charges high prices. The green trace (individual prices from Thompson Sampling) is noisy early on because the posterior is wide, producing diverse parameter samples and diverse prices. As the posterior concentrates, the prices converge to the true optimum \(p^* \approx 20.8\). The blue line (posterior-optimal price, computed from the posterior mean) shows the smooth convergence of the algorithm's "best guess."

**Bottom-left (cumulative revenue)**: the Bayesian TS revenue tracks below the oracle initially — the overpricing period costs revenue. But the gap narrows as the algorithm learns, and by the end of 200 rounds, the per-period revenue nearly matches the oracle. The total revenue loss (the gap between the curves) is the cumulative cost of learning — the "tuition" the algorithm pays to discover the demand curve.

**Bottom-right (cumulative regret)**: the regret curve rises steeply in the first 20-30 rounds (the overpricing period) and then **flattens**. The flattening indicates that per-period regret has dropped to near zero — the algorithm has learned enough to price near-optimally. This is the hallmark of a good bandit algorithm: exploration cost is **front-loaded**. You pay the regret cost early, and it pays for itself many times over through better pricing in later rounds. The flattening regret curve is a visual confirmation of the sublinear regret guarantee — \(R_T = O(\sqrt{T})\) for Bayesian TS, where the slope of the cumulative regret curve decays toward zero.

---

## 21. The Complete Stack

Let's tie the entire five-part series together. We started with the most basic question — how should a firm set its price? — and built the answer layer by layer:

**Part 1: The Lerner Index**. Given known demand elasticity \(\varepsilon\), the optimal markup is:

$$
\frac{p - c}{p} = \frac{1}{|\varepsilon|}
$$

This is the foundation. Every pricing decision reduces to knowing your elasticity and applying this formula. Inelastic demand (low \(|\varepsilon|\)) → high markup. Elastic demand (high \(|\varepsilon|\)) → low markup.

**Part 2: Price Discrimination**. A single price leaves money on the table because customers have heterogeneous willingness to pay. First-degree discrimination (charge each customer their WTP), second-degree (versioning, quantity discounts), and third-degree (segment-based pricing) all extract more surplus. The Lerner index applies *per segment* — each customer group has its own elasticity and its own optimal markup.

**Part 3: Competition**. Other firms compress your margins. Bertrand competition drives prices to marginal cost. Differentiation, capacity constraints, and repeated interaction sustain higher equilibria. Game theory tells you how much margin competition destroys and what strategies preserve it. The Buy Box rotation on Amazon is a live mechanism that sustains supra-competitive prices through tacit algorithmic coordination.

**Part 4: Causal Estimation**. The entire stack depends on knowing \(\varepsilon\), but estimating it from data is hard because prices are endogenous — firms set prices in response to demand signals, creating confounding. Instrumental variables, difference-in-differences, regression discontinuity, and double/debiased ML solve this identification problem and give you unbiased elasticity estimates.

**Part 5: Bandits and Dynamic Pricing**. Even with good causal estimates, demand changes over time and differs across contexts. Multi-armed bandits (UCB, Thompson Sampling) learn the optimal price online while minimizing regret. Contextual bandits personalize prices based on customer and market features. MDPs handle inventory constraints and inter-temporal dependencies. At scale, this becomes the automated pricing engine running at Amazon (2.5 million price changes/day), Uber (real-time surge across thousands of zones), and airlines (revenue management across millions of itineraries).

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 400" style="max-width:700px; width:100%; height:auto;">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#d4d4d4"/>
    </marker>
  </defs>
  <rect width="700" height="400" fill="#1a1a2e" rx="8"/>
  <!-- Title -->
  <text x="350" y="35" text-anchor="middle" fill="#d4d4d4" font-size="16" font-weight="bold">The Complete Pricing Stack</text>
  <!-- Boxes -->
  <rect x="50" y="60" width="180" height="50" rx="6" fill="#2d3436" stroke="#636e72" stroke-width="1.5"/>
  <text x="140" y="82" text-anchor="middle" fill="#d4d4d4" font-size="11" font-weight="bold">Part 1: Lerner Index</text>
  <text x="140" y="100" text-anchor="middle" fill="#b2bec3" font-size="10">p* = c / (1 + 1/|ε|)</text>

  <rect x="260" y="60" width="180" height="50" rx="6" fill="#2d3436" stroke="#636e72" stroke-width="1.5"/>
  <text x="350" y="82" text-anchor="middle" fill="#d4d4d4" font-size="11" font-weight="bold">Part 2: Discrimination</text>
  <text x="350" y="100" text-anchor="middle" fill="#b2bec3" font-size="10">Segment → per-group ε → markup</text>

  <rect x="470" y="60" width="180" height="50" rx="6" fill="#2d3436" stroke="#636e72" stroke-width="1.5"/>
  <text x="560" y="82" text-anchor="middle" fill="#d4d4d4" font-size="11" font-weight="bold">Part 3: Competition</text>
  <text x="560" y="100" text-anchor="middle" fill="#b2bec3" font-size="10">Game theory → margin pressure</text>

  <!-- Row 2 -->
  <rect x="155" y="160" width="180" height="50" rx="6" fill="#2d3436" stroke="#636e72" stroke-width="1.5"/>
  <text x="245" y="182" text-anchor="middle" fill="#d4d4d4" font-size="11" font-weight="bold">Part 4: Causal Estimation</text>
  <text x="245" y="200" text-anchor="middle" fill="#b2bec3" font-size="10">IV / DML → unbiased ε</text>

  <rect x="365" y="160" width="180" height="50" rx="6" fill="#2d3436" stroke="#636e72" stroke-width="1.5"/>
  <text x="455" y="182" text-anchor="middle" fill="#d4d4d4" font-size="11" font-weight="bold">Part 5: Bandits & RL</text>
  <text x="455" y="200" text-anchor="middle" fill="#b2bec3" font-size="10">Learn ε online + optimize</text>

  <!-- Bottom: unified system -->
  <rect x="150" y="280" width="400" height="65" rx="8" fill="#0a3d62" stroke="#3498db" stroke-width="2"/>
  <text x="350" y="305" text-anchor="middle" fill="#d4d4d4" font-size="13" font-weight="bold">Automated Pricing Engine</text>
  <text x="350" y="325" text-anchor="middle" fill="#b2bec3" font-size="10">Amazon: 2.5M changes/day | Uber: real-time surge | Airlines: revenue mgmt</text>

  <!-- Arrows -->
  <line x1="140" y1="110" x2="220" y2="160" stroke="#636e72" stroke-width="1.5" marker-end="url(#arrowhead)"/>
  <line x1="350" y1="110" x2="280" y2="160" stroke="#636e72" stroke-width="1.5" marker-end="url(#arrowhead)"/>
  <line x1="350" y1="110" x2="430" y2="160" stroke="#636e72" stroke-width="1.5" marker-end="url(#arrowhead)"/>
  <line x1="560" y1="110" x2="480" y2="160" stroke="#636e72" stroke-width="1.5" marker-end="url(#arrowhead)"/>
  <line x1="245" y1="210" x2="300" y2="280" stroke="#636e72" stroke-width="1.5" marker-end="url(#arrowhead)"/>
  <line x1="455" y1="210" x2="400" y2="280" stroke="#636e72" stroke-width="1.5" marker-end="url(#arrowhead)"/>
</svg>

This is what modern algorithmic pricing looks like: economic theory (Parts 1-3) provides the structural framework, causal inference (Part 4) grounds it in data, and online learning (Part 5) makes it adaptive and autonomous.

**The frontier**: Large language models are beginning to enter pricing — understanding product descriptions, customer reviews, and competitive positioning in natural language to inform elasticity estimates. Reinforcement learning from human feedback (RLHF) is being explored for pricing fairness — training pricing agents to satisfy both revenue objectives and equity constraints. And the regulatory challenge of algorithmic collusion — the Buy Box phenomenon we discussed — is one of the most active areas in antitrust economics.

The toolkit is complete. The Lerner index tells you *what* to charge. Causal estimation tells you *how to measure* what you need. Bandits tell you *how to learn* while earning. And the MDP framework tells you *how to plan* when today's decisions shape tomorrow's state. Every automated pricing system in the world is some combination of these four ideas, operating at the scale that only algorithms can achieve.

*This concludes the 5-part pricing strategy series. [Back to Part 1: Demand, Elasticity & Markup](/2026/04/13/demand-elasticity-monopolist-markup.html)*
