---
layout: post
title: "Time Series Foundations for Forecasting: Stationarity, Autocorrelation, and the Wold Decomposition"
date: 2026-04-18
category: math
---

*This is Part 1 of a 5-part series on time series forecasting. **Part 1: Foundations** | [Part 2: ARIMA & Box-Jenkins Forecasting](/2026/04/19/arima-box-jenkins-forecasting.html) | [Part 3: Exponential Smoothing, ETS & Theta](/2026/04/20/exponential-smoothing-ets-theta.html) | [Part 4: State-Space Models & Kalman Filtering](/2026/04/21/state-space-kalman-filtering.html) | [Part 5: Modern Forecasting — GARCH, Gradient Boosting, N-BEATS & Hierarchies](/2026/04/22/modern-forecasting-garch-gbm-nbeats-hierarchical.html)*

Forecasting is the point. A demand-planning team at a retailer does not care whether the sales series is "covariance stationary" — they care whether the model they ship on Monday will predict Friday's SKU-level demand closely enough to avoid a $10M stockout. A quant fund does not want to "characterize the spectral density of returns" — they want a conditional distribution of tomorrow's price that beats the implied quote. A cloud operator does not want to "decompose the latency series" — they want a 24-hour forecast accurate enough to autoscale without paging the on-call. Every concept in this series exists because it buys you a better forecast.

But forecasts are only as reliable as the assumptions behind them, and the most important assumption in every time series model is some form of stationarity — the idea that the structure generating past observations is the same structure that will generate the next one. If that fails, your point forecast is biased, your prediction interval is miscalibrated, your backtest lies to you, and your model silently overfits to a regime that is already over. So before we fit a single forecasting model (Part 2 onward), we have to build the vocabulary for describing and testing that structure.

This post does that. We define the stochastic process; define weak and strict stationarity; derive the autocovariance, autocorrelation, and partial autocorrelation functions — the three objects every forecaster reads before fitting anything; introduce white noise (the target shape of forecast residuals) and the random walk (the benchmark every forecaster must beat); state and sketch the **Wold decomposition**, which says every stationary process is a filtered white noise and is therefore the formal reason linear forecasting models exist at all; develop ergodicity, which is why we can estimate forecast-relevant quantities from one sample path; describe the trend/seasonality preprocessing that makes real series suitable for model fitting; cover the diagnostic tests (Ljung-Box, ADF, KPSS) that gate the transition from data to model; close with a section on **industry best practices** — what forecasters actually do with this machinery in production.

---

## Table of Contents

1. [What Is a Time Series?](#1-what-is-a-time-series)
2. [Stationarity: Strict and Weak](#2-stationarity-strict-and-weak)
3. [Autocovariance and Autocorrelation](#3-autocovariance-and-autocorrelation)
4. [White Noise and the Random Walk](#4-white-noise-and-the-random-walk)
5. [The Partial Autocorrelation Function](#5-the-partial-autocorrelation-function)
6. [Linear Processes and the Wold Decomposition](#6-linear-processes-and-the-wold-decomposition)
7. [Ergodicity: When Time Averages Equal Ensemble Averages](#7-ergodicity-when-time-averages-equal-ensemble-averages)
8. [Trends, Seasonality, and Differencing](#8-trends-seasonality-and-differencing)
9. [Testing for White Noise and Unit Roots](#9-testing-for-white-noise-and-unit-roots)
10. [Python Simulations](#10-python-simulations)
11. [Industry Best Practices](#11-industry-best-practices)

---

## 1. What Is a Time Series?

Formally, a **time series** is a collection of random variables indexed by time:

$$
\{X_t : t \in \mathcal{T}\}
$$

all defined on a common probability space \((\Omega, \mathcal{F}, \mathbb{P})\). The index set \(\mathcal{T}\) is typically \(\mathbb{Z}\) (discrete time) or \(\mathbb{R}\) (continuous time). We will focus on discrete time, so \(t \in \mathbb{Z}\) or \(t \in \{1, 2, \ldots, T\}\).

Two things are worth pausing on. First, \(X_t\) is a *random variable*, not a number. When you look at a time plot — say, daily closing prices for AAPL over 2020–2025 — you are looking at *one realization* \(\{x_t(\omega^*)\}\) of the underlying process, for a single outcome \(\omega^* \in \Omega\). Different "outcomes" \(\omega\) produce different possible histories. In the real world we only get to see one history, and the central methodological problem of time series is how to infer properties of the distribution \(\mathbb{P}\) from a single path.

Second, the process is really a function \(X : \mathbb{Z} \times \Omega \to \mathbb{R}\). For fixed \(t\), \(X_t(\cdot)\) is a random variable. For fixed \(\omega\), \(X_\cdot(\omega)\) is a deterministic sequence called a **sample path** or **realization**. The distinction matters because statistical inference is almost always about averages over \(\omega\) (expectations), but we only have data that averages over \(t\). The bridge between these two is *ergodicity*, which we treat in Section 7.

### Finite-Dimensional Distributions

A stochastic process is specified by its **finite-dimensional distributions**: for every finite collection of times \(t_1 < t_2 < \ldots < t_n\), the joint distribution

$$
F_{t_1, \ldots, t_n}(x_1, \ldots, x_n) = \mathbb{P}(X_{t_1} \le x_1, \ldots, X_{t_n} \le x_n).
$$

**Kolmogorov's extension theorem** says that if a family of finite-dimensional distributions satisfies two consistency conditions — permutation invariance and a marginalization condition — then there exists a probability space and a stochastic process with exactly those distributions. In practice we never work with the full joint distribution; we work with its first two moments, because that is enough for linear forecasting.

### Why One Realization Is Enough (Sometimes)

Consider the question: what is \(\mathbb{E}[X_t]\)? In principle you would average over many independent realizations: \(\hat{\mu} = (1/N) \sum_{i=1}^N X_t^{(i)}\). But we only have one realization. The natural alternative is the **time average** \(\bar{X}_T = (1/T) \sum_{t=1}^T X_t\). These two averages estimate different things in general — the first estimates \(\mathbb{E}[X_t]\), the second estimates something like an orbit average along the realization. They coincide *only* when the process is stationary and ergodic, as Birkhoff's theorem (1931) makes precise. Every applied piece of time series inference implicitly uses this equivalence.

---

## 2. Stationarity: Strict and Weak

Stationarity is the time-series version of i.i.d. — it is the condition that makes statistical inference possible from a single sample path. There are two definitions of stationarity, and the weaker one is what we use in practice.

### Strict Stationarity

A process \(\{X_t\}\) is **strictly stationary** if for every \(n \ge 1\), every \(t_1 < \ldots < t_n\), and every \(h \in \mathbb{Z}\),

$$
(X_{t_1}, \ldots, X_{t_n}) \stackrel{d}{=} (X_{t_1 + h}, \ldots, X_{t_n + h}).
$$

Shifting time does not change the joint distribution. This is a very strong condition — it applies to the entire distribution, not just its moments.

### Weak (Covariance) Stationarity

A process is **weakly stationary** or **covariance stationary** if:

1. \(\mathbb{E}[X_t^2] < \infty\) for all \(t\) (second moments exist),
2. \(\mathbb{E}[X_t] = \mu\) for all \(t\) (constant mean),
3. \(\mathrm{Cov}(X_t, X_{t+h}) = \gamma(h)\) depends only on \(h\), not on \(t\).

Weak stationarity is what classical time series assumes. Note two things:

- Strict stationarity plus finite second moments implies weak stationarity.
- Weak stationarity does *not* imply strict stationarity in general — you might have a process with constant mean and autocovariance depending only on the lag, but with higher-moment dependence that shifts over time (e.g., ARCH/GARCH models).
- For Gaussian processes, the two notions coincide, because a Gaussian distribution is completely determined by its first two moments.

### Why Stationarity Matters

Consider what happens without stationarity. Suppose \(\mathbb{E}[X_t] = \mu_t\) changes over time. Then any estimator we build by averaging — say the sample mean — is estimating some weighted average of the \(\mu_t\)s, which has no clean interpretation. Forecasting becomes impossible in principle: the future may be drawn from a different distribution than anything we have seen. Stationarity is the condition under which the future resembles the past in a statistical sense, and therefore the past can inform the future.

In practice, most real-world series are *not* stationary as observed. Stock prices drift; GDP grows; temperature has seasonal cycles. The first step of classical time series analysis is almost always to *transform* the observed series into something approximately stationary — via differencing, log transforms, or explicit trend removal. The entire apparatus of ARIMA (Part 2) is organized around this preprocessing step.

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg">
  <rect width="700" height="280" fill="#1a1a1a"/>
  <text x="350" y="20" fill="#e8e8e8" font-size="15" text-anchor="middle" font-family="Georgia, serif" font-weight="bold">Stationary vs. Non-Stationary Realizations</text>
  <!-- Stationary panel -->
  <rect x="40" y="40" width="300" height="200" fill="none" stroke="#444" stroke-width="1"/>
  <text x="190" y="60" fill="#6db3f2" font-size="13" text-anchor="middle" font-family="Georgia, serif">Stationary (constant mean, bounded variance)</text>
  <polyline points="50,140 70,130 90,150 110,135 130,155 150,140 170,125 190,145 210,160 230,140 250,130 270,150 290,145 310,135 330,150" fill="none" stroke="#6db3f2" stroke-width="1.5"/>
  <line x1="50" y1="140" x2="330" y2="140" stroke="#888" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="335" y="144" fill="#888" font-size="11" font-family="Georgia, serif">μ</text>
  <!-- Non-stationary panel -->
  <rect x="370" y="40" width="300" height="200" fill="none" stroke="#444" stroke-width="1"/>
  <text x="520" y="60" fill="#f2a5a5" font-size="13" text-anchor="middle" font-family="Georgia, serif">Random walk (variance grows with t)</text>
  <polyline points="380,150 400,140 420,160 440,145 460,170 480,160 500,175 520,180 540,195 560,185 580,210 600,195 620,215 640,225 660,220" fill="none" stroke="#f2a5a5" stroke-width="1.5"/>
  <line x1="380" y1="150" x2="660" y2="150" stroke="#888" stroke-width="1" stroke-dasharray="3,3"/>
</svg>

---

## 3. Autocovariance and Autocorrelation

For a weakly stationary process, the **autocovariance function** at lag \(h\) is

$$
\gamma(h) = \mathrm{Cov}(X_t, X_{t+h}) = \mathbb{E}[(X_t - \mu)(X_{t+h} - \mu)].
$$

It quantifies the linear relationship between the process at two times separated by \(h\) units. By stationarity this depends only on \(h\), not on the absolute time \(t\). The autocovariance has three immediate properties:

- **Symmetry**: \(\gamma(h) = \gamma(-h)\), because \(\mathrm{Cov}(X_t, X_{t+h}) = \mathrm{Cov}(X_{t+h}, X_t)\).
- **Bound**: \(|\gamma(h)| \le \gamma(0) = \mathrm{Var}(X_t)\), by Cauchy-Schwarz.
- **Positive semi-definiteness**: for any \(n\), any times \(t_1, \ldots, t_n\), and any real \(a_1, \ldots, a_n\),

$$
\sum_{i,j=1}^n a_i a_j \gamma(t_i - t_j) \ge 0.
$$

This last property is the functional analogue of a covariance matrix being positive semi-definite. It will play a central role in Part 3 when we introduce spectral density via Bochner's theorem.

### The Autocorrelation Function (ACF)

The **autocorrelation function** simply normalizes the autocovariance by the variance:

$$
\rho(h) = \frac{\gamma(h)}{\gamma(0)}.
$$

So \(\rho(0) = 1\) and \(|\rho(h)| \le 1\) for all \(h\). The ACF at lag \(h\) is the correlation coefficient between \(X_t\) and \(X_{t+h}\). It is scale-free, so two series with the same autocorrelation structure but different variances have the same ACF.

### Sample Autocovariance and ACF

Given a realization \(x_1, \ldots, x_T\), the natural estimators are

$$
\hat{\gamma}(h) = \frac{1}{T} \sum_{t=1}^{T-h} (x_t - \bar{x})(x_{t+h} - \bar{x}), \qquad \hat{\rho}(h) = \frac{\hat{\gamma}(h)}{\hat{\gamma}(0)}.
$$

A few subtleties. Notice we divide by \(T\), not \(T - h\). Dividing by \(T\) makes the estimator biased but guarantees that \(\hat{\gamma}\) is a valid positive semi-definite function — critical for downstream uses like computing spectral estimates. Dividing by \(T - h\) gives an unbiased estimator but it can fail to be positive semi-definite. Most software (including `statsmodels.tsa`) uses the \(T\)-normalization by default.

For large \(T\) and under mild conditions (linear process, finite fourth moments), Bartlett's formula gives the asymptotic variance of \(\hat{\rho}(h)\):

$$
\mathrm{Var}(\hat{\rho}(h)) \approx \frac{1}{T} \sum_{k=-\infty}^{\infty} \left[\rho(k)^2 + \rho(k-h)\rho(k+h) - 4\rho(h)\rho(k)\rho(k-h) + 2\rho(h)^2 \rho(k)^2\right].
$$

For a white noise process (\(\rho(k) = 0\) for \(k \ne 0\)), this collapses to \(1/T\), giving the familiar \(\pm 1.96/\sqrt{T}\) confidence bands on ACF plots. When you see blue dashed lines on a correlogram, that's what they are.

---

## 4. White Noise and the Random Walk

Two processes play the role of the hydrogen atom in time series: white noise and the random walk. Everything else is built from them.

### White Noise

A process \(\{\varepsilon_t\}\) is **white noise** if

$$
\mathbb{E}[\varepsilon_t] = 0, \quad \mathbb{E}[\varepsilon_t^2] = \sigma^2, \quad \mathbb{E}[\varepsilon_t \varepsilon_s] = 0 \text{ for } t \ne s.
$$

We write \(\varepsilon_t \sim \mathrm{WN}(0, \sigma^2)\). Zero mean, constant variance, uncorrelated across time. Note: white noise is *not* required to be i.i.d. or Gaussian. A GARCH residual sequence is white noise but has dependent squares.

- **Gaussian white noise**: \(\varepsilon_t \sim \mathcal{N}(0, \sigma^2)\) i.i.d. This is what most simulations use.
- **i.i.d. white noise**: same distribution, independent — stronger than WN.
- **Strong white noise**: usually synonymous with i.i.d. WN.

The autocovariance is \(\gamma(0) = \sigma^2\), \(\gamma(h) = 0\) for \(h \ne 0\). In spectral terms (Part 3), its spectral density is constant: flat across all frequencies — hence the name "white," by analogy with white light.

### The Random Walk

The **random walk** is the cumulative sum of white noise:

$$
Y_t = Y_{t-1} + \varepsilon_t = Y_0 + \sum_{s=1}^t \varepsilon_s.
$$

With \(Y_0 = 0\) for concreteness, we get

$$
\mathbb{E}[Y_t] = 0, \qquad \mathrm{Var}(Y_t) = t\sigma^2, \qquad \mathrm{Cov}(Y_t, Y_{t+h}) = t\sigma^2 \text{ for } h \ge 0.
$$

The variance grows linearly with time — the process is *not* stationary. Yet it is central to finance (log-returns), physics (diffusion), and ecology. The appropriate way to make it stationary is to take first differences: \(\Delta Y_t = Y_t - Y_{t-1} = \varepsilon_t\), which is white noise. This motivates the "I" (for "integrated") in ARIMA: an I(1) process becomes stationary after one differencing.

### Random Walk with Drift

Add a constant term:

$$
Y_t = \delta + Y_{t-1} + \varepsilon_t.
$$

By recursion, \(Y_t = Y_0 + \delta t + \sum \varepsilon_s\). The drift gives the series a deterministic linear trend, on top of the stochastic accumulation. Differencing still makes it stationary: \(\Delta Y_t = \delta + \varepsilon_t\).

### The ACF Signature

The ACF is one of the most useful diagnostic tools. Different processes have very different ACFs:

- **White noise**: ACF is zero at all lags except lag 0.
- **Random walk**: sample ACF decays very slowly and linearly — a classic fingerprint of non-stationarity.
- **AR(1) with \(\phi = 0.8\)**: ACF decays geometrically as \(0.8^h\).
- **MA(1)**: ACF is nonzero only at lag 1.

We build up this catalogue in Part 2.

---

## 5. The Partial Autocorrelation Function

The ACF at lag \(h\) measures total correlation between \(X_t\) and \(X_{t+h}\) — but some of that correlation is indirect, passing through the intermediate lags. The **partial autocorrelation function (PACF)** at lag \(h\) measures the correlation *after removing the linear effect* of the intermediate variables \(X_{t+1}, \ldots, X_{t+h-1}\).

Formally, the PACF at lag \(h\), denoted \(\phi_{hh}\), is the coefficient of \(X_{t-h}\) in the best linear predictor of \(X_t\) using \(X_{t-1}, \ldots, X_{t-h}\):

$$
X_t = \phi_{h1} X_{t-1} + \phi_{h2} X_{t-2} + \ldots + \phi_{hh} X_{t-h} + u_t.
$$

This can be computed recursively via the **Yule-Walker equations** or the **Durbin-Levinson algorithm**. The key fact is:

- For an **AR(p)** process, \(\phi_{hh} = 0\) for \(h > p\). The PACF "cuts off" at lag \(p\).
- For an **MA(q)** process, the PACF decays geometrically without cutting off.

This complementarity is the backbone of the Box-Jenkins method (Part 2): look at ACF and PACF together to identify model orders.

### Computing the PACF: Durbin-Levinson

Given autocorrelations \(\rho(1), \rho(2), \ldots\), the Durbin-Levinson recursion computes \(\phi_{hh}\) in \(O(h^2)\) time:

$$
\phi_{hh} = \frac{\rho(h) - \sum_{j=1}^{h-1} \phi_{h-1,j} \rho(h-j)}{1 - \sum_{j=1}^{h-1} \phi_{h-1,j} \rho(j)},
$$

$$
\phi_{h,j} = \phi_{h-1,j} - \phi_{hh} \phi_{h-1,h-j}, \quad j = 1, \ldots, h-1.
$$

You start from \(\phi_{11} = \rho(1)\) and iterate. This is how `statsmodels` computes the sample PACF internally (or via OLS on lagged regressors — equivalent asymptotically).

---

## 6. Linear Processes and the Wold Decomposition

We now arrive at one of the most beautiful theorems in time series analysis. Informally, every (mean-zero, weakly stationary, purely non-deterministic) process can be written as an infinite moving average of white noise. This is the **Wold decomposition** (1938), and it is the theoretical foundation for everything in Part 2.

### Linear Processes

A process \(\{X_t\}\) is called **linear** or a **moving average representation** if

$$
X_t = \sum_{j=0}^\infty \psi_j \varepsilon_{t-j}, \qquad \sum_{j=0}^\infty \psi_j^2 < \infty,
$$

where \(\varepsilon_t \sim \mathrm{WN}(0, \sigma^2)\). The convergence of the series in mean square is guaranteed by the square-summability of \(\{\psi_j\}\). With \(\psi_0 = 1\), the autocovariance can be computed directly:

$$
\gamma(h) = \sigma^2 \sum_{j=0}^\infty \psi_j \psi_{j+h}.
$$

This formula will be used constantly.

### The Wold Decomposition Theorem

**Theorem (Wold, 1938).** Any zero-mean weakly stationary process \(\{X_t\}\) can be written uniquely as

$$
X_t = \sum_{j=0}^\infty \psi_j \varepsilon_{t-j} + V_t,
$$

where:

1. \(\psi_0 = 1\) and \(\sum \psi_j^2 < \infty\);
2. \(\{\varepsilon_t\}\) is white noise with variance \(\sigma^2\);
3. \(\varepsilon_t\) is the one-step-ahead linear forecast error: \(\varepsilon_t = X_t - P_{t-1} X_t\), where \(P_{t-1}\) is the projection onto the closed linear span of past \(X_s\)s;
4. \(V_t\) is a **deterministic** component — predictable from its own infinite past with zero error — and is uncorrelated with \(\varepsilon_s\) for all \(s\).

### Sketch of Proof

Let \(\mathcal{H}_t\) be the closed linear span of \(\{X_s : s \le t\}\) in \(L^2\). Define the **innovation**

$$
\varepsilon_t = X_t - P_{\mathcal{H}_{t-1}} X_t,
$$

where \(P_{\mathcal{H}_{t-1}}\) is orthogonal projection. By construction \(\varepsilon_t \perp \mathcal{H}_{t-1}\), and by stationarity \(\mathbb{E}[\varepsilon_t^2] = \sigma^2\) is constant. The \(\varepsilon_t\) are mutually orthogonal (innovations at different times live in different orthogonal subspaces), hence white noise.

Now project \(X_t\) onto \(\overline{\mathrm{span}}\{\varepsilon_s : s \le t\}\):

$$
P_{\mathcal{H}^\varepsilon_t} X_t = \sum_{j=0}^\infty \psi_j \varepsilon_{t-j}, \qquad \psi_j = \frac{\langle X_t, \varepsilon_{t-j}\rangle}{\sigma^2}.
$$

The residual \(V_t = X_t - P_{\mathcal{H}^\varepsilon_t} X_t\) lies in \(\mathcal{H}_t\) but is orthogonal to every \(\varepsilon_s\), so it is in \(\bigcap_t \mathcal{H}_t\), the **remote past**. Any element of the remote past is perfectly predictable from its own past — hence deterministic. □

### What Wold Really Says — and Why Forecasters Care

Three interpretations, all pointed at forecasting:

1. **Every stationary process is AR(\(\infty\))/MA(\(\infty\))**: once you remove a possibly deterministic component, the rest is a moving average of white noise innovations. ARMA models (Part 2) are finite-parameter approximations to this infinite representation — and the quality of that approximation is exactly the forecast gain over the naive mean forecast.
2. **Innovations are the forecast residuals, by construction**: \(\varepsilon_t = X_t - P_{\mathcal{H}_{t-1}} X_t\). This is not an assumption imposed on the data; it is what you *extract* by optimal one-step-ahead linear projection. "White residuals" is not an aesthetic preference, it is the guarantee that no further linear signal remains. A forecaster whose residuals still autocorrelate is, by Wold, leaving predictable variance on the table.
3. **Linear forecasting is optimal for Gaussian processes**: the Wold innovations coincide with the best *nonlinear* forecast residuals when the joint distribution is Gaussian. Nonlinear models (Part 5) can only help when the joint distribution is non-Gaussian or conditional variance is time-varying.

The deterministic component \(V_t\) is usually ignored in practice. In macroeconomics, where very long cycles or demographic trends exist, it can matter. In operational forecasting — demand, traffic, latency — \(V_t = 0\) is a safe default.

### The Lag Operator and Polynomial Form

Define the **lag operator** \(L\) by \(LX_t = X_{t-1}\), so \(L^j X_t = X_{t-j}\). A linear process can be written

$$
X_t = \psi(L) \varepsilon_t, \qquad \psi(L) = \sum_{j=0}^\infty \psi_j L^j.
$$

This notation lets us manipulate time series algebraically. ARMA models are rational lag polynomials: \(\phi(L) X_t = \theta(L) \varepsilon_t\), where \(\phi\) and \(\theta\) are finite-order polynomials. Inverting \(\phi(L)\), when possible, yields the MA(\(\infty\)) representation directly:

$$
X_t = \frac{\theta(L)}{\phi(L)} \varepsilon_t.
$$

We will exploit this machinery heavily in Part 2.

---

## 7. Ergodicity: When Time Averages Equal Ensemble Averages

Stationarity says the *distribution* does not change over time. But we still only have one realization. Can we estimate \(\mathbb{E}[X_t] = \mu\) from the time average \(\bar{X}_T = (1/T)\sum X_t\)? The answer depends on **ergodicity**.

### Definition

A weakly stationary process is **mean ergodic** if

$$
\bar{X}_T \xrightarrow{\mathbb{P}} \mu \quad \text{as } T \to \infty.
$$

A sufficient (and intuitive) condition is that \(\sum_{h=-\infty}^\infty |\gamma(h)| < \infty\) — the autocovariances are summable. Then

$$
\mathrm{Var}(\bar{X}_T) = \frac{1}{T^2} \sum_{s,t=1}^T \gamma(s-t) \approx \frac{1}{T} \sum_{h=-\infty}^\infty \gamma(h) \to 0.
$$

The intuition: if memory decays fast enough, far-apart observations are nearly independent, so the time average satisfies something like a law of large numbers. A process that is *not* mean ergodic: let \(X_t = Z\) for all \(t\), where \(Z \sim \mathcal{N}(0, 1)\). This is stationary — every \(X_t\) has the same distribution, and autocovariances depend only on the lag — but the time average equals \(Z\), not \(0 = \mathbb{E}[Z]\).

### Ergodicity for Second Moments

For estimating autocovariances we need a stronger property: **covariance ergodicity**. It requires that \(\mathbb{E}[X_t X_{t+h}]\) can be consistently estimated by time averages. For linear processes driven by i.i.d. white noise with finite fourth moments, this holds automatically under summability of \(\gamma\). For nonlinear processes (GARCH, stochastic volatility) the conditions are more delicate.

### Birkhoff's Ergodic Theorem

The deepest statement is **Birkhoff's ergodic theorem** (1931): for any strictly stationary, ergodic process and any measurable \(f\) with \(\mathbb{E}|f(X_t)| < \infty\),

$$
\frac{1}{T} \sum_{t=1}^T f(X_t) \xrightarrow{\text{a.s.}} \mathbb{E}[f(X_0)].
$$

This is *the* theorem that justifies replacing ensemble averages with time averages. Every applied inference procedure — from estimating a mean, to an ACF, to a density — leans on it. Ergodicity is to time series what i.i.d. is to cross-sectional statistics.

### Practical Consequence

You almost never test ergodicity directly. You test stationarity (via unit-root tests), and if the process is stationary and mixing (which follows from mild regularity conditions), it is ergodic. The practical warning: if your series has structural breaks or regime shifts, ergodicity fails, and your estimates are estimating a confused mixture of regimes rather than a well-defined population quantity.

---

## 8. Trends, Seasonality, and Differencing

Real-world series typically violate stationarity in two predictable ways: they trend, and they repeat. A classical decomposition writes

$$
X_t = T_t + S_t + R_t,
$$

where \(T_t\) is a slowly-varying **trend**, \(S_t\) is a **seasonal** component with period \(s\) (e.g., \(s = 12\) for monthly data with annual seasonality), and \(R_t\) is a stationary **remainder**. The statistical task is to estimate and remove \(T_t\) and \(S_t\), leaving a stationary residual to model.

### Removing the Trend

Two approaches:

1. **Detrending**: fit a deterministic trend (linear, polynomial, or smoother like a Hodrick-Prescott or LOESS) and subtract. This assumes the trend is a smooth deterministic function.
2. **Differencing**: take \(\Delta X_t = X_t - X_{t-1}\). This removes a linear trend exactly; applied \(d\) times, it removes a polynomial trend of degree \(d\).

The fundamental question: is the trend *deterministic* (best removed by detrending) or *stochastic* (best removed by differencing)? Mistaking one for the other produces biased inference. A stochastic trend (unit root) detrended linearly produces spurious persistence in residuals; a deterministic trend differenced produces a non-invertible MA component with unusual error dynamics.

The **augmented Dickey-Fuller test** and the **KPSS test** (below) are designed to discriminate these cases.

### Seasonal Differencing

For monthly data with annual cycle, apply

$$
\Delta_{12} X_t = X_t - X_{t-12}.
$$

This removes a constant seasonal pattern. Combined with first differencing, the common transformation is \(\Delta \Delta_{12} X_t = (X_t - X_{t-1}) - (X_{t-12} - X_{t-13})\), which is the backbone of SARIMA (Part 2).

### STL Decomposition

The modern standard is **STL — Seasonal and Trend decomposition using Loess** (Cleveland et al., 1990). It iteratively smooths the seasonal and trend components using locally-weighted regression, handles non-constant seasonal amplitude, and is robust to outliers. We show it in the Python section.

<svg viewBox="0 0 700 340" xmlns="http://www.w3.org/2000/svg">
  <rect width="700" height="340" fill="#1a1a1a"/>
  <text x="350" y="20" fill="#e8e8e8" font-size="15" text-anchor="middle" font-family="Georgia, serif" font-weight="bold">Classical Additive Decomposition: X = T + S + R</text>
  <!-- Original -->
  <text x="50" y="60" fill="#e8e8e8" font-size="12" font-family="Georgia, serif">X_t</text>
  <polyline points="80,70 110,55 140,75 170,45 200,80 230,48 260,85 290,50 320,90 350,55 380,95 410,60 440,100 470,65 500,105 530,70 560,108 590,75 620,112" fill="none" stroke="#6db3f2" stroke-width="1.5"/>
  <!-- Trend -->
  <text x="50" y="130" fill="#e8e8e8" font-size="12" font-family="Georgia, serif">T_t</text>
  <line x1="80" y1="138" x2="620" y2="170" stroke="#a4d08a" stroke-width="2"/>
  <!-- Seasonal -->
  <text x="50" y="210" fill="#e8e8e8" font-size="12" font-family="Georgia, serif">S_t</text>
  <polyline points="80,210 110,195 140,220 170,195 200,220 230,195 260,220 290,195 320,220 350,195 380,220 410,195 440,220 470,195 500,220 530,195 560,220 590,195 620,220" fill="none" stroke="#f2c894" stroke-width="1.5"/>
  <!-- Residual -->
  <text x="50" y="290" fill="#e8e8e8" font-size="12" font-family="Georgia, serif">R_t</text>
  <polyline points="80,285 110,288 140,282 170,292 200,285 230,295 260,285 290,280 320,290 350,285 380,292 410,282 440,293 470,286 500,295 530,282 560,290 590,287 620,289" fill="none" stroke="#f2a5a5" stroke-width="1.5"/>
  <line x1="80" y1="285" x2="620" y2="285" stroke="#555" stroke-width="0.8" stroke-dasharray="3,3"/>
</svg>

---

## 9. Testing for White Noise and Unit Roots

Three tests together cover the diagnostic workflow.

### Ljung-Box Test

Tests the null that the first \(h\) autocorrelations are jointly zero — i.e., the series is white noise through lag \(h\). The statistic is

$$
Q_h = T(T+2) \sum_{k=1}^{h} \frac{\hat{\rho}(k)^2}{T - k},
$$

which is asymptotically \(\chi^2_{h}\) under white noise (or \(\chi^2_{h - p - q}\) when applied to residuals of an ARMA(\(p,q\)) fit). You use it on residuals after model fitting: if the model has captured the dynamics, residuals should be white, and \(Q_h\) should not reject.

### Augmented Dickey-Fuller (ADF)

Tests the null of a unit root (non-stationary) against stationary alternative. The regression

$$
\Delta X_t = \alpha + \beta t + \rho X_{t-1} + \sum_{i=1}^p \gamma_i \Delta X_{t-i} + \varepsilon_t
$$

has the null \(\rho = 0\) (unit root) versus \(\rho < 0\) (stationary). Critical values are non-standard (Dickey-Fuller distribution) because the OLS \(t\)-statistic does not have a \(t\) distribution under the null. The test has notoriously low power when \(\rho\) is close to but below zero — near-unit-root processes can look indistinguishable from true random walks.

### KPSS

Reverses the nulls: null is stationary, alternative is unit root. Complements ADF because they test different hypotheses and often disagree in the borderline zone where power of both tests is low. The common strategy is to use both: ADF rejects unit root + KPSS fails to reject stationary ⇒ confident stationary. Neither rejects ⇒ indeterminate. Both reject ⇒ the series may be neither.

---

## 10. Python Simulations

We will simulate white noise, a random walk, and an AR(1) process; estimate autocorrelations; and test for stationarity. Code uses `numpy`, `matplotlib`, `statsmodels`, and the rendering conventions from the style guide (LaTeX in labels, `plt.subplots`).

### Simulating the Three Processes

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

rng = np.random.default_rng(42)
T = 1000
sigma = 1.0

# White noise
wn = rng.normal(0, sigma, T)

# Random walk
rw = np.cumsum(wn)

# AR(1) with phi = 0.8
phi = 0.8
ar1 = np.zeros(T)
for t in range(1, T):
    ar1[t] = phi * ar1[t-1] + rng.normal(0, sigma)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(wn, color='#6db3f2', lw=0.8)
axes[0].set_title(r'White noise $\varepsilon_t \sim \mathcal{N}(0, 1)$')
axes[0].set_ylabel(r'$X_t$')

axes[1].plot(rw, color='#f2a5a5', lw=0.8)
axes[1].set_title(r'Random walk $Y_t = Y_{t-1} + \varepsilon_t$')
axes[1].set_ylabel(r'$Y_t$')

axes[2].plot(ar1, color='#a4d08a', lw=0.8)
axes[2].set_title(r'AR(1) with $\phi = 0.8$')
axes[2].set_ylabel(r'$X_t$')
axes[2].set_xlabel(r'Time $t$')

for ax in axes:
    ax.grid(alpha=0.3)
fig.tight_layout()
```

### Sample ACF and PACF

```python
lags = 40

fig, axes = plt.subplots(3, 2, figsize=(12, 9))
for i, (data, name) in enumerate([(wn, 'White noise'),
                                   (rw, 'Random walk'),
                                   (ar1, r'AR(1), $\phi = 0.8$')]):
    acf_vals = acf(data, nlags=lags, fft=True)
    pacf_vals = pacf(data, nlags=lags, method='ywm')
    ci = 1.96 / np.sqrt(len(data))

    axes[i,0].stem(range(lags+1), acf_vals, basefmt=' ')
    axes[i,0].axhspan(-ci, ci, alpha=0.15, color='gray')
    axes[i,0].set_title(f'ACF: {name}')
    axes[i,0].set_ylabel(r'$\hat{\rho}(h)$')

    axes[i,1].stem(range(lags+1), pacf_vals, basefmt=' ')
    axes[i,1].axhspan(-ci, ci, alpha=0.15, color='gray')
    axes[i,1].set_title(f'PACF: {name}')
    axes[i,1].set_ylabel(r'$\hat{\phi}_{hh}$')

for ax in axes[-1]:
    ax.set_xlabel(r'Lag $h$')
fig.tight_layout()
```

Expected patterns:
- **White noise**: ACF and PACF all inside the band at non-zero lags.
- **Random walk**: ACF decays linearly and very slowly — close to 1 for many lags.
- **AR(1) with \(\phi=0.8\)**: ACF decays geometrically as \(0.8^h\); PACF cuts off sharply after lag 1.

### The Three Tests

```python
def describe_series(x, name):
    print(f"\n=== {name} ===")
    adf_stat, adf_p, *_ = adfuller(x, autolag='AIC')
    print(f"ADF:        stat = {adf_stat:7.3f}, p = {adf_p:.4f}")
    kpss_stat, kpss_p, *_ = kpss(x, regression='c', nlags='auto')
    print(f"KPSS:       stat = {kpss_stat:7.3f}, p = {kpss_p:.4f}")
    lb = acorr_ljungbox(x, lags=[10], return_df=True).iloc[0]
    print(f"Ljung-Box(10): Q = {lb['lb_stat']:.2f}, p = {lb['lb_pvalue']:.4f}")

describe_series(wn, "White noise")
describe_series(rw, "Random walk")
describe_series(ar1, "AR(1), phi=0.8")
```

You should see:
- **WN**: ADF rejects strongly (stationary), KPSS fails to reject (stationary), Ljung-Box fails to reject (white).
- **RW**: ADF fails to reject (has a unit root), KPSS rejects (non-stationary), Ljung-Box strongly rejects (highly correlated).
- **AR(1)**: ADF rejects (stationary); KPSS fails to reject; Ljung-Box rejects (correlated, but that is the correct answer — the *levels* are autocorrelated, which is what the Ljung-Box detects. After fitting an AR(1), residuals should pass Ljung-Box.)

### STL Decomposition on a Real Series

```python
from statsmodels.tsa.seasonal import STL
import pandas as pd

# Simulate a series with trend + seasonal + noise
n = 240  # 20 years of monthly
t = np.arange(n)
trend = 0.05 * t
seasonal = 2.0 * np.sin(2 * np.pi * t / 12)
noise = rng.normal(0, 0.5, n)
x = trend + seasonal + noise
idx = pd.date_range('2000-01', periods=n, freq='MS')
series = pd.Series(x, index=idx)

stl = STL(series, period=12, robust=True).fit()

fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
axes[0].plot(series, color='#6db3f2', lw=1.0);  axes[0].set_ylabel(r'$X_t$')
axes[1].plot(stl.trend, color='#a4d08a', lw=1.2); axes[1].set_ylabel(r'$\hat{T}_t$')
axes[2].plot(stl.seasonal, color='#f2c894', lw=1.0); axes[2].set_ylabel(r'$\hat{S}_t$')
axes[3].plot(stl.resid, color='#f2a5a5', lw=0.8); axes[3].set_ylabel(r'$\hat{R}_t$')
axes[3].axhline(0, color='#555', lw=0.8, ls='--')
axes[3].set_xlabel('Date')
fig.suptitle('STL decomposition of simulated trend + seasonal + noise')
fig.tight_layout()
```

The residual component \(\hat{R}_t\) should look approximately like white noise. Run Ljung-Box on it to check.

### Estimating Mean and the Effective Sample Size

A subtle point: if your process is autocorrelated, the variance of the sample mean is not \(\sigma^2 / T\) but

$$
\mathrm{Var}(\bar{X}_T) \approx \frac{\sigma^2}{T} \cdot \underbrace{\sum_{h=-\infty}^\infty \rho(h)}_{\text{long-run variance factor}} = \frac{\sigma^2}{T_{\mathrm{eff}}}.
$$

The **effective sample size** \(T_{\mathrm{eff}}\) can be much smaller than \(T\) for persistent processes. For AR(1) with \(\phi = 0.8\), \(\sum \rho(h) = (1+\phi)/(1-\phi) = 9\), so \(T_{\mathrm{eff}} = T/9\). Ignoring this effectively makes your confidence intervals 3x too narrow.

```python
from statsmodels.tsa.stattools import acf

def effective_n(x, max_lag=None):
    n = len(x)
    if max_lag is None:
        max_lag = int(10 * np.log10(n))
    rho = acf(x, nlags=max_lag, fft=True)
    factor = 1 + 2 * sum(rho[1:])
    return n / max(factor, 1.0)

print(f"AR(1) phi=0.8: T = {T}, T_eff ~ {effective_n(ar1):.0f}")
```

You should see \(T_{\mathrm{eff}} \approx 100\) for \(T = 1000\), matching the theoretical factor of 9.

---

## 11. Industry Best Practices

The machinery in this post looks academic, but every item below is something a practitioner will actually do on the first day they inherit a forecasting problem. Skipping any of them is how you ship a model that passes code review and fails in production.

### 11.1 Always Plot the Series First — Four Views

Before any test or model:

1. **Levels**: the raw series. Look for trend, level shifts, outliers, gaps, changes in volatility.
2. **Differences** \(\Delta X_t\): often reveals heteroskedasticity and event days that hide at the level.
3. **ACF/PACF**: on levels *and* on residuals after any transformation. If ACF of the raw series decays very slowly, you likely have a unit root.
4. **Rolling mean and rolling standard deviation**: a cheap visual stationarity check. Expanding windows are misleading; use rolling windows of a full seasonal period.

A surprising amount of forecasting failure comes from not looking at the data. Structural breaks, DST transitions, unit changes, and reporting delays all show up in plots long before they show up in test metrics.

### 11.2 Cleaning Checklist Before Modeling

- **Time zone and calendar**: align to a single timezone; document DST handling (forecast errors spike at DST transitions if unhandled). Business-day vs. calendar-day indexing changes ACF structure.
- **Missing values**: distinguish *structurally missing* (holidays, weekends for business metrics) from *reporting lag*. Don't forward-fill without thought — it manufactures fake autocorrelation.
- **Outliers**: flag but do not delete. Winsorize only if you understand why. Large true moves (COVID, product launches) are data, not noise; hiding them biases the model toward overconfidence.
- **Level shifts**: detected via CUSUM, Bai-Perron, or simple domain inspection. Either include dummy regressors or split the series at the break — do not fit a single stationary model across a break.
- **Revisions**: GDP, inventory, and many business metrics get revised months after publication. For honest backtests, use the *vintage* data available at forecast time, not the final revised series.

### 11.3 Always Establish a Benchmark Forecast

No model is "good" in isolation — it is good relative to something naive. Standard benchmarks:

- **Naive-1**: \(\hat{X}_{T+h} = X_T\). Your model must beat this or it is not earning its complexity.
- **Seasonal naive**: \(\hat{X}_{T+h} = X_{T+h-s}\). The relevant benchmark for seasonal series.
- **Random walk with drift**: \(\hat{X}_{T+h} = X_T + h\hat{\delta}\). The benchmark for trending series.
- **Historical mean / median**: for series that are truly stationary around a level.

Report Mean Absolute Scaled Error (MASE) — the MAE of your model divided by the MAE of the seasonal naive on the training set. MASE < 1 means you beat the benchmark. This scales across heterogeneous items and is the metric M-competitions use.

### 11.4 Use Proper Time Series Cross-Validation

*Never* use random K-fold on time series. It leaks the future into the past. Two correct schemes:

- **Expanding window** (rolling origin): fit on \([1, t]\), forecast \([t+1, t+h]\), advance \(t\), repeat. This is the Hyndman-Athanasopoulos gold standard.
- **Sliding window**: same but the training window slides forward with fixed length. Use when you suspect non-stationarity and want to discard old regimes.

Always evaluate at the forecast horizon you care about. A one-step MAE tells you almost nothing about 30-step performance.

### 11.5 Separate Three Kinds of Uncertainty

A production forecast has three distinct sources of error:

1. **Innovation uncertainty** \(\sigma^2\) — the irreducible noise. Shows up in prediction intervals even if the model is perfectly specified.
2. **Parameter uncertainty** — \(\hat{\phi}\) is not \(\phi\). Usually small for long series, large for short/seasonal ones.
3. **Model uncertainty** — the wrong family. The biggest of the three, and the one standard prediction intervals ignore entirely.

Bayesian methods (Part 4) and ensembles (Part 5) are the two production-grade responses to model uncertainty. Most shops under-report intervals by a factor of 2–3× because they account only for (1).

### 11.6 Test Stationarity With Two Tests, Not One

Run both **ADF** (null = unit root) and **KPSS** (null = stationary). They are complementary:

- ADF reject + KPSS not reject ⇒ confidently stationary.
- ADF not reject + KPSS reject ⇒ confidently non-stationary; difference once.
- Both reject ⇒ suspect fractional integration, structural break, or near-integrated behavior.
- Neither rejects ⇒ low power zone; rely on domain knowledge and model diagnostics.

Neither test has good power in samples under ~100, and neither handles level shifts well. Do not treat them as oracles — treat them as votes alongside the ACF plot and domain priors.

### 11.7 Log-Transform Before Differencing, Not After

For strictly positive series with multiplicative seasonality (revenue, bookings, traffic), take logs *first*. Differencing stabilizes the mean; logging stabilizes the variance. Applied in the wrong order, you get an arithmetically-differenced series with exploding variance, which then requires more complicated models than the problem deserves.

Box-Cox (\(\lambda\)) is the parametric generalization:

$$
y(\lambda) = \begin{cases} (X^\lambda - 1)/\lambda & \lambda \ne 0 \\ \ln X & \lambda = 0 \end{cases}
$$

Guerrero's method picks \(\lambda\) to minimize the coefficient of variation across seasonal blocks; `forecast::BoxCox.lambda` in R and the `scipy.stats.boxcox` helper are the standard implementations. Remember to *back-transform* your forecasts and adjust for the Jensen gap: \(\mathbb{E}[\exp Y] > \exp(\mathbb{E} Y)\).

### 11.8 Monitor Residuals in Production

A model that passes Ljung-Box on the training set can still fail on live data — the environment drifts. Every production forecasting system should:

- Log residuals at every forecast horizon.
- Run a rolling Ljung-Box and report p-values weekly.
- Track forecast bias: \(\text{mean}(\hat{X}_{t+h|t} - X_{t+h})\). If it drifts away from zero, something is wrong with the mean equation.
- Track coverage: what fraction of realized values fall inside the nominal 80% / 95% prediction interval? For calibrated models, coverage should match the nominal level to within Monte Carlo error.

Under-coverage in production is almost universal. The rolling CRPS (Part 5) is a more robust tracking metric than MAE because it reflects the entire predictive distribution rather than a point.

### 11.9 Treat Seasonality As a Modeling Choice, Not a Fact

"Weekly seasonality" is not a property of the data. It is a *model* you impose. Real human and economic systems have multiple overlapping periodicities (daily, weekly, monthly billing cycles, quarterly business rhythms, annual holidays), and forcing a single seasonality into ARIMA is often wrong. Options:

- **SARIMA** for a single period (Part 2).
- **TBATS / Fourier regressors** for multiple seasonalities. You include pairs \(\sin(2\pi k t / s)\), \(\cos(2\pi k t / s)\) as exogenous regressors for each relevant period \(s\).
- **STL with multiple periods (MSTL)** for exploratory decomposition.
- **Calendar features** (holiday dummies, payday effects, Black Friday) for known discrete events.

Misspecifying seasonality shows up as residual autocorrelation at the missing frequency. Check the residual ACF at lag \(s\) for your candidate periods.

### 11.10 Keep Model Simplicity a Variable You Optimize

The M4 and M5 competitions both found that combining simple models (ETS, ARIMA, Theta) in an equal-weight ensemble beats any single complex model on most operational forecasting problems. The heuristic for production:

- **Fewer than 50 observations**: no ARIMA, no ML — seasonal naive or simple exponential smoothing.
- **50–500 observations, clean seasonality**: ETS or SARIMA.
- **500–5000, covariates matter**: SARIMAX, regression with ARIMA errors, or gradient-boosted trees with lag features.
- **> 5000, many related series**: hierarchical reconciliation and/or global ML models (Part 5).

Complexity must be *earned* against the benchmark MASE on expanding-window CV. A model that wins on train and ties on CV is a worse choice than the simpler model it ties with.

---

## Summary and What's Next

We have built the first layer of forecasting vocabulary:

- A time series is a collection of random variables indexed by time; we observe one realization, and the job of a forecaster is to push that realization one step (or \(h\) steps) into the future.
- **Stationarity** (weak) means constant mean and lag-dependent-only autocovariance; it is the structural assumption under which past informs future and forecasts have a well-defined target.
- The **autocovariance/autocorrelation** \(\gamma(h)/\rho(h)\) summarizes linear memory; sample versions are consistent under ergodicity and are the first quantities a forecaster reads off the data.
- The **PACF** cleans autocorrelation of indirect effects via intermediate lags; combined with the ACF, it identifies the order of ARMA models.
- **White noise** and the **random walk** are the two prototypes: white noise is the target shape of well-specified forecast residuals; the random walk is the benchmark every forecaster must beat.
- **Wold decomposition** says every stationary process is a linear filter of its own innovations — the formal reason ARMA-type forecasting works.
- **Ergodicity** is what lets you estimate forecast-relevant quantities from one sample path.
- Real series have trends and seasonality; differencing, logs, and STL preprocess them into something a stationary model can consume; ADF and KPSS gate the transition.
- **Industry practice** is a discipline, not a decoration: plot first, benchmark always, cross-validate on time, log coverage in production.

In [Part 2](/2026/04/19/arima-box-jenkins-forecasting.html) we take finite-parameter approximations to Wold's MA(\(\infty\)) and build the ARIMA family — autoregressive, moving average, integrated, seasonal — together with the Box-Jenkins identification/estimation/diagnosis cycle, AIC/BIC model selection, forecasting formulas, and prediction intervals. [Part 3](/2026/04/20/exponential-smoothing-ets-theta.html) develops exponential smoothing and the state-space ETS family (Holt-Winters, damped trend, Theta), which win on shorter series. [Part 4](/2026/04/21/state-space-kalman-filtering.html) unifies everything under state-space representation and the Kalman filter/smoother. [Part 5](/2026/04/22/modern-forecasting-garch-gbm-nbeats-hierarchical.html) covers modern practice: GARCH for volatility forecasting, gradient-boosted trees with lag features, N-BEATS (a deep MLP architecture), hierarchical reconciliation, and the calibration / proper-scoring-rule framework for probabilistic forecasts.
