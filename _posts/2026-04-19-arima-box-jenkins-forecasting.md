---
layout: post
title: "ARIMA and the Box-Jenkins Method: Forecasting with Linear Time Series Models"
date: 2026-04-19
category: math
---

*This is Part 2 of a 5-part series on time series forecasting. [Part 1: Foundations](/2026/04/18/time-series-foundations-stationarity.html) | **Part 2: ARIMA & Box-Jenkins** | [Part 3: Exponential Smoothing, ETS & Theta](/2026/04/20/exponential-smoothing-ets-theta.html) | [Part 4: State-Space & Kalman](/2026/04/21/state-space-kalman-filtering.html) | [Part 5: Modern Forecasting](/2026/04/22/modern-forecasting-garch-gbm-nbeats-hierarchical.html)*

In Part 1 we proved, via Wold's decomposition, that every weakly stationary process can be written as an infinite-order moving average of its own innovations. That result is beautiful and useless on its own — you cannot estimate infinitely many parameters from a finite sample. ARIMA is the practical engineering compromise: we approximate that MA(\(\infty\)) by a *ratio* of two finite-order polynomials in the lag operator, squeeze most of the useful memory into a handful of parameters, and close the loop with an estimator, a forecasting formula, and a diagnostic procedure.

ARIMA is still, despite forty years of subsequent development, the most consequential time series forecasting model in industry. It wins on short-to-medium series, on clean data, on series with modest nonlinearity, and — critically — on series where you need calibrated prediction intervals, not just point forecasts. It is the model the forecasting competitions use as a baseline, the model a manager wants you to compare a neural net against, and the model you will ship in quarter one of any new forecasting project. This post builds it from first principles.

We start with AR, MA, and ARMA processes — their definitions, stationarity conditions, and ACF/PACF signatures. We then handle non-stationarity via integration (the "I"), give the full ARIMA and SARIMA specifications, and derive the recursive forecasting formula and its prediction intervals. We cover estimation (MLE, conditional sum of squares, Yule-Walker), model selection (AIC/BIC/AICc), and the Box-Jenkins iterative identification-estimation-diagnosis cycle. We then implement everything in Python on a real-ish demand series and close with industry best practices.

---

## Table of Contents

1. [AR, MA, and ARMA Processes](#1-ar-ma-and-arma-processes)
2. [Stationarity, Invertibility, and the Characteristic Polynomial](#2-stationarity-invertibility-and-the-characteristic-polynomial)
3. [ACF and PACF Signatures](#3-acf-and-pacf-signatures)
4. [ARIMA: Handling Non-Stationarity via Integration](#4-arima-handling-non-stationarity-via-integration)
5. [SARIMA and Seasonal Structure](#5-sarima-and-seasonal-structure)
6. [Estimation: MLE, CSS, and Yule-Walker](#6-estimation-mle-css-and-yule-walker)
7. [Model Selection: AIC, BIC, AICc](#7-model-selection-aic-bic-aicc)
8. [Forecasting: Point and Interval](#8-forecasting-point-and-interval)
9. [The Box-Jenkins Methodology](#9-the-box-jenkins-methodology)
10. [ARIMAX and Regression with ARIMA Errors](#10-arimax-and-regression-with-arima-errors)
11. [Python: Full Forecasting Pipeline](#11-python-full-forecasting-pipeline)
12. [Industry Best Practices](#12-industry-best-practices)

---

## 1. AR, MA, and ARMA Processes

Throughout, \(\varepsilon_t \sim \mathrm{WN}(0, \sigma^2)\) is white noise, and \(L\) is the lag operator (\(LX_t = X_{t-1}\)).

### Autoregressive AR(p)

An **autoregressive process of order \(p\)** is defined by

$$
X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \ldots + \phi_p X_{t-p} + \varepsilon_t.
$$

Equivalently, in lag-operator form:

$$
\phi(L) X_t = \varepsilon_t, \qquad \phi(L) = 1 - \phi_1 L - \phi_2 L^2 - \ldots - \phi_p L^p.
$$

The AR(\(p\)) model says: the current value is a linear combination of the previous \(p\) values plus an innovation. It captures persistence of arbitrary decay shape (within the family of mixtures of exponentials and oscillating terms), which is why it wins on most macroeconomic and operational forecasting problems.

### Moving Average MA(q)

A **moving average process of order \(q\)** is defined by

$$
X_t = \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \ldots + \theta_q \varepsilon_{t-q}.
$$

In lag-operator form: \(X_t = \theta(L) \varepsilon_t\), with \(\theta(L) = 1 + \theta_1 L + \ldots + \theta_q L^q\). MA processes model short-run shocks that linger for exactly \(q\) periods and then vanish. Classic applications: reporting corrections, revisions, short-lived shocks.

### ARMA(p,q)

The combined **ARMA(\(p, q\))** process is

$$
\phi(L) X_t = \theta(L) \varepsilon_t.
$$

Equivalently \(X_t = \phi_1 X_{t-1} + \ldots + \phi_p X_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \ldots + \theta_q \varepsilon_{t-q}\).

ARMA is the practical workhorse. By appropriately choosing \(p\) and \(q\), it approximates any stationary linear process — Wold's MA(\(\infty\)) becomes a rational function \(\theta(L)/\phi(L)\) that can reproduce exponential decay, damped oscillation, and moving-window shock dynamics with a few parameters each.

### Include a Mean

In practice we almost always model the deviation from a mean:

$$
\phi(L)(X_t - \mu) = \theta(L) \varepsilon_t.
$$

Expanding, this is \(\phi(L) X_t = \phi(1) \mu + \theta(L) \varepsilon_t\), where \(\phi(1) \mu\) is an intercept term. Most software estimates \(\mu\) explicitly; conceptually, subtract the mean, model the centered series, add back.

---

## 2. Stationarity, Invertibility, and the Characteristic Polynomial

Writing down an ARMA model does not make the implied process stationary. The model \(X_t = 1.2 X_{t-1} + \varepsilon_t\) explodes. For forecasting we need two properties: **causal stationarity** (so the mean and variance are finite and constant) and **invertibility** (so the innovations can be recovered from the observables — critical for forecasting).

### Causal Stationarity

An ARMA process is **causal stationary** if it can be written as a one-sided MA(\(\infty\)) in past innovations:

$$
X_t = \sum_{j=0}^\infty \psi_j \varepsilon_{t-j}, \qquad \sum \psi_j^2 < \infty.
$$

Algebraically this amounts to being able to invert \(\phi(L)\): \(X_t = \phi(L)^{-1} \theta(L) \varepsilon_t\). The formal condition is on the roots of the AR polynomial:

$$
\phi(z) = 1 - \phi_1 z - \phi_2 z^2 - \ldots - \phi_p z^p = 0 \quad\Rightarrow\quad |z| > 1 \text{ for all roots}.
$$

All roots of \(\phi(z)\) must lie **strictly outside the unit circle** in the complex plane. Some references equivalently state the condition on the reverse polynomial \(z^p \phi(1/z)\); in that formulation the roots must lie strictly inside the unit disk. Either way, the essential content is: the AR operator must be invertible as a convergent geometric series in \(L\).

**AR(1) example.** For \(X_t = \phi X_{t-1} + \varepsilon_t\), the root of \(\phi(z) = 1 - \phi z\) is \(z = 1/\phi\). Stationarity requires \(|1/\phi| > 1\), i.e., \(|\phi| < 1\). When \(|\phi| < 1\):

$$
X_t = \sum_{j=0}^\infty \phi^j \varepsilon_{t-j}, \quad \mathbb{E}[X_t] = 0, \quad \mathrm{Var}(X_t) = \frac{\sigma^2}{1 - \phi^2}.
$$

**AR(2) example.** For \(X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \varepsilon_t\), stationarity requires the triangle

$$
\phi_1 + \phi_2 < 1, \quad \phi_2 - \phi_1 < 1, \quad |\phi_2| < 1.
$$

Equivalently, the roots of \(1 - \phi_1 z - \phi_2 z^2 = 0\) lie outside the unit circle. This triangular region in the \(\phi_1\)-\(\phi_2\) plane is one of the most plotted diagrams in time series textbooks — inside it, the process is stationary; outside it, explosive. Along the upper curve \(\phi_2 = 1 - \phi_1\) sits a unit root.

### Invertibility

By symmetry with stationarity, the MA operator \(\theta(L)\) is **invertible** if the roots of \(\theta(z) = 0\) lie strictly outside the unit circle. When this holds, we can write the process as an AR(\(\infty\)) in past observations:

$$
\varepsilon_t = \theta(L)^{-1} \phi(L) X_t = \sum_{j=0}^\infty \pi_j X_{t-j},
$$

for some \(\pi_j\). Invertibility matters for *forecasting* because we need to recover \(\hat{\varepsilon}_t\) — the unobserved innovation — from observed \(X_t\)s to make forecasts. A non-invertible MA model is not identifiable from the second-moment structure alone: an MA(1) with parameter \(\theta\) and \(\sigma^2\) has the exact same ACF as an MA(1) with parameter \(1/\theta\) and \(\theta^2 \sigma^2\). We always pick the invertible representation.

### Summary of Parameter Space

For a usable ARMA(\(p, q\)):

- Roots of \(\phi(z) = 0\) lie outside the unit circle ⇒ causal stationary.
- Roots of \(\theta(z) = 0\) lie outside the unit circle ⇒ invertible.
- \(\phi(z)\) and \(\theta(z)\) share no common roots ⇒ parameters are identified.

Software enforces these with reparameterizations (partial autocorrelation parameterization of Jones, 1980) so that optimizers cannot wander into non-stationary regions.

---

## 3. ACF and PACF Signatures

Box-Jenkins identification rests on one pattern: AR and MA leave complementary fingerprints on the ACF and PACF. Reading the correlogram identifies the model family. We work out the theoretical ACF/PACF for the basic cases, then state the general rule.

### MA(q) ACF

For \(X_t = \varepsilon_t + \theta_1 \varepsilon_{t-1} + \ldots + \theta_q \varepsilon_{t-q}\):

$$
\gamma(h) = \begin{cases} \sigma^2 \sum_{j=0}^{q-h} \theta_j \theta_{j+h} & |h| \le q \\ 0 & |h| > q \end{cases}
$$

(with \(\theta_0 = 1\)). The ACF **cuts off** sharply at lag \(q\): nonzero for \(h \le q\), exactly zero for \(h > q\). Visually, an MA(2) correlogram has two significant spikes at lags 1 and 2 and noise thereafter.

### AR(p) ACF

The AR(\(p\)) ACF satisfies the **Yule-Walker equations**:

$$
\rho(h) = \phi_1 \rho(h-1) + \phi_2 \rho(h-2) + \ldots + \phi_p \rho(h-p), \quad h \ge 1.
$$

This is a linear difference equation whose solution is a mixture of exponentials and damped sinusoids — the roots of \(\phi(z)\) determine the decay. The ACF **decays** (possibly oscillating) without a clean cutoff. Visually: AR(1) with \(\phi = 0.7\) gives a geometric decay \(0.7^h\); AR(2) with complex roots gives damped oscillation.

### AR(p) PACF

The PACF of an AR(\(p\)) **cuts off** at lag \(p\): \(\phi_{hh} = 0\) for \(h > p\). Intuition: once you condition on \(X_{t-1}, \ldots, X_{t-p}\), there is no residual linear information in deeper lags because the AR model *is* the best linear predictor from those.

### MA(q) PACF

The PACF of an invertible MA(\(q\)) **decays** geometrically (and can oscillate) without cutting off. Symmetric to AR(\(p\)) ACF behavior.

### The Box-Jenkins Table

| Process | ACF | PACF |
|---|---|---|
| White noise | zero at all lags \(h \ge 1\) | zero at all lags \(h \ge 1\) |
| AR(\(p\)) | tails off | cuts off after lag \(p\) |
| MA(\(q\)) | cuts off after lag \(q\) | tails off |
| ARMA(\(p, q\)) | tails off | tails off |

Knowing this table is the first half of model identification. The second half is *verifying* your identification by fitting candidate models and comparing AIC/BIC and residual diagnostics — never trust the correlogram alone.

<svg viewBox="0 0 720 320" xmlns="http://www.w3.org/2000/svg">
  <rect width="720" height="320" fill="#1a1a1a"/>
  <text x="360" y="22" fill="#e8e8e8" font-size="15" text-anchor="middle" font-family="Georgia, serif" font-weight="bold">Theoretical ACF and PACF for AR(1), MA(1), ARMA(1,1)</text>
  <!-- Three panels: ACF of AR(1), MA(1), ARMA(1,1) -->
  <!-- AR(1) phi = 0.7, ACF = 0.7^h, PACF lag1 = 0.7, 0 after -->
  <g transform="translate(20,50)">
    <text x="100" y="-5" fill="#6db3f2" font-size="12" text-anchor="middle" font-family="Georgia, serif">AR(1), φ=0.7</text>
    <line x1="0" y1="120" x2="200" y2="120" stroke="#888" stroke-width="1"/>
    <!-- ACF bars decaying -->
    <rect x="10" y="0" width="8" height="120" fill="#6db3f2"/>
    <rect x="28" y="36" width="8" height="84" fill="#6db3f2"/>
    <rect x="46" y="60" width="8" height="60" fill="#6db3f2"/>
    <rect x="64" y="78" width="8" height="42" fill="#6db3f2"/>
    <rect x="82" y="91" width="8" height="29" fill="#6db3f2"/>
    <rect x="100" y="100" width="8" height="20" fill="#6db3f2"/>
    <rect x="118" y="106" width="8" height="14" fill="#6db3f2"/>
    <rect x="136" y="110" width="8" height="10" fill="#6db3f2"/>
    <rect x="154" y="113" width="8" height="7" fill="#6db3f2"/>
    <rect x="172" y="115" width="8" height="5" fill="#6db3f2"/>
    <text x="100" y="250" fill="#e8e8e8" font-size="11" text-anchor="middle" font-family="Georgia, serif">ACF (tails off)</text>
    <!-- PACF below -->
    <g transform="translate(0,140)">
      <line x1="0" y1="120" x2="200" y2="120" stroke="#888" stroke-width="1"/>
      <rect x="10" y="0" width="8" height="120" fill="#a4d08a"/>
      <rect x="28" y="36" width="8" height="84" fill="#a4d08a"/>
      <!-- remaining ~0 -->
      <text x="100" y="155" fill="#e8e8e8" font-size="11" text-anchor="middle" font-family="Georgia, serif">PACF (cuts off at 1)</text>
    </g>
  </g>
  <!-- MA(1) theta = 0.7 -->
  <g transform="translate(260,50)">
    <text x="100" y="-5" fill="#f2c894" font-size="12" text-anchor="middle" font-family="Georgia, serif">MA(1), θ=0.7</text>
    <line x1="0" y1="120" x2="200" y2="120" stroke="#888" stroke-width="1"/>
    <rect x="10" y="0" width="8" height="120" fill="#f2c894"/>
    <rect x="28" y="60" width="8" height="60" fill="#f2c894"/>
    <!-- rest zero -->
    <text x="100" y="250" fill="#e8e8e8" font-size="11" text-anchor="middle" font-family="Georgia, serif">ACF (cuts off at 1)</text>
    <g transform="translate(0,140)">
      <line x1="0" y1="120" x2="200" y2="120" stroke="#888" stroke-width="1"/>
      <rect x="10" y="0" width="8" height="120" fill="#a4d08a"/>
      <rect x="28" y="40" width="8" height="80" fill="#a4d08a"/>
      <rect x="46" y="65" width="8" height="55" fill="#a4d08a"/>
      <rect x="64" y="84" width="8" height="36" fill="#a4d08a"/>
      <rect x="82" y="97" width="8" height="23" fill="#a4d08a"/>
      <rect x="100" y="106" width="8" height="14" fill="#a4d08a"/>
      <text x="100" y="155" fill="#e8e8e8" font-size="11" text-anchor="middle" font-family="Georgia, serif">PACF (tails off)</text>
    </g>
  </g>
  <!-- ARMA(1,1) -->
  <g transform="translate(500,50)">
    <text x="100" y="-5" fill="#f2a5a5" font-size="12" text-anchor="middle" font-family="Georgia, serif">ARMA(1,1)</text>
    <line x1="0" y1="120" x2="200" y2="120" stroke="#888" stroke-width="1"/>
    <rect x="10" y="0" width="8" height="120" fill="#f2a5a5"/>
    <rect x="28" y="25" width="8" height="95" fill="#f2a5a5"/>
    <rect x="46" y="45" width="8" height="75" fill="#f2a5a5"/>
    <rect x="64" y="65" width="8" height="55" fill="#f2a5a5"/>
    <rect x="82" y="80" width="8" height="40" fill="#f2a5a5"/>
    <rect x="100" y="92" width="8" height="28" fill="#f2a5a5"/>
    <rect x="118" y="101" width="8" height="19" fill="#f2a5a5"/>
    <rect x="136" y="107" width="8" height="13" fill="#f2a5a5"/>
    <text x="100" y="250" fill="#e8e8e8" font-size="11" text-anchor="middle" font-family="Georgia, serif">ACF (tails off)</text>
    <g transform="translate(0,140)">
      <line x1="0" y1="120" x2="200" y2="120" stroke="#888" stroke-width="1"/>
      <rect x="10" y="0" width="8" height="120" fill="#a4d08a"/>
      <rect x="28" y="30" width="8" height="90" fill="#a4d08a"/>
      <rect x="46" y="60" width="8" height="60" fill="#a4d08a"/>
      <rect x="64" y="82" width="8" height="38" fill="#a4d08a"/>
      <rect x="82" y="97" width="8" height="23" fill="#a4d08a"/>
      <rect x="100" y="106" width="8" height="14" fill="#a4d08a"/>
      <text x="100" y="155" fill="#e8e8e8" font-size="11" text-anchor="middle" font-family="Georgia, serif">PACF (tails off)</text>
    </g>
  </g>
</svg>

---

## 4. ARIMA: Handling Non-Stationarity via Integration

Most real series are not stationary in level. Revenue grows, traffic trends, temperatures shift. ARIMA handles this by *differencing*.

### The I(d) Process

A series \(X_t\) is **integrated of order \(d\)**, written \(I(d)\), if the \(d\)-th difference \(\Delta^d X_t\) is stationary, where \(\Delta = 1 - L\) and \(\Delta^d = (1 - L)^d\). Specifically:

- \(I(0)\): stationary as observed.
- \(I(1)\): differencing once gives stationary. The prototypical example is the random walk.
- \(I(2)\): differencing twice. Rare in practice — mostly arises with strong quadratic trends.

Higher orders of integration are almost never needed. An \(I(3)\) hypothesis is a signal that you have the wrong model.

### ARIMA(p, d, q)

An **ARIMA(\(p, d, q\))** model applies ARMA(\(p, q\)) to the \(d\)-th difference:

$$
\phi(L)(1 - L)^d X_t = \theta(L) \varepsilon_t.
$$

The differencing operator \(\left(1 - L\right)^d\) introduces \(d\) roots at \(z = 1\) — the unit circle. These are "unit roots," and their presence is what makes \(X_t\) non-stationary while \(\Delta^d X_t\) is stationary.

### How to Choose d

Two practical rules:

1. **Plot \(X_t\), \(\Delta X_t\), \(\Delta^2 X_t\) with ACFs.** The correct \(d\) is the smallest one for which the ACF of \(\Delta^d X_t\) decays quickly (not linearly) to zero.
2. **Use unit-root tests sequentially.** Test \(X_t\) with ADF; if it fails to reject, difference and test \(\Delta X_t\). Stop when ADF rejects. Cross-check with KPSS (Part 1).

Over-differencing is harmful: \(\Delta \varepsilon_t\) is an MA(1) with \(\theta = -1\), a non-invertible model. If you difference a stationary series, you inject a unit root into the MA side. So: *difference only if you must*.

### ARIMA(p, 1, q) and Drift

For an ARIMA(\(p, 1, q\)) with a constant term:

$$
\phi(L)(1 - L)X_t = c + \theta(L)\varepsilon_t
$$

the constant \(c\) induces a deterministic linear trend in levels. This is the standard model for trending series. If the true trend is stochastic (i.e., truly \(I(1)\) with drift), leaving in a level-form constant gives a quadratic trend in levels — almost always wrong. Most software defaults to "no constant" when \(d \ge 1\); add one only when domain knowledge supports a deterministic drift.

---

## 5. SARIMA and Seasonal Structure

Many series have seasonal structure: monthly data with annual cycle, daily data with weekly cycle, hourly with daily cycle. **SARIMA** — seasonal ARIMA — extends ARIMA with an additional seasonal block. The general form is

$$
\text{SARIMA}(p, d, q) \times (P, D, Q)_s,
$$

where \(s\) is the seasonal period and \(\left(P, D, Q\right)\) are the orders of the seasonal AR, seasonal differencing, and seasonal MA. The full model:

$$
\phi(L)\Phi(L^s)(1 - L)^d (1 - L^s)^D X_t = \theta(L) \Theta(L^s) \varepsilon_t,
$$

where \(\Phi\) and \(\Theta\) are polynomials in \(L^s\). The operator \(\left(1 - L^s\right)\) is **seasonal differencing**: \(\Delta_s X_t = X_t - X_{t-s}\). It removes constant seasonal patterns and deterministic seasonal trends.

### Standard Choices

For monthly data with annual seasonality (\(s = 12\)):

- **Airline model: SARIMA(0,1,1)(0,1,1)\(_{12}\)**. The famous Box-Jenkins parameterization for Airline Passenger data, and still a default starting point for monthly seasonal series. Both a non-seasonal MA(1) and a seasonal MA(1), first-difference and seasonal-first-difference — six parameters in total including variance.
- **Simple seasonal ARIMA: SARIMA(1,0,0)(1,1,0)\(_{12}\)** for series with a persistent but bounded seasonal pattern.

For daily data with weekly seasonality (\(s = 7\)): substitute 7 for 12 in all of the above.

### Multiple Seasonalities

When you have both weekly and annual patterns (daily data, \(s_1 = 7\), \(s_2 = 365\)), SARIMA with one \(s\) is insufficient. Options:

- **SARIMA with Fourier regressors**: model weekly seasonality with SARIMA and annual seasonality with external \(\sin(2\pi k t / 365)\), \(\cos(2\pi k t / 365)\) regressors.
- **TBATS** (Trigonometric Box-Cox ARMA with Trend and Seasonality): native multi-seasonal support.
- **STL + ARIMA**: decompose seasonally via STL, model residual with ARIMA, recompose forecasts.

Multi-seasonal SARIMA — e.g., SARIMA\(\left(p,d,q\right)\left(P_1, D_1, Q_1\right)_{s_1}\left(P_2, D_2, Q_2\right)_{s_2}\) — is possible but the parameter space balloons and estimation is unreliable.

---

## 6. Estimation: MLE, CSS, and Yule-Walker

### Maximum Likelihood Estimation

For Gaussian \(\varepsilon_t\), the joint density of \(\left(X_1, \ldots, X_T\right)\) is multivariate normal with covariance matrix \(\Sigma(\boldsymbol{\eta})\) determined by the ARMA parameters \(\boldsymbol{\eta} = \left(\phi_1, \ldots, \phi_p, \theta_1, \ldots, \theta_q, \sigma^2\right)\). The log-likelihood is

$$
\ell(\boldsymbol{\eta}) = -\frac{T}{2}\log(2\pi) - \frac{1}{2}\log|\Sigma| - \frac{1}{2} \mathbf{X}^\top \Sigma^{-1} \mathbf{X}.
$$

Direct evaluation is \(O(T^3)\), which is too expensive for long series. The trick is to compute \(\ell\) recursively via the **Kalman filter** (Part 4): ARMA models can be written in state-space form, and the filter produces one-step-ahead prediction errors and their variances in \(O(T(p+q)^2)\) time, from which \(\ell\) is built. This is how `statsmodels` and `arima` in R compute exact MLE.

Under regularity conditions (stationarity, invertibility, identifiability), \(\hat{\boldsymbol{\eta}}_{\text{MLE}}\) is consistent and asymptotically normal:

$$
\sqrt{T}(\hat{\boldsymbol{\eta}} - \boldsymbol{\eta}) \xrightarrow{d} \mathcal{N}(0, I(\boldsymbol{\eta})^{-1}),
$$

where \(I\) is the information matrix. Standard errors in software output come from the observed Hessian.

### Conditional Sum of Squares (CSS)

A computationally lighter approximation: treat the first \(p\) observations (and the first \(q\) innovations, set to zero) as initial conditions, then minimize

$$
S(\boldsymbol{\eta}) = \sum_{t=p+1}^T \hat{\varepsilon}_t^2,
$$

where \(\hat{\varepsilon}_t\) is computed recursively from the ARMA equation given current parameter estimates. This is asymptotically equivalent to MLE for large \(T\), cheaper per iteration, but loses information from the first \(p+q\) observations. Default for `auto.arima` when \(T\) is very large or MLE fails to converge.

### Yule-Walker (AR only)

For pure AR(\(p\)), the Yule-Walker equations give closed-form estimates from sample autocorrelations:

$$
\begin{pmatrix} \hat{\rho}(0) & \hat{\rho}(1) & \ldots & \hat{\rho}(p-1) \\ \hat{\rho}(1) & \hat{\rho}(0) & \ldots & \hat{\rho}(p-2) \\ \vdots & & \ddots & \vdots \\ \hat{\rho}(p-1) & \ldots & & \hat{\rho}(0) \end{pmatrix} \begin{pmatrix} \hat{\phi}_1 \\ \hat{\phi}_2 \\ \vdots \\ \hat{\phi}_p \end{pmatrix} = \begin{pmatrix} \hat{\rho}(1) \\ \hat{\rho}(2) \\ \vdots \\ \hat{\rho}(p) \end{pmatrix}.
$$

Solve the Toeplitz system with Durbin-Levinson in \(O(p^2)\). Useful as a starting point for MLE and as a quick diagnostic for small-\(p\) models.

---

## 7. Model Selection: AIC, BIC, AICc

Having fit multiple candidate ARMA(\(p, q\)) models, we pick one by minimizing an information criterion.

### AIC

**Akaike's Information Criterion:**

$$
\mathrm{AIC} = -2 \ell(\hat{\boldsymbol{\eta}}) + 2k,
$$

where \(k = p + q + 1\) (plus a constant if fitted). AIC estimates the Kullback-Leibler divergence between the fitted model and the true data-generating process, penalizing each added parameter by 2. Optimal in a predictive sense: asymptotically, AIC-selected models have minimum one-step-ahead prediction error.

### BIC

**Bayesian Information Criterion:**

$$
\mathrm{BIC} = -2 \ell(\hat{\boldsymbol{\eta}}) + k \log T.
$$

Penalty grows with \(T\). BIC is consistent for model selection when the true model is in the candidate set: as \(T \to \infty\), BIC selects the true model with probability 1. AIC is not consistent — it over-selects — but is often better for prediction in small samples.

### AICc

**Corrected AIC** for small \(T\):

$$
\mathrm{AICc} = \mathrm{AIC} + \frac{2k(k+1)}{T - k - 1}.
$$

Hyndman's `forecast::auto.arima` and `pmdarima` default to AICc. When \(T / k > 40\), AIC ≈ AICc. For short series (T < 100), always use AICc.

### Practical Advice

- Use **AICc** for short series (say \(T < 100\)).
- Use **BIC** when parsimony matters and you expect the true model is relatively simple.
- Use **AIC** or **AICc** when forecast accuracy matters more than recovering the true model.
- Do not compare ICs across differences. An ARIMA(1,0,0) and an ARIMA(1,1,0) are fit to different data (levels vs. differences); their likelihoods are not comparable.

---

## 8. Forecasting: Point and Interval

This is the part that matters. You have fit \(\hat{\phi}(L)\hat{\Delta}^d X_t = \hat{\theta}(L) \hat{\varepsilon}_t\). Now produce \(\hat{X}_{T+h|T}\) and a prediction interval.

### The Recursion

Write the model in causal MA(\(\infty\)) form: \(X_t = \sum_j \psi_j \varepsilon_{t-j}\). Then

$$
X_{T+h} = \sum_{j=0}^{h-1} \psi_j \varepsilon_{T+h-j} + \sum_{j=h}^\infty \psi_j \varepsilon_{T+h-j}.
$$

The first sum involves future (unknown) innovations; its conditional expectation given \(\mathcal{F}_T\) is zero. The second sum uses past (observable) innovations. The **minimum MSE point forecast** is

$$
\hat{X}_{T+h|T} = \mathbb{E}[X_{T+h} \mid \mathcal{F}_T] = \sum_{j=h}^\infty \psi_j \varepsilon_{T+h-j}.
$$

In practice we compute it recursively. For ARMA(\(p, q\)):

$$
\hat{X}_{T+h|T} = \hat{\phi}_1 \hat{X}_{T+h-1|T} + \ldots + \hat{\phi}_p \hat{X}_{T+h-p|T} + \hat{\theta}_1 \hat{\varepsilon}_{T+h-1|T} + \ldots + \hat{\theta}_q \hat{\varepsilon}_{T+h-q|T},
$$

with the convention \(\hat{X}_{s|T} = X_s\) for \(s \le T\) and \(\hat{\varepsilon}_{s|T} = 0\) for \(s > T\) (future innovations are unpredictable, expectation zero) and \(\hat{\varepsilon}_{s|T} = \hat{\varepsilon}_s\) for \(s \le T\).

### Forecast Error Variance

The \(h\)-step-ahead forecast error is

$$
e_{T+h|T} = X_{T+h} - \hat{X}_{T+h|T} = \sum_{j=0}^{h-1} \psi_j \varepsilon_{T+h-j},
$$

with variance

$$
\mathrm{Var}(e_{T+h|T}) = \sigma^2 \sum_{j=0}^{h-1} \psi_j^2.
$$

This variance **grows with \(h\)**. It converges to \(\mathrm{Var}(X_t) = \sigma^2 \sum_{j=0}^\infty \psi_j^2\) for stationary ARMA, meaning uncertainty grows until it reaches the unconditional variance, then plateaus. For ARIMA with \(d \ge 1\), the \(\psi_j\) do not decay and the variance grows without bound — *the point forecast becomes uninformative at long horizons*. This is the mathematical reason you cannot forecast a random walk 100 steps ahead with useful precision.

### Prediction Intervals

Under Gaussian innovations, the \(1 - \alpha\) prediction interval is

$$
\hat{X}_{T+h|T} \pm z_{1-\alpha/2} \sqrt{\hat{\sigma}^2 \sum_{j=0}^{h-1} \hat{\psi}_j^2}.
$$

A textbook 80% interval uses \(z = 1.28\); 95% uses 1.96. These are *nominal* levels. Three sources of miscalibration in practice:

1. **Parameter uncertainty** ignored: true variance is larger because \(\hat{\boldsymbol{\eta}}\) has estimation error. Correction: simulate parameter draws, compute forecast for each, report quantiles.
2. **Non-Gaussian innovations**: heavy tails widen real intervals. Use bootstrap intervals instead of Gaussian.
3. **Model misspecification**: nothing fixes this except trying alternative models and picking the one whose intervals actually cover.

### Bootstrap Prediction Intervals

For robust intervals without assuming Gaussian innovations:

1. Fit ARIMA, obtain residuals \(\lbrace \hat{\varepsilon}_t\rbrace\).
2. Sample residuals with replacement to build a bootstrap innovation sequence.
3. Feed through the fitted recursion to simulate \(X_{T+1}^*, \ldots, X_{T+h}^*\).
4. Repeat \(B\) times; quantiles of \(\lbrace X_{T+h}^{*(b)}\rbrace\) give a bootstrap interval.

This is the `simulate` + quantile pattern, and it is the correct default for finance, where returns have fat tails.

<svg viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg">
  <rect width="720" height="300" fill="#1a1a1a"/>
  <text x="360" y="22" fill="#e8e8e8" font-size="15" text-anchor="middle" font-family="Georgia, serif" font-weight="bold">Fan Chart — Forecast Uncertainty Grows With Horizon</text>
  <!-- Axes -->
  <line x1="50" y1="260" x2="690" y2="260" stroke="#888" stroke-width="1"/>
  <line x1="50" y1="260" x2="50" y2="40" stroke="#888" stroke-width="1"/>
  <!-- In-sample -->
  <polyline points="50,200 80,190 110,210 140,195 170,205 200,185 230,200 260,180 290,195 320,175 350,185 380,170 410,180 440,175" fill="none" stroke="#6db3f2" stroke-width="1.8"/>
  <line x1="440" y1="260" x2="440" y2="40" stroke="#555" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="440" y="275" fill="#888" font-size="11" text-anchor="middle" font-family="Georgia, serif">T</text>
  <!-- Fans -->
  <polygon points="440,175 500,160 560,140 620,125 680,110 680,240 620,230 560,215 500,195 440,175" fill="#6db3f2" fill-opacity="0.15"/>
  <polygon points="440,175 500,168 560,160 620,148 680,135 680,218 620,206 560,195 500,183 440,175" fill="#6db3f2" fill-opacity="0.3"/>
  <polyline points="440,175 500,170 560,165 620,158 680,152" fill="none" stroke="#6db3f2" stroke-width="2" stroke-dasharray="4,3"/>
  <text x="690" y="150" fill="#6db3f2" font-size="11" font-family="Georgia, serif">point</text>
  <text x="690" y="132" fill="#6db3f2" font-size="11" font-family="Georgia, serif" fill-opacity="0.8">80%</text>
  <text x="690" y="112" fill="#6db3f2" font-size="11" font-family="Georgia, serif" fill-opacity="0.6">95%</text>
  <text x="370" y="280" fill="#d4d4d4" font-size="13" text-anchor="middle" font-family="Georgia, serif">Time</text>
</svg>

---

## 9. The Box-Jenkins Methodology

Box and Jenkins (1970) proposed a three-stage iterative cycle that remains the dominant workflow for ARIMA fitting.

### Stage 1: Identification

1. **Plot the series**. Look for trend, seasonality, variance changes, outliers.
2. **Stabilize variance** if needed (log, Box-Cox).
3. **Choose \(d\)**. Apply differencing until the series looks stationary; confirm with ADF/KPSS.
4. **Examine ACF/PACF of the differenced series**. Use the table in Section 3 to propose candidate orders \(p\) and \(q\).
5. For seasonal data, examine the ACF/PACF at multiples of \(s\) to choose \(P\), \(Q\) and \(D\).

### Stage 2: Estimation

Fit candidate models by MLE. Check parameter estimates are well inside stationary/invertible regions. Fit a few alternative orders near your initial choice; compare by AICc.

### Stage 3: Diagnosis

After fitting, examine residuals:

- **ACF/PACF of residuals**: should look like white noise.
- **Ljung-Box test** on residuals at lags 10–20 (adjusted for \(p + q\) parameters). If \(Q\) is significant, the model is inadequate.
- **Residual plot**: any systematic pattern (trend, heteroskedasticity, regime changes) indicates misspecification.
- **Residual histogram and Q-Q plot**: check for approximate normality. Heavy tails indicate you should use bootstrap intervals or a different innovation distribution.
- **Check over-fitting**: fit a larger model; if coefficients of additional lags are insignificant, stick with the smaller.

If diagnostics fail, return to Stage 1 with a revised identification. Iterate until a model passes.

### auto.arima

Hyndman's `auto.arima` and the `pmdarima` Python port automate the search:

1. Determine \(d\) and \(D\) via KPSS/seasonal strength tests.
2. Grid-search \(\left(p, q, P, Q\right)\) up to specified maxima, using stepwise search to control combinatorial explosion.
3. Return the AICc-minimizing model.

It is fast, reasonable, and wrong about 10–20% of the time. Use it as a starting point and always run residual diagnostics yourself. Do not ship a model just because `auto.arima` picked it.

---

## 10. ARIMAX and Regression with ARIMA Errors

Real forecasting problems almost always have covariates: price, promotions, weather, holidays, day-of-week dummies. Two equivalent-looking but subtly different models:

### ARIMAX

$$
\phi(L)(1 - L)^d X_t = \boldsymbol{\beta}^\top \mathbf{z}_t + \theta(L) \varepsilon_t,
$$

where \(\mathbf{z}_t\) are covariates. Equivalently, regress differenced \(X\) on differenced \(z\) with ARMA errors. **Problem:** the covariates enter inside the \(\phi(L)\) operator, so their coefficients \(\boldsymbol{\beta}\) are not easily interpretable as partial effects.

### Regression with ARIMA Errors (Preferred)

$$
X_t = \boldsymbol{\beta}^\top \mathbf{z}_t + \eta_t, \qquad \phi(L)(1 - L)^d \eta_t = \theta(L) \varepsilon_t.
$$

The covariates explain a deterministic part; the residual follows ARIMA. Now \(\boldsymbol{\beta}\) has the clean interpretation "effect of \(z\) on \(X\), with the autocorrelation structure absorbed in \(\eta\)."

Both are identifiable and can be fit via MLE in state-space form. `statsmodels.tsa.SARIMAX` implements the latter when you pass `exog=` and set `trend='n'`.

### Practical Use of Exogenous Regressors

Typical operational regressors:

- **Calendar**: day-of-week, month, holiday indicators. Use one-hot encoded dummies.
- **Fourier terms**: for multiple seasonalities, \(\sin(2\pi k t / s), \cos(2\pi k t / s)\) for \(k = 1, \ldots, K\) where \(K\) controls smoothness.
- **Promotions / interventions**: binary indicators with lag/lead effects.
- **Temperature, competitor price**: continuous regressors in many applied problems.

Warning: exogenous regressors must be *available at forecast time*. If you use weather as a regressor, you need forecasted weather for the forecast horizon — otherwise you are cheating on your backtest.

---

## 11. Python: Full Forecasting Pipeline

We will simulate a monthly series with trend + seasonality + noise, fit an auto-ARIMA, produce forecasts with intervals, and run diagnostics. Real code, real plots.

### Data Simulation and Initial Inspection

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

rng = np.random.default_rng(7)
n = 144  # 12 years of monthly data
t = np.arange(n)
trend = 0.3 * t + 50
season = 8.0 * np.sin(2*np.pi*t/12) + 4.0*np.cos(2*np.pi*t/6)
noise = rng.normal(0, 2.0, n)
# AR(1) component in the noise to make it ARMA-realistic
ar_noise = np.zeros(n)
for i in range(1, n):
    ar_noise[i] = 0.6 * ar_noise[i-1] + rng.normal(0, 1.5)
y = trend + season + ar_noise
idx = pd.date_range('2014-01', periods=n, freq='MS')
series = pd.Series(y, index=idx, name='Demand')

# Train / test split
train = series.iloc[:-24]
test = series.iloc[-24:]
```

### Identification

```python
fig, axes = plt.subplots(3, 2, figsize=(13, 9))
axes[0,0].plot(train, color='#6db3f2'); axes[0,0].set_title(r'Training series $X_t$')
plot_acf(train, lags=40, ax=axes[0,1]); axes[0,1].set_title('ACF of levels')

d1 = train.diff().dropna()
axes[1,0].plot(d1, color='#a4d08a'); axes[1,0].set_title(r'$\Delta X_t$ (first difference)')
plot_acf(d1, lags=40, ax=axes[1,1]); axes[1,1].set_title(r'ACF of $\Delta X_t$')

d12 = d1.diff(12).dropna()
axes[2,0].plot(d12, color='#f2c894'); axes[2,0].set_title(r'$\Delta_{12}\Delta X_t$ (seasonal differenced)')
plot_acf(d12, lags=40, ax=axes[2,1]); axes[2,1].set_title(r'ACF of $\Delta_{12}\Delta X_t$')
for ax in axes.flat: ax.grid(alpha=0.3)
fig.tight_layout()

# Stationarity tests
for name, s in [('levels', train), ('first diff', d1), ('both diffs', d12)]:
    adf_p = adfuller(s, autolag='AIC')[1]
    kpss_p = kpss(s, regression='c', nlags='auto')[1]
    print(f"{name:12s}  ADF p={adf_p:.3f}   KPSS p={kpss_p:.3f}")
```

Expected: ADF fails to reject on levels (\(p\) high), KPSS rejects (\(p\) low) — non-stationary. After \(\Delta\), still rejects KPSS at the seasonal frequency. After \(\Delta \Delta_{12}\), stationary.

### Fit SARIMA

```python
# Start with SARIMA(1,1,1)(1,1,1)_12 — a sensible default for monthly data
model = SARIMAX(train,
                order=(1,1,1),
                seasonal_order=(1,1,1,12),
                enforce_stationarity=True,
                enforce_invertibility=True)
res = model.fit(disp=False)
print(res.summary())
```

Reading the output: check that all AR/MA coefficients have \(|z|\)-statistics > 2 (statistically significant), that the Ljung-Box p-value on residuals is > 0.05, and that AIC/BIC are lower than simpler alternatives.

### Model Selection by Grid Search (AICc)

```python
from itertools import product

best_aicc = np.inf
best_order = None
for (p, q, P, Q) in product(range(3), range(3), range(2), range(2)):
    try:
        m = SARIMAX(train, order=(p,1,q), seasonal_order=(P,1,Q,12),
                    enforce_stationarity=True, enforce_invertibility=True)
        r = m.fit(disp=False, method='lbfgs', maxiter=200)
        k = p + q + P + Q + 1  # variance
        aicc = r.aic + 2*k*(k+1) / max(len(train) - k - 1, 1)
        if aicc < best_aicc:
            best_aicc = aicc
            best_order = (p, q, P, Q)
            best_res = r
    except Exception:
        continue

print(f"Best: (p,q,P,Q) = {best_order}, AICc = {best_aicc:.1f}")
```

### Residual Diagnostics

```python
resid = best_res.resid.iloc[best_res.loglikelihood_burn:]

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
axes[0,0].plot(resid, color='#6db3f2', lw=0.8); axes[0,0].axhline(0, color='#555')
axes[0,0].set_title('Residuals vs. time')
plot_acf(resid, lags=40, ax=axes[0,1]); axes[0,1].set_title('Residual ACF')

axes[1,0].hist(resid, bins=30, color='#a4d08a', density=True, alpha=0.8)
x_pdf = np.linspace(resid.min(), resid.max(), 200)
axes[1,0].plot(x_pdf, stats.norm.pdf(x_pdf, resid.mean(), resid.std()), 'r--')
axes[1,0].set_title('Residual histogram vs. normal')

stats.probplot(resid, dist='norm', plot=axes[1,1])
axes[1,1].set_title('Q-Q plot')
fig.tight_layout()

# Ljung-Box
lb = acorr_ljungbox(resid, lags=[10, 20, 30], return_df=True)
print(lb)
```

A well-specified model should give:

- Residuals hovering around zero with no pattern.
- ACF spikes all inside the blue bands.
- Histogram roughly normal.
- Q-Q plot nearly linear.
- Ljung-Box p-values all > 0.05.

### Forecast and Intervals

```python
fcst = best_res.get_forecast(steps=24)
mean = fcst.predicted_mean
ci80 = fcst.conf_int(alpha=0.20)
ci95 = fcst.conf_int(alpha=0.05)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(train, color='#6db3f2', label='Train')
ax.plot(test, color='#a4d08a', label='Test (actual)')
ax.plot(mean, color='#f2a5a5', label=r'$\hat{X}_{T+h|T}$ (point forecast)')
ax.fill_between(mean.index, ci95.iloc[:,0], ci95.iloc[:,1], color='#f2a5a5', alpha=0.15, label='95% PI')
ax.fill_between(mean.index, ci80.iloc[:,0], ci80.iloc[:,1], color='#f2a5a5', alpha=0.30, label='80% PI')
ax.set_ylabel(r'$X_t$')
ax.set_xlabel('Date')
ax.set_title(r'SARIMA$(p,1,q)(P,1,Q)_{12}$ Forecast with Prediction Intervals')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
fig.tight_layout()

# Accuracy metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(test, mean)
rmse = np.sqrt(mean_squared_error(test, mean))
# MASE = MAE / MAE of seasonal naive on train
naive = train.shift(12).dropna()
naive_mae = mean_absolute_error(train.iloc[12:], naive)
mase = mae / naive_mae
print(f"MAE = {mae:.2f}   RMSE = {rmse:.2f}   MASE = {mase:.3f}")
print(f"Coverage at 80%: {((test.values >= ci80.iloc[:,0].values) & (test.values <= ci80.iloc[:,1].values)).mean():.1%}")
print(f"Coverage at 95%: {((test.values >= ci95.iloc[:,0].values) & (test.values <= ci95.iloc[:,1].values)).mean():.1%}")
```

MASE < 1 ⇒ we beat seasonal naive. Coverage close to nominal ⇒ calibration is OK.

### Bootstrap Prediction Intervals

```python
def bootstrap_forecast(res, steps, n_boot=500, rng=None):
    if rng is None: rng = np.random.default_rng()
    resid = res.resid.iloc[res.loglikelihood_burn:].values
    sims = np.empty((n_boot, steps))
    for b in range(n_boot):
        # Resample innovations from residuals
        eps = rng.choice(resid, size=steps, replace=True)
        sims[b] = res.simulate(steps, anchor='end', repetitions=1,
                               measurement_shocks=eps.reshape(-1, 1)).flatten()
    return sims

sims = bootstrap_forecast(best_res, steps=24, n_boot=2000)
p_lo = np.percentile(sims, 2.5, axis=0)
p_hi = np.percentile(sims, 97.5, axis=0)
```

### Rolling Origin Cross-Validation

```python
def rolling_cv(series, order, seasonal_order, horizon=12, min_train=60):
    errors = []
    for end in range(min_train, len(series) - horizon):
        tr = series.iloc[:end]
        te = series.iloc[end:end+horizon]
        try:
            r = SARIMAX(tr, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=True, enforce_invertibility=True
                       ).fit(disp=False, method='lbfgs')
            f = r.forecast(horizon)
            errors.append(te.values - f.values)
        except Exception:
            continue
    errors = np.array(errors)
    mae_by_h = np.mean(np.abs(errors), axis=0)
    return mae_by_h

mae_h = rolling_cv(series, order=(best_order[0], 1, best_order[1]),
                   seasonal_order=(best_order[2], 1, best_order[3], 12),
                   horizon=12)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(np.arange(1, 13), mae_h, marker='o', color='#6db3f2')
ax.set_xlabel(r'Forecast horizon $h$')
ax.set_ylabel(r'Mean absolute error')
ax.set_title('MAE grows with forecast horizon (rolling CV)')
ax.grid(alpha=0.3)
```

MAE grows with horizon — the fan chart is not lying. This is the real quantity you should report to stakeholders.

---

## 12. Industry Best Practices

### 12.1 Always Fit a Benchmark

Fit seasonal naive, simple exponential smoothing, and auto-ARIMA at minimum. Report MASE against seasonal naive. If your fancy ARIMA has MASE \(\ge\) 1, revisit the model — you are not beating the benchmark. Large MASE variance across items in a panel indicates some items are structurally easier than others and deserves investigation.

### 12.2 Use Rolling-Origin Evaluation

Static train/test split is a *single* realization of the evaluation noise. Rolling origin with step size 1–\(s\) gives you the distribution of errors across many test points. Report median, 80th percentile, and worst-case of the error distribution — not just the mean. Stakeholders care about tail performance, not average.

### 12.3 Fit At the Cadence You Will Forecast

If you run the model daily and forecast 14 days ahead, your CV should mirror that. Do not validate on one long test window when production will use daily re-estimation — the error characteristics are different because the model sees fresh data each day.

### 12.4 Prefer Regression with ARIMA Errors for Interpretable Covariates

ARIMAX confounds covariate effects with autoregressive dynamics. Regression-with-ARIMA-errors keeps the covariate coefficients interpretable and gives cleaner intervention analysis.

### 12.5 Do Not Over-Difference

Each additional differencing operation injects a unit root on the MA side if the series did not actually need it. A bloated model fits the training set but forecasts poorly because the inflated \(\psi_j\) weights amplify innovation noise. Test the necessity of \(d\) and \(D\) with KPSS.

### 12.6 Watch Out for Seasonal Unit Roots

Seasonal differencing \(\left(1 - L^s\right)\) is much less robust to misspecification than ordinary differencing. Applied when not needed, it absorbs the intercept and produces odd trending behavior. Use the **OCSB** or **HEGY** seasonal unit-root tests if you are unsure. Most software defaults (including `auto.arima`) apply seasonal differencing aggressively, often more than is warranted.

### 12.7 Cap Maximum Order Search

Grid-searching over \(\left(p, q, P, Q\right) \in [0, 5]^4\) on short data almost always selects an overparameterized model with spurious estimates. Cap \(p, q \le 3\) and \(P, Q \le 2\) for monthly data unless you have strong domain evidence for longer memory.

### 12.8 Refit on a Schedule

ARIMA parameters drift. A model fit once and deployed forever will have degrading MASE. Refit monthly for monthly data, weekly for daily data. Automate the refit, monitor parameter stability, alert on large jumps — these can indicate regime changes.

### 12.9 Report Intervals, Not Just Points

A point forecast is a distribution summary that hides the uncertainty you care about. Retailers make ordering decisions on the 95th percentile, not the mean. Service level agreements are written against a tail quantile. Always provide \(P(X_{T+h} \le x)\) or a wide interval, and monitor coverage in production. Under-coverage is a five-alarm fire in any forecasting system.

### 12.10 Log-Transform for Positive Multiplicative Series

Revenue, demand, page views — all are multiplicatively seasonal, strictly positive, and heteroskedastic. Fit ARIMA to \(\log X_t\); back-transform at forecast time, correcting the Jensen gap:

$$
\hat{X}_{T+h|T}^{\text{back}} \approx \exp(\hat{Y}_{T+h|T}) \cdot \exp(\hat{\sigma}_h^2 / 2),
$$

where \(\hat{\sigma}_h^2\) is the variance of the log-scale forecast. Without this correction your point forecast is systematically biased low by 1–5% at long horizons.

### 12.11 Do Not Trust auto.arima Blindly

`auto.arima` is a great baseline, not a decision. Review the selected model: are the orders sensible given your data? Do the coefficients pass significance? Does the residual ACF look white? Has it selected a near-non-invertible MA order because of over-differencing? Log the selected \(\left(p, d, q, P, D, Q\right)\) over time — if it keeps changing, your series is not stationary in a deeper sense and ARIMA is the wrong tool.

### 12.12 For Many Related Series, Batch Carefully

Most operational forecasting problems involve thousands of SKUs, zones, or sensors — each gets a SARIMA fit. Common failures at scale:

- **Cold items** with too few observations: fall back to a pooled or hierarchical model (Part 5).
- **Intermittent demand** (many zeros): ARIMA is a bad fit. Use Croston's method or TSB, or reframe as a count model.
- **Parameter instability** across items: accept it. Aggressive per-item optimization overfits. Consider a shared seasonal structure with item-specific intercepts instead.

### 12.13 Document the Preprocessing

The forecast is the output of a pipeline: clean → transform → difference → model → forecast → back-transform. Every step must be reversible and reproducible. Keep a single preprocessing module that both training and inference call. The most common production bug is a subtle divergence between training-time and inference-time transformation.

---

## Summary and What's Next

ARIMA is the finite-parameter realization of Wold's MA(\(\infty\)) theorem: a rational lag polynomial \(\theta(L)/\phi(L)\) approximating the innovation filter of any stationary process. We added differencing to handle unit roots, seasonal blocks for periodic structure, exogenous regressors for covariates, and MLE + AICc for estimation + selection. The forecasting formulas are recursive; prediction intervals are built from the \(\psi_j\) weights; and the whole thing is held together by the Box-Jenkins identify-estimate-diagnose loop.

What ARIMA is good at: short-to-medium horizon forecasts on series with modest nonlinearity and clean seasonality; calibrated prediction intervals under Gaussianity; interpretable coefficients; cheap to fit.

What ARIMA struggles with: very short series (< 50 observations), intermittent demand, multi-seasonal patterns, strong nonlinearity, abrupt regime changes, and very long horizons. For those you reach for other tools in the forecasting toolbox.

In [Part 3](/2026/04/20/exponential-smoothing-ets-theta.html) we cover exponential smoothing and the state-space ETS family — Holt's method, Holt-Winters, damped trend, and Theta. These models win on shorter series and are the workhorses of operational forecasting. In [Part 4](/2026/04/21/state-space-kalman-filtering.html) we unify ARIMA and ETS under a common state-space framework and introduce the Kalman filter as the exact-MLE engine. [Part 5](/2026/04/22/modern-forecasting-garch-gbm-nbeats-hierarchical.html) covers modern forecasting: GARCH, gradient boosting with lag features, N-BEATS (deep MLPs without attention), hierarchical reconciliation, and calibrated probabilistic forecasting.
