---
layout: post
title: "State-Space Models and the Kalman Filter: The Unifying Framework for Forecasting"
date: 2026-04-21
category: math
---

*This is Part 4 of a 5-part series on time series forecasting. [Part 1: Foundations](/2026/04/18/time-series-foundations-stationarity.html) | [Part 2: ARIMA](/2026/04/19/arima-box-jenkins-forecasting.html) | [Part 3: Exponential Smoothing & ETS](/2026/04/20/exponential-smoothing-ets-theta.html) | **Part 4: State-Space & Kalman** | [Part 5: Modern Forecasting](/2026/04/22/modern-forecasting-garch-gbm-nbeats-hierarchical.html)*

Every forecasting model in Parts 2 and 3 — AR, MA, ARIMA, SARIMA, SES, Holt, Holt-Winters, ETS — is secretly the same kind of object wearing different clothes. The object is a **linear state-space model**, and the algorithm that unites them is the **Kalman filter**. This is not an accident and it is not a retrofit. The state-space formulation is the correct mathematical container for forecasting because it separates *what you believe about the world* (the latent state: level, trend, seasonal factor, local deviation) from *what you observe* (the noisy series) and provides an exact, recursive, \\(O(T)\\) algorithm for updating the former given the latter.

Once you have the Kalman filter, many things that were previously ad hoc become principled. Exact maximum likelihood for ARIMA reduces to running the filter and summing the one-step prediction errors. Exact MLE for ETS likewise. Structural time series models — where you decompose a series into explicitly-interpretable level + slope + seasonal + cycle components — are a state-space formulation that can actually identify and forecast each component separately. Missing data becomes a filter that skips a step, not an imputation hack. Exogenous regressors with time-varying effects become states that evolve and are updated. Multivariate forecasting, where several related series share latent dynamics, is a direct extension — and delivers some of the largest accuracy gains in practice.

This post builds the state-space framework from scratch. We derive the Kalman filter step by step, prove its optimality, add the backward smoother, show how to estimate parameters via the likelihood it produces, apply it to structural time series (the trend + seasonal + cycle decomposition), handle missing data, extend to regression with time-varying coefficients, and sketch the non-linear cases (extended / unscented Kalman filters, particle filters). Python implementations throughout. We close with industry best practices for shipping state-space models in production.

---

## Table of Contents

1. [The Linear Gaussian State-Space Model](#1-the-linear-gaussian-state-space-model)
2. [The Forecasting Problem in State-Space Form](#2-the-forecasting-problem-in-state-space-form)
3. [Deriving the Kalman Filter](#3-deriving-the-kalman-filter)
4. [The Kalman Smoother (RTS)](#4-the-kalman-smoother-rts)
5. [Likelihood and Parameter Estimation](#5-likelihood-and-parameter-estimation)
6. [Structural Time Series Models](#6-structural-time-series-models)
7. [ARIMA and ETS in State-Space Form](#7-arima-and-ets-in-state-space-form)
8. [Missing Data and Outliers](#8-missing-data-and-outliers)
9. [Extensions: Non-Linear and Non-Gaussian](#9-extensions-non-linear-and-non-gaussian)
10. [Python: Structural Forecasting Pipeline](#10-python-structural-forecasting-pipeline)
11. [Industry Best Practices](#11-industry-best-practices)

---

## 1. The Linear Gaussian State-Space Model

A linear Gaussian state-space model has two equations:

$$
\mathbf{x}_t = \mathbf{F}_t \mathbf{x}_{t-1} + \mathbf{G}_t \mathbf{w}_t, \qquad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_t), \tag{state}
$$

$$
y_t = \mathbf{H}_t \mathbf{x}_t + v_t, \qquad v_t \sim \mathcal{N}(0, R_t). \tag{observation}
$$

Names and meanings:

- \\(\mathbf{x}_t \in \mathbb{R}^n\\) is the **state** at time \\(t\\) — the thing we want to estimate. Typical contents: level, trend, seasonal factors, regression coefficients, latent AR lags.
- \\(y_t \in \mathbb{R}\\) (or \\(\mathbb{R}^p\\) for multivariate) is the **observation** — the actual data point we record.
- \\(\mathbf{F}_t\\) is the **state transition matrix** — how the state evolves deterministically.
- \\(\mathbf{G}_t\\) selects / scales which state components get stochastic shocks.
- \\(\mathbf{w}_t\\) is the **state (process) noise** with covariance \\(\mathbf{Q}_t\\).
- \\(\mathbf{H}_t\\) is the **observation matrix** — maps state to what we measure.
- \\(v_t\\) is the **measurement noise** with variance \\(R_t\\).

An initial distribution \\(\mathbf{x}_0 \sim \mathcal{N}(\mathbf{m}_0, \mathbf{P}_0)\\) closes the specification.

### Why Separate State from Observation

A time series is almost always a noisy proxy for a smooth underlying process. Temperature readings have instrument noise; sales figures get reclassified; economic indicators get revised. Treating the observation as identical to the quantity of interest is how you get brittle models that overreact to measurement errors. State-space models formalize the distinction: \\(y_t\\) is what you see; \\(\mathbf{x}_t\\) is what you want to know; \\(\mathbf{H}_t\\) is the lens between them.

### The Two Noise Sources

Most of the confusion around state-space models comes from mixing up \\(\mathbf{w}_t\\) and \\(v_t\\):

- \\(\mathbf{w}_t\\): **process** noise. Changes the true state. "The actual demand level drifted up this week."
- \\(v_t\\): **measurement** noise. Changes only the observation. "We mis-recorded sales by 3%."

Their covariances \\(\mathbf{Q}\\) and \\(R\\) together determine how much of an observed jump is "real" (update state) versus "noise" (ignore). The Kalman filter does this decomposition optimally.

---

## 2. The Forecasting Problem in State-Space Form

Given observations \\(y_{1:t} = (y_1, \ldots, y_t)\\), we want three things:

1. **Filtering**: the posterior \\(p(\mathbf{x}_t \mid y_{1:t})\\) — current state given current data. Updates as \\(t\\) advances, producing \\(\hat{\mathbf{m}}_{t|t}, \hat{\mathbf{P}}_{t|t}\\).
2. **Prediction**: the predictive distribution \\(p(\mathbf{x}_{t+h} \mid y_{1:t})\\) and hence \\(p(y_{t+h} \mid y_{1:t})\\). This is the forecasting target.
3. **Smoothing**: the posterior \\(p(\mathbf{x}_t \mid y_{1:T})\\) using the *full* data including future observations. Used for retrospective analysis, component decomposition, and imputation.

In the linear Gaussian case all three are Gaussian and have closed-form mean-covariance recursions. The Kalman filter is the filtering recursion; the RTS smoother is the smoothing recursion. Prediction is just running the state equation forward without updating.

---

## 3. Deriving the Kalman Filter

Assume the linear Gaussian model of Section 1 and that the filtering distribution at time \\(t - 1\\) is

$$
\mathbf{x}_{t-1} \mid y_{1:t-1} \sim \mathcal{N}(\hat{\mathbf{m}}_{t-1|t-1}, \hat{\mathbf{P}}_{t-1|t-1}).
$$

We want the filtering distribution at time \\(t\\) after observing \\(y_t\\). The derivation has two steps — predict and update.

### Step 1: Predict (Time Update)

Before seeing \\(y_t\\), propagate the state through the state equation:

$$
\mathbf{x}_t = \mathbf{F}_t \mathbf{x}_{t-1} + \mathbf{G}_t \mathbf{w}_t.
$$

Since \\(\mathbf{x}_{t-1}\\) is Gaussian and the noise is independent Gaussian, so is \\(\mathbf{x}_t \mid y_{1:t-1}\\). Taking expectation:

$$
\hat{\mathbf{m}}_{t|t-1} = \mathbf{F}_t \hat{\mathbf{m}}_{t-1|t-1},
$$

$$
\hat{\mathbf{P}}_{t|t-1} = \mathbf{F}_t \hat{\mathbf{P}}_{t-1|t-1} \mathbf{F}_t^\top + \mathbf{G}_t \mathbf{Q}_t \mathbf{G}_t^\top.
$$

The mean advances by the deterministic part of the state transition; the covariance grows by both the propagated prior uncertainty and the new process noise.

### Step 2: Predict the Observation

Given the predicted state, predict the observation:

$$
\hat{y}_{t|t-1} = \mathbf{H}_t \hat{\mathbf{m}}_{t|t-1},
$$

$$
S_t = \mathbf{H}_t \hat{\mathbf{P}}_{t|t-1} \mathbf{H}_t^\top + R_t.
$$

The quantity \\(S_t\\) is the **innovation variance** — how uncertain we are about \\(y_t\\) before seeing it.

### Step 3: Update (Measurement Update)

Compute the **innovation** (aka prediction error):

$$
e_t = y_t - \hat{y}_{t|t-1}.
$$

This is the information content of the new observation. The **Kalman gain**

$$
\mathbf{K}_t = \hat{\mathbf{P}}_{t|t-1} \mathbf{H}_t^\top S_t^{-1}
$$

is the optimal weight on this information. The updated (posterior) state distribution is

$$
\hat{\mathbf{m}}_{t|t} = \hat{\mathbf{m}}_{t|t-1} + \mathbf{K}_t e_t,
$$

$$
\hat{\mathbf{P}}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t) \hat{\mathbf{P}}_{t|t-1}.
$$

### Sketch of Why This Is Optimal

The derivation is joint Gaussian conditioning. The predicted state and the observation, jointly, are

$$
\begin{pmatrix} \mathbf{x}_t \\ y_t \end{pmatrix} \mid y_{1:t-1} \sim \mathcal{N}\left( \begin{pmatrix} \hat{\mathbf{m}}_{t|t-1} \\ \hat{y}_{t|t-1} \end{pmatrix}, \begin{pmatrix} \hat{\mathbf{P}}_{t|t-1} & \hat{\mathbf{P}}_{t|t-1} \mathbf{H}_t^\top \\ \mathbf{H}_t \hat{\mathbf{P}}_{t|t-1} & S_t \end{pmatrix} \right).
$$

By the standard Gaussian conditioning formula, the posterior \\(\mathbf{x}_t \mid y_{1:t}\\) has mean and covariance exactly as given above. The Kalman gain \\(\mathbf{K}_t\\) is the regression coefficient of \\(\mathbf{x}_t\\) on \\(y_t\\) — it is both the minimum-MSE linear estimator and (under Gaussianity) the conditional expectation.

### Interpretation of the Kalman Gain

Expand \\(\mathbf{K}_t\\):

$$
\mathbf{K}_t = \frac{\hat{\mathbf{P}}_{t|t-1} \mathbf{H}_t^\top}{\mathbf{H}_t \hat{\mathbf{P}}_{t|t-1} \mathbf{H}_t^\top + R_t}.
$$

- If \\(R_t\\) (measurement noise) is large relative to the state uncertainty, \\(\mathbf{K}_t\\) is small — trust the prior, nearly ignore the new observation.
- If \\(R_t\\) is small relative to state uncertainty, \\(\mathbf{K}_t\\) is near \\((\mathbf{H}_t \hat{\mathbf{P}}_{t|t-1} \mathbf{H}_t^\top)^{-1} \hat{\mathbf{P}}_{t|t-1} \mathbf{H}_t^\top\\) — trust the new observation, nearly replace the prior.

The filter continuously trades off prior state estimate against new measurements, weighted by their relative precisions. When measurements are very precise, the filter tracks them closely; when they're noisy, the filter smooths. All without hyperparameter tuning beyond \\(\mathbf{Q}\\) and \\(R\\).

### The Joseph Form

For numerical stability, the covariance update should use the **Joseph form**:

$$
\hat{\mathbf{P}}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t) \hat{\mathbf{P}}_{t|t-1} (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t)^\top + \mathbf{K}_t R_t \mathbf{K}_t^\top.
$$

This is mathematically equivalent but preserves positive semi-definiteness under floating-point arithmetic. Production Kalman filter implementations always use the Joseph form — it's the difference between a stable filter and one that silently drifts into negative variances after a few thousand steps.

---

## 4. The Kalman Smoother (RTS)

The filter uses only past and present data. For retrospective analysis — component decomposition, outlier detection, historical state reconstruction — we want \\(p(\mathbf{x}_t \mid y_{1:T})\\) using all data. The **Rauch-Tung-Striebel (RTS) smoother** computes this with a backward pass.

### Algorithm

Run the Kalman filter forward, store \\(\hat{\mathbf{m}}_{t|t}, \hat{\mathbf{P}}_{t|t}, \hat{\mathbf{m}}_{t+1|t}, \hat{\mathbf{P}}_{t+1|t}\\) for all \\(t\\). Then run backward: initialize \\(\hat{\mathbf{m}}^s_T = \hat{\mathbf{m}}_{T|T}\\), \\(\hat{\mathbf{P}}^s_T = \hat{\mathbf{P}}_{T|T}\\), and for \\(t = T-1, T-2, \ldots, 0\\):

$$
\mathbf{C}_t = \hat{\mathbf{P}}_{t|t} \mathbf{F}_{t+1}^\top \hat{\mathbf{P}}_{t+1|t}^{-1},
$$

$$
\hat{\mathbf{m}}^s_t = \hat{\mathbf{m}}_{t|t} + \mathbf{C}_t (\hat{\mathbf{m}}^s_{t+1} - \hat{\mathbf{m}}_{t+1|t}),
$$

$$
\hat{\mathbf{P}}^s_t = \hat{\mathbf{P}}_{t|t} + \mathbf{C}_t (\hat{\mathbf{P}}^s_{t+1} - \hat{\mathbf{P}}_{t+1|t}) \mathbf{C}_t^\top.
$$

Here \\(\mathbf{C}_t\\) is the **smoother gain** — analogous to the Kalman gain but pushing information backward in time.

### Why Smooth?

- **Component decomposition**: for a structural model, \\(\hat{\mathbf{m}}^s_t\\) gives the best estimate of each latent component (level, trend, seasonal) at every historical \\(t\\). Essential for interpretation.
- **Outlier detection**: compare the smoothed observation \\(\mathbf{H}_t \hat{\mathbf{m}}^s_t\\) to \\(y_t\\); large discrepancies identify outliers that couldn't have been noise *after seeing the full data*.
- **Missing data imputation**: for missing \\(y_t\\), the smoothed mean is the best estimate given everything else.
- **Model diagnostics**: look at the smoothed innovations and their pattern; structural misspecification shows up clearly.

---

## 5. Likelihood and Parameter Estimation

The Kalman filter gives, as a byproduct, the **prediction error decomposition** of the log-likelihood. For the model with unknown parameters \\(\boldsymbol{\theta}\\) (entries of \\(\mathbf{F}, \mathbf{G}, \mathbf{H}, \mathbf{Q}, R\\), initial state):

$$
\log p(y_{1:T} \mid \boldsymbol{\theta}) = \sum_{t=1}^T \log p(y_t \mid y_{1:t-1}, \boldsymbol{\theta}) = -\frac{1}{2} \sum_{t=1}^T \left[\log(2\pi S_t) + \frac{e_t^2}{S_t}\right].
$$

Run the filter with parameters \\(\boldsymbol{\theta}\\), collect \\(e_t\\) and \\(S_t\\), compute the likelihood. Maximize over \\(\boldsymbol{\theta}\\) by any gradient-based optimizer (L-BFGS, Newton). The gradient of the likelihood with respect to \\(\boldsymbol{\theta}\\) is also computed from filter quantities (Harvey, 1989, gives the closed-form gradient; automatic differentiation works too).

### Why This Is a Huge Deal

The computation is \\(O(T n^3)\\) where \\(n\\) is the state dimension — linear in sample size, polynomial in state dimension. For ARIMA this is the only known way to compute the *exact* likelihood efficiently (naive computation is \\(O(T^3)\\)). For ETS, the innovations form of Hyndman et al. is essentially a Kalman filter with a specific, simple state structure. For any compound model (structural + regression + ARIMA errors) the state-space form is the only practical tool for exact inference.

### Initial State Treatment

Three options for \\(\mathbf{x}_0\\):

1. **Known**: \\(\mathbf{m}_0, \mathbf{P}_0\\) fixed. Useful only in engineering contexts where you know initial conditions.
2. **MLE**: include \\(\mathbf{m}_0\\) as parameters in \\(\boldsymbol{\theta}\\). Standard for most econometric applications. `statsmodels.tsa.statespace` does this.
3. **Diffuse**: set \\(\mathbf{P}_0 \to \infty \mathbf{I}\\) on non-stationary state components. The **exact diffuse filter** (Koopman, 1997) handles the infinite prior analytically. This is the correct treatment when the state is initialized from a non-stationary distribution (unit-root processes, initial level, seasonal factors). All modern packages support it.

### EM Algorithm

For very high-dimensional state spaces, direct gradient-based MLE can be slow. The EM algorithm (Shumway and Stoffer, 1982) exploits the state-space structure: the E-step runs the Kalman smoother, the M-step has closed-form updates for \\(\mathbf{F}, \mathbf{Q}, \mathbf{H}, R\\). Converges slowly near the optimum but is robust for pathological starting values.

---

## 6. Structural Time Series Models

The most valuable application of state-space methods for forecasting is the **structural time series** or **unobserved components (UCM)** model. Rather than modeling the series as a flat ARMA, decompose it explicitly:

$$
y_t = \mu_t + \tau_t + \gamma_t + \psi_t + \varepsilon_t,
$$

where \\(\mu_t\\) is a local **level**, \\(\tau_t\\) is a local **trend (slope)**, \\(\gamma_t\\) is a **seasonal** component, \\(\psi_t\\) is a **cycle**, and \\(\varepsilon_t\\) is measurement noise. Each component has its own state equations.

### Local Linear Trend

$$
\mu_t = \mu_{t-1} + \tau_{t-1} + \eta^\mu_t, \qquad \eta^\mu_t \sim \mathcal{N}(0, \sigma^2_\mu),
$$

$$
\tau_t = \tau_{t-1} + \eta^\tau_t, \qquad \eta^\tau_t \sim \mathcal{N}(0, \sigma^2_\tau).
$$

The slope \\(\tau_t\\) walks randomly; the level \\(\mu_t\\) picks up the slope plus its own random walk component. Choosing \\(\sigma^2_\mu = 0\\) (no level shocks, only trend shocks) gives the **smooth trend** model used in demographic forecasting.

### Seasonal Component

Two parameterizations:

**Dummy seasonal**: \\(\gamma_t + \gamma_{t-1} + \ldots + \gamma_{t-m+1} = \eta^\gamma_t\\). Equivalently, \\(\gamma_t = -\sum_{j=1}^{m-1} \gamma_{t-j} + \eta^\gamma_t\\). Forces the seasonal factors to sum (approximately) to zero over any complete period. State vector has \\(m-1\\) seasonal components.

**Trigonometric seasonal**: a sum of \\(\lfloor m/2 \rfloor\\) sinusoidal components at the fundamental frequency and harmonics. Each component is:

$$
\begin{pmatrix} \gamma_{j,t} \\ \gamma^*_{j,t} \end{pmatrix} = \begin{pmatrix} \cos \lambda_j & \sin \lambda_j \\ -\sin \lambda_j & \cos \lambda_j \end{pmatrix} \begin{pmatrix} \gamma_{j,t-1} \\ \gamma^*_{j,t-1} \end{pmatrix} + \begin{pmatrix} \eta^\gamma_{j,t} \\ \eta^{\gamma*}_{j,t} \end{pmatrix},
$$

with frequencies \\(\lambda_j = 2\pi j / m\\). The trigonometric form handles non-integer seasonal periods cleanly (e.g., 365.25 days/year) and is the standard for modern structural time series software (`bsts` in R, `statsmodels.tsa.UnobservedComponents`).

### Cycle Component

A stationary oscillation at a slower frequency than seasonality — useful for modeling business cycles:

$$
\begin{pmatrix} \psi_t \\ \psi^*_t \end{pmatrix} = \rho \begin{pmatrix} \cos \lambda & \sin \lambda \\ -\sin \lambda & \cos \lambda \end{pmatrix} \begin{pmatrix} \psi_{t-1} \\ \psi^*_{t-1} \end{pmatrix} + \begin{pmatrix} \eta^\psi_t \\ \eta^{\psi*}_t \end{pmatrix},
$$

with damping \\(\rho \in (0, 1)\\) and frequency \\(\lambda\\) both estimated. Used in macroeconomics to separate business-cycle fluctuations from trend and seasonal.

### Why Structural Models Matter

The components are **interpretable and separately forecastable**. Your retail forecast isn't a single number — it's "level at \$100M, growing at \$2M/month, Black Friday adds \$15M to November, seasonal factors gradually strengthening 3% year-over-year." That decomposition is the actual output your business wants.

The components also **forecast differently at long horizons**: the trend keeps drifting, the seasonal repeats, the cycle decays. A purely-ARMA forecast homogenizes these and loses information. Structural forecasts stay meaningful at horizons where ARMA prediction intervals have already exploded.

<svg viewBox="0 0 720 380" xmlns="http://www.w3.org/2000/svg">
  <rect width="720" height="380" fill="#1a1a1a"/>
  <text x="360" y="20" fill="#e8e8e8" font-size="15" text-anchor="middle" font-family="Georgia, serif" font-weight="bold">Structural Decomposition: y_t = μ_t + τ_t-drift + γ_t + ε_t</text>
  <!-- Observed -->
  <text x="50" y="55" fill="#e8e8e8" font-size="11" font-family="Georgia, serif">y_t</text>
  <polyline points="80,55 110,70 140,45 170,75 200,50 230,80 260,55 290,85 320,60 350,90 380,65 410,95 440,70 470,100 500,75 530,105 560,80 590,110 620,85 650,115" fill="none" stroke="#6db3f2" stroke-width="1.5"/>
  <!-- Level -->
  <text x="50" y="130" fill="#e8e8e8" font-size="11" font-family="Georgia, serif">μ_t</text>
  <polyline points="80,140 110,137 140,135 170,132 200,128 230,125 260,121 290,117 320,113 350,110 380,107 410,103 440,99 470,95 500,92 530,88 560,84 590,80 620,76 650,72" fill="none" stroke="#a4d08a" stroke-width="2"/>
  <!-- Trend slope (displayed as low-slope line) -->
  <!-- Seasonal -->
  <text x="50" y="215" fill="#e8e8e8" font-size="11" font-family="Georgia, serif">γ_t</text>
  <polyline points="80,215 110,200 140,230 170,205 200,225 230,210 260,220 290,215 320,225 350,205 380,230 410,210 440,220 470,215 500,225 530,205 560,230 590,210 620,220 650,215" fill="none" stroke="#f2c894" stroke-width="1.5"/>
  <line x1="80" y1="215" x2="650" y2="215" stroke="#555" stroke-width="0.8" stroke-dasharray="3,3"/>
  <!-- Residual -->
  <text x="50" y="300" fill="#e8e8e8" font-size="11" font-family="Georgia, serif">ε_t</text>
  <polyline points="80,300 110,303 140,298 170,305 200,297 230,302 260,301 290,299 320,304 350,298 380,302 410,297 440,303 470,299 500,302 530,298 560,304 590,299 620,303 650,300" fill="none" stroke="#f2a5a5" stroke-width="1.2"/>
  <line x1="80" y1="300" x2="650" y2="300" stroke="#555" stroke-width="0.8" stroke-dasharray="3,3"/>
  <text x="360" y="360" fill="#d4d4d4" font-size="11" text-anchor="middle" font-family="Georgia, serif">time</text>
</svg>

---

## 7. ARIMA and ETS in State-Space Form

Every ARMA and ETS model has a linear Gaussian state-space representation. The specific form depends on the parameterization.

### AR(p) in State-Space Form

The state is the last \\(p\\) observations:

$$
\mathbf{x}_t = \begin{pmatrix} X_t \\ X_{t-1} \\ \vdots \\ X_{t-p+1} \end{pmatrix}, \quad \mathbf{F} = \begin{pmatrix} \phi_1 & \phi_2 & \ldots & \phi_p \\ 1 & 0 & \ldots & 0 \\ 0 & 1 & \ldots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & \ldots & 1 & 0 \end{pmatrix}, \quad \mathbf{H} = (1, 0, \ldots, 0).
$$

Process noise \\(\mathbf{w}_t\\) lives only in the first component with variance \\(\sigma^2\\); measurement noise is zero. Running the Kalman filter on this reproduces AR(\\(p\\)) one-step-ahead forecasts.

### ARMA(p, q) — Harvey Form

For ARMA, the state combines partial observations and innovations. Writing \\(r = \max(p, q+1)\\), Harvey's representation uses state

$$
\mathbf{x}_t = \begin{pmatrix} X_t \\ \phi_2 X_{t-1} + \ldots + \phi_p X_{t-p+1} + \theta_1 \varepsilon_t + \ldots + \theta_q \varepsilon_{t-q+1} \\ \vdots \end{pmatrix}
$$

with a specific \\(\mathbf{F}\\) involving both AR and MA coefficients. The details are in Harvey (1989, *Forecasting, Structural Time Series Models and the Kalman Filter*). The takeaway: ARMA fits by MLE through this Kalman filter, and that's what `statsmodels.tsa.arima.ARIMA` does internally.

### ETS in State-Space Form

ETS has its own innovations state-space form (Hyndman-Koehler-Snyder-Grose). For ETS(A, A, N) — additive Holt:

$$
y_t = \ell_{t-1} + b_{t-1} + \varepsilon_t,
$$

$$
\mathbf{x}_t = \begin{pmatrix} \ell_t \\ b_t \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix} \mathbf{x}_{t-1} + \begin{pmatrix} \alpha \\ \alpha\beta \end{pmatrix} \varepsilon_t.
$$

This is a single-innovations state-space model where process noise and measurement noise are the same scalar. A special case of the general linear Gaussian model, with the same Kalman filter recursions applying.

### Integrated Models

For ARIMA(\\(p, d, q\\)) with \\(d = 1\\), the state can be augmented with one extra dimension to handle the non-stationarity. Initialization of the unit-root component uses the diffuse filter (Section 5).

---

## 8. Missing Data and Outliers

### Missing Data

One of the great practical wins of state-space models is **missing data is free**. If \\(y_t\\) is missing at time \\(t\\):

1. Predict step: run as normal, producing \\(\hat{\mathbf{m}}_{t|t-1}, \hat{\mathbf{P}}_{t|t-1}\\).
2. Update step: **skip**. Set \\(\hat{\mathbf{m}}_{t|t} = \hat{\mathbf{m}}_{t|t-1}\\) and \\(\hat{\mathbf{P}}_{t|t} = \hat{\mathbf{P}}_{t|t-1}\\).

The filter continues forward; the missing observation contributes no information to the state, but the state uncertainty grows for that step. The smoother then interpolates retrospectively using future data.

Compare this to ARIMA which has no clean way to handle missing values — the standard approach is to impute before fitting, which is ad hoc. For state-space models it's principled, exact, and automatic.

### Outlier Detection

A large standardized innovation \\(|e_t| / \sqrt{S_t} > 3\\) flags an anomaly. Two responses:

- **Treat as missing**: ignore the observation, don't update state. The series continues undisturbed.
- **Model as an intervention**: add a time-varying covariate or an explicit "additive outlier" to the observation equation.

The **innovations outlier test** in Harvey (1989) computes standardized residuals and tests for anomalies jointly with parameter estimation.

---

## 9. Extensions: Non-Linear and Non-Gaussian

The linear Gaussian case is tractable and covers a huge amount of ground. When it doesn't fit, three extensions:

### Extended Kalman Filter (EKF)

For non-linear state or observation equations \\(\mathbf{x}_t = f(\mathbf{x}_{t-1}) + \mathbf{w}_t\\), \\(y_t = h(\mathbf{x}_t) + v_t\\), linearize around the current mean:

$$
\mathbf{F}_t \approx \frac{\partial f}{\partial \mathbf{x}}\bigg|_{\hat{\mathbf{m}}_{t-1|t-1}}, \qquad \mathbf{H}_t \approx \frac{\partial h}{\partial \mathbf{x}}\bigg|_{\hat{\mathbf{m}}_{t|t-1}}.
$$

Run the linear Kalman filter with these time-varying matrices. EKF works when the non-linearities are mild. It fails (diverges, underestimates uncertainty) when the state distribution is not well-approximated by a Gaussian around the mean — strong nonlinearities, multi-modal posteriors.

### Unscented Kalman Filter (UKF)

Instead of linearizing, propagate **sigma points** — carefully chosen points around the mean — through the nonlinear functions and fit a Gaussian to the results. Captures second-order nonlinearities exactly. The go-to upgrade when EKF diverges but the posterior is still roughly unimodal.

### Particle Filters (Sequential Monte Carlo)

For arbitrarily non-linear, non-Gaussian state-space models, the posterior is represented by weighted samples (particles) propagated and reweighted at each step. The bootstrap particle filter:

1. Sample \\(N\\) particles \\(\{\mathbf{x}^{(i)}_0\}\\) from the initial distribution.
2. At each \\(t\\): propagate each particle through \\(f\\); compute weights \\(w^{(i)}_t \propto p(y_t \mid \mathbf{x}^{(i)}_t)\\); resample from the weighted particles.

Consistent as \\(N \to \infty\\); computationally expensive; prone to particle degeneracy in high dimensions. Used for stochastic volatility models, tracking applications, and stochastic differential equation inference.

For forecasting at moderate dimensions with non-Gaussian innovations, **particle filters with parameter estimation via SMC\\(^2\\) or IF2** (iterated filtering) are the modern state of the art. For most business forecasting problems, the linear Gaussian Kalman filter is still enough — and cheaper, simpler, and more robust.

---

## 10. Python: Structural Forecasting Pipeline

Fit a local-linear-trend-plus-trigonometric-seasonal structural model to a simulated series; run Kalman filter, smoother, component decomposition; forecast; compare to ETS and auto-ARIMA.

### Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.arima.model import ARIMA

rng = np.random.default_rng(13)
n = 180
t = np.arange(n)
trend = 50 + 0.2 * t + rng.normal(0, 0.5, n).cumsum() * 0.3
seasonal = 6 * np.sin(2*np.pi*t/12) + 2 * np.cos(2*np.pi*t/6)
noise = rng.normal(0, 1.5, n)
y = trend + seasonal + noise
idx = pd.date_range('2011-01', periods=n, freq='MS')
series = pd.Series(y, index=idx, name='y')
train = series.iloc[:-24]
test = series.iloc[-24:]
```

### Fit a Structural Model

```python
ucm = UnobservedComponents(train,
                           level='local linear trend',
                           freq_seasonal=[{'period': 12, 'harmonics': 3}])
res = ucm.fit(disp=False)
print(res.summary())
```

`freq_seasonal=[{'period': 12, 'harmonics': 3}]` uses a trigonometric seasonal with 3 harmonics at period 12. The summary reports estimated variances for level, slope, and seasonal innovations — interpretable directly as "how fast does each component change."

### Extract Smoothed Components

```python
ss = res.get_smoothed_decomposition()
# Or equivalently:
smoothed = res.states.smoothed
# smoothed has columns: level, trend, freq_seasonal.period_12

fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
axes[0].plot(train, color='#6db3f2', label='Observed'); axes[0].set_ylabel(r'$y_t$')
axes[0].legend(loc='upper left')
axes[1].plot(smoothed['level'], color='#a4d08a'); axes[1].set_ylabel(r'$\mu_t$')
axes[1].set_title('Smoothed level')
axes[2].plot(smoothed['trend'], color='#f2c894'); axes[2].set_ylabel(r'$\tau_t$')
axes[2].set_title('Smoothed slope')
# Seasonal: sum the sin/cos pairs
seasonal_cols = [c for c in smoothed.columns if 'seasonal' in c]
seasonal_hat = smoothed[seasonal_cols].iloc[:, ::2].sum(axis=1)  # pick real parts
axes[3].plot(seasonal_hat, color='#f2a5a5'); axes[3].set_ylabel(r'$\gamma_t$')
axes[3].set_title('Smoothed seasonal')
for ax in axes: ax.grid(alpha=0.3)
fig.tight_layout()
```

### Forecast with Components

```python
h = 24
fcst = res.get_forecast(steps=h)
mean = fcst.predicted_mean
ci80 = fcst.conf_int(alpha=0.20)
ci95 = fcst.conf_int(alpha=0.05)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(train, color='#6db3f2', label='Train', lw=1)
ax.plot(test,  color='#a4d08a', label='Test (actual)', lw=1.2)
ax.plot(mean,  color='#f2a5a5', label=r'Structural $\hat{y}_{T+h|T}$', lw=1.8)
ax.fill_between(mean.index, ci95.iloc[:,0], ci95.iloc[:,1], color='#f2a5a5', alpha=0.12)
ax.fill_between(mean.index, ci80.iloc[:,0], ci80.iloc[:,1], color='#f2a5a5', alpha=0.25)
ax.set_title('Structural time series forecast')
ax.legend(loc='upper left'); ax.grid(alpha=0.3)
fig.tight_layout()
```

### Compare to ETS and Auto-ARIMA

```python
# ETS
ets_res = ETSModel(train, error='add', trend='add', damped_trend=True,
                   seasonal='add', seasonal_periods=12).fit(disp=False)
ets_fcst = ets_res.forecast(h)

# Basic ARIMA
arima_res = ARIMA(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
arima_fcst = arima_res.forecast(h)

for name, fc in [('Structural', mean), ('ETS', ets_fcst), ('SARIMA', arima_fcst)]:
    mae = np.mean(np.abs(test.values - fc.values))
    print(f'{name:12s} test MAE = {mae:.3f}')
```

Typically on clean structural data, all three are within 5–10% of each other. On series with irregular interventions or long memory, differences grow.

### Missing Data Example

```python
train_missing = train.copy()
idx_missing = rng.choice(len(train_missing), 10, replace=False)
train_missing.iloc[idx_missing] = np.nan

ucm_m = UnobservedComponents(train_missing,
                              level='local linear trend',
                              freq_seasonal=[{'period': 12, 'harmonics': 3}])
res_m = ucm_m.fit(disp=False)

# Smoothed fill for missing values
smooth_pred = res_m.smoothed_signal_mean
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(train, color='#6db3f2', alpha=0.3, label='Full truth')
ax.plot(train_missing, color='#6db3f2', label='Observed')
ax.plot(smooth_pred, color='#f2a5a5', label='Smoothed estimate', lw=1.2)
ax.scatter(train_missing.index[idx_missing], train.iloc[idx_missing],
           color='#a4d08a', s=35, zorder=3, label='True values at missing times')
ax.legend(); ax.grid(alpha=0.3)
```

The smoother fills in missing observations optimally, using future data — MCAR imputation for free.

### Intervention / Outlier Analysis

```python
std_resid = res.standardized_forecasts_error.flatten()
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(train.index[1:], std_resid, color='#6db3f2')
ax.axhline(3, color='#f2a5a5', ls='--'); ax.axhline(-3, color='#f2a5a5', ls='--')
ax.set_title('Standardized one-step innovations')
outliers = train.index[1:][np.abs(std_resid) > 3]
for t in outliers: ax.axvline(t, color='#a4d08a', alpha=0.3)
```

Points outside ±3 are candidate outliers or interventions; investigate before re-fitting.

---

## 11. Industry Best Practices

### 11.1 Use Structural Models When Interpretation Matters

For operational or executive reporting, structural decomposition is worth its weight in gold. "Our forecast is \$112M next month, of which \$98M is baseline level, \$10M is seasonal lift, \$4M is trend continuation" is a sentence a CFO can act on. "Our SARIMA(2,1,2)(0,1,1) forecast is \$112M" is not. Ship structural models for anything human-facing.

### 11.2 Always Use Diffuse Initialization for Non-Stationary States

If your state contains a random walk, integrated component, seasonal factors, or regression coefficients, initialize them with the diffuse Kalman filter (`initialization='diffuse'` in `statsmodels` or `exact_diffuse=TRUE` in `bsts`). Non-diffuse initialization injects prior information that isn't real and biases the first ~\\(2m\\) observations.

### 11.3 Use Joseph-Form Covariance Updates

Numerical Kalman filter implementations that use the simple covariance update \\((\mathbf{I} - \mathbf{K}\mathbf{H}) \mathbf{P}\\) can drift into non-PSD territory after thousands of steps. The Joseph form costs one extra multiplication and permanently fixes this. Most production libraries use it; if you roll your own, don't skip it.

### 11.4 Check State Variance Stability

After fitting, plot \\(\hat{\mathbf{P}}_{t|t}\\) diagonals over time. They should converge to a steady state (the **algebraic Riccati equation** solution). If they don't — growing unboundedly, oscillating — the model is not stabilizable with the observation schedule, and forecasts will diverge. Common cause: observation matrix doesn't make all state components observable.

### 11.5 Profile the Likelihood Before Calling MLE Done

State-space likelihoods often have flat ridges, especially when \\(\sigma^2_\mu\\) and \\(\sigma^2_\tau\\) are both small. Fit from several starting values and check they all converge to the same MLE. If the likelihood is near-flat along a ridge, fix one variance (e.g., \\(\sigma^2_\mu = 0\\) for smooth trend) and re-fit.

### 11.6 Watch for Invariance to Scaling

If you scale \\(y_t\\) by a constant, the MLEs of variances scale by the square. Check your parameter estimates are order-of-magnitude reasonable given the data scale. Some packages default to unconstrained optimization in log-variances; make sure your starting values aren't astronomical.

### 11.7 Refit Periodically; Monitor Innovation Sequence

After deployment, track one-step innovations in production. Standardized innovations should have mean zero, variance 1, and pass a Ljung-Box test. When any of these drift, refit. Structural models are more robust than ARIMA to regime changes because they can *learn* a new level or slope through the state equations, but they can't adapt to changes in the noise structure — that requires refit.

### 11.8 Start Simple; Add Components Incrementally

- Start: local level only.
- Add local slope → local linear trend.
- Add seasonal component.
- Add cycle component only if business cycle is relevant.
- Add regressors last.

Each step should produce a measurable AICc improvement. If adding a seasonal doesn't lower AICc, the series doesn't have seasonal structure worth modeling — stop and ship the simpler model.

### 11.9 Combine State-Space Forecasts With ETS and ARIMA

Ensembling across model classes dominates any single-class ensemble. Average UCM + ETS + ARIMA; the three disagree in different ways and the mean tends to win. A production pipeline I have seen ship: equal-weight mean of 4 methods (UCM, ETS, SARIMA, Theta), median over last 4 refits, monitored daily.

### 11.10 Use Bayesian Structural Time Series for Small Data

`bsts` (Scott and Varian, 2014) and PyMC/PyStan-based approaches put priors on variance parameters and integrate over them via MCMC or variational methods. This is critical when you have few observations — uncertainty in the variance parameters is the dominant source of forecast uncertainty, and plug-in MLE under-reports it. For short series with executive visibility, a Bayesian structural model returns honest prediction intervals where a frequentist fit gives over-confident ones.

### 11.11 Be Careful With Multivariate Models

Multivariate state-space models (VAR, dynamic factor models) can deliver big accuracy gains by borrowing strength across series. They can also overfit catastrophically at state dimension much above \\(p \cdot q\\) — the number of free parameters grows quadratically. Use shrinkage priors, restrict cross-equation dynamics (lower-triangular \\(\mathbf{F}\\)), or adopt a factor structure. Never run unconstrained VAR on > 10 series without regularization.

### 11.12 Document the State Space

Every production state-space model should have a one-page document describing: state vector composition, transition matrix (possibly time-varying), observation equation, innovation covariances, initialization method, estimator used, parameter constraints, and refit cadence. Complex state-space models are second only to deep-learning models for post-deployment debugging nightmares. A clear state diagram — drawn once and checked in — saves you ten hours when something breaks in production.

---

## Summary and What's Next

State-space models unify the forecasting machinery of this series:

- A **linear Gaussian state-space model** separates latent state (what you want to know) from observation (what you see) with a transition dynamic and an observation map.
- The **Kalman filter** computes exact filtering distributions in \\(O(T)\\) time; the RTS **smoother** does the retrospective version.
- The filter's byproducts (innovations and their variances) give exact likelihoods, gradients, and MLE infrastructure.
- **Structural time series** — level + trend + seasonal + cycle, each as an explicit unobserved component — give interpretable, separately-forecastable decompositions that ARMA cannot.
- Every ARIMA and ETS model has a state-space representation; the state-space formulation is how all of them are estimated in practice.
- Missing data, outliers, and interventions are handled natively in state-space.
- Non-linear extensions (EKF, UKF, particle filters) exist for when the Gaussian assumption fails.

In [Part 5](/2026/04/22/modern-forecasting-garch-gbm-nbeats-hierarchical.html), the final post, we cover modern forecasting practice that sits on top of this foundation: **GARCH** for volatility forecasting (the right tool when the variance is itself a time series), **gradient-boosted trees** with lag features (the workhorse of competitive forecasting since M5), **N-BEATS** (a deep MLP architecture without any attention), **hierarchical reconciliation** (for forecasting thousands of related series that must aggregate consistently), and the **proper scoring rules / calibration framework** that tells you whether your probabilistic forecasts are honest.
