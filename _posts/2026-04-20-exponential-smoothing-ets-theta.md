---
layout: post
title: "Exponential Smoothing, ETS, and the Theta Method: The Forecasters' Swiss Army Knife"
date: 2026-04-20
category: math
---

*This is Part 3 of a 5-part series on time series forecasting. [Part 1: Foundations](/2026/04/18/time-series-foundations-stationarity.html) | [Part 2: ARIMA & Box-Jenkins](/2026/04/19/arima-box-jenkins-forecasting.html) | **Part 3: Exponential Smoothing, ETS & Theta** | [Part 4: State-Space & Kalman](/2026/04/21/state-space-kalman-filtering.html) | [Part 5: Modern Forecasting](/2026/04/22/modern-forecasting-garch-gbm-nbeats-hierarchical.html)*

ARIMA is beautiful, rigorous, and — on the kind of short, seasonal, slightly-ratty series that constitute 90% of real-world forecasting problems — often slower, less accurate, and more brittle than a seventy-year-old heuristic from an inventory-management paper that nobody reads any more. The Makridakis forecasting competitions (M, M3, M4, M5, held between 1982 and 2020) have repeatedly found that simple exponential smoothing methods, and their modern state-space refinement ETS, win on most operational forecasting series. The 2018 M4 competition's top entries were either ETS, or combinations *including* ETS, or hybrid models that used ETS as a core component. The same was true in M5.

Why? Because exponential smoothing makes precisely the bets that match how real business series behave: the recent past matters more than the distant past, trends decay rather than persist forever, seasonal patterns are mostly stable but allow gentle drift, and the cost of a misspecified likelihood is higher than the cost of an unbiased heuristic. These bets are informal in the original 1957 Holt paper; they became formal and extensible when Hyndman, Koehler, Snyder, and Grose fit them inside a state-space framework in 2002, producing the **innovations state-space** family usually called **ETS** (Error, Trend, Seasonal).

This post covers four things. First, classical exponential smoothing — simple, Holt, Holt-Winters, damped trend — as recursions, with geometric intuition and smoothing-parameter interpretations. Second, the state-space ETS framework that turns these heuristics into a likelihood-based family with thirty variants (additive / multiplicative; damped / not; additive noise / multiplicative noise), selection by AICc, and closed-form forecast distributions. Third, the **Theta method**, a deceptively trivial decomposition that won M3 and whose performance remains hard to beat. Fourth, industry best practices — how these methods get used, combined, and monitored in production.

---

## Table of Contents

1. [Simple Exponential Smoothing](#1-simple-exponential-smoothing)
2. [Holt's Linear Trend Method](#2-holts-linear-trend-method)
3. [Damped Trend](#3-damped-trend)
4. [Holt-Winters Seasonal Methods](#4-holt-winters-seasonal-methods)
5. [The ETS State-Space Framework](#5-the-ets-state-space-framework)
6. [Estimation and Model Selection](#6-estimation-and-model-selection)
7. [Forecasts and Prediction Intervals](#7-forecasts-and-prediction-intervals)
8. [The Theta Method](#8-the-theta-method)
9. [When to Use ETS vs. ARIMA](#9-when-to-use-ets-vs-arima)
10. [Python: End-to-End ETS Pipeline](#10-python-end-to-end-ets-pipeline)
11. [Industry Best Practices](#11-industry-best-practices)

---

## 1. Simple Exponential Smoothing

Start with the simplest case: a series with no trend and no seasonality, just local-level variation around a slowly-drifting mean. Call the latent level \(\ell_t\). **Simple exponential smoothing (SES)** estimates it as an exponentially weighted average of the observations:

$$
\ell_t = \alpha X_t + (1 - \alpha)\ell_{t-1},
$$

where \(\alpha \in [0, 1]\) is the **smoothing parameter**. Forecasts at all horizons are flat at the current level: \(\hat{X}_{T+h|T} = \ell_T\). Unrolling the recursion,

$$
\ell_T = \alpha \sum_{j=0}^{T-1} (1 - \alpha)^j X_{T-j} + (1 - \alpha)^T \ell_0.
$$

The weights on past observations decay geometrically. With \(\alpha = 0.1\) the half-life is about 6.6 periods; with \(\alpha = 0.5\) it's roughly one period. This is exactly the IIR low-pass filter familiar from signal processing, and the reason for the name: the influence of past observations decays *exponentially* with their distance from now.

### What \(\alpha\) Means

- **\(\alpha \to 0\)**: pure mean, recent observations barely update the level. Appropriate when the series is nearly stationary and noise is high.
- **\(\alpha \to 1\)**: level equals the latest observation. Appropriate when the series changes rapidly and the signal-to-noise ratio is high. At the limit, SES becomes the naive-1 forecast.

The optimal \(\alpha\) is chosen to minimize one-step-ahead squared error, usually by numerical optimization. This is equivalent to MLE under Gaussian innovations in the state-space form (Section 5).

### Error-Correction Form

Rewrite the update:

$$
\ell_t = \ell_{t-1} + \alpha(X_t - \ell_{t-1}) = \ell_{t-1} + \alpha e_t,
$$

where \(e_t = X_t - \ell_{t-1}\) is the **one-step-ahead forecast error**. The smoothing parameter \(\alpha\) is the *learning rate* of the level — how aggressively it corrects toward the last observation. This formulation is the bridge to the state-space representation and also to gradient-descent intuitions: SES is an online SGD update on the level with learning rate \(\alpha\).

### Model Implied by SES

SES is the optimal forecast for an ARIMA(0,1,1) — a random walk plus an MA(1) error. Specifically, if

$$
X_t = X_{t-1} + \varepsilon_t - (1 - \alpha) \varepsilon_{t-1},
$$

then the SES recursion produces exactly the minimum-MSE one-step forecast. This equivalence is not coincidence: the Wold / lag-polynomial machinery of Part 2 and the recursive smoothing of Part 3 are two views of the same object. ETS in Section 5 makes this precise.

---

## 2. Holt's Linear Trend Method

SES assumes a flat forecast — useless for series with trend. **Holt's method** (1957) adds a local slope:

$$
\ell_t = \alpha X_t + (1 - \alpha)(\ell_{t-1} + b_{t-1}), \qquad b_t = \beta(\ell_t - \ell_{t-1}) + (1 - \beta) b_{t-1},
$$

where \(b_t\) is the **trend** (slope per period). The forecast is linear in horizon:

$$
\hat{X}_{T+h|T} = \ell_T + h \cdot b_T.
$$

Two smoothing parameters now: \(\alpha\) controls the level, \(\beta\) controls the trend. The error-correction form:

$$
\ell_t = \ell_{t-1} + b_{t-1} + \alpha e_t, \qquad b_t = b_{t-1} + \alpha\beta e_t.
$$

Both level and trend are driven by the same one-step-ahead error, just with different gains. If \(\beta\) is small, the trend is very stable; if \(\beta\) is near 1, the trend tracks the most recent two-period change closely.

### Why Holt Is Dangerous at Long Horizons

The linear forecast \(\hat{X}_{T+h|T} = \ell_T + h b_T\) **grows without bound**. For a retail series, this will happily extrapolate next year's sales into the stratosphere if the recent trend was steep. Holt with untamed slope is the leading cause of "wildly optimistic revenue forecast" in automated forecasting systems. The cure is Section 3.

### Holt as ARIMA(0,2,2)

Holt's method is the optimal forecast for ARIMA(0,2,2) with specific parameter constraints. Two unit roots absorb a linear trend; the double integration is why Holt tracks persistent trends faithfully. Equivalent to thinking of Holt as a local linear regression with decaying weights.

---

## 3. Damped Trend

Gardner and McKenzie (1985) fixed Holt's long-horizon pathology with a **damping parameter** \(\phi \in (0, 1)\):

$$
\ell_t = \alpha X_t + (1 - \alpha)(\ell_{t-1} + \phi b_{t-1}), \qquad b_t = \beta(\ell_t - \ell_{t-1}) + (1 - \beta)\phi b_{t-1}.
$$

The trend is multiplied by \(\phi\) at each step, so the \(h\)-step-ahead forecast becomes

$$
\hat{X}_{T+h|T} = \ell_T + (\phi + \phi^2 + \ldots + \phi^h) b_T = \ell_T + \phi\, \frac{1 - \phi^h}{1 - \phi}\, b_T.
$$

As \(h \to \infty\), this converges to \(\ell_T + \frac{\phi}{1 - \phi} b_T\), a finite asymptote. The trend *decays* rather than persisting forever — which is what economic series actually do. Growth rates revert; consumer trends saturate; demand plateaus.

### Typical Damping Values

- **\(\phi = 1\)**: standard Holt (no damping).
- **\(\phi \in [0.8, 0.98]\)**: most production settings. The M3 competition found damped trend models systematically beat non-damped ones.
- **\(\phi < 0.8\)**: rarely needed; usually indicates the trend component is not really useful and simple level smoothing is enough.

The data selects \(\phi\) by MLE or AICc. In practice, one of the biggest accuracy lifts you can get on a large forecasting system is: switch Holt to damped Holt, and re-estimate \(\phi\) from data.

<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg">
  <rect width="680" height="300" fill="#1a1a1a"/>
  <text x="340" y="22" fill="#e8e8e8" font-size="15" text-anchor="middle" font-family="Georgia, serif" font-weight="bold">Linear Trend vs. Damped Trend Forecasts</text>
  <line x1="50" y1="260" x2="650" y2="260" stroke="#888" stroke-width="1"/>
  <line x1="50" y1="260" x2="50" y2="40" stroke="#888" stroke-width="1"/>
  <!-- In-sample -->
  <polyline points="50,230 90,225 130,218 170,210 210,200 250,195 290,188 330,182 370,180" fill="none" stroke="#6db3f2" stroke-width="1.8"/>
  <line x1="370" y1="260" x2="370" y2="40" stroke="#555" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="370" y="275" fill="#888" font-size="11" text-anchor="middle" font-family="Georgia, serif">T</text>
  <!-- Linear forecast -->
  <polyline points="370,180 410,160 450,140 490,120 530,100 570,80 610,60 650,40" fill="none" stroke="#f2a5a5" stroke-width="2" stroke-dasharray="5,3"/>
  <text x="650" y="55" fill="#f2a5a5" font-size="11" font-family="Georgia, serif" text-anchor="end">Holt (no damp)</text>
  <!-- Damped forecast -->
  <polyline points="370,180 410,168 450,160 490,155 530,151 570,148 610,147 650,146" fill="none" stroke="#a4d08a" stroke-width="2"/>
  <text x="650" y="160" fill="#a4d08a" font-size="11" font-family="Georgia, serif" text-anchor="end">Damped, φ=0.9</text>
</svg>

---

## 4. Holt-Winters Seasonal Methods

Add a seasonal component \(s_t\) with period \(m\) (12 for monthly, 4 for quarterly, 7 for daily with weekly cycle). There are two canonical variants, additive and multiplicative.

### Additive Seasonality

When the seasonal amplitude is roughly constant in absolute units:

$$
\begin{aligned}
\ell_t &= \alpha(X_t - s_{t-m}) + (1 - \alpha)(\ell_{t-1} + b_{t-1}), \\
b_t &= \beta(\ell_t - \ell_{t-1}) + (1 - \beta) b_{t-1}, \\
s_t &= \gamma(X_t - \ell_{t-1} - b_{t-1}) + (1 - \gamma) s_{t-m}, \\
\hat{X}_{T+h|T} &= \ell_T + h \cdot b_T + s_{T + h - m(\lfloor (h-1)/m \rfloor + 1)}.
\end{aligned}
$$

Three smoothing parameters: \(\alpha\) for level, \(\beta\) for trend, \(\gamma\) for seasonal. The seasonal component repeats every \(m\) periods, with slow updating.

### Multiplicative Seasonality

When the seasonal amplitude scales with the level (common for revenue, bookings, page views):

$$
\begin{aligned}
\ell_t &= \alpha \frac{X_t}{s_{t-m}} + (1 - \alpha)(\ell_{t-1} + b_{t-1}), \\
b_t &= \beta(\ell_t - \ell_{t-1}) + (1 - \beta) b_{t-1}, \\
s_t &= \gamma \frac{X_t}{\ell_{t-1} + b_{t-1}} + (1 - \gamma) s_{t-m}, \\
\hat{X}_{T+h|T} &= (\ell_T + h \cdot b_T) \cdot s_{T + h - m(\lfloor (h-1)/m \rfloor + 1)}.
\end{aligned}
$$

Division rather than subtraction; multiplication rather than addition. The series must be strictly positive for multiplicative seasonality to be defined — you cannot divide by zero.

### When to Choose Which

A simple diagnostic: plot the series and its log. If the original has seasonality whose amplitude grows with the level, and the log has constant-amplitude seasonality, use **multiplicative** seasonality (or equivalently, log-transform then use additive). If the original already has constant-amplitude seasonality, use **additive**.

Multiplicative Holt-Winters with log pre-transform is often equivalent to additive Holt-Winters on logs, up to Jensen's-gap corrections. In practice many shops just log-transform and do additive everywhere; it's numerically stable and the exponential back-transform is straightforward.

### Damped Seasonal Variants

The damping parameter \(\phi\) extends naturally: \(b_{t-1}\) gets multiplied by \(\phi\) in the level update, \(\phi b_{t-1}\) in the trend inherited value. This gives the damped-Holt-Winters model used as a default in industry.

---

## 5. The ETS State-Space Framework

The recursions above are heuristic. In 2002, Hyndman, Koehler, Snyder, and Grose embedded them in a formal **innovations state-space model** with one source of random error — this is the **ETS** framework that now underpins the `forecast` package in R and the `ETSModel` class in `statsmodels`.

### Taxonomy

ETS models are named by three letters: **Error \(\times\) Trend \(\times\) Seasonal**, each of which can be:

- **N** — none
- **A** — additive
- **M** — multiplicative
- **Ad** — additive damped (only for trend)
- **Md** — multiplicative damped (rare)

So ETS(A, N, N) is SES with additive noise. ETS(A, A, A) is additive Holt-Winters with additive noise. ETS(M, Ad, M) is multiplicative Holt-Winters with damped multiplicative trend and multiplicative noise. Thirty permutations in total, of which about fifteen are practically used.

### State-Space Form

Every ETS model has the form

$$
X_t = w(\mathbf{x}_{t-1}) + r(\mathbf{x}_{t-1}) \varepsilon_t, \qquad \mathbf{x}_t = f(\mathbf{x}_{t-1}) + g(\mathbf{x}_{t-1}) \varepsilon_t,
$$

where \(\mathbf{x}_t\) is the state (level, trend, seasonal components), \(\varepsilon_t \sim \mathcal{N}(0, \sigma^2)\) is a single innovation, and \(w, r, f, g\) are known functions of the state and model choice. The "innovations" part means the same \(\varepsilon_t\) drives both the observation and the state — no separate measurement noise. This is the defining simplification vs. general state-space models (Part 4).

### ETS(A, N, N) Worked Out

For SES with additive error:

$$
X_t = \ell_{t-1} + \varepsilon_t, \qquad \ell_t = \ell_{t-1} + \alpha \varepsilon_t.
$$

Observation equation: \(X_t\) equals the previous level plus innovation. State equation: the level updates by \(\alpha\) times the innovation. This is exactly the error-correction form of Section 1, with \(\varepsilon_t = X_t - \ell_{t-1}\) identified as the innovation.

### ETS(A, A, N) Worked Out

For Holt with additive error:

$$
X_t = \ell_{t-1} + b_{t-1} + \varepsilon_t,
$$

$$
\ell_t = \ell_{t-1} + b_{t-1} + \alpha \varepsilon_t, \qquad b_t = b_{t-1} + \alpha\beta \varepsilon_t.
$$

The state vector is \(\mathbf{x}_t = (\ell_t, b_t)\). Innovation propagates to both components with gains \(\alpha\) and \(\alpha\beta\). Same structure extends to seasonal variants.

### ETS with Multiplicative Error

ETS(M, N, N) has \(X_t = \ell_{t-1}(1 + \varepsilon_t)\), meaning error is proportional to level. For revenue/demand this is often more realistic: a \$1M company has noise ≈ \$100K, a \$100M company has noise ≈ \$10M; the *relative* noise is stable. Multiplicative-error models often have better coverage on heteroskedastic positive series than additive ones.

### Why ETS Is a Big Deal

Three practical consequences of moving from heuristic smoothing to the state-space form:

1. **Likelihood-based estimation** (Section 6): MLE of \(\alpha, \beta, \gamma, \phi, \sigma^2\) and initial state \(\mathbf{x}_0\).
2. **Principled model selection**: AICc picks among thirty variants automatically.
3. **Analytic forecast distributions**: for additive-error variants, exact Gaussian forecast distributions; for multiplicative-error variants, simulation-based forecast distributions. *No more back-of-envelope intervals*.

---

## 6. Estimation and Model Selection

### Likelihood

For additive-error ETS, the one-step-ahead prediction errors \(e_t = X_t - \hat{X}_{t|t-1}\) are i.i.d. \(\mathcal{N}(0, \sigma^2)\) under the model. The concentrated log-likelihood (with \(\sigma^2\) profiled out) becomes

$$
\ell_c(\boldsymbol{\eta}) = -\frac{T}{2} \log \left(\frac{1}{T} \sum_{t=1}^T e_t(\boldsymbol{\eta})^2 \right),
$$

up to constants. Minimizing this is equivalent to minimizing mean squared one-step error — so MLE and "least-squares recursion fitting" coincide here.

For multiplicative-error ETS, the log-likelihood is slightly more involved because the observation variance depends on the state:

$$
\ell(\boldsymbol{\eta}) = -\frac{T}{2}\log\sigma^2 - \frac{1}{2\sigma^2}\sum_t \varepsilon_t^2 - \sum_t \log|r(\mathbf{x}_{t-1})|.
$$

The last term is the log-Jacobian from the transformation between \(X_t\) and \(\varepsilon_t\). This term is what prevents multiplicative-error ETS from being compared to additive-error ETS by MSE alone.

### Initial State

The recursion needs \(\mathbf{x}_0\). Three options:

1. **Heuristic**: set \(\ell_0\) = first observation, \(b_0\) = average early slope, \(s_0\) = average deviations from a linear fit on the first \(2m\) observations.
2. **MLE**: treat \(\mathbf{x}_0\) as parameters and optimize jointly. This is what `forecast::ets` and `statsmodels.ETS` do by default.
3. **Exact initialization**: diffuse or flat prior, integrated out analytically. Gives identical asymptotic behavior to MLE.

For short series (\(T < 50\)), the choice of initial state materially affects forecasts. For longer series it matters less.

### AICc Selection Across Variants

For each candidate ETS model \(m\), compute

$$
\mathrm{AICc}_m = -2\ell(\hat{\boldsymbol{\eta}}_m) + 2k_m + \frac{2k_m(k_m+1)}{T - k_m - 1}.
$$

Select the model with minimum AICc. Hyndman's `ets()` in R automates this: it tries all valid ETS variants given the data (if the series has zeros it disables multiplicative variants; if length \(< 2m\) it disables seasonal), fits each, picks minimum AICc. `statsmodels.ETSModel` with `auto=True` does the same. This is the standard workflow — not running an auto-ETS is leaving accuracy on the table.

### Multiplicative vs. Additive: Which Wins

Empirically on M3 and M4:

- Multiplicative error variants win on strictly positive trending series with clear seasonality.
- Additive error variants win on series with mild/no trend and possible zero-or-negative values (returns, deviations, temperature anomalies).
- Damped-trend variants win over non-damped variants nearly always.

A reasonable default when AICc is unavailable: ETS(M, Ad, M) for revenue/volume; ETS(A, Ad, A) for balanced series; ETS(A, N, N) (SES) for stationary-around-a-level series.

---

## 7. Forecasts and Prediction Intervals

### Point Forecasts

For any ETS model, the point forecast \(\hat{X}_{T+h|T} = \mathbb{E}[X_{T+h} \mid \mathcal{F}_T]\) is computed by iterating the state equations forward with future innovations set to zero:

$$
\hat{\mathbf{x}}_{T+1} = f(\mathbf{x}_T), \quad \hat{\mathbf{x}}_{T+2} = f(\hat{\mathbf{x}}_{T+1}), \quad \ldots
$$

and applying the observation function \(w\) at each step. For additive-error models this is an exact conditional expectation; for multiplicative-error models it is an approximation (because the expectation of a product is not the product of expectations when the terms share noise), but the error is small.

### Prediction Interval Variance

For additive-error models, the \(h\)-step forecast variance has a known analytic form. For ETS(A, N, N):

$$
\mathrm{Var}(X_{T+h} - \hat{X}_{T+h|T}) = \sigma^2 \left[ 1 + (h-1) \alpha^2 \right].
$$

For ETS(A, A, N) (Holt) with damping \(\phi\):

$$
\mathrm{Var}(X_{T+h} - \hat{X}_{T+h|T}) = \sigma^2 \left[1 + \sum_{j=1}^{h-1} \left( \alpha + \alpha \beta \frac{\phi(1 - \phi^j)}{1 - \phi}\right)^2 \right].
$$

These grow with horizon but often more slowly than ARIMA, because ETS damping reins in trend uncertainty. Hyndman, Koehler, Ord and Snyder (2008, *Forecasting with Exponential Smoothing*) catalogue the full set of variance formulas.

### Multiplicative-Error PIs: Simulation

For multiplicative-error variants, the forecast distribution is non-Gaussian and skewed. Analytical PIs do not exist in closed form. The production pattern:

1. Draw \(B\) independent simulations from the fitted model, each running the state-space equations forward with fresh i.i.d. Gaussian innovations.
2. At each horizon, compute empirical quantiles across the \(B\) simulations.

```python
sims = res.simulate(nsimulations=horizon, repetitions=5000)
p05 = np.quantile(sims, 0.05, axis=1)
p95 = np.quantile(sims, 0.95, axis=1)
```

This gives coverage that's calibrated even when the forecast distribution is log-normal-ish. For the cost of 5000 Monte Carlo runs (cheap), you get correct PIs — use this over analytic approximations for any multiplicative or skewed series.

### Calibration Is the Metric That Matters

The point forecast is a number. The forecast *distribution* is the thing your business decisions actually use — when you calculate a service-level inventory target, you're asking for the 95th percentile. Coverage diagnostic:

$$
\mathrm{Coverage}(\alpha) = \frac{1}{H} \sum_{h=1}^H \mathbf{1}[Q^{(\alpha/2)}_h \le X_{T+h} \le Q^{(1-\alpha/2)}_h],
$$

where \(Q^{(p)}_h\) is the predicted \(p\)-quantile at horizon \(h\). For a nominal 80% PI, the realized coverage across a rolling-origin evaluation should be near 80%. Under-coverage (e.g., 65% realized) means the model's uncertainty is *too tight* — a silent failure mode common in production forecasts.

---

## 8. The Theta Method

In M3 (2000), a method no one had heard of from Assimakopoulos and Nikolopoulos beat every other entry in accuracy averaged over 3003 series. It is trivially simple:

1. Deseasonalize the series (classical additive decomposition with period \(m\)).
2. Fit two "theta lines": \(L_0\) is the linear regression of \(X\) on \(t\); \(L_2\) is the series \(X\) itself with curvature doubled (\(\theta = 2\)) — equivalently, \(X + (X - L_0)\).
3. Forecast \(L_0\) by extending the linear regression, and \(L_2\) by simple exponential smoothing.
4. Average the two forecasts and re-seasonalize.

That's it. Why does it work? Because it decomposes the series into "long-run linear trend" (\(L_0\)) and "short-run noise + local curvature" (\(L_2\)), extrapolates each with the right method (linear / SES), and averages — an implicit ensemble. The M3 winner, later M4 top performer, and persistent inclusion in production ensembles trace back to this idea.

### Theta as a State-Space Model

Hyndman and Billah (2003) showed that the Theta method is equivalent to a specific ETS model: ETS(A, N, N) applied to the \(\theta = 2\) line with drift added. This connects it back to the unified framework: Theta isn't magic, it's ETS + a deterministic trend.

### Generalized Theta

Fiorucci et al. (2016) introduced a **dynamic optimized theta** (DOTM) model that lets \(\theta\) itself vary. This is what many modern production forecasting stacks actually use under the "theta" name. `nixtla/statsforecast` implements DOTM; on M4 it outperforms classical theta.

### Why Theta Is in Every Modern Ensemble

Theta is cheap (linear regression + SES), robust (few parameters, stable estimates), and complementary to ETS/ARIMA (different inductive biases). A vanilla average of ETS, ARIMA, and Theta forecasts beat every individual method in M4. The practitioner's rule: fit all three, equal-weight average.

---

## 9. When to Use ETS vs. ARIMA

Both produce calibrated forecasts for stationary series; their strengths diverge:

| Scenario | Prefer |
|---|---|
| Short series (\(T < 50\)) | ETS |
| Long series (\(T > 500\)) with memory | ARIMA |
| Strong multiplicative seasonality | ETS(*,*,M) |
| Need exogenous regressors | ARIMAX (or regression + ARIMA errors) |
| Need damped trend | ETS(A, Ad, *) |
| Heteroskedastic positive series | ETS with M error |
| Need interpretable coefficients | ARIMA |
| Automated large-scale fitting | ETS (faster, fewer failures) |
| Complex correlation at many lags | ARIMA(p, d, q) with high p |

ETS wins by a small margin on M4 / M5 benchmarks on average, but the two are best combined. Equal-weight averaging of ETS + ARIMA + Theta is the industry default for an automated, high-volume forecasting pipeline where per-item tuning is impossible.

---

## 10. Python: End-to-End ETS Pipeline

Simulate a series with trend + multiplicative seasonality + noise, fit an auto-ETS, produce forecasts with simulation-based PIs, and compare to Holt-Winters, SES, and Theta.

### Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.metrics import mean_absolute_error

rng = np.random.default_rng(11)
n = 156  # 13 years of monthly data
t = np.arange(n)
level = 100.0 + 0.5 * t
season = 1.0 + 0.25 * np.sin(2*np.pi*t/12) + 0.10 * np.cos(2*np.pi*t/6)
noise = rng.normal(0, 0.03, n)
x = level * season * (1 + noise)
idx = pd.date_range('2013-01', periods=n, freq='MS')
series = pd.Series(x, index=idx, name='demand')
train = series.iloc[:-24]
test = series.iloc[-24:]
```

### Fit Several ETS Variants and Select by AICc

```python
candidates = [
    ('ETS(A,N,N)',   dict(error='add', trend=None, seasonal=None)),
    ('ETS(A,A,N)',   dict(error='add', trend='add', seasonal=None)),
    ('ETS(A,Ad,N)',  dict(error='add', trend='add', damped_trend=True, seasonal=None)),
    ('ETS(A,A,A)',   dict(error='add', trend='add', seasonal='add', seasonal_periods=12)),
    ('ETS(A,Ad,A)',  dict(error='add', trend='add', damped_trend=True,
                          seasonal='add', seasonal_periods=12)),
    ('ETS(M,A,M)',   dict(error='mul', trend='add', seasonal='mul', seasonal_periods=12)),
    ('ETS(M,Ad,M)',  dict(error='mul', trend='add', damped_trend=True,
                          seasonal='mul', seasonal_periods=12)),
]

best = (np.inf, None, None, None)
for name, kw in candidates:
    try:
        m = ETSModel(train, **kw)
        r = m.fit(disp=False)
        k = len(r.params) + 1
        aicc = r.aic + 2*k*(k+1) / max(len(train) - k - 1, 1)
        print(f"{name:12s}  AIC={r.aic:8.1f}  AICc={aicc:8.1f}")
        if aicc < best[0]:
            best = (aicc, name, r, kw)
    except Exception as ex:
        print(f"{name:12s}  FAIL: {ex}")

print(f"\nBest model: {best[1]}")
```

Expected winner on this kind of series: ETS(M, Ad, M) — multiplicative error, damped additive trend, multiplicative seasonality.

### Forecast with Simulation-Based PIs

```python
res = best[2]
h = 24

# Point forecast
fcst = res.forecast(steps=h)

# Simulation PIs
sims = res.simulate(nsimulations=h, repetitions=5000, anchor='end')
# sims shape: (h, 5000)
p05 = np.quantile(sims, 0.05, axis=1)
p20 = np.quantile(sims, 0.20, axis=1)
p80 = np.quantile(sims, 0.80, axis=1)
p95 = np.quantile(sims, 0.95, axis=1)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(train, color='#6db3f2', label='Train', lw=1.2)
ax.plot(test,  color='#a4d08a', label='Test (actual)', lw=1.2)
ax.plot(fcst,  color='#f2a5a5', label=r'$\hat{X}_{T+h|T}$', lw=1.8)
ax.fill_between(fcst.index, p05, p95, color='#f2a5a5', alpha=0.15, label='90% PI')
ax.fill_between(fcst.index, p20, p80, color='#f2a5a5', alpha=0.30, label='60% PI')
ax.set_title(f'{best[1]} — Forecast with simulation PIs')
ax.set_ylabel(r'$X_t$')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
fig.tight_layout()
```

### Theta Method

```python
from statsforecast.models import Theta as ThetaMethod
from statsforecast import StatsForecast

# Simple hand-rolled Theta for clarity:
def theta_forecast(y, h, m):
    # Deseasonalize (additive) via classical decomposition
    s = np.array([np.mean([y[i] for i in range(j, len(y), m)]) for j in range(m)])
    s -= s.mean()
    s_tiled = np.tile(s, int(np.ceil(len(y)/m)))[:len(y)]
    y_ds = y - s_tiled

    # Linear reg on deseasonalized
    t = np.arange(len(y_ds))
    slope, intercept = np.polyfit(t, y_ds, 1)
    L0_future = intercept + slope * (len(y_ds) + np.arange(h))

    # Theta-2 line: SES on deseasonalized
    ses = ExponentialSmoothing(y_ds, trend=None, seasonal=None).fit(optimized=True)
    L2_future = np.full(h, ses.forecast(h).values[0])
    # For pedagogical clarity we flat-extrapolate the SES; statsforecast does it properly.

    # Average and re-seasonalize
    fcst = 0.5 * (L0_future + L2_future) + np.tile(s, int(np.ceil(h/m)))[:h]
    return fcst

theta_fcst = theta_forecast(train.values, 24, 12)
```

### Compare Methods on Rolling Origin

```python
def rolling_origin_eval(series, method_fn, horizon=12, min_train=48, step=1):
    errors = []
    for end in range(min_train, len(series) - horizon, step):
        tr = series.iloc[:end]
        te = series.iloc[end:end+horizon]
        try:
            fc = method_fn(tr, horizon)
            errors.append(np.abs(te.values - fc))
        except Exception:
            continue
    return np.array(errors)  # (n_windows, horizon)

def fit_ets(tr, h, **kw):
    r = ETSModel(tr, **kw).fit(disp=False)
    return r.forecast(h).values

def fit_hw(tr, h):
    return ExponentialSmoothing(tr, trend='add', seasonal='mul',
                                seasonal_periods=12, damped_trend=True
                               ).fit(optimized=True).forecast(h).values

def fit_ses(tr, h):
    return ExponentialSmoothing(tr, trend=None, seasonal=None
                               ).fit(optimized=True).forecast(h).values

def fit_naive(tr, h):
    return np.full(h, tr.iloc[-1])

def fit_seasonal_naive(tr, h):
    vals = tr.values
    return np.array([vals[-12 + i % 12] for i in range(h)])

methods = {
    'SES':       fit_ses,
    'HW (add/mul, damped)': fit_hw,
    'Naive-1':   fit_naive,
    'Seasonal naive': fit_seasonal_naive,
    'Theta':     lambda tr, h: theta_forecast(tr.values, h, 12),
}

results = {}
for name, fn in methods.items():
    errs = rolling_origin_eval(series, fn, horizon=12, min_train=48)
    mae = np.mean(np.abs(errs))
    # MASE = MAE / MAE of seasonal naive
    sn_errs = rolling_origin_eval(series, fit_seasonal_naive, horizon=12, min_train=48)
    mase = mae / np.mean(np.abs(sn_errs))
    results[name] = {'MAE': mae, 'MASE': mase}

print(pd.DataFrame(results).T)
```

On this simulated series you should see Holt-Winters and Theta with MASE < 1 (beat seasonal naive), SES with MASE around 1.5 (no seasonality), naive-1 much worse.

### Simple Ensemble

```python
def ensemble(tr, h):
    f1 = fit_hw(tr, h)
    f2 = theta_forecast(tr.values, h, 12)
    f3 = fit_ets(tr, h, error='mul', trend='add', damped_trend=True,
                 seasonal='mul', seasonal_periods=12)
    return (f1 + f2 + f3) / 3

errs = rolling_origin_eval(series, ensemble, horizon=12, min_train=48)
mae = np.mean(np.abs(errs))
print(f'Ensemble MAE: {mae:.3f}')
```

Equal-weight ensemble of Holt-Winters + Theta + ETS(M,Ad,M) is usually the hardest-to-beat simple approach on monthly data.

---

## 11. Industry Best Practices

### 11.1 Default to Auto-ETS Before Anything Else

On any new operational forecasting problem, fit an auto-ETS as your first model. It will run quickly, produce something reasonable, and set the baseline every more complex model must beat. Report AICc + rolling-origin MASE; if your fancy model doesn't improve both, ship auto-ETS.

### 11.2 Always Include Damped Trend in the Candidate Set

Across M3 and M4, damped-trend variants win or tie non-damped variants on nearly every trending series. Many production forecasting systems disable damping "for simplicity" and eat the accuracy cost permanently. Don't. Include ETS(*, Ad, *) in your auto search.

### 11.3 Use Multiplicative Error for Positive Series

If your series is strictly positive and has variance that scales with the level (revenue, demand, impressions), multiplicative error models produce calibrated PIs that additive models cannot match. Additive models systematically over-estimate uncertainty at low levels and under-estimate at high levels.

### 11.4 Prefer Simulation-Based PIs

Analytical PI formulas exist for additive ETS variants, but not cleanly for multiplicative variants. Run simulations (\(\ge 2000\) paths) and compute empirical quantiles. This uniformizes the code path across model families and handles non-Gaussian forecast distributions correctly. The cost is negligible for ETS; the robustness gain is large.

### 11.5 Ensembles Beat Single Models on Average

Hard-won lesson from every M-competition: ensembles win. For batch forecasting across many items, average ETS + ARIMA + Theta (equal weights). For higher-accuracy single-series work, add naive benchmarks and use stacked regression weights learned on a holdout. Expect 5–15% MASE improvement over the best single model, with no extra variance.

### 11.6 Watch Out for Very Short Series

ETS breaks (or silently returns nonsense) on series with fewer than \(2m\) observations (less than two seasons). For such series, fall back to:

- Naive-1 / seasonal naive as the primary
- A pooled/hierarchical model using information from related series
- An ad-hoc "no model" forecast with human review

Do not let an auto-ETS run loose on short series — you will ship random forecasts with falsely tight intervals.

### 11.7 Log Before Multiplicative Doesn't Just Work

The trick of "just log the series and use additive ETS" usually works, but:

- Log of zero is undefined. Shift by a small constant, or use a different model.
- Back-transforming the point forecast needs Jensen correction: \(\hat{X} = \exp(\hat{Y}) \exp(\hat{\sigma}^2 / 2)\).
- Back-transforming PIs does *not* need correction: just exponentiate the quantiles of the log-scale PI.
- The Jensen correction bias can be 1–5% on monthly data with \(\hat\sigma^2 / 2 \sim 0.01\)–0.05. Silent, persistent under-forecast.

Multiplicative-error ETS handles all this natively.

### 11.8 Monitor Smoothing Parameter Drift

Rolling refit with a schedule. Log \(\hat{\alpha}, \hat{\beta}, \hat{\gamma}, \hat{\phi}\) at each refit. Sudden large changes indicate regime shift — investigate before shipping. A stable system has \(\hat\alpha\) with CV < 20% across refits; \(\hat\phi\) usually stabilizes above 0.9 and below 0.98. Alert if any parameter jumps to an extreme (\(\hat\alpha \to 1\), \(\hat\phi \to 0\)) — usually means the model is fitting noise.

### 11.9 Do Not Mix Training and Forecast Seasonal Structure

The most common seasonal-ETS bug: training used \(m = 12\) (monthly), forecasting uses daily. Or the seasonal period was detected wrong on cold items. Always assert that `seasonal_periods` matches the actual frequency of the data, and that \(T > 2m\), before training.

### 11.10 Intermittent Demand Requires a Different Tool

ETS assumes smooth, non-zero series. For series with many zeros (slow-moving inventory SKUs), use:

- **Croston's method** (1972): separate forecasts for demand size and inter-demand intervals.
- **TSB (Teunter-Syntetos-Babai)** correction for bias.
- **Zero-inflated or count models** in a GLM framework.

Forcing ETS on intermittent demand gives systematically biased, over-smoothed forecasts that cause stockouts.

### 11.11 Use `forecast::ets` / `statsmodels.ETSModel` Correctly

Common pitfalls:

- **`model='ZZZ'`** in R auto-selects. Use it. Don't specify a model unless you have a good reason.
- **`biasadj=TRUE`** for log-transformed series corrects the Jensen gap. Default is FALSE in older versions — check.
- **`damped=NULL`** (R) or `damped_trend=None` lets auto-selection decide. Don't force a choice unless you have evidence.
- **Bootstrap PIs** (`bootstrap=TRUE`) when you suspect non-Gaussian innovations. More robust than simulated parametric PIs.

### 11.12 Benchmark Against Naive Before Shipping

Every production forecast should have a MASE computed vs. seasonal naive on a rolling-origin CV *at the forecast horizon you actually use*. A MASE \(\ge 1\) means your complex model is not improving over a method that a spreadsheet can execute — ship the spreadsheet instead, reduce complexity debt, and move on.

### 11.13 Document the Full Specification

"We use ETS" is not a spec. "We use ETS(M, Ad, M) with `seasonal_periods=12`, `damped_trend=True`, `initialization_method='estimated'`, refit monthly, PIs at 80% and 95% via 5000-path simulation, backtested on expanding-window rolling origin at horizon \(h = 1, 6, 12\)" is a spec. Write it down; make it a runnable configuration. Future-you, and every teammate, will thank you.

---

## Summary and What's Next

Exponential smoothing started as a 1957 inventory heuristic and ended as the **ETS innovations state-space family** — a fully parametric, likelihood-based, auto-selectable collection of thirty variants with closed-form forecasts and (for additive-error variants) closed-form prediction intervals. We added the damped-trend fix for long-horizon sanity and the multiplicative-seasonality variant for positive heteroskedastic series; we covered the Theta method, a decomposition-plus-SES ensemble that won M3 and is still in every production forecaster's ensemble. ETS + ARIMA + Theta (equal-weight average) is the closest thing forecasting has to a default.

In [Part 4](/2026/04/21/state-space-kalman-filtering.html) we unify ARIMA and ETS under the general state-space framework and derive the **Kalman filter** — the \(O(T)\) algorithm that underlies exact MLE in both, and the workhorse of modern structural time series models (trend + seasonal + cycle as explicit unobserved components). Part 5 finishes the series with modern practice: GARCH for volatility forecasting, gradient boosting with lag features, N-BEATS (deep MLP without attention), hierarchical reconciliation, and proper scoring rules for probabilistic forecast evaluation.
