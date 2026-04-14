---
layout: post
title: "Modern Forecasting: GARCH, Gradient Boosting, N-BEATS, and Hierarchical Reconciliation"
date: 2026-04-22
category: math
---

*This is Part 5 of a 5-part series on time series forecasting. [Part 1: Foundations](/2026/04/18/time-series-foundations-stationarity.html) | [Part 2: ARIMA](/2026/04/19/arima-box-jenkins-forecasting.html) | [Part 3: Exponential Smoothing & ETS](/2026/04/20/exponential-smoothing-ets-theta.html) | [Part 4: State-Space & Kalman](/2026/04/21/state-space-kalman-filtering.html) | **Part 5: Modern Forecasting***

The first four parts of this series built the linear-Gaussian forecasting tradition: ARIMA for memory and integration, ETS for smoothing-style state-space models, and Kalman filtering as the unifying engine. Those models cover most operational forecasting needs. But there are four important problems they don't handle well: forecasting **volatility** (the variance is itself a time series), forecasting with **rich exogenous structure** (calendar, weather, prices, promotions, item attributes), forecasting **many related series** that must aggregate consistently (sales by SKU × store × region, all rolling up), and producing **probabilistic forecasts** with proper calibration.

Each of these has a modern solution that has stabilized in industry over the last decade. **GARCH** family models (Engle 1982, Bollerslev 1986) handle conditional volatility — the workhorse of finance and risk management. **Gradient-boosted decision trees** with lag features (LightGBM, XGBoost) won the 2020 M5 competition by a wide margin and are now the industry default for retail/demand forecasting at scale. **N-BEATS** (Oreshkin et al. 2020) showed that a deep MLP architecture — no attention, no recurrence — can beat ETS+ARIMA on M3 and M4 benchmarks. **Hierarchical reconciliation** (Hyndman et al. 2011, Wickramasuriya et al. 2019) takes independent forecasts at every level of an aggregation tree and projects them back onto a coherent forecast that adds up. **Proper scoring rules** — CRPS, log-score, quantile loss — give you the right metric to evaluate probabilistic forecasts.

This post covers all five. We derive GARCH and its volatility forecasts, develop the gradient-boosting recipe with lag features and the leakage rules that make it work, sketch N-BEATS and explain why it works without attention, derive the MinT optimal reconciliation formula, define and discuss CRPS / log-score / quantile loss, and close with industry best practices for shipping these models in production.

---

## Table of Contents

1. [GARCH and Conditional Volatility](#1-garch-and-conditional-volatility)
2. [Gradient Boosting with Lag Features](#2-gradient-boosting-with-lag-features)
3. [N-BEATS: Deep Learning Without Attention](#3-n-beats-deep-learning-without-attention)
4. [Hierarchical Forecasting and Reconciliation](#4-hierarchical-forecasting-and-reconciliation)
5. [Probabilistic Forecasting and Proper Scoring Rules](#5-probabilistic-forecasting-and-proper-scoring-rules)
6. [Backtesting at Scale](#6-backtesting-at-scale)
7. [Python Pipelines](#7-python-pipelines)
8. [Industry Best Practices](#8-industry-best-practices)

---

## 1. GARCH and Conditional Volatility

Financial returns have a defining property: levels are nearly uncorrelated (markets are roughly efficient at the daily scale), but their *squares* are strongly autocorrelated. Big returns cluster — a volatile day is more likely to be followed by another volatile day. ARIMA models the conditional mean; **GARCH** models the conditional variance.

### ARCH(p)

Engle's (1982) **Autoregressive Conditional Heteroskedasticity** model:

$$
r_t = \mu + \varepsilon_t, \quad \varepsilon_t = \sigma_t z_t, \quad z_t \sim \mathcal{N}(0, 1) \text{ i.i.d.},
$$

$$
\sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i \varepsilon_{t-i}^2,
$$

with \\(\omega > 0\\) and \\(\alpha_i \ge 0\\) for stationarity. The conditional variance is a moving average of past squared shocks. Big past shocks → high current variance → fatter-tailed return distribution.

### GARCH(p, q)

Bollerslev (1986) added a moving-average analogue of past variances:

$$
\sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i \varepsilon_{t-i}^2 + \sum_{j=1}^q \beta_j \sigma_{t-j}^2.
$$

GARCH(1,1) — \\(p = q = 1\\) — is by far the most common specification in practice. The recursion

$$
\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2
$$

reads like an exponentially weighted moving average of squared shocks. **Stationarity** requires \\(\alpha + \beta < 1\\); the **unconditional variance** is \\(\omega / (1 - \alpha - \beta)\\). Most fitted equity-return GARCH(1,1) models give \\(\alpha + \beta \approx 0.97\)–\\(0.99\\), meaning volatility is highly persistent — shocks decay slowly back to the long-run level.

### Variance Forecasts

The \\(h\\)-step-ahead variance forecast is recursive:

$$
\hat{\sigma}^2_{T+1|T} = \omega + \alpha \varepsilon_T^2 + \beta \sigma_T^2,
$$

$$
\hat{\sigma}^2_{T+h|T} = \omega + (\alpha + \beta) \hat{\sigma}^2_{T+h-1|T}, \quad h \ge 2.
$$

This converges geometrically to the unconditional variance \\(\omega / (1 - \alpha - \beta)\\). Volatility *mean-reverts* — a key empirical fact and the reason GARCH dominates in option pricing and Value-at-Risk applications.

### EGARCH and the Leverage Effect

Standard GARCH treats positive and negative shocks symmetrically. Equity volatility responds asymmetrically — a 5% drop raises future volatility more than a 5% rise. Nelson's (1991) **Exponential GARCH** captures this:

$$
\log \sigma_t^2 = \omega + \alpha (|z_{t-1}| - \mathbb{E}|z_{t-1}|) + \gamma z_{t-1} + \beta \log \sigma_{t-1}^2.
$$

The asymmetry parameter \\(\gamma\\) is typically negative in equity returns, capturing the leverage effect. EGARCH and similar asymmetric variants (GJR-GARCH, TGARCH) are the production-standard for risk management.

### Why You Need GARCH Even If You Have ARIMA

ARIMA gives a constant prediction interval (under Gaussian innovations), which is wrong on returns: realized intervals widen and narrow with market regime. A GARCH-augmented model gives intervals that adapt — the same point forecast, but a 95% interval that's twice as wide on a volatile day. For Value-at-Risk, options pricing, position sizing, and risk-parity portfolios, this adaptation is the entire point.

### Python: Fit a GARCH(1,1)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

rng = np.random.default_rng(7)
T = 2000
omega, alpha, beta = 0.05, 0.10, 0.85
sigma2 = np.zeros(T)
ret = np.zeros(T)
sigma2[0] = omega / (1 - alpha - beta)
for t in range(1, T):
    sigma2[t] = omega + alpha * ret[t-1]**2 + beta * sigma2[t-1]
    ret[t] = np.sqrt(sigma2[t]) * rng.standard_normal()

m = arch_model(ret * 100, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
res = m.fit(disp='off')
print(res.summary())

# Forecast volatility 30 steps ahead
fc = res.forecast(horizon=30, reindex=False)
sigma_fc = np.sqrt(fc.variance.values[-1])
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(np.sqrt(sigma2[-200:]) * 100, color='#6db3f2', label=r'In-sample $\\sigma_t$')
ax.plot(range(200, 230), sigma_fc, color='#f2a5a5', label=r'Forecast $\\hat{\\sigma}_{T+h|T}$')
ax.set_xlabel('t'); ax.set_ylabel(r'$\\sigma_t$ (%)'); ax.legend(); ax.grid(alpha=0.3)
```

Use `vol='EGARCH'` for asymmetric models; `dist='studentst'` for fat-tailed innovations (recommended for equity returns).

---

## 2. Gradient Boosting with Lag Features

The 2020 M5 forecasting competition (Walmart hierarchical sales, 42K series, daily) was won by LightGBM ensembles by a margin large enough that the result reset industry norms. Since then, gradient-boosted trees with carefully constructed lag features are the default for high-volume, high-cardinality forecasting at retailers, ad networks, and ride-sharing platforms.

### Why GBMs Win at Scale

- **No stationarity assumption**: trees handle trend and seasonality through split rules, not through differencing.
- **Native covariate support**: prices, promotions, weather, item metadata enter as features without special treatment.
- **Cross-series learning**: a single global model trained on millions of (series, time) rows transfers patterns across items, regions, days. Local ETS / ARIMA cannot do this.
- **Robust to noisy data**: trees are insensitive to outliers and irrelevant features.
- **Cheap inference**: a fitted LightGBM model predicts millions of rows per second on commodity hardware.

### The Feature Construction Recipe

Given target \\(y_t\\) for series \\(s\\), the feature row at forecast horizon \\(h\\) is:

$$
\mathbf{x}_{s,t,h} = \big[ \underbrace{y_{s, t-h-l}}_{\text{lag features}}, \underbrace{\bar{y}_{s, t-h-l:t-h}}_{\text{rolling stats}}, \underbrace{\text{calendar}_t}_{\text{day, month, holiday}}, \underbrace{\text{covariates}_t}_{\text{price, promo, weather}}, \underbrace{\text{static}_s}_{\text{item attributes}} \big].
$$

Forecast horizon \\(h\\) is itself a feature, OR you train a separate model per horizon (the "direct" strategy — see below).

### Lag Features

For each series, include lags at 1, 7, 14, 28, 35, 56, 91, 365 days (depending on which seasonalities matter). Critical: every lag must use information available at *forecast time*, not at *fit time*. If you're forecasting day \\(t+h\\) from day \\(t\\), the lag must be \\(\ge h\\) — never use values that wouldn't be available.

### Rolling Statistics

Features like rolling mean, std, max, skew, kurtosis at windows of 7, 14, 28, 91 days. These capture local trend and volatility. Same anti-leakage rule: window must end at \\(t - h\\), not \\(t\\).

### Calendar Features

Day-of-week (one-hot), month, day-of-year, week-of-year, holiday indicators (with lead/lag for Black Friday-type effects), payday indicators. These let the tree learn weekly and seasonal effects directly.

### Encoding Static Features

For panel data with thousands of series, item ID, store, category, region are categorical features with thousands of levels. Use:

- **Native categorical support** in LightGBM (`categorical_feature=`), which optimally splits categorical variables with up to ~1000 levels.
- **Target encoding** (mean of target per category, computed only on training folds to avoid leakage) for very high-cardinality features.
- **Embeddings** via factorization machines or shallow neural networks if categorical cardinality is huge (>100K).

### Forecasting Strategy: Direct vs. Recursive

Two ways to use a model trained on one-step-ahead targets to forecast \\(h\\) steps ahead:

- **Recursive**: fit a one-step model; forecast \\(t+1\\), append to history, forecast \\(t+2\\) using the predicted \\(t+1\\), etc. Cumulative error grows with \\(h\\); fast to train.
- **Direct**: train a separate model for each horizon \\(h\\). No error accumulation; \\(H\\) times the training cost; loses cross-horizon information.

**DirRec** (Sorjamaa et al. 2007) and **MIMO** (multiple-output) hybridize. M5 winners used direct multi-horizon models with shared features.

### Anti-Leakage Discipline

The single biggest gradient-boosting forecasting bug: leakage. Specifically:

1. **Future information in features**: rolling statistics computed using future data look magical in train, terrible in test.
2. **Target encoding without out-of-fold construction**: target encoded on the full series leaks the target into features.
3. **Cross-validation that doesn't respect time**: random K-fold across rows from the same series leaks future-into-past.

**Always** do feature construction inside a function that takes (series, target_time) as input and only uses data with timestamp \\(< \text{target\\_time} - h\\). Build a unit test that confirms swapping in random future data doesn't change the feature row for any historical row.

### Quantile Regression for Probabilistic Forecasts

LightGBM supports the **quantile loss** directly:

$$
\mathcal{L}_q(y, \hat{y}) = \begin{cases} q (y - \hat{y}) & y \ge \hat{y} \\ (q - 1)(y - \hat{y}) & y < \hat{y} \end{cases}
$$

Train one model per quantile (\\(q = 0.1, 0.25, 0.5, 0.75, 0.9, 0.95\\)) to get a full predictive distribution. The M5 competition graded on the **pinball loss** averaged across quantiles — most top entries used quantile-regression LightGBMs.

### Why Not Just Use GBMs Always?

- **Short series**: < 100 observations per series gives no signal across lags. ETS dominates.
- **No covariates**: when calendar / promo / weather aren't useful, GBMs lose their main advantage.
- **Need calibrated PIs without quantile training**: parametric models (ETS, ARIMA, structural) give analytic intervals; GBMs need quantile regression or post-hoc conformal calibration.
- **Long-horizon forecasts**: GBMs degrade faster than ETS/ARIMA at horizons much beyond the longest training lag.
- **Interpretation**: trees are harder to explain in business terms than "level + trend + seasonal."

For most retail/demand problems with thousands of items and rich features: GBMs win. For univariate forecasting on a clean monthly series: ETS/ARIMA likely win.

---

## 3. N-BEATS: Deep Learning Without Attention

In 2020 Oreshkin et al. published a deep neural architecture, **N-BEATS** (Neural Basis Expansion Analysis), that beat the M4 winner (Smyl's hybrid ES-RNN) without using recurrence, attention, or any explicit time-series prior. It is purely fully-connected MLPs, stacked with residual connections and a basis-expansion interpretation. The fact that this works tells you something about how much time series forecasting depends on attention or state space — apparently very little.

### Architecture

The model takes a fixed-length **lookback window** \\(\mathbf{x} \in \mathbb{R}^L\\) and outputs a **forecast** \\(\mathbf{y} \in \mathbb{R}^H\\) of horizon \\(H\\). The network is organized in **blocks** and **stacks**:

- **Block**: a 4-layer fully-connected MLP with \\(\mathrm{ReLU}\\) activations. It outputs two vectors of expansion coefficients \\(\boldsymbol{\theta}^b\\) (backcast) and \\(\boldsymbol{\theta}^f\\) (forecast). These coefficients are passed through a **basis function** \\(g^b, g^f\\) to produce the block backcast \\(\hat{\mathbf{x}}^b = g^b(\boldsymbol{\theta}^b)\\) and forecast \\(\hat{\mathbf{y}}^b = g^f(\boldsymbol{\theta}^f)\\).
- **Residual connection**: the block input is updated as \\(\mathbf{x}_{l+1} = \mathbf{x}_l - \hat{\mathbf{x}}^b_l\\). Each subsequent block models the residual of what came before.
- **Stack**: blocks of the same basis function. The stack forecast is \\(\sum_b \hat{\mathbf{y}}^b\\).
- **Model**: stacks summed.

### Generic vs. Interpretable Basis

Two basis-function choices:

- **Generic basis**: \\(g^f(\boldsymbol{\theta}) = \mathbf{V}^f \boldsymbol{\theta}\\) — a learned linear projection. Maximum flexibility, less interpretable.
- **Interpretable basis**: stacks have separate **trend** (polynomial basis: \\(t^0, t^1, \ldots, t^d\\)) and **seasonality** (Fourier basis: \\(\sin(2\pi k t / T), \cos(2\pi k t / T)\\)) blocks. Makes the components readable.

The interpretable variant is essentially a deep version of the structural decomposition from Part 4 — explicit trend + seasonal — with neural network coefficient estimation per block.

### Why It Works

Three things do most of the work:

1. **Doubly residual stacking**: each block sees residuals not predicted by previous blocks. Equivalent to boosting in functional space.
2. **Fixed lookback / fixed horizon**: same architecture across all training samples. A single global model trained on millions of (lookback, forecast) windows from many series.
3. **Ensembling**: the M4 result used 180 N-BEATS models trained with different random seeds, lookback windows, and loss functions, averaged. Single N-BEATS is competitive; the ensemble is what wins.

### N-HiTS: Faster, Often Better

Challu et al. (2023) introduced **N-HiTS** (Hierarchical Interpolation for Time Series) which adds multi-rate signal sampling and interpolation between blocks. It's faster to train, often more accurate at long horizons, and handles multi-scale seasonality cleanly. For long-horizon forecasting, N-HiTS is the better starting point in 2026.

### Where N-BEATS / N-HiTS Fit in the Stack

These methods **win on problems where**:

- You have many series (\\(> 1000\\)) sharing latent dynamics — global training transfers structure.
- Series are long enough (\\(L > 100\\)) for fixed lookback windows.
- You want non-parametric forecasts without writing structural assumptions.
- You can afford training a deep model (GPU recommended).

They **lose on problems where**:

- Series are short or heterogeneous in length.
- Calibrated prediction intervals matter (vanilla N-BEATS is point-forecast only; quantile variants exist but underperform GBM quantile regression).
- Interpretability matters more than peak accuracy.
- You need to incorporate covariates (vanilla N-BEATS is univariate; extensions exist).

### Python Skeleton

`nixtla/neuralforecast` provides a clean N-BEATS / N-HiTS implementation:

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
from neuralforecast.losses.pytorch import MAE

models = [
    NBEATS(input_size=2*12, h=12, max_steps=500, loss=MAE()),
    NHITS(input_size=2*12, h=12, max_steps=500, loss=MAE()),
]
nf = NeuralForecast(models=models, freq='MS')
nf.fit(df=long_df)  # long_df: columns [unique_id, ds, y]
fcst = nf.predict()
```

For real production use: train an ensemble (≥ 5 seeds), add quantile-regression heads for probabilistic forecasts, and benchmark against an ETS+ARIMA+Theta ensemble before committing.

---

## 4. Hierarchical Forecasting and Reconciliation

Most operational forecasting problems have **hierarchical** structure: total sales = sum over regions; regional sales = sum over stores; store sales = sum over SKUs. You can forecast at any level. The forecasts at different levels almost never agree — sum of SKU forecasts ≠ store forecast, etc. The **reconciliation problem** is producing a coherent set of forecasts that respect the hierarchy.

### Setup

Let \\(\mathbf{y}_t \in \mathbb{R}^n\\) be all series at time \\(t\\): the leaves (bottom level) plus all aggregations. There are \\(m\\) leaves and \\(n - m\\) aggregations. The **summing matrix** \\(\mathbf{S} \in \mathbb{R}^{n \times m}\\) maps leaves to all levels:

$$
\mathbf{y}_t = \mathbf{S} \mathbf{b}_t,
$$

where \\(\mathbf{b}_t \in \mathbb{R}^m\\) are the bottom-level (leaf) values. Coherence: any forecast \\(\hat{\mathbf{y}}\\) is **coherent** iff \\(\hat{\mathbf{y}} = \mathbf{S} \tilde{\mathbf{b}}\\) for some \\(\tilde{\mathbf{b}}\\).

### Bottom-Up

Forecast each leaf, then sum: \\(\tilde{\mathbf{y}} = \mathbf{S} \hat{\mathbf{b}}\\). Pros: simple, coherent by construction. Cons: leaf-level series are noisier and harder to forecast accurately; aggregating noisy forecasts gives noisy aggregate forecasts.

### Top-Down

Forecast the aggregate, allocate to leaves by historical proportions: \\(\tilde{b}_i = p_i \hat{y}_{\text{total}}\\). Pros: aggregate is easier to forecast. Cons: assumes proportions are stable; bias when item mix shifts.

### Middle-Out

Forecast at an intermediate level (e.g., region), aggregate up (sum), disaggregate down (proportions). A compromise.

None of these uses information from all levels jointly. The modern approach does.

### MinT (Minimum Trace) Reconciliation

Hyndman and colleagues (2011, with Wickramasuriya 2019) showed there is a unique **optimal** reconciliation in the MSE sense.

Given independent base forecasts \\(\hat{\mathbf{y}} \in \mathbb{R}^n\\) at every level (from any model), the reconciled forecast is

$$
\tilde{\mathbf{y}} = \mathbf{S}(\mathbf{S}^\top \mathbf{W}^{-1} \mathbf{S})^{-1} \mathbf{S}^\top \mathbf{W}^{-1} \hat{\mathbf{y}},
$$

where \\(\mathbf{W}\\) is the covariance matrix of base-forecast errors. This is a generalized least-squares projection of the base forecasts onto the coherent subspace spanned by \\(\mathbf{S}\\). The projection minimizes the trace of the covariance of reconciled forecast errors — hence **MinT**.

### Choosing W

Wickramasuriya et al. discuss four weight matrices in increasing order of complexity:

1. **OLS**: \\(\mathbf{W} = \mathbf{I}\\). Simplest; ignores error correlations.
2. **WLS variance scaling**: \\(\mathbf{W} = \mathrm{diag}(\hat{\sigma}_i^2)\\). Diagonal, uses base-forecast error variances.
3. **WLS structural**: \\(\mathbf{W} = \mathrm{diag}(\mathbf{S} \mathbf{1})\\). Treats leaf and aggregate errors based on the number of leaves they represent.
4. **MinT shrink**: full covariance \\(\hat{\mathbf{W}}\\) of in-sample residuals, with shrinkage toward the diagonal for stability.

**MinT shrink** typically gives the largest accuracy improvement, especially on hierarchies with > 100 series.

### Why Reconciliation Helps

Even when each base forecast is unbiased, reconciliation pools information across levels. The aggregate forecast benefits from the signal in leaf forecasts (which see sub-pattern structure); leaf forecasts benefit from the aggregate's lower-noise level estimate. Improvements of 5–15% MASE are typical.

### Probabilistic Reconciliation

For probabilistic forecasts (full predictive distributions, not just points), the linear MinT projection extends — Panagiotelis et al. (2020) gives the formal framework. In practice you draw paths from each base forecast distribution, project each path through the MinT projection matrix, and the resulting paths give a coherent probabilistic forecast.

### Library Support

R: `hts` and `fable` packages. Python: `hierarchicalforecast` (Nixtla), and `pyhts`. Both implement bottom-up, top-down, middle-out, MinT, and probabilistic variants.

---

## 5. Probabilistic Forecasting and Proper Scoring Rules

A point forecast is a single number. A **probabilistic forecast** is a full predictive distribution \\(F_h\\) for the value at horizon \\(h\\). To evaluate one against realized \\(y\\), use a **scoring rule** \\(S(F, y)\\). A scoring rule is **proper** if its expectation is minimized (negatively oriented) by the true distribution: forecasting honestly is the best you can do. **Strictly proper** if uniquely so.

### Why Proper Matters

If your scoring rule is improper, the forecast that minimizes expected score is not the true distribution — it's some distortion. You will train on a proper rule and evaluate on the same, OR you will reward miscalibration. Use only proper rules.

### Three Proper Rules

**1. Logarithmic score** (negative log-likelihood):

$$
S^{\log}(F, y) = -\log f(y),
$$

where \\(f\\) is the density of \\(F\\). The MLE objective. Sensitive to tails (outlier with small \\(f(y)\\) gives huge score).

**2. Continuous Ranked Probability Score (CRPS)**:

$$
\mathrm{CRPS}(F, y) = \int_{-\infty}^{\infty} (F(z) - \mathbf{1}[y \le z])^2 \, dz.
$$

CRPS measures the integrated squared distance between the predictive CDF and the empirical CDF of the realized value (a step function). Smaller is better. Reduces to MAE for point forecasts; generalizes to distributions cleanly. Less sensitive to tails than log-score, easier to interpret in the same units as the target. **The default proper score for forecast evaluation in 2026.**

For Gaussian \\(F = \mathcal{N}(\mu, \sigma^2)\\):

$$
\mathrm{CRPS}(\mathcal{N}(\mu, \sigma^2), y) = \sigma \left[ \frac{y - \mu}{\sigma}\left(2\Phi\!\left(\frac{y-\mu}{\sigma}\right) - 1\right) + 2\phi\!\left(\frac{y-\mu}{\sigma}\right) - \frac{1}{\sqrt{\pi}}\right].
$$

**3. Quantile (pinball) loss** for a single quantile \\(q\\):

$$
\mathrm{QL}_q(F, y) = (q - \mathbf{1}[y < Q^{(q)}])(y - Q^{(q)}),
$$

where \\(Q^{(q)}\\) is the predicted \\(q\\)-quantile. Average over quantile grid \\(\lbrace 0.1, 0.5, 0.9\rbrace\\) (or finer) to evaluate a quantile forecast. M5 used quantile loss across nine quantiles.

### Calibration: PIT

Beyond scoring, check **calibration** with the **Probability Integral Transform**:

$$
u_t = F_t(y_t).
$$

If \\(F_t\\) is correct, the \\(u_t\\) should be Uniform\\([0, 1]\\) — under the null of perfect calibration. Histogram of PIT values should look flat. U-shaped histogram = under-dispersed (intervals too tight). Inverted-U = over-dispersed. Skew = location bias.

Pair PIT diagnostics with rolling coverage check (Section 11 of Part 1): the reliable production tests for forecast calibration.

### Reliability Diagrams

For probabilistic-binary or quantile forecasts, plot predicted vs. realized frequency. Production forecasts should fall on the 45° line. Deviations indicate systematic miscalibration that no single accuracy metric will catch.

---

## 6. Backtesting at Scale

For a single series, rolling-origin CV (Part 1, Section 11) is the standard. For thousands of series, the protocol scales but adds two concerns:

### Aggregating Across Series

Different series have different scales, different lengths, different difficulty. Pooled MAE is dominated by big-volume items. Better aggregates:

- **MASE per series, then median or mean across series.** MASE normalizes by the seasonal naive error, making per-item errors comparable.
- **Item-weighted vs. equal-weighted.** A retailer cares more about high-volume items; equal-weight gives small items disproportionate influence on the aggregate.
- **Quantile of MASE** (e.g., 90th percentile): captures tail performance — usually what stakeholders complain about.

### Compute Budget

Refit \\(N\\) models \\(\times\\) \\(K\\) windows is expensive at scale. Tactics:

- **Sub-sample series for CV**: pick a stratified sample (e.g., 10% across volume bins), CV on those, assume conclusions generalize.
- **Last-windows-only**: instead of full rolling-origin, evaluate on the last \\(K\\) windows. Faster but less robust.
- **Cached features**: feature construction often dominates CV runtime; cache lag features per series and recompute only when a CV window crosses a lag.

### Stability vs. Accuracy

Two models with similar mean MASE may have very different *variance* of MASE across CV windows. The stable model is usually the better production choice — predictable error properties make alerting and remediation easier. Report std of MASE alongside the mean.

---

## 7. Python Pipelines

We illustrate three pipelines: GARCH for volatility, LightGBM for hierarchical demand, and MinT reconciliation across a small hierarchy.

### Volatility Forecasting with GARCH

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# Simulate GARCH(1,1)
rng = np.random.default_rng(42)
T = 3000
omega, alpha, beta = 0.05, 0.08, 0.90
sigma2 = np.full(T, omega/(1-alpha-beta))
ret = np.zeros(T)
for t in range(1, T):
    sigma2[t] = omega + alpha * ret[t-1]**2 + beta * sigma2[t-1]
    ret[t] = np.sqrt(sigma2[t]) * rng.standard_normal()

# Fit
m = arch_model(ret*100, mean='Constant', vol='GARCH', p=1, q=1, dist='studentst')
res = m.fit(disp='off')
print(res.summary().tables[1])

# 30-day vol forecast
fc = res.forecast(horizon=30, reindex=False)
sig_fc = np.sqrt(fc.variance.values[-1])

# VaR at 5%
nu = res.params['nu']  # student-t df
from scipy.stats import t
var_5 = -res.params['mu'] - sig_fc * t.ppf(0.05, df=nu)

fig, axes = plt.subplots(2, 1, figsize=(11, 6))
axes[0].plot(np.sqrt(sigma2[-300:])*100, color='#6db3f2', label='Realized vol')
axes[0].plot(range(300, 330), sig_fc, color='#f2a5a5', label='Forecast vol')
axes[0].set_ylabel(r'$\\sigma_t$ (%)'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(range(300, 330), var_5, color='#f2c894')
axes[1].set_ylabel('5% VaR (%)'); axes[1].set_xlabel('horizon'); axes[1].grid(alpha=0.3)
fig.tight_layout()
```

### LightGBM with Lag Features

```python
import lightgbm as lgb
import pandas as pd
import numpy as np

def build_features(df, target='y', max_lag=28, rolling_windows=(7, 28)):
    df = df.sort_values(['series_id', 'ds']).copy()
    g = df.groupby('series_id')[target]
    for L in [1, 7, 14, 28]:
        df[f'lag_{L}'] = g.shift(L)
    for w in rolling_windows:
        df[f'rmean_{w}'] = g.shift(1).rolling(w).mean().reset_index(level=0, drop=True)
        df[f'rstd_{w}'] = g.shift(1).rolling(w).std().reset_index(level=0, drop=True)
    df['dow'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    df['weekofyear'] = df['ds'].dt.isocalendar().week.astype(int)
    return df

# df: long-format with columns [series_id, ds, y, price, promo, ...]
df_feat = build_features(df).dropna()
features = [c for c in df_feat.columns if c not in ('series_id', 'ds', 'y')]
cutoff = df_feat['ds'].max() - pd.Timedelta(days=28)
train = df_feat[df_feat['ds'] <= cutoff]
val   = df_feat[df_feat['ds'] >  cutoff]

dtrain = lgb.Dataset(train[features], label=train['y'],
                     categorical_feature=['series_id', 'dow', 'month'])
dval   = lgb.Dataset(val[features],   label=val['y'])

params = dict(
    objective='tweedie', tweedie_variance_power=1.1,  # for non-negative skewed counts
    learning_rate=0.05, num_leaves=128,
    min_data_in_leaf=200, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5,
    metric='mae', verbose=-1,
)
booster = lgb.train(params, dtrain, num_boost_round=5000,
                    valid_sets=[dtrain, dval],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
```

For probabilistic forecasts: train one booster per quantile with `objective='quantile', alpha=q`. Concatenate to form a quantile forecast.

### Hierarchical Reconciliation (MinT)

```python
import numpy as np
from scipy.linalg import lstsq

def mint_reconcile(yhat, S, W):
    """yhat: (n,) base forecasts at all levels.
       S: (n, m) summing matrix.
       W: (n, n) base-forecast error covariance.
       Returns reconciled (n,) coherent forecast."""
    Wi = np.linalg.inv(W)
    M  = S @ np.linalg.inv(S.T @ Wi @ S) @ S.T @ Wi
    return M @ yhat

# Toy hierarchy: total -> {A, B}; A -> {a1, a2}; B -> {b1, b2}
# Bottom-level series order: a1, a2, b1, b2
m = 4
S = np.array([
    [1, 1, 1, 1],   # total
    [1, 1, 0, 0],   # A
    [0, 0, 1, 1],   # B
    [1, 0, 0, 0],   # a1
    [0, 1, 0, 0],   # a2
    [0, 0, 1, 0],   # b1
    [0, 0, 0, 1],   # b2
])  # n = 7

# Suppose base forecasts (incoherent)
yhat = np.array([100, 55, 40, 30, 23, 18, 25], dtype=float)
print('sum of A children =', yhat[3] + yhat[4], 'vs A =', yhat[1])  # 53 vs 55
W = np.diag([4.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])  # variances
y_rec = mint_reconcile(yhat, S, W)
print('reconciled total =', y_rec[0])
print('reconciled A =', y_rec[1], 'vs sum of children =', y_rec[3] + y_rec[4])
```

### CRPS for a Gaussian Forecast

```python
from scipy.stats import norm

def crps_gaussian(y, mu, sigma):
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1/np.sqrt(np.pi))

# Empirical CRPS from a sample
def crps_empirical(samples, y):
    # samples: (n_samples,), y: scalar
    s = np.sort(samples)
    n = len(s)
    return np.mean(np.abs(s - y)) - 0.5 * np.mean(np.abs(s[:, None] - s[None, :]))
```

`crps_empirical` is \\(O(n^2)\\) — use it for evaluation, not training.

### PIT Histogram

```python
def pit_histogram(forecast_cdfs, y_true, bins=20):
    pit = np.array([F(y) for F, y in zip(forecast_cdfs, y_true)])
    plt.hist(pit, bins=bins, edgecolor='black')
    plt.axhline(len(pit) / bins, color='red', linestyle='--', label='Uniform expectation')
    plt.xlabel('PIT'); plt.legend()
```

A flat histogram = calibrated. U-shape = under-dispersed (intervals too tight). Inverted-U = over-dispersed.

---

## 8. Industry Best Practices

### 8.1 Choose the Right Tool for the Problem Class

Map the problem to the tool:

| Problem | Best Choice |
|---|---|
| Univariate, monthly, < 200 obs | ETS / Auto-ARIMA / Theta ensemble |
| Univariate, daily, > 1 year | SARIMA + ETS ensemble; structural model for interpretation |
| Volatility forecasting | GARCH (often EGARCH with Student-t) |
| Many series + covariates + scale | LightGBM / XGBoost with lag features |
| Many series, no covariates, deep learning available | N-BEATS / N-HiTS ensemble |
| Hierarchical structure | Per-level model + MinT reconciliation |
| Need full predictive distribution | Quantile regression GBM, or Bayesian structural time series |

### 8.2 Always Benchmark Against Naive

Even with a sophisticated stack: report MASE against seasonal naive on a rolling-origin CV. If your model doesn't beat seasonal naive by a meaningful margin (\\(> 5\%\\) on the metric you actually use), simplify.

### 8.3 Ensemble Across Model Classes

Across M3, M4, M5: equal-weight ensembling across heterogeneous models (ETS + ARIMA + Theta + LightGBM + N-BEATS) consistently beats any single model. Cost is minimal once each component is built. **Don't ship a single model when you can ship an ensemble.**

### 8.4 Anti-Leakage Discipline Is Non-Negotiable

For tree models with engineered features, leakage is the #1 source of "looks great, fails in production." Build features through a function that takes (series, target_time, horizon) and only uses data with timestamp \\(\le \text{target\\_time} - \text{horizon}\\). Unit test by feeding it future data and confirming features at past timestamps don't change.

### 8.5 Quantile Forecasts Require Quantile Loss

Training a model on MAE/RMSE and post-hoc fitting a Gaussian for intervals gives miscalibrated PIs on real data. Use quantile loss directly during training; for tree models, train a separate booster per quantile. CRPS as the evaluation metric, quantile loss as the training metric, PIT histogram as the calibration diagnostic.

### 8.6 Reconcile When You Have a Hierarchy

If your forecasts are consumed at multiple aggregation levels (regional planners, store managers, item buyers), they MUST be coherent — sum of child forecasts = parent forecast. MinT reconciliation typically delivers 5–15% accuracy improvement on top of free coherence. Use it.

### 8.7 Refit on a Schedule, Monitor Always

ETS, ARIMA, GBM, N-BEATS — all need periodic refit. Cadence depends on data velocity:

- High-frequency intraday: refit hourly, monitor model output continuously.
- Daily retail: refit weekly, monitor coverage and bias daily.
- Monthly business KPIs: refit monthly, review residuals quarterly.

Alerts: rolling MAE drift, coverage drift, parameter drift, residual autocorrelation appearing.

### 8.8 Probabilistic Calibration Beats Point Accuracy

For business decisions, the 90th-percentile forecast is what determines stock orders, capacity provisioning, and risk capital. A model with marginally worse point accuracy but well-calibrated tails is more valuable than a model with great point accuracy but tight intervals. Optimize for the metric your decision uses — usually a quantile loss or CRPS, not MAE.

### 8.9 Don't Forget Backwards Tests for Regime Stability

A model fit on 2014–2019 retail data, deployed in 2020, misbehaved spectacularly (COVID). The forecasting community absorbed this lesson: include regime-aware diagnostics:

- Rolling MAE plotted over the entire history (does the model degrade in any specific regime?).
- Coverage in each calendar quarter (does any quarter under-cover?).
- Parameter trajectories across refits (is the model's view of seasonality changing?).

These diagnostics are cheap; they catch model drift years before the next regime shock.

### 8.10 GBMs Need Robust Categorical Handling

LightGBM categorical splits are powerful but unstable on rare categories. For an item with three observations in training, the categorical split may overfit. Mitigations: minimum-occurrences thresholding, target encoding with smoothing, hashing for very high-cardinality features. Always cross-validate the categorical handling.

### 8.11 N-BEATS / Deep Models Need Real Validation Discipline

Easy to overfit a deep model on a few hundred series. Always:

- Hold out the last \\(H\\) observations of every series for validation.
- Early-stop on validation MASE (not MSE — scale-invariant metric matches what you ship).
- Ensemble across seeds (\\(\ge 5\\)) — single-seed runs have meaningfully different error rates.
- Beat a strong baseline (ETS+ARIMA ensemble, or LightGBM) before deploying.

### 8.12 Document the Forecasting System, Not Just the Model

A production forecasting system has: data pipeline, feature pipeline, model(s), reconciliation, evaluation harness, alerting. Each must be documented. Most "model failure" outages come from upstream data changes or pipeline drift, not from the model itself. The model is one component; the system is the deliverable.

### 8.13 Forecast for the Decision

The single highest-leverage shift in any forecasting team's maturity: stop optimizing the model in isolation; start optimizing the *decision* the forecast informs. Inventory ordering responds to the 95th percentile of demand; capacity planning to the 80th; pricing to the median. Build the metric, the loss, and the evaluation around the decision. The model that minimizes inventory costs is not the model that minimizes MAE.

---

## Summary and the Series

Across five posts we have built the modern forecaster's toolkit:

- **Part 1**: stationarity, ACF/PACF, white noise, random walk, Wold decomposition, ergodicity, trend/seasonality preprocessing, ADF/KPSS/Ljung-Box, and the discipline of plotting and benchmarking before modeling.
- **Part 2**: AR/MA/ARMA/ARIMA/SARIMA, Box-Jenkins identification, MLE/AICc/BIC, the recursive forecasting formula, parametric and bootstrap prediction intervals.
- **Part 3**: SES, Holt, damped trend, Holt-Winters, the ETS innovations state-space family, AICc selection across thirty variants, the Theta method, and the empirical superiority of model averaging.
- **Part 4**: the linear Gaussian state-space model, the Kalman filter and RTS smoother, exact MLE via the prediction error decomposition, structural decomposition into level + trend + seasonal + cycle, and the unification of ARIMA and ETS under one framework.
- **Part 5**: GARCH for volatility, gradient-boosted trees with lag features for scale, N-BEATS for global learning across many series, hierarchical reconciliation with MinT, and proper scoring rules / calibration for probabilistic forecasts.

Throughout, the same operational discipline:

1. Plot the series before anything else.
2. Establish a benchmark (seasonal naive) and report MASE.
3. Use rolling-origin cross-validation, not random splits.
4. Report intervals and monitor coverage in production.
5. Ensemble across heterogeneous models — it almost always wins.
6. Forecast for the decision the forecast informs, not in isolation.

Forecasting is one of those rare technical disciplines where the boring practices buy you most of the accuracy. The mathematics matters; the engineering matters more; the discipline of evaluating against the right benchmark on the right metric on the right rolling-origin schedule matters most. Master that, and the model choice becomes a nearly-secondary detail.

This is the end of the series. Pick a forecasting problem at work; fit Auto-ETS, Auto-ARIMA, and Theta; ensemble them; backtest with rolling origin and report MASE; monitor coverage in production. You now have the vocabulary, the algorithms, and the practices to do it well.
