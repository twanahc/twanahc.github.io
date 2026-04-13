---
layout: post
title: "Causal Demand Estimation: From Endogeneity to Double Machine Learning"
date: 2026-04-16
category: business
---

*This is Part 4 of a 5-part series on pricing strategy. [Part 1: Demand, Elasticity & Markup](/2026/04/13/demand-elasticity-monopolist-markup.html) | [Part 2: Price Discrimination](/2026/04/14/price-discrimination-extracting-surplus.html) | [Part 3: Game Theory of Pricing](/2026/04/15/game-theory-competitive-pricing.html) | **Part 4: Causal Demand Estimation** | [Part 5: Algorithmic Dynamic Pricing](/2026/04/17/algorithmic-dynamic-pricing-bandits.html)*

Parts 1 through 3 of this series built a beautiful theoretical tower. The Lerner index tells you the optimal markup given elasticity. Price discrimination theory tells you how to segment consumers and capture surplus. Game theory tells you how to set prices when competitors react. Every result depends on one critical input: the demand function. We wrote \\(Q(P)\\) or \\(Q = a - bP\\) and proceeded to optimize. But where does this function come from?

In practice, you have historical data. Prices you charged. Quantities you sold. Maybe some information about the time of year, what your competitors were doing, and whether it rained. The obvious approach is to regress quantity on price and read off the slope. This approach is almost always wrong. Not a little wrong. Sometimes it gives you the wrong sign --- it tells you that raising prices *increases* demand. Understanding why this happens and how to fix it is arguably the most important technical skill in applied pricing. Get it wrong and every downstream optimization is garbage in, garbage out.

The problem has a name: **endogeneity**. The classical fix is instrumental variables, a technique from the 1920s that remains a workhorse. But the cutting edge has moved far beyond, merging flexible machine learning with the rigorous causal framework of econometrics. The result is **Double Machine Learning** (DML), a method that lets you use random forests, gradient boosting, or neural networks to control for confounders while still producing valid causal estimates with proper confidence intervals. This post builds the entire pipeline from scratch.

---

## Table of Contents

1. [The Demand Estimation Problem](#the-demand-estimation-problem)
2. [Why Naive Regression Fails: Endogeneity](#why-naive-regression-fails-endogeneity)
3. [The Simultaneity Bias](#the-simultaneity-bias)
4. [Instrumental Variables --- The Classical Fix](#instrumental-variables--the-classical-fix)
5. [Two-Stage Least Squares (2SLS)](#two-stage-least-squares-2sls)
6. [The Frisch-Waugh-Lovell Theorem](#the-frisch-waugh-lovell-theorem)
7. [From Linear Models to Machine Learning](#from-linear-models-to-machine-learning)
8. [Double Machine Learning (DML)](#double-machine-learning-dml)
9. [Cross-Fitting and Sample Splitting](#cross-fitting-and-sample-splitting)
10. [The Interactive Model and Beyond Partial Linearity](#the-interactive-model-and-beyond-partial-linearity)
11. [Causal Forests for Heterogeneous Price Sensitivity](#causal-forests-for-heterogeneous-price-sensitivity)
12. [Difference-in-Differences for Pricing Policy Evaluation](#difference-in-differences-for-pricing-policy-evaluation)
13. [The BLP Model --- Structural Demand Estimation at Scale](#the-blp-model--structural-demand-estimation-at-scale)
14. [Python --- Causal Forest for Heterogeneous Elasticity](#python--causal-forest-for-heterogeneous-elasticity)
15. [Demand Censoring](#demand-censoring)
16. [Conjoint Analysis and Discrete Choice Models](#conjoint-analysis-and-discrete-choice-models)
17. [Python Implementation](#python-implementation)
18. [The State of the Art](#the-state-of-the-art)

---

## The Demand Estimation Problem

We want to estimate the relationship between price and quantity demanded. Write it as:

$$Q = f(P, X) + \varepsilon$$

where \\(P\\) is the price, \\(X\\) is a vector of **observables** --- everything else we can measure that might affect demand (product features, season, day of week, weather, advertising spend, competitor prices) --- and \\(\varepsilon\\) is the **unobserved demand shock**. This last term captures everything that affects demand but that we cannot see in our data: a viral TikTok video about the product, a shift in consumer sentiment, an unrecorded promotion.

What we specifically want is the **causal effect** of \\(P\\) on \\(Q\\). Not the correlation. The causal effect. If we reached into the world and changed the price by 1%, holding everything else fixed, how much would quantity change? This is the price elasticity:

$$\varepsilon_P = \frac{\partial Q}{\partial P} \cdot \frac{P}{Q}$$

This elasticity is the input to the Lerner index from Part 1, to the price discrimination schemes from Part 2, and to the equilibrium calculations from Part 3. It is the foundation of everything.

The problem: in observational data --- the transaction records sitting in your database --- prices are not randomly assigned. They are *set* by a profit-maximizing firm. The product manager, the pricing algorithm, or the CEO looked at the state of the world (including information we might not have in our dataset) and chose a price to maximize revenue or profit. This means the price in our data is a *function* of demand conditions, which means it is correlated with the demand shock \\(\varepsilon\\). And that correlation is where everything goes sideways.

---

## Why Naive Regression Fails: Endogeneity

**Endogeneity** means that an explanatory variable in your regression is correlated with the error term. In our context, it means:

$$\mathbb{E}[\varepsilon \mid P] \neq 0$$

The price is not independent of the things affecting demand that we cannot observe. There are three classical sources of endogeneity. All three are relevant to demand estimation.

**Source 1: Omitted Variable Bias.** Suppose a variable affects both price and quantity, and you do not include it in your regression. The classic example: ice cream. Ice cream sales spike in summer. Ice cream prices also tend to be higher in summer (higher demand, higher input costs, seasonal pricing). If you regress sales on price without controlling for season, you see high prices coinciding with high sales. The naive estimate of the price coefficient comes out positive --- it looks like raising prices increases demand. The omitted variable is temperature (or season), which drives both price and quantity upward simultaneously.

Here is a subtler example. Compare MacBooks and Chromebooks. MacBooks are expensive. In many markets, Apple sells enormous volumes of MacBooks. Chromebooks are cheap. Their volumes are often lower. A naive cross-sectional regression of quantity on price across product categories would suggest that higher prices lead to higher sales. The omitted variable is product quality, brand value, ecosystem lock-in --- all the things that simultaneously justify higher prices and generate higher demand.

**Source 2: Simultaneity.** Price and quantity are determined simultaneously in a market. Demand depends on price, but price also depends on demand (through the firm's pricing rule or through market equilibrium). We will dedicate the next section to this because it is the deepest source of endogeneity in demand estimation.

**Source 3: Measurement Error.** If price is measured with error, classical errors-in-variables theory tells us the coefficient estimate is attenuated (biased toward zero). In practice, measurement error in price is less common than the other two sources, but it matters in markets where the effective price differs from the listed price (think coupons, negotiated discounts, bundled pricing).

Formally, recall that the OLS estimator for a simple regression \\(Q = \alpha + \beta P + \varepsilon\\) is:

$$\hat{\beta}_{OLS} = \frac{\text{Cov}(P, Q)}{\text{Var}(P)} = \beta + \frac{\text{Cov}(P, \varepsilon)}{\text{Var}(P)}$$

The second term is the **bias**. It equals zero only if \\(\text{Cov}(P, \varepsilon) = 0\\). When prices are high precisely when demand is high (which is how rational firms price), \\(\text{Cov}(P, \varepsilon) > 0\\), so the bias is positive. Since the true \\(\beta\\) is negative (higher prices reduce demand), a positive bias pushes the estimate upward --- making it less negative, or even positive. You underestimate price sensitivity. In some cases, you conclude that customers *want* to pay more. This is not a theoretical curiosity; it happens routinely with real-world pricing data.

---

## The Simultaneity Bias

This deserves its own section because it is the heart of the identification problem in demand estimation and because the geometric intuition is beautiful.

In a market, there are two curves: demand and supply. Demand says how much consumers want to buy at each price. Supply says how much producers want to sell at each price. The observed price and quantity are determined by the intersection --- the **equilibrium**.

Write the system:

$$\text{Demand:} \quad Q = \alpha_0 + \alpha_1 P + \varepsilon_d$$

$$\text{Supply:} \quad Q = \beta_0 + \beta_1 P + \varepsilon_s$$

where \\(\alpha_1 < 0\\) (demand slopes down) and \\(\beta_1 > 0\\) (supply slopes up). The terms \\(\varepsilon_d\\) and \\(\varepsilon_s\\) are demand and supply shocks respectively.

Setting the two equal gives the equilibrium price:

$$P^* = \frac{\alpha_0 - \beta_0 + \varepsilon_d - \varepsilon_s}{\beta_1 - \alpha_1}$$

Notice that \\(P^*\\) depends on both \\(\varepsilon_d\\) and \\(\varepsilon_s\\). Substituting back gives \\(Q^*\\), which also depends on both shocks. So the observed data point \\((P^*, Q^*)\\) is a function of both the demand shock and the supply shock.

Now here is the key insight. Suppose we observe many data points over time.

- When the **demand curve shifts** (\\(\varepsilon_d\\) changes, \\(\varepsilon_s\\) is constant), the equilibrium moves along the supply curve. You trace out the supply curve, not the demand curve.
- When the **supply curve shifts** (\\(\varepsilon_s\\) changes, \\(\varepsilon_d\\) is constant), the equilibrium moves along the demand curve. Now you trace the demand curve.
- When **both shift simultaneously**, the observed points trace out neither curve. They scatter across the \\((P, Q)\\) plane, and a regression through those scattered points gives you a meaningless hybrid.

<svg viewBox="0 0 720 440" xmlns="http://www.w3.org/2000/svg" style="max-width: 720px; display: block; margin: 2em auto;">
  <defs>
    <marker id="arrow-s" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#d4d4d4"/>
    </marker>
  </defs>
  <text x="360" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" font-weight="bold" fill="#d4d4d4">Simultaneity: Which Curve Are You Tracing?</text>

  <!-- Left panel: Supply shifts -->
  <text x="180" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#d4d4d4">Supply shifts &#x2192; trace Demand</text>
  <!-- Axes -->
  <line x1="60" y1="380" x2="320" y2="380" stroke="#d4d4d4" stroke-width="1.5" marker-end="url(#arrow-s)"/>
  <line x1="60" y1="380" x2="60" y2="70" stroke="#d4d4d4" stroke-width="1.5" marker-end="url(#arrow-s)"/>
  <text x="190" y="415" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#d4d4d4">Quantity Q</text>
  <text x="30" y="225" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#d4d4d4" transform="rotate(-90, 30, 225)">Price P</text>
  <!-- Demand curve (fixed) -->
  <line x1="80" y1="100" x2="290" y2="350" stroke="#4fc3f7" stroke-width="2.5"/>
  <text x="295" y="358" font-family="Arial, sans-serif" font-size="12" fill="#4fc3f7" font-weight="bold">D</text>
  <!-- Supply curves (shifting) -->
  <line x1="80" y1="340" x2="250" y2="120" stroke="#e57373" stroke-width="1.5" stroke-dasharray="6,3"/>
  <text x="255" y="115" font-family="Arial, sans-serif" font-size="11" fill="#e57373">S&#x2081;</text>
  <line x1="120" y1="340" x2="290" y2="120" stroke="#e57373" stroke-width="1.5" stroke-dasharray="6,3"/>
  <text x="295" y="115" font-family="Arial, sans-serif" font-size="11" fill="#e57373">S&#x2082;</text>
  <line x1="160" y1="340" x2="310" y2="160" stroke="#e57373" stroke-width="1.5" stroke-dasharray="6,3"/>
  <text x="315" y="155" font-family="Arial, sans-serif" font-size="11" fill="#e57373">S&#x2083;</text>
  <!-- Equilibrium dots on demand curve -->
  <circle cx="145" cy="175" r="5" fill="#66bb6a"/>
  <circle cx="185" cy="220" r="5" fill="#66bb6a"/>
  <circle cx="220" cy="265" r="5" fill="#66bb6a"/>

  <!-- Right panel: Both shift -->
  <text x="540" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#d4d4d4">Both shift &#x2192; trace neither</text>
  <!-- Axes -->
  <line x1="400" y1="380" x2="660" y2="380" stroke="#d4d4d4" stroke-width="1.5" marker-end="url(#arrow-s)"/>
  <line x1="400" y1="380" x2="400" y2="70" stroke="#d4d4d4" stroke-width="1.5" marker-end="url(#arrow-s)"/>
  <text x="530" y="415" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#d4d4d4">Quantity Q</text>
  <text x="370" y="225" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#d4d4d4" transform="rotate(-90, 370, 225)">Price P</text>
  <!-- Multiple D and S curves -->
  <line x1="420" y1="100" x2="600" y2="320" stroke="#4fc3f7" stroke-width="1.2" stroke-dasharray="4,3"/>
  <line x1="450" y1="90" x2="630" y2="310" stroke="#4fc3f7" stroke-width="1.2" stroke-dasharray="4,3"/>
  <line x1="420" y1="330" x2="570" y2="120" stroke="#e57373" stroke-width="1.2" stroke-dasharray="4,3"/>
  <line x1="460" y1="340" x2="620" y2="110" stroke="#e57373" stroke-width="1.2" stroke-dasharray="4,3"/>
  <line x1="500" y1="340" x2="650" y2="130" stroke="#e57373" stroke-width="1.2" stroke-dasharray="4,3"/>
  <!-- Scattered equilibrium dots -->
  <circle cx="490" cy="200" r="5" fill="#66bb6a"/>
  <circle cx="530" cy="170" r="5" fill="#66bb6a"/>
  <circle cx="555" cy="230" r="5" fill="#66bb6a"/>
  <circle cx="510" cy="250" r="5" fill="#66bb6a"/>
  <circle cx="575" cy="195" r="5" fill="#66bb6a"/>
  <!-- Naive regression line -->
  <line x1="470" y1="270" x2="590" y2="170" stroke="#ffd54f" stroke-width="2" stroke-dasharray="8,4"/>
  <text x="600" y="165" font-family="Arial, sans-serif" font-size="11" fill="#ffd54f">Naive OLS</text>
</svg>

The left panel shows the situation we want: supply shifts while demand stays put. The green equilibrium dots trace out the demand curve, and a regression through them recovers the demand slope. The right panel shows reality: both curves shift, the equilibrium points scatter, and the naive OLS line (yellow dashed) is some meaningless weighted average of the two slopes.

This is why **identification** matters. To estimate the demand curve, you need variation that shifts supply but not demand. If you can find such variation, you can trace out the demand curve from the data.

---

## Instrumental Variables --- The Classical Fix

The solution is to find a variable \\(Z\\) --- called an **instrument** --- that provides exogenous variation in price. Specifically, \\(Z\\) must satisfy two conditions:

1. **Relevance**: \\(\text{Cov}(Z, P) \neq 0\\). The instrument must actually affect prices.
2. **Exclusion restriction**: \\(\text{Cov}(Z, \varepsilon) = 0\\). The instrument affects quantity *only* through its effect on price, not directly.

The intuition: the instrument gives us variation in price that is "as good as random" from demand's perspective. By using only this variation to estimate the price-quantity relationship, we isolate the causal effect.

In the supply-demand framework, an instrument for demand estimation is something that shifts the supply curve without shifting demand. Classic examples:

**Cost shifters.** Input costs affect the supply side --- they change the cost of producing the good --- but (in principle) do not directly affect consumers' willingness to pay. Fuel prices for airlines, commodity prices for food manufacturers, exchange rates for importers, tax changes. When fuel prices spike, airlines raise ticket prices. If the fuel price shock does not directly change travelers' desire to fly (it changes their cost of driving, which is a subtlety we will ignore for now), then fuel prices are a valid instrument.

**Hausman instruments.** Prices of the same product in *other* geographic markets. The idea: a cost shock hits Coca-Cola nationally (e.g., a change in sugar prices), so Coke prices in Denver and Boston move together. But a local demand shock in Denver (say, a heatwave) should not directly affect demand in Boston. So the price of Coke in Boston is correlated with the price of Coke in Denver (through the common cost shock) but uncorrelated with Denver-specific demand shocks.

**BLP instruments.** Berry, Levinsohn, and Pakes (1995) proposed using the characteristics of competing products as instruments. The logic: the number and type of competitors a firm faces affects its pricing (through competitive pressure), but the characteristics of other firms' products should not directly affect demand for the focal product (conditional on the focal product's own characteristics).

For the simple case with one endogenous variable and one instrument, the IV estimator is:

$$\hat{\beta}_{IV} = \frac{\text{Cov}(Z, Q)}{\text{Cov}(Z, P)}$$

Why does this work? If \\(Z\\) satisfies the exclusion restriction, the only reason \\(Z\\) and \\(Q\\) are correlated is through the causal chain \\(Z \to P \to Q\\). So:

$$\text{Cov}(Z, Q) = \text{Cov}(Z, \beta P + \varepsilon) = \beta \cdot \text{Cov}(Z, P) + \text{Cov}(Z, \varepsilon) = \beta \cdot \text{Cov}(Z, P)$$

Dividing both sides by \\(\text{Cov}(Z, P)\\) gives us \\(\beta\\). Clean, elegant, and powerful --- provided the two conditions hold.

The relevance condition is testable. The exclusion restriction is not. You can never prove from data alone that your instrument does not have a direct effect on demand. This is the Achilles' heel of instrumental variables: the most important assumption is the one you cannot test.

---

## Two-Stage Least Squares (2SLS)

When you have multiple instruments and control variables, the workhorse IV procedure is **Two-Stage Least Squares**.

Set up: We want to estimate:

$$Q_i = \theta P_i + X_i'\gamma + \varepsilon_i$$

where \\(P\\) is endogenous, \\(X\\) is a vector of exogenous controls, and we have instruments \\(Z\\) (which may be a vector).

**Stage 1: Purify the price.** Regress \\(P\\) on the instruments \\(Z\\) and the controls \\(X\\):

$$P_i = Z_i'\delta + X_i'\pi + v_i$$

Compute the predicted (fitted) value \\(\hat{P}_i = Z_i'\hat{\delta} + X_i'\hat{\pi}\\). This predicted price contains only the variation in \\(P\\) explained by the exogenous instruments and controls. All the endogenous variation --- the part correlated with the demand shock \\(\varepsilon\\) --- is in the residual \\(\hat{v}_i\\), which we discard.

**Stage 2: Estimate the structural equation.** Regress \\(Q\\) on \\(\hat{P}\\) and \\(X\\):

$$Q_i = \theta \hat{P}_i + X_i'\gamma + u_i$$

The coefficient \\(\hat{\theta}\\) on \\(\hat{P}\\) is the 2SLS estimate of the causal effect of price on quantity.

**Why this works.** Because \\(\hat{P}\\) is constructed from exogenous variables only, it is by construction uncorrelated with \\(\varepsilon\\) (assuming the exclusion restriction holds). So the OLS regression in Stage 2 is unbiased.

More formally, the 2SLS estimator is consistent:

$$\hat{\theta}_{2SLS} \xrightarrow{p} \theta \quad \text{as } n \to \infty$$

provided the instruments are relevant and satisfy the exclusion restriction.

**Diagnostics.** Two critical tests:

1. **First-stage F-statistic.** Test whether the instruments actually predict price. The rule of thumb (Staiger and Stock, 1997): the F-statistic from the first-stage regression should be well above 10. If \\(F < 10\\), the instruments are "weak," and the 2SLS estimator is badly biased --- often worse than OLS. Weak instruments are one of the most common problems in applied IV work.

2. **Overidentification test (Sargan/Hansen).** If you have more instruments than endogenous variables, you can test whether the instruments are consistent with each other. Under the null that all instruments are valid, the Sargan statistic is distributed \\(\chi^2\\) with degrees of freedom equal to the number of overidentifying restrictions. A rejection suggests at least one instrument violates the exclusion restriction.

The fundamental weakness of 2SLS: it relies on linear models in both stages. If the true relationships are nonlinear, the first stage misses variation, and the second stage imposes a functional form that may not hold. This is where machine learning enters.

---

## The Frisch-Waugh-Lovell Theorem

Before we get to Double Machine Learning, we need a theorem that provides the conceptual bridge from classical econometrics to ML-based causal inference. The **Frisch-Waugh-Lovell (FWL) theorem** is one of the most elegant results in regression theory and it underpins the entire DML framework.

**Statement.** Consider the regression:

$$Y = X_1 \beta_1 + X_2 \beta_2 + \varepsilon$$

The coefficient \\(\beta_1\\) is identical to the coefficient obtained from the following two-step procedure:

1. Regress \\(Y\\) on \\(X_2\\) and compute the residual \\(\tilde{Y} = Y - X_2 \hat{\gamma}_Y\\)
2. Regress \\(X_1\\) on \\(X_2\\) and compute the residual \\(\tilde{X}_1 = X_1 - X_2 \hat{\gamma}_X\\)
3. Regress \\(\tilde{Y}\\) on \\(\tilde{X}_1\\). The coefficient is exactly \\(\hat{\beta}_1\\).

In words: **controlling for confounders is equivalent to removing their influence from both sides first, then estimating the relationship on what remains.**

**Proof.** Define the projection matrix onto the column space of \\(X_2\\) as \\(P_2 = X_2(X_2'X_2)^{-1}X_2'\\) and the annihilator matrix as \\(M_2 = I - P_2\\). Applying \\(M_2\\) to both sides of the full regression:

$$M_2 Y = M_2 X_1 \beta_1 + M_2 X_2 \beta_2 + M_2 \varepsilon$$

Since \\(M_2 X_2 = 0\\) by construction (the annihilator removes the \\(X_2\\) component), we get:

$$\tilde{Y} = \tilde{X}_1 \beta_1 + M_2 \varepsilon$$

where \\(\tilde{Y} = M_2 Y\\) and \\(\tilde{X}_1 = M_2 X_1\\). OLS on this residualized equation gives:

$$\hat{\beta}_1 = (\tilde{X}_1'\tilde{X}_1)^{-1}\tilde{X}_1'\tilde{Y}$$

which is exactly the coefficient on \\(X_1\\) in the full regression. \\(\square\\)

**Why this matters for DML.** The FWL theorem tells us that the "partialing out" procedure is exact for linear regression. The leap to DML is this: what if we replace the linear regressions in steps 1 and 2 with flexible ML models? If those ML models do a *better* job of removing the confounders' influence than a linear model would, the residuals are cleaner, and the final estimate of \\(\beta_1\\) is better controlled. That is exactly the DML idea.

---

## From Linear Models to Machine Learning

The limitation of classical IV and 2SLS is that they use **linear** first and second stages. In the real world, confounders affect price and demand in complex, nonlinear ways.

Consider a concrete example. Suppose demand for hotel rooms depends on the day of week, whether there is a local event, the season, and the weather. Season affects demand nonlinearly --- there might be a peak in summer, a smaller peak around holidays, and a trough in January. If your regression includes only a linear "month" variable, you have not captured this pattern. The residual \\(\varepsilon\\) still contains the nonlinear seasonal component, price is correlated with season, and you have endogeneity. You need interaction terms, polynomial terms, splines --- or you need a model that automatically captures nonlinear relationships.

Machine learning models --- random forests, gradient-boosted trees, neural networks --- excel at exactly this. They find complex nonlinear patterns in data. They handle high-dimensional feature spaces. They do not require you to specify the functional form in advance.

But there is a problem. ML models optimize for prediction, not causal inference. They do not give you a coefficient that you can interpret as "the causal effect of a 1% price change." They are prone to overfitting. And if you naively plug ML predictions into a causal framework, the regularization bias from the ML model contaminates your causal estimate.

Specifically, ML estimators converge to the truth at a rate slower than \\(n^{-1/2}\\) (the parametric rate). They might converge at \\(n^{-1/4}\\) or even slower, depending on the complexity of the function they are estimating. If you use an ML estimator directly to estimate the causal parameter, this slow rate infects your estimate, and you cannot construct valid confidence intervals.

What we need is a framework that uses ML for what it is good at --- flexible prediction, removing confounders --- while preserving the \\(n^{-1/2}\\) convergence rate and the asymptotic normality that we need for causal inference. Enter Double Machine Learning.

---

## Double Machine Learning (DML)

**Double Machine Learning** was developed by Chernozhukov, Chetty, Demirer, Duflo, Hansen, Newey, and Robins (2018). It is, in my view, one of the most important methodological contributions of the last decade, because it rigorously unites the strengths of machine learning with the discipline of causal inference.

The setup is the **partially linear model**:

$$Q = \theta P + g(X) + \varepsilon \quad \text{where } \mathbb{E}[\varepsilon \mid X, P] = 0$$

Here \\(\theta\\) is the causal effect of price on quantity --- the parameter we want. The function \\(g(X)\\) is an *arbitrary* function of the confounders; we make no assumptions about its form. We also write:

$$P = m(X) + V \quad \text{where } m(X) = \mathbb{E}[P \mid X] \text{ and } \mathbb{E}[V \mid X] = 0$$

The variable \\(V\\) is the part of price variation that is *not explained by the observables*. If our observables are rich enough that the only remaining source of price variation is unrelated to demand shocks, then estimating \\(\theta\\) from the residual variation identifies the causal effect.

The DML procedure applies Frisch-Waugh-Lovell logic using ML:

**Step 1: Partial out \\(X\\) from \\(Q\\).** Train an ML model to predict \\(Q\\) from \\(X\\). Compute the residual:

$$\tilde{Q} = Q - \hat{g}(X)$$

This residual is the part of demand variation not explained by the observables.

**Step 2: Partial out \\(X\\) from \\(P\\).** Train a separate ML model to predict \\(P\\) from \\(X\\). Compute the residual:

$$\tilde{P} = P - \hat{m}(X)$$

This residual is the part of price variation not explained by the observables.

**Step 3: Estimate \\(\theta\\).** Regress \\(\tilde{Q}\\) on \\(\tilde{P}\\):

$$\hat{\theta} = \frac{\sum_i \tilde{P}_i \tilde{Q}_i}{\sum_i \tilde{P}_i^2}$$

Why "double"? Because there are two ML models --- one for the outcome \\(Q\\), one for the treatment \\(P\\). Both are **nuisance parameters**: we do not care about them intrinsically; they are means to an end.

**Why it works.** By partialing out \\(X\\) from both sides, any confounding *through* \\(X\\) is eliminated. The residuals \\(\tilde{Q}\\) and \\(\tilde{P}\\) contain only variation that is orthogonal to \\(X\\). If the observables \\(X\\) capture all confounders (conditional exogeneity), then the remaining correlation between \\(\tilde{Q}\\) and \\(\tilde{P}\\) is the causal effect \\(\theta\\).

**The Neyman Orthogonality Condition.** This is the theoretical heart of DML and the reason it works despite using imperfect ML estimates. DML constructs a **score function** (or moment condition) \\(\psi(W; \theta, \eta)\\), where \\(W = (Q, P, X)\\) is the data and \\(\eta = (g, m)\\) are the nuisance parameters. The key property is:

$$\frac{\partial}{\partial \eta} \mathbb{E}[\psi(W; \theta_0, \eta)] \bigg|_{\eta = \eta_0} = 0$$

This says that the expected score is **insensitive** to small perturbations in the nuisance parameter estimates, evaluated at the true values. In other words, first-order errors in estimating \\(g\\) and \\(m\\) do not bias the estimate of \\(\theta\\).

For the partially linear model, the Neyman-orthogonal score is:

$$\psi(W; \theta, g, m) = (Q - g(X) - \theta P)(P - m(X))$$

$$= \tilde{Q} \cdot \tilde{P} - \theta \tilde{P}^2$$

You can verify the orthogonality condition by computing the Gateaux derivative with respect to \\(g\\) and \\(m\\) --- the cross-terms vanish at the true parameter values.

The practical consequence: the bias from ML estimation of the nuisance functions enters only at second order. If \\(\hat{g}\\) and \\(\hat{m}\\) converge at rate \\(n^{-1/4}\\) each, the product of their errors converges at rate \\(n^{-1/2}\\), which is the parametric rate. So \\(\hat{\theta}\\) is \\(\sqrt{n}\\)-consistent and asymptotically normal:

$$\sqrt{n}(\hat{\theta} - \theta_0) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$$

This means you get valid confidence intervals and hypothesis tests, even though you used black-box ML models for the nuisance estimation. This is remarkable.

---

## Cross-Fitting and Sample Splitting

There is a critical implementation detail that makes DML work in finite samples: **cross-fitting**.

If you use the same data to train the ML models and to compute the residuals, overfitting bias contaminates the estimate. The ML model might fit the noise in the training data, producing residuals that are artificially small. This overfitting bias does not vanish fast enough and can dominate the causal estimate.

The solution is cross-fitting, which is conceptually similar to cross-validation but serves a different purpose: **debiasing** rather than model selection.

**Algorithm: DML with K-fold cross-fitting.**

1. **Split** the data into \\(K\\) folds of approximately equal size (typically \\(K = 5\\)).
2. **For each fold** \\(k = 1, \ldots, K\\):
   - Let \\(I_k\\) denote the indices in fold \\(k\\), and \\(I_{-k}\\) denote all other indices.
   - Train \\(\hat{g}_{-k}\\) on \\(\{(Q_i, X_i) : i \in I_{-k}\}\\) --- predict \\(Q\\) from \\(X\\) using all data *except* fold \\(k\\).
   - Train \\(\hat{m}_{-k}\\) on \\(\{(P_i, X_i) : i \in I_{-k}\}\\) --- predict \\(P\\) from \\(X\\) using all data except fold \\(k\\).
   - Compute residuals for fold \\(k\\):
     - \\(\tilde{Q}_i = Q_i - \hat{g}_{-k}(X_i)\\) for \\(i \in I_k\\)
     - \\(\tilde{P}_i = P_i - \hat{m}_{-k}(X_i)\\) for \\(i \in I_k\\)
3. **Stack** all residuals across folds. Every observation now has a residual computed from an ML model that was not trained on that observation.
4. **Estimate** \\(\hat{\theta}\\) by regressing \\(\tilde{Q}\\) on \\(\tilde{P}\\):

$$\hat{\theta} = \frac{\sum_{i=1}^{n} \tilde{P}_i \tilde{Q}_i}{\sum_{i=1}^{n} \tilde{P}_i^2}$$

The standard error is computed as:

$$\hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^{n} (\tilde{Q}_i - \hat{\theta} \tilde{P}_i)^2 \cdot \tilde{P}_i^2 \bigg/ \left(\frac{1}{n} \sum_{i=1}^{n} \tilde{P}_i^2\right)^2$$

Cross-fitting ensures that the ML predictions are genuinely out-of-sample, which prevents overfitting from polluting the causal estimate. Without cross-fitting, DML can fail badly. With it, the theoretical guarantees hold.

---

## The Interactive Model and Beyond Partial Linearity

The partially linear model \\(Q = \theta P + g(X) + \varepsilon\\) we built in the DML section assumes something strong: the causal effect \\(\theta\\) is a **constant**. Every customer, every product, every context has the same price sensitivity. This is almost never true. A college student and a hedge fund manager do not react the same way to a $5 price increase. A price-sensitive shopper browsing discount sites and a brand-loyal repeat customer have fundamentally different elasticities. If you estimate a single average \\(\theta\\) and use it to set one price, you leave enormous surplus on the table.

The **interactive model** generalizes the partially linear model by letting the treatment effect depend on covariates:

$$Q = \theta(X) \cdot P + g(X) + \varepsilon$$

Here \\(\theta(X)\\) is a function, not a number. It maps from the covariate space to the real line. For a customer with features \\(X = x\\), the price elasticity is \\(\theta(x)\\). This function is called the **Conditional Average Treatment Effect (CATE)** --- it tells you how the causal effect of price varies across observable segments.

Why does this matter for pricing? Recall from Part 2 that third-degree price discrimination works by charging different prices to segments with different elasticities. The Lerner index says the optimal markup for segment \\(x\\) is:

$$\text{markup}(x) = \frac{1}{|\theta(x)|}$$

If \\(\theta(x)\\) is more negative (highly elastic customers), you set a small markup. If \\(\theta(x)\\) is close to zero (inelastic customers), you set a large markup. Estimating \\(\theta(X)\\) is therefore the statistical engine behind personalized pricing. The question is how to estimate a *function-valued* causal parameter from observational data.

### The DR-Learner (Doubly Robust Learner)

The DR-learner is a meta-algorithm for CATE estimation that inherits the doubly robust property from the semiparametric statistics literature. It proceeds in four steps.

**Step 1: Estimate the generalized propensity score.** In the binary treatment case, the propensity score is \\(e(X) = \mathbb{E}[D \mid X]\\) --- the probability of receiving treatment given covariates. For our continuous treatment (price), the analog is the **conditional density** of price given covariates, or more practically, the conditional mean \\(e(X) = \mathbb{E}[P \mid X]\\). We estimate this with any flexible ML model:

$$\hat{e}(X) = \hat{\mathbb{E}}[P \mid X]$$

This is identical to the \\(\hat{m}(X)\\) from the DML procedure.

**Step 2: Estimate the outcome model.** Fit a model for the full conditional expectation of the outcome:

$$\hat{\mu}(X, P) = \hat{\mathbb{E}}[Q \mid X, P]$$

This can be any flexible regression (random forest, gradient boosted trees, neural network) that takes both features and price as inputs.

**Step 3: Construct the pseudo-outcome.** The key insight is the **augmented inverse propensity weighting (AIPW)** construction. For each observation, compute:

$$\tilde{Y}_i = \hat{\mu}(X_i, P_i) + \frac{P_i - \hat{e}(X_i)}{\widehat{\text{Var}}(P \mid X_i)} \Big(Q_i - \hat{\mu}(X_i, P_i)\Big)$$

This pseudo-outcome has a remarkable property: its conditional expectation, as a function of \\(X\\), equals the CATE \\(\theta(X)\\), *even if one of the two models (propensity or outcome) is misspecified*. This is the **doubly robust** property. If the outcome model \\(\hat{\mu}\\) is correct, the second term has conditional mean zero and the first term gives you the CATE. If the propensity model \\(\hat{e}\\) is correct, the reweighting corrects for the outcome model's error. The estimator is consistent if *either* model is correct (but not necessarily both).

The derivation of double robustness proceeds as follows. Write the true conditional expectation:

$$\mathbb{E}[\tilde{Y} \mid X] = \mathbb{E}\left[\hat{\mu}(X, P) + \frac{P - \hat{e}(X)}{\widehat{\text{Var}}(P \mid X)} \big(Q - \hat{\mu}(X, P)\big) \;\middle|\; X\right]$$

If \\(\hat{\mu} = \mu\\) (the outcome model is correct), then \\(\mathbb{E}[Q - \mu(X, P) \mid X, P] = 0\\), so the second term vanishes in expectation, and \\(\mathbb{E}[\tilde{Y} \mid X] = \mathbb{E}[\mu(X, P) \mid X] = \theta(X) \cdot \mathbb{E}[P \mid X] + g(X)\\). After appropriate centering, this isolates \\(\theta(X)\\).

Alternatively, if \\(\hat{e} = e\\) (the propensity model is correct), the reweighting by \\((P - e(X))/\text{Var}(P \mid X)\\) acts as a local instrumental variable, projecting the outcome residual onto the price residual conditional on \\(X\\). The expectation of the price residual times the demand shock is zero by conditional exogeneity, and what remains is \\(\theta(X)\\).

**Step 4: Regress the pseudo-outcome on \\(X\\).** Fit a final ML model:

$$\hat{\theta}(X) = \text{ML model fit of } \tilde{Y}_i \text{ on } X_i$$

This final model directly estimates the CATE function. You can use any flexible regressor --- a random forest, a gradient-boosted model, or even a neural network.

### The R-Learner (Robinson's Decomposition Extended)

The R-learner takes a different approach, extending Robinson's (1988) partial residualization idea. Start from the interactive model:

$$Q = \theta(X) \cdot P + g(X) + \varepsilon$$

Take conditional expectations given \\(X\\):

$$\mathbb{E}[Q \mid X] = \theta(X) \cdot \mathbb{E}[P \mid X] + g(X)$$

Subtract to get:

$$Q - \mathbb{E}[Q \mid X] = \theta(X) \cdot (P - \mathbb{E}[P \mid X]) + \varepsilon$$

Define residuals \\(\tilde{Q} = Q - \hat{\ell}(X)\\) and \\(\tilde{P} = P - \hat{m}(X)\\), where \\(\hat{\ell}\\) and \\(\hat{m}\\) are ML estimates of the conditional means. Then:

$$\tilde{Q} = \theta(X) \cdot \tilde{P} + \varepsilon$$

The R-learner estimates \\(\theta(X)\\) by minimizing a weighted loss:

$$\hat{\theta} = \arg\min_{\theta(\cdot)} \sum_{i=1}^{n} \left(\tilde{Q}_i - \theta(X_i) \cdot \tilde{P}_i\right)^2$$

This is a weighted regression problem where the weights come from the price residuals. In regions of \\(X\\)-space where the price residual \\(\tilde{P}\\) is large (meaning there is substantial residual price variation after controlling for \\(X\\)), the estimate of \\(\theta(X)\\) is well-identified. Where \\(\tilde{P}\\) is small, identification is weak and the estimate is noisy.

The practical implementation uses regularized regression or a local estimator. A common choice: parameterize \\(\theta(X) = f_w(X)\\) as a neural network or boosted tree model with parameters \\(w\\), and minimize the loss above with respect to \\(w\\). The `econml` package from Microsoft Research implements both the DR-learner and R-learner.

### Connection to Optimal Pricing

Once you have \\(\hat{\theta}(X)\\), pricing follows immediately. For customer segment \\(x\\), the estimated price elasticity of demand is \\(\hat{\theta}(x)\\), and the Lerner index gives the optimal markup:

$$\frac{P^*(x) - MC}{P^*(x)} = \frac{1}{|\hat{\theta}(x)|}$$

Customers with \\(\hat{\theta}(x) = -3\\) (highly elastic) get a markup of \\(1/3 \approx 33\%\\). Customers with \\(\hat{\theta}(x) = -1.2\\) (relatively inelastic) get a markup of \\(1/1.2 \approx 83\%\\). This is the statistical implementation of the price discrimination theory from Part 2, now grounded in causal estimates from observational data.

---

## Causal Forests for Heterogeneous Price Sensitivity

The DR-learner and R-learner require you to choose an ML model for the final stage --- the model that maps \\(X\\) to \\(\theta(X)\\). Athey and Imbens (2018) proposed a purpose-built algorithm for this: the **Generalized Random Forest (GRF)**, and its special case, the **causal forest**.

The key insight behind causal forests is that ordinary random forests solve the wrong problem for causal inference. A standard random forest predicts \\(\mathbb{E}[Q \mid X]\\) --- it finds neighborhoods in \\(X\\)-space where the *outcome* is locally homogeneous. But for CATE estimation, we need neighborhoods where the *treatment effect* is locally homogeneous. These are not the same thing. A region of \\(X\\)-space might have very similar average demand but very different price sensitivities.

### The Splitting Criterion

In a standard regression tree, each node is split by choosing the variable and threshold that maximizes the reduction in variance of the outcome \\(Q\\). In a causal tree, the goal is different: choose the split that maximizes **heterogeneity in the treatment effect** across the two child nodes.

Formally, at a node containing observations \\(\{(Q_i, P_i, X_i)\}_{i \in \mathcal{N}}\\), consider a candidate split into left child \\(\mathcal{L}\\) and right child \\(\mathcal{R}\\). Estimate the treatment effect in each child using a simple IV-style estimator (regress \\(Q\\) on \\(P\\) within the node, or use residualized outcomes). Let \\(\hat{\theta}_\mathcal{L}\\) and \\(\hat{\theta}_\mathcal{R}\\) be these estimates. The causal tree chooses the split that maximizes:

$$\Delta(\mathcal{L}, \mathcal{R}) = \frac{n_\mathcal{L} \cdot n_\mathcal{R}}{(n_\mathcal{L} + n_\mathcal{R})^2} \left(\hat{\theta}_\mathcal{L} - \hat{\theta}_\mathcal{R}\right)^2$$

This criterion rewards splits where the treatment effect differs substantially between the two children, weighted by the sample sizes to avoid tiny leaves.

### Honesty: Separate Construction from Estimation

Here is a subtle but critical point. If you use the same data to determine the tree structure (where to split) and to estimate the treatment effect in each leaf, you get overfitting. The splits will chase noise in the treatment effect, and the leaf estimates will be biased.

The solution is **honesty**. Split the available data into two halves:
- The **structure sample**: used to determine the tree topology (which variables to split on, at what thresholds).
- The **estimation sample**: used to estimate the treatment effect within each leaf.

Because the estimation sample was not used to choose the splits, the treatment effect estimates are unbiased conditional on the tree structure. This separation is what gives causal forests valid confidence intervals --- a property that standard random forests lack for causal quantities.

### The Forest Weighting Interpretation

A single honest causal tree is noisy. A causal forest aggregates many such trees, each built on a bootstrap subsample. The aggregation produces a smooth estimate with a beautiful interpretation.

For a query point \\(x\\), define the **adaptive kernel weight**:

$$\alpha_i(x) = \frac{1}{B} \sum_{b=1}^{B} \frac{\mathbf{1}(X_i \in L_b(x))}{|L_b(x)|}$$

where \\(B\\) is the number of trees, \\(L_b(x)\\) is the leaf of tree \\(b\\) that contains \\(x\\), and \\(|L_b(x)|\\) is the number of estimation-sample observations in that leaf. The weight \\(\alpha_i(x)\\) measures how often observation \\(i\\) lands in the same leaf as the query point \\(x\\), averaged across all trees. Observations with covariates similar to \\(x\\) (in the sense defined by the forest's splitting criteria) receive higher weights.

The causal forest estimate at \\(x\\) is then a **locally weighted IV estimator**:

$$\hat{\theta}(x) = \frac{\sum_{i=1}^{n} \alpha_i(x) \cdot (Q_i - \bar{Q}_\alpha)(P_i - \bar{P}_\alpha)}{\sum_{i=1}^{n} \alpha_i(x) \cdot (P_i - \bar{P}_\alpha)^2}$$

where \\(\bar{Q}_\alpha = \sum_i \alpha_i(x) Q_i\\) and \\(\bar{P}_\alpha = \sum_i \alpha_i(x) P_i\\) are the weighted means. This is just the formula for a weighted regression coefficient of \\(Q\\) on \\(P\\), with weights determined by the forest. The forest has learned, from the data, which observations are "relevant" for estimating the treatment effect at point \\(x\\).

### Asymptotic Normality and Confidence Intervals

Under regularity conditions (the forest must be grown with sufficient randomness, the minimum leaf size must grow with \\(n\\), and the number of trees \\(B\\) must be large enough), the causal forest estimate is asymptotically normal:

$$\frac{\hat{\theta}(x) - \theta(x)}{\hat{\sigma}(x)} \xrightarrow{d} \mathcal{N}(0, 1)$$

where \\(\hat{\sigma}(x)\\) is a consistent variance estimate that can be computed from the forest itself using the infinitesimal jackknife or the bootstrap of little bags (half-sampling). This means you can construct pointwise confidence intervals:

$$\hat{\theta}(x) \pm z_{\alpha/2} \cdot \hat{\sigma}(x)$$

This is extraordinary for a nonparametric estimator. You get a flexible, data-adaptive estimate of a heterogeneous treatment effect *with valid statistical inference* at every point in the covariate space.

### Practical Significance for Pricing

With a causal forest, you can estimate a **different price elasticity for every customer segment**, along with a confidence interval for each. This enables:

1. **Targeted pricing with quantified uncertainty.** For segment \\(x\\), the estimated elasticity is \\(\hat{\theta}(x) \pm 1.96 \cdot \hat{\sigma}(x)\\). If the confidence interval is tight (say \\(-2.5 \pm 0.3\\)), you can confidently set the optimal markup. If the interval is wide (say \\(-2.5 \pm 2.0\\)), the estimate is too uncertain for aggressive price discrimination --- you should price conservatively or gather more data for that segment.

2. **Segment discovery.** The forest reveals *which covariates drive heterogeneity* in price sensitivity. By examining the splitting variables and their importance scores, you learn that, for example, customer tenure and geographic region are the primary drivers of elasticity differences --- information that guides marketing and product strategy.

3. **Policy simulation.** With \\(\hat{\theta}(x)\\) in hand, you can simulate the revenue impact of any proposed pricing policy --- a 10% across-the-board increase, a targeted discount for high-elasticity segments, or a tiered pricing structure --- before implementing it.

---

## Difference-in-Differences for Pricing Policy Evaluation

Everything so far has focused on estimating the *structural* relationship between price and demand: if I change the price for a specific product by 1%, how does demand respond? But there is a different and equally important question: **what was the causal effect of a pricing policy change?**

Suppose a retailer switches from fixed markups to algorithmic dynamic pricing in 50 of its 200 stores, while the other 150 continue with the old system. Six months later, management wants to know: did the new pricing system increase revenue? By how much? This is a **policy evaluation** question, and the right tool is **difference-in-differences (DiD)**.

### The Setup

You observe units (stores, products, regions) indexed by \\(i\\), at times \\(t\\). Some units receive a treatment \\(D_{it} = 1\\) (the new pricing policy) starting at time \\(t^*\\), while others remain untreated \\(D_{it} = 0\\). You observe an outcome \\(Q_{it}\\) (revenue, quantity, or profit) for all units at all times, both before and after the policy change.

The fundamental problem of causal inference: you cannot observe the treated stores' revenue *had they not been treated*. That counterfactual is missing. DiD constructs it using the control group.

### The Parallel Trends Assumption

DiD rests on one critical assumption: **absent the treatment, the treatment and control groups would have followed the same trend in outcomes.** Formally:

$$\mathbb{E}[Q_{it}(0) - Q_{it'}(0) \mid D_i = 1] = \mathbb{E}[Q_{it}(0) - Q_{it'}(0) \mid D_i = 0]$$

for all pre-treatment periods \\(t, t'\\) and the post-treatment period. Here \\(Q_{it}(0)\\) denotes the potential outcome under no treatment. The assumption says that the *change* in outcomes would have been the same in both groups, even though the *levels* may differ. The treatment and control groups do not need to start at the same revenue level --- they just need to have been moving in the same direction at the same rate.

This is weaker than requiring the two groups to be identical. It allows for permanent differences (some stores are just bigger or in better locations) as long as those differences are *stable* over time.

### The DiD Estimator

The estimator is beautifully simple. Compute the average outcome for each group (treatment and control) in each period (before and after). Then take the difference of the differences:

$$\hat{\theta}_{DiD} = \underbrace{(\bar{Q}_{treat,post} - \bar{Q}_{treat,pre})}_{\text{change in treated group}} - \underbrace{(\bar{Q}_{control,post} - \bar{Q}_{control,pre})}_{\text{change in control group}}$$

The first difference removes time-invariant characteristics of the treated group (store size, location quality). The second difference removes common time trends (seasonal demand shifts, macroeconomic conditions). What remains is the causal effect of the treatment.

Why does this work? Write the potential outcomes model:

$$Q_{it} = \alpha_i + \gamma_t + \theta \cdot D_{it} + \varepsilon_{it}$$

where \\(\alpha_i\\) is a **unit fixed effect** (captures all time-invariant differences between units), \\(\gamma_t\\) is a **time fixed effect** (captures all common temporal shocks), \\(\theta\\) is the treatment effect, and \\(\varepsilon_{it}\\) is idiosyncratic noise. Taking the double difference:

$$\hat{\theta}_{DiD} = (\bar{Q}_{1,post} - \bar{Q}_{1,pre}) - (\bar{Q}_{0,post} - \bar{Q}_{0,pre})$$

The \\(\alpha_i\\) terms cancel within each group (they are the same before and after). The \\(\gamma_t\\) terms cancel across groups (they are the same for treatment and control). What survives is \\(\theta\\).

### Two-Way Fixed Effects Regression

In practice, with panel data (multiple units observed over multiple time periods), the DiD estimator is implemented via **two-way fixed effects (TWFE)** regression:

$$Q_{it} = \alpha_i + \gamma_t + \theta \cdot D_{it} + \varepsilon_{it}$$

where \\(\alpha_i\\) are unit dummies (one per store) and \\(\gamma_t\\) are time dummies (one per period). OLS on this specification gives the TWFE estimate \\(\hat{\theta}_{TWFE}\\). In the simple 2-period, 2-group case, this is exactly the DiD estimator. With multiple periods and groups, it generalizes naturally.

You can add time-varying covariates \\(X_{it}\\) (e.g., local advertising spend, weather, competitor actions):

$$Q_{it} = \alpha_i + \gamma_t + \theta \cdot D_{it} + X_{it}'\beta + \varepsilon_{it}$$

These covariates improve precision and help the parallel trends assumption hold conditionally.

### The Staggered Adoption Problem

The simple DiD story works cleanly when treatment happens to all treated units at the same time. In practice, rollouts are often **staggered**: different stores adopt the new pricing at different dates. Store A starts in January, Store B in March, Store C in June.

The standard TWFE regression still includes unit and time fixed effects with \\(D_{it}\\) as the treatment indicator. For years, practitioners assumed this was fine. Then a series of papers --- Goodman-Bacon (2021), Callaway and Sant'Anna (2021), Sun and Abraham (2021), de Chaisemartin and D'Haultfoeuille (2020) --- showed that **TWFE is biased under staggered adoption when the treatment effect is heterogeneous**.

The problem is subtle and important. TWFE with staggered adoption implicitly computes a weighted average of many 2x2 DiD comparisons. Some of those comparisons use **already-treated units as controls** --- for example, comparing stores treated in March to stores treated in January (which are already under treatment by March). If the treatment effect changes over time (e.g., the new pricing system works better as the algorithm learns), using already-treated units as controls introduces bias. Worse, some of the implicit weights can be **negative**, meaning that a positive treatment effect in one comparison can enter the overall estimate with a negative sign.

Goodman-Bacon (2021) provided an exact decomposition: the TWFE estimator is a weighted average of all possible 2x2 DiD estimates from the data, with weights that depend on group sizes and treatment timing. When those weights are negative (which happens when treatment effects are heterogeneous over time), the TWFE estimate can even have the wrong sign.

**Modern solutions:**

- **Callaway and Sant'Anna (2021):** Estimate group-time-specific treatment effects \\(\text{ATT}(g, t)\\) --- the average effect for cohort \\(g\\) (units first treated at time \\(g\\)) at time \\(t\\). These building blocks are then aggregated with user-chosen (non-negative) weights to form summary measures.
- **Sun and Abraham (2021):** An interaction-weighted estimator that avoids contamination from heterogeneous effects across cohorts.
- **Synthetic DiD (Arkhangelsky et al., 2021):** Combines the synthetic control idea (reweight the control group to match the treated group's pre-treatment trajectory) with the DiD framework. This relaxes the parallel trends assumption by constructing a synthetic control that actually tracks the treated group's pre-treatment path.

### Application to Pricing

Consider our retailer rolling out dynamic pricing to stores in waves. The staggered DiD framework lets you:
1. Estimate the revenue effect for each cohort (wave of stores) at each post-treatment time.
2. Test whether the effect grows over time (as the algorithm learns) or fades (as competitors respond).
3. Aggregate into an overall policy effect with proper statistical inference.

<svg viewBox="0 0 720 420" xmlns="http://www.w3.org/2000/svg" style="max-width: 720px; display: block; margin: 2em auto;">
  <defs>
    <marker id="arrow-did" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#d4d4d4"/>
    </marker>
  </defs>
  <text x="360" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" font-weight="bold" fill="#d4d4d4">Difference-in-Differences: Parallel Trends Logic</text>

  <!-- Axes -->
  <line x1="80" y1="370" x2="670" y2="370" stroke="#d4d4d4" stroke-width="1.5" marker-end="url(#arrow-did)"/>
  <line x1="80" y1="370" x2="80" y2="40" stroke="#d4d4d4" stroke-width="1.5" marker-end="url(#arrow-did)"/>
  <text x="375" y="405" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#d4d4d4">Time</text>
  <text x="30" y="205" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#d4d4d4" transform="rotate(-90, 30, 205)">Revenue</text>

  <!-- Treatment time marker -->
  <line x1="370" y1="55" x2="370" y2="370" stroke="#ffd54f" stroke-width="1.5" stroke-dasharray="8,4"/>
  <text x="370" y="48" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#ffd54f">Policy Change</text>

  <!-- Pre-period labels -->
  <text x="220" y="390" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#d4d4d4">Pre-treatment</text>
  <text x="520" y="390" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#d4d4d4">Post-treatment</text>

  <!-- Control group (solid throughout) -->
  <line x1="120" y1="280" x2="370" y2="220" stroke="#4fc3f7" stroke-width="2.5"/>
  <line x1="370" y1="220" x2="630" y2="160" stroke="#4fc3f7" stroke-width="2.5"/>
  <text x="640" y="160" font-family="Arial, sans-serif" font-size="12" fill="#4fc3f7" font-weight="bold">Control</text>

  <!-- Treatment group pre-period -->
  <line x1="120" y1="230" x2="370" y2="170" stroke="#e57373" stroke-width="2.5"/>

  <!-- Treatment group post-period (actual, diverging upward) -->
  <line x1="370" y1="170" x2="630" y2="80" stroke="#e57373" stroke-width="2.5"/>
  <text x="640" y="80" font-family="Arial, sans-serif" font-size="12" fill="#e57373" font-weight="bold">Treated</text>

  <!-- Counterfactual (dashed, parallel to control) -->
  <line x1="370" y1="170" x2="630" y2="110" stroke="#e57373" stroke-width="1.5" stroke-dasharray="6,4"/>
  <text x="640" y="115" font-family="Arial, sans-serif" font-size="11" fill="#e57373" font-style="italic">Counterfactual</text>

  <!-- Treatment effect brace -->
  <line x1="600" y1="83" x2="600" y2="108" stroke="#66bb6a" stroke-width="2.5"/>
  <text x="615" y="100" font-family="Arial, sans-serif" font-size="12" fill="#66bb6a" font-weight="bold">&#x03B8;</text>

  <!-- Dots at treatment time -->
  <circle cx="370" cy="220" r="4" fill="#4fc3f7"/>
  <circle cx="370" cy="170" r="4" fill="#e57373"/>
</svg>

The diagram captures the core logic. Before the policy change, both groups trend in parallel (same slope). After the policy change, the treated group diverges from the control group. The counterfactual (dashed red line) shows where the treated group would have been without the treatment, extrapolating the parallel pre-treatment trend. The treatment effect \\(\theta\\) is the gap between the actual treated outcome and this counterfactual.

---

## The BLP Model --- Structural Demand Estimation at Scale

The Berry-Levinsohn-Pakes (1995) model --- universally known as BLP --- is the workhorse of demand estimation in industrial organization. If DML is the tool for estimating causal effects from observational data, BLP is the tool for estimating the entire demand system for a market with many differentiated products. It tells you not just your own-price elasticity, but the full matrix of own-price and cross-price elasticities for every product against every other product. This is what you need to optimally price a product portfolio.

### The Setup

Consider a market with \\(J\\) products. Consumer \\(i\\) can choose one of the \\(J\\) products or an outside option (buy nothing). Consumer \\(i\\)'s utility from product \\(j\\) is:

$$u_{ij} = \delta_j + \mu_{ij} + \varepsilon_{ij}$$

This utility has three components.

**Mean utility** \\(\delta_j\\). This is common to all consumers and captures the "average" appeal of product \\(j\\):

$$\delta_j = X_j'\bar{\beta} - \alpha p_j + \xi_j$$

where \\(X_j\\) is a vector of observed product characteristics (horsepower, fuel efficiency, screen size --- whatever is relevant), \\(p_j\\) is price, \\(\alpha > 0\\) is the mean price sensitivity, and \\(\xi_j\\) is **unobserved product quality** --- the brand cachet, design quality, advertising effect, or any demand-relevant attribute not captured by \\(X_j\\). This unobserved quality is the source of the endogeneity problem: better products (higher \\(\xi_j\\)) tend to have higher prices.

**Individual taste variation** \\(\mu_{ij}\\). This captures the fact that different consumers have different preferences over product characteristics:

$$\mu_{ij} = X_j' \Sigma v_i + \sigma_\alpha \nu_i p_j$$

where \\(v_i \sim \mathcal{N}(0, I)\\) and \\(\nu_i \sim \mathcal{N}(0, 1)\\) are consumer-specific taste shocks, and \\(\Sigma\\) and \\(\sigma_\alpha\\) are parameters governing the dispersion of preferences. The random coefficient on characteristics means that some consumers care more about feature A, others about feature B. The random coefficient on price means that some consumers are more price-sensitive than others.

**Idiosyncratic shock** \\(\varepsilon_{ij}\\). This is i.i.d. Type I extreme value (Gumbel distributed). This distributional assumption gives the logit structure to the choice probabilities, which makes the model computationally tractable.

### Market Shares

The probability that consumer \\(i\\) chooses product \\(j\\) is (by the logit formula):

$$\pi_{ij} = \frac{\exp(\delta_j + \mu_{ij})}{1 + \sum_{k=1}^{J} \exp(\delta_k + \mu_{ik})}$$

The 1 in the denominator represents the outside option (utility normalized to zero). The **predicted market share** of product \\(j\\) is obtained by integrating over the distribution of consumer types:

$$s_j(\delta, \Sigma, \sigma_\alpha) = \int \frac{\exp(\delta_j + \mu_{ij})}{1 + \sum_{k=1}^{J} \exp(\delta_k + \mu_{ik})} \; dF(v_i, \nu_i)$$

This integral has no closed-form solution because the denominator depends on \\(v_i\\) and \\(\nu_i\\) through all \\(J\\) products simultaneously. It must be computed by **simulation**: draw \\(R\\) consumers from the distribution \\(F\\), compute each one's choice probability, and average:

$$s_j \approx \frac{1}{R} \sum_{r=1}^{R} \frac{\exp(\delta_j + \mu_{rj})}{1 + \sum_{k=1}^{J} \exp(\delta_k + \mu_{rk})}$$

Typical values of \\(R\\) range from 200 to 1000 simulation draws.

### The Contraction Mapping (Berry 1994)

Here is the computational heart of BLP. We observe actual market shares \\(S_j\\) (from sales data). We want to find the vector of mean utilities \\(\delta = (\delta_1, \ldots, \delta_J)\\) such that the model-predicted shares equal the observed shares: \\(s_j(\delta, \Sigma, \sigma_\alpha) = S_j\\) for all \\(j\\).

This is a system of \\(J\\) nonlinear equations in \\(J\\) unknowns. Berry (1994) showed that the mapping:

$$\delta_j^{(t+1)} = \delta_j^{(t)} + \ln S_j - \ln s_j(\delta^{(t)}, \Sigma, \sigma_\alpha)$$

is a **contraction mapping** and converges to a unique fixed point \\(\delta^*(\Sigma, \sigma_\alpha)\\). The intuition: if the predicted share of product \\(j\\) is too low (\\(s_j < S_j\\)), increase its mean utility (\\(\delta_j\\) goes up). If predicted share is too high, decrease it. The logarithmic adjustment ensures convergence.

This is remarkable because it means that for any given set of random coefficient parameters \\((\Sigma, \sigma_\alpha)\\), we can *invert* the observed market shares to recover the mean utilities. And from the mean utilities, we can back out the unobserved quality:

$$\xi_j = \delta_j^* - X_j'\bar{\beta} + \alpha p_j$$

### The Endogeneity Problem and Instruments

The unobserved quality \\(\xi_j\\) is the econometric headache. Products with high \\(\xi_j\\) (think Apple products, luxury brands) tend to have high prices. So \\(p_j\\) and \\(\xi_j\\) are correlated, and naive estimation of \\(\alpha\\) (the price coefficient) is biased.

The solution: instruments. The moment condition is:

$$\mathbb{E}[Z_j' \xi_j] = 0$$

where \\(Z_j\\) is a vector of instruments. BLP proposed using functions of other products' characteristics as instruments. The logic: the number and configuration of competing products affects firm \\(j\\)'s pricing (through competitive pressure), but the characteristics of competitors' products do not directly enter the demand for product \\(j\\) (conditional on \\(j\\)'s own characteristics). Common BLP instruments include:

1. **Sum of characteristics of other products by the same firm:** \\(\sum_{k \neq j, k \in \mathcal{F}_j} X_k\\), where \\(\mathcal{F}_j\\) is the set of products made by \\(j\\)'s firm. These capture within-firm cannibalization effects on pricing.
2. **Sum of characteristics of rival firms' products:** \\(\sum_{k \notin \mathcal{F}_j} X_k\\). These capture competitive pressure on pricing.

### The GMM Objective

Stack the moment conditions across products and markets:

$$\hat{\xi}(\theta) = \delta^*(\Sigma, \sigma_\alpha) - X\bar{\beta} + \alpha p$$

where \\(\theta = (\bar{\beta}, \alpha, \Sigma, \sigma_\alpha)\\). The GMM estimator minimizes:

$$\hat{\theta}_{GMM} = \arg\min_\theta \; \hat{\xi}(\theta)' Z \, W \, Z' \hat{\xi}(\theta)$$

where \\(W\\) is a weighting matrix (typically the optimal GMM weighting matrix \\(W = (\frac{1}{n} Z' \hat{\xi} \hat{\xi}' Z)^{-1}\\), estimated in a two-step procedure).

The estimation proceeds by nested optimization: the outer loop searches over \\((\Sigma, \sigma_\alpha)\\), and for each candidate value, the inner loop runs the contraction mapping to solve for \\(\delta^*\\), then computes \\(\bar{\beta}\\) and \\(\alpha\\) by linear IV regression, then evaluates the GMM objective. This is computationally intensive --- a single evaluation requires multiple contraction mapping iterations, each requiring simulation of market shares --- but modern implementations handle markets with hundreds of products.

### Elasticities from BLP

Once the model is estimated, you can compute the full elasticity matrix. The own-price elasticity of product \\(j\\) is:

$$\varepsilon_{jj} = -\frac{\alpha \, p_j}{s_j} \int \pi_{ij}(1 - \pi_{ij}) \, dF(v_i, \nu_i)$$

The cross-price elasticity of product \\(j\\) with respect to the price of product \\(k\\) is:

$$\varepsilon_{jk} = \frac{\alpha \, p_k}{s_j} \int \pi_{ij} \cdot \pi_{ik} \, dF(v_i, \nu_i)$$

These elasticities depend on the distribution of consumer types through the integral, and they reflect realistic substitution patterns: products that are "close" in characteristic space (similar features, similar prices) have high cross-price elasticities, while dissimilar products have low cross-elasticities. This is a major improvement over the plain logit model, where the cross-price elasticities depend only on market shares (the "independence of irrelevant alternatives" property) and cannot capture the fact that a Honda Civic competes more with a Toyota Corolla than with a BMW 7 Series.

### Why BLP Matters for Pricing

BLP gives you the complete demand system. With the full elasticity matrix in hand, you can:

1. **Optimize the prices of an entire product portfolio simultaneously.** The optimal price for product \\(j\\) depends on its own elasticity *and* on the cross-elasticities with all other products in the firm's portfolio. Raising the price of product \\(j\\) pushes some customers to substitute to product \\(k\\) (which may also be yours), and this cannibalization must be accounted for.

2. **Simulate competitive responses.** If you know competitors' cost structures (or can estimate them from the first-order conditions of their profit maximization), you can predict how they will adjust prices in response to your changes.

3. **Evaluate mergers and acquisitions.** Post-merger, a firm internalizes the cross-price effects between the merging firms' products. BLP is the standard tool for predicting post-merger price changes in antitrust analysis.

---

## Python --- Causal Forest for Heterogeneous Elasticity

The following code demonstrates heterogeneous treatment effect estimation in the pricing context. We simulate a market where price sensitivity varies with customer income, then use an R-learner approach with DML-within-bins to recover the heterogeneous elasticity.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

np.random.seed(42)
n = 6000

# Customer features: income (main source of heterogeneity) + noise features
income = np.random.uniform(20, 150, n)  # thousands of dollars
x2 = np.random.normal(0, 1, n)
x3 = np.random.normal(0, 1, n)
X = np.column_stack([income, x2, x3])

# True heterogeneous price effect: theta(X) depends on income
# Low income -> more price sensitive (theta more negative)
# High income -> less price sensitive (theta closer to zero)
theta_true = -3.0 + 2.0 * (income / 150)  # ranges from ~ -2.73 to ~ -1.0

# Confounders affect both price and demand
g_X = 50 + 0.3 * income + 2.0 * np.sin(income / 20) + 1.5 * x2
m_X = 10 + 0.05 * income + 0.8 * np.cos(income / 25) + 0.5 * x3

# Price: function of confounders + noise
V = np.random.normal(0, 2, n)
P = m_X + V

# Demand: heterogeneous effect of price + confounders + noise
epsilon = np.random.normal(0, 3, n)
Q = theta_true * P + g_X + epsilon

# ================================================================
# Method: DML within income quintiles (binned heterogeneity)
# ================================================================
n_bins = 10
income_edges = np.percentile(income, np.linspace(0, 100, n_bins + 1))
bin_indices = np.digitize(income, income_edges[1:-1])

theta_hat_bins = np.zeros(n_bins)
theta_true_bins = np.zeros(n_bins)
se_bins = np.zeros(n_bins)
income_midpoints = np.zeros(n_bins)

for b in range(n_bins):
    mask = bin_indices == b
    n_b = mask.sum()
    X_b, P_b, Q_b = X[mask], P[mask], Q[mask]
    income_midpoints[b] = income[mask].mean()
    theta_true_bins[b] = theta_true[mask].mean()

    # DML with 5-fold cross-fitting within this bin
    Q_tilde_b = np.zeros(n_b)
    P_tilde_b = np.zeros(n_b)
    kf = KFold(n_splits=5, shuffle=True, random_state=b)

    for train_idx, test_idx in kf.split(X_b):
        rf_q = RandomForestRegressor(
            n_estimators=150, max_depth=8,
            min_samples_leaf=10, random_state=42
        )
        rf_q.fit(X_b[train_idx], Q_b[train_idx])
        Q_tilde_b[test_idx] = Q_b[test_idx] - rf_q.predict(X_b[test_idx])

        rf_p = RandomForestRegressor(
            n_estimators=150, max_depth=8,
            min_samples_leaf=10, random_state=42
        )
        rf_p.fit(X_b[train_idx], P_b[train_idx])
        P_tilde_b[test_idx] = P_b[test_idx] - rf_p.predict(X_b[test_idx])

    # Estimate theta within this bin
    theta_hat_bins[b] = (
        np.sum(P_tilde_b * Q_tilde_b) / np.sum(P_tilde_b ** 2)
    )

    # Standard error
    resid = Q_tilde_b - theta_hat_bins[b] * P_tilde_b
    se_bins[b] = np.sqrt(
        np.mean(resid ** 2 * P_tilde_b ** 2)
        / (np.mean(P_tilde_b ** 2) ** 2 * n_b)
    )

# ================================================================
# Visualization
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left panel: Estimated vs true theta(X) by income ---
ax = axes[0]
ax.plot(income_midpoints, theta_true_bins, 'g-o', linewidth=2,
        markersize=7, label=r'True $\theta(X)$', zorder=3)
ax.errorbar(income_midpoints, theta_hat_bins, yerr=1.96 * se_bins,
            fmt='s-', color='#4fc3f7', linewidth=2, markersize=6,
            capsize=4, capthick=1.5,
            label=r'DML estimate $\hat{\theta}(X) \pm 1.96 \, SE$',
            zorder=2)
ax.axhline(y=0, color='#888888', linewidth=0.8, linestyle=':')
ax.set_xlabel(r'Income (\$k)', fontsize=13)
ax.set_ylabel(r'Price elasticity $\theta(X)$', fontsize=13)
ax.set_title('Heterogeneous Price Sensitivity by Income', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# --- Right panel: Implied optimal markup ---
ax = axes[1]
markup_true = 1.0 / np.abs(theta_true_bins)
markup_hat = 1.0 / np.abs(theta_hat_bins)
markup_upper = 1.0 / np.abs(theta_hat_bins + 1.96 * se_bins)
markup_lower = 1.0 / np.abs(theta_hat_bins - 1.96 * se_bins)

ax.plot(income_midpoints, markup_true, 'g-o', linewidth=2,
        markersize=7, label=r'True optimal markup $1/|\theta|$', zorder=3)
ax.fill_between(income_midpoints, markup_lower, markup_upper,
                alpha=0.2, color='#4fc3f7', zorder=1)
ax.plot(income_midpoints, markup_hat, 's-', color='#4fc3f7', linewidth=2,
        markersize=6, label=r'Estimated markup $1/|\hat{\theta}|$',
        zorder=2)
ax.set_xlabel(r'Income (\$k)', fontsize=13)
ax.set_ylabel(r'Lerner index $1/|\theta|$', fontsize=13)
ax.set_title('Implied Optimal Markup by Income Segment', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('heterogeneous_elasticity.png', dpi=150, bbox_inches='tight')
plt.show()

# Print results table
print(f"{'Income ($k)':>12} {'True θ':>8} {'Est θ':>8} {'SE':>8} "
      f"{'True Markup':>12} {'Est Markup':>12}")
print("-" * 68)
for i in range(n_bins):
    print(f"{income_midpoints[i]:>12.1f} {theta_true_bins[i]:>8.3f} "
          f"{theta_hat_bins[i]:>8.3f} {se_bins[i]:>8.4f} "
          f"{markup_true[i]:>12.3f} {markup_hat[i]:>12.3f}")
```

The left panel shows the estimated price elasticity \\(\hat{\theta}(X)\\) as a function of income, with 95% confidence intervals. Low-income customers (around $30k) have elasticities near \\(-2.6\\), while high-income customers (around $140k) have elasticities near \\(-1.1\\). The DML estimates track the true function closely.

The right panel translates these elasticities into optimal markups via the Lerner index \\(1/|\theta|\\). The pattern is exactly what economic theory predicts: high-income, less price-sensitive customers should face markups of roughly 80--90%, while low-income, highly elastic customers should face markups of roughly 35--40%. This is the quantitative foundation for the third-degree price discrimination strategies discussed in Part 2 --- now estimated from observational data with proper causal identification and uncertainty quantification.

---

## Demand Censoring

There is a practical complication that arises constantly in retail and travel: **demand censoring**, also called **stockout bias**.

When inventory runs out, you do not observe the true demand. You observe that all units sold. If a product was priced at $10 and you stocked 100 units and all 100 sold, the true demand at $10 might have been 120, or 150, or 500. You only know it was at least 100.

This is a classic **right-censoring** problem (the same structure as survival analysis in biostatistics). The observed quantity is:

$$Q^{obs} = \min(Q^{true}, \text{Inventory})$$

If popular products at low prices tend to stock out, you systematically underestimate demand at low prices. This biases the demand curve: it looks flatter (less elastic) than it truly is. You underestimate how much demand would increase if you lowered prices, and you set prices too high.

**Solutions:**

1. **Tobit models.** Explicitly model the censoring. The likelihood for an uncensored observation is the normal density; for a censored observation (sold out), it is the survival function \\(P(Q^{true} > \text{Inventory})\\). Maximum likelihood estimation handles the mixture.

2. **Survival analysis.** Treat "time to stockout" or "demand exceeding inventory" as a survival problem. Cox proportional hazards or accelerated failure time models can accommodate censoring while estimating the price coefficient.

3. **Modified DML.** Adapt the DML framework by replacing the standard least-squares residualization with a censored regression in the outcome model. This is an active research area.

4. **Data engineering.** Often the simplest fix: restrict the sample to observations where you are confident no stockout occurred (inventory was well above sales). This discards data but avoids the bias. For airlines and hotels, use booking data before the flight/night is full.

Demand censoring is especially important in industries with perishable inventory: airlines (finite seats), hotels (finite rooms), event tickets, and fashion retail (seasonal with no replenishment).

---

## Conjoint Analysis and Discrete Choice Models

Everything so far has used observational data --- the transaction records from actual sales. There is an entirely different approach: **directly measure willingness to pay** through structured experiments.

**Conjoint analysis** works as follows. Show a sample of consumers a series of hypothetical products, each described by a set of attributes: brand, features, size, and --- critically --- price. Ask them to rank these products, rate them, or choose their preferred option from a set. From the pattern of choices across many scenarios, you can statistically decompose the total utility into contributions from each attribute, including price.

For example, you might show a consumer:

| Product A | Product B |
|---|---|
| Brand X | Brand Y |
| 128 GB storage | 256 GB storage |
| $799 | $699 |

If the consumer chooses Product A despite the higher price, you infer that the brand and lower storage are worth at least $100 to them. Across thousands of such choices from hundreds of consumers, you estimate the distribution of willingness to pay for each attribute.

**Discrete choice models** formalize this. The workhorse is the **conditional logit** (McFadden, 1974; he won the Nobel Prize for this). Consumer \\(i\\) chooses product \\(j\\) from a set of \\(J\\) alternatives. The probability of choosing \\(j\\) is:

$$P(y_i = j) = \frac{\exp(V_{ij})}{\sum_{k=1}^{J} \exp(V_{ik})}$$

where \\(V_{ij} = \beta_{price} \cdot \text{Price}_j + \beta_{features}' \cdot \text{Features}_j\\) is the **deterministic utility**. The coefficient \\(\beta_{price}\\) directly gives you price sensitivity. The own-price elasticity of product \\(j\\) is:

$$\varepsilon_{jj} = \beta_{price} \cdot P_j \cdot (1 - s_j)$$

where \\(s_j\\) is the market share of product \\(j\\).

The **mixed logit** (or random coefficients logit) extends this by allowing \\(\beta_{price}\\) to vary across consumers, drawn from some distribution (e.g., normal). This captures preference heterogeneity --- some consumers are price-sensitive, others are not.

**BLP (Berry, Levinsohn, and Pakes, 1995)** is the state of the art for estimating demand from aggregate market share data (rather than individual-level choice data). It combines a structural model of consumer choice with instrumental variables to handle the endogeneity of prices in observed market data. BLP is the dominant framework in industrial organization for studying markets with differentiated products (cars, cereals, airlines). The method involves solving a fixed-point problem to invert observed market shares into mean utility levels, then using IV to estimate the structural parameters. It is computationally intensive but incredibly powerful.

**Trade-offs.** Conjoint analysis gives you direct willingness-to-pay estimates without needing to solve the endogeneity problem (because prices are randomized by design). But it is expensive, requires careful experimental design, and suffers from hypothetical bias (what people say they would do is not always what they actually do). Observational methods like DML use real purchase behavior but require strong assumptions about identification. The best practice is to use both: conjoint for initial estimates and directional guidance, and observational methods (DML, BLP) for ongoing demand tracking with real data.

---

## Python Implementation

### Demonstration 1: Endogeneity Bias and the IV Fix

We start by simulating a market where price is endogenous and showing that OLS gives the wrong answer while IV gets it right.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n = 2000

# True demand: Q = 100 - 2*P + epsilon
# The true price coefficient is -2
theta_true = -2.0
alpha = 100.0

# Unobserved demand shock
epsilon = np.random.normal(0, 10, n)

# Cost shifter instrument: affects price but not demand directly
Z = np.random.normal(50, 15, n)

# Price is set based on costs AND the demand shock (endogeneity!)
# Firm observes epsilon and raises price when demand is high
P = 20 + 0.5 * Z + 0.6 * epsilon + np.random.normal(0, 3, n)

# Observed quantity
Q = alpha + theta_true * P + epsilon

# --- OLS (biased) ---
ols = LinearRegression()
ols.fit(P.reshape(-1, 1), Q)
theta_ols = ols.coef_[0]

# --- IV estimation (2SLS by hand) ---
# Stage 1: regress P on Z
stage1 = LinearRegression()
stage1.fit(Z.reshape(-1, 1), P)
P_hat = stage1.predict(Z.reshape(-1, 1))

# Stage 2: regress Q on P_hat
stage2 = LinearRegression()
stage2.fit(P_hat.reshape(-1, 1), Q)
theta_iv = stage2.coef_[0]

print(f"True theta:  {theta_true:.3f}")
print(f"OLS theta:   {theta_ols:.3f}  (biased toward zero / positive)")
print(f"IV theta:    {theta_iv:.3f}  (close to true value)")

# --- Visualization ---
fig, ax = plt.subplots(1, 1, figsize=(9, 6))

ax.scatter(P, Q, alpha=0.15, s=8, color='#888888', label='Observed data')

P_grid = np.linspace(P.min(), P.max(), 100)
Q_true = alpha + theta_true * P_grid
Q_ols = ols.intercept_ + theta_ols * P_grid
Q_iv = stage2.intercept_ + theta_iv * P_grid

ax.plot(P_grid, Q_true, 'g-', linewidth=2.5, label=rf'True demand ($\theta = {theta_true:.1f}$)')
ax.plot(P_grid, Q_ols, 'r--', linewidth=2, label=rf'OLS estimate ($\hat{{\theta}} = {theta_ols:.2f}$)')
ax.plot(P_grid, Q_iv, 'b-.', linewidth=2, label=rf'IV estimate ($\hat{{\theta}} = {theta_iv:.2f}$)')

ax.set_xlabel(r'Price $P$', fontsize=13)
ax.set_ylabel(r'Quantity $Q$', fontsize=13)
ax.set_title('Endogeneity Bias: OLS vs. Instrumental Variables', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iv_vs_ols.png', dpi=150, bbox_inches='tight')
plt.show()
```

The OLS line will be noticeably flatter than the true demand curve (or even upward-sloping), because the positive correlation between price and the demand shock biases the coefficient toward zero or positive. The IV line, using only the exogenous variation from the cost shifter, recovers the true slope.

### Demonstration 2: Double Machine Learning from Scratch

Now we implement DML with cross-fitting, handling nonlinear confounders that OLS cannot control for.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

np.random.seed(123)
n = 5000
d = 5  # number of confounders

# Generate confounders
X = np.random.normal(0, 1, (n, d))

# Complex nonlinear confounder effects on demand
# g(X) = 3*sin(X1) + 2*X2^2 - X3*X4 + exp(0.5*X5)
g_X = (3 * np.sin(X[:, 0]) + 2 * X[:, 1]**2
       - X[:, 2] * X[:, 3] + np.exp(0.5 * X[:, 4]))

# Complex nonlinear confounder effects on price
# m(X) = 2*cos(X1) + X2*X3 + |X4|^1.5 + X5
m_X = (2 * np.cos(X[:, 0]) + X[:, 1] * X[:, 2]
       + np.abs(X[:, 3])**1.5 + X[:, 4])

# True causal effect
theta_true = -1.5

# Price: nonlinear function of X plus noise
V = np.random.normal(0, 1, n)
P = m_X + V

# Demand: causal effect of price + nonlinear confounders + noise
epsilon = np.random.normal(0, 0.5, n)
Q = theta_true * P + g_X + epsilon

# ============================================================
# Method 1: Naive OLS (Q on P only, ignoring X)
# ============================================================
ols_naive = LinearRegression()
ols_naive.fit(P.reshape(-1, 1), Q)
theta_naive = ols_naive.coef_[0]

# ============================================================
# Method 2: OLS with linear controls (Q on P and X)
# ============================================================
PX = np.column_stack([P, X])
ols_controls = LinearRegression()
ols_controls.fit(PX, Q)
theta_linear = ols_controls.coef_[0]

# ============================================================
# Method 3: Double Machine Learning with cross-fitting
# ============================================================
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

Q_tilde = np.zeros(n)
P_tilde = np.zeros(n)

for train_idx, test_idx in kf.split(X):
    # Train outcome model: predict Q from X
    rf_q = RandomForestRegressor(n_estimators=200, max_depth=10,
                                  min_samples_leaf=5, random_state=42)
    rf_q.fit(X[train_idx], Q[train_idx])
    Q_tilde[test_idx] = Q[test_idx] - rf_q.predict(X[test_idx])

    # Train treatment model: predict P from X
    rf_p = RandomForestRegressor(n_estimators=200, max_depth=10,
                                  min_samples_leaf=5, random_state=42)
    rf_p.fit(X[train_idx], P[train_idx])
    P_tilde[test_idx] = P[test_idx] - rf_p.predict(X[test_idx])

# Final estimate: regress Q_tilde on P_tilde
theta_dml = np.sum(P_tilde * Q_tilde) / np.sum(P_tilde**2)

# Standard error
residuals = Q_tilde - theta_dml * P_tilde
se_dml = np.sqrt(np.mean(residuals**2 * P_tilde**2) /
                 (np.mean(P_tilde**2)**2 * n))

print(f"True theta:          {theta_true:.3f}")
print(f"Naive OLS:           {theta_naive:.3f}")
print(f"OLS + linear ctrl:   {theta_linear:.3f}")
print(f"DML estimate:        {theta_dml:.3f} (SE: {se_dml:.4f})")
print(f"DML 95% CI:          [{theta_dml - 1.96*se_dml:.3f}, "
      f"{theta_dml + 1.96*se_dml:.3f}]")

# --- Visualization ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Residualized scatter
ax = axes[0]
ax.scatter(P_tilde, Q_tilde, alpha=0.1, s=6, color='#888888')
P_grid = np.linspace(P_tilde.min(), P_tilde.max(), 100)
ax.plot(P_grid, theta_dml * P_grid, 'b-', linewidth=2,
        label=rf'DML: $\hat{{\theta}} = {theta_dml:.3f}$')
ax.plot(P_grid, theta_true * P_grid, 'g--', linewidth=2,
        label=rf'True: $\theta = {theta_true:.3f}$')
ax.set_xlabel(r'Price residual $\tilde{P}$', fontsize=12)
ax.set_ylabel(r'Quantity residual $\tilde{Q}$', fontsize=12)
ax.set_title('DML: Residualized Regression', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Right: Comparison of estimates
ax = axes[1]
methods = ['Naive OLS', 'OLS + Linear\nControls', 'DML']
estimates = [theta_naive, theta_linear, theta_dml]
colors = ['#e57373', '#ffa726', '#4fc3f7']

bars = ax.bar(methods, estimates, color=colors, edgecolor='white', width=0.5)
ax.axhline(y=theta_true, color='#66bb6a', linewidth=2, linestyle='--',
           label=rf'True $\theta = {theta_true}$')
ax.set_ylabel(r'Estimated $\hat{\theta}$', fontsize=12)
ax.set_title('Comparison of Estimation Methods', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

for bar, est in zip(bars, estimates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{est:.2f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('dml_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

The naive OLS estimate will be substantially biased because the nonlinear confounders create endogeneity that a simple price regression cannot handle. OLS with linear controls will do better but still be biased because the true confounder effects are nonlinear. DML, using random forests to capture those nonlinearities, will recover the true \\(\theta = -1.5\\) within a tight confidence interval.

### Demonstration 3: Instrument Strength and 2SLS Performance

Finally, we visualize how instrument strength (measured by the first-stage F-statistic) affects the quality of IV estimates.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(99)
n = 1000
n_simulations = 500
theta_true = -2.0

# Vary instrument strength by changing Cov(Z, P)
# gamma controls Z -> P relationship: P = 20 + gamma*Z + 0.6*eps + noise
gammas = np.linspace(0.02, 1.0, 30)

median_estimates = []
iqr_lower = []
iqr_upper = []
mean_F_stats = []

for gamma in gammas:
    estimates = []
    F_stats = []

    for sim in range(n_simulations):
        eps = np.random.normal(0, 10, n)
        Z = np.random.normal(50, 15, n)
        P = 20 + gamma * Z + 0.6 * eps + np.random.normal(0, 3, n)
        Q = 100 + theta_true * P + eps

        # Stage 1
        s1 = LinearRegression().fit(Z.reshape(-1, 1), P)
        P_hat = s1.predict(Z.reshape(-1, 1))

        # F-statistic for first stage
        SS_res = np.sum((P - P_hat)**2)
        SS_tot = np.sum((P - P.mean())**2)
        R2 = 1 - SS_res / SS_tot
        F_stat = R2 / (1 - R2) * (n - 2)
        F_stats.append(F_stat)

        # Stage 2
        s2 = LinearRegression().fit(P_hat.reshape(-1, 1), Q)
        estimates.append(s2.coef_[0])

    estimates = np.array(estimates)
    median_estimates.append(np.median(estimates))
    iqr_lower.append(np.percentile(estimates, 25))
    iqr_upper.append(np.percentile(estimates, 75))
    mean_F_stats.append(np.mean(F_stats))

median_estimates = np.array(median_estimates)
iqr_lower = np.array(iqr_lower)
iqr_upper = np.array(iqr_upper)
mean_F_stats = np.array(mean_F_stats)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Estimates vs F-statistic
ax = axes[0]
ax.fill_between(mean_F_stats, iqr_lower, iqr_upper,
                alpha=0.25, color='#4fc3f7', label='IQR (25th–75th)')
ax.plot(mean_F_stats, median_estimates, 'b-', linewidth=2,
        label='Median 2SLS estimate')
ax.axhline(y=theta_true, color='#66bb6a', linewidth=2, linestyle='--',
           label=rf'True $\theta = {theta_true}$')
ax.axvline(x=10, color='#e57373', linewidth=1.5, linestyle=':',
           label=r'$F = 10$ threshold')
ax.set_xlabel(r'First-stage $F$-statistic', fontsize=12)
ax.set_ylabel(r'2SLS estimate $\hat{\theta}$', fontsize=12)
ax.set_title('2SLS Bias and Variance vs. Instrument Strength', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(mean_F_stats) + 5)

# Right: Spread (IQR width) vs F-statistic
ax = axes[1]
iqr_width = iqr_upper - iqr_lower
ax.plot(mean_F_stats, iqr_width, 'o-', color='#e57373',
        markersize=4, linewidth=1.5)
ax.axvline(x=10, color='#e57373', linewidth=1.5, linestyle=':',
           label=r'$F = 10$ threshold')
ax.set_xlabel(r'First-stage $F$-statistic', fontsize=12)
ax.set_ylabel(r'IQR width of $\hat{\theta}$', fontsize=12)
ax.set_title('Estimation Precision vs. Instrument Strength', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(mean_F_stats) + 5)

plt.tight_layout()
plt.savefig('iv_strength.png', dpi=150, bbox_inches='tight')
plt.show()
```

This simulation shows the dramatic difference between weak and strong instruments. When \\(F < 10\\), the 2SLS estimates are wildly spread out and badly biased (often biased toward the OLS estimate). As \\(F\\) increases past 10, 20, 50, the estimates converge tightly around the true value. The right panel makes the precision gain explicit: the interquartile range shrinks rapidly with instrument strength. This is why the "F > 10" rule is so important in practice.

---

## The State of the Art

The methods in this post --- IV, 2SLS, DML --- are the backbone of causal demand estimation. But the frontier keeps moving. Here is where things stand in 2026.

**DML with transformer-based nuisance models.** The ML models used in the DML partialing-out step are typically random forests or gradient-boosted trees. But there is nothing in the theory that restricts the choice. Recent work has used transformer-based forecasting models (temporal fusion transformers, time-series foundation models) as the nuisance estimators, especially for demand estimation in settings with complex temporal patterns. These models capture long-range dependencies and seasonality better than tree-based methods, leading to cleaner residuals and more precise causal estimates.

**Two-stage deep learning for airlines.** The airline industry, where demand estimation is both critical and extremely difficult (complex fare structures, competitive dynamics, and severe inventory constraints), has seen a shift from traditional econometric methods to deep learning approaches. A two-stage architecture --- deep networks for the nuisance estimation, followed by a linear causal stage --- has been reported to reduce estimation error from roughly 25% to about 4% on held-out data, compared to classical log-linear demand models. The key is the first stage's ability to control for the extremely high-dimensional confounder space (day of week, time of year, route characteristics, competitive capacity, booking curve dynamics).

**A/B testing as the gold standard.** Randomized price experiments eliminate endogeneity entirely by construction. If you randomly assign prices to customers, price is independent of the demand shock, and OLS is unbiased. Companies like Amazon, Uber, and Airbnb run large-scale price experiments. But there are costs: lost revenue during the experiment (you are deliberately setting some prices "wrong"), ethical concerns (charging different people different prices for identical products can generate backlash), and statistical challenges (interference between experimental units, long-run effects that differ from short-run responses). In practice, A/B testing is used for periodic calibration, while DML-type methods provide ongoing demand estimates between experiments.

**Synthetic instruments.** A growing line of research uses the structure of the problem itself to generate instruments, rather than relying on finding "natural" instruments in the wild. For example, in multi-product firms, the cost structure across products can generate instruments for individual product prices. In dynamic settings, lagged cost shocks or policy changes provide temporal variation.

**Bridge to Part 5.** We now have the tools to estimate demand from observational data. We can recover the true price elasticity even when prices are endogenous. The next question: how do we *use* this estimate in real time? If demand is uncertain and changes over time, how do we set prices that balance exploitation (charging the price we think is optimal) with exploration (learning more about demand)? This is the **dynamic pricing** problem, and it takes us into the world of multi-armed bandits, Thompson sampling, and reinforcement learning. That is Part 5.

---

*This is Part 4 of a 5-part series on pricing strategy. [Part 1: Demand, Elasticity & Markup](/2026/04/13/demand-elasticity-monopolist-markup.html) | [Part 2: Price Discrimination](/2026/04/14/price-discrimination-extracting-surplus.html) | [Part 3: Game Theory of Pricing](/2026/04/15/game-theory-competitive-pricing.html) | **Part 4: Causal Demand Estimation** | [Part 5: Algorithmic Dynamic Pricing](/2026/04/17/algorithmic-dynamic-pricing-bandits.html)*
