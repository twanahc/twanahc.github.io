---
layout: post
title: "Demand, Elasticity, and the Monopolist's Markup: The Mathematics of Pricing Power"
date: 2026-04-13
category: business
---

*This is Part 1 of a 5-part series on pricing strategy. **Part 1: Demand, Elasticity & Markup** | [Part 2: Price Discrimination](/2026/04/14/price-discrimination-extracting-surplus.html) | [Part 3: Game Theory of Pricing](/2026/04/15/game-theory-competitive-pricing.html) | [Part 4: Causal Demand Estimation](/2026/04/16/causal-demand-estimation-ml.html) | [Part 5: Algorithmic Dynamic Pricing](/2026/04/17/algorithmic-dynamic-pricing-bandits.html)*

Every business decision about what to charge — whether it's a SaaS company agonizing over $29/month versus $49/month, a pharmaceutical company pricing a new drug, or Amazon making 2.5 million automated price changes per day — ultimately rests on a single mathematical relationship: how the quantity people want to buy responds to the price you set. That relationship is captured in a derivative, and the entire edifice of pricing strategy is built on top of it.

This might sound reductive. Surely pricing involves brand positioning, competitive dynamics, cost structures, customer psychology? It does. But every one of those factors operates *through* the demand function. Brand loyalty makes demand less sensitive to price. A competitor's price cut shifts your demand curve. Customer psychology determines the shape of the curve. The math doesn't ignore these forces — it absorbs them. Once you understand the mathematical machinery, every pricing concept you've ever encountered clicks into place as a special case.

This post builds that machinery from scratch. We'll start with what a demand function actually is, derive the concept of elasticity that makes demand comparable across products and markets, connect elasticity to revenue, and arrive at the result that sits at the heart of all pricing theory: the monopolist's optimal markup is the reciprocal of demand elasticity. Along the way, we'll build geometric intuition with diagrams, verify everything with Python simulations, and set up the framework that the rest of this series will build on.

---

## Table of Contents

1. [The Demand Function](#1-the-demand-function)
2. [Price Elasticity of Demand](#2-price-elasticity-of-demand)
3. [Arc vs. Point Elasticity](#3-arc-vs-point-elasticity)
4. [The Revenue-Elasticity Connection](#4-the-revenue-elasticity-connection)
5. [Consumer and Producer Surplus](#5-consumer-and-producer-surplus)
6. [The Profit-Maximizing Firm: MR = MC](#6-the-profit-maximizing-firm-mr--mc)
7. [The Lerner Index — Markup as a Function of Elasticity](#7-the-lerner-index--markup-as-a-function-of-elasticity)
8. [Cross-Price and Income Elasticity](#8-cross-price-and-income-elasticity)
9. [Python Simulations](#9-python-simulations)
10. [From Theory to Practice](#10-from-theory-to-practice)

---

## 1. The Demand Function

Let's start at the very beginning. A **demand function** is a mathematical expression that tells you how many units of a product consumers will purchase, given the relevant economic variables. In its most general form:

$$
Q = f(P, \; I, \; P_s, \; P_c, \; T, \; N, \; \ldots)
$$

where:

- \(Q\) is the **quantity demanded** — the number of units consumers want to buy
- \(P\) is the **price** of the good itself
- \(I\) is **consumer income** (or aggregate income in a market)
- \(P_s\) represents the **prices of substitute goods** — products that serve a similar purpose (Pepsi is a substitute for Coca-Cola; AWS is a substitute for GCP)
- \(P_c\) represents the **prices of complementary goods** — products consumed together (printers and ink cartridges; game consoles and games)
- \(T\) captures **tastes and preferences** — things like brand perception, trends, seasonal effects
- \(N\) is the **number of buyers** in the market

The most fundamental empirical regularity in economics is the **law of demand**: for virtually all goods, quantity demanded decreases when price increases, holding everything else constant. Mathematically:

$$
\frac{\partial Q}{\partial P} < 0
$$

This partial derivative is negative. When you raise the price, people buy less. This isn't a theorem — it's an empirical observation so robust that economists call it a "law." There are theoretical exceptions (Giffen goods, Veblen goods), but they're rare enough in practice that the law of demand is as close to a universal truth as economics gets.

### The Linear Demand Model

The simplest and most commonly used demand specification is **linear demand**:

$$
Q = a - bP
$$

where \(a\) and \(b\) are positive constants. What do these parameters mean?

- \(a\) is the **quantity intercept** — the quantity demanded if the price were zero. Think of it as the total "appetite" for the product when it's free. For a streaming service, \(a\) might represent how many people would sign up if it cost nothing.
- \(b\) is the **price sensitivity** — how many units of demand you lose for each one-unit increase in price. A large \(b\) means demand is very responsive to price changes.

The law of demand is automatically satisfied here because \(\partial Q / \partial P = -b < 0\).

We can rearrange this into the **inverse demand function**, which expresses price as a function of quantity:

$$
P = \frac{a}{b} - \frac{1}{b}Q
$$

The inverse demand tells you: given that you want to sell \(Q\) units, what's the highest price you can charge? The quantity intercept in inverse demand is \(a/b\) — the maximum price anyone would pay for the first unit, sometimes called the **choke price** or **reservation price**.

### Demand Curve vs. Demand Function

A crucial distinction that trips up many people: the **demand curve** is a two-dimensional plot of \(Q\) versus \(P\) (or equivalently \(P\) versus \(Q\)), holding all other variables fixed. When we draw a downward-sloping line on a graph, we're slicing through the full demand function at specific values of income, competitor prices, preferences, and so on.

When one of those other variables changes — say income rises, or a competitor raises their price — the entire demand curve **shifts**. A movement along the curve (caused by a change in own price) is called a "change in quantity demanded." A shift of the curve (caused by a change in income, tastes, or other prices) is called a "change in demand." This distinction matters enormously for pricing analysis: if your sales dropped, was it because you raised price (movement along the curve) or because a competitor launched a better product (shift of the curve)?

<svg viewBox="0 0 500 400" xmlns="http://www.w3.org/2000/svg">
  <rect width="500" height="400" fill="#1a1a1a"/>
  <!-- Axes -->
  <line x1="70" y1="20" x2="70" y2="340" stroke="#d4d4d4" stroke-width="1.5"/>
  <line x1="70" y1="340" x2="470" y2="340" stroke="#d4d4d4" stroke-width="1.5"/>
  <!-- Axis labels -->
  <text x="270" y="380" fill="#d4d4d4" font-size="14" text-anchor="middle" font-family="Georgia, serif">Quantity (Q)</text>
  <text x="20" y="180" fill="#d4d4d4" font-size="14" text-anchor="middle" font-family="Georgia, serif" transform="rotate(-90, 20, 180)">Price (P)</text>
  <!-- Demand curve -->
  <line x1="70" y1="40" x2="430" y2="340" stroke="#6db3f2" stroke-width="2.5"/>
  <!-- Intercept labels -->
  <text x="55" y="40" fill="#e8e8e8" font-size="13" text-anchor="end" font-family="Georgia, serif">a/b</text>
  <text x="430" y="365" fill="#e8e8e8" font-size="13" text-anchor="middle" font-family="Georgia, serif">a</text>
  <!-- Dashed lines to intercepts -->
  <line x1="70" y1="40" x2="70" y2="40" stroke="#555" stroke-width="1" stroke-dasharray="4,3"/>
  <line x1="430" y1="340" x2="430" y2="340" stroke="#555" stroke-width="1" stroke-dasharray="4,3"/>
  <!-- Arrowheads -->
  <polygon points="470,340 460,335 460,345" fill="#d4d4d4"/>
  <polygon points="70,20 65,30 75,30" fill="#d4d4d4"/>
  <!-- Label the curve -->
  <text x="310" y="155" fill="#6db3f2" font-size="14" font-family="Georgia, serif">Q = a − bP</text>
  <!-- Slope annotation -->
  <text x="175" y="130" fill="#999" font-size="12" font-family="Georgia, serif">slope = −b</text>
  <line x1="160" y1="140" x2="200" y2="180" stroke="#999" stroke-width="1" stroke-dasharray="3,3" marker-end="url(#arrowGray)"/>
  <defs>
    <marker id="arrowGray" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#999"/>
    </marker>
  </defs>
  <!-- Title -->
  <text x="270" y="16" fill="#e8e8e8" font-size="15" text-anchor="middle" font-family="Georgia, serif" font-weight="bold">Linear Demand Curve</text>
</svg>

The diagram above shows a linear demand curve. At a price of \(a/b\), quantity demanded is zero — nobody buys. At a price of zero, quantity demanded is \(a\) — everyone who could possibly want it takes one. Between these extremes, the relationship is a straight line with slope \(-b\) (in \(Q\)-\(P\) space, the slope is \(-b\); in the more conventional \(P\)-\(Q\) space used in economics textbooks, the slope of the inverse demand is \(-1/b\)).

---

## 2. Price Elasticity of Demand

The slope of the demand curve, \(dQ/dP = -b\), tells you something useful: for each $1 increase in price, you lose \(b\) units of sales. But it has a serious problem — **it depends on the units you choose**.

If you measure quantity in gallons and price in dollars, you get one number. Switch to liters and euros, and you get a completely different number. You can't compare the price sensitivity of gasoline in the US (measured in gallons and dollars) to gasoline in France (measured in liters and euros) using raw slopes. And you certainly can't compare the price sensitivity of gasoline to the price sensitivity of software subscriptions using slopes.

We need a **unit-free** measure of price sensitivity. The solution is to measure everything in percentages. This gives us the **price elasticity of demand** (PED):

$$
\varepsilon = \frac{dQ}{dP} \cdot \frac{P}{Q} = \frac{\%\Delta Q}{\%\Delta P}
$$

Elasticity asks: if price increases by 1%, by what percentage does quantity demanded change? Since \(dQ/dP < 0\) for normal goods, elasticity is typically negative. Many textbooks work with the absolute value \(|\varepsilon|\) to avoid carrying minus signs everywhere, but I'll keep the sign explicit and use absolute values only when the context demands it.

### The Key Ranges

The magnitude of elasticity tells you the nature of demand:

- **\(|\varepsilon| > 1\)**: demand is **elastic**. A 1% price increase causes more than a 1% drop in quantity. Consumers are very responsive to price. Examples: luxury goods, products with close substitutes, non-essentials.
- **\(|\varepsilon| < 1\)**: demand is **inelastic**. A 1% price increase causes less than a 1% drop in quantity. Consumers are relatively unresponsive. Examples: insulin, gasoline (short-run), addictive substances, products with no close substitutes.
- **\(|\varepsilon| = 1\)**: demand is **unit elastic**. The percentage change in quantity exactly matches the percentage change in price. This turns out to be the point where revenue is maximized — a fact we'll derive shortly.

### Elasticity Along a Linear Demand Curve

Here's a fact that surprises many people: **elasticity is not constant along a linear demand curve**, even though the slope is constant. Let's see why.

For \(Q = a - bP\), we have \(dQ/dP = -b\). Plugging into the elasticity formula:

$$
\varepsilon = -b \cdot \frac{P}{a - bP}
$$

This depends on where you are on the curve — specifically, on the price \(P\) (which determines the ratio \(P/Q\)). Let's evaluate at the key points:

- **At \(P = 0\)** (bottom of the curve, \(Q = a\)): \(\varepsilon = -b \cdot 0/a = 0\). Perfectly inelastic.
- **At \(Q = 0\)** (top of the curve, \(P = a/b\)): \(\varepsilon = -b \cdot (a/b)/0 \to -\infty\). Perfectly elastic.
- **At the midpoint \(P = a/(2b)\)**, meaning \(Q = a/2\): \(\varepsilon = -b \cdot \frac{a/(2b)}{a/2} = -1\). Unit elastic.

So the upper half of a linear demand curve (high prices, low quantities) is elastic, and the lower half (low prices, high quantities) is inelastic. The midpoint is unit elastic. This is a critical insight: a firm operating on the upper (elastic) part of the curve faces very different pricing incentives than one on the lower (inelastic) part.

<svg viewBox="0 0 500 420" xmlns="http://www.w3.org/2000/svg">
  <rect width="500" height="420" fill="#1a1a1a"/>
  <!-- Axes -->
  <line x1="70" y1="20" x2="70" y2="350" stroke="#d4d4d4" stroke-width="1.5"/>
  <line x1="70" y1="350" x2="470" y2="350" stroke="#d4d4d4" stroke-width="1.5"/>
  <!-- Axis labels -->
  <text x="270" y="390" fill="#d4d4d4" font-size="14" text-anchor="middle" font-family="Georgia, serif">Quantity (Q)</text>
  <text x="20" y="185" fill="#d4d4d4" font-size="14" text-anchor="middle" font-family="Georgia, serif" transform="rotate(-90, 20, 185)">Price (P)</text>
  <!-- Arrowheads -->
  <polygon points="470,350 460,345 460,355" fill="#d4d4d4"/>
  <polygon points="70,20 65,30 75,30" fill="#d4d4d4"/>
  <!-- Elastic region (upper half) -->
  <line x1="70" y1="40" x2="250" y2="195" stroke="#f2736d" stroke-width="2.5"/>
  <!-- Unit elastic point -->
  <circle cx="250" cy="195" r="5" fill="#f0c040" stroke="#1a1a1a" stroke-width="1"/>
  <!-- Inelastic region (lower half) -->
  <line x1="250" y1="195" x2="430" y2="350" stroke="#6db3f2" stroke-width="2.5"/>
  <!-- Region labels -->
  <text x="140" y="95" fill="#f2736d" font-size="13" font-family="Georgia, serif" font-weight="bold">Elastic</text>
  <text x="130" y="112" fill="#f2736d" font-size="12" font-family="Georgia, serif">|ε| > 1</text>
  <text x="350" y="290" fill="#6db3f2" font-size="13" font-family="Georgia, serif" font-weight="bold">Inelastic</text>
  <text x="350" y="307" fill="#6db3f2" font-size="12" font-family="Georgia, serif">|ε| &lt; 1</text>
  <!-- Unit elastic label -->
  <text x="265" y="185" fill="#f0c040" font-size="12" font-family="Georgia, serif" font-weight="bold">|ε| = 1</text>
  <!-- Intercept labels -->
  <text x="55" y="40" fill="#e8e8e8" font-size="12" text-anchor="end" font-family="Georgia, serif">a/b</text>
  <text x="430" y="370" fill="#e8e8e8" font-size="12" text-anchor="middle" font-family="Georgia, serif">a</text>
  <!-- Midpoint label -->
  <text x="55" y="195" fill="#e8e8e8" font-size="12" text-anchor="end" font-family="Georgia, serif">a/2b</text>
  <line x1="60" y1="195" x2="70" y2="195" stroke="#555" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="250" y="370" fill="#e8e8e8" font-size="12" text-anchor="middle" font-family="Georgia, serif">a/2</text>
  <line x1="250" y1="350" x2="250" y2="195" stroke="#555" stroke-width="1" stroke-dasharray="3,3"/>
  <!-- Title -->
  <text x="270" y="16" fill="#e8e8e8" font-size="15" text-anchor="middle" font-family="Georgia, serif" font-weight="bold">Elasticity Varies Along a Linear Demand Curve</text>
</svg>

The diagram makes this visual. As you slide down the demand curve from high price to low price, you cross from the elastic regime into the inelastic regime, passing through unit elasticity at the midpoint.

---

## 3. Arc vs. Point Elasticity

The elasticity formula \(\varepsilon = (dQ/dP)(P/Q)\) is the **point elasticity** — it uses calculus and gives you the exact elasticity at a specific point on the demand curve. This is precise and elegant when you know the functional form of demand.

But in practice, you often don't have a nice differentiable demand function. You have data: "when we charged $20, we sold 500 units; when we raised to $25, we sold 400 units." You need to compute elasticity from two discrete observations. The naive approach would be:

$$
\varepsilon \approx \frac{(Q_2 - Q_1)/Q_1}{(P_2 - P_1)/P_1}
$$

This has an ugly problem: the result depends on which point you use as the "base." If you compute the percentage change starting from \((P_1, Q_1)\), you get one number; starting from \((P_2, Q_2)\), you get a different number. This asymmetry is unacceptable.

The **arc elasticity** (also called the **midpoint method**) fixes this by using the average of the two points as the base:

$$
\varepsilon_{\text{arc}} = \frac{(Q_2 - Q_1) / \left(\frac{Q_1 + Q_2}{2}\right)}{(P_2 - P_1) / \left(\frac{P_1 + P_2}{2}\right)} = \frac{Q_2 - Q_1}{P_2 - P_1} \cdot \frac{P_1 + P_2}{Q_1 + Q_2}
$$

This gives the same answer regardless of which point you call "1" and which you call "2." It's the standard approach for empirical elasticity estimation from discrete price experiments.

### Constant-Elasticity Demand

For the linear demand model, elasticity varies along the curve. But there's another widely used specification where elasticity is constant everywhere — the **constant-elasticity** (or **iso-elastic**) demand function:

$$
Q = AP^{\varepsilon}
$$

where \(A > 0\) is a scale parameter and \(\varepsilon < 0\) is the elasticity. Let's verify that elasticity is indeed constant:

$$
\frac{dQ}{dP} = A\varepsilon P^{\varepsilon - 1}
$$

$$
\varepsilon_{\text{PED}} = \frac{dQ}{dP} \cdot \frac{P}{Q} = A\varepsilon P^{\varepsilon - 1} \cdot \frac{P}{AP^{\varepsilon}} = \varepsilon
$$

The elasticity is just the parameter \(\varepsilon\), regardless of price. Taking logs of both sides gives the **log-log specification**:

$$
\ln Q = \ln A + \varepsilon \ln P
$$

This is a straight line in log-log space, which is why constant-elasticity demand is so popular in empirical work — you just run a linear regression of log-quantity on log-price, and the slope coefficient is the elasticity directly. This log-log regression is one of the workhorses of demand estimation and will reappear in Part 4 when we discuss causal methods.

---

## 4. The Revenue-Elasticity Connection

Now we arrive at the result that connects elasticity to the thing businesses actually care about most directly: **revenue**. Total revenue is simply price times quantity:

$$
R = P \cdot Q
$$

What happens to revenue when you raise the price? There are two opposing effects. The higher price means you earn more per unit sold (good for revenue). But by the law of demand, the higher price also means you sell fewer units (bad for revenue). Which effect wins? The answer depends entirely on elasticity.

### Deriving dR/dP

Let's differentiate revenue with respect to price. Since \(Q\) depends on \(P\), we use the product rule:

$$
\frac{dR}{dP} = \frac{d(PQ)}{dP} = Q + P\frac{dQ}{dP}
$$

The first term \(Q\) is the **price effect** — selling existing units at a higher price. The second term \(P \cdot dQ/dP\) is the **quantity effect** — losing sales due to the price increase (this term is negative).

Factor out \(Q\):

$$
\frac{dR}{dP} = Q\left(1 + \frac{P}{Q}\frac{dQ}{dP}\right) = Q\left(1 + \frac{1}{\varepsilon}\right)
$$

Wait — I need to be careful with the sign convention. We defined \(\varepsilon = (dQ/dP)(P/Q)\), which is negative. So the expression inside the parentheses is \(1 + 1/\varepsilon\), where \(\varepsilon < 0\). Let's think through the cases:

**Case 1: Elastic demand (\(|\varepsilon| > 1\), so \(\varepsilon < -1\))**

The term \(1/\varepsilon\) is between \(-1\) and \(0\), so \(1 + 1/\varepsilon\) is between \(0\) and \(1\). Since \(Q > 0\), we get \(dR/dP > 0\)... wait, that doesn't seem right. Let me recheck.

Actually, if \(\varepsilon = -2\), then \(1/\varepsilon = -0.5\), so \(1 + 1/\varepsilon = 0.5 > 0\), meaning \(dR/dP > 0\). But that says raising price increases revenue when demand is elastic, which contradicts standard results. The issue is that I wrote \(\varepsilon\) where I should be more careful.

Let me redo this properly. With \(\varepsilon < 0\):

- If \(|\varepsilon| > 1\) (elastic), then \(\varepsilon < -1\), so \(1/\varepsilon > -1\), so \(1 + 1/\varepsilon > 0\). This means \(dR/dP > 0\)... 

Hmm — actually, this is correct when we think about it from the revenue-quantity side. Let me instead derive \(dR/dQ\), which is marginal revenue, and then think about the price side more carefully.

The confusion arises because a price *increase* causes a quantity *decrease* on the elastic portion, and total revenue falls. Let's rewrite:

$$
\frac{dR}{dP} = Q + P\frac{dQ}{dP}
$$

When \(|\varepsilon| > 1\): the quantity effect \(P \cdot dQ/dP\) (negative, large in magnitude) dominates the price effect \(Q\) (positive). So \(dR/dP < 0\): raising price decreases revenue. To verify: if \(\varepsilon = -3\) at a point where \(P = 10, Q = 100\), then \(dQ/dP = \varepsilon Q/P = -30\). So \(dR/dP = 100 + 10(-30) = 100 - 300 = -200 < 0\). Correct.

When \(|\varepsilon| < 1\): the price effect dominates. \(dR/dP > 0\): raising price increases revenue.

When \(|\varepsilon| = 1\): the two effects exactly cancel. \(dR/dP = 0\): revenue is at a maximum (or minimum, but it's a maximum for reasonable demand curves).

Let me reconcile with the factored form. We have:

$$
\frac{dR}{dP} = Q\left(1 + \frac{1}{\varepsilon}\right)
$$

With \(\varepsilon = -3\): \(1 + 1/(-3) = 1 - 1/3 = 2/3 > 0\). But we just showed \(dR/dP < 0\). The issue is that I factored incorrectly. Let me redo it:

$$
\frac{dR}{dP} = Q + P\frac{dQ}{dP}
$$

Hmm, but we can write \(P \cdot dQ/dP = (P/Q) \cdot Q \cdot (dQ/dP) \cdot (Q/Q)\)... No, more simply:

$$
\frac{dR}{dP} = Q + P\frac{dQ}{dP} = Q\left(1 + \frac{P}{Q}\frac{dQ}{dP}\right) = Q(1 + \varepsilon)
$$

Wait — no. Let me be very careful:

$$
\frac{P}{Q}\frac{dQ}{dP} = \varepsilon
$$

So:

$$
P\frac{dQ}{dP} = \varepsilon \cdot Q
$$

Therefore:

$$
\frac{dR}{dP} = Q + \varepsilon Q = Q(1 + \varepsilon)
$$

Now with \(\varepsilon = -3\): \(Q(1 + (-3)) = Q(-2) < 0\). With \(\varepsilon = -0.5\): \(Q(1 + (-0.5)) = Q(0.5) > 0\). With \(\varepsilon = -1\): \(Q(1 + (-1)) = 0\). This all checks out. So the correct factored form is:

$$
\boxed{\frac{dR}{dP} = Q(1 + \varepsilon)}
$$

Now the summary is clean:

| Demand Type | Elasticity | \(1 + \varepsilon\) | \(dR/dP\) | Effect of Price Increase |
|---|---|---|---|---|
| Elastic | \(\varepsilon < -1\) | Negative | Negative | Revenue **falls** |
| Unit elastic | \(\varepsilon = -1\) | Zero | Zero | Revenue **unchanged** (max) |
| Inelastic | \(-1 < \varepsilon < 0\) | Positive | Positive | Revenue **rises** |

This table is the single most important result for practical pricing. It tells you which side of the revenue hill you're sitting on. If your product has elastic demand and you raise prices, you'll lose revenue. If it has inelastic demand and you raise prices, you'll *gain* revenue. Revenue is maximized at the unit-elastic point.

### Marginal Revenue

Now let's derive **marginal revenue** — the additional revenue from selling one more unit. This requires thinking in terms of quantity, using the inverse demand function \(P(Q)\):

$$
R = P(Q) \cdot Q
$$

$$
MR = \frac{dR}{dQ} = P + Q\frac{dP}{dQ}
$$

The first term \(P\) is the revenue from selling the additional unit. The second term \(Q \cdot dP/dQ\) is negative — to sell one more unit, you must lower the price, and that lower price applies to *all* units, not just the marginal one. (This is assuming uniform pricing, where you charge everyone the same price. Price discrimination, the topic of Part 2, relaxes this.)

We can factor this using the inverse of elasticity. Since \(dP/dQ = 1/(dQ/dP)\):

$$
MR = P + Q\frac{dP}{dQ} = P\left(1 + \frac{Q}{P}\frac{dP}{dQ}\right) = P\left(1 + \frac{1}{\varepsilon}\right)
$$

So:

$$
\boxed{MR = P\left(1 + \frac{1}{\varepsilon}\right)}
$$

Note that \(MR = 0\) when \(\varepsilon = -1\), confirming that revenue is maximized at the unit-elastic point. When demand is elastic (\(\varepsilon < -1\)), marginal revenue is positive — selling more increases revenue. When demand is inelastic (\(-1 < \varepsilon < 0\)), marginal revenue is negative — selling more actually *decreases* revenue because the price cut needed to induce the extra sale hurts too much on all the other units.

For the linear inverse demand \(P = a/b - Q/b\):

$$
MR = \frac{a}{b} - \frac{2Q}{b}
$$

Notice that marginal revenue has the same intercept as the inverse demand curve but **twice the slope**. This is a general property of linear demand and is geometrically useful — the MR curve bisects the horizontal distance between the price axis and the demand curve.

---

## 5. Consumer and Producer Surplus

Before we optimize the firm's pricing decision, we need to understand the concept of **surplus** — the economic value that trade creates and how it's divided.

### Consumer Surplus

Think about buying a cup of coffee. You might be willing to pay up to $5 for it — that's your **reservation price** or **willingness to pay**. If the actual price is $3, you get the coffee AND you keep $2 that you would have been willing to spend. That $2 is your **consumer surplus** — the difference between what you were willing to pay and what you actually paid.

Aggregated over all consumers, consumer surplus is the area between the demand curve and the market price. The demand curve traces out the willingness to pay for each successive unit. Everyone whose willingness to pay exceeds the market price buys the good and gets a surplus equal to the difference.

For linear demand \(Q = a - bP\) at a market price \(P^*\):

$$
CS = \int_{P^*}^{a/b} (a - bP) \, dP = \left[aP - \frac{b}{2}P^2\right]_{P^*}^{a/b}
$$

$$
CS = \left(\frac{a^2}{b} - \frac{a^2}{2b}\right) - \left(aP^* - \frac{b}{2}P^{*2}\right) = \frac{a^2}{2b} - aP^* + \frac{b}{2}P^{*2}
$$

$$
CS = \frac{(a - bP^*)^2}{2b} = \frac{Q^{*2}}{2b}
$$

Geometrically, this is the area of the triangle between the demand curve, the price line, and the vertical axis: base \(Q^*\), height \(a/b - P^*\), so area \(= \frac{1}{2} Q^* (a/b - P^*)\). Since \(Q^* = a - bP^*\), we can verify: \(\frac{1}{2}(a - bP^*)(a/b - P^*) = \frac{(a-bP^*)^2}{2b}\). Checks out.

### Producer Surplus

**Producer surplus** is the analogous concept for sellers. It's the difference between the price received and the minimum price at which the seller would have been willing to supply — which is the marginal cost. For a firm with constant marginal cost \(c\):

$$
PS = (P^* - c) \cdot Q^*
$$

This is simply the firm's profit (when there are no fixed costs) — a rectangle with height \(P^* - c\) and width \(Q^*\).

### Total Surplus and Deadweight Loss

**Total surplus** is \(CS + PS\). In a perfectly competitive market where \(P = MC\), total surplus is maximized — there's no way to rearrange production or consumption to make someone better off without making someone else worse off (this is the First Welfare Theorem).

A monopolist, however, charges a price above marginal cost and produces less than the competitive quantity. This creates a **deadweight loss** — a triangle of surplus that nobody gets. Some consumers who valued the good above its marginal cost are priced out of the market. The surplus they would have generated vanishes.

<svg viewBox="0 0 520 420" xmlns="http://www.w3.org/2000/svg">
  <rect width="520" height="420" fill="#1a1a1a"/>
  <!-- Axes -->
  <line x1="70" y1="20" x2="70" y2="360" stroke="#d4d4d4" stroke-width="1.5"/>
  <line x1="70" y1="360" x2="480" y2="360" stroke="#d4d4d4" stroke-width="1.5"/>
  <text x="280" y="400" fill="#d4d4d4" font-size="14" text-anchor="middle" font-family="Georgia, serif">Quantity (Q)</text>
  <text x="18" y="190" fill="#d4d4d4" font-size="14" text-anchor="middle" font-family="Georgia, serif" transform="rotate(-90, 18, 190)">Price (P)</text>
  <polygon points="480,360 470,355 470,365" fill="#d4d4d4"/>
  <polygon points="70,20 65,30 75,30" fill="#d4d4d4"/>
  <!-- Demand curve: from (70,40) to (430,360) -->
  <line x1="70" y1="40" x2="430" y2="360" stroke="#6db3f2" stroke-width="2"/>
  <!-- MC line: horizontal at y=260 -->
  <line x1="70" y1="260" x2="460" y2="260" stroke="#66cc66" stroke-width="2" stroke-dasharray="6,3"/>
  <!-- MR curve: from (70,40) to (250,360) — steeper -->
  <line x1="70" y1="40" x2="250" y2="360" stroke="#cc66cc" stroke-width="1.5" stroke-dasharray="4,3"/>
  <!-- Monopoly quantity Q* where MR=MC: MR at y=260 => x = 70 + (260-40)/(360-40)*180 = 70 + 220/320*180 = 70+123.75 ≈ 194 -->
  <!-- Monopoly price P* on demand at Q*=194: y = 40 + (194-70)/(430-70)*(360-40) = 40 + 124/360*320 = 40+110 = 150 -->
  <!-- Competitive Q_c where P=MC on demand: y=260 => x = 70 + (260-40)/(360-40)*360 = 70 + 220/320*360 = 70+247.5 = 317.5 -->
  <!-- Consumer surplus: triangle above P* below demand, from x=70 to x=194 -->
  <polygon points="70,40 194,150 70,150" fill="#6db3f2" fill-opacity="0.25"/>
  <!-- Producer surplus: rectangle from MC to P*, Q=0 to Q* -->
  <polygon points="70,150 194,150 194,260 70,260" fill="#66cc66" fill-opacity="0.25"/>
  <!-- Deadweight loss triangle -->
  <polygon points="194,150 317,260 194,260" fill="#f2736d" fill-opacity="0.35"/>
  <!-- Dashed lines for monopoly Q* and P* -->
  <line x1="194" y1="150" x2="194" y2="360" stroke="#999" stroke-width="1" stroke-dasharray="4,3"/>
  <line x1="70" y1="150" x2="194" y2="150" stroke="#999" stroke-width="1" stroke-dasharray="4,3"/>
  <!-- Dashed line for competitive quantity -->
  <line x1="317" y1="260" x2="317" y2="360" stroke="#999" stroke-width="1" stroke-dasharray="4,3"/>
  <!-- Labels -->
  <text x="55" y="150" fill="#e8e8e8" font-size="12" text-anchor="end" font-family="Georgia, serif">P*</text>
  <text x="55" y="260" fill="#66cc66" font-size="12" text-anchor="end" font-family="Georgia, serif">MC</text>
  <text x="194" y="378" fill="#e8e8e8" font-size="12" text-anchor="middle" font-family="Georgia, serif">Q*</text>
  <text x="317" y="378" fill="#999" font-size="12" text-anchor="middle" font-family="Georgia, serif">Q_c</text>
  <!-- Region labels -->
  <text x="110" y="110" fill="#6db3f2" font-size="12" font-family="Georgia, serif">CS</text>
  <text x="120" y="215" fill="#66cc66" font-size="12" font-family="Georgia, serif">PS</text>
  <text x="235" y="225" fill="#f2736d" font-size="12" font-family="Georgia, serif" font-weight="bold">DWL</text>
  <!-- Curve labels -->
  <text x="440" y="345" fill="#6db3f2" font-size="12" font-family="Georgia, serif">D</text>
  <text x="260" y="370" fill="#cc66cc" font-size="12" font-family="Georgia, serif">MR</text>
  <text x="462" y="252" fill="#66cc66" font-size="12" font-family="Georgia, serif">MC</text>
  <!-- Title -->
  <text x="280" y="16" fill="#e8e8e8" font-size="15" text-anchor="middle" font-family="Georgia, serif" font-weight="bold">Monopoly Surplus and Deadweight Loss</text>
</svg>

The blue region is consumer surplus — what's left for consumers after paying the monopoly price. The green region is producer surplus — the monopolist's profit. The red triangle is deadweight loss — the transactions that would have been mutually beneficial (the buyer values the good above its marginal cost) but don't happen because the monopolist restricts output to keep prices high.

---

## 6. The Profit-Maximizing Firm: MR = MC

Now we're ready to solve the monopolist's optimization problem. A firm's **profit** is revenue minus cost:

$$
\pi = R(Q) - C(Q) = P(Q) \cdot Q - C(Q)
$$

where \(C(Q)\) is the total cost function. To maximize profit, we take the first derivative with respect to quantity and set it to zero:

$$
\frac{d\pi}{dQ} = \frac{dR}{dQ} - \frac{dC}{dQ} = MR - MC = 0
$$

This gives us the fundamental optimality condition:

$$
\boxed{MR = MC}
$$

**Marginal revenue equals marginal cost.** The firm should keep producing additional units as long as the revenue from the next unit exceeds its cost. When MR drops to equal MC, stop.

The **second-order condition** ensures this is a maximum, not a minimum:

$$
\frac{d^2\pi}{dQ^2} = \frac{dMR}{dQ} - \frac{dMC}{dQ} < 0
$$

This requires the MR curve to be falling faster than the MC curve at the optimum — which is typically satisfied for downward-sloping MR and non-decreasing MC.

### Worked Example

Let's work through a complete numerical example to make this concrete. Suppose:

- **Demand**: \(Q = 100 - 2P\), which gives inverse demand \(P = 50 - Q/2\)
- **Marginal cost**: \(MC = c = 10\) (constant)
- **No fixed costs**: \(C(Q) = 10Q\)

**Step 1: Find marginal revenue.** From the inverse demand \(P = 50 - Q/2\):

$$
R = PQ = \left(50 - \frac{Q}{2}\right)Q = 50Q - \frac{Q^2}{2}
$$

$$
MR = \frac{dR}{dQ} = 50 - Q
$$

As expected, MR has the same intercept (50) but twice the slope (\(-1\) vs \(-1/2\)) compared to inverse demand.

**Step 2: Set MR = MC.**

$$
50 - Q^* = 10 \implies Q^* = 40
$$

**Step 3: Find the monopoly price.** Plug \(Q^* = 40\) into inverse demand:

$$
P^* = 50 - \frac{40}{2} = 50 - 20 = 30
$$

**Step 4: Calculate profit.**

$$
\pi = (P^* - c) \cdot Q^* = (30 - 10)(40) = 800
$$

**Step 5: Calculate surplus.**

Consumer surplus:

$$
CS = \frac{1}{2}(50 - 30)(40) = \frac{1}{2}(20)(40) = 400
$$

Producer surplus (= profit with no fixed costs):

$$
PS = (30 - 10)(40) = 800
$$

Competitive outcome (where \(P = MC = 10\)): \(Q_c = 100 - 2(10) = 80\). Total surplus under competition:

$$
TS_{\text{comp}} = \frac{1}{2}(50 - 10)(80) = 1600
$$

Deadweight loss from monopoly:

$$
DWL = TS_{\text{comp}} - CS - PS = 1600 - 400 - 800 = 400
$$

Alternatively, the deadweight loss triangle has base \(Q_c - Q^* = 80 - 40 = 40\) and height \(P^* - MC = 30 - 10 = 20\), giving \(DWL = \frac{1}{2}(40)(20) = 400\). Consistent.

So the monopolist produces 40 units at $30 each, earning $800 in profit. But this comes at the cost of $400 in deadweight loss — value destroyed by the restriction of output.

---

## 7. The Lerner Index — Markup as a Function of Elasticity

We've established that the monopolist sets \(MR = MC\), and we've derived that \(MR = P(1 + 1/\varepsilon)\). Now we combine these two results to obtain the crown jewel of pricing theory.

Starting from the optimality condition:

$$
MR = MC
$$

$$
P\left(1 + \frac{1}{\varepsilon}\right) = MC
$$

Rearrange to isolate the relationship between price and marginal cost:

$$
P + \frac{P}{\varepsilon} = MC
$$

$$
P - MC = -\frac{P}{\varepsilon}
$$

$$
\frac{P - MC}{P} = -\frac{1}{\varepsilon}
$$

The left-hand side is the **Lerner Index**, denoted \(L\):

$$
\boxed{L = \frac{P - MC}{P} = -\frac{1}{\varepsilon} = \frac{1}{|\varepsilon|}}
$$

This is the fundamental pricing equation. It says: **the optimal proportional markup over marginal cost equals the reciprocal of the absolute value of demand elasticity**.

Let's unpack what this means with concrete examples:

- **Pharmaceutical drug with \(|\varepsilon| = 1.5\)** (few substitutes, medical necessity): \(L = 1/1.5 = 0.67\). The price is 67% above marginal cost. If MC = $10, the price is $30.
- **Restaurant meal with \(|\varepsilon| = 3\)**: \(L = 1/3 = 0.33\). A 33% markup. If MC = $10, price is $15.
- **Commodity wheat with \(|\varepsilon| = 10\)** (many substitutes, standardized product): \(L = 1/10 = 0.10\). Only a 10% markup. If MC = $10, price is $11.11.
- **SaaS tool with \(|\varepsilon| = 2\)** (some switching costs, moderate substitutes): \(L = 0.5\). A 50% markup.

### Why a Monopolist Never Prices on the Inelastic Portion

Here's a powerful corollary. If \(|\varepsilon| < 1\), then \(L = 1/|\varepsilon| > 1\), which means \((P - MC)/P > 1\), which means \(MC < 0\). But marginal cost can't be negative (producing one more unit can't save you money in any normal sense). Therefore, **a profit-maximizing monopolist will never operate where demand is inelastic**.

The intuition is direct: on the inelastic portion of the demand curve, marginal revenue is negative (recall that \(MR < 0\) when \(|\varepsilon| < 1\)). If marginal revenue is negative, selling one more unit *decreases* revenue. And producing one more unit *increases* costs. So the firm would be losing money on both fronts — less revenue and higher costs. It should cut output (raise price) until it reaches the elastic portion of the curve.

This is a testable prediction: monopolists should always price in the elastic region of demand. Empirically, this holds remarkably well.

### What the Lerner Index Really Tells Us

The Lerner Index connects **market structure** to **pricing power** through a single intermediate variable: elasticity.

- **More substitutes** → consumers can easily switch → demand is more elastic → higher \(|\varepsilon|\) → lower \(L\) → lower markup
- **Fewer substitutes** → consumers are stuck → demand is more inelastic → lower \(|\varepsilon|\) → higher \(L\) → higher markup
- **Stronger brand** → perceived as less substitutable → lower \(|\varepsilon|\) → higher \(L\)
- **Network effects** → switching costs → lower \(|\varepsilon|\) → higher \(L\)
- **Patent protection** → no substitutes → very low \(|\varepsilon|\) → very high \(L\)

In a perfectly competitive market, firms are price-takers (\(P = MC\)), so \(L = 0\). The Lerner Index ranges from 0 (perfect competition) toward 1 (extreme monopoly power). It's a direct measure of how far from the competitive ideal a market is.

---

## 8. Cross-Price and Income Elasticity

So far we've focused on own-price elasticity — how a product's demand responds to its own price. But the demand function \(Q = f(P, I, P_s, P_c, \ldots)\) depends on other variables too, and we can define elasticities with respect to each.

### Cross-Price Elasticity

The **cross-price elasticity** between good \(x\) and good \(y\) measures how the demand for \(x\) responds to a change in the price of \(y\):

$$
\varepsilon_{xy} = \frac{\partial Q_x}{\partial P_y} \cdot \frac{P_y}{Q_x}
$$

The sign tells you the relationship:

- **\(\varepsilon_{xy} > 0\)**: goods \(x\) and \(y\) are **substitutes**. When Coca-Cola raises its price (\(P_y\) up), demand for Pepsi rises (\(Q_x\) up). Consumers switch.
- **\(\varepsilon_{xy} < 0\)**: goods \(x\) and \(y\) are **complements**. When printer prices rise (\(P_y\) up), demand for ink cartridges falls (\(Q_x\) down). They're consumed together.
- **\(\varepsilon_{xy} = 0\)**: the goods are **independent**. The price of oranges doesn't affect demand for car insurance.

The magnitude tells you *how strongly* the goods are related. Cross-price elasticities are essential for:

- **Competitive strategy**: if your product has a high cross-price elasticity with a competitor, you're in a fierce price war. A small price cut by them steals a lot of your customers.
- **Bundling decisions**: bundling makes sense when goods are complements (\(\varepsilon_{xy} < 0\)). Microsoft bundles Word, Excel, and PowerPoint because they're complementary — demand for each increases when the others are available cheaply.
- **Merger analysis**: antitrust regulators use cross-price elasticities to define relevant markets. If the cross-price elasticity between two firms' products is high, they're in the same market, and a merger would reduce competition.

### Income Elasticity

The **income elasticity** of demand measures how demand responds to changes in consumer income:

$$
\varepsilon_I = \frac{\partial Q}{\partial I} \cdot \frac{I}{Q}
$$

This classifies goods into categories:

- **Normal goods** (\(\varepsilon_I > 0\)): demand increases with income. Most goods are normal.
  - **Necessities** (\(0 < \varepsilon_I < 1\)): demand increases, but less than proportionally. Food staples, utilities, basic clothing. As you get richer, you spend a *smaller* share of income on these.
  - **Luxuries** (\(\varepsilon_I > 1\)): demand increases more than proportionally. Fine dining, luxury cars, premium SaaS tiers. As you get richer, you spend a *larger* share of income on these.
- **Inferior goods** (\(\varepsilon_I < 0\)): demand *decreases* with income. Instant ramen, bus tickets (when you can afford a car), budget airlines.

### Engel Curves

An **Engel curve** plots the quantity demanded of a good against income, holding prices constant. For necessities, the Engel curve rises steeply at low income and flattens out at high income. For luxuries, it's the reverse — flat at low income and steep at high income. For inferior goods, it eventually slopes downward.

Income elasticities matter for pricing because they tell you how different customer segments respond to pricing. A product that's a necessity for low-income users but a luxury for high-income users has different pricing leverage across segments — a key input to the price discrimination strategies we'll explore in Part 2.

They also have macroeconomic implications: during recessions (falling incomes), demand for luxury goods collapses (\(\varepsilon_I > 1\) means large demand drop), demand for necessities barely budges (\(\varepsilon_I < 1\)), and demand for inferior goods actually rises (\(\varepsilon_I < 0\)). If you're pricing a product, knowing where it sits on the income-elasticity spectrum tells you how recession-proof your revenue is.

---

## 9. Python Simulations

Let's verify and visualize everything we've derived. Three simulations follow.

### 9.1 Demand, Elasticity, and Revenue Along a Linear Demand Curve

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters for linear demand Q = a - bP
a, b = 100, 2

# Price range (avoiding P=0 for elasticity computation)
P = np.linspace(0.01, a / b - 0.01, 500)
Q = a - b * P

# Elasticity along the curve
epsilon = -b * P / (a - b * P)

# Revenue
R = P * Q

fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=False)
fig.suptitle(r'Linear Demand: $Q = 100 - 2P$', fontsize=16, color='white', y=0.95)

# --- Plot 1: Demand Curve ---
ax1 = axes[0]
# Color the curve by elasticity regime
elastic_mask = np.abs(epsilon) > 1
inelastic_mask = np.abs(epsilon) < 1

ax1.plot(Q[elastic_mask], P[elastic_mask], color='#f2736d', linewidth=2.5, label=r'Elastic ($|\epsilon| > 1$)')
ax1.plot(Q[inelastic_mask], P[inelastic_mask], color='#6db3f2', linewidth=2.5, label=r'Inelastic ($|\epsilon| < 1$)')

# Unit elastic point
P_unit = a / (2 * b)
Q_unit = a / 2
ax1.plot(Q_unit, P_unit, 'o', color='#f0c040', markersize=10, zorder=5, label=r'Unit elastic ($|\epsilon| = 1$)')

ax1.set_xlabel(r'Quantity $Q$', fontsize=13)
ax1.set_ylabel(r'Price $P$', fontsize=13)
ax1.set_title('Demand Curve with Elasticity Regions', fontsize=14, color='white')
ax1.legend(fontsize=11)
ax1.set_facecolor('#1a1a1a')
ax1.tick_params(colors='#d4d4d4')
ax1.spines['bottom'].set_color('#555')
ax1.spines['left'].set_color('#555')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.xaxis.label.set_color('#d4d4d4')
ax1.yaxis.label.set_color('#d4d4d4')
ax1.title.set_color('#e8e8e8')

# --- Plot 2: Elasticity vs Price ---
ax2 = axes[1]
ax2.plot(P, np.abs(epsilon), color='#cc66cc', linewidth=2.5)
ax2.axhline(y=1, color='#f0c040', linestyle='--', linewidth=1.5, label=r'$|\epsilon| = 1$')
ax2.axvline(x=P_unit, color='#f0c040', linestyle=':', linewidth=1, alpha=0.5)

ax2.fill_between(P, np.abs(epsilon), 1, where=np.abs(epsilon) > 1, alpha=0.15, color='#f2736d', label='Elastic region')
ax2.fill_between(P, np.abs(epsilon), 1, where=np.abs(epsilon) < 1, alpha=0.15, color='#6db3f2', label='Inelastic region')

ax2.set_xlabel(r'Price $P$', fontsize=13)
ax2.set_ylabel(r'$|\epsilon|$', fontsize=13)
ax2.set_title('Elasticity Magnitude Along the Demand Curve', fontsize=14)
ax2.set_ylim(0, 10)
ax2.legend(fontsize=11)
ax2.set_facecolor('#1a1a1a')
ax2.tick_params(colors='#d4d4d4')
ax2.spines['bottom'].set_color('#555')
ax2.spines['left'].set_color('#555')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.xaxis.label.set_color('#d4d4d4')
ax2.yaxis.label.set_color('#d4d4d4')
ax2.title.set_color('#e8e8e8')

# --- Plot 3: Revenue vs Quantity ---
ax3 = axes[2]
ax3.plot(Q, R, color='#66cc66', linewidth=2.5, label=r'Revenue $R = PQ$')
ax3.axvline(x=Q_unit, color='#f0c040', linestyle='--', linewidth=1.5, label=r'$Q$ at $|\epsilon|=1$')
ax3.plot(Q_unit, P_unit * Q_unit, 'o', color='#f0c040', markersize=10, zorder=5)
ax3.annotate(rf'Max Revenue = {P_unit * Q_unit:.0f}', xy=(Q_unit, P_unit * Q_unit),
             xytext=(Q_unit + 12, P_unit * Q_unit + 50),
             fontsize=11, color='#f0c040',
             arrowprops=dict(arrowstyle='->', color='#f0c040'))

ax3.set_xlabel(r'Quantity $Q$', fontsize=13)
ax3.set_ylabel(r'Revenue $R$', fontsize=13)
ax3.set_title('Revenue Curve (Bell-Shaped)', fontsize=14)
ax3.legend(fontsize=11)
ax3.set_facecolor('#1a1a1a')
ax3.tick_params(colors='#d4d4d4')
ax3.spines['bottom'].set_color('#555')
ax3.spines['left'].set_color('#555')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.xaxis.label.set_color('#d4d4d4')
ax3.yaxis.label.set_color('#d4d4d4')
ax3.title.set_color('#e8e8e8')

fig.patch.set_facecolor('#111')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('demand_elasticity_revenue.png', dpi=150, facecolor='#111', bbox_inches='tight')
plt.show()
```

This produces three panels. The top panel shows the demand curve color-coded by elasticity regime — red for elastic (upper portion), blue for inelastic (lower portion), with the unit-elastic midpoint marked in gold. The middle panel shows how elasticity magnitude varies with price: it's near zero at low prices and shoots toward infinity at high prices. The bottom panel shows the revenue curve, which is bell-shaped and maximized exactly at the unit-elastic quantity.

### 9.2 The Lerner Index vs. Elasticity

```python
import numpy as np
import matplotlib.pyplot as plt

# Elasticity range (must be > 1 for monopolist)
eps = np.linspace(1.01, 20, 500)
L = 1.0 / eps

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(eps, L, color='#6db3f2', linewidth=2.5, label=r'$L = 1/|\epsilon|$')
ax.fill_between(eps, L, alpha=0.1, color='#6db3f2')

# Industry annotations
industries = [
    (1.3, 'Pharmaceuticals\n(patented drugs)', '#f2736d'),
    (1.8, 'Software / SaaS', '#cc66cc'),
    (2.5, 'Branded consumer\ngoods', '#f0c040'),
    (4.0, 'Airline tickets', '#66cc66'),
    (8.0, 'Retail groceries', '#d4d4d4'),
    (15.0, 'Commodity markets\n(wheat, oil)', '#999'),
]

for e_val, label, color in industries:
    l_val = 1.0 / e_val
    ax.plot(e_val, l_val, 'o', color=color, markersize=9, zorder=5)
    # Adjust text position based on elasticity value
    y_offset = 0.04 if e_val < 5 else 0.02
    ax.annotate(label, xy=(e_val, l_val), xytext=(e_val + 0.5, l_val + y_offset),
                fontsize=10, color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

ax.set_xlabel(r'Demand Elasticity $|\epsilon|$', fontsize=14)
ax.set_ylabel(r'Lerner Index $L = (P - MC)/P$', fontsize=14)
ax.set_title('Optimal Markup Decreases with Demand Elasticity', fontsize=15)
ax.legend(fontsize=12, loc='upper right')

# Styling
ax.set_facecolor('#1a1a1a')
ax.tick_params(colors='#d4d4d4')
ax.spines['bottom'].set_color('#555')
ax.spines['left'].set_color('#555')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.label.set_color('#d4d4d4')
ax.yaxis.label.set_color('#d4d4d4')
ax.title.set_color('#e8e8e8')
fig.patch.set_facecolor('#111')

ax.set_xlim(1, 20)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('lerner_index.png', dpi=150, facecolor='#111', bbox_inches='tight')
plt.show()
```

This plot shows the hyperbolic relationship \(L = 1/|\varepsilon|\). The key takeaway is visible at a glance: firms with inelastic demand (few substitutes) enjoy enormous markups, while firms facing elastic demand (many substitutes) are forced to price close to cost. The annotated industry examples ground the math in reality — pharmaceutical companies with patented drugs can mark up 60-70%, while commodity sellers eke out margins of 5-10%.

### 9.3 Monopoly Profit Optimization with Surplus Areas

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
a, b, c = 100, 2, 10  # Q = a - bP, MC = c

# Derived monopoly solution
Q_star = (a - b * c) / 2  # = 40
P_star = (a / b + c) / 2  # = 30
profit = (P_star - c) * Q_star  # = 800
Q_comp = a - b * c  # = 80, competitive quantity

# --- Plot 1: Profit as a function of Q ---
Q_range = np.linspace(0, a, 500)
P_of_Q = a / b - Q_range / b  # Inverse demand
Revenue = P_of_Q * Q_range
Cost = c * Q_range
Profit = Revenue - Cost

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax1 = axes[0]
ax1.plot(Q_range, Profit, color='#f0c040', linewidth=2.5, label=r'Profit $\pi(Q)$')
ax1.plot(Q_range, Revenue, color='#66cc66', linewidth=2, linestyle='--', alpha=0.7, label=r'Revenue $R(Q)$')
ax1.plot(Q_range, Cost, color='#f2736d', linewidth=2, linestyle='--', alpha=0.7, label=r'Cost $C(Q)$')
ax1.axvline(x=Q_star, color='#f0c040', linestyle=':', linewidth=1.5, alpha=0.7)
ax1.plot(Q_star, profit, 'o', color='#f0c040', markersize=10, zorder=5)
ax1.annotate(rf'$Q^* = {Q_star:.0f}$, $\pi^* = {profit:.0f}$',
             xy=(Q_star, profit), xytext=(Q_star + 8, profit + 50),
             fontsize=12, color='#f0c040',
             arrowprops=dict(arrowstyle='->', color='#f0c040'))

ax1.set_xlabel(r'Quantity $Q$', fontsize=13)
ax1.set_ylabel(r'Dollars ($)', fontsize=13)
ax1.set_title(r'Profit Maximization: $Q = 100 - 2P$, $MC = 10$', fontsize=14)
ax1.legend(fontsize=11, loc='upper right')
ax1.set_facecolor('#1a1a1a')
ax1.tick_params(colors='#d4d4d4')
ax1.spines['bottom'].set_color('#555')
ax1.spines['left'].set_color('#555')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.xaxis.label.set_color('#d4d4d4')
ax1.yaxis.label.set_color('#d4d4d4')
ax1.title.set_color('#e8e8e8')

# --- Plot 2: Demand, MR, MC with surplus shading ---
ax2 = axes[1]
Q_plot = np.linspace(0, a, 500)
P_demand = a / b - Q_plot / b  # Inverse demand
MR_curve = a / b - 2 * Q_plot / b  # MR

ax2.plot(Q_plot, P_demand, color='#6db3f2', linewidth=2.5, label='Demand (Inverse)')
ax2.plot(Q_plot[Q_plot <= a/2], MR_curve[Q_plot <= a/2], color='#cc66cc', linewidth=2, linestyle='--', label='MR')
ax2.axhline(y=c, color='#66cc66', linewidth=2, linestyle='-.', label=rf'MC = {c}')

# Shade consumer surplus
Q_cs = np.linspace(0, Q_star, 100)
P_cs = a / b - Q_cs / b
ax2.fill_between(Q_cs, P_star, P_cs, alpha=0.25, color='#6db3f2', label='CS')

# Shade producer surplus
ax2.fill_between(Q_cs, c, P_star, alpha=0.25, color='#66cc66', label='PS (Profit)')

# Shade deadweight loss
Q_dwl = np.linspace(Q_star, Q_comp, 100)
P_dwl = a / b - Q_dwl / b
ax2.fill_between(Q_dwl, c, P_dwl, alpha=0.3, color='#f2736d', label='DWL')

# Mark optimal point
ax2.plot(Q_star, P_star, 's', color='#f0c040', markersize=10, zorder=5)
ax2.annotate(rf'$(Q^*={Q_star:.0f},\; P^*={P_star:.0f})$',
             xy=(Q_star, P_star), xytext=(Q_star + 5, P_star + 4),
             fontsize=12, color='#f0c040',
             arrowprops=dict(arrowstyle='->', color='#f0c040'))

# Dashed lines from optimal point
ax2.plot([Q_star, Q_star], [0, P_star], ':', color='#999', linewidth=1)
ax2.plot([0, Q_star], [P_star, P_star], ':', color='#999', linewidth=1)

ax2.set_xlabel(r'Quantity $Q$', fontsize=13)
ax2.set_ylabel(r'Price / Cost ($)', fontsize=13)
ax2.set_title('Surplus Under Monopoly Pricing', fontsize=14)
ax2.legend(fontsize=10, loc='upper right')
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 55)
ax2.set_facecolor('#1a1a1a')
ax2.tick_params(colors='#d4d4d4')
ax2.spines['bottom'].set_color('#555')
ax2.spines['left'].set_color('#555')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.xaxis.label.set_color('#d4d4d4')
ax2.yaxis.label.set_color('#d4d4d4')
ax2.title.set_color('#e8e8e8')

fig.patch.set_facecolor('#111')
plt.tight_layout()
plt.savefig('monopoly_optimization.png', dpi=150, facecolor='#111', bbox_inches='tight')
plt.show()
```

The left panel shows profit as a function of quantity — it's a parabola that peaks at \(Q^* = 40\) with maximum profit of $800. Revenue and cost curves are overlaid to show how the gap between them (which is profit) first widens, then narrows. The right panel shows the classic diagram: inverse demand, marginal revenue, and marginal cost, with consumer surplus (blue), producer surplus (green), and deadweight loss (red) all shaded. The monopoly optimum sits where MR crosses MC, and the price is read off the demand curve above it.

---

## 10. From Theory to Practice

We've built the complete mathematical foundation for pricing theory in a single sitting. Let's recap the key results:

1. **The demand function** \(Q = f(P, I, P_s, P_c, \ldots)\) captures how quantity responds to all relevant economic variables.
2. **Price elasticity** \(\varepsilon = (dQ/dP)(P/Q)\) is the unit-free measure of price sensitivity, and it varies along a linear demand curve.
3. **The revenue-elasticity connection**: \(dR/dP = Q(1 + \varepsilon)\). Revenue rises with price when demand is inelastic, falls when elastic, and is maximized at unit elasticity.
4. **The MR = MC condition** gives the profit-maximizing output for any firm with market power.
5. **The Lerner Index** \(L = 1/|\varepsilon|\) is the ultimate pricing formula: optimal markup is inversely proportional to demand elasticity.

But this entire analysis assumes a single, known demand curve and a monopolist selling one product at one uniform price to a homogeneous market. Real firms face a much messier world, and the rest of this series tackles those complications one by one.

### What Comes Next

**Part 2: Price Discrimination — Extracting Surplus.** The Lerner formula gives one price for one market. But customers differ in their willingness to pay — the demand curve aggregates over a heterogeneous population. Price discrimination is the art of charging different prices to different customers (or for different units) to capture more of the consumer surplus triangle. We'll cover first-, second-, and third-degree discrimination, two-part tariffs, bundling, and versioning, all derived from the demand theory built here.

**Part 3: Game Theory of Pricing.** A monopolist faces no competitors. But most firms do. When you set a price, your competitors react, and you must anticipate their reactions. This takes us into game theory: Bertrand competition (price wars), Cournot competition (quantity setting), Stackelberg leadership, and Nash equilibria in pricing games. The Lerner Index generalizes to account for strategic interaction.

**Part 4: Causal Demand Estimation.** Everything above assumes you *know* the demand function. In reality, you must estimate it from data. This is harder than it sounds because of the **endogeneity problem**: prices aren't randomly assigned — firms set prices in response to demand, creating a simultaneous equation that naive regression can't untangle. We'll cover instrumental variables, natural experiments, difference-in-differences, and modern ML approaches to demand estimation.

**Part 5: Algorithmic Dynamic Pricing.** Even with a perfectly estimated demand function, the optimal price changes over time as demand shifts, inventories fluctuate, and competitors move. Dynamic pricing algorithms — from simple rule-based systems to multi-armed bandits and reinforcement learning — must balance exploitation (pricing optimally given current knowledge) with exploration (experimenting to learn the demand curve). We'll build a Thompson sampling dynamic pricer from scratch.

The journey from "how much should we charge?" to a production pricing algorithm passes through all of microeconomic theory. This first post gave you the mathematical bedrock. The next four will build the house.
