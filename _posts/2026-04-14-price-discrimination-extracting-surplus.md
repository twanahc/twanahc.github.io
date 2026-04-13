---
layout: post
title: "Price Discrimination: The Mathematics of Extracting Consumer Surplus"
date: 2026-04-14
category: business
---

*This is Part 2 of a 5-part series on pricing strategy. [Part 1: Demand, Elasticity & Markup](/2026/04/13/demand-elasticity-monopolist-markup.html) | **Part 2: Price Discrimination** | [Part 3: Game Theory of Pricing](/2026/04/15/game-theory-competitive-pricing.html) | [Part 4: Causal Demand Estimation](/2026/04/16/causal-demand-estimation-ml.html) | [Part 5: Algorithmic Dynamic Pricing](/2026/04/17/algorithmic-dynamic-pricing-bandits.html)*

In Part 1 we derived the monopolist's optimal uniform price using the Lerner index: set marginal revenue equal to marginal cost, mark up inversely proportional to demand elasticity, collect your profit, go home. But uniform pricing is a blunt instrument. It leaves money on the table in two distinct ways. First, there are customers who value the product far above the price you charge — they would have paid more, and the difference between what they would have paid and what they actually paid is surplus you handed them for free. Second, there are customers who value the product above your marginal cost but below your uniform price — transactions that would be profitable never happen. That gap is deadweight loss, value destroyed by the pricing mechanism itself.

Price discrimination is the systematic attempt to close both of those gaps. The idea is simple in principle: charge different prices to different customers, or for different units, so that you capture more of the total surplus. In practice, it ranges from the perfectly personalized (every customer pays their exact willingness to pay) to the subtly designed (a menu of options that makes customers reveal their own price sensitivity through their choices). Airlines, SaaS companies, movie theaters, academic publishers, and theme parks all do it — they just use different mechanisms. What unifies them is the mathematics.

This post works through the full theory. We will start with why uniform pricing is suboptimal, then cover the three classical degrees of price discrimination (in a pedagogically useful order), then move to the practical mechanisms — two-part tariffs, bundling, versioning, and behavioral pricing. Each concept is defined before it is used, each result is derived from first principles, and we end with Python simulations so you can see the numbers move.

---

## Table of Contents

1. [Why Uniform Pricing Is Suboptimal](#why-uniform-pricing-is-suboptimal)
2. [First-Degree Price Discrimination](#first-degree-price-discrimination)
3. [Third-Degree Price Discrimination](#third-degree-price-discrimination)
4. [Second-Degree Price Discrimination and Mechanism Design](#second-degree-price-discrimination-and-mechanism-design)
5. [Two-Part Tariffs](#two-part-tariffs)
6. [Bundling Theory](#bundling-theory)
7. [Versioning and Damaged Goods](#versioning-and-damaged-goods)
8. [Behavioral Pricing Psychology](#behavioral-pricing-psychology)
9. [Python Simulations](#python-simulations)
10. [The Practical Landscape](#the-practical-landscape)

---

## Why Uniform Pricing Is Suboptimal

Recall from Part 1 the monopolist's problem. The firm faces a downward-sloping demand curve \\(Q(P)\\) and has marginal cost \\(MC\\). The optimal uniform price \\(P^*\\) satisfies \\(MR = MC\\), which by the Lerner index gives:

$$\frac{P^* - MC}{P^*} = -\frac{1}{\varepsilon(P^*)}$$

where \\(\varepsilon\\) is the price elasticity of demand. At this price, the firm sells \\(Q^* = Q(P^*)\\) units and earns profit \\(\pi = (P^* - MC) \cdot Q^*\\).

But look at what is left on the table. The demand curve tells us the willingness to pay (WTP) of every potential customer. Customers with WTP above \\(P^*\\) buy the product, but each of them pays only \\(P^*\\), not their full WTP. The gap — the area between the demand curve and the price line, from \\(Q = 0\\) to \\(Q = Q^*\\) — is **consumer surplus** (CS):

$$CS = \int_0^{Q^*} \big[P(Q) - P^*\big] \, dQ$$

where \\(P(Q)\\) is the inverse demand function (the demand curve written as price as a function of quantity). This is money consumers would have been willing to spend but didn't have to.

Meanwhile, there are customers with WTP between \\(MC\\) and \\(P^*\\). Each of these transactions would generate positive surplus (the buyer values the good above cost), but they never happen because the price is too high. The lost surplus from these missing transactions is the **deadweight loss** (DWL):

$$DWL = \int_{Q^*}^{Q_c} \big[P(Q) - MC\big] \, dQ$$

where \\(Q_c\\) is the competitive quantity (where \\(P(Q_c) = MC\\)).

The monopolist's producer surplus (PS) is the rectangle \\((P^* - MC) \cdot Q^*\\). Total surplus is \\(CS + PS + DWL\\) in a competitive market (where DWL = 0), and the monopolist captures only the PS rectangle.

<svg viewBox="0 0 500 340" xmlns="http://www.w3.org/2000/svg" style="max-width:520px; display:block; margin:auto;">
  <!-- Axes -->
  <line x1="60" y1="20" x2="60" y2="290" stroke="#d4d4d4" stroke-width="1.5"/>
  <line x1="60" y1="290" x2="470" y2="290" stroke="#d4d4d4" stroke-width="1.5"/>
  <text x="25" y="160" fill="#d4d4d4" font-size="13" transform="rotate(-90,25,160)">Price</text>
  <text x="260" y="320" fill="#d4d4d4" font-size="13" text-anchor="middle">Quantity</text>

  <!-- Demand curve: from (60,40) to (440,290) -->
  <line x1="60" y1="40" x2="440" y2="290" stroke="#e74c3c" stroke-width="2.5"/>
  <text x="445" y="285" fill="#e74c3c" font-size="12">D</text>

  <!-- MC line at y=220 -->
  <line x1="60" y1="220" x2="440" y2="220" stroke="#3498db" stroke-width="2" stroke-dasharray="6,3"/>
  <text x="445" y="224" fill="#3498db" font-size="12">MC</text>

  <!-- MR curve: steeper, from (60,40) to (250,290) -->
  <line x1="60" y1="40" x2="250" y2="290" stroke="#9b59b6" stroke-width="2" stroke-dasharray="4,2"/>
  <text x="255" y="288" fill="#9b59b6" font-size="12">MR</text>

  <!-- P* line at y=130, Q* at x=195 -->
  <line x1="60" y1="130" x2="195" y2="130" stroke="#d4d4d4" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="195" y1="130" x2="195" y2="290" stroke="#d4d4d4" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="35" y="134" fill="#d4d4d4" font-size="12">P*</text>
  <text x="190" y="305" fill="#d4d4d4" font-size="12">Q*</text>

  <!-- Qc at x=316 (where demand meets MC) -->
  <line x1="316" y1="220" x2="316" y2="290" stroke="#d4d4d4" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="308" y="305" fill="#d4d4d4" font-size="12">Q_c</text>

  <!-- Consumer Surplus: triangle above P*, below demand, left of Q* -->
  <polygon points="60,40 60,130 195,130" fill="#2ecc71" opacity="0.35"/>
  <text x="90" y="100" fill="#2ecc71" font-size="12" font-weight="bold">CS</text>

  <!-- Producer Surplus: rectangle P* to MC, from 0 to Q* -->
  <polygon points="60,130 195,130 195,220 60,220" fill="#e67e22" opacity="0.35"/>
  <text x="110" y="182" fill="#e67e22" font-size="12" font-weight="bold">PS</text>

  <!-- Deadweight Loss: triangle between Q* and Qc -->
  <polygon points="195,130 316,220 195,220" fill="#e74c3c" opacity="0.3"/>
  <text x="220" y="200" fill="#e74c3c" font-size="12" font-weight="bold">DWL</text>

  <!-- Intersection dots -->
  <circle cx="195" cy="130" r="4" fill="#d4d4d4"/>
  <circle cx="195" cy="220" r="4" fill="#d4d4d4"/>
  <circle cx="316" cy="220" r="4" fill="#d4d4d4"/>
</svg>

The monopolist's dream is to convert that green CS triangle and red DWL triangle into producer surplus. Price discrimination is the collection of techniques for doing exactly that.

Three conditions must hold for price discrimination to be feasible:

1. **Market power.** The firm must be able to set price above marginal cost. A perfectly competitive firm is a price-taker and cannot discriminate.
2. **Ability to segment or screen.** The firm must either be able to identify which customers have high WTP (direct segmentation) or design a mechanism that makes customers reveal their type through their choices (screening).
3. **Arbitrage prevention.** Customers who get the low price must not be able to resell to customers who face the high price. This is why price discrimination works well for services (haircuts, airline seats) and poorly for commodity goods (wheat, oil).

---

## First-Degree Price Discrimination

**First-degree price discrimination** — also called **perfect price discrimination** — is the theoretical ideal. The firm charges each customer exactly their willingness to pay.

Formally, suppose customer \\(i\\) has willingness to pay \\(v_i\\) and the firm's marginal cost is \\(c\\). Under first-degree discrimination, the firm charges:

$$P_i = v_i \quad \text{for all } i \text{ with } v_i \geq c$$

Every customer whose WTP exceeds cost is served. The firm captures the entire surplus from every transaction. Consumer surplus is zero. Deadweight loss is also zero — every efficient transaction occurs.

If the distribution of WTP across the population is described by a density function \\(f(v)\\), then total profit under first-degree discrimination is:

$$\pi_{\text{1st}} = \int_c^{v_{\max}} (v - c) \, f(v) \, dv$$

This is equal to the total surplus that a competitive market would generate — but now the firm captures all of it instead of splitting it with consumers. The quantity sold equals the competitive quantity \\(Q_c\\). It is allocatively efficient (no DWL) but distributionally extreme (zero CS).

<svg viewBox="0 0 500 340" xmlns="http://www.w3.org/2000/svg" style="max-width:520px; display:block; margin:auto;">
  <!-- Axes -->
  <line x1="60" y1="20" x2="60" y2="290" stroke="#d4d4d4" stroke-width="1.5"/>
  <line x1="60" y1="290" x2="470" y2="290" stroke="#d4d4d4" stroke-width="1.5"/>
  <text x="25" y="160" fill="#d4d4d4" font-size="13" transform="rotate(-90,25,160)">Price</text>
  <text x="260" y="320" fill="#d4d4d4" font-size="13" text-anchor="middle">Quantity</text>

  <!-- Demand curve -->
  <line x1="60" y1="40" x2="440" y2="290" stroke="#e74c3c" stroke-width="2.5"/>
  <text x="445" y="285" fill="#e74c3c" font-size="12">D</text>

  <!-- MC line -->
  <line x1="60" y1="220" x2="440" y2="220" stroke="#3498db" stroke-width="2" stroke-dasharray="6,3"/>
  <text x="445" y="224" fill="#3498db" font-size="12">MC</text>

  <!-- Entire area between demand and MC is producer surplus -->
  <polygon points="60,40 316,220 60,220" fill="#e67e22" opacity="0.45"/>
  <text x="120" y="170" fill="#e67e22" font-size="14" font-weight="bold">All Producer</text>
  <text x="120" y="188" fill="#e67e22" font-size="14" font-weight="bold">Surplus</text>

  <!-- Qc mark -->
  <line x1="316" y1="220" x2="316" y2="290" stroke="#d4d4d4" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="308" y="305" fill="#d4d4d4" font-size="12">Q_c</text>

  <!-- Labels -->
  <text x="100" y="80" fill="#2ecc71" font-size="12">CS = 0</text>
  <text x="330" y="260" fill="#e74c3c" font-size="12">DWL = 0</text>

  <circle cx="316" cy="220" r="4" fill="#d4d4d4"/>
</svg>

Why is first-degree discrimination nearly impossible in practice? Because it requires the firm to know every individual customer's exact WTP — and to prevent arbitrage. If Alice's WTP is $80 and Bob's is $40, you need to charge Alice $80 and Bob $40, and you need to prevent Bob from buying two units and reselling one to Alice.

Approximations to first-degree discrimination exist in the wild:

- **Auctions** (eBay, Google Ads): each buyer reveals their WTP through bidding.
- **Individual negotiation** (car dealerships, B2B enterprise sales): the salesperson probes to learn the buyer's reservation price.
- **Personalized online pricing** (controversial): using browsing history, location, and behavioral data to show different prices to different users. Regulators are increasingly skeptical of this.

The gap between first-degree (theoretical ideal) and uniform pricing (practical baseline) motivates the more realistic discrimination strategies that follow.

---

## Third-Degree Price Discrimination

We cover third-degree before second-degree because it is conceptually simpler and sets up the screening problem that makes second-degree interesting.

**Third-degree price discrimination** means dividing customers into identifiable groups with different demand elasticities, then charging each group a different price. The key word is *identifiable* — the firm can directly observe which segment a customer belongs to.

Examples are everywhere: student discounts (show your ID), senior discounts, geographic pricing (different prices in different countries), weekday vs. weekend pricing at movie theaters, business vs. leisure airfares (Saturday night stay requirement separates them).

### The Mathematics

Suppose the firm sells to two segments. Segment 1 has demand \\(Q_1(P_1)\\) and segment 2 has demand \\(Q_2(P_2)\\). The firm's total cost is \\(C(Q_1 + Q_2)\\) with constant marginal cost \\(MC = c\\). The firm's problem is:

$$\max_{P_1, P_2} \; \pi = P_1 \cdot Q_1(P_1) + P_2 \cdot Q_2(P_2) - c \cdot (Q_1 + Q_2)$$

Taking the first-order condition with respect to \\(P_1\\) (and similarly for \\(P_2\\)):

$$\frac{\partial \pi}{\partial P_1} = Q_1 + P_1 \frac{dQ_1}{dP_1} - c \frac{dQ_1}{dP_1} = 0$$

This can be rewritten as:

$$Q_1 + (P_1 - c) \frac{dQ_1}{dP_1} = 0$$

Dividing through by \\(Q_1\\) and using the definition of price elasticity \\(\varepsilon_1 = \frac{dQ_1}{dP_1} \cdot \frac{P_1}{Q_1}\\):

$$1 + (P_1 - c) \cdot \frac{\varepsilon_1}{P_1} = 0$$

which gives us the Lerner index for segment 1:

$$\frac{P_1 - c}{P_1} = -\frac{1}{\varepsilon_1}$$

The same condition holds for segment 2. Since both segments face the same marginal cost, the key result is:

$$MR_1 = MR_2 = MC$$

Using \\(MR_i = P_i(1 + 1/\varepsilon_i)\\), we can write:

$$P_1 \left(1 + \frac{1}{\varepsilon_1}\right) = P_2 \left(1 + \frac{1}{\varepsilon_2}\right)$$

Rearranging to get the price ratio:

$$\frac{P_1}{P_2} = \frac{1 + 1/\varepsilon_2}{1 + 1/\varepsilon_1}$$

Since demand elasticities are negative (\\(\varepsilon < 0\\)), this means: **the segment with more inelastic demand (lower \\(|\varepsilon|\\)) gets charged a higher price.** Business travelers have inelastic demand for flights (they must travel on specific dates); leisure travelers are elastic (they can shift dates or choose not to go). So business travelers pay more. Students are price-sensitive; working professionals less so. So students get discounts.

### A Numerical Example

Let us work through a concrete problem. Two segments:

- **Segment A** (business travelers): \\(Q_A = 100 - P_A\\)
- **Segment B** (leisure travelers): \\(Q_B = 200 - 4P_B\\)

Marginal cost \\(MC = 10\\).

**With discrimination**, we solve \\(MR_A = MC\\) and \\(MR_B = MC\\) separately.

For segment A, total revenue is \\(TR_A = P_A \cdot Q_A = P_A(100 - P_A) = 100P_A - P_A^2\\). Marginal revenue in terms of quantity: invert to get \\(P_A = 100 - Q_A\\), so \\(TR_A = 100Q_A - Q_A^2\\) and \\(MR_A = 100 - 2Q_A\\). Setting \\(MR_A = 10\\):

$$100 - 2Q_A = 10 \implies Q_A = 45, \quad P_A = 100 - 45 = 55$$

For segment B, \\(P_B = 50 - Q_B/4\\), so \\(TR_B = 50Q_B - Q_B^2/4\\) and \\(MR_B = 50 - Q_B/2\\). Setting \\(MR_B = 10\\):

$$50 - Q_B/2 = 10 \implies Q_B = 80, \quad P_B = 50 - 80/4 = 30$$

Discriminating profit:

$$\pi_{\text{disc}} = (55 - 10)(45) + (30 - 10)(80) = 2025 + 1600 = 3625$$

**Without discrimination** (uniform price), we must aggregate the demands. Total demand at price \\(P\\) is:

$$Q(P) = Q_A(P) + Q_B(P) = (100 - P) + (200 - 4P) = 300 - 5P$$

Note: this is valid only for \\(P \leq 50\\) (both segments active). The inverse demand is \\(P = 60 - Q/5\\), total revenue \\(TR = 60Q - Q^2/5\\), and \\(MR = 60 - 2Q/5\\). Setting \\(MR = 10\\):

$$60 - 2Q/5 = 10 \implies Q = 125, \quad P = 60 - 125/5 = 35$$

Uniform profit:

$$\pi_{\text{uniform}} = (35 - 10)(125) = 3125$$

The profit gain from discrimination is \\(3625 - 3125 = 500\\), a 16% increase.

### Welfare Analysis

Does third-degree discrimination improve social welfare? The answer is ambiguous — it depends on whether discrimination opens new markets.

If both segments would be served under uniform pricing, discrimination reshuffles surplus without necessarily increasing total quantity. It can decrease total welfare (the inelastic segment is charged more, the elastic segment is charged less, but the distortion in the inelastic segment may outweigh the gain in the elastic segment).

However, if discrimination causes a segment to be served that would not have been served at the uniform price (the uniform price exceeds the maximum WTP of that segment), then total quantity rises and welfare can increase. This "new market" effect is the strongest argument in favor of third-degree discrimination.

A necessary condition for welfare to increase is that total output rises under discrimination compared to uniform pricing.

---

## Second-Degree Price Discrimination and Mechanism Design

This is the hardest and most intellectually rewarding case. The firm cannot directly observe customer types — it cannot check whether you are a business traveler or a leisure traveler, a high-WTP or low-WTP buyer. Instead, it designs a **menu of options** and lets customers **self-select**. The customers' own choices reveal their type.

This is the domain of **mechanism design**, also called **screening theory**. The firm's problem is not just to set a price but to design a mechanism (a menu of contracts) that extracts maximum surplus given information asymmetry.

### The Setup

Suppose there are two customer types:

- **Type H** (high WTP): values quality \\(q\\) at \\(v_H \cdot q\\). There are \\(\lambda N\\) of them, where \\(\lambda\\) is the proportion of high types and \\(N\\) is the total market.
- **Type L** (low WTP): values quality \\(q\\) at \\(v_L \cdot q\\), with \\(v_L < v_H\\). There are \\((1 - \lambda) N\\) of them.

The cost of providing quality \\(q\\) is \\(c(q) = q^2 / 2\\) (increasing and convex — higher quality costs disproportionately more).

If the firm could observe types directly, it would offer each type their **first-best** contract. For type \\(i\\), the first-best quality \\(q_i^{FB}\\) maximizes the per-customer surplus \\(v_i q - q^2/2\\):

$$\frac{d}{dq}(v_i q - q^2/2) = v_i - q = 0 \implies q_i^{FB} = v_i$$

The firm would set quality \\(q_H^{FB} = v_H\\), \\(q_L^{FB} = v_L\\), and charge each type their full surplus: \\(P_H = v_H \cdot q_H^{FB} = v_H^2\\), \\(P_L = v_L \cdot q_L^{FB} = v_L^2\\).

But the firm cannot observe types. If it offers these two contracts, type H would look at the type L contract \\((q_L^{FB}, P_L)\\) and compute their surplus from taking it: \\(v_H \cdot q_L^{FB} - P_L = v_H v_L - v_L^2 = v_L(v_H - v_L) > 0\\). Compared to the type H contract, which gives them zero surplus (\\(v_H^2 - v_H^2 = 0\\)). So type H would **mimic** type L — pretend to be a low-WTP customer to get a better deal.

### The Constraints

The firm offers a menu \\(\{(q_H, P_H), (q_L, P_L)\}\\) and must satisfy four constraints:

**Individual Rationality (IR):** Each type must prefer buying to not buying (getting zero surplus).

$$\text{IR-H:} \quad v_H q_H - P_H \geq 0$$

$$\text{IR-L:} \quad v_L q_L - P_L \geq 0$$

**Incentive Compatibility (IC):** Each type must prefer their intended contract to the other type's contract.

$$\text{IC-H:} \quad v_H q_H - P_H \geq v_H q_L - P_L$$

$$\text{IC-L:} \quad v_L q_L - P_L \geq v_L q_H - P_H$$

### Which Constraints Bind?

This is where the elegance lies. We can show that in the optimal solution, **IC-H** and **IR-L** are the binding (tight) constraints, while IC-L and IR-H are slack.

**Why does IR-L bind?** Type L has the lowest WTP. If we could charge them more, we would. We charge them exactly enough to make them indifferent between buying and not buying: \\(v_L q_L - P_L = 0\\), so \\(P_L = v_L q_L\\).

**Why does IC-H bind?** Type H is the "dangerous" type — the one tempted to mimic type L. We need type H to be exactly indifferent between their contract and type L's contract. If IC-H were slack (type H strictly preferred their contract), we could raise \\(P_H\\) and extract more surplus.

From the binding IC-H: \\(v_H q_H - P_H = v_H q_L - P_L\\), which gives:

$$P_H = v_H q_H - v_H q_L + P_L = v_H q_H - v_H q_L + v_L q_L$$

$$P_H = v_H q_H - (v_H - v_L) q_L$$

The term \\((v_H - v_L) q_L\\) is the **information rent** — the surplus type H captures because the firm cannot verify their type. It is positive and increasing in \\(q_L\\).

### The Key Distortion

The firm's profit is:

$$\pi = \lambda \big[P_H - c(q_H)\big] + (1-\lambda)\big[P_L - c(q_L)\big]$$

Substituting our expressions for \\(P_H\\) and \\(P_L\\):

$$\pi = \lambda \big[v_H q_H - (v_H - v_L)q_L - q_H^2/2\big] + (1-\lambda)\big[v_L q_L - q_L^2/2\big]$$

Taking the FOC with respect to \\(q_H\\):

$$\frac{\partial \pi}{\partial q_H} = \lambda(v_H - q_H) = 0 \implies q_H = v_H = q_H^{FB}$$

Type H gets the **efficient quality** — no distortion at the top. This is a general result in screening theory.

Taking the FOC with respect to \\(q_L\\):

$$\frac{\partial \pi}{\partial q_L} = -\lambda(v_H - v_L) + (1-\lambda)(v_L - q_L) = 0$$

$$q_L = v_L - \frac{\lambda}{1-\lambda}(v_H - v_L)$$

This is **below the first-best** \\(q_L^{FB} = v_L\\). The firm deliberately degrades type L's quality. Why? Because every unit of quality in the low-end product increases the information rent type H captures. By making the cheap option worse, the firm makes it less tempting for type H to mimic type L, which allows the firm to charge type H closer to their true WTP.

This is the mathematical foundation of a pattern we see everywhere:

- **Airlines**: Economy class is deliberately uncomfortable (tight seats, no legroom, restricted bags) not primarily because it is cheaper to produce, but because comfortable economy would cannibalize business class.
- **Software tiers**: The free tier is restricted not just to save costs, but to make the paid tier more attractive to high-WTP users.
- **Textbook publishing**: The hardcover comes first at a high price; the paperback comes later at a low price. The delay screens the impatient (high-WTP) from the patient (low-WTP).
- **Intel 486SX**: The infamous case where Intel took its 486DX processor, **disabled** the math coprocessor at additional manufacturing cost, and sold it as the cheaper 486SX. The cheap version literally cost more to make because of the active degradation step.

---

## Two-Part Tariffs

A **two-part tariff** charges customers a fixed fee \\(F\\) (for access or membership) plus a per-unit price \\(p\\) (for each unit consumed). The total payment for \\(Q\\) units is:

$$T(Q) = F + p \cdot Q$$

This is a powerful pricing mechanism because it gives the firm two instruments instead of one. The per-unit price \\(p\\) controls the quantity consumed, and the fixed fee \\(F\\) extracts surplus.

### Homogeneous Consumers

If all consumers are identical with demand curve \\(Q(p)\\), the optimal two-part tariff is elegant. Set the per-unit price equal to marginal cost:

$$p^* = MC$$

At this price, each consumer buys the competitive quantity and generates consumer surplus:

$$CS(MC) = \int_0^{Q(MC)} \big[P(Q) - MC\big] \, dQ$$

Now set the fixed fee equal to the entire consumer surplus:

$$F^* = CS(MC)$$

The result is equivalent to first-degree price discrimination: every efficient transaction occurs (\\(p = MC\\) means no DWL), and the firm captures all surplus through the fixed fee. The consumer is left with exactly zero surplus — indifferent between participating and not.

### Heterogeneous Consumers

With heterogeneous consumers, the problem becomes a tradeoff. Suppose there are two types with different demand curves: type H (heavy user) and type L (light user), with \\(CS_H(p) > CS_L(p)\\) for all \\(p\\).

If the firm sets \\(F = CS_H(MC)\\), type L drops out (the fixed fee exceeds their surplus). If it sets \\(F = CS_L(MC)\\), both types participate, but type H retains surplus \\(CS_H(MC) - CS_L(MC)\\).

The optimal two-part tariff balances these forces. Let \\(N_H\\) and \\(N_L\\) be the numbers of each type. The firm solves:

$$\max_{p, F} \; \pi = (N_H + N_L)[F + (p - MC) \cdot Q_L(p)] \quad \text{if } F \leq CS_L(p)$$

or, if it decides to serve only type H:

$$\pi = N_H[CS_H(p) + (p - MC) \cdot Q_H(p)]$$

The general result is that the optimal per-unit price exceeds marginal cost (\\(p^* > MC\\)) — the firm sacrifices some per-unit efficiency to raise the consumer surplus of the heavy user, enabling a higher fixed fee from both types.

Real-world examples of two-part tariffs:

- **Costco**: annual membership fee ($65) + near-wholesale prices on products
- **Amusement parks**: entry fee + per-ride tickets (or all-inclusive entry = pure fixed fee)
- **Cell phone plans**: monthly fee + per-minute/per-GB charges
- **Country clubs**: membership fee + greens fees per round

---

## Bundling Theory

**Bundling** means selling multiple products together as a package. **Pure bundling** offers only the bundle, not individual components. **Mixed bundling** offers both the bundle and individual products, with the bundle priced below the sum of individual prices.

The core question: when does bundling increase profit compared to selling products separately?

### The Key Insight

The answer, established by Adams & Yellen (1976) and formalized by Schmalensee (1984), is: **bundling works when customers have negatively correlated valuations across goods.**

Here is the intuition. Suppose you sell two goods, A and B:

| Customer | WTP for A | WTP for B | WTP for Bundle |
|----------|-----------|-----------|----------------|
| Alice    | $200      | $50       | $250           |
| Bob      | $50       | $200      | $250           |

**Selling separately at $200 each:** Alice buys only A, Bob buys only B. Revenue = $400.

**Selling separately at $50 each:** Alice buys both, Bob buys both. Revenue = $200. (Worse.)

**Bundling at $250:** Both buy the bundle. Revenue = $500.

The bundle captures more surplus because it reduces the heterogeneity in total WTP. When valuations are negatively correlated, high WTP for one good tends to coincide with low WTP for the other, so the sum is more homogeneous than the components. A more homogeneous distribution of WTP is easier to price against (you lose less to either consumer surplus or foregone sales).

### The Formal Condition

Let \\(V_A\\) and \\(V_B\\) be random variables representing a customer's WTP for goods A and B. The WTP for the bundle is \\(V_A + V_B\\). By the variance addition rule:

$$\text{Var}(V_A + V_B) = \text{Var}(V_A) + \text{Var}(V_B) + 2\text{Cov}(V_A, V_B)$$

When \\(\text{Cov}(V_A, V_B) < 0\\) (negative correlation), the variance of the bundle valuation is less than the sum of individual variances. Lower variance means a more concentrated distribution of WTP, which means less money left on the table by any single price point.

In the limit, if valuations are perfectly negatively correlated, the bundle WTP has zero variance — every customer values the bundle at exactly the same amount, and the firm can extract 100% of surplus with a single bundle price. This is the bundling equivalent of first-degree discrimination.

### Real-World Bundling

- **Microsoft Office**: Word, Excel, PowerPoint sold as a bundle. Customers who mainly need spreadsheets have high WTP for Excel, low for Word; writers have the opposite. The bundle captures both.
- **Cable TV**: You get 200 channels even though you watch 15. Each subscriber values a different subset, but the bundle price captures the average.
- **Amazon Prime**: Shipping + Video + Music + Reading. You might join for the shipping and consider the video a bonus; another customer joins for the video and appreciates the shipping. Negative correlation across use cases makes the bundle robust.

---

## Versioning and Damaged Goods

**Versioning** is the practice of offering multiple versions of a product at different price-quality points. It is the practical implementation of second-degree price discrimination — the versions are the menu, and customers self-select.

The deep insight is that the quality differentiation need not reflect cost differentiation. In fact, sometimes **it costs more to make the cheap version**.

This is the paradox of **damaged goods**. The firm starts with a high-quality product and deliberately removes features or degrades performance to create a low-quality version. The degradation itself may require engineering effort (and therefore cost). But the firm does it anyway because the screening value — preventing high-WTP customers from buying the cheap version — exceeds the cost of degradation.

Classic examples:

- **Intel 486SX/DX**: The 486SX was a 486DX with the math coprocessor physically disabled. Manufacturing cost was identical or higher (the disabling required an extra step). The price was lower. Intel was paying to make a worse product.
- **Software editions** (Professional vs. Home, Enterprise vs. Standard): The code base is often the same; the cheap version has features turned off.
- **Airlines**: The physical cost difference between economy and business seats is much smaller than the price difference. Much of economy's discomfort is a feature, not a bug — it screens out price-insensitive travelers.
- **Academic journals**: Institutional subscriptions cost 10-50x individual subscriptions. The product is identical.
- **Printers**: Some manufacturers sell "home" printers that are artificially slowed down compared to "office" models using the same hardware.

The mathematics is exactly the second-degree discrimination model from above. The firm distorts \\(q_L\\) downward (degrades the cheap version) to reduce the information rent captured by type H. The optimal amount of degradation balances the lost surplus from type L (who gets a worse product) against the increased surplus extraction from type H (who now finds it less attractive to mimic type L).

Denhardt's Rule (informal): if you see a product line where the cheap version seems *unnecessarily* bad, you are looking at a screening mechanism, not a cost-saving measure.

---

## Behavioral Pricing Psychology

Everything above assumes rational consumers who maximize utility given prices and budget constraints. Real humans are more interesting. Behavioral economics, rooted in the work of Kahneman and Tversky, reveals systematic departures from rationality that pricing strategists exploit — or at least must account for.

### Reference Pricing and Prospect Theory

People do not evaluate prices in absolute terms. They evaluate them relative to a **reference point** — what they expect to pay, what they paid last time, or what they see as the "normal" price. **Prospect theory** (Kahneman & Tversky, 1979) says that losses (paying more than the reference) hurt roughly twice as much as equivalent gains (paying less than the reference) feel good.

This has concrete pricing implications. A price increase from $10 to $12 feels like a $2 loss, which stings more than a $2 savings from a price drop feels good. This **loss aversion** means firms should avoid salient price increases and instead use shrinkflation (reduce quantity), rebundling, or fee restructuring to obscure the loss.

### Anchoring

The **anchoring effect** means that the first number a person encounters dominates their subsequent judgments, even when it is irrelevant. In pricing:

- Restaurants put the $65 steak at the top of the menu. The $28 pasta suddenly looks reasonable by comparison.
- Real estate agents show an overpriced house first, making the target property look like a bargain.
- SaaS pricing pages show the "Enterprise" tier first (at $499/month) so that the $99/month "Professional" tier feels affordable.

The anchor need not be a price anyone actually pays. Its function is to shift the reference point.

### The Decoy Effect (Asymmetric Dominance)

Adding a product that is **dominated** by one option but not the other can shift preferences toward the dominating option. This is the **decoy effect**, also called **asymmetric dominance**.

The famous example is The Economist's subscription pricing (circa 2009):

- Web-only: $59
- Print-only: $125
- Web + Print: $125

The print-only option at $125 is dominated by web + print at $125 (you get strictly more for the same price). No rational person should choose print-only. But its presence shifts preferences from web-only toward web + print. In Dan Ariely's experiments, removing the print-only option caused the majority of subjects to switch from web + print to web-only, drastically reducing revenue.

The decoy works by making the target option (web + print) look like an obvious bargain. Without the decoy, people compare web-only ($59) to web + print ($125) and focus on the $66 price difference. With the decoy, they compare print-only ($125) to web + print ($125) and focus on the free web access.

### Charm Pricing

Why does virtually every retail price end in .99? The **left-digit effect**: consumers process the leftmost digit first and anchor on it. $9.99 is perceived as "about $9," while $10.00 is "about $10." Research by Anderson and Simester (2003) demonstrated that changing a price from $34 to $39 actually *increased* demand for a mail-order catalog item — the left digit stayed at 3.

This is not rational, but it is real. The demand function itself is shaped by psychological factors, and any pricing model that ignores them is incomplete.

### Pay-What-You-Want

An extreme form of voluntary price discrimination: let the customer choose their own price, including zero. The puzzle is that people routinely pay positive amounts even when they could pay nothing.

Radiohead's *In Rainbows* (2007) was offered as pay-what-you-want and generated substantial revenue. Humble Bundle lets buyers choose their price for game bundles. Restaurants have experimented with pay-what-you-want with mixed results.

The mechanism relies on fairness norms, guilt, social signaling, and identity. People pay because paying nothing would conflict with their self-image as fair-minded individuals. The model works best when: the marginal cost is near zero (digital goods), social visibility is high, and the buyer has an ongoing relationship with the seller.

These behavioral findings do not replace the elasticity-based framework from Part 1. They modify it. The demand function \\(Q(P)\\) that the economist writes down is itself the result of psychological processes — anchoring, reference dependence, framing. A complete pricing theory must sit at the intersection.

---

## Python Simulations

### Simulation 1: Third-Degree Discrimination Surplus Comparison

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Parameters
MC = 10

# Segment A (business): Q_A = 100 - P_A  =>  P_A = 100 - Q_A
# Segment B (leisure):  Q_B = 200 - 4*P_B  =>  P_B = 50 - Q_B/4

# --- Discrimination ---
# Segment A: MR_A = 100 - 2Q_A = MC => Q_A=45, P_A=55
Q_A_disc, P_A_disc = 45, 55
# Segment B: MR_B = 50 - Q_B/2 = MC => Q_B=80, P_B=30
Q_B_disc, P_B_disc = 80, 30

# --- Uniform pricing ---
# Combined demand: Q = 300 - 5P => P = 60 - Q/5
# MR = 60 - 2Q/5 = MC => Q=125, P=35
Q_uni, P_uni = 125, 35

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Helper: shade surplus regions for a linear demand segment
def plot_segment(ax, intercept_p, slope_inv, Q_opt, P_opt, mc, title, color_demand):
    """
    Demand: P = intercept_p + slope_inv * Q  (slope_inv < 0)
    """
    Q_max = (intercept_p - mc) / abs(slope_inv)  # competitive quantity
    q_arr = np.linspace(0, Q_max * 1.15, 300)
    p_arr = intercept_p + slope_inv * q_arr

    # Demand curve
    ax.plot(q_arr, np.maximum(p_arr, mc - 5), color=color_demand, lw=2.5,
            label=r'Demand $P(Q)$')
    # MC line
    ax.axhline(mc, color='#3498db', ls='--', lw=1.5, label=r'$MC$')

    # Consumer surplus
    q_cs = np.linspace(0, Q_opt, 200)
    p_cs = intercept_p + slope_inv * q_cs
    ax.fill_between(q_cs, P_opt, p_cs, alpha=0.35, color='#2ecc71', label='CS')

    # Producer surplus
    ax.fill_between([0, Q_opt], [mc, mc], [P_opt, P_opt], alpha=0.35,
                    color='#e67e22', label='PS')

    # Deadweight loss
    q_dwl = np.linspace(Q_opt, Q_max, 200)
    p_dwl = intercept_p + slope_inv * q_dwl
    ax.fill_between(q_dwl, mc, p_dwl, alpha=0.3, color='#e74c3c', label='DWL')

    ax.set_xlim(0, Q_max * 1.2)
    ax.set_ylim(0, intercept_p * 1.1)
    ax.set_xlabel(r'Quantity $Q$', fontsize=12)
    ax.set_ylabel(r'Price $P$', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')

    # Annotate price and quantity
    ax.plot(Q_opt, P_opt, 'ko', ms=6)
    ax.annotate(rf'$P^*={P_opt}$, $Q^*={Q_opt}$',
                xy=(Q_opt, P_opt), xytext=(Q_opt + 5, P_opt + 8),
                fontsize=10, color='#d4d4d4',
                arrowprops=dict(arrowstyle='->', color='#d4d4d4'))

# Segment A with discrimination
plot_segment(axes[0], 100, -1, Q_A_disc, P_A_disc, MC,
             'Segment A (Business)\nDiscrimination', '#e74c3c')

# Segment B with discrimination
plot_segment(axes[1], 50, -0.25, Q_B_disc, P_B_disc, MC,
             'Segment B (Leisure)\nDiscrimination', '#9b59b6')

# Uniform pricing (combined demand: P = 60 - Q/5)
plot_segment(axes[2], 60, -0.2, Q_uni, P_uni, MC,
             'Combined Market\nUniform Pricing', '#e74c3c')

# Profit annotations
pi_disc = (P_A_disc - MC) * Q_A_disc + (P_B_disc - MC) * Q_B_disc
pi_uni = (P_uni - MC) * Q_uni
fig.suptitle(
    rf'Third-Degree Discrimination: $\pi_{{disc}}=\${pi_disc}$ vs '
    rf'$\pi_{{uniform}}=\${pi_uni}$  '
    rf'(+{(pi_disc - pi_uni)/pi_uni*100:.1f}\%)',
    fontsize=14, fontweight='bold', y=1.02
)

plt.tight_layout()
plt.savefig('third_degree_surplus.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.show()
```

### Simulation 2: Bundling Revenue vs. Correlation

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_customers = 5000
correlations = np.linspace(-0.95, 0.95, 50)

mean_v = [100, 100]  # mean WTP for goods A and B
std_v = 30           # std of WTP

rev_separate = []
rev_bundle = []

for rho in correlations:
    # Covariance matrix
    cov = [[std_v**2, rho * std_v**2],
           [rho * std_v**2, std_v**2]]
    valuations = np.random.multivariate_normal(mean_v, cov, n_customers)
    v_A = np.clip(valuations[:, 0], 0, None)
    v_B = np.clip(valuations[:, 1], 0, None)
    v_bundle = v_A + v_B

    # Optimal separate pricing: try a grid of prices
    best_rev_sep = 0
    for p_A in np.linspace(10, 200, 200):
        for p_B in np.linspace(10, 200, 200):
            rev = np.sum(v_A >= p_A) * p_A + np.sum(v_B >= p_B) * p_B
            if rev > best_rev_sep:
                best_rev_sep = rev
    rev_separate.append(best_rev_sep / n_customers)

    # Optimal bundle pricing: try a grid
    best_rev_bun = 0
    for p_bun in np.linspace(50, 300, 500):
        rev = np.sum(v_bundle >= p_bun) * p_bun
        if rev > best_rev_bun:
            best_rev_bun = rev
    rev_bundle.append(best_rev_bun / n_customers)

rev_separate = np.array(rev_separate)
rev_bundle = np.array(rev_bundle)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(correlations, rev_separate, 'o-', color='#e74c3c', lw=2, ms=4,
        label=r'Optimal Separate Pricing')
ax.plot(correlations, rev_bundle, 's-', color='#3498db', lw=2, ms=4,
        label=r'Optimal Pure Bundling')
ax.fill_between(correlations, rev_separate, rev_bundle,
                where=rev_bundle > rev_separate,
                alpha=0.2, color='#3498db', label='Bundling advantage')
ax.fill_between(correlations, rev_separate, rev_bundle,
                where=rev_separate > rev_bundle,
                alpha=0.2, color='#e74c3c', label='Separate advantage')

ax.set_xlabel(r'Correlation $\rho$ between $V_A$ and $V_B$', fontsize=13)
ax.set_ylabel(r'Revenue per customer (\$)', fontsize=13)
ax.set_title('Bundling vs. Separate Pricing:\nRevenue as a Function of Valuation Correlation',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.axvline(0, color='#888', ls=':', lw=1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bundling_correlation.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.show()
```

### Simulation 3: The Decoy Effect

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)
n = 500

# Two real products:
#   Product A: quality=7, price=5  (value-oriented)
#   Product B: quality=9, price=9  (premium)
# Decoy D:    quality=8, price=9  (dominated by B on quality, similar price)

# Consumer utility: U = alpha * quality - price + noise
# alpha ~ Uniform(0.5, 1.5) captures heterogeneous quality sensitivity

alpha = np.random.uniform(0.5, 1.5, n)
noise_A = np.random.normal(0, 0.3, n)
noise_B = np.random.normal(0, 0.3, n)
noise_D = np.random.normal(0, 0.3, n)

U_A = alpha * 7 - 5 + noise_A
U_B = alpha * 9 - 9 + noise_B
U_D = alpha * 8 - 9 + noise_D

# Without decoy: choose max(U_A, U_B)
choice_no_decoy = np.where(U_A > U_B, 'A', 'B')
frac_A_no = np.mean(choice_no_decoy == 'A')
frac_B_no = np.mean(choice_no_decoy == 'B')

# With decoy: choose max(U_A, U_B, U_D)
U_all = np.column_stack([U_A, U_B, U_D])
choice_idx = np.argmax(U_all, axis=1)
labels = np.array(['A', 'B', 'D'])
choice_with_decoy = labels[choice_idx]
frac_A_wd = np.mean(choice_with_decoy == 'A')
frac_B_wd = np.mean(choice_with_decoy == 'B')
frac_D_wd = np.mean(choice_with_decoy == 'D')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart: market shares
x_pos = [0, 1]
width = 0.35

# Without decoy
axes[0].bar([0 - width/2, 1 - width/2], [frac_A_no * 100, frac_B_no * 100],
            width, color=['#2ecc71', '#3498db'], edgecolor='white', lw=1.5)
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Product A\n(Value)', 'Product B\n(Premium)'], fontsize=12)
axes[0].set_ylabel('Market Share (%)', fontsize=12)
axes[0].set_title('Without Decoy', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, 80)
for i, v in enumerate([frac_A_no, frac_B_no]):
    axes[0].text(i - width/2, v * 100 + 1.5, f'{v*100:.1f}%', fontsize=13,
                fontweight='bold', ha='center')

# With decoy
bars = axes[1].bar([0, 1, 2],
                   [frac_A_wd * 100, frac_B_wd * 100, frac_D_wd * 100],
                   0.5, color=['#2ecc71', '#3498db', '#e74c3c'],
                   edgecolor='white', lw=1.5)
axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(['Product A\n(Value)', 'Product B\n(Premium)',
                         'Decoy D\n(Dominated)'], fontsize=12)
axes[1].set_ylabel('Market Share (%)', fontsize=12)
axes[1].set_title('With Decoy', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, 80)
for i, v in enumerate([frac_A_wd, frac_B_wd, frac_D_wd]):
    axes[1].text(i, v * 100 + 1.5, f'{v*100:.1f}%', fontsize=13,
                fontweight='bold', ha='center')

# Arrow showing the shift
shift = frac_B_wd - frac_B_no
fig.text(0.5, 0.02,
         f'Decoy shifts B share from {frac_B_no*100:.1f}% to {frac_B_wd*100:.1f}% '
         f'(+{shift*100:.1f} pp) — asymmetric dominance at work',
         ha='center', fontsize=12, style='italic', color='#d4d4d4')

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('decoy_effect.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.show()
```

---

## The Practical Landscape

Let us step back and survey which industries use which discrimination techniques, and why.

**Airlines** are the masters of price discrimination across all three degrees. They practice third-degree discrimination through fare classes segmented by observable characteristics (advance purchase requirements, Saturday-night stay rules, refundability). They practice second-degree discrimination through the economy/premium-economy/business/first-class menu, where seat quality is the screening device. They approximate first-degree discrimination through yield management systems that adjust prices continuously based on remaining inventory and estimated demand. An empty seat at departure has zero value, so the incentive to discriminate is enormous.

**SaaS companies** rely heavily on versioning (second-degree discrimination): Free, Pro, Enterprise tiers where the quality dimension is feature access, support level, and usage limits. They also practice third-degree discrimination through geographic pricing (lower prices in developing markets), educational discounts, and startup programs. Usage-based pricing (per API call, per seat) functions as a two-part tariff with the subscription as the fixed fee.

**Retail** combines dynamic pricing (adjusting prices over time based on demand signals), bundling (product bundles, "buy 2 get 1 free"), and couponing (third-degree discrimination where the cost of clipping coupons screens price-sensitive customers from price-insensitive ones — a form of second-degree discrimination through hassle).

**Entertainment** uses temporal discrimination (movie theaters charge more on opening weekend, books launch in hardcover before paperback), bundling (streaming services that aggregate content), and geographic pricing (different prices for the same Netflix subscription in different countries).

**Academic publishing** practices some of the most extreme price discrimination in any industry. Institutional subscriptions to journals can cost 50x what an individual subscription costs. The product is identical — the screening device is the identity of the buyer, making this third-degree discrimination between individuals and institutions.

| Industry | 1st Degree | 2nd Degree | 3rd Degree | Two-Part | Bundling |
|----------|:----------:|:----------:|:----------:|:--------:|:--------:|
| Airlines | Yield mgmt | Fare classes | Advance purchase | Loyalty programs | Route bundles |
| SaaS | Custom enterprise | Tier menus | Geographic, edu | Subscription + usage | Feature bundles |
| Retail | Personalized coupons | Product lines | Couponing, location | Membership (Costco) | Product bundles |
| Entertainment | Auctions | Temporal release | Student, senior | Streaming subscriptions | Content libraries |
| Academic | Negotiated site licenses | Print vs. digital | Institutional vs. individual | Society membership + access | Journal bundles |

### The Thread Forward

Everything in this post assumes the firm operates in isolation — it has market power and faces no strategic response from competitors. But in the real world, competitors react. If you price-discriminate and a rival undercuts your low segment, your screening mechanism collapses. If you bundle and a competitor offers a superior unbundled alternative for one component, your bundle loses its value.

Understanding how competitors respond requires game theory — Nash equilibria, Bertrand and Cournot competition, and the strategic dynamics of price wars. That is Part 3.

---

*This is Part 2 of a 5-part series on pricing strategy. [Part 1: Demand, Elasticity & Markup](/2026/04/13/demand-elasticity-monopolist-markup.html) | **Part 2: Price Discrimination** | [Part 3: Game Theory of Pricing](/2026/04/15/game-theory-competitive-pricing.html) | [Part 4: Causal Demand Estimation](/2026/04/16/causal-demand-estimation-ml.html) | [Part 5: Algorithmic Dynamic Pricing](/2026/04/17/algorithmic-dynamic-pricing-bandits.html)*
