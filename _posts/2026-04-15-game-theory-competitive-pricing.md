---
layout: post
title: "Game Theory of Competitive Pricing: Nash Equilibria, Price Wars, and Tacit Collusion"
date: 2026-04-15
category: business
---

*This is Part 3 of a 5-part series on pricing strategy. [Part 1: Demand, Elasticity & Markup](/2026/04/13/demand-elasticity-monopolist-markup.html) | [Part 2: Price Discrimination](/2026/04/14/price-discrimination-extracting-surplus.html) | **Part 3: Game Theory of Pricing** | [Part 4: Causal Demand Estimation](/2026/04/16/causal-demand-estimation-ml.html) | [Part 5: Algorithmic Dynamic Pricing](/2026/04/17/algorithmic-dynamic-pricing-bandits.html)*

Parts 1 and 2 analyzed pricing from the perspective of a monopolist --- a firm that sets prices in isolation, facing a demand curve it can exploit without interference. That is a clean mathematical problem: find the price that maximizes \(\text{profit} = (P - MC) \times Q(P)\), take the first-order condition, and you are done. But most markets are not monopolies. When you set a price, your competitors observe it and respond. When they respond, you respond to their response. This recursive strategic interaction is the domain of **game theory**, and it fundamentally changes the mathematics of optimal pricing.

The central question becomes: what price should you charge, knowing that your competitors are simultaneously solving the same problem? Your optimal price depends on their prices, which depend on your price, which depends on their prices. This circular reasoning is not a flaw --- it is the defining feature of strategic interaction. The resolution comes from a concept called **Nash equilibrium**, which identifies the price profile where everyone's strategy is simultaneously optimal given everyone else's strategy. No one wants to deviate, so the system is stable.

This post builds the game-theoretic foundations of competitive pricing from scratch. We start with the formal definition of a game, work through the two canonical oligopoly models (Bertrand and Cournot), analyze sequential competition (Stackelberg), explain why firms are trapped in a prisoner's dilemma over pricing, show how repeated interaction can sustain collusion, and then explore the contemporary problem of algorithmic collusion. The mathematics is clean, the results are surprising, and the implications for real pricing strategy are immediate.

---

## Table of Contents

1. [Games, Strategies, and Nash Equilibrium](#games-strategies-and-nash-equilibrium)
2. [Bertrand Competition --- The Price War Paradox](#bertrand-competition----the-price-war-paradox)
3. [Bertrand with Differentiated Products](#bertrand-with-differentiated-products)
4. [Cournot Competition --- Choosing Quantities](#cournot-competition----choosing-quantities)
5. [The Cournot-Nash Markup](#the-cournot-nash-markup)
6. [Stackelberg Leadership](#stackelberg-leadership)
7. [The Prisoner's Dilemma of Pricing](#the-prisoners-dilemma-of-pricing)
8. [Repeated Games and Tacit Collusion](#repeated-games-and-tacit-collusion)
9. [Algorithmic Collusion](#algorithmic-collusion)
10. [Two-Sided Market Pricing](#two-sided-market-pricing)
11. [Python Simulations](#python-simulations)
12. [From Theory to Real Markets](#from-theory-to-real-markets)

---

## Games, Strategies, and Nash Equilibrium

Before we can analyze competitive pricing, we need to define what a "game" means in the mathematical sense. A **game** is any situation where the outcome for each participant depends not only on their own decisions but also on the decisions of others. Formally, a game has three elements:

1. **Players**: a set of decision-makers \(\{1, 2, \ldots, n\}\). In pricing, these are the competing firms.
2. **Strategy sets**: each player \(i\) has a set \(S_i\) of available strategies. For a price-setting firm, \(S_i\) might be the interval \([0, \infty)\) of possible prices. For a quantity-setting firm, it might be the set of possible production levels.
3. **Payoff functions**: each player \(i\) has a payoff function \(\pi_i(s_1, s_2, \ldots, s_n)\) that maps a strategy profile (one strategy per player) to a real number. For firms, this is profit.

The key feature is the interdependence: player \(i\)'s payoff depends on the strategies chosen by *all* players, not just their own. This is what makes it a game rather than an optimization problem.

Given this structure, each player wants to choose the strategy that maximizes their own payoff. But the optimal strategy depends on what the other players choose. This leads to the concept of a **best response**. Player \(i\)'s best response to the strategies of all other players \(s_{-i} = (s_1, \ldots, s_{i-1}, s_{i+1}, \ldots, s_n)\) is the strategy that maximizes their payoff:

$$\text{BR}_i(s_{-i}) = \arg\max_{s_i \in S_i} \pi_i(s_i, s_{-i})$$

The best response is a function: given what everyone else is doing, it tells you what you should do. The problem is that everyone is computing their best response simultaneously. We need a concept that resolves this circularity.

A **Nash Equilibrium** (NE) is a strategy profile \((s_1^*, s_2^*, \ldots, s_n^*)\) where every player is playing their best response to everyone else's strategy. Formally:

$$\pi_i(s_i^*, s_{-i}^*) \geq \pi_i(s_i, s_{-i}^*) \quad \text{for all } s_i \in S_i \text{ and all } i$$

In words: no player can improve their payoff by unilaterally changing their strategy, given that all other players keep their strategies fixed. This is not about optimality in any global sense --- it is about **stability**. A Nash equilibrium is a state from which no one has an incentive to deviate. It is a fixed point of the best-response mapping: if you start at a NE, the best-response dynamics leave you there.

An important subtlety: Nash equilibrium does not mean everyone is happy. It does not mean the outcome is socially optimal. It means the outcome is self-enforcing --- no individual can do better by changing their own behavior alone. As we will see, NE can be collectively terrible (the prisoner's dilemma is the canonical example).

**Existence.** John Nash proved in 1950 that every finite game (finitely many players, finitely many strategies) has at least one Nash equilibrium, possibly in **mixed strategies** (probability distributions over pure strategies). For the continuous games we study in pricing (where strategy sets are intervals of prices or quantities), existence follows from more general fixed-point theorems, and the equilibria are typically in pure strategies.

---

## Bertrand Competition --- The Price War Paradox

The simplest model of price competition is due to Joseph Bertrand (1883). Consider two firms selling **identical products** with the same constant marginal cost \(c\). Each firm simultaneously chooses a price. Consumers are perfectly rational and buy from whichever firm charges less. If both charge the same price, demand splits equally. Market demand at price \(P\) is \(Q(P) = a - bP\), where \(a > 0\) and \(b > 0\).

Each firm's demand is:

$$q_i(P_i, P_j) = \begin{cases} a - bP_i & \text{if } P_i < P_j \\ \frac{a - bP_i}{2} & \text{if } P_i = P_j \\ 0 & \text{if } P_i > P_j \end{cases}$$

And each firm's profit is \(\pi_i = (P_i - c) \cdot q_i(P_i, P_j)\).

**Claim**: the unique Nash equilibrium is \(P_1^* = P_2^* = c\). Both firms price at marginal cost, earning zero profit.

**Proof.** We need to show two things: (1) the profile \(P_1 = P_2 = c\) is a Nash equilibrium, and (2) no other price profile is a Nash equilibrium.

**(1) \(P_1 = P_2 = c\) is a NE.** At this profile, both firms earn zero profit. If firm 1 raises its price above \(c\), it sells nothing (all consumers go to firm 2), so its profit remains zero. If firm 1 lowers its price below \(c\), it captures the entire market but sells at a loss, so profit is negative. Therefore firm 1 cannot improve its payoff by deviating. By symmetry, neither can firm 2. This is a Nash equilibrium.

**(2) No other profile is a NE.** Consider the possible cases:

- *Case: \(P_1 = P_2 = P > c\).* Both firms earn positive profit \(\frac{(P-c)(a-bP)}{2}\). But firm 1 could charge \(P - \varepsilon\) for tiny \(\varepsilon > 0\), stealing the entire market. Its profit would be approximately \((P - c)(a - bP)\), nearly double its current profit. So this is not a NE.

- *Case: \(P_1 > P_2 > c\).* Firm 1 sells nothing and earns zero. It could undercut firm 2 by setting \(P_1 = P_2 - \varepsilon\), earning positive profit. So firm 1 has an incentive to deviate. Not a NE.

- *Case: \(P_1 > P_2 = c\).* Firm 1 sells nothing, firm 2 earns zero. Firm 2 could raise its price slightly above \(c\) and earn positive profit (since firm 1's price is even higher, firm 2 still captures all demand). So firm 2 wants to deviate. Not a NE.

- *Case: \(P_i < c\) for some firm.* That firm is making negative profit and would prefer to raise its price to \(c\). Not a NE.

Every case other than \(P_1 = P_2 = c\) admits a profitable deviation. Therefore the unique NE is marginal-cost pricing. \(\square\)

This is the **Bertrand Paradox**: with just **two** firms selling identical products, competition drives the price all the way down to marginal cost --- the same outcome as **perfect competition** with infinitely many firms. The standard monopoly markup of Part 1 vanishes entirely. Going from one firm to two firms destroys all market power.

This seems absurd. We observe real oligopolies --- airlines, telecoms, cereal manufacturers --- with just a handful of firms and healthy profit margins. The paradox tells us that the Bertrand model's assumptions are too strong. It breaks down because of:

- **Capacity constraints**: firms cannot serve the entire market by undercutting (Edgeworth, 1897)
- **Product differentiation**: Coke and Pepsi are substitutes, not identical (next section)
- **Repeated interaction**: firms compete not once but repeatedly, enabling collusion (Section 8)
- **Search costs**: consumers do not instantly switch to the cheapest firm

Each relaxation restores positive markups, and we explore the most important ones below.

---

## Bertrand with Differentiated Products

The Bertrand paradox dissolves once we recognize that competing products are rarely identical. Coca-Cola and Pepsi, iPhone and Samsung Galaxy, AWS and Google Cloud --- these are substitutes, but not perfect ones. Some consumers prefer one over the other and will not switch just because the rival is a penny cheaper.

We model this by giving each firm a demand function that depends on both its own price and its rival's price. For two firms:

$$Q_1 = a - bP_1 + dP_2$$

$$Q_2 = a - bP_2 + dP_1$$

Here \(b > 0\) is the own-price sensitivity (higher \(P_1\) reduces firm 1's demand) and \(d > 0\) is the cross-price sensitivity (higher \(P_2\) pushes demand toward firm 1). The parameter \(d\) captures the degree of substitutability: when \(d = 0\), the firms are independent monopolists; as \(d \to b\), the products become nearly identical and we approach the original Bertrand setting.

Firm 1 maximizes profit:

$$\pi_1 = (P_1 - c)(a - bP_1 + dP_2)$$

Taking the first-order condition with respect to \(P_1\):

$$\frac{\partial \pi_1}{\partial P_1} = (a - bP_1 + dP_2) + (P_1 - c)(-b) = 0$$

$$a - bP_1 + dP_2 - bP_1 + bc = 0$$

$$a - 2bP_1 + dP_2 + bc = 0$$

Solving for \(P_1\):

$$P_1^* = \frac{a + bc + dP_2}{2b}$$

This is firm 1's **best response function**. Notice that it is an *upward-sloping* function of \(P_2\): when your rival raises their price, your optimal response is to raise yours too (though not by as much --- the slope is \(d/(2b) < 1\)). Prices are **strategic complements**. This makes economic sense: if Pepsi raises its price, some Pepsi drinkers switch to Coke, so Coke can afford to raise its price and still gain customers.

By symmetry, firm 2's best response is:

$$P_2^* = \frac{a + bc + dP_1}{2b}$$

To find the Nash equilibrium, we solve these two equations simultaneously. Substituting the second into the first:

$$P_1^* = \frac{a + bc + d \cdot \frac{a + bc + dP_1^*}{2b}}{2b} = \frac{a + bc}{2b} + \frac{d(a + bc)}{4b^2} + \frac{d^2 P_1^*}{4b^2}$$

$$P_1^* \left(1 - \frac{d^2}{4b^2}\right) = \frac{(a + bc)(2b + d)}{4b^2}$$

$$P_1^* = \frac{(a + bc)(2b + d)}{4b^2 - d^2} = \frac{(a+bc)(2b+d)}{(2b-d)(2b+d)} = \frac{a + bc}{2b - d}$$

By symmetry, \(P_2^* = \frac{a + bc}{2b - d}\). Both firms charge the same price in the symmetric equilibrium:

$$\boxed{P^* = \frac{a + bc}{2b - d}}$$

The markup over marginal cost is:

$$P^* - c = \frac{a + bc}{2b - d} - c = \frac{a + bc - 2bc + dc}{2b - d} = \frac{a - (2b - d)c + 2c(b - b) + dc}{2b - d}$$

Let us simplify more carefully:

$$P^* - c = \frac{a + bc - c(2b - d)}{2b - d} = \frac{a - bc + dc}{2b - d} = \frac{a - c(b - d)}{2b - d}$$

Since \(a > bc\) (market exists at marginal cost) and \(d < b\) (own-price effect dominates cross-price effect), we have \(a - c(b - d) > a - bc + 0 > 0\) and \(2b - d > 0\), so the markup is **positive**. The Bertrand paradox is resolved.

The key comparative statics: **more differentiation** (lower \(d\), meaning products are less substitutable) leads to **higher markups**. As \(d \to 0\), the equilibrium price approaches the monopoly price \((a + bc)/(2b)\). As \(d \to b\), the markup shrinks toward zero, recovering the Bertrand paradox for identical products.

<svg viewBox="0 0 500 400" xmlns="http://www.w3.org/2000/svg">
  <rect width="500" height="400" fill="#1a1a2e"/>
  <!-- Axes -->
  <line x1="60" y1="340" x2="460" y2="340" stroke="#d4d4d4" stroke-width="1.5"/>
  <line x1="60" y1="340" x2="60" y2="40" stroke="#d4d4d4" stroke-width="1.5"/>
  <text x="260" y="385" fill="#d4d4d4" font-size="14" text-anchor="middle">P₁</text>
  <text x="25" y="190" fill="#d4d4d4" font-size="14" text-anchor="middle" transform="rotate(-90 25 190)">P₂</text>
  <!-- Best response 1: P1 = (a+bc+dP2)/(2b), so P2 = (2bP1 - a - bc)/d => upward sloping line in (P1, P2) space -->
  <!-- BR1: P1 as function of P2: starts at P1=(a+bc)/(2b) when P2=0, slope d/(2b) -->
  <!-- Plot as P2 as function of P1: P2 = (2bP1 - a - bc)/d -->
  <!-- BR1 line: from (120, 40) to (420, 340) -->
  <line x1="120" y1="340" x2="440" y2="60" stroke="#4fc3f7" stroke-width="2.5"/>
  <text x="445" y="50" fill="#4fc3f7" font-size="12">BR₁(P₂)</text>
  <!-- BR2: P2 as function of P1: P2 = (a+bc+dP1)/(2b), slope d/(2b) -->
  <!-- BR2 line: from (60, 220) to (440, 120) but as upward in P1 axis -->
  <line x1="60" y1="250" x2="380" y2="60" stroke="#ff8a65" stroke-width="2.5"/>
  <text x="385" y="75" fill="#ff8a65" font-size="12">BR₂(P₁)</text>
  <!-- Nash equilibrium intersection -->
  <circle cx="230" cy="155" r="6" fill="#e040fb" stroke="white" stroke-width="1.5"/>
  <text x="248" y="148" fill="#e040fb" font-size="13" font-weight="bold">Nash Equilibrium</text>
  <text x="248" y="166" fill="#e040fb" font-size="11">(P*, P*)</text>
  <!-- 45-degree line for reference -->
  <line x1="60" y1="340" x2="400" y2="0" stroke="#d4d4d4" stroke-width="0.5" stroke-dasharray="6,4"/>
  <text x="395" y="15" fill="#d4d4d4" font-size="10" opacity="0.6">45°</text>
  <!-- MC reference -->
  <line x1="100" y1="340" x2="100" y2="30" stroke="#66bb6a" stroke-width="1" stroke-dasharray="4,4"/>
  <text x="105" y="30" fill="#66bb6a" font-size="10">P = c</text>
  <line x1="60" y1="300" x2="460" y2="300" stroke="#66bb6a" stroke-width="1" stroke-dasharray="4,4"/>
  <text x="462" y="304" fill="#66bb6a" font-size="10">c</text>
  <!-- Title -->
  <text x="260" y="25" fill="#d4d4d4" font-size="15" text-anchor="middle" font-weight="bold">Best Response Functions — Differentiated Bertrand</text>
</svg>

The diagram above shows the two best response functions in \((P_1, P_2)\) space. Both are upward-sloping (strategic complements). Their intersection is the Nash equilibrium, which lies above the marginal cost line --- both firms earn positive profit.

---

## Cournot Competition --- Choosing Quantities

An alternative model, due to Antoine Augustin Cournot (1838), assumes that firms choose **quantities** rather than prices, and the market price adjusts to clear total supply. Why would firms choose quantities? In industries like oil, steel, semiconductors, and airlines, firms effectively commit to production levels (drilling plans, factory output, seat capacity) and the market price results from the aggregate supply hitting the demand curve.

Consider two firms with constant marginal cost \(c\). Market inverse demand is:

$$P = \frac{a - (Q_1 + Q_2)}{b}$$

where \(Q_1\) and \(Q_2\) are the quantities produced by firms 1 and 2. (This is just \(Q = a - bP\) solved for \(P\).)

Firm 1 maximizes profit:

$$\pi_1 = (P - c)Q_1 = \left(\frac{a - Q_1 - Q_2}{b} - c\right)Q_1 = \frac{(a - bc) - Q_1 - Q_2}{b} \cdot Q_1$$

Let us define \(A = a - bc\) to simplify notation. Then:

$$\pi_1 = \frac{Q_1(A - Q_1 - Q_2)}{b}$$

First-order condition:

$$\frac{\partial \pi_1}{\partial Q_1} = \frac{A - 2Q_1 - Q_2}{b} = 0$$

$$Q_1^* = \frac{A - Q_2}{2}$$

This is firm 1's best response function. Notice that it is **downward-sloping** in \(Q_2\): when your rival produces more, you produce less. Quantities are **strategic substitutes**. This is the opposite of the differentiated Bertrand model where prices were strategic complements. The intuition is straightforward: if your rival floods the market with output, the price falls, making it less profitable for you to produce as much.

Firm 2's best response is symmetric: \(Q_2^* = \frac{A - Q_1}{2}\).

Solving simultaneously:

$$Q_1 = \frac{A - \frac{A - Q_1}{2}}{2} = \frac{2A - A + Q_1}{4} = \frac{A + Q_1}{4}$$

$$4Q_1 = A + Q_1 \implies 3Q_1 = A \implies Q_1^* = \frac{A}{3} = \frac{a - bc}{3}$$

By symmetry, \(Q_2^* = \frac{a - bc}{3}\). Total output:

$$Q^* = Q_1^* + Q_2^* = \frac{2(a - bc)}{3}$$

Equilibrium price:

$$P^* = \frac{a - Q^*}{b} = \frac{a - \frac{2(a - bc)}{3}}{b} = \frac{3a - 2a + 2bc}{3b} = \frac{a + 2bc}{3b}$$

Each firm's profit:

$$\pi^* = (P^* - c) Q_i^* = \left(\frac{a + 2bc}{3b} - c\right)\frac{a - bc}{3} = \frac{a - bc}{3b} \cdot \frac{a - bc}{3} = \frac{(a - bc)^2}{9b}$$

Let us compare this to the benchmark cases. The **monopoly** outcome (one firm, or two firms colluding) gives \(Q_m = \frac{a - bc}{2}\), \(P_m = \frac{a + bc}{2b}\). The **perfectly competitive** outcome gives \(Q_c = a - bc\), \(P_c = c\). Cournot duopoly sits between them:

$$Q_m = \frac{A}{2} < Q^* = \frac{2A}{3} < Q_c = A$$

$$P_c = c < P^* = \frac{a + 2bc}{3b} < P_m = \frac{a + bc}{2b}$$

The Cournot outcome is intermediate: more competition than monopoly, but less than perfect competition. This is much more realistic than the Bertrand result.

**Generalization to \(n\) firms.** With \(n\) symmetric Cournot competitors, each firm's best response leads to:

$$Q_i^* = \frac{a - bc}{n + 1}, \qquad Q^* = \frac{n(a - bc)}{n+1}, \qquad P^* = \frac{a + nbc}{(n+1)b}$$

As \(n \to \infty\), \(P^* \to c\): the Cournot equilibrium converges to perfect competition. With a small number of firms, the markup is significant. With many firms, it erodes. This is the intuitive behavior we expect from real markets.

---

## The Cournot-Nash Markup

The Cournot model yields a beautiful connection between market structure and pricing power. From the \(n\)-firm Cournot equilibrium, the **Lerner index** (the relative markup over marginal cost) is:

$$L = \frac{P - MC}{P}$$

From the \(n\)-firm solution \(P^* = \frac{a + nbc}{(n+1)b}\), and noting that the market demand elasticity at the equilibrium price is \(\varepsilon = -\frac{bP}{Q} = -\frac{bP}{a - bP}\), we can derive (after some algebra that mirrors the monopoly case from Part 1):

$$\boxed{L = \frac{1}{n|\varepsilon|}}$$

where \(|\varepsilon|\) is the absolute value of the market demand elasticity. This is a remarkable formula. It says the markup depends on exactly two things:

1. **\(n\)**: the number of firms. More competitors means lower markup.
2. **\(|\varepsilon|\)**: the demand elasticity. More elastic demand means lower markup.

When \(n = 1\), this reduces to the monopoly markup formula \(L = 1/|\varepsilon|\) from Part 1. When \(n \to \infty\), the markup goes to zero (perfect competition).

The connection to market concentration runs even deeper. Define the **Herfindahl-Hirschman Index** (HHI), a standard measure of market concentration used by antitrust authorities worldwide. The HHI is the sum of squared market shares:

$$\text{HHI} = \sum_{i=1}^n s_i^2$$

where \(s_i\) is firm \(i\)'s market share. For symmetric Cournot firms, each has share \(1/n\), so:

$$\text{HHI} = n \cdot \left(\frac{1}{n}\right)^2 = \frac{1}{n}$$

Substituting into the Lerner index:

$$L = \frac{\text{HHI}}{|\varepsilon|}$$

This is why antitrust authorities care about HHI. It is not an arbitrary measure --- it maps directly to expected markups via the Cournot model. A merger that increases HHI from 0.10 to 0.25 (say, from 10 symmetric firms to 4) is expected to roughly 2.5x the markup. The U.S. Department of Justice and FTC use HHI thresholds of 0.15 (moderately concentrated) and 0.25 (highly concentrated) to evaluate mergers, and this Cournot-Nash relationship is the theoretical justification.

---

## Stackelberg Leadership

In both Bertrand and Cournot, firms move simultaneously. But what if one firm moves first? This is the **Stackelberg model** (1934): one firm (the **leader**) commits to a strategy before the other (the **follower**) observes it and responds. This sequential structure changes the equilibrium dramatically.

We solve it by **backward induction**: first, determine what the follower will do as a function of the leader's choice; then, the leader maximizes profit knowing the follower's response.

Using the Cournot setup with linear demand, inverse demand \(P = (a - Q_1 - Q_2)/b\), and marginal cost \(c\):

**Step 1: Follower's problem.** Given the leader's quantity \(Q_L\), the follower maximizes:

$$\pi_F = \left(\frac{a - Q_L - Q_F}{b} - c\right)Q_F$$

This is exactly the Cournot best response:

$$Q_F^*(Q_L) = \frac{a - bc - Q_L}{2}$$

**Step 2: Leader's problem.** The leader substitutes the follower's best response into its own profit function:

$$\pi_L = \left(\frac{a - Q_L - Q_F^*(Q_L)}{b} - c\right)Q_L = \left(\frac{a - Q_L - \frac{a - bc - Q_L}{2}}{b} - c\right)Q_L$$

Simplifying the price:

$$P = \frac{a - Q_L - \frac{a - bc - Q_L}{2}}{b} = \frac{2a - 2Q_L - a + bc + Q_L}{2b} = \frac{a + bc - Q_L}{2b}$$

So:

$$\pi_L = \left(\frac{a + bc - Q_L}{2b} - c\right)Q_L = \frac{(a - bc - Q_L)}{2b} \cdot Q_L$$

First-order condition:

$$\frac{\partial \pi_L}{\partial Q_L} = \frac{a - bc - 2Q_L}{2b} = 0 \implies Q_L^* = \frac{a - bc}{2}$$

The follower produces:

$$Q_F^* = \frac{a - bc - \frac{a - bc}{2}}{2} = \frac{a - bc}{4}$$

Total output: \(Q = \frac{3(a-bc)}{4}\). Compare this to the Cournot duopoly total output of \(\frac{2(a-bc)}{3}\): the Stackelberg market produces **more** output and has a **lower** price. The leader produces more than the Cournot quantity (\(\frac{a-bc}{2}\) vs \(\frac{a-bc}{3}\)), while the follower produces less (\(\frac{a-bc}{4}\) vs \(\frac{a-bc}{3}\)).

The leader earns higher profit than a Cournot duopolist: \(\pi_L = \frac{(a-bc)^2}{8b}\) versus \(\pi_C = \frac{(a-bc)^2}{9b}\). The follower earns less: \(\pi_F = \frac{(a-bc)^2}{16b}\). Being the first mover is a significant advantage.

This has practical implications: **market entry deterrence**. An incumbent firm can commit to high output (or low prices, or large capacity investment) to make the market less attractive for a potential entrant. The entrant, observing the incumbent's commitment, optimally produces less --- or may decide not to enter at all if the remaining profit is below its fixed costs. This is the theoretical basis for predatory pricing and penetration pricing strategies.

---

## The Prisoner's Dilemma of Pricing

Let us now reframe competitive pricing as a **Prisoner's Dilemma** --- the most famous game in all of game theory. The Prisoner's Dilemma is a game where individual rationality leads to a collectively suboptimal outcome. It captures exactly the tension that firms face when setting prices.

Suppose two firms can each choose one of two strategies: **High Price** (the collusive price that maximizes joint profit) or **Low Price** (the competitive price that undercuts the rival). The payoff matrix looks like this:

<svg viewBox="0 0 500 320" xmlns="http://www.w3.org/2000/svg">
  <rect width="500" height="320" fill="#1a1a2e"/>
  <!-- Title -->
  <text x="250" y="30" fill="#d4d4d4" font-size="15" text-anchor="middle" font-weight="bold">Pricing Prisoner's Dilemma</text>
  <!-- Column labels -->
  <text x="310" y="65" fill="#4fc3f7" font-size="13" text-anchor="middle">Firm 2: High</text>
  <text x="430" y="65" fill="#4fc3f7" font-size="13" text-anchor="middle">Firm 2: Low</text>
  <!-- Row labels -->
  <text x="140" y="130" fill="#ff8a65" font-size="13" text-anchor="middle">Firm 1: High</text>
  <text x="140" y="210" fill="#ff8a65" font-size="13" text-anchor="middle">Firm 1: Low</text>
  <!-- Grid -->
  <rect x="250" y="80" width="120" height="70" fill="none" stroke="#d4d4d4" stroke-width="1"/>
  <rect x="370" y="80" width="120" height="70" fill="none" stroke="#d4d4d4" stroke-width="1"/>
  <rect x="250" y="150" width="120" height="70" fill="none" stroke="#d4d4d4" stroke-width="1"/>
  <rect x="370" y="150" width="120" height="70" fill="none" stroke="#d4d4d4" stroke-width="1"/>
  <!-- Payoffs -->
  <text x="310" y="115" fill="#66bb6a" font-size="14" text-anchor="middle" font-weight="bold">$50M, $50M</text>
  <text x="310" y="133" fill="#d4d4d4" font-size="10" text-anchor="middle">(cooperate)</text>
  <text x="430" y="115" fill="#ef5350" font-size="14" text-anchor="middle" font-weight="bold">$10M, $70M</text>
  <text x="430" y="133" fill="#d4d4d4" font-size="10" text-anchor="middle">(sucker)</text>
  <text x="310" y="195" fill="#ef5350" font-size="14" text-anchor="middle" font-weight="bold">$70M, $10M</text>
  <text x="310" y="213" fill="#d4d4d4" font-size="10" text-anchor="middle">(temptation)</text>
  <text x="430" y="195" fill="#ffa726" font-size="14" text-anchor="middle" font-weight="bold">$30M, $30M</text>
  <text x="430" y="213" fill="#d4d4d4" font-size="10" text-anchor="middle">(Nash eq.)</text>
  <!-- Arrows showing dominant strategy -->
  <text x="250" y="275" fill="#d4d4d4" font-size="12" text-anchor="middle">Each firm's dominant strategy: Low Price</text>
  <text x="250" y="295" fill="#d4d4d4" font-size="12" text-anchor="middle">NE: (Low, Low) = $30M each — but (High, High) = $50M each!</text>
</svg>

Let us walk through the logic. From firm 1's perspective:

- **If firm 2 chooses High**: firm 1 gets $50M from High and $70M from Low. Low is better.
- **If firm 2 chooses Low**: firm 1 gets $10M from High and $30M from Low. Low is better.

Regardless of what firm 2 does, firm 1 is better off choosing Low Price. Low Price is a **dominant strategy** for firm 1. By symmetry, it is also a dominant strategy for firm 2. The unique Nash equilibrium is (Low, Low), giving each firm $30M.

But notice: both firms would be better off at (High, High), which gives $50M each. The collectively optimal outcome is not individually stable. If both firms agreed to keep prices high, each would have an incentive to secretly undercut and grab $70M while the rival gets only $10M. The agreement unravels.

This is the fundamental tension of oligopoly pricing: **individual rationality leads to collective suboptimality**. The "invisible hand" here does not guide the market to efficiency in a way that benefits the firms --- it guides them to a competitive outcome where consumers capture most of the surplus. From the firms' perspective, competition is a trap. From society's perspective, this is exactly what we want.

---

## Repeated Games and Tacit Collusion

The Prisoner's Dilemma analysis above assumes the game is played **once**. But real firms do not interact once and walk away. Airlines compete on the same routes quarter after quarter. Gas stations compete on the same street year after year. This changes the analysis profoundly.

The key insight is that in a **repeated game**, firms can use **future behavior** to enforce cooperation in the present. If you defect today, I will punish you tomorrow. This threat of future punishment can make cooperation (both charging high prices) sustainable.

Consider the pricing game repeated infinitely many times, with both firms discounting future profits by a **discount factor** \(\delta \in (0, 1)\). A dollar of profit next period is worth \(\delta\) dollars today. The discount factor captures both the time value of money and the probability that the game continues (if the market might disappear, future payoffs matter less).

The simplest enforcement mechanism is a **grim trigger strategy**: both firms charge the collusive (high) price. If anyone deviates, both revert to the competitive (low) price **forever**. Let us derive when this sustains collusion.

Define the per-period payoffs:
- \(\pi_{\text{collude}}\): profit when both charge high (e.g., $50M)
- \(\pi_{\text{deviate}}\): profit from undercutting when the rival charges high (e.g., $70M)
- \(\pi_{\text{compete}}\): profit when both charge low (e.g., $30M)

**If firm 1 cooperates forever**, its discounted payoff is:

$$V_{\text{cooperate}} = \pi_{\text{collude}} + \delta \pi_{\text{collude}} + \delta^2 \pi_{\text{collude}} + \cdots = \frac{\pi_{\text{collude}}}{1 - \delta}$$

**If firm 1 deviates in period 1**, it gets the deviation payoff once, then the competitive payoff forever after (the rival triggers punishment):

$$V_{\text{deviate}} = \pi_{\text{deviate}} + \delta \pi_{\text{compete}} + \delta^2 \pi_{\text{compete}} + \cdots = \pi_{\text{deviate}} + \frac{\delta \pi_{\text{compete}}}{1 - \delta}$$

Cooperation is sustainable if \(V_{\text{cooperate}} \geq V_{\text{deviate}}\):

$$\frac{\pi_{\text{collude}}}{1 - \delta} \geq \pi_{\text{deviate}} + \frac{\delta \pi_{\text{compete}}}{1 - \delta}$$

$$\pi_{\text{collude}} \geq (1 - \delta)\pi_{\text{deviate}} + \delta \pi_{\text{compete}}$$

$$\pi_{\text{collude}} - \delta \pi_{\text{compete}} \geq (1 - \delta)\pi_{\text{deviate}}$$

$$\pi_{\text{collude}} - \delta \pi_{\text{compete}} \geq \pi_{\text{deviate}} - \delta \pi_{\text{deviate}}$$

$$\delta(\pi_{\text{deviate}} - \pi_{\text{compete}}) \geq \pi_{\text{deviate}} - \pi_{\text{collude}}$$

$$\boxed{\delta \geq \frac{\pi_{\text{deviate}} - \pi_{\text{collude}}}{\pi_{\text{deviate}} - \pi_{\text{compete}}}}$$

The right-hand side is the **critical discount factor** \(\delta^*\). Collusion is sustainable if and only if firms are sufficiently patient (high \(\delta\)).

Using our numerical example: \(\pi_{\text{collude}} = 50\), \(\pi_{\text{deviate}} = 70\), \(\pi_{\text{compete}} = 30\):

$$\delta^* = \frac{70 - 50}{70 - 30} = \frac{20}{40} = 0.5$$

If \(\delta \geq 0.5\), collusion is sustainable. Since real discount factors are typically close to 1 (firms care a lot about future profits), collusion is easily sustainable in this example.

This result is a special case of the **Folk Theorem**, one of the most powerful results in game theory: in an infinitely repeated game with sufficiently patient players, **any** payoff profile that gives each player more than their one-shot Nash equilibrium payoff can be sustained as a subgame-perfect equilibrium. The folk theorem tells us that patience + repeated interaction = an enormous range of sustainable outcomes, including full collusion.

**Factors that facilitate collusion:**

- **Fewer firms**: with more firms, the deviation payoff is larger relative to the collusion payoff (you steal from more rivals), so \(\delta^*\) is higher. Collusion is harder to sustain.
- **Transparent pricing**: if you cannot observe your rival's price, you cannot detect deviations and trigger punishment. Opacity undermines collusion.
- **Frequent interaction**: the more often firms interact, the faster punishment arrives, making deviation less attractive.
- **Symmetric firms**: asymmetric costs or capacities create disagreement about the "fair" collusive price, making coordination harder.
- **High barriers to entry**: if collusion attracts new entrants who undercut the cartel, the collusive price is not sustainable.

This explains why explicit cartels (OPEC, the lysine cartel, the LCD panel price-fixing conspiracy) tend to involve a small number of firms in concentrated industries with observable prices and high entry barriers.

---

## Algorithmic Collusion

Here is where classical game theory meets the modern economy. Increasingly, prices are not set by human managers but by **pricing algorithms**. Amazon has millions of products whose prices are adjusted automatically by software. Airlines use revenue management systems that update fares in real time. Gas stations use electronic price signs connected to competitor-monitoring services.

The question is: can these algorithms learn to **collude** without being explicitly programmed to do so?

The answer, alarmingly, appears to be **yes**. Research by Calvano, Calzolari, Denicolo, and Pastorello (2020) showed that simple **Q-learning algorithms** (a basic reinforcement learning method) consistently learn to maintain supra-competitive prices in experimental oligopoly settings. The algorithms were not programmed to collude. They were programmed only to maximize their own profit. Through repeated interaction, they independently discovered that keeping prices high was a better long-term strategy than undercutting.

The mechanism is exactly the repeated-game logic of the previous section, but implemented by algorithms rather than humans. The algorithms learn from experience that undercutting triggers price wars (because the rival algorithm retaliates), while maintaining high prices leads to stable, high profits. They effectively learn grim-trigger-like strategies without being told to.

This creates a profound regulatory challenge. Traditional antitrust law requires **agreement** for price-fixing to be illegal. Two CEOs meeting in a hotel room to fix prices is a felony. But if two pricing algorithms independently converge on collusive prices through trial and error --- without any human communication --- is that illegal? The algorithms were not programmed to collude; they learned to. There was no meeting, no communication, no agreement in any traditional legal sense.

Consider the **Amazon Buy Box** --- the "Add to Cart" button that goes to one seller. Multiple sellers compete for the Buy Box, and most use automated repricing software. These algorithms sometimes converge on rotation patterns that resemble turn-taking collusion: seller A wins the Buy Box for a while at a high price, then lets seller B win at a similar price. This pattern maximizes joint profit without any explicit coordination.

The European Commission and the U.S. Federal Trade Commission are actively grappling with this issue. Some legal scholars argue that using an algorithm that predictably leads to collusive outcomes should itself constitute an agreement. Others argue that you cannot make it illegal for firms to use profit-maximizing software --- that would be absurd. The resolution is far from settled, and it is one of the most important pricing policy questions of the 2020s.

The practical implication for pricing strategy: if you operate in an oligopoly and your competitors use algorithmic pricing, your own algorithm's behavior will affect the equilibrium. An overly aggressive algorithm that always undercuts may trigger a price war. A more measured algorithm that matches competitor moves may sustain higher margins. Understanding the game-theoretic dynamics is essential for designing pricing algorithms that do not accidentally destroy your own profitability.

---

## Two-Sided Market Pricing

The game theory of pricing takes on an entirely new dimension in **platform markets** --- businesses that serve two distinct groups of users and create value by facilitating interactions between them. Google connects searchers with advertisers. Uber connects riders with drivers. Visa connects cardholders with merchants. Amazon Marketplace connects buyers with sellers.

The defining feature of a platform is **cross-side network effects**: the value to users on one side depends on the number of users on the other side. More riders make the platform more attractive to drivers (shorter idle time), and more drivers make it more attractive to riders (shorter wait times). These cross-side effects fundamentally change the pricing calculus.

The foundational model is due to Rochet and Tirole (2003). A platform charges price \(p_B\) to buyers and \(p_S\) to sellers. The total price is \(P = p_B + p_S\). Buyer demand depends on the price they pay and the number of sellers (and vice versa):

$$N_B = D_B(p_B, N_S), \qquad N_S = D_S(p_S, N_B)$$

The platform's profit is:

$$\pi = (p_B - c_B)N_B + (p_S - c_S)N_S$$

The key insight from Rochet-Tirole is that the **total price level** follows standard monopoly pricing logic (the generalization from Part 1), but the **allocation** of the price between the two sides depends on demand elasticities and cross-side externalities. The platform wants to subsidize the side that generates more cross-side value.

More precisely, the optimal price structure satisfies:

$$\frac{p_B - c_B + \alpha_S \cdot p_S}{p_B} = \frac{1}{|\varepsilon_B|}$$

where \(\alpha_S\) captures the marginal effect of an additional buyer on seller participation. The platform sets \(p_B\) below what a standard monopolist would charge, because each additional buyer generates positive externalities for the seller side (which the platform captures through \(p_S\)).

This explains several patterns that seem paradoxical from a single-sided perspective:

- **Google charges advertisers but not users.** User demand is highly elastic (many alternative search engines), and each additional user generates large value for advertisers (more eyeballs). So users are the "subsidy side" and advertisers are the "money side."

- **Credit card companies charge merchants, not cardholders.** Card networks often give cashback rewards to cardholders (effectively negative prices!) because each additional cardholder makes the network more valuable to merchants, who pay interchange fees.

- **Uber initially subsidized both riders and drivers.** In the growth phase, the platform needed to build both sides simultaneously. This is the **chicken-and-egg problem** of platforms: you need sellers to attract buyers, but you need buyers to attract sellers.

The competitive dynamics between platforms add another game-theoretic layer. When Uber and Lyft compete, they are playing a game where each platform's pricing affects not just its own demand but also its rival's demand on both sides. The equilibrium involves complex cross-platform externalities that make standard oligopoly models look simple by comparison.

For pricing practitioners, the key takeaway is: **on a platform, the optimal price on one side can be zero or even negative**. You are not leaving money on the table --- you are investing in cross-side network effects that you monetize on the other side.

---

## Python Simulations

### Simulation 1: Bertrand vs Cournot Best Responses

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
a, b, c_cost, d = 100, 2, 10, 1.0  # differentiated Bertrand params
A = a - b * c_cost  # = 80

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left panel: Differentiated Bertrand ---
ax = axes[0]
P_range = np.linspace(c_cost, 40, 200)

# Best responses: P_i = (a + bc + dP_j) / (2b)
BR1_P = (a + b * c_cost + d * P_range) / (2 * b)
BR2_P = (a + b * c_cost + d * P_range) / (2 * b)  # symmetric, but plot P2 vs P1

# NE
P_NE = (a + b * c_cost) / (2 * b - d)

ax.plot(P_range, BR1_P, color='#4fc3f7', linewidth=2.5, label=r'$BR_1(P_2)$: $P_1 = \frac{a+bc+dP_2}{2b}$')
ax.plot(BR2_P, P_range, color='#ff8a65', linewidth=2.5, label=r'$BR_2(P_1)$: $P_2 = \frac{a+bc+dP_1}{2b}$')
ax.plot(P_NE, P_NE, 'o', color='#e040fb', markersize=12, zorder=5, label=f'Nash Eq. $P^* = {P_NE:.1f}$')
ax.axhline(y=c_cost, color='#66bb6a', linestyle='--', alpha=0.7, label=f'MC = {c_cost}')
ax.axvline(x=c_cost, color='#66bb6a', linestyle='--', alpha=0.7)
ax.set_xlabel(r'$P_1$', fontsize=13)
ax.set_ylabel(r'$P_2$', fontsize=13)
ax.set_title('Differentiated Bertrand\n(Strategic Complements)', fontsize=13)
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(c_cost, 40)
ax.set_ylim(c_cost, 40)
ax.grid(True, alpha=0.3)

# --- Right panel: Cournot ---
ax = axes[1]
Q_range = np.linspace(0, A, 200)

# Best responses: Q_i = (A - Q_j) / 2
BR1_Q = (A - Q_range) / 2
BR2_Q = (A - Q_range) / 2

Q_NE = A / 3

ax.plot(Q_range, BR1_Q, color='#4fc3f7', linewidth=2.5, label=r'$BR_1(Q_2)$: $Q_1 = \frac{A - Q_2}{2}$')
ax.plot(BR2_Q, Q_range, color='#ff8a65', linewidth=2.5, label=r'$BR_2(Q_1)$: $Q_2 = \frac{A - Q_1}{2}$')
ax.plot(Q_NE, Q_NE, 'o', color='#e040fb', markersize=12, zorder=5, label=f'Nash Eq. $Q^* = {Q_NE:.1f}$')
ax.set_xlabel(r'$Q_1$', fontsize=13)
ax.set_ylabel(r'$Q_2$', fontsize=13)
ax.set_title('Cournot Duopoly\n(Strategic Substitutes)', fontsize=13)
ax.legend(fontsize=9, loc='upper right')
ax.set_xlim(0, A)
ax.set_ylim(0, A)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bertrand_vs_cournot.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Simulation 2: Collusion Sustainability vs Number of Firms

```python
import numpy as np
import matplotlib.pyplot as plt

# Symmetric Cournot oligopoly with linear demand
# With n firms, collusion gives each firm monopoly_profit / n
# Deviation payoff: best response to (n-1) firms producing collusive quantity
# Competitive payoff: Cournot-Nash per-firm profit

a, b, c_cost = 100, 2, 10
A = a - b * c_cost  # = 80

n_firms = np.arange(2, 21)
delta_critical = np.zeros_like(n_firms, dtype=float)

for idx, n in enumerate(n_firms):
    # Collusion: split monopoly output equally
    Q_monopoly = A / 2
    q_collude = Q_monopoly / n
    P_collude = (a - Q_monopoly) / b
    pi_collude = (P_collude - c_cost) * q_collude

    # Deviation: best response to (n-1) firms each producing q_collude
    # Q_others = (n-1) * q_collude
    Q_others = (n - 1) * q_collude
    q_deviate = (A - Q_others) / 2  # best response
    Q_total_dev = Q_others + q_deviate
    P_deviate = (a - Q_total_dev) / b
    pi_deviate = (P_deviate - c_cost) * q_deviate

    # Nash (Cournot): each firm produces A/(n+1)
    q_nash = A / (n + 1)
    P_nash = (a - n * q_nash) / b
    pi_nash = (P_nash - c_cost) * q_nash

    # Critical discount factor
    delta_critical[idx] = (pi_deviate - pi_collude) / (pi_deviate - pi_nash)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(n_firms, delta_critical, 'o-', color='#e040fb', linewidth=2.5, markersize=8)
ax.axhline(y=0.9, color='#ff8a65', linestyle='--', alpha=0.7, label=r'Typical $\delta = 0.9$ (quarterly)')
ax.axhline(y=0.99, color='#4fc3f7', linestyle='--', alpha=0.7, label=r'Typical $\delta = 0.99$ (monthly)')
ax.fill_between(n_firms, delta_critical, 1, alpha=0.15, color='#ef5350', label='Collusion unsustainable')
ax.fill_between(n_firms, 0, delta_critical, alpha=0.15, color='#66bb6a', label='Collusion sustainable')
ax.set_xlabel('Number of firms $n$', fontsize=13)
ax.set_ylabel(r'Critical discount factor $\delta^*$', fontsize=13)
ax.set_title('Collusion Sustainability: Critical Discount Factor vs Market Structure', fontsize=14)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_xlim(2, 20)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('collusion_sustainability.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Simulation 3: Cournot Convergence to Perfect Competition

```python
import numpy as np
import matplotlib.pyplot as plt

a, b, c_cost = 100, 2, 10
A = a - b * c_cost

n_range = np.arange(1, 51)

# Cournot equilibrium price: P = (a + n*b*c) / ((n+1)*b)
P_cournot = (a + n_range * b * c_cost) / ((n_range + 1) * b)

# Benchmarks
P_monopoly = (a + b * c_cost) / (2 * b)  # n=1
P_competitive = c_cost  # n -> infinity

# Per-firm profit
q_i = A / (n_range + 1)
pi_i = (P_cournot - c_cost) * q_i

# Lerner index
L = (P_cournot - c_cost) / P_cournot

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Price convergence
ax = axes[0]
ax.plot(n_range, P_cournot, color='#4fc3f7', linewidth=2.5, label='Cournot $P^*$')
ax.axhline(y=P_monopoly, color='#e040fb', linestyle='--', linewidth=1.5, label=f'Monopoly $P_m = {P_monopoly:.1f}$')
ax.axhline(y=P_competitive, color='#66bb6a', linestyle='--', linewidth=1.5, label=f'Competitive $P_c = {P_competitive}$')
ax.set_xlabel('Number of firms $n$', fontsize=12)
ax.set_ylabel(r'Equilibrium Price $P^*$', fontsize=12)
ax.set_title('Price Convergence to Competition', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Lerner index
ax = axes[1]
ax.plot(n_range, L, color='#ff8a65', linewidth=2.5)
ax.set_xlabel('Number of firms $n$', fontsize=12)
ax.set_ylabel(r'Lerner Index $L = \frac{P - MC}{P}$', fontsize=12)
ax.set_title('Markup Erosion with Competition', fontsize=13)
ax.grid(True, alpha=0.3)

# Per-firm profit
ax = axes[2]
ax.plot(n_range, pi_i, color='#e040fb', linewidth=2.5)
ax.set_xlabel('Number of firms $n$', fontsize=12)
ax.set_ylabel(r'Per-firm profit $\pi_i$', fontsize=12)
ax.set_title('Profit Erosion with Competition', fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cournot_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## From Theory to Real Markets

Real markets do not neatly fit into any single model. They blend elements of all the frameworks we have discussed:

**Bertrand-like features** appear when firms set prices explicitly and consumers can easily compare (e-commerce, airline booking sites, gas stations). In these settings, differentiation and switching costs are the primary sources of markup. Without them, Bertrand competition drives prices toward cost.

**Cournot-like features** appear when firms commit to capacity or production levels and the market price adjusts. Semiconductor fabs, oil producers, and airlines (which set seat capacity months in advance) behave more like Cournot competitors. The HHI-markup relationship provides a useful rule of thumb for these industries.

**Stackelberg dynamics** arise when one firm has a credible first-mover advantage --- a dominant market share, a technological lead, or a reputation for aggressive competition. Amazon's pricing strategy in many categories follows Stackelberg logic: it commits to low prices, and smaller retailers must accept the follower role.

**Repeated-game collusion** is pervasive in concentrated industries with stable participants and transparent pricing. It does not require a smoke-filled room --- tacit collusion through mutual understanding of punishment strategies is sufficient and legal (though antitrust authorities watch for "facilitating practices" that make coordination easier).

**Algorithmic pricing** is rapidly becoming the norm, and it creates new equilibrium dynamics that combine elements of all the above models with the added complexity of machine learning and reinforcement learning.

The game-theoretic framework gives us a unified language for analyzing all of these patterns. The central lessons are:

1. **Markup depends on market structure.** The Cournot-Nash formula \(L = 1/(n|\varepsilon|)\) gives the first-order approximation.
2. **Differentiation creates market power.** Without it, Bertrand competition destroys margins even with just two firms.
3. **Repeated interaction enables collusion.** The folk theorem tells us that patience and repeated play can sustain almost any outcome, including collusion.
4. **Sequential moves create asymmetry.** First-mover advantage is real and quantifiable.
5. **Platforms are fundamentally different.** Two-sided network effects mean optimal prices on one side can be zero or negative.

But all of this analysis assumes we **know the demand function**. Every formula in this post --- the Bertrand equilibrium, the Cournot markup, the collusion threshold --- takes the demand curve as given. In practice, we must **estimate** it from data. And estimating demand is fiendishly difficult, because the prices we observe in data are themselves equilibrium outcomes of the very game we are trying to model. This is the identification problem, and in Part 4, we tackle it head-on using causal inference and machine learning.
