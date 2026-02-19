---
layout: post
title: "CAC, LTV, and Churn: The Complete SaaS Metrics Handbook for AI Video Platforms"
date: 2026-01-11
category: market
---

Every AI video platform founder I talk to can tell me their generation quality, their latency, their model lineup. Almost none can tell me their net revenue retention rate, their LTV:CAC ratio by acquisition channel, or their cohort-level payback period. This is a problem, because the companies that win in AI video won't be the ones with the best models --- they'll be the ones that understand their unit economics deeply enough to scale profitably.

This post is the complete SaaS metrics reference for AI video platforms. Every formula, every benchmark, every worked example. Bookmark it.

---

## Table of Contents

1. [Churn: The Silent Killer](#1-churn-the-silent-killer)
2. [Customer Lifetime Value (LTV)](#2-customer-lifetime-value-ltv)
3. [Customer Acquisition Cost (CAC)](#3-customer-acquisition-cost-cac)
4. [LTV:CAC Ratio and Payback Period](#4-ltvcac-ratio-and-payback-period)
5. [The SaaS Quick Ratio](#5-the-saas-quick-ratio)
6. [AI-Video-Specific Metrics](#6-ai-video-specific-metrics)
7. [Cohort Analysis](#7-cohort-analysis)
8. [Survival Analysis and Predicting User Lifetime](#8-survival-analysis-and-predicting-user-lifetime)
9. [Putting It All Together: The Metrics Dashboard](#9-putting-it-all-together-the-metrics-dashboard)

---

## 1. Churn: The Silent Killer

Churn is the single most important metric for any subscription business. In AI video, it's especially brutal because most platforms are fighting against a novelty curve: users sign up excited, generate a dozen videos, realize the outputs aren't production-ready for their use case, and cancel.

### 1.1 Logo Churn vs Revenue Churn

There are two fundamentally different kinds of churn, and confusing them will mislead you.

**Logo churn** (customer churn) counts the number of customers lost:

$$
\text{Logo Churn Rate} = \frac{\text{Customers lost during period}}{\text{Customers at start of period}}
$$

**Revenue churn** (MRR churn) counts the dollars lost:

$$
\text{Gross Revenue Churn Rate} = \frac{\text{MRR lost from churned + contracted customers}}{\text{MRR at start of period}}
$$

Why the distinction matters: if your $10/month users churn at 15% monthly but your $100/month users churn at 3%, your logo churn might be 12% but your revenue churn might be 5%. The business is healthier than logo churn suggests.

For AI video platforms specifically, the pattern is even more pronounced. Free-to-paid conversion users (who typically land on the cheapest tier) churn at 2-4x the rate of users who start on mid-tier plans. Your cheapest plan is your leakiest bucket.

### 1.2 Net Revenue Retention (NRR)

Net revenue retention is the single best health metric for a SaaS business. It answers: "If I acquired zero new customers, would my revenue grow or shrink?"

$$
\text{NRR} = \frac{\text{Starting MRR} + \text{Expansion MRR} - \text{Churned MRR} - \text{Contraction MRR}}{\text{Starting MRR}} \times 100\%
$$

The components:

- **Starting MRR**: Revenue from existing customers at period start
- **Expansion MRR**: Revenue gained from existing customers upgrading or purchasing add-ons
- **Churned MRR**: Revenue lost from customers who left entirely
- **Contraction MRR**: Revenue lost from customers who downgraded

**Benchmarks:**

| NRR Range | Interpretation | AI Video Context |
|---|---|---|
| < 80% | Leaky bucket, unsustainable | Typical of consumer-only video tools |
| 80-100% | Stable but not growing | Most AI video platforms today |
| 100-120% | Healthy expansion | Platforms with enterprise tiers + credit upsells |
| > 120% | Excellent, "negative churn" | Top SaaS companies (rare in AI video so far) |

**Worked example for an AI video platform:**

Starting MRR: $100,000 from 2,000 customers. During the month:
- 160 customers churned (8% logo churn), losing $4,800 MRR (mostly $20/mo and $30/mo plans)
- 40 customers downgraded from Pro (\(50/mo) to Basic (\)20/mo), losing $1,200 MRR
- 100 customers upgraded from Basic (\(20/mo) to Pro (\)50/mo), gaining $3,000 MRR
- 200 customers purchased additional credit packs ($15 average), adding $3,000 one-time MRR

$$
\text{NRR} = \frac{100{,}000 + 3{,}000 + 3{,}000 - 4{,}800 - 1{,}200}{100{,}000} \times 100\% = 100.0\%
$$

Exactly 100%. This platform is running in place. Expansion revenue exactly offsets churn and contraction. Not terrible, but not a path to growth without continuously acquiring new customers.

### 1.3 Monthly vs Annual Churn

Monthly and annual churn are not interchangeable. A common mistake is to multiply monthly churn by 12 to get annual churn. This is wrong.

The correct conversion:

$$
\text{Annual Churn} = 1 - (1 - \text{Monthly Churn})^{12}
$$

At 5% monthly churn:

$$
\text{Annual Churn} = 1 - (1 - 0.05)^{12} = 1 - 0.95^{12} = 1 - 0.5404 = 45.96\%
$$

Not \(5\% \times 12 = 60\%\). The compounding matters.

| Monthly Churn | Naive Annual (x12) | Actual Annual | Average Customer Lifetime |
|---|---|---|---|
| 2% | 24% | 21.5% | 50 months |
| 5% | 60% | 46.0% | 20 months |
| 8% | 96% | 63.2% | 12.5 months |
| 10% | 120% (???) | 71.8% | 10 months |
| 15% | 180% (???) | 85.9% | 6.7 months |

The "naive annual" column shows why multiplication is absurd --- you can't lose 120% of your customers.

Average customer lifetime in months (assuming constant churn rate):

$$
\bar{L} = \frac{1}{\text{Monthly Churn Rate}}
$$

At 8% monthly churn, the average customer stays \(1/0.08 = 12.5\) months. This is a crucial input to LTV calculations.

### 1.4 Retention Curve

A retention curve plots the percentage of a cohort that remains active over time. It's the most honest view of your product's stickiness.

<svg viewBox="0 0 700 420" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 700px; font-family: 'Segoe UI', system-ui, sans-serif;">
  <!-- Grid -->
  <defs>
    <pattern id="grid1" width="50" height="30" patternUnits="userSpaceOnUse">
      <path d="M 50 0 L 0 0 0 30" fill="none" stroke="#e0e0e0" stroke-width="0.5"/>
    </pattern>
  </defs>
  <rect x="80" y="30" width="600" height="330" fill="url(#grid1)"/>

  <!-- Axes -->
  <line x1="80" y1="360" x2="680" y2="360" stroke="#333" stroke-width="2"/>
  <line x1="80" y1="30" x2="80" y2="360" stroke="#333" stroke-width="2"/>

  <!-- Y-axis labels -->
  <text x="70" y="365" text-anchor="end" font-size="11" fill="#333">0%</text>
  <text x="70" y="299" text-anchor="end" font-size="11" fill="#333">20%</text>
  <text x="70" y="233" text-anchor="end" font-size="11" fill="#333">40%</text>
  <text x="70" y="167" text-anchor="end" font-size="11" fill="#333">60%</text>
  <text x="70" y="101" text-anchor="end" font-size="11" fill="#333">80%</text>
  <text x="70" y="35" text-anchor="end" font-size="11" fill="#333">100%</text>

  <!-- X-axis labels -->
  <text x="80" y="378" text-anchor="middle" font-size="11" fill="#333">0</text>
  <text x="130" y="378" text-anchor="middle" font-size="11" fill="#333">1</text>
  <text x="180" y="378" text-anchor="middle" font-size="11" fill="#333">2</text>
  <text x="230" y="378" text-anchor="middle" font-size="11" fill="#333">3</text>
  <text x="280" y="378" text-anchor="middle" font-size="11" fill="#333">4</text>
  <text x="330" y="378" text-anchor="middle" font-size="11" fill="#333">5</text>
  <text x="380" y="378" text-anchor="middle" font-size="11" fill="#333">6</text>
  <text x="430" y="378" text-anchor="middle" font-size="11" fill="#333">7</text>
  <text x="480" y="378" text-anchor="middle" font-size="11" fill="#333">8</text>
  <text x="530" y="378" text-anchor="middle" font-size="11" fill="#333">9</text>
  <text x="580" y="378" text-anchor="middle" font-size="11" fill="#333">10</text>
  <text x="630" y="378" text-anchor="middle" font-size="11" fill="#333">11</text>
  <text x="680" y="378" text-anchor="middle" font-size="11" fill="#333">12</text>

  <!-- Axis titles -->
  <text x="380" y="405" text-anchor="middle" font-size="13" fill="#333" font-weight="bold">Month</text>
  <text x="25" y="195" text-anchor="middle" font-size="13" fill="#333" font-weight="bold" transform="rotate(-90, 25, 195)">Retention %</text>

  <!-- High churn (15% monthly) - red -->
  <polyline points="80,30 130,79.5 180,122.1 230,158.3 280,189.0 330,215.1 380,237.3 430,256.1 480,272.1 530,285.7 580,297.3 630,307.1 680,315.5" fill="none" stroke="#ef5350" stroke-width="2.5"/>

  <!-- Medium churn (8% monthly) - orange -->
  <polyline points="80,30 130,56.4 180,80.3 230,102.3 280,122.4 330,140.9 380,157.9 430,173.6 480,188.0 530,201.2 580,213.4 630,224.7 680,235.0" fill="none" stroke="#ffa726" stroke-width="2.5"/>

  <!-- Low churn (3% monthly) - green -->
  <polyline points="80,30 130,39.9 180,49.5 230,58.8 280,67.8 330,76.5 380,84.9 430,93.1 480,100.9 530,108.5 580,115.9 630,123.0 680,129.9" fill="none" stroke="#8bc34a" stroke-width="2.5"/>

  <!-- Excellent (1% monthly) - blue -->
  <polyline points="80,30 130,33.3 180,36.6 230,39.8 280,43.0 330,46.2 380,49.3 430,52.4 480,55.4 530,58.4 580,61.3 630,64.2 680,67.1" fill="none" stroke="#4fc3f7" stroke-width="2.5"/>

  <!-- Legend -->
  <rect x="450" y="50" width="220" height="100" fill="white" stroke="#ccc" rx="4"/>
  <line x1="460" y1="68" x2="490" y2="68" stroke="#ef5350" stroke-width="2.5"/>
  <text x="495" y="72" font-size="11" fill="#333">15%/mo churn (consumer)</text>
  <line x1="460" y1="88" x2="490" y2="88" stroke="#ffa726" stroke-width="2.5"/>
  <text x="495" y="92" font-size="11" fill="#333">8%/mo churn (prosumer)</text>
  <line x1="460" y1="108" x2="490" y2="108" stroke="#8bc34a" stroke-width="2.5"/>
  <text x="495" y="112" font-size="11" fill="#333">3%/mo churn (SMB)</text>
  <line x1="460" y1="128" x2="490" y2="128" stroke="#4fc3f7" stroke-width="2.5"/>
  <text x="495" y="132" font-size="11" fill="#333">1%/mo churn (enterprise)</text>

  <!-- Title -->
  <text x="380" y="18" text-anchor="middle" font-size="15" fill="#333" font-weight="bold">Retention Curves by Churn Rate</text>
</svg>

**Figure 1.** Retention curves for different monthly churn rates typical of AI video platform segments. At 15% monthly churn (typical consumer), fewer than 15% of users remain after 12 months. At 1% monthly churn (enterprise contracts), nearly 89% remain.

The shape of the retention curve tells you as much as the level. A curve that flattens after month 2-3 (steep initial drop, then stabilization) indicates you have a "core user" base that finds real value --- your problem is activation, not retention. A curve that decays linearly forever indicates no one finds sticky value.

---

## 2. Customer Lifetime Value (LTV)

LTV is the total revenue you expect to collect from a customer over their entire relationship with your product. It determines how much you can afford to spend on acquisition.

### 2.1 Simple LTV Formula

The simplest formulation, assuming constant ARPU and constant churn:

$$
\text{LTV}_{\text{simple}} = \frac{\text{ARPU}}{\text{Monthly Churn Rate}}
$$

where ARPU is average revenue per user per month. If ARPU = $25/month and monthly churn = 8%:

$$
\text{LTV}_{\text{simple}} = \frac{25}{0.08} = \$312.50
$$

This formula assumes:
- ARPU is constant (no upgrades, no credit packs, no price changes)
- Churn is constant (same probability of leaving every month)
- No discount rate (a dollar next year is worth a dollar today)

All three assumptions are wrong for AI video platforms. Let's fix them.

### 2.2 DCF-Adjusted LTV

A more rigorous approach uses the discounted cash flow method, accounting for the time value of money and variable survival probabilities:

$$
\text{LTV} = \sum_{t=1}^{T} \frac{\text{ARPU}_t \times S(t)}{(1 + d)^t}
$$

where:

- \(\text{ARPU}_t\) is the average revenue per user in month \(t\) (can change due to upgrades, credits, price changes)
- \(S(t)\) is the survival probability at month \(t\) --- the probability that a customer acquired at time 0 is still active at time \(t\)
- \(d\) is the monthly discount rate (annual discount rate / 12, approximately)
- \(T\) is the time horizon (typically 36-60 months)

For constant churn rate \(c\), the survival function simplifies to:

$$
S(t) = (1 - c)^t
$$

And with constant ARPU and an infinite horizon, the DCF LTV converges to:

$$
\text{LTV}_{\text{DCF}} = \sum_{t=1}^{\infty} \frac{\text{ARPU} \times (1-c)^t}{(1+d)^t} = \frac{\text{ARPU} \times (1-c)}{c + d}
$$

This is the formula you should actually use. It's still clean but accounts for the time value of money.

**Worked example:** ARPU = $25/month, monthly churn = 8%, annual discount rate = 10% (monthly \(d \approx 0.00833\)):

$$
\text{LTV}_{\text{DCF}} = \frac{25 \times 0.92}{0.08 + 0.00833} = \frac{23.0}{0.08833} = \$260.36
$$

Compare to the simple formula's $312.50. The discount rate knocks 17% off the LTV. This matters when you're deciding how much to spend on acquisition.

### 2.3 LTV with Expansion Revenue

AI video platforms have a natural expansion mechanism: credit packs. Users who hit their tier limits buy additional credits. Users who love the product upgrade to higher tiers. This means ARPU grows over time for surviving customers.

If ARPU grows at rate \(g\) per month for surviving users:

$$
\text{LTV}_{\text{expansion}} = \sum_{t=1}^{\infty} \frac{\text{ARPU}_0 \times (1+g)^t \times (1-c)^t}{(1+d)^t} = \frac{\text{ARPU}_0 \times (1+g)(1-c)}{c + d - g + cg}
$$

For the case where \(g < c + d\) (expansion rate less than churn + discount, which it should be for convergence):

**Worked example:** Same as above, but with 2% monthly ARPU expansion (\(g = 0.02\)):

$$
\text{LTV}_{\text{expansion}} = \frac{25 \times 1.02 \times 0.92}{0.08 + 0.00833 - 0.02 + 0.0016} = \frac{23.46}{0.06993} = \$335.47
$$

The expansion revenue adds 29% to the LTV compared to the flat-ARPU DCF calculation. This is why upselling credit packs and higher tiers is so important.

### 2.4 LTV Sensitivity Analysis

LTV is extremely sensitive to churn rate. Small improvements in retention create large increases in customer value. Let's model this across the range of churn rates seen in AI video platforms.

**Assumptions:** ARPU = $25/month, annual discount rate = 10%, no expansion.

| Monthly Churn | Avg Lifetime (mo) | Simple LTV | DCF LTV | Annual LTV Equiv |
|---|---|---|---|---|
| 2% | 50.0 | $1,250.00 | $880.28 | $211.27 |
| 3% | 33.3 | $833.33 | $632.18 | $227.58 |
| 5% | 20.0 | $500.00 | $396.55 | $238.93 |
| 8% | 12.5 | $312.50 | $260.36 | $250.02 |
| 10% | 10.0 | $250.00 | $212.77 | $255.32 |
| 12% | 8.3 | $208.33 | $179.83 | $259.91 |
| 15% | 6.7 | $166.67 | $145.89 | $262.60 |

<svg viewBox="0 0 700 420" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 700px; font-family: 'Segoe UI', system-ui, sans-serif;">
  <!-- Grid -->
  <defs>
    <pattern id="grid2" width="50" height="35" patternUnits="userSpaceOnUse">
      <path d="M 50 0 L 0 0 0 35" fill="none" stroke="#e0e0e0" stroke-width="0.5"/>
    </pattern>
  </defs>
  <rect x="90" y="30" width="590" height="330" fill="url(#grid2)"/>

  <!-- Axes -->
  <line x1="90" y1="360" x2="680" y2="360" stroke="#333" stroke-width="2"/>
  <line x1="90" y1="30" x2="90" y2="360" stroke="#333" stroke-width="2"/>

  <!-- Y-axis: $0 to $1400 -->
  <text x="80" y="365" text-anchor="end" font-size="11" fill="#333">$0</text>
  <text x="80" y="318" text-anchor="end" font-size="11" fill="#333">$200</text>
  <text x="80" y="271" text-anchor="end" font-size="11" fill="#333">$400</text>
  <text x="80" y="224" text-anchor="end" font-size="11" fill="#333">$600</text>
  <text x="80" y="177" text-anchor="end" font-size="11" fill="#333">$800</text>
  <text x="80" y="130" text-anchor="end" font-size="11" fill="#333">$1000</text>
  <text x="80" y="83" text-anchor="end" font-size="11" fill="#333">$1200</text>

  <!-- X-axis labels -->
  <text x="135" y="378" text-anchor="middle" font-size="11" fill="#333">2%</text>
  <text x="225" y="378" text-anchor="middle" font-size="11" fill="#333">5%</text>
  <text x="315" y="378" text-anchor="middle" font-size="11" fill="#333">8%</text>
  <text x="405" y="378" text-anchor="middle" font-size="11" fill="#333">10%</text>
  <text x="495" y="378" text-anchor="middle" font-size="11" fill="#333">12%</text>
  <text x="585" y="378" text-anchor="middle" font-size="11" fill="#333">15%</text>

  <!-- Axis titles -->
  <text x="385" y="405" text-anchor="middle" font-size="13" fill="#333" font-weight="bold">Monthly Churn Rate</text>
  <text x="25" y="195" text-anchor="middle" font-size="13" fill="#333" font-weight="bold" transform="rotate(-90, 25, 195)">LTV ($)</text>

  <!-- Simple LTV bars -->
  <rect x="115" y="63.4" width="18" height="296.6" fill="#4fc3f7" opacity="0.8"/>
  <rect x="205" y="242" width="18" height="118" fill="#4fc3f7" opacity="0.8"/>
  <rect x="295" y="286" width="18" height="74" fill="#4fc3f7" opacity="0.8"/>
  <rect x="385" y="301" width="18" height="59" fill="#4fc3f7" opacity="0.8"/>
  <rect x="475" y="311" width="18" height="49" fill="#4fc3f7" opacity="0.8"/>
  <rect x="565" y="321" width="18" height="39" fill="#4fc3f7" opacity="0.8"/>

  <!-- DCF LTV bars -->
  <rect x="138" y="152" width="18" height="208" fill="#8bc34a" opacity="0.8"/>
  <rect x="228" y="267" width="18" height="93" fill="#8bc34a" opacity="0.8"/>
  <rect x="318" y="299" width="18" height="61" fill="#8bc34a" opacity="0.8"/>
  <rect x="408" y="310" width="18" height="50" fill="#8bc34a" opacity="0.8"/>
  <rect x="498" y="318" width="18" height="42" fill="#8bc34a" opacity="0.8"/>
  <rect x="588" y="326" width="18" height="34" fill="#8bc34a" opacity="0.8"/>

  <!-- Legend -->
  <rect x="460" y="45" width="200" height="55" fill="white" stroke="#ccc" rx="4"/>
  <rect x="470" y="58" width="14" height="14" fill="#4fc3f7" opacity="0.8"/>
  <text x="490" y="70" font-size="11" fill="#333">Simple LTV</text>
  <rect x="470" y="78" width="14" height="14" fill="#8bc34a" opacity="0.8"/>
  <text x="490" y="90" font-size="11" fill="#333">DCF-adjusted LTV</text>

  <!-- Title -->
  <text x="385" y="18" text-anchor="middle" font-size="15" fill="#333" font-weight="bold">LTV Sensitivity to Monthly Churn Rate (ARPU = $25)</text>
</svg>

**Figure 2.** LTV sensitivity to monthly churn rate. Reducing churn from 8% to 5% increases DCF LTV by 52% ($260 to $397). Reducing from 8% to 3% increases it by 143%. At any realistic ARPU, churn reduction is the highest-leverage activity for increasing customer value.

The key takeaway: **a 1 percentage point reduction in monthly churn from 8% to 7% increases LTV by approximately 15%.** That's worth more than almost any feature you could build.

### 2.5 LTV by Segment

In practice, you shouldn't calculate a single LTV. Different user segments have wildly different values:

| Segment | ARPU | Monthly Churn | DCF LTV | Notes |
|---|---|---|---|---|
| Free-to-paid converts | $15 | 15% | $82 | High churn, low ARPU |
| Organic signups (paid) | $25 | 8% | $260 | Your "average" user |
| Prosumer creators | $50 | 5% | $793 | Power users, regular generators |
| Small business | $100 | 4% | $1,913 | Use for marketing content |
| Enterprise (annual) | $500 | 1.5%/mo | $19,619 | Low churn, high ARPU |

The spread is enormous --- an enterprise customer is worth 240x a free-to-paid convert. This should directly inform your acquisition strategy: every dollar spent acquiring enterprise users is worth far more than a dollar spent on consumer acquisition.

---

## 3. Customer Acquisition Cost (CAC)

CAC is the total cost to acquire one paying customer. Simple concept, surprisingly hard to calculate correctly.

### 3.1 Basic CAC Formula

$$
\text{CAC} = \frac{\text{Total Sales \& Marketing Spend in Period}}{\text{New Customers Acquired in Period}}
$$

The numerator should include:
- Paid advertising spend (Google, Meta, TikTok, YouTube, Twitter/X)
- Content marketing costs (writers, designers, tools)
- SEO costs (tools, contractors)
- Sales team fully loaded cost (salary + benefits + tools + commissions)
- Marketing team fully loaded cost
- Marketing software (analytics, email, CRM)
- Event and conference costs
- Referral program costs
- Free trial infrastructure cost (GPU compute for free generations)

That last one is specific to AI video and often forgotten. If you offer free generations to convert users, the compute cost of those free generations is part of CAC. At $0.05-0.30 per generation and 20-50 free generations per trial user, that's $1-15 per trial user in compute cost before they've paid anything.

### 3.2 CAC by Channel

Different channels have wildly different acquisition costs. Here's what I've seen across AI video platforms:

| Channel | Typical CAC | Conversion Rate | Pros | Cons |
|---|---|---|---|---|
| Organic search / SEO | $3-10 | 3-8% visitor-to-trial | Low cost, compounds | Slow to build, competitive |
| Content marketing | $8-20 | 2-5% | Builds authority, SEO benefit | Slow, requires consistency |
| Paid search (Google) | $15-40 | 5-12% click-to-trial | High intent, scalable | Expensive, competitive keywords |
| Paid social (Meta) | $8-25 | 1-4% | Good targeting, visual format | Lower intent, ad fatigue |
| Paid social (TikTok) | $5-15 | 2-6% | AI video demos perform well | Younger demo, lower ARPU |
| YouTube ads | $12-30 | 3-7% | Video format perfect for demos | Production cost, creative fatigue |
| Influencer / creator | $10-35 | 3-8% | Trust signal, authentic demos | Hard to scale, variable quality |
| Partnerships / integrations | $5-15 | Varies | High quality leads | Slow to establish |
| Product-led (viral / referral) | $1-5 | 10-25% | Highest quality, cheapest | Requires product virality |

### 3.3 Attribution Models

The fundamental question: when a user sees your TikTok ad, then reads your blog post, then clicks a Google ad, then signs up --- which channel gets the credit?

**First-touch attribution:**

$$
\text{Credit}_{c} = \begin{cases} 1 & \text{if } c = \text{first touchpoint} \\ 0 & \text{otherwise} \end{cases}
$$

All credit goes to the first interaction (TikTok ad). Simple but overvalues awareness channels.

**Last-touch attribution:**

$$
\text{Credit}_{c} = \begin{cases} 1 & \text{if } c = \text{last touchpoint before conversion} \\ 0 & \text{otherwise} \end{cases}
$$

All credit goes to the last interaction (Google ad). Simple but overvalues bottom-of-funnel channels.

**Linear attribution:**

$$
\text{Credit}_{c} = \frac{1}{n}
$$

where \(n\) is the total number of touchpoints. Each interaction gets equal credit ($1/3$ each). Fair but probably wrong --- the last click matters more than the first impression.

**Time-decay attribution:**

$$
\text{Credit}_{c_i} = \frac{e^{-\lambda(T - t_i)}}{\sum_{j=1}^{n} e^{-\lambda(T - t_j)}}
$$

where \(T\) is the conversion time, \(t_i\) is the time of touchpoint \(i\), and \(\lambda\) is the decay rate. More recent touchpoints get more credit. This is the most mathematically defensible model for most SaaS businesses.

**Worked example (time-decay):** Three touchpoints: TikTok ad (day 1), blog post (day 10), Google ad (day 14), conversion (day 14). With \(\lambda = 0.1\):

- TikTok: \(e^{-0.1 \times 13} = e^{-1.3} = 0.2725\)
- Blog: \(e^{-0.1 \times 4} = e^{-0.4} = 0.6703\)
- Google: \(e^{-0.1 \times 0} = e^{0} = 1.0\)

Total: \(0.2725 + 0.6703 + 1.0 = 1.9428\)

Credits: TikTok = 14%, Blog = 34.5%, Google = 51.5%

This feels about right --- the Google ad that directly preceded conversion gets the most credit, but the earlier touchpoints still contributed.

### 3.4 Blended CAC vs Paid CAC

**Blended CAC** includes all customers, including organic (who cost "$0" to acquire directly). **Paid CAC** only counts customers from paid channels against paid spend.

$$
\text{Blended CAC} = \frac{\text{Total S\&M Spend}}{\text{All New Customers}}
$$

$$
\text{Paid CAC} = \frac{\text{Paid Channel Spend}}{\text{Customers from Paid Channels}}
$$

If 60% of your customers come from organic channels and you spend $50K/month on paid to acquire 500 paid customers and 750 organic customers:

$$
\text{Blended CAC} = \frac{50{,}000}{1{,}250} = \$40
$$

$$
\text{Paid CAC} = \frac{50{,}000}{500} = \$100
$$

Both numbers matter. Blended CAC tells you overall efficiency. Paid CAC tells you marginal efficiency --- what the next dollar of paid spend returns.

---

## 4. LTV:CAC Ratio and Payback Period

### 4.1 The Magic 3x Ratio

The LTV:CAC ratio answers: "For every dollar I spend acquiring a customer, how many dollars do I get back?"

$$
\text{LTV:CAC Ratio} = \frac{\text{LTV}}{\text{CAC}}
$$

**Benchmarks:**

| Ratio | Interpretation | Action |
|---|---|---|
| < 1.0 | Losing money on every customer | Stop spending. Fix product or economics. |
| 1.0 - 2.0 | Unprofitable (after OpEx) | Reduce CAC or reduce churn |
| 2.0 - 3.0 | Approaching profitability | Optimize, but careful about scaling spend |
| 3.0 - 5.0 | Healthy | Scale spend confidently |
| > 5.0 | Under-investing in growth | Spend more on acquisition. You're leaving growth on the table. |

The >3x target comes from the fact that a SaaS business has costs beyond COGS and S&M: engineering, G&A, support. A 3x ratio means roughly $1 goes to acquisition, $1 goes to serving the customer and running the business, and $1 is profit.

For AI video platforms, the 3x target is harder to hit because COGS (GPU compute for generations) is high. With 60% gross margins, you need a higher LTV:CAC ratio to achieve the same unit profitability:

$$
\text{Gross-Profit-Adjusted LTV:CAC} = \frac{\text{LTV} \times \text{Gross Margin \%}}{\text{CAC}}
$$

At 60% gross margin, you need a 5x LTV:CAC ratio to get the same effective 3x on gross profit.

### 4.2 Payback Period

The payback period is how many months it takes to recover your CAC:

$$
\text{Payback Period} = \frac{\text{CAC}}{\text{ARPU} \times \text{Gross Margin \%}}
$$

**Worked example:** CAC = $80, ARPU = $25, Gross Margin = 60%:

$$
\text{Payback Period} = \frac{80}{25 \times 0.60} = \frac{80}{15} = 5.33 \text{ months}
$$

**Benchmarks:**

| Payback Period | Interpretation |
|---|---|
| < 6 months | Excellent --- fast capital recovery |
| 6-12 months | Good --- standard for B2B SaaS |
| 12-18 months | Concerning --- need strong retention to work |
| > 18 months | Dangerous --- high capital requirements |

A 5.33-month payback with 8% monthly churn means \(S(5.33) = 0.92^{5.33} = 0.643\) --- 64.3% of customers are still around to "pay back" their acquisition cost. This is fine. But at 15% monthly churn, \(S(5.33) = 0.85^{5.33} = 0.421\) --- only 42% make it to payback. That's unsustainable.

### 4.3 Payback-Adjusted LTV:CAC

A more nuanced metric combines LTV:CAC with the payback period to account for the risk that customers churn before paying back:

$$
\text{Expected Payback} = \text{CAC} \times S\left(\frac{\text{CAC}}{\text{ARPU} \times \text{GM}}\right)
$$

If \(S(t_{\text{payback}}) > 0.5\), more than half your customers reach payback, which is typically the minimum viable threshold.

---

## 5. The SaaS Quick Ratio

The Quick Ratio measures growth efficiency --- how much new revenue you're adding relative to how much you're losing:

$$
\text{Quick Ratio} = \frac{\text{New MRR} + \text{Expansion MRR}}{\text{Churned MRR} + \text{Contraction MRR}}
$$

**Benchmarks:**

| Quick Ratio | Interpretation | Growth Character |
|---|---|---|
| < 1.0 | Shrinking | MRR is declining --- fix churn immediately |
| 1.0 - 2.0 | Growing slowly | Growth is inefficient, leaky bucket |
| 2.0 - 4.0 | Healthy growth | Strong growth with manageable churn |
| > 4.0 | Excellent | Rapid growth, strong retention |

**Worked example for an AI video platform:**

- New MRR (new subscribers this month): $12,000
- Expansion MRR (upgrades + credit pack upsells): $4,000
- Churned MRR (cancelled subscribers): $6,000
- Contraction MRR (downgrades): $1,500

$$
\text{Quick Ratio} = \frac{12{,}000 + 4{,}000}{6{,}000 + 1{,}500} = \frac{16{,}000}{7{,}500} = 2.13
$$

A Quick Ratio of 2.13 means for every dollar of MRR lost, you're adding $2.13. This is decent but not exceptional. The platform is growing, but it's working hard to replace churned revenue.

**Decomposing the Quick Ratio:**

You can break it down further to understand the growth sources:

$$
\text{QR} = \underbrace{\frac{\text{New MRR}}{\text{Lost MRR}}}_{\text{Acquisition Efficiency}} + \underbrace{\frac{\text{Expansion MRR}}{\text{Lost MRR}}}_{\text{Expansion Efficiency}}
$$

In our example: \(12{,}000/7{,}500 + 4{,}000/7{,}500 = 1.60 + 0.53\)

Acquisition accounts for 75% of growth, expansion for 25%. A mature SaaS business would want expansion contributing 40%+ of growth. This platform is still too dependent on new customer acquisition.

---

## 6. AI-Video-Specific Metrics

Beyond standard SaaS metrics, AI video platforms need to track metrics unique to generative AI products.

### 6.1 Cost Per Generation

$$
\text{Cost Per Generation} = \frac{\text{Total Compute Cost}}{\text{Total Generations}}
$$

But this average hides enormous variance. A 5-second 480p generation on a fast model might cost $0.02. A 10-second 1080p generation on Veo 3 might cost $0.50. You need to track cost per generation by:

- **Model**: Kling costs differ from Runway costs differ from Veo costs
- **Resolution**: 480p vs 720p vs 1080p vs 4K
- **Duration**: 3s vs 5s vs 10s vs 30s
- **Mode**: Standard vs high-quality vs turbo

$$
\text{Cost}_{m,r,d,q} = \text{API Price}_{m}(r, d, q) + \text{Infrastructure Overhead}
$$

Infrastructure overhead includes: queue management, storage for outputs, CDN delivery, encoding/transcoding, and a share of platform infrastructure.

### 6.2 Margin Per Generation

$$
\text{Margin Per Generation} = \text{Revenue Per Generation} - \text{Cost Per Generation}
$$

Revenue per generation depends on your pricing model. If you sell credits at $0.10/credit and a generation costs 5 credits:

$$
\text{Revenue Per Generation} = 5 \times \$0.10 = \$0.50
$$

If the API cost is $0.15 and infrastructure overhead is $0.03:

$$
\text{Margin Per Generation} = \$0.50 - \$0.18 = \$0.32 \quad (64\% \text{ gross margin})
$$

Track this over time. As API prices drop (they have been, rapidly), your margin per generation expands --- or you pass savings to users via lower credit prices to drive volume.

### 6.3 Generations Per User Per Month (GPUPM)

$$
\text{GPUPM} = \frac{\text{Total Generations in Month}}{\text{Active Users in Month}}
$$

This is your engagement metric. More generations per user correlates with lower churn and higher LTV. Benchmarks from the platforms I've tracked:

| Segment | GPUPM | Interpretation |
|---|---|---|
| Casual users | 5-15 | Experimenting, not integrated into workflow |
| Regular users | 30-80 | Using for specific projects |
| Power users | 100-300 | Integrated into daily workflow |
| Enterprise/API | 500-5000+ | Automated pipelines |

A bimodal distribution (lots of users at 5-10 and a smaller cluster at 100+) is common and healthy. It means you have a power-user core. A single peak at 5-10 with a long tail to zero is concerning --- nobody is finding deep value.

### 6.4 Credit Utilization Rate

For credit-based pricing:

$$
\text{Credit Utilization} = \frac{\text{Credits Used}}{\text{Credits Available (Purchased + Included)}}
$$

| Utilization | Interpretation | Action |
|---|---|---|
| < 30% | Users not engaging | Activation problem. Improve onboarding. |
| 30-60% | Moderate engagement | Normal for casual users |
| 60-80% | Good engagement | Users finding value |
| 80-95% | High engagement | Users may need upsell prompts |
| > 95% | Maxed out | These users are candidates for plan upgrades |

Users with < 20% utilization are high churn risk. Users at > 90% utilization are upsell candidates. Build alerts for both.

### 6.5 Model Preference Distribution

Track which models users prefer (in a multi-model platform):

$$
\text{Model Share}_m = \frac{\text{Generations on Model } m}{\text{Total Generations}} \times 100\%
$$

This data is strategically critical. It tells you:
- Which API providers have leverage over you (high share = high dependency)
- Where to focus quality optimization
- Which models to prioritize for caching/optimization
- User aesthetic preferences

If 60% of generations go to one model, that provider has enormous leverage. Diversification is a strategic imperative.

### 6.6 Generation Success Rate

$$
\text{Success Rate} = \frac{\text{Generations Completed Successfully}}{\text{Generations Attempted}}
$$

Failures include: API timeouts, content policy rejections, model errors, queue timeouts. Target > 95%. Below 90% and you have a product quality problem that will drive churn.

### 6.7 Time to First Generation (TTFG)

$$
\text{TTFG} = t_{\text{first generation completed}} - t_{\text{account created}}
$$

The analog of "time to value" for AI video. Shorter TTFG correlates strongly with conversion and retention. Benchmark: under 5 minutes from signup to first completed generation.

---

## 7. Cohort Analysis

Cohort analysis is the most powerful analytical tool for understanding your business. It groups users by their signup month and tracks their behavior over time.

### 7.1 Building a Cohort Retention Table

A cohort retention table shows what percentage of each monthly cohort is still active in subsequent months:

| Cohort | Month 0 | Month 1 | Month 2 | Month 3 | Month 4 | Month 5 | Month 6 |
|---|---|---|---|---|---|---|---|
| Jul 2025 | 100% | 72% | 58% | 50% | 45% | 42% | 40% |
| Aug 2025 | 100% | 70% | 55% | 47% | 42% | 39% | -- |
| Sep 2025 | 100% | 74% | 62% | 55% | 50% | -- | -- |
| Oct 2025 | 100% | 78% | 66% | 59% | -- | -- | -- |
| Nov 2025 | 100% | 80% | 69% | -- | -- | -- | -- |
| Dec 2025 | 100% | 82% | -- | -- | -- | -- | -- |

Reading this table: Of all users who signed up in July 2025, 72% were still active after 1 month, 58% after 2 months, and 40% after 6 months.

The beautiful thing about cohort analysis: you can see improvement over time. The Sep-Dec cohorts are retaining better than Jul-Aug, which means something is working (product improvement, better onboarding, different acquisition channels).

<svg viewBox="0 0 720 400" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 720px; font-family: 'Segoe UI', system-ui, sans-serif;">
  <!-- Title -->
  <text x="360" y="22" text-anchor="middle" font-size="15" fill="#333" font-weight="bold">Cohort Retention Heatmap</text>

  <!-- Column headers -->
  <text x="185" y="52" text-anchor="middle" font-size="10" fill="#333" font-weight="bold">M0</text>
  <text x="255" y="52" text-anchor="middle" font-size="10" fill="#333" font-weight="bold">M1</text>
  <text x="325" y="52" text-anchor="middle" font-size="10" fill="#333" font-weight="bold">M2</text>
  <text x="395" y="52" text-anchor="middle" font-size="10" fill="#333" font-weight="bold">M3</text>
  <text x="465" y="52" text-anchor="middle" font-size="10" fill="#333" font-weight="bold">M4</text>
  <text x="535" y="52" text-anchor="middle" font-size="10" fill="#333" font-weight="bold">M5</text>
  <text x="605" y="52" text-anchor="middle" font-size="10" fill="#333" font-weight="bold">M6</text>

  <!-- Row labels -->
  <text x="100" y="88" text-anchor="middle" font-size="11" fill="#333">Jul 2025</text>
  <text x="100" y="133" text-anchor="middle" font-size="11" fill="#333">Aug 2025</text>
  <text x="100" y="178" text-anchor="middle" font-size="11" fill="#333">Sep 2025</text>
  <text x="100" y="223" text-anchor="middle" font-size="11" fill="#333">Oct 2025</text>
  <text x="100" y="268" text-anchor="middle" font-size="11" fill="#333">Nov 2025</text>
  <text x="100" y="313" text-anchor="middle" font-size="11" fill="#333">Dec 2025</text>

  <!-- Jul row - 100,72,58,50,45,42,40 -->
  <rect x="155" y="65" width="60" height="38" fill="#4fc3f7" rx="3"/>
  <text x="185" y="88" text-anchor="middle" font-size="11" fill="white" font-weight="bold">100%</text>
  <rect x="225" y="65" width="60" height="38" fill="#4fc3f7" opacity="0.72" rx="3"/>
  <text x="255" y="88" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">72%</text>
  <rect x="295" y="65" width="60" height="38" fill="#ffa726" opacity="0.8" rx="3"/>
  <text x="325" y="88" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">58%</text>
  <rect x="365" y="65" width="60" height="38" fill="#ffa726" opacity="0.7" rx="3"/>
  <text x="395" y="88" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">50%</text>
  <rect x="435" y="65" width="60" height="38" fill="#ef5350" opacity="0.6" rx="3"/>
  <text x="465" y="88" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">45%</text>
  <rect x="505" y="65" width="60" height="38" fill="#ef5350" opacity="0.55" rx="3"/>
  <text x="535" y="88" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">42%</text>
  <rect x="575" y="65" width="60" height="38" fill="#ef5350" opacity="0.5" rx="3"/>
  <text x="605" y="88" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">40%</text>

  <!-- Aug row - 100,70,55,47,42,39 -->
  <rect x="155" y="110" width="60" height="38" fill="#4fc3f7" rx="3"/>
  <text x="185" y="133" text-anchor="middle" font-size="11" fill="white" font-weight="bold">100%</text>
  <rect x="225" y="110" width="60" height="38" fill="#4fc3f7" opacity="0.70" rx="3"/>
  <text x="255" y="133" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">70%</text>
  <rect x="295" y="110" width="60" height="38" fill="#ffa726" opacity="0.75" rx="3"/>
  <text x="325" y="133" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">55%</text>
  <rect x="365" y="110" width="60" height="38" fill="#ef5350" opacity="0.62" rx="3"/>
  <text x="395" y="133" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">47%</text>
  <rect x="435" y="110" width="60" height="38" fill="#ef5350" opacity="0.55" rx="3"/>
  <text x="465" y="133" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">42%</text>
  <rect x="505" y="110" width="60" height="38" fill="#ef5350" opacity="0.5" rx="3"/>
  <text x="535" y="133" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">39%</text>

  <!-- Sep row - 100,74,62,55,50 -->
  <rect x="155" y="155" width="60" height="38" fill="#4fc3f7" rx="3"/>
  <text x="185" y="178" text-anchor="middle" font-size="11" fill="white" font-weight="bold">100%</text>
  <rect x="225" y="155" width="60" height="38" fill="#4fc3f7" opacity="0.74" rx="3"/>
  <text x="255" y="178" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">74%</text>
  <rect x="295" y="155" width="60" height="38" fill="#8bc34a" opacity="0.72" rx="3"/>
  <text x="325" y="178" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">62%</text>
  <rect x="365" y="155" width="60" height="38" fill="#ffa726" opacity="0.75" rx="3"/>
  <text x="395" y="178" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">55%</text>
  <rect x="435" y="155" width="60" height="38" fill="#ffa726" opacity="0.7" rx="3"/>
  <text x="465" y="178" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">50%</text>

  <!-- Oct row - 100,78,66,59 -->
  <rect x="155" y="200" width="60" height="38" fill="#4fc3f7" rx="3"/>
  <text x="185" y="223" text-anchor="middle" font-size="11" fill="white" font-weight="bold">100%</text>
  <rect x="225" y="200" width="60" height="38" fill="#4fc3f7" opacity="0.78" rx="3"/>
  <text x="255" y="223" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">78%</text>
  <rect x="295" y="200" width="60" height="38" fill="#8bc34a" opacity="0.76" rx="3"/>
  <text x="325" y="223" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">66%</text>
  <rect x="365" y="200" width="60" height="38" fill="#ffa726" opacity="0.78" rx="3"/>
  <text x="395" y="223" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">59%</text>

  <!-- Nov row - 100,80,69 -->
  <rect x="155" y="245" width="60" height="38" fill="#4fc3f7" rx="3"/>
  <text x="185" y="268" text-anchor="middle" font-size="11" fill="white" font-weight="bold">100%</text>
  <rect x="225" y="245" width="60" height="38" fill="#4fc3f7" opacity="0.80" rx="3"/>
  <text x="255" y="268" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">80%</text>
  <rect x="295" y="245" width="60" height="38" fill="#8bc34a" opacity="0.79" rx="3"/>
  <text x="325" y="268" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">69%</text>

  <!-- Dec row - 100,82 -->
  <rect x="155" y="290" width="60" height="38" fill="#4fc3f7" rx="3"/>
  <text x="185" y="313" text-anchor="middle" font-size="11" fill="white" font-weight="bold">100%</text>
  <rect x="225" y="290" width="60" height="38" fill="#4fc3f7" opacity="0.82" rx="3"/>
  <text x="255" y="313" text-anchor="middle" font-size="11" fill="#333" font-weight="bold">82%</text>

  <!-- Color legend -->
  <rect x="155" y="345" width="25" height="15" fill="#4fc3f7" rx="2"/>
  <text x="185" y="357" font-size="10" fill="#333">70-100%</text>
  <rect x="255" y="345" width="25" height="15" fill="#8bc34a" opacity="0.75" rx="2"/>
  <text x="285" y="357" font-size="10" fill="#333">60-69%</text>
  <rect x="355" y="345" width="25" height="15" fill="#ffa726" opacity="0.75" rx="2"/>
  <text x="385" y="357" font-size="10" fill="#333">45-59%</text>
  <rect x="455" y="345" width="25" height="15" fill="#ef5350" opacity="0.6" rx="2"/>
  <text x="485" y="357" font-size="10" fill="#333">&lt;45%</text>
</svg>

**Figure 3.** Cohort retention heatmap. Each row represents a signup cohort. The improving M1 retention across later cohorts (72% -> 82%) indicates product/onboarding improvements are working. The July/August cohorts show classic AI-video novelty decay: steep month-1 drop, then gradual stabilization.

### 7.2 Cohort Revenue Table

Retention tells you who stays. Revenue cohorts tell you how much they spend:

| Cohort (n=1000) | M0 Rev | M1 Rev | M2 Rev | M3 Rev | M4 Rev | M5 Rev | M6 Rev |
|---|---|---|---|---|---|---|---|
| Jul 2025 | $25,000 | $19,800 | $17,110 | $15,500 | $14,625 | $14,280 | $14,000 |
| Aug 2025 | $25,000 | $19,250 | $16,225 | $14,570 | $13,650 | $13,260 | -- |
| Sep 2025 | $25,000 | $20,350 | $18,290 | $17,050 | $16,500 | -- | -- |
| Oct 2025 | $25,000 | $21,450 | $19,470 | $18,290 | -- | -- | -- |

Notice that the per-user revenue for surviving users *increases* over time. In July, 1000 users generated $25,000 (ARPU $25). By month 6, 400 users generated $14,000 (ARPU $35). Surviving users become more valuable --- they upgrade, buy credits, and use the product more intensively.

### 7.3 Cumulative LTV by Cohort

Summing cumulative revenue per original cohort member gives you realized LTV over time:

$$
\text{Cumulative LTV}_t = \sum_{i=0}^{t} \frac{\text{Cohort Revenue}_i}{\text{Original Cohort Size}}
$$

For the July 2025 cohort (n=1000):

| Month | Period Revenue | Cumulative Revenue | Cum. LTV per User |
|---|---|---|---|
| 0 | $25,000 | $25,000 | $25.00 |
| 1 | $19,800 | $44,800 | $44.80 |
| 2 | $17,110 | $61,910 | $61.91 |
| 3 | $15,500 | $77,410 | $77.41 |
| 4 | $14,625 | $92,035 | $92.04 |
| 5 | $14,280 | $106,315 | $106.32 |
| 6 | $14,000 | $120,315 | $120.32 |

With a CAC of $40 and 60% gross margins, the gross-profit-adjusted cumulative LTV at month 6 is $120.32 x 0.60 = $72.19, which exceeds CAC at around month 4. Payback period: ~4 months. Healthy.

### 7.4 Identifying Product-Market Fit from Cohort Data

Three signals of product-market fit in cohort data:

1. **Retention curve flattening**: If your retention curve flattens (stops declining) at a meaningful level (>20%), you have a core user base that finds lasting value. The curve should resemble a "hockey stick" turned on its side --- steep early decline, then horizontal.

2. **Improving cohorts over time**: Newer cohorts should retain better than older ones. If they don't, your product improvements aren't moving the needle.

3. **Revenue expansion within cohorts**: If surviving users spend more over time (ARPU increases), the product is becoming more valuable to retained users, not less.

If all three signals are present, you likely have product-market fit. If none are present, you're in trouble. If 1-2 are present, you're on the path but not there yet.

---

## 8. Survival Analysis and Predicting User Lifetime

For more sophisticated lifetime predictions, we can borrow from biostatistics: survival analysis.

### 8.1 The Kaplan-Meier Estimator

The Kaplan-Meier (KM) estimator constructs a survival function from observed data, handling the fact that some users are still active (censored --- we don't know when they'll churn).

Given ordered event times \(t_1 < t_2 < \cdots < t_k\) where at each \(t_i\):
- \(d_i\) = number of churns at time \(t_i\)
- \(n_i\) = number of users at risk just before time \(t_i\) (still active and not censored)

$$
\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)
$$

**Worked example:**

| Month \(t_i\) | At Risk \(n_i\) | Churned \(d_i\) | Censored | \(\frac{d_i}{n_i}\) | \(1 - \frac{d_i}{n_i}\) | \(\hat{S}(t_i)\) |
|---|---|---|---|---|---|---|
| 1 | 500 | 130 | 5 | 0.260 | 0.740 | 0.740 |
| 2 | 365 | 62 | 3 | 0.170 | 0.830 | 0.614 |
| 3 | 300 | 33 | 4 | 0.110 | 0.890 | 0.547 |
| 4 | 263 | 26 | 2 | 0.099 | 0.901 | 0.493 |
| 5 | 235 | 19 | 6 | 0.081 | 0.919 | 0.453 |
| 6 | 210 | 15 | 3 | 0.071 | 0.929 | 0.421 |

The censored users are those who signed up recently and haven't had the opportunity to churn yet. The KM estimator correctly handles this --- they're removed from the at-risk pool without being counted as churned.

**Interpreting the KM curve:**

- \(\hat{S}(3) = 0.547\): After 3 months, 54.7% of users are still active
- \(\hat{S}(6) = 0.421\): After 6 months, 42.1% are still active
- The median survival time is the value of \(t\) where \(\hat{S}(t) = 0.5\), approximately 4 months in this example

### 8.2 Confidence Intervals (Greenwood's Formula)

The KM estimator's variance is given by Greenwood's formula:

$$
\widehat{\text{Var}}[\hat{S}(t)] = [\hat{S}(t)]^2 \sum_{t_i \leq t} \frac{d_i}{n_i(n_i - d_i)}
$$

The 95% confidence interval:

$$
\hat{S}(t) \pm 1.96 \times \sqrt{\widehat{\text{Var}}[\hat{S}(t)]}
$$

This gives you error bars on your retention predictions. With small cohorts, the uncertainty is substantial --- don't make million-dollar decisions based on 50-user cohort data.

### 8.3 Using Survival Analysis for LTV Prediction

Once you have \(\hat{S}(t)\) from the KM estimator, plug it directly into the LTV formula:

$$
\text{LTV} = \sum_{t=1}^{T} \frac{\text{ARPU}_t \times \hat{S}(t)}{(1+d)^t}
$$

This is more accurate than the closed-form formula because it uses your actual retention curve shape rather than assuming constant churn. Your real retention curve is typically concave (steep early churn, flattening later), which means the constant-churn formula underestimates LTV for surviving users.

### 8.4 Python Implementation

```python
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

# Prepare data: each row is a user
# 'duration' = months active, 'observed' = 1 if churned, 0 if still active
data = pd.DataFrame({
    'duration': [1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 8, 10, 12, 12, 14],
    'observed': [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0]
})

kmf = KaplanMeierFitter()
kmf.fit(data['duration'], event_observed=data['observed'])

# Survival function
survival = kmf.survival_function_
print(survival)

# Median survival time
print(f"Median lifetime: {kmf.median_survival_time_:.1f} months")

# LTV calculation
arpu = 25  # $/month
discount_rate = 0.10 / 12  # monthly

ltv = 0
for t in range(1, 37):  # 36-month horizon
    if t in survival.index:
        s_t = survival.loc[t, 'KM_estimate']
    else:
        # Interpolate or use last known value
        s_t = survival.iloc[-1]['KM_estimate']

    ltv += arpu * s_t / (1 + discount_rate) ** t

print(f"36-month DCF LTV: ${ltv:.2f}")
```

### 8.5 Segmented Survival Analysis

Run separate KM estimators for different user segments to compare survival curves:

```python
# Compare survival by acquisition channel
for channel in ['organic', 'paid_search', 'paid_social', 'referral']:
    mask = data['channel'] == channel
    kmf.fit(
        data.loc[mask, 'duration'],
        event_observed=data.loc[mask, 'observed'],
        label=channel
    )
    kmf.plot()

# Log-rank test for statistical significance
from lifelines.statistics import logrank_test
results = logrank_test(
    organic_durations, paid_durations,
    event_observed_A=organic_events,
    event_observed_B=paid_events
)
print(f"p-value: {results.p_value:.4f}")
```

If the log-rank test shows \(p < 0.05\), the survival curves are statistically significantly different. This justifies different LTV assumptions for different channels.

---

## 9. Putting It All Together: The Metrics Dashboard

Every AI video platform founder should have a dashboard that surfaces these metrics in real-time. Here's the layout I recommend:

<svg viewBox="0 0 750 560" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 750px; font-family: 'Segoe UI', system-ui, sans-serif;">
  <!-- Title -->
  <text x="375" y="25" text-anchor="middle" font-size="16" fill="#333" font-weight="bold">AI Video Platform Metrics Dashboard</text>

  <!-- Row 1: KPI Cards -->
  <!-- MRR -->
  <rect x="15" y="45" width="170" height="90" fill="#f8f9fa" stroke="#dee2e6" rx="6"/>
  <text x="100" y="68" text-anchor="middle" font-size="10" fill="#666" font-weight="bold">MRR</text>
  <text x="100" y="95" text-anchor="middle" font-size="22" fill="#333" font-weight="bold">$127.4K</text>
  <text x="100" y="115" text-anchor="middle" font-size="11" fill="#8bc34a">+8.2% MoM</text>

  <!-- NRR -->
  <rect x="195" y="45" width="170" height="90" fill="#f8f9fa" stroke="#dee2e6" rx="6"/>
  <text x="280" y="68" text-anchor="middle" font-size="10" fill="#666" font-weight="bold">NET REVENUE RETENTION</text>
  <text x="280" y="95" text-anchor="middle" font-size="22" fill="#333" font-weight="bold">103.2%</text>
  <text x="280" y="115" text-anchor="middle" font-size="11" fill="#8bc34a">+1.1pp vs last month</text>

  <!-- LTV:CAC -->
  <rect x="375" y="45" width="170" height="90" fill="#f8f9fa" stroke="#dee2e6" rx="6"/>
  <text x="460" y="68" text-anchor="middle" font-size="10" fill="#666" font-weight="bold">LTV:CAC RATIO</text>
  <text x="460" y="95" text-anchor="middle" font-size="22" fill="#333" font-weight="bold">3.4x</text>
  <text x="460" y="115" text-anchor="middle" font-size="11" fill="#8bc34a">Above 3x target</text>

  <!-- Quick Ratio -->
  <rect x="555" y="45" width="170" height="90" fill="#f8f9fa" stroke="#dee2e6" rx="6"/>
  <text x="640" y="68" text-anchor="middle" font-size="10" fill="#666" font-weight="bold">QUICK RATIO</text>
  <text x="640" y="95" text-anchor="middle" font-size="22" fill="#333" font-weight="bold">2.4</text>
  <text x="640" y="115" text-anchor="middle" font-size="11" fill="#ffa726">Target: 4.0</text>

  <!-- Row 2: Secondary KPIs -->
  <!-- Logo Churn -->
  <rect x="15" y="150" width="115" height="75" fill="#f8f9fa" stroke="#dee2e6" rx="6"/>
  <text x="72" y="170" text-anchor="middle" font-size="9" fill="#666" font-weight="bold">LOGO CHURN</text>
  <text x="72" y="195" text-anchor="middle" font-size="18" fill="#ef5350" font-weight="bold">7.2%</text>
  <text x="72" y="212" text-anchor="middle" font-size="9" fill="#8bc34a">-0.8pp</text>

  <!-- Rev Churn -->
  <rect x="140" y="150" width="115" height="75" fill="#f8f9fa" stroke="#dee2e6" rx="6"/>
  <text x="197" y="170" text-anchor="middle" font-size="9" fill="#666" font-weight="bold">REV CHURN</text>
  <text x="197" y="195" text-anchor="middle" font-size="18" fill="#ffa726" font-weight="bold">4.8%</text>
  <text x="197" y="212" text-anchor="middle" font-size="9" fill="#8bc34a">-0.3pp</text>

  <!-- Blended CAC -->
  <rect x="265" y="150" width="115" height="75" fill="#f8f9fa" stroke="#dee2e6" rx="6"/>
  <text x="322" y="170" text-anchor="middle" font-size="9" fill="#666" font-weight="bold">BLENDED CAC</text>
  <text x="322" y="195" text-anchor="middle" font-size="18" fill="#333" font-weight="bold">$38</text>
  <text x="322" y="212" text-anchor="middle" font-size="9" fill="#ef5350">+$4 MoM</text>

  <!-- Payback -->
  <rect x="390" y="150" width="115" height="75" fill="#f8f9fa" stroke="#dee2e6" rx="6"/>
  <text x="447" y="170" text-anchor="middle" font-size="9" fill="#666" font-weight="bold">PAYBACK PERIOD</text>
  <text x="447" y="195" text-anchor="middle" font-size="18" fill="#333" font-weight="bold">4.8 mo</text>
  <text x="447" y="212" text-anchor="middle" font-size="9" fill="#8bc34a">Below 6mo target</text>

  <!-- GPUPM -->
  <rect x="515" y="150" width="115" height="75" fill="#f8f9fa" stroke="#dee2e6" rx="6"/>
  <text x="572" y="170" text-anchor="middle" font-size="9" fill="#666" font-weight="bold">GEN/USER/MO</text>
  <text x="572" y="195" text-anchor="middle" font-size="18" fill="#333" font-weight="bold">34.7</text>
  <text x="572" y="212" text-anchor="middle" font-size="9" fill="#8bc34a">+12%</text>

  <!-- Margin -->
  <rect x="640" y="150" width="85" height="75" fill="#f8f9fa" stroke="#dee2e6" rx="6"/>
  <text x="682" y="170" text-anchor="middle" font-size="9" fill="#666" font-weight="bold">GROSS MARGIN</text>
  <text x="682" y="195" text-anchor="middle" font-size="18" fill="#333" font-weight="bold">62%</text>
  <text x="682" y="212" text-anchor="middle" font-size="9" fill="#8bc34a">+2pp</text>

  <!-- Row 3: Charts area -->
  <!-- Retention Chart placeholder -->
  <rect x="15" y="240" width="350" height="150" fill="#f8f9fa" stroke="#dee2e6" rx="6"/>
  <text x="190" y="262" text-anchor="middle" font-size="11" fill="#666" font-weight="bold">COHORT RETENTION CURVES</text>
  <!-- Mini retention curves -->
  <polyline points="35,370 75,340 115,320 155,310 195,305 235,302 275,300 315,299" fill="none" stroke="#4fc3f7" stroke-width="1.5"/>
  <polyline points="35,370 75,345 115,330 155,322 195,318 235,316" fill="none" stroke="#8bc34a" stroke-width="1.5"/>
  <polyline points="35,370 75,350 115,338 155,332" fill="none" stroke="#ffa726" stroke-width="1.5"/>
  <text x="320" y="302" font-size="8" fill="#4fc3f7">Jul</text>
  <text x="240" y="319" font-size="8" fill="#8bc34a">Sep</text>
  <text x="160" y="335" font-size="8" fill="#ffa726">Nov</text>

  <!-- MRR Waterfall placeholder -->
  <rect x="375" y="240" width="350" height="150" fill="#f8f9fa" stroke="#dee2e6" rx="6"/>
  <text x="550" y="262" text-anchor="middle" font-size="11" fill="#666" font-weight="bold">MRR WATERFALL</text>
  <!-- Mini waterfall bars -->
  <rect x="405" y="290" width="40" height="80" fill="#4fc3f7" opacity="0.8" rx="2"/>
  <text x="425" y="380" text-anchor="middle" font-size="8" fill="#333">Start</text>
  <rect x="455" y="280" width="40" height="40" fill="#8bc34a" opacity="0.8" rx="2"/>
  <text x="475" y="380" text-anchor="middle" font-size="8" fill="#333">New</text>
  <rect x="505" y="295" width="40" height="25" fill="#8bc34a" opacity="0.6" rx="2"/>
  <text x="525" y="380" text-anchor="middle" font-size="8" fill="#333">Expand</text>
  <rect x="555" y="305" width="40" height="30" fill="#ef5350" opacity="0.7" rx="2"/>
  <text x="575" y="380" text-anchor="middle" font-size="8" fill="#333">Churn</text>
  <rect x="605" y="310" width="40" height="10" fill="#ef5350" opacity="0.5" rx="2"/>
  <text x="625" y="380" text-anchor="middle" font-size="8" fill="#333">Contract</text>
  <rect x="655" y="282" width="40" height="88" fill="#4fc3f7" opacity="0.8" rx="2"/>
  <text x="675" y="380" text-anchor="middle" font-size="8" fill="#333">End</text>

  <!-- Row 4: AI-video specific -->
  <!-- Model distribution -->
  <rect x="15" y="405" width="230" height="140" fill="#f8f9fa" stroke="#dee2e6" rx="6"/>
  <text x="130" y="428" text-anchor="middle" font-size="11" fill="#666" font-weight="bold">MODEL DISTRIBUTION</text>
  <!-- Mini pie approximation with stacked bar -->
  <rect x="40" y="440" width="90" height="20" fill="#4fc3f7" rx="2"/>
  <text x="135" y="455" font-size="9" fill="#333">Veo 3 - 38%</text>
  <rect x="40" y="465" width="65" height="20" fill="#ef5350" opacity="0.8" rx="2"/>
  <text x="110" y="480" font-size="9" fill="#333">Kling 3 - 27%</text>
  <rect x="40" y="490" width="45" height="20" fill="#8bc34a" rx="2"/>
  <text x="90" y="505" font-size="9" fill="#333">Runway - 19%</text>
  <rect x="40" y="515" width="38" height="20" fill="#ffa726" rx="2"/>
  <text x="83" y="530" font-size="9" fill="#333">Other - 16%</text>

  <!-- Cost per gen -->
  <rect x="255" y="405" width="235" height="140" fill="#f8f9fa" stroke="#dee2e6" rx="6"/>
  <text x="372" y="428" text-anchor="middle" font-size="11" fill="#666" font-weight="bold">COST/GEN BY MODEL</text>
  <text x="290" y="455" font-size="10" fill="#333">Veo 3 (1080p):</text>
  <text x="430" y="455" font-size="10" fill="#333" font-weight="bold">$0.28</text>
  <text x="290" y="475" font-size="10" fill="#333">Kling 3 (1080p):</text>
  <text x="430" y="475" font-size="10" fill="#333" font-weight="bold">$0.15</text>
  <text x="290" y="495" font-size="10" fill="#333">Runway (720p):</text>
  <text x="430" y="495" font-size="10" fill="#333" font-weight="bold">$0.12</text>
  <text x="290" y="515" font-size="10" fill="#333">Luma (720p):</text>
  <text x="430" y="515" font-size="10" fill="#333" font-weight="bold">$0.08</text>
  <text x="290" y="535" font-size="10" fill="#333">Blended avg:</text>
  <text x="430" y="535" font-size="10" fill="#8bc34a" font-weight="bold">$0.17</text>

  <!-- Success rate & TTFG -->
  <rect x="500" y="405" width="225" height="140" fill="#f8f9fa" stroke="#dee2e6" rx="6"/>
  <text x="612" y="428" text-anchor="middle" font-size="11" fill="#666" font-weight="bold">GENERATION HEALTH</text>
  <text x="530" y="460" font-size="10" fill="#333">Success Rate:</text>
  <text x="660" y="460" font-size="14" fill="#8bc34a" font-weight="bold">96.3%</text>
  <text x="530" y="490" font-size="10" fill="#333">Avg Queue Time:</text>
  <text x="660" y="490" font-size="14" fill="#333" font-weight="bold">12s</text>
  <text x="530" y="520" font-size="10" fill="#333">TTFG (median):</text>
  <text x="660" y="520" font-size="14" fill="#8bc34a" font-weight="bold">3.2 min</text>
</svg>

**Figure 4.** Recommended metrics dashboard layout for an AI video platform. Top row: four headline KPIs. Second row: supporting metrics. Bottom: charts for cohort retention, MRR waterfall, model distribution, and generation health.

### 9.1 Alert Thresholds

Set automated alerts for:

| Metric | Green | Yellow | Red |
|---|---|---|---|
| Monthly logo churn | < 6% | 6-10% | > 10% |
| NRR | > 105% | 95-105% | < 95% |
| LTV:CAC | > 3x | 2-3x | < 2x |
| Quick Ratio | > 3.0 | 2.0-3.0 | < 2.0 |
| Payback Period | < 6 mo | 6-12 mo | > 12 mo |
| Gen Success Rate | > 97% | 93-97% | < 93% |
| TTFG (median) | < 3 min | 3-10 min | > 10 min |
| Credit Utilization | 40-85% | 20-40% or 85-95% | < 20% or > 95% |
| Gross Margin | > 60% | 45-60% | < 45% |

### 9.2 Metrics Cadence

Not every metric needs daily attention:

| Frequency | Metrics |
|---|---|
| Real-time | Success rate, queue time, active generations |
| Daily | New signups, conversions, generations, revenue |
| Weekly | Churn (leading indicators), CAC by channel, GPUPM |
| Monthly | NRR, LTV:CAC, Quick Ratio, cohort analysis, payback |
| Quarterly | Survival analysis, LTV recalculation, strategic review |

---

## Summary: The Metrics That Matter Most

If you only track five metrics for your AI video platform, track these:

1. **Net Revenue Retention (NRR)**: Are existing customers growing in value? Target > 100%.
2. **LTV:CAC Ratio (gross-profit-adjusted)**: Is your acquisition engine profitable? Target > 3x.
3. **Payback Period**: How fast do you recover acquisition cost? Target < 6 months.
4. **Generations Per User Per Month**: Are users engaged? Higher = stickier.
5. **Gross Margin Per Generation**: Is the underlying unit economics healthy? Target > 55%.

Everything else is derivative or supporting. These five tell you whether you have a viable business.

The math is clear: in AI video, churn reduction is the single highest-leverage activity. A 2 percentage point reduction in monthly churn from 8% to 6% increases DCF LTV by roughly 35%, improves payback probability, and compounds across every cohort you've ever acquired. Before building the next feature, ask: will this reduce churn? If not, reconsider.

---

*This post is part of a series on the business side of AI video platforms. Next: [The API Aggregator Model]({% raw %}{% post_url 2026-01-10-api-aggregator-business-model %}{% endraw %}) and [Enterprise vs Consumer AI Video]({% raw %}{% post_url 2026-01-09-enterprise-vs-consumer-ai-video %}{% endraw %}).*
