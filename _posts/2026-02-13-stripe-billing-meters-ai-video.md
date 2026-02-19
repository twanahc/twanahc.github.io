---
layout: post
title: "Building Usage-Based Billing for AI Video: Stripe Meters, Credit Systems, and the Math of Sustainable Pricing"
date: 2026-02-13
category: infrastructure
---

Billing is the least glamorous and most critical piece of any AI video platform. Get it wrong and you either bleed money on every generation or price yourself out of the market. This post is the definitive technical guide to building usage-based billing for AI video, covering everything from the mathematical foundations of credit-based pricing through full Stripe Billing Meters implementations to revenue analytics formulas.

I am writing this now because there is a hard deadline: **March 31, 2026**, when Stripe removes the legacy usage records API entirely. If you are running metered billing on Stripe today, you must migrate. But rather than just cover the migration, I want to lay out the complete billing architecture for an AI video SaaS from first principles.

---

## Table of Contents

1. [The Fundamental Billing Problem for AI Video](#the-fundamental-billing-problem-for-ai-video)
2. [Three Billing Models: Deep Analysis with Math](#three-billing-models-deep-analysis-with-math)
3. [The Hybrid Model: Design and Implementation](#the-hybrid-model-design-and-implementation)
4. [Cost Modeling: From Provider Costs to User Prices](#cost-modeling-from-provider-costs-to-user-prices)
5. [Stripe Legacy vs. Billing Meters: Technical Comparison](#stripe-legacy-vs-billing-meters-technical-comparison)
6. [Stripe Billing Meters: Full Implementation](#stripe-billing-meters-full-implementation)
7. [Stripe Workflows for AI Billing Automation](#stripe-workflows-for-ai-billing-automation)
8. [Revenue Analytics: Formulas and Dashboards](#revenue-analytics-formulas-and-dashboards)
9. [Pricing Psychology for AI Video](#pricing-psychology-for-ai-video)
10. [Competitor Pricing Analysis](#competitor-pricing-analysis)

---

## The Fundamental Billing Problem for AI Video

AI video generation has a billing problem that most SaaS products do not have: **wildly variable cost per user action**. Consider:

| User Action | Model Used | Duration | Resolution | Your API Cost |
|---|---|---|---|---|
| Quick social clip | Luma Ray3.14 | 5s | 720p | $0.10 |
| Product showcase | Kling 2.1 Pro | 10s | 1080p | $1.60 |
| Cinematic hero video | Google Veo 3 | 8s | 4K | $2.40 |
| Storyboard preview | Luma Ray3.14 | 5s | 480p | $0.06 |
| Background removal + regen | Runway Gen-4 | 5s | 1080p | $1.00 |

The cheapest action costs $0.06. The most expensive costs $2.40. That is a **40x range** in cost for actions that the user perceives as "generating a video." In a traditional SaaS product, the cost to serve a user action varies by maybe 2-3x. A 40x range makes flat-rate pricing extremely difficult.

### Why Flat-Rate Fails

Suppose you offer an "Unlimited" plan at $49/month. You need to model the expected cost per user:

$$E[\text{cost}] = \sum_{i} P(\text{action}_i) \times C(\text{action}_i) \times N_{\text{generations}}$$

Where \(P(\text{action}_i)\) is the probability of choosing model \(i\), \(C(\text{action}_i)\) is the cost, and \(N_{\text{generations}}\) is monthly generation count.

For a typical active user:

| Model Choice | Probability | Cost | Expected Contribution |
|---|---|---|---|
| Luma Ray3.14 (fast/cheap) | 45% | $0.10 | $0.045 |
| Kling 2.1 Standard | 25% | $0.50 | $0.125 |
| Kling 2.1 Pro | 15% | $1.60 | $0.240 |
| Google Veo 3 | 10% | $2.40 | $0.240 |
| Runway Gen-4 | 5% | $1.00 | $0.050 |
| **Weighted average** | | | **$0.70 per generation** |

At $0.70 per generation, a $49/month plan breaks even at:

$$N_{\text{breakeven}} = \frac{49}{0.70} = 70 \text{ generations/month}$$

That sounds manageable. But here is the problem: **usage is not normally distributed**. It follows a power law. Most users generate 10-30 videos per month. A small percentage generate 200-500. On a flat-rate plan:

- 80% of users generate <50 videos: you make money
- 15% of users generate 50-100 videos: you roughly break even
- 5% of users generate 100-500 videos: **you lose $21-$301 per user per month**

The top 5% of users can destroy your unit economics. This is why every serious AI video platform uses some form of usage-based billing.

### The Three Pricing Dimensions

AI video has three natural dimensions to price on:

1. **Volume**: Number of generations or total seconds of video
2. **Quality**: Which model (cheap vs. premium), resolution, duration
3. **Features**: Special capabilities like start/end frame, character consistency, video editing

Most successful billing models use a combination. Credits are the cleanest abstraction because they can encode all three dimensions into a single unit.

---

## Three Billing Models: Deep Analysis with Math

### Model 1: Credit-Based (Pre-Purchased)

Users buy credit packs upfront. Each generation deducts credits based on model, duration, and features used.

**Structure:**

```
Credit Packs:
  Starter:  100 credits  @ $9.99   ($0.0999/credit)
  Pro:      500 credits  @ $39.99  ($0.0800/credit)
  Ultra:   2000 credits  @ $129.99 ($0.0650/credit)

Credit Costs per Generation:
  Luma Ray3.14 5s:        4 credits
  Kling 2.1 Standard 5s: 10 credits
  Kling 2.1 Pro 5s:      19 credits
  Google Veo 3 5s:       34 credits
  Flux 2.0 Pro (image):   2 credits
```

**Margin analysis at the Pro tier ($0.08/credit):**

| Operation | Credits | Revenue | Provider Cost | Infra Cost | Margin |
|---|---|---|---|---|---|
| Luma Ray3.14 5s | 4 | $0.32 | $0.12 | $0.02 | 56% |
| Kling 2.1 Std 5s | 10 | $0.80 | $0.50 | $0.02 | 35% |
| Kling 2.1 Pro 5s | 19 | $1.52 | $1.60 | $0.02 | -7% |
| Veo 3 5s | 34 | $2.72 | $2.40 | $0.02 | 11% |
| Flux 2.0 Pro | 2 | $0.16 | $0.04 | $0.01 | 69% |

Notice that Kling 2.1 Pro has a **negative margin** at this pricing. This is common: premium models often operate at a loss, subsidized by profits from cheaper models. The key metric is **blended margin** across the actual usage mix:

$$\text{Blended margin} = \frac{\sum_i P_i \times (R_i - C_i)}{\sum_i P_i \times R_i}$$

Using our earlier probability distribution:

$$\text{Revenue per avg generation} = 0.45(0.32) + 0.25(0.80) + 0.15(1.52) + 0.10(2.72) + 0.05(0.16) = \$0.90$$

$$\text{Cost per avg generation} = 0.45(0.14) + 0.25(0.52) + 0.15(1.62) + 0.10(2.42) + 0.05(0.05) = \$0.64$$

$$\text{Blended margin} = \frac{0.90 - 0.64}{0.90} = 28.9\%$$

That is too low for a healthy SaaS business. You need 50-65% blended margin. The fix is either to increase credit costs for premium models or increase the credit pack price. Let me recalculate with adjusted credit costs:

**Adjusted credit table (target 55% blended margin):**

| Operation | Old Credits | New Credits | Revenue (Pro) | Cost | Margin |
|---|---|---|---|---|---|
| Luma Ray3.14 5s | 4 | 4 | $0.32 | $0.14 | 56% |
| Kling 2.1 Std 5s | 10 | 14 | $1.12 | $0.52 | 54% |
| Kling 2.1 Pro 5s | 19 | 38 | $3.04 | $1.62 | 47% |
| Veo 3 5s | 34 | 56 | $4.48 | $2.42 | 46% |
| Flux 2.0 Pro | 2 | 2 | $0.16 | $0.05 | 69% |

New blended margin:

$$\text{Revenue} = 0.45(0.32) + 0.25(1.12) + 0.15(3.04) + 0.10(4.48) + 0.05(0.16) = \$1.40$$

$$\text{Cost} = \$0.64 \text{ (unchanged)}$$

$$\text{Blended margin} = \frac{1.40 - 0.64}{1.40} = 54.3\%$$

Now we are in the target range. The tradeoff is that premium models consume credits much faster, which users may perceive as expensive. But this is correct: premium models genuinely cost more, and the credit system makes that visible.

**Advantages of credit-based billing:**
- Simple mental model for users ("I have X credits left")
- Revenue recognized upfront (better cash flow)
- Users self-regulate usage
- Price changes only require updating the credit table, not user-facing prices

**Disadvantages:**
- Users must commit to a purchase before using the product (friction)
- Credit expiration policies are annoying and feel punitive
- Users with unused credits may churn ("I still have credits, why would I buy more?")
- Reconciliation complexity when credits span billing periods

### Model 2: Subscription + Included Credits

Users pay a monthly subscription that includes a set number of credits. Overage is billed per-credit.

**Structure:**

```
Subscription Tiers:
  Free:     $0/mo     30 credits included    No overage
  Starter:  $14.99/mo 150 credits included   $0.12/credit overage
  Pro:      $49.99/mo 600 credits included   $0.10/credit overage
  Business: $149.99/mo 2000 credits included $0.08/credit overage
```

**Calculating the right number of included credits:**

The included credits should cover the expected usage of the **median user** at each tier, with some buffer. The formula:

$$\text{Included credits} = E[\text{monthly generations}] \times E[\text{credits per generation}] \times (1 + \text{buffer})$$

For the Pro tier, assuming median users generate 50 videos/month at an average of 8 credits per generation:

$$\text{Included credits} = 50 \times 8 \times 1.5 = 600 \text{ credits}$$

The 1.5x buffer (50% more than expected usage) ensures that most Pro users stay within their included credits, which feels good. The overage pricing at \(0.10/credit is higher than the effective included rate (\)49.99 / 600 = $0.083/credit), incentivizing users to upgrade to the next tier rather than paying overage.

**Margin analysis at the Pro tier:**

Let us model three user archetypes:

**Light user (25 generations/month, avg 6 credits each = 150 credits):**

| | Amount |
|---|---|
| Subscription revenue | $49.99 |
| Overage revenue | $0.00 |
| Total revenue | $49.99 |
| Provider costs (150 credits worth) | ~$10.50 |
| Infrastructure costs | ~$3.00 |
| **Margin** | **$36.49 (73%)** |

**Median user (50 generations/month, avg 8 credits each = 400 credits):**

| | Amount |
|---|---|
| Subscription revenue | $49.99 |
| Overage revenue | $0.00 |
| Total revenue | $49.99 |
| Provider costs (400 credits worth) | ~$28.00 |
| Infrastructure costs | ~$5.00 |
| **Margin** | **$16.99 (34%)** |

**Power user (120 generations/month, avg 10 credits each = 1200 credits):**

| | Amount |
|---|---|
| Subscription revenue | $49.99 |
| Overage revenue | 600 credits * $0.10 = $60.00 |
| Total revenue | $109.99 |
| Provider costs (1200 credits worth) | ~$84.00 |
| Infrastructure costs | ~$12.00 |
| **Margin** | **$13.99 (13%)** |

The problem is clear: power users have thin margins even with overage charges, because the included credits were consumed at a below-market rate. The overage rate ($0.10/credit) needs to be high enough to cover costs with margin.

**The overage pricing formula:**

$$\text{Overage rate} = \frac{E[\text{cost per credit for power users}]}{1 - \text{target margin for overage}}$$

If power users tend to use premium models more (average cost per credit of $0.09):

$$\text{Overage rate} = \frac{0.09}{1 - 0.45} = \$0.164$$

Rounding to $0.15/credit for the Pro tier overage would yield healthier power-user margins. But this is a psychological tension: $0.15/credit overage vs \(0.083/credit included feels punitive. The solution is to make the upgrade to Business tier (\)149.99/month for 2000 credits = $0.075/credit) more attractive than paying overage.

**Upgrade decision point for a Pro user:**

At what usage level is upgrading to Business cheaper than paying Pro + overage?

$$49.99 + (N - 600) \times 0.15 = 149.99$$

$$0.15N - 90 = 100$$

$$N = \frac{190}{0.15} = 1267 \text{ credits}$$

So a Pro user consuming more than ~1267 credits/month should upgrade to Business. You should trigger an upgrade prompt when they hit 80% of this threshold (~1000 credits).

### Model 3: Pure Usage-Based (Pay-As-You-Go)

Users pay per generation with no subscription commitment.

**Structure:**

```
Per-generation pricing:
  Luma Ray3.14 5s:       $0.35
  Kling 2.1 Standard 5s: $1.15
  Kling 2.1 Pro 5s:      $3.50
  Google Veo 3 5s:       $5.00
  Flux 2.0 Pro (image):  $0.15

Minimum monthly charge: $0 (true pay-as-you-go)
Prepaid balance option: 10% discount on deposits >$50
```

**Margin analysis:**

$$\text{Price per generation} = \frac{\text{provider cost} + \text{infra cost}}{1 - \text{target margin}}$$

For Luma Ray3.14 at 55% target margin:

$$\text{Price} = \frac{0.12 + 0.02}{1 - 0.55} = \frac{0.14}{0.45} = \$0.311 \approx \$0.35 \text{ (rounded up)}$$

For Veo 3:

$$\text{Price} = \frac{2.40 + 0.02}{1 - 0.55} = \frac{2.42}{0.45} = \$5.38 \approx \$5.00 \text{ (rounded down for psychology)}$$

Note that I rounded Veo 3 down to $5.00 even though the formula says $5.38. This is deliberate -- $5.00 is a psychologically cleaner price point. The actual margin is:

$$\text{Margin} = 1 - \frac{2.42}{5.00} = 51.6\%$$

Still within the target range.

**Advantages of pure usage-based:**
- Zero friction to start (no commitment)
- Users pay exactly for what they use (feels fair)
- Revenue scales linearly with usage (predictable unit economics)
- No unused credits to manage or expire

**Disadvantages:**
- Unpredictable revenue (hard to forecast)
- No recurring revenue component (lower valuation multiples)
- Users may be hesitant to experiment (each generation has a visible cost)
- High churn risk (no switching cost)

---

## The Hybrid Model: Design and Implementation

The optimal billing model for most AI video platforms is a **hybrid** that combines subscription + included credits + usage-based overage + credit packs. Here is the complete design.

### Tier Structure

```
┌──────────────────────────────────────────────────────────────┐
│                    PRICING ARCHITECTURE                       │
│                                                               │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │  Free    │  │ Starter  │  │  Pro     │  │  Business    │ │
│  │  $0/mo   │  │ $14.99/mo│  │ $49.99/mo│  │ $149.99/mo  │ │
│  │          │  │          │  │          │  │              │ │
│  │ 30 cr    │  │ 150 cr   │  │ 600 cr   │  │ 2000 cr     │ │
│  │ incl.    │  │ incl.    │  │ incl.    │  │ incl.        │ │
│  │          │  │          │  │          │  │              │ │
│  │ No       │  │ $0.15/cr │  │ $0.12/cr │  │ $0.09/cr    │ │
│  │ overage  │  │ overage  │  │ overage  │  │ overage      │ │
│  │          │  │          │  │          │  │              │ │
│  │ Luma     │  │ Luma +   │  │ All      │  │ All models + │ │
│  │ only     │  │ Kling Std│  │ models   │  │ priority     │ │
│  └─────────┘  └──────────┘  └──────────┘  └──────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              ADD-ON CREDIT PACKS                         │ │
│  │  100 credits: $9.99  |  500 credits: $39.99             │ │
│  │  Never expire while subscription is active               │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### Credit Table (Final)

This is the production credit table that balances user perception with sustainable margins:

| Operation | Duration | Credits | Effective Price (Pro) | Your Cost | Margin |
|---|---|---|---|---|---|
| **Luma Ray3.14** (fast) | 5s | 4 | $0.33 | $0.14 | 58% |
| **Luma Ray3.14** (start+end) | 5s | 5 | $0.42 | $0.20 | 52% |
| **Luma Ray3.14** (9s) | 9s | 7 | $0.58 | $0.24 | 59% |
| **Kling 2.1 Standard** | 5s | 12 | $1.00 | $0.52 | 48% |
| **Kling 2.1 Standard** | 10s | 22 | $1.83 | $1.02 | 44% |
| **Kling 2.1 Pro** | 5s | 30 | $2.50 | $1.62 | 35% |
| **Kling 2.1 Pro** | 10s | 55 | $4.58 | $3.22 | 30% |
| **Google Veo 3** | 5s | 42 | $3.50 | $2.42 | 31% |
| **Google Veo 3** | 8s | 64 | $5.33 | $3.82 | 28% |
| **Runway Gen-4** | 5s | 24 | $2.00 | $1.02 | 49% |
| **Runway Gen-4** | 10s | 44 | $3.67 | $2.02 | 45% |
| **Flux 2.0 Pro** (image) | -- | 2 | $0.17 | $0.05 | 71% |
| **Flux 2.0 Dev** (image) | -- | 1 | $0.08 | $0.02 | 75% |

**Blended margin calculation using typical Pro user mix:**

| Model | Usage % | Credits | Revenue | Cost | Contribution |
|---|---|---|---|---|---|
| Luma Ray3.14 5s | 35% | 4 | $0.33 | \(0.14 | +\)0.067 |
| Luma Ray3.14 start+end | 10% | 5 | $0.42 | \(0.20 | +\)0.022 |
| Kling 2.1 Std 5s | 20% | 12 | $1.00 | \(0.52 | +\)0.096 |
| Kling 2.1 Pro 5s | 10% | 30 | $2.50 | \(1.62 | +\)0.088 |
| Veo 3 5s | 5% | 42 | $3.50 | \(2.42 | +\)0.054 |
| Runway Gen-4 5s | 5% | 24 | $2.00 | \(1.02 | +\)0.049 |
| Flux images | 15% | 1.5 avg | $0.13 | \(0.04 | +\)0.014 |

$$\text{Weighted revenue per generation} = \$0.82$$

$$\text{Weighted cost per generation} = \$0.43$$

$$\text{Blended margin} = \frac{0.82 - 0.43}{0.82} = 47.6\%$$

With the subscription base revenue added on top, overall margin reaches approximately 55-60%, which is healthy.

---

## Cost Modeling: From Provider Costs to User Prices

The general pricing formula for any AI video operation:

$$P_{\text{user}} = \frac{C_{\text{provider}} + C_{\text{infra}} + C_{\text{support}}}{1 - M_{\text{target}}}$$

Where:
- \(C_{\text{provider}}\) = API cost from the model provider
- \(C_{\text{infra}}\) = your infrastructure cost (API gateway, queue, storage, CDN, monitoring)
- \(C_{\text{support}}\) = allocated customer support cost per generation
- \(M_{\text{target}}\) = target gross margin (0.50 to 0.65 for SaaS)

### Infrastructure Cost Breakdown

For a typical AI video SaaS processing 50,000 generations per month:

| Component | Monthly Cost | Per Generation |
|---|---|---|
| **API Gateway** (Cloudflare Workers / AWS API Gateway) | $50 | $0.001 |
| **Queue System** (Redis / SQS) | $100 | $0.002 |
| **Video Storage** (S3, 500GB/month) | $12 | $0.0002 |
| **CDN Delivery** (Cloudfront, 2TB/month) | $170 | $0.003 |
| **Database** (Firestore / Postgres) | $80 | $0.002 |
| **Monitoring** (Datadog / self-hosted) | $200 | $0.004 |
| **Auth** (Firebase Auth / Clerk) | $50 | $0.001 |
| **Background Jobs** (Cloud Run / Lambda) | $150 | $0.003 |
| **Webhook Processing** | $30 | $0.001 |
| **Total Infrastructure** | **\(842** | **\)0.017** |

So \(C_{\text{infra}} \approx \\)0.02$ per generation is a reasonable estimate at moderate scale. At higher scale (500K+ generations/month), this drops to ~$0.005-0.01 due to better amortization.

### The Support Cost Factor

Customer support is easy to overlook. For an AI video platform:

- **Failed generations**: Users contact support when generations fail or produce garbage. At a 5% contact rate and $5/support ticket (automated + human blend), that is $0.25 per contacted generation, or $0.0125 per generation amortized across all generations.
- **Billing questions**: At a 2% monthly contact rate per user and $3/ticket, roughly $0.005 per generation.
- **Feature questions**: Minimal, ~$0.002 per generation.

Total: \(C_{\text{support}} \approx \\)0.02$ per generation.

### Complete Pricing Model

Putting it all together for each model tier:

$$P_{\text{user}} = \frac{C_{\text{provider}} + 0.02 + 0.02}{1 - 0.55}$$

| Model | Provider Cost | + Infra | + Support | Total Cost | / (1-0.55) | User Price |
|---|---|---|---|---|---|---|
| Luma Ray3.14 5s | $0.10 | $0.12 | $0.14 | $0.14 | \(0.31 | **\)0.35** |
| Kling 2.1 Std 5s | $0.50 | $0.52 | $0.54 | $0.54 | \(1.20 | **\)1.25** |
| Kling 2.1 Pro 5s | $1.60 | $1.62 | $1.64 | $1.64 | \(3.64 | **\)3.75** |
| Veo 3 5s | $2.40 | $2.42 | $2.44 | $2.44 | \(5.42 | **\)5.50** |
| Runway Gen-4 5s | $1.00 | $1.02 | $1.04 | $1.04 | \(2.31 | **\)2.50** |

These are the per-generation prices at 55% gross margin. In a credit system, you convert these to credit amounts by dividing by your price-per-credit.

---

## Stripe Legacy vs. Billing Meters: Technical Comparison

Let me explain exactly what is changing in Stripe's billing infrastructure and why it matters.

### The Legacy System (Deprecated)

The old metered billing flow:

```
┌──────────┐     ┌──────────────┐     ┌────────────────────┐
│ Your App  │────▶│ POST /v1/    │────▶│ Stripe aggregates  │
│ (after    │     │ subscription_│     │ at end of billing   │
│ each gen) │     │ items/{id}/  │     │ period, creates     │
│           │     │ usage_records│     │ invoice             │
└──────────┘     └──────────────┘     └────────────────────┘
```

**API call (legacy):**

```typescript
// DEPRECATED - Breaks after March 31, 2026
await stripe.subscriptionItems.createUsageRecord(
  'si_xxx', // subscription item ID
  {
    quantity: 5, // 5 seconds of video
    timestamp: Math.floor(Date.now() / 1000),
    action: 'increment', // or 'set'
  }
);
```

**Problems with the legacy system:**

1. **Throughput limit**: ~100 requests/second across your entire account. For a platform with thousands of concurrent users, this is a bottleneck.

2. **Aggregation latency**: ~3 minutes. Users see their credit balance update with a 3-minute delay. This is confusing in an interactive tool where they expect near-instant feedback.

3. **Tied to subscription items**: Usage records must reference a specific subscription item ID. If a user changes plans mid-cycle, you need to handle the subscription item ID change. This creates race conditions.

4. **No test clock support for v2**: You cannot properly test billing cycles without waiting for real time to pass. (Legacy usage records work with v1 test clocks but with limitations.)

5. **No event deduplication**: If your webhook handler retries and sends the same usage record twice, you get double-counted usage. You must implement idempotency yourself.

### The New System: Billing Meters

```
┌──────────┐     ┌──────────────┐     ┌────────────────────┐
│ Your App  │────▶│ POST /v2/    │────▶│ Stripe aggregates  │
│ (after    │     │ billing/     │     │ in ~30 seconds,    │
│ each gen) │     │ meter_events │     │ attaches to sub    │
│           │     │              │     │ automatically      │
└──────────┘     └──────────────┘     └────────────────────┘
```

**Key differences:**

| Feature | Legacy Usage Records | Billing Meters |
|---|---|---|
| **Throughput** | ~100/s account-wide | 100,000/s account-wide |
| **Aggregation latency** | ~3 minutes | ~30 seconds |
| **Event deduplication** | Manual (you implement) | Built-in (idempotency key) |
| **Test clock support** | Limited | Full (v2 events) |
| **Subscription coupling** | Must reference sub item ID | Reference customer ID only |
| **Historical queries** | Limited | Full event history |
| **Real-time dashboard** | No | Yes (meter summary) |
| **Webhook events** | `invoice.created` only | `billing.meter.usage_reported` |

The architecture improvement is significant. Billing Meters are a proper event-sourcing system: you emit events, Stripe aggregates them, and the aggregation drives invoicing. Your application code becomes simpler because you do not need to track subscription item IDs -- you just emit events keyed by customer ID and event name.

### Migration Timeline

| Date | Action Required |
|---|---|
| **Now** | All new features should use Billing Meters |
| **February 2026** | Begin migrating existing metered subscriptions |
| **March 15, 2026** | Complete migration testing with test clocks |
| **March 31, 2026** | Legacy usage records API removed in `2025-03-31.basil` |
| **April 2026** | If you pinned API version to `2025-02-24.acacia`, you still work but receive no updates |

---

## Stripe Billing Meters: Full Implementation

Here is the complete TypeScript implementation for billing an AI video platform using Stripe Billing Meters.

### Step 1: Create Billing Meters

You need one meter per billable dimension. For an AI video platform, I recommend two meters: one for video generation credits and one for image generation credits.

```typescript
// billing-setup.ts
import Stripe from "stripe";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: "2025-02-24.acacia", // Use latest stable version
});

/**
 * One-time setup: Create billing meters for video and image generation.
 * Run this once during initial setup.
 */
async function createBillingMeters() {
  // Video generation meter (measured in credits)
  const videoMeter = await stripe.billing.meters.create({
    display_name: "Video Generation Credits",
    event_name: "video_generation_credits",
    default_aggregation: {
      formula: "sum",
    },
    customer_mapping: {
      event_payload_key: "stripe_customer_id",
      type: "by_id",
    },
    value_settings: {
      event_payload_key: "credits",
    },
  });

  console.log(`Video meter created: ${videoMeter.id}`);

  // Image generation meter (measured in credits)
  const imageMeter = await stripe.billing.meters.create({
    display_name: "Image Generation Credits",
    event_name: "image_generation_credits",
    default_aggregation: {
      formula: "sum",
    },
    customer_mapping: {
      event_payload_key: "stripe_customer_id",
      type: "by_id",
    },
    value_settings: {
      event_payload_key: "credits",
    },
  });

  console.log(`Image meter created: ${imageMeter.id}`);

  return { videoMeter, imageMeter };
}
```

### Step 2: Create Prices Backed by Meters

Each subscription tier needs a metered price component backed by the billing meter. This is the price that Stripe uses to calculate overage charges.

```typescript
/**
 * Create tiered pricing backed by the video generation meter.
 * Each tier has different overage rates.
 */
async function createMeterPrices(videoMeterId: string) {
  // Starter tier overage: $0.15/credit
  const starterOverage = await stripe.prices.create({
    currency: "usd",
    unit_amount: 15, // $0.15 in cents
    billing_scheme: "per_unit",
    recurring: {
      interval: "month",
      usage_type: "metered",
      meter: videoMeterId,
    },
    product_data: {
      name: "Video Credits - Starter Overage",
    },
    metadata: {
      tier: "starter",
      included_credits: "150",
    },
  });

  // Pro tier overage: $0.12/credit
  const proOverage = await stripe.prices.create({
    currency: "usd",
    unit_amount: 12, // $0.12 in cents
    billing_scheme: "per_unit",
    recurring: {
      interval: "month",
      usage_type: "metered",
      meter: videoMeterId,
    },
    product_data: {
      name: "Video Credits - Pro Overage",
    },
    metadata: {
      tier: "pro",
      included_credits: "600",
    },
  });

  // Business tier overage: $0.09/credit
  const businessOverage = await stripe.prices.create({
    currency: "usd",
    unit_amount: 9, // $0.09 in cents
    billing_scheme: "per_unit",
    recurring: {
      interval: "month",
      usage_type: "metered",
      meter: videoMeterId,
    },
    product_data: {
      name: "Video Credits - Business Overage",
    },
    metadata: {
      tier: "business",
      included_credits: "2000",
    },
  });

  return { starterOverage, proOverage, businessOverage };
}

/**
 * Create a subscription with both a fixed recurring component
 * (the base subscription) and a metered component (overage).
 */
async function createSubscription(
  customerId: string,
  basePriceId: string, // Fixed monthly price ($14.99, $49.99, etc.)
  overagePriceId: string // Metered price for overage
) {
  const subscription = await stripe.subscriptions.create({
    customer: customerId,
    items: [
      { price: basePriceId }, // Fixed component
      { price: overagePriceId }, // Metered overage component
    ],
    payment_behavior: "default_incomplete",
    payment_settings: {
      save_default_payment_method: "on_subscription",
    },
    expand: ["latest_invoice.payment_intent"],
  });

  return subscription;
}
```

### Step 3: Record Usage Events

This is the core of the billing integration -- recording meter events every time a user generates content.

```typescript
// billing-events.ts
import Stripe from "stripe";
import { randomUUID } from "crypto";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!);

interface GenerationEvent {
  userId: string;
  stripeCustomerId: string;
  model: string;
  type: "video" | "image";
  duration?: number; // seconds, for video
  credits: number;
  generationId: string;
  metadata?: Record<string, string>;
}

/**
 * Record a generation event in Stripe's billing meter.
 * This function should be called AFTER the generation succeeds.
 *
 * IMPORTANT: Use the generationId as the idempotency key to prevent
 * double-counting on retries.
 */
async function recordGenerationEvent(
  event: GenerationEvent
): Promise<void> {
  const eventName =
    event.type === "video"
      ? "video_generation_credits"
      : "image_generation_credits";

  try {
    await stripe.v2.billing.meterEvents.create({
      event_name: eventName,
      payload: {
        stripe_customer_id: event.stripeCustomerId,
        credits: String(event.credits),
        // Additional metadata for analytics (not used in billing)
        model: event.model,
        duration: String(event.duration || 0),
        generation_id: event.generationId,
      },
      // Idempotency key prevents double-counting on retry
      identifier: event.generationId,
      // Timestamp when the generation actually occurred
      timestamp: new Date().toISOString(),
    });

    console.log(
      `Recorded ${event.credits} credits for customer ` +
        `${event.stripeCustomerId} (${event.model}, ${eventName})`
    );
  } catch (error) {
    if (
      error instanceof Stripe.errors.StripeError &&
      error.code === "resource_already_exists"
    ) {
      // Duplicate event -- this is fine, it was already recorded
      console.log(
        `Duplicate event for generation ${event.generationId}, skipping`
      );
      return;
    }
    throw error;
  }
}

/**
 * Record a video generation. Calculates credits based on model and duration.
 */
async function recordVideoGeneration(
  userId: string,
  stripeCustomerId: string,
  model: string,
  durationSeconds: number,
  generationId: string,
  options?: {
    startEndFrame?: boolean;
  }
): Promise<number> {
  const credits = calculateVideoCredits(
    model,
    durationSeconds,
    options?.startEndFrame || false
  );

  await recordGenerationEvent({
    userId,
    stripeCustomerId,
    model,
    type: "video",
    duration: durationSeconds,
    credits,
    generationId,
  });

  return credits;
}

/**
 * Credit calculation logic. This is the single source of truth
 * for how many credits each operation costs.
 */
function calculateVideoCredits(
  model: string,
  durationSeconds: number,
  startEndFrame: boolean
): number {
  // Base credits per 5-second increment
  const baseCreditsPerIncrement: Record<string, number> = {
    "ray-3-14": 4,
    "kling-2.1-standard": 12,
    "kling-2.1-pro": 30,
    "veo-3": 42,
    "gen-4": 24,
  };

  const base = baseCreditsPerIncrement[model];
  if (!base) {
    throw new Error(`Unknown model: ${model}`);
  }

  // Calculate increments (round up to nearest 5s)
  const increments = Math.ceil(durationSeconds / 5);

  // Start+end frame adds 25% surcharge (rounded up)
  let credits = base * increments;
  if (startEndFrame) {
    credits = Math.ceil(credits * 1.25);
  }

  return credits;
}

/**
 * Get current credit usage for a customer in the current billing period.
 * Uses Stripe's meter summary API.
 */
async function getCurrentUsage(
  customerId: string,
  meterId: string
): Promise<{
  totalCreditsUsed: number;
  includedCredits: number;
  overageCredits: number;
}> {
  const summary = await stripe.billing.meters.listEventSummaries(
    meterId,
    {
      customer: customerId,
      start_time: Math.floor(getBillingPeriodStart().getTime() / 1000),
      end_time: Math.floor(Date.now() / 1000),
    }
  );

  const totalUsed = summary.data.reduce(
    (sum, s) => sum + s.aggregated_value,
    0
  );

  // Look up the customer's subscription to find included credits
  const subscriptions = await stripe.subscriptions.list({
    customer: customerId,
    status: "active",
    limit: 1,
  });

  const sub = subscriptions.data[0];
  const includedCredits = sub
    ? parseInt(
        sub.items.data[0]?.price?.metadata?.included_credits || "0"
      )
    : 0;

  const overageCredits = Math.max(0, totalUsed - includedCredits);

  return {
    totalCreditsUsed: totalUsed,
    includedCredits,
    overageCredits,
  };
}

function getBillingPeriodStart(): Date {
  // Simplified -- in production, use the subscription's
  // current_period_start
  const now = new Date();
  return new Date(now.getFullYear(), now.getMonth(), 1);
}

export {
  recordGenerationEvent,
  recordVideoGeneration,
  calculateVideoCredits,
  getCurrentUsage,
  GenerationEvent,
};
```

### Step 4: Handle Included Credits with the Overage Model

The tricky part of the subscription + overage model is that Stripe's meters track ALL usage, but you only want to charge for overage (usage above the included credits). There are two approaches:

**Approach A: Stripe Tiers (Recommended)**

Use Stripe's tiered pricing to make the first N credits free:

```typescript
/**
 * Create a tiered price where the first N credits are free
 * (covered by the subscription) and subsequent credits are charged.
 */
async function createTieredOveragePrice(
  meterId: string,
  includedCredits: number,
  overageRate: number // in cents
) {
  const price = await stripe.prices.create({
    currency: "usd",
    billing_scheme: "tiered",
    tiers_mode: "graduated",
    tiers: [
      {
        // First N credits: free (included in subscription)
        up_to: includedCredits,
        unit_amount: 0,
      },
      {
        // Everything above N: charged at overage rate
        up_to: "inf",
        unit_amount: overageRate,
      },
    ],
    recurring: {
      interval: "month",
      usage_type: "metered",
      meter: meterId,
    },
    product_data: {
      name: "Video Generation Credits",
    },
  });

  return price;
}

// Example: Pro tier with 600 included credits, $0.12/credit overage
// createTieredOveragePrice("mtr_xxx", 600, 12);
```

This is the cleanest approach because Stripe handles the math entirely. You send ALL credit usage as meter events, and Stripe automatically charges $0 for the first 600 and $0.12/credit for everything above.

**Approach B: Client-Side Tracking**

If you need more control (e.g., different credit pools for video vs. image), track included credits in your own database and only emit meter events for overage:

```typescript
/**
 * Track credit usage and only emit Stripe meter events for overage.
 * Requires your own credit balance tracking in your database.
 */
async function recordUsageWithIncludedCredits(
  userId: string,
  stripeCustomerId: string,
  credits: number,
  generationId: string,
  db: DatabaseClient
): Promise<{
  creditsCharged: number;
  fromIncluded: number;
  fromOverage: number;
}> {
  // Get user's current credit balance from your database
  const balance = await db.getCreditBalance(userId);

  let fromIncluded = 0;
  let fromOverage = 0;

  if (balance.remainingIncluded >= credits) {
    // Fully covered by included credits
    fromIncluded = credits;
    fromOverage = 0;
  } else {
    // Partially covered by included, rest is overage
    fromIncluded = balance.remainingIncluded;
    fromOverage = credits - fromIncluded;
  }

  // Update your database
  await db.deductCredits(userId, fromIncluded, fromOverage);

  // Only emit a Stripe meter event for the overage portion
  if (fromOverage > 0) {
    await recordGenerationEvent({
      userId,
      stripeCustomerId,
      model: "aggregate",
      type: "video",
      credits: fromOverage,
      generationId,
    });
  }

  return {
    creditsCharged: credits,
    fromIncluded,
    fromOverage,
  };
}
```

I recommend **Approach A** unless you have specific requirements that demand client-side tracking. Approach A has fewer moving parts and fewer opportunities for billing discrepancies.

### Step 5: Pre-Generation Credit Check

Before starting a generation, verify the user has sufficient credits (either included or payment method for overage):

```typescript
/**
 * Check if a user can afford a generation before starting it.
 * Returns true if the user has included credits remaining
 * or has a valid payment method for overage charges.
 */
async function canAffordGeneration(
  stripeCustomerId: string,
  requiredCredits: number,
  meterId: string
): Promise<{
  allowed: boolean;
  reason?: string;
  remainingIncluded: number;
}> {
  // Get current usage
  const usage = await getCurrentUsage(stripeCustomerId, meterId);
  const remainingIncluded = Math.max(
    0,
    usage.includedCredits - usage.totalCreditsUsed
  );

  // If user has enough included credits, allow immediately
  if (remainingIncluded >= requiredCredits) {
    return { allowed: true, remainingIncluded };
  }

  // If user will go into overage, check they have a payment method
  const customer = await stripe.customers.retrieve(stripeCustomerId);
  if (customer.deleted) {
    return {
      allowed: false,
      reason: "Customer account not found",
      remainingIncluded,
    };
  }

  const paymentMethods = await stripe.paymentMethods.list({
    customer: stripeCustomerId,
    type: "card",
    limit: 1,
  });

  if (paymentMethods.data.length === 0) {
    return {
      allowed: false,
      reason:
        "No payment method on file. Add a card to use overage credits.",
      remainingIncluded,
    };
  }

  // Check for a spending cap (optional)
  const subscriptions = await stripe.subscriptions.list({
    customer: stripeCustomerId,
    status: "active",
    limit: 1,
  });

  const sub = subscriptions.data[0];
  if (sub?.metadata?.spending_cap) {
    const cap = parseFloat(sub.metadata.spending_cap);
    const currentOverageSpend =
      usage.overageCredits *
      (sub.items.data[1]?.price?.unit_amount || 0) / 100;

    if (currentOverageSpend >= cap) {
      return {
        allowed: false,
        reason: `Monthly spending cap of $${cap} reached. ` +
          `Increase your cap or upgrade your plan.`,
        remainingIncluded,
      };
    }
  }

  return { allowed: true, remainingIncluded };
}
```

---

## Stripe Workflows for AI Billing Automation

Stripe Workflows allow you to build event-driven automation directly in Stripe, without writing backend code. For AI video billing, there are several high-value automations.

### Workflow 1: Low Credit Warning

**Trigger**: When meter usage reaches 80% of included credits.

```
Trigger: billing.meter.usage_reported
Condition: aggregated_value >= included_credits * 0.8

Actions:
  1. Send email via Stripe:
     Subject: "You've used 80% of your monthly credits"
     Body: "You have {remaining} credits left this period.
            Consider upgrading to {next_tier} for more included
            credits at a lower rate."

  2. Send webhook to your app:
     URL: https://your-app.com/webhooks/low-credits
     Payload: { customer_id, usage, remaining, tier }
```

In Stripe's dashboard, this is configured as:

```typescript
// Programmatic workflow creation (Stripe API)
const workflow = await stripe.workflows.create({
  name: "Low Credit Warning",
  trigger: {
    type: "billing.meter.usage_reported",
    filter: {
      meter: "mtr_video_xxx",
    },
  },
  steps: [
    {
      type: "condition",
      condition: {
        field: "aggregated_value",
        operator: "gte",
        // This would be dynamic per customer, so in practice
        // you'd use the webhook approach below
        value: "480", // 80% of 600 for Pro tier
      },
    },
    {
      type: "send_email",
      template: "low_credit_warning",
      to: "{{customer.email}}",
    },
    {
      type: "webhook",
      url: "https://your-app.com/webhooks/low-credits",
    },
  ],
});
```

In practice, the threshold varies by tier, so the most reliable approach is to handle the `billing.meter.usage_reported` webhook in your own code and implement the comparison logic there:

```typescript
// webhook-handler.ts
import { calculateVideoCredits, getCurrentUsage } from "./billing-events";

async function handleMeterUsageWebhook(event: Stripe.Event) {
  if (event.type !== "billing.meter.usage_reported") return;

  const meterEvent = event.data.object;
  const customerId = meterEvent.customer;

  const usage = await getCurrentUsage(customerId, meterEvent.meter);
  const usagePercent = usage.totalCreditsUsed / usage.includedCredits;

  if (usagePercent >= 0.8 && usagePercent < 0.85) {
    // Send 80% warning (only once, use the range to prevent duplicates)
    await sendLowCreditEmail(customerId, usage);
    await notifyAppLowCredits(customerId, usage);
  }

  if (usage.totalCreditsUsed >= usage.includedCredits) {
    // User has exhausted included credits
    await sendCreditsExhaustedEmail(customerId, usage);
  }
}
```

### Workflow 2: Auto-Pause on Unpaid Overage

When a user's overage charges exceed their spending cap, pause their generation capability:

```typescript
async function handleOverageLimitReached(
  customerId: string,
  currentOverage: number,
  cap: number
) {
  // Update customer metadata to flag as paused
  await stripe.customers.update(customerId, {
    metadata: {
      generation_paused: "true",
      pause_reason: "overage_limit",
      paused_at: new Date().toISOString(),
    },
  });

  // Your app checks this flag before allowing new generations
  // Send notification
  await sendOverageLimitEmail(customerId, currentOverage, cap);
}
```

### Workflow 3: Automatic Tier Upgrade Suggestion

When a user would save money by upgrading, prompt them:

```typescript
/**
 * Calculate whether upgrading to the next tier would save money.
 * Run this at the end of each billing period or when usage crosses
 * a threshold.
 */
function shouldUpgrade(
  currentTier: "starter" | "pro" | "business",
  monthlyCreditsUsed: number
): {
  shouldUpgrade: boolean;
  currentCost: number;
  upgradedCost: number;
  savings: number;
} {
  const tiers = {
    starter: {
      base: 14.99,
      included: 150,
      overageRate: 0.15,
      next: "pro" as const,
    },
    pro: {
      base: 49.99,
      included: 600,
      overageRate: 0.12,
      next: "business" as const,
    },
    business: {
      base: 149.99,
      included: 2000,
      overageRate: 0.09,
      next: null,
    },
  };

  const current = tiers[currentTier];
  const currentOverage = Math.max(0, monthlyCreditsUsed - current.included);
  const currentCost = current.base + currentOverage * current.overageRate;

  if (!current.next) {
    return {
      shouldUpgrade: false,
      currentCost,
      upgradedCost: currentCost,
      savings: 0,
    };
  }

  const next = tiers[current.next];
  const nextOverage = Math.max(0, monthlyCreditsUsed - next.included);
  const upgradedCost = next.base + nextOverage * next.overageRate;

  return {
    shouldUpgrade: upgradedCost < currentCost,
    currentCost,
    upgradedCost,
    savings: currentCost - upgradedCost,
  };
}

// Example:
// A Starter user consuming 400 credits/month:
// Current: $14.99 + (400-150) * $0.15 = $14.99 + $37.50 = $52.49
// Pro:     $49.99 + (400-600) * $0.12 = $49.99 + $0     = $49.99
// Savings: $2.50/month -- marginal, but Pro also gives access to more models
```

---

## Revenue Analytics: Formulas and Dashboards

These are the metrics every AI video platform must track, with exact formulas.

### Core Metrics

**1. Average Revenue Per User (ARPU)**

$$\text{ARPU} = \frac{\text{Total Revenue}}{\text{Active Users}}$$

For a usage-based product, break this into components:

$$\text{ARPU} = \text{ARPU}_{\text{subscription}} + \text{ARPU}_{\text{overage}} + \text{ARPU}_{\text{credit packs}}$$

Target: $30-80/month for a prosumer AI video tool.

**2. Cost Per Generation (CPG)**

$$\text{CPG} = \frac{\text{Total Provider API Costs} + \text{Infrastructure Costs}}{\text{Total Generations}}$$

Track this broken down by model:

$$\text{CPG}_{\text{model}} = \frac{\sum C_{\text{provider, model}} + C_{\text{infra, model}}}{N_{\text{model}}}$$

Target: Track trend over time. Should decrease as you negotiate better API rates and optimize infrastructure.

**3. Margin Per Model**

$$M_{\text{model}} = 1 - \frac{\text{CPG}_{\text{model}}}{\text{Revenue per generation}_{\text{model}}}$$

This tells you which models make money and which lose money. Example dashboard:

| Model | Generations/mo | CPG | Rev/Gen | Margin | $ Contribution |
|---|---|---|---|---|---|
| Luma Ray3.14 | 25,000 | $0.14 | $0.33 | 58% | $4,750 |
| Kling 2.1 Std | 10,000 | $0.52 | $1.00 | 48% | $4,800 |
| Kling 2.1 Pro | 3,000 | $1.62 | $2.50 | 35% | $2,640 |
| Veo 3 | 1,500 | $2.42 | $3.50 | 31% | $1,620 |
| Flux (images) | 40,000 | $0.04 | $0.13 | 69% | $3,600 |
| **Total** | **79,500** | | | **49%** | **$17,410** |

**4. Credit Utilization Rate**

$$\text{Utilization} = \frac{\text{Credits Consumed}}{\text{Credits Purchased or Included}}$$

If utilization is too low (<60%), users feel they are wasting money and may churn. If utilization is too high (>95%), users are constantly running out and feeling constrained.

Target: 70-85% utilization.

**5. Overage Revenue as % of Total**

$$\text{Overage \%} = \frac{\text{Overage Revenue}}{\text{Total Revenue}} \times 100$$

If this is too high (>30%), your included credit amounts are too low and users feel nickel-and-dimed. If too low (<10%), you might be leaving money on the table.

Target: 15-25% overage revenue.

**6. Cost of Goods Sold (COGS) Ratio**

$$\text{COGS Ratio} = \frac{\text{Total API Costs} + \text{Infra Costs}}{\text{Total Revenue}}$$

For a healthy AI video SaaS:

$$\text{COGS Ratio} = 1 - \text{Gross Margin}$$

Target: 35-50% COGS ratio (50-65% gross margin).

**7. Customer Lifetime Value (LTV)**

$$\text{LTV} = \text{ARPU} \times \frac{1}{\text{Monthly Churn Rate}}$$

For a $50/month ARPU and 5% monthly churn:

$$\text{LTV} = 50 \times \frac{1}{0.05} = \$1,000$$

This means you can spend up to $300-400 to acquire a customer (targeting 3:1 LTV:CAC ratio).

**8. Revenue Per Generation (RPG)**

$$\text{RPG} = \frac{\text{Total Revenue}}{\text{Total Generations}}$$

This is the inverse perspective of CPG. Track RPG / CPG ratio:

$$\text{RPG/CPG Ratio} = \frac{\text{RPG}}{\text{CPG}} = \frac{1}{1 - \text{Gross Margin}}$$

At 55% margin: RPG/CPG = 2.22x (you earn $2.22 for every $1 you spend).

### Building the Analytics Dashboard

```typescript
// analytics.ts

interface BillingAnalytics {
  period: { start: Date; end: Date };
  revenue: {
    subscription: number;
    overage: number;
    creditPacks: number;
    total: number;
  };
  costs: {
    provider: Record<string, number>; // by model
    infrastructure: number;
    total: number;
  };
  usage: {
    totalGenerations: number;
    byModel: Record<string, number>;
    totalCreditsConsumed: number;
    totalCreditsAvailable: number;
  };
  users: {
    active: number;
    paying: number;
    free: number;
    churned: number;
  };
  derived: {
    arpu: number;
    cpg: number;
    rpg: number;
    grossMargin: number;
    cogsRatio: number;
    creditUtilization: number;
    overagePercent: number;
    marginByModel: Record<string, number>;
    ltv: number;
  };
}

async function calculateAnalytics(
  period: { start: Date; end: Date },
  db: DatabaseClient
): Promise<BillingAnalytics> {
  // Fetch raw data
  const revenue = await db.getRevenue(period);
  const costs = await db.getCosts(period);
  const usage = await db.getUsage(period);
  const users = await db.getUserMetrics(period);
  const churnRate = await db.getChurnRate(period);

  // Calculate derived metrics
  const arpu = revenue.total / users.active;
  const cpg = costs.total / usage.totalGenerations;
  const rpg = revenue.total / usage.totalGenerations;
  const grossMargin = 1 - costs.total / revenue.total;
  const creditUtilization =
    usage.totalCreditsConsumed / usage.totalCreditsAvailable;
  const overagePercent = revenue.overage / revenue.total;

  // Margin by model
  const marginByModel: Record<string, number> = {};
  for (const [model, genCount] of Object.entries(usage.byModel)) {
    const modelCost = costs.provider[model] || 0;
    const modelRevenue = (rpg * genCount); // Simplified
    marginByModel[model] =
      modelRevenue > 0 ? 1 - modelCost / modelRevenue : 0;
  }

  const ltv = arpu / churnRate;

  return {
    period,
    revenue,
    costs,
    usage,
    users,
    derived: {
      arpu,
      cpg,
      rpg,
      grossMargin,
      cogsRatio: 1 - grossMargin,
      creditUtilization,
      overagePercent,
      marginByModel,
      ltv,
    },
  };
}
```

---

## Pricing Psychology for AI Video

The way you present prices matters as much as the prices themselves. Here are the psychological principles that apply specifically to AI video billing.

### Credit Packs vs. Per-Generation Pricing

Consider these two presentations of the same price:

**Option A**: "Generate a video for $0.10"

**Option B**: "100 credits for $10" (where 1 video = 1 credit)

They are mathematically identical. But Option B converts better. Why?

1. **Mental accounting**: Credits create a separate "spending account" in the user's mind. Spending credits feels less painful than spending dollars. This is the same reason casinos use chips.

2. **Decoupling**: When spending dollars, each generation triggers a "is this worth $0.10?" evaluation. With credits, the payment happened at purchase time. Each generation feels "free" because the credits are already bought.

3. **Anchoring**: "\(10 for 100 credits" anchors to the round number (100) rather than the unit price (\)0.10). 100 feels like a lot. $0.10 per generation also feels cheap, but the 100 anchor is more memorable.

4. **Loss aversion reversal**: With per-generation pricing, each generation is a loss ($0.10 gone). With credits, NOT using them is the loss ("I paid for 100 credits, I should use them"). This drives engagement.

### The Power of Tier Naming

Research on SaaS pricing consistently shows that three-tier pricing with a highlighted "recommended" tier increases conversion to the middle tier by 20-40%.

For AI video, name your tiers to signal the user type, not the feature set:

**Good**: Free / Creator / Professional / Studio
**Bad**: Basic / Standard / Premium / Enterprise

"Creator" and "Professional" tell users which tier they belong in. "Standard" and "Premium" are generic and make users feel they need to compare feature lists.

### The "Penny Gap"

The difference between free and $0.01 is larger than the difference between $0.01 and $1.00, psychologically. This is why the free tier is critical: it gets users over the penny gap. Once they are paying anything, price sensitivity drops dramatically.

For AI video, the free tier should be:
- Generous enough to demonstrate value (30 credits = ~7 video generations)
- Limited enough that active users outgrow it within 1-2 weeks
- Restricted in model access (free gets Luma only, paid gets all models)

### Price Ending Effects

In SaaS:
- $9.99 outsells $10.00 by ~8% (the "left digit effect")
- $49 outsells $50 by ~5%
- $149.99 outsells $150 by ~3% (effect diminishes at higher prices)

Our tier prices ($14.99, $49.99, $149.99) follow this pattern deliberately.

### The Upgrade Prompt Timing

When should you prompt a user to upgrade? Not when they run out of credits (they are frustrated). Prompt when they are at **75% consumption and actively generating** -- they are engaged, productive, and receptive.

```
"You're on a roll! You've used 75% of your monthly credits with
10 days left in your billing period. Upgrade to Pro for 4x more
credits and access to premium models like Veo 3."
```

---

## Competitor Pricing Analysis

Let us analyze the pricing pages of major AI video platforms and reverse-engineer their cost structures and margins.

### Runway

**Pricing (as of February 2026):**
- Free: 125 credits/month
- Standard: $15/month (625 credits)
- Pro: $35/month (2,250 credits)
- Unlimited: $95/month (unlimited Standard generations, 5,000 credits for premium)

**Credit costs:**
- Gen-4 (10s, 720p): 100 credits
- Gen-4 Turbo (10s, 720p): 50 credits
- Image: 5 credits

**Analysis:**

At the Pro tier ($35/month for 2,250 credits):
- Effective price per credit: $0.0156
- Cost per Gen-4 10s generation (100 credits): $1.56
- Runway's likely API cost for Gen-4 (based on their infrastructure): ~$0.80-1.00

$$\text{Estimated margin} = 1 - \frac{0.90}{1.56} = 42\%$$

Runway's margins are moderate. They can afford this because their web app has high engagement (users return daily) and low churn.

### Pika

**Pricing (as of February 2026):**
- Free: 150 credits/day (resets daily)
- Standard: $10/month (700 credits/month)
- Pro: $30/month (2,100 credits/month)
- Ultra: $60/month (unlimited standard, 4,500 pro credits)

**Credit costs:**
- Standard video (3-4s): 10 credits
- High-quality video (4s): 20 credits

**Analysis:**

At Standard ($10/month for 700 credits):
- Effective price per credit: $0.0143
- Cost per standard video (10 credits): $0.143
- Pika's likely generation cost: ~$0.05-0.08

$$\text{Estimated margin} = 1 - \frac{0.065}{0.143} = 55\%$$

Pika's margins are higher because they use their own model infrastructure (lower costs) and price aggressively at the low end to attract consumers.

### Luma (API)

**Pricing (as of February 2026):**

Luma uses a different model for API users vs. consumer app users. API pricing is based on generation type and resolution.

- Ray3.14 Standard (720p, 5s): $0.04 per generation
- Ray3.14 HD (1080p, 5s): $0.08 per generation
- Ray3.14 Start+End Frame (1080p, 5s): $0.12 per generation
- Credit-based billing available at volume discounts

**Analysis:**

Luma's API pricing is the lowest in the market. At $0.04-0.12 per generation, they are almost certainly operating at thin margins or at a loss to gain market share, subsidized by their $900M in funding. This is a classic land-grab strategy: acquire developers now, raise prices later once switching costs are high.

### What Pricing Implies About Costs

We can use these public prices to estimate the industry's actual costs:

| Provider | User Price (5s video) | Estimated Cost | Implied Margin | Strategy |
|---|---|---|---|---|
| Runway Gen-4 | $1.56 | $0.80-1.00 | 36-49% | Premium brand, web-first |
| Pika Standard | $0.14 | $0.05-0.08 | 43-64% | Consumer growth |
| Luma Ray3.14 | $0.04-0.12 | $0.03-0.08 | 25-33% | API market share grab |
| Kling 2.1 | $0.50-1.60 | $0.30-0.80 | 40-50% | Quality positioning |
| Google Veo 3 | $2.40 | $1.00-1.50 | 38-58% | Enterprise, quality-first |

The convergence is clear: for standard quality 5-second video generation, the actual compute cost is **$0.03-0.10**, and providers are adding 30-65% margin on top. As compute costs continue to fall, expect prices to converge toward $0.01-0.05 per generation within 12 months.

---

## Summary: The Billing Architecture Decision Tree

```
Does your product have multiple AI models with different costs?
├── Yes → Use credit-based billing (normalizes across models)
│   ├── Do users generate daily?
│   │   ├── Yes → Subscription + included credits + overage
│   │   └── No → Credit packs (no recurring commitment)
│   └── Implement with Stripe Billing Meters (tiered pricing)
│
└── No (single model) → Per-generation pricing is simpler
    ├── Subscription + included generations + overage
    └── Implement with Stripe Billing Meters (flat per-unit pricing)

For all paths:
  ✓ Migrate from legacy usage records before March 31, 2026
  ✓ Use Stripe Workflows for automated notifications
  ✓ Track blended margin by model monthly
  ✓ Set credit prices with 18-month cost decline baked in
  ✓ Price based on user value, not your cost
```

The billing system is not a set-and-forget component. As model costs drop (see the [Ray3.14 cost collapse analysis](/2026/02/14/luma-ray314-cost-collapse.html)), you will need to regularly reassess your credit table, tier structure, and included credit amounts. Build the analytics to track this from day one.

---

*Previous post: [Flux 2.0 and the Image Generation Arms Race](/2026/02/12/flux-2-image-generation-arms-race.html)*

*Next post: [Luma Ray3.14: The 3x Cost Collapse](/2026/02/14/luma-ray314-cost-collapse.html)*
