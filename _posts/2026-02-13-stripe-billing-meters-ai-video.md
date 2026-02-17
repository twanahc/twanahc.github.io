---
layout: post
title: "Building Usage-Based Billing for AI Video: The Stripe Meters Migration You Can't Ignore"
date: 2026-02-13
category: infrastructure
---

If you're running a credit-based or per-generation billing model on Stripe, there's a hard deadline you need to know about: **March 31, 2026**. Stripe is removing the legacy usage records API entirely in API version `2025-03-31.basil`. If your billing uses the old metered subscription pattern, it breaks.

The replacement — Stripe Billing Meters — is actually better for AI video platforms. Here's why, and how to migrate.

## What's Changing

The old pattern: create a metered price, then submit usage records via `POST /v1/subscription_items/{id}/usage_records` after each generation. Stripe aggregates and invoices at the end of the billing period.

The new pattern: create a Billing Meter, then send meter events via the new API. Same concept, dramatically better infrastructure:

- **100,000 events per second** throughput (the old API couldn't handle high-frequency events)
- **30-second aggregation latency** (down from ~3 minutes)
- **Test clock support** for v2 meter events (faster testing cycles)

For an AI video platform processing thousands of generations per day, the old 3-minute aggregation latency meant users couldn't see their credit balance update in real-time. With 30-second latency, you can show near-real-time credit consumption in the UI.

## Why This Matters for AI Video Billing

AI video platforms typically bill in one of three ways:

**Credit packs**: User buys 100 credits, each generation costs N credits based on duration/quality. Simple, predictable.

**Per-second metered**: User pays per second of video generated. Maps directly to your API costs. More granular.

**Tiered subscriptions with overage**: Free/Pro/Ultra tiers with included generations, pay-per-use above the cap.

All three patterns benefit from Billing Meters:

**Credit packs** can use meters to track credit consumption and trigger automatic top-ups when the balance drops below a threshold. The new Workflows feature lets you build this logic directly in Stripe: "if credits < 10, send a notification" or "if credits = 0, pause the generation queue."

**Per-second metered** billing now works at scale. At 100K events/second, you can emit a meter event for every second of every generation without throttling. The 30-second aggregation means your dashboard shows near-live costs.

**Tiered with overage** can use meters to track included-unit consumption and automatically switch to overage pricing when the tier limit is hit. Stripe handles the math.

## Migration Steps

1. **Create a Billing Meter** for each billable dimension (video seconds, image generations, etc.):

```
stripe billing meters create \
  --display-name="Video Seconds" \
  --event-name="video_generation" \
  --default-aggregation.formula=sum
```

2. **Create new prices** backed by the meter (your old metered prices won't work):

```
stripe prices create \
  --currency=usd \
  --billing-scheme=per_unit \
  --unit-amount=5 \
  --recurring.interval=month \
  --recurring.usage-type=metered \
  --recurring.meter="mtr_xxx"
```

3. **Update your subscription creation** to use the new prices. You'll need to create new subscriptions or migrate existing ones.

4. **Replace usage record calls** with meter event calls:

```typescript
// Old (breaks March 31, 2026)
await stripe.subscriptionItems.createUsageRecord(itemId, {
  quantity: videoSeconds,
  timestamp: Math.floor(Date.now() / 1000),
});

// New
await stripe.v2.billing.meterEvents.create({
  event_name: 'video_generation',
  payload: {
    stripe_customer_id: customerId,
    value: String(videoSeconds),
  },
});
```

5. **Test with test clocks** — Stripe's v2 meter events now support test clocks, so you can simulate a full billing cycle without waiting for a real month to pass.

## The Workflows Opportunity

Stripe's new Workflows feature is particularly useful for AI video billing. You can build multi-step automation directly in Stripe:

- **Low credit warning**: When meter consumption approaches the plan limit, trigger an email or in-app notification
- **Auto-pause on zero credits**: When credits are exhausted, update a flag that your API checks before starting a new generation
- **Automatic tier upgrade prompts**: When a free user hits their limit, trigger a checkout session for Pro
- **Usage-based cost alerts**: Notify yourself when a user's monthly spend exceeds a threshold (helps catch abuse)

These are all things you'd otherwise build in your own backend. Having them in Stripe means less custom code and fewer race conditions around billing state.

## Timeline

- **Now**: Start building on the new Billing Meters API for any new features
- **Before March 31, 2026**: Migrate all existing metered subscriptions to the new API
- **API version `2025-02-24.acacia`**: Last version that supports legacy usage records. Pin to this if you need more time, but don't push past March.

The migration is mechanical but not trivial. Budget a sprint for it. The payoff is better real-time billing visibility for your users and dramatically higher throughput for meter events.
