---
layout: post
title: "The $2.5B AI Video Market: Who's Raising, Who's Shipping, and What's Next"
date: 2026-02-09
category: market
---

In November 2025, Luma AI raised $900 million. Let that sink in. A video generation startup — not a foundation model lab, not a cloud provider — raised nearly a billion dollars. The AI video market has gone from "interesting research" to one of the most heavily funded sectors in tech.

Here's where the money is going, who's actually making revenue, and what the landscape looks like for builders entering this market in 2026.

## The Funding Landscape

Major rounds in the last 12 months:

| Company | Round | Amount | Valuation |
|---|---|---|---|
| Luma AI | Series (Nov 2025) | $900M | — |
| Runway | Multiple rounds | $308M total | ~$4B |
| Synthesia | Series D | $180M | $2.1B |
| PixVerse | Multiple (Alibaba-led) | $60M+ | — |
| Hedra | Seed+ | $32M | — |

And then there are the big tech players — Google (Veo), OpenAI (Sora), and Meta (Movie Gen) — who aren't raising external funding but are investing billions in compute and research.

Total venture capital deployed into AI video generation in 2025 exceeded $1.5 billion. The market size projections vary: $716M in 2025 growing to $2.5B by 2032 (Grand View Research), or $1.07B growing to $1.97B by 2030 (alternative estimates). Either way, we're in the early innings of a multi-billion dollar market.

## Who's Actually Making Money

Funding is one thing. Revenue is another. Here's what we know:

**Kling / Kuaishou**: ~$100M revenue in the first three quarters of 2025. The clearest commercial success in the space, driven by Kuaishou's massive user base in China (700M+ MAU on their short video platform).

**PixVerse**: $40M ARR as of October 2025, with 16M monthly active users. Growing aggressively with Alibaba distribution.

**Runway**: Revenue figures aren't public, but their enterprise customer list (major studios, agencies, brands) and $4B valuation suggest meaningful revenue. They've been commercial longer than most competitors.

**Synthesia**: $180M Series D at $2.1B valuation implies significant revenue. Their focus on enterprise "talking head" video (corporate training, marketing) is a different market from creative video generation.

**Pika, Luma, Hailuo**: Consumer traction varies. All have millions of users but revenue figures aren't public. The consumer video generation market is proving harder to monetize than enterprise.

## The Revenue Models

Three billing models are emerging:

**Credit-based (most common)**: User buys credit packs, each generation consumes credits based on duration and quality. Pika, Runway, and most platforms use this. Simple to understand, predictable revenue per user. Downside: users hoard credits and churn when they run out.

**Subscription with included generations**: Fixed monthly fee with N included generations. Midjourney, Luma, and others use this. Predictable revenue, but the cost floor per subscriber limits how generous you can be. Works best when users generate consistently.

**Usage-based metered**: Pay per second or per generation with no upfront commitment. Most API-first platforms use this. Best for B2B/platform customers, worst for consumer UX (unpredictable costs).

For platform builders, the hybrid model is winning: subscription tiers with included credits, additional credit packs available for purchase, and API access for power users with usage-based billing. This covers casual users (subscription), power users (credit packs), and developers (API metering).

## What the Funding Signals

The capital flowing into AI video tells us several things:

**The models aren't commoditizing yet.** If they were, investors wouldn't be funding multiple model companies at billion-dollar valuations. There's still meaningful differentiation between Runway, Luma, Kling, and Veo on quality, speed, style, and capabilities.

**Enterprise is the monetization path.** Consumer video generation has millions of users but the average revenue per user is low. The big money is in enterprise: studios, agencies, brands, and education. Synthesia's $2.1B valuation on enterprise video proves this.

**Infrastructure plays are emerging.** Not everyone needs to build a model. Platforms that route across multiple models, add editing and workflow tools on top, and handle the billing/delivery/collaboration layer have a viable business without training their own weights.

**Asia is a legitimate market.** Kling's $100M revenue and PixVerse's 16M MAU are largely driven by Chinese and Asian markets. The AI video market is global from day one, unlike previous tech waves that started in the US and expanded later.

## What's Next

**Consolidation is coming.** With this many funded players and a market that's still under $1B in total revenue, some companies will merge and others will die. The models with the deepest integration into existing workflows (Adobe's Flux integration, Google's Workspace integration, Runway's studio partnerships) have the strongest moats.

**The API layer wins.** Most of the value in AI video will accrue to the application layer, not the model layer. Just as Stripe captured more value than any individual payment processor, the platforms that abstract multi-model video generation into a clean, billable service will capture more value than any individual model.

**Real-time changes the game.** When video generation becomes real-time (PixVerse R1 is the first credible demo), the product category expands from "video creation tool" to "interactive experience platform." This is a much larger market.

**B2B will outpace B2C.** Consumer video generation will grow, but the revenue-per-user math is challenging at current costs. B2B applications — marketing content, training videos, product demos, localization — have clearer ROI and higher willingness to pay.

## For Builders

If you're entering this market in 2026:

1. **Don't train your own model** unless you have a genuine technical advantage. Use APIs. The model landscape changes too fast to bet on a single architecture.

2. **Build the workflow, not the model.** The gap in the market is creative tools, project management, collaboration, and delivery — not raw generation capability.

3. **Plan for multi-model from day one.** No single model is best at everything. Your platform should route to the right model for each request.

4. **Price on value, not cost.** Your API costs will drop every quarter. Your prices should reflect what the output is worth to users, not what it costs you to generate.

5. **Go enterprise early.** Consumer is fun but enterprise pays. A single studio contract can be worth more than 10,000 consumer subscriptions.
