---
layout: post
title: "A Technical Founder's Guide to VCs, Equity, and Not Getting Screwed"
date: 2026-02-19
category: business
---

You built something. Maybe it is a model that actually works, or an infrastructure tool that solves a real problem, or a product that users keep coming back to. Now someone in a fleece vest wants to give you two million dollars for a piece of your company, and you are suddenly swimming in terms you half-understand: liquidation preferences, anti-dilution provisions, SAFEs, 409A valuations, the option pool shuffle.

Here is the thing. The people on the other side of the table do this every day. They have done hundreds of deals. You have done zero. That asymmetry of experience is where founders get crushed — not because VCs are evil, but because this is their craft and you have not yet learned the rules of the game you are about to play.

This post is long. Deliberately so. It is the reference I wish existed before my first fundraise: everything from fund mechanics to cap table math to the subtle ways you can lose control of the company you started. Read it end to end, or use the table of contents to jump to what you need right now.

---

## Table of Contents

- [Part 1: Understanding VCs](#part-1-understanding-vcs)
  - [1. What VCs Actually Are](#1-what-vcs-actually-are)
  - [2. The VC Business Model](#2-the-vc-business-model)
  - [3. What VCs Look For](#3-what-vcs-look-for)
  - [4. Types of Investors](#4-types-of-investors)
- [Part 2: The Fundraising Process](#part-2-the-fundraising-process)
  - [5. When to Raise](#5-when-to-raise)
  - [6. How Much to Raise](#6-how-much-to-raise)
  - [7. Valuation](#7-valuation)
  - [8. Term Sheets](#8-term-sheets)
  - [9. SAFEs and Convertible Notes](#9-safes-and-convertible-notes)
- [Part 3: Equity with Co-founders, Employees, and Partners](#part-3-equity-with-co-founders-employees-and-partners)
  - [10. Founder Equity Splits](#10-founder-equity-splits)
  - [11. Employee Equity](#11-employee-equity)
  - [12. Cap Table Management](#12-cap-table-management)
  - [13. Advisor Equity](#13-advisor-equity)
- [Part 4: The Uncomfortable Truths](#part-4-the-uncomfortable-truths)
  - [14. Alignment and Misalignment](#14-alignment-and-misalignment)
  - [15. Control](#15-control)
  - [16. The Down Round](#16-the-down-round)
  - [17. Due Diligence Red Flags](#17-due-diligence-red-flags)
- [Part 5: Practical Advice](#part-5-practical-advice)
  - [18. Negotiation Tactics](#18-negotiation-tactics)
  - [19. References and Back-Channeling](#19-references-and-back-channeling)
  - [20. The Relationship After the Wire](#20-the-relationship-after-the-wire)

---

# Part 1: Understanding VCs

## 1. What VCs Actually Are

A **venture capital firm** is a financial intermediary. That sentence is boring but foundational, so let us unpack it.

There are three actors in the VC ecosystem. First, the **Limited Partners** (LPs). These are the people and institutions who actually have the money: university endowments (Harvard, Yale, Stanford), pension funds (CalPERS, the Ontario Teachers' Pension Plan), sovereign wealth funds, family offices, and wealthy individuals. LPs have hundreds of millions or billions of dollars and they need those dollars to grow. They allocate a small percentage — typically 5-15% of their total portfolio — to "alternative assets," which includes venture capital.

Second, the **General Partners** (GPs). These are the people whose names are on the door of the VC firm. They do not (mostly) invest their own money. Instead, they raise a **fund** — a pool of capital from LPs — and invest that fund into startups. A fund is a legal entity, usually a limited partnership, with a defined size and a defined lifespan (typically 10 years, with optional extensions).

Third, there is you: the **startup founder**. You receive the money. You build the thing. Ideally, you make everyone rich.

The economics work like this. The GP charges the LPs a **management fee**, typically **2% of the fund size per year**. On a \$100M fund, that is \$2M per year. Over a 10-year fund life, that is \$20M. This pays for salaries, offices, flights to Demo Day, and all the operational costs of running the firm. Then, when the fund generates returns — when startups get acquired or go public — the GP takes **carried interest** (or "carry"), typically **20% of the profits**. The LPs get the remaining 80%.

Let us make this concrete. Say a GP raises a \$100M fund. Over 10 years:

- Management fees: ~\$20M total (2% x \$100M x 10 years, though the fee basis often steps down after the investment period)
- The GP invests the remaining ~\$80M into startups
- Suppose the fund returns \$300M total to the LPs
- Profit = \$300M - \$100M = \$200M
- GP carry = 20% x \$200M = \$40M
- LPs get: \$300M - \$40M = \$260M (a 2.6x return on their \$100M)

Why does this matter to you? Because it tells you what drives VC behavior. The GP makes money in two ways: management fees (which are guaranteed, regardless of performance) and carry (which requires actual returns). A partner at a top firm might earn \$500K–\$2M per year from management fees alone. Carry is where the real wealth is — but only if the fund performs.

This creates an important dynamic: **VCs have a fiduciary duty to their LPs, not to you.** Their legal obligation is to maximize returns for the people who gave them money. You are the vehicle through which they fulfill that obligation. This is not cynicism. It is the structural reality of the relationship, and understanding it will save you from a lot of confusion about why your VC behaves the way they do.

---

## 2. The VC Business Model

Venture capital follows a **power law** distribution of returns. This is not a nice-to-have insight — it is the single most important thing to understand about how VCs think.

In a normal distribution (the classic bell curve), most outcomes cluster around the average. Power law distributions are radically different: a tiny number of outcomes generate almost all the value. This is the mathematical structure of venture returns.

Here is what a typical fund portfolio looks like. Out of, say, 30 investments:

| Outcome | Number of Companies | Return Multiple | Contribution to Fund |
|---------|-------------------|-----------------|---------------------|
| Total loss | ~10 | 0x | \$0 |
| Mostly dead (acqui-hire, fire sale) | ~10 | 0.1-0.5x | Negligible |
| Modest return | ~5 | 1-3x | Small |
| Good return | ~3 | 3-10x | Moderate |
| Home run | ~2 | 10-50x | Most of the fund's returns |
| Unicorn | 0-1 | 50-100x+ | Could return the entire fund alone |

Let us do the math. A \$100M fund (investing ~\$80M after fees) needs to return **at least \$300M** to be considered a good fund (a 3x net return to LPs). Most of the 30 investments will return nothing or nearly nothing. The fund's entire performance hinges on those 2-3 companies that generate 10x+ returns.

Now consider what this means for you as a founder seeking investment. If a VC invests \$3M into your company at a \$15M valuation (buying 20% ownership), they need your company to generate a return large enough to matter to their fund. If their fund is \$100M, they need your company to eventually be worth enough that their 20% (which will be diluted in later rounds — maybe down to 10%) represents a significant chunk of that \$300M target.

Say their stake gets diluted to 10% by the time you exit. For your company to return \$30M to the fund (10x on their \$3M investment), you need to exit at \$300M. That is a meaningful outcome but not fund-defining. For your company to "return the fund" — the holy grail for a VC — you need to exit at \$1B+ so their 10% is worth \$100M+.

This is why **VCs do not care about your "nice lifestyle business."** If you are building a company that will comfortably generate \$5M per year in profit and you want to run it for 20 years, that is a fantastic personal outcome. But it does nothing for the VC's fund. They need billion-dollar exits. They need you to swing for the fences.

This creates a fundamental **misalignment** that permeates the entire founder-VC relationship. You might be perfectly happy with a \$50M acquisition. Your VC would rather you turn that down and keep going, because a \$50M exit on their \$3M investment returns only \$5M to a \$100M fund — it barely moves the needle. They want you to keep playing the game for a shot at a \$500M or \$1B outcome, even if that means a higher probability of total failure.

Understand this math cold. It explains almost every confusing VC behavior you will ever encounter.

---

## 3. What VCs Look For

Ask a VC what they look for and they will recite some version of: "team, market, traction, and defensibility." This is true but incomplete. Let me break down what these actually mean in practice and then tell you what they will not say out loud.

**Team.** VCs are betting on people, especially at the early stages when the product barely exists. They want founders who are (a) deeply knowledgeable about the problem space, (b) relentlessly resourceful, and (c) able to recruit great people. Technical founders get a boost here — if you built the thing yourself, that signals capability and conviction. But VCs also want to see that you can communicate clearly, sell a vision, and lead a team. The "brilliant but cannot talk to humans" stereotype is not an asset. It is a risk factor.

**Market.** This is about the Total Addressable Market (TAM) — the total revenue opportunity if you captured 100% of the market. VCs want TAMs of \$1B+ because that is the only way to generate the returns their fund model requires. But here is the nuance: the best companies create or redefine markets. Uber's TAM was not "the taxi market" — it was "all urban transportation." Airbnb's TAM was not "cheap hotels" — it was "all short-term accommodation." If your market looks small today, you need a credible narrative for why it will be enormous tomorrow.

**Traction.** Evidence that the thing works and people want it. At the earliest stages, this might be user interviews, waitlist signups, or a working prototype with a few dozen users. At later stages, it is revenue, growth rate, retention, and unit economics. Revenue growth rate matters more than absolute revenue. A company doing \$20K/month MRR growing 20% month-over-month is more exciting to a VC than a company doing \$100K/month MRR growing 5% month-over-month, because the first company will overtake the second within a year.

**Defensibility.** Why will this be hard to copy? Network effects (the product gets better as more people use it), proprietary data (you have training data or datasets nobody else can get), technical moats (you solved a genuinely hard problem), regulatory advantages (you have licenses or approvals that take years to obtain), or switching costs (your users' data and workflows are so embedded in your product that leaving is painful). Pure technology is rarely a durable moat — someone can replicate most code in 6-12 months. But technology combined with data, distribution, or network effects can be extremely defensible.

Now here is what VCs will not tell you directly: they are, to a significant degree, **pattern matching**. They have seen hundreds of pitches. They have an internal model — often unconscious — of what a successful founder looks like, sounds like, and acts like. This model is built from their portfolio's winners and the broader ecosystem's successes.

The patterns they match on include: pedigree (Stanford/MIT, Google/Meta/Stripe alumni), previous startup experience (especially previous exits), domain expertise (you worked in the industry for 10 years before starting the company), and, frankly, demographic factors that are slowly but not yet fully corrected for. The warm introduction matters because it is a signal — someone the VC trusts is vouching for you, which reduces their perceived risk.

**Narrative** is underrated by technical founders. Your company is not just a product — it is a story. Why now? Why you? Why is this inevitable? The best pitches make the outcome feel like a foregone conclusion. The VC should walk away thinking "this is going to happen with or without me, and I want to be on the right side of it." That framing — inevitability — is more powerful than any metric you can show.

---

## 4. Types of Investors

Not all money is the same. Here is the landscape, roughly ordered by stage:

**Angel Investors.** Individuals investing their own money, typically \$10K-\$250K. Former founders, executives, domain experts. The best angels bring genuine expertise and connections. The worst bring nothing but opinions. Angels usually invest via SAFEs (we will cover these later) and expect less formality. They typically cannot lead a round or set terms.

**Pre-Seed Funds.** Small funds (\$10M-\$50M) that invest \$100K-\$500K at the earliest stages — sometimes just a team and an idea. Examples: Precursor Ventures, Hustle Fund. They are comfortable with extreme risk and minimal traction. They often focus on specific theses (developer tools, AI, climate) and can be very hands-on.

**Seed Funds.** Funds (\$50M-\$300M) investing \$500K-\$3M. This is where the fundraising process starts feeling "real." You will likely need a deck, some traction, and a clear plan. Examples: First Round Capital, Floodgate, Homebrew. They typically want to see product-market fit signals — users who love the product, even if there are not many of them yet.

**Series A Firms.** The big step up. Funds (\$300M-\$2B+) investing \$5M-\$20M. They want meaningful traction: typically \$1M+ ARR (Annual Recurring Revenue, which is just your monthly recurring revenue times 12) with strong growth. This is where governance gets real — they will want a board seat, information rights, and protective provisions. Examples: Andreessen Horowitz (a16z), Sequoia, Benchmark.

**Series B/C/Growth Firms.** Even larger checks (\$20M-\$100M+) for companies with proven business models that need capital to scale. The conversation shifts from "can this work?" to "how fast can this grow?" and "what are the unit economics?" Examples: General Catalyst, Tiger Global, Coatue.

**Corporate VC (CVC).** The investment arms of large corporations: Google Ventures, Salesforce Ventures, Intel Capital. CVCs can offer strategic value (distribution, partnerships, technical resources) but come with strings. Your competitor's corporate parent might get board observation rights or information about your roadmap. They may also lose interest if the parent company's strategy shifts. Use CVCs carefully: take their money when the strategic alignment is genuine and short-term, but do not depend on them.

The practical takeaway: match the investor to your stage. Pitching Sequoia when you have no revenue is (usually) a waste of everyone's time. Pitching an angel when you need \$10M is, too. Build relationships one stage ahead — talk to Series A investors when you are raising your seed, so they know your story when you are ready.

---

# Part 2: The Fundraising Process

## 5. When to Raise

Timing matters more than most founders realize. Raise too early and you will give away too much of your company for too little money at a low valuation. Raise too late and you might run out of cash before you close the round — and desperation is the worst negotiating position.

Here are rough benchmarks by stage. These are not rules — they are central tendencies that shift by market, sector, and era:

**Pre-Seed (raising \$250K-\$1M)**
- You have: a team, a clear problem statement, possibly a prototype
- Traction: user interviews, waitlist, early alpha users
- Valuation: \$3M-\$10M pre-money

**Seed (raising \$1M-\$4M)**
- You have: a working product with real users
- Traction: \$10K-\$50K MRR, or strong engagement metrics for non-revenue models
- Valuation: \$8M-\$20M pre-money

**Series A (raising \$5M-\$20M)**
- You have: product-market fit and a scalable go-to-market motion
- Traction: \$100K+ MRR, ideally \$1M+ ARR, with consistent month-over-month growth (15%+ for the strongest companies)
- Valuation: \$30M-\$100M pre-money

The general principle: **raise when you have leverage, not when you need money.** Leverage means you have strong momentum (growing fast), multiple interested investors (competition drives terms in your favor), or enough runway that you can walk away from a bad deal.

Raising too early is the more common mistake among technical founders. You have an idea, maybe some early code, and someone offers you money. The temptation is obvious. But if you raise \$500K at a \$4M pre-money valuation, you have given away 11% of your company before you have proven anything. If you waited six months, built the product, got 50 paying customers, and then raised \$1.5M at a \$12M pre-money valuation, you would give away 11% but have 3x the capital. The math strongly favors patience, as long as you can afford to be patient (i.e., you have savings, consulting income, or a grant to bridge the gap).

Raising too late is the other failure mode. If you have 3 months of runway left and you start fundraising, you are in trouble. Fundraising typically takes 3-6 months. VCs can smell desperation. If they know you are running out of money, they will either pass (why invest in a sinking ship?) or offer predatory terms (because they know you cannot say no).

---

## 6. How Much to Raise

The standard advice is to raise **18-24 months of runway.** Runway is the number of months you can operate before running out of cash. Here is how to calculate it:

**Monthly burn rate** = total monthly expenses - total monthly revenue

If you spend \$80K/month (salaries, rent, servers, software) and earn \$15K/month in revenue, your burn rate is \$65K/month. To get 18 months of runway, you need: \$65K x 18 = \$1.17M.

But you should plan for your burn rate to increase. You are raising money to grow, which means hiring, which means higher expenses. A reasonable approach: model your burn rate 6 months out (after you have made the hires you plan to make with the new capital) and use that number.

If your current burn is \$65K/month but you plan to hire 3 engineers (\$15K/month each fully loaded), your projected burn becomes \$110K/month. For 18 months of runway at the projected rate: \$110K x 18 = \$1.98M. Round up to \$2M.

There is a tension here. Raising more money gives you more runway, which reduces existential risk. But raising more money usually means either (a) giving away more equity at the same valuation, or (b) needing a higher valuation to limit dilution, which sets a higher bar for your next round.

The relationship between amount raised and valuation is roughly: investors typically want to own **15-25% of the company** in a given round. So if you raise \$2M and the investor wants 20% ownership, the implied post-money valuation is \$10M (and the pre-money is \$8M). If you raise \$4M and the investor still wants 20%, the post-money is \$20M (pre-money \$16M). A higher valuation is not free — it means your next round (Series A or B) needs to be at an even higher valuation, or you face a "down round" (covered later), which is painful.

Rule of thumb: raise enough to hit the milestones that will make your next round a no-brainer, plus a buffer. If you need \$1M ARR for a Series A, and you are at \$100K ARR, figure out what it takes to get to \$1M ARR (hiring, marketing spend, time) and raise for that plus 6 months of cushion.

---

## 7. Valuation

Valuation is where the math meets the vibes. Let us start with the mechanics and then talk about strategy.

**Pre-money valuation** is what your company is worth *before* the investment. **Post-money valuation** is what it is worth *after* the investment. The relationship is simple:

> Post-money = Pre-money + Investment Amount

Example: you negotiate a \$10M pre-money valuation and raise \$2M. The post-money valuation is \$12M. The investor owns \$2M / \$12M = **16.67%** of the company. You (and any existing shareholders) own the remaining 83.33%.

Another example to make sure this is crisp. If you have 10 million shares outstanding before the round, each share is worth \$1 (pre-money \$10M / 10M shares). The investor puts in \$2M, which buys 2M new shares at \$1 each. Now there are 12M total shares. Investor owns 2M / 12M = 16.67%. Your 10M shares are now worth \$10M / \$12M of the company = 83.33%.

**How valuations are set in practice:**

At the earliest stages (pre-seed, seed), valuation is largely a function of (a) comparable transactions (what are similar companies raising at?), (b) the quality of the team and idea, and (c) supply/demand (how many investors want in?). There is no formula. It is a negotiation.

At later stages (Series A and beyond), valuations become more quantitative. Common approaches:

- **Revenue multiples.** SaaS companies might be valued at 10-30x their ARR, depending on growth rate, market size, and retention. A company doing \$2M ARR growing 3x year-over-year might command a 25x multiple = \$50M valuation.
- **Comparable company analysis.** What did similar companies raise at recently? If three competitors raised Series A rounds at \$40-60M valuations, you are probably in that band.
- **DCF (Discounted Cash Flow).** Rare at early stages because the projections are pure fiction, but occasionally used at growth stages.

**Why a higher valuation is not always better.** This is counterintuitive. You might think "get the highest valuation possible to minimize dilution." But consider:

If you raise your seed at a \$20M pre-money valuation, your Series A needs to be at a significantly higher valuation — say \$60M+ — to avoid a "down round." A down round means your Series A valuation is *lower* than your seed valuation. This is devastating for several reasons: it triggers anti-dilution protections (which dilute founders, covered in the term sheet section), it demoralizes your team (their equity is worth less than they thought), and it signals to the market that something went wrong.

A seed valuation of \$12M with strong growth to a \$50M Series A is a much healthier trajectory than a \$20M seed followed by a flat or down \$20M Series A. Be ambitious but honest. The right valuation is the one that gives you enough capital with acceptable dilution and sets you up for a clean next round.

---

## 8. Term Sheets

The term sheet is a non-binding document that outlines the key terms of an investment. It is typically 5-10 pages and covers economics (how the money works) and control (who makes decisions). Let us go through every term that matters.

**Valuation and Price Per Share.** Already covered above. The term sheet will specify pre-money valuation, the investment amount, and the resulting price per share.

**Liquidation Preference.** This is the most important economic term after valuation, and the one founders most often misunderstand.

A liquidation preference determines who gets paid first — and how much — when the company is sold or liquidated. Standard terms give investors a **1x non-participating** liquidation preference. Here is what that means:

In a **liquidation event** (sale of the company, IPO, or winding down), the investor gets to choose: either (a) take their money back (1x their investment), or (b) convert their preferred shares to common shares and take their pro-rata share of the total proceeds. They choose whichever is higher.

**Example — 1x Non-Participating:**

- Investor puts in \$5M for 20% of the company
- Company sells for \$100M
- Option A: take \$5M back (1x preference)
- Option B: convert and take 20% of \$100M = \$20M
- Investor chooses Option B: \$20M
- Founders/employees split the remaining 80%: \$80M

Now change the exit price:

- Company sells for \$15M
- Option A: take \$5M back (1x preference)
- Option B: convert and take 20% of \$15M = \$3M
- Investor chooses Option A: \$5M
- Founders/employees split: \$10M

The liquidation preference protects the investor's downside. In a bad outcome, they get their money back before anyone else. In a good outcome, they convert and share proportionally. This is fair and standard. **Accept 1x non-participating without much negotiation.**

Now let us talk about **participating preferred**, which is much worse for founders:

With participating preferred, the investor gets their money back first (1x) AND then also participates in the remaining proceeds pro-rata. This is sometimes called "double dipping."

**Example — 1x Participating:**

- Investor puts in \$5M for 20% of the company
- Company sells for \$30M
- Step 1: Investor takes \$5M off the top (1x preference)
- Step 2: Remaining \$25M is split pro-rata. Investor gets 20% of \$25M = \$5M
- Investor total: \$5M + \$5M = \$10M (33% of proceeds, despite owning 20%)
- Founders/employees get: \$20M (67%, despite owning 80%)

At a \$100M exit:
- Step 1: Investor takes \$5M
- Step 2: Investor gets 20% of \$95M = \$19M
- Investor total: \$24M (24% of proceeds vs 20% ownership)
- Founders/employees: \$76M

Participating preferred is bad for founders at every exit price. **Push back hard on this.** If an investor insists, negotiate a cap (e.g., participating up to 3x, then converts to common). But the standard is 1x non-participating, and any deviation should require a concession elsewhere.

**Anti-Dilution Protection.** If the company raises a future round at a lower valuation (a down round), anti-dilution provisions adjust the conversion price of the existing investor's preferred shares, effectively giving them more shares for the same investment. There are two flavors:

**Full ratchet:** the conversion price drops to the new round's price, regardless of how much is raised. This is extremely aggressive and rare. Example: investor bought shares at \$10/share. The next round prices shares at \$5/share. Full ratchet reprices all the investor's shares to \$5, doubling their share count. This massively dilutes founders.

**Weighted average:** the conversion price is adjusted based on a formula that accounts for how many shares are issued in the down round relative to the total. This is much more founder-friendly and is the standard. The dilution is proportional to the severity of the down round, not a cliff.

**Accept weighted average. Reject full ratchet.** Full ratchet is a red flag that suggests the investor is either unsophisticated or adversarial.

**Board Seats.** The term sheet specifies how many board seats there are and who controls them. A typical post-seed structure is 3 seats: 2 founders, 1 investor. Post-Series A: 5 seats: 2 founders, 2 investors, 1 independent (mutually agreed). Board composition matters enormously — this is where control lives. More on this in Part 4.

**Protective Provisions.** These are veto rights that investors get regardless of board composition. Typical protective provisions require investor approval for:

- Selling the company
- Raising more money (issuing new shares)
- Changing the certificate of incorporation
- Taking on significant debt
- Paying dividends
- Changing the size of the board

These are standard and reasonable. But watch out for overreach — some investors try to add provisions requiring their approval for hiring/firing executives, changing the business model, or spending above a threshold. Push back on anything that micromanages operations.

**Pro-Rata Rights.** The right to invest in future rounds to maintain their ownership percentage. If an investor owns 15% and you raise a Series A, pro-rata rights let them invest enough in the Series A to still own 15% afterward. This is standard and generally fine — it means your existing investors are willing to double down. It can become a problem if too many small investors all exercise pro-rata, leaving little room for a new lead investor.

**Drag-Along Rights.** If shareholders holding a majority (typically 50%+) of shares vote to sell the company, drag-along rights force all shareholders to participate in the sale on the same terms. This prevents a minority shareholder from blocking an acquisition. Standard and reasonable, though negotiate for a high threshold (e.g., 60-70% of shares must approve).

**Tag-Along Rights (Co-Sale Rights).** If a founder sells their shares, other shareholders can "tag along" and sell a proportional amount on the same terms. This prevents founders from secretly cashing out while leaving investors stuck.

**Information Rights.** Investors get the right to receive regular financial reports (monthly or quarterly), annual budgets, and cap table updates. Standard. The main thing to watch: make sure information rights do not extend to competitive or sensitive data that could leak to the investor's other portfolio companies.

**Right of First Refusal (ROFR).** If a shareholder wants to sell their shares to a third party, existing shareholders (usually the company first, then investors) get the right to buy those shares first at the same price. This lets investors control who joins the cap table.

---

## 9. SAFEs and Convertible Notes

At the earliest stages (pre-seed and seed), you often do not want to negotiate a full term sheet. You want to get money in quickly and cheaply, without the legal fees and time of a priced round. Two instruments exist for this: **SAFEs** and **convertible notes.**

A **convertible note** is a loan. You borrow money from the investor, and instead of paying it back with interest, the loan converts into equity at your next priced round. Key terms:

- **Interest rate:** Usually 2-8%. The interest accrues and converts into additional equity.
- **Maturity date:** When the loan comes due if no conversion happens (typically 18-24 months). If the note matures without a conversion event, the investor can technically demand their money back. In practice, this is usually renegotiated.
- **Valuation cap:** A maximum valuation at which the note converts. If the cap is \$10M and your Series A is at \$50M, the note converts at the \$10M valuation, giving the investor more shares.
- **Discount:** A percentage discount on the Series A price. A 20% discount means the note converts at 80% of whatever the Series A investors pay.

A **SAFE** (Simple Agreement for Future Equity) was created by Y Combinator to simplify this further. It is not a loan — there is no interest, no maturity date, no repayment obligation. It is a contract that says: "I give you money now, and in the future when you raise a priced round, this money converts into equity."

SAFEs have two key terms:

- **Valuation cap:** Same as above — the maximum valuation at which the SAFE converts.
- **Discount:** Same as above — a percentage discount on the next round's price.

A SAFE can have a cap only, a discount only, or both (in which case the investor gets the more favorable conversion). The most common SAFE today is a **post-money SAFE with a valuation cap**, which is what YC uses. "Post-money" here means the cap represents the valuation *including* all SAFE money raised — this makes dilution more predictable.

**Conversion Example (Post-Money SAFE):**

You raise \$1M on a post-money SAFE with a \$10M valuation cap. Then you raise a Series A at a \$40M pre-money valuation.

- The SAFE converts as if the company were valued at \$10M (the cap) at the time of conversion
- \$1M / \$10M = the SAFE holders get 10% of the company (on a post-money basis at the time the SAFE was issued)
- The Series A then prices on top of this

If you had used a pre-money SAFE with a \$10M cap, the math is slightly different because the cap refers to the pre-money valuation excluding other SAFEs, which can create stacking effects. Post-money SAFEs are cleaner because each investor knows exactly what percentage they are buying.

**Conversion Example with Discount:**

You raise \$500K on a SAFE with a 20% discount and no cap. Your Series A prices shares at \$10/share. The SAFE converts at \$10 x (1 - 0.20) = \$8/share. So the \$500K buys 62,500 shares instead of the 50,000 shares a Series A investor would get for the same money.

**When to use SAFEs vs. Convertible Notes vs. Priced Rounds:**

- **SAFEs** for pre-seed and early seed when you want speed and simplicity. Low legal costs (\$0-\$5K).
- **Convertible notes** if investors insist (some institutional investors prefer the additional protections of a note). Legal costs: \$5-\$15K.
- **Priced round** for seed rounds of \$2M+ and all later stages. You need a full term sheet and preferred stock purchase agreement. Legal costs: \$15-\$40K+.

YC loves SAFEs because they align incentives in a clean way: the founder gets money fast, the investor gets a fair conversion, and nobody wastes time or money on legal negotiations at a stage where the company might pivot entirely.

---

# Part 3: Equity with Co-founders, Employees, and Partners

## 10. Founder Equity Splits

This is the conversation that ruins friendships. Let us make it precise instead of emotional.

**Equal splits** (50/50 for two founders, 33/33/33 for three) are the simplest and most common approach. The argument for equal: it signals trust, avoids resentment, and acknowledges that building a company is a long game where everyone's contribution evolves. The early research from Noam Wasserman at Harvard found that equal splits correlated with lower founder conflict.

**Unequal splits** make sense when the contributions are clearly asymmetric from day one. If one founder has been working on this for two years, built the prototype, and has deep domain expertise while the other is joining fresh, a 60/40 or 70/30 split may be warranted. The argument: equity should reflect contribution, risk assumed, and opportunity cost.

My take: default to equal unless there is a genuinely compelling reason not to. The resentment from an unequal split that the junior founder considers unfair is corrosive and slow-acting — it may not surface for two years, but it will surface.

**Vesting** is the mechanism that makes equity grants safe. Here is how it works.

A standard vesting schedule is **4 years with a 1-year cliff:**

- **Vesting period:** 4 years total. Your equity "vests" (becomes fully yours) over this period.
- **Cliff:** No equity vests for the first 12 months. On the 1-year anniversary, 25% of your total equity vests all at once. After that, the remaining 75% vests monthly (1/48 of the total each month, or equivalently 1/36 of the remaining grant).

Why does vesting exist? Because co-founders leave. Without vesting, a co-founder who contributes for 3 months and then quits still owns their full equity stake. You are left building the company while they collect value they did not earn. Vesting solves this.

**What happens when a co-founder leaves at month 8 (before the cliff)?**

They get nothing. Zero. Their unvested shares return to the company (or are never issued in the first place). This is the entire point of the cliff — it is a trial period. If the partnership does not work out in the first year, the departing founder walks away without equity.

**What happens when a co-founder leaves at month 18?**

They have passed the cliff, so 25% vested at month 12. Then 6 additional months of monthly vesting: 6/48 = 12.5%. Total vested: 25% + 12.5% = 37.5% of their original equity grant. The unvested 62.5% returns to the company.

**Critical point: founders should vest too.** Some first-time founders skip vesting for themselves, thinking "this is my company, I should not have to earn my shares." Investors will require founder vesting. More importantly, if your co-founder does not vest and leaves early, you have a serious problem.

---

## 11. Employee Equity

You need great people to build a great company, and equity is how startups compete with Google salaries. But employee equity is a minefield of tax implications and misunderstandings. Let us walk through it.

**Stock Options** are the most common form of employee equity. An option is the right — not the obligation — to purchase shares of the company at a fixed price (the **exercise price** or **strike price**) at some future date.

There are two types:

- **Incentive Stock Options (ISOs):** Available only to employees (not contractors or advisors). They receive favorable tax treatment: no ordinary income tax when you exercise, and if you hold the shares for 1 year after exercise and 2 years after grant, you pay long-term capital gains rates on the profit. However, the spread between exercise price and fair market value at exercise counts as income for **Alternative Minimum Tax (AMT)** purposes — more on this below.

- **Non-Qualified Stock Options (NSOs):** Available to anyone (employees, contractors, advisors). The spread at exercise is taxed as ordinary income. Simpler but more expensive tax-wise for the recipient.

The **exercise price** is set by a **409A valuation** — an independent appraisal of the company's fair market value. This is required by the IRS (the "409A" refers to Section 409A of the Internal Revenue Code). The 409A valuation is almost always much lower than the preferred stock price (the price VCs pay), because common stock lacks the liquidation preferences and other protections of preferred stock. A typical 409A might be 25-35% of the preferred price.

Example: your Series A prices preferred shares at \$10/share. The 409A might value common stock at \$3/share. Employees receiving options have an exercise price of \$3/share. If the company eventually goes public at \$50/share, the employee's profit is \$47/share.

**The Tax Trap (AMT).** If an employee exercises ISOs while the company is still private, the difference between the exercise price and the fair market value at the time of exercise is treated as income for AMT purposes, even though the employee has not actually received any cash. This can create a massive tax bill on paper gains.

Example: employee exercises 100,000 ISOs at \$1/share (exercise price) when the 409A fair market value is \$10/share. The spread is \$9/share x 100,000 = \$900,000 of AMT income. Depending on the employee's total income and state, this could generate a six-figure tax bill — for shares they cannot even sell yet because the company is private.

This is why **early exercise** (exercising options immediately when granted, before the 409A goes up) paired with an **83(b) election** is so valuable. I will cover 83(b) elections in Part 4.

**How much equity to give employees:**

This varies by stage, role, and seniority. Here are rough guidelines for early-stage companies (seed through Series A):

| Role | Stage: Seed | Stage: Series A |
|------|-------------|-----------------|
| VP/C-level (non-founder) | 1-3% | 0.5-1.5% |
| Director/Senior Engineer | 0.4-1.0% | 0.1-0.5% |
| Senior Engineer | 0.1-0.5% | 0.05-0.2% |
| Junior Engineer | 0.05-0.2% | 0.01-0.1% |
| Early key hire (#1-5) | 0.5-2.0% | — |

These are percentages of the company *at the time of the grant*. As the company raises more money and the total shares increase, these percentages get diluted (but hopefully the per-share value increases more than enough to compensate).

**The Option Pool Shuffle.** This is one of the most important concepts for founders to understand, and VCs will never bring it up proactively.

When a VC invests, the term sheet will specify that an **option pool** (a reserve of shares for future employee grants) must be created or expanded *before* the investment closes. The option pool is typically 10-20% of the post-money shares.

Here is why this matters. The option pool comes out of the founders' shares, not the investors' shares. It is created before the investment prices the shares, which means it dilutes the founders but not the investor.

**Example — The Option Pool Shuffle:**

Suppose you have 8M shares outstanding (all founder shares). A VC offers to invest \$4M at a \$16M pre-money valuation.

Without an option pool:
- Pre-money: \$16M / 8M shares = \$2/share
- New investor shares: \$4M / \$2 = 2M shares
- Post-money: 10M total shares
- Founders: 8M / 10M = 80%
- Investor: 2M / 10M = 20%

Now the VC says they want a 15% option pool created before closing:

- 15% of post-money (10M shares) = 1.5M shares for the option pool
- These 1.5M shares are carved out of the pre-money (from the founder side)
- The \$16M pre-money now covers: 8M founder shares + 1.5M option pool shares = 9.5M pre-money shares
- Price per share: \$16M / 9.5M = \$1.684/share (lower!)
- Investor gets: \$4M / \$1.684 = 2.376M shares
- Post-money total: 9.5M + 2.376M = 11.876M shares
- Founders: 8M / 11.876M = 67.4%
- Option pool: 1.5M / 11.876M = 12.6%
- Investor: 2.376M / 11.876M = 20.0%

Notice what happened. The VC still gets 20% — that was the deal. But the founders went from 80% to 67.4%, not 80%. The option pool diluted *only the founders*. The "real" pre-money valuation for the founders is effectively \$16M - the option pool value, which is closer to \$13.3M.

**How to push back:** negotiate the option pool size based on a realistic 18-month hiring plan, not an arbitrary percentage. If you only plan to hire 5 people in the next 18 months, you might only need a 10% pool, not 20%. Every percentage point of unnecessary option pool comes directly out of your pocket.

---

## 12. Cap Table Management

A **cap table** (capitalization table) is a spreadsheet that records who owns what in your company. Every share, every option, every SAFE, every convertible note. It is the definitive record of ownership.

Let us work through a full multi-round example to see how dilution plays out over the life of a company. This is the single most important piece of math in this post.

**Founding (Day 0):**

Two co-founders split equity 50/50. The company issues 10M shares total.

| Shareholder | Shares | Ownership |
|-------------|--------|-----------|
| Founder A | 5,000,000 | 50.0% |
| Founder B | 5,000,000 | 50.0% |
| **Total** | **10,000,000** | **100.0%** |

**Seed Round:**

The company raises \$2M on a \$8M pre-money valuation. The investors also require a 10% option pool.

- Option pool: 10% of post-money. Post-money = \$8M + \$2M = \$10M. So 10% = \$1M worth of shares.
- Price per share: we need to account for the option pool being created pre-money.
- Pre-money shares: 10M existing + new option pool shares. Let the option pool be X shares.
- Post-money shares: 10M + X + investor shares.
- Option pool must be 10% of post-money: X / (10M + X + investor shares) = 10%.
- Investor ownership: \$2M / \$10M = 20%.
- Solving: investors get 20%, option pool gets 10%, founders get 70%.
- Total post-money shares: let us call it S. Founders = 10M shares = 70% of S. So S = 10M / 0.70 = 14,285,714.
- Option pool: 10% x 14,285,714 = 1,428,571 shares.
- Investor shares: 20% x 14,285,714 = 2,857,143 shares.
- Price per share: \$2M / 2,857,143 = \$0.70/share.

| Shareholder | Shares | Ownership |
|-------------|--------|-----------|
| Founder A | 5,000,000 | 35.0% |
| Founder B | 5,000,000 | 35.0% |
| Option Pool | 1,428,571 | 10.0% |
| Seed Investors | 2,857,143 | 20.0% |
| **Total** | **14,285,714** | **100.0%** |

Each founder went from 50% to 35%. That is 30% relative dilution. It hurts, but the per-share value is now \$0.70 (vs. essentially \$0 before).

**Series A:**

The company raises \$10M at a \$30M pre-money valuation. The Series A lead requires expanding the option pool to 15% post-money. Currently the option pool has 1,428,571 shares. Some have been granted (say 500,000 to early employees), leaving 928,571 unallocated.

- Post-money: \$30M + \$10M = \$40M.
- 15% option pool post-money = need option pool to be 15% of total shares.
- Series A investor ownership: \$10M / \$40M = 25%.
- Founders + seed + existing option grants take up the rest: 100% - 25% - 15% = 60%.
- Current non-pool shares: 10M (founders) + 2,857,143 (seed) + 500,000 (granted options) = 13,357,143 shares.
- These represent 60% of post-money. Total post-money shares: 13,357,143 / 0.60 = 22,261,905.
- New option pool shares needed: 15% x 22,261,905 = 3,339,286 total pool. Subtract unallocated existing pool of 928,571 = 2,410,715 new shares.
- Series A investor shares: 25% x 22,261,905 = 5,565,476 shares.
- Price per share: \$10M / 5,565,476 = \$1.797/share.

| Shareholder | Shares | Ownership |
|-------------|--------|-----------|
| Founder A | 5,000,000 | 22.5% |
| Founder B | 5,000,000 | 22.5% |
| Granted Options (employees) | 500,000 | 2.2% |
| Option Pool (unallocated) | 3,339,286 | 15.0% |
| Seed Investors | 2,857,143 | 12.8% |
| Series A Investors | 5,565,476 | 25.0% |
| **Total** | **22,261,905** | **100.0%** |

Each founder is now at 22.5%. Their shares have not changed (still 5M each), but the total pie has grown. The good news: each share is now worth \$1.80 instead of \$0.70. Each founder's stake is worth \$9M (5M x \$1.80), up from \$3.5M post-seed.

**Series B:**

The company raises \$30M at a \$120M pre-money valuation. Option pool refreshed to 12% post-money.

- Post-money: \$150M.
- Series B ownership: \$30M / \$150M = 20%.
- Option pool: 12% of post-money.
- Everyone else: 100% - 20% - 12% = 68%.

Skipping the full share math and jumping to the ownership table:

| Shareholder | Ownership | Value at \$150M |
|-------------|-----------|----------------|
| Founder A | 15.3% | \$22.95M |
| Founder B | 15.3% | \$22.95M |
| Early Employees | ~3% | ~\$4.5M |
| Option Pool | 12.0% | \$18.0M |
| Seed Investors | 8.7% | \$13.05M |
| Series A Investors | 17.0% | \$25.5M |
| Series B Investors | 20.0% | \$30.0M |

Each founder started at 50% and is now at ~15% after three rounds. This is normal. If the company reaches a \$500M exit, each founder's 15% is worth \$75M. Dilution is not inherently bad — dilution with increasing per-share value is the whole game.

**Keep your cap table clean.** This means: no informal equity promises (everything in writing), no complex share structures that confuse future investors, proper documentation for every grant and transfer, and use proper cap table management software (Carta, Pulley, or AngelList) instead of a spreadsheet that one person maintains.

---

## 13. Advisor Equity

Advisors can be genuinely valuable or completely useless. The difference is whether they actually do the work.

**Typical advisory grants:**

| Level of Involvement | Equity | Vesting |
|---------------------|--------|---------|
| Light (quarterly call, occasional intros) | 0.1-0.25% | 2 years, monthly |
| Standard (monthly meeting, regular intros, strategic input) | 0.25-0.5% | 2 years, monthly |
| Heavy (weekly involvement, deep operational help, board observer) | 0.5-1.0% | 2-4 years, monthly |

Advisor equity should always vest. A common structure is a 2-year vesting schedule with no cliff (or a 1-month cliff). This is shorter than employee vesting because the advisory relationship is inherently less committed.

**When advisors are worth it:**

- They have deep domain expertise in your market and will make specific, actionable introductions to customers, partners, or investors.
- They have operational experience scaling the exact type of company you are building and will help you avoid specific pitfalls.
- They lend credibility that directly helps with fundraising or customer acquisition.

**When advisors are not worth it:**

- They want the title and the equity but will not commit to specific deliverables or a regular cadence.
- Their "network" is vague and their introductions are low-quality.
- They are collecting advisory positions at 10+ companies and cannot give you meaningful attention.
- You are giving them equity because they are famous, not because they are useful.

The FAST Agreement (Founder/Advisor Standard Template) by the Founder Institute is a good starting template for advisor agreements. Always put the terms in writing. Define the scope of work, the expected time commitment, and the vesting schedule.

---

# Part 4: The Uncomfortable Truths

## 14. Alignment and Misalignment

Let us return to the power law math from Part 1 and trace its implications for your day-to-day relationship with your investor.

**When your interests align with your VC's:**

- You both want the company to grow rapidly and become extremely valuable.
- You both want to hire the best people.
- You both want the company to raise subsequent rounds at increasing valuations.
- You both want a large, successful exit.

**When your interests diverge:**

**Outcome size.** You might be thrilled with a \$30M acquisition that makes you personally wealthy. Your VC is not. On their \$3M investment, a \$30M exit returns maybe \$6M (20% ownership). For a \$200M fund, that is noise. They would rather you reject the offer and keep swinging for a \$300M+ outcome, even if that means a higher probability of total failure. Your personal risk tolerance and the fund's required returns are in direct conflict here.

**Timeline.** Most VC funds have a 10-year life (sometimes with 2-3 year extensions). If they invested in your seed round in year 3 of their fund, they need you to exit by year 10-13. If you are building a slow-growing, durable business that will be most valuable in 15 years, there is a mismatch. The VC may push you toward a premature exit or an acquisition you do not want.

**Growth vs. profitability.** VCs typically want you to prioritize growth over profitability, because growth is what drives valuation multiples. "Spend the money. Grow faster. We will raise more." This advice can be correct (invest aggressively when you have product-market fit and a scalable channel) or catastrophic (burn cash on growth that does not stick while the market tightens). The VC's downside is capped at their investment — they lose \$3M, which is 1-3% of their fund. Your downside is your company, your equity, and years of your life.

**Competing portfolio companies.** Some VCs invest in multiple companies in the same space. They may frame this as "we are betting on the market." In practice, it means they have options and you have a competitor who knows your investor's thinking. Most term sheets include a clause about competitive investments — read it carefully.

---

## 15. Control

Control is not about ego. It is about who gets to make the decisions that determine the company's future. Many founders lose control gradually, through a series of individually reasonable decisions, and do not realize it until it is too late.

**Board composition** is the primary mechanism of control. The board of directors has the legal authority to hire and fire the CEO (yes, you), approve major transactions, set executive compensation, and make strategic decisions.

A typical progression:

- **Pre-seed/Seed:** 2-3 person board, all founders. No investor board seats.
- **Series A:** 5 person board: 2 founders, 2 investors, 1 independent.
- **Series B:** 5-7 person board: 2 founders, 2-3 investors, 1-2 independents.

At Series A, you have parity. At Series B, you might already be outnumbered. If the two investor board members and the "independent" (who the investors helped select) vote together, they control the board 3-2.

**Protective provisions** are the second mechanism. Even if you control the board, investors with protective provisions can veto key decisions: raising money, selling the company, changing the charter, issuing new shares. These provisions are standard, but their scope is negotiable. Narrow protective provisions give you operational freedom. Broad protective provisions give investors a chokehold.

**Voting rights.** Preferred stock often carries votes equal to the common stock it converts into, plus additional votes via protective provisions. In extreme cases, investors can engineer a situation where they have effective control despite owning a minority of shares.

**How founders lose control:**

1. They give up a board seat at each round without getting one back.
2. They agree to broad protective provisions because "the investor seems friendly."
3. They fail to negotiate for super-voting stock (dual-class shares where founder shares carry 10x voting power).
4. They bring in too many investors, each of whom gets some control rights, creating a Frankenstein governance structure.
5. The board fires them. This happens more often than anyone talks about. Investors can replace the CEO if they control the board.

**What to do about it:**

- Consider dual-class stock structures (like Google and Facebook used) where founder shares carry 10x voting power. Investors do not love this, but at early stages it is negotiable.
- Negotiate board composition carefully at each round. A 5-person board with 2 founders, 1 investor, and 2 mutually agreed independents is better than 2 founders, 2 investors, 1 independent.
- Keep protective provisions narrow and specific.
- Build a relationship with your independent board members. They are often the swing votes.

Some founders — most famously, Mark Zuckerberg — have taken less money or less favorable terms to maintain control. This is not irrational. Losing control of your company is an existential risk, and no amount of capital compensates for it if the people in control make decisions you believe are wrong.

---

## 16. The Down Round

A **down round** occurs when a company raises money at a valuation lower than its previous round. If your Series A was at a \$50M pre-money and your Series B is at a \$35M pre-money, that is a down round.

Why it is devastating:

**Anti-dilution kicks in.** If your Series A investors have weighted-average anti-dilution protection (and they do), their conversion price adjusts downward. They get more shares, which dilutes everyone else — primarily the founders and employees. If they have full ratchet (rare but it happens), the dilution is brutal.

**Employee morale collapses.** Your employees have stock options with an exercise price based on the old, higher 409A valuation. In a down round, the new 409A will be lower, meaning their existing options are "underwater" — the exercise price is higher than the current fair market value. Those options are currently worthless. Your best people will start interviewing at Google.

**Signal damage.** The startup ecosystem talks. A down round signals that something went wrong — you overpromised, the market shifted, or the business is not working. Future investors, customers, and recruits will all notice. It becomes harder to raise subsequent rounds, close enterprise deals, and hire senior talent.

**How to avoid it:**

- Do not raise at an inflated valuation in the first place. The number one cause of down rounds is an up round that was too high.
- Hit your milestones. If you told investors you would be at \$2M ARR in 18 months, get there.
- Manage your burn rate so you do not need to raise from a position of weakness.
- If you are heading toward a down round, consider alternatives: bridge financing from existing investors (a small note to extend runway), revenue-based financing, or cost-cutting to reach profitability.
- If a down round is unavoidable, negotiate hard on the anti-dilution adjustment. Push for broad-based weighted average (the most founder-friendly standard), and ask investors to waive or reduce anti-dilution to preserve the cap table.

---

## 17. Due Diligence Red Flags

When a VC decides to invest, they (or their lawyers) will conduct **due diligence** — a thorough review of your company's legal, financial, and operational status. Founders who have not been meticulous about corporate hygiene get caught here. The deal does not always fall apart, but it can result in worse terms, renegotiated valuations, or delayed closings.

Here is what VCs check and what you should have in order:

**Clean Cap Table.** Every share issued, every option granted, every SAFE signed should be properly documented and accounted for. No handshake deals. No "we promised our friend 2% but never put it in writing." Ambiguity on the cap table terrifies investors because it creates future legal liability.

**IP Assignment.** All intellectual property created for the company must be legally assigned to the company — not to individual founders or contractors. Every employee and contractor should have signed a **CIIA (Confidentiality and Invention Assignment Agreement)**. If your CTO built the core technology before the company was incorporated, there needs to be a formal IP assignment agreement transferring that code to the company. Without this, the company does not actually own its own product.

**Proper Incorporation.** For US-based startups seeking VC investment, the standard is a **Delaware C-Corporation.** Not an LLC, not an S-corp, not incorporated in your home state (unless that is Delaware). There are specific, practical reasons for this: Delaware has well-established corporate law, VCs are structured to invest in C-corps (not LLCs), and the convertible preferred stock that VCs require is a C-corp instrument. If you are incorporated as anything else, you will need to convert, which costs time and legal fees.

**83(b) Elections.** When founders receive shares subject to vesting, there is a tax event. Without an 83(b) election, the IRS treats each vesting tranche as taxable income at the then-current fair market value. If your shares are worth \$0.001 each at founding but \$5 each when they vest three years later, you owe income tax on \$5/share at each vesting date.

An **83(b) election** tells the IRS: "I want to be taxed on the full value of these shares *now*, at the current (very low) value, rather than as they vest." If you file within 30 days of receiving the shares, you pay taxes on \$0.001/share for all shares, even though most have not vested yet. If the shares increase in value, all future gains are taxed at capital gains rates when you eventually sell.

**This is a 30-day deadline with no exceptions.** Miss it and you cannot undo the damage. It is one of the most important administrative tasks at company founding. VCs will check that all founders filed their 83(b) elections. If you did not, it is a red flag (it suggests sloppy corporate governance) and it creates a personal tax liability for the affected founders.

**No Side Agreements.** No verbal promises of equity, no secret revenue-sharing deals with friends, no informal arrangements that could create a claim on the company's assets. Everything should be in the corporate records.

**Employment Matters.** Proper employment agreements for all employees, proper contractor agreements with IP assignment clauses, compliance with employment law (particularly in California, where non-competes are unenforceable and certain IP assignment provisions are limited by law).

**Corporate Records.** Board meeting minutes, written consents for major decisions, proper bylaws, and a current certificate of incorporation. These should be stored in a secure, accessible location (most lawyers use a virtual data room).

---

# Part 5: Practical Advice

## 18. Negotiation Tactics

Fundraising is a negotiation, and most founders approach it badly because they are too grateful to be getting funded. Here is how to do it right.

**Create FOMO (Fear of Missing Out).** The single most powerful lever in fundraising is competition. If multiple VCs want to invest, each one is afraid of losing the deal to the others. This drives better terms, faster decisions, and more founder-friendly dynamics.

How to create competition: run a **process**. Instead of talking to one VC at a time, sequentially, talk to 15-25 VCs in the same 2-3 week period. Schedule your first meetings in weeks 1-2, partner meetings in weeks 3-4, and aim to have term sheets in weeks 4-6. When one VC expresses strong interest, use it to accelerate others: "We are seeing strong interest from several firms and expect to have term sheets within two weeks. We would love to include you in the process."

This is not dishonest. It is standard practice. VCs expect it. They do the same thing on their side — talking to multiple companies in the same space before choosing which one to fund.

**Know your BATNA.** BATNA is the "Best Alternative To a Negotiated Agreement." It is what you do if you walk away from this deal. If your BATNA is "we run out of money in 3 months and die," your negotiating position is terrible. If your BATNA is "we keep growing with our existing revenue and raise next quarter at a higher valuation," your position is strong.

The best BATNA is not needing the money at all. Companies that are growing fast and have 12+ months of runway can afford to be picky. This is why the advice to raise when you do not need the money is so critical — it transforms your negotiating position.

**What is negotiable:**

- Valuation (within reason — the range is often narrower than founders think)
- Option pool size (argue for a realistic hiring plan, not an arbitrary 20%)
- Liquidation preference (push for 1x non-participating)
- Anti-dilution (push for broad-based weighted average)
- Board composition (fight for an extra independent seat)
- Protective provisions (narrow the scope)
- Pro-rata rights (limit to major investors only)

**What is generally not negotiable:**

- Standard information rights
- Standard drag-along/tag-along
- Right of first refusal
- The existence of a liquidation preference (some preference is standard in any preferred stock deal)

**When to use a lawyer:** Always. From the moment you receive a term sheet, have a lawyer who specializes in startup financing review it. This is not optional. A good startup lawyer costs \$15-40K for a financing round, and they will pay for themselves by catching unfavorable terms that would cost you millions later. Use a firm that has done hundreds of startup deals — Wilson Sonsini, Cooley, Gunderson Dettmer, Fenwick & West, or Goodwin Procter are the usual names. Do not use your uncle's real estate attorney.

Your lawyer reviews the terms and gives you advice. You make the decisions. Some founders defer entirely to their lawyers, which can kill deals (lawyers are trained to minimize risk, not to optimize for your specific situation). Other founders ignore their lawyers entirely, which is how you end up with participating preferred and full ratchet anti-dilution. Find the balance.

**One more thing:** never bluff about having competing term sheets if you do not. VCs talk to each other. If they discover you lied, the deal is dead and your reputation is damaged. You can create urgency without being dishonest: "We are in active discussions with several firms and expect to make a decision in the next two weeks" is both true and effective.

---

## 19. References and Back-Channeling

Here is something that catches first-time founders off guard: VCs will extensively research you before writing a check. This goes far beyond googling your name.

**What VCs do:**

- **Formal references.** They will ask you for 3-5 references. Choose these carefully — former bosses, co-workers, customers, and investors who will speak enthusiastically and specifically about your abilities.
- **Back-channel references.** This is the one that matters more. The VC will look at your LinkedIn, find mutual connections, and reach out to people you did *not* offer as references. They will talk to former colleagues, investors who passed on you, other founders in your space, and anyone in their network who has crossed paths with you. They are looking for patterns: does this person deliver? Are they honest? Can they recruit? Are they coachable?
- **Customer references.** If you have customers, the VC will call them. "How did you find this product? What would you do if it disappeared tomorrow? Have you recommended it to anyone?"
- **Technical diligence.** For deep-tech or AI companies, the VC may bring in a technical advisor to evaluate your technology, review your codebase, or assess your research claims.

**How to manage this:**

- Assume everything you have ever done professionally will surface. If you were fired from a previous job, had a co-founder dispute, or have a lawsuit in your past, disclose it proactively. VCs can forgive almost anything except being surprised.
- Prepare your references. Tell them the call is coming. Brief them on what the VC is likely to ask. This is not cheating — it is standard.
- Maintain relationships with former colleagues and bosses, even if you left on imperfect terms. A neutral reference is much better than a negative one.

**Do your own due diligence on VCs.** This is the part most founders skip, and they should not. The investor is going to be on your cap table (and possibly your board) for 7-10 years. You need to know who you are getting into bed with.

How to diligence a VC:

- **Talk to their portfolio founders.** Not just the ones on their website (those are the success stories). Find founders whose companies failed or were acqui-hired. How did the VC behave when things got hard? Did they help, disappear, or make things worse? Did they push for premature exits? Were they supportive of pivots?
- **Check their fund structure.** How old is the fund? A VC investing from a fund that is already 7 years old may push you toward a fast exit. How large is the fund? A partner at a \$2B fund will not spend much time on your \$2M seed investment.
- **Ask about their decision-making process.** How many partners need to approve? What is their typical timeline from first meeting to term sheet? A VC that takes 4 months to make a decision is not ideal when you are running a tight fundraise process.
- **Google them.** Search for their name plus "founder" or "startup" to find any public conflicts or concerns.

The best VC relationships are genuine partnerships. The worst are adversarial. Due diligence on both sides makes the good outcomes more likely.

---

## 20. The Relationship After the Wire

The money hits your bank account. The term sheet is signed. Now what?

The fundraising relationship gets all the attention, but the post-investment relationship is what actually matters. You are going to work with this investor for years. Here is how to make it productive.

**Board Meetings.** If your investor has a board seat, you will hold formal board meetings, typically quarterly. A good board meeting is not a performance — it is a working session. Structure it as:

1. Financial update (revenue, burn rate, runway, key metrics) — 15 minutes
2. Key decisions and strategic discussion — 45-60 minutes (this is the valuable part)
3. Asks (what do you need from the board?) — 15 minutes

Send the deck and financial materials 3-5 days before the meeting so board members come prepared. The worst board meetings are 90-minute slide presentations. The best are focused discussions about the 2-3 most important decisions facing the company.

**Investor Updates.** Even if your investor does not have a board seat, send regular updates. Monthly or quarterly, depending on stage. A good investor update includes:

- Key metrics (revenue, growth rate, customers, runway)
- Wins (closed a big customer, shipped a major feature, hired a key person)
- Challenges (what is not working, where you need help)
- Specific asks (introductions, advice, resources)

The "asks" section is the most important. VCs are most useful when you give them a specific, actionable request: "Can you introduce me to the VP of Engineering at Stripe?" is a good ask. "Help me with sales" is not. VCs have broad networks but limited time. Make it easy for them to help you.

**When to ask for help (and when not to).** VCs can be genuinely useful for: fundraising introductions (for future rounds), customer introductions, executive recruiting, strategic advice (especially if they have seen 10 other companies face the same decision), and navigating corporate development conversations (M&A interest from larger companies).

VCs are generally not useful for: day-to-day operational decisions, product strategy (unless they have specific domain expertise), technical architecture, or anything that requires deep, sustained engagement. They sit on 8-12 boards. They cannot go deep on your company the way you can.

**Managing conflict.** There will be disagreements. You will want to do something the investor does not support, or vice versa. The foundation of managing conflict is transparency. If you are going to make a decision the board might disagree with, surface it early, explain your reasoning, and listen to their concerns. You do not have to follow their advice, but you should demonstrate that you have heard it and considered it thoughtfully.

The worst thing you can do is surprise your investor with bad news. If you are going to miss a quarter, tell them in month 2, not month 3. If you are losing a key customer, flag it immediately. Bad news that is managed proactively builds trust. Bad news that is hidden or delayed destroys it.

**The long game.** Remember that the VC ecosystem is small and reputation-based. Your investor will be a reference for you for the rest of your career. Even if the company fails, how you behave — your transparency, your work ethic, your integrity — determines whether this investor will back you again, introduce you to their best founders, and speak well of you in the market.

The best founders treat their investors as partners: not friends, not bosses, not ATMs, but genuine partners with aligned (if not identical) interests. Set expectations clearly, communicate proactively, and ask for help without hesitation or ego. This is a long relationship. Invest in it accordingly.

---

## Conclusion

If you have read this far, you now have a working understanding of how venture capital works, how equity is structured, and where the traps are hidden. Let me leave you with the compressed version — the things that matter most:

1. **Understand the VC business model.** Their fund structure drives their behavior. They are not charities. They are not your friends. They are professional investors with specific return requirements. Understand those requirements and you will understand everything they do.

2. **Keep your cap table clean.** Proper incorporation (Delaware C-corp), 83(b) elections filed on time, IP assigned to the company, all equity grants documented. This is boring. It is also non-negotiable.

3. **Negotiate from strength.** Raise when you do not need the money. Run a process. Know your BATNA. Have a lawyer. Do not be grateful — be professional.

4. **Protect your control.** Board composition, voting rights, protective provisions — these are not boilerplate. They determine who makes the decisions that matter. Pay attention.

5. **Think in terms of dilution, not valuation.** A \$20M valuation means nothing if your terms are terrible. Focus on ownership percentage, liquidation preferences, and control rights. Those are what determine your outcome.

6. **Do the math.** Run the cap table forward. Model the liquidation waterfall. Understand what your equity is actually worth under different exit scenarios. The math is not hard. Ignoring it is expensive.

This game has rules. Now you know them. Go build something worth funding.
