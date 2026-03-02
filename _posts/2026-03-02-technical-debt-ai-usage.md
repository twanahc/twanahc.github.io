---
layout: post
title: "Technical Debt in the Age of AI: The Compounding Cost of Code You Didn't Write"
date: 2026-03-02
category: infra
---

You shipped a feature in two hours. Cursor generated the endpoint, Claude wrote the validation logic, you glued together a retry mechanism from a snippet that looked reasonable, and the tests passed. Product was thrilled. The PR merged at 4 PM and was live by 5. A month later, that feature is the source of 40% of your production incidents. The retry logic silently swallows a class of errors that only surface under load. The validation has three branches nobody on the team can explain --- they work, empirically, but nobody knows *why* they work, and nobody wants to touch them because the last person who tried caused a two-hour outage.

Technical debt is the most abused metaphor in software engineering. Everyone uses it. Almost nobody defines it precisely. The original formulation, from Ward Cunningham in 1992, was narrow and deliberate: you ship code that you know is not the ideal design, because shipping now and refactoring later is a rational trade-off --- the same way taking on financial debt to capture an opportunity is rational if you have a plan to pay it back. What most people call technical debt is not that. It is code nobody understands, written under pressure, with no plan to revisit. That is not debt. That is a mess. The distinction matters because the remediation strategies are completely different.

This post defines technical debt rigorously, breaks it into the categories that actually matter for engineering decision-making, and then asks the question that matters right now: what happens when AI tools generate most of your code?

The thesis is straightforward. AI tools do not create a new kind of debt. They accelerate the accumulation of every existing kind, while simultaneously making the debt harder to detect because the code *looks* right. Syntactically clean, well-commented, plausible. The surface quality goes up. The structural quality --- the part that determines whether your system survives contact with production at scale --- is uncorrelated with how the code reads.

---

## Table of Contents

1. [What Technical Debt Actually Means](#what-technical-debt-actually-means)
2. [The Mathematics of Debt Accumulation](#the-mathematics-of-debt-accumulation)
3. [How AI Tools Change the Debt Equation](#how-ai-tools-change-the-debt-equation)
4. [The Six Patterns of AI-Generated Debt](#the-six-patterns-of-ai-generated-debt)
5. [The Detection Problem](#the-detection-problem)
6. [Quantifying the Cost: When Debt Kills You](#quantifying-the-cost-when-debt-kills-you)
7. [Managing AI-Generated Debt](#managing-ai-generated-debt)
8. [Conclusion](#conclusion)

---
