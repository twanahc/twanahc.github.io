# Blog Writing Style Guide

## Voice & Pedagogy
- Write as if a physicist with ML experience is explaining concepts — build everything from the ground up
- Define every single concept before using it; never assume the reader knows a term without explanation
- Do NOT explicitly state "as a physicist" or reference the author's background — let the rigor and style speak for itself
- Build intuition first, then formalize with math; always explain WHY something is true, not just THAT it is true
- Treat derivations as narratives — walk through each step, explain what's happening and why

## Mathematical Content
- Use LaTeX via MathJax (inline: `$...$`, display: `$$...$$`)
- Include rigorous derivations from first principles
- Add Python code simulations/plots (matplotlib) where they help build intuition
- Use SVG diagrams for geometric and conceptual illustrations
- Include a Table of Contents for long posts

## Post Format
- Front matter: `layout: post`, `title`, `date`, `category`
- Category options: `math`, `infra`, `business`, `tools`, `creative`, `landscape`
- Conversational but precise tone — no fluff, no hedging, no buzzwords
- Use `---` horizontal rules between major sections

## Site Structure
- Jekyll site at twanahc.github.io
- Layout: `_layouts/default.html` (has MathJax support)
- Stylesheet: `assets/style.css`
- Posts go in `_posts/` with filename format `YYYY-MM-DD-slug.md`
