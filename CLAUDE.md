# Blog Writing Style Guide

## Voice & Pedagogy
- Write as if a physicist with ML experience is explaining concepts — build everything from the ground up
- Define every single concept before using it; never assume the reader knows a term without explanation
- Do NOT explicitly state "as a physicist" or reference the author's background — let the rigor and style speak for itself
- Build intuition first, then formalize with math; always explain WHY something is true, not just THAT it is true
- Treat derivations as narratives — walk through each step, explain what's happening and why

## Mathematical Content
- Use LaTeX via MathJax (inline: `\(...\)`, display: `$$...$$`)
- NEVER use `$...$` for inline math — it conflicts with currency dollar signs
- Use plain `$` for currency (e.g., `$100M`, `$0.05/sec`) — MathJax won't touch it
- For posts with NO math at all, add `mathjax: false` to front matter
- Include rigorous derivations from first principles
- Use SVG diagrams for geometric and conceptual illustrations
- Include a Table of Contents for long posts

## Python Code & Figures
- Use numpy for all numerical computation
- Use matplotlib for all plots and figures
- ALL figures with math must use proper LaTeX formatting in labels, titles, annotations, and legends
  - Enable with `plt.rcParams['text.usetex'] = False` and use matplotlib's built-in mathtext: `r'$\alpha$'`, `r'$\nabla f(x)$'`, etc.
  - Axis labels: e.g., `r'Loss $L(N)$'`, `r'Parameters $N$'`
  - Titles: e.g., `r'Convergence of $\mathbb{E}[X_n]$ to $\mu$'`
  - Legends: e.g., `r'$\eta = 0.01$'`
  - Annotations: use `ax.annotate(r'$x^* = \arg\min f$', ...)`
- Use a clean, publication-quality style: `plt.style.use('default')` with dark backgrounds where appropriate
- Label every axis, include legends when multiple lines are plotted
- Use `fig, ax = plt.subplots()` pattern (not `plt.plot()` directly)

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
