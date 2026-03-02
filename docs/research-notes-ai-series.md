# Research Notes — AI Research Assistant Series

> Temporary file. Delete after articles are written.

## ArXiv Volume
- Total submissions: ~16,000-18,000/month (~500-600/day total)
- CS specifically: ~600-800/day (largest category, ~700/day is correct)
- Total cumulative papers: exceeded 2.4M by end of 2024
- Growth rate: ~10-15% per year

## Global Researchers
- UNESCO Science Report (2021 data): ~9M FTE researchers globally
- Breakdown: China ~2.3M, EU ~1.9M, USA ~1.5M, Japan ~0.7M, rest ~2.6M
- Including grad students: 15-20M+

## Market Data
- Traditional academic info services: ~$30B+ market
- Elsevier revenue: ~$3.5B/year
- Clarivate (Web of Science): ~$2.7B revenue
- Typical research university library spend: $5-15M/year
- AI-specific research tools: ~$500M-$1B (2024), growing rapidly

## Funding
- Elicit: $9M Series A (2023)
- Consensus: $11.5M Series A (2023)
- Perplexity: $73.6M Series B (2024 Q1, $520M val), $250M+ (2024 Q4, $9B+ val)
- Sakana AI: $200M+ Series A (2024)

## Tool Details

### Semantic Scholar
- 200M+ papers, billions of citation edges
- Free API: paper search, author search, citation graph, recommendations
- Rate limits: 100 req/s authenticated, 100/5min unauth
- TLDR (SciTLDR/BART), SPECTER embeddings (SciBERT)

### Elicit
- Q&A over papers, structured data extraction into tables
- PRISMA-style systematic review workflows
- Full-text PDF analysis, not just abstracts

### Consensus
- Consensus Meter: classifies papers as Yes/No/Possibly on claims
- Best for empirical yes/no questions
- Available as ChatGPT plugin

### ResearchRabbit
- "Spotify for papers" — seed papers → recommendations
- Co-citation analysis + semantic similarity
- Zotero integration, free

### Connected Papers
- Similarity graph (NOT just citations)
- Based on co-citation and bibliographic coupling
- Force-directed layout, color by year, size by citations

### Sakana AI Scientist
- Autonomous loop: idea → code → experiment → paper → peer review
- Cost: ~$15/paper (Sonnet 3.5)
- Automated reviewer: ~65% agreement with human ICLR reviewers
- Limitations: variations on known themes, small-scale compute, no physical experiments

### ChemCrow
- GPT-4 + 18 chemistry tools
- Nature Machine Intelligence (2024)
- Reaction planning, safety checks, molecular property prediction
- ReAct-style agent loop

### STORM (Stanford)
- Multi-perspective research → Wikipedia-style articles
- Simulates different "experts", question-asking approach
- Open source

### GPT-Researcher
- Planner → researcher → reviewer → writer agents
- LangChain/LangGraph, 10K+ GitHub stars
- Open source

### ChatGPT Deep Research
- Launched Feb 2025
- Uses o3 model variant
- 5-30 minutes, browses dozens to hundreds of pages
- Multi-hop research, structured output with citations

### Perplexity Pro
- Search-augmented LLM, inline citations
- Pro Search: multi-step with clarifying questions
- Spaces, Pages features
- $20/month, multiple model backends

## Citation Accuracy
- Without RAG: 30-70% fabrication rate
- With RAG: 5-15% fabrication, but misattribution errors
- Perplexity: 60-80% citations accurately support claims

## Hallucination Rates
- Summarization: 3-15% contain hallucinated facts
- Open-ended: 15-30%+ factual claims may be inaccurate
- Medical Q&A: 80-90% accuracy (10-20% error)
- ChemCrow with tools: ~5% error (vs ~30% base GPT-4)

## Claude Code Features

### CLAUDE.md
- Persistent instructions injected into system prompt
- Layered: repo root → directory-level → user-level
- Walks up directory tree for monorepo support

### Hooks
- PreToolUse, PostToolUse, Notification
- Configured in .claude/settings.json or settings.local.json
- Matcher regex against tool names (Bash, Edit, Write, etc.)

### MCP
- Model Context Protocol — universal interface for external tools
- Client-server architecture
- Official servers: filesystem, github, postgres, sqlite, brave-search, puppeteer, memory
- Configured via CLI or settings.json

### Skills
- Reusable workflows invoked via /slash-commands
- Built-in: /commit, /review-pr
- Custom skills can be defined

### Subagents
- Separate Claude instances for specialized tasks
- Own context, specialized system prompts
- Good for parallel work, context isolation
