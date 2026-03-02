---
layout: post
title: "The AI Research Assistant: Landscape, Architecture, and the Road to Computational Scientific Discovery"
date: 2026-03-09
category: landscape
---

*This is Part 1 of a 3-part series. **Part 1: Landscape** | [Part 2: Claude Code Best Practices](/posts/2026-03-10-claude-code-production-best-practices/) | [Part 3: Building the Assistant](/posts/2026-03-11-building-ai-research-assistant-claude-code/)*

A computational biologist at a mid-size research university opens her laptop on Monday morning. Over the weekend, 2,100 papers were posted to ArXiv across her areas of interest: machine learning, computational neuroscience, and statistical methods. She has five days before the next batch. If she reads deeply --- working through derivations, checking methodology, assessing whether results generalize to her own work --- she can absorb maybe four papers this week. Five if she skips lunch. That leaves 2,096 papers unread, any one of which might contain the technique that solves the problem she has been stuck on for three months.

This is not a time management problem. It is not a search problem. Google Scholar, Semantic Scholar, and ArXiv search will find papers matching her keywords with reasonable precision. The problem is that keyword matching does not tell her which papers *matter* --- which ones introduce methodology shifts relevant to her work, which ones contradict the assumptions underlying her current approach, which ones apply techniques from adjacent fields that she has never thought to look at. The bottleneck is not access to information. It is the cognitive bandwidth to synthesize information across hundreds of papers and extract the patterns that drive research forward.

The thesis of this article is direct. An AI research assistant is not a search engine with a chatbot bolted on top. It is a cognitive augmentation layer that must handle five distinct capabilities: literature synthesis, hypothesis generation, experimental design, code-assisted analysis, and structured knowledge management. The market today is fragmented, with most tools solving one or two of these capabilities well and ignoring the rest. Nobody has the complete loop. Understanding why --- and what the complete architecture looks like --- is the first step toward building one.

---

## Table of Contents

1. [The Research Bottleneck](#the-research-bottleneck)
2. [What Exists Today](#what-exists-today)
3. [The Five Capabilities](#the-five-capabilities)
4. [Architecture of a Research Assistant](#architecture-of-a-research-assistant)
5. [Market Analysis](#market-analysis)
6. [The Hard Problems](#the-hard-problems)
7. [Where This Series Goes](#where-this-series-goes)

---

## The Research Bottleneck

ArXiv receives roughly 700 new submissions per day in computer science alone. Across all fields --- physics, mathematics, quantitative biology, statistics, electrical engineering, economics --- the platform processes between 16,000 and 18,000 papers per month, adding to a corpus that has exceeded 2.4 million papers total. And ArXiv is one platform. PubMed indexes over 1.5 million new biomedical articles per year. The total volume of scientific output globally is estimated at over 3 million peer-reviewed papers annually.

A researcher engaged in deep reading --- working through proofs, evaluating experimental methodology, assessing whether conclusions generalize beyond the specific dataset and configuration tested --- can process three to five papers per week. "Deep reading" here means more than skimming an abstract. It means understanding the mathematical framework, checking the experimental design for confounds, and mentally connecting the results to your own work. This is the rate-limiting cognitive step that no reading speed improvement fixes.

The arithmetic is stark. A researcher in machine learning faces approximately 700 new papers per day in her field and can deeply process maybe 4 per week. That is a ratio of roughly 1,200 to 1. The gap is not closing. Publication volume grows at 10 to 15 percent per year. Human cognitive bandwidth does not.

What gets missed in this gap is not random noise. Cross-domain connections are the first casualty --- a technique developed in computational chemistry that would accelerate training of the specific neural network architecture you are working on, except you never read computational chemistry papers. Contradictory findings go unnoticed --- three papers confirm a result, but a fourth published last month shows it fails under the exact conditions of your experiment, and you will not find it because it uses different keywords. Methodological innovations spread slowly --- a new statistical test that eliminates a confound in your analysis exists, but it was published in a journal you do not follow, cited by authors you have not heard of.

### How the Bottleneck Shifted

The nature of the research bottleneck has shifted three times, and each shift changed what tools researchers needed.

**Pre-internet: The access problem.** Before digital repositories, finding relevant work meant physically visiting libraries, browsing journal shelves, following citation chains by hand, and attending conferences. The bottleneck was knowing that a relevant paper *existed*. The tools that solved this were indexes, abstracts journals, and eventually digital databases like MEDLINE (1971) and INSPEC.

**Pre-search engines: The discovery problem.** Digital databases put papers online, but finding the right ones required knowing the right keywords and the right databases to search. A paper about "attention mechanisms" published in a natural language processing venue would not surface when searching for "selective information routing" in a computer vision context, even though the underlying technique was identical. The tools that solved this were Google Scholar (2004), Semantic Scholar, and modern full-text search engines that rank by relevance rather than exact match.

**Now: The synthesis problem.** Discovery is largely solved. A well-constructed search query will surface most of the relevant literature within a few pages of results. The new bottleneck is synthesis --- integrating information across many papers to identify patterns, consensus, contradictions, and gaps. Current search tools return papers. They do not tell you what those papers *collectively mean*.

### What Synthesis Actually Means

Synthesis is not summarization. A summary tells you what one paper says. Synthesis tells you what many papers *mean together*. The distinction is critical because it determines the technical requirements of a research assistant.

**Consensus detection:** Papers A, B, and C all report that technique X improves performance by 15 to 20 percent on benchmark Y. This convergence across independent studies is evidence that the result is robust, not a statistical fluke.

**Contradiction detection:** Paper D, published three months after A, B, and C, reports that technique X actually *degrades* performance when applied to data with property Z. This is not a refutation --- it is a boundary condition. The research assistant needs to flag both the consensus and the exception.

**Gap identification:** Twenty papers study technique X on benchmarks Y1 through Y5. No paper tests it on domain Z, which has structural similarities to Y3 but different noise characteristics. This gap is a research opportunity. Identifying it requires mapping the space of what *has been tested* against the space of what *could be tested*.

**Methodology evolution:** The standard approach to problem P shifted from method M1 (2020) to M2 (2022) to M3 (2024). Understanding *why* each transition happened --- what limitation of the predecessor each new method addressed --- tells you where M4 is likely to come from.

None of these operations can be performed by reading one paper at a time. They require holding multiple papers in context simultaneously and reasoning across them. This is precisely the cognitive task that does not scale with publication volume, and precisely the task that current tools do not address.

---

## What Exists Today

The landscape of research-oriented AI tools falls into three categories: literature-focused tools that help you find and organize papers, AI-native research agents that attempt autonomous scientific workflows, and general-purpose LLMs that researchers adapt to their needs. Each category solves part of the problem. None solves it completely. Understanding what each tool does well --- and where each falls short --- reveals the architecture that a complete research assistant requires.

### Literature Tools

**Semantic Scholar** (Allen Institute for AI) is the infrastructure layer that many other tools build on. It indexes over 200 million academic papers across all fields and exposes a free API with endpoints for paper search, author lookup, citation graph traversal, and paper recommendations. The citation graph contains billions of edges. Two features distinguish it from Google Scholar: SPECTER embeddings, which are document-level vector representations trained on citation signal (papers that cite each other get similar embeddings, enabling semantic rather than keyword-based search), and TLDR summaries, single-sentence paper descriptions generated by a fine-tuned language model. The API handles 100 requests per second for authenticated users. What Semantic Scholar does not do: it does not synthesize across papers, generate hypotheses, or reason about what you should read next in the context of your specific research question.

**Elicit** (formerly Ought) is a research assistant focused on systematic literature review. You enter a natural language research question --- "What is the effect of sleep deprivation on working memory in adults over 60?" --- and Elicit finds relevant papers, then extracts structured data from each one. The extraction capability is the core differentiator: you define columns (sample size, methodology, effect size, population demographics) and Elicit fills a structured table by reading each paper and pulling out the relevant data points. This turns unstructured papers into a queryable dataset. Elicit supports PRISMA-style screening workflows for systematic reviews and can process uploaded PDFs at the full-text level, not just abstracts. Where it falls short: Elicit is a data extraction tool. It does not generate hypotheses, design experiments, or maintain a knowledge graph across sessions.

**Consensus** takes a different approach: it measures scientific agreement. For empirical yes-or-no questions --- "Does creatine supplementation improve exercise performance?" --- Consensus classifies each relevant paper's finding as supporting, opposing, or neutral on the claim, then shows a percentage breakdown. The Consensus Meter is their signature feature. The classifier is trained on paper findings and searches across approximately 200 million papers (via a Semantic Scholar partnership). It works well for well-posed empirical questions with measurable outcomes. It works poorly for open-ended, theoretical, or methodology questions, which constitute most of actual research.

**ResearchRabbit** has been described as "Spotify for research papers." You seed it with papers you find interesting, and it recommends related papers using a combination of co-citation analysis (papers frequently cited together are likely related) and semantic similarity. It also builds author network visualizations and sends alerts when new papers matching your interests are published. It integrates with Zotero for reference management and is free to use. ResearchRabbit solves the discovery problem --- finding papers you did not know to search for --- but does not help with synthesis.

**Connected Papers** builds a visual similarity graph around a seed paper, and its construction is worth understanding because it reveals a general principle. The graph does *not* simply show citation links. It computes a similarity score between papers based on co-citation and bibliographic coupling --- the overlap between the papers that cite them and the papers they cite. Two papers can be connected in the graph even if they never cite each other, as long as they occupy a similar position in the citation network. The graph uses a force-directed layout where strongly related papers cluster together, with color coding by publication year and node size by citation count. This reveals the intellectual neighborhood of a paper in a way that a citation list cannot.

### AI-Native Research Tools

These tools attempt to automate parts of the research process itself, not just the literature search.

**The AI Scientist** (Sakana AI, August 2024) is the most ambitious attempt at autonomous research. It implements a complete research loop: generate a research idea by reviewing literature and brainstorming, write code to test the idea, run the experiment, analyze results, write a full scientific paper in LaTeX with figures and citations, then run an automated peer review on its own paper. The system produced papers on diffusion models and language model fine-tuning at a cost of approximately $15 per paper using Claude 3.5 Sonnet as the backbone. Its automated reviewer achieved roughly 65 percent agreement with human reviewers on accept-or-reject decisions at ICLR-level papers. The limitations are instructive: it generates variations on known themes rather than genuinely novel ideas, its experiments are limited to small-scale compute (single GPU, short training runs), it cannot perform physical experiments, and its generated papers sometimes contain errors in reasoning or hallucinated references. In one run, the agent modified its own training code in unintended ways. The AI Scientist is a proof of concept that the full research loop *can* be automated. It is not yet evidence that it *should* be.

**ChemCrow** (published in Nature Machine Intelligence, 2024) takes the opposite approach: instead of trying to be a general research agent, it is a domain-specific agent built on GPT-4 and augmented with 18 expert-designed chemistry tools. These include molecular property prediction, reaction planning and retrosynthesis, safety assessment (toxicity, explosiveness, regulatory status), and patent and literature search. ChemCrow can plan multi-step organic syntheses, assess the safety of proposed reactions, and suggest modifications to improve molecular properties. Expert chemists evaluating its outputs rated them as largely correct and useful for drug discovery and materials science. The architecture is a ReAct-style agent loop: the LLM decides which tool to call, processes the result, then decides the next step. ChemCrow demonstrates that domain-specific tool augmentation dramatically reduces the error rate of LLMs on scientific tasks --- from roughly 30 percent error for base GPT-4 in chemistry to roughly 5 percent with tools.

**STORM** (Stanford NLP, 2024) generates long-form, Wikipedia-style articles on a given topic. Its architecture is interesting: it simulates multiple "experts" who each bring different perspectives to the topic, then has them conduct independent internet research by asking targeted questions from their respective viewpoints. The different perspectives engage in simulated dialogues to resolve contradictions and build comprehensive understanding. Finally, it synthesizes everything into a structured, cited article. Human evaluators rated STORM-generated articles as comparable to Wikipedia articles on topic coverage and organization, though lower on factual accuracy. STORM shows that multi-perspective research produces more comprehensive coverage than single-pass approaches.

**GPT-Researcher** is an open-source autonomous research agent that decomposes a research question into sub-questions, dispatches separate search agents for each, and then synthesizes results into a report. Built on LangChain and LangGraph, it uses a planner-researcher-reviewer-writer architecture: the planner breaks down the question, researchers search the web in parallel, a reviewer evaluates quality and relevance, and a writer produces the final report with citations. It has accumulated over 10,000 GitHub stars and is actively maintained.

### General-Purpose LLM Workflows

**ChatGPT Deep Research** (launched February 2025) is OpenAI's entry into autonomous research. When triggered, it creates a research plan, conducts multiple web searches (browsing dozens to hundreds of pages), takes 5 to 30 minutes depending on complexity, and produces a structured report with inline citations. It uses a variant of the o3 model fine-tuned for research tasks and can follow citation chains --- finding a claim, then verifying it by checking the source. The limitation that matters: it cannot access paywalled academic papers, which is most of the literature that researchers actually need.

**Perplexity Pro** is a search-augmented LLM that answers questions with inline citations to web sources. Its Pro Search mode conducts multi-step searches with clarifying questions, and its Spaces feature allows persistent research contexts with saved sources. Perplexity is good for surface-level research --- finding facts, getting oriented in a new topic, checking claims. Independent evaluations found that roughly 60 to 80 percent of its inline citations accurately support the claim they are attached to. The remaining 20 to 40 percent are partially relevant, tangentially related, or occasionally unsupportive. For research where citation accuracy is non-negotiable, that error rate is too high.

**Claude Projects and Claude Code** take a different approach: instead of autonomous research, they provide a configurable environment where the researcher maintains control. Claude Projects offers a 200K-token context window with document upload and project-level system prompts, making it useful for close reading of specific paper sets and drafting literature review sections. Claude Code provides a CLI-based development environment with file access, code execution, web search, and extensible tool use via MCP servers. Neither is a research assistant out of the box, but both can be configured into one --- which is what Article 3 in this series will do.

### The Capability Matrix

How does each tool score across the five capabilities a research assistant needs? The five capabilities --- which the next section will define in detail --- are literature synthesis, hypothesis generation, experimental design, code execution, and knowledge management.

| Tool | Literature | Hypothesis | Experiment | Code | Knowledge |
|------|-----------|------------|------------|------|-----------|
| Semantic Scholar | ◐ | ○ | ○ | ○ | ○ |
| Elicit | ● | ○ | ○ | ○ | ◐ |
| Consensus | ◐ | ○ | ○ | ○ | ○ |
| ResearchRabbit | ◐ | ○ | ○ | ○ | ◐ |
| Connected Papers | ◐ | ○ | ○ | ○ | ○ |
| AI Scientist | ◐ | ◐ | ◐ | ● | ○ |
| ChemCrow | ◐ | ◐ | ◐ | ◐ | ○ |
| STORM | ● | ○ | ○ | ○ | ○ |
| GPT-Researcher | ◐ | ○ | ○ | ○ | ○ |
| ChatGPT Deep Research | ◐ | ○ | ○ | ○ | ○ |
| Perplexity Pro | ◐ | ○ | ○ | ○ | ◐ |
| Claude Projects | ◐ | ◐ | ○ | ◐ | ◐ |

● = strong capability  ◐ = partial capability  ○ = absent

### The Gap

The matrix makes the state of the landscape visible. The literature column has the most filled circles --- this is the problem most tools are solving, because it is the most straightforward: take papers, apply NLP, return results. The hypothesis and experiment columns are nearly empty. The knowledge management column --- persistent state across research sessions --- is almost entirely absent.

No tool in the matrix scores ● in more than two columns. The AI Scientist comes closest to breadth but sacrifices depth --- its literature capability is partial (limited to a narrow domain), its hypothesis generation is constrained to variations on known ideas, and it has no persistent knowledge management. ChemCrow demonstrates that domain-specific tool augmentation works but is limited to chemistry.

The gap is not a missing feature. It is a missing architecture. Building a research assistant that covers all five capabilities requires a system design that integrates retrieval, reasoning, tool use, human judgment, and persistent memory. The tools that exist today are components. Nobody has assembled the complete system.

---

## The Five Capabilities

The capability matrix in the previous section previewed five columns. This section defines each one precisely --- what it is, why it matters, what "good" looks like, and what the system needs to implement it. These are not aspirational features. They are the minimum set of capabilities required for a research assistant to be more useful than a search engine with a chatbot.

### 1. Literature Ingestion and Synthesis

**What it is.** The ability to process a corpus of papers and extract structured understanding that spans individual documents. This is not summarization --- summarization compresses one document into fewer words. Synthesis identifies patterns, agreements, contradictions, methodology trends, and knowledge gaps *across* documents.

**Why it matters.** A researcher reading one paper at a time builds a mental model of their field incrementally. That mental model is lossy --- you forget details, you miss connections, you develop blind spots in areas you have not read recently. A research assistant that can synthesize across hundreds of papers simultaneously does not have these limitations. It can hold the entire corpus in context and answer questions that require cross-document reasoning.

**What "good" looks like.** You ask: "What methods have been used to improve transformer efficiency, and where do the results disagree?" The assistant returns: "Papers A, B, C, and D all report that pruning reduces inference time by 30 to 50 percent with less than 2 percent accuracy loss. Paper E contradicts this on models below 1B parameters --- pruning in that regime causes 8 to 12 percent accuracy degradation. Papers F and G use quantization instead, reporting similar latency improvements with no accuracy loss above 7B parameters but significant loss below 3B. No paper has tested the combination of pruning and quantization on models in the 1B to 3B range, which is a gap."

**Technical requirements.** PDF parsing with section-boundary detection. Chunking strategies that preserve document structure rather than splitting at arbitrary token windows. Embedding generation for semantic retrieval. A vector store for efficient similarity search. Cross-document reasoning that can hold retrieved chunks from multiple papers and identify patterns across them.

### 2. Hypothesis Generation

**What it is.** Structured brainstorming that produces specific, testable hypotheses with novelty assessment and supporting evidence. This is not "give me ideas" --- it is a systematic exploration of the hypothesis space with checks against existing literature.

**Why it matters.** Generating research hypotheses is a cognitive task that benefits from breadth of knowledge. A researcher knows her own field deeply but may not see connections to adjacent fields. An assistant with access to a broad corpus can identify analogies --- a technique used in field A that has structural similarities to an unsolved problem in field B --- that a human specialist would miss.

**What "good" looks like.** The assistant produces structured output: a hypothesis statement ("Applying spectral normalization from GAN training to the attention weights in vision transformers will reduce training instability on small datasets"), supporting evidence (three papers showing spectral normalization stabilizes GAN training, two papers documenting attention weight instability in vision transformers on ImageNet subsets), a novelty assessment ("No existing paper applies spectral normalization to vision transformer attention. The closest work is Paper X, which uses a different normalization approach on a different architecture"), and suggested experiments ("Compare training stability metrics on CIFAR-10 and ImageNet-100 with and without spectral normalization on ViT-Small").

**Technical requirements.** Retrieval over the literature corpus to check novelty. Structured output templates that enforce hypothesis format. Cross-domain retrieval to find analogies. A human-in-the-loop checkpoint where the researcher validates, redirects, or discards each hypothesis before any experiment is designed.

### 3. Experimental Design

**What it is.** The ability to design rigorous experiments: statistical power analysis, control group specification, confound identification, and protocol generation. Not "run this experiment" but "here is why this experimental design controls for confounds X, Y, and Z, and here is the minimum sample size needed to detect an effect of this magnitude."

**Why it matters.** Experimental design is where many research projects go wrong. Underpowered studies, missing controls, unidentified confounds --- these are systematic errors that waste months of work and produce unreliable results. An assistant that can check experimental design against statistical requirements before the experiment runs catches errors that would otherwise surface during peer review, or worse, never surface at all.

**What "good" looks like.** You describe an experiment: "I want to test whether data augmentation strategy X improves accuracy on task Y." The assistant returns: a power analysis calculating the minimum sample size given your expected effect size and desired statistical power, a list of confounds to control for (hardware variation, random seed sensitivity, data ordering effects, hyperparameter sensitivity), a suggested control group design, and a concrete protocol with randomization strategy, evaluation metrics, and stopping criteria.

**Technical requirements.** Statistical computation (power analysis, sample size calculation, effect size estimation). Domain-aware constraint generation --- knowing which confounds matter in ML experiments versus clinical trials versus social science studies. Code generation for analysis pipelines that implement the designed experiment.

### 4. Code Execution and Analysis

**What it is.** Sandboxed environments for running data analysis, generating visualizations, running simulations, and executing the computational components of research. The critical requirement is reproducibility: every analysis must generate code with fixed random seeds, pinned dependency versions, and logged data provenance.

**Why it matters.** Research increasingly involves computation --- running models, processing datasets, generating figures, performing statistical tests. An assistant that can execute analysis code and present results with full provenance eliminates the gap between "I wonder what this data looks like" and actually seeing it. The reproducibility requirement is non-negotiable because irreproducible computational results are a growing crisis in science.

**What "good" looks like.** You ask: "Run a correlation analysis between training dataset size and model accuracy across the papers in my corpus that report both metrics." The assistant writes a Python script that extracts the relevant data points, runs the analysis, generates a scatter plot with confidence intervals, computes the correlation coefficient and p-value, and saves all of this with a script that can be re-executed to reproduce the exact result. The script includes the data source, extraction method, library versions, and random seed.

**Technical requirements.** A sandboxed execution environment (containerized Python with standard scientific libraries). Code generation that follows strict reproducibility rules. Version pinning for all dependencies. Data provenance logging. Output capture and presentation (figures, tables, statistical results).

### 5. Knowledge Management

**What it is.** Persistent memory across research sessions: a structured record of what you have read, what you have concluded, what hypotheses you have tested, and how papers connect to each other. This is the research assistant's equivalent of a lab notebook.

**Why it matters.** Research is iterative. You read papers in week one, form a hypothesis in week two, design an experiment in week three, and revisit your literature review in week four when results surprise you. Without persistent memory, the assistant starts from scratch each session. With it, the assistant knows your research context: what you have already read, what you are working on, what you have tried and what worked.

**What "good" looks like.** You open a new session. The assistant knows: "Last session, you were investigating the contradiction between Paper D's results and the consensus from Papers A, B, and C. You hypothesized that the difference is due to dataset scale. You designed an experiment to test this but had not yet run it. Your citation graph currently tracks 47 papers with 3 unresolved contradictions and 2 identified gaps."

**Technical requirements.** A structured research journal with typed entries (hypothesis, experiment, finding, note). A citation graph maintained as a queryable database (papers as nodes, citations as directed edges, shared methodology as undirected edges). A memory architecture with different persistence tiers: stable patterns in project configuration (small, permanent), research state in a project database (large, persistent), paper embeddings in a vector store (large, searchable), and conversation context (ephemeral).

### How the Five Capabilities Connect

The five capabilities are not independent modules. They form a workflow with feedback loops.

<svg viewBox="0 0 700 420" xmlns="http://www.w3.org/2000/svg" style="max-width:750px; display:block; margin:2em auto; font-family:Georgia,serif;">
  <!-- Background -->
  <rect width="700" height="420" rx="8" fill="#1a1a2e"/>
  <!-- Title -->
  <text x="350" y="30" text-anchor="middle" fill="#e8e8e8" font-size="16" font-weight="bold">The Five Capabilities: Research Workflow</text>

  <!-- Capability boxes -->
  <!-- 1. Literature -->
  <rect x="40" y="60" width="160" height="70" rx="6" fill="#1e3a2a" stroke="#6dc98c" stroke-width="1.5"/>
  <text x="120" y="88" text-anchor="middle" fill="#6dc98c" font-size="13" font-weight="bold">1. Literature</text>
  <text x="120" y="108" text-anchor="middle" fill="#9ec9a8" font-size="11">Ingestion &amp; Synthesis</text>

  <!-- 2. Hypothesis -->
  <rect x="270" y="60" width="160" height="70" rx="6" fill="#1e2a3a" stroke="#5b9bd5" stroke-width="1.5"/>
  <text x="350" y="88" text-anchor="middle" fill="#5b9bd5" font-size="13" font-weight="bold">2. Hypothesis</text>
  <text x="350" y="108" text-anchor="middle" fill="#8bb8d9" font-size="11">Generation &amp; Novelty</text>

  <!-- 3. Experiment -->
  <rect x="500" y="60" width="160" height="70" rx="6" fill="#3a2a1e" stroke="#d4944a" stroke-width="1.5"/>
  <text x="580" y="88" text-anchor="middle" fill="#d4944a" font-size="13" font-weight="bold">3. Experiment</text>
  <text x="580" y="108" text-anchor="middle" fill="#d9b88b" font-size="11">Design &amp; Validation</text>

  <!-- 4. Code -->
  <rect x="150" y="220" width="160" height="70" rx="6" fill="#2a1e3a" stroke="#a05bd5" stroke-width="1.5"/>
  <text x="230" y="248" text-anchor="middle" fill="#a05bd5" font-size="13" font-weight="bold">4. Code Execution</text>
  <text x="230" y="268" text-anchor="middle" fill="#c08bd9" font-size="11">Analysis &amp; Simulation</text>

  <!-- 5. Knowledge -->
  <rect x="390" y="220" width="160" height="70" rx="6" fill="#3a1e1e" stroke="#e06060" stroke-width="1.5"/>
  <text x="470" y="248" text-anchor="middle" fill="#e06060" font-size="13" font-weight="bold">5. Knowledge</text>
  <text x="470" y="268" text-anchor="middle" fill="#d98b8b" font-size="11">Memory &amp; Citation Graph</text>

  <!-- Forward arrows (main flow) -->
  <!-- Literature → Hypothesis -->
  <line x1="200" y1="95" x2="268" y2="95" stroke="#6dc98c" stroke-width="2" marker-end="url(#arrowGreen)"/>
  <!-- Hypothesis → Experiment -->
  <line x1="430" y1="95" x2="498" y2="95" stroke="#5b9bd5" stroke-width="2" marker-end="url(#arrowBlue)"/>
  <!-- Experiment → Code -->
  <line x1="540" y1="130" x2="310" y2="222" stroke="#d4944a" stroke-width="2" marker-end="url(#arrowOrange)"/>
  <!-- Code → Knowledge -->
  <line x1="310" y1="255" x2="388" y2="255" stroke="#a05bd5" stroke-width="2" marker-end="url(#arrowPurple)"/>

  <!-- Feedback arrows -->
  <!-- Knowledge → Literature (feedback loop) -->
  <path d="M 470 290 L 470 350 L 120 350 L 120 132" fill="none" stroke="#e06060" stroke-width="1.5" stroke-dasharray="6,3" marker-end="url(#arrowRed)"/>
  <!-- Knowledge → Hypothesis (feedback loop) -->
  <path d="M 430 290 L 430 330 L 350 330 L 350 132" fill="none" stroke="#e06060" stroke-width="1.5" stroke-dasharray="6,3" marker-end="url(#arrowRed)"/>

  <!-- Human checkpoint markers -->
  <rect x="215" y="150" width="90" height="28" rx="4" fill="#2a2a4e" stroke="#8888cc" stroke-width="1"/>
  <text x="260" y="168" text-anchor="middle" fill="#8888cc" font-size="10">Human Review</text>

  <rect x="445" y="150" width="90" height="28" rx="4" fill="#2a2a4e" stroke="#8888cc" stroke-width="1"/>
  <text x="490" y="168" text-anchor="middle" fill="#8888cc" font-size="10">Human Review</text>

  <!-- Legend -->
  <text x="40" y="395" fill="#888" font-size="11">── Main flow</text>
  <text x="200" y="395" fill="#888" font-size="11">- - Feedback loop</text>
  <rect x="370" y="384" width="60" height="16" rx="3" fill="#2a2a4e" stroke="#8888cc" stroke-width="1"/>
  <text x="400" y="396" text-anchor="middle" fill="#8888cc" font-size="9">Human</text>
  <text x="445" y="396" fill="#888" font-size="11">= checkpoint</text>

  <!-- Arrow markers -->
  <defs>
    <marker id="arrowGreen" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#6dc98c"/></marker>
    <marker id="arrowBlue" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#5b9bd5"/></marker>
    <marker id="arrowOrange" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#d4944a"/></marker>
    <marker id="arrowPurple" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#a05bd5"/></marker>
    <marker id="arrowRed" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#e06060"/></marker>
  </defs>
</svg>

Literature feeds hypothesis generation --- you cannot hypothesize without knowing what has already been established. Hypotheses feed experimental design --- each hypothesis implies specific tests. Experimental designs feed code execution --- someone has to actually run the analysis. Results feed knowledge management --- every experiment updates your understanding. And knowledge management feeds back into everything: your evolving understanding shapes what literature you seek, what hypotheses you generate, and how you interpret results. The human-in-the-loop checkpoints sit at two critical transitions: between hypothesis generation and experimental design (validating that the hypothesis is worth testing) and between experimental design and code execution (approving the experiment before resources are spent).

---

## Architecture of a Research Assistant

The five capabilities define *what* the system must do. The architecture defines *how*. A research assistant is not a single model answering questions. It is a pipeline of specialized components connected by a language model that acts as an orchestrator, with human checkpoints at every decision boundary.

### The RAG Pipeline

RAG --- retrieval-augmented generation --- is the foundation. Instead of asking an LLM to answer research questions from its training data (which is stale, incomplete, and prone to hallucination), you retrieve relevant documents first and provide them as context. The quality of a RAG system is determined almost entirely by the quality of its retrieval, and retrieval quality depends on three engineering decisions: how you parse documents, how you chunk them, and how you embed them.

**PDF ingestion and text extraction.** Academic papers are PDFs, and PDFs are hostile to machine reading. Tables render as scattered text fragments. Equations break across lines. Two-column layouts interleave unrelated paragraphs. Figures have captions that are spatially separate from the text that references them. Two tools handle this at different fidelity levels: PyMuPDF is fast and extracts raw text reliably but loses document structure (it does not know where sections begin and end). Grobid is an academic-specific parser that extracts section boundaries, reference lists, and citation metadata, preserving the structure that matters for downstream reasoning. The trade-off is speed versus structure: PyMuPDF processes a paper in milliseconds, Grobid in seconds. For a research assistant that needs to reason about which section of a paper contains a particular claim, Grobid's structure preservation is worth the latency.

**Chunking strategy.** This is where most RAG systems fail silently. The naive approach is to split text into fixed-size token windows with some overlap (e.g., 512 tokens with 50 tokens of overlap). This is easy to implement and terrible for research papers. It splits mid-sentence, mid-paragraph, and mid-argument. A chunk might contain the end of one section's methodology description and the beginning of the next section's results, creating a context that never existed in the original paper. Section-aware chunking splits at section boundaries instead, keeping each chunk as a self-contained unit with its section title and the paper's title and abstract as metadata. Citation-preserving chunking goes further: when a chunk references another paper, the citation metadata (title, authors, year) is included in the chunk, so the retrieval system knows not just what was said but what was cited as evidence.

**Embedding generation.** Each chunk gets converted into a dense vector representation --- an embedding --- that captures its semantic meaning. The embedding model determines retrieval quality. General-purpose models like text-embedding-3-large work but miss domain-specific nuance. Scientific embedding models like SPECTER (trained on citation signal from Semantic Scholar) capture the similarity structure that matters for research: papers that cite each other get similar embeddings, even if they use different terminology. For a corpus of 10,000 papers with an average of 20 chunks per paper, embedding generation is a batch job processing 200,000 chunks. At current API pricing, this costs between $2 and $20 depending on the model and provider.

**Vector storage.** The embeddings need to be stored for fast similarity search. Three options span the complexity spectrum. pgvector is a PostgreSQL extension that adds vector similarity search to your existing database --- simple, integrated, good enough for corpora under a million chunks. Chroma is a lightweight, local vector database designed for AI applications --- more features than pgvector, still simple to deploy. Pinecone is a managed cloud service that handles scaling, replication, and index optimization --- appropriate when you need to serve concurrent users against a corpus of millions of chunks.

### The Tool-Augmented LLM Backbone

The language model is the orchestrator, not the database. It does not store knowledge --- it routes queries, calls tools, interprets results, and generates structured outputs. The tools it needs access to:

**Code execution.** A sandboxed Python environment with scientific libraries (numpy, scipy, pandas, matplotlib, statsmodels). The sandbox must be isolated --- the LLM should not be able to modify system files, access the network arbitrarily, or execute persistent processes.

**Web search.** For retrieving current papers, checking recent results, and verifying claims against primary sources. This supplements the local vector store with real-time access to new publications.

**Database queries.** The knowledge management layer stores research state in a structured database. The LLM needs read and write access to log findings, update the citation graph, and retrieve research context from previous sessions.

**Structured output generation.** The LLM produces hypothesis templates, experiment specifications, and literature review sections as structured data (JSON, YAML, or formatted Markdown), not free-form text. Structure makes outputs parseable, versionable, and comparable across sessions.

### Human-in-the-Loop Checkpoints

The most important architectural decision is where autonomy ends. A research assistant that operates fully autonomously is a research assistant that will hallucinate citations, pursue dead-end hypotheses, and generate plausible but wrong conclusions without anyone noticing. The researcher is the domain expert. The assistant is the computational engine. The design principle is simple: **the assistant proposes, the researcher disposes.**

Three checkpoints are non-negotiable:

**Hypothesis validation.** Every generated hypothesis must be reviewed by the researcher before experimental design begins. The researcher checks: does this hypothesis make domain sense? Is the novelty assessment accurate? Is this worth the computational cost of testing?

**Experiment approval.** Every experimental design must be approved before code execution. The researcher checks: are the controls sufficient? Is the sample size realistic given available compute? Are the evaluation metrics appropriate?

**Result interpretation.** Every set of results must be interpreted by the researcher. The assistant presents data; the researcher decides what it means. This is where domain expertise is irreplaceable --- statistical significance does not imply scientific significance, and the difference requires human judgment.

<svg viewBox="0 0 800 520" xmlns="http://www.w3.org/2000/svg" style="max-width:850px; display:block; margin:2em auto; font-family:Georgia,serif;">
  <!-- Background -->
  <rect width="800" height="520" rx="8" fill="#1a1a2e"/>
  <!-- Title -->
  <text x="400" y="28" text-anchor="middle" fill="#e8e8e8" font-size="15" font-weight="bold">Research Assistant: System Architecture</text>

  <!-- Data Sources (top left) -->
  <text x="100" y="58" text-anchor="middle" fill="#888" font-size="11" font-weight="bold">DATA SOURCES</text>
  <rect x="20" y="68" width="80" height="36" rx="4" fill="#1e2a3a" stroke="#5b9bd5" stroke-width="1"/>
  <text x="60" y="90" text-anchor="middle" fill="#5b9bd5" font-size="10">ArXiv</text>
  <rect x="110" y="68" width="80" height="36" rx="4" fill="#1e2a3a" stroke="#5b9bd5" stroke-width="1"/>
  <text x="150" y="90" text-anchor="middle" fill="#5b9bd5" font-size="10">Semantic Scholar</text>
  <rect x="65" y="112" width="80" height="36" rx="4" fill="#1e2a3a" stroke="#5b9bd5" stroke-width="1"/>
  <text x="105" y="134" text-anchor="middle" fill="#5b9bd5" font-size="10">Uploaded PDFs</text>

  <!-- Ingestion Pipeline -->
  <text x="310" y="58" text-anchor="middle" fill="#888" font-size="11" font-weight="bold">INGESTION PIPELINE</text>
  <rect x="230" y="68" width="70" height="36" rx="4" fill="#1e3a2a" stroke="#6dc98c" stroke-width="1"/>
  <text x="265" y="84" text-anchor="middle" fill="#6dc98c" font-size="9">PDF Parse</text>
  <text x="265" y="96" text-anchor="middle" fill="#6dc98c" font-size="8">(Grobid)</text>
  <rect x="310" y="68" width="70" height="36" rx="4" fill="#1e3a2a" stroke="#6dc98c" stroke-width="1"/>
  <text x="345" y="84" text-anchor="middle" fill="#6dc98c" font-size="9">Chunk</text>
  <text x="345" y="96" text-anchor="middle" fill="#6dc98c" font-size="8">(section-aware)</text>
  <rect x="390" y="68" width="70" height="36" rx="4" fill="#1e3a2a" stroke="#6dc98c" stroke-width="1"/>
  <text x="425" y="84" text-anchor="middle" fill="#6dc98c" font-size="9">Embed</text>
  <text x="425" y="96" text-anchor="middle" fill="#6dc98c" font-size="8">(SPECTER)</text>

  <!-- Arrows through pipeline -->
  <line x1="195" y1="86" x2="228" y2="86" stroke="#6dc98c" stroke-width="1.5" marker-end="url(#arrowG2)"/>
  <line x1="300" y1="86" x2="308" y2="86" stroke="#6dc98c" stroke-width="1.5" marker-end="url(#arrowG2)"/>
  <line x1="380" y1="86" x2="388" y2="86" stroke="#6dc98c" stroke-width="1.5" marker-end="url(#arrowG2)"/>

  <!-- Vector Store -->
  <rect x="500" y="62" width="120" height="50" rx="6" fill="#2a2a3e" stroke="#8888cc" stroke-width="1.5"/>
  <text x="560" y="84" text-anchor="middle" fill="#8888cc" font-size="11" font-weight="bold">Vector Store</text>
  <text x="560" y="100" text-anchor="middle" fill="#8888cc" font-size="9">(pgvector / Chroma)</text>
  <line x1="460" y1="86" x2="498" y2="86" stroke="#6dc98c" stroke-width="1.5" marker-end="url(#arrowG2)"/>

  <!-- LLM Backbone (center) -->
  <rect x="280" y="170" width="240" height="80" rx="8" fill="#2a1e3a" stroke="#a05bd5" stroke-width="2"/>
  <text x="400" y="198" text-anchor="middle" fill="#a05bd5" font-size="14" font-weight="bold">LLM Orchestrator</text>
  <text x="400" y="216" text-anchor="middle" fill="#c08bd9" font-size="10">Routes queries, calls tools,</text>
  <text x="400" y="230" text-anchor="middle" fill="#c08bd9" font-size="10">generates structured outputs</text>

  <!-- Vector store to LLM -->
  <line x1="560" y1="112" x2="560" y2="190" stroke="#8888cc" stroke-width="1.5"/>
  <line x1="560" y1="190" x2="522" y2="190" stroke="#8888cc" stroke-width="1.5" marker-end="url(#arrowViolet)"/>
  <text x="570" y="155" fill="#8888cc" font-size="9">retrieval</text>

  <!-- Tools (below LLM) -->
  <text x="400" y="278" text-anchor="middle" fill="#888" font-size="11" font-weight="bold">TOOLS</text>
  <rect x="140" y="290" width="100" height="40" rx="4" fill="#1e3a2a" stroke="#6dc98c" stroke-width="1"/>
  <text x="190" y="310" text-anchor="middle" fill="#6dc98c" font-size="10">Code Sandbox</text>
  <text x="190" y="322" text-anchor="middle" fill="#6dc98c" font-size="8">(Python)</text>

  <rect x="260" y="290" width="100" height="40" rx="4" fill="#1e2a3a" stroke="#5b9bd5" stroke-width="1"/>
  <text x="310" y="310" text-anchor="middle" fill="#5b9bd5" font-size="10">Web Search</text>
  <text x="310" y="322" text-anchor="middle" fill="#5b9bd5" font-size="8">(current papers)</text>

  <rect x="380" y="290" width="100" height="40" rx="4" fill="#3a2a1e" stroke="#d4944a" stroke-width="1"/>
  <text x="430" y="310" text-anchor="middle" fill="#d4944a" font-size="10">Knowledge DB</text>
  <text x="430" y="322" text-anchor="middle" fill="#d4944a" font-size="8">(research state)</text>

  <rect x="500" y="290" width="100" height="40" rx="4" fill="#3a1e1e" stroke="#e06060" stroke-width="1"/>
  <text x="550" y="310" text-anchor="middle" fill="#e06060" font-size="10">Structured Output</text>
  <text x="550" y="322" text-anchor="middle" fill="#e06060" font-size="8">(JSON/YAML)</text>

  <!-- LLM to tools -->
  <line x1="340" y1="250" x2="190" y2="288" stroke="#6dc98c" stroke-width="1.5" marker-end="url(#arrowG2)"/>
  <line x1="370" y1="250" x2="310" y2="288" stroke="#5b9bd5" stroke-width="1.5" marker-end="url(#arrowB2)"/>
  <line x1="430" y1="250" x2="430" y2="288" stroke="#d4944a" stroke-width="1.5" marker-end="url(#arrowO2)"/>
  <line x1="460" y1="250" x2="550" y2="288" stroke="#e06060" stroke-width="1.5" marker-end="url(#arrowR2)"/>

  <!-- Output Layer -->
  <text x="400" y="370" text-anchor="middle" fill="#888" font-size="11" font-weight="bold">OUTPUTS</text>
  <rect x="100" y="380" width="120" height="36" rx="4" fill="#1e3a2a" stroke="#6dc98c" stroke-width="1"/>
  <text x="160" y="402" text-anchor="middle" fill="#6dc98c" font-size="10">Synthesis Reports</text>

  <rect x="240" y="380" width="120" height="36" rx="4" fill="#1e2a3a" stroke="#5b9bd5" stroke-width="1"/>
  <text x="300" y="402" text-anchor="middle" fill="#5b9bd5" font-size="10">Hypotheses</text>

  <rect x="380" y="380" width="120" height="36" rx="4" fill="#3a2a1e" stroke="#d4944a" stroke-width="1"/>
  <text x="440" y="402" text-anchor="middle" fill="#d4944a" font-size="10">Experiment Designs</text>

  <rect x="520" y="380" width="120" height="36" rx="4" fill="#2a1e3a" stroke="#a05bd5" stroke-width="1"/>
  <text x="580" y="402" text-anchor="middle" fill="#a05bd5" font-size="10">Analysis Results</text>

  <!-- Human checkpoint bar -->
  <rect x="80" y="440" width="580" height="32" rx="4" fill="#2a2a4e" stroke="#8888cc" stroke-width="1.5" stroke-dasharray="4,2"/>
  <text x="370" y="460" text-anchor="middle" fill="#8888cc" font-size="11" font-weight="bold">HUMAN-IN-THE-LOOP CHECKPOINTS</text>
  <text x="160" y="490" text-anchor="middle" fill="#8888cc" font-size="9">Validate hypotheses</text>
  <text x="370" y="490" text-anchor="middle" fill="#8888cc" font-size="9">Approve experiments</text>
  <text x="580" y="490" text-anchor="middle" fill="#8888cc" font-size="9">Interpret results</text>

  <!-- Researcher icon markers -->
  <circle cx="160" y="500" r="4" fill="#8888cc"/>
  <circle cx="370" y="500" r="4" fill="#8888cc"/>
  <circle cx="580" y="500" r="4" fill="#8888cc"/>

  <!-- Arrow defs -->
  <defs>
    <marker id="arrowG2" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><path d="M0,0 L7,2.5 L0,5" fill="#6dc98c"/></marker>
    <marker id="arrowB2" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><path d="M0,0 L7,2.5 L0,5" fill="#5b9bd5"/></marker>
    <marker id="arrowO2" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><path d="M0,0 L7,2.5 L0,5" fill="#d4944a"/></marker>
    <marker id="arrowR2" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><path d="M0,0 L7,2.5 L0,5" fill="#e06060"/></marker>
    <marker id="arrowViolet" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><path d="M0,0 L7,2.5 L0,5" fill="#8888cc"/></marker>
  </defs>
</svg>

The architecture is a stack: data sources feed an ingestion pipeline, which populates a vector store, which serves the LLM orchestrator, which uses tools to produce outputs, which pass through human checkpoints before feeding back into the knowledge database. Every component has a defined interface. Every interface can be tested independently. This testability is not a nice-to-have --- it is how you build a system that you can trust with scientific reasoning.

---

## Market Analysis

The demand side of this market is large and growing. UNESCO estimates approximately 9 million full-time equivalent researchers globally, with China contributing roughly 2.3 million, the EU 1.9 million, the US 1.5 million, Japan 0.7 million, and the rest of the world making up the balance. If you include graduate students actively conducting research, the number exceeds 15 million.

Current spending on research information services is substantial. Elsevier (parent company RELX) generates roughly $3.5 billion in annual revenue from academic publishing and data analytics. Clarivate, which operates Web of Science, generates approximately $2.7 billion. A typical research university spends $5 to $15 million per year on library subscriptions to journal databases. The total traditional academic information services market exceeds $30 billion.

The AI-specific research tools market is much smaller --- estimated at $500 million to $1 billion in 2024 --- but growing rapidly. Perplexity's valuation jumped from $520 million (Series B, early 2024) to over $9 billion (late 2024). Sakana AI raised over $200 million. Elicit and Consensus raised $9 million and $11.5 million respectively, both in 2023. The funding trajectory suggests that investors see this as a market in its pre-inflection phase.

Willingness to pay segments cleanly by user type:

**Academic researchers ($20-50/month).** Price-sensitive, often paying out of personal or grant funds. Need literature synthesis and knowledge management most. Elicit and Consensus target this segment.

**Enterprise R&D ($200-500/seat/month).** Corporate research teams in technology, pharmaceuticals, and materials science. Need all five capabilities, especially code execution and experimental design. Willing to pay for reliability and integration with existing workflows.

**Pharma and biotech ($1,000+/seat/month).** The highest willingness to pay, driven by the extreme cost of failed experiments. A clinical trial that costs $50 million can be de-risked by better experimental design and literature synthesis. At this price point, the research assistant competes with hiring additional research staff.

The competitive dynamics split along a familiar axis: vertical specialists versus horizontal platforms. Vertical tools (ChemCrow for chemistry, domain-specific fine-tuned models) know more about one field. Horizontal tools (general-purpose LLMs, broad retrieval systems) make cross-domain connections. Research is inherently cross-domain --- the most impactful discoveries often come from applying techniques from one field to problems in another. This suggests that horizontal platforms with domain-specific extensions will win long-term, the same pattern that played out in enterprise software (Salesforce started vertical, then went horizontal with a platform model).

---

## The Hard Problems

Six problems remain unsolved. Each one is a research problem in itself, and each constrains what a research assistant can reliably do today.

### 1. Hallucination in Scientific Context

All language models hallucinate. In a scientific context, not all hallucinations are equally dangerous.

**Citation hallucination is catastrophic.** The model generates a reference to a paper that does not exist --- plausible author names, a reasonable title, a specific journal and year. Without retrieval augmentation, LLMs fabricate 30 to 70 percent of academic citations. With RAG (retrieving real papers and providing them as context), the fabrication rate drops to 5 to 15 percent, but a new error mode emerges: misattribution, where the model cites a real paper but attributes a claim to it that the paper does not actually make. In research, a single hallucinated citation in a literature review can propagate through an entire line of investigation.

**Methodology suggestion errors are annoying but survivable.** The model suggests using a statistical test that is inappropriate for your data distribution, or recommends a hyperparameter range that is reasonable but suboptimal. These errors are caught during the experimental design review checkpoint. They waste time but do not corrupt the research record.

The implication for architecture: every claim the research assistant makes must be traceable to a specific source. The system must distinguish between claims derived from retrieved documents (verifiable) and claims generated from the model's training data (unverifiable). The human checkpoint is the last line of defense, but the system should make verification as easy as possible by providing inline citations with page numbers.

### 2. Evaluation

How do you measure whether a hypothesis is "good"? There is no ground truth. You cannot train a classifier on good-hypothesis / bad-hypothesis pairs because the quality of a hypothesis is only determined retrospectively, often years later.

Proxy metrics exist but are imperfect:

**Novelty** --- is this hypothesis already in the existing literature? This is measurable via retrieval: search the corpus for the hypothesis statement and check if existing papers already test it. But novelty alone is not quality. A novel hypothesis can be trivially true, obviously wrong, or untestable.

**Plausibility** --- is this hypothesis consistent with known results? A hypothesis that contradicts well-established findings is probably wrong (though occasionally, those are the most important hypotheses). Plausibility can be estimated by checking whether the hypothesis's assumptions align with the consensus in the literature.

**Testability** --- can this hypothesis be verified experimentally with available resources? A hypothesis about dark matter interactions is not testable in a computational neuroscience lab. Testability requires matching the hypothesis to the researcher's available methods, data, and compute.

None of these metrics individually captures hypothesis quality. Their combination provides a useful signal but not a definitive score. The research assistant can rank hypotheses by these proxies, but the researcher must make the final judgment.

### 3. Reproducibility

The research assistant generates code for data analysis, statistical tests, and visualizations. That code must be exactly reproducible --- running it again, on any machine, must produce the same result. This is harder than it sounds.

Sources of irreproducibility: random number generators without fixed seeds, floating-point arithmetic differences across hardware, dependency version drift (the same library function behaving differently in version 2.1 versus 2.3), data pipeline ordering effects (shuffling data differently changes training dynamics), and implicit state from interactive sessions (running cells in a notebook out of order).

The solution is engineering discipline: mandatory random seeds for every stochastic operation, pinned dependency versions in a lock file, data provenance logging (recording the exact input data, its source, and any preprocessing applied), and deterministic execution environments (containers with fixed OS and library versions). The research assistant must enforce these requirements in every piece of code it generates, which means the rules must be embedded in its configuration, not left to the model's judgment.

### 4. Citation Accuracy

Every claim the research assistant makes about the literature must trace to a specific paper, a specific section, and a specific claim within that section. "Paper X shows that Y" is insufficient --- the system must be able to point to where in Paper X the claim Y appears.

This requires chunk-level provenance: when the system retrieves a chunk from the vector store, it must retain the chunk's metadata (paper title, section, page number, paragraph). When the system synthesizes across multiple chunks, it must maintain attribution for each synthesized claim. This is technically straightforward but operationally demanding --- every retrieval and reasoning step must carry its provenance chain forward.

### 5. Depth Versus Breadth

Domain-specific models (fine-tuned on chemistry papers, trained on medical literature) know more about their field than general-purpose models. They use terminology correctly, understand domain conventions, and make fewer errors on domain-specific reasoning. General-purpose models make cross-domain connections that domain-specific models miss, because they have seen a broader range of fields during training.

This trade-off is fundamental, not solvable by training a bigger model. The architecturally interesting question is how to compose specialized and general capabilities: using a general-purpose model as the orchestrator with domain-specific retrieval and validation tools, or using a domain-specific model with cross-domain retrieval augmentation. The right answer likely varies by application.

### 6. Trust Calibration

The researcher needs to know when to trust the assistant and when to verify independently. Confidence scores ("I am 85% confident in this synthesis") are necessary but insufficient, because language model confidence scores are poorly calibrated --- they express the model's output probability, not the factual accuracy of the claim.

Better calibration requires grounding confidence in evidence: "This synthesis is based on 12 retrieved papers, 9 of which agree. The 3 disagreeing papers all use a different methodology, which may explain the discrepancy." This is more useful than a number because it gives the researcher the information needed to make their own trust decision. The research assistant should present evidence, not conclusions about its own reliability.

---

## Where This Series Goes

This article defined the problem (the synthesis bottleneck), surveyed the landscape (fragmented, partially solved), specified the requirements (five capabilities), and sketched the architecture (RAG pipeline with tool-augmented LLM and human checkpoints). The hard problems are real and unsolved, but the architectural framework is sound.

The next two articles in this series turn this framework into a working system.

**Article 2: Claude Code for Production** covers the development tool we will use to build the research assistant. Claude Code is not autocomplete --- it is a configurable development environment that enforces engineering discipline through CLAUDE.md conventions, hook systems, test-driven workflows, and code review patterns. The difference between a Claude Code project that ships clean, maintainable software and one that generates technical debt is entirely in the configuration. Article 2 is the setup guide.

**Article 3: Building the Research Assistant** takes the requirements from this article and the development discipline from Article 2 and builds the system. Every architectural component maps to a Claude Code best practice. Every component is built test-first. Every failure mode is anticipated because the configuration tells the development tool to anticipate it. The result is a five-component research assistant with full observability and load testing --- not a prototype, but a system designed for production use.
