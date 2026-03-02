---
layout: post
title: "Building an AI Research Assistant: Claude Code Best Practices in Practice"
date: 2026-03-11
category: infra
---

*This is Part 3 of a 3-part series. [Part 1: AI Research Assistant Landscape](/posts/2026-03-09-ai-research-assistant-landscape/) | [Part 2: Claude Code Best Practices](/posts/2026-03-10-claude-code-production-best-practices/) | **Part 3: Building the Assistant***

We defined what an AI research assistant needs: five capabilities spanning literature synthesis, hypothesis generation, experimental design, code execution, and knowledge management. We defined how to build production software with Claude Code: CLAUDE.md conventions, hook systems, test-driven development, and semantic code review. Now we build the thing.

This article is a construction log. Every architectural decision maps to a best practice from the Claude Code guide. Every component is built test-first. Every failure mode is anticipated because the project's CLAUDE.md tells Claude to anticipate it. The goal is not a prototype --- it is a system designed to run in production, with observability, load testing, and clear boundaries between what the AI decides and what the human decides.

The research assistant we are building has five components: a literature ingestion pipeline that turns PDFs into searchable, semantically chunked vector embeddings; a semantic search and synthesis engine that answers research questions by reasoning across multiple papers simultaneously; a hypothesis generation engine that produces structured, novelty-assessed hypotheses; an experiment design assistant that handles statistical rigor; and a knowledge management layer that maintains persistent state across research sessions. Each component is a module with a defined interface, tested independently and integrated through the LLM orchestrator.

---

## Table of Contents

1. [The Project Setup](#the-project-setup)
2. [Literature Ingestion Pipeline](#literature-ingestion-pipeline)
3. [Semantic Search and Synthesis](#semantic-search-and-synthesis)
4. [The Hypothesis Engine](#the-hypothesis-engine)
5. [Experiment Design Assistant](#experiment-design-assistant)
6. [Knowledge Management](#knowledge-management)
7. [Observability for AI Systems](#observability-for-ai-systems)
8. [Load Testing the Research Pipeline](#load-testing-the-research-pipeline)
9. [The Complete Architecture](#the-complete-architecture)

---

## The Project Setup

Before writing any code, we set up the environment. This means writing the CLAUDE.md, defining the directory structure, configuring hooks, and selecting MCP servers. The setup is the most important phase --- it determines the quality ceiling for everything that follows.

### The CLAUDE.md

Here is the complete CLAUDE.md for this project. Every rule is a constraint derived from a requirement:

```markdown
# AI Research Assistant — Project Standards

## Stack
- Language: Python 3.12, strict type hints (mypy --strict)
- Framework: FastAPI for API layer, asyncio for concurrency
- Database: PostgreSQL 16 with pgvector extension for embeddings
- Embedding model: text-embedding-3-large (OpenAI) for paper embeddings
- LLM: Claude 3.5 Sonnet for synthesis and hypothesis generation
- Testing: pytest with pytest-asyncio, hypothesis for property-based tests
- Load testing: k6

## Coding Standards
- All functions must have type annotations (no `Any` unless unavoidable)
- All public functions must have a docstring with Args, Returns, and Raises
- NEVER use `print()` — use the structured logger from `src/observability/logger.py`
- NEVER use mutable default arguments
- NEVER catch bare `Exception` — catch specific exception types
- All database queries must use parameterized queries (no f-strings in SQL)
- All API endpoints must validate input with Pydantic models

## Research-Specific Rules
- NEVER generate a citation without a source chunk ID from the vector store
- NEVER present a synthesis claim without listing the supporting paper IDs
- All hypothesis outputs must use the structured template in `src/hypothesis/schema.py`
- All generated analysis code must include: random seed, library versions, data source path
- Confidence scores must be grounded in evidence counts, not model probabilities

## Performance
- Query-to-answer latency must be < 2 seconds for interactive use
- Embedding generation must process at least 100 chunks per second in batch mode
- Vector search must return results in < 100ms for corpora up to 1M chunks
- API endpoints must respond in < 200ms at P99 (excluding LLM calls)

## Testing
- Every new function must have at least one unit test
- Retrieval pipeline must have accuracy tests (precision@10, recall@10)
- Hypothesis generation must have structured output validation tests
- Integration tests must use a real PostgreSQL instance with pgvector
- Property-based tests for all data transformation functions
```

Every rule in this file traces to a requirement. "NEVER generate a citation without a source chunk ID" prevents citation hallucination. "Query-to-answer latency must be < 2 seconds" ensures interactive usability. "All generated analysis code must include random seed, library versions, data source path" enforces reproducibility.

### Directory Structure

```
research-assistant/
├── CLAUDE.md
├── .claude/
│   ├── settings.json          # Hooks, permissions, MCP servers
│   └── settings.local.json    # Local overrides (git-ignored)
├── src/
│   ├── ingestion/             # Component 1: Literature pipeline
│   │   ├── parser.py          # PDF parsing (Grobid integration)
│   │   ├── chunker.py         # Section-aware chunking
│   │   ├── embedder.py        # Embedding generation
│   │   └── store.py           # Vector store operations
│   ├── search/                # Component 2: Semantic search
│   │   ├── query.py           # Query decomposition
│   │   ├── retriever.py       # Vector retrieval + re-ranking
│   │   └── synthesizer.py     # Cross-paper synthesis
│   ├── hypothesis/            # Component 3: Hypothesis engine
│   │   ├── generator.py       # Hypothesis generation
│   │   ├── schema.py          # Structured output templates
│   │   └── novelty.py         # Novelty assessment
│   ├── experiment/            # Component 4: Experiment design
│   │   ├── power.py           # Power analysis
│   │   ├── design.py          # Experiment design generation
│   │   └── codegen.py         # Analysis code generation
│   ├── knowledge/             # Component 5: Knowledge management
│   │   ├── journal.py         # Research journal
│   │   ├── graph.py           # Citation graph
│   │   └── memory.py          # Session persistence
│   ├── observability/         # Cross-cutting: observability
│   │   ├── logger.py          # Structured logging
│   │   ├── metrics.py         # Quality and cost metrics
│   │   └── tracing.py         # Correlation ID tracing
│   └── api/                   # API layer
│       ├── routes.py          # FastAPI routes
│       └── models.py          # Pydantic request/response models
├── tests/
│   ├── unit/                  # Unit tests (mirror src/ structure)
│   ├── integration/           # Integration tests (real DB)
│   ├── load/                  # k6 load test scripts
│   └── fixtures/              # Test data (sample papers, embeddings)
├── migrations/                # Database migrations (Alembic)
├── pyproject.toml
└── docker-compose.yml         # PostgreSQL + pgvector for development
```

The structure is feature-based: each component has its own directory with all related code. Tests mirror the source structure. Cross-cutting concerns (observability) get their own directory. This organization means Claude only needs to read one directory when working on one component.

### Hook Configuration

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [{
          "type": "command",
          "command": "cd /path/to/research-assistant && python -m mypy src/ --strict 2>&1 | tail -20"
        }]
      },
      {
        "matcher": "Edit|Write",
        "hooks": [{
          "type": "command",
          "command": "cd /path/to/research-assistant && python -m ruff check src/ 2>&1 | tail -20"
        }]
      },
      {
        "matcher": "Edit|Write",
        "hooks": [{
          "type": "command",
          "command": "cd /path/to/research-assistant && python -m pytest tests/unit/ -x -q 2>&1 | tail -30"
        }]
      }
    ]
  }
}
```

Three hooks run on every file change: mypy for type checking, ruff for linting, and pytest for unit tests. The `-x` flag stops pytest on the first failure, and `tail` limits output to keep hook feedback concise. Every edit gets immediate feedback on type errors, style violations, and test failures.

### MCP Server Selection

Three MCP servers extend Claude's capabilities for this project:

**PostgreSQL with pgvector.** Claude can query the vector store directly, inspect embeddings, and verify retrieval results without writing throwaway scripts.

**Code execution sandbox.** A sandboxed Python environment where Claude can run analysis scripts, generate visualizations, and test statistical computations.

**ArXiv API access.** Claude can search for and retrieve current papers to augment the local corpus during research sessions.

---

## Literature Ingestion Pipeline

The ingestion pipeline turns PDFs into searchable, semantically chunked vector embeddings. This is the foundation --- every other component retrieves from the vector store that this pipeline populates. If the chunks are poorly formed, retrieval is noisy. If the embeddings are weak, semantically related papers do not cluster. If the metadata is lost, citations cannot be traced back to their source. Getting the pipeline right is the highest-leverage engineering decision in the entire system.

### Pipeline Stages

The pipeline has four stages: parse, chunk, embed, store. Each stage has a defined input, a defined output, and a set of failure modes.

**Stage 1: PDF Parsing.** The input is a PDF file. The output is structured text with section boundaries and metadata.

Two tools handle this. PyMuPDF (also called fitz) is a general-purpose PDF parser. It extracts text quickly and reliably but treats the document as a flat stream of characters --- it does not know where sections begin or end, does not extract reference lists as structured data, and does not distinguish between body text, figure captions, and table contents. Processing time: milliseconds per paper.

Grobid is a machine learning-based parser designed specifically for scientific documents. It identifies section titles, section boundaries, paragraph breaks, reference strings, and citation markers within the text. It outputs structured XML with labeled elements: title, abstract, section headers, body paragraphs, and a parsed reference list with author names, publication years, and venues. Processing time: 1 to 3 seconds per paper. Grobid runs as a local service (Java-based, Docker-deployable).

The trade-off is clear: PyMuPDF is faster and simpler but loses the structure that downstream components need. Grobid preserves structure at the cost of complexity and latency. For a research assistant that needs to know *which section* of a paper contains a claim and *which reference* supports it, Grobid's structure preservation is worth the cost.

```python
# Simplified Grobid integration
from dataclasses import dataclass

@dataclass
class ParsedSection:
    title: str
    text: str
    references: list[str]  # Reference IDs cited in this section

@dataclass
class ParsedPaper:
    title: str
    authors: list[str]
    abstract: str
    sections: list[ParsedSection]
    references: list[dict]  # Parsed reference list

async def parse_pdf(pdf_path: str) -> ParsedPaper:
    """Send PDF to Grobid and parse the structured XML response."""
    async with aiohttp.ClientSession() as session:
        with open(pdf_path, "rb") as f:
            resp = await session.post(
                "http://localhost:8070/api/processFulltextDocument",
                data={"input": f},
            )
        xml = await resp.text()
    return _parse_grobid_xml(xml)
```

**Stage 2: Section-Aware Chunking.** The input is a `ParsedPaper`. The output is a list of chunks, each with metadata.

Naive chunking --- splitting text into fixed-size windows of 512 tokens with 50-token overlap --- is the most common approach and the worst for research papers. It splits mid-sentence, mid-paragraph, and mid-argument. A chunk might contain the conclusion of one section and the introduction to the next, creating a semantic chimera that confuses retrieval.

Section-aware chunking splits at section boundaries. Each chunk is a self-contained unit: a single section (or a subsection if the section is long), prefixed with the paper's title and the section's title as metadata. If a section exceeds the maximum chunk size (typically 1,000 to 2,000 tokens), it is split at paragraph boundaries, never mid-sentence.

Citation-preserving chunking adds one more layer: when a chunk references another paper, the reference metadata (title, authors, year) is appended to the chunk. This means the embedding captures not just what the chunk says but what evidence it cites.

```python
@dataclass
class Chunk:
    paper_id: str
    paper_title: str
    section_title: str
    text: str
    cited_references: list[dict]
    token_count: int

def chunk_paper(paper: ParsedPaper, max_tokens: int = 1500) -> list[Chunk]:
    """Split a parsed paper into section-aware chunks."""
    chunks = []
    for section in paper.sections:
        if count_tokens(section.text) <= max_tokens:
            chunks.append(Chunk(
                paper_id=generate_id(paper),
                paper_title=paper.title,
                section_title=section.title,
                text=section.text,
                cited_references=[
                    paper.references[ref_id]
                    for ref_id in section.references
                    if ref_id in paper.references
                ],
                token_count=count_tokens(section.text),
            ))
        else:
            # Split at paragraph boundaries
            for para_group in split_paragraphs(section.text, max_tokens):
                chunks.append(Chunk(
                    paper_id=generate_id(paper),
                    paper_title=paper.title,
                    section_title=section.title,
                    text=para_group,
                    cited_references=[],  # Paragraph-level ref tracking is lossy
                    token_count=count_tokens(para_group),
                ))
    return chunks
```

**Stage 3: Embedding Generation.** The input is a list of chunks. The output is a list of vector embeddings, one per chunk.

The embedding model determines retrieval quality. Three options span the quality-cost spectrum:

- **BAAI/bge-large-en-v1.5**: Open-source, runs locally, good quality, no API cost. Slower on CPU, fast on GPU.
- **Voyage-3**: API-based, optimized for retrieval tasks, excellent quality, moderate cost.
- **text-embedding-3-large** (OpenAI): API-based, very high quality, higher cost. 3,072-dimensional embeddings.

For a corpus of 10,000 papers with an average of 20 chunks per paper (200,000 chunks), embedding costs at current API pricing range from $0 (local model) to approximately $2 (text-embedding-3-large at $0.00001 per token, ~300 tokens per chunk). The cost scales linearly and is a one-time batch operation per paper.

Batch processing matters at scale. Sequential API calls embed one chunk at a time. Batch API calls embed up to 2,048 chunks per request, reducing network overhead and improving throughput from roughly 10 chunks per second (sequential) to over 100 chunks per second (batched).

**Stage 4: Vector Storage.** The input is chunks with their embeddings. The output is a queryable vector store.

pgvector --- the PostgreSQL extension for vector similarity search --- is the right choice for this system because the knowledge management layer already requires PostgreSQL. Using pgvector means embeddings, metadata, and research state all live in one database, queryable with standard SQL.

```sql
-- Schema for the vector store
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE paper_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id TEXT NOT NULL,
    paper_title TEXT NOT NULL,
    section_title TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    cited_references JSONB DEFAULT '[]',
    embedding vector(3072),  -- text-embedding-3-large dimensionality
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX ON paper_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

The HNSW (Hierarchical Navigable Small World) index enables approximate nearest neighbor search. For a corpus of 1 million chunks, a query returns the top 20 most similar chunks in under 100 milliseconds. The `m` and `ef_construction` parameters control the trade-off between index build time, query speed, and recall accuracy.

### The TDD Workflow

This component demonstrates the TDD workflow from Article 2. The human writes the test --- defining what correct retrieval looks like --- before Claude implements the pipeline.

```python
# Human writes this test
def test_retrieval_returns_relevant_chunks():
    """Given 5 papers about transformer efficiency, retrieving
    'what methods improve transformer efficiency?' should return
    chunks from the papers about pruning and quantization."""
    # Setup: ingest 5 known papers
    papers = load_test_fixtures("transformer_efficiency_papers")
    for paper in papers:
        pipeline.ingest(paper)

    # Query
    results = retriever.search("what methods improve transformer efficiency?", top_k=10)

    # Assert: papers about pruning and quantization should be in top results
    result_paper_ids = {r.paper_id for r in results}
    assert "pruning_paper_2024" in result_paper_ids
    assert "quantization_paper_2024" in result_paper_ids

    # Assert: results should include section-level metadata
    for result in results:
        assert result.section_title != ""
        assert result.paper_title != ""
```

The test defines the contract: given these papers, this query should return these results, with this metadata structure. Claude implements the pipeline to satisfy the contract. The human reviews: does the chunking preserve section boundaries? Are citations tracked? Is the embedding model configured correctly?

---

## Semantic Search and Synthesis

The vector store is populated. Now we need to get information out of it --- not just "find relevant chunks" but "answer research questions by reasoning across multiple papers simultaneously." This is the transition from search to synthesis, and it is where the research assistant becomes more than a glorified search engine.

### Query Decomposition

Research questions are complex. "What methods have been used to improve transformer efficiency, and which show the best latency-quality trade-off?" is not a single search query. It contains multiple sub-questions, each requiring different retrieval patterns:

1. "What methods improve transformer efficiency?" --- broad survey question
2. "What are the latency improvements of each method?" --- specific metric extraction
3. "What are the quality trade-offs of each method?" --- specific metric extraction
4. "How do the methods compare on latency versus quality?" --- cross-result synthesis

The query decomposition step breaks a complex research question into sub-queries, each targeted at a specific aspect of the question. Each sub-query hits the vector store independently. The results are merged, deduplicated (the same chunk may be relevant to multiple sub-queries), and ranked by aggregate relevance.

```python
async def decompose_query(question: str) -> list[str]:
    """Break a complex research question into targeted sub-queries."""
    response = await llm.generate(
        system="You are a research librarian. Decompose the following research "
               "question into 2-5 specific sub-queries that, together, would "
               "answer the original question. Return as a JSON array of strings.",
        user=question,
        response_format={"type": "json_object"},
    )
    return json.loads(response)["sub_queries"]
```

### Re-Ranking and Relevance

Initial retrieval from the vector store uses cosine similarity between the query embedding and chunk embeddings. This is fast (sub-100ms for million-chunk corpora with HNSW indexes) but noisy --- cosine similarity captures semantic relatedness, not query-answer relevance. A chunk that discusses transformer efficiency *in passing* has similar cosine similarity to a chunk that *directly answers* the query about transformer efficiency methods.

The two-stage retrieval pattern fixes this:

**Stage 1: Recall.** Retrieve the top-K chunks by cosine similarity (K = 50 to 100). This cast a wide net. Some results will be highly relevant, some tangentially related, some noise.

**Stage 2: Precision.** Re-rank the top-K results using a cross-encoder model. A cross-encoder takes a (query, chunk) pair as input and outputs a relevance score. Unlike the embedding model (which encodes query and chunk independently), the cross-encoder processes them together, enabling fine-grained relevance judgment. Cross-encoders are slower (50 to 200ms per pair) but dramatically more accurate.

```python
async def search(query: str, top_k: int = 20) -> list[SearchResult]:
    """Two-stage retrieval: vector search for recall, cross-encoder for precision."""
    # Stage 1: broad recall
    query_embedding = await embed(query)
    candidates = await vector_store.search(query_embedding, limit=100)

    # Stage 2: precise re-ranking
    scored = []
    for chunk in candidates:
        relevance = cross_encoder.score(query, chunk.text)
        scored.append((chunk, relevance))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [
        SearchResult(chunk=chunk, relevance=score)
        for chunk, score in scored[:top_k]
    ]
```

### Cross-Paper Synthesis

This is the hard part and the core value of the research assistant. Synthesis is not summarizing retrieved chunks. It is reasoning *across* them to identify patterns that no single chunk contains.

The synthesis prompt architecture has three phases:

**Phase 1: Structured extraction.** For each retrieved chunk, extract the key claim, the evidence supporting it, the methodology used, and the paper's metadata. This produces a structured representation of each chunk's contribution:

```json
{
  "chunk_id": "abc123",
  "paper": "Smith et al., 2024",
  "claim": "Pruning reduces inference time by 40% with 1.5% accuracy loss",
  "evidence_type": "experimental",
  "methodology": "structured pruning on ViT-Base, ImageNet-1K",
  "metrics": {"latency_reduction": 0.40, "accuracy_loss": 0.015}
}
```

**Phase 2: Pattern detection.** Across the structured extractions, identify:

- **Agreement:** Multiple papers report consistent results. "Papers A, B, and C all report 30 to 50 percent latency reduction from pruning, with accuracy loss below 2 percent."
- **Contradiction:** Papers disagree on a result. "Paper D reports that pruning *increases* latency on models below 1B parameters, contradicting the consensus from A, B, and C which tested on models above 3B."
- **Gaps:** Areas where no evidence exists. "No paper has tested the combination of pruning and quantization on models in the 1B to 3B parameter range."

**Phase 3: Synthesis generation.** Produce a structured synthesis report that presents the patterns, cites the supporting evidence, and flags uncertainties:

```python
@dataclass
class SynthesisReport:
    question: str
    consensus: list[ConsensusPoint]   # Agreed-upon findings with citations
    contradictions: list[Contradiction]  # Disagreements with context
    gaps: list[Gap]                    # Identified research opportunities
    confidence: str                    # Evidence-grounded confidence statement
    sources: list[SourceChunk]         # All chunks used, with traceability
```

The critical rule from CLAUDE.md applies here: "NEVER present a synthesis claim without listing the supporting paper IDs." Every sentence in the synthesis report traces to specific chunks in the vector store, which trace to specific sections in specific papers. The chain of provenance is unbroken.

### Performance Budget

Interactive use requires fast response times. Here is where the latency goes in a typical query:

| Stage | Latency | Notes |
|-------|---------|-------|
| Query embedding | 50ms | Single API call |
| Vector search | 50-100ms | pgvector HNSW index |
| Cross-encoder re-ranking | 300-500ms | 100 candidates, batched |
| LLM synthesis | 1,000-2,000ms | Claude Sonnet, structured output |
| **Total** | **1,400-2,650ms** | **Within 2s budget (usually)** |

The LLM synthesis step dominates. Optimizations: pre-compute embeddings for common query patterns, cache re-ranking scores for recently retrieved chunks, and batch cross-encoder scoring. For queries that exceed the 2-second budget, return the retrieval results immediately with a "synthesis in progress" indicator and stream the synthesis as it completes.

---

## The Hypothesis Engine

The synthesis engine tells you what the literature says. The hypothesis engine asks: given what the literature says, what should we test next?

Hypothesis generation is where AI assistance is most valuable and most dangerous. Most valuable because the assistant has broader knowledge than any individual researcher --- it has read (through its training data and the vector store) thousands of papers across multiple fields and can identify cross-domain analogies that a specialist would miss. Most dangerous because a plausible-sounding hypothesis that is actually trivial, already tested, or physically impossible wastes months of research effort. The hypothesis engine must produce structured, assessable output --- not free-form brainstorming.

### Prompt Architecture

The hypothesis engine uses structured outputs, not free-form text. Every hypothesis follows a template that forces the model to provide assessable claims:

```python
@dataclass
class Hypothesis:
    statement: str             # Clear, falsifiable claim
    supporting_evidence: list[Evidence]  # Papers that motivate this hypothesis
    novelty_assessment: NoveltyScore     # Is this already explored?
    research_stage: str        # exploratory | confirmatory | mechanistic
    suggested_experiments: list[ExperimentSketch]
    confidence: str            # Evidence-grounded confidence
    assumptions: list[str]     # What must be true for this to work

@dataclass
class Evidence:
    paper_id: str
    claim: str
    relevance: str  # How this evidence supports the hypothesis

@dataclass
class NoveltyScore:
    score: str      # novel | partially_explored | well_established
    similar_work: list[str]  # Papers that have tested similar hypotheses
    differentiation: str     # How this hypothesis differs from prior work
```

The template enforces rigor. A hypothesis without supporting evidence is speculation. A hypothesis without a novelty assessment might duplicate existing work. A hypothesis without suggested experiments is untestable.

### Research Stage Variants

Different stages of research require different kinds of hypotheses. The hypothesis engine adapts its prompt based on the research stage:

**Exploratory:** "What patterns exist in this data?" The assistant generates broad hypotheses aimed at identifying structure. These are appropriate early in a research project when the question is still being defined.

**Confirmatory:** "Does X cause Y?" The assistant generates specific, falsifiable hypotheses with clear predictions. These require existing evidence suggesting the relationship and a concrete experimental test.

**Mechanistic:** "How does X cause Y?" The assistant generates hypotheses about underlying mechanisms, requiring deeper domain knowledge and more detailed experimental designs.

### Novelty Assessment

The most critical check: is this hypothesis already in the literature? The novelty assessment queries the vector store for the hypothesis statement and checks whether existing papers already test it:

```python
async def assess_novelty(hypothesis: str) -> NoveltyScore:
    """Check if a hypothesis has already been explored in the corpus."""
    # Search for the hypothesis statement in the literature
    results = await search(hypothesis, top_k=20)

    # Check for direct matches (someone already tested this)
    direct_matches = [
        r for r in results
        if r.relevance > 0.85  # High relevance = likely already explored
    ]

    if len(direct_matches) >= 3:
        return NoveltyScore(
            score="well_established",
            similar_work=[r.chunk.paper_id for r in direct_matches],
            differentiation="This hypothesis has been extensively tested.",
        )
    elif len(direct_matches) >= 1:
        return NoveltyScore(
            score="partially_explored",
            similar_work=[r.chunk.paper_id for r in direct_matches],
            differentiation="Similar work exists but with different conditions.",
        )
    else:
        return NoveltyScore(
            score="novel",
            similar_work=[],
            differentiation="No existing work directly tests this hypothesis.",
        )
```

### The Human-in-the-Loop Checkpoint

Every generated hypothesis passes through a human review before any experimental design begins. The researcher validates:

1. Does this hypothesis make domain sense? (The assistant might generate a hypothesis that is syntactically valid but physically impossible.)
2. Is the novelty assessment accurate? (The assistant might miss a relevant paper, or might flag a genuinely novel hypothesis as already explored.)
3. Is this worth the computational cost of testing? (Even a novel, plausible hypothesis might not be worth pursuing if the expected impact is low.)

The checkpoint is not optional. It is built into the workflow: the hypothesis engine produces candidates, the researcher filters, and only approved hypotheses proceed to experimental design.

---

## Experiment Design Assistant

A hypothesis without a rigorous experimental design is a wish. The experiment design assistant turns approved hypotheses into testable protocols with statistical grounding.

### Statistical Power Analysis

The first question for any experiment: how large does the study need to be? Power analysis answers this. Given the expected effect size (\(\delta\)), the desired significance level (\(\alpha\)), and the target statistical power (\(1 - \beta\)), the minimum sample size is:

$$n = \frac{(z_{\alpha/2} + z_{\beta})^2 \cdot 2\sigma^2}{\delta^2}$$

where \(z_{\alpha/2}\) is the critical value for the significance level (1.96 for \(\alpha = 0.05\)), \(z_{\beta}\) is the critical value for the desired power (0.84 for power = 0.80), \(\sigma^2\) is the population variance, and \(\delta\) is the minimum effect size you want to detect.

The formula is straightforward. The hard part is choosing the inputs. The expected effect size comes from prior literature (the synthesis engine retrieves relevant benchmarks). The population variance comes from the researcher's data or from published baselines. The significance level and power are conventional (0.05 and 0.80 respectively) unless the domain requires stricter thresholds.

```python
from scipy import stats

def power_analysis(
    effect_size: float,
    variance: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Calculate minimum sample size for a two-sample t-test."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) ** 2 * 2 * variance) / (effect_size ** 2)
    return int(np.ceil(n))
```

### Control Group Design

The experiment design assistant identifies what must be controlled. For ML experiments, common confounds include:

- **Hardware variation:** Same model trained on different GPUs can produce different results due to floating-point ordering differences. Control: fix the hardware or use deterministic training modes.
- **Random seed sensitivity:** Some results only hold for specific random seeds. Control: run each experiment with 5 different seeds and report mean and standard deviation.
- **Data ordering effects:** The order of training samples affects convergence. Control: fix the shuffle seed or report results across multiple orderings.
- **Hyperparameter sensitivity:** The result might depend on a specific hyperparameter configuration. Control: either use the same hyperparameters as the baseline or perform a hyperparameter sweep for both conditions.

The assistant generates a confound checklist specific to the experiment type:

```python
@dataclass
class ExperimentDesign:
    hypothesis: str
    independent_variable: str
    dependent_variables: list[str]
    control_conditions: list[str]
    confounds: list[Confound]
    sample_size: int          # From power analysis
    randomization_strategy: str
    evaluation_metrics: list[Metric]
    stopping_criteria: str
    estimated_compute_cost: str

@dataclass
class Confound:
    name: str
    risk: str        # How it could affect results
    mitigation: str  # How to control for it
```

### Code Generation Rules

The experiment design assistant generates analysis code --- Python scripts that implement the designed experiment. The CLAUDE.md rules for generated analysis code are strict:

```markdown
# Generated Analysis Code Requirements
- Every script must set a random seed at the top: `np.random.seed(42); torch.manual_seed(42)`
- Every script must log library versions: `print(f"numpy=={np.__version__}")`
- Every script must log the data source: path, hash, and modification date
- No hardcoded paths — use environment variables or command-line arguments
- Results must be saved to a structured JSON file, not just printed
- All plots must be saved as both PNG and SVG
```

These rules are not suggestions. They are enforced by the PostToolUse hook that runs mypy and the tests after every edit. If Claude generates a script without a random seed, the test fails, and Claude fixes it.

---

## Knowledge Management

A research assistant without memory is a research assistant that starts over every session. The knowledge management layer gives the system persistent state: what you have read, what you have concluded, what hypotheses you have tested, and how papers connect to each other. This is the difference between a tool you use for isolated questions and a tool that understands your research context.

### Research Journal

The research journal is a structured log of the research process. Not a chat transcript --- a typed, timestamped record with entries categorized by type.

```python
from enum import Enum
from datetime import datetime

class EntryType(Enum):
    HYPOTHESIS = "hypothesis"
    EXPERIMENT = "experiment"
    FINDING = "finding"
    NOTE = "note"
    LITERATURE_REVIEW = "literature_review"

@dataclass
class JournalEntry:
    id: str
    entry_type: EntryType
    timestamp: datetime
    content: str
    citations: list[str]       # Paper IDs referenced
    status: str                # proposed | in_progress | completed | abandoned
    parent_id: str | None      # Links to the hypothesis this experiment tests
    tags: list[str]
    metadata: dict             # Type-specific fields
```

The schema enforces structure. A hypothesis entry must have a novelty assessment. An experiment entry must link to its parent hypothesis. A finding entry must cite the experiment that produced it. This structure makes the journal queryable: "Show me all hypotheses about transformer efficiency that are still in proposed status" is a database query, not a search problem.

```sql
-- Research journal table
CREATE TABLE journal_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entry_type TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    content TEXT NOT NULL,
    citations TEXT[] DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'proposed',
    parent_id UUID REFERENCES journal_entries(id),
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    embedding vector(3072)    -- For semantic search over journal
);

CREATE INDEX ON journal_entries (entry_type, status);
CREATE INDEX ON journal_entries USING gin (tags);
```

### Citation Graph

The citation graph tracks relationships between papers. It has two types of edges:

**Citation edges (directed).** Paper A cites Paper B. These come from the parsed reference lists in the ingestion pipeline.

**Methodology edges (undirected).** Papers A and B use the same methodology but do not cite each other. These are inferred from the embedding similarity of methodology sections.

```python
@dataclass
class CitationGraph:
    papers: dict[str, PaperNode]
    citation_edges: list[tuple[str, str]]      # (citing, cited)
    methodology_edges: list[tuple[str, str]]   # (paper_a, paper_b)

    def papers_supporting(self, claim: str) -> list[str]:
        """Find papers that support a given claim."""
        # Search for the claim in all paper chunks
        results = search(claim, top_k=20)
        return [r.paper_id for r in results if r.relevance > 0.7]

    def papers_contradicting(self, claim: str) -> list[str]:
        """Find papers that contradict a given claim."""
        negated = f"evidence against: {claim}"
        results = search(negated, top_k=20)
        return [r.paper_id for r in results if r.relevance > 0.7]

    def citation_chain(self, paper_a: str, paper_b: str) -> list[str]:
        """Find the shortest citation path between two papers."""
        # BFS over citation edges
        return bfs(self.citation_edges, paper_a, paper_b)
```

The graph enables queries that the vector store alone cannot answer: "What is the citation chain from Paper A to Paper B?" "What papers use the same methodology as Paper C but apply it to a different domain?" "What papers cite all three of Papers D, E, and F?"

### Session Persistence and Memory Architecture

Research context must survive across sessions. The memory architecture has four tiers, each with different persistence, size, and access characteristics:

**Tier 1: Project configuration (CLAUDE.md auto-memory).** Small, permanent, loaded at session start. Contains stable patterns: project conventions, research domain definitions, recurring terminology. This is the CLAUDE.md file and the auto-memory directory. Updated rarely, read every session.

**Tier 2: Research state (PostgreSQL database).** Large, persistent, queried on demand. Contains the research journal, citation graph, experiment records, and findings. This is the structured state of the research project. Updated frequently, queried by specific operations.

**Tier 3: Paper embeddings (pgvector store).** Large, persistent, searched via similarity. Contains all ingested paper chunks and their embeddings. Updated when new papers are ingested, searched by every retrieval operation.

**Tier 4: Session context (conversation).** Ephemeral, limited by context window. Contains the current conversation, in-progress reasoning, and temporary state. Lost when the session ends.

The design principle: important state goes in the database, not in the conversation. When the researcher closes a session and opens a new one, the assistant should know the research context by querying Tiers 1 through 3. The new session starts with: "Last session, you were investigating the contradiction between Paper D's results and the consensus from Papers A, B, and C. You hypothesized that the difference is due to dataset scale. You designed an experiment to test this but had not yet run it."

This handoff is implemented as a session start routine:

```python
async def load_research_context() -> str:
    """Build a context summary from persistent state."""
    recent_entries = await journal.get_recent(limit=10)
    open_hypotheses = await journal.get_by_status("proposed")
    in_progress = await journal.get_by_status("in_progress")
    unresolved = await graph.get_contradictions()

    return format_context(
        recent_entries=recent_entries,
        open_hypotheses=open_hypotheses,
        in_progress=in_progress,
        unresolved_contradictions=unresolved,
    )
```

---

## Observability for AI Systems

Traditional observability monitors three things: logs, metrics, and traces. AI systems need all three, plus domain-specific quality metrics that traditional software does not have. You need to know not just whether the system is running, but whether it is producing *correct* results --- and "correct" in a research context is harder to measure than response codes.

### Structured Logging for LLM Interactions

Every LLM call gets a structured log entry:

```python
@dataclass
class LLMCallLog:
    correlation_id: str     # Traces this call through the full pipeline
    timestamp: datetime
    model: str              # e.g., "claude-3-5-sonnet"
    prompt_hash: str        # SHA-256 of the prompt (not the full text, for cost)
    prompt_tokens: int
    response_tokens: int
    latency_ms: int
    cost_usd: float         # Calculated from token counts and model pricing
    tool_calls: list[str]   # Tools the LLM invoked
    error: str | None

async def log_llm_call(call: LLMCallLog) -> None:
    """Log an LLM call with structured fields."""
    logger.info(
        "llm_call",
        correlation_id=call.correlation_id,
        model=call.model,
        prompt_tokens=call.prompt_tokens,
        response_tokens=call.response_tokens,
        latency_ms=call.latency_ms,
        cost_usd=call.cost_usd,
    )
```

Every retrieval operation gets logged too: the query, the number of results returned, the relevance scores, and the latency. The correlation ID ties everything together: a single research question flows from query decomposition through retrieval through re-ranking through synthesis, and the correlation ID lets you trace the full path.

### Quality Metrics

Three metrics specific to AI research systems:

**Retrieval precision.** What fraction of the chunks returned by a query were actually relevant? Measure this with a labeled test set: queries paired with known-relevant chunks. Run the test set periodically and track precision@10 over time. If precision drops below a threshold (e.g., 0.7), something has changed --- the embedding model drifted, the corpus changed distribution, or a code change broke the retrieval pipeline.

**Hypothesis novelty rate.** What fraction of generated hypotheses are assessed as "novel" (not already in the literature)? A low novelty rate means the assistant is generating ideas that have already been explored. Track this over time to detect whether the hypothesis engine is becoming less creative as the corpus grows.

**User acceptance rate.** What fraction of the assistant's suggestions (hypotheses, experiment designs, synthesis reports) does the researcher actually use? This is the ultimate quality metric --- if the researcher ignores the output, the system is not useful. Track acceptance by logging which outputs the researcher acts on (proceeds to the next step) versus which they discard.

### Cost Tracking

AI system costs are variable, unpredictable, and can spike dramatically. Track costs at three granularities:

**Per-query cost.** Break down the cost of a single research question: embedding generation ($0.001), vector search ($0.000 --- local compute), cross-encoder re-ranking ($0.002), LLM synthesis ($0.01 to $0.05). A typical query costs $0.01 to $0.06. This is the unit economics of the system.

**Per-session cost.** Total tokens consumed, total API calls, total compute time for a research session. A typical 1-hour research session might cost $1 to $5 depending on query volume and synthesis complexity.

**Running total with budget alerts.** Cumulative cost with configurable alerts: "warn at $50/month, block at $100/month." This prevents runaway costs from loops, retries, or unexpectedly large prompts.

```python
class CostTracker:
    def __init__(self, monthly_budget: float = 100.0):
        self.monthly_budget = monthly_budget
        self.monthly_spend = 0.0

    async def record(self, cost: float, operation: str) -> None:
        self.monthly_spend += cost
        await metrics.gauge("cost_monthly_usd", self.monthly_spend)
        await metrics.counter("cost_per_operation", cost, labels={"op": operation})

        if self.monthly_spend > self.monthly_budget * 0.8:
            logger.warning("cost_alert", spend=self.monthly_spend, budget=self.monthly_budget)
        if self.monthly_spend > self.monthly_budget:
            raise BudgetExceededError(f"Monthly budget of ${self.monthly_budget} exceeded")
```

### Alert Patterns

Four alert conditions specific to AI research systems:

**Quality degradation.** Retrieval precision drops below threshold. Trigger: re-run the retrieval test set and compare to baseline. This catches silent failures --- the system still responds, but the answers are wrong.

**Cost spikes.** A single query costs more than 10x the average. Trigger: the LLM entered a loop, generated an unexpectedly long response, or the query decomposition produced too many sub-queries. Alert and investigate.

**Latency anomalies.** A query takes longer than 5 seconds (2.5x the budget). Trigger: the LLM API is rate-limited, the vector store index needs rebuilding, or the corpus grew past the index's efficient range.

**Citation failure rate.** More than 10 percent of synthesis claims lack traceable citations. Trigger: the synthesis prompt is not enforcing the citation requirement, or the retrieval is not returning sufficient source material.

---

## Load Testing the Research Pipeline

An AI research system has different bottlenecks than traditional web applications. The database is not the bottleneck --- the LLM API is. Load testing must target the actual constraints.

### Simulating Concurrent Research Sessions

Multiple researchers using the system simultaneously. Each session generates queries, triggers retrievals, and calls the LLM for synthesis. The load test simulates this:

```javascript
// k6 load test: concurrent research sessions
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 10 },   // Ramp to 10 concurrent users
    { duration: '5m', target: 10 },   // Sustain 10 users
    { duration: '2m', target: 50 },   // Ramp to 50 concurrent users
    { duration: '5m', target: 50 },   // Sustain 50 users
    { duration: '2m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(99)<5000'],  // P99 < 5s (includes LLM time)
    http_req_failed: ['rate<0.01'],     // Error rate < 1%
  },
};

const queries = [
  'What methods improve transformer efficiency?',
  'How does pruning affect model accuracy?',
  'Compare quantization approaches for language models',
  // ... more realistic queries
];

export default function () {
  const query = queries[Math.floor(Math.random() * queries.length)];
  const res = http.post('http://localhost:8000/api/search', JSON.stringify({
    question: query,
    top_k: 20,
  }), { headers: { 'Content-Type': 'application/json' } });

  check(res, {
    'status is 200': (r) => r.status === 200,
    'has results': (r) => JSON.parse(r.body).results.length > 0,
    'has citations': (r) => JSON.parse(r.body).sources.length > 0,
  });

  sleep(Math.random() * 5 + 2);  // 2-7 second think time between queries
}
```

### Vector Database Under Load

Concurrent reads (queries) and writes (new papers being ingested) stress the vector store differently than either alone. Test with:

- **Read-only load:** 50 concurrent queries per second against a corpus of 100K, 500K, and 1M chunks. Measure: P50, P95, P99 latency, and recall@10 (does the index return the same results under load as it does at rest?).
- **Write-during-read:** Ingest new papers while queries are running. Measure: does ingestion degrade query latency? Does the HNSW index need rebuilding after bulk ingestion?
- **Index size scaling:** How does query latency scale as the corpus grows from 10K to 1M chunks? pgvector's HNSW index has sub-linear scaling, but the constants matter.

### LLM API Resilience

The LLM API is the single point of failure with the most variable behavior. Test three scenarios:

**Rate limiting.** Send queries faster than the API allows. Expected behavior: the system queues excess requests, serves results from cache where possible, and degrades gracefully (returns retrieval results without synthesis when the LLM is unavailable).

**Elevated latency.** Simulate the API responding 3 to 5x slower than normal (common during peak hours). Expected behavior: the system respects the latency budget by returning partial results (retrieval without synthesis) and streaming synthesis when it completes.

**Retry with exponential backoff.** When the API returns a 429 (rate limit) or 503 (service unavailable), the system retries with exponential backoff and jitter:

```python
async def call_llm_with_retry(
    prompt: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> str:
    """Call the LLM API with exponential backoff and jitter."""
    for attempt in range(max_retries):
        try:
            return await llm.generate(prompt)
        except (RateLimitError, ServiceUnavailableError) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(
                "llm_retry",
                attempt=attempt + 1,
                delay_s=delay,
                error=str(e),
            )
            await asyncio.sleep(delay)
```

### Embedding Throughput

Bulk paper ingestion (onboarding a new research corpus) stresses the embedding pipeline. For 10,000 papers with 20 chunks each (200,000 chunks), sequential embedding takes approximately 5.5 hours at 10 chunks per second. Batched embedding at 100 chunks per second takes 33 minutes. The cost difference is negligible (both process the same number of tokens), but the time difference determines whether onboarding a new corpus is a "start and go to lunch" operation or a "start and go home for the day" operation.

Test the embedding pipeline at scale:
- 1,000 papers (20K chunks): should complete in under 5 minutes batched
- 10,000 papers (200K chunks): should complete in under 35 minutes batched
- 100,000 papers (2M chunks): should complete in under 6 hours batched

At 100,000 papers, the cost of text-embedding-3-large at approximately $0.00001 per token (300 tokens per chunk, 2M chunks) is roughly $6. This is affordable for a one-time corpus onboarding.

---

## The Complete Architecture

We have built five components, an observability layer, and a load testing suite. Now step back and see the full system.

<svg viewBox="0 0 850 680" xmlns="http://www.w3.org/2000/svg" style="max-width:900px; display:block; margin:2em auto; font-family:Georgia,serif;">
  <!-- Background -->
  <rect width="850" height="680" rx="8" fill="#1a1a2e"/>
  <!-- Title -->
  <text x="425" y="28" text-anchor="middle" fill="#e8e8e8" font-size="15" font-weight="bold">AI Research Assistant: Complete System Architecture</text>

  <!-- External Data Sources -->
  <text x="120" y="55" text-anchor="middle" fill="#888" font-size="10" font-weight="bold">EXTERNAL SOURCES</text>
  <rect x="20" y="62" width="75" height="30" rx="4" fill="#1e2a3a" stroke="#5b9bd5" stroke-width="1"/>
  <text x="57" y="81" text-anchor="middle" fill="#5b9bd5" font-size="9">ArXiv API</text>
  <rect x="105" y="62" width="75" height="30" rx="4" fill="#1e2a3a" stroke="#5b9bd5" stroke-width="1"/>
  <text x="142" y="81" text-anchor="middle" fill="#5b9bd5" font-size="9">Semantic Scholar</text>
  <rect x="190" y="62" width="75" height="30" rx="4" fill="#1e2a3a" stroke="#5b9bd5" stroke-width="1"/>
  <text x="227" y="81" text-anchor="middle" fill="#5b9bd5" font-size="9">Uploaded PDFs</text>

  <!-- Component 1: Ingestion -->
  <rect x="20" y="110" width="250" height="85" rx="6" fill="#1e3a2a" stroke="#6dc98c" stroke-width="1.5"/>
  <text x="145" y="130" text-anchor="middle" fill="#6dc98c" font-size="12" font-weight="bold">1. Literature Ingestion</text>
  <text x="70" y="152" text-anchor="middle" fill="#9ec9a8" font-size="9">PDF Parse</text>
  <text x="145" y="152" text-anchor="middle" fill="#9ec9a8" font-size="9">Chunk</text>
  <text x="215" y="152" text-anchor="middle" fill="#9ec9a8" font-size="9">Embed</text>
  <line x1="95" y1="148" x2="118" y2="148" stroke="#6dc98c" stroke-width="1" marker-end="url(#aG)"/>
  <line x1="168" y1="148" x2="192" y2="148" stroke="#6dc98c" stroke-width="1" marker-end="url(#aG)"/>
  <text x="145" y="180" text-anchor="middle" fill="#6dc98c" font-size="8">Grobid → Section-aware → text-embedding-3-large</text>

  <!-- Arrow: sources → ingestion -->
  <line x1="145" y1="92" x2="145" y2="108" stroke="#5b9bd5" stroke-width="1.5" marker-end="url(#aB)"/>

  <!-- Vector Store (center) -->
  <rect x="310" y="110" width="160" height="55" rx="6" fill="#2a2a3e" stroke="#8888cc" stroke-width="2"/>
  <text x="390" y="135" text-anchor="middle" fill="#8888cc" font-size="12" font-weight="bold">pgvector Store</text>
  <text x="390" y="152" text-anchor="middle" fill="#8888cc" font-size="9">Embeddings + Metadata</text>

  <!-- Arrow: ingestion → vector store -->
  <line x1="270" y1="140" x2="308" y2="140" stroke="#6dc98c" stroke-width="1.5" marker-end="url(#aG)"/>

  <!-- Component 2: Search & Synthesis -->
  <rect x="510" y="110" width="250" height="85" rx="6" fill="#1e2a3a" stroke="#5b9bd5" stroke-width="1.5"/>
  <text x="635" y="130" text-anchor="middle" fill="#5b9bd5" font-size="12" font-weight="bold">2. Search &amp; Synthesis</text>
  <text x="565" y="152" text-anchor="middle" fill="#8bb8d9" font-size="9">Query Decomp</text>
  <text x="635" y="152" text-anchor="middle" fill="#8bb8d9" font-size="9">Re-rank</text>
  <text x="710" y="152" text-anchor="middle" fill="#8bb8d9" font-size="9">Synthesize</text>
  <text x="635" y="180" text-anchor="middle" fill="#5b9bd5" font-size="8">Two-stage retrieval + cross-paper reasoning</text>

  <!-- Arrow: vector store → search -->
  <line x1="470" y1="140" x2="508" y2="140" stroke="#8888cc" stroke-width="1.5" marker-end="url(#aV)"/>

  <!-- LLM Orchestrator (center row) -->
  <rect x="280" y="230" width="220" height="60" rx="8" fill="#2a1e3a" stroke="#a05bd5" stroke-width="2"/>
  <text x="390" y="255" text-anchor="middle" fill="#a05bd5" font-size="13" font-weight="bold">LLM Orchestrator</text>
  <text x="390" y="272" text-anchor="middle" fill="#c08bd9" font-size="9">Claude Sonnet — routes, reasons, generates</text>

  <!-- Arrows: search ↔ LLM -->
  <line x1="580" y1="195" x2="480" y2="235" stroke="#5b9bd5" stroke-width="1.5" marker-end="url(#aB)"/>
  <line x1="500" y1="250" x2="590" y2="195" stroke="#a05bd5" stroke-width="1" stroke-dasharray="4,2"/>

  <!-- Component 3: Hypothesis Engine -->
  <rect x="20" y="320" width="190" height="70" rx="6" fill="#3a2a1e" stroke="#d4944a" stroke-width="1.5"/>
  <text x="115" y="345" text-anchor="middle" fill="#d4944a" font-size="12" font-weight="bold">3. Hypothesis Engine</text>
  <text x="115" y="365" text-anchor="middle" fill="#d9b88b" font-size="9">Structured generation + novelty check</text>
  <text x="115" y="380" text-anchor="middle" fill="#d4944a" font-size="8">Templates, cross-domain analogies</text>

  <!-- Component 4: Experiment Design -->
  <rect x="240" y="320" width="190" height="70" rx="6" fill="#2a1e3a" stroke="#a05bd5" stroke-width="1.5"/>
  <text x="335" y="345" text-anchor="middle" fill="#a05bd5" font-size="12" font-weight="bold">4. Experiment Design</text>
  <text x="335" y="365" text-anchor="middle" fill="#c08bd9" font-size="9">Power analysis, controls, confounds</text>
  <text x="335" y="380" text-anchor="middle" fill="#a05bd5" font-size="8">Code generation for analysis</text>

  <!-- Component 5: Knowledge Management -->
  <rect x="460" y="320" width="190" height="70" rx="6" fill="#3a1e1e" stroke="#e06060" stroke-width="1.5"/>
  <text x="555" y="345" text-anchor="middle" fill="#e06060" font-size="12" font-weight="bold">5. Knowledge Mgmt</text>
  <text x="555" y="365" text-anchor="middle" fill="#d98b8b" font-size="9">Research journal, citation graph</text>
  <text x="555" y="380" text-anchor="middle" fill="#e06060" font-size="8">Session persistence, memory tiers</text>

  <!-- Arrows: LLM → components -->
  <line x1="330" y1="290" x2="130" y2="318" stroke="#d4944a" stroke-width="1.5" marker-end="url(#aO)"/>
  <line x1="390" y1="290" x2="335" y2="318" stroke="#a05bd5" stroke-width="1.5" marker-end="url(#aP)"/>
  <line x1="440" y1="290" x2="540" y2="318" stroke="#e06060" stroke-width="1.5" marker-end="url(#aR)"/>

  <!-- Human-in-the-loop checkpoints -->
  <rect x="680" y="230" width="140" height="60" rx="6" fill="#2a2a4e" stroke="#8888cc" stroke-width="1.5" stroke-dasharray="5,3"/>
  <text x="750" y="253" text-anchor="middle" fill="#8888cc" font-size="11" font-weight="bold">Human-in-Loop</text>
  <text x="750" y="270" text-anchor="middle" fill="#8888cc" font-size="8">Validate hypotheses</text>
  <text x="750" y="282" text-anchor="middle" fill="#8888cc" font-size="8">Approve experiments</text>

  <!-- Arrow: LLM → Human -->
  <line x1="500" y1="260" x2="678" y2="260" stroke="#8888cc" stroke-width="1.5" marker-end="url(#aV)"/>

  <!-- Observability Layer (bottom) -->
  <rect x="20" y="420" width="630" height="55" rx="6" fill="#1e1e2e" stroke="#d4944a" stroke-width="1.5" stroke-dasharray="4,2"/>
  <text x="335" y="445" text-anchor="middle" fill="#d4944a" font-size="12" font-weight="bold">Observability Layer</text>
  <text x="120" y="463" text-anchor="middle" fill="#d9b88b" font-size="9">Structured Logs</text>
  <text x="260" y="463" text-anchor="middle" fill="#d9b88b" font-size="9">Quality Metrics</text>
  <text x="400" y="463" text-anchor="middle" fill="#d9b88b" font-size="9">Cost Tracking</text>
  <text x="540" y="463" text-anchor="middle" fill="#d9b88b" font-size="9">Correlation Tracing</text>

  <!-- Arrows: components → observability -->
  <line x1="115" y1="390" x2="115" y2="418" stroke="#d4944a" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="335" y1="390" x2="335" y2="418" stroke="#d4944a" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="555" y1="390" x2="555" y2="418" stroke="#d4944a" stroke-width="1" stroke-dasharray="3,2"/>

  <!-- Knowledge DB -->
  <rect x="680" y="340" width="140" height="50" rx="6" fill="#2a2a3e" stroke="#e06060" stroke-width="1.5"/>
  <text x="750" y="362" text-anchor="middle" fill="#e06060" font-size="10" font-weight="bold">PostgreSQL</text>
  <text x="750" y="378" text-anchor="middle" fill="#d98b8b" font-size="8">Journal + Graph + Config</text>

  <!-- Arrow: knowledge → DB -->
  <line x1="650" y1="355" x2="678" y2="360" stroke="#e06060" stroke-width="1.5" marker-end="url(#aR)"/>

  <!-- Feedback loop: Knowledge → Literature -->
  <path d="M 555 390 L 555 500 L 145 500 L 145 197" fill="none" stroke="#e06060" stroke-width="1.5" stroke-dasharray="6,3" marker-end="url(#aR)"/>
  <text x="350" y="515" text-anchor="middle" fill="#e06060" font-size="9">Feedback: findings update corpus and hypotheses</text>

  <!-- Legend -->
  <text x="30" y="555" fill="#888" font-size="9">── Data flow</text>
  <text x="150" y="555" fill="#888" font-size="9">- - Feedback / monitoring</text>
  <rect x="300" y="545" width="50" height="14" rx="3" fill="#2a2a4e" stroke="#8888cc" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="360" y="556" fill="#888" font-size="9">= Human checkpoint</text>

  <!-- Arrow defs -->
  <defs>
    <marker id="aG" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><path d="M0,0 L7,2.5 L0,5" fill="#6dc98c"/></marker>
    <marker id="aB" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><path d="M0,0 L7,2.5 L0,5" fill="#5b9bd5"/></marker>
    <marker id="aO" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><path d="M0,0 L7,2.5 L0,5" fill="#d4944a"/></marker>
    <marker id="aP" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><path d="M0,0 L7,2.5 L0,5" fill="#a05bd5"/></marker>
    <marker id="aR" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><path d="M0,0 L7,2.5 L0,5" fill="#e06060"/></marker>
    <marker id="aV" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><path d="M0,0 L7,2.5 L0,5" fill="#8888cc"/></marker>
  </defs>
</svg>

### Best Practice to System Property Mapping

Every Claude Code best practice from Article 2 manifests as a concrete system property in this research assistant. The table makes the mapping explicit:

| Best Practice | System Property | Concrete Example |
|--------------|----------------|------------------|
| CLAUDE.md conventions | Reproducible analysis code | Random seeds, version pinning, data provenance enforced by CLAUDE.md rules |
| Hook systems | Automated quality gates | mypy, ruff, and pytest run on every file change --- type errors and test failures caught at edit time |
| TDD workflow | Verified components | Retrieval accuracy tests define "correct" before pipeline implementation |
| Code review discipline | Domain correctness | Reviews check for citation provenance and statistical validity, not just syntax |
| MCP servers | Extended capabilities | pgvector MCP for direct Claude access to the vector store during development |
| Performance awareness | Latency budgets | < 2s interactive query response enforced by observability alerts |
| Skills and workflows | Reusable research operations | Structured hypothesis generation follows a repeatable template |
| Project structure | Efficient context loading | Feature-based directories mean Claude reads one component at a time |

The mapping is not coincidental. The system was designed so that each best practice has a measurable effect. If you remove the CLAUDE.md rules about random seeds, the analysis code becomes irreproducible. If you remove the hooks, type errors and test failures slip through to commits. If you remove the TDD workflow, the retrieval pipeline is tested after implementation rather than before --- which means "tested" means "we hope it works" rather than "we defined what correct means and verified it."

### What We Built

A five-component research assistant with:

- **Literature ingestion** that turns PDFs into section-aware, citation-preserving chunks with semantic embeddings, searchable via pgvector.
- **Semantic search and synthesis** that decomposes complex research questions, retrieves relevant chunks via two-stage retrieval, and synthesizes cross-paper patterns with full citation provenance.
- **Hypothesis generation** that produces structured, novelty-assessed hypotheses with supporting evidence and suggested experiments, filtered through a human-in-the-loop checkpoint.
- **Experiment design** that calculates statistical power, identifies confounds, and generates reproducible analysis code with mandatory random seeds and version pinning.
- **Knowledge management** that maintains a structured research journal, a queryable citation graph, and a four-tier memory architecture that persists context across sessions.
- **Observability** covering structured logging for every LLM call and retrieval, quality metrics (precision, novelty rate, acceptance rate), cost tracking with budget alerts, and correlation tracing across the full pipeline.
- **Load testing** targeting the actual bottlenecks: LLM API rate limits, vector database scaling, embedding throughput, and concurrent session handling.

### What We Would Do Differently

No retrospective is complete without honest assessment.

**Grobid dependency is heavy.** The Java-based parser requires its own Docker container and adds operational complexity. For a v2, we would evaluate newer ML-based parsers (Nougat, Marker) that run as Python libraries and may offer comparable section extraction with less overhead.

**The hypothesis novelty assessment is coarse.** Checking "is this hypothesis in the literature" via vector similarity is a useful heuristic but not a rigorous novelty check. A paper might test a closely related hypothesis using different terminology that the embedding model does not capture. A v2 would add a structured novelty check that parses hypothesis components (independent variable, dependent variable, mechanism) and searches for each independently.

**Cost tracking should be proactive, not reactive.** The current system alerts when budgets are exceeded. A v2 would estimate the cost of a query *before* executing it and ask for confirmation if the estimated cost is high --- particularly for queries that decompose into many sub-queries.

**The human-in-the-loop checkpoints are binary.** The researcher either approves or rejects. A v2 would support partial approval --- "this hypothesis is interesting but test it on a different dataset" --- with structured feedback that the hypothesis engine can incorporate.

### Where This Goes

The research assistant described in this series is a starting point, not a finished product. The architecture is general: replace the paper corpus with patent filings and you have a patent analysis assistant. Replace it with clinical trial reports and you have a drug discovery assistant. Replace it with internal company documents and you have a competitive intelligence system. The five capabilities --- literature synthesis, hypothesis generation, experimental design, code execution, and knowledge management --- apply to any domain where humans need to reason across large corpora of structured documents.

The tools to build these systems exist now. The engineering discipline to build them well --- CLAUDE.md conventions, hook systems, test-driven development, semantic code review, observability, and load testing --- is what separates a prototype that works in a demo from a system that works in production. This series described both the destination and the path.
