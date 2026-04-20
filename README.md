# RAG-Enabled FOMC Policy Analysis

A RAG-based Q&A system that, given a natural-language query about Fed monetary policy, retrieves relevant passages from FOMC documents and generates a grounded, context-aware answer. The system targets FOMC meeting minutes and statements from 2020–2025, covering the COVID-19 pandemic response, the historic tightening cycle, and the subsequent rate-cutting period.

## Motivation

Staying current with Fed policy requires reading lengthy FOMC documents — minutes alone can exceed 20 pages per meeting. This pipeline democratises access to quality financial NLP by enabling a lightweight 7B-parameter open-source model to produce accurate, grounded policy analysis without API costs or proprietary dependencies. RAG is essential here: the base model alone frequently hallucinates specific dates, statistics, and other verifiable details.

## Data

- **Source:** `vtasca/fomc-statements-minutes` (HuggingFace) — FOMC statements and minutes from 2000–2025
- **Filtered to:** 2020–2025 (~96 documents: 48 meetings × 2 document types)
- **Text cleaning:** encoding artifact repair, boilerplate removal (attendance lists, vote sections, footnotes), section header preservation

## Pipeline Architecture

```
User Query
    └─> Query Date Parser (regex-based temporal extraction)
           ├─ date found → Pre-filter chunks by date range
           │                  └─> Semantic Search (FAISS) + Keyword Search (BM25)
           └─ no date   → Full Corpus Search
                              └─> Semantic Search (FAISS) + Keyword Search (BM25)
                                         └─> RRF Merge → Top-K Passages → Mistral 7B Instruct
```

### Key Components

| Stage | Approach | Detail |
|---|---|---|
| Chunking | Paragraph-aware | 512-word target, 50-word overlap; statements kept whole |
| Embedding | BAAI/bge-base-en-v1.5 | 768-dim, hard-negative mining, retrieval-optimised |
| Indexing | FAISS (IndexFlatIP) | Cosine similarity via normalised inner product |
| Keyword search | BM25Okapi | k1=1.5, b=0.75 (original BM25 defaults) |
| Fusion | Reciprocal Rank Fusion | k=60; merges semantic and BM25 rankings |
| Generation | Mistral 7B Instruct v0.3 | 4-bit NF4 quantisation via bitsandbytes (~4 GB VRAM) |

Date pre-filtering was a critical optimisation: without it, relevant early-2024 chunks ranked at position 182+ because FOMC documents use highly similar language across years.

## Evaluation

Evaluated across 15 queries spanning 9 topics and 3 difficulty levels using automatic metrics and LLM-as-judge scoring (Claude Haiku 4.5 via API).

### RAG vs Base Model

| Metric | RAG | Base | Δ |
|---|---|---|---|
| ROUGE-L | 0.236 | 0.140 | +69% |
| Faithfulness (1–5) | 2.667 | 1.267 | +111% |
| Relevance (1–5) | 4.667 | 4.467 | +0.200 |
| Completeness (1–5) | 3.400 | 3.667 | −0.267 |

### Retrieval Performance

| Query Type | Mean Precision@k | Mean MRR |
|---|---|---|
| With date reference (n=6) | 0.667 | 1.000 |
| Without date reference (n=9) | 0.378 | 0.806 |
| Overall | 0.493 | 0.883 |

RAG's biggest win is faithfulness (+111%) — it grounds every claim in an actual source document. Date-specific queries achieve perfect MRR (1.0), confirming the pre-filtering mechanism works as intended. The base model scores slightly higher on completeness, as it freely generates elaborations that are often unverifiable.

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

```env
ANTHROPIC_API_KEY=sk-ant-...   # for Claude Haiku evaluation judge
HF_TOKEN=hf_...                # for loading the FOMC dataset from HuggingFace
```

### Running the pipeline

```bash
python main.py --query "What was the Fed's stance on inflation in March 2023?"
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--query` | required | Natural language question about FOMC policy |
| `--top-k` | 5 | Number of passages to retrieve |
| `--no-date-filter` | off | Disable date-based pre-filtering |

On first run, `main.py` downloads the FOMC dataset and both embedding/generation models from HuggingFace — this takes a few minutes. Subsequent runs reuse cached models.

The notebook (`fomc_rag_pipeline.ipynb`) is designed to run on Google Colab with a free T4 GPU. Models are cached to Google Drive to avoid re-downloading across sessions. The `src/` modules can be used independently of the notebook.

## Limitations & Future Work

- Completeness is slightly lower for RAG due to context window constraints
- BM25 parameters (k1, b) were not tuned on a domain-specific relevance dataset
- Temporal queries without explicit dates still struggle with semantic similarity across years
- Future directions: cross-encoder reranking, query expansion, larger/fine-tuned embedding models
