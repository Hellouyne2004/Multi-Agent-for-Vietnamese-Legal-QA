# Multi-Agent RAG for Vietnamese Legal QA

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/framework-LangGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/vector--db-Qdrant-red.svg)](https://qdrant.tech/)
[![FastAPI](https://img.shields.io/badge/api-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

Vietnamese legal question-answering system built with **LangGraph multi-agent RAG**, **Qdrant hybrid retrieval**, **FastAPI**, **React**, citation grounding, and hallucination checks.

## Features

- LangGraph workflow with Router, Retriever, Grader, Web Searcher, Generator, and Hallucination Grader nodes.
- Legal-aware ingestion with PDF/OCR extraction, article/clause/point metadata, and smart chunking for Vietnamese legal documents.
- Qdrant hybrid retrieval with dense vectors, BM25-style sparse vectors, metadata filters, reciprocal rank fusion, and optional reranking.
- Citation validation and deterministic hallucination checks before optional LLM review.
- FastAPI endpoints with synchronous and SSE streaming responses, plus a React/Vite frontend.

## Evaluation Snapshot

The evaluation suite is designed to measure the system by layer instead of treating the multi-agent graph as a black box. Under Gemini free-tier limits, the current public results focus on a **100-case retrieval benchmark** and a smaller **20-case end-to-end benchmark** for quota-aware validation.

| Area | Metric | Current snapshot |
| --- | ---: | ---: |
| Corpus | Registry documents | 4 |
| Corpus | Chunks | 527 |
| Corpus | Missing metadata | 0 |
| Benchmark | Retrieval cases | 100 |
| Benchmark | E2E subset cases | 20 |
| Retrieval report | Metrics | Doc Hit@5, Article Hit@5, Clause Hit@5, MRR, latency |

The benchmark covers the current 4-document indexed corpus with direct legal lookup, procedural questions, table lookup, out-of-scope routing, unsafe requests, and hallucination traps. Failure analysis is used to separate corpus issues, retrieval misses, citation problems, and unsupported-answer behavior.

## Architecture

![System architecture](docs/assets/system-architecture.png)

```text
User question
  -> Router Agent
  -> Retriever Agent: Qdrant hybrid retrieval + metadata filters
  -> Grader Agent: context relevance / completeness
  -> Web Searcher Agent: fallback when internal context is insufficient
  -> Generator Agent: grounded answer with citations
  -> Hallucination Grader: rule checks + optional LLM self-check
  -> Final answer + citations + traceable state
```

### Main Agents

- **Router**: classifies legal, non-legal, and conversational inputs.
- **Retriever**: combines semantic search, sparse keyword retrieval, metadata filters, and context expansion.
- **Grader**: checks whether retrieved context is enough for answer generation.
- **Web Searcher**: optional Tavily fallback when local legal data is insufficient.
- **Generator**: synthesizes a Vietnamese answer with source IDs and citations.
- **Hallucination Grader**: rejects missing citations, invalid source IDs, unsupported numbers, and malformed citation metadata before optional LLM review.

## Tech Stack

- **LLM orchestration**: LangGraph, LangChain, Gemini
- **Retrieval**: Qdrant, multilingual sentence embeddings, BM25-style sparse vectors
- **Backend**: FastAPI, Pydantic, SSE streaming
- **Frontend**: React, Vite, Tailwind CSS
- **Data processing**: PyMuPDF, pdfplumber, Tesseract OCR, custom legal chunking
- **Testing/evaluation**: pytest, offline JSONL benchmark, generated evaluation reports

## Repository Layout

```text
api/                         FastAPI app and QA router
data/evaluation/             Public benchmark seed for repeatable evaluation
data/processed/              Document registry and ingestion quality summary
eval_reports/                Generated evaluation reports and failure analysis
frontend/                    React/Vite UI
scripts/                     Indexing, evaluation, observability, and utility scripts
src/                         Agents, graph state, data pipeline, models, utilities
tests/                       Smoke tests for API contracts and deterministic checks
docker-compose.yml           Qdrant service definition
requirements.txt             Python dependencies
```

## Setup

### 1. Create Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Secrets

Copy `.env.example` to `.env` and set keys locally:

```bash
GEMINI_API_KEY=...
GEMINI_API_KEY_1=...
TAVILY_API_KEY=...
QDRANT_URL=http://localhost:6333
```

Security checklist:
- Keep `.env` local only.
- Confirm `.env` is ignored by git.
- Do not print raw API keys in logs.

### 3. Start Qdrant

```bash
docker-compose up -d qdrant
```

### 4. Build the Index

```bash
python scripts/build_index.py
```

### 5. Run the API

```bash
python api/main.py
```

API endpoints:

- `GET /`
- `POST /api/v1/qa/ask`
- `GET /api/v1/qa/stream?question=...`

### 6. Run the Frontend

```bash
cd frontend
npm install
npm run dev
```

## Evaluation

The evaluator is organized around three quality layers:

- **Corpus quality**: metadata completeness, chunk length distribution, legal/table structure coverage, OCR quality summary.
- **Component quality**: router intent accuracy, retrieval Doc/Article/Clause Hit@k, MRR, grader decision accuracy, citation validity, deterministic hallucination checks.
- **End-to-end quality**: fact coverage, forbidden fact rate, grounded answer rate, refusal accuracy, unsupported numeric claims, retry count, web fallback usage, latency, and error rate.

The current retrieval report has full prediction coverage on 100 benchmark cases and records latency, quality gates, and failure categories. Reports also track `prediction_coverage`, so partial E2E runs are not mistaken for full benchmark failures.

Generated reports are written to `eval_reports/retrieval_100.md` and `eval_reports/failure_analysis.md` after running the retrieval evaluator.

## Tests

```bash
pytest -q
```

The smoke tests focus on deterministic contracts:
- FastAPI root response shape.
- Rule-based hallucination/citation checks.
- Retrieval filter parsing for article/clause queries.
- Offline evaluation metrics for retrieval, answer quality, refusal accuracy, and quality gates.
- Quota-friendly component evaluation with `--component retrieval`, `--component e2e`, and `--only-predicted`.

## Key Rotation Observability

The LLM factory does not assign one fixed API key per agent. It hashes the agent purpose, selects a stable primary key index, and then tries the remaining keys as fallbacks. Logs show only key indices, never raw keys.

Check the routing plan without calling Gemini:

```bash
python scripts/check_key_rotation_observability.py
```

Example safe log shape:

```text
[LLM] agent=generator model=gemini-1.5-flash primary_key_index=2 fallback_key_indices=[3, 0, 1] json_mode=True
[LLM] agent=generator key_index=2 failed reason=quota/rate_limit
[LLM] agent=generator key_index=3 succeeded
```

## Limitations

- The default report remains offline and deterministic; full graph evaluation requires Qdrant plus configured LLM credentials.
- Gemini free-tier quotas make a full 100-case E2E run impractical; use 100-case retrieval metrics and the 20-case E2E subset for CV-ready reporting.
- LLM-as-judge is optional and should be used as a secondary signal, not as the only source of truth.
- Ablation comparison is scored from prediction files; missing variant files are reported as missing rather than filled with synthetic scores.
- The CV-ready evaluation bullet: "Designed and implemented a component-wise evaluation framework for a Vietnamese Legal Multi-Agent RAG system, covering corpus quality, 100-case hybrid retrieval benchmarking, quota-aware E2E evaluation, prediction coverage tracking, citation validation, hallucination checks, latency metrics, ablation studies, and failure analysis."
