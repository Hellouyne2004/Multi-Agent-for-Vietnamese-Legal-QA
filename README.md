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
- FastAPI endpoints with synchronous, SSE streaming, and trace/debug responses, plus a React/Vite frontend.

## Evaluation Snapshot

The project uses component-wise evaluation so that corpus, retrieval, routing,
generation, and runtime failures are measured independently. See the
[`Evaluation Results`](docs/EVALUATION_RESULTS.md) for denominators, limitations,
and reproduction commands.

| Component | Scope | Current result |
| --- | --- | ---: |
| Corpus | 4 documents, 527 chunks | 0 missing required metadata fields |
| Retrieval | 100/100 benchmark predictions | **97.80% Doc Hit@5** |
| Retrieval | 91 eligible cases | **97.25% standard MRR@5** |
| Retrieval | 100 predictions | **0% runtime errors**, 526.55 ms P95 |
| Router | 30-case blind holdout | **100% intent and policy accuracy** |
| Router policy gates | 5 cases per action | **100% recall across all 6 actions** |
| Grader | 20-case frozen-context holdout | **90.00% accuracy**, 89.90% macro F1 |
| Grader safety | 10 insufficient contexts | **80.00% recall**, 20.00% false-positive rate |

All Router functional gates pass on the completed frozen holdout. Router P95
latency is 10.46 seconds, which currently remains above the strict 6-second target. Future optimizations will focus on reducing this latency to meet high-throughput production requirements.
The Grader also remains pre-production: it accepts all 10 sufficient contexts,
but incorrectly accepts 2/10 insufficient contexts and has 16.02-second P95
latency.

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
eval_reports/                Local generated reports (git-ignored)
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
- `GET /api/v1/qa/trace/{trace_id}`

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

The current retrieval report has full prediction coverage on 100 benchmark
cases and records latency, quality gates, and failure categories. Reports also
track `prediction_coverage`, so partial runs are not mistaken for complete
benchmark results.

To execute comprehensive end-to-end evaluation, build the curated 40-case benchmark:

```bash
python scripts/prepare_e2e_benchmark.py
python scripts/run_e2e_eval.py --dataset data/evaluation/legal_qa_eval_e2e_40.jsonl --out eval_reports/e2e_predictions_40.jsonl --skip-existing
python scripts/evaluate_legal_qa.py --dataset data/evaluation/legal_qa_eval_e2e_40.jsonl --predictions eval_reports/e2e_predictions_40.jsonl --component e2e --only-predicted --out-json eval_reports/e2e_40.json --out-md eval_reports/e2e_40.md
```

The curated public summary is in
[`docs/EVALUATION_RESULTS.md`](docs/EVALUATION_RESULTS.md). Raw predictions and
generated reports stay local under `eval_reports/` and can be reproduced with
the evaluation scripts.

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
- E2E-40 benchmark schema, route-action scoring, hallucination retry metrics, and API trace events.

## Observability

Every API answer includes a `trace_id` and compact `agent_events` for recent
requests. The trace endpoint returns the same per-agent audit trail so router,
retriever, grader, generator, and hallucination behavior can be debugged without
printing raw secrets:

```bash
curl http://localhost:8000/api/v1/qa/trace/<trace_id>
```

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
- Gemini free-tier quotas can interrupt full E2E runs; use `--skip-existing` and publish only completed prediction sets.
- LLM-as-judge is optional and should be used as a secondary signal, not as the only source of truth.
- Ablation comparison is scored from prediction files; missing variant files are reported as missing rather than filled with synthetic scores.
- **Evaluation Summary**: This repository implements a rigorous component-wise evaluation framework for Vietnamese Legal Agentic RAG. It benchmarks hybrid retrieval on 100 cases (achieving 97.8% Doc Hit@5 and 97.25% MRR@5), demonstrates 100% intent/policy accuracy on a frozen 30-case Router holdout, and evaluates CRAG context sufficiency on 20 frozen cases (90% accuracy) with explicit safety and latency gates.
