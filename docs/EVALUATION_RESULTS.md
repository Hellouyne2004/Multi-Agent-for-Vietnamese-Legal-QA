# Evaluation Results

Last updated: 20 June 2026

This page contains the curated, reproducible evaluation snapshot for the
Vietnamese Legal Multi-Agent RAG project. Only completed benchmark results are
published; partial API runs remain local until their frozen evaluation set is
complete.

## Highlights

| Area | Scope | Result |
| --- | --- | ---: |
| Corpus | 4 legal documents | 527 traceable chunks |
| Metadata | Required document, page, source, and hierarchy fields | 0 missing |
| Retrieval | 100/100 benchmark predictions | **97.80% Doc Hit@5** |
| Ranking | 91 eligible retrieval cases | **97.25% standard MRR@5** |
| Reliability | 100 retrieval predictions | **0% runtime errors** |
| Latency | 100 retrieval predictions | **526.55 ms P95** |

## Corpus

| Metric | Result |
| --- | ---: |
| Registry documents | 4 |
| Total chunks | 527 |
| Article / clause / point chunks | 236 / 219 / 57 |
| Table chunks | 6 |
| Missing required metadata | 0 |
| Chunk length, min / mean / max | 201 / 843.4 / 2,199 characters |

The ingestion pipeline preserves document and page traceability and represents
article, clause, point, and table structures. OCR is still the main corpus risk:
458/527 chunks originate from OCR and some contain character corruption or
broken legal headings.

## Retrieval

Benchmark: `data/evaluation/legal_qa_eval_100.jsonl`  
Configuration: hybrid dense + sparse retrieval, reciprocal-rank fusion, Top-5

| Metric | Numerator / denominator | Result |
| --- | ---: | ---: |
| Prediction coverage | 100/100 cases | **100.00%** |
| Doc Hit@5 | 89/91 eligible cases | **97.80%** |
| Standard MRR@5 | 88.5/91 | **97.25%** |
| Runtime error rate | 0/100 | **0.00%** |
| Median latency | 100 samples | **430.5 ms** |
| Warm mean latency | 99 samples | **448.9 ms** |
| P95 latency | 100 samples | **526.55 ms** |

Document selection is strong, but answer-ready evidence remains the next
engineering target:

| Evidence metric | Numerator / denominator | Result |
| --- | ---: | ---: |
| Context Fact Coverage@5 | 110/186 facts | 59.14% |
| Full Fact Case Rate@5 | 47/90 cases | 52.22% |
| Forbidden Fact in Context@5 | 4/90 cases | 4.44% |

The audit found that 151/186 expected facts are exact-matchable in their source
document. Retrieval covers 110/151 of that measurable ceiling (72.85%). The
remaining gap combines ranking misses, OCR noise, benchmark paraphrases,
article metadata defects, and multi-row table contamination.

## Evaluation Coverage

- **Published:** corpus structure and 100-case retrieval benchmark.
- **Prepared:** a frozen 30-case Router holdout with six balanced policy
  actions; results will be published after 30/30 prediction coverage.
- **Deferred:** generation, grader, hallucination grader, web search, full E2E,
  and ablation. No score is claimed for these components yet.

## Reproduce

```powershell
python scripts\validate_legal_corpus.py
python scripts\check_ingestion_quality.py
python scripts\run_retrieval_eval.py --dataset data\evaluation\legal_qa_eval_100.jsonl --out eval_reports\retrieval_predictions.jsonl
python scripts\evaluate_legal_qa.py --dataset data\evaluation\legal_qa_eval_100.jsonl --predictions eval_reports\retrieval_predictions.jsonl --component retrieval --out-json eval_reports\retrieval_100.json --out-md eval_reports\retrieval_100.md
python scripts\validate_router_benchmark.py --dataset data\evaluation\router_holdout_30_v1.jsonl
python -m pytest -q
```

Generated predictions and reports are intentionally git-ignored. The repository
keeps the benchmark data, evaluation code, tests, and this curated summary.
