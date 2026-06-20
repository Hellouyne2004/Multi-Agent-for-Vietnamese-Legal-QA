# Evaluation Results

Last updated: 20 June 2026

This report presents the latest reproducible results for the Vietnamese Legal
Multi-Agent RAG project. Metrics are reported by component so that retrieval,
routing, generation, and runtime failures are not mixed into one score.

## Executive Summary

| Component | Evaluation scope | Result | Status |
| --- | --- | --- | --- |
| Corpus | 4 legal documents, 527 chunks | Complete trace metadata; legal hierarchy and tables preserved | Measured with OCR caveats |
| Retrieval | 100-case benchmark, 100/100 predictions | Doc Hit@5 97.80%; standard MRR@5 97.25%; runtime errors 0% | Strong document routing; evidence coverage needs improvement |
| Router | Blind 30-case holdout, 18/30 rows collected | 17/17 successful predictions correct for both intent and policy action | Partial, blocked by API quota |
| Generation / Grader / Hallucination | No valid benchmark run yet | Not reported | Deferred to preserve API quota |
| Full E2E / Ablation | No comparable complete run yet | Not reported | Deferred until upstream gates improve |

**Readiness:** the project is suitable for a controlled internal demo and for
showing the evaluation architecture in an interview. It is not presented as
production-ready because retrieval evidence coverage is below target and the
Router holdout is incomplete.

## Portfolio Highlights

- Built a deterministic, component-wise evaluation pipeline with explicit
  prediction coverage, eligible denominators, quality gates, latency, failure
  taxonomy, and quota-aware resume support.
- Evaluated hybrid retrieval on **100/100 benchmark cases** with **97.80%
  Doc Hit@5**, **97.25% standard MRR@5**, and **0/100 runtime errors**.
- Created a blind Router policy holdout covering six actions; the current run
  achieved **17/17 correct successful predictions** before API quota exhaustion.
- Traced low evidence coverage to retrieval ranking, OCR, benchmark wording,
  chunk metadata, and table-row contamination instead of attributing every miss
  to the embedding model.

## 1. Corpus Quality

| Metric | Result |
| --- | ---: |
| Registry documents | 4 |
| Total chunks | 527 |
| Article / clause / point chunks | 236 / 219 / 57 |
| Table chunks | 6 |
| Missing required metadata | 0 |
| Chunk length, min / mean / max | 201 / 843.4 / 2,199 characters |
| OCR-derived chunks | 458/527 |

All chunks contain document, page, source path, source URL, and required legal
hierarchy fields. The corpus is structurally traceable, but it is not declared
fully clean: OCR-heavy documents still contain corrupted Vietnamese characters,
broken headings, and digit/letter substitutions. These defects affect exact
fact matching and some article metadata.

## 2. Retrieval Evaluation

Benchmark: `data/evaluation/legal_qa_eval_100.jsonl`  
Predictions: 100/100 cases  
Retriever: dense + sparse hybrid retrieval with reciprocal-rank fusion, Top-5

| Metric | Numerator / denominator | Result | Gate |
| --- | ---: | ---: | --- |
| Doc Hit@5 | 89/91 eligible cases | **97.80%** | PASS, >=95% |
| Standard MRR@5 | 88.5/91 | **97.25%** | PASS, >=80% |
| Context Fact Coverage@5 | 110/186 facts | 59.14% | Below 80% target |
| Full Fact Case Rate@5 | 47/90 cases | 52.22% | Below 70% target |
| Forbidden Fact in Context@5 | 4/90 cases | 4.44% | PASS, <=5% |
| Runtime error rate | 0/100 | **0.00%** | PASS, <=5% |
| Median retrieval latency | 100 samples | **430.5 ms** | Measured |
| Warm mean latency | 99 samples | **448.9 ms** | Measured |
| P95 retrieval latency | 100 samples | **526.55 ms** | Measured |

The retriever reliably selects the correct document, but correct-document hits
do not guarantee answer-ready evidence. Of 186 expected facts, only 151 are
exact-matchable in their expected source document; retrieval covers 110/151
of this measurable ceiling (**72.85%**). Among 43 cases initially flagged for
missing context facts, the audit identifies 16 retrieval-only cases, 5 mixed
cases, and 22 source-label or exact-matching cases.

Diagnostic metadata results are Article Hit@5 `51/75 = 68.00%`, Clause Hit@5
`1/3 = 33.33%`, and Level Hit@5 `80/91 = 87.91%`. Point Hit@5 remains
`N/A_NO_LABEL` because there are no eligible point-level gold labels.

## 3. Router Evaluation

Benchmark: `router-holdout-30-v1.0`  
Prompt: `router-policy-v2.2`  
Model: `gemini-2.5-flash`, temperature `0.1`

| Metric | Numerator / denominator | Result | Status |
| --- | ---: | ---: | --- |
| Collected prediction rows | 18/30 | 60.00% | INVALID_PARTIAL |
| Successful predictions | 17/18 rows | 94.44% | One quota failure |
| Intent accuracy | 17/17 successful | **100.00%** | Partial |
| Policy-action accuracy | 17/17 successful | **100.00%** | Partial |
| Retrieve recall | 5/5 | **100.00%** | Measured |
| Out-of-scope redirect recall | 5/5 | **100.00%** | Measured |
| General-chat response recall | 5/5 | **100.00%** | Measured |
| Unsafe refusal recall | 2/2 observed | **100.00%** | Sample risk |
| Unsupported refusal recall | 0/5 run | N/A_NOT_RUN | Blocked by quota |
| Web-required recall | 0/5 run | N/A_NOT_RUN | Blocked by quota |
| Runtime error rate | 1/18 | 5.56% | Quota-related |
| P95 observed latency | 17 successful | 10,729 ms | Fallback affected |

The observed classification quality is promising across four policy actions.
The run is not reported as 100% Router accuracy because 13 successful calls are
still required to complete the holdout. Every successful request used API-key
fallback, so the observed latency measures retry behavior rather than clean
single-attempt model latency.

## 4. Deferred Components

Generation, relevance grading, hallucination grading, web-search quality, full
E2E response quality, and ablation comparisons are intentionally not assigned
scores. Existing API quota is reserved for completing the Router holdout, and
the retrieval evidence gates already explain a major upstream limitation.

This is recorded as `N/A_NOT_RUN` or `BLOCKED_QUOTA`, never as zero, PASS, or
FAIL. A new E2E result will only be published when predictions match the current
benchmark IDs and provide enough valid cases for meaningful denominators.

## 5. Main Findings

1. **Document selection is strong:** 89/91 Doc Hit@5 and 97.25% standard MRR.
2. **Evidence completeness is the main quality bottleneck:** only 110/186 gold
   facts appear in Top-5 context under the current deterministic matcher.
3. **Measurement quality matters:** 35/186 expected facts are not exact-matchable
   in their own source due mainly to paraphrase/label wording and partly to OCR.
4. **Router policy behavior is promising but incomplete:** all 17 successful
   holdout predictions are correct, while unsupported and web actions remain
   unmeasured.
5. **Quota affects reliability and latency:** Router fallback occurred on 17/17
   successful holdout requests and caused the only recorded Router error.

## Reproduction

```powershell
python scripts\validate_legal_corpus.py
python scripts\check_ingestion_quality.py
python scripts\run_retrieval_eval.py --dataset data\evaluation\legal_qa_eval_100.jsonl --out eval_reports\retrieval_predictions.jsonl
python scripts\evaluate_legal_qa.py --dataset data\evaluation\legal_qa_eval_100.jsonl --predictions eval_reports\retrieval_predictions.jsonl --component retrieval --out-json eval_reports\retrieval_100.json --out-md eval_reports\retrieval_100.md
python scripts\validate_router_benchmark.py --dataset data\evaluation\router_holdout_30_v1.jsonl
python scripts\score_router_eval.py --dataset data\evaluation\router_holdout_30_v1.jsonl --predictions eval_reports\router_holdout_30_predictions.jsonl --out-json eval_reports\router_holdout_30.json --out-md eval_reports\router_holdout_30.md
```

Detailed evidence is available in:

- `data/processed/ingestion_quality_report.md`
- `eval_reports/retrieval_100.md`
- `eval_reports/retrieval_audit.md`
- `eval_reports/failure_analysis.md`
- `eval_reports/router_holdout_30.md`
- `eval_reports/router_overall_evaluation.md`
