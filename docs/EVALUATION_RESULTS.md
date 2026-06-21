# Evaluation Results

Last updated: 21 June 2026

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
| Router | 30/30 blind holdout predictions | **100% intent accuracy** |
| Router policy | Six actions, 5 cases each | **100% action accuracy** |
| Grader | 20/20 frozen-context predictions | **90.00% accuracy** |
| Grader safety | 10 insufficient contexts | **80.00% recall**, 20.00% FPR |

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

## Router

Benchmark: `data/evaluation/router_holdout_30_v1.jsonl`<br>
Configuration: `router-policy-v2.2`, `gemini-2.5-flash`, temperature `0.1`

| Metric | Numerator / denominator | Result | Gate |
| --- | ---: | ---: | --- |
| Prediction coverage | 30/30 | **100.00%** | MEASURED |
| Intent accuracy | 30/30 | **100.00%** | PASS, >=90% |
| Intent macro F1 | 30 eligible | **100.00%** | PASS, >=85% |
| Policy-action accuracy | 30/30 | **100.00%** | PASS, >=90% |
| Unsafe refusal recall | 5/5 | **100.00%** | PASS, =100% |
| Unsupported refusal recall | 5/5 | **100.00%** | PASS, >=90% |
| Web-required recall | 5/5 | **100.00%** | PASS, >=90% |
| Final prediction error rate | 0/30 | **0.00%** | PASS, <=2% |
| Intent calibration ECE | 30 predictions | **1.77%** | PASS, <=10% |
| Router P95 latency | 30 predictions | 10,461 ms | FAIL, <=6,000 ms |

All four intents and all six policy actions have at least five gold cases. The
confusion matrices are diagonal: there are no false accepts, false rejects, or
observed classification failures. Functional quality is ready for a controlled
internal demo on this benchmark.

Latency remains the limiting factor. Mean/P50/P95 latency is
`6,938 / 6,521 / 10,461 ms`. Thirteen single-attempt cases average 6,122 ms;
17 fallback cases average 7,561 ms. The average key-attempt count is 2.27.
One quota failure occurred during collection and was later retried successfully;
the final scored prediction set contains no error rows.

All confidence values fall in the `0.9-1.0` bucket. Mean intent/policy
confidence is 98.23%/98.10% and observed accuracy is 100% for both. ECE is low,
but calibration at medium and low confidence remains unmeasured.

## Grader

Benchmark: `data/evaluation/grader_eval_20.jsonl`<br>
Configuration: `grader-policy-v1.0`, `gemini-2.5-flash`, temperature `0.1`

The Grader was evaluated only on frozen retrieval contexts. Gold
`context_sufficient` labels were assigned from expected evidence; `requires_web`
was not used as a proxy label.

| Metric | Numerator / denominator | Result | Gate |
| --- | ---: | ---: | --- |
| Prediction coverage | 20/20 | **100.00%** | MEASURED |
| Accuracy | 18/20 | **90.00%** | PASS, >=85% |
| Macro F1 | 20 cases | **89.90%** | Diagnostic |
| Sufficient-context recall | 10/10 | **100.00%** | PASS |
| Insufficient-context recall | 8/10 | **80.00%** | FAIL, >=90% |
| False-positive rate | 2/10 | **20.00%** | FAIL, <=10% |
| False-negative rate | 0/10 | **0.00%** | PASS |
| Runtime error rate | 0/20 | **0.00%** | PASS, <=2% |
| Grader P95 latency | 20 predictions | 16,024 ms | FAIL, <=6,000 ms |

Both errors are unsafe `no -> yes` decisions. One inferred a missing territorial
tax rule from background legal context; the other ignored a direct contradiction
after selecting the more trustworthy source. No false negative or unnecessary
web fallback was observed.

Mean/P50/P95 latency is `7,501 / 4,708 / 16,024 ms`. Three fallback cases
average 12,193 ms; sixteen single-attempt cases average 7,090 ms. Calibration is
not production-ready (ECE 24.15%). The emitted `relevance_score` is not treated
as a context-sufficiency probability, so no deployment threshold is claimed.

Accuracy has a Wilson 95% confidence interval of 69.90%-97.21%. All labels were
assigned by one annotator, and slices with fewer than five cases remain
diagnostic only.

## Evaluation Coverage

- **Published:** corpus structure, the 100-case retrieval benchmark, the completed
  30-case Router holdout with six balanced policy actions, and the completed
  20-case frozen-context Grader holdout.
- **Deferred:** generation, hallucination grader, web search, full E2E, and
  ablation. No score is claimed for these components yet.

## Reproduce

```powershell
python scripts\validate_legal_corpus.py
python scripts\check_ingestion_quality.py
python scripts\run_retrieval_eval.py --dataset data\evaluation\legal_qa_eval_100.jsonl --out eval_reports\retrieval_predictions.jsonl
python scripts\evaluate_legal_qa.py --dataset data\evaluation\legal_qa_eval_100.jsonl --predictions eval_reports\retrieval_predictions.jsonl --component retrieval --out-json eval_reports\retrieval_100.json --out-md eval_reports\retrieval_100.md
python scripts\validate_router_benchmark.py --dataset data\evaluation\router_holdout_30_v1.jsonl
python scripts\run_router_eval.py --dataset data\evaluation\router_holdout_30_v1.jsonl --out eval_reports\router_holdout_30_predictions.jsonl --append --skip-existing --limit 5 --max-errors 1
python scripts\score_router_eval.py --dataset data\evaluation\router_holdout_30_v1.jsonl --predictions eval_reports\router_holdout_30_predictions.jsonl --out-json eval_reports\router_holdout_30.json --out-md eval_reports\router_holdout_30.md
python scripts\validate_grader_benchmark.py --dataset data\evaluation\grader_eval_20.jsonl
python scripts\run_grader_eval.py --dataset data\evaluation\grader_eval_20.jsonl --out eval_reports\grader_predictions.jsonl --append --skip-existing --limit 5 --max-errors 1
python scripts\score_grader_eval.py --dataset data\evaluation\grader_eval_20.jsonl --predictions eval_reports\grader_predictions.jsonl --out-json eval_reports\grader_eval_20.json --out-md eval_reports\grader_eval_20.md
python -m pytest -q
```

Generated predictions and reports are intentionally git-ignored. The repository
keeps the benchmark data, evaluation code, tests, and this curated summary.
