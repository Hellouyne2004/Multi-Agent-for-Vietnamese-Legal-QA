# Legal QA Evaluation Report

- Dataset: `data\evaluation\legal_qa_eval_100.jsonl` (100 cases)
- Mode: scored_retrieval_predictions
- Predictions: `eval_reports\retrieval_predictions.jsonl`
- Component: `retrieval`
- Prediction coverage: 100/100 dataset cases (100.00%)
- Prediction rows: 100 rows, 100 unique case IDs

## Quality Gates

| Gate | Value | Rule | Status |
| --- | --- | --- | --- |
| corpus_missing_metadata_total | 0 | lte 0 | PASS |
| corpus_chunk_avg_chars | 843.4 | between (500, 1200) | PASS |
| router_intent_accuracy | n/a | gte 0.9 | N/A |
| router_refusal_accuracy | n/a | gte 0.9 | N/A |
| retrieval_doc_hit_at_k | 97.80% | gte 0.95 | PASS |
| retrieval_article_hit_at_k | 68.00% | gte 0.85 | FAIL |
| retrieval_clause_hit_at_k | 33.33% | gte 0.75 | FAIL |
| retrieval_mrr | 99.44% | gte 0.8 | PASS |
| generation_fact_coverage | n/a | gte 0.75 | N/A |
| generation_forbidden_fact_rate | n/a | lte 0.05 | N/A |
| generation_display_citation_valid | n/a | gte 0.95 | N/A |
| generation_citation_url_valid | n/a | gte 0.95 | N/A |
| e2e_grounded_answer_rate | n/a | gte 0.8 | N/A |
| e2e_unsupported_claim_rate | n/a | lte 0.08 | N/A |
| reliability_error_rate | n/a | lte 0.05 | N/A |

## Corpus Quality

- Source: `data/processed/chunks.jsonl`
- Registry documents: 4
- Chunks: 527
- Chunk chars: min=201, avg=843.4, max=2199
- Missing metadata total: 0
- Levels: `{'preamble': 3, 'article': 236, 'clause': 219, 'point': 57, 'document': 6, 'table': 6}`

## Router And Agent Decisions

| Metric | Value |
| --- | ---: |
| Router cases | 0 |
| Intent accuracy | n/a |
| Refusal/out-of-scope accuracy | n/a |
| Grader accuracy | n/a |

## Retrieval Summary

| Metric | Value |
| --- | ---: |
| Cases | 100 |
| Doc Hit@5 | 97.80% |
| Article Hit@5 | 68.00% |
| Clause Hit@5 | 33.33% |
| Point Hit@5 | n/a |
| Level Hit@5 | 87.91% |
| MRR | 99.44% |
| Avg retrieval latency | 537.39 ms |
| P95 retrieval latency | 526.55 ms |

## Answer And E2E Summary

| Metric | Value |
| --- | ---: |
| Cases | 0 |
| Fact Coverage | n/a |
| Forbidden Fact Rate | n/a |
| Display Citation Valid | n/a |
| Citation URL Valid | n/a |
| Unsupported Claim Rate | n/a |
| Refusal Accuracy | n/a |
| Grounded Answer Rate | n/a |
| Web Fallback Rate | n/a |
| Error Rate | n/a |
| Quota/Rate-limit Error Rate | n/a |
| Runtime Error Rate | n/a |
| Avg processing latency | n/a ms |
| P95 processing latency | n/a ms |
| Avg generation attempts | n/a |

## Failure Analysis

| Category | Cases | Sample IDs |
| --- | --- | --- |
| wrong_article | 24 | labor_contract_types_008, labor_employee_termination_009, labor_strike_definition_015, labor_collective_bargaining_016, labor_contract_content_018, labor_assignment_020, labor_part_time_022, labor_illegal_termination_024 |
| wrong_clause | 2 | labor_working_time_001, labor_annual_leave_005 |
| wrong_doc | 2 | labor_wage_scale_027, cyber_transition_previous_law_078 |

## Ablation Summary

| Variant | Cases | Doc Hit@5 | Article Hit@5 | Clause Hit@5 | Fact Coverage | Grounded Rate |
| --- | --- | --- | --- | --- | --- | --- |
| dense | 0 | n/a | n/a | n/a | n/a | n/a |
| sparse | 0 | n/a | n/a | n/a | n/a | n/a |
| hybrid | 0 | n/a | n/a | n/a | n/a | n/a |
| full_graph | 0 | n/a | n/a | n/a | n/a | n/a |

## Notes

- Predictions scored from eval_reports\retrieval_predictions.jsonl.
- Deterministic metrics are always reported; LLM-as-judge is opt-in.
- Retrieval component mode skips router, grader, generation, and E2E answer metrics.

## Reproduce

```bash
python scripts/evaluate_legal_qa.py --dataset data/evaluation/legal_qa_eval_100.jsonl
python scripts/run_retrieval_eval.py --dataset data/evaluation/legal_qa_eval_100.jsonl --out eval_reports/retrieval_predictions.jsonl
python scripts/evaluate_legal_qa.py --dataset data/evaluation/legal_qa_eval_100.jsonl --predictions eval_reports/retrieval_predictions.jsonl --component retrieval --out-json eval_reports/retrieval_100.json --out-md eval_reports/retrieval_100.md
python scripts/run_e2e_eval.py --dataset data/evaluation/legal_qa_eval_e2e_20.jsonl --out eval_reports/e2e_predictions_20.jsonl --skip-existing
python scripts/evaluate_legal_qa.py --dataset data/evaluation/legal_qa_eval_e2e_20.jsonl --predictions eval_reports/e2e_predictions_20.jsonl --component e2e --only-predicted --out-json eval_reports/e2e_20.json --out-md eval_reports/e2e_20.md
python scripts/compare_ablation_runs.py --dataset data/evaluation/legal_qa_eval_100.jsonl --run dense=eval_reports/dense_predictions.jsonl --run sparse=eval_reports/sparse_predictions.jsonl --run hybrid=eval_reports/hybrid_predictions.jsonl --run full_graph=eval_reports/e2e_predictions.jsonl
```
