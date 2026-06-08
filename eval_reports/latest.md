# Legal QA Evaluation Report

- Created at: 2026-06-08T17:26:59
- Dataset seed: `data/evaluation/legal_qa_eval_30.jsonl`
- Mode: baseline_snapshot

## Corpus

- Registry documents: 4
- Chunks: 527
- Chunk length average: 843.4 chars

## Retrieval Summary

| Metric | Value |
| --- | ---: |
| Cases | 32 |
| Doc Hit@5 | 100.00% |
| Article Hit@5 | 93.75% |
| Clause Hit@5 | 84.38% |
| Point Hit@5 | 100.00% |
| Level Hit@5 | 100.00% |
| MRR | 90.63% |
| Avg retrieval latency | n/a ms |

## Answer Summary

| Metric | Value |
| --- | ---: |
| Cases | 5 |
| Fact Coverage | 70.00% |
| Forbidden OK | n/a |
| Display Citation Valid | 100.00% |
| Citation URL Valid | 100.00% |
| Avg answer latency | n/a ms |

## Notes

- Retrieval was evaluated with top_k=5 on an internal 32-question set.
- Generation was evaluated on the first 5 cases to limit LLM cost.
- Full graph evaluation, LLM-as-a-judge, and RAGAS were not run in this snapshot.

## Reproduce

```bash
python scripts/evaluate_legal_qa.py
python scripts/evaluate_legal_qa.py --predictions eval_reports/my_run_predictions.jsonl
```
