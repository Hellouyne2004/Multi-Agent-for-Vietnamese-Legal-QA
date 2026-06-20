# Router Benchmark

## Purpose

`data/evaluation/router_eval_72.jsonl` (`router-eval-72-v1.1`) evaluates the
router as two separate systems:

1. A four-class intent classifier.
2. A policy gate represented by `expected.expected_route_action`.

The benchmark is aligned with the current corpus domains: labor law, personal income tax, cybersecurity law, and the land-price appendix. It also includes out-of-scope, conversational, unsafe, unsupported, ambiguous, and time-sensitive questions.

`data/evaluation/router_holdout_30_v1.jsonl` is the compact blind holdout for
overall evaluation. It was authored after prompt v2.2 was frozen and contains
five cases for each route action. Do not use its failures to edit the prompt
until the complete 30-case report has been saved.

## Labels

Intent labels match `src/agents/router.py` exactly:

- `legal_query`
- `procedural`
- `out_of_scope`
- `general_chat`

Policy labels produced by the router are:

- `retrieve`
- `redirect_out_of_scope`
- `respond_chat`
- `refuse_unsafe`
- `refuse_unsupported`
- `web_required`

Runtime failures use `router_error`. It is not a gold benchmark class and must
be counted in error rate rather than mapped to an intent or policy label.

Do not silently map policy labels back into intent labels. A case can correctly have `expected_intent=legal_query` and `expected_route_action=refuse_unsafe`.

## Coverage

| Dimension | Distribution |
| --- | --- |
| Intent | legal_query 30; procedural 14; out_of_scope 14; general_chat 14 |
| Route action | retrieve 27; redirect 14; chat 14; unsafe 6; unsupported 5; web 6 |
| Web requirement | false 66; true 6 |
| Corpus domains | labor, tax, cybersecurity, land_price |

Every intent and policy class has at least five cases. This is enough for a first macro-F1 report, but not for a narrow confidence interval.

Version `v1.1` adjudicates `router_unsupported_062` from
`refuse_unsupported` to `retrieve`: a potentially false claim within corpus
scope must be retrieved and corrected rather than fact-checked by the router.

## Validate Without API Calls

```powershell
python scripts/validate_router_benchmark.py
```

The validator checks JSONL syntax, unique IDs/questions, allowed labels, boolean fields, intent coverage, route-action coverage, and policy consistency.

## Quota-Aware Pilot

The following 20 cases cover all four intent classes and all six route actions:

```powershell
python scripts/run_router_eval.py --out eval_reports/router_predictions_v2_2.jsonl --case-ids "router_labor_legal_001,router_tax_legal_004,router_cyber_legal_007,router_land_legal_010,router_labor_proc_013,router_tax_proc_017,router_cyber_proc_021,router_land_proc_024,router_oos_027,router_oos_030,router_oos_033,router_chat_041,router_chat_046,router_chat_051,router_unsafe_055,router_unsafe_059,router_unsupported_061,router_unsupported_065,router_web_067,router_web_071" --max-errors 1
```

This pilot is a runtime/schema check, not a final quality measurement. Minority policy classes have only two pilot examples.

## Resume Full Collection

```powershell
python scripts/run_router_eval.py --out eval_reports/router_predictions_v2_2.jsonl --append --skip-existing --limit 10 --max-errors 1
```

Repeat on later quota windows until all 72 IDs have successful predictions. Do not convert failed calls into `general_chat`; rows with `error` remain runtime failures.

With `--skip-existing`, successful IDs are skipped while error rows are retried
and atomically replaced by ID. This keeps the JSONL unique across quota windows
and prevents transient quota errors from becoming permanently skipped cases.

Minimum prediction schema:

```json
{"id":"case_id","intent":"legal_query","intent_confidence":0.92,"route_action":"retrieve","route_confidence":0.95,"router_attempt_count":1,"router_key_index":2,"benchmark_version":"router-eval-72-v1.1","prompt_version":"router-policy-v2.2","model":"gemini-2.5-flash","temperature":0.1,"router_ms":35,"error":null}
```

Use a new output file after changing the router contract. Do not append v2
predictions to the legacy file that has no policy fields:

```powershell
python scripts/run_router_eval.py --out eval_reports/router_predictions_v2_2.jsonl --case-ids "router_unsafe_055,router_unsafe_056,router_unsafe_057,router_unsafe_058,router_unsafe_059,router_unsafe_060" --max-errors 1
```

Score any partial or complete run without API calls:

```powershell
python scripts/score_router_eval.py --dataset data/evaluation/router_holdout_30_v1.jsonl --predictions eval_reports/router_holdout_30_predictions.jsonl --out-json eval_reports/router_holdout_30.json --out-md eval_reports/router_holdout_30.md
```

The scorer reports `INVALID_PARTIAL` until all 72 case IDs have predictions.
It reports policy metrics as `N/A_NOT_RUN` when predictions do not contain
`route_action`; missing policy output is never converted to zero or PASS.

## Scoring Rules

- Report coverage and runtime errors before classifier metrics.
- Count errors separately; do not treat fallback `general_chat` as a valid model prediction.
- Report accuracy, macro/micro precision, recall and F1, per-class metrics, and the confusion matrix.
- Report false accepts and false rejects against `expected_route_action` separately from intent errors.
- Confidence calibration requires confidence buckets with enough samples and ECE over successful predictions.
- Slice by domain, category, difficulty, ambiguity, `requires_web`, and answer policy.
- Mark a class `SAMPLE_RISK` whenever its valid prediction denominator falls below five.
