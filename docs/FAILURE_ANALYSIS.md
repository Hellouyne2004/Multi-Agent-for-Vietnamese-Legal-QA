# Failure Analysis Guide

Generated failure reports are written to `eval_reports/failure_analysis.md`.
This committed guide explains how to interpret them to aid in debugging and future
system development.

## Failure Categories

| Category | Meaning | First place to inspect |
| --- | --- | --- |
| `wrong_intent` | Router chose the wrong broad intent | Router prompt and holdout labels |
| `wrong_route_action` | Router chose the wrong policy action | Route policy rules and safety examples |
| `wrong_doc` | Retriever missed the expected document | Query filters, hybrid search, corpus metadata |
| `missing_context_fact` | Retrieved context lacks expected facts | Chunking, OCR quality, context expansion |
| `forbidden_context_fact` | Retrieved context contains misleading facts | Ranking and metadata filters |
| `wrong_grader_verdict` | Grader accepted or rejected context incorrectly | Grader prompt and sufficiency labels |
| `missing_fact` | Answer omitted expected facts | Generator prompt after retrieval is confirmed |
| `invalid_citation` | Answer lacks source IDs when required | Generator citation format and validator |
| `unsupported_claim` | Answer contains unsupported numeric claims | Hallucination grader and citation grounding |
| `refusal_error` | Refusal/out-of-scope behavior is wrong | Router policy and response template |
| `quota_or_rate_limit` | Provider quota interrupted the run | Re-run later with `--skip-existing` |
| `runtime_error` | Service or dependency failed | Qdrant, graph exceptions, API logs |

## System Diagnostics

This report facilitates component-by-component diagnostics of the system. By isolating failure modes, retrieval errors can be systematically debugged prior to assessing generation quality, and infrastructural issues (e.g., provider quotas or rate limits) are explicitly separated from model quality failures.
