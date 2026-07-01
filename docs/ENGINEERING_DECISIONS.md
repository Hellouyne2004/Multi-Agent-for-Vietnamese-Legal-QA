# Engineering Decisions

This project is designed as a robust, production-minded multi-agent RAG system tailored for Vietnamese Legal Question Answering. The architecture prioritizes reproducibility, high reliability, deterministic evaluation, and comprehensive traceability.

## Current Priorities

1. **Evaluation before more agents.** New agents or retrieval changes should be
   accepted only when they improve measurable quality, latency, safety, or
   debuggability.
2. **Compact graph state.** Large retrieved documents, web snippets, citations,
   and trace events stay in `runtime_store`; LangGraph state passes compact IDs.
3. **Deterministic scoring first.** LLM-as-judge is optional. Public metrics must
   be reproducible from committed datasets and local prediction files.
4. **Traceable failures.** Each request should expose enough per-agent trace data
   to explain router, retriever, grader, generator, and hallucination decisions.

## Accepted Tradeoffs

- The E2E-40 benchmark includes two `web_required` cases, but live web answers
  are not published as stable results until predictions are fully collected.
- Router and Grader quality is reported separately from full E2E quality because
  Gemini quota and web dependencies can make graph-level runs incomplete.
- The current corpus is intentionally small; improving evaluation and debugging
  comes before adding more legal documents.

## Next Engineering Targets

- Improve Context Fact Coverage@5 before claiming answer-ready retrieval.
- Add ablation runs for dense-only, sparse-only, hybrid, and hybrid plus rerank.
- Publish failure analysis from scored E2E predictions once the E2E-40 run is
  complete.
