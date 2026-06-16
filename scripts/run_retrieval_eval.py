"""Run retrieval-only benchmark cases without calling any LLM agents.

This runner is intended for API-key-limited evaluation. It exercises the same
retriever implementation used by the graph, stores retrieved chunks in the
standard prediction schema, and can be scored with:

    python scripts/evaluate_legal_qa.py --predictions eval_reports/retrieval_predictions.jsonl --component retrieval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "evaluation" / "legal_qa_eval_100.jsonl"
DEFAULT_OUT = ROOT / "eval_reports" / "retrieval_predictions.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def compact_document(doc: dict[str, Any], max_content_chars: int) -> dict[str, Any]:
    content = str(doc.get("content", ""))
    if max_content_chars and len(content) > max_content_chars:
        content = content[:max_content_chars]
    return {
        "content": content,
        "metadata": doc.get("metadata", {}),
        "score": doc.get("score", 0.0),
    }


def parse_case_ids(case_id_args: list[str], case_ids_arg: str) -> set[str]:
    ids: list[str] = []
    ids.extend(case_id_args or [])
    ids.extend(item.strip() for item in (case_ids_arg or "").split(",") if item.strip())
    return set(ids)


def completed_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["id"] for row in load_jsonl(path) if row.get("id")}


def select_cases(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected_ids = parse_case_ids(args.case_id, args.case_ids)
    if selected_ids:
        cases = [case for case in cases if case.get("id") in selected_ids]
    if args.skip_existing:
        done = completed_case_ids(args.out)
        cases = [case for case in cases if case.get("id") not in done]
    if args.limit:
        cases = cases[: args.limit]
    return cases


def run_case(case: dict[str, Any], *, max_content_chars: int) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from src.agents.retriever import retriever_node
    from src.graph.runtime_store import get_documents
    from src.graph.state import create_initial_state

    started = time.perf_counter()
    initial_state = create_initial_state(case["question"], user_id="retrieval-evaluation")
    try:
        update = retriever_node(initial_state)
        final_state = {**initial_state, **update}
        documents = get_documents(final_state)
        retrieval_ms = int((time.perf_counter() - started) * 1000)
        return {
            "id": case["id"],
            "question": case["question"],
            "component": "retrieval",
            "answer": "",
            "citations": [],
            "retrieved_documents": [compact_document(doc, max_content_chars) for doc in documents],
            "web_results": [],
            "intent": None,
            "grader_verdict": None,
            "hallucination_verdict": None,
            "generation_attempt": 0,
            "query_filters": final_state.get("query_filters"),
            "query_preferences": final_state.get("query_preferences"),
            "retrieved_chunk_ids": final_state.get("retrieved_chunk_ids", []),
            "selected_context_ids": final_state.get("selected_context_ids", []),
            "retrieval_ms": retrieval_ms,
            "answer_ms": None,
            "processing_time_ms": retrieval_ms,
            "agent_events": [],
            "error": update.get("error"),
        }
    except Exception as exc:
        retrieval_ms = int((time.perf_counter() - started) * 1000)
        return {
            "id": case.get("id", ""),
            "question": case.get("question", ""),
            "component": "retrieval",
            "answer": "",
            "citations": [],
            "retrieved_documents": [],
            "web_results": [],
            "intent": None,
            "grader_verdict": None,
            "hallucination_verdict": None,
            "generation_attempt": 0,
            "retrieval_ms": retrieval_ms,
            "answer_ms": None,
            "processing_time_ms": retrieval_ms,
            "agent_events": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_dataset(args: argparse.Namespace) -> None:
    cases = select_cases(load_jsonl(args.dataset), args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append or args.skip_existing else "w"
    if not cases:
        print("No retrieval cases to run.")
        return
    with args.out.open(mode, encoding="utf-8") as file:
        for index, case in enumerate(cases, 1):
            prediction = run_case(case, max_content_chars=args.max_content_chars)
            file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            file.flush()
            print(f"[{index}/{len(cases)}] {case['id']} error={bool(prediction.get('error'))}")
    print(f"Saved retrieval predictions: {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval-only legal QA evaluation without LLM calls.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--case-id", action="append", default=[], help="Run one case ID. Can be repeated.")
    parser.add_argument("--case-ids", default="", help="Comma-separated case IDs to run.")
    parser.add_argument("--max-content-chars", type=int, default=4000)
    args = parser.parse_args()
    run_dataset(args)


if __name__ == "__main__":
    main()
