"""Run end-to-end benchmark cases through the LangGraph app.

This script requires the normal runtime dependencies for the project. It writes
a predictions JSONL file that ``scripts/evaluate_legal_qa.py`` can score
offline afterwards.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "evaluation" / "legal_qa_eval_e2e_40.jsonl"
DEFAULT_OUT = ROOT / "eval_reports" / "e2e_predictions.jsonl"
QUOTA_ERROR_TERMS = (
    "RESOURCE_EXHAUSTED",
    "quota",
    "rate_limit",
    "rate limit",
    "429",
    "generate_content_free_tier_requests",
)


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


def compact_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_id": event.get("trace_id"),
        "step": event.get("step"),
        "agent": event.get("agent"),
        "input_summary": event.get("input_summary"),
        "output_summary": event.get("output_summary"),
        "chunk_ids": event.get("chunk_ids", []),
        "scores": event.get("scores", {}),
        "latency_ms": event.get("latency_ms", 0),
        "error": event.get("error"),
        "created_at": event.get("created_at"),
    }


def latency_for(events: list[dict[str, Any]], agents: set[str]) -> int | None:
    values = [
        int(event.get("latency_ms", 0))
        for event in events
        if event.get("agent") in agents and isinstance(event.get("latency_ms", 0), (int, float))
    ]
    return sum(values) if values else None


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


def is_quota_error(prediction: dict[str, Any]) -> bool:
    text = " ".join(
        str(value or "")
        for value in [
            prediction.get("error"),
            prediction.get("answer"),
            prediction.get("hallucinations"),
        ]
    )
    normalized = text.lower()
    return any(term.lower() in normalized for term in QUOTA_ERROR_TERMS)


async def run_case(case: dict[str, Any], *, max_content_chars: int) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from src.graph.graph import app as graph_app
    from src.graph.runtime_store import get_agent_events, get_citations, get_documents, get_web_results
    from src.graph.state import create_initial_state

    started = time.perf_counter()
    initial_state = create_initial_state(case["question"], user_id="evaluation")
    try:
        final_state = await graph_app.ainvoke(initial_state)
        processing_time_ms = int((time.perf_counter() - started) * 1000)
        documents = get_documents(final_state)
        web_results = get_web_results(final_state)
        citations = get_citations(final_state)
        events = get_agent_events(final_state)

        answer = final_state.get("answer") or "Xin lỗi, tôi không thể tìm thấy câu trả lời phù hợp."
        return {
            "id": case["id"],
            "trace_id": final_state.get("trace_id") or final_state.get("request_id"),
            "question": case["question"],
            "answer": answer,
            "citations": citations,
            "retrieved_documents": [compact_document(doc, max_content_chars) for doc in documents],
            "web_results": web_results,
            "intent": final_state.get("intent"),
            "intent_confidence": final_state.get("intent_confidence"),
            "route_action": final_state.get("route_action"),
            "route_confidence": final_state.get("route_confidence"),
            "query_filters": final_state.get("query_filters"),
            "query_preferences": final_state.get("query_preferences"),
            "retrieved_chunk_ids": final_state.get("retrieved_chunk_ids", []),
            "selected_context_ids": final_state.get("selected_context_ids", []),
            "grader_verdict": final_state.get("grader_verdict"),
            "grader_score": final_state.get("grader_score"),
            "hallucination_verdict": final_state.get("hallucination_verdict"),
            "hallucinations": final_state.get("hallucinations"),
            "hallucination_retry_count": final_state.get("hallucination_retry_count", 0),
            "generation_attempt": final_state.get("generation_attempt", 0),
            "retrieval_ms": latency_for(events, {"retriever"}),
            "answer_ms": latency_for(events, {"generator", "hallucination_grader"}),
            "processing_time_ms": processing_time_ms,
            "agent_events": [compact_event(event) for event in events],
            "error": final_state.get("error"),
        }
    except Exception as exc:
        processing_time_ms = int((time.perf_counter() - started) * 1000)
        return {
            "id": case.get("id", ""),
            "trace_id": initial_state.get("trace_id") or initial_state.get("request_id"),
            "question": case.get("question", ""),
            "answer": "",
            "citations": [],
            "retrieved_documents": [],
            "web_results": [],
            "intent": None,
            "grader_verdict": None,
            "hallucination_verdict": None,
            "hallucination_retry_count": 0,
            "generation_attempt": 0,
            "retrieval_ms": None,
            "answer_ms": None,
            "processing_time_ms": processing_time_ms,
            "agent_events": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


async def run_dataset(args: argparse.Namespace) -> None:
    cases = select_cases(load_jsonl(args.dataset), args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append or args.skip_existing else "w"
    completed = 0
    quota_errors = 0
    if not cases:
        print("No E2E cases to run.")
        return
    with args.out.open(mode, encoding="utf-8") as file:
        for case in cases:
            prediction = await run_case(case, max_content_chars=args.max_content_chars)
            file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            file.flush()
            completed += 1
            print(f"[{completed}/{len(cases)}] {case['id']} error={bool(prediction.get('error'))}")
            if is_quota_error(prediction):
                quota_errors += 1
                print(f"Quota/rate-limit detected ({quota_errors}/{args.max_quota_errors}).")
                if args.stop_on_quota and quota_errors >= args.max_quota_errors:
                    print("Stopping early because Gemini quota/rate limit was reached. Re-run later with --skip-existing.")
                    break
    print(f"Saved predictions: {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end legal QA evaluation.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--case-id", action="append", default=[], help="Run one case ID. Can be repeated.")
    parser.add_argument("--case-ids", default="", help="Comma-separated case IDs to run.")
    parser.add_argument("--stop-on-quota", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-quota-errors", type=int, default=1)
    parser.add_argument("--max-content-chars", type=int, default=4000)
    args = parser.parse_args()
    asyncio.run(run_dataset(args))


if __name__ == "__main__":
    main()
