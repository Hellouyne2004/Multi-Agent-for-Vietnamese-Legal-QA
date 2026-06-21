"""Run only grader_node on a frozen, labelled context benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "evaluation" / "grader_eval_20.jsonl"
DEFAULT_OUT = ROOT / "eval_reports" / "grader_predictions.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, raw_line in enumerate(file, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def parse_case_ids(values: list[str], csv_value: str) -> set[str]:
    case_ids = {value.strip() for value in values if value.strip()}
    case_ids.update(value.strip() for value in csv_value.split(",") if value.strip())
    return case_ids


def select_cases(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    requested = parse_case_ids(args.case_id, args.case_ids)
    if requested:
        known = {case["id"] for case in cases}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown case IDs: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in requested]
    if args.skip_existing:
        completed = {
            row.get("id")
            for row in load_jsonl(args.out)
            if row.get("id") and not row.get("error")
        }
        cases = [case for case in cases if case["id"] not in completed]
    if args.limit is not None:
        cases = cases[: args.limit]
    return cases


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from src.agents.grader import GRADER_PROMPT_VERSION, grader_node
    from src.config import LLM_MODEL, LLM_REQUEST_TIMEOUT, LLM_TEMPERATURE
    from src.graph.state import create_initial_state

    state = create_initial_state(case["question"], user_id="grader-evaluation")
    state["documents"] = case.get("grader_documents", [])
    started = time.perf_counter()
    try:
        result = grader_node(state)
        grader_ms = int((time.perf_counter() - started) * 1000)
        return {
            "id": case["id"],
            "grader_verdict": result.get("grader_verdict"),
            "grader_score": result.get("grader_score"),
            "grader_reasoning": result.get("grader_reasoning", ""),
            "grader_attempt_count": result.get("grader_attempt_count"),
            "grader_key_index": result.get("grader_key_index"),
            "grader_ms": grader_ms,
            "benchmark_version": case.get("benchmark_version"),
            "context_version": case.get("context_version"),
            "context_hash": case.get("context_hash"),
            "prompt_version": GRADER_PROMPT_VERSION,
            "model": LLM_MODEL,
            "temperature": LLM_TEMPERATURE,
            "request_timeout": LLM_REQUEST_TIMEOUT,
            "error": result.get("error"),
        }
    except Exception as exc:
        grader_ms = int((time.perf_counter() - started) * 1000)
        return {
            "id": case["id"],
            "grader_verdict": None,
            "grader_score": None,
            "grader_reasoning": "",
            "grader_attempt_count": None,
            "grader_key_index": None,
            "grader_ms": grader_ms,
            "benchmark_version": case.get("benchmark_version"),
            "context_version": case.get("context_version"),
            "context_hash": case.get("context_hash"),
            "prompt_version": GRADER_PROMPT_VERSION,
            "model": LLM_MODEL,
            "temperature": LLM_TEMPERATURE,
            "request_timeout": LLM_REQUEST_TIMEOUT,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run(args: argparse.Namespace) -> None:
    cases = select_cases(load_jsonl(args.dataset), args)
    if not cases:
        print("No Grader cases to run.")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = load_jsonl(args.out) if args.append or args.skip_existing else []
    predictions = {row["id"]: row for row in existing if row.get("id")}
    errors = 0
    for index, case in enumerate(cases, 1):
        prediction = run_case(case)
        predictions[case["id"]] = prediction
        write_jsonl_atomic(args.out, list(predictions.values()))
        has_error = bool(prediction.get("error"))
        errors += int(has_error)
        print(
            f"[{index}/{len(cases)}] {case['id']} "
            f"verdict={prediction.get('grader_verdict')} "
            f"score={prediction.get('grader_score')} error={has_error}"
        )
        if args.max_errors is not None and errors >= args.max_errors:
            print(f"Stopped after {errors} error(s) to protect quota.")
            break
    print(f"Saved Grader predictions: {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-errors", type=int, default=1)
    args = parser.parse_args()
    if args.max_errors is not None and args.max_errors < 0:
        args.max_errors = None
    run(args)


if __name__ == "__main__":
    main()
