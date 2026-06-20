"""Run only router_node for a labelled router benchmark.

This runner never executes graph edges, retrieval, generation, graders, or web
search. It stops after a configurable number of errors to protect API quota.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "evaluation" / "router_eval_72.jsonl"
DEFAULT_OUT = ROOT / "eval_reports" / "router_predictions_v2_2.jsonl"
BENCHMARK_VERSION = "router-eval-72-v1.1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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


def parse_case_ids(values: list[str], csv_value: str) -> set[str]:
    case_ids = {value.strip() for value in values if value.strip()}
    case_ids.update(value.strip() for value in csv_value.split(",") if value.strip())
    return case_ids


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def select_cases(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    requested_ids = parse_case_ids(args.case_id, args.case_ids)
    if requested_ids:
        known_ids = {case["id"] for case in cases}
        missing_ids = requested_ids - known_ids
        if missing_ids:
            raise ValueError(f"Unknown case IDs: {sorted(missing_ids)}")
        cases = [case for case in cases if case["id"] in requested_ids]
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
    from src.agents.router import ROUTER_PROMPT_VERSION, router_node
    from src.config import LLM_MODEL, LLM_TEMPERATURE
    from src.graph.state import create_initial_state

    state = create_initial_state(case["question"], user_id="router-evaluation")
    started = time.perf_counter()
    try:
        result = router_node(state)
        router_ms = int((time.perf_counter() - started) * 1000)
        return {
            "id": case["id"],
            "intent": result.get("intent"),
            "intent_confidence": result.get("intent_confidence"),
            "route_action": result.get("route_action"),
            "route_confidence": result.get("route_confidence"),
            "router_attempt_count": result.get("router_attempt_count"),
            "router_key_index": result.get("router_key_index"),
            "benchmark_version": case.get("benchmark_version", BENCHMARK_VERSION),
            "prompt_version": ROUTER_PROMPT_VERSION,
            "model": LLM_MODEL,
            "temperature": LLM_TEMPERATURE,
            "router_ms": router_ms,
            "error": result.get("error"),
        }
    except Exception as exc:
        router_ms = int((time.perf_counter() - started) * 1000)
        return {
            "id": case["id"],
            "intent": None,
            "intent_confidence": None,
            "route_action": "router_error",
            "route_confidence": 0.0,
            "router_attempt_count": None,
            "router_key_index": None,
            "benchmark_version": case.get("benchmark_version", BENCHMARK_VERSION),
            "prompt_version": ROUTER_PROMPT_VERSION,
            "model": LLM_MODEL,
            "temperature": LLM_TEMPERATURE,
            "router_ms": router_ms,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run(args: argparse.Namespace) -> None:
    cases = select_cases(load_jsonl(args.dataset), args)
    if not cases:
        print("No router cases to run.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = load_jsonl(args.out) if args.append or args.skip_existing else []
    predictions_by_id = {
        row["id"]: row for row in existing_rows if row.get("id")
    }
    errors = 0
    for index, case in enumerate(cases, 1):
        prediction = run_case(case)
        predictions_by_id[case["id"]] = prediction
        write_jsonl_atomic(args.out, list(predictions_by_id.values()))
        has_error = bool(prediction.get("error"))
        errors += int(has_error)
        print(
            f"[{index}/{len(cases)}] {case['id']} "
            f"intent={prediction.get('intent')} "
            f"action={prediction.get('route_action')} error={has_error}"
        )
        if args.max_errors is not None and errors >= args.max_errors:
            print(f"Stopped after {errors} error(s) to protect quota.")
            break
    print(f"Saved router predictions: {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run router-only evaluation cases.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--max-errors",
        type=int,
        default=1,
        help="Stop after this many error rows. Use a negative value to disable.",
    )
    args = parser.parse_args()
    if args.max_errors is not None and args.max_errors < 0:
        args.max_errors = None
    run(args)


if __name__ == "__main__":
    main()
