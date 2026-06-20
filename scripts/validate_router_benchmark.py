"""Validate the standalone router benchmark without calling any model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "evaluation" / "router_eval_72.jsonl"

INTENTS = {"legal_query", "procedural", "out_of_scope", "general_chat"}
ROUTE_ACTIONS = {
    "retrieve",
    "redirect_out_of_scope",
    "respond_chat",
    "refuse_unsafe",
    "refuse_unsupported",
    "web_required",
}
DIFFICULTIES = {"easy", "medium", "hard"}
REQUIRED_FIELDS = {
    "id",
    "question",
    "domain",
    "category",
    "type",
    "difficulty",
    "answer_policy",
    "requires_web",
    "is_ambiguous",
    "expected",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, raw_line in enumerate(file, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def validate_policy(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = row.get("expected") or {}
    intent = expected.get("expected_intent")
    action = expected.get("expected_route_action")
    policy = row.get("answer_policy")
    requires_web = row.get("requires_web")

    if requires_web is True and action != "web_required":
        errors.append("requires_web=true must map to web_required")
    if policy == "unsafe_refusal" and action != "refuse_unsafe":
        errors.append("unsafe_refusal must map to refuse_unsafe")
    if policy == "unsupported_refusal" and action != "refuse_unsupported":
        errors.append("unsupported_refusal must map to refuse_unsupported")
    if intent == "out_of_scope" and action != "redirect_out_of_scope":
        errors.append("out_of_scope must map to redirect_out_of_scope")
    if intent == "general_chat" and action != "respond_chat":
        errors.append("general_chat must map to respond_chat")
    if (
        intent in {"legal_query", "procedural"}
        and policy == "grounded_answer"
        and requires_web is False
        and action != "retrieve"
    ):
        errors.append("grounded legal/procedural case must map to retrieve")
    return errors


def validate(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, Counter[str]]]:
    errors: list[str] = []
    ids: set[str] = set()
    questions: set[str] = set()

    for row in rows:
        line_no = row.get("_line_no", "?")
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            errors.append(f"line {line_no}: missing fields {sorted(missing)}")

        case_id = str(row.get("id", "")).strip()
        question = " ".join(str(row.get("question", "")).casefold().split())
        if not case_id:
            errors.append(f"line {line_no}: empty id")
        elif case_id in ids:
            errors.append(f"line {line_no}: duplicate id {case_id}")
        ids.add(case_id)

        if not question:
            errors.append(f"line {line_no}: empty question")
        elif question in questions:
            errors.append(f"line {line_no}: duplicate normalized question")
        questions.add(question)

        expected = row.get("expected") or {}
        intent = expected.get("expected_intent")
        action = expected.get("expected_route_action")
        if intent not in INTENTS:
            errors.append(f"line {line_no}: invalid intent {intent!r}")
        if action not in ROUTE_ACTIONS:
            errors.append(f"line {line_no}: invalid route action {action!r}")
        if row.get("difficulty") not in DIFFICULTIES:
            errors.append(f"line {line_no}: invalid difficulty {row.get('difficulty')!r}")
        if not isinstance(row.get("requires_web"), bool):
            errors.append(f"line {line_no}: requires_web must be boolean")
        if not isinstance(row.get("is_ambiguous"), bool):
            errors.append(f"line {line_no}: is_ambiguous must be boolean")
        for policy_error in validate_policy(row):
            errors.append(f"line {line_no}: {policy_error}")

    counters = {
        "intent": Counter(row["expected"]["expected_intent"] for row in rows),
        "route_action": Counter(row["expected"]["expected_route_action"] for row in rows),
        "category": Counter(str(row.get("category")) for row in rows),
        "domain": Counter(str(row.get("domain")) for row in rows),
        "difficulty": Counter(str(row.get("difficulty")) for row in rows),
        "requires_web": Counter(str(row.get("requires_web")).lower() for row in rows),
        "ambiguous": Counter(str(row.get("is_ambiguous")).lower() for row in rows),
        "benchmark_version": Counter(
            str(row.get("benchmark_version", "legacy_unspecified")) for row in rows
        ),
    }
    return errors, counters


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate router benchmark labels and coverage.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--min-class-size", type=int, default=5)
    args = parser.parse_args()

    rows = load_jsonl(args.dataset)
    errors, counters = validate(rows)
    print(f"Dataset: {args.dataset}")
    print(f"Cases: {len(rows)}")
    for name, counter in counters.items():
        print(f"{name}: {dict(sorted(counter.items()))}")

    for label_type in ("intent", "route_action"):
        for label, count in counters[label_type].items():
            if count < args.min_class_size:
                print(f"SAMPLE_RISK {label_type}={label}: {count} < {args.min_class_size}")

    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Validation: PASS")


if __name__ == "__main__":
    main()
