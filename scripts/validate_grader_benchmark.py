"""Validate frozen Grader benchmark structure without calling an LLM."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "evaluation" / "grader_eval_20.jsonl"
VALID_LABELS = {"yes", "no", "uncertain"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def calculate_hash(documents: list[dict[str, Any]]) -> str:
    payload = json.dumps(documents, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    ids = [row.get("id") for row in rows]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"Duplicate IDs: {duplicates}")
    for index, row in enumerate(rows, 1):
        prefix = f"row {index} ({row.get('id')})"
        label = row.get("context_sufficient")
        if label not in VALID_LABELS:
            errors.append(f"{prefix}: invalid context_sufficient={label!r}")
        for key in ["benchmark_version", "context_version", "question", "gold_reason", "slice"]:
            if not row.get(key):
                errors.append(f"{prefix}: missing {key}")
        if not isinstance(row.get("expected_facts"), list):
            errors.append(f"{prefix}: expected_facts must be a list")
        if not isinstance(row.get("missing_facts"), list):
            errors.append(f"{prefix}: missing_facts must be a list")
        documents = row.get("grader_documents")
        if not isinstance(documents, list):
            errors.append(f"{prefix}: grader_documents must be a list")
            continue
        if row.get("context_hash") != calculate_hash(documents):
            errors.append(f"{prefix}: context_hash mismatch")
        if label == "no" and row.get("slice") == "empty_context" and documents:
            errors.append(f"{prefix}: empty_context contains documents")
    labels = Counter(row.get("context_sufficient") for row in rows)
    if labels["yes"] != 10 or labels["no"] != 10:
        errors.append(f"Expected balanced yes/no 10/10, got {dict(labels)}")
    origins = Counter(row.get("context_origin") for row in rows)
    if origins["retrieval_snapshot"] < 15:
        errors.append("At least 15 cases must use actual retrieval snapshots")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    rows = load_jsonl(args.dataset)
    errors = validate(rows)
    print(f"Dataset: {args.dataset}")
    print(f"Cases: {len(rows)}")
    print(f"labels: {dict(Counter(row.get('context_sufficient') for row in rows))}")
    print(f"slices: {dict(Counter(row.get('slice') for row in rows))}")
    print(f"origins: {dict(Counter(row.get('context_origin') for row in rows))}")
    print(f"agreement: {dict(Counter(row.get('annotation', {}).get('agreement_status') for row in rows))}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Validation: PASS")


if __name__ == "__main__":
    main()
