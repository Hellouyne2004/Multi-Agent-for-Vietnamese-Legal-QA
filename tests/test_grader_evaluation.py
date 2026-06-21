from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_grader_eval import score
from scripts.validate_grader_benchmark import load_jsonl, validate


def test_frozen_grader_benchmark_is_balanced_and_valid() -> None:
    dataset = load_jsonl(ROOT / "data" / "evaluation" / "grader_eval_20.jsonl")
    assert len(dataset) == 20
    assert validate(dataset) == []
    assert sum(case["context_sufficient"] == "yes" for case in dataset) == 10
    assert sum(case["context_sufficient"] == "no" for case in dataset) == 10


def test_grader_scorer_tracks_dangerous_yes_and_extra_fallback() -> None:
    dataset = [
        {"id": "a", "context_sufficient": "yes", "slice": "sufficient", "category": "x", "missing_facts": []},
        {"id": "b", "context_sufficient": "yes", "slice": "sufficient", "category": "x", "missing_facts": []},
        {"id": "c", "context_sufficient": "no", "slice": "incomplete", "category": "x", "missing_facts": ["fact-c"]},
        {"id": "d", "context_sufficient": "no", "slice": "incomplete", "category": "x", "missing_facts": ["fact-d"]},
    ]
    base = {
        "benchmark_version": "v1", "context_version": "ctx1",
        "prompt_version": "p1", "model": "m1", "temperature": 0.1,
        "request_timeout": 30.0,
        "grader_ms": 100, "grader_attempt_count": 1, "error": None,
    }
    predictions = [
        {**base, "id": "a", "grader_verdict": "yes", "grader_score": 0.9},
        {**base, "id": "b", "grader_verdict": "no", "grader_score": 0.2},
        {**base, "id": "c", "grader_verdict": "yes", "grader_score": 0.8},
        {**base, "id": "d", "grader_verdict": "no", "grader_score": 0.1},
    ]
    report = score(dataset, predictions, assumed_web_ms=2500)
    assert report["status"] == "MEASURED"
    assert report["classification"]["accuracy"]["numerator"] == 2
    assert report["false_positive_rate"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert report["false_negative_rate"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert report["false_negative_cost"]["extra_web_calls"] == 1
    assert report["false_negative_cost"]["estimated_extra_latency_ms"] == 2500
    failure_types = {failure["failure_type"] for failure in report["failures"]}
    assert failure_types == {"false_positive_unsafe_yes", "false_negative_extra_fallback"}
    assert report["recommended_threshold"] is None
    assert report["threshold_assessment"]["status"] == "N/A_NOT_APPLICABLE"
    assert report["runtime"]["single_attempt"]["cases"] == 4


def test_grader_scorer_marks_empty_run_without_false_failures() -> None:
    dataset = [
        {"id": "a", "context_sufficient": "yes", "slice": "sufficient", "category": "x"},
        {"id": "b", "context_sufficient": "no", "slice": "incomplete", "category": "x"},
    ]
    report = score(dataset, [], incomplete_reason="BLOCKED_RUNTIME")
    assert report["status"] == "BLOCKED_RUNTIME"
    assert report["classification"]["accuracy"]["value"] is None
    assert report["threshold_sensitivity"] == []
    assert all(gate["status"] == "N/A_NOT_RUN" for gate in report["gates"])


def test_grader_scorer_marks_partial_coverage() -> None:
    dataset = [
        {"id": "a", "context_sufficient": "yes", "slice": "sufficient", "category": "x"},
        {"id": "b", "context_sufficient": "no", "slice": "incomplete", "category": "x"},
    ]
    prediction = {
        "id": "a", "grader_verdict": "yes", "grader_score": 0.9,
        "benchmark_version": "v1", "context_version": "ctx1",
        "prompt_version": "p1", "model": "m1", "temperature": 0.1,
        "grader_ms": 100, "grader_attempt_count": 1, "error": None,
    }
    report = score(dataset, [prediction])
    assert report["status"] == "INVALID_PARTIAL"
    assert report["coverage"]["missing_ids"] == ["b"]


def test_grader_empty_context_is_deterministic_no() -> None:
    try:
        grader_module = importlib.import_module("src.agents.grader")
    except ModuleNotFoundError as exc:
        pytest.skip(f"Optional Grader dependencies unavailable: {exc}")
    result = grader_module.grader_node({"question": "Test", "documents": []})
    assert result["grader_verdict"] == "no"
    assert result["grader_score"] == 0.0
    assert result["grader_attempt_count"] == 0
    assert result["error"] is None
