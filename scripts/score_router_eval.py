"""Score router intent and policy predictions without calling any model."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "evaluation" / "router_eval_72.jsonl"
DEFAULT_PREDICTIONS = ROOT / "eval_reports" / "router_predictions_v2_2.jsonl"
DEFAULT_JSON = ROOT / "eval_reports" / "router_v2_2.json"
DEFAULT_MD = ROOT / "eval_reports" / "router_v2_2.md"

INTENTS = ["legal_query", "procedural", "out_of_scope", "general_chat"]
ROUTE_ACTIONS = [
    "retrieve",
    "redirect_out_of_scope",
    "respond_chat",
    "refuse_unsafe",
    "refuse_unsupported",
    "web_required",
]
ACTIVE_ACTIONS = {"retrieve", "web_required"}
REFUSAL_ACTIONS = {"refuse_unsafe", "refuse_unsupported"}
STOP_ACTIONS = REFUSAL_ACTIONS | {"redirect_out_of_scope", "respond_chat"}


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


def ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def nearest_rank(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def classification_summary(
    joined: list[dict[str, Any]],
    labels: list[str],
    gold_getter: Callable[[dict[str, Any]], Any],
    prediction_key: str,
    confidence_key: str,
    report_status: str,
) -> dict[str, Any]:
    eligible = [
        row
        for row in joined
        if not row["prediction"].get("error")
        and gold_getter(row["case"]) in labels
        and row["prediction"].get(prediction_key) in labels
    ]
    if not eligible:
        return {
            "status": "N/A_NOT_RUN",
            "eligible_cases": 0,
            "accuracy": ratio(0, 0),
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "micro_precision": None,
            "micro_recall": None,
            "micro_f1": None,
            "per_class": {},
            "confusion_matrix": {},
            "calibration": {"status": "N/A_NOT_RUN", "ece": None, "buckets": []},
        }

    confusion = {gold: {pred: 0 for pred in labels} for gold in labels}
    correct = 0
    for row in eligible:
        gold = gold_getter(row["case"])
        predicted = row["prediction"][prediction_key]
        confusion[gold][predicted] += 1
        correct += int(gold == predicted)

    per_class: dict[str, Any] = {}
    macro_precision_values: list[float] = []
    macro_recall_values: list[float] = []
    macro_f1_values: list[float] = []
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted_count = sum(confusion[gold][label] for gold in labels)
        fp = predicted_count - tp
        fn = support - tp
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        if support and precision is None:
            precision = 0.0
        f1 = f1_score(precision, recall)
        if support:
            if precision is not None:
                macro_precision_values.append(precision)
            if recall is not None:
                macro_recall_values.append(recall)
            if f1 is not None:
                macro_f1_values.append(f1)
        per_class[label] = {
            "support": support,
            "predicted": predicted_count,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "sample_risk": support < 5,
        }

    calibration_rows = []
    for row in eligible:
        raw_confidence = row["prediction"].get(confidence_key)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            continue
        if 0.0 <= confidence <= 1.0:
            calibration_rows.append(
                (
                    confidence,
                    int(gold_getter(row["case"]) == row["prediction"][prediction_key]),
                )
            )

    buckets = []
    weighted_gap = 0.0
    for bucket_index in range(10):
        lower = bucket_index / 10
        upper = (bucket_index + 1) / 10
        bucket = [
            item
            for item in calibration_rows
            if lower <= item[0] <= upper
            and (bucket_index == 9 or item[0] < upper)
        ]
        if not bucket:
            continue
        avg_confidence = mean(item[0] for item in bucket)
        accuracy = mean(item[1] for item in bucket)
        weighted_gap += len(bucket) / len(calibration_rows) * abs(avg_confidence - accuracy)
        buckets.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(bucket),
                "avg_confidence": avg_confidence,
                "accuracy": accuracy,
            }
        )

    micro = correct / len(eligible)
    return {
        "status": report_status,
        "eligible_cases": len(eligible),
        "accuracy": ratio(correct, len(eligible)),
        "macro_precision": mean(macro_precision_values) if macro_precision_values else None,
        "macro_recall": mean(macro_recall_values) if macro_recall_values else None,
        "macro_f1": mean(macro_f1_values) if macro_f1_values else None,
        "micro_precision": micro,
        "micro_recall": micro,
        "micro_f1": micro,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "calibration": {
            "status": "MEASURED" if len(calibration_rows) >= 30 else "INVALID_PARTIAL",
            "eligible_cases": len(calibration_rows),
            "ece": weighted_gap if calibration_rows else None,
            "buckets": buckets,
        },
    }


def build_slices(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dimensions = {
        "domain": lambda case: case.get("domain"),
        "category": lambda case: case.get("category"),
        "difficulty": lambda case: case.get("difficulty"),
        "ambiguous": lambda case: str(case.get("is_ambiguous")).lower(),
        "requires_web": lambda case: str(case.get("requires_web")).lower(),
    }
    output = []
    for dimension, getter in dimensions.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in joined:
            groups[str(getter(row["case"]))].append(row)
        for value, rows in groups.items():
            intent_eligible = [
                row
                for row in rows
                if not row["prediction"].get("error")
                and row["prediction"].get("intent") in INTENTS
            ]
            action_eligible = [
                row
                for row in rows
                if not row["prediction"].get("error")
                and row["prediction"].get("route_action") in ROUTE_ACTIONS
            ]
            intent_correct = sum(
                row["prediction"]["intent"]
                == row["case"]["expected"]["expected_intent"]
                for row in intent_eligible
            )
            action_correct = sum(
                row["prediction"]["route_action"]
                == row["case"]["expected"]["expected_route_action"]
                for row in action_eligible
            )
            output.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "cases": len(rows),
                    "intent_accuracy": ratio(intent_correct, len(intent_eligible)),
                    "route_action_accuracy": ratio(action_correct, len(action_eligible)),
                    "sample_risk": len(rows) < 5,
                }
            )
    return sorted(output, key=lambda row: (row["dimension"], row["value"]))


def gate(name: str, value: float | None, operator: str, threshold: float, status: str) -> dict[str, Any]:
    if value is None:
        result = "N/A_NOT_RUN"
    elif status != "MEASURED":
        result = status
    elif operator == "gte":
        result = "PASS" if value >= threshold else "FAIL"
    else:
        result = "PASS" if value <= threshold else "FAIL"
    return {
        "gate": name,
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "status": result,
    }


def build_report(
    dataset_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    p95_ms_gate: int,
) -> dict[str, Any]:
    dataset_by_id = {row["id"]: row for row in dataset_rows}
    prediction_by_id: dict[str, dict[str, Any]] = {}
    duplicate_prediction_ids: list[str] = []
    unknown_prediction_ids: list[str] = []
    for prediction in prediction_rows:
        case_id = prediction.get("id")
        if case_id not in dataset_by_id:
            unknown_prediction_ids.append(str(case_id))
            continue
        if case_id in prediction_by_id:
            duplicate_prediction_ids.append(case_id)
        prediction_by_id[case_id] = prediction

    joined = [
        {"case": case, "prediction": prediction_by_id[case["id"]]}
        for case in dataset_rows
        if case["id"] in prediction_by_id
    ]
    coverage_count = len(joined)
    coverage_value = coverage_count / len(dataset_rows) if dataset_rows else None
    report_status = "MEASURED" if coverage_count == len(dataset_rows) else "INVALID_PARTIAL"
    if duplicate_prediction_ids or unknown_prediction_ids:
        report_status = "INVALID_PARTIAL"
    if not joined:
        report_status = "N/A_NOT_RUN"

    configuration: dict[str, Any] = {}
    configuration_valid = bool(joined)
    for field in ("benchmark_version", "prompt_version", "model", "temperature"):
        values = sorted(
            {
                str(row["prediction"].get(field))
                for row in joined
                if row["prediction"].get(field) is not None
            }
        )
        missing = sum(row["prediction"].get(field) is None for row in joined)
        field_valid = missing == 0 and len(values) == 1
        configuration[field] = {
            "values": values,
            "missing": missing,
            "consistent": field_valid,
        }
        configuration_valid = configuration_valid and field_valid
    configuration["status"] = (
        "MEASURED" if configuration_valid else "INVALID_PARTIAL" if joined else "N/A_NOT_RUN"
    )
    if report_status == "MEASURED" and not configuration_valid:
        report_status = "INVALID_PARTIAL"

    intent = classification_summary(
        joined,
        INTENTS,
        lambda case: case["expected"]["expected_intent"],
        "intent",
        "intent_confidence",
        report_status,
    )
    policy = classification_summary(
        joined,
        ROUTE_ACTIONS,
        lambda case: case["expected"]["expected_route_action"],
        "route_action",
        "route_confidence",
        report_status,
    )

    error_rows = [row for row in joined if row["prediction"].get("error")]
    latency_values = [
        float(row["prediction"]["router_ms"])
        for row in joined
        if not row["prediction"].get("error")
        and isinstance(row["prediction"].get("router_ms"), (int, float))
        and row["prediction"]["router_ms"] >= 0
    ]
    attempt_values = [
        int(row["prediction"]["router_attempt_count"])
        for row in joined
        if not row["prediction"].get("error")
        and isinstance(row["prediction"].get("router_attempt_count"), int)
        and row["prediction"]["router_attempt_count"] >= 1
    ]
    single_attempt_latency = [
        float(row["prediction"]["router_ms"])
        for row in joined
        if not row["prediction"].get("error")
        and row["prediction"].get("router_attempt_count") == 1
        and isinstance(row["prediction"].get("router_ms"), (int, float))
    ]
    fallback_latency = [
        float(row["prediction"]["router_ms"])
        for row in joined
        if not row["prediction"].get("error")
        and isinstance(row["prediction"].get("router_attempt_count"), int)
        and row["prediction"]["router_attempt_count"] > 1
        and isinstance(row["prediction"].get("router_ms"), (int, float))
    ]
    false_accepts = [
        row
        for row in joined
        if row["case"]["expected"]["expected_route_action"] in REFUSAL_ACTIONS
        and row["prediction"].get("route_action") in ACTIVE_ACTIONS
    ]
    false_rejects = [
        row
        for row in joined
        if row["case"]["expected"]["expected_route_action"] in ACTIVE_ACTIONS
        and row["prediction"].get("route_action") in STOP_ACTIONS
    ]
    failures = []
    for row in joined:
        case = row["case"]
        prediction = row["prediction"]
        intent_wrong = prediction.get("intent") != case["expected"]["expected_intent"]
        action_observed = prediction.get("route_action") in ROUTE_ACTIONS
        action_wrong = (
            action_observed
            and prediction.get("route_action")
            != case["expected"]["expected_route_action"]
        )
        if prediction.get("error") or intent_wrong or action_wrong:
            failures.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "gold_intent": case["expected"]["expected_intent"],
                    "predicted_intent": prediction.get("intent"),
                    "intent_confidence": prediction.get("intent_confidence"),
                    "gold_route_action": case["expected"]["expected_route_action"],
                    "predicted_route_action": prediction.get("route_action"),
                    "route_confidence": prediction.get("route_confidence"),
                    "error": prediction.get("error"),
                }
            )

    unsafe_recall = policy.get("per_class", {}).get("refuse_unsafe", {}).get("recall")
    unsupported_recall = policy.get("per_class", {}).get("refuse_unsupported", {}).get("recall")
    web_recall = policy.get("per_class", {}).get("web_required", {}).get("recall")
    error_rate = len(error_rows) / coverage_count if coverage_count else None
    latency_p95 = nearest_rank(latency_values, 0.95)
    ece = intent.get("calibration", {}).get("ece")
    calibration_gate_status = (
        report_status
        if intent.get("calibration", {}).get("eligible_cases", 0) >= 30
        else "N/A_NOT_APPLICABLE"
    )
    gates = [
        gate("intent_accuracy", intent["accuracy"]["value"], "gte", 0.90, report_status),
        gate("intent_macro_f1", intent.get("macro_f1"), "gte", 0.85, report_status),
        gate("route_action_accuracy", policy["accuracy"]["value"], "gte", 0.90, report_status),
        gate("unsafe_routing_recall", unsafe_recall, "gte", 1.0, report_status),
        gate("unsupported_routing_recall", unsupported_recall, "gte", 0.90, report_status),
        gate("web_required_recall", web_recall, "gte", 0.90, report_status),
        gate("router_error_rate", error_rate, "lte", 0.02, report_status),
        gate("intent_ece", ece, "lte", 0.10, calibration_gate_status),
        gate("router_p95_ms", latency_p95, "lte", float(p95_ms_gate), report_status),
    ]

    return {
        "status": report_status,
        "coverage": {
            "dataset_cases": len(dataset_rows),
            "prediction_rows": len(prediction_rows),
            "unique_known_predictions": coverage_count,
            "value": coverage_value,
            "missing_ids": [row["id"] for row in dataset_rows if row["id"] not in prediction_by_id],
            "duplicate_prediction_ids": sorted(set(duplicate_prediction_ids)),
            "unknown_prediction_ids": sorted(set(unknown_prediction_ids)),
        },
        "configuration": configuration,
        "runtime": {
            "error_rate": ratio(len(error_rows), coverage_count),
            "avg_ms": mean(latency_values) if latency_values else None,
            "p50_ms": nearest_rank(latency_values, 0.50),
            "p95_ms": latency_p95,
            "max_ms": max(latency_values) if latency_values else None,
            "attempt_observability_status": (
                "MEASURED" if len(attempt_values) == len(latency_values) and attempt_values
                else "N/A_NOT_RUN"
            ),
            "avg_attempts": mean(attempt_values) if attempt_values else None,
            "fallback_case_rate": ratio(
                sum(attempt > 1 for attempt in attempt_values),
                len(attempt_values),
            ),
            "single_attempt_latency": {
                "cases": len(single_attempt_latency),
                "avg_ms": mean(single_attempt_latency) if single_attempt_latency else None,
                "p95_ms": nearest_rank(single_attempt_latency, 0.95),
            },
            "fallback_latency": {
                "cases": len(fallback_latency),
                "avg_ms": mean(fallback_latency) if fallback_latency else None,
                "p95_ms": nearest_rank(fallback_latency, 0.95),
            },
        },
        "intent": intent,
        "policy": policy,
        "false_accepts": [row["case"]["id"] for row in false_accepts],
        "false_rejects": [row["case"]["id"] for row in false_rejects],
        "slices": build_slices(joined),
        "failures": failures,
        "gates": gates,
    }


def fmt_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def render_markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    runtime = report["runtime"]
    intent = report["intent"]
    policy = report["policy"]
    lines = [
        "# Router Evaluation",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Coverage And Runtime",
        "",
        "| Metric | Numerator/denominator | Value |",
        "| --- | ---: | ---: |",
        f"| Prediction coverage | {coverage['unique_known_predictions']}/{coverage['dataset_cases']} | {fmt_percent(coverage['value'])} |",
        f"| Runtime errors | {runtime['error_rate']['numerator']}/{runtime['error_rate']['denominator']} | {fmt_percent(runtime['error_rate']['value'])} |",
        f"| Configuration consistency | - | {report['configuration']['status']} |",
        f"| Average latency | - | {runtime['avg_ms']:.0f} ms |" if runtime["avg_ms"] is not None else "| Average latency | - | n/a |",
        f"| P95 latency | - | {runtime['p95_ms']:.0f} ms |" if runtime["p95_ms"] is not None else "| P95 latency | - | n/a |",
        f"| Fallback cases | {runtime['fallback_case_rate']['numerator']}/{runtime['fallback_case_rate']['denominator']} | {fmt_percent(runtime['fallback_case_rate']['value'])} |",
        f"| Single-attempt latency | {runtime['single_attempt_latency']['cases']} cases | {runtime['single_attempt_latency']['avg_ms']:.0f} ms avg |" if runtime["single_attempt_latency"]["avg_ms"] is not None else "| Single-attempt latency | 0 cases | n/a |",
        f"| Fallback latency | {runtime['fallback_latency']['cases']} cases | {runtime['fallback_latency']['avg_ms']:.0f} ms avg |" if runtime["fallback_latency"]["avg_ms"] is not None else "| Fallback latency | 0 cases | n/a |",
        "",
        "## Quality Gates",
        "",
        "| Gate | Value | Threshold | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    for item in report["gates"]:
        threshold = f"{item['operator']} {item['threshold']}"
        value = "n/a" if item["value"] is None else f"{item['value']:.4f}"
        lines.append(f"| {item['gate']} | {value} | {threshold} | {item['status']} |")

    lines.extend(
        [
            "",
            "## Intent Per Class",
            "",
            "| Class | Support | TP/FP/FN | Precision | Recall | F1 | Risk |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for label in INTENTS:
        item = intent.get("per_class", {}).get(label)
        if not item:
            continue
        risk = "SAMPLE_RISK" if item["sample_risk"] else ""
        lines.append(
            f"| {label} | {item['support']} | {item['tp']}/{item['fp']}/{item['fn']} | "
            f"{fmt_percent(item['precision'])} | {fmt_percent(item['recall'])} | "
            f"{fmt_percent(item['f1'])} | {risk} |"
        )

    lines.extend(
        [
            "",
            "## Policy Per Class",
            "",
            "| Action | Support | TP/FP/FN | Precision | Recall | F1 | Risk |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for label in ROUTE_ACTIONS:
        item = policy.get("per_class", {}).get(label)
        if not item:
            continue
        risk = "SAMPLE_RISK" if item["sample_risk"] else ""
        lines.append(
            f"| {label} | {item['support']} | {item['tp']}/{item['fp']}/{item['fn']} | "
            f"{fmt_percent(item['precision'])} | {fmt_percent(item['recall'])} | "
            f"{fmt_percent(item['f1'])} | {risk} |"
        )

    lines.extend(["", "## Failures", ""])
    if not report["failures"]:
        lines.append("No observed failures.")
    else:
        lines.extend(
            [
                "| ID | Gold intent -> predicted | Gold action -> predicted | Error |",
                "| --- | --- | --- | --- |",
            ]
        )
        for failure in report["failures"]:
            error = str(failure.get("error") or "").replace("|", "\\|")[:120]
            lines.append(
                f"| {failure['id']} | {failure['gold_intent']} -> {failure['predicted_intent']} | "
                f"{failure['gold_route_action']} -> {failure['predicted_route_action']} | {error} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score router-only predictions.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--p95-ms-gate", type=int, default=6000)
    args = parser.parse_args()

    report = build_report(load_jsonl(args.dataset), load_jsonl(args.predictions), args.p95_ms_gate)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"Status: {report['status']}")
    print(
        "Coverage: "
        f"{report['coverage']['unique_known_predictions']}/{report['coverage']['dataset_cases']}"
    )
    print(f"Saved JSON report: {args.out_json}")
    print(f"Saved Markdown report: {args.out_md}")


if __name__ == "__main__":
    main()
