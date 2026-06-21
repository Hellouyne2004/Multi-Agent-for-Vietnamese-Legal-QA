"""Score frozen-context Grader predictions without calling an LLM."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "evaluation" / "grader_eval_20.jsonl"
DEFAULT_PREDICTIONS = ROOT / "eval_reports" / "grader_predictions.jsonl"
DEFAULT_JSON = ROOT / "eval_reports" / "grader_eval_20.json"
DEFAULT_MD = ROOT / "eval_reports" / "grader_eval_20.md"
LABELS = ["yes", "no"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def nearest_rank(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def latency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["prediction"]["grader_ms"])
        for row in rows
        if row["prediction"].get("grader_ms") is not None
    ]
    return {
        "cases": len(values),
        "avg_ms": mean(values) if values else None,
        "p50_ms": nearest_rank(values, 0.50),
        "p95_ms": nearest_rank(values, 0.95),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float | None]:
    if not total:
        return [None, None]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for row in rows:
        confusion[row["case"]["context_sufficient"]][row["prediction"]["grader_verdict"]] += 1
    correct = sum(confusion[label][label] for label in LABELS)
    per_class: dict[str, Any] = {}
    f1_values = []
    for label in LABELS:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[gold][label] for gold in LABELS)
        fp = predicted - tp
        fn = support - tp
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        class_f1 = f1(precision, recall)
        if class_f1 is not None:
            f1_values.append(class_f1)
        per_class[label] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": class_f1,
            "recall_ci95": wilson_interval(tp, support),
        }
    return {
        "accuracy": ratio(correct, len(rows)),
        "accuracy_ci95": wilson_interval(correct, len(rows)),
        "macro_f1": mean(f1_values) if f1_values else None,
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def threshold_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        tp = tn = fp = fn = 0
        for row in rows:
            gold = row["case"]["context_sufficient"]
            predicted = "yes" if float(row["prediction"]["grader_score"]) >= threshold else "no"
            tp += int(gold == "yes" and predicted == "yes")
            tn += int(gold == "no" and predicted == "no")
            fp += int(gold == "no" and predicted == "yes")
            fn += int(gold == "yes" and predicted == "no")
        results.append({
            "threshold": threshold,
            "accuracy": safe_div(tp + tn, len(rows)),
            "false_positive_rate": safe_div(fp, fp + tn),
            "false_negative_rate": safe_div(fn, fn + tp),
            "fp": fp,
            "fn": fn,
            "weighted_error": 2 * fp + fn,
        })
    return results


def calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    for row in rows:
        try:
            score = float(row["prediction"].get("grader_score"))
        except (TypeError, ValueError):
            continue
        if 0 <= score <= 1:
            values.append((score, int(row["case"]["context_sufficient"] == "yes")))
    buckets = []
    ece = 0.0
    for index in range(5):
        lower, upper = index / 5, (index + 1) / 5
        bucket = [
            item for item in values
            if lower <= item[0] <= upper and (index == 4 or item[0] < upper)
        ]
        if not bucket:
            continue
        avg_score = mean(item[0] for item in bucket)
        accuracy = mean(item[1] for item in bucket)
        ece += len(bucket) / len(values) * abs(avg_score - accuracy)
        buckets.append({
            "lower": lower,
            "upper": upper,
            "count": len(bucket),
            "avg_score": avg_score,
            "yes_rate": accuracy,
        })
    brier = mean((score - gold) ** 2 for score, gold in values) if values else None
    return {"eligible_cases": len(values), "ece": ece if values else None, "brier": brier, "buckets": buckets}


def configuration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["benchmark_version", "context_version", "prompt_version", "model", "temperature", "request_timeout"]
    result: dict[str, Any] = {}
    for field in fields:
        values = sorted({str(row["prediction"].get(field)) for row in rows if row["prediction"].get(field) is not None})
        missing = sum(1 for row in rows if row["prediction"].get(field) is None)
        result[field] = {"values": values, "missing": missing, "consistent": len(values) == 1 and missing == 0}
    result["status"] = "MEASURED" if all(item.get("consistent") for item in result.values() if isinstance(item, dict)) else "INVALID_CONFIG"
    return result


def build_slices(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for dimension in ["slice", "context_origin", "category"]:
            groups[(dimension, str(row["case"].get(dimension)))].append(row)
    output = []
    for (dimension, value), group in sorted(groups.items()):
        correct = sum(row["case"]["context_sufficient"] == row["prediction"]["grader_verdict"] for row in group)
        output.append({
            "dimension": dimension,
            "value": value,
            "cases": len(group),
            "accuracy": ratio(correct, len(group)),
            "sample_risk": len(group) < 5,
        })
    return output


def score(
    dataset: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    assumed_web_ms: int = 3000,
    incomplete_reason: str | None = None,
) -> dict[str, Any]:
    case_by_id = {case["id"]: case for case in dataset}
    prediction_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        prediction_groups[str(prediction.get("id"))].append(prediction)
    duplicates = sorted(case_id for case_id, rows in prediction_groups.items() if len(rows) > 1)
    unknown = sorted(case_id for case_id in prediction_groups if case_id not in case_by_id)
    known_predictions = {case_id: rows[-1] for case_id, rows in prediction_groups.items() if case_id in case_by_id}
    missing = [case_id for case_id in case_by_id if case_id not in known_predictions]
    joined = [{"case": case_by_id[case_id], "prediction": prediction} for case_id, prediction in known_predictions.items()]
    uncertain = [row for row in joined if row["case"].get("context_sufficient") == "uncertain"]
    eligible = [
        row for row in joined
        if row["case"].get("context_sufficient") in LABELS
        and not row["prediction"].get("error")
        and row["prediction"].get("grader_verdict") in LABELS
    ]
    errors = [row for row in joined if row["prediction"].get("error")]
    config = configuration(joined)
    complete = not missing and not duplicates and not unknown and config["status"] == "MEASURED"
    status = "MEASURED" if complete else (incomplete_reason or "INVALID_PARTIAL")
    classification = classification_metrics(eligible) if eligible else {
        "accuracy": ratio(0, 0), "accuracy_ci95": [None, None], "macro_f1": None,
        "confusion_matrix": {}, "per_class": {},
    }
    false_positives = [
        row for row in eligible
        if row["case"]["context_sufficient"] == "no" and row["prediction"]["grader_verdict"] == "yes"
    ]
    false_negatives = [
        row for row in eligible
        if row["case"]["context_sufficient"] == "yes" and row["prediction"]["grader_verdict"] == "no"
    ]
    no_support = sum(row["case"]["context_sufficient"] == "no" for row in eligible)
    yes_support = sum(row["case"]["context_sufficient"] == "yes" for row in eligible)
    latencies = [float(row["prediction"]["grader_ms"]) for row in eligible if row["prediction"].get("grader_ms") is not None]
    attempts = [float(row["prediction"]["grader_attempt_count"]) for row in eligible if row["prediction"].get("grader_attempt_count") is not None]
    fallback_rows = [row for row in eligible if (row["prediction"].get("grader_attempt_count") or 0) > 1]
    threshold_rows = [row for row in eligible if row["prediction"].get("grader_score") is not None]
    thresholds = threshold_metrics(threshold_rows) if threshold_rows else []
    best_observed_threshold = min(
        thresholds,
        key=lambda row: (row["weighted_error"], -row["accuracy"], abs(row["threshold"] - 0.5)),
    ) if thresholds else None
    failures = []
    for row in [*false_positives, *false_negatives, *errors]:
        if row in errors:
            failure_type = "runtime_or_parse_error"
        elif row in false_positives:
            failure_type = "false_positive_unsafe_yes"
        else:
            failure_type = "false_negative_extra_fallback"
        failures.append({
            "id": row["case"]["id"],
            "question": row["case"].get("question"),
            "slice": row["case"].get("slice"),
            "gold": row["case"].get("context_sufficient"),
            "predicted": row["prediction"].get("grader_verdict"),
            "score": row["prediction"].get("grader_score"),
            "missing_facts": row["case"].get("missing_facts", []),
            "gold_reason": row["case"].get("gold_reason"),
            "reasoning": row["prediction"].get("grader_reasoning"),
            "error": row["prediction"].get("error"),
            "failure_type": failure_type,
        })
    no_recall = classification.get("per_class", {}).get("no", {}).get("recall")
    accuracy = classification["accuracy"]["value"]
    error_rate = safe_div(len(errors), len(joined))
    p95 = nearest_rank(latencies, 0.95)
    fp_rate = safe_div(len(false_positives), no_support)
    deterministic_rows = [row for row in eligible if (row["prediction"].get("grader_attempt_count") or 0) == 0]
    single_attempt_rows = [row for row in eligible if row["prediction"].get("grader_attempt_count") == 1]
    def gate_status(value: float | None, passed: bool) -> str:
        return "N/A_NOT_RUN" if value is None else ("PASS" if passed else "FAIL")

    gates = [
        {"gate": "accuracy", "value": accuracy, "rule": ">=0.85", "status": gate_status(accuracy, accuracy is not None and accuracy >= 0.85)},
        {"gate": "insufficient_context_recall", "value": no_recall, "rule": ">=0.90", "status": gate_status(no_recall, no_recall is not None and no_recall >= 0.90)},
        {"gate": "false_positive_rate", "value": fp_rate, "rule": "<=0.10", "status": gate_status(fp_rate, fp_rate is not None and fp_rate <= 0.10)},
        {"gate": "error_rate", "value": error_rate, "rule": "<=0.02", "status": gate_status(error_rate, error_rate is not None and error_rate <= 0.02)},
        {"gate": "p95_ms", "value": p95, "rule": "<=6000", "status": gate_status(p95, p95 is not None and p95 <= 6000)},
    ]
    return {
        "status": status,
        "coverage": {
            "dataset_cases": len(dataset), "prediction_rows": len(predictions),
            "unique_known_predictions": len(known_predictions), "missing_ids": missing,
            "duplicate_ids": duplicates, "unknown_ids": unknown,
            "uncertain_cases": len(uncertain),
        },
        "configuration": config,
        "classification": classification,
        "false_positive_rate": ratio(len(false_positives), no_support),
        "false_negative_rate": ratio(len(false_negatives), yes_support),
        "runtime": {
            "error_rate": ratio(len(errors), len(joined)),
            "avg_ms": mean(latencies) if latencies else None,
            "p50_ms": nearest_rank(latencies, 0.50), "p95_ms": p95,
            "avg_attempts": mean(attempts) if attempts else None,
            "fallback_rate": ratio(len(fallback_rows), len(eligible)),
            "deterministic": latency_summary(deterministic_rows),
            "single_attempt": latency_summary(single_attempt_rows),
            "fallback": latency_summary(fallback_rows),
        },
        "calibration": calibration(eligible),
        "threshold_sensitivity": thresholds,
        "best_observed_threshold": best_observed_threshold,
        "recommended_threshold": None,
        "threshold_assessment": {
            "status": "N/A_NOT_APPLICABLE",
            "reason": "grader_score is relevance_score, not a calibrated probability that context is sufficient",
        },
        "false_negative_cost": {
            "extra_web_calls": len(false_negatives),
            "extra_api_calls": len(false_negatives),
            "assumed_web_ms_per_call": assumed_web_ms,
            "estimated_extra_latency_ms": len(false_negatives) * assumed_web_ms,
            "status": "SCENARIO_ESTIMATE",
        },
        "failures": failures,
        "slices": build_slices(eligible),
        "gates": gates,
        "annotation": {
            "protocol": "yes: all core facts and no serious conflict; no: missing core fact, wrong/expired version, freshness need, or empty/irrelevant context",
            "agreement_status": dict(Counter(case.get("annotation", {}).get("agreement_status") for case in dataset)),
        },
    }


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def render_markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    classification = report["classification"]
    runtime = report["runtime"]
    avg_ms = "n/a" if runtime["avg_ms"] is None else f"{runtime['avg_ms']:.0f}"
    p50_ms = "n/a" if runtime["p50_ms"] is None else f"{runtime['p50_ms']:.0f}"
    p95_ms = "n/a" if runtime["p95_ms"] is None else f"{runtime['p95_ms']:.0f}"
    avg_attempts = "n/a" if runtime["avg_attempts"] is None else f"{runtime['avg_attempts']:.2f}"
    lines = [
        "# Grader Evaluation", "", f"Status: **{report['status']}**", "",
        "## Coverage", "",
        "| Metric | Value |", "| --- | ---: |",
        f"| Prediction coverage | {coverage['unique_known_predictions']}/{coverage['dataset_cases']} |",
        f"| Uncertain gold labels | {coverage['uncertain_cases']} |",
        f"| Configuration | {report['configuration']['status']} |", "",
        "## Quality Gates", "", "| Gate | Value | Rule | Status |", "| --- | ---: | ---: | --- |",
    ]
    for gate in report["gates"]:
        value = gate["value"]
        rendered = f"{value:.4f}" if isinstance(value, float) and gate["gate"] != "p95_ms" else ("n/a" if value is None else f"{value:.0f}")
        lines.append(f"| {gate['gate']} | {rendered} | {gate['rule']} | {gate['status']} |")
    ci = classification["accuracy_ci95"]
    lines += [
        "", "## Classification", "",
        f"Accuracy: **{classification['accuracy']['numerator']}/{classification['accuracy']['denominator']} = {percent(classification['accuracy']['value'])}** "
        f"(Wilson 95% CI {percent(ci[0])}-{percent(ci[1])}).", "",
        "| Gold / predicted | yes | no |", "| --- | ---: | ---: |",
        f"| yes | {classification['confusion_matrix'].get('yes', {}).get('yes', 0)} | {classification['confusion_matrix'].get('yes', {}).get('no', 0)} |",
        f"| no | {classification['confusion_matrix'].get('no', {}).get('yes', 0)} | {classification['confusion_matrix'].get('no', {}).get('no', 0)} |", "",
        "| Class | TP/FP/FN | Precision | Recall | F1 |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in LABELS:
        item = classification.get("per_class", {}).get(label, {})
        lines.append(f"| {label} | {item.get('tp', 0)}/{item.get('fp', 0)}/{item.get('fn', 0)} | {percent(item.get('precision'))} | {percent(item.get('recall'))} | {percent(item.get('f1'))} |")
    lines += [
        "", "## Runtime", "", "| Metric | Value |", "| --- | ---: |",
        f"| Error rate | {runtime['error_rate']['numerator']}/{runtime['error_rate']['denominator']} = {percent(runtime['error_rate']['value'])} |",
        f"| Avg / P50 / P95 latency | {avg_ms} / {p50_ms} / {p95_ms} ms |",
        f"| Average attempts | {avg_attempts} |",
        f"| Fallback rate | {runtime['fallback_rate']['numerator']}/{runtime['fallback_rate']['denominator']} = {percent(runtime['fallback_rate']['value'])} |",
        f"| Deterministic no-call cases | {runtime['deterministic']['cases']} |",
        f"| Single-attempt cases, avg/P95 | {runtime['single_attempt']['cases']}, {runtime['single_attempt']['avg_ms'] or 0:.0f}/{runtime['single_attempt']['p95_ms'] or 0:.0f} ms |",
        f"| Fallback cases, avg/P95 | {runtime['fallback']['cases']}, {runtime['fallback']['avg_ms'] or 0:.0f}/{runtime['fallback']['p95_ms'] or 0:.0f} ms |",
        "", "## Failures", "",
    ]
    if not report["failures"]:
        lines.append("No failures were measured." if report["status"] == "MEASURED" else "N/A_NOT_RUN: no successful predictions are available for failure analysis.")
    else:
        lines += ["| ID | Type | Gold -> predicted | Missing facts |", "| --- | --- | --- | --- |"]
        for failure in report["failures"]:
            missing = "; ".join(failure["missing_facts"]) or "-"
            lines.append(f"| {failure['id']} | {failure['failure_type']} | {failure['gold']} -> {failure['predicted']} | {missing} |")
    best_threshold = report.get("best_observed_threshold")
    if best_threshold:
        lines += [
            "", "## Threshold", "",
            "**N/A_NOT_APPLICABLE:** `grader_score` measures relevance rather than a calibrated probability of context sufficiency.",
            f"The best observed diagnostic sweep point was {best_threshold['threshold']:.1f}, "
            f"with {best_threshold['fp']} FP and {best_threshold['fn']} FN; it is not a deployment recommendation.",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--assumed-web-ms", type=int, default=3000)
    parser.add_argument(
        "--incomplete-reason",
        choices=["N/A_NOT_RUN", "N/A_NO_LABEL", "INVALID_PARTIAL", "BLOCKED_QUOTA", "BLOCKED_RUNTIME"],
        default=None,
    )
    args = parser.parse_args()
    report = score(
        load_jsonl(args.dataset),
        load_jsonl(args.predictions),
        args.assumed_web_ms,
        incomplete_reason=args.incomplete_reason,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"Status: {report['status']}")
    print(f"Coverage: {report['coverage']['unique_known_predictions']}/{report['coverage']['dataset_cases']}")
    print(f"Saved JSON report: {args.out_json}")
    print(f"Saved Markdown report: {args.out_md}")


if __name__ == "__main__":
    main()
