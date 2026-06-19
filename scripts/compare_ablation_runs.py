"""Compare multiple evaluation prediction files as ablation variants."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_legal_qa import (
    DEFAULT_DATASET,
    load_jsonl,
    score_answers,
    score_retrieval,
    fmt_percent,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_JSON = ROOT / "eval_reports" / "ablation_report.json"
DEFAULT_OUT_MD = ROOT / "eval_reports" / "ablation_report.md"


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be formatted as label=path")
    label, path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("Run label cannot be empty")
    return label.strip(), Path(path)


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in load_jsonl(path)}


def summarize_run(label: str, path: Path, cases: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    if not path.exists():
        return {
            "variant": label,
            "path": str(path),
            "status": "missing",
            "cases": 0,
        }
    predictions = load_predictions(path)
    retrieval = score_retrieval(cases, predictions, top_k=top_k)
    answer = score_answers(cases, predictions)
    return {
        "variant": label,
        "path": str(path),
        "status": "scored",
        "cases": len(predictions),
        "doc_hit_at_5": retrieval.get("doc_hit_at_k"),
        "context_fact_coverage_at_5": retrieval.get("context_fact_coverage_at_k"),
        "full_context_fact_case_rate_at_5": retrieval.get("full_context_fact_case_rate_at_k"),
        "forbidden_fact_in_context_rate_at_5": retrieval.get("forbidden_fact_in_context_rate_at_k"),
        "article_hit_at_5": retrieval.get("article_hit_at_k"),
        "clause_hit_at_5": retrieval.get("clause_hit_at_k"),
        "mrr": retrieval.get("mrr"),
        "fact_coverage": answer.get("fact_coverage"),
        "grounded_answer_rate": answer.get("grounded_answer_rate"),
        "unsupported_claim_rate": answer.get("unsupported_claim_rate"),
        "error_rate": answer.get("error_rate"),
        "avg_processing_time_ms": answer.get("avg_processing_time_ms"),
    }


def write_markdown(report: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# Retrieval And E2E Ablation Report",
        "",
        f"- Created at: {report['created_at']}",
        f"- Dataset: `{report['dataset']}`",
        "",
        "| Variant | Status | Cases | Doc Hit@5 | Context Fact Coverage@5 | Full Fact Cases@5 | Forbidden Context Facts@5 | Article Hit@5 diag | Clause Hit@5 diag | MRR | Answer Fact Coverage | Grounded Rate | Unsupported Claims | Error Rate | Avg latency ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report.get("runs", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(run.get("variant", "n/a")),
                    str(run.get("status", "n/a")),
                    str(run.get("cases", 0)),
                    fmt_percent(run.get("doc_hit_at_5")),
                    fmt_percent(run.get("context_fact_coverage_at_5")),
                    fmt_percent(run.get("full_context_fact_case_rate_at_5")),
                    fmt_percent(run.get("forbidden_fact_in_context_rate_at_5")),
                    fmt_percent(run.get("article_hit_at_5")),
                    fmt_percent(run.get("clause_hit_at_5")),
                    fmt_percent(run.get("mrr")),
                    fmt_percent(run.get("fact_coverage")),
                    fmt_percent(run.get("grounded_answer_rate")),
                    fmt_percent(run.get("unsupported_claim_rate")),
                    fmt_percent(run.get("error_rate")),
                    "n/a" if run.get("avg_processing_time_ms") is None else f"{run.get('avg_processing_time_ms'):.2f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Recommended Variants",
            "",
            "- `dense`: semantic/vector-only retrieval predictions, if produced by a separate retrieval runner.",
            "- `sparse`: keyword/BM25-style predictions, if produced by a separate retrieval runner.",
            "- `hybrid`: current Qdrant RRF hybrid retrieval predictions.",
            "- `full_graph`: full LangGraph multi-agent predictions from `scripts/run_e2e_eval.py`.",
        ]
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ablation prediction runs.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run", type=parse_run, action="append", default=[])
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    default_runs = [
        ("dense", ROOT / "eval_reports" / "dense_predictions.jsonl"),
        ("sparse", ROOT / "eval_reports" / "sparse_predictions.jsonl"),
        ("hybrid", ROOT / "eval_reports" / "hybrid_predictions.jsonl"),
        ("full_graph", ROOT / "eval_reports" / "e2e_predictions.jsonl"),
    ]
    runs = args.run or default_runs
    cases = load_jsonl(args.dataset)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "runs": [summarize_run(label, path, cases, args.top_k) for label, path in runs],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, args.out_md)
    print(f"Saved ablation JSON: {args.out_json}")
    print(f"Saved ablation Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
