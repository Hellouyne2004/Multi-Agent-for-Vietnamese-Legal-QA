"""Offline evaluator for the public Vietnamese legal QA benchmark seed.

By default this script regenerates ``eval_reports/latest.md`` from the
committed baseline snapshot. When a predictions JSONL file is provided, it
scores retrieval, citation, and lightweight answer coverage deterministically.
It does not call Qdrant, Gemini, Tavily, or any external service.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "evaluation" / "legal_qa_eval_30.jsonl"
DEFAULT_BASELINE = ROOT / "eval_reports" / "baseline_metrics.json"
DEFAULT_OUT_JSON = ROOT / "eval_reports" / "latest.json"
DEFAULT_OUT_MD = ROOT / "eval_reports" / "latest.md"


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def get_metadata(doc: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(doc, dict):
        return {}
    return doc.get("metadata") or doc.get("payload") or doc


def equivalent(expected: Any, actual: Any) -> bool:
    if expected in (None, "", []):
        return True
    if actual in (None, "", []):
        return False
    return normalize_text(expected) == normalize_text(actual)


def first_relevant_rank(expected: dict[str, Any], docs: list[dict[str, Any]]) -> int | None:
    expected_doc_id = expected.get("doc_id")
    if not expected_doc_id:
        return None
    for index, doc in enumerate(docs, 1):
        metadata = get_metadata(doc)
        if equivalent(expected_doc_id, metadata.get("doc_id")):
            return index
    return None


def hit_for_field(expected: dict[str, Any], docs: list[dict[str, Any]], field: str) -> bool | None:
    expected_value = expected.get(field)
    if expected_value in (None, "", []):
        return None
    for doc in docs:
        metadata = get_metadata(doc)
        if equivalent(expected.get("doc_id"), metadata.get("doc_id")) and equivalent(
            expected_value,
            metadata.get(field),
        ):
            return True
    return False


def score_retrieval(cases: list[dict[str, Any]], predictions: dict[str, dict[str, Any]], top_k: int) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for case in cases:
        expected = case.get("expected", {})
        prediction = predictions.get(case["id"], {})
        docs = as_list(prediction.get("retrieved_documents"))[:top_k]
        rank = first_relevant_rank(expected, docs)
        scored.append(
            {
                "id": case["id"],
                "doc_hit": rank is not None if expected.get("doc_id") else None,
                "article_hit": hit_for_field(expected, docs, "article_number"),
                "clause_hit": hit_for_field(expected, docs, "clause_number"),
                "point_hit": hit_for_field(expected, docs, "point_label"),
                "level_hit": hit_for_field(expected, docs, "level"),
                "mrr": 1 / rank if rank else None,
                "retrieval_ms": prediction.get("retrieval_ms"),
            }
        )
    return {
        "cases": len(cases),
        "top_k": top_k,
        "doc_hit_at_k": bool_mean(scored, "doc_hit"),
        "article_hit_at_k": bool_mean(scored, "article_hit"),
        "clause_hit_at_k": bool_mean(scored, "clause_hit"),
        "point_hit_at_k": bool_mean(scored, "point_hit"),
        "level_hit_at_k": bool_mean(scored, "level_hit"),
        "mrr": float_mean(scored, "mrr"),
        "avg_retrieval_ms": float_mean(scored, "retrieval_ms"),
        "results": scored,
    }


def citation_ids(text: str) -> set[str]:
    return {match.group(1).replace(" ", "") for match in re.finditer(r"\[(S\d+|Web\s+\d+)\]", text or "")}


def score_answers(cases: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for case in cases:
        expected = case.get("expected", {})
        prediction = predictions.get(case["id"], {})
        answer = prediction.get("answer", "")
        normalized_answer = normalize_text(answer)
        expected_facts = [normalize_text(item) for item in expected.get("expected_facts", [])]
        forbidden_facts = [normalize_text(item) for item in expected.get("forbidden_facts", [])]
        covered = [fact for fact in expected_facts if fact and fact in normalized_answer]
        forbidden_found = [fact for fact in forbidden_facts if fact and fact in normalized_answer]
        citations = as_list(prediction.get("citations"))
        source_ids = citation_ids(answer)
        for citation in citations:
            if isinstance(citation, dict):
                source_ids.update(citation_ids(str(citation.get("source", ""))))
                source_ids.update(citation_ids(str(citation.get("text", ""))))
        needs_citation = bool(expected.get("doc_id"))
        citations_have_url = all(
            not isinstance(item, dict) or not citation_ids(str(item.get("source", ""))) or bool(item.get("url"))
            for item in citations
        )
        scored.append(
            {
                "id": case["id"],
                "fact_coverage": len(covered) / len(expected_facts) if expected_facts else None,
                "forbidden_ok": not forbidden_found,
                "display_citation_valid": bool(source_ids) if needs_citation else True,
                "citation_url_valid": citations_have_url,
                "answer_ms": prediction.get("answer_ms"),
            }
        )
    return {
        "cases": len(cases),
        "fact_coverage": float_mean(scored, "fact_coverage"),
        "forbidden_ok": bool_mean(scored, "forbidden_ok"),
        "display_citation_valid": bool_mean(scored, "display_citation_valid"),
        "citation_url_valid": bool_mean(scored, "citation_url_valid"),
        "avg_answer_ms": float_mean(scored, "answer_ms"),
        "results": scored,
    }


def bool_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(1 for value in values if bool(value)) / len(values)


def float_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return mean(values) if values else None


def fmt_percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.2f}%"


def fmt_number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


def repo_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_report(report: dict[str, Any], out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Legal QA Evaluation Report",
        "",
        f"- Created at: {report['created_at']}",
        f"- Dataset seed: `{report['dataset']}`",
        f"- Mode: {report['mode']}",
        "",
        "## Corpus",
        "",
    ]
    corpus = report.get("corpus", {})
    if corpus:
        lines.extend(
            [
                f"- Registry documents: {corpus.get('registry_documents', 'n/a')}",
                f"- Chunks: {corpus.get('chunks', 'n/a')}",
                f"- Chunk length average: {corpus.get('chunk_chars_avg', 'n/a')} chars",
                "",
            ]
        )

    retrieval = report.get("retrieval_summary", {})
    top_k = retrieval.get("top_k", 5)
    lines.extend(
        [
            "## Retrieval Summary",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Cases | {retrieval.get('cases', 'n/a')} |",
            f"| Doc Hit@{top_k} | {fmt_percent(retrieval.get('doc_hit_at_k', retrieval.get('doc_hit_at_5')))} |",
            f"| Article Hit@{top_k} | {fmt_percent(retrieval.get('article_hit_at_k', retrieval.get('article_hit_at_5')))} |",
            f"| Clause Hit@{top_k} | {fmt_percent(retrieval.get('clause_hit_at_k', retrieval.get('clause_hit_at_5')))} |",
            f"| Point Hit@{top_k} | {fmt_percent(retrieval.get('point_hit_at_k', retrieval.get('point_hit_at_5')))} |",
            f"| Level Hit@{top_k} | {fmt_percent(retrieval.get('level_hit_at_k', retrieval.get('level_hit_at_5')))} |",
            f"| MRR | {fmt_percent(retrieval.get('mrr'))} |",
            f"| Avg retrieval latency | {fmt_number(retrieval.get('avg_retrieval_ms'))} ms |",
            "",
        ]
    )

    answer = report.get("answer_summary", {})
    lines.extend(
        [
            "## Answer Summary",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Cases | {answer.get('cases', 'n/a')} |",
            f"| Fact Coverage | {fmt_percent(answer.get('fact_coverage'))} |",
            f"| Forbidden OK | {fmt_percent(answer.get('forbidden_ok'))} |",
            f"| Display Citation Valid | {fmt_percent(answer.get('display_citation_valid'))} |",
            f"| Citation URL Valid | {fmt_percent(answer.get('citation_url_valid', answer.get('source_mapping_valid')))} |",
            f"| Avg answer latency | {fmt_number(answer.get('avg_answer_ms'))} ms |",
            "",
        ]
    )

    if report.get("notes"):
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in report["notes"])
        lines.append("")

    lines.extend(
        [
            "## Reproduce",
            "",
            "```bash",
            "python scripts/evaluate_legal_qa.py",
            "python scripts/evaluate_legal_qa.py --predictions eval_reports/my_run_predictions.jsonl",
            "```",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def build_baseline_report(dataset: Path, baseline_path: Path) -> dict[str, Any]:
    baseline = load_json(baseline_path)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": repo_path(dataset),
        "mode": "baseline_snapshot",
        "notes": baseline.get("notes", []),
        "corpus": baseline.get("corpus", {}),
        "retrieval_summary": baseline.get("retrieval_summary", {}),
        "answer_summary": baseline.get("answer_summary", {}),
    }


def build_scored_report(dataset: Path, predictions_path: Path, top_k: int) -> dict[str, Any]:
    cases = load_jsonl(dataset)
    predictions = {row["id"]: row for row in load_jsonl(predictions_path)}
    retrieval_summary = score_retrieval(cases, predictions, top_k)
    answer_summary = score_answers(cases, predictions)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": repo_path(dataset),
        "mode": "scored_predictions",
        "notes": [
            f"Predictions scored from {predictions_path}.",
            "The evaluator uses deterministic string/metadata checks and does not call external services.",
        ],
        "retrieval_summary": retrieval_summary,
        "answer_summary": answer_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Vietnamese legal QA predictions.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.predictions:
        report = build_scored_report(args.dataset, args.predictions, args.top_k)
    else:
        report = build_baseline_report(args.dataset, args.baseline)

    write_report(report, args.out_json, args.out_md)
    print(f"Saved JSON report: {args.out_json}")
    print(f"Saved Markdown report: {args.out_md}")


if __name__ == "__main__":
    main()
