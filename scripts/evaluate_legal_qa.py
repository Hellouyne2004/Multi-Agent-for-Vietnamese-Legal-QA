"""Evaluation suite for the Vietnamese Legal Multi-Agent RAG system.

The default path is fully offline: it can regenerate public reports from the
committed dataset and baseline snapshot without Qdrant, Gemini, Tavily, or an
API server. When a predictions JSONL file is provided, it scores retrieval,
agent decisions, citation grounding, answer quality, latency, and failures.

Optional LLM-as-judge scoring is intentionally opt-in via ``--llm-judge`` so
the deterministic evaluator remains reproducible and cheap by default.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "evaluation" / "legal_qa_eval_100.jsonl"
DEFAULT_BASELINE = ROOT / "eval_reports" / "baseline_metrics.json"
DEFAULT_OUT_JSON = ROOT / "eval_reports" / "latest.json"
DEFAULT_OUT_MD = ROOT / "eval_reports" / "latest.md"
DEFAULT_FAILURE_MD = ROOT / "eval_reports" / "failure_analysis.md"
DEFAULT_ABLATION = ROOT / "eval_reports" / "ablation_report.json"
DEFAULT_REGISTRY = ROOT / "data" / "processed" / "document_registry.jsonl"
DEFAULT_CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
DEFAULT_INGESTION_REPORT = ROOT / "data" / "processed" / "ingestion_quality_report.md"

QUALITY_GATES = {
    "corpus_missing_metadata_total": ("lte", 0),
    "corpus_chunk_avg_chars": ("between", (500, 1200)),
    "router_intent_accuracy": ("gte", 0.90),
    "router_refusal_accuracy": ("gte", 0.90),
    "retrieval_doc_hit_at_k": ("gte", 0.95),
    "retrieval_article_hit_at_k": ("gte", 0.85),
    "retrieval_clause_hit_at_k": ("gte", 0.75),
    "retrieval_mrr": ("gte", 0.80),
    "generation_fact_coverage": ("gte", 0.75),
    "generation_forbidden_fact_rate": ("lte", 0.05),
    "generation_display_citation_valid": ("gte", 0.95),
    "generation_citation_url_valid": ("gte", 0.95),
    "e2e_grounded_answer_rate": ("gte", 0.80),
    "e2e_unsupported_claim_rate": ("lte", 0.08),
    "reliability_error_rate": ("lte", 0.05),
}


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
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def normalize_text(text: Any) -> str:
    raw = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    normalized = unicodedata.normalize("NFD", raw).replace("đ", "d")
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", without_marks)


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


def bool_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(1 for value in values if bool(value)) / len(values)


def float_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return mean(values) if values else None


def percentile(values: list[float], p: float) -> float | None:
    clean = sorted(float(value) for value in values if isinstance(value, (int, float)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return clean[low]
    return clean[low] * (high - rank) + clean[high] * (rank - low)


def fmt_percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.2f}%"


def fmt_number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


def repo_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


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


def citation_ids(text: str) -> set[str]:
    return {
        match.group(1).replace(" ", "")
        for match in re.finditer(r"\[(S\d+|Web\s+\d+)\]", text or "", flags=re.IGNORECASE)
    }


def numeric_facts(text: str) -> set[str]:
    normalized = (text or "").replace(",", ".")
    return {
        match.group(0).strip(".")
        for match in re.finditer(r"\b\d+(?:\.\d+)?%?\b", normalized)
    }


def docs_context_text(prediction: dict[str, Any]) -> str:
    parts: list[str] = []
    for doc in as_list(prediction.get("retrieved_documents")):
        if not isinstance(doc, dict):
            continue
        parts.append(str(doc.get("content", "")))
        metadata = get_metadata(doc)
        parts.append(" ".join(str(value) for value in metadata.values()))
    for result in as_list(prediction.get("web_results")):
        if isinstance(result, dict):
            parts.extend([str(result.get("title", "")), str(result.get("content", "")), str(result.get("url", ""))])
    return "\n".join(parts)


def expected_intent_for_case(case: dict[str, Any]) -> str | None:
    expected = case.get("expected", {})
    explicit = expected.get("expected_intent") or case.get("expected_intent")
    if explicit:
        return explicit
    policy = case.get("answer_policy") or expected.get("answer_policy")
    category = case.get("category", "")
    case_type = case.get("type", "")
    if policy in {"refuse_or_redirect", "unsafe_refusal", "unsupported_refusal"}:
        return "out_of_scope"
    if category in {"out_of_scope", "unsafe", "unsupported"}:
        return "out_of_scope"
    if case_type == "general_chat":
        return "general_chat"
    if case_type == "procedure" or category == "procedural":
        return "procedural"
    if expected.get("doc_id"):
        return "legal_query"
    return None


def expected_grader_verdict_for_case(case: dict[str, Any]) -> str | None:
    expected = case.get("expected", {})
    explicit = expected.get("expected_grader_verdict") or case.get("expected_grader_verdict")
    if explicit:
        return explicit
    if case.get("requires_web") is True:
        return "no"
    if expected.get("doc_id"):
        return "yes"
    return None


def expected_refusal(case: dict[str, Any]) -> bool:
    expected = case.get("expected", {})
    policy = case.get("answer_policy") or expected.get("answer_policy")
    category = case.get("category", "")
    return bool(
        policy in {"refuse_or_redirect", "unsafe_refusal", "unsupported_refusal"}
        or category in {"out_of_scope", "unsafe", "unsupported"}
        or (expected.get("doc_id") is None and policy != "grounded_answer")
    )


def prediction_error_category(prediction: dict[str, Any]) -> str | None:
    error = str(prediction.get("error") or "")
    if not error:
        return None
    normalized = normalize_text(error)
    quota_terms = [
        "resource_exhausted",
        "quota",
        "rate_limit",
        "rate limit",
        "429",
        "generate_content_free_tier_requests",
    ]
    if any(term in normalized for term in quota_terms):
        return "quota_or_rate_limit"
    return "runtime_error"


def is_refusal_text(answer: str) -> bool:
    normalized = normalize_text(answer)
    refusal_terms = [
        "ngoai pham vi",
        "khong thuoc pham vi",
        "khong du can cu",
        "khong the tra loi",
        "khong tim thay tai lieu",
        "hoi chuyen gia",
        "chuyen gia y te",
        "khong ho tro",
    ]
    return any(term in normalized for term in refusal_terms)


def score_router(cases: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for case in cases:
        expected_intent = expected_intent_for_case(case)
        if not expected_intent:
            continue
        prediction = predictions.get(case["id"], {})
        actual_intent = prediction.get("intent")
        is_correct = actual_intent == expected_intent
        scored.append(
            {
                "id": case["id"],
                "expected_intent": expected_intent,
                "actual_intent": actual_intent,
                "correct": is_correct,
                "is_refusal_case": expected_intent in {"out_of_scope", "general_chat"},
                "failure_reasons": [] if is_correct else ["wrong_intent"],
            }
        )
    return {
        "cases": len(scored),
        "intent_accuracy": bool_mean(scored, "correct"),
        "refusal_accuracy": bool_mean([row for row in scored if row["is_refusal_case"]], "correct"),
        "results": scored,
    }


def score_grader(cases: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for case in cases:
        expected_verdict = expected_grader_verdict_for_case(case)
        if not expected_verdict:
            continue
        prediction = predictions.get(case["id"], {})
        actual = prediction.get("grader_verdict")
        is_correct = actual == expected_verdict
        scored.append(
            {
                "id": case["id"],
                "expected_grader_verdict": expected_verdict,
                "actual_grader_verdict": actual,
                "correct": is_correct,
                "grader_score": prediction.get("grader_score"),
                "failure_reasons": [] if is_correct else ["wrong_grader_verdict"],
            }
        )
    return {
        "cases": len(scored),
        "grader_accuracy": bool_mean(scored, "correct"),
        "avg_grader_score": float_mean(scored, "grader_score"),
        "results": scored,
    }


def score_retrieval(cases: list[dict[str, Any]], predictions: dict[str, dict[str, Any]], top_k: int) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for case in cases:
        expected = case.get("expected", {})
        prediction = predictions.get(case["id"], {})
        docs = as_list(prediction.get("retrieved_documents"))[:top_k]
        rank = first_relevant_rank(expected, docs)
        doc_hit = rank is not None if expected.get("doc_id") else None
        article_hit = hit_for_field(expected, docs, "article_number")
        clause_hit = hit_for_field(expected, docs, "clause_number")
        point_hit = hit_for_field(expected, docs, "point_label")
        level_hit = hit_for_field(expected, docs, "level")
        failure_reasons: list[str] = []
        if doc_hit is False:
            failure_reasons.append("wrong_doc")
        if article_hit is False:
            failure_reasons.append("wrong_article")
        if clause_hit is False:
            failure_reasons.append("wrong_clause")
        if point_hit is False:
            failure_reasons.append("wrong_point")
        scored.append(
            {
                "id": case["id"],
                "doc_hit": doc_hit,
                "article_hit": article_hit,
                "clause_hit": clause_hit,
                "point_hit": point_hit,
                "level_hit": level_hit,
                "mrr": 1 / rank if rank else None,
                "retrieval_ms": prediction.get("retrieval_ms"),
                "failure_reasons": failure_reasons,
            }
        )
    latencies = [row["retrieval_ms"] for row in scored if isinstance(row.get("retrieval_ms"), (int, float))]
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
        "p95_retrieval_ms": percentile(latencies, 0.95),
        "results": scored,
    }


def score_answers(cases: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for case in cases:
        expected = case.get("expected", {})
        prediction = predictions.get(case["id"], {})
        answer = prediction.get("answer", "")
        error_category = prediction_error_category(prediction)
        if error_category:
            scored.append(
                {
                    "id": case["id"],
                    "fact_coverage": None,
                    "forbidden_ok": None,
                    "forbidden_fact_rate": None,
                    "display_citation_valid": None,
                    "citation_url_valid": None,
                    "unsupported_claim": None,
                    "unsupported_numbers": [],
                    "refusal_case": expected_refusal(case),
                    "refusal_ok": None,
                    "grounded_answer": None,
                    "answer_ms": prediction.get("answer_ms"),
                    "processing_time_ms": prediction.get("processing_time_ms"),
                    "generation_attempt": prediction.get("generation_attempt"),
                    "web_used": bool(prediction.get("web_results") or prediction.get("web_result_ids")),
                    "error": error_category != "quota_or_rate_limit",
                    "quota_error": error_category == "quota_or_rate_limit",
                    "runtime_error": error_category == "runtime_error",
                    "llm_faithfulness": None,
                    "llm_completeness": None,
                    "llm_usefulness": None,
                    "llm_citation_support": None,
                    "failure_reasons": [error_category],
                }
            )
            continue
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

        needs_citation = bool(expected.get("doc_id")) and not expected_refusal(case)
        citations_have_url = all(
            not isinstance(item, dict)
            or not citation_ids(str(item.get("source", "")) + str(item.get("text", "")))
            or bool(item.get("url"))
            for item in citations
        )

        context_text = docs_context_text(prediction)
        answer_numbers = numeric_facts(answer)
        context_numbers = numeric_facts(context_text)
        expected_numbers = numeric_facts(" ".join(expected.get("expected_facts", [])))
        citation_numbers = {re.sub(r"\D", "", source_id) for source_id in source_ids}
        unsupported_numbers = answer_numbers - context_numbers - expected_numbers - citation_numbers
        unsupported_claim = bool(unsupported_numbers) if answer_numbers else False

        refusal_case = expected_refusal(case)
        refusal_ok = is_refusal_text(answer) if refusal_case else None
        fact_coverage = len(covered) / len(expected_facts) if expected_facts else None
        forbidden_ok = not forbidden_found
        display_citation_valid = bool(source_ids) if needs_citation else True
        grounded_answer = (
            (fact_coverage is None or fact_coverage >= 0.75)
            and forbidden_ok
            and display_citation_valid
            and citations_have_url
            and not unsupported_claim
            and (refusal_ok is not False)
        )

        failure_reasons: list[str] = []
        if fact_coverage is not None and fact_coverage < 0.75:
            failure_reasons.append("missing_fact")
        if forbidden_found:
            failure_reasons.append("forbidden_fact")
        if not display_citation_valid:
            failure_reasons.append("invalid_citation")
        if not citations_have_url:
            failure_reasons.append("citation_missing_url")
        if unsupported_claim:
            failure_reasons.append("unsupported_claim")
        if refusal_ok is False:
            failure_reasons.append("refusal_error")

        llm_judge = prediction.get("llm_judge") if isinstance(prediction.get("llm_judge"), dict) else {}
        scored.append(
            {
                "id": case["id"],
                "fact_coverage": fact_coverage,
                "forbidden_ok": forbidden_ok,
                "forbidden_fact_rate": 0.0 if forbidden_ok else 1.0,
                "display_citation_valid": display_citation_valid,
                "citation_url_valid": citations_have_url,
                "unsupported_claim": unsupported_claim,
                "unsupported_numbers": sorted(unsupported_numbers),
                "refusal_case": refusal_case,
                "refusal_ok": refusal_ok,
                "grounded_answer": grounded_answer,
                "answer_ms": prediction.get("answer_ms"),
                "processing_time_ms": prediction.get("processing_time_ms"),
                "generation_attempt": prediction.get("generation_attempt"),
                "web_used": bool(prediction.get("web_results") or prediction.get("web_result_ids")),
                "error": False,
                "quota_error": False,
                "runtime_error": False,
                "llm_faithfulness": llm_judge.get("faithfulness"),
                "llm_completeness": llm_judge.get("completeness"),
                "llm_usefulness": llm_judge.get("usefulness"),
                "llm_citation_support": llm_judge.get("citation_support"),
                "failure_reasons": failure_reasons,
            }
        )

    processing_latencies = [
        row["processing_time_ms"]
        for row in scored
        if isinstance(row.get("processing_time_ms"), (int, float))
    ]
    return {
        "cases": len(cases),
        "fact_coverage": float_mean(scored, "fact_coverage"),
        "forbidden_ok": bool_mean(scored, "forbidden_ok"),
        "forbidden_fact_rate": float_mean(scored, "forbidden_fact_rate"),
        "display_citation_valid": bool_mean(scored, "display_citation_valid"),
        "citation_url_valid": bool_mean(scored, "citation_url_valid"),
        "unsupported_claim_rate": bool_mean(scored, "unsupported_claim"),
        "refusal_accuracy": bool_mean([row for row in scored if row["refusal_case"]], "refusal_ok"),
        "grounded_answer_rate": bool_mean(scored, "grounded_answer"),
        "avg_answer_ms": float_mean(scored, "answer_ms"),
        "avg_processing_time_ms": float_mean(scored, "processing_time_ms"),
        "p95_processing_time_ms": percentile(processing_latencies, 0.95),
        "avg_generation_attempts": float_mean(scored, "generation_attempt"),
        "web_fallback_rate": bool_mean(scored, "web_used"),
        "error_rate": bool_mean(scored, "error"),
        "quota_error_rate": bool_mean(scored, "quota_error"),
        "runtime_error_rate": bool_mean(scored, "runtime_error"),
        "llm_judge": {
            "faithfulness": float_mean(scored, "llm_faithfulness"),
            "completeness": float_mean(scored, "llm_completeness"),
            "usefulness": float_mean(scored, "llm_usefulness"),
            "citation_support": float_mean(scored, "llm_citation_support"),
        },
        "results": scored,
    }


def parse_ingestion_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    parsed: dict[str, Any] = {}
    missing_match = re.search(r"Missing metadata:\s+`([^`]+)`", text)
    if missing_match:
        try:
            parsed["missing"] = ast.literal_eval(missing_match.group(1))
        except Exception:
            parsed["missing"] = {}
    levels_match = re.search(r"Levels:\s+`([^`]+)`", text)
    if levels_match:
        try:
            parsed["levels"] = ast.literal_eval(levels_match.group(1))
        except Exception:
            parsed["levels"] = {}
    chars_match = re.search(r"Chunk chars:\s+min=(\d+),\s+avg=([\d.]+),\s+max=(\d+)", text)
    if chars_match:
        parsed["chunk_chars_min"] = int(chars_match.group(1))
        parsed["chunk_chars_avg"] = float(chars_match.group(2))
        parsed["chunk_chars_max"] = int(chars_match.group(3))
    registry_match = re.search(r"Registry documents:\s+(\d+)", text)
    chunks_match = re.search(r"Chunks:\s+(\d+)", text)
    if registry_match:
        parsed["registry_documents"] = int(registry_match.group(1))
    if chunks_match:
        parsed["chunks"] = int(chunks_match.group(1))
    return parsed


def build_corpus_summary(
    *,
    baseline: dict[str, Any] | None = None,
    chunks_path: Path = DEFAULT_CHUNKS,
    registry_path: Path = DEFAULT_REGISTRY,
    ingestion_report_path: Path = DEFAULT_INGESTION_REPORT,
) -> dict[str, Any]:
    baseline = baseline or {}
    if chunks_path.exists():
        chunks = load_jsonl(chunks_path)
        registry = load_jsonl(registry_path) if registry_path.exists() else []
        lengths = [len(str(chunk.get("content") or chunk.get("chunk_text") or "")) for chunk in chunks]
        levels = Counter(chunk.get("level", "") for chunk in chunks)
        missing = {
            "missing_doc_id": sum(1 for chunk in chunks if not chunk.get("doc_id")),
            "missing_page": sum(1 for chunk in chunks if not chunk.get("page_start")),
            "missing_source_url": sum(1 for chunk in chunks if not chunk.get("source_url")),
            "missing_article_label": sum(
                1
                for chunk in chunks
                if str(chunk.get("level", "")).startswith(("article", "clause", "point"))
                and not (chunk.get("dieu") or chunk.get("article_number"))
            ),
        }
        return {
            "source": repo_path(chunks_path),
            "registry_documents": len(registry),
            "chunks": len(chunks),
            "levels": dict(levels),
            "missing": missing,
            "missing_metadata_total": sum(missing.values()),
            "chunk_chars_min": min(lengths) if lengths else None,
            "chunk_chars_avg": round(mean(lengths), 1) if lengths else None,
            "chunk_chars_max": max(lengths) if lengths else None,
            "short_chunks": sum(1 for value in lengths if value < 120),
            "long_chunks": sum(1 for value in lengths if value > 2200),
        }

    report = parse_ingestion_report(ingestion_report_path)
    corpus = baseline.get("corpus", {}) if isinstance(baseline, dict) else {}
    missing = report.get("missing", {})
    levels = report.get("levels", {})
    return {
        "source": repo_path(ingestion_report_path) if ingestion_report_path.exists() else "baseline_metrics.json",
        "registry_documents": report.get("registry_documents", corpus.get("registry_documents")),
        "chunks": report.get("chunks", corpus.get("chunks")),
        "levels": levels,
        "missing": missing,
        "missing_metadata_total": sum(value for value in missing.values() if isinstance(value, int)) if missing else None,
        "chunk_chars_min": report.get("chunk_chars_min", corpus.get("chunk_chars_min")),
        "chunk_chars_avg": report.get("chunk_chars_avg", corpus.get("chunk_chars_avg")),
        "chunk_chars_max": report.get("chunk_chars_max", corpus.get("chunk_chars_max")),
        "short_chunks": None,
        "long_chunks": None,
    }


def gate_status(value: Any, rule: tuple[str, Any]) -> str:
    if value is None:
        return "N/A"
    op, threshold = rule
    if op == "gte":
        return "PASS" if float(value) >= float(threshold) else "FAIL"
    if op == "lte":
        return "PASS" if float(value) <= float(threshold) else "FAIL"
    if op == "between":
        low, high = threshold
        return "PASS" if low <= float(value) <= high else "FAIL"
    return "N/A"


def build_quality_gates(report: dict[str, Any]) -> list[dict[str, Any]]:
    corpus = report.get("corpus", {})
    router = report.get("router_summary", {})
    retrieval = report.get("retrieval_summary", {})
    answer = report.get("answer_summary", {})
    values = {
        "corpus_missing_metadata_total": corpus.get("missing_metadata_total"),
        "corpus_chunk_avg_chars": corpus.get("chunk_chars_avg"),
        "router_intent_accuracy": router.get("intent_accuracy"),
        "router_refusal_accuracy": router.get("refusal_accuracy") or answer.get("refusal_accuracy"),
        "retrieval_doc_hit_at_k": retrieval.get("doc_hit_at_k", retrieval.get("doc_hit_at_5")),
        "retrieval_article_hit_at_k": retrieval.get("article_hit_at_k", retrieval.get("article_hit_at_5")),
        "retrieval_clause_hit_at_k": retrieval.get("clause_hit_at_k", retrieval.get("clause_hit_at_5")),
        "retrieval_mrr": retrieval.get("mrr"),
        "generation_fact_coverage": answer.get("fact_coverage"),
        "generation_forbidden_fact_rate": answer.get("forbidden_fact_rate"),
        "generation_display_citation_valid": answer.get("display_citation_valid"),
        "generation_citation_url_valid": answer.get("citation_url_valid", answer.get("source_mapping_valid")),
        "e2e_grounded_answer_rate": answer.get("grounded_answer_rate"),
        "e2e_unsupported_claim_rate": answer.get("unsupported_claim_rate"),
        "reliability_error_rate": answer.get("error_rate"),
    }
    gates: list[dict[str, Any]] = []
    for name, rule in QUALITY_GATES.items():
        op, threshold = rule
        gates.append(
            {
                "gate": name,
                "value": values.get(name),
                "rule": f"{op} {threshold}",
                "status": gate_status(values.get(name), rule),
            }
        )
    return gates


def merge_failure_results(*summaries: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_case: dict[str, dict[str, Any]] = defaultdict(lambda: {"id": "", "failure_reasons": []})
    for summary in summaries:
        for row in summary.get("results", []):
            case_id = row.get("id")
            if not case_id:
                continue
            by_case[case_id]["id"] = case_id
            by_case[case_id]["failure_reasons"].extend(row.get("failure_reasons", []))
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case_id, row in by_case.items():
        unique = sorted(set(row["failure_reasons"]))
        if not unique:
            continue
        for reason in unique:
            categories[reason].append({"id": case_id})
    return dict(categories)


def load_ablation_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return {"error": f"Could not parse {path}"}


def llm_judge_prediction(case: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    """Judge one answer with Gemini using a small rubric. Opt-in only."""
    sys.path.insert(0, str(ROOT))
    from src.utils.llm_factory import get_model_with_fallback, parse_json_response

    context = docs_context_text(prediction)[:6000]
    expected = case.get("expected", {})
    prompt = f"""
You are evaluating a Vietnamese legal QA RAG answer. Return JSON only.

Score each dimension from 1 to 5:
- faithfulness: answer is supported by provided sources.
- completeness: answer covers the expected legal facts.
- usefulness: answer is directly helpful to the user.
- citation_support: citations actually support the claims.

Question: {case.get("question", "")}
Expected facts: {json.dumps(expected.get("expected_facts", []), ensure_ascii=False)}
Forbidden facts: {json.dumps(expected.get("forbidden_facts", []), ensure_ascii=False)}
Answer: {prediction.get("answer", "")}
Sources: {context}

Return:
{{
  "faithfulness": <1-5>,
  "completeness": <1-5>,
  "usefulness": <1-5>,
  "citation_support": <1-5>,
  "notes": "<short reason>"
}}
"""
    llm = get_model_with_fallback(purpose="evaluation_judge", json_mode=True)
    response = llm.invoke(prompt)
    result = parse_json_response(response.content)
    return {
        "faithfulness": int(result.get("faithfulness", 0) or 0),
        "completeness": int(result.get("completeness", 0) or 0),
        "usefulness": int(result.get("usefulness", 0) or 0),
        "citation_support": int(result.get("citation_support", 0) or 0),
        "notes": str(result.get("notes", "")),
    }


def apply_llm_judge(
    cases: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    limit: int | None,
) -> dict[str, dict[str, Any]]:
    judged: dict[str, dict[str, Any]] = {}
    count = 0
    for case in cases:
        prediction = dict(predictions.get(case["id"], {}))
        if not prediction:
            continue
        if limit is not None and count >= limit:
            judged[case["id"]] = prediction
            continue
        try:
            prediction["llm_judge"] = llm_judge_prediction(case, prediction)
        except Exception as exc:
            prediction["llm_judge"] = {"error": str(exc)}
        judged[case["id"]] = prediction
        count += 1
    return {**predictions, **judged}


def empty_summary() -> dict[str, Any]:
    return {"cases": 0, "results": []}


def filter_cases_for_predictions(
    cases: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    *,
    only_predicted: bool,
) -> list[dict[str, Any]]:
    if not only_predicted:
        return cases
    return [case for case in cases if case.get("id") in predictions]


def build_prediction_coverage(
    dataset_cases: list[dict[str, Any]],
    scored_cases: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset_ids = [case["id"] for case in dataset_cases if case.get("id")]
    scored_ids = [case["id"] for case in scored_cases if case.get("id")]
    prediction_ids = [row["id"] for row in prediction_rows if row.get("id")]
    dataset_set = set(dataset_ids)
    prediction_set = set(prediction_ids)
    coverage = len(scored_ids) / len(dataset_ids) if dataset_ids else None
    return {
        "dataset_cases": len(dataset_ids),
        "prediction_rows": len(prediction_rows),
        "predicted_cases": len(prediction_set),
        "scored_cases": len(scored_ids),
        "coverage": coverage,
        "missing_case_count": len(dataset_set - prediction_set),
        "extra_prediction_count": len(prediction_set - dataset_set),
        "missing_case_ids_sample": sorted(dataset_set - prediction_set)[:20],
        "extra_prediction_ids_sample": sorted(prediction_set - dataset_set)[:20],
    }


def build_baseline_report(dataset: Path, baseline_path: Path) -> dict[str, Any]:
    dataset_cases = load_jsonl(dataset) if dataset.exists() else []
    baseline = load_json(baseline_path)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": repo_path(dataset),
        "dataset_cases": len(dataset_cases),
        "mode": "baseline_snapshot",
        "notes": baseline.get("notes", []),
        "corpus": build_corpus_summary(baseline=baseline),
        "router_summary": baseline.get("router_summary", {}),
        "retrieval_summary": baseline.get("retrieval_summary", {}),
        "grader_summary": baseline.get("grader_summary", {}),
        "answer_summary": baseline.get("answer_summary", {}),
        "ablation_summary": load_ablation_report(DEFAULT_ABLATION),
    }
    report["quality_gates"] = build_quality_gates(report)
    report["failure_categories"] = {}
    return report


def build_scored_report(
    dataset: Path,
    predictions_path: Path,
    top_k: int,
    *,
    llm_judge: bool = False,
    llm_judge_limit: int | None = None,
    only_predicted: bool = False,
    component: str = "all",
) -> dict[str, Any]:
    cases = load_jsonl(dataset)
    prediction_rows = load_jsonl(predictions_path)
    predictions = {row["id"]: row for row in prediction_rows if row.get("id")}
    scored_cases = filter_cases_for_predictions(cases, predictions, only_predicted=only_predicted)
    if llm_judge and component != "retrieval":
        predictions = apply_llm_judge(scored_cases, predictions, llm_judge_limit)
    baseline = load_json(DEFAULT_BASELINE)
    retrieval_summary = score_retrieval(scored_cases, predictions, top_k)
    if component == "retrieval":
        router_summary = empty_summary()
        grader_summary = empty_summary()
        answer_summary = empty_summary()
    else:
        router_summary = score_router(scored_cases, predictions)
        grader_summary = score_grader(scored_cases, predictions)
        answer_summary = score_answers(scored_cases, predictions)

    coverage = build_prediction_coverage(cases, scored_cases, prediction_rows)
    quota_count = sum(
        1
        for prediction in predictions.values()
        if prediction_error_category(prediction) == "quota_or_rate_limit"
    )
    notes = [
        f"Predictions scored from {predictions_path}.",
        "Deterministic metrics are always reported; LLM-as-judge is opt-in.",
    ]
    if only_predicted:
        notes.append("Only cases present in the predictions file were scored; missing dataset cases are reported as coverage, not failures.")
    if component == "retrieval":
        notes.append("Retrieval component mode skips router, grader, generation, and E2E answer metrics.")
    if quota_count:
        notes.append(f"{quota_count} prediction(s) contain quota/rate-limit errors; these are tracked separately from model quality errors.")

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": repo_path(dataset),
        "dataset_cases": len(cases),
        "mode": "scored_predictions" if component == "all" else f"scored_{component}_predictions",
        "component": component,
        "predictions": repo_path(predictions_path),
        "prediction_coverage": coverage,
        "quota_or_rate_limit_predictions": quota_count,
        "notes": notes,
        "corpus": build_corpus_summary(baseline=baseline),
        "router_summary": router_summary,
        "retrieval_summary": retrieval_summary,
        "grader_summary": grader_summary,
        "answer_summary": answer_summary,
        "ablation_summary": load_ablation_report(DEFAULT_ABLATION),
    }
    report["quality_gates"] = build_quality_gates(report)
    report["failure_categories"] = merge_failure_results(router_summary, retrieval_summary, grader_summary, answer_summary)
    return report


def markdown_table(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    header = rows[0]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def write_report(report: dict[str, Any], out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Legal QA Evaluation Report",
        "",
        f"- Created at: {report['created_at']}",
        f"- Dataset: `{report['dataset']}` ({report.get('dataset_cases', 'n/a')} cases)",
        f"- Mode: {report['mode']}",
    ]
    if report.get("predictions"):
        lines.append(f"- Predictions: `{report['predictions']}`")
    if report.get("component"):
        lines.append(f"- Component: `{report['component']}`")
    coverage = report.get("prediction_coverage")
    if coverage:
        lines.append(
            "- Prediction coverage: "
            f"{coverage.get('scored_cases', 'n/a')}/{coverage.get('dataset_cases', 'n/a')} dataset cases "
            f"({fmt_percent(coverage.get('coverage'))})"
        )
        lines.append(
            "- Prediction rows: "
            f"{coverage.get('prediction_rows', 'n/a')} rows, "
            f"{coverage.get('predicted_cases', 'n/a')} unique case IDs"
        )
    if report.get("quota_or_rate_limit_predictions"):
        lines.append(f"- Quota/rate-limit predictions: {report['quota_or_rate_limit_predictions']}")
    lines.append("")

    lines.extend(["## Quality Gates", ""])
    gate_rows = [["Gate", "Value", "Rule", "Status"]]
    for gate in report.get("quality_gates", []):
        value = gate.get("value")
        gate_rows.append([
            gate["gate"],
            fmt_percent(value) if isinstance(value, float) and "chars" not in gate["gate"] and "metadata" not in gate["gate"] else str(value if value is not None else "n/a"),
            gate["rule"],
            gate["status"],
        ])
    lines.extend(markdown_table(gate_rows))
    lines.append("")

    corpus = report.get("corpus", {})
    lines.extend(
        [
            "## Corpus Quality",
            "",
            f"- Source: `{corpus.get('source', 'n/a')}`",
            f"- Registry documents: {corpus.get('registry_documents', 'n/a')}",
            f"- Chunks: {corpus.get('chunks', 'n/a')}",
            f"- Chunk chars: min={corpus.get('chunk_chars_min', 'n/a')}, avg={corpus.get('chunk_chars_avg', 'n/a')}, max={corpus.get('chunk_chars_max', 'n/a')}",
            f"- Missing metadata total: {corpus.get('missing_metadata_total', 'n/a')}",
            f"- Levels: `{corpus.get('levels', {})}`",
            "",
        ]
    )

    router = report.get("router_summary", {})
    if router:
        lines.extend(
            [
                "## Router And Agent Decisions",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Router cases | {router.get('cases', 'n/a')} |",
                f"| Intent accuracy | {fmt_percent(router.get('intent_accuracy'))} |",
                f"| Refusal/out-of-scope accuracy | {fmt_percent(router.get('refusal_accuracy'))} |",
                f"| Grader accuracy | {fmt_percent(report.get('grader_summary', {}).get('grader_accuracy'))} |",
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
            f"| P95 retrieval latency | {fmt_number(retrieval.get('p95_retrieval_ms'))} ms |",
            "",
        ]
    )

    answer = report.get("answer_summary", {})
    lines.extend(
        [
            "## Answer And E2E Summary",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Cases | {answer.get('cases', 'n/a')} |",
            f"| Fact Coverage | {fmt_percent(answer.get('fact_coverage'))} |",
            f"| Forbidden Fact Rate | {fmt_percent(answer.get('forbidden_fact_rate'))} |",
            f"| Display Citation Valid | {fmt_percent(answer.get('display_citation_valid'))} |",
            f"| Citation URL Valid | {fmt_percent(answer.get('citation_url_valid', answer.get('source_mapping_valid')))} |",
            f"| Unsupported Claim Rate | {fmt_percent(answer.get('unsupported_claim_rate'))} |",
            f"| Refusal Accuracy | {fmt_percent(answer.get('refusal_accuracy'))} |",
            f"| Grounded Answer Rate | {fmt_percent(answer.get('grounded_answer_rate'))} |",
            f"| Web Fallback Rate | {fmt_percent(answer.get('web_fallback_rate'))} |",
            f"| Error Rate | {fmt_percent(answer.get('error_rate'))} |",
            f"| Quota/Rate-limit Error Rate | {fmt_percent(answer.get('quota_error_rate'))} |",
            f"| Runtime Error Rate | {fmt_percent(answer.get('runtime_error_rate'))} |",
            f"| Avg processing latency | {fmt_number(answer.get('avg_processing_time_ms'))} ms |",
            f"| P95 processing latency | {fmt_number(answer.get('p95_processing_time_ms'))} ms |",
            f"| Avg generation attempts | {fmt_number(answer.get('avg_generation_attempts'))} |",
            "",
        ]
    )

    llm_judge = answer.get("llm_judge", {})
    if any(value is not None for value in llm_judge.values()):
        lines.extend(
            [
                "## LLM-As-Judge",
                "",
                "| Metric | Avg score / 5 |",
                "| --- | ---: |",
                f"| Faithfulness | {fmt_number(llm_judge.get('faithfulness'))} |",
                f"| Completeness | {fmt_number(llm_judge.get('completeness'))} |",
                f"| Usefulness | {fmt_number(llm_judge.get('usefulness'))} |",
                f"| Citation support | {fmt_number(llm_judge.get('citation_support'))} |",
                "",
            ]
        )

    failures = report.get("failure_categories", {})
    lines.extend(["## Failure Analysis", ""])
    if failures:
        failure_rows = [["Category", "Cases", "Sample IDs"]]
        for category, cases in sorted(failures.items(), key=lambda item: len(item[1]), reverse=True):
            failure_rows.append([category, str(len(cases)), ", ".join(row["id"] for row in cases[:8])])
        lines.extend(markdown_table(failure_rows))
    else:
        lines.append("- No case-level failure data available for this mode.")
    lines.append("")

    ablation = report.get("ablation_summary")
    if ablation:
        lines.extend(["## Ablation Summary", ""])
        if ablation.get("runs"):
            rows = [["Variant", "Cases", "Doc Hit@5", "Article Hit@5", "Clause Hit@5", "Fact Coverage", "Grounded Rate"]]
            for run in ablation["runs"]:
                rows.append(
                    [
                        str(run.get("variant", "n/a")),
                        str(run.get("cases", "n/a")),
                        fmt_percent(run.get("doc_hit_at_5")),
                        fmt_percent(run.get("article_hit_at_5")),
                        fmt_percent(run.get("clause_hit_at_5")),
                        fmt_percent(run.get("fact_coverage")),
                        fmt_percent(run.get("grounded_answer_rate")),
                    ]
                )
            lines.extend(markdown_table(rows))
        else:
            lines.append("- Ablation runs have not been generated yet.")
        lines.append("")

    if report.get("notes"):
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in report["notes"])
        lines.append("")

    lines.extend(
        [
            "## Reproduce",
            "",
            "```bash",
            "python scripts/evaluate_legal_qa.py --dataset data/evaluation/legal_qa_eval_100.jsonl",
            "python scripts/run_retrieval_eval.py --dataset data/evaluation/legal_qa_eval_100.jsonl --out eval_reports/retrieval_predictions.jsonl",
            "python scripts/evaluate_legal_qa.py --dataset data/evaluation/legal_qa_eval_100.jsonl --predictions eval_reports/retrieval_predictions.jsonl --component retrieval --out-json eval_reports/retrieval_100.json --out-md eval_reports/retrieval_100.md",
            "python scripts/run_e2e_eval.py --dataset data/evaluation/legal_qa_eval_e2e_20.jsonl --out eval_reports/e2e_predictions_20.jsonl --skip-existing",
            "python scripts/evaluate_legal_qa.py --dataset data/evaluation/legal_qa_eval_e2e_20.jsonl --predictions eval_reports/e2e_predictions_20.jsonl --component e2e --only-predicted --out-json eval_reports/e2e_20.json --out-md eval_reports/e2e_20.md",
            "python scripts/compare_ablation_runs.py --dataset data/evaluation/legal_qa_eval_100.jsonl --run dense=eval_reports/dense_predictions.jsonl --run sparse=eval_reports/sparse_predictions.jsonl --run hybrid=eval_reports/hybrid_predictions.jsonl --run full_graph=eval_reports/e2e_predictions.jsonl",
            "```",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def write_failure_analysis(report: dict[str, Any], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Legal QA Failure Analysis",
        "",
        f"- Created at: {report['created_at']}",
        f"- Dataset: `{report['dataset']}`",
        f"- Mode: {report['mode']}",
        "",
    ]
    failures = report.get("failure_categories", {})
    if not failures:
        lines.append("No case-level failure data is available. Run the evaluator with `--predictions` to populate this report.")
    else:
        lines.extend(["## Top Failure Categories", ""])
        rows = [["Category", "Count", "Case IDs"]]
        for category, cases in sorted(failures.items(), key=lambda item: len(item[1]), reverse=True):
            rows.append([category, str(len(cases)), ", ".join(row["id"] for row in cases[:20])])
        lines.extend(markdown_table(rows))
        lines.extend(
            [
                "",
                "## How To Use",
                "",
                "- `wrong_intent`: inspect router prompt and intent labels.",
                "- `wrong_doc`, `wrong_article`, `wrong_clause`: inspect retrieval filters, chunk metadata, and ranking.",
                "- `missing_fact`: inspect generator prompt and retrieved context coverage.",
                "- `unsupported_claim`: inspect hallucination grader and citation grounding.",
                "- `refusal_error`: inspect out-of-scope and unsafe request handling.",
                "- `quota_or_rate_limit`: rerun later with `--skip-existing`; do not treat this as model quality failure.",
                "- `runtime_error`: inspect service dependencies, Qdrant, graph exceptions, and API/runtime logs.",
            ]
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def resolve_dataset(path: Path) -> Path:
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Vietnamese legal QA predictions.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--failure-md", type=Path, default=DEFAULT_FAILURE_MD)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--component",
        choices=["all", "retrieval", "e2e"],
        default="all",
        help="Score all metrics, retrieval-only metrics, or E2E predictions.",
    )
    parser.add_argument(
        "--only-predicted",
        action="store_true",
        help="Score only dataset cases present in the predictions file.",
    )
    parser.add_argument("--llm-judge", action="store_true", help="Run optional Gemini judge for predictions.")
    parser.add_argument("--llm-judge-limit", type=int, default=None)
    args = parser.parse_args()

    dataset = resolve_dataset(args.dataset)
    if args.predictions:
        report = build_scored_report(
            dataset,
            args.predictions,
            args.top_k,
            llm_judge=args.llm_judge,
            llm_judge_limit=args.llm_judge_limit,
            only_predicted=args.only_predicted,
            component=args.component,
        )
    else:
        report = build_baseline_report(dataset, args.baseline)

    write_report(report, args.out_json, args.out_md)
    write_failure_analysis(report, args.failure_md)
    print(f"Saved JSON report: {args.out_json}")
    print(f"Saved Markdown report: {args.out_md}")
    print(f"Saved failure analysis: {args.failure_md}")


if __name__ == "__main__":
    main()
