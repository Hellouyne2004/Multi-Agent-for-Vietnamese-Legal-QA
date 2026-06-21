"""Build a manually adjudicated Grader benchmark from frozen retrieval contexts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "data" / "evaluation" / "legal_qa_eval_100.jsonl"
DEFAULT_RETRIEVAL = ROOT / "eval_reports" / "retrieval_predictions.jsonl"
DEFAULT_OUT = ROOT / "data" / "evaluation" / "grader_eval_20.jsonl"
BENCHMARK_VERSION = "grader-holdout-20-v1.0"
CONTEXT_VERSION = "retrieval-100-2026-06-16-grader-view-500"


ACTUAL_SPECS = [
    ("grader_yes_001", "labor_working_time_001", "yes", "relevant_sufficient", [], "Article 105 context contains both daily and weekly limits."),
    ("grader_yes_002", "labor_holidays_004", "yes", "relevant_sufficient", [], "The frozen context contains all three core paid-holiday facts."),
    ("grader_yes_003", "labor_strike_definition_015", "yes", "mixed_relevant_irrelevant", [], "Relevant expanded context contains all three definition facts despite distractors."),
    ("grader_yes_004", "labor_notice_indefinite_044", "yes", "procedure_sufficient", [], "The notice-period fact required for the procedure is present."),
    ("grader_yes_005", "land_price_table_083", "yes", "table_sufficient", [], "The requested table row and value are preserved in the frozen context."),
    ("grader_yes_006", "land_price_compare_089", "yes", "table_sufficient", [], "Both requested table rows and values are present for comparison."),
    ("grader_yes_007", "tax_taxpayer_064", "yes", "relevant_sufficient", [], "Both taxpayer-scope facts are visible to the Grader."),
    ("grader_yes_008", "tax_effective_date_062", "yes", "ocr_sufficient", [], "The effective-date fact is present despite OCR noise."),
    ("grader_yes_009", "cyber_transition_074", "yes", "mixed_relevant_irrelevant", [], "Expanded context contains both transitional-provision facts."),
    ("grader_yes_010", "cyber_personal_use_080", "yes", "relevant_sufficient", [], "Both personal-use obligations are present in the frozen context."),
    ("grader_no_011", "labor_contract_content_018", "no", "relevant_incomplete", ["công việc và địa điểm làm việc", "mức lương"], "Related contract articles are retrieved, but the core Article 21 details are incomplete in the Grader-visible text."),
    ("grader_no_012", "labor_employer_obligations_053", "no", "relevant_incomplete", ["thực hiện hợp đồng lao động", "tôn trọng danh dự, nhân phẩm của người lao động", "đối thoại, trao đổi với người lao động"], "Retrieved articles discuss employers but omit the three general obligation facts."),
    ("grader_no_013", "labor_wage_scale_027", "no", "wrong_document", ["xây dựng thang lương, bảng lương", "định mức lao động"], "All retrieved documents are land-price appendix tables, not labor law."),
    ("grader_no_014", "cyber_transition_previous_law_078", "no", "empty_context", ["Luật An toàn thông tin mạng số 86/2015/QH13"], "Retrieval returned no documents."),
    ("grader_no_015", "tax_non_resident_072", "no", "relevant_incomplete", ["thu nhập chịu thuế phát sinh trong lãnh thổ Việt Nam"], "The context contains non-resident tax articles but not the requested general territorial rule."),
    ("grader_no_016", "labor_contract_types_008", "no", "relevant_incomplete", ["hợp đồng lao động không xác định thời hạn", "hợp đồng lao động xác định thời hạn"], "Related contract articles are present, but Article 20 and both contract types are missing."),
]


CONTROLLED_SPECS = [
    {
        "id": "grader_no_017",
        "case_id": "labor_working_time_001",
        "docs_id": "labor_working_time_001",
        "transform": "expired",
        "slice": "wrong_expired_version",
        "missing_facts": [],
        "gold_reason": "The facts are present, but the only relevant legal context is explicitly marked expired.",
    },
    {
        "id": "grader_no_018",
        "case_id": "labor_working_time_001",
        "docs_id": "land_price_table_083",
        "transform": "irrelevant",
        "slice": "irrelevant",
        "missing_facts": ["không quá 08 giờ trong 01 ngày", "không quá 48 giờ trong 01 tuần"],
        "gold_reason": "The question asks about working time, while every context item is a land-price table.",
    },
    {
        "id": "grader_no_019",
        "case_id": "labor_holidays_004",
        "docs_id": "labor_holidays_004",
        "transform": "partial",
        "slice": "relevant_incomplete",
        "missing_facts": ["Tết Âm lịch", "Quốc khánh"],
        "gold_reason": "A controlled relevant fragment contains only New Year's Day and omits two core holiday facts.",
    },
    {
        "id": "grader_no_020",
        "case_id": "labor_working_time_001",
        "docs_id": "labor_working_time_001",
        "transform": "contradictory",
        "slice": "mixed_contradictory",
        "missing_facts": [],
        "gold_reason": "Correct limits are mixed with a directly conflicting 60-hour weekly limit.",
    },
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def freeze_document(document: dict[str, Any]) -> dict[str, Any]:
    metadata = document.get("metadata", {})
    frozen_metadata = {
        key: metadata.get(key)
        for key in [
            "chunk_id",
            "doc_id",
            "ten_van_ban",
            "so_hieu_van_ban",
            "dieu",
            "khoang",
            "trang_thai",
            "ngay_hieu_luc",
            "page_start",
            "page_end",
            "level",
        ]
    }
    return {
        "content": str(document.get("content", ""))[:500],
        "metadata": frozen_metadata,
        "score": float(document.get("score") or 0.0),
    }


def transform_documents(
    transform: str,
    documents: list[dict[str, Any]],
    case: dict[str, Any],
) -> list[dict[str, Any]]:
    documents = copy.deepcopy(documents)
    if transform == "irrelevant":
        return documents
    if transform == "expired":
        for document in documents:
            document["metadata"]["trang_thai"] = "Hết hiệu lực"
        if documents:
            documents[0]["content"] = (
                "Trạng thái văn bản: Hết hiệu lực. " + documents[0]["content"]
            )[:500]
        return documents
    if transform == "partial":
        first_fact = case["expected"]["expected_facts"][0]
        metadata = documents[0]["metadata"] if documents else {}
        return [{
            "content": f"Trích đoạn liên quan nhưng chưa đầy đủ: {first_fact}.",
            "metadata": metadata,
            "score": 0.95,
        }]
    if transform == "contradictory":
        conflict = {
            "content": "Nguồn xung đột: thời giờ làm việc bình thường có thể lên đến 60 giờ trong 01 tuần.",
            "metadata": {
                "chunk_id": "controlled_conflict_working_time",
                "doc_id": "controlled_conflict",
                "ten_van_ban": "Nguồn pháp lý không xác minh",
                "dieu": "Quy định xung đột",
                "trang_thai": "Không xác định",
            },
            "score": 0.99,
        }
        return [conflict, *documents]
    raise ValueError(f"Unknown transform: {transform}")


def context_hash(documents: list[dict[str, Any]]) -> str:
    payload = json.dumps(documents, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_row(
    *,
    benchmark_id: str,
    case: dict[str, Any],
    documents: list[dict[str, Any]],
    gold: str,
    slice_name: str,
    missing_facts: list[str],
    reason: str,
    origin: str,
) -> dict[str, Any]:
    expected = case.get("expected", {})
    return {
        "id": benchmark_id,
        "benchmark_version": BENCHMARK_VERSION,
        "context_version": CONTEXT_VERSION,
        "source_case_id": case["id"],
        "context_origin": origin,
        "slice": slice_name,
        "category": case.get("category"),
        "difficulty": case.get("difficulty"),
        "question": case["question"],
        "expected_facts": expected.get("expected_facts", []),
        "forbidden_facts": expected.get("forbidden_facts", []),
        "context_sufficient": gold,
        "missing_facts": missing_facts,
        "gold_reason": reason,
        "annotation": {
            "protocol_version": "grader-gold-v1.0",
            "annotator_count": 1,
            "agreement_status": "N/A_SINGLE_ANNOTATOR",
        },
        "context_hash": context_hash(documents),
        "grader_documents": documents,
    }


def build(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases = {row["id"]: row for row in load_jsonl(args.cases)}
    predictions = {row["id"]: row for row in load_jsonl(args.retrieval_predictions)}
    rows: list[dict[str, Any]] = []

    for benchmark_id, case_id, gold, slice_name, missing, reason in ACTUAL_SPECS:
        case = cases[case_id]
        documents = [
            freeze_document(document)
            for document in predictions[case_id].get("retrieved_documents", [])
        ]
        rows.append(build_row(
            benchmark_id=benchmark_id,
            case=case,
            documents=documents,
            gold=gold,
            slice_name=slice_name,
            missing_facts=missing,
            reason=reason,
            origin="retrieval_snapshot",
        ))

    for spec in CONTROLLED_SPECS:
        case = cases[spec["case_id"]]
        source_documents = [
            freeze_document(document)
            for document in predictions[spec["docs_id"]].get("retrieved_documents", [])
        ]
        documents = transform_documents(spec["transform"], source_documents, case)
        rows.append(build_row(
            benchmark_id=spec["id"],
            case=case,
            documents=documents,
            gold="no",
            slice_name=spec["slice"],
            missing_facts=spec["missing_facts"],
            reason=spec["gold_reason"],
            origin=f"controlled_{spec['transform']}",
        ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--retrieval-predictions", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    rows = build(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} frozen Grader cases: {args.out}")


if __name__ == "__main__":
    main()
