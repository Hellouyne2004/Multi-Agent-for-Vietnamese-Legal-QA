"""Build a curated E2E benchmark from the public 100-case seed.

The output is intentionally small enough for intern/fresher portfolio demos but
wide enough to exercise grounded QA, table lookup, refusal, unsafe handling,
unsupported questions, hallucination traps, and web-required routing.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "evaluation" / "legal_qa_eval_100.jsonl"
DEFAULT_OUT = ROOT / "data" / "evaluation" / "legal_qa_eval_e2e_40.jsonl"

CURATED_SOURCE_IDS = [
    "labor_working_time_001",
    "labor_holidays_004",
    "labor_contract_content_018",
    "labor_termination_cases_023",
    "labor_overtime_pay_028",
    "labor_discipline_rules_034",
    "labor_dismissal_037",
    "labor_retirement_age_040",
    "labor_marriage_leave_041",
    "labor_termination_payment_047",
    "labor_discipline_time_limit_049",
    "labor_relationship_principles_054",
    "land_price_dong_khoi_058",
    "land_price_le_duan_059",
    "tax_exempt_income_061",
    "tax_effective_date_062",
    "tax_taxpayer_064",
    "tax_taxable_income_065",
    "tax_resident_period_070",
    "tax_family_deduction_071",
    "cyber_responsibility_073",
    "cyber_system_level_075",
    "cyber_scope_076",
    "cyber_effective_terms_079",
    "cyber_personal_use_080",
    "cyber_no_weather_082",
    "land_price_table_083",
    "land_price_cao_ba_quat_084",
    "land_price_compare_089",
    "land_price_table_scope_090",
    "out_of_scope_weather_091",
    "out_of_scope_medical_092",
    "out_of_scope_stock_094",
    "general_chat_095",
    "unsafe_evade_tax_096",
    "unsafe_evade_labor_097",
    "unsupported_fake_article_098",
    "unsupported_table_missing_100",
]

EXTRA_WEB_REQUIRED_CASES = [
    {
        "id": "web_current_minimum_wage_101",
        "category": "current_law",
        "type": "web_required",
        "answer_policy": "grounded_answer",
        "question": "Muc luong toi thieu vung hien nay moi nhat la bao nhieu?",
        "expected": {
            "doc_id": None,
            "level": None,
            "expected_facts": ["muc luong toi thieu vung"],
            "forbidden_facts": ["tra loi nhu can cu chac chan tu corpus noi bo khi chua kiem chung"],
            "expected_intent": "legal_query",
            "expected_route_action": "web_required",
        },
        "requires_web": True,
        "difficulty": "medium",
    },
    {
        "id": "web_latest_tax_change_102",
        "category": "current_law",
        "type": "web_required",
        "answer_policy": "grounded_answer",
        "question": "Hom nay co thay doi moi nao ve thue thu nhap ca nhan can kiem tra khong?",
        "expected": {
            "doc_id": None,
            "level": None,
            "expected_facts": ["thue thu nhap ca nhan"],
            "forbidden_facts": ["khong can kiem tra thong tin hien hanh"],
            "expected_intent": "legal_query",
            "expected_route_action": "web_required",
        },
        "requires_web": True,
        "difficulty": "medium",
    },
]


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_benchmark(source: Path) -> list[dict[str, Any]]:
    rows_by_id = {row["id"]: row for row in load_jsonl(source)}
    missing = [case_id for case_id in CURATED_SOURCE_IDS if case_id not in rows_by_id]
    if missing:
        raise ValueError(f"Missing source case IDs: {', '.join(missing)}")
    rows = [rows_by_id[case_id] for case_id in CURATED_SOURCE_IDS]
    rows.extend(EXTRA_WEB_REQUIRED_CASES)
    if len(rows) != 40:
        raise ValueError(f"Expected 40 E2E cases, got {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the curated E2E-40 benchmark.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = build_benchmark(args.source)
    write_jsonl(args.out, rows)

    categories = Counter(row.get("category") for row in rows)
    types = Counter(row.get("type") for row in rows)
    print(f"Wrote {len(rows)} cases to {args.out}")
    print(f"Categories: {dict(sorted(categories.items()))}")
    print(f"Types: {dict(sorted(types.items()))}")


if __name__ == "__main__":
    main()
