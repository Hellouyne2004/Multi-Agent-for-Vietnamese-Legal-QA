import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_rule_based_hallucination_check_rejects_missing_citation():
    pytest.importorskip("langchain_core")
    from src.agents.hallucination_grader import _rule_based_hallucination_check

    docs = [
        {
            "content": "Thời giờ làm việc bình thường không quá 08 giờ trong 01 ngày.",
            "metadata": {"doc_id": "45_2019_qh14", "source_url": "https://example.test/law"},
        }
    ]

    reason = _rule_based_hallucination_check("Người lao động làm không quá 08 giờ mỗi ngày.", docs)

    assert "không có citation" in reason


def test_rule_based_hallucination_check_accepts_grounded_answer():
    pytest.importorskip("langchain_core")
    from src.agents.hallucination_grader import _rule_based_hallucination_check

    docs = [
        {
            "content": "Thời giờ làm việc bình thường không quá 08 giờ trong 01 ngày.",
            "metadata": {"doc_id": "45_2019_qh14", "source_url": "https://example.test/law"},
        }
    ]
    citations = [{"source": "[S1]", "text": "Điều 105", "url": "https://example.test/law"}]

    reason = _rule_based_hallucination_check(
        "Theo tài liệu, thời giờ làm việc bình thường không quá 08 giờ trong 01 ngày [S1].",
        docs,
        citations=citations,
    )

    assert reason == ""


def test_retriever_extracts_article_and_clause_filters():
    pytest.importorskip("qdrant_client")
    from src.agents.retriever import _parse_query_filters

    filters = _parse_query_filters("Theo Điều 105 khoản 1 Bộ luật Lao động thì làm việc tối đa bao lâu?")

    assert filters["article_number"] == 105
    assert filters["clause_number"] == 1


def test_api_root_contract():
    pytest.importorskip("fastapi")
    pytest.importorskip("loguru")
    from api.main import root

    response = asyncio.run(root())

    assert response["status"] == "online"
    assert response["service"] == "Vietnamese Legal RAG API"
    assert "timestamp" in response


def test_offline_evaluator_scores_retrieval_and_answer():
    module_path = ROOT / "scripts" / "evaluate_legal_qa.py"
    spec = importlib.util.spec_from_file_location("evaluate_legal_qa", module_path)
    evaluator = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(evaluator)

    cases = [
        {
            "id": "case_1",
            "expected": {
                "doc_id": "45_2019_qh14",
                "article_number": 105,
                "clause_number": 1,
                "level": "clause",
                "expected_facts": ["08 giờ"],
                "forbidden_facts": ["60 giờ"],
            },
        }
    ]
    predictions = {
        "case_1": {
            "retrieved_documents": [
                {
                    "metadata": {
                        "doc_id": "45_2019_qh14",
                        "article_number": 105,
                        "clause_number": 1,
                        "level": "clause",
                    }
                }
            ],
            "answer": "Thời giờ làm việc bình thường không quá 08 giờ trong 01 ngày [S1].",
            "citations": [{"source": "[S1]", "url": "https://example.test/law"}],
        }
    }

    retrieval = evaluator.score_retrieval(cases, predictions, top_k=5)
    answer = evaluator.score_answers(cases, predictions)

    assert retrieval["doc_hit_at_k"] == 1.0
    assert retrieval["article_hit_at_k"] == 1.0
    assert answer["fact_coverage"] == 1.0
    assert answer["display_citation_valid"] == 1.0


def test_eval_100_dataset_schema_is_loadable():
    module_path = ROOT / "scripts" / "evaluate_legal_qa.py"
    spec = importlib.util.spec_from_file_location("evaluate_legal_qa", module_path)
    evaluator = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(evaluator)

    rows = evaluator.load_jsonl(ROOT / "data" / "evaluation" / "legal_qa_eval_100.jsonl")

    assert len(rows) == 100
    for row in rows:
        assert row["id"]
        assert row["question"]
        assert row["answer_policy"]
        assert "expected" in row
        assert "expected_facts" in row["expected"]
        assert "forbidden_facts" in row["expected"]
        assert "expected_intent" in row["expected"]
        assert "requires_web" in row
        assert "difficulty" in row


def test_eval_dataset_matches_indexed_corpus_scope():
    module_path = ROOT / "scripts" / "evaluate_legal_qa.py"
    spec = importlib.util.spec_from_file_location("evaluate_legal_qa", module_path)
    evaluator = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(evaluator)

    registry = evaluator.load_jsonl(ROOT / "data" / "processed" / "document_registry.jsonl")
    indexed_doc_ids = {row["doc_id"] for row in registry}
    rows = evaluator.load_jsonl(ROOT / "data" / "evaluation" / "legal_qa_eval_100.jsonl")

    expected_doc_ids = {
        row["expected"]["doc_id"]
        for row in rows
        if row.get("expected", {}).get("doc_id")
    }
    assert expected_doc_ids.issubset(indexed_doc_ids)
    assert "luat_viec_lam_2025" not in expected_doc_ids


def test_eval_e2e_20_subset_schema_is_loadable():
    module_path = ROOT / "scripts" / "evaluate_legal_qa.py"
    spec = importlib.util.spec_from_file_location("evaluate_legal_qa", module_path)
    evaluator = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(evaluator)

    rows = evaluator.load_jsonl(ROOT / "data" / "evaluation" / "legal_qa_eval_e2e_20.jsonl")

    assert len(rows) == 20
    assert {row["id"] for row in rows}.issubset(
        {row["id"] for row in evaluator.load_jsonl(ROOT / "data" / "evaluation" / "legal_qa_eval_100.jsonl")}
    )
    assert any(row["type"] == "table_lookup" for row in rows)
    assert any(row["category"] == "out_of_scope" for row in rows)
    assert any(row["type"] == "hallucination_trap" for row in rows)


def test_offline_evaluator_scores_router_refusal_and_quality_gates():
    module_path = ROOT / "scripts" / "evaluate_legal_qa.py"
    spec = importlib.util.spec_from_file_location("evaluate_legal_qa", module_path)
    evaluator = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(evaluator)

    cases = [
        {
            "id": "case_legal",
            "answer_policy": "grounded_answer",
            "expected": {
                "doc_id": "45_2019_qh14",
                "expected_intent": "legal_query",
                "expected_facts": ["08 giờ"],
                "forbidden_facts": ["60 giờ"],
            },
        },
        {
            "id": "case_refuse",
            "category": "out_of_scope",
            "answer_policy": "refuse_or_redirect",
            "expected": {
                "doc_id": None,
                "expected_intent": "out_of_scope",
                "expected_facts": ["ngoài phạm vi pháp lý"],
                "forbidden_facts": ["kê đơn thuốc"],
            },
        },
    ]
    predictions = {
        "case_legal": {
            "intent": "legal_query",
            "answer": "Thời giờ làm việc không quá 08 giờ [S1].",
            "citations": [{"source": "[S1]", "url": "https://example.test/law"}],
            "retrieved_documents": [{"content": "không quá 08 giờ", "metadata": {"doc_id": "45_2019_qh14"}}],
            "processing_time_ms": 100,
        },
        "case_refuse": {
            "intent": "out_of_scope",
            "answer": "Câu hỏi này ngoài phạm vi pháp lý, bạn nên hỏi chuyên gia y tế.",
            "citations": [],
            "retrieved_documents": [],
            "processing_time_ms": 50,
        },
    }

    router = evaluator.score_router(cases, predictions)
    answer = evaluator.score_answers(cases, predictions)
    report = {
        "corpus": {"missing_metadata_total": 0, "chunk_chars_avg": 843.4},
        "router_summary": router,
        "retrieval_summary": {"doc_hit_at_k": 1.0, "article_hit_at_k": 0.9, "clause_hit_at_k": 0.8, "mrr": 0.9},
        "answer_summary": answer,
    }
    gates = evaluator.build_quality_gates(report)

    assert router["intent_accuracy"] == 1.0
    assert answer["refusal_accuracy"] == 1.0
    assert answer["grounded_answer_rate"] == 1.0
    assert any(gate["gate"] == "router_intent_accuracy" and gate["status"] == "PASS" for gate in gates)


def test_offline_evaluator_prediction_coverage_only_predicted():
    module_path = ROOT / "scripts" / "evaluate_legal_qa.py"
    spec = importlib.util.spec_from_file_location("evaluate_legal_qa", module_path)
    evaluator = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(evaluator)

    cases = [{"id": "case_1"}, {"id": "case_2"}]
    predictions = {"case_1": {"id": "case_1"}}
    scored = evaluator.filter_cases_for_predictions(cases, predictions, only_predicted=True)
    coverage = evaluator.build_prediction_coverage(cases, scored, [{"id": "case_1"}])

    assert [case["id"] for case in scored] == ["case_1"]
    assert coverage["scored_cases"] == 1
    assert coverage["dataset_cases"] == 2
    assert coverage["coverage"] == 0.5


def test_offline_evaluator_tracks_quota_without_answer_quality_penalty():
    module_path = ROOT / "scripts" / "evaluate_legal_qa.py"
    spec = importlib.util.spec_from_file_location("evaluate_legal_qa", module_path)
    evaluator = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(evaluator)

    cases = [
        {
            "id": "case_quota",
            "expected": {
                "doc_id": "45_2019_qh14",
                "expected_facts": ["08 giờ"],
                "forbidden_facts": [],
            },
        }
    ]
    predictions = {
        "case_quota": {
            "error": "429 RESOURCE_EXHAUSTED: quota exceeded for generate_content_free_tier_requests",
            "answer": "",
        }
    }

    answer = evaluator.score_answers(cases, predictions)

    assert answer["quota_error_rate"] == 1.0
    assert answer["error_rate"] == 0.0
    assert answer["grounded_answer_rate"] is None
    assert answer["fact_coverage"] is None


def test_new_eval_cli_modules_import_without_runtime_services():
    for script_name in ["run_e2e_eval.py", "run_retrieval_eval.py", "compare_ablation_runs.py"]:
        module_path = ROOT / "scripts" / script_name
        spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
