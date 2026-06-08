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
