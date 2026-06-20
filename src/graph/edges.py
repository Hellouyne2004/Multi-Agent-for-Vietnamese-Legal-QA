"""
src/graph/edges.py
Định nghĩa logic điều kiện (conditional edges) để định tuyến giữa các node trong LangGraph.
"""
from typing import Literal
from langgraph.graph import END
from src.graph.state import GraphState
from src.utils.logger import logger
from src.config import MAX_RETRIES


def decide_to_retrieve(state: GraphState) -> Literal["retriever", "web_searcher", END]:
    """
    Ưu tiên policy action từ router để chọn corpus, web hoặc dừng workflow.
    
    Args:
        state: Trạng thái hiện tại của graph
        
    Returns:
        Tên node tiếp theo hoặc END
    """
    route_action = state.get("route_action")
    intent = state.get("intent")

    if route_action == "retrieve":
        logger.info("[EDGE] route_action=retrieve. Routing to retriever.")
        return "retriever"

    if route_action == "web_required":
        logger.info("[EDGE] route_action=web_required. Routing to web_searcher.")
        return "web_searcher"

    if route_action in {
        "redirect_out_of_scope",
        "respond_chat",
        "refuse_unsafe",
        "refuse_unsupported",
        "router_error",
    }:
        logger.info("[EDGE] route_action='{}' stops the RAG flow.", route_action)
        return END

    # Backward compatibility for old saved states that predate route_action.
    if route_action is None and intent in ["legal_query", "procedural"]:
        logger.warning(
            "[EDGE] Missing route_action; falling back to legacy intent='{}'.",
            intent,
        )
        return "retriever"

    logger.warning(
        "[EDGE] Invalid or missing route_action='{}' for intent='{}'. Routing to END.",
        route_action,
        intent,
    )
    return END


def grade_documents(state: GraphState) -> Literal["web_searcher", "generator"]:
    """
    Dựa trên kết quả đánh giá của grader để chọn generator hay web_searcher (fallback).
    
    Args:
        state: Trạng thái hiện tại của graph
        
    Returns:
        "generator" nếu tài liệu đủ, ngược lại "web_searcher"
    """
    verdict = state.get("grader_verdict")
    
    if verdict == "yes":
        logger.info("[EDGE] Documents are relevant. Routing to generator.")
        return "generator"
    
    logger.info("[EDGE] Documents are NOT relevant. Routing to web_searcher fallback.")
    return "web_searcher"


def check_hallucination(state: GraphState) -> Literal["generator", END]:
    """
    Kiểm tra kết quả của hallucination grader hoặc lỗi từ generator.
    Nếu fail hoặc gặp lỗi parse và chưa quá số lần retry -> generate lại.
    Nếu pass hoặc đã quá số lần retry -> kết thúc.
    """
    verdict = state.get("hallucination_verdict")
    attempt = state.get("generation_attempt", 0)
    error = state.get("error", "")
    
    # 1. Nếu có lỗi ParseError từ generator -> Ưu tiên retry
    if error and "ParseError" in str(error):
        if attempt < MAX_RETRIES:
            logger.warning(f"[EDGE] Generator encountered ParseError (attempt {attempt}/{MAX_RETRIES}). Retrying generation...")
            return "generator"
        else:
            logger.error(f"[EDGE] Max retries reached for ParseError ({MAX_RETRIES}). Routing to END.")
            return END

    # 2. Nếu grader cho qua -> END
    if verdict == "pass":
        logger.info("[EDGE] No hallucinations detected. Routing to END.")
        return END
    
    # 3. Nếu grader báo lỗi ảo giác -> Kiểm tra số lần retry
    if attempt < MAX_RETRIES:
        logger.warning(f"[EDGE] Hallucinations detected (attempt {attempt}/{MAX_RETRIES}). Retrying generation...")
        return "generator"
    
    logger.error(f"[EDGE] Hallucinations detected but max retries reached ({MAX_RETRIES}). Routing to END.")
    return END

