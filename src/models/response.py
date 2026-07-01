"""
src/models/response.py
Pydantic models for API responses.
"""
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class Citation(BaseModel):
    """Citation model for referencing legal documents."""
    text: str = Field(..., description="The segment of text being cited.")
    source: str = Field(..., description="The name or reference of the document source.")
    position: int = Field(..., description="The position in the answer where the citation occurs.")


class WebResult(BaseModel):
    """Web search result model."""
    url: str
    title: str
    content: str
    source_type: str = "web"


class AgentEvent(BaseModel):
    """Compact trace event for one graph node execution."""
    trace_id: Optional[str] = None
    step: Optional[int] = None
    agent: Optional[str] = None
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    chunk_ids: List[str] = Field(default_factory=list)
    scores: Dict[str, float] = Field(default_factory=dict)
    latency_ms: int = 0
    error: Optional[str] = None
    created_at: Optional[str] = None


class AnswerResponse(BaseModel):
    """Response model for a legal question."""
    trace_id: Optional[str] = None
    question: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    web_results: List[WebResult] = Field(default_factory=list)
    agent_events: List[AgentEvent] = Field(default_factory=list)
    confidence: float = 0.0
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    route_action: Optional[str] = None
    route_confidence: Optional[float] = None
    generation_attempt: int = 1
    processing_time_ms: int = 0
    error: Optional[str] = None


class StreamEvent(BaseModel):
    """Container for SSE events sent during graph execution."""
    event: str  # 'status', 'data', 'error', 'end'
    node: Optional[str] = None
    message: Optional[str] = None
    data: Optional[Any] = None
