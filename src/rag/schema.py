# src/rag/schema.py
from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class OptionForecast(BaseModel):
    """
    Your model's output for a specific option/expiry.
    Extend as needed (delta/gamma, prob_up, etc.).
    """
    ticker: str
    expiry: str                 # ISO date (YYYY-MM-DD)
    strike: float
    option_type: Literal["CALL", "PUT"]
    model_price: float
    iv: Optional[float] = None
    spot: Optional[float] = None
    source: Optional[str] = None   # e.g., "black-scholes-v1"


class ResearchChunk(BaseModel):
    """
    Normalized view of a retrieved chunk (for prompts & UI).
    """
    text: str
    source: str
    page: Optional[int] = None
    ticker: Optional[str] = None
    source_type: Optional[str] = None
    asof_date: Optional[str] = None


class Thesis(BaseModel):
    """
    Final structured output for the UI card.
    """
    ticker: str
    one_line_view: str = Field(..., description="Concise, plain-English thesis in one line.")
    view_confidence_10pt: int = Field(..., ge=1, le=10)
    key_drivers: List[str]
    risks: List[str]
    suggested_actions: List[str]
    references: List[str] = Field(default_factory=list)  # ["file.pdf:12", ...]


class QueryRequest(BaseModel):
    ticker: str
    question: str
    top_k: int = 5
    use_mmr: bool = True
    bm25_weight: float = 0.5
    forecasts: List[OptionForecast] = Field(default_factory=list)


class QueryResponse(BaseModel):
    thesis: Thesis
    retrieved: List[ResearchChunk] = Field(default_factory=list)


def to_ref_string(source: str, page: Optional[int]) -> str:
    return f"{source}:{page}" if page else source
