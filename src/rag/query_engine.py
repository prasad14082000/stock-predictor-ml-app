# src/rag/query_engine.py
from __future__ import annotations

import json
import re
from typing import List, Optional, Sequence

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.rag.ollama_runner import generate
from src.rag.chroma_db import similarity_search as chroma_sim, mmr_search as chroma_mmr
from src.rag.schema import (
    QueryRequest,
    OptionForecast,
    ResearchChunk,
    Thesis,
    QueryResponse,
)

# Optional: hybrid keyword + vector retrieval if you have BM25 configured
try:
    from src.rag.hybrid_retriever import hybrid_retrieve as _hybrid_retrieve
    _HAS_HYBRID = True
except Exception:
    _HAS_HYBRID = False


# ---------------------------
# Helpers: mapping & refs
# ---------------------------

def to_ref_string(source: str, page: Optional[int]) -> str:
    """Standardize reference string used both in context headers and thesis.references."""
    if page is None:
        return f"{source}:1"
    return f"{source}:{page}"


def to_research_chunk(doc: Document) -> ResearchChunk:
    """Map a LangChain Document into our typed ResearchChunk."""
    md = doc.metadata or {}
    return ResearchChunk(
        text = doc.page_content or "",
        source=str(md.get("source", "")),
        page=int(md.get("page", 1)) if md.get("page") is not None else 1,
        ticker=str(md.get("ticker", "")),
        source_type=str(md.get("source_type", "")),
        asof_date=md.get("asof_date", None),
    )


def _allowed_refs_from(chunks: Sequence[ResearchChunk]) -> set[str]:
    """Compute the set of legal refs from retrieved chunks."""
    return {to_ref_string(c.source, c.page) for c in chunks}


def filter_thesis_references(thesis: Thesis, chunks: Sequence[ResearchChunk]) -> Thesis:
    """Drop any LLM-invented references not present in the provided context headers."""
    allowed = _allowed_refs_from(chunks)
    if getattr(thesis, "references", None) is None:
        thesis.references = []
    thesis.references = [r for r in thesis.references if r in allowed]
    return thesis


# ---------------------------
# Retrieval
# ---------------------------

def _build_where(req: QueryRequest) -> Optional[dict]:
    """Build a Chroma metadata filter, currently by ticker if provided."""
    where = {}
    if getattr(req, "ticker", None):
        # Chroma filter expects exact match on the field stored in metadata
        where["ticker"] = req.ticker
    return where or None

_CE = None
def rerank_with_cross_encoder(query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    global _CE
    if _CE is None:
        _CE = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # fast CPU model
    pairs = [[query, d.page_content] for d in docs]
    scores = _CE.predict(pairs).tolist()
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]

def expand_queries(question: str, model: str) -> List[str]:
    prompt = f"""Give 3 alternative phrasings of the following research question as a compact JSON array of strings only. Question: {question}"""
    raw = generate(prompt=prompt, model=model)
    return json.loads(raw)[:3]

def _retrieve(db, req: QueryRequest, bm25_corpus: Optional[List[Document]] = None) -> List[Document]:
    """Retrieve candidate chunks with vector (MMR or cosine). Optionally merge BM25 if available."""
    where = _build_where(req)
    k = getattr(req, "top_k", 5) or 5
    queries = [req.question]
    try:
        queries += expand_queries(req.question, model="llama3.1:8b-instruct-q4_K_M")
    except Exception:
        pass

    
    pool: List[Document] = []
    for q in queries:
        if req.use_mmr:
            pool += chroma_mmr(db, q, k=max(5*k, 20), fetch_k=max(5*k, 40), where=where)
        else:
            pool += chroma_sim(db, q, k=max(5*k, 20), where=where)

    # If the caller supplied a BM25 corpus and hybrid module is present, use it
    if bm25_corpus and _HAS_HYBRID:
        pool += _hybrid_retrieve(db, bm25_corpus, req.question, where=where, k=max(5*k, 20))

    # Deduplicate by (source,page,text)
    seen, uniq = set(), []
    for d in pool:
        key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:60])
        if key not in seen:
            seen.add(key)
            uniq.append(d)

    # Finally re-rank and keep top k
    return rerank_with_cross_encoder(req.question, uniq, top_k=k)

# ---------------------------
# Context formatting
# ---------------------------

def _format_context_for_llm(docs: Sequence[Document]) -> tuple[str, List[ResearchChunk]]:
    """
    Build a deterministic, citation-friendly context block.

    Each chunk is wrapped like:

    [file.pdf:12]
    actual text ...

    So the model can copy refs exactly into the JSON "references" list.
    """
    lines: List[str] = []
    chunks: List[ResearchChunk] = []
    for d in docs:
        rc = to_research_chunk(d)
        chunks.append(rc)
        header = f"[{to_ref_string(rc.source, rc.page)}]"
        body = rc.snippet.strip()
        # Keep chunks tidy—trim excessive whitespace
        body = re.sub(r"\s+\n", "\n", body)
        body = re.sub(r"\n\s+", "\n", body)
        lines.append(header)
        lines.append(body)
        lines.append("")  # spacer
    return "\n".join(lines).strip(), chunks


def _summarize_forecasts(forecasts: Optional[List[OptionForecast]]) -> str:
    """Compact text summary of option forecasts for prompting."""
    if not forecasts:
        return "(No model forecasts supplied.)"
    bullets = []
    for f in forecasts:
        parts = [
            f.ticker or "",
            f"exp:{f.expiry}" if f.expiry else "",
            f"K:{f.strike}" if f.strike is not None else "",
            f"type:{f.option_type}" if f.option_type else "",
            f"model_price:{f.model_price}" if f.model_price is not None else "",
            f"IV:{f.iv}" if f.iv is not None else "",
            f"spot:{f.spot}" if f.spot is not None else "",
            f"src:{f.source}" if f.source else "",
        ]
        bullets.append(" - " + " ".join(p for p in parts if p))
    return "Model forecasts:\n" + "\n".join(bullets)


# ---------------------------
# LLM prompt & call
# ---------------------------

_SYSTEM_INSTRUCTIONS = """You are an equity research analyst.
Use ONLY the provided Context. If something is not in Context, say "insufficient context" (don’t guess).
Think through the facts and their references silently. Do not invent citations or facts. Then output ONLY the final JSON exactly as specified."""

_JSON_FORMAT_HINT = """Return STRICT JSON with the following keys:
- "ticker": string
- "one_line_view": string (<=140 chars)
- "view_confidence_10pt": integer 1..10
- "key_drivers": array of 2..5 short strings
- "risks": array of 2..5 short strings
- "suggested_actions": array of 1..5 short strings
- "references": array of strings like "file.pdf:12" copied VERBATIM from the bracketed headers in the context.
You MUST include at least 1 reference if any context was provided.
No extra keys. No commentary. Only JSON.
If you use a fact, include its source page in "references". No guessing. No extra keys. Only JSON.
"""

def ask_llm_for_thesis(
    question: str,
    context_block: str,
    forecasts_block: str,
    ticker: Optional[str],
    model: str = "llama3.1:8b-instruct-q4_K_M",
) -> Thesis:
    """Call local Ollama with a tightly-scoped prompt and parse a Thesis."""
    prompt = f"""{_SYSTEM_INSTRUCTIONS}

Question:
{question}

{_JSON_FORMAT_HINT}

{forecasts_block}

Context (each section starts with a bracketed source reference; copy these into "references"):
{context_block}
"""
    raw = generate(
        prompt=prompt,
        model=model,
        options={"temperature": 0.1, "num_ctx": 8192, "repeat_penalty": 1.05, "top_p": 0.9},
        expect_json=True,     # <— key change
        timeout=600,)

    # Extract JSON robustly (strip non-JSON tokens if any)
    json_text = _extract_json(raw)
    data = json.loads(raw)

    # Ticker fallback
    if not data.get("ticker") and ticker:
        data["ticker"] = ticker

    # Build Thesis dataclass
    return Thesis(
        ticker=data.get("ticker", ticker or ""),
        one_line_view=data.get("one_line_view", "").strip(),
        view_confidence_10pt=int(data.get("view_confidence_10pt", 5)),
        key_drivers=[s.strip() for s in data.get("key_drivers", []) if isinstance(s, str)],
        risks=[s.strip() for s in data.get("risks", []) if isinstance(s, str)],
        suggested_actions=[s.strip() for s in data.get("suggested_actions", []) if isinstance(s, str)],
        references=[s.strip() for s in data.get("references", []) if isinstance(s, str)],
    )


def _extract_json(text: str) -> str:
    """Pull out the first well-formed {...} block; raise nicely if none found."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    # Try to find a JSON object within
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    raise ValueError("LLM did not return JSON. Got:\n" + text[:500])


# ---------------------------
# Public entry point
# ---------------------------

def run_query(
    db,
    req: QueryRequest,
    bm25_corpus: Optional[List[Document]] = None,
    model: str = "llama3",
) -> QueryResponse:
    """
    Main orchestrator:
      1) retrieve chunks
      2) format context
      3) call LLM
      4) enforce reference validity
      5) return typed response
    """
    # 1) retrieve
    docs = _retrieve(db, req, bm25_corpus=bm25_corpus)

    # 2) format context (+ structured chunks)
    context_block, chunks = _format_context_for_llm(docs)

    # 3) forecasts block
    forecasts_block = _summarize_forecasts(getattr(req, "forecasts", None))

    # 4) call LLM
    thesis = ask_llm_for_thesis(
        question=req.question,
        context_block=context_block,
        forecasts_block=forecasts_block,
        ticker=getattr(req, "ticker", None),
        model=model,
    )

    # 5) post-filter references to only those present in the headers we provided
    thesis = filter_thesis_references(thesis, chunks)

    # 6) wrap & return
    return QueryResponse(thesis=thesis, retrieved=chunks)


# ---------------------------
# (Optional) utility for hybrid
# ---------------------------

def to_bm25_corpus(docs: Sequence[Document]) -> List[Document]:
    """
    Convenience: pass the same chunk list you sent to Chroma into a BM25 retriever.
    If you already have a custom BM25 corpus, ignore this.
    """
    return list(docs)
