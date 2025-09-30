from __future__ import annotations

import os
import json
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Your modules ---
from src.rag.doc_indexer import files_to_documents, chunk_documents
from src.rag.chroma_db import build_chroma, load_chroma
from src.rag.query_engine import run_query
from src.rag.schema import QueryRequest, OptionForecast

from src.options_pricing.black_scholes import BlackScholes

APP = FastAPI(title="Stock Predictor ML API", version="0.1.0")

# CORS so you can call from web apps / notebooks easily
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT  = REPO_ROOT / "data" / "research"
REPORTS_DIR = REPO_ROOT / "reports"
for p in [DATA_ROOT, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---------- Schemas ----------

class ForecastInput(BaseModel):
    symbol: str
    start: str
    end: str
    forecast_days: int = 7

class ForecastOutput(BaseModel):
    symbol: str
    cmd: str
    ok: bool
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    files: List[str] = Field(default_factory=list)

class OptionInput(BaseModel):
    S: float
    K: float
    T: float
    sigma: float
    r: float

class OptionOutput(BaseModel):
    call_price: float
    put_price: float
    call_delta: float
    put_delta: float
    call_gamma: float
    put_gamma: float

class RAGQueryIn(BaseModel):
    ticker: str
    question: str
    top_k: int = 5
    use_mmr: bool = True
    forecasts: List[OptionForecast] = Field(default_factory=list)

# ---------- Health ----------

@APP.get("/health")
def health() -> dict:
    return {"status": "ok"}

# ---------- Forecast endpoint ----------

@APP.post("/forecast", response_model=ForecastOutput)
def forecast(inp: ForecastInput) -> ForecastOutput:
    """
    Reuses your run_pipeline.py exactly like the Streamlit tab.
    Writes the same report files under /reports.
    """
    cmd = [
        "python", str(REPO_ROOT / "run_pipeline.py"),
        "--symbol", f"{inp.symbol}.NS",
        "--start", inp.start,
        "--end", inp.end,
        "--forecast_days", str(inp.forecast_days),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, shell=False)

    # Collect expected output files
    base = inp.symbol.upper()
    expected = [
        REPORTS_DIR / f"{base}_lstm_forecast_plot.png",
        REPORTS_DIR / f"{base}_elasticnet_multi_step_forecast.csv",
        REPORTS_DIR / f"{base}_elasticnet_multi_step_forecast_plot.png",
        REPORTS_DIR / f"{base}.NS_actual_price.csv",
    ]
    files = [str(p) for p in expected if p.exists()]
    return ForecastOutput(
        symbol=inp.symbol,
        cmd=" ".join(cmd),
        ok=(proc.returncode == 0),
        stdout=proc.stdout[-4000:],  # tail to keep payload light
        stderr=proc.stderr[-4000:],
        files=files,
    )

# ---------- Options pricing endpoint ----------

@APP.post("/options/black-scholes", response_model=OptionOutput)
def price_options(inp: OptionInput) -> OptionOutput:
    model = BlackScholes(inp.T, inp.K, inp.S, inp.sigma, inp.r)
    model.calculate_prices()
    return OptionOutput(
        call_price=model.call_price,
        put_price=model.put_price,
        call_delta=model.call_delta,
        put_delta=model.put_delta,
        call_gamma=model.call_gamma,
        put_gamma=model.put_gamma,
    )

# ---------- RAG: index ----------

@APP.post("/rag/index")
async def rag_index(
    ticker: str = Form(...),
    files: List[UploadFile] = File(...),
) -> dict:
    """
    Upload PDFs/TXT/CSV/XLSX and build a Chroma index: data/research/{ticker}
    """
    persist_dir = DATA_ROOT / ticker
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Read all uploads into memory (same as your Streamlit path)
    tupled = []
    for up in files:
        content = await up.read()
        tupled.append((up.filename, content, up.content_type or ""))

    page_docs = files_to_documents(tupled)
    chunks = chunk_documents(page_docs, chunk_size=800, chunk_overlap=120)

    # Use the langchain-chroma implementation (no .persist() call)
    build_chroma(chunks, persist_dir=str(persist_dir))
    return {"ticker": ticker, "chunks_indexed": len(chunks), "persist_dir": str(persist_dir)}

# ---------- RAG: query ----------

@APP.post("/rag/query")
def rag_query(q: RAGQueryIn) -> dict:
    """
    Query existing index for `q.ticker` and return thesis + retrieved chunks.
    """
    persist_dir = DATA_ROOT / q.ticker
    if not persist_dir.exists():
        raise HTTPException(status_code=404, detail=f"Index not found at {persist_dir}")

    db = load_chroma(persist_dir=str(persist_dir))
    resp = run_query(db, QueryRequest(**q.model_dump()))
    # Return friendly JSON
    thesis = resp.thesis.model_dump()
    retrieved = [c.model_dump() for c in resp.retrieved]
    return {"thesis": thesis, "retrieved": retrieved}

# ---------- Dev runner ----------

if __name__ == "__main__":
    import uvicorn
    # bind to all interfaces so other devices on LAN can hit it
    uvicorn.run("src.apps.fastapi:APP", host="0.0.0.0", port=8000, reload=False)
