from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
from pathlib import Path
import uvicorn

from src.rag.doc_indexer import files_to_documents, chunk_documents
from src.rag.chroma_db import build_chroma, load_chroma
from src.rag.schema import QueryRequest, OptionForecast
from src.rag.query_engine import run_query

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = APP_ROOT / "data" / "research"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Stock RAG API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/index")
async def index_docs(ticker: str = Form(...), files: List[UploadFile] = File(...)):
    ticker = ticker.upper()
    persist_dir = DATA_ROOT / ticker
    persist_dir.mkdir(parents=True, exist_ok=True)

    tupled = []
    for f in files:
        bytes_ = await f.read()
        tupled.append((f.filename, bytes_, f.content_type or ""))

    docs = files_to_documents(tupled)
    chunks = chunk_documents(docs, chunk_size=1200, chunk_overlap=120)
    build_chroma(chunks, persist_dir=str(persist_dir))
    return {"ticker": ticker, "indexed_chunks": len(chunks), "persist_dir": str(persist_dir)}

@app.post("/query")
async def query_rag(
    ticker: str = Form(...),
    question: str = Form(...),
    top_k: int = Form(5),
    use_mmr: bool = Form(True),
    spot: Optional[float] = Form(None),
    strike: Optional[float] = Form(None),
    expiry: Optional[str] = Form(None),
    option_type: Optional[str] = Form(None),
    iv: Optional[float] = Form(None),
    model_price: Optional[float] = Form(None),
):
    ticker = ticker.upper()
    persist_dir = DATA_ROOT / ticker
    db = load_chroma(persist_dir=str(persist_dir))

    forecasts = []
    if spot and strike and expiry and option_type:
        forecasts.append(OptionForecast(
            ticker=ticker, expiry=expiry, strike=float(strike),
            option_type=option_type, model_price=model_price, iv=iv, spot=spot, source="api"
        ))

    req = QueryRequest(
        ticker=ticker, question=question, top_k=top_k, use_mmr=use_mmr, forecasts=forecasts
    )
    resp = run_query(db, req)
    return JSONResponse(resp.model_dump())

if __name__ == "__main__":
    uvicorn.run("src.apps.api:app", host="0.0.0.0", port=8000, reload=False)
