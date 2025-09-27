from pathlib import Path
from langchain_core.documents import Document
from src.rag.doc_indexer import files_to_documents, chunk_documents
from src.rag.chroma_db import build_chroma, load_chroma
from src.rag.query_engine import run_query
from src.rag.schema import QueryRequest

# 1) Collect files from a folder (drop your concalls, credit notes, results here)
DATA_DIR = Path("data/research/raw")
persist_dir = "data/research/index"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Example: ensure at least one file exists
# Put files like: TITAN_q1fy26_concall.pdf, crisil_titan_2025.pdf, results_TITAN_Q1FY26.pdf, etc.

files = []
for p in DATA_DIR.glob("*.*"):
    mime = "application/pdf" if p.suffix.lower()==".pdf" else "text/plain"
    files.append((p.name, p.read_bytes(), mime))

docs = files_to_documents(files)
chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=120)

# 2) Build chroma index (persist)
db = build_chroma(chunks, persist_dir=persist_dir)

# 3) Query
req = QueryRequest(
    ticker="TITAN",
    question="What does management say about margins in H2? Any guidance or drivers?",
    top_k=5,
    use_mmr=True,
)
resp = run_query(db, req)
print(resp.thesis.model_dump_json(indent=2))
