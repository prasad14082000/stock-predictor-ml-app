# src/rag/chroma_db.py
from __future__ import annotations
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_chroma(
    docs: List[Document],
    persist_dir: Optional[str] = "data/research/index",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Chroma:
    """
    Build a Chroma index from docs. If persist_dir is provided, it persists to disk.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    if persist_dir:
        db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
        db.persist()
        return db
    else:
        return Chroma.from_documents(documents=docs, embedding=embeddings)

def load_chroma(
    persist_dir: str = "data/research/index",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

def similarity_search(db: Chroma, query: str, k: int = 4, where: Optional[Dict[str, Any]] = None):
    # Chroma (LangChain) supports metadata filtering via filter=dict
    return db.similarity_search(query, k=k, filter=where)

def mmr_search(db: Chroma, query: str, k: int = 6, fetch_k: int = 20, where: Optional[Dict[str, Any]] = None):
    return db.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, filter=where)
