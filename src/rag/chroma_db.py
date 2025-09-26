# src/rag/chroma_db.py
from typing import List, Optional
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

def build_chroma(
    docs: List[Document],
    persist_dir: Optional[str] = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Chroma:
    """
    Build a Chroma index from docs. If persist_dir is None, it stays in-memory.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    if persist_dir:
        db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        db.persist()
        return db
    else:
        return Chroma.from_documents(
            documents=docs,
            embedding=embeddings
        )

def similarity_search(db: Chroma, query: str, k: int = 4, where: dict | None = None):
    # LangChain Chroma supports filter=dict -> WHERE on metadata
    return db.similarity_search(query, k=k, filter=where)

def mmr_search(db: Chroma, query: str, k: int = 6, fetch_k: int = 20, where: dict | None = None):
    return db.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, filter=where)
