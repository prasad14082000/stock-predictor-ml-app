# src/rag/hybrid_retriever.py

from langchain_community.retrievers import BM25Retriever

def hybrid_retrieve(chroma_db, bm25_corpus, query, where=None, k=5):
    # 1) vector
    v_hits = chroma_db.similarity_search(query, k=k, filter=where)
    # 2) bm25
    bm25 = BM25Retriever.from_documents(bm25_corpus)
    b_hits = bm25.get_relevant_documents(query)
    # 3) merge (dedupe by (source,page) and keep top k)
    seen, merged = set(), []
    for d in v_hits + b_hits:
        key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:40])
        if key in seen: continue
        seen.add(key); merged.append(d)
        if len(merged) >= k: break
    return merged
