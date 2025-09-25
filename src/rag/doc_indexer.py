# src/rag/doc_indexer.py
from io import BytesIO, StringIO
from typing import List, Tuple
import pandas as pd
from pypdf import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---- loaders (in-memory) ----

def _load_pdf_bytes(file_bytes: bytes, file_name: str) -> str:
    """
    Read PDF bytes (no disk I/O) and return concatenated text.
    """
    try:
        reader = PdfReader(BytesIO(file_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        text = "\n".join(pages)
        if not text.strip():
            text = "(No extractable text found in PDF)"
        return text
    except Exception as e:
        return f"(Failed to read PDF: {file_name}. Error: {e})"

def _load_text_bytes(file_bytes: bytes, encoding: str = "utf-8") -> str:
    try:
        return file_bytes.decode(encoding, errors="ignore")
    except Exception:
        return file_bytes.decode("latin-1", errors="ignore")

def _load_csv_bytes(file_bytes: bytes) -> str:
    try:
        df = pd.read_csv(BytesIO(file_bytes))
        return df.to_csv(index=False)
    except Exception:
        # Try Excel fallback if someone uploaded xlsx with csv mimetype
        try:
            df = pd.read_excel(BytesIO(file_bytes))
            return df.to_csv(index=False)
        except Exception:
            return "(Failed to parse CSV/Excel)"

# ---- public API ----

def files_to_documents(
    files: List[Tuple[str, bytes, str]]
) -> List[Document]:
    """
    Convert a list of (filename, bytes, mimetype) to LangChain Documents.
    Keeps all content in-memory.
    """
    docs: List[Document] = []
    for fname, fbytes, mimetype in files:
        name_lower = fname.lower()
        if name_lower.endswith(".pdf") or "pdf" in mimetype:
            text = _load_pdf_bytes(fbytes, fname)
        elif name_lower.endswith((".txt", ".md")) or ("text" in mimetype):
            text = _load_text_bytes(fbytes)
        elif name_lower.endswith((".csv", ".xlsx")) or ("csv" in mimetype or "excel" in mimetype):
            text = _load_csv_bytes(fbytes)
        else:
            # fallback as text
            text = _load_text_bytes(fbytes)

        # Create a single Document; metadata can carry file info
        docs.append(Document(
            page_content=text,
            metadata={"source": fname, "bytes": len(fbytes)}
        ))
    return docs

def chunk_documents(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120
) -> List[Document]:
    """
    Split documents into overlapping chunks suitable for retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)
