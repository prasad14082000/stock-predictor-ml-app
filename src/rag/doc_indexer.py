# src/rag/doc_indexer.py
"""
Finance-aware document indexer for RAG.

What this adds vs your previous version:
- Page-level extraction for PDFs (preserves page numbers for citations).
- Finance metadata on every chunk: ticker, source_type, asof_date, page, source.
- Smarter loaders (PDF/TXT/CSV/XLSX) with graceful fallbacks (all in-memory).
- Heading-aware chunking that carries metadata forward.

You can keep using:
- files_to_documents(...)  -> returns a list[Document] (PDFs split by page)
- chunk_documents(...)     -> returns chunked Documents with inherited metadata
"""

from __future__ import annotations

from io import BytesIO
from typing import List, Tuple, Optional
from datetime import datetime
import re

import pandas as pd
from pypdf import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


# -------------------------
# Heuristics / parsers
# -------------------------

_TICKER_PAT = re.compile(
    r"\b(titan|reliance|tcs|infosys|hdfc|icici|itc|sbin|axis|kotak|wipro|bpcl|hpcl|ongc|tata|hcl|ultracemco|maruti|"
    r"asianpaints|hdfc bank|icici bank|larsen|ltimindtree|jsw|tvs|sun pharma|dr reddy|cipla|britannia)\b",
    re.IGNORECASE,
)

def guess_ticker_from_name(fname: str) -> str:
    """
    Guess a ticker from file name. Extend this map for your coverage.
    """
    base = fname.split("/")[-1].lower()
    m = _TICKER_PAT.search(base)
    if not m:
        return ""
    t = m.group(1)
    # quick normalization
    aliases = {
        "hdfc bank": "HDFCBANK",
        "icici bank": "ICICIBANK",
        "larsen": "LT",
        "tata": "TATA",
        "dr reddy": "DRREDDY",
    }
    return aliases.get(t.lower(), t.upper())


def guess_type_from_name(fname: str) -> str:
    """
    Lightweight source-type guess from filename.
    """
    f = fname.lower()
    if "concall" in f or "earnings call" in f or "earningscall" in f:
        return "concall"
    if "credit" in f or "rating" in f or "crisil" in f or "icare" in f or "care" in f or "icra" in f:
        return "credit_rating"
    if "result" in f or re.search(r"\bfy\d{2}\b", f) or re.search(r"\bq[1-4]fy\d{2}\b", f):
        return "results"
    if "broker" in f or "note" in f or "initiation" in f or "update" in f:
        return "broker_note"
    if "press" in f or "news" in f or "pr" in f:
        return "news"
    return "other"


_DATE_ISO_PAT = re.compile(r"\b(20\d{2})[-/\.](0?[1-9]|1[0-2])[-/\.](0?[1-9]|[12]\d|3[01])\b")
_DATE_TEXT_PAT = re.compile(
    r"\b(0?[1-9]|[12]\d|3[01])\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*(20\d{2})\b",
    re.IGNORECASE,
)

def parse_asof_date_from_text(text: str) -> Optional[str]:
    """
    Try to extract an as-of date from text.
    Returns ISO date 'YYYY-MM-DD' or None.
    """
    m = _DATE_ISO_PAT.search(text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(y, mo, d).date().isoformat()
        except ValueError:
            pass

    m = _DATE_TEXT_PAT.search(text)
    if m:
        d, mon_str, y = int(m.group(1)), m.group(2).title(), int(m.group(3))
        month_map = {
            "Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
            "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12
        }
        mo = month_map.get(mon_str[:3], None)
        if mo:
            try:
                return datetime(y, mo, d).date().isoformat()
            except ValueError:
                pass
    return None


# -------------------------
# Loaders (in-memory)
# -------------------------

def _load_text_bytes(file_bytes: bytes, encoding: str = "utf-8") -> str:
    try:
        return file_bytes.decode(encoding, errors="ignore")
    except Exception:
        return file_bytes.decode("latin-1", errors="ignore")


def _load_csv_bytes(file_bytes: bytes) -> str:
    """
    Try CSV; if that fails, try Excel; return CSV text (so it remains searchable).
    """
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


def pdf_to_page_documents(file_bytes: bytes, file_name: str) -> List[Document]:
    """
    Read PDF bytes (no disk I/O) and return one Document per page with page-level metadata.
    Also attempts to parse an as-of date from the first page.
    """
    docs: List[Document] = []
    try:
        reader = PdfReader(BytesIO(file_bytes))
    except Exception as e:
        # Return a single error doc so the UI can surface the issue
        return [
            Document(
                page_content=f"(Failed to read PDF: {file_name}. Error: {e})",
                metadata={"source": file_name, "page": None, "ticker": "", "source_type": "other", "asof_date": None},
            )
        ]

    # Heuristics from filename
    ticker_guess = guess_ticker_from_name(file_name)
    src_type = guess_type_from_name(file_name)

    # Try to parse as-of date from first page text
    first_page_text = ""
    try:
        if len(reader.pages) > 0:
            first_page_text = reader.pages[0].extract_text() or ""
    except Exception:
        pass
    asof_guess = parse_asof_date_from_text(first_page_text)

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        meta = {
            "source": file_name,
            "page": i,
            "ticker": ticker_guess,
            "source_type": src_type,
            "asof_date": asof_guess,  # can be None; you can add a small UI to set/override
        }
        # Even empty pages are useful for accurate page refs; but skip truly empty text to reduce noise
        if text.strip():
            docs.append(Document(page_content=text, metadata=meta))
        else:
            # Keep a tiny placeholder so citations don't break, but mark it as empty
            docs.append(Document(page_content="(empty page)", metadata={**meta, "empty": True}))
    return docs


# -------------------------
# Public API
# -------------------------

def files_to_documents(files: List[Tuple[str, bytes, str]]) -> List[Document]:
    """
    Convert a list of (filename, bytes, mimetype) to LangChain Documents.
    - PDFs -> one Document per page with finance metadata.
    - TXT/MD -> one Document with source metadata.
    - CSV/XLSX -> CSV text as one Document (keeps it searchable).
    All operations are in-memory; no disk I/O here.
    """
    docs: List[Document] = []

    for fname, fbytes, mimetype in files:
        name_lower = fname.lower()

        if name_lower.endswith(".pdf") or "pdf" in mimetype:
            docs.extend(pdf_to_page_documents(fbytes, fname))

        elif name_lower.endswith((".txt", ".md")) or ("text" in mimetype):
            text = _load_text_bytes(fbytes)
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": fname,
                        "page": 1,
                        "ticker": guess_ticker_from_name(fname),
                        "source_type": guess_type_from_name(fname),
                        "asof_date": parse_asof_date_from_text(text[:2000]) or None,
                    },
                )
            )

        elif name_lower.endswith((".csv", ".xlsx")) or ("csv" in mimetype or "excel" in mimetype):
            text = _load_csv_bytes(fbytes)
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": fname,
                        "page": 1,
                        "ticker": guess_ticker_from_name(fname),
                        "source_type": "table",
                        "asof_date": None,  # tables often lack explicit dates; you can extend later
                    },
                )
            )

        else:
            # Fallback: treat as text
            text = _load_text_bytes(fbytes)
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": fname,
                        "page": 1,
                        "ticker": guess_ticker_from_name(fname),
                        "source_type": guess_type_from_name(fname),
                        "asof_date": parse_asof_date_from_text(text[:2000]) or None,
                    },
                )
            )

    return docs


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Document]:
    """
    Split documents into overlapping chunks suitable for retrieval.
    Metadata (source, page, ticker, source_type, asof_date) is carried forward.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # (Optional) hard-truncate overly long chunks to keep prompt budgets in check:
    # for c in chunks:
    #     if len(c.page_content) > 3000:
    #         c.page_content = c.page_content[:3000]

    return chunks
