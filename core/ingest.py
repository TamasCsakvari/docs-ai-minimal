import io, uuid
from typing import List, Tuple
from pypdf import PdfReader
from core.llm import embed_texts
from db.pg import insert_embeddings

def pdf_to_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk(text: str, max_chars=1800, overlap=200) -> List[str]:
    if not text:
        return []
    step = max(1, max_chars - overlap)
    return [text[i:i+max_chars] for i in range(0, len(text), step)]

def ingest_pdf(pdf_bytes: bytes, source: str) -> int:
    text = pdf_to_text(pdf_bytes)
    chunks = [c.strip() for c in chunk(text) if c.strip()]
    if not chunks:
        return 0
    vecs = embed_texts(chunks)
    rows: List[Tuple[str, str, list, str]] = [
        (str(uuid.uuid4()), c, v, source) for c, v in zip(chunks, vecs)
    ]
    insert_embeddings(rows)
    return len(rows)
