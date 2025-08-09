# core/llm.py
import os
from typing import List, Literal

# --- Generation (keep your current SDK) ---
import google.generativeai as genai_old
# --- Embeddings (new SDK) ---
from google import genai as genai_new
from google.genai import types as genai_types

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

# Configure both clients
genai_old.configure(api_key=API_KEY)
_client = genai_new.Client(api_key=API_KEY)

# Models
_TEXT_MODEL = genai_old.GenerativeModel("gemini-1.5-flash")
_EMBED_MODEL = "gemini-embedding-001"
_EMBED_DIM = 768  # keep in sync with DB: vector(768)

def generate(prompt: str) -> str:
    resp = _TEXT_MODEL.generate_content(prompt)
    return getattr(resp, "text", "")

def _embed_batch(texts: List[str], task_type: Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]) -> List[List[float]]:
    # Batch call with fixed dimensionality for pgvector
    resp = _client.models.embed_content(
        model=_EMBED_MODEL,
        contents=texts,
        config=genai_types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=_EMBED_DIM,
        ),
    )
    # resp.embeddings is a list; each item has .values
    return [e.values for e in resp.embeddings]

def embed_texts(
    texts: List[str],
    task: Literal["retrieval_document", "retrieval_query"] = "retrieval_document",
) -> List[List[float]]:
    """Public generic helper (task-aware) with batching."""
    batch_size = 128
    out: List[List[float]] = []
    ttype = "RETRIEVAL_DOCUMENT" if task == "retrieval_document" else "RETRIEVAL_QUERY"
    for i in range(0, len(texts), batch_size):
        out.extend(_embed_batch(texts[i:i+batch_size], ttype))
    return out

# Convenience wrappers
def embed_docs(texts: List[str]) -> List[List[float]]:
    return embed_texts(texts, task="retrieval_document")

def embed_query(text: str) -> List[float]:
    return embed_texts([text], task="retrieval_query")[0]
