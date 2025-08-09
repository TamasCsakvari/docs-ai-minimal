# core/llm.py
import os
from typing import List, Literal

# Text generation uses the google-generativeai SDK (current stable for content generation)
import google.generativeai as genai

# Embeddings use the google-genai SDK (newer API, supports batching + output_dimensionality)
from google import genai as genai_emb
from google.genai import types as genai_types

# --- Configuration ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

# Configure both clients
genai.configure(api_key=API_KEY)         # for generation
emb_client = genai_emb.Client(api_key=API_KEY)  # for embeddings

# --- Models ---
TEXT_MODEL = genai.GenerativeModel("gemini-1.5-flash")
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768  # must match pgvector column size

# --- Generation ---
def generate(prompt: str) -> str:
    resp = TEXT_MODEL.generate_content(prompt)
    return getattr(resp, "text", "")

# --- Embeddings ---
def _embed_batch(
    texts: List[str],
    task_type: Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
) -> List[List[float]]:
    """Batch embed text using google-genai."""
    resp = emb_client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=genai_types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBED_DIM,
        ),
    )
    return [e.values for e in resp.embeddings]

def embed_texts(
    texts: List[str],
    task: Literal["retrieval_document", "retrieval_query"] = "retrieval_document",
) -> List[List[float]]:
    """Embed a list of texts with batching."""
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
