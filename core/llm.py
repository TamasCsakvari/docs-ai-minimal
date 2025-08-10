# core/llm.py
import os
import time
import random
from typing import List, Literal
import google.generativeai as genai_text
from google import genai as genai_emb
from google.genai import types as genai_types
from google.genai import errors as genai_errors

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

# Configure clients
genai_text.configure(api_key=API_KEY)               # generation
_emb_client = genai_emb.Client(api_key=API_KEY)     # embeddings

TEXT_MODEL = genai_text.GenerativeModel("gemini-1.5-flash")
EMBED_MODEL = "models/embedding-001"  # google-genai model name
EMBED_DIM = 768                       # must match pgvector schema: vector(768)

# Optional tuning via env
DEFAULT_BATCH = 24
EMBED_BATCH_SIZE = max(1, min(int(os.getenv("EMBED_BATCH_SIZE", DEFAULT_BATCH)), 100))  # API cap is 100

def generate(prompt: str) -> str:
    resp = TEXT_MODEL.generate_content(prompt)
    return getattr(resp, "text", "")


def _embed_batch(
    texts: List[str], task_type: Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
) -> List[List[float]]:
    """
    Embeds a batch of text with exponential backoff for transient errors/rate limits

    Args:
        texts: list of text strings to embed
        task_type: task type for embedding (RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY)

    Returns:
        list of embedding vectors (list of lists of floats)
    """
    # Exponential backoff for transient errors/rate limits
    attempts = 0
    delay = 1.0
    while True:
        try:
            resp = _emb_client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=genai_types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBED_DIM,
                ),
            )
            return [e.values for e in resp.embeddings]
        except genai_errors.ClientError as e:
            # 429 or 5xx → backoff; 4xx (other than quota-ish) → raise
            status = getattr(e, "status_code", None)
            if status in (429, 500, 502, 503, 504) or "exhausted" in str(e).lower():
                attempts += 1
                if attempts > 6:
                    raise
                # jittered exponential backoff
                time.sleep(delay + random.uniform(0, 0.3))
                delay = min(delay * 2, 16)
            else:
                raise


def embed_texts(
    texts: List[str],
    task: Literal["retrieval_document", "retrieval_query"] = "retrieval_document",
) -> List[List[float]]:
    """
    Embed a list of texts. Uses batching with API-safe limits and backoff.
    """
    out: List[List[float]] = []
    ttype = "RETRIEVAL_DOCUMENT" if task == "retrieval_document" else "RETRIEVAL_QUERY"
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        out.extend(_embed_batch(batch, ttype))
    return out

# Convenience wrappers
def embed_docs(texts: List[str]) -> List[List[float]]:
    """Embeddings for documents/chunks."""
    return embed_texts(texts, task="retrieval_document")

def embed_query(text: str) -> List[float]:
    """Embedding for a single user query."""
    return embed_texts([text], task="retrieval_query")[0]
