# core/llm.py
import os, time
from typing import List, Literal

# Text generation (stable SDK)
import google.generativeai as genai_old

# Embeddings (new SDK)  ✅ correct import path
import google.genai as genai_new
from google.genai import types as genai_types
from google.genai import errors as genai_errors

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

genai_old.configure(api_key=API_KEY)
_client = genai_new.Client(api_key=API_KEY)

_TEXT_MODEL = genai_old.GenerativeModel("gemini-1.5-flash")
_EMBED_MODEL = "gemini-embedding-001"
_EMBED_DIM = 768  # must match vector(768) in DB

def generate(prompt: str) -> str:
    resp = _TEXT_MODEL.generate_content(prompt)
    return getattr(resp, "text", "")

def _embed_batch(texts: List[str], task_type: Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]) -> List[List[float]]:
    resp = _client.models.embed_content(
        model=_EMBED_MODEL,
        contents=texts,
        config=genai_types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=_EMBED_DIM,
        ),
    )
    return [e.values for e in resp.embeddings]

def embed_texts(
    texts: List[str],
    task: Literal["retrieval_document", "retrieval_query"] = "retrieval_document",
) -> List[List[float]]:
    # Hard API cap is 100 per request
    batch_size = 100
    out: List[List[float]] = []
    ttype = "RETRIEVAL_DOCUMENT" if task == "retrieval_document" else "RETRIEVAL_QUERY"

    # gentle throttle + retries to avoid 429
    sleep_between_batches = 0.4   # increase to 0.6–1.0 if you still hit 429
    max_retries = 4

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        attempt = 0
        delay = 0.5
        while True:
            try:
                out.extend(_embed_batch(batch, ttype))
                break
            except genai_errors.ClientError as e:
                # retry only on 429
                if getattr(e, "status_code", None) == 429 and attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2
                    attempt += 1
                    continue
                raise

        if i + batch_size < len(texts):
            time.sleep(sleep_between_batches)

    return out

def embed_docs(texts: List[str]) -> List[List[float]]:
    return embed_texts(texts, task="retrieval_document")

def embed_query(text: str) -> List[float]:
    return embed_texts([text], task="retrieval_query")[0]
