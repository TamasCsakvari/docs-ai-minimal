import os
import google.generativeai as genai
from typing import List

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")
genai.configure(api_key=API_KEY)

_TEXT_MODEL = genai.GenerativeModel("gemini-1.5-flash")
_EMBED_MODEL = "text-embedding-004"  # 768 dims

def generate(prompt: str) -> str:
    resp = _TEXT_MODEL.generate_content(prompt)
    return getattr(resp, "text", "")

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed each text individually and normalize the response to List[List[float]].
    Handles all known google-generativeai shapes.
    """
    def extract_values(resp) -> List[float]:
        # --- Dict shapes first ---
        if isinstance(resp, dict):
            # {"embedding": {"values": [...]}}
            emb = resp.get("embedding")
            if isinstance(emb, dict) and isinstance(emb.get("values"), list):
                return list(emb["values"])
            # {"values": [...]}
            vals = resp.get("values")
            if isinstance(vals, list):
                return list(vals)
            # {"embeddings": [ ... ]}  (batch-like)
            embs = resp.get("embeddings")
            if isinstance(embs, list) and embs:
                return extract_values(embs[0])

        # --- Object-style responses ---
        # resp.embedding.values or resp.embedding.values()
        if hasattr(resp, "embedding"):
            emb = getattr(resp, "embedding")
            if hasattr(emb, "values"):
                v = getattr(emb, "values")
                v = v() if callable(v) else v
                return list(v)

        # resp.values or resp.values()
        if hasattr(resp, "values"):
            v = getattr(resp, "values")
            v = v() if callable(v) else v
            # accept list/tuple; if it's dict_values, coerce to list
            if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
                v = list(v)
            if v and isinstance(v[0], (int, float)):
                return v  # looks like the vector
            # if it's a nested structure, try first item
            if v:
                return extract_values(v[0])

        raise TypeError(f"Unexpected embedding response shape: {type(resp)} -> {repr(resp)[:200]}")

    vectors: List[List[float]] = []
    for t in texts:
        r = genai.embed_content(
            model=_EMBED_MODEL,
            content=t,
            task_type="retrieval_document",
        )
        vectors.append(extract_values(r))
    return vectors

