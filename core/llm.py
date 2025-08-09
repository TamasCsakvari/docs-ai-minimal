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
    Embed each text individually and normalize the result to List[List[float]].
    Handles: list of floats, dict['values'], dict['embedding']['values'],
    dict['embeddings'][0], object.embedding.values, object.values(),
    and dict_values([...]) wrappers.
    """
    def normalize_vector(x) -> List[float]:
        # Already a numeric list
        if isinstance(x, list) and x and all(isinstance(a, (int, float)) for a in x):
            return [float(a) for a in x]
        # A tuple of numbers
        if isinstance(x, tuple) and x and all(isinstance(a, (int, float)) for a in x):
            return [float(a) for a in x]
        # A 1-item list/tuple that itself contains the vector
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
            return normalize_vector(x[0])
        # dict_values([...]) → cast to list then normalize
        if str(type(x)).endswith("dict_values'>"):
            x = list(x)
            return normalize_vector(x)
        raise TypeError(f"Not a numeric vector: {type(x)} -> {repr(x)[:200]}")

    def extract_values(resp) -> List[float]:
        # 1) Plain list/tuple of floats
        if isinstance(resp, (list, tuple)):
            return normalize_vector(resp)

        # 2) dict shapes
        if isinstance(resp, dict):
            # {"embedding": {"values": [...]}}
            if "embedding" in resp and isinstance(resp["embedding"], dict):
                emb = resp["embedding"].get("values", resp["embedding"])
                return normalize_vector(emb)
            # {"values": [...]}
            if "values" in resp:
                return normalize_vector(resp["values"])
            # {"embeddings": [ ... ]} (batch-like)
            if "embeddings" in resp and isinstance(resp["embeddings"], list) and resp["embeddings"]:
                return extract_values(resp["embeddings"][0])

        # 3) object-style: .embedding.values or .values
        if hasattr(resp, "embedding"):
            emb = getattr(resp, "embedding")
            if hasattr(emb, "values"):
                v = emb.values() if callable(emb.values) else emb.values
                return normalize_vector(v)

        if hasattr(resp, "values"):
            v = resp.values() if callable(resp.values) else resp.values
            # Could be list/tuple/dict_values or nested list-of-list
            return normalize_vector(v)

        # 4) Last resort: if it's iterable and yields one item, try that item
        try:
            from collections.abc import Iterable
            if isinstance(resp, Iterable) and not isinstance(resp, (str, bytes, dict)):
                lst = list(resp)
                if lst:
                    return extract_values(lst[0])
        except Exception:
            pass

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

