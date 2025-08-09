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
    # Batch embed; google-generativeai returns different shapes per version
    out = genai.embed_content(model=_EMBED_MODEL, content=texts, task_type="retrieval_document")
    if isinstance(out, dict) and "embeddings" in out:
        return [e["values"] for e in out["embeddings"]]
    if hasattr(out, "embeddings"):
        return [e.values for e in out.embeddings]
    # Single fallback
    return [out["embedding"]["values"]]
