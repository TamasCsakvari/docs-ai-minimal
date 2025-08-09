from typing import List, Tuple
from sqlalchemy import text
from .session import get_session
from core.llm import embed_texts

# rows: (id, text, embedding, source)
def insert_embeddings(rows: List[Tuple[str, str, list, str]]):
    sql = text("""
        INSERT INTO embeddings (id, text, embedding, source)
        VALUES (:id, :text, :embedding, :source)
    """)
    with get_session() as s:
        for r in rows:
            s.execute(sql, {"id": r[0], "text": r[1], "embedding": r[2], "source": r[3]})
        s.commit()

def similarity_search(query: str, k: int = 4) -> List[str]:
    qvec = embed_texts([query])[0]
    sql = text("""
        WITH q AS (
          SELECT :qvec::vector AS qemb
        )
        SELECT e.text
        FROM q, embeddings e
        ORDER BY (e.embedding <=> q.qemb)
        LIMIT :k
    """)
    with get_session() as s:
        rows = s.execute(sql, {"qvec": qvec, "k": k}).fetchall()
        return [r[0] for r in rows]
