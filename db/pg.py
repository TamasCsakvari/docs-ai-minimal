# db/pg.py
from typing import List, Tuple
from sqlalchemy import text
from .session import get_session
from core.llm import embed_docs, embed_query  # task-aware wrappers

# rows: (id, text, embedding, source)
def insert_embeddings(rows: List[Tuple[str, str, list, str]]):
    """
    Insert pre-computed (doc) embeddings into pgvector.
    Each row: (uuid, chunk_text, embedding_vector, source_name)
    """
    sql = text("""
        INSERT INTO embeddings (id, text, embedding, source)
        VALUES (:id, :text, :embedding, :source)
    """)
    with get_session() as s:
        for r in rows:
            s.execute(sql, {"id": r[0], "text": r[1], "embedding": r[2], "source": r[3]})
        s.commit()

def similarity_search(query: str, k: int = 4) -> List[str]:
    """
    Embed the user query with RETRIEVAL_QUERY, then cosine search against stored doc embeddings.
    """
    qvec = embed_query(query)  # task_type="retrieval_query"
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

