CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS embeddings (
  id uuid PRIMARY KEY,
  text TEXT,
  embedding vector(768),
  source TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', coalesce(text, ''))) STORED
);
CREATE INDEX IF NOT EXISTS idx_embeddings_embedding
ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_embeddings_tsv ON embeddings USING gin (tsv);