# Docs-AI Minimal

### What 🔎  
RAG API for document Q&A that ingests PDFs, finds the most relevant chunks, and returns grounded answers with low latency.

### How ⚙️  
On upload, PDFs are extracted, chunked with overlap, embedded (768-d), and stored in PostgreSQL/pgvector for cosine search; on ask, the query is embedded, top-k chunks are retrieved, and **Gemini 1.5 Flash** generates a context-bound answer with Redis caching and SQLAlchemy-managed sessions; endpoints: `POST /upload`, `POST /ask`.

### Why 💡  
Cuts manual reading time, keeps answers faithful to your documents, and runs locally with simple ops—fast to prototype, easy to productionize.

### Techstack 🧰  
FastAPI, LangGraph, Google GenAI **Embedding-001 (768)**, **Gemini 1.5 Flash**, PostgreSQL + **pgvector (IVFFLAT)**, Redis cache, SQLAlchemy, Docker Compose, `.env` config (`DATABASE_URL`, `REDIS_URL`, `GEMINI_API_KEY`), batching + exponential backoff, cosine similarity search.

![Docs-AI Minimal](docs/docs-ai-minimal-architecture.drawio.png)

Local RAG stack using **FastAPI**, **Postgres + pgvector**, **Redis**, and **Gemini** models.

We use:
* **Gemini 1.5 Flash** for text generation
* **Gemini embeddings** (via `google-genai`) for vector search
* **pgvector** for similarity search
* **Redis** for caching
* **Docker Compose** to run Postgres + Redis locally

## Requirements
* Python 3.11+
* Docker Desktop
* Google AI Studio API key (`GEMINI_API_KEY`)

## Set up

Create and activate virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1 # Windows PowerShell
source .venv/bin/activate # macOS/Linux
```

Install dependencies
```bash
pip install -r requirements.txt
```

Set up environment variables
Create `.env` in the repo root:
```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/postgres
REDIS_URL=redis://localhost:6379
GEMINI_API_KEY=YOUR_GOOGLE_AI_STUDIO_KEY
```

Start Postgres + Redis
```bash
docker compose -f docker-compose.dev.yml up -d
```

Initialize the database schema:
```bash
docker cp infra/sql_init.sql docs-ai-minimal-pg-1:/sql_init.sql
docker exec -it docs-ai-minimal-pg-1 psql -U postgres -f /sql_init.sql
```

Run the API server
```bash
python -m uvicorn api.main:app --reload
```

## Usage
Upload a PDF
```bash
curl -X POST -F "file=@data/examplepdf.pdf" http://localhost:8000/upload
```
Ask a question
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"Summarize the document in 2 sentences.\"}"
```

## How It Works
* **/upload** → Extracts PDF text, chunks it (with overlap), generates embeddings via `google-genai` (output\_dimensionality=768), stores in Postgres pgvector.
* **/ask** → Embeds the query, searches similar chunks in pgvector, sends top results to Gemini 1.5 Flash for the answer.

Chunking config (in `core/ingest.py`):
```python
def chunk(text: str, max_chars=2400, overlap=150):
    ...
```
* `max_chars` → chunk size
* `overlap` → shared characters between chunks



## Reset Local Data
This removes all DB + Redis data.
```bash
docker compose -f docker-compose.dev.yml down -v
```



## Code Architecture & Functionality

**High-level flow**

1. `POST /upload` → extract PDF text → chunk → embed chunks (batching + backoff) → store text + vectors in Postgres (pgvector).
2. `POST /ask` → embed query → cosine search in pgvector → pass top docs as context to Gemini → return grounded answer.

**Directories**

* `api/`

  * `main.py` – FastAPI app + router registration.
  * `routes.py` – HTTP endpoints: `/healthz`, `/upload`, `/ask`.
* `core/`

  * `ingest.py` – PDF → text, chunking, calls `embed_docs()` and persists rows.
  * `llm.py` – Text generation (Gemini 1.5 Flash) and embeddings (google‑genai) with batching + retry/backoff.
  * `workflows.py` – Minimal RAG pipeline (retrieve → generate) built with LangGraph.
* `db/`

  * `session.py` – SQLAlchemy engine/session from `DATABASE_URL`.
  * `pg.py` – `insert_embeddings()` and `similarity_search()` with pgvector.
  * `redis.py` – tiny JSON cache for retrieved docs.
* `infra/`

  * `sql_init.sql` – pgvector schema (768‑d vectors + IVFFLAT index).

**Endpoints**

* `GET /healthz` – liveness probe.
* `POST /upload` (multipart form) – field `file` (PDF). Returns `{status, chunks}`.
* `POST /ask` (JSON) – body `{"question": "..."}`. Returns `{answer}`.

**Chunking** (`core/ingest.py`)

* `chunk(text, max_chars=2400, overlap=150)` → \~10–15% overlap preserves context across boundaries.

**Embeddings** (`core/llm.py`)

* Model: `models/embedding-001` with `output_dimensionality=768` (matches `vector(768)`).
* Task pairing: docs → `RETRIEVAL_DOCUMENT`, queries → `RETRIEVAL_QUERY`.
* Batching: default `EMBED_BATCH_SIZE=24` (env override) with exponential backoff on 429/5xx.

**Retrieval** (`db/pg.py`)

* Cosine distance: `ORDER BY (e.embedding <=> :query_vec)`.
* Returns top‑k chunk texts to the generator.

**Generation** (`core/llm.py` → `generate()`)

* Model: `gemini-1.5-flash`.
* Prompt instructs “answer strictly from provided context; say you can’t find it if missing”.

**Config knobs**

* `.env` – `DATABASE_URL`, `REDIS_URL`, `GEMINI_API_KEY`.
* (Optional) `EMBED_BATCH_SIZE` – tune request burst to avoid throttling.
* Adjust `max_chars/overlap` in `core/ingest.py` for ingestion speed vs. recall.

