# Docs-AI Minimal

Local RAG stack using **FastAPI**, **Postgres + pgvector**, **Redis**, and **Gemini** models.

We use:

* **Gemini 1.5 Flash** for text generation
* **Gemini embeddings** (via `google-genai`) for vector search
* **pgvector** for similarity search
* **Redis** for caching
* **Docker Compose** to run Postgres + Redis locally

---

## 1. Prerequisites

* Python 3.11+
* Docker Desktop
* Google AI Studio API key (`GEMINI_API_KEY`)

---

## 2. Clone & Environment Setup

```bash
git clone <this-repo-url>
cd docs-ai-minimal
```

### Create and activate virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment variables

Create `.env` in the repo root:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/postgres
REDIS_URL=redis://localhost:6379
GEMINI_API_KEY=YOUR_GOOGLE_AI_STUDIO_KEY
```

---

## 3. Start Postgres + Redis

```bash
docker compose -f docker-compose.dev.yml up -d
```

Initialize the database schema:

```bash
docker cp infra/sql_init.sql docs-ai-minimal-pg-1:/sql_init.sql
docker exec -it docs-ai-minimal-pg-1 psql -U postgres -f /sql_init.sql
```

---

## 4. Run the API server

```bash
python -m uvicorn api.main:app --reload
```

* Health check → [http://localhost:8000/healthz](http://localhost:8000/healthz)
* Swagger UI → [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 5. Usage

### Upload a PDF

```bash
curl -X POST -F "file=@data/examplepdf.pdf" http://localhost:8000/upload
```

**Response:**

```json
{"status": "ok", "chunks": 909}
```

### Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Summarize the document in 2 sentences."}'
```

---

## 6. How It Works

* **/upload** → Extracts PDF text, chunks it (with overlap), generates embeddings via `google-genai` (output\_dimensionality=768), stores in Postgres pgvector.
* **/ask** → Embeds the query, searches similar chunks in pgvector, sends top results to Gemini 1.5 Flash for the answer.

Chunking config (in `core/ingest.py`):

```python
def chunk(text: str, max_chars=2400, overlap=150):
    ...
```

* `max_chars` → chunk size
* `overlap` → shared characters between chunks

---

## 7. Performance Notes

* **Embedding size:** We use 768-d vectors to match `vector(768)` in DB.
* **Batching:** Implemented with `google-genai` for faster ingestion.
* **Overlap tuning:** Keep 10–15% overlap to preserve context.

---

## 8. Troubleshooting

* **`python-multipart` missing** → `pip install python-multipart`
* **`cannot adapt type 'builtin_function_or_method'`** → Ensure embeddings return `List[float]`, not a method.
* **Slow ingestion** → Increase chunk size or batch size, ensure batching is enabled.
* **No text in PDF** → Must be a text-based PDF, not scanned images.

---

## 9. Reset Local Data

```bash
docker compose -f docker-compose.dev.yml down -v
```

This removes all DB + Redis data.

---

## License

AGPL-3.0
