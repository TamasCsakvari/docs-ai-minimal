from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="Docs-AI Minimal")
app.include_router(router)

@app.get("/healthz")
async def healthz():
    return {"ok": True}
