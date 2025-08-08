import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from core.ingest import ingest_pdf
from core.workflows import qa_graph

router = APIRouter()

class AskRequest(BaseModel):
    question: str

@router.post("/ask")
async def ask(payload: AskRequest):
    if not payload.question.strip():
        raise HTTPException(400, "question required")
    result = await asyncio.to_thread(qa_graph.invoke, {"question": payload.question})
    return {"answer": result["answer"]}

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF supported")
    content = await file.read()
    count = await asyncio.to_thread(ingest_pdf, content, file.filename)
    return {"status": "ok", "chunks": count}