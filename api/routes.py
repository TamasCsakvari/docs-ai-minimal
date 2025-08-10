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
    """
    Ask a question about a previously uploaded document.

    Args:
        payload: an AskRequest with a question field.

    Returns:
        a dictionary with a single key "answer" containing a string answer.

    Raises:
        400: if the question is empty.
    """
    if not payload.question.strip():
        raise HTTPException(400, "question required")
    result = await asyncio.to_thread(qa_graph.invoke, {"question": payload.question})
    return {"answer": result["answer"]}

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Upload a PDF document to be queried later.

    Args:
        file: the UploadFile to be ingested.

    Returns:
        a dictionary with a single key "chunks" containing the number of chunks
        extracted from the PDF, and a "status" key with value "ok" if successful.

    Raises:
        400: if the uploaded file is not a PDF.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF supported")
    content = await file.read()
    count = await asyncio.to_thread(ingest_pdf, content, file.filename)
    return {"status": "ok", "chunks": count}