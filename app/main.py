import hashlib
import os
import time
import json
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app import config, db, rag, vector_store
from app.ingest import file_hash, ingest_pdf
from app.llm import LLMError
from app.pdf_utils import PDFValidationError, validate_upload

def _warm_up() -> None:
    """Load the embedding model in the background so the first question is not slow."""
    try:
        from app.embeddings import get_model

        get_model()
    except Exception:
        pass  # it will simply load on first use instead


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="AI PDF Chatbot", version="4.0", lifespan=lifespan)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(5, ge=1, le=10)
    doc_ids: list[str] | None = None  # None = search all documents
    history: list[ChatTurn] = []  # earlier turns, for follow-up questions


class FeedbackRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    answer: str = Field(min_length=1, max_length=5000)
    rating: Literal["up", "down"]
    comment: str | None = Field(None, max_length=1000)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    """Upload one PDF from the app: validate, extract, chunk, embed, store."""
    contents = await file.read()

    try:
        validate_upload(file.filename or "", len(contents))
    except PDFValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    doc_id = hashlib.sha256(contents).hexdigest()[:32]
    file_path = os.path.join(config.UPLOAD_DIR, f"{doc_id}.pdf")
    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        return await run_in_threadpool(
            ingest_pdf, file_path, file.filename, doc_id, "upload"
        )
    except PDFValidationError as e:
        os.remove(file_path)  # do not keep unreadable files
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/documents")
def list_documents():
    return {"documents": db.list_documents()}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document together with its vectors."""
    existing = db.get_document(doc_id)
    removed = await run_in_threadpool(vector_store.delete_document, doc_id)
    if existing is None and removed == 0:
        raise HTTPException(status_code=404, detail="Document not found.")

    db.delete_document(doc_id)
    uploaded_copy = os.path.join(config.UPLOAD_DIR, f"{doc_id}.pdf")
    if os.path.exists(uploaded_copy):
        os.remove(uploaded_copy)
    return {"doc_id": doc_id, "deleted": True, "chunks_removed": removed}


@app.post("/index-folder")
async def index_folder():
    """Index every PDF in the company's existing folder (LIBRARY_DIR in .env).
    Files that are already indexed are skipped."""
    os.makedirs(config.LIBRARY_DIR, exist_ok=True)
    results = []
    for name in sorted(os.listdir(config.LIBRARY_DIR)):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(config.LIBRARY_DIR, name)
        try:
            validate_upload(name, os.path.getsize(path))
            result = await run_in_threadpool(
                ingest_pdf, path, name, file_hash(path), "folder"
            )
        except PDFValidationError as e:
            result = {"filename": name, "status": "skipped", "reason": str(e)}
        results.append(result)
    return {"folder": config.LIBRARY_DIR, "results": results}


@app.get("/search")
async def search_documents(
    q: str = Query(..., min_length=2, description="Your question or search phrase"),
    top_k: int = Query(5, ge=1, le=20),
    doc_id: list[str] | None = Query(None, description="Limit to these documents"),
    page: int | None = Query(None, ge=1, description="Limit to one page number"),
    mode: Literal["hybrid", "semantic"] | None = Query(
        None, description="hybrid = meaning + keywords (default), semantic = meaning only"
    ),
):
    """Search without any LLM: returns the best passages with page citations."""
    started = time.perf_counter()
    results = await run_in_threadpool(vector_store.search, q, top_k, doc_id, page, mode)
    return {
        "query": q,
        "mode": mode or config.SEARCH_MODE,
        "took_ms": round((time.perf_counter() - started) * 1000),
        "results": results,
    }


@app.post("/ask")
async def ask_question(request: AskRequest):
    """Answer a question using only the documents, with page citations.
    Replies 'Not found in document.' when the documents do not contain the answer."""
    try:
        return await run_in_threadpool(
            rag.answer_question,
            request.question,
            request.top_k,
            request.doc_ids,
            [turn.model_dump() for turn in request.history],
        )
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/ask/stream")
def ask_question_stream(request: AskRequest):
    """Same as /ask, but sent as server-sent events: many {"type": "token"} lines while
    the answer is written, then one {"type": "final"} line with the checked answer."""
    def events():
        try:
            for event in rag.stream_answer(
                request.question, request.top_k, request.doc_ids,
                [turn.model_dump() for turn in request.history],
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except LLMError as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/feedback")
def send_feedback(request: FeedbackRequest):
    """Thumbs up / down for an answer."""
    feedback_id = db.add_feedback(
        request.question, request.answer, request.rating, request.comment
    )
    return {"id": feedback_id, "saved": True}


@app.get("/feedback/summary")
def feedback_summary():
    return db.feedback_summary()
