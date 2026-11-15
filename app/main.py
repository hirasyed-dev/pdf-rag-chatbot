import hashlib
import os
import time

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from app import config, db, vector_store
from app.ingest import file_hash, ingest_pdf
from app.pdf_utils import PDFValidationError, validate_upload

app = FastAPI(title="AI PDF Chatbot", version="2.0")


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
):
    """Semantic search without any LLM: returns the best passages with page citations."""
    started = time.perf_counter()
    results = await run_in_threadpool(vector_store.search, q, top_k, doc_id, page)
    return {
        "query": q,
        "took_ms": round((time.perf_counter() - started) * 1000),
        "results": results,
    }
