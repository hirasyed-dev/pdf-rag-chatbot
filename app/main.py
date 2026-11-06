import os
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.chunking import chunk_pages
from app.pdf_utils import PDFValidationError, extract_pages, validate_upload

app = FastAPI(title="AI PDF Chatbot", version="1.0")

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    contents = await file.read()

    # 1. Validate extension and size before saving anything
    try:
        validate_upload(file.filename or "", len(contents))
    except PDFValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Save under a unique id (no overwrites, no unsafe filenames)
    doc_id = uuid.uuid4().hex
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    with open(file_path, "wb") as f:
        f.write(contents)

    # 3. Extract per page and chunk; delete the file if it is unreadable
    try:
        pages = extract_pages(file_path)
        chunks = chunk_pages(pages, doc_id)
    except PDFValidationError as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "status": "extracted",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
