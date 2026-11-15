"""One shared pipeline used by both upload and folder indexing:
extract per page -> chunk (never crossing pages) -> embed -> store."""
import hashlib

from app import db, vector_store
from app.chunking import chunk_pages
from app.pdf_utils import extract_pages


def file_hash(path: str) -> str:
    """Same file content always gives the same id, so duplicates are skipped."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:32]


def ingest_pdf(path: str, filename: str, doc_id: str, source: str = "upload") -> dict:
    if vector_store.doc_exists(doc_id):
        existing = db.get_document(doc_id) or {"doc_id": doc_id, "filename": filename}
        return {**existing, "status": "already_indexed"}

    pages = extract_pages(path)  # raises PDFValidationError for bad/scanned files
    chunks = chunk_pages(pages, doc_id)
    vector_store.add_chunks(chunks, filename)
    db.upsert_document(doc_id, filename, len(pages), len(chunks), source)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "source": source,
        "status": "indexed",
    }
