import pymupdf as fitz  # PyMuPDF

MAX_FILE_SIZE_MB = 20
MAX_PAGES = 200
MIN_TEXT_CHARS = 50  # below this we treat the PDF as scanned/unreadable


class PDFValidationError(Exception):
    """Raised when an uploaded PDF fails validation."""


def validate_upload(filename: str, size_bytes: int) -> None:
    """Check extension and size BEFORE touching the file contents."""
    if not filename.lower().endswith(".pdf"):
        raise PDFValidationError("Only .pdf files are allowed.")
    if size_bytes == 0:
        raise PDFValidationError("The file is empty.")
    if size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise PDFValidationError(f"File is larger than {MAX_FILE_SIZE_MB} MB.")


def extract_pages(pdf_path: str) -> list[dict]:
    """
    Returns a list like:
    [{"page_number": 1, "text": "..."}, {"page_number": 2, "text": "..."}]
    Page numbers start at 1, so citations match what the user sees in a PDF viewer.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise PDFValidationError("Could not open the PDF. It may be corrupted.") from e

    try:
        if doc.needs_pass:
            raise PDFValidationError("Password-protected PDFs are not supported.")
        if doc.page_count > MAX_PAGES:
            raise PDFValidationError(f"PDF has more than {MAX_PAGES} pages.")

        pages = []
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            pages.append({"page_number": index, "text": text})
    finally:
        doc.close()

    total_chars = sum(len(p["text"]) for p in pages)
    if total_chars < MIN_TEXT_CHARS:
        raise PDFValidationError(
            "No readable text found. This looks like a scanned PDF, which is not supported."
        )

    return pages
