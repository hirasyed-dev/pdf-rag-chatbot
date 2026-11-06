# AI PDF Chatbot (Internal)

Internal web app that lets users upload PDFs (or index an existing folder of PDFs),
ask questions in natural language, and get answers with page-level citations (RAG).

## Status
- [x] Phase 1: FastAPI setup, PDF validation, page-wise extraction (PyMuPDF), page-bound chunking, pytest
- [ ] Phase 2: Local embeddings (sentence-transformers) + ChromaDB + semantic search endpoint
- [ ] Phase 3: LLM answering with citations + Streamlit UI
- [ ] Phase 4: Multi-document, 20-question benchmark, final docs

## Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```
API docs: http://127.0.0.1:8000/docs

## Tests
```bash
pytest
```

## Design decisions
- Chunks never cross page boundaries, so every citation points to the exact page.
- Chunk size is about 450 tokens with a 50 token overlap.
- Scanned (image-only) PDFs are detected and rejected with a clear message.
