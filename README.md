# AI PDF Chatbot (Internal)

Internal web app that lets users upload PDFs (or index an existing folder of PDFs),
ask questions in natural language, and get answers with page-level citations (RAG).

## Status
- [x] Phase 1: FastAPI setup, PDF validation, page-wise extraction (PyMuPDF), page-bound chunking, pytest
- [x] Phase 2: Local embeddings (sentence-transformers), ChromaDB with metadata filtering, semantic search API, document list/delete, SQLite tracking, folder indexing
- [ ] Phase 3: LLM answering with citations + Streamlit UI
- [ ] Phase 4: Multi-document, 20-question benchmark, final docs

## Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env       # then edit if needed
uvicorn app.main:app --reload
```
API docs: http://127.0.0.1:8000/docs

## API
| Method | Endpoint | What it does |
|---|---|---|
| POST | `/documents` | Upload a PDF (validate, extract, chunk, embed, store) |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{doc_id}` | Delete a document and its vectors |
| POST | `/index-folder` | Index all PDFs in the existing folder (`LIBRARY_DIR`) |
| GET | `/search?q=...&top_k=5&doc_id=...&page=...` | Semantic search with page citations (no LLM) |

## Tests
```bash
pytest
```

## Design decisions
- Chunks never cross page boundaries, so every citation points to the exact page.
- Chunk size is about 450 tokens with a 50 token overlap (spec: 300-500).
- Embeddings use `BAAI/bge-small-en-v1.5` (local, reads up to 512 tokens, so chunks are not cut off).
- A document id is the hash of the file content, so uploading the same PDF twice never creates duplicates.
- Scanned (image-only) PDFs are detected and rejected with a clear message.
