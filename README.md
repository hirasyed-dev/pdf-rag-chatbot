# AI PDF Chatbot (Internal)

Internal web app: upload PDFs (or index an existing folder of PDFs), ask questions in
natural language, and get answers with page-level citations (RAG).
If the documents do not contain the answer, the app says "Not found in document."

## Status
- [x] Phase 1: FastAPI setup, PDF validation, page-wise extraction (PyMuPDF), page-bound chunking, pytest
- [x] Phase 2: Local embeddings (sentence-transformers), ChromaDB with metadata filtering, semantic search API, document list/delete, SQLite tracking, folder indexing
- [x] Phase 3: LLM answers with citations (Gemini), "Not found in document" fallback, Streamlit UI
- [x] Phase 4: Hybrid search, streaming answers, thumbs feedback, multi-document polish, 20-question benchmark, final docs

## Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env       # then put your Gemini key in .env
python check_llm.py          # checks that the key works
```

## Run (two terminals)
```bash
# Terminal 1: backend
uvicorn app.main:app --reload

# Terminal 2: web UI
streamlit run frontend/streamlit_app.py
```
Backend docs: http://127.0.0.1:8000/docs  |  Web UI: http://localhost:8501

## API
| Method | Endpoint | What it does |
|---|---|---|
| POST | `/documents` | Upload a PDF (validate, extract, chunk, embed, store) |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{doc_id}` | Delete a document and its vectors |
| POST | `/index-folder` | Index all PDFs in the existing folder (`LIBRARY_DIR`) |
| GET | `/search?q=...&top_k=5&doc_id=...&page=...&mode=hybrid` | Search with page citations (no LLM). `mode` = `hybrid` or `semantic` |
| POST | `/ask` | Cited answer from the LLM, or "Not found in document." |
| POST | `/ask/stream` | Same as `/ask`, sent as server-sent events (tokens first, then the checked answer) |
| POST | `/feedback` | Thumbs up / down for an answer (`rating` = `up` or `down`) |
| GET | `/feedback/summary` | Number of thumbs up and down |

## Tests
```bash
pytest
```
Tests never call the real LLM or download the embedding model.

## Benchmark (acceptance criteria)
With the backend running:
```bash
python benchmark.py --search-only   # no LLM: semantic vs hybrid search comparison
python benchmark.py                 # full 20-question benchmark (15 answerable, 5 not)
python benchmark.py --upload-test   # times a generated 50-page PDF (target: under 60 s)
python benchmark.py --quick         # only 5 questions, quick speed check
python benchmark.py --only 18,20    # only these question numbers
```
Questions live in `benchmark/questions.json` (`expected_pages` lists every page that contains the
fact; an empty list means the PDF cannot answer it). Results are written to `benchmark/results.md`.
How it is scored:
- An answer counts as correct when at least one of its citations is on an expected page.
  Individual citation precision is also reported.
- An unanswerable question passes only if the reply is "Not found in document."
- First-token time is measured on the streaming endpoint, including retrieval.

Note: the free Gemini tier allows only about 20 requests per model per day, so a full run can use
up a whole day's quota. Use `--only` or `--quick` for repeat checks, or switch `LLM_MODEL`.

## Benchmark results (measured 2026-11-22)
Setup: `gemini-3.5-flash-lite`, `LLM_THINKING=minimal`, hybrid search, `AI_PDF_Chatbot.pdf` (5 pages), one run on a normal laptop.

| Acceptance criterion | Target | Measured | |
|---|---|---|---|
| Answers citing the exact page (15 answerable questions) | at least 85% | 15/15 (100%) | PASS |
| Individual citations on a correct page | for information | 18/20 (90%) | |
| Unanswerable questions answered "Not found in document" | at least 4 of 5 | 5/5 | PASS |
| Vector search time (slowest of 20) | under 2 s | 264 ms | PASS |
| First answer token | under 5 s | median 2.6 s, slowest 8.6 s | PASS (median) |
| 50-page PDF upload and indexing | under 60 s | 32.1 s | PASS |
| Unit tests | all pass | 66 passed | PASS |
| API key in the repository | none | none (`.env` is git-ignored) | PASS |

Honest notes:
- The first-token target is judged on the median. The slowest question (8.6 s) was slower because the free Gemini tier is shared and its speed varies during the day.
- The two citations that are not on an expected page (questions 8 and 12) point to page 4, which also mentions related content.
- The test document is small (5 pages), so retrieval is easy here. Semantic and hybrid search both put the right page in the top 5 for 15/15 questions and in first place for 11/15 questions. Hybrid search matters more on larger collections.
- Free-tier Gemini models allow only about 20 requests per day for the bigger models, and 500 per day for the "flash-lite" models, which is why this project defaults to a lite model for benchmarking.
- If an answer comes back without citations, the app asks the model once to add them. The final answer is always checked, so an uncited answer is never shown.

## How an answer is produced
1. The question is searched in two ways and the rankings are merged (reciprocal rank fusion):
   meaning (embeddings in ChromaDB) and exact words (BM25 keywords). Identical passages are shown once.
2. Passages below `MIN_SCORE` are dropped. If nothing is left, the LLM is not called.
3. The LLM gets the numbered passages and strict rules: use only these sources, cite as [S1], or reply "Not found in document."
4. Citations are mapped back to the stored file name and page number (never taken from the LLM's own words).
5. An answer with no valid citation is not shown; the app replies "Not found in document." instead.

## Using several documents
Leave "Ask only these documents" empty to search every indexed PDF, or pick one or more.
Each citation shows its own file name and page. If two different files share a name,
the sidebar shows a short id so they can be told apart.

## Settings (`.env`)
| Setting | Meaning |
|---|---|
| `LLM_API_KEY` | Gemini key (only ever in `.env`) |
| `LLM_MODEL` | Gemini model name. Empty = auto-pick. Pin a fast one if answers are slow (`python check_models.py` times them) |
| `LLM_THINKING` | `minimal` (default), `low`, `medium`, `high` or empty. Lower = faster first words |
| `SEARCH_MODE` | `hybrid` (default) or `semantic` |
| `MIN_SCORE` | Passages below this similarity are ignored (default 0.35) |
| `LIBRARY_DIR` | Folder of existing company PDFs for "Index existing folder" |

## Design decisions
- Chunks never cross page boundaries, so every citation points to the exact page.
- Chunk size is about 450 tokens with a 50 token overlap (spec: 300-500).
- Embeddings use `BAAI/bge-small-en-v1.5` (local, reads up to 512 tokens, so chunks are not cut off).
- A document id is the hash of the file content, so uploading the same PDF twice never creates duplicates.
- Scanned (image-only) PDFs are detected and rejected with a clear message.
- The API key is read only from `.env` (git-ignored). `app/llm.py` is the only file that talks to the LLM provider.
- Streaming shows the answer while it is written, but the final answer is always checked afterwards. If it has no valid citation it is replaced by "Not found in document."
- A follow-up question uses the previous question as context, unless that previous question was answered with "Not found".
- Tests use fake embeddings and a fake LLM, so they are fast and free.

## Limits
- Text PDFs only (no OCR), up to 20 MB and 200 pages.
- No user login (internal tool). Anyone who can open the app can see all indexed documents.
- Quality of answers depends on the Gemini model you choose.
