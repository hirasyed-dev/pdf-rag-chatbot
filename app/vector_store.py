"""ChromaDB storage. Every chunk is saved with doc_id, filename and page_number
so that search results can always be cited and filtered."""
import chromadb

from app import config
from app.embeddings import embed_documents, embed_query
from app.keyword_search import BM25, reciprocal_rank_fusion

COLLECTION_NAME = "pdf_chunks"

_client = None
_keyword_cache: dict = {}  # filter key -> (chunk count, ids, BM25 index)


def get_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return _client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def add_chunks(chunks: list[dict], filename: str) -> None:
    if not chunks:
        return
    _keyword_cache.clear()
    texts = [c["text"] for c in chunks]
    get_collection().add(
        ids=[f"{c['doc_id']}_{c['chunk_index']}" for c in chunks],
        documents=texts,
        embeddings=embed_documents(texts),
        metadatas=[
            {
                "doc_id": c["doc_id"],
                "filename": filename,
                "page_number": c["page_number"],
                "chunk_index": c["chunk_index"],
            }
            for c in chunks
        ],
    )


def doc_exists(doc_id: str) -> bool:
    result = get_collection().get(where={"doc_id": doc_id}, limit=1)
    return len(result["ids"]) > 0


def delete_document(doc_id: str) -> int:
    """Remove every chunk of a document. Returns how many chunks were removed."""
    _keyword_cache.clear()
    collection = get_collection()
    ids = collection.get(where={"doc_id": doc_id})["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def _build_filter(doc_ids: list[str] | None, page_number: int | None):
    conditions = []
    if doc_ids:
        conditions.append({"doc_id": {"$in": doc_ids}})
    if page_number is not None:
        conditions.append({"page_number": page_number})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def _dense_search(collection, query, n_results, where):
    """Semantic search. Returns dict id -> hit, ordered best first."""
    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return [
        (id_, _to_hit(text, meta, 1 - distance))
        for id_, text, meta, distance in zip(
            result["ids"][0], result["documents"][0],
            result["metadatas"][0], result["distances"][0],
        )
    ]


def _to_hit(text: str, meta: dict, score: float) -> dict:
    return {
        "doc_id": meta["doc_id"],
        "filename": meta["filename"],
        "page_number": meta["page_number"],
        "score": round(score, 4),  # cosine similarity, higher is better
        "snippet": text[:300],
        "text": text,
    }


def _keyword_ranking(collection, query, where, total) -> list[str]:
    """Chunk ids that match the query words, best first (BM25)."""
    key = repr(where)
    cached = _keyword_cache.get(key)
    if cached is None or cached[0] != total:
        data = collection.get(where=where, include=["documents"])
        cached = (total, data["ids"], data["documents"], BM25(data["documents"]))
        _keyword_cache[key] = cached
    _, ids, _, index = cached
    scored = sorted(zip(index.scores(query), ids), reverse=True)
    return [id_ for score, id_ in scored if score > 0]


def _drop_duplicates(hits: list[dict]) -> list[dict]:
    """The same text indexed twice (e.g. a re-saved copy of one PDF) must not fill
    several of the few slots we send to the LLM."""
    seen, unique = set(), []
    for hit in hits:
        fingerprint = " ".join(hit["text"].split()).lower()
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(hit)
    return unique


def search(
    query: str,
    top_k: int = 5,
    doc_ids: list[str] | None = None,
    page_number: int | None = None,
    mode: str | None = None,
) -> list[dict]:
    """Find the best passages with their page citations.
    mode "semantic" = meaning only, "hybrid" = meaning + keywords (default)."""
    mode = mode or config.SEARCH_MODE
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []
    where = _build_filter(doc_ids, page_number)

    pool = min(max(top_k * 4, 20), total)
    dense = _dense_search(collection, query, pool, where)
    if mode != "hybrid":
        return _drop_duplicates([hit for _, hit in dense])[:top_k]

    hits = dict(dense)
    keyword_ids = _keyword_ranking(collection, query, where, total)[:pool]

    # keyword-only matches still need a semantic score (used for the MIN_SCORE check)
    missing = [i for i in keyword_ids if i not in hits]
    if missing:
        query_vector = embed_query(query)
        data = collection.get(ids=missing, include=["documents", "metadatas", "embeddings"])
        for id_, text, meta, vector in zip(
            data["ids"], data["documents"], data["metadatas"], data["embeddings"]
        ):
            score = sum(float(a) * float(b) for a, b in zip(query_vector, vector))
            hits[id_] = _to_hit(text, meta, score)

    ranked = reciprocal_rank_fusion([[i for i, _ in dense], keyword_ids])
    return _drop_duplicates([hits[i] for i in ranked])[:top_k]
