"""ChromaDB storage. Every chunk is saved with doc_id, filename and page_number
so that search results can always be cited and filtered."""
import chromadb

from app import config
from app.embeddings import embed_documents, embed_query

COLLECTION_NAME = "pdf_chunks"

_client = None


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


def search(
    query: str,
    top_k: int = 5,
    doc_ids: list[str] | None = None,
    page_number: int | None = None,
) -> list[dict]:
    """Semantic search. Returns the best passages with their page citations."""
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=min(top_k, total),
        where=_build_filter(doc_ids, page_number),
    )

    hits = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "doc_id": meta["doc_id"],
                "filename": meta["filename"],
                "page_number": meta["page_number"],
                "score": round(1 - distance, 4),  # cosine similarity, higher is better
                "snippet": text[:300],
                "text": text,
            }
        )
    return hits
