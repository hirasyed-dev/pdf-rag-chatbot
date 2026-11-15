"""Local embeddings with sentence-transformers (no API key, text stays on this machine)."""
from app import config

# bge models work best when search queries carry this prefix (documents do not)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def get_model():
    """Load the model once and reuse it, because loading is slow."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # imported here so tests stay fast

        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    vectors = get_model().encode(
        texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    vector = get_model().encode(
        QUERY_PREFIX + query, normalize_embeddings=True, show_progress_bar=False
    )
    return vector.tolist()
