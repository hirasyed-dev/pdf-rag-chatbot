"""Shared test helpers. Tests use a tiny fake embedding so they run fast and
never download the real model."""
import math

import pytest

from app import config, vector_store


def fake_embed(text: str) -> list[float]:
    vector = [0.0] * 26
    for ch in text.lower():
        if "a" <= ch <= "z":
            vector[ord(ch) - 97] += 1
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point all storage at a temporary folder and use fake embeddings."""
    monkeypatch.setattr(config, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(config, "CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "SQLITE_PATH", str(tmp_path / "meta.db"))
    monkeypatch.setattr(config, "LIBRARY_DIR", str(tmp_path / "library"))
    monkeypatch.setattr(vector_store, "_client", None)
    monkeypatch.setattr(
        vector_store, "embed_documents", lambda texts: [fake_embed(t) for t in texts]
    )
    monkeypatch.setattr(vector_store, "embed_query", fake_embed)
    return tmp_path
