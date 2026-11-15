"""SQLite table that tracks uploaded documents (name, pages, chunks, status)."""
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from app import config


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(config.SQLITE_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id      TEXT PRIMARY KEY,
            filename    TEXT NOT NULL,
            page_count  INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL,
            source      TEXT NOT NULL,
            status      TEXT NOT NULL,
            created_at  TEXT NOT NULL
        )
        """
    )
    return conn


def upsert_document(
    doc_id: str, filename: str, page_count: int, chunk_count: int, source: str
) -> None:
    with closing(_connect()) as conn:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id,
                    filename,
                    page_count,
                    chunk_count,
                    source,
                    "indexed",
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )


def get_document(doc_id: str) -> dict | None:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
    return dict(row) if row else None


def list_documents() -> list[dict]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_document(doc_id: str) -> bool:
    with closing(_connect()) as conn:
        with conn:
            cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    return cursor.rowcount > 0
