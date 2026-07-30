"""SQLiteSearchIndex: FTS5 keyword search over record body and tags.

A SearchIndex implementation with zero model dependency — BM25 over the
tokens of ``body`` and ``tags``. Re-indexing a record id replaces its row.
Scores are ``-bm25`` (higher is better, matching the port convention); only
actual matches are returned — retrieval discards non-positive relevance, so
padding the tail with zero-score ids would cost a full-table scan and buy
nothing.

FTS5 ships with the standard sqlite3 build on every mainstream platform; the
constructor raises StoreError with a clear message where it is unavailable.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path

from ..errors import StoreError
from ..ports import SearchIndex
from ..records import MemoryRecord

# Query text is reduced to plain alphanumeric tokens and OR-joined, so raw
# user text can never inject FTS5 query syntax.
_TOKEN = re.compile(r"[a-zA-Z0-9]+")


class SQLiteSearchIndex(SearchIndex):
    """Keyword SearchIndex over FTS5. See module docstring."""

    def __init__(self, path: Path | str = ":memory:") -> None:
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA busy_timeout=5000")
        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS record_fts"
                " USING fts5(record_id UNINDEXED, body, tags)"
            )
        except sqlite3.OperationalError as exc:
            raise StoreError(f"sqlite build lacks FTS5: {exc}") from exc
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def index(self, record: MemoryRecord) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM record_fts WHERE record_id = ?", (record.id,)
            )
            self._conn.execute(
                "INSERT INTO record_fts (record_id, body, tags) VALUES (?, ?, ?)",
                (record.id, record.body, " ".join(record.tags)),
            )

    def candidates(self, query: str, k: int) -> tuple[tuple[str, float], ...]:
        if k <= 0:
            return ()
        tokens = _TOKEN.findall(query)
        if not tokens:
            return ()
        fts_query = " OR ".join(f'"{token}"' for token in tokens)
        with self._lock:
            rows = self._conn.execute(
                "SELECT record_id, -bm25(record_fts) FROM record_fts"
                " WHERE record_fts MATCH ?",
                (fts_query,),
            ).fetchall()
        # bm25 can round to exactly zero; keep every match strictly positive
        # so a match is never mistaken for irrelevance downstream.
        matched = [(record_id, max(score, 1e-9)) for record_id, score in rows]
        matched.sort(key=lambda pair: (-pair[1], pair[0]))
        return tuple(matched[:k])
