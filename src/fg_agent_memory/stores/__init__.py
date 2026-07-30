"""Persistent storage adapters for the RecordStore and SearchIndex ports.

Two backends ship with the reference, both stdlib-only:

- :class:`FileRecordStore` — a readable directory of canonical-JSON files
  you can commit to git. The default when memory should be inspectable.
- :class:`SQLiteRecordStore` — a single-file SQLite database with queryable
  state/type/tags, plus :class:`SQLiteSearchIndex`, an FTS5 keyword index.

Both pass the same port contract suite as the in-memory references —
the standard is the record format and lifecycle, never the backend.
"""

from .file_store import FileRecordStore
from .sqlite_index import SQLiteSearchIndex
from .sqlite_store import SQLiteRecordStore

__all__ = ["FileRecordStore", "SQLiteRecordStore", "SQLiteSearchIndex"]
