"""SQLiteRecordStore: single-file persistence with queryable metadata.

Schema — the wire form stays authoritative, columns are just query keys:

    records(record_id PK, seq, state, type, payload)   -- latest version
    record_tags(record_id, tag)                        -- latest version's tags
    history(record_id, version, payload)               -- every version, append-only

``payload`` is the canonical JSON of the record, so a row round-trips to the
exact signed bytes. ``seq`` preserves insertion order for ``list``. WAL mode
is enabled for concurrent readers; every ``put`` is one transaction, so a
version is either fully written (history + latest + tags) or absent.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from fg_agent_id import canonical_json

from ..errors import StoreError
from ..ports import RecordStore
from ..records import MemoryRecord, RecordState, RecordType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    record_id TEXT PRIMARY KEY,
    seq       INTEGER NOT NULL,
    state     TEXT NOT NULL,
    type      TEXT NOT NULL,
    payload   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS record_tags (
    record_id TEXT NOT NULL,
    tag       TEXT NOT NULL,
    PRIMARY KEY (record_id, tag)
);
CREATE TABLE IF NOT EXISTS history (
    record_id TEXT NOT NULL,
    version   INTEGER NOT NULL,
    payload   TEXT NOT NULL,
    PRIMARY KEY (record_id, version)
);
CREATE INDEX IF NOT EXISTS idx_records_state ON records(state);
CREATE INDEX IF NOT EXISTS idx_records_type ON records(type);
"""


class SQLiteRecordStore(RecordStore):
    """Append-only record store over stdlib sqlite3. See module docstring."""

    # A benign version-number race with a concurrent process retries this
    # many times before giving up loudly.
    _PUT_RETRIES = 16

    def __init__(self, path: Path | str = ":memory:") -> None:
        # One connection guarded by a lock: safe to share across threads
        # (e.g. an MCP server dispatching tool calls on a worker pool).
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def put(self, record: MemoryRecord) -> None:
        payload = canonical_json(record.model_dump()).decode()
        with self._lock:
            for _ in range(self._PUT_RETRIES):
                try:
                    # BEGIN IMMEDIATE takes the write lock up front, so the
                    # MAX(version) read and the insert are one atomic unit
                    # against other processes on the same file.
                    self._conn.execute("BEGIN IMMEDIATE")
                    self._write(record, payload)
                    self._conn.commit()
                    return
                except sqlite3.IntegrityError:
                    # Another writer claimed this version number between our
                    # read and insert; re-read and retry.
                    self._conn.rollback()
                except sqlite3.OperationalError as exc:
                    self._conn.rollback()
                    # Real cross-process contention surfaces here (BEGIN
                    # IMMEDIATE times out with "database is locked"), not as
                    # IntegrityError — retry it; re-raise anything else.
                    if "locked" not in str(exc).lower():
                        raise
                except BaseException:
                    self._conn.rollback()
                    raise
        raise StoreError(
            f"could not append a version for {record.id} after "
            f"{self._PUT_RETRIES} attempts (concurrent writers)"
        )

    def _write(self, record: MemoryRecord, payload: str) -> None:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM history WHERE record_id = ?",
            (record.id,),
        ).fetchone()
        self._conn.execute(
            "INSERT INTO history (record_id, version, payload) VALUES (?, ?, ?)",
            (record.id, row[0] + 1, payload),
        )
        seq_row = self._conn.execute(
            "SELECT seq FROM records WHERE record_id = ?", (record.id,)
        ).fetchone()
        if seq_row is None:
            max_seq = self._conn.execute("SELECT COALESCE(MAX(seq), 0) FROM records")
            seq = max_seq.fetchone()[0] + 1
        else:
            seq = seq_row[0]
        self._conn.execute(
            "INSERT OR REPLACE INTO records (record_id, seq, state, type, payload)"
            " VALUES (?, ?, ?, ?, ?)",
            (record.id, seq, record.state.value, record.type.value, payload),
        )
        self._conn.execute("DELETE FROM record_tags WHERE record_id = ?", (record.id,))
        self._conn.executemany(
            "INSERT OR IGNORE INTO record_tags (record_id, tag) VALUES (?, ?)",
            [(record.id, tag) for tag in record.tags],
        )

    def redact(self, record_id: str, reason: str) -> MemoryRecord:
        """§11 governed redaction: replace every history payload and the
        latest-version row with one tombstone, in one transaction — content
        destroyed, existence and version count preserved."""
        tomb = MemoryRecord.tombstone(self.get(record_id), reason=reason)
        payload = canonical_json(tomb.model_dump()).decode()
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE history SET payload = ? WHERE record_id = ?",
                (payload, record_id),
            )
            self._conn.execute(
                "UPDATE records SET state = ?, type = ?, payload = ?"
                " WHERE record_id = ?",
                (tomb.state.value, tomb.type.value, payload, record_id),
            )
            self._conn.execute(
                "DELETE FROM record_tags WHERE record_id = ?", (record_id,)
            )
        return tomb

    def get(self, record_id: str) -> MemoryRecord:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM records WHERE record_id = ?", (record_id,)
            ).fetchone()
        if row is None:
            raise StoreError(f"unknown record id: {record_id}")
        return _parse(row[0])

    def history(self, record_id: str) -> tuple[MemoryRecord, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM history WHERE record_id = ? ORDER BY version",
                (record_id,),
            ).fetchall()
        if not rows:
            raise StoreError(f"unknown record id: {record_id}")
        return tuple(_parse(row[0]) for row in rows)

    def list(
        self,
        *,
        state: RecordState | None = None,
        type: RecordType | None = None,
        tag: str | None = None,
    ) -> tuple[MemoryRecord, ...]:
        clauses, params = [], []
        if state is not None:
            clauses.append("state = ?")
            params.append(RecordState(state).value)
        if type is not None:
            clauses.append("type = ?")
            params.append(RecordType(type).value)
        if tag is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM record_tags t"
                " WHERE t.record_id = records.record_id AND t.tag = ?)"
            )
            params.append(tag)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT payload FROM records{where} ORDER BY seq", params
            ).fetchall()
        return tuple(_parse(row[0]) for row in rows)


def _parse(payload: str) -> MemoryRecord:
    try:
        data = json.loads(payload)
    except ValueError as exc:
        raise StoreError(f"corrupt record payload: {exc}") from exc
    return MemoryRecord.model_validate(data)
