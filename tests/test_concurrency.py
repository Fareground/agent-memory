"""Concurrent-writer safety: no store may ever lose a version silently."""

from __future__ import annotations

import threading

from fg_agent_memory import Provenance, RecordType
from fg_agent_memory.records import MemoryRecord
from fg_agent_memory.stores import SQLiteRecordStore, SQLiteSearchIndex
from fg_agent_memory.stores.file_store import FileRecordStore

_WRITERS = 8
_PUTS_PER_WRITER = 5


def _record() -> MemoryRecord:
    return MemoryRecord.create(
        RecordType.FACT,
        "A contested memory.",
        Provenance(kind="conversation"),
    )


def _hammer(store, record: MemoryRecord) -> list[BaseException]:
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            for _ in range(_PUTS_PER_WRITER):
                store.put(record)
        except BaseException as exc:  # noqa: BLE001 — surfacing every failure
            errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(_WRITERS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


def test_file_store_concurrent_puts_lose_no_versions(tmp_path) -> None:
    store = FileRecordStore(tmp_path / "records")
    record = _record()
    errors = _hammer(store, record)
    assert not errors
    assert len(store.history(record.id)) == _WRITERS * _PUTS_PER_WRITER


def test_sqlite_store_concurrent_puts_lose_no_versions(tmp_path) -> None:
    store = SQLiteRecordStore(tmp_path / "memory.db")
    record = _record()
    errors = _hammer(store, record)
    assert not errors
    assert len(store.history(record.id)) == _WRITERS * _PUTS_PER_WRITER
    store.close()


def test_sqlite_store_two_connections_same_file(tmp_path) -> None:
    """Two processes' worth of connections to one database file: the version
    read-modify-write must serialize, never clobber or crash unretried."""
    path = tmp_path / "memory.db"
    first, second = SQLiteRecordStore(path), SQLiteRecordStore(path)
    record = _record()
    errors = _hammer(first, record) + _hammer(second, record)
    assert not errors
    assert len(first.history(record.id)) == 2 * _WRITERS * _PUTS_PER_WRITER
    first.close()
    second.close()


def test_sqlite_index_is_usable_across_threads(tmp_path) -> None:
    index = SQLiteSearchIndex(tmp_path / "index.sqlite")
    record = _record()
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(_PUTS_PER_WRITER):
                index.index(record)
                index.candidates("contested memory", 4)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(_WRITERS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert index.candidates("contested", 4)[0][0] == record.id
    index.close()
