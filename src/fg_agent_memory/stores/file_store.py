"""FileRecordStore: a readable directory of records you can commit to git.

Layout — one directory per record id, one canonical-JSON file per version:

    <root>/
      <id-hex>/
        000001.json     # first version (birth)
        000002.json     # each later put appends the next number
        ...

This layout is chosen for git-friendliness: paths are stable (derived from
the content-addressed id), history is append-only (a lifecycle change adds a
new file instead of rewriting an old one, so diffs are pure additions), and
no index file exists to churn on every write. Files are canonical JSON, so
the bytes on disk are exactly the signed wire form.

Writes are atomic AND concurrent-writer safe: content goes to a temp file in
the same directory (fsynced before it becomes visible), then ``os.link``
claims the next version filename — link fails if another writer already took
that number, so the version is recounted and retried rather than silently
clobbered. A crash mid-write leaves at worst a stray ``.tmp-*`` file, never
a torn or lost version.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

from fg_agent_id import canonical_json

from ..errors import StoreError
from ..ports import RecordStore
from ..records import MemoryRecord, RecordState, RecordType

# Version filenames are zero-padded so lexicographic order IS version order.
_VERSION_DIGITS = 6
_VERSION_FILE = re.compile(rf"^\d{{{_VERSION_DIGITS}}}\.json$")
# Record ids are `mem:` + lowercase hex; only the hex names the directory,
# which also guarantees the derived path can never escape the root.
_ID_PATTERN = re.compile(r"^mem:([0-9a-f]+)$")


def _record_dirname(record_id: str) -> str:
    match = _ID_PATTERN.match(record_id)
    if not match:
        raise StoreError(f"malformed record id: {record_id!r}")
    return match.group(1)


class FileRecordStore(RecordStore):
    """Append-only record store over a plain directory. See module docstring
    for the layout; passes the same contract suite as every other store."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def put(self, record: MemoryRecord) -> None:
        record_dir = self._root / _record_dirname(record.id)
        record_dir.mkdir(exist_ok=True)
        payload = canonical_json(record.model_dump())
        tmp = record_dir / f".tmp-{uuid.uuid4().hex}"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.write(fd, payload)
            os.fsync(fd)  # data durable before the version becomes visible
        finally:
            os.close(fd)
        try:
            while True:
                version = len(self._version_files(record_dir)) + 1
                final = record_dir / f"{version:0{_VERSION_DIGITS}d}.json"
                try:
                    # link (not replace): claiming a version number fails if a
                    # concurrent writer took it first, so no version is ever
                    # silently overwritten — recount and retry instead.
                    os.link(tmp, final)
                    break
                except FileExistsError:
                    continue
        finally:
            os.unlink(tmp)
        dir_fd = os.open(record_dir, os.O_RDONLY)
        try:
            os.fsync(dir_fd)  # the new directory entry survives power loss
        finally:
            os.close(dir_fd)

    def get(self, record_id: str) -> MemoryRecord:
        files = self._existing_version_files(record_id)
        return self._load(files[-1])

    def history(self, record_id: str) -> tuple[MemoryRecord, ...]:
        files = self._existing_version_files(record_id)
        return tuple(self._load(path) for path in files)

    def list(
        self,
        *,
        state: RecordState | None = None,
        type: RecordType | None = None,
        tag: str | None = None,
    ) -> tuple[MemoryRecord, ...]:
        latest = []
        for record_dir in self._root.iterdir():
            files = self._version_files(record_dir)
            if not files:
                continue
            latest.append(self._load(files[-1]))
        # The filesystem keeps no insertion order, so listing order is
        # deterministic instead: birth time, then id.
        latest.sort(key=lambda r: (r.created_at, r.id))
        out = []
        for record in latest:
            if state is not None and record.state is not RecordState(state):
                continue
            if type is not None and record.type is not RecordType(type):
                continue
            if tag is not None and tag not in record.tags:
                continue
            out.append(record)
        return tuple(out)

    def redact(self, record_id: str, reason: str) -> MemoryRecord:
        """§11 governed redaction: rewrite every version file of the record
        with one tombstone. The only operation that ever rewrites a stored
        file — content destroyed, existence and version count preserved."""
        tomb = MemoryRecord.tombstone(self.get(record_id), reason=reason)
        payload = canonical_json(tomb.model_dump())
        record_dir = self._root / _record_dirname(record_id)
        for version_file in self._version_files(record_dir):
            tmp = record_dir / f".tmp-{uuid.uuid4().hex}"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                os.write(fd, payload)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, version_file)
        dir_fd = os.open(record_dir, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        return tomb

    def _existing_version_files(self, record_id: str) -> list[Path]:
        record_dir = self._root / _record_dirname(record_id)
        files = self._version_files(record_dir) if record_dir.is_dir() else []
        if not files:
            raise StoreError(f"unknown record id: {record_id}")
        return files

    @staticmethod
    def _version_files(record_dir: Path) -> list[Path]:
        if not record_dir.is_dir():
            return []
        return sorted(p for p in record_dir.iterdir() if _VERSION_FILE.match(p.name))

    @staticmethod
    def _load(path: Path) -> MemoryRecord:
        try:
            data = json.loads(path.read_bytes())
        except (OSError, ValueError) as exc:
            raise StoreError(
                f"unreadable record file {path}: {exc} — restore the file "
                "from backup, or remove this version file to drop the record"
            ) from exc
        return MemoryRecord.model_validate(data)
