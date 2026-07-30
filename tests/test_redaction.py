"""§11 redaction: governed content destruction with a permanent tombstone."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fg_agent_memory import (
    InMemoryRecordStore,
    Memory,
    Provenance,
    RecordError,
    RecordState,
    RecordType,
    StoreError,
)
from fg_agent_memory.records import MemoryRecord
from fg_agent_memory.stores import SQLiteRecordStore

SECRET = "The prod API key is sk-live-abc123."


def _stores(tmp_path):
    return (InMemoryRecordStore(), SQLiteRecordStore(tmp_path / "memory.db"))


class TestTombstoneRecord:
    def test_tombstone_keeps_identity_and_destroys_content(self) -> None:
        prov = Provenance(kind="conversation", ref="chat-42")
        original = MemoryRecord.create(RecordType.FACT, SECRET, prov, tags=("secret",))
        tomb = MemoryRecord.tombstone(original, reason="leaked credential")
        assert tomb.id == original.id
        assert tomb.state is RecordState.ARCHIVED
        assert tomb.redacted and tomb.redaction_reason == "leaked credential"
        assert tomb.birth_digest and len(tomb.birth_digest) == 64
        wire = json.dumps(tomb.model_dump())
        assert "sk-live" not in wire and "chat-42" not in wire and "secret" not in wire

    def test_tombstone_round_trips_on_the_wire(self) -> None:
        prov = Provenance(kind="conversation")
        tomb = MemoryRecord.tombstone(
            MemoryRecord.create(RecordType.FACT, SECRET, prov), reason="cleanup"
        )
        parsed = MemoryRecord.model_validate(tomb.model_dump())
        assert parsed == tomb

    def test_tombstone_invariants_are_enforced(self) -> None:
        prov = Provenance(kind="conversation")
        original = MemoryRecord.create(RecordType.FACT, SECRET, prov)
        with pytest.raises(RecordError):
            MemoryRecord.tombstone(original, reason="")
        tomb = MemoryRecord.tombstone(original, reason="once")
        with pytest.raises(RecordError):
            MemoryRecord.tombstone(tomb, reason="twice")
        # Redaction fields on a non-redacted record are rejected outright.
        with pytest.raises(RecordError):
            MemoryRecord.create(RecordType.FACT, "ok", prov, extra={}).model_copy(
                update={"birth_digest": "ab" * 32}
            )

    def test_non_redacted_wire_form_is_unchanged(self) -> None:
        prov = Provenance(kind="conversation")
        record = MemoryRecord.create(RecordType.FACT, "Plain fact.", prov)
        assert "redacted" not in record.model_dump()


class TestStoreRedact:
    def test_every_version_is_rewritten(self, tmp_path) -> None:
        from fg_agent_memory.lifecycle import archive

        for store in _stores(tmp_path):
            prov = Provenance(kind="conversation")
            record = MemoryRecord.create(RecordType.FACT, SECRET, prov)
            store.put(record)
            store.put(archive(record, reason="decay"))
            tomb = store.redact(record.id, "leaked credential")
            history = store.history(record.id)
            assert len(history) == 2  # version count preserved
            for version in history:
                assert version == tomb
                assert "sk-live" not in version.body
            assert store.get(record.id).redacted

    def test_file_store_leaves_no_original_bytes_on_disk(self, tmp_path: Path) -> None:
        memory = Memory(tmp_path / "memory", auto_consolidate=False)
        result = memory.remember(SECRET)
        memory.redact(result.record_id, "leaked credential")
        on_disk = b"".join(
            path.read_bytes()
            for path in (tmp_path / "memory" / "records").rglob("*.json")
        )
        assert b"sk-live" not in on_disk

    def test_redact_unknown_id_raises(self, tmp_path) -> None:
        for store in _stores(tmp_path):
            with pytest.raises(StoreError):
                store.redact("mem:" + "0" * 32, "nothing there")


class TestPorcelainRedact:
    def test_redacted_content_leaves_recall(self, tmp_path: Path) -> None:
        memory = Memory(tmp_path / "memory", auto_consolidate=False)
        result = memory.remember(SECRET)
        assert "sk-live" in memory.recall("prod API key").as_prompt_block()
        memory.redact(result.record_id, "leaked credential")
        assert "sk-live" not in memory.recall("prod API key").as_prompt_block()
        # Even history-inclusive recall never resurrects destroyed content.
        block = memory.recall(
            "prod API key", include_history=True, include_archived=True
        ).as_prompt_block()
        assert "sk-live" not in block

    def test_redacting_rule_evidence_flags_the_rule(self, tmp_path: Path) -> None:
        store = InMemoryRecordStore()
        prov = Provenance(kind="observation")
        episode = MemoryRecord.create(RecordType.EPISODE, "Observed the secret.", prov)
        store.put(episode)
        rule = MemoryRecord.create(
            RecordType.RULE,
            "Derived from the secret episode.",
            Provenance(kind="consolidation"),
            evidence=(episode.id,),
        )
        store.put(rule)
        memory = Memory(store=store, auto_consolidate=False)
        memory.redact(episode.id, "sensitive observation")
        memory.consolidate()
        assert memory.store.get(rule.id).flagged is True

    def test_redact_requires_reason_and_store_support(self, tmp_path: Path) -> None:
        memory = Memory(tmp_path / "memory")
        result = memory.remember("Something harmless.")
        with pytest.raises(ValueError):
            memory.redact(result.record_id, "")

        class NoRedactStore(InMemoryRecordStore):
            redact = None

        bare = Memory(store=NoRedactStore(), auto_consolidate=False)
        stored = bare.remember("Held hostage.")
        with pytest.raises(StoreError):
            bare.redact(stored.record_id, "cannot")

    def test_tombstones_survive_export_and_load(self, tmp_path: Path) -> None:
        memory = Memory(tmp_path / "memory", auto_consolidate=False)
        kept = memory.remember("A perfectly fine memory.")
        gone = memory.remember(SECRET)
        memory.redact(gone.record_id, "leaked credential")
        artifact = tmp_path / "brain.json"
        memory.export(artifact)
        assert b"sk-live" not in artifact.read_bytes()
        loaded = Memory.load(artifact, verify=False)
        assert loaded.store.get(gone.record_id).redacted
        assert loaded.store.get(kept.record_id).body == "A perfectly fine memory."
