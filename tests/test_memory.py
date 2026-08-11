"""Porcelain Memory API: default experience, dedup, cadence, portability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from fg_agent_id import KeyPair, SignatureError, canonical_json

from fg_agent_memory import Memory, MemoryFile, RecordError


@pytest.fixture
def identity() -> KeyPair:
    return KeyPair(
        signing_key=Ed25519PrivateKey.from_private_bytes(bytes(range(32, 64))),
        agreement_key=X25519PrivateKey.from_private_bytes(bytes(range(64, 96))),
    )


class TestDefaultExperience:
    def test_default_path_writes_a_readable_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "memory"
        memory = Memory(root)
        result = memory.remember("Sandro lives in Austin.")

        record_files = list((root / "records").rglob("*.json"))
        assert len(record_files) == 1
        on_disk = json.loads(record_files[0].read_text())
        assert on_disk["id"] == result.record_id
        assert on_disk["body"] == "Sandro lives in Austin."
        assert on_disk["state"] == "active"
        assert (root / "index.sqlite").exists()

    def test_recall_finds_what_was_remembered(self, tmp_path: Path) -> None:
        memory = Memory(tmp_path / "memory")
        memory.remember("The API key rotation happens every Friday.")
        result = memory.recall("when does key rotation happen?")
        assert "every Friday" in result.as_prompt_block()
        assert len(result) == 1

    def test_empty_recall_renders_cleanly(self, tmp_path: Path) -> None:
        memory = Memory(tmp_path / "memory")
        assert memory.recall("anything").as_prompt_block() == "(no relevant memories)"

    def test_remember_rejects_empty_text(self, tmp_path: Path) -> None:
        memory = Memory(tmp_path / "memory")
        with pytest.raises(ValueError):
            memory.remember("   ")

    def test_reopening_the_directory_keeps_the_memory(self, tmp_path: Path) -> None:
        root = tmp_path / "memory"
        Memory(root).remember("The retro board lives in Linear.")
        reopened = Memory(root)
        assert "Linear" in reopened.recall("retro board").as_prompt_block()

    def test_tags_and_source_land_on_the_record(self, tmp_path: Path) -> None:
        memory = Memory(tmp_path / "memory")
        result = memory.remember(
            "Deploys must go through CI.", tags=["policy"], source="onboarding-doc"
        )
        record = memory.store.get(result.record_id)
        assert record.tags == ("policy",)
        assert record.provenance.ref == "onboarding-doc"


class TestDedupAndCadence:
    def test_same_content_reinforces_instead_of_duplicating(self, tmp_path: Path) -> None:
        memory = Memory(tmp_path / "memory")
        first = memory.remember("The standup is at 9:30.")
        second = memory.remember("The standup is at 9:30.")
        assert second.deduped and not first.deduped
        assert second.record_id == first.record_id
        assert memory.status()["records"] == 1

    def test_auto_consolidate_cadence(self, tmp_path: Path) -> None:
        memory = Memory(tmp_path / "memory", auto_consolidate_every=3)
        results = [memory.remember(f"Note number {i} is stored.") for i in range(1, 6)]
        # Consolidation fires on the 3rd accepted remember, then not again
        # until three more accumulate.
        assert [r.consolidation is not None for r in results] == [
            False, False, True, False, False,
        ]

    def test_auto_consolidate_can_be_disabled(self, tmp_path: Path) -> None:
        memory = Memory(tmp_path / "memory", auto_consolidate=False, auto_consolidate_every=1)
        result = memory.remember("Solo note.")
        assert result.consolidation is None

    def test_dedup_does_not_advance_the_cadence(self, tmp_path: Path) -> None:
        memory = Memory(tmp_path / "memory", auto_consolidate_every=2)
        memory.remember("Only one distinct thing.")
        repeat = memory.remember("Only one distinct thing.")
        assert repeat.consolidation is None


class TestResolveGuards:
    def test_resolve_requires_a_real_dispute(self, tmp_path: Path) -> None:
        memory = Memory(tmp_path / "memory")
        a = memory.remember("The cache TTL is one hour.")
        b = memory.remember("Deploy window opens at noon.")
        with pytest.raises(RecordError):
            memory.resolve(a.record_id, b.record_id, "not actually disputed")

    def test_resolve_requires_a_reason_and_distinct_sides(self, tmp_path: Path) -> None:
        memory = Memory(tmp_path / "memory")
        a = memory.remember("The feature flag is on.")
        with pytest.raises(ValueError):
            memory.resolve(a.record_id, a.record_id, "")
        with pytest.raises(RecordError):
            memory.resolve(a.record_id, a.record_id, "same record twice")


class TestPortability:
    def test_export_load_round_trip_with_signature(
        self, tmp_path: Path, identity: KeyPair
    ) -> None:
        memory = Memory(tmp_path / "memory", identity=identity)
        memory.remember("The billing service owns invoices.")
        memory.remember("The ledger service owns balances.")

        artifact = tmp_path / "brain.json"
        exported = memory.export(artifact)
        assert exported.signature and exported.agent
        exported.verify()  # file-level signature verifies
        for record in exported.records:
            record.verify()  # every record individually signed

        loaded = Memory.load(artifact)
        assert loaded.status()["records"] == 2
        assert "invoices" in loaded.recall("who owns invoices?").as_prompt_block()

    def test_load_rejects_a_tampered_file(self, tmp_path: Path, identity: KeyPair) -> None:
        memory = Memory(tmp_path / "memory", identity=identity)
        memory.remember("Only the truth is signed.")
        artifact = tmp_path / "brain.json"
        memory.export(artifact)

        # Rewriting a body breaks its content address.
        data = json.loads(artifact.read_text())
        data["records"][0]["body"] = "A quietly rewritten memory."
        artifact.write_text(json.dumps(data))
        with pytest.raises(RecordError):
            Memory.load(artifact)

        # Re-stamping the export time breaks the file-level signature.
        data = json.loads(artifact.read_text())
        data["records"][0]["body"] = "Only the truth is signed."
        data["created_at"] = "2020-01-01T00:00:00.000Z"
        artifact.write_text(json.dumps(data))
        with pytest.raises(SignatureError):
            Memory.load(artifact)

    def test_unsigned_export_loads_only_with_explicit_opt_out(self, tmp_path: Path) -> None:
        memory = Memory(tmp_path / "memory")
        memory.remember("Unsigned but portable.")
        artifact = tmp_path / "plain.json"
        exported = memory.export(artifact)
        assert exported.signature == ""
        assert isinstance(exported, MemoryFile)
        # Default verify=True refuses an unsigned file outright.
        with pytest.raises(RecordError):
            Memory.load(artifact, path=tmp_path / "second")
        loaded = Memory.load(artifact, path=tmp_path / "second", verify=False)
        assert loaded.status()["records"] == 1
        assert ((tmp_path / "second") / "records").exists()

    def test_load_rejects_a_stripped_signature(self, tmp_path: Path, identity: KeyPair) -> None:
        """Stripping the signature must not silently downgrade verification —
        the signature-downgrade forgery vector."""
        memory = Memory(tmp_path / "memory", identity=identity)
        memory.remember("Only the truth is signed.")
        artifact = tmp_path / "brain.json"
        memory.export(artifact)
        data = json.loads(artifact.read_text())
        data["signature"] = ""
        artifact.write_text(json.dumps(data))
        with pytest.raises(RecordError):
            Memory.load(artifact)

    def test_load_rejects_a_forged_record_inside_a_signed_file(
        self, tmp_path: Path, identity: KeyPair
    ) -> None:
        """A record claiming a signer it was not signed by must fail on load."""
        memory = Memory(tmp_path / "memory", identity=identity)
        memory.remember("Only the truth is signed.")
        artifact = tmp_path / "brain.json"
        memory.export(artifact)
        data = json.loads(artifact.read_text())
        # Corrupt one record's signature while leaving its content address
        # intact; re-sign the file wrapper so only record verification can
        # catch it.
        data["records"][0]["signature"] = "A" + data["records"][0]["signature"][1:]
        tampered = MemoryFile.model_validate(data).sign(identity)
        artifact.write_bytes(canonical_json(tampered.model_dump()))
        with pytest.raises((RecordError, SignatureError)):
            Memory.load(artifact)


class TestBackgroundPasses:
    def test_decay_reports_without_deleting(self, tmp_path: Path) -> None:
        memory = Memory(tmp_path / "memory")
        result = memory.remember("Fresh memories do not decay.")
        report = memory.decay()
        assert report.archived == ()
        assert result.record_id in report.saliences

    def test_status_counts_open_contradictions(self, tmp_path: Path) -> None:
        memory = Memory(tmp_path / "memory")
        memory.remember("The linter is strict.")
        memory.remember("The linter is not strict.")
        memory.consolidate()
        status = memory.status()
        assert status["open_contradictions"] == 1
        assert status["by_state"]["transitional"] == 2


class TestDedupScaling:
    def test_repeat_remember_with_new_tags_merges_them(self, tmp_path: Path) -> None:
        memory = Memory(tmp_path / "memory")
        first = memory.remember("Sandro uses Zed.")
        second = memory.remember("Sandro uses Zed.", tags=("editor",))
        assert second.deduped and second.record_id == first.record_id
        assert "editor" in memory.store.get(first.record_id).tags
        # Tag merge is a new version of the same record, never a new record.
        assert len(memory.store.history(first.record_id)) == 2

    def test_dedup_survives_consolidation_state_changes(self, tmp_path: Path) -> None:
        memory = Memory(tmp_path / "memory", auto_consolidate=False)
        memory.remember("The deploy needs approval.")
        memory.remember("The deploy needs no approval.")
        memory.consolidate()  # both sides now transitional, not active
        result = memory.remember("The deploy needs approval.")
        # The transitional record is no longer live truth; remembering the
        # same text again must not silently dedup against it.
        assert not result.deduped


class TestKeyfileIdentity:
    """``identity="agent.key"`` — string/Path keyfile convenience."""

    def test_string_path_creates_keyfile_and_signs(self, tmp_path: Path) -> None:
        keyfile = tmp_path / "agent.key"
        memory = Memory(tmp_path / "memory", identity=str(keyfile))
        assert keyfile.exists()

        memory.remember("Signed on first run.")
        exported = memory.export(tmp_path / "brain.json")
        assert exported.signature and exported.agent
        exported.verify()

    def test_same_keyfile_yields_same_agent_address(self, tmp_path: Path) -> None:
        keyfile = tmp_path / "agent.key"
        first = Memory(tmp_path / "m1", identity=keyfile)
        second = Memory(tmp_path / "m2", identity=keyfile)

        first.remember("One agent.")
        second.remember("Two directories.")
        a = first.export(tmp_path / "a.json")
        b = second.export(tmp_path / "b.json")
        assert a.agent == b.agent
