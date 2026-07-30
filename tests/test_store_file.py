"""FileRecordStore: the port contract suite plus on-disk layout guarantees."""

import json

import pytest

from fg_agent_memory import StoreError, archive
from fg_agent_memory.stores import FileRecordStore

from . import test_ports as contract
from .conftest import make_record


@pytest.fixture
def store(tmp_path):
    return FileRecordStore(tmp_path / "memory")


# The full RecordStore contract and the proposal pipeline, re-collected here
# against the file-backed store (the local `store` fixture overrides the
# parametrized one in test_ports).
test_contract_put_get_roundtrip = contract.test_store_put_get_roundtrip
test_contract_get_unknown_raises = contract.test_store_get_unknown_raises
test_contract_is_append_only = contract.test_store_is_append_only
test_contract_list_filters = contract.test_store_list_filters
test_contract_apply_create = contract.test_apply_create
test_contract_apply_supersede = contract.test_apply_supersede
test_contract_apply_supersede_validates = contract.test_apply_supersede_validates
test_contract_apply_transition_and_archive = contract.test_apply_transition_and_archive
test_contract_apply_rejects_illegal = contract.test_apply_rejects_illegal_lifecycle_without_writing
test_contract_apply_rejects_malformed = contract.test_apply_rejects_malformed_proposals
test_contract_apply_proposals_in_order = contract.test_apply_proposals_in_order


def test_layout_is_versioned_files_under_stable_path(tmp_path, provenance):
    store = FileRecordStore(tmp_path / "memory")
    record = make_record(provenance)
    store.put(record)
    record_dir = tmp_path / "memory" / record.id.removeprefix("mem:")
    assert [p.name for p in sorted(record_dir.iterdir())] == ["000001.json"]

    store.put(archive(record, reason="decay"))
    # Append-only on disk: a lifecycle change adds a file, never rewrites one.
    assert [p.name for p in sorted(record_dir.iterdir())] == ["000001.json", "000002.json"]
    first = json.loads((record_dir / "000001.json").read_bytes())
    assert first["state"] == "active"  # the original bytes are untouched


def test_files_are_canonical_json_wire_form(tmp_path, provenance):
    store = FileRecordStore(tmp_path / "memory")
    record = make_record(provenance)
    store.put(record)
    record_dir = tmp_path / "memory" / record.id.removeprefix("mem:")
    on_disk = json.loads((record_dir / "000001.json").read_bytes())
    assert on_disk == record.model_dump()


def test_no_temp_files_left_behind(tmp_path, provenance):
    store = FileRecordStore(tmp_path / "memory")
    record = make_record(provenance)
    store.put(record)
    store.put(archive(record, reason="decay"))
    leftovers = [p for p in (tmp_path / "memory").rglob("*") if p.name.startswith(".tmp-")]
    assert leftovers == []


def test_reopen_sees_everything(tmp_path, provenance):
    root = tmp_path / "memory"
    first = FileRecordStore(root)
    record = make_record(provenance)
    first.put(record)
    first.put(archive(record, reason="decay"))

    reopened = FileRecordStore(root)
    assert reopened.get(record.id).state.value == "archived"
    assert len(reopened.history(record.id)) == 2
    assert len(reopened.list()) == 1


def test_malformed_id_rejected(store):
    for bad in ("", "mem:", "not-an-id", "mem:../escape", "mem:ABCDEF"):
        with pytest.raises(StoreError, match="malformed|unknown"):
            store.get(bad)
