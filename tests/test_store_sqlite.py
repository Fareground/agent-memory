"""SQLiteRecordStore and SQLiteSearchIndex: contract suite plus FTS behavior."""

import pytest

from fg_agent_memory import RecordState, archive
from fg_agent_memory.stores import SQLiteRecordStore, SQLiteSearchIndex

from . import test_ports as contract
from .conftest import make_record


@pytest.fixture
def store():
    return SQLiteRecordStore()


@pytest.fixture
def index():
    return SQLiteSearchIndex()


# The full RecordStore contract, the SearchIndex contract, and the proposal
# pipeline, re-collected against the SQLite implementations (the local
# fixtures override the parametrized ones in test_ports).
test_contract_put_get_roundtrip = contract.test_store_put_get_roundtrip
test_contract_get_unknown_raises = contract.test_store_get_unknown_raises
test_contract_is_append_only = contract.test_store_is_append_only
test_contract_list_filters = contract.test_store_list_filters
test_contract_index_ranks_overlap_first = contract.test_index_ranks_overlapping_text_first
test_contract_apply_create = contract.test_apply_create
test_contract_apply_supersede = contract.test_apply_supersede
test_contract_apply_supersede_validates = contract.test_apply_supersede_validates
test_contract_apply_transition_and_archive = contract.test_apply_transition_and_archive
test_contract_apply_rejects_illegal = contract.test_apply_rejects_illegal_lifecycle_without_writing
test_contract_apply_rejects_malformed = contract.test_apply_rejects_malformed_proposals
test_contract_apply_proposals_in_order = contract.test_apply_proposals_in_order


def test_store_persists_across_reopen(tmp_path, provenance):
    path = tmp_path / "memory.db"
    first = SQLiteRecordStore(path)
    record = make_record(provenance, tags=("alpha",))
    first.put(record)
    first.put(archive(record, reason="decay"))
    first.close()

    reopened = SQLiteRecordStore(path)
    assert reopened.get(record.id).state is RecordState.ARCHIVED
    assert len(reopened.history(record.id)) == 2
    assert [r.id for r in reopened.list(tag="alpha")] == [record.id]
    reopened.close()


def test_index_matches_tags_not_just_body(index, provenance):
    tagged = make_record(provenance, body="Nothing notable here.", tags=("deployment",))
    plain = make_record(provenance, body="Also nothing notable.")
    index.index(tagged)
    index.index(plain)
    results = index.candidates("deployment", k=2)
    assert [record_id for record_id, _ in results] == [tagged.id]
    assert results[0][1] > 0.0


def test_reindex_replaces_previous_row(index, provenance):
    record = make_record(provenance, body="The cat sat on the mat.")
    index.index(record)
    index.index(record)  # idempotent, not duplicated
    results = index.candidates("cat", k=5)
    assert [record_id for record_id, _ in results] == [record.id]


def test_unmatched_ids_are_not_returned(index, provenance):
    match = make_record(provenance, body="Solar panels on the roof.")
    miss = make_record(provenance, body="Quarterly revenue numbers.")
    index.index(match)
    index.index(miss)
    results = index.candidates("solar roof", k=2)
    # Only actual matches come back — retrieval discards non-positive
    # relevance, so padding the tail would cost a scan and buy nothing.
    assert [record_id for record_id, _ in results] == [match.id]
    assert results[0][1] > 0.0


def test_index_ignores_fts_query_syntax(index, provenance):
    record = make_record(provenance, body="A perfectly normal fact.")
    index.index(record)
    # Raw FTS operators/quotes must be treated as text, never as syntax.
    assert index.candidates('normal" OR record_id:*', k=1)[0][0] == record.id
    assert index.candidates("(((", k=1) == ()
    assert index.candidates("", k=1) == ()
