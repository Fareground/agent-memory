"""Port contracts, parametrized so any implementation can run them, plus the
proposal pipeline. The in-memory reference implementations are the fixtures."""

import pytest

from fg_agent_memory import (
    HashEmbedder,
    InMemoryRecordStore,
    InMemorySearchIndex,
    LifecycleError,
    Proposal,
    ProposalError,
    ProposalKind,
    RecordState,
    RecordType,
    StoreError,
    apply_proposal,
    apply_proposals,
    archive,
)

from .conftest import make_record

# Any RecordStore/SearchIndex implementation can be added here to run the
# full contract suite against it.
STORE_FACTORIES = [InMemoryRecordStore]
INDEX_FACTORIES = [InMemorySearchIndex]


@pytest.fixture(params=STORE_FACTORIES, ids=lambda f: f.__name__)
def store(request):
    return request.param()


@pytest.fixture(params=INDEX_FACTORIES, ids=lambda f: f.__name__)
def index(request):
    return request.param()


def test_store_put_get_roundtrip(store, provenance):
    record = make_record(provenance)
    store.put(record)
    assert store.get(record.id) == record


def test_store_get_unknown_raises(store):
    with pytest.raises(StoreError):
        store.get("mem:" + "0" * 32)
    with pytest.raises(StoreError):
        store.history("mem:" + "0" * 32)


def test_store_is_append_only(store, provenance):
    record = make_record(provenance)
    store.put(record)
    archived = archive(record, reason="decay")
    store.put(archived)
    assert store.get(record.id) == archived  # latest wins
    assert store.history(record.id) == (record, archived)  # prior stays addressable


def test_store_list_filters(store, provenance):
    fact = make_record(provenance, body="A fact is here.", tags=("alpha",))
    episode = make_record(
        provenance, body="A thing happened.", type=RecordType.EPISODE, tags=("beta",)
    )
    store.put(fact)
    store.put(episode)
    store.put(archive(episode, reason="decay"))
    assert list(store.list(type=RecordType.FACT)) == [fact]
    assert list(store.list(state=RecordState.ACTIVE)) == [fact]
    assert [r.id for r in store.list(state=RecordState.ARCHIVED)] == [episode.id]
    assert list(store.list(tag="alpha")) == [fact]
    assert store.list(state=RecordState.ACTIVE, tag="beta") == ()
    assert len(store.list()) == 2
    assert tuple(iter(store)) == store.list()


def test_hash_embedder_deterministic_and_normalized():
    embedder = HashEmbedder(dimensions=64)
    a = embedder.embed("the deploy failed at noon")
    assert a == embedder.embed("the deploy failed at noon")
    assert len(a) == 64
    assert abs(sum(v * v for v in a) - 1.0) < 1e-9
    assert embedder.embed("") == (0.0,) * 64
    with pytest.raises(ValueError):
        HashEmbedder(dimensions=0)


def test_index_ranks_overlapping_text_first(index, provenance):
    weather = make_record(provenance, body="Austin weather is hot in July.")
    deploy = make_record(provenance, body="The deploy pipeline broke on Tuesday.")
    index.index(weather)
    index.index(deploy)
    results = index.candidates("hot weather in Austin", k=2)
    assert [record_id for record_id, _ in results][0] == weather.id
    # An index may return only matches (FTS) or everything scored (vector);
    # the contract is ranking, not tail behavior.
    if len(results) > 1:
        assert results[0][1] > results[1][1]
    assert index.candidates("anything", k=0) == ()
    assert len(index.candidates("anything", k=1)) <= 1


def test_apply_create(store, provenance):
    record = make_record(provenance)
    written = apply_proposal(store, Proposal(kind=ProposalKind.CREATE, record=record))
    assert written == (record,)
    assert store.get(record.id) == record


def test_apply_supersede(store, provenance):
    old = make_record(provenance, body="Sandro lives in Boston.")
    store.put(old)
    new = make_record(provenance, body="Sandro lives in Austin.", supersedes=old.id)
    written = apply_proposal(
        store, Proposal(kind=ProposalKind.SUPERSEDE, record=new, reason="user moved")
    )
    assert store.get(old.id).state is RecordState.SUPERSEDED
    assert store.get(old.id).superseded_by == new.id
    assert store.get(new.id) == new
    assert len(written) == 2


def test_apply_supersede_validates(store, provenance):
    unlinked = make_record(provenance, body="No back reference.")
    with pytest.raises(ProposalError, match="supersedes"):
        apply_proposal(
            store, Proposal(kind=ProposalKind.SUPERSEDE, record=unlinked, reason="r")
        )
    ghost = make_record(provenance, body="Successor.", supersedes="mem:" + "0" * 32)
    with pytest.raises(ProposalError, match="not in store"):
        apply_proposal(store, Proposal(kind=ProposalKind.SUPERSEDE, record=ghost, reason="r"))


def test_apply_transition_and_archive(store, provenance):
    a = make_record(provenance, body="It is safe.")
    b = make_record(provenance, body="It is risky.")
    store.put(a)
    store.put(b)
    apply_proposal(
        store,
        Proposal(
            kind=ProposalKind.TRANSITION, target_id=a.id, other_id=b.id, reason="conflict"
        ),
    )
    assert store.get(a.id).state is RecordState.TRANSITIONAL
    assert store.get(b.id).state is RecordState.TRANSITIONAL

    apply_proposal(
        store, Proposal(kind=ProposalKind.ARCHIVE, target_id=a.id, reason="decay")
    )
    assert store.get(a.id).state is RecordState.ARCHIVED


def test_apply_rejects_illegal_lifecycle_without_writing(store, provenance):
    record = make_record(provenance)
    store.put(record)
    apply_proposal(store, Proposal(kind=ProposalKind.ARCHIVE, target_id=record.id, reason="d"))
    with pytest.raises(LifecycleError):
        apply_proposal(
            store, Proposal(kind=ProposalKind.ARCHIVE, target_id=record.id, reason="again")
        )
    assert len(store.history(record.id)) == 2  # nothing extra written


def test_apply_rejects_malformed_proposals(store):
    with pytest.raises(ProposalError):
        apply_proposal(store, Proposal(kind=ProposalKind.CREATE))
    with pytest.raises(ProposalError):
        apply_proposal(store, Proposal(kind=ProposalKind.TRANSITION, target_id="x"))
    with pytest.raises(ProposalError):
        apply_proposal(store, Proposal(kind=ProposalKind.ARCHIVE))


def test_apply_proposals_in_order(store, provenance):
    record = make_record(provenance)
    written = apply_proposals(
        store,
        [
            Proposal(kind=ProposalKind.CREATE, record=record),
            Proposal(kind=ProposalKind.ARCHIVE, target_id=record.id, reason="decay"),
        ],
    )
    assert len(written) == 2
    assert store.get(record.id).state is RecordState.ARCHIVED
