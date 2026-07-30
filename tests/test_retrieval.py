"""Retrieval invariants: state ordering, both-sides, budget, determinism."""

from datetime import timedelta

import pytest

from fg_agent_memory import (
    InMemoryRecordStore,
    InMemorySearchIndex,
    SearchIndex,
    archive,
    begin_transition,
    supersede,
)
from fg_agent_memory.retrieval import TouchCounts, approx_tokens, retrieve
from fg_agent_memory.stores import SQLiteSearchIndex

from .conftest import FIXED_TIME, make_record

NOW = FIXED_TIME + timedelta(days=1)
BIG_BUDGET = 10_000


class StubIndex(SearchIndex):
    """Fixed relevance scores, for pinning every other factor in tests."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = dict(scores)

    def index(self, record) -> None:
        raise NotImplementedError("stub is preloaded")

    def candidates(self, query: str, k: int):
        ranked = sorted(self._scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return tuple(ranked[:k])


@pytest.fixture
def store():
    return InMemoryRecordStore()


def test_active_outranks_superseded_at_equal_relevance(store, provenance):
    old = make_record(provenance, body="The API limit is 100.")
    new = make_record(provenance, body="The API limit is 500.", supersedes=old.id)
    store.put(old)
    store.put(new)
    store.put(supersede(old, new, reason="limit raised", now=FIXED_TIME))
    index = StubIndex({old.id: 1.0, new.id: 1.0})

    block = retrieve(
        "api limit",
        store=store,
        indexes=[index],
        budget_tokens=BIG_BUDGET,
        now=NOW,
        include_history=True,
    )
    assert block.ids[0] == new.id
    scores = {item.record.id: item.score for item in block.records}
    assert scores[new.id] > scores[old.id]


def test_superseded_excluded_by_default_but_rides_history_trail(store, provenance):
    old = make_record(provenance, body="Sandro lives in Boston.")
    new = make_record(provenance, body="Sandro lives in Austin.", supersedes=old.id)
    store.put(old)
    store.put(new)
    store.put(supersede(old, new, reason="moved", now=FIXED_TIME))

    # As a direct candidate, the superseded record is invisible by default.
    only_old = retrieve(
        "where does Sandro live",
        store=store,
        indexes=[StubIndex({old.id: 1.0})],
        budget_tokens=BIG_BUDGET,
        now=NOW,
    )
    assert only_old.records == ()

    # With history requested, the successor pulls its predecessor along.
    with_history = retrieve(
        "where does Sandro live",
        store=store,
        indexes=[StubIndex({new.id: 1.0})],
        budget_tokens=BIG_BUDGET,
        now=NOW,
        include_history=True,
    )
    assert with_history.ids == (new.id, old.id)
    predecessor = with_history.records[1]
    assert predecessor.reasons == (f"superseded predecessor of {new.id}",)


def test_transitional_surfaces_both_sides(store, provenance):
    a = make_record(provenance, body="The rollout is safe.")
    b = make_record(provenance, body="The rollout is risky.")
    a_marked, b_marked = begin_transition(a, b, reason="conflict", now=FIXED_TIME)
    store.put(a_marked)
    store.put(b_marked)

    block = retrieve(
        "rollout safety",
        store=store,
        indexes=[StubIndex({a.id: 1.0})],  # only ONE side matches the query
        budget_tokens=BIG_BUDGET,
        now=NOW,
    )
    assert set(block.ids) == {a.id, b.id}
    partner = next(item for item in block.records if item.record.id == b.id)
    assert partner.reasons == (f"other side of the transition with {a.id}",)


def test_transition_pair_is_budgeted_atomically(store, provenance):
    a = make_record(provenance, body="Side A of the dispute, long enough to cost tokens.")
    b = make_record(provenance, body="Side B of the dispute, long enough to cost tokens.")
    a_marked, b_marked = begin_transition(a, b, reason="conflict", now=FIXED_TIME)
    store.put(a_marked)
    store.put(b_marked)

    # Budget fits one side but not both: the pair must not be split.
    budget = approx_tokens(a_marked)
    block = retrieve(
        "dispute",
        store=store,
        indexes=[StubIndex({a.id: 1.0})],
        budget_tokens=budget,
        now=NOW,
    )
    assert block.records == ()


def test_archived_excluded_by_default_included_by_flag(store, provenance):
    record = make_record(provenance, body="An old detail nobody needs.")
    store.put(record)
    store.put(archive(record, reason="decay", now=FIXED_TIME))
    index = StubIndex({record.id: 1.0})
    common = dict(store=store, indexes=[index], budget_tokens=BIG_BUDGET, now=NOW)

    assert retrieve("old detail", **common).records == ()
    included = retrieve("old detail", include_archived=True, **common)
    assert included.ids == (record.id,)


def test_budget_truncates_by_whole_records(store, provenance):
    records = [
        make_record(provenance, body=f"Record number {i} with a body of steady size.")
        for i in range(4)
    ]
    for record in records:
        store.put(record)
    index = StubIndex({record.id: 1.0 for record in records})
    per_record = approx_tokens(records[0])

    block = retrieve(
        "record",
        store=store,
        indexes=[index],
        budget_tokens=per_record * 2 + 1,  # room for two, not three
        now=NOW,
    )
    assert len(block.records) == 2
    assert block.total_tokens == per_record * 2 <= block.budget_tokens
    assert all(item.tokens == per_record for item in block.records)


def test_confidence_and_relevance_weigh_in(store, provenance):
    sure = make_record(provenance, body="Certain fact.", confidence=0.9)
    shaky = make_record(provenance, body="Doubtful fact.", confidence=0.2)
    store.put(sure)
    store.put(shaky)

    block = retrieve(
        "fact",
        store=store,
        indexes=[StubIndex({sure.id: 0.5, shaky.id: 0.5})],
        budget_tokens=BIG_BUDGET,
        now=NOW,
    )
    assert block.ids == (sure.id, shaky.id)


def test_touch_reinforcement_lifts_repeatedly_retrieved_records(store, provenance):
    quiet = make_record(provenance, body="Rarely used fact.")
    favorite = make_record(provenance, body="Constantly used fact.")
    store.put(quiet)
    store.put(favorite)
    index = StubIndex({quiet.id: 1.0, favorite.id: 1.0})
    touches = TouchCounts()
    for _ in range(10):
        touches.touch(favorite.id)

    block = retrieve(
        "fact",
        store=store,
        indexes=[index],
        budget_tokens=BIG_BUDGET,
        now=NOW,
        touches=touches,
    )
    assert block.ids[0] == favorite.id
    # Retrieval itself reinforces: both included records got touched.
    assert touches.count(favorite.id) == 11
    assert touches.count(quiet.id) == 1


def test_union_across_indexes_takes_best_normalized_score(store, provenance):
    first = make_record(provenance, body="One record, two indexes.")
    second = make_record(provenance, body="Another record entirely.")
    store.put(first)
    store.put(second)
    # Index A: first is a weak second place (0.1 of a 0.5 top -> 0.2).
    # Index B: first is the top hit (normalizes to 1.0). Best wins.
    index_a = StubIndex({first.id: 0.1, second.id: 0.5})
    index_b = StubIndex({first.id: 0.9, second.id: 0.3})

    block = retrieve(
        "record",
        store=store,
        indexes=[index_a, index_b],
        budget_tokens=BIG_BUDGET,
        now=NOW,
    )
    by_id = {item.record.id: item for item in block.records}
    assert "relevance 1.0000" in by_id[first.id].reasons[0]


def test_deterministic_across_repeated_calls(store, provenance):
    records = [make_record(provenance, body=f"Fact number {i}.") for i in range(5)]
    for record in records:
        store.put(record)
    index = StubIndex({record.id: 0.5 for record in records})
    common = dict(store=store, indexes=[index], budget_tokens=BIG_BUDGET, now=NOW)

    first = retrieve("fact", **common)
    second = retrieve("fact", **common)
    assert first == second
    assert first.ids == tuple(sorted(first.ids))  # equal scores tie-break by id


def test_stale_index_entries_are_skipped(store, provenance):
    record = make_record(provenance, body="Still stored.")
    store.put(record)
    index = StubIndex({record.id: 1.0, "mem:" + "f" * 32: 1.0})

    block = retrieve(
        "stored",
        store=store,
        indexes=[index],
        budget_tokens=BIG_BUDGET,
        now=NOW,
    )
    assert block.ids == (record.id,)


def test_end_to_end_store_index_retrieve(provenance):
    """Write records via the store, index them, retrieve over real indexes."""
    store = InMemoryRecordStore()
    fts = SQLiteSearchIndex()
    vector = InMemorySearchIndex()
    weather = make_record(provenance, body="Austin weather is hot in July.")
    old_deploy = make_record(provenance, body="Deploys run from the laptop.")
    new_deploy = make_record(
        provenance, body="Deploys run from CI now.", supersedes=old_deploy.id
    )
    for record in (weather, old_deploy, new_deploy):
        store.put(record)
    store.put(supersede(old_deploy, new_deploy, reason="CI adopted", now=FIXED_TIME))
    for record in (weather, old_deploy, new_deploy):
        fts.index(record)
        vector.index(record)

    block = retrieve(
        "how do deploys run",
        store=store,
        indexes=[fts, vector],
        budget_tokens=BIG_BUDGET,
        now=NOW,
    )
    assert block.ids[0] == new_deploy.id
    assert old_deploy.id not in block.ids  # superseded stays out by default
    assert block.total_tokens <= block.budget_tokens


class TestHybridRelevanceNormalization:
    def test_incomparable_index_scales_do_not_dominate(self) -> None:
        """A bm25-scale index (scores ~10) beside a cosine-scale index
        (scores ~0.9) must not win on magnitude alone: each index is
        normalized to [0, 1] before the best-across-indexes union."""
        from fg_agent_memory.ports import InMemoryRecordStore, SearchIndex
        from fg_agent_memory.records import MemoryRecord, Provenance, RecordType
        from fg_agent_memory.retrieval import retrieve

        prov = Provenance(kind="conversation")
        loud = MemoryRecord.create(RecordType.FACT, "Loudly indexed fact.", prov)
        quiet = MemoryRecord.create(RecordType.FACT, "Quietly indexed fact.", prov)
        store = InMemoryRecordStore()
        store.put(loud)
        store.put(quiet)

        class Fixed(SearchIndex):
            def __init__(self, scores: dict[str, float]) -> None:
                self._scores = scores

            def index(self, record: MemoryRecord) -> None:  # pragma: no cover
                pass

            def candidates(self, query: str, k: int):
                return tuple(sorted(self._scores.items(), key=lambda p: -p[1]))[:k]

        # bm25-ish index prefers quiet (8 < 10); cosine-ish prefers quiet
        # too (0.9 > 0.5) — but raw max-union would rank loud first on the
        # bm25 index's sheer scale if quiet's cosine won on its own index.
        bm25_like = Fixed({loud.id: 10.0, quiet.id: 12.0})
        cosine_like = Fixed({loud.id: 0.2, quiet.id: 0.9})
        block = retrieve(
            "fact",
            store=store,
            indexes=(bm25_like, cosine_like),
            budget_tokens=1000,
        )
        assert block.ids[0] == quiet.id
        # Both indexes agree quiet is best; its normalized relevance is 1.0.
        assert block.records[0].score >= block.records[1].score
