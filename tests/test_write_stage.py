"""Write stage: extraction validation, content-address dedup, reinforcement."""

from __future__ import annotations

import pytest

from fg_agent_memory import (
    InMemoryRecordStore,
    MemoryRecord,
    Operator,
    Proposal,
    ProposalKind,
    Provenance,
    RecordState,
    RecordType,
)
from fg_agent_memory.pipeline import SalienceScores, ingest

from .conftest import FIXED_TIME, make_record


class FixedOperator(Operator):
    """Extractor that returns a canned proposal tuple."""

    def __init__(self, proposals: tuple[Proposal, ...]) -> None:
        self._proposals = proposals

    def propose(self, text: str, provenance: Provenance) -> tuple[Proposal, ...]:
        return self._proposals


class EchoOperator(Operator):
    """Extractor that proposes one fact per input line — used to check what
    text the write stage actually hands the extractor."""

    def propose(self, text: str, provenance: Provenance) -> tuple[Proposal, ...]:
        return tuple(
            Proposal(
                kind=ProposalKind.CREATE,
                record=MemoryRecord.create(
                    RecordType.FACT, line, provenance, created_at=FIXED_TIME
                ),
            )
            for line in text.splitlines()
        )


@pytest.fixture
def store() -> InMemoryRecordStore:
    return InMemoryRecordStore()


def test_ingest_accepts_valid_facts(store, provenance):
    record = make_record(provenance, body="Sandro lives in Austin.")
    report = ingest(
        "irrelevant",
        extractor=FixedOperator((Proposal(kind=ProposalKind.CREATE, record=record),)),
        store=store,
        provenance=provenance,
    )
    assert report.accepted == (record.id,)
    assert report.deduped == ()
    assert report.rejected == ()
    assert store.get(record.id).state is RecordState.ACTIVE


def test_ingest_dedupes_by_content_address_and_reinforces(store, provenance):
    record = make_record(provenance, body="Sandro lives in Austin.")
    proposal = Proposal(kind=ProposalKind.CREATE, record=record)
    extractor = FixedOperator((proposal,))
    scores = SalienceScores()

    first = ingest("x", extractor=extractor, store=store, provenance=provenance, scores=scores)
    second = ingest(
        "x",
        extractor=extractor,
        store=store,
        provenance=provenance,
        scores=scores,
        now=FIXED_TIME,
    )

    assert first.accepted == (record.id,)
    assert second.accepted == ()
    assert second.deduped == (record.id,)
    # Dedup is a reinforcement touch, not a rewrite: one stored version.
    assert len(store.history(record.id)) == 1
    assert scores.touches(record.id) == 1
    assert scores.last_touched(record.id) == FIXED_TIME


def test_ingest_rejects_rule_creation(store, provenance):
    episode = make_record(provenance, body="It happened.", type=RecordType.EPISODE)
    store.put(episode)
    rule = make_record(
        provenance,
        body="Always X.",
        type=RecordType.RULE,
        evidence=(episode.id,),
    )
    report = ingest(
        "x",
        extractor=FixedOperator((Proposal(kind=ProposalKind.CREATE, record=rule),)),
        store=store,
        provenance=provenance,
    )
    assert report.accepted == ()
    assert len(report.rejected) == 1
    assert "promotion" in report.rejected[0].error


def test_ingest_rejects_non_create_proposals(store, provenance):
    target = make_record(provenance, body="Existing fact.")
    store.put(target)
    report = ingest(
        "x",
        extractor=FixedOperator(
            (
                Proposal(kind=ProposalKind.ARCHIVE, target_id=target.id, reason="nope"),
                Proposal(kind=ProposalKind.CREATE, record=None),
            )
        ),
        store=store,
        provenance=provenance,
    )
    assert report.accepted == ()
    assert len(report.rejected) == 2
    assert "only create" in report.rejected[0].error
    assert "no record" in report.rejected[1].error
    # Nothing was written or mutated.
    assert store.get(target.id).state is RecordState.ACTIVE
    assert len(store.history(target.id)) == 1


def test_ingest_rejects_records_born_with_lifecycle_links(store, provenance):
    predecessor = make_record(provenance, body="Old fact.")
    store.put(predecessor)
    entangled = make_record(provenance, body="New fact.", supersedes=predecessor.id)
    report = ingest(
        "x",
        extractor=FixedOperator((Proposal(kind=ProposalKind.CREATE, record=entangled),)),
        store=store,
        provenance=provenance,
    )
    assert report.accepted == ()
    assert "lifecycle links" in report.rejected[0].error


def test_ingest_renders_structured_events_deterministically(store, provenance):
    report = ingest(
        {"b": 2, "a": 1},
        extractor=EchoOperator(),
        store=store,
        provenance=provenance,
    )
    bodies = [store.get(rid).body for rid in report.accepted]
    assert bodies == ["a: 1", "b: 2"]  # sorted keys, key: value lines


def test_ingest_with_heuristic_extractor_end_to_end(store, provenance):
    from fg_agent_memory import HeuristicExtractor

    report = ingest(
        "Sandro lives in Austin. The deploy failed on Tuesday.",
        extractor=HeuristicExtractor(),
        store=store,
        provenance=provenance,
    )
    assert len(report.accepted) == 2
    types = {store.get(rid).type for rid in report.accepted}
    assert types == {RecordType.FACT, RecordType.EPISODE}
