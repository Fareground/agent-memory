"""Consolidation engine: dedup, contradiction protocol, hostile operators,
the ghost-memory end-to-end, and determinism."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fg_agent_memory import (
    InMemoryRecordStore,
    Proposal,
    ProposalKind,
    Provenance,
    RecordState,
    RecordType,
)
from fg_agent_memory.pipeline import (
    ConsolidationOperators,
    HeuristicContradictionOperator,
    HeuristicResolutionOperator,
    TrigramNearDupOperator,
    consolidate,
    default_operators,
    resolve,
)
from fg_agent_memory.pipeline.consolidate import (
    ContradictionOperator,
    NearDupOperator,
    trigram_similarity,
)
from fg_agent_memory.records import MemoryRecord

from .conftest import FIXED_TIME, make_record

NOW = datetime(2026, 2, 1, 0, 0, 0, tzinfo=UTC)


@pytest.fixture
def store() -> InMemoryRecordStore:
    return InMemoryRecordStore()


def snapshot(store: InMemoryRecordStore) -> dict[str, str]:
    return {record.id: record.state.value for record in store.list()}


# ---------------------------------------------------------------- exact dedup


def test_exact_dedup_collapses_identical_content(store, provenance):
    a = make_record(provenance, body="The build is green.")
    b = make_record(
        Provenance(kind="observation"), body="The build is green."
    )  # same content, different provenance → different content address
    store.put(a)
    store.put(b)
    assert a.id != b.id

    report = consolidate(store, now=NOW)

    actives = store.list(state=RecordState.ACTIVE)
    assert len(actives) == 1
    keeper = actives[0]
    loser_id = b.id if keeper.id == a.id else a.id
    loser = store.get(loser_id)
    assert loser.state is RecordState.SUPERSEDED
    assert loser.superseded_by == keeper.id
    assert "exact duplicate" in loser.state_reason
    dedup_actions = report.by_pass("dedup.exact")
    assert len(dedup_actions) == 1
    assert dedup_actions[0].before == ("active",)
    assert dedup_actions[0].after == ("superseded",)


# ------------------------------------------------------------------- near dup


def test_trigram_similarity_bounds():
    assert trigram_similarity("same text", "same text") == 1.0
    assert trigram_similarity("aaaa", "zzzz") == 0.0
    assert trigram_similarity("", "anything") == 0.0


def test_near_dup_merge_supersedes_both_sides_with_evidence(store, provenance):
    a = make_record(provenance, body="Sandro prefers small focused files.")
    b = make_record(
        provenance,
        body="Sandro prefers small, focused files!",
        confidence=0.9,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    store.put(a)
    store.put(b)

    report = consolidate(store, now=NOW)

    actives = store.list(state=RecordState.ACTIVE)
    assert len(actives) == 1
    merged = actives[0]
    assert set(merged.evidence) == {a.id, b.id}
    assert merged.provenance.kind == "consolidation"
    for old_id in (a.id, b.id):
        old = store.get(old_id)
        assert old.state is RecordState.SUPERSEDED
        assert old.superseded_by == merged.id  # both sides addressable + chained
    assert len(report.by_pass("dedup.near")) == 2
    assert report.rejections == ()


def test_near_dup_below_threshold_untouched(store, provenance):
    a = make_record(provenance, body="Sandro lives in Austin, Texas today.")
    b = make_record(provenance, body="The deploy pipeline uses GitHub Actions.")
    store.put(a)
    store.put(b)
    report = consolidate(store, now=NOW)
    assert report.by_pass("dedup.near") == ()
    assert store.get(a.id).state is RecordState.ACTIVE
    assert store.get(b.id).state is RecordState.ACTIVE


# -------------------------------------------------------- contradiction & ghost


def test_contradiction_routes_through_transitional_never_overwrites(store, provenance):
    a = make_record(provenance, body="Server region.", slots={"region": "us-east-1"})
    b = make_record(provenance, body="Server region!", slots={"region": "eu-west-1"})
    store.put(a)
    store.put(b)

    operators = ConsolidationOperators(
        near_dup=TrigramNearDupOperator(threshold=1.01),  # keep near-dup out of the way
        contradiction=HeuristicContradictionOperator(),
        resolver=None,  # detection only: no auto-resolution
    )
    report = consolidate(store, operators=operators, now=NOW)

    for rid in (a.id, b.id):
        record = store.get(rid)
        assert record.state is RecordState.TRANSITIONAL
        assert set(record.transition.sides) == {a.id, b.id}  # both sides visible
    assert len(report.by_pass("contradiction")) == 1
    assert report.by_pass("resolution") == ()


def test_ghost_memory_end_to_end(store):
    """X asserted, not-X later → transitional (both retrievable) → resolved →
    old fact superseded-but-addressable with a reason chain."""
    inference = Provenance(kind="conversation", ref="session-1")
    correction = Provenance(kind="correction", ref="session-2")
    x = make_record(inference, body="Sandro lives in Boston.", slots={"city": "Boston"})
    not_x = make_record(
        correction,
        body="Sandro lives in Austin.",
        slots={"city": "Austin"},
        created_at=datetime(2026, 1, 5, tzinfo=UTC),
    )
    store.put(x)
    store.put(not_x)

    # Pass 1: detection only — the contradiction becomes a transition pair.
    detect_only = ConsolidationOperators(
        near_dup=TrigramNearDupOperator(threshold=1.01),
        contradiction=HeuristicContradictionOperator(),
        resolver=None,
    )
    first = consolidate(store, operators=detect_only, now=NOW)
    assert store.get(x.id).state is RecordState.TRANSITIONAL
    assert store.get(not_x.id).state is RecordState.TRANSITIONAL
    assert len(first.by_pass("contradiction")) == 1
    # Both sides retrievable and mutually discoverable while in dispute.
    assert set(store.get(x.id).transition.sides) == {x.id, not_x.id}

    # Explicit resolution: correction wins; the loser is superseded, not gone.
    winner, loser = resolve(
        store,
        winner_id=not_x.id,
        loser_id=x.id,
        reason="user corrected the city",
        now=NOW,
    )
    assert winner.state is RecordState.ACTIVE
    assert loser.state is RecordState.SUPERSEDED
    assert loser.superseded_by == not_x.id
    assert loser.state_reason == "user corrected the city"
    # Ghost check: the old fact never competes as truth again but every
    # version of it stays addressable.
    assert store.get(x.id).state is RecordState.SUPERSEDED
    states = [version.state for version in store.history(x.id)]
    assert states == [
        RecordState.ACTIVE,
        RecordState.TRANSITIONAL,
        RecordState.SUPERSEDED,
    ]


def test_auto_resolver_correction_beats_inference(store):
    x = make_record(
        Provenance(kind="conversation"), body="Region fact.", slots={"r": "a"}
    )
    not_x = make_record(
        Provenance(kind="correction"), body="Region fact!", slots={"r": "b"}
    )
    store.put(x)
    store.put(not_x)
    operators = ConsolidationOperators(
        near_dup=TrigramNearDupOperator(threshold=1.01),
        contradiction=HeuristicContradictionOperator(),
        resolver=HeuristicResolutionOperator(),
    )
    report = consolidate(store, operators=operators, now=NOW)
    assert store.get(not_x.id).state is RecordState.ACTIVE
    assert store.get(x.id).state is RecordState.SUPERSEDED
    resolutions = report.by_pass("resolution")
    assert len(resolutions) == 1
    assert "correction" in resolutions[0].reason


def test_auto_resolver_newer_wins_only_within_same_provenance_kind(store):
    """Temporal precedence is applied cautiously: different provenance kinds
    → no auto-resolution, the pair stays transitional."""
    older = make_record(
        Provenance(kind="conversation"),
        body="Flag state.",
        slots={"flag": "on"},
        created_at=FIXED_TIME,
    )
    newer_other_kind = make_record(
        Provenance(kind="observation"),
        body="Flag state!",
        slots={"flag": "off"},
        created_at=datetime(2026, 1, 9, tzinfo=UTC),
    )
    store.put(older)
    store.put(newer_other_kind)
    consolidate(store, now=NOW)
    assert store.get(older.id).state is RecordState.TRANSITIONAL
    assert store.get(newer_other_kind.id).state is RecordState.TRANSITIONAL

    # Same provenance kind and identical slot keys → newer wins.
    store2 = InMemoryRecordStore()
    a = make_record(
        Provenance(kind="conversation"),
        body="Flag state.",
        slots={"flag": "on"},
        created_at=FIXED_TIME,
    )
    b = make_record(
        Provenance(kind="conversation"),
        body="Flag state!",
        slots={"flag": "off"},
        created_at=datetime(2026, 1, 9, tzinfo=UTC),
    )
    store2.put(a)
    store2.put(b)
    consolidate(store2, now=NOW)
    assert store2.get(b.id).state is RecordState.ACTIVE
    assert store2.get(a.id).state is RecordState.SUPERSEDED


# ------------------------------------------------------------ hostile operator


class HostileNearDup(NearDupOperator):
    """Oversteps in every way it can construct: archives instead of merging,
    merges without evidence links, supersedes records outside the pair."""

    def propose_merge(self, a, b, *, now):
        return Proposal(kind=ProposalKind.ARCHIVE, target_id=a.id, reason="be gone")


class HostileMergeWithoutEvidence(NearDupOperator):
    def propose_merge(self, a, b, *, now):
        merged = make_record(
            Provenance(kind="consolidation"),
            body=a.body,
            supersedes=b.id,
            created_at=now,
        )
        return Proposal(kind=ProposalKind.SUPERSEDE, record=merged, reason="merge")


class HostileContradiction(ContradictionOperator):
    """Names a record outside the candidate pair — a fabricated dispute."""

    def propose_contradiction(self, a, b):
        return Proposal(
            kind=ProposalKind.TRANSITION,
            target_id=a.id,
            other_id="mem:" + "f" * 32,
            reason="fabricated",
        )


class HostileResolver(HeuristicResolutionOperator):
    def propose_winner(self, a, b):
        return "mem:00000000000000000000000000000000", "I said so"


def test_hostile_operators_fully_rejected_with_report_entries(store, provenance):
    a = make_record(provenance, body="Sandro prefers small focused files.")
    b = make_record(provenance, body="Sandro prefers small, focused files!")
    store.put(a)
    store.put(b)
    before = snapshot(store)

    operators = ConsolidationOperators(
        near_dup=HostileNearDup(),
        contradiction=HostileContradiction(),
        resolver=None,
    )
    report = consolidate(store, operators=operators, now=NOW)

    assert snapshot(store) == before  # nothing applied, nothing mutated
    assert report.actions == ()
    errors = [rejection.error for rejection in report.rejections]
    assert any("only propose supersede" in error for error in errors)
    assert any("exactly the candidate pair" in error for error in errors)


def test_hostile_merge_without_evidence_rejected(store, provenance):
    a = make_record(provenance, body="Sandro prefers small focused files.")
    b = make_record(provenance, body="Sandro prefers small, focused files!")
    store.put(a)
    store.put(b)
    before = snapshot(store)
    operators = ConsolidationOperators(
        near_dup=HostileMergeWithoutEvidence(),
        contradiction=HeuristicContradictionOperator(),
        resolver=None,
    )
    report = consolidate(store, operators=operators, now=NOW)
    assert snapshot(store) == before
    assert any("BOTH sides" in rejection.error for rejection in report.rejections)


def test_hostile_resolver_rejected(store):
    a = make_record(Provenance(kind="conversation"), body="Pair.", slots={"k": "1"})
    b = make_record(Provenance(kind="conversation"), body="Pair!", slots={"k": "2"})
    store.put(a)
    store.put(b)
    detect = ConsolidationOperators(
        near_dup=TrigramNearDupOperator(threshold=1.01),
        contradiction=HeuristicContradictionOperator(),
        resolver=None,
    )
    consolidate(store, operators=detect, now=NOW)

    hostile = ConsolidationOperators(
        near_dup=TrigramNearDupOperator(threshold=1.01),
        contradiction=HeuristicContradictionOperator(),
        resolver=HostileResolver(),
    )
    report = consolidate(store, operators=hostile, now=NOW)
    assert store.get(a.id).state is RecordState.TRANSITIONAL  # dispute untouched
    assert any("outside the pair" in rejection.error for rejection in report.rejections)


# ---------------------------------------------------------------- determinism


def _seed(store: InMemoryRecordStore) -> None:
    conversation = Provenance(kind="conversation")
    store.put(make_record(conversation, body="Sandro prefers small focused files."))
    store.put(make_record(conversation, body="Sandro prefers small, focused files!"))
    store.put(make_record(conversation, body="Mode fact.", slots={"mode": "on"}))
    store.put(
        make_record(
            conversation,
            body="Mode fact!",
            slots={"mode": "off"},
            created_at=datetime(2026, 1, 9, tzinfo=UTC),
        )
    )
    for day in range(1, 4):
        store.put(
            make_record(
                conversation,
                body=f"Deploy failed on day {day}.",
                type=RecordType.EPISODE,
                tags=("deploy", "failure"),
                created_at=datetime(2026, 1, day, tzinfo=UTC),
            )
        )


def test_consolidate_is_deterministic():
    reports = []
    snapshots = []
    for _ in range(2):
        store = InMemoryRecordStore()
        _seed(store)
        reports.append(consolidate(store, operators=default_operators(), now=NOW))
        snapshots.append(snapshot(store))
    assert reports[0] == reports[1]
    assert snapshots[0] == snapshots[1]
    assert reports[0].actions  # the pass actually did things


class TestContradictionScope:
    def test_episodes_are_not_forced_into_contradiction(self, store) -> None:
        """Distinct events with opposite polarity are history, not a dispute."""
        prov = Provenance(kind="observation")
        monday = MemoryRecord.create(
            RecordType.EPISODE, "The build passed today.", prov
        )
        tuesday = MemoryRecord.create(
            RecordType.EPISODE, "The build did not pass today.", prov
        )
        store.put(monday)
        store.put(tuesday)
        consolidate(store, operators=default_operators())
        assert store.get(monday.id).state is RecordState.ACTIVE
        assert store.get(tuesday.id).state is RecordState.ACTIVE

    def test_multiway_contradiction_is_deferred_not_silent(self, store) -> None:
        """A contradicts B and C; only one pair can transition per pass —
        the other must surface in the report, never vanish."""
        prov = Provenance(kind="conversation")
        a = MemoryRecord.create(RecordType.FACT, "The staging deploy needs approval.", prov)
        b = MemoryRecord.create(
            RecordType.FACT, "The staging deploy needs no approval.", prov
        )
        c = MemoryRecord.create(
            RecordType.FACT, "The staging deploy needs no approval at all.", prov
        )
        for record in (a, b, c):
            store.put(record)
        no_resolver = ConsolidationOperators(
            near_dup=TrigramNearDupOperator(threshold=0.99),
            contradiction=HeuristicContradictionOperator(),
            resolver=None,
        )
        report = consolidate(store, operators=no_resolver)
        transitioned = [r for r in (a, b, c) if store.get(r.id).state is RecordState.TRANSITIONAL]
        assert len(transitioned) == 2
        deferrals = [
            rejection
            for rejection in report.rejections
            if rejection.pass_name == "contradiction" and "deferred" in rejection.error
        ]
        assert deferrals


class TestPartialEvidenceFlags:
    def test_rule_with_live_and_missing_evidence_stays_unflagged(self, store) -> None:
        prov = Provenance(kind="observation")
        live = MemoryRecord.create(RecordType.EPISODE, "Observed once.", prov)
        store.put(live)
        rule = MemoryRecord.create(
            RecordType.RULE,
            "A pattern with partly missing evidence.",
            Provenance(kind="consolidation"),
            evidence=(live.id, "mem:" + "0" * 32),
        )
        store.put(rule)
        consolidate(store, operators=default_operators())
        assert store.get(rule.id).flagged is False

    def test_rule_with_only_missing_or_dead_evidence_is_flagged(self, store) -> None:
        rule = MemoryRecord.create(
            RecordType.RULE,
            "A pattern whose evidence is gone.",
            Provenance(kind="consolidation"),
            evidence=("mem:" + "0" * 32,),
        )
        store.put(rule)
        consolidate(store, operators=default_operators())
        flagged = store.get(rule.id)
        assert flagged.flagged is True
        assert "no live evidence" in (flagged.flag_reason or "")


class TestEventTimeResolution:
    def test_resolver_prefers_newer_event_over_newer_ingest(self, store) -> None:
        """Learned about Tuesday's change on Wednesday, but yesterday's
        correction arrived today: event time decides, not ingest order."""
        prov = Provenance(kind="observation")
        newer_event = MemoryRecord.create(
            RecordType.FACT,
            "The primary region is us-east-1.",
            prov,
            created_at=datetime(2026, 7, 1, tzinfo=UTC),
            occurred_at=datetime(2026, 7, 20, tzinfo=UTC),
        )
        older_event = MemoryRecord.create(
            RecordType.FACT,
            "The primary region is not us-east-1 anymore, sadly.",
            prov,
            created_at=datetime(2026, 7, 10, tzinfo=UTC),  # ingested later
            occurred_at=datetime(2026, 7, 5, tzinfo=UTC),  # happened earlier
        )
        resolver = HeuristicResolutionOperator()
        verdict = resolver.propose_winner(newer_event, older_event)
        assert verdict is not None
        winner_id, reason = verdict
        assert winner_id == newer_event.id
        assert "newer event" in reason
