"""Mechanical decay: salience arithmetic, archival, reinforcement, protections."""

from __future__ import annotations

from datetime import timedelta

import pytest

from fg_agent_memory import (
    InMemoryRecordStore,
    Provenance,
    RecordState,
    RecordType,
)
from fg_agent_memory.pipeline import (
    DecayConfig,
    SalienceScores,
    consolidate,
    decay,
    salience,
)

from .conftest import FIXED_TIME, make_record

CONFIG = DecayConfig(half_life_days=30.0, reinforcement_bonus=0.25, archive_threshold=0.05)
LONG_AGO = FIXED_TIME + timedelta(days=365)  # far past every half-life


@pytest.fixture
def store() -> InMemoryRecordStore:
    return InMemoryRecordStore()


def test_salience_halves_per_half_life(provenance):
    record = make_record(provenance, confidence=0.8)
    fresh = salience(
        record, touches=0, last_touched=None, now=FIXED_TIME, config=CONFIG
    )
    aged = salience(
        record,
        touches=0,
        last_touched=None,
        now=FIXED_TIME + timedelta(days=30),
        config=CONFIG,
    )
    assert fresh == pytest.approx(0.8)
    assert aged == pytest.approx(0.4)


def test_salience_reinforcement_bonus_and_touch_recency(provenance):
    record = make_record(provenance, confidence=0.8)
    touched_at = FIXED_TIME + timedelta(days=30)
    boosted = salience(
        record, touches=2, last_touched=touched_at, now=touched_at, config=CONFIG
    )
    # Age counts from the touch, so freshness is 1.0; two touches add 50%.
    assert boosted == pytest.approx(0.8 * 1.5)


def test_untouched_record_archives_and_touched_record_survives(store, provenance):
    stale = make_record(provenance, body="Stale fact nobody uses.")
    used = make_record(provenance, body="Frequently used fact.")
    store.put(stale)
    store.put(used)
    scores = SalienceScores()
    for _ in range(3):
        scores.touch(used.id, now=LONG_AGO)

    report = decay(store, scores=scores, now=LONG_AGO, config=CONFIG)

    assert store.get(stale.id).state is RecordState.ARCHIVED
    assert "salience" in store.get(stale.id).state_reason
    assert store.get(used.id).state is RecordState.ACTIVE
    archived_ids = [action.record_ids for action in report.archived]
    assert archived_ids == [(stale.id,)]
    assert report.saliences[stale.id] < CONFIG.archive_threshold


def test_decay_never_deletes(store, provenance):
    record = make_record(provenance)
    store.put(record)
    decay(store, now=LONG_AGO, config=CONFIG)
    # Archived, still addressable, full history intact.
    assert store.get(record.id).state is RecordState.ARCHIVED
    assert len(store.history(record.id)) == 2


def test_unflagged_rule_protected_flagged_rule_decays(store, provenance):
    episode = make_record(provenance, body="It happened.", type=RecordType.EPISODE)
    store.put(episode)
    rule = make_record(
        provenance, body="Always X.", type=RecordType.RULE, evidence=(episode.id,)
    )
    store.put(rule)

    report = decay(store, now=LONG_AGO, config=CONFIG)
    assert store.get(rule.id).state is RecordState.ACTIVE  # unflagged rules never decay
    assert any(
        action.record_ids == (rule.id,) and "unflagged rule" in action.reason
        for action in report.protected
    )

    # Once evidence is invalidated and the rule is flagged, it may decay.
    from fg_agent_memory import archive

    store.put(archive(store.get(episode.id), reason="invalidated in test", now=LONG_AGO))
    consolidate(store, now=LONG_AGO)  # flag sweep flags the rule
    much_later = LONG_AGO + timedelta(days=365)  # let the flagged rule go stale
    report2 = decay(store, now=much_later, config=CONFIG)
    assert store.get(rule.id).state is RecordState.ARCHIVED
    assert any(action.record_ids == (rule.id,) for action in report2.archived)


def test_evidence_of_active_rule_protected(store, provenance):
    episodes = [
        make_record(
            provenance,
            body=f"Deploy failed on day {day}.",
            type=RecordType.EPISODE,
            tags=("deploy",),
            created_at=FIXED_TIME + timedelta(days=day),
        )
        for day in range(3)
    ]
    for episode in episodes:
        store.put(episode)
    consolidate(store, now=FIXED_TIME + timedelta(days=3))  # promotes a rule
    assert len(store.list(type=RecordType.RULE)) == 1

    report = decay(store, now=LONG_AGO, config=CONFIG)

    for episode in episodes:
        assert store.get(episode.id).state is not RecordState.ARCHIVED
    assert any("evidence by an active rule" in action.reason for action in report.protected)


def test_transitional_records_never_decay(store):
    a = make_record(Provenance(kind="conversation"), body="Pair.", slots={"k": "1"})
    b = make_record(Provenance(kind="conversation"), body="Pair!", slots={"k": "2"})
    store.put(a)
    store.put(b)
    from fg_agent_memory import begin_transition

    marked_a, marked_b = begin_transition(a, b, reason="dispute")
    store.put(marked_a)
    store.put(marked_b)

    report = decay(store, now=LONG_AGO, config=CONFIG)

    assert store.get(a.id).state is RecordState.TRANSITIONAL
    assert store.get(b.id).state is RecordState.TRANSITIONAL
    assert all("both sides" in action.reason for action in report.protected)


def test_decay_is_deterministic(provenance):
    def run():
        store = InMemoryRecordStore()
        store.put(make_record(provenance, body="One fact."))
        store.put(make_record(provenance, body="Two fact."))
        return decay(store, now=LONG_AGO, config=CONFIG)

    first, second = run(), run()
    assert first.archived == second.archived
    assert first.saliences == second.saliences


class TestSaliencePersistence:
    def test_reinforcement_survives_reopening_the_directory(self, tmp_path) -> None:
        from fg_agent_memory import Memory

        root = tmp_path / "memory"
        memory = Memory(root)
        result = memory.remember("A frequently used fact is stored here.")
        for _ in range(3):
            memory.recall("frequently used fact")
        assert (root / "salience.json").exists()

        reopened = Memory(root)
        assert reopened._scores.touches(result.record_id) >= 3

    def test_corrupt_sidecar_degrades_gracefully(self, tmp_path) -> None:
        from fg_agent_memory import Memory

        root = tmp_path / "memory"
        Memory(root).remember("Still fine.")
        (root / "salience.json").write_text("{not json")
        reopened = Memory(root)  # must not raise
        assert "Still fine" in reopened.recall("fine").as_prompt_block()
