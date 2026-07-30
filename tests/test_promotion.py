"""Promotion: repeated episodes → evidence-linked rules, and the flag sweep."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fg_agent_memory import (
    InMemoryRecordStore,
    Provenance,
    RecordState,
    RecordType,
    archive,
)
from fg_agent_memory.pipeline import ConsolidationConfig, consolidate

from .conftest import make_record

NOW = datetime(2026, 2, 1, 0, 0, 0, tzinfo=UTC)


@pytest.fixture
def store() -> InMemoryRecordStore:
    return InMemoryRecordStore()


def _episodes(store, count: int, tags=("deploy", "failure"), start: int = 1):
    records = []
    for day in range(start, start + count):
        episode = make_record(
            Provenance(kind="observation"),
            body=f"Deploy failed on day {day}.",
            type=RecordType.EPISODE,
            tags=tags,
            created_at=datetime(2026, 1, day, tzinfo=UTC),
        )
        store.put(episode)
        records.append(episode)
    return records


def test_promotion_creates_rule_with_evidence_to_all_episodes(store):
    episodes = _episodes(store, 3)
    report = consolidate(store, now=NOW)

    rules = store.list(state=RecordState.ACTIVE, type=RecordType.RULE)
    assert len(rules) == 1
    rule = rules[0]
    assert set(rule.evidence) == {episode.id for episode in episodes}
    assert rule.tags == ("deploy", "failure")
    assert rule.provenance.kind == "consolidation"
    promotions = report.by_pass("promotion")
    assert len(promotions) == 1
    assert promotions[0].after == ("active",)


def test_promotion_respects_threshold(store):
    _episodes(store, 2)
    consolidate(store, config=ConsolidationConfig(promotion_threshold=3), now=NOW)
    assert store.list(type=RecordType.RULE) == ()

    _episodes(store, 1, start=3)  # third distinct episode, same signature
    consolidate(store, config=ConsolidationConfig(promotion_threshold=3), now=NOW)
    assert len(store.list(type=RecordType.RULE)) == 1


def test_promotion_is_idempotent_per_signature(store):
    _episodes(store, 3)
    consolidate(store, now=NOW)
    consolidate(store, now=NOW)  # second pass must not duplicate the rule
    assert len(store.list(type=RecordType.RULE)) == 1


def test_flag_sweep_flags_rule_when_evidence_invalidated(store):
    episodes = _episodes(store, 3)
    consolidate(store, now=NOW)
    for episode in episodes:
        store.put(archive(store.get(episode.id), reason="test decay", now=NOW))

    report = consolidate(store, now=NOW)

    rule = store.list(type=RecordType.RULE)[0]
    assert rule.flagged
    assert "archived or superseded" in rule.flag_reason
    sweeps = report.by_pass("rule-flags")
    assert len(sweeps) == 1
    assert sweeps[0].action == "flag"
