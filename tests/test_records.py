"""Record format: content addressing, round-trips, canonical bytes, invariants."""

from datetime import UTC, datetime

import pytest
from fg_agent_id import canonical_json

from fg_agent_memory import (
    MemoryFile,
    MemoryRecord,
    Provenance,
    RecordError,
    RecordState,
    RecordType,
    Transition,
    confidence_from_wire,
    confidence_to_wire,
)

from .conftest import FIXED_TIME, make_record


def test_id_is_content_address(provenance):
    record = make_record(provenance)
    assert record.id.startswith("mem:")
    assert len(record.id) == 4 + 32
    # Same birth content, same id — dedup by content identity.
    twin = make_record(provenance)
    assert twin.id == record.id


def test_id_ignores_lifecycle_fields(provenance):
    record = make_record(provenance)
    changed = record.model_copy(
        update={"confidence": 0.1, "tags": ("x",), "state_reason": "why not"}
    )
    assert changed.id == record.id


def test_id_changes_with_birth_content(provenance):
    a = make_record(provenance, body="Sandro lives in Austin.")
    b = make_record(provenance, body="Sandro lives in Boston.")
    assert a.id != b.id


def test_forged_id_rejected(provenance):
    record = make_record(provenance)
    with pytest.raises(RecordError, match="content address"):
        MemoryRecord.model_validate({**record.model_dump(), "id": "mem:" + "0" * 32})


def test_round_trip(provenance):
    record = make_record(
        provenance,
        slots={"subject": "sky", "location": "Austin"},
        confidence=0.85,
        tags=("weather", "austin"),
    )
    parsed = MemoryRecord.model_validate(record.model_dump())
    assert parsed == record
    assert canonical_json(parsed.model_dump()) == canonical_json(record.model_dump())


def test_unknown_fields_preserved_and_round_tripped(provenance):
    data = make_record(provenance).model_dump()
    data["x_saliency"] = 7
    parsed = MemoryRecord.model_validate(data)
    assert parsed.extra == {"x_saliency": 7}
    assert parsed.model_dump()["x_saliency"] == 7


def test_extra_must_not_shadow_fields(provenance):
    with pytest.raises(RecordError, match="shadow"):
        make_record(provenance, extra={"state": "archived"})


def test_confidence_wire_form():
    assert confidence_to_wire(0.5) == "0.5000"
    assert confidence_to_wire(1) == "1.0000"
    assert confidence_from_wire("0.2500") == 0.25
    for bad in (-0.1, 1.5, "2.0", "nope", None, True):
        with pytest.raises(RecordError):
            confidence_from_wire(bad)


def test_canonical_bytes_stable(provenance):
    record = make_record(provenance, confidence=0.9)
    first = canonical_json(record.model_dump())
    again = canonical_json(MemoryRecord.model_validate(record.model_dump()).model_dump())
    assert first == again


def test_rule_requires_evidence(provenance):
    with pytest.raises(RecordError, match="evidence"):
        make_record(provenance, type=RecordType.RULE)
    fact = make_record(provenance)
    rule = make_record(
        provenance, body="Always check.", type=RecordType.RULE, evidence=(fact.id,)
    )
    assert rule.evidence == (fact.id,)


def test_transition_block_state_coupling(provenance):
    record = make_record(provenance)
    with pytest.raises(RecordError, match="transition"):
        record.model_copy(update={"state": RecordState.TRANSITIONAL})
    block = Transition(sides=(record.id, "mem:" + "b" * 32), reason="conflict")
    with pytest.raises(RecordError, match="transition"):
        record.model_copy(update={"transition": block})


def test_superseded_requires_successor(provenance):
    record = make_record(provenance)
    with pytest.raises(RecordError, match="successor"):
        record.model_copy(update={"state": RecordState.SUPERSEDED})
    with pytest.raises(RecordError, match="superseded_by"):
        record.model_copy(update={"superseded_by": "mem:" + "c" * 32})


def test_flagged_only_on_rules(provenance):
    with pytest.raises(RecordError, match="rule"):
        make_record(provenance).model_copy(update={"flagged": True})


def test_transition_validation():
    with pytest.raises(RecordError):
        Transition(sides=("a", "a"), reason="dup sides")
    with pytest.raises(RecordError):
        Transition(sides=("a", "b"), reason="")


def test_naive_timestamp_rejected(provenance):
    with pytest.raises(ValueError, match="timezone"):
        make_record(provenance, created_at=datetime(2026, 1, 1))


def test_memory_file_round_trip(provenance):
    records = (make_record(provenance), make_record(provenance, body="Two things happened."))
    mf = MemoryFile.create(records, agent=None, created_at=FIXED_TIME)
    parsed = MemoryFile.model_validate(mf.model_dump())
    assert parsed == mf
    assert parsed.canonical_bytes() == mf.canonical_bytes()


def test_memory_file_unknown_header_fields(provenance):
    data = MemoryFile.create((make_record(provenance),), created_at=FIXED_TIME).model_dump()
    data["x_engine"] = "reference"
    parsed = MemoryFile.model_validate(data)
    assert parsed.extra == {"x_engine": "reference"}
    assert parsed.model_dump()["x_engine"] == "reference"


def test_provenance_requires_kind():
    with pytest.raises(RecordError):
        Provenance(kind="")


def test_created_at_normalized_to_ms(provenance):
    record = make_record(
        provenance, created_at=datetime(2026, 1, 1, 0, 0, 0, 123456, tzinfo=UTC)
    )
    assert record.created_at.microsecond == 123000
    assert record.model_dump()["created_at"] == "2026-01-01T00:00:00.123Z"


class TestOccurredAt:
    def test_occurred_at_round_trips_and_keeps_the_id(self) -> None:
        from datetime import UTC, datetime

        prov = Provenance(kind="observation")
        event_time = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
        record = MemoryRecord.create(
            RecordType.EPISODE, "The deploy finished.", prov, occurred_at=event_time
        )
        wire = record.model_dump()
        assert wire["occurred_at"] == "2026-07-01T12:00:00.000Z"
        parsed = MemoryRecord.model_validate(wire)
        assert parsed.occurred_at == event_time
        assert parsed.id == record.id  # not part of the content address

    def test_absent_occurred_at_is_omitted_from_the_wire(self) -> None:
        prov = Provenance(kind="conversation")
        record = MemoryRecord.create(RecordType.FACT, "No event time here.", prov)
        assert "occurred_at" not in record.model_dump()
        # Byte-stable round trip — the back-compat guarantee for existing
        # signed records.
        assert (
            MemoryRecord.model_validate(record.model_dump()).model_dump()
            == record.model_dump()
        )
