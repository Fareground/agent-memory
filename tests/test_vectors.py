"""Golden cross-implementation vectors (spec/vectors.json).

These pin the exact bytes an independent implementation must reproduce:
content-addressed ids, canonical record bytes, domain-separated signing
inputs, signed artifacts, and the full lifecycle legality table.
"""

import json
from pathlib import Path

import pytest
from fg_agent_id import canonical_json

from fg_agent_memory import (
    MemoryFile,
    MemoryRecord,
    RecordState,
    is_legal,
    signing_input,
    verify_by_address,
)

VECTORS = json.loads((Path(__file__).parent.parent / "spec" / "vectors.json").read_text())


def test_record_id_vector():
    v = VECTORS["record_id"]
    record = MemoryRecord.model_validate(
        {
            **v["birth"],
            "updated_at": v["birth"]["created_at"],
        }
    )
    assert record.id == v["expected"]


def test_canonical_record_vector():
    v = VECTORS["canonical_record"]
    record = MemoryRecord.model_validate(v["record"])
    assert canonical_json(record.model_dump()).hex() == v["canonical_hex"]


def test_signing_input_vectors():
    for case in VECTORS["signing_input"]:
        assert signing_input(case["context"], case["payload"]).hex() == case["expected_hex"]


@pytest.mark.parametrize("name", sorted(VECTORS["artifacts"]))
def test_artifact_signing_payload_vectors(name):
    """Each signed artifact's exact signing bytes and signature are pinned."""
    case = VECTORS["artifacts"][name]
    data = signing_input(case["context"], case["payload"])
    assert data.hex() == case["signing_input_hex"]
    verify_by_address(
        VECTORS["keys"]["address_a"], case["context"], case["payload"], case["signature_b64"]
    )


def test_signed_record_vector_verifies_as_object():
    record = MemoryRecord.model_validate(VECTORS["artifacts"]["signed_record"]["record"])
    record.verify()


def test_memory_file_vector_verifies_as_object():
    case = VECTORS["artifacts"]["memory_file"]
    mf = MemoryFile.model_validate(case["file"])
    mf.verify()
    assert mf.canonical_bytes().hex() == case["canonical_bytes_hex"]


def test_lifecycle_legality_vectors():
    cases = VECTORS["lifecycle"]["legality"]
    assert len(cases) == len(RecordState) ** 2  # the table is total
    for case in cases:
        assert (
            is_legal(RecordState(case["from"]), RecordState(case["to"])) == case["legal"]
        ), f"{case['from']} -> {case['to']}"
