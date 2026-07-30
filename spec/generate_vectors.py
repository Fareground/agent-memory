"""Regenerate spec/vectors.json.

Every vector is derived from fixed seeds and fixed timestamps, so the output
is byte-stable across runs. Run this after any intentional wire-format
change:

    python spec/generate_vectors.py

Then review the diff — an unexpected change here means the wire format moved.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from fg_agent_id import KeyPair, address_from_signing_key, canonical_json

from fg_agent_memory import (
    CONTEXT_MEMORY_FILE,
    CONTEXT_RECORD,
    MemoryFile,
    MemoryRecord,
    Provenance,
    RecordType,
    is_legal,
    signing_input,
)
from fg_agent_memory.records import RecordState

SEED_A = bytes(range(32))
AGREEMENT_SEED = bytes(range(96, 128))

FIXED_TIME = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def keypair(seed: bytes) -> KeyPair:
    return KeyPair(
        signing_key=Ed25519PrivateKey.from_private_bytes(seed),
        agreement_key=X25519PrivateKey.from_private_bytes(AGREEMENT_SEED),
    )


def signed(context: str, payload: dict, keys: KeyPair) -> dict:
    data = signing_input(context, payload)
    return {
        "context": context,
        "payload": payload,
        "signing_input_hex": data.hex(),
        "signature_b64": base64.b64encode(keys.sign(data)).decode(),
    }


def build() -> dict:
    keys_a = keypair(SEED_A)
    addr_a = address_from_signing_key(keys_a.public.signing)

    provenance = Provenance(kind="conversation", ref="session-0001", agent=addr_a)
    fact = MemoryRecord.create(
        RecordType.FACT,
        "The conformance suite runs on fixed seeds.",
        provenance,
        slots={"subject": "conformance suite", "predicate": "runs on fixed seeds"},
        confidence=0.9,
        tags=("conformance",),
        created_at=FIXED_TIME,
    )
    episode = MemoryRecord.create(
        RecordType.EPISODE,
        "Ran the vector generator and reviewed the diff.",
        provenance,
        confidence=0.75,
        created_at=FIXED_TIME,
    )
    rule = MemoryRecord.create(
        RecordType.RULE,
        "Always review the vector diff after a wire-format change.",
        provenance,
        confidence=0.8,
        evidence=(fact.id, episode.id),
        created_at=FIXED_TIME,
    )
    signed_fact = fact.sign(keys_a, addr_a)

    memory_file = MemoryFile.create(
        (fact, episode, rule), agent=addr_a, created_at=FIXED_TIME
    ).sign(keys_a)

    states = [state.value for state in RecordState]
    legality = [
        {"from": frm, "to": to, "legal": is_legal(RecordState(frm), RecordState(to))}
        for frm, to in product(states, states)
    ]

    return {
        "_": "Golden conformance vectors for fg-agent-memory. Regenerate with "
        "spec/generate_vectors.py; every value derives from fixed seeds.",
        "keys": {"seed_a_hex": SEED_A.hex(), "address_a": addr_a},
        "record_id": {
            "birth": {
                "type": fact.type.value,
                "body": fact.body,
                "slots": fact.slots,
                "provenance": fact.provenance.model_dump(),
                "created_at": "2026-01-01T00:00:00.000Z",
            },
            "expected": fact.id,
        },
        "canonical_record": {
            "record": fact.model_dump(),
            "canonical_hex": canonical_json(fact.model_dump()).hex(),
        },
        "signing_input": [
            {
                "context": CONTEXT_RECORD,
                "payload": {"a": 1},
                "expected_hex": signing_input(CONTEXT_RECORD, {"a": 1}).hex(),
            },
            {
                "context": CONTEXT_MEMORY_FILE,
                "payload": {"b": "two"},
                "expected_hex": signing_input(CONTEXT_MEMORY_FILE, {"b": "two"}).hex(),
            },
        ],
        "artifacts": {
            "signed_record": {
                **signed(CONTEXT_RECORD, signed_fact._payload(), keys_a),
                "record": signed_fact.model_dump(),
            },
            "memory_file": {
                **signed(CONTEXT_MEMORY_FILE, memory_file._payload(), keys_a),
                "file": memory_file.model_dump(),
                "canonical_bytes_hex": memory_file.canonical_bytes().hex(),
            },
        },
        "lifecycle": {"legality": legality},
    }


def main() -> None:
    out = Path(__file__).parent / "vectors.json"
    out.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
