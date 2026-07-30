"""Shared fixtures: fixed keys and record builders."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from fg_agent_id import KeyPair, address_from_signing_key

from fg_agent_memory import MemoryRecord, Provenance, RecordType

FIXED_TIME = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


@pytest.fixture
def keys() -> KeyPair:
    return KeyPair(
        signing_key=Ed25519PrivateKey.from_private_bytes(bytes(range(32))),
        agreement_key=X25519PrivateKey.from_private_bytes(bytes(range(96, 128))),
    )


@pytest.fixture
def address(keys: KeyPair) -> str:
    return address_from_signing_key(keys.public.signing)


@pytest.fixture
def provenance() -> Provenance:
    return Provenance(kind="conversation", ref="session-42")


def make_record(
    provenance: Provenance,
    body: str = "The sky over Austin is clear.",
    type: RecordType = RecordType.FACT,
    **kwargs,
) -> MemoryRecord:
    kwargs.setdefault("created_at", FIXED_TIME)
    return MemoryRecord.create(type, body, provenance, **kwargs)
