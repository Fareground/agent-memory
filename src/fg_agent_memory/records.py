"""MemoryRecord and MemoryFile: the portable record format.

The record format IS the standard. A record is a typed, timestamped,
provenance-carrying unit of agent memory with an explicit lifecycle state;
a memory file is the signed, portable container of records an agent owns.
Every invariant of the standard is stateable over these structures alone —
no storage backend is ever load-bearing.

Identity is content-addressed: a record's id is derived from its immutable
birth content (type, body, slots, provenance, created_at), so two agents that
extract the same content at the same instant name it identically, and dedup
by content identity is a byte comparison. Lifecycle fields (state, links,
confidence, flags) are versioned *around* that stable id — a state change
never changes what the record is called.

Unknown fields are preserved and signed (``extra``), so a newer record
carrying extension fields still verifies on an older peer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, fields, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from fg_agent_id import KeyPair, canonical_json

# Internal-API dependency: fg_agent_id.serde helpers are not re-exported at
# the package root; the fg-agent-id version bound in pyproject pins them.
from fg_agent_id.serde import (
    canonical_timestamp,
    parse_datetime,
    require_mapping,
    require_str,
)

from .errors import RecordError
from .signing import CONTEXT_MEMORY_FILE, CONTEXT_RECORD, sign_payload, verify_by_address
from .version import FORMAT_VERSION

# 128 bits of the content digest — collision-safe at any plausible store size
# while keeping ids short enough to read in logs and evidence lists.
_ID_HEX_CHARS = 32
_ID_PREFIX = "mem:"


class RecordType(StrEnum):
    """What kind of memory a record holds. Rules are special: they are
    promoted from evidence and must always keep links to it."""

    FACT = "fact"
    EPISODE = "episode"
    PROCEDURE = "procedure"
    RULE = "rule"


class RecordState(StrEnum):
    """The explicit lifecycle state every record carries. Transitions are
    enforced in :mod:`fg_agent_memory.lifecycle`, never by convention."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    TRANSITIONAL = "transitional"
    ARCHIVED = "archived"


def confidence_to_wire(value: float) -> str:
    """Render a confidence for canonical serialization.

    Canonical JSON forbids floats (their formatting differs across languages),
    so confidence travels as a string with exactly four decimal digits —
    one spelling per value, always. Four digits is deliberate headroom over
    what any scoring pipeline meaningfully distinguishes.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RecordError(f"confidence must be a number, got {type(value).__name__}")
    if not 0.0 <= float(value) <= 1.0:
        raise RecordError(f"confidence must be within [0, 1], got {value!r}")
    return f"{float(value):.4f}"


def confidence_from_wire(value: Any) -> float:
    """Parse a wire confidence string back to a float in [0, 1]."""
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError as exc:
            raise RecordError(f"confidence is not a valid number: {value!r}") from exc
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)
    else:
        raise RecordError(f"confidence must be a number or string, got {type(value).__name__}")
    if not 0.0 <= parsed <= 1.0:
        raise RecordError(f"confidence must be within [0, 1], got {value!r}")
    return parsed


@dataclass(frozen=True, kw_only=True)
class Provenance:
    """Where a record came from: a source kind (open vocabulary — e.g.
    ``conversation``, ``observation``, ``tool``, ``consolidation``,
    ``import``), an optional source reference, and the address of the agent
    that wrote it."""

    kind: str
    ref: str | None = None
    agent: str | None = None

    def __post_init__(self) -> None:
        require_str("provenance.kind", self.kind)
        if not self.kind:
            raise RecordError("provenance.kind must be non-empty")

    def model_dump(self) -> dict[str, Any]:
        return {"kind": self.kind, "ref": self.ref, "agent": self.agent}

    @classmethod
    def model_validate(cls, data: Any) -> Provenance:
        data = require_mapping("provenance", data)
        return cls(kind=data.get("kind", ""), ref=data.get("ref"), agent=data.get("agent"))


@dataclass(frozen=True, kw_only=True)
class Transition:
    """The two sides of an unresolved contradiction, plus why it exists.
    Both records in the pair carry the same block — a reader holding either
    side can always find the other."""

    sides: tuple[str, str]
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sides", tuple(self.sides))
        if len(self.sides) != 2 or self.sides[0] == self.sides[1]:
            raise RecordError("transition.sides must name exactly two distinct record ids")
        for side in self.sides:
            require_str("transition side", side)
        require_str("transition.reason", self.reason)
        if not self.reason:
            raise RecordError("transition.reason must be non-empty")

    def model_dump(self) -> dict[str, Any]:
        return {"sides": list(self.sides), "reason": self.reason}

    @classmethod
    def model_validate(cls, data: Any) -> Transition:
        data = require_mapping("transition", data)
        return cls(sides=tuple(data.get("sides", ())), reason=data.get("reason", ""))


@dataclass(frozen=True, kw_only=True)
class MemoryRecord:
    id: str = ""  # content address; computed from birth content when empty
    type: RecordType
    body: str
    slots: dict[str, Any] = field(default_factory=dict)  # optional structured view of body
    state: RecordState = RecordState.ACTIVE
    confidence: float = 1.0
    created_at: datetime
    updated_at: datetime
    # When the remembered event happened (event time), as opposed to
    # created_at (ingest time — when the agent learned it). Optional,
    # immutable after birth, signed, NOT part of the content address, and
    # omitted from the wire form when unset so pre-existing signed records
    # re-serialize byte-identically.
    occurred_at: datetime | None = None
    provenance: Provenance
    evidence: tuple[str, ...] = ()  # record ids; REQUIRED non-empty for rules
    supersedes: str | None = None
    superseded_by: str | None = None
    transition: Transition | None = None
    state_reason: str | None = None  # why the record entered its current state
    tags: tuple[str, ...] = ()
    flagged: bool = False  # rule whose evidence was invalidated
    flag_reason: str | None = None
    signed_by: str | None = None  # signer's agent address
    signature: str = ""  # base64 ed25519 over the canonical record sans signature
    # §11 redaction tombstone: content destroyed under an explicit, audited
    # action; the record's existence, id, and link graph remain forever.
    # Emitted on the wire only when redacted, so non-redacted records
    # serialize byte-identically to pre-redaction implementations.
    redacted: bool = False
    redaction_reason: str | None = None
    birth_digest: str | None = None  # full sha256 hex of the original birth content
    # Unknown wire fields, preserved and included in the signed payload.
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_str("body", self.body)
        require_mapping("slots", self.slots)
        require_mapping("extra", self.extra)
        object.__setattr__(self, "type", RecordType(self.type))
        object.__setattr__(self, "state", RecordState(self.state))
        object.__setattr__(self, "confidence", confidence_from_wire(self.confidence))
        object.__setattr__(self, "created_at", parse_datetime("created_at", self.created_at))
        object.__setattr__(self, "updated_at", parse_datetime("updated_at", self.updated_at))
        if self.occurred_at is not None:
            object.__setattr__(
                self, "occurred_at", parse_datetime("occurred_at", self.occurred_at)
            )
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "tags", tuple(self.tags))
        for eid in self.evidence:
            require_str("evidence id", eid)
        for tag in self.tags:
            require_str("tag", tag)
        if self.type is RecordType.RULE and not self.evidence:
            raise RecordError("a rule record must carry non-empty evidence links")
        if (self.state is RecordState.TRANSITIONAL) != (self.transition is not None):
            raise RecordError(
                "transition block is required exactly when state is transitional"
            )
        if self.state is RecordState.SUPERSEDED and not self.superseded_by:
            raise RecordError("a superseded record must name its successor (superseded_by)")
        if self.superseded_by and self.state not in (
            RecordState.SUPERSEDED,
            RecordState.ARCHIVED,
        ):
            raise RecordError("superseded_by is only meaningful on superseded/archived records")
        if self.flagged and self.type is not RecordType.RULE:
            raise RecordError("only rule records can be flagged")
        # `extra` is merged over the known fields in model_dump, so a key that
        # collides with a real field would shadow it there. Wire input can't
        # do this (model_validate partitions by field name), but a hand-built
        # record could.
        shadowed = {f.name for f in fields(type(self))} & set(self.extra)
        if shadowed:
            raise RecordError(f"extra must not shadow known record fields: {sorted(shadowed)}")
        if self.redacted:
            # A tombstone's content is destroyed, so the id can no longer be
            # recomputed — it is explained by birth_digest instead, and the
            # tamper-evidence trade is deliberate (§11).
            if self.state is not RecordState.ARCHIVED:
                raise RecordError("a redacted record must be archived")
            if not self.birth_digest:
                raise RecordError("a redacted record must carry its birth_digest")
            if not self.id:
                raise RecordError("a redacted record must keep its original id")
            if not self.redaction_reason:
                raise RecordError("redaction requires a reason")
            return
        if self.birth_digest is not None or self.redaction_reason is not None:
            raise RecordError("redaction fields are only meaningful on a redacted record")
        computed = self._compute_id()
        if not self.id:
            object.__setattr__(self, "id", computed)
        elif self.id != computed:
            raise RecordError(
                f"record id does not match its content address: {self.id!r} != {computed!r}"
            )

    def _birth(self) -> dict[str, Any]:
        """The immutable birth content the id is derived from."""
        return {
            "type": self.type.value,
            "body": self.body,
            "slots": self.slots,
            "provenance": self.provenance.model_dump(),
            "created_at": canonical_timestamp(self.created_at),
        }

    def _compute_id(self) -> str:
        """Content address over the immutable birth content only — lifecycle
        fields are versioned around the id and must never change it."""
        digest = hashlib.sha256(canonical_json(self._birth())).hexdigest()
        return _ID_PREFIX + digest[:_ID_HEX_CHARS]

    def model_dump(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "type": self.type.value,
            "body": self.body,
            "slots": self.slots,
            "state": self.state.value,
            "confidence": confidence_to_wire(self.confidence),
            "created_at": canonical_timestamp(self.created_at),
            "updated_at": canonical_timestamp(self.updated_at),
            "provenance": self.provenance.model_dump(),
            "evidence": list(self.evidence),
            "supersedes": self.supersedes,
            "superseded_by": self.superseded_by,
            "transition": self.transition.model_dump() if self.transition else None,
            "state_reason": self.state_reason,
            "tags": list(self.tags),
            "flagged": self.flagged,
            "flag_reason": self.flag_reason,
            "signed_by": self.signed_by,
            "signature": self.signature,
        }
        # Optional extension fields are emitted only when set: a record that
        # never carried them re-serializes byte-identically, so existing
        # signatures keep verifying.
        if self.occurred_at is not None:
            data["occurred_at"] = canonical_timestamp(self.occurred_at)
        if self.redacted:
            data["redacted"] = True
            data["redaction_reason"] = self.redaction_reason
            data["birth_digest"] = self.birth_digest
        data.update(self.extra)
        return data

    @classmethod
    def model_validate(cls, data: Any) -> MemoryRecord:
        data = require_mapping("record", data)
        known = {f.name for f in fields(cls)}
        extra = {k: v for k, v in data.items() if k not in known}
        provenance = Provenance.model_validate(data.get("provenance", {}))
        transition = data.get("transition")
        return cls(
            id=data.get("id", ""),
            type=data.get("type", ""),
            body=data.get("body", ""),
            slots=data.get("slots", {}),
            state=data.get("state", RecordState.ACTIVE),
            confidence=data.get("confidence", "1.0000"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            occurred_at=data.get("occurred_at"),
            provenance=provenance,
            evidence=tuple(data.get("evidence", ()) or ()),
            supersedes=data.get("supersedes"),
            superseded_by=data.get("superseded_by"),
            transition=Transition.model_validate(transition) if transition else None,
            state_reason=data.get("state_reason"),
            tags=tuple(data.get("tags", ()) or ()),
            flagged=bool(data.get("flagged", False)),
            flag_reason=data.get("flag_reason"),
            signed_by=data.get("signed_by"),
            signature=data.get("signature", ""),
            redacted=bool(data.get("redacted", False)),
            redaction_reason=data.get("redaction_reason"),
            birth_digest=data.get("birth_digest"),
            extra=extra,
        )

    def model_copy(self, update: dict[str, Any] | None = None) -> MemoryRecord:
        return replace(self, **(update or {}))

    def evolve(self, update: dict[str, Any]) -> MemoryRecord:
        """A new lifecycle version of this record. A record signature covers
        the full payload, so any lifecycle mutation invalidates it — evolve
        clears ``signed_by``/``signature`` rather than carrying a stale
        signature that no longer verifies. The birth version keeps its
        signature in history; re-signing an evolved version is the engine's
        explicit choice, never implicit."""
        if self.signature or self.signed_by:
            update = {**update, "signed_by": None, "signature": ""}
        return replace(self, **update)

    @classmethod
    def create(
        cls,
        type: RecordType,
        body: str,
        provenance: Provenance,
        *,
        slots: dict[str, Any] | None = None,
        confidence: float = 1.0,
        evidence: tuple[str, ...] = (),
        supersedes: str | None = None,
        tags: tuple[str, ...] = (),
        created_at: datetime | None = None,
        occurred_at: datetime | None = None,
        extra: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """A fresh active record. ``created_at`` defaults to now (UTC);
        ``occurred_at`` is the optional event time (episodes SHOULD carry
        it) as opposed to when the agent learned it."""
        now = created_at if created_at is not None else datetime.now(UTC)
        return cls(
            type=type,
            body=body,
            slots=slots or {},
            confidence=confidence,
            created_at=now,
            updated_at=now,
            occurred_at=occurred_at,
            provenance=provenance,
            evidence=evidence,
            supersedes=supersedes,
            tags=tags,
            extra=extra or {},
        )

    REDACTED_BODY = "[redacted]"

    @classmethod
    def tombstone(
        cls,
        original: MemoryRecord,
        *,
        reason: str,
        now: datetime | None = None,
    ) -> MemoryRecord:
        """The §11 redaction tombstone for ``original``: content destroyed,
        existence preserved. Keeps the id, type, timestamps, and the link
        graph (evidence/supersession ids are not content); wipes body,
        slots, tags, event time, and provenance detail; records the full
        birth-content digest so the id stays explainable. Unsigned — the
        original signature covered content that no longer exists."""
        if original.redacted:
            raise RecordError(f"record {original.id} is already redacted")
        if not reason:
            raise RecordError("redaction requires a reason")
        digest = hashlib.sha256(canonical_json(original._birth())).hexdigest()
        return cls(
            id=original.id,
            type=original.type,
            body=cls.REDACTED_BODY,
            state=RecordState.ARCHIVED,
            confidence=0.0,
            created_at=original.created_at,
            updated_at=now if now is not None else datetime.now(UTC),
            provenance=Provenance(kind="redaction"),
            evidence=original.evidence,
            supersedes=original.supersedes,
            superseded_by=original.superseded_by,
            redacted=True,
            redaction_reason=reason,
            birth_digest=digest,
        )

    def _payload(self) -> dict[str, Any]:
        """The signed view of this record: everything except the signature."""
        data = self.model_dump()
        data.pop("signature")
        return data

    def sign(self, keys: KeyPair, address: str) -> MemoryRecord:
        """Sign this record as the agent at ``address``. The signer's address
        is inside the signed payload, so a signature cannot be re-attributed."""
        unsigned = self.model_copy(update={"signed_by": address, "signature": ""})
        signature = sign_payload(keys, CONTEXT_RECORD, unsigned._payload())
        return unsigned.model_copy(update={"signature": signature})

    def verify(self) -> None:
        """Record must be signed by the key its ``signed_by`` address certifies."""
        if not self.signed_by or not self.signature:
            raise RecordError("record is not signed")
        verify_by_address(self.signed_by, CONTEXT_RECORD, self._payload(), self.signature)


@dataclass(frozen=True, kw_only=True)
class MemoryFile:
    """THE portable artifact: a header plus records, with canonical bytes and
    an optional file-level signature by the owning agent. An agent's memory
    is a file it owns — exportable, importable, and verifiable anywhere."""

    format: str = FORMAT_VERSION
    agent: str | None = None  # owning agent's address
    created_at: datetime
    records: tuple[MemoryRecord, ...] = ()
    signature: str = ""  # base64 ed25519 over canonical file sans signature
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_str("format", self.format)
        require_mapping("extra", self.extra)
        object.__setattr__(self, "created_at", parse_datetime("created_at", self.created_at))
        object.__setattr__(self, "records", tuple(self.records))
        for record in self.records:
            if not isinstance(record, MemoryRecord):
                raise RecordError("records must be MemoryRecord instances")
        shadowed = {f.name for f in fields(type(self))} & set(self.extra)
        if shadowed:
            raise RecordError(f"extra must not shadow known file fields: {sorted(shadowed)}")

    def model_dump(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "format": self.format,
            "agent": self.agent,
            "created_at": canonical_timestamp(self.created_at),
            "records": [record.model_dump() for record in self.records],
            "signature": self.signature,
        }
        data.update(self.extra)
        return data

    @classmethod
    def model_validate(cls, data: Any) -> MemoryFile:
        data = require_mapping("memory file", data)
        known = {f.name for f in fields(cls)}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            format=data.get("format", ""),
            agent=data.get("agent"),
            created_at=data.get("created_at"),
            records=tuple(
                MemoryRecord.model_validate(item) for item in data.get("records", ())
            ),
            signature=data.get("signature", ""),
            extra=extra,
        )

    def model_copy(self, update: dict[str, Any] | None = None) -> MemoryFile:
        return replace(self, **(update or {}))

    @classmethod
    def create(
        cls,
        records: tuple[MemoryRecord, ...] | list[MemoryRecord],
        *,
        agent: str | None = None,
        created_at: datetime | None = None,
    ) -> MemoryFile:
        now = created_at if created_at is not None else datetime.now(UTC)
        return cls(agent=agent, created_at=now, records=tuple(records))

    def _payload(self) -> dict[str, Any]:
        data = self.model_dump()
        data.pop("signature")
        return data

    def canonical_bytes(self) -> bytes:
        """The deterministic bytes of this file, signature excluded — the
        form the file-level signature covers."""
        return canonical_json(self._payload())

    def sign(self, keys: KeyPair) -> MemoryFile:
        """Sign the whole file as its owning agent (``agent`` must be set)."""
        if not self.agent:
            raise RecordError("a memory file must name its agent before signing")
        signature = sign_payload(keys, CONTEXT_MEMORY_FILE, self._payload())
        return self.model_copy(update={"signature": signature})

    def verify(self) -> None:
        """File must be signed by the key its ``agent`` address certifies."""
        if not self.agent or not self.signature:
            raise RecordError("memory file is not signed")
        verify_by_address(self.agent, CONTEXT_MEMORY_FILE, self._payload(), self.signature)
