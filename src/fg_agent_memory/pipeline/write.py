"""The write stage: raw input → validated records in the store.

An extractor operator turns raw text (or a structured event) into *create*
proposals; the write stage validates every one before anything is stored.
Extraction may only create fresh fact/episode/procedure records — never
rules (rules exist only through evidence-linked promotion) and never
lifecycle mutations. A proposal whose record already exists by content
address is a reinforcement, not a duplicate: the existing record is touched
in the salience sidecar and nothing is re-written. The write stage is never
destructive; every outcome lands in the :class:`IngestReport`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..errors import StoreError
from ..ports import Operator, Proposal, ProposalKind, RecordStore
from ..records import MemoryRecord, Provenance, RecordState, RecordType
from .decay import SalienceScores
from .report import Rejection

_PASS = "write"


@dataclass(frozen=True, kw_only=True)
class IngestReport:
    """Everything one ingest call did: record ids admitted, record ids that
    reinforced an existing record instead, and proposals refused (with
    reasons). Never destructive — there is no "removed" list by design."""

    accepted: tuple[str, ...]
    deduped: tuple[str, ...]
    rejected: tuple[Rejection, ...]


def _render_event(event: Mapping[str, Any]) -> str:
    """Deterministic text view of a structured event: sorted `key: value`
    lines, so the same event always reaches the extractor as the same text."""
    return "\n".join(f"{key}: {event[key]}" for key in sorted(event))


def _validate_extraction(proposal: Proposal) -> str | None:
    """Why a write-stage proposal is inadmissible, or None if it is fine.

    The write stage admits exactly one shape: a *create* proposal carrying a
    fresh, active, unentangled record. Everything else — lifecycle mutations,
    rules, records born superseded or flagged — is the extractor overstepping
    the write stage's authority.
    """
    if proposal.kind is not ProposalKind.CREATE:
        return f"write stage admits only create proposals, got {proposal.kind.value!r}"
    record = proposal.record
    if record is None:
        return "create proposal carries no record"
    if record.type is RecordType.RULE:
        return "extraction must not create rules; rules exist only through promotion"
    if record.state is not RecordState.ACTIVE:
        return f"extracted records must be born active, got {record.state.value!r}"
    if record.supersedes or record.superseded_by or record.transition is not None:
        return "extracted records must not carry lifecycle links"
    if record.flagged:
        return "extracted records must not be born flagged"
    return None


def ingest(
    source: str | Mapping[str, Any],
    *,
    extractor: Operator,
    store: RecordStore,
    provenance: Provenance,
    now: datetime | None = None,
    scores: SalienceScores | None = None,
) -> IngestReport:
    """Run the extractor over ``source`` and admit its valid proposals.

    ``source`` is raw text or a structured event (mapping), rendered
    deterministically. Dedup is by content address: a proposed record whose
    id is already in the store reinforces the existing record (a ``touch``
    in the ``scores`` sidecar, when one is provided) instead of duplicating
    it. Invalid proposals are rejected with reasons, never repaired.
    """
    stamp = now if now is not None else datetime.now(UTC)
    text = _render_event(source) if isinstance(source, Mapping) else source

    accepted: list[str] = []
    deduped: list[str] = []
    rejected: list[Rejection] = []

    for proposal in extractor.propose(text, provenance):
        problem = _validate_extraction(proposal)
        if problem is not None:
            rejected.append(
                Rejection(pass_name=_PASS, detail=_describe(proposal), error=problem)
            )
            continue
        record = proposal.record
        assert record is not None  # _validate_extraction guarantees it
        if _exists(store, record.id):
            deduped.append(record.id)
            if scores is not None:
                scores.touch(record.id, now=stamp)
            continue
        store.put(record)
        accepted.append(record.id)

    return IngestReport(
        accepted=tuple(accepted), deduped=tuple(deduped), rejected=tuple(rejected)
    )


def _exists(store: RecordStore, record_id: str) -> bool:
    try:
        store.get(record_id)
    except StoreError:
        return False
    return True


def _describe(proposal: Proposal) -> str:
    record = proposal.record
    if isinstance(record, MemoryRecord):
        return f"{proposal.kind.value} {record.type.value} {record.id}"
    return f"{proposal.kind.value} target={proposal.target_id} other={proposal.other_id}"
