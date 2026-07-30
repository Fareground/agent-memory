"""The fact lifecycle as a code-enforced state machine.

Every record carries an explicit state; the transitions between states are
deterministic and enforced here — never by convention, never by an LLM.
Superseding names its successor and reason; contradiction produces a
transition pair, not a silent overwrite; archival is decay, and records are
never deleted. This is the direct attack on the consolidation failure mode:
ghost memory cannot form when stale/current/in-dispute are machine states.

All functions are pure: they take records, validate the transition, and
return new record versions. No storage dependency — every invariant here is
stateable over the record format alone (a rule that only exists as a query on
some backend would be a leak).
"""

from __future__ import annotations

from datetime import UTC, datetime

from .errors import LifecycleError
from .records import MemoryRecord, RecordState, RecordType, Transition

# The complete legality table. Anything not listed raises.
#   active → superseded      (must name successor + reason)
#   active → transitional    (must carry the transition block, both sides)
#   transitional → active    (resolution names the winner)
#   transitional → superseded (resolution's loser)
#   any non-archived → archived (decay only; archived is terminal)
LEGAL_TRANSITIONS: frozenset[tuple[RecordState, RecordState]] = frozenset(
    {
        (RecordState.ACTIVE, RecordState.SUPERSEDED),
        (RecordState.ACTIVE, RecordState.TRANSITIONAL),
        (RecordState.TRANSITIONAL, RecordState.ACTIVE),
        (RecordState.TRANSITIONAL, RecordState.SUPERSEDED),
        (RecordState.ACTIVE, RecordState.ARCHIVED),
        (RecordState.SUPERSEDED, RecordState.ARCHIVED),
        (RecordState.TRANSITIONAL, RecordState.ARCHIVED),
    }
)


def is_legal(current: RecordState, target: RecordState) -> bool:
    return (RecordState(current), RecordState(target)) in LEGAL_TRANSITIONS


def assert_transition(current: RecordState, target: RecordState) -> None:
    if not is_legal(current, target):
        raise LifecycleError(f"illegal lifecycle transition: {current} -> {target}")


def _now(now: datetime | None) -> datetime:
    return now if now is not None else datetime.now(UTC)


def supersede(
    old: MemoryRecord,
    successor: MemoryRecord,
    *,
    reason: str,
    now: datetime | None = None,
) -> MemoryRecord:
    """Retire ``old`` in favor of ``successor``; returns the superseded old.

    The successor must already reference the record it replaces
    (``successor.supersedes == old.id``) — supersession without a named
    predecessor is exactly the silent overwrite this standard forbids.
    The superseded record stays addressable forever.
    """
    assert_transition(old.state, RecordState.SUPERSEDED)
    if not reason:
        raise LifecycleError("supersession requires a reason")
    if successor.supersedes != old.id:
        raise LifecycleError(
            f"successor {successor.id} does not reference the record it supersedes "
            f"({old.id}); set supersedes on the successor first"
        )
    if successor.state is not RecordState.ACTIVE:
        raise LifecycleError("a successor must be an active record")
    return old.evolve(
        update={
            "state": RecordState.SUPERSEDED,
            "superseded_by": successor.id,
            "state_reason": reason,
            "updated_at": _now(now),
        }
    )


def begin_transition(
    a: MemoryRecord,
    b: MemoryRecord,
    *,
    reason: str,
    now: datetime | None = None,
) -> tuple[MemoryRecord, MemoryRecord]:
    """Mark two contradicting records transitional. Both sides carry the same
    transition block, so either record leads a reader to the other."""
    assert_transition(a.state, RecordState.TRANSITIONAL)
    assert_transition(b.state, RecordState.TRANSITIONAL)
    if a.id == b.id:
        raise LifecycleError("a record cannot contradict itself")
    if not reason:
        raise LifecycleError("a transition requires a reason")
    block = Transition(sides=(a.id, b.id), reason=reason)
    stamp = _now(now)
    update = {"state": RecordState.TRANSITIONAL, "transition": block, "updated_at": stamp}
    return a.evolve(update), b.evolve(update)


def resolve_transition(
    winner: MemoryRecord,
    loser: MemoryRecord,
    *,
    reason: str,
    now: datetime | None = None,
) -> tuple[MemoryRecord, MemoryRecord]:
    """Resolve a transition pair: the winner returns to active, the loser is
    superseded by it. Both records must be the two sides of the same
    transition block."""
    assert_transition(winner.state, RecordState.ACTIVE)
    assert_transition(loser.state, RecordState.SUPERSEDED)
    if not reason:
        raise LifecycleError("resolution requires a reason")
    if winner.transition is None or loser.transition is None:
        raise LifecycleError("both records must carry a transition block")
    if set(winner.transition.sides) != {winner.id, loser.id} or (
        winner.transition.sides != loser.transition.sides
    ):
        raise LifecycleError(
            "winner and loser are not the two sides of the same transition"
        )
    stamp = _now(now)
    resolved_winner = winner.evolve(
        update={
            "state": RecordState.ACTIVE,
            "transition": None,
            "state_reason": reason,
            "updated_at": stamp,
        }
    )
    resolved_loser = loser.evolve(
        update={
            "state": RecordState.SUPERSEDED,
            "transition": None,
            "superseded_by": winner.id,
            "state_reason": reason,
            "updated_at": stamp,
        }
    )
    return resolved_winner, resolved_loser


def archive(
    record: MemoryRecord,
    *,
    reason: str,
    now: datetime | None = None,
) -> MemoryRecord:
    """Decay a record into the archive. Archived is terminal — the record
    stays addressable forever, it just no longer competes for retrieval.
    Records are never deleted."""
    assert_transition(record.state, RecordState.ARCHIVED)
    if not reason:
        raise LifecycleError("archival requires a reason")
    return record.evolve(
        update={
            "state": RecordState.ARCHIVED,
            "transition": None,
            "state_reason": reason,
            "updated_at": _now(now),
        }
    )


def refresh_rule_flag(
    rule: MemoryRecord,
    evidence_records: dict[str, MemoryRecord],
    *,
    now: datetime | None = None,
) -> MemoryRecord:
    """Re-evaluate a rule against the current state of its evidence.

    A rule whose every evidence record has been invalidated (archived or
    superseded) is flagged — it still applies until an operator supersedes or
    archives it, but the flag is machine-visible: invalidate the evidence and
    the rule cannot silently keep its authority. Any live evidence (active or
    transitional) clears the flag.
    """
    if rule.type is not RecordType.RULE:
        raise LifecycleError(f"only rule records carry evidence flags, got {rule.type}")
    missing = [eid for eid in rule.evidence if eid not in evidence_records]
    if missing:
        raise LifecycleError(f"evidence records not supplied for rule {rule.id}: {missing}")
    invalidated = {RecordState.ARCHIVED, RecordState.SUPERSEDED}
    all_invalid = all(
        evidence_records[eid].state in invalidated for eid in rule.evidence
    )
    if all_invalid == rule.flagged:
        return rule
    update: dict[str, object] = {"flagged": all_invalid, "updated_at": _now(now)}
    update["flag_reason"] = (
        "all evidence records are archived or superseded" if all_invalid else None
    )
    return rule.evolve(update)
