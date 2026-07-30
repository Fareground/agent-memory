"""Retrieval: the state-aware, budgeted read stage of the pipeline.

Indexes answer "what looks like this query"; retrieval turns that into what
an agent should actually load, and the difference is the whole point of the
lifecycle. Each candidate is scored

    score = relevance × state_weight × recency × salience × confidence

where relevance comes from the SearchIndex ports, the state weight makes
active outrank transitional outrank superseded at equal relevance, recency
decays from ``updated_at`` with a floor (old truth is discounted, never
erased), and salience grows with how often a record has been retrieved
before — the retrieval-reinforcement hook, tracked by a :class:`TouchCounts`
sidecar the caller owns.

State semantics are non-negotiable here:

- **Transitional records surface both sides.** Including one side of an
  unresolved contradiction pulls its transition partner into the same block —
  an agent must never see half of a dispute as if it were settled truth.
- **Superseded records ride the provenance trail.** They are excluded as
  direct candidates unless history is explicitly requested, in which case an
  included record's ``supersedes`` chain is walked and surfaced with it.
- **Archived records are excluded by default** and includable by flag; decay
  means losing the competition for attention, not addressability.

The token budget selects whole records — compression here is selection,
never rewriting — and every included record carries human-readable reasons,
so a retrieval result is explainable after the fact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .errors import StoreError
from .ports import RecordStore, SearchIndex
from .records import MemoryRecord, RecordState

# ~4 characters per token is the standard rough heuristic for English text;
# budgets here are advisory ceilings, not tokenizer-exact accounting.
APPROX_CHARS_PER_TOKEN = 4
# Recency half-life: a record untouched for this many days scores half.
RECENCY_HALF_LIFE_DAYS = 30.0
# Recency never decays below this floor — old records stay findable.
RECENCY_FLOOR = 0.1
# Touches needed for salience to reach half of its maximum boost.
SALIENCE_HALF_SATURATION = 5.0
# State weights: the invariant is the strict ordering
# active > transitional > superseded > archived, not the constants.
STATE_WEIGHTS: dict[RecordState, float] = {
    RecordState.ACTIVE: 1.0,
    RecordState.TRANSITIONAL: 0.8,
    RecordState.SUPERSEDED: 0.3,
    RecordState.ARCHIVED: 0.1,
}
# How many candidates to pull from each index before scoring.
DEFAULT_CANDIDATES_PER_INDEX = 32


def approx_tokens(record: MemoryRecord) -> int:
    """Rough token cost of loading a record (its body), never below one."""
    return max(1, len(record.body) // APPROX_CHARS_PER_TOKEN)


class TouchCounts:
    """The retrieval-reinforcement sidecar: how often each record has been
    retrieved. ``retrieve`` bumps it for every record it returns, so records
    that keep proving useful float upward. The caller owns persistence —
    the counts are salience state, not part of the signed record."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def touch(self, record_id: str) -> None:
        self._counts[record_id] = self._counts.get(record_id, 0) + 1

    def count(self, record_id: str) -> int:
        return self._counts.get(record_id, 0)


@dataclass(frozen=True, kw_only=True)
class RetrievedRecord:
    """One record in a retrieval result, with the score that ranked it, its
    token cost, and why it is here."""

    record: MemoryRecord
    score: float
    tokens: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class RetrievedBlock:
    """The ranked, budgeted result of one retrieval."""

    query: str
    records: tuple[RetrievedRecord, ...] = ()
    budget_tokens: int = 0
    total_tokens: int = 0

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(item.record.id for item in self.records)


@dataclass
class _Unit:
    """A primary candidate plus the companions that must travel with it
    (transition partner, provenance-trail predecessors). Budgeted atomically:
    either the whole unit fits or none of it is included."""

    members: list[RetrievedRecord] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(member.tokens for member in self.members)


def retrieve(
    query: str,
    *,
    store: RecordStore,
    indexes: Sequence[SearchIndex],
    budget_tokens: int,
    now: datetime | None = None,
    include_archived: bool = False,
    include_history: bool = False,
    touches: TouchCounts | None = None,
    candidates_per_index: int = DEFAULT_CANDIDATES_PER_INDEX,
) -> RetrievedBlock:
    """Run the retrieval stage: union index candidates, score them with
    state/recency/salience/confidence, and pack whole records into the token
    budget, best first. See the module docstring for the semantics."""
    moment = now if now is not None else datetime.now(UTC)
    relevance = _union_candidates(query, indexes, candidates_per_index)

    scored: list[tuple[float, str, MemoryRecord]] = []
    for record_id in sorted(relevance):
        record = _get_or_none(store, record_id)
        if record is None:
            continue  # index knows an id the store no longer serves
        if record.state is RecordState.ARCHIVED and not include_archived:
            continue
        if record.state is RecordState.SUPERSEDED and not include_history:
            continue
        score = _score(record, relevance[record_id], moment, touches)
        if score <= 0.0:
            continue
        scored.append((score, record_id, record))
    scored.sort(key=lambda item: (-item[0], item[1]))

    selected: list[RetrievedRecord] = []
    seen: set[str] = set()
    used = 0
    for score, record_id, record in scored:
        if record_id in seen:
            continue
        unit = _build_unit(
            record,
            score,
            relevance[record_id],
            store,
            seen,
            include_history,
            moment,
            touches,
        )
        if used + unit.tokens > budget_tokens:
            continue  # whole-record budgeting: a unit that does not fit is skipped
        used += unit.tokens
        for member in unit.members:
            seen.add(member.record.id)
            selected.append(member)
            if touches is not None:
                touches.touch(member.record.id)

    return RetrievedBlock(
        query=query,
        records=tuple(selected),
        budget_tokens=budget_tokens,
        total_tokens=used,
    )


def _union_candidates(
    query: str, indexes: Sequence[SearchIndex], k: int
) -> dict[str, float]:
    """Each id's best relevance across all indexes, floored at zero.

    Raw scores from different index families are incomparable (bm25 is
    unbounded, cosine lives in [-1, 1]), so each index's positive scores are
    normalized by that index's best score before combining — every index
    speaks in [0, 1] and no backend dominates by scale alone."""
    best: dict[str, float] = {}
    for index in indexes:
        candidates = [
            (record_id, score)
            for record_id, score in index.candidates(query, k)
            if score > 0.0
        ]
        if not candidates:
            continue
        top = max(score for _, score in candidates)
        for record_id, score in candidates:
            normalized = score / top
            if normalized > best.get(record_id, 0.0):
                best[record_id] = normalized
    return best


def _score(
    record: MemoryRecord,
    relevance: float,
    now: datetime,
    touches: TouchCounts | None,
) -> float:
    age_days = max(0.0, (now - record.updated_at).total_seconds() / 86400.0)
    recency = max(RECENCY_FLOOR, 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS))
    salience = 1.0
    if touches is not None:
        count = touches.count(record.id)
        salience += count / (count + SALIENCE_HALF_SATURATION)
    return relevance * STATE_WEIGHTS[record.state] * recency * salience * record.confidence


def _build_unit(
    record: MemoryRecord,
    score: float,
    relevance: float,
    store: RecordStore,
    seen: set[str],
    include_history: bool,
    moment: datetime,
    touches: TouchCounts | None,
) -> _Unit:
    """Companions ride along because of the primary, but each is scored on
    its own state, recency, salience, and confidence — a superseded
    predecessor never reports its successor's score."""
    unit = _Unit()
    unit_ids = {record.id}
    unit.members.append(
        RetrievedRecord(
            record=record,
            score=score,
            tokens=approx_tokens(record),
            reasons=(f"matched query (relevance {relevance:.4f})",),
        )
    )

    if record.transition is not None:
        partner_id = next(side for side in record.transition.sides if side != record.id)
        partner = _get_or_none(store, partner_id) if partner_id not in seen else None
        if partner is not None:
            unit_ids.add(partner.id)
            unit.members.append(
                RetrievedRecord(
                    record=partner,
                    score=_score(partner, relevance, moment, touches),
                    tokens=approx_tokens(partner),
                    reasons=(f"other side of the transition with {record.id}",),
                )
            )

    if include_history:
        successor = record
        while successor.supersedes:
            if successor.supersedes in seen or successor.supersedes in unit_ids:
                break
            predecessor = _get_or_none(store, successor.supersedes)
            if predecessor is None:
                break
            unit_ids.add(predecessor.id)
            unit.members.append(
                RetrievedRecord(
                    record=predecessor,
                    score=_score(predecessor, relevance, moment, touches),
                    tokens=approx_tokens(predecessor),
                    reasons=(f"superseded predecessor of {successor.id}",),
                )
            )
            successor = predecessor

    return unit


def _get_or_none(store: RecordStore, record_id: str) -> MemoryRecord | None:
    try:
        return store.get(record_id)
    except StoreError:
        return None
