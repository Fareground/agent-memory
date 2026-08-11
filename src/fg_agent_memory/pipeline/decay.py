"""Mechanical decay: salience scoring and code-driven archival.

Decay is arithmetic, never LLM mood: salience is a pure function of
confidence, reinforcement count, and age since last touch, with every
constant named in :class:`DecayConfig`. Records whose salience falls below
the archive threshold are archived through the lifecycle machine — never
deleted — and hard protections keep load-bearing records alive: in-dispute
(transitional) records, unflagged rules, and anything an active rule cites
as evidence never decay out from under their dependents.

Reinforcement lives in the :class:`SalienceScores` sidecar rather than on
the record, deliberately: a touch count changes hourly and signing a new
record version per retrieval hit would be noise. The record format stays the
interop standard; salience is engine-local working state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

# Internal-API dependency: fg_agent_id.serde helpers are not re-exported at
# the package root; the fg-agent-id version bound in pyproject pins them.
from fg_agent_id.serde import canonical_timestamp, parse_datetime

from ..lifecycle import archive
from ..ports import RecordStore
from ..records import MemoryRecord, RecordState, RecordType
from .report import Action

_PASS = "decay"

_SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True, kw_only=True)
class DecayConfig:
    """Named decay constants. The defaults are SHOULD-level suggestions from
    the spec, not part of the interop contract — tune them per deployment.

    - ``half_life_days``: age at which an untouched record's salience halves.
    - ``reinforcement_bonus``: linear salience bonus per recorded touch.
    - ``archive_threshold``: salience below which a record is archived.
    """

    half_life_days: float = 30.0
    reinforcement_bonus: float = 0.25
    archive_threshold: float = 0.05


class SalienceScores:
    """Sidecar reinforcement counter: record id → (touches, last touched).

    This is the retrieval hook: call :meth:`touch` whenever a record is
    used, and decay reads the count and recency back out. Kept outside the
    record on purpose (see module docstring) — it is engine-local state, and
    losing it degrades gracefully to age-since-``updated_at`` decay.
    """

    def __init__(self) -> None:
        self._touches: dict[str, int] = {}
        self._last: dict[str, datetime] = {}

    def touch(self, record_id: str, *, now: datetime | None = None) -> None:
        """Record one reinforcement of ``record_id`` at ``now``."""
        stamp = now if now is not None else datetime.now(UTC)
        self._touches[record_id] = self._touches.get(record_id, 0) + 1
        self._last[record_id] = stamp

    def touches(self, record_id: str) -> int:
        return self._touches.get(record_id, 0)

    def last_touched(self, record_id: str) -> datetime | None:
        return self._last.get(record_id)

    def to_dict(self) -> dict[str, dict[str, object]]:
        """JSON-ready snapshot, so an engine can persist the sidecar and
        reinforcement survives a restart (losing it still degrades
        gracefully to age-since-update decay)."""
        return {
            record_id: {
                "touches": count,
                "last_touched": canonical_timestamp(self._last[record_id]),
            }
            for record_id, count in sorted(self._touches.items())
        }

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, object]]) -> SalienceScores:
        scores = cls()
        for record_id, entry in data.items():
            count = int(entry.get("touches", 0))
            if count < 1:
                continue
            scores._touches[record_id] = count
            scores._last[record_id] = parse_datetime(
                "last_touched", entry.get("last_touched")
            )
        return scores


@dataclass(frozen=True, kw_only=True)
class DecayReport:
    """What one decay pass did: records archived, records that scored below
    threshold but were protected (with the protection named), and the
    salience computed for every record considered."""

    archived: tuple[Action, ...]
    protected: tuple[Action, ...]
    saliences: dict[str, float]


def salience(
    record: MemoryRecord,
    *,
    touches: int,
    last_touched: datetime | None,
    now: datetime,
    config: DecayConfig,
) -> float:
    """Pure salience score: confidence, halved per half-life of idleness,
    boosted linearly per reinforcement touch. Age counts from the most
    recent of the record's ``updated_at`` and its last recorded touch."""
    reference = record.updated_at
    if last_touched is not None and last_touched > reference:
        reference = last_touched
    age_days = max(0.0, (now - reference).total_seconds() / _SECONDS_PER_DAY)
    freshness = 0.5 ** (age_days / config.half_life_days)
    return record.confidence * freshness * (1.0 + config.reinforcement_bonus * touches)


def _protection(
    record: MemoryRecord, evidence_of_active_rules: frozenset[str]
) -> str | None:
    """Why this record must not be archived, or None if it may decay."""
    if record.state is RecordState.TRANSITIONAL:
        return "in an unresolved transition; both sides must stay visible"
    if record.type is RecordType.RULE and not record.flagged:
        return "unflagged rule; rules decay only after their evidence is invalidated"
    if record.id in evidence_of_active_rules:
        return "cited as evidence by an active rule"
    return None


def decay(
    store: RecordStore,
    *,
    scores: SalienceScores | None = None,
    now: datetime | None = None,
    config: DecayConfig | None = None,
) -> DecayReport:
    """One mechanical decay pass over the store.

    Considers active and superseded records (transitional records are
    protected in place; archived is terminal). Anything scoring below
    ``config.archive_threshold`` is archived via the lifecycle machine
    unless a protection applies — protections are reported, not silent.
    """
    stamp = now if now is not None else datetime.now(UTC)
    cfg = config if config is not None else DecayConfig()
    board = scores if scores is not None else SalienceScores()

    evidence_of_active_rules = frozenset(
        eid
        for rule in store.list(state=RecordState.ACTIVE, type=RecordType.RULE)
        for eid in rule.evidence
    )

    candidates = sorted(
        list(store.list(state=RecordState.ACTIVE))
        + list(store.list(state=RecordState.SUPERSEDED))
        + list(store.list(state=RecordState.TRANSITIONAL)),
        key=lambda record: record.id,
    )

    archived: list[Action] = []
    protected: list[Action] = []
    saliences: dict[str, float] = {}

    for record in candidates:
        score = salience(
            record,
            touches=board.touches(record.id),
            last_touched=board.last_touched(record.id),
            now=stamp,
            config=cfg,
        )
        saliences[record.id] = score
        if score >= cfg.archive_threshold:
            continue
        shield = _protection(record, evidence_of_active_rules)
        if shield is not None:
            protected.append(
                Action(
                    pass_name=_PASS,
                    action="protect",
                    record_ids=(record.id,),
                    before=(record.state.value,),
                    after=(record.state.value,),
                    reason=f"salience {score:.4f} below threshold but {shield}",
                )
            )
            continue
        reason = f"salience {score:.4f} fell below archive threshold {cfg.archive_threshold}"
        decayed = archive(record, reason=reason, now=stamp)
        store.put(decayed)
        archived.append(
            Action(
                pass_name=_PASS,
                action="archive",
                record_ids=(record.id,),
                before=(record.state.value,),
                after=(decayed.state.value,),
                reason=reason,
            )
        )

    return DecayReport(
        archived=tuple(archived), protected=tuple(protected), saliences=saliences
    )
