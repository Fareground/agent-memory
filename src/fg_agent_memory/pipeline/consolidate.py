"""The consolidation engine: the background pass that keeps memory sane.

Consolidation is composed of deterministic sub-passes — exact dedup,
near-dup merge, contradiction detection, resolution, promotion, and the
rule-flag sweep — each producing typed proposals that are validated against
the lifecycle machine before apply. This is the "operator inside the
pipeline" principle made concrete: an operator output that names an illegal
transition (or oversteps its sub-pass's authority) is rejected and logged in
the report, never applied and never repaired.

Detection quality is an operator port per sub-pass; the shipped defaults are
deliberately dumb heuristics (trigram Jaccard, slot/polarity matching) so
the whole engine runs and is testable with zero model dependency. Swap in an
LLM operator and the validation gate is exactly the same.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from typing import Any

from ..errors import AgentMemoryError, StoreError
from ..lifecycle import refresh_rule_flag, resolve_transition, supersede
from ..ports import Proposal, ProposalKind, RecordStore, apply_proposal
from ..records import MemoryRecord, Provenance, RecordState, RecordType
from ..text import content_terms as _content_terms
from .report import Action, Rejection

_KIND_CONSOLIDATION = "consolidation"

# Jaccard overlap (over stemmed content words) at or above which two
# opposite-polarity bodies are treated as talking about the same core
# claim. SHOULD-level default per SPEC §9.2.
_CORE_OVERLAP_THRESHOLD = 0.5


@dataclass(frozen=True, kw_only=True)
class ConsolidationConfig:
    """Tunables for the deterministic sub-passes. Defaults are SHOULD-level
    starting points, not interop requirements.

    - ``near_dup_threshold``: trigram-Jaccard similarity at or above which
      two records are near-duplicates.
    - ``promotion_threshold``: episodes sharing a signature required before
      a rule is proposed.
    """

    near_dup_threshold: float = 0.6
    promotion_threshold: int = 3


@dataclass(frozen=True, kw_only=True)
class ConsolidationReport:
    """The audit trail of one consolidation pass: every mutation applied
    (with before/after states and reasons) and every operator proposal
    rejected. A consolidation that cannot be reconstructed from its report
    did not happen."""

    actions: tuple[Action, ...]
    rejections: tuple[Rejection, ...]
    # Every record id written during the pass (new versions and new records),
    # so callers can refresh indexes/sidecars incrementally instead of
    # rescanning the whole store.
    touched: tuple[str, ...] = ()

    def by_pass(self, pass_name: str) -> tuple[Action, ...]:
        return tuple(action for action in self.actions if action.pass_name == pass_name)


class NearDupOperator(ABC):
    """Near-duplicate port: given a candidate pair, propose a merge (a
    supersede proposal whose merged record carries evidence links to both
    sides) or abstain with None. The pipeline validates the proposal shape
    and the lifecycle transition before anything is written."""

    @abstractmethod
    def propose_merge(
        self, a: MemoryRecord, b: MemoryRecord, *, now: datetime
    ) -> Proposal | None: ...


class ContradictionOperator(ABC):
    """Contradiction-detection port: given a candidate pair, propose moving
    it into the transitional protocol (a transition proposal over exactly
    these two ids) or abstain. Detection NEVER resolves — a detected
    contradiction makes both sides visible; it never silently overwrites."""

    @abstractmethod
    def propose_contradiction(
        self, a: MemoryRecord, b: MemoryRecord
    ) -> Proposal | None: ...


class ResolutionOperator(ABC):
    """Auto-resolution port: given a transitional pair, name a winner and a
    reason, or abstain (None) to leave the dispute for an explicit
    :func:`resolve` call. Resolution is applied through the lifecycle
    machine; the loser is superseded-but-addressable, never erased."""

    @abstractmethod
    def propose_winner(
        self, a: MemoryRecord, b: MemoryRecord
    ) -> tuple[str, str] | None:
        """(winner_id, reason) or None."""


def _trigrams(text: str) -> frozenset[str]:
    normalized = " ".join(text.lower().split())
    if len(normalized) < 3:
        return frozenset({normalized} if normalized else ())
    return frozenset(normalized[i : i + 3] for i in range(len(normalized) - 2))


def trigram_similarity(a: str, b: str) -> float:
    """Jaccard similarity over character trigrams — the model-free
    near-duplicate heuristic."""
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _negation_contradiction(
    body_a: str, body_b: str, *, overlap_threshold: float = _CORE_OVERLAP_THRESHOLD
) -> bool:
    """True when two bodies carry opposite polarity over a shared content
    core: exactly one side is negated (negator tokens or a "no X needed"
    polarity frame), and their stemmed content words overlap at or above
    the threshold (Jaccard). Deterministic and model-free — this is the
    shape the default contradiction pass claims. Paraphrase or semantic
    contradictions without a negation marker need an LLM operator on the
    ContradictionOperator port."""
    terms_a, negative_a = _content_terms(body_a)
    terms_b, negative_b = _content_terms(body_b)
    if negative_a == negative_b or not terms_a or not terms_b:
        return False
    overlap = len(terms_a & terms_b) / len(terms_a | terms_b)
    return overlap >= overlap_threshold


def _strength(record: MemoryRecord) -> tuple[float, datetime, str]:
    """Deterministic strength ordering: confidence, then recency, then id."""
    return (record.confidence, record.created_at, record.id)


class TrigramNearDupOperator(NearDupOperator):
    """Default near-dup operator: trigram Jaccard at/above the threshold →
    merge proposal. The merged record keeps the stronger side's body, unions
    slots and tags, links BOTH sides as evidence, and supersedes the weaker.

    Two deliberate abstentions: episodes are never merged (similar text on
    distinct events is not duplication — exact dedup covers true copies),
    and a pair whose shared slots disagree is a *contradiction* candidate —
    merging it away would be exactly the silent overwrite this standard
    forbids, so the contradiction pass gets it instead."""

    def __init__(self, *, threshold: float) -> None:
        self._threshold = threshold

    def propose_merge(
        self, a: MemoryRecord, b: MemoryRecord, *, now: datetime
    ) -> Proposal | None:
        if a.type is RecordType.EPISODE:
            return None
        if any(a.slots[key] != b.slots[key] for key in set(a.slots) & set(b.slots)):
            return None
        # A pair that differs by negation is a contradiction candidate;
        # merging it away would be the silent overwrite this standard forbids.
        if _negation_contradiction(a.body, b.body):
            return None
        if trigram_similarity(a.body, b.body) < self._threshold:
            return None
        strong, weak = sorted((a, b), key=_strength, reverse=True)
        merged = MemoryRecord.create(
            strong.type,
            strong.body,
            Provenance(kind=_KIND_CONSOLIDATION, ref=f"merge:{weak.id}"),
            slots={**weak.slots, **strong.slots},
            confidence=strong.confidence,
            evidence=tuple(sorted((strong.id, weak.id))),
            supersedes=weak.id,
            tags=tuple(sorted(set(strong.tags) | set(weak.tags))),
            created_at=now,
        )
        return Proposal(
            kind=ProposalKind.SUPERSEDE,
            record=merged,
            reason=f"near-duplicate of {strong.id} (trigram similarity ≥ {self._threshold})",
        )


class HeuristicContradictionOperator(ContradictionOperator):
    """Default contradiction detector: same-slot different-value, or a
    negation-polarity pair — bodies with opposite polarity over a shared
    stemmed content-word core (see :func:`_negation_contradiction`). This
    deliberately catches only negation-marked contradictions; semantic or
    paraphrase contradictions without a polarity marker need an LLM
    operator plugged into the :class:`ContradictionOperator` port."""

    def propose_contradiction(
        self, a: MemoryRecord, b: MemoryRecord
    ) -> Proposal | None:
        reason = self._detect(a, b)
        if reason is None:
            return None
        return Proposal(
            kind=ProposalKind.TRANSITION, target_id=a.id, other_id=b.id, reason=reason
        )

    def _detect(self, a: MemoryRecord, b: MemoryRecord) -> str | None:
        for key in sorted(set(a.slots) & set(b.slots)):
            if a.slots[key] != b.slots[key]:
                return (
                    f"slot {key!r} disagrees: {a.slots[key]!r} vs {b.slots[key]!r}"
                )
        if _negation_contradiction(a.body, b.body):
            return "one side negates the other"
        return None


class HeuristicResolutionOperator(ResolutionOperator):
    """Default auto-resolver, deliberately cautious about temporal
    precedence: an explicit user correction beats inference outright, but
    "newer wins" applies ONLY when both sides share the same provenance kind
    and the same slot keys — recency across different sources or shapes is
    not evidence of truth."""

    CORRECTION_KIND = "correction"

    def propose_winner(
        self, a: MemoryRecord, b: MemoryRecord
    ) -> tuple[str, str] | None:
        a_corr = a.provenance.kind == self.CORRECTION_KIND
        b_corr = b.provenance.kind == self.CORRECTION_KIND
        if a_corr != b_corr:
            winner = a if a_corr else b
            return winner.id, "explicit user correction beats inference"
        if a.provenance.kind == b.provenance.kind and set(a.slots) == set(b.slots):
            # "Newer" means the newer EVENT when both sides carry event
            # time; ingest order is only a proxy for it.
            if a.occurred_at is not None and b.occurred_at is not None:
                if a.occurred_at != b.occurred_at:
                    winner = a if a.occurred_at > b.occurred_at else b
                    return (
                        winner.id,
                        "newer event wins (same provenance kind, identical slots)",
                    )
                return None
            if a.created_at != b.created_at:
                winner = a if a.created_at > b.created_at else b
                return (
                    winner.id,
                    "newer assertion wins (same provenance kind, identical slots)",
                )
        return None


@dataclass(frozen=True, kw_only=True)
class ConsolidationOperators:
    """The operator ports one consolidation pass runs with. ``resolver`` is
    optional — without one, detected contradictions stay transitional until
    an explicit :func:`resolve`."""

    near_dup: NearDupOperator
    contradiction: ContradictionOperator
    resolver: ResolutionOperator | None = None


def default_operators(
    config: ConsolidationConfig | None = None,
) -> ConsolidationOperators:
    cfg = config if config is not None else ConsolidationConfig()
    return ConsolidationOperators(
        near_dup=TrigramNearDupOperator(threshold=cfg.near_dup_threshold),
        contradiction=HeuristicContradictionOperator(),
        resolver=HeuristicResolutionOperator(),
    )


def resolve(
    store: RecordStore,
    *,
    winner_id: str,
    loser_id: str,
    reason: str,
    now: datetime | None = None,
) -> tuple[MemoryRecord, MemoryRecord]:
    """Explicitly resolve a transitional pair: the winner returns to active,
    the loser becomes superseded-but-addressable with the reason on record.
    All lifecycle invariants (same transition block, legal states) apply."""
    stamp = now if now is not None else datetime.now(UTC)
    winner, loser = store.get(winner_id), store.get(loser_id)
    resolved_winner, resolved_loser = resolve_transition(
        winner, loser, reason=reason, now=stamp
    )
    store.put(resolved_winner)
    store.put(resolved_loser)
    return resolved_winner, resolved_loser


@dataclass
class _Run:
    """Mutable working state of one consolidation pass."""

    store: RecordStore
    now: datetime
    config: ConsolidationConfig
    actions: list[Action] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)

    def act(
        self,
        pass_name: str,
        action: str,
        records_before: tuple[MemoryRecord, ...],
        records_after: tuple[MemoryRecord, ...],
        reason: str,
    ) -> None:
        self.actions.append(
            Action(
                pass_name=pass_name,
                action=action,
                record_ids=tuple(r.id for r in records_after),
                before=tuple(r.state.value for r in records_before),
                after=tuple(r.state.value for r in records_after),
                reason=reason,
            )
        )

    def reject(self, pass_name: str, detail: str, error: str) -> None:
        self.rejections.append(
            Rejection(pass_name=pass_name, detail=detail, error=error)
        )


def _active(store: RecordStore) -> list[MemoryRecord]:
    return sorted(store.list(state=RecordState.ACTIVE), key=lambda r: r.id)


def _candidate_pairs(
    records: list[MemoryRecord],
) -> list[tuple[MemoryRecord, MemoryRecord]]:
    """Blocked candidate generation instead of all-pairs comparison.

    Two records are worth comparing only when they share at least one
    stemmed content word or slot key — the same signal both default
    operators score on (trigram overlap implies shared words; contradiction
    requires shared core terms or a shared slot). Blocking on that signal
    keeps a pass near-linear on real stores where all-pairs would be
    quadratic, with identical detections for the shipped heuristics.
    Deterministic: records arrive id-sorted and pairs are emitted id-sorted.
    """
    blocks: dict[str, list[int]] = {}
    for position, record in enumerate(records):
        terms, _ = _content_terms(record.body)
        keys = set(terms) | {f"slot:{key}" for key in record.slots}
        for key in keys:
            blocks.setdefault(key, []).append(position)
    pair_positions: set[tuple[int, int]] = set()
    for members in blocks.values():
        if len(members) < 2:
            continue
        pair_positions.update(combinations(members, 2))
    return [(records[a], records[b]) for a, b in sorted(pair_positions)]


def _content_key(record: MemoryRecord) -> tuple[Any, ...]:
    return (
        record.type.value,
        record.body,
        tuple(sorted((k, repr(v)) for k, v in record.slots.items())),
    )


def _pass_exact_dedup(run: _Run) -> None:
    """Pure-code dedup: identical (type, body, slots) under different content
    addresses (differing provenance/birth time) collapse to the strongest
    copy; each weaker copy is superseded by a new version of the keeper that
    names it. Existing-id dedup never reaches here — the write stage turns
    that into a reinforcement touch."""
    groups: dict[tuple[Any, ...], list[MemoryRecord]] = {}
    for record in _active(run.store):
        groups.setdefault(_content_key(record), []).append(record)
    for key in sorted(groups, key=repr):
        group = groups[key]
        if len(group) < 2:
            continue
        group.sort(key=_strength, reverse=True)
        keeper, losers = group[0], group[1:]
        for loser in losers:
            successor = keeper.evolve({"supersedes": loser.id, "updated_at": run.now})
            retired = supersede(
                loser,
                successor,
                reason=f"exact duplicate of {keeper.id}",
                now=run.now,
            )
            run.store.put(successor)
            run.store.put(retired)
            run.act(
                "dedup.exact",
                "supersede",
                (loser,),
                (retired,),
                f"exact duplicate of {keeper.id}",
            )
            keeper = successor


def _validate_merge(
    proposal: Proposal, a: MemoryRecord, b: MemoryRecord
) -> str | None:
    """A near-dup operator may only ask for one thing: a supersede whose
    merged record links both sides as evidence and replaces one of them."""
    if proposal.kind is not ProposalKind.SUPERSEDE:
        return "near-dup operator may only propose supersede merges"
    record = proposal.record
    if record is None:
        return "merge proposal carries no record"
    if record.supersedes not in (a.id, b.id):
        return "merged record must supersede one side of the pair"
    if not {a.id, b.id} <= set(record.evidence):
        return "merged record must carry evidence links to BOTH sides"
    return None


def _pass_near_dup(run: _Run, operator: NearDupOperator) -> None:
    superseded: set[str] = set()
    records = _active(run.store)
    for a, b in _candidate_pairs(records):
        if a.id in superseded or b.id in superseded or a.type is not b.type:
            continue
        proposal = operator.propose_merge(a, b, now=run.now)
        if proposal is None:
            continue
        problem = _validate_merge(proposal, a, b)
        if problem is not None:
            run.reject("dedup.near", f"merge of {a.id}+{b.id}", problem)
            continue
        merged = proposal.record
        assert merged is not None
        weak = a if merged.supersedes == a.id else b
        strong = b if weak is a else a
        try:
            written = apply_proposal(run.store, proposal)
        except AgentMemoryError as exc:
            run.reject("dedup.near", f"merge of {a.id}+{b.id}", str(exc))
            continue
        run.act(
            "dedup.near",
            "supersede",
            (weak,),
            written[1:],
            proposal.reason,
        )
        # Collapse the stronger side under the merged record too, so one
        # pass leaves a single active survivor. Forward `supersedes` points
        # at the stronger side; the weaker side keeps its superseded_by
        # back-pointer, which is the authoritative chain.
        successor = merged.evolve({"supersedes": strong.id, "updated_at": run.now})
        retired_strong = supersede(
            strong, successor, reason=f"merged into {merged.id}", now=run.now
        )
        run.store.put(successor)
        run.store.put(retired_strong)
        run.act(
            "dedup.near",
            "supersede",
            (strong,),
            (retired_strong,),
            f"merged into {merged.id}",
        )
        superseded.update({weak.id, strong.id})


def _validate_contradiction(
    proposal: Proposal, a: MemoryRecord, b: MemoryRecord
) -> str | None:
    if proposal.kind is not ProposalKind.TRANSITION:
        return "contradiction operator may only propose transitions"
    if {proposal.target_id, proposal.other_id} != {a.id, b.id}:
        return "transition proposal must name exactly the candidate pair"
    return None


def _pass_contradictions(run: _Run, operator: ContradictionOperator) -> None:
    """Detect contradictions among active records.

    Episodes are excluded alongside rules: an episode is a timestamped
    occurrence, not a competing truth-claim — "the build passed" (Monday)
    and "the build did not pass" (Tuesday) are two events, and forcing them
    transitional would suppress legitimate history (the same reasoning the
    near-dup operator applies when it abstains on episodes).

    A transition names exactly two sides, so a record can join only one
    transition per pass. When a detected pair would involve an
    already-entangled record (A contradicts both B and C), the extra pair is
    NOT silently dropped — it lands in the report as a deferral, to be
    re-detected once the first dispute is resolved.
    """
    entangled: set[str] = set()
    records = [
        r
        for r in _active(run.store)
        if r.type not in (RecordType.RULE, RecordType.EPISODE)
    ]
    for a, b in _candidate_pairs(records):
        if a.type is not b.type:
            continue
        if a.id in entangled or b.id in entangled:
            deferred = operator.propose_contradiction(a, b)
            if deferred is not None:
                run.reject(
                    "contradiction",
                    f"pair {a.id}+{b.id}",
                    "deferred: a side already joined a transition this pass; "
                    "re-run consolidation after that dispute resolves",
                )
            continue
        proposal = operator.propose_contradiction(a, b)
        if proposal is None:
            continue
        problem = _validate_contradiction(proposal, a, b)
        if problem is not None:
            run.reject("contradiction", f"pair {a.id}+{b.id}", problem)
            continue
        try:
            written = apply_proposal(run.store, proposal)
        except AgentMemoryError as exc:
            run.reject("contradiction", f"pair {a.id}+{b.id}", str(exc))
            continue
        run.act("contradiction", "transition", (a, b), written, proposal.reason)
        entangled.update({a.id, b.id})


def _transitional_pairs(
    store: RecordStore,
) -> list[tuple[MemoryRecord, MemoryRecord]]:
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[MemoryRecord, MemoryRecord]] = []
    for record in sorted(
        store.list(state=RecordState.TRANSITIONAL), key=lambda r: r.id
    ):
        assert record.transition is not None  # state invariant on the record
        sides = record.transition.sides
        if sides in seen:
            continue
        seen.add(sides)
        try:
            a, b = store.get(sides[0]), store.get(sides[1])
        except StoreError:
            continue
        if a.state is RecordState.TRANSITIONAL and b.state is RecordState.TRANSITIONAL:
            pairs.append((a, b))
    return pairs


def _pass_resolution(run: _Run, resolver: ResolutionOperator | None) -> None:
    if resolver is None:
        return
    for a, b in _transitional_pairs(run.store):
        verdict = resolver.propose_winner(a, b)
        if verdict is None:
            continue
        winner_id, reason = verdict
        if winner_id not in (a.id, b.id):
            run.reject(
                "resolution",
                f"pair {a.id}+{b.id}",
                "resolver named a winner outside the pair",
            )
            continue
        if not reason:
            run.reject("resolution", f"pair {a.id}+{b.id}", "resolution requires a reason")
            continue
        loser_id = b.id if winner_id == a.id else a.id
        try:
            resolved_winner, resolved_loser = resolve(
                run.store,
                winner_id=winner_id,
                loser_id=loser_id,
                reason=reason,
                now=run.now,
            )
        except AgentMemoryError as exc:
            run.reject("resolution", f"pair {a.id}+{b.id}", str(exc))
            continue
        run.act(
            "resolution",
            "resolve",
            (a, b) if winner_id == a.id else (b, a),
            (resolved_winner, resolved_loser),
            reason,
        )


def _signature(record: MemoryRecord) -> tuple[str, ...]:
    if record.tags:
        return tuple(sorted(record.tags))
    return tuple(sorted(record.slots))


def _pass_promotion(run: _Run) -> None:
    """Repeated episodes → a rule with evidence links to ALL of them, created
    through record invariants (non-empty evidence is enforced at birth)."""
    groups: dict[tuple[str, ...], list[MemoryRecord]] = {}
    for episode in _active(run.store):
        if episode.type is not RecordType.EPISODE:
            continue
        signature = _signature(episode)
        if signature:
            groups.setdefault(signature, []).append(episode)
    existing = {
        _signature(rule)
        for rule in run.store.list(type=RecordType.RULE)
        if rule.state is not RecordState.ARCHIVED
    }
    for signature in sorted(groups):
        episodes = groups[signature]
        if len(episodes) < run.config.promotion_threshold or signature in existing:
            continue
        evidence = tuple(sorted(episode.id for episode in episodes))
        rule = MemoryRecord.create(
            RecordType.RULE,
            f"Recurring pattern across {len(episodes)} episodes "
            f"({', '.join(signature)}); latest: "
            f"{max(episodes, key=lambda e: (e.created_at, e.id)).body}",
            Provenance(kind=_KIND_CONSOLIDATION, ref=f"promotion:{'+'.join(signature)}"),
            confidence=min(episode.confidence for episode in episodes),
            evidence=evidence,
            tags=signature,
            created_at=run.now,
        )
        try:
            apply_proposal(
                run.store, Proposal(kind=ProposalKind.CREATE, record=rule)
            )
        except AgentMemoryError as exc:  # pragma: no cover — birth invariants hold
            run.reject("promotion", f"signature {signature}", str(exc))
            continue
        run.act(
            "promotion",
            "create-rule",
            (),
            (rule,),
            f"{len(episodes)} episodes share signature {signature} "
            f"(threshold {run.config.promotion_threshold})",
        )


def _pass_rule_flags(run: _Run) -> None:
    """The invariant sweep: a rule is flagged exactly when NO evidence record
    is live — missing evidence counts as invalidated, but any live (active or
    transitional) evidence keeps the rule unflagged, per §3.4. A rule with
    one missing and one active evidence record has live support and must not
    be flagged."""
    for rule in sorted(
        run.store.list(state=RecordState.ACTIVE, type=RecordType.RULE),
        key=lambda r: r.id,
    ):
        evidence: dict[str, MemoryRecord] = {}
        missing: list[str] = []
        for eid in rule.evidence:
            try:
                evidence[eid] = run.store.get(eid)
            except StoreError:
                missing.append(eid)
        if not missing:
            refreshed = refresh_rule_flag(rule, evidence, now=run.now)
        else:
            live = {RecordState.ACTIVE, RecordState.TRANSITIONAL}
            any_live = any(record.state in live for record in evidence.values())
            should_flag = not any_live
            if should_flag == rule.flagged:
                continue
            reason = (
                "no live evidence: records missing from store "
                f"({sorted(missing)}) and the rest archived or superseded"
                if should_flag
                else None
            )
            refreshed = rule.evolve(
                {"flagged": should_flag, "flag_reason": reason, "updated_at": run.now}
            )
        if refreshed.flagged != rule.flagged:
            run.store.put(refreshed)
            run.act(
                "rule-flags",
                "flag" if refreshed.flagged else "unflag",
                (rule,),
                (refreshed,),
                refreshed.flag_reason or "live evidence restored",
            )


def consolidate(
    store: RecordStore,
    *,
    operators: ConsolidationOperators | None = None,
    now: datetime | None = None,
    config: ConsolidationConfig | None = None,
) -> ConsolidationReport:
    """One deterministic consolidation pass over the store.

    Sub-pass order: exact dedup → near-dup merge → contradiction detection →
    resolution → promotion → rule-flag sweep. Every operator proposal is
    validated against the lifecycle machine before apply; every mutation and
    every rejection lands in the returned report.
    """
    cfg = config if config is not None else ConsolidationConfig()
    ops = operators if operators is not None else default_operators(cfg)
    recording = _RecordingStore(store)
    run = _Run(
        store=recording,
        now=now if now is not None else datetime.now(UTC),
        config=cfg,
    )
    _pass_exact_dedup(run)
    _pass_near_dup(run, ops.near_dup)
    _pass_contradictions(run, ops.contradiction)
    _pass_resolution(run, ops.resolver)
    _pass_promotion(run)
    _pass_rule_flags(run)
    return ConsolidationReport(
        actions=tuple(run.actions),
        rejections=tuple(run.rejections),
        touched=tuple(sorted(recording.touched)),
    )


class _RecordingStore(RecordStore):
    """Pass-through store that remembers which ids were written, so a
    consolidation report can name everything it changed."""

    def __init__(self, inner: RecordStore) -> None:
        self._inner = inner
        self.touched: set[str] = set()

    def put(self, record: MemoryRecord) -> None:
        self._inner.put(record)
        self.touched.add(record.id)

    def get(self, record_id: str) -> MemoryRecord:
        return self._inner.get(record_id)

    def history(self, record_id: str) -> tuple[MemoryRecord, ...]:
        return self._inner.history(record_id)

    def list(
        self,
        *,
        state: RecordState | None = None,
        type: RecordType | None = None,
        tag: str | None = None,
    ) -> tuple[MemoryRecord, ...]:
        return self._inner.list(state=state, type=type, tag=tag)
