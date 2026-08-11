"""The porcelain API: two calls, zero lifecycle vocabulary required.

:class:`Memory` is the front door of fg-agent-memory. The default experience
is deliberately simple —

    memory = Memory("./memory")
    memory.remember("Sandro's favorite editor is Zed.")
    print(memory.recall("what editor does Sandro use?").as_prompt_block())

— and everything underneath is the full standard: records land in a readable
directory of canonical-JSON files (commit it to git if you like), a SQLite
FTS index rides beside it, consolidation runs automatically every few
remembers, and nothing is ever deleted. Every layer is swappable through
constructor kwargs (store, indexes, extractor, operators, configs, identity)
for callers who want the machinery; none of it is required.

Contradictions are first-class here, not a porcelain casualty: remember X,
then remember not-X, and recall shows *both* sides marked as disputed until
:meth:`Memory.resolve` names a winner — after which the loser is superseded
but forever addressable through ``include_history``. The porcelain never
auto-picks winners; deciding what is true is the caller's call.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fg_agent_id import (
    KeyPair,
    address_from_signing_key,
    canonical_json,
    load_or_create_keys,
)

from .errors import AgentMemoryError, RecordError, StoreError
from .pipeline import (
    ConsolidationConfig,
    ConsolidationOperators,
    ConsolidationReport,
    DecayConfig,
    DecayReport,
    HeuristicContradictionOperator,
    IngestReport,
    SalienceScores,
    TrigramNearDupOperator,
)
from .pipeline import consolidate as run_consolidate
from .pipeline import decay as run_decay
from .pipeline import ingest as run_ingest
from .pipeline import resolve as run_resolve
from .ports import (
    InMemoryRecordStore,
    InMemorySearchIndex,
    Operator,
    Proposal,
    ProposalKind,
    RecordStore,
    SearchIndex,
)
from .records import MemoryFile, MemoryRecord, Provenance, RecordState, RecordType
from .retrieval import RetrievedBlock, RetrievedRecord, TouchCounts, retrieve
from .stores import FileRecordStore, SQLiteSearchIndex
from .text import infer_slots

# Subdirectory / sidecar names inside a Memory directory.
_RECORDS_DIR = "records"
_INDEX_FILE = "index.sqlite"
_SALIENCE_FILE = "salience.json"

# How many remembers between automatic consolidation passes, by default.
# Small enough that a casual user sees contradictions surfaced within one
# short session; large enough that single remembers stay O(1).
DEFAULT_AUTO_CONSOLIDATE_EVERY = 4

DEFAULT_RECALL_BUDGET_TOKENS = 2000


class VerbatimExtractor(Operator):
    """Default write-stage operator for the porcelain: the caller said
    "remember this", so this is remembered — one fact record carrying the
    text verbatim. No sentence heuristics, no silent drops. Swap in
    :class:`~fg_agent_memory.HeuristicExtractor` (or an LLM operator) via
    ``Memory(extractor=...)`` when extraction should be smarter."""

    def __init__(self, *, tags: tuple[str, ...] = (), confidence: float = 0.9) -> None:
        self._tags = tags
        self._confidence = confidence

    def propose(self, text: str, provenance: Provenance) -> tuple[Proposal, ...]:
        body = text.strip()
        record = MemoryRecord.create(
            RecordType.FACT,
            body,
            provenance,
            # A coarse subject→value slot lets the contradiction pass catch
            # value swaps that carry no negation marker ("the DB is Postgres"
            # vs "the DB is MySQL").
            slots=infer_slots(body),
            confidence=self._confidence,
            tags=self._tags,
        )
        return (Proposal(kind=ProposalKind.CREATE, record=record),)


class _SigningExtractor(Operator):
    """Wraps any extractor so every created record leaves signed by the
    Memory's identity. Signatures never change record ids (the id covers
    birth content only), so dedup semantics are untouched."""

    def __init__(self, inner: Operator, keys: KeyPair, address: str) -> None:
        self._inner = inner
        self._keys = keys
        self._address = address

    def propose(self, text: str, provenance: Provenance) -> tuple[Proposal, ...]:
        out = []
        for proposal in self._inner.propose(text, provenance):
            if proposal.kind is ProposalKind.CREATE and proposal.record is not None:
                signed = proposal.record.sign(self._keys, self._address)
                proposal = Proposal(kind=ProposalKind.CREATE, record=signed)
            out.append(proposal)
        return tuple(out)


class _RecallTouches(TouchCounts):
    """Retrieval-reinforcement bridge: every recall hit also lands in the
    decay sidecar, so records that keep proving useful resist archival."""

    def __init__(self, scores: SalienceScores) -> None:
        super().__init__()
        self._scores = scores

    def touch(self, record_id: str) -> None:
        super().touch(record_id)
        self._scores.touch(record_id)


@dataclass(frozen=True, kw_only=True)
class RememberResult:
    """What one :meth:`Memory.remember` call did."""

    record_id: str
    deduped: bool  # True when the content was already known (reinforced, not re-written)
    ingest: IngestReport
    consolidation: ConsolidationReport | None = None  # present when auto-consolidation ran


@dataclass(frozen=True, kw_only=True)
class RecallResult:
    """A :meth:`Memory.recall` answer: the underlying retrieval block plus a
    clean, model-ready string rendering."""

    block: RetrievedBlock

    @property
    def records(self) -> tuple[RetrievedRecord, ...]:
        return self.block.records

    @property
    def ids(self) -> tuple[str, ...]:
        return self.block.ids

    def __len__(self) -> int:
        return len(self.block.records)

    def __iter__(self):
        return iter(self.block.records)

    def as_prompt_block(self) -> str:
        """The recalled memories as a compact block ready to paste into a
        prompt. Disputed and superseded records are labeled so the model
        never mistakes one side of a contradiction for settled truth."""
        if not self.block.records:
            return "(no relevant memories)"
        lines = [f"Relevant memories (query: {self.block.query}):"]
        for item in self.block.records:
            record = item.record
            marker = ""
            if record.state is RecordState.TRANSITIONAL:
                assert record.transition is not None
                marker = f" [disputed: {record.transition.reason}]"
            elif record.state is RecordState.SUPERSEDED:
                marker = " [superseded — kept for history]"
            elif record.state is RecordState.ARCHIVED:
                marker = " [archived]"
            flag = f" [flagged: {record.flag_reason}]" if record.flagged else ""
            lines.append(f"- {record.body}{marker}{flag}")
        return "\n".join(lines)


class Memory:
    """One agent's memory, as one object. See the module docstring for the
    two-call experience; every underlying layer is an optional kwarg.

    :param path: Directory the memory lives in. Created if missing; records
        are readable canonical-JSON files under ``<path>/records/`` with a
        SQLite FTS sidecar at ``<path>/index.sqlite``.
    :param store: Bring your own :class:`RecordStore` (overrides ``path``
        for record storage).
    :param indexes: Bring your own :class:`SearchIndex` sequence.
    :param extractor: Write-stage operator; defaults to
        :class:`VerbatimExtractor`.
    :param operators: Consolidation operator set; the porcelain default has
        **no auto-resolver** — contradictions stay visible until
        :meth:`resolve`.
    :param identity: fg-agent-id keypair; when present, every remembered
        record and every export is signed. A ``str`` or :class:`Path` is
        treated as a keyfile path and resolved with fg-agent-id's
        ``load_or_create_keys`` — the file is created on first run, so
        ``Memory("./memory", identity="agent.key")`` is a complete
        persistent-identity setup.
    :param auto_consolidate: Run a consolidation pass automatically every
        ``auto_consolidate_every`` remembers (default on).
    """

    def __init__(
        self,
        path: str | Path | None = "./memory",
        *,
        store: RecordStore | None = None,
        indexes: tuple[SearchIndex, ...] | list[SearchIndex] | None = None,
        extractor: Operator | None = None,
        operators: ConsolidationOperators | None = None,
        consolidation_config: ConsolidationConfig | None = None,
        decay_config: DecayConfig | None = None,
        identity: KeyPair | str | Path | None = None,
        source_kind: str = "conversation",
        auto_consolidate: bool = True,
        auto_consolidate_every: int = DEFAULT_AUTO_CONSOLIDATE_EVERY,
    ) -> None:
        if auto_consolidate_every < 1:
            raise ValueError("auto_consolidate_every must be at least 1")
        if store is None and path is None:
            store = InMemoryRecordStore()

        self._path = Path(path) if path is not None else None
        # The salience sidecar lives beside the records only when this
        # Memory owns the directory (default path-based store) — a custom
        # store brings its own persistence story.
        self._owns_path = store is None and self._path is not None
        if store is not None:
            self._store = store
        else:
            assert self._path is not None
            self._path.mkdir(parents=True, exist_ok=True)
            self._store = FileRecordStore(self._path / _RECORDS_DIR)

        if indexes is not None:
            self._indexes: tuple[SearchIndex, ...] = tuple(indexes)
        elif self._path is not None and store is None:
            self._indexes = (SQLiteSearchIndex(self._path / _INDEX_FILE),)
        else:
            self._indexes = (InMemorySearchIndex(),)

        if isinstance(identity, (str, Path)):
            identity = load_or_create_keys(identity)
        self._identity = identity
        self._address = (
            address_from_signing_key(identity.public.signing) if identity else None
        )
        self._base_extractor = extractor if extractor is not None else VerbatimExtractor()

        self._consolidation_config = (
            consolidation_config if consolidation_config is not None else ConsolidationConfig()
        )
        # Porcelain default: detect contradictions, never auto-pick winners.
        self._operators = operators if operators is not None else ConsolidationOperators(
            near_dup=TrigramNearDupOperator(
                threshold=self._consolidation_config.near_dup_threshold
            ),
            contradiction=HeuristicContradictionOperator(),
            resolver=None,
        )
        self._decay_config = decay_config if decay_config is not None else DecayConfig()
        self._source_kind = source_kind
        self._auto_consolidate = auto_consolidate
        self._auto_every = auto_consolidate_every
        self._remembers_since_consolidate = 0

        self._scores = self._load_salience()
        self._touches = _RecallTouches(self._scores)
        # One lock around every public operation: a Memory instance is safe
        # to share across threads (e.g. an MCP server's worker pool), and a
        # consolidation pass never interleaves with a remember.
        self._op_lock = threading.RLock()
        # body -> active record id, so porcelain dedup is a dict lookup
        # instead of a full-store scan per remember.
        self._active_bodies: dict[str, str] = {}

        # Make everything already in the store findable (covers reopening an
        # existing directory, or a directory copied without its index sidecar).
        self._reindex_all()

    # ------------------------------------------------------- salience sidecar

    def _salience_path(self) -> Path | None:
        return self._path / _SALIENCE_FILE if self._owns_path else None

    def _load_salience(self) -> SalienceScores:
        path = self._salience_path()
        if path is None or not path.exists():
            return SalienceScores()
        try:
            data = json.loads(path.read_bytes())
            return SalienceScores.from_dict(data.get("scores", {}))
        except (ValueError, OSError, AgentMemoryError):
            # A corrupt sidecar degrades gracefully to age-based decay; it
            # is engine-local state, never part of the record standard.
            return SalienceScores()

    def _save_salience(self) -> None:
        path = self._salience_path()
        if path is None:
            return
        payload = json.dumps({"scores": self._scores.to_dict()}, sort_keys=True)
        # Unique temp name: two Memory instances on one directory must not
        # interleave writes through a shared temp file.
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(payload)
        tmp.replace(path)

    # ------------------------------------------------------------------ core

    def remember(
        self,
        text: str,
        *,
        tags: tuple[str, ...] | list[str] | None = None,
        source: str | None = None,
    ) -> RememberResult:
        """Store something. Dedup is automatic: remembering the same content
        twice reinforces the existing record instead of duplicating it."""
        if not text or not text.strip():
            raise ValueError("cannot remember empty text")
        with self._op_lock:
            return self._remember_locked(text, tags=tags, source=source)

    def _remember_locked(
        self,
        text: str,
        *,
        tags: tuple[str, ...] | list[str] | None,
        source: str | None,
    ) -> RememberResult:
        # Porcelain-level dedup: the same live content is one memory, however
        # many times it is said. (Content addresses include birth time, so
        # the write stage's id-dedup alone only catches same-instant repeats;
        # the consolidation pass would collapse the copy later, but a
        # remember() caller should never create one in the first place.)
        body = text.strip()
        existing = self._active_body(body)
        if existing is not None:
            self._scores.touch(existing.id)
            # A repeat remember still contributes its tags: merge them onto
            # the existing record (tags are lifecycle data, not birth
            # content, so the id is stable) instead of dropping them.
            merged_tags = tuple(dict.fromkeys(existing.tags + tuple(tags or ())))
            if merged_tags != existing.tags:
                updated = existing.evolve(
                    {"tags": merged_tags, "updated_at": datetime.now(UTC)}
                )
                self._store.put(updated)
                self._note_record(updated)
            self._save_salience()
            return RememberResult(
                record_id=existing.id,
                deduped=True,
                ingest=IngestReport(accepted=(), deduped=(existing.id,), rejected=()),
            )
        provenance = Provenance(kind=self._source_kind, ref=source, agent=self._address)
        # Composition order matters: tags first, signature last — the
        # signature must cover the tags it ships with.
        extractor = self._base_extractor
        if tags:
            extractor = _TaggingExtractor(extractor, tuple(tags))
        if self._identity is not None and self._address is not None:
            extractor = _SigningExtractor(extractor, self._identity, self._address)
        report = run_ingest(
            text,
            extractor=extractor,
            store=self._store,
            provenance=provenance,
            scores=self._scores,
        )
        for record_id in report.accepted:
            self._note_record(self._store.get(record_id))

        record_id = (report.accepted + report.deduped + ("",))[0]
        consolidation: ConsolidationReport | None = None
        if report.accepted:
            self._remembers_since_consolidate += 1
            if self._auto_consolidate and self._remembers_since_consolidate >= self._auto_every:
                consolidation = self.consolidate()
        self._save_salience()
        return RememberResult(
            record_id=record_id,
            deduped=bool(report.deduped) and not report.accepted,
            ingest=report,
            consolidation=consolidation,
        )

    def recall(
        self,
        query: str,
        *,
        budget_tokens: int = DEFAULT_RECALL_BUDGET_TOKENS,
        include_history: bool = False,
        include_archived: bool = False,
    ) -> RecallResult:
        """What the memory knows about ``query``, ranked and budgeted.

        Current, undisputed facts come first. Both sides of any unresolved
        contradiction are always surfaced together, labeled as disputed.
        Superseded (resolved-away) records appear only with
        ``include_history=True``; archived ones only with
        ``include_archived=True``."""
        with self._op_lock:
            block = retrieve(
                query,
                store=self._store,
                indexes=self._indexes,
                budget_tokens=budget_tokens,
                include_history=include_history,
                include_archived=include_archived,
                touches=self._touches,
            )
            if block.records:
                self._save_salience()
        return RecallResult(block=block)

    # ------------------------------------------------------- background passes

    def consolidate(self) -> ConsolidationReport:
        """One background pass: dedup, contradiction detection, promotion,
        rule-flag sweep. Runs automatically every few remembers (see
        ``auto_consolidate``); calling it directly is always safe."""
        with self._op_lock:
            report = run_consolidate(
                self._store, operators=self._operators, config=self._consolidation_config
            )
            self._remembers_since_consolidate = 0
            # Refresh only what the pass wrote — merged/rule records get
            # indexed, retired records leave the dedup map.
            for record_id in report.touched:
                self._note_record(self._store.get(record_id))
            return report

    def decay(self) -> DecayReport:
        """One mechanical decay pass: stale, unreinforced records are archived
        (never deleted); disputed records, live rules, and rule evidence are
        protected. Protections are reported, not silent."""
        with self._op_lock:
            report = run_decay(
                self._store, scores=self._scores, config=self._decay_config
            )
            for action in report.archived:
                for record_id in action.record_ids:
                    self._note_record(self._store.get(record_id))
            self._save_salience()
            return report

    def resolve(self, record_id: str, winner_id: str, reason: str) -> None:
        """Settle a dispute between two records by naming the winner.

        ``record_id`` and ``winner_id`` must be the two sides of the same
        unresolved contradiction. The loser is superseded — out of default
        recall, forever addressable via ``include_history``."""
        if not reason:
            raise ValueError("resolution requires a reason")
        if record_id == winner_id:
            raise RecordError("winner and loser must be two distinct records")
        with self._op_lock:
            self._resolve_locked(record_id, winner_id, reason)

    def _resolve_locked(self, record_id: str, winner_id: str, reason: str) -> None:
        winner = self._store.get(winner_id)
        if winner.transition is None:
            raise RecordError(f"record {winner_id} is not in an unresolved contradiction")
        if set(winner.transition.sides) != {record_id, winner_id}:
            raise RecordError(
                f"records {record_id} and {winner_id} are not sides of the same contradiction"
            )
        resolved_winner, resolved_loser = run_resolve(
            self._store, winner_id=winner_id, loser_id=record_id, reason=reason
        )
        self._note_record(resolved_winner)
        self._note_record(resolved_loser)

    def redact(self, record_id: str, reason: str) -> MemoryRecord:
        """Destroy a record's content under an explicit, audited action —
        the escape hatch for secrets or personal data remembered by mistake.

        Every stored version is replaced by one tombstone: the id, type,
        timestamps, and link graph survive (so nothing dangles and the
        redaction itself is visible forever), but body, slots, tags, and
        provenance detail are gone. This is NOT decay: archival keeps
        content, redaction destroys it, and only this call may do so."""
        if not reason:
            raise ValueError("redaction requires a reason")
        with self._op_lock:
            redactor = getattr(self._store, "redact", None)
            if redactor is None:
                raise StoreError(
                    "the configured record store does not support redaction"
                )
            tomb = redactor(record_id, reason)
            self._note_record(tomb)
            return tomb

    # ------------------------------------------------------------ portability

    def export(self, path: str | Path) -> MemoryFile:
        """Write the memory to one portable file — the latest version of
        every record, all lifecycle states included, signed when this Memory
        has an identity — and return the in-memory artifact."""
        records = self._store.list()
        memory_file = MemoryFile.create(records, agent=self._address)
        if self._identity is not None:
            memory_file = memory_file.sign(self._identity)
        Path(path).write_bytes(canonical_json(memory_file.model_dump()))
        return memory_file

    @classmethod
    def load(
        cls,
        file_path: str | Path,
        *,
        path: str | Path | None = None,
        verify: bool = True,
        **kwargs,
    ) -> Memory:
        """Open a portable memory file as a live :class:`Memory`.

        With ``verify=True`` (the default) the file-level signature is
        REQUIRED and checked before anything is loaded, and every record
        naming a signer is verified too — a missing or stripped signature is
        an error, never a silent skip, so attacker-controlled file content
        can never downgrade verification. Pass ``verify=False`` only for
        unsigned files from trusted local sources. ``path`` gives the loaded
        memory a directory of its own; without it the records live in memory
        only. Other kwargs pass through to :class:`Memory`."""
        data = json.loads(Path(file_path).read_bytes())
        memory_file = MemoryFile.model_validate(data)
        if verify:
            memory_file.verify()  # raises on an unsigned or tampered file
            for record in memory_file.records:
                if record.signed_by or record.signature:
                    record.verify()
        memory = cls(path, **kwargs) if path is not None else cls(None, **kwargs)
        for record in memory_file.records:
            try:
                memory._store.get(record.id)
            except StoreError:
                memory._store.put(record)
                memory._note_record(record)
        return memory

    # -------------------------------------------------------------- inspection

    def status(self) -> dict[str, object]:
        """A compact snapshot: record counts by state and type, plus how many
        unresolved contradictions are waiting on :meth:`resolve`."""
        records = self._store.list()
        by_state: dict[str, int] = {}
        by_type: dict[str, int] = {}
        disputes: set[tuple[str, str]] = set()
        for record in records:
            by_state[record.state.value] = by_state.get(record.state.value, 0) + 1
            by_type[record.type.value] = by_type.get(record.type.value, 0) + 1
            if record.transition is not None:
                disputes.add(record.transition.sides)
        return {
            "records": len(records),
            "by_state": by_state,
            "by_type": by_type,
            "open_contradictions": len(disputes),
            "path": str(self._path) if self._path is not None else None,
            "agent": self._address,
        }

    @property
    def store(self) -> RecordStore:
        """The underlying record store, for callers who want the machinery."""
        return self._store

    # ---------------------------------------------------------------- internals

    def _index(self, record: MemoryRecord) -> None:
        for index in self._indexes:
            index.index(record)

    def _note_record(self, record: MemoryRecord) -> None:
        """Keep the search indexes and the active-body dedup map in step
        with one written record version."""
        self._index(record)
        if record.state is RecordState.ACTIVE:
            self._active_bodies[record.body] = record.id
        elif self._active_bodies.get(record.body) == record.id:
            del self._active_bodies[record.body]

    def _active_body(self, body: str) -> MemoryRecord | None:
        """The live record for an exact body, or None. Verifies against the
        store so a stale map entry can never resurrect a retired record."""
        record_id = self._active_bodies.get(body)
        if record_id is None:
            return None
        try:
            record = self._store.get(record_id)
        except StoreError:
            record = None
        if record is None or record.state is not RecordState.ACTIVE or record.body != body:
            del self._active_bodies[body]
            return None
        return record

    def _reindex_all(self) -> None:
        for record in self._store.list():
            self._note_record(record)


class _TaggingExtractor(Operator):
    """Adds per-call tags on top of whatever the base extractor proposes."""

    def __init__(self, inner: Operator, tags: tuple[str, ...]) -> None:
        self._inner = inner
        self._tags = tags

    def propose(self, text: str, provenance: Provenance) -> tuple[Proposal, ...]:
        out = []
        for proposal in self._inner.propose(text, provenance):
            if proposal.kind is ProposalKind.CREATE and proposal.record is not None:
                record = proposal.record
                merged_tags = tuple(dict.fromkeys(record.tags + self._tags))
                if merged_tags != record.tags:
                    record = MemoryRecord.create(
                        record.type,
                        record.body,
                        record.provenance,
                        slots=dict(record.slots),
                        confidence=record.confidence,
                        evidence=record.evidence,
                        tags=merged_tags,
                        created_at=record.created_at,
                        extra=dict(record.extra),
                    )
                proposal = Proposal(kind=ProposalKind.CREATE, record=record)
            out.append(proposal)
        return tuple(out)
