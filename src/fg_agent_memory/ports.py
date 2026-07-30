"""Adapter ports: storage, search, and the LLM touchpoint.

The standard is the record format and the lifecycle semantics — never the
backend. Storage and indexing are ports; users bring SQLite, Postgres, a
vector DB, or a directory of files, and the framework applies unchanged.
The in-memory implementations here are the reference proof that no backend
is load-bearing, and double as the fixtures for the port contract tests.

The ``Operator`` port is where an LLM (or any heuristic) plugs in: it takes
typed input and returns typed *proposals*, which the pipeline validates
against the lifecycle machine before applying. The LLM is an operator inside
the pipeline, never a free rewriter of the store.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum

from .errors import ProposalError, StoreError
from .lifecycle import archive, begin_transition, supersede
from .records import MemoryRecord, Provenance, RecordState, RecordType
from .text import infer_slots


class RecordStore(ABC):
    """Append-only record storage.

    ``put`` never overwrites: each put of an existing id appends a new
    version, and every prior version stays addressable via ``history``.
    ``get`` returns the latest version. Deleting is not part of the contract —
    decay is a lifecycle state, not a removal.
    """

    @abstractmethod
    def put(self, record: MemoryRecord) -> None:
        """Append ``record`` as the newest version of its id."""

    @abstractmethod
    def get(self, record_id: str) -> MemoryRecord:
        """The latest version of ``record_id``; raises StoreError if unknown."""

    @abstractmethod
    def history(self, record_id: str) -> tuple[MemoryRecord, ...]:
        """Every stored version of ``record_id``, oldest first."""

    @abstractmethod
    def list(
        self,
        *,
        state: RecordState | None = None,
        type: RecordType | None = None,
        tag: str | None = None,
    ) -> tuple[MemoryRecord, ...]:
        """Latest versions matching every given filter, insertion-ordered."""

    def __iter__(self) -> Iterator[MemoryRecord]:
        return iter(self.list())


class InMemoryRecordStore(RecordStore):
    """Reference store: a dict of version lists. The contract, minus any
    persistence — proof that the standard has no hidden backend dependency."""

    def __init__(self) -> None:
        self._versions: dict[str, list[MemoryRecord]] = {}

    def put(self, record: MemoryRecord) -> None:
        self._versions.setdefault(record.id, []).append(record)

    def get(self, record_id: str) -> MemoryRecord:
        versions = self._versions.get(record_id)
        if not versions:
            raise StoreError(f"unknown record id: {record_id}")
        return versions[-1]

    def history(self, record_id: str) -> tuple[MemoryRecord, ...]:
        versions = self._versions.get(record_id)
        if not versions:
            raise StoreError(f"unknown record id: {record_id}")
        return tuple(versions)

    def list(
        self,
        *,
        state: RecordState | None = None,
        type: RecordType | None = None,
        tag: str | None = None,
    ) -> tuple[MemoryRecord, ...]:
        out = []
        for versions in self._versions.values():
            record = versions[-1]
            if state is not None and record.state is not RecordState(state):
                continue
            if type is not None and record.type is not RecordType(type):
                continue
            if tag is not None and tag not in record.tags:
                continue
            out.append(record)
        return tuple(out)

    def redact(self, record_id: str, reason: str) -> MemoryRecord:
        """§11 governed redaction: every stored version of ``record_id`` is
        replaced by one tombstone — content destroyed, existence and version
        count preserved. Deliberately NOT part of the RecordStore port:
        redaction is an explicit maintenance action, never pipeline reach."""
        tomb = MemoryRecord.tombstone(self.get(record_id), reason=reason)
        self._versions[record_id] = [tomb] * len(self._versions[record_id])
        return tomb


class Embedder(ABC):
    """Text → fixed-dimension vector. The embedding function is itself a
    port: bring a model if you have one; the hash fallback below means
    nothing in the framework *requires* one."""

    @property
    @abstractmethod
    def dimensions(self) -> int: ...

    @abstractmethod
    def embed(self, text: str) -> tuple[float, ...]: ...


class HashEmbedder(Embedder):
    """Deterministic, model-free embedding: tokens hashed into a fixed-size
    bag-of-words vector, L2-normalized. Not semantically smart — exact and
    overlapping tokens score high — but fully reproducible, dependency-free,
    and good enough to run and test the whole pipeline without any model."""

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self._dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return tuple(vector)
        return tuple(v / norm for v in vector)


class SearchIndex(ABC):
    """Candidate ranking over records. Retrieval policy (state-awareness,
    recency, token budgets) lives above this port; the index only answers
    "which record ids look most like this query, and how much"."""

    @abstractmethod
    def index(self, record: MemoryRecord) -> None: ...

    @abstractmethod
    def candidates(self, query: str, k: int) -> tuple[tuple[str, float], ...]:
        """Top-``k`` (record_id, score) pairs, best first."""


class InMemorySearchIndex(SearchIndex):
    """Reference index: cosine similarity over an Embedder port (the hash
    fallback by default). Re-indexing a record id replaces its vector."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder = embedder if embedder is not None else HashEmbedder()
        self._vectors: dict[str, tuple[float, ...]] = {}

    def index(self, record: MemoryRecord) -> None:
        self._vectors[record.id] = self._embedder.embed(record.body)

    def candidates(self, query: str, k: int) -> tuple[tuple[str, float], ...]:
        if k <= 0:
            return ()
        query_vector = self._embedder.embed(query)
        scored = [
            (record_id, sum(a * b for a, b in zip(query_vector, vector, strict=True)))
            for record_id, vector in self._vectors.items()
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return tuple(scored[:k])


class ProposalKind(StrEnum):
    """What an operator may ask the pipeline to do. Nothing else exists —
    there is no free-rewrite proposal by design."""

    CREATE = "create"
    SUPERSEDE = "supersede"
    TRANSITION = "transition"
    ARCHIVE = "archive"


@dataclass(frozen=True, kw_only=True)
class Proposal:
    """A typed, validatable unit of operator output.

    - ``create``: ``record`` is a new record to admit.
    - ``supersede``: ``record`` replaces the stored record it names in its
      ``supersedes`` field; ``reason`` required.
    - ``transition``: ``target_id``/``other_id`` name two stored records that
      contradict; ``reason`` required.
    - ``archive``: ``target_id`` names the stored record to decay; ``reason``
      required.
    """

    kind: ProposalKind
    record: MemoryRecord | None = None
    target_id: str | None = None
    other_id: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ProposalKind(self.kind))


class Operator(ABC):
    """The LLM touchpoint. An operator sees typed input and emits proposals;
    it never touches the store. The pipeline validates every proposal against
    the lifecycle machine before applying — an invalid proposal is rejected,
    not repaired."""

    @abstractmethod
    def propose(self, text: str, provenance: Provenance) -> tuple[Proposal, ...]: ...


def apply_proposal(store: RecordStore, proposal: Proposal) -> tuple[MemoryRecord, ...]:
    """Validate a proposal against the lifecycle machine and apply it.

    Returns the record versions written. Raises ProposalError (bad shape or
    unknown targets) or LifecycleError (illegal transition) without writing.
    """
    if proposal.kind is ProposalKind.CREATE:
        if proposal.record is None:
            raise ProposalError("create proposal requires a record")
        store.put(proposal.record)
        return (proposal.record,)

    if proposal.kind is ProposalKind.SUPERSEDE:
        if proposal.record is None or not proposal.record.supersedes:
            raise ProposalError("supersede proposal requires a successor naming supersedes")
        try:
            old = store.get(proposal.record.supersedes)
        except StoreError as exc:
            raise ProposalError(f"supersede target not in store: {exc}") from exc
        retired = supersede(old, proposal.record, reason=proposal.reason)
        store.put(proposal.record)
        store.put(retired)
        return (proposal.record, retired)

    if proposal.kind is ProposalKind.TRANSITION:
        if not proposal.target_id or not proposal.other_id:
            raise ProposalError("transition proposal requires target_id and other_id")
        try:
            a = store.get(proposal.target_id)
            b = store.get(proposal.other_id)
        except StoreError as exc:
            raise ProposalError(f"transition target not in store: {exc}") from exc
        a_marked, b_marked = begin_transition(a, b, reason=proposal.reason)
        store.put(a_marked)
        store.put(b_marked)
        return (a_marked, b_marked)

    if proposal.kind is ProposalKind.ARCHIVE:
        if not proposal.target_id:
            raise ProposalError("archive proposal requires target_id")
        try:
            record = store.get(proposal.target_id)
        except StoreError as exc:
            raise ProposalError(f"archive target not in store: {exc}") from exc
        archived = archive(record, reason=proposal.reason)
        store.put(archived)
        return (archived,)

    raise ProposalError(f"unknown proposal kind: {proposal.kind}")


def apply_proposals(
    store: RecordStore, proposals: Iterable[Proposal]
) -> tuple[MemoryRecord, ...]:
    """Apply proposals in order; every write that happens is validated."""
    written: list[MemoryRecord] = []
    for proposal in proposals:
        written.extend(apply_proposal(store, proposal))
    return tuple(written)


# Sentence shapes the heuristic extractor recognizes. Deliberately dumb:
# the point is a working default operator with zero model dependency, not
# extraction quality.
_FACT_PATTERN = re.compile(
    r"\b(is|are|was|were|has|have|prefers?|likes?|uses?|lives?|means?)\b", re.IGNORECASE
)
_EPISODE_PATTERN = re.compile(
    r"\b(did|ran|tried|failed|succeeded|happened|deployed|fixed|broke|met|went)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class HeuristicExtractor(Operator):
    """Default write-stage operator: regex/rule-based, no LLM.

    Splits input into sentences and proposes fact/episode records for the
    shapes it recognizes. It never proposes rules — rules exist only through
    evidence-linked promotion, which no extractor is entitled to shortcut.
    """

    def __init__(self, *, confidence: float = 0.5, tags: tuple[str, ...] = ()) -> None:
        self._confidence = confidence
        self._tags = tags

    def propose(self, text: str, provenance: Provenance) -> tuple[Proposal, ...]:
        proposals: list[Proposal] = []
        for sentence in _SENTENCE_SPLIT.split(text.strip()):
            sentence = sentence.strip()
            if len(sentence) < 8:
                continue
            if _EPISODE_PATTERN.search(sentence):
                record_type = RecordType.EPISODE
            elif _FACT_PATTERN.search(sentence):
                record_type = RecordType.FACT
            else:
                continue
            record = MemoryRecord.create(
                record_type,
                sentence,
                provenance,
                # Coarse subject→value slots make no-negator value swaps
                # detectable by the default contradiction pass.
                slots=infer_slots(sentence) if record_type is RecordType.FACT else {},
                confidence=self._confidence,
                tags=self._tags,
            )
            proposals.append(Proposal(kind=ProposalKind.CREATE, record=record))
        return tuple(proposals)
