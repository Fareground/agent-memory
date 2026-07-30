"""fg-agent-memory — portable, signed agent memory.

Typed memory records with an explicit, code-enforced fact lifecycle
(active | superseded | transitional | archived), evidence-linked rules,
storage-agnostic pipeline ports, and a signed portable memory file —
the memory standard in the Fareground agent-standards family, signed
with fg-agent-id identities.
"""

from __future__ import annotations

from .errors import (
    AgentMemoryError,
    LifecycleError,
    ProposalError,
    RecordError,
    StoreError,
)
from .lifecycle import (
    LEGAL_TRANSITIONS,
    archive,
    assert_transition,
    begin_transition,
    is_legal,
    refresh_rule_flag,
    resolve_transition,
    supersede,
)
from .memory import (
    Memory,
    RecallResult,
    RememberResult,
    VerbatimExtractor,
)
from .pipeline import (
    ConsolidationConfig,
    ConsolidationOperators,
    ConsolidationReport,
    DecayConfig,
    DecayReport,
    IngestReport,
    SalienceScores,
    consolidate,
    decay,
    default_operators,
    ingest,
    resolve,
)
from .ports import (
    Embedder,
    HashEmbedder,
    HeuristicExtractor,
    InMemoryRecordStore,
    InMemorySearchIndex,
    Operator,
    Proposal,
    ProposalKind,
    RecordStore,
    SearchIndex,
    apply_proposal,
    apply_proposals,
)
from .records import (
    MemoryFile,
    MemoryRecord,
    Provenance,
    RecordState,
    RecordType,
    Transition,
    confidence_from_wire,
    confidence_to_wire,
)
from .retrieval import RetrievedBlock, RetrievedRecord, TouchCounts, retrieve
from .signing import (
    CONTEXT_MEMORY_FILE,
    CONTEXT_RECORD,
    DOMAIN,
    domain_tag,
    sign_payload,
    signing_input,
    verify_by_address,
    verify_payload,
)
from .stores import FileRecordStore, SQLiteRecordStore, SQLiteSearchIndex
from .version import FORMAT_VERSION

__all__ = [
    "CONTEXT_MEMORY_FILE",
    "CONTEXT_RECORD",
    "DOMAIN",
    "FORMAT_VERSION",
    "LEGAL_TRANSITIONS",
    "AgentMemoryError",
    "ConsolidationConfig",
    "ConsolidationOperators",
    "ConsolidationReport",
    "DecayConfig",
    "DecayReport",
    "Embedder",
    "FileRecordStore",
    "IngestReport",
    "Memory",
    "RecallResult",
    "RememberResult",
    "RetrievedBlock",
    "RetrievedRecord",
    "SQLiteRecordStore",
    "SQLiteSearchIndex",
    "SalienceScores",
    "TouchCounts",
    "VerbatimExtractor",
    "consolidate",
    "decay",
    "default_operators",
    "ingest",
    "resolve",
    "retrieve",
    "HashEmbedder",
    "HeuristicExtractor",
    "InMemoryRecordStore",
    "InMemorySearchIndex",
    "LifecycleError",
    "MemoryFile",
    "MemoryRecord",
    "Operator",
    "Proposal",
    "ProposalError",
    "ProposalKind",
    "Provenance",
    "RecordError",
    "RecordState",
    "RecordStore",
    "RecordType",
    "SearchIndex",
    "StoreError",
    "Transition",
    "apply_proposal",
    "apply_proposals",
    "archive",
    "assert_transition",
    "begin_transition",
    "confidence_from_wire",
    "confidence_to_wire",
    "domain_tag",
    "is_legal",
    "refresh_rule_flag",
    "resolve_transition",
    "sign_payload",
    "signing_input",
    "supersede",
    "verify_by_address",
    "verify_payload",
]
