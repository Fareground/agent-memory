# Changelog

All notable changes to `fg-agent-memory`. Format loosely follows Keep a
Changelog. The wire format version is tracked separately from the package
version and remains `0.1` until the v1.0 freeze.

## [0.2.0] — 2026-08-11

### Release preparation (2026-08-10)

- `Memory(identity=...)` now also accepts a keyfile path (`str`/`Path`),
  resolved with fg-agent-id's `load_or_create_keys` — created on first
  run, reloaded after — so `Memory("./memory", identity="agent.key")` is
  a complete persistent signed-memory setup. A `KeyPair` still works
  unchanged.
- First PyPI-ready release: `.github/workflows/release.yml` (tag-driven
  build, wheel smoke test, trusted publishing) and `SECURITY.md`.
- `fg-agent-id` dependency pinned to `>=0.2,<0.3` — a few internal
  fg_agent_id submodule helpers are used (marked at each import site), so
  new minors need a deliberate bump.
- `TrigramNearDupOperator`, `HeuristicContradictionOperator`, and
  `HeuristicResolutionOperator` exported at the package root (they were
  documented but only importable from `fg_agent_memory.pipeline`).
- `examples/`: `quickstart.py` (the README hello world) and
  `llm_operator.py` (a provider-agnostic LLM `ContradictionOperator` with
  the full proposal plumbing real and the model call stubbed); both wired
  into the test suite.

### Event time + redaction (2026-07-23)

- **`occurred_at`** — optional, signed event time on every record (episodes
  SHOULD carry it), distinct from `created_at` ingest time. Not part of the
  content address; omitted from the wire when unset so existing signed
  records re-serialize byte-identically. The heuristic resolver's "newer
  wins" compares event time when both sides carry it. (SPEC §2.)
- **§11 Redaction** — governed content destruction: `redact(record_id,
  reason)` replaces every stored version with a permanent tombstone (id,
  type, timestamps, link graph preserved; body/slots/tags/provenance detail
  destroyed; full birth digest recorded). Available on all three stores,
  the `Memory` porcelain, and as an MCP tool; deliberately not on the
  RecordStore port and unreachable by operators. Redacted evidence flags
  dependent rules via the §9.3 sweep.

### Audit hardening (2026-07-23)

Full-diagnostics audit (architecture, correctness, security) closed in one
pass; every item spec-reflected where it changes normative behavior.

- **SECURITY — load verification downgrade closed.** `Memory.load` with
  `verify=True` (the default) now *requires* a file signature — stripping
  the signature no longer silently skips verification — and every record
  naming a signer is individually verified. Unsigned loads take an explicit
  `verify=False`. (SPEC §4.)
- **Signing × lifecycle.** Lifecycle transitions (`supersede`, transition,
  resolve, archive, flag) clear `signed_by`/`signature` on the evolved
  version instead of carrying a stale signature that no longer verifies;
  the signed birth version stays verifiable in history. New
  `MemoryRecord.evolve`. (SPEC §2.3.)
- **Concurrent-writer safety.** FileRecordStore claims version filenames
  with fsync + `os.link` (a taken number retries, never clobbers);
  SQLiteRecordStore serializes puts with `BEGIN IMMEDIATE` + retry and a
  lock-guarded shared connection; SQLiteSearchIndex likewise;
  `Memory` holds one lock across public operations. New concurrency test
  module.
- **Scaling.** Consolidation near-dup/contradiction passes compare blocked
  candidate pairs (shared stemmed content word or slot key) instead of all
  pairs; `remember` dedup is a verified body→id map instead of a full scan;
  consolidation reports name every `touched` id so the porcelain refreshes
  indexes incrementally instead of re-indexing the store.
- **Contradiction scope.** Episodes (timestamped events, not truth-claims)
  are excluded from contradiction detection; multi-way contradictions are
  reported as deferrals instead of silently dropped. (SPEC §9.2.)
- **Antonym/value-swap gap.** New `text.py` with `infer_slots`: equative
  "X is Y" facts get a coarse subject→value slot from both default
  extractors, so "staging DB is Postgres" vs "…is MySQL" is detected with
  no LLM. Copula-only, multi-word subjects only — non-exclusive claims
  never manufacture disputes.
- **Rule flags.** A rule is flagged exactly when *no* evidence is live;
  missing evidence counts as invalidated but one live record keeps the rule
  unflagged. (SPEC §9.3.)
- **Retrieval.** Per-index relevance normalized to [0, 1] before the
  best-across-indexes union (bm25 and cosine scales are incomparable); the
  FTS index no longer pads results with zero-score non-matches.
- **Salience persistence.** The reinforcement sidecar persists atomically to
  `salience.json`; repeat remembers merge their tags onto the existing
  record instead of dropping them.

Phases 2–3: the running system over the phase-1 foundation, and the
user-facing layer over the full machinery.

- **Porcelain `Memory` API** (`fg_agent_memory.Memory`): the two-call
  experience — `remember(text)` / `recall(query)` — over the complete
  pipeline. Defaults: a readable directory of canonical-JSON records
  (`FileRecordStore`) with a SQLite FTS sidecar index, a verbatim extractor
  (what you say is what is stored), auto-consolidation every few remembers,
  and no auto-resolver — contradictions surface both sides in recall until
  an explicit `resolve(record_id, winner_id, reason)`. Every layer is an
  optional kwarg (`store=`, `indexes=`, `extractor=`, `operators=`, configs,
  `identity=` for signing). `export(path)` / `Memory.load(path)` round-trip
  the signed portable `MemoryFile`; `recall(...).as_prompt_block()` renders
  a clean, model-ready block with disputed/superseded records labeled.
- **Storage adapters** (SPEC §6): `FileRecordStore` — one directory per
  record id, one canonical-JSON file per version, atomic
  temp-file-plus-rename writes, git-friendly append-only layout — and
  `SQLiteRecordStore` (WAL, one transaction per put, queryable
  state/type/tags beside an append-only history table), plus
  `SQLiteSearchIndex`, an FTS5 keyword index with query-syntax injection
  ruled out by construction. All pass the shared port contract suite.
- **State-aware retrieval** (SPEC §7): multiplicative scoring
  (relevance × state weight × recency × salience × confidence),
  transition pairs budgeted atomically so a dispute never surfaces
  one-sided, superseded/archived excluded by default and includable
  explicitly, whole-record token budgeting, deterministic tie-breaking,
  and machine-readable inclusion reasons. `TouchCounts` is the
  retrieval-reinforcement sidecar.
- **Pipeline** (SPEC §8–§10): `ingest` (create-only write stage,
  content-address dedup as reinforcement), `consolidate` (exact dedup →
  near-dup merge → contradiction detection → resolution → promotion →
  rule-flag sweep, every operator proposal validated against the lifecycle
  machine, full audit report), `resolve` (explicit winner naming), and
  `decay` (mechanical salience-driven archival with protections for
  transitional records, unflagged rules, and rule evidence — reported,
  never silent).
- **MCP server** (`fg-agent-memory-mcp`, optional `[mcp]` extra): stdio
  server exposing `remember` / `recall` / `consolidate` / `status` over a
  `Memory` rooted at `--path`. Import-guarded — the core package never
  requires the MCP SDK.

## [0.1.0]

- **Record format** (SPEC §2): typed `MemoryRecord`
  (`fact`/`episode`/`procedure`/`rule`) with content-addressed ids
  (`mem:` + 128-bit SHA-256 of the birth content), explicit lifecycle state,
  string-encoded confidence, provenance, evidence links (required non-empty
  for rules), supersession links, transition blocks, tags, and
  preserve-and-sign unknown fields. Records are individually signable with
  fg-agent-id keys under the domain-separated context
  `fg-agent-memory/v1/record`; `MemoryFile` is the signed portable container.
- **Lifecycle state machine** (SPEC §3): deterministic, code-enforced
  transitions over `active | superseded | transitional | archived` — pure
  functions, no storage dependency. Supersession must name successor and
  reason; contradiction produces a shared transition block; resolution names
  the winner; archival is terminal and records are never deleted. Rules are
  flagged when all their evidence is archived/superseded.
- **Adapter ports** (SPEC §5): append-only `RecordStore` (versioned history,
  state/type/tag filters), `SearchIndex` over an `Embedder` port with a
  deterministic hash-based default, and the `Operator` port — typed proposals
  validated against the lifecycle machine before applying. Reference
  in-memory implementations and a model-free `HeuristicExtractor` included.
- Wire spec (`spec/SPEC.md`, RFC 2119) and golden conformance vectors
  (`spec/vectors.json`) covering record ids, canonical bytes, signing inputs,
  signed artifacts, and the full lifecycle legality table.
