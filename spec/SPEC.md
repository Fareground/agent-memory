# Agent-Memory Wire Specification

**Version:** `fg-agent-memory/0.1` · **Status:** draft · **Companion:** `spec/vectors.json` (golden conformance vectors)

This document specifies the agent-memory record format, lifecycle semantics,
and adapter contracts precisely enough for an independent implementation to
interoperate byte-for-byte with the Python reference. Every `MUST`/`SHOULD`/
`MAY` is [RFC 2119]. Where this document and the code disagree, the golden
vectors in `spec/vectors.json` are authoritative for the byte-level
constructions they cover.

Identity primitives (canonical JSON, Ed25519 keys, `amp:key:` addresses,
base64 signature canonicality) are inherited unchanged from the sibling
standard [fg-agent-id]; this document specifies only what memory adds.

## 0. Rationale

Memory is one agent's private, persistent experience across sessions —
distinct from shared, governed knowledge. Everything shipping today stores
and retrieves; **consolidation** — contradiction resolution, episode→rule
promotion, decay — is measured broken everywhere, because production systems
implement it as an LLM freely rewriting a store. This standard's answer, in
five commitments:

Cross-agent sharing lives in the sibling **fg-agent-knowledge** standard, not
here: importing another agent's memory file (§4) merges only disjoint record
ids and never fuses two agents' records. Promoting a private memory into shared
knowledge is a NEW claim the promoting agent authors in the knowledge layer,
where governed convergence (contradiction, confidence, promotion) — not a
structural cross-author merge — reconciles it. This standard owns one agent's
memory; it deliberately defines no multi-agent merge.

1. **Fact lifecycle as a state machine, not metadata.** Every record carries
   an explicit state with deterministic, code-enforced transitions.
   Contradiction produces a transition pair, never a silent overwrite.
2. **Evidence-linked promotion.** A rule keeps machine-readable links to the
   episodes that justify it; invalidate the evidence and the rule is flagged.
3. **Mechanical decay.** Archival is a lifecycle state driven by code, not
   LLM mood — and records are never deleted.
4. **The portable record IS the standard.** An agent's memory is a signed
   file it owns, usable across runtimes. Engines compete on pipelines; the
   record format is the interop contract.
5. **The LLM is an operator inside the pipeline, never a free rewriter.**
   Model output is a typed proposal, validated against the state machine
   before anything is written.

## 1. Inherited primitives

| Primitive | Source |
|---|---|
| Canonical JSON (sorted keys, NFC, no floats, no whitespace) | fg-agent-id SPEC §3 |
| Ed25519 signatures, base64 canonical spelling | fg-agent-id SPEC §1, §3 |
| Self-certifying `amp:key:<base58>` addresses | fg-agent-id SPEC §2 |
| Timestamp spelling `YYYY-MM-DDTHH:MM:SS.mmmZ` (UTC, truncate to ms) | fg-agent-id SPEC §3.2 |

Implementations MUST reproduce these byte-for-byte; the fg-agent-id golden
vectors cover them.

### 1.1 Domain-separated signing input

Every signature in this standard is computed over:

```
signing_input(context, payload) =
    uint16be(len(tag)) || tag || canonical_json(payload)

tag = UTF-8("fg-agent-memory/v1/" || context)
```

`context` is exactly one of:

| Context | Artifact |
|---|---|
| `record` | §2 `MemoryRecord` |
| `memory-file` | §4 `MemoryFile` |

A verifier MUST verify under the context of the artifact type it expects and
MUST NOT accept a signature that verifies only under a different context.
The domain differs from fg-agent-id's by construction, so a memory signature
can never be replayed as an identity artifact or vice versa. New artifact
types MUST take new context strings; contexts MUST NOT be reused.

See `vectors.signing_input` and `vectors.artifacts.*.signing_input_hex`.

## 2. Record format

A `MemoryRecord` is a JSON object with these fields:

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Content address (§2.1) |
| `type` | string | `fact` \| `episode` \| `procedure` \| `rule` |
| `body` | string | The memory itself, free text |
| `slots` | object | OPTIONAL structured view of the body; free JSON |
| `state` | string | `active` \| `superseded` \| `transitional` \| `archived` |
| `confidence` | string | §2.2 wire confidence in `[0,1]` |
| `created_at` | string | RFC 3339 ms UTC; ingest time (when the agent learned it); immutable after birth |
| `updated_at` | string | RFC 3339 ms UTC; bumped on every state change |
| `occurred_at` | string | OPTIONAL event time (when the remembered event happened), RFC 3339 ms UTC; immutable after birth; episodes SHOULD carry it |
| `provenance` | object | `{kind, ref, agent}` — source kind (open vocabulary, e.g. `conversation`, `observation`, `tool`, `consolidation`, `import`), optional source reference, optional writer address |
| `evidence` | array | Record ids justifying this record; MUST be non-empty when `type` is `rule` |
| `supersedes` | string \| null | Id of the record this one replaces |
| `superseded_by` | string \| null | Id of this record's successor |
| `transition` | object \| null | §3.2 transition block; present iff `state` is `transitional` |
| `state_reason` | string \| null | Why the record entered its current state |
| `tags` | array | Free-form string tags |
| `flagged` | boolean | §3.4 evidence-invalidation flag; rules only |
| `flag_reason` | string \| null | Why the record is flagged |
| `signed_by` | string \| null | Signer's `amp:key:` address |
| `signature` | string | Base64 Ed25519 over the canonical record sans `signature` |

Unknown fields MUST be preserved on parse, re-emitted on serialization, and
included in the signed payload, so a newer record carrying extension fields
still verifies on an older implementation. An unknown field whose name
collides with a defined field MUST be rejected.

Optional fields marked OPTIONAL in the table (`occurred_at`, and the §11
redaction fields) MUST be emitted only when set — a record that never
carried them serializes byte-identically to one produced before the field
existed, so signatures over pre-existing records keep verifying.
`occurred_at` is signed but NOT part of the content address (§2.1): event
time refines a memory, it does not rename it. Where both sides of a
temporal-precedence resolution (§9.2) carry `occurred_at`, "newer wins"
MUST compare event time, not ingest time.

Producers MUST NOT emit `superseded_by` on a record whose state is `active`
or `transitional`, and MUST NOT set `flagged` on a non-rule record.

### 2.1 Content-addressed identity

`id` is derived from the record's immutable birth content:

```
id = "mem:" || hex(sha256(canonical_json(birth)))[0:32]

birth = {"type": ..., "body": ..., "slots": ...,
         "provenance": {...}, "created_at": "..."}
```

(32 lowercase hex characters = 128 digest bits.) Lifecycle fields — state,
links, confidence, flags, `updated_at` — are versioned *around* the id: a
state change MUST NOT change what a record is called. Two records extracting
the same content from the same source at the same instant name it
identically, so dedup by content identity is a byte comparison.

An implementation MUST reject a record whose `id` does not equal the content
address recomputed from its birth fields — except a §11 redacted tombstone,
whose content is destroyed and whose id is explained by `birth_digest`
instead. See `vectors.record_id`.

### 2.2 Confidence on the wire

Canonical JSON forbids floats. Confidence MUST be rendered as a decimal
string with exactly four fractional digits (`"0.5000"`, `"1.0000"`), value
within `[0, 1]`. Implementations MUST reject values outside the range.

### 2.3 Record signatures

A record MAY be signed by an agent identity (fg-agent-id keypair). The
signature is computed under context `record` (§1.1) over the canonical
record with the `signature` field removed and `signed_by` set to the
signer's address. Because `signed_by` is inside the signed payload, a
signature cannot be re-attributed to another agent. Verification resolves
the public key from the self-certifying address alone.

A signature covers the full record payload, so ANY lifecycle mutation
(state, links, flags, `updated_at`) invalidates it. An implementation
producing a new lifecycle version of a signed record MUST NOT carry the
stale signature forward: it MUST clear `signed_by` and `signature` on the
evolved version (the signed birth version stays addressable in history), and
MAY re-sign the evolved version explicitly. A record whose signature does
not verify against its own payload MUST be treated as unsigned-invalid, not
silently accepted.

## 3. Lifecycle

Every record is in exactly one state. The complete legality table — anything
not listed MUST be rejected:

| From | To | Condition |
|---|---|---|
| `active` | `superseded` | Successor MUST exist, MUST name this record in `supersedes`, and a reason MUST be given (§3.1) |
| `active` | `transitional` | MUST carry a transition block naming both sides (§3.2) |
| `transitional` | `active` | Resolution: this record is the winner (§3.3) |
| `transitional` | `superseded` | Resolution: this record is the loser, superseded by the winner (§3.3) |
| `active` \| `superseded` \| `transitional` | `archived` | Decay only; a reason MUST be given |

`archived` is terminal. Records are NEVER deleted: superseded and archived
records MUST stay addressable forever. The legality table is encoded as
vectors in `vectors.lifecycle.legality`.

### 3.1 Supersession

Replacing a memory MUST name what is replaced: the successor record carries
`supersedes = old.id`, and the retired record gains `state = superseded`,
`superseded_by = successor.id`, and a `state_reason`. A supersession whose
successor does not reference the superseded record MUST be rejected — that
is precisely the silent overwrite this standard exists to prevent.

### 3.2 Transition (contradiction without a winner)

When two records contradict and neither is yet known to win, both MUST move
to `transitional` carrying the same transition block:

```json
{"sides": ["<id-a>", "<id-b>"], "reason": "..."}
```

`sides` MUST name exactly two distinct record ids and MUST include the
carrying record's own id. A record in `transitional` state without a
transition block, or a transition block on any other state, MUST be
rejected. Retrieval SHOULD surface both sides of a transition.

### 3.3 Resolution

Resolving a transition names the winner: the winner returns to `active` with
its transition block cleared; the loser becomes `superseded` with
`superseded_by = winner.id`. Both records MUST be the two sides of the same
transition block, and a reason MUST be given.

### 3.4 Rule evidence invariants

A record of type `rule` MUST carry a non-empty `evidence` list of record
ids. When every evidence record of a rule is invalidated (state `archived`
or `superseded`), the rule MUST be flagged: `flagged = true` with a
`flag_reason`. A flagged rule remains in its lifecycle state — flagging is
machine-visible doubt, not an automatic demotion; retiring the rule is a
separate, explicit lifecycle step. If any evidence record is live (`active`
or `transitional`) the flag MUST be cleared. Flag evaluation over a partial
evidence set MUST be rejected.

## 4. Memory file

The portable artifact: everything an agent remembers, in a file it owns.

| Field | Type | Meaning |
|---|---|---|
| `format` | string | `fg-agent-memory/0.1` |
| `agent` | string \| null | Owning agent's `amp:key:` address |
| `created_at` | string | RFC 3339 ms UTC export time |
| `records` | array | `MemoryRecord` objects (§2) |
| `signature` | string | Base64 Ed25519 over the canonical file sans `signature`, context `memory-file` |

The canonical bytes of a file are `canonical_json` of the object with
`signature` removed. A file-level signature is OPTIONAL but a signed file
MUST name its `agent`, and record-level signatures inside the file remain
independently verifiable. Unknown header fields follow §2's
preserve-and-sign rule.

A verifying loader MUST NOT let file content decide whether verification
runs: when the caller requests verification, a missing or empty `signature`
is a verification FAILURE, never a silent skip — otherwise stripping the
signature downgrades a signed file to an unverified one. Accepting unsigned
files MUST be an explicit caller opt-out. A verifying loader MUST also
verify every record in the file that names a `signed_by` signer.

## 5. Adapters (ports)

The standard is the record format and lifecycle semantics — never the
backend. Every invariant in this document is stateable over the record
format alone; an invariant that only exists as a query on a particular
backend is a leak and MUST NOT be introduced.

### 5.1 RecordStore

Append-only storage keyed by record id. `put` MUST NOT overwrite: writing an
existing id appends a new version, and every prior version MUST stay
addressable (`history`, oldest first). `get` returns the latest version.
`list` filters latest versions by state, type, and tag. Deletion is not part
of the contract.

### 5.2 SearchIndex and Embedder

`index(record)` makes a record findable; `candidates(query, k)` returns up
to `k` `(record_id, score)` pairs, best first. The embedding function is
itself a port. Implementations MUST ship a deterministic model-free default
(the reference uses token-hash bag-of-words, L2-normalized) so that nothing
in the framework requires a model. Retrieval policy — state-awareness
(active outranks superseded; transitional surfaces both sides), recency,
token budgets — lives above this port.

### 5.3 Operator

The LLM touchpoint. An operator receives typed input and returns typed
proposals (`create` | `supersede` | `transition` | `archive`); it MUST NOT
have write access to the store. The pipeline MUST validate every proposal
against the lifecycle machine (§3) before applying, and MUST reject — not
repair — invalid proposals. **The LLM is an operator inside the pipeline,
never a free rewriter.** Implementations MUST be runnable with no model:
the reference ships a regex-based `HeuristicExtractor` that proposes only
facts and episodes — rules exist only through evidence-linked promotion,
which no extractor may shortcut.

[RFC 2119]: https://www.rfc-editor.org/rfc/rfc2119
[fg-agent-id]: https://github.com/Fareground/agent-id

## 6. Storage adapters

A storage adapter is any concrete `RecordStore` (§5.1). The contract below
binds every adapter; the parametrized port contract tests are its executable
form, and an adapter that passes them is conformant regardless of backend.

- **Append-only.** `put` MUST NOT overwrite or delete. Writing an existing
  id MUST append a new version; every prior version MUST remain addressable
  through `history` (oldest first) and MUST NOT be mutated after the write
  completes. Deletion MUST NOT be offered — decay is a lifecycle state.
- **Atomicity.** A version MUST be either fully persisted or absent. An
  interrupted write MUST NOT leave a torn or partially visible version
  (file adapters SHOULD use temp-file-plus-rename; database adapters SHOULD
  use a transaction per put).
- **Wire fidelity.** Adapters MUST persist the canonical wire form (§2) so
  that a stored record round-trips byte-for-byte, signatures included.
  Query-oriented projections (state/type/tag columns, index rows) are
  derived data and MUST NOT be authoritative over the stored payload.
- **History addressability.** `get` returns the latest version; `history`
  MUST return every version in write order. Both MUST fail on unknown ids
  rather than answering emptily.

The reference ships two adapters. The **file store** lays records out as
one directory per id containing one canonical-JSON file per version, with
zero-padded version filenames so lexicographic order is version order. This
layout is REQUIRED for file-store interoperability: paths are stable
functions of the record id, history is append-only on disk (a lifecycle
change adds a file, never rewrites one — diffs under version control are
pure additions), and no index file churns on writes. The **SQLite store**
keeps a latest-version table with queryable state/type/tags beside an
append-only history table, in WAL mode, one transaction per put.

An FTS keyword `SearchIndex` (the reference uses SQLite FTS5 over body and
tags) is OPTIONAL: keyword search is an index choice, never part of the
storage contract. An FTS index MUST treat raw query text as text — query
syntax injection MUST NOT be possible — and MUST rank matches above
non-matches with deterministic id tie-breaking.

## 7. Retrieval

Retrieval is the pipeline stage that turns index candidates into what an
agent loads. Implementations MUST score each candidate as the product of
independent factors:

```
score = relevance × state_weight × recency × salience × confidence
```

The constants are implementation-chosen; the invariants are not:

- **Relevance** comes from one or more `SearchIndex` ports; with several
  indexes, a record's relevance is its best score across them, floored at
  zero.
- **State weight** MUST order states strictly: active > transitional >
  superseded > archived. Given equal relevance, recency, salience, and
  confidence, an active record MUST outrank a superseded one.
- **Recency** MUST decay from `updated_at` toward a positive floor — age
  discounts a record, it MUST NOT erase it.
- **Salience** MUST NOT decrease with retrieval reinforcement: each time a
  record is returned, its touch count increments, and a higher count MUST
  never lower its score. Touch counts are retrieval state owned by the
  caller, never part of the signed record.
- **Confidence** is the record's own field.

State-aware semantics:

- **Transitional records surface both sides.** Including one side of a
  transition MUST pull its partner (the other id in `transition.sides`)
  into the same result, budget permitting — the pair is budgeted
  atomically, and a result MUST NOT present one side of an unresolved
  contradiction without the other.
- **Superseded records** MUST be excluded as direct candidates by default.
  They MAY be included when history is explicitly requested, in which case
  an included record's `supersedes` chain SHOULD be surfaced with it,
  each entry marked as a predecessor of its successor.
- **Archived records** MUST be excluded by default and MAY be includable by
  an explicit flag.

Budgeting and determinism:

- The token budget selects **whole records only** — a record (or an atomic
  pair) that does not fit is skipped, never truncated mid-record.
  Compression is selection, not rewriting: retrieval MUST NOT alter record
  bodies. Token cost MAY be approximate (the reference uses a
  characters-per-token heuristic).
- Ranking MUST be deterministic: equal scores tie-break on record id.
- Every included record MUST carry a machine-readable reason for its
  inclusion (query match, transition partner, predecessor), so a retrieval
  result is explainable after the fact.

## 8. Write stage

The write stage turns raw input (text or a structured event, rendered
deterministically as sorted `key: value` lines) into stored records via an
extractor operator. Its authority is deliberately narrow:

- The write stage MUST admit only `create` proposals. Lifecycle mutations
  (`supersede`, `transition`, `archive`) proposed by an extractor MUST be
  rejected.
- An extractor MUST NOT create rules. Rules exist only through
  evidence-linked promotion (§9.3); a rejected rule-creation MUST appear in
  the ingest report.
- Extracted records MUST be born `active`, with no lifecycle links
  (`supersedes`, `superseded_by`, `transition`) and unflagged. Confidence
  bounds are enforced by the record format (§2).
- Dedup is by content address: a proposed record whose id already exists in
  the store MUST be treated as a reinforcement of the existing record (a
  touch in the salience sidecar, §10), MUST NOT be re-written, and MUST be
  reported as deduplicated.
- The write stage MUST NOT be destructive, and MUST return a report naming
  every accepted, deduplicated, and rejected proposal with reasons.
  Rejection is total: a rejected proposal leaves the store untouched.

## 9. Consolidation

Consolidation is a background pass composed of deterministic sub-passes:
exact dedup → near-dup merge → contradiction detection → resolution →
promotion → rule-flag sweep. The governing principle is §5.3's: detection
quality is an operator port per sub-pass, but every operator output is a
typed proposal that MUST be validated against the lifecycle machine (§3)
before apply. A proposal naming an illegal transition, or overstepping its
sub-pass's authority, MUST be rejected and logged in the report — never
applied, never repaired.

The consolidation report is a first-class output: every mutation MUST be
recorded with the records' before/after states and a reason, and every
rejection with its cause. Given the same store contents, operators, clock,
and configuration, a consolidation pass MUST be deterministic — same input,
same report.

### 9.1 Dedup

Exact dedup is pure code: active records identical in type, body, and slots
(distinct content addresses via differing provenance or birth time) collapse
to the strongest copy; each weaker copy is superseded by a version of the
keeper that names it, with a reason.

Near-dup detection is an operator port. The reference default uses character
trigram Jaccard similarity against a configured threshold. A merge proposal
MUST be a supersede whose merged record carries evidence links to BOTH sides
of the pair and supersedes one of them; anything else MUST be rejected. Both
originals remain addressable, with `superseded_by` chains pointing at the
merged record. A near-dup operator SHOULD abstain when shared slots disagree
(that pair is a contradiction candidate, §9.2 — merging it away would be a
silent overwrite) and SHOULD abstain on episodes (similar text on distinct
events is not duplication).

### 9.2 Contradiction

Candidate-pair detection is an operator port. The reference default is
deterministic negation-polarity detection: same slot key with different
values, or a body pair whose polarity differs (negator tokens or a
"no X needed/required" polarity frame on exactly one side) over a shared
content-word core — normalized, lightly stemmed content words whose
Jaccard overlap meets a configured threshold. This catches negation-marked
contradictions, including morphological variants and "no X needed"-style
phrasings; semantic or paraphrase contradictions carrying no polarity
marker are beyond the default and need an LLM operator on the port.

Episodes are excluded from contradiction detection alongside rules: an
episode is a timestamped occurrence, not a competing truth-claim — opposite
polarity across two events is history, not a dispute. Because a transition
names exactly two sides, a record joins at most one transition per pass;
a further detected pair involving an already-entangled record MUST be
reported as deferred, never silently dropped.

A detected contradiction MUST route through the transitional protocol
of §3: both records move to `transitional`, carrying the same transition
block, both sides retrievable. A contradiction MUST NEVER silently overwrite
either side.

Resolution is a separate step — `resolve(pair, winner, reason)` — applied
via lifecycle transitions: the winner returns to `active`, the loser becomes
superseded-but-addressable with the reason on record. An auto-resolver
operator MAY propose a winner. The reference resolver encodes the researched
caution about temporal precedence: an explicit user correction beats
inference outright, but "newer wins" applies ONLY when both sides share the
same provenance kind and identical slot keys — recency across differing
sources or shapes is not evidence of truth, and such pairs stay transitional
until explicitly resolved.

**Normative example (ghost memory).** X is asserted; not-X is asserted
later. Detection moves the pair to `transitional` — both sides retrievable,
each carrying the transition block naming the other. Resolution names not-X
the winner with a reason. End state: not-X is `active`; X is `superseded`
with `superseded_by` = not-X and the resolution reason as `state_reason`,
and every prior version of X (active → transitional → superseded) remains
addressable via history. X can never again be retrieved as current truth,
and it can never silently return — the ghost cannot form.

### 9.3 Promotion

Repeated episodes become rules. When at least the configured threshold of
active episodes share a signature (tags, else slot keys), consolidation
proposes ONE rule record whose `evidence` MUST link ALL supporting episodes;
non-empty evidence is enforced by the record format at birth (§2). Promotion
MUST be idempotent per signature — an existing non-archived rule with the
same signature suppresses re-promotion.

Every consolidation pass MUST also run the rule-flag sweep of §3.4: a rule
is flagged with a reason exactly when NO evidence record is live — a record
missing from the store counts as invalidated, but any live evidence keeps
the rule unflagged. Flag changes are report-recorded mutations like any
other.

## 10. Decay

Decay is mechanical: salience is a pure function of confidence,
reinforcement count, and age since last touch, with every constant named in
a decay configuration. The reference scoring — confidence halved per
half-life of idleness, plus a linear bonus per reinforcement touch — and its
default constants are SHOULD-level recommendations, not interop
requirements; the protections below are MUSTs.

- A record whose salience falls below the archive threshold is archived via
  the lifecycle machine (§3). Records MUST NEVER be deleted; archived
  records stay addressable with their full version history.
- `transitional` records MUST NOT decay: an unresolved dispute keeps both
  sides visible regardless of age.
- An unflagged rule MUST NOT decay. A rule may be archived by decay only
  after the flag sweep (§9.3) has flagged it.
- A record cited as `evidence` by an active rule MUST NOT decay, whatever
  its own salience — evidence links are load-bearing.
- Protections MUST be reported, not silent: a below-threshold record that
  survives appears in the decay report with the protection named.

Reinforcement is a sidecar, not a record field: a `touch(record_id)` hook
(intended for retrieval hits) bumps an engine-local counter and recency
stamp that decay reads back. This is deliberate — touch counts are volatile
working state, and signing a new record version per retrieval hit would
bury the audit trail; losing the sidecar degrades gracefully to
age-since-update decay. The portable record format (§2) is unchanged by
reinforcement.

## 11. Redaction

Records are never deleted — but content may be destroyed under an explicit,
audited **redaction**, the escape hatch for secrets or personal data
remembered by mistake. Redaction is a compliance action, categorically
distinct from decay: archival keeps content and loses attention; redaction
destroys content and keeps existence.

Redacting a record replaces EVERY stored version of its id with one
**tombstone**:

| Field | Tombstone value |
|---|---|
| `id`, `type`, `created_at` | preserved |
| `evidence`, `supersedes`, `superseded_by` | preserved (ids are not content; the link graph must not dangle) |
| `state` | `archived` |
| `body` | `"[redacted]"` |
| `slots`, `tags`, `occurred_at` | wiped |
| `provenance` | `{kind: "redaction"}` — original provenance detail is content |
| `confidence` | `"0.0000"` |
| `redacted` | `true` |
| `redaction_reason` | REQUIRED non-empty |
| `birth_digest` | full sha256 hex of the original canonical birth content |
| `signed_by`, `signature` | cleared — the signature covered destroyed content |

`redacted`, `redaction_reason`, and `birth_digest` follow §2's
emit-only-when-set rule, so non-redacted records are wire-unchanged. A
redacted record's `id` cannot be recomputed from its fields; verifiers MUST
accept it on the strength of `birth_digest` (of which the id is a prefix
by construction §2.1) and MUST treat the loss of content tamper-evidence as
the deliberate cost of destruction. A record carrying redaction fields
without `redacted: true`, a redacted record not in `archived` state, or a
redaction without a reason MUST be rejected.

Redaction MUST NOT be part of the `RecordStore` port (§5.1) or reachable by
any operator proposal — it is a separate, explicitly-invoked maintenance
operation. The redaction itself is permanently visible: anyone reading the
store sees that something was redacted, when, and why — never what.
Version count and addressability are preserved (every version of the id
resolves to the tombstone). A rule whose evidence is redacted loses live
evidence like any other invalidation and is flagged by the §9.3 sweep.

For deployments on hostile or unerasable storage, per-record encryption
with key destruction (crypto-shredding) is the stronger profile; it is out
of scope for this document and does not change the tombstone semantics
above.
