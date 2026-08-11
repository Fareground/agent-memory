<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/wordmark-dark.svg" />
    <source media="(prefers-color-scheme: light)" srcset="assets/wordmark.svg" />
    <img alt="Fareground" src="assets/wordmark.svg" width="320" />
  </picture>
</p>

<h1 align="center">agent-memory</h1>

<p align="center"><em>Portable, signed agent memory with a code-enforced fact lifecycle.</em></p>

<p align="center">
  <a href="https://github.com/Fareground/agent-memory/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/Fareground/agent-memory/ci.yml?branch=main&style=flat-square&label=CI" /></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11+-3b82f6?style=flat-square" />
  <a href="https://pypi.org/project/fg-agent-memory/"><img alt="PyPI" src="https://img.shields.io/pypi/v/fg-agent-memory?style=flat-square" /></a>
</p>

---

## Overview

**agent-memory** is the memory standard in the Fareground agent-standards family: one agent's persistent experience across sessions, as a file the agent owns. Typed records (`fact` / `episode` / `procedure` / `rule`) carry provenance, confidence, and an explicit lifecycle state, are content-addressed and individually signable with [fg-agent-id](https://github.com/Fareground/agent-id) identities, and travel in a signed, canonical memory file that any runtime can verify. Engines compete on pipelines; the record format is the interop contract.

The differentiator is **consolidation as code, not vibes**. Most memory systems implement contradiction resolution, promotion, and decay as an LLM freely rewriting a store. Here the fact lifecycle is a deterministic state machine: superseding must name its successor and reason, contradiction produces a transition pair instead of a silent overwrite, rules keep machine-readable evidence links and are flagged the moment their evidence is invalidated, and archival is a terminal state — records are never deleted. The LLM is an operator *inside* the pipeline whose typed proposals are validated against the machine before anything is written; it is never a free rewriter.

The standard is storage-agnostic by construction: `RecordStore`, `SearchIndex`, `Embedder`, and `Operator` are adapter ports, every invariant is stateable over the record format alone, and the reference ships in-memory implementations plus a deterministic hash embedder and model-free operators so the whole framework runs and tests with no model and no backend.

## Status

**Alpha.** The library is functional end to end, and the byte-level constructions are pinned by golden wire vectors. The wire format version is pinned at `0.1` and is expected to move before a `1.0` freeze — treat on-disk records and signatures as not yet stable across versions, and expect APIs to change. See [`CHANGELOG.md`](CHANGELOG.md) for the current state and recent changes.

## Install

The package is distributed as `fg-agent-memory` (import path `fg_agent_memory`); its only dependency, the sibling [fg-agent-id](https://github.com/Fareground/agent-id) identity library, resolves automatically.

```bash
pip install fg-agent-memory

# with the optional MCP stdio server
pip install "fg-agent-memory[mcp]"
```

For development from a checkout of this repo:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Usage

Two calls. No lifecycle vocabulary required.

```python
from fg_agent_memory import Memory

memory = Memory("./memory")          # writes a readable directory you can commit to git
memory.remember("Sandro's favorite editor is Zed.")
print(memory.recall("what editor does Sandro use?").as_prompt_block())
```

Under the hood that was: a canonical-JSON record directory (`FileRecordStore`), a SQLite FTS index sidecar, content-address dedup, and auto-consolidation every few remembers. All of it is swappable — `Memory(store=..., indexes=..., extractor=..., operators=..., identity=...)` — none of it is required.

Want every record and export signed by a persistent [fg-agent-id](https://github.com/Fareground/agent-id) identity? Pass a keyfile path — it is created on first run and reloaded ever after:

```python
memory = Memory("./memory", identity="agent.key")
```

(An existing `KeyPair` works too; the string form is `load_or_create_keys` under the hood.)

### The ghost-memory demo

The failure mode this standard exists to kill: you learn X, later learn not-X, and the stale X silently haunts retrieval forever. Here a contradiction is a first-class state — both sides stay visible until *you* decide, and the loser is retired addressably, never deleted.

```python
from fg_agent_memory import Memory

memory = Memory("./memory")

# Two assertions that cannot both be true.
a = memory.remember("The staging database is Postgres.")
b = memory.remember("The staging database is not Postgres.")

# Consolidation detects the contradiction. Neither side is dropped.
memory.consolidate()
print(memory.recall("staging database").as_prompt_block())
# - The staging database is Postgres. [disputed: one side negates the other]
# - The staging database is not Postgres. [disputed: one side negates the other]

# You name the winner, with a reason that goes on the record.
memory.resolve(a.record_id, b.record_id, "checked infra: staging moved off Postgres in March")

print(memory.recall("staging database").as_prompt_block())
# - The staging database is not Postgres.

# The loser is superseded, not gone — history is one flag away.
print(memory.recall("staging database", include_history=True).as_prompt_block())
# - The staging database is not Postgres.
# - The staging database is Postgres. [superseded — kept for history]
```

This exact snippet is executable as written. The default detector is deterministic negation-polarity matching over shared content words (it catches "X" vs "not X" and "no X needed" phrasings, morphology included); contradictions with no negation marker need an LLM operator plugged into the contradiction port.

### MCP server

Any MCP client gets `remember` / `recall` / `consolidate` / `redact` / `status` tools over one memory directory (requires the `[mcp]` extra):

```bash
fg-agent-memory-mcp --path ~/agent-memory
```

```json
{ "mcpServers": { "memory": { "command": "fg-agent-memory-mcp", "args": ["--path", "/home/me/agent-memory"] } } }
```

## Concepts

```
            porcelain: Memory — remember / recall / consolidate / decay / resolve / export
            ───────────────────────────────────────────────────────────────────────────────
  raw text ──► write stage ────► RecordStore ◄──── consolidation ◄──── decay
               (extractor        (append-only,     dedup → contradiction   (mechanical
                Operator,         versioned         → resolution →          salience;
                create-only)      history)          promotion → flags)      protections
                                      │                                     reported)
                                      ▼
                SearchIndex ───► retrieval (state-aware, budgeted) ───► prompt block
                                      │
                                      ▼
                                 MemoryFile (signed, portable — the file the agent owns)
```

Every arrow into the store passes through the lifecycle state machine (`active | transitional | superseded | archived`); operators — LLM or heuristic — only ever emit typed proposals that are validated first and rejected, never repaired.

The reference ships implementations for every port, so the whole framework runs with no model and no backend:

| Port | Shipped implementations | Bring your own |
|---|---|---|
| `RecordStore` | `FileRecordStore` (git-friendly canonical-JSON directory), `SQLiteRecordStore` (WAL, queryable), `InMemoryRecordStore` | any append-only versioned backend |
| `SearchIndex` | `SQLiteSearchIndex` (FTS5 keyword), `InMemorySearchIndex` (cosine over an `Embedder`) | vector DBs, hybrid search |
| `Embedder` | `HashEmbedder` (deterministic, model-free) | any embedding model |
| `Operator` (write) | `VerbatimExtractor` (porcelain default), `HeuristicExtractor` (regex) | LLM extractors |
| `Operator` (consolidation) | `TrigramNearDupOperator`, `HeuristicContradictionOperator`, `HeuristicResolutionOperator` | LLM detectors / resolvers — see [`examples/llm_operator.py`](examples/llm_operator.py) |

Every store implementation satisfies one shared port contract; conformance is the contract, not the backend.

The wire format, lifecycle legality table, adapter contracts, retrieval scoring, and pipeline authority rules are normatively specified in [`spec/SPEC.md`](spec/SPEC.md) (RFC 2119); byte-level constructions are pinned by the golden vectors in [`spec/vectors.json`](spec/vectors.json). Where prose and code disagree, the vectors win.

## Project structure

- `src/fg_agent_memory/memory.py` — the porcelain `Memory` API
- `src/fg_agent_memory/records.py` — `MemoryRecord` + `MemoryFile`, the portable format
- `src/fg_agent_memory/lifecycle.py` — the state machine (pure functions)
- `src/fg_agent_memory/ports.py` — storage/search/operator ports + reference impls
- `src/fg_agent_memory/stores/` — file + SQLite adapters, FTS keyword index
- `src/fg_agent_memory/retrieval.py` — state-aware, budgeted retrieval
- `src/fg_agent_memory/pipeline/` — write stage, consolidation, decay
- `src/fg_agent_memory/mcp_server.py` — the MCP stdio server (`[mcp]` extra)
- `spec/SPEC.md` + `spec/vectors.json` — the wire spec and golden vectors
- `examples/` — the quickstart and an LLM consolidation operator skeleton

## Contributing

Development setup, test commands, and style conventions are in [CONTRIBUTING.md](CONTRIBUTING.md).

---

<p align="center"><sub>Built by <a href="https://github.com/Fareground">Fareground</a> · Licensed under <a href="LICENSE">Apache-2.0</a>.</sub></p>
