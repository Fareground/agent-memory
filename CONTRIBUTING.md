# Contributing to agent-memory

Thanks for your interest in improving the memory standard. This document covers
local setup, tests, and the conventions the codebase follows.

## Development setup

The package depends on the sibling [fg-agent-id](https://github.com/Fareground/agent-id)
identity library. Check it out next to this repository, then install both
editable into a virtual environment:

```bash
python -m venv .venv
.venv/bin/pip install -e ../fg-agent-id -e ".[dev]"

# add the MCP server extra if you are working on the stdio server
.venv/bin/pip install -e ../fg-agent-id -e ".[dev,mcp]"
```

Requires Python 3.11 or newer.

## Tests

The suite is fast, deterministic, and runs with no model and no external
backend:

```bash
.venv/bin/python -m pytest -q
```

Please keep it green and add tests with every change. In particular:

- The shared **port-contract suite** is the definition of conformance — new
  `RecordStore` / `SearchIndex` implementations must pass it.
- The **golden vectors** in `spec/vectors.json` pin the wire format. If a change
  legitimately alters the wire format, regenerate them with
  `.venv/bin/python spec/generate_vectors.py` and update `spec/SPEC.md` in the
  same change. Where prose and vectors disagree, the vectors win.
- New behavior on the porcelain `Memory` API should come with an example-style
  test, in the spirit of `tests/test_ghost_memory_demo.py`.

## Style and lint

Formatting and linting are handled by [ruff](https://docs.astral.sh/ruff/),
configured in `pyproject.toml` (line length 100, rule sets `E`, `F`, `I`, `UP`,
`B`, target `py311`):

```bash
.venv/bin/ruff check .
.venv/bin/ruff format .
```

Beyond the linter:

- Prefer many small, focused files over few large ones; organize by feature.
- Favor immutable data and pure functions — the lifecycle state machine is
  deliberately a set of pure functions.
- Public types are exported through `src/fg_agent_memory/__init__.py`; keep the
  `py.typed` marker honest with accurate annotations.

## Commits and pull requests

- Follow [Conventional Commits](https://www.conventionalcommits.org/):
  `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `perf:`, `ci:`.
- Keep commits small and focused; write a clear body explaining the *why*.
- **Do not include AI/assistant co-author attribution or `Co-Authored-By`
  trailers naming an assistant** in commit messages.
- Make sure `pytest` and `ruff check` pass before opening a pull request, and
  describe how you verified the change.

## Licensing of contributions

This project is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).
By contributing you agree that your contributions are licensed under the same
terms.
