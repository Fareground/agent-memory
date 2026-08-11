"""MCP stdio server: fg-agent-memory as tools any MCP client can call.

Exposes ``remember`` / ``recall`` / ``consolidate`` / ``status`` over one
:class:`~fg_agent_memory.Memory` rooted at ``--path``. The ``mcp`` SDK is an
optional extra — install with ``pip install fg-agent-memory[mcp]`` — and this
module import-guards it so the core package never needs it.

Run directly, or register with an MCP client:

    fg-agent-memory-mcp --path ~/agent-memory
"""

from __future__ import annotations

import argparse
import importlib.util
import warnings

from .memory import Memory

try:  # pragma: no cover — exercised via the import-guard test
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    FastMCP = None  # type: ignore[assignment]


def _missing_mcp_message() -> str:
    """Distinguish "mcp is not installed" from "mcp is installed but
    incompatible" (mcp 2.0 removed ``mcp.server.fastmcp``) so the error
    never tells users to install an extra they already have."""
    try:
        installed = importlib.util.find_spec("mcp") is not None
        compatible = installed and importlib.util.find_spec("mcp.server.fastmcp") is not None
    except (ImportError, ValueError):  # broken installation probes as absent
        installed = compatible = False
    if installed and not compatible:
        return (
            "the installed 'mcp' package is incompatible with this server "
            "(mcp.server.fastmcp is unavailable — removed in mcp 2.0); "
            "install a supported version with: pip install 'mcp>=1.0,<2'"
        )
    return (
        "the MCP server requires the optional 'mcp' dependency; "
        "install with: pip install fg-agent-memory[mcp]"
    )


def build_server(memory: Memory) -> FastMCP:
    """A FastMCP server wired to ``memory``. Raises ImportError without the
    ``mcp`` extra installed."""
    if FastMCP is None:
        raise ImportError(_missing_mcp_message())

    server = FastMCP("fg-agent-memory")

    @server.tool()
    def remember(text: str, tags: list[str] | None = None, source: str | None = None) -> dict:
        """Store something in memory. Duplicate content reinforces the
        existing memory instead of creating a copy."""
        result = memory.remember(text, tags=tuple(tags) if tags else None, source=source)
        return {"record_id": result.record_id, "deduped": result.deduped}

    @server.tool()
    def recall(
        query: str, budget_tokens: int = 2000, include_history: bool = False
    ) -> dict:
        """Retrieve memories relevant to a query, ranked and token-budgeted.
        Both sides of any unresolved contradiction are surfaced together."""
        result = memory.recall(
            query, budget_tokens=budget_tokens, include_history=include_history
        )
        return {
            "prompt_block": result.as_prompt_block(),
            "records": [
                {
                    "id": item.record.id,
                    "body": item.record.body,
                    "state": item.record.state.value,
                    "score": item.score,
                    "reasons": list(item.reasons),
                }
                for item in result.records
            ],
        }

    @server.tool()
    def consolidate() -> dict:
        """Run a consolidation pass: dedup, contradiction detection,
        episode-to-rule promotion. Nothing is ever deleted."""
        report = memory.consolidate()
        return {
            "actions": [
                {
                    "pass": action.pass_name,
                    "action": action.action,
                    "records": list(action.record_ids),
                    "reason": action.reason,
                }
                for action in report.actions
            ],
            "rejections": len(report.rejections),
        }

    @server.tool()
    def redact(record_id: str, reason: str) -> dict:
        """Destroy a memory's content (secrets/PII remembered by mistake).
        The record's existence, id, and links remain as an audited
        tombstone; the content is unrecoverable. This is permanent."""
        tomb = memory.redact(record_id, reason)
        return {"record_id": tomb.id, "redacted": True, "reason": reason}

    @server.tool()
    def status() -> dict:
        """Memory snapshot: record counts by state and type, and how many
        contradictions await resolution."""
        return memory.status()

    return server


def main(argv: list[str] | None = None) -> None:
    """Console entry point: ``fg-agent-memory-mcp --path <dir>``."""
    parser = argparse.ArgumentParser(description="fg-agent-memory MCP stdio server")
    parser.add_argument(
        "--path",
        default="./memory",
        help="directory the memory lives in (default: ./memory)",
    )
    args = parser.parse_args(argv)
    # mcp 1.x's pydantic-settings usage emits IncompleteFieldDefinitionWarning
    # noise on startup; it is upstream and harmless on a stdio server.
    warnings.filterwarnings(
        "ignore", message=".*", module="pydantic_settings.*"
    )
    server = build_server(Memory(args.path))
    server.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
