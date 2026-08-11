"""MCP server: tool surface over a Memory, plus the import guard.

Skips cleanly when the optional ``mcp`` extra is not installed — only the
import-guard test runs everywhere.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from fg_agent_memory import mcp_server
from fg_agent_memory.memory import Memory


def test_import_guard_when_mcp_is_absent(tmp_path: Path, monkeypatch) -> None:
    """The core package imports fine without the SDK; only building the
    server fails, with an actionable message. Runs everywhere."""
    import importlib.util

    monkeypatch.setattr(mcp_server, "FastMCP", None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match=r"fg-agent-memory\[mcp\]"):
        mcp_server.build_server(Memory(tmp_path / "memory"))


def test_import_guard_when_mcp_is_incompatible(tmp_path: Path, monkeypatch) -> None:
    """mcp installed but mcp.server.fastmcp gone (mcp 2.x): the message must
    say the version is wrong, not tell users to install the extra again."""
    import importlib.util

    fake_spec = object()
    monkeypatch.setattr(mcp_server, "FastMCP", None)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: fake_spec if name == "mcp" else None,
    )
    with pytest.raises(ImportError, match=r"mcp>=1\.0,<2"):
        mcp_server.build_server(Memory(tmp_path / "memory"))


@pytest.fixture
def server(tmp_path: Path):
    pytest.importorskip("mcp", reason="optional [mcp] extra not installed")
    return mcp_server.build_server(Memory(tmp_path / "memory"))


def _call(server, tool: str, args: dict):
    """Call a tool in-process and return its JSON payload."""
    contents = asyncio.run(server.call_tool(tool, args))
    assert len(contents) == 1 and contents[0].type == "text"
    return json.loads(contents[0].text)


def test_exposes_the_five_tools(server) -> None:
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {
        "remember",
        "recall",
        "consolidate",
        "status",
        "redact",
    }


def test_remember_recall_round_trip(server) -> None:
    stored = _call(server, "remember", {"text": "The MCP bridge works end to end."})
    assert stored["record_id"].startswith("mem:")
    assert stored["deduped"] is False

    again = _call(server, "remember", {"text": "The MCP bridge works end to end."})
    assert again["deduped"] is True

    recalled = _call(server, "recall", {"query": "does the MCP bridge work?"})
    assert "end to end" in recalled["prompt_block"]
    assert recalled["records"][0]["state"] == "active"


def test_consolidate_and_status(server) -> None:
    _call(server, "remember", {"text": "The gateway is rate limited."})
    _call(server, "remember", {"text": "The gateway is not rate limited."})
    report = _call(server, "consolidate", {})
    assert any(action["action"] == "transition" for action in report["actions"])
    status = _call(server, "status", {})
    assert status["open_contradictions"] == 1
