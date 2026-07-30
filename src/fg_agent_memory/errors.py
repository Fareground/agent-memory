"""Error hierarchy for the agent-memory standard."""


class AgentMemoryError(Exception):
    """Base class for all agent-memory errors."""


class RecordError(AgentMemoryError):
    """Malformed memory record or memory file."""


class LifecycleError(AgentMemoryError):
    """Illegal lifecycle transition or violated lifecycle invariant."""


class StoreError(AgentMemoryError):
    """Record store contract violation (missing record, bad version)."""


class ProposalError(AgentMemoryError):
    """Operator proposal rejected by pipeline validation."""
