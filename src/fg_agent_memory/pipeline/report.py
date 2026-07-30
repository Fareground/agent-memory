"""Report primitives shared by every pipeline stage.

The audit trail is a first-class output: every mutation a stage performs is
recorded as an :class:`Action` (with before/after states and a reason), and
every operator proposal the pipeline refused is recorded as a
:class:`Rejection`. A report is the complete answer to "what did this pass
do, and why" — nothing mutates the store without leaving a line here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Action:
    """One applied mutation: which pass did what to which records.

    ``before`` and ``after`` hold the lifecycle-state values of the records
    named in ``record_ids``, index-aligned, so the report alone reconstructs
    the state change without re-reading the store.
    """

    pass_name: str
    action: str
    record_ids: tuple[str, ...]
    before: tuple[str, ...]
    after: tuple[str, ...]
    reason: str


@dataclass(frozen=True, kw_only=True)
class Rejection:
    """One refused operator output: what was proposed and why it was
    rejected. Rejected proposals are never partially applied — the store is
    untouched by anything that lands here."""

    pass_name: str
    detail: str
    error: str
