"""The README ghost-memory demo, executed end-to-end as a test.

Porcelain API only — no internal imports. If this test needs anything but
``from fg_agent_memory import Memory, RecordState``, the two-call promise is broken.
The snippet here must stay in lockstep with README.md's ghost-memory block.
"""

from __future__ import annotations

from pathlib import Path

from fg_agent_memory import Memory, RecordState


def test_readme_ghost_memory_snippet(tmp_path: Path) -> None:
    memory = Memory(tmp_path / "memory")

    # Two assertions that cannot both be true.
    a = memory.remember("The staging database is Postgres.")
    b = memory.remember("The staging database is not Postgres.")

    # Consolidation detects the contradiction. Neither side is dropped.
    memory.consolidate()
    disputed = memory.recall("staging database")

    # Both sides are visible, both labeled as disputed.
    assert set(disputed.ids) == {a.record_id, b.record_id}
    block = disputed.as_prompt_block()
    assert "The staging database is Postgres." in block
    assert "The staging database is not Postgres." in block
    assert block.count("[disputed:") == 2

    # You name the winner, with a reason that goes on the record.
    memory.resolve(a.record_id, b.record_id, "checked infra: staging moved off Postgres in March")

    # Default recall returns only the current fact, ranked first.
    current = memory.recall("staging database")
    assert current.ids[0] == b.record_id
    assert a.record_id not in current.ids
    assert "[disputed:" not in current.as_prompt_block()

    # The loser is superseded, not gone — history is one flag away.
    history = memory.recall("staging database", include_history=True)
    assert history.ids[0] == b.record_id
    assert a.record_id in history.ids
    assert "[superseded — kept for history]" in history.as_prompt_block()

    # Nothing was deleted: every version of the loser is still addressable,
    # and its trail records the whole dispute.
    versions = memory.store.history(a.record_id)
    assert [v.state.value for v in versions] == ["active", "transitional", "superseded"]
    assert versions[-1].superseded_by == b.record_id
    assert versions[-1].state_reason == "checked infra: staging moved off Postgres in March"


def test_auto_consolidation_surfaces_the_dispute_without_calling_consolidate(
    tmp_path: Path,
) -> None:
    """A two-call user never types the word 'consolidate': with the default
    cadence, the contradiction still becomes visible on its own."""
    memory = Memory(tmp_path / "memory", auto_consolidate_every=2)
    a = memory.remember("The deploy pipeline is green.")
    b = memory.remember("The deploy pipeline is not green.")
    assert b.consolidation is not None  # cadence hit: consolidation ran inside remember

    block = memory.recall("deploy pipeline").as_prompt_block()
    assert block.count("[disputed:") == 2
    assert {a.record_id, b.record_id} == set(memory.recall("deploy pipeline").ids)


# ------------------------------------------------- polarity-over-shared-core


def _dispute_detected(memory: Memory, query: str, a, b) -> bool:
    disputed = memory.recall(query, include_history=True)
    block = disputed.as_prompt_block()
    return {a.record_id, b.record_id} <= set(disputed.ids) and block.count("[disputed:") >= 2


def test_morphological_variant_negation_is_detected(tmp_path: Path) -> None:
    """'requires' vs 'does not require' — the inflection change must not
    defeat the default detector."""
    memory = Memory(tmp_path / "memory")
    a = memory.remember("The backend requires a restart after edits.")
    b = memory.remember("The backend does not require a restart after edits.")
    memory.consolidate()
    assert _dispute_detected(memory, "backend restart", a, b)


def test_no_x_needed_polarity_frame_is_detected(tmp_path: Path) -> None:
    """'no restart needed' vs 'requires a restart' — a polarity frame over
    the same core claim is a contradiction candidate."""
    memory = Memory(tmp_path / "memory")
    a = memory.remember("The operator backend hot-reloads; no restart needed.")
    b = memory.remember("The operator backend requires a restart after edits.")
    memory.consolidate()
    assert _dispute_detected(memory, "operator backend restart", a, b)


def test_unrelated_statements_do_not_pair(tmp_path: Path) -> None:
    memory = Memory(tmp_path / "memory")
    memory.remember("The deploy target is us-east-1.")
    memory.remember("The staging password rotates weekly.")
    memory.consolidate()
    block = memory.recall("deploy staging", include_history=True).as_prompt_block()
    assert "[disputed:" not in block


def test_same_polarity_difference_is_not_a_contradiction(tmp_path: Path) -> None:
    """'likes coffee' vs 'likes tea' — no negation marker, no dispute."""
    memory = Memory(tmp_path / "memory")
    memory.remember("Sandro likes coffee.")
    memory.remember("Sandro likes tea.")
    memory.consolidate()
    block = memory.recall("Sandro likes", include_history=True).as_prompt_block()
    assert "[disputed:" not in block


def test_value_swap_without_negation_is_detected(tmp_path: Path) -> None:
    """The antonym gap: 'X is A' vs 'X is B' carries no negator, so polarity
    detection alone misses it. Inferred subject→value slots close it."""
    memory = Memory(tmp_path / "memory", auto_consolidate=False)
    first = memory.remember("The staging database is Postgres.")
    second = memory.remember("The staging database is MySQL.")
    memory.consolidate()
    a = memory.store.get(first.record_id)
    b = memory.store.get(second.record_id)
    assert a.state is RecordState.TRANSITIONAL
    assert b.state is RecordState.TRANSITIONAL
    rendered = memory.recall("what is the staging database?").as_prompt_block()
    assert "Postgres" in rendered and "MySQL" in rendered
    assert "disputed" in rendered
