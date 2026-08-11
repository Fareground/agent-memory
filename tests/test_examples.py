"""The examples/ directory runs as part of the suite so it cannot rot.

- quickstart.py is the README hello world, executed for real.
- llm_operator.py has its full Proposal-emitting plumbing exercised with a
  fake model; only the provider call is a stub.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from fg_agent_memory import (
    ConsolidationOperators,
    Memory,
    MemoryRecord,
    ProposalKind,
    Provenance,
    RecordType,
    TrigramNearDupOperator,
)

_EXAMPLES = Path(__file__).parent.parent / "examples"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


quickstart = _load("quickstart")
llm_operator = _load("llm_operator")


def _record(body: str) -> MemoryRecord:
    return MemoryRecord.create(
        RecordType.FACT, body, Provenance(kind="conversation")
    )


class TestQuickstart:
    def test_readme_hello_world_runs(self, tmp_path):
        block = quickstart.main(str(tmp_path / "memory"))
        assert "Zed" in block


class TestLLMOperatorPlumbing:
    def test_stub_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            llm_operator.call_llm("any prompt")

    def test_contradiction_verdict_emits_transition_proposal(self):
        def fake_llm(prompt: str) -> str:
            assert "Postgres" in prompt  # both bodies reach the model
            return '{"contradiction": true, "reason": "cannot both be true"}'

        op = llm_operator.LLMContradictionOperator(fake_llm)
        a = _record("The staging database is Postgres.")
        b = _record("The staging database is MySQL.")
        proposal = op.propose_contradiction(a, b)
        assert proposal is not None
        assert proposal.kind is ProposalKind.TRANSITION
        assert proposal.target_id == a.id
        assert proposal.other_id == b.id
        assert proposal.reason == "cannot both be true"

    def test_no_contradiction_abstains(self):
        op = llm_operator.LLMContradictionOperator(
            lambda _: '{"contradiction": false}'
        )
        assert op.propose_contradiction(_record("A."), _record("B.")) is None

    @pytest.mark.parametrize(
        "response",
        [
            "not json at all",
            "[1, 2, 3]",
            '{"contradiction": "yes"}',
            '{"contradiction": true}',  # missing reason
            '{"contradiction": true, "reason": "   "}',  # blank reason
        ],
    )
    def test_malformed_response_abstains(self, response):
        op = llm_operator.LLMContradictionOperator(lambda _: response)
        assert op.propose_contradiction(_record("A."), _record("B.")) is None

    def test_model_failure_abstains(self):
        def broken(_: str) -> str:
            raise ConnectionError("provider down")

        op = llm_operator.LLMContradictionOperator(broken)
        assert op.propose_contradiction(_record("A."), _record("B.")) is None

    def test_default_model_is_the_stub_and_fails_closed(self):
        op = llm_operator.LLMContradictionOperator()
        assert op.propose_contradiction(_record("A."), _record("B.")) is None

    def test_end_to_end_through_consolidation(self, tmp_path):
        """The operator's proposal survives pipeline validation: both sides
        of the dispute surface in recall, neither is dropped."""
        operators = ConsolidationOperators(
            near_dup=TrigramNearDupOperator(threshold=0.9),
            contradiction=llm_operator.LLMContradictionOperator(
                lambda _: '{"contradiction": true, "reason": "model says so"}'
            ),
        )
        memory = Memory(tmp_path / "memory", operators=operators)
        memory.remember("The staging environment feels stable.")
        memory.remember("The staging environment keeps crashing.")
        memory.consolidate()
        block = memory.recall("staging environment").as_prompt_block()
        assert "disputed" in block
