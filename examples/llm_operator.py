"""An LLM as a validated operator in the consolidation pipeline.

The core idea of this standard: the model never touches the store. An
operator sees a typed candidate pair, emits a typed ``Proposal``, and the
pipeline validates that proposal against the lifecycle machine before
anything is written — an invalid proposal is rejected, not repaired.

This example implements the ``ContradictionOperator`` port with an LLM
doing the detection. Everything around the model call is real and tested:
prompt construction, strict response parsing, fail-closed error handling,
and the transition ``Proposal`` the pipeline expects. Only the model call
itself is a stub — wire ``call_llm`` to any provider (Anthropic, OpenAI,
a local model) and the rest works unchanged.

Run the plumbing against a fake model:  see tests/test_examples.py.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from fg_agent_memory import MemoryRecord, Proposal, ProposalKind
from fg_agent_memory.pipeline.consolidate import ContradictionOperator

_PROMPT_TEMPLATE = """\
Two memory records may contradict each other. Decide whether they can both
be true at the same time.

Record A: {body_a}
Record B: {body_b}

Respond with ONLY a JSON object, no prose:
  {{"contradiction": true, "reason": "<one short sentence>"}}
or
  {{"contradiction": false}}
"""


def call_llm(prompt: str) -> str:
    """Send ``prompt`` to your model and return its text response.

    Deliberately unimplemented: this example is provider-agnostic. Wire it
    to your SDK of choice; the operator only needs the raw response string.
    """
    raise NotImplementedError(
        "call_llm is a stub — connect it to your LLM provider "
        "(e.g. return client.messages.create(...).content)"
    )


class LLMContradictionOperator(ContradictionOperator):
    """Contradiction detection delegated to a model, fail-closed.

    Abstains (returns None) on any model failure or malformed response —
    a flaky model must never manufacture a dispute. When the model does
    report a contradiction, the emitted proposal is exactly what the
    pipeline validates: a transition over the two record ids with the
    model's reason on the record.
    """

    def __init__(self, llm: Callable[[str], str] = call_llm) -> None:
        self._llm = llm

    def propose_contradiction(
        self, a: MemoryRecord, b: MemoryRecord
    ) -> Proposal | None:
        prompt = _PROMPT_TEMPLATE.format(body_a=a.body, body_b=b.body)
        try:
            response = self._llm(prompt)
        except Exception:
            return None  # fail closed: no model, no dispute
        verdict = self._parse(response)
        if verdict is None:
            return None
        contradiction, reason = verdict
        if not contradiction:
            return None
        return Proposal(
            kind=ProposalKind.TRANSITION,
            target_id=a.id,
            other_id=b.id,
            reason=reason,
        )

    @staticmethod
    def _parse(response: str) -> tuple[bool, str] | None:
        """Strictly parse the model response; anything malformed → abstain."""
        try:
            payload = json.loads(response.strip())
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        contradiction = payload.get("contradiction")
        if not isinstance(contradiction, bool):
            return None
        if not contradiction:
            return (False, "")
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return None  # a dispute without a reason is not actionable
        return (True, reason.strip())


# Wiring it into Memory — the near-dup pass keeps its default operator, the
# contradiction port becomes the model:
#
#     from fg_agent_memory import ConsolidationOperators, Memory
#
#     operators = ConsolidationOperators(contradiction=LLMContradictionOperator())
#     memory = Memory("./memory", operators=operators)
#     memory.consolidate()  # LLM verdicts, validated by the lifecycle machine
