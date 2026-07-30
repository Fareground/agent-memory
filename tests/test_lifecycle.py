"""Lifecycle machine: every legal and illegal transition, plus invariants."""

from itertools import product

import pytest

from fg_agent_memory import (
    LEGAL_TRANSITIONS,
    LifecycleError,
    RecordState,
    RecordType,
    archive,
    assert_transition,
    begin_transition,
    is_legal,
    refresh_rule_flag,
    resolve_transition,
    supersede,
)

from .conftest import make_record

EXPECTED_LEGAL = {
    (RecordState.ACTIVE, RecordState.SUPERSEDED),
    (RecordState.ACTIVE, RecordState.TRANSITIONAL),
    (RecordState.TRANSITIONAL, RecordState.ACTIVE),
    (RecordState.TRANSITIONAL, RecordState.SUPERSEDED),
    (RecordState.ACTIVE, RecordState.ARCHIVED),
    (RecordState.SUPERSEDED, RecordState.ARCHIVED),
    (RecordState.TRANSITIONAL, RecordState.ARCHIVED),
}


@pytest.mark.parametrize("frm,to", list(product(RecordState, RecordState)))
def test_legality_table_is_complete(frm, to):
    """Every (from, to) pair is decided; anything outside the table raises."""
    assert is_legal(frm, to) == ((frm, to) in EXPECTED_LEGAL)
    if (frm, to) in EXPECTED_LEGAL:
        assert_transition(frm, to)
    else:
        with pytest.raises(LifecycleError, match="illegal"):
            assert_transition(frm, to)


def test_legal_transitions_constant_matches():
    assert set(LEGAL_TRANSITIONS) == EXPECTED_LEGAL


def test_supersede_happy_path(provenance):
    old = make_record(provenance, body="Sandro lives in Boston.")
    new = make_record(provenance, body="Sandro lives in Austin.", supersedes=old.id)
    retired = supersede(old, new, reason="user moved")
    assert retired.state is RecordState.SUPERSEDED
    assert retired.superseded_by == new.id
    assert retired.state_reason == "user moved"
    assert retired.id == old.id  # still addressable under the same name


def test_supersede_requires_back_reference(provenance):
    old = make_record(provenance, body="Old fact.")
    unlinked = make_record(provenance, body="New fact.")  # no supersedes
    with pytest.raises(LifecycleError, match="reference"):
        supersede(old, unlinked, reason="drift")


def test_supersede_requires_reason(provenance):
    old = make_record(provenance, body="Old fact.")
    new = make_record(provenance, body="New fact.", supersedes=old.id)
    with pytest.raises(LifecycleError, match="reason"):
        supersede(old, new, reason="")


def test_supersede_requires_active_successor(provenance):
    old = make_record(provenance, body="Old fact.")
    new = make_record(provenance, body="New fact.", supersedes=old.id)
    archived_new = archive(new, reason="decayed")
    with pytest.raises(LifecycleError, match="active"):
        supersede(old, archived_new, reason="drift")


def test_supersede_archived_record_is_illegal(provenance):
    old = archive(make_record(provenance, body="Old fact."), reason="decay")
    new = make_record(provenance, body="New fact.", supersedes=old.id)
    with pytest.raises(LifecycleError, match="illegal"):
        supersede(old, new, reason="drift")


def test_begin_transition(provenance):
    a = make_record(provenance, body="The deploy is safe.")
    b = make_record(provenance, body="The deploy is risky.")
    a2, b2 = begin_transition(a, b, reason="contradicting observations")
    for side in (a2, b2):
        assert side.state is RecordState.TRANSITIONAL
        assert side.transition is not None
        assert set(side.transition.sides) == {a.id, b.id}
    assert a2.transition == b2.transition


def test_begin_transition_rejects_self_and_no_reason(provenance):
    a = make_record(provenance, body="One thing.")
    b = make_record(provenance, body="Another thing.")
    with pytest.raises(LifecycleError, match="itself"):
        begin_transition(a, a, reason="huh")
    with pytest.raises(LifecycleError, match="reason"):
        begin_transition(a, b, reason="")


def test_begin_transition_from_archived_is_illegal(provenance):
    a = archive(make_record(provenance, body="Gone."), reason="decay")
    b = make_record(provenance, body="Here.")
    with pytest.raises(LifecycleError, match="illegal"):
        begin_transition(a, b, reason="conflict")


def test_resolve_transition(provenance):
    a = make_record(provenance, body="The deploy is safe.")
    b = make_record(provenance, body="The deploy is risky.")
    a2, b2 = begin_transition(a, b, reason="conflict")
    winner, loser = resolve_transition(a2, b2, reason="postmortem confirmed safe")
    assert winner.state is RecordState.ACTIVE
    assert winner.transition is None
    assert loser.state is RecordState.SUPERSEDED
    assert loser.superseded_by == winner.id
    assert loser.transition is None


def test_resolve_requires_matching_pair(provenance):
    a = make_record(provenance, body="A thing.")
    b = make_record(provenance, body="B thing.")
    c = make_record(provenance, body="C thing.")
    a2, b2 = begin_transition(a, b, reason="conflict one")
    c2, _ = begin_transition(c, b, reason="conflict two")
    with pytest.raises(LifecycleError, match="same transition"):
        resolve_transition(a2, c2, reason="mismatched")


def test_resolve_requires_transitional_records(provenance):
    a = make_record(provenance, body="A thing.")
    b = make_record(provenance, body="B thing.")
    with pytest.raises(LifecycleError, match="illegal"):
        resolve_transition(a, b, reason="never began")


def test_archive_from_every_non_terminal_state(provenance):
    active = make_record(provenance, body="Active one.")
    assert archive(active, reason="decay").state is RecordState.ARCHIVED

    old = make_record(provenance, body="Old two.")
    new = make_record(provenance, body="New two.", supersedes=old.id)
    superseded = supersede(old, new, reason="replaced")
    assert archive(superseded, reason="decay").state is RecordState.ARCHIVED

    a, b = begin_transition(
        make_record(provenance, body="Side a."),
        make_record(provenance, body="Side b."),
        reason="conflict",
    )
    archived = archive(a, reason="decay")
    assert archived.state is RecordState.ARCHIVED
    assert archived.transition is None


def test_archived_is_terminal(provenance):
    archived = archive(make_record(provenance, body="Done."), reason="decay")
    with pytest.raises(LifecycleError, match="illegal"):
        archive(archived, reason="again")


def test_archive_requires_reason(provenance):
    with pytest.raises(LifecycleError, match="reason"):
        archive(make_record(provenance, body="No reason."), reason="")


def _rule_with_evidence(provenance):
    e1 = make_record(provenance, body="Episode one happened.", type=RecordType.EPISODE)
    e2 = make_record(provenance, body="Episode two happened.", type=RecordType.EPISODE)
    rule = make_record(
        provenance,
        body="Always do the thing.",
        type=RecordType.RULE,
        evidence=(e1.id, e2.id),
    )
    return rule, e1, e2


def test_rule_flagged_when_all_evidence_invalidated(provenance):
    rule, e1, e2 = _rule_with_evidence(provenance)
    dead1 = archive(e1, reason="decay")
    dead2 = archive(e2, reason="decay")
    flagged = refresh_rule_flag(rule, {e1.id: dead1, e2.id: dead2})
    assert flagged.flagged is True
    assert flagged.flag_reason


def test_rule_not_flagged_with_live_evidence(provenance):
    rule, e1, e2 = _rule_with_evidence(provenance)
    dead1 = archive(e1, reason="decay")
    same = refresh_rule_flag(rule, {e1.id: dead1, e2.id: e2})
    assert same.flagged is False
    assert same is rule  # no-op returns the record unchanged


def test_rule_flag_clears_when_evidence_revives(provenance):
    rule, e1, e2 = _rule_with_evidence(provenance)
    dead = {e1.id: archive(e1, reason="d"), e2.id: archive(e2, reason="d")}
    flagged = refresh_rule_flag(rule, dead)
    cleared = refresh_rule_flag(flagged, {e1.id: e1, e2.id: dead[e2.id]})
    assert cleared.flagged is False
    assert cleared.flag_reason is None


def test_rule_flag_rejects_partial_evidence(provenance):
    rule, e1, _ = _rule_with_evidence(provenance)
    with pytest.raises(LifecycleError, match="not supplied"):
        refresh_rule_flag(rule, {e1.id: e1})


def test_rule_flag_rejects_non_rule(provenance):
    fact = make_record(provenance)
    with pytest.raises(LifecycleError, match="rule"):
        refresh_rule_flag(fact, {})


class TestSignatureInvalidationOnEvolve:
    """A lifecycle transition invalidates a record signature; evolved
    versions must never carry a stale signature that no longer verifies."""

    def test_signatures_are_cleared_across_the_full_lifecycle_chain(self) -> None:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
        from fg_agent_id import KeyPair, address_from_signing_key

        from fg_agent_memory import Provenance, RecordType
        from fg_agent_memory.lifecycle import (
            archive,
            begin_transition,
            resolve_transition,
        )
        from fg_agent_memory.records import MemoryRecord

        keys = KeyPair(
            signing_key=Ed25519PrivateKey.from_private_bytes(bytes(range(32, 64))),
            agreement_key=X25519PrivateKey.from_private_bytes(bytes(range(64, 96))),
        )
        address = address_from_signing_key(keys.public.signing)
        prov = Provenance(kind="conversation")
        a = MemoryRecord.create(RecordType.FACT, "X is true.", prov).sign(keys, address)
        b = MemoryRecord.create(RecordType.FACT, "X is not true.", prov).sign(keys, address)
        a.verify()
        b.verify()

        ta, tb = begin_transition(a, b, reason="contradiction")
        assert ta.signature == "" and ta.signed_by is None
        assert tb.signature == "" and tb.signed_by is None

        winner, loser = resolve_transition(tb, ta, reason="b wins")
        assert winner.signature == "" and loser.signature == ""

        gone = archive(loser, reason="decayed")
        assert gone.signature == "" and gone.signed_by is None
        # The signed birth versions still verify on their own.
        a.verify()
        b.verify()
