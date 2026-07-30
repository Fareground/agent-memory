"""HeuristicExtractor sanity: the default model-free write-stage operator."""

from fg_agent_memory import (
    HeuristicExtractor,
    InMemoryRecordStore,
    ProposalKind,
    Provenance,
    RecordType,
    apply_proposals,
)


def extract(text: str) -> tuple:
    return HeuristicExtractor().propose(text, Provenance(kind="conversation"))


def test_extracts_facts_and_episodes():
    proposals = extract(
        "Sandro lives in Austin. We deployed the new build yesterday. Ok."
    )
    assert [p.kind for p in proposals] == [ProposalKind.CREATE, ProposalKind.CREATE]
    types = [p.record.type for p in proposals]
    assert types == [RecordType.FACT, RecordType.EPISODE]
    assert proposals[0].record.body == "Sandro lives in Austin."


def test_never_proposes_rules():
    proposals = extract(
        "Always review the diff before merging. You should never skip tests."
    )
    assert all(p.record.type is not RecordType.RULE for p in proposals)


def test_ignores_noise():
    assert extract("Hmm. Ok! Wow?") == ()
    assert extract("") == ()


def test_confidence_and_tags_applied():
    extractor = HeuristicExtractor(confidence=0.3, tags=("auto",))
    (proposal,) = extractor.propose(
        "The backend uses SQLite.", Provenance(kind="observation")
    )
    assert proposal.record.confidence == 0.3
    assert proposal.record.tags == ("auto",)
    assert proposal.record.provenance.kind == "observation"


def test_extractor_output_applies_cleanly():
    store = InMemoryRecordStore()
    proposals = extract("The venv lives in backend. We ran the suite twice.")
    written = apply_proposals(store, proposals)
    assert len(written) == len(proposals) == 2
    assert {record.id for record in store.list()} == {r.id for r in written}


class TestInferredSlots:
    def test_copula_facts_carry_a_subject_value_slot(self) -> None:
        from fg_agent_memory.text import infer_slots

        assert infer_slots("The staging database is Postgres.") == {
            "stag databas": "postgr"
        }
        # Same subject, different value -> same key, competing values.
        assert list(infer_slots("The staging database is MySQL.")) == ["stag databas"]

    def test_non_exclusive_and_negated_shapes_stay_slotless(self) -> None:
        from fg_agent_memory.text import infer_slots

        assert infer_slots("Sandro likes coffee.") == {}  # not equative
        assert infer_slots("Sandro is tired.") == {}  # bare one-word subject
        assert infer_slots("The staging database is not Postgres.") == {}  # polarity owns it
        assert infer_slots("Ship it!") == {}
        # Descriptive (non-identity) values never slot: 'fast' coexists
        # with 'Vim', they are not competing values for the same attribute.
        assert infer_slots("My favorite editor is fast.") == {}
        assert list(infer_slots("My favorite editor is Vim.")) == ["my favorit editor"]
