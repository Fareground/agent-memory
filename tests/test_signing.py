"""Record and file signatures: domain separation, verification, tamper."""

import pytest
from fg_agent_id import SignatureError
from fg_agent_id.signing import signing_input as id_signing_input

from fg_agent_memory import (
    CONTEXT_MEMORY_FILE,
    CONTEXT_RECORD,
    DOMAIN,
    MemoryFile,
    RecordError,
    domain_tag,
    signing_input,
)

from .conftest import FIXED_TIME, make_record


def test_domain_tag_is_memory_specific():
    tag = domain_tag(CONTEXT_RECORD)
    assert tag == b"fg-agent-memory/v1/record"
    assert DOMAIN == "fg-agent-memory/v1"


def test_signing_input_differs_from_agent_id_domain():
    payload = {"a": 1}
    assert signing_input("record", payload) != id_signing_input("record", payload)


def test_signing_input_differs_across_contexts():
    payload = {"a": 1}
    assert signing_input(CONTEXT_RECORD, payload) != signing_input(
        CONTEXT_MEMORY_FILE, payload
    )


def test_record_sign_and_verify(keys, address, provenance):
    signed = make_record(provenance).sign(keys, address)
    assert signed.signed_by == address
    signed.verify()  # does not raise


def test_record_signature_keeps_content_address(keys, address, provenance):
    record = make_record(provenance)
    assert record.sign(keys, address).id == record.id


def test_unsigned_record_verify_rejected(provenance):
    with pytest.raises(RecordError, match="not signed"):
        make_record(provenance).verify()


def test_tampered_record_fails_verification(keys, address, provenance):
    signed = make_record(provenance).sign(keys, address)
    tampered = signed.model_copy(update={"confidence": 0.01})
    with pytest.raises(SignatureError):
        tampered.verify()


def test_signature_not_reattributable(keys, address, provenance):
    other = "amp:key:" + "1" * 44
    signed = make_record(provenance).sign(keys, address)
    stolen = signed.model_copy(update={"signed_by": other})
    with pytest.raises(Exception):  # bad address or signature failure  # noqa: B017
        stolen.verify()


def test_memory_file_sign_and_verify(keys, address, provenance):
    mf = MemoryFile.create(
        (make_record(provenance).sign(keys, address),),
        agent=address,
        created_at=FIXED_TIME,
    ).sign(keys)
    mf.verify()
    mf.records[0].verify()  # record signatures stay independently verifiable


def test_memory_file_requires_agent_to_sign(keys, provenance):
    mf = MemoryFile.create((make_record(provenance),), created_at=FIXED_TIME)
    with pytest.raises(RecordError, match="agent"):
        mf.sign(keys)


def test_tampered_file_fails_verification(keys, address, provenance):
    record = make_record(provenance)
    mf = MemoryFile.create((record,), agent=address, created_at=FIXED_TIME).sign(keys)
    tampered = mf.model_copy(update={"records": ()})
    with pytest.raises(SignatureError):
        tampered.verify()


def test_record_signature_not_valid_as_file_signature(keys, address, provenance):
    """Domain separation: a record signature must not verify under memory-file."""
    record = make_record(provenance).sign(keys, address)
    mf = MemoryFile.create((), agent=address, created_at=FIXED_TIME)
    forged = mf.model_copy(update={"signature": record.signature})
    with pytest.raises(SignatureError):
        forged.verify()
