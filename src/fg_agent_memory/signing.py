"""Domain-separated signing input for memory artifacts.

Identical construction to fg-agent-id's signing input, under this standard's
own domain, so a signature produced over a memory record can never be replayed
as an identity artifact (or vice versa) — even when the payloads canonicalize
to the same bytes:

    uint16be(len(tag)) || tag || canonical_json(payload)

where ``tag = "fg-agent-memory/v1/<context>"`` encoded as UTF-8. Canonical
JSON, base64 signature decoding, and key handling are imported from
fg-agent-id — this module defines only the memory-specific domain and
contexts. Never reuse a context across artifact types, and never sign a bare
payload.
"""

from __future__ import annotations

import base64
from typing import Any

from fg_agent_id import KeyPair, PublicKeys, canonical_json, signing_key_from_address

# Internal-API dependency: decode_signature is not re-exported at the
# fg_agent_id package root; the fg-agent-id version bound in pyproject pins it.
from fg_agent_id.signing import decode_signature

DOMAIN = "fg-agent-memory/v1"

# One constant per signed artifact type. These strings are wire-visible.
CONTEXT_RECORD = "record"
CONTEXT_MEMORY_FILE = "memory-file"

_MAX_TAG_BYTES = 0xFFFF


def domain_tag(context: str) -> bytes:
    """The full UTF-8 domain tag for a context (``<DOMAIN>/<context>``)."""
    if not context:
        raise ValueError("signing context must be a non-empty string")
    tag = f"{DOMAIN}/{context}".encode()
    if len(tag) > _MAX_TAG_BYTES:
        raise ValueError("signing context tag is too long")
    return tag


def signing_input(context: str, payload: Any) -> bytes:
    """The exact bytes to sign or verify for ``payload`` under ``context``."""
    tag = domain_tag(context)
    return len(tag).to_bytes(2, "big") + tag + canonical_json(payload)


def sign_payload(keys: KeyPair, context: str, payload: Any) -> str:
    """Sign ``payload`` under ``context``; returns a base64 signature."""
    return base64.b64encode(keys.sign(signing_input(context, payload))).decode()


def verify_payload(public: PublicKeys, context: str, payload: Any, signature: str) -> None:
    """Verify a base64 signature over ``payload`` under ``context``."""
    public.verify(decode_signature(signature), signing_input(context, payload))


def verify_by_address(address: str, context: str, payload: Any, signature: str) -> None:
    """Verify against the signing key an AMP address self-certifies."""
    public = PublicKeys(signing=signing_key_from_address(address), agreement=_NO_AGREEMENT)
    verify_payload(public, context, payload, signature)


# Signature verification only ever touches the Ed25519 half; this placeholder
# keeps PublicKeys' shape without implying a usable X25519 key.
_NO_AGREEMENT = b"\x00" * 32
