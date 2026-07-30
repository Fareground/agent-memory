"""Deterministic, model-free text normalization shared by the pipeline.

Everything here is stdlib-only and reproducible: light suffix stemming,
content-word extraction with polarity detection, and coarse subject→value
slot inference for "X is Y"-shaped sentences. Detection quality beyond these
heuristics belongs to LLM operators on the pipeline's ports — this module is
the floor every implementation can stand on with zero model dependency.
"""

from __future__ import annotations

import re

# Tokens that flip the polarity of a sentence. A side containing any of
# these is treated as negative-polarity for contradiction detection.
NEGATORS = frozenset(
    {
        "not",
        "no",
        "never",
        "n't",
        "none",
        "neither",
        "nor",
        "without",
        "isnt",
        "arent",
        "dont",
        "doesnt",
        "didnt",
        "cant",
        "cannot",
        "wont",
        "wasnt",
        "werent",
    }
)

# Function words carrying no content: ignored when computing the shared
# core of a candidate pair.
STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "done",
        "has",
        "have",
        "had",
        "having",
        "will",
        "would",
        "can",
        "could",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "from",
        "with",
        "and",
        "or",
        "but",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "as",
        "so",
        "if",
        "then",
        "than",
        "after",
        "before",
        "when",
        "any",
        "some",
        "still",
        "now",
        "also",
        "just",
    }
)

# Polarity-frame stems: "requires X" / "no X needed" / "X is necessary"
# express the same modal claim; the modal word itself is part of the
# polarity frame, not of the content core. Stems (post-stem()).
POLARITY_FRAME_STEMS = frozenset({"need", "requir", "necessari", "necessar"})

# Only copulas split a sentence into a subject slot and a value: an
# equative claim ("X is Y") is exclusive in a way preference/usage verbs
# ("likes coffee" and "likes tea" can both be true) never are.
_RELATION = re.compile(r"\b(is|are|was|were)\b", re.IGNORECASE)


def stem(token: str) -> str:
    """Light deterministic suffix stemming (s/es/ed/ing + trailing 'e'),
    stdlib only. Just enough to fold morphological variants like
    requires/require/required onto one stem."""
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    if token.endswith("e") and len(token) >= 4:
        token = token[:-1]
    return token


def content_terms(body: str) -> tuple[frozenset[str], bool]:
    """Normalize a body into (stemmed content-word set, negative-polarity).

    Lowercases, splits contractions, strips punctuation, drops negators,
    stopwords, and polarity-frame modals, and stems what remains."""
    lowered = body.lower().replace("n't", " not ")
    cleaned = "".join(ch if ch.isalnum() else " " for ch in lowered)
    tokens = cleaned.split()
    negative = any(token in NEGATORS for token in tokens)
    stems = {
        stem(token)
        for token in tokens
        if token not in NEGATORS and token not in STOPWORDS
    }
    return frozenset(stems - POLARITY_FRAME_STEMS), negative


def normalize_phrase(phrase: str) -> str:
    """A phrase reduced to its in-order stemmed content words — the stable
    spelling used for inferred slot keys and values."""
    cleaned = "".join(ch if ch.isalnum() else " " for ch in phrase.lower())
    stems = [
        stem(token)
        for token in cleaned.split()
        if token not in STOPWORDS and token not in NEGATORS
    ]
    return " ".join(s for s in stems if s not in POLARITY_FRAME_STEMS)


def infer_slots(body: str) -> dict[str, str]:
    """A coarse subject→value slot for an "X is Y"-shaped sentence.

    This is what lets the default contradiction pass catch value swaps that
    carry no negation marker ("staging DB is Postgres" vs "staging DB is
    MySQL"): both normalize to the same slot key with different values.
    Negated sentences are left slotless — polarity detection owns them — and
    anything that doesn't split cleanly into a non-empty subject and value
    yields no slot at all. Coarse by design; a smarter extractor operator
    can always emit richer slots."""
    lowered = body.lower().replace("n't", " not ")
    if any(token in NEGATORS for token in re.findall(r"[a-z0-9]+", lowered)):
        return {}
    match = _RELATION.search(body)
    if match is None:
        return {}
    subject = normalize_phrase(body[: match.start()])
    raw_value = body[match.end() :]
    value = normalize_phrase(raw_value)
    # A bare one-word subject ("Sandro is tired" / "Sandro is hungry") names
    # a person or thing with many simultaneous properties, not an exclusive
    # attribute — inferring a slot there manufactures false disputes. Two or
    # more content words ("staging database", "Sandro's favorite editor")
    # name a specific attribute whose value genuinely competes.
    if not value or len(subject.split()) < 2:
        return {}
    # Only identity-like values (proper nouns, versions, numbers — anything
    # carrying an uppercase letter or digit) compete for a slot. A plain
    # descriptive value ("my favorite editor is fast") coexists with an
    # identity ("…is Vim"); slotting it would manufacture a false dispute.
    # Adjective-state swaps ("open" vs "closed") are deliberately left to an
    # LLM operator — precision beats recall when a false positive blocks
    # recall of both records.
    if not re.search(r"[A-Z0-9]", raw_value):
        return {}
    return {subject: value}
