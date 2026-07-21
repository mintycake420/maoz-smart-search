"""Conservative Hebrew normalization for retrieval and regression checks."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping

NORMALIZATION_VERSION = "hebrew-nfkc-v1"

_TOKEN_RE = re.compile(r"[\u0590-\u05FF]+|[A-Za-z]+(?:[-._/][A-Za-z0-9]+)*|\d+", re.UNICODE)
_HEBREW_RE = re.compile(r"^[\u0590-\u05FF]+$")
_NIQQUD_RE = re.compile(r"[\u0591-\u05C7]")
_PREFIX_CLITICS = frozenset("ובלכמשה")
_QUOTE_TRANSLATION = str.maketrans({
    "׳": "'",
    "״": '"',
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "־": "-",
    "–": "-",
    "—": "-",
})


def normalize_text(value: str) -> str:
    """Normalize compatibility forms and punctuation without transliterating text."""

    value = unicodedata.normalize("NFKC", value).translate(_QUOTE_TRANSLATION)
    value = _NIQQUD_RE.sub("", value)
    return " ".join(value.casefold().split())


def canonical_tokens(value: str) -> tuple[str, ...]:
    """Return surface tokens used for honest no-lexical-overlap assertions."""

    return tuple(_TOKEN_RE.findall(normalize_text(value)))


def clitic_variants(token: str) -> tuple[str, ...]:
    """Add, never replace, cautious single-prefix Hebrew variants.

    Hebrew prefix removal is ambiguous.  Keeping the original token avoids turning
    this pragmatic POC fallback into a claim of morphological analysis.
    """

    variants = [token]
    candidate = token
    if _HEBREW_RE.fullmatch(token):
        # Two prefixes are enough for the explicit POC robustness probe (e.g.
        # ולמתנדבים).  Going further increases false stems quickly.
        for _ in range(2):
            if len(candidate) < 4 or candidate[0] not in _PREFIX_CLITICS:
                break
            candidate = candidate[1:]
            variants.append(candidate)
    return tuple(dict.fromkeys(variants))


def lexical_tokens(value: str, gazetteer: Mapping[str, Iterable[str]] | None = None) -> tuple[str, ...]:
    tokens: list[str] = []
    for token in canonical_tokens(value):
        tokens.extend(clitic_variants(token))

    if gazetteer:
        normalized = normalize_text(value)
        for phrase, expansions in gazetteer.items():
            if normalize_text(phrase) in normalized:
                for expansion in expansions:
                    for token in canonical_tokens(str(expansion)):
                        tokens.extend(clitic_variants(token))

    return tuple(tokens)


def token_intersection(left: str, right: str) -> frozenset[str]:
    return frozenset(canonical_tokens(left)).intersection(canonical_tokens(right))
