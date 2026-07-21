"""Small exact BM25 leg used alongside dense retrieval in the POC."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from .normalization import canonical_tokens, clitic_variants, lexical_tokens, normalize_text


def _is_latin(term: str) -> bool:
    return any("a" <= character <= "z" for character in term)


class LexicalIndex:
    """In-memory BM25 whose statistics are computed inside the allowed relation."""

    def __init__(self, documents: Sequence[str], gazetteer: Mapping[str, Iterable[str]] | None = None) -> None:
        self._gazetteer = gazetteer or {}
        self._documents = tuple(Counter(lexical_tokens(text, self._gazetteer)) for text in documents)
        self._lengths = tuple(sum(document.values()) for document in self._documents)

    def scores(self, query: str, candidate_indices: Sequence[int]) -> dict[int, float]:
        query_terms = Counter(lexical_tokens(query, self._gazetteer))
        if not query_terms or not candidate_indices:
            return {}

        candidates = tuple(dict.fromkeys(candidate_indices))
        average_length = sum(self._lengths[index] for index in candidates) / max(len(candidates), 1)
        average_length = average_length or 1.0
        document_frequency = {
            term: sum(1 for index in candidates if term in self._documents[index])
            for term in query_terms
        }

        k1 = 1.2
        b = 0.75
        scores: dict[int, float] = {}
        for index in candidates:
            document = self._documents[index]
            length = self._lengths[index]
            score = 0.0
            for term, query_frequency in query_terms.items():
                frequency = document.get(term, 0)
                if not frequency:
                    continue
                df = document_frequency[term]
                inverse_document_frequency = math.log(1.0 + (len(candidates) - df + 0.5) / (df + 0.5))
                denominator = frequency + k1 * (1.0 - b + b * length / average_length)
                score += inverse_document_frequency * (frequency * (k1 + 1.0) / denominator) * query_frequency
            if score > 0:
                scores[index] = score
        return scores

    # A lexical hit only overrides the dense confidence gate when it is close to a
    # whole-query match: most of the query's surface words are present, and at
    # least one of them is genuinely discriminative in this corpus.
    MIN_QUERY_COVERAGE = 0.6
    DISCRIMINATIVE_DF_RATIO = 0.08

    def strong_match_indices(self, query: str, candidate_indices: Sequence[int]) -> frozenset[int]:
        """Find whole-query lexical evidence strong enough to override the dense gate.

        This path exists for entity and acronym queries — ``NOVA Skills``,
        ``מתנ"ס``, ``ולמתנדבים`` — where the dense leg alone is unreliable but the
        surface string is unambiguous.  It is deliberately *not* a general
        keyword-matching path.

        Rarity alone cannot decide this.  In a corpus this small almost every term
        looks rare (``נוער`` and ``nova`` both appear in one document), so an
        earlier document-frequency rule let ordinary sentences such as
        ``כניסה של בני נוער לשוק העבודה`` fire the gate on two incidental words and
        return confident, unrelated people.  Coverage is the signal that actually
        separates "the query *is* this entity" from "the query happens to share a
        word": an entity query matches nearly all of its own tokens, a
        natural-language query matches a small fraction.

        Statistics are computed only inside the already-allowed candidate relation.
        """

        candidates = tuple(dict.fromkeys(candidate_indices))
        surface_terms = tuple(
            dict.fromkeys(term for term in canonical_tokens(query) if len(term) >= 2)
        )
        if not surface_terms or not candidates:
            return frozenset()

        # Each surface word is realised either by one of its own clitic forms, or by a
        # gazetteer expansion the whole query triggered.  An expansion counts only when
        # *all* of its tokens are present: ``מתנ"ס`` expands to ``מרכז קהילתי``, and a
        # profile reading ``סיוע קהילתי`` shares one generic word with that phrase
        # without being a community centre.  Matching per token let the shorter,
        # weaker document win on BM25 length normalisation.
        own_variants = {term: set(clitic_variants(term)) for term in surface_terms}
        expansion_groups: dict[str, list[list[set[str]]]] = {term: [] for term in surface_terms}
        normalized_query = normalize_text(query)
        for phrase, expansions in self._gazetteer.items():
            if normalize_text(phrase) not in normalized_query:
                continue
            phrase_terms = [term for term in canonical_tokens(phrase) if term in expansion_groups]
            if not phrase_terms:
                continue
            for expansion in expansions:
                group = [set(clitic_variants(token)) for token in canonical_tokens(str(expansion))]
                if not group:
                    continue
                for term in phrase_terms:
                    expansion_groups[term].append(group)

        def covers(term: str, document) -> bool:
            if own_variants[term] & document.keys():
                return True
            return any(
                all(token_variants & document.keys() for token_variants in group)
                for group in expansion_groups[term]
            )

        document_frequency = {
            term: sum(1 for index in candidates if covers(term, self._documents[index]))
            for term in surface_terms
        }
        discriminative_limit = max(1, int(len(candidates) * self.DISCRIMINATIVE_DF_RATIO))

        strong: set[int] = set()
        for index in candidates:
            document = self._documents[index]
            covered = [term for term in surface_terms if covers(term, document)]
            if not covered:
                continue
            coverage = len(covered) / len(surface_terms)
            discriminative = any(document_frequency[term] <= discriminative_limit for term in covered)
            # The gate overrides a *dense* abstention, so it must earn that override:
            # something about the query has to be beyond what the embedding handles
            # well.  A plain Hebrew word that the dense leg already scores fine (say
            # ``קהילה``) does not qualify, or the gate becomes a keyword search that
            # silently outranks the semantic one.
            rescued = any(
                _is_latin(term)
                or term not in document  # matched only through a clitic or gazetteer variant
                for term in covered
            )
            if coverage >= self.MIN_QUERY_COVERAGE and discriminative and rescued:
                strong.add(index)
        return frozenset(strong)
