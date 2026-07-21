"""Dense-only ranking, the lexical confidence gate, evidence, and abstention."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .domain import Aspect, Profile, SourceSpan
from .embeddings import OnnxBgeEncoder, TextEncoder
from .index import ProfileIndex
from .lexical import LexicalIndex
from .normalization import canonical_tokens, normalize_text, token_intersection
from .projection import project_profile

_PROVENANCE_LABELS = {
    "member_confirmed": "אושר על ידי החבר/ה",
    "staff_verified": "אומת על ידי הצוות",
    "self_described": "תיאור עצמי",
    "salesforce_structured": "שדה מובנה ב-Salesforce",
    "demo_added": "נוסף בהדגמה זו",
}

_GUEST_ID_PREFIX = "003SYNG"

# Ceiling on runtime additions held in the in-memory overlay.  Locally this is
# unreachable — a demo adds one or two people.  It matters when the UI is exposed
# beyond the machine that started it, e.g. behind a tunnel, where every visitor
# writes into the same overlay: the cap bounds both memory and how far one visitor
# can push the corpus the next visitor searches.  `reset()` clears it.
_MAX_LIVE_ADDITIONS = 25


@dataclass(frozen=True, slots=True)
class SearchResult:
    profile_id: str
    name: str
    title: str
    organisation: str
    sector: str
    region: str
    winning_aspect: str
    winning_aspect_label: str
    evidence_span: str
    evidence_highlight: str | None
    evidence_field: str
    provenance: str
    confidence_tier: str
    semantic_only: bool
    match_mechanism: str
    concept_bridge: str | None
    dense_score: float
    # Retained as a diagnostic only.  Since the fusion leg was removed this value
    # never influences rank; it records what the lexical index saw.
    lexical_score: float

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for private in ("dense_score", "lexical_score"):
            payload.pop(private)
        return payload


@dataclass(frozen=True, slots=True)
class SearchResponse:
    status: str
    message: str
    query: str
    results: tuple[SearchResult, ...]
    meta: dict[str, Any]

    def public_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "query": self.query,
            "results": [result.public_dict() for result in self.results],
            "meta": self.meta,
        }


@dataclass(frozen=True, slots=True)
class _LiveState:
    """One immutable snapshot of everything a single search reads.

    The sealed base artifacts never change after load.  Adding a demo profile
    builds a *new* snapshot and swaps it in atomically, so a search that started
    on the previous snapshot finishes on a consistent view instead of mixing two
    corpus states mid-ranking.
    """

    aspects: tuple[Aspect, ...]
    aspect_vectors: np.ndarray
    profiles_by_id: dict[str, Profile]
    source_vectors: dict[str, np.ndarray]
    lexical: LexicalIndex
    added_profile_ids: tuple[str, ...]


class SearchEngine:
    """Importable POC search service over an immutable synthetic index.

    Runtime additions (`add_profile`) live in a swapped in-memory overlay and
    disappear on restart or on `reset()`; the hashed artifacts on disk stay
    authoritative.
    """

    def __init__(
        self,
        index: ProfileIndex,
        encoder: TextEncoder,
        *,
        max_live_additions: int = _MAX_LIVE_ADDITIONS,
    ) -> None:
        self.index = index
        self.encoder = encoder
        expected_model = index.manifest["embedding"]["model_id"]
        expected_dimension = int(index.manifest["embedding"]["dimension"])
        if encoder.model_id != expected_model or encoder.dimension != expected_dimension:
            raise ValueError("Query encoder identity does not match profile embeddings")
        self._concept_vectors = index.concept_vectors_by_text
        self._write_lock = threading.Lock()
        self.max_live_additions = int(max_live_additions)
        # The sealed corpus exactly as it loaded.  `add_profile` copies out of a
        # snapshot and never mutates one, so this stays a faithful "before any
        # visitor touched it" view for `reset()` to restore — no reload, no
        # re-encode, and the manifest is verified once at startup either way.
        self._base_state = _LiveState(
            aspects=index.aspects,
            aspect_vectors=index.aspect_vectors,
            profiles_by_id=index.profiles_by_id,
            source_vectors=index.sources_by_key,
            lexical=index.lexical,
            added_profile_ids=(),
        )
        self._state = self._base_state

    @classmethod
    def from_default(cls, root: Path | None = None, model_dir: Path | None = None) -> "SearchEngine":
        index = ProfileIndex.load(root)
        configured_model_dir = model_dir or index.root / index.manifest["embedding"]["runtime_model_dir"]
        return cls(
            index=index,
            encoder=OnnxBgeEncoder(
                configured_model_dir,
                max_length=int(index.manifest["embedding"].get("max_length", 512)),
                batch_size=int(index.manifest["embedding"].get("inference_batch_size", 32)),
                expected_artifact_hash=index.manifest["embedding"].get("runtime_artifact_sha256"),
            ),
        )

    @property
    def profiles(self) -> tuple[Profile, ...]:
        """Every profile a search can currently return, base corpus first."""

        return tuple(self._state.profiles_by_id.values())

    @property
    def added_profile_ids(self) -> tuple[str, ...]:
        return self._state.added_profile_ids

    def directory_snapshot(self) -> tuple[tuple[Profile, ...], frozenset[str]]:
        """Every visible profile, and which of them were added at runtime, from **one** state.

        `profiles` and `added_profile_ids` each re-read the live state, so a caller
        that reads both straddles any `add_profile` or `reset` landing in between
        and describes two different corpora in one payload.  The benign direction
        reports additions the payload does not contain; the harmful direction hands
        back a visitor's profile with no `added` flag, presenting invented text as
        part of the measured corpus.

        Unreachable while the UI is bound to `127.0.0.1` and driven by one person.
        Reachable as soon as the link is shared — which is exactly why `search()`
        pins a snapshot for the duration of a request, and this does the same.
        """

        state = self._state
        return tuple(state.profiles_by_id.values()), frozenset(state.added_profile_ids)

    def add_profile(self, record: Mapping[str, Any]) -> Profile:
        """Validate, embed and index one synthetic record at runtime.

        The record goes through exactly the fail-closed validation the shipped
        corpus went through (`_synthetic: true`, a `003SYN` id, typed fields), is
        projected into the same four aspects, and is embedded with the same local
        encoder.  Nothing is written to disk: the sealed artifacts remain
        authoritative and the addition vanishes on restart.  That is deliberate —
        a demo visitor should be able to test the system on a person *they*
        invented without acquiring the power to mutate the measured corpus.

        If the record carries no ``Id``, a guest id in the reserved
        ``003SYNG…`` range is assigned under the write lock.
        """

        prepared = dict(record)
        with self._write_lock:
            state = self._state
            if len(state.added_profile_ids) >= self.max_live_additions:
                raise ValueError(
                    f"נוספו כבר {self.max_live_additions} פרופילים בהדגמה הזו — "
                    "אפשר לאפס את ההדגמה ולהתחיל מחדש"
                )
            if not prepared.get("Id"):
                prepared["Id"] = self._next_guest_id(state)
            profile = Profile.from_salesforce(prepared)
            if profile.profile_id in state.profiles_by_id:
                raise ValueError(f"Profile id {profile.profile_id} is already indexed")
            # Aspects with no populated source field embed the empty string, which
            # produces a meaningless vector that can still win a ranking.  The
            # sealed corpus never hits this (every field is populated); a form
            # submission legitimately can, so empty aspects are dropped here.
            aspects = tuple(
                aspect for aspect in project_profile(profile) if aspect.embedding_text
            )
            if not aspects:
                raise ValueError("הפרופיל ריק — נדרש תוכן באחד משדות התיאור לפחות")
            aspect_vectors = self.encoder.encode([aspect.embedding_text for aspect in aspects])
            source_items = [
                (f"{aspect.key}:{source.field}", source.text)
                for aspect in aspects
                for source in aspect.sources
            ]
            source_vectors = self.encoder.encode([text for _, text in source_items])
            merged_sources = dict(state.source_vectors)
            merged_sources.update(
                {key: vector for (key, _), vector in zip(source_items, source_vectors, strict=True)}
            )
            profiles = dict(state.profiles_by_id)
            profiles[profile.profile_id] = profile
            combined_aspects = state.aspects + aspects
            self._state = _LiveState(
                aspects=combined_aspects,
                aspect_vectors=np.vstack([state.aspect_vectors, aspect_vectors]),
                profiles_by_id=profiles,
                source_vectors=merged_sources,
                lexical=LexicalIndex(
                    [aspect.lexical_text for aspect in combined_aspects],
                    self.index.gazetteer_aliases,
                ),
                added_profile_ids=state.added_profile_ids + (profile.profile_id,),
            )
        return profile

    def reset(self) -> int:
        """Drop every runtime addition and return to the sealed corpus.

        Restores the snapshot built at construction, so what a search ranks
        afterwards is byte-for-byte the measured corpus again — the same state a
        restart would give, without paying the encoder load a second time.
        Returns how many additions were discarded.

        This exists because the overlay is process-global: when the UI is reached
        by more than one person, one visitor's invented profile is in the index
        everyone else searches.  That is the right behaviour for a shared demo and
        the wrong thing to leave lying around before a session that matters.
        """

        with self._write_lock:
            discarded = len(self._state.added_profile_ids)
            self._state = self._base_state
        return discarded

    @staticmethod
    def _next_guest_id(state: _LiveState) -> str:
        highest = 0
        for profile_id in state.profiles_by_id:
            if profile_id.startswith(_GUEST_ID_PREFIX):
                suffix = profile_id[len(_GUEST_ID_PREFIX):]
                if suffix.isdigit():
                    highest = max(highest, int(suffix))
        return f"{_GUEST_ID_PREFIX}{highest + 1:08d}"

    def search(
        self,
        query: str,
        *,
        filters: Mapping[str, str] | None = None,
        allowed_profile_ids: Iterable[str] | None = None,
        top_k: int = 5,
    ) -> SearchResponse:
        query = normalize_text(query)
        if not query:
            raise ValueError("Search query cannot be empty")
        if len(query) > 200:
            raise ValueError("Search query is limited to 200 characters")
        top_k = max(1, min(int(top_k), 5))

        # One snapshot for the whole request: an add_profile swap that lands
        # mid-search must not change what this search is ranking.
        state = self._state

        candidate_indices = self._candidate_indices(state, filters or {}, allowed_profile_ids)
        if not candidate_indices:
            return self._empty_response(query, "אין פרופילים זמינים תחת המסננים שנבחרו")

        concept_matches = self.index.concepts.match(query)
        expanded_queries: list[str] = [query]
        variant_concepts: list[Any | None] = [None]
        seen_variants = {normalize_text(query)}
        for concept_match in concept_matches:
            for expansion in concept_match.expansions:
                normalized_expansion = normalize_text(expansion)
                if normalized_expansion in seen_variants:
                    continue
                seen_variants.add(normalized_expansion)
                expanded_queries.append(expansion)
                variant_concepts.append(concept_match)
        original_query_vector = self.encoder.encode([query])[0]
        expansion_vectors = [
            self._concept_vectors[normalize_text(expansion)]
            for expansion in expanded_queries[1:]
        ]
        query_vectors = np.vstack([original_query_vector, *expansion_vectors])
        score_matrix = state.aspect_vectors[candidate_indices] @ query_vectors.T
        best_query_variant = np.argmax(score_matrix, axis=1)
        dense_scores = np.max(score_matrix, axis=1)
        dense_by_index = {index: float(score) for index, score in zip(candidate_indices, dense_scores, strict=True)}
        query_vector_by_index = {
            index: query_vectors[int(variant)]
            for index, variant in zip(candidate_indices, best_query_variant, strict=True)
        }
        winning_concept_by_index = {
            index: variant_concepts[int(variant)]
            for index, variant in zip(candidate_indices, best_query_variant, strict=True)
        }
        lexical_by_index = state.lexical.scores(query, candidate_indices)
        strong_lexical_indices = state.lexical.strong_match_indices(query, candidate_indices)

        # Ranking is dense-only.  A reciprocal-rank fusion over a BM25 leg used to sit
        # here behind three tuned weights.  An ablation across the whole golden set
        # measured its contribution as exactly zero: 8/8 acceptance, 7/7 abstentions
        # and 7/9 held-out with it and without it.  On the held-out set it did not add
        # a correct answer, it merely swapped which one it got -- winning
        # ``visual_accessibility`` while losing ``youth_employment``, a query the bare
        # model already ranked first.  Part A.2's own staging rule says anything that
        # cannot show a measured margin does not ship, so it does not.
        #
        # Deleting it also removes a defect class structurally rather than by patch:
        # the fused ranking and the dense ranking could pick different aspects of the
        # same profile, letting a weaker span inherit a stronger sibling's label.  With
        # one ranking there is nothing left to disagree.
        #
        # The lexical index stays.  ``strong_match_indices`` still does real work below
        # as a gate-opener -- it is what rescues an exact acronym match from abstention
        # when the dense score alone would refuse -- and its scores remain a diagnostic.
        # It simply no longer ranks.
        best_dense_by_profile: dict[str, int] = {}
        for index in candidate_indices:
            profile_id = state.aspects[index].profile_id
            incumbent = best_dense_by_profile.get(profile_id)
            if incumbent is None or (dense_by_index[index], state.aspects[index].key) > (
                dense_by_index[incumbent],
                state.aspects[incumbent].key,
            ):
                best_dense_by_profile[profile_id] = index
        dense_profile_ranking = sorted(
            best_dense_by_profile.values(),
            key=lambda index: (-dense_by_index[index], state.aspects[index].profile_id),
        )
        # One ranking now serves gate, tier, score, evidence and concept bridge.
        profile_ranking = dense_profile_ranking

        confidence = self.index.manifest["confidence"]
        threshold = float(confidence["dense_threshold"])
        result_floor = float(confidence.get("result_floor", threshold))
        # Concept expansion lifts the best score of a *covered* query by roughly
        # +0.09..+0.13, and never lifts an uncovered one.  Calibrating the gate on
        # expanded positives alone therefore sets a bar that ordinary queries cannot
        # clear; the threshold below is calibrated against out-of-domain negatives
        # instead, which are never expanded.  See manifest["confidence"].
        relative_margin = float(confidence.get("result_relative_margin", 1.0))
        # Deliberately the best dense score anywhere in the corpus, not the best
        # displayed one.  This gate answers "is there any strong semantic match at
        # all", so suppressing it because a weaker aspect happened to be displayed
        # would abstain on a query the model actually answered.  The per-result floor
        # below asks a different question -- "is *this* result good" -- and must use
        # the score of the aspect being shown.
        top_dense = dense_by_index[dense_profile_ranking[0]]
        dense_gate = top_dense >= threshold
        lexical_gate = bool(strong_lexical_indices)
        if not dense_gate and not lexical_gate:
            return self._empty_response(query, "לא נמצאה התאמה חזקה", concept_matches)

        # Padding guard.  An absolute floor alone cannot do this job: when a concept
        # fires, its expansion phrases lift every loosely related profile at once, so
        # a fixed floor lets four near-miss people ride along behind one good match.
        # Requiring results to stay within a margin of the best result scales with
        # whatever the query actually achieved.
        #
        # Everything below is keyed on the aspect in ``profile_ranking``, which is also
        # the aspect whose score, concept bridge and evidence are reported.  Scoring a
        # result against a *different* aspect of the same profile would let an unrelated
        # span inherit a strong label from elsewhere in the record.  That used to be a
        # live hazard -- the fused and dense rankings disagreed for 19 profile/query
        # pairs on this corpus, by up to +0.067 -- and it is now structurally impossible
        # rather than merely guarded, because with fusion removed there is only one
        # ranking.  The ceiling is taken over the same quantity, so the comparison stays
        # like with like.
        #
        # The margin is applied per intent, not per query.  A query can carry more
        # than one intent ("informal education and circular economy"), and those
        # intents legitimately peak at different heights; measuring the weaker one
        # against the stronger one's ceiling deletes a correct answer.  Each result is
        # therefore compared against the best score achieved by the same query variant
        # that won it.
        intent_by_index: dict[int, str | None] = {}
        intent_ceiling: dict[str | None, float] = {}
        for candidate in profile_ranking:
            concept = winning_concept_by_index[candidate]
            intent = concept.concept_id if concept is not None else None
            intent_by_index[candidate] = intent
            intent_ceiling[intent] = max(intent_ceiling.get(intent, -1.0), dense_by_index[candidate])

        results: list[SearchResult] = []
        for index in profile_ranking:
            if len(results) >= top_k:
                break
            profile_dense = dense_by_index[index]
            ceiling = intent_ceiling[intent_by_index[index]]
            display_floor = max(result_floor, ceiling - relative_margin)
            if profile_dense < display_floor and index not in strong_lexical_indices:
                continue
            ranked_aspect = state.aspects[index]
            profile = state.profiles_by_id[ranked_aspect.profile_id]
            winning_concept = winning_concept_by_index[index]
            aspect, evidence = self._select_result_evidence(
                state,
                ranked_aspect,
                query_vector_by_index[index],
                winning_concept.expansions if winning_concept is not None else (),
            )
            overlap = token_intersection(query, evidence.text)
            highlight = self._concept_highlight(
                evidence,
                winning_concept.expansions if winning_concept is not None else (),
            ) or self._surface_highlight(evidence.text, overlap)
            # "Strong" has to mean something on a corpus this small.  Cosine scores
            # here compress into roughly 0.31–0.71, so an absolute bar alone moves
            # with the threshold and starts calling near misses strong: after the
            # recalibration, two clearly unrelated profiles cleared it on the
            # flagship query.  A strong result must also be near the best result for
            # its own intent.
            near_best = profile_dense >= ceiling - relative_margin / 2
            tier = (
                "חזקה"
                if (profile_dense >= threshold + 0.05 and near_best)
                or index in strong_lexical_indices
                else "אפשרית"
            )
            if winning_concept is not None:
                mechanism = "מילון מושגים + וקטור סמנטי"
            elif index in strong_lexical_indices:
                mechanism = "התאמה מילולית נדירה + וקטור סמנטי"
            else:
                mechanism = "וקטור סמנטי + חיפוש מילולי"
            results.append(
                SearchResult(
                    profile_id=profile.profile_id,
                    name=profile.display_name,
                    title=profile.title,
                    organisation=profile.organisation,
                    sector=profile.sector,
                    region=profile.region,
                    winning_aspect=aspect.kind,
                    winning_aspect_label=aspect.label_he,
                    evidence_span=evidence.text,
                    evidence_highlight=highlight,
                    evidence_field=evidence.field,
                    provenance=_PROVENANCE_LABELS.get(evidence.provenance, "מקור פרופיל"),
                    confidence_tier=tier,
                    semantic_only=not overlap,
                    match_mechanism=mechanism,
                    concept_bridge=winning_concept.bridge_label if winning_concept is not None else None,
                    dense_score=dense_by_index[index],
                    lexical_score=lexical_by_index.get(index, 0.0),
                )
            )

        if not results:
            return self._empty_response(query, "לא נמצאה התאמה חזקה", concept_matches)
        # The headline has to agree with the cards underneath it.  Announcing "strong
        # matches" above a column of "אפשרית" tiers reads as a bug to anyone watching
        # the demo, and on the flagship query that is exactly what happened.
        message = (
            "נמצאו התאמות חזקות"
            if any(result.confidence_tier == "חזקה" for result in results)
            else "נמצאו התאמות אפשריות"
        )
        return SearchResponse(
            status="ok",
            message=message,
            query=query,
            results=tuple(results),
            meta=self._response_meta(concept_matches),
        )

    def _candidate_indices(
        self,
        state: _LiveState,
        filters: Mapping[str, str],
        allowed_profile_ids: Iterable[str] | None,
    ) -> list[int]:
        allowed = set(allowed_profile_ids) if allowed_profile_ids is not None else None
        # ``cohort`` and ``availability`` filters existed here for a while with no
        # caller anywhere -- not the web UI, not the evaluators, not the tests.  Public
        # API with no consumer is a maintenance liability, so they were cut rather than
        # left undecided; the fields they filtered on still exist in the fixtures and
        # can be re-exposed the day a caller actually needs them.
        supported_filters = {
            "sector": "sector",
            "region": "region",
        }
        unknown = set(filters).difference(supported_filters)
        if unknown:
            raise ValueError(f"Unsupported filters: {', '.join(sorted(unknown))}")

        indices: list[int] = []
        for index, aspect in enumerate(state.aspects):
            profile = state.profiles_by_id[aspect.profile_id]
            if allowed is not None and profile.profile_id not in allowed:
                continue
            rejected = False
            for filter_name, expected in filters.items():
                if expected in (None, ""):
                    continue
                attribute = supported_filters[filter_name]
                if str(getattr(profile, attribute)) != str(expected):
                    rejected = True
                    break
            if not rejected:
                indices.append(index)
        return indices

    def _select_evidence(self, state: _LiveState, aspect: Aspect, query_vector: np.ndarray) -> SourceSpan:
        candidates: list[tuple[float, SourceSpan]] = []
        for source in aspect.sources:
            key = f"{aspect.key}:{source.field}"
            vector = state.source_vectors.get(key)
            if vector is not None:
                candidates.append((float(vector @ query_vector), source))
        if candidates:
            candidates.sort(key=lambda pair: (-pair[0], pair[1].field))
            return candidates[0][1]
        if not aspect.sources:
            raise RuntimeError(f"Aspect {aspect.key} has no evidence source")
        return aspect.sources[0]

    def _select_result_evidence(
        self,
        state: _LiveState,
        ranked_aspect: Aspect,
        query_vector: np.ndarray,
        concept_expansions: Sequence[str],
    ) -> tuple[Aspect, SourceSpan]:
        """Choose a faithful evidence span without feeding it back into ranking.

        When a transparent concept bridge fired, prefer a profile span that
        directly contains one of that bridge's phrases.  This makes the displayed
        explanation inspectable (for example ``תנועות נוער``), while the profile
        rank, confidence gate, and score remain untouched.  Expansion order is the
        staff-authored evidence priority; otherwise source-vector similarity is the
        fallback.
        """

        direct: list[tuple[tuple[int, float, int, int], Aspect, SourceSpan]] = []
        for aspect in state.aspects:
            if aspect.profile_id != ranked_aspect.profile_id:
                continue
            for source in aspect.sources:
                source_normalized = normalize_text(source.text)
                source_tokens = set(canonical_tokens(source.text))
                for expansion_order, expansion in enumerate(concept_expansions):
                    expansion_normalized = normalize_text(expansion)
                    expansion_tokens = set(canonical_tokens(expansion))
                    if not expansion_tokens:
                        continue
                    overlap = source_tokens.intersection(expansion_tokens)
                    exact = int(expansion_normalized in source_normalized)
                    coverage = len(overlap) / len(expansion_tokens)
                    if exact or coverage == 1.0:
                        priority = (exact, coverage, -expansion_order, len(overlap))
                        direct.append((priority, aspect, source))
        if direct:
            direct.sort(key=lambda item: (item[0], item[1].key, item[2].field), reverse=True)
            _, aspect, source = direct[0]
            return aspect, source
        return ranked_aspect, self._select_evidence(state, ranked_aspect, query_vector)

    @staticmethod
    def _surface_highlight(evidence: str, overlap: frozenset[str]) -> str | None:
        if not overlap:
            return None
        for token in canonical_tokens(evidence):
            if token in overlap:
                return token
        return None

    @staticmethod
    def _concept_highlight(evidence: SourceSpan, concept_expansions: Sequence[str]) -> str | None:
        """Highlight the concept phrase that bridged the query to this span.

        This replaces an earlier lookup into ``golden_queries.json``.  That version
        produced the right phrase on the flagship query, but it produced it by
        reading the evaluation set's own answer — and because the flagship is
        deliberately zero-overlap, the surface fallback could never have found it.
        The highlight was therefore an artefact of the test fixture rather than of
        the search.  The staff-owned concept vocabulary is a legitimate source for
        the same phrase: it is what actually bridged the query, it is reviewable in
        ``config/concepts.json``, and it works for any query, judged or not.

        The returned phrase is always a literal substring of ``evidence.text``; the
        UI highlights by ``indexOf``, so a normalized-but-absent phrase would render
        as no highlight at all.
        """

        for expansion in concept_expansions:
            if expansion in evidence.text:
                return expansion
            # Fall back to a normalization-tolerant scan for spans that differ only
            # in niqqud, quote form or spacing.  normalize_text drops and collapses
            # characters, so offsets cannot be mapped directly; a short bounded
            # window is searched instead.
            normalized_expansion = normalize_text(expansion)
            if not normalized_expansion or normalized_expansion not in normalize_text(evidence.text):
                continue
            width = len(normalized_expansion)
            for start in range(len(evidence.text)):
                for end in range(start + width, min(len(evidence.text), start + width + 8) + 1):
                    candidate = evidence.text[start:end]
                    if normalize_text(candidate) == normalized_expansion:
                        return candidate
        return None

    def _response_meta(self, concept_matches: Sequence[Any] = ()) -> dict[str, Any]:
        return {
            "encoder_mode": "local_onnx",
            "model": self.index.manifest["embedding"]["model_id"],
            "synthetic_only": True,
            "generated_explanations": False,
            "applied_concepts": [match.concept_id for match in concept_matches],
        }

    def _empty_response(self, query: str, message: str, concept_matches: Sequence[Any] = ()) -> SearchResponse:
        return SearchResponse(
            status="no_strong_match",
            message=message,
            query=query,
            results=(),
            meta=self._response_meta(concept_matches),
        )
