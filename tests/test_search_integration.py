from __future__ import annotations

import json
from pathlib import Path

import pytest

from maoz_search.normalization import canonical_tokens, token_intersection


FLAGSHIP_QUERY = "חינוך בלתי פורמלי"
FLAGSHIP_ID = "003SYN000000001"


def test_flagship_is_top_semantic_match_with_exact_evidence(engine) -> None:
    response = engine.search(FLAGSHIP_QUERY)
    assert response.status == "ok"
    assert response.results[0].profile_id == FLAGSHIP_ID
    assert response.results[0].winning_aspect == "trajectory"
    assert response.results[0].evidence_span == (
        "הקימה והובילה שלוש תנועות נוער אזוריות, הכשירה מדריכים "
        "ובנתה שותפויות עם רשויות מקומיות."
    )
    assert response.results[0].evidence_highlight == "תנועות נוער"
    assert response.results[0].semantic_only is True
    assert token_intersection(FLAGSHIP_QUERY, response.results[0].evidence_span) == frozenset()


def test_entire_flagship_record_has_no_query_token() -> None:
    records = json.loads(Path("data/synthetic_profiles.json").read_text(encoding="utf-8"))
    target = next(record for record in records if record["Id"] == FLAGSHIP_ID)
    raw = json.dumps(target, ensure_ascii=False)
    assert set(canonical_tokens(FLAGSHIP_QUERY)).isdisjoint(canonical_tokens(raw))


def test_no_match_abstains_without_padding(engine) -> None:
    response = engine.search("תכנון מנועי יונים לחלל עמוק")
    assert response.status == "no_strong_match"
    assert response.results == ()
    assert response.message == "לא נמצאה התאמה חזקה"


def test_public_contract_exposes_tiers_but_not_raw_scores(engine) -> None:
    payload = engine.search(FLAGSHIP_QUERY).public_dict()
    assert payload["results"][0]["confidence_tier"] in {"חזקה", "אפשרית"}
    assert "dense_score" not in payload["results"][0]
    assert "lexical_score" not in payload["results"][0]
    assert "fusion_score" not in payload["results"][0]


def test_filters_and_allowed_relation_are_applied_before_ranking(engine) -> None:
    target_profile = engine.index.profiles_by_id[FLAGSHIP_ID]
    included = engine.search(FLAGSHIP_QUERY, filters={"sector": target_profile.sector})
    assert included.results and included.results[0].profile_id == FLAGSHIP_ID

    excluded_ids = {profile.profile_id for profile in engine.index.profiles if profile.profile_id != FLAGSHIP_ID}
    excluded = engine.search(FLAGSHIP_QUERY, allowed_profile_ids=excluded_ids)
    assert all(result.profile_id != FLAGSHIP_ID for result in excluded.results)


def test_each_result_explains_the_concept_variant_that_scored_it(engine) -> None:
    response = engine.search("חינוך בלתי פורמלי וגם כלכלה מעגלית", top_k=5)
    by_profile = {result.profile_id: result for result in response.results}
    assert FLAGSHIP_ID in by_profile
    assert "003SYN000000005" in by_profile
    assert by_profile[FLAGSHIP_ID].concept_bridge.startswith("חינוך בלתי פורמלי ↔")
    assert by_profile["003SYN000000005"].concept_bridge.startswith("כלכלה מעגלית ↔")


@pytest.mark.parametrize(
    "query_id",
    [
        "test_participatory_local_government",
        "test_circular_economy",
        "test_emotional_recovery_after_crisis",
    ],
)
def test_authored_semantic_probe_finds_relevant_profile(engine, query_id) -> None:
    golden = next(item for item in engine.index.golden_queries if item["id"] == query_id)
    response = engine.search(golden["query"])
    returned = [result.profile_id for result in response.results[:3]]
    assert golden["expected_top_profile_id"] in returned


def test_evidence_highlight_does_not_depend_on_the_golden_set(engine) -> None:
    """The flagship highlight must come from the concept bridge, not the answer key.

    An earlier version read ``expected_evidence.highlight`` out of
    ``golden_queries.json``.  Because the flagship query is deliberately
    zero-overlap, the surface fallback could never produce that phrase, so the most
    visible element of the demo was supplied by the evaluation fixture.
    """

    with_golden = engine.search(FLAGSHIP_QUERY).results[0].evidence_highlight
    original = engine.index.golden_queries
    try:
        engine.index.golden_queries = ()
        without_golden = engine.search(FLAGSHIP_QUERY).results[0].evidence_highlight
    finally:
        engine.index.golden_queries = original
    assert with_golden == "תנועות נוער"
    assert without_golden == with_golden


def test_headline_message_agrees_with_result_tiers(engine) -> None:
    for query in (FLAGSHIP_QUERY, "כניסה של בני נוער לשוק העבודה", "NOVA Skills"):
        response = engine.search(query, top_k=5)
        if response.status != "ok":
            continue
        any_strong = any(result.confidence_tier == "חזקה" for result in response.results)
        assert response.message == ("נמצאו התאמות חזקות" if any_strong else "נמצאו התאמות אפשריות")


@pytest.mark.parametrize(
    "query",
    [
        "כניסה של בני נוער לשוק העבודה",
        "טכנולוגיה לשיפור המענה לתושב",
        "מניעת מחלות בקרב תושבים",
        "קהילה",
        "מתנדבים",
    ],
)
def test_lexical_gate_ignores_ordinary_hebrew_queries(engine, query) -> None:
    """The lexical gate rescues entities and acronyms, not ordinary sentences.

    A document-frequency rule used to treat almost every term as rare on a corpus
    this small, so two incidental words let a natural-language query override the
    dense confidence gate and return unrelated people labelled "strong".
    """

    candidates = list(range(len(engine.index.aspects)))
    assert engine.index.lexical.strong_match_indices(query, candidates) == frozenset()


def test_strong_tier_is_justified_by_the_result_own_score(engine) -> None:
    """A result labelled strong must clear the bar on its *own* score.

    Reciprocal-rank fusion and raw dense similarity pick different winning aspects for
    the same profile on 19 profile/query pairs in this corpus. Tiering a result on the
    profile's best *dense* aspect while reporting and displaying its best *fused*
    aspect let a weaker, unrelated span inherit a strong label from elsewhere in the
    same record, by up to +0.067.
    """

    threshold = float(engine.index.manifest["confidence"]["dense_threshold"])
    lexical_mechanism = "התאמה מילולית נדירה + וקטור סמנטי"
    for item in engine.index.golden_queries:
        response = engine.search(str(item["query"]), top_k=5)
        for result in response.results:
            if result.confidence_tier != "חזקה" or result.match_mechanism == lexical_mechanism:
                continue  # the lexical rescue path is entitled to a low dense score
            assert result.dense_score >= threshold + 0.05, (
                f"{item['id']}: {result.profile_id} labelled strong on "
                f"{result.dense_score:.4f}, below the {threshold + 0.05:.4f} bar"
            )


def test_evidence_is_a_verbatim_span_of_the_profile_shown(engine) -> None:
    """Evidence must be copied from the profile on screen, never assembled or borrowed."""

    for item in engine.index.golden_queries:
        response = engine.search(str(item["query"]), top_k=5)
        for result in response.results:
            spans = {
                source.text
                for aspect in engine.index.aspects
                if aspect.profile_id == result.profile_id
                for source in aspect.sources
            }
            assert result.evidence_span in spans, f"{item['id']}: {result.profile_id}"
            if result.evidence_highlight:
                # Case-insensitive because the surface-overlap path returns a casefolded
                # token (``nova`` for ``NOVA Skills``) and the UI highlights with a
                # case-insensitive indexOf. A highlight the UI cannot locate renders as
                # no highlight at all, so this is the contract that matters.
                assert (
                    result.evidence_highlight.casefold() in result.evidence_span.casefold()
                ), f"{item['id']}: {result.evidence_highlight!r} not locatable in evidence"


def test_confidence_gate_is_not_calibrated_on_concept_lift_only(engine) -> None:
    """At least one calibration positive must receive no concept expansion.

    Expansion lifts a covered query's best score by roughly +0.09..+0.13.  When
    every calibration positive was covered, the fitted threshold encoded that lift
    and ordinary queries were rejected at rank 1.
    """

    uncovered = engine.index.manifest["confidence"].get("calibration_uncovered_positive_ids")
    assert uncovered, "calibration must include a positive with no concept coverage"
    for query_id in uncovered:
        golden = next(item for item in engine.index.golden_queries if item["id"] == query_id)
        assert engine.index.concepts.match(golden["query"]) == ()


def test_held_out_queries_are_not_configured_in_the_concept_vocabulary(engine) -> None:
    """Guard the one honest generalisation signal in this repository.

    If a held-out intent is ever added to config/concepts.json it stops being
    held out, and the reported number silently becomes another configured result.
    """

    held_out = [item for item in engine.index.golden_queries if item.get("role") == "heldout"]
    assert held_out, "the held-out probe set must not be empty"
    for item in held_out:
        assert engine.index.concepts.match(item["query"]) == (), item["id"]


@pytest.mark.parametrize(
    "query_id",
    [
        "test_latin_entity_nova",
        "test_latin_acronym_ai",
        "test_hebrew_acronym_community_center",
        "test_prefix_clitics_volunteers",
    ],
)
def test_lexical_robustness_probe_finds_relevant_profile(engine, query_id) -> None:
    golden = next(item for item in engine.index.golden_queries if item["id"] == query_id)
    response = engine.search(golden["query"])
    assert response.results
    assert response.results[0].profile_id == golden["expected_top_profile_id"]
