"""Runtime profile addition: the demo flow that proves the system is not
answering from a dataset its authors prepared.

Each test builds a private ``SearchEngine`` over the shared index and encoder, so
additions never leak into the session-scoped engine other tests rank against.
"""

from __future__ import annotations

import pytest

from maoz_search import SearchEngine
from maoz_search.domain import ProfileValidationError
from maoz_search.web import create_app


def fresh_engine(engine) -> SearchEngine:
    return SearchEngine(engine.index, engine.encoder)


def guest_record(**overrides) -> dict:
    record = {
        "_synthetic": True,
        "FirstName": "יונתן",
        "LastName": "תפוחי",
        "Title": "סמנכ\"ל מיזוגים ורכישות",
        "Description": "",
        "Account": {"Id": "", "Name": "קרן צמיחה"},
        "Sector__c": "פיננסים",
        "Region__c": "מרכז",
        "Cohort__c": "",
        "Areas_of_Activity__c": "ליווי חברות פינטק; עסקאות חוצות גבולות",
        "Experience__c": "הוביל עשרות עסקאות מיזוג ורכישה בתחום הפינטק, ואלוף ארצי בבאולינג.",
        "Interests__c": "פינטק; באולינג תחרותי",
        "Values__c": "",
        "Affiliations__c": "",
        "Field_Provenance__c": {
            "Title": "demo_added",
            "Account.Name": "demo_added",
            "Sector__c": "demo_added",
            "Experience__c": "demo_added",
            "Areas_of_Activity__c": "demo_added",
            "Interests__c": "demo_added",
        },
    }
    record.update(overrides)
    return record


def test_added_profile_is_found_with_verbatim_evidence(engine) -> None:
    private = fresh_engine(engine)
    profile = private.add_profile(guest_record())
    assert profile.profile_id.startswith("003SYNG")

    response = private.search("מיזוגים ורכישות בפינטק")
    assert response.status == "ok"
    assert response.results[0].profile_id == profile.profile_id
    own_spans = {
        source.text
        for aspect in private._state.aspects
        if aspect.profile_id == profile.profile_id
        for source in aspect.sources
    }
    assert response.results[0].evidence_span in own_spans
    assert response.results[0].provenance == "נוסף בהדגמה זו"


def test_added_profile_does_not_mutate_the_sealed_base(engine) -> None:
    private = fresh_engine(engine)
    base_aspects = len(private.index.aspects)
    base_vector_rows = private.index.aspect_vectors.shape[0]
    private.add_profile(guest_record())

    # The sealed index object is untouched; only the live snapshot grew.
    assert len(private.index.aspects) == base_aspects
    assert private.index.aspect_vectors.shape[0] == base_vector_rows
    assert len(private._state.aspects) > base_aspects
    # And the session engine, which shares that index, never sees the guest.
    assert engine.added_profile_ids == ()


def test_added_profiles_get_distinct_generated_guest_ids(engine) -> None:
    private = fresh_engine(engine)
    first = private.add_profile(guest_record())
    second = private.add_profile(guest_record(FirstName="דנה", LastName="לביא"))
    assert first.profile_id != second.profile_id
    assert {first.profile_id, second.profile_id} == {"003SYNG00000001", "003SYNG00000002"}


def test_add_profile_rejects_a_non_synthetic_record(engine) -> None:
    private = fresh_engine(engine)
    with pytest.raises(ProfileValidationError, match="synthetic"):
        private.add_profile(guest_record(_synthetic=False))


def test_add_profile_rejects_an_empty_profile(engine) -> None:
    private = fresh_engine(engine)
    empty = guest_record(
        Title="",
        Description="",
        Account={"Id": "", "Name": ""},
        Sector__c="",
        Region__c="",
        Areas_of_Activity__c="",
        Experience__c="",
        Interests__c="",
    )
    with pytest.raises(ValueError, match="ריק"):
        private.add_profile(empty)


def test_sparse_profile_drops_empty_aspects_and_still_searches(engine) -> None:
    sparse = guest_record(
        Description="",
        Areas_of_Activity__c="",
        Experience__c="",
        Interests__c="",
        Values__c="",
        Affiliations__c="",
        Cohort__c="",
    )
    private = fresh_engine(engine)
    profile = private.add_profile(sparse)
    added = [aspect for aspect in private._state.aspects if aspect.profile_id == profile.profile_id]
    assert [aspect.kind for aspect in added] == ["role_org"]
    # No RuntimeError from an evidence-less aspect; the search simply runs.
    private.search("מיזוגים ורכישות")


def test_api_endpoint_adds_and_finds_a_person(engine) -> None:
    client = create_app(fresh_engine(engine)).test_client()
    created = client.post("/api/profiles", json={
        "first_name": "יונתן",
        "last_name": "תפוחי",
        "title": "סמנכ\"ל מיזוגים ורכישות",
        "organisation": "קרן צמיחה",
        "sector": "פיננסים",
        "region": "מרכז",
        "experience": "הוביל עשרות עסקאות מיזוג ורכישה בתחום הפינטק, ואלוף ארצי בבאולינג.",
        "areas_of_activity": "ליווי חברות פינטק",
        "interests": "פינטק; באולינג תחרותי",
    })
    assert created.status_code == 201
    assert created.json["name"] == "יונתן תפוחי"
    profile_id = created.json["profile_id"]

    found = client.post("/api/search", json={"query": "מיזוגים ורכישות בפינטק", "filters": {}})
    assert found.status_code == 200
    assert found.json["results"][0]["profile_id"] == profile_id

    meta = client.get("/api/meta").json
    assert "פיננסים" in meta["filters"]["sectors"]
    assert meta["added_profile_count"] == 1


def test_form_contact_details_are_stored_but_never_indexed(engine) -> None:
    """A form-added person carries contact details the search can never see."""

    private = fresh_engine(engine)
    client = create_app(private).test_client()
    created = client.post("/api/profiles", json={
        "first_name": "יונתן",
        "last_name": "תפוחי",
        "title": "סמנכ\"ל מיזוגים ורכישות",
        "email": "yonatan@appleseed.example",
        "phone": "050-000-0099",
        "experience": "הוביל עסקאות מיזוג ורכישה בתחום הפינטק.",
    })
    assert created.status_code == 201
    profile_id = created.json["profile_id"]

    row = next(
        profile for profile in client.get("/api/profiles").json["profiles"]
        if profile["profile_id"] == profile_id
    )
    assert row["email"] == "yonatan@appleseed.example"
    assert row["phone"] == "050-000-0099"

    indexed = " ".join(
        aspect.embedding_text + " " + aspect.lexical_text
        for aspect in private._state.aspects
        if aspect.profile_id == profile_id
    ).casefold()
    assert "appleseed.example" not in indexed
    assert "050-000-0099" not in indexed

    # Searching the address must not retrieve them: it was never encoded.
    by_email = client.post("/api/search", json={
        "query": "yonatan@appleseed.example", "filters": {},
    })
    assert by_email.status_code == 200
    assert by_email.json["status"] == "no_strong_match"


def test_a_new_sector_joins_the_facet_vocabulary(engine) -> None:
    """Sector is an open combobox, not a closed picklist.

    The form suggests the sectors already in the index and accepts anything
    else; a new value must then become selectable for everyone, because the
    search filter and the form's own suggestion list read this one payload.
    The browser renders a dropdown arrow that reads as a fixed ``<select>``, so
    this behaviour is easy to assume away — it is asserted here, not described.
    """

    client = create_app(fresh_engine(engine)).test_client()
    novel = "משפטים"
    before = client.get("/api/meta").json["filters"]["sectors"]
    assert novel not in before, "pick a sector the shipped corpus does not already use"

    created = client.post("/api/profiles", json={
        "first_name": "אורי",
        "last_name": "אברהם",
        "title": "יועץ משפטי",
        "sector": novel,
        "experience": "ליווה חקיקה בתחום הרגולציה הפיננסית והופיע בוועדות הכנסת.",
    })
    assert created.status_code == 201

    after = client.get("/api/meta").json["filters"]["sectors"]
    assert novel in after
    assert set(before) < set(after), "existing sectors must survive the addition"

    # And the new value works as a real facet, not just a label in a list.
    scoped = client.post("/api/search", json={
        "query": "ליווי חקיקה ורגולציה",
        "filters": {"sector": novel},
    })
    assert scoped.status_code == 200
    assert scoped.json["results"][0]["profile_id"] == created.json["profile_id"]


def test_directory_endpoint_lists_base_and_added_profiles(engine) -> None:
    client = create_app(fresh_engine(engine)).test_client()

    before = client.get("/api/profiles").json
    assert before["count"] == 18
    assert before["added_count"] == 0
    first = before["profiles"][0]
    assert {"profile_id", "name", "title", "organisation", "sector", "region",
            "experience", "areas_of_activity", "interests", "values",
            "affiliations", "description", "cohort", "added"} <= set(first)
    assert all(profile["added"] is False for profile in before["profiles"])

    created = client.post("/api/profiles", json={
        "first_name": "יונתן",
        "last_name": "תפוחי",
        "title": "סמנכ\"ל מיזוגים ורכישות",
        "experience": "הוביל עסקאות מיזוג ורכישה בתחום הפינטק.",
    })
    assert created.status_code == 201

    after = client.get("/api/profiles").json
    assert after["count"] == 19
    assert after["added_count"] == 1
    added_rows = [profile for profile in after["profiles"] if profile["added"]]
    assert [row["profile_id"] for row in added_rows] == [created.json["profile_id"]]
    assert added_rows[0]["experience"] == "הוביל עסקאות מיזוג ורכישה בתחום הפינטק."


def test_form_description_keeps_self_described_provenance(engine) -> None:
    """The free-form blurb maps to the Contact Description and stays dense-only.

    In the sealed corpus Description carries self_described provenance and is
    excluded from the lexical keyword-rescue leg — an anti-stuffing control, not
    a relevance judgment. A form submission must inherit exactly that split:
    embedded for semantic ranking, absent from the lexical projection.
    """

    private = fresh_engine(engine)
    client = create_app(private).test_client()
    blurb = "אוהב לחבר בין אנשים ורעיונות"
    created = client.post("/api/profiles", json={
        "first_name": "דנה",
        "last_name": "לביא",
        "title": "מנהלת שותפויות",
        "description": blurb,
    })
    assert created.status_code == 201
    trajectory = next(
        aspect for aspect in private._state.aspects
        if aspect.profile_id == created.json["profile_id"] and aspect.kind == "trajectory"
    )
    assert blurb in trajectory.embedding_text
    assert blurb not in trajectory.lexical_text
    description_span = next(span for span in trajectory.sources if span.field == "Description")
    assert description_span.provenance == "self_described"
    assert description_span.include_in_lexical is False


def test_api_endpoint_rejects_invalid_submissions(engine) -> None:
    client = create_app(fresh_engine(engine)).test_client()
    assert client.post("/api/profiles", data="not json", content_type="application/json").status_code == 400
    missing_name = client.post("/api/profiles", json={"first_name": "", "last_name": "כהן", "title": "מנהל"})
    assert missing_name.status_code == 400
    no_content = client.post("/api/profiles", json={"first_name": "דנה", "last_name": "כהן"})
    assert no_content.status_code == 400
    oversized = client.post("/api/profiles", json={
        "first_name": "דנה",
        "last_name": "כהן",
        "experience": "א" * 301,
    })
    assert oversized.status_code == 400


def test_reset_returns_the_index_to_the_sealed_corpus(engine) -> None:
    """The overlay is process-global, so one visitor's invention is in everyone's index.

    That is correct for a shared demo and wrong to leave standing before a session
    that matters, so the overlay has to be droppable without a restart — a restart
    costs a fresh 580 MB ONNX session load.
    """

    private = fresh_engine(engine)
    sealed_ids = {profile.profile_id for profile in private.profiles}
    profile = private.add_profile(guest_record())

    assert profile.profile_id in {p.profile_id for p in private.profiles}
    found = private.search("מיזוגים ורכישות בפינטק")
    assert found.results[0].profile_id == profile.profile_id

    assert private.reset() == 1
    assert private.added_profile_ids == ()
    assert {p.profile_id for p in private.profiles} == sealed_ids
    after = private.search("מיזוגים ורכישות בפינטק")
    assert all(result.profile_id != profile.profile_id for result in after.results)
    assert private.reset() == 0, "resetting an untouched index discards nothing"


def test_live_additions_are_capped_and_reset_clears_the_cap(engine) -> None:
    """Unbounded additions are only unreachable while the UI is on one machine.

    Reached over a shared link the add endpoint is an open, unauthenticated write
    that runs the encoder, so the overlay needs a ceiling. The cap is checked
    before any encoding happens — refusing is cheap on purpose.
    """

    private = SearchEngine(engine.index, engine.encoder, max_live_additions=1)
    private.add_profile(guest_record())

    with pytest.raises(ValueError, match="לאפס"):
        private.add_profile(guest_record(FirstName="נועה", LastName="ברק"))

    assert private.reset() == 1
    private.add_profile(guest_record(FirstName="נועה", LastName="ברק"))
    assert len(private.added_profile_ids) == 1


def test_reset_endpoint_reports_what_it_discarded(engine) -> None:
    client = create_app(fresh_engine(engine)).test_client()
    created = client.post("/api/profiles", json={
        "first_name": "יונתן",
        "last_name": "תפוחי",
        "title": "סמנכ\"ל מיזוגים ורכישות",
        "experience": "הוביל עשרות עסקאות מיזוג ורכישה בתחום הפינטק.",
    })
    assert created.status_code == 201
    assert client.get("/api/meta").json["added_profile_count"] == 1

    reset = client.post("/api/reset")
    assert reset.status_code == 200
    assert reset.json["status"] == "ok"
    assert reset.json["discarded"] == 1

    meta = client.get("/api/meta").json
    assert meta["added_profile_count"] == 0
    # The UI reads this to decide whether to offer the reset control at all.
    assert meta["max_added_profiles"] >= 1
    assert client.get("/api/profiles").json["added_count"] == 0


def test_directory_snapshot_is_one_consistent_state(engine) -> None:
    """The directory payload must describe one corpus, not two read a moment apart.

    `profiles` and `added_profile_ids` each re-read the live state, so a caller that
    reads both straddles any concurrent add or reset. The harmful direction returns
    a visitor's profile with no ``added`` flag — invented text presented as part of
    the measured corpus — which is reachable as soon as the UI is shared.
    """

    private = fresh_engine(engine)
    profile = private.add_profile(guest_record())

    profiles, added = private.directory_snapshot()
    assert added == frozenset({profile.profile_id})
    assert profile.profile_id in {candidate.profile_id for candidate in profiles}

    # A write landing afterwards must not retroactively edit what was handed out.
    private.reset()
    assert added == frozenset({profile.profile_id}), "an issued snapshot must not follow later writes"
    assert profile.profile_id in {candidate.profile_id for candidate in profiles}

    after_profiles, after_added = private.directory_snapshot()
    assert after_added == frozenset()
    assert profile.profile_id not in {candidate.profile_id for candidate in after_profiles}


def test_directory_payload_never_reports_additions_it_does_not_contain(engine) -> None:
    client = create_app(fresh_engine(engine)).test_client()
    created = client.post("/api/profiles", json={
        "first_name": "יונתן",
        "last_name": "תפוחי",
        "experience": "הוביל עשרות עסקאות מיזוג ורכישה בתחום הפינטק.",
    })
    assert created.status_code == 201

    payload = client.get("/api/profiles").json
    flagged = [row for row in payload["profiles"] if row["added"]]
    assert payload["added_count"] == len(flagged) == 1
    # The two endpoints must agree, because the UI decides whether to offer the
    # reset control from one and renders the badges from the other.
    assert client.get("/api/meta").json["added_profile_count"] == payload["added_count"]
