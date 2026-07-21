import pytest

from maoz_search.domain import Profile, ProfileValidationError
from maoz_search.projection import project_profile


def synthetic_record() -> dict:
    return {
        "_synthetic": True,
        "Id": "003SYNTEST000001",
        "FirstName": "נועה",
        "LastName": "בדיקה",
        "Title": "מנהלת",
        "Description": "תיאור עצמי",
        "Account": {"Id": "001SYNTEST000001", "Name": "ארגון בדיקה"},
        "Sector__c": "חברה אזרחית",
        "Region__c": "צפון",
        "Cohort__c": "מחזור בדיקה",
        "Areas_of_Activity__c": "מנהיגות",
        "Experience__c": "הובלת צוותים",
        "Interests__c": "קהילה",
        "Values__c": "שותפות",
        "Affiliations__c": "רשת בדיקה",
        "Available_for_Introductions__c": True,
        "Field_Provenance__c": {
            "Description": "self_described",
            "Interests__c": "self_described",
            "Experience__c": "staff_verified",
        },
        "Private_Assessment__c": "RESTRICTED_CANARY_NEVER_INDEX",
    }


def test_projection_has_four_aspects_and_drops_restricted_canary() -> None:
    aspects = project_profile(Profile.from_salesforce(synthetic_record()))
    assert [aspect.kind for aspect in aspects] == [
        "role_org",
        "trajectory",
        "interests_values",
        "affiliations",
    ]
    combined = " ".join(aspect.embedding_text + " " + aspect.lexical_text for aspect in aspects)
    assert "restricted_canary" not in combined.casefold()


def test_contact_details_never_enter_the_index() -> None:
    """Email and phone are carried on the record and excluded from retrieval.

    The projection is an allow-list, so this holds structurally rather than by
    a rule someone has to remember: a field the spec does not name cannot reach
    an aspect, an embedding, or the lexical leg. Contact details are disclosed
    only after a match has been made on other grounds, which is the
    data-minimisation position Part A.3 argues.
    """

    record = synthetic_record()
    record["Email"] = "noa.barak@nitzan.example"
    record["Phone"] = "050-000-0001"
    profile = Profile.from_salesforce(record)
    assert profile.email == "noa.barak@nitzan.example"
    assert profile.phone == "050-000-0001"

    aspects = project_profile(profile)
    haystack = " ".join(
        aspect.embedding_text + " " + aspect.lexical_text for aspect in aspects
    ).casefold()
    assert "nitzan.example" not in haystack
    assert "050-000-0001" not in haystack
    spans = {source.field for aspect in aspects for source in aspect.sources}
    assert not spans.intersection({"Email", "Phone"})


def test_shipped_corpus_carries_contact_details_outside_the_index() -> None:
    """The same guarantee, asserted against the corpus that actually ships."""

    import json
    from pathlib import Path

    records = json.loads(
        Path("data/synthetic_profiles.json").read_text(encoding="utf-8")
    )
    profiles = [Profile.from_salesforce(record) for record in records]
    assert all(profile.email and profile.phone for profile in profiles)
    # RFC 2606 reserves .example, so no address here can ever be routable.
    assert all(profile.email.endswith(".example") for profile in profiles)

    for profile in profiles:
        haystack = " ".join(
            aspect.embedding_text + " " + aspect.lexical_text
            for aspect in project_profile(profile)
        ).casefold()
        assert profile.email.casefold() not in haystack, profile.profile_id
        assert profile.phone not in haystack, profile.profile_id


def test_self_authored_source_is_dense_only() -> None:
    aspects = {aspect.kind: aspect for aspect in project_profile(Profile.from_salesforce(synthetic_record()))}
    assert "תיאור עצמי" in aspects["trajectory"].embedding_text
    assert "תיאור עצמי" not in aspects["trajectory"].lexical_text


def test_real_record_fails_closed() -> None:
    record = synthetic_record()
    record["_synthetic"] = False
    with pytest.raises(ProfileValidationError, match="synthetic"):
        Profile.from_salesforce(record)
