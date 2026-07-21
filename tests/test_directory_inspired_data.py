import json
from pathlib import Path

from maoz_search.domain import Profile
from maoz_search.projection import project_profiles


def test_directory_inspired_corpus_is_synthetic_and_projection_compatible() -> None:
    path = Path("data/directory_inspired_synthetic_profiles.json")
    records = json.loads(path.read_text(encoding="utf-8"))

    assert len(records) == 20
    assert all(record["_synthetic"] is True for record in records)
    assert all(
        record["Source_Basis__c"] == "public_directory_field_shape_only; no_person_level_copy"
        for record in records
    )
    assert all(record["FirstName"] == "פרופיל" for record in records)
    assert all("סינתטי" in record["LastName"] for record in records)
    assert all("סינתטי" in record["Account"]["Name"] for record in records)

    profiles = tuple(Profile.from_salesforce(record) for record in records)
    assert len({profile.profile_id for profile in profiles}) == len(profiles)
    assert len({profile.account_id for profile in profiles}) == len(profiles)
    assert len({profile.sector for profile in profiles}) >= 12
    assert {profile.cohort for profile in profiles} >= {
        "מחזור א",
        "מחזור ב",
        "מחזור ג",
        "מחזור ד",
        "מחזור ה",
        "מחזור ו",
        "מחזור ז",
        "מחזור ח",
        "מחזור ט",
        "מחזור י",
        "מחזור יא",
        "מחזור יב",
        "מחזור יג",
    }
    assert len(project_profiles(profiles)) == len(profiles) * 4
