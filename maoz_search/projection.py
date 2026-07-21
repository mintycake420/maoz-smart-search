"""The narrow seam between Salesforce-shaped fixtures and searchable aspects."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from .domain import Aspect, Profile, SourceSpan
from .normalization import NORMALIZATION_VERSION, normalize_text

PROJECTION_VERSION = "four-aspect-v1"

_SELF_AUTHORED = "self_described"

_ASPECT_SPECS: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    (
        "role_org",
        "תפקיד וארגון",
        (
            ("Title", "title"),
            ("Account.Name", "organisation"),
            ("Sector__c", "sector"),
        ),
    ),
    (
        "trajectory",
        "ניסיון",
        (
            ("Experience__c", "experience"),
            ("Description", "description"),
        ),
    ),
    (
        "interests_values",
        "תחומי עניין וערכים",
        (
            ("Areas_of_Activity__c", "areas_of_activity"),
            ("Interests__c", "interests"),
            ("Values__c", "values"),
        ),
    ),
    (
        "affiliations",
        "שיוכים ורשתות",
        (
            ("Affiliations__c", "affiliations"),
            ("Cohort__c", "cohort"),
        ),
    ),
)


def projection_contract_hash() -> str:
    payload = {
        "projection_version": PROJECTION_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "aspects": _ASPECT_SPECS,
        "self_authored_excluded_from_lexical": True,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def project_profile(profile: Profile) -> tuple[Aspect, ...]:
    aspects: list[Aspect] = []
    for kind, label_he, fields in _ASPECT_SPECS:
        spans: list[SourceSpan] = []
        for salesforce_field, attribute in fields:
            value = getattr(profile, attribute)
            if not value:
                continue
            provenance = profile.provenance.get(salesforce_field, "member_confirmed")
            spans.append(
                SourceSpan(
                    field=salesforce_field,
                    text=value,
                    provenance=provenance,
                    include_in_lexical=provenance != _SELF_AUTHORED,
                )
            )

        embedding_parts = [span.text for span in spans]
        lexical_parts = [span.text for span in spans if span.include_in_lexical]
        aspects.append(
            Aspect(
                key=f"{profile.profile_id}:{kind}",
                profile_id=profile.profile_id,
                kind=kind,
                label_he=label_he,
                embedding_text=normalize_text(" · ".join(embedding_parts)),
                lexical_text=normalize_text(" · ".join(lexical_parts)),
                sources=tuple(spans),
            )
        )
    return tuple(aspects)


def project_profiles(profiles: Iterable[Profile]) -> tuple[Aspect, ...]:
    return tuple(aspect for profile in profiles for aspect in project_profile(profile))
