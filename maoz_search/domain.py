"""Domain objects at the Salesforce projection boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class ProfileValidationError(ValueError):
    """Raised when a fixture crosses the deliberately narrow POC boundary."""


@dataclass(frozen=True, slots=True)
class Profile:
    profile_id: str
    first_name: str
    last_name: str
    title: str
    description: str
    account_id: str
    organisation: str
    # Standard Contact contact details. Deliberately absent from
    # projection._ASPECT_SPECS: they are never embedded, never searchable, and
    # are surfaced only after a match has been made on other grounds. The
    # projection is an allow-list, so this exclusion is structural rather than
    # a rule someone has to remember.
    email: str
    phone: str
    sector: str
    region: str
    cohort: str
    areas_of_activity: str
    experience: str
    interests: str
    values: str
    affiliations: str
    provenance: Mapping[str, str]

    @property
    def display_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    @classmethod
    def from_salesforce(cls, record: Mapping[str, Any]) -> "Profile":
        """Validate and project one synthetic Salesforce-shaped Contact record.

        A real Contact is intentionally rejected.  This makes accidentally dropping
        production data into the take-home fail closed instead of silently indexing it.
        """

        if record.get("_synthetic") is not True:
            raise ProfileValidationError("Part B accepts synthetic fixtures only")

        account = record.get("Account")
        if not isinstance(account, Mapping):
            raise ProfileValidationError("Account must be an object with Id and Name")

        provenance = record.get("Field_Provenance__c", {})
        if not isinstance(provenance, Mapping):
            raise ProfileValidationError("Field_Provenance__c must be an object")

        def text(field: str, source: Mapping[str, Any] = record) -> str:
            value = source.get(field, "")
            if value is None:
                return ""
            if not isinstance(value, str):
                raise ProfileValidationError(f"{field} must be text")
            return value.strip()

        profile_id = text("Id")
        if not profile_id.startswith("003SYN"):
            raise ProfileValidationError("Synthetic Contact Id must start with 003SYN")

        return cls(
            profile_id=profile_id,
            first_name=text("FirstName"),
            last_name=text("LastName"),
            title=text("Title"),
            description=text("Description"),
            account_id=text("Id", account),
            organisation=text("Name", account),
            email=text("Email"),
            phone=text("Phone"),
            sector=text("Sector__c"),
            region=text("Region__c"),
            cohort=text("Cohort__c"),
            areas_of_activity=text("Areas_of_Activity__c"),
            experience=text("Experience__c"),
            interests=text("Interests__c"),
            values=text("Values__c"),
            affiliations=text("Affiliations__c"),
            provenance={str(key): str(value) for key, value in provenance.items()},
        )


@dataclass(frozen=True, slots=True)
class SourceSpan:
    field: str
    text: str
    provenance: str
    include_in_lexical: bool


@dataclass(frozen=True, slots=True)
class Aspect:
    key: str
    profile_id: str
    kind: str
    label_he: str
    embedding_text: str
    lexical_text: str
    sources: tuple[SourceSpan, ...]
