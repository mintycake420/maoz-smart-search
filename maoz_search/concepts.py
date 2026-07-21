"""Transparent, staff-owned domain concept expansion.

The lexicon exists because the measured dense model and its paired reranker did
not reliably bridge MAOZ's required flagship concept.  Entries contain concepts,
never profile identifiers, and affect dense query formulation only.  Evidence is
always quoted from the original profile.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .artifacts import load_json
from .normalization import normalize_text

_SALESFORCE_CONTACT_ID = re.compile(r"\b003[A-Za-z0-9]{12,15}\b")


@dataclass(frozen=True, slots=True)
class ConceptMatch:
    concept_id: str
    label_he: str
    trigger: str
    expansions: tuple[str, ...]

    @property
    def bridge_label(self) -> str:
        return f"{self.trigger} ↔ {' · '.join(self.expansions)}"


class ConceptLexicon:
    def __init__(self, concepts: tuple[dict, ...]) -> None:
        self._concepts = concepts

    @classmethod
    def load(cls, path: Path) -> "ConceptLexicon":
        payload = load_json(path)
        concepts = payload.get("concepts")
        if not isinstance(concepts, list):
            raise ValueError("concepts.json must contain a concepts array")
        guardrails = payload.get("guardrails")
        if not isinstance(guardrails, dict) or guardrails.get("profile_ids_forbidden") is not True:
            raise ValueError("concepts.json must enforce the profile_ids_forbidden guardrail")
        concept_ids: set[str] = set()
        for item in concepts:
            if not isinstance(item, dict) or not all(key in item for key in ("id", "label_he", "triggers", "expansions")):
                raise ValueError("Every concept needs id, label_he, triggers, and expansions")
            if not isinstance(item["id"], str) or not item["id"].strip():
                raise ValueError("Every concept id must be a non-empty string")
            if item["id"] in concept_ids:
                raise ValueError(f"Duplicate concept id: {item['id']}")
            concept_ids.add(item["id"])
            if not isinstance(item["label_he"], str) or not item["label_he"].strip():
                raise ValueError(f"Concept {item['id']} needs a non-empty Hebrew label")
            if not isinstance(item["triggers"], list) or not isinstance(item["expansions"], list):
                raise ValueError(f"Concept {item['id']} triggers and expansions must be arrays")
            if not item["triggers"] or not item["expansions"]:
                raise ValueError(f"Concept {item.get('id')} cannot have empty triggers or expansions")
            phrases = [item["id"], item["label_he"], *item["triggers"], *item["expansions"]]
            if not all(isinstance(value, str) and value.strip() for value in phrases):
                raise ValueError(f"Concept {item['id']} contains a blank or non-string phrase")
            if any(_SALESFORCE_CONTACT_ID.search(value) for value in phrases):
                raise ValueError(f"Concept {item['id']} contains a forbidden Salesforce Contact ID")
        return cls(tuple(concepts))

    def match(self, query: str) -> tuple[ConceptMatch, ...]:
        normalized_query = normalize_text(query)
        matches: list[ConceptMatch] = []
        for item in self._concepts:
            for trigger in item["triggers"]:
                normalized_trigger = normalize_text(str(trigger))
                if normalized_trigger and normalized_trigger in normalized_query:
                    matches.append(
                        ConceptMatch(
                            concept_id=str(item["id"]),
                            label_he=str(item["label_he"]),
                            trigger=str(trigger),
                            expansions=tuple(str(value) for value in item["expansions"]),
                        )
                    )
                    break
        return tuple(matches)

    def expanded_queries(self, query: str) -> tuple[str, ...]:
        expansions = [query]
        for match in self.match(query):
            expansions.extend(match.expansions)
        return tuple(dict.fromkeys(expansions))
