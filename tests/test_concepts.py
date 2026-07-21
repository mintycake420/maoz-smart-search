from __future__ import annotations

import json

import pytest

from maoz_search.concepts import ConceptLexicon


def write_lexicon(tmp_path, concept):
    path = tmp_path / "concepts.json"
    path.write_text(
        json.dumps(
            {
                "concepts": [concept],
                "guardrails": {"profile_ids_forbidden": True},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_concept_schema_requires_phrase_arrays(tmp_path) -> None:
    path = write_lexicon(
        tmp_path,
        {"id": "example", "label_he": "דוגמה", "triggers": "לא מערך", "expansions": ["ביטוי"]},
    )
    with pytest.raises(ValueError, match="must be arrays"):
        ConceptLexicon.load(path)


def test_concept_schema_enforces_no_profile_ids(tmp_path) -> None:
    path = write_lexicon(
        tmp_path,
        {
            "id": "example",
            "label_he": "דוגמה",
            "triggers": ["חיפוש"],
            "expansions": ["003SYN000000001"],
        },
    )
    with pytest.raises(ValueError, match="forbidden Salesforce Contact ID"):
        ConceptLexicon.load(path)
