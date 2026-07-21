import json
from pathlib import Path

from maoz_search.normalization import canonical_tokens, clitic_variants, normalize_text, token_intersection


def test_normalizes_hebrew_punctuation_and_niqqud() -> None:
    assert normalize_text("  תְּנוּעוֹת־נוֹעַר  ") == "תנועות-נוער"


def test_prefix_variant_is_additive() -> None:
    assert clitic_variants("בחברה") == ("בחברה", "חברה")
    assert clitic_variants("ולמתנדבים") == ("ולמתנדבים", "למתנדבים", "מתנדבים")
    assert clitic_variants("נוער") == ("נוער",)


def test_flagship_surface_terms_do_not_overlap() -> None:
    query = "חינוך בלתי פורמלי"
    evidence = "הקימה והובילה שלוש תנועות נוער אזוריות"
    assert token_intersection(query, evidence) == frozenset()
    assert canonical_tokens(query) == ("חינוך", "בלתי", "פורמלי")


def test_no_corpus_profile_contains_the_flagship_query() -> None:
    """The zero-overlap premise has to hold against the data, not against a literal.

    The assertion above compares two frozen strings and never opens a corpus, so it
    stayed green while a companion-corpus profile carried ``חינוך בלתי פורמלי`` verbatim
    in ``Sector__c`` -- which silently turned the headline demo into an exact string
    match on that corpus while still reporting rank 1 and a strong tier. A control that
    cannot fail on real data is not a control, so this reads every shipped corpus.
    """
    root = Path(__file__).resolve().parents[1]
    query_tokens = set(canonical_tokens("חינוך בלתי פורמלי"))

    corpora = [
        root / "data" / "synthetic_profiles.json",
        root / "data" / "directory_inspired_synthetic_profiles.json",
    ]
    for corpus in corpora:
        if not corpus.exists():  # companion corpus is optional
            continue
        for record in json.loads(corpus.read_text(encoding="utf-8")):
            searchable = " ".join(
                value["Name"] if isinstance(value, dict) and "Name" in value else str(value)
                for field, value in record.items()
                if field != "Field_Provenance__c"
            )
            covered = query_tokens.intersection(canonical_tokens(searchable))
            assert covered != query_tokens, (
                f"{corpus.name} record {record['Id']} contains every token of the "
                f"flagship query, so the no-shared-words demo would be satisfied by "
                f"literal matching on that corpus"
            )
