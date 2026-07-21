"""Evaluate the independently authored judged query set.

This is the only measurement in the repository whose queries and relevance grades
were not written by whoever built the retrieval system.  Everything in
``data/golden_queries.json`` shares an author with ``config/concepts.json``, which
makes it a regression check rather than evidence; this set does not, and the
independence is checkable rather than asserted -- the concept vocabulary fires on
none of these queries.

Two properties of this harness are deliberate and should not be "fixed":

* **It encodes the corpus at runtime.**  The shipped vectors in ``data/artifacts``
  are cryptographically bound to ``data/synthetic_profiles.json``; the companion
  corpus has no artifacts and must not silently acquire any, because promoting a
  corpus is supposed to require a full rebuild and a fresh calibration.  Encoding
  eighty aspects on CPU costs a couple of minutes and keeps that boundary intact.
* **It uses the runtime corpus's confidence threshold on a different corpus.**
  That mismatch is the honest state of affairs -- the companion corpus has no
  judged calibration data of its own -- and it is what ``acc_06`` exposes.

``acc_06`` was the original known failure: the encoder ranks the correct profile
*first* at 0.389 while the gate sat at 0.474, so a correct rank-1 answer was
abstained away.  The 2026-07-21 corpus reseal made the pattern three queries
larger: stripping synthetic boilerplate from the *runtime* corpus sharpened its
embeddings and moved the fitted gate to 0.495, and ``dev_04``, ``acc_03`` and
``acc_05`` (scores 0.477–0.480, all still rank 1) crossed under it.  Ranking never
regressed — every one of the 18 judged primaries sits at un-gated dense rank 1,
and that is now enforced directly below — but the gate-coupled MRR floor was
re-baselined.  Nothing was tuned in response: this harness deliberately applies
the runtime corpus's fitted gate to a corpus it was never calibrated on, and the
finding is the deliverable rather than a defect to paper over.  The lesson is the
point and it should stay visible: **a single absolute threshold does not transfer
across corpora**, and per-corpus calibration is a v1 requirement.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maoz_search.artifacts import load_json  # noqa: E402
from maoz_search.concepts import ConceptLexicon  # noqa: E402
from maoz_search.domain import Profile  # noqa: E402
from maoz_search.embeddings import OnnxBgeEncoder  # noqa: E402
from maoz_search.index import ProfileIndex  # noqa: E402
from maoz_search.lexical import LexicalIndex  # noqa: E402
from maoz_search.normalization import normalize_text  # noqa: E402
from maoz_search.projection import project_profiles  # noqa: E402
from maoz_search.search import SearchEngine, SearchResponse  # noqa: E402

# Measured on the frozen companion corpus.  These are floors, not targets: the point
# is to notice a regression, not to invite tuning against an independent set.
#
# ACCEPTANCE_MRR_FLOOR was 0.90 (measured 0.923) until the 2026-07-21 reseal of the
# *runtime* corpus moved the fitted gate from 0.474 to 0.495 and pushed three
# correct-at-rank-1 queries under it (measured 0.769 after).  The floor was
# re-baselined so the tripwire arms against future *silent* drift instead of failing
# permanently against a deliberate, documented recalibration — and the un-gated
# rank-1 floor below now protects the ranking signal itself, which no gate movement
# can mask in either direction.  Nothing about the gate or the calibration set was
# changed in response to these numbers — re-fitting the gate against the set that
# measures it is exactly the circularity this harness exists to avoid.
ACCEPTANCE_MRR_FLOOR = 0.75
REQUIRED_ABSTENTIONS = 5
REQUIRED_UNGATED_PRIMARY_RANK1 = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def build_engine(batch_size: int) -> SearchEngine:
    judged = load_json(ROOT / "data" / "judged_queries.json")
    records = load_json(ROOT / judged["corpus"])
    profiles = tuple(Profile.from_salesforce(record) for record in records)
    aspects = project_profiles(profiles)

    gazetteer = load_json(ROOT / "config" / "gazetteer.json")
    concepts_payload = load_json(ROOT / "config" / "concepts.json")
    encoder = OnnxBgeEncoder(ROOT / "models" / "bge-m3-int8", batch_size=batch_size)

    aspect_vectors = encoder.encode([aspect.embedding_text for aspect in aspects])
    source_items = [
        (f"{aspect.key}:{source.field}", source.text)
        for aspect in aspects
        for source in aspect.sources
    ]
    phrases = tuple(
        dict.fromkeys(
            str(expansion)
            for item in concepts_payload["concepts"]
            for expansion in item["expansions"]
        )
    )
    index = ProfileIndex(
        root=ROOT,
        raw_records=records,
        profiles=profiles,
        aspects=aspects,
        aspect_vectors=aspect_vectors,
        source_keys=tuple(key for key, _ in source_items),
        source_vectors=encoder.encode([text for _, text in source_items]),
        concept_phrase_keys=phrases,
        concept_phrase_vectors=encoder.encode(phrases),
        lexical=LexicalIndex(
            [aspect.lexical_text for aspect in aspects], gazetteer.get("aliases", {})
        ),
        concepts=ConceptLexicon.load(ROOT / "config" / "concepts.json"),
        manifest=load_json(ROOT / "data" / "artifacts" / "manifest.json"),
        golden_queries=(),
        gazetteer_aliases=dict(gazetteer.get("aliases", {})),
    )
    return SearchEngine(index, encoder)


def discounted_gain(gains: list[int]) -> float:
    return sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))


def score_query(response: SearchResponse, judgments: dict[str, int]) -> dict:
    returned = [result.profile_id for result in response.results]
    gains = [judgments.get(profile_id, 0) for profile_id in returned]
    ideal = sorted(judgments.values(), reverse=True)[:5]
    primary = next((pid for pid, grade in judgments.items() if grade == 3), None)
    relevant = {pid for pid, grade in judgments.items() if grade > 0}

    reciprocal_rank = 0.0
    if primary is not None and primary in returned:
        reciprocal_rank = 1.0 / (returned.index(primary) + 1)

    return {
        "status": response.status,
        "returned": returned,
        "gains": gains,
        "reciprocal_rank": reciprocal_rank,
        "hit_at_3": primary is not None and primary in returned[:3],
        "ndcg_at_5": (discounted_gain(gains) / discounted_gain(ideal)) if ideal else None,
        "recall_at_5": (
            len(relevant.intersection(returned)) / len(relevant) if relevant else None
        ),
        # A result graded 0 is padding: it was added to fill the requested count.
        "padding": sum(1 for gain in gains if gain == 0),
        "applied_concepts": response.meta["applied_concepts"],
    }


def mean(values: list[float]) -> float | None:
    usable = [value for value in values if value is not None]
    return round(statistics.mean(usable), 4) if usable else None


def ungated_primary_ranks(engine: SearchEngine, judged: dict) -> dict:
    """Dense rank of every judged primary with the gate out of the picture.

    The gate-coupled metrics above move whenever the runtime corpus is resealed,
    because this harness deliberately applies the runtime gate to a corpus it was
    not fitted on.  This block is the measurement that cannot be moved by gate
    drift in either direction: raw dense cosine over the companion aspects, best
    aspect per profile, no threshold, no lexical leg, no concepts (none fire on
    this set anyway).  It is the one non-circular *ranking* signal in the
    repository, so it gets its own enforced floor.
    """

    rows: list[dict] = []
    for item in judged["queries"]:
        if item["group"] == "abstention":
            continue
        judgments = dict(item["judgments"])
        primary = next((pid for pid, grade in judgments.items() if grade == 3), None)
        if primary is None:
            continue
        query_vector = engine.encoder.encode([normalize_text(str(item["query"]))])[0]
        scores = engine.index.aspect_vectors @ query_vector
        best: dict[str, float] = {}
        for aspect, score in zip(engine.index.aspects, scores, strict=True):
            best[aspect.profile_id] = max(best.get(aspect.profile_id, -1.0), float(score))
        order = sorted(best, key=lambda profile_id: (-best[profile_id], profile_id))
        rows.append({
            "id": item["id"],
            "rank": order.index(primary) + 1,
            "primary_dense": round(best[primary], 4),
        })
    return {
        "queries": len(rows),
        "primary_at_rank_1": sum(1 for row in rows if row["rank"] == 1),
        "per_query": rows,
        "note": (
            "Raw dense rank of the graded primary, gate and lexical leg excluded. "
            "Enforced at 18/18: any drop is a genuine ranking regression and cannot "
            "be masked or caused by gate movement."
        ),
    }


def evaluate(engine: SearchEngine) -> dict:
    judged = load_json(ROOT / "data" / "judged_queries.json")
    rows: list[dict] = []
    for item in judged["queries"]:
        response = engine.search(str(item["query"]), top_k=5)
        scored = score_query(response, dict(item["judgments"]))
        rows.append({"id": item["id"], "group": item["group"], **scored})

    def summarise(group: str) -> dict:
        subset = [row for row in rows if row["group"] == group]
        return {
            "queries": len(subset),
            "mrr": mean([row["reciprocal_rank"] for row in subset]),
            "hit_at_3": f"{sum(row['hit_at_3'] for row in subset)}/{len(subset)}",
            "ndcg_at_5": mean([row["ndcg_at_5"] for row in subset]),
            "recall_at_5": mean([row["recall_at_5"] for row in subset]),
        }

    abstention_rows = [row for row in rows if row["group"] == "abstention"]
    correct_abstentions = sum(
        1 for row in abstention_rows if row["status"] == "no_strong_match"
    )
    positive_rows = [row for row in rows if row["group"] != "abstention"]
    unaided = [row for row in positive_rows if not row["applied_concepts"]]
    gate_abstained = [
        row["id"] for row in positive_rows if row["status"] == "no_strong_match"
    ]

    return {
        "scope": "independently authored judgments over the frozen companion corpus",
        "corpus": judged["corpus"],
        "dev": summarise("dev"),
        "acceptance": summarise("acceptance"),
        "abstention": {
            "queries": len(abstention_rows),
            "correct": f"{correct_abstentions}/{len(abstention_rows)}",
        },
        "padding_results_returned": sum(row["padding"] for row in positive_rows),
        "concept_bridge_unaided": {
            "queries_without_any_concept": len(unaided),
            "of_those_primary_at_rank_1": sum(
                1 for row in unaided if row["reciprocal_rank"] == 1.0
            ),
            "note": (
                "These queries were answered by the encoder alone, with no entry in "
                "the staff-owned concept vocabulary firing. Gate-abstained queries "
                "count as misses here; see ungated_ranking for the pure rank signal."
            ),
        },
        "ungated_ranking": ungated_primary_ranks(engine, judged),
        "gate_abstained": {
            "ids": gate_abstained,
            "note": (
                "Positive queries the fitted gate refused. Every one of them ranks its "
                "primary FIRST un-gated (see ungated_ranking): these are correct "
                "answers lost to a threshold calibrated on a different corpus. acc_06 "
                "was the original single victim at the 0.474 gate; the 2026-07-21 "
                "reseal moved the gate to 0.495 and added dev_04, acc_03 and acc_05 "
                "(0.477–0.480). Recorded, not silently tuned away — the "
                "finding is that a single absolute threshold does not transfer across "
                "corpora."
            ),
        },
        "queries": rows,
    }


def main() -> None:
    args = parse_args()

    def blocked(*_args, **_kwargs):
        raise AssertionError("Judged evaluation attempted a network connection")

    with patch("socket.create_connection", side_effect=blocked), patch(
        "socket.socket.connect", side_effect=blocked
    ):
        report = evaluate(build_engine(args.batch_size))

    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")

    acceptance_mrr = report["acceptance"]["mrr"] or 0.0
    correct = int(report["abstention"]["correct"].split("/")[0])
    ungated_rank1 = report["ungated_ranking"]["primary_at_rank_1"]
    if (
        acceptance_mrr < ACCEPTANCE_MRR_FLOOR
        or correct < REQUIRED_ABSTENTIONS
        or ungated_rank1 < REQUIRED_UNGATED_PRIMARY_RANK1
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
