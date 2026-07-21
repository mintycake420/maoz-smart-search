"""Reproduce the frozen Part B acceptance measurements without network access.

These results describe only the synthetic POC corpus.  They are deliberately not
presented as a benchmark or as a substitute for MAOZ-owned query judgments.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maoz_search import SearchEngine  # noqa: E402
from maoz_search.normalization import normalize_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-runs", type=int, default=5)
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    return parser.parse_args()


def ablation(engine: SearchEngine) -> dict:
    """Report what the one remaining ranking component actually contributes.

    Ranking is dense-only: the BM25 fusion leg was removed after this harness
    measured its contribution at exactly zero.  That leaves the staff-owned concept
    vocabulary as the only thing that can still move a rank, so it has to justify
    itself on every run rather than on the day it was added.  Part A.2 Stage 2 says a
    component that cannot show a measured margin does not ship; this is that check,
    wired in permanently so the claim cannot quietly stop being true.
    """
    real_match = engine.index.concepts.match

    def tally_with(enabled: bool) -> dict[str, str]:
        engine.index.concepts.match = real_match if enabled else (lambda query: [])
        try:
            rows = [
                (
                    item["role"],
                    item.get("expected_top_profile_id") is None,
                    _passed(engine.search(str(item["query"]), top_k=5), item),
                )
                for item in engine.index.golden_queries
            ]
        finally:
            engine.index.concepts.match = real_match

        def group(roles: set[str], negative: bool) -> str:
            subset = [row for row in rows if row[0] in roles and row[1] is negative]
            return f"{sum(1 for row in subset if row[2])}/{len(subset)}"

        return {
            "acceptance_top1": group({"demo", "test"}, False),
            "acceptance_abstentions": group({"demo", "test"}, True),
            "heldout_top1": group({"heldout"}, False),
        }

    with_bridge = tally_with(True)
    without_bridge = tally_with(False)
    ranks = concept_rank_effect(engine)

    # The gated tally alone cannot support the claim being made. A query that starts
    # passing might have moved up the ranking, or might have sat at rank 1 all along
    # and merely been lifted over dense_threshold by the expansion -- and the lexical
    # gate is live in both arms either way. Those are different components doing
    # different jobs, so the rank measurement below is what actually tests "the concept
    # bridge is a ranking component", and both have to agree before the claim stands.
    gated_gain = _numerator(with_bridge["acceptance_top1"]) - _numerator(
        without_bridge["acceptance_top1"]
    )
    earns_its_place = (
        gated_gain > 0 and ranks["improved"] > 0 and ranks["regressed"] == 0
    )
    return {
        "with_concept_bridge": with_bridge,
        "without_concept_bridge": without_bridge,
        "rank_only_effect": ranks,
        "earns_its_place": earns_its_place,
        "note": (
            "The bare encoder is the baseline. The concept bridge is the only remaining "
            "ranking component and must beat it; the removed BM25 fusion leg could not, "
            "which is why it is gone. Held-out figures are expected to be equal - the "
            "bridge is deliberately not configured for held-out intents. This check is "
            "enforced, not merely reported: main() exits non-zero when earns_its_place "
            "is false, so a later model, artifact or lexicon change cannot quietly "
            "invalidate the justification while the command stays green."
        ),
    }


def _numerator(tally: str) -> int:
    return int(tally.split("/")[0])


def concept_rank_effect(engine: SearchEngine) -> dict:
    """Measure the bridge's effect on *rank*, with the gate taken out of the picture.

    Scores every acceptance positive by dense similarity alone -- best over the plain
    query, then best over the query plus its approved concept phrases -- and compares
    where the expected profile lands. No threshold, no lexical leg, no abstention: just
    the ordering the encoder produces. This is the measurement that justifies calling
    the concept vocabulary a ranking component rather than a gate-opener.
    """
    real_match = engine.index.concepts.match
    concept_vectors = engine.index.concept_vectors_by_text

    def dense_rank(query: str, use_concepts: bool, target: str) -> int:
        vectors = [engine.encoder.encode([normalize_text(query)])[0]]
        if use_concepts:
            for match in real_match(query):
                for expansion in match.expansions:
                    vectors.append(concept_vectors[normalize_text(expansion)])
        scores = (engine.index.aspect_vectors @ np.vstack(vectors).T).max(axis=1)
        best: dict[str, float] = {}
        for aspect, score in zip(engine.index.aspects, scores, strict=True):
            best[aspect.profile_id] = max(best.get(aspect.profile_id, -1e9), float(score))
        order = sorted(best, key=lambda profile_id: (-best[profile_id], profile_id))
        return order.index(target) + 1

    rows = []
    for item in engine.index.golden_queries:
        target = item.get("expected_top_profile_id")
        if item["role"] not in {"demo", "test"} or target is None:
            continue
        query = str(item["query"])
        off = dense_rank(query, False, target)
        on = dense_rank(query, True, target)
        rows.append({"id": item["id"], "rank_without": off, "rank_with": on})

    return {
        "queries": len(rows),
        "improved": sum(1 for row in rows if row["rank_with"] < row["rank_without"]),
        "unchanged": sum(1 for row in rows if row["rank_with"] == row["rank_without"]),
        "regressed": sum(1 for row in rows if row["rank_with"] > row["rank_without"]),
        "per_query": rows,
        "note": (
            "Dense rank of the expected profile, gate and lexical leg excluded. Queries "
            "whose rank is unchanged and already 1 are carried by the encoder alone; a "
            "query stuck at a poor rank in both arms is passing acceptance through the "
            "lexical strong-match gate, not through ranking."
        ),
    }


def _passed(response, item: dict) -> bool:
    returned = [result.profile_id for result in response.results]
    if item.get("should_abstain"):
        return response.status == "no_strong_match"
    return bool(returned and returned[0] == item.get("expected_top_profile_id"))


def evaluate(engine: SearchEngine, timing_runs: int) -> dict:
    rows: list[dict] = []
    for item in engine.index.golden_queries:
        response = engine.search(str(item["query"]), top_k=5)
        returned = [result.profile_id for result in response.results]
        expected = item.get("expected_top_profile_id")
        should_abstain = bool(item.get("should_abstain"))
        passed = response.status == "no_strong_match" if should_abstain else bool(returned and returned[0] == expected)
        rows.append(
            {
                "id": item["id"],
                "role": item["role"],
                "query_kind": item["query_kind"],
                "status": response.status,
                "expected_profile_id": expected,
                "top_profile_id": returned[0] if returned else None,
                "passed": passed,
                "applied_concepts": response.meta["applied_concepts"],
            }
        )

    # The frozen MiniLM / official-FP32 model-comparison block used to be reported
    # here.  It was removed with the comparison feature itself.  Its measured result
    # — official FP32 BGE-M3 ranks the flagship target second, the English-default
    # MiniLM control ranks it eleventh — is a recorded measurement from that run and
    # is no longer re-derivable from shipped artifacts.
    flagship = next(
        item for item in engine.index.golden_queries if item["id"] == "demo_flagship_informal_education"
    )
    timing_query = str(flagship["query"])
    engine.search(timing_query)  # warm session and tokenizer
    durations: list[float] = []
    for _ in range(max(1, timing_runs)):
        started = time.perf_counter()
        engine.search(timing_query)
        durations.append(time.perf_counter() - started)

    # Three groups, reported separately because they carry different evidential weight
    # and adding them together would overstate all three.
    #
    #   calibration - the examples that *selected* dense_threshold.  Reporting these as
    #                 results is circular: the gate was fitted so they would pass.
    #   acceptance  - demo/test rows.  Independent of the fit, but every semantic intent
    #                 is configured in config/concepts.json, so this is a regression
    #                 check against the system's own vocabulary.
    #   held-out    - targets with no concept entry, excluded from the fit.  The only
    #                 group that speaks to generalisation, and still author-written.
    def tally(subset: list[dict]) -> str:
        return f"{sum(row['passed'] for row in subset)}/{len(subset)}"

    calibration_rows = [row for row in rows if row["role"] == "calibration"]
    acceptance_rows = [row for row in rows if row["role"] in {"demo", "test"}]
    heldout_rows = [row for row in rows if row["role"] == "heldout"]
    positive = lambda subset: [row for row in subset if row["expected_profile_id"] is not None]
    negative = lambda subset: [row for row in subset if row["expected_profile_id"] is None]
    return {
        "scope": "frozen synthetic POC only",
        "all_acceptance_checks_passed": all(row["passed"] for row in acceptance_rows),
        "acceptance_top1": tally(positive(acceptance_rows)),
        "acceptance_abstentions": tally(negative(acceptance_rows)),
        "acceptance_note": (
            "Independent of threshold fitting, but every semantic intent here is "
            "configured in config/concepts.json, so this is a regression check against "
            "the system's own vocabulary rather than evidence of generalisation."
        ),
        "heldout_top1": tally(positive(heldout_rows)),
        "heldout_note": (
            "Held-out queries target profiles with no concept-vocabulary entry and were "
            "not used to fit the confidence gate. They are not a MAOZ-judged relevance "
            "measurement and do not estimate production quality."
        ),
        "calibration_fit": {
            "positives_retained": tally(positive(calibration_rows)),
            "negatives_abstained": tally(negative(calibration_rows)),
            "note": (
                "These examples selected dense_threshold. They are reported for "
                "regression only and are NOT evidence: the gate was fitted so they pass."
            ),
        },
        "calibration_consistent": all(row["passed"] for row in calibration_rows),
        "component_ablation": ablation(engine),
        "warm_flagship_latency_seconds": {
            "runs": len(durations),
            "median": round(statistics.median(durations), 3),
            "minimum": round(min(durations), 3),
            "maximum": round(max(durations), 3),
        },
        "queries": rows,
    }


def main() -> None:
    args = parse_args()

    def blocked(*_args, **_kwargs):
        raise AssertionError("Evaluation attempted a network connection")

    with patch("socket.create_connection", side_effect=blocked), patch(
        "socket.socket.connect", side_effect=blocked
    ):
        report = evaluate(SearchEngine.from_default(ROOT), args.timing_runs)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    # Calibration is not evidence, but a calibration row that stops behaving means the
    # fitted gate has drifted from the artifacts, so it still fails the run.
    #
    # The ablation is enforced for a different reason. The BM25 fusion leg was removed
    # for failing to show a margin; leaving the surviving component's justification as
    # a printed number would let the same failure recur silently after a model, artifact
    # or lexicon change. Either the claim is checked on every run or it should not be
    # stated as continuously true.
    if (
        not report["all_acceptance_checks_passed"]
        or not report["calibration_consistent"]
        or not report["component_ablation"]["earns_its_place"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
