"""Build immutable profile vectors and calibration metadata.

This is an explicit preparation command, not part of application startup.  It may
load local model artifacts; it never calls a hosted inference API.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maoz_search.artifacts import canonical_json_bytes, load_json, sha256_bytes, sha256_file  # noqa: E402
from maoz_search.concepts import ConceptLexicon  # noqa: E402
from maoz_search.domain import Profile  # noqa: E402
from maoz_search.embeddings import OnnxBgeEncoder  # noqa: E402
from maoz_search.projection import project_profiles, projection_contract_hash  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bge-model-dir", type=Path, default=ROOT / "models" / "bge-m3-int8")
    parser.add_argument("--upstream-revision", required=True, help="verified Hugging Face commit SHA")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def combined_model_hash(model_dir: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in model_dir.iterdir() if path.is_file())
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def profile_best_from_values(aspects, scores: np.ndarray) -> tuple[float, str]:
    best: dict[str, float] = {}
    for aspect, score in zip(aspects, scores, strict=True):
        best[aspect.profile_id] = max(best.get(aspect.profile_id, -1.0), float(score))
    profile_id, score = max(best.items(), key=lambda pair: (pair[1], pair[0]))
    return score, profile_id


def calibrate(
    aspects,
    vectors: np.ndarray,
    encoder: OnnxBgeEncoder,
    concepts: ConceptLexicon,
    queries: list[dict],
) -> dict:
    calibration = [item for item in queries if item.get("role") == "calibration"]
    if not calibration:
        raise ValueError("At least one calibration query is required")
    positives: list[float] = []
    negatives: list[float] = []
    observations: list[dict] = []
    for item in calibration:
        variants = concepts.expanded_queries(str(item["query"]))
        query_vectors = encoder.encode(variants)
        score_matrix = vectors @ query_vectors.T
        best_scores = np.max(score_matrix, axis=1)
        top_score, top_profile = profile_best_from_values(aspects, best_scores)
        observation = {
            "query_id": item["id"],
            "top_dense": round(top_score, 6),
            "top_profile_id": top_profile,
            "should_abstain": bool(item.get("should_abstain")),
            "concept_ids": [match.concept_id for match in concepts.match(str(item["query"]))],
        }
        observations.append(observation)
        (negatives if item.get("should_abstain") else positives).append(top_score)
    if not positives or not negatives:
        raise ValueError("Calibration requires positive and abstention examples")

    # Build-failing guard for a bug this calibration already shipped once.  Concept
    # expansion lifts a covered query's best score by roughly +0.09..+0.13, and never
    # lifts an out-of-domain negative.  When every calibration positive was concept-
    # covered, the fitted threshold silently encoded that lift, and ordinary queries
    # -- which get no expansion -- were judged against a bar they could not reach.
    # Correct answers at rank 1 were being discarded as "no strong match".
    uncovered_positives = [
        observation
        for observation in observations
        if not observation["should_abstain"] and not observation["concept_ids"]
    ]
    if not uncovered_positives:
        raise ValueError(
            "Calibration needs at least one positive with no concept coverage, "
            "otherwise the threshold is fitted to concept-expanded scores and "
            "un-expanded queries are systematically rejected"
        )

    min_positive = min(positives)
    max_negative = max(negatives)
    if max_negative < min_positive:
        threshold = (max_negative + min_positive) / 2.0
    else:
        threshold = max_negative + 1e-4
    retained = sum(score >= threshold for score in positives)
    return {
        "dense_threshold": round(threshold, 6),
        "result_floor": round(max(-1.0, threshold - 0.035), 6),
        # An absolute floor cannot stop concept-expansion padding: a firing concept
        # lifts every loosely related profile at once, so near misses ride along
        # behind one good match.  Results must also stay near the best result.
        "result_relative_margin": 0.04,
        "basis": "max over the original query and any approved concept expansions",
        "calibration_query_ids": [item["id"] for item in calibration],
        "calibration_positive_retained": f"{retained}/{len(positives)}",
        "calibration_uncovered_positive_ids": [
            observation["query_id"] for observation in uncovered_positives
        ],
        "calibration_observations": observations,
        "scope": "synthetic POC only; must be recalibrated before real-data use",
    }


def main() -> None:
    args = parse_args()
    data_dir = ROOT / "data"
    artifact_dir = data_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    raw_records = load_json(data_dir / "synthetic_profiles.json")
    profiles = tuple(Profile.from_salesforce(record) for record in raw_records)
    aspects = project_profiles(profiles)
    golden_payload = load_json(data_dir / "golden_queries.json")
    queries = list(golden_payload["queries"])
    gazetteer_payload = load_json(ROOT / "config" / "gazetteer.json")
    concepts_payload = load_json(ROOT / "config" / "concepts.json")
    concepts = ConceptLexicon.load(ROOT / "config" / "concepts.json")

    bge = OnnxBgeEncoder(args.bge_model_dir, batch_size=args.batch_size)
    aspect_vectors = bge.encode([aspect.embedding_text for aspect in aspects])
    source_items = [
        (f"{aspect.key}:{source.field}", source.text)
        for aspect in aspects
        for source in aspect.sources
    ]
    source_vectors = bge.encode([text for _, text in source_items])
    concept_phrases = tuple(
        dict.fromkeys(
            str(expansion)
            for item in concepts_payload["concepts"]
            for expansion in item["expansions"]
        )
    )
    concept_phrase_vectors = bge.encode(concept_phrases)
    np.savez_compressed(
        artifact_dir / "bge_m3_profiles.npz",
        aspect_keys=np.asarray([aspect.key for aspect in aspects]),
        aspect_vectors=aspect_vectors,
        source_keys=np.asarray([key for key, _ in source_items]),
        source_vectors=source_vectors,
        concept_phrase_keys=np.asarray(concept_phrases),
        concept_phrase_vectors=concept_phrase_vectors,
    )

    confidence = calibrate(aspects, aspect_vectors, bge, concepts, queries)

    # The MiniLM and official-FP32 comparison artifacts used to be built and sealed
    # here.  The frozen model comparison was cut from the product (the live
    # add-a-profile flow argues generalisation more directly than a comparison over
    # fixtures we authored), and removing it is also what makes this rebuild
    # self-contained: the official FP32 weights were never shipped, so any build
    # step that regenerated their artifact could not run from a clean checkout.
    # The measured result it produced, on the flagship query against these same 18
    # profiles: official FP32 BGE-M3 ranked the target second, the English-default
    # MiniLM control ranked it eleventh, and the shipped weight-only 8-bit graph
    # recovers the FP32 position.  Recorded here because it is no longer
    # re-derivable from shipped artifacts.

    runtime_model_dir = args.bge_model_dir.resolve().relative_to(ROOT).as_posix()
    manifest = {
        "schema_version": 1,
        "built_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "synthetic_only": True,
        "corpus_sha256": sha256_bytes(canonical_json_bytes(raw_records)),
        "golden_queries_sha256": sha256_bytes(canonical_json_bytes(golden_payload)),
        "gazetteer_sha256": sha256_bytes(canonical_json_bytes(gazetteer_payload)),
        "projection_contract_sha256": projection_contract_hash(),
        "concept_lexicon_sha256": sha256_bytes(canonical_json_bytes(concepts_payload)),
        "embedding": {
            "model_id": "BAAI/bge-m3",
            "upstream_revision": args.upstream_revision,
            "dimension": 1024,
            "pooling": "CLS",
            "normalized": True,
            "max_length": 512,
            "inference_batch_size": args.batch_size,
            "runtime": f"ONNX Runtime CPU {importlib.metadata.version('onnxruntime')}",
            "quantization": "8-bit weight-only MatMulNBits (asymmetric blocks of 128; embedding tables int8)",
            "runtime_model_dir": runtime_model_dir,
            "runtime_artifact_sha256": combined_model_hash(args.bge_model_dir),
            "profile_artifact": "bge_m3_profiles.npz",
            "profile_artifact_sha256": sha256_file(artifact_dir / "bge_m3_profiles.npz"),
        },
        "retrieval": {
            "dense": "exact cosine",
            "ranking": "dense only",
            "lexical": "BM25, used as a gate-opener and diagnostic; never ranks",
            "fusion": "removed - measured zero contribution on the golden set",
            "note": "POC default, not tuned on MAOZ data",
        },
        "confidence": confidence,
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
