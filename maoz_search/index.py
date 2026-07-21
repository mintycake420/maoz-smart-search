"""Load the synthetic corpus and its immutable exact-search artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import ArtifactMismatchError, canonical_json_bytes, load_json, sha256_bytes, sha256_file
from .concepts import ConceptLexicon
from .domain import Aspect, Profile
from .lexical import LexicalIndex
from .normalization import normalize_text
from .projection import project_profiles, projection_contract_hash


@dataclass(slots=True)
class ProfileIndex:
    root: Path
    raw_records: list[dict[str, Any]]
    profiles: tuple[Profile, ...]
    aspects: tuple[Aspect, ...]
    aspect_vectors: np.ndarray
    source_keys: tuple[str, ...]
    source_vectors: np.ndarray
    concept_phrase_keys: tuple[str, ...]
    concept_phrase_vectors: np.ndarray
    lexical: LexicalIndex
    concepts: ConceptLexicon
    manifest: dict[str, Any]
    golden_queries: tuple[dict[str, Any], ...]
    # Retained parsed gazetteer aliases so a runtime profile addition can rebuild
    # the in-memory BM25 leg under the same alias rules the sealed corpus used.
    gazetteer_aliases: dict[str, Any]

    @classmethod
    def load(cls, root: Path | None = None) -> "ProfileIndex":
        root = Path(root or Path(__file__).resolve().parents[1]).resolve()
        data_dir = root / "data"
        artifact_dir = data_dir / "artifacts"

        raw_records = load_json(data_dir / "synthetic_profiles.json")
        if not isinstance(raw_records, list):
            raise ArtifactMismatchError("synthetic_profiles.json must contain an array")
        profiles = tuple(Profile.from_salesforce(record) for record in raw_records)
        aspects = project_profiles(profiles)

        gazetteer_payload = load_json(root / "config" / "gazetteer.json")
        aliases = gazetteer_payload.get("aliases", {})
        if not isinstance(aliases, dict):
            raise ArtifactMismatchError("gazetteer aliases must be an object")
        lexical = LexicalIndex([aspect.lexical_text for aspect in aspects], aliases)
        concepts_path = root / "config" / "concepts.json"
        concepts_payload = load_json(concepts_path)
        concepts = ConceptLexicon.load(concepts_path)
        golden_payload = load_json(data_dir / "golden_queries.json")

        manifest = load_json(artifact_dir / "manifest.json")
        expected_corpus_hash = sha256_bytes(canonical_json_bytes(raw_records))
        if manifest.get("corpus_sha256") != expected_corpus_hash:
            raise ArtifactMismatchError("Profile vectors do not match the synthetic corpus")
        expected_golden_hash = sha256_bytes(canonical_json_bytes(golden_payload))
        if manifest.get("golden_queries_sha256") != expected_golden_hash:
            raise ArtifactMismatchError("Golden queries do not match calibration and comparison artifacts")
        expected_gazetteer_hash = sha256_bytes(canonical_json_bytes(gazetteer_payload))
        if manifest.get("gazetteer_sha256") != expected_gazetteer_hash:
            raise ArtifactMismatchError("Gazetteer does not match the calibrated lexical gate")
        if manifest.get("projection_contract_sha256") != projection_contract_hash():
            raise ArtifactMismatchError("Profile vectors do not match the projection contract")
        expected_concepts_hash = sha256_bytes(canonical_json_bytes(concepts_payload))
        if manifest.get("concept_lexicon_sha256") != expected_concepts_hash:
            raise ArtifactMismatchError("Confidence calibration does not match the concept lexicon")

        profile_artifact_path = artifact_dir / str(
            manifest["embedding"].get("profile_artifact", "bge_m3_profiles.npz")
        )
        if sha256_file(profile_artifact_path) != manifest["embedding"].get("profile_artifact_sha256"):
            raise ArtifactMismatchError("Profile vector artifact hash does not match the manifest")
        with np.load(profile_artifact_path, allow_pickle=False) as artifact:
            artifact_aspect_keys = tuple(str(value) for value in artifact["aspect_keys"].tolist())
            expected_aspect_keys = tuple(aspect.key for aspect in aspects)
            if artifact_aspect_keys != expected_aspect_keys:
                raise ArtifactMismatchError("Aspect order or identity changed; rebuild vectors")
            aspect_vectors = np.asarray(artifact["aspect_vectors"], dtype=np.float32)
            source_keys = tuple(str(value) for value in artifact["source_keys"].tolist())
            source_vectors = np.asarray(artifact["source_vectors"], dtype=np.float32)
            concept_phrase_keys = tuple(str(value) for value in artifact["concept_phrase_keys"].tolist())
            concept_phrase_vectors = np.asarray(artifact["concept_phrase_vectors"], dtype=np.float32)

        expected_shape = (len(aspects), int(manifest["embedding"]["dimension"]))
        if aspect_vectors.shape != expected_shape:
            raise ArtifactMismatchError(f"Unexpected aspect vector shape: {aspect_vectors.shape}")
        if source_vectors.shape != (len(source_keys), expected_shape[1]):
            raise ArtifactMismatchError("Unexpected source vector shape")
        if concept_phrase_vectors.shape != (len(concept_phrase_keys), expected_shape[1]):
            raise ArtifactMismatchError("Unexpected concept vector shape")
        expected_concept_phrases = {
            normalize_text(str(expansion))
            for item in concepts_payload["concepts"]
            for expansion in item["expansions"]
        }
        if {normalize_text(key) for key in concept_phrase_keys} != expected_concept_phrases:
            raise ArtifactMismatchError("Concept vectors do not match the staff-owned vocabulary")
        if (
            not np.all(np.isfinite(aspect_vectors))
            or not np.all(np.isfinite(source_vectors))
            or not np.all(np.isfinite(concept_phrase_vectors))
        ):
            raise ArtifactMismatchError("Embedding artifacts contain non-finite values")

        golden_queries = tuple(golden_payload.get("queries", ()))
        return cls(
            root=root,
            raw_records=raw_records,
            profiles=profiles,
            aspects=aspects,
            aspect_vectors=aspect_vectors,
            source_keys=source_keys,
            source_vectors=source_vectors,
            concept_phrase_keys=concept_phrase_keys,
            concept_phrase_vectors=concept_phrase_vectors,
            lexical=lexical,
            concepts=concepts,
            manifest=manifest,
            golden_queries=golden_queries,
            gazetteer_aliases=dict(aliases),
        )

    @property
    def profiles_by_id(self) -> dict[str, Profile]:
        return {profile.profile_id: profile for profile in self.profiles}

    @property
    def sources_by_key(self) -> dict[str, np.ndarray]:
        return {key: self.source_vectors[index] for index, key in enumerate(self.source_keys)}

    @property
    def concept_vectors_by_text(self) -> dict[str, np.ndarray]:
        return {
            normalize_text(key): self.concept_phrase_vectors[index]
            for index, key in enumerate(self.concept_phrase_keys)
        }

