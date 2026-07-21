"""Local-only embedding backends.

The runtime deliberately has no download branch.  The ONNX model must already be
present on disk; otherwise search fails with an actionable local-artifact error
naming the path it expected.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


class EncoderUnavailableError(RuntimeError):
    pass


class TextEncoder(Protocol):
    model_id: str
    dimension: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


class OnnxBgeEncoder:
    """Lazy CPU encoder for a local quantized BGE-M3 ONNX snapshot."""

    model_id = "BAAI/bge-m3"
    dimension = 1024

    def __init__(
        self,
        model_dir: Path,
        *,
        max_length: int = 512,
        batch_size: int = 32,
        expected_artifact_hash: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.batch_size = max(1, int(batch_size))
        self.expected_artifact_hash = expected_artifact_hash
        self._tokenizer = None
        self._session = None
        self._input_names: frozenset[str] = frozenset()
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def _load(self) -> None:
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            model_path = self.model_dir / "model.onnx"
            # The 580 MB encoder graph ships as a release asset rather than in the
            # repository, so a fresh clone legitimately arrives without it.  That is a
            # one-time setup step, never a runtime download: nothing on this path
            # reaches the network, and the manifest digest below is what makes a
            # manually placed file safe to trust.
            if not model_path.is_file():
                raise EncoderUnavailableError(
                    f"Local BGE-M3 artifact is missing. Expected {model_path}. "
                    "Download model.onnx from this repository's Releases page and place "
                    "it at that path. Runtime downloads are intentionally disabled."
                )
            if self.expected_artifact_hash:
                digest = hashlib.sha256()
                for path in sorted(item for item in self.model_dir.iterdir() if item.is_file()):
                    digest.update(path.name.encode("utf-8"))
                    file_digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        while chunk := handle.read(1024 * 1024):
                            file_digest.update(chunk)
                    digest.update(file_digest.hexdigest().encode("ascii"))
                if digest.hexdigest() != self.expected_artifact_hash:
                    raise EncoderUnavailableError(
                        f"The contents of {self.model_dir} do not match the digest sealed "
                        "into data/artifacts/manifest.json. The likeliest cause is an "
                        "incomplete or interrupted download of model.onnx: check its size "
                        "against the release asset and fetch it again. Note that the digest "
                        "covers every file in the directory, so a stray extra file there "
                        "will also trip this."
                    )
            try:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                import onnxruntime as ort
                from transformers import AutoTokenizer
            except ImportError as exc:  # pragma: no cover - depends on evaluator environment
                raise EncoderUnavailableError(
                    "Local inference requires onnxruntime and transformers; no hosted fallback is allowed"
                ) from exc

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                local_files_only=True,
                use_fast=True,
            )
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._input_names = frozenset(item.name for item in self._session.get_inputs())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        self._load()
        assert self._tokenizer is not None and self._session is not None

        # The shipped graph uses weight-only MatMulNBits quantization: activations
        # remain floating-point, and its batch-vs-single invariance is regression
        # measured before packaging. Runtime traffic is normally one query; the
        # larger bound speeds the explicit offline artifact rebuild.
        rows: list[np.ndarray] = []
        for offset in range(0, len(texts), self.batch_size):
            batch = list(texts[offset : offset + self.batch_size])
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            inputs = {
                name: np.asarray(value, dtype=np.int64)
                for name, value in encoded.items()
                if name in self._input_names
            }
            outputs = self._session.run(None, inputs)
            vectors = np.asarray(outputs[0][:, 0, :], dtype=np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.clip(norms, 1e-12, None)
            rows.extend(vectors)
        return np.ascontiguousarray(np.vstack(rows), dtype=np.float32)
