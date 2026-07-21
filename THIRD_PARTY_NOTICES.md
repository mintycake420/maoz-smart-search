# Third-party model notices

This file records the model artifacts distributed with, or used to prepare, the
MAOZ Part B proof of concept. It does not replace the licence files or terms supplied
by the respective upstream projects.

## BAAI/bge-m3

- Use in this repository: primary profile, source-span, concept-phrase and runtime
  query embeddings.
- Distributed runtime files: the local ONNX/tokenizer snapshot under
  `models/bge-m3-int8/`. The tokenizer, configs and licence are in the repository;
  the 580,393,844-byte `model.onnx` is attached to the release instead, so the
  directory measures ~22 MB as cloned and 602,549,616 bytes once assembled. The
  manifest records a combined artifact hash over all of it, verified before load.
- Distributed derived files: 72 profile-aspect vectors plus source-span and
  concept-phrase vectors in `data/artifacts/bge_m3_profiles.npz`. Official FP32
  model weights are not distributed. A narrow official-FP32 comparison artifact
  (`bge_fp32_demo.npz`) was distributed until the frozen model-comparison demo was
  removed in July 2026. It is not distributed here, and the comparison it supported is
  therefore a recorded measurement rather than a reproducible one.
- Upstream revision: recorded in `data/artifacts/manifest.json`.
- Local modification recorded by the manifest: 8-bit weight-only asymmetric
  `MatMulNBits` quantisation in blocks of 128, with int8 embedding tables. The graph
  runs on ONNX Runtime CPU 1.26.0.
- Licence: MIT. The upstream licence text and copyright notice are retained at
  `models/bge-m3-int8/LICENSE`.

## sentence-transformers/all-MiniLM-L6-v2 — evaluated, no longer distributed

- Use in this repository: an illustrative English-default control, measured once
  against the frozen flagship query (target rank 11 versus BGE-M3's rank 2); never
  primary ranking.
- Distributed files: none. Derived vectors were previously shipped in
  `data/artifacts/minilm_demo.npz`; the frozen model-comparison demo it backed was
  removed in July 2026, and the rank-11 result above is a recorded measurement from
  that run rather than something re-derivable from the artifacts shipped here.
- Licence: the upstream model card declares Apache-2.0.

## BAAI/bge-reranker-v2-m3 — evaluated, not distributed

- Use in this repository: a local experiment tested the paired cross-encoder against
  the frozen flagship query. It worsened the target rank and was rejected.
- Distributed files: none.
- Licence: Apache-2.0, verified per the reranker repository rather than inferred from
  the BGE-M3 embedding-model licence.

Python libraries named in `pyproject.toml` are installed as dependencies rather than
vendored into this repository. Their own upstream licences continue to apply.
