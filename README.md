# maoz-smart-search

Hebrew semantic search over member profiles. It finds people by *what they do*, not by
keyword — a search for **חינוך בלתי פורמלי** ("informal education") returns a profile that
says **תנועות נוער** ("youth movements"), which shares no words with the query. When nothing
matches well it says so, rather than padding the results with weak guesses.

Inference is local: profile and query text are embedded on the CPU with a bundled ONNX
model. No hosted model API, no embedding service, no outbound network call at runtime.

All profile data in this repository is synthetic.

## Running it

```powershell
gh release download v1.0 --pattern model.onnx --dir models/bge-m3-int8   # 580 MB encoder
python -m pip install -e .
python -m maoz_search
```

The encoder is attached to the [v1.0 release](../../releases/tag/v1.0) rather than committed
(it is 580 MB); the release notes cover setup and download verification. The service binds to
`127.0.0.1:8765` and opens the UI. `--host`, `--port`, `--model-dir`, `--no-browser` and
`--no-warmup` are available.

```powershell
python -m pytest                    # 58 tests
python scripts/evaluate.py          # golden set + concept-bridge ablation
python scripts/evaluate_judged.py   # independently judged set
```

## How a query flows

A query passes through the modules in `maoz_search/` roughly in file order. Each stage owns
one concern:

| Stage | Module | What it does |
|---|---|---|
| Ingest | `domain.py` | Validates a Salesforce-shaped record and projects it to a `Profile`. Rejects anything not marked synthetic — fails closed. |
| Project | `projection.py` | Splits each `Profile` into four searchable `Aspect`s. Contact details are excluded structurally, by an allow-list. |
| Normalize | `normalization.py` | Conservative Hebrew normalisation, tokenisation, and cautious prefix-clitic variants. |
| Expand | `concepts.py` | Bridges query vocabulary to domain phrases (`informal education` → `youth movements`) via a transparent, staff-owned lexicon. |
| Encode | `embeddings.py` | Embeds text with the local BGE-M3 ONNX model. Lazy, CPU-only, offline. |
| Index | `index.py` | Loads the sealed vectors and verifies every hash before serving. |
| Rank | `lexical.py`, `search.py` | Dense cosine ranking with abstention and extractive evidence. |
| Serve | `web.py` | Flask JSON API and the Hebrew RTL UI. |

## The classes, module by module

### `domain.py` — the Salesforce boundary
- **`Profile`** — a frozen, validated Contact record. `Profile.from_salesforce(record)` is the
  only way to make one; it rejects any record missing `_synthetic: true` or a `003SYN` id
  prefix, so production data cannot be indexed by accident. Contact fields (`email`, `phone`)
  live on the record but are deliberately never projected into search.
- **`Aspect`** — one of the four searchable facets of a profile, carrying separate
  `embedding_text` and `lexical_text` plus the `SourceSpan`s it came from.
- **`SourceSpan`** — one field's contribution to an aspect, tagged with provenance and whether
  it may participate in keyword matching.
- **`ProfileValidationError`** — raised when a record crosses the synthetic-only boundary.

### `projection.py` — profile → four aspects
- **`project_profile` / `project_profiles`** — turn a `Profile` into its `Aspect`s using
  `_ASPECT_SPECS`: *role & organisation*, *trajectory*, *interests & values*, *affiliations &
  cohort*. Self-described text is embedded but excluded from the keyword leg, so repetition
  can't buy a lexical boost.
- **`projection_contract_hash`** — a digest of the projection rules, sealed into the manifest
  so the shipped vectors can't silently drift from the code that produced them.

### `normalization.py` — Hebrew text handling
- **`normalize_text`** — NFKC, niqqud stripping, quote canonicalisation.
- **`canonical_tokens` / `lexical_tokens`** — tokenisation, the latter applying gazetteer
  aliases.
- **`clitic_variants`** — cautious prefix-clitic expansion (the ו/ב/ל/כ/מ/ש/ה prefixes).

### `concepts.py` — the concept bridge
- **`ConceptLexicon`** — loads `config/concepts.json`, enforcing that entries contain phrases
  only, never profile identifiers. `expanded_queries(query)` adds domain phrases when a trigger
  fires; it changes only how the *dense* query is formed and never touches ranking directly.
- **`ConceptMatch`** — one fired concept: its trigger and the expansion phrases it contributes.

### `embeddings.py` — the encoder
- **`OnnxBgeEncoder`** — lazy CPU encoder over the local quantised BGE-M3 graph. Verifies a
  SHA-256 digest of the model directory before loading; there is no download fallback.
- **`TextEncoder`** — the `Protocol` the engine depends on, so the encoder can be substituted.
- **`EncoderUnavailableError`** — raised when the model is missing or its hash doesn't match.

### `index.py` — the sealed index
- **`ProfileIndex`** — `ProfileIndex.load()` reads the corpus, projects it, loads the
  precomputed vectors, and checks the corpus, golden queries, gazetteer, concept vocabulary,
  projection contract and vector artifact each against `manifest.json`. Any mismatch fails the
  load rather than serving stale results.

### `lexical.py` — keyword leg
- **`LexicalIndex`** — an in-memory BM25 index used *only* to open the confidence gate on an
  exact match and to record a diagnostic. It never contributes to the ranking score.

### `search.py` — the engine
- **`SearchEngine`** — the public entry point. `SearchEngine.from_default()` builds one from the
  shipped artifacts; `search(query, filters, allowed_profile_ids, top_k)` runs the pipeline and
  returns a `SearchResponse`. `add_profile(record)` indexes a synthetic record live into an
  in-memory overlay (the sealed base never changes); `reset()` drops the overlay.
- **`SearchResult`** — one ranked person: name, role, the winning aspect, a verbatim
  `evidence_span`, the confidence tier, and the concept bridge if one fired. Internal scores are
  stripped from the public form.
- **`SearchResponse`** — the whole answer: `status` (`ok` or `no_strong_match`), a message, the
  results, and diagnostic `meta`.

### `web.py` — the service
- **`create_app(engine)`** — builds the Flask app. Routes: `GET /` (UI), `GET /api/meta`,
  `GET /api/health`, `POST /api/search`, `GET`/`POST /api/profiles`, `POST /api/reset`.

### `__main__.py` — the CLI
- **`python -m maoz_search`** — loads the encoder, warms it with one query so the UI doesn't
  open onto a spinner, and serves the app.

## Layout

```
maoz_search/   the package (classes above) + templates/ and static/ for the UI
scripts/       build_artifacts.py (rebuild vectors + manifest), evaluate*.py (the two harnesses)
config/        concepts.json (concept vocabulary), gazetteer.json (entity aliases)
data/          synthetic corpora, query sets, and data/artifacts/ (sealed vectors + manifest)
models/        bge-m3-int8/ — tokenizer & config committed; model.onnx from the release
tests/         58 contract and regression tests
```

Four files are hashed into `data/artifacts/manifest.json` — `config/concepts.json`,
`config/gazetteer.json`, `data/synthetic_profiles.json` and `data/golden_queries.json` — and
verified at startup, so editing any of them fails the load by design. Regenerate them with
`scripts/build_artifacts.py`, which re-seals the manifest, rather than by hand.

## Data boundary

Every record must carry `_synthetic: true` and a `003SYN` contact-id prefix or projection
raises `ProfileValidationError`. The service binds to loopback, returns extractive source text
(never a generated summary), and strips internal scores from public responses. Don't place real
member data here, and don't use `--host` to expose the demo without real authentication in
front of it.
