# MAOZ Hebrew Smart Search — Part B POC

A local, Hebrew-first semantic search over **synthetic Salesforce-shaped profiles only**.

## The problem

MAOZ holds information about network members in Salesforce. Staff need to find relevant
people for collaboration and consultation — not only by role or organisation, but by areas
of activity, experience, interests and shared challenges. Keyword search cannot do this,
because the words staff search with are not the words members wrote.

The brief's own success criterion is one example: searching
<span dir="rtl" lang="he">חינוך בלתי פורמלי</span> (*informal education*) must retrieve a
profile that says <span dir="rtl" lang="he">תנועות נוער</span> (*youth movements*), with no
shared words. This POC does that, and can also **decline to answer** when nothing matches
well — which matters more for staff trust than any single good result.

## Quick start

```powershell
git clone https://github.com/mintycake420/maoz-smart-search.git
cd maoz-smart-search

# One-time: the 580 MB encoder graph is attached to the release rather than committed.
gh release download v1.0 --pattern model.onnx --dir models/bge-m3-int8
# — or download model.onnx from the repository's Releases page by hand and save it to
#   models/bge-m3-int8/model.onnx

python -m pip install -e .
python -m maoz_search
```

Binds to `127.0.0.1:8765` and opens the local UI. Fetching the encoder is a one-time setup
step; **at runtime there is no model download, no hosted inference API and no outbound
network call at all.** After `pip install -e .` the console script `maoz-search` runs the
same thing. `--no-browser`, `--no-warmup`, `--port`, `--host` and `--model-dir` are
available.

Startup loads the encoder before opening the browser and prints how long it took — ONNX
session creation over the 580 MB graph, paid once at startup rather than on the first
search, so a live demo does not open onto a spinner. Budget for it: warm file cache takes
tens of seconds; a genuinely cold cache has been measured at **146 s** on the dev machine.
Start the server before the call, not during it.

> **If the encoder step is skipped**, startup stops and names the exact path it expected.
> A truncated or interrupted download is caught separately, by the manifest hash check.
> There is no network fallback in either case, which is the intended behaviour rather than
> a limitation.

<details>
<summary><strong>Why the encoder is not in the repository, and the Windows hazard around it</strong></summary>

`model.onnx` is 580 MB — past what belongs in a git object database, and past GitHub's
warning threshold. It is attached to the release instead. That choice is not purely
logistical: the manifest already pins a SHA-256 digest over *every* file in
`models/bge-m3-int8/`, so a manually placed file is verified before it is ever loaded. A
wrong or half-downloaded copy fails loudly at startup rather than silently degrading
retrieval quality, which is the failure mode that actually matters here.

The remaining encoder files *are* committed, and they carry the hazard. `.gitattributes`
marks that directory `-text`, and the line is load-bearing rather than cosmetic: under the
repository's `* text=auto` rule, a Windows clone (`core.autocrlf=true` is the platform
default) rewrites `tokenizer.json` and the JSON configs to CRLF, the directory digest
changes, and the encoder refuses to start with an artifact-hash error — on a machine where
nothing is actually wrong. It is invisible in the authoring checkout and does not reproduce
on macOS or Linux.

**If that pin is ever regenerated, commit `.gitattributes` on its own.** Re-adding the
model files while the working tree holds CRLF copies writes CRLF into the blobs and moves
the problem upstream instead of fixing it.

</details>

## Sharing the demo over a link

The POC binds to `127.0.0.1` and stays there. To let someone on another machine use it —
a reviewer on a call, or anyone holding the link — put a tunnel in front of the local
server rather than deploying it:

```powershell
winget install --id Cloudflare.cloudflared    # once

# terminal 1 — the POC, exactly as above
python -m maoz_search --no-browser

# terminal 2 — a public HTTPS front door for it
cloudflared tunnel --url http://127.0.0.1:8765
```

`cloudflared` prints a `https://<random-words>.trycloudflare.com` URL. It needs no
Cloudflare account, the hostname is new on every run, and it stops resolving when you stop
the process. **Start the server first and let it finish loading the encoder** — the tunnel
will happily publish a URL that 502s while ONNX session creation is still running.

**What this is and is not.** The link works from anywhere, but only while your machine is
awake and both processes are running. It is a link for a scheduled session, not a
deployment.

*What it does not change:* the encoder. Inference still runs on your CPU under ONNX Runtime
and no text is sent to a hosted model API, so the A.3 argument about **model hosting** holds
exactly as written.

*What it does change:* the transport, and this is a real boundary shift rather than a
detail. Profile text now leaves the machine by design — search results and the whole
directory travel to the visitor's browser, and whatever they type into the add form travels
back — and every byte passes through **Cloudflare's edge, which terminates TLS**. That is a
third party in the path who is simply not there when the UI is bound to `127.0.0.1`. Fine
for a synthetic corpus over a scheduled call; it would need an answer of its own before any
real record went near it. Do not read "local inference" as "nothing left the machine" —
those were the same sentence on localhost and are two different claims here.

**What the link exposes.** The POC has no authentication — [POC boundaries](#poc-boundaries)
says so, and this is where it bites. Anyone with the URL can search, read the whole
directory, add a profile and reset the index. That is acceptable *here* only because the
**shipped** corpus is synthetic and the hostname is an unguessable one-off: it is obscurity,
not access control, and it would be the wrong answer for real member data.

> **`_synthetic: true` does not make it so.** The add endpoint stamps that flag onto
> whatever arrives, so it records the *intent* of the demo, not the provenance of the text.
> A visitor who types a real person's name, phone and biography gets a real record —
> validated, embedded, searchable, and served from the public directory until someone
> resets. The guarantee that everything here is synthetic covers the corpus in git; it has
> never covered the remote write path, and sharing the URL is what makes that gap
> reachable. **Say out loud that the form is not for real people, and reset afterwards.**
> This is deliberately not enforced in code: a validator that claimed to detect "real"
> personal data would be wrong often enough to be worse than the honest warning.

Two guards make it survivable rather than safe:

- Runtime additions are capped at **25** per process (`SearchEngine.max_live_additions`).
- **איפוס ההדגמה**, in the profiles dialog, drops every addition and restores the sealed
  corpus without a restart — which matters because a restart costs another full encoder
  load.

The in-memory overlay is process-global, so whatever one visitor adds is in the index the
next visitor searches. That is the right behaviour for a shared demo and the wrong thing to
leave standing beforehand: **reset before a session that matters.**

## Methodology

Each validated Contact-shaped fixture becomes **four separately searchable aspects**: role
and organisation; professional trajectory; interests, areas of activity and values;
affiliations and cohort. Self-authored spans stay available to the dense leg but are
excluded from the lexical leg, so repetition cannot buy a keyword boost.

At query time:

1. Normalise Hebrew conservatively; add cautious prefix-clitic variants.
2. Limit candidates by the caller-supplied scope and hard filters.
3. Match only explicit triggers in the transparent, staff-owned concept vocabulary.
4. Embed the query with the repository-local BGE-M3 ONNX model; when a concept fires, add
   its already-precomputed expansion vectors.
5. Exact cosine search over all eligible aspect vectors, taking the best score across the
   original query and approved concept phrases.
6. Run an in-memory BM25 leg — **used only to open the confidence gate on an exact match
   and to record a diagnostic, never to rank.**
7. Keep the best aspect per profile by dense score. That single ranking drives gate, tier,
   score, evidence and concept bridge alike.
8. Apply a calibrated threshold, and **return no strong match** if nothing clears it.
9. Drop results outside a margin of the best result *for the intent they matched*, so a
   firing concept cannot pad the list with near misses.
10. Select a verbatim source span as evidence. **It does not generate a summary.**

**Ranking is dense-only, and that is a result rather than a preference.** BM25 fusion was
built, measured at zero contribution, and removed; the paired cross-encoder reranker was
built, measured as a regression on the flagship query (it moved the correct target from
rank 2 to rank 4), and rejected. Neither is packaged. The surviving ablation —
concept-bridge on versus off — is re-run on every invocation of `scripts/evaluate.py`,
which gates its own exit code on the bridge still earning its place.

`config/concepts.json` maps concepts to **phrases, never to people or profile IDs**. It
changes dense query formulation and leaves the lexical query untouched.

## Repository structure

The package is ordered as the pipeline runs:

| Stage | Module | Responsibility |
|---|---|---|
| Ingest | `maoz_search/domain.py` | Salesforce-shaped validation and the synthetic-only boundary |
| Preprocess | `maoz_search/projection.py` | Four-aspect projection, provenance-aware lexical exclusion |
| Preprocess | `maoz_search/normalization.py` | Hebrew normalisation, tokenisation, cautious clitic variants |
| Expand | `maoz_search/concepts.py` | Trigger-to-phrase concept bridges, no people or profile IDs |
| Embed | `maoz_search/embeddings.py` | Lazy, local-only BGE-M3 ONNX encoder |
| Index | `maoz_search/index.py` | Artifact verification and immutable in-memory index loading |
| Retrieve | `maoz_search/lexical.py` | Candidate-scoped BM25 — gate-opener and diagnostic only |
| Retrieve | `maoz_search/search.py` | Dense ranking, abstention, extractive evidence |
| Serve | `maoz_search/web.py`, `templates/`, `static/` | Local Flask API and Hebrew RTL demo |

Supporting directories:

| Path | Contents |
|---|---|
| `scripts/build_artifacts.py` | **Maintainer only.** Rebuilds vectors, calibration and manifest |
| `scripts/evaluate.py` | Golden-set evaluation and the concept-bridge ablation |
| `scripts/evaluate_judged.py` | Independently judged evaluation |
| `config/` | `concepts.json` (staff-owned vocabulary), `gazetteer.json` (acronym/entity aliases) |
| `data/` | Synthetic corpora and query sets — see [`data/README.md`](data/README.md) |
| `data/artifacts/` | Precomputed vectors + `manifest.json`, the hash that binds them together |
| `models/bge-m3-int8/` | The local encoder. `model.onnx` arrives from Releases — see [Quick start](#quick-start) |
| `tests/` | 58 contract and regression tests |
| `docs/` | The Part A.1–A.3 and Part B deliverable documents |

> **Four files are cryptographically bound to the shipped vectors:**
> `config/concepts.json`, `config/gazetteer.json`, `data/synthetic_profiles.json` and
> `data/golden_queries.json`. `ProfileIndex.load()` hashes each against `manifest.json`
> before the encoder will start, so editing any of them fails the application closed by
> design — changing them requires the explicit rebuild below, not a text edit.
>
> **`data/judged_queries.json` and `data/directory_inspired_synthetic_profiles.json` are
> not bound.** Nothing reads them at runtime; only `scripts/evaluate_judged.py` does.
> Editing either silently changes that evaluation's results instead of failing closed,
> which is the more dangerous failure of the two — treat them as frozen by convention.

## Current results

These are the figures this repository stands behind, re-measured 2026-07-21 after the
corpus was renamed and resealed — which re-fitted the confidence gate from 0.474 to 0.495
and moved several numbers below. Everything else is exploratory. Every figure here is
reproducible from a clean checkout with the two commands under
[Reproducing](#reproducing); none is asserted from memory.

### Golden set — `python scripts/evaluate.py`

| Group | Result | Evidential weight |
|---|---|---|
| Acceptance — rank 1 | **8/8** | Independent of threshold fitting, but every intent is configured in `config/concepts.json` — a regression check against the system's own vocabulary |
| Acceptance — abstention | **7/7** | Out-of-domain queries return the explicit no-match state. Includes a deliberately health-adjacent probe, since the corpus has health profiles and a near-domain miss is the harder case |
| Held-out — rank 1 | **7/9** | The only generalisation signal here. Targets have no concept entry and were excluded from the fit. One of the two misses (`heldout_youth_employment`) abstains at the stricter gate rather than mis-ranking |
| Calibration fit | 4/4 positives, 5/5 negatives | **Not evidence.** These selected `dense_threshold`; the gate was fitted so they pass. Regression only |

**The three groups are deliberately not summed.** An earlier version reported "12/12" and
"6/6" — but 4 of those 12 and 5 of those 6 were the examples used to fit the gate, which
made the headline partly circular.

The same run reports the **concept-bridge ablation: 8/8 with it, 6/8 without**, and a
`rank_only_effect` block isolating rank from gating — improved on 2, unchanged on 6,
regressed on 0. `earns_its_place` gates the exit code, so the justification cannot quietly
stop being true.

### Independently judged set — `python scripts/evaluate_judged.py`

The one measurement whose queries and relevance grades do **not** share an author with
`config/concepts.json`. 23 queries over the 20-profile companion corpus, which was not
touched by the rename.

**The headline: every one of the 18 judged primaries sits at un-gated dense rank 1 —
18/18, enforced by the evaluator's exit code.** That is the encoder working unaided (the
concept vocabulary fires on none of these queries), on queries written by someone who had
not seen the answer key. It is the one figure here that gate movement cannot touch.

| Group | MRR | Hit@3 | nDCG@5 | Recall@5 |
|---|---|---|---|---|
| dev (5) | 0.800 | 4/5 | 0.800 | 0.800 |
| acceptance (13) | 0.769 | 10/13 | 0.655 | 0.577 |
| abstention (5) | — | — | — | **5/5 correct** |

The gated metrics are **lower than the 2026-07-20 run** (acceptance MRR was 0.923), and
the entire drop has one cause, verified query by query: this harness deliberately applies
the runtime corpus's fitted gate to a corpus it was never calibrated on, and the resealed
gate is stricter. `dev_04`, `acc_03` and `acc_05` (scores 0.477–0.480) joined `acc_06`
(0.389) as **correct-at-rank-1 answers the gate abstains on**. Nothing was tuned in
response — the finding is the deliverable: **a single absolute confidence threshold does
not transfer across corpora**, so per-corpus calibration is a v1 requirement, not a
refinement.

What was deliberately *not* done about it: the gate was not re-fitted against this set.
Re-fitting the threshold using the one evaluation whose queries were written by someone
else would convert the only independent measurement here into another configured result —
exactly the circularity the three-way split above exists to avoid. The floor in
`scripts/evaluate_judged.py` was re-baselined instead, so the tripwire still arms against
future silent drift rather than failing permanently against a deliberate recalibration.

### Measured local runtime

On an Intel Core i5-9400F, CPU only, warm flagship medians have been measured between
**0.74 s and 2.37 s** across campaigns — the figure moves with machine load (the top of
that range was measured while another evaluation saturated the CPU), so treat it as a
range rather than a number. First-query cost is dominated by ONNX session creation:
tens of seconds warm, and **146 s measured once on a genuinely cold file cache**. These
measure this ~603 MB bundle under ONNX Runtime CPU 1.26.0; they are not latency promises
for another machine or corpus.

## Three screen-share beats

1. **The flagship match.** Search <span dir="rtl" lang="he">חינוך בלתי פורמלי</span>. The
   concept bridge ranks נועה ברק first, labels her
   <span dir="rtl" lang="he">חזקה</span>, and shows the original evidence
   <span dir="rtl" lang="he">תנועות נוער</span> — the profile and query share no canonical
   token.
2. **Add a person live, then find them.** Open הוספת פרופיל, invent someone — a fintech
   M&A director who is also a national bowling champion, in Hebrew or with English mixed
   in — and submit. The same fail-closed validation, the same four-aspect projection and
   the same local encoder index them in seconds, in memory only. Then search a paraphrase
   of what you wrote and watch them come back with a verbatim evidence span. Nothing on
   this path is precomputed, which is the point: the demo is not leaning on fixtures its
   authors prepared. (Search is by *content*, not by name — person names are deliberately
   not embedded, so searching the new person's name abstains. And phrase the need as a
   fuller clause: two-word queries often score inside the gate's noise band, close enough
   to the 0.495 threshold that a correct match can land under it — and the abstention
   state says so when it happens rather than guessing.)
3. **An honest refusal.** The out-of-domain query returns `no_strong_match` with no padded
   results.

Beat 2 replaced a frozen MiniLM/FP32 model comparison on 2026-07-21, on the grounds that a
live addition argues generalisation more directly than a comparison over fixtures this
repository's own author wrote. That superseded measurement — on the flagship query, official
FP32 BGE-M3 ranked the target **second**, the shipped weight-only 8-bit graph recovers that
same position, and an English-default MiniLM control ranked it **eleventh** — is quoted here
as a recorded result. Its artifacts were removed with the feature, so unlike every figure
above it is **not** re-derivable from this checkout. Treat it accordingly.

## Reproducing

```powershell
python -m pytest                        # 58 tests
python scripts/evaluate.py              # golden set + concept ablation
python scripts/evaluate_judged.py       # independently judged set
```

All three need `models/bge-m3-int8/model.onnx` in place first — see
[Quick start](#quick-start). They exercise the real local encoder rather than a stub, so
without it the integration tests and both evaluators fail immediately, naming the path they
expected.

Both evaluators run with outbound sockets blocked and are deterministic.

Tests cover the synthetic-only projection guard, four-aspect projection, exclusion of
restricted and self-authored text from the lexical projection, Hebrew punctuation and
prefix handling, RTL/self-contained HTML, safe DOM rendering, and the runtime
add-a-profile flow — including that an addition never mutates the sealed base index and
is validated by the same fail-closed path as the corpus. The integration and API checks
warm the real local encoder with outbound connections blocked.

**Seven tests exist specifically to stop review-found defects from returning:** the evidence
highlight must survive deletion of the golden set; the headline message must agree with the
tiers under it; the lexical gate must ignore ordinary Hebrew sentences; calibration must
include a positive with no concept coverage; a result labelled strong must clear the bar on
its own score rather than a sibling aspect's; evidence must be a verbatim span of the
profile shown; and **no `heldout` query may have a `config/concepts.json` entry** — that
last one matters most, because adding a held-out intent to the vocabulary would silently
convert the only generalisation number here into another configured result.

## Importable API

The UI and Python callers use the same contract:

```python
from maoz_search import SearchEngine

engine = SearchEngine.from_default()
response = engine.search(
    "חינוך בלתי פורמלי",
    filters={"region": "צפון"},
    allowed_profile_ids={"003SYN000000001", "003SYN000000002"},
    top_k=3,
)

if response.status == "no_strong_match":
    print(response.message)
else:
    for result in response.results:
        print(result.name, result.winning_aspect_label, result.evidence_span)
```

`engine.add_profile(record)` indexes one additional synthetic record at runtime — the
same fail-closed validation and four-aspect projection as the corpus, embedded live by
the local encoder. Additions live in a swapped in-memory snapshot: the sealed artifacts
on disk never change, concurrent searches keep a consistent view, and everything added
vanishes on restart. The web UI's הוספת פרופיל form calls `POST /api/profiles`, which
builds the record server-side (a visitor supplies fields, never an id or a synthetic
marker), and `GET /api/profiles` returns the browsable directory — base corpus plus
live additions — that backs the UI's פרופילים באינדקס panel and the "full profile"
expansion on every result card.

**How the form maps to Salesforce, and what each field feeds.** Form inputs map 1:1
onto the MAOZ-confirmed Contact shape: name → `FirstName`/`LastName`, תפקיד → `Title`,
ארגון → `Account.Name`, אימייל → `Email`, טלפון → `Phone`, and the custom fields
`Sector__c`, `Region__c`, `Cohort__c`, `Experience__c`, `Areas_of_Activity__c`,
`Interests__c`, `Values__c`, `Affiliations__c`, plus the standard free-form
`Description`. **Every narrative field —
interests and values included — is embedded and ranked semantically**; the only
field-level distinction is provenance-driven and affects the *lexical keyword-rescue
leg only*: self-described text (the Description blurb) cannot buy an exact-keyword
rescue, an anti-stuffing control from Part A.3, not a relevance judgment. Form-entered
fields are stamped `demo_added` (treated as staff-entered CRM data, trusted on both
legs); `Description` keeps `self_described`, mirroring its corpus counterpart.

**`Email`, `Phone` and `Region__c` are carried on the record and never enter the
index.** The projection is an **allow-list** — `_ASPECT_SPECS` names the ten fields that
become searchable text — so a field it does not name cannot reach an aspect, an
embedding or the lexical leg. Contact details are therefore disclosed only *after* a
match has been made on other grounds, which is Part A.3's data-minimisation position
demonstrated rather than asserted: searching an indexed person's email address returns
`no_strong_match`, because that address was never encoded. Two regression tests hold the
line — one on a fixture, one reading the corpus that actually ships. (`Region__c` is
excluded for a different reason: it serves the structured filter, where an exact facet
beats a fuzzy vector.) Every shipped address uses the RFC 2606-reserved `.example` TLD,
which can never be registered, and phone numbers are a sequential `050-000-00NN`
placeholder pattern rather than plausible Israeli mobiles — a plausible number belongs
to somebody. Known fidelity gaps against a real org, stated rather than
papered over: real `Areas`/`Interests` are likely multi-select picklists with a
controlled vocabulary rather than free text, `Account` is a lookup rather than a
string, and production provenance comes from Salesforce field history rather than a
server-side stamp.

`allowed_profile_ids` is an injectable scope applied **before** ranking. It demonstrates the
retrieval boundary; it is **not** a substitute for the Salesforce revalidation and database
row-level security specified for production. The web POC exposes sector and region facets
and has no authentication.

## POC boundaries

Part A describes the production direction. Part B keeps the same projection and retrieval
seams but narrows the operational surface:

| Production direction | This POC |
|---|---|
| Salesforce extraction, deltas and erasure | Versioned synthetic JSON fixtures only |
| PostgreSQL + pgvector inside a permission-bound query | In-process NumPy exact scan plus in-memory BM25 |
| Per-request Salesforce revalidation and database RLS | Caller-injected `allowed_profile_ids`; no identity system |
| Local cross-encoder reranking | Measured on the flagship, where it moved the target from rank 2 to rank 4; rejected and not packaged |
| Operational audit, reconciliation, restore and lifecycle jobs | Out of scope |
| Real-data confidence and relevance calibration | Synthetic hypotheses only; must be replaced before any pilot |

The NumPy adapter is a correctness-oriented exact baseline, **not** evidence about
PostgreSQL latency or scale. The concept vocabulary is a POC mechanism whose entries require
MAOZ ownership and judged-query review before any real-data pilot.

## Synthetic-only boundary

Every fixture must carry `_synthetic: true` and each Contact ID must begin with `003SYN`,
or projection raises `ProfileValidationError`. The restricted-field canary is excluded by
the allow-listed projection. These are take-home guardrails, not production DLP.

The local service binds to loopback, returns extractive source text, strips internal numeric
scores from public responses, and sets restrictive browser headers. **No profile or query
text is sent to an embedding API or an instruction-following model.** Do not place real MAOZ
member data in this repository, and do not use `--host` to expose the demo without
production-grade authentication and transport controls.

## Rebuilding artifacts (maintainer operation)

Never part of startup. Since the model-comparison removal (2026-07-21) the rebuild is
**self-contained**: it needs only the repository-local BGE-M3 ONNX directory and the
verified upstream revision, and runs from a clean `pip install -e .` checkout.

```powershell
$upstreamRevision = (Get-Content data/artifacts/manifest.json | ConvertFrom-Json).embedding.upstream_revision
python scripts/build_artifacts.py `
  --bge-model-dir models/bge-m3-int8 `
  --batch-size 32 `
  --upstream-revision $upstreamRevision
```

This rewrites the 72 aspect vectors, source vectors, confidence calibration metadata and
`manifest.json`, and hashes the corpus, golden queries, gazetteer, projection contract and
concept vocabulary into it. Self-contained does not mean casual: the rebuild **re-fits the
confidence gate**, so it moves numbers in documents. After any rebuild, re-run
`scripts/evaluate.py` and `scripts/evaluate_judged.py` and update every figure they
contradict.

A gazetteer edit does not mathematically require new dense vectors, but it changes the
lexical gate and therefore invalidates the sealed manifest; use the explicit rebuild to
recalibrate and reseal.

## The deliverables

Open the four HTML documents in a browser — they are self-contained, with no external
assets, and print cleanly to PDF.

| Document | What it answers |
|---|---|
| [`docs/part-a1-discovery-questions.html`](docs/part-a1-discovery-questions.html) | A.1 — five questions for the network managers |
| [`docs/part-a2-architecture.html`](docs/part-a2-architecture.html) | A.2 — solution, data flow, and the AI-tools disclosure |
| [`docs/part-a3-information-security.html`](docs/part-a3-information-security.html) | A.3 — security and authorisation design |
| [`docs/part-b-poc.html`](docs/part-b-poc.html) | B — POC rationale and its trace to B.1–B.3 |
| [`ASSIGNMENT.md`](ASSIGNMENT.md) | The verbatim brief and the deliverables checklist |
| [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | Bundled model provenance and licences |

The AI-tools disclosure required by the brief lives in Parts A.2 and A.3 and is maintained
as a living document.
