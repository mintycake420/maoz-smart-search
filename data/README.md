# Data fixtures

`synthetic_profiles.json` is the immutable 18-profile corpus used by the shipped
search artifacts and acceptance measurements. The people and organisations in it
are **fictional**: the names are ordinary Israeli names, chosen across the
communities MAOZ works with so the demo reads like a real directory, and any
resemblance to a real person is coincidental. Every record still carries
`_synthetic: true` and a `003SYN` id, and projection rejects records without them.

`directory_inspired_synthetic_profiles.json` is a larger, schema-compatible
companion corpus. It contains 20 invented Hebrew profiles shaped for the same
Salesforce projection boundary. The public [MAOZ network directory](https://maoz-il.org/%d7%a8%d7%a9%d7%aa-%d7%9e%d7%a2%d7%95%d7%96/)
informed only these aggregate design choices:

- a compact public profile has a name, role and organisation, short professional
  background, and cohort;
- the network spans public, civic, health, education, local-government, business,
  technology, regional-development and community roles;
- search fixtures need enough variety to test cross-sector connections.

No public name, organisation, identifier, biography, image or one-to-one profile
mapping appears in the companion corpus. Every person and organisation label is
explicitly synthetic, and every record carries both `_synthetic: true` and a
`Source_Basis__c` non-copy marker.

**No profile in either corpus may contain every token of the flagship query.** The
companion corpus originally carried `Sector__c: "חינוך בלתי פורמלי"` — the flagship query
verbatim — on profile `003SYN000000111`. That profile is the correct answer, so the demo
still returned it at rank 1 with a strong tier, but it was doing so by exact string match
on a structured field rather than by bridging to <span dir="rtl">תנועות נוער</span>. The
headline capability would have been silently replaced by keyword lookup with nothing on
screen to show it. The value is now `חינוך חברתי־קהילתי`, and
`tests/test_normalization.py::test_no_corpus_profile_contains_the_flagship_query` reads
every shipped corpus and fails the build on a recurrence. The older probe next to it
compares two frozen strings and never opens a corpus, which is why it stayed green
throughout.

The companion file is deliberately not the runtime default. The shipped vectors and
confidence calibration are cryptographically bound to `synthetic_profiles.json`.
Promoting a different corpus therefore requires a complete artifact rebuild and a
fresh confidence calibration; changing only the JSON would make the application fail
closed, as intended. Profiles added live through the demo UI are a different thing:
they exist in memory only, never touch these files, and vanish on restart.

`judged_queries.json` is **not** a calibration input and must never become one. Its
queries and relevance grades were authored without reading `config/concepts.json`, which
makes it the only measurement in this repository that does not share an author with the
system. Its dev group measured perfect on the first run, so the acceptance and abstention
groups are clean held-out numbers — tuning anything against them would destroy the only
non-circular evidence here. `scripts/evaluate_judged.py` reads it; nothing else may.
