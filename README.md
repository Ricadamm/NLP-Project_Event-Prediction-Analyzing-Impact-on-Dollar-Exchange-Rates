# Global Geopolitical Event Prediction: Analyzing Impact on Dollar Exchange Rates

## Project Overview
An end-to-end NLP and analytical pipeline to investigate whether global geopolitical news significantly influences and helps predict fluctuations in the US Dollar (USD) exchange rate.

## Repository Structure
```
├── data/
│   ├── raw/                  # Original BI XLSX + raw GDELT/CNBC attempts
│   ├── interim/              # Validated JISDOR and cleaned candidate metadata
│   └── processed/            # Cleaned, aligned final dataset
├── src/
│   ├── scraper_news.py       # GDELT collection CLI wrapper
│   ├── acquisition/          # GDELT pipeline + bounded CNBC archive collector
│   ├── alignment/            # Strict exact-timestamp/JISDOR alignment
│   ├── pipeline/             # Timestamp-enriched CNBC pilot orchestrator
│   ├── preprocessing/        # JISDOR and news metadata cleaners
│   ├── preprocess_USD-Exchange-Rate.py  # Existing JISDOR CLI, preserved
│   ├── data_alignment.py     # WIB-aware temporal JISDOR alignment
│   └── run_task1.py          # Seven-day CNBC proof-of-concept orchestrator
├── config/                  # API settings, sources and human topic taxonomy
├── tests/                   # Offline pytest suite
├── requirements.txt
└── README.md
```

## Task 1: Data Acquisition & Strategic Preprocessing

### Data Sources
- **News Data**: Existing GDELT DOC 2.0 pipeline plus CNBC historical article site-map discovery
- **Exchange Rate Data**: Bank Indonesia JISDOR (USD/IDR daily rate, Sep 2021 – Sep 2026)

### Pipeline Steps
1. **Acquisition** — Validate the original BI workbook; keep the GDELT pipeline intact; and acquire CNBC title/URL discoveries from a maximum of seven daily archive pages.
2. **Metadata cleaning** — Clean each source independently, deduplicate canonical URLs, preserve raw provenance, and quarantine invalid or out-of-scope records.
3. **Publisher enrichment and filtering** — Cache exact CNBC publisher timestamps/sections and label every article with the unchanged geopolitical taxonomy.
4. **Temporal alignment** — Convert exact publisher timestamps to WIB and map them to actual JISDOR dates using the configured research cutoff. The original date-only v1 output is retained for comparison.
5. **Later research stages** — Human relevance review and model-specific text preparation remain future work. Lowercasing, stopword removal and lemmatization are not applied to source titles.

## Setup
```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
```

## Usage
```bash
# 1. Clean and validate the unchanged BI workbook
python -m src.preprocessing.clean_jisdor

# 2. Collect and clean only the authorized seven-day pilot
python -m src.acquisition.collect_gdelt --start-date 2021-09-01 --end-date 2021-09-07 --pilot

# 3. Run the CNBC acquisition, cleaning, and JISDOR alignment proof-of-concept
python -m src.run_task1 --start-date 2021-09-01 --end-date 2021-09-07

# Reproduce CNBC cleaning/alignment from cached raw evidence with no HTTP
python -m src.run_task1 --cached --skip-jisdor-clean

# Enrich only the existing seven-day CNBC URL set
python -m src.acquisition.collect_cnbc --start-date 2021-09-01 --end-date 2021-09-07 --pilot --enrich-existing

# Run the timestamp-enriched pipeline (reuses discovery and metadata caches)
python -m src.pipeline.run_task1 --source cnbc --start-date 2021-09-01 --end-date 2021-09-07 --pilot --cache-only --skip-jisdor-clean

# Re-run strict alignment or deterministic review sampling independently
python -m src.alignment.align_news_jisdor
python -m src.preprocessing.sample_cnbc_review --n 100 --seed 42

# 4. Run deterministic offline tests
python -m pytest -q
```

## Team Members
- [Member 1]
- [Member 2]
- [Member 3]

## Task 1 data foundation

This implementation extends the existing team repository. The source audit at
`5f812bc` found an implemented `src/preprocess_USD-Exchange-Rate.py`, scraper and
alignment placeholders, the BI workbook, and raw/processed directories. The
README had named a `preprocessing.py` file that was absent. The existing rate
cleaner provided the extraction/parsing/sorting foundation; it now delegates to
the validated module instead of silently dropping invalid or duplicate rows.
No original tracked file was deleted. The alignment placeholder is now the
CNBC/JISDOR temporal-alignment stage, while the working GDELT implementation
remains independent. Existing requirements were retained; only `openpyxl`, `pyyaml`, and
`pytest` were added. Task 1 does not need the NLTK corpus download.

### JISDOR source and validation

Bank Indonesia supplied the original XLSX at
`data/raw/Informasi Kurs Jisdor.xlsx`. It remains at its tracked location rather
than being moved or copied into a new `jisdor/` directory. Its SHA-256 is
`0a8492663f686e4857c33c08923a3c039941c12ae27f03ea31e2227d4ce07c49`.

The inspected workbook has one sheet, `Informasi Kurs Jisdor`, 1,207 sheet rows,
three blank rows, a title row, and `NO`, `Tanggal`, `Kurs` headers on row 5. Its
1,202 actual observations were in descending date order. `Tanggal` contains
month/day/year date strings with time; `Kurs` contains numeric cells. The fourth
column is empty below the title/header area. The output has **exactly**:

```csv
date,jisdor
2021-09-01,14284
2021-09-02,14281
2021-09-03,14261
```

The cleaner discovers a unique XLSX or accepts `--input`, scans sheet contents
for the column names, and extracts by those names rather than fixed positions.
Use `--sheet` for an explicit table if a future workbook has multiple candidate
sheets. It drops `NO`, empty columns, and wholly blank table rows, then parses
dates with pandas and normalizes them to daily resolution. String dates must
include a full ISO or month/day/year calendar date. Numeric Excel serials without
a real date cell and ambiguous/incomplete dates fail instead of being guessed.

Numeric cells are retained; string rates may include `Rp`/`IDR`, whitespace,
thousands separators, or decimal marks according to the policy recorded in QA.
Every observation must have a finite, positive numeric rate. Missing/invalid
values and **all** duplicate dates, including identical duplicates, fail with
source-row diagnostics. No averaging, imputation, scaling, outlier removal,
returns, targets, or synthetic weekend/holiday rows are produced. Missing dates
in QA means null date cells, not gaps in the calendar.

```bash
python -m src.preprocessing.clean_jisdor
# Existing team entry point also works:
python src/preprocess_USD-Exchange-Rate.py
# Explicit alternate source:
python -m src.preprocessing.clean_jisdor --input "data/raw/Informasi Kurs Jisdor.xlsx"
```

Outputs are `data/interim/jisdor/jisdor_clean.csv` and
`data/interim/jisdor/jisdor_cleaning_report.json`. The report calculates row
counts, bounds, nulls, duplicates, numeric dtype, order, parsing policies, and
source/output hashes. Expected values (1,202 rows; 2021-09-01–2026-09-01) are
comparisons only. A failed rerun records `output_valid: false` and removes its
old generated CSV so a stale success cannot be mistaken for current output.
The original XLSX is always read only.

### GDELT collection and reproducibility

`config/gdelt.yaml` controls endpoint, source domains, language, timeouts,
retries, window size, and pacing. `config/geopolitical_topics.yaml` contains the
team's six initial categories unchanged: `armed_conflict`, `sanctions`,
`trade_conflict`, `energy_geopolitics`, `political_instability`, and
`monetary_geoeconomic`. These are transparent **candidate retrieval** choices,
not validated relevance labels. No taxonomy expansion or LLM filtering occurs.

Queries retain configured keyword order, use uppercase `OR`, quote phrases,
and append `domain:reuters.com sourcelang:english` by default. Article List uses
`mode=artlist`, `format=json`, `sort=dateasc`, and `maxrecords=250`. Request
dates use UTC `YYYYMMDDHHMMSS`. Both CLI calendar dates are **inclusive**; the
seven-day pilot means logical `[2021-09-01T00:00:00Z, 2021-09-08T00:00:00Z)`.

Collection starts with daily windows. A response at the configured cap is
potentially truncated, so the window splits into smaller intervals. Midpoints
use minimum-window units, allowing daily trees to reach 15-minute leaves
without zero-duration calls or gaps. Capped terminal intervals retain their
data, log a critical warning, and make QA `collection_complete: false`.
If every record in a capped response has a valid timestamp outside the actual
submitted bounds, the full response is saved as failed, retryable evidence and
splitting stops. Mixed timestamps and ordinary boundary discrepancies continue
through explicit QA; no raw timestamp is adjusted.

The API documentation does not establish a formal endpoint inclusivity
contract. Each logical interval is padded by one second on both sides; actual
`requested_start`/`requested_end` and unpadded `logical_start`/`logical_end` are
retained in raw provenance. Terminal duplicates are combined by normalized URL.
The integrated collector explicitly counts any records outside the overall
requested UTC period as `excluded_boundary_records` and retains them in raw
files. This convention prevents silent gaps or unreported scope expansion; it
does not establish exhaustive archive coverage.

HTTP 429, 5xx, timeouts and connection errors receive bounded exponential
backoff (default 5, 10, 20, 40, 80 seconds). `Retry-After` is respected; a delay
beyond the configured cap leaves a retryable failure. Permanent client errors
and malformed/non-JSON responses are failures, not empty successes. A valid
`{"articles": []}` is the only successful empty response. A live preflight on
2026-09-09 returned HTTP 429 asking for five seconds between calls. Therefore
the configuration keeps a one-second post-success pause and also enforces at
least five idle seconds after **each response or transport failure** before
starting another attempt. Slow responses therefore do not consume the idle gap.

Each window has an atomic checkpoint under
`data/raw/news/gdelt/_checkpoints/`. Raw JSON attempt files are stored under
`data/raw/news/gdelt/YYYY/YYYY-MM-DD/category/`. They preserve GDELT fields and
query/category, request bounds, logical bounds, and retrieval time. Request
identity includes the query and API configuration, so a changed configuration
does not reuse old state. Status is `pending`, `completed`, `failed`,
`saturated`, or internal `split`. Completed leaf requests are skipped on resume;
failed/pending requests retry; split parents traverse their children. Saturated
leaves remain flagged and are retried only with `--force`, which preserves
earlier numbered raw attempt files. Run only one collector per raw directory
at a time; checkpoint writes are atomic, not a multi-process locking system.

After the configured number of consecutive failed daily jobs (default three),
the collector stops making requests and lists every remaining daily job as
pending. Rerun the same command to retry. Exit codes are `0` for completed
requests without saturated intervals, `2` for incomplete collection, and `1`
for configuration/storage errors. Logs go to `logs/gdelt_collection.log`.
`generated_at_utc` and retrieval timestamps vary between runs; identities,
configuration hashes, query strings, category serialization and data-cleaning
rules are deterministic. Reports include the full configuration snapshot.

```bash
# Pilot; rerun unchanged to resume:
python -m src.acquisition.collect_gdelt --start-date 2021-09-01 --end-date 2021-09-07 --pilot
# Compatibility wrapper:
python src/scraper_news.py --pilot
# Rebuild the same scoped CSV and QA offline, without HTTP or checkpoint writes:
python -m src.acquisition.collect_gdelt --pilot --report-only
# Intentionally re-fetch the pilot, preserving earlier raw attempts:
python -m src.acquisition.collect_gdelt --pilot --force
# Narrow diagnostic selection, if the team chooses it:
python -m src.acquisition.collect_gdelt --pilot --topics armed_conflict sanctions --domains reuters.com
```

`--report-only` audits every requested job from retained checkpoints. Missing or
invalid caches become pending in the report; saved failures remain failures.
When checkpoint state is absent (for example in a fresh clone), it reconstructs
the same scoped state in memory from matching numbered raw attempt files. The
latest attempt determines status and records; recorded attempt counters are
summed. Malformed evidence is reported as pending, and raw/checkpoint files are
never written by this mode.
It preserves historical counters, reports zero new HTTP activity, and cannot be
combined with `--force`. This is the reproducible way to inspect a partial pilot
without immediately retrying the service.

The **manual full historical command** below is supplied for the team and was
not executed during implementation. Validate live access and pilot QA first.
No arguments without `--pilot` produce a full-range default run.

```bash
python -m src.acquisition.collect_gdelt --start-date 2021-09-01 --end-date 2026-09-01
```

### Candidate cleaning and QA

The collector automatically writes `data/interim/news/gdelt_news_clean.csv`,
`gdelt_news_cleaning_report.json`, `gdelt_news_quarantine.json`, and
`gdelt_pilot_report.json` (`gdelt_collection_report.json` for non-pilot runs).
CSV avoids adding a Parquet dependency. Raw files are never edited by cleaning.

The clean schema retains `article_id`, `original_url`, `normalized_url`,
`title`, `published_at_utc`, `published_at_wib`, `domain`, `language`,
`sourcecountry`, `categories`, `socialimage`, and `retrieved_at_utc`. Additional
columns preserve mobile URLs, original `seendate`, every original URL, source
record counts, quality flags, and the full JSON retrieval provenance.

**Timestamp caveat:** these requested `published_at_*` column names contain
GDELT `seendate`, its first-seen/index time, **not a verified publisher timestamp**.
`timestamp_semantics=gdelt_first_seen` makes this explicit in every row. Parse
as aware UTC, retain UTC, and convert to `Asia/Jakarta` with its explicit offset.
For a repeated URL use its earliest valid first-seen time; preserve other times
in provenance. Later research must evaluate timestamp suitability before
alignment or claims about prediction.

URL cleaning removes only `utm_source`, `utm_medium`, `utm_campaign`,
`utm_term`, and `utm_content`. It preserves the original URL and does not change
path, fragment, ports, remaining parameter order, encoded keys, or article
identity. This conservative exact deduplication may leave URL aliases as
separate articles. IDs are SHA-256 of the normalized URL. Categories are sorted
JSON arrays; overlapping topics, windows and resumed attempts do not erase
membership. Titles receive NFC Unicode and whitespace normalization only;
case, punctuation, numbers and semantic content are preserved.

Invalid/missing URLs are retained in a separate quarantine JSON because they
cannot be assigned a valid URL-based identity. Missing titles and invalid
timestamps on valid URLs stay in the output with flags. No noisy result is
excluded through an undocumented relevance rule.

The QA report includes source/topic scope, terminal raw records, unique URLs,
duplicate occurrences, category/language/domain counts, earliest/latest valid
times, missing URLs/titles, invalid timestamps, HTTP/retry counters, failed
windows, saturated leaves, pending jobs, and boundary exclusions. Saturated
parent responses are raw evidence but do not count as terminal returned
records. Counts reconcile as terminal raw = boundary exclusions + unique
articles + duplicate occurrences + quarantined occurrences. Historic HTTP
counters include selected checkpoint history; `this_run_http` counts only new
network activity. A forced process termination can leave an interrupted attempt's
counters uncheckpointed; runtime log observations must be reported separately,
without guessing unknown in-flight outcomes. `all_raw_response_records` includes
saved split-parent and failed-response evidence for visited windows as well as
terminal records. An empty CSV after failure is a schema-bearing artifact,
not proof that there was no news.

Standalone recleaning is also available:

```bash
python -m src.preprocessing.clean_news --input data/raw/news/gdelt
```

Standalone cleaning reads active/latest terminal attempts across its input
directory; it is not scoped to one CLI date selection. Its default outputs are
`gdelt_news_all_candidates.csv`, `gdelt_news_all_candidates_report.json`, and
`gdelt_news_all_candidates_quarantine.json`, keeping the collector's scoped
artifacts and pilot report intact. Use the collector command for study-scoped QA.

QA distinguishes `request_windows_completed`, `unsaturated_retrieval`, and
`timestamp_scope_consistent`; `collection_complete` requires all three. Unexpected
`seendate` bounds are preserved and flagged, including near-boundary values.
The [GDELT metadata documentation](https://blog.gdeltproject.org/new-gkg-2-0-article-metadata-fields/)
describes a 15-minute processing cycle, which may help explain boundary effects,
but it does not establish the exact DOC behavior observed here. No timestamps
are adjusted. A normal resume reuses successful responses and retains their
warnings; use `--force` only when intentionally re-querying after investigating
endpoint behavior.

### Tests and data hygiene

```bash
python -m pytest -q
```

Normal tests use temporary synthetic workbook fixtures and mocked HTTP clients;
they never contact GDELT. They cover extraction, invalid/null/duplicate handling,
sorting, query/phrase syntax, UTC request parameters, 429/5xx/transport retries,
retry exhaustion, JSON validation, window termination, resume/force behavior,
URL identity, category aggregation, timestamp conversion and QA accounting.
There is no live integration test in the default suite. The pilot command is
the explicit live integration run.

The implementation was verified with Python 3.13.3, pandas 2.2.3, openpyxl 3.1.5,
requests 2.32.3, PyYAML 6.0.3, and pytest 9.1.1. Python 3.10 or newer is required
by the code's type syntax. These are observed validation versions; the existing
minimum-version dependency policy in `requirements.txt` is preserved.

The original workbook, compact clean JISDOR CSV, QA, and seven-day pilot raw
evidence can be tracked. Python caches, logs, checkpoint state, and other
historical raw date chunks are ignored. `data/` itself is not ignored. Review
the size of generated outputs before committing any later full collection.

### API and research limitations

The [official 2018 historical-search update](https://blog.gdeltproject.org/doc-2-0-updates-1-5-year-searching-and-updated-mobile-interface/)
describes search from 2017-01-01 onward and an Article List restriction to the
last three months **within a requested window**. Daily 2021 requests follow
that published design. The older rolling-cutoff wording in the
[DOC 2.0 launch documentation](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)
must not by itself be used to reject the pilot dates. Nevertheless, current
endpoint behavior, historical archive coverage and Reuters availability must
be verified by the live pilot; published capability does not guarantee access.

The endpoint can rate limit or fail, capped minimum windows may omit articles,
and even uncapped responses do not prove complete historical coverage. Query
matches remain candidates and may be unrelated. Article bodies are not fetched
or provided by this pipeline. No body crawler, paywall/anti-bot bypass, relevance
classifier, sentiment model, FinBERT, embeddings, features, train/test split,
returns, interpolation, or target creation is performed. The CNBC proof-of-concept
does perform a declared date-level JISDOR alignment; GDELT outputs are not
automatically mixed into it. Source selection, taxonomy, relevance validation,
target construction and model design remain human research decisions.

### Recorded implementation results (2026-09-09)

JISDOR cleaning passed: 1,202 source observations, 2021-09-01 through 2026-09-01,
zero duplicate dates or missing date/rate values, and unchanged source SHA-256.
The CSV independently reconciles to every original workbook observation.

The requested seven-day GDELT pilot is **incomplete**. After bounded retries and
a resumed run, the service continued returning HTTP 429; the final live run
stopped at the configured three-consecutive-failed-job threshold.

| Pilot metric | Recorded result |
|---|---:|
| Planned daily topic/source jobs | 42 |
| Completed / failed / pending | 14 / 7 / 21 |
| Raw terminal records / unique articles / duplicates | 418 / 376 / 42 |
| Missing URLs / titles / invalid timestamps | 0 / 0 / 0 |
| Minimum-window saturation | 0 |
| Returned timestamps beyond individual request bounds | 7 |
| Checkpointed HTTP 429 / retries | 67 / 64 |
| Observed HTTP 429 / retry announcements in pipeline logs | 68 / 66 |

The seven timestamp discrepancies are 15-minute boundary spillovers, retained
unchanged. No minimum-window saturation occurred live; cap detection and
adaptive splitting were exercised with mocked responses in the offline tests.
`gdelt_pilot_report.json` lists every failed/pending window and exact scope.
`gdelt_pilot_execution_observations.json` reconciles checkpointed counters with
observed log events; one observed 429 and two retry announcements belong to the
interrupted attempt. A separate initial preflight 429 is outside both pipeline
counter sets. Unknown in-flight outcomes are not inferred.

The final QA was rebuilt with `--report-only`. A copy of the raw files with no
checkpoint directory reproduced the exact scoped CSV and counts, without HTTP
or raw/checkpoint writes. **213 tests passed; 0 failed;
0 skipped.** The five-year command was not executed. Live API
availability and the unfinished pilot must be resolved before a full run.

### CNBC discovery proof-of-concept results (v1, 2026-09-14)

The bounded CNBC orchestrator was run only for **2021-09-01 through
2021-09-07**. It made seven archive requests with zero retries and did not
invoke GDELT or crawl article bodies. CNBC's site-map payload is internally
based on `updatedDate`, so raw evidence can contain canonical URL dates outside
the requested publication interval. Cleaning retains those records in QA but
excludes them from the scoped dataset.

| CNBC validation metric | Result |
|---|---:|
| Archive days completed / failed | 7 / 0 |
| HTTP requests / retries | 7 / 0 |
| Raw archive discoveries | 400 |
| Clean Sep 1–7 canonical-URL articles | 377 |
| Invalid/undated quarantined records | 1 |
| Valid records excluded as outside publication range | 22 |
| Same-day / next-trading-day JISDOR alignments | 350 / 27 |
| Unaligned clean rows | 0 |

The archive payload has no intraday publication time. `publication_date` is
therefore parsed from each validated CNBC canonical URL; `published_at_utc` and
`published_at_wib` remain empty and the alignment basis is recorded as
`publication_date_only`. No time is fabricated. The collector rejects any
request longer than seven inclusive days, so the five-year CNBC scrape was not
run and cannot be launched through this proof-of-concept command. A cached
no-network rerun reproduced all 377 aligned rows. The current offline suite is
**232 passed, 0 failed**.

### CNBC timestamp-enriched pilot results (v2, 2026-09-15)

The v2 enrichment reused exactly the 377 scoped v1 URLs and fetched their CNBC
article pages with a two-second post-response delay, 30-second timeout, three
bounded retries, and one compact checkpoint JSON per article ID. Raw HTML is
not retained. All 377 pages returned publisher timestamps through
`NewsArticle.datePublished`; UTC is retained and WIB is derived with an aware
timezone conversion. Section extraction succeeded for all rows (372 from
JSON-LD and five from the publisher section meta tag). Compared in the original
publisher timestamp offset, 50 sitemap dates and 14 canonical URL dates differ
from the exact published date; these are QA observations, not automatic errors.

The existing `config/geopolitical_topics.yaml` is unchanged. A deterministic
boundary-aware filter selected 28 candidates (7.43%) and retained all 349
non-candidates. Counts by category are armed conflict 4, monetary/geoeconomic
22, political instability 1, trade conflict 1, and zero for sanctions and
energy geopolitics. Section priors add a score signal but cannot create a
candidate without a title or publisher-keyword taxonomy match.

Strict alignment reads only actual dates in `jisdor_clean.csv`. The configured
15:00 WIB cutoff is explicitly a September 2021 research assumption, not a
market-hour fact inferred from the data. Of 377 articles, 151 map to the same
JISDOR date before cutoff, 179 map forward after cutoff, and 47 weekend articles
map to the next actual observation; none are unresolved. The v2 output also
attaches unscaled `previous_jisdor`, `change_idr`, and `return_pct`, with no
interpolation or synthetic dates. The original date-only aligned CSV remains
unchanged.

Because only 28 articles satisfy the deterministic candidate rule, the manual
review CSV includes all 28 rather than inventing non-candidates to reach 100.
Its three human-label fields are blank. Ordering is deterministic with seed 42,
category round-robin balancing, publication-date/section round-robin strata,
and a SHA-256 tie-break. The labeling protocol is in
`docs/cnbc_manual_review_protocol.md`. Ordinary tests remain fully offline.
The completed suite reports **245 passed, 0 failed, 0 skipped**.

Known limitations remain: CNBC can change historical page markup or access
behavior; publisher sections can be broad; the keyword filter can produce both
false positives and false negatives; and manual relevance labels are still
pending. All pilot pages exposed JSON-LD timestamps during this run, but future
reproduction may encounter inaccessible or structurally changed pages. The
15:00 WIB cutoff is a configurable research assumption and needs domain review
before causal claims are made.

To reproduce the complete bounded pilot (reusing caches when present and
fetching only missing metadata for the same 377 URLs):

```bash
python -m src.pipeline.run_task1 --source cnbc --start-date 2021-09-01 --end-date 2021-09-07 --pilot
```
