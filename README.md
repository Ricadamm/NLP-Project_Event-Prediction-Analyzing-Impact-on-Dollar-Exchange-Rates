# Global Geopolitical Event Prediction: Analyzing Impact on Dollar Exchange Rates

## Project Overview

This project builds an end-to-end NLP and data-engineering pipeline to study whether global geopolitical news is associated with, and can help predict, movements in the USD/IDR exchange rate.

For Task 1, the pipeline combines:

- **CNBC historical news** from 1 September 2021 to 1 September 2026
- **Bank Indonesia JISDOR** USD/IDR observations over the same period
- Deterministic geopolitical filtering
- Publisher timestamp enrichment
- WIB-aware temporal alignment to actual JISDOR trading days

The current production pipeline has completed successfully for the full five-year period.

---

## Task 1 Production Results

| Stage | Result |
|---|---:|
| CNBC raw discovery records | 124,479 |
| Unique CNBC articles | 120,034 |
| Title-prefilter candidates | 14,763 |
| Exact publisher timestamps | 14,761 |
| Missing/unresolved timestamps | 2 |
| Final geopolitical articles | 11,003 |
| Successfully aligned articles | 10,997 |
| Unaligned articles | 6 |
| JISDOR trading-day rows | 1,202 |
| Trading days with aligned geopolitical news | 1,185 |

The final aligned article dataset and daily aggregate are written to:

```text
data/processed/cnbc_jisdor_aligned.csv
data/processed/cnbc_jisdor_daily.csv
```

---

## Research Pipeline

```text
CNBC historical daily sitemap
        ↓
Discovery: title + URL
        ↓
Cleaning / URL normalization / deduplication
        ↓
Broad geopolitical title prefilter
        ↓
Publisher metadata enrichment
        ↓
Exact datePublished timestamp + section + keywords + authors
        ↓
Deterministic geopolitical filtering
        ↓
UTC → Asia/Jakarta (WIB)
        ↓
Cutoff-aware alignment to actual JISDOR observation dates
        ↓
Article-level aligned dataset
        ↓
Trading-day aggregate dataset
```

### Why two news stages?

**Discovery** identifies which CNBC articles existed during the historical period. It collects article URLs, titles, archive dates, and provenance from CNBC's historical daily article sitemap.

**Enrichment** visits only the prefiltered candidate article pages and extracts publisher metadata required for rigorous temporal alignment, especially CNBC's exact `datePublished` timestamp.

The pipeline does not scrape full article bodies for Task 1.

---

## Data Sources

### CNBC

Historical CNBC article pages are discovered from daily article sitemaps such as:

```text
https://www.cnbc.com/site-map/articles/YYYY/Month/D/
```

The production collector supports:

- resumable daily checkpoints
- retry and exponential backoff
- archive pagination
- completeness validation
- atomic raw evidence
- candidate-only metadata enrichment
- bounded parallel enrichment

### Bank Indonesia JISDOR

The exchange-rate source is the Bank Indonesia JISDOR workbook.

The validated series contains:

- **1,202 observations**
- start date: **2021-09-01**
- end date: **2026-09-01**
- no missing JISDOR values
- no duplicate observation dates

The cleaned output is:

```text
data/interim/jisdor/jisdor_clean.csv
```

with schema:

```text
date,jisdor
```

A higher JISDOR value means more Indonesian rupiah are required per US dollar, i.e. the rupiah is weaker against the USD.

---

## Geopolitical Taxonomy

The deterministic taxonomy contains six categories:

```text
armed_conflict
sanctions
trade_conflict
energy_geopolitics
political_instability
monetary_geoeconomic
```

Articles may receive more than one category.

For example, an article about sanctions affecting Russian oil exports may be counted under both:

```text
sanctions
energy_geopolitics
```

Therefore, the sum of category counts on a trading day may exceed `news_count`.

The taxonomy is defined in:

```text
config/geopolitical_topics.yaml
```

and should be treated as a transparent candidate-labeling rule rather than a fully validated ground-truth annotation scheme.

---

## Temporal Alignment

News is published continuously, while JISDOR is observed only on Bank Indonesia trading days.

All exact CNBC publisher timestamps are converted to:

```text
Asia/Jakarta
```

The pipeline uses a configurable research cutoff of:

```text
15:00 WIB
```

Alignment logic:

| Situation | Effective JISDOR date |
|---|---|
| JISDOR day, before cutoff | same trading day |
| JISDOR day, at/after cutoff | next actual JISDOR date |
| Weekend | next actual JISDOR date |
| Non-JISDOR weekday / holiday | next actual JISDOR date |
| Missing exact publisher timestamp | unaligned |
| News after final available JISDOR observation | unaligned |

Production alignment results:

```text
after_cutoff_next_trade_day          4608
same_day_before_cutoff               4501
non_jisdor_day_next_trade_day         983
weekend_next_trade_day                905
beyond_jisdor_range_unaligned           4
missing_exact_timestamp_unaligned       2
```

The four articles beyond the JISDOR range were published after 15:00 WIB on 2026-09-01, so the next JISDOR date falls outside the available exchange-rate window.

---

## Final Daily Dataset

The main modeling-ready daily dataset is:

```text
data/processed/cnbc_jisdor_daily.csv
```

It contains one row for each actual JISDOR observation date.

### Columns

| Column | Meaning |
|---|---|
| `date` | Actual JISDOR observation date |
| `jisdor` | USD/IDR JISDOR rate |
| `previous_jisdor` | Previous available JISDOR observation |
| `change_idr` | `jisdor - previous_jisdor` |
| `return_pct` | Percentage change from the previous JISDOR observation |
| `news_count` | Number of aligned geopolitical CNBC articles |
| `armed_conflict_count` | Count of armed-conflict articles |
| `sanctions_count` | Count of sanctions articles |
| `trade_conflict_count` | Count of trade-conflict articles |
| `energy_geopolitics_count` | Count of energy-geopolitics articles |
| `political_instability_count` | Count of political-instability articles |
| `monetary_geoeconomic_count` | Count of monetary/geoeconomic articles |

For modeling, `change_idr` or `return_pct` can be used as exchange-rate targets, while the news counts can be used as explanatory or predictive features.

---

## Final Article-Level Dataset

The article-level aligned dataset is:

```text
data/processed/cnbc_jisdor_aligned.csv
```

It preserves article-level metadata, geopolitical labels, publication timestamps, effective trading dates, and alignment reasons.

This dataset is useful for later NLP stages such as:

- sentiment analysis
- event classification
- embedding generation
- topic modeling
- event-intensity scoring

---

## Repository Structure

```text
.
├── config/
│   ├── cnbc.yaml
│   ├── gdelt.yaml
│   └── geopolitical_topics.yaml
├── data/
│   ├── raw/
│   │   └── Informasi Kurs Jisdor.xlsx
│   ├── interim/
│   │   ├── jisdor/
│   │   └── news/
│   └── processed/
├── docs/
├── src/
│   ├── acquisition/
│   ├── alignment/
│   ├── pipeline/
│   └── preprocessing/
├── tests/
├── pytest.ini
├── requirements.txt
└── README.md
```

Large runtime caches and historical raw acquisition evidence should not be committed to GitHub. Keep only representative samples and reproducible processed outputs required by the assignment.

---

## Setup

```bash
python -m venv .venv
```

Activate the virtual environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## Running the Pipeline

### Clean JISDOR

```bash
python -m src.preprocessing.clean_jisdor
```

### Full CNBC Production Pipeline

Once discovery is complete and cached:

```bash
python -m src.pipeline.run_task1 \
    --source cnbc \
    --start-date 2021-09-01 \
    --end-date 2026-09-01 \
    --full-range \
    --skip-jisdor-clean \
    --enrichment-workers 4
```

Do not use `--refresh-discovery` unless an intentional full rediscovery is required.

### Run Tests

```bash
python -m pytest -q
```

On Windows, if the default pytest temporary directory causes permission errors:

```powershell
New-Item -ItemType Directory -Force C:\pytest-farhan | Out-Null
python -m pytest -q --basetemp="C:\pytest-farhan\run1"
```

---

## Quality Assurance

The production pipeline explicitly reports unresolved cases rather than silently imputing them.

Current production exceptions:

- **2 articles** could not obtain an exact publisher timestamp
- **4 articles** were published after the cutoff on the final JISDOR date and therefore had no future JISDOR observation available

These six rows remain visible in the article-level dataset with explicit alignment reasons.

A deterministic manual-review sample is also generated at:

```text
data/interim/news/cnbc_manual_review_sample.csv
```

Human validation of the geopolitical filter should be completed before treating the automated labels as final ground truth.

---


## Detailed Production QA

The final production run includes explicit reconciliation checks across acquisition, deduplication, filtering, metadata enrichment, alignment, and trading-day aggregation.

### Acquisition and filtering reconciliation

| Metric | Value |
|---|---:|
| Requested calendar days | 1,827 |
| CNBC archive days completed | 1,827 |
| CNBC archive days failed | 0 |
| CNBC archive days pending | 0 |
| Raw sitemap records | 124,479 |
| Unique normalized URLs | 120,034 |
| Prefilter candidates | 14,763 |
| Prefilter rejects | 105,271 |
| Prefilter rate | 12.30% |
| Final geopolitical candidates | 11,003 |
| Final candidate rate vs. unique discoveries | 9.17% |

All production reconciliation checks passed:

```text
archive_days       = true
discovery_index    = true
prefilter          = true
metadata_queue     = true
metadata           = true
final_filter       = true
alignment          = true
trading_days       = true
```

### Metadata enrichment QA

The title prefilter reduced the 120,034-article discovery index to 14,763 articles requiring publisher-page metadata enrichment.

| Metric | Value |
|---|---:|
| Exact publisher timestamps | 14,761 |
| Missing publisher timestamps | 2 |
| Metadata cache hits | 7,518 |
| Live metadata fetches in final run | 7,245 |
| HTTP requests in final run | 7,275 |
| Metadata failures | 2 |
| Enrichment workers | 4 |

The cache-hit and live-fetch counts reconcile exactly to the 14,763-row enrichment queue. Failed requests are preserved as unresolved records rather than silently removed.

### Final geopolitical category distribution

The 11,003 final geopolitical articles have the following multi-label category counts:

| Category | Articles |
|---|---:|
| `monetary_geoeconomic` | 5,110 |
| `armed_conflict` | 3,362 |
| `trade_conflict` | 2,428 |
| `energy_geopolitics` | 525 |
| `sanctions` | 298 |
| `political_instability` | 78 |

These counts sum to more than 11,003 because the taxonomy is **multi-label**: one article may match multiple geopolitical categories.

### CNBC section distribution

The most common publisher sections among enriched candidate articles are:

| CNBC section | Articles |
|---|---:|
| Markets | 4,482 |
| Politics | 2,366 |
| PRO Home | 1,176 |
| Business News | 825 |
| Investing | 693 |
| Technology | 388 |
| CNBC Investing Club | 281 |
| CNBC TV | 275 |
| Make It | 167 |
| Asia Economy | 133 |

Only 3 enriched candidate articles have a missing publisher section.

---

## Alignment Diagnostics

The strict alignment stage processes all 11,003 final geopolitical candidates.

| Alignment outcome | Articles |
|---|---:|
| Same JISDOR day, before cutoff | 4,501 |
| After cutoff → next JISDOR day | 4,608 |
| Non-JISDOR weekday/holiday → next JISDOR day | 983 |
| Weekend → next JISDOR day | 905 |
| Beyond available JISDOR range | 4 |
| Missing exact timestamp | 2 |

Summary:

```text
articles with exact publisher timestamps = 11,001
articles aligned                         = 10,997
articles unaligned                       = 6
JISDOR observations                      = 1,202
timezone                                 = Asia/Jakarta
default cutoff                           = 15:00:00 WIB
```

The cutoff is explicitly a **research assumption**, not a value inferred from CNBC or Bank Indonesia data:

> 15:00 WIB research cutoff for the 2021-09-01 through 2026-09-01 study; not inferred from article or JISDOR data.

The alignment policy is:

> No exact publisher timestamp means no strict alignment; no midnight substitution or interpolation.

There is exactly **1 aligned row without a previous JISDOR value**. This is expected for the first JISDOR observation in the study window, because no earlier observation exists inside the dataset.

### Financial feature definitions

The aligned and daily datasets use the following deterministic definitions:

```text
previous_jisdor = previous actual JISDOR observation
change_idr      = jisdor - previous_jisdor
return_pct      = ((jisdor / previous_jisdor) - 1) * 100
```

Positive `change_idr` / `return_pct` means the USD/IDR rate increased, which corresponds to rupiah depreciation against the US dollar. Negative values indicate rupiah appreciation.

---

## Known Limitations

The production QA report records the following limitations:

1. Articles rejected by the broad title prefilter are not publisher-metadata enriched.
2. The complete 120,034-article index is retained so rejected rows remain auditable and can be queued for enrichment later.
3. The deterministic geopolitical filters are candidate-retrieval rules, not human-validated relevance labels.
4. The 15:00 WIB cutoff is a configured research assumption.

For that reason, the generated 200-row manual-review sample should be used to estimate the precision and recall of the deterministic geopolitical filtering before the labels are treated as research ground truth.


## Current Task 1 Status

**Status: Production pipeline passed**

Completed:

- five-year CNBC discovery
- CNBC archive pagination handling
- URL deduplication
- geopolitical title prefilter
- publisher metadata enrichment
- exact timestamp extraction
- deterministic geopolitical filtering
- WIB conversion
- cutoff-aware JISDOR alignment
- trading-day aggregation
- QA reports
- manual-review sample generation

Next research stages include validating the geopolitical labels and building NLP/model features such as sentiment, event representations, and lagged news effects.

---

## Reproducibility Notes

The pipeline is designed to avoid hidden assumptions:

- no interpolation of weekend or holiday JISDOR values
- no synthetic exchange-rate trading days
- no fake midnight timestamps
- no timestamp imputation for unresolved articles
- no silent acceptance of incomplete CNBC archive pages
- no full-body article scraping during Task 1
- no LLM-based geopolitical filtering in the data-acquisition pipeline

The actual JISDOR observation calendar is used as the trading calendar.

---

## Team

Add team-member names and roles here before final submission.
