# Global Geopolitical Event Prediction: Analyzing Impact on Dollar Exchange Rates

## Project Overview

This project investigates whether global geopolitical news influences and can help predict fluctuations in the **USD/IDR exchange rate**.

Task 1 focuses on building a reproducible data pipeline for:

- historical geopolitical news acquisition,
- CNBC article discovery and metadata enrichment,
- deterministic geopolitical filtering,
- Bank Indonesia JISDOR preprocessing,
- timestamp-aware news-to-trading-day alignment, and
- generation of daily datasets for downstream NLP and predictive modeling.

**Study period:** 1 September 2021 – 1 September 2026.

---

## Data Sources

### News Data

The main historical news source is **CNBC**. The pipeline discovers CNBC articles through historical sitemap/archive records, deduplicates URLs, applies a broad geopolitical prefilter, enriches candidate articles with publisher metadata, and performs deterministic geopolitical filtering.

The current geopolitical categories are:

- Armed Conflict
- Sanctions
- Trade Conflict
- Energy Geopolitics
- Political Instability
- Monetary / Geoeconomic Events

These categories are **candidate-retrieval/filtering labels**, not manually verified ground-truth annotations.

A small GDELT sample and the earlier GDELT acquisition implementation are also retained for reproducibility.

### Exchange Rate Data

USD/IDR exchange-rate data are sourced from **Bank Indonesia JISDOR**.

The repository contains JISDOR observations for the study period and retains only actual JISDOR observation dates. Weekend and holiday rows are not artificially interpolated.

---

## Current Repository Structure

```text
.
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── .gitkeep
│   │   ├── Informasi Kurs Jisdor.xlsx
│   │   └── sample_GDELTS_Raw_Data.tsv
│   │
│   └── processed/
│       ├── .gitkeep
│       ├── cnbc_full_collection_report.json
│       ├── cnbc_jisdor_daily.csv
│       └── usd_idr_jisdor_cleaned.csv
│
└── src/
    ├── __init__.py
    │
    ├── acquisition/
    │   ├── __init__.py
    │   ├── cnbc_client.py
    │   ├── collect_cnbc.py
    │   ├── collect_gdelt.py
    │   ├── enrich_cnbc.py
    │   └── gdelt_client.py
    │
    ├── alignment/
    │   ├── __init__.py
    │   └── align_news_jisdor.py
    │
    ├── pipeline/
    │   ├── __init__.py
    │   └── run_task1.py
    │
    ├── preprocessing/
    │   ├── __init__.py
    │   ├── clean_cnbc.py
    │   ├── clean_jisdor.py
    │   ├── clean_news.py
    │   ├── filter_cnbc.py
    │   ├── prefilter_cnbc.py
    │   └── sample_cnbc_review.py
    │
    ├── data_alignment.py
    ├── preprocess_USDExchangeRate.py
    ├── run_task1.py
    └── scraper_news.py
```

### Source-Code Organization

- `src/acquisition/` — CNBC and GDELT acquisition utilities.
- `src/preprocessing/` — JISDOR cleaning, CNBC cleaning, prefiltering, filtering, and validation sampling.
- `src/alignment/` — strict CNBC timestamp-to-JISDOR temporal alignment and daily aggregation.
- `src/pipeline/` — Task 1 pipeline orchestration.
- `src/data_alignment.py` — shared JISDOR validation utilities plus the earlier alignment implementation currently referenced by the strict aligner.
- `src/preprocess_USDExchangeRate.py` — original JISDOR preprocessing script retained from the initial project implementation.
- `src/run_task1.py` — compatibility entry point for `src.pipeline.run_task1`.
- `src/scraper_news.py` — compatibility entry point for the GDELT collector.

---

## Task 1 Pipeline

### 1. CNBC Historical Discovery

CNBC historical sitemap/archive records are collected across the requested study period.

The full collection report currently records:

- **1,827** requested calendar days
- **124,479** raw sitemap records
- **120,034** unique CNBC URLs

### 2. Prefiltering and Metadata Enrichment

A broad title/URL prefilter is applied before article metadata enrichment.

Current full-run results:

- **14,763** prefilter candidates
- **14,761** exact publisher timestamps
- **2** missing timestamps

### 3. Geopolitical Filtering

The deterministic geopolitical filter produced:

- **11,003** final geopolitical candidate articles

Current multi-label category counts:

| Category | Count |
|---|---:|
| Monetary / Geoeconomic | 5,110 |
| Armed Conflict | 3,362 |
| Trade Conflict | 2,428 |
| Energy Geopolitics | 525 |
| Sanctions | 298 |
| Political Instability | 78 |

Because categories are multi-label, category counts do not sum to the total number of candidate articles.

### 4. JISDOR Preprocessing

The Bank Indonesia workbook is parsed and validated before alignment.

The processing workflow checks:

- date parsing,
- numeric exchange-rate conversion,
- invalid or missing rows,
- duplicate dates,
- chronological ordering, and
- actual JISDOR observation dates.

The final trading calendar contains **1,202 JISDOR observations**.

### 5. Temporal Alignment

CNBC publication timestamps are interpreted in **WIB (`Asia/Jakarta`)** and mapped to actual JISDOR observation dates.

The current research configuration uses a **15:00 WIB cutoff**:

```text
Published before 15:00 WIB on a JISDOR day
        ↓
Same trading day

Published at/after 15:00 WIB
        ↓
Next available JISDOR trading day

Published on weekend / non-JISDOR day
        ↓
Next available JISDOR trading day
```

Current alignment results:

- **10,997** aligned articles
- **6** unaligned articles
- **4,501** same-day-before-cutoff articles
- **4,608** after-cutoff articles mapped to the next trading day
- **905** weekend articles mapped forward
- **983** other non-JISDOR-day articles mapped forward

No weekend or holiday exchange-rate observations are synthesized.

### 6. Daily Aggregation

Aligned article-level records are aggregated to the JISDOR trading-day level.

The daily dataset includes:

```text
date
previous_jisdor
jisdor
change_idr
return_pct
news_count
armed_conflict_count
sanctions_count
trade_conflict_count
energy_geopolitics_count
political_instability_count
monetary_geoeconomic_count
```

The current output contains news on **1,185 of 1,202 JISDOR trading days**.

---

## Main Processed Outputs

### Daily CNBC + JISDOR Dataset

```text
data/processed/cnbc_jisdor_daily.csv
```

This is the main Task 1 daily dataset intended for later NLP feature engineering, statistical analysis, and predictive modeling.

### CNBC Full Collection Report

```text
data/processed/cnbc_full_collection_report.json
```

Contains collection, filtering, enrichment, alignment, category, and reconciliation statistics.

### Original Cleaned JISDOR Dataset

```text
data/processed/usd_idr_jisdor_cleaned.csv
```

Contains the cleaned USD/IDR JISDOR observations from the original preprocessing workflow.

---

## Setup

Clone the repository:

```bash
git clone https://github.com/Ricadamm/NLP-Project_Event-Prediction-Analyzing-Impact-on-Dollar-Exchange-Rates.git
cd NLP-Project_Event-Prediction-Analyzing-Impact-on-Dollar-Exchange-Rates
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## Usage

### Task 1 Pipeline Entry Point

```bash
python src/run_task1.py --help
```

or:

```bash
python -m src.pipeline.run_task1 --help
```

### CNBC Acquisition

```bash
python -m src.acquisition.collect_cnbc --help
```

### GDELT Collector

```bash
python src/scraper_news.py --pilot
```

### Validated JISDOR Preprocessing

```bash
python -m src.preprocessing.clean_jisdor --help
```

The original JISDOR preprocessing implementation is retained as:

```text
src/preprocess_USDExchangeRate.py
```

### CNBC-to-JISDOR Alignment

```bash
python -m src.alignment.align_news_jisdor --help
```

---

## Important Reproducibility Note

The pipeline source currently references configuration files such as:

```text
config/cnbc.yaml
config/geopolitical_topics.yaml
```

These configuration files are **not currently tracked on the `main` branch**. They must be restored or supplied before rerunning the complete historical CNBC pipeline end-to-end from a fresh clone.

The already-generated Task 1 processed outputs are available under `data/processed/`.

---

## Notes

- Exact CNBC publisher timestamps are preferred for strict alignment.
- All alignment is performed relative to WIB.
- Actual JISDOR observation dates define the trading calendar.
- No weekend/holiday JISDOR values are interpolated.
- The **15:00 WIB cutoff** is a configured research assumption.
- Deterministic geopolitical categories are retrieval/filtering labels rather than manual relevance labels.
- GDELT remains in the repository as an earlier/alternative acquisition approach; the final historical pipeline uses CNBC.

---

## Team Members

- Farhan Adiwidya Pradana
- Adam Rizky
- Muhammad Javier
