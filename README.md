# Global Geopolitical Event Prediction: Analyzing Impact on Dollar Exchange Rates

## Project Overview

This project develops an end-to-end NLP and analytical pipeline to investigate whether global geopolitical events reported in the news influence and help predict fluctuations in the **USD/IDR exchange rate**.

The Task 1 pipeline focuses on:

- Historical geopolitical news acquisition
- News filtering and preprocessing
- Bank Indonesia JISDOR exchange-rate preprocessing
- Timezone-aware temporal alignment between news and JISDOR trading dates
- Generation of article-level and daily datasets for downstream NLP and predictive modeling

The study period covers **1 September 2021 – 1 September 2026**.

---

## Data Sources

### News Data

Historical news articles are collected from **CNBC** using CNBC's historical sitemap/archive infrastructure.

The acquisition pipeline:

1. Discovers historical CNBC article URLs
2. Removes duplicate URLs
3. Applies a broad geopolitical prefilter
4. Retrieves publisher metadata
5. Extracts exact publication timestamps when available
6. Applies deterministic geopolitical-category filtering

The geopolitical categories currently include:

- Armed Conflict
- Sanctions
- Trade Conflict
- Energy Geopolitics
- Political Instability
- Monetary / Geoeconomic Events

These categories are deterministic retrieval and filtering labels, not manually verified ground-truth annotations.

### Exchange Rate Data

USD/IDR exchange-rate data are obtained from **Bank Indonesia JISDOR**.

The dataset contains official JISDOR observations from:

```text
2021-09-01 → 2026-09-01
```

Only actual JISDOR observation dates are retained. Weekends and holidays are not artificially interpolated.

---

## Repository Structure

```text
├── config/
│   ├── cnbc.yaml
│   └── geopolitical_topics.yaml
│
├── data/
│   ├── raw/                         # Original source data
│   ├── interim/                     # Intermediate cleaned/enriched datasets
│   └── processed/                   # Final aligned and aggregated datasets
│
├── src/
│   ├── __init__.py
│   │
│   ├── acquisition/
│   │   ├── __init__.py
│   │   ├── cnbc_client.py
│   │   ├── collect_cnbc.py
│   │   ├── collect_gdelt.py
│   │   ├── enrich_cnbc.py
│   │   └── gdelt_client.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── clean_cnbc.py
│   │   ├── clean_jisdor.py
│   │   ├── clean_news.py
│   │   ├── prefilter_cnbc.py
│   │   ├── filter_cnbc.py
│   │   └── sample_cnbc_review.py
│   │
│   ├── alignment/
│   │   ├── __init__.py
│   │   ├── common.py
│   │   └── align_news_jisdor.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── run_task1.py
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── preprocess_jisdor.py
│   │   ├── run_task1.py
│   │   └── scraper_news.py
│   │
│   └── legacy/
│       ├── __init__.py
│       └── preprocess_USDExchangeRate.py
│
├── tests/
├── requirements.txt
└── README.md
```

---

## Task 1 Pipeline

### 1. Data Acquisition

Historical CNBC articles are discovered across the study period and stored with provenance information such as:

- Article ID
- Article URL
- Headline
- Publication timestamp
- Publisher
- Discovery date
- Retrieval timestamp
- Timestamp quality/status

### 2. News Prefiltering and Enrichment

A broad title and URL prefilter reduces the number of articles requiring additional metadata retrieval.

Selected articles are then enriched with publisher metadata, including exact publication timestamps whenever available.

### 3. Geopolitical Filtering

Articles are filtered using deterministic keyword and topic rules. An article may belong to more than one geopolitical category.

The resulting dataset represents **geopolitical candidate articles** rather than manually labeled ground truth.

### 4. JISDOR Preprocessing

The Bank Indonesia JISDOR dataset is cleaned by:

- Identifying the actual data table in the source file
- Parsing dates
- Converting JISDOR rates to numeric values
- Removing invalid rows
- Checking duplicate dates
- Sorting observations chronologically

No synthetic weekend or holiday observations are created.

### 5. Temporal Alignment

CNBC publication timestamps are converted to **WIB (`Asia/Jakarta`)** before alignment.

The alignment process uses the actual JISDOR trading calendar.

```text
Article published before cutoff
        ↓
Eligible for the same JISDOR trading date

Article published at/after cutoff
        ↓
Next available JISDOR trading date

Article published on weekend/holiday
        ↓
Next available JISDOR trading date
```

The current Task 1 methodology uses a **15:00 WIB cutoff** as the configured research assumption.

No exchange-rate observations are interpolated for non-trading days.

### 6. Exchange-Rate Features

The aligned dataset includes JISDOR movement features such as:

```text
previous_jisdor
jisdor
change_idr
return_pct
```

`return_pct` represents the percentage movement relative to the previous JISDOR observation.

### 7. Daily Aggregation

Article-level observations are aggregated into a daily JISDOR dataset containing features such as:

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

This dataset is intended to support the next stages of NLP feature extraction, hypothesis testing, and predictive modeling.

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Ricadamm/NLP-Project_Event-Prediction-Analyzing-Impact-on-Dollar-Exchange-Rates.git
cd NLP-Project_Event-Prediction-Analyzing-Impact-on-Dollar-Exchange-Rates
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Usage

### Run the Complete Task 1 Pipeline

```bash
python -m src.cli.run_task1 --help
```

The main pipeline implementation is located at:

```text
src/pipeline/run_task1.py
```

### Run CNBC / News Acquisition

```bash
python -m src.acquisition.collect_cnbc --help
```

Legacy GDELT collection remains available for reproducibility:

```bash
python -m src.cli.scraper_news --help
```

### Preprocess JISDOR

```bash
python -m src.cli.preprocess_jisdor --help
```

The validated JISDOR cleaning implementation is located at:

```text
src/preprocessing/clean_jisdor.py
```

The team's original JISDOR preprocessing implementation is retained under:

```text
src/legacy/preprocess_USDExchangeRate.py
```

for project history and reproducibility.

### Align CNBC News with JISDOR

```bash
python -m src.alignment.align_news_jisdor --help
```

This module performs the final timestamp-aware CNBC-to-JISDOR alignment.

---

## Main Processed Output

The primary daily dataset generated by Task 1 is:

```text
data/processed/cnbc_jisdor_daily.csv
```

A pipeline summary and quality-control report is stored in:

```text
data/processed/cnbc_full_collection_report.json
```

---

## Testing

Install the test dependency if necessary:

```bash
python -m pip install pytest
```

Run the test suite with:

```bash
python -m pytest -q
```

---

## Notes

- Exact CNBC publisher timestamps are preferred for temporal alignment.
- All timestamps used for alignment are interpreted in WIB.
- Actual JISDOR observation dates define the trading calendar.
- Weekend and holiday JISDOR values are not interpolated.
- Geopolitical categories are deterministic candidate-retrieval labels.
- The 15:00 WIB cutoff is treated as a configured research assumption for Task 1.

---

## Team Members

- Farhan Adiwidya Pradana
- Adam Rizky
- Muhammad Javier
