# Global Geopolitical Event Prediction: Analyzing Impact on Dollar Exchange Rates

## Project Overview

This project investigates whether global geopolitical news can help predict movements in the **USD/IDR exchange rate** (Bank Indonesia JISDOR).

- **Task 1 — Data pipeline.** Collect five years of CNBC articles, filter them into six geopolitical categories, and align every headline to a JISDOR trading day using its exact publication timestamp.
- **Task 2 — NLP features and modeling.** Turn headlines into features (category counts, Loughran-McDonald tone, TF-IDF topics) and test whether they improve **next-day** USD/IDR direction prediction over a price-history-only baseline.

**Study period:** 1 September 2021 – 1 September 2026.

**Current result:** no evidence that CNBC headline features improve next-day direction prediction over the price-only baseline at this sample size (McNemar p = 0.358 on 178 test days; confirmed by a 48-month walk-forward check). Details in [`reports/task2_results.md`](reports/task2_results.md).

---

## Where to Find the Course Deliverables

| Requirement | Location |
|---|---|
| Split dataset (train / validation / test) | [`data/splits/`](data/splits/) |
| Modular code: feature extraction | [`src/features/`](src/features/) |
| Modular code: baseline model training | [`src/modeling/baselines.py`](src/modeling/baselines.py), [`src/modeling/train.py`](src/modeling/train.py) |
| Modular code: evaluation | [`src/modeling/evaluate.py`](src/modeling/evaluate.py) |
| EDA and initial experimental results | [`notebook/task2_eda_and_experiments.ipynb`](notebook/task2_eda_and_experiments.ipynb) |
| Task formulation, feature justification, pipeline design | [`docs/task2_plan.md`](docs/task2_plan.md) (source material for the written report) |
| Results write-up | [`reports/task2_results.md`](reports/task2_results.md) |

---

## Data Sources

### News: CNBC

All news comes from **CNBC**. The Task 1 pipeline discovers articles through CNBC's historical sitemap/archive pages, deduplicates URLs, applies a broad title/URL prefilter, enriches the remaining candidates with publisher metadata (exact publication timestamp, section, keyword tags), and assigns deterministic geopolitical categories:

- Armed Conflict
- Sanctions
- Trade Conflict
- Energy Geopolitics
- Political Instability
- Monetary / Geoeconomic Events

These categories are **keyword-retrieval labels** from [`config/geopolitical_topics.yaml`](config/geopolitical_topics.yaml), not manually verified annotations. Only article **headlines** are used; article bodies were not collected.

### Exchange rate: Bank Indonesia JISDOR

USD/IDR rates come from **Bank Indonesia JISDOR** (`data/raw/Informasi Kurs Jisdor.xlsx`). Only actual JISDOR observation dates are kept; weekend and holiday rates are never interpolated.

### Sentiment dictionary: Loughran-McDonald

Task 2 uses the **Loughran-McDonald Master Dictionary (1993–2025)** from the Notre Dame Software Repository for Accounting and Finance ([SRAF](https://sraf.nd.edu/loughranmcdonald-master-dictionary/)), stored in `data/external/`. It is free for academic use; cite Loughran & McDonald (2011), *Journal of Finance*.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── pytest.ini
│

│
├── data/
│   ├── raw/
│   │   ├── Informasi Kurs Jisdor.xlsx          # Bank Indonesia JISDOR workbook
│   │   └── cnbc_article.gz                     # gzip CSV: all 120,034 discovered CNBC URLs (pre-filter)
│   ├── external/
│   │   └── Loughran-McDonald_MasterDictionary_1993-2025.csv
│   ├── interim/
│   │   └── features/cnbc_headline_features.csv # per-headline category flags + LM scores
│   ├── processed/
│   │   ├── cnbc_jisdor_daily.csv               # Task 1 output: daily JISDOR + news counts
│   │   ├── cnbc_headlines_aligned.csv          # Task 1 output: 10,997 headlines with trade dates
│   │   ├── cnbc_full_collection_report.json    # Task 1 collection/QA statistics
│   │   ├── nlp_daily_features.csv              # Task 2: daily counts + LM tone on the JISDOR calendar
│   │   ├── nlp_feature_report.json             # Task 2: coverage and reconciliation checks
│   │   ├── task2_test_predictions.csv          # Task 2: per-day test predictions, every model
│   │   └── usd_idr_jisdor_cleaned.csv          # original cleaned JISDOR series (legacy script)
│   └── splits/
│       ├── train.csv                           # 825 trading days
│       ├── validation.csv                      # 176 trading days
│       ├── test.csv                            # 178 trading days
│       └── split_summary.json
│
├── notebook/
│   └── task2_eda_and_experiments.ipynb         # EDA + initial experimental results
│
├── src/
│   ├── acquisition/       # CNBC archive discovery (collect_cnbc, cnbc_client) and metadata enrichment
│   ├── preprocessing/     # JISDOR cleaning, CNBC cleaning, prefilter, geopolitical filter, review sampling
│   ├── alignment/         # timestamp → JISDOR trading-day alignment and daily aggregation
│   ├── features/          # Task 2 feature extraction: text cleaning, LM tone, TF-IDF/SVD, daily features
│   ├── modeling/          # Task 2: dataset + splits, baselines, training, evaluation, split export
│   ├── pipeline/          # run_task1.py and run_task2.py orchestrators
│   ├── data_alignment.py  # JISDOR validation helpers used by the strict aligner
│   └── ...                # legacy/compatibility scripts, see "Legacy code"
│
├── reports/
│   ├── task2_results.md / .json                # results write-up and machine-readable results
│   ├── task2_tuning_log.csv                    # all 144 grid configurations tried
│   ├── task2_walk_forward.csv                  # monthly out-of-sample accuracy (B2 vs M4)
│   ├── task2_coefficients.json, svd_top_terms.json
│   └── figures/                                # model comparison and probability plots
│

│
└── tests/                                      # Task 1 test suite (pytest)
```

---

## Task 1: CNBC → JISDOR Data Pipeline

Orchestrated by [`src/pipeline/run_task1.py`](src/pipeline/run_task1.py). Every stage writes a CSV plus a JSON report, and the run fails if the counts between stages don't reconcile.

| Stage | Result (full run) |
|---|---|
| 1. CNBC historical discovery | 1,827 calendar days → 124,479 sitemap records → **120,034 unique URLs** |
| 2. Title/URL prefilter + metadata enrichment | **14,763** candidates; 14,761 with exact publisher timestamps, 2 missing |
| 3. Geopolitical filtering (title + publisher tags) | **11,003** candidate articles |
| 4. JISDOR preprocessing | **1,202** JISDOR trading days |
| 5. Timestamp alignment | **10,997** aligned, 6 unaligned |
| 6. Daily aggregation | news on **1,185 of 1,202** trading days |

Category counts (multi-label, so they sum to more than 11,003):

| Category | Articles |
|---|--:|
| Monetary / Geoeconomic | 5,110 |
| Armed Conflict | 3,362 |
| Trade Conflict | 2,428 |
| Energy Geopolitics | 525 |
| Sanctions | 298 |
| Political Instability | 78 |

### Alignment rule

Publication timestamps are converted to **WIB (`Asia/Jakarta`)** and mapped to actual JISDOR dates with a **15:00 WIB cutoff** (a configured research assumption in `config/cnbc.yaml`):

```text
Published before 15:00 WIB on a JISDOR day   →  same trading day           (4,501 headlines)
Published at/after 15:00 WIB                 →  next JISDOR trading day    (4,608)
Published on a weekend                       →  next JISDOR trading day    (905)
Published on another non-JISDOR day          →  next JISDOR trading day    (983)
```

---

## Task 2: NLP Features and Direction Modeling

Orchestrated by [`src/pipeline/run_task2.py`](src/pipeline/run_task2.py); all parameters live in [`config/task2.yaml`](config/task2.yaml). Full rationale: [`docs/task2_plan.md`](docs/task2_plan.md).

### Task formulation

- **Target:** `y_t = 1` if the **next** trading day's JISDOR return is positive (USD strengthens / rupiah weakens), else 0. Zero-return days (13) count as 0.
- **Why next-day:** news published up to 15:00 WIB on day *t* is assigned to day *t*, which is after that day's JISDOR fixing. Predicting the same day's direction would use news published after the rate being predicted.
- **Metrics:**
  - **Directional accuracy** (primary), always reported next to the baselines.
  - **Macro-F1** (secondary). Positive-class F1 would reward always predicting "up": at a ~55% up-rate that alone scores ≈ 0.71.
  - MCC and balanced accuracy as supporting metrics; McNemar test, block-bootstrap confidence intervals, and a walk-forward check to judge whether differences are real.

### NLP feature extraction

| Feature group | Method | Learned from data? |
|---|---|---|
| Geopolitical categories | Reuses Task 1's `matched_categories` (multi-label) | No |
| News volume | `log1p` of total and per-category daily headline counts | No |
| Finance tone | Loughran-McDonald Negative / Positive / Uncertainty ratios and net tone per headline, with a 3-token negation rule; averaged per day, overall and per category | No |
| Topics | TF-IDF (unigrams + bigrams, English stopwords) → daily mean → TruncatedSVD with k ∈ {10, 20, 50} tuned on validation | **Yes: fit on training-window headlines only** |

**Why these methods:** headlines are short (8–15 tokens), and there are only ~825 training days. The dictionary and TF-IDF approaches are interpretable and cheap, and they need no pretrained language model. Loughran-McDonald is a finance-specific tone lexicon. SVD compresses thousands of TF-IDF terms into a few topics, so the models are not overwhelmed by features on such a small sample.

### Pipeline

```text
JISDOR calendar (1,202 trading days)                   cnbc_headlines_aligned.csv (10,997 headlines)
   │                                                         │
   ├─ price features at t: lagged returns, momentum,         ├─ categories (reused from Task 1)
   │  volatility, distance from 20-day MA, weekday            ├─ LM tone per headline (fixed dictionary)
   ├─ target: y_t = 1 if return_{t+1} > 0                     └─ cleaned text kept for TF-IDF
   ├─ drop 20 warm-up days and the last day (no t+1)              │
   └─ chronological 70/15/15 split, 1-day purge per boundary ─────┤
                                                                   ▼
                            aggregate by trading day: counts + mean LM tone (overall and per category)
                            left-join onto the calendar (1,181 modeling rows; no-news days → 0)
                                                                   │
                            per model and grid point:
                              fit TF-IDF → SVD on TRAIN headlines only
                              fit scaler + logistic regression on TRAIN, score on VALIDATION
                                                                   │
                            refit best config on TRAIN+VALIDATION → predict TEST once
                                                                   │
                            B0/B1/B2 (no news) vs M1–M4 (news): accuracy, macro-F1, MCC,
                            McNemar, bootstrap CIs, walk-forward
```

### Splits

| Split | Trading days | Dates | Next-day "up" share |
|---|--:|---|--:|
| Train | 825 | 2021-09-29 → 2025-02-24 | 55.8% |
| Validation | 176 | 2025-02-26 → 2025-11-24 | 51.7% |
| Test | 178 | 2025-11-26 → 2026-08-31 | 57.9% |

The purged boundary days (2025-02-25 and 2025-11-25) belong to no split. The files in `data/splits/` contain the target, price features, counts and LM tone. They **do not** contain TF-IDF/SVD columns, because those are refit on each training window inside `src/modeling/train.py`.

### Models

| ID | Model | Features |
|---|---|---|
| B0 | Majority class | — |
| B1 | Persistence | tomorrow's direction = today's |
| B2 | Logistic regression | price only |
| M1 | Logistic regression | price + news counts |
| M2 | Logistic regression | price + LM tone |
| M3 | Logistic regression | price + TF-IDF/SVD |
| M4 | Logistic regression | price + counts + LM tone + TF-IDF/SVD |

All logistic models share one grid (C, L1/L2 penalty, class weight, plus SVD k for M3/M4). They are selected on validation accuracy, refit on train + validation, and scored **once** on test.

### Results (test set, 178 days)

| Model | Accuracy | Macro-F1 | MCC | 95% CI (accuracy) |
|---|--:|--:|--:|---|
| B0 majority class | 0.579 | 0.367 | 0.000 | 0.511 – 0.652 |
| B1 persistence | 0.545 | 0.533 | 0.065 | 0.489 – 0.618 |
| **B2 price only** | **0.545** | **0.470** | −0.005 | 0.472 – 0.624 |
| M1 + counts | 0.511 | 0.500 | −0.001 | 0.444 – 0.584 |
| M2 + LM tone | 0.472 | 0.471 | −0.043 | 0.404 – 0.551 |
| M3 + TF-IDF/SVD | 0.522 | 0.476 | −0.027 | 0.444 – 0.601 |
| **M4 all NLP** | **0.500** | **0.488** | −0.024 | 0.433 – 0.568 |

- **M4 vs. B2:** M4 is lower on accuracy and slightly higher on macro-F1. McNemar gives **p = 0.358**, so the difference is not significant.
- **Walk-forward (48 months):** mean monthly accuracy is 0.558 for B2 and 0.534 for M4.
- **B0's accuracy:** the test period happens to be one-sided (57.9% up-days), which inflates majority-class accuracy. Its macro-F1 of 0.367 shows it has no skill.

---

## Setup

Tested with Python 3.9.

```bash
git clone https://github.com/Ricadamm/NLP-Project_Event-Prediction-Analyzing-Impact-on-Dollar-Exchange-Rates.git
cd NLP-Project_Event-Prediction-Analyzing-Impact-on-Dollar-Exchange-Rates

python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .\.venv\Scripts\Activate.ps1     # Windows PowerShell

python -m pip install -r requirements.txt
python -m pip install jupyter      # only needed to open the notebook
```

Run every command from the repository root, so the `src.*` imports and the paths in `config/` resolve.

---

## Usage

### Task 2 (uses the committed Task 1 outputs, no network needed)

```bash
python -m src.pipeline.run_task2           # features → tuning → single test evaluation → reports/ (~30 s)
python -m src.modeling.export_splits       # write data/splits/{train,validation,test}.csv
python -m src.features.build_daily_features  # NLP features only, no modeling
jupyter notebook notebook/task2_eda_and_experiments.ipynb
```

Runs are deterministic (`random_state = 42`). A rerun reproduces the numbers above.

### Task 1 (re-collects CNBC data over the network)

```bash
python -m src.pipeline.run_task1 --pilot                                   # 2021-09-01..07 pilot
python -m src.pipeline.run_task1 --full-range --start-date 2021-09-01 --end-date 2026-09-01 --enrichment-workers 4
python -m src.pipeline.run_task1 --pilot --cache-only --skip-jisdor-clean  # rerun from local caches

python -m src.acquisition.collect_cnbc --help
python -m src.preprocessing.clean_jisdor --help
python -m src.alignment.align_news_jisdor --help
```

The raw CNBC archive cache (`data/raw/news/cnbc/`) is not committed. A full Task 1 rerun re-fetches it. Task 2 does not need it.

### Tests

```bash
python -m pytest -q
```

This runs the Task 1 test suite. 8 tests in `tests/test_news_cleaning.py` currently fail; they cover the legacy GDELT news cleaner (`src/preprocessing/clean_news.py`), which the CNBC pipeline does not use. Task 2 has no unit tests. Instead, each run fails if one of its built-in checks fails: target construction, split ordering and purge, exact reconciliation of daily counts with Task 1, and finite TF-IDF/SVD output. Training-only TF-IDF fitting is enforced in the code (`fit_mask` in `src/features/tfidf_features.py`).

---

## Notes and Limitations

- Only headlines are used (no article bodies), so each document has little context for TF-IDF and the tone dictionary.
- The Loughran-McDonald dictionary was built for 10-K filings, not news headlines. About 53% of headlines contain at least one dictionary word.
- CNBC is a US outlet, so Indonesia-specific drivers (Bank Indonesia policy, domestic politics) are thinly covered.
- The news mix shifts across the splits. Trade conflict is ~6% of training headlines but ~72% of validation headlines (the 2025 tariff period).
- The 15:00 WIB cutoff is a research assumption, and the categories are retrieval labels, not ground truth.
- A recurring newsletter prefix, "CNBC Daily Open:" (396 headlines), is stripped before text features are computed, so it doesn't dominate the TF-IDF topics.

### Legacy code

An earlier version of the project collected news from **GDELT**. The final pipeline uses **CNBC only**, but the GDELT code is still in the repository and is not called by either pipeline:
- `src/acquisition/collect_gdelt.py`, `src/acquisition/gdelt_client.py`
- `src/preprocessing/clean_news.py`
- `src/scraper_news.py` (a wrapper for the GDELT collector)
- `config/gdelt.yaml` and the related tests

Other compatibility files:
- `src/run_task1.py` is a thin wrapper around `src.pipeline.run_task1`.
- `src/preprocess_USDExchangeRate.py` is the original JISDOR cleaning script. It produced `data/processed/usd_idr_jisdor_cleaned.csv` and has been superseded by `src/preprocessing/clean_jisdor.py`.

---

## Team Members

- Farhan Adiwidya Pradana
- Adam Rizky
- Muhammad Javier
