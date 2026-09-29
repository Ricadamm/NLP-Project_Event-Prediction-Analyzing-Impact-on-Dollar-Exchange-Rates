# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NLP research project (university course) testing whether geopolitical news can explain/predict USD/IDR moves. Task 1 is a reproducible data pipeline: CNBC news → geopolitical filtering → alignment to Bank Indonesia JISDOR trading days → daily dataset `data/processed/cnbc_jisdor_daily.csv`. Task 2 builds NLP features (category counts, Loughran-McDonald tone, TF-IDF/SVD) from the article-level headlines and tests whether they improve next-day USD/IDR direction prediction over a price-only baseline. Study period: 2021-09-01 to 2026-09-01. Full Task 2 methodology: `docs/task2_plan.md`; results: `reports/task2_results.md`.

## Commands

Python package rooted at the repo root; run modules with `python -m` from the repo root so `src.*` imports resolve.

```bash
python -m pip install -r requirements.txt

# Task 1
python -m src.pipeline.run_task1 --pilot                                    # 2021-09-01..07 pilot
python -m src.pipeline.run_task1 --full-range --start-date 2021-09-01 --end-date 2026-09-01 --enrichment-workers 4
python -m src.pipeline.run_task1 --pilot --cache-only --skip-jisdor-clean   # rerun offline from caches
python -m src.preprocessing.clean_jisdor --help
python -m src.alignment.align_news_jisdor --help
python -m src.acquisition.collect_cnbc --help
python src/scraper_news.py --pilot          # legacy GDELT collector

# Task 2 (needs data/processed/cnbc_headlines_aligned.csv and data/external/Loughran-McDonald_MasterDictionary_*.csv)
python -m src.features.build_daily_features   # NLP features only, no modeling
python -m src.pipeline.run_task2              # full pipeline: features + tuning + test eval + reports

# Tests (Task 1 only; Task 2 has no test suite)
python -m pytest -q
```

Config files: `config/cnbc.yaml`, `config/geopolitical_topics.yaml`, `config/gdelt.yaml` (Task 1) and `config/task2.yaml` (Task 2) are all tracked. `tests/` (Task 1) has 267 tests; 8 in `tests/test_news_cleaning.py` currently fail on a pre-existing fixture issue in `src/preprocessing/clean_news.py` unrelated to any GDELT/JISDOR/CNBC code path Task 2 touches.

## Architecture

`src/pipeline/run_task1.py` orchestrates the stages in order, each a function in its own module that reads a CSV and writes a CSV plus a JSON report:

1. `preprocessing/clean_jisdor.clean_workbook` — parses `data/raw/Informasi Kurs Jisdor.xlsx` → `data/interim/jisdor/jisdor_clean.csv`. Only real observation dates; never interpolate weekends/holidays.
2. `acquisition/collect_cnbc.collect` (uses `cnbc_client`) — per-day CNBC archive discovery, cached under `data/raw/news/cnbc/<year>/<date>/` with `_checkpoints/`. Completed days are immutable cache hits; `--refresh-discovery` resumes missing/failed days, it does not refetch.
3. `preprocessing/clean_cnbc` — dedup/quarantine into `cnbc_article_index.csv`.
4. `preprocessing/prefilter_cnbc` — cheap title/URL keyword prefilter producing the enrichment queue (reuses `filter_cnbc.keyword_pattern`/`validate_taxonomy`).
5. `acquisition/enrich_cnbc` — fetches article pages for exact publisher timestamps; cache in `data/raw/news/cnbc/article_metadata`.
6. `preprocessing/filter_cnbc` — deterministic multi-label geopolitical categories from the taxonomy YAML. These are retrieval labels, not ground truth.
7. `alignment/align_news_jisdor` — timestamps in WIB (`Asia/Jakarta`); before cutoff (default 15:00 WIB, configurable with date-specific `cutoff_regimes` in `cnbc.yaml`) → same JISDOR day, otherwise/non-trading day → next JISDOR day. `write_daily_output` aggregates per trading day with per-category counts.
8. `preprocessing/sample_cnbc_review` — seeded manual-review sample.

The orchestrator writes a run manifest (`data/interim/task1_cnbc_*_run_report.json`) and a QA report whose `counts_reconcile` checks require every stage's in/out counts to balance; a mismatch fails the run. When changing a stage, keep its report keys consistent with what `run_task1.py` reads. JSON outputs are written atomically (temp file + `os.replace`).

Legacy/compat files: `src/run_task1.py` and `src/scraper_news.py` are thin wrappers; `src/data_alignment.py` is the older aligner but still provides `validate_jisdor`/`AlignmentError` used by the strict aligner; `src/preprocess_USDExchangeRate.py` produced `usd_idr_jisdor_cleaned.csv`. GDELT (`collect_gdelt`, `gdelt_client`, `clean_news`) is an independent fallback not invoked by the orchestrator.

### Task 2: NLP features and modeling

`src/pipeline/run_task2.py` orchestrates, same conventions as Task 1 (JSON reports, atomic writes, a manifest, and built-in checks that fail the run rather than silently produce a leaky result):

1. `features/build_daily_features.py` — reads `data/processed/cnbc_headlines_aligned.csv`, reuses Task 1's `matched_categories` (no re-filtering), scores every headline with `features/lm_sentiment.py` (Loughran-McDonald tone; dictionary in `data/external/`, no learned parameters). Writes article-level scores to `data/interim/features/cnbc_headline_features.csv` and a daily aggregate (counts + LM tone, no TF-IDF) to `data/processed/nlp_daily_features.csv`. **Hard-fails if the daily counts don't reconcile exactly with Task 1's `cnbc_jisdor_daily.csv`.**
2. `modeling/dataset.py` — builds price features (lagged returns, momentum, volatility, MA distance) from `cnbc_jisdor_daily.csv`, the next-day direction target (`y_t = 1` if `return_{t+1} > 0` — NOT same-day, to avoid using news published after that day's JISDOR fixing), and the chronological 70/15/15 split with a 1-day purge at each boundary.
3. `features/tfidf_features.py` — TF-IDF + TruncatedSVD, **fit only on the current training window's headlines** (never validation/test). This is refit for every `svd_k` tried during tuning and again on train+val for the final model — it is deliberately not a static file, unlike the counts/LM features.
4. `modeling/train.py` — grid search (logistic regression: `C`, `penalty`, `class_weight`, plus SVD `k` for TF-IDF models) on validation for B2 (price-only) and M1-M4 (price + one or more NLP feature groups); refits the winner on train+val and predicts test once.
5. `modeling/evaluate.py` — accuracy/macro-F1/MCC, McNemar (M4 vs. B2, the primary significance test), block-bootstrap CIs, and a supplementary walk-forward check (expanding window, monthly refits, not used for model selection).

Outputs land in `reports/` (`task2_results.json`/`.md`, `task2_tuning_log.csv`, `svd_top_terms.json`, `figures/`) and `data/processed/task2_test_predictions.csv`. `config/task2.yaml` holds every tunable (split ratios, TF-IDF params, grids, LM negation rules). Current result: no evidence CNBC headline features improve on the price-only baseline at this sample size (~178 test days) — see `reports/task2_results.md` before assuming a code bug if you rerun and see similar near-chance numbers.
