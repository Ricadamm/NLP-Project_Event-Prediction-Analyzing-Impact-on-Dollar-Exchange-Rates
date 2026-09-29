# Task 2 Plan: NLP Features, Modeling, and Evaluation

Status: implemented and run. Results: `reports/task2_results.md` (verdict: no evidence of added predictive value at this sample size). Run it again with `python3 -m src.pipeline.run_task2`.
Research question: do CNBC geopolitical headlines add predictive value for the USD/IDR direction beyond price history alone?

## Pipeline overview

The team's proposed flow, revised so that nothing learned from text can see validation/test data. The key change: **the split dates are fixed first**, from the JISDOR calendar, and TF-IDF/SVD/scaling are fit inside the modeling step on training days only. LM scoring and category counts use no learned parameters, so they are computed once up front.

```text
JISDOR calendar (1,202 trading days)                 cnbc_headlines_aligned.csv (10,997 headlines)
        │                                                        │
        ├─ price features at t (lagged returns,                  ├─ category: reuse matched_categories (Task 1)
        │  momentum, volatility, MA distance)                    ├─ LM tone per headline (dictionary, no fitting)
        ├─ target: y_t = 1 if return_{t+1} > 0                   └─ cleaned text kept for TF-IDF
        ├─ drop warm-up (~20 days) and last day (no t+1)                 │
        └─ FIX SPLIT DATES: chronological 70/15/15,                      │
           1-day purge at each boundary  ───────────────────────────────┤
                                                                         ▼
                                   aggregate by effective_trade_date (+ category):
                                   counts, mean LM (all news + per category)
                                                                         │
                                   LEFT JOIN onto the calendar (~1,180 rows; no-news days → 0)
                                                                         │
                                   modeling loop, per model and grid point:
                                     fit TF-IDF → daily mean → SVD(k) on TRAIN headlines/days only
                                     fit scaler + logistic regression on TRAIN
                                     score on VALIDATION → pick config
                                                                         │
                                   refit chosen config on TRAIN+VAL → predict TEST once
                                                                         │
                                   B0/B1/B2 (price only) vs M1–M4 (price + NLP):
                                   accuracy, macro-F1, MCC, McNemar, bootstrap CIs, walk-forward
```

What changed versus the team's diagram, and why:

| Team diagram | Revised | Reason |
|---|---|---|
| TF-IDF computed at headline level before the split | TF-IDF (and SVD) fit inside the modeling loop on training headlines only | Fitting vocabulary/IDF on all 10,997 headlines lets test-period words and frequencies shape the features. That is leakage and would inflate the NLP model's score |
| Split happens last | Split dates fixed first, from the calendar | Every learned step needs to know where training ends. The split is a property of the calendar, not of the feature table |
| "direction" (unspecified) | Next-day direction, `y_t = 1` if `return_{t+1} > 0` | News up to 15:00 WIB on day `t` is assigned to `t`, after that day's JISDOR fixing. A same-day target would use news published after the rate it predicts (section 2.1) |
| TF-IDF aggregated by category | TF-IDF aggregated over all news per day, then SVD | Per-category TF-IDF gives 6 × k features on ~820 training days and very sparse categories (political instability: 78 headlines in 5 years). Category detail is kept through counts and per-category LM tone, which are only a few columns |
| ~1,202 daily rows | ~1,180 modeling rows | The first ~20 days lack rolling-window history and the last day has no next-day target |
| (not shown) | Statistical checks and walk-forward | About 177 test days gives roughly ±7 pp uncertainty on accuracy, so a single test number cannot establish that news helps |

Efficiency: LM scoring and aggregation run once and are cached in `data/interim/features/`. Only TF-IDF + SVD is refit per grid point, and it runs on ~6.7k training headlines (~9.1k for the final train+val refit), so the full grid takes seconds, not minutes.

---

## 0. Prerequisites (blocking)

### 0.1 Article-level headlines — resolved, full fidelity

A teammate supplied `data/processed/cnbc_headlines_aligned.csv` (10,997 rows: `article_id, title, published_at_wib, effective_trade_date, matched_categories, original_url, previous_jisdor, jisdor, change_idr, return_pct, alignment_reason`). This is the real article-level output of the strict aligner (`src/alignment/align_news_jisdor.py`), not a reconstruction. Verified three ways, not just inspected:
- Row count and `alignment_reason` breakdown match the README exactly: 4,501 `same_day_before_cutoff`, 4,608 `after_cutoff_next_trade_day`, 905 `weekend_next_trade_day`, 983 `non_jisdor_day_next_trade_day`.
- `published_at_wib` carries a real time-of-day, and `effective_trade_date` is already computed with the actual 15:00 WIB cutoff — no date-only approximation needed.
- **Decisive check:** aggregating this file by `effective_trade_date` (news_count + the 6 category counts) and comparing cell-by-cell against the already-committed `data/processed/cnbc_jisdor_daily.csv` gives **zero mismatches across all 1,202 trading days**. This file is the exact source of the committed daily dataset.

(An earlier, incomplete supply attempt — `data/raw/cnbc_article_index.csv`, the pre-enrichment discovery output with day-only dates and no publisher keywords — was superseded by this file and has been deleted from the working tree.)

This is a small file (~4.2MB): commit it to the repo.

Acceptance check (passed): 10,997 rows, `alignment_reason` counts as above, zero nulls in `title`/`published_at_wib`/`effective_trade_date`, and the daily-aggregate reconciliation against `cnbc_jisdor_daily.csv` above.

### 0.2 Restore configs on `main` — done

`config/cnbc.yaml`, `config/geopolitical_topics.yaml` and `config/gdelt.yaml` were recovered from branch `origin/task1-cnbc-pipeline` and copied into this working tree (not yet committed). That branch also had a `tests/` suite, `pytest.ini`, and `docs/cnbc_manual_review_protocol.md`, none of which were on `main`; those were copied in too since they're needed to validate Task 1 before building on top of it.

Note: neither `origin/task1-cnbc-pipeline` nor `origin/task1-data-pipeline` contains the article-level headline files (`cnbc_jisdor_aligned.csv`, `cnbc_news_candidates.csv`, `cnbc_news_enriched.csv`, `cnbc_article_index.csv`) — only aggregated daily counts and JSON reports were ever committed anywhere. Section 0.1 is still blocking.

### 0.3 Dependencies

Add these to `requirements.txt`: `scikit-learn`, `scipy`, `PyYAML`, `openpyxl`, `statsmodels` (for McNemar), and `matplotlib` (for plots). PyYAML and openpyxl are already imported by Task 1 but not listed.

### 0.4 LM dictionary

- Download the Loughran-McDonald Master Dictionary CSV from the Notre Dame Software Repository for Accounting and Finance (SRAF).
- Store it under `data/external/`. Record the version/year and the source URL in the feature report.
- It is free for academic use. Cite Loughran & McDonald (2011).

---

## 1. NLP Feature Extraction

Text source: CNBC headlines only, because article bodies were not collected. Headlines are short (about 8–15 tokens), and this limits both methods. Section 1.6 covers this.

### 1.1 Step 1: category assignment (reuse Task 1, do not rebuild)

Task 1's `src/preprocessing/filter_cnbc.py` already assigns each headline to the 6 categories by keyword matching on `config/geopolitical_topics.yaml`. The result is stored in `matched_categories`.
- Reuse that column so category membership stays identical to the Task 1 counts.
- Labels are **multi-label**: one headline can be in several categories.
- Every article in the aligned file is, by construction, in at least one category.

Planned sanity checks:
- Category distribution per split. Categories whose counts drift strongly over time will behave differently in train and test.
- A precision check using the existing `cnbc_manual_review_sample.csv`. The team fills in `human_relevant`, and the observed precision gets reported as a limitation. Some taxonomy keywords ("conflict", "military") are broad.

### 1.2 Text preprocessing

Use two separate normalizations, because the two methods need different input.

| Step | TF-IDF | LM |
|---|---|---|
| Lowercase | yes | yes (match against the uppercased dictionary) |
| Strip CNBC prefixes (e.g. "CNBC Pro:", "Watch:") | yes | yes |
| Tokenize | sklearn word regex | same regex |
| Stopword removal | yes (English) | no (the dictionary is only content words anyway) |
| Stemming/lemmatization | no. Rely on bigrams and `min_df` | **no**: the LM dictionary already lists inflected forms |
| Near-duplicate headlines on the same trade day | checked, not dropped (see note) | same |

Note on duplicates: checked directly against `cnbc_headlines_aligned.csv` — only 6 rows (3 same-day exact-title pairs) out of 10,997. Dropping them would break the harder invariant in section 4 (daily counts must reconcile exactly with `cnbc_jisdor_daily.csv`), for a change too small to affect any feature. They are counted in the feature report but kept in the data.

### 1.3 Step 2a: TF-IDF (topic/entity signal)

- Use `TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.5, sublinear_tf=True, max_features=5000)`.
- **Fit on training-period headlines only.** The vocabulary and IDF must not see validation or test text.
- **Dimensionality problem:** there are about 5,000 TF-IDF dimensions but only about 820 training days. Raw TF-IDF would overfit badly. Plan:
  1. Headline → TF-IDF vector.
  2. Daily vector = mean of that day's headline vectors.
  3. `TruncatedSVD` (LSA), fit on the training days, keeping `k` components. Tune `k` over {10, 20, 50} on validation.
- Interpretability: for each SVD component, report the top positively and negatively loaded terms, e.g. "component 3 ≈ tariffs / China / trade war".
- Deviation from the original summary: TF-IDF is aggregated over **all news per day**, not per category. Per-category TF-IDF would give 6 × k features on an already small sample. It stays as an optional ablation. Per-category aggregation is kept for the LM tone and the counts, where it adds only a few columns.

### 1.4 Step 2b: Loughran-McDonald tone

For each headline, count dictionary hits in **Negative**, **Positive** and **Uncertainty**. Also count Litigious and Constraining, but only for exploration.

Per-headline scores:
- `lm_neg = neg_hits / n_tokens`, and the same for `lm_pos` and `lm_unc`
- `lm_net = (pos_hits − neg_hits) / (pos_hits + neg_hits + 1)`

Negation: flip a Positive hit to Negative if "no / not / never / none / without" appears in the 3 preceding tokens. This follows the LM convention.

Report **coverage**: the share of headlines with at least one LM hit. Expect many zeros, because headlines are short.

Known overlap: some category keywords (e.g. "crisis", "conflict") are themselves LM Negative words. So `lm_neg` will partly repeat the category counts. Check the correlation and report it.

### 1.5 Step 3: daily aggregation and merge

The key is `effective_trade_date` from `cnbc_headlines_aligned.csv` (section 0.1). It already applies the 15:00 WIB cutoff and pushes weekend/holiday news forward to the next trading day.

Per trading day `t`:

| Feature group | Columns | Count |
|---|---|---|
| Volume | `log1p(news_count)`, `log1p(<category>_count)` × 6 | 7 |
| LM, all news | mean `lm_neg`, `lm_pos`, `lm_unc`, `lm_net` | 4 |
| LM, per category | mean `lm_net` and `lm_neg` for each of the 6 categories | 12 |
| TF-IDF | SVD components 1..k (daily mean) | k |

Rules:
- A day or category with no news gets 0 for its means, and its count column is 0, so the model can tell "no news" apart from "neutral news". There are 17 trading days with no news at all.
- Left-join onto the JISDOR daily table, so every trading day is kept.

Outputs:
- `data/interim/features/cnbc_headline_features.csv`: article-level LM scores
- `data/processed/nlp_daily_features.csv`
- `data/processed/nlp_feature_report.json`: coverage, vocabulary size, SVD explained variance, top terms, LM version

TF-IDF/SVD columns depend on the training window. The final model refits them (section 3.3), so the saved daily file is tied to a specific fit. Record the fit window in the report.

### 1.6 Known limitations

- Headlines only: little context, and the LM dictionary was built for 10-K filings, not news headlines.
- Keyword categories are retrieval labels, not verified annotations.
- CNBC is a US outlet, so coverage of Indonesia-specific drivers (BI policy, domestic politics) is thin.

---

## 2. Baseline Modeling and Task Formulation

### 2.1 Target: next-day direction (important change)

Definition: **`y_t = 1` if `return_pct_{t+1} > 0`** (USD/IDR rises, i.e. the rupiah weakens), else 0. The features use only information available by trading day `t`.

Why next-day rather than same-day `return_pct_t`:
- Task 1 assigns news published up to 15:00 WIB on day `t` to day `t` (`same_day_before_cutoff` in `cnbc_headlines_aligned.csv`); news at/after 15:00 WIB is pushed to day `t+1` (`after_cutoff_next_trade_day`).
- The JISDOR rate for day `t` is fixed earlier in the day. Confirm the exact fixing time with Bank Indonesia's documentation.
- So a same-day model would partly use news published *after* the rate it "predicts". That is look-ahead leakage.
- Next-day prediction removes this problem, and it is what "predict" means in the research question.

Optional secondary analysis: a same-day model, clearly labeled as **contemporaneous/explanatory**. It asks "does news co-move with the rupiah?", not "does news predict it?".

Other target details:
- Ties: 13 days have `return_pct == 0`. Label them 0 ("not UP") and state this.
- Class balance: 55.3% of days are UP over the whole period. Report the balance per split, because the majority-class floor changes by split.

### 2.2 Price-only features (used by the baseline and the combined model)

All are computed from JISDOR up to day `t`:
- lagged returns `r_t, r_{t-1}, …, r_{t-4}`
- 5- and 20-day rolling mean return (momentum)
- 20-day rolling standard deviation of returns (volatility)
- `jisdor_t / MA20_t − 1` (distance from the moving average)
- day-of-week dummies

The first 20 rows are warm-up and get dropped.

### 2.3 Baselines

| ID | Model | Purpose |
|---|---|---|
| B0 | Majority class (from train) | Floor. Any model must beat this |
| B1 | Persistence: predict tomorrow's direction = today's | Classic naive FX baseline |
| **B2** | **Logistic regression on price-only features** | **Main comparison.** The "history only" model |

### 2.4 Combined models (same model type as B2)

Each model uses the same logistic regression, the same hyperparameter grid and the same split. Only the features change.

| ID | Features |
|---|---|
| M1 | price + news counts |
| M2 | price + LM tone (all-news + per-category) |
| M3 | price + TF-IDF SVD |
| **M4** | **price + counts + LM + TF-IDF (full model)** |

The ablation shows *which* NLP representation, if any, adds value.

Model spec:
- sklearn `Pipeline(StandardScaler → LogisticRegression)`, with the scaler fit on train only.
- Grid: `C ∈ {0.01, 0.1, 1, 10}`, `penalty ∈ {l2, l1}` (solver `liblinear`/`saga`), `class_weight ∈ {None, balanced}`, plus the SVD `k` for M3/M4.
- Fixed `random_state = 42`.

Optional robustness family: `HistGradientBoostingClassifier` on the same B2 vs. M4 feature sets, to check that conclusions don't depend on linearity.

---

## 3. Data Splitting and Evaluation

### 3.1 Chronological 70/15/15 split

Usable rows: 1,202 trading days, minus the first day (no previous rate), minus the last day (no next-day target), minus about 20 warm-up days, leaves about 1,180.

| Split | Share | Rows (approx.) | Approx. dates* |
|---|---|---|---|
| Train | 70% | ~826 | 2021-10 → 2025-02 |
| Validation | 15% | ~177 | 2025-02 → 2025-11 |
| Test | 15% | ~177 | 2025-11 → 2026-08 |

\*Exact dates come from the code, not from this table.

Rules:
- Strict time order, no shuffling.
- **One-day purge** at each boundary. The target at `t` uses the price at `t+1`, so the last row of train must not use the first price of validation, and likewise at the validation/test boundary.
- Everything learned is fit on **train only** during tuning: TF-IDF vocabulary/IDF, SVD, scaler, and the majority class for B0.

### 3.2 Tuning protocol

1. For each model ID, fit every grid point on train and score it on validation.
2. Pick the configuration with the best **validation directional accuracy**. Break ties with macro-F1.
3. Log every configuration tried to `reports/task2_tuning_log.csv`, so the reported results are transparent about how many were tried.

### 3.3 Final test (run once)

1. Refit each selected configuration on **train + validation**. This includes refitting TF-IDF and SVD on the train+val headlines.
2. Predict the test set **once**. Do not iterate on test results.
3. Save the per-day predictions and probabilities for every model: `data/processed/task2_test_predictions.csv`.

### 3.4 Metrics

| Metric | Role | Note |
|---|---|---|
| Directional accuracy | Primary | Always shown next to B0, B1 and B2, never alone |
| **Macro-F1** | Secondary | Use macro, not positive-class F1: always predicting UP already gets positive-class F1 ≈ 0.71 at a 55% base rate |
| Balanced accuracy, MCC | Supporting | Harder to game when classes are imbalanced |
| Confusion matrix | Supporting | Shows whether a model collapses to one class |

### 3.5 Is the difference real? (statistical checks)

With about 177 test days, the 95% confidence interval on accuracy is roughly **±7 percentage points**. Small gains will not be distinguishable from noise, so report this plainly.

- **McNemar test**, M4 vs. B2 on the same test days: the main "does news add value?" test.
- **Binomial test** of each model's accuracy against the B0 rate.
- **Block bootstrap** (block of about 5 days, 1,000 resamples) for CIs on accuracy and macro-F1.
- **Walk-forward robustness check** (supplementary): expanding window with monthly refits over the whole period, reporting B2 vs. M4 accuracy per month. It gives far more out-of-sample days than the single test window. It is reported separately and not used for model selection.

Decision rule, stated before running:
- "News adds predictive value" means M4 (or the best of M1–M4) beats B2 on test in **both** accuracy and macro-F1, and McNemar gives p < 0.05.
- A gain that is not significant is reported as "no evidence of added value at this sample size", not as a positive result.

### 3.6 Interpretation outputs

- Logistic regression coefficients (standardized) for M4, with the top features and their sign.
- SVD component → top terms table, so TF-IDF effects can be read in words.
- A plot of predicted probability against the actual direction over the test period.

---

## 4. Implementation Layout (follows Task 1 conventions)

Each stage reads a CSV and writes a CSV plus a JSON report, with atomic writes. A `run_task2` orchestrator records a manifest.

```text
config/task2.yaml                  # split ratios, purge, seeds, TF-IDF params, grids, LM path
src/features/
  text_preprocess.py               # normalization shared by TF-IDF and LM
  lm_sentiment.py                  # dictionary loading, per-headline scores, negation
  tfidf_features.py                # fit-on-window vectorizer + SVD, top-terms export
  build_daily_features.py          # daily aggregation + JISDOR merge
src/modeling/
  dataset.py                       # target shift, price features, warm-up drop, split + purge
  baselines.py                     # B0, B1
  train.py                         # grid search on validation, refit on train+val
  evaluate.py                      # metrics, McNemar, binomial, bootstrap, walk-forward
src/pipeline/run_task2.py          # orchestration + manifest
reports/                           # task2_results.json, tuning log, figures
```

Built-in leakage checks. The run fails if any of these fail, in the same spirit as Task 1's `counts_reconcile`:
- TF-IDF, SVD and scaler were fit only on dates ≤ the fit-window end.
- `y_t` equals `sign(return_pct_{t+1}) > 0` for every row.
- There is no date overlap between splits, and the purge rows are absent.
- The daily feature news counts equal the Task 1 `cnbc_jisdor_daily.csv` counts.

---

## 5. Order of Work

1. Section 0: get the headlines, restore the configs, add the dependencies, download the LM dictionary.
2. `dataset.py` + B0/B1/B2. This gives the price-only reference numbers before any NLP work.
3. LM features, then M2.
4. TF-IDF + SVD, then M3.
5. M1 and M4, then the full tuning run.
6. Final refit and a single test evaluation, then the statistical checks and walk-forward check.
7. Write-up: results table (B0, B1, B2, M1–M4 × accuracy, macro-F1, MCC, CI), the McNemar result, interpretation, limitations.

## 6. Decisions to confirm with the team

1. **Next-day target** instead of same-day `return_pct` (recommended, see 2.1).
2. **TF-IDF over all news + SVD** instead of per-category TF-IDF (recommended, see 1.3).
3. **Macro-F1** as the F1 variant (recommended, see 3.4).
4. Who provides the article-level aligned CSV (section 0.1).
