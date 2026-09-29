# Task 2 Results: Do CNBC Geopolitical Headlines Predict USD/IDR Direction?

Pipeline: `python3 -m src.pipeline.run_task2` (config: `config/task2.yaml`). Full methodology and rationale: `docs/task2_plan.md`.

## Verdict

**No evidence that CNBC headline features add predictive value for next-day USD/IDR direction, at this sample size.**

The pre-registered decision rule (docs/task2_plan.md, section 3.5) requires the full model (M4) to beat the price-only baseline (B2) on test in *both* accuracy and macro-F1, with a McNemar test p < 0.05. M4 beat B2 on macro-F1 only (0.488 vs. 0.470), not on accuracy (0.500 vs. 0.545), and McNemar gave p = 0.358 (not significant). None of the four NLP feature variants (M1-M4) beat B2 on accuracy.

This is reported as a negative result, not reframed as a positive one — see Limitations for why this is plausible rather than a pipeline defect.

## Data and splits

| | |
|---|---|
| Headlines | 10,997 (`data/processed/cnbc_headlines_aligned.csv`), full timestamp fidelity, verified against Task 1's committed daily counts (zero mismatches) |
| Modeling rows | 1,181 trading days (1,202 JISDOR days − 20 warm-up − 1 no-target day) |
| Train | 825 days, 2021-09-29 → 2025-02-24 |
| Validation | 176 days, 2025-02-26 → 2025-11-24 |
| Test | 178 days, 2025-11-26 → 2026-08-31 |
| Purge | 1 trading day dropped at each boundary (2025-02-25, 2025-11-25), because its next-day target would use the first price of the following split |
| Class balance (y=1, next day up) | train 55.8%, validation 51.7%, test 57.9% |

## Test-set metrics

| Model | Accuracy | Macro-F1 | MCC | 95% bootstrap CI (accuracy) | Binomial p vs. B0 |
|---|--:|--:|--:|---|--:|
| B0 (majority class) | 0.579 | 0.367 | 0.000 | [0.511, 0.652] | — |
| B1 (persistence) | 0.545 | 0.533 | 0.065 | [0.489, 0.618] | 0.662 |
| **B2 (price-only)** | **0.545** | **0.470** | −0.005 | [0.472, 0.624] | 0.662 |
| M1 (price + counts) | 0.511 | 0.500 | −0.001 | [0.444, 0.584] | 0.906 |
| M2 (price + LM tone) | 0.472 | 0.471 | −0.043 | [0.404, 0.551] | 0.991 |
| M3 (price + TF-IDF/SVD) | 0.522 | 0.476 | −0.027 | [0.444, 0.601] | 0.846 |
| **M4 (price + all NLP)** | **0.500** | **0.488** | −0.024 | [0.433, 0.568] | 0.947 |

Every 95% CI spans roughly 13-16 points and overlaps every other model's — at 178 test days, none of these differences are distinguishable from noise (docs/task2_plan.md, section 3.5 anticipated exactly this).

Note B0 wins on raw accuracy only because the test period happens to be unusually one-sided (57.9% up-days, the highest of the three splits); B0's macro-F1 (0.367) shows it has no real skill, which is why accuracy is never read alone in this plan.

## Primary significance test

McNemar (M4 vs. B2, same 178 test days): **p = 0.358**. B2 was right on 33 days where M4 was wrong; M4 was right on 25 days where B2 was wrong. That 8-day gap is the whole accuracy difference (97 vs. 89 correct), and it is not statistically distinguishable from chance.

## Walk-forward robustness check (supplementary)

Expanding-window, monthly refits, 48 out-of-sample months (far more days than the single 178-day test window), using each model's already-tuned hyperparameters:

| Model | Mean monthly accuracy | Std dev | Months |
|---|--:|--:|--:|
| B2 | 0.558 | 0.115 | 48 |
| M4 | 0.534 | 0.080 | 48 |

B2 outperforms M4 on average across this much larger out-of-sample window too. This was not used for model selection — it exists only to check whether the single-test-window result generalizes, and it points the same direction.

## Interpretation

**M4 standardized logistic coefficients (top by magnitude):**

| Feature | Coefficient | Reading |
|---|--:|---|
| `log_news_count` | −0.378 | More total news that day → lower P(up) next day |
| `political_instability_lm_neg` | −0.255 | Negative-toned political-instability headlines → lower P(up) |
| `log_trade_conflict_count` | +0.229 | More trade-conflict headlines → higher P(up) |
| `log_monetary_geoeconomic_count` | +0.228 | More monetary/geoeconomic headlines → higher P(up) |
| `ma_distance_20` | −0.175 | Price above its 20-day average → lower P(up) (mean-reversion) |

These are directionally plausible individually, but with MCC ≈ 0 on test, the model is not usefully separating up-days from down-days overall — see the M4 probability plot (`reports/figures/m4_test_predictions.png`): predicted probabilities cluster tightly around 0.5 all period, with no visible separation between actual up/down days.

**TF-IDF/SVD components (final train+val fit, top terms):** after removing a "CNBC Daily Open:" newsletter-boilerplate prefix found in 396/10,997 headlines (3.6% — checked directly, not assumed, and now stripped in `config/task2.yaml`), the top components are substantively topical:
- Component 0: inflation, Fed, tariffs, rate, treasury, war
- Component 1: Trump, war, Ukraine, tariffs, Russia, "trump tariffs", "ukraine war", China, trade
- Component 2: tariffs, "trump tariffs", trade, markets, Canada

Full table: `reports/svd_top_terms.json`.

## Limitations (see also docs/task2_plan.md, section 1.6)

- **Headlines only, no article body.** 8-15 tokens per headline gives TF-IDF and LM little to work with; LM dictionary coverage is 52.7% (at least one hit), but most headlines have 0-1 hits total.
- **LM dictionary built for 10-K filings, not news headlines** — tone words like "crisis" or "conflict" are also taxonomy category keywords, so `lm_neg` correlates only weakly (|r| ≤ 0.11) with category membership, not strongly enough to be redundant, but not a clean independent signal either.
- **US-outlet source.** CNBC's coverage of Indonesia-specific drivers (Bank Indonesia policy, domestic politics) is thin; most candidate headlines are about US/global macro (Fed, tariffs) rather than IDR-specific events.
- **Small effective sample for the model complexity.** The final M4 uses 56 features (13 price + 7 count + 16 LM + 20 SVD) against 825 training days; some of the apparent M1-M4 underperformance relative to B2 is consistent with overfitting during validation-based hyperparameter selection, not necessarily "news has zero information content" — a larger news corpus or article bodies could plausibly change this.
- **Categories are retrieval labels, not verified relevance** (Task 1 finding, inherited here).

## Reproducibility

- Every TF-IDF vocabulary/IDF and SVD projection is fit on training-window headlines only (enforced in `src/features/tfidf_features.py`); validation and test headlines are transformed, never fit on.
- Leakage checks (`src/pipeline/run_task2.py::run_leakage_checks`) confirm: target matches `sign(return_{t+1})`, no duplicate rows across splits, ≥1-day purge at each boundary, daily NLP feature counts reconcile exactly with Task 1's committed `cnbc_jisdor_daily.csv`.
- All 144 grid configurations tried (B2/M1/M2: 16 each; M3/M4: 48 each, ×3 SVD-k values) are logged in `reports/task2_tuning_log.csv`.
- `random_state = 42` throughout; three independent runs produced bit-identical results.

## Outputs

| File | Contents |
|---|---|
| `data/interim/features/cnbc_headline_features.csv` | Article-level LM scores + category flags |
| `data/processed/nlp_daily_features.csv` | Daily counts + LM tone (no TF-IDF — that's fit per model, see plan section 1.5) |
| `data/processed/nlp_feature_report.json` | Coverage, category counts, reconciliation check |
| `data/processed/task2_test_predictions.csv` | Per-day predictions/probabilities, every model, test period |
| `reports/task2_tuning_log.csv` | Every grid configuration tried |
| `reports/task2_walk_forward.csv` | Month-by-month B2/M4 accuracy |
| `reports/task2_coefficients.json` | Standardized logistic coefficients, all models |
| `reports/svd_top_terms.json` | Top terms per SVD component, M3 and M4 |
| `reports/task2_results.json` | Full machine-readable results (source for this document) |
| `reports/figures/` | Model comparison bar chart; M4/B2 probability-vs-actual plots |
| `reports/task2_manifest.json` | Run status per stage |
| `data/splits/{train,validation,test}.csv` | The exact modeling rows per split (`python -m src.modeling.export_splits`); excludes TF-IDF/SVD, which is refit per training window |
| `notebook/task2_eda_and_experiments.ipynb` | EDA and the experimental results above, with figures |
