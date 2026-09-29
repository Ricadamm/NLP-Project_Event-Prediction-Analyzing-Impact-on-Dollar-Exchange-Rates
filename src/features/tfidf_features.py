"""TF-IDF + SVD headline features, fit on a training window only.

Unlike LM scoring, TF-IDF vocabulary/IDF and the SVD projection are learned
from data, so they must never see validation or test headlines (docs/task2_plan.md,
sections 1.3 and 3.1). Every public function here takes an explicit boolean
`fit_mask` and fits only on the rows where it is True; all rows are still
transformed, since applying a fixed, already-fitted transform to unseen text
is not leakage.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf_svd(
    texts: pd.Series,
    fit_mask: pd.Series,
    *,
    ngram_range: tuple[int, int],
    min_df: int,
    max_df: float,
    sublinear_tf: bool,
    max_features: int,
    svd_k: int,
    random_state: int,
) -> tuple[TfidfVectorizer, TruncatedSVD, np.ndarray]:
    """Fit TF-IDF + TruncatedSVD on `texts[fit_mask]`, transform every row.

    Returns (vectorizer, svd, svd_matrix) where svd_matrix has one row per
    input text (aligned with `texts.index`) and `svd_k` columns.
    """
    # This machine's BLAS backend (verified: Apple Accelerate under Python
    # 3.9) emits spurious "divide by zero"/"overflow encountered in matmul"
    # RuntimeWarnings from sklearn's internal sparse@dense products -- inside
    # TfidfTransformer's idf scaling as well as TruncatedSVD. Checked
    # directly: the vectorizer, IDF weights and SVD output all come out
    # finite and sane (see docs/task2_plan.md, section 1.3), so this is
    # display noise from this machine's linear algebra library, not a
    # numeric fault. Suppressed narrowly across the whole fit/transform
    # sequence, with a hard finiteness check below as a safety net in case
    # that is ever not true.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)

        vectorizer = TfidfVectorizer(
            ngram_range=tuple(ngram_range),
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            max_features=max_features,
            stop_words="english",
        )
        fit_texts = texts[fit_mask]
        vectorizer.fit(fit_texts)
        tfidf_all = vectorizer.transform(texts)

        n_features = len(vectorizer.vocabulary_)
        effective_k = max(1, min(svd_k, n_features - 1, int(fit_mask.sum()) - 1))
        svd = TruncatedSVD(n_components=effective_k, random_state=random_state, algorithm="arpack")
        svd.fit(vectorizer.transform(fit_texts))
        svd_matrix = svd.transform(tfidf_all)

    if not np.isfinite(svd_matrix).all():
        raise FloatingPointError("Non-finite values in SVD output despite the known-benign warning path")
    return vectorizer, svd, svd_matrix


def daily_mean_svd(svd_matrix: np.ndarray, dates: pd.Series) -> pd.DataFrame:
    """Aggregate per-headline SVD components to a daily mean, indexed by date."""
    columns = [f"svd_{i}" for i in range(svd_matrix.shape[1])]
    frame = pd.DataFrame(svd_matrix, columns=columns, index=dates.index)
    frame["effective_trade_date"] = dates.values
    return frame.groupby("effective_trade_date")[columns].mean()


def top_terms_per_component(
    vectorizer: TfidfVectorizer, svd: TruncatedSVD, n_terms: int = 10
) -> dict[str, dict[str, list[str]]]:
    """For each SVD component, the top positively/negatively loaded terms."""
    terms = np.array(vectorizer.get_feature_names_out())
    report = {}
    for component_index, loadings in enumerate(svd.components_):
        top_positive = terms[np.argsort(loadings)[::-1][:n_terms]].tolist()
        top_negative = terms[np.argsort(loadings)[:n_terms]].tolist()
        report[f"svd_{component_index}"] = {
            "top_positive_terms": top_positive,
            "top_negative_terms": top_negative,
            "explained_variance_ratio": float(svd.explained_variance_ratio_[component_index]),
        }
    return report
