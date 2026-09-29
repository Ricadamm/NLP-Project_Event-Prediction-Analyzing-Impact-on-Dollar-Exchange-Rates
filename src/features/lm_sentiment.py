"""Loughran-McDonald finance-tone scoring for CNBC headlines.

Reference: Loughran, T. and McDonald, B. (2011), "When Is a Liability Not a
Liability? Textual Analysis, Dictionaries, and 10-Ks", Journal of Finance.
Dictionary: Loughran-McDonald Master Dictionary, Notre Dame Software
Repository for Accounting and Finance (SRAF), free for academic use.

No parameters are learned from the article corpus here: the dictionary is
fixed, so this stage runs once over every headline regardless of the
train/validation/test split (docs/task2_plan.md, section 1.4).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.text_preprocess import tokenize

LM_CATEGORIES = ("Negative", "Positive", "Uncertainty", "Litigious", "Constraining")


def load_lexicon(path: str | Path) -> dict[str, set[str]]:
    """Load the LM Master Dictionary CSV into {category: {WORD, ...}}.

    A word belongs to a category when that column is non-zero (the LM
    convention stores the dictionary vintage year the word was tagged, not a
    count).
    """
    table = pd.read_csv(path, usecols=["Word", *LM_CATEGORIES])
    lexicon = {}
    for category in LM_CATEGORIES:
        words = table.loc[table[category] > 0, "Word"]
        lexicon[category] = set(words.astype(str).str.upper())
    return lexicon


def score_tokens(
    tokens: list[str],
    lexicon: dict[str, set[str]],
    negation_words: list[str],
    negation_window: int,
) -> dict[str, int]:
    """Count dictionary hits in one tokenized headline.

    A Positive hit is flipped to Negative if a negation word appears in the
    `negation_window` tokens immediately before it (LM convention).
    """
    negation_set = {word.upper() for word in negation_words}
    upper_tokens = [token.upper() for token in tokens]
    neg_hits = pos_hits = unc_hits = lit_hits = constraining_hits = 0
    for index, token in enumerate(upper_tokens):
        window_start = max(0, index - negation_window)
        negated = any(t in negation_set for t in upper_tokens[window_start:index])
        if token in lexicon["Positive"]:
            if negated:
                neg_hits += 1
            else:
                pos_hits += 1
        elif token in lexicon["Negative"]:
            neg_hits += 1
        if token in lexicon["Uncertainty"]:
            unc_hits += 1
        if token in lexicon["Litigious"]:
            lit_hits += 1
        if token in lexicon["Constraining"]:
            constraining_hits += 1
    return {
        "n_tokens": len(tokens),
        "neg_hits": neg_hits,
        "pos_hits": pos_hits,
        "unc_hits": unc_hits,
        "lit_hits": lit_hits,
        "constraining_hits": constraining_hits,
    }


def score_headline(
    title: str,
    lexicon: dict[str, set[str]],
    negation_words: list[str],
    negation_window: int,
) -> dict[str, float]:
    """Return the per-headline LM ratios defined in docs/task2_plan.md, section 1.4."""
    tokens = tokenize(title)
    hits = score_tokens(tokens, lexicon, negation_words, negation_window)
    n_tokens = hits["n_tokens"]
    lm_neg = hits["neg_hits"] / n_tokens if n_tokens else 0.0
    lm_pos = hits["pos_hits"] / n_tokens if n_tokens else 0.0
    lm_unc = hits["unc_hits"] / n_tokens if n_tokens else 0.0
    lm_net = (hits["pos_hits"] - hits["neg_hits"]) / (hits["pos_hits"] + hits["neg_hits"] + 1)
    return {
        "n_tokens": n_tokens,
        "lm_neg": lm_neg,
        "lm_pos": lm_pos,
        "lm_unc": lm_unc,
        "lm_net": lm_net,
        "lm_has_hit": bool(hits["neg_hits"] or hits["pos_hits"] or hits["unc_hits"]),
    }


def score_dataframe(
    titles: pd.Series,
    lexicon: dict[str, set[str]],
    negation_words: list[str],
    negation_window: int,
) -> pd.DataFrame:
    """Vectorized wrapper: one row of LM scores per input title."""
    records = [
        score_headline(title, lexicon, negation_words, negation_window)
        for title in titles
    ]
    return pd.DataFrame.from_records(records, index=titles.index)
