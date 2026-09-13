"""Two reference points for intent classification, scored the same way as the LLM.

Neither can train on anything the LLM produced: `pairs.parquet` carries no intent
column, only `data/labels/golden.csv`'s ~200 hand-labelled rows do. So both baselines
are fit and scored entirely within the golden set, via leave-one-out cross-validation
- every row is predicted once, by a model that never saw it during training.

LOOCV rather than k-fold: several intents have only a handful of labelled rows, and
two currently have exactly one. Stratified k-fold cannot split a class with fewer
members than there are folds, and plain k-fold still starves every fold of training
data it doesn't need to give up. LOOCV has no such floor and gives every held-out
prediction the largest training set available (n-1 of n rows). It does not, and
cannot, fix the one-example intents themselves: whichever scheme is used, the one
time a singleton's row is held out, its label is entirely absent from training, so a
classifier fit only on this data has no way to predict it. That is a fact about
having one example, not a shortcoming of any particular cross-validation strategy.

Scored with `classification.score()`, not a second accuracy/macro-F1 implementation
- the point of a baseline is a number that means exactly the same thing as the number
it is compared against.

`message` is used raw, uncleaned - the same input `online/classify.py` hands the LLM
(`pipeline.handle()` never calls `clean_text()` before `classify()`), so the baseline
and the LLM see the same text.
"""

from __future__ import annotations

from typing import TypedDict

import pandas as pd

from support_agent.eval import classification
from support_agent.eval.classification import ClassificationScore


class BaselineResults(TypedDict):
    majority: ClassificationScore
    tfidf_logreg: ClassificationScore
    singleton_intents: list[str]


def predict_majority(train_intents: pd.Series) -> str:
    """The training fold's most common intent.

    Ties break alphabetically (`Series.mode()`'s own tiebreak) - arbitrary, but
    deterministic, which is all a baseline needs.
    """
    return train_intents.mode().iloc[0]


def majority_loocv(intents: pd.Series) -> list[str]:
    """One majority-vote prediction per row, each computed from the *other* rows only.

    Recomputed per fold rather than once globally: the global mode can differ from a
    fold's mode (a near-tied distribution can flip once the held-out row's own vote is
    removed), and using the global mode would leak that row's own label into its own
    "prediction" by way of the count it contributed.
    """
    from sklearn.model_selection import LeaveOneOut

    return [
        predict_majority(intents.iloc[train_idx])
        for train_idx, _ in LeaveOneOut().split(intents)
    ]


def tfidf_logreg_loocv(messages: pd.Series, intents: pd.Series) -> list[str]:
    """One TF-IDF + Logistic Regression prediction per row, via leave-one-out CV.

    The vectorizer is fit inside the pipeline `cross_val_predict` drives, so it never
    sees a held-out row's text before that row is predicted.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            # max_iter raised from sklearn's default (100) only to avoid a spurious
            # non-convergence warning on sparse TF-IDF features - not a performance
            # tune. random_state fixes the one place LogisticRegression can otherwise
            # vary run to run.
            ("logreg", LogisticRegression(max_iter=1000, random_state=0)),
        ]
    )
    predictions = cross_val_predict(pipeline, messages, intents, cv=LeaveOneOut())
    return list(predictions)


def run(golden: pd.DataFrame) -> BaselineResults:
    """golden: `golden.load_golden()`'s output - needs `message` and `intent`."""
    messages = golden["message"].reset_index(drop=True)
    intents = golden["intent"].reset_index(drop=True)

    majority_predictions = majority_loocv(intents)
    tfidf_predictions = tfidf_logreg_loocv(messages, intents)

    counts = intents.value_counts()
    singleton_intents = sorted(counts[counts == 1].index)

    return BaselineResults(
        majority=classification.score(pd.Series(majority_predictions), intents),
        tfidf_logreg=classification.score(pd.Series(tfidf_predictions), intents),
        singleton_intents=singleton_intents,
    )
