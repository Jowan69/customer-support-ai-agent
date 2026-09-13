"""Trivial and TF-IDF baselines for intent classification, scored via leave-one-out CV."""

from __future__ import annotations

import pandas as pd

from support_agent.eval import baselines

# Two well-separated classes, three examples each, so every LOOCV fold still has both
# classes with at least one example left to train on.
_MESSAGES = pd.Series(
    [
        "where is my package",
        "package never arrived",
        "still no delivery",
        "you charged my card twice",
        "unauthorized charge on my account",
        "someone used my card",
    ]
)
_INTENTS = pd.Series(
    [
        "delivery_late",
        "delivery_late",
        "delivery_late",
        "fraud_or_unauthorized",
        "fraud_or_unauthorized",
        "fraud_or_unauthorized",
    ]
)


def test_majority_uses_training_fold_only():
    """Each fold's prediction must come from the *other* rows, not a cached global mode.

    Global mode over ["a","a","b","b","b"] is "b" (3 of 5). If a fold's own row leaked
    into its training count, every "b" row would still see two other "b"s plus itself
    and keep predicting "b". Recomputed correctly, removing one "b" leaves a 2-2 tie
    that breaks alphabetically to "a" - so the three "b" rows must predict "a", not "b".
    """
    intents = pd.Series(["a", "a", "b", "b", "b"])
    predictions = baselines.majority_loocv(intents)
    assert predictions == ["b", "b", "a", "a", "a"]


def test_tfidf_returns_one_prediction_per_row():
    predictions = baselines.tfidf_logreg_loocv(_MESSAGES, _INTENTS)
    assert len(predictions) == len(_MESSAGES)
    assert set(predictions) <= set(_INTENTS)


def test_predictions_are_from_training_labels():
    """A classifier can only ever predict a label it was fit on."""
    predictions = baselines.tfidf_logreg_loocv(_MESSAGES, _INTENTS)
    assert set(predictions).issubset(set(_INTENTS.unique()))


def test_singleton_fold_cannot_predict_absent_label():
    """A one-example intent is missing from training in the one fold that matters.

    When `packaging_issue`'s own row is held out, its label is entirely absent from
    that fold's training data - the classifier has no way to predict it there, no
    matter how the text looks.
    """
    messages = pd.concat(
        [_MESSAGES, pd.Series(["this is a one-off complaint about packaging"])],
        ignore_index=True,
    )
    intents = pd.concat([_INTENTS, pd.Series(["packaging_issue"])], ignore_index=True)

    predictions = baselines.tfidf_logreg_loocv(messages, intents)

    singleton_idx = intents[intents == "packaging_issue"].index[0]
    assert predictions[singleton_idx] != "packaging_issue"


def test_run_scores_over_all_golden_rows_via_classification_score():
    golden = pd.DataFrame(
        {
            "message": pd.concat(
                [_MESSAGES, pd.Series(["this is a one-off complaint about packaging"])],
                ignore_index=True,
            ),
            "intent": pd.concat([_INTENTS, pd.Series(["packaging_issue"])], ignore_index=True),
        }
    )

    result = baselines.run(golden)

    assert result["majority"]["n"] == len(golden)
    assert result["tfidf_logreg"]["n"] == len(golden)
    assert result["singleton_intents"] == ["packaging_issue"]
