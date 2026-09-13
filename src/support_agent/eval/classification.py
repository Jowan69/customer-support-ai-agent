"""Per-intent precision / recall / F1 for classify(), scored off a frozen run.

Accuracy alone was the wrong headline. The corpus is dominated by a handful of
delivery intents, so a classifier that never once predicts `fraud_or_unauthorized` -
the intent whose miss costs the most - still scores well. Per-intent numbers make that
failure visible in the row it belongs to.
"""

from __future__ import annotations

from typing import TypedDict

import pandas as pd


class IntentRow(TypedDict):
    intent: str
    precision: float
    recall: float
    f1: float
    support: int
    predicted: int


class ClassificationScore(TypedDict):
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_intent: list[IntentRow]
    n: int
    n_intents_evaluated: int
    unpredicted_intents: list[str]


def score(
    predicted: pd.Series, actual: pd.Series, *, all_intents: list[str] | None = None
) -> ClassificationScore:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    # Macro-F1 is averaged over intents that actually occur in the labelled set, not
    # over all 19 in the taxonomy. Averaging in seven intents with zero support would
    # score the model on rows nobody labelled and drag the headline number toward zero
    # for a reason that has nothing to do with the model.
    present = sorted(set(actual))
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=present, zero_division=0
    )
    predicted_counts = predicted.value_counts()

    per_intent = [
        IntentRow(
            intent=intent,
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
            support=int(support[i]),
            predicted=int(predicted_counts.get(intent, 0)),
        )
        for i, intent in enumerate(present)
    ]
    per_intent.sort(key=lambda r: (r["f1"], r["support"]))

    # Intents the taxonomy defines, that appear in the labelled set, and that the model
    # never once predicted. This is the "it learned to ignore the rare classes" check.
    never_predicted = [i for i in present if predicted_counts.get(i, 0) == 0]

    return ClassificationScore(
        accuracy=float(accuracy_score(actual, predicted)),
        macro_f1=float(f1_score(actual, predicted, labels=present, average="macro",
                                zero_division=0)),
        weighted_f1=float(f1_score(actual, predicted, labels=present, average="weighted",
                                   zero_division=0)),
        per_intent=per_intent,
        n=len(actual),
        n_intents_evaluated=len(present),
        unpredicted_intents=never_predicted,
    )


def confusion_pairs(predicted: pd.Series, actual: pd.Series, top: int = 10) -> list[dict]:
    """The most frequent (gold, predicted) disagreements - the error analysis shortlist."""
    wrong = pd.DataFrame({"actual": actual, "predicted": predicted})
    wrong = wrong[wrong["actual"] != wrong["predicted"]]
    if wrong.empty:
        return []
    counts = wrong.groupby(["actual", "predicted"]).size().reset_index(name="count")
    counts = counts.sort_values("count", ascending=False).head(top)
    return counts.to_dict("records")


def run(merged: pd.DataFrame) -> ClassificationScore:
    """merged: golden joined to the frozen run (intent_gold / intent_pred)."""
    return score(merged["intent_pred"], merged["intent_gold"])
