"""How good the auto/escalate call is, scored off a frozen run.

`escalate` is the positive class throughout, because the two errors are not
symmetrical and the report should not average them into one number:

  false auto      the agent auto-sent something that needed a person. This is the
                  expensive one - a wrong reply went out unread.
  false escalate  the agent asked for a human who was not needed. This costs queue
                  time and nothing else.

Recall on `escalate` is therefore the safety metric, and precision is the efficiency
metric. A system that escalates everything scores 1.0 recall and is useless, which is
why the majority-class baseline sits next to this table in the report.
"""

from __future__ import annotations

from typing import TypedDict

import pandas as pd

ACTIONS = ("auto", "escalate")


class DecisionScore(TypedDict):
    escalate_precision: float
    escalate_recall: float
    escalate_f1: float
    accuracy: float
    confusion_matrix: dict[str, dict[str, int]]
    false_auto: int
    false_escalate: int
    n: int
    gold_escalate_rate: float
    predicted_escalate_rate: float


def score(predicted: pd.Series, actual: pd.Series) -> DecisionScore:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

    cm = confusion_matrix(actual, predicted, labels=list(ACTIONS))
    cm_dict = {
        gold: {pred: int(cm[i][j]) for j, pred in enumerate(ACTIONS)}
        for i, gold in enumerate(ACTIONS)
    }

    precision = float(
        precision_score(actual, predicted, pos_label="escalate", zero_division=0)
    )
    recall = float(recall_score(actual, predicted, pos_label="escalate", zero_division=0))
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return DecisionScore(
        escalate_precision=precision,
        escalate_recall=recall,
        escalate_f1=f1,
        accuracy=float(accuracy_score(actual, predicted)),
        confusion_matrix=cm_dict,
        false_auto=cm_dict["escalate"]["auto"],
        false_escalate=cm_dict["auto"]["escalate"],
        n=len(actual),
        gold_escalate_rate=float((actual == "escalate").mean()) if len(actual) else 0.0,
        predicted_escalate_rate=float((predicted == "escalate").mean()) if len(actual) else 0.0,
    )


def reason_breakdown(merged: pd.DataFrame) -> pd.DataFrame:
    """Which rule fired, and how often it was right.

    `decide()` returns a reason for every call; this is the only place that checks
    whether the reasons it gives are the reasons it should be giving.
    """
    if "reason" not in merged.columns:
        return pd.DataFrame(columns=["reason", "action", "n", "n_correct", "correct_rate"])

    df = merged.copy()
    df["correct"] = df["action"] == df["expected_action"]
    out = (
        df.groupby(["reason", "action"])
        .agg(n=("correct", "size"), n_correct=("correct", "sum"))
        .reset_index()
    )
    out["correct_rate"] = out["n_correct"] / out["n"]
    return out.sort_values("n", ascending=False)


def false_auto_rows(merged: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """The rows that auto-sent when a human was needed. Read these by hand."""
    mask = (merged["expected_action"] == "escalate") & (merged["action"] == "auto")
    columns = [c for c in ("tweet_id", "intent_gold", "intent_pred", "reason", "message", "reply")
               if c in merged.columns]
    return merged.loc[mask, columns].head(limit)


def always_escalate_baseline(actual: pd.Series) -> DecisionScore:
    """The trivial policy: escalate every row, auto-send nothing.

    Scored with the same score() function as the agent, so the two numbers mean
    exactly the same thing. This is the obvious question any reviewer will ask
    - "why not just escalate everything?" - answered with a number instead of
    left for them to work out: this policy gets zero false-autos and accuracy
    equal to the labelled escalate rate, for zero automation. The agent has to
    be read against this, not just admired on its own confusion matrix.
    """
    predicted = pd.Series(["escalate"] * len(actual), index=actual.index)
    return score(predicted, actual)


def run(merged: pd.DataFrame) -> DecisionScore:
    return score(merged["action"], merged["expected_action"])
