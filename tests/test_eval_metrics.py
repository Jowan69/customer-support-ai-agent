"""Classification and decision scoring."""

from __future__ import annotations

import pandas as pd

from support_agent.eval import classification, decisions


def test_per_intent_exposes_a_class_the_model_never_predicts():
    actual = pd.Series(["delivery_late"] * 8 + ["fraud_or_unauthorized"] * 2)
    predicted = pd.Series(["delivery_late"] * 10)

    score = classification.score(predicted, actual)

    assert score["accuracy"] == 0.8  # looks fine
    assert "fraud_or_unauthorized" in score["unpredicted_intents"]  # is not fine
    fraud = next(r for r in score["per_intent"] if r["intent"] == "fraud_or_unauthorized")
    assert fraud["recall"] == 0.0
    assert fraud["support"] == 2
    # worst F1 first, so the failure is the first row of the table
    assert score["per_intent"][0]["intent"] == "fraud_or_unauthorized"


def test_macro_f1_ignores_intents_absent_from_the_labels():
    actual = pd.Series(["a", "a", "b", "b"])
    predicted = pd.Series(["a", "a", "b", "b"])
    score = classification.score(predicted, actual)
    assert score["macro_f1"] == 1.0
    assert score["n_intents_evaluated"] == 2


def test_confusion_pairs_ranks_the_common_mistakes():
    actual = pd.Series(["delivery_late"] * 3 + ["refund_status"])
    predicted = pd.Series(["delivery_not_received"] * 3 + ["refund_status"])
    pairs = classification.confusion_pairs(predicted, actual)
    assert pairs[0] == {
        "actual": "delivery_late",
        "predicted": "delivery_not_received",
        "count": 3,
    }


def test_escalate_is_the_positive_class():
    actual = pd.Series(["escalate", "escalate", "auto", "auto"])
    predicted = pd.Series(["escalate", "auto", "auto", "escalate"])

    score = decisions.score(predicted, actual)

    assert score["escalate_recall"] == 0.5
    assert score["escalate_precision"] == 0.5
    assert score["false_auto"] == 1
    assert score["false_escalate"] == 1


def test_escalate_everything_scores_perfect_recall():
    """The baseline the report has to be read against."""
    actual = pd.Series(["escalate"] * 7 + ["auto"] * 3)
    predicted = pd.Series(["escalate"] * 10)

    score = decisions.score(predicted, actual)

    assert score["escalate_recall"] == 1.0
    assert score["accuracy"] == 0.7
    assert score["false_auto"] == 0
    assert score["predicted_escalate_rate"] == 1.0


def test_always_escalate_baseline_matches_escalate_everything():
    """The wrapper must be exactly the same computation as calling score()
    directly with an all-escalate series - no separate logic to drift from it."""
    actual = pd.Series(["escalate"] * 7 + ["auto"] * 3)

    baseline = decisions.always_escalate_baseline(actual)
    direct = decisions.score(pd.Series(["escalate"] * 10), actual)

    assert baseline == direct
    assert baseline["escalate_recall"] == 1.0
    assert baseline["false_auto"] == 0
    assert baseline["accuracy"] == 0.7


def test_always_escalate_baseline_preserves_index():
    """predicted must align with actual by position, not by a reset index -
    a mismatched index would silently score every row against the wrong label."""
    actual = pd.Series(["escalate", "auto"], index=[5, 9])
    baseline = decisions.always_escalate_baseline(actual)
    assert baseline["n"] == 2
    assert baseline["false_auto"] == 0


def test_reason_breakdown_scores_each_rule():
    merged = pd.DataFrame(
        {
            "reason": ["unsafe_reply:url", "unsafe_reply:url", "confident_and_grounded"],
            "action": ["escalate", "escalate", "auto"],
            "expected_action": ["escalate", "auto", "auto"],
        }
    )
    out = decisions.reason_breakdown(merged).set_index("reason")
    assert out.loc["unsafe_reply:url", "n"] == 2
    assert out.loc["unsafe_reply:url", "correct_rate"] == 0.5
    assert out.loc["confident_and_grounded", "correct_rate"] == 1.0


def test_false_auto_rows_are_the_dangerous_ones():
    merged = pd.DataFrame(
        {
            "tweet_id": ["1", "2"],
            "expected_action": ["escalate", "auto"],
            "action": ["auto", "auto"],
            "reason": ["confident_and_grounded", "confident_and_grounded"],
            "message": ["m1", "m2"],
        }
    )
    rows = decisions.false_auto_rows(merged)
    assert rows["tweet_id"].tolist() == ["1"]
