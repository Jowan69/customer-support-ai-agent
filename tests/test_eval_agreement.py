"""Kappa, and the two ways it lies about a small ordinal scale."""

from __future__ import annotations

import json

import pandas as pd

from support_agent.eval import agreement, judge


def _frame(**columns):
    return pd.DataFrame({"tweet_id": [str(i) for i in range(len(next(iter(columns.values()))))],
                         **columns})


def test_perfect_agreement():
    scores = [1, 2, 3, 4, 5, 5, 4, 3]
    rows = agreement.score(_frame(groundedness=scores), _frame(groundedness=scores))
    row = next(r for r in rows if r["dimension"] == "groundedness")
    assert row["kappa"] == 1.0
    assert row["exact_agreement"] == 1.0


def test_off_by_one_is_penalised_less_than_off_by_four():
    human = [1, 2, 3, 4, 5]
    near = agreement.score(_frame(tone=human), _frame(tone=[2, 3, 4, 5, 4]))[0]
    far = agreement.score(_frame(tone=human), _frame(tone=[5, 4, 3, 2, 1]))[0]
    assert near["kappa"] > far["kappa"]
    assert near["within_one"] == 1.0


def test_constant_rater_is_flagged_not_silently_zero():
    """The trap: high agreement, kappa near zero, because there is no variance left."""
    rows = agreement.score(
        _frame(safety=[5, 5, 5, 5, 5, 5, 5, 4]),
        _frame(safety=[5, 5, 5, 5, 5, 5, 5, 5]),
    )
    row = rows[0]
    assert row["degenerate"] is True
    assert row["exact_agreement"] > 0.8
    assert "agreement columns" in row["note"] or row["kappa"] is None


def test_both_raters_constant_gives_none_not_nan():
    rows = agreement.score(_frame(relevance=[4, 4, 4]), _frame(relevance=[4, 4, 4]))
    assert rows[0]["kappa"] is None
    assert "undefined" in rows[0]["note"]


def test_only_shared_rows_are_compared():
    human = pd.DataFrame({"tweet_id": ["1", "2", "3"], "tone": [4, 4, 4]})
    judged = pd.DataFrame({"tweet_id": ["1", "2"], "tone": [4, 5]})
    assert agreement.score(human, judged)[0]["n"] == 2


def test_score_sheet_is_blank_and_only_covers_rows_with_replies():
    merged = pd.DataFrame(
        {
            "tweet_id": ["1", "2", "3"],
            "message": ["a", "b", "c"],
            "reply": ["r1", None, "  "],
            "grounded_pair_ids": ["[]", "[]", "[]"],
        }
    )
    sheet = agreement.make_score_sheet(merged, n=10)
    assert sheet["tweet_id"].tolist() == ["1"]
    for dimension in judge.DIMENSIONS:
        assert (sheet[dimension] == "").all()


def test_score_sheet_carries_the_sources_the_judge_saw():
    """Both raters must see the same sources or groundedness kappa is meaningless."""
    merged = pd.DataFrame(
        {
            "tweet_id": ["1"],
            "message": ["where is my order"],
            "reply": ["a reply"],
            "grounded_pair_ids": [json.dumps(["p1", "p2"])],
        }
    )
    sheet = agreement.make_score_sheet(
        merged, n=1, sources_by_pair={"p1": "past reply one", "p2": "past reply two"}
    )
    assert "past reply one" in sheet.loc[0, "sources"]
    assert "past reply two" in sheet.loc[0, "sources"]
    # rendered by the same function the prompt uses, so the two views are identical
    assert sheet.loc[0, "sources"] == judge.format_sources(["past reply one", "past reply two"])


def test_score_sheet_says_none_when_nothing_was_cited():
    merged = pd.DataFrame(
        {"tweet_id": ["1"], "message": ["m"], "reply": ["r"], "grounded_pair_ids": ["[]"]}
    )
    sheet = agreement.make_score_sheet(merged, n=1, sources_by_pair={"p1": "unused"})
    assert sheet.loc[0, "sources"] == "(none)"
