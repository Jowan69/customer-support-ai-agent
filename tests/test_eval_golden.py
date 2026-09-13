"""Loading rules: the eval must refuse a half-labelled set rather than score it."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from support_agent.eval import golden

COLUMNS = ["tweet_id", "message", "intent", "expected_action"]


def _write(tmp_path, rows, name="golden.csv"):
    path = tmp_path / name
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


def test_missing_file_names_the_fix(tmp_path):
    with pytest.raises(golden.GoldenError, match="make_golden_sample"):
        golden.load_golden(tmp_path / "nope.csv")


def test_blank_intent_is_an_error_not_a_dropped_row(tmp_path):
    path = _write(tmp_path, [
        ["1", "a", "order_status", "auto"],
        ["2", "b", "", "escalate"],
    ])
    with pytest.raises(golden.GoldenError, match="no `intent`"):
        golden.load_golden(path)


def test_blank_expected_action_is_an_error(tmp_path):
    path = _write(tmp_path, [["1", "a", "order_status", ""]])
    with pytest.raises(golden.GoldenError, match="no `expected_action`"):
        golden.load_golden(path)


def test_unknown_action_rejected(tmp_path):
    path = _write(tmp_path, [["1", "a", "order_status", "maybe"]])
    with pytest.raises(golden.GoldenError, match="unrecognised expected_action"):
        golden.load_golden(path)


def test_unknown_intent_rejected_against_the_taxonomy(tmp_path):
    path = _write(tmp_path, [["1", "a", "invented_intent", "auto"]])
    with pytest.raises(golden.GoldenError, match="not in config/intents.json"):
        golden.load_golden(path, valid_intents=["order_status"])


def test_labels_are_stripped_and_lowercased(tmp_path):
    path = _write(tmp_path, [["1", "a", " order_status ", " Escalate "]])
    loaded = golden.load_golden(path)
    assert loaded.loc[0, "intent"] == "order_status"
    assert loaded.loc[0, "expected_action"] == "escalate"


def test_tweet_id_stays_a_string_so_the_join_works(tmp_path):
    """The join key is numeric in both CSVs; inferred dtypes are how it silently
    produces zero rows."""
    gold_path = _write(tmp_path, [["2293392", "a", "order_status", "auto"]])
    run_path = tmp_path / "replies.csv"
    pd.DataFrame(
        [{"tweet_id": 2293392, "intent": "order_status", "action": "auto"}]
    ).to_csv(run_path, index=False)

    merged, unscored = golden.join(
        golden.load_golden(gold_path), golden.load_run(run_path)
    )
    assert len(merged) == 1
    assert unscored == []


def test_duplicate_run_rows_rejected(tmp_path):
    run_path = tmp_path / "replies.csv"
    pd.DataFrame(
        [
            {"tweet_id": "1", "intent": "order_status", "action": "auto"},
            {"tweet_id": "1", "intent": "order_status", "action": "auto"},
        ]
    ).to_csv(run_path, index=False)
    with pytest.raises(golden.GoldenError, match="duplicate tweet_ids"):
        golden.load_run(run_path)


def test_join_reports_labelled_rows_the_run_missed(tmp_path):
    gold_path = _write(tmp_path, [
        ["1", "a", "order_status", "auto"],
        ["2", "b", "refund_status", "escalate"],
    ])
    run_path = tmp_path / "replies.csv"
    pd.DataFrame(
        [{"tweet_id": "1", "intent": "order_status", "action": "auto"}]
    ).to_csv(run_path, index=False)

    merged, unscored = golden.join(
        golden.load_golden(gold_path), golden.load_run(run_path)
    )
    assert len(merged) == 1
    assert unscored == ["2"]


def test_load_intents_reads_the_taxonomy(tmp_path):
    path = tmp_path / "intents.json"
    path.write_text(json.dumps({"intents": [{"name": "a"}, {"name": "b"}]}))
    assert golden.load_intents(path) == ["a", "b"]
