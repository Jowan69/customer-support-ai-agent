"""Grounding scoring: the self-match trap, and the artifact-free frozen path."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from support_agent.eval import retrieval


def test_cluster_intent_map_takes_the_modal_human_label():
    golden = pd.DataFrame(
        {
            "cluster_id": [0, 0, 0, 1, 1],
            "intent": ["delivery_late", "delivery_late", "order_status",
                       "refund_status", "refund_status"],
        }
    )
    assert retrieval.cluster_intent_map(golden) == {0: "delivery_late", 1: "refund_status"}


def test_hit_when_any_retrieved_neighbour_shares_the_intent():
    rows = [("delivery_late", ["p1", "p2"], [0.8, 0.7])]
    score = retrieval.score_rows(rows, {"p1": "order_status", "p2": "delivery_late"})
    assert score["top_k_match_rate"] == 1.0
    assert score["top_1_match_rate"] == 0.0  # rank 1 was the wrong kind of case


def test_similarity_is_split_by_hit_and_miss():
    rows = [
        ("delivery_late", ["p1"], [0.9]),
        ("refund_status", ["p1"], [0.3]),
    ]
    score = retrieval.score_rows(rows, {"p1": "delivery_late"})
    assert score["mean_top1_score_on_hit"] == 0.9
    assert score["mean_top1_score_on_miss"] == 0.3


def test_unmapped_pairs_never_count_as_hits():
    rows = [("delivery_late", ["unknown"], [0.9])]
    assert retrieval.score_rows(rows, {})["top_k_match_rate"] == 0.0


def test_absent_similarity_is_none_not_zero():
    """Zero would read as "the neighbours were dissimilar" rather than "unknown"."""
    score = retrieval.score_rows([("delivery_late", ["p1"], [])], {"p1": "delivery_late"})
    assert score["mean_top1_score"] is None
    assert score["mean_top1_score_on_hit"] is None


def test_excluded_rows_are_counted_not_scored():
    score = retrieval.score_rows(
        [("delivery_late", ["p1"], [])], {"p1": "delivery_late"}, n_rows=10
    )
    assert score["n"] == 1
    assert score["n_rows"] == 10
    assert score["n_no_neighbour"] == 9


# --- the self-match trap -----------------------------------------------------------


def test_strip_self_pair_drops_the_querys_own_pair():
    """online/retrieve.py has no self-match filter, so the row's own pair comes back
    at rank 1 with score ~1.0 - in the frozen run, on 181 of 187 rows."""
    assert retrieval.strip_self_pair("123", ["123_456", "789_101"]) == ["789_101"]


def test_strip_self_pair_matches_on_the_customer_id_not_a_substring():
    # "1234_5" belongs to a different customer and must survive a filter for "123".
    assert retrieval.strip_self_pair("123", ["1234_5"]) == ["1234_5"]


def test_parse_pair_ids_handles_json_blank_and_missing():
    assert retrieval.parse_pair_ids(json.dumps(["a", "b"])) == ["a", "b"]
    assert retrieval.parse_pair_ids("") == []
    assert retrieval.parse_pair_ids(float("nan")) == []
    assert retrieval.parse_pair_ids("not json") == []


# --- the frozen path ---------------------------------------------------------------


def _merged() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tweet_id": ["1", "2", "3"],
            "intent_gold": ["delivery_late", "refund_status", "order_status"],
            "cluster_id": [0, 1, 2],
            "grounded_pair_ids": [
                json.dumps(["1_99", "50_51"]),   # own pair first, then a real neighbour
                json.dumps(["2_98"]),            # own pair only -> nothing left to score
                json.dumps([]),                  # cited nothing
            ],
        }
    )


def _assignments() -> pd.DataFrame:
    return pd.DataFrame({"pair_id": ["1_99", "50_51", "2_98"], "cluster_id": [0, 0, 1]})


def test_run_frozen_scores_only_rows_with_a_usable_neighbour():
    golden = pd.DataFrame(
        {"cluster_id": [0, 1, 2], "intent": ["delivery_late", "refund_status", "order_status"]}
    )
    score = retrieval.run_frozen(_merged(), golden, _assignments())

    assert score["source"] == "frozen_run"
    assert score["n"] == 1          # row 1 only
    assert score["n_rows"] == 3
    assert score["n_no_neighbour"] == 2
    assert score["top_1_match_rate"] == 1.0  # 50_51 is in cluster 0 -> delivery_late


def test_run_frozen_reports_no_similarity():
    """`top1_sim` in the frozen run is the self-match, so there is nothing to report."""
    golden = pd.DataFrame({"cluster_id": [0], "intent": ["delivery_late"]})
    score = retrieval.run_frozen(_merged(), golden, _assignments())
    assert score["mean_top1_score"] is None


def test_load_pair_clusters_prefers_the_tracked_copy(tmp_path):
    frozen, artifacts = tmp_path / "frozen", tmp_path / "artifacts"
    frozen.mkdir()
    artifacts.mkdir()
    pd.DataFrame({"pair_id": ["a"], "cluster_id": [0]}).to_parquet(
        frozen / "pair_clusters.parquet", index=False
    )
    pd.DataFrame({"pair_id": ["b"], "cluster_id": [1]}).to_parquet(
        artifacts / "pair_clusters.parquet", index=False
    )
    assert retrieval.load_pair_clusters(frozen, artifacts)["pair_id"].tolist() == ["a"]


def test_load_pair_clusters_falls_back_to_the_artifact_dir(tmp_path):
    frozen, artifacts = tmp_path / "frozen", tmp_path / "artifacts"
    artifacts.mkdir()
    pd.DataFrame({"pair_id": ["b"], "cluster_id": [1]}).to_parquet(
        artifacts / "pair_clusters.parquet", index=False
    )
    assert retrieval.load_pair_clusters(frozen, artifacts)["pair_id"].tolist() == ["b"]


def test_load_pair_clusters_says_where_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError, match="pair_clusters.parquet"):
        retrieval.load_pair_clusters(tmp_path / "frozen", tmp_path / "artifacts")
