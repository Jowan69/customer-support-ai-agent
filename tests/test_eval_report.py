"""The report renders, and says what it is computed over."""

from __future__ import annotations

import pandas as pd

from support_agent.eval import classification, decisions, report

META = {
    "n_golden": 200,
    "n_run": 180,
    "n_scored": 180,
    "unscored": ["1", "2"],
    "golden_path": "golden.csv",
    "run_path": "replies.csv",
    "artifact_version": "v2",
    "intents_version": "v2",
    "generated_at": "2026-09-10 00:00 UTC",
}


_BASELINES = {
    "majority": classification.score(
        pd.Series(["delivery_late"] * 20), pd.Series(["delivery_late"] * 18 + ["other"] * 2)
    ),
    "tfidf_logreg": classification.score(
        pd.Series(["delivery_late"] * 19 + ["other"]),
        pd.Series(["delivery_late"] * 18 + ["other"] * 2),
    ),
    "singleton_intents": ["prime_membership"],
}


def _render(baselines=None, decision_baseline=None, retrieval=None):
    actual = pd.Series(["delivery_late"] * 8 + ["fraud_or_unauthorized"] * 2)
    predicted = pd.Series(["delivery_late"] * 10)
    gold_actions = pd.Series(["escalate"] * 6 + ["auto"] * 4)
    pred_actions = pd.Series(["escalate"] * 5 + ["auto"] * 5)

    return report.build_report(
        meta=META,
        classification=classification.score(predicted, actual),
        confusions=classification.confusion_pairs(predicted, actual),
        decisions=decisions.score(pred_actions, gold_actions),
        reasons=pd.DataFrame(columns=["reason", "action", "n", "n_correct", "correct_rate"]),
        drafting={
            "mean_cosine_sim": 0.42,
            "median_cosine_sim": 0.40,
            "judge_means": {"groundedness": 4.1, "safety": 4.4},
            "judge_low_count": {"groundedness": 3, "safety": 11},
            "reply_rate": 0.9,
            "n_scored": 162,
            "n_judged": 162,
            "n_total": 180,
        },
        retrieval=retrieval,
        agreement=[],
        baselines=baselines,
        decision_baseline=decision_baseline,
    )


def test_coverage_is_stated_before_any_metric():
    text = _render()
    assert text.index("hand-labelled rows") < text.index("## Intent classification")
    assert "2 labelled rows are not in the run" in text


def test_unpredicted_intent_is_called_out():
    assert "Never predicted:" in _render()
    assert "`fraud_or_unauthorized`" in _render()


def test_false_auto_is_named_as_the_error_that_matters():
    assert "false auto" in _render()


def test_missing_agreement_says_the_judge_is_ungrounded():
    assert "a model grading a model" in _render()


def test_missing_grounding_does_not_break_the_report():
    assert "_Not scored - no `pair_clusters.parquet`" in _render()


_FROZEN_GROUNDING = {
    "top_k_match_rate": 0.458,
    "top_1_match_rate": 0.411,
    "mean_top1_score": None,
    "mean_top1_score_on_hit": None,
    "mean_top1_score_on_miss": None,
    "n": 107,
    "k": 4,
    "n_clusters_mapped": 18,
    "source": "frozen_run",
    "n_rows": 200,
    "n_no_neighbour": 93,
}


def test_frozen_grounding_says_it_needs_no_artifacts():
    text = _render(retrieval=_FROZEN_GROUNDING)
    assert "regenerates from a clone" in text
    assert "107" in text


def test_frozen_grounding_states_what_was_excluded_and_why():
    text = _render(retrieval=_FROZEN_GROUNDING)
    assert "**93** rows are excluded" in text
    assert "not counted as misses" in text


def test_frozen_grounding_refuses_to_print_a_similarity_it_does_not_have():
    text = _render(retrieval=_FROZEN_GROUNDING)
    assert "Similarity is not reported on this path" in text
    assert "self-match at ~1.0" in text


def test_live_grounding_is_labelled_as_measuring_the_retriever():
    live = dict(_FROZEN_GROUNDING, source="live_index", mean_top1_score=0.71,
                mean_top1_score_on_hit=0.78, mean_top1_score_on_miss=0.66)
    text = _render(retrieval=live)
    assert "measures the retriever" in text
    assert "0.780" in text


def test_baselines_section_is_omitted_when_not_run():
    assert "## Baselines" not in _render()


def test_baselines_section_states_both_ns_honestly():
    text = _render(_BASELINES)
    assert "## Baselines" in text
    # the baselines' own n (20) and the classifier row's n (10) are different
    # populations, and the report must say so rather than implying one number.
    assert "n=20" in text
    assert "n=10" in text


def test_baselines_names_the_singleton_intent_that_cannot_be_predicted():
    text = _render(_BASELINES)
    assert "`prime_membership`" in text
    assert "leave-one-out cross-validation" in text


def test_decision_baseline_section_is_omitted_when_not_run():
    assert "### Compared to the trivial policy" not in _render()


def test_decision_baseline_section_compares_agent_to_always_escalate():
    gold_actions = pd.Series(["escalate"] * 6 + ["auto"] * 4)
    baseline = decisions.always_escalate_baseline(gold_actions)

    text = _render(decision_baseline=baseline)

    assert "### Compared to the trivial policy" in text
    assert "Always escalate (trivial)" in text
    assert "Agent (frozen run)" in text


def test_writes_the_file(tmp_path):
    path = report.write_report(_render(), tmp_path / "results" / "eval_report.md")
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("# Evaluation report")
