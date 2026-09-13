import pandas as pd
import pytest

from support_agent.offline.corpus_filter import filter_pairs

MIN_CHARS = 10
MIN_WORDS = 3


def _pairs(*customer_msgs):
    return pd.DataFrame(
        [
            {
                "pair_id": f"p{i}",
                "customer_tweet_id": str(i),
                "customer_msg": msg,
                "customer_msg_clean": msg,
                "agent_tweet_id": f"a{i}",
                "agent_reply": "we can help with that",
            }
            for i, msg in enumerate(customer_msgs)
        ]
    )


def _run(df):
    return filter_pairs(df, min_chars=MIN_CHARS, min_words=MIN_WORDS)


def test_empty_after_cleaning_is_dropped():
    kept, counts = _run(_pairs("", "my order never arrived at all"))

    assert counts["removed_empty"] == 1
    assert list(kept["customer_msg_clean"]) == ["my order never arrived at all"]


def test_emoji_only_message_counts_as_empty():
    """':thumbs_up:' survives cleaning but is not content."""
    kept, counts = _run(_pairs(":thumbs_up:", "where is my package please"))

    assert counts["removed_empty"] == 1
    assert len(kept) == 1


def test_too_short_messages_are_dropped():
    kept, counts = _run(_pairs("thanks", "any update", "my parcel is still missing"))

    assert counts["removed_too_short"] == 2
    assert len(kept) == 1


def test_word_count_gate_catches_long_but_contentless_text():
    kept, _ = _run(_pairs("aaaaaaaaaaaaaaaaaaaa", "order not delivered yet"))

    assert list(kept["customer_msg_clean"]) == ["order not delivered yet"]


def test_exact_duplicates_are_removed_keeping_the_first():
    kept, counts = _run(
        _pairs("where is my order now", "where is my order now", "refund my money please")
    )

    assert counts["removed_duplicate"] == 1
    assert list(kept["pair_id"]) == ["p0", "p2"]


def test_duplicate_matching_ignores_case_and_spacing():
    kept, counts = _run(_pairs("Where Is  My Order", "where is my order"))

    assert counts["removed_duplicate"] == 1
    assert len(kept) == 1


def test_counts_add_up():
    kept, counts = _run(
        _pairs("", "thanks", "where is my order now", "where is my order now", "refund me now")
    )

    assert counts["total"] == 5
    assert counts["kept"] == len(kept)
    assert (
        counts["removed_empty"] + counts["removed_too_short"] + counts["removed_duplicate"]
        + counts["kept"]
        == counts["total"]
    )


def test_missing_clean_column_is_an_error_not_a_silent_pass():
    df = _pairs("where is my order now").drop(columns=["customer_msg_clean"])

    with pytest.raises(RuntimeError, match="customer_msg_clean"):
        _run(df)


@pytest.mark.parametrize(
    "message",
    [
        "注文した商品が届きません",  # Japanese
        "我的包裹还没有送到怎么办",  # Chinese
        "พัสดุของฉันยังไม่มาถึงเลย",  # Thai
    ],
)
def test_unspaced_scripts_survive_the_word_count_gate(message):
    """`\\w+` sees these as one token; a word-count gate would delete the language."""
    kept, counts = _run(_pairs(message))

    assert counts["removed_too_short"] == 0
    assert list(kept["customer_msg_clean"]) == [message]
