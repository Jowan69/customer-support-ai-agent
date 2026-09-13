import pandas as pd

from support_agent.offline.chains import build_chains

# Matches settings.min_message_chars / settings.min_message_words, hardcoded here
# (rather than imported) so these tests don't silently change meaning if the
# settings default ever moves.
_MIN_CHARS = 10
_MIN_WORDS = 3


def _msg(tweet_id, author_id, in_response_to, response_to_id, inbound, text="a substantive message right here"):
    return {
        "tweet_id": tweet_id,
        "author_id": author_id,
        "text": text,
        "in_response_to_tweet_id": in_response_to,
        "response_tweet_id": response_to_id,
        "inbound": inbound,
    }


def _build(messages):
    return build_chains(messages, min_chars=_MIN_CHARS, min_words=_MIN_WORDS)


def test_resolved_thread_is_not_repeat_contact():
    messages = pd.DataFrame(
        [
            _msg("1", "cust1", None, "2", True),
            _msg("2", "AmazonHelp", "1", None, False),
        ]
    )

    chains = _build(messages)

    assert len(chains) == 1
    assert bool(chains.iloc[0]["is_repeat_contact"]) is False
    assert chains.iloc[0]["tweet_ids"] == ["1", "2"]


def test_substantive_message_after_agent_reply_is_repeat_contact():
    messages = pd.DataFrame(
        [
            _msg("1", "cust1", None, "2", True),
            _msg("2", "AmazonHelp", "1", "3", False),
            _msg(
                "3", "cust1", "2", None, True,
                text="it still has not arrived and tracking has not updated in four days",
            ),
        ]
    )

    chains = _build(messages)

    assert len(chains) == 1
    assert bool(chains.iloc[0]["is_repeat_contact"]) is True


def test_bare_acknowledgement_after_agent_reply_is_not_repeat_contact():
    """The whole point of this fix: a short thank-you must not count."""
    messages = pd.DataFrame(
        [
            _msg("1", "cust1", None, "2", True),
            _msg("2", "AmazonHelp", "1", "3", False),
            _msg("3", "cust1", "2", None, True, text="Thanks!"),
        ]
    )

    chains = _build(messages)

    assert len(chains) == 1
    assert bool(chains.iloc[0]["is_repeat_contact"]) is False


def test_emoji_only_reply_after_agent_reply_is_not_repeat_contact():
    """An emoji alone demojizes to a name token, which is empty content - the
    same reasoning corpus_filter.py uses for the retrieval corpus."""
    messages = pd.DataFrame(
        [
            _msg("1", "cust1", None, "2", True),
            _msg("2", "AmazonHelp", "1", "3", False),
            _msg("3", "cust1", "2", None, True, text="\U0001F44D"),
        ]
    )

    chains = _build(messages)

    assert len(chains) == 1
    assert bool(chains.iloc[0]["is_repeat_contact"]) is False


def test_long_but_sparse_repeated_word_is_not_repeat_contact():
    """Clears the character gate but not the word gate on its own - both must
    hold, matching corpus_filter.filter_pairs()'s too_short logic exactly."""
    messages = pd.DataFrame(
        [
            _msg("1", "cust1", None, "2", True),
            _msg("2", "AmazonHelp", "1", "3", False),
            _msg("3", "cust1", "2", None, True, text="pleeeeeease"),
        ]
    )

    chains = _build(messages)

    assert len(chains) == 1
    assert bool(chains.iloc[0]["is_repeat_contact"]) is False
