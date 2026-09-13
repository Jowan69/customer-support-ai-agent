import pandas as pd

from support_agent.offline.pairs import build_pairs


def test_single_hop_join_on_synthetic_thread():
    messages = pd.DataFrame(
        [
            {
                "tweet_id": "1",
                "author_id": "cust1",
                "text": "My order is late",
                "in_response_to_tweet_id": None,
                "response_tweet_id": "2",
                "inbound": True,
            },
            {
                "tweet_id": "2",
                "author_id": "AmazonHelp",
                "text": "Sorry to hear that, can you DM us your order number?",
                "in_response_to_tweet_id": "1",
                "response_tweet_id": None,
                "inbound": False,
            },
            {
                "tweet_id": "3",
                "author_id": "cust2",
                "text": "Unrelated message with no reply",
                "in_response_to_tweet_id": None,
                "response_tweet_id": None,
                "inbound": True,
            },
        ]
    )

    pairs = build_pairs(messages)

    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["customer_tweet_id"] == "1"
    assert row["agent_tweet_id"] == "2"
    assert row["customer_msg"] == "My order is late"
    assert row["agent_reply"].startswith("Sorry to hear that")
