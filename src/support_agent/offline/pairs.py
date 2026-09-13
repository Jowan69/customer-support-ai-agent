"""Single-hop join: each inbound customer message -> its first AmazonHelp reply."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUPPORT_AUTHOR = "AmazonHelp"


def build_pairs(messages: pd.DataFrame, support_author: str = SUPPORT_AUTHOR) -> pd.DataFrame:
    customer = messages[messages["inbound"]].copy()
    agent = messages[messages["author_id"] == support_author].copy()

    # Support often answers one tweet with several: keep only the first reply per
    # parent, otherwise the join fans out and the corpus fills with near-duplicates.
    agent = agent.dropna(subset=["in_response_to_tweet_id"])
    if "created_at" in agent.columns:
        agent = agent.sort_values("created_at")
    agent = agent.drop_duplicates(subset=["in_response_to_tweet_id"], keep="first")

    agent_by_parent = agent.set_index("in_response_to_tweet_id")

    pairs = customer.join(
        agent_by_parent[["tweet_id", "text"]].rename(
            columns={"tweet_id": "agent_tweet_id", "text": "agent_reply"}
        ),
        on="tweet_id",
        how="inner",
    )

    pairs = pairs.rename(columns={"tweet_id": "customer_tweet_id", "text": "customer_msg"})
    pairs["pair_id"] = pairs["customer_tweet_id"] + "_" + pairs["agent_tweet_id"]

    return pairs[["pair_id", "customer_tweet_id", "customer_msg", "agent_tweet_id", "agent_reply"]]


def run(processed_dir: Path) -> pd.DataFrame:
    messages = pd.read_parquet(processed_dir / "messages.parquet")
    pairs = build_pairs(messages)
    pairs.to_parquet(processed_dir / "pairs.parquet", index=False)
    return pairs


if __name__ == "__main__":
    from config.settings import settings

    run(settings.processed_dir)
