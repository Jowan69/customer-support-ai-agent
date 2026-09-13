"""twcs.csv -> messages.parquet, restricted to threads touching @AmazonHelp.

`sample_frac` subsamples whole conversations, never individual rows. Sampling rows
would strand agent replies whose customer message was dropped, and pairs.py would
silently lose them - a subsample that quietly changes what the corpus *is*. Seeding
the thread walk from a fraction of the support tweets keeps every kept thread intact.

Source columns (Kaggle "Customer Support on Twitter" twcs.csv):
tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUPPORT_AUTHOR = "AmazonHelp"


def _thread_tweet_ids(
    df: pd.DataFrame,
    support_author: str,
    sample_frac: float | None = None,
    seed: int = 0,
) -> set[str]:
    """All tweet_ids that belong to a thread where `support_author` replied at least once.

    With `sample_frac`, the walk is seeded from that fraction of the support tweets and
    then expands outward as usual, so each surviving thread is whole.
    """
    support_tweets = df[df["author_id"] == support_author]
    if sample_frac is not None:
        support_tweets = support_tweets.sample(frac=sample_frac, random_state=seed)

    thread_ids: set[str] = set(support_tweets["tweet_id"])
    thread_ids |= set(support_tweets["in_response_to_tweet_id"].dropna())

    # walk outward: any tweet that responds to something already in the set joins it too
    frontier = thread_ids.copy()
    while frontier:
        responders = df[df["in_response_to_tweet_id"].isin(frontier)]["tweet_id"]
        new_ids = set(responders) - thread_ids
        if not new_ids:
            break
        thread_ids |= new_ids
        frontier = new_ids
    return thread_ids


def ingest(
    raw_csv: Path,
    out_path: Path,
    support_author: str = SUPPORT_AUTHOR,
    sample_frac: float | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    df = pd.read_csv(
        raw_csv,
        dtype={
            "tweet_id": str,
            "author_id": str,
            "response_tweet_id": str,
            "in_response_to_tweet_id": str,
        },
        parse_dates=["created_at"],
    )
    df["inbound"] = df["inbound"].astype(bool)

    keep_ids = _thread_tweet_ids(df, support_author, sample_frac, seed)
    out = df[df["tweet_id"].isin(keep_ids)].copy()

    out = out[
        [
            "tweet_id",
            "author_id",
            "text",
            "created_at",
            "in_response_to_tweet_id",
            "response_tweet_id",
            "inbound",
        ]
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out


if __name__ == "__main__":
    from config.settings import settings

    ingest(settings.raw_dir / "twcs.csv", settings.processed_dir / "messages.parquet")
