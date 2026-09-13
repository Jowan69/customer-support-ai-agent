"""Forward walk each thread via response_tweet_id -> chains.parquet.

is_repeat_contact: the customer sent a SUBSTANTIVE further inbound message after
the first agent reply in the chain. "Substantive" means the message clears the
same length/word gate corpus_filter.py already applies before a message is
usable for retrieval (settings.min_message_chars / settings.min_message_words),
computed the same way corpus_filter.py computes it: clean_text() first, then its
own content/word-count helpers. One gate, reused - not a second one invented to
mean roughly the same thing, and not a keyword blacklist either: "thanks so much
for looking into this, really appreciate the quick help" is long and wordy and
would still pass, which is correct - a substantive message is substantive
whatever its sentiment.

It still does not mean the reply failed to resolve the problem, and it must not
be read that way even after this narrowing - a detailed thank-you sets it exactly
as readily as a detailed complaint. What it excludes now is bare acknowledgements
and single-emoji replies ("thanks!", "danke", "\U0001F44D"), which used to set it
just the same as a genuine follow-up.

Before this narrowing, ANY further inbound message counted, acknowledgements
included - cluster 5 of the v2 taxonomy is 5,200 messages of exactly that shape.
online/decide.py wired this column in unnarrowed once, and its own eval showed
that made repeat_contact the worst-performing signal it had (3.8% correct on the
auto rows it drove), so it was dropped from decide() rather than trusted. This is
the fix that makes it safe to reconsider wiring back in.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from support_agent.offline.corpus_filter import _content_only, _word_count
from support_agent.text import clean_text

SUPPORT_AUTHOR = "AmazonHelp"


def _split_response_ids(raw: str | float) -> list[str]:
    if not isinstance(raw, str) or not raw:
        return []
    return raw.split(",")


def _is_substantive(text: str, *, min_chars: int, min_words: int) -> bool:
    """Same gate corpus_filter.filter_pairs() applies to retrieval pairs, applied
    here to one candidate follow-up message."""
    content = _content_only(clean_text(text))
    return len(content) >= min_chars and _word_count(content) >= min_words


def build_chains(
    messages: pd.DataFrame,
    support_author: str = SUPPORT_AUTHOR,
    *,
    min_chars: int,
    min_words: int,
) -> pd.DataFrame:
    by_id = messages.set_index("tweet_id")
    roots = messages[
        messages["inbound"] & messages["in_response_to_tweet_id"].isna()
    ]

    rows = []
    for root_id in roots["tweet_id"]:
        chain = [root_id]
        frontier = _split_response_ids(by_id.loc[root_id, "response_tweet_id"])
        seen_agent_reply = False
        is_repeat_contact = False

        while frontier:
            next_frontier: list[str] = []
            for tid in frontier:
                if tid not in by_id.index:
                    continue
                chain.append(tid)
                row = by_id.loc[tid]
                if row["author_id"] == support_author:
                    seen_agent_reply = True
                elif (
                    row["inbound"]
                    and seen_agent_reply
                    and _is_substantive(row["text"], min_chars=min_chars, min_words=min_words)
                ):
                    is_repeat_contact = True
                next_frontier.extend(_split_response_ids(row["response_tweet_id"]))
            frontier = next_frontier

        rows.append(
            {
                "chain_id": root_id,
                "root_tweet_id": root_id,
                "tweet_ids": chain,
                "is_repeat_contact": is_repeat_contact,
            }
        )

    return pd.DataFrame(rows)


def run(processed_dir: Path, *, min_chars: int, min_words: int) -> pd.DataFrame:
    messages = pd.read_parquet(processed_dir / "messages.parquet")
    chains = build_chains(messages, min_chars=min_chars, min_words=min_words)
    chains.to_parquet(processed_dir / "chains.parquet", index=False)
    return chains


if __name__ == "__main__":
    from config.settings import settings

    run(
        settings.processed_dir,
        min_chars=settings.min_message_chars,
        min_words=settings.min_message_words,
    )
