import pandas as pd

from support_agent.offline.ingest import _thread_tweet_ids

SUPPORT = "AmazonHelp"


def _thread(root: str, customer: str) -> list[dict]:
    """customer opens -> agent replies -> customer follows up."""
    return [
        {"tweet_id": f"{root}a", "author_id": customer,
         "in_response_to_tweet_id": None, "inbound": True},
        {"tweet_id": f"{root}b", "author_id": SUPPORT,
         "in_response_to_tweet_id": f"{root}a", "inbound": False},
        {"tweet_id": f"{root}c", "author_id": customer,
         "in_response_to_tweet_id": f"{root}b", "inbound": True},
    ]


def _corpus(n_threads: int) -> pd.DataFrame:
    rows = []
    for i in range(n_threads):
        rows += _thread(f"t{i}", f"cust{i}")
    return pd.DataFrame(rows)


def test_without_sampling_every_thread_is_kept():
    df = _corpus(10)

    assert len(_thread_tweet_ids(df, SUPPORT)) == 30


def test_sampling_keeps_whole_threads_not_loose_messages():
    """A stranded agent reply whose customer message was dropped would be silently
    lost by pairs.py - the subsample would change what the corpus is."""
    df = _corpus(20)

    kept = _thread_tweet_ids(df, SUPPORT, sample_frac=0.5, seed=0)

    roots = {tid[:-1] for tid in kept}
    for root in roots:
        assert {f"{root}a", f"{root}b", f"{root}c"} <= kept, f"{root} came back in pieces"


def test_sampling_keeps_roughly_the_requested_fraction():
    df = _corpus(100)

    kept = _thread_tweet_ids(df, SUPPORT, sample_frac=0.2, seed=0)

    assert 15 <= len({tid[:-1] for tid in kept}) <= 25


def test_sampling_is_reproducible():
    df = _corpus(50)

    assert _thread_tweet_ids(df, SUPPORT, 0.3, seed=7) == _thread_tweet_ids(df, SUPPORT, 0.3, seed=7)


def test_a_different_seed_selects_a_different_subset():
    df = _corpus(50)

    assert _thread_tweet_ids(df, SUPPORT, 0.3, seed=1) != _thread_tweet_ids(df, SUPPORT, 0.3, seed=2)
