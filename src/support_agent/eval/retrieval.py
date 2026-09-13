"""Was the drafter handed the right kind of past case?

There are two ways to ask that, and they are not the same question.

`run_frozen` reads the pairs the frozen run actually grounded on - the
`grounded_pair_ids` column of `results/replies.csv` - and asks whether they were the
right kind of case. It needs no FAISS index and no encoder, only the frozen run and
`frozen/pair_clusters.parquet` (1.5MB, in the repo). So it is the path that works on a
fresh clone, and - like every other table in the report - it describes the one frozen
run rather than a fresh pass over the corpus.

`run_live` re-queries the index for each golden message and scores the top-k that comes
back. That measures the retriever rather than the run, needs the full ~500MB artifact
set, and is opt-in (`run_eval --retrieval-live`).

The report says which one produced its numbers, because they measure different things:
the frozen path scores what the drafter cited, the live path scores what the index
returned whether it was cited or not.

The intent of a retrieved pair is its cluster's modal hand-labelled intent, taken from
the golden rows that fell in that cluster. The sampler's `cluster_floor` guarantees at
least six labelled rows per cluster, so each mapping rests on six human judgements
rather than one.

This is a proxy and the report says so. A cluster is not an intent - some clusters split
across two - so a "miss" here can mean the retrieved case was fine and the cluster is
mixed. Read it as: did the draft rest on the right neighbourhood of the corpus?
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import pandas as pd

_CLUSTER_FILE = "pair_clusters.parquet"


class RetrievalScore(TypedDict):
    top_k_match_rate: float
    top_1_match_rate: float
    mean_top1_score: float | None
    mean_top1_score_on_hit: float | None
    mean_top1_score_on_miss: float | None
    n: int
    k: int
    n_clusters_mapped: int
    source: str
    n_rows: int
    n_no_neighbour: int


def cluster_intent_map(golden: pd.DataFrame) -> dict[int, str]:
    """cluster_id -> the intent humans most often gave rows in that cluster."""
    if "cluster_id" not in golden.columns:
        return {}
    modes = (
        golden.groupby("cluster_id")["intent"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )
    return {int(k): str(v) for k, v in modes.items()}


def pair_intent_map(golden: pd.DataFrame, assignments: pd.DataFrame) -> dict[str, str]:
    """pair_id -> proxy intent, via the pair's cluster."""
    cluster_to_intent = cluster_intent_map(golden)
    return {
        str(row.pair_id): cluster_to_intent[int(row.cluster_id)]
        for row in assignments.itertuples()
        if int(row.cluster_id) in cluster_to_intent
    }


def load_pair_clusters(*candidates: Path) -> pd.DataFrame:
    """First `pair_clusters.parquet` found among the given directories.

    `frozen/` is checked before `artifacts/<version>/` so a clone with no artifacts
    still scores retrieval, and a working copy that has rebuilt its artifacts still
    reads the assignment the golden labels were drawn against.
    """
    for directory in candidates:
        path = Path(directory) / _CLUSTER_FILE
        if path.exists():
            return pd.read_parquet(path)
    looked = ", ".join(str(Path(c) / _CLUSTER_FILE) for c in candidates)
    raise FileNotFoundError(f"no {_CLUSTER_FILE} found - looked in: {looked}")


def strip_self_pair(tweet_id: str, pair_ids: list[str]) -> list[str]:
    """Drop the query's own pair from a retrieved list.

    `online/retrieve.py` has no self-match filter, and the golden rows were drawn from
    the indexed corpus, so a message's own pair sits in the index and comes back at
    rank 1 with a similarity of ~1.0. In the frozen run that happened for 181 of the
    187 rows that retrieved anything. Counting it would score the index on its ability
    to find the row it was handed.

    A pair_id is `"<customer_tweet_id>_<reply_tweet_id>"`, so the row's own pair is
    identifiable from the id alone - which is what makes this path artifact-free.
    `pairs_meta.parquet`, the file that holds the message text, is 38MB and not in the
    repo.
    """
    prefix = f"{tweet_id}_"
    return [p for p in pair_ids if not str(p).startswith(prefix)]


def parse_pair_ids(raw: object) -> list[str]:
    """The `grounded_pair_ids` cell, which run_online writes as a JSON array."""
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(x) for x in parsed] if isinstance(parsed, list) else []


def score_rows(
    rows: list[tuple[str, list[str], list[float]]],
    pair_to_intent: dict[str, str],
    *,
    source: str = "live_index",
    n_rows: int | None = None,
) -> RetrievalScore:
    """rows: (gold_intent, retrieved_pair_ids, retrieved_scores) per scored query.

    Rows with no usable neighbour are expected to be excluded by the caller, not passed
    in empty: a row that retrieved nothing is not a retrieval miss, and counting it as
    one would report the drafter's citation rate as a retrieval failure. The count of
    what was dropped is carried in `n_no_neighbour` so the report can say so.
    """
    hits = top1_hits = 0
    top1_scores: list[float] = []
    hit_scores: list[float] = []
    miss_scores: list[float] = []

    for gold_intent, pair_ids, scores in rows:
        intents = [pair_to_intent.get(p) for p in pair_ids]
        is_hit = gold_intent in intents
        hits += int(is_hit)
        top1_hits += int(bool(intents) and intents[0] == gold_intent)

        if scores:
            top1_scores.append(scores[0])
            (hit_scores if is_hit else miss_scores).append(scores[0])

    n = len(rows)
    total = n if n_rows is None else n_rows

    def mean(xs: list[float]) -> float | None:
        # None, not 0.0: the frozen path has no per-neighbour similarity to report, and
        # a zero there reads as "the neighbours were dissimilar" rather than "unknown".
        return float(sum(xs) / len(xs)) if xs else None

    return RetrievalScore(
        top_k_match_rate=hits / n if n else 0.0,
        top_1_match_rate=top1_hits / n if n else 0.0,
        mean_top1_score=mean(top1_scores),
        mean_top1_score_on_hit=mean(hit_scores),
        mean_top1_score_on_miss=mean(miss_scores),
        n=n,
        k=max((len(p) for _, p, _ in rows), default=0),
        n_clusters_mapped=len(set(pair_to_intent.values())),
        source=source,
        n_rows=total,
        n_no_neighbour=max(total - n, 0),
    )


def run_frozen(
    merged: pd.DataFrame,
    golden: pd.DataFrame,
    assignments: pd.DataFrame,
) -> RetrievalScore:
    """Score the pairs the frozen run grounded on. No index, no encoder, no API.

    `merged` is the golden set joined to the frozen run, so `intent_gold` is the human
    label and `grounded_pair_ids` is what the drafter cited.

    No similarity numbers come out of this path. The run's `top1_sim` column is the
    self-match score - ~1.0 on all 200 rows - so it says nothing about the neighbours
    that remain once the row's own pair is removed, and there is nothing else in the
    frozen CSV to put in its place. Reported as unavailable rather than filled in.
    """
    pair_to_intent = pair_intent_map(golden, assignments)

    rows: list[tuple[str, list[str], list[float]]] = []
    for _, row in merged.iterrows():
        cited = parse_pair_ids(row.get("grounded_pair_ids"))
        pair_ids = strip_self_pair(str(row["tweet_id"]), cited)
        if not pair_ids:
            continue
        rows.append((str(row["intent_gold"]), pair_ids, []))

    return score_rows(rows, pair_to_intent, source="frozen_run", n_rows=len(merged))


def run_live(golden: pd.DataFrame, artifact_dir: Path, k: int | None = None) -> RetrievalScore:
    """Re-query the index for every golden message. Needs the full artifact set."""
    from support_agent.offline.clusters import pair_cluster_assignments
    from support_agent.online.retrieve import retrieve

    pair_to_intent = pair_intent_map(golden, pair_cluster_assignments(artifact_dir))

    rows: list[tuple[str, list[str], list[float]]] = []
    for _, row in golden.iterrows():
        message = str(row["message"])
        retrieved = retrieve(message, k=k)
        # Same self-match trap as the frozen path, caught on the message text here
        # because the live call has it to hand.
        retrieved = [p for p in retrieved if str(p["customer_msg"]).strip() != message.strip()]
        if not retrieved:
            continue
        rows.append(
            (
                str(row["intent"]),
                [str(p["pair_id"]) for p in retrieved],
                [float(p["score"]) for p in retrieved],
            )
        )
    return score_rows(rows, pair_to_intent, source="live_index", n_rows=len(golden))


# The old name, from when re-querying the index was the only way to score retrieval.
run = run_live
