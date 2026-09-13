"""Reply quality: similarity to the brand's real reply, plus the judge's five dimensions.

Cosine similarity against `real_reply` is reported but deliberately not led with. Two
replies can solve the same problem in different words and score 0.4; a reply that
copies the real one's phrasing while promising a refund it cannot give scores 0.9.
It is a drift check - "does this sound like this brand at all" - and the judge
dimensions are the quality measure.

Nothing here calls the drafter. Replies come from the frozen run, so the reply scored
in this table is the reply that produced the action in the decisions table.

The similarity numbers are cached to `results/reply_similarity.csv` and read back from
there. Computing them needs the sentence-transformers encoder - a 470MB download on a
machine that has never run the offline build - and the default eval path is supposed to
run on a fresh clone in seconds. The cache is a deterministic function of the frozen
replies and the encoder name, so reading it back is not a shortcut, it is the same
answer without the download. If it is absent and the encoder cannot be loaded, the
similarity is reported as not computed rather than as zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from support_agent.eval.judge import DIMENSIONS

CACHE_NAME = "reply_similarity.csv"


class DraftingScore(TypedDict):
    mean_cosine_sim: float | None
    median_cosine_sim: float | None
    judge_means: dict[str, float]
    judge_low_count: dict[str, int]
    reply_rate: float
    n_scored: int
    n_judged: int
    n_total: int


def cosine_similarities(replies: list[str], references: list[str], encoder) -> list[float]:
    if not replies:
        return []
    left = encoder.encode(replies, normalize_embeddings=True)
    right = encoder.encode(references, normalize_embeddings=True)
    return [float(np.dot(a, b)) for a, b in zip(left, right, strict=True)]


def load_cached_similarity(path: Path) -> dict[str, float]:
    """tweet_id -> cosine similarity, from a previous run that had the encoder."""
    if not path.exists():
        return {}
    cached = pd.read_csv(path, dtype={"tweet_id": str})
    if not {"tweet_id", "cosine_sim"} <= set(cached.columns):
        return {}
    return {
        str(r.tweet_id): float(r.cosine_sim)
        for r in cached.itertuples()
        if pd.notna(r.cosine_sim)
    }


def write_cached_similarity(path: Path, sims: dict[str, float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {"tweet_id": list(sims), "cosine_sim": [sims[k] for k in sims]}
    ).sort_values("tweet_id")
    frame.to_csv(path, index=False)
    return path


def run(
    merged: pd.DataFrame,
    judge_scores: pd.DataFrame | None = None,
    *,
    cache_path: Path | None = None,
) -> DraftingScore:
    def _present(column: str) -> pd.Series:
        if column not in merged.columns:
            return pd.Series(False, index=merged.index)
        return merged[column].notna() & (merged[column].astype(str).str.strip() != "")

    has_reply = _present("reply")
    # A blank `real_reply` is not a zero-similarity reply, it is a row with no
    # reference to compare against - averaging it in would drag the mean toward a
    # number about missing data.
    scorable = merged[has_reply & _present("real_reply")]

    cached = load_cached_similarity(cache_path) if cache_path else {}
    ids = [str(t) for t in scorable["tweet_id"]] if "tweet_id" in scorable.columns else []

    sims: list[float] = []
    if not scorable.empty and ids and all(i in cached for i in ids):
        sims = [cached[i] for i in ids]
    elif not scorable.empty:
        try:
            from config.settings import settings
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(settings.embedding_model)
            sims = cosine_similarities(
                scorable["reply"].astype(str).tolist(),
                scorable["real_reply"].astype(str).tolist(),
                encoder,
            )
            if cache_path and ids:
                write_cached_similarity(cache_path, dict(zip(ids, sims, strict=True)))
        except Exception:  # noqa: BLE001
            # No encoder and no cache. The drift check is the one number in this report
            # that cannot be recovered from the frozen files alone, and a missing
            # optional metric must not take the other eleven tables down with it.
            sims = []

    judge_means: dict[str, float] = {}
    judge_low: dict[str, int] = {}
    n_judged = 0
    if judge_scores is not None and not judge_scores.empty:
        # Same reasoning as `scorable` above: judge_scores can carry rows from earlier
        # or unrelated runs (it is a cached, append-only-by-resume file), so scope it
        # to this run's tweet_ids before aggregating, exactly like the cosine path.
        judge_scores = judge_scores[judge_scores["tweet_id"].isin(merged["tweet_id"])]
    if judge_scores is not None and not judge_scores.empty:
        n_judged = len(judge_scores)
        for dimension in DIMENSIONS:
            if dimension in judge_scores.columns:
                judge_means[dimension] = float(judge_scores[dimension].mean())
                # 1s and 2s are the replies that should never have been drafted.
                # A mean of 4.1 with eleven 1s on `safety` is not a passing grade.
                judge_low[dimension] = int((judge_scores[dimension] <= 2).sum())

    return DraftingScore(
        mean_cosine_sim=float(np.mean(sims)) if sims else None,
        median_cosine_sim=float(np.median(sims)) if sims else None,
        judge_means=judge_means,
        judge_low_count=judge_low,
        reply_rate=float(has_reply.mean()) if len(merged) else 0.0,
        n_scored=len(sims),
        n_judged=n_judged,
        n_total=len(merged),
    )
