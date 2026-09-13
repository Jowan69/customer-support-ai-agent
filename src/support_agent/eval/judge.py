"""The LLM-judge scoring pass: five 1-5 dimensions per drafted reply.

Kept separate from the metrics, and cached to disk, for two reasons. It is the only
part of the eval that costs API calls, so rebuilding the report must not re-spend
them. And Phase 7 scores the same replies by hand: the human and the judge have to be
looking at an identical, frozen set of replies for the agreement number to mean
anything.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "judge.txt"

DIMENSIONS = ("groundedness", "relevance", "actionability", "tone", "safety")

_DIMENSION_PROPERTIES = {
    dimension: {"type": "integer", "minimum": 1, "maximum": 5} for dimension in DIMENSIONS
}

# Only the numeric dimensions are requested (and validated) - a `*_reason` field costs
# an API call's worth of tokens per dimension and, since nothing downstream reads it,
# was generated only to be thrown away in run().
JUDGE_SCHEMA = {
    "type": "object",
    "properties": _DIMENSION_PROPERTIES,
    "required": list(DIMENSIONS),
    "additionalProperties": False,
}


def build_prompt(message: str, reply: str, sources: list[str]) -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8").format(
        message=message, reply=reply, sources_block=format_sources(sources)
    )


def judge_one(message: str, reply: str, sources: list[str], client) -> dict:
    prompt = build_prompt(message, reply, sources)
    result = client.complete_json(prompt, json_schema=JUDGE_SCHEMA, schema_name="judge")
    for dimension in DIMENSIONS:
        # A judge that returns 7 or "good" would otherwise land in the mean and move
        # it. Fail on the row rather than average in a number off the scale. bool is a
        # subclass of int in Python, so True/False would otherwise pass as 1/0.
        value = result.get(dimension)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"judge returned {dimension}={value!r}, expected an integer 1-5")
    return result


def load_sources(artifact_dir: Path) -> dict[str, str]:
    """pair_id -> the agent reply that pair records, from the built index metadata.

    This is what "grounded in" means operationally: the drafter cited these pair_ids,
    so these are the replies its claims are supposed to trace back to. Both raters -
    the judge and the human scoring by hand in Phase 7 - have to be shown the same
    ones, or `groundedness` is two different questions and its kappa is meaningless.
    """
    meta_path = artifact_dir / "pairs_meta.parquet"
    if not meta_path.exists():
        return {}
    meta = pd.read_parquet(meta_path, columns=["pair_id", "agent_reply"])
    return {str(r.pair_id): str(r.agent_reply) for r in meta.itertuples()}


def format_sources(sources: list[str]) -> str:
    """The exact block the judge prompt renders, reused verbatim on the human sheet."""
    return "\n".join(f"- {s}" for s in sources) or "(none)"


def sources_for(row: pd.Series, sources_by_pair: dict[str, str]) -> list[str]:
    raw = row.get("grounded_pair_ids")
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        pair_ids = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [sources_by_pair[p] for p in pair_ids if p in sources_by_pair]


def run(
    merged: pd.DataFrame,
    client,
    *,
    out_path: Path,
    sources_by_pair: dict[str, str] | None = None,
    limit: int | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Score every row that produced a reply. Writes after each row so a failed run
    partway through is resumable rather than discarded."""
    sources_by_pair = sources_by_pair or {}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    scored: list[dict] = []
    if resume and out_path.exists():
        previous = pd.read_csv(out_path, dtype={"tweet_id": str})
        scored = previous.to_dict("records")
        done = set(previous["tweet_id"])

    todo = merged[merged["reply"].notna() & (merged["reply"].astype(str).str.strip() != "")]
    todo = todo[~todo["tweet_id"].isin(done)]
    # None means no limit; 0 must mean zero rows, not "no limit" (`if limit:` treats 0
    # as falsy and skips the head() entirely); a negative limit must be rejected rather
    # than silently reinterpreted by pandas' negative-head slicing.
    if limit is not None:
        if limit < 0:
            raise ValueError(f"judge-limit must be >= 0, got {limit}")
        todo = todo.head(limit)

    for _, row in todo.iterrows():
        result = judge_one(
            row["message"], row["reply"], sources_for(row, sources_by_pair), client
        )
        scored.append({"tweet_id": row["tweet_id"], **{d: result[d] for d in DIMENSIONS}})
        _write_csv_atomic(pd.DataFrame(scored), out_path)

    return pd.DataFrame(scored)


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write via a temp file plus rename so an interrupted write cannot leave a
    partially written `path` - the resume logic above trusts that if `path` exists it
    holds a complete, valid CSV."""
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)
