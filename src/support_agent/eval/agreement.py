"""Cohen's kappa between the human scores and the LLM judge, per dimension.

The judge is the only thing scoring 200 replies, so the report leans on it. This is
the check on whether it can be leaned on. Without it, "mean groundedness 4.1" is a
number produced by the same family of model that wrote the replies, agreeing with
itself.

Quadratic weights, because the scale is ordinal: a judge that says 4 where the human
said 5 is nearly right, and unweighted kappa would score that identically to saying 1.

Two traps this module reports rather than hides:

  degenerate  if the human gave nearly every reply the same score, kappa collapses
              toward 0 no matter how well the two agree, because there is no variance
              left to explain. A kappa of 0.05 next to 92% exact agreement means the
              dimension is uninformative, not that the judge failed. Both are reported
              side by side so the pair can be read together.
  small n     50 items gives a wide confidence interval. The number is a sanity check
              on the judge, not a published psychometric.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pandas as pd

from support_agent.eval import judge
from support_agent.eval.judge import DIMENSIONS

# The 1-5 integer scale every dimension is scored on (see judge.JUDGE_SCHEMA).
_SCORE_MIN, _SCORE_MAX = 1, 5


class AgreementRow(TypedDict):
    dimension: str
    kappa: float | None
    exact_agreement: float
    within_one: float
    mean_human: float
    mean_judge: float
    n: int
    degenerate: bool
    note: str


def _kappa(human: pd.Series, judge_scores: pd.Series) -> float | None:
    from sklearn.metrics import cohen_kappa_score

    # Both raters constant -> kappa is undefined (0/0), and sklearn returns nan.
    # Report None and say why rather than printing nan into a table.
    if human.nunique() == 1 and judge_scores.nunique() == 1:
        return None
    value = cohen_kappa_score(human, judge_scores, weights="quadratic")
    return None if pd.isna(value) else float(value)


def _reject_duplicate_tweet_ids(frame: pd.DataFrame, label: str) -> None:
    """Matches the duplicate guard in golden.load_run(): a duplicated tweet_id means
    the scores file was appended to rather than replaced, and merging it in as-is would
    silently multiply rows instead of raising."""
    duplicated = frame["tweet_id"].duplicated()
    if duplicated.any():
        ids = frame.loc[duplicated, "tweet_id"].head(5).tolist()
        raise ValueError(
            f"{duplicated.sum()} duplicate tweet_ids in the {label} scores "
            f"(e.g. {ids}) - each tweet_id must appear once."
        )


def _validated_scores(pair: pd.DataFrame, column: str, rater: str) -> pd.Series:
    """Coerce a scored column to int, rejecting anything that is not actually an
    integer 1-5 rather than silently truncating or accepting bools (bool is a subclass
    of int in Python, so `.astype(int)` alone would accept True/False as 1/0)."""
    values: dict = {}
    invalid: list[tuple] = []
    for index, value in pair[column].items():
        if isinstance(value, bool):
            invalid.append((index, value))
            continue
        if isinstance(value, int):
            int_value = value
        elif isinstance(value, float) and value.is_integer():
            int_value = int(value)
        else:
            invalid.append((index, value))
            continue
        if not _SCORE_MIN <= int_value <= _SCORE_MAX:
            invalid.append((index, value))
            continue
        values[index] = int_value
    if invalid:
        bad = invalid[:5]
        raise ValueError(
            f"{rater} {column!r} has {len(invalid)} value(s) that are not integers "
            f"{_SCORE_MIN}-{_SCORE_MAX}: {bad}"
        )
    return pd.Series(values, dtype=int)


def score(human: pd.DataFrame, judge_scores: pd.DataFrame) -> list[AgreementRow]:
    """Both frames keyed by tweet_id, one column per dimension, integers 1-5."""
    _reject_duplicate_tweet_ids(human, "human")
    _reject_duplicate_tweet_ids(judge_scores, "judge")
    merged = human.merge(judge_scores, on="tweet_id", suffixes=("_human", "_judge"))

    rows: list[AgreementRow] = []
    for dimension in DIMENSIONS:
        human_col, judge_col = f"{dimension}_human", f"{dimension}_judge"
        if human_col not in merged.columns or judge_col not in merged.columns:
            continue

        pair = merged[[human_col, judge_col]].dropna()
        if pair.empty:
            continue
        h = _validated_scores(pair, human_col, "human")
        j = _validated_scores(pair, judge_col, "judge")

        diff = (h - j).abs()
        degenerate = h.nunique() == 1 or j.nunique() == 1
        kappa = _kappa(h, j)

        if kappa is None:
            note = "undefined - both raters gave a single value throughout"
        elif degenerate:
            note = "one rater used a single value; read the agreement columns instead"
        elif kappa < 0.2:
            note = "poor agreement - the judge is not measuring what you are"
        elif kappa < 0.4:
            note = "fair"
        elif kappa < 0.6:
            note = "moderate"
        elif kappa < 0.8:
            note = "substantial"
        else:
            note = "almost perfect"

        rows.append(
            AgreementRow(
                dimension=dimension,
                kappa=kappa,
                exact_agreement=float((diff == 0).mean()),
                within_one=float((diff <= 1).mean()),
                mean_human=float(h.mean()),
                mean_judge=float(j.mean()),
                n=len(pair),
                degenerate=degenerate,
                note=note,
            )
        )
    return rows


def run(human_path: Path, judge_path: Path) -> list[AgreementRow]:
    if not human_path.exists() or not judge_path.exists():
        return []
    human = pd.read_csv(human_path, dtype={"tweet_id": str})
    judge_scores = pd.read_csv(judge_path, dtype={"tweet_id": str})
    return score(human, judge_scores)


def describe(human_path: Path, judge_path: Path) -> str:
    """Why `run()` might have returned no rows, without changing its return type.

    "missing_files" - the human scores file, the judge scores file, or both, do not
                       exist yet
    "no_overlap"    - both files exist, but share no tweet_ids, or no dimension column
                       names overlap, so there is nothing to compare
    "ok"            - rows are available
    """
    if not human_path.exists() or not judge_path.exists():
        return "missing_files"
    return "ok" if run(human_path, judge_path) else "no_overlap"


def make_score_sheet(
    merged: pd.DataFrame,
    n: int,
    seed: int = 0,
    sources_by_pair: dict[str, str] | None = None,
) -> pd.DataFrame:
    """The blank sheet for Phase 7: `n` replies to score by hand.

    Carries a `sources` column holding the same source block the judge was shown,
    rendered by the same function. Without it the two raters are answering different
    questions on `groundedness` - the judge checks claims against the retrieved cases,
    while a human with only the message and the reply can only guess at what was
    retrieved - and the kappa for that dimension measures the gap between the two
    views rather than between the two raters.

    Judge scores are deliberately not pre-filled, for the same reason the intent
    labels were not: agreement with a number already on the page in front of you is
    not agreement.
    """
    pool = merged[merged["reply"].notna() & (merged["reply"].astype(str).str.strip() != "")]
    picked = pool.sample(n=min(n, len(pool)), random_state=seed)

    sheet = picked[["tweet_id", "message", "reply"]].copy()
    sheet["sources"] = [
        judge.format_sources(judge.sources_for(row, sources_by_pair or {}))
        for _, row in picked.iterrows()
    ]
    for dimension in DIMENSIONS:
        sheet[dimension] = ""
    sheet["note"] = ""
    return sheet.reset_index(drop=True)
