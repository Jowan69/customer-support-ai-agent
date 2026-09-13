"""Loading and validating the two files every eval stage reads.

There is one labelled input - `data/labels/golden.csv` - and one frozen run -
`results/replies.csv`. Every metric in the report is computed from those two joined
together, and nothing in `eval/` calls the pipeline.

That is the whole point. The old harness re-ran `classify()` for the classification
score, `handle()` for the decision score and `draft()` again for the drafting score:
three passes over a nondeterministic model, so the intent that scored in one table was
not the intent that produced the action in the next. Numbers computed off different
runs cannot be read against each other. Freeze one run, score that run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

GOLDEN_COLUMNS = ("tweet_id", "message", "intent", "expected_action")
RUN_COLUMNS = ("tweet_id", "intent", "action")
ACTIONS = ("auto", "escalate")


class GoldenError(ValueError):
    """The labelled set is not usable - raised early and loudly, never worked around."""


def load_intents(intents_path: Path) -> list[str]:
    payload = json.loads(intents_path.read_text(encoding="utf-8"))
    return [i["name"] for i in payload["intents"]]


def load_golden(path: Path, *, valid_intents: list[str] | None = None) -> pd.DataFrame:
    """The hand-labelled set. Blank or unknown labels are an error, not a dropped row.

    Silently dropping unlabelled rows would quietly shrink the evaluation and report a
    confident number over whatever happened to be finished.
    """
    if not path.exists():
        raise GoldenError(f"{path} does not exist - run scripts/make_golden_sample.py first")

    # tweet_id is numeric in the CSV and is the join key; letting pandas infer it here
    # and as int64 elsewhere is how the join silently produces zero rows.
    golden = pd.read_csv(path, dtype={"tweet_id": str})

    missing = [c for c in GOLDEN_COLUMNS if c not in golden.columns]
    if missing:
        raise GoldenError(f"{path} is missing columns: {missing}")

    for column in ("intent", "expected_action"):
        blank = golden[column].isna() | (golden[column].astype(str).str.strip() == "")
        if blank.any():
            ids = golden.loc[blank, "tweet_id"].head(5).tolist()
            raise GoldenError(
                f"{blank.sum()} of {len(golden)} rows have no `{column}` "
                f"(e.g. tweet_id {ids}). Finish labelling before running the eval."
            )

    golden["intent"] = golden["intent"].astype(str).str.strip()
    golden["expected_action"] = golden["expected_action"].astype(str).str.strip().str.lower()

    bad_actions = sorted(set(golden["expected_action"]) - set(ACTIONS))
    if bad_actions:
        raise GoldenError(f"unrecognised expected_action values: {bad_actions}")

    if valid_intents is not None:
        unknown = sorted(set(golden["intent"]) - set(valid_intents))
        if unknown:
            raise GoldenError(
                f"intent labels not in config/intents.json: {unknown}. "
                "Either fix the label or add the intent to the taxonomy."
            )

    return golden


def load_run(path: Path) -> pd.DataFrame:
    """The frozen agent run being scored."""
    if not path.exists():
        raise GoldenError(f"{path} does not exist - freeze a run with scripts/run_online.py --out")

    run = pd.read_csv(path, dtype={"tweet_id": str})
    missing = [c for c in RUN_COLUMNS if c not in run.columns]
    if missing:
        raise GoldenError(f"{path} is missing columns: {missing}")

    # A rerun that appended instead of replacing would double every row and halve
    # nothing visibly - the metrics would just quietly be computed twice.
    duplicated = run["tweet_id"].duplicated()
    if duplicated.any():
        raise GoldenError(
            f"{duplicated.sum()} duplicate tweet_ids in {path} - the run was appended to "
            "rather than replaced. Delete it and re-run."
        )
    return run


def join(golden: pd.DataFrame, run: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Inner join on tweet_id, plus the ids the run did not cover.

    Returned rather than warned about: coverage belongs in the report. A run over 140
    of 200 rows is a legitimate thing to score, but not a legitimate thing to describe
    as "the golden set".
    """
    merged = golden.merge(run, on="tweet_id", how="inner", suffixes=("_gold", "_pred"))
    unscored = sorted(set(golden["tweet_id"]) - set(run["tweet_id"]))
    return merged, unscored
