"""Scores a frozen run against the golden set and writes results/eval_report.md.

    python -m scripts.run_eval                   # the default: no API calls, no artifacts
    python -m scripts.run_eval --judge           # also run the LLM judge (costs calls)
    python -m scripts.run_eval --retrieval-live  # re-query the index (needs artifacts/)
    python -m scripts.run_eval --no-retrieval    # skip grounding entirely

The default path reads only files that are in the repo - `data/labels/golden.csv`,
`results/replies.csv`, `results/judge_scores.csv`, `data/labels/human_scores.csv` and
`frozen/pair_clusters.parquet` - and calls no model at all. That is deliberate: a clone
can regenerate the whole report in seconds without the 500MB artifact set, an API key,
or an hour of paced API calls.

`--judge` and `--retrieval-live` are the two opt-ins that need more than a clone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from config.settings import settings

from support_agent.eval import (
    agreement,
    baselines,
    classification,
    decisions,
    drafting,
    golden,
    report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=settings.labels_dir / "golden.csv")
    parser.add_argument("--run", type=Path, default=settings.root_dir / "results" / "replies.csv")
    parser.add_argument(
        "--out", type=Path, default=settings.root_dir / "results" / "eval_report.md"
    )
    parser.add_argument("--judge", action="store_true", help="run the LLM judge (costs API calls)")
    parser.add_argument("--judge-limit", type=int, default=None)
    parser.add_argument(
        "--retrieval-live",
        action="store_true",
        help="re-query the FAISS index instead of scoring the frozen run's citations "
        "(needs the full artifacts/<version>/ set)",
    )
    parser.add_argument("--no-retrieval", action="store_true")
    args = parser.parse_args()

    results_dir = args.out.parent
    judge_path = results_dir / "judge_scores.csv"
    human_path = settings.labels_dir / "human_scores.csv"

    intents = golden.load_intents(settings.intents_path)
    gold = golden.load_golden(args.golden, valid_intents=intents)
    run_df = golden.load_run(args.run)
    merged, unscored = golden.join(gold, run_df)

    if merged.empty:
        raise SystemExit("golden set and run share no tweet_ids - check the join key")

    print(f"scoring {len(merged)} rows ({len(unscored)} labelled rows not in the run)")

    if args.judge:
        from support_agent.eval import judge as judge_module
        from support_agent.llm.grok_client import get_client

        # Without this the judge was scoring groundedness against "(none)" - every
        # reply trivially unsupported, and the dimension measuring nothing.
        sources = judge_module.load_sources(settings.artifact_dir)
        if not sources:
            print("warning: no pairs_meta.parquet - judging groundedness without sources")

        print("running the judge (cached; safe to interrupt and resume)")
        judge_module.run(
            merged,
            get_client(),
            out_path=judge_path,
            sources_by_pair=sources,
            limit=args.judge_limit,
        )

    judge_scores = (
        pd.read_csv(judge_path, dtype={"tweet_id": str}) if judge_path.exists() else None
    )

    retrieval_score = None
    if not args.no_retrieval:
        from support_agent.eval import retrieval

        try:
            if args.retrieval_live:
                # Measures the retriever, not this run: it re-queries the index and
                # scores whatever comes back now, cited or not.
                print("scoring retrieval against the live index")
                retrieval_score = retrieval.run_live(gold, settings.artifact_dir)
            else:
                # frozen_dir first: a clone has frozen/ and no artifacts, and a working
                # copy that has rebuilt artifacts must still score against the partition
                # the golden labels were drawn from.
                print("scoring grounding from the frozen run")
                assignments = retrieval.load_pair_clusters(
                    settings.frozen_dir, settings.artifact_dir
                )
                retrieval_score = retrieval.run_frozen(merged, gold, assignments)
        except Exception as exc:  # noqa: BLE001 - a missing file must not sink the report
            print(f"grounding skipped: {exc}")

    decision_score = decisions.run(merged)
    decision_baseline_score = decisions.always_escalate_baseline(merged["expected_action"])

    print(f"scoring baselines (leave-one-out over {len(gold)} golden rows)")
    baseline_scores = baselines.run(gold)

    text = report.build_report(
        meta={
            "n_golden": len(gold),
            "n_run": len(run_df),
            "n_scored": len(merged),
            "unscored": unscored,
            "golden_path": args.golden.name,
            "run_path": args.run.name,
            "artifact_version": settings.artifact_version,
            "intents_version": settings.intents_version,
            "generated_at": report.now(),
        },
        classification=classification.run(merged),
        confusions=classification.confusion_pairs(merged["intent_pred"], merged["intent_gold"]),
        decisions=decision_score,
        reasons=decisions.reason_breakdown(merged),
        drafting=drafting.run(
            merged, judge_scores, cache_path=results_dir / drafting.CACHE_NAME
        ),
        retrieval=retrieval_score,
        agreement=agreement.run(human_path, judge_path),
        baselines=baseline_scores,
        decision_baseline=decision_baseline_score,
    )
    path = report.write_report(text, args.out)
    print(f"wrote {path}")

    false_auto = decisions.false_auto_rows(merged)
    if not false_auto.empty:
        false_auto.to_csv(results_dir / "false_auto.csv", index=False)
        print(f"{len(false_auto)} false-auto rows -> {results_dir / 'false_auto.csv'}")

    print(json.dumps({k: v for k, v in decision_score.items()
                      if k != "confusion_matrix"}, indent=2))


if __name__ == "__main__":
    main()
