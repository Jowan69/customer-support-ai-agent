"""Blank sheet for Phase 7: N frozen replies to score by hand against the judge anchors.

    python -m scripts.make_score_sheet --n 50

Scores are left blank on purpose. Agreement with a number already printed on the page
is not agreement. The anchors are in src/support_agent/prompts/judge.txt - score
against those, not against your own scale, or kappa measures the gap between two
rubrics rather than between two raters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config.settings import settings

from support_agent.eval import agreement, golden, judge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--golden", type=Path, default=settings.labels_dir / "golden.csv")
    parser.add_argument("--run", type=Path, default=settings.root_dir / "results" / "replies.csv")
    parser.add_argument("--out", type=Path, default=settings.labels_dir / "human_scores.csv")
    args = parser.parse_args()

    gold = golden.load_golden(args.golden)
    merged, _ = golden.join(gold, golden.load_run(args.run))
    sources = judge.load_sources(settings.artifact_dir)
    if not sources:
        print("warning: no pairs_meta.parquet - the `sources` column will be empty, and "
              "groundedness scores will not be comparable with the judge's")
    sheet = agreement.make_score_sheet(merged, args.n, seed=args.seed, sources_by_pair=sources)

    if args.out.exists():
        raise SystemExit(f"{args.out} already exists - refusing to overwrite your scores")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(args.out, index=False)
    print(f"wrote {len(sheet)} rows -> {args.out}")
    print("score each reply 1-5 on: " + ", ".join(agreement.DIMENSIONS))
    print("the `sources` column is what the judge saw - score groundedness against it")


if __name__ == "__main__":
    main()
