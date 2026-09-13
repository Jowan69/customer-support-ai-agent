"""Batch-runs pipeline.handle over a CSV and freezes the result to results/replies.csv.

    python -m scripts.run_online --input data/labels/golden.csv --out results/replies.csv
    python -m scripts.run_online --replay --limit 5    # no API key, no index, seconds
    python -m scripts.run_online --record --limit 5    # re-record those fixtures

`--replay` runs the whole pipeline from `fixtures/replay/*.json` - recorded model
responses and recorded retrieval - so a clone with no key and no artifacts can still
watch classify -> retrieve -> draft -> signals -> decide happen on a real message. It
writes to `results/demo_replies.csv`, never to the frozen run the eval scores.

`--record` is the other half: it runs live and saves what came back. Run it when the
prompts, the taxonomy or the model change, and commit the refreshed fixtures.

Paced on purpose. Every provider caps tokens and requests per minute, and each message
costs two calls; firing them as fast as the API answers exhausts the quota in seconds
and the rest of the batch fails. --delay spaces them out instead. Groq's free tier is
6k tokens/min and a message costs roughly 2k, so the default 20s keeps a run inside it.
Raise the quota, lower the delay; --delay 0 disables it.

The frozen CSV is what the eval scores. Two consequences worth knowing:

  - Rows are written as they complete, and an existing --out is resumed rather than
    restarted. At 20s a row a 200-row run takes over an hour; losing it to a rate
    limit at row 190 is not an acceptable failure mode.
  - Every metric in the report comes from this one file, so classification, decisions
    and reply quality all describe the same run of a nondeterministic model. Re-running
    the agent between tables is how an eval ends up internally inconsistent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from config.settings import settings

from support_agent.online import replay
from support_agent.online.pipeline import handle

_FIELDS = (
    "tweet_id", "intent", "confidence", "language", "top1_sim", "is_repeat",
    "reply", "grounded_pair_ids", "action", "reason", "latency_ms",
)


def repeat_flags(chains_path: Path) -> dict[str, bool]:
    """root_tweet_id -> is_repeat_contact, from the offline chain walk.

    The golden CSV carries no repeat-contact column - it is a property of the thread,
    not the message - so it is joined back on here. Without it every row would be
    scored as a first contact and the `repeat_contact` concern would never fire.
    """
    if not chains_path.exists():
        return {}
    chains = pd.read_parquet(chains_path)
    return {
        str(r.root_tweet_id): bool(r.is_repeat_contact) for r in chains.itertuples()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=settings.labels_dir / "golden.csv")
    # Resolved after parsing, because the default depends on --replay: a demo run must
    # not land on top of the frozen run that every number in the report comes from.
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=20.0)
    parser.add_argument("--no-resume", action="store_true", help="start over, ignoring --out")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="run from recorded fixtures - no API key, no index, no delay",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="run live and save each row as a fixture (needs a key and the artifacts)",
    )
    parser.add_argument("--fixtures", type=Path, default=settings.root_dir / replay.DEFAULT_DIR)
    args = parser.parse_args()

    if args.replay and args.record:
        raise SystemExit("--replay and --record are opposites; pick one")

    results_dir = settings.root_dir / "results"
    if args.out is None:
        # --record runs live, so without this it would default onto results/replies.csv
        # and overwrite the frozen run with five rows.
        demo = args.replay or args.record
        args.out = results_dir / ("demo_replies.csv" if demo else "replies.csv")
    if args.replay:
        args.delay = 0.0  # nothing is being called; pacing would be theatre

    df = pd.read_csv(args.input, dtype={"tweet_id": str})
    if args.limit:
        df = df.head(args.limit)

    done: set[str] = set()
    rows: list[dict] = []
    if args.out.exists() and not args.no_resume:
        previous = pd.read_csv(args.out, dtype={"tweet_id": str})
        rows = previous.to_dict("records")
        done = set(previous["tweet_id"])
        print(f"resuming: {len(done)} rows already in {args.out}")

    fixtures = replay.load_fixtures(args.fixtures) if args.replay else {}

    flags = repeat_flags(settings.processed_dir / "chains.parquet")
    if not args.replay and "is_repeat_contact" not in df.columns and not flags:
        print("warning: no chains.parquet - every row treated as a first contact")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    todo = [r for _, r in df.iterrows() if str(r["tweet_id"]) not in done]

    for position, row in enumerate(todo):
        tweet_id = str(row["tweet_id"])
        is_repeat = bool(row.get("is_repeat_contact", flags.get(tweet_id, False)))

        client = None
        retriever = None
        captured: list[dict] = []

        if args.replay:
            fixture = fixtures.get(tweet_id)
            if fixture is None:
                print(f"skipped {tweet_id}\tno fixture in {args.fixtures}")
                continue
            # The fixture carries the flag: a clone has no chains.parquet to join it
            # from, and defaulting it to False would replay a different decision than
            # the one that was recorded.
            is_repeat = fixture.is_repeat
            client = replay.ReplayClient(fixture)
            retriever = replay.replay_retriever(fixture)
        elif args.record:
            from support_agent.llm.grok_client import get_client
            from support_agent.online.retrieve import retrieve as live_retrieve

            client = replay.RecordingClient(get_client())
            retriever = replay.recording_retriever(live_retrieve, captured)

        decision = handle(tweet_id, row["message"], is_repeat, client=client, retriever=retriever)

        if args.record:
            path = replay.write_fixture(
                args.fixtures,
                replay.new_fixture(
                    tweet_id,
                    str(row["message"]),
                    is_repeat,
                    captured,
                    client.responses,
                    artifact_version=settings.artifact_version,
                    model=settings.grok_model,
                ),
            )
            print(f"recorded\t{path}")
        rows.append(
            {
                "tweet_id": decision.tweet_id,
                "intent": decision.intent,
                "confidence": decision.classify_confidence,
                "language": decision.language,
                "top1_sim": decision.top1_sim,
                "is_repeat": int(decision.is_repeat),
                "reply": decision.reply,
                "grounded_pair_ids": json.dumps(decision.grounded_pair_ids),
                "action": decision.action,
                "reason": decision.reason,
                "latency_ms": decision.latency_ms,
            }
        )
        # Written every row, not at the end: an hour of API calls should not be lost
        # to a rate limit on the last one.
        pd.DataFrame(rows, columns=list(_FIELDS)).to_csv(args.out, index=False)

        print(f"{position + 1}/{len(todo)}\t{decision.tweet_id}\t{decision.intent}\t"
              f"{decision.action}\t{decision.reason}")

        if args.delay > 0 and position < len(todo) - 1:
            import time

            time.sleep(args.delay)

    print(f"\nfroze {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
