
"""ingest -> pairs -> chains -> clean -> filter -> embed -> taxonomy -> index.

The taxonomy step clusters the embeddings and writes artifacts/<v>/clusters.parquet.
Naming those clusters is manual and stays manual: open
notebooks/01_taxonomy_inspect.ipynb, read each cluster's sample messages, and call
taxonomy.write_intents() to produce config/intents.json. Until that is done,
config/intents.json holds whatever was there before — the online classifier will be
sorting real messages into whatever labels that file contains.
"""

from __future__ import annotations

import argparse
import json

from config.settings import settings

from support_agent.offline import (
    build_index,
    chains,
    clean,
    corpus_filter,
    embed,
    ingest,
    pairs,
    taxonomy,
)


def _intents_are_placeholder() -> bool:
    """True if config/intents.json wasn't produced from this run's clusters."""
    if not settings.intents_path.exists():
        return True
    clusters_path = settings.artifact_dir / "clusters.parquet"
    if not clusters_path.exists():
        return True
    return settings.intents_path.stat().st_mtime < clusters_path.stat().st_mtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=12,
        help="number of KMeans clusters to dump for manual inspection (default: 12)",
    )
    args = parser.parse_args()

    print("[1/8] ingest")
    ingest.ingest(settings.raw_dir / "twcs.csv", settings.processed_dir / "messages.parquet")

    print("[2/8] pairs")
    pairs.run(settings.processed_dir)

    print("[3/8] chains")
    chains.run(
        settings.processed_dir,
        min_chars=settings.min_message_chars,
        min_words=settings.min_message_words,
    )

    print("[4/8] clean")
    clean.run(settings.processed_dir)

    print("[5/8] filter")
    counts = corpus_filter.run(
        settings.processed_dir,
        min_chars=settings.min_message_chars,
        min_words=settings.min_message_words,
    )
    print(
        f"      {counts['total']:,} pairs -> {counts['kept']:,} kept "
        f"(empty {counts['removed_empty']:,}, "
        f"short {counts['removed_too_short']:,}, "
        f"duplicate {counts['removed_duplicate']:,})"
    )

    print("[6/8] embed")
    embed.run(settings.processed_dir, settings.artifact_dir, settings.embedding_model)

    print(f"[7/8] taxonomy (clustering into {args.n_clusters} groups)")
    clusters = taxonomy.run(settings.artifact_dir, settings.processed_dir, args.n_clusters)
    print(f"      -> {settings.artifact_dir / 'clusters.parquet'} ({len(clusters)} clusters)")

    print("[8/8] build_index")
    build_index.run(settings.artifact_dir, settings.processed_dir)

    print(f"done -> {settings.artifact_dir}")

    if _intents_are_placeholder():
        current = "missing"
        if settings.intents_path.exists():
            names = [i["name"] for i in json.loads(settings.intents_path.read_text())["intents"]]
            current = ", ".join(names)
        print()
        print("ACTION REQUIRED: config/intents.json has not been rebuilt from these clusters.")
        print(f"  it currently holds: {current}")
        print("  open notebooks/01_taxonomy_inspect.ipynb, name each cluster, and run")
        print("  taxonomy.write_intents() before running the online pipeline.")


if __name__ == "__main__":
    main()
