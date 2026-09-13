"""Build the golden evaluation set: a stratified sample ready for hand-labelling.

Uniform random sampling is the obvious thing and the wrong thing here. Drawn evenly
from 148k pairs, `account_access` (2.3% of the corpus) lands about four times in 200
rows and `fraud_or_unauthorized` - which has no cluster at all - never appears. You
cannot report per-intent performance on a class with four examples, and the intent
that always escalates would go entirely unmeasured.

So rows are drawn in four passes, rarest first, and each row records the pass that
selected it in `sampled_by`. That column is the sampling note: it says exactly why
every example is in the set.

  1. fraud_seed          keyword-seeded; the only way to measure an intent with no cluster
  2. language_floor      every language with a real pool gets a minimum, so the
                         language gate can be evaluated rather than assumed
  3. cluster_floor       every embedding cluster gets a minimum - clusters are how the
                         taxonomy was derived, so this is the closest available proxy
                         for even intent coverage before any labels exist
  4. cluster_fill        remaining budget, proportional to cluster size, so the set
                         still reflects what real traffic looks like

Labels are deliberately left blank. Pre-filling them with the model's own predictions
is far quicker and quietly circular - you would be measuring the system against labels
it anchored. Label blind.

    python -m scripts.make_golden_sample --n 200
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from config.settings import settings

# Customers rarely use the word "fraud". They describe it: things arriving they did
# not buy, charges they do not recognise, someone else in the account. Kept short and
# high-precision - this pass seeds candidates for a human to judge, it does not label.
_FRAUD_PATTERNS = [
    r"n[o']?t\s+order", r"never\s+order", r"didn'?t\s+(?:order|buy|purchase|authori)",
    r"did\s+not\s+(?:order|buy|purchase|authori)", r"unauthoris|unauthoriz",
    r"hack(?:ed|ing)", r"someone\s+(?:else|has|is)\s+(?:us|access|order|buy)",
    r"don'?t\s+recogni[sz]e", r"do\s+not\s+recogni[sz]e", r"fraud",
    r"注文した覚え", r"身に覚えのない", r"不正アクセス",
    r"no\s+reconozco", r"no\s+he\s+(?:pedido|comprado)", r"nunca\s+ped[ií]",
    r"n[aã]o\s+(?:pedi|reconhe[cç]o)", r"nicht\s+bestellt", r"nie\s+bestellt",
]
_FRAUD_RE = re.compile("|".join(_FRAUD_PATTERNS), flags=re.IGNORECASE)


def _cluster_assignments(artifact_dir: Path) -> pd.DataFrame:
    """pair_id -> cluster_id, with the same k and seed taxonomy.py used.

    Delegated so the retrieval eval and this sampler cannot drift onto different
    partitions of the corpus; the shared version also caches the result, which turns
    the two-minute KMeans into a one-off.
    """
    from support_agent.offline.clusters import pair_cluster_assignments

    return pair_cluster_assignments(artifact_dir)


def _take(pool: pd.DataFrame, n: int, taken: set[str], label: str, seed: int) -> pd.DataFrame:
    """Up to `n` rows from `pool` that are not already selected, tagged with `label`."""
    fresh = pool[~pool["pair_id"].isin(taken)]
    if fresh.empty or n <= 0:
        return fresh.head(0).assign(sampled_by=label)
    picked = fresh.sample(n=min(n, len(fresh)), random_state=seed)
    taken.update(picked["pair_id"])
    return picked.assign(sampled_by=label)


def build_sample(
    pairs: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    n: int,
    fraud_n: int,
    language_floor: int,
    language_min_pool: int,
    cluster_floor: int,
    seed: int,
) -> pd.DataFrame:
    df = pairs.merge(clusters, on="pair_id", how="inner")
    taken: set[str] = set()
    parts: list[pd.DataFrame] = []

    # 1. fraud - rarest, and invisible to every other pass
    fraud_pool = df[df["customer_msg_clean"].astype(str).str.contains(_FRAUD_RE, na=False)]
    parts.append(_take(fraud_pool, fraud_n, taken, "fraud_seed", seed))

    # 2. language - so the gate that drives most escalations can be measured
    counts = df["customer_language"].value_counts()
    for language in counts[counts >= language_min_pool].index:
        pool = df[df["customer_language"] == language]
        parts.append(_take(pool, language_floor, taken, "language_floor", seed))

    # 3. cluster floor - even coverage of the structure the taxonomy came from
    for cluster_id in sorted(df["cluster_id"].unique()):
        pool = df[df["cluster_id"] == cluster_id]
        parts.append(_take(pool, cluster_floor, taken, "cluster_floor", seed))

    # 4. proportional fill - keeps the set shaped like real traffic
    remaining = n - sum(len(p) for p in parts)
    if remaining > 0:
        sizes = df["cluster_id"].value_counts(normalize=True)
        for cluster_id, share in sizes.items():
            pool = df[df["cluster_id"] == cluster_id]
            parts.append(_take(pool, int(round(remaining * share)), taken, "cluster_fill", seed))

    sample = pd.concat(parts, ignore_index=True)
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed)

    # shuffled so the labeller does not see all of one cluster in a row, which
    # would drag their judgement toward whatever they just decided
    return sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def to_golden(sample: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "tweet_id": sample["customer_tweet_id"],
            "message": sample["customer_msg"],
            "real_reply": sample["agent_reply"],
            "language": sample["customer_language"],
            "cluster_id": sample["cluster_id"],
            "sampled_by": sample["sampled_by"],
            "intent": "",           # <- you, by hand
            "expected_action": "",  # <- you, by hand: auto | escalate
            "note": "",
        }
    )
    return out


def summarise(golden: pd.DataFrame) -> str:
    by_pass = golden["sampled_by"].value_counts()
    by_lang = golden["language"].value_counts()
    lines = [
        "# Golden set - how it was sampled",
        "",
        f"{len(golden)} examples drawn from data/processed/pairs.parquet "
        f"({settings.artifact_version} artifacts).",
        "",
        "Uniform random sampling would leave the rare intents unmeasurable: "
        "`account_access` is 2.3% of the corpus and `fraud_or_unauthorized` has no "
        "cluster at all. Rows were drawn in four passes, rarest first.",
        "",
        "## Selection pass",
        "",
        "| pass | rows | why |",
        "| --- | --- | --- |",
        f"| fraud_seed | {by_pass.get('fraud_seed', 0)} | keyword-seeded; the only way to "
        "measure an intent no cluster produced |",
        f"| language_floor | {by_pass.get('language_floor', 0)} | the language gate causes "
        "most escalations, so it needs evaluating |",
        f"| cluster_floor | {by_pass.get('cluster_floor', 0)} | even coverage of the clusters "
        "the taxonomy was derived from |",
        f"| cluster_fill | {by_pass.get('cluster_fill', 0)} | proportional remainder, so the "
        "set still looks like real traffic |",
        "",
        "## Language coverage",
        "",
        "| language | rows |",
        "| --- | --- |",
    ]
    lines += [f"| {lang} | {count} |" for lang, count in by_lang.items()]
    lines += [
        "",
        "## Labelling",
        "",
        "`intent` and `expected_action` were filled in by hand, blind - the columns are "
        "written empty on purpose. Pre-filling them with the agent's own predictions is "
        "faster and circular: the set would be measuring the system against labels it "
        "anchored. `real_reply` is the brand's actual historical reply, carried over from "
        "the corpus rather than written.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="target size (brief asks 150-250)")
    parser.add_argument("--fraud-n", type=int, default=10)
    parser.add_argument("--language-floor", type=int, default=5)
    parser.add_argument("--language-min-pool", type=int, default=200)
    parser.add_argument("--cluster-floor", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    pairs = pd.read_parquet(settings.processed_dir / "pairs.parquet")
    print(f"corpus: {len(pairs):,} pairs")

    print("assigning clusters (KMeans over the saved embeddings - a minute or two)")
    clusters = _cluster_assignments(settings.artifact_dir)

    sample = build_sample(
        pairs,
        clusters,
        n=args.n,
        fraud_n=args.fraud_n,
        language_floor=args.language_floor,
        language_min_pool=args.language_min_pool,
        cluster_floor=args.cluster_floor,
        seed=args.seed,
    )
    golden = to_golden(sample)

    out_path = Path(args.out) if args.out else settings.labels_dir / "golden.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    golden.to_csv(out_path, index=False)

    note_path = out_path.with_name("golden_sampling_note.md")
    note_path.write_text(summarise(golden), encoding="utf-8")

    print(f"\nwrote {len(golden)} rows -> {out_path}")
    print(f"sampling note   -> {note_path}")
    print("\nby pass:")
    print(golden["sampled_by"].value_counts().to_string())
    print("\nby language:")
    print(golden["language"].value_counts().to_string())
    print("\nNow fill in `intent` and `expected_action`. Label blind - do not run the "
          "agent first.")


if __name__ == "__main__":
    main()
