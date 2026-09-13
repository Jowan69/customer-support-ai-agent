"""Clusters embeddings and dumps sample messages per cluster for manual inspection.

Workflow: run `cluster()` here, inspect artifacts/<v>/clusters.parquet in
notebooks/01_taxonomy_inspect.ipynb, hand-label each cluster with an intent name,
then call `write_intents()` to produce the versioned config/intents.json that
online/classify.py is templated against.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def cluster(
    embeddings: np.ndarray,
    pair_ids: pd.Series,
    messages: pd.Series,
    n_clusters: int,
    samples_per_cluster: int = 8,
    random_state: int = 0,
) -> pd.DataFrame:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(embeddings)

    df = pd.DataFrame({"pair_id": pair_ids, "message": messages, "cluster_id": labels})
    rows = []
    for cluster_id, group in df.groupby("cluster_id"):
        sample = group["message"].head(samples_per_cluster).tolist()
        rows.append({"cluster_id": int(cluster_id), "size": len(group), "sample_msgs": sample})
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def write_intents(
    cluster_labels: dict[int, str],
    descriptions: dict[str, str],
    out_path: Path,
    version: str,
) -> None:
    """cluster_labels: {cluster_id: intent_name}, hand-assigned after inspecting clusters."""
    intents = sorted(set(cluster_labels.values()))
    payload = {
        "version": version,
        "intents": [
            {"name": name, "description": descriptions.get(name, "")} for name in intents
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=4))


def run(artifact_dir: Path, processed_dir: Path, n_clusters: int = 12) -> pd.DataFrame:
    embeddings = np.load(artifact_dir / "embeddings.npy")
    ids = pd.read_parquet(artifact_dir / "embedding_ids.parquet")
    pairs = pd.read_parquet(processed_dir / "pairs.parquet").set_index("pair_id")
    messages = ids["pair_id"].map(pairs["customer_msg"])

    clusters = cluster(embeddings, ids["pair_id"], messages, n_clusters)
    clusters.to_parquet(artifact_dir / "clusters.parquet", index=False)
    return clusters


if __name__ == "__main__":
    from config.settings import settings

    run(settings.artifact_dir, settings.processed_dir)
