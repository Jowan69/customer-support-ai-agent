"""Builds a FAISS index over customer_msg embeddings + writes pairs_meta for query-time lookup."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd


def build_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # embeddings are normalized -> inner product = cosine sim
    index.add(embeddings.astype(np.float32))
    return index


def run(artifact_dir: Path, processed_dir: Path) -> None:
    embeddings = np.load(artifact_dir / "embeddings.npy")
    ids = pd.read_parquet(artifact_dir / "embedding_ids.parquet")
    pairs = pd.read_parquet(processed_dir / "pairs.parquet")

    index = build_index(embeddings)
    faiss.write_index(index, str(artifact_dir / "pairs.faiss"))

    meta = ids.merge(pairs, on="pair_id", how="left")
    meta.to_parquet(artifact_dir / "pairs_meta.parquet", index=False)


if __name__ == "__main__":
    from config.settings import settings

    run(settings.artifact_dir, settings.processed_dir)
