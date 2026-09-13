"""Encodes pairs.customer_msg_clean with one sentence-transformer -> embeddings.npy + id map.

Requires `clean.py` to have run first — the raw column is never vectorised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def embed(pairs: pd.DataFrame, model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(
        pairs["customer_msg_clean"].tolist(),
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def run(processed_dir: Path, artifact_dir: Path, model_name: str) -> None:
    pairs = pd.read_parquet(processed_dir / "pairs.parquet")
    if "customer_msg_clean" not in pairs.columns:
        raise RuntimeError("pairs.parquet has no customer_msg_clean column - run clean.py first")
    vectors = embed(pairs, model_name)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifact_dir / "embeddings.npy", vectors)
    pairs[["pair_id"]].to_parquet(artifact_dir / "embedding_ids.parquet", index=False)


if __name__ == "__main__":
    from config.settings import settings

    run(settings.processed_dir, settings.artifact_dir, settings.embedding_model)
