"""retrieve(msg) -> [pairs with score]. Loads the FAISS index + encoder once per process."""

from __future__ import annotations

from functools import lru_cache

import faiss
import numpy as np
import pandas as pd
from config.settings import settings

from support_agent.online.types import RetrievedPair
from support_agent.text import clean_text

__all__ = ["RetrievedPair", "retrieve"]


@lru_cache(maxsize=1)
def _load_index() -> faiss.Index:
    return faiss.read_index(str(settings.artifact_dir / "pairs.faiss"))


@lru_cache(maxsize=1)
def _load_meta() -> pd.DataFrame:
    return pd.read_parquet(settings.artifact_dir / "pairs_meta.parquet")


@lru_cache(maxsize=1)
def _load_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def retrieve(message: str, k: int | None = None) -> list[RetrievedPair]:
    k = k or settings.top_k
    index = _load_index()
    meta = _load_meta()
    encoder = _load_encoder()

    # the index was built over cleaned text; the query must be cleaned identically
    vector = encoder.encode([clean_text(message)], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(vector, k)

    results: list[RetrievedPair] = []
    for score, idx in zip(scores[0], indices[0], strict=True):
        if idx < 0:
            continue
        row = meta.iloc[idx]
        results.append(
            RetrievedPair(
                pair_id=row["pair_id"],
                customer_msg=row["customer_msg"],
                agent_reply=row["agent_reply"],
                score=float(score),
            )
        )
    return results
