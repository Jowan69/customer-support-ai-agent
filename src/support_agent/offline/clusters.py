"""Canonical pair -> cluster assignment, shared by the sampler and the retrieval eval.

The offline build persists only the cluster *summaries* (clusters.parquet), not which
cluster each pair landed in. Two places need the per-pair assignment: the golden
sampler, to guarantee even coverage, and the retrieval eval, to say what a retrieved
neighbour is about. Recomputing it in both would be the same code twice with the same
seed - until one of them changed and the eval silently started scoring against a
different partition than the sample was drawn from. So it lives here once.

The result is cached to `pair_clusters.parquet` in the artifact directory: KMeans over
230MB of embeddings takes a couple of minutes, and nothing about it changes between
runs.

The v2 cache is also committed to `frozen/pair_clusters.parquet`, which is what
`eval/retrieval.py` reads. `artifacts/` is too large for the repo, and without that one
1.5MB file a fresh clone could not score grounding at all. See `frozen/README.md`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_CACHE_NAME = "pair_clusters.parquet"

# Fixed, and must stay fixed: taxonomy.py clustered with these, so changing them here
# would repartition the corpus underneath a taxonomy that was hand-labelled from the
# old partition.
_RANDOM_STATE = 0


def pair_cluster_assignments(artifact_dir: Path, *, use_cache: bool = True) -> pd.DataFrame:
    """pair_id -> cluster_id, using the same k and seed taxonomy.py used."""
    cache_path = artifact_dir / _CACHE_NAME
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    from sklearn.cluster import KMeans

    embeddings = np.load(artifact_dir / "embeddings.npy")
    ids = pd.read_parquet(artifact_dir / "embedding_ids.parquet")
    k = len(pd.read_parquet(artifact_dir / "clusters.parquet"))

    labels = KMeans(n_clusters=k, random_state=_RANDOM_STATE, n_init="auto").fit_predict(
        embeddings
    )
    out = pd.DataFrame({"pair_id": ids["pair_id"].astype(str), "cluster_id": labels})

    if use_cache:
        out.to_parquet(cache_path, index=False)
    return out
