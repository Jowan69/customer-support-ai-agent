# frozen/

Two small files the evaluation reads, committed because `artifacts/` is not.

| file | size | what it is |
| --- | --- | --- |
| `pair_clusters.parquet` | 1.5 MB | `pair_id -> cluster_id` for all 148,161 pairs |
| `clusters.parquet` | 18 KB | the 22 clusters with their sizes and sample messages |

Both are outputs of the `v2` offline build. They live here, and not in
`artifacts/v2/`, for one reason: the rest of that directory is ~500 MB of embeddings
and FAISS index, which cannot go in a git repo, and without these two files
`make eval` could not score grounding on a fresh clone. Everything else the eval
reads is already small enough to commit.

`eval/retrieval.py` looks here first and falls back to `artifacts/<version>/`, so a
working copy that has rebuilt its artifacts still scores against the partition the
golden labels were drawn from rather than a fresh one.

These are **not** rebuilt by `make offline`; that writes to `artifacts/<version>/`. If
you rebuild and mean to publish results from the new build, copy the new
`pair_clusters.parquet` and `clusters.parquet` here deliberately - and know that doing
so invalidates `data/labels/golden.csv`, whose intent labels belong to the taxonomy the
old clusters produced.
