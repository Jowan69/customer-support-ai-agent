# fixtures/replay/

Recorded runs, one JSON per `tweet_id`, for `run_online --replay`.

Each fixture holds the pairs the index returned for that message and the JSON the
model replied with, captured during a real run. Replaying them runs the whole
pipeline - classify, retrieve, draft, signals, decide - with no API key, no FAISS
index, and no encoder, which is the only way someone who just cloned this repo can
watch it work at all.

To record or refresh them, on a machine that has the artifacts and a key:

    make record-fixtures        # python -m scripts.run_online --record --limit 5 --delay 0

The prompts are unchanged from the frozen run, so the LLM cache in `.cache/` answers
most of it without spending a request. Recording writes to
`results/demo_replies.csv`, never to the frozen `results/replies.csv`.

Re-record when the prompt templates, the taxonomy, or the model change. A fixture
stamps the `artifact_version` and `model` it was recorded against so a stale one is
identifiable; nothing checks it automatically.
