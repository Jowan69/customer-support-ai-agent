.PHONY: eval demo judge eval-live record-fixtures score-sheet offline online test lint

# --- reproducing the published report: no API key, no artifacts, seconds ---

# Scores the frozen run in results/replies.csv against the 200 hand labels in
# data/labels/golden.csv and writes results/eval_report.md. Everything it reads is in
# the repo.
eval:
	python -m scripts.run_eval

# Runs the pipeline end to end on 5 recorded messages - no key, no index, no encoder.
# Writes results/demo_replies.csv, never the frozen run.
demo:
	python -m scripts.run_online --replay --limit 5

test:
	pytest

lint:
	ruff check .

# --- opt-ins that need a key, the artifacts, or both ---

# Re-runs the LLM judge over the frozen replies (costs API calls; cached, resumable).
judge:
	python -m scripts.run_eval --judge

# Scores retrieval by re-querying the FAISS index instead of scoring what the frozen
# run cited. Needs the full artifacts/<version>/ set.
eval-live:
	python -m scripts.run_eval --retrieval-live

# Refreshes fixtures/replay/ after a prompt, taxonomy or model change. Needs a key and
# the artifacts; the LLM cache answers most of it for free.
record-fixtures:
	python -m scripts.run_online --record --limit 5 --delay 0

# Blank sheet of 50 replies for hand-scoring, into data/labels/human_scores.csv.
score-sheet:
	python -m scripts.make_score_sheet --n 50

# --- how the frozen run was produced. Hours. Not a step in reproducing the report,
# --- and `offline` invalidates data/labels/golden.csv - read the README first.

offline:
	python -m scripts.run_offline

online:
	python -m scripts.run_online --input data/labels/golden.csv --out results/replies.csv