"""Renders every metric into one markdown file: results/eval_report.md.

The report leads with what the numbers are computed over - how many rows, how many
were unscored, what the labels were - because every figure below it is meaningless
without that, and a table with no denominator is how an eval quietly overstates
itself.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from support_agent.eval.judge import DIMENSIONS


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _f(x: float | None, places: int = 3) -> str:
    return "-" if x is None else f"{x:.{places}f}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return out


def _coverage(meta: dict[str, Any]) -> list[str]:
    lines = [
        "## What this is computed over",
        "",
        f"- golden set: **{meta['n_golden']}** hand-labelled rows "
        f"(`{meta['golden_path']}`)",
        f"- frozen run: **{meta['n_run']}** rows (`{meta['run_path']}`)",
        f"- scored: **{meta['n_scored']}** rows present in both",
    ]
    if meta.get("unscored"):
        lines.append(
            f"- **{len(meta['unscored'])} labelled rows are not in the run** and are "
            "excluded from every table below"
        )
    lines += [
        f"- artifacts `{meta['artifact_version']}`, taxonomy `{meta['intents_version']}`",
        f"- generated {meta['generated_at']}",
        "",
        "Labels were assigned by hand, blind, before the agent was run over the set; "
        "the sampling and labelling method is in `data/labels/golden_sampling_note.md` "
        "and `data/labels/LABELLING.md`.",
        "",
    ]
    return lines


def _classification(score: dict, confusions: list[dict]) -> list[str]:
    lines = [
        "## Intent classification",
        "",
        f"Accuracy **{_pct(score['accuracy'])}** · macro-F1 **{_f(score['macro_f1'])}** · "
        f"weighted-F1 **{_f(score['weighted_f1'])}** over {score['n']} rows and "
        f"{score['n_intents_evaluated']} intents present in the labels.",
        "",
        "Macro-F1 averages only over intents that occur in the labelled set. The "
        "taxonomy defines more; scoring the model on intents nobody labelled would "
        "move the headline number for a reason unrelated to the model.",
        "",
    ]
    lines += _table(
        ["intent", "P", "R", "F1", "support", "predicted"],
        [
            [
                r["intent"],
                _f(r["precision"], 2),
                _f(r["recall"], 2),
                _f(r["f1"], 2),
                str(r["support"]),
                str(r["predicted"]),
            ]
            for r in score["per_intent"]
        ],
    )
    lines.append("")
    lines.append("Worst F1 first - that ordering is the point of the table.")
    lines.append("")

    if score["unpredicted_intents"]:
        lines += [
            f"**Never predicted:** {', '.join('`' + i + '`' for i in score['unpredicted_intents'])}"
            " - these intents appear in the labels and the classifier never once chose "
            "them.",
            "",
        ]
    if confusions:
        lines += ["### Most frequent confusions", ""]
        lines += _table(
            ["labelled", "predicted", "n"],
            [[c["actual"], c["predicted"], str(c["count"])] for c in confusions],
        )
        lines.append("")
    return lines


def _baselines(baselines: dict[str, Any], grok_score: dict) -> list[str]:
    lines = [
        "## Baselines",
        "",
        "Two reference points for the intent classifier above, scored with the same "
        "`classification.score()` function and the same metric definitions. Neither can "
        "train on anything the LLM produced - `pairs.parquet` carries no intent labels, "
        "only the golden set does - so both are fit and scored entirely within the golden "
        "set via leave-one-out cross-validation: every row predicted once, by a model that "
        "never saw it during training.",
        "",
        f"That makes the baselines' **n={baselines['majority']['n']}** a different "
        f"population from the frozen run's **n={grok_score['n']}** row above - the golden "
        "set has more labelled rows than the frozen run currently covers. The gap is real, "
        "not a formatting inconsistency, and the two are not directly comparable until the "
        "run covers the same rows.",
        "",
    ]
    lines += _table(
        ["model", "accuracy", "macro-F1", "weighted-F1", "n"],
        [
            [
                "Majority class (LOOCV)",
                _pct(baselines["majority"]["accuracy"]),
                _f(baselines["majority"]["macro_f1"]),
                _f(baselines["majority"]["weighted_f1"]),
                str(baselines["majority"]["n"]),
            ],
            [
                "TF-IDF + LogisticRegression (LOOCV)",
                _pct(baselines["tfidf_logreg"]["accuracy"]),
                _f(baselines["tfidf_logreg"]["macro_f1"]),
                _f(baselines["tfidf_logreg"]["weighted_f1"]),
                str(baselines["tfidf_logreg"]["n"]),
            ],
            [
                "Grok (frozen run)",
                _pct(grok_score["accuracy"]),
                _f(grok_score["macro_f1"]),
                _f(grok_score["weighted_f1"]),
                str(grok_score["n"]),
            ],
        ],
    )
    lines.append("")
    if baselines["singleton_intents"]:
        names = ", ".join("`" + i + "`" for i in baselines["singleton_intents"])
        verb = "has" if len(baselines["singleton_intents"]) == 1 else "have"
        lines += [
            f"{names} {verb} only one labelled example. In its LOOCV fold that label "
            "is entirely absent from training, so the supervised TF-IDF baseline cannot "
            "learn to predict it - a fact about having one example, not a shortcoming of "
            "leave-one-out cross-validation.",
            "",
        ]
    return lines


def _decisions(score: dict, reasons: pd.DataFrame, baseline: dict | None = None) -> list[str]:
    cm = score["confusion_matrix"]
    lines = [
        "## Auto vs escalate",
        "",
        f"`escalate` precision **{_f(score['escalate_precision'])}** · recall "
        f"**{_f(score['escalate_recall'])}** · F1 **{_f(score['escalate_f1'])}** · "
        f"accuracy **{_pct(score['accuracy'])}** over {score['n']} rows.",
        "",
        f"Labelled escalate rate **{_pct(score['gold_escalate_rate'])}**, predicted "
        f"**{_pct(score['predicted_escalate_rate'])}**. Compare both against the "
        "majority-class baseline before reading the accuracy as skill.",
        "",
    ]
    lines += _table(
        ["", "predicted auto", "predicted escalate"],
        [
            ["**labelled auto**", str(cm["auto"]["auto"]), str(cm["auto"]["escalate"])],
            ["**labelled escalate**", str(cm["escalate"]["auto"]), str(cm["escalate"]["escalate"])],
        ],
    )
    lines += [
        "",
        f"**{score['false_auto']} false auto** - sent without a human when one was "
        f"needed. This is the error that matters. **{score['false_escalate']} false "
        "escalate** - a human was asked for unnecessarily, which costs queue time only.",
        "",
    ]
    if baseline:
        lines += [
            "### Compared to the trivial policy",
            "",
            "`always_escalate_baseline()` escalates every row: zero automation, zero "
            "false-autos, accuracy equal to the labelled escalate rate. The agent's "
            "value is in the rows below it can safely move into the auto column, not "
            "in beating this on accuracy or escalate-recall alone - both are easy to "
            "win by escalating more.",
            "",
        ]
        lines += _table(
            [
                "policy", "accuracy", "escalate P", "escalate R", "escalate F1",
                "false auto", "false escalate", "n",
            ],
            [
                [
                    "Always escalate (trivial)",
                    _pct(baseline["accuracy"]),
                    _f(baseline["escalate_precision"]),
                    _f(baseline["escalate_recall"]),
                    _f(baseline["escalate_f1"]),
                    str(baseline["false_auto"]),
                    str(baseline["false_escalate"]),
                    str(baseline["n"]),
                ],
                [
                    "Agent (frozen run)",
                    _pct(score["accuracy"]),
                    _f(score["escalate_precision"]),
                    _f(score["escalate_recall"]),
                    _f(score["escalate_f1"]),
                    str(score["false_auto"]),
                    str(score["false_escalate"]),
                    str(score["n"]),
                ],
            ],
        )
        lines.append("")
    if not reasons.empty:
        lines += ["### Which rule fired, and whether it was right", ""]
        lines += _table(
            ["reason", "action", "n", "correct", "rate"],
            [
                [
                    f"`{r['reason']}`",
                    r["action"],
                    str(int(r["n"])),
                    str(int(r["n_correct"])),
                    _pct(r["correct_rate"]),
                ]
                for _, r in reasons.iterrows()
            ],
        )
        lines.append("")
    return lines


def _retrieval(score: dict | None) -> list[str]:
    if not score:
        return [
            "## Grounding",
            "",
            "_Not scored - no `pair_clusters.parquet` found in `frozen/` or "
            "`artifacts/`._",
            "",
        ]

    frozen = score.get("source") == "frozen_run"
    lines = ["## Grounding", ""]

    if frozen:
        lines += [
            f"Scored from the pairs the frozen run cited (`grounded_pair_ids`), over "
            f"**{score['n']}** of {score['n_rows']} rows. No index and no encoder are "
            "involved, so this table regenerates from a clone with nothing but "
            "`frozen/`.",
            "",
        ]
    else:
        lines += [
            f"Scored by re-querying the index for each golden message, over "
            f"**{score['n']}** of {score['n_rows']} rows. This measures the retriever, "
            "not the frozen run - the pairs scored here are whatever the index returns "
            "now, cited by the drafter or not.",
            "",
        ]

    lines += [
        f"Top-{score['k']} intent match **{_pct(score['top_k_match_rate'])}** · top-1 "
        f"**{_pct(score['top_1_match_rate'])}**.",
        "",
    ]

    if score.get("n_no_neighbour"):
        what = "cited no pair" if frozen else "retrieved nothing"
        lines += [
            f"**{score['n_no_neighbour']}** rows are excluded: they {what} once the "
            "query's own pair was removed. They are not counted as misses - a row with "
            "nothing to ground on is a coverage fact, not a wrong neighbour - so read "
            f"the rates above as over the {score['n']} rows that had one.",
            "",
        ]

    if score.get("mean_top1_score") is None:
        lines += [
            "Similarity is not reported on this path. The run's `top1_sim` column "
            "records the self-match at ~1.0 on every row, so it says nothing about the "
            "neighbours left after that pair is removed, and the frozen CSV holds no "
            "per-neighbour score to use instead. Run `--retrieval-live` against the "
            "full artifacts for a similarity split.",
            "",
        ]
    else:
        lines += [
            f"Mean top-1 similarity **{_f(score['mean_top1_score'])}** - "
            f"**{_f(score['mean_top1_score_on_hit'])}** when the neighbourhood matched, "
            f"**{_f(score['mean_top1_score_on_miss'])}** when it did not. If those two "
            "are close, the similarity score is not carrying the information "
            "`retrieval_score_threshold` assumes it does.",
            "",
        ]

    lines += [
        "Retrieved pairs are labelled by their cluster's modal hand-labelled intent, "
        "so this is a proxy: a mixed cluster can make a good retrieval look like a "
        "miss. A query's own pair is excluded, or the index would be scored on finding "
        "the row it was handed.",
        "",
    ]
    return lines


def _drafting(score: dict) -> list[str]:
    lines = [
        "## Reply quality",
        "",
        f"The drafter produced a reply for **{_pct(score['reply_rate'])}** of rows "
        f"({score['n_scored']} scored for similarity, {score['n_judged']} judged).",
        "",
    ]
    if score["n_scored"]:
        lines += [
            f"Cosine similarity to the brand's real reply: mean "
            f"**{_f(score['mean_cosine_sim'])}**, median "
            f"**{_f(score['median_cosine_sim'])}**. Reported as a drift check, not a "
            "quality measure - solving the same problem in different words scores low, "
            "and copying the phrasing while promising something the agent cannot "
            "deliver scores high.",
            "",
        ]
    else:
        lines += [
            "_Similarity not computed._ Either no row had both a drafted reply and a "
            "reference reply, or `results/reply_similarity.csv` is absent and the "
            "sentence-transformers encoder could not be loaded. Every other table on "
            "this page comes from the frozen files; this is the one number that needs "
            "the encoder, so it is reported as missing rather than as zero.",
            "",
        ]
    if score["judge_means"]:
        lines += ["### Judge dimensions (1-5)", ""]
        lines += _table(
            ["dimension", "mean", "scored <=2"],
            [
                [d, _f(score["judge_means"][d], 2), str(score["judge_low_count"].get(d, 0))]
                for d in DIMENSIONS
                if d in score["judge_means"]
            ],
        )
        lines += [
            "",
            "The `scored <=2` column is the one to read. A mean of 4.1 on `safety` with "
            "eleven 1s is not a passing grade - it is eleven replies that should never "
            "have been drafted.",
            "",
        ]
    return lines


_AGREEMENT_STATUS_DETAIL = {
    "missing_files": "`data/labels/human_scores.csv` (or the judge scores file) does not exist yet",
    "no_overlap": (
        "the human and judge score files both exist, but share no scored tweet_ids "
        "or no matching dimension columns"
    ),
}


def _agreement(rows: list[dict], status: str | None = None) -> list[str]:
    if not rows:
        # `status` distinguishes "the file isn't there yet" from "the file exists but
        # there is nothing to compare" - both would otherwise report as a missing file,
        # which is only true for the first. When the caller does not pass a status
        # (e.g. it only has the rows, not the paths that produced them), say so rather
        # than asserting a specific cause we do not actually know.
        detail = _AGREEMENT_STATUS_DETAIL.get(
            status,
            "`data/labels/human_scores.csv` may not exist yet, or it and the judge "
            "scores share no scored tweet_ids",
        )
        return [
            "## Judge agreement",
            "",
            f"_Not computed - {detail}._ Until it does, every judge number above is a "
            "model grading a model.",
            "",
        ]
    lines = [
        "## Judge agreement (Cohen's kappa, quadratic weights)",
        "",
        "Human scores on a sample of the frozen replies, scored blind against the same "
        "anchors. Sample size is given per dimension in the `n` column below - "
        "`agreement.score()` drops unscored rows independently for each dimension, so "
        "it is not the same count throughout.",
        "",
    ]
    lines += _table(
        ["dimension", "kappa", "exact", "within 1", "mean human", "mean judge", "n", "reading"],
        [
            [
                r["dimension"],
                _f(r["kappa"]),
                _pct(r["exact_agreement"]),
                _pct(r["within_one"]),
                _f(r["mean_human"], 2),
                _f(r["mean_judge"], 2),
                str(r["n"]),
                r["note"],
            ]
            for r in rows
        ],
    )
    lines += [
        "",
        "Where kappa is low but exact agreement is high, the dimension has almost no "
        "variance to explain and kappa collapses - read the agreement columns for those "
        "rows, and treat the dimension as uninformative rather than the judge as wrong.",
        "",
    ]
    return lines


def build_report(
    *,
    meta: dict[str, Any],
    classification: dict,
    confusions: list[dict],
    decisions: dict,
    reasons: pd.DataFrame,
    drafting: dict,
    retrieval: dict | None = None,
    agreement: list[dict] | None = None,
    agreement_status: str | None = None,
    baselines: dict[str, Any] | None = None,
    decision_baseline: dict | None = None,
) -> str:
    lines = ["# Evaluation report", ""]
    lines += _coverage(meta)
    lines += _classification(classification, confusions)
    if baselines:
        lines += _baselines(baselines, classification)
    lines += _decisions(decisions, reasons, decision_baseline)
    lines += _retrieval(retrieval)
    lines += _drafting(drafting)
    lines += _agreement(agreement or [], agreement_status)
    return "\n".join(lines).rstrip() + "\n"


def write_report(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


def action_counts(db_path: Path) -> pd.DataFrame:
    """The old summary: action counts by intent and reason, straight from decisions.db.

    Kept because it covers every decision ever recorded, not just the frozen run.
    """
    from support_agent.storage.db import fetch_all

    rows = [dict(r) for r in fetch_all(db_path)]
    if not rows:
        return pd.DataFrame(columns=["intent", "action", "reason", "count"])
    return (
        pd.DataFrame(rows)
        .groupby(["intent", "action", "reason"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
