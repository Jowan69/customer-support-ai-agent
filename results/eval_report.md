# Evaluation report

## What this is computed over

- golden set: **200** hand-labelled rows (`golden.csv`)
- frozen run: **200** rows (`replies.csv`)
- scored: **200** rows present in both
- artifacts `v2`, taxonomy `v2`
- generated 2026-09-13 20:45 UTC

Labels were assigned by hand, blind, before the agent was run over the set; the sampling and labelling method is in `data/labels/golden_sampling_note.md` and `data/labels/LABELLING.md`.

## Intent classification

Accuracy **51.5%** · macro-F1 **0.509** · weighted-F1 **0.519** over 200 rows and 19 intents present in the labels.

Macro-F1 averages only over intents that occur in the labelled set. The taxonomy defines more; scoring the model on intents nobody labelled would move the headline number for a reason unrelated to the model.

| intent | P | R | F1 | support | predicted |
| --- | --- | --- | --- | --- | --- |
| account_manage | 0.00 | 0.00 | 0.00 | 1 | 3 |
| no_response_received | 0.33 | 0.20 | 0.25 | 5 | 3 |
| order_status | 0.19 | 0.60 | 0.29 | 5 | 16 |
| account_access | 0.20 | 1.00 | 0.33 | 1 | 5 |
| service_feedback | 0.45 | 0.28 | 0.34 | 18 | 11 |
| delivery_courier_issue | 0.38 | 0.43 | 0.40 | 7 | 8 |
| fraud_or_unauthorized | 0.38 | 0.43 | 0.40 | 7 | 8 |
| no_action_needed | 0.36 | 0.50 | 0.42 | 10 | 14 |
| prime_membership | 0.33 | 0.67 | 0.44 | 3 | 6 |
| other | 0.64 | 0.41 | 0.50 | 73 | 47 |
| delivery_not_received | 0.50 | 0.57 | 0.53 | 7 | 8 |
| refund_status | 0.67 | 0.57 | 0.62 | 7 | 6 |
| pricing_or_promotion | 0.60 | 0.75 | 0.67 | 4 | 5 |
| item_wrong_or_missing | 0.75 | 0.60 | 0.67 | 5 | 4 |
| contact_request | 0.56 | 0.83 | 0.67 | 6 | 9 |
| delivery_late | 0.63 | 0.71 | 0.67 | 24 | 27 |
| item_damaged | 0.80 | 0.67 | 0.73 | 6 | 5 |
| device_or_content_issue | 0.64 | 0.90 | 0.75 | 10 | 14 |
| delivery_wrong_location | 1.00 | 1.00 | 1.00 | 1 | 1 |

Worst F1 first - that ordering is the point of the table.

### Most frequent confusions

| labelled | predicted | n |
| --- | --- | --- |
| other | no_action_needed | 8 |
| service_feedback | other | 5 |
| delivery_late | order_status | 4 |
| other | delivery_late | 4 |
| other | order_status | 4 |
| other | service_feedback | 4 |
| other | prime_membership | 4 |
| other | device_or_content_issue | 4 |
| no_action_needed | other | 3 |
| other | delivery_not_received | 3 |

## Baselines

Two reference points for the intent classifier above, scored with the same `classification.score()` function and the same metric definitions. Neither can train on anything the LLM produced - `pairs.parquet` carries no intent labels, only the golden set does - so both are fit and scored entirely within the golden set via leave-one-out cross-validation: every row predicted once, by a model that never saw it during training.

That makes the baselines' **n=200** a different population from the frozen run's **n=200** row above - the golden set has more labelled rows than the frozen run currently covers. The gap is real, not a formatting inconsistency, and the two are not directly comparable until the run covers the same rows.

| model | accuracy | macro-F1 | weighted-F1 | n |
| --- | --- | --- | --- | --- |
| Majority class (LOOCV) | 36.5% | 0.028 | 0.195 | 200 |
| TF-IDF + LogisticRegression (LOOCV) | 41.0% | 0.052 | 0.260 | 200 |
| Grok (frozen run) | 51.5% | 0.509 | 0.519 | 200 |

`account_access`, `account_manage`, `delivery_wrong_location` have only one labelled example. In its LOOCV fold that label is entirely absent from training, so the supervised TF-IDF baseline cannot learn to predict it - a fact about having one example, not a shortcoming of leave-one-out cross-validation.

## Auto vs escalate

`escalate` precision **0.891** · recall **0.943** · F1 **0.916** · accuracy **85.0%** over 200 rows.

Labelled escalate rate **87.0%**, predicted **92.0%**. Compare both against the majority-class baseline before reading the accuracy as skill.

|  | predicted auto | predicted escalate |
| --- | --- | --- |
| **labelled auto** | 6 | 20 |
| **labelled escalate** | 10 | 164 |

**10 false auto** - sent without a human when one was needed. This is the error that matters. **20 false escalate** - a human was asked for unnecessarily, which costs queue time only.

### Compared to the trivial policy

`always_escalate_baseline()` escalates every row: zero automation, zero false-autos, accuracy equal to the labelled escalate rate. The agent's value is in the rows below it can safely move into the auto column, not in beating this on accuracy or escalate-recall alone - both are easy to win by escalating more.

| policy | accuracy | escalate P | escalate R | escalate F1 | false auto | false escalate | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Always escalate (trivial) | 87.0% | 0.870 | 1.000 | 0.930 | 0 | 26 | 200 |
| Agent (frozen run) | 85.0% | 0.891 | 0.943 | 0.916 | 10 | 20 | 200 |

### Which rule fired, and whether it was right

| reason | action | n | correct | rate |
| --- | --- | --- | --- | --- |
| `account_specific_intent` | escalate | 55 | 48 | 87.3% |
| `is_other` | escalate | 35 | 34 | 97.1% |
| `account_specific_intent+repeat_contact` | escalate | 21 | 21 | 100.0% |
| `unsupported_language:es` | escalate | 13 | 12 | 92.3% |
| `llm_unavailable` | escalate | 13 | 10 | 76.9% |
| `confident_and_grounded` | auto | 12 | 4 | 33.3% |
| `unsupported_language:ja` | escalate | 9 | 7 | 77.8% |
| `always_escalate:fraud_or_unauthorized` | escalate | 8 | 6 | 75.0% |
| `unsupported_language:fr` | escalate | 8 | 7 | 87.5% |
| `unsupported_language:unknown` | escalate | 5 | 5 | 100.0% |
| `unsupported_language:de` | escalate | 4 | 4 | 100.0% |
| `unsupported_language:pt` | escalate | 4 | 3 | 75.0% |
| `unsupported_language:pcm` | escalate | 3 | 2 | 66.7% |
| `below_escalation_threshold:repeat_contact` | auto | 3 | 1 | 33.3% |
| `payment_intent+repeat_contact` | escalate | 2 | 2 | 100.0% |
| `unsupported_language:it` | escalate | 2 | 2 | 100.0% |
| `below_escalation_threshold:refund_intent` | auto | 1 | 1 | 100.0% |
| `low_confidence+account_specific_intent` | escalate | 1 | 0 | 0.0% |
| `unsupported_language:hi` | escalate | 1 | 1 | 100.0% |

## Grounding

Scored from the pairs the frozen run cited (`grounded_pair_ids`), over **107** of 200 rows. No index and no encoder are involved, so this table regenerates from a clone with nothing but `frozen/`.

Top-4 intent match **45.8%** · top-1 **41.1%**.

**93** rows are excluded: they cited no pair once the query's own pair was removed. They are not counted as misses - a row with nothing to ground on is a coverage fact, not a wrong neighbour - so read the rates above as over the 107 rows that had one.

Similarity is not reported on this path. The run's `top1_sim` column records the self-match at ~1.0 on every row, so it says nothing about the neighbours left after that pair is removed, and the frozen CSV holds no per-neighbour score to use instead. Run `--retrieval-live` against the full artifacts for a similarity split.

Retrieved pairs are labelled by their cluster's modal hand-labelled intent, so this is a proxy: a mixed cluster can make a good retrieval look like a miss. A query's own pair is excluded, or the index would be scored on finding the row it was handed.

## Reply quality

The drafter produced a reply for **93.5%** of rows (187 scored for similarity, 199 judged).

Cosine similarity to the brand's real reply: mean **0.537**, median **0.537**. Reported as a drift check, not a quality measure - solving the same problem in different words scores low, and copying the phrasing while promising something the agent cannot deliver scores high.

### Judge dimensions (1-5)

| dimension | mean | scored <=2 |
| --- | --- | --- |
| groundedness | 4.38 | 6 |
| relevance | 3.47 | 20 |
| actionability | 3.53 | 12 |
| tone | 3.57 | 1 |
| safety | 4.98 | 0 |

The `scored <=2` column is the one to read. A mean of 4.1 on `safety` with eleven 1s is not a passing grade - it is eleven replies that should never have been drafted.

## Judge agreement (Cohen's kappa, quadratic weights)

Human scores on a sample of the frozen replies, scored blind against the same anchors. Sample size is given per dimension in the `n` column below - `agreement.score()` drops unscored rows independently for each dimension, so it is not the same count throughout.

| dimension | kappa | exact | within 1 | mean human | mean judge | n | reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| groundedness | -0.050 | 40.0% | 68.0% | 4.32 | 4.22 | 50 | poor agreement - the judge is not measuring what you are |
| relevance | 0.221 | 28.0% | 76.0% | 4.06 | 3.46 | 50 | fair |
| actionability | -0.052 | 16.0% | 72.0% | 4.20 | 3.56 | 50 | poor agreement - the judge is not measuring what you are |
| tone | 0.090 | 18.0% | 76.0% | 4.54 | 3.60 | 50 | poor agreement - the judge is not measuring what you are |
| safety | -0.020 | 84.0% | 94.0% | 4.76 | 4.98 | 50 | poor agreement - the judge is not measuring what you are |

Where kappa is low but exact agreement is high, the dimension has almost no variance to explain and kappa collapses - read the agreement columns for those rows, and treat the dimension as uninformative rather than the judge as wrong.
