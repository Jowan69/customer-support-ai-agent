# Golden set - how it was sampled

200 examples drawn from data/processed/pairs.parquet (v2 artifacts).

Uniform random sampling would leave the rare intents unmeasurable: `account_access` is 2.3% of the corpus and `fraud_or_unauthorized` has no cluster at all. Rows were drawn in four passes, rarest first.

## Selection pass

| pass | rows | why |
| --- | --- | --- |
| fraud_seed | 10 | keyword-seeded; the only way to measure an intent no cluster produced |
| language_floor | 44 | the language gate causes most escalations, so it needs evaluating |
| cluster_floor | 130 | even coverage of the clusters the taxonomy was derived from |
| cluster_fill | 16 | proportional remainder, so the set still looks like real traffic |

## Language coverage

| language | rows |
| --- | --- |
| en | 124 |
| es | 19 |
| ja | 12 |
| fr | 12 |
| pt | 9 |
| unknown | 8 |
| de | 6 |
| pcm | 5 |
| it | 4 |
| hi | 1 |

## Labelling

`intent` and `expected_action` were filled in by hand, blind - the columns are written empty on purpose. Pre-filling them with the agent's own predictions is faster and circular: the set would be measuring the system against labels it anchored. `real_reply` is the brand's actual historical reply, carried over from the corpus rather than written.