# Labelling the golden set

The "how you labelled" half of the note the brief asks for. Written before any row was
labelled, so it is a method rather than a description of what happened.

## 0. Before you start

```bash
python -m scripts.make_golden_sample --n 200
```

Check the printed breakdown before labelling anything:

- `fraud_seed` should be ~10. If it is 0 the keyword pass matched nothing and
  `fraud_or_unauthorized` will be unmeasurable — stop and widen the patterns.
- `language_floor` should cover more than `en`. If only `en` appears, no language has a
  pool of 200, and the language gate (which drives most escalations) cannot be evaluated.
- `cluster_floor` should be ~132 (k=22 x 6). Nothing should be missing a cluster.

**Hide `real_reply` while you label.** It is in the CSV as the reference for Phase 6/7
drafting scores, not as an answer key. Reading Amazon's actual reply before you decide
`expected_action` anchors you to what Amazon did, which is not the same question. Label
from a view showing only `tweet_id`, `message`, `language`.

**Do not run the agent first.** Blind, or the numbers mean nothing.

## 1. Label from the message alone

You are labelling what the *incoming message* is and what *this system* should do with
it. Not what a good agent would eventually discover, not what the thread turned out to
be about. One tweet, one judgement.

## 2. Decision order — first match wins

Ambiguity is resolved by order, not by feel. Walk down the list and stop at the first
rule that fits.

1. Reply fragment with no standalone meaning ("yes please", "DM sent", a bare mention) -> `other`
2. Nothing is being asked — thanks, or confirmation it is resolved -> `no_action_needed`
3. Customer says they did not authorise it — didn't order, don't recognise the charge,
   account hacked, someone else in the account -> `fraud_or_unauthorized`
   *(beats every delivery and refund label below — this is the highest-cost miss)*
4. Asking for a channel, not an answer — phone number, live chat, a human -> `contact_request`
5. Did what they were told and heard nothing, or waiting on a promised email/callback
   -> `no_response_received`
6. Physical package, in this order:
   a. tracking says delivered, customer doesn't have it -> `delivery_not_received`
   b. complaint about what the *driver* did — left in the rain, gave it to a neighbour,
      threw it -> `delivery_courier_issue`
   c. went to the wrong address/country/depot, or wants it rerouted -> `delivery_wrong_location`
   d. still in transit, past or about to miss its date -> `delivery_late`
7. Contents: arrived broken/counterfeit -> `item_damaged`; wrong item or part of the
   order missing -> `item_wrong_or_missing`
8. Money: owed money that hasn't arrived or is the wrong amount -> `refund_status`;
   disputing the price/discount/promo itself -> `pricing_or_promotion`
9. Account: cannot log in -> `account_access`; can log in, wants to change something
   -> `account_manage`
10. `prime_membership`, `device_or_content_issue`, `order_status`, `service_feedback`
11. Nothing above fits -> `other`

## 3. The tie-breaks that will actually bite

| Pair | The question that settles it |
| --- | --- |
| `delivery_late` vs `delivery_not_received` | What does tracking say? In transit -> late. Delivered -> not_received. |
| `delivery_not_received` vs `delivery_courier_issue` | Is the complaint about the driver's *action*? Yes -> courier_issue. Just "says delivered, isn't here" -> not_received. |
| `fraud_or_unauthorized` vs `refund_status` | Did the customer make the purchase? Yes -> refund_status. No -> fraud. |
| `order_status` vs `delivery_late` | Has a delivery promise been missed? No, or it's a general order question -> order_status. |
| `refund_status` vs `pricing_or_promotion` | Is money owed back, or is the amount charged itself disputed? |
| `account_access` vs `account_manage` | Can they get in right now? |
| `service_feedback` vs `no_action_needed` | Any unresolved ask attached? No ask + evaluative -> service_feedback. No ask + gratitude -> no_action_needed. |

**On `other`:** it means *unclassifiable*, not *hard*. It maps to `is_other` and forces
escalation, so leaning on it when you're unsure quietly inflates the escalation rate and
makes the system look safer than it is. If you can name the intent at 60% confidence,
name it and put your doubt in `note`.

## 4. `expected_action` — auto or escalate

Two readings are possible and they give different labels: *what a good human agent would
do*, or *what this system should do given what it can and cannot see*. **Use the second**
— that is what `decide()` is being measured against.

Mark `escalate` if any of these hold:

- the message is not in English (the reviewer can't read it, so nobody can check the reply)
- `fraud_or_unauthorized`, or `other`
- a correct reply needs account-specific facts the agent cannot see — this order's real
  status, whether a refund actually went out, why a specific charge appeared
- a correct reply needs an *action* only a human or backend can take — issue the refund,
  cancel the order, reroute the parcel, unlock the account
- the customer is already in a failed support loop (`no_response_received`, or visible
  repeat contact); a canned reply makes it worse

Mark `auto` only if the right reply is procedural, identical for anyone in that
situation, needs no lookup, and is in English — how to start a return, what to do when
tracking stalls, where the Prime cancel setting lives, acknowledging thanks.

Expect this to come out heavily skewed toward `escalate` — plausibly 70%+. That is a
finding, not a failure: it is the real ceiling on automation for a public Twitter channel
with no account access, and it belongs in the README. It is also exactly why the
majority-class baseline exists — without it, "80% accurate" could just be a model that
always says escalate.

## 5. Multi-issue tweets

Label the intent the customer wants *resolved*, not the first one mentioned. If two are
genuinely co-equal, take the one higher in the decision order and record the other in
`note`.

## 6. Drift control

Labelling 200 rows over hours means row 1 and row 180 get judged by different standards.
Three cheap defences:

- Do 20 rows, stop, re-read sections 2-4, then continue. Drift is worst at the start.
- Two sittings maximum. Fatigue shows up as everything becoming `escalate`.
- **Re-label the first 20 at the end**, without looking at your first pass. Where they
  disagree is your own drift — report that count as a self-consistency figure. It costs
  20 minutes and it is the honest upper bound on how good the κ in Phase 7 can be.

Fill `note` for any row you hesitated on for more than ~10 seconds. Those rows are the
error analysis in Phase 8, and you will not remember them later.

Never leave `intent` blank. `other` is a decision; blank is a skipped row.
