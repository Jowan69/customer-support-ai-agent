"""decide(signals) -> {action, reason}. The one swappable piece of combination logic.

No imports from other `online/` modules — this can be replaced (rule table, learned
policy, whatever) without touching pipeline.py.

Five outright escalations: the drafted reply failed a safety check, the LLM was
unavailable, the message didn't land in the taxonomy (`is_other`), the intent is one
that always needs a person, or the customer wrote in a language nobody here can review. Those are decisions of kind, not degree -
no combination of other signals should be able to outvote them, which is why they sit
above the weight table rather than in it.

Everything else is a weighted concern, and escalation needs the
weights to reach a threshold. `is_repeat_contact` carries a weight below that threshold
on purpose: it contributes to the call but never makes it alone. How the remaining
signals combine is the piece the design deliberately left open — the weights below are
a placeholder for it, which is why they live here and in settings, not in pipeline.py.
"""

from __future__ import annotations

from typing import TypedDict

from config.settings import settings

from support_agent.storage.schemas import Action, Signals


class DecideResult(TypedDict):
    action: Action
    reason: str


def _concerns(signals: Signals) -> list[tuple[str, int]]:
    """(name, weight) for every concern the signals raise."""
    found: list[tuple[str, int]] = []

    if signals.confidence < settings.classify_confidence_threshold:
        found.append(("low_confidence", 2))
    if signals.top1_sim < settings.retrieval_score_threshold:
        found.append(("weak_top_match", 2))
    if signals.mean_sim < settings.mean_retrieval_score_threshold:
        found.append(("weak_retrieval_support", 1))
    # Weighted at the escalation threshold itself: a message in one of these
    # intents needs a human regardless of how confident the classifier was or
    # how close the retrieved neighbours were - those signals speak to whether
    # the model recognised the message, not to whether it can act on it.
    if signals.intent in settings.account_specific_intents:
        found.append(("account_specific_intent", 2))
    if signals.intent_flags.refund:
        found.append(("refund_intent", 1))
    if signals.intent_flags.payment:
        found.append(("payment_intent", 1))
    # Restored 2026-09-13 (previously dropped: see chains.py's docstring for
    # why). chains.py now only sets is_repeat_contact true for a substantive
    # follow-up, so decide() can trust it again without re-deriving that
    # narrowing itself. Weight unchanged from before it was dropped.
    if signals.is_repeat_contact:
        found.append(("repeat_contact", 1))

    return found


def decide(signals: Signals) -> DecideResult:
    # First, because it is the only rule about what we are on the verge of *sending*.
    # A reply carrying someone else's tracking link must not auto-send even when every
    # other signal is confident - and the reason has to name what leaked.
    if signals.safety_violations:
        return DecideResult(
            action="escalate", reason="unsafe_reply:" + "+".join(signals.safety_violations)
        )

    if signals.llm_unavailable:
        return DecideResult(action="escalate", reason="llm_unavailable")

    if signals.is_other:
        return DecideResult(action="escalate", reason="is_other")

    if signals.intent in settings.escalate_always_intents:
        return DecideResult(action="escalate", reason=f"always_escalate:{signals.intent}")

    # A reply nobody on the team can read is a reply nobody can check.
    if signals.language not in settings.auto_send_languages:
        return DecideResult(action="escalate", reason=f"unsupported_language:{signals.language}")

    concerns = _concerns(signals)
    weight = sum(w for _, w in concerns)

    if weight >= settings.escalation_weight_threshold:
        return DecideResult(action="escalate", reason="+".join(name for name, _ in concerns))

    if concerns:
        return DecideResult(
            action="auto",
            reason="below_escalation_threshold:" + "+".join(name for name, _ in concerns),
        )

    return DecideResult(action="auto", reason="confident_and_grounded")
