"""Assembles the signals dict that decide() consumes. No decision logic lives here.

`is_other` means exactly one thing: the classifier didn't place the message in the
taxonomy. Low confidence is a separate signal that decide() weighs — the two are
deliberately not merged.
"""

from __future__ import annotations

from config.settings import settings

from support_agent.online.types import RetrievedPair
from support_agent.storage.schemas import IntentFlags, Signals


def build_signals(
    intent: str,
    language: str,
    confidence: float,
    retrieved: list[RetrievedPair],
    is_repeat_contact: bool,
    safety_violations: list[str] | None = None,
    llm_unavailable: bool = False,
) -> Signals:
    scores = [p["score"] for p in retrieved]
    top1_sim = scores[0] if scores else 0.0
    mean_sim = sum(scores) / len(scores) if scores else 0.0

    return Signals(
        intent=intent,
        language=language,
        confidence=confidence,
        top1_sim=top1_sim,
        mean_sim=mean_sim,
        intent_flags=IntentFlags(
            refund=intent in settings.refund_intents,
            payment=intent in settings.payment_intents,
        ),
        is_repeat_contact=is_repeat_contact,
        is_other=(intent == "other"),
        safety_violations=list(safety_violations or []),
        llm_unavailable=llm_unavailable,
    )
