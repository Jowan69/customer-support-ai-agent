from support_agent.online.decide import decide
from support_agent.storage.schemas import IntentFlags, Signals


def _signals(**overrides):
    # Neutral on purpose: not fraud, not "other", and not one of
    # settings.account_specific_intents, so tests that aren't specifically
    # about intent-driven escalation don't trip that rule incidentally.
    base = dict(
        intent="no_action_needed",
        language="en",
        confidence=0.9,
        top1_sim=0.9,
        mean_sim=0.8,
        intent_flags=IntentFlags(),
        is_repeat_contact=False,
        is_other=False,
        safety_violations=[],
        llm_unavailable=False,
    )
    base.update(overrides)
    return Signals(**base)


def test_confident_and_grounded_auto_sends():
    result = decide(_signals())
    assert result["action"] == "auto"
    assert result["reason"] == "confident_and_grounded"


def test_is_other_escalates():
    result = decide(_signals(is_other=True))
    assert result["action"] == "escalate"
    assert result["reason"] == "is_other"


def test_llm_unavailable_escalates():
    result = decide(_signals(llm_unavailable=True))
    assert result["action"] == "escalate"
    assert result["reason"] == "llm_unavailable"


def test_repeat_contact_alone_does_not_escalate():
    """Restored (see decide.py's comment): chains.py now only sets
    is_repeat_contact true for a substantive follow-up, so decide() can read it
    again. Weight unchanged from before it was dropped - repeat_contact alone
    (1) stays below the escalation threshold (2); it contributes but never
    decides alone."""
    result = decide(_signals(is_repeat_contact=True))
    assert result["action"] == "auto"
    assert "repeat_contact" in result["reason"]


def test_repeat_contact_combined_with_another_concern_escalates():
    result = decide(_signals(is_repeat_contact=True, intent_flags=IntentFlags(refund=True)))
    assert result["action"] == "escalate"
    assert "repeat_contact" in result["reason"]
    assert "refund_intent" in result["reason"]


def test_repeat_contact_false_contributes_nothing():
    """decide() trusts the caller's is_repeat_contact as already-corrected -
    it does no narrowing of its own, so False must be a true no-op."""
    result = decide(_signals(is_repeat_contact=False))
    assert result["action"] == "auto"
    assert result["reason"] == "confident_and_grounded"


def test_low_confidence_escalates():
    result = decide(_signals(confidence=0.2))
    assert result["action"] == "escalate"
    assert "low_confidence" in result["reason"]


def test_weak_top_match_escalates():
    result = decide(_signals(top1_sim=0.1, mean_sim=0.8))
    assert result["action"] == "escalate"
    assert "weak_top_match" in result["reason"]


def test_weak_mean_support_alone_does_not_escalate():
    result = decide(_signals(mean_sim=0.1))
    assert result["action"] == "auto"
    assert "weak_retrieval_support" in result["reason"]


def test_always_escalate_intent_bypasses_the_weight_table():
    """fraud has no cluster behind it, so it can never be outvoted by other signals."""
    result = decide(_signals(intent="fraud_or_unauthorized"))

    assert result["action"] == "escalate"
    assert result["reason"] == "always_escalate:fraud_or_unauthorized"


def test_unsupported_language_escalates():
    result = decide(_signals(language="es"))

    assert result["action"] == "escalate"
    assert result["reason"] == "unsupported_language:es"


def test_unknown_language_escalates():
    """An undetectable language is not a reviewable one."""
    result = decide(_signals(language="unknown"))

    assert result["action"] == "escalate"
    assert result["reason"] == "unsupported_language:unknown"


def test_supported_language_still_auto_sends():
    result = decide(_signals(language="en"))

    assert result["action"] == "auto"


def test_unsafe_reply_escalates_even_when_everything_else_is_confident():
    """auto means nobody reads it, so a leaked link must never reach a customer."""
    result = decide(_signals(safety_violations=["url"]))

    assert result["action"] == "escalate"
    assert result["reason"] == "unsafe_reply:url"


def test_account_specific_intents_escalate_alone():
    """refund_status / account_manage / order_status / delivery_not_received
    need account-specific facts or a backend action - see settings.py's
    account_specific_intents. Confidence and retrieval strength must not be
    able to outvote that, so every confident/well-grounded default is kept."""
    from config.settings import settings

    for intent in settings.account_specific_intents:
        result = decide(_signals(intent=intent))
        assert result["action"] == "escalate", intent
        assert "account_specific_intent" in result["reason"], intent


def test_pricing_or_promotion_alone_does_not_escalate():
    """Deliberately left at weight 1, unlike account_specific_intents - only
    refund_status/account_manage/order_status/delivery_not_received were
    recalibrated."""
    result = decide(_signals(intent="pricing_or_promotion", intent_flags=IntentFlags(refund=True)))
    assert result["action"] == "auto"
    assert "refund_intent" in result["reason"]


def test_prime_membership_alone_does_not_escalate():
    result = decide(_signals(intent="prime_membership", intent_flags=IntentFlags(payment=True)))
    assert result["action"] == "auto"
    assert "payment_intent" in result["reason"]


def test_unsafe_reply_reason_names_every_violation():
    result = decide(_signals(safety_violations=["url", "order_number"]))

    assert result["reason"] == "unsafe_reply:url+order_number"


def test_safety_outranks_the_other_hard_gates():
    """The reason should name what we were about to send, not why it was odd."""
    result = decide(_signals(is_other=True, llm_unavailable=True, safety_violations=["url"]))

    assert result["reason"] == "unsafe_reply:url"
