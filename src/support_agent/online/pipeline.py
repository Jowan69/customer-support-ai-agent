"""handle(msg) -> runs classify -> retrieve -> draft -> signals -> decide, writes a Decision row.

`client` and `retriever` are the pipeline's only two doors to the outside world, and
both are injectable so the whole thing can run from recorded fixtures with no API key
and no FAISS index - see `online/replay.py` and `run_online --replay`. Nothing else in
here changes behaviour between a live run and a replayed one.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from config.settings import settings

from support_agent.llm.grok_client import get_client
from support_agent.online.classify import classify
from support_agent.online.decide import decide
from support_agent.online.draft import DraftResult, draft
from support_agent.online.safety import check_reply
from support_agent.online.signals import build_signals
from support_agent.storage.db import insert_decision
from support_agent.storage.schemas import Decision
from support_agent.text import clean_text, detect_language


def handle(
    tweet_id: str,
    message: str,
    is_repeat_contact: bool = False,
    *,
    client: object | None = None,
    retriever: Callable[..., list] | None = None,
) -> Decision:
    start = time.perf_counter()
    # `or get_client()`, not an unconditional call: get_client() raises when no API key
    # is set, and a replayed run is allowed to have no key.
    client = client or get_client()
    # Imported here, not at module scope: `online.retrieve` pulls in faiss and the
    # index, and a replayed run supplies its own retriever and should not need either.
    if retriever is None:
        from support_agent.online.retrieve import retrieve as retrieve_fn
    else:
        retrieve_fn = retriever

    # Detected here, not passed in: a live message arrives as raw text with no
    # metadata, and this is the same detector the offline corpus was labelled with.
    language = detect_language(clean_text(message))

    classification = classify(message, client=client)
    retrieved = retrieve_fn(message)

    # If the first call could not reach the model, the second one almost certainly
    # cannot either - and decide() already returns escalate on llm_unavailable, so
    # the answer is fixed whatever comes back. Skipping it saves three more attempts,
    # each of which may now wait out a Retry-After, to reach a foregone conclusion.
    draft_result: DraftResult = (
        DraftResult(reply=None, grounded_pair_ids=[])
        if classification["llm_unavailable"]
        else draft(
            message, retrieved, classification["intent"], language=language, client=client
        )
    )

    # Checked before the decision, never after: the point is to stop an unsafe reply
    # auto-sending, and by the time decide() has answered it is too late to matter.
    safety_violations = check_reply(draft_result["reply"])

    signals = build_signals(
        intent=classification["intent"],
        language=language,
        confidence=classification["confidence"],
        retrieved=retrieved,
        is_repeat_contact=is_repeat_contact,
        safety_violations=safety_violations,
        llm_unavailable=classification["llm_unavailable"] or draft_result["reply"] is None,
    )
    decision_result = decide(signals)

    latency_ms = (time.perf_counter() - start) * 1000

    decision = Decision(
        decision_id=str(uuid.uuid4()),
        tweet_id=tweet_id,
        message=message,
        intent=classification["intent"],
        language=language,
        classify_confidence=classification["confidence"],
        top1_sim=signals.top1_sim,
        is_repeat=signals.is_repeat_contact,
        reply=draft_result["reply"],
        grounded_pair_ids=draft_result["grounded_pair_ids"],
        action=decision_result["action"],
        reason=decision_result["reason"],
        artifact_version=settings.artifact_version,
        intents_version=settings.intents_version,
        latency_ms=latency_ms,
        created_at=datetime.now(UTC),
    )

    insert_decision(settings.db_path, decision)
    return decision
