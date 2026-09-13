"""draft(msg, pairs, intent) -> {reply, grounded_pair_ids}."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from support_agent.llm.grok_client import GrokClient, GrokUnavailableError, get_client
from support_agent.online.types import RetrievedPair
from support_agent.text import UNKNOWN_LANGUAGE

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "draft.txt"

_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "grounded_pair_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reply", "grounded_pair_ids"],
}


class DraftResult(TypedDict):
    reply: str | None
    grounded_pair_ids: list[str]


def _language_instruction(language: str) -> str:
    """The retrieved examples are mostly English, so without this the model drifts
    into replying in the examples' language rather than the customer's."""
    if language == UNKNOWN_LANGUAGE:
        return (
            "Write the reply in the same language the customer used. The examples "
            "above may be in a different language - copy their approach, not their "
            "language."
        )
    return (
        f"The customer wrote in '{language}' (ISO 639-1). Write the reply in that "
        "language, even when the examples above are in another one: use them for "
        "what to say and how the problem was resolved, never for which language to "
        "say it in."
    )


def _build_prompt(
    message: str, pairs: list[RetrievedPair], intent: str, language: str
) -> str:
    block = "\n".join(
        f"[{p['pair_id']}] customer: {p['customer_msg']!r} -> agent: {p['agent_reply']!r}"
        for p in pairs
    ) or "(none found)"
    return _PROMPT_PATH.read_text().format(
        message=message,
        intent=intent,
        retrieved_pairs_block=block,
        language_instruction=_language_instruction(language),
    )


def draft(
    message: str,
    pairs: list[RetrievedPair],
    intent: str,
    language: str = UNKNOWN_LANGUAGE,
    client: GrokClient | None = None,
) -> DraftResult:
    prompt = _build_prompt(message, pairs, intent, language)
    client = client or get_client()

    try:
        result = client.complete_json(prompt, json_schema=_SCHEMA, schema_name="draft")
    except GrokUnavailableError:
        return DraftResult(reply=None, grounded_pair_ids=[])

    valid_ids = {p["pair_id"] for p in pairs}
    grounded = [pid for pid in result.get("grounded_pair_ids", []) if pid in valid_ids]
    return DraftResult(reply=result.get("reply"), grounded_pair_ids=grounded)
