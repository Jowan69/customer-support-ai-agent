"""classify(msg) -> {intent, confidence, model, latency_ms}. Reads config/intents.json only."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TypedDict

from config.settings import settings

from support_agent.llm.grok_client import GrokClient, GrokUnavailableError, get_client

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "classify.txt"

def _schema(valid_names: list[str]) -> dict:
    """Built per call from intents.json rather than fixed at import.

    With the enum the model cannot emit a label outside the taxonomy at all. The
    old unconstrained string let it invent one, which was then quietly remapped to
    "other" - hiding exactly the drift the intents_version stamp exists to catch.
    """
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": valid_names},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["intent", "confidence"],
    }


class ClassifyResult(TypedDict):
    intent: str
    confidence: float
    model: str
    latency_ms: float
    llm_unavailable: bool


def _load_intents() -> list[dict]:
    return json.loads(settings.intents_path.read_text())["intents"]


def _build_prompt(message: str, intents: list[dict]) -> str:
    intents_block = "\n".join(f"- {i['name']}: {i['description']}" for i in intents)
    return _PROMPT_PATH.read_text().format(intents_block=intents_block, message=message)


def classify(message: str, client: GrokClient | None = None) -> ClassifyResult:
    intents = _load_intents()
    valid_names = [i["name"] for i in intents]
    prompt = _build_prompt(message, intents)

    client = client or get_client()
    start = time.perf_counter()
    llm_unavailable = False
    try:
        result = client.complete_json(
            prompt, json_schema=_schema(valid_names), schema_name="classify"
        )
        # Kept as a backstop: enum enforcement is the provider's promise, not ours.
        intent = result["intent"] if result["intent"] in set(valid_names) else "other"
        confidence = float(result["confidence"])
    except (GrokUnavailableError, KeyError, TypeError, ValueError):
        # "other" is only a safe placeholder here, NOT a classification. Without the
        # flag a dead API is indistinguishable in the decisions table from a message
        # the classifier genuinely could not place, which would silently corrupt eval.
        # KeyError/TypeError/ValueError cover a response whose shape is wrong - a
        # replayed cache entry from an older schema, say. That is unusable rather
        # than unavailable, but the right answer is the same: escalate, never crash.
        intent, confidence = "other", 0.0
        llm_unavailable = True
    latency_ms = (time.perf_counter() - start) * 1000

    return ClassifyResult(
        intent=intent,
        confidence=confidence,
        model=settings.grok_model,
        latency_ms=latency_ms,
        llm_unavailable=llm_unavailable,
    )
