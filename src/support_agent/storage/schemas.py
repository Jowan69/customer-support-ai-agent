"""Pydantic models shared across offline/online/eval. No I/O here."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class Message(BaseModel):
    tweet_id: str
    author_id: str
    text: str
    text_clean: str | None = None
    language: str | None = None
    created_at: datetime
    in_response_to_tweet_id: str | None = None
    response_tweet_id: str | None = None
    inbound: bool


class Pair(BaseModel):
    """A single customer message -> first AmazonHelp reply."""

    pair_id: str
    customer_tweet_id: str
    customer_msg: str
    customer_language: str | None = None
    agent_tweet_id: str
    agent_reply: str


class Chain(BaseModel):
    """A full forward walk from the customer's opening message."""

    chain_id: str
    root_tweet_id: str
    tweet_ids: list[str]
    is_repeat_contact: bool


class IntentFlags(BaseModel):
    """Sensitive-action flags derived from the classified intent."""

    refund: bool = False
    payment: bool = False


class Signals(BaseModel):
    """Assembled evidence used by decide()."""

    intent: str
    language: str
    confidence: float
    top1_sim: float
    mean_sim: float
    intent_flags: IntentFlags = IntentFlags()
    is_repeat_contact: bool
    is_other: bool
    safety_violations: list[str] = []
    llm_unavailable: bool = False


Action = Literal["auto", "escalate"]


class Decision(BaseModel):
    decision_id: str
    tweet_id: str
    message: str
    intent: str
    language: str
    classify_confidence: float
    top1_sim: float
    is_repeat: bool
    reply: str | None = None
    grounded_pair_ids: list[str] = []
    action: Action
    reason: str
    artifact_version: str
    intents_version: str
    latency_ms: float
    created_at: datetime
