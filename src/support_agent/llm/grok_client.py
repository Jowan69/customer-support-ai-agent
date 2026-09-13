"""The only module that talks to the Grok API. Nothing else knows about the key.

Wraps: env-based auth, retry x3 with backoff, timeout, JSON-schema-constrained parsing,
and an optional sha256(prompt) response cache.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import httpx
from config.settings import settings
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from support_agent.llm.cache import LLMCache

_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GrokUnavailableError(Exception):
    """Raised when the API fails after all retries, or returns unparseable JSON."""


def _is_retryable(exc: BaseException) -> bool:
    """Only retry what a second attempt could actually fix.

    A 4xx is a verdict, not a hiccup: a bad model name or a rejected key returns the
    same answer three times and just triples the latency of every failure. Rate
    limits and server errors are the ones worth waiting out.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, httpx.TransportError)  # timeouts, connection resets


# Blind exponential backoff is a guess. When the server tells us how long the
# bucket needs, that number is right and the guess is not - retrying early just
# spends another request against a quota that has not refilled yet.
_MAX_RETRY_WAIT_S = 60.0
_BACKOFF = wait_exponential(multiplier=1, min=1, max=10)


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Seconds the server asked us to wait, when it says so in a form we understand."""
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    header = exc.response.headers.get("retry-after")
    if not header:
        return None
    try:
        return min(float(header), _MAX_RETRY_WAIT_S)
    except ValueError:
        return None  # HTTP-date form; fall back to backoff rather than parsing dates


def _wait(retry_state) -> float:
    outcome = retry_state.outcome
    exc = outcome.exception() if outcome is not None else None
    if exc is not None:
        asked = _retry_after_seconds(exc)
        if asked is not None:
            return asked
    return _BACKOFF(retry_state)


def _cache_key(prompt: str, schema_name: str, json_schema: dict[str, Any]) -> str:
    """Everything that changes what a valid response looks like.

    Keying on the prompt alone replays an answer produced by a different model, or
    against a different schema, as though it were this one's. This project has already
    switched providers once; the cache must not survive that silently.
    """
    return "\x00".join(
        [
            settings.grok_model,
            schema_name,
            json.dumps(json_schema, sort_keys=True, separators=(",", ":")),
            prompt,
        ]
    )


class GrokClient:
    def __init__(self, cache: LLMCache | None = None):
        if not settings.grok_api_key:
            raise RuntimeError("GROK_API_KEY is not set (see .env.example)")
        self._cache = cache or LLMCache(settings.cache_dir / "llm_cache.db")
        self._client = httpx.Client(timeout=settings.llm_timeout_s)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GrokClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def complete_json(
        self,
        prompt: str,
        *,
        json_schema: dict[str, Any],
        schema_name: str = "response",
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Call Grok and parse the reply as JSON matching `json_schema`.

        Raises GrokUnavailableError on network/timeout failure after retries,
        or on a response that doesn't parse as valid JSON.
        """
        key = _cache_key(prompt, schema_name, json_schema)
        if use_cache:
            cached = self._cache.get(key)
            if cached is not None:
                return json.loads(cached)

        try:
            raw = self._call(prompt, json_schema, schema_name)
        except httpx.HTTPError as exc:
            raise GrokUnavailableError(str(exc)) from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GrokUnavailableError(f"non-JSON response: {exc}") from exc

        if use_cache:
            self._cache.set(key, raw)
        return parsed

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=_wait,
        reraise=True,
    )
    def _call(self, prompt: str, json_schema: dict[str, Any], schema_name: str) -> str:
        response = self._client.post(
            _API_URL,
            headers={"Authorization": f"Bearer {settings.grok_api_key}"},
            json={
                "model": settings.grok_model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
    "name": schema_name,
    "strict": True,
    "schema": json_schema,
},
                },
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


@lru_cache(maxsize=1)
def get_client() -> GrokClient:
    """The shared client for this process.

    `handle()` used to build one per message and never close it, so every httpx
    connection pool it opened leaked; a few hundred eval rows is a few hundred
    sockets. One client, reused, keep-alive intact.
    """
    return GrokClient()
