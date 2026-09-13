import httpx
import pytest
from config.settings import settings

from support_agent.llm.cache import LLMCache
from support_agent.llm.grok_client import (
    GrokClient,
    GrokUnavailableError,
    _retry_after_seconds,
)

_SCHEMA = {"type": "object", "properties": {"intent": {"type": "string"}}}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "grok_api_key", "test-key")
    return GrokClient(cache=LLMCache(tmp_path / "cache.db"))


_URL = "https://api.groq.com/openai/v1/chat/completions"


def _mock_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}]},
        request=httpx.Request("POST", _URL),
    )


def _error_response(status: int) -> httpx.Response:
    return httpx.Response(status, json={"error": "nope"}, request=httpx.Request("POST", _URL))


def test_timeout_retries_three_times_then_raises_unavailable(client, monkeypatch):
    calls = {"n": 0}

    def always_times_out(*args, **kwargs):
        calls["n"] += 1
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(client._client, "post", always_times_out)

    with pytest.raises(GrokUnavailableError):
        client.complete_json("prompt", json_schema=_SCHEMA, use_cache=False)

    assert calls["n"] == 3


def test_bad_json_response_raises_unavailable(client, monkeypatch):
    monkeypatch.setattr(client._client, "post", lambda *a, **k: _mock_response("not json"))

    with pytest.raises(GrokUnavailableError):
        client.complete_json("prompt", json_schema=_SCHEMA, use_cache=False)


def test_valid_response_is_parsed_and_cached(client, monkeypatch):
    calls = {"n": 0}

    def post(*args, **kwargs):
        calls["n"] += 1
        return _mock_response('{"intent": "order_status"}')

    monkeypatch.setattr(client._client, "post", post)

    first = client.complete_json("prompt", json_schema=_SCHEMA)
    second = client.complete_json("prompt", json_schema=_SCHEMA)

    assert first == {"intent": "order_status"}
    assert second == {"intent": "order_status"}
    assert calls["n"] == 1  # second call hit the cache


def test_client_error_is_not_retried(client, monkeypatch):
    """A 400 is a verdict, not a hiccup - retrying it just triples the failure time."""
    calls = {"n": 0}

    def post(*args, **kwargs):
        calls["n"] += 1
        return _error_response(400)

    monkeypatch.setattr(client._client, "post", post)

    with pytest.raises(GrokUnavailableError):
        client.complete_json("prompt", json_schema=_SCHEMA, use_cache=False)

    assert calls["n"] == 1


def test_server_error_is_retried(client, monkeypatch):
    calls = {"n": 0}

    def post(*args, **kwargs):
        calls["n"] += 1
        return _error_response(503)

    monkeypatch.setattr(client._client, "post", post)

    with pytest.raises(GrokUnavailableError):
        client.complete_json("prompt", json_schema=_SCHEMA, use_cache=False)

    assert calls["n"] == 3


def test_rate_limit_is_retried(client, monkeypatch):
    calls = {"n": 0}

    def post(*args, **kwargs):
        calls["n"] += 1
        return _error_response(429)

    monkeypatch.setattr(client._client, "post", post)

    with pytest.raises(GrokUnavailableError):
        client.complete_json("prompt", json_schema=_SCHEMA, use_cache=False)

    assert calls["n"] == 3


def _status_error(status: int, headers: dict | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", _URL)
    response = httpx.Response(status, headers=headers or {}, request=request)
    return httpx.HTTPStatusError("rate limited", request=request, response=response)


def test_retry_after_is_honoured_when_the_server_sends_it():
    """The server knows when its bucket refills; our backoff is only a guess."""
    assert _retry_after_seconds(_status_error(429, {"retry-after": "7.5"})) == 7.5


def test_retry_after_is_capped():
    assert _retry_after_seconds(_status_error(429, {"retry-after": "3600"})) == 60.0


def test_missing_retry_after_falls_back_to_backoff():
    assert _retry_after_seconds(_status_error(429)) is None


def test_http_date_retry_after_falls_back_to_backoff():
    """Ollama and some proxies send the HTTP-date form; backoff is close enough."""
    header = {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}

    assert _retry_after_seconds(_status_error(429, header)) is None


def test_non_http_errors_have_no_retry_after():
    assert _retry_after_seconds(httpx.ReadTimeout("timed out")) is None
