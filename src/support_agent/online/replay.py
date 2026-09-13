"""Run the online pipeline from recorded responses: no API key, no index, no encoder.

The pipeline has exactly two doors to the outside world - the LLM client and
`retrieve()` - and both are injectable into `handle()`. This module supplies recorded
stand-ins for each, plus the recorder that produced them.

Why this exists: a clone of this repo has neither the 500MB FAISS index nor a Groq key,
so without fixtures there is no way for a reader to watch the pipeline run at all. The
alternative on offer was rebuilding the artifacts from the 500MB corpus first, which is
hours of work to see five messages go through. A fixture is 4KB.

What a fixture is and is not. It is a real recording: the pairs the index actually
returned for that message and the JSON the model actually replied with, captured by
`--record` during a real run. It is not a simulation, and nothing here invents a
response - a message with no fixture raises rather than quietly degrading, because a
demo that silently escalates everything looks like a working agent making a decision.

What it cannot show: whether retrieval finds good neighbours today, or how the model
answers a message nobody recorded. Replay reproduces recorded rows and nothing else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_DIR = Path("fixtures") / "replay"


class FixtureError(RuntimeError):
    """No recording for something the pipeline asked for.

    Deliberately not a `GrokUnavailableError`, `KeyError`, `TypeError` or `ValueError`:
    `classify()` catches all four and answers "other / escalate / llm_unavailable". A
    missing fixture would then look exactly like a dead API, and the demo would print a
    plausible escalation for a row it never actually ran.
    """


@dataclass(frozen=True)
class Fixture:
    tweet_id: str
    message: str
    is_repeat: bool
    retrieved: list[dict[str, Any]]
    responses: dict[str, Any]
    meta: dict[str, Any]

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Fixture:
        return cls(
            tweet_id=str(payload["tweet_id"]),
            message=str(payload["message"]),
            is_repeat=bool(payload.get("is_repeat", False)),
            retrieved=list(payload.get("retrieved", [])),
            responses=dict(payload.get("responses", {})),
            meta=dict(payload.get("meta", {})),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "tweet_id": self.tweet_id,
            "message": self.message,
            "is_repeat": self.is_repeat,
            "retrieved": self.retrieved,
            "responses": self.responses,
            "meta": self.meta,
        }


def load_fixtures(directory: Path) -> dict[str, Fixture]:
    """tweet_id -> Fixture, from `<directory>/<tweet_id>.json`."""
    directory = Path(directory)
    if not directory.exists():
        raise FixtureError(
            f"no fixtures at {directory} - record some with "
            "`python -m scripts.run_online --record --limit 5` on a machine that has "
            "the artifacts and an API key"
        )
    fixtures: dict[str, Fixture] = {}
    for path in sorted(directory.glob("*.json")):
        fixture = Fixture.from_json(json.loads(path.read_text(encoding="utf-8")))
        fixtures[fixture.tweet_id] = fixture
    if not fixtures:
        raise FixtureError(f"{directory} contains no *.json fixtures")
    return fixtures


def write_fixture(directory: Path, fixture: Fixture) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{fixture.tweet_id}.json"
    path.write_text(json.dumps(fixture.to_json(), indent=2, ensure_ascii=False), "utf-8")
    return path


class ReplayClient:
    """Stands in for GrokClient, answering from one row's recording.

    Keyed by `schema_name` ("classify", "draft") rather than by a hash of the prompt.
    A prompt hash would be the stricter check, but it also makes every fixture in the
    repo expire the moment a prompt template gains a comma - and the fixtures exist so
    a reader can watch the pipeline run, not to pin the prompt text. `tests/` is where
    prompt changes get caught.
    """

    def __init__(self, fixture: Fixture):
        self._fixture = fixture

    def complete_json(
        self,
        prompt: str,  # noqa: ARG002 - the recording is keyed by schema, not by prompt
        *,
        json_schema: dict[str, Any],  # noqa: ARG002
        schema_name: str = "response",
        use_cache: bool = True,  # noqa: ARG002
    ) -> dict[str, Any]:
        if schema_name not in self._fixture.responses:
            raise FixtureError(
                f"fixture {self._fixture.tweet_id} has no recorded '{schema_name}' "
                f"response (has: {sorted(self._fixture.responses) or 'nothing'})"
            )
        return self._fixture.responses[schema_name]

    def close(self) -> None:
        return None


class RecordingClient:
    """Wraps a real client and keeps the last response for each schema_name."""

    def __init__(self, inner: Any):
        self._inner = inner
        self.responses: dict[str, Any] = {}

    def complete_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        result = self._inner.complete_json(prompt, **kwargs)
        self.responses[kwargs.get("schema_name", "response")] = result
        return result

    def close(self) -> None:
        return None


def replay_retriever(fixture: Fixture):
    """A `retrieve`-shaped callable that returns the pairs recorded for this row."""

    def retrieve(message: str, k: int | None = None):  # noqa: ARG001
        pairs = fixture.retrieved
        return list(pairs[:k] if k else pairs)

    return retrieve


def recording_retriever(inner, captured: list[dict[str, Any]]):
    """A `retrieve`-shaped callable that also stores what came back."""

    def retrieve(message: str, k: int | None = None):
        pairs = inner(message, k=k)
        captured.clear()
        captured.extend(dict(p) for p in pairs)
        return pairs

    return retrieve


def new_fixture(
    tweet_id: str,
    message: str,
    is_repeat: bool,
    retrieved: list[dict[str, Any]],
    responses: dict[str, Any],
    *,
    artifact_version: str,
    model: str,
) -> Fixture:
    return Fixture(
        tweet_id=str(tweet_id),
        message=message,
        is_repeat=bool(is_repeat),
        retrieved=retrieved,
        responses=responses,
        meta={
            # Stamped so a fixture recorded against another taxonomy or another model
            # is identifiable as such rather than silently replayed as this one's.
            "recorded_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "artifact_version": artifact_version,
            "model": model,
        },
    )
