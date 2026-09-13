"""Replay: the pipeline runs from recorded fixtures, and says so when one is missing."""

from __future__ import annotations

import json

import pytest

from support_agent.online import replay

_FIXTURE = {
    "tweet_id": "42",
    "message": "@AmazonHelp where is my parcel",
    "is_repeat": True,
    "retrieved": [
        {
            "pair_id": "7_8",
            "customer_msg": "where is my order",
            "agent_reply": "we are looking into it",
            "score": 0.81,
        }
    ],
    "responses": {
        "classify": {"intent": "delivery_late", "confidence": 0.82},
        "draft": {"reply": "Sorry about that - we are checking.", "grounded_pair_ids": ["7_8"]},
    },
    "meta": {"artifact_version": "v2", "model": "openai/gpt-oss-20b"},
}


def _write(tmp_path, payload=None):
    directory = tmp_path / "replay"
    directory.mkdir()
    payload = payload or _FIXTURE
    (directory / f"{payload['tweet_id']}.json").write_text(json.dumps(payload), "utf-8")
    return directory


def test_fixtures_load_by_tweet_id(tmp_path):
    fixtures = replay.load_fixtures(_write(tmp_path))
    assert fixtures["42"].message.endswith("parcel")
    assert fixtures["42"].is_repeat is True


def test_missing_directory_says_how_to_record_one(tmp_path):
    with pytest.raises(replay.FixtureError, match="--record"):
        replay.load_fixtures(tmp_path / "nope")


def test_empty_directory_is_an_error_not_an_empty_run(tmp_path):
    (tmp_path / "replay").mkdir()
    with pytest.raises(replay.FixtureError):
        replay.load_fixtures(tmp_path / "replay")


def test_replay_client_answers_from_the_recording(tmp_path):
    fixture = replay.load_fixtures(_write(tmp_path))["42"]
    client = replay.ReplayClient(fixture)
    result = client.complete_json("any prompt", json_schema={}, schema_name="classify")
    assert result["intent"] == "delivery_late"


def test_missing_response_raises_rather_than_looking_like_a_dead_api(tmp_path):
    """classify() catches GrokUnavailableError/KeyError/TypeError/ValueError and answers
    "other, escalate, llm_unavailable". A missing fixture must not be able to hide in
    that - a demo printing a plausible escalation for a row it never ran is worse than
    a crash."""
    fixture = replay.load_fixtures(_write(tmp_path))["42"]
    client = replay.ReplayClient(fixture)

    with pytest.raises(replay.FixtureError):
        client.complete_json("p", json_schema={}, schema_name="judge")

    assert not issubclass(replay.FixtureError, KeyError | TypeError | ValueError)


def test_replay_retriever_returns_the_recorded_pairs(tmp_path):
    fixture = replay.load_fixtures(_write(tmp_path))["42"]
    pairs = replay.replay_retriever(fixture)("ignored")
    assert [p["pair_id"] for p in pairs] == ["7_8"]
    assert pairs[0]["score"] == 0.81


def test_recording_client_keeps_each_schemas_last_response():
    class Inner:
        def complete_json(self, prompt, **kwargs):
            return {"echo": kwargs["schema_name"]}

    client = replay.RecordingClient(Inner())
    client.complete_json("p", schema_name="classify", json_schema={})
    client.complete_json("p", schema_name="draft", json_schema={})
    assert client.responses == {"classify": {"echo": "classify"}, "draft": {"echo": "draft"}}


def test_recording_retriever_captures_what_came_back():
    captured: list[dict] = []
    inner = lambda message, k=None: [{"pair_id": "1_2", "score": 0.5}]  # noqa: E731
    pairs = replay.recording_retriever(inner, captured)("msg")
    assert captured == pairs


def test_round_trip_through_disk_preserves_the_recording(tmp_path):
    fixture = replay.load_fixtures(_write(tmp_path))["42"]
    out = replay.write_fixture(tmp_path / "out", fixture)
    assert replay.load_fixtures(out.parent)["42"] == fixture
