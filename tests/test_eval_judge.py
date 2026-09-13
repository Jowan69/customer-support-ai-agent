"""The judge scoring pass: schema discipline, caching, resume."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from support_agent.eval import judge


class FakeClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts = []

    def complete_json(self, prompt, json_schema=None, schema_name=None):
        self.prompts.append(prompt)
        return self.payloads.pop(0)


def _payload(**overrides):
    base = {d: 4 for d in judge.DIMENSIONS}
    base.update({f"{d}_reason": "because" for d in judge.DIMENSIONS})
    base.update(overrides)
    return base


def test_all_five_dimensions_are_required_by_the_schema():
    assert set(judge.DIMENSIONS) <= set(judge.JUDGE_SCHEMA["required"])
    for dimension in judge.DIMENSIONS:
        prop = judge.JUDGE_SCHEMA["properties"][dimension]
        assert prop == {"type": "integer", "minimum": 1, "maximum": 5}


def test_off_scale_score_raises_rather_than_entering_the_mean():
    client = FakeClient([_payload(tone=7)])
    with pytest.raises(ValueError, match="tone=7"):
        judge.judge_one("m", "r", [], client)


def test_prompt_carries_sources_and_reply():
    client = FakeClient([_payload()])
    judge.judge_one("customer text", "drafted text", ["a past reply"], client)
    prompt = client.prompts[0]
    assert "customer text" in prompt
    assert "drafted text" in prompt
    assert "a past reply" in prompt


def test_rows_without_a_reply_are_not_judged(tmp_path):
    merged = pd.DataFrame(
        {
            "tweet_id": ["1", "2"],
            "message": ["a", "b"],
            "reply": ["a reply", None],
            "grounded_pair_ids": ["[]", "[]"],
        }
    )
    client = FakeClient([_payload()])
    out = judge.run(merged, client, out_path=tmp_path / "judge.csv")
    assert out["tweet_id"].tolist() == ["1"]


def test_resume_skips_rows_already_scored(tmp_path):
    out_path = tmp_path / "judge.csv"
    pd.DataFrame([{"tweet_id": "1", **{d: 3 for d in judge.DIMENSIONS}}]).to_csv(
        out_path, index=False
    )
    merged = pd.DataFrame(
        {
            "tweet_id": ["1", "2"],
            "message": ["a", "b"],
            "reply": ["r1", "r2"],
            "grounded_pair_ids": ["[]", "[]"],
        }
    )
    client = FakeClient([_payload()])  # exactly one payload: row 1 must not be re-judged
    out = judge.run(merged, client, out_path=out_path)
    assert sorted(out["tweet_id"]) == ["1", "2"]
    assert len(client.prompts) == 1


def test_sources_resolve_from_grounded_pair_ids(tmp_path):
    merged = pd.DataFrame(
        {
            "tweet_id": ["1"],
            "message": ["a"],
            "reply": ["r"],
            "grounded_pair_ids": [json.dumps(["p1", "missing"])],
        }
    )
    client = FakeClient([_payload()])
    judge.run(
        merged, client, out_path=tmp_path / "j.csv", sources_by_pair={"p1": "past reply text"}
    )
    assert "past reply text" in client.prompts[0]


def test_prompt_source_block_matches_the_sheet_rendering():
    """The judge prompt and the human score sheet must render sources identically."""
    client = FakeClient([_payload()])
    judge.judge_one("m", "r", ["one", "two"], client)
    assert judge.format_sources(["one", "two"]) in client.prompts[0]


def test_no_sources_renders_as_none():
    assert judge.format_sources([]) == "(none)"


def test_load_sources_is_empty_without_the_artifact(tmp_path):
    assert judge.load_sources(tmp_path) == {}


def test_load_sources_maps_pair_id_to_agent_reply(tmp_path):
    pd.DataFrame(
        {"pair_id": ["p1", "p2"], "agent_reply": ["r1", "r2"], "customer_msg": ["a", "b"]}
    ).to_parquet(tmp_path / "pairs_meta.parquet", index=False)
    assert judge.load_sources(tmp_path) == {"p1": "r1", "p2": "r2"}
