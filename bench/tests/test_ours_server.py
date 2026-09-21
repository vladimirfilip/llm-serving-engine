"""The OpenAI-style wrapper around the engine, with the engine faked."""

import json
import types

import pytest
from fastapi.testclient import TestClient

from bench.engines.ours_server import create_app
from llm_serving_engine.engine import EngineUnavailable, InvalidPrompt, Submission
from llm_serving_engine.scheduling.dispatch import (
    ABORTED,
    DONE,
    new_output_channel,
    output_channels,
)


class FakeTokenizer:
    def encode(self, text):
        return [1, *map(ord, text)]

    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids)

    def decode_incremental(self, _seq_id, generated, skip_special_tokens=True):
        if generated[-1] == 100:
            return "" if skip_special_tokens else "<eot>"
        return "" if generated[-1] == 99 else chr(97 + generated[-1] % 26)

    def forget(self, seq_id):
        self.forgotten = seq_id


class FakeHandle:
    healthy = True

    def __init__(self, items=(1, 2, 3, DONE), logprobs=None, error=None):
        self.items, self.logprobs, self.error = list(items), logprobs, error
        self.tokenizer = FakeTokenizer()
        self.submitted = []
        self.engine = types.SimpleNamespace(
            tokenizer=self.tokenizer,
            stats=lambda: {"running": 2},
            model_runner=types.SimpleNamespace(score=lambda ids: [-1.0] * (len(ids) - 1)),
        )

    def bind_loop(self, loop):
        self.loop = loop

    def submit_tokens(self, tokens, params, logprobs=False):
        if self.error:
            raise self.error
        self.submitted.append((tokens, params, logprobs))
        seq_id, queue = new_output_channel(64)
        for item in self.items:
            queue.put_nowait(item)
        return Submission(seq_id, queue, len(tokens), self.tokenizer,
                          list(self.logprobs) if logprobs else None)


def post(handle, **body):
    body = {"model": "m", "prompt": [1, 2, 3], "max_tokens": 3, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, **body}
    with TestClient(create_app(handle, max_model_len=100)) as client:
        return client.post("/v1/completions", json=body)


def events(response) -> list[str]:
    return [line[6:] for line in response.text.split("\n\n") if line.startswith("data: ")]


def test_a_stream_has_one_event_per_token_then_usage_then_done():
    response = post(FakeHandle())
    lines = events(response)
    assert lines[-1] == "[DONE]"
    tokens, usage = lines[:3], json.loads(lines[3])
    assert [json.loads(t)["choices"][0]["text"] for t in tokens] == ["b", "c", "d"]
    assert usage == {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 3,
                                              "total_tokens": 6}}


def test_the_request_reaches_the_engine_as_token_ids_with_greedy_params_and_ignore_eos():
    handle = FakeHandle()
    post(handle, ignore_eos=True)
    tokens, params, logprobs = handle.submitted[0]
    assert tokens == [1, 2, 3] and logprobs is False
    assert (params.temperature, params.max_tokens, params.ignore_eos) == (0, 3, True)


def test_a_text_prompt_is_tokenized_with_the_engines_tokenizer():
    handle = FakeHandle()
    post(handle, prompt="ab")
    assert handle.submitted[0][0] == [1, 97, 98]


def test_logprob_requests_carry_token_ids_and_the_chosen_logprob_per_event():
    handle = FakeHandle(logprobs=[-0.5, -1.5, -2.5])
    lines = events(post(handle, logprobs=1))
    chosen = [json.loads(line)["choices"][0]["logprobs"] for line in lines[:3]]
    assert chosen == [{"tokens": [f"token_id:{i}"], "token_logprobs": [lp]}
                      for i, lp in zip((1, 2, 3), (-0.5, -1.5, -2.5), strict=True)]


def test_a_held_back_character_still_yields_an_event_with_empty_text():
    lines = events(post(FakeHandle(items=(99, 2, 3, DONE))))
    assert [json.loads(line)["choices"][0]["text"] for line in lines[:3]] == ["", "c", "d"]


def test_special_tokens_get_text_only_when_the_request_asks_for_them():
    items = (1, 100, 3, DONE)
    skipped = events(post(FakeHandle(items=items)))
    kept = events(post(FakeHandle(items=items), skip_special_tokens=False))
    assert [json.loads(line)["choices"][0]["text"] for line in skipped[:3]] == ["b", "", "d"]
    assert [json.loads(line)["choices"][0]["text"] for line in kept[:3]] == ["b", "<eot>", "d"]


def test_a_logprob_stream_that_lost_a_token_ends_in_an_error_not_shifted_logprobs():
    handle = FakeHandle(items=(1, 2, DONE), logprobs=[-0.5, -1.5, -2.5])  # 3 logprobs, 2 tokens
    lines = events(post(handle, logprobs=1))
    assert json.loads(lines[-1]) == {"error": "1 token(s) dropped from a logprob stream"}
    assert "[DONE]" not in lines


def test_an_aborted_request_ends_with_an_error_event_and_no_done():
    lines = events(post(FakeHandle(items=(1, ABORTED))))
    assert json.loads(lines[-1]) == {"error": "request aborted by the engine"}
    assert "[DONE]" not in lines


def test_the_stream_pops_its_output_channel_when_it_ends():
    handle = FakeHandle()
    post(handle)
    assert output_channels == {} and handle.tokenizer.forgotten is not None


def test_a_non_streamed_request_returns_json_with_usage():
    response = post(FakeHandle(), stream=False)
    body = response.json()
    assert body["choices"][0]["text"] == "bcd"
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}


def test_a_stop_sequence_cuts_a_non_streamed_completion_at_its_first_occurrence():
    response = post(FakeHandle(items=(1, 2, 3, 4, DONE)), stream=False, stop=["d"])
    assert response.json()["choices"][0]["text"] == "bc"
    assert post(FakeHandle(items=(1, 2, 3, DONE)), stream=False, stop="zz").json()[
        "choices"][0]["text"] == "bcd"


def test_a_request_past_max_model_len_is_rejected_before_the_engine_sees_it():
    handle = FakeHandle()
    response = post(handle, max_tokens=98)
    assert response.status_code == 400 and "max_model_len" in response.text
    assert handle.submitted == []


@pytest.mark.parametrize(("error", "status"),
                         [(EngineUnavailable("stopping"), 503), (InvalidPrompt("bad"), 400)])
def test_engine_refusals_map_to_http_statuses(error, status):
    assert post(FakeHandle(error=error)).status_code == status


def test_health_stats_and_score_endpoints():
    handle = FakeHandle()
    with TestClient(create_app(handle, 100)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/internal/stats").json() == {"running": 2}
        assert client.post("/internal/score", json={"token_ids": [1, 2, 3, 4]}).json() == {
            "logprobs": [-1.0, -1.0, -1.0]}
        handle.healthy = False
        assert client.get("/health").status_code == 503
