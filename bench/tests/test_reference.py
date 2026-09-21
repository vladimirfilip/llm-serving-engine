"""The HF reference on a tiny random Llama, on CPU."""

import json

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from bench import reference

VOCAB = 128


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256, pad_token_id=0)
    path = tmp_path_factory.mktemp("tiny")
    LlamaForCausalLM(config).save_pretrained(path)
    return str(path)


def prompt(i: int, length: int) -> dict:
    ids = [1, *np.random.default_rng(i).integers(2, VOCAB, size=length - 1).tolist()]
    return {"id": f"p{i}", "prompt_token_ids": ids}


def model_of(path):
    return reference.load_model(path, "cpu", torch.float32)


def test_generation_is_greedy_and_records_the_logprob_of_each_chosen_token(tiny_model):
    model = model_of(tiny_model)
    (record,) = reference.generate(model, [prompt(0, 10)], 6, 3, pad_id=0)
    assert len(record["token_ids"]) == 6 == len(record["logprobs"])
    ids = torch.tensor([prompt(0, 10)["prompt_token_ids"]])
    for token, lp in zip(record["token_ids"], record["logprobs"], strict=True):
        logits = model(ids).logits[0, -1].float()
        assert token == int(logits.argmax())
        assert lp == pytest.approx(float(logits.log_softmax(-1)[token]), abs=1e-4)
        ids = torch.cat([ids, torch.tensor([[token]])], dim=1)


def test_batched_generation_matches_generating_each_prompt_alone(tiny_model):
    model = model_of(tiny_model)
    prompts = [prompt(1, 6), prompt(2, 11), prompt(3, 9)]
    batched = reference.generate(model, prompts, 5, 3, pad_id=0)
    for p in prompts:
        alone = reference.generate(model, [p], 5, 3, pad_id=0)[0]
        assert next(r for r in batched if r["id"] == p["id"])["token_ids"] == alone["token_ids"]


def test_generation_runs_past_the_stop_token(tiny_model):
    model = model_of(tiny_model)
    model.generation_config.eos_token_id = int(reference.generate(
        model, [prompt(4, 8)], 1, 1, pad_id=0)[0]["token_ids"][0])
    (record,) = reference.generate(model, [prompt(4, 8)], 8, 3, pad_id=0)
    assert len(record["token_ids"]) == 8


def test_teacher_forcing_the_references_own_tokens_agrees_with_its_generation(tiny_model):
    model = model_of(tiny_model)
    p = prompt(5, 12)
    (gen,) = reference.generate(model, [p], 6, 3, pad_id=0)
    forced = reference.teacher_forced(model, p["prompt_token_ids"], gen["token_ids"])
    assert forced["ref_argmax"] == gen["token_ids"]
    assert forced["ref_logprob"] == pytest.approx(gen["logprobs"], abs=1e-3)
    assert forced["ref_top1"] == pytest.approx(forced["ref_logprob"], abs=1e-5)
    assert all(t2 <= t1 for t1, t2 in zip(forced["ref_top1"], forced["ref_top2"], strict=True))


def test_teacher_forcing_another_token_gives_its_logprob_and_a_smaller_top1_gap(tiny_model):
    model = model_of(tiny_model)
    p = prompt(6, 9)
    forced = reference.teacher_forced(model, p["prompt_token_ids"], [3, 4])
    n = len(p["prompt_token_ids"])
    logits = model(torch.tensor([p["prompt_token_ids"] + [3, 4]])).logits[0].float().log_softmax(-1)
    assert forced["ref_logprob"] == pytest.approx(
        [logits[n - 1, 3].item(), logits[n, 4].item()], abs=1e-4)


def test_perplexity_windows_are_non_overlapping_each_behind_a_bos_and_drop_the_remainder():
    windows = reference.ppl_windows(np.arange(10, 30, dtype=np.int32), window=8, bos=1)
    assert windows == [[1, *range(10, 18)], [1, *range(18, 26)]]


def test_window_logprobs_match_a_full_forward_and_perplexity_is_exp_of_the_mean(tiny_model):
    model = model_of(tiny_model)
    ids = [1, *range(5, 21)]
    lps = reference.window_logprobs(model, ids)
    logits = model(torch.tensor([ids])).logits[0, :-1].float().log_softmax(-1)
    expected = logits.gather(-1, torch.tensor(ids[1:]).unsqueeze(-1)).squeeze(-1)
    assert lps == pytest.approx(expected.tolist(), abs=1e-4)
    assert reference.perplexity([lps, lps]) == pytest.approx(float(np.exp(-np.mean(lps))))


def test_tasks_cache_generations_by_prompt_and_regenerate_a_changed_prompt(tiny_model, tmp_path):
    out = tmp_path / "gen.jsonl"
    prompts = [prompt(7, 8), prompt(8, 9)]
    args = dict(model_path=tiny_model, out_path=str(out), new_tokens=4, long_new_tokens=2,
                device="cpu", dtype="float32")
    assert reference.task_generate(prompts=prompts, **args) == {"generated": 2, "cached": 0}
    assert reference.task_generate(prompts=prompts, **args) == {"generated": 0, "cached": 2}
    changed = [prompts[0], prompt(9, 9) | {"id": "p8"}]
    assert reference.task_generate(prompts=changed, **args) == {"generated": 1, "cached": 1}
    assert {r["id"] for r in reference.read_jsonl(out)} == {"p7", "p8"}


def test_score_task_writes_one_record_per_engine_generation(tiny_model, tmp_path):
    gen = tmp_path / "engine.jsonl"
    reference.write_jsonl(gen, [{"id": "p10", "token_ids": [3, 4, 5]}])
    result = reference.task_score(tiny_model, [prompt(10, 8)], str(gen),
                                  str(tmp_path / "scored.jsonl"), device="cpu", dtype="float32")
    assert result == {"scored": 1}
    record = json.loads((tmp_path / "scored.jsonl").read_text())
    assert len(record["ref_logprob"]) == 3 and record["id"] == "p10"


def test_ppl_task_reports_the_perplexity_of_the_test_windows(tiny_model, tmp_path):
    ids = np.random.default_rng(0).integers(2, VOCAB, size=100).astype(np.int32)
    np.save(tmp_path / "test.npy", ids)
    result = reference.task_ppl(tiny_model, str(tmp_path / "test.npy"), 16, 1,
                                str(tmp_path / "ppl.json"), device="cpu", dtype="float32")
    assert result["windows"] == 6 and 1 < result["ppl"] < VOCAB * 3
    assert json.loads((tmp_path / "ppl.json").read_text()) == result
