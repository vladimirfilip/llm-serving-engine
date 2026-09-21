import json
from pathlib import Path

import pytest
import yaml

from bench.reference import read_jsonl, write_jsonl
from bench.run import Run
from bench.suites import correctness


@pytest.fixture
def run(fast_config, tmp_path, synthetic_datasets):
    return Run.open(fast_config, "cor", results_dir=tmp_path / "results")


def test_the_prompt_selection_takes_the_configured_short_and_long_prompts(run):
    prompts = correctness.select_prompts(run)
    kinds = [p["kind"] for p in prompts]
    assert kinds == ["sharegpt"] * 6 + ["long"]
    assert len(prompts[-1]["prompt_token_ids"]) == 600


def test_a_fake_engine_is_checked_for_batch_invariance_and_gates_are_not_evaluated(run):
    correctness.execute(run, ["mock"])
    summary = json.loads((run.dir / "correctness" / "summary.json").read_text())
    invariance = summary["mock"]["batch_invariance"]
    assert invariance["run_to_run_identical_fraction"] == 1.0
    assert invariance["identical_fraction_vs_batched"] == 1.0
    assert invariance["identical_fraction_vs_shuffled"] == 1.0
    assert summary["mock"]["generated"] == 7 == summary["mock"]["requested"]
    assert "ppl" not in summary["mock"] and "abs_dlogprob_mean" not in summary["mock"]
    assert all(g["passed"] is None for g in summary["gates"].values())
    gens = read_jsonl(run.dir / "correctness" / "mock" / "gen.jsonl")
    assert [len(g["token_ids"]) for g in gens] == [24] * 6 + [8]
    assert all(len(g["logprobs"]) == len(g["token_ids"]) for g in gens)


@pytest.fixture
def real_model_run(fast_config_dir, tmp_path, synthetic_datasets, monkeypatch):
    """The mock declared a real model, with the reference tasks and lm_eval faked at their
    process boundary: the reference agrees with the mock exactly."""
    from bench.config import load_config

    path = fast_config_dir / "engines" / "mock.yaml"
    spec = yaml.safe_load(path.read_text()) | {"real_model": True}
    path.write_text(yaml.safe_dump(spec))
    run = Run.open(load_config(fast_config_dir), "cor-real", results_dir=tmp_path / "results")

    from bench.engines.mock_server import logprob_at, token_at

    def fake_task(task, **args):
        ids = set(args["ids"]) if "ids" in args else set()
        prompts = {p["id"]: p for p in read_jsonl(synthetic_datasets / "correctness_prompts.jsonl")
                   if p["id"] in ids}
        if task == "reference-generate":
            def n(prompt_id):
                key = "long_new_tokens" if prompt_id.startswith("long") else "new_tokens"
                return args[key]

            write_jsonl(Path(args["out_path"]), [
                {"id": i, "token_ids": [token_at(p["prompt_token_ids"], k) for k in range(n(i))],
                 "logprobs": [logprob_at(p["prompt_token_ids"], k) for k in range(n(i))]}
                for i, p in prompts.items()])
        elif task == "reference-score":
            records = []
            for g in read_jsonl(Path(args["engine_gen_path"])):
                lps = g["logprobs"]
                records.append({"id": g["id"], "ref_logprob": lps, "ref_argmax": g["token_ids"],
                                "ref_top1": lps, "ref_top2": [lp - 5 for lp in lps]})
            write_jsonl(Path(args["out_path"]), records)
        elif task == "reference-ppl":
            Path(args["out_path"]).write_text(json.dumps({"ppl": 5.0}))
        return {}

    monkeypatch.setattr(correctness, "run_task", fake_task)
    monkeypatch.setattr(correctness, "run_gsm8k",
                        lambda *_a, **_k: {"gsm8k_strict": 0.5, "gsm8k_flexible": 0.6})
    return run


def test_an_engine_that_matches_the_reference_passes_the_gates_it_can_be_judged_on(
    real_model_run,
):
    correctness.execute(real_model_run, ["mock"])
    summary = json.loads((real_model_run.dir / "correctness" / "summary.json").read_text())
    mock = summary["mock"]
    assert mock["abs_dlogprob_mean"] == 0 and mock["tf_top1_agree"] == 1.0
    assert mock["confident_mismatch_rate"] == 0 and mock["no_divergence_fraction"] == 1.0
    assert mock["ref_ppl"] == 5.0 and mock["gsm8k_flexible"] == 0.6 and mock["ppl"] > 0
    # the gates judge the engine named `ours`; a fake named mock is measured but not gated
    assert all(g["passed"] is None for g in summary["gates"].values())
    assert (real_model_run.dir / "tables" / "correctness.csv").exists()


def test_batch_invariance_is_unavailable_not_perfect_when_an_engine_returns_no_token_ids(run):
    from bench.engines.base import Launch
    from bench.suites.common import engine_session

    data = correctness.load_datasets(run)
    launch = Launch(args_add=["--token-ids=false"])
    with engine_session(run, "mock", "t", launch, checks=False) as s:
        out = correctness.run_batch_invariance(s, correctness.select_prompts(run), data)
    assert out == {"unavailable": "the engine returned no token ids, so runs cannot be compared"}


def test_generation_keeps_a_record_with_no_ids_but_drops_one_with_missing_ids():
    assert correctness.complete_tokens({"token_ids": [], "completion_tokens_usage": 5})
    assert correctness.complete_tokens({"token_ids": [1, 2], "completion_tokens_usage": 2})
    assert not correctness.complete_tokens({"token_ids": [1], "completion_tokens_usage": 2})
