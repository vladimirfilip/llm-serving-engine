import json

from llm_serving_engine.loadgen import cli


def test_sampling_params_none_when_no_overrides_given():
    args = cli._parse_args(["--target-qps", "2", "--duration-s", "1"])
    assert cli._sampling_params_from_args(args) is None


def test_sampling_params_built_from_given_overrides_only():
    args = cli._parse_args(["--target-qps", "2", "--duration-s", "1", "--top-p", "0.5"])
    params = cli._sampling_params_from_args(args)
    assert params.top_p == 0.5
    assert params.temperature == 1.0  # the SamplingParams default


def test_max_tokens_overrides_every_request_shape():
    args = cli._parse_args(["--target-qps", "2", "--duration-s", "1", "--max-tokens", "8"])
    workload = cli._workload_from_args(args)
    assert len(workload) == len(cli.WORKLOAD)
    assert {shape.max_tokens for shape in workload} == {8}


def test_without_max_tokens_each_shape_keeps_its_own():
    args = cli._parse_args(["--target-qps", "2", "--duration-s", "1"])
    assert cli._workload_from_args(args) == cli.WORKLOAD


def test_main_writes_the_run_envelope_plotting_reads(tmp_path, monkeypatch):
    out = tmp_path / "run.json"
    result = {
        "shape": "chat", "latency": 0.5, "first_token_latency": 0.1, "scheduled_at": 0.0,
        "completed_at": 0.5, "success": True,
    }
    monkeypatch.setattr(cli, "open_loop_load_gen", lambda *_: _immediate([result]))

    cli.main(["--target-qps", "3", "--duration-s", "2", "--out", str(out)])

    run = json.loads(out.read_text())
    assert run["target_qps"] == 3
    assert run["duration_s"] == 2
    assert run["results"] == [result]


async def _immediate(value):
    return value
