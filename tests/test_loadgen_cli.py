import json

from llm_serving_engine.loadgen import cli


def test_sampling_params_none_when_no_overrides_given():
    args = cli._parse_args(["--target-qps", "2", "--duration-s", "1"])
    assert cli._sampling_params_from_args(args) is None


def test_sampling_params_built_from_given_overrides_only():
    args = cli._parse_args(["--target-qps", "2", "--duration-s", "1", "--max-tokens", "8"])
    params = cli._sampling_params_from_args(args)
    assert params.max_tokens == 8
    assert params.temperature == 1.0  # the SamplingParams default


def test_main_writes_the_run_envelope_plotting_reads(tmp_path, monkeypatch):
    out = tmp_path / "run.json"
    monkeypatch.setattr(cli, "_run", lambda config: _immediate([{"latency": 0.5, "success": True}]))

    cli.main(["--target-qps", "3", "--duration-s", "2", "--out", str(out)])

    run = json.loads(out.read_text())
    assert run["target_qps"] == 3
    assert run["duration_s"] == 2
    assert run["results"] == [{"latency": 0.5, "success": True}]


async def _immediate(value):
    return value
