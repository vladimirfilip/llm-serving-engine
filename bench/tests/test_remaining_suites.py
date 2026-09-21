import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from bench.config import load_config
from bench.engines.base import Launch
from bench.run import Run
from bench.suites import ablation, coldstart, precision, probe, soak
from bench.suites.common import SuiteSkipped


@pytest.fixture
def run(fast_config, tmp_path, synthetic_datasets):
    return Run.open(fast_config, "rs", results_dir=tmp_path / "results")


def test_the_first_request_penalty_is_its_ttft_less_the_median_of_the_last_five():
    ttfts = [0.50, 0.10, 0.12, 0.11, 0.13, 0.09, 0.10, 0.11, 0.12, 0.10]
    assert coldstart.first_request_penalty(ttfts) == pytest.approx(0.50 - 0.10)


def test_coldstart_launches_repeatedly_and_labels_only_the_first_cold(run, monkeypatch):
    monkeypatch.setattr(coldstart, "drop_page_cache", lambda: True)
    coldstart.execute(run, ["mock"])
    table = pd.read_csv(run.dir / "tables" / "coldstart.csv")
    assert table.cache.tolist() == ["cold_cache", "warm_cache"]
    assert (table.wait_ready_s > 0).all() and table.first_request_penalty_s.notna().all()


def test_a_refused_cache_drop_labels_every_launch_warm(run, monkeypatch):
    monkeypatch.setattr(coldstart, "drop_page_cache", lambda: False)
    coldstart.execute(run, ["mock"])
    assert set(pd.read_csv(run.dir / "tables" / "coldstart.csv").cache) == {"warm_cache"}


def write_steps(run, steps):
    (run.cfg.dir / "engines" / "ours_ablation.yaml").write_text(yaml.safe_dump({"steps": steps}))


def test_an_ablation_step_with_a_placeholder_flag_exits_with_an_error_naming_the_step(run):
    write_steps(run, [{"name": "naive_baseline", "args_add": []},
                      {"name": "+paged_kv", "args_add": ["--<flag>"]}])
    with pytest.raises(ValueError, match=r"\+paged_kv.*<flag>"):
        ablation.load_steps(run)
    with pytest.raises(ValueError, match=r"\+paged_kv"):
        ablation.execute(run, ["ours"])


def test_ablation_steps_are_cumulative_in_arguments_and_environment():
    steps = [{"name": "a", "args_add": ["--x"], "env": {"K": "1"}},
             {"name": "b", "args_add": ["--y"]},
             {"name": "c", "env": {"K": "2", "L": 3}}]
    launches = [launch for _step, launch in ablation.cumulative(steps, Launch(4096, ["--base"]))]
    assert [launch.args_add for launch in launches] == [
        ["--base", "--x"], ["--base", "--x", "--y"], ["--base", "--x", "--y"]]
    assert [launch.env for launch in launches] == [{"K": "1"}, {"K": "1"}, {"K": "2", "L": "3"}]
    assert {launch.token_budget for launch in launches} == {4096}


def test_the_shipped_ablation_uses_only_flags_the_engine_has():
    config = load_config()
    steps = yaml.safe_load((config.dir / "engines" / "ours_ablation.yaml").read_text())["steps"]
    known = {"LLM_SCHEDULER", "LLM_KV_ALLOCATOR", "LLM_USE_CUSTOM_KERNELS", "LLM_USE_CUDA_GRAPHS"}
    assert [s["name"] for s in steps] == ["naive_baseline", "+continuous_batching",
                                          "+fused_kernels", "+paged_kv", "+cuda_graphs"]
    assert {k for s in steps for k in s.get("env", {})} <= known
    assert all(ablation.PLACEHOLDER not in a for s in steps for a in s.get("args_add", []))


def test_the_ablation_measures_each_step_and_skips_an_optional_step_the_engine_rejects(
    run, monkeypatch
):
    quick = dataclasses.replace(run.cfg, quick=True)
    run = Run.open(quick, "abl", results_dir=run.dir.parent)
    write_steps(run, [
        {"name": "naive", "args_add": ["--decode-ms=16"]},
        {"name": "+faster", "args_add": ["--decode-ms=4"]},
        {"name": "+unsupported", "args_add": ["--no-such-flag"], "optional": True},
    ])
    monkeypatch.setattr(ablation, "ABLATED_ENGINE", "mock")
    ablation.execute(run, ["mock"])
    table = pd.read_csv(run.dir / "tables" / "ablation.csv").set_index("step")
    assert table.loc["naive", "tpot_s"] > 2 * table.loc["+faster", "tpot_s"]
    assert table.loc["+faster", "out_tok_s_min"] <= table.loc["+faster", "out_tok_s"] <= \
        table.loc["+faster", "out_tok_s_max"]
    assert "exited during startup" in table.loc["+unsupported", "skipped"]


def test_a_required_step_the_engine_rejects_fails_the_phase(run, monkeypatch):
    write_steps(run, [{"name": "naive", "args_add": ["--no-such-flag"]}])
    monkeypatch.setattr(ablation, "ABLATED_ENGINE", "mock")
    with pytest.raises(RuntimeError, match="exited during startup"):
        ablation.execute(run, ["mock"])


def test_the_ablation_runs_on_ours_only(run):
    with pytest.raises(SuiteSkipped, match="ours only"):
        ablation.execute(run, ["vllm"])


def test_precision_variants_are_skipped_with_a_single_precision(run):
    with pytest.raises(SuiteSkipped, match="single precision"):
        precision.execute(run, ["ours"])


def test_precision_reports_capacity_and_perplexity_per_precision(run, monkeypatch):
    run.cfg.model["precisions"] = ["bfloat16", "float16"]
    monkeypatch.setattr(precision, "PRECISIONED_ENGINE", "mock")
    monkeypatch.setattr(precision, "run_gsm8k", lambda *_a, **_k: {"gsm8k_flexible": 0.5})
    monkeypatch.setattr(precision, "measure_perplexity", lambda _s: 7.0)
    precision.execute(run, ["mock"])
    table = pd.read_csv(run.dir / "tables" / "precision.csv")
    assert table.precision.tolist() == ["bfloat16", "float16"]
    assert (table.capacity_tok_s > 0).all() and table.ppl_delta_vs_first.tolist() == [0.0, 0.0]


LIMITS = {"gpu_mem_growth_max_frac": 0.01, "throughput_drift_max_frac": 0.03,
          "p99_tpot_drift_max_frac": 0.10}


def steady(duration_s: float, tok=100.0, p99=0.05, mem=8e9, step=10.0):
    t = np.arange(0, duration_s, step)
    tokens = pd.DataFrame({"t_s": t, "out_tok_s": tok})
    latency = pd.DataFrame({"t_s": t, "tpot_p99": p99})
    memory = pd.DataFrame({"t_s": t, "mem_used_bytes": mem})
    return tokens, latency, memory


def test_a_steady_run_passes_every_soak_criterion():
    tokens, latency, memory = steady(2400)
    verdict = soak.soak_verdict(tokens, latency, memory, 0, 2400, LIMITS)
    assert verdict["passed"] and all(verdict[k]["passed"] for k in
                                     ("gpu_mem_growth", "throughput_drift", "p99_tpot_growth",
                                      "errors"))


def test_each_soak_criterion_fails_on_its_own_violation():
    duration = 2400  # reference is 10 to 60 of 240 -> [100 s, 600 s), evaluation from 600 s
    tokens, latency, memory = steady(duration)
    late = memory.t_s >= 600
    grown = memory.assign(mem_used_bytes=np.where(late, 8e9 * 1.02, 8e9))
    assert not soak.soak_verdict(tokens, latency, grown, 0, duration, LIMITS)["gpu_mem_growth"][
        "passed"]
    slow = tokens.assign(out_tok_s=np.where(tokens.t_s >= 600, 95.0, 100.0))
    assert not soak.soak_verdict(slow, latency, memory, 0, duration, LIMITS)["throughput_drift"][
        "passed"]
    tail = latency.assign(tpot_p99=np.where(latency.t_s >= 600, 0.056, 0.05))
    assert not soak.soak_verdict(tokens, tail, memory, 0, duration, LIMITS)["p99_tpot_growth"][
        "passed"]
    verdict = soak.soak_verdict(tokens, latency, memory, 1, duration, LIMITS)
    assert not verdict["errors"]["passed"] and not verdict["passed"]


def test_soak_windows_count_tokens_and_completions_per_window():
    df = pd.DataFrame([
        {"status": "ok", "t_send": 0.0, "t_first": 0.1, "t_done": 1.0, "completion_tokens_usage": 3,
         "token_times": [0.1, 0.5, 1.0]},
        {"status": "ok", "t_send": 1.0, "t_first": 1.2, "t_done": 3.0, "completion_tokens_usage": 3,
         "token_times": [1.2, 2.0, 3.0]},
        {"status": "timeout", "t_send": 2.0, "t_first": np.nan, "t_done": np.nan,
         "completion_tokens_usage": 0, "token_times": []},
    ])
    out = soak.windowed(df, 2.0, 4.0)
    assert out.out_tok_s.tolist() == [pytest.approx(4 / 2), pytest.approx(2 / 2)]
    assert out.tpot_p50[0] == pytest.approx((1.0 - 0.1) / 2)
    assert out.ttft_p50[1] == pytest.approx(0.2)


def test_a_short_soak_on_a_fake_engine_writes_its_windows_and_verdict(run, monkeypatch):
    run.cfg.suite["soak"].update(duration_h=16 / 3600, engines=["mock"], sample_s=1)
    monkeypatch.setattr(soak, "CHUNK_S", 2.0)
    monkeypatch.setattr(soak, "THROUGHPUT_WINDOW_S", 2.0)
    monkeypatch.setattr(soak, "LATENCY_WINDOW_S", 4.0)
    probe.execute(run, ["mock"])
    soak.execute(run, ["mock"])
    folder = run.dir / "soak" / "mock"
    assert {p.name for p in folder.iterdir()} >= {"throughput.csv", "latency.csv", "verdict.json"}
    throughput = pd.read_csv(folder / "throughput.csv")
    assert len(throughput) == 8 and throughput.out_tok_s.max() > 0
    assert "cumulative_errors" in pd.read_csv(folder / "memory.csv")
    assert list(pd.read_csv(run.dir / "tables" / "soak.csv").engine) == ["mock"]


def test_soak_is_skipped_for_engines_it_does_not_cover(run):
    with pytest.raises(SuiteSkipped, match="soak runs on"):
        soak.execute(run, ["vllm"])


def test_a_soak_criterion_with_no_samples_is_not_evaluated_rather_than_failed():
    tokens, latency, memory = steady(2400)
    verdict = soak.soak_verdict(tokens, latency, memory.iloc[0:0], 0, 2400, LIMITS)
    assert verdict["gpu_mem_growth"]["passed"] is None
    assert "no mem_used_bytes samples" in verdict["gpu_mem_growth"]["reason"]
    assert verdict["throughput_drift"]["passed"] is True and verdict["passed"] is None
    failing = soak.soak_verdict(tokens, latency, memory.iloc[0:0], 3, 2400, LIMITS)
    assert failing["passed"] is False  # a real failure still fails the run


def test_a_refused_cache_drop_is_recorded_in_the_coldstart_table(run, monkeypatch):
    monkeypatch.setattr(coldstart, "drop_page_cache", lambda: False)
    coldstart.execute(run, ["mock"])
    table = pd.read_csv(run.dir / "tables" / "coldstart.csv")
    assert table.cache_drop_refused.tolist() == [True, False]


def test_a_soak_engine_that_aborts_leaves_no_files_and_borrows_nothing(
    fast_config_dir, tmp_path, synthetic_datasets, monkeypatch
):
    engines = fast_config_dir / "engines"
    bad = yaml.safe_load((engines / "mock.yaml").read_text())
    bad["name"] = "bad"
    bad["launch"] += ["--burst=4"]
    (engines / "bad.yaml").write_text(yaml.safe_dump(bad))
    run = Run.open(load_config(fast_config_dir), "soak-bad", results_dir=tmp_path / "results")
    probe.execute(run, ["mock"])
    capacity = json.loads(probe.capacity_path(run).read_text())
    capacity["bad"] = capacity["mock"]
    probe.capacity_path(run).write_text(json.dumps(capacity))
    run.cfg.suite["soak"].update(duration_h=8 / 3600, engines=["bad", "mock"], sample_s=1)
    monkeypatch.setattr(soak, "CHUNK_S", 2.0)
    monkeypatch.setattr(soak, "THROUGHPUT_WINDOW_S", 2.0)
    monkeypatch.setattr(soak, "LATENCY_WINDOW_S", 4.0)
    soak.execute(run, ["bad", "mock"])
    assert run.aborted("soak") == ["bad"]
    assert list(pd.read_csv(run.dir / "tables" / "soak.csv").engine) == ["mock"]
    assert not (run.dir / "soak" / "bad" / "verdict.json").exists()
