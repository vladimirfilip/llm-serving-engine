import pytest

from bench.engines.base import CheckFailed, Launch
from bench.run import Run
from bench.suites import checks
from bench.suites.common import engine_session, load_env


@pytest.fixture
def run(fast_config, tmp_path, synthetic_datasets):
    return Run.open(fast_config, "checks", results_dir=tmp_path / "results")


def session_for(run, *mock_args):
    return engine_session(run, "mock", "t", Launch(args_add=list(mock_args)), checks=False)


def test_a_plain_mock_passes_every_check(run):
    with session_for(run) as s:
        assert checks.check_prefix_cache_off(s).passed
        assert checks.check_no_bursts(s).passed
        assert checks.check_token_accounting(s).passed
        assert checks.check_tokenizer_identity(s).skipped


def test_the_prefix_cache_check_fails_on_a_mock_with_a_prefix_cache(run):
    with session_for(run, "--prefix-cache=true") as s:
        result = checks.check_prefix_cache_off(s)
    assert not result.passed and "TTFT" in result.detail


def test_the_burst_check_fails_on_a_mock_that_releases_tokens_in_bursts(run):
    for burst in (2, 4):
        with session_for(run, f"--burst={burst}") as s:
            assert not checks.check_no_bursts(s).passed, burst


def test_bundled_events_clear_itl_valid_but_the_run_survives(run):
    """Token accounting is not a MUST: an engine that packs several tokens into each event is
    still measured, with ITL percentiles withheld."""
    with session_for(run, "--bundle-events=4") as s:
        accounting = checks.check_token_accounting(s)
        assert not accounting.passed and not accounting.must
        assert checks.check_no_bursts(s).passed  # bundled events still arrive evenly
    with engine_session(run, "mock", "t2", Launch(args_add=["--bundle-events=4"])) as s:
        assert not s.adapter.itl_valid
    assert checks.load_checks(run)["mock"]["itl_valid"] is False


def test_a_busy_gpu_aborts_the_engine_and_the_result_is_recorded(run, monkeypatch):
    from bench import env

    monkeypatch.setattr(env, "gpu_available", lambda: True)
    monkeypatch.setattr(env, "compute_pids", lambda: [4242])
    monkeypatch.setattr(env, "gpu_utilization_pct", lambda _index: 60)
    monkeypatch.setattr(checks, "IDLE_SETTLE_S", 0.0)  # read when the check runs, not at import
    with pytest.raises(CheckFailed, match="GPU not idle"):
        checks.require_idle_gpu(run, "ours")
    saved = checks.load_checks(run)["ours"]["gpu_idle"]
    assert not saved["passed"] and "4242" in saved["detail"]

    monkeypatch.setattr(env, "compute_pids", lambda: [])
    monkeypatch.setattr(env, "gpu_utilization_pct", lambda _index: 4)
    checks.require_idle_gpu(run, "ours")
    assert checks.load_checks(run)["ours"]["gpu_idle"]["passed"]
    monkeypatch.setattr(env, "gpu_utilization_pct", lambda _index: 5)
    with pytest.raises(CheckFailed):
        checks.require_idle_gpu(run, "ours")


def test_a_failed_must_check_aborts_the_engine_and_is_cached(run):
    with pytest.raises(CheckFailed, match="no_bursts"):
        with engine_session(run, "mock", "t", Launch(args_add=["--burst=4"])):
            pass
    saved = checks.load_checks(run)["mock"]
    assert not saved["itl_valid"]
    assert [r["name"] for r in saved["results"] if not r["passed"]] == ["no_bursts"]


def test_passing_checks_are_run_once_and_set_itl_valid(run):
    with engine_session(run, "mock", "t") as s:
        assert s.adapter.itl_valid and s.client_procs >= 1
    first = checks.checks_path(run).read_text()
    with engine_session(run, "mock", "t2") as s:
        assert s.adapter.itl_valid
    assert checks.checks_path(run).read_text() == first


def test_the_null_server_check_passes_a_client_that_keeps_up_and_fails_one_that_cannot(fast_config):
    limits = fast_config.suite["client_check"]
    ok, detail = checks.null_server_check(limits, procs=1, cpus=None)
    assert ok, detail
    impossible = limits | {"send_lag_p99_ms": 0.001, "jitter_p99_ms": 0.001}
    assert not checks.null_server_check(impossible, procs=1, cpus=None)[0]


def test_env_json_is_written_once_with_the_resolved_fairness_rules(run):
    with engine_session(run, "mock", "t", checks=False):
        pass
    env = load_env(run)
    assert env["fairness"]["F4"]["max_num_seqs"] == 16
    assert env["model"]["spec"]["n_kv_heads"] == 8
    assert env["clocks_locked"] is False and env["gpu_prepared"] is False
    assert "bw_read_gbs" not in env


def test_a_gpu_that_goes_quiet_within_the_settle_window_passes(monkeypatch):
    from bench import env

    readings = iter([40, 20, 3])
    monkeypatch.setattr(env, "compute_pids", lambda: [])
    monkeypatch.setattr(env, "gpu_utilization_pct", lambda _index: next(readings))
    monkeypatch.setattr(checks.time, "sleep", lambda _s: None)
    assert checks.check_gpu_idle(0, settle_s=60).passed
