import copy
import shutil

import pytest
import yaml

from bench.config import CONFIG_DIR, load_config
from bench.modelspec import ModelSpec
from bench.workloads.synthetic import write_synthetic

FIXTURE_MODEL = "bench/tests/fixtures/model"


def fast_suite(suite: dict) -> dict:
    """The full suite scaled down so a mock sweep finishes in seconds."""
    s = copy.deepcopy(suite)
    sweep = s["load_sweep"]
    sweep.update(target_duration_s=2, warmup_requests=4, probe_requests=24)
    for w in sweep["per_workload"].values():
        w.update(fractions=[0.5, 1.0], repeats=1, min_requests=8, max_requests=40)
    s["workloads"]["short_long"]["output"] = 48
    s["workloads"]["sharegpt"].update(max_prompt=128, max_total=256)
    # Plumbing tests must not depend on how quiet the host is; test_checks proves the check fails.
    s["client_check"].update(streams=16, send_lag_p99_ms=250, jitter_p99_ms=250)
    sweep["rate_grid_rps_override"] = {"sharegpt": [4, 8], "short_short": [4], "long_short": [2],
                                       "short_long": [2]}
    s["single_stream"].update(contexts=[128, 512], decode_tokens=48, skip_tokens=8, repeats=2,
                              ttft_prompts=[128, 512], ttft_repeats=2, batch_sizes=[1, 2, 4],
                              batch_warmup_s=1, batch_total_s=3)
    s["memory"].update(grid_batches=[1, 2], grid_lengths=[512, 1024], cell_output_tokens=16,
                       kv_util_duration_s=4, exhaustion_batch=8, exhaustion_output_tokens=16,
                       hang_s=5, recover_s=5, cell_timeout_s=30)
    s["scheduler"]["interference"].update(background_streams=2, background_output=200,
                                          warm_s=1, inject_every_s=1, inject_for_s=3, tail_s=1,
                                          inject_prompt=512, inject_output=4)
    s["scheduler"]["overload"].update(duration_s=3, timeout_s=60)
    s["coldstart"]["launches"] = 2
    s["correctness"].update(n_prompts=6, n_long_prompts=1, long_prompt_len=600, max_new_tokens=24,
                            long_max_new_tokens=8, batch_invariance_prompts=4,
                            batch_invariance_tokens=16, batch_invariance_background=4)
    return s


@pytest.fixture
def fast_config_dir(tmp_path):
    """A copy of the configs with the mock engine running at speed and a small suite."""
    target = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, target)
    suite = yaml.safe_load((target / "suite.yaml").read_text())
    (target / "suite.yaml").write_text(yaml.safe_dump(fast_suite(suite)))
    model = yaml.safe_load((target / "model.yaml").read_text())
    model.update(local_path=str(FIXTURE_MODEL), max_model_len=8192, max_num_seqs=16)
    (target / "model.yaml").write_text(yaml.safe_dump(model))
    mock = yaml.safe_load((target / "engines" / "mock.yaml").read_text())
    mock["launch"] += ["--prefill-ms-per-1k=5", "--decode-ms=4", "--decode-ms-per-seq=0.05"]
    (target / "engines" / "mock.yaml").write_text(yaml.safe_dump(mock))
    return target


@pytest.fixture
def fast_config(fast_config_dir):
    return load_config(fast_config_dir)


@pytest.fixture
def synthetic_datasets(tmp_path, monkeypatch, fast_config):
    directory = tmp_path / "datasets"
    spec = ModelSpec.from_dir(FIXTURE_MODEL, "bfloat16")
    write_synthetic(directory, fast_config.suite, spec.vocab)
    monkeypatch.setenv("BENCH_DATASETS_DIR", str(directory))
    return directory
