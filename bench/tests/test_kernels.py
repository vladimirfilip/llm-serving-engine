import pandas as pd
import pytest

from bench.config import load_config
from bench.run import Run
from bench.suites import kernels
from bench.suites.common import SuiteSkipped

pytestmark = pytest.mark.cuda


def test_timing_reports_ordered_percentiles_of_a_known_workload():
    import torch

    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    t = kernels.time_closure(lambda: x @ x, warmup=3, iters=20, flush_mib=8)
    assert 0 < t["ms_p10"] <= t["ms_median"] <= t["ms_p90"] < 50


def test_every_contender_of_ours_matches_the_fp32_reference_and_gemms_report_their_ratio(tmp_path):
    cfg = load_config().suite["kernels"] | {
        "warmup_iters": 2, "timed_iters": 5, "decode_batches": [1, 4], "decode_contexts": [512],
        "prefill_lengths": [128, 512], "gemm_m": [1, 128]}
    summary = kernels.task_kernels("bench/tests/fixtures/model", "bfloat16", cfg, 600.0,
                                   str(tmp_path))
    assert "ours" in summary["contenders"] and summary["dropped_for_error"] == []
    decode = pd.read_csv(tmp_path / "kernels_decode.csv")
    assert set(decode.contender) == {"ours"} and (decode.max_abs_error < 2e-2).all()
    assert (decode.pct_of_bw_read > 0).all() and (decode.gb_s > 0).all()
    prefill = pd.read_csv(tmp_path / "kernels_prefill.csv")
    assert (prefill.tflop_s > 0).all() and (prefill.max_abs_error < 2e-2).all()
    gemm = pd.read_csv(tmp_path / "kernels_gemm.csv")
    assert set(gemm.contender) == {"ours", "torch"} and set(gemm["shape"]) == {
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head"}
    assert (gemm.groupby("contender").size() == 10).all()


def test_decode_cells_that_would_not_fit_are_skipped_and_named(tmp_path):
    cfg = load_config().suite["kernels"] | {
        "warmup_iters": 1, "timed_iters": 2, "decode_batches": [1, 100000],
        "decode_contexts": [512], "prefill_lengths": [128], "gemm_m": [1]}
    summary = kernels.task_kernels("bench/tests/fixtures/model", "bfloat16", cfg, None,
                                   str(tmp_path))
    assert summary["skipped_decode_cells"] == [(100000, 512)]


def test_the_suite_is_skipped_without_ours_or_without_a_gpu(fast_config, tmp_path):
    run = Run.open(fast_config, "k", results_dir=tmp_path)
    with pytest.raises(SuiteSkipped, match="ours"):
        kernels.execute(run, ["vllm"])


def test_a_contender_that_raises_is_named_with_its_reason_and_the_others_still_run(tmp_path):
    from bench.kernels import reference_kernels
    class Broken:
        name = "flashinfer"
        notes = "broken on purpose"

        def decode_attention(self, batch, ctx, spec):
            raise RuntimeError("nvcc not found")

        def prefill_attention(self, seq_len, spec):
            raise RuntimeError("nvcc not found")

        def decode_error(self, *args):
            return None

        prefill_error = decode_error

        def gemm(self, which, m, spec):
            return None

    original = reference_kernels.available_contenders
    reference_kernels.available_contenders = lambda: ([Broken()], {})
    try:
        cfg = load_config().suite["kernels"] | {
            "warmup_iters": 1, "timed_iters": 2, "decode_batches": [1], "decode_contexts": [512],
            "prefill_lengths": [128], "gemm_m": [1]}
        summary = kernels.task_kernels("bench/tests/fixtures/model", "bfloat16", cfg, None,
                                       str(tmp_path))
    finally:
        reference_kernels.available_contenders = original
    reason = summary["unavailable"]["flashinfer (failed at run time)"]
    assert "RuntimeError: nvcc not found" in reason
    assert set(pd.read_csv(tmp_path / "kernels_decode.csv").contender) == {"ours"}
