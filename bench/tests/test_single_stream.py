import pandas as pd
import pytest

from bench.run import Run
from bench.suites import single_stream
from bench.suites.single_stream import decode_tpot, saturation_batch


def rows(**by_batch):
    return [{"batch": int(b[1:]), "out_tok_s": v} for b, v in by_batch.items()]


def test_saturation_is_the_first_batch_whose_doubling_gains_under_five_percent():
    assert saturation_batch(rows(b1=100, b2=190, b4=200, b8=205)) == 4  # 2 -> 4 gains 5.3%
    assert saturation_batch(rows(b1=100, b2=190, b4=300)) is None
    assert saturation_batch(rows(b1=100, b2=102)) == 1
    assert saturation_batch(rows(b1=100)) is None


def test_decode_tpot_skips_the_first_tokens_and_counts_the_rest():
    times = [1.0 + 0.5 * i for i in range(10)]  # first token slow to arrive, then even
    times[1:] = [2.0 + 0.01 * i for i in range(9)]
    record = {"token_times": times, "completion_tokens_usage": 10}
    assert decode_tpot(record, skip=2, itl_valid=True) == pytest.approx(0.01)
    assert decode_tpot(record, skip=2, itl_valid=False) == pytest.approx((2.08 - 1.0) / 9)


@pytest.fixture
def run(fast_config, tmp_path, synthetic_datasets):
    return Run.open(fast_config, "ss", results_dir=tmp_path / "results")


def test_the_suite_writes_decode_ttft_and_batch_tables(run):
    single_stream.execute(run, ["mock"])
    tables = run.dir / "tables"
    decode = pd.read_csv(tables / "single_decode.csv")
    assert list(decode.context) == [128, 512]
    assert decode.tpot_s.between(0.0035, 0.006).all()  # 4 ms step + 0.05 ms per sequence
    assert decode.bound_fraction.isna().all()  # no bandwidth probe on a fake engine

    ttft = pd.read_csv(tables / "single_ttft.csv")
    assert list(ttft.prompt) == [128, 512] and (ttft.ttft_s > 0).all()
    assert ttft.ttft_s.is_monotonic_increasing and (ttft.prefill_mfu > 0).all()

    batch = pd.read_csv(tables / "single_batch.csv")
    assert list(batch.batch) == [1, 2, 4]
    assert batch.out_tok_s.is_monotonic_increasing
    assert batch.out_tok_s.iloc[0] == pytest.approx(1 / 0.00405, rel=0.25)
    assert (batch.per_stream_tok_s > 100).all()


def test_contexts_past_the_model_length_are_skipped(run):
    run.cfg.suite["single_stream"]["contexts"] = [128, 16384]
    single_stream.execute(run, ["mock"])
    assert list(pd.read_csv(run.dir / "tables" / "single_decode.csv").context) == [128]
