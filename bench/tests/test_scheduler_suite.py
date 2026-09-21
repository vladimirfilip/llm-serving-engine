import math

import pandas as pd
import pytest
import yaml

from bench.config import load_config
from bench.run import Run
from bench.suites import probe, scheduler
from bench.suites.scheduler import (
    during_prefill,
    label_gaps,
    overload_metrics,
    summarize_interference,
    windowed_tpot,
)
from bench.tests.factories import record


def test_a_gap_counts_as_during_prefill_only_if_it_overlaps_an_injected_request():
    injected = [(10.0, 11.0)]
    assert during_prefill(9.5, 10.5, injected)      # starts before, ends inside
    assert during_prefill(10.2, 10.4, injected)     # inside
    assert during_prefill(10.9, 12.0, injected)     # starts inside, ends after
    assert during_prefill(9.0, 12.0, injected)      # spans it
    assert not during_prefill(9.0, 10.0, injected)  # ends exactly when it starts
    assert not during_prefill(11.0, 12.0, injected)
    assert not during_prefill(1.0, 2.0, [])


def test_gaps_are_labelled_per_stream_and_summarised_per_label():
    stream = {"token_times": [0.0, 0.1, 0.2, 5.2, 5.3]}  # one 5 s stall while a prefill runs
    injected = [(0.15, 5.1)]
    gaps = label_gaps([stream], injected)
    assert gaps.label.tolist() == ["baseline", "during_prefill", "during_prefill", "baseline"]
    rows = {r["label"]: r for r in summarize_interference(
        gaps, [record(0, 0.15, ttft=4.95, n_tokens=2)])}
    assert rows["during_prefill"]["itl_max"] == pytest.approx(5.0)
    assert rows["baseline"]["itl_p50"] == pytest.approx(0.1)
    assert rows["baseline"]["n"] == 2 and rows["during_prefill"]["n"] == 2
    assert rows["baseline"]["inject_ttft_p50"] == pytest.approx(4.95)


def test_windowed_tpot_stands_in_for_itl_when_events_are_not_per_token():
    stream = {"token_times": [i * 0.1 for i in range(45)]}  # 10 tokens/s for 4.4 s
    frame = windowed_tpot([stream], 2.0)
    assert len(frame) == 3 and frame.itl.tolist() == pytest.approx([0.1, 0.1, 0.1])


def test_overload_metrics_follow_their_definitions():
    records = [record(0, 0.0, ttft=1.0, prompt_len=10), record(1, 1.0, ttft=2.0, prompt_len=20),
               record(2, 2.0, ttft=40.0, prompt_len=30), record(3, 3.0, ttft=50.0, prompt_len=40),
               record(4, 4.0, status="timeout", prompt_len=50)]
    for r in records:
        r["token_times"] = r["token_times"] or []
    m = overload_metrics(records, starvation_ttft_s=30)
    assert m["n"] == 5 and m["timeout_rate"] == pytest.approx(0.2)
    assert m["starved_fraction"] == pytest.approx(2 / 5)  # two of five waited over 30 s
    assert m["ttft_p50"] == pytest.approx(21.0) and m["ttft_max"] == pytest.approx(50.0)
    assert m["ttft_p99_over_p50"] == pytest.approx(m["ttft_p99"] / 21.0)
    assert m["spearman_prompt_len_ttft"] == pytest.approx(1.0)  # longer prompts waited longer
    flat = overload_metrics([record(i, float(i), ttft=1.0, prompt_len=10) for i in range(4)], 30)
    assert math.isnan(flat["spearman_prompt_len_ttft"]) or flat["starved_fraction"] == 0


@pytest.fixture
def run(fast_config_dir, tmp_path, synthetic_datasets):
    engines = fast_config_dir / "engines" / "mock.yaml"
    spec = yaml.safe_load(engines.read_text())
    spec["variants"] = {"chunked_prefill_off": {"args_add": ["--chunk-tokens=0"]}}
    spec["launch"] += ["--chunk-tokens=512", "--prefill-ms-per-1k=200"]
    engines.write_text(yaml.safe_dump(spec))
    suite = yaml.safe_load((fast_config_dir / "suite.yaml").read_text())
    suite["scheduler"]["interference"]["inject_prompt"] = 2048
    (fast_config_dir / "suite.yaml").write_text(yaml.safe_dump(suite))
    return Run.open(load_config(fast_config_dir), "sch", results_dir=tmp_path / "results")


def test_variants_are_the_tuned_default_plus_each_the_engine_defines(run):
    assert list(scheduler.variants_of(run, "mock")) == ["default", "chunked_prefill_off"]
    off = scheduler.variants_of(run, "mock")["chunked_prefill_off"]
    assert off.args_add == ["--chunk-tokens=0"]


def test_the_suite_measures_interference_per_variant_and_overload_fairness(run):
    probe.execute(run, ["mock"])
    scheduler.execute(run, ["mock"])
    tables = run.dir / "tables"
    interference = pd.read_csv(tables / "scheduler_interference.csv")
    assert set(interference.variant) == {"default", "chunked_prefill_off"}
    assert set(interference.label) == {"baseline", "during_prefill"}
    stall = interference[(interference.label == "during_prefill")].set_index("variant").itl_max
    # unchunked, one 2048-token prefill (about 410 ms) stalls every decoding stream at once
    assert stall["chunked_prefill_off"] > stall["default"]
    overload = pd.read_csv(tables / "scheduler_overload.csv").iloc[0]
    assert overload.n > 0 and 0 <= overload.timeout_rate <= 1
    assert (run.dir / "scheduler" / "mock" / "interference_default.parquet").exists()
