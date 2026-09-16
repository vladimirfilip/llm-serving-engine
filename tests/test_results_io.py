import csv
import json

from llm_serving_engine.loadgen.report import build_point, build_report
from llm_serving_engine.loadgen.results_io import write_pooled_latency, write_run


def _result(**fields) -> dict:
    return {
        "scheduled_at": 0.0, "shape": "chat", "success": True, "latency": 0.5,
        "first_token_latency": 0.1, "completed_at": 0.5, "output_tokens": 3,
        "prompt_tokens": 10, **fields,
    }


def test_fractional_qps_stems_each_keep_their_own_files(tmp_path):
    for qps in (0.03, 0.05):
        results = [_result(latency=qps)]
        stem = tmp_path / f"qps_{qps:g}"
        write_run(stem, {"target_qps": qps, "results": results}, build_report(results))

    for qps in (0.03, 0.05):
        run = json.loads((tmp_path / f"qps_{qps:g}.json").read_text())
        assert run["target_qps"] == qps
        summary = json.loads((tmp_path / f"qps_{qps:g}_summary.json").read_text())
        assert summary["e2e_latency_max"] == qps
        assert (tmp_path / f"qps_{qps:g}.csv").exists()


def test_raw_csv_has_one_row_per_request_and_summary_carries_run_fields(tmp_path):
    results = [_result(), _result(success=False, error="ReadTimeout: ")]
    write_run(tmp_path / "run", {"results": results}, build_report(results), target_qps=2.0)

    with (tmp_path / "run.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [row["error"] for row in rows] == ["", "ReadTimeout: "]
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["target_qps"] == 2.0
    assert summary["failures"] == 1


def test_pooled_latency_summary_counts_every_repeat(tmp_path):
    point = build_point([[_result()], [_result(), _result()]])
    write_pooled_latency(tmp_path / "qps_0.5", point.latency, target_qps=0.5, repeats=2)
    pooled = json.loads((tmp_path / "qps_0.5_pooled_summary.json").read_text())
    assert pooled["ttft_chat_count"] == 3
    assert pooled["repeats"] == 2
