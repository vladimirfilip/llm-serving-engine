import pytest

from bench.metrics.correctness import (
    batch_invariance,
    evaluate_gates,
    first_divergence,
    free_run_divergence,
    token_metrics,
)

GATES = {"confident_mismatch_rate_max": 0.001,
         "mean_abs_dlogprob_vs_baseline_median_max_ratio": 1.5,
         "ppl_vs_ref_max_ratio": 1.005, "gsm8k_flexible_vs_vllm_min_delta_points": -2.0}


def test_token_metrics_match_hand_computed_values():
    engine = [{"id": "a", "token_ids": [5, 6, 7], "logprobs": [-1.0, -2.0, -3.0]},
              {"id": "b", "token_ids": [8], "logprobs": [-0.5]}]
    scored = [
        # a: position 1 disagrees, and the reference is confident there (gap 2.5 > 1)
        {"id": "a", "ref_logprob": [-1.5, -2.0, -2.5], "ref_argmax": [5, 9, 7],
         "ref_top1": [-1.5, -0.5, -2.5], "ref_top2": [-3.0, -3.0, -4.0]},
        # b: disagrees but the reference is unsure (gap 0.2 < 1), so it is not confident
        {"id": "b", "ref_logprob": [-0.5], "ref_argmax": [2], "ref_top1": [-0.4],
         "ref_top2": [-0.6]},
    ]
    m = token_metrics(engine, scored, confident_gap=1.0)
    assert m["positions"] == 4
    assert m["abs_dlogprob_mean"] == pytest.approx((0.5 + 0.0 + 0.5 + 0.0) / 4)
    assert m["abs_dlogprob_max"] == 0.5 and m["abs_dlogprob_p50"] == pytest.approx(0.25)
    assert m["tf_top1_agree"] == pytest.approx(2 / 4)
    assert m["confident_mismatch_rate"] == pytest.approx(1 / 4)


def test_first_divergence_is_the_first_differing_index_or_the_shorter_length():
    assert first_divergence([1, 2, 3], [1, 2, 4]) == 2
    assert first_divergence([1, 2, 3], [1, 2, 3]) == 3
    assert first_divergence([1, 2], [1, 2, 3]) == 2
    assert first_divergence([9], [1]) == 0


def test_free_run_divergence_reports_the_median_index_and_the_identical_share():
    engine = [{"id": "a", "token_ids": [1, 2, 3, 4]}, {"id": "b", "token_ids": [1, 2, 3, 4]},
              {"id": "c", "token_ids": [1, 9, 3, 4]}]
    reference = [{"id": "a", "token_ids": [1, 2, 3, 4]}, {"id": "b", "token_ids": [1, 2, 7, 4]},
                 {"id": "c", "token_ids": [1, 2, 3, 4]}]
    out = free_run_divergence(engine, reference)
    assert out == {"first_divergence_median": 2.0, "no_divergence_fraction": pytest.approx(1 / 3)}


def test_batch_invariance_separates_run_to_run_noise_from_batch_dependence():
    a1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 1, 1]]
    a2 = [[1, 2, 3], [4, 5, 6], [7, 8, 0], [1, 1, 1]]  # one prompt differs run to run
    b = [[1, 2, 3], [4, 0, 6], [7, 8, 9], [1, 1, 1]]  # one differs at index 1
    c = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 1]]  # one differs at index 0
    out = batch_invariance(a1, a2, b, c)
    assert out["run_to_run_identical_fraction"] == 0.75
    assert out["identical_fraction_vs_batched"] == 0.75
    assert out["identical_fraction_vs_shuffled"] == 0.75
    assert out["mean_first_divergence_vs_batched"] == 1.0
    assert out["mean_first_divergence_vs_shuffled"] == 0.0
    same = batch_invariance(a1, a1, a1, a1)
    assert same["mean_first_divergence_vs_batched"] is None


def summary(**ours):
    base = {"abs_dlogprob_mean": 0.01, "gsm8k_flexible": 0.60}
    return {"ours": ours, "vllm": base, "sglang": {"abs_dlogprob_mean": 0.03}}


def test_every_gate_passes_when_ours_is_within_its_limits():
    s = summary(confident_mismatch_rate=0.0005, abs_dlogprob_mean=0.02, ppl=7.02, ref_ppl=7.0,
                gsm8k_flexible=0.59)
    gates = evaluate_gates(s, GATES)
    assert {name: g["passed"] for name, g in gates.items()} == {
        "confident_mismatch_rate": True, "abs_dlogprob_vs_baselines": True,
        "ppl_vs_reference": True, "gsm8k_vs_vllm": True}


def test_each_gate_fails_on_its_own_violation():
    good = dict(confident_mismatch_rate=0.0005, abs_dlogprob_mean=0.02, ppl=7.02, ref_ppl=7.0,
                gsm8k_flexible=0.59)
    failing = {
        "confident_mismatch_rate": dict(good, confident_mismatch_rate=0.002),
        "abs_dlogprob_vs_baselines": dict(good, abs_dlogprob_mean=0.031),  # 1.5 x median 0.02
        "ppl_vs_reference": dict(good, ppl=7.036),  # > 1.005 x 7.0
        "gsm8k_vs_vllm": dict(good, gsm8k_flexible=0.579),  # 2.1 points below vllm
    }
    for name, ours in failing.items():
        gates = evaluate_gates(summary(**ours), GATES)
        assert gates[name]["passed"] is False, name
        assert all(g["passed"] for other, g in gates.items() if other != name), name


def test_gates_are_not_evaluated_rather_than_passed_when_their_inputs_are_missing():
    gates = evaluate_gates({"ours": {"confident_mismatch_rate": 0.0}}, GATES)
    assert gates["confident_mismatch_rate"]["passed"] is True
    for name in ("abs_dlogprob_vs_baselines", "ppl_vs_reference", "gsm8k_vs_vllm"):
        assert gates[name]["passed"] is None and "not evaluated" in gates[name]["detail"]
    assert evaluate_gates({}, GATES)["confident_mismatch_rate"]["passed"] is None
