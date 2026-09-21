"""Correctness metrics and gates. Pure functions over per-prompt token and logprob lists."""

from __future__ import annotations

import numpy as np

from .request_metrics import percentile


def token_metrics(engine: list[dict], scored: list[dict], confident_gap: float) -> dict:
    """Engine tokens judged by the reference over the same prefixes. `engine` records carry
    `token_ids` and `logprobs`; `scored` records the reference's `ref_logprob`, `ref_argmax`,
    `ref_top1` and `ref_top2`. Both are matched by `id`."""
    ref = {r["id"]: r for r in scored}
    diffs, agree, confident, count = [], 0, 0, 0
    for record in engine:
        r = ref[record["id"]]
        tokens = np.asarray(record["token_ids"])
        diffs.append(np.abs(np.asarray(record["logprobs"]) - np.asarray(r["ref_logprob"])))
        differs = np.asarray(r["ref_argmax"]) != tokens
        gap = np.asarray(r["ref_top1"]) - np.asarray(r["ref_top2"])
        agree += int((~differs).sum())
        confident += int((differs & (gap > confident_gap)).sum())
        count += len(tokens)
    flat = np.concatenate(diffs)
    return {
        "abs_dlogprob_mean": float(flat.mean()), "abs_dlogprob_p50": percentile(flat, 50),
        "abs_dlogprob_p99": percentile(flat, 99), "abs_dlogprob_max": float(flat.max()),
        "tf_top1_agree": agree / count, "confident_mismatch_rate": confident / count,
        "positions": count,
    }


def first_divergence(a: list[int], b: list[int]) -> int:
    """Index of the first differing token; the shorter length when one is a prefix of the other."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))


def free_run_divergence(engine: list[dict], reference: list[dict]) -> dict:
    """Free-running greedy of the engine against the reference's: where each prompt first
    differs, and the share that never does."""
    ref = {r["id"]: r["token_ids"] for r in reference}
    firsts = [first_divergence(r["token_ids"], ref[r["id"]]) for r in engine]
    identical = [f == len(r["token_ids"]) == len(ref[r["id"]]) for f, r in zip(firsts, engine,
                                                                            strict=True)]
    return {"first_divergence_median": float(np.median(firsts)),
            "no_divergence_fraction": float(np.mean(identical))}


def batch_invariance(a1: list[list[int]], a2: list[list[int]], b: list[list[int]],
                     c: list[list[int]]) -> dict:
    """Prompts run alone twice (A, A2), amid filler (B) and amid filler shuffled (C): whether
    the tokens are the same regardless of the batch they shared."""
    def identical(x, y) -> float:
        return float(np.mean([p == q for p, q in zip(x, y, strict=True)]))

    def mean_divergence(x, y) -> float | None:
        firsts = [first_divergence(p, q) for p, q in zip(x, y, strict=True) if p != q]
        return float(np.mean(firsts)) if firsts else None

    return {"run_to_run_identical_fraction": identical(a1, a2),
            "identical_fraction_vs_batched": identical(a1, b),
            "identical_fraction_vs_shuffled": identical(a1, c),
            "mean_first_divergence_vs_batched": mean_divergence(a1, b),
            "mean_first_divergence_vs_shuffled": mean_divergence(a1, c)}


def evaluate_gates(summary: dict, gates: dict) -> dict[str, dict]:
    """The four gates on the engine named `ours`. Each is `passed` True or False, or None with
    a reason when its inputs were not measured."""
    ours = summary.get("ours", {})
    baselines = [s for name, s in summary.items() if name != "ours" and "abs_dlogprob_mean" in s]

    def gate(passed: bool | None, detail: str) -> dict:
        return {"passed": passed, "detail": detail}

    out = {}
    rate = ours.get("confident_mismatch_rate")
    limit = gates["confident_mismatch_rate_max"]
    out["confident_mismatch_rate"] = (
        gate(rate <= limit, f"{rate:.5f} <= {limit}") if rate is not None
        else gate(None, "not evaluated: no teacher-forced scoring of ours"))
    mean = ours.get("abs_dlogprob_mean")
    if mean is None or not baselines:
        out["abs_dlogprob_vs_baselines"] = gate(None, "not evaluated: needs ours and a baseline")
    else:
        median = float(np.median([b["abs_dlogprob_mean"] for b in baselines]))
        ratio = gates["mean_abs_dlogprob_vs_baseline_median_max_ratio"]
        out["abs_dlogprob_vs_baselines"] = gate(
            mean <= ratio * median, f"{mean:.5f} <= {ratio} x median baseline {median:.5f}")
    ppl, ref_ppl = ours.get("ppl"), ours.get("ref_ppl")
    ratio = gates["ppl_vs_ref_max_ratio"]
    out["ppl_vs_reference"] = (
        gate(ppl <= ratio * ref_ppl, f"{ppl:.4f} <= {ratio} x reference {ref_ppl:.4f}")
        if ppl is not None and ref_ppl is not None
        else gate(None, "not evaluated: no perplexity for ours or the reference"))
    ours_gsm, vllm_gsm = ours.get("gsm8k_flexible"), summary.get("vllm", {}).get("gsm8k_flexible")
    delta = gates["gsm8k_flexible_vs_vllm_min_delta_points"]
    out["gsm8k_vs_vllm"] = (
        gate((ours_gsm - vllm_gsm) * 100 >= delta,
             f"{(ours_gsm - vllm_gsm) * 100:+.2f} points (need >= {delta})")
        if ours_gsm is not None and vllm_gsm is not None
        else gate(None, "not evaluated: needs GSM8K for ours and vllm"))
    return out


def gates_verdict(gates: dict[str, dict]) -> str:
    """`pass` only when all four gates were evaluated and passed; `fail` if any failed;
    otherwise `not evaluated`. Publishing and the headline table share this rule."""
    if any(g["passed"] is False for g in gates.values()):
        return "fail"
    if len(gates) < 4 or any(g["passed"] is None for g in gates.values()):
        return "not evaluated"
    return "pass"
