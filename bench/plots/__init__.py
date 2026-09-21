"""One module per plot. Each exposes `draw(ctx) -> Path | None`; None means the suite that
feeds it produced no data, and the report lists the plot as not run."""

from __future__ import annotations

import importlib
from pathlib import Path

PLOTS = {
    "p01": ("p01_throughput_latency_sharegpt", "throughput against TPOT under load"),
    "p02": ("p02_batch1_bound_fraction", "batch-1 decode against the bandwidth bound"),
    "p03": ("p03_ablation_waterfall", "ablation of ours' optimizations"),
    "p04": ("p04_memory_breakdown", "device memory: breakdown and peaks"),
    "p05": ("p05_correctness", "logprob agreement and perplexity"),
    "p06": ("p06_goodput_sharegpt", "goodput against offered load"),
    "p07": ("p07_ttft_vs_rate_sharegpt", "time to first token against offered load"),
    "p08": ("p08_latency_percentiles_sharegpt", "latency percentiles against offered load"),
    "p09": ("p09_fixed_shapes", "tail TPOT on the fixed-shape workloads"),
    "p10": ("p10_tpot_vs_context", "decode latency against context length"),
    "p11": ("p11_ttft_vs_prompt", "prefill latency and MFU against prompt length"),
    "p12": ("p12_throughput_vs_batch", "closed-loop throughput against batch size"),
    "p13": ("p13_roofline", "roofline of ours' kernels"),
    "p14": ("p14_attn_decode", "decode attention bandwidth"),
    "p15": ("p15_attn_prefill", "prefill attention throughput"),
    "p16": ("p16_gemm", "projection GEMMs against torch"),
    "p17": ("p17_gpu_busy", "GPU busy fraction and idle gaps"),
    "p18": ("p18_ours_step_breakdown", "ours' step time breakdown"),
    "p19": ("p19_capacity_heatmap", "batch and length the engine holds"),
    "p20": ("p20_kv_util", "KV utilisation and fragmentation"),
    "p21": ("p21_prefill_interference", "decode stalls during a long prefill"),
    "p22": ("p22_overload_ttft", "who waits under overload"),
    "p23": ("p23_energy", "output tokens per joule"),
    "p24": ("p24_soak", "long-run stability"),
    "p25": ("p25_coldstart", "cold start and first request"),
}


def draw_all(ctx) -> dict[str, Path | None]:
    """Draw every plot that has data; the value is the PNG path, or None if not run."""
    out = {}
    for key, (module, _title) in PLOTS.items():
        out[key] = importlib.import_module(f"{__package__}.{module}").draw(ctx)
    return out
