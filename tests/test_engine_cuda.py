"""The whole engine on the GPU: tiny Llama, paged KV, decode graphs and the real scheduler
threads."""

from __future__ import annotations

import asyncio

import pytest

from llm_serving_engine.config import KVCacheConfig, ModelConfig
from llm_serving_engine.engine import InferenceEngine
from llm_serving_engine.model.model_runner import ModelRunner
from llm_serving_engine.model.sampling import SamplingParams
from llm_serving_engine.observability.metrics_export import REGISTRY
from llm_serving_engine.scheduling.allocator import BlockAllocator
from tests.test_engine import make_config, read_stream

pytestmark = pytest.mark.cuda

BLOCK_SIZE = 4
PROMPTS = ["abcdef", "ghijkl", "mnopqr"]


class VocabTokenizer:
    """Characters mapped into the tiny model's 64-token vocabulary."""

    def encode_prompt(self, text: str) -> list[int]:
        return [ord(c) % 64 for c in text]


async def generate(tiny_llama_dir, num_blocks: int) -> list[list]:
    config = make_config(
        model=ModelConfig(model_name_or_path=str(tiny_llama_dir), dtype="float32"),
        kv_cache=KVCacheConfig(block_size=BLOCK_SIZE),
        max_concurrent_sequences=4,
    )
    runner = ModelRunner(config.model)
    engine = InferenceEngine(
        config, VocabTokenizer(), runner, allocator=BlockAllocator(num_blocks, BLOCK_SIZE)
    )
    assert runner._decode_graphs.graphs
    engine.bind_loop(asyncio.get_running_loop())
    engine.start()
    try:
        params = SamplingParams(temperature=0.0, max_tokens=20)
        submissions = [engine.submit(prompt, params) for prompt in PROMPTS]
        return await asyncio.gather(*(read_stream(s.output_queue) for s in submissions))
    finally:
        engine.stop()


@pytest.mark.asyncio
async def test_preemption_under_kv_pressure_changes_no_output(tiny_llama_dir):
    unconstrained = await generate(tiny_llama_dir, num_blocks=64)
    before = REGISTRY.get_sample_value("llm_preemptions_total") or 0

    # 26 tokens per request need 7 blocks each; 10 blocks forces preemption.
    constrained = await generate(tiny_llama_dir, num_blocks=10)

    assert REGISTRY.get_sample_value("llm_preemptions_total") > before
    assert constrained == unconstrained
    assert all(len(stream) == 21 for stream in constrained)  # 20 tokens + DONE
