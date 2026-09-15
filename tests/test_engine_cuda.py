"""The whole engine on the GPU: tiny Llama, KV pool sizing, decode graphs and the real
scheduler threads."""

from __future__ import annotations

import asyncio

import pytest

from llm_serving_engine.config import KVCacheConfig, ModelConfig
from llm_serving_engine.engine import InferenceEngine
from llm_serving_engine.model.model_runner import ModelRunner
from llm_serving_engine.model.sampling import SamplingParams
from llm_serving_engine.observability.metrics_export import REGISTRY
from llm_serving_engine.scheduling.allocator import BlockAllocator
from tests.factories import make_config, read_stream

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


def test_a_contiguous_pool_leaves_the_eager_activation_peak_free(tiny_llama_dir, monkeypatch):
    free_bytes = 64 * 1024**2
    monkeypatch.setattr("llm_serving_engine.engine._free_memory_bytes", lambda runner: free_bytes)
    config = make_config(
        model=ModelConfig(model_name_or_path=str(tiny_llama_dir), dtype="float32"),
        kv_cache=KVCacheConfig(block_size=BLOCK_SIZE),
        kv_allocator="contiguous",
        max_concurrent_sequences=4,
    )
    runner = ModelRunner(config.model)
    reserves = []
    measure = runner.bytes_beyond_kv_pool

    def recorded_measure(*args):
        reserves.append(measure(*args))
        return reserves[-1]

    monkeypatch.setattr(runner, "bytes_beyond_kv_pool", recorded_measure)
    engine = InferenceEngine(config, VocabTokenizer(), runner)

    [reserved] = reserves
    assert reserved > 0
    capacity_tokens = config.kv_cache.num_blocks(free_bytes - reserved) * BLOCK_SIZE
    assert engine.allocator.capacity_tokens == capacity_tokens


# Runs in a subprocess: the device-side assert it triggers poisons CUDA for the whole process.
_DEVICE_ASSERT_SCRIPT = """
import asyncio, sys, time
from llm_serving_engine.config import KVCacheConfig, ModelConfig
from llm_serving_engine.engine import IngressRequest, InferenceEngine
from llm_serving_engine.model.model_runner import ModelRunner
from llm_serving_engine.model.sampling import SamplingParams
from llm_serving_engine.scheduling.allocator import BlockAllocator
from llm_serving_engine.scheduling.dispatch import ABORTED, new_output_channel
from tests.factories import make_config

async def main():
    config = make_config(model=ModelConfig(model_name_or_path=sys.argv[1], dtype="float32"))
    runner = ModelRunner(config.model)
    engine = InferenceEngine(config, None, runner, allocator=BlockAllocator(16, 4))
    engine.bind_loop(asyncio.get_running_loop())
    engine.start()
    # Past submit's vocabulary check, as a tokenizer/model mismatch would be.
    seq_id, q = new_output_channel(maxsize=8)
    engine._submitted += 1
    request = IngressRequest(seq_id, [runner.vocab_size], SamplingParams(), time.monotonic())
    engine.ingress.put(request)
    assert await asyncio.wait_for(q.get(), timeout=30) is ABORTED
    deadline = time.monotonic() + 10
    while engine.healthy and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    assert not engine.healthy
    print("engine unhealthy after device loss")

asyncio.run(main())
"""


def test_a_device_side_assert_fails_the_engine(tiny_llama_dir):
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-c", _DEVICE_ASSERT_SCRIPT, str(tiny_llama_dir)],
        capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert "engine unhealthy after device loss" in completed.stdout
