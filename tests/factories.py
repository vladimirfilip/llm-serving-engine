from __future__ import annotations

import asyncio
import time

from llm_serving_engine.config import EngineConfig, KVCacheConfig, ModelConfig, ServerConfig
from llm_serving_engine.model.model_runner import ModelRunner
from llm_serving_engine.model.sampling import SamplingParams
from llm_serving_engine.observability.metrics import RequestMetrics
from llm_serving_engine.scheduling.allocator import KVAllocator
from llm_serving_engine.scheduling.batch_plan import BatchEntry
from llm_serving_engine.scheduling.dispatch import ABORTED, DONE
from llm_serving_engine.scheduling.sequence import Sequence


def make_sequence(**overrides) -> Sequence:
    fields = dict(
        seq_id=1,
        prompt_tokens=[1, 2, 3, 4, 5],
        sampling_params=SamplingParams(),
        metrics=RequestMetrics(enqueue_time=0.0),
    )
    fields.update(overrides)
    return Sequence(**fields)


def decoding_sequence(allocator: KVAllocator, seq_id: int, prompt_len: int) -> Sequence:
    """A sequence as scheduler_step leaves it after a full prefill and one result: one
    generated token, whose KV slot the next decode step allocates."""
    seq = make_sequence(
        seq_id=seq_id, prompt_tokens=list(range(3, 3 + prompt_len)), generated_tokens=[9],
        status="DECODING", prefill_progress=prompt_len,
    )
    assert allocator.allocate(seq, prompt_len)
    return seq


def prefilling_sequence(
    allocator: KVAllocator, seq_id: int, prompt_len: int, progress: int
) -> Sequence:
    """A sequence mid-prefill, holding blocks for the tokens it has already chunked."""
    seq = make_sequence(
        seq_id=seq_id, prompt_tokens=[0] * prompt_len, status="PREFILLING",
        prefill_progress=progress,
    )
    assert allocator.allocate(seq, progress)
    return seq


def admit(seq: Sequence, n_tokens: int) -> BatchEntry:
    """scheduler_step's bookkeeping for one prefill chunk of `seq`."""
    seq.prefill_progress += n_tokens
    seq.status = "DECODING" if seq.prefill_progress == seq.num_tokens else "PREFILLING"
    return BatchEntry(seq.seq_id, n_tokens, is_prefill_chunk=True)


def greedy(seq_id: int, prompt: list[int]):
    params = SamplingParams(temperature=0.0, max_tokens=8)
    return make_sequence(seq_id=seq_id, prompt_tokens=prompt, sampling_params=params)


def load_runner(path, **overrides) -> ModelRunner:
    fields = dict(model_name_or_path=str(path), device="cuda", dtype="float32")
    fields.update(overrides)
    return ModelRunner(ModelConfig(**fields))


def make_config(**overrides) -> EngineConfig:
    model = overrides.pop("model", ModelConfig())
    kv_cache = overrides.pop("kv_cache", KVCacheConfig())
    server = overrides.pop("server", ServerConfig())
    return EngineConfig(model=model, kv_cache=kv_cache, server=server, **overrides)


TOKEN = 7


class FakeTokenizer:
    def encode_prompt(self, text: str) -> list[int]:
        return [ord(c) for c in text]


class FakeModelRunner:
    """No weights: every sequence that owes a token gets TOKEN, finishing at max_tokens."""

    device = "cpu"
    vocab_size = 256

    def __init__(self):
        self.setup_calls: list[tuple] = []  # in call order
        self.freed: list[int] = []
        self.fail_next_forward = False
        self.device_lost = False

    def device_usable(self):
        return not self.device_lost

    def forward(self, plan, seqs):
        if self.fail_next_forward:
            self.fail_next_forward = False
            raise RuntimeError("injected forward failure")
        results = []
        for entry in plan:
            seq = seqs[entry.seq_id]
            if entry.is_prefill_chunk and seq.status == "PREFILLING":
                continue
            finished = len(seq.generated_tokens) + 1 >= seq.sampling_params.max_tokens
            results.append((seq.seq_id, TOKEN, finished))
        return results

    def allocate_kv_pool(self, num_blocks, block_size):
        self.setup_calls.append(("allocate_kv_pool", num_blocks, block_size))

    def capture_decode_graphs(self, bucket_sizes):
        self.setup_calls.append(("capture_decode_graphs", bucket_sizes))

    def capture_piecewise_graphs(self, bucket_sizes):
        self.setup_calls.append(("capture_piecewise_graphs", bucket_sizes))

    def free(self, seq_id):
        self.freed.append(seq_id)

    def memory_bytes(self):
        return {"weights": 10, "kv_cache": 20, "activations": 3, "workspace": 0,
                "cuda_graph_pool": 4, "other": 1}


async def read_stream(q: asyncio.Queue) -> list:
    items = []
    while not items or items[-1] not in (DONE, ABORTED):
        items.append(await asyncio.wait_for(q.get(), timeout=5))
    return items


def wait_until(condition, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition never held"
        time.sleep(0.005)
