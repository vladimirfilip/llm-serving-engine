from __future__ import annotations

from llm_serving_engine.model.sampling import SamplingParams
from llm_serving_engine.observability.metrics import RequestMetrics
from llm_serving_engine.scheduling.allocator import KVAllocator
from llm_serving_engine.scheduling.batch_plan import BatchEntry
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


def admit(seq: Sequence, n_tokens: int) -> BatchEntry:
    """scheduler_step's bookkeeping for one prefill chunk of `seq`."""
    seq.prefill_progress += n_tokens
    seq.status = "DECODING" if seq.prefill_progress == seq.num_tokens else "PREFILLING"
    return BatchEntry(seq.seq_id, n_tokens, is_prefill_chunk=True)
