"""Piecewise-graph iterations against eager forward_fused on the same state: mixed decode and
prefill plans, chunked prefill, padding, and dispatch."""

from __future__ import annotations

import copy
from itertools import pairwise

import pytest

from llm_serving_engine.model.model_runner import ModelRunner
from llm_serving_engine.model.piecewise_graph import piecewise_graph_buckets
from llm_serving_engine.scheduling.allocator import BlockAllocator
from llm_serving_engine.scheduling.batch_plan import BatchEntry, BatchPlan
from tests.factories import admit, greedy, load_runner
from tests.test_decode_graph import blocks_untouched, poison_blocks

pytestmark = pytest.mark.cuda

NUM_BLOCKS, BLOCK_SIZE = 64, 4


@pytest.fixture(scope="module")
def piecewise_runner(tiny_llama_dir) -> ModelRunner:
    runner = load_runner(tiny_llama_dir)
    runner.allocate_kv_pool(NUM_BLOCKS, BLOCK_SIZE)
    runner.capture_piecewise_graphs(piecewise_graph_buckets(48))
    assert set(runner._piecewise_graphs.graphs) == {16, 24, 32, 48}, "capture or self-check failed"
    return runner


@pytest.fixture
def allocator() -> BlockAllocator:
    return BlockAllocator(NUM_BLOCKS, BLOCK_SIZE)


def prefilled(runner: ModelRunner, allocator: BlockAllocator, seq_id: int, prompt_len: int):
    seq = greedy(seq_id, list(range(3, 3 + prompt_len)))
    assert allocator.allocate(seq, prompt_len)
    plan = BatchPlan(entries=[admit(seq, prompt_len)])
    [(_, token, _)] = runner.forward_fused(plan, {seq_id: seq})
    seq.generated_tokens.append(token)
    assert allocator.allocate(seq, 1)
    return seq


def assert_piecewise_matches_eager(runner: ModelRunner, plan: BatchPlan, seqs: dict, monkeypatch):
    eager_seqs = copy.deepcopy(seqs)
    eager = runner.forward_fused(plan, eager_seqs)

    def fail(*args):
        raise AssertionError("the plan ran eagerly")

    with monkeypatch.context() as patch:
        patch.setattr(runner, "forward_fused", fail)
        piecewise = runner.forward(plan, seqs)
    assert piecewise == eager
    return piecewise


def test_buckets_pad_by_under_1_5x_and_end_at_the_maximum():
    buckets = piecewise_graph_buckets(4160)
    assert buckets[:6] == [16, 24, 32, 48, 64, 96]
    assert buckets[-1] == 4160
    assert all(b / a <= 1.5 for a, b in pairwise(buckets))


def test_decodes_beside_a_prefill_chunk_match_eager(piecewise_runner, allocator, monkeypatch):
    decoders = [prefilled(piecewise_runner, allocator, 600 + i, n) for i, n in enumerate([3, 7, 5])]
    newcomer = greedy(610, list(range(20, 34)))  # 14 tokens; 17 in total pads to 24
    assert allocator.allocate(newcomer, 14)
    plan = BatchPlan(
        entries=[BatchEntry(s.seq_id, 1, is_prefill_chunk=False) for s in decoders]
        + [admit(newcomer, 14)]
    )
    seqs = {s.seq_id: s for s in [*decoders, newcomer]}

    results = assert_piecewise_matches_eager(piecewise_runner, plan, seqs, monkeypatch)

    assert len(results) == 4


def test_a_chunked_prefill_continues_through_piecewise_graphs(
    piecewise_runner, allocator, monkeypatch
):
    seq = greedy(620, list(range(5, 35)))  # 30 tokens in chunks of 20 and 10
    seqs = {seq.seq_id: seq}
    assert allocator.allocate(seq, 20)
    assert assert_piecewise_matches_eager(
        piecewise_runner, BatchPlan(entries=[admit(seq, 20)]), seqs, monkeypatch
    ) == []  # a partial chunk samples nothing

    assert allocator.allocate(seq, 10)
    [result] = assert_piecewise_matches_eager(
        piecewise_runner, BatchPlan(entries=[admit(seq, 10)]), seqs, monkeypatch
    )
    assert result[0] == seq.seq_id


def test_padding_positions_write_only_to_the_scratch_block(piecewise_runner, allocator):
    seq = greedy(630, list(range(3, 20)))  # 17 tokens, bucket 24: 7 padding positions
    assert allocator.allocate(seq, 17)
    unowned = [50]
    poison_blocks(piecewise_runner, unowned)

    piecewise_runner.forward(BatchPlan(entries=[admit(seq, 17)]), {seq.seq_id: seq})

    assert blocks_untouched(piecewise_runner, unowned)


def test_a_plan_above_the_largest_bucket_runs_eagerly(piecewise_runner, allocator, monkeypatch):
    seq = greedy(640, list(range(3, 52)))  # 49 tokens
    assert allocator.allocate(seq, 49)
    ran_eagerly = []
    real_forward_fused = piecewise_runner.forward_fused

    def record(plan, seqs):
        ran_eagerly.append(True)
        return real_forward_fused(plan, seqs)

    monkeypatch.setattr(piecewise_runner, "forward_fused", record)
    piecewise_runner.forward(BatchPlan(entries=[admit(seq, 49)]), {seq.seq_id: seq})

    assert ran_eagerly
