"""Graphed decode against eager forward_fused on the same state: padding, rows changing
occupant between replays, stale block tables after a preemption, and dispatch."""

from __future__ import annotations

import copy

import pytest
import torch

from llm_serving_engine.model.decode_graph import decode_graph_buckets
from llm_serving_engine.model.model_runner import ModelRunner
from llm_serving_engine.scheduling.allocator import BlockAllocator
from llm_serving_engine.scheduling.batch_plan import BatchEntry, BatchPlan
from tests.factories import admit
from tests.test_fused_forward import greedy, load_runner

pytestmark = pytest.mark.cuda

NUM_BLOCKS, BLOCK_SIZE = 64, 4
SENTINEL = 12345.0


@pytest.fixture(scope="module")
def graphed_runner(tiny_llama_dir) -> ModelRunner:
    runner = load_runner(tiny_llama_dir)
    runner.allocate_kv_pool(NUM_BLOCKS, BLOCK_SIZE)
    runner.capture_decode_graphs(decode_graph_buckets(8))
    assert runner._decode_graphs.graphs, "capture or self-check failed"
    return runner


@pytest.fixture
def allocator() -> BlockAllocator:
    return BlockAllocator(NUM_BLOCKS, BLOCK_SIZE)


def prefilled(runner: ModelRunner, allocator: BlockAllocator, seq_id: int, prompt_len: int):
    """A sequence as the scheduler leaves it for its first decode: prompt prefilled, first
    token generated, a slot allocated for the next."""
    seq = greedy(seq_id, list(range(3, 3 + prompt_len)))
    assert allocator.allocate(seq, prompt_len)
    plan = BatchPlan(entries=[admit(seq, prompt_len)])
    [(_, token, _)] = runner.forward_fused(plan, {seq_id: seq})
    seq.generated_tokens.append(token)
    assert allocator.allocate(seq, 1)
    return seq


def decode_plan(seqs) -> BatchPlan:
    return BatchPlan(entries=[BatchEntry(seq.seq_id, 1, is_prefill_chunk=False) for seq in seqs])


def assert_graphed_matches_eager(runner: ModelRunner, seqs: list) -> list:
    plan = decode_plan(seqs)
    by_id = {seq.seq_id: seq for seq in seqs}
    eager = runner.forward_fused(plan, copy.deepcopy(by_id))
    graphed = runner.forward(plan, by_id)
    assert graphed == eager
    return graphed


def advance(allocator: BlockAllocator, seqs: list, results: list) -> None:
    by_id = {seq.seq_id: seq for seq in seqs}
    for seq_id, token, _finished in results:
        by_id[seq_id].generated_tokens.append(token)
        assert allocator.allocate(by_id[seq_id], 1)


def poison_blocks(runner: ModelRunner, block_ids: list[int]) -> None:
    for layer in range(len(runner._k_pool)):
        runner._k_pool[layer][block_ids] = SENTINEL
        runner._v_pool[layer][block_ids] = SENTINEL


def blocks_untouched(runner: ModelRunner, block_ids: list[int]) -> bool:
    return all(
        bool(torch.all(runner._k_pool[layer][block_ids] == SENTINEL))
        and bool(torch.all(runner._v_pool[layer][block_ids] == SENTINEL))
        for layer in range(len(runner._k_pool))
    )


def test_batch_exactly_filling_a_bucket_matches_eager(graphed_runner, allocator):
    seqs = [prefilled(graphed_runner, allocator, 400 + i, n) for i, n in enumerate([3, 5, 2, 4])]
    assert_graphed_matches_eager(graphed_runner, seqs)


def test_padded_batch_matches_eager(graphed_runner, allocator):
    seqs = [prefilled(graphed_runner, allocator, 410 + i, n) for i, n in enumerate([6, 2, 9])]
    assert_graphed_matches_eager(graphed_runner, seqs)


def test_padding_rows_write_only_to_the_scratch_block(graphed_runner, allocator):
    seq = prefilled(graphed_runner, allocator, 420, 3)
    unowned = [50]
    poison_blocks(graphed_runner, unowned)

    graphed_runner.forward(decode_plan([seq]), {seq.seq_id: seq})  # 1 real row, 7 padding

    assert blocks_untouched(graphed_runner, unowned)


def test_rows_that_change_occupant_between_replays_match_eager(graphed_runner, allocator):
    a, b, c, e = (
        prefilled(graphed_runner, allocator, 430 + i, n) for i, n in enumerate([4, 7, 5, 6])
    )
    results = graphed_runner.forward(decode_plan([a, b, c, e]), {s.seq_id: s for s in (a, b, c, e)})
    advance(allocator, [a, b, c, e], results)

    # a and e finish; d arrives. Next replay: row 0 goes a -> d, row 3 goes e -> padding.
    finished_blocks = a.block_table.physical_blocks + e.block_table.physical_blocks
    for seq in (a, e):
        allocator.free(seq.block_table)
        graphed_runner.free(seq.seq_id)
    d = prefilled(graphed_runner, allocator, 434, 5)
    poison_blocks(graphed_runner, finished_blocks)

    assert_graphed_matches_eager(graphed_runner, [d, b, c])
    assert blocks_untouched(graphed_runner, finished_blocks)


def test_a_preempted_sequence_replays_against_its_new_blocks(graphed_runner, allocator):
    seq = prefilled(graphed_runner, allocator, 440, 6)
    results = graphed_runner.forward(decode_plan([seq]), {seq.seq_id: seq})
    advance(allocator, [seq], results)

    old_blocks = list(seq.block_table.physical_blocks)
    allocator.free(seq.block_table)
    graphed_runner.free(seq.seq_id)
    seq.prefill_progress, seq.status = 0, "WAITING"
    assert allocator.allocate(seq, seq.num_tokens)
    [(_, token, _)] = graphed_runner.forward_fused(
        BatchPlan(entries=[admit(seq, seq.num_tokens)]), {seq.seq_id: seq}
    )
    seq.generated_tokens.append(token)
    assert allocator.allocate(seq, 1)
    assert set(seq.block_table.physical_blocks).isdisjoint(old_blocks)
    poison_blocks(graphed_runner, old_blocks)

    assert_graphed_matches_eager(graphed_runner, [seq])


def test_replay_reads_the_current_inputs(graphed_runner, allocator):
    seq_a = prefilled(graphed_runner, allocator, 450, 3)
    seq_b = prefilled(graphed_runner, allocator, 451, 11)
    graph = graphed_runner._decode_graphs.graphs[1]

    graphed_runner.forward(decode_plan([seq_a]), {seq_a.seq_id: seq_a})
    logits_a = graph.logits.clone()
    graphed_runner.forward(decode_plan([seq_b]), {seq_b.seq_id: seq_b})

    assert not torch.allclose(logits_a, graph.logits)


def test_a_plan_with_a_prefill_entry_runs_eagerly(graphed_runner, allocator, monkeypatch):
    decoding = prefilled(graphed_runner, allocator, 460, 3)
    prefilling = greedy(461, [1, 2, 3, 4, 5])
    assert allocator.allocate(prefilling, 5)
    plan = BatchPlan(entries=[BatchEntry(460, 1, is_prefill_chunk=False), admit(prefilling, 5)])

    def fail(*args):
        raise AssertionError("a plan with a prefill entry replayed a graph")

    monkeypatch.setattr(graphed_runner._decode_graphs, "replay", fail)
    assert len(graphed_runner.forward(plan, {460: decoding, 461: prefilling})) == 2


def test_a_decode_batch_above_the_largest_bucket_runs_eagerly(
    graphed_runner, allocator, monkeypatch
):
    seqs = [prefilled(graphed_runner, allocator, 470 + i, 2) for i in range(9)]

    def fail(*args):
        raise AssertionError("a batch above every bucket replayed a graph")

    monkeypatch.setattr(graphed_runner._decode_graphs, "replay", fail)
    assert len(graphed_runner.forward(decode_plan(seqs), {s.seq_id: s for s in seqs})) == 9


def test_a_full_concurrency_decode_batch_replays_a_graph(tiny_llama_dir, monkeypatch):
    runner = load_runner(tiny_llama_dir)
    runner.allocate_kv_pool(num_blocks=256, block_size=BLOCK_SIZE)
    runner.capture_decode_graphs(decode_graph_buckets(64))
    allocator = BlockAllocator(256, BLOCK_SIZE)
    seqs = [prefilled(runner, allocator, 500 + i, 2) for i in range(64)]

    def fail(*args):
        raise AssertionError("a 64-sequence decode ran eagerly")

    monkeypatch.setattr(runner, "forward_fused", fail)
    assert len(runner.forward(decode_plan(seqs), {s.seq_id: s for s in seqs})) == 64
