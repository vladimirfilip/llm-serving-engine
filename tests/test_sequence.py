from llm_serving_engine.metrics import RequestMetrics
from llm_serving_engine.sampling import SamplingParams
from llm_serving_engine.sequence import BlockTable, Sequence


def make_sequence(**overrides) -> Sequence:
    defaults = dict(
        seq_id=1,
        prompt_tokens=[1, 2, 3],
        sampling_params=SamplingParams(),
        arrival_time=0.0,
        metrics=RequestMetrics(enqueue_time=0.0),
    )
    defaults.update(overrides)
    return Sequence(**defaults)


def test_defaults():
    seq = make_sequence()
    assert seq.status == "WAITING"
    assert seq.generated_tokens == []
    assert seq.block_table.physical_blocks == []
    assert seq.prefill_progress == 0
    assert not seq.is_finished


def test_num_tokens_counts_prompt_and_generated():
    seq = make_sequence(prompt_tokens=[1, 2, 3])
    seq.generated_tokens.extend([4, 5])
    assert seq.num_tokens == 5


def test_is_finished_reflects_status():
    seq = make_sequence(status="FINISHED")
    assert seq.is_finished


def test_no_output_channel_field():
    # Sequence must not carry a live asyncio.Queue.
    assert "output_channel" not in Sequence.__slots__


def test_block_table_is_independent_per_instance():
    a, b = make_sequence(), make_sequence(seq_id=2)
    a.block_table.physical_blocks.append(0)
    assert b.block_table.physical_blocks == []


def test_block_table_defaults():
    t = BlockTable()
    assert t.physical_blocks == []
    assert t.num_tokens == 0
