from llm_serving_engine.scheduling.sequence import Sequence
from tests.factories import make_sequence


def test_a_new_sequence_waits_with_no_state():
    seq = make_sequence()
    assert seq.status == "WAITING"
    assert seq.generated_tokens == []
    assert seq.block_table.physical_blocks == []
    assert seq.block_table.num_tokens == 0
    assert seq.prefill_progress == 0


def test_num_tokens_counts_prompt_and_generated():
    seq = make_sequence(prompt_tokens=[1, 2, 3])
    seq.generated_tokens.extend([4, 5])
    assert seq.num_tokens == 5


def test_prefill_token_ids_continue_from_the_prompt_into_generated_tokens():
    seq = make_sequence(prompt_tokens=[1, 2, 3], generated_tokens=[4, 5])
    assert seq.prefill_token_ids(0, 2) == [1, 2]
    assert seq.prefill_token_ids(2, 5) == [3, 4, 5]


def test_sequence_carries_no_output_channel():
    assert "output_channel" not in Sequence.__slots__


def test_block_tables_are_independent_per_sequence():
    a, b = make_sequence(), make_sequence(seq_id=2)
    a.block_table.physical_blocks.append(0)
    assert b.block_table.physical_blocks == []


def test_sequences_compare_by_identity():
    assert make_sequence() != make_sequence()
