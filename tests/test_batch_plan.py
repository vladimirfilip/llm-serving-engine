from types import SimpleNamespace

from llm_serving_engine.scheduling.batch_plan import BatchPlan


def test_empty_plan():
    plan = BatchPlan()
    assert len(plan) == 0
    assert plan.total_tokens == 0
    assert list(plan) == []


def test_add_accumulates_entries():
    plan = BatchPlan()
    plan.add(SimpleNamespace(seq_id=1), n_tokens=1)
    plan.add(SimpleNamespace(seq_id=2), n_tokens=32, is_prefill_chunk=True)
    assert len(plan) == 2
    assert plan.total_tokens == 33
    entries = list(plan)
    assert entries[0].seq_id == 1 and entries[0].is_prefill_chunk is False
    assert entries[1].seq_id == 2 and entries[1].is_prefill_chunk is True
