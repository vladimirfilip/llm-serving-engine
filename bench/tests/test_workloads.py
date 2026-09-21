import numpy as np

from bench.workloads.workloads import Datasets, fixed_requests, make_requests, sharegpt_requests

POOL = [{"id": i, "prompt_token_ids": [1, i], "output_len": 10 + i} for i in range(20)]


def test_sharegpt_draws_without_replacement_while_the_pool_lasts():
    picked = sharegpt_requests(POOL, 20, 1234, 0, 0)
    assert sorted(r.prompt_token_ids[1] for r in picked) == list(range(20))


def test_sharegpt_draws_with_replacement_past_the_pool():
    assert len(sharegpt_requests(POOL, 50, 1234, 0, 0)) == 50


def test_requests_are_identical_across_calls_and_carry_the_reference_output_length():
    a, b = sharegpt_requests(POOL, 10, 1234, 0, 1), sharegpt_requests(POOL, 10, 1234, 0, 1)
    assert a == b
    assert all(r.output_len == 10 + r.prompt_token_ids[1] for r in a)
    assert [r.req_id for r in a] == list(range(10))


def test_fixed_prompts_start_with_bos_and_have_the_requested_length():
    train = np.arange(1000, 6000, dtype=np.int32)
    reqs = fixed_requests(train, bos=7, prompt=128, output=64, n=5, seed=1234,
                          workload_index=1, repeat=0)
    for r in reqs:
        assert len(r.prompt_token_ids) == 128 and r.prompt_token_ids[0] == 7 and r.output_len == 64
        body = r.prompt_token_ids[1:]
        assert body == list(range(body[0], body[0] + 127))


def test_make_requests_dispatches_on_workload_kind():
    data = Datasets(POOL, np.arange(5000, dtype=np.int32), bos_token_id=7)
    fixed = make_requests({"kind": "fixed", "prompt": 16, "output": 4}, 1, 3, 1234, 0, data)
    share = make_requests({"kind": "sharegpt"}, 0, 3, 1234, 0, data)
    assert len(fixed[0].prompt_token_ids) == 16
    assert share[0].prompt_token_ids in [p["prompt_token_ids"] for p in POOL]
