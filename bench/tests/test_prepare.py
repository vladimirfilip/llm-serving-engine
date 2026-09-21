import numpy as np

from bench.workloads.prepare import correctness_prompts, filter_sharegpt, first_exchange, split_pool


def conv(cid, *turns):
    return {"id": cid, "conversations": [{"from": f, "value": v} for f, v in turns]}


def encode_prompts(texts):
    return [[0, *range(len(t))] for t in texts]  # BOS plus one token per character


def encode_replies(texts):
    return [list(range(len(t))) for t in texts]


def test_first_exchange_is_the_first_human_turn_and_the_reply_after_it():
    c = conv("a", ("system", "s"), ("human", "q1"), ("gpt", "a1"), ("human", "q2"), ("gpt", "a2"))
    assert first_exchange(c) == ("q1", "a1")
    assert first_exchange(conv("b", ("human", "only"))) is None
    assert first_exchange(conv("c", ("gpt", "x"), ("human", "y"))) is None


def test_filter_keeps_only_records_within_every_length_bound():
    convs = [
        conv("ok", ("human", "x" * 10), ("gpt", "y" * 10)),
        conv("short_prompt", ("human", "x" * 2), ("gpt", "y" * 10)),  # 3 tokens with BOS
        conv("short_reply", ("human", "x" * 10), ("gpt", "y" * 3)),
        conv("long_prompt", ("human", "x" * 60), ("gpt", "y" * 10)),
        conv("long_total", ("human", "x" * 40), ("gpt", "y" * 40)),
    ]
    kept = filter_sharegpt(convs, encode_prompts, encode_replies, max_prompt=50, max_total=70)
    assert [r["id"] for r in kept] == ["ok"]
    assert kept[0]["prompt_token_ids"][0] == 0 and kept[0]["output_len"] == 10


def test_pool_split_is_seeded_and_disjoint():
    records = [{"id": i} for i in range(30)]
    pool, rest = split_pool(records, 10, seed=1234)
    assert pool == split_pool(records, 10, seed=1234)[0]
    assert len(pool) == 10 and len(rest) == 20
    assert {r["id"] for r in pool}.isdisjoint(r["id"] for r in rest)
    assert pool != split_pool(records, 10, seed=1)[0]


def test_correctness_prompts_avoid_the_pool_and_include_long_wikitext_slices():
    records = [{"id": i, "prompt_token_ids": [0] * (40 + i)} for i in range(60)]
    pool, rest = split_pool(records, 20, seed=1)
    cfg = {"n_prompts": 10, "prompt_len_range": [32, 1024], "n_long_prompts": 2,
           "long_prompt_len": 50}
    out = correctness_prompts(rest, pool, np.arange(1000, dtype=np.int32), bos=7, cfg=cfg)
    sharegpt = [o for o in out if o["kind"] == "sharegpt"]
    assert len(sharegpt) == 10 and {o["id"] for o in sharegpt}.isdisjoint(
        f"sharegpt-{r['id']}" for r in pool)
    long = [o for o in out if o["kind"] == "long"]
    assert len(long) == 2 and all(len(o["prompt_token_ids"]) == 50 and o["prompt_token_ids"][0] == 7
                                  for o in long)
