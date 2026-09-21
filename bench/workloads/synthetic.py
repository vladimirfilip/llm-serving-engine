"""Seeded random datasets shaped like the real ones, for runs with no model files or network:
the mock engine's CPU-only runs and the unit tests. Stamped `synthetic` in `meta.json`, so
every report says so."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .prepare import correctness_prompts, split_pool, write_jsonl

SYNTHETIC_BOS = 1
TRAIN_TOKENS = 400_000
TEST_TOKENS = 60_000


def write_synthetic(directory: Path, suite: dict, vocab: int, bos: int = SYNTHETIC_BOS) -> None:
    rng = np.random.default_rng(suite["seed"])
    wl = suite["workloads"]["sharegpt"]
    records = []
    for i in range(wl["n_pool"] * 2):
        prompt = int(np.clip(rng.lognormal(4.2, 0.9), 4, wl["max_prompt"]))
        output = int(np.clip(rng.lognormal(5.3, 0.8), 4, wl["max_total"] - prompt))
        ids = [bos, *rng.integers(3, vocab, size=prompt - 1).tolist()]
        records.append({"id": f"synthetic-{i}", "prompt_token_ids": ids, "output_len": output})
    pool, rest = split_pool(records, wl["n_pool"], suite["seed"])
    directory.mkdir(parents=True, exist_ok=True)
    train = rng.integers(3, vocab, size=TRAIN_TOKENS).astype(np.int32)
    np.save(directory / "wikitext2_train_ids.npy", train)
    test = rng.integers(3, vocab, size=TEST_TOKENS).astype(np.int32)
    np.save(directory / "wikitext2_test_ids.npy", test)
    write_jsonl(directory / "sharegpt_pool.jsonl", pool)
    write_jsonl(directory / "correctness_prompts.jsonl",
                correctness_prompts(rest, pool, train, bos, suite["correctness"]))
    (directory / "meta.json").write_text(json.dumps({"bos_token_id": bos, "synthetic": True}))
