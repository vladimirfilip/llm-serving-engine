"""Request generators. A request is `{req_id, prompt_token_ids, output_len}`; for one
(workload, rate index, repeat) the list is identical for every engine."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


@dataclass(slots=True)
class Request:
    req_id: int
    prompt_token_ids: list[int]
    output_len: int


@dataclass(frozen=True, slots=True)
class Datasets:
    sharegpt_pool: list[dict]  # {id, prompt_token_ids, output_len}
    wikitext_train_ids: np.ndarray
    bos_token_id: int

    @classmethod
    def load(cls, bos_token_id: int, directory: Path = DATASETS_DIR) -> Datasets:
        lines = (directory / "sharegpt_pool.jsonl").read_text().splitlines()
        pool = [json.loads(line) for line in lines]
        train = np.load(directory / "wikitext2_train_ids.npy")
        return cls(pool, train, bos_token_id)


def sharegpt_requests(pool: list[dict], n: int, seed: int, workload_index: int,
                      repeat: int) -> list[Request]:
    """Without replacement while the pool lasts, with replacement past it."""
    rng = np.random.default_rng([seed, workload_index, repeat])
    picks = rng.choice(len(pool), size=n, replace=n > len(pool))
    return [Request(i, pool[p]["prompt_token_ids"], pool[p]["output_len"])
            for i, p in enumerate(picks)]


def fixed_requests(train_ids: np.ndarray, bos: int, prompt: int, output: int, n: int, seed: int,
                   workload_index: int, repeat: int) -> list[Request]:
    rng = np.random.default_rng([seed, workload_index, repeat])
    offsets = rng.integers(0, len(train_ids) - prompt, size=n)
    return [Request(i, [bos, *train_ids[o : o + prompt - 1].tolist()], output)
            for i, o in enumerate(offsets)]


def make_requests(workload: dict, workload_index: int, n: int, seed: int, repeat: int,
                  data: Datasets) -> list[Request]:
    if workload["kind"] == "sharegpt":
        return sharegpt_requests(data.sharegpt_pool, n, seed, workload_index, repeat)
    return fixed_requests(data.wikitext_train_ids, data.bos_token_id, workload["prompt"],
                          workload["output"], n, seed, workload_index, repeat)
