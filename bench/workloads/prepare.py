"""`bench prepare-data`: builds the seeded, reproducible datasets under `bench/datasets/`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import Config
from .workloads import DATASETS_DIR

SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
WIKITEXT_REPO = "Salesforce/wikitext"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"
CORRECTNESS_SEED = 4321
MIN_TOKENS = 4

Encode = Callable[[list[str]], list[list[int]]]


def first_exchange(conversation: dict) -> tuple[str, str] | None:
    """The first human turn and the first assistant turn after it."""
    turns = conversation.get("conversations", [])
    human = next((i for i, t in enumerate(turns) if t["from"] == "human"), None)
    if human is None:
        return None
    reply = next((t for t in turns[human + 1 :] if t["from"] == "gpt"), None)
    return (turns[human]["value"], reply["value"]) if reply else None


def filter_sharegpt(conversations: list[dict], encode_prompts: Encode, encode_replies: Encode,
                    max_prompt: int, max_total: int) -> list[dict]:
    """Records `{id, prompt_token_ids, output_len}` whose lengths fit the workload. Prompts
    keep their BOS; `output_len` is the reference reply's token count."""
    exchanges = [(c["id"], ex) for c in conversations if (ex := first_exchange(c))]
    prompts = encode_prompts([ex[0] for _, ex in exchanges])
    replies = encode_replies([ex[1] for _, ex in exchanges])
    return [
        {"id": cid, "prompt_token_ids": p, "output_len": len(r)}
        for (cid, _), p, r in zip(exchanges, prompts, replies, strict=True)
        if len(p) >= MIN_TOKENS and len(r) >= MIN_TOKENS and len(p) <= max_prompt
        and len(p) + len(r) <= max_total
    ]


def split_pool(records: list[dict], n_pool: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Shuffled with `seed`: the load-test pool, then everything left over."""
    order = np.random.default_rng(seed).permutation(len(records))
    shuffled = [records[i] for i in order]
    return shuffled[:n_pool], shuffled[n_pool:]


def correctness_prompts(rest: list[dict], pool: list[dict], train_ids: np.ndarray, bos: int,
                        cfg: dict) -> list[dict]:
    """ShareGPT prompts within the length range, drawn from outside the load-test pool while
    enough remain, and long prompts sliced from WikiText."""
    rng = np.random.default_rng(CORRECTNESS_SEED)
    lo, hi = cfg["prompt_len_range"]

    def in_range(records: list[dict]) -> list[dict]:
        return [r for r in records if lo <= len(r["prompt_token_ids"]) <= hi]

    candidates = in_range(rest)
    if len(candidates) < cfg["n_prompts"]:
        candidates += in_range(pool)
    picks = rng.choice(len(candidates), size=cfg["n_prompts"], replace=False)
    out = [{"id": f"sharegpt-{candidates[i]['id']}", "kind": "sharegpt",
            "prompt_token_ids": candidates[i]["prompt_token_ids"]} for i in picks]
    length = cfg["long_prompt_len"]
    for k, o in enumerate(rng.integers(0, len(train_ids) - length, size=cfg["n_long_prompts"])):
        out.append({"id": f"long-{k}", "kind": "long",
                    "prompt_token_ids": [bos, *train_ids[o : o + length - 1].tolist()]})
    return out


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def prepare_data(cfg: Config, directory: Path = DATASETS_DIR) -> None:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

    def encode_prompts(texts: list[str]) -> list[list[int]]:
        return tokenizer(texts).input_ids

    def encode_replies(texts: list[str]) -> list[list[int]]:
        return tokenizer(texts, add_special_tokens=False).input_ids

    wikitext = load_dataset(WIKITEXT_REPO, "wikitext-2-raw-v1")
    for split in ("train", "test"):
        ids = tokenizer("\n\n".join(wikitext[split]["text"]), add_special_tokens=False).input_ids
        np.save(directory / f"wikitext2_{split}_ids.npy", np.asarray(ids, dtype=np.int32))

    wl = cfg.suite["workloads"]["sharegpt"]
    path = hf_hub_download(SHAREGPT_REPO, SHAREGPT_FILE, repo_type="dataset")
    raw = json.loads(Path(path).read_text())
    kept = filter_sharegpt(raw, encode_prompts, encode_replies, wl["max_prompt"], wl["max_total"])
    pool, rest = split_pool(kept, wl["n_pool"], cfg.suite["seed"])
    write_jsonl(directory / "sharegpt_pool.jsonl", pool)
    train = np.load(directory / "wikitext2_train_ids.npy")
    write_jsonl(directory / "correctness_prompts.jsonl",
                correctness_prompts(rest, pool, train, tokenizer.bos_token_id,
                                    cfg.suite["correctness"]))
    (directory / "meta.json").write_text(
        json.dumps({"bos_token_id": tokenizer.bos_token_id, "synthetic": False}))
