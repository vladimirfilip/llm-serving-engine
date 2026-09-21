"""The HuggingFace reference model: greedy generation, teacher-forced scoring of another
engine's tokens, and perplexity. Runs in a child process (`bench.gpu_tasks`) so the harness
never holds a CUDA context, and the model is gone from the GPU before any engine starts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

BATCH_PROMPTS = 8
LONG_PROMPT_TOKENS = 4096  # prompts at least this long are generated one at a time
SCORE_CHUNK = 512


def load_model(model_path: str, device: str, dtype: torch.dtype = torch.bfloat16):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, attn_implementation="sdpa"
    )
    return model.to(device).eval()


@torch.no_grad()
def greedy_batch(model, prompts: list[list[int]], new_tokens: int, pad_id: int) -> list[dict]:
    """Greedy continuation of each prompt for exactly `new_tokens` tokens, past any stop token,
    with the logprob of each chosen token. Prompts are left-padded into one batch."""
    device = model.device
    width = max(map(len, prompts))
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(prompts), width), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        ids[row, width - len(prompt) :] = torch.tensor(prompt)
        mask[row, width - len(prompt) :] = 1
    out = model.generate(
        input_ids=ids.to(device), attention_mask=mask.to(device), max_new_tokens=new_tokens,
        min_new_tokens=new_tokens, do_sample=False, eos_token_id=None, pad_token_id=pad_id,
        output_logits=True, return_dict_in_generate=True,
    )
    tokens = out.sequences[:, width:]
    logprobs = torch.stack([step.float().log_softmax(-1) for step in out.logits], dim=1)
    chosen = logprobs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return [{"token_ids": tokens[i].tolist(), "logprobs": chosen[i].tolist()}
            for i in range(len(prompts))]


def generate(model, prompts: list[dict], new_tokens: int, long_new_tokens: int,
             pad_id: int) -> list[dict]:
    """A generation record per prompt: shorter prompts batched together, long ones alone."""
    short = [p for p in prompts if len(p["prompt_token_ids"]) < LONG_PROMPT_TOKENS]
    long = [p for p in prompts if len(p["prompt_token_ids"]) >= LONG_PROMPT_TOKENS]
    short.sort(key=lambda p: len(p["prompt_token_ids"]))
    records = []
    for i in range(0, len(short), BATCH_PROMPTS):
        batch = short[i : i + BATCH_PROMPTS]
        outs = greedy_batch(model, [p["prompt_token_ids"] for p in batch], new_tokens, pad_id)
        records += [{"id": p["id"], **o} for p, o in zip(batch, outs, strict=True)]
    for p in long:
        out = greedy_batch(model, [p["prompt_token_ids"]], long_new_tokens, pad_id)[0]
        records.append({"id": p["id"], **out})
    return records


@torch.no_grad()
def teacher_forced(model, prompt: list[int], generated: list[int]) -> dict:
    """The reference's view of `generated` given `prompt`: per generated position the logprob
    of the token the other engine chose, the reference argmax and its top-2 logprobs."""
    ids = torch.tensor([prompt + generated], device=model.device)
    logits = model(ids).logits[0, len(prompt) - 1 : -1].float()
    logprobs = logits.log_softmax(-1)
    targets = ids[0, len(prompt) :]
    top2 = logprobs.topk(2, dim=-1)
    return {
        "ref_logprob": logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).tolist(),
        "ref_argmax": top2.indices[:, 0].tolist(),
        "ref_top1": top2.values[:, 0].tolist(),
        "ref_top2": top2.values[:, 1].tolist(),
    }


def ppl_windows(test_ids: np.ndarray, window: int, bos: int) -> list[list[int]]:
    """Non-overlapping windows of `window` tokens, each behind a BOS; the remainder is dropped."""
    count = len(test_ids) // window
    return [[bos, *test_ids[i * window : (i + 1) * window].tolist()] for i in range(count)]


@torch.no_grad()
def window_logprobs(model, ids: list[int]) -> list[float]:
    """Logprob of every token after the first, given its prefix."""
    tensor = torch.tensor([ids], device=model.device)
    hidden = model.model(tensor).last_hidden_state[0, :-1]
    out = []
    for start in range(0, len(hidden), SCORE_CHUNK):
        rows = hidden[start : start + SCORE_CHUNK]
        logits = model.lm_head(rows).float()
        targets = tensor[0, start + 1 : start + 1 + len(rows)].unsqueeze(-1)
        out.append(logits.gather(-1, targets).squeeze(-1) - logits.logsumexp(-1))
    return torch.cat(out).tolist()


def perplexity(logprobs_per_window: list[list[float]]) -> float:
    """exp of the negative mean logprob over every predicted token of every window."""
    flat = np.concatenate([np.asarray(w, dtype=np.float64) for w in logprobs_per_window])
    return float(np.exp(-flat.mean()))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def prompt_hash(prompt: dict) -> str:
    return hashlib.sha256(json.dumps(prompt["prompt_token_ids"]).encode()).hexdigest()[:12]


def task_generate(model_path: str, prompts: list[dict], out_path: str, new_tokens: int,
                  long_new_tokens: int, device: str = "cuda", dtype: str = "bfloat16") -> dict:
    """Generates the reference continuation of every prompt not already in `out_path`, and
    appends it. Cached records are kept only if their prompt is unchanged."""
    out = Path(out_path)
    cached = {r["id"]: r for r in read_jsonl(out)} if out.exists() else {}
    missing = [p for p in prompts if cached.get(p["id"], {}).get("prompt_hash") != prompt_hash(p)]
    if missing:
        model = load_model(model_path, device, getattr(torch, dtype))
        pad_id = model.config.pad_token_id or 0
        for record in generate(model, missing, new_tokens, long_new_tokens, pad_id):
            by_id = next(p for p in missing if p["id"] == record["id"])
            cached[record["id"]] = record | {"prompt_hash": prompt_hash(by_id)}
        write_jsonl(out, list(cached.values()))
    return {"generated": len(missing), "cached": len(prompts) - len(missing)}


def task_score(model_path: str, prompts: list[dict], engine_gen_path: str, out_path: str,
               device: str = "cuda", dtype: str = "bfloat16") -> dict:
    """Teacher-forces the reference over an engine's generated tokens, one record per prompt."""
    by_id = {p["id"]: p for p in prompts}
    model = load_model(model_path, device, getattr(torch, dtype))
    records = []
    for r in read_jsonl(Path(engine_gen_path)):
        prompt_ids = by_id[r["id"]]["prompt_token_ids"]
        records.append({"id": r["id"], **teacher_forced(model, prompt_ids, r["token_ids"])})
    write_jsonl(Path(out_path), records)
    return {"scored": len(records)}


def task_ppl(model_path: str, test_ids_path: str, window: int, bos: int, out_path: str,
             device: str = "cuda", dtype: str = "bfloat16") -> dict:
    model = load_model(model_path, device, getattr(torch, dtype))
    windows = ppl_windows(np.load(test_ids_path), window, bos)
    result = {"ppl": perplexity([window_logprobs(model, w) for w in windows]),
              "windows": len(windows)}
    Path(out_path).write_text(json.dumps(result))
    return result
