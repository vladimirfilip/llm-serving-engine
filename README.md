# llm-serving-engine

A single-GPU LLM serving engine built like a trading system: lock-free queues, continuous
batching, paged KV cache, custom CUDA/Triton kernels on the hot path, and honest
tail-latency measurement.

Full design: [LLM_Serving_Engine_Spec.md](LLM_Serving_Engine_Spec.md). Theory it applies:
[LLM_Systems_Technical_Reference.md](LLM_Systems_Technical_Reference.md). Division of
labor and code style: [CLAUDE.md](CLAUDE.md).

## Status

The HTTP/SSE server, tokenizer, model loading, config, metrics, threading model,
continuous- and static-batching schedulers, and both KV-cache allocators (paged and a
contiguous max-length baseline) are implemented. The one piece still unimplemented is
the load generator's open-loop timing loop:

- [`src/llm_serving_engine/loadgen/timing.py`](src/llm_serving_engine/loadgen/timing.py) — `open_loop_load_gen` raises `NotImplementedError("open-loop Poisson arrivals, timed from intended_send_time")`

Until that lands, `llm-loadgen` and `scripts/benchmark.py` both fail at their first
request — everything up to that call (server startup, model loading, admission,
generation) works end to end.

## Configuration

Key env vars (see [`config.py`](src/llm_serving_engine/config.py) for the full set):

- `LLM_MODEL`, `LLM_DEVICE`, `LLM_DTYPE`, `LLM_QUANTIZE` (`none`/`int8`), `LLM_USE_CUSTOM_KERNELS`
- `LLM_SCHEDULER` (`continuous`/`static`), `LLM_STATIC_BATCH_SIZE`
- `LLM_KV_ALLOCATOR` (`paged`/`contiguous`)
- `LLM_BLOCK_SIZE`, `LLM_GPU_MEM_UTIL`, `LLM_TOKEN_BUDGET`

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

## Running

```bash
LLM_MODEL=sshleifer/tiny-gpt2 llm-serve
curl -N -X POST localhost:8000/v1/generate \
  -H 'Content-Type: application/json' -d '{"prompt": "hello"}'
```

## Load testing

```bash
llm-loadgen --base-url http://localhost:8000 --target-qps 10 --duration-s 30 --out results/run1.json
```

Raises `NotImplementedError` until `loadgen/timing.py` is implemented. Once it is, raw
per-request results land on disk as JSON and `plotting.py` regenerates plots from those
files — a plot is never hand-edited, only its input file and a rerun.

## Benchmarks

```bash
python scripts/benchmark.py all --model sshleifer/tiny-gpt2 --device cpu
```

Launches `llm-serve` under different config knobs, drives each instance with the real
load generator, and writes raw results plus plots under `results/`:

- `pareto` — sweeps `--qps`, plots p50/p99 against achieved throughput
- `scheduler` — `LLM_SCHEDULER=continuous` vs. `static`, same offered load
- `allocator` — `LLM_KV_ALLOCATOR=paged` vs. `contiguous`, same memory budget

## Development

```bash
ruff check src/llm_serving_engine tests
pytest -q
```
